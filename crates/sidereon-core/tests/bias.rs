#![cfg(sidereon_repo_tests)]

//! Fixture provenance:
//! `fixtures/bias/CODE.BIA`
//! Source URL: `http://ftp.aiub.unibe.ch/CODE/CODE.BIA`
//! Retrieval date: 2026-07-01
//! Product license: public CODE/IGS bias product, citation requested by product
//! text, no SPDX license stated in file.
//! SHA-256: `3cc61c9dbf40fbb121826ba398a6f22cc398ceeb5f1dce365602a159f676d955`
//!
//! `fixtures/bias/P1C1_RINEX.DCB`
//! Source URL: `http://ftp.aiub.unibe.ch/CODE/P1C1_RINEX.DCB`
//! Retrieval date: 2026-07-01
//! Product license: public CODE DCB product, citation requested by CODE product
//! references, no SPDX license stated in file.
//! SHA-256: `ce0a19d8a842e7c49d63348d422baca3ce712fadd72f75b52cdd50d529a55abc`

use std::collections::BTreeMap;

use sidereon_core::astro::time::model::{Instant, JulianDateSplit, TimeScale};
use sidereon_core::bias::{
    bias_epoch_instant, ionosphere_free_coefficients, write_bias_sinex, write_bias_sinex_bytes,
    write_code_dcb, write_code_dcb_bytes, BiasDeparture, BiasEpoch, BiasError, BiasKind,
    BiasLineRole, BiasLookup, BiasMode, BiasNotice, BiasObservableFamily, BiasReadPolicy, BiasSet,
    BiasSinexHeader, BiasSlopeReference, BiasTarget, BiasTargetKey, BiasUnit, CodeDcbOptions,
    FieldError, SkipReason, WarningKind,
};
use sidereon_core::constants::{C_M_S, F_L1_HZ, F_L2_HZ, NS_TO_S};
use sidereon_core::frequencies;
use sidereon_core::{GnssSatelliteId, GnssSystem};
const BIA: &[u8] = include_bytes!("fixtures/bias/CODE.BIA");
const DCB: &[u8] = include_bytes!("fixtures/bias/P1C1_RINEX.DCB");

fn sat(system: GnssSystem, prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(system, prn).unwrap()
}

fn epoch(year: i32, doy: u16, sod: u32) -> sidereon_core::astro::time::model::Instant {
    bias_epoch_instant(BiasEpoch::new(year, doy, sod).unwrap(), TimeScale::Gpst).unwrap()
}

fn dcb_options() -> CodeDcbOptions {
    let mut options = CodeDcbOptions::new(
        ("P1".to_string(), "C1".to_string()),
        2026,
        6,
        TimeScale::Gpst,
    );
    options.receiver_system = None;
    options
}

fn ns(value: f64) -> f64 {
    value * 1.0e-9
}

fn edge_bias_sinex() -> &'static [u8] {
    b"\
%=BIA 1.00 TST 2020:001:00000 TST 2020:001:00000 2020:004:00000 A 00000011
+FILE/REFERENCE
 DESCRIPTION EDGE CASE PRODUCT
-FILE/REFERENCE
+BIAS/DESCRIPTION
 BIAS_MODE ABSOLUTE
 TIME_SYSTEM G
 SATELLITE_CLOCK_REFERENCE_OBSERVABLES G C1W C2W
 SATELLITE_CLOCK_REFERENCE_OBSERVABLES E C1C C5Q
 OBSERVATION_SAMPLING 30
 PARAMETER_SPACING 86400
 DETERMINATION_METHOD TEST
-BIAS/DESCRIPTION
+BIAS/SOLUTION
*BIAS SVN_ PRN STATION__ OBS1 OBS2 BIAS_START____ BIAS_END______ UNIT __ESTIMATED_VALUE____ _STD_DEV___
 OSB  G063 G             C1C       2020:001:00000 2020:002:00000 ns      1.000000000000E-01 1.00000E-02
 OSB  G063 G01           C1C       2020:001:00000 2020:002:00000 ns     -1.234567890000E+00 2.00000E-02    8.640000000000E-01 1.00000E-02
 OSB  G063 G01           C1W       2020:001:00000 2020:002:00000 ns      5.600000000000E-01 2.00000E-02
 DSB  G063 G01           C1C  C1W  2020:001:00000 2020:002:00000 ns     -1.794567890000E+00 3.00000E-02
 ISB  G063 G01           C1C  C2W  2020:001:00000 2020:002:00000 ns      2.500000000000E-01 4.00000E-02
 OSB  G063 G01           L1C       2020:001:00000 2020:002:00000 cyc    -1.050000000000E-01 1.00000E-02
 OSB       G   ALGO      C1C       2020:001:00000 2020:002:00000 ns      3.100000000000E+00 5.00000E-02
 OSB       E   ALGO      C1C       2020:001:00000 2020:002:00000 ns      4.200000000000E+00 6.00000E-02
 OSB  G063 G01 ALGO      C1C       2020:001:00000 2020:002:00000 ns      9.900000000000E+00 7.00000E-02
 OSB  E011 E11           C1C       2020:001:00000 2020:002:00000 ns      1.500000000000E+00 2.00000E-02
 OSB  G063 G01           C2W       2020:002:00000 2020:003:86399 ns     -3.000000000000E-01 2.00000E-02
-BIAS/SOLUTION
%=ENDBIA
"
}

#[test]
fn bias_sinex_parse_round_trips_fixture() {
    let parsed = BiasSet::parse_bias_sinex(BIA).expect("parse Bias-SINEX fixture");
    let set = parsed.value;
    assert_eq!(set.mode(), BiasMode::Absolute);
    assert_eq!(set.records().len(), 351);
    // FILE/COMMENT and INPUT/ACKNOWLEDGMENTS are blocks Bias-SINEX 1.00
    // section 2.1 defines. They are kept line for line, so they are not
    // reported as unknown blocks.
    assert!(!set
        .diagnostics()
        .skips
        .iter()
        .any(|skip| matches!(skip.reason, SkipReason::UnknownBlock(_))));
    assert!(set.source_lines().iter().any(|line| {
        line.role == BiasLineRole::BlockBody
            && line
                .text
                .starts_with(" CODE final product series for the IGS.")
    }));
    assert_eq!(
        set.clock_reference().per_system.get(&GnssSystem::Gps),
        Some(&("C1W".to_string(), "C2W".to_string()))
    );
    assert_eq!(set.time_scale(), Some(TimeScale::Gpst));
    assert_eq!(set.time_system_label(), Some("G"));

    // The 74-column header line of section 4.1, field by field.
    let header = set.header().sinex.as_ref().expect("header line");
    assert_eq!(header.version, "1.00");
    assert_eq!(header.file_agency.as_deref(), Some("COD"));
    assert_eq!(header.creation_time.as_deref(), Some("2026:182:31588"));
    assert_eq!(header.data_agency.as_deref(), Some("IGS"));
    assert_eq!(header.start.as_deref(), Some("2026:152:00000"));
    assert_eq!(header.end.as_deref(), Some("2026:182:00000"));
    assert_eq!(header.mode.as_deref(), Some("A"));
    assert_eq!(header.estimate_count_value(), Some(351));
    assert!(!set.notices().iter().any(|notice| matches!(
        notice,
        BiasNotice::Departure(BiasDeparture::EstimateCountMismatch { .. })
    )));

    // Line 29 carries a byte that is not UTF-8, in a citation.
    assert_eq!(set.notices(), &[BiasNotice::InvalidUtf8 { line: 29 }]);
    let counts = set.line_counts();
    assert_eq!(counts.lines, 423);
    assert_eq!(counts.records, 351);
    assert_eq!(counts.skipped, 0);
    assert_eq!(counts.header_footer, 2);
    assert_eq!(
        counts.header_footer
            + counts.comments
            + counts.blank
            + counts.block_delimiters
            + counts.info_rows
            + counts.records
            + counts.skipped
            + counts.block_body
            + counts.other,
        counts.lines
    );

    let g01 = sat(GnssSystem::Gps, 1);
    assert_eq!(
        set.code_osb_seconds(g01, "C1C", epoch(2026, 181, 0))
            .value()
            .unwrap()
            .to_bits(),
        ns(-6.2069).to_bits()
    );
    assert_eq!(
        set.code_osb_seconds(g01, "C1W", epoch(2026, 181, 0))
            .value()
            .unwrap()
            .to_bits(),
        ns(-5.2579).to_bits()
    );
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Glonass, 2), "C1P", epoch(2026, 181, 0))
            .value()
            .unwrap()
            .to_bits(),
        ns(1.7840).to_bits()
    );

    // The product restates its input byte for byte. A string cannot hold the
    // non-UTF-8 byte of line 29, so the string writer names that line instead
    // of replacing the byte.
    assert_eq!(
        write_bias_sinex(&set),
        Err(BiasError::InvalidUtf8Line { line: 29 })
    );
    let encoded = write_bias_sinex_bytes(&set).expect("write Bias-SINEX");
    assert_eq!(encoded, BIA);
    let reparsed = BiasSet::parse_bias_sinex(&encoded)
        .expect("reparse Bias-SINEX")
        .value;
    assert_eq!(set.records(), reparsed.records());
    assert_eq!(reparsed.skipped_records(), 0);
}

#[test]
fn code_dcb_parse_round_trips_fixture() {
    let parsed = BiasSet::parse_code_dcb(DCB, None).expect("parse DCB fixture");
    let set = parsed.value;
    assert_eq!(set.mode(), BiasMode::Relative);
    // The file's two satellite numbers above the operational roster - G34 on
    // line 40 and R28 on line 67 - are ordinary satellite tokens, so every
    // record in the file is retained and nothing is skipped.
    assert_eq!(set.skipped_records(), 0);
    assert_eq!(set.records().len(), 498);

    let g01 = sat(GnssSystem::Gps, 1);
    assert_eq!(
        set.code_dsb_seconds(g01, "C1W", "C1C", epoch(2026, 153, 0))
            .value()
            .unwrap()
            .to_bits(),
        ns(0.626).to_bits()
    );
    assert_eq!(
        set.code_dsb_seconds(g01, "C1C", "C1W", epoch(2026, 153, 0))
            .value()
            .unwrap()
            .to_bits(),
        (-ns(0.626)).to_bits()
    );
    assert_eq!(
        set.code_dsb_seconds(
            sat(GnssSystem::Glonass, 2),
            "C1P",
            "C1C",
            epoch(2026, 153, 0)
        )
        .value()
        .unwrap()
        .to_bits(),
        ns(0.291).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "algo", "C1W", "C1C", epoch(2026, 153, 0))
            .value()
            .unwrap()
            .to_bits(),
        ns(-1.314).to_bits()
    );
    // The product names the station by code and DOMES number. A
    // nine-character identifier with the same code is not answered from it:
    // the two identify the station differently, and the product states no
    // link between them.
    assert_eq!(
        set.receiver_code_dsb_seconds(
            GnssSystem::Gps,
            "algo00xxx",
            "C1W",
            "C1C",
            epoch(2026, 153, 0),
        ),
        BiasLookup::Absent
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(
            GnssSystem::Glonass,
            "ALGO",
            "C1P",
            "C1C",
            epoch(2026, 153, 0),
        )
        .value()
        .unwrap()
        .to_bits(),
        ns(0.218).to_bits()
    );

    // The two satellites above the operational roster keep their own values,
    // in the file's own nanosecond units, with the sign convention the
    // neighbouring records use.
    let g34 = sat(GnssSystem::Gps, 34);
    assert_eq!(
        set.code_dsb_seconds(g34, "C1W", "C1C", epoch(2026, 153, 0))
            .value()
            .unwrap()
            .to_bits(),
        ns(0.941).to_bits()
    );
    assert_eq!(
        set.code_dsb_seconds(g34, "C1C", "C1W", epoch(2026, 153, 0))
            .value()
            .unwrap()
            .to_bits(),
        (-ns(0.941)).to_bits()
    );
    let r28 = sat(GnssSystem::Glonass, 28);
    assert_eq!(
        set.code_dsb_seconds(r28, "C1P", "C1C", epoch(2026, 153, 0))
            .value()
            .unwrap()
            .to_bits(),
        ns(1.184).to_bits()
    );
    assert_eq!(
        set.code_dsb_seconds(r28, "C1C", "C1P", epoch(2026, 153, 0))
            .value()
            .unwrap()
            .to_bits(),
        (-ns(1.184)).to_bits()
    );
    // Their neighbours are unchanged: G32 and R27 keep their own values, so
    // nothing has shifted by a record.
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 32), "C1W", "C1C", epoch(2026, 153, 0))
            .value()
            .unwrap()
            .to_bits(),
        ns(0.343).to_bits()
    );
    assert_eq!(
        set.code_dsb_seconds(
            sat(GnssSystem::Glonass, 27),
            "C1P",
            "C1C",
            epoch(2026, 153, 0)
        )
        .value()
        .unwrap()
        .to_bits(),
        (-ns(1.160)).to_bits()
    );
    // G33 is absent from the file and stays absent; no record is invented for
    // the gap between G32 and G34.
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 33), "C1W", "C1C", epoch(2026, 153, 0)),
        BiasLookup::Absent
    );

    // The retained records survive a write and read back unchanged.
    let written = write_code_dcb(&set).expect("write DCB");
    assert!(
        written.lines().any(|line| line.starts_with("G34")),
        "G34 must be written back"
    );
    assert!(
        written.lines().any(|line| line.starts_with("R28")),
        "R28 must be written back"
    );
    let reparsed = BiasSet::parse_code_dcb(written.as_bytes(), None)
        .expect("reparse written DCB")
        .value;
    assert_eq!(reparsed.skipped_records(), 0);
    assert_eq!(reparsed.records().len(), set.records().len());
    assert_eq!(
        reparsed
            .code_dsb_seconds(g34, "C1W", "C1C", epoch(2026, 153, 0))
            .value()
            .unwrap()
            .to_bits(),
        ns(0.941).to_bits()
    );
    assert_eq!(
        reparsed
            .code_dsb_seconds(r28, "C1P", "C1C", epoch(2026, 153, 0))
            .value()
            .unwrap()
            .to_bits(),
        ns(1.184).to_bits()
    );
}

#[test]
fn code_dcb_requires_metadata_when_title_is_not_self_describing() {
    let err = BiasSet::parse_code_dcb(b" G01  1.000  0.100\n", None).unwrap_err();
    assert_eq!(err, BiasError::MissingDcbMetadata);
}

#[test]
fn forgiving_bias_sinex_parse_reports_typed_skips() {
    let text = "\
%=BIA 1.00 TST 2020:001:00000 TST 2020:001:00000 2020:002:00000 A 00000004
+FILE/REFERENCE
 DESCRIPTION TEST
-FILE/REFERENCE
+BIAS/DESCRIPTION
 BIAS_MODE ABSOLUTE
 TIME_SYSTEM G
 SATELLITE_CLOCK_REFERENCE_OBSERVABLES G C1W C2W
-BIAS/DESCRIPTION
+UNKNOWN/BLOCK
 vendor text
-UNKNOWN/BLOCK
+BIAS/SOLUTION
 OSB  G063 G01           C1C       2020:001:00000 2020:002:00000 ns      1.000000000000E+00 1.00000E-02
 OSB  G063 G01           C1W       bad-start       2020:002:00000 ns      1.000000000000E+00 1.00000E-02
 OSB  G063 G01           C2W       2020:001:00000 2020:002:00000 bad     1.000000000000E+00 1.00000E-02
 OSB  G063 G01           C5Q       2020:001:00000 2020:002:00000 ns      bad-value            1.00000E-02
-BIAS/SOLUTION
%=ENDBIA
";
    // Section 2.1 allows no other blocks, so a strict read refuses the file.
    assert_eq!(
        BiasSet::parse_bias_sinex(text.as_bytes()).unwrap_err(),
        BiasError::Departure {
            departure: BiasDeparture::UnknownBlock {
                name: "UNKNOWN/BLOCK".to_string(),
                line: 10,
            }
        }
    );
    let parsed = BiasSet::parse_bias_sinex_with_policy(text.as_bytes(), BiasReadPolicy::Lenient)
        .expect("forgiving parse");
    assert!(parsed
        .value
        .notices()
        .contains(&BiasNotice::Departure(BiasDeparture::UnknownBlock {
            name: "UNKNOWN/BLOCK".to_string(),
            line: 10,
        })));
    assert_eq!(parsed.value.records().len(), 1);
    assert_eq!(parsed.value.skipped_records(), 4);
    assert!(matches!(
        parsed.value.diagnostics().skips[0].reason,
        SkipReason::UnknownBlock(_)
    ));
    assert!(matches!(
        parsed.value.diagnostics().skips[1].reason,
        SkipReason::MalformedField(_)
    ));
    assert!(matches!(
        parsed.value.diagnostics().skips[2].reason,
        SkipReason::UnsupportedUnit(_)
    ));
    assert!(matches!(
        parsed.value.diagnostics().skips[3].reason,
        SkipReason::MalformedField(_)
    ));
    assert_eq!(parsed.value.diagnostics().skips[1].at.line, Some(15));
    // The unknown block's body and the three skipped rows keep their text.
    let lines = parsed.value.source_lines();
    assert_eq!(lines[10].text, " vendor text");
    assert_eq!(lines[10].role, BiasLineRole::BlockBody);
    let skipped: Vec<usize> = parsed
        .value
        .skipped_lines()
        .map(|line| line.number)
        .collect();
    assert_eq!(skipped, vec![15, 16, 17]);
    assert_eq!(
        lines[15].text,
        " OSB  G063 G01           C2W       2020:001:00000 2020:002:00000 bad     1.000000000000E+00 1.00000E-02"
    );
    assert_eq!(
        write_bias_sinex(&parsed.value).expect("restate"),
        text,
        "skipped rows and the unknown block are written back"
    );
}

#[test]
fn bias_sinex_targets_units_validity_and_slope_resolve() {
    let set = BiasSet::parse_bias_sinex(edge_bias_sinex())
        .expect("parse edge Bias-SINEX")
        .value;
    let records = set.records();
    assert!(records
        .iter()
        .any(|record| record.target == BiasTarget::System(GnssSystem::Gps)));
    assert!(records
        .iter()
        .any(|record| record.target == BiasTarget::Satellite(sat(GnssSystem::Gps, 1))));
    assert!(records.iter().any(|record| {
        record.target
            == BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ALGO".to_string(),
            }
    }));
    assert!(records.iter().any(|record| {
        record.target
            == BiasTarget::SatelliteReceiver {
                sat: sat(GnssSystem::Gps, 1),
                station: "ALGO".to_string(),
            }
    }));
    assert!(records.iter().any(|record| {
        record.kind == BiasKind::Isb
            && record.obs1 == "C1C"
            && record.obs2.as_deref() == Some("C2W")
    }));

    assert_eq!(
        set.receiver_code_osb_seconds(GnssSystem::Gps, "ALGO", "C1C", epoch(2020, 1, 0))
            .value()
            .unwrap()
            .to_bits(),
        ns(3.100000000000).to_bits()
    );
    assert_eq!(
        set.receiver_code_osb_seconds(GnssSystem::Galileo, "ALGO", "C1C", epoch(2020, 1, 0))
            .value()
            .unwrap()
            .to_bits(),
        ns(4.200000000000).to_bits()
    );
    assert_eq!(
        set.sat_receiver_code_osb_seconds(
            sat(GnssSystem::Gps, 1),
            "ALGO",
            "C1C",
            epoch(2020, 1, 0),
        )
        .value()
        .unwrap()
        .to_bits(),
        ns(9.900000000000).to_bits()
    );
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 1), "L1C", epoch(2020, 1, 0)),
        BiasLookup::Absent
    );
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 1), "C1C", epoch(2019, 365, 86_399)),
        BiasLookup::Absent
    );
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 1), "C1C", epoch(2020, 2, 0)),
        BiasLookup::Absent
    );
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 1), "C2W", epoch(2020, 3, 86_399))
            .value()
            .unwrap()
            .to_bits(),
        ns(-0.300000000000).to_bits()
    );
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 1), "C2W", epoch(2020, 4, 0)),
        BiasLookup::Absent
    );

    let drifted = set
        .code_osb_seconds(sat(GnssSystem::Gps, 1), "C1C", epoch(2020, 1, 10))
        .value()
        .unwrap();
    // Bias-SINEX 1.00 section 5.1: with a slope, the bias refers to the
    // middle of its validity interval. This record is valid over
    // 2020:001:00000..2020:002:00000, so its value holds at 2020:001:43200
    // and the query ten seconds into the day lies 43190 s before that. This
    // test used to measure the drift from the interval start, which section
    // 5.1 uses only when the end is undefined.
    let expected = ns(-1.234567890000) + ns(0.864) * (10.0 - 43_200.0);
    assert_eq!(drifted.to_bits(), expected.to_bits());
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 1), "C1C", epoch(2020, 1, 43_200))
            .value()
            .unwrap()
            .to_bits(),
        ns(-1.234567890000).to_bits()
    );
}

#[test]
fn overlap_selection_uses_latest_covering_start_and_warns() {
    let lines = vec![
        "%=BIA 1.00 TST 2020:001:00000 TST 2020:001:00000 2020:004:00000 A 00000002".to_string(),
        "+FILE/REFERENCE".to_string(),
        " DESCRIPTION TEST".to_string(),
        "-FILE/REFERENCE".to_string(),
        "+BIAS/DESCRIPTION".to_string(),
        " BIAS_MODE ABSOLUTE".to_string(),
        " TIME_SYSTEM G".to_string(),
        " SATELLITE_CLOCK_REFERENCE_OBSERVABLES G C1W C2W".to_string(),
        "-BIAS/DESCRIPTION".to_string(),
        "+BIAS/SOLUTION".to_string(),
        sinex_line(SinexLine {
            kind: "OSB",
            svn: "G063",
            prn: "G01",
            station: "",
            obs1: "C1W",
            obs2: "",
            start: "2020:001:00000",
            end: "2020:003:00000",
            unit: "ns",
            value: 1.0,
        }),
        sinex_line(SinexLine {
            kind: "OSB",
            svn: "G063",
            prn: "G01",
            station: "",
            obs1: "C1W",
            obs2: "",
            start: "2020:002:00000",
            end: "2020:004:00000",
            unit: "ns",
            value: 2.0,
        }),
        "-BIAS/SOLUTION".to_string(),
        "%=ENDBIA".to_string(),
    ];
    let text = lines.join("\n");
    let set = BiasSet::parse_bias_sinex(text.as_bytes()).unwrap().value;
    assert!(set
        .diagnostics()
        .warnings
        .iter()
        .any(|warning| warning.kind == WarningKind::Overlap));
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 1), "C1W", epoch(2020, 2, 0))
            .value()
            .unwrap()
            .to_bits(),
        (2.0e-9_f64).to_bits()
    );
}

#[test]
fn units_sign_and_bias_model_match_golden_bits() {
    let set = BiasSet::parse_bias_sinex(edge_bias_sinex())
        .expect("parse edge Bias-SINEX")
        .value;
    let g01 = sat(GnssSystem::Gps, 1);
    let osb_m = set
        .code_osb_seconds(g01, "C1W", epoch(2020, 1, 0))
        .value()
        .unwrap()
        * C_M_S;
    assert_eq!(osb_m.to_bits(), (ns(0.560000000000) * C_M_S).to_bits());
    // A phase bias stated in cycles is returned as stated; the carrier
    // frequency is needed only for a phase bias stated in nanoseconds.
    let phase_m = set
        .phase_osb_cycles(g01, "L1C", epoch(2020, 1, 0), None)
        .value()
        .unwrap()
        * (C_M_S / F_L1_HZ);
    assert_eq!(
        phase_m.to_bits(),
        (-0.105000000000_f64 * (C_M_S / F_L1_HZ)).to_bits()
    );

    let dcb_s = ns(4.2);
    let gamma = (F_L1_HZ / F_L2_HZ) * (F_L1_HZ / F_L2_HZ);
    let tgd_s = dcb_s / (1.0 - gamma);
    assert_eq!(tgd_s.to_bits(), 0xbe3be217807ad49e);

    let (alpha, beta) = ionosphere_free_coefficients(F_L1_HZ, F_L2_HZ).unwrap();
    let if_used = alpha * ns(-1.234567890000) + beta * ns(-0.300000000000);
    let if_ref = alpha * ns(0.560000000000) + beta * ns(-0.300000000000);
    let model = (if_used - if_ref) * C_M_S;
    assert_eq!(model.to_bits(), 0xbff5e9ddc13e45e7);
}

#[test]
fn relative_dsb_path_resolves_multi_hop_deterministically() {
    let lines = vec![
        "%=BIA 1.00 TST 2020:001:00000 TST 2020:001:00000 2020:004:00000 R 00000002".to_string(),
        "+FILE/REFERENCE".to_string(),
        " DESCRIPTION TEST".to_string(),
        "-FILE/REFERENCE".to_string(),
        "+BIAS/DESCRIPTION".to_string(),
        " BIAS_MODE RELATIVE".to_string(),
        " TIME_SYSTEM G".to_string(),
        " SATELLITE_CLOCK_REFERENCE_OBSERVABLES G C1W C2W".to_string(),
        "-BIAS/DESCRIPTION".to_string(),
        "+BIAS/SOLUTION".to_string(),
        sinex_line(SinexLine {
            kind: "DSB",
            svn: "G063",
            prn: "G01",
            station: "",
            obs1: "C1C",
            obs2: "C1P",
            start: "2020:001:00000",
            end: "2020:002:00000",
            unit: "ns",
            value: 1.0,
        }),
        sinex_line(SinexLine {
            kind: "DSB",
            svn: "G063",
            prn: "G01",
            station: "",
            obs1: "C1P",
            obs2: "C1W",
            start: "2020:001:00000",
            end: "2020:002:00000",
            unit: "ns",
            value: 2.0,
        }),
        "-BIAS/SOLUTION".to_string(),
        "%=ENDBIA".to_string(),
    ];
    let text = lines.join("\n");
    let set = BiasSet::parse_bias_sinex(text.as_bytes()).unwrap().value;
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 1), "C1C", "C1W", epoch(2020, 1, 0))
            .value()
            .unwrap()
            .to_bits(),
        (ns(1.0) + ns(2.0)).to_bits()
    );
}

struct SinexLine<'a> {
    kind: &'a str,
    svn: &'a str,
    prn: &'a str,
    station: &'a str,
    obs1: &'a str,
    obs2: &'a str,
    start: &'a str,
    end: &'a str,
    unit: &'a str,
    value: f64,
}

fn sinex_line(line: SinexLine<'_>) -> String {
    format!(
        " {:<4} {:<4} {:<3} {:<9} {:<4} {:<4} {:<14} {:<14} {:<4} {:>21.12E} {:>11.5E}",
        line.kind,
        line.svn,
        line.prn,
        line.station,
        line.obs1,
        line.obs2,
        line.start,
        line.end,
        line.unit,
        line.value,
        1.0e-2
    )
}

#[test]
fn code_bias_model_uses_relative_dsb_when_absolute_osbs_are_absent() {
    let parsed = BiasSet::parse_code_dcb(DCB, Some(dcb_options())).expect("parse DCB fixture");
    let set = parsed.value;
    let g01 = sat(GnssSystem::Gps, 1);
    let value_m = set
        .code_bias_model_m(
            g01,
            ("C1C", "C2W"),
            (F_L1_HZ, F_L2_HZ),
            None,
            ("C1W", "C2W"),
            epoch(2026, 153, 0),
        )
        .value()
        .unwrap();
    let (alpha, _beta) = ionosphere_free_coefficients(F_L1_HZ, F_L2_HZ).unwrap();
    let expected = alpha * -ns(0.626) * C_M_S;
    assert_eq!(value_m.to_bits(), expected.to_bits());
}

#[test]
fn matched_clock_datum_returns_exact_zero_without_bias_records() {
    let parsed = BiasSet::parse_code_dcb(DCB, Some(dcb_options())).expect("parse DCB fixture");
    let set = parsed.value;
    assert_eq!(
        set.code_bias_model_m(
            sat(GnssSystem::Gps, 1),
            ("C1W", "C2W"),
            (F_L1_HZ, F_L2_HZ),
            None,
            ("C1W", "C2W"),
            epoch(2026, 153, 0),
        ),
        BiasLookup::Available {
            value: 0.0,
            records: vec![],
            overridden: vec![],
        }
    );
}

#[test]
fn receiver_station_keys_are_system_scoped() {
    let set = BiasSet::parse_bias_sinex(edge_bias_sinex())
        .expect("parse edge Bias-SINEX")
        .value;
    let mut got = BTreeMap::new();
    got.insert(
        "G",
        set.receiver_code_osb_seconds(GnssSystem::Gps, "ALGO", "C1C", epoch(2020, 1, 0))
            .value()
            .unwrap(),
    );
    got.insert(
        "E",
        set.receiver_code_osb_seconds(GnssSystem::Galileo, "ALGO", "C1C", epoch(2020, 1, 0))
            .value()
            .unwrap(),
    );
    assert_ne!(got["G"].to_bits(), got["E"].to_bits());
}

#[test]
fn code_dcb_parse_write_parse_systemless_stations_regression() {
    let dcb_text = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
      abmf                   -1.365       0.050
      ab-1                    2.500       0.010
      st_01                  -0.750       0.020
      in sp 01                1.000       0.000
";
    let mut opts = dcb_options();
    opts.receiver_system = Some(GnssSystem::Gps);

    let parsed = BiasSet::parse_code_dcb(dcb_text.as_bytes(), Some(opts.clone())).unwrap();
    let set = parsed.value;
    assert_eq!(set.skipped_records(), 0);
    assert_eq!(set.records().len(), 4);

    let rec_abmf = set
        .records()
        .iter()
        .find(|r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "abmf"))
        .expect("find abmf");
    assert_eq!(rec_abmf.value.to_bits(), (-1.365 * 1.0e-9_f64).to_bits());
    assert_eq!(
        rec_abmf.sigma.unwrap().to_bits(),
        (0.050 * 1.0e-9_f64).to_bits()
    );

    let rec_hyphen = set
        .records()
        .iter()
        .find(|r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "ab-1"))
        .expect("find ab-1");
    assert_eq!(rec_hyphen.value.to_bits(), (2.500 * 1.0e-9_f64).to_bits());
    assert_eq!(
        rec_hyphen.sigma.unwrap().to_bits(),
        (0.010 * 1.0e-9_f64).to_bits()
    );

    let rec_under = set
        .records()
        .iter()
        .find(|r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "st_01"))
        .expect("find st_01");
    assert_eq!(rec_under.value.to_bits(), (-0.750 * 1.0e-9_f64).to_bits());
    assert_eq!(
        rec_under.sigma.unwrap().to_bits(),
        (0.020 * 1.0e-9_f64).to_bits()
    );

    let rec_space = set
        .records()
        .iter()
        .find(
            |r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "in sp 01"),
        )
        .expect("find in sp 01");
    assert_eq!(rec_space.value.to_bits(), (1.000 * 1.0e-9_f64).to_bits());
    assert_eq!(
        rec_space.sigma.unwrap().to_bits(),
        (0.000 * 1.0e-9_f64).to_bits()
    );

    let t0 = epoch(2026, 153, 0);
    // Canonical lookup succeeds with exact case and uppercase
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "abmf", "C1W", "C1C", t0)
            .value()
            .unwrap()
            .to_bits(),
        (-1.365 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF", "C1W", "C1C", t0)
            .value()
            .unwrap()
            .to_bits(),
        (-1.365 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "ab-1", "C1W", "C1C", t0)
            .value()
            .unwrap()
            .to_bits(),
        (2.500 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "AB-1", "C1W", "C1C", t0)
            .value()
            .unwrap()
            .to_bits(),
        (2.500 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "st_01", "C1W", "C1C", t0)
            .value()
            .unwrap()
            .to_bits(),
        (-0.750 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "ST_01", "C1W", "C1C", t0)
            .value()
            .unwrap()
            .to_bits(),
        (-0.750 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "in sp 01", "C1W", "C1C", t0)
            .value()
            .unwrap()
            .to_bits(),
        (1.000 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "IN SP 01", "C1W", "C1C", t0)
            .value()
            .unwrap()
            .to_bits(),
        (1.000 * 1.0e-9_f64).to_bits()
    );

    // A set read from DCB is written back as its source lines, so the
    // system-less rows read back with the same options.
    let written = write_code_dcb(&set).unwrap();
    assert_eq!(written, dcb_text);
    let reparsed = BiasSet::parse_code_dcb(written.as_bytes(), Some(opts))
        .unwrap()
        .value;
    assert_eq!(reparsed.skipped_records(), 0);
    assert_eq!(reparsed.records().len(), 4);

    let rep_abmf = reparsed
        .records()
        .iter()
        .find(|r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "abmf"))
        .expect("find abmf in reparsed");
    assert_eq!(rep_abmf.value.to_bits(), rec_abmf.value.to_bits());
    assert_eq!(
        rep_abmf.sigma.unwrap().to_bits(),
        rec_abmf.sigma.unwrap().to_bits()
    );

    let rep_hyphen = reparsed
        .records()
        .iter()
        .find(|r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "ab-1"))
        .expect("find ab-1 in reparsed");
    assert_eq!(rep_hyphen.value.to_bits(), rec_hyphen.value.to_bits());
    assert_eq!(
        rep_hyphen.sigma.unwrap().to_bits(),
        rec_hyphen.sigma.unwrap().to_bits()
    );

    let rep_under = reparsed
        .records()
        .iter()
        .find(|r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "st_01"))
        .expect("find st_01 in reparsed");
    assert_eq!(rep_under.value.to_bits(), rec_under.value.to_bits());
    assert_eq!(
        rep_under.sigma.unwrap().to_bits(),
        rec_under.sigma.unwrap().to_bits()
    );

    let rep_space = reparsed
        .records()
        .iter()
        .find(
            |r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "in sp 01"),
        )
        .expect("find in sp 01 in reparsed");
    assert_eq!(rep_space.value.to_bits(), rec_space.value.to_bits());
    assert_eq!(
        rep_space.sigma.unwrap().to_bits(),
        rec_space.sigma.unwrap().to_bits()
    );

    assert_eq!(
        reparsed
            .receiver_code_dsb_seconds(GnssSystem::Gps, "abmf", "C1W", "C1C", t0)
            .value()
            .unwrap()
            .to_bits(),
        (-1.365 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        reparsed
            .receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF", "C1W", "C1C", t0)
            .value()
            .unwrap()
            .to_bits(),
        (-1.365 * 1.0e-9_f64).to_bits()
    );
}

#[test]
fn code_dcb_negative_zero_sigma_bit_preserving_roundtrip() {
    let dcb_text = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
G01                           0.626      -0.000
G     ABMF 97103M001         -1.365      -0.000
";
    let parsed = BiasSet::parse_code_dcb(dcb_text.as_bytes(), None).unwrap();
    let set = parsed.value;
    assert_eq!(set.skipped_records(), 0);
    assert_eq!(set.records().len(), 2);

    let rec_g01 = &set.records()[0];
    assert_eq!(rec_g01.value.to_bits(), (0.626 * 1.0e-9_f64).to_bits());
    assert_eq!(rec_g01.sigma.unwrap().to_bits(), (-0.0_f64).to_bits());

    let rec_abmf = &set.records()[1];
    assert_eq!(rec_abmf.value.to_bits(), (-1.365 * 1.0e-9_f64).to_bits());
    assert_eq!(rec_abmf.sigma.unwrap().to_bits(), (-0.0_f64).to_bits());

    let written = write_code_dcb(&set).unwrap();
    assert!(written
        .lines()
        .any(|l| l.starts_with("G01") && l.ends_with("   -0.000")));
    assert!(written
        .lines()
        .any(|l| l.starts_with("G     ABMF 97103M001") && l.ends_with("   -0.000")));

    let reparsed = BiasSet::parse_code_dcb(written.as_bytes(), None)
        .unwrap()
        .value;
    assert_eq!(reparsed.skipped_records(), 0);
    assert_eq!(reparsed.records().len(), 2);

    let rep_g01 = &reparsed.records()[0];
    assert_eq!(rep_g01.value.to_bits(), rec_g01.value.to_bits());
    assert_eq!(rep_g01.sigma.unwrap().to_bits(), (-0.0_f64).to_bits());

    let rep_abmf = &reparsed.records()[1];
    assert_eq!(rep_abmf.value.to_bits(), rec_abmf.value.to_bits());
    assert_eq!(rep_abmf.sigma.unwrap().to_bits(), (-0.0_f64).to_bits());
}

#[test]
fn code_dcb_negative_sigma_read_write_read_roundtrip() {
    let dcb_text = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
G01                           0.626      -0.050
G02                           0.626      -0.000
G     ABMF 97103M001         -1.365      -0.050
G     ALIC 50137M001         -1.689      -0.000
";
    let parsed = BiasSet::parse_code_dcb(dcb_text.as_bytes(), None).unwrap();
    let set = parsed.value;
    assert_eq!(set.skipped_records(), 0);
    assert_eq!(set.records().len(), 4);

    let rec_g01 = &set.records()[0];
    assert_eq!(rec_g01.value.to_bits(), (0.626 * NS_TO_S).to_bits());
    assert_eq!(
        rec_g01.sigma.unwrap().to_bits(),
        (-0.050 * NS_TO_S).to_bits()
    );
    assert!(rec_g01.sigma.unwrap().is_sign_negative());

    let rec_g02 = &set.records()[1];
    assert_eq!(rec_g02.value.to_bits(), (0.626 * NS_TO_S).to_bits());
    assert_eq!(rec_g02.sigma.unwrap().to_bits(), (-0.0_f64).to_bits());
    assert!(rec_g02.sigma.unwrap().is_sign_negative());

    let rec_abmf = &set.records()[2];
    assert_eq!(
        rec_abmf.target,
        BiasTarget::Receiver {
            system: GnssSystem::Gps,
            station: "ABMF 97103M001".to_string(),
        }
    );
    assert_eq!(rec_abmf.value.to_bits(), (-1.365 * NS_TO_S).to_bits());
    assert_eq!(
        rec_abmf.sigma.unwrap().to_bits(),
        (-0.050 * NS_TO_S).to_bits()
    );
    assert!(rec_abmf.sigma.unwrap().is_sign_negative());

    let rec_alic = &set.records()[3];
    assert_eq!(
        rec_alic.target,
        BiasTarget::Receiver {
            system: GnssSystem::Gps,
            station: "ALIC 50137M001".to_string(),
        }
    );
    assert_eq!(rec_alic.value.to_bits(), (-1.689 * NS_TO_S).to_bits());
    assert_eq!(rec_alic.sigma.unwrap().to_bits(), (-0.0_f64).to_bits());
    assert!(rec_alic.sigma.unwrap().is_sign_negative());

    let written = write_code_dcb(&set).unwrap();
    assert!(written
        .lines()
        .any(|l| l.starts_with("G01") && l.ends_with("   -0.050")));
    assert!(written
        .lines()
        .any(|l| l.starts_with("G02") && l.ends_with("   -0.000")));
    assert!(written
        .lines()
        .any(|l| l.starts_with("G     ABMF 97103M001") && l.ends_with("   -0.050")));
    assert!(written
        .lines()
        .any(|l| l.starts_with("G     ALIC 50137M001") && l.ends_with("   -0.000")));

    let reparsed = BiasSet::parse_code_dcb(written.as_bytes(), None)
        .unwrap()
        .value;
    assert_eq!(reparsed.skipped_records(), 0);
    assert_eq!(reparsed.records().len(), 4);

    let rep_g01 = &reparsed.records()[0];
    assert_eq!(rep_g01.target, rec_g01.target);
    assert_eq!(rep_g01.value.to_bits(), rec_g01.value.to_bits());
    assert_eq!(
        rep_g01.sigma.unwrap().to_bits(),
        (-0.050 * NS_TO_S).to_bits()
    );
    assert!(rep_g01.sigma.unwrap().is_sign_negative());

    let rep_g02 = &reparsed.records()[1];
    assert_eq!(rep_g02.target, rec_g02.target);
    assert_eq!(rep_g02.value.to_bits(), rec_g02.value.to_bits());
    assert_eq!(rep_g02.sigma.unwrap().to_bits(), (-0.0_f64).to_bits());
    assert!(rep_g02.sigma.unwrap().is_sign_negative());

    let rep_abmf = &reparsed.records()[2];
    assert_eq!(
        rep_abmf.target,
        BiasTarget::Receiver {
            system: GnssSystem::Gps,
            station: "ABMF 97103M001".to_string(),
        }
    );
    assert_eq!(rep_abmf.value.to_bits(), rec_abmf.value.to_bits());
    assert_eq!(
        rep_abmf.sigma.unwrap().to_bits(),
        (-0.050 * NS_TO_S).to_bits()
    );
    assert!(rep_abmf.sigma.unwrap().is_sign_negative());

    let rep_alic = &reparsed.records()[3];
    assert_eq!(
        rep_alic.target,
        BiasTarget::Receiver {
            system: GnssSystem::Gps,
            station: "ALIC 50137M001".to_string(),
        }
    );
    assert_eq!(rep_alic.value.to_bits(), rec_alic.value.to_bits());
    assert_eq!(rep_alic.sigma.unwrap().to_bits(), (-0.0_f64).to_bits());
    assert!(rep_alic.sigma.unwrap().is_sign_negative());
}

#[test]
fn code_dcb_malformed_value_candidate_diagnostics() {
    let mut opts = dcb_options();
    opts.receiver_system = Some(GnssSystem::Gps);

    // Satellite candidate with malformed VALUE
    let sat_val = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
G01                           VALUE       0.000
";
    let parsed = BiasSet::parse_code_dcb(sat_val.as_bytes(), None).unwrap();
    assert_eq!(parsed.value.records().len(), 0);
    assert_eq!(parsed.value.skipped_records(), 1);
    assert!(matches!(
        parsed.value.diagnostics().skips[0].reason,
        SkipReason::MalformedField(FieldError::FloatParse {
            field: "dcb value",
            ref value,
        }) if value == "VALUE"
    ));

    // Explicit receiver candidate with malformed VALUE
    let rec_explicit_val = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
G     ABMF 97103M001          VALUE       0.050
";
    let parsed = BiasSet::parse_code_dcb(rec_explicit_val.as_bytes(), None).unwrap();
    assert_eq!(parsed.value.records().len(), 0);
    assert_eq!(parsed.value.skipped_records(), 1);
    assert!(matches!(
        parsed.value.diagnostics().skips[0].reason,
        SkipReason::MalformedField(FieldError::FloatParse {
            field: "dcb value",
            ref value,
        }) if value == "VALUE"
    ));

    // Implicit receiver candidate with system option and malformed VALUE
    let rec_sysless_val = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
      ABMF 97103M001          VALUE       0.050
";
    let parsed = BiasSet::parse_code_dcb(rec_sysless_val.as_bytes(), Some(opts.clone())).unwrap();
    assert_eq!(parsed.value.records().len(), 0);
    assert_eq!(parsed.value.skipped_records(), 1);
    assert!(matches!(
        parsed.value.diagnostics().skips[0].reason,
        SkipReason::MalformedField(FieldError::FloatParse {
            field: "dcb value",
            ref value,
        }) if value == "VALUE"
    ));

    // Headers alone without data candidates
    let headers_alone = "\
# DCB P1-C1 2026-06 G
CODE'S MONTHLY GNSS P1-C1 DCB SOLUTION, YEAR 2026, MONTH 06      01-JUL-26 08:42
--------------------------------------------------------------------------------
DIFFERENTIAL (P1-C1) CODE BIASES FOR SATELLITES AND RECEIVERS:
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
# Comment line
";
    let parsed_none = BiasSet::parse_code_dcb(headers_alone.as_bytes(), None).unwrap();
    assert_eq!(parsed_none.value.records().len(), 0);
    assert_eq!(parsed_none.value.skipped_records(), 0);

    let parsed_opts = BiasSet::parse_code_dcb(headers_alone.as_bytes(), Some(opts)).unwrap();
    assert_eq!(parsed_opts.value.records().len(), 0);
    assert_eq!(parsed_opts.value.skipped_records(), 0);
}

#[test]
fn code_dcb_station_name_edge_cases_and_unpadded_roundtrip() {
    let mut opts = dcb_options();
    opts.receiver_system = Some(GnssSystem::Gps);
    let t0 = epoch(2026, 153, 0);

    let cases = [
        (
            "explicit STATION NAME RMS",
            format!(
                "G     {:<16}    {:9.3}   {:9.3}",
                "STATION NAME RMS", -1.365, 0.050
            ),
            "STATION NAME RMS",
        ),
        (
            "implicit STATION NAME RMS",
            "      STATION NAME RMS        -1.365       0.050".to_string(),
            "STATION NAME RMS",
        ),
        (
            "implicit CODE'S SOLUTION",
            "      CODE'S SOLUTION         -1.365       0.050".to_string(),
            "CODE'S SOLUTION",
        ),
        (
            "unpadded long station ABMF97103M001",
            "ABMF97103M001                 -1.365       0.050".to_string(),
            "ABMF97103M001",
        ),
        (
            "explicit -abmf",
            "G     -abmf                   -1.365       0.050".to_string(),
            "-abmf",
        ),
        (
            "implicit -abmf",
            "      -abmf                   -1.365       0.050".to_string(),
            "-abmf",
        ),
        (
            "unpadded -abmf",
            "-abmf                         -1.365       0.050".to_string(),
            "-abmf",
        ),
        (
            "explicit #abmf",
            "G     #abmf                   -1.365       0.050".to_string(),
            "#abmf",
        ),
        (
            "implicit #abmf",
            "      #abmf                   -1.365       0.050".to_string(),
            "#abmf",
        ),
        (
            "unpadded #abmf",
            "#abmf                         -1.365       0.050".to_string(),
            "#abmf",
        ),
    ];

    for (name, line, expected_station) in &cases {
        let text = format!(
            "# DCB P1-C1 2026-06 G\n\
             PRN / STATION NAME        VALUE (ns)  RMS (ns)\n\
            ***   ****************    *****.***   *****.***\n\
            # Comment line\n\
            --------------------------------------------------------------------------------\n\
            {line}\n"
        );
        let parsed = BiasSet::parse_code_dcb(text.as_bytes(), Some(opts.clone()))
            .unwrap_or_else(|e| panic!("{name}: parse error: {e:?}"));
        let set = parsed.value;
        assert_eq!(set.skipped_records(), 0, "{name}: expected 0 skips");
        assert_eq!(set.records().len(), 1, "{name}: expected 1 record");

        let rec = &set.records()[0];
        match &rec.target {
            BiasTarget::Receiver { system, station } => {
                assert_eq!(*system, GnssSystem::Gps, "{name}: expected GPS system");
                assert_eq!(station, expected_station, "{name}: station mismatch");
            }
            other => panic!("{name}: expected Receiver target, got {other:?}"),
        }
        assert_eq!(
            rec.value.to_bits(),
            ns(-1.365).to_bits(),
            "{name}: value mismatch"
        );
        assert_eq!(
            rec.sigma.unwrap().to_bits(),
            ns(0.050).to_bits(),
            "{name}: sigma mismatch"
        );

        let written = write_code_dcb(&set).unwrap_or_else(|e| panic!("{name}: write error: {e:?}"));
        let reparsed = BiasSet::parse_code_dcb(written.as_bytes(), Some(opts.clone()))
            .unwrap_or_else(|e| panic!("{name}: reparse error: {e:?}"))
            .value;
        assert_eq!(
            reparsed.skipped_records(),
            0,
            "{name}: expected 0 skips on reparse"
        );
        assert_eq!(
            reparsed.records().len(),
            1,
            "{name}: expected 1 record on reparse"
        );

        let rep_rec = &reparsed.records()[0];
        assert_eq!(
            rep_rec.target, rec.target,
            "{name}: target preserved on reparse"
        );
        assert_eq!(
            rep_rec.value.to_bits(),
            rec.value.to_bits(),
            "{name}: value preserved"
        );
        assert_eq!(
            rep_rec.sigma.unwrap().to_bits(),
            rec.sigma.unwrap().to_bits(),
            "{name}: sigma preserved"
        );
    }

    // Concrete long station ABMF97103M001: verify full station and canonical ABMF lookup survive parse/write/parse
    let unpadded_text = "# DCB P1-C1 2026-06 G\nABMF97103M001                 -1.365       0.050\n";
    let set_abmf = BiasSet::parse_code_dcb(unpadded_text.as_bytes(), Some(opts.clone()))
        .unwrap()
        .value;
    assert_eq!(set_abmf.skipped_records(), 0);
    assert_eq!(set_abmf.records().len(), 1);
    assert_eq!(
        set_abmf.records()[0].target,
        BiasTarget::Receiver {
            system: GnssSystem::Gps,
            station: "ABMF97103M001".to_string(),
        }
    );
    assert_eq!(
        set_abmf
            .receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF97103M001", "C1W", "C1C", t0)
            .value()
            .unwrap()
            .to_bits(),
        ns(-1.365).to_bits()
    );
    assert_eq!(
        set_abmf
            .receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF", "C1W", "C1C", t0)
            .value()
            .unwrap()
            .to_bits(),
        ns(-1.365).to_bits()
    );

    let written_abmf = write_code_dcb(&set_abmf).unwrap();
    assert_eq!(written_abmf, unpadded_text);
    let reparsed_abmf = BiasSet::parse_code_dcb(written_abmf.as_bytes(), Some(opts.clone()))
        .unwrap()
        .value;
    assert_eq!(reparsed_abmf.skipped_records(), 0);
    assert_eq!(reparsed_abmf.records().len(), 1);
    assert_eq!(
        reparsed_abmf.records()[0].target,
        BiasTarget::Receiver {
            system: GnssSystem::Gps,
            station: "ABMF97103M001".to_string(),
        }
    );
    assert_eq!(
        reparsed_abmf
            .receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF97103M001", "C1W", "C1C", t0)
            .value()
            .unwrap()
            .to_bits(),
        ns(-1.365).to_bits()
    );
    assert_eq!(
        reparsed_abmf
            .receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF", "C1W", "C1C", t0)
            .value()
            .unwrap()
            .to_bits(),
        ns(-1.365).to_bits()
    );

    // Data candidates with malformed VALUE still emit typed diagnostics
    let malformed_cases = [
        "G     STATION NAME RMS        VALUE       0.050",
        "      STATION NAME RMS        VALUE       0.050",
        "      CODE'S SOLUTION         VALUE       0.050",
        "      -abmf                   VALUE       0.050",
        "      #abmf                   VALUE       0.050",
    ];
    for line in malformed_cases {
        let text = format!("# DCB P1-C1 2026-06 G\n{line}\n");
        let parsed = BiasSet::parse_code_dcb(text.as_bytes(), Some(opts.clone())).unwrap();
        assert_eq!(parsed.value.records().len(), 0);
        assert_eq!(parsed.value.skipped_records(), 1);
        assert!(matches!(
            parsed.value.diagnostics().skips[0].reason,
            SkipReason::MalformedField(FieldError::FloatParse {
                field: "dcb value",
                ref value,
            }) if value == "VALUE"
        ));
    }
}

// Fixed-column cases. Rows are placed at the Bias-SINEX 1.00 section 4.8
// columns by `solution_row`, whose layout is checked against a row of the CODE
// product in `solution_row_builder_matches_code_columns`.

/// Fields of one `BIAS/SOLUTION` row as text. `solution_row` places them at
/// the section 4.8 columns (0-based byte ranges): BIAS 1..5, SVN 6..10,
/// PRN 11..14, STATION 15..24, OBS1 25..29, OBS2 30..34, BIAS_START 35..49,
/// BIAS_END 50..64, UNIT 65..69, estimate 70..91, standard deviation 92..103,
/// slope 104..125 and slope standard deviation 126..137. Text fields are
/// left-aligned and numeric fields right-aligned.
#[derive(Clone, Copy)]
struct Row<'a> {
    kind: &'a str,
    svn: &'a str,
    prn: &'a str,
    station: &'a str,
    obs1: &'a str,
    obs2: &'a str,
    start: &'a str,
    end: &'a str,
    unit: &'a str,
    value: &'a str,
    sigma: &'a str,
    slope: &'a str,
    slope_sigma: &'a str,
}

impl<'a> Row<'a> {
    /// An OSB valid over 2020:001:00000..2020:002:00000 with no SVN, station,
    /// uncertainty or slope.
    fn osb(prn: &'a str, obs1: &'a str, unit: &'a str, value: &'a str) -> Self {
        Self {
            kind: "OSB",
            svn: "",
            prn,
            station: "",
            obs1,
            obs2: "",
            start: "2020:001:00000",
            end: "2020:002:00000",
            unit,
            value,
            sigma: "",
            slope: "",
            slope_sigma: "",
        }
    }
}

fn put(line: &mut [u8], (start, end): (usize, usize), text: &str, right_aligned: bool) {
    assert!(
        text.is_ascii() && text.len() <= end - start,
        "{text:?} does not fit columns {start}..{end}"
    );
    let at = if right_aligned {
        end - text.len()
    } else {
        start
    };
    line[at..at + text.len()].copy_from_slice(text.as_bytes());
}

fn solution_row(row: Row<'_>) -> String {
    let mut line = vec![b' '; 137];
    put(&mut line, (1, 5), row.kind, false);
    put(&mut line, (6, 10), row.svn, false);
    put(&mut line, (11, 14), row.prn, false);
    put(&mut line, (15, 24), row.station, false);
    put(&mut line, (25, 29), row.obs1, false);
    put(&mut line, (30, 34), row.obs2, false);
    put(&mut line, (35, 49), row.start, false);
    put(&mut line, (50, 64), row.end, false);
    put(&mut line, (65, 69), row.unit, false);
    put(&mut line, (70, 91), row.value, true);
    put(&mut line, (92, 103), row.sigma, true);
    put(&mut line, (104, 125), row.slope, true);
    put(&mut line, (126, 137), row.slope_sigma, true);
    String::from_utf8(line).unwrap().trim_end().to_string()
}

/// The 74-column header line of section 4.1 for a TST product.
fn header_line(mode: char, count: usize) -> String {
    let line = format!(
        "%=BIA 1.00 TST 2020:001:00000 TST 2020:001:00000 2020:011:00000 {mode} {count:08}"
    );
    assert_eq!(line.len(), 74);
    line
}

/// A Bias-SINEX document: header line, the three mandatory blocks, footer.
fn assemble(
    header: String,
    file_reference: &[&str],
    description: &[&str],
    rows: &[String],
) -> String {
    let mut lines = vec![header, "+FILE/REFERENCE".to_string()];
    lines.extend(file_reference.iter().map(|line| line.to_string()));
    lines.push("-FILE/REFERENCE".to_string());
    lines.push("+BIAS/DESCRIPTION".to_string());
    lines.extend(description.iter().map(|line| line.to_string()));
    lines.push("-BIAS/DESCRIPTION".to_string());
    lines.push("+BIAS/SOLUTION".to_string());
    lines.extend(rows.iter().cloned());
    lines.push("-BIAS/SOLUTION".to_string());
    lines.push("%=ENDBIA".to_string());
    let mut text = lines.join("\n");
    text.push('\n');
    text
}

const TEST_REFERENCE: [&str; 1] = [" DESCRIPTION        TEST"];
const ABSOLUTE_G: [&str; 2] = [
    " BIAS_MODE                               ABSOLUTE",
    " TIME_SYSTEM                             G",
];

/// Document whose lines are: 1 header, 2..4 FILE/REFERENCE, 5 block start,
/// the description rows, the block end, the solution block start, then the
/// solution rows.
fn document(mode: char, description: &[&str], rows: &[String]) -> String {
    assemble(
        header_line(mode, rows.len()),
        &TEST_REFERENCE,
        description,
        rows,
    )
}

fn parse(text: &str) -> BiasSet {
    BiasSet::parse_bias_sinex(text.as_bytes())
        .expect("strict Bias-SINEX parse")
        .value
}

#[test]
fn solution_row_builder_matches_code_columns() {
    let text = String::from_utf8_lossy(BIA);
    let real = text
        .lines()
        .find(|line| line.starts_with(" OSB  G080 G01           C1C"))
        .expect("CODE row");
    let built = solution_row(Row {
        svn: "G080",
        start: "2026:152:00000",
        end: "2026:182:00000",
        sigma: "0.0046",
        ..Row::osb("G01", "C1C", "ns", "-6.2069")
    });
    assert_eq!(built, real);
}

#[test]
fn slope_reference_epoch_follows_section_5_1() {
    // Bias-SINEX 1.00 section 5.1: a sloped bias refers to the middle of a
    // closed interval, to the start when the end is undefined, and to the end
    // when the start is undefined. Every row states 10 ns and 1 ns/s.
    let rows = [
        solution_row(Row {
            end: "2020:001:00020",
            slope: "1.0",
            ..Row::osb("G01", "C1C", "ns", "10.0")
        }),
        solution_row(Row {
            start: "0000:000:00000",
            end: "2020:001:00020",
            slope: "1.0",
            ..Row::osb("G01", "C1W", "ns", "10.0")
        }),
        solution_row(Row {
            end: "0000:000:00000",
            slope: "1.0",
            ..Row::osb("G01", "C2W", "ns", "10.0")
        }),
        solution_row(Row {
            start: "0000:000:00000",
            end: "0000:000:00000",
            slope: "1.0",
            ..Row::osb("G01", "C5Q", "ns", "10.0")
        }),
    ];
    let set = parse(&document('A', &ABSOLUTE_G, &rows));
    let g01 = sat(GnssSystem::Gps, 1);
    let at = |obs: &str, second: u32| set.code_osb_seconds(g01, obs, epoch(2020, 1, second));
    let start = BiasEpoch::new(2020, 1, 0).unwrap();
    let end = BiasEpoch::new(2020, 1, 20).unwrap();

    // Closed 2020:001:00000..2020:001:00020: the value holds at second 10.
    assert_eq!(
        set.records()[0].slope_reference(),
        BiasSlopeReference::Midpoint { start, end }
    );
    assert_eq!(at("C1C", 10).value().unwrap().to_bits(), ns(10.0).to_bits());
    assert_eq!(at("C1C", 0).value().unwrap().to_bits(), 0.0_f64.to_bits());
    assert_eq!(at("C1C", 20), BiasLookup::Absent);

    // Undefined start: the value holds at the end, second 20.
    assert_eq!(
        set.records()[1].slope_reference(),
        BiasSlopeReference::End(end)
    );
    assert_eq!(at("C1W", 10).value().unwrap().to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        at("C1W", 0).value().unwrap().to_bits(),
        (-ns(10.0)).to_bits()
    );

    // Undefined end: the value holds at the start.
    assert_eq!(
        set.records()[2].slope_reference(),
        BiasSlopeReference::Start(start)
    );
    assert_eq!(at("C2W", 0).value().unwrap().to_bits(), ns(10.0).to_bits());
    assert_eq!(at("C2W", 10).value().unwrap().to_bits(), ns(20.0).to_bits());

    // Neither bound: section 5.1 gives the value no epoch.
    assert_eq!(
        set.records()[3].slope_reference(),
        BiasSlopeReference::Undefined
    );
    assert_eq!(
        at("C5Q", 10),
        BiasLookup::UndefinedSlopeReference { record: 3 }
    );
}

#[test]
fn slope_intervals_are_the_exact_time_from_the_reference_epoch() {
    // A sloped bias referred to 2020:001:00000, queried at 06:30:15.1 on that
    // day. The interval is 23415.1 s exactly; the difference of the two split
    // Julian dates, each rounded, is 23415.100000000002 s.
    let rows = [solution_row(Row {
        end: "0000:000:00000",
        slope: "1.0",
        ..Row::osb("G01", "C1C", "ns", "10.0")
    })];
    let set = parse(&document('A', &ABSOLUTE_G, &rows));
    let record = &set.records()[0];
    let slope = record.slope.expect("sloped record");
    let query = Instant::from_julian_date(TimeScale::Gpst, {
        let (jd_whole, fraction) =
            sidereon_core::astro::time::split_julian_date(2020, 1, 1, 6, 30, 15.1);
        JulianDateSplit::new(jd_whole, fraction).unwrap()
    });
    let split = query.julian_date().unwrap();
    assert_eq!(split.fraction * 86_400.0, 23_415.100_000_000_002);
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 1), "C1C", query)
            .value()
            .unwrap()
            .to_bits(),
        (record.value + slope * 23_415.1).to_bits()
    );
    // A nanosecond query is taken at its count, and a query just before the
    // record's start is outside it.
    let nanos_from_j2000 = (sidereon_core::astro::time::j2000_seconds(2020, 1, 1, 6, 30, 15.0)
        as i128)
        * 1_000_000_000
        + 100_000_000;
    assert_eq!(
        set.code_osb_seconds(
            sat(GnssSystem::Gps, 1),
            "C1C",
            Instant::from_nanos(TimeScale::Gpst, nanos_from_j2000)
        )
        .value()
        .unwrap()
        .to_bits(),
        (record.value + slope * 23_415.1).to_bits()
    );
    let just_before = Instant::from_julian_date(TimeScale::Gpst, {
        let (jd_whole, fraction) =
            sidereon_core::astro::time::split_julian_date(2019, 12, 31, 23, 59, 59.999_999_999_9);
        JulianDateSplit::new(jd_whole, fraction).unwrap()
    });
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 1), "C1C", just_before),
        BiasLookup::Absent
    );
}

#[test]
fn phase_family_comes_from_the_observable_and_nanoseconds_need_a_carrier() {
    let rows = [
        // As in the CODE daily product: a phase bias stated in nanoseconds.
        solution_row(Row {
            svn: "G080",
            sigma: "0.0001",
            ..Row::osb("G01", "L1C", "ns", "-0.99559")
        }),
        solution_row(Row::osb("G01", "L2W", "cyc", "0.125")),
        solution_row(Row::osb("R02", "L1P", "ns", "1.5")),
        // Section 4.8: code biases are stated in ns.
        solution_row(Row::osb("G01", "C1W", "cyc", "0.5")),
        // A DSB between a code and a phase observable is kept as a mixed
        // record.
        solution_row(Row {
            kind: "DSB",
            obs2: "L1C",
            ..Row::osb("G01", "C1C", "ns", "0.5")
        }),
    ];
    let set = parse(&document('A', &ABSOLUTE_G, &rows));
    let records = set.records();
    assert_eq!(records.len(), 4);
    assert_eq!(records[3].family, BiasObservableFamily::Mixed);
    assert_eq!(records[3].line, Some(14));
    assert_eq!(records[0].family, BiasObservableFamily::Phase);
    assert_eq!(records[0].unit, BiasUnit::Nanoseconds);
    assert!(records[0].is_phase());
    assert_eq!(records[0].value.to_bits(), ns(-0.99559).to_bits());
    assert_eq!(records[1].unit, BiasUnit::Cycles);
    let skips: Vec<(Option<usize>, SkipReason)> = set
        .diagnostics()
        .skips
        .iter()
        .map(|skip| (skip.at.line, skip.reason.clone()))
        .collect();
    assert_eq!(
        skips,
        vec![(
            Some(13),
            SkipReason::InconsistentRecord("code bias is not stated in ns")
        )]
    );

    let t = epoch(2020, 1, 0);
    let g01 = sat(GnssSystem::Gps, 1);
    assert_eq!(set.code_osb_seconds(g01, "L1C", t), BiasLookup::Absent);
    // Code and phase lookups do not use the mixed record.
    assert_eq!(
        set.code_dsb_seconds(g01, "C1C", "L1C", t),
        BiasLookup::Absent
    );
    assert_eq!(
        set.phase_osb_cycles(g01, "L1C", t, None),
        BiasLookup::CarrierFrequencyRequired { record: 0 }
    );
    assert_eq!(
        set.phase_osb_cycles(g01, "L1C", t, Some(F_L1_HZ))
            .value()
            .unwrap()
            .to_bits(),
        (ns(-0.99559) * F_L1_HZ).to_bits()
    );
    assert_eq!(
        set.phase_osb_cycles(g01, "L1C", t, Some(-1.0)),
        BiasLookup::InvalidCarrierFrequency
    );
    assert_eq!(
        set.phase_osb_cycles(g01, "L2W", t, None),
        BiasLookup::Available {
            value: 0.125,
            records: vec![1],
            overridden: vec![],
        }
    );

    // A GLONASS FDMA carrier depends on the satellite's frequency channel, so
    // a nanosecond phase bias converts only with the channel's frequency.
    let r02 = sat(GnssSystem::Glonass, 2);
    assert_eq!(
        frequencies::rinex_observation_frequency_hz(GnssSystem::Glonass, "L1P", 3.04, None),
        None
    );
    assert_eq!(
        set.phase_osb_cycles(r02, "L1P", t, None),
        BiasLookup::CarrierFrequencyRequired { record: 2 }
    );
    let channel_hz =
        frequencies::rinex_observation_frequency_hz(GnssSystem::Glonass, "L1P", 3.04, Some(-4))
            .unwrap();
    assert_eq!(
        set.phase_osb_cycles(r02, "L1P", t, Some(channel_hz))
            .value()
            .unwrap()
            .to_bits(),
        (ns(1.5) * channel_hz).to_bits()
    );
}

#[test]
fn time_system_declarations_are_kept_and_checked() {
    let rows = [solution_row(Row::osb("G01", "C1C", "ns", "1.0"))];
    let mode = " BIAS_MODE                               ABSOLUTE";
    let text_with = |time_rows: &[&str]| {
        let mut description = vec![mode];
        description.extend_from_slice(time_rows);
        document('A', &description, &rows)
    };
    let strict = |time_rows: &[&str]| BiasSet::parse_bias_sinex(text_with(time_rows).as_bytes());
    let lenient = |time_rows: &[&str]| {
        BiasSet::parse_bias_sinex_with_policy(
            text_with(time_rows).as_bytes(),
            BiasReadPolicy::Lenient,
        )
        .expect("lenient read")
        .value
    };
    let g01 = sat(GnssSystem::Gps, 1);
    let t = epoch(2020, 1, 0);

    // Missing: section 4.6 makes TIME_SYSTEM mandatory, so a strict read
    // refuses the file and a lenient read assumes no scale.
    let missing = BiasDeparture::MissingDeclaration {
        keyword: "TIME_SYSTEM",
    };
    assert_eq!(
        strict(&[]).unwrap_err(),
        BiasError::Departure {
            departure: missing.clone()
        }
    );
    let set = lenient(&[]);
    assert_eq!(set.time_scale(), None);
    assert!(set.notices().contains(&BiasNotice::Departure(missing)));
    assert_eq!(
        set.code_osb_seconds(g01, "C1C", t),
        BiasLookup::UnsupportedScale {
            product: None,
            query: TimeScale::Gpst,
        }
    );

    // A label section 4.6 does not define: refused strictly; read leniently,
    // kept as written, with the scale it names if it names one.
    for (label, scale) in [
        ("XYZ", None),
        ("GPS", Some(TimeScale::Gpst)),
        ("GLO", Some(TimeScale::Utc)),
        ("TCG", Some(TimeScale::Tcg)),
        ("TCB", Some(TimeScale::Tcb)),
    ] {
        let row = format!(" TIME_SYSTEM                             {label}");
        let departure = BiasDeparture::NonStandardTimeSystem {
            line: 7,
            label: label.to_string(),
        };
        assert_eq!(
            strict(&[row.as_str()]).unwrap_err(),
            BiasError::Departure {
                departure: departure.clone()
            },
            "{label}"
        );
        let set = lenient(&[row.as_str()]);
        assert_eq!(set.time_scale(), scale, "{label}");
        assert_eq!(set.time_system_label(), Some(label));
        assert!(set.notices().contains(&BiasNotice::Departure(departure)));
    }

    // The section 4.6 flags `S` and `I` read as GPS time: RINEX tags SBAS
    // data in GPS time, and RTKLIB, like the SP3 reader of this library,
    // treats IRNSS system time as GPS-aligned. The label is kept as written.
    for label in ["S", "I"] {
        let row = format!(" TIME_SYSTEM {label}");
        let set = strict(&[row.as_str()]).unwrap().value;
        assert_eq!(set.time_scale(), Some(TimeScale::Gpst), "{label}");
        assert_eq!(set.time_system_label(), Some(label));
    }

    // Repeated with the same meaning.
    let set = strict(&[" TIME_SYSTEM G", " TIME_SYSTEM G"]).unwrap().value;
    assert_eq!(set.time_scale(), Some(TimeScale::Gpst));
    assert!(set.notices().contains(&BiasNotice::RepeatedDeclaration {
        line: 8,
        keyword: "TIME_SYSTEM",
    }));

    // Repeated with another meaning: undetermined.
    let set = strict(&[" TIME_SYSTEM G", " TIME_SYSTEM UTC"])
        .unwrap()
        .value;
    assert_eq!(set.time_scale(), None);
    assert!(set.notices().contains(&BiasNotice::ConflictingDeclaration {
        line: 8,
        keyword: "TIME_SYSTEM",
    }));

    // `R` and `UTC` both read as UTC, and each is restated as written.
    for label in ["R", "UTC"] {
        let row = format!(" TIME_SYSTEM                             {label}");
        let text = document('A', &[mode, row.as_str()], &rows);
        let set = parse(&text);
        assert_eq!(set.time_scale(), Some(TimeScale::Utc));
        assert_eq!(set.time_system_label(), Some(label));
        assert_eq!(write_bias_sinex(&set).unwrap(), text);
        let t_utc =
            bias_epoch_instant(BiasEpoch::new(2020, 1, 0).unwrap(), TimeScale::Utc).unwrap();
        assert_eq!(
            set.code_osb_seconds(g01, "C1C", t_utc)
                .value()
                .unwrap()
                .to_bits(),
            ns(1.0).to_bits()
        );
        assert_eq!(
            set.code_osb_seconds(g01, "C1C", t),
            BiasLookup::UnsupportedScale {
                product: Some(TimeScale::Utc),
                query: TimeScale::Gpst,
            }
        );
    }
}

#[test]
fn writers_refuse_time_scales_without_a_label_that_reads_back() {
    let text = "# DCB P1-C1 2026-06\nG01                           0.626       0.000\n";
    for scale in [TimeScale::Tt, TimeScale::Tdb, TimeScale::Glonasst] {
        let options = CodeDcbOptions::new(("P1".to_string(), "C1".to_string()), 2026, 6, scale);
        let mut set = BiasSet::parse_code_dcb(text.as_bytes(), Some(options))
            .unwrap()
            .value;
        assert_eq!(set.time_scale(), Some(scale));
        // A set read from DCB restates its own lines, which state no label.
        assert_eq!(write_code_dcb(&set).unwrap(), text);
        // Written as Bias-SINEX, the writer states a section 4.6 label, and
        // no such label reads back as TT, TDB or GLONASS time.
        set.set_sinex_header(BiasSinexHeader::new(
            "TST",
            BiasEpoch::new(2026, 182, 0).unwrap(),
            "TST",
            Some(BiasEpoch::new(2026, 152, 0).unwrap()),
            Some(BiasEpoch::new(2026, 182, 0).unwrap()),
        ))
        .unwrap();
        assert_eq!(
            write_bias_sinex(&set),
            Err(BiasError::UnsupportedTimeSystem { scale: Some(scale) })
        );
    }
}

#[test]
fn dcb_set_written_as_bias_sinex_reads_back_exactly() {
    let text = "\
# DCB P1-C1 2026-06 G
G01                           0.626       0.050
G02                          -2.069
";
    let mut set = BiasSet::parse_code_dcb(text.as_bytes(), None)
        .unwrap()
        .value;
    assert_eq!(
        write_bias_sinex(&set),
        Err(BiasError::MissingWriterMetadata {
            field: "sinex header"
        })
    );
    set.set_sinex_header(BiasSinexHeader::new(
        "TST",
        BiasEpoch::new(2026, 182, 0).unwrap(),
        "TST",
        Some(BiasEpoch::new(2026, 152, 0).unwrap()),
        Some(BiasEpoch::new(2026, 182, 0).unwrap()),
    ))
    .unwrap();
    let written = write_bias_sinex(&set).unwrap();
    assert!(written.starts_with(
        "%=BIA 1.00 TST 2026:182:00000 TST 2026:152:00000 2026:182:00000 R 00000002\n"
    ));
    assert!(written.ends_with("-BIAS/SOLUTION\n%=ENDBIA\n"));
    // Section 4.6 requires clock-reference observables only for products
    // consistent with the ionosphere-free combination, so none are invented.
    assert!(!written.contains("SATELLITE_CLOCK_REFERENCE_OBSERVABLES"));

    let reparsed = BiasSet::parse_bias_sinex(written.as_bytes())
        .expect("strict read of the written product")
        .value;
    assert_eq!(reparsed.mode(), BiasMode::Relative);
    assert_eq!(reparsed.time_scale(), Some(TimeScale::Gpst));
    assert_eq!(reparsed.records().len(), 2);
    for (read, source) in reparsed.records().iter().zip(set.records()) {
        assert_eq!(read.kind, source.kind);
        assert_eq!(read.target, source.target);
        assert_eq!(read.obs1, source.obs1);
        assert_eq!(read.obs2, source.obs2);
        assert_eq!(read.valid_from, source.valid_from);
        assert_eq!(read.valid_until, source.valid_until);
        assert_eq!(read.value.to_bits(), source.value.to_bits());
        assert_eq!(read.sigma.map(f64::to_bits), source.sigma.map(f64::to_bits));
        assert_eq!(read.unit, BiasUnit::Nanoseconds);
        assert_eq!(read.family, BiasObservableFamily::Code);
    }
}

#[test]
fn header_line_and_footer_follow_section_4_1() {
    let rows = [
        solution_row(Row::osb("G01", "C1C", "ns", "1.0")),
        solution_row(Row::osb("G01", "C1W", "ns", "2.0")),
    ];
    let text = document('A', &ABSOLUTE_G, &rows);
    let set = parse(&text);
    assert!(set.notices().is_empty(), "{:?}", set.notices());
    assert!(set.clock_reference().per_system.is_empty());
    assert_eq!(write_bias_sinex(&set).unwrap(), text);
    let lines = set.source_lines();
    assert_eq!(lines[0].role, BiasLineRole::Header);
    assert_eq!(lines.last().unwrap().role, BiasLineRole::Footer);
    assert_eq!(lines.last().unwrap().text, "%=ENDBIA");

    // A declared count that differs from the rows (section 4.1): refused
    // strictly, reported leniently.
    let miscounted = assemble(header_line('A', 3), &TEST_REFERENCE, &ABSOLUTE_G, &rows);
    let count_departure = BiasDeparture::EstimateCountMismatch {
        declared: 3,
        solution_rows: 2,
    };
    assert_eq!(
        BiasSet::parse_bias_sinex(miscounted.as_bytes()).unwrap_err(),
        BiasError::Departure {
            departure: count_departure.clone()
        }
    );
    let set = BiasSet::parse_bias_sinex_with_policy(miscounted.as_bytes(), BiasReadPolicy::Lenient)
        .unwrap()
        .value;
    assert!(set
        .notices()
        .contains(&BiasNotice::Departure(count_departure)));
    assert!(set
        .diagnostics()
        .warnings
        .iter()
        .any(|warning| warning.kind == WarningKind::Mismatch && warning.at.line == Some(1)));

    // A header mode that disagrees with BIAS_MODE (section 4.1).
    let mismatched_mode = document('R', &ABSOLUTE_G, &rows);
    let mode_departure = BiasDeparture::HeaderModeMismatch {
        header: "R".to_string(),
        description: BiasMode::Absolute,
    };
    assert_eq!(
        BiasSet::parse_bias_sinex(mismatched_mode.as_bytes()).unwrap_err(),
        BiasError::Departure {
            departure: mode_departure.clone()
        }
    );
    let set =
        BiasSet::parse_bias_sinex_with_policy(mismatched_mode.as_bytes(), BiasReadPolicy::Lenient)
            .unwrap()
            .value;
    assert!(set
        .notices()
        .contains(&BiasNotice::Departure(mode_departure)));

    // The count earlier versions of this library wrote after +BIAS/SOLUTION.
    let suffixed = text.replacen("+BIAS/SOLUTION\n", "+BIAS/SOLUTION 2\n", 1);
    assert_eq!(
        BiasSet::parse_bias_sinex(suffixed.as_bytes()).unwrap_err(),
        BiasError::Departure {
            departure: BiasDeparture::BlockStartSuffix { line: 9 }
        }
    );
    let set = BiasSet::parse_bias_sinex_with_policy(suffixed.as_bytes(), BiasReadPolicy::Lenient)
        .unwrap()
        .value;
    assert_eq!(set.records().len(), 2);
    assert_eq!(write_bias_sinex(&set).unwrap(), suffixed);

    // No footer: refused strictly, read and reported leniently, and restated
    // as read.
    let no_footer = text.replace("%=ENDBIA\n", "");
    assert_eq!(
        BiasSet::parse_bias_sinex(no_footer.as_bytes()).unwrap_err(),
        BiasError::Departure {
            departure: BiasDeparture::MissingFooter
        }
    );
    let lenient =
        BiasSet::parse_bias_sinex_with_policy(no_footer.as_bytes(), BiasReadPolicy::Lenient)
            .unwrap()
            .value;
    assert!(lenient
        .notices()
        .contains(&BiasNotice::Departure(BiasDeparture::MissingFooter)));
    assert_eq!(lenient.records().len(), 2);
    assert_eq!(write_bias_sinex(&lenient).unwrap(), no_footer);

    // The short header earlier versions of this library wrote.
    let short = text.replacen(&header_line('A', 2), "%=BIA 1.00 TST", 1);
    assert_eq!(
        BiasSet::parse_bias_sinex(short.as_bytes()).unwrap_err(),
        BiasError::Departure {
            departure: BiasDeparture::HeaderLayout {
                reason: "header line is not 74 columns"
            }
        }
    );
    let lenient = BiasSet::parse_bias_sinex_with_policy(short.as_bytes(), BiasReadPolicy::Lenient)
        .unwrap()
        .value;
    let header = lenient.header().sinex.as_ref().unwrap();
    assert_eq!(header.file_agency.as_deref(), Some("TST"));
    assert_eq!(header.creation_time, None);
    assert_eq!(lenient.records().len(), 2);

    // Unmatched blocks.
    let unclosed = text.replace("-BIAS/SOLUTION\n", "");
    assert_eq!(
        BiasSet::parse_bias_sinex(unclosed.as_bytes()).unwrap_err(),
        BiasError::Departure {
            departure: BiasDeparture::UnclosedBlock {
                name: "BIAS/SOLUTION".to_string(),
                line: 9,
            }
        }
    );
    let mismatched = text.replace("-BIAS/DESCRIPTION\n", "-FILE/COMMENT\n");
    assert_eq!(
        BiasSet::parse_bias_sinex(mismatched.as_bytes()).unwrap_err(),
        BiasError::Departure {
            departure: BiasDeparture::MismatchedBlockEnd {
                open: "BIAS/DESCRIPTION".to_string(),
                close: "FILE/COMMENT".to_string(),
                line: 8,
            }
        }
    );
    for trailing_line in ["trailing text", ""] {
        let trailing = format!("{text}{trailing_line}\n");
        let departure = BiasDeparture::ContentAfterFooter {
            line: text.lines().count() + 1,
        };
        assert_eq!(
            BiasSet::parse_bias_sinex(trailing.as_bytes()).unwrap_err(),
            BiasError::Departure {
                departure: departure.clone()
            },
            "{trailing_line:?}"
        );
        let set =
            BiasSet::parse_bias_sinex_with_policy(trailing.as_bytes(), BiasReadPolicy::Lenient)
                .unwrap()
                .value;
        assert!(set.notices().contains(&BiasNotice::Departure(departure)));
        assert_eq!(write_bias_sinex(&set).unwrap(), trailing);
    }

    // Bias-SINEX 1.00 defines version 1.00 only: refused strictly, read
    // leniently with a notice.
    let other_version = text.replacen("%=BIA 1.00", "%=BIA 1.01", 1);
    assert_eq!(
        BiasSet::parse_bias_sinex(other_version.as_bytes()).unwrap_err(),
        BiasError::UnsupportedVersion {
            version: "1.01".to_string()
        }
    );
    let set =
        BiasSet::parse_bias_sinex_with_policy(other_version.as_bytes(), BiasReadPolicy::Lenient)
            .unwrap()
            .value;
    assert!(set
        .notices()
        .contains(&BiasNotice::Departure(BiasDeparture::OtherVersion {
            version: "1.01".to_string()
        })));
    assert_eq!(set.records().len(), 2);
    assert_eq!(write_bias_sinex(&set).unwrap(), other_version);
}

#[test]
fn nine_character_stations_stay_distinct() {
    let rows = [
        solution_row(Row {
            station: "ABMF00GLP",
            ..Row::osb("G", "C1C", "ns", "1.0")
        }),
        solution_row(Row {
            station: "ABMF00FRA",
            ..Row::osb("G", "C1C", "ns", "2.0")
        }),
        solution_row(Row {
            station: "@MP1JAV-1",
            ..Row::osb("G", "C1W", "ns", "3.0")
        }),
        solution_row(Row {
            svn: "G080",
            station: "ALGO00CAN",
            ..Row::osb("G01", "C1C", "ns", "4.0")
        }),
        // A legacy four-character code.
        solution_row(Row {
            station: "WTZZ",
            ..Row::osb("G", "C1C", "ns", "5.0")
        }),
    ];
    let set = parse(&document('A', &ABSOLUTE_G, &rows));
    let station = |index: usize| match &set.records()[index].target {
        BiasTarget::Receiver { station, .. } | BiasTarget::SatelliteReceiver { station, .. } => {
            station.clone()
        }
        other => panic!("expected a station target, got {other:?}"),
    };
    assert_eq!(station(0), "ABMF00GLP");
    assert_eq!(station(1), "ABMF00FRA");
    assert_eq!(station(2), "@MP1JAV-1");
    assert_eq!(station(3), "ALGO00CAN");
    assert_eq!(
        BiasTargetKey::receiver(GnssSystem::Gps, " abmf00glp ").station,
        Some("ABMF00GLP".to_string())
    );

    let t = epoch(2020, 1, 0);
    let receiver =
        |name: &str, obs: &str| set.receiver_code_osb_seconds(GnssSystem::Gps, name, obs, t);
    assert_eq!(
        receiver("ABMF00GLP", "C1C").value().unwrap().to_bits(),
        ns(1.0).to_bits()
    );
    assert_eq!(
        receiver("abmf00fra", "C1C").value().unwrap().to_bits(),
        ns(2.0).to_bits()
    );
    // Both stations carry the legacy code ABMF, so a query by that code is
    // ambiguous.
    assert_eq!(
        receiver("ABMF", "C1C"),
        BiasLookup::Ambiguous {
            records: vec![0, 1]
        }
    );
    // A nine-character identifier never answers for a different one, even
    // with the same first four characters.
    assert_eq!(receiver("ABMF00XXX", "C1C"), BiasLookup::Absent);
    // A nine-character query answers from the legacy code it carries.
    assert_eq!(
        receiver("WTZZ00DEU", "C1C").value().unwrap().to_bits(),
        ns(5.0).to_bits()
    );
    assert_eq!(
        receiver("@MP1JAV-1", "C1W").value().unwrap().to_bits(),
        ns(3.0).to_bits()
    );
    // A legacy code answers from the one nine-character identifier carrying
    // it.
    assert_eq!(
        set.sat_receiver_code_osb_seconds(sat(GnssSystem::Gps, 1), "ALGO", "C1C", t)
            .value()
            .unwrap()
            .to_bits(),
        ns(4.0).to_bits()
    );
}

#[test]
fn header_rows_keep_order_repeats_and_spacing() {
    let file_reference = [
        " DESCRIPTION        CODE,  Astronomical   Institute",
        " DESCRIPTION        Second  line",
        " CONTACT            code@example.org",
    ];
    let description = [
        " OBSERVATION_SAMPLING                             300",
        " OBSERVATION_SAMPLING                              30",
        " BIAS_MODE                               ABSOLUTE",
        " TIME_SYSTEM                             G  ",
        " SATELLITE_CLOCK_REFERENCE_OBSERVABLES   G  C1W  C2W ",
        " SATELLITE_CLOCK_REFERENCE_OBSERVABLES   E  C1C  C5Q ",
        " SATELLITE_CLOCK_REFERENCE_OBSERVABLES   E  C1X  C5X ",
    ];
    let rows = [solution_row(Row::osb("G01", "C1C", "ns", "1.0"))];
    let text = assemble(header_line('A', 1), &file_reference, &description, &rows);
    let set = parse(&text);
    let header = set.header();
    assert_eq!(header.file_reference.len(), 3);
    assert_eq!(
        header
            .file_reference_values("DESCRIPTION")
            .collect::<Vec<_>>(),
        vec!["CODE,  Astronomical   Institute", "Second  line"]
    );
    assert_eq!(
        header
            .description_values("OBSERVATION_SAMPLING")
            .collect::<Vec<_>>(),
        vec!["300", "30"]
    );
    assert_eq!(header.description.len(), 7);
    assert_eq!(set.time_system_label(), Some("G"));
    // Galileo's two declarations disagree, so neither is used.
    assert_eq!(
        set.clock_reference().per_system.get(&GnssSystem::Gps),
        Some(&("C1W".to_string(), "C2W".to_string()))
    );
    assert_eq!(
        set.clock_reference().per_system.get(&GnssSystem::Galileo),
        None
    );
    assert!(set.notices().contains(&BiasNotice::ConflictingDeclaration {
        line: 14,
        keyword: "SATELLITE_CLOCK_REFERENCE_OBSERVABLES",
    }));
    let written = write_bias_sinex(&set).unwrap();
    assert_eq!(written, text);
    assert_eq!(
        written
            .lines()
            .filter(|line| line.contains("SATELLITE_CLOCK_REFERENCE_OBSERVABLES   G"))
            .count(),
        1
    );
}

#[test]
fn uncertainties_are_kept_whether_or_not_a_slope_is_given() {
    let rows = [
        solution_row(Row::osb("G01", "C1C", "ns", "1.0")),
        solution_row(Row {
            sigma: "0.0000",
            ..Row::osb("G01", "C1W", "ns", "1.0")
        }),
        // The E11.6 spelling section 4.8 shows.
        solution_row(Row {
            sigma: ".398201E-01",
            ..Row::osb("G01", "C2W", "ns", "1.0")
        }),
        solution_row(Row {
            slope_sigma: "0.000100",
            ..Row::osb("G01", "C5Q", "ns", "1.0")
        }),
        solution_row(Row {
            sigma: "-0.0000",
            ..Row::osb("G01", "C1X", "ns", "1.0")
        }),
    ];
    let text = document('A', &ABSOLUTE_G, &rows);
    let set = parse(&text);
    let records = set.records();
    assert_eq!(records[0].sigma, None);
    assert_eq!(records[1].sigma.unwrap().to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        records[2].sigma.unwrap().to_bits(),
        (0.0398201_f64 * NS_TO_S).to_bits()
    );
    assert_eq!(records[3].slope, None);
    assert_eq!(
        records[3].slope_sigma.unwrap().to_bits(),
        ns(0.0001).to_bits()
    );
    assert!(records[4].sigma.unwrap().is_sign_negative());
    assert_eq!(write_bias_sinex(&set).unwrap(), text);
}

#[test]
fn full_precision_extreme_exponents_and_signed_zero_are_read_exactly() {
    let rows = [
        solution_row(Row::osb("G01", "L1C", "cyc", "1.234567890123456E+00")),
        solution_row(Row::osb("G01", "C1C", "ns", "-1.23456789012345E+00")),
        solution_row(Row::osb("G01", "C1W", "ns", "1.0E-300")),
        solution_row(Row::osb("G01", "C2W", "ns", "1.0E+300")),
        solution_row(Row::osb("G01", "C5Q", "ns", "-0.0")),
        solution_row(Row::osb("G01", "L2W", "cyc", "0.000000000000000E+00")),
    ];
    let text = document('A', &ABSOLUTE_G, &rows);
    let set = parse(&text);
    let values: Vec<u64> = set
        .records()
        .iter()
        .map(|record| record.value.to_bits())
        .collect();
    assert_eq!(
        values,
        vec![
            1.234567890123456_f64.to_bits(),
            (-1.23456789012345_f64 * NS_TO_S).to_bits(),
            (1.0e-300_f64 * NS_TO_S).to_bits(),
            (1.0e300_f64 * NS_TO_S).to_bits(),
            (-0.0_f64).to_bits(),
            0.0_f64.to_bits(),
        ]
    );
    assert_eq!(write_bias_sinex(&set).unwrap(), text);
}

#[test]
fn conflicting_duplicates_are_ambiguous_and_every_overlap_is_reported() {
    let rows = [
        solution_row(Row::osb("G01", "C1C", "ns", "1.0")),
        solution_row(Row::osb("G01", "C1C", "ns", "2.0")),
        solution_row(Row::osb("G01", "C1W", "ns", "3.0")),
        solution_row(Row::osb("G01", "C1W", "ns", "3.0")),
        solution_row(Row {
            start: "2020:001:00000",
            end: "2020:011:00000",
            ..Row::osb("G02", "C1C", "ns", "1.0")
        }),
        solution_row(Row {
            start: "2020:003:00000",
            end: "2020:004:00000",
            ..Row::osb("G02", "C1C", "ns", "2.0")
        }),
        solution_row(Row {
            start: "2020:005:00000",
            end: "2020:006:00000",
            ..Row::osb("G02", "C1C", "ns", "3.0")
        }),
        solution_row(Row {
            kind: "DSB",
            obs2: "C1W",
            ..Row::osb("G03", "C1C", "ns", "1.0")
        }),
        solution_row(Row {
            kind: "DSB",
            obs2: "C1W",
            ..Row::osb("G03", "C1C", "ns", "1.5")
        }),
    ];
    let set = parse(&document('A', &ABSOLUTE_G, &rows));
    assert_eq!(set.records().len(), 9);
    let g01 = sat(GnssSystem::Gps, 1);
    let g02 = sat(GnssSystem::Gps, 2);
    let t = epoch(2020, 1, 0);

    // Same interval, different values: neither is chosen.
    assert_eq!(
        set.code_osb_seconds(g01, "C1C", t),
        BiasLookup::Ambiguous {
            records: vec![0, 1]
        }
    );
    // Same interval, same value: one answer.
    assert_eq!(
        set.code_osb_seconds(g01, "C1W", t)
            .value()
            .unwrap()
            .to_bits(),
        ns(3.0).to_bits()
    );
    // Nested intervals: the latest covering start applies.
    let at = |doy: u16, second: u32| {
        set.code_osb_seconds(g02, "C1C", epoch(2020, doy, second))
            .value()
            .unwrap()
            .to_bits()
    };
    assert_eq!(at(2, 0), ns(1.0).to_bits());
    assert_eq!(at(3, 10), ns(2.0).to_bits());
    assert_eq!(at(5, 43_200), ns(3.0).to_bits());
    // Parallel DSB records that disagree.
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 3), "C1C", "C1W", t),
        BiasLookup::Ambiguous {
            records: vec![7, 8]
        }
    );

    // Every overlapping pair, including the long record against each nested
    // one that does not overlap its neighbour.
    let overlaps: Vec<(usize, usize)> = set
        .notices()
        .iter()
        .filter_map(|notice| match notice {
            BiasNotice::Overlap { first, second } => Some((*first, *second)),
            _ => None,
        })
        .collect();
    for pair in [(0, 1), (2, 3), (4, 5), (4, 6), (7, 8)] {
        assert!(overlaps.contains(&pair), "missing overlap {pair:?}");
    }
    assert!(!overlaps.contains(&(5, 6)));
}

#[test]
fn optional_and_unknown_blocks_are_kept_in_place() {
    fn line(input: &mut Vec<u8>, text: &[u8]) {
        input.extend_from_slice(text);
        input.push(b'\n');
    }
    let mut input: Vec<u8> = Vec::new();
    line(&mut input, header_line('A', 1).as_bytes());
    line(
        &mut input,
        b"*-------------------------------------------------------------------------------",
    );
    line(&mut input, b"+FILE/REFERENCE");
    line(&mut input, b" DESCRIPTION        TEST");
    line(&mut input, b"-FILE/REFERENCE");
    line(&mut input, b"+FILE/COMMENT");
    // Line 7: a Latin-1 byte, as in the CODE products' citations.
    line(&mut input, b" Villiger, A., A. J\xe4ggi, 2019:");
    line(&mut input, b"*  a comment inside a block");
    line(&mut input, b"-FILE/COMMENT");
    line(&mut input, b"+INPUT/ACKNOWLEDGMENTS");
    line(
        &mut input,
        b" COD Center for Orbit Determination in Europe\r",
    );
    line(&mut input, b"-INPUT/ACKNOWLEDGMENTS");
    line(&mut input, b"+BIAS/DESCRIPTION");
    line(&mut input, ABSOLUTE_G[0].as_bytes());
    line(&mut input, ABSOLUTE_G[1].as_bytes());
    line(&mut input, b"-BIAS/DESCRIPTION");
    line(&mut input, b"+BIAS/RECEIVER_INFORMATION");
    line(
        &mut input,
        b" MAO0      G @MP0      2015:276:00000 2015:276:86399 JAVAD TRE-G3TH DELTA 3.6.4",
    );
    line(&mut input, b"-BIAS/RECEIVER_INFORMATION");
    line(&mut input, b"+VENDOR/EXTRA");
    line(&mut input, b" anything  at   all");
    line(&mut input, b"-VENDOR/EXTRA");
    line(&mut input, b"+BIAS/SOLUTION");
    line(
        &mut input,
        solution_row(Row::osb("G01", "C1C", "ns", "1.0")).as_bytes(),
    );
    line(&mut input, b"-BIAS/SOLUTION");
    line(&mut input, b"%=ENDBIA");

    // VENDOR/EXTRA is not a block section 2.1 allows: a strict read refuses
    // the file, and a lenient read keeps the block and reports it.
    assert_eq!(
        BiasSet::parse_bias_sinex(&input).unwrap_err(),
        BiasError::Departure {
            departure: BiasDeparture::UnknownBlock {
                name: "VENDOR/EXTRA".to_string(),
                line: 20,
            }
        }
    );
    let set = BiasSet::parse_bias_sinex_with_policy(&input, BiasReadPolicy::Lenient)
        .expect("lenient parse")
        .value;
    assert!(set
        .notices()
        .contains(&BiasNotice::Departure(BiasDeparture::UnknownBlock {
            name: "VENDOR/EXTRA".to_string(),
            line: 20,
        })));
    assert_eq!(set.records().len(), 1);
    // Only the block Bias-SINEX 1.00 section 2.1 does not define is unknown.
    let skips: Vec<(Option<usize>, SkipReason)> = set
        .diagnostics()
        .skips
        .iter()
        .map(|skip| (skip.at.line, skip.reason.clone()))
        .collect();
    assert_eq!(
        skips,
        vec![(
            Some(20),
            SkipReason::UnknownBlock("VENDOR/EXTRA".to_string())
        )]
    );
    assert!(set.notices().contains(&BiasNotice::InvalidUtf8 { line: 7 }));
    let lines = set.source_lines();
    assert_eq!(
        lines[6].bytes.as_deref(),
        Some(&b" Villiger, A., A. J\xe4ggi, 2019:"[..])
    );
    assert_eq!(lines[6].role, BiasLineRole::BlockBody);
    assert_eq!(lines[7].role, BiasLineRole::Comment);
    assert_eq!(
        lines[10].text,
        " COD Center for Orbit Determination in Europe"
    );
    assert_eq!(
        lines[17].text,
        " MAO0      G @MP0      2015:276:00000 2015:276:86399 JAVAD TRE-G3TH DELTA 3.6.4"
    );
    assert_eq!(lines[17].role, BiasLineRole::BlockBody);
    assert_eq!(lines[20].text, " anything  at   all");
    assert_eq!(lines[20].role, BiasLineRole::BlockBody);

    let counts = set.line_counts();
    assert_eq!(counts.lines, 26);
    assert_eq!(counts.block_body, 4);
    assert_eq!(counts.comments, 2);
    assert_eq!(counts.records, 1);

    assert_eq!(
        write_bias_sinex(&set),
        Err(BiasError::InvalidUtf8Line { line: 7 })
    );
    assert_eq!(write_bias_sinex_bytes(&set).unwrap(), input);
}

#[test]
fn dcb_lines_are_all_kept_and_counted() {
    let text = "\
CODE'S MONTHLY GNSS P1-C1 DCB SOLUTION, YEAR 2026, MONTH 06      01-JUL-26 08:42
--------------------------------------------------------------------------------

DIFFERENTIAL (P1-C1) CODE BIASES FOR SATELLITES AND RECEIVERS:

PRN / STATION NAME        VALUE (NS)  RMS (NS)
***   ****************    *****.***   *****.***
G01                           0.626       0.005
G02                           VALUE       0.005
G     ALGO 40104M002         -1.314       0.010
";
    let parsed = BiasSet::parse_code_dcb(text.as_bytes(), None).unwrap();
    let set = parsed.value;
    assert_eq!(set.records().len(), 2);
    assert_eq!(set.skipped_records(), 1);
    assert!(set.notices().contains(&BiasNotice::DcbTimeSystemAssumed));
    let skipped: Vec<&str> = set.skipped_lines().map(|line| line.text.as_str()).collect();
    assert_eq!(
        skipped,
        vec!["G02                           VALUE       0.005"]
    );
    let counts = set.line_counts();
    assert_eq!(counts.lines, 10);
    assert_eq!(counts.records, 2);
    assert_eq!(counts.skipped, 1);
    assert_eq!(counts.blank, 2);
    assert_eq!(counts.other, 5);
    assert_eq!(set.records()[1].line, Some(10));
}

#[test]
fn dcb_metadata_must_describe_every_record() {
    let relative_g = [
        " BIAS_MODE                               RELATIVE",
        " TIME_SYSTEM                             G",
    ];
    let june = |row: Row<'static>| Row {
        kind: "DSB",
        obs2: "C1C",
        start: "2026:152:00000",
        end: "2026:182:00000",
        ..row
    };
    let meta = || {
        CodeDcbOptions::new(
            ("P1".to_string(), "C1".to_string()),
            2026,
            6,
            TimeScale::Gpst,
        )
    };
    let base = june(Row::osb("G01", "C1W", "ns", "0.626"));

    // A record the metadata describes exactly can be written as DCB.
    let mut set = parse(&document('R', &relative_g, &[solution_row(base)]));
    set.set_dcb_meta(meta()).unwrap();
    let written = write_code_dcb(&set).unwrap();
    let reparsed = BiasSet::parse_code_dcb(written.as_bytes(), None)
        .unwrap()
        .value;
    assert_eq!(
        reparsed.records()[0].value.to_bits(),
        set.records()[0].value.to_bits()
    );

    // An end at second 86399 reads as the following midnight, so it is the
    // same month.
    let mut set = parse(&document(
        'R',
        &relative_g,
        &[solution_row(Row {
            end: "2026:181:86399",
            ..base
        })],
    ));
    assert!(set.set_dcb_meta(meta()).is_ok());

    for (row, field) in [
        (
            Row {
                slope: "0.001",
                ..base
            },
            "slope",
        ),
        (
            Row {
                slope_sigma: "0.001",
                ..base
            },
            "slope sigma",
        ),
        (
            Row {
                svn: "G080",
                ..base
            },
            "svn",
        ),
        (
            Row {
                obs1: "C2W",
                obs2: "C2C",
                ..base
            },
            "observables",
        ),
        (
            Row {
                end: "2026:153:00000",
                ..base
            },
            "validity interval",
        ),
    ] {
        let mut set = parse(&document('R', &relative_g, &[solution_row(row)]));
        assert_eq!(
            set.set_dcb_meta(meta()),
            Err(BiasError::DcbRecordMismatch { record: 0, field }),
            "{field}"
        );
        assert_eq!(set.header().dcb_meta, None);
        assert_eq!(
            write_code_dcb(&set),
            Err(BiasError::MissingWriterMetadata { field: "dcb_meta" })
        );
    }

    let mut utc = parse(&document(
        'R',
        &[
            relative_g[0],
            " TIME_SYSTEM                             UTC",
        ],
        &[solution_row(base)],
    ));
    assert_eq!(
        utc.set_dcb_meta(meta()),
        Err(BiasError::UnsupportedTimeSystem {
            scale: Some(TimeScale::Utc)
        })
    );

    // A DCB product cannot be moved to another month under its records.
    let mut dcb = BiasSet::parse_code_dcb(DCB, None).unwrap().value;
    let july = CodeDcbOptions::new(
        ("P1".to_string(), "C1".to_string()),
        2026,
        7,
        TimeScale::Gpst,
    );
    // A DCB product restates its own title, so its metadata stays as read.
    let refused = BiasError::InvalidInput {
        field: "dcb_meta",
        reason: "a CODE DCB product restates its own title",
    };
    assert_eq!(dcb.set_dcb_meta(july), Err(refused.clone()));
    let current = dcb.header().dcb_meta.clone().unwrap();
    assert_eq!(
        dcb.set_dcb_meta(current.clone().with_receiver_system(GnssSystem::Gps)),
        Err(refused)
    );
    assert_eq!(dcb.set_dcb_meta(current), Ok(()));
    assert_eq!(
        dcb.header().dcb_meta.as_ref().map(|meta| meta.month),
        Some(6)
    );
}

#[test]
fn a_later_covering_start_overrides_and_says_so() {
    let rows = [
        solution_row(Row {
            end: "2020:001:00010",
            ..Row::osb("G03", "C1C", "ns", "1.0")
        }),
        solution_row(Row {
            start: "2020:001:00005",
            end: "2020:001:00015",
            ..Row::osb("G03", "C1C", "ns", "2.0")
        }),
    ];
    let set = parse(&document('A', &ABSOLUTE_G, &rows));
    // At second 7 both cover the query; the later start applies, and the
    // record it overrides is named.
    match set.code_osb_seconds(sat(GnssSystem::Gps, 3), "C1C", epoch(2020, 1, 7)) {
        BiasLookup::Available {
            value,
            records,
            overridden,
        } => {
            assert_eq!(value.to_bits(), ns(2.0).to_bits());
            assert_eq!(records, vec![1]);
            assert_eq!(overridden, vec![0]);
        }
        other => panic!("expected an available value, got {other:?}"),
    }
    // At second 3 only the first covers it, and nothing is overridden.
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 3), "C1C", epoch(2020, 1, 3)),
        BiasLookup::Available {
            value: ns(1.0),
            records: vec![0],
            overridden: vec![],
        }
    );
}

#[test]
fn epochs_compare_as_instants_and_86399_ends_the_day() {
    let rows = [
        // 00000..86399: the end reads as the following midnight, so with a
        // slope the value holds at noon, second 43200. RTKLIB reads no
        // validity or slope from Bias-SINEX, so it gives no convention here.
        solution_row(Row {
            end: "2020:001:86399",
            slope: "1.0",
            ..Row::osb("G01", "C1C", "ns", "10.0")
        }),
        // Second 86400 of day 1 is the same instant as second 0 of day 2.
        solution_row(Row {
            start: "2020:001:86400",
            end: "2020:003:00000",
            ..Row::osb("G02", "C1C", "ns", "1.0")
        }),
        solution_row(Row {
            start: "2020:002:00000",
            end: "2020:003:00000",
            ..Row::osb("G02", "C1C", "ns", "2.0")
        }),
    ];
    let set = parse(&document('A', &ABSOLUTE_G, &rows));
    assert_eq!(
        set.records()[0].slope_reference(),
        BiasSlopeReference::Midpoint {
            start: BiasEpoch::new(2020, 1, 0).unwrap(),
            end: BiasEpoch::new(2020, 2, 0).unwrap(),
        }
    );
    let g01 = sat(GnssSystem::Gps, 1);
    assert_eq!(
        set.code_osb_seconds(g01, "C1C", epoch(2020, 1, 43_200))
            .value()
            .unwrap()
            .to_bits(),
        ns(10.0).to_bits()
    );
    assert_eq!(
        set.code_osb_seconds(g01, "C1C", epoch(2020, 1, 0))
            .value()
            .unwrap()
            .to_bits(),
        (ns(10.0) + ns(1.0) * (0.0 - 43_200.0)).to_bits()
    );

    // Both G02 records start at the same instant, so neither overrides the
    // other: they overlap, and their different values are ambiguous.
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 2), "C1C", epoch(2020, 2, 10)),
        BiasLookup::Ambiguous {
            records: vec![1, 2]
        }
    );
    assert!(set.notices().contains(&BiasNotice::Overlap {
        first: 1,
        second: 2
    }));
}

#[test]
fn a_byte_that_is_not_utf8_shifts_no_other_column() {
    // Bias-SINEX: a Latin-1 byte in the station field (columns 15..24).
    let rows = [solution_row(Row {
        station: "ABXC00DEU",
        sigma: "0.0046",
        ..Row::osb("G", "C1C", "ns", "1.25")
    })];
    let text = document('A', &ABSOLUTE_G, &rows);
    let mut input = text.into_bytes();
    let at = input
        .windows(9)
        .position(|window| window == b"ABXC00DEU")
        .unwrap()
        + 2;
    input[at] = 0xe4;
    let set = BiasSet::parse_bias_sinex(&input)
        .expect("strict parse")
        .value;
    let row_line = set.records()[0].line.unwrap();
    assert!(set
        .notices()
        .contains(&BiasNotice::InvalidUtf8 { line: row_line }));
    let record = &set.records()[0];
    assert_eq!(
        record.target,
        BiasTarget::Receiver {
            system: GnssSystem::Gps,
            station: "AB\u{fffd}C00DEU".to_string(),
        }
    );
    assert_eq!(record.obs1, "C1C");
    assert_eq!(record.unit, BiasUnit::Nanoseconds);
    assert_eq!(record.value.to_bits(), ns(1.25).to_bits());
    assert_eq!(record.sigma.unwrap().to_bits(), ns(0.0046).to_bits());
    assert_eq!(write_bias_sinex_bytes(&set).unwrap(), input);

    // CODE DCB: a Latin-1 byte in the station field (columns 6..22).
    let mut dcb =
        b"# DCB P1-C1 2026-06 G\nG     ABXC 97103M001         -1.365       0.050\n".to_vec();
    let at = dcb.windows(4).position(|window| window == b"ABXC").unwrap() + 2;
    dcb[at] = 0xe4;
    let set = BiasSet::parse_code_dcb(&dcb, None).unwrap().value;
    assert_eq!(set.records().len(), 1);
    let record = &set.records()[0];
    assert_eq!(
        record.target,
        BiasTarget::Receiver {
            system: GnssSystem::Gps,
            station: "AB\u{fffd}C 97103M001".to_string(),
        }
    );
    assert_eq!(record.value.to_bits(), ns(-1.365).to_bits());
    assert_eq!(record.sigma.unwrap().to_bits(), ns(0.050).to_bits());
    assert_eq!(
        write_code_dcb(&set),
        Err(BiasError::InvalidUtf8Line { line: 2 })
    );
    assert_eq!(write_code_dcb_bytes(&set).unwrap(), dcb);
}

#[test]
fn code_plus_domes_names_match_only_their_own_spellings() {
    let text = "\
# DCB P1-C1 2026-06 G
G     ABMF 97103M001         -1.365       0.050
G     BRST10004M004           0.250       0.010
";
    let set = BiasSet::parse_code_dcb(text.as_bytes(), None)
        .unwrap()
        .value;
    let t = epoch(2026, 153, 0);
    let dsb =
        |station: &str| set.receiver_code_dsb_seconds(GnssSystem::Gps, station, "C1W", "C1C", t);
    // A nine-character identifier never answers from a code-plus-DOMES name.
    assert_eq!(dsb("ABMF00GLP"), BiasLookup::Absent);
    // The same code and DOMES number, with and without the blank.
    assert_eq!(
        dsb("ABMF97103M001").value().unwrap().to_bits(),
        ns(-1.365).to_bits()
    );
    assert_eq!(
        dsb("BRST 10004M004").value().unwrap().to_bits(),
        ns(0.250).to_bits()
    );
    // Another DOMES number with the same code is another monument.
    assert_eq!(dsb("ABMF 97103M002"), BiasLookup::Absent);
    // The bare code answers from either spelling.
    assert_eq!(dsb("ABMF").value().unwrap().to_bits(), ns(-1.365).to_bits());
    assert_eq!(dsb("BRST").value().unwrap().to_bits(), ns(0.250).to_bits());
}

#[test]
fn an_isb_and_a_dsb_on_one_pair_do_not_overlap() {
    // Bias-SINEX 1.00 example A.5.2 (4B) gives G01 an ISB and a DSB on the
    // same C1W/C2W pair over the same day. They are different quantities.
    let day = |row: Row<'static>| Row {
        start: "2016:271:00000",
        end: "2016:272:00000",
        svn: "G063",
        ..row
    };
    let rows = [
        solution_row(Row {
            kind: "ISB",
            obs2: "C2W",
            sigma: "0",
            ..day(Row::osb("G01", "C1W", "ns", "0"))
        }),
        solution_row(Row {
            kind: "DSB",
            obs2: "C1C",
            sigma: "0.0183",
            ..day(Row::osb("G01", "C1W", "ns", "1.4295"))
        }),
        solution_row(Row {
            kind: "DSB",
            obs2: "C2W",
            sigma: "0.0183",
            ..day(Row::osb("G01", "C1W", "ns", "-7.4709"))
        }),
    ];
    let relative_g = [
        " BIAS_MODE                               RELATIVE",
        " TIME_SYSTEM                             G",
    ];
    let set = parse(&document('R', &relative_g, &rows));
    assert_eq!(set.records().len(), 3);
    assert!(!set
        .notices()
        .iter()
        .any(|notice| matches!(notice, BiasNotice::Overlap { .. })));
    assert!(!set
        .diagnostics()
        .warnings
        .iter()
        .any(|warning| warning.kind == WarningKind::Overlap));
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 1), "C1W", "C2W", epoch(2016, 271, 0)),
        BiasLookup::Available {
            value: ns(-7.4709),
            records: vec![2],
            overridden: vec![],
        }
    );
}

#[test]
fn dcb_titles_read_only_the_time_labels_a_generated_title_states() {
    // "GPS" in a CODE title names the constellation, not a time system.
    let text = "\
CODE'S MONTHLY GPS P1-C1 DCB SOLUTION, YEAR 2026, MONTH 06      01-JUL-26 08:42
G01                           0.626       0.000
";
    let set = BiasSet::parse_code_dcb(text.as_bytes(), None)
        .unwrap()
        .value;
    assert!(set.notices().contains(&BiasNotice::DcbTimeSystemAssumed));
    assert_eq!(set.time_scale(), Some(TimeScale::Gpst));
    // So options on another scale do not contradict the title.
    let utc = CodeDcbOptions::new(
        ("P1".to_string(), "C1".to_string()),
        2026,
        6,
        TimeScale::Utc,
    );
    let set = BiasSet::parse_code_dcb(text.as_bytes(), Some(utc))
        .unwrap()
        .value;
    assert_eq!(set.time_scale(), Some(TimeScale::Utc));

    // A year a DCB title cannot state is refused, not overflowed.
    let far = CodeDcbOptions::new(
        ("P1".to_string(), "C1".to_string()),
        i32::MAX,
        12,
        TimeScale::Gpst,
    );
    assert_eq!(
        BiasSet::parse_code_dcb(b"G01                           0.626\n", Some(far)).unwrap_err(),
        BiasError::InvalidInput {
            field: "year",
            reason: "out of range",
        }
    );
}

#[test]
fn a_generated_dcb_title_states_utc_and_reads_back_as_utc() {
    let rows = [solution_row(Row {
        kind: "DSB",
        obs2: "C1C",
        start: "2026:152:00000",
        end: "2026:182:00000",
        ..Row::osb("G01", "C1W", "ns", "0.626")
    })];
    let description = [
        " BIAS_MODE                               RELATIVE",
        " TIME_SYSTEM                             UTC",
    ];
    let mut set = parse(&document('R', &description, &rows));
    set.set_dcb_meta(CodeDcbOptions::new(
        ("P1".to_string(), "C1".to_string()),
        2026,
        6,
        TimeScale::Utc,
    ))
    .unwrap();
    let written = write_code_dcb(&set).unwrap();
    assert!(written.starts_with("# DCB P1-C1 2026-06 UTC\n"));
    let reparsed = BiasSet::parse_code_dcb(written.as_bytes(), None)
        .unwrap()
        .value;
    assert_eq!(reparsed.time_scale(), Some(TimeScale::Utc));
    assert!(!reparsed
        .notices()
        .contains(&BiasNotice::DcbTimeSystemAssumed));
    assert_eq!(
        reparsed.records()[0].value.to_bits(),
        set.records()[0].value.to_bits()
    );
}

#[test]
fn the_header_line_is_read_at_its_byte_columns() {
    let rows = [solution_row(Row::osb("G01", "C1C", "ns", "1.0"))];
    let text = document('A', &ABSOLUTE_G, &rows);
    // A Latin-1 byte in the creating agency (columns 11..14).
    let mut input = text.into_bytes();
    assert_eq!(&input[11..14], b"TST");
    input[12] = 0xe4;
    let set = BiasSet::parse_bias_sinex(&input)
        .expect("strict parse")
        .value;
    assert!(set.notices().contains(&BiasNotice::InvalidUtf8 { line: 1 }));
    let header = set.header().sinex.as_ref().unwrap();
    assert_eq!(header.file_agency.as_deref(), Some("T\u{fffd}T"));
    assert_eq!(header.creation_time.as_deref(), Some("2020:001:00000"));
    assert_eq!(header.data_agency.as_deref(), Some("TST"));
    assert_eq!(header.mode.as_deref(), Some("A"));
    assert_eq!(header.estimate_count_value(), Some(1));
    assert_eq!(write_bias_sinex_bytes(&set).unwrap(), input);
}

#[test]
fn dsb_routes_agree_hop_by_hop_or_by_exact_decimal_closure() {
    let dsb = |prn: &'static str, obs1: &'static str, obs2: &'static str, value: &'static str| {
        solution_row(Row {
            kind: "DSB",
            obs2,
            ..Row::osb(prn, obs1, "ns", value)
        })
    };
    let sloped = |prn: &'static str,
                  obs1: &'static str,
                  obs2: &'static str,
                  value: &'static str,
                  slope: &'static str| {
        solution_row(Row {
            kind: "DSB",
            obs2,
            slope,
            ..Row::osb(prn, obs1, "ns", value)
        })
    };
    let rows = [
        // G04: two parallel records on one pair, 1e-16 s apart.
        dsb("G04", "C1C", "C1W", "1.0"),
        dsb("G04", "C1C", "C1W", "1.0000001"),
        // G05: two two-hop routes through different observables whose stated
        // values close exactly, 0.1 + 0.2 = 0.15 + 0.15, although their
        // binary sums differ.
        dsb("G05", "C1C", "C1P", "0.1"),
        dsb("G05", "C1P", "C1W", "0.2"),
        dsb("G05", "C1C", "C1X", "0.15"),
        dsb("G05", "C1X", "C1W", "0.15"),
        // G06: parallel first hops whose sums with the second hop round to
        // the same bits, although the hops differ.
        dsb("G06", "C1C", "C1P", "0.001"),
        dsb("G06", "C1C", "C1P", "0.001000000000000002"),
        dsb("G06", "C1P", "C1W", "1.0"),
        // G08: routes whose binary sums fall within the rounding bound but
        // whose stated decimals do not close: 0.1 + 0.2 against 0.3 + 1E-20.
        dsb("G08", "C1C", "C1P", "0.1"),
        dsb("G08", "C1P", "C1W", "0.2"),
        dsb("G08", "C1C", "C1X", "0.3"),
        dsb("G08", "C1X", "C1W", "1E-20"),
        // G09: sloped routes that are the same function of time.
        sloped("G09", "C1C", "C1P", "1.0", "0.1"),
        sloped("G09", "C1P", "C1W", "2.0", "0.2"),
        sloped("G09", "C1C", "C1X", "1.5", "0.15"),
        sloped("G09", "C1X", "C1W", "1.5", "0.15"),
        // G10: sloped routes whose slopes do not close.
        sloped("G10", "C1C", "C1P", "1.0", "0.1"),
        sloped("G10", "C1P", "C1W", "2.0", "0.2"),
        sloped("G10", "C1C", "C1X", "1.5", "0.15"),
        sloped("G10", "C1X", "C1W", "1.5", "0.16"),
    ];
    let relative_g = [
        " BIAS_MODE                               RELATIVE",
        " TIME_SYSTEM                             G",
    ];
    let set = parse(&document('R', &relative_g, &rows));
    let t = epoch(2020, 1, 0);
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 4), "C1C", "C1W", t),
        BiasLookup::Ambiguous {
            records: vec![0, 1]
        }
    );

    // A comparison of the binary sums alone would call G05 a conflict.
    assert_ne!(
        (0.0 + ns(0.1) + ns(0.2)).to_bits(),
        (0.0 + ns(0.15) + ns(0.15)).to_bits()
    );
    // The route first in observable order, through C1P, gives the value.
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 5), "C1C", "C1W", t),
        BiasLookup::Available {
            value: 0.0 + ns(0.1) + ns(0.2),
            records: vec![2, 3],
            overridden: vec![],
        }
    );

    assert_eq!(
        (0.0 + ns(0.001) + ns(1.0)).to_bits(),
        (0.0 + ns(0.001000000000000002) + ns(1.0)).to_bits()
    );
    // The parallel G06 hop disagrees; the conflict names every record on the
    // shortest-route graph.
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 6), "C1C", "C1W", t),
        BiasLookup::Ambiguous {
            records: vec![6, 7, 8]
        }
    );

    let bound =
        |a: f64, b: f64, magnitude: f64| (a - b).abs() <= 4.0 * (f64::EPSILON / 2.0) * magnitude;
    assert!(bound(
        0.0 + ns(0.1) + ns(0.2),
        0.0 + ns(0.3) + ns(1.0e-20),
        ns(0.1) + ns(0.2) + ns(0.3) + ns(1.0e-20)
    ));
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 8), "C1C", "C1W", t),
        BiasLookup::Ambiguous {
            records: vec![9, 10, 11, 12]
        }
    );

    // Slopes close (0.1 + 0.2 = 0.15 + 0.15) and so do value - slope * t_ref,
    // all four sharing one validity day; the C1P route gives the value.
    let hop = |value: f64, slope: f64| ns(value) + ns(slope) * (0.0 - 43_200.0);
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 9), "C1C", "C1W", t),
        BiasLookup::Available {
            value: 0.0 + hop(1.0, 0.1) + hop(2.0, 0.2),
            records: vec![13, 14],
            overridden: vec![],
        }
    );
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 10), "C1C", "C1W", t),
        BiasLookup::Ambiguous {
            records: vec![17, 18, 19, 20]
        }
    );
}

#[test]
fn a_long_chain_of_parallel_dsb_records_resolves_hop_by_hop() {
    // Ten hops with five equal parallel records each: 5^10 record routes,
    // but one route over observables.
    let observables = [
        "C1A", "C1B", "C1D", "C1E", "C1F", "C1G", "C1H", "C1I", "C1J", "C1K", "C1L",
    ];
    let mut rows = Vec::new();
    for pair in observables.windows(2) {
        for _ in 0..5 {
            rows.push(solution_row(Row {
                kind: "DSB",
                obs2: pair[1],
                ..Row::osb("G07", pair[0], "ns", "1.0")
            }));
        }
    }
    let relative_g = [
        " BIAS_MODE                               RELATIVE",
        " TIME_SYSTEM                             G",
    ];
    let set = parse(&document('R', &relative_g, &rows));
    let expected = (0..10).fold(0.0, |sum, _| sum + ns(1.0));
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 7), "C1A", "C1L", epoch(2020, 1, 0)),
        BiasLookup::Available {
            value: expected,
            records: (0..50).collect(),
            overridden: vec![],
        }
    );
}

#[test]
fn a_generated_dcb_title_states_its_time_system_in_its_label() {
    let row = "G01                           0.626       0.000\n";
    let read = |title: &str| {
        BiasSet::parse_code_dcb(format!("{title}\n{row}").as_bytes(), None)
            .unwrap()
            .value
    };
    // Version 2.1.1 of this library wrote UTC as `R`; the month may have one
    // digit.
    let set = read("# DCB P1-C1 2026-06 R");
    assert_eq!(set.time_scale(), Some(TimeScale::Utc));
    assert!(!set.notices().contains(&BiasNotice::DcbTimeSystemAssumed));
    assert_eq!(
        read("# DCB P1-C1 2026-6 G").time_scale(),
        Some(TimeScale::Gpst)
    );
    assert_eq!(
        read("# DCB P1-C1 2026-06 TT").time_scale(),
        Some(TimeScale::Tt)
    );
    assert_eq!(
        read("# DCB P1-C1 2026-06 TDB").time_scale(),
        Some(TimeScale::Tdb)
    );
    // A constellation name is read as the scale it names, with a notice.
    let set = read("# DCB P1-C1 2026-06 GLO");
    assert_eq!(set.time_scale(), Some(TimeScale::Utc));
    assert!(set.notices().contains(&BiasNotice::DcbTimeSystemAlias {
        line: 1,
        label: "GLO".to_string(),
    }));

    // An unknown label: strict refuses, lenient reads the rows with no time
    // scale and reports the departure.
    let unknown = format!("# DCB P1-C1 2026-06 XYZ\n{row}");
    let departure = BiasDeparture::UnknownDcbTimeSystem {
        line: 1,
        label: "XYZ".to_string(),
    };
    assert_eq!(
        BiasSet::parse_code_dcb(unknown.as_bytes(), None).unwrap_err(),
        BiasError::Departure {
            departure: departure.clone()
        }
    );
    let set =
        BiasSet::parse_code_dcb_with_policy(unknown.as_bytes(), None, BiasReadPolicy::Lenient)
            .unwrap()
            .value;
    assert_eq!(set.time_scale(), None);
    assert_eq!(set.header().dcb_meta, None);
    assert_eq!(set.records().len(), 1);
    assert!(set
        .notices()
        .contains(&BiasNotice::Departure(departure.clone())));
    assert_eq!(write_code_dcb(&set).unwrap(), unknown);
    // With options, the options decide under either policy.
    let utc = CodeDcbOptions::new(
        ("P1".to_string(), "C1".to_string()),
        2026,
        6,
        TimeScale::Utc,
    );
    let set = BiasSet::parse_code_dcb(unknown.as_bytes(), Some(utc))
        .unwrap()
        .value;
    assert_eq!(set.time_scale(), Some(TimeScale::Utc));
    assert!(set.notices().contains(&BiasNotice::Departure(departure)));

    // A prose title states no time system, whatever its words.
    let prose = format!("CODE'S MONTHLY UTC P1-C1 DCB SOLUTION, YEAR 2026, MONTH 06\n{row}");
    let set = BiasSet::parse_code_dcb(prose.as_bytes(), None)
        .unwrap()
        .value;
    assert!(set.notices().contains(&BiasNotice::DcbTimeSystemAssumed));
}

#[test]
fn sloped_routes_close_only_with_their_reference_epochs() {
    let dsb = |prn: &'static str,
               obs1: &'static str,
               obs2: &'static str,
               window: (&'static str, &'static str),
               value: &'static str,
               slope: &'static str| {
        solution_row(Row {
            kind: "DSB",
            obs2,
            start: window.0,
            end: window.1,
            slope,
            ..Row::osb(prn, obs1, "ns", value)
        })
    };
    let day = ("2020:001:00000", "2020:002:00000");
    let rows = [
        // G11: t_ref is the midpoint 10 s, the start (open end), the
        // half-second midpoint 0.5 s, and the end 30 s (open start). The
        // slopes close (0.2 + 0.4 = 0.1 + 0.5) and so do value - slope *
        // t_ref (1.0 - 2.0 + 2.0 = 1.05 - 0.05 + 15.0 - 15.0); with the start
        // taken for every reference they would not (3.0 against 16.05).
        dsb(
            "G11",
            "C1C",
            "C1P",
            ("2020:001:00000", "2020:001:00020"),
            "1.0",
            "0.2",
        ),
        dsb(
            "G11",
            "C1P",
            "C1W",
            ("2020:001:00000", "0000:000:00000"),
            "2.0",
            "0.4",
        ),
        dsb(
            "G11",
            "C1C",
            "C1X",
            ("2020:001:00000", "2020:001:00001"),
            "1.05",
            "0.1",
        ),
        dsb(
            "G11",
            "C1X",
            "C1W",
            ("0000:000:00000", "2020:001:00030"),
            "15.0",
            "0.5",
        ),
        // G12: the slope analogue of G08, 0.1 + 0.05 against 0.15 + 1E-20.
        dsb("G12", "C1C", "C1P", day, "1.0", "0.1"),
        dsb("G12", "C1P", "C1W", day, "2.0", "0.05"),
        dsb("G12", "C1C", "C1X", day, "1.5", "0.15"),
        dsb("G12", "C1X", "C1W", day, "1.5", "1E-20"),
    ];
    let relative_g = [
        " BIAS_MODE                               RELATIVE",
        " TIME_SYSTEM                             G",
    ];
    let set = parse(&document('R', &relative_g, &rows));
    let t = epoch(2020, 1, 0);
    let first_hop = ns(1.0) + ns(0.2) * (0.0 - 10.0);
    let second_hop = ns(2.0) + ns(0.4) * 0.0;
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 11), "C1C", "C1W", t),
        BiasLookup::Available {
            value: 0.0 + first_hop + second_hop,
            records: vec![0, 1],
            overridden: vec![],
        }
    );
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 12), "C1C", "C1W", t),
        BiasLookup::Ambiguous {
            records: vec![4, 5, 6, 7]
        }
    );
}

#[test]
fn a_diamond_chain_of_two_to_the_forty_routes_resolves() {
    // Forty diamonds N_k -> {A_k, B_k} -> N_k+1, every hop 1 ns: 160 rows
    // and 2^40 shortest routes, all agreeing.
    let name = |index: usize| format!("C{index:03X}");
    let names: Vec<String> = (0..=120).map(name).collect();
    let mut rows = Vec::new();
    for k in 0..40 {
        let (node, a, b, next) = (3 * k, 3 * k + 1, 3 * k + 2, 3 * k + 3);
        for (from, to) in [(node, a), (a, next), (node, b), (b, next)] {
            rows.push(solution_row(Row {
                kind: "DSB",
                obs2: names[to].as_str(),
                ..Row::osb("G13", names[from].as_str(), "ns", "1.0")
            }));
        }
    }
    let relative_g = [
        " BIAS_MODE                               RELATIVE",
        " TIME_SYSTEM                             G",
    ];
    let set = parse(&document('R', &relative_g, &rows));
    let expected = (0..80).fold(0.0, |sum, _| sum + ns(1.0));
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 13), "C000", "C078", epoch(2020, 1, 0))
            .value()
            .unwrap()
            .to_bits(),
        expected.to_bits()
    );
}

#[test]
fn a_very_long_dsb_chain_resolves_without_recursion() {
    let names: Vec<String> = (0..=3000).map(|index| format!("C{index:03X}")).collect();
    let rows: Vec<String> = names
        .windows(2)
        .map(|pair| {
            solution_row(Row {
                kind: "DSB",
                obs2: pair[1].as_str(),
                ..Row::osb("G14", pair[0].as_str(), "ns", "1.0")
            })
        })
        .collect();
    let relative_g = [
        " BIAS_MODE                               RELATIVE",
        " TIME_SYSTEM                             G",
    ];
    let set = parse(&document('R', &relative_g, &rows));
    let expected = (0..3000).fold(0.0, |sum, _| sum + ns(1.0));
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 14), "C000", "CBB8", epoch(2020, 1, 0))
            .value()
            .unwrap()
            .to_bits(),
        expected.to_bits()
    );
}

#[test]
fn generated_dcb_titles_on_tt_and_tdb_read_back() {
    for (label, scale) in [("TT", TimeScale::Tt), ("TDB", TimeScale::Tdb)] {
        let rows = [solution_row(Row {
            kind: "DSB",
            obs2: "C1C",
            start: "2026:152:00000",
            end: "2026:182:00000",
            ..Row::osb("G01", "C1W", "ns", "0.626")
        })];
        let time_row = format!(" TIME_SYSTEM                             {label}");
        let description = [
            " BIAS_MODE                               RELATIVE",
            time_row.as_str(),
        ];
        // TT and TDB are not section 4.6 labels, so the Bias-SINEX product is
        // read leniently.
        let mut set = BiasSet::parse_bias_sinex_with_policy(
            document('R', &description, &rows).as_bytes(),
            BiasReadPolicy::Lenient,
        )
        .unwrap()
        .value;
        assert_eq!(set.time_scale(), Some(scale));
        set.set_dcb_meta(CodeDcbOptions::new(
            ("P1".to_string(), "C1".to_string()),
            2026,
            6,
            scale,
        ))
        .unwrap();
        let written = write_code_dcb(&set).unwrap();
        assert!(written.starts_with(&format!("# DCB P1-C1 2026-06 {label}\n")));
        let reparsed = BiasSet::parse_code_dcb(written.as_bytes(), None)
            .unwrap()
            .value;
        assert_eq!(reparsed.time_scale(), Some(scale));
        assert!(!reparsed
            .notices()
            .contains(&BiasNotice::DcbTimeSystemAssumed));
        assert_eq!(
            reparsed.records()[0].value.to_bits(),
            set.records()[0].value.to_bits()
        );
    }
}

#[test]
fn parallel_dsb_records_stated_in_opposite_directions_compare_as_one_hop() {
    let dsb = |prn: &'static str, obs1: &'static str, obs2: &'static str, value: &'static str| {
        solution_row(Row {
            kind: "DSB",
            obs2,
            ..Row::osb(prn, obs1, "ns", value)
        })
    };
    let rows = [
        // C1C - C1W = 1 ns, stated once each way: the same bias.
        dsb("G15", "C1C", "C1W", "1.0"),
        dsb("G15", "C1W", "C1C", "-1.0"),
        // C1C - C1W = 1 ns and C1W - C1C = 1 ns contradict each other.
        dsb("G16", "C1C", "C1W", "1.0"),
        dsb("G16", "C1W", "C1C", "1.0"),
    ];
    let relative_g = [
        " BIAS_MODE                               RELATIVE",
        " TIME_SYSTEM                             G",
    ];
    let set = parse(&document('R', &relative_g, &rows));
    let t = epoch(2020, 1, 0);
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 15), "C1C", "C1W", t),
        BiasLookup::Available {
            value: 0.0 + ns(1.0),
            records: vec![0, 1],
            overridden: vec![],
        }
    );
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 16), "C1C", "C1W", t),
        BiasLookup::Ambiguous {
            records: vec![2, 3]
        }
    );
}

#[test]
fn a_dcb_title_alias_that_disagrees_with_the_options_is_reported() {
    let text = "# DCB P1-C1 2026-06 GAL\nG01                           0.626       0.000\n";
    let gpst = CodeDcbOptions::new(
        ("P1".to_string(), "C1".to_string()),
        2026,
        6,
        TimeScale::Gpst,
    );
    let set = BiasSet::parse_code_dcb(text.as_bytes(), Some(gpst))
        .expect("the options decide")
        .value;
    assert_eq!(set.time_scale(), Some(TimeScale::Gpst));
    assert!(set.notices().contains(&BiasNotice::DcbTimeSystemAlias {
        line: 1,
        label: "GAL".to_string(),
    }));
}
