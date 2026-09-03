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

use sidereon_core::astro::time::model::TimeScale;
use sidereon_core::bias::{
    bias_epoch_instant, ionosphere_free_coefficients, write_bias_sinex, BiasEpoch, BiasError,
    BiasKind, BiasMode, BiasSet, BiasTarget, CodeDcbOptions, SkipReason, WarningKind,
};
use sidereon_core::constants::{C_M_S, F_L1_HZ, F_L2_HZ};
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
%=BIA 1.00 TST
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
+BIAS/SOLUTION 11
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
"
}

#[test]
fn bias_sinex_parse_round_trips_fixture() {
    let parsed = BiasSet::parse_bias_sinex(BIA).expect("parse Bias-SINEX fixture");
    let set = parsed.value;
    assert_eq!(set.mode, BiasMode::Absolute);
    assert_eq!(set.records().len(), 351);
    assert!(set.diagnostics().skips.iter().any(
        |skip| matches!(skip.reason, SkipReason::UnknownBlock(ref name) if name == "FILE/COMMENT")
    ));
    assert_eq!(
        set.clock_reference.per_system.get(&GnssSystem::Gps),
        Some(&("C1W".to_string(), "C2W".to_string()))
    );

    let g01 = sat(GnssSystem::Gps, 1);
    assert_eq!(
        set.code_osb_seconds(g01, "C1C", epoch(2026, 181, 0))
            .unwrap()
            .to_bits(),
        ns(-6.2069).to_bits()
    );
    assert_eq!(
        set.code_osb_seconds(g01, "C1W", epoch(2026, 181, 0))
            .unwrap()
            .to_bits(),
        ns(-5.2579).to_bits()
    );
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Glonass, 2), "C1P", epoch(2026, 181, 0))
            .unwrap()
            .to_bits(),
        ns(1.7840).to_bits()
    );

    let encoded = write_bias_sinex(&set).expect("write Bias-SINEX");
    let reparsed = BiasSet::parse_bias_sinex(encoded.as_bytes())
        .expect("reparse Bias-SINEX")
        .value;
    assert_eq!(set.records(), reparsed.records());
    assert_eq!(reparsed.skipped_records(), 0);
}

#[test]
fn code_dcb_parse_round_trips_fixture() {
    let parsed = BiasSet::parse_code_dcb(DCB, None).expect("parse DCB fixture");
    let set = parsed.value;
    assert_eq!(set.mode, BiasMode::Relative);
    assert_eq!(set.skipped_records(), 2);
    assert_eq!(set.records().len(), 496);

    let g01 = sat(GnssSystem::Gps, 1);
    assert_eq!(
        set.code_dsb_seconds(g01, "C1W", "C1C", epoch(2026, 153, 0))
            .unwrap()
            .to_bits(),
        ns(0.626).to_bits()
    );
    assert_eq!(
        set.code_dsb_seconds(g01, "C1C", "C1W", epoch(2026, 153, 0))
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
        .unwrap()
        .to_bits(),
        ns(0.291).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(
            GnssSystem::Gps,
            "algo00xxx",
            "C1W",
            "C1C",
            epoch(2026, 153, 0),
        )
        .unwrap()
        .to_bits(),
        ns(-1.314).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(
            GnssSystem::Glonass,
            "ALGO",
            "C1P",
            "C1C",
            epoch(2026, 153, 0),
        )
        .unwrap()
        .to_bits(),
        ns(0.218).to_bits()
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
%=BIA 1.00 TST
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
+BIAS/SOLUTION 4
 OSB  G063 G01           C1C       2020:001:00000 2020:002:00000 ns      1.000000000000E+00 1.00000E-02
 OSB  G063 G01           C1W       bad-start       2020:002:00000 ns      1.000000000000E+00 1.00000E-02
 OSB  G063 G01           C2W       2020:001:00000 2020:002:00000 bad     1.000000000000E+00 1.00000E-02
 OSB  G063 G01           C5Q       2020:001:00000 2020:002:00000 ns      bad-value            1.00000E-02
-BIAS/SOLUTION
";
    let parsed = BiasSet::parse_bias_sinex(text.as_bytes()).expect("forgiving parse");
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
            .unwrap()
            .to_bits(),
        ns(3.100000000000).to_bits()
    );
    assert_eq!(
        set.receiver_code_osb_seconds(GnssSystem::Galileo, "ALGO", "C1C", epoch(2020, 1, 0))
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
        .unwrap()
        .to_bits(),
        ns(9.900000000000).to_bits()
    );
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 1), "L1C", epoch(2020, 1, 0)),
        None
    );
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 1), "C1C", epoch(2019, 365, 86_399)),
        None
    );
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 1), "C1C", epoch(2020, 2, 0)),
        None
    );
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 1), "C2W", epoch(2020, 3, 86_399))
            .unwrap()
            .to_bits(),
        ns(-0.300000000000).to_bits()
    );
    assert_eq!(
        set.code_osb_seconds(sat(GnssSystem::Gps, 1), "C2W", epoch(2020, 4, 0)),
        None
    );

    let drifted = set
        .code_osb_seconds(sat(GnssSystem::Gps, 1), "C1C", epoch(2020, 1, 10))
        .unwrap();
    let expected = ns(-1.234567890000) + ns(0.864) * 10.0;
    assert_eq!(drifted.to_bits(), expected.to_bits());
}

#[test]
fn overlap_selection_uses_latest_covering_start_and_warns() {
    let lines = vec![
        "%=BIA 1.00 TST".to_string(),
        "+FILE/REFERENCE".to_string(),
        " DESCRIPTION TEST".to_string(),
        "-FILE/REFERENCE".to_string(),
        "+BIAS/DESCRIPTION".to_string(),
        " BIAS_MODE ABSOLUTE".to_string(),
        " TIME_SYSTEM G".to_string(),
        " SATELLITE_CLOCK_REFERENCE_OBSERVABLES G C1W C2W".to_string(),
        "-BIAS/DESCRIPTION".to_string(),
        "+BIAS/SOLUTION 2".to_string(),
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
    let osb_m = set.code_osb_seconds(g01, "C1W", epoch(2020, 1, 0)).unwrap() * C_M_S;
    assert_eq!(osb_m.to_bits(), (ns(0.560000000000) * C_M_S).to_bits());
    let phase_m = set.phase_osb_cycles(g01, "L1C", epoch(2020, 1, 0)).unwrap() * (C_M_S / F_L1_HZ);
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
        "%=BIA 1.00 TST".to_string(),
        "+FILE/REFERENCE".to_string(),
        " DESCRIPTION TEST".to_string(),
        "-FILE/REFERENCE".to_string(),
        "+BIAS/DESCRIPTION".to_string(),
        " BIAS_MODE RELATIVE".to_string(),
        " TIME_SYSTEM G".to_string(),
        " SATELLITE_CLOCK_REFERENCE_OBSERVABLES G C1W C2W".to_string(),
        "-BIAS/DESCRIPTION".to_string(),
        "+BIAS/SOLUTION 2".to_string(),
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
    ];
    let text = lines.join("\n");
    let set = BiasSet::parse_bias_sinex(text.as_bytes()).unwrap().value;
    assert_eq!(
        set.code_dsb_seconds(sat(GnssSystem::Gps, 1), "C1C", "C1W", epoch(2020, 1, 0))
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
        Some(0.0)
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
            .unwrap(),
    );
    got.insert(
        "E",
        set.receiver_code_osb_seconds(GnssSystem::Galileo, "ALGO", "C1C", epoch(2020, 1, 0))
            .unwrap(),
    );
    assert_ne!(got["G"].to_bits(), got["E"].to_bits());
}
