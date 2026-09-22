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
    bias_epoch_instant, ionosphere_free_coefficients, write_bias_sinex, write_code_dcb, BiasEpoch,
    BiasError, BiasKind, BiasMode, BiasSet, BiasTarget, CodeDcbOptions, FieldError, SkipReason,
    WarningKind,
};
use sidereon_core::constants::{C_M_S, F_L1_HZ, F_L2_HZ, NS_TO_S};
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

    let parsed = BiasSet::parse_code_dcb(dcb_text.as_bytes(), Some(opts)).unwrap();
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
            .unwrap()
            .to_bits(),
        (-1.365 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF", "C1W", "C1C", t0)
            .unwrap()
            .to_bits(),
        (-1.365 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "ab-1", "C1W", "C1C", t0)
            .unwrap()
            .to_bits(),
        (2.500 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "AB-1", "C1W", "C1C", t0)
            .unwrap()
            .to_bits(),
        (2.500 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "st_01", "C1W", "C1C", t0)
            .unwrap()
            .to_bits(),
        (-0.750 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "ST_01", "C1W", "C1C", t0)
            .unwrap()
            .to_bits(),
        (-0.750 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "in sp 01", "C1W", "C1C", t0)
            .unwrap()
            .to_bits(),
        (1.000 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        set.receiver_code_dsb_seconds(GnssSystem::Gps, "IN SP 01", "C1W", "C1C", t0)
            .unwrap()
            .to_bits(),
        (1.000 * 1.0e-9_f64).to_bits()
    );

    // Writing produces explicit system prefix in columns 0..6
    let written = write_code_dcb(&set).unwrap();
    assert!(written
        .lines()
        .any(|l| l.starts_with("G     abmf            ") && l.ends_with("    0.050")));
    assert!(written
        .lines()
        .any(|l| l.starts_with("G     ab-1            ") && l.ends_with("    0.010")));
    assert!(written
        .lines()
        .any(|l| l.starts_with("G     st_01           ") && l.ends_with("    0.020")));
    assert!(written
        .lines()
        .any(|l| l.starts_with("G     in sp 01        ") && l.ends_with("    0.000")));

    // Reparsing without options parses the explicit system records
    let reparsed = BiasSet::parse_code_dcb(written.as_bytes(), None)
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
            .unwrap()
            .to_bits(),
        (-1.365 * 1.0e-9_f64).to_bits()
    );
    assert_eq!(
        reparsed
            .receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF", "C1W", "C1C", t0)
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
            .unwrap()
            .to_bits(),
        ns(-1.365).to_bits()
    );
    assert_eq!(
        set_abmf
            .receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF", "C1W", "C1C", t0)
            .unwrap()
            .to_bits(),
        ns(-1.365).to_bits()
    );

    let written_abmf = write_code_dcb(&set_abmf).unwrap();
    assert!(written_abmf
        .lines()
        .any(|l| l.starts_with("G     ABMF97103M001")));
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
            .unwrap()
            .to_bits(),
        ns(-1.365).to_bits()
    );
    assert_eq!(
        reparsed_abmf
            .receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF", "C1W", "C1C", t0)
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
