#![cfg(sidereon_repo_tests)]

//! RINEX clock product evaluation tests.
//!
//! Real CLK/SP3 fixtures are public IGS final products mirrored in
//! `tests/fixtures/clk/IGS0OPSFIN_20261330000_90M_30S_CLK.CLK` and
//! `tests/fixtures/sp3/IGS0OPSFIN_20261330000_03H_15M_ORB.SP3`.
//! The real-product identity checks compare evaluated values against parsed
//! record rows, and compare shared CLK/SP3 record epochs to the SP3 clock field
//! resolution.

use sidereon_core::astro::time::civil::seconds_between_splits;
use sidereon_core::astro::time::model::{Instant, TimeScale};
use sidereon_core::constants::SECONDS_PER_DAY;
use sidereon_core::ephemeris::Sp3;
use sidereon_core::rinex::clock::{
    civil_to_clock_instant, civil_to_gps_seconds, ClockEpoch, ClockPoint, RinexClock,
    RinexClockError, RinexClockSkip,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

const CLK: &str = include_str!("fixtures/clk/synthetic_rinex_clock.clk");
const REAL_CLK: &str = include_str!("fixtures/clk/IGS0OPSFIN_20261330000_90M_30S_CLK.CLK");
const REAL_SP3: &str = include_str!("fixtures/sp3/IGS0OPSFIN_20261330000_03H_15M_ORB.SP3");

fn gps(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS PRN")
}

fn real_clk_source_rows() -> Vec<(String, Instant, f64)> {
    REAL_CLK
        .lines()
        .filter(|line| line.starts_with("AS "))
        .map(|line| {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            assert!(
                fields.len() >= 10,
                "real CLK AS row has too few fields: {line}"
            );
            let year = fields[2].parse::<i32>().expect("CLK source year");
            let month = fields[3].parse::<u8>().expect("CLK source month");
            let day = fields[4].parse::<u8>().expect("CLK source day");
            let hour = fields[5].parse::<u8>().expect("CLK source hour");
            let minute = fields[6].parse::<u8>().expect("CLK source minute");
            let second = fields[7].parse::<f64>().expect("CLK source second");
            let epoch =
                civil_to_clock_instant(TimeScale::Gpst, year, month, day, hour, minute, second)
                    .expect("CLK source epoch");
            let bias_s = fields[9].parse::<f64>().expect("CLK source bias");
            (fields[1].to_string(), epoch, bias_s)
        })
        .collect()
}

#[test]
fn real_clk_record_epochs_evaluate_to_parsed_rows() {
    let clock = RinexClock::parse(REAL_CLK).expect("real RINEX clock");
    let mut checked = 0usize;

    for (satellite, epoch, bias_s) in real_clk_source_rows() {
        let evaluated = clock
            .clock_s_at_instant(&satellite, epoch)
            .expect("valid clock query")
            .expect("clock record at source epoch");
        assert_eq!(
            evaluated.to_bits(),
            bias_s.to_bits(),
            "{satellite} source-epoch clock identity"
        );
        checked += 1;
    }

    assert_eq!(checked, 5_792);
}

#[test]
fn real_clk_and_sp3_clocks_match_at_shared_record_epochs() {
    let clock = RinexClock::parse(REAL_CLK).expect("real RINEX clock");
    let sp3 = Sp3::parse(REAL_SP3.as_bytes()).expect("real SP3");
    let mut checked = 0usize;

    for epoch_index in [0usize, 6] {
        let epoch = sp3.epochs[epoch_index];
        for prn in 1..=32 {
            let satellite = gps(prn);
            let sp3_clock_s = sp3
                .state(satellite, epoch_index)
                .expect("SP3 satellite state")
                .clock_s
                .expect("SP3 clock record");
            let clk_clock_s = clock
                .clock_s_at_instant(&satellite.to_string(), epoch)
                .expect("valid CLK query")
                .expect("CLK clock at shared source epoch");
            assert!(
                (clk_clock_s - sp3_clock_s).abs() <= 5.0e-13,
                "{satellite} epoch {epoch_index} CLK {clk_clock_s:e} SP3 {sp3_clock_s:e}"
            );
            checked += 1;
        }
    }

    assert_eq!(checked, 64);
}

#[test]
fn parses_satellite_clock_records_and_ignores_receivers() {
    let clock = RinexClock::parse(CLK).expect("RINEX clock");
    assert_eq!(clock.time_scale, TimeScale::Gpst);
    let sats = clock.series.keys().cloned().collect::<Vec<_>>();
    assert_eq!(sats, vec!["G05".to_string(), "G24".to_string()]);
    assert_eq!(clock.series["G05"].len(), 3);
    assert_eq!(clock.series["G24"].len(), 2);
}

#[test]
fn exact_and_interpolated_biases_match_legacy_bits() {
    let clock = RinexClock::parse(CLK).expect("RINEX clock");

    let g05 = clock
        .clock_s("G05", epoch(2026, 5, 13, 0, 0, 30.0))
        .expect("valid clock query")
        .expect("G05 exact clock");
    assert_eq!(g05.to_bits(), 0xbf2a36e36f0d4275);

    let g24_exact = clock
        .clock_s("G24", epoch(2026, 5, 13, 0, 0, 0.0))
        .expect("valid clock query")
        .expect("G24 exact clock");
    assert_eq!(g24_exact.to_bits(), 0x3f0a36e2eb1c432d);

    let g24_mid = clock
        .clock_s("G24", epoch(2026, 5, 13, 0, 0, 15.0))
        .expect("valid clock query")
        .expect("G24 interpolated clock");
    assert_eq!(g24_mid.to_bits(), 0x3f0a36e4a2ea40ca);
}

#[test]
fn outside_span_and_unknown_satellite_have_no_clock() {
    let clock = RinexClock::parse(CLK).expect("RINEX clock");
    assert_eq!(
        clock
            .clock_s("G99", epoch(2026, 5, 13, 0, 0, 15.0))
            .expect("valid clock query"),
        None
    );
    assert_eq!(
        clock
            .clock_s("G05", epoch(2026, 5, 12, 23, 59, 0.0))
            .expect("valid clock query"),
        None
    );
    assert_eq!(
        clock
            .clock_s("G05", epoch(2026, 5, 13, 1, 0, 0.0))
            .expect("valid clock query"),
        None
    );
}

#[test]
fn duplicate_time_tags_keep_the_last_record() {
    let text = "AS G05  2026 05 13 00 00  0.000000  1   1.0e-04\n\
                AS G05  2026 05 13 00 00  0.000000  1   2.0e-04\n";
    let clock = RinexClock::parse(text).expect("RINEX clock");
    let bias = clock
        .clock_s("G05", epoch(2026, 5, 13, 0, 0, 0.0))
        .expect("valid clock query")
        .expect("duplicate point");
    assert_eq!(bias.to_bits(), (2.0e-4_f64).to_bits());
}

#[test]
fn rounded_fractional_second_carries_to_next_second() {
    let text = "AS G05  2026 05 13 00 00 59.9999996  1   1.0e-04\n";
    let clock = RinexClock::parse(text).expect("rounded clock epoch must parse");
    let expected = civil_to_gps_seconds(2026, 5, 13, 0, 1, 0.0).expect("next minute");

    assert_eq!(
        clock.series["G05"][0]
            .gps_seconds()
            .expect("GPST sample")
            .to_bits(),
        expected.to_bits()
    );
    assert_eq!(
        civil_to_gps_seconds(2026, 5, 13, 0, 0, 59.9999996)
            .expect("rounded public epoch")
            .to_bits(),
        expected.to_bits()
    );
}

#[test]
fn utc_time_system_preserves_scale_and_queries_by_utc_instant() {
    let text = " 3.00           C                                       RINEX VERSION / TYPE\n\
                UTC                                                     TIME SYSTEM ID\n\
                                                                    END OF HEADER\n\
                AS G05  2017 01 01 00 00  0.000000  1   1.0e-04\n\
                AS G05  2017 01 01 00 00 30.000000  1   2.0e-04\n";
    let clock = RinexClock::parse(text).expect("UTC RINEX clock");

    assert_eq!(clock.time_scale, TimeScale::Utc);
    assert_eq!(clock.series["G05"][0].epoch.scale, TimeScale::Utc);
    assert_eq!(clock.series_rows(), vec![("G05".to_string(), vec![])]);
    let interpolated = clock
        .clock_s("G05", epoch(2017, 1, 1, 0, 0, 15.0))
        .expect("valid clock query")
        .expect("UTC interpolated clock");
    assert!((interpolated - 1.5e-4).abs() < 1.0e-18);

    let gpst_query =
        civil_to_clock_instant(TimeScale::Gpst, 2017, 1, 1, 0, 0, 15.0).expect("GPST instant");
    assert_eq!(
        clock
            .clock_s_at_instant("G05", gpst_query)
            .expect("valid clock query"),
        None
    );

    let rows = clock.instant_series_rows();
    assert_eq!(rows[0].1[0].0.scale, TimeScale::Utc);
    let rebuilt = RinexClock::from_instant_series_rows(clock.time_scale, rows)
        .expect("valid manual RINEX clock rows");
    assert_eq!(rebuilt, clock);
}

#[test]
fn rinex_clock_utc_leap_second_interval_to_midnight_interpolates_forward() {
    let text = " 3.00           C                                       RINEX VERSION / TYPE\n\
                UTC                                                     TIME SYSTEM ID\n\
                                                                    END OF HEADER\n\
                AS G05  2016 12 31 23 59 60.250000  1   1.0e-04\n\
                AS G05  2017 01 01 00 00  0.000000  1   4.0e-04\n";
    let clock = RinexClock::parse(text).expect("UTC leap-second RINEX clock");
    let points = &clock.series["G05"];
    assert_eq!(points.len(), 2);

    let leap = points[0].epoch.julian_date().expect("leap-second split");
    let midnight = points[1].epoch.julian_date().expect("midnight split");
    assert_eq!(leap.jd_whole.to_bits(), midnight.jd_whole.to_bits());
    assert!(leap.fraction < midnight.fraction);
    let span_s = seconds_between_splits(
        midnight.jd_whole,
        midnight.fraction,
        leap.jd_whole,
        leap.fraction,
    );
    assert!((span_s - 0.75).abs() < 1.0e-12);

    let interpolated = clock
        .clock_s("G05", epoch(2016, 12, 31, 23, 59, 60.625))
        .expect("valid clock query")
        .expect("leap-second interpolation");
    assert!((interpolated - 2.5e-4).abs() < 1.0e-18);
}

#[test]
fn rejects_gps_time_leap_second_label() {
    let text = "AS G05  2016 12 31 23 59 60.000000  1   1.0e-04\n";
    let err = RinexClock::parse(text).expect_err("GPS-time clock leap second must error");
    assert_eq!(
        err,
        RinexClockError::BadField {
            line: 1,
            field: "epoch",
            value: "2016 12 31 23 59 60".to_string(),
        }
    );
    assert_eq!(civil_to_gps_seconds(2016, 12, 31, 23, 59, 60.0), None);
}

#[test]
fn strict_parse_reports_short_as_records() {
    let text = "AS G05  2026 05 13 00 00  0.000000  1\n";
    let err = RinexClock::parse(text).expect_err("short AS record must error");
    assert_eq!(
        err,
        RinexClockError::MalformedAsRecord {
            line: 1,
            reason: "expected at least 10 fields",
            record: "AS G05  2026 05 13 00 00  0.000000  1".to_string(),
        }
    );
}

#[test]
fn strict_parse_reports_bad_as_fields() {
    let text = "AS G05  2026 05 13 00 00  bad-second  1   1.0e-04\n";
    let err = RinexClock::parse(text).expect_err("bad AS field must error");
    assert_eq!(
        err,
        RinexClockError::BadField {
            line: 1,
            field: "second",
            value: "bad-second".to_string(),
        }
    );
}

#[test]
fn strict_parse_rejects_malformed_fractional_second() {
    let text = "AS G05  2026 05 13 00 00  59.  1   1.0e-04\n";
    let err = RinexClock::parse(text).expect_err("malformed AS fraction must error");
    assert_eq!(
        err,
        RinexClockError::BadField {
            line: 1,
            field: "second",
            value: "59.".to_string(),
        }
    );
}

#[test]
fn strict_parse_rejects_invalid_leap_second_range() {
    for second in ["61.000000", "-1.000000"] {
        let text = format!("AS G05  2016 12 31 23 59 {second:>10}  1   1.0e-04\n");
        let err = RinexClock::parse(&text).expect_err("invalid AS second must error");
        assert_eq!(
            err,
            RinexClockError::BadField {
                line: 1,
                field: "epoch",
                value: format!("2016 12 31 23 59 {}", second.parse::<f64>().unwrap()),
            }
        );
    }
}

#[test]
fn strict_parse_rejects_invalid_civil_date() {
    let text = "AS G05  2026 13 31 23 59  0.000000  1   1.0e-04\n";
    let err = RinexClock::parse(text).expect_err("invalid AS date must error");
    assert_eq!(
        err,
        RinexClockError::BadField {
            line: 1,
            field: "epoch",
            value: "2026 13 31 23 59 0".to_string(),
        }
    );
}

#[test]
fn parse_lossy_keeps_legacy_skip_behavior() {
    let text = "AS G05  2026 05 13 00 00  0.000000  1   1.0e-04\n\
                AS G06  2026 05 13 00 00  bad-second  1   2.0e-04\n";
    let clock = RinexClock::parse_lossy(text);
    assert_eq!(
        clock.series.keys().cloned().collect::<Vec<_>>(),
        vec!["G05"]
    );
    assert_eq!(
        clock
            .clock_s("G05", epoch(2026, 5, 13, 0, 0, 0.0))
            .expect("valid clock query")
            .expect("G05 clock")
            .to_bits(),
        (1.0e-4_f64).to_bits()
    );
    assert_eq!(clock.diagnostics.len(), 1);
    assert_eq!(clock.diagnostics[0].line, 2);
}

#[test]
fn civil_gps_seconds_match_gps_epoch_boundary() {
    assert_eq!(
        civil_to_gps_seconds(1980, 1, 6, 0, 0, 0.0).expect("GPS epoch"),
        0.0
    );
    assert_eq!(
        civil_to_gps_seconds(1980, 1, 7, 0, 0, 0.0).expect("next day"),
        SECONDS_PER_DAY
    );
}

fn epoch(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: f64) -> ClockEpoch {
    ClockEpoch {
        year,
        month,
        day,
        hour,
        minute,
        second,
    }
}

#[test]
fn parse_reports_unmodelled_clock_records_while_retaining_modelled_satellites() {
    let text = "\
 3.00           C                                       RINEX VERSION / TYPE
 2    AR    AS                                          # / TYPES OF DATA
                                                        END OF HEADER
AS G01  2026 05 13 00 00  0.000000  1   1.000000000000e-04
AR ALIC 2026 05 13 00 00  0.000000  2   2.000000000000e-04 1.0e-10
AS G02  2026 05 13 00 00  0.000000  1   3.000000000000e-04
CR ALGO 2026 05 13 00 00  0.000000  2   4.000000000000e-04 1.0e-10
DR AREQ 2026 05 13 00 00  0.000000  2   5.000000000000e-04 1.0e-10
MS ASCG 2026 05 13 00 00  0.000000  2   6.000000000000e-04 1.0e-10
";
    let clock = RinexClock::parse(text).expect("parse clock file with mixed records");
    assert_eq!(clock.series.len(), 2);
    assert_eq!(clock.series["G01"].len(), 1);
    assert_eq!(clock.series["G02"].len(), 1);

    assert_eq!(clock.skipped_records.len(), 4);
    assert_eq!(
        clock.skipped_records,
        vec![
            RinexClockSkip {
                line: 5,
                record_type: "AR".to_string(),
            },
            RinexClockSkip {
                line: 7,
                record_type: "CR".to_string(),
            },
            RinexClockSkip {
                line: 8,
                record_type: "DR".to_string(),
            },
            RinexClockSkip {
                line: 9,
                record_type: "MS".to_string(),
            },
        ]
    );
}

#[test]
fn blank_and_whitespace_lines_compatibility() {
    let text = concat!(
        "\n",
        " 3.00           C                                       RINEX VERSION / TYPE\n",
        "   \n",
        "                                                        END OF HEADER\n",
        "   \n",
        "AS G01  2026 05 13 00 00  0.000000  1    0.100000000000E-03\n",
        "\n",
        "AS G01  2026 05 13 00 00 30.000000  1    0.200000000000E-03\n",
        "   \n",
    );
    let strict = RinexClock::parse(text).expect("blank lines must parse in strict mode");
    assert_eq!(strict.series["G01"].len(), 2);

    let lossy = RinexClock::parse_lossy(text);
    assert_eq!(lossy.series["G01"].len(), 2);
    assert!(lossy.diagnostics.is_empty());
}

#[test]
fn official_300_ar_areq_six_value_parent_only_skip_followed_by_as() {
    let text = " 3.00           CLOCK DATA          GPS                 RINEX VERSION / TYPE
 2    AS    AR                                          # / TYPES OF DATA
                                                        END OF HEADER
AR AREQ 1994 07 14 20 59  0.000000  6   -0.123456789012E+00 -0.123456789012E+01
-0.123456789012E+02 -0.123456789012E+03 -0.123456789012E+04 -0.123456789012E+05
AS G16  1994 07 14 20 59  0.000000  2    -.123456789012E+00  -.123456789012E-01
";
    let clock = RinexClock::parse(text).expect("parse official 3.00 mixed lines");
    assert_eq!(
        clock.skipped_records,
        vec![RinexClockSkip {
            line: 4,
            record_type: "AR".to_string(),
        }]
    );
    assert_eq!(clock.series.len(), 1);
    let point = &clock.series["G16"][0];
    assert_eq!(point.bias_s, -0.123456789012);
    assert_eq!(point.additional_values, vec![-0.0123456789012]);
}

#[test]
fn as_records_round_trip_counts_1_2_4_6_retaining_all_values() {
    let text = " 3.00           C                                       RINEX VERSION / TYPE
    GPS                                                         TIME SYSTEM ID
                                                                END OF HEADER
AS G01  2026 05 13 00 00  0.000000  1   -0.123456789012E+00
AS G02  2026 05 13 00 00  0.000000  2   -0.123456789012E+00 -0.123456789012E+01
AS G03  2026 05 13 00 00  0.000000  4   -0.123456789012E+00 -0.123456789012E+01
-0.123456789012E+02 -0.123456789012E+03
AS G04  2026 05 13 00 00  0.000000  6   -0.123456789012E+00 -0.123456789012E+01
-0.123456789012E+02 -0.123456789012E+03 -0.123456789012E+04 -0.123456789012E+05
";
    let clock = RinexClock::parse(text).expect("parse records of counts 1, 2, 4, 6");
    assert_eq!(clock.series["G01"][0].additional_values.len(), 0);
    assert_eq!(clock.series["G02"][0].additional_values.len(), 1);
    assert_eq!(clock.series["G03"][0].additional_values.len(), 3);
    assert_eq!(clock.series["G04"][0].additional_values.len(), 5);

    let serialized = clock
        .to_rinex_string()
        .expect("serialize clock with continuations");
    let reparsed = RinexClock::parse(&serialized).expect("re-parse serialized clock");
    assert_eq!(reparsed, clock);

    assert_eq!(
        reparsed.series["G04"][0].additional_values,
        vec![
            -1.23456789012,
            -12.3456789012,
            -123.456789012,
            -1234.56789012,
            -12345.6789012,
        ]
    );
}

#[test]
fn missing_continuation_followed_by_valid_as_record() {
    let text = " 3.00           C                                       RINEX VERSION / TYPE
                                                                END OF HEADER
AS G01  2026 05 13 00 00  0.000000  4   -0.123456789012E+00 -0.123456789012E+01
AS G02  2026 05 13 00 00  0.000000  1   -0.123456789012E+00
";
    let err = RinexClock::parse(text).expect_err("missing continuation must error in strict mode");
    assert_eq!(
        err,
        RinexClockError::MissingContinuation {
            line: 3,
            record_type: "AS".to_string(),
        }
    );

    let lossy = RinexClock::parse_lossy(text);
    assert!(!lossy.series.contains_key("G01"));
    assert!(lossy.series.contains_key("G02"));
    assert_eq!(lossy.series["G02"].len(), 1);
    assert_eq!(lossy.diagnostics.len(), 1);
    assert_eq!(lossy.diagnostics[0].line, 3);
    assert_eq!(lossy.diagnostics[0].error, err);
}

#[test]
fn malformed_continuation_followed_by_valid_as_record() {
    let text = " 3.00           C                                       RINEX VERSION / TYPE
                                                                END OF HEADER
AS G01  2026 05 13 00 00  0.000000  4   -0.123456789012E+00 -0.123456789012E+01
not-a-number not-a-number
AS G02  2026 05 13 00 00  0.000000  1   -0.123456789012E+00
";
    let err =
        RinexClock::parse(text).expect_err("malformed continuation must error in strict mode");
    assert_eq!(
        err,
        RinexClockError::MalformedContinuation {
            line: 4,
            reason: "invalid numeric field",
            record: "not-a-number not-a-number".to_string(),
        }
    );

    let lossy = RinexClock::parse_lossy(text);
    assert!(!lossy.series.contains_key("G01"));
    assert!(lossy.series.contains_key("G02"));
    assert_eq!(lossy.series["G02"].len(), 1);
    assert_eq!(lossy.diagnostics.len(), 1);
    assert_eq!(lossy.diagnostics[0].line, 4);
    assert_eq!(lossy.diagnostics[0].error, err);
}

#[test]
fn unknown_standalone_numeric_line_is_rejected() {
    let text = " 3.00           C                                       RINEX VERSION / TYPE
                                                                END OF HEADER
 -0.123456789012E+02 -0.123456789012E+03
";
    let err = RinexClock::parse(text).expect_err("unknown standalone numeric line must error");
    assert_eq!(
        err,
        RinexClockError::BadField {
            line: 3,
            field: "record_type",
            value: "-0.123456789012E+02".to_string(),
        }
    );
    let lossy = RinexClock::parse_lossy(text);
    assert!(lossy.series.is_empty());
    assert_eq!(lossy.diagnostics.len(), 1);
    assert_eq!(lossy.diagnostics[0].line, 3);
    assert_eq!(lossy.diagnostics[0].error, err);
}

#[test]
fn fixed_column_adjacent_full_width_values() {
    let text = " 3.00           C                                       RINEX VERSION / TYPE
                                                                END OF HEADER
AS G01  2026 05 13 00 00  0.000000  2   -0.123456789012E+00 -0.123456789012E+01
";
    let clock = RinexClock::parse(text).expect("parse fixed-column adjacent full-width values");
    let pt = &clock.series["G01"][0];
    assert_eq!(pt.bias_s, -0.123456789012);
    assert_eq!(pt.additional_values, vec![-1.23456789012]);
}

#[test]
fn official_304_parent_and_continuation_layout_compatibility() {
    let text =
        " 3.04                 C                    G                      RINEX VERSION / TYPE
    GPS                                                           TIME SYSTEM ID
                                                                  END OF HEADER
AR AREQ00USA 1994 07 14 20 59  0.000000  6   -0.123456789012E+00  -0.123456789012E+01
   -0.123456789012E+02  -0.123456789012E+03  -0.123456789012E+04  -0.123456789012E+05
AS G16       1994 07 14 20 59  0.000000  6   -0.123456789012E+00  -0.123456789012E+01
   -0.123456789012E+02  -0.123456789012E+03  -0.123456789012E+04  -0.123456789012E+05
";
    let clock = RinexClock::parse(text).expect("parse 3.04 layout");
    assert_eq!(
        clock.skipped_records,
        vec![RinexClockSkip {
            line: 4,
            record_type: "AR".to_string(),
        }]
    );
    assert_eq!(clock.series.len(), 1);
    let pt = &clock.series["G16"][0];
    assert_eq!(pt.bias_s, -0.123456789012);
    assert_eq!(
        pt.additional_values,
        vec![
            -1.23456789012,
            -12.3456789012,
            -123.456789012,
            -1234.56789012,
            -12345.6789012,
        ]
    );
}

#[test]
fn public_validation_and_refusal_for_additional_values() {
    let epoch = civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 0.0).unwrap();

    let too_many = ClockPoint {
        epoch,
        bias_s: 1.0e-4,
        additional_values: vec![1.0e-5, 2.0e-6, 3.0e-7, 4.0e-8, 5.0e-9, 6.0e-10],
    };
    assert_eq!(
        too_many.validate(),
        Err(RinexClockError::InvalidInput {
            field: "additional_values",
            reason: "cannot exceed 5 additional values (maximum count is 6)",
        })
    );

    let nan_val = ClockPoint {
        epoch,
        bias_s: 1.0e-4,
        additional_values: vec![1.0e-5, f64::NAN],
    };
    assert_eq!(
        nan_val.validate(),
        Err(RinexClockError::InvalidInput {
            field: "rate",
            reason: "must be finite",
        })
    );

    let unrep = ClockPoint {
        epoch,
        bias_s: 1.0e-4,
        additional_values: vec![1.23456789012345e-4],
    };
    let unrep_clock = RinexClock::from_instant_series_rows(
        TimeScale::Gpst,
        vec![("G01".to_string(), vec![(epoch, 1.0e-4)])],
    )
    .unwrap();
    let mut bad_series = unrep_clock;
    bad_series.series.get_mut("G01").unwrap()[0] = unrep;
    assert_eq!(
        bad_series.to_rinex_string(),
        Err(RinexClockError::InvalidInput {
            field: "sigma",
            reason:
                "value cannot be represented in Fortran E19.12 format without loss of precision",
        })
    );
}

#[test]
fn parent_declared_count_strict_and_lossy_validation() {
    // 1. AR count 0 followed by valid AS
    let text_0 = " 3.00           C                                       RINEX VERSION / TYPE
                                                            END OF HEADER
AR AREQ 1994 07 14 20 59  0.000000  0   -0.123456789012E+00
AS G01  2026 05 13 00 00  0.000000  1   -0.123456789012E+00
";
    let err_0 = RinexClock::parse(text_0).expect_err("count 0 must error");
    assert_eq!(
        err_0,
        RinexClockError::BadField {
            line: 3,
            field: "count",
            value: "0".to_string(),
        }
    );
    let lossy_0 = RinexClock::parse_lossy(text_0);
    assert_eq!(lossy_0.diagnostics.len(), 1);
    assert_eq!(lossy_0.diagnostics[0].line, 3);
    assert_eq!(lossy_0.diagnostics[0].error, err_0);
    assert_eq!(lossy_0.series.len(), 1);
    assert_eq!(lossy_0.series["G01"].len(), 1);
    assert!(lossy_0.skipped_records.is_empty());

    // 2. AR count 7 followed by valid AS
    let text_7 = " 3.00           C                                       RINEX VERSION / TYPE
                                                            END OF HEADER
AR AREQ 1994 07 14 20 59  0.000000  7   -0.123456789012E+00
AS G01  2026 05 13 00 00  0.000000  1   -0.123456789012E+00
";
    let err_7 = RinexClock::parse(text_7).expect_err("count 7 must error");
    assert_eq!(
        err_7,
        RinexClockError::BadField {
            line: 3,
            field: "count",
            value: "7".to_string(),
        }
    );
    let lossy_7 = RinexClock::parse_lossy(text_7);
    assert_eq!(lossy_7.diagnostics.len(), 1);
    assert_eq!(lossy_7.diagnostics[0].line, 3);
    assert_eq!(lossy_7.diagnostics[0].error, err_7);
    assert_eq!(lossy_7.series["G01"].len(), 1);
    assert!(lossy_7.skipped_records.is_empty());

    // 3. AR huge count followed by valid AS
    let text_huge = " 3.00           C                                       RINEX VERSION / TYPE
                                                            END OF HEADER
AR AREQ 1994 07 14 20 59 0.000000 99999999999999999999999999999999 -0.123456789012E+00
AS G01  2026 05 13 00 00  0.000000  1   -0.123456789012E+00
";
    let err_huge = RinexClock::parse(text_huge).expect_err("huge count must error");
    assert!(matches!(
        err_huge,
        RinexClockError::BadField {
            line: 3,
            field: "count",
            ..
        }
    ));
    let lossy_huge = RinexClock::parse_lossy(text_huge);
    assert_eq!(lossy_huge.diagnostics.len(), 1);
    assert_eq!(lossy_huge.diagnostics[0].line, 3);
    assert_eq!(lossy_huge.series["G01"].len(), 1);
    assert!(lossy_huge.skipped_records.is_empty());

    // 4. AR malformed count followed by valid AS
    let text_malformed =
        " 3.00           C                                       RINEX VERSION / TYPE
                                                            END OF HEADER
AR AREQ 1994 07 14 20 59  0.000000 xyz   -0.123456789012E+00
AS G01  2026 05 13 00 00  0.000000  1   -0.123456789012E+00
";
    let err_malformed = RinexClock::parse(text_malformed).expect_err("malformed count must error");
    assert_eq!(
        err_malformed,
        RinexClockError::BadField {
            line: 3,
            field: "count",
            value: "xyz".to_string(),
        }
    );
    let lossy_malformed = RinexClock::parse_lossy(text_malformed);
    assert_eq!(lossy_malformed.diagnostics.len(), 1);
    assert_eq!(lossy_malformed.diagnostics[0].line, 3);
    assert_eq!(lossy_malformed.series["G01"].len(), 1);
    assert!(lossy_malformed.skipped_records.is_empty());

    // 5. AS count 0, 7, huge, malformed
    let text_as_0 = " 3.00           C                                       RINEX VERSION / TYPE
                                                            END OF HEADER
AS G01  2026 05 13 00 00  0.000000  0   -0.123456789012E+00
AS G02  2026 05 13 00 00  0.000000  1   -0.123456789012E+00
";
    let err_as_0 = RinexClock::parse(text_as_0).expect_err("AS count 0 must error");
    assert_eq!(
        err_as_0,
        RinexClockError::BadField {
            line: 3,
            field: "count",
            value: "0".to_string(),
        }
    );
    let lossy_as_0 = RinexClock::parse_lossy(text_as_0);
    assert_eq!(lossy_as_0.diagnostics.len(), 1);
    assert_eq!(lossy_as_0.diagnostics[0].line, 3);
    assert!(!lossy_as_0.series.contains_key("G01"));
    assert_eq!(lossy_as_0.series["G02"].len(), 1);

    // 6. Missing count in parent
    let text_missing_count =
        " 3.00           C                                       RINEX VERSION / TYPE
                                                            END OF HEADER
AR AREQ 1994 07 14 20 59 0.000000
AS G01  2026 05 13 00 00  0.000000  1   -0.123456789012E+00
";
    let err_missing = RinexClock::parse(text_missing_count).expect_err("missing count must error");
    assert!(matches!(
        err_missing,
        RinexClockError::BadField {
            line: 3,
            field: "count",
            ..
        }
    ));
    let lossy_missing = RinexClock::parse_lossy(text_missing_count);
    assert_eq!(lossy_missing.diagnostics.len(), 1);
    assert_eq!(lossy_missing.diagnostics[0].line, 3);
    assert_eq!(lossy_missing.series["G01"].len(), 1);
}

#[test]
fn valid_unsupported_parent_count_6_followed_by_as() {
    let text = " 3.00           C                                       RINEX VERSION / TYPE
                                                            END OF HEADER
AR AREQ 1994 07 14 20 59  0.000000  6   -0.123456789012E+00 -0.123456789012E+01
-0.123456789012E+02 -0.123456789012E+03 -0.123456789012E+04 -0.123456789012E+05
AS G01  2026 05 13 00 00  0.000000  1   -0.123456789012E+00
";
    let clock = RinexClock::parse(text).expect("valid unsupported count 6 must parse");
    assert_eq!(
        clock.skipped_records,
        vec![RinexClockSkip {
            line: 3,
            record_type: "AR".to_string(),
        }]
    );
    assert_eq!(clock.series.len(), 1);
    assert_eq!(clock.series["G01"].len(), 1);
    assert!(clock.diagnostics.is_empty());

    let lossy = RinexClock::parse_lossy(text);
    assert_eq!(lossy.skipped_records, clock.skipped_records);
    assert_eq!(lossy.series, clock.series);
    assert!(lossy.diagnostics.is_empty());
}

#[test]
fn continuation_internal_hole_rejected_300_and_304() {
    // 3.00 continuation internal hole: count 4 requires values 3 and 4.
    // Value 3 (cols 0..19) is empty, value 4 (cols 20..39) is populated.
    let text_300 = " 3.00           C                                       RINEX VERSION / TYPE
                                                            END OF HEADER
AS G01  2026 05 13 00 00  0.000000  4   -0.123456789012E+00 -0.123456789012E+01
                    -0.123456789012E+03
";
    let err_300 =
        RinexClock::parse(text_300).expect_err("internal hole in 3.00 continuation must error");
    assert_eq!(
        err_300,
        RinexClockError::MalformedContinuation {
            line: 4,
            reason: "missing required continuation value",
            record: "-0.123456789012E+03".to_string(),
        }
    );

    // 3.04 continuation internal hole: starts with 3 spaces.
    // Cols 3..22 is empty, cols 24..43 is populated.
    let text_304 =
        " 3.04                 C                    G                      RINEX VERSION / TYPE
                                                                  END OF HEADER
AS G16       1994 07 14 20 59  0.000000  4   -0.123456789012E+00  -0.123456789012E+01
                        -0.123456789012E+03
";
    let err_304 =
        RinexClock::parse(text_304).expect_err("internal hole in 3.04 continuation must error");
    assert_eq!(
        err_304,
        RinexClockError::MalformedContinuation {
            line: 4,
            reason: "missing required continuation value",
            record: "-0.123456789012E+03".to_string(),
        }
    );
}

#[test]
fn continuation_excess_values_rejected_300_and_304() {
    // 3.00 continuation excess values: count 3 requires 1 value (value 3).
    // Cols 20..39 has extra value (value 4).
    let text_300 = " 3.00           C                                       RINEX VERSION / TYPE
                                                            END OF HEADER
AS G01  2026 05 13 00 00  0.000000  3   -0.123456789012E+00 -0.123456789012E+01
-0.123456789012E+02 -0.123456789012E+03
";
    let err_300 =
        RinexClock::parse(text_300).expect_err("excess values in 3.00 continuation must error");
    assert_eq!(
        err_300,
        RinexClockError::MalformedContinuation {
            line: 4,
            reason: "excess values in continuation line",
            record: "-0.123456789012E+02 -0.123456789012E+03".to_string(),
        }
    );

    // 3.04 continuation excess values: count 3 requires 1 value. Extra value present.
    let text_304 =
        " 3.04                 C                    G                      RINEX VERSION / TYPE
                                                                  END OF HEADER
AS G16       1994 07 14 20 59  0.000000  3   -0.123456789012E+00  -0.123456789012E+01
   -0.123456789012E+02  -0.123456789012E+03
";
    let err_304 =
        RinexClock::parse(text_304).expect_err("excess values in 3.04 continuation must error");
    assert_eq!(
        err_304,
        RinexClockError::MalformedContinuation {
            line: 4,
            reason: "excess values in continuation line",
            record: "-0.123456789012E+02  -0.123456789012E+03".to_string(),
        }
    );

    // Compact continuation fallback excess values:
    let text_compact =
        " 3.00           C                                       RINEX VERSION / TYPE
                                                            END OF HEADER
AS G01 2026 05 13 00 00 0.000000 3 -0.123456789012E+00 -0.123456789012E+01
-0.123456789012E+02 -0.123456789012E+03
";
    let err_compact =
        RinexClock::parse(text_compact).expect_err("compact excess continuation values must error");
    assert_eq!(
        err_compact,
        RinexClockError::MalformedContinuation {
            line: 4,
            reason: "excess values in continuation line",
            record: "-0.123456789012E+02 -0.123456789012E+03".to_string(),
        }
    );
}

#[test]
fn parent_count_1_with_sigma_and_compact_excess_rejected() {
    // 3.00 fixed parent count 1 with populated sigma field
    let text_300 = " 3.00           C                                       RINEX VERSION / TYPE
                                                            END OF HEADER
AS G01  2026 05 13 00 00  0.000000  1   -0.123456789012E+00 -0.123456789012E+01
";
    let err_300 = RinexClock::parse(text_300).expect_err("3.00 count 1 with sigma must error");
    assert_eq!(
        err_300,
        RinexClockError::BadField {
            line: 3,
            field: "sigma",
            value: "-0.123456789012E+01".to_string(),
        }
    );

    // 3.04 fixed parent count 1 with populated sigma field
    let text_304 =
        " 3.04                 C                    G                      RINEX VERSION / TYPE
                                                                  END OF HEADER
AS G16       1994 07 14 20 59  0.000000  1   -0.123456789012E+00  -0.123456789012E+01
";
    let err_304 = RinexClock::parse(text_304).expect_err("3.04 count 1 with sigma must error");
    assert_eq!(
        err_304,
        RinexClockError::BadField {
            line: 3,
            field: "sigma",
            value: "-0.123456789012E+01".to_string(),
        }
    );

    // Compact parent count 1 with extra value
    let text_compact_1 =
        " 3.00           C                                       RINEX VERSION / TYPE
                                                            END OF HEADER
AS G01 2026 05 13 00 00 0.000000 1 -0.123456789012E+00 -0.123456789012E+01
";
    let err_compact_1 =
        RinexClock::parse(text_compact_1).expect_err("compact count 1 with extra value must error");
    assert_eq!(
        err_compact_1,
        RinexClockError::MalformedAsRecord {
            line: 3,
            reason: "excess values in parent record",
            record: "AS G01 2026 05 13 00 00 0.000000 1 -0.123456789012E+00 -0.123456789012E+01"
                .to_string(),
        }
    );

    // Compact parent count 2 with extra value
    let text_compact_2 =
        " 3.00           C                                       RINEX VERSION / TYPE
                                                            END OF HEADER
AS G01 2026 05 13 00 00 0.000000 2 -0.123456789012E+00 -0.123456789012E+01 -0.123456789012E+02
";
    let err_compact_2 =
        RinexClock::parse(text_compact_2).expect_err("compact count 2 with extra value must error");
    assert_eq!(
        err_compact_2,
        RinexClockError::MalformedAsRecord {
            line: 3,
            reason: "excess values in parent record",
            record: "AS G01 2026 05 13 00 00 0.000000 2 -0.123456789012E+00 -0.123456789012E+01 -0.123456789012E+02".to_string(),
        }
    );
}

#[test]
fn as_records_round_trip_counts_3_and_5() {
    let text = " 3.00           C                                       RINEX VERSION / TYPE
    GPS                                                         TIME SYSTEM ID
                                                                END OF HEADER
AS G03  2026 05 13 00 00  0.000000  3   -0.123456789012E+00 -0.123456789012E+01
-0.123456789012E+02
AS G05  2026 05 13 00 00  0.000000  5   -0.123456789012E+00 -0.123456789012E+01
-0.123456789012E+02 -0.123456789012E+03 -0.123456789012E+04
";
    let clock = RinexClock::parse(text).expect("parse records of counts 3 and 5");
    assert_eq!(clock.series["G03"][0].additional_values.len(), 2);
    assert_eq!(clock.series["G05"][0].additional_values.len(), 4);
    assert_eq!(
        clock.series["G03"][0].additional_values,
        vec![-1.23456789012, -12.3456789012]
    );
    assert_eq!(
        clock.series["G05"][0].additional_values,
        vec![
            -1.23456789012,
            -12.3456789012,
            -123.456789012,
            -1234.56789012,
        ]
    );

    let serialized = clock
        .to_rinex_string()
        .expect("serialize clock with counts 3 and 5");
    let reparsed = RinexClock::parse(&serialized).expect("re-parse serialized clock");
    assert_eq!(reparsed, clock);
}

#[test]
fn lossy_parse_preserves_tab_delimited_compact_as_after_malformed() {
    let text = " 3.00           C                                       RINEX VERSION / TYPE
                                                            END OF HEADER
AR BAD RECORD LINE
AS\tG01\t2026\t05\t13\t00\t00\t0.000000\t1\t-0.123456789012E+00
";
    let lossy = RinexClock::parse_lossy(text);
    assert_eq!(lossy.diagnostics.len(), 1);
    assert_eq!(lossy.diagnostics[0].line, 3);
    assert_eq!(lossy.series.len(), 1);
    assert_eq!(lossy.series["G01"].len(), 1);
    assert_eq!(lossy.series["G01"][0].bias_s, -0.123456789012);
}

#[test]
fn positive_13digit_3digit_exponent_and_negative_zero_serialize_and_roundtrip() {
    let text = concat!(
        " 3.00           C                                       RINEX VERSION / TYPE\n",
        "    GPS                                                         TIME SYSTEM ID\n",
        "                                                                END OF HEADER\n",
        "AS G01  2026 05 13 00 00  0.000000  4   1.234567890123E+100  2.761547232975E-04\n",
        "-0.000000000000E+00 1.234567890123E-100\n",
    );
    let clock = RinexClock::parse(text).expect("parse positive 13-digit clock with 3-digit exp");
    let pt = &clock.series["G01"][0];
    assert_eq!(pt.bias_s, 1.234567890123e100);
    assert_eq!(pt.bias_s.to_bits(), (1.234567890123e100_f64).to_bits());
    assert_eq!(pt.additional_values[0], 2.761547232975e-4);
    assert_eq!(
        pt.additional_values[0].to_bits(),
        (2.761547232975e-4_f64).to_bits()
    );
    assert_eq!(pt.additional_values[1], -0.0);
    assert_eq!(pt.additional_values[1].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(pt.additional_values[2], 1.234567890123e-100);
    assert_eq!(
        pt.additional_values[2].to_bits(),
        (1.234567890123e-100_f64).to_bits()
    );

    let serialized = clock
        .to_rinex_string()
        .expect("serialize clock with 13-digit numbers");
    let reparsed = RinexClock::parse(&serialized).expect("re-parse serialized text");
    assert_eq!(reparsed, clock);

    let reparsed_pt = &reparsed.series["G01"][0];
    assert_eq!(reparsed_pt.bias_s.to_bits(), pt.bias_s.to_bits());
    assert_eq!(
        reparsed_pt.additional_values[0].to_bits(),
        pt.additional_values[0].to_bits()
    );
    assert_eq!(
        reparsed_pt.additional_values[1].to_bits(),
        pt.additional_values[1].to_bits()
    );
    assert_eq!(
        reparsed_pt.additional_values[2].to_bits(),
        pt.additional_values[2].to_bits()
    );

    let as_line = serialized
        .lines()
        .find(|l| l.starts_with("AS G01"))
        .expect("serialized AS line");
    assert_eq!(as_line.len(), 79);
    assert_eq!(&as_line[40..59], "1.234567890123E+100");
    assert_eq!(&as_line[59..60], " ");
    assert_eq!(&as_line[60..79], " 2.761547232975E-04");

    let cont_line = serialized
        .lines()
        .find(|l| l.starts_with("-0.000000000000E+00"))
        .expect("serialized continuation line");
    assert_eq!(cont_line.len(), 39);
    assert_eq!(&cont_line[0..19], "-0.000000000000E+00");
    assert_eq!(&cont_line[19..20], " ");
    assert_eq!(&cont_line[20..39], "1.234567890123E-100");
}

#[test]
fn negative_13digit_3digit_exponent_and_signed_zero_serialize_and_roundtrip() {
    let text = concat!(
        " 3.00           C                                       RINEX VERSION / TYPE\n",
        "    GPS                                                         TIME SYSTEM ID\n",
        "                                                                END OF HEADER\n",
        "AS G01  2026 05 13 00 00  0.000000  4   -1.234567890123E100   2.761547232975E-04\n",
        "-0.000000000000E+00 -.1234567890123E-99\n",
        "AS G02  2026 05 13 00 00  0.000000  6   -.1234567890123E-99 -1.234567890123E100\n",
        "-0.000000000000E+00  2.761547232975E-04 1.234567890123E+100 1.234567890123E-100\n",
    );
    let clock = RinexClock::parse(text).expect("parse negative 13-digit clock with 3-digit exp");
    let pt1 = &clock.series["G01"][0];
    assert_eq!(pt1.bias_s, -1.234567890123e100);
    assert_eq!(pt1.bias_s.to_bits(), (-1.234567890123e100_f64).to_bits());
    assert_eq!(pt1.additional_values[0], 2.761547232975e-4);
    assert_eq!(
        pt1.additional_values[0].to_bits(),
        (2.761547232975e-4_f64).to_bits()
    );
    assert_eq!(pt1.additional_values[1], -0.0);
    assert_eq!(pt1.additional_values[1].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(pt1.additional_values[2], -1.234567890123e-100);
    assert_eq!(
        pt1.additional_values[2].to_bits(),
        (-1.234567890123e-100_f64).to_bits()
    );

    let pt2 = &clock.series["G02"][0];
    assert_eq!(pt2.bias_s, -1.234567890123e-100);
    assert_eq!(pt2.bias_s.to_bits(), (-1.234567890123e-100_f64).to_bits());
    assert_eq!(pt2.additional_values[0], -1.234567890123e100);
    assert_eq!(
        pt2.additional_values[0].to_bits(),
        (-1.234567890123e100_f64).to_bits()
    );
    assert_eq!(pt2.additional_values[1].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(
        pt2.additional_values[2].to_bits(),
        (2.761547232975e-4_f64).to_bits()
    );
    assert_eq!(
        pt2.additional_values[3].to_bits(),
        (1.234567890123e100_f64).to_bits()
    );
    assert_eq!(
        pt2.additional_values[4].to_bits(),
        (1.234567890123e-100_f64).to_bits()
    );

    let serialized = clock
        .to_rinex_string()
        .expect("serialize clock with fallback candidates");
    let reparsed = RinexClock::parse(&serialized).expect("re-parse serialized text");
    assert_eq!(reparsed, clock);

    let reparsed_pt1 = &reparsed.series["G01"][0];
    assert_eq!(reparsed_pt1.bias_s.to_bits(), pt1.bias_s.to_bits());
    assert_eq!(
        reparsed_pt1.additional_values[0].to_bits(),
        pt1.additional_values[0].to_bits()
    );
    assert_eq!(
        reparsed_pt1.additional_values[1].to_bits(),
        pt1.additional_values[1].to_bits()
    );
    assert_eq!(
        reparsed_pt1.additional_values[2].to_bits(),
        pt1.additional_values[2].to_bits()
    );

    let reparsed_pt2 = &reparsed.series["G02"][0];
    assert_eq!(reparsed_pt2.bias_s.to_bits(), pt2.bias_s.to_bits());
    assert_eq!(
        reparsed_pt2.additional_values[0].to_bits(),
        pt2.additional_values[0].to_bits()
    );
    assert_eq!(
        reparsed_pt2.additional_values[1].to_bits(),
        pt2.additional_values[1].to_bits()
    );
    assert_eq!(
        reparsed_pt2.additional_values[2].to_bits(),
        pt2.additional_values[2].to_bits()
    );
    assert_eq!(
        reparsed_pt2.additional_values[3].to_bits(),
        pt2.additional_values[3].to_bits()
    );
    assert_eq!(
        reparsed_pt2.additional_values[4].to_bits(),
        pt2.additional_values[4].to_bits()
    );

    let as_line1 = serialized
        .lines()
        .find(|l| l.starts_with("AS G01"))
        .expect("serialized AS G01 line");
    assert_eq!(as_line1.len(), 79);
    assert_eq!(as_line1[40..59].len(), 19);
    assert!(as_line1[40..59].contains('.'));
    assert!(as_line1[40..59].contains('E'));
    assert_eq!(
        as_line1[40..59].trim().parse::<f64>().unwrap().to_bits(),
        pt1.bias_s.to_bits()
    );
    assert_eq!(&as_line1[59..60], " ");
    assert_eq!(&as_line1[60..79], " 2.761547232975E-04");

    let cont_line1 = serialized
        .lines()
        .find(|l| l.starts_with("-0.000000000000E+00 -"))
        .expect("serialized G01 continuation line");
    assert_eq!(cont_line1.len(), 39);
    assert_eq!(&cont_line1[0..19], "-0.000000000000E+00");
    assert_eq!(&cont_line1[19..20], " ");
    assert_eq!(cont_line1[20..39].len(), 19);
    assert!(cont_line1[20..39].contains('.'));
    assert!(cont_line1[20..39].contains('E'));
    assert_eq!(
        cont_line1[20..39].trim().parse::<f64>().unwrap().to_bits(),
        pt1.additional_values[2].to_bits()
    );

    let as_line2 = serialized
        .lines()
        .find(|l| l.starts_with("AS G02"))
        .expect("serialized AS G02 line");
    assert_eq!(as_line2.len(), 79);
    assert_eq!(as_line2[40..59].len(), 19);
    assert!(as_line2[40..59].contains('.'));
    assert!(as_line2[40..59].contains('E'));
    assert_eq!(
        as_line2[40..59].trim().parse::<f64>().unwrap().to_bits(),
        pt2.bias_s.to_bits()
    );
    assert_eq!(&as_line2[59..60], " ");
    assert_eq!(as_line2[60..79].len(), 19);
    assert!(as_line2[60..79].contains('.'));
    assert!(as_line2[60..79].contains('E'));
    assert_eq!(
        as_line2[60..79].trim().parse::<f64>().unwrap().to_bits(),
        pt2.additional_values[0].to_bits()
    );

    let cont_line2 = serialized
        .lines()
        .find(|l| l.starts_with("-0.000000000000E+00  2.761547232975E-04"))
        .expect("serialized G02 continuation line");
    assert_eq!(cont_line2.len(), 79);
    assert_eq!(&cont_line2[0..19], "-0.000000000000E+00");
    assert_eq!(&cont_line2[19..20], " ");
    assert_eq!(&cont_line2[20..39], " 2.761547232975E-04");
    assert_eq!(&cont_line2[39..40], " ");
    assert_eq!(&cont_line2[40..59], "1.234567890123E+100");
    assert_eq!(&cont_line2[59..60], " ");
    assert_eq!(&cont_line2[60..79], "1.234567890123E-100");
}

#[test]
fn negative_14digit_shifted_exponent_serialize_and_roundtrip() {
    let epoch = civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 0.0).unwrap();
    let val_14d = -1.2345678901234e100;
    let point = ClockPoint {
        epoch,
        bias_s: val_14d,
        additional_values: vec![val_14d, 2.761547232975e-4, -0.0],
    };
    let clock = RinexClock::from_instant_series_rows(
        TimeScale::Gpst,
        vec![("G01".to_string(), vec![(epoch, val_14d)])],
    )
    .expect("valid instant series rows");
    let mut clock_with_additional = clock;
    clock_with_additional.series.get_mut("G01").unwrap()[0] = point;

    let serialized = clock_with_additional
        .to_rinex_string()
        .expect("serialize clock with 14-digit shifted exponent candidate");

    let reparsed = RinexClock::parse(&serialized).expect("re-parse serialized text");
    assert_eq!(reparsed, clock_with_additional);

    let reparsed_pt = &reparsed.series["G01"][0];
    assert_eq!(reparsed_pt.bias_s.to_bits(), val_14d.to_bits());
    assert_eq!(
        reparsed_pt.additional_values[0].to_bits(),
        val_14d.to_bits()
    );
    assert_eq!(
        reparsed_pt.additional_values[1].to_bits(),
        (2.761547232975e-4_f64).to_bits()
    );
    assert_eq!(
        reparsed_pt.additional_values[2].to_bits(),
        (-0.0_f64).to_bits()
    );

    let as_line = serialized
        .lines()
        .find(|l| l.starts_with("AS G01"))
        .expect("serialized AS G01 line");
    assert_eq!(as_line.len(), 79);
    assert_eq!(&as_line[40..59], "-12.345678901234E99");
    assert_eq!(&as_line[59..60], " ");
    assert_eq!(&as_line[60..79], "-12.345678901234E99");

    let cont_line = serialized
        .lines()
        .find(|l| l.starts_with(" 2.761547232975E-04"))
        .expect("serialized continuation line");
    assert_eq!(cont_line.len(), 39);
    assert_eq!(&cont_line[0..19], " 2.761547232975E-04");
    assert_eq!(&cont_line[19..20], " ");
    assert_eq!(&cont_line[20..39], "-0.000000000000E+00");
}

#[test]
fn unrepresentable_precision_exceeding_candidate_budget_refused() {
    let epoch = civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 0.0).unwrap();
    // Candidate-set exhaustion evidence for -1.2345678901234e-100:
    // Exact f64 roundtrip bit equality requires 14 significant digits (12345678901234).
    // In the documented finite grammar, decimal-bearing mantissas with sign ('-')
    // and explicit point ('.') require at least 16 ASCII bytes:
    //   - Standard normalized: -1.2345678901234 (16 B) + E-100 (5 B) = 21 bytes
    //   - Shifted left 1 digit: -.12345678901234 (16 B) + E-99 (4 B) = 20 bytes
    //   - Shifted right: -12.345678901234 (16 B) + E-101 (5 B) = 21 bytes
    // No supported spelling can fit within the 19-column width budget; therefore
    // the value cannot be represented without silent precision loss and is refused.
    let neg_val = -1.2345678901234e-100;
    let clock = RinexClock::from_instant_series_rows(
        TimeScale::Gpst,
        vec![("G01".to_string(), vec![(epoch, neg_val)])],
    )
    .expect("valid instant series rows");
    let err = clock
        .to_rinex_string()
        .expect_err("14-digit negative value with exponent -100 must be refused");
    assert_eq!(
        err,
        RinexClockError::InvalidInput {
            field: "bias",
            reason:
                "value cannot be represented in Fortran E19.12 format without loss of precision",
        }
    );

    let clock_sigma = RinexClock::from_instant_series_rows(
        TimeScale::Gpst,
        vec![("G01".to_string(), vec![(epoch, 1.0e-4)])],
    )
    .expect("valid instant series rows");
    let mut bad_sigma_series = clock_sigma;
    bad_sigma_series.series.get_mut("G01").unwrap()[0].additional_values = vec![neg_val];
    let sigma_err = bad_sigma_series
        .to_rinex_string()
        .expect_err("14-digit negative value with exponent -100 in sigma must be refused");
    assert_eq!(
        sigma_err,
        RinexClockError::InvalidInput {
            field: "sigma",
            reason:
                "value cannot be represented in Fortran E19.12 format without loss of precision",
        }
    );
}
