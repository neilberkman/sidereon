#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use super::epoch::{
    epoch_cmp, instant_to_j2000_seconds, interpolate, nearest_microsecond_civil, EpochSource,
};
use super::numeric::format_e19_12;
use super::record::{read_parent, render_record, EpochContext, SigmaGap, TypedEpoch, TypedRecord};
use super::*;
use crate::astro::time::model::JulianDateSplit;
use crate::constants::{GPS_EPOCH_TO_J2000_S, SECONDS_PER_DAY};
use std::cmp::Ordering;

fn as_record(satellite: &str, bias: &str) -> String {
    format!("AS {satellite} 2020 01 01 00 00 00.000000 1 {bias}")
}

fn gpst_context() -> EpochContext {
    EpochContext {
        scale: Some(TimeScale::Gpst),
        policy: validate::CivilSecondPolicy::Continuous,
    }
}

fn typed_as(epoch: Instant, values: Vec<f64>) -> TypedRecord {
    TypedRecord {
        record_type: ClockRecordType::As,
        name: "G01".to_string(),
        epoch: TypedEpoch::Instant {
            instant: epoch,
            source: super::epoch::EpochSource::Instant,
        },
        values,
    }
}

#[test]
fn parse_rejects_non_finite_as_bias() {
    let err = RinexClock::parse(&as_record("G01", "NaN")).unwrap_err();
    assert_eq!(
        err,
        RinexClockError::BadField {
            line: 1,
            field: "bias",
            value: "NaN".to_string(),
        }
    );
}

#[test]
fn parse_rejects_malformed_as_satellite_token() {
    let err = RinexClock::parse(&as_record("X01", "1.0e-9")).unwrap_err();
    assert_eq!(
        err,
        RinexClockError::BadField {
            line: 1,
            field: "satellite",
            value: "X01".to_string(),
        }
    );
}

#[test]
fn explicit_utc_time_system_preserves_clock_epoch_scale() {
    let text = " 3.00           C                                       RINEX VERSION / TYPE\n\
                UTC                                                     TIME SYSTEM ID\n\
                                                                    END OF HEADER\n\
                AS G05  2017 01 01 00 00  0.000000  1   1.0e-04\n\
                AS G05  2017 01 01 00 00 30.000000  1   2.0e-04\n";
    let clock = RinexClock::parse(text).expect("UTC RINEX clock");

    assert_eq!(clock.time_scale(), Some(TimeScale::Utc));
    assert_eq!(clock.series()["G05"][0].epoch.scale, TimeScale::Utc);
    let interpolated = clock
        .clock_s(
            "G05",
            ClockEpoch {
                year: 2017,
                month: 1,
                day: 1,
                hour: 0,
                minute: 0,
                second: 15.0,
            },
        )
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

    // A product rebuilt from the instant rows carries the same series; it does
    // not carry the source header, so the products themselves differ.
    let rows = clock.instant_series_rows();
    assert_eq!(rows[0].1[0].0.scale, TimeScale::Utc);
    let rebuilt = RinexClock::from_instant_series_rows(TimeScale::Utc, rows)
        .expect("valid manual RINEX clock rows");
    assert_eq!(rebuilt.series(), clock.series());
    assert_eq!(rebuilt.time_scale(), clock.time_scale());
}

#[test]
fn manual_series_rows_reject_non_finite_inputs() {
    assert_eq!(
        RinexClock::from_series_rows(vec![("G05".to_string(), vec![(f64::NAN, 1.0e-4)])])
            .unwrap_err(),
        RinexClockError::InvalidInput {
            field: "gps_seconds",
            reason: "must be finite",
        }
    );
    assert_eq!(
        RinexClock::from_series_rows(vec![(
            "G05".to_string(),
            vec![(1_463_904_000.0, f64::INFINITY)]
        )])
        .unwrap_err(),
        RinexClockError::InvalidInput {
            field: "bias_s",
            reason: "must be finite",
        }
    );
}

#[test]
fn manual_series_rows_reject_unsorted_gps_seconds() {
    assert_eq!(
        RinexClock::from_series_rows(vec![(
            "G05".to_string(),
            vec![(1_463_904_030.0, 1.0e-4), (1_463_904_000.0, 2.0e-4)]
        )])
        .unwrap_err(),
        RinexClockError::InvalidInput {
            field: "gps_seconds",
            reason: "must be strictly increasing",
        }
    );
}

#[test]
fn large_finite_gps_seconds_are_refused_without_panicking() {
    // A finite GPS-seconds value far past any civil year used to reach an
    // `expect` on the split Julian date. It is refused by name instead.
    let huge = 3.487_425_075_154_047e123;
    let refusal = RinexClockError::InvalidInput {
        field: "gps_seconds",
        reason: "outside the civil years 1 through 9999 that a clock epoch can name",
    };
    let clock =
        RinexClock::from_series_rows(vec![("G05".to_string(), vec![(1_463_904_000.0, 1.0e-4)])])
            .expect("valid manual RINEX clock rows");
    for value in [huge, -huge, f64::MAX, f64::MIN, 1.0e12] {
        assert_eq!(
            clock.clock_s_at_gps_seconds("G05", value).unwrap_err(),
            refusal,
            "{value}"
        );
        assert_eq!(
            RinexClock::from_series_rows(vec![("G05".to_string(), vec![(value, 1.0e-4)])])
                .unwrap_err(),
            refusal,
            "{value}"
        );
    }
    // The edges of the civil range still answer.
    assert!(clock.clock_s_at_gps_seconds("G05", 0.0).is_ok());
    assert!(clock.clock_s_at_gps_seconds("G05", -6.0e10).is_ok());
    assert!(clock.clock_s_at_gps_seconds("G05", 2.5e11).is_ok());
}

#[test]
fn manual_instant_rows_reject_non_finite_inputs() {
    let bad_epoch = Instant::from_julian_date(
        TimeScale::Gpst,
        JulianDateSplit {
            jd_whole: f64::NAN,
            fraction: 0.0,
        },
    );
    assert_eq!(
        RinexClock::from_instant_series_rows(
            TimeScale::Gpst,
            vec![("G05".to_string(), vec![(bad_epoch, 1.0e-4)])],
        )
        .unwrap_err(),
        RinexClockError::InvalidInput {
            field: "epoch",
            reason: "must be finite",
        }
    );

    let good_epoch =
        civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 0.0).expect("GPST instant");
    assert_eq!(
        RinexClock::from_instant_series_rows(
            TimeScale::Gpst,
            vec![("G05".to_string(), vec![(good_epoch, f64::NAN)])],
        )
        .unwrap_err(),
        RinexClockError::InvalidInput {
            field: "bias_s",
            reason: "must be finite",
        }
    );
}

#[test]
fn manual_instant_rows_reject_unsorted_epochs() {
    let later =
        civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 30.0).expect("later epoch");
    let earlier =
        civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 0.0).expect("earlier epoch");

    assert_eq!(
        RinexClock::from_instant_series_rows(
            TimeScale::Gpst,
            vec![("G05".to_string(), vec![(later, 1.0e-4), (earlier, 2.0e-4)])],
        )
        .unwrap_err(),
        RinexClockError::InvalidInput {
            field: "epoch",
            reason: "must be strictly increasing",
        }
    );
}

#[test]
fn rinex_clock_queries_reject_non_finite_inputs() {
    let clock =
        RinexClock::from_series_rows(vec![("G05".to_string(), vec![(1_463_904_000.0, 1.0e-4)])])
            .expect("valid manual RINEX clock rows");
    let bad_epoch = Instant::from_julian_date(
        TimeScale::Gpst,
        JulianDateSplit {
            jd_whole: f64::INFINITY,
            fraction: 0.0,
        },
    );
    assert_eq!(
        clock.clock_s_at_instant("G05", bad_epoch).unwrap_err(),
        RinexClockError::InvalidInput {
            field: "epoch",
            reason: "must be finite",
        }
    );
    assert_eq!(
        clock.clock_s_at_gps_seconds("G05", f64::NAN).unwrap_err(),
        RinexClockError::InvalidInput {
            field: "gps_seconds",
            reason: "must be finite",
        }
    );
    assert_eq!(
        clock
            .clock_s(
                "G05",
                ClockEpoch {
                    year: 2026,
                    month: 5,
                    day: 13,
                    hour: 0,
                    minute: 0,
                    second: f64::NAN,
                },
            )
            .unwrap_err(),
        RinexClockError::InvalidInput {
            field: "epoch",
            reason: "invalid civil clock epoch",
        }
    );
}

#[test]
fn interpolation_rejects_non_positive_bracket_span() {
    let day = 2_457_753.5;
    let p0 = Instant::from_julian_date(
        TimeScale::Utc,
        JulianDateSplit::new(day, 1.0).expect("valid split Julian date"),
    );
    let p1 = Instant::from_julian_date(
        TimeScale::Utc,
        JulianDateSplit::new(day + 1.0, 0.0).expect("valid split Julian date"),
    );
    let query = Instant::from_julian_date(
        TimeScale::Utc,
        JulianDateSplit::new(day + 1.0, 0.5 / SECONDS_PER_DAY).expect("valid split Julian date"),
    );
    let records = [
        ClockPoint::new(p0, 1.0e-4, Vec::new()),
        ClockPoint::new(p1, 2.0e-4, Vec::new()),
    ];

    assert_eq!(interpolate(&records, query, EpochSource::Instant), None);
}

#[test]
fn utc_interpolation_across_a_leap_second_uses_elapsed_time() {
    // 2016-12-31 ends with a positive leap second. 23:59:59 -> 23:59:60 is one
    // elapsed second, as is 23:59:60 -> 2017-01-01 00:00:00, although the
    // Julian-date labels of 23:59:59 and 23:59:60 coincide.
    let text = "     3.00           C                                       RINEX VERSION / TYPE\n\
                   UTC                                                      TIME SYSTEM ID\n\
                                                                            END OF HEADER\n\
AS G05  2016 12 31 23 59 59.000000  1    0.100000000000E-03\n\
AS G05  2016 12 31 23 59 60.000000  1    0.200000000000E-03\n\
AS G05  2017 01 01 00 00  0.000000  1    0.400000000000E-03\n";
    let clock = RinexClock::parse(text).expect("UTC leap-second clock");
    assert_eq!(clock.time_scale(), Some(TimeScale::Utc));
    let at = |second: f64, day: u8, year: i32, month: u8, hour: u8, minute: u8| {
        clock
            .clock_s(
                "G05",
                ClockEpoch {
                    year,
                    month,
                    day,
                    hour,
                    minute,
                    second,
                },
            )
            .expect("valid UTC query")
    };
    assert_eq!(at(60.0, 31, 2016, 12, 23, 59), Some(2.0e-4));
    let before_leap = at(59.5, 31, 2016, 12, 23, 59).expect("bracketed by 59 and 60");
    assert!((before_leap - 1.5e-4).abs() < 1.0e-15, "{before_leap}");
    let in_leap = at(60.5, 31, 2016, 12, 23, 59).expect("bracketed by 60 and midnight");
    assert!((in_leap - 3.0e-4).abs() < 1.0e-15, "{in_leap}");
}

#[test]
fn utc_interpolation_over_a_leap_second_counts_the_inserted_second() {
    // 23:59:30 to 00:00:00 across the 2016 leap second is 31 elapsed seconds;
    // 23:59:60.0 is 30 of them in.
    let text = "     3.00           C                                       RINEX VERSION / TYPE\n\
                   UTC                                                      TIME SYSTEM ID\n\
                                                                            END OF HEADER\n\
AS G05  2016 12 31 23 59 30.000000  1    0.000000000000E+00\n\
AS G05  2017 01 01 00 00  0.000000  1    0.310000000000E+02\n";
    let clock = RinexClock::parse(text).expect("UTC clock");
    let value = clock
        .clock_s(
            "G05",
            ClockEpoch {
                year: 2016,
                month: 12,
                day: 31,
                hour: 23,
                minute: 59,
                second: 60.0,
            },
        )
        .expect("valid UTC query")
        .expect("bracketed epoch");
    assert!((value - 30.0).abs() < 1.0e-9, "{value}");
}

#[test]
fn qzsst_rows_are_queryable_on_the_gpst_timeline() {
    let p0 =
        civil_to_clock_instant(TimeScale::Qzsst, 2026, 5, 13, 0, 0, 0.0).expect("QZSST instant");
    let p1 =
        civil_to_clock_instant(TimeScale::Qzsst, 2026, 5, 13, 0, 0, 30.0).expect("QZSST instant");
    let clock = RinexClock::from_instant_series_rows(
        TimeScale::Qzsst,
        vec![("J02".to_string(), vec![(p0, 1.0e-4), (p1, 3.0e-4)])],
    )
    .expect("QZSST clock builds");

    let mid = civil_to_gps_seconds(2026, 5, 13, 0, 0, 15.0).expect("gps seconds");
    let bias = clock
        .clock_s_at_gps_seconds("J02", mid)
        .expect("query succeeds")
        .expect("QZSST row interpolates on the GPST timeline");
    assert!(
        (bias - 2.0e-4).abs() < 1.0e-12,
        "expected midpoint interpolation 2.0e-4, got {bias}"
    );

    let start = civil_to_gps_seconds(2026, 5, 13, 0, 0, 0.0).expect("gps seconds");
    assert_eq!(
        clock
            .clock_s_at_gps_seconds("J02", start)
            .expect("query succeeds"),
        Some(1.0e-4)
    );
}

#[test]
fn qzsst_rows_project_to_the_gps_seconds_they_answer_at() {
    // The GPS-seconds projection and the GPS-seconds query use one timeline:
    // every exported QZSST row is answered at its own GPS seconds.
    let text =
        " 3.04                 C                    J                      RINEX VERSION / TYPE\n\
                   QZS                                                           TIME SYSTEM ID\n\
                                                                                 END OF HEADER\n\
AS J02       2026 05 13 00 00  0.000000  1   -0.232835122007E-05\n\
AS J02       2026 05 13 00 05  0.000000  1   -0.232835122107E-05\n";
    let clock = RinexClock::parse(text).expect("QZS clock");
    assert_eq!(clock.time_scale(), Some(TimeScale::Qzsst));
    let rows = clock.series_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].1.len(), 2);
    assert_eq!(
        rows[0].1[0].0,
        civil_to_gps_seconds(2026, 5, 13, 0, 0, 0.0).expect("gps seconds")
    );
    for &(gps_seconds, bias_s) in &rows[0].1 {
        assert_eq!(
            clock
                .clock_s_at_gps_seconds("J02", gps_seconds)
                .expect("valid query"),
            Some(bias_s)
        );
    }
}

#[test]
fn to_rinex_string_round_trips_through_parse() {
    let text = "     3.00           C                                       RINEX VERSION / TYPE\n\
                GPS                                                         TIME SYSTEM ID\n\
                                                                    END OF HEADER\n\
                AS G05  2026 05 13 00 00  0.000000  1   -2.000000000000e-04\n\
                AS G05  2026 05 13 00 00 30.500000  1   -2.000000600000e-04\n\
                AS G24  2026 05 13 00 01  0.000000  1    5.000000000000e-05\n\
                AS E11  2026 05 13 00 00  0.000000  1    1.234500000000e-09\n";
    let clock = RinexClock::parse(text).expect("parse GPST RINEX clock");
    let serialized = clock.to_rinex_string().expect("serialize RINEX clock");
    assert_eq!(serialized, text, "an unedited product restates its input");
    let reparsed = RinexClock::parse(&serialized).expect("re-parse serialized");
    assert_eq!(reparsed, clock, "serializer must round-trip through parse");
    assert_eq!(
        reparsed
            .to_rinex_string()
            .expect("serialize reparsed clock"),
        serialized
    );
}

#[test]
fn to_rinex_string_round_trips_utc_time_scale() {
    let text = "     3.00           C                                       RINEX VERSION / TYPE\n\
                UTC                                                         TIME SYSTEM ID\n\
                                                                    END OF HEADER\n\
                AS G05  2017 01 01 00 00  0.000000  1    1.000000000000e-04\n\
                AS G05  2017 01 01 00 00 30.000000  1    2.000000000000e-04\n";
    let clock = RinexClock::parse(text).expect("parse UTC RINEX clock");
    assert_eq!(clock.time_scale(), Some(TimeScale::Utc));
    let serialized = clock.to_rinex_string().expect("serialize RINEX clock");
    let reparsed = RinexClock::parse(&serialized).expect("re-parse serialized");
    assert_eq!(reparsed.time_scale(), Some(TimeScale::Utc));
    assert_eq!(reparsed, clock);
}

#[test]
fn to_rinex_string_rejects_unsupported_time_scale() {
    let epoch =
        civil_to_clock_instant(TimeScale::Tcg, 2026, 5, 13, 0, 0, 0.0).expect("TCG instant");
    let clock = RinexClock::from_instant_series_rows(
        TimeScale::Tcg,
        vec![("G05".to_string(), vec![(epoch, 1.0e-4)])],
    )
    .expect("TCG clock builds");

    assert_eq!(
        clock.to_rinex_string(),
        Err(RinexClockError::UnsupportedTimeScale {
            scale: TimeScale::Tcg
        })
    );
}

#[test]
fn to_rinex_string_rejects_unsupported_row_time_scale() {
    let epoch =
        civil_to_clock_instant(TimeScale::Tcg, 2026, 5, 13, 0, 0, 0.0).expect("TCG instant");
    let clock = RinexClock::from_instant_series_rows(
        TimeScale::Gpst,
        vec![("G05".to_string(), vec![(epoch, 1.0e-4)])],
    )
    .expect("mixed-scale clock builds");

    assert_eq!(
        clock.to_rinex_string(),
        Err(RinexClockError::UnsupportedTimeScale {
            scale: TimeScale::Tcg
        })
    );
}

#[test]
fn nanos_repr_epoch_serializes_to_true_civil_time() {
    let jd_epoch =
        civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 30.0).expect("GPST instant");
    let j2000_s = instant_to_j2000_seconds(&jd_epoch).expect("J2000 seconds");
    let nanos = (j2000_s * 1.0e9).round() as i128;
    let nanos_epoch = Instant::from_nanos(TimeScale::Gpst, nanos);

    let nanos_clock = RinexClock::from_instant_series_rows(
        TimeScale::Gpst,
        vec![("G05".to_string(), vec![(nanos_epoch, 1.0e-4)])],
    )
    .expect("nanos clock builds");
    let jd_clock = RinexClock::from_instant_series_rows(
        TimeScale::Gpst,
        vec![("G05".to_string(), vec![(jd_epoch, 1.0e-4)])],
    )
    .expect("jd clock builds");

    let serialized = nanos_clock
        .to_rinex_string()
        .expect("serialize nanos RINEX clock");
    assert!(
        serialized.contains("2026 05 13 00 00 30.000000"),
        "Nanos epoch must serialize to its true civil time, got:\n{serialized}"
    );
    assert_eq!(
        serialized,
        jd_clock
            .to_rinex_string()
            .expect("serialize JD RINEX clock"),
        "Nanos- and Julian-date-repr epochs of the same instant must serialize identically"
    );

    let reparsed = RinexClock::parse(&serialized).expect("re-parse serialized Nanos product");
    assert_eq!(reparsed.series(), jd_clock.series());
}

#[test]
fn to_rinex_string_round_trips_utc_leap_second_epoch() {
    let text = "     3.00           C                                       RINEX VERSION / TYPE\n\
                UTC                                                         TIME SYSTEM ID\n\
                                                                    END OF HEADER\n\
                AS G05  2016 12 31 23 59 60.000000  1    1.000000000000e-04\n\
                AS G05  2016 12 31 23 59 60.500000  1    2.000000000000e-04\n";
    let clock = RinexClock::parse(text).expect("parse UTC leap-second RINEX clock");
    let serialized = clock.to_rinex_string().expect("serialize RINEX clock");
    assert!(serialized.contains("23 59 60.000000"));
    assert!(serialized.contains("23 59 60.500000"));
    let reparsed = RinexClock::parse(&serialized).expect("re-parse serialized leap second");
    assert_eq!(reparsed, clock);

    // A product built from the parsed samples writes the leap-second label too.
    let rebuilt = RinexClock::from_clock_points(
        TimeScale::Utc,
        clock
            .series()
            .iter()
            .map(|(sat, points)| (sat.clone(), points.clone()))
            .collect(),
    )
    .expect("rebuild UTC samples");
    let written = rebuilt.to_rinex_string().expect("write rebuilt samples");
    assert!(written.contains("2016 12 31 23 59 60.000000"), "{written}");
    assert!(written.contains("2016 12 31 23 59 60.500000"), "{written}");
    assert_eq!(
        RinexClock::parse(&written)
            .expect("reparse rebuilt")
            .series(),
        clock.series()
    );
}

#[test]
fn parse_fixed_column_satellite_with_internal_space() {
    let text = "AS G  1 2026 05 13 00 00  0.000000  1   1.000000000000e-04\n";
    let clock = RinexClock::parse(text).expect("parse satellite with internal space");
    assert!(clock.series().contains_key("G01"));
    assert_eq!(clock.series()["G01"].len(), 1);
}

#[test]
fn parse_fixed_column_abutting_fields() {
    let text = "AS G01  2026 05 13 00 00  0.000000  2    2.761547232975e-04 4.197517456140e-11\n";
    let clock = RinexClock::parse(text).expect("parse abutting bias and sigma");
    assert!(clock.series().contains_key("G01"));
    let point = &clock.series()["G01"][0];
    assert_eq!(point.bias_s.to_bits(), (2.761547232975e-4_f64).to_bits());
}

#[test]
fn strict_parse_rejects_unrecognized_record_type() {
    let text = "XX G01  2026 05 13 00 00  0.000000  1   1.0e-04\n";
    let err = RinexClock::parse(text).expect_err("strict parse must reject unknown record type");
    assert_eq!(
        err,
        RinexClockError::BadField {
            line: 1,
            field: "record_type",
            value: "XX".to_string(),
        }
    );
    let lossy = RinexClock::parse_lossy(text);
    assert!(lossy.series().is_empty());
    assert_eq!(lossy.record_count(), 0);
    assert_eq!(lossy.diagnostics().len(), 1);
    assert_eq!(lossy.to_rinex_string().expect("restate"), text);
}

#[test]
fn render_record_formats_fixed_columns_single_digit_seconds() {
    let instant = civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 5.123456).unwrap();
    let lines = render_record(
        &typed_as(instant, vec![2.761547232975e-4]),
        ClockLayout::V300,
        Some(TimeScale::Gpst),
        SigmaGap::One,
    )
    .unwrap();
    assert_eq!(lines.len(), 1);
    assert!(
        lines[0].starts_with("AS G01  2026 05 13 00 00  5.123456  1"),
        "expected fixed-column layout without space split in seconds: {}",
        lines[0]
    );
    let read = read_parent(1, &lines[0], Some(ClockLayout::V300), &gpst_context())
        .expect("rendered record reads back");
    assert_eq!(read.reading, ClockRecordReading::Columns(ClockLayout::V300));
    assert_eq!(read.values, vec![2.761547232975e-4]);
}

#[test]
fn render_record_formats_exact_19_column_fields_and_continuation() {
    let instant = civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 0.0).unwrap();
    let lines = render_record(
        &typed_as(
            instant,
            vec![
                1.234567890123e100,
                2.761547232975e-4,
                -0.0,
                1.234567890123e-100,
            ],
        ),
        ClockLayout::V300,
        Some(TimeScale::Gpst),
        SigmaGap::One,
    )
    .unwrap();
    assert_eq!(lines.len(), 2);

    let parent = &lines[0];
    assert_eq!(parent.len(), 79);
    assert_eq!(&parent[40..59], "1.234567890123E+100");
    assert_eq!(&parent[59..60], " ");
    assert_eq!(&parent[60..79], " 2.761547232975E-04");

    let cont = &lines[1];
    assert_eq!(cont.len(), 39);
    assert_eq!(&cont[0..19], "-0.000000000000E+00");
    assert_eq!(&cont[19..20], " ");
    assert_eq!(&cont[20..39], "1.234567890123E-100");
}

#[test]
fn render_record_writes_the_304_columns() {
    // Table A16 (3.04): A2,1X,A9,1X,I4,1X,4(I2,1X),F9.6,1X,I2,3X,E19.12, then the
    // sigma; the continuation is 3X, then E19.12,2X three times, then E19.12.
    // The sigma follows one blank (columns 66-84), where the IGS combination
    // example places it and RTKLIB reads it, unless the product's own lines
    // use the two blanks of Table A16.
    let instant = civil_to_clock_instant(TimeScale::Gpst, 1994, 7, 14, 20, 59, 0.0).unwrap();
    let values = vec![
        -0.123456789012,
        -1.23456789012,
        -12.3456789012,
        -123.456789012,
        -1234.56789012,
        -12345.6789012,
    ];
    let continuation =
        "   -0.123456789012E+02  -0.123456789012E+03  -0.123456789012E+04  -0.123456789012E+05";
    let one = render_record(
        &typed_as(instant, values.clone()),
        ClockLayout::V304,
        Some(TimeScale::Gpst),
        SigmaGap::One,
    )
    .unwrap();
    assert_eq!(
        one,
        vec![
            "AS G01       1994 07 14 20 59  0.000000  6   -0.123456789012E+00 -0.123456789012E+01"
                .to_string(),
            continuation.to_string(),
        ]
    );
    assert_eq!(one[0].len(), 84);
    assert_eq!(&one[0][65..84], "-0.123456789012E+01");
    assert_eq!(one[1].len(), 85);
    let two = render_record(
        &typed_as(instant, values),
        ClockLayout::V304,
        Some(TimeScale::Gpst),
        SigmaGap::Two,
    )
    .unwrap();
    assert_eq!(
        two[0],
        "AS G01       1994 07 14 20 59  0.000000  6   -0.123456789012E+00  -0.123456789012E+01"
    );
    assert_eq!(two[0].len(), 85);
    // Both spacings read back to the same values.
    for line in [&one[0], &two[0]] {
        let read = read_parent(1, line, Some(ClockLayout::V304), &gpst_context())
            .expect("rendered 3.04 record reads back");
        assert_eq!(read.reading, ClockRecordReading::Columns(ClockLayout::V304));
        assert_eq!(read.values, vec![-0.123456789012, -1.23456789012]);
    }
}

#[test]
fn render_record_refuses_an_instant_the_epoch_field_cannot_restate() {
    // 0.5 microseconds past the minute: the seconds field states microseconds,
    // so the instant is refused rather than rounded.
    let nanos = Instant::from_nanos(TimeScale::Gpst, 500);
    assert_eq!(
        render_record(
            &typed_as(nanos, vec![1.0e-4]),
            ClockLayout::V300,
            Some(TimeScale::Gpst),
            SigmaGap::One,
        ),
        Err(RinexClockError::InvalidInput {
            field: "epoch",
            reason: "the epoch field cannot restate this instant without rounding it",
        })
    );
    let whole = Instant::from_nanos(TimeScale::Gpst, 1_000);
    let lines = render_record(
        &typed_as(whole, vec![1.0e-4]),
        ClockLayout::V300,
        Some(TimeScale::Gpst),
        SigmaGap::One,
    )
    .expect("a whole microsecond is stated");
    assert!(
        lines[0].contains("2000 01 01 12 00  0.000001"),
        "{}",
        lines[0]
    );

    let civil = super::epoch::clock_epoch_to_civil(
        ClockEpoch {
            year: 2026,
            month: 5,
            day: 13,
            hour: 0,
            minute: 0,
            second: 5.1234567,
        },
        validate::CivilSecondPolicy::Continuous,
    )
    .expect("valid civil epoch");
    assert_eq!(civil.second, 5);
    assert_eq!(civil.microsecond, 123_456);
    assert_eq!(civil.femtosecond, 700_000_000);
    let record = TypedRecord {
        record_type: ClockRecordType::As,
        name: "G01".to_string(),
        epoch: TypedEpoch::Civil {
            civil,
            second_text: None,
        },
        values: vec![1.0e-4],
    };
    assert_eq!(
        render_record(
            &record,
            ClockLayout::V300,
            Some(TimeScale::Gpst),
            SigmaGap::One
        ),
        Err(RinexClockError::InvalidInput {
            field: "epoch",
            reason: "the seconds field states microseconds and this epoch carries finer digits",
        })
    );
    // The same epoch edited from a source line keeps the source text, which
    // fits the seconds field with a blank before it and restates it exactly.
    let edited = TypedRecord {
        epoch: TypedEpoch::Civil {
            civil,
            second_text: Some("5.1234567".to_string()),
        },
        ..record.clone()
    };
    let lines = render_record(
        &edited,
        ClockLayout::V300,
        Some(TimeScale::Gpst),
        SigmaGap::One,
    )
    .expect("source seconds text fits");
    assert!(
        lines[0].starts_with("AS G01  2026 05 13 00 00 5.1234567  1"),
        "{}",
        lines[0]
    );
    let lines = render_record(
        &edited,
        ClockLayout::V304,
        Some(TimeScale::Gpst),
        SigmaGap::One,
    )
    .expect("source seconds text fits");
    assert!(
        lines[0].starts_with("AS G01       2026 05 13 00 00 5.1234567  1"),
        "{}",
        lines[0]
    );
    // A ten-character text would meet the minute in the 3.00 layout.
    let crowded = TypedRecord {
        epoch: TypedEpoch::Civil {
            civil,
            second_text: Some("5.12345670".to_string()),
        },
        ..record
    };
    assert!(render_record(
        &crowded,
        ClockLayout::V300,
        Some(TimeScale::Gpst),
        SigmaGap::One
    )
    .is_err());
}

#[test]
fn sub_microsecond_epochs_are_read_exactly_and_kept_apart() {
    // Two records 0.5 microseconds apart are two samples: the seconds field is
    // read with every digit it states rather than rounded to the microsecond.
    let text = "AS G05  2026 05 13 00 00 59.9999990  1   1.0e-04\n\
                AS G05  2026 05 13 00 00 59.9999995  1   2.0e-04\n";
    let clock = RinexClock::parse(text).expect("sub-microsecond epochs");
    let points = &clock.series()["G05"];
    assert_eq!(points.len(), 2);
    assert_eq!(
        epoch_cmp(&points[0].epoch, &points[1].epoch),
        Ordering::Less
    );
    let civil = clock.records().nth(1).unwrap().civil;
    assert_eq!(
        (civil.second, civil.microsecond, civil.femtosecond),
        (59, 999_999, 500_000_000)
    );
    // Neither epoch can be written in the 3.00 seconds field without rounding
    // or meeting the minute, so an edit of it is refused and nothing changes.
    let mut edited = clock.clone();
    assert!(edited.set_record_values(1, vec![3.0e-4]).is_err());
    assert_eq!(edited, clock);
    assert_eq!(edited.to_rinex_string().unwrap(), text);
}

#[test]
fn format_e19_12_formats_standard_examples_and_rejects_unrepresentable() {
    assert_eq!(
        format_e19_12(-0.123456789012, "bias").unwrap(),
        "-0.123456789012E+00"
    );
    assert_eq!(
        format_e19_12(-1.23456789012, "bias").unwrap(),
        "-0.123456789012E+01"
    );
    assert_eq!(
        format_e19_12(-12.3456789012, "bias").unwrap(),
        "-0.123456789012E+02"
    );
    assert_eq!(format_e19_12(0.0, "bias").unwrap(), " 0.000000000000E+00");
    assert_eq!(format_e19_12(-0.0, "bias").unwrap(), "-0.000000000000E+00");
    assert_eq!(
        format_e19_12(1.0e-4, "bias").unwrap(),
        " 0.100000000000E-03"
    );
    assert_eq!(
        format_e19_12(2.761547232975e-4, "bias").unwrap(),
        " 2.761547232975E-04"
    );
    assert_eq!(
        format_e19_12(1.0e-105, "bias").unwrap(),
        " .100000000000E-104"
    );
    assert_eq!(
        format_e19_12(1.234567890123e100, "bias").unwrap(),
        "1.234567890123E+100"
    );
    assert_eq!(
        format_e19_12(1.234567890123e-100, "bias").unwrap(),
        "1.234567890123E-100"
    );

    assert!(format_e19_12(f64::NAN, "bias").is_err());
    assert!(format_e19_12(f64::INFINITY, "bias").is_err());
    assert!(format_e19_12(f64::NEG_INFINITY, "bias").is_err());

    let fb_neg_e100 = format_e19_12(-1.234567890123e100, "bias").unwrap();
    assert_eq!(fb_neg_e100.len(), 19);
    assert_eq!(
        fb_neg_e100.trim().parse::<f64>().unwrap().to_bits(),
        (-1.234567890123e100_f64).to_bits()
    );

    let fb_neg_em100 = format_e19_12(-1.234567890123e-100, "bias").unwrap();
    assert_eq!(fb_neg_em100.len(), 19);
    assert_eq!(
        fb_neg_em100.trim().parse::<f64>().unwrap().to_bits(),
        (-1.234567890123e-100_f64).to_bits()
    );

    assert_eq!(
        format_e19_12(-1.2345678901234e100, "bias").unwrap(),
        "-12.345678901234E99"
    );

    assert!(format_e19_12(1.23456789012345e-4, "bias").is_err());
    assert!(format_e19_12(-1.2345678901234e-100, "bias").is_err());
}

#[test]
fn validate_clock_point_bounds_and_finite_checks() {
    let epoch = civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 0.0).unwrap();
    let valid = ClockPoint::new(epoch, 1.0e-4, vec![1.0e-5, 2.0e-6, 3.0e-7, 4.0e-8, 5.0e-9]);
    assert!(valid.validate().is_ok());

    let mut too_many = valid.clone();
    too_many.additional_values.push(6.0e-10);
    assert_eq!(
        too_many.validate(),
        Err(RinexClockError::InvalidInput {
            field: "additional_values",
            reason: "cannot exceed 5 additional values (maximum count is 6)",
        })
    );

    let mut non_finite = valid;
    non_finite.additional_values[2] = f64::NAN;
    assert_eq!(
        non_finite.validate(),
        Err(RinexClockError::InvalidInput {
            field: "rate_sigma",
            reason: "must be finite",
        })
    );
}

#[test]
fn line_terminators_and_a_missing_final_newline_are_restated() {
    let crlf =
        "     3.00           C                                       RINEX VERSION / TYPE\r\n\
                                                                            END OF HEADER\r\n\
AS G01  2026 05 13 00 00  0.000000  1    0.100000000000E-03\r\n\
\r\n\
AS G01  2026 05 13 00 00 30.000000  1    0.200000000000E-03";
    let clock = RinexClock::parse(crlf).expect("CRLF clock");
    assert_eq!(clock.series()["G01"].len(), 2);
    assert_eq!(clock.to_rinex_string().expect("restate"), crlf);

    // An inserted record after the unterminated last line is written on its own
    // line with the product's terminator.
    let mut edited = clock.clone();
    let record = ClockRecord::new(
        ClockRecordType::As,
        "G02",
        ClockEpoch {
            year: 2026,
            month: 5,
            day: 13,
            hour: 0,
            minute: 1,
            second: 0.0,
        },
        vec![3.0e-4],
    )
    .expect("record");
    edited.insert_record(2, record).expect("insert at end");
    let written = edited.to_rinex_string().expect("write edited");
    assert!(written.starts_with(crlf));
    assert_eq!(
        &written[crlf.len()..],
        "\r\nAS G02  2026 05 13 00 01  0.000000  1    0.300000000000E-03\r\n"
    );
}

#[test]
fn source_lines_are_available_by_line_number() {
    let text = "AS G01  2026 05 13 00 00  0.000000  1    0.100000000000E-03\r\nXX junk\n";
    let clock = RinexClock::parse_lossy(text);
    assert_eq!(
        clock.source_line(1),
        Some("AS G01  2026 05 13 00 00  0.000000  1    0.100000000000E-03")
    );
    assert_eq!(clock.source_line(2), Some("XX junk"));
    assert_eq!(clock.source_line(0), None);
    assert_eq!(clock.source_line(3), None);
}

#[test]
fn parse_reports_unmodelled_clock_records_while_returning_modelled_satellite_series() {
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
AS G01  2026 05 13 00 00 30.000000  1   1.500000000000e-04
";
    let clock = RinexClock::parse(text).expect("parse clock file with mixed records");
    assert_eq!(clock.series().len(), 2);
    assert_eq!(clock.series()["G01"].len(), 2);
    assert_eq!(clock.series()["G02"].len(), 1);
    assert_eq!(clock.series()["G01"][0].bias_s, 1.0e-4);
    assert_eq!(clock.series()["G01"][1].bias_s, 1.5e-4);
    assert_eq!(clock.series()["G02"][0].bias_s, 3.0e-4);

    assert_eq!(clock.skipped_records().len(), 4);
    assert_eq!(
        clock.skipped_records()[0],
        RinexClockSkip {
            line: 5,
            record_type: "AR".to_string(),
        }
    );
    assert_eq!(
        clock.skipped_records()[1],
        RinexClockSkip {
            line: 7,
            record_type: "CR".to_string(),
        }
    );
    assert_eq!(
        clock.skipped_records()[2],
        RinexClockSkip {
            line: 8,
            record_type: "DR".to_string(),
        }
    );
    assert_eq!(
        clock.skipped_records()[3],
        RinexClockSkip {
            line: 9,
            record_type: "MS".to_string(),
        }
    );

    let lossy = RinexClock::parse_lossy(text);
    assert_eq!(lossy.series().len(), 2);
    assert_eq!(lossy.skipped_records(), clock.skipped_records());
    assert!(lossy.diagnostics().is_empty());
}

#[test]
fn instants_built_from_gps_seconds_are_written_and_read_back_unchanged() {
    // GPS seconds on a 0.1 s grid and with microsecond digits name instants
    // whose split parts differ in the last places from the parser's reading of
    // the microsecond text, yet they are the same GPS second count. They are
    // written, and read back to the same GPS seconds.
    let mut rows = Vec::new();
    for k in 0..1000u32 {
        rows.push(1_463_904_000.0 + f64::from(k) * 30.0 + f64::from(k % 10) * 0.1);
    }
    for k in 0..1000u32 {
        rows.push(1_463_990_400.0 + f64::from(k) * 30.0 + 0.123456);
    }
    let clock = RinexClock::from_series_rows(vec![(
        "G05".to_string(),
        rows.iter().map(|&gps| (gps, 1.0e-4)).collect(),
    )])
    .expect("GPS-second rows");
    let text = clock
        .to_rinex_string()
        .expect("every row is a microsecond epoch");
    assert!(text.contains("2026 05 27 08 00 30.100000"));
    assert!(text.contains("2026 05 28 08 00  0.123456"));
    let reread = RinexClock::parse(&text).expect("reread");
    let reread_rows: Vec<f64> = reread.series_rows()[0]
        .1
        .iter()
        .map(|&(gps, _)| gps)
        .collect();
    assert_eq!(reread_rows, rows);
}

#[test]
fn a_microsecond_product_survives_export_and_rebuild_through_gps_seconds() {
    // A product of microsecond epochs exported with series_rows and rebuilt
    // with from_series_rows is written with the same microsecond texts.
    let mut text = String::from("     3.00           C                   G                   RINEX VERSION / TYPE\n   GPS                                                      TIME SYSTEM ID\n                                                            END OF HEADER\n");
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut expected = Vec::new();
    for minute in 0..60u32 {
        for step in 0..17u32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let micro = state % 1_000_000;
            let second = step * 3;
            let epoch = format!("2026 05 13 00 {minute:02} {second:>2}.{micro:06}");
            text.push_str(&format!("AS G05  {epoch}  1    0.100000000000E-03\n"));
            expected.push(epoch);
        }
    }
    let parsed = RinexClock::parse(&text).expect("microsecond product");
    let rebuilt = RinexClock::from_series_rows(parsed.series_rows()).expect("rebuild");
    let written = rebuilt.to_rinex_string().expect("write rebuilt product");
    for epoch in &expected {
        assert!(written.contains(epoch.as_str()), "{epoch}");
    }
    assert_eq!(
        RinexClock::parse(&written).unwrap().series_rows(),
        parsed.series_rows()
    );
}

#[test]
fn instants_stepped_by_adding_day_fractions_are_refused_or_reported() {
    // 30 s epochs built by adding 30/86400 to the day fraction carry rounding
    // in the last places. Where the sum is not the split any of the crate's
    // conversions builds from the microsecond text, the strict writer refuses
    // it; the lenient policy writes the nearest microsecond and reports each.
    let day = 2_461_173.5;
    let mut fraction = 0.0;
    let mut points = Vec::new();
    let mut expected = Vec::new();
    for k in 0..2880u32 {
        let epoch = Instant::from_julian_date(
            TimeScale::Gpst,
            JulianDateSplit::new(day, fraction).expect("split"),
        );
        let exact = f64::from(k * 30) / SECONDS_PER_DAY;
        if fraction.to_bits() != exact.to_bits() {
            expected.push((k as usize, epoch));
        }
        points.push((epoch, 1.0e-4));
        fraction += 30.0 / SECONDS_PER_DAY;
    }
    assert!(!expected.is_empty());
    let clock =
        RinexClock::from_instant_series_rows(TimeScale::Gpst, vec![("G05".to_string(), points)])
            .expect("stepped epochs");
    assert_eq!(
        clock.to_rinex_string(),
        Err(RinexClockError::InvalidInput {
            field: "epoch",
            reason: "the epoch field cannot restate this instant without rounding it",
        })
    );
    let (text, departures) = clock
        .to_rinex_string_with_policy(ClockWritePolicy::lenient())
        .expect("nearest microsecond allowed");
    assert_eq!(departures.len(), expected.len());
    for (departure, (record, epoch)) in departures.iter().zip(&expected) {
        let seconds = record * 30;
        // The written epoch fields, separated by single blanks: the seconds
        // field carries no leading zero.
        let written = format!(
            "2026 05 13 {:02} {:02} {}.000000",
            seconds / 3600,
            seconds % 3600 / 60,
            seconds % 60
        );
        assert_eq!(
            departure,
            &ClockWriteDeparture::EpochAtNearestMicrosecond {
                record: *record,
                name: "G05".to_string(),
                epoch: Some(*epoch),
                written,
            }
        );
    }
    let reread = RinexClock::parse(&text).expect("reread");
    assert_eq!(reread.series()["G05"].len(), 2880);
}

#[test]
fn one_rounded_epoch_is_refused_strict_and_reported_under_the_policy() {
    let one_tenth_microsecond =
        civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 30.0000001)
            .expect("exact civil epoch");
    let mut stepped = 0.0;
    for _ in 0..5 {
        stepped += 30.0 / SECONDS_PER_DAY;
    }
    let stepped = Instant::from_julian_date(
        TimeScale::Gpst,
        JulianDateSplit::new(2_461_173.5, stepped).expect("split"),
    );
    for (epoch, written) in [
        (one_tenth_microsecond, "2026 05 13 00 00 30.000000"),
        (stepped, "2026 05 13 00 02 30.000000"),
    ] {
        let clock = RinexClock::from_instant_series_rows(
            TimeScale::Gpst,
            vec![("G05".to_string(), vec![(epoch, 1.0e-4)])],
        )
        .expect("row");
        assert_eq!(
            clock.to_rinex_string(),
            Err(RinexClockError::InvalidInput {
                field: "epoch",
                reason: "the epoch field cannot restate this instant without rounding it",
            })
        );
        let policy =
            ClockWritePolicy::strict().with_nearest_microsecond_epochs(ClockWriteLeniency::Allow);
        let (text, departures) = clock
            .to_rinex_string_with_policy(policy)
            .expect("nearest microsecond allowed");
        assert_eq!(
            departures,
            vec![ClockWriteDeparture::EpochAtNearestMicrosecond {
                record: 0,
                name: "G05".to_string(),
                epoch: Some(epoch),
                written: written.to_string(),
            }]
        );
        assert!(text.contains(&format!("AS G05  {written}")), "{text}");
    }
}

#[test]
fn whole_j2000_second_instants_are_written_strict() {
    // Splits built by the whole-J2000-second conversion sit on the noon day
    // boundary; each is the split that conversion builds from the whole second
    // its microsecond text states, including epochs near J2000 where the
    // reader's split does not sum to a whole second as a double.
    let seconds: [i64; 6] = [0, 30, 45_296, 86_399, 832_032_000, 832_032_030];
    let points = seconds
        .iter()
        .map(|&n| {
            let (jd_whole, fraction) =
                crate::astro::time::civil::split_julian_date_from_j2000_seconds(n);
            (
                Instant::from_julian_date(
                    TimeScale::Gpst,
                    JulianDateSplit::new(jd_whole, fraction).expect("split"),
                ),
                1.0e-4,
            )
        })
        .collect();
    let clock =
        RinexClock::from_instant_series_rows(TimeScale::Gpst, vec![("G05".to_string(), points)])
            .expect("whole J2000 seconds");
    let (text, departures) = clock
        .to_rinex_string_with_policy(ClockWritePolicy::default())
        .expect("strict write");
    assert!(departures.is_empty());
    assert_eq!(clock.to_rinex_string().unwrap(), text);
    for written in [
        "AS G05  2000 01 01 12 00  0.000000",
        "AS G05  2000 01 01 12 00 30.000000",
        "AS G05  2000 01 02 00 34 56.000000",
        "AS G05  2000 01 02 11 59 59.000000",
    ] {
        assert!(text.contains(written), "{written}\n{text}");
    }
}

#[test]
fn write_policy_defaults_to_strict() {
    assert_eq!(ClockWritePolicy::default(), ClockWritePolicy::strict());
    assert_eq!(
        ClockWritePolicy::strict().nearest_microsecond_epochs,
        ClockWriteLeniency::Strict
    );
    assert_eq!(
        ClockWritePolicy::lenient(),
        ClockWritePolicy::strict().with_nearest_microsecond_epochs(ClockWriteLeniency::Allow)
    );
    // A product read from text writes the same bytes under any policy, with no
    // departure.
    let text = "AS G05  2026 05 13 00 00 59.9999996  1   1.0e-04\n";
    let clock = RinexClock::parse(text).unwrap();
    assert_eq!(
        clock.to_rinex_string_with_policy(ClockWritePolicy::lenient()),
        Ok((text.to_string(), Vec::new()))
    );
}

#[test]
fn a_sub_microsecond_instant_without_source_text_is_refused() {
    let epoch = civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 59.9999995)
        .expect("exact civil epoch");
    let clock = RinexClock::from_instant_series_rows(
        TimeScale::Gpst,
        vec![("G05".to_string(), vec![(epoch, 1.0e-4)])],
    )
    .expect("row");
    assert_eq!(
        clock.to_rinex_string(),
        Err(RinexClockError::InvalidInput {
            field: "epoch",
            reason: "the epoch field cannot restate this instant without rounding it",
        })
    );
}

#[test]
fn queries_read_the_second_as_stated() {
    // A query second is read with every digit, as a record's seconds field is,
    // so a query at a record's stated epoch lands on that record.
    let text = "AS G05  2026 05 13 00 00 59.9999996  1   1.0e-04\n\
                AS G05  2026 05 13 00 01  0.000000  1   2.0e-04\n";
    let clock = RinexClock::parse(text).expect("clock");
    let at = |second: f64| {
        clock
            .clock_s(
                "G05",
                ClockEpoch {
                    year: 2026,
                    month: 5,
                    day: 13,
                    hour: 0,
                    minute: 0,
                    second,
                },
            )
            .expect("valid query")
    };
    assert_eq!(at(59.9999996), Some(1.0e-4));
    let next_minute = civil_to_gps_seconds(2026, 5, 13, 0, 1, 0.0).unwrap();
    let stated = civil_to_gps_seconds(2026, 5, 13, 0, 0, 59.9999996).unwrap();
    assert!(stated < next_minute, "{stated} {next_minute}");
    assert_eq!(
        civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 59.9999996),
        clock.series()["G05"].first().map(|point| point.epoch)
    );
}

#[test]
fn sigma_spacing_is_taken_from_a_one_value_record_with_a_sigma() {
    // The first 3.04 record carrying a sigma declares one value (the EMR
    // shape) and writes the sigma after two blanks; an edited record follows.
    let text =
        "3.04                 C                    G                      RINEX VERSION / TYPE
   GPS                                                           TIME SYSTEM ID
                                                                 END OF HEADER
AS G01       2026 05 13 00 00  0.000000  1   -0.123456789012E+00  -0.123456789012E-01
AS G02       2026 05 13 00 00  0.000000  1   -0.223456789012E+00
";
    let mut clock = RinexClock::parse(text).expect("3.04 clock");
    clock
        .set_record_values(1, vec![-0.223456789012, 1.0e-11])
        .expect("edit G02");
    assert!(clock.to_rinex_string().unwrap().ends_with(
        "\nAS G02       2026 05 13 00 00  0.000000  2   -0.223456789012E+00   0.100000000000E-10\n"
    ));
}

/// A small xorshift generator so the sequence below is the same on every run.
struct Sequence(u64);

impl Sequence {
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % bound as u64) as usize
    }
}

fn assert_derived_matches_rebuild(clock: &RinexClock, step: &str) {
    let mut rebuilt = clock.clone();
    rebuilt.rebuild();
    assert_eq!(rebuilt.derived, clock.derived, "{step}");
    assert_eq!(rebuilt.notices, clock.notices, "{step}");
    assert_eq!(rebuilt.diagnostics, clock.diagnostics, "{step}");
    assert_eq!(rebuilt.context, clock.context, "{step}");
    let positions: Vec<usize> = clock
        .body
        .iter()
        .enumerate()
        .filter(|(_, entry)| matches!(entry, BodyEntry::Record { .. } | BodyEntry::Typed(_)))
        .map(|(position, _)| position)
        .collect();
    assert_eq!(positions, clock.record_positions, "{step}");
    assert_eq!(clock.order_keys.len(), clock.body.len(), "{step}");
    assert!(
        clock.order_keys.windows(2).all(|pair| pair[0] < pair[1]),
        "{step}"
    );
}

fn run_edit_sequence(text: &str, date: (i32, u8, u8, u8), seed: u64, label: &str) {
    let mut clock = RinexClock::parse_lossy(text);
    assert_derived_matches_rebuild(&clock, label);
    let mut sequence = Sequence(seed);
    let names = ["G01", "G02", "R05", "AREQ", "USNO"];
    let (year, month, day, hour) = date;

    // Edit the last source record of each duplicated name and epoch: for an
    // `AS` record that is the one the series shows, and for a pair of `AR`
    // records it is the second record of a section 4 discontinuity.
    let records: Vec<ClockRecord> = clock.records().collect();
    let shown_duplicates: Vec<usize> = records
        .iter()
        .enumerate()
        .filter(|(index, record)| {
            record.line.is_some()
                && records[..*index].iter().any(|earlier| {
                    earlier.name == record.name
                        && earlier.record_type == record.record_type
                        && earlier.civil == record.civil
                })
                && !records[index + 1..].iter().any(|later| {
                    later.name == record.name
                        && later.record_type == record.record_type
                        && later.civil == record.civil
                })
        })
        .map(|(index, _)| index)
        .collect();
    for index in shown_duplicates {
        let mut values = vec![9.0e-4];
        if let Some(last) = records[index]
            .surplus
            .iter()
            .map(|value| value.position)
            .max()
        {
            values.resize(last + 1, 1.0e-11);
        }
        clock
            .set_record_values(index, values)
            .expect("edit a shown duplicate");
        assert_derived_matches_rebuild(&clock, &format!("{label}: shown duplicate {index}"));
    }

    // Remove every source record that repeats another's epoch.
    loop {
        let records: Vec<ClockRecord> = clock.records().collect();
        let duplicate = records.iter().enumerate().position(|(index, record)| {
            record.line.is_some()
                && records[..index].iter().any(|earlier| {
                    earlier.name == record.name
                        && earlier.record_type == record.record_type
                        && earlier.civil == record.civil
                })
        });
        let Some(index) = duplicate else {
            break;
        };
        clock.remove_record(index).expect("remove duplicate");
        assert_derived_matches_rebuild(&clock, &format!("{label}: duplicate removal"));
    }

    for step in 0..400 {
        let label = format!("{label}: step {step}");
        let count = clock.record_count();
        let operation = if step < 25 { 0 } else { sequence.below(10) };
        match operation {
            // Insert, at index 0 for the first 25 steps so the order keys run
            // out of room and are reassigned.
            0..=3 => {
                let index = if step < 25 {
                    0
                } else {
                    sequence.below(count + 1)
                };
                let name = names[sequence.below(names.len())];
                let record_type = if name.len() == 3 {
                    ClockRecordType::As
                } else {
                    ClockRecordType::Ar
                };
                let minute = [0u8, 5, 10][sequence.below(3)];
                let bias = (sequence.below(90) as f64 + 1.0) / 1.0e4;
                let values = if sequence.below(2) == 0 {
                    vec![bias]
                } else {
                    vec![bias, 1.0e-11]
                };
                let record = ClockRecord::new(
                    record_type,
                    name,
                    ClockEpoch {
                        year,
                        month,
                        day,
                        hour,
                        minute,
                        second: 0.0,
                    },
                    values,
                )
                .expect("new record");
                clock.insert_record(index, record).expect("insert");
            }
            4..=5 if count > 0 => {
                clock.remove_record(sequence.below(count)).expect("remove");
            }
            6..=8 if count > 0 => {
                let index = sequence.below(count);
                let record = clock.records().nth(index).expect("record");
                let last_surplus = record.surplus.iter().map(|value| value.position).max();
                let values = match last_surplus {
                    Some(last) if sequence.below(2) == 0 => {
                        let mut values = vec![7.0e-4];
                        values.resize(last + 1, 1.0e-11);
                        values
                    }
                    _ => vec![(sequence.below(90) as f64 + 1.0) / 1.0e4],
                };
                let before = clock.clone();
                let before_derived = clock.derived.clone();
                if clock.set_record_values(index, values).is_err() {
                    assert!(last_surplus.is_some(), "{label}");
                    assert_eq!(clock, before, "{label}");
                    assert_eq!(clock.derived, before_derived, "{label}");
                }
            }
            9 => {
                let system = if sequence.below(2) == 0 {
                    ClockTimeSystem::Utc
                } else {
                    ClockTimeSystem::Gps
                };
                clock.set_time_system(system).expect("set time system");
            }
            _ => {}
        }
        assert_derived_matches_rebuild(&clock, &label);
        if step % 50 == 49 {
            let written = clock.to_rinex_string().expect("write");
            assert_eq!(
                RinexClock::parse_lossy(&written).series(),
                clock.series(),
                "{label}"
            );
        }
    }
}

#[test]
fn record_by_record_maintenance_matches_a_full_rebuild() {
    run_edit_sequence(
        include_str!("../../tests/fixtures/clk/lossless/rinex_clock300_table_a17.clk"),
        (1994, 7, 14, 20),
        0x9e37_79b9_7f4a_7c15,
        "3.00 Table A17",
    );
    run_edit_sequence(
        include_str!(
            "../../tests/fixtures/clk/lossless/EMR0OPSRAP_20262600000_first_epoch_excerpt.clk"
        ),
        (2026, 9, 17, 0),
        0xd1b5_4a32_d192_ed03,
        "EMR excerpt",
    );
    run_edit_sequence(
        "     3.00           C                   G                   RINEX VERSION / TYPE
   GPS                                                      TIME SYSTEM ID
                                                            END OF HEADER
AS G01  2026 05 13 00 00  0.000000  1    0.100000000000E-03
AS G01  2026 05 13 00 00  0.000000  1    0.200000000000E-03
XX not a record
AR AREQ 2026 05 13 00 00  0.000000  1    0.300000000000E-03
AS G01  2026 05 13 00 05  0.000000  4    0.400000000000E-03  0.100000000000E-10
 0.100000000000E-12  0.100000000000E-14
     \x20
AS G02  2026 05 13 00 00  0.000000  1    0.500000000000E-03  0.100000000000E-10
AS G01  2026 05 13 00 00  0.000000  1    0.600000000000E-03
AR AREQ 2026 05 13 00 00  0.000000  1    0.700000000000E-03
garbage line
",
        (2026, 5, 13, 0),
        0x94d0_49bb_1331_11eb,
        "lossy text",
    );
}

#[test]
fn batch_edits_rebuild_once_and_change_nothing_on_refusal() {
    let text = include_str!(
        "../../tests/fixtures/clk/lossless/EMR0OPSRAP_20262600000_first_epoch_excerpt.clk"
    );
    let mut clock = RinexClock::parse(text).expect("EMR excerpt");
    let series = clock.series().clone();
    let removed = clock.retain_records(|record| record.record_type() != ClockRecordType::Ar);
    assert_eq!(removed, 104);
    assert_eq!(clock.record_count(), 49);
    assert!(clock.skipped_records().is_empty());
    assert_eq!(clock.series(), &series);
    assert_derived_matches_rebuild(&clock, "after retain");
    let written = clock.to_rinex_string().unwrap();
    assert!(!written.contains("\nAR "));
    assert!(written.starts_with(&text[..text.find("END OF HEADER").unwrap()]));

    // One refused edit refuses the batch: bias alone would drop the sigma.
    let before = clock.clone();
    assert!(clock
        .edit_records(|record| Some(vec![record.bias_s() * 2.0]))
        .is_err());
    assert_eq!(clock, before);
    let edited = clock
        .edit_records(|record| {
            let sigma = record.surplus_values().first()?.value;
            Some(vec![record.bias_s(), sigma])
        })
        .expect("restating every sigma");
    assert_eq!(edited, 49);
    assert!(clock.notices().is_empty(), "{:?}", clock.notices());
    assert_eq!(
        clock.series()["G01"][0].additional_values,
        vec![5.556437046250e-12]
    );
    assert_derived_matches_rebuild(&clock, "after edit_records");
}

#[test]
fn correctly_rounded_gps_seconds_are_written_strict() {
    // GPS seconds a caller writes as decimals with microsecond digits are the
    // correctly rounded doubles of those decimals, the same doubles the
    // product's own seconds export gives for records read from that text.
    // They are written as their microsecond text, and the product's series
    // rows stay the doubles it was built from.
    let mut state = 0x853c_49e6_748f_ea9b_u64;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    let mut decimals: Vec<String> = (0..4000)
        .map(|_| {
            format!(
                "{}.{:06}",
                1_300_000_000 + next(200_000_000),
                next(1_000_000)
            )
        })
        .collect();
    decimals.push("1463904030.123456".to_string());
    decimals.sort_by(|a, b| {
        a.parse::<f64>()
            .unwrap()
            .total_cmp(&b.parse::<f64>().unwrap())
    });
    decimals.dedup();
    let rows: Vec<(f64, f64)> = decimals
        .iter()
        .map(|decimal| (decimal.parse::<f64>().unwrap(), 1.0e-4))
        .collect();
    let clock = RinexClock::from_series_rows(vec![("G01".to_string(), rows.clone())])
        .expect("GPS-second rows");
    assert_eq!(clock.series_rows(), vec![("G01".to_string(), rows)]);
    let text = clock
        .to_rinex_string()
        .expect("every row is written strict");
    assert!(text.contains("AS G01  2026 05 27 08 00 30.123456"));
    let written: Vec<String> = RinexClock::parse(&text)
        .expect("reread")
        .records()
        .map(|record| {
            let epoch = record.civil_epoch();
            let gps = civil_to_gps_seconds(
                epoch.year,
                epoch.month,
                epoch.day,
                epoch.hour,
                epoch.minute,
                0.0,
            )
            .unwrap() as i64;
            let micro = (epoch.second * 1.0e6).round() as i64;
            format!("{}.{:06}", gps + micro / 1_000_000, micro % 1_000_000)
        })
        .collect();
    assert_eq!(written, decimals);
}

#[test]
fn a_leap_second_instant_just_before_midnight_rounds_to_midnight() {
    // 0.2 microseconds before the midnight that ends the 2016 leap second. The
    // nearest microsecond text is 2017-01-01 00:00:00.000000, never a second 61.
    let instant = Instant::from_julian_date(
        TimeScale::Utc,
        JulianDateSplit::new(2_457_754.5, -0.2e-6 / SECONDS_PER_DAY).expect("split"),
    );
    let clock = RinexClock::from_instant_series_rows(
        TimeScale::Utc,
        vec![("G05".to_string(), vec![(instant, 1.0e-4)])],
    )
    .expect("row");
    assert!(clock.to_rinex_string().is_err());
    let (text, departures) = clock
        .to_rinex_string_with_policy(ClockWritePolicy::lenient())
        .expect("nearest microsecond allowed");
    assert_eq!(
        departures,
        vec![ClockWriteDeparture::EpochAtNearestMicrosecond {
            record: 0,
            name: "G05".to_string(),
            epoch: Some(instant),
            written: "2017 01 01 00 00 0.000000".to_string(),
        }]
    );
    assert!(
        text.contains("AS G05  2017 01 01 00 00  0.000000"),
        "{text}"
    );
    // Half a second into the leap second is still written as 23:59:60.
    let leap = civil_to_clock_instant(TimeScale::Utc, 2016, 12, 31, 23, 59, 60.5).unwrap();
    let clock = RinexClock::from_instant_series_rows(
        TimeScale::Utc,
        vec![("G05".to_string(), vec![(leap, 1.0e-4)])],
    )
    .unwrap();
    assert!(clock
        .to_rinex_string()
        .unwrap()
        .contains("AS G05  2016 12 31 23 59 60.500000"));
}

/// Year, month, day, hour and minute of a civil tag.
type CivilMinute = (i64, i64, i64, i64, i64);

/// Days from 1980-01-06 to a proleptic Gregorian date, counted without the
/// crate's calendar code (days-from-civil over 400-year eras).
fn reference_days_since_gps_epoch(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let year_of_era = y - era * 400;
    let day_of_year = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    // 719_468 days from 0000-03-01 to 1970-01-01, then 3_657 to 1980-01-06.
    era * 146_097 + day_of_era - 719_468 - 3_657
}

/// The GPS seconds a civil tag states, as decimal text read by the correctly
/// rounded `str::parse`: the exact rational value rounded once.
fn reference_gps_seconds(
    (year, month, day, hour, minute): CivilMinute,
    whole_second: i64,
    fraction_digits: &str,
) -> f64 {
    const SCALE: i128 = 10_000_000_000_000;
    assert!(fraction_digits.len() <= 13);
    let whole = i128::from(reference_days_since_gps_epoch(year, month, day)) * 86_400
        + i128::from(hour * 3_600 + minute * 60 + whole_second);
    let fraction = format!("{fraction_digits:0<13}").parse::<i128>().unwrap();
    let total = whole * SCALE + fraction;
    let sign = if total < 0 { "-" } else { "" };
    let magnitude = total.unsigned_abs();
    format!(
        "{sign}{}.{:013}",
        magnitude / SCALE as u128,
        magnitude % SCALE as u128
    )
    .parse()
    .unwrap()
}

#[test]
fn a_seven_digit_tag_has_one_gps_second_value_on_every_path() {
    // 2026-05-13 00:00:59.9999996 GPST is 1462665659.9999996 GPS seconds,
    // whose nearest double is 0x41d5cba06efffffe. Adding the microseconds and
    // then the sub-microsecond digits as doubles gave the next double up for
    // the civil conversion, while the record read from the same text gave
    // this one.
    const STATED: u64 = 0x41d5_cba0_6eff_fffe;
    assert_eq!(
        "1462665659.9999996".parse::<f64>().unwrap().to_bits(),
        STATED
    );
    let civil = civil_to_gps_seconds(2026, 5, 13, 0, 0, 59.9999996).expect("GPS seconds");
    assert_eq!(civil.to_bits(), STATED);

    let text = "AS G05  2026 05 13 00 00 59.9999996  1   1.0e-04\n";
    let clock = RinexClock::parse(text).expect("seven-digit clock epoch");
    let record = clock.series()["G05"][0].gps_seconds().expect("GPST sample");
    assert_eq!(record.to_bits(), STATED);
    assert_eq!(clock.series_rows()[0].1[0].0.to_bits(), STATED);
    assert_eq!(
        clock
            .clock_s_at_gps_seconds("G05", civil)
            .expect("valid query"),
        Some(1.0e-4)
    );
}

#[test]
fn a_record_answers_a_query_at_its_own_gps_seconds() {
    // 2026-05-13 00:00:30.126705 is 1462665630.126705 GPS seconds, nearest
    // double 0x41d5cba067881bef. The split the GPS-seconds constructor builds
    // from that double falls just before the record's own split, so the query
    // instant lies before the first sample; the record answers at the GPS
    // seconds it exports.
    let text = "AS G05  2026 05 13 00 00 30.126705  1   1.0e-04\n\
                AS G05  2026 05 13 00 01  0.000000  1   2.0e-04\n";
    let clock = RinexClock::parse(text).expect("clock");
    let rows = clock.series_rows();
    let exported = rows[0].1[0].0;
    assert_eq!(exported.to_bits(), 0x41d5_cba0_6788_1bef);
    assert_eq!(
        exported,
        reference_gps_seconds((2026, 5, 13, 0, 0), 30, "126705")
    );
    let query = gps_seconds_to_instant(exported).expect("query instant");
    assert_eq!(
        epoch_cmp(&query, &clock.series()["G05"][0].epoch),
        Ordering::Less
    );
    for &(gps_seconds, bias_s) in &rows[0].1 {
        assert_eq!(
            clock
                .clock_s_at_gps_seconds("G05", gps_seconds)
                .expect("valid query"),
            Some(bias_s)
        );
    }
}

#[test]
fn the_stated_second_of_a_record_is_its_nearest_double() {
    // 30 + 0.000357 + 0.0000001 summed as doubles is 30.000357100000002; the
    // nearest double to 30.0003571 is the one the literal reads to.
    let text = "AS G05  2026 05 13 00 00 30.0003571  1   1.0e-04\n";
    let clock = RinexClock::parse(text).expect("clock");
    let second = clock.records().next().unwrap().civil_epoch().second;
    assert_eq!(second.to_bits(), 30.000_357_1_f64.to_bits());
}

#[test]
fn the_nearest_microsecond_rounds_the_stated_digits() {
    let civil = |minute: u32, second: u32, microsecond: u32, femtosecond: u32| Civil {
        year: 2026,
        month: 5,
        day: 13,
        hour: 0,
        minute,
        second,
        microsecond,
        femtosecond,
    };
    let fields = |c: Civil| (c.minute, c.second, c.microsecond, c.femtosecond);
    // 30.1234565 is half a microsecond past 30.123456 and rounds up; summed
    // as doubles it read 123456.49999999964 microseconds and rounded down.
    let tie = nearest_microsecond_civil(civil(0, 30, 123_456, 500_000_000)).unwrap();
    assert_eq!(fields(tie), (0, 30, 123_457, 0));
    let below = nearest_microsecond_civil(civil(0, 30, 123_456, 499_999_999)).unwrap();
    assert_eq!(fields(below), (0, 30, 123_456, 0));
    let carry = nearest_microsecond_civil(civil(0, 59, 999_999, 500_000_000)).unwrap();
    assert_eq!(fields(carry), (1, 0, 0, 0));
}

#[test]
fn gps_seconds_are_the_correctly_rounded_count_on_every_path() {
    // Civil tags from year 1 to 9999 with up to thirteen fractional second
    // digits. Each conversion is compared with the stated count as decimal
    // text read by the correctly rounded `str::parse`: the civil conversion
    // and the record path (a record stating the tag read into a GPST and a
    // QZSST product, then `ClockPoint::gps_seconds`) for every tag, and a sample
    // built from the reader's instant alone for tags of up to ten digits,
    // the finest the reader's split tells apart.
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    let days_in_month = |year: i64, month: i64| match month {
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    let mut cases: Vec<(CivilMinute, i64, String)> = vec![
        ((2026, 5, 13, 0, 0), 59, "9999996".to_string()),
        ((2026, 5, 13, 0, 0), 59, "9999999999999".to_string()),
        ((1999, 12, 31, 23, 59), 59, "9999999999".to_string()),
        ((9999, 12, 31, 23, 59), 59, "999999".to_string()),
        ((1, 1, 1, 0, 0), 0, "0000000000001".to_string()),
        ((1980, 1, 6, 0, 0), 0, "0000001".to_string()),
        ((1980, 1, 5, 23, 59), 59, "9999999".to_string()),
    ];
    for _ in 0..20_000 {
        let year = if next(2) == 0 {
            1980 + next(120) as i64
        } else {
            1 + next(9999) as i64
        };
        let month = 1 + next(12) as i64;
        let day = 1 + next(days_in_month(year, month) as u64) as i64;
        let digits = next(14) as usize;
        let fraction: String = (0..digits)
            .map(|_| char::from(b'0' + next(10) as u8))
            .collect();
        cases.push((
            (year, month, day, next(24) as i64, next(60) as i64),
            next(60) as i64,
            fraction,
        ));
    }
    for (fields, whole_second, fraction) in cases {
        let (year, month, day, hour, minute) = fields;
        let second: f64 = format!("{whole_second}.{fraction}0").parse().unwrap();
        let want = reference_gps_seconds(fields, whole_second, &fraction);
        let label = format!(
            "{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{whole_second:02}.{fraction}"
        );
        let args = (
            year as i32,
            month as u8,
            day as u8,
            hour as u8,
            minute as u8,
        );
        let got = civil_to_gps_seconds(args.0, args.1, args.2, args.3, args.4, second)
            .unwrap_or_else(|| panic!("{label}"));
        assert_eq!(got.to_bits(), want.to_bits(), "{label}: {got:e} {want:e}");
        for scale in [TimeScale::Gpst, TimeScale::Qzsst] {
            assert_eq!(
                record_gps_seconds(scale, fields, &format!("{whole_second}.{fraction}0"))
                    .map(f64::to_bits),
                Some(want.to_bits()),
                "{label} {scale:?} record"
            );
            if fraction.trim_end_matches('0').len() <= 10 {
                let epoch =
                    civil_to_clock_instant(scale, args.0, args.1, args.2, args.3, args.4, second)
                        .unwrap_or_else(|| panic!("{label}"));
                assert_eq!(
                    ClockPoint::new(epoch, 0.0, Vec::new())
                        .gps_seconds()
                        .map(f64::to_bits),
                    Some(want.to_bits()),
                    "{label} {scale:?} instant"
                );
            }
        }
    }
    // GPS time has no leap-second label: a second of 60 is refused.
    assert_eq!(civil_to_gps_seconds(2016, 12, 31, 23, 59, 60.0), None);
    assert_eq!(civil_to_gps_seconds(2016, 12, 31, 23, 59, 60.5), None);
}

#[test]
fn gps_seconds_rows_are_exported_as_they_were_given() {
    // Any double from about 24 days after the GPS epoch onward, where the
    // double grid is coarser than the day split's resolution, is exported as
    // the double the product was built from, on either side of the epoch.
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut rows = Vec::new();
    for _ in 0..20_000 {
        let unit = (next() >> 11) as f64 / (1_u64 << 53) as f64;
        let seconds = if next().is_multiple_of(2) {
            2.0e6 + unit * 2.4e11
        } else {
            -2.0e6 - unit * 6.0e10
        };
        rows.push(seconds);
    }
    rows.sort_by(f64::total_cmp);
    rows.dedup();
    let rows: Vec<(f64, f64)> = rows.into_iter().map(|seconds| (seconds, 1.0e-4)).collect();
    let clock =
        RinexClock::from_series_rows(vec![("G01".to_string(), rows.clone())]).expect("rows");
    let exported = clock.series_rows();
    for (&(got, _), &(want, _)) in exported[0].1.iter().zip(&rows) {
        assert_eq!(got.to_bits(), want.to_bits(), "{want:e}");
    }
    assert_eq!(exported[0].1.len(), rows.len());
}

/// GPS seconds of the one sample of a GPST or QZSST product read from a
/// record stating `seconds` as its seconds field, whitespace-separated.
fn record_gps_seconds(
    scale: TimeScale,
    (year, month, day, hour, minute): CivilMinute,
    seconds: &str,
) -> Option<f64> {
    let (header, name) = match scale {
        TimeScale::Qzsst => (
            " 3.04                 C                    J                      RINEX VERSION / TYPE\n\
                QZS                                                           TIME SYSTEM ID\n\
                                                                              END OF HEADER\n",
            "J02",
        ),
        _ => ("", "G05"),
    };
    let text = format!(
        "{header}AS {name} {year:04} {month:02} {day:02} {hour:02} {minute:02} {seconds} 1 1.0e-04\n"
    );
    let clock = RinexClock::parse(&text).ok()?;
    assert_eq!(clock.time_scale(), Some(scale), "{text}");
    clock.series().get(name)?.first()?.gps_seconds()
}

#[test]
fn tags_the_split_cannot_tell_apart_keep_their_own_gps_seconds() {
    // Under the pre-3.0.0 arithmetic, which rounded the split twice, each
    // tag's reader split was also the reader split of a ten-digit tag on the
    // other side of a rounding boundary of the GPS-seconds grid, so the
    // instant alone gave the neighbouring double. A record read from text, in
    // a GPST or a QZSST product, keeps its tag and gives the correctly rounded
    // value. The split is now rounded once, within 5e-12 s of the tag, and
    // the instant alone gives the same value for these tags. (`ClockRecord::new`
    // refuses a tag finer than a microsecond on insertion, so only a read
    // record can hold one.)
    for (fields, whole_second, fraction) in [
        ((2002, 10, 2, 19, 39), 7, "79081088304"),
        ((2022, 9, 26, 21, 6), 52, "513679146771"),
    ] {
        let (year, month, day, hour, minute) = fields;
        let want = reference_gps_seconds(fields, whole_second, fraction);
        let second: f64 = format!("{whole_second}.{fraction}").parse().unwrap();
        let tag = ClockEpoch {
            year: year as i32,
            month: month as u8,
            day: day as u8,
            hour: hour as u8,
            minute: minute as u8,
            second,
        };
        let civil = civil_to_gps_seconds(
            tag.year, tag.month, tag.day, tag.hour, tag.minute, tag.second,
        )
        .unwrap();
        assert_eq!(civil.to_bits(), want.to_bits());
        assert_eq!(
            record_gps_seconds(
                TimeScale::Qzsst,
                fields,
                &format!("{whole_second}.{fraction}")
            )
            .map(f64::to_bits),
            Some(want.to_bits())
        );
        let text = format!(
            "AS G05 {year:04} {month:02} {day:02} {hour:02} {minute:02} {whole_second}.{fraction} 1 1.0e-04\n"
        );
        let clock = RinexClock::parse(&text).expect("whitespace record");
        let point = &clock.series()["G05"][0];
        assert_eq!(
            point.gps_seconds().map(f64::to_bits),
            Some(want.to_bits()),
            "{text}"
        );
        assert_eq!(clock.series_rows()[0].1[0].0.to_bits(), want.to_bits());
        assert_eq!(
            clock
                .clock_s_at_gps_seconds("G05", want)
                .expect("valid query"),
            Some(1.0e-4)
        );
        let lookup = ClockPoint::new(point.epoch, 1.0e-4, Vec::new())
            .gps_seconds()
            .unwrap();
        assert_eq!(lookup.to_bits(), want.to_bits(), "the instant alone");
    }
}

#[test]
fn a_qzsst_record_answers_a_query_at_its_own_gps_seconds() {
    let text =
        " 3.04                 C                    J                      RINEX VERSION / TYPE\n\
                   QZS                                                           TIME SYSTEM ID\n\
                                                                                 END OF HEADER\n\
AS J02       2026 05 13 00 00 30.126705  1   -0.232835122007E-05\n\
AS J02       2026 05 13 00 01  0.000000  1   -0.232835122107E-05\n";
    let clock = RinexClock::parse(text).expect("QZS clock");
    assert_eq!(clock.time_scale(), Some(TimeScale::Qzsst));
    let rows = clock.series_rows();
    assert_eq!(rows[0].1[0].0.to_bits(), 0x41d5_cba0_6788_1bef);
    for &(gps_seconds, bias_s) in &rows[0].1 {
        assert_eq!(
            clock
                .clock_s_at_gps_seconds("J02", gps_seconds)
                .expect("valid query"),
            Some(bias_s)
        );
    }
}

#[test]
fn samples_sharing_one_gps_seconds_value_are_interpolated_between() {
    // 30.1261057 and 30.1261058 are 0.1 microseconds apart and both round to
    // 0x41d5cba06788121e GPS seconds. A query at that value names neither
    // sample alone and is interpolated at its own instant.
    let text = "AS G05  2026 05 13 00 00 30.1261057  1   1.0e-04\n\
                AS G05  2026 05 13 00 00 30.1261058  1   2.0e-04\n\
                AS G05  2026 05 13 00 01  0.000000  1   3.0e-04\n";
    let clock = RinexClock::parse(text).expect("clock");
    let shared = clock.series_rows()[0].1[0].0;
    assert_eq!(shared.to_bits(), 0x41d5_cba0_6788_121e);
    assert_eq!(clock.series_rows()[0].1[1].0.to_bits(), shared.to_bits());
    let query = gps_seconds_to_instant(shared).expect("query instant");
    assert_eq!(
        clock
            .clock_s_at_gps_seconds("G05", shared)
            .expect("valid query"),
        clock.clock_s_at_instant("G05", query).expect("valid query")
    );
}

#[test]
fn gps_seconds_exported_before_3_0_0_are_written_as_their_tags() {
    // Before 3.0.0 `series_rows` summed the reader split's two parts in f64.
    // For these tags that sum is one unit in the last place from the
    // correctly rounded double. A product rebuilt from it is written strict
    // as the tag, with no departure under the lenient policy, and reading the
    // text back gives the correctly rounded double.
    let tags = ["30.095029", "30.118786", "30.126705", "30.142543"];
    let mut rows = Vec::new();
    let mut correct = Vec::new();
    for tag in tags {
        let second: f64 = tag.parse().unwrap();
        // The 2.x reader's split: the clock fields summed in f64, then divided
        // by the day. Every tag here is on the microsecond grid.
        let whole = second.trunc();
        let microsecond = ((second - whole) * 1.0e6).round();
        let fraction_2x = (whole + microsecond / 1.0e6) / SECONDS_PER_DAY;
        let jd_whole = crate::astro::time::scales::julian_day_number(2026, 5, 13) as f64 - 0.5;
        let split_2x = Instant::from_julian_date(
            TimeScale::Gpst,
            JulianDateSplit::new(jd_whole, fraction_2x).expect("2.x split"),
        );
        let exported_2x = instant_to_j2000_seconds(&split_2x).unwrap() + GPS_EPOCH_TO_J2000_S;
        let rounded = civil_to_gps_seconds(2026, 5, 13, 0, 0, second).unwrap();
        assert_ne!(exported_2x, rounded, "{tag}");
        rows.push((exported_2x, 1.0e-4));
        correct.push((rounded, 1.0e-4));
    }
    let clock =
        RinexClock::from_series_rows(vec![("G05".to_string(), rows.clone())]).expect("2.x rows");
    assert_eq!(clock.series_rows()[0].1, rows);
    let text = clock.to_rinex_string().expect("written strict");
    for tag in tags {
        let line = format!("AS G05  2026 05 13 00 00 {tag}");
        assert!(text.contains(&line), "{line}\n{text}");
    }
    let (lenient, departures) = clock
        .to_rinex_string_with_policy(ClockWritePolicy::lenient())
        .expect("written lenient");
    assert!(departures.is_empty(), "{departures:?}");
    assert_eq!(lenient, text);
    let reread = RinexClock::parse(&text).expect("reread");
    assert_eq!(reread.series_rows()[0].1, correct);
}

#[test]
fn intervals_between_clock_tags_are_exact() {
    // A day of 30 s records. Each record's split is rounded, so the day
    // fractions of two records 30 s apart do not differ by exactly 30 s; the
    // interval between their tags is.
    let mut text = String::from("     3.00           C                   G                   RINEX VERSION / TYPE\n   GPS                                                      TIME SYSTEM ID\n                                                            END OF HEADER\n");
    for k in 0..2880_u32 {
        let seconds = k * 30;
        text.push_str(&format!(
            "AS G05  2026 05 13 {:02} {:02} {:2}.000000  1   {}.0e-06\n",
            seconds / 3600,
            seconds % 3600 / 60,
            seconds % 60,
            k % 7
        ));
    }
    let clock = RinexClock::parse(&text).expect("clock");
    let records = &clock.series()["G05"];
    let mut split_intervals_off = 0;
    for pair in records.windows(2) {
        let (p0, p1) = (&pair[0], &pair[1]);
        assert_eq!(
            super::epoch::seconds_between((&p1.epoch, p1.source), (&p0.epoch, p0.source)),
            Some(30.0)
        );
        let (a, b) = (
            p0.epoch.julian_date().unwrap(),
            p1.epoch.julian_date().unwrap(),
        );
        if crate::astro::time::civil::seconds_between_splits(
            b.jd_whole, b.fraction, a.jd_whole, a.fraction,
        ) != 30.0
        {
            split_intervals_off += 1;
        }
    }
    assert!(split_intervals_off > 0);

    // A query by civil tag, by GPS seconds and by instant is taken at its
    // exact time: 10 s into a 30 s span is exactly a third of it.
    let expected = crate::astro::math::interp::lerp_ratio(
        records[100].bias_s,
        records[101].bias_s,
        10.0,
        30.0,
    );
    let tag = ClockEpoch {
        year: 2026,
        month: 5,
        day: 13,
        hour: 0,
        minute: 50,
        second: 10.0,
    };
    assert_eq!(clock.clock_s("G05", tag).unwrap(), Some(expected));
    let instant = civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 50, 10.0).unwrap();
    assert_eq!(
        clock.clock_s_at_instant("G05", instant).unwrap(),
        Some(expected)
    );
    let gps_seconds = civil_to_gps_seconds(2026, 5, 13, 0, 50, 10.0).unwrap();
    assert_eq!(
        clock.clock_s_at_gps_seconds("G05", gps_seconds).unwrap(),
        Some(expected)
    );
}

#[test]
fn intervals_between_fractional_clock_tags_are_the_decimal_difference() {
    let text = "AS G05  2026 05 13 00 00 30.1261057  1   1.0e-04\n\
                AS G05  2026 05 13 00 00 30.3261058  1   2.0e-04\n";
    let clock = RinexClock::parse(text).expect("clock");
    let records = &clock.series()["G05"];
    let (p0, p1) = (&records[0], &records[1]);
    assert_eq!(
        super::epoch::seconds_between((&p1.epoch, p1.source), (&p0.epoch, p0.source)),
        Some(0.2000001)
    );
    // A split Julian date that is no tag's reading is taken at the exact time
    // its two parts hold.
    let split = p0.epoch.julian_date().unwrap();
    let off_grid = Instant::from_julian_date(
        TimeScale::Gpst,
        JulianDateSplit::new(split.jd_whole, split.fraction + f64::EPSILON / 8.0).unwrap(),
    );
    let seconds = super::epoch::seconds_between(
        (&off_grid, EpochSource::Instant),
        (&p0.epoch, EpochSource::Instant),
    )
    .unwrap();
    assert!(seconds != 0.0 && seconds.abs() < 1.0e-11, "{seconds:e}");
}

#[test]
fn an_instant_the_2x_reader_built_is_written_as_its_tag() {
    // Before 3.0.0 the reader summed the clock fields in f64 and divided by
    // the day; for 00:00:30.007919 that split is one unit in the last place
    // from the correctly rounded one the reader builds now. A product holding
    // the 2.x split is written strict as the tag.
    let tag = civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 30.007_919).unwrap();
    let split = tag.julian_date().unwrap();
    assert_eq!(split.fraction.to_bits(), 0x3f36_c2f5_be98_79a2);
    let split_2x = Instant::from_julian_date(
        TimeScale::Gpst,
        JulianDateSplit::new(split.jd_whole, f64::from_bits(0x3f36_c2f5_be98_79a3)).unwrap(),
    );
    assert_eq!(
        split_2x.julian_date().unwrap().fraction,
        (30.0 + 7_919.0 / 1.0e6) / SECONDS_PER_DAY
    );
    let clock = RinexClock::from_instant_series_rows(
        TimeScale::Gpst,
        vec![("G05".to_string(), vec![(split_2x, 1.0e-4)])],
    )
    .expect("2.x instant");
    let text = clock.to_rinex_string().expect("written strict");
    assert!(
        text.contains("AS G05  2026 05 13 00 00 30.007919"),
        "{text}"
    );
}

#[test]
fn samples_held_off_the_midnight_boundary_are_ordered_by_time() {
    // 2026-09-22 18:00 held as (2461306.0, 0.25): later than 17:00 and
    // earlier than 19:00 of the same day, whose splits sit on the midnight
    // boundary 2461305.5.
    let at = |hour: u8, minute: u8| {
        civil_to_clock_instant(TimeScale::Gpst, 2026, 9, 22, hour, minute, 0.0).unwrap()
    };
    let six_pm = Instant::from_julian_date(
        TimeScale::Gpst,
        JulianDateSplit::new(2_461_306.0, 0.25).unwrap(),
    );
    let clock = RinexClock::from_instant_series_rows(
        TimeScale::Gpst,
        vec![(
            "G05".to_string(),
            vec![(at(17, 0), 1.0e-4), (six_pm, 2.0e-4), (at(19, 0), 4.0e-4)],
        )],
    )
    .expect("increasing in time");
    assert_eq!(
        clock.clock_s_at_instant("G05", at(18, 30)).unwrap(),
        Some(crate::astro::math::interp::lerp_ratio(
            2.0e-4, 4.0e-4, 1_800.0, 3_600.0
        ))
    );
    assert_eq!(
        clock.clock_s_at_instant("G05", six_pm).unwrap(),
        Some(2.0e-4)
    );
}

#[test]
fn a_query_between_a_samples_reading_and_its_tag_is_bracketed_by_time() {
    // The reader's split of 00:01:30 lies 1.25e-15 s before the tag. A query
    // split between the two (held on a boundary 90 s into the day) is before
    // the sample the interpolation measures at its tag, so it is bracketed by
    // 00:00:00 and 00:01:30, not by 00:01:30 and 00:03:00.
    let text = "     3.00           C                   G                   RINEX VERSION / TYPE\n   GPS                                                      TIME SYSTEM ID\n                                                            END OF HEADER\n\
                AS G05  2026 05 13 00 00  0.000000  1   1.0e-04\n\
                AS G05  2026 05 13 00 01 30.000000  1   2.0e-04\n\
                AS G05  2026 05 13 00 03  0.000000  1   3.0e-04\n";
    let clock = RinexClock::parse(text).expect("clock");
    let sample = clock.series()["G05"][1].epoch.julian_date().unwrap();
    assert_eq!(
        (sample.jd_whole, sample.fraction.to_bits()),
        (2_461_173.5, 0x3f51_1111_1111_1111)
    );
    let query = Instant::from_julian_date(
        TimeScale::Gpst,
        JulianDateSplit::new(
            f64::from_bits(0x4142_c6fa_c022_2222),
            f64::from_bits(0x3dd1_1111_1108_8889),
        )
        .unwrap(),
    );
    let bias = clock
        .clock_s_at_instant("G05", query)
        .unwrap()
        .expect("bracketed");
    assert_eq!(
        bias,
        crate::astro::math::interp::lerp_ratio(1.0e-4, 2.0e-4, 90.0, 90.0)
    );
}
