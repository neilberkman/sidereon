//! AIUB's short-name CODE products before GPS week 2238, one real file per sampling
//! era, read through the crate's readers and checked against the catalog identity of
//! their date: the SP3 orbits through exact validation, the clock and IONEX excerpts
//! through the clock and IONEX readers. The files and their trims are recorded in
//! `fixtures/{sp3,clk,ionex}/PROVENANCE.md`.

use sha2::{Digest, Sha256};
use sidereon_core::atmosphere::Ionex;
use sidereon_core::data::{mgex_clk, mgex_ionex, mgex_sp3, AnalysisCenter, ProductDate};
use sidereon_core::ephemeris::{
    parse_exact_sp3, ExactSp3Coverage, ExactSp3Request, ExactSp3ValidationError,
};
use sidereon_core::rinex::clock::{civil_to_gps_seconds, RinexClock, RinexClockNotice};

fn date(year: i32, month: u8, day: u8) -> ProductDate {
    ProductDate::new(year, month, day).expect("valid date")
}

fn fixture(dir: &str, name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{dir}/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|error| panic!("read {path}: {error}"))
}

fn fixture_text(dir: &str, name: &str) -> String {
    String::from_utf8(fixture(dir, name)).expect("ASCII fixture")
}

#[test]
fn code_ionex_excerpt_bytes_match_recorded_provenance() {
    for (name, expected) in [
        (
            "CODG0010.95I",
            "fe02575ef376c2d550970e861d8cf6987cdf9ee4133d667ad180973c0a51beaa",
        ),
        (
            "CODG0330_maps1-2.97I",
            "464af65493c50972dbe6a91951394a0226cfa8f489d40a72aa15639543da6e09",
        ),
        (
            "CODG0550.97I",
            "93b0106a056504279542862b5cef9816797799cadc0f3f0e3ae32adbf45b2a17",
        ),
        (
            "CODG0870_maps1-2.98I",
            "2482130ee6ab9826195d5c4870c71b634000395f29a0b9a7586c906648c9f9d2",
        ),
        (
            "CODG3070_maps1-2.02I",
            "bbb4b45f993103cd6f39a691bc3b4d702428f857530a639b1ddfe7399ca55a28",
        ),
        (
            "CODG2910_maps1-3.14I",
            "17fff5734f3933e4e9c03420122f1d6fef89560a9015998ae0ace682141e8590",
        ),
        (
            "CODG2920_maps1-3.14I",
            "f1eadde3fb4bcd0572c40b73f65a0e785c5dab5ea2c28aecadfbc663f1ecdcf6",
        ),
    ] {
        let bytes = fixture("ionex", name);
        assert!(bytes.is_ascii(), "{name}");
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected, "{name}");
    }
}

/// Seconds a sampling token states (`15M`, `30S`, `02H`, ...).
fn token_seconds(token: &str) -> f64 {
    let (amount, unit) = token.split_at(2);
    let amount: f64 = amount.parse().expect("two-digit amount");
    amount
        * match unit {
            "S" => 1.0,
            "M" => 60.0,
            "H" => 3600.0,
            "D" => 86_400.0,
            other => panic!("unit {other}"),
        }
}

/// The first 15-minute and the first 5-minute short-name MGEX orbit files pass exact
/// validation against the catalog identity of their date, midnight to midnight
/// inclusive (97 and 289 epochs), agency `AIUB`, and fail it at the other sampling.
#[test]
fn code_short_name_sp3_files_pass_exact_validation_for_their_identity() {
    for (name, product_date, sample, other) in [
        ("COM17733.EPH", date(2014, 1, 1), "15M", "05M"),
        ("COM19610.EPH", date(2017, 8, 6), "05M", "15M"),
    ] {
        let identity = mgex_sp3(AnalysisCenter::Cod, product_date, None)
            .expect("short-name CODE SP3")
            .identity()
            .expect("identity");
        assert_eq!(identity.official_filename, name);
        assert_eq!(identity.sample, sample);
        let bytes = fixture("sp3", name);
        let request = ExactSp3Request::from_identity(&identity).expect("request");
        assert_eq!(request.expected_agency(), Some("AIUB"));
        let (_, coverage) =
            parse_exact_sp3(&bytes, &request).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(coverage, ExactSp3Coverage::Inclusive, "{name}");

        let at_other = ExactSp3Request::new(product_date, None, "01D", other).expect("request");
        assert!(
            matches!(
                parse_exact_sp3(&bytes, &at_other),
                Err(ExactSp3ValidationError::CadenceMismatch { .. })
            ),
            "{name} is not a {other} product"
        );
    }
}

/// The short-name MGEX clock excerpts read with the crate's strict clock reader, and
/// every satellite clock series steps at the sampling the catalog identity of the date
/// states: 300 s on 2014-01-01, 30 s from 2017-08-13. The excerpts keep the first three
/// satellite epochs from midnight.
#[test]
fn code_short_name_clock_excerpts_step_at_their_identity_sampling() {
    for (name, (year, month, day), official, sample) in [
        (
            "COM17733_0000-0010.CLK",
            (2014, 1, 1),
            "COM17733.CLK",
            "05M",
        ),
        (
            "COM19620_0000-0100.CLK",
            (2017, 8, 13),
            "COM19620.CLK",
            "30S",
        ),
    ] {
        let identity = mgex_clk(AnalysisCenter::Cod, date(year, month, day), None)
            .expect("short-name CODE clock")
            .identity()
            .expect("identity");
        assert_eq!(identity.official_filename, official);
        assert_eq!(identity.sample, sample);
        let step_s = token_seconds(sample);

        let clock = RinexClock::parse(&fixture_text("clk", name))
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        assert!(
            clock.diagnostics().is_empty(),
            "{name}: {:?}",
            clock.diagnostics()
        );
        // AIUB flags some satellite records with a letter in column 83, past the 80
        // columns of a version 2.00 record: C12 at 00:05 and 00:10 in the 2014 file.
        let trailing: Vec<&RinexClockNotice> = clock
            .notices()
            .iter()
            .filter(|notice| matches!(notice, RinexClockNotice::TrailingTextRecords { .. }))
            .collect();
        if name == "COM17733_0000-0010.CLK" {
            assert_eq!(
                trailing,
                [&RinexClockNotice::TrailingTextRecords {
                    records: 2,
                    first_line: 489,
                }],
                "{name}"
            );
            let c12 = &clock.series()["C12"];
            assert_eq!(c12.len(), 3, "{name}");
            assert_eq!(c12[1].bias_s.to_bits(), 0.535575502280E-03_f64.to_bits());

            let (record_index, flagged) = clock
                .records()
                .enumerate()
                .find(|(_, record)| record.satellite() == Some("C12") && record.line() == Some(489))
                .expect("first flagged C12 record");
            let original = fixture_text("clk", name);
            let original_line = original.lines().nth(488).expect("source line 489");
            let suffix = original_line.get(80..).expect("ASCII suffix");
            assert!(suffix.contains('E'), "{suffix:?}");
            assert_eq!(flagged.declared_count(), 2);
            let mut edited = clock.clone();
            let mut values = flagged.values().to_vec();
            values[0] += 1.0e-9;
            edited
                .set_record_values(record_index, values)
                .expect("edit flagged clock record");
            let written = edited.to_rinex_string().expect("write edited clock");
            let written_line = written.lines().nth(488).expect("written line 489");
            assert_ne!(&written_line[..80], &original_line[..80]);
            assert_eq!(&written_line[80..], suffix);
            let reparsed = RinexClock::parse(&written).expect("read edited clock");
            assert!(reparsed.notices().iter().any(|notice| matches!(
                notice,
                RinexClockNotice::TrailingTextRecords {
                    records: 2,
                    first_line: 489
                }
            )));

            let mut one_value_edit = clock.clone();
            one_value_edit
                .set_record_values(record_index, vec![flagged.values()[0] + 2.0e-9])
                .expect("reduce flagged clock record to one value");
            let one_value_text = one_value_edit
                .to_rinex_string()
                .expect("write one-value clock record");
            let one_value_line = one_value_text.lines().nth(488).expect("written line 489");
            assert_eq!(&one_value_line[79..], &original_line[79..]);
            let one_value_reparsed =
                RinexClock::parse(&one_value_text).expect("read one-value clock record");
            assert!(one_value_reparsed.notices().iter().any(|notice| matches!(
                notice,
                RinexClockNotice::TrailingTextRecords {
                    records: 2,
                    first_line: 489
                }
            )));
        } else {
            assert!(trailing.is_empty(), "{name}: {trailing:?}");
        }
        let midnight = civil_to_gps_seconds(year, month, day, 0, 0, 0.0).expect("midnight");
        let rows = clock.series_rows();
        assert!(rows.len() >= 60, "{name}: {} satellites", rows.len());
        let mut epochs = std::collections::BTreeSet::new();
        for (satellite, points) in &rows {
            assert!(!points.is_empty(), "{name} {satellite}");
            for pair in points.windows(2) {
                assert_eq!(pair[1].0 - pair[0].0, step_s, "{name} {satellite}");
            }
            for (t, _) in points {
                epochs.insert((t - midnight).to_bits());
            }
        }
        let expected: std::collections::BTreeSet<u64> = [0.0, step_s, 2.0 * step_s]
            .into_iter()
            .map(f64::to_bits)
            .collect();
        assert_eq!(epochs, expected, "{name}");
    }
}

/// Real short-name IONEX products from each published interval era read with the crate's
/// IONEX reader, and their map interval is the sampling the catalog identity states.
#[test]
fn code_short_name_ionex_excerpts_read_at_their_identity_map_interval() {
    let mut first_epochs = Vec::new();
    for (name, product_date, official, sample, map_count, rms_count) in [
        (
            "CODG0010.95I",
            date(1995, 1, 1),
            "CODG0010.95I",
            "01D",
            1,
            0,
        ),
        (
            "CODG0330_maps1-2.97I",
            date(1997, 2, 2),
            "CODG0330.97I",
            "02H",
            2,
            2,
        ),
        (
            "CODG0550.97I",
            date(1997, 2, 24),
            "CODG0550.97I",
            "01D",
            1,
            0,
        ),
        (
            "CODG0870_maps1-2.98I",
            date(1998, 3, 28),
            "CODG0870.98I",
            "02H",
            2,
            0,
        ),
        (
            "CODG3070_maps1-2.02I",
            date(2002, 11, 3),
            "CODG3070.02I",
            "02H",
            2,
            2,
        ),
        (
            "CODG2910_maps1-3.14I",
            date(2014, 10, 18),
            "CODG2910.14I",
            "02H",
            3,
            3,
        ),
        (
            "CODG2920_maps1-3.14I",
            date(2014, 10, 19),
            "CODG2920.14I",
            "01H",
            3,
            3,
        ),
    ] {
        let identity = mgex_ionex(AnalysisCenter::Cod, product_date, None)
            .expect("short-name CODE IONEX")
            .identity()
            .expect("identity");
        assert_eq!(identity.official_filename, official);
        assert_eq!(identity.sample, sample);
        let step_s = token_seconds(sample);

        let ionex = Ionex::parse_str(&fixture_text("ionex", name))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(f64::from(ionex.header().interval_s), step_s, "{name}");
        assert_eq!(ionex.tec_maps().len(), map_count, "{name}");
        assert_eq!(ionex.rms_maps().len(), rms_count, "{name}");
        let epochs = ionex.map_epochs_s();
        assert_eq!(epochs.len(), map_count, "{name}");
        for pair in epochs.windows(2) {
            assert_eq!((pair[1] - pair[0]) as f64, step_s, "{name}");
        }
        first_epochs.push(epochs[0]);
    }
    // The two 2014 boundary samples begin at midnight on consecutive days.
    assert_eq!(first_epochs[6] - first_epochs[5], 86_400);
}
