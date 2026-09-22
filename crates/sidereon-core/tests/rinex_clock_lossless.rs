#![cfg(sidereon_repo_tests)]

//! RINEX clock products keep their source lines as the authority.
//!
//! Fixtures in `tests/fixtures/clk/lossless/` (provenance, hashes and
//! extraction in `clock_fixture_provenance.json` there):
//!
//! - `rinex_clock300_table_a17.clk`, `rinex_clock300_table_a18.clk`: the
//!   Table A17 and A18 examples of the RINEX clock 3.00 specification
//!   (<https://files.igs.org/pub/data/format/rinex_clock300.txt>, lines 628-661
//!   and 674-686), byte for byte.
//! - `rinex_clock304_table_a17.clk`, `rinex_clock304_another_example.clk`,
//!   `rinex_clock304_table_a18.clk`: the Table A17, "ANOTHER EXAMPLE" and
//!   Table A18 examples of the RINEX clock 3.04 specification
//!   (<https://files.igs.org/pub/data/format/rinex_clock304.txt>, lines 622-655,
//!   667-714 and 727-739), byte for byte with their CRLF terminators.
//! - `*_first_epoch_excerpt.clk`: the header and the first epoch of four IGS
//!   analysis-centre products from
//!   <https://igs.bkg.bund.de/root_ftp/IGSac/products/2436/>
//!   (COD0OPSRAP, EMR0OPSRAP, ESA0OPSRAP and GBM0MGXRAP
//!   `_20262600000_01D_*_CLK.CLK`), each a byte prefix of the product.
//!
//! Measured on the complete products (not committed): every data line of all
//! six public products examined (the four above, GRG0OPSRAP_20222800000 and
//! IGS0OPSFIN_20262140000) fits the 3.00 record columns; every one of the
//! 140,974 AS records in EMR0OPSRAP declares one value and carries a bias sigma
//! in the sigma columns; no product repeats a record for one type, name and
//! epoch; ESA0OPSRAP (3.02), GBM0MGXRAP (3.00) and IGS0OPSFIN (3.00) have no
//! TIME SYSTEM ID record and a blank satellite system field.

use sidereon_core::astro::time::model::TimeScale;
use sidereon_core::rinex::clock::{
    civil_to_clock_instant, ClockEpoch, ClockHeaderField, ClockHeaderReading, ClockLayout,
    ClockRecord, ClockRecordReading, ClockRecordType, ClockSurplusValue, ClockTimeSystem,
    ClockTimeSystemStatus, RinexClock, RinexClockError, RinexClockNotice, RinexClockSkip,
};

macro_rules! fixture {
    ($name:literal) => {
        include_str!(concat!("fixtures/clk/lossless/", $name))
    };
}

const SPEC_300_A17: &str = fixture!("rinex_clock300_table_a17.clk");
const SPEC_300_A18: &str = fixture!("rinex_clock300_table_a18.clk");
const SPEC_304_A17: &str = fixture!("rinex_clock304_table_a17.clk");
const SPEC_304_ANOTHER: &str = fixture!("rinex_clock304_another_example.clk");
const SPEC_304_A18: &str = fixture!("rinex_clock304_table_a18.clk");
const COD: &str = fixture!("COD0OPSRAP_20262600000_first_epoch_excerpt.clk");
const EMR: &str = fixture!("EMR0OPSRAP_20262600000_first_epoch_excerpt.clk");
const ESA: &str = fixture!("ESA0OPSRAP_20262600000_first_epoch_excerpt.clk");
const GBM: &str = fixture!("GBM0MGXRAP_20262600000_first_epoch_excerpt.clk");
const SYNTHETIC: &str = include_str!("fixtures/clk/synthetic_rinex_clock.clk");
const IGS_FINAL: &str = include_str!("fixtures/clk/IGS0OPSFIN_20261330000_90M_30S_CLK.CLK");

const ALL_FIXTURES: [(&str, &str); 11] = [
    ("rinex_clock300_table_a17", SPEC_300_A17),
    ("rinex_clock300_table_a18", SPEC_300_A18),
    ("rinex_clock304_table_a17", SPEC_304_A17),
    ("rinex_clock304_another_example", SPEC_304_ANOTHER),
    ("rinex_clock304_table_a18", SPEC_304_A18),
    ("COD0OPSRAP", COD),
    ("EMR0OPSRAP", EMR),
    ("ESA0OPSRAP", ESA),
    ("GBM0MGXRAP", GBM),
    ("synthetic", SYNTHETIC),
    ("IGS0OPSFIN_90M", IGS_FINAL),
];

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

fn header_field(clock: &RinexClock, label: &str, payload_prefix: &str) -> ClockHeaderField {
    clock
        .header_records()
        .into_iter()
        .find(|record| record.label() == label && record.payload().starts_with(payload_prefix))
        .and_then(|record| record.field().cloned())
        .unwrap_or_else(|| panic!("typed {label} record starting {payload_prefix:?}"))
}

fn record_named(clock: &RinexClock, name: &str) -> ClockRecord {
    clock
        .records()
        .find(|record| record.name() == name)
        .unwrap_or_else(|| panic!("record {name}"))
}

fn skip(line: usize, record_type: &str) -> RinexClockSkip {
    RinexClockSkip::new(line, record_type)
}

#[test]
fn every_fixture_restates_its_bytes() {
    for (name, text) in ALL_FIXTURES {
        let strict = RinexClock::parse(text).unwrap_or_else(|err| panic!("{name}: {err}"));
        assert_eq!(strict.to_rinex_string().as_deref(), Ok(text), "{name}");
        let lossy = RinexClock::parse_lossy(text);
        assert!(lossy.diagnostics().is_empty(), "{name}");
        assert_eq!(lossy, strict, "{name}");
        assert_eq!(lossy.to_rinex_string().as_deref(), Ok(text), "{name}");
        let reparsed = RinexClock::parse(&strict.to_rinex_string().unwrap()).unwrap();
        assert_eq!(reparsed, strict, "{name}");
    }
}

#[test]
fn spec_300_example_reads_at_the_300_columns() {
    let clock = RinexClock::parse(SPEC_300_A17).expect("3.00 Table A17");
    assert_eq!(clock.version(), Some(3.00));
    assert_eq!(clock.layout(), Some(ClockLayout::V300));
    assert_eq!(clock.satellite_system(), Some('G'));
    assert_eq!(clock.time_system(), Some(ClockTimeSystem::Gps));
    assert_eq!(clock.time_system_status(), &ClockTimeSystemStatus::Declared);
    assert_eq!(clock.time_scale(), Some(TimeScale::Gpst));
    assert!(clock.notices().is_empty(), "{:?}", clock.notices());

    let header = clock.header_records();
    assert_eq!(header.len(), 26);
    assert!(header
        .iter()
        .all(|record| record.reading() == ClockHeaderReading::Columns));
    assert!(header.iter().all(|record| record.label_column() == 60));
    assert_eq!(
        header[0].field(),
        Some(&ClockHeaderField::VersionType {
            version: 3.00,
            file_type: "CLOCK DATA".to_string(),
            satellite_system: "GPS".to_string(),
        })
    );
    assert_eq!(
        header_field(&clock, "SYS / # / OBS TYPES", "G"),
        ClockHeaderField::ObservationTypes {
            system: Some('G'),
            count: Some(4),
            descriptors: ["C1W", "L1W", "C2W", "L2W"].map(String::from).to_vec(),
        }
    );
    assert_eq!(
        header_field(&clock, "LEAP SECONDS", ""),
        ClockHeaderField::LeapSeconds(10)
    );
    assert_eq!(
        header_field(&clock, "SYS / DCBS APPLIED", "G"),
        ClockHeaderField::DcbsApplied {
            system: "G".to_string(),
            program: "CC2NONCC".to_string(),
            source: "p1c1bias.hist @ goby.nrl.navy.mil".to_string(),
        }
    );
    assert_eq!(
        header_field(&clock, "# OF CLK REF", ""),
        ClockHeaderField::ClockRefCount {
            count: 1,
            start: Some(epoch(1994, 7, 14, 0, 0, 0.0)),
            stop: Some(epoch(1994, 7, 14, 20, 59, 0.0)),
        }
    );
    assert_eq!(
        header_field(&clock, "ANALYSIS CLK REF", "TIDB"),
        ClockHeaderField::AnalysisClockRef {
            name: "TIDB".to_string(),
            identifier: "50103M108".to_string(),
            constraint_s: Some(-0.123456789012),
        }
    );
    // Table A15 (3.00): A4,1X,A20,I11,X,I11,X,I11.
    assert_eq!(
        header_field(&clock, "SOLN STA NAME / NUM", "AREQ"),
        ClockHeaderField::SolutionStation {
            name: "AREQ".to_string(),
            identifier: "42202M005".to_string(),
            xyz_mm: [-1234567890, 1234567890, -1234567890],
        }
    );
    assert_eq!(
        header_field(&clock, "# OF SOLN SATS", ""),
        ClockHeaderField::SolutionSatelliteCount(27)
    );
    match header_field(&clock, "PRN LIST", "G01") {
        ClockHeaderField::PrnList(prns) => assert_eq!(prns.len(), 15),
        other => panic!("{other:?}"),
    }

    let records: Vec<ClockRecord> = clock.records().collect();
    assert_eq!(records.len(), 5);
    let areq = &records[0];
    assert_eq!(areq.record_type(), ClockRecordType::Ar);
    assert_eq!(areq.line(), Some(27));
    assert_eq!(areq.line_count(), 2);
    assert_eq!(
        areq.reading(),
        ClockRecordReading::Columns(ClockLayout::V300)
    );
    assert_eq!(
        areq.continuation_reading(),
        Some(ClockRecordReading::Columns(ClockLayout::V300))
    );
    assert_eq!(
        areq.values(),
        &[
            -0.123456789012,
            -1.23456789012,
            -12.3456789012,
            -123.456789012,
            -1234.56789012,
            -12345.6789012
        ]
    );
    let gold = &records[2];
    assert_eq!(
        gold.values(),
        &[
            -0.0123456789012,
            -0.00123456789012,
            -0.000123456789012,
            -0.0000123456789012
        ]
    );
    assert_eq!(clock.series().len(), 1);
    assert_eq!(
        clock.series()["G16"][0].additional_values,
        vec![-0.0123456789012]
    );
    assert_eq!(
        clock.skipped_records(),
        [
            skip(27, "AR"),
            skip(30, "AR"),
            skip(32, "AR"),
            skip(33, "AR")
        ]
    );
}

#[test]
fn spec_304_example_reads_at_the_304_columns() {
    let clock = RinexClock::parse(SPEC_304_A17).expect("3.04 Table A17");
    assert_eq!(clock.version(), Some(3.04));
    assert_eq!(clock.layout(), Some(ClockLayout::V304));
    assert_eq!(clock.time_scale(), Some(TimeScale::Gpst));
    let header = clock.header_records();
    assert!(header.iter().all(|record| record.label_column() == 65));
    // The 3.04 example prints SYS / DCBS APPLIED, SYS / PCVS APPLIED and both
    // # OF CLK REF records in the 3.00 columns, against its own Table A15.
    assert_eq!(
        clock.notices(),
        [
            RinexClockNotice::HeaderRecordNonconforming { line: 9 },
            RinexClockNotice::HeaderRecordNonconforming { line: 10 },
            RinexClockNotice::HeaderRecordNonconforming { line: 13 },
            RinexClockNotice::HeaderRecordNonconforming { line: 15 },
        ]
    );
    assert_eq!(
        header[12].reading(),
        ClockHeaderReading::OtherVersionColumns
    );
    assert_eq!(
        header[12].field(),
        Some(&ClockHeaderField::ClockRefCount {
            count: 1,
            start: Some(epoch(1994, 7, 14, 0, 0, 0.0)),
            stop: Some(epoch(1994, 7, 14, 20, 59, 0.0)),
        })
    );
    // Table A15 (3.04): A9,1X (or A4,6X),A20,I11,1X,I11,1X,I11.
    assert_eq!(
        header_field(&clock, "SOLN STA NAME / NUM", "GOLD"),
        ClockHeaderField::SolutionStation {
            name: "GOLD".to_string(),
            identifier: "40405S031".to_string(),
            xyz_mm: [1234567890, -1234567890, -1234567890],
        }
    );
    assert_eq!(
        header_field(&clock, "ANALYSIS CLK REF", "USNO"),
        ClockHeaderField::AnalysisClockRef {
            name: "USNO".to_string(),
            identifier: "40451S003".to_string(),
            constraint_s: Some(-0.123456789012),
        }
    );
    // 3.04: A1,2X,I3,2X,14(A3,1X): descriptors start in column 9.
    assert_eq!(
        header_field(&clock, "SYS / # / OBS TYPES", "G"),
        ClockHeaderField::ObservationTypes {
            system: Some('G'),
            count: Some(4),
            descriptors: ["C1W", "L1W", "C2W", "L2W"].map(String::from).to_vec(),
        }
    );
    match header_field(&clock, "PRN LIST", "G01") {
        ClockHeaderField::PrnList(prns) => assert_eq!(prns.len(), 16),
        other => panic!("{other:?}"),
    }

    let areq = record_named(&clock, "AREQ00USA");
    assert_eq!(
        areq.reading(),
        ClockRecordReading::Columns(ClockLayout::V304)
    );
    assert_eq!(
        areq.continuation_reading(),
        Some(ClockRecordReading::Columns(ClockLayout::V304))
    );

    // The 3.00 and 3.04 examples state the same records in two layouts.
    let v300 = RinexClock::parse(SPEC_300_A17).expect("3.00 Table A17");
    let pairs: Vec<(ClockRecord, ClockRecord)> = v300.records().zip(clock.records()).collect();
    assert_eq!(pairs.len(), 5);
    for (a, b) in &pairs {
        assert_eq!(a.record_type(), b.record_type());
        assert_eq!(a.civil_epoch(), b.civil_epoch());
        assert_eq!(a.epoch(), b.epoch());
        assert_eq!(a.values(), b.values());
        assert_eq!(a.line(), b.line());
    }
    assert_eq!(v300.series(), clock.series());
}

#[test]
fn spec_304_another_example_reads_leap_seconds_and_nine_character_names() {
    let clock = RinexClock::parse(SPEC_304_ANOTHER).expect("3.04 example");
    assert_eq!(clock.time_system(), Some(ClockTimeSystem::Gps));
    assert_eq!(
        header_field(&clock, "LEAP SECONDS", ""),
        ClockHeaderField::LeapSeconds(37)
    );
    assert_eq!(
        header_field(&clock, "LEAP SECONDS GNSS", ""),
        ClockHeaderField::LeapSecondsGnss(18)
    );
    assert_eq!(
        header_field(&clock, "SOLN STA NAME / NUM", "DGAR00GBR"),
        ClockHeaderField::SolutionStation {
            name: "DGAR00GBR".to_string(),
            identifier: "30802M001".to_string(),
            xyz_mm: [1916268889, 6029977675, -801719507],
        }
    );
    assert_eq!(
        header_field(&clock, "SYS / PCVS APPLIED", "G"),
        ClockHeaderField::PcvsApplied {
            system: "G".to_string(),
            program: String::new(),
            source: "igs14_1935.atx".to_string(),
        }
    );
    assert_eq!(
        clock.notices(),
        [RinexClockNotice::HeaderRecordNonconforming { line: 41 }]
    );
    assert_eq!(record_named(&clock, "DGAR00GBR").line(), Some(45));
    let g01 = &clock.series()["G01"][0];
    assert_eq!(g01.bias_s, 0.175309377613e-8);
    assert_eq!(g01.additional_values, vec![0.183422207046e-10]);
    assert_eq!(
        g01.epoch,
        civil_to_clock_instant(TimeScale::Gpst, 2017, 3, 11, 0, 0, 0.0).unwrap()
    );
}

#[test]
fn spec_304_file_without_time_system_reads_with_the_300_default() {
    // 3.04 requires TIME SYSTEM ID and its own Table A18 example omits it. The
    // file is read with the 3.00 default (GPS for a file that is not pure
    // GLONASS or pure Galileo) and reported.
    let clock = RinexClock::parse(SPEC_304_A18).expect("3.04 Table A18");
    assert_eq!(clock.time_system(), Some(ClockTimeSystem::Gps));
    assert_eq!(
        clock.time_system_status(),
        &ClockTimeSystemStatus::Defaulted
    );
    assert_eq!(clock.time_scale(), Some(TimeScale::Gpst));
    assert_eq!(
        clock.notices(),
        [
            RinexClockNotice::HeaderRecordNonconforming { line: 7 },
            RinexClockNotice::TimeSystemDefaulted {
                system: ClockTimeSystem::Gps
            },
            RinexClockNotice::TimeSystemMissing,
        ]
    );
    let records: Vec<ClockRecord> = clock.records().collect();
    assert_eq!(records.len(), 4);
    assert_eq!(records[2].record_type(), ClockRecordType::Dr);
    assert_eq!(records[2].civil_epoch(), epoch(1995, 7, 14, 22, 23, 14.5));
    assert_eq!(
        records[2].epoch(),
        civil_to_clock_instant(TimeScale::Gpst, 1995, 7, 14, 22, 23, 14.5)
    );
    assert_eq!(records[2].values(), &[-1.23456789012, 0.123456789012]);
    assert!(clock.series().is_empty());

    // Declaring the time system writes one TIME SYSTEM ID record at the 3.04
    // columns, before LEAP SECONDS GNSS as Table A15 orders them, with the
    // file's CRLF terminator; every other line is unchanged.
    let mut declared = clock.clone();
    declared
        .set_time_system(ClockTimeSystem::Gps)
        .expect("declare GPS");
    let tsid = format!("{:<65}TIME SYSTEM ID\r\n", "   GPS");
    let split_at = SPEC_304_A18
        .match_indices("\r\n")
        .nth(3)
        .map(|(index, _)| index + 2)
        .unwrap();
    let expected = format!(
        "{}{}{}",
        &SPEC_304_A18[..split_at],
        tsid,
        &SPEC_304_A18[split_at..]
    );
    assert_eq!(declared.to_rinex_string().unwrap(), expected);
    assert_eq!(
        declared.time_system_status(),
        &ClockTimeSystemStatus::Declared
    );
    let inserted = &declared.header_records()[4];
    assert_eq!(inserted.line(), None);
    assert_eq!(inserted.reading(), ClockHeaderReading::Columns);
    let reparsed = RinexClock::parse(&expected).expect("reparse declared");
    assert_eq!(reparsed.time_scale(), Some(TimeScale::Gpst));
    // The inserted record moves STATION NAME / NUM to line 8.
    assert_eq!(
        reparsed.notices(),
        [RinexClockNotice::HeaderRecordNonconforming { line: 8 }]
    );
}

#[test]
fn spec_300_calibration_example_defaults_to_gps_time() {
    let clock = RinexClock::parse(SPEC_300_A18).expect("3.00 Table A18 (version 2.00)");
    assert_eq!(clock.version(), Some(2.00));
    assert_eq!(clock.layout(), Some(ClockLayout::V300));
    assert_eq!(
        clock.time_system_status(),
        &ClockTimeSystemStatus::Defaulted
    );
    assert_eq!(clock.time_scale(), Some(TimeScale::Gpst));
    assert_eq!(
        clock.notices(),
        [RinexClockNotice::TimeSystemDefaulted {
            system: ClockTimeSystem::Gps
        }]
    );
    assert_eq!(
        header_field(&clock, "STATION NAME / NUM", "USNO"),
        ClockHeaderField::StationNameNum {
            name: "USNO".to_string(),
            identifier: "40451S003".to_string(),
        }
    );
    assert_eq!(
        clock.skipped_records(),
        [
            skip(10, "CR"),
            skip(11, "CR"),
            skip(12, "DR"),
            skip(13, "CR")
        ]
    );
    let dr = clock.records().nth(2).unwrap();
    assert_eq!(
        dr.epoch(),
        civil_to_clock_instant(TimeScale::Gpst, 1994, 7, 14, 22, 23, 14.5)
    );
}

#[test]
fn emr_product_keeps_the_bias_sigma_of_one_value_records() {
    // EMR0OPSRAP declares one value on every AS record and writes the bias
    // sigma in the sigma columns. Each record reads, its declared value list is
    // the bias, and the sigma is kept as a surplus value at position 1.
    let clock = RinexClock::parse(EMR).expect("EMR excerpt");
    assert_eq!(clock.version(), Some(2.00));
    assert_eq!(clock.time_system_status(), &ClockTimeSystemStatus::Declared);
    assert_eq!(clock.record_count(), 153);
    assert_eq!(
        clock.notices(),
        [RinexClockNotice::SurplusValues {
            records: 49,
            first_line: 239,
        }]
    );
    let as_records: Vec<ClockRecord> = clock
        .records()
        .filter(|record| record.record_type() == ClockRecordType::As)
        .collect();
    assert_eq!(as_records.len(), 49);
    assert!(as_records
        .iter()
        .all(|record| record.declared_count() == 1 && record.surplus_values().len() == 1));
    let g01 = &as_records[0];
    assert_eq!(g01.line(), Some(239));
    assert_eq!(g01.bias_s(), 0.170710878415e-3);
    assert_eq!(
        g01.surplus_values(),
        &[ClockSurplusValue {
            position: 1,
            value: 5.556437046250e-12,
        }]
    );
    assert_eq!(clock.series().len(), 49);
    assert!(clock.series()["G01"][0].additional_values.is_empty());
    let ar = record_named(&clock, "NRC1");
    assert_eq!(ar.line(), Some(135));
    assert_eq!(ar.values(), &[-0.121218628367e-4, 0.387783342482e-11]);
}

#[test]
fn esa_302_product_without_time_system_reads_as_gps_time() {
    let clock = RinexClock::parse(ESA).expect("ESA excerpt");
    assert_eq!(clock.version(), Some(3.02));
    assert_eq!(clock.layout(), Some(ClockLayout::V300));
    assert_eq!(clock.satellite_system(), None);
    assert_eq!(clock.time_system(), Some(ClockTimeSystem::Gps));
    assert_eq!(
        clock.time_system_status(),
        &ClockTimeSystemStatus::Defaulted
    );
    // The PCVS program field runs past its 17 columns, and the ANALYSIS CENTER
    // designator is four characters.
    assert_eq!(
        clock.notices(),
        [
            RinexClockNotice::HeaderRecordUninterpreted { line: 4 },
            RinexClockNotice::HeaderRecordUninterpreted { line: 5 },
            RinexClockNotice::HeaderRecordNonconforming { line: 6 },
            RinexClockNotice::TimeSystemDefaulted {
                system: ClockTimeSystem::Gps
            },
        ]
    );
    let header = clock.header_records();
    assert_eq!(header[3].field(), None);
    assert_eq!(
        header[3].text(),
        "G EPNS 1.4.1 11/08/2026     ESA23_2429                      SYS / PCVS APPLIED  "
    );
    assert_eq!(
        header[5].field(),
        Some(&ClockHeaderField::AnalysisCenter {
            designator: "ESOC".to_string(),
            name: "USING EPNS".to_string(),
        })
    );
    let records: Vec<ClockRecord> = clock.records().collect();
    assert_eq!(records.len(), 196);
    let count = |record_type: ClockRecordType| {
        records
            .iter()
            .filter(|record| record.record_type() == record_type)
            .count()
    };
    assert_eq!(count(ClockRecordType::Cr), 7);
    assert_eq!(count(ClockRecordType::Ar), 136);
    assert_eq!(count(ClockRecordType::As), 53);
    assert_eq!(records[0].line(), Some(155));
    assert_eq!(records[0].civil_epoch(), epoch(2026, 9, 17, 0, 0, 0.0));
    assert_eq!(clock.skipped_records()[0], skip(155, "CR"));
    assert_eq!(clock.skipped_records().len(), 143);
}

#[test]
fn gbm_multi_system_product_keeps_every_system_and_name() {
    let clock = RinexClock::parse(GBM).expect("GBM excerpt");
    assert_eq!(clock.time_scale(), Some(TimeScale::Gpst));
    assert_eq!(
        clock.notices(),
        [RinexClockNotice::TimeSystemDefaulted {
            system: ClockTimeSystem::Gps
        }]
    );
    assert_eq!(clock.record_count(), 262);
    assert_eq!(clock.series().len(), 121);
    for (letter, expected) in [('C', 35), ('E', 30), ('G', 31), ('J', 4), ('R', 21)] {
        let found = clock
            .series()
            .keys()
            .filter(|sat| sat.starts_with(letter))
            .count();
        assert_eq!(found, expected, "{letter}");
    }
    let lpgs = record_named(&clock, "lpgs");
    assert_eq!(lpgs.line(), Some(254));
    assert_eq!(lpgs.values(), &[-0.227699702757e-3]);
    assert_eq!(clock.series()["J02"][0].bias_s, -0.232835122007e-5);
}

#[test]
fn cod_product_with_padded_records_reads_at_the_300_columns() {
    let clock = RinexClock::parse(COD).expect("COD excerpt");
    assert!(clock.notices().is_empty(), "{:?}", clock.notices());
    assert_eq!(clock.record_count(), 188);
    assert!(clock
        .records()
        .all(|record| record.reading() == ClockRecordReading::Columns(ClockLayout::V300)));
    let g01 = record_named(&clock, "G01");
    assert_eq!(g01.line(), Some(268));
    assert_eq!(g01.values(), &[0.170710719870e-3, 0.133801203962e-10]);
    assert_eq!(clock.source_line(268).map(str::len), Some(89));
}

#[test]
fn two_ar_records_at_one_epoch_are_both_retained() {
    // RINEX clock section 4: a discontinuity found in analysis is reported as
    // two AR records for one station and epoch with different values.
    for (text, layout) in [
        (
            "     3.00           C                   G                   RINEX VERSION / TYPE
   GPS                                                      TIME SYSTEM ID
                                                            END OF HEADER
AR AREQ 1994 07 14 20 59  0.000000  2    0.100000000000E-03  0.100000000000E-10
AR AREQ 1994 07 14 20 59  0.000000  2    0.200000000000E-03  0.200000000000E-10
AS G16  1994 07 14 20 59  0.000000  1   -0.123456789012E+00
",
            ClockLayout::V300,
        ),
        (
            "3.04                 C                    G                      RINEX VERSION / TYPE
   GPS                                                           TIME SYSTEM ID
                                                                 END OF HEADER
AR AREQ00USA 1994 07 14 20 59  0.000000  2    0.100000000000E-03   0.100000000000E-10
AR AREQ00USA 1994 07 14 20 59  0.000000  2    0.200000000000E-03   0.200000000000E-10
AS G16       1994 07 14 20 59  0.000000  1   -0.123456789012E+00
",
            ClockLayout::V304,
        ),
    ] {
        let clock = RinexClock::parse(text).expect("discontinuity pair");
        assert_eq!(clock.layout(), Some(layout));
        let ar: Vec<ClockRecord> = clock
            .records()
            .filter(|record| record.record_type() == ClockRecordType::Ar)
            .collect();
        assert_eq!(ar.len(), 2);
        assert_eq!(ar[0].civil_epoch(), ar[1].civil_epoch());
        assert_eq!(ar[0].values(), &[1.0e-4, 1.0e-11]);
        assert_eq!(ar[1].values(), &[2.0e-4, 2.0e-11]);
        assert_eq!(
            ar[0].reading(),
            ClockRecordReading::Columns(layout),
            "{text}"
        );
        assert_eq!(clock.skipped_records(), [skip(4, "AR"), skip(5, "AR")]);
        assert_eq!(clock.to_rinex_string().unwrap(), text);
    }
}

#[test]
fn unknown_record_types_and_unreadable_lines_are_retained_and_restated() {
    let text = "     3.00           C                   G                   RINEX VERSION / TYPE
   GPS                                                      TIME SYSTEM ID
                                                            END OF HEADER
AS G01  2026 05 13 00 00  0.000000  1    0.100000000000E-03
XX G01  2026 05 13 00 00  0.000000  1    0.100000000000E-03
IB G01  2026 05 13 00 00  0.000000  3    0.100000000000E-03  0.1E-10
   0.1E-11
free text that is not a record
AR AREQ 2026 05 13 00 00  0.000000  2    0.100000000000E-03  not-a-number

AS G01  2026 05 13 00 00 30.000000  1    0.200000000000E-03
";
    assert_eq!(
        RinexClock::parse(text).unwrap_err(),
        RinexClockError::BadField {
            line: 5,
            field: "record_type",
            value: "XX".to_string(),
        }
    );
    let clock = RinexClock::parse_lossy(text);
    assert_eq!(clock.to_rinex_string().unwrap(), text);
    let diagnostic_lines: Vec<usize> = clock.diagnostics().iter().map(|d| d.line).collect();
    assert_eq!(diagnostic_lines, vec![5, 6, 7, 8, 9]);
    assert_eq!(
        clock.diagnostics()[4].error,
        RinexClockError::BadField {
            line: 9,
            field: "sigma",
            value: "not-a-number".to_string(),
        }
    );
    assert_eq!(clock.record_count(), 2);
    assert_eq!(clock.series()["G01"].len(), 2);
    assert_eq!(clock.source_line(8), Some("free text that is not a record"));

    // Removing a record leaves the retained unreadable lines in place.
    let mut edited = clock.clone();
    let removed = edited
        .remove_record(1)
        .expect("remove the second AS record");
    assert_eq!(removed.line(), Some(11));
    let expected = text.replace(
        "\nAS G01  2026 05 13 00 00 30.000000  1    0.200000000000E-03\n",
        "\n",
    );
    assert_eq!(edited.to_rinex_string().unwrap(), expected);
}

#[test]
fn restating_a_product_keeps_its_header_and_glo_time_system() {
    // The previous writer emitted three header lines and wrote GLO as UTC. A
    // product read from text now restates every header record and the GLO label.
    let text = "     3.00           C                   R                   RINEX VERSION / TYPE
TESTPGM             TESTAGENCY          20260917 000000 UTC PGM / RUN BY / DATE
A COMMENT THAT MUST SURVIVE                                 COMMENT
   GLO                                                      TIME SYSTEM ID
    18                                                      LEAP SECONDS
     2    AR    AS                                          # / TYPES OF DATA
                                                            END OF HEADER
AR ONSA 2026 09 17 00 00  0.000000  1    0.100000000000E-03
AS R01  2026 09 17 00 00  0.000000  1    0.200000000000E-03
AS R01  2026 09 17 00 00 30.000000  1    0.300000000000E-03
";
    let clock = RinexClock::parse(text).expect("GLO 3.00 clock");
    assert_eq!(clock.time_system(), Some(ClockTimeSystem::Glo));
    // GLO epochs have the hours of UTC (RINEX clock 3.00 Table A15, RINEX 3.05
    // section 4.1.2, RTKLIB).
    assert_eq!(clock.time_scale(), Some(TimeScale::Utc));
    assert_eq!(clock.to_rinex_string().unwrap(), text);
    assert!(clock.notices().is_empty(), "{:?}", clock.notices());

    // A 3.04 GLO file is UTC too, so a leap-second label on a leap-second day
    // is an epoch, and the label reads back unchanged.
    let text_304 =
        "3.04                 C                    R                      RINEX VERSION / TYPE
   GLO                                                           TIME SYSTEM ID
                                                                 END OF HEADER
AS R01       2016 12 31 23 59 60.000000  1    0.200000000000E-03
AS R01       2017 01 01 00 00  0.000000  1    0.300000000000E-03
";
    let clock_304 = RinexClock::parse(text_304).expect("GLO 3.04 clock");
    assert_eq!(clock_304.time_scale(), Some(TimeScale::Utc));
    assert_eq!(clock_304.series()["R01"].len(), 2);
    let half = clock_304
        .clock_s("R01", epoch(2016, 12, 31, 23, 59, 60.5))
        .unwrap()
        .expect("bracketed by 23:59:60 and midnight");
    assert!((half - 2.5e-4).abs() < 1.0e-15, "{half}");
    assert_eq!(clock_304.to_rinex_string().unwrap(), text_304);

    // GLONASS system time (UTC(SU) + 3 h) has no RINEX clock label: a product
    // built in it is refused by name, as before.
    let epoch_glonasst =
        civil_to_clock_instant(TimeScale::Glonasst, 2026, 9, 17, 3, 0, 0.0).unwrap();
    let built = RinexClock::from_instant_series_rows(
        TimeScale::Glonasst,
        vec![("R01".to_string(), vec![(epoch_glonasst, 2.0e-4)])],
    )
    .expect("GLONASST rows");
    assert_eq!(
        built.to_rinex_string(),
        Err(RinexClockError::UnsupportedTimeScale {
            scale: TimeScale::Glonasst
        })
    );
}

#[test]
fn beidou_and_navic_time_systems() {
    let text_bds =
        "3.04                 C                    C                      RINEX VERSION / TYPE
   BDS                                                           TIME SYSTEM ID
                                                                 END OF HEADER
AS C01       2026 05 13 00 00  0.000000  1   -0.799538377154E-06
";
    let bds = RinexClock::parse(text_bds).expect("BDS clock");
    assert_eq!(bds.time_system(), Some(ClockTimeSystem::Bds));
    assert_eq!(bds.time_scale(), Some(TimeScale::Bdt));
    let written = RinexClock::from_instant_series_rows(TimeScale::Bdt, bds.instant_series_rows())
        .unwrap()
        .to_rinex_string()
        .unwrap();
    assert!(written.contains(&format!("{:<65}TIME SYSTEM ID\n", "   BDS")));
    assert_eq!(RinexClock::parse(&written).unwrap().series(), bds.series());

    // IRN has no core time scale: records keep civil epochs, no instant is
    // invented, and the file restates exactly.
    let text_irn =
        "3.04                 C                    I                      RINEX VERSION / TYPE
   IRN                                                           TIME SYSTEM ID
                                                                 END OF HEADER
AS I01       2026 05 13 00 00  0.000000  1   -0.232835122007E-05
";
    let irn = RinexClock::parse(text_irn).expect("IRN clock");
    assert_eq!(irn.time_system(), Some(ClockTimeSystem::Irn));
    assert_eq!(irn.time_scale(), None);
    assert_eq!(
        irn.notices(),
        [RinexClockNotice::TimeSystemWithoutScale {
            system: ClockTimeSystem::Irn
        }]
    );
    let record = irn.records().next().unwrap();
    assert_eq!(record.epoch(), None);
    assert_eq!(record.civil_epoch(), epoch(2026, 5, 13, 0, 0, 0.0));
    assert!(irn.series().is_empty());
    assert_eq!(irn.to_rinex_string().unwrap(), text_irn);

    // IRNSS time is continuous: a 23:59:60 label names no epoch in it.
    let text_irn_leap =
        text_irn.replace("2026 05 13 00 00  0.000000", "2016 12 31 23 59 60.000000");
    assert_eq!(
        RinexClock::parse(&text_irn_leap).unwrap_err(),
        RinexClockError::BadField {
            line: 4,
            field: "epoch",
            value: "2016 12 31 23 59 60".to_string(),
        }
    );
}

#[test]
fn unrecognised_time_system_is_not_read_as_gps_time() {
    let text = "     3.00           C                   G                   RINEX VERSION / TYPE
   XYZ                                                      TIME SYSTEM ID
                                                            END OF HEADER
AS G01  2026 05 13 00 00  0.000000  1    0.100000000000E-03
";
    let error = RinexClockError::BadField {
        line: 2,
        field: "time_system",
        value: "XYZ".to_string(),
    };
    assert_eq!(RinexClock::parse(text).unwrap_err(), error);

    let mut clock = RinexClock::parse_lossy(text);
    assert_eq!(
        clock.time_system_status(),
        &ClockTimeSystemStatus::Unrecognized {
            label: "XYZ".to_string()
        }
    );
    assert_eq!(clock.time_scale(), None);
    assert!(clock.series().is_empty());
    assert_eq!(clock.diagnostics()[0].error, error);
    assert_eq!(clock.to_rinex_string().unwrap(), text);

    clock
        .set_time_system(ClockTimeSystem::Gps)
        .expect("declare GPS");
    assert_eq!(clock.time_scale(), Some(TimeScale::Gpst));
    assert_eq!(clock.series()["G01"].len(), 1);
    assert!(clock.diagnostics().is_empty());
    assert_eq!(
        clock.to_rinex_string().unwrap(),
        text.replace("   XYZ", "   GPS")
    );
}

#[test]
fn files_without_time_system_take_the_300_defaults() {
    // RINEX clock 3.00 Table A15: GPS for pure GPS files, GLO for pure GLONASS
    // files, GAL for pure Galileo files; every other file reads in GPS time
    // (Table A16, RTKLIB). The same rule applies to 3.04 files, which are also
    // reported for lacking the record.
    for (system_code, satellite, system, scale) in [
        ('R', "R01", ClockTimeSystem::Glo, TimeScale::Utc),
        ('E', "E11", ClockTimeSystem::Gal, TimeScale::Gst),
        ('G', "G01", ClockTimeSystem::Gps, TimeScale::Gpst),
        ('C', "C01", ClockTimeSystem::Gps, TimeScale::Gpst),
        ('J', "J02", ClockTimeSystem::Gps, TimeScale::Gpst),
        ('I', "I01", ClockTimeSystem::Gps, TimeScale::Gpst),
        ('M', "G01", ClockTimeSystem::Gps, TimeScale::Gpst),
    ] {
        let text = format!(
            "{:<60}RINEX VERSION / TYPE\n{:<60}END OF HEADER\nAS {satellite}  2026 05 13 00 00  0.000000  1    0.100000000000E-03\n",
            format!("     3.00           C                   {system_code}"),
            ""
        );
        let clock = RinexClock::parse(&text).expect("3.00 clock");
        assert_eq!(clock.time_system(), Some(system), "{system_code}");
        assert_eq!(clock.time_scale(), Some(scale), "{system_code}");
        assert_eq!(
            clock.time_system_status(),
            &ClockTimeSystemStatus::Defaulted
        );
        assert_eq!(
            clock.notices(),
            [RinexClockNotice::TimeSystemDefaulted { system }]
        );

        let text_304 = format!(
            "{:<65}RINEX VERSION / TYPE\n{:<65}END OF HEADER\nAS {satellite}       2026 05 13 00 00  0.000000  1    0.100000000000E-03\n",
            format!("3.04                 C                    {system_code}"),
            ""
        );
        let clock_304 = RinexClock::parse(&text_304).expect("3.04 clock");
        assert_eq!(clock_304.time_system(), Some(system), "{system_code}");
        assert_eq!(clock_304.time_scale(), Some(scale), "{system_code}");
        assert_eq!(
            clock_304.notices(),
            [
                RinexClockNotice::TimeSystemDefaulted { system },
                RinexClockNotice::TimeSystemMissing,
            ]
        );
        assert_eq!(clock_304.series()[satellite].len(), 1);
    }
}

#[test]
fn records_in_the_other_layout_are_read_and_reported() {
    // A 3.00 file with one record written in the 3.04 columns.
    let text = "     3.00           C                   G                   RINEX VERSION / TYPE
   GPS                                                      TIME SYSTEM ID
                                                            END OF HEADER
AS G01  2026 05 13 00 00  0.000000  1    0.100000000000E-03
AS G02       2026 05 13 00 00  0.000000  1    0.200000000000E-03
";
    let clock = RinexClock::parse(text).expect("mixed layouts");
    let readings: Vec<ClockRecordReading> = clock.records().map(|r| r.reading()).collect();
    assert_eq!(
        readings,
        vec![
            ClockRecordReading::Columns(ClockLayout::V300),
            ClockRecordReading::Columns(ClockLayout::V304),
        ]
    );
    assert_eq!(
        clock.notices(),
        [RinexClockNotice::OtherLayoutRecords {
            records: 1,
            first_line: 5,
        }]
    );
    assert_eq!(clock.series()["G02"][0].bias_s, 2.0e-4);
    assert_eq!(clock.to_rinex_string().unwrap(), text);
}

#[test]
fn value_edits_replace_the_record_lines_and_nothing_else() {
    let mut clock = RinexClock::parse(SPEC_300_A17).expect("3.00 Table A17");
    clock
        .set_record_values(1, vec![1.0e-4, 2.0e-5])
        .expect("edit G16");
    let expected = SPEC_300_A17.replace(
        "AS G16  1994 07 14 20 59  0.000000  2    -.123456789012E+00  -.123456789012E-01\n",
        "AS G16  1994 07 14 20 59  0.000000  2    0.100000000000E-03  0.200000000000E-04\n",
    );
    assert_ne!(expected, SPEC_300_A17);
    assert_eq!(clock.to_rinex_string().unwrap(), expected);
    let g16 = clock.records().nth(1).unwrap();
    assert_eq!(g16.reading(), ClockRecordReading::Edited);
    assert_eq!(g16.line(), None);
    assert_eq!(clock.series()["G16"][0].bias_s, 1.0e-4);
    assert_eq!(clock.series()["G16"][0].additional_values, vec![2.0e-5]);
    assert_eq!(
        clock.skipped_records(),
        [
            skip(27, "AR"),
            skip(30, "AR"),
            skip(32, "AR"),
            skip(33, "AR")
        ]
    );

    // Removing a record removes its continuation line with it.
    let removed = clock.remove_record(0).expect("remove AREQ");
    assert_eq!(removed.name(), "AREQ");
    assert_eq!(removed.declared_count(), 6);
    let expected = expected.replace(
        "AR AREQ 1994 07 14 20 59  0.000000  6   -0.123456789012E+00 -0.123456789012E+01\n\
-0.123456789012E+02 -0.123456789012E+03 -0.123456789012E+04 -0.123456789012E+05\n",
        "",
    );
    assert_eq!(clock.to_rinex_string().unwrap(), expected);
    assert_eq!(
        clock.skipped_records(),
        [skip(30, "AR"), skip(32, "AR"), skip(33, "AR")]
    );

    // An edit the writer would refuse is refused, and nothing changes: the
    // product can never become unwritable through an accepted edit.
    let before = clock.clone();
    assert_eq!(
        clock.set_record_values(0, vec![1.23456789012345e-4]),
        Err(RinexClockError::InvalidInput {
            field: "bias",
            reason:
                "value cannot be represented in Fortran E19.12 format without loss of precision",
        })
    );
    assert_eq!(clock, before);
    assert_eq!(clock.to_rinex_string().unwrap(), expected);
    assert_eq!(
        clock.set_record_values(0, vec![f64::NAN]),
        Err(RinexClockError::InvalidInput {
            field: "bias_s",
            reason: "must be finite",
        })
    );
    assert_eq!(
        clock.set_record_values(99, vec![1.0]),
        Err(RinexClockError::InvalidInput {
            field: "index",
            reason: "no record at this index",
        })
    );
}

#[test]
fn value_edits_keep_or_refuse_surplus_values() {
    // Every EMR AS record declares one value and carries a sigma. Replacing the
    // bias alone would drop the sigma, so it is refused; restating the sigma as
    // a declared value is accepted and written as a two-value record.
    let mut clock = RinexClock::parse(EMR).expect("EMR excerpt");
    let g01 = clock
        .records()
        .position(|record| record.name() == "G01")
        .unwrap();
    let before = clock.clone();
    assert_eq!(
        clock.set_record_values(g01, vec![1.0e-4]),
        Err(RinexClockError::InvalidInput {
            field: "values",
            reason:
                "the record carries values beyond its declared count; the new values must restate them",
        })
    );
    assert_eq!(clock, before);
    clock
        .set_record_values(g01, vec![0.170710878415e-3, 5.556437046250e-12])
        .expect("restating the sigma");
    let written = clock.to_rinex_string().unwrap();
    assert!(written.contains(
        "\nAS G01  2026 09 17 00 00  0.000000  2    0.170710878415E-03  0.555643704625E-11\n"
    ));
    assert_eq!(
        clock.series()["G01"][0].additional_values,
        vec![5.556437046250e-12]
    );
    assert_eq!(
        clock.notices(),
        [RinexClockNotice::SurplusValues {
            records: 48,
            first_line: 240,
        }]
    );
}

#[test]
fn edited_304_records_follow_the_sigma_spacing_of_their_product() {
    // The Table A17 example writes the sigma after two blanks; the IGS
    // combination example after one. An edited record follows its product.
    let mut two = RinexClock::parse(SPEC_304_A17).expect("3.04 Table A17");
    two.set_record_values(1, vec![1.0e-4, 2.0e-5]).unwrap();
    assert!(two.to_rinex_string().unwrap().contains(
        "\r\nAS G16       1994 07 14 20 59  0.000000  2    0.100000000000E-03   0.200000000000E-04\r\n"
    ));
    let mut one = RinexClock::parse(SPEC_304_ANOTHER).expect("3.04 example");
    let g01 = one
        .records()
        .position(|record| record.name() == "G01")
        .unwrap();
    one.set_record_values(g01, vec![1.0e-4, 2.0e-5]).unwrap();
    assert!(one.to_rinex_string().unwrap().contains(
        "\r\nAS G01       2017 03 11 00 00  0.000000  2    0.100000000000E-03  0.200000000000E-04\r\n"
    ));
}

#[test]
fn record_by_record_edits_agree_with_a_full_read() {
    // Build a product one record at a time, with duplicates, out-of-order
    // epochs, removals and value edits, and compare every derived view with a
    // fresh read of the text it writes.
    let mut clock = RinexClock::parse(
        "     3.00           C                   G                   RINEX VERSION / TYPE
   GPS                                                      TIME SYSTEM ID
                                                            END OF HEADER
",
    )
    .expect("empty product");
    let record = |name: &str, minute: u8, bias: f64| {
        ClockRecord::new(
            if name.len() == 3 {
                ClockRecordType::As
            } else {
                ClockRecordType::Ar
            },
            name,
            epoch(2026, 5, 13, 0, minute, 0.0),
            vec![bias],
        )
        .unwrap()
    };
    for minute in 0..40u8 {
        let count = clock.record_count();
        clock
            .insert_record(count, record("G01", minute, f64::from(minute)))
            .unwrap();
    }
    clock.insert_record(0, record("G01", 50, 50.0)).unwrap();
    clock.insert_record(3, record("G01", 5, 105.0)).unwrap();
    clock.insert_record(10, record("G01", 5, 205.0)).unwrap();
    clock.insert_record(7, record("AREQ", 5, 1.0)).unwrap();
    clock.insert_record(7, record("G02", 5, 2.0)).unwrap();
    assert_eq!(clock.record_count(), 45);
    // The last G01 record at minute 5 in file order is the sample.
    let at_five = |clock: &RinexClock| {
        clock
            .clock_s("G01", epoch(2026, 5, 13, 0, 5, 0.0))
            .unwrap()
            .unwrap()
    };
    assert_eq!(at_five(&clock), 205.0);
    let removed = clock.remove_record(12).unwrap();
    assert_eq!(removed.bias_s(), 205.0);
    assert_eq!(at_five(&clock), 5.0);
    clock.set_record_values(3, vec![305.0]).unwrap();
    assert_eq!(at_five(&clock), 5.0);
    let shown = clock
        .records()
        .position(|record| record.bias_s() == 5.0)
        .unwrap();
    let removed = clock.remove_record(shown).unwrap();
    assert_eq!(removed.civil_epoch(), epoch(2026, 5, 13, 0, 5, 0.0));
    assert_eq!(at_five(&clock), 305.0);

    let text = clock.to_rinex_string().unwrap();
    let reread = RinexClock::parse(&text).unwrap();
    assert_eq!(reread.series(), clock.series());
    assert_eq!(reread.record_count(), clock.record_count());
    let biases = |clock: &RinexClock| clock.records().map(|r| r.bias_s()).collect::<Vec<_>>();
    assert_eq!(biases(&reread), biases(&clock));
    assert_eq!(clock.series()["G01"].len(), 41);
    assert_eq!(clock.series()["G02"].len(), 1);
}

#[test]
fn inserted_records_are_written_in_the_file_layout() {
    let mut clock = RinexClock::parse(SPEC_304_A17).expect("3.04 Table A17");
    let record = ClockRecord::new(
        ClockRecordType::Ar,
        "AREQ00USA",
        epoch(1994, 7, 14, 21, 0, 0.0),
        vec![0.5],
    )
    .expect("new AR record");
    clock.insert_record(5, record).expect("append");
    let expected = format!(
        "{SPEC_304_A17}AR AREQ00USA 1994 07 14 21 00  0.000000  1    0.500000000000E+00\r\n"
    );
    assert_eq!(clock.to_rinex_string().unwrap(), expected);
    assert_eq!(clock.record_count(), 6);

    // A nine-character name does not fit the 3.00 layout.
    let mut v300 = RinexClock::parse(SPEC_300_A17).expect("3.00 Table A17");
    let record = ClockRecord::new(
        ClockRecordType::Ar,
        "AREQ00USA",
        epoch(1994, 7, 14, 21, 0, 0.0),
        vec![0.5],
    )
    .unwrap();
    assert_eq!(
        v300.insert_record(0, record),
        Err(RinexClockError::InvalidInput {
            field: "name",
            reason: "wider than the name field of the layout",
        })
    );
    assert_eq!(v300.to_rinex_string().unwrap(), SPEC_300_A17);

    // A satellite record goes into the series.
    let record = ClockRecord::new(
        ClockRecordType::As,
        "G05",
        epoch(1994, 7, 14, 20, 59, 0.0),
        vec![0.25, 0.5],
    )
    .unwrap();
    v300.insert_record(1, record).expect("insert G05");
    assert_eq!(v300.series()["G05"][0].bias_s, 0.25);
    assert!(v300.to_rinex_string().unwrap().contains(
        "\nAS G05  1994 07 14 20 59  0.000000  2    0.250000000000E+00  0.500000000000E+00\nAS G16"
    ));
}

#[test]
fn a_time_system_change_that_invalidates_an_epoch_changes_nothing() {
    let text = "     3.00           C                   G                   RINEX VERSION / TYPE
   UTC                                                      TIME SYSTEM ID
                                                            END OF HEADER
AS G05  2016 12 31 23 59 60.000000  1    0.100000000000E-03
";
    let mut clock = RinexClock::parse(text).expect("UTC leap-second clock");
    assert_eq!(
        clock.set_time_system(ClockTimeSystem::Gps),
        Err(RinexClockError::BadField {
            line: 4,
            field: "epoch",
            value: "2016 12 31 23 59 60".to_string(),
        })
    );
    assert_eq!(clock.time_scale(), Some(TimeScale::Utc));
    assert_eq!(clock.to_rinex_string().unwrap(), text);
    assert_eq!(clock.series()["G05"].len(), 1);
}

#[test]
fn a_utc_leap_second_label_is_queryable() {
    let text = "     3.00           C                   G                   RINEX VERSION / TYPE
   UTC                                                      TIME SYSTEM ID
                                                            END OF HEADER
AS G05  2016 12 31 23 59 30.000000  1    0.000000000000E+00
AS G05  2016 12 31 23 59 60.000000  1    0.300000000000E+02
AS G05  2017 01 01 00 00  0.000000  1    0.310000000000E+02
";
    let clock = RinexClock::parse(text).expect("UTC clock");
    assert_eq!(
        clock
            .clock_s("G05", epoch(2016, 12, 31, 23, 59, 60.0))
            .unwrap(),
        Some(30.0)
    );
    let half = clock
        .clock_s("G05", epoch(2016, 12, 31, 23, 59, 60.5))
        .unwrap()
        .unwrap();
    assert!((half - 30.5).abs() < 1.0e-9, "{half}");
    let before = clock
        .clock_s("G05", epoch(2016, 12, 31, 23, 59, 45.0))
        .unwrap()
        .unwrap();
    assert!((before - 15.0).abs() < 1.0e-9, "{before}");
    // A GPST product has no 23:59:60.
    let gps = RinexClock::parse(&text.replace("   UTC", "   GPS").replace(
        "AS G05  2016 12 31 23 59 60.000000  1    0.300000000000E+02\n",
        "",
    ))
    .unwrap();
    assert_eq!(
        gps.clock_s("G05", epoch(2016, 12, 31, 23, 59, 60.0)),
        Err(RinexClockError::InvalidInput {
            field: "epoch",
            reason: "invalid civil clock epoch",
        })
    );
}
