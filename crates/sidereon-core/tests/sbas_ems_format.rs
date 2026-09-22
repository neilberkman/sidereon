//! Blank-separated EMS data records.
//!
//! The record layout is specified by the EGNOS Message Server User Interface
//! Document, E-RD-SYS-E31-011-ESA Issue 2 Revision 0 (2004-11-26), section 3,
//! page 14, figures 3 and 4: blank characters separate the PRN, the two-digit
//! year, month, day, hour, minute and second, the message type, and the
//! message in hexadecimal. Issue 2 Revision 0 placed the time stamp on GPS
//! time; earlier issues used UTC.
//!
//! Provenance: `fixtures/sbas_ems/glab_ems_format_example.ems` holds the three
//! records displayed on the gLAB EGNOS v2.0 file-description page, copied byte
//! for byte. It is an authored example of the format rather than an
//! independently acquired archive file; `glab_ems_format_example_provenance.json`
//! records that distinction and the source digests.
//!
//! The expected week and seconds of week below were computed from the printed
//! calendar fields against the Sunday 1980-01-06 GPS week origin, and the
//! expected bytes were converted from the hexadecimal text displayed on that
//! page. Neither came from the reader under test.

use sidereon_core::astro::time::{GnssWeekTow, TimeScale};
use sidereon_core::sbas::{parse_ems_lines, SbasWireForm};

const GLAB_EXAMPLE: &str = include_str!("fixtures/sbas_ems/glab_ems_format_example.ems");

/// `120 18 03 31 23 59 58 4 C611C003FC0003FF40000000000000000000000000039797BB80000017CA1640`
const RECORD_0: [u8; 32] = [
    0xC6, 0x11, 0xC0, 0x03, 0xFC, 0x00, 0x03, 0xFF, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x97, 0x97, 0xBB, 0x80, 0x00, 0x00, 0x17, 0xCA, 0x16, 0x40,
];
/// `120 18 03 31 23 59 59 4 A91EEE7E7EE7777EEEE777E777777EEEE77EEE700000000000000000234C75C0`
const RECORD_1: [u8; 32] = [
    0xA9, 0x1E, 0xEE, 0x7E, 0x7E, 0xE7, 0x77, 0x7E, 0xEE, 0xE7, 0x77, 0xE7, 0x77, 0x77, 0x7E, 0xEE,
    0xE7, 0x7E, 0xEE, 0x70, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x23, 0x4C, 0x75, 0xC0,
];
/// `120 18 04 01 00 00 00 4 C60AC000000003FCC003FFC000013FF8000000000003BB97BB9BBBBB805CAC40`
const RECORD_2: [u8; 32] = [
    0xC6, 0x0A, 0xC0, 0x00, 0x00, 0x00, 0x03, 0xFC, 0xC0, 0x03, 0xFF, 0xC0, 0x00, 0x01, 0x3F, 0xF8,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xBB, 0x97, 0xBB, 0x9B, 0xBB, 0xBB, 0x80, 0x5C, 0xAC, 0x40,
];

/// The three records cross midnight and the GPS week rollover from Saturday
/// 2018-03-31 into Sunday 2018-04-01, so week 1994 seconds 604798 and 604799
/// are followed by week 1995 second 0.
#[test]
fn glab_example_records_parse_across_the_gps_week_boundary() {
    let expected: [(u32, f64, &[u8]); 3] = [
        (1994, 604_798.0, &RECORD_0),
        (1994, 604_799.0, &RECORD_1),
        (1995, 0.0, &RECORD_2),
    ];

    let parsed = parse_ems_lines(GLAB_EXAMPLE).expect("gLAB EMS format example parses");
    assert_eq!(
        parsed.len(),
        expected.len(),
        "blank-separated EMS records must be retained, not skipped as non-record lines"
    );

    for (block, (week, tow_s, bytes)) in parsed.iter().zip(expected) {
        let epoch = GnssWeekTow::new(TimeScale::Gpst, week, tow_s).expect("valid GPST epoch");
        assert_eq!(block.satellite_id.to_string(), "S20");
        assert_eq!(block.epoch, epoch);
        assert_eq!(block.form, SbasWireForm::Framed250);
        assert_eq!(block.bytes.len(), 32);
        assert_eq!(block.bytes, bytes);
    }
}
