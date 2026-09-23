use std::path::PathBuf;

use sidereon_core::constants::{
    BDS_EPOCH_MINUS_GPS_EPOCH_S, GPST_MINUS_BDT_S, GPS_EPOCH_TO_J2000_S, SECONDS_PER_WEEK,
};
use sidereon_core::ephemeris::{BroadcastEphemeris, BroadcastRecord, EphemerisSource, Sp3};
use sidereon_core::rtcm::{decode_messages, BeidouEphemeris, GalileoFnavEphemeris, Message};
use sidereon_core::GnssSystem;

const GPS_WEEK_AT_BCEP_CAPTURE: u32 = 2426;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn hex_bytes(line: &str) -> Vec<u8> {
    assert_eq!(line.len() % 2, 0, "hex line must have whole bytes");
    (0..line.len())
        .step_by(2)
        .map(|idx| u8::from_str_radix(&line[idx..idx + 2], 16).expect("valid hex byte"))
        .collect()
}

fn load_rtcm_frames(name: &str) -> Vec<Vec<u8>> {
    let text = std::fs::read_to_string(fixture_path(name)).expect("read RTCM fixture");
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(hex_bytes)
        .collect()
}

fn load_sp3(name: &str) -> Sp3 {
    let bytes = std::fs::read(fixture_path(name)).expect("read SP3 fixture");
    Sp3::parse(&bytes).expect("parse SP3 trim")
}

fn toe_j2000_s(record: &BroadcastRecord) -> f64 {
    let continuous = f64::from(record.week) * SECONDS_PER_WEEK + record.elements.toe_sow;
    if record.satellite_id.system == GnssSystem::BeiDou {
        continuous + BDS_EPOCH_MINUS_GPS_EPOCH_S + GPST_MINUS_BDT_S - GPS_EPOCH_TO_J2000_S
    } else {
        continuous - GPS_EPOCH_TO_J2000_S
    }
}

/// The comparison epoch: `toe`, or one second after it for Galileo, since the store
/// serves a Galileo record only after its `toe`, as RTKLIB `seleph` skips a record
/// whose age of data is not positive.
fn query_j2000_s(record: &BroadcastRecord) -> f64 {
    if record.satellite_id.system == GnssSystem::Galileo {
        toe_j2000_s(record) + 1.0
    } else {
        toe_j2000_s(record)
    }
}

fn position_error_m(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn assert_sp3_agreement(records: Vec<BroadcastRecord>, sp3: &Sp3, ceiling_m: f64) -> (usize, f64) {
    let broadcast = BroadcastEphemeris::new(records.clone()).expect("build broadcast store");
    let mut max_error_m = 0.0_f64;
    for record in &records {
        let query = query_j2000_s(record);
        let (broadcast_position, _) = broadcast
            .position_clock_at_j2000_s(record.satellite_id, query)
            .expect("broadcast state at toe");
        let precise = sp3
            .position_at_j2000_seconds(record.satellite_id, query)
            .expect("SP3 state at toe");
        let error_m = position_error_m(broadcast_position, precise.position.as_array());
        max_error_m = max_error_m.max(error_m);
        assert!(
            error_m < ceiling_m,
            "{} broadcast-vs-SP3 error {error_m:.3} m exceeds {ceiling_m:.1} m ceiling",
            record.satellite_id
        );
    }
    (records.len(), max_error_m)
}

fn assert_corrupted_breaks_sp3(
    record: BroadcastRecord,
    sp3: &Sp3,
    ceiling_m: f64,
    min_break_m: f64,
) {
    let bad_broadcast = BroadcastEphemeris::new(vec![record]).expect("build corrupted store");
    let query = query_j2000_s(&record);
    let (bad_position, _) = bad_broadcast
        .position_clock_at_j2000_s(record.satellite_id, query)
        .expect("corrupted broadcast state at toe");
    let precise = sp3
        .position_at_j2000_seconds(record.satellite_id, query)
        .expect("SP3 state for corrupted comparison");
    let corrupted_error_m = position_error_m(bad_position, precise.position.as_array());
    assert!(
        corrupted_error_m > min_break_m.max(ceiling_m),
        "corrupted sqrtA should break the SP3 oracle, got {corrupted_error_m:.3} m"
    );
}

#[test]
fn real_beidou_1042_decodes_and_propagates_against_sp3() {
    let frames = load_rtcm_frames("rtcm/BCEP00BKG0_20260708_1636_1042_frames.hex");
    let mut raw_messages = Vec::new();
    let mut records = Vec::new();

    for frame in &frames {
        let messages = decode_messages(frame);
        assert_eq!(messages.len(), 1, "each fixture frame carries one message");
        let Message::BeidouEphemeris(eph) = messages[0] else {
            panic!("expected BeiDou 1042");
        };
        assert_eq!(messages[0].to_frame().expect("re-frame 1042"), *frame);
        assert!(
            eph.sqrt_a > 2_700_000_000 && eph.eccentricity > 0,
            "decoded orbital fields must be non-trivial"
        );
        raw_messages.push(eph);
        records.push(eph.to_broadcast_record().expect("1042 to broadcast record"));
    }

    assert!(
        records
            .iter()
            .all(|record| record.satellite_id.system == GnssSystem::BeiDou),
        "fixture should contain only BeiDou ephemerides"
    );

    let sp3 = load_sp3("sp3/GRG0OPSULT_20261880600_02D_05M_ORB_C19_C23_E02_E05_1500_1730.SP3");
    // BeiDou broadcast D1/D2 orbits should be close to the same-day GRG
    // ultra-rapid SP3. The 100 m loose ceiling matches broadcast-ephemeris
    // accuracy expectations while still allowing product differences.
    let (count, max_error_m) = assert_sp3_agreement(records, &sp3, 100.0);

    let mut corrupted: BeidouEphemeris = raw_messages[0];
    corrupted.sqrt_a += 50_000_000;
    assert_corrupted_breaks_sp3(
        corrupted
            .to_broadcast_record()
            .expect("corrupted 1042 still maps to a record"),
        &sp3,
        100.0,
        100_000.0,
    );

    eprintln!("BeiDou 1042 SP3 validation: {count} records, max error {max_error_m:.3} m");
}

#[test]
fn real_qzss_1044_decodes_and_propagates_against_sp3() {
    let frames = load_rtcm_frames("rtcm/BCEP00BKG0_20260708_1636_1044_frames.hex");
    let mut raw_messages = Vec::new();
    let mut records = Vec::new();

    for frame in &frames {
        let messages = decode_messages(frame);
        assert_eq!(messages.len(), 1, "each fixture frame carries one message");
        let Message::QzssEphemeris(eph) = messages[0] else {
            panic!("expected QZSS 1044");
        };
        assert_eq!(messages[0].to_frame().expect("re-frame 1044"), *frame);
        assert!(
            eph.sqrt_a > 2_700_000_000 && eph.eccentricity > 0,
            "decoded orbital fields must be non-trivial"
        );
        raw_messages.push(eph);
        records.push(
            eph.to_broadcast_record(GPS_WEEK_AT_BCEP_CAPTURE)
                .expect("1044 to broadcast record"),
        );
    }

    assert!(
        records
            .iter()
            .all(|record| record.satellite_id.system == GnssSystem::Qzss),
        "fixture should contain only QZSS ephemerides"
    );

    let sp3 = load_sp3("sp3/qzu24263_06_J02_J04_J08_1500_1900.sp3");
    // QZSS is compared against the official QZU 15-minute ultra-rapid SP3. The
    // 100 m loose ceiling is wide for broadcast ephemeris but catches scale
    // mistakes, including the t_oe conversion exercised by this capture.
    let (count, max_error_m) = assert_sp3_agreement(records, &sp3, 100.0);

    let mut corrupted = raw_messages[0];
    corrupted.sqrt_a += 50_000_000;
    assert_corrupted_breaks_sp3(
        corrupted
            .to_broadcast_record(GPS_WEEK_AT_BCEP_CAPTURE)
            .expect("corrupted 1044 still maps to a record"),
        &sp3,
        100.0,
        100_000.0,
    );

    eprintln!("QZSS 1044 SP3 validation: {count} records, max error {max_error_m:.3} m");
}

#[test]
fn real_galileo_fnav_1045_decodes_and_propagates_against_sp3() {
    let frames = load_rtcm_frames("rtcm/BCEP00BKG0_20260708_1636_1045_frames.hex");
    let mut raw_messages = Vec::new();
    let mut records = Vec::new();

    for frame in &frames {
        let messages = decode_messages(frame);
        assert_eq!(messages.len(), 1, "each fixture frame carries one message");
        let Message::GalileoFnavEphemeris(eph) = messages[0] else {
            panic!("expected Galileo F/NAV 1045");
        };
        assert_eq!(messages[0].to_frame().expect("re-frame 1045"), *frame);
        assert!(
            eph.sqrt_a > 2_800_000_000 && eph.eccentricity > 0,
            "decoded orbital fields must be non-trivial"
        );
        raw_messages.push(eph);
        records.push(eph.to_broadcast_record().expect("1045 to broadcast record"));
    }

    assert!(
        records
            .iter()
            .all(|record| record.satellite_id.system == GnssSystem::Galileo),
        "fixture should contain only Galileo ephemerides"
    );
    let sp3 = load_sp3("sp3/GRG0OPSULT_20261880600_02D_05M_ORB_C19_C23_E02_E05_1500_1730.SP3");
    // Galileo F/NAV uses the same broadcast dynamics as I/NAV and is compared
    // against GRG ultra-rapid SP3 with the same 100 m loose ceiling as 1046.
    let (count, max_error_m) = assert_sp3_agreement(records, &sp3, 100.0);

    let mut corrupted: GalileoFnavEphemeris = raw_messages[0];
    corrupted.sqrt_a += 50_000_000;
    assert_corrupted_breaks_sp3(
        corrupted
            .to_broadcast_record()
            .expect("corrupted 1045 still maps to a record"),
        &sp3,
        100.0,
        100_000.0,
    );

    eprintln!("Galileo F/NAV 1045 SP3 validation: {count} records, max error {max_error_m:.3} m");
}
