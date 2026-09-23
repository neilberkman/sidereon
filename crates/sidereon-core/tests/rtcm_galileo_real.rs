use std::path::PathBuf;

use sidereon_core::constants::{GPS_EPOCH_TO_J2000_S, SECONDS_PER_WEEK};
use sidereon_core::ephemeris::{BroadcastEphemeris, BroadcastRecord, EphemerisSource, Sp3};
use sidereon_core::rtcm::{decode_messages, Message};
use sidereon_core::{GnssSatelliteId, GnssSystem};

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

fn load_rtcm_frames() -> Vec<Vec<u8>> {
    let text = std::fs::read_to_string(fixture_path(
        "rtcm/SSRA00EUH0_20260708_1402_1046_frames.hex",
    ))
    .expect("read RTCM 1046 fixture");
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(hex_bytes)
        .collect()
}

fn load_sp3() -> Sp3 {
    let bytes = std::fs::read(fixture_path(
        "sp3/GRG0OPSULT_20261880600_02D_05M_ORB_E03_E10_1300_1430.SP3",
    ))
    .expect("read Galileo SP3 trim");
    Sp3::parse(&bytes).expect("parse Galileo SP3 trim")
}

fn toe_j2000_s(record: &BroadcastRecord) -> f64 {
    f64::from(record.week) * SECONDS_PER_WEEK + record.elements.toe_sow - GPS_EPOCH_TO_J2000_S
}

/// The comparison epoch: one second after `toe`. The store serves a Galileo record only
/// after its `toe`, as RTKLIB `seleph` skips a record whose age of data is not positive.
fn query_j2000_s(record: &BroadcastRecord) -> f64 {
    toe_j2000_s(record) + 1.0
}

fn position_error_m(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[test]
fn real_galileo_1046_decodes_and_propagates_against_sp3() {
    let frames = load_rtcm_frames();
    let mut records = Vec::new();
    let mut raw_messages = Vec::new();

    for frame in &frames {
        let messages = decode_messages(frame).expect("fixture frame decodes in full");
        assert_eq!(messages.len(), 1, "each fixture frame carries one message");
        let Message::GalileoInavEphemeris(eph) = &messages[0] else {
            panic!("expected Galileo I/NAV 1046");
        };
        assert_eq!(messages[0].to_frame().expect("re-frame 1046"), *frame);
        assert!(
            eph.sqrt_a > 2_800_000_000 && eph.eccentricity > 0,
            "decoded orbital fields must be non-trivial"
        );
        raw_messages.push(eph.clone());
        records.push(eph.to_broadcast_record().expect("1046 to broadcast record"));
    }

    assert_eq!(records.len(), 8);
    assert_eq!(
        records
            .iter()
            .map(|record| record.satellite_id)
            .collect::<Vec<_>>(),
        (3..=10)
            .map(|prn| GnssSatelliteId::new(GnssSystem::Galileo, prn).unwrap())
            .collect::<Vec<_>>()
    );

    let broadcast = BroadcastEphemeris::new(records.clone()).expect("build broadcast store");
    let sp3 = load_sp3();

    let mut max_error_m = 0.0_f64;
    for record in &records {
        let query = query_j2000_s(record);
        let (broadcast_position, _) = broadcast
            .position_clock_at_j2000_s(record.satellite_id, query)
            .expect("broadcast state after Galileo toe");
        let precise = sp3
            .position_at_j2000_seconds(record.satellite_id, query)
            .expect("SP3 state after Galileo toe");
        let error_m = position_error_m(broadcast_position, precise.position.as_array());
        max_error_m = max_error_m.max(error_m);
        assert!(
            error_m < 100.0,
            "{} broadcast-vs-SP3 error {error_m:.3} m exceeds loose ceiling",
            record.satellite_id
        );
    }

    let mut corrupted = raw_messages[0].clone();
    corrupted.sqrt_a += 50_000_000;
    let bad_record = corrupted
        .to_broadcast_record()
        .expect("corrupted 1046 still maps to a record");
    let bad_broadcast = BroadcastEphemeris::new(vec![bad_record]).expect("build corrupted store");
    let query = query_j2000_s(&bad_record);
    let (bad_position, _) = bad_broadcast
        .position_clock_at_j2000_s(bad_record.satellite_id, query)
        .expect("corrupted broadcast state after toe");
    let precise = sp3
        .position_at_j2000_seconds(bad_record.satellite_id, query)
        .expect("SP3 state for corrupted comparison");
    let corrupted_error_m = position_error_m(bad_position, precise.position.as_array());
    assert!(
        corrupted_error_m > 100_000.0,
        "corrupting sqrtA should break the SP3 oracle, got {corrupted_error_m:.3} m"
    );

    eprintln!("max Galileo 1046 broadcast-vs-SP3 error: {max_error_m:.3} m");
}
