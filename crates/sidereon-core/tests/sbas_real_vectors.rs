//! Provenance: real SBAS bodies captured in RTKLIB
//! `test/data/rcvraw/gw10_20110121.sbas`.
//! RTKLIB `src/rcv/gw10.c` copies receiver message id `0x03` bytes 7 through 35
//! into `sbsmsg.msg[0..28]`. Expected decoded integers are pinned against the
//! offsets in RTKLIB `src/sbas.c` decode_sbstype2, decode_sbstype9,
//! decode_sbstype18, decode_sbstype25, and decode_sbstype26.
//!
//! The capture used here did not contain MT24. MT24 is covered by an
//! RTKLIB-offset unit test in the SBAS message codec.

use sidereon_core::sbas::{SbasBlock, SbasLongTermRecord, SbasMessage, SbasWireForm};

fn body(hex: &str) -> Vec<u8> {
    assert!(hex.len().is_multiple_of(2));
    (0..hex.len())
        .step_by(2)
        .map(|idx| u8::from_str_radix(&hex[idx..idx + 2], 16).expect("hex byte"))
        .collect()
}

fn decode(hex: &str) -> SbasMessage {
    SbasBlock::decode(&body(hex), SbasWireForm::Body226)
        .expect("captured SBAS body decodes")
        .message
}

fn active_positions(mask: &[bool; 201]) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .filter_map(|(idx, active)| active.then_some(idx))
        .collect()
}

fn assert_velocity_record(
    record: &SbasLongTermRecord,
    expected: (u8, u8, [i32; 3], [i32; 3], i32, i32, u32),
) {
    let (monitored_index, iode, delta_pos, delta_rate, delta_a_f0, delta_a_f1, tod) = expected;
    assert_eq!(record.monitored_index, monitored_index);
    assert_eq!(record.iode, iode);
    assert_eq!([record.delta_x, record.delta_y, record.delta_z], delta_pos);
    assert_eq!(
        [
            record.delta_x_rate,
            record.delta_y_rate,
            record.delta_z_rate,
        ],
        delta_rate
    );
    assert_eq!(record.delta_a_f0, delta_a_f0);
    assert_eq!(record.delta_a_f1, delta_a_f1);
    assert_eq!(record.time_of_day_s, Some(tod));
}

#[test]
fn captured_mt2_fast_corrections_match_rtklib_offsets() {
    let msg = decode("5308DFFC010005FFC00DFFC009FFDFFC001FFDFFDFFFBABBBBBB9BBB80");
    let SbasMessage::FastCorrections(fast) = msg else {
        panic!("expected MT2 fast corrections");
    };
    assert_eq!(fast.preamble, 0x53);
    assert_eq!(fast.message_type, 2);
    assert_eq!(fast.iodf, 0);
    assert_eq!(fast.iodp, 3);
    assert_eq!(
        fast.prc,
        [2047, 4, 1, 2047, 3, 2047, 2, 2047, 2047, 0, 2047, 2047, 2047]
    );
    assert_eq!(
        fast.udrei,
        [14, 14, 10, 14, 14, 14, 14, 14, 14, 6, 14, 14, 14]
    );
}

#[test]
fn captured_mt9_geo_navigation_match_rtklib_offsets() {
    let msg = decode("9A25C80C8D3F574632853C69A015EEBFF2D7DF580018FE3FCFF79C38C0");
    let SbasMessage::GeoNav(geo) = msg else {
        panic!("expected MT9 GEO navigation");
    };
    assert_eq!(geo.preamble, 0x9A);
    assert_eq!(geo.reserved.0.first(), Some(&(114, 8)));
    assert_eq!(geo.time_of_day_s, 100);
    assert_eq!(geo.ura, 6);
    assert_eq!(
        [geo.x_m, geo.y_m, geo.z_m],
        [-404_035_386, 338_289_485, 89_835]
    );
    assert_eq!(
        [geo.x_rate_m_s, geo.y_rate_m_s, geo.z_rate_m_s],
        [-422, -2090, 24]
    );
    assert_eq!(
        [geo.x_accel_m_s2, geo.y_accel_m_s2, geo.z_accel_m_s2,],
        [-8, -4, -3]
    );
    assert_eq!([geo.a_gf0_s, geo.a_gf1_s_s], [-400, -29]);
}

#[test]
fn captured_mt18_igp_mask_match_rtklib_offsets() {
    let msg = decode("5348DF0000000000FC0000FFC0007FF0003FFC001FFC0007FF8003FF80");
    let SbasMessage::IgpMask(mask) = msg else {
        panic!("expected MT18 IGP mask");
    };
    assert_eq!(mask.preamble, 0x53);
    assert_eq!(mask.reserved.0.first(), Some(&(3, 4)));
    assert_eq!(mask.band_number, 7);
    assert_eq!(mask.iodi, 3);
    let active = active_positions(&mask.mask);
    assert_eq!(active.len(), 73);
    assert_eq!(&active[..10], [40, 41, 42, 43, 44, 45, 64, 65, 66, 67]);
}

#[test]
fn captured_mt25_long_term_corrections_match_rtklib_offsets() {
    let msg = decode("5366819010029EE7ED83018202819BBE1A08BF8008FFA00000004066C0");
    let SbasMessage::LongTermCorrections(long) = msg else {
        panic!("expected MT25 long-term corrections");
    };
    assert!(long.halves[0].velocity_code);
    assert_eq!(long.halves[0].iodp, 3);
    assert_velocity_record(
        &long.halves[0].records[0],
        (16, 50, [16, 20, -71], [6, 3, 4], -37, 5, 102),
    );
    assert!(long.halves[1].velocity_code);
    assert_eq!(long.halves[1].iodp, 3);
    assert_velocity_record(
        &long.halves[1].records[0],
        (31, 13, [34, -16, 8], [0, 0, 0], -3, 2, 102),
    );
}

#[test]
fn captured_mt26_iono_delays_match_rtklib_offsets() {
    let msg = decode("9A680053E21F17F897C000000000000000000000000000000000006000");
    let SbasMessage::IonoDelays(iono) = msg else {
        panic!("expected MT26 ionosphere delays");
    };
    assert_eq!(iono.preamble, 0x9A);
    assert_eq!(iono.band_number, 0);
    assert_eq!(iono.block_id, 0);
    assert_eq!(iono.iodi, 3);
    assert_eq!(iono.entries[0].vertical_delay, 41);
    assert_eq!(iono.entries[0].givei, 15);
    assert_eq!(iono.entries[1].vertical_delay, 33);
    assert_eq!(iono.entries[1].givei, 15);
    assert_eq!(iono.entries[2].vertical_delay, 47);
    assert_eq!(iono.entries[2].givei, 15);
}

/// Every captured body re-encodes to the 29 bytes it was decoded from,
/// velocity-code long-term halves and reserved bits included.
#[test]
fn captured_bodies_restate_byte_for_byte() {
    for hex in [
        "5308DFFC010005FFC00DFFC009FFDFFC001FFDFFDFFFBABBBBBB9BBB80",
        "9A25C80C8D3F574632853C69A015EEBFF2D7DF580018FE3FCFF79C38C0",
        "5348DF0000000000FC0000FFC0007FF0003FFC001FFC0007FF8003FF80",
        "5366819010029EE7ED83018202819BBE1A08BF8008FFA00000004066C0",
        "9A680053E21F17F897C000000000000000000000000000000000006000",
    ] {
        let bytes = body(hex);
        let block = SbasBlock::decode(&bytes, SbasWireForm::Body226).expect("decode");
        assert_eq!(block.encode().expect("re-encode"), bytes, "{hex}");
    }
}
