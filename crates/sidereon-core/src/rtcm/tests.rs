//! Unit tests for the RTCM 3 codec: CRC vectors, frame sync, and IR round-trips.

use super::bits::{BitReader, BitWriter};
use super::crc::{crc24q, crc24q_with_init};
use super::*;

// ---------------------------------------------------------------------------
// Bit codec
// ---------------------------------------------------------------------------

#[test]
fn bit_reader_writer_unsigned_round_trip() {
    let mut w = BitWriter::new();
    w.push_u(0b101, 3);
    w.push_u(0x3FF, 10);
    w.push_u(0xFFFF_FFFF_FFFF_FFFF, 64);
    let bytes = w.into_bytes();

    let mut r = BitReader::new(&bytes);
    assert_eq!(r.u(3).unwrap(), 0b101);
    assert_eq!(r.u(10).unwrap(), 0x3FF);
    assert_eq!(r.u(64).unwrap(), 0xFFFF_FFFF_FFFF_FFFF);
}

#[test]
fn bit_reader_writer_twos_complement() {
    let mut w = BitWriter::new();
    w.push_i(-1, 8);
    w.push_i(-2048, 12);
    w.push_i(2047, 12);
    let bytes = w.into_bytes();

    let mut r = BitReader::new(&bytes);
    assert_eq!(r.i(8).unwrap(), -1);
    assert_eq!(r.i(12).unwrap(), -2048);
    assert_eq!(r.i(12).unwrap(), 2047);
}

#[test]
fn bit_reader_writer_sign_magnitude() {
    let mut w = BitWriter::new();
    w.push_ism(-5, 5); // sign + 4-bit magnitude
    w.push_ism(5, 5);
    w.push_ism(0, 11);
    w.push_ism(-1023, 11);
    let bytes = w.into_bytes();

    let mut r = BitReader::new(&bytes);
    assert_eq!(r.ism(5).unwrap(), -5);
    assert_eq!(r.ism(5).unwrap(), 5);
    assert_eq!(r.ism(11).unwrap(), 0);
    assert_eq!(r.ism(11).unwrap(), -1023);
}

#[test]
fn bit_reader_reports_truncation() {
    let bytes = [0xFFu8];
    let mut r = BitReader::new(&bytes);
    assert!(r.u(9).is_err());
}

// ---------------------------------------------------------------------------
// CRC-24Q
// ---------------------------------------------------------------------------

#[test]
fn crc24q_algorithm_matches_published_openpgp_vector() {
    // Same polynomial and bit mechanics as RTCM's CRC-24Q; starting the
    // register at 0xB704CE reproduces the published CRC-24/OPENPGP check value
    // over "123456789", which anchors the polynomial and bit order.
    assert_eq!(crc24q_with_init(0xB704CE, b"123456789"), 0x0021_CF02);
}

#[test]
fn crc24q_rtcm_init_zero_empty_is_zero() {
    assert_eq!(crc24q(b""), 0);
}

// ---------------------------------------------------------------------------
// Framing
// ---------------------------------------------------------------------------

#[test]
fn frame_round_trip_and_crc_check() {
    let body = [0x43u8, 0x10, 0x00, 0xAB, 0xCD];
    let frame = encode_frame(&body).unwrap();
    assert_eq!(frame[0], PREAMBLE);
    assert_eq!(frame.len(), body.len() + FRAME_OVERHEAD);

    let decoded = decode_frame(&frame).unwrap();
    assert_eq!(decoded.body, &body);
    assert_eq!(decoded.frame_len, frame.len());
}

#[test]
fn frame_detects_corruption() {
    let body = [0x12u8, 0x34, 0x56];
    let mut frame = encode_frame(&body).unwrap();
    // Flip a payload bit; the CRC must now fail.
    frame[4] ^= 0x01;
    assert!(decode_frame(&frame).is_err());
}

#[test]
fn frame_rejects_oversize_body() {
    let body = vec![0u8; MAX_BODY_LEN + 1];
    assert!(encode_frame(&body).is_err());
}

#[test]
fn scanner_resyncs_past_junk_and_partial_frames() {
    let a = encode_frame(&[0x01, 0x02, 0x03]).unwrap();
    let b = encode_frame(&[0x09, 0x08]).unwrap();

    let mut stream = Vec::new();
    stream.extend_from_slice(&[0x00, 0xD3, 0x99]); // a stray 0xD3 that is not a frame
    stream.extend_from_slice(&a);
    stream.extend_from_slice(&[0xAA, 0xBB]); // junk between frames
    stream.extend_from_slice(&b);

    let frames: Vec<_> = FrameScanner::new(&stream).collect();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].body, &[0x01, 0x02, 0x03]);
    assert_eq!(frames[1].body, &[0x09, 0x08]);
}

// ---------------------------------------------------------------------------
// Station coordinates 1005 / 1006
// ---------------------------------------------------------------------------

fn sample_station(message_number: u16, height: Option<u16>) -> StationCoordinates {
    StationCoordinates {
        message_number,
        reference_station_id: 2003,
        itrf_realization_year: 21,
        gps_indicator: true,
        glonass_indicator: true,
        galileo_indicator: false,
        reference_station_indicator: true,
        ecef_x: 11_446_021_400,
        single_receiver_oscillator: false,
        reserved: false,
        ecef_y: -7_415_136_500,
        quarter_cycle_indicator: 2,
        ecef_z: 12_602_528_900,
        antenna_height: height,
    }
}

#[test]
fn station_1005_round_trip() {
    let station = sample_station(1005, None);
    let body = station.encode();
    // 1005 body is exactly 19 bytes (152 bits).
    assert_eq!(body.len(), 19);
    assert_eq!(message_number(&body).unwrap(), 1005);
    let decoded = StationCoordinates::decode(&body).unwrap();
    assert_eq!(decoded, station);
}

#[test]
fn station_1006_round_trip_and_meters() {
    let station = sample_station(1006, Some(15_000));
    let body = station.encode();
    // 1006 body is exactly 21 bytes (168 bits).
    assert_eq!(body.len(), 21);
    let decoded = StationCoordinates::decode(&body).unwrap();
    assert_eq!(decoded, station);

    assert!((decoded.x_m() - 1_144_602.14).abs() < 1e-6);
    assert!((decoded.z_m() - 1_260_252.89).abs() < 1e-6);
    assert!((decoded.antenna_height_m().unwrap() - 1.5).abs() < 1e-9);
}

#[test]
fn station_full_frame_round_trip() {
    let message = Message::StationCoordinates(sample_station(1006, Some(0)));
    let frame = message.to_frame().unwrap();
    let decoded = decode_messages(&frame);
    assert_eq!(decoded, vec![message]);
}

// ---------------------------------------------------------------------------
// Antenna / receiver descriptors 1007 / 1008 / 1033
// ---------------------------------------------------------------------------

#[test]
fn antenna_1007_round_trip() {
    let descriptor = AntennaDescriptor {
        message_number: 1007,
        reference_station_id: 100,
        antenna_descriptor: "TRM59800.00     NONE".to_string(),
        antenna_setup_id: 1,
        antenna_serial_number: None,
        receiver_type: None,
        receiver_firmware_version: None,
        receiver_serial_number: None,
    };
    let body = descriptor.encode();
    assert_eq!(AntennaDescriptor::decode(&body).unwrap(), descriptor);
}

#[test]
fn antenna_1008_round_trip() {
    let descriptor = AntennaDescriptor {
        message_number: 1008,
        reference_station_id: 100,
        antenna_descriptor: "ASH701945C_M    SCIS".to_string(),
        antenna_setup_id: 3,
        antenna_serial_number: Some("CR12345".to_string()),
        receiver_type: None,
        receiver_firmware_version: None,
        receiver_serial_number: None,
    };
    let body = descriptor.encode();
    assert_eq!(AntennaDescriptor::decode(&body).unwrap(), descriptor);
}

#[test]
fn antenna_1033_round_trip() {
    let descriptor = AntennaDescriptor {
        message_number: 1033,
        reference_station_id: 4095,
        antenna_descriptor: "LEIAR25.R4      LEIT".to_string(),
        antenna_setup_id: 0,
        antenna_serial_number: Some("09120119".to_string()),
        receiver_type: Some("LEICA GR50".to_string()),
        receiver_firmware_version: Some("4.50".to_string()),
        receiver_serial_number: Some("1830080".to_string()),
    };
    let frame = Message::AntennaDescriptor(descriptor.clone())
        .to_frame()
        .unwrap();
    assert_eq!(
        decode_messages(&frame),
        vec![Message::AntennaDescriptor(descriptor)]
    );
}

// ---------------------------------------------------------------------------
// GPS ephemeris 1019
// ---------------------------------------------------------------------------

#[test]
fn gps_ephemeris_1019_round_trip() {
    let eph = GpsEphemeris {
        satellite_id: 14,
        week_number: 1023,
        sv_accuracy: 0,
        code_on_l2: 1,
        idot: -1234,
        iode: 42,
        t_oc: 30_000,
        a_f2: 0,
        a_f1: -7,
        a_f0: 123_456,
        iodc: 42,
        c_rs: -8000,
        delta_n: 4500,
        m0: -1_073_741_824,
        c_uc: -512,
        eccentricity: 21_000_000,
        c_us: 600,
        sqrt_a: 2_705_000_000,
        t_oe: 30_000,
        c_ic: -10,
        omega0: 1_000_000_000,
        c_is: 12,
        i0: 600_000_000,
        c_rc: 7000,
        omega: -900_000_000,
        omega_dot: -2000,
        t_gd: -5,
        sv_health: 0,
        l2_p_data_flag: false,
        fit_interval: true,
    };
    let body = eph.encode();
    // 1019 body is exactly 61 bytes (488 bits).
    assert_eq!(body.len(), 61);
    let decoded = GpsEphemeris::decode(&body).unwrap();
    assert_eq!(decoded, eph);
    assert_eq!(
        decoded.satellite().unwrap(),
        crate::id::GnssSatelliteId::new(crate::id::GnssSystem::Gps, 14).unwrap()
    );
}

// ---------------------------------------------------------------------------
// GLONASS ephemeris 1020 (sign-magnitude fields)
// ---------------------------------------------------------------------------

#[test]
fn glonass_ephemeris_1020_round_trip() {
    let eph = GlonassEphemeris {
        satellite_id: 7,
        frequency_channel: 4,
        almanac_health: true,
        almanac_health_availability: true,
        p1: 1,
        t_k: 1234,
        b_n_msb: false,
        p2: true,
        t_b: 76,
        xn_dot: -123_456,
        xn: 67_108_000,
        xn_dot_dot: -3,
        yn_dot: 7777,
        yn: -67_000_000,
        yn_dot_dot: 2,
        zn_dot: -1,
        zn: 12_345_678,
        zn_dot_dot: -1,
        p3: true,
        gamma_n: -1000,
        m_p: 2,
        m_l_n_third: false,
        tau_n: -2_000_000,
        delta_tau_n: 4,
        e_n: 10,
        m_p4: true,
        m_f_t: 6,
        m_n_t: 1500,
        m_m: 1,
        additional_data_available: true,
        n_a: 700,
        tau_c: -1_500_000_000,
        m_n4: 5,
        m_tau_gps: -987_654,
        m_l_n_fifth: false,
        reserved: 0,
    };
    let body = eph.encode();
    // 1020 body is exactly 45 bytes (360 bits).
    assert_eq!(body.len(), 45);
    let decoded = GlonassEphemeris::decode(&body).unwrap();
    assert_eq!(decoded, eph);
}

// ---------------------------------------------------------------------------
// MSM observations
// ---------------------------------------------------------------------------

fn msm_header() -> MsmHeader {
    MsmHeader {
        reference_station_id: 0,
        epoch_time: 86_400_000,
        multiple_message: false,
        iods: 0,
        reserved: 0,
        clock_steering: 0,
        external_clock: 0,
        divergence_free_smoothing: false,
        smoothing_interval: 0,
    }
}

#[test]
fn msm4_gps_round_trip() {
    // Satellites G03, G14, G22; G22 carries no signal cells (mask bit set,
    // every cell absent) to exercise the empty-row path.
    let satellites = vec![
        MsmSatellite {
            id: 3,
            rough_range_ms: 75,
            rough_range_mod1: 512,
            extended_info: None,
            rough_phase_range_rate_m_s: None,
        },
        MsmSatellite {
            id: 14,
            rough_range_ms: 80,
            rough_range_mod1: 1000,
            extended_info: None,
            rough_phase_range_rate_m_s: None,
        },
        MsmSatellite {
            id: 22,
            rough_range_ms: 255,
            rough_range_mod1: 0,
            extended_info: None,
            rough_phase_range_rate_m_s: None,
        },
    ];
    // Signals 2 and 15; cells for (G03,s2), (G03,s15), (G14,s2).
    let signals = vec![
        MsmSignal {
            satellite_id: 3,
            signal_id: 2,
            fine_pseudorange: -4000,
            fine_phase_range: 100_000,
            lock_time_indicator: 9,
            half_cycle_ambiguity: false,
            cnr: 45,
            fine_phase_range_rate: None,
        },
        MsmSignal {
            satellite_id: 3,
            signal_id: 15,
            fine_pseudorange: 4000,
            fine_phase_range: -100_000,
            lock_time_indicator: 3,
            half_cycle_ambiguity: true,
            cnr: 38,
            fine_phase_range_rate: None,
        },
        MsmSignal {
            satellite_id: 14,
            signal_id: 2,
            fine_pseudorange: 16,
            fine_phase_range: -7,
            lock_time_indicator: 15,
            half_cycle_ambiguity: false,
            cnr: 50,
            fine_phase_range_rate: None,
        },
    ];
    let message = MsmMessage {
        message_number: 1074,
        system: crate::id::GnssSystem::Gps,
        kind: MsmKind::Msm4,
        header: msm_header(),
        satellites,
        signals,
    };
    let body = message.encode();
    let decoded = MsmMessage::decode(&body).unwrap();
    assert_eq!(decoded, message);
}

#[test]
fn msm7_glonass_round_trip_with_extended_info() {
    let satellites = vec![
        MsmSatellite {
            id: 1,
            rough_range_ms: 70,
            rough_range_mod1: 256,
            extended_info: Some(8),
            rough_phase_range_rate_m_s: Some(-1500),
        },
        MsmSatellite {
            id: 9,
            rough_range_ms: 90,
            rough_range_mod1: 900,
            extended_info: Some(2),
            rough_phase_range_rate_m_s: Some(3000),
        },
    ];
    let signals = vec![
        MsmSignal {
            satellite_id: 1,
            signal_id: 2,
            fine_pseudorange: -500_000,
            fine_phase_range: 8_000_000,
            lock_time_indicator: 700,
            half_cycle_ambiguity: false,
            cnr: 800,
            fine_phase_range_rate: Some(-12_000),
        },
        MsmSignal {
            satellite_id: 9,
            signal_id: 2,
            fine_pseudorange: 500_000,
            fine_phase_range: -8_000_000,
            lock_time_indicator: 1,
            half_cycle_ambiguity: true,
            cnr: 640,
            fine_phase_range_rate: Some(16_000),
        },
    ];
    let message = MsmMessage {
        message_number: 1087,
        system: crate::id::GnssSystem::Glonass,
        kind: MsmKind::Msm7,
        header: msm_header(),
        satellites,
        signals,
    };
    let frame = Message::Msm(message.clone()).to_frame().unwrap();
    let decoded = decode_messages(&frame);
    assert_eq!(decoded, vec![Message::Msm(message)]);
}

#[test]
fn msm_kind_maps_constellation_and_type() {
    use crate::id::GnssSystem::*;
    let cases = [
        (1074, Gps, MsmKind::Msm4),
        (1077, Gps, MsmKind::Msm7),
        (1084, Glonass, MsmKind::Msm4),
        (1087, Glonass, MsmKind::Msm7),
        (1094, Galileo, MsmKind::Msm4),
        (1097, Galileo, MsmKind::Msm7),
        (1124, BeiDou, MsmKind::Msm4),
        (1127, BeiDou, MsmKind::Msm7),
    ];
    for (num, sys, kind) in cases {
        let m = MsmMessage {
            message_number: num,
            system: sys,
            kind,
            header: msm_header(),
            satellites: Vec::new(),
            signals: Vec::new(),
        };
        let body = m.encode();
        let decoded = MsmMessage::decode(&body).unwrap();
        assert_eq!(decoded.system, sys);
        assert_eq!(decoded.kind, kind);
        assert_eq!(decoded.message_number, num);
    }
}

// ---------------------------------------------------------------------------
// Unsupported messages and dispatch
// ---------------------------------------------------------------------------

#[test]
fn unsupported_message_round_trips_verbatim() {
    // Message 1230 (GLONASS code-phase biases) is not decoded; build a body
    // whose first 12 bits are 1230 and check it survives a frame round-trip.
    let mut w = BitWriter::new();
    w.push_u(1230, 12);
    w.push_u(0xABCD, 16);
    let body = w.into_bytes();

    let message = Message::decode(&body).unwrap();
    match &message {
        Message::Unsupported(u) => assert_eq!(u.message_number, 1230),
        _ => panic!("expected Unsupported"),
    }
    assert_eq!(message.encode(), body);
    assert_eq!(message.message_number(), 1230);

    let frame = message.to_frame().unwrap();
    assert_eq!(decode_messages(&frame), vec![message]);
}

#[test]
fn multiple_messages_in_one_stream() {
    let station = Message::StationCoordinates(sample_station(1005, None));
    let eph = Message::GpsEphemeris(GpsEphemeris {
        satellite_id: 1,
        week_number: 100,
        sv_accuracy: 0,
        code_on_l2: 0,
        idot: 0,
        iode: 0,
        t_oc: 0,
        a_f2: 0,
        a_f1: 0,
        a_f0: 0,
        iodc: 0,
        c_rs: 0,
        delta_n: 0,
        m0: 0,
        c_uc: 0,
        eccentricity: 0,
        c_us: 0,
        sqrt_a: 0,
        t_oe: 0,
        c_ic: 0,
        omega0: 0,
        c_is: 0,
        i0: 0,
        c_rc: 0,
        omega: 0,
        omega_dot: 0,
        t_gd: 0,
        sv_health: 0,
        l2_p_data_flag: false,
        fit_interval: false,
    });

    let mut stream = station.to_frame().unwrap();
    stream.extend_from_slice(&eph.to_frame().unwrap());

    assert_eq!(decode_messages(&stream), vec![station, eph]);
}

// ---------------------------------------------------------------------------
// Public construction + encode path (the binding-facing API)
//
// These build each supported message from scratch through the public structs
// and the public per-type and `Message`-level encode entry points, decode the
// result back, and assert both field-for-field IR equality and byte-for-byte
// round-trip equality at the body and frame levels.
// ---------------------------------------------------------------------------

/// Encode `message`, decode it back, and assert field-for-field IR equality plus
/// byte round-trip equality on both the body and the full transport frame.
fn assert_round_trips(message: Message) {
    // Body: build -> encode -> decode is field-for-field identical, and
    // re-encoding the decoded value reproduces the body bytes.
    let body = message.encode();
    let decoded = Message::decode(&body).unwrap();
    assert_eq!(decoded, message, "decoded IR must equal the constructed IR");
    assert_eq!(decoded.encode(), body, "re-encode must be byte-identical");
    assert_eq!(decoded.message_number(), message.message_number());

    // Frame: wrapping, scanning, and re-framing all round-trip byte-for-byte.
    let frame = message.to_frame().unwrap();
    let scanned = decode_messages(&frame);
    assert_eq!(scanned, vec![message.clone()]);
    assert_eq!(scanned[0].to_frame().unwrap(), frame);
}

#[test]
fn build_station_from_scratch_round_trips() {
    for (number, height) in [(1005u16, None), (1006u16, Some(15_000u16))] {
        let station = sample_station(number, height);
        // Exercise the public per-type encode/decode directly.
        let body = station.encode();
        let decoded = StationCoordinates::decode(&body).unwrap();
        assert_eq!(decoded, station);
        assert_eq!(decoded.encode(), body);
        // And the same value through the `Message` wrapper.
        assert_round_trips(Message::StationCoordinates(station));
    }
}

#[test]
fn build_antenna_from_scratch_round_trips() {
    let descriptors = [
        AntennaDescriptor {
            message_number: 1007,
            reference_station_id: 100,
            antenna_descriptor: "TRM59800.00     NONE".to_string(),
            antenna_setup_id: 1,
            antenna_serial_number: None,
            receiver_type: None,
            receiver_firmware_version: None,
            receiver_serial_number: None,
        },
        AntennaDescriptor {
            message_number: 1008,
            reference_station_id: 200,
            antenna_descriptor: "ASH701945C_M    SCIS".to_string(),
            antenna_setup_id: 3,
            antenna_serial_number: Some("CR12345".to_string()),
            receiver_type: None,
            receiver_firmware_version: None,
            receiver_serial_number: None,
        },
        AntennaDescriptor {
            message_number: 1033,
            reference_station_id: 4095,
            antenna_descriptor: "LEIAR25.R4      LEIT".to_string(),
            antenna_setup_id: 0,
            antenna_serial_number: Some("09120119".to_string()),
            receiver_type: Some("LEICA GR50".to_string()),
            receiver_firmware_version: Some("4.50".to_string()),
            receiver_serial_number: Some("1830080".to_string()),
        },
    ];
    for descriptor in descriptors {
        let body = descriptor.encode();
        let decoded = AntennaDescriptor::decode(&body).unwrap();
        assert_eq!(decoded, descriptor);
        assert_eq!(decoded.encode(), body);
        assert_round_trips(Message::AntennaDescriptor(descriptor));
    }
}

#[test]
fn build_gps_ephemeris_from_scratch_round_trips() {
    let eph = GpsEphemeris {
        satellite_id: 14,
        week_number: 1023,
        sv_accuracy: 0,
        code_on_l2: 1,
        idot: -1234,
        iode: 42,
        t_oc: 30_000,
        a_f2: 0,
        a_f1: -7,
        a_f0: 123_456,
        iodc: 42,
        c_rs: -8000,
        delta_n: 4500,
        m0: -1_073_741_824,
        c_uc: -512,
        eccentricity: 21_000_000,
        c_us: 600,
        sqrt_a: 2_705_000_000,
        t_oe: 30_000,
        c_ic: -10,
        omega0: 1_000_000_000,
        c_is: 12,
        i0: 600_000_000,
        c_rc: 7000,
        omega: -900_000_000,
        omega_dot: -2000,
        t_gd: -5,
        sv_health: 0,
        l2_p_data_flag: false,
        fit_interval: true,
    };
    let body = eph.encode();
    let decoded = GpsEphemeris::decode(&body).unwrap();
    assert_eq!(decoded, eph);
    assert_eq!(decoded.encode(), body);
    assert_round_trips(Message::GpsEphemeris(eph));
}

#[test]
fn build_glonass_ephemeris_from_scratch_round_trips() {
    let eph = GlonassEphemeris {
        satellite_id: 7,
        frequency_channel: 4,
        almanac_health: true,
        almanac_health_availability: true,
        p1: 1,
        t_k: 1234,
        b_n_msb: false,
        p2: true,
        t_b: 76,
        xn_dot: -123_456,
        xn: 67_108_000,
        xn_dot_dot: -3,
        yn_dot: 7777,
        yn: -67_000_000,
        yn_dot_dot: 2,
        zn_dot: -1,
        zn: 12_345_678,
        zn_dot_dot: -1,
        p3: true,
        gamma_n: -1000,
        m_p: 2,
        m_l_n_third: false,
        tau_n: -2_000_000,
        delta_tau_n: 4,
        e_n: 10,
        m_p4: true,
        m_f_t: 6,
        m_n_t: 1500,
        m_m: 1,
        additional_data_available: true,
        n_a: 700,
        tau_c: -1_500_000_000,
        m_n4: 5,
        m_tau_gps: -987_654,
        m_l_n_fifth: false,
        reserved: 0,
    };
    let body = eph.encode();
    let decoded = GlonassEphemeris::decode(&body).unwrap();
    assert_eq!(decoded, eph);
    assert_eq!(decoded.encode(), body);
    assert_round_trips(Message::GlonassEphemeris(eph));
}

#[test]
fn build_msm4_from_scratch_round_trips() {
    let satellites = vec![
        MsmSatellite {
            id: 3,
            rough_range_ms: 75,
            rough_range_mod1: 512,
            extended_info: None,
            rough_phase_range_rate_m_s: None,
        },
        MsmSatellite {
            id: 14,
            rough_range_ms: 80,
            rough_range_mod1: 1000,
            extended_info: None,
            rough_phase_range_rate_m_s: None,
        },
    ];
    let signals = vec![
        MsmSignal {
            satellite_id: 3,
            signal_id: 2,
            fine_pseudorange: -4000,
            fine_phase_range: 100_000,
            lock_time_indicator: 9,
            half_cycle_ambiguity: false,
            cnr: 45,
            fine_phase_range_rate: None,
        },
        MsmSignal {
            satellite_id: 14,
            signal_id: 2,
            fine_pseudorange: 16,
            fine_phase_range: -7,
            lock_time_indicator: 15,
            half_cycle_ambiguity: true,
            cnr: 50,
            fine_phase_range_rate: None,
        },
    ];
    let message = MsmMessage {
        message_number: 1074,
        system: crate::id::GnssSystem::Gps,
        kind: MsmKind::Msm4,
        header: msm_header(),
        satellites,
        signals,
    };
    let body = message.encode();
    let decoded = MsmMessage::decode(&body).unwrap();
    assert_eq!(decoded, message);
    assert_eq!(decoded.encode(), body);
    assert_round_trips(Message::Msm(message));
}

#[test]
fn build_msm7_from_scratch_round_trips() {
    let satellites = vec![MsmSatellite {
        id: 1,
        rough_range_ms: 70,
        rough_range_mod1: 256,
        extended_info: Some(8),
        rough_phase_range_rate_m_s: Some(-1500),
    }];
    let signals = vec![MsmSignal {
        satellite_id: 1,
        signal_id: 2,
        fine_pseudorange: -500_000,
        fine_phase_range: 8_000_000,
        lock_time_indicator: 700,
        half_cycle_ambiguity: false,
        cnr: 800,
        fine_phase_range_rate: Some(-12_000),
    }];
    let message = MsmMessage {
        message_number: 1077,
        system: crate::id::GnssSystem::Gps,
        kind: MsmKind::Msm7,
        header: msm_header(),
        satellites,
        signals,
    };
    let body = message.encode();
    let decoded = MsmMessage::decode(&body).unwrap();
    assert_eq!(decoded, message);
    assert_eq!(decoded.encode(), body);
    assert_round_trips(Message::Msm(message));
}

#[test]
fn message_enum_is_matched_exhaustively_without_wildcard() {
    // With `#[non_exhaustive]` removed, a caller can match every variant
    // without a catch-all arm; this compiles only while the set is exhaustive.
    let message = Message::StationCoordinates(sample_station(1005, None));
    let number = match &message {
        Message::Msm(m) => m.message_number,
        Message::StationCoordinates(s) => s.message_number,
        Message::AntennaDescriptor(a) => a.message_number,
        Message::GpsEphemeris(_) => 1019,
        Message::GlonassEphemeris(_) => 1020,
        Message::Unsupported(u) => u.message_number,
    };
    assert_eq!(number, 1005);
}
