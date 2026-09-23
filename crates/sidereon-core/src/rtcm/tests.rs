//! Unit tests for the RTCM 3 codec: CRC vectors, frame sync, and IR round-trips.

use super::bits::{BitReader, BitWriter};
use super::crc::{crc24q, crc24q_with_init};
use super::*;
use crate::error::Error;

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
    let body = eph.encode().unwrap();
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
    let body = eph.encode().unwrap();
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
    let body = message.encode().unwrap();
    let decoded = MsmMessage::decode(&body).unwrap();
    assert_eq!(decoded, message);
}

/// The MSM satellite mask holds ids 1..=64 and the signal mask ids 1..=32. An
/// id outside those, a satellite or cell listed twice, or a signal whose
/// satellite is not listed cannot be stated in the masks, so the encoder
/// refuses it by name instead of shifting it onto another bit or dropping it.
#[test]
fn msm_encode_refuses_satellite_and_signal_lists_its_masks_cannot_state() {
    let satellite = |id| MsmSatellite {
        id,
        rough_range_ms: 70,
        rough_range_mod1: 256,
        extended_info: None,
        rough_phase_range_rate_m_s: None,
    };
    let signal = |satellite_id, signal_id| MsmSignal {
        satellite_id,
        signal_id,
        fine_pseudorange: 16,
        fine_phase_range: -7,
        lock_time_indicator: 15,
        half_cycle_ambiguity: false,
        cnr: 50,
        fine_phase_range_rate: None,
    };
    let message = |satellites, signals| MsmMessage {
        message_number: 1074,
        system: crate::id::GnssSystem::Gps,
        kind: MsmKind::Msm4,
        header: msm_header(),
        satellites,
        signals,
    };

    // The edges of both masks encode and decode.
    let edges = message(
        vec![satellite(1), satellite(64)],
        vec![signal(1, 1), signal(64, 32)],
    );
    let body = edges.encode().expect("ids 1 and 64, signals 1 and 32 fit");
    assert_eq!(MsmMessage::decode(&body).unwrap(), edges);

    for (satellites, signals, needle) in [
        (
            vec![satellite(0)],
            vec![],
            "outside the 1..=64 satellite mask",
        ),
        (
            vec![satellite(65)],
            vec![],
            "outside the 1..=64 satellite mask",
        ),
        (
            vec![satellite(3)],
            vec![signal(3, 0)],
            "outside the 1..=32 signal mask",
        ),
        (
            vec![satellite(3)],
            vec![signal(3, 33)],
            "outside the 1..=32 signal mask",
        ),
        (vec![satellite(3), satellite(3)], vec![], "listed twice"),
        (
            vec![satellite(3)],
            vec![signal(3, 2), signal(3, 2)],
            "listed twice",
        ),
        (vec![satellite(3)], vec![signal(4, 2)], "does not hold"),
    ] {
        let err = message(satellites, signals)
            .encode()
            .expect_err("the masks cannot state this message");
        assert!(
            matches!(err, Error::InvalidInput(ref text) if text.contains(needle)),
            "expected {needle:?}, got {err}"
        );
    }
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
        let body = m.encode().unwrap();
        let decoded = MsmMessage::decode(&body).unwrap();
        assert_eq!(decoded.system, sys);
        assert_eq!(decoded.kind, kind);
        assert_eq!(decoded.message_number, num);
    }
}

fn lli_msm(
    system: crate::id::GnssSystem,
    kind: MsmKind,
    epoch_time: u32,
    satellite_id: u8,
    signal_id: u8,
    lock_time_indicator: u16,
    half_cycle_ambiguity: bool,
) -> MsmMessage {
    let message_number = match (system, kind) {
        (crate::id::GnssSystem::Gps, MsmKind::Msm4) => 1074,
        (crate::id::GnssSystem::Gps, MsmKind::Msm7) => 1077,
        (crate::id::GnssSystem::Glonass, MsmKind::Msm4) => 1084,
        (crate::id::GnssSystem::Glonass, MsmKind::Msm7) => 1087,
        (crate::id::GnssSystem::Galileo, MsmKind::Msm4) => 1094,
        (crate::id::GnssSystem::Galileo, MsmKind::Msm7) => 1097,
        (crate::id::GnssSystem::Sbas, MsmKind::Msm4) => 1104,
        (crate::id::GnssSystem::Sbas, MsmKind::Msm7) => 1107,
        (crate::id::GnssSystem::Qzss, MsmKind::Msm4) => 1114,
        (crate::id::GnssSystem::Qzss, MsmKind::Msm7) => 1117,
        (crate::id::GnssSystem::BeiDou, MsmKind::Msm4) => 1124,
        (crate::id::GnssSystem::BeiDou, MsmKind::Msm7) => 1127,
        (crate::id::GnssSystem::Navic, MsmKind::Msm4) => 1134,
        (crate::id::GnssSystem::Navic, MsmKind::Msm7) => 1137,
    };
    let mut header = msm_header();
    header.epoch_time = epoch_time;
    MsmMessage {
        message_number,
        system,
        kind,
        header,
        satellites: vec![MsmSatellite {
            id: satellite_id,
            rough_range_ms: 75,
            rough_range_mod1: 512,
            extended_info: (kind == MsmKind::Msm7).then_some(0),
            rough_phase_range_rate_m_s: (kind == MsmKind::Msm7).then_some(0),
        }],
        signals: vec![MsmSignal {
            satellite_id,
            signal_id,
            fine_pseudorange: 0,
            fine_phase_range: 0,
            lock_time_indicator,
            half_cycle_ambiguity,
            cnr: 40,
            fine_phase_range_rate: (kind == MsmKind::Msm7).then_some(0),
        }],
    }
}

#[test]
fn msm_lock_time_tables_and_signal_helpers_are_pinned() {
    let df402 = [
        0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072, 262_144,
        524_288,
    ];
    for (indicator, expected) in df402.into_iter().enumerate() {
        assert_eq!(
            minimum_lock_time_ms(MsmKind::Msm4, indicator as u16),
            Some(expected),
            "DF402 {indicator}"
        );
    }
    assert_eq!(minimum_lock_time_ms(MsmKind::Msm4, 16), None);
    assert_eq!(minimum_lock_time_ms(MsmKind::Msm7, 1024), None);

    let signal = MsmSignal {
        satellite_id: 1,
        signal_id: 2,
        fine_pseudorange: 0,
        fine_phase_range: 0,
        lock_time_indicator: 6,
        half_cycle_ambiguity: false,
        cnr: 0,
        fine_phase_range_rate: None,
    };
    assert_eq!(signal.minimum_lock_time_ms(MsmKind::Msm4), Some(1024));
    assert_eq!(signal.minimum_lock_time_ms(MsmKind::Msm7), Some(6));
}

#[test]
fn derive_lli_pins_loss_of_lock_truth_table() {
    let prev_some = PreviousLock {
        min_lock_time_ms: Some(512),
        elapsed_ms: 400,
    };
    assert_eq!(derive_lli(None, Some(0), false), 0);
    assert_eq!(derive_lli(None, None, true), LLI_HALF_CYCLE);
    assert_eq!(
        derive_lli(Some(prev_some), Some(256), false),
        LLI_LOSS_OF_LOCK
    );
    assert_eq!(
        derive_lli(
            Some(PreviousLock {
                min_lock_time_ms: Some(512),
                elapsed_ms: 600,
            }),
            Some(512),
            false
        ),
        LLI_LOSS_OF_LOCK
    );
    assert_eq!(derive_lli(Some(prev_some), Some(512), false), 0);
    assert_eq!(
        derive_lli(
            Some(PreviousLock {
                min_lock_time_ms: None,
                elapsed_ms: 600,
            }),
            Some(512),
            false
        ),
        LLI_LOSS_OF_LOCK
    );
    assert_eq!(
        derive_lli(
            Some(PreviousLock {
                min_lock_time_ms: None,
                elapsed_ms: 400,
            }),
            Some(512),
            false
        ),
        0
    );
    assert_eq!(derive_lli(Some(prev_some), None, false), LLI_LOSS_OF_LOCK);
    assert_eq!(
        derive_lli(Some(prev_some), None, true),
        LLI_LOSS_OF_LOCK | LLI_HALF_CYCLE
    );
}

#[test]
fn derive_lli_pins_same_bucket_and_half_cycle_cases() {
    let previous = PreviousLock {
        min_lock_time_ms: Some(512),
        elapsed_ms: 400,
    };
    assert_eq!(derive_lli(Some(previous), Some(512), false), 0);
    assert_eq!(
        derive_lli(
            Some(PreviousLock {
                elapsed_ms: 512,
                ..previous
            }),
            Some(512),
            false
        ),
        0
    );
    assert_eq!(
        derive_lli(
            Some(PreviousLock {
                elapsed_ms: 600,
                ..previous
            }),
            Some(512),
            false
        ),
        LLI_LOSS_OF_LOCK
    );
    assert_eq!(
        derive_lli(
            Some(PreviousLock {
                min_lock_time_ms: Some(0),
                elapsed_ms: 30,
            }),
            Some(0),
            false
        ),
        LLI_LOSS_OF_LOCK
    );
    assert_eq!(
        [
            derive_lli(Some(previous), Some(512), false),
            derive_lli(Some(previous), Some(512), true),
            derive_lli(Some(previous), Some(512), false),
        ],
        [0, LLI_HALF_CYCLE, 0]
    );
}

#[test]
fn lock_time_tracker_handles_mixed_msm_rollovers_duplicates_and_reset() {
    use crate::id::GnssSystem::*;

    let mut tracker = LockTimeTracker::new();
    let first = lli_msm(Gps, MsmKind::Msm4, 10_000, 3, 2, 6, false);
    assert_eq!(tracker.observe(&first)[0].lli, 0);

    let mixed_same_raw_decrease = lli_msm(Gps, MsmKind::Msm7, 11_000, 3, 2, 6, false);
    assert_eq!(
        tracker.observe(&mixed_same_raw_decrease)[0].lli,
        LLI_LOSS_OF_LOCK,
        "DF402 6 means 1024 ms, DF407 6 means 6 ms"
    );

    let duplicate = lli_msm(Gps, MsmKind::Msm7, 11_000, 3, 2, 700, false);
    assert_eq!(tracker.observe(&duplicate)[0].lli, 0);
    let after_duplicate = lli_msm(Gps, MsmKind::Msm7, 11_010, 3, 2, 20, false);
    assert_eq!(
        tracker.observe(&after_duplicate)[0].lli,
        0,
        "duplicate epoch must not replace the stored low lock time"
    );

    let week_wrap = msm_epoch_dt_ms(Gps, 604_799_000, 1_000);
    assert_eq!(week_wrap, 2_000);

    let glonass_prev = 6u32 << 27 | 86_399_000;
    let glonass_now = 1_000;
    assert_eq!(msm_epoch_dt_ms(Glonass, glonass_prev, glonass_now), 2_000);
    let glonass_unknown_prev = 7u32 << 27 | 86_399_000;
    assert_eq!(msm_epoch_dt_ms(Glonass, glonass_unknown_prev, 1_000), 2_000);

    let galileo = lli_msm(Galileo, MsmKind::Msm4, 20_000, 3, 2, 1, false);
    assert_eq!(
        tracker.observe(&galileo)[0].lli,
        0,
        "same satellite/signal id in another constellation has separate state"
    );

    tracker.reset();
    assert_eq!(tracker.observe(&mixed_same_raw_decrease)[0].lli, 0);
}

#[test]
fn msm_signal_rinex_code_has_anchor_mappings() {
    use crate::id::GnssSystem::*;
    let cases = [
        (Gps, 2, Some("1C")),
        (Gps, 10, Some("2W")),
        (Gps, 22, Some("5I")),
        (Gps, 23, Some("5Q")),
        (Gps, 24, Some("5X")),
        (Galileo, 2, Some("1C")),
        (Glonass, 2, Some("1C")),
        (BeiDou, 2, Some("2I")),
        (Sbas, 2, Some("1C")),
        (Gps, 1, None),
        (Gps, 33, None),
    ];
    for (system, signal, expected) in cases {
        assert_eq!(
            msm_signal_rinex_code(system, signal),
            expected,
            "{system:?} signal {signal}"
        );
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
    assert_eq!(message.encode().unwrap(), body);
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

#[test]
fn decode_stream_surfaces_skipped_frames_without_dropping_unsupported_messages() {
    let valid = Message::Msm(lli_msm(
        crate::id::GnssSystem::Gps,
        MsmKind::Msm7,
        10_000,
        3,
        2,
        64,
        false,
    ));
    let valid_frame = valid.to_frame().unwrap();

    let mut unsupported_body = BitWriter::new();
    unsupported_body.push_u(1230, 12);
    unsupported_body.push_u(0xABCD, 16);
    let unsupported = Message::Unsupported(UnsupportedMessage {
        message_number: 1230,
        body: unsupported_body.into_bytes(),
    });
    let unsupported_frame = unsupported.to_frame().unwrap();

    let mut truncated_body = BitWriter::new();
    truncated_body.push_u(1005, 12);
    let truncated_frame = encode_frame(&truncated_body.into_bytes()).unwrap();
    let garbage = [0xAA, 0xD3, 0x00, 0x00, 0x12, 0x34, 0x56];

    let mut stream_bytes = Vec::new();
    stream_bytes.extend_from_slice(&valid_frame);
    stream_bytes.extend_from_slice(&unsupported_frame);
    stream_bytes.extend_from_slice(&garbage);
    let truncated_offset = stream_bytes.len();
    stream_bytes.extend_from_slice(&truncated_frame);
    stream_bytes.extend_from_slice(&valid_frame);

    let stream = decode_stream(&stream_bytes);
    assert_eq!(
        stream.messages,
        vec![valid.clone(), unsupported.clone(), valid.clone()]
    );
    assert_eq!(decode_messages(&stream_bytes), stream.messages);
    assert_eq!(stream.diagnostics.resync_bytes, garbage.len());
    assert_eq!(
        stream.diagnostics.skipped_frames,
        vec![FrameSkip {
            offset: truncated_offset,
            message_number: Some(1005),
            reason: FrameSkipReason::Truncated,
        }]
    );

    let frames: Vec<_> = FrameScanner::new(&stream_bytes).collect();
    assert_eq!(frames.len(), 4);
}

#[test]
fn decode_stream_and_frame_scanner_agree_after_overlength_preamble() {
    let station = Message::StationCoordinates(sample_station(1005, None));
    let real_frame = station.to_frame().unwrap();

    let mut stream_bytes = vec![PREAMBLE, 0x03, 0xFF, 0xAA, 0xBB];
    stream_bytes.extend_from_slice(&real_frame);

    let stream = decode_stream(&stream_bytes);
    assert_eq!(stream.messages, vec![station]);

    let frames: Vec<_> = FrameScanner::new(&stream_bytes).collect();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].body, decode_frame(&real_frame).unwrap().body);
    assert_eq!(
        stream
            .messages
            .iter()
            .map(|message| message.encode().expect("a decoded message encodes"))
            .collect::<Vec<_>>(),
        frames
            .iter()
            .map(|frame| frame.body.to_vec())
            .collect::<Vec<_>>()
    );
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
    let body = message.encode().unwrap();
    let decoded = Message::decode(&body).unwrap();
    assert_eq!(decoded, message, "decoded IR must equal the constructed IR");
    assert_eq!(
        decoded.encode().unwrap(),
        body,
        "re-encode must be byte-identical"
    );
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
    let body = eph.encode().unwrap();
    let decoded = GpsEphemeris::decode(&body).unwrap();
    assert_eq!(decoded, eph);
    assert_eq!(decoded.encode().unwrap(), body);
    assert_round_trips(Message::GpsEphemeris(eph));
}

#[test]
fn build_glonass_ephemeris_from_scratch_round_trips() {
    let eph = valid_glonass_ephemeris();
    let body = eph.encode().unwrap();
    let decoded = GlonassEphemeris::decode(&body).unwrap();
    assert_eq!(decoded, eph);
    assert_eq!(decoded.encode().unwrap(), body);
    assert_round_trips(Message::GlonassEphemeris(eph));
}

#[test]
fn build_galileo_fnav_ephemeris_from_scratch_round_trips_and_evaluates() {
    let eph = GalileoFnavEphemeris {
        satellite_id: 11,
        week_number: 1380,
        iod_nav: 321,
        sisa: 73,
        idot: -120,
        t_oc: 1200,
        a_f2: -2,
        a_f1: 3456,
        a_f0: -456_789,
        c_rs: -2000,
        delta_n: 3200,
        m0: -500_000_000,
        c_uc: -120,
        eccentricity: 85_899_345,
        c_us: 145,
        sqrt_a: 2_852_126_720,
        t_oe: 1200,
        c_ic: -55,
        omega0: 400_000_000,
        c_is: 42,
        i0: 650_000_000,
        c_rc: 9000,
        omega: -300_000_000,
        omega_dot: -2400,
        bgd_e5a_e1: -8,
        e5a_signal_health: 0,
        e5a_data_validity: false,
        reserved: 0,
    };
    let body = eph.encode().unwrap();
    assert_eq!(body.len(), 62);
    let decoded = GalileoFnavEphemeris::decode(&body).unwrap();
    assert_eq!(decoded, eph);
    assert_eq!(decoded.encode().unwrap(), body);
    assert_round_trips(Message::GalileoFnavEphemeris(eph));
    assert_broadcast_record_is_nontrivial(decoded.to_broadcast_record().unwrap());
}

#[test]
fn build_galileo_inav_ephemeris_from_scratch_round_trips_and_evaluates() {
    let eph = GalileoInavEphemeris {
        satellite_id: 12,
        week_number: 1380,
        iod_nav: 322,
        sisa_index: 74,
        idot: -125,
        t_oc: 1200,
        a_f2: -1,
        a_f1: 2345,
        a_f0: -345_678,
        c_rs: -2100,
        delta_n: 3100,
        m0: -450_000_000,
        c_uc: -118,
        eccentricity: 82_000_000,
        c_us: 140,
        sqrt_a: 2_852_126_720,
        t_oe: 1200,
        c_ic: -53,
        omega0: 420_000_000,
        c_is: 41,
        i0: 650_000_000,
        c_rc: 8800,
        omega: -310_000_000,
        omega_dot: -2300,
        bgd_e5a_e1: -8,
        bgd_e5b_e1: 6,
        e5b_signal_health: 0,
        e5b_data_validity: false,
        e1b_signal_health: 0,
        e1b_data_validity: false,
        reserved: 0,
    };
    let body = eph.encode().unwrap();
    assert_eq!(body.len(), 63);
    let decoded = GalileoInavEphemeris::decode(&body).unwrap();
    assert_eq!(decoded, eph);
    assert_eq!(decoded.encode().unwrap(), body);
    assert_round_trips(Message::GalileoInavEphemeris(eph));
    assert_broadcast_record_is_nontrivial(decoded.to_broadcast_record().unwrap());
}

#[test]
fn build_beidou_ephemeris_from_scratch_round_trips_and_evaluates() {
    let eph = BeidouEphemeris {
        satellite_id: 19,
        week_number: 1070,
        sv_urai: 2,
        idot: 115,
        aode: 17,
        t_oc: 9000,
        a_f2: -12,
        a_f1: 45_000,
        a_f0: -123_456,
        aodc: 18,
        c_rs: -12_000,
        delta_n: 2200,
        m0: 300_000_000,
        c_uc: -11_000,
        eccentricity: 70_000_000,
        c_us: 10_000,
        sqrt_a: 3_404_333_056,
        t_oe: 9000,
        c_ic: -9000,
        omega0: -600_000_000,
        c_is: 8500,
        i0: 620_000_000,
        c_rc: 11_500,
        omega: 500_000_000,
        omega_dot: -1800,
        t_gd1: -16,
        t_gd2: 12,
        sv_health: false,
    };
    let body = eph.encode().unwrap();
    assert_eq!(body.len(), 64);
    let decoded = BeidouEphemeris::decode(&body).unwrap();
    assert_eq!(decoded, eph);
    assert_eq!(decoded.encode().unwrap(), body);
    assert_round_trips(Message::BeidouEphemeris(eph));
    assert_broadcast_record_is_nontrivial(decoded.to_broadcast_record().unwrap());
}

#[test]
fn build_qzss_ephemeris_from_scratch_round_trips_and_evaluates() {
    let eph = QzssEphemeris {
        satellite_id: 2,
        t_oc: 4500,
        a_f2: -1,
        a_f1: 1234,
        a_f0: -234_567,
        iode: 44,
        c_rs: -1500,
        delta_n: 2400,
        m0: 250_000_000,
        c_uc: -100,
        eccentricity: 65_000_000,
        c_us: 130,
        sqrt_a: 2_701_770_752,
        t_oe: 3000,
        c_ic: -40,
        omega0: 350_000_000,
        c_is: 38,
        i0: 610_000_000,
        c_rc: 7100,
        omega: -260_000_000,
        omega_dot: -1600,
        idot: 110,
        codes_on_l2: 2,
        week_number: 386,
        ura: 1,
        sv_health: 0,
        t_gd: -4,
        iodc: 44,
        fit_interval: false,
    };
    let body = eph.encode().unwrap();
    assert_eq!(body.len(), 61);
    let decoded = QzssEphemeris::decode(&body).unwrap();
    assert_eq!(decoded, eph);
    assert_eq!(decoded.encode().unwrap(), body);
    assert_round_trips(Message::QzssEphemeris(eph));
    assert_broadcast_record_is_nontrivial(decoded.to_broadcast_record(2434).unwrap());

    // The fit flag reads as RTKLIB reads it from RTCM and from RINEX (0: 2 h, 1: 4 h), so
    // the record written to RINEX reads back with the same flag and fit interval.
    for (flag, hours) in [(false, 2.0), (true, 4.0)] {
        let record = QzssEphemeris {
            fit_interval: flag,
            ..eph
        }
        .to_broadcast_record(2434)
        .unwrap();
        assert_eq!(record.fit_interval_s, Some(hours * 3600.0));
        let encoded = crate::rinex_nav::encode_nav(&[record]).expect("encode QZSS record");
        let reparsed = crate::rinex_nav::parse_nav(&encoded).expect("reparse QZSS record");
        assert_eq!(reparsed[0].fit_interval_s, record.fit_interval_s);
        assert_eq!(
            reparsed[0].stated.orbit7_field2,
            record.stated.orbit7_field2
        );
    }
}

fn assert_broadcast_record_is_nontrivial(record: crate::rinex_nav::BroadcastRecord) {
    use crate::spp::EphemerisSource;

    assert!(record.elements.sqrt_a > 4000.0);
    assert!(record.elements.e > 0.0);
    assert!(record.elements.i0.abs() > 0.1);
    let toe_continuous =
        f64::from(record.toe.week) * crate::constants::SECONDS_PER_WEEK + record.toe.tow_s;
    let query = if record.satellite_id.system == crate::id::GnssSystem::BeiDou {
        toe_continuous
            + crate::constants::GPST_MINUS_BDT_S
            + crate::constants::BDS_EPOCH_MINUS_GPS_EPOCH_S
            - crate::constants::GPS_EPOCH_TO_J2000_S
    } else if record.satellite_id.system == crate::id::GnssSystem::Galileo {
        // A Galileo record is served only after its `toe` (RTKLIB `seleph`'s
        // age-of-data rule), so it is evaluated one second later.
        toe_continuous - crate::constants::GPS_EPOCH_TO_J2000_S + 1.0
    } else {
        toe_continuous - crate::constants::GPS_EPOCH_TO_J2000_S
    };
    let store = crate::rinex_nav::BroadcastStore::new(vec![record]).unwrap();
    let (position, clock) = store
        .position_clock_at_j2000_s(record.satellite_id, query)
        .expect("converted RTCM broadcast record evaluates at its toe");
    assert!(position.iter().all(|value| value.is_finite()));
    assert!(clock.is_finite());
    let radius =
        (position[0] * position[0] + position[1] * position[1] + position[2] * position[2]).sqrt();
    assert!(
        (20_000_000.0..=50_000_000.0).contains(&radius),
        "broadcast radius {radius}"
    );
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
    let body = message.encode().unwrap();
    let decoded = MsmMessage::decode(&body).unwrap();
    assert_eq!(decoded, message);
    assert_eq!(decoded.encode().unwrap(), body);
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
    let body = message.encode().unwrap();
    let decoded = MsmMessage::decode(&body).unwrap();
    assert_eq!(decoded, message);
    assert_eq!(decoded.encode().unwrap(), body);
    assert_round_trips(Message::Msm(message));
}

#[test]
fn msm7_absent_phase_range_rate_distinguished_from_zero() {
    let satellites = vec![
        MsmSatellite {
            id: 1,
            rough_range_ms: 70,
            rough_range_mod1: 256,
            extended_info: Some(0),
            rough_phase_range_rate_m_s: None,
        },
        MsmSatellite {
            id: 2,
            rough_range_ms: 72,
            rough_range_mod1: 512,
            extended_info: Some(0),
            rough_phase_range_rate_m_s: Some(0),
        },
    ];
    let signals = vec![
        MsmSignal {
            satellite_id: 1,
            signal_id: 1,
            fine_pseudorange: 100,
            fine_phase_range: 200,
            lock_time_indicator: 50,
            half_cycle_ambiguity: false,
            cnr: 400,
            fine_phase_range_rate: None,
        },
        MsmSignal {
            satellite_id: 2,
            signal_id: 1,
            fine_pseudorange: 300,
            fine_phase_range: 400,
            lock_time_indicator: 60,
            half_cycle_ambiguity: false,
            cnr: 450,
            fine_phase_range_rate: Some(0),
        },
    ];
    let message = MsmMessage {
        message_number: 1077,
        system: crate::id::GnssSystem::Gps,
        kind: MsmKind::Msm7,
        header: msm_header(),
        satellites,
        signals,
    };
    let body = message.encode().unwrap();
    let decoded = MsmMessage::decode(&body).unwrap();
    assert_eq!(decoded.satellites[0].rough_phase_range_rate_m_s, None);
    assert_eq!(decoded.satellites[1].rough_phase_range_rate_m_s, Some(0));
    assert_eq!(decoded.signals[0].fine_phase_range_rate, None);
    assert_eq!(decoded.signals[1].fine_phase_range_rate, Some(0));
    assert_eq!(decoded, message);
    assert_eq!(decoded.encode().unwrap(), body);
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
        Message::BeidouEphemeris(_) => 1042,
        Message::QzssEphemeris(_) => 1044,
        Message::GalileoFnavEphemeris(_) => 1045,
        Message::GalileoInavEphemeris(_) => 1046,
        Message::Ssr(s) => s.message_number,
        Message::Unsupported(u) => u.message_number,
    };
    assert_eq!(number, 1005);
}

fn valid_gps_ephemeris() -> GpsEphemeris {
    GpsEphemeris {
        satellite_id: 14,
        week_number: 386,
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
    }
}

fn valid_beidou_ephemeris() -> BeidouEphemeris {
    BeidouEphemeris {
        satellite_id: 19,
        week_number: 1070,
        sv_urai: 0,
        idot: 115,
        aode: 17,
        t_oc: 9000,
        a_f2: -12,
        a_f1: 45_000,
        a_f0: -123_456,
        aodc: 18,
        c_rs: -12_000,
        delta_n: 2200,
        m0: 300_000_000,
        c_uc: -11_000,
        eccentricity: 70_000_000,
        c_us: 10_000,
        sqrt_a: 3_404_333_056,
        t_oe: 9000,
        c_ic: -9000,
        omega0: -600_000_000,
        c_is: 8500,
        i0: 620_000_000,
        c_rc: 11_500,
        omega: 500_000_000,
        omega_dot: -1800,
        t_gd1: -16,
        t_gd2: 12,
        sv_health: false,
    }
}

fn valid_qzss_ephemeris() -> QzssEphemeris {
    QzssEphemeris {
        satellite_id: 2,
        t_oc: 4500,
        a_f2: -1,
        a_f1: 1234,
        a_f0: -234_567,
        iode: 44,
        c_rs: -1500,
        delta_n: 2400,
        m0: 250_000_000,
        c_uc: -100,
        eccentricity: 65_000_000,
        c_us: 130,
        sqrt_a: 2_701_770_752,
        t_oe: 3000,
        c_ic: -40,
        omega0: 350_000_000,
        c_is: 38,
        i0: 610_000_000,
        c_rc: 7100,
        omega: -260_000_000,
        omega_dot: -1600,
        idot: 110,
        codes_on_l2: 2,
        week_number: 386,
        ura: 0,
        sv_health: 0,
        t_gd: -4,
        iodc: 44,
        fit_interval: false,
    }
}

fn valid_galileo_fnav_ephemeris() -> GalileoFnavEphemeris {
    GalileoFnavEphemeris {
        satellite_id: 11,
        week_number: 1380,
        iod_nav: 321,
        sisa: 0,
        idot: -120,
        t_oc: 1200,
        a_f2: -2,
        a_f1: 3456,
        a_f0: -456_789,
        c_rs: -2000,
        delta_n: 3200,
        m0: -500_000_000,
        c_uc: -120,
        eccentricity: 85_899_345,
        c_us: 145,
        sqrt_a: 2_852_126_720,
        t_oe: 1200,
        c_ic: -55,
        omega0: 400_000_000,
        c_is: 42,
        i0: 650_000_000,
        c_rc: 9000,
        omega: -300_000_000,
        omega_dot: -2400,
        bgd_e5a_e1: -8,
        e5a_signal_health: 0,
        e5a_data_validity: false,
        reserved: 0,
    }
}

fn valid_galileo_inav_ephemeris() -> GalileoInavEphemeris {
    GalileoInavEphemeris {
        satellite_id: 12,
        week_number: 1380,
        iod_nav: 322,
        sisa_index: 0,
        idot: -125,
        t_oc: 1200,
        a_f2: -1,
        a_f1: 2345,
        a_f0: -345_678,
        c_rs: -2100,
        delta_n: 3100,
        m0: -450_000_000,
        c_uc: -118,
        eccentricity: 82_000_000,
        c_us: 140,
        sqrt_a: 2_852_126_720,
        t_oe: 1200,
        c_ic: -53,
        omega0: 420_000_000,
        c_is: 41,
        i0: 650_000_000,
        c_rc: 8800,
        omega: -310_000_000,
        omega_dot: -2300,
        bgd_e5a_e1: -8,
        bgd_e5b_e1: 6,
        e5b_signal_health: 0,
        e5b_data_validity: false,
        e1b_signal_health: 0,
        e1b_data_validity: false,
        reserved: 0,
    }
}

fn valid_glonass_ephemeris() -> GlonassEphemeris {
    GlonassEphemeris {
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
    }
}

/// `expect_err` with the message number and the raw value in the panic text.
fn expect_refusal(
    result: crate::error::Result<crate::id::GnssSatelliteId>,
    message: &str,
    satellite_id: u8,
) -> Error {
    match result {
        Err(err) => err,
        Ok(sat) => panic!("{message} accepted satellite id {satellite_id} as {sat}"),
    }
}

/// Call `satellite()` on each ephemeris message with a given raw satellite
/// field, returning one result per message in a fixed order:
/// 1019, 1020, 1042, 1045, 1046 (all six-bit), then 1044 (four-bit).
fn ephemeris_satellites(
    satellite_id: u8,
) -> Vec<(
    &'static str,
    crate::error::Result<crate::id::GnssSatelliteId>,
)> {
    let mut gps = valid_gps_ephemeris();
    gps.satellite_id = satellite_id;
    let mut glonass = valid_glonass_ephemeris();
    glonass.satellite_id = satellite_id;
    let mut beidou = valid_beidou_ephemeris();
    beidou.satellite_id = satellite_id;
    let mut fnav = valid_galileo_fnav_ephemeris();
    fnav.satellite_id = satellite_id;
    let mut inav = valid_galileo_inav_ephemeris();
    inav.satellite_id = satellite_id;
    let mut qzss = valid_qzss_ephemeris();
    qzss.satellite_id = satellite_id;
    vec![
        ("1019", gps.satellite()),
        ("1020", glonass.satellite()),
        ("1042", beidou.satellite()),
        ("1045", fnav.satellite()),
        ("1046", inav.satellite()),
        ("1044", qzss.satellite()),
    ]
}

/// The shared satellite-token range is 1..=99 for every constellation, so the
/// raw field is what actually bounds these messages. 63 is the widest value a
/// six-bit field carries and 15 the widest a four-bit one carries; both
/// convert, and converting says only that the number is well formed and
/// transmissible, not that the constellation flies that satellite. GPS 1019
/// values 40..=63 convert too, to the SBAS satellites DF009 names with them.
#[test]
fn ephemeris_satellite_accepts_every_value_its_raw_field_can_carry() {
    for satellite_id in 1..=63u8 {
        for (message, result) in ephemeris_satellites(satellite_id) {
            // 1044 is the only four-bit field; the rest take the whole six-bit
            // range.
            let expected_ok = message != "1044" || satellite_id <= 15;
            assert_eq!(
                result.is_ok(),
                expected_ok,
                "{message} satellite id {satellite_id}"
            );
            if let Ok(sat) = result {
                if message == "1019" && satellite_id >= 40 {
                    assert_eq!(sat.system, crate::id::GnssSystem::Sbas, "{satellite_id}");
                    assert_eq!(sat.prn, satellite_id - 20, "{satellite_id}");
                } else {
                    assert_eq!(sat.prn, satellite_id, "{message}");
                }
            }
        }
    }

    // Numbers above the operational roster are ordinary field values: an
    // extended GLONASS slot, and the GPS satellite number CODE's DCB tables
    // carry as G34.
    let mut glonass = valid_glonass_ephemeris();
    glonass.satellite_id = 28;
    assert_eq!(glonass.satellite().unwrap().to_string(), "R28");
    let mut gps = valid_gps_ephemeris();
    gps.satellite_id = 34;
    assert_eq!(gps.satellite().unwrap().to_string(), "G34");
}

/// Zero is not a spellable satellite token, and a value wider than the raw
/// field is not a satellite the message can name at all. Both are refused,
/// per message, with a typed error.
#[test]
fn ephemeris_satellite_refuses_zero_and_values_wider_than_the_raw_field() {
    for (message, result) in ephemeris_satellites(0) {
        let err = result.expect_err("satellite id 0 is not a spellable token");
        assert!(
            matches!(err, Error::Parse(_)),
            "{message}: refusal must be typed, got {err}"
        );
        assert!(
            err.to_string().contains("1..=99"),
            "{message}: zero is refused as a token, got {err}"
        );
    }

    for satellite_id in [64u8, 65, 100, 199, 255] {
        for (message, result) in ephemeris_satellites(satellite_id) {
            let err = expect_refusal(result, message, satellite_id);
            assert!(
                matches!(err, Error::Parse(_)),
                "{message} {satellite_id}: refusal must be typed, got {err}"
            );
            assert!(
                err.to_string().contains("raw satellite field"),
                "{message} {satellite_id}: refusal must name the raw field, got {err}"
            );
        }
    }

    // QZSS 1044 is four bits, so it stops at 15 while the six-bit messages
    // still accept 16..=63.
    for satellite_id in [16u8, 20, 63] {
        let mut qzss = valid_qzss_ephemeris();
        qzss.satellite_id = satellite_id;
        let err = qzss
            .satellite()
            .expect_err("1044 carries a four-bit satellite field")
            .to_string();
        assert!(err.contains("4-bit"), "{satellite_id}: {err}");
        let mut beidou = valid_beidou_ephemeris();
        beidou.satellite_id = satellite_id;
        assert!(beidou.satellite().is_ok(), "1042 {satellite_id}");
    }
}

/// DF009 values 40..=63 name SBAS satellites, broadcast PRN `value + 80`,
/// which is how RTKLIB `decode_type1019` reads them after reading the rest of
/// the message the same way. The satellite is the SBAS one; the GPS LNAV
/// broadcast record is refused, since no such record exists for an SBAS
/// satellite. 33..=39 are GPS numbers above the operational roster.
#[test]
fn gps_1019_df009_sbas_range_names_sbas_satellites() {
    for satellite_id in 33..=39u8 {
        let mut gps = valid_gps_ephemeris();
        gps.satellite_id = satellite_id;
        let sat = gps.satellite().expect("DF009 33..=39 is a GPS number");
        assert_eq!(sat.system, crate::id::GnssSystem::Gps);
        assert_eq!(sat.prn, satellite_id);
    }
    for (satellite_id, token) in [(40u8, "S20"), (58, "S38"), (63, "S43")] {
        let mut gps = valid_gps_ephemeris();
        gps.satellite_id = satellite_id;
        assert_eq!(
            gps.satellite().expect("DF009 SBAS satellite").to_string(),
            token
        );
        let full_week = 2048 + u32::from(gps.week_number);
        let err = gps
            .to_broadcast_record(full_week)
            .expect_err("no GPS LNAV record for an SBAS satellite");
        assert!(
            err.to_string().contains("no GPS LNAV broadcast record"),
            "{satellite_id}: {err}"
        );
    }
}

/// A refused satellite id is refused, not repaired. The raw struct keeps the
/// value it held, `to_broadcast_record` fails instead of emitting a record for
/// the wrong satellite, and a decoded body still round-trips byte for byte.
#[test]
fn ephemeris_satellite_refusal_leaves_the_raw_record_untouched() {
    let mut gps = valid_gps_ephemeris();
    gps.satellite_id = 100;
    assert!(gps.satellite().is_err());
    assert!(gps.to_broadcast_record(2434).is_err());
    assert_eq!(
        gps.satellite_id, 100,
        "the raw field is not zeroed or clamped"
    );

    // A body that really was transmitted with the widest six-bit value decodes
    // to that value, keeps its reserved bits, and re-encodes identically.
    let mut widest = valid_gps_ephemeris();
    widest.satellite_id = 63;
    let body = widest.encode().unwrap();
    let decoded = GpsEphemeris::decode(&body).unwrap();
    assert_eq!(decoded, widest);
    assert_eq!(decoded.satellite_id, 63);
    assert_eq!(decoded.encode().unwrap(), body);
    assert_eq!(decoded.satellite().unwrap().to_string(), "S43");

    let mut qzss = valid_qzss_ephemeris();
    qzss.satellite_id = 15;
    let qzss_body = qzss.encode().unwrap();
    let qzss_decoded = QzssEphemeris::decode(&qzss_body).unwrap();
    assert_eq!(qzss_decoded, qzss);
    assert_eq!(qzss_decoded.encode().unwrap(), qzss_body);
    assert_eq!(qzss_decoded.satellite().unwrap().to_string(), "J15");
}

/// A satellite id wider than its message's raw field is refused by the
/// encoder instead of being written as its low bits, which would name another
/// satellite: 100 would become 36 in six bits and 20 would become 4 in four.
#[test]
fn raw_ephemeris_encoders_refuse_out_of_width_satellite_ids() {
    let mut gps = valid_gps_ephemeris();
    gps.satellite_id = 100;
    let mut glonass = valid_glonass_ephemeris();
    glonass.satellite_id = 64;
    let mut beidou = valid_beidou_ephemeris();
    beidou.satellite_id = 64;
    let mut fnav = valid_galileo_fnav_ephemeris();
    fnav.satellite_id = 255;
    let mut inav = valid_galileo_inav_ephemeris();
    inav.satellite_id = 64;
    let mut qzss = valid_qzss_ephemeris();
    qzss.satellite_id = 20;
    for (message, result) in [
        ("1019", gps.encode()),
        ("1020", glonass.encode()),
        ("1042", beidou.encode()),
        ("1045", fnav.encode()),
        ("1046", inav.encode()),
        ("1044", qzss.encode()),
    ] {
        let err = result.expect_err("an out-of-width satellite id is refused");
        assert!(
            matches!(err, Error::InvalidInput(ref text)
                if text.contains("raw satellite field") && text.contains(message)),
            "{message}: {err}"
        );
    }
    assert!(
        Message::GpsEphemeris(gps).to_frame().is_err(),
        "the framed path refuses it too"
    );

    // The widest value each field carries still encodes.
    let mut qzss = valid_qzss_ephemeris();
    qzss.satellite_id = 15;
    assert!(qzss.encode().is_ok());
    let mut gps = valid_gps_ephemeris();
    gps.satellite_id = 63;
    assert!(gps.encode().is_ok());
}

#[test]
fn broadcast_conversion_refuses_ura_absence_and_range_for_gps_beidou_qzss() {
    // GPS (message 1019)
    let mut gps_eph = valid_gps_ephemeris();
    gps_eph.sv_accuracy = 0;
    assert_eq!(
        gps_eph
            .to_broadcast_record(2434)
            .unwrap()
            .sv_accuracy_m
            .unwrap(),
        2.4
    );
    gps_eph.sv_accuracy = 14;
    assert_eq!(
        gps_eph
            .to_broadcast_record(2434)
            .unwrap()
            .sv_accuracy_m
            .unwrap(),
        6144.0
    );
    gps_eph.sv_accuracy = 15;
    match gps_eph.to_broadcast_record(2434) {
        Err(Error::InvalidInput(msg)) => {
            assert!(msg.contains("GPS"));
            assert!(msg.contains("URA index 15"));
            assert!(msg.contains("no accuracy prediction"));
        }
        other => panic!("expected InvalidInput for GPS URA 15 absence, got {other:?}"),
    }
    gps_eph.sv_accuracy = 16;
    match gps_eph.to_broadcast_record(2434) {
        Err(Error::InvalidInput(msg)) => {
            assert!(msg.contains("GPS"));
            assert!(msg.contains("URA index 16"));
            assert!(msg.contains("exceeds 4-bit range"));
        }
        other => panic!("expected InvalidInput for GPS URA 16 out of range, got {other:?}"),
    }

    // BeiDou (message 1042)
    let mut bds_eph = valid_beidou_ephemeris();
    bds_eph.sv_urai = 0;
    assert_eq!(
        bds_eph
            .to_broadcast_record()
            .unwrap()
            .sv_accuracy_m
            .unwrap(),
        2.4
    );
    bds_eph.sv_urai = 14;
    assert_eq!(
        bds_eph
            .to_broadcast_record()
            .unwrap()
            .sv_accuracy_m
            .unwrap(),
        6144.0
    );
    bds_eph.sv_urai = 15;
    match bds_eph.to_broadcast_record() {
        Err(Error::InvalidInput(msg)) => {
            assert!(msg.contains("BeiDou"));
            assert!(msg.contains("URA index 15"));
            assert!(msg.contains("no accuracy prediction"));
        }
        other => panic!("expected InvalidInput for BeiDou URA 15 absence, got {other:?}"),
    }
    bds_eph.sv_urai = 16;
    match bds_eph.to_broadcast_record() {
        Err(Error::InvalidInput(msg)) => {
            assert!(msg.contains("BeiDou"));
            assert!(msg.contains("URA index 16"));
            assert!(msg.contains("exceeds 4-bit range"));
        }
        other => panic!("expected InvalidInput for BeiDou URA 16 out of range, got {other:?}"),
    }

    // QZSS (message 1044)
    let mut qzss_eph = valid_qzss_ephemeris();
    qzss_eph.ura = 0;
    assert_eq!(
        qzss_eph
            .to_broadcast_record(2434)
            .unwrap()
            .sv_accuracy_m
            .unwrap(),
        2.4
    );
    qzss_eph.ura = 14;
    assert_eq!(
        qzss_eph
            .to_broadcast_record(2434)
            .unwrap()
            .sv_accuracy_m
            .unwrap(),
        6144.0
    );
    qzss_eph.ura = 15;
    match qzss_eph.to_broadcast_record(2434) {
        Err(Error::InvalidInput(msg)) => {
            assert!(msg.contains("QZSS"));
            assert!(msg.contains("URA index 15"));
            assert!(msg.contains("no accuracy prediction"));
        }
        other => panic!("expected InvalidInput for QZSS URA 15 absence, got {other:?}"),
    }
    qzss_eph.ura = 16;
    match qzss_eph.to_broadcast_record(2434) {
        Err(Error::InvalidInput(msg)) => {
            assert!(msg.contains("QZSS"));
            assert!(msg.contains("URA index 16"));
            assert!(msg.contains("exceeds 4-bit range"));
        }
        other => panic!("expected InvalidInput for QZSS URA 16 out of range, got {other:?}"),
    }
}

#[test]
fn broadcast_conversion_refuses_galileo_sisa_spare_and_napa_and_accepts_valid_indices() {
    let mut fnav = valid_galileo_fnav_ephemeris();
    let mut inav = valid_galileo_inav_ephemeris();

    // Valid characterized indices: 0 => 0.00 m, 15 => 0.15 m, 125 => 6.00 m
    for (sisa, expected_m) in [(0, 0.0), (15, 0.15), (125, 6.00)] {
        fnav.sisa = sisa;
        assert_eq!(
            fnav.to_broadcast_record().unwrap().sv_accuracy_m.unwrap(),
            expected_m
        );
        inav.sisa_index = sisa;
        assert_eq!(
            inav.to_broadcast_record().unwrap().sv_accuracy_m.unwrap(),
            expected_m
        );
    }

    // Spare indices (126..=254): test lower and upper bounds
    for spare_idx in [126, 254] {
        fnav.sisa = spare_idx;
        match fnav.to_broadcast_record() {
            Err(Error::InvalidInput(msg)) => {
                assert!(msg.contains("Galileo"));
                assert!(msg.contains(&format!("SISA index {spare_idx}")));
                assert!(msg.contains("spare with no defined accuracy"));
            }
            other => {
                panic!("expected spare InvalidInput for F/NAV SISA {spare_idx}, got {other:?}")
            }
        }
        inav.sisa_index = spare_idx;
        match inav.to_broadcast_record() {
            Err(Error::InvalidInput(msg)) => {
                assert!(msg.contains("Galileo"));
                assert!(msg.contains(&format!("SISA index {spare_idx}")));
                assert!(msg.contains("spare with no defined accuracy"));
            }
            other => {
                panic!("expected spare InvalidInput for I/NAV SISA {spare_idx}, got {other:?}")
            }
        }
    }

    // NAPA index (255)
    fnav.sisa = 255;
    match fnav.to_broadcast_record() {
        Err(Error::InvalidInput(msg)) => {
            assert!(msg.contains("Galileo"));
            assert!(msg.contains("SISA index 255"));
            assert!(msg.contains("no accuracy prediction available (NAPA)"));
        }
        other => panic!("expected NAPA InvalidInput for F/NAV SISA 255, got {other:?}"),
    }
    inav.sisa_index = 255;
    match inav.to_broadcast_record() {
        Err(Error::InvalidInput(msg)) => {
            assert!(msg.contains("Galileo"));
            assert!(msg.contains("SISA index 255"));
            assert!(msg.contains("no accuracy prediction available (NAPA)"));
        }
        other => panic!("expected NAPA InvalidInput for I/NAV SISA 255, got {other:?}"),
    }
}

#[test]
fn galileo_sisa_piecewise_boundaries_map_accurately() {
    let mut fnav = valid_galileo_fnav_ephemeris();
    let mut inav = valid_galileo_inav_ephemeris();
    let boundaries = [
        (49, 0.49),
        (50, 0.50),
        (74, 0.98),
        (75, 1.00),
        (99, 1.96),
        (100, 2.00),
        (125, 6.00),
    ];
    for (sisa, expected_m) in boundaries {
        fnav.sisa = sisa;
        let fnav_rec = fnav.to_broadcast_record().unwrap();
        assert!(
            (fnav_rec.sv_accuracy_m.unwrap() - expected_m).abs() < 1e-12,
            "F/NAV SISA index {sisa} mapped to {}, expected {expected_m}",
            fnav_rec.sv_accuracy_m.unwrap()
        );
        inav.sisa_index = sisa;
        let inav_rec = inav.to_broadcast_record().unwrap();
        assert!(
            (inav_rec.sv_accuracy_m.unwrap() - expected_m).abs() < 1e-12,
            "I/NAV SISA index {sisa} mapped to {}, expected {expected_m}",
            inav_rec.sv_accuracy_m.unwrap()
        );
    }
}

#[test]
fn raw_codec_round_trips_retain_absence_and_spare_accuracy_indices() {
    // GPS with URA 15 (absence)
    let mut gps = valid_gps_ephemeris();
    gps.sv_accuracy = 15;
    let bytes = gps.encode().unwrap();
    let decoded = GpsEphemeris::decode(&bytes).unwrap();
    assert_eq!(decoded.sv_accuracy, 15);
    assert_eq!(decoded, gps);

    // BeiDou with URA 15 (absence)
    let mut bds = valid_beidou_ephemeris();
    bds.sv_urai = 15;
    let bytes = bds.encode().unwrap();
    let decoded = BeidouEphemeris::decode(&bytes).unwrap();
    assert_eq!(decoded.sv_urai, 15);
    assert_eq!(decoded, bds);

    // QZSS with URA 15 (absence)
    let mut qzss = valid_qzss_ephemeris();
    qzss.ura = 15;
    let bytes = qzss.encode().unwrap();
    let decoded = QzssEphemeris::decode(&bytes).unwrap();
    assert_eq!(decoded.ura, 15);
    assert_eq!(decoded, qzss);

    // Galileo F/NAV with spare 126, spare 254, and NAPA 255
    let mut fnav = valid_galileo_fnav_ephemeris();
    for sisa in [126, 254, 255] {
        fnav.sisa = sisa;
        let bytes = fnav.encode().unwrap();
        let decoded = GalileoFnavEphemeris::decode(&bytes).unwrap();
        assert_eq!(decoded.sisa, sisa);
        assert_eq!(decoded, fnav);
    }

    // Galileo I/NAV with spare 126, spare 254, and NAPA 255
    let mut inav = valid_galileo_inav_ephemeris();
    for sisa in [126, 254, 255] {
        inav.sisa_index = sisa;
        let bytes = inav.encode().unwrap();
        let decoded = GalileoInavEphemeris::decode(&bytes).unwrap();
        assert_eq!(decoded.sisa_index, sisa);
        assert_eq!(decoded, inav);
    }
}
