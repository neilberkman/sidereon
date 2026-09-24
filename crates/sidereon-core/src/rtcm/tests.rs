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
        trailing_bits: Vec::new(),
    }
}

#[test]
fn station_1005_round_trip() {
    let station = sample_station(1005, None);
    let body = station.encode().unwrap();
    // 1005 body is exactly 19 bytes (152 bits).
    assert_eq!(body.len(), 19);
    assert_eq!(message_number(&body).unwrap(), 1005);
    let decoded = StationCoordinates::decode(&body).unwrap();
    assert_eq!(decoded, station);
}

#[test]
fn station_1006_round_trip_and_meters() {
    let station = sample_station(1006, Some(15_000));
    let body = station.encode().unwrap();
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
    let decoded = decode_messages(&frame).unwrap();
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
        trailing_bits: Vec::new(),
    };
    let body = descriptor.encode().unwrap();
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
        trailing_bits: Vec::new(),
    };
    let body = descriptor.encode().unwrap();
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
        trailing_bits: Vec::new(),
    };
    let frame = Message::AntennaDescriptor(descriptor.clone())
        .to_frame()
        .unwrap();
    assert_eq!(
        decode_messages(&frame).unwrap(),
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
        trailing_bits: Vec::new(),
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
        negative_zero: 0,
        trailing_bits: Vec::new(),
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
            rough_range_ms: Some(75),
            rough_range_mod1: 512,
            extended_info: None,
            rough_phase_range_rate_m_s: None,
        },
        MsmSatellite {
            id: 14,
            rough_range_ms: Some(80),
            rough_range_mod1: 1000,
            extended_info: None,
            rough_phase_range_rate_m_s: None,
        },
        MsmSatellite {
            id: 22,
            rough_range_ms: Some(255),
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
            fine_pseudorange: Some(-4000),
            fine_phase_range: Some(100_000),
            lock_time_indicator: Some(9),
            half_cycle_ambiguity: Some(false),
            cnr: Some(45),
            fine_phase_range_rate: None,
        },
        MsmSignal {
            satellite_id: 3,
            signal_id: 15,
            fine_pseudorange: Some(4000),
            fine_phase_range: Some(-100_000),
            lock_time_indicator: Some(3),
            half_cycle_ambiguity: Some(true),
            cnr: Some(38),
            fine_phase_range_rate: None,
        },
        MsmSignal {
            satellite_id: 14,
            signal_id: 2,
            fine_pseudorange: Some(16),
            fine_phase_range: Some(-7),
            lock_time_indicator: Some(15),
            half_cycle_ambiguity: Some(false),
            cnr: Some(50),
            fine_phase_range_rate: None,
        },
    ];
    let message = MsmMessage {
        message_number: 1074,
        system: crate::id::GnssSystem::Gps,
        kind: MsmKind::Msm4,
        header: msm_header(),
        signal_mask: msm_signal_mask(&signals),
        satellites,
        signals,
        trailing_bits: Vec::new(),
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
        rough_range_ms: Some(70),
        rough_range_mod1: 256,
        extended_info: None,
        rough_phase_range_rate_m_s: None,
    };
    let signal = |satellite_id, signal_id| MsmSignal {
        satellite_id,
        signal_id,
        fine_pseudorange: Some(16),
        fine_phase_range: Some(-7),
        lock_time_indicator: Some(15),
        half_cycle_ambiguity: Some(false),
        cnr: Some(50),
        fine_phase_range_rate: None,
    };
    let message = |satellites: Vec<MsmSatellite>, signals: Vec<MsmSignal>| MsmMessage {
        message_number: 1074,
        system: crate::id::GnssSystem::Gps,
        kind: MsmKind::Msm4,
        header: msm_header(),
        signal_mask: msm_signal_mask(&signals),
        satellites,
        signals,
        trailing_bits: Vec::new(),
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
            matches!(err, Error::RtcmEncode(ref e) if e.to_string().contains(needle)),
            "expected {needle:?}, got {err}"
        );
    }
}

#[test]
fn msm7_glonass_round_trip_with_extended_info() {
    let satellites = vec![
        MsmSatellite {
            id: 1,
            rough_range_ms: Some(70),
            rough_range_mod1: 256,
            extended_info: Some(8),
            rough_phase_range_rate_m_s: Some(-1500),
        },
        MsmSatellite {
            id: 9,
            rough_range_ms: Some(90),
            rough_range_mod1: 900,
            extended_info: Some(2),
            rough_phase_range_rate_m_s: Some(3000),
        },
    ];
    let signals = vec![
        MsmSignal {
            satellite_id: 1,
            signal_id: 2,
            fine_pseudorange: Some(-500_000),
            fine_phase_range: Some(8_000_000),
            lock_time_indicator: Some(700),
            half_cycle_ambiguity: Some(false),
            cnr: Some(800),
            fine_phase_range_rate: Some(-12_000),
        },
        MsmSignal {
            satellite_id: 9,
            signal_id: 2,
            fine_pseudorange: Some(500_000),
            fine_phase_range: Some(-8_000_000),
            lock_time_indicator: Some(1),
            half_cycle_ambiguity: Some(true),
            cnr: Some(640),
            fine_phase_range_rate: Some(16_000),
        },
    ];
    let message = MsmMessage {
        message_number: 1087,
        system: crate::id::GnssSystem::Glonass,
        kind: MsmKind::Msm7,
        header: msm_header(),
        signal_mask: msm_signal_mask(&signals),
        satellites,
        signals,
        trailing_bits: Vec::new(),
    };
    let frame = Message::Msm(message.clone()).to_frame().unwrap();
    let decoded = decode_messages(&frame).unwrap();
    assert_eq!(decoded, vec![Message::Msm(message)]);
}

#[test]
fn msm_kind_maps_constellation_and_type() {
    use crate::id::GnssSystem::*;
    let cases = [
        (1071, Gps, MsmKind::Msm1),
        (1072, Gps, MsmKind::Msm2),
        (1073, Gps, MsmKind::Msm3),
        (1074, Gps, MsmKind::Msm4),
        (1075, Gps, MsmKind::Msm5),
        (1076, Gps, MsmKind::Msm6),
        (1077, Gps, MsmKind::Msm7),
        (1105, Sbas, MsmKind::Msm5),
        (1113, Qzss, MsmKind::Msm3),
        (1131, Navic, MsmKind::Msm1),
        (1136, Navic, MsmKind::Msm6),
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
            signal_mask: 0,
            satellites: Vec::new(),
            signals: Vec::new(),
            trailing_bits: Vec::new(),
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
    let group: u16 = match system {
        crate::id::GnssSystem::Gps => 0,
        crate::id::GnssSystem::Glonass => 1,
        crate::id::GnssSystem::Galileo => 2,
        crate::id::GnssSystem::Sbas => 3,
        crate::id::GnssSystem::Qzss => 4,
        crate::id::GnssSystem::BeiDou => 5,
        crate::id::GnssSystem::Navic => 6,
    };
    let message_number = 1070 + 10 * group + u16::from(kind.number());
    let mut header = msm_header();
    header.epoch_time = epoch_time;
    MsmMessage {
        message_number,
        system,
        kind,
        header,
        signal_mask: 1u32 << (32 - u32::from(signal_id)),
        satellites: vec![MsmSatellite {
            id: satellite_id,
            rough_range_ms: Some(75),
            rough_range_mod1: 512,
            extended_info: (kind == MsmKind::Msm7).then_some(0),
            rough_phase_range_rate_m_s: (kind == MsmKind::Msm7).then_some(0),
        }],
        signals: vec![MsmSignal {
            satellite_id,
            signal_id,
            fine_pseudorange: Some(0),
            fine_phase_range: Some(0),
            lock_time_indicator: Some(lock_time_indicator),
            half_cycle_ambiguity: Some(half_cycle_ambiguity),
            cnr: Some(40),
            fine_phase_range_rate: (kind == MsmKind::Msm7).then_some(0),
        }],
        trailing_bits: Vec::new(),
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
        fine_pseudorange: Some(0),
        fine_phase_range: Some(0),
        lock_time_indicator: Some(6),
        half_cycle_ambiguity: Some(false),
        cnr: Some(0),
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
    // Message 4094 (a proprietary number) is not decoded; build a body whose
    // first 12 bits are 4094 and check it survives a frame round-trip.
    let mut w = BitWriter::new();
    w.push_u(4094, 12);
    w.push_u(0xABCD, 16);
    let body = w.into_bytes();

    let message = Message::decode(&body).unwrap();
    match &message {
        Message::Unsupported(u) => assert_eq!(u.message_number, 4094),
        _ => panic!("expected Unsupported"),
    }
    assert_eq!(message.encode().unwrap(), body);
    assert_eq!(message.message_number(), 4094);

    let frame = message.to_frame().unwrap();
    assert_eq!(decode_messages(&frame).unwrap(), vec![message]);
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
        trailing_bits: Vec::new(),
    });

    let mut stream = station.to_frame().unwrap();
    stream.extend_from_slice(&eph.to_frame().unwrap());

    assert_eq!(decode_messages(&stream).unwrap(), vec![station, eph]);
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
    unsupported_body.push_u(4094, 12);
    unsupported_body.push_u(0xABCD, 16);
    let unsupported = Message::Unsupported(UnsupportedMessage {
        message_number: 4094,
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
    // The strict whole-stream reader refuses a stream it cannot read in full.
    assert!(decode_messages(&stream_bytes).is_err());
    assert_eq!(stream.diagnostics.resync_bytes, garbage.len());
    // The stray 0xD3 declares an empty body whose CRC-24Q fails.
    assert_eq!(stream.diagnostics.crc_failures, 1);
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
    let scanned = decode_messages(&frame).unwrap();
    assert_eq!(scanned, vec![message.clone()]);
    assert_eq!(scanned[0].to_frame().unwrap(), frame);
}

#[test]
fn build_station_from_scratch_round_trips() {
    for (number, height) in [(1005u16, None), (1006u16, Some(15_000u16))] {
        let station = sample_station(number, height);
        // Exercise the public per-type encode/decode directly.
        let body = station.encode().unwrap();
        let decoded = StationCoordinates::decode(&body).unwrap();
        assert_eq!(decoded, station);
        assert_eq!(decoded.encode().unwrap(), body);
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
            trailing_bits: Vec::new(),
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
            trailing_bits: Vec::new(),
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
            trailing_bits: Vec::new(),
        },
    ];
    for descriptor in descriptors {
        let body = descriptor.encode().unwrap();
        let decoded = AntennaDescriptor::decode(&body).unwrap();
        assert_eq!(decoded, descriptor);
        assert_eq!(decoded.encode().unwrap(), body);
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
        trailing_bits: Vec::new(),
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
        trailing_bits: Vec::new(),
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
        trailing_bits: Vec::new(),
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
        trailing_bits: Vec::new(),
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
        trailing_bits: Vec::new(),
    };
    let body = eph.encode().unwrap();
    assert_eq!(body.len(), 61);
    let decoded = QzssEphemeris::decode(&body).unwrap();
    assert_eq!(decoded, eph);
    assert_eq!(decoded.encode().unwrap(), body);
    assert_round_trips(Message::QzssEphemeris(eph.clone()));
    assert_broadcast_record_is_nontrivial(decoded.to_broadcast_record(2434).unwrap());

    // The fit flag reads as RTKLIB reads it from RTCM and from RINEX (0: 2 h, 1: 4 h), so
    // the record written to RINEX reads back with the same flag and fit interval.
    for (flag, hours) in [(false, 2.0), (true, 4.0)] {
        let record = QzssEphemeris {
            fit_interval: flag,
            ..eph.clone()
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
            rough_range_ms: Some(75),
            rough_range_mod1: 512,
            extended_info: None,
            rough_phase_range_rate_m_s: None,
        },
        MsmSatellite {
            id: 14,
            rough_range_ms: Some(80),
            rough_range_mod1: 1000,
            extended_info: None,
            rough_phase_range_rate_m_s: None,
        },
    ];
    let signals = vec![
        MsmSignal {
            satellite_id: 3,
            signal_id: 2,
            fine_pseudorange: Some(-4000),
            fine_phase_range: Some(100_000),
            lock_time_indicator: Some(9),
            half_cycle_ambiguity: Some(false),
            cnr: Some(45),
            fine_phase_range_rate: None,
        },
        MsmSignal {
            satellite_id: 14,
            signal_id: 2,
            fine_pseudorange: Some(16),
            fine_phase_range: Some(-7),
            lock_time_indicator: Some(15),
            half_cycle_ambiguity: Some(true),
            cnr: Some(50),
            fine_phase_range_rate: None,
        },
    ];
    let message = MsmMessage {
        message_number: 1074,
        system: crate::id::GnssSystem::Gps,
        kind: MsmKind::Msm4,
        header: msm_header(),
        signal_mask: msm_signal_mask(&signals),
        satellites,
        signals,
        trailing_bits: Vec::new(),
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
        rough_range_ms: Some(70),
        rough_range_mod1: 256,
        extended_info: Some(8),
        rough_phase_range_rate_m_s: Some(-1500),
    }];
    let signals = vec![MsmSignal {
        satellite_id: 1,
        signal_id: 2,
        fine_pseudorange: Some(-500_000),
        fine_phase_range: Some(8_000_000),
        lock_time_indicator: Some(700),
        half_cycle_ambiguity: Some(false),
        cnr: Some(800),
        fine_phase_range_rate: Some(-12_000),
    }];
    let message = MsmMessage {
        message_number: 1077,
        system: crate::id::GnssSystem::Gps,
        kind: MsmKind::Msm7,
        header: msm_header(),
        signal_mask: msm_signal_mask(&signals),
        satellites,
        signals,
        trailing_bits: Vec::new(),
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
            rough_range_ms: Some(70),
            rough_range_mod1: 256,
            extended_info: Some(0),
            rough_phase_range_rate_m_s: None,
        },
        MsmSatellite {
            id: 2,
            rough_range_ms: Some(72),
            rough_range_mod1: 512,
            extended_info: Some(0),
            rough_phase_range_rate_m_s: Some(0),
        },
    ];
    let signals = vec![
        MsmSignal {
            satellite_id: 1,
            signal_id: 1,
            fine_pseudorange: Some(100),
            fine_phase_range: Some(200),
            lock_time_indicator: Some(50),
            half_cycle_ambiguity: Some(false),
            cnr: Some(400),
            fine_phase_range_rate: None,
        },
        MsmSignal {
            satellite_id: 2,
            signal_id: 1,
            fine_pseudorange: Some(300),
            fine_phase_range: Some(400),
            lock_time_indicator: Some(60),
            half_cycle_ambiguity: Some(false),
            cnr: Some(450),
            fine_phase_range_rate: Some(0),
        },
    ];
    let message = MsmMessage {
        message_number: 1077,
        system: crate::id::GnssSystem::Gps,
        kind: MsmKind::Msm7,
        header: msm_header(),
        signal_mask: msm_signal_mask(&signals),
        satellites,
        signals,
        trailing_bits: Vec::new(),
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
        Message::LegacyObservations(o) => o.message_number,
        Message::StationCoordinates(s) => s.message_number,
        Message::AntennaDescriptor(a) => a.message_number,
        Message::SystemParameters(_) => 1013,
        Message::Text(_) => 1029,
        Message::NetworkAuxiliaryStation(_) => 1014,
        Message::NetworkCorrectionDifferences(m) => m.message_number,
        Message::HelmertTransformation(m) => m.message_number,
        Message::ResidualGrid(m) => m.message_number,
        Message::Projection(m) => m.message_number(),
        Message::NetworkResiduals(m) => m.message_number,
        Message::PhysicalReferenceStation(_) => 1032,
        Message::FkpGradients(m) => m.message_number,
        Message::GpsEphemeris(_) => 1019,
        Message::GlonassEphemeris(_) => 1020,
        Message::NavicEphemeris(_) => 1041,
        Message::GlonassCodePhaseBiases(_) => 1230,
        Message::BeidouEphemeris(_) => 1042,
        Message::QzssEphemeris(_) => 1044,
        Message::GalileoFnavEphemeris(_) => 1045,
        Message::GalileoInavEphemeris(_) => 1046,
        Message::Ssr(s) => s.message_number,
        Message::SsrVtec(v) => v.message_number,
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
        trailing_bits: Vec::new(),
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
        trailing_bits: Vec::new(),
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
        trailing_bits: Vec::new(),
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
        trailing_bits: Vec::new(),
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
        trailing_bits: Vec::new(),
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
        negative_zero: 0,
        trailing_bits: Vec::new(),
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
            matches!(err, Error::RtcmConversion(_)),
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
                matches!(err, Error::RtcmConversion(_)),
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
            matches!(err, Error::RtcmEncode(ref e)
                if e.to_string().contains("raw satellite field") && e.to_string().contains(message)),
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
        Err(Error::RtcmConversion(refusal)) => {
            let msg = refusal.to_string();
            assert!(msg.contains("GPS"));
            assert!(msg.contains("URA index 15"));
            assert!(msg.contains("no accuracy prediction"));
        }
        other => panic!("expected RtcmConversion for GPS URA 15 absence, got {other:?}"),
    }
    gps_eph.sv_accuracy = 16;
    match gps_eph.to_broadcast_record(2434) {
        Err(Error::RtcmConversion(refusal)) => {
            let msg = refusal.to_string();
            assert!(msg.contains("GPS"));
            assert!(msg.contains("URA index 16"));
            assert!(msg.contains("exceeds 4-bit range"));
        }
        other => panic!("expected RtcmConversion for GPS URA 16 out of range, got {other:?}"),
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
        Err(Error::RtcmConversion(refusal)) => {
            let msg = refusal.to_string();
            assert!(msg.contains("BeiDou"));
            assert!(msg.contains("URA index 15"));
            assert!(msg.contains("no accuracy prediction"));
        }
        other => panic!("expected RtcmConversion for BeiDou URA 15 absence, got {other:?}"),
    }
    bds_eph.sv_urai = 16;
    match bds_eph.to_broadcast_record() {
        Err(Error::RtcmConversion(refusal)) => {
            let msg = refusal.to_string();
            assert!(msg.contains("BeiDou"));
            assert!(msg.contains("URA index 16"));
            assert!(msg.contains("exceeds 4-bit range"));
        }
        other => panic!("expected RtcmConversion for BeiDou URA 16 out of range, got {other:?}"),
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
        Err(Error::RtcmConversion(refusal)) => {
            let msg = refusal.to_string();
            assert!(msg.contains("QZSS"));
            assert!(msg.contains("URA index 15"));
            assert!(msg.contains("no accuracy prediction"));
        }
        other => panic!("expected RtcmConversion for QZSS URA 15 absence, got {other:?}"),
    }
    qzss_eph.ura = 16;
    match qzss_eph.to_broadcast_record(2434) {
        Err(Error::RtcmConversion(refusal)) => {
            let msg = refusal.to_string();
            assert!(msg.contains("QZSS"));
            assert!(msg.contains("URA index 16"));
            assert!(msg.contains("exceeds 4-bit range"));
        }
        other => panic!("expected RtcmConversion for QZSS URA 16 out of range, got {other:?}"),
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
            Err(Error::RtcmConversion(refusal)) => {
                let msg = refusal.to_string();
                assert!(msg.contains("Galileo"));
                assert!(msg.contains(&format!("SISA index {spare_idx}")));
                assert!(msg.contains("spare with no defined accuracy"));
            }
            other => {
                panic!("expected spare RtcmConversion for F/NAV SISA {spare_idx}, got {other:?}")
            }
        }
        inav.sisa_index = spare_idx;
        match inav.to_broadcast_record() {
            Err(Error::RtcmConversion(refusal)) => {
                let msg = refusal.to_string();
                assert!(msg.contains("Galileo"));
                assert!(msg.contains(&format!("SISA index {spare_idx}")));
                assert!(msg.contains("spare with no defined accuracy"));
            }
            other => {
                panic!("expected spare RtcmConversion for I/NAV SISA {spare_idx}, got {other:?}")
            }
        }
    }

    // NAPA index (255)
    fnav.sisa = 255;
    match fnav.to_broadcast_record() {
        Err(Error::RtcmConversion(refusal)) => {
            let msg = refusal.to_string();
            assert!(msg.contains("Galileo"));
            assert!(msg.contains("SISA index 255"));
            assert!(msg.contains("no accuracy prediction available (NAPA)"));
        }
        other => panic!("expected NAPA RtcmConversion for F/NAV SISA 255, got {other:?}"),
    }
    inav.sisa_index = 255;
    match inav.to_broadcast_record() {
        Err(Error::RtcmConversion(refusal)) => {
            let msg = refusal.to_string();
            assert!(msg.contains("Galileo"));
            assert!(msg.contains("SISA index 255"));
            assert!(msg.contains("no accuracy prediction available (NAPA)"));
        }
        other => panic!("expected NAPA RtcmConversion for I/NAV SISA 255, got {other:?}"),
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

// ---------------------------------------------------------------------------
// Format audit: framing diagnostics, departures and policy
// ---------------------------------------------------------------------------

fn unsupported_4094_frame() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.push_u(4094, 12);
    w.push_u(0xABCD, 16);
    encode_frame(&w.into_bytes()).unwrap()
}

/// The six reserved header bits are read into `DecodedFrame::reserved` and
/// written back by `encode_frame_with_reserved`. A nonzero value is refused by
/// name under the strict policy and read and reported under the lenient one;
/// it was ignored and dropped before.
#[test]
fn frame_reserved_bits_are_kept_refused_strictly_and_reported_leniently() {
    let station = sample_station(1005, None);
    let body = station.encode().unwrap();
    let frame = encode_frame_with_reserved(&body, 5).unwrap();
    assert_eq!(frame[1] >> 2, 5);
    let decoded = decode_frame(&frame).unwrap();
    assert_eq!(decoded.reserved, 5);
    assert_eq!(decoded.body, body.as_slice());
    assert_eq!(
        encode_frame_with_reserved(decoded.body, decoded.reserved).unwrap(),
        frame
    );
    assert_eq!(
        decode_frame(&encode_frame(&body).unwrap())
            .unwrap()
            .reserved,
        0
    );
    assert!(encode_frame_with_reserved(&body, 64).is_err());

    let departure = RtcmDeparture::FrameReservedBits { reserved: 5 };
    let strict = decode_stream(&frame);
    assert!(strict.messages.is_empty());
    assert_eq!(
        strict.diagnostics.skipped_frames,
        vec![FrameSkip {
            offset: 0,
            message_number: Some(1005),
            reason: FrameSkipReason::Departure(departure.clone()),
        }]
    );
    assert!(decode_messages(&frame).is_err());

    let lenient = decode_stream_with_policy(&frame, RtcmPolicy::Lenient);
    assert_eq!(lenient.messages, vec![Message::StationCoordinates(station)]);
    assert!(lenient.diagnostics.skipped_frames.is_empty());
    assert_eq!(
        lenient.diagnostics.departures,
        vec![StreamDeparture {
            offset: 0,
            departure,
        }]
    );
}

/// A preamble whose declared frame lies in the buffer but fails its CRC-24Q is
/// counted, by the stream decoder, the frame scanner and the chunked
/// assembler alike, and its bytes are counted as resynchronized. The
/// assembler counted neither before.
#[test]
fn crc_failures_and_resync_bytes_are_counted_by_every_scanner() {
    let a = Message::StationCoordinates(sample_station(1005, None))
        .to_frame()
        .unwrap();
    let mut bad = unsupported_4094_frame();
    let last = bad.len() - 1;
    bad[last] ^= 0x01;
    assert!(
        !bad[1..].contains(&PREAMBLE),
        "one preamble in the bad frame"
    );
    let c = unsupported_4094_frame();
    let mut bytes = vec![0x00, 0x01];
    bytes.extend_from_slice(&a);
    bytes.extend_from_slice(&bad);
    bytes.extend_from_slice(&c);

    let stream = decode_stream(&bytes);
    assert_eq!(stream.messages.len(), 2);
    assert_eq!(stream.diagnostics.crc_failures, 1);
    assert_eq!(stream.diagnostics.resync_bytes, 2 + bad.len());
    assert!(!stream.diagnostics.is_clean());

    let mut scanner = FrameScanner::new(&bytes);
    assert_eq!(scanner.by_ref().count(), 2);
    assert_eq!(scanner.crc_failures(), 1);
    assert_eq!(scanner.resync_bytes(), 2 + bad.len());

    let mut assembler = SsrStreamAssembler::new();
    let split = 2 + a.len() + 3;
    let mut decoded = assembler.push(&bytes[..split]);
    decoded.extend(assembler.push(&bytes[split..]));
    assert_eq!(decoded.len(), 2);
    assert!(decoded.iter().all(|result| result.is_ok()));
    assert_eq!(assembler.diagnostics().crc_failures, 1);
    assert_eq!(assembler.diagnostics().resync_bytes, 2 + bad.len());
    assert_eq!(assembler.retained_len(), 0);

    let err = decode_messages(&bytes).expect_err("the stream is not read in full");
    assert!(err.to_string().contains("1 CRC-24Q failures"), "{err}");
    assert_eq!(decode_messages(&a).unwrap().len(), 1);
}

/// The chunked assembler reports lenient departures with offsets counted from
/// the first byte pushed, across chunks, and refuses them under the strict
/// policy.
#[test]
fn assembler_departures_carry_stream_offsets_across_chunks() {
    let body = sample_station(1005, None).encode().unwrap();
    let plain = encode_frame(&body).unwrap();
    let marked = encode_frame_with_reserved(&body, 1).unwrap();

    let mut lenient = SsrStreamAssembler::with_policy(RtcmPolicy::Lenient);
    assert_eq!(lenient.push(&plain).len(), 1);
    let mut chunk = vec![0xAA];
    chunk.extend_from_slice(&marked);
    let out = lenient.push(&chunk);
    assert_eq!(out.len(), 1);
    assert!(out[0].is_ok());
    assert_eq!(
        lenient.diagnostics().departures,
        vec![StreamDeparture {
            offset: plain.len() + 1,
            departure: RtcmDeparture::FrameReservedBits { reserved: 1 },
        }]
    );
    assert_eq!(lenient.diagnostics().resync_bytes, 1);

    let mut strict = SsrStreamAssembler::new();
    let out = strict.push(&marked);
    assert_eq!(out.len(), 1);
    let err = out[0]
        .as_ref()
        .expect_err("strict refuses the reserved bits");
    assert!(err.to_string().contains("reserved bits"), "{err}");
}

/// Bits after a message's last field other than the fewer than eight zero bits
/// that align the body are a departure: refused by name under the strict
/// policy, read under the lenient one into the message's `trailing_bits`, and
/// written back by the lenient encoder so the body re-encodes byte for byte.
/// They were dropped without a trace before, so a decode followed by an encode
/// silently shortened the body.
#[test]
fn trailing_bits_after_the_last_field_are_a_departure() {
    // 1005 fills exactly 152 bits; one more byte is eight trailing bits.
    let station = sample_station(1005, None);
    let mut body = station.encode().unwrap();
    body.push(0x00);
    let err = StationCoordinates::decode(&body).expect_err("strict per-type decode");
    assert!(
        err.to_string().contains("8 bits after its last field"),
        "{err}"
    );
    assert!(Message::decode(&body).is_err());
    let departure = RtcmDeparture::TrailingBits {
        message_number: 1005,
        bits: vec![false; 8],
    };
    let (message, departures) = Message::decode_with_policy(&body, RtcmPolicy::Lenient).unwrap();
    let mut expected = station.clone();
    expected.trailing_bits = vec![false; 8];
    assert_eq!(message, Message::StationCoordinates(expected.clone()));
    assert_eq!(departures, vec![departure.clone()]);
    let err = message
        .encode()
        .expect_err("strict encode refuses the tail");
    assert!(err.to_string().contains("strict policy"), "{err}");
    assert_eq!(
        message.encode_with_policy(RtcmPolicy::Lenient).unwrap(),
        (body.clone(), vec![departure])
    );
    assert_eq!(
        expected.encode_with_policy(RtcmPolicy::Lenient).unwrap().0,
        body
    );
    let frame = encode_frame(&body).unwrap();
    assert!(matches!(
        decode_stream(&frame).diagnostics.skipped_frames[0].reason,
        FrameSkipReason::Departure(RtcmDeparture::TrailingBits { .. })
    ));
    let lenient = decode_stream_with_policy(&frame, RtcmPolicy::Lenient);
    assert_eq!(
        encode_frame(
            &lenient.messages[0]
                .encode_with_policy(RtcmPolicy::Lenient)
                .unwrap()
                .0
        )
        .unwrap(),
        frame
    );

    // 1044 fills 485 bits, so three pad bits close its 61 bytes; a set pad bit
    // is a departure, three zero bits are not.
    let qzss = valid_qzss_ephemeris();
    let body = qzss.encode().unwrap();
    assert_eq!(body.len(), 61);
    assert_eq!(QzssEphemeris::decode(&body).unwrap(), qzss);
    let mut marked = body.clone();
    marked[60] |= 0x01;
    assert!(QzssEphemeris::decode(&marked).is_err());
    let (message, departures) = Message::decode_with_policy(&marked, RtcmPolicy::Lenient).unwrap();
    let mut expected = qzss.clone();
    expected.trailing_bits = vec![false, false, true];
    assert_eq!(message, Message::QzssEphemeris(expected));
    assert_eq!(
        departures,
        vec![RtcmDeparture::TrailingBits {
            message_number: 1044,
            bits: vec![false, false, true],
        }]
    );
    assert_eq!(
        message.encode_with_policy(RtcmPolicy::Lenient).unwrap().0,
        marked
    );

    // Every other type keeps and writes back its tail the same way.
    let mut msm = msm4(vec![msm4_satellite(3)], vec![msm4_signal(3, 2)]);
    let mut body = msm.encode().unwrap();
    body.extend_from_slice(&[0xA5, 0x00]);
    let (message, _) = Message::decode_with_policy(&body, RtcmPolicy::Lenient).unwrap();
    assert_eq!(
        message.encode_with_policy(RtcmPolicy::Lenient).unwrap().0,
        body
    );
    let Message::Msm(read) = &message else {
        panic!("expected MSM");
    };
    assert!(read.trailing_bits.len() >= 16);
    // A tail of zero bits that would read back as the alignment alone is
    // refused under both policies, since it would not be read back.
    msm.trailing_bits = vec![false];
    assert!(msm.encode_with_policy(RtcmPolicy::Lenient).is_err());
    for body in [
        valid_gps_ephemeris().encode().unwrap(),
        valid_glonass_ephemeris().encode().unwrap(),
        valid_beidou_ephemeris().encode().unwrap(),
        valid_galileo_fnav_ephemeris().encode().unwrap(),
        valid_galileo_inav_ephemeris().encode().unwrap(),
        sample_station(1006, Some(7)).encode().unwrap(),
        AntennaDescriptor {
            message_number: 1033,
            reference_station_id: 4095,
            antenna_descriptor: "LEIAR25.R4      LEIT".to_string(),
            antenna_setup_id: 0,
            antenna_serial_number: Some("09120119".to_string()),
            receiver_type: Some("LEICA GR50".to_string()),
            receiver_firmware_version: Some("4.50".to_string()),
            receiver_serial_number: Some("1830080".to_string()),
            trailing_bits: Vec::new(),
        }
        .encode()
        .unwrap(),
    ] {
        let mut body = body;
        body.push(0x80);
        assert!(Message::decode(&body).is_err());
        let (message, departures) =
            Message::decode_with_policy(&body, RtcmPolicy::Lenient).unwrap();
        assert_eq!(departures.len(), 1, "{}", message.message_number());
        assert_eq!(
            message.encode_with_policy(RtcmPolicy::Lenient).unwrap().0,
            body,
            "{}",
            message.message_number()
        );
    }

    // SSR keeps its tail in `padding_bits` and is held to the same rule.
    let ssr = Message::Ssr(crate::rtcm::SsrMessage {
        message_number: 1058,
        igs_ssr_version: None,
        system: crate::id::GnssSystem::Gps,
        kind: SsrKind::Clock,
        header: SsrHeader {
            epoch_time_s: 1,
            update_interval: 0,
            multiple_message: false,
            iod_ssr: 0,
            provider_id: 0,
            solution_id: 0,
            satellite_reference_datum: None,
            dispersive_bias_consistency: None,
            mw_consistency: None,
            satellite_count: 0,
        },
        orbit: Vec::new(),
        clock: Vec::new(),
        code_bias: Vec::new(),
        phase_bias: Vec::new(),
        ura: Vec::new(),
        padding_bits: Vec::new(),
    });
    let mut body = ssr.encode().unwrap();
    body.push(0x00);
    assert!(Message::decode(&body).is_err());
    let (message, departures) = Message::decode_with_policy(&body, RtcmPolicy::Lenient).unwrap();
    assert!(matches!(
        departures[..],
        [RtcmDeparture::TrailingBits { .. }]
    ));
    assert!(message.encode().is_err());
    assert_eq!(
        message.encode_with_policy(RtcmPolicy::Lenient).unwrap().0,
        body
    );
}

fn msm4_satellite(id: u8) -> MsmSatellite {
    MsmSatellite {
        id,
        rough_range_ms: Some(70),
        rough_range_mod1: 256,
        extended_info: None,
        rough_phase_range_rate_m_s: None,
    }
}

fn msm4_signal(satellite_id: u8, signal_id: u8) -> MsmSignal {
    MsmSignal {
        satellite_id,
        signal_id,
        fine_pseudorange: Some(16),
        fine_phase_range: Some(-7),
        lock_time_indicator: Some(15),
        half_cycle_ambiguity: Some(false),
        cnr: Some(50),
        fine_phase_range_rate: None,
    }
}

fn msm4(satellites: Vec<MsmSatellite>, signals: Vec<MsmSignal>) -> MsmMessage {
    MsmMessage {
        message_number: 1074,
        system: crate::id::GnssSystem::Gps,
        kind: MsmKind::Msm4,
        header: msm_header(),
        signal_mask: msm_signal_mask(&signals),
        satellites,
        signals,
        trailing_bits: Vec::new(),
    }
}

/// A signal-mask bit with no cell is kept in `signal_mask` and written back.
/// The encoder rebuilt the mask from the cells before, so such a message
/// re-encoded with the bit cleared and a narrower cell mask.
#[test]
fn msm_signal_mask_bit_without_cells_round_trips() {
    let mut message = msm4(vec![msm4_satellite(3)], vec![msm4_signal(3, 2)]);
    // Signal 5 is listed but carries no cell.
    message.signal_mask |= 1 << (32 - 5);
    let body = message.encode().unwrap();
    let decoded = MsmMessage::decode(&body).unwrap();
    assert_eq!(decoded, message);
    assert_eq!(decoded.encode().unwrap(), body);

    // A cell whose signal is not in the mask is refused, not added to it.
    let mut unlisted = message.clone();
    unlisted.signal_mask = 1 << (32 - 5);
    let err = unlisted.encode().expect_err("signal 2 is not in the mask");
    assert!(
        err.to_string().contains("not set in the signal mask"),
        "{err}"
    );
}

/// A cell mask over 64 bits departs from RTCM 10403, and RTKLIB
/// `decode_msm_head` refuses it. The strict policy refuses it on both decode
/// and encode; the lenient policy reads and writes it and reports it.
#[test]
fn msm_cell_mask_over_64_bits_is_a_departure() {
    // Nine satellites by eight signals is 72 cells.
    let satellites: Vec<_> = (1..=9).map(msm4_satellite).collect();
    let signals: Vec<_> = (1..=9).map(|id| msm4_signal(id, id.min(8))).collect();
    let mut message = msm4(satellites, signals);
    message.signal_mask = 0xFF00_0000;
    let departure = RtcmDeparture::MsmCellMaskOver64 {
        message_number: 1074,
        cells: 72,
    };

    let err = message
        .encode()
        .expect_err("strict encode refuses 72 cells");
    assert!(err.to_string().contains("72 bits"), "{err}");
    let (body, departures) = message.encode_with_policy(RtcmPolicy::Lenient).unwrap();
    assert_eq!(departures, vec![departure.clone()]);

    assert!(MsmMessage::decode(&body).is_err());
    assert!(Message::decode(&body).is_err());
    let (decoded, departures) = Message::decode_with_policy(&body, RtcmPolicy::Lenient).unwrap();
    assert_eq!(decoded, Message::Msm(message.clone()));
    assert_eq!(departures, vec![departure.clone()]);
    assert_eq!(
        decoded.encode_with_policy(RtcmPolicy::Lenient).unwrap().0,
        body
    );

    let frame = encode_frame(&body).unwrap();
    assert_eq!(
        decode_stream(&frame).diagnostics.skipped_frames[0].reason,
        FrameSkipReason::Departure(departure)
    );
    assert_eq!(
        decode_stream_with_policy(&frame, RtcmPolicy::Lenient).messages,
        vec![Message::Msm(message)]
    );
}

/// The MSM encoder refuses by name every value it would otherwise truncate,
/// fill or drop, and a message number that names another layout.
#[test]
fn msm_encode_refuses_values_it_would_truncate_fill_or_drop() {
    use crate::id::GnssSystem;

    let base = msm4(vec![msm4_satellite(3)], vec![msm4_signal(3, 2)]);
    base.encode().expect("the base message encodes");
    let refused = |edit: &dyn Fn(&mut MsmMessage), expected: RtcmEncodeError| {
        let mut m = base.clone();
        edit(&mut m);
        let err = m.encode().expect_err("invalid MSM must be refused");
        let Error::RtcmEncode(actual) = err else {
            panic!("expected a typed RTCM encode refusal, got {err}");
        };
        assert_eq!(*actual, expected);
    };
    refused(
        &|m| m.message_number = 1077,
        RtcmEncodeError::MessageNumber {
            message_number: 1077,
            record: RtcmRecordKind::Msm {
                system: GnssSystem::Gps,
                kind: MsmKind::Msm4,
            },
        },
    );
    refused(
        &|m| m.kind = MsmKind::Msm7,
        RtcmEncodeError::MessageNumber {
            message_number: 1074,
            record: RtcmRecordKind::Msm {
                system: GnssSystem::Gps,
                kind: MsmKind::Msm7,
            },
        },
    );
    refused(
        &|m| m.header.reference_station_id = 4096,
        RtcmEncodeError::FieldOutOfRange {
            message_number: 1074,
            field: "reference station ID".into(),
            value: 4096,
            width: 12,
            encoding: RtcmFieldEncoding::Unsigned,
        },
    );
    refused(
        &|m| m.header.epoch_time = 1 << 30,
        RtcmEncodeError::FieldOutOfRange {
            message_number: 1074,
            field: "epoch time".into(),
            value: 1 << 30,
            width: 30,
            encoding: RtcmFieldEncoding::Unsigned,
        },
    );
    refused(
        &|m| m.header.iods = 8,
        RtcmEncodeError::FieldOutOfRange {
            message_number: 1074,
            field: "IODS".into(),
            value: 8,
            width: 3,
            encoding: RtcmFieldEncoding::Unsigned,
        },
    );
    refused(
        &|m| m.header.smoothing_interval = 8,
        RtcmEncodeError::FieldOutOfRange {
            message_number: 1074,
            field: "smoothing interval".into(),
            value: 8,
            width: 3,
            encoding: RtcmFieldEncoding::Unsigned,
        },
    );
    refused(
        &|m| m.satellites[0].rough_range_mod1 = 1024,
        RtcmEncodeError::FieldOutOfRange {
            message_number: 1074,
            field: "satellite 3 rough range modulo 1 ms".into(),
            value: 1024,
            width: 10,
            encoding: RtcmFieldEncoding::Unsigned,
        },
    );
    let mut extended = base.clone();
    extended.satellites[0].extended_info = Some(1);
    assert!(matches!(
        extended.encode(),
        Err(Error::RtcmEncode(error))
            if matches!(
                *error,
                RtcmEncodeError::MsmOptional {
                    message_number: 1074,
                    kind: MsmKind::Msm4,
                    satellite: 3,
                    signal: None,
                    field: MsmOptionalField::ExtendedInfo,
                    problem: MsmOptionalProblem::NotCarried,
                }
            )
    ));
    refused(
        &|m| m.satellites[0].rough_phase_range_rate_m_s = Some(1),
        RtcmEncodeError::MsmOptional {
            message_number: 1074,
            kind: MsmKind::Msm4,
            satellite: 3,
            signal: None,
            field: MsmOptionalField::RoughPhaseRangeRate,
            problem: MsmOptionalProblem::NotCarried,
        },
    );
    refused(
        &|m| m.signals[0].fine_phase_range_rate = Some(1),
        RtcmEncodeError::MsmOptional {
            message_number: 1074,
            kind: MsmKind::Msm4,
            satellite: 3,
            signal: Some(2),
            field: MsmOptionalField::FinePhaseRangeRate,
            problem: MsmOptionalProblem::NotCarried,
        },
    );
    refused(
        &|m| m.signals[0].fine_pseudorange = Some(1 << 14),
        RtcmEncodeError::FieldOutOfRange {
            message_number: 1074,
            field: "satellite 3 signal 2 fine pseudorange".into(),
            value: 1 << 14,
            width: 15,
            encoding: RtcmFieldEncoding::TwosComplement,
        },
    );
    refused(
        &|m| m.signals[0].fine_phase_range = Some(-(1 << 21) - 1),
        RtcmEncodeError::FieldOutOfRange {
            message_number: 1074,
            field: "satellite 3 signal 2 fine phase range".into(),
            value: -(1 << 21) - 1,
            width: 22,
            encoding: RtcmFieldEncoding::TwosComplement,
        },
    );
    refused(
        &|m| m.signals[0].lock_time_indicator = Some(16),
        RtcmEncodeError::FieldOutOfRange {
            message_number: 1074,
            field: "satellite 3 signal 2 lock-time indicator".into(),
            value: 16,
            width: 4,
            encoding: RtcmFieldEncoding::Unsigned,
        },
    );
    refused(
        &|m| m.signals[0].cnr = Some(64),
        RtcmEncodeError::FieldOutOfRange {
            message_number: 1074,
            field: "satellite 3 signal 2 CNR".into(),
            value: 64,
            width: 6,
            encoding: RtcmFieldEncoding::Unsigned,
        },
    );

    // MSM7: extended info is carried and must be given; `Some` of an invalid
    // value is the spelling of `None` and is refused.
    let mut msm7 = base.clone();
    msm7.message_number = 1077;
    msm7.kind = MsmKind::Msm7;
    msm7.satellites[0].extended_info = Some(0);
    msm7.encode().expect("a well-formed MSM7 encodes");
    let mut missing = msm7.clone();
    missing.satellites[0].extended_info = None;
    assert!(matches!(
        missing.encode(),
        Err(Error::RtcmEncode(error))
            if matches!(
                *error,
                RtcmEncodeError::MsmOptional {
                    message_number: 1077,
                    kind: MsmKind::Msm7,
                    satellite: 3,
                    signal: None,
                    field: MsmOptionalField::ExtendedInfo,
                    problem: MsmOptionalProblem::Missing,
                }
            )
    ));
    let mut wide = msm7.clone();
    wide.satellites[0].extended_info = Some(16);
    assert!(matches!(
        wide.encode(),
        Err(Error::RtcmEncode(error))
            if matches!(
                *error,
                RtcmEncodeError::FieldOutOfRange {
                    message_number: 1077,
                    ref field,
                    value: 16,
                    width: 4,
                    encoding: RtcmFieldEncoding::Unsigned,
                } if field == "satellite 3 extended info"
            )
    ));
    let mut rough = msm7.clone();
    rough.satellites[0].rough_phase_range_rate_m_s = Some(MSM_ROUGH_PHASE_RANGE_RATE_INVALID);
    assert!(matches!(
        rough.encode(),
        Err(Error::RtcmEncode(error))
            if matches!(
                *error,
                RtcmEncodeError::MsmOptional {
                    message_number: 1077,
                    kind: MsmKind::Msm7,
                    satellite: 3,
                    signal: None,
                    field: MsmOptionalField::RoughPhaseRangeRate,
                    problem: MsmOptionalProblem::InvalidValue(value),
                } if value == i64::from(MSM_ROUGH_PHASE_RANGE_RATE_INVALID)
            )
    ));
    let mut fine = msm7.clone();
    fine.signals[0].fine_phase_range_rate = Some(MSM_FINE_PHASE_RANGE_RATE_INVALID);
    assert!(matches!(
        fine.encode(),
        Err(Error::RtcmEncode(error))
            if matches!(
                *error,
                RtcmEncodeError::MsmOptional {
                    message_number: 1077,
                    kind: MsmKind::Msm7,
                    satellite: 3,
                    signal: Some(2),
                    field: MsmOptionalField::FinePhaseRangeRate,
                    problem: MsmOptionalProblem::InvalidValue(value),
                } if value == i64::from(MSM_FINE_PHASE_RANGE_RATE_INVALID)
            )
    ));
    let mut cnr = msm7.clone();
    cnr.signals[0].cnr = Some(1024);
    assert!(matches!(
        cnr.encode(),
        Err(Error::RtcmEncode(error))
            if matches!(
                *error,
                RtcmEncodeError::FieldOutOfRange {
                    message_number: 1077,
                    ref field,
                    value: 1024,
                    width: 10,
                    encoding: RtcmFieldEncoding::Unsigned,
                } if field == "satellite 3 signal 2 CNR"
            )
    ));
    let mut prr = msm7;
    prr.signals[0].fine_phase_range_rate = Some(i16::MAX);
    assert!(matches!(
        prr.encode(),
        Err(Error::RtcmEncode(error))
            if matches!(
                *error,
                RtcmEncodeError::FieldOutOfRange {
                    message_number: 1077,
                    ref field,
                    value: 32767,
                    width: 15,
                    encoding: RtcmFieldEncoding::TwosComplement,
                } if field == "satellite 3 signal 2 fine phase-range rate"
            )
    ));
}

/// The MSM invalid values decode as transmitted and are exported by name.
#[test]
fn msm_invalid_values_round_trip_as_transmitted() {
    let mut message = msm4(vec![msm4_satellite(3)], vec![msm4_signal(3, 2)]);
    message.satellites[0].rough_range_ms = Some(MSM_ROUGH_RANGE_INVALID);
    message.signals[0].fine_pseudorange = Some(MSM4_FINE_PSEUDORANGE_INVALID);
    message.signals[0].fine_phase_range = Some(MSM4_FINE_PHASE_RANGE_INVALID);
    let body = message.encode().unwrap();
    assert_eq!(MsmMessage::decode(&body).unwrap(), message);

    let mut msm7 = message.clone();
    msm7.message_number = 1077;
    msm7.kind = MsmKind::Msm7;
    msm7.satellites[0].extended_info = Some(0);
    msm7.signals[0].fine_pseudorange = Some(MSM7_FINE_PSEUDORANGE_INVALID);
    msm7.signals[0].fine_phase_range = Some(MSM7_FINE_PHASE_RANGE_INVALID);
    let body = msm7.encode().unwrap();
    assert_eq!(MsmMessage::decode(&body).unwrap(), msm7);
}

/// A GLONASS sign-magnitude field transmitted as negative zero reads as `0`,
/// as RTKLIB `getbitg` reads it, and its sign is kept in `negative_zero` so the
/// body re-encodes as transmitted. It was written back with the sign clear.
#[test]
fn glonass_negative_zero_is_kept_and_written_back() {
    let mut eph = valid_glonass_ephemeris();
    eph.delta_tau_n = 0;
    let body = eph.encode().unwrap();
    // DF125 delta_tau_n starts at bit 253: its sign bit is bit 5 of byte 31.
    let mut flipped = body.clone();
    assert_eq!(flipped[31] & 0x04, 0);
    flipped[31] |= 0x04;
    let decoded = GlonassEphemeris::decode(&flipped).unwrap();
    assert_eq!(decoded.delta_tau_n, 0);
    assert_eq!(
        decoded.negative_zero,
        GlonassEphemeris::NEGATIVE_ZERO_DELTA_TAU_N
    );
    assert_eq!(decoded.encode().unwrap(), flipped);
    let mut expected = eph.clone();
    expected.negative_zero = GlonassEphemeris::NEGATIVE_ZERO_DELTA_TAU_N;
    assert_eq!(decoded, expected);

    // Negative zero on a nonzero value, or a bit naming no field, is refused.
    let mut contradictory = eph.clone();
    contradictory.negative_zero = GlonassEphemeris::NEGATIVE_ZERO_XN;
    assert!(contradictory
        .encode()
        .unwrap_err()
        .to_string()
        .contains("marked negative zero"));
    let mut undefined = eph;
    undefined.negative_zero = 1 << 14;
    assert!(undefined
        .encode()
        .unwrap_err()
        .to_string()
        .contains("name no sign-magnitude field"));
}

/// Every ephemeris field is written in its own width or refused by name; the
/// encoders kept only the low bits before.
#[test]
fn ephemeris_encoders_refuse_values_wider_than_their_fields() {
    let check = |result: crate::error::Result<Vec<u8>>, needle: &str| {
        let err = result.expect_err(needle);
        assert!(
            matches!(err, Error::RtcmEncode(ref e) if e.to_string().contains(needle)),
            "expected {needle:?}, got {err}"
        );
    };
    let mut gps = valid_gps_ephemeris();
    gps.week_number = 1024;
    check(gps.encode(), "week_number 1024");
    let mut gps = valid_gps_ephemeris();
    gps.idot = 1 << 13;
    check(gps.encode(), "idot 8192");
    let mut gps = valid_gps_ephemeris();
    gps.m0 = 1 << 31;
    check(gps.encode(), "m0 2147483648");
    let mut gps = valid_gps_ephemeris();
    gps.eccentricity = 1 << 32;
    check(gps.encode(), "eccentricity 4294967296");

    let mut glonass = valid_glonass_ephemeris();
    glonass.xn = 1 << 26;
    check(glonass.encode(), "xn 67108864");
    let mut glonass = valid_glonass_ephemeris();
    glonass.frequency_channel = 32;
    check(glonass.encode(), "frequency_channel 32");
    let mut glonass = valid_glonass_ephemeris();
    glonass.tau_c = -(1 << 31);
    check(glonass.encode(), "tau_c -2147483648");

    let mut beidou = valid_beidou_ephemeris();
    beidou.t_oc = 1 << 17;
    check(beidou.encode(), "t_oc 131072");
    let mut qzss = valid_qzss_ephemeris();
    qzss.codes_on_l2 = 4;
    check(qzss.encode(), "codes_on_l2 4");
    let mut fnav = valid_galileo_fnav_ephemeris();
    fnav.a_f0 = 1 << 30;
    check(fnav.encode(), "a_f0 1073741824");
    let mut inav = valid_galileo_inav_ephemeris();
    inav.reserved = 4;
    check(inav.encode(), "reserved 4");
}

/// Station coordinates are written in their own widths, and the antenna
/// height with 1006 only.
#[test]
fn station_encode_refuses_what_it_would_truncate_drop_or_misplace() {
    let refused = |station: StationCoordinates, needle: &str| {
        let err = station.encode().expect_err(needle);
        assert!(
            matches!(err, Error::RtcmEncode(ref e) if e.to_string().contains(needle)),
            "expected {needle:?}, got {err}"
        );
    };
    refused(
        sample_station(1005, Some(1)),
        "1005 carries no antenna height",
    );
    refused(sample_station(1006, None), "1006 carries an antenna height");
    refused(sample_station(1007, None), "is not station coordinates");
    let mut s = sample_station(1005, None);
    s.reference_station_id = 4096;
    refused(s, "reference station ID 4096");
    let mut s = sample_station(1005, None);
    s.itrf_realization_year = 64;
    refused(s, "ITRF realization year 64");
    let mut s = sample_station(1005, None);
    s.quarter_cycle_indicator = 4;
    refused(s, "quarter-cycle indicator 4");
    let mut s = sample_station(1005, None);
    s.ecef_x = 1 << 37;
    refused(s, "ECEF X 137438953472");
    // The 38-bit extremes are written.
    let mut s = sample_station(1005, None);
    s.ecef_y = -(1 << 37);
    s.ecef_z = (1 << 37) - 1;
    assert_eq!(StationCoordinates::decode(&s.encode().unwrap()).unwrap(), s);
}

/// Descriptor strings are read one byte per character, `U+0000`..=`U+00FF`,
/// and written back one byte per character. The encoder wrote UTF-8 before, so
/// a byte above 0x7F came back as two bytes and a longer count.
#[test]
fn antenna_strings_round_trip_every_byte_value() {
    let mut w = BitWriter::new();
    w.push_u(1007, 12);
    w.push_u(1, 12);
    w.push_u(3, 8);
    for byte in [b'A', 0xB0, b'Z'] {
        w.push_u(u64::from(byte), 8);
    }
    w.push_u(0, 8);
    let body = w.into_bytes();
    let decoded = AntennaDescriptor::decode(&body).unwrap();
    assert_eq!(decoded.antenna_descriptor, "A\u{B0}Z");
    assert_eq!(decoded.encode().unwrap(), body);

    let refused = |descriptor: AntennaDescriptor, needle: &str| {
        let err = descriptor.encode().expect_err(needle);
        assert!(
            matches!(err, Error::RtcmEncode(ref e) if e.to_string().contains(needle)),
            "expected {needle:?}, got {err}"
        );
    };
    let mut wide = decoded.clone();
    wide.antenna_descriptor = "A\u{263A}".to_string();
    refused(wide, "is not an 8-bit character");
    let mut long = decoded.clone();
    long.antenna_descriptor = "A".repeat(256);
    refused(long, "character count 256");
    let mut serial = decoded.clone();
    serial.antenna_serial_number = Some("X".to_string());
    refused(serial, "1007 carries no antenna serial number");
    let mut missing = decoded.clone();
    missing.message_number = 1008;
    refused(missing, "1008 carries the antenna serial number");
    let mut receiver = decoded.clone();
    receiver.message_number = 1033;
    receiver.antenna_serial_number = Some(String::new());
    refused(receiver, "1033 carries the receiver type");
    let mut number = decoded.clone();
    number.message_number = 1005;
    refused(number, "is not an antenna descriptor");
    let mut station = decoded;
    station.reference_station_id = 4096;
    refused(station, "reference station ID 4096");
    let mut longest = AntennaDescriptor::decode(&body).unwrap();
    longest.antenna_descriptor = "\u{FF}".repeat(255);
    let body = longest.encode().unwrap();
    assert_eq!(AntennaDescriptor::decode(&body).unwrap(), longest);
}

/// An unsupported message encodes only a body that decodes back to it.
#[test]
fn unsupported_encode_refuses_a_body_that_decodes_as_something_else() {
    let frame = unsupported_4094_frame();
    let body = decode_frame(&frame).unwrap().body.to_vec();
    let ok = UnsupportedMessage {
        message_number: 4094,
        body: body.clone(),
    };
    assert_eq!(ok.encode().unwrap(), body);
    let mismatched = UnsupportedMessage {
        message_number: 4093,
        body: body.clone(),
    };
    assert!(mismatched
        .encode()
        .unwrap_err()
        .to_string()
        .contains("carries message number 4094"));
    let short = UnsupportedMessage {
        message_number: 4094,
        body: vec![0x4C],
    };
    assert!(short.encode().is_err());
    let typed_body = sample_station(1005, None).encode().unwrap();
    let typed = Message::Unsupported(UnsupportedMessage {
        message_number: 1005,
        body: typed_body,
    });
    assert!(typed
        .encode()
        .unwrap_err()
        .to_string()
        .contains("decoded into its typed variant"));
}

/// At the end of a stream the assembler reads past a byte that only looks like
/// a preamble, whose declared frame runs past the data, to the whole frame
/// behind it, and counts the bytes it passes over.
#[test]
fn assembler_finish_reads_past_an_unfinished_preamble() {
    let frame = Message::StationCoordinates(sample_station(1005, None))
        .to_frame()
        .unwrap();
    let mut bytes = vec![PREAMBLE, 0x03, 0xFF];
    bytes.extend_from_slice(&frame);
    let mut assembler = SsrStreamAssembler::new();
    assert!(assembler.push(&bytes).is_empty());
    assert_eq!(assembler.retained_len(), bytes.len());
    let out = assembler.finish();
    assert_eq!(out.len(), 1);
    assert!(out[0].is_ok());
    assert_eq!(assembler.retained_len(), 0);
    assert_eq!(assembler.diagnostics().resync_bytes, 3);
    assert_eq!(assembler.diagnostics().crc_failures, 0);

    // A real partial frame at the end is counted as passed over.
    let mut assembler = SsrStreamAssembler::new();
    assert_eq!(assembler.push(&frame[..frame.len() - 1]).len(), 0);
    assert!(assembler.finish().is_empty());
    assert_eq!(assembler.diagnostics().resync_bytes, frame.len() - 1);
}

/// Every encoder refusal is an `Error::RtcmEncode` whose fields name the
/// message, the field and the value, and whose text is the refusal's message.
#[test]
fn encoder_refusals_are_typed() {
    let refusal = |result: crate::error::Result<Vec<u8>>| match result {
        Err(Error::RtcmEncode(refusal)) => *refusal,
        other => panic!("expected an RTCM encode refusal, got {other:?}"),
    };

    let mut station = sample_station(1005, None);
    station.reference_station_id = 4096;
    let error = refusal(station.encode());
    assert_eq!(
        error,
        RtcmEncodeError::FieldOutOfRange {
            message_number: 1005,
            field: "reference station ID".to_string(),
            value: 4096,
            width: 12,
            encoding: RtcmFieldEncoding::Unsigned,
        }
    );
    assert_eq!(
        error.to_string(),
        "RTCM 1005 reference station ID 4096 does not fit its 12-bit unsigned field (0..=4095)"
    );
    let mut station = sample_station(1005, None);
    station.ecef_x = 1 << 37;
    assert_eq!(
        refusal(station.encode()).to_string(),
        "RTCM 1005 ECEF X 137438953472 does not fit its 38-bit two's-complement field \
         (-137438953472..=137438953471)"
    );

    assert_eq!(
        refusal(sample_station(1005, Some(1)).encode()),
        RtcmEncodeError::FieldPresence {
            message_number: 1005,
            record: RtcmRecordKind::StationCoordinates,
            field: "antenna height",
            carried: false,
        }
    );
    assert_eq!(
        refusal(sample_station(1007, None).encode()),
        RtcmEncodeError::MessageNumber {
            message_number: 1007,
            record: RtcmRecordKind::StationCoordinates,
        }
    );

    let mut gps = valid_gps_ephemeris();
    gps.satellite_id = 64;
    let error = refusal(gps.encode());
    assert_eq!(
        error,
        RtcmEncodeError::SatelliteIdOutOfRange {
            message_number: 1019,
            field: "GPS PRN",
            value: 64,
            width: 6,
        }
    );
    assert_eq!(
        error.to_string(),
        "GPS PRN 64 in 1019 does not fit the 6-bit raw satellite field (0..=63)"
    );

    let error = refusal(msm4(vec![msm4_satellite(3)], vec![msm4_signal(4, 2)]).encode());
    assert_eq!(
        error,
        RtcmEncodeError::MsmMask {
            message_number: 1074,
            problem: MsmMaskProblem::SignalSatelliteNotListed {
                signal: 2,
                satellite: 4,
            },
        }
    );
    let mut msm = msm4(vec![msm4_satellite(3)], vec![msm4_signal(3, 2)]);
    msm.signals[0].fine_phase_range_rate = Some(1);
    assert_eq!(
        refusal(msm.encode()),
        RtcmEncodeError::MsmOptional {
            message_number: 1074,
            kind: MsmKind::Msm4,
            satellite: 3,
            signal: Some(2),
            field: MsmOptionalField::FinePhaseRangeRate,
            problem: MsmOptionalProblem::NotCarried,
        }
    );
    let mut msm = msm4(vec![msm4_satellite(3)], vec![msm4_signal(3, 2)]);
    msm.trailing_bits = vec![true];
    assert!(matches!(
        refusal(msm.encode()),
        RtcmEncodeError::StrictDeparture(RtcmDeparture::TrailingBits {
            message_number: 1074,
            ..
        })
    ));

    assert_eq!(
        refusal(encode_frame(&[0u8; 1024])),
        RtcmEncodeError::FrameBodyTooLong { len: 1024 }
    );
    assert_eq!(
        refusal(encode_frame_with_reserved(&[0x3E, 0xD0], 64)),
        RtcmEncodeError::FrameReservedOutOfRange { value: 64 }
    );
}

/// Every ephemeris conversion refusal is an `Error::RtcmConversion` whose
/// fields name the cause.
#[test]
fn ephemeris_conversion_refusals_are_typed() {
    let refusal = |result: crate::error::Result<crate::rinex_nav::BroadcastRecord>| match result {
        Err(Error::RtcmConversion(refusal)) => *refusal,
        other => panic!("expected an RTCM conversion refusal, got {other:?}"),
    };

    let gps = valid_gps_ephemeris();
    let week = gps.week_number;
    let wrong_week = u32::from(week) + 2 * 1024 + 1;
    assert_eq!(
        refusal(gps.to_broadcast_record(wrong_week)),
        RtcmConversionError::WeekMismatch {
            message_number: 1019,
            full_week: wrong_week,
            week,
        }
    );
    let full_week = u32::from(week) + 2 * 1024;
    let mut ura = valid_gps_ephemeris();
    ura.sv_accuracy = 15;
    assert_eq!(
        refusal(ura.to_broadcast_record(full_week)),
        RtcmConversionError::UraNoPrediction {
            system: crate::GnssSystem::Gps,
            index: 15,
        }
    );
    let mut wide = valid_gps_ephemeris();
    wide.satellite_id = 100;
    let error = refusal(wide.to_broadcast_record(full_week));
    assert_eq!(
        error,
        RtcmConversionError::SatelliteIdOutOfRange {
            message_number: 1019,
            field: "GPS PRN",
            value: 100,
            width: 6,
        }
    );
    assert_eq!(
        error.to_string(),
        "GPS PRN 100 in 1019 does not fit the 6-bit raw satellite field (0..=63)"
    );
    let mut sbas = valid_gps_ephemeris();
    sbas.satellite_id = 40;
    assert!(matches!(
        refusal(sbas.to_broadcast_record(full_week)),
        RtcmConversionError::NoLnavRecord { value: 40, .. }
    ));

    let mut spare = valid_galileo_fnav_ephemeris();
    spare.sisa = 200;
    assert_eq!(
        refusal(spare.to_broadcast_record()),
        RtcmConversionError::SisaSpare { index: 200 }
    );
    spare.sisa = 255;
    assert_eq!(
        refusal(spare.to_broadcast_record()),
        RtcmConversionError::SisaNoPrediction
    );

    let navic = navic_ephemeris();
    let full_week = u32::from(navic.week_number) + 1;
    assert_eq!(
        refusal(navic.to_broadcast_record(full_week)),
        RtcmConversionError::NavicWeekMismatch {
            full_week,
            week: navic.week_number,
        }
    );
}

#[test]
fn new_family_encoder_refusals_are_typed() {
    let refusal = |result: crate::error::Result<Vec<u8>>| match result {
        Err(Error::RtcmEncode(refusal)) => *refusal,
        other => panic!("expected an RTCM encode refusal, got {other:?}"),
    };

    let mut legacy = legacy_message(1004);
    legacy.satellite_count = 1;
    assert_eq!(
        refusal(legacy.encode()),
        RtcmEncodeError::CountMismatch {
            message_number: 1004,
            field: "satellite record",
            expected: 1,
            actual: 2,
        }
    );
    let mut legacy = legacy_message(1001);
    legacy.satellites[0].l2 = legacy_message(1003).satellites[0].l2;
    assert_eq!(
        refusal(legacy.encode()),
        RtcmEncodeError::SatelliteFieldPresence {
            message_number: 1001,
            record: RtcmRecordKind::LegacyObservations,
            satellite: 5,
            field: "L2 observables",
            carried: false,
        }
    );

    let parameters = SystemParameters {
        reference_station_id: 1,
        mjd: 60_000,
        seconds_of_day: 1,
        announcement_count: 1,
        leap_seconds: 18,
        announcements: Vec::new(),
        trailing_bits: Vec::new(),
    };
    assert_eq!(
        refusal(parameters.encode()),
        RtcmEncodeError::CountMismatch {
            message_number: 1013,
            field: "header record",
            expected: 1,
            actual: 0,
        }
    );

    let biases = GlonassCodePhaseBiases {
        reference_station_id: 1,
        aligned: true,
        reserved: 8,
        l1_ca: None,
        l1_p: None,
        l2_ca: None,
        l2_p: None,
        trailing_bits: Vec::new(),
    };
    assert_eq!(
        refusal(biases.encode()),
        RtcmEncodeError::FieldOutOfRange {
            message_number: 1230,
            field: "reserved".into(),
            value: 8,
            width: 3,
            encoding: RtcmFieldEncoding::Unsigned,
        }
    );
    let text = TextMessage {
        reference_station_id: 1,
        mjd: 60_000,
        seconds_of_day: 1,
        character_count: 0,
        code_units: vec![b'x'; 256],
        trailing_bits: Vec::new(),
    };
    assert!(matches!(
        refusal(text.encode()),
        RtcmEncodeError::FieldOutOfRange {
            message_number: 1029,
            ..
        }
    ));
    let mut network = correction_differences(1015);
    network.satellites[0].geometric = Some(1);
    assert!(matches!(
        refusal(network.encode()),
        RtcmEncodeError::SatelliteFieldPresence {
            message_number: 1015,
            record: RtcmRecordKind::Network {
                family: "network correction-difference message",
            },
            satellite: 5,
            field: "geometric difference",
            carried: false,
        }
    ));

    let mut transformation = helmert(1021);
    transformation.source_name = "A\u{263a}".to_string();
    assert!(matches!(
        refusal(transformation.encode()),
        RtcmEncodeError::NonLatin1Character {
            field,
            character: '\u{263a}',
        } if field == "source name"
    ));

    let mut vtec = vtec_message(1264);
    vtec.layers[0].degree = 0;
    assert!(matches!(
        refusal(vtec.encode()),
        RtcmEncodeError::ValueOutOfRange {
            message_number: 1264,
            value: 0,
            minimum: 1,
            maximum: 16,
            ..
        }
    ));
}

/// An MSM message of `kind` for `system` with two satellites and two signals,
/// every carried field set to a distinct value and every field the type does
/// not carry `None`.
fn msm_of_kind(system: crate::id::GnssSystem, kind: MsmKind) -> MsmMessage {
    let group: u16 = match system {
        crate::id::GnssSystem::Gps => 0,
        crate::id::GnssSystem::Glonass => 1,
        crate::id::GnssSystem::Galileo => 2,
        crate::id::GnssSystem::Sbas => 3,
        crate::id::GnssSystem::Qzss => 4,
        crate::id::GnssSystem::BeiDou => 5,
        crate::id::GnssSystem::Navic => 6,
    };
    let rate = kind.carries_phase_range_rate();
    let phase = kind.carries_phase_range();
    let extended = kind.is_extended_resolution();
    let satellites: Vec<_> = [(4u8, 71u8, 300u16, 9u8, -512i16), (17, 80, 1023, 13, 777)]
        .into_iter()
        .map(|(id, ms, mod1, ext, prr)| MsmSatellite {
            id,
            rough_range_ms: kind.carries_rough_range_ms().then_some(ms),
            rough_range_mod1: mod1,
            extended_info: rate.then_some(ext),
            rough_phase_range_rate_m_s: rate.then_some(prr),
        })
        .collect();
    let signals: Vec<_> = [(4u8, 2u8, 1i32), (4, 15, -2), (17, 2, 3)]
        .into_iter()
        .map(|(satellite_id, signal_id, k)| MsmSignal {
            satellite_id,
            signal_id,
            fine_pseudorange: kind
                .carries_pseudorange()
                .then_some(k * (if extended { 100_000 } else { 5_000 })),
            fine_phase_range: phase.then_some(-k * (if extended { 2_000_000 } else { 500_000 })),
            lock_time_indicator: phase
                .then_some((if extended { 600 } else { 11 }) + k.unsigned_abs() as u16),
            half_cycle_ambiguity: phase.then_some(k > 1),
            cnr: kind
                .carries_cnr()
                .then_some((if extended { 700 } else { 40 }) + k.unsigned_abs() as u16),
            fine_phase_range_rate: rate.then_some(k as i16 * 1_000),
        })
        .collect();
    MsmMessage {
        message_number: 1070 + 10 * group + u16::from(kind.number()),
        system,
        kind,
        header: msm_header(),
        signal_mask: msm_signal_mask(&signals),
        satellites,
        signals,
        trailing_bits: Vec::new(),
    }
}

/// Every MSM type of every system round-trips, and its body holds exactly the
/// fields RTCM 10403.3 Tables 3.5-78 to 3.5-99 give it: the 169-bit header and
/// masks, the cell mask, then per satellite and per cell the widths of its
/// type. MSM1..MSM3 have 10 satellite bits (DF398); MSM4 and MSM6 18
/// (DF397, DF398); MSM5 and MSM7 36 (DF397, DF419, DF398, DF399). Per cell:
/// MSM1 15, MSM2 27, MSM3 42, MSM4 48, MSM5 63, MSM6 65, MSM7 80.
#[test]
fn every_msm_type_round_trips_with_its_field_widths() {
    use crate::id::GnssSystem::*;
    let kinds: [(MsmKind, usize, usize); 7] = [
        (MsmKind::Msm1, 10, 15),
        (MsmKind::Msm2, 10, 27),
        (MsmKind::Msm3, 10, 42),
        (MsmKind::Msm4, 18, 48),
        (MsmKind::Msm5, 36, 63),
        (MsmKind::Msm6, 18, 65),
        (MsmKind::Msm7, 36, 80),
    ];
    for system in [Gps, Glonass, Galileo, Sbas, Qzss, BeiDou, Navic] {
        for (kind, sat_bits, cell_bits) in kinds {
            let message = msm_of_kind(system, kind);
            let body = message.encode().unwrap();
            // Header 12+12+30+1+3+7+2+2+1+3, satellite mask 64, signal mask
            // 32, cell mask 2 satellites x 2 signals.
            let bits = 169 + 4 + 2 * sat_bits + 3 * cell_bits;
            assert_eq!(body.len(), bits.div_ceil(8), "{system:?} {kind:?}");
            let decoded = MsmMessage::decode(&body).unwrap();
            assert_eq!(decoded, message, "{system:?} {kind:?}");
            assert_eq!(
                Message::decode(&body).unwrap(),
                Message::Msm(message),
                "{system:?} {kind:?}"
            );
        }
    }
    // The last digits 8, 9 and 0 are not MSM types.
    for number in [1078u16, 1079, 1080, 1138] {
        let mut w = BitWriter::new();
        w.push_u(u64::from(number), 12);
        w.push_u(0, 12);
        assert!(matches!(
            Message::decode(&w.into_bytes()).unwrap(),
            Message::Unsupported(_)
        ));
    }
}

/// A field an MSM type does not carry is refused when given, and a field it
/// carries is refused when missing, rather than written or left out.
#[test]
fn msm_types_refuse_fields_they_do_not_carry() {
    let gps = crate::id::GnssSystem::Gps;
    let refusal = |message: &MsmMessage| match message.encode() {
        Err(Error::RtcmEncode(refusal)) => *refusal,
        other => panic!("expected an RTCM encode refusal, got {other:?}"),
    };
    let mut msm1 = msm_of_kind(gps, MsmKind::Msm1);
    msm1.signals[0].fine_phase_range = Some(0);
    assert_eq!(
        refusal(&msm1),
        RtcmEncodeError::FieldPresence {
            message_number: 1071,
            record: RtcmRecordKind::Msm {
                system: gps,
                kind: MsmKind::Msm1,
            },
            field: "fine phase range",
            carried: false,
        }
    );

    let mut msm2 = msm_of_kind(gps, MsmKind::Msm2);
    msm2.signals[0].fine_pseudorange = Some(0);
    assert_eq!(
        refusal(&msm2),
        RtcmEncodeError::FieldPresence {
            message_number: 1072,
            record: RtcmRecordKind::Msm {
                system: gps,
                kind: MsmKind::Msm2,
            },
            field: "fine pseudorange",
            carried: false,
        }
    );

    let mut msm3 = msm_of_kind(gps, MsmKind::Msm3);
    msm3.satellites[0].rough_range_ms = Some(70);
    assert_eq!(
        refusal(&msm3),
        RtcmEncodeError::FieldPresence {
            message_number: 1073,
            record: RtcmRecordKind::Msm {
                system: gps,
                kind: MsmKind::Msm3,
            },
            field: "rough range",
            carried: false,
        }
    );

    let mut msm5 = msm_of_kind(gps, MsmKind::Msm5);
    msm5.signals[1].cnr = None;
    assert_eq!(
        refusal(&msm5),
        RtcmEncodeError::FieldPresence {
            message_number: 1075,
            record: RtcmRecordKind::Msm {
                system: gps,
                kind: MsmKind::Msm5,
            },
            field: "CNR",
            carried: true,
        }
    );

    let mut msm6 = msm_of_kind(gps, MsmKind::Msm6);
    msm6.signals[0].half_cycle_ambiguity = None;
    assert_eq!(
        refusal(&msm6),
        RtcmEncodeError::FieldPresence {
            message_number: 1076,
            record: RtcmRecordKind::Msm {
                system: gps,
                kind: MsmKind::Msm6,
            },
            field: "half-cycle ambiguity indicator",
            carried: true,
        }
    );
    let mut msm6 = msm_of_kind(gps, MsmKind::Msm6);
    msm6.signals[0].lock_time_indicator = Some(1024);
    assert!(matches!(
        refusal(&msm6),
        RtcmEncodeError::FieldOutOfRange {
            message_number: 1076,
            value: 1024,
            width: 10,
            encoding: RtcmFieldEncoding::Unsigned,
            ..
        }
    ));

    // MSM5 carries the phase-range rates, whose invalid values read as None.
    let mut msm5 = msm_of_kind(gps, MsmKind::Msm5);
    msm5.satellites[0].rough_phase_range_rate_m_s = None;
    msm5.signals[0].fine_phase_range_rate = None;
    let body = msm5.encode().unwrap();
    assert_eq!(MsmMessage::decode(&body).unwrap(), msm5);
}

/// MSM1 carries no lock-time or half-cycle indicator, so the lock tracker
/// derives no LLI from it and keeps no state for it; MSM2, MSM3 and MSM5 use
/// DF402 and MSM6 DF407.
#[test]
fn lock_tracker_reads_each_msm_type_by_its_lock_field() {
    let gps = crate::id::GnssSystem::Gps;
    let mut tracker = LockTimeTracker::new();
    assert!(tracker.observe(&msm_of_kind(gps, MsmKind::Msm1)).is_empty());
    for kind in [MsmKind::Msm2, MsmKind::Msm3, MsmKind::Msm5, MsmKind::Msm6] {
        let message = msm_of_kind(gps, kind);
        let cells = LockTimeTracker::new().observe(&message);
        assert_eq!(cells.len(), message.signals.len(), "{kind:?}");
        for (cell, signal) in cells.iter().zip(&message.signals) {
            assert_eq!(
                cell.min_lock_time_ms,
                minimum_lock_time_ms(kind, signal.lock_time_indicator.unwrap()),
                "{kind:?}"
            );
        }
    }
    assert_eq!(minimum_lock_time_ms(MsmKind::Msm1, 0), None);
    assert_eq!(minimum_lock_time_ms(MsmKind::Msm3, 6), Some(1024));
    assert_eq!(minimum_lock_time_ms(MsmKind::Msm6, 6), Some(6));
}

/// A legacy observation message `number` with two satellites, every carried
/// field set and every part the number's layout does not carry `None`.
fn legacy_message(number: u16) -> LegacyObservations {
    let glonass = number >= 1009;
    let extended = matches!(number, 1002 | 1004 | 1010 | 1012);
    let l2 = matches!(number, 1003 | 1004 | 1011 | 1012);
    let satellites = [(5u8, 1u32), (24, 2)]
        .into_iter()
        .map(|(satellite_id, k)| LegacySatellite {
            satellite_id,
            frequency_channel: glonass.then_some(3 + k as u8),
            l1: LegacyL1 {
                code_indicator: k == 2,
                pseudorange: 1_000_000 * k,
                phase_range_minus_pseudorange: -300_000 + k as i32,
                lock_time_indicator: 100 + k as u8,
                pseudorange_modulus_ambiguity: extended.then_some(70 + k as u8),
                cnr: extended.then_some(160 + k as u8),
            },
            l2: l2.then_some(LegacyL2 {
                code_indicator: k as u8,
                pseudorange_difference: -4_000 * k as i16,
                phase_range_minus_l1_pseudorange: 400_000 - k as i32,
                lock_time_indicator: 120 + k as u8,
                cnr: extended.then_some(150 + k as u8),
            }),
        })
        .collect();
    LegacyObservations {
        message_number: number,
        reference_station_id: 2003,
        epoch_time: if glonass { 86_399_000 } else { 604_799_000 },
        synchronous_gnss: true,
        satellite_count: 2,
        divergence_free_smoothing: true,
        smoothing_interval: 5,
        satellites,
        trailing_bits: Vec::new(),
    }
}

/// Every legacy observation message round-trips, and its body is the header
/// and records RTCM 10403.3 Tables 3.5-2 to 3.5-15 give it: a 64-bit GPS
/// header (DF002..DF008) and 58, 74, 101 or 125 bits per satellite for 1001,
/// 1002, 1003, 1004; a 61-bit GLONASS header (DF002, DF003, DF034..DF037) and
/// 64, 79, 107 or 130 bits per satellite for 1009, 1010, 1011, 1012.
#[test]
fn legacy_observations_round_trip_with_their_field_widths() {
    for (number, header, record) in [
        (1001u16, 64usize, 58usize),
        (1002, 64, 74),
        (1003, 64, 101),
        (1004, 64, 125),
        (1009, 61, 64),
        (1010, 61, 79),
        (1011, 61, 107),
        (1012, 61, 130),
    ] {
        let message = legacy_message(number);
        let body = message.encode().unwrap();
        assert_eq!(body.len(), (header + 2 * record).div_ceil(8), "{number}");
        assert_eq!(
            LegacyObservations::decode(&body).unwrap(),
            message,
            "{number}"
        );
        assert_eq!(
            Message::decode(&body).unwrap(),
            Message::LegacyObservations(message.clone()),
            "{number}"
        );
        assert_eq!(
            Message::LegacyObservations(message).message_number(),
            number
        );
    }
    let gps = legacy_message(1004);
    assert_eq!(gps.system(), Some(crate::id::GnssSystem::Gps));
    assert_eq!(
        legacy_message(1011).system(),
        Some(crate::id::GnssSystem::Glonass)
    );
}

/// A legacy body that ends before the records its header counts is refused
/// as truncated under the strict policy. Under the lenient policy the
/// complete records are read, as RTKLIB `decode_type1004` reads them, the
/// header count and the bits of the cut record are kept, and the lenient
/// encoder writes the body back as read.
#[test]
fn short_legacy_body_is_refused_strictly_and_read_leniently() {
    let full = legacy_message(1004);
    let body = full.encode().unwrap();
    // Header 64 bits and one 125-bit record end at bit 189; cut inside the
    // second record.
    let short = body[..30].to_vec();
    let err = LegacyObservations::decode(&short).unwrap_err();
    assert!(err.to_string().contains("truncated"), "{err}");

    let departure = RtcmDeparture::RecordsShort {
        message_number: 1004,
        declared: 2,
        read: 1,
    };
    let (read, departures) =
        LegacyObservations::decode_with_policy(&short, RtcmPolicy::Lenient).unwrap();
    assert_eq!(departures, vec![departure.clone()]);
    assert_eq!(read.satellite_count, 2);
    assert_eq!(read.satellites, full.satellites[..1].to_vec());
    assert_eq!(read.trailing_bits.len(), 30 * 8 - 189);
    assert!(matches!(
        read.encode(),
        Err(Error::RtcmEncode(error))
            if matches!(
                *error,
                RtcmEncodeError::CountMismatch {
                    message_number: 1004,
                    field: "satellite record",
                    expected: 2,
                    actual: 1,
                }
            )
    ));
    let (written, departures) = read.encode_with_policy(RtcmPolicy::Lenient).unwrap();
    assert_eq!(written, short);
    assert_eq!(departures, vec![departure]);

    // Bits after a complete set of records are a trailing-bits departure.
    let mut long = body.clone();
    long.push(0x80);
    assert!(LegacyObservations::decode(&long).is_err());
    let (read, departures) =
        LegacyObservations::decode_with_policy(&long, RtcmPolicy::Lenient).unwrap();
    assert!(matches!(
        departures[..],
        [RtcmDeparture::TrailingBits { .. }]
    ));
    assert_eq!(
        read.encode_with_policy(RtcmPolicy::Lenient).unwrap().0,
        long
    );
}

/// The legacy encoder refuses a part its layout does not carry, a missing
/// part it does, a count other than the records, and values wider than their
/// fields.
#[test]
fn legacy_encoder_refuses_what_its_layout_cannot_state() {
    let refused = |message: LegacyObservations, expected: RtcmEncodeError| {
        let actual = match message.encode() {
            Err(Error::RtcmEncode(error)) => *error,
            other => panic!("expected {expected:?}, got {other:?}"),
        };
        assert_eq!(actual, expected);
    };
    let mut m = legacy_message(1001);
    m.satellites[0].l2 = legacy_message(1003).satellites[0].l2;
    refused(
        m,
        RtcmEncodeError::SatelliteFieldPresence {
            message_number: 1001,
            record: RtcmRecordKind::LegacyObservations,
            satellite: 5,
            field: "L2 observables",
            carried: false,
        },
    );
    let mut m = legacy_message(1012);
    m.satellites[1].frequency_channel = None;
    refused(
        m,
        RtcmEncodeError::SatelliteFieldPresence {
            message_number: 1012,
            record: RtcmRecordKind::LegacyObservations,
            satellite: 24,
            field: "GLONASS frequency channel",
            carried: true,
        },
    );
    let mut m = legacy_message(1004);
    m.satellites[0].frequency_channel = Some(7);
    refused(
        m,
        RtcmEncodeError::SatelliteFieldPresence {
            message_number: 1004,
            record: RtcmRecordKind::LegacyObservations,
            satellite: 5,
            field: "GLONASS frequency channel",
            carried: false,
        },
    );
    let mut m = legacy_message(1002);
    m.satellites[0].l1.cnr = None;
    refused(
        m,
        RtcmEncodeError::SatelliteFieldPresence {
            message_number: 1002,
            record: RtcmRecordKind::LegacyObservations,
            satellite: 5,
            field: "L1 CNR",
            carried: true,
        },
    );
    let mut m = legacy_message(1003);
    m.satellites[0].l2.as_mut().unwrap().cnr = Some(1);
    refused(
        m,
        RtcmEncodeError::SatelliteFieldPresence {
            message_number: 1003,
            record: RtcmRecordKind::LegacyObservations,
            satellite: 5,
            field: "L2 CNR",
            carried: false,
        },
    );
    let mut m = legacy_message(1004);
    m.satellite_count = 1;
    refused(
        m,
        RtcmEncodeError::CountMismatch {
            message_number: 1004,
            field: "satellite record",
            expected: 1,
            actual: 2,
        },
    );
    let mut m = legacy_message(1004);
    m.satellites[0].l1.pseudorange = 1 << 24;
    refused(
        m,
        RtcmEncodeError::FieldOutOfRange {
            message_number: 1004,
            field: "satellite 5 L1 pseudorange".to_string(),
            value: 1 << 24,
            width: 24,
            encoding: RtcmFieldEncoding::Unsigned,
        },
    );
    // GLONASS DF041 is one bit wider, DF044 one bit narrower.
    let mut m = legacy_message(1012);
    m.satellites[0].l1.pseudorange = 1 << 24;
    m.encode().expect("DF041 is 25 bits");
    m.satellites[0].l1.pseudorange_modulus_ambiguity = Some(128);
    refused(
        m,
        RtcmEncodeError::FieldOutOfRange {
            message_number: 1012,
            field: "satellite 5 L1 pseudorange modulus ambiguity".to_string(),
            value: 128,
            width: 7,
            encoding: RtcmFieldEncoding::Unsigned,
        },
    );
    let mut m = legacy_message(1004);
    m.message_number = 1005;
    refused(
        m,
        RtcmEncodeError::MessageNumber {
            message_number: 1005,
            record: RtcmRecordKind::LegacyObservations,
        },
    );
}

fn navic_ephemeris() -> NavicEphemeris {
    NavicEphemeris {
        satellite_id: 9,
        week_number: 389,
        a_f0: -1_234_567,
        a_f1: -12_345,
        a_f2: -3,
        ura: 2,
        t_oc: 10_821,
        t_gd: -5,
        delta_n: 1_234_567,
        iodec: 161,
        reserved: 0x2A5,
        l5_flag: true,
        s_flag: false,
        c_uc: -16_000,
        c_us: 15_000,
        c_ic: -1,
        c_is: 2,
        c_rc: 16_383,
        c_rs: -16_384,
        idot: -8_000,
        m0: -2_000_000_000,
        t_oe: 10_821,
        eccentricity: 3_000_000,
        sqrt_a: 3_404_000_000,
        omega0: 1_500_000_000,
        omega: -1_000_000_000,
        omega_dot: -2_000_000,
        i0: 400_000_000,
        spare_df544: 3,
        spare_df545: 1,
        trailing_bits: Vec::new(),
    }
}

/// A 1041 body is the 482 bits of RTCM 10403.3 Amendment 2's table (DF002,
/// DF516..DF545) and round-trips, spare and reserved bits included; its
/// health word is the L5 flag in bit 1 and the S flag in bit 0, as RTKLIB
/// stores it, and a PRN wider than DF516 is refused.
#[test]
fn navic_ephemeris_round_trips_with_its_field_widths() {
    let eph = navic_ephemeris();
    let body = eph.encode().unwrap();
    assert_eq!(body.len(), 482usize.div_ceil(8));
    assert_eq!(NavicEphemeris::decode(&body).unwrap(), eph);
    assert_eq!(
        Message::decode(&body).unwrap(),
        Message::NavicEphemeris(eph.clone())
    );
    assert_eq!(Message::NavicEphemeris(eph.clone()).message_number(), 1041);
    assert_eq!(eph.health(), 2);
    assert_eq!(eph.satellite().unwrap().to_string(), "I09");

    let record = eph.to_broadcast_record(1024 + 389).unwrap();
    assert_eq!(record.message, crate::rinex_nav::NavMessage::NavicLnav);
    assert_eq!(record.issue_of_data.unwrap().issue, 161);
    assert_eq!(record.sv_health, 2.0);
    assert_eq!(record.sv_accuracy_m, Some(4.85));
    assert_eq!(record.elements.toe_sow, 173_136.0);
    assert_eq!(record.clock.af0, -1_234_567.0 * 2f64.powi(-31));
    assert_eq!(
        record.elements.delta_n,
        1_234_567.0 * 2f64.powi(-41) * core::f64::consts::PI
    );
    assert_eq!(record.elements.crc, 16_383.0 / 16.0);
    assert!(eph.to_broadcast_record(389 + 1).is_err());

    let mut wide = eph;
    wide.satellite_id = 64;
    assert!(wide.encode().unwrap_err().to_string().contains("6-bit"));
}

/// A 1230 body carries a bias for each set bit of its four-bit signal mask,
/// in mask order. A mask with no bit set is a complete 32-bit message, which
/// RTKLIB `decode_type1230` refuses as short and RTCM 10403.3 gives no bias
/// fields; it is read here. The invalid value decodes as transmitted and
/// reads as no bias in metres.
#[test]
fn glonass_code_phase_biases_round_trip_by_mask() {
    let empty = GlonassCodePhaseBiases {
        reference_station_id: 7,
        aligned: true,
        reserved: 0,
        l1_ca: None,
        l1_p: None,
        l2_ca: None,
        l2_p: None,
        trailing_bits: Vec::new(),
    };
    let body = empty.encode().unwrap();
    assert_eq!(body.len(), 4);
    assert_eq!(empty.signal_mask(), 0);
    assert_eq!(GlonassCodePhaseBiases::decode(&body).unwrap(), empty);

    let some = GlonassCodePhaseBiases {
        l1_ca: Some(953),
        l2_ca: Some(GLONASS_CODE_PHASE_BIAS_INVALID),
        l2_p: Some(-1),
        reserved: 5,
        ..empty.clone()
    };
    let body = some.encode().unwrap();
    assert_eq!(some.signal_mask(), 0b1011);
    assert_eq!(body.len(), (32 + 3 * 16) / 8);
    let decoded = GlonassCodePhaseBiases::decode(&body).unwrap();
    assert_eq!(decoded, some);
    assert_eq!(
        Message::decode(&body).unwrap(),
        Message::GlonassCodePhaseBiases(some.clone())
    );
    assert_eq!(
        decoded.biases_m(),
        [Some(953.0 * 0.02), None, None, Some(-0.02)]
    );
    let mut wide = some;
    wide.reserved = 8;
    assert!(wide
        .encode()
        .unwrap_err()
        .to_string()
        .contains("reserved 8"));
}

fn igs_header(kind: SsrKind, count: u8) -> SsrHeader {
    SsrHeader {
        epoch_time_s: 345_600,
        update_interval: 3,
        multiple_message: true,
        iod_ssr: 7,
        provider_id: 258,
        solution_id: 2,
        satellite_reference_datum: matches!(kind, SsrKind::Orbit | SsrKind::CombinedOrbitClock)
            .then_some(true),
        dispersive_bias_consistency: (kind == SsrKind::PhaseBias).then_some(true),
        mw_consistency: (kind == SsrKind::PhaseBias).then_some(false),
        satellite_count: count,
    }
}

/// An IGS SSR message of every group for `system`, one satellite, with an
/// eight-bit issue.
fn igs_message(system: crate::id::GnssSystem, kind: SsrKind) -> SsrMessage {
    let orbit = SsrOrbitRecord {
        satellite_id: 36,
        iode: 0xA5,
        iod_crc: None,
        delta_radial: -12_345,
        delta_along: 23_456,
        delta_cross: -34_567,
        dot_delta_radial: 456,
        dot_delta_along: -567,
        dot_delta_cross: 678,
    };
    let clock = SsrClockRecord {
        satellite_id: 36,
        c0: -78_901,
        c1: 89_012,
        c2: -9_012_345,
    };
    let mut message = SsrMessage {
        message_number: IGS_SSR_MESSAGE_NUMBER,
        igs_ssr_version: Some(1),
        system,
        kind,
        header: igs_header(kind, 1),
        orbit: Vec::new(),
        clock: Vec::new(),
        code_bias: Vec::new(),
        phase_bias: Vec::new(),
        ura: Vec::new(),
        padding_bits: Vec::new(),
    };
    match kind {
        SsrKind::Orbit => message.orbit.push(orbit),
        SsrKind::Clock => message.clock.push(clock),
        SsrKind::CombinedOrbitClock => {
            message.orbit.push(orbit);
            message.clock.push(clock);
        }
        SsrKind::HighRateClock => message.clock.push(SsrClockRecord {
            c1: 0,
            c2: 0,
            ..clock
        }),
        SsrKind::CodeBias => message.code_bias.push(SsrCodeBiasRecord {
            satellite_id: 36,
            biases: vec![(0, -1234), (9, 2345), (11, 1)],
        }),
        SsrKind::PhaseBias => message.phase_bias.push(SsrPhaseBiasRecord {
            satellite_id: 36,
            yaw_angle: 300,
            yaw_rate: -7,
            biases: vec![SsrPhaseBiasSignal {
                signal_id: 5,
                integer_indicator: 1,
                wide_lane_integer_indicator: 2,
                discontinuity_counter: 9,
                bias: -123_456,
            }],
        }),
        SsrKind::Ura => message.ura.push((36, 41)),
    }
    message
}

/// Every IGS SSR satellite message round-trips, and its body is the layout of
/// IGS SSR v1.00 Tables 8-23: a 79-bit orbit or combined header (the CRS
/// indicator after the solution ID), a 78-bit clock, high-rate clock, code-bias
/// or URA header, an 80-bit phase-bias header, and per satellite 135 (orbit),
/// 76 (clock), 205 (combined), 28 (high-rate clock), 11 + 19 per bias (code
/// bias), 28 + 32 per bias (phase bias) or 12 (URA) bits, with a six-bit
/// satellite ID and an eight-bit IOD for every system.
#[test]
fn igs_ssr_messages_round_trip_with_their_field_widths() {
    use crate::id::GnssSystem::*;
    for (system, offset) in [
        (Gps, 20u8),
        (Glonass, 40),
        (Galileo, 60),
        (Qzss, 80),
        (BeiDou, 100),
        (Sbas, 120),
    ] {
        for (kind, digit, bits) in [
            (SsrKind::Orbit, 1u8, 79 + 135),
            (SsrKind::Clock, 2, 78 + 76),
            (SsrKind::CombinedOrbitClock, 3, 79 + 205),
            (SsrKind::HighRateClock, 4, 78 + 28),
            (SsrKind::CodeBias, 5, 78 + 11 + 3 * 19),
            (SsrKind::PhaseBias, 6, 80 + 28 + 32),
            (SsrKind::Ura, 7, 78 + 12),
        ] {
            let message = igs_message(system, kind);
            assert_eq!(message.igs_ssr_subtype(), Some(offset + digit));
            let body = message.encode().unwrap();
            let at = format!("{system:?} {kind:?}");
            assert_eq!(body.len(), (bits as usize).div_ceil(8), "{at}");
            let mut r = BitReader::new(&body);
            assert_eq!(r.u(12).unwrap(), 4076, "{at}");
            assert_eq!(r.u(3).unwrap(), 1, "{at} version");
            assert_eq!(r.u(8).unwrap(), u64::from(offset + digit), "{at} subtype");
            let mut decoded = SsrMessage::decode(&body).unwrap();
            assert!(decoded.padding_bits.iter().all(|bit| !bit), "{at}");
            decoded.padding_bits.clear();
            assert_eq!(decoded, message, "{at}");
            let decoded = Message::decode(&body).unwrap();
            assert!(matches!(decoded, Message::Ssr(_)), "{at}");
            assert_eq!(decoded.encode().unwrap(), body, "{at}");
        }
    }

    // The orbit header ends with the CRS indicator after the solution ID.
    let body = igs_message(Galileo, SsrKind::Orbit).encode().unwrap();
    let mut r = BitReader::new(&body);
    r.u(12 + 3 + 8 + 20).unwrap();
    r.u(4 + 1 + 4 + 16 + 4).unwrap();
    assert_eq!(r.u(1).unwrap(), 1, "CRS indicator");
    assert_eq!(r.u(6).unwrap(), 1, "satellite count");
    assert_eq!(r.u(6).unwrap(), 36, "satellite ID");
    assert_eq!(
        r.u(8).unwrap(),
        0xA5,
        "IDF012: the eight low bits of IODnav"
    );
}

/// A 4076 message is written only in the IGS SSR layout and an RTCM SSR one
/// only in its own: an IGS SSR version missing, given for an RTCM number, a
/// NavIC group (IGS SSR has none) and an issue wider than eight bits are
/// refused. A 4076 subtype this codec does not decode is kept as unsupported
/// and written back; a decoded one is not held as unsupported.
#[test]
fn igs_ssr_layout_refusals_and_unsupported_subtypes() {
    use crate::id::GnssSystem::*;
    let refusal = |message: SsrMessage| match message.encode() {
        Err(Error::RtcmEncode(error)) => *error,
        other => panic!("expected an SSR encode refusal, got {other:?}"),
    };
    let mut m = igs_message(Gps, SsrKind::Orbit);
    m.igs_ssr_version = None;
    assert_eq!(
        refusal(m),
        RtcmEncodeError::FieldPresence {
            message_number: 4076,
            record: RtcmRecordKind::Ssr {
                system: Gps,
                kind: SsrKind::Orbit,
            },
            field: "IGS SSR version",
            carried: true,
        }
    );
    let mut m = igs_message(Gps, SsrKind::Orbit);
    m.message_number = 1057;
    assert_eq!(
        refusal(m),
        RtcmEncodeError::FieldPresence {
            message_number: 1057,
            record: RtcmRecordKind::Ssr {
                system: Gps,
                kind: SsrKind::Orbit,
            },
            field: "IGS SSR version",
            carried: false,
        }
    );
    let m = igs_message(Navic, SsrKind::Orbit);
    assert_eq!(
        refusal(m),
        RtcmEncodeError::MessageNumber {
            message_number: 4076,
            record: RtcmRecordKind::Ssr {
                system: Navic,
                kind: SsrKind::Orbit,
            },
        }
    );
    let mut m = igs_message(Galileo, SsrKind::Orbit);
    m.orbit[0].iode = 0x1A5;
    assert!(matches!(
        refusal(m),
        RtcmEncodeError::FieldOutOfRange {
            message_number: 4076,
            value: 0x1A5,
            width: 8,
            ..
        }
    ));
    let mut m = igs_message(Glonass, SsrKind::Clock);
    m.clock[0].satellite_id = 64;
    assert_eq!(
        refusal(m),
        RtcmEncodeError::SsrSatelliteIdOutOfRange {
            message_number: 4076,
            value: 64,
            width: 6,
        }
    );

    for subtype in [0u8, 28, 140, 200, 255] {
        let mut w = BitWriter::new();
        w.push_u(4076, 12);
        w.push_u(1, 3);
        w.push_u(u64::from(subtype), 8);
        w.push_u(0xABC, 12);
        let body = w.into_bytes();
        let decoded = Message::decode(&body).unwrap();
        assert!(
            matches!(&decoded, Message::Unsupported(u) if u.body == body),
            "{subtype}"
        );
        assert_eq!(decoded.encode().unwrap(), body, "{subtype}");
    }
    let body = igs_message(Qzss, SsrKind::Ura).encode().unwrap();
    let held = UnsupportedMessage {
        message_number: 4076,
        body,
    };
    assert!(matches!(
        held.encode(),
        Err(Error::RtcmEncode(error))
            if matches!(
                *error,
                RtcmEncodeError::UnsupportedDecodedNumber {
                    message_number: 4076,
                }
            )
    ));
}

fn vtec_message(message_number: u16) -> SsrVtecMessage {
    SsrVtecMessage {
        message_number,
        igs_ssr_version: (message_number == 4076).then_some(1),
        epoch_time_s: 345_600,
        update_interval: 5,
        multiple_message: false,
        iod_ssr: 3,
        provider_id: 258,
        solution_id: 1,
        quality_indicator: 511,
        layers: vec![
            SsrVtecLayer {
                height: 45,
                degree: 3,
                order: 2,
                cosine: (0..9).map(|k| 100 * k - 400).collect(),
                sine: (0..5).map(|k| -7 * k).collect(),
            },
            SsrVtecLayer {
                height: 100,
                degree: 1,
                order: 1,
                cosine: vec![i16::MIN, 1, i16::MAX],
                sine: vec![-1],
            },
        ],
        trailing_bits: Vec::new(),
    }
}

/// The VTEC coefficient counts follow the sequence IGS SSR v1.00 and RTCM
/// 10403.3 state, `C_nm` for `m = 0..=M`, `n = m..=N` and `S_nm` for
/// `m = 1..=M`, `n = m..=N`: degree 3 and order 2 carry 9 and 5 (the example
/// of IGS SSR v1.00 section 8.4.1); an order above the degree carries no term
/// for `m > N`.
#[test]
fn vtec_coefficient_counts_follow_the_stated_sequence() {
    assert_eq!(SsrVtecLayer::coefficient_counts(3, 2), (9, 5));
    assert_eq!(SsrVtecLayer::coefficient_counts(1, 1), (3, 1));
    assert_eq!(SsrVtecLayer::coefficient_counts(16, 16), (153, 136));
    assert_eq!(SsrVtecLayer::coefficient_counts(1, 3), (3, 1));
    for n in 1..=16u8 {
        for m in 1..=n {
            let (c, s) = SsrVtecLayer::coefficient_counts(n, m);
            let (n, m) = (usize::from(n), usize::from(m));
            assert_eq!(c, (n + 1) * (n + 2) / 2 - (n - m) * (n - m + 1) / 2);
            assert_eq!(s, c - (n + 1));
        }
    }
}

/// The IGS SSR VTEC message (4076 subtype 201) and RTCM 1264 round-trip; their
/// bodies differ by the IGS identification only: an 83-bit (4076) or 72-bit
/// (1264) header, then per layer 16 bits and 16 per coefficient.
#[test]
fn vtec_messages_round_trip_with_their_field_widths() {
    for (number, header) in [(4076u16, 83usize), (1264, 72)] {
        let message = vtec_message(number);
        let body = message.encode().unwrap();
        let bits = header + 2 * 16 + 16 * (9 + 5 + 3 + 1);
        assert_eq!(body.len(), bits.div_ceil(8), "{number}");
        assert_eq!(SsrVtecMessage::decode(&body).unwrap(), message, "{number}");
        assert_eq!(
            Message::decode(&body).unwrap(),
            Message::SsrVtec(message.clone()),
            "{number}"
        );
        assert_eq!(Message::SsrVtec(message).message_number(), number);
    }
    let refused = |message: SsrVtecMessage, expected: RtcmEncodeError| {
        let err = message
            .encode()
            .expect_err("invalid VTEC message must be refused");
        let Error::RtcmEncode(actual) = err else {
            panic!("expected a typed RTCM encode refusal, got {err}");
        };
        assert_eq!(*actual, expected);
    };
    let mut m = vtec_message(4076);
    m.layers[0].sine.pop();
    refused(
        m,
        RtcmEncodeError::CountMismatch {
            message_number: 4076,
            field: "VTEC sine coefficient",
            expected: 5,
            actual: 4,
        },
    );
    let mut m = vtec_message(1264);
    m.layers[1].degree = 17;
    refused(
        m,
        RtcmEncodeError::ValueOutOfRange {
            message_number: 1264,
            field: "VTEC layer 1 degree".into(),
            value: 17,
            minimum: 1,
            maximum: 16,
        },
    );
    let mut m = vtec_message(1264);
    m.layers.clear();
    refused(
        m,
        RtcmEncodeError::ValueOutOfRange {
            message_number: 1264,
            field: "VTEC layer count".into(),
            value: 0,
            minimum: 1,
            maximum: 4,
        },
    );
    let mut m = vtec_message(1264);
    m.igs_ssr_version = Some(1);
    refused(
        m,
        RtcmEncodeError::FieldPresence {
            message_number: 1264,
            record: RtcmRecordKind::SsrVtec {
                message_number: 1264,
            },
            field: "IGS SSR version",
            carried: false,
        },
    );
    let mut m = vtec_message(4076);
    m.quality_indicator = 512;
    refused(
        m,
        RtcmEncodeError::FieldOutOfRange {
            message_number: 4076,
            field: "VTEC quality indicator".into(),
            value: 512,
            width: 9,
            encoding: RtcmFieldEncoding::Unsigned,
        },
    );
}

fn correction_differences(number: u16) -> NetworkCorrectionDifferences {
    let geometric = matches!(number, 1016 | 1017 | 1038 | 1039);
    let ionospheric = !matches!(number, 1016 | 1038);
    NetworkCorrectionDifferences {
        message_number: number,
        network_id: 200,
        subnetwork_id: 9,
        epoch_time: if number >= 1037 { 864_000 } else { 6_047_999 },
        multiple_message: true,
        master_station_id: 4095,
        auxiliary_station_id: 17,
        satellite_count: 2,
        satellites: [(5u8, 1i32), (21, -1)]
            .into_iter()
            .map(|(satellite_id, k)| NetworkCorrectionDifference {
                satellite_id,
                ambiguity_status: 2,
                non_sync_count: 7,
                geometric: geometric.then_some(65_535 * k),
                iod: geometric.then_some(200),
                ionospheric: ionospheric.then_some(-65_536 * k.max(0)),
            })
            .collect(),
        trailing_bits: Vec::new(),
    }
}

/// The correction-difference messages carry the ionospheric difference
/// (1015, 1037), the geometric difference with its IOD (1016, 1038), or both
/// (1017, 1039): 36 + 40 header bits (DF065 23 bits for GPS, DF233 20 for
/// GLONASS) and 28, 36 or 53 bits per satellite. A part the number does not
/// carry is refused when given, and a body shorter than its satellite count
/// is refused strictly and read leniently to the last complete record.
#[test]
fn network_correction_differences_round_trip_by_layout() {
    for (number, header, record) in [
        (1015u16, 76usize, 28usize),
        (1016, 76, 36),
        (1017, 76, 53),
        (1037, 73, 28),
        (1038, 73, 36),
        (1039, 73, 53),
    ] {
        let message = correction_differences(number);
        let body = message.encode().unwrap();
        assert_eq!(body.len(), (header + 2 * record).div_ceil(8), "{number}");
        assert_eq!(
            NetworkCorrectionDifferences::decode(&body).unwrap(),
            message,
            "{number}"
        );
        assert_eq!(
            Message::decode(&body).unwrap(),
            Message::NetworkCorrectionDifferences(message),
            "{number}"
        );
    }
    let mut m = correction_differences(1015);
    m.satellites[0].geometric = Some(1);
    assert!(m
        .encode()
        .unwrap_err()
        .to_string()
        .contains("geometric difference is given, and 1015 does not carry it"));
    let mut m = correction_differences(1038);
    m.satellites[1].iod = None;
    assert!(m
        .encode()
        .unwrap_err()
        .to_string()
        .contains("IOD is not given, and 1038 carries it"));

    let full = correction_differences(1017);
    let body = full.encode().unwrap();
    let short = body[..body.len() - 4].to_vec();
    assert!(NetworkCorrectionDifferences::decode(&short)
        .unwrap_err()
        .to_string()
        .contains("truncated"));
    let (read, departures) =
        NetworkCorrectionDifferences::decode_with_policy(&short, RtcmPolicy::Lenient).unwrap();
    let departure = RtcmDeparture::RecordsShort {
        message_number: 1017,
        declared: 2,
        read: 1,
    };
    assert_eq!(departures, vec![departure.clone()]);
    assert_eq!(read.satellites, full.satellites[..1].to_vec());
    assert!(read.encode().is_err());
    assert_eq!(
        read.encode_with_policy(RtcmPolicy::Lenient).unwrap(),
        (short, vec![departure])
    );
}

/// The network residual, physical station, FKP and auxiliary-station messages
/// round-trip with the widths RTCM 10403.3 gives them.
#[test]
fn network_rtk_messages_round_trip_with_their_field_widths() {
    let residual = NetworkResidual {
        satellite_id: 63,
        s_oc: 255,
        s_od: 511,
        s_oh: 63,
        s_lc: 1023,
        s_ld: 0,
    };
    for (number, header) in [(1030u16, 56usize), (1031, 53)] {
        let message = NetworkResiduals {
            message_number: number,
            epoch_time: if number == 1030 { 604_799 } else { 86_399 },
            reference_station_id: 12,
            reference_station_count: 127,
            satellite_count: 2,
            satellites: vec![residual, residual],
            trailing_bits: Vec::new(),
        };
        let body = message.encode().unwrap();
        assert_eq!(body.len(), (header + 2 * 49).div_ceil(8), "{number}");
        assert_eq!(NetworkResiduals::decode(&body).unwrap(), message);
        assert_eq!(
            Message::decode(&body).unwrap(),
            Message::NetworkResiduals(message)
        );
    }
    let gradient = FkpGradient {
        satellite_id: 1,
        iod: 255,
        geometric_north: -2048,
        geometric_east: 2047,
        ionospheric_north: -8192,
        ionospheric_east: 8191,
    };
    for (number, header) in [(1034u16, 49usize), (1035, 46)] {
        let message = FkpGradients {
            message_number: number,
            reference_station_id: 4095,
            epoch_time: 3,
            satellite_count: 1,
            satellites: vec![gradient],
            trailing_bits: Vec::new(),
        };
        let body = message.encode().unwrap();
        assert_eq!(body.len(), (header + 66).div_ceil(8), "{number}");
        assert_eq!(FkpGradients::decode(&body).unwrap(), message);
        let mut wide = message;
        wide.satellites[0].geometric_north = -2049;
        assert!(wide.encode().unwrap_err().to_string().contains("-2049"));
    }
    let station = PhysicalReferenceStation {
        non_physical_station_id: 100,
        physical_station_id: 4000,
        itrf_realization_year: 20,
        ecef_x: -(1 << 37),
        ecef_y: (1 << 37) - 1,
        ecef_z: 12_602_528_900,
        trailing_bits: Vec::new(),
    };
    let body = station.encode().unwrap();
    assert_eq!(body.len(), 156usize.div_ceil(8));
    assert_eq!(PhysicalReferenceStation::decode(&body).unwrap(), station);
    let auxiliary = NetworkAuxiliaryStation {
        network_id: 255,
        subnetwork_id: 15,
        auxiliary_station_count: 31,
        master_station_id: 1,
        auxiliary_station_id: 2,
        delta_latitude: -(1 << 19),
        delta_longitude: (1 << 20) - 1,
        delta_height: -1,
        trailing_bits: Vec::new(),
    };
    let body = auxiliary.encode().unwrap();
    assert_eq!(body.len(), 117usize.div_ceil(8));
    assert_eq!(NetworkAuxiliaryStation::decode(&body).unwrap(), auxiliary);
}

fn helmert(number: u16) -> HelmertTransformation {
    HelmertTransformation {
        message_number: number,
        source_name: "ETRF2000".to_string(),
        target_name: "\u{e9}\u{ff}".to_string(),
        system_id: 3,
        utilized_messages: 0x3FF,
        plate_number: 31,
        computation_indicator: 15,
        height_indicator: 3,
        validity_latitude: -(1 << 18),
        validity_longitude: (1 << 19) - 1,
        validity_extension_latitude: 16_383,
        validity_extension_longitude: 0,
        dx: -(1 << 22),
        dy: 1,
        dz: (1 << 22) - 1,
        r1: i32::MIN,
        r2: i32::MAX,
        r3: 0,
        ds: -(1 << 24),
        rotation_point: (number == 1022).then_some(RotationPoint {
            x: -(1 << 34),
            y: (1 << 34) - 1,
            z: 7,
        }),
        add_as: (1 << 24) - 1,
        add_bs: (1 << 25) - 1,
        add_at: 0,
        add_bt: 1,
        horizontal_quality: 7,
        vertical_quality: 0,
        trailing_bits: Vec::new(),
    }
}

/// 1021 and 1022 round-trip, their names as 8-bit characters (every byte
/// value kept), 1022 with its 105-bit rotation point; a rotation point held
/// against the number, a character above U+00FF and a 32-character name are
/// refused. The residual grids and projections round-trip with their widths.
#[test]
fn transformation_messages_round_trip_with_their_field_widths() {
    let mut scale = helmert(1021);
    scale.ds = 1_234_567;
    assert!((scale.scale_correction_ppm() - 12.34567).abs() < 1.0e-12);
    for (number, rotation) in [(1021u16, 0usize), (1022, 105)] {
        let message = helmert(number);
        let body = message.encode().unwrap();
        // 12 + 5 + 8*8 + 5 + 2*8 + 8+10+5+4+2 + 19+20+14+14 + 3*23 + 3*32 + 25
        // + rotation + 24+25+24+25 + 3+3.
        let bits = 12 + 5 + 64 + 5 + 16 + 29 + 67 + 69 + 96 + 25 + rotation + 98 + 6;
        assert_eq!(body.len(), bits.div_ceil(8), "{number}");
        assert_eq!(HelmertTransformation::decode(&body).unwrap(), message);
        assert_eq!(
            Message::decode(&body).unwrap(),
            Message::HelmertTransformation(message)
        );
    }
    let mut m = helmert(1021);
    m.rotation_point = helmert(1022).rotation_point;
    assert!(m
        .encode()
        .unwrap_err()
        .to_string()
        .contains("carries no rotation point"));
    let mut m = helmert(1022);
    m.target_name = "\u{100}".to_string();
    assert!(m
        .encode()
        .unwrap_err()
        .to_string()
        .contains("not an 8-bit character"));
    let mut m = helmert(1022);
    m.source_name = "x".repeat(32);
    assert!(m
        .encode()
        .unwrap_err()
        .to_string()
        .contains("source name character count 32"));

    for (number, origin_bits, offset_bits) in [(1023u16, 43usize, 16usize), (1024, 51, 20)] {
        let mut residuals = [GridResidual::default(); RESIDUAL_GRID_POINTS];
        for (index, r) in residuals.iter_mut().enumerate() {
            *r = GridResidual {
                horizontal_1: index as i16 - 256,
                horizontal_2: 255 - index as i16,
                height: -(index as i16),
            };
        }
        let grid = ResidualGrid {
            message_number: number,
            system_id: 9,
            horizontal_shift: true,
            vertical_shift: false,
            origin_1: -1000,
            origin_2: if number == 1024 { (1 << 26) - 1 } else { -1 },
            extension_1: 4095,
            extension_2: 1,
            mean_offset_1: -128,
            mean_offset_2: 127,
            mean_height_offset: -16_384,
            residuals,
            horizontal_interpolation: 3,
            vertical_interpolation: 2,
            horizontal_quality: 7,
            vertical_quality: 1,
            mjd: 60_000,
            trailing_bits: Vec::new(),
        };
        let body = grid.encode().unwrap();
        let bits = 12 + 10 + origin_bits + 24 + offset_bits + 15 + 16 * 27 + 26;
        assert_eq!(body.len(), bits.div_ceil(8), "{number}");
        assert_eq!(ResidualGrid::decode(&body).unwrap(), grid);
        if number == 1024 {
            let mut negative = grid;
            negative.origin_2 = -1;
            assert!(negative
                .encode()
                .unwrap_err()
                .to_string()
                .contains("unsigned"));
        }
    }

    for (parameters, bits) in [
        (
            ProjectionParameters::NaturalOrigin {
                latitude: -(1 << 33),
                longitude: (1 << 34) - 1,
                add_scale: (1 << 30) - 1,
                false_easting: (1 << 36) - 1,
                false_northing: -(1 << 34),
            },
            26 + 34 + 35 + 30 + 36 + 35,
        ),
        (
            ProjectionParameters::LambertConicConformal {
                latitude: 1,
                longitude: -1,
                standard_parallel_1: 2,
                standard_parallel_2: -2,
                false_easting: 3,
                false_northing: -3,
            },
            26 + 34 + 35 + 34 + 34 + 36 + 35,
        ),
        (
            ProjectionParameters::ObliqueMercator {
                rectification: true,
                latitude: 5,
                longitude: 6,
                azimuth: (1 << 35) - 1,
                rectified_to_skew: -(1 << 25),
                add_scale: 9,
                easting: 10,
                northing: -11,
            },
            26 + 1 + 34 + 35 + 35 + 26 + 30 + 36 + 35,
        ),
    ] {
        let projection = Projection {
            system_id: 1,
            projection_type: 63,
            parameters,
            trailing_bits: Vec::new(),
        };
        let body = projection.encode().unwrap();
        let bits: usize = bits;
        assert_eq!(body.len(), bits.div_ceil(8), "{:?}", projection.parameters);
        assert_eq!(Projection::decode(&body).unwrap(), projection);
        assert_eq!(
            Message::decode(&body).unwrap().message_number(),
            projection.message_number()
        );
    }
}

/// 1013 announcements and a 1029 text round-trip; the 1029 text is the DF139
/// count of UTF-8 code units, the DF138 character count kept apart from it.
#[test]
fn system_parameters_and_text_round_trip() {
    let parameters = SystemParameters {
        reference_station_id: 4095,
        mjd: 61_000,
        seconds_of_day: 86_399,
        announcement_count: 2,
        leap_seconds: 18,
        announcements: vec![
            MessageAnnouncement {
                message_number: 1077,
                synchronous: true,
                interval: 10,
            },
            MessageAnnouncement {
                message_number: 1230,
                synchronous: false,
                interval: 65_535,
            },
        ],
        trailing_bits: Vec::new(),
    };
    let body = parameters.encode().unwrap();
    assert_eq!(
        body.len(),
        (12 + 12 + 16 + 17 + 5 + 8 + 2 * 29usize).div_ceil(8)
    );
    assert_eq!(SystemParameters::decode(&body).unwrap(), parameters);
    let mut wrong = parameters.clone();
    wrong.announcement_count = 1;
    assert!(wrong.encode().is_err());

    let text = TextMessage {
        reference_station_id: 1,
        mjd: 61_000,
        seconds_of_day: 1,
        character_count: 5,
        code_units: "Zürch".as_bytes().to_vec(),
        trailing_bits: Vec::new(),
    };
    let body = text.encode().unwrap();
    assert_eq!(
        body.len(),
        (12 + 12 + 16 + 17 + 7 + 8 + 6 * 8usize).div_ceil(8)
    );
    let decoded = TextMessage::decode(&body).unwrap();
    assert_eq!(decoded, text);
    assert_eq!(decoded.text().unwrap(), "Zürch");
    assert_eq!(Message::decode(&body).unwrap(), Message::Text(text.clone()));
    let mut long = text;
    long.code_units = vec![b'x'; 256];
    assert!(long
        .encode()
        .unwrap_err()
        .to_string()
        .contains("code unit count 256"));
}
