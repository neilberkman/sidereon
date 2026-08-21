// Deferred NTRIP acceptance vectors: captured live-caster sourcetables and gpsd
// cross-checks stay out of this hermetic test file because they require external
// service captures or a local gpsd install. The committed SSR fixture below
// covers the payload-to-store path without network dependencies.

use sidereon_core::astro::time::{GnssWeekTow, TimeScale};
use sidereon_core::ntrip::{
    NtripClientMachine, NtripConfig, NtripEvent, NtripHandshake, NtripVersion,
};
use sidereon_core::rtcm::{self, Message, SsrStreamAssembler};
use sidereon_core::ssr::SsrCorrectionStore;
use sidereon_core::{GnssSatelliteId, GnssSystem};

const REAL_SSRA02IGS0_1060_FRAME_HEX: &str =
    include_str!("fixtures/ssr/SSRA02IGS0_2026181234930_1060.hex");
const REAL_SSR_WEEK: u32 = 2425;
const REAL_SSR_EPOCH_TOW_S: f64 = 344_970.0;

#[test]
fn ntrip_machine_payload_feeds_rtcm_assembler() {
    let frame = rtcm::encode_frame(&[0xff, 0xf0]).unwrap();
    let mut wire =
        b"HTTP/1.1 200 OK\r\nContent-Type: gnss/data\r\nTransfer-Encoding: chunked\r\n\r\n"
            .to_vec();
    wire.extend_from_slice(format!("{:X}\r\n", frame.len()).as_bytes());
    wire.extend_from_slice(&frame);
    wire.extend_from_slice(b"\r\n0\r\n\r\n");

    let mut machine = NtripClientMachine::new(NtripConfig {
        host: "caster.example.test".into(),
        port: 2101,
        mountpoint: "MOUNT".into(),
        version: NtripVersion::Rev2,
        credentials: None,
        user_agent_product: "test-client/0".into(),
        gga_interval_s: None,
    });
    machine.connection_request().unwrap();
    let events = machine.push(&wire);
    assert!(matches!(
        &events[0],
        NtripEvent::Connected(NtripHandshake {
            version: NtripVersion::Rev2,
            chunked: true,
            ..
        })
    ));

    let mut assembler = SsrStreamAssembler::new();
    let messages: Vec<_> = events
        .into_iter()
        .filter_map(|event| match event {
            NtripEvent::Payload(bytes) => Some(bytes),
            _ => None,
        })
        .flat_map(|bytes| assembler.push(&bytes))
        .collect();
    assert!(matches!(
        &messages[0],
        Ok(Message::Unsupported(message)) if message.message_number == 4095
    ));
}

#[test]
fn ntrip_machine_payload_feeds_ssr_correction_store() {
    let frame = hex_bytes(REAL_SSRA02IGS0_1060_FRAME_HEX);
    let split = frame.len() / 2;
    let mut first =
        b"HTTP/1.1 200 OK\r\nContent-Type: gnss/data\r\nTransfer-Encoding: chunked\r\n\r\n"
            .to_vec();
    first.extend_from_slice(format!("{:X}\r\n", frame.len()).as_bytes());
    first.extend_from_slice(&frame[..split]);

    let mut second = frame[split..].to_vec();
    second.extend_from_slice(b"\r\n0\r\n\r\n");

    let mut machine = NtripClientMachine::new(NtripConfig {
        host: "caster.example.test".into(),
        port: 2101,
        mountpoint: "MOUNT".into(),
        version: NtripVersion::Rev2,
        credentials: None,
        user_agent_product: "test-client/0".into(),
        gga_interval_s: None,
    });
    machine.connection_request().unwrap();

    let mut assembler = SsrStreamAssembler::new();
    let mut store = SsrCorrectionStore::new();
    let week = GnssWeekTow::new(TimeScale::Gpst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S).unwrap();
    let mut ssr_messages = 0;

    for events in [machine.push(&first), machine.push(&second)] {
        for event in events {
            if let NtripEvent::Payload(bytes) = event {
                for decoded in assembler.push(&bytes) {
                    let message = decoded.expect("decode SSR fixture frame");
                    if matches!(message, Message::Ssr(_)) {
                        ssr_messages += 1;
                    }
                    store.ingest(&message, week).expect("ingest SSR message");
                }
            }
        }
    }

    assert_eq!(assembler.retained_len(), 0);
    assert_eq!(ssr_messages, 1);
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
    assert!(store.orbit(sat).is_some());
    assert!(store.clock(sat).is_some());
}

fn hex_bytes(hex: &str) -> Vec<u8> {
    let compact: String = hex.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    assert_eq!(compact.len() % 2, 0);
    compact
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| {
            let hi = (chunk[0] as char).to_digit(16).unwrap();
            let lo = (chunk[1] as char).to_digit(16).unwrap();
            ((hi << 4) | lo) as u8
        })
        .collect()
}
