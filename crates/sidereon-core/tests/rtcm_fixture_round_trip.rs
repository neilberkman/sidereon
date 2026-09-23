//! Every real RTCM 3 frame in the fixtures decodes under the strict policy and
//! re-encodes to its exact bytes: the body the message encodes to is the body
//! the frame carries, and the frame built around it is the frame read.

use std::path::PathBuf;

use sidereon_core::rtcm::{
    self, decode_stream, encode_frame_with_reserved, FrameScanner, Message, RtcmPolicy,
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// The bytes of a hex fixture: every hex digit on its non-comment lines.
fn hex_fixture(name: &str) -> Vec<u8> {
    let text = std::fs::read_to_string(fixture(name)).expect("read hex fixture");
    let digits: Vec<u8> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'))
        .flat_map(|line| line.bytes().filter(u8::is_ascii_hexdigit))
        .collect();
    assert_eq!(digits.len() % 2, 0, "{name}: whole bytes");
    digits
        .chunks(2)
        .map(|pair| {
            u8::from_str_radix(std::str::from_utf8(pair).expect("ascii"), 16).expect("hex byte")
        })
        .collect()
}

/// Decode `bytes` strictly and check that every frame round-trips; return the
/// message numbers in stream order.
fn assert_round_trips(name: &str, bytes: &[u8], resync_bytes: usize) -> Vec<u16> {
    let stream = decode_stream(bytes);
    let diagnostics = &stream.diagnostics;
    assert!(
        diagnostics.skipped_frames.is_empty(),
        "{name}: skipped {:?}",
        diagnostics.skipped_frames
    );
    assert!(diagnostics.departures.is_empty(), "{name}");
    assert_eq!(diagnostics.crc_failures, 0, "{name}");
    assert_eq!(diagnostics.resync_bytes, resync_bytes, "{name}");

    let mut scanner = FrameScanner::new(bytes);
    let frames: Vec<_> = scanner.by_ref().collect();
    assert_eq!(scanner.resync_bytes(), resync_bytes, "{name}");
    assert_eq!(scanner.crc_failures(), 0, "{name}");
    assert_eq!(frames.len(), stream.messages.len(), "{name}");

    for (index, (frame, message)) in frames.iter().zip(&stream.messages).enumerate() {
        assert_eq!(frame.reserved, 0, "{name} frame {index}");
        assert!(
            !matches!(message, Message::Unsupported(_)),
            "{name} frame {index}: message {} is not decoded",
            message.message_number()
        );
        let body = message
            .encode()
            .unwrap_or_else(|err| panic!("{name} frame {index}: {err}"));
        assert_eq!(body, frame.body, "{name} frame {index} body");
        // The CRC-24Q is a function of the header and body, and the frame's
        // CRC verified, so equal header and body give the frame as read.
        let offset = frame.body.as_ptr() as usize - bytes.as_ptr() as usize - 3;
        assert_eq!(
            encode_frame_with_reserved(&body, frame.reserved).expect("re-frame"),
            &bytes[offset..offset + frame.frame_len],
            "{name} frame {index} bytes"
        );
        assert_eq!(
            Message::decode_with_policy(frame.body, RtcmPolicy::Strict)
                .expect("strict decode")
                .1,
            Vec::new(),
            "{name} frame {index}"
        );
    }
    stream
        .messages
        .iter()
        .map(Message::message_number)
        .collect()
}

#[test]
fn recorded_msm_and_station_stream_round_trips_frame_for_frame() {
    let bytes = std::fs::read(fixture("rtcm/gmsd7_20121014.rtcm3")).expect("read capture");
    // The capture is cut at 256 KiB, in the middle of a frame; those last 302
    // bytes are the only bytes outside a CRC-valid frame.
    let numbers = assert_round_trips("gmsd7", &bytes, 302);
    assert_eq!(numbers.len(), 1143);
    for (number, count) in [
        (1007, 28),
        (1008, 28),
        (1019, 15),
        (1020, 16),
        (1033, 28),
        (1077, 257),
        (1087, 257),
        (1117, 257),
        (1127, 257),
    ] {
        assert_eq!(
            numbers.iter().filter(|&&n| n == number).count(),
            count,
            "message {number}"
        );
    }
}

#[test]
fn recorded_ssr_stream_round_trips_frame_for_frame() {
    let bytes = std::fs::read(fixture("ssr/SSRA03IGS0_2026188140760_3epoch.rtcm3"))
        .expect("read SSR capture");
    let numbers = assert_round_trips("SSRA03IGS0", &bytes, 0);
    assert_eq!(numbers.len(), 29);
    for number in [1059, 1060, 1065, 1066, 1242, 1243, 1260, 1261] {
        assert!(numbers.contains(&number), "message {number}");
    }
}

#[test]
fn recorded_ephemeris_and_ssr_frames_round_trip_frame_for_frame() {
    for (name, number, frames) in [
        ("rtcm/BCEP00BKG0_20260708_1636_1042_frames.hex", 1042, 5),
        ("rtcm/BCEP00BKG0_20260708_1636_1044_frames.hex", 1044, 4),
        ("rtcm/BCEP00BKG0_20260708_1636_1045_frames.hex", 1045, 4),
        ("rtcm/SSRA00EUH0_20260708_1402_1046_frames.hex", 1046, 8),
        ("ssr/SSRA02IGS0_2026181234930_1060.hex", 1060, 1),
        ("ssr/SSRA02IGS0_2026181234930_1243.hex", 1243, 1),
    ] {
        let bytes = hex_fixture(name);
        let numbers = assert_round_trips(name, &bytes, 0);
        assert_eq!(numbers, vec![number; frames], "{name}");
        assert_eq!(
            rtcm::decode_messages(&bytes).expect("clean fixture").len(),
            frames,
            "{name}"
        );
    }
}
