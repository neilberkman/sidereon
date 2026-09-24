#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::rtcm::{self, FrameScanner, LockTimeTracker, Message, RtcmPolicy};
use sidereon_core::Error;

fuzz_target!(|data: &[u8]| {
    for policy in [RtcmPolicy::Strict, RtcmPolicy::Lenient] {
        let stream = rtcm::decode_stream_with_policy(data, policy);
        assert!(stream.diagnostics.resync_bytes <= data.len());
        assert!(stream.diagnostics.skipped_frames.len() <= data.len());
        assert!(stream.diagnostics.crc_failures <= stream.diagnostics.resync_bytes);
        if policy == RtcmPolicy::Strict {
            assert!(stream.diagnostics.departures.is_empty());
        }

        let mut tracker = LockTimeTracker::new();
        for message in &stream.messages {
            if let Message::Msm(msm) = message {
                let cells = tracker.observe(msm);
                let phase_cells = msm
                    .signals
                    .iter()
                    .filter(|signal| signal.lock_time_indicator.is_some())
                    .count();
                assert_eq!(cells.len(), phase_cells);
            }
        }

        // Every message decoded from a frame re-encodes under the same policy
        // to that frame's body, or its encoder refuses it by name.
        for frame in FrameScanner::new(data) {
            let Ok((message, _)) = Message::decode_with_policy(frame.body, policy) else {
                continue;
            };
            match message.encode_with_policy(policy) {
                Ok((body, _)) => {
                    assert_eq!(body, frame.body, "message {}", message.message_number())
                }
                Err(error) => assert!(
                    matches!(error, Error::RtcmEncode(ref refusal) if !refusal.to_string().is_empty()),
                    "message {}: {error}",
                    message.message_number()
                ),
            }
        }
    }
});
