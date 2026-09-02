//! RTCM 3 differential-GNSS stream decoding and encoding.
//!
//! RTCM 10403.x ("RTCM Standard for Differential GNSS Services, Version 3") is
//! the dominant wire format for real-time GNSS correction and observation
//! streams: base-station observations, reference coordinates, antenna metadata,
//! and broadcast ephemerides flow from a caster to a rover as a sequence of
//! framed binary messages. This module is a sans-I/O codec for that stream,
//! built to the same shape as the crate's RINEX / SP3 / IONEX parsers:
//!
//! 1. a forgiving byte-level frame layer (`framing`) that syncs on the `0xD3`
//!    preamble, reads the 10-bit length, and verifies the 24-bit CRC-24Q;
//! 2. a format-agnostic canonical IR ([`Message`] and its typed variants) that
//!    stores each field as its raw transmitted integer; and
//! 3. an encoder that turns the IR back into bytes, so a decode followed by an
//!    encode round-trips byte-for-byte.
//!
//! ## Message coverage
//!
//! Decoded and encoded:
//!
//! | Message            | Numbers                                  | IR type |
//! |--------------------|------------------------------------------|---------|
//! | MSM4 observations  | 1074 / 1084 / 1094 / 1104 / 1114 / 1124 / 1134 | [`MsmMessage`] |
//! | MSM7 observations  | 1077 / 1087 / 1097 / 1107 / 1117 / 1127 / 1137 | [`MsmMessage`] |
//! | Station coordinates| 1005 / 1006                              | [`StationCoordinates`] |
//! | Antenna / receiver | 1007 / 1008 / 1033                       | [`AntennaDescriptor`] |
//! | GPS ephemeris      | 1019                                     | [`GpsEphemeris`] |
//! | GLONASS ephemeris  | 1020                                     | [`GlonassEphemeris`] |
//! | BeiDou ephemeris   | 1042                                     | [`BeidouEphemeris`] |
//! | QZSS ephemeris     | 1044                                     | [`QzssEphemeris`] |
//! | Galileo ephemeris  | 1045 / 1046                              | [`GalileoFnavEphemeris`] / [`GalileoInavEphemeris`] |
//!
//! Any other message number is preserved losslessly as [`Message::Unsupported`]
//! (its raw body is kept so the frame still round-trips). Deferred message types
//! include the other MSM variants (MSM1/2/3/5/6), the legacy L1/L1-L2
//! observation messages (1001-1004, 1009-1012), the network-RTK and SSR
//! correction families. They decode as `Unsupported` rather than erroring.
//!
//! ## Quick start
//!
//! ```
//! use sidereon_core::rtcm::{self, Message, StationCoordinates};
//!
//! // Build a 1006 reference-coordinate message and frame it.
//! let station = StationCoordinates {
//!     message_number: 1006,
//!     reference_station_id: 2003,
//!     itrf_realization_year: 0,
//!     gps_indicator: true,
//!     glonass_indicator: true,
//!     galileo_indicator: false,
//!     reference_station_indicator: false,
//!     ecef_x: 11_446_021_400,
//!     single_receiver_oscillator: false,
//!     reserved: false,
//!     ecef_y: -7_415_136_500,
//!     quarter_cycle_indicator: 0,
//!     ecef_z: 12_602_528_900,
//!     antenna_height: Some(15_000),
//! };
//! // A constructed message encodes either directly on the typed value or
//! // through the [`Message`] wrapper; both produce the same body bytes.
//! let body = station.encode();
//! assert_eq!(body, Message::StationCoordinates(station).encode());
//! let frame = rtcm::encode_frame(&body).unwrap();
//!
//! // Decode it back out of the framed stream.
//! let decoded = rtcm::decode_messages(&frame);
//! assert_eq!(decoded.len(), 1);
//! match &decoded[0] {
//!     Message::StationCoordinates(s) => assert_eq!(s.reference_station_id, 2003),
//!     _ => panic!("expected station coordinates"),
//! }
//! ```

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod antenna;
pub(crate) mod bits;
pub(crate) mod crc;
mod ephemeris;
mod framing;
mod lli;
mod msm;
mod ssr;
mod station;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;

use crate::error::Result;

use bits::BitReader;

pub use antenna::AntennaDescriptor;
pub use ephemeris::{
    BeidouEphemeris, GalileoFnavEphemeris, GalileoInavEphemeris, GlonassEphemeris, GpsEphemeris,
    QzssEphemeris,
};
pub use framing::{
    decode_frame, encode_frame, DecodedFrame, FrameScanner, FRAME_OVERHEAD, MAX_BODY_LEN, PREAMBLE,
};
pub use lli::{
    derive_lli, minimum_lock_time_ms, msm_epoch_dt_ms, msm_signal_rinex_code, CellLli,
    LockTimeTracker, PreviousLock, LLI_HALF_CYCLE, LLI_LOSS_OF_LOCK,
};
pub use msm::{MsmHeader, MsmKind, MsmMessage, MsmSatellite, MsmSignal};
pub use ssr::{
    SsrClockRecord, SsrCodeBiasRecord, SsrHeader, SsrKind, SsrMessage, SsrOrbitRecord,
    SsrPhaseBiasRecord, SsrPhaseBiasSignal,
};
pub use station::StationCoordinates;

/// A message whose number is recognized but whose body this codec does not
/// decode. The raw body is preserved so the frame still round-trips.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnsupportedMessage {
    /// The RTCM message number (read from the first 12 bits of the body).
    pub message_number: u16,
    /// The undecoded message body.
    pub body: Vec<u8>,
}

/// A decoded RTCM byte stream plus diagnostics for skipped frames.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RtcmStream {
    /// Every message decoded from a CRC-valid frame, in stream order.
    pub messages: Vec<Message>,
    /// Forgiving stream diagnostics for skipped bytes and skipped frames.
    pub diagnostics: StreamDiagnostics,
}

/// Diagnostics collected while scanning an RTCM byte stream.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StreamDiagnostics {
    /// Bytes skipped while resynchronizing on the next valid frame.
    pub resync_bytes: usize,
    /// CRC-valid frames whose body could not be decoded into the message IR.
    pub skipped_frames: Vec<FrameSkip>,
}

/// One CRC-valid frame that could not be decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameSkip {
    /// Byte offset of the frame preamble in the scanned buffer.
    pub offset: usize,
    /// RTCM message number when the body was long enough to carry one.
    pub message_number: Option<u16>,
    /// Why the body did not decode.
    pub reason: FrameSkipReason,
}

/// Typed reason for a skipped CRC-valid frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameSkipReason {
    /// The body ended before all required fields of its recognized type.
    Truncated,
    /// The body is internally inconsistent for its recognized type.
    Malformed(String),
}

pub(crate) type DecodeResult<T> = std::result::Result<T, DecodeError>;

#[derive(Debug)]
pub(crate) enum DecodeError {
    OutOfInput(bits::OutOfInput),
    Error(crate::error::Error),
}

impl From<bits::OutOfInput> for DecodeError {
    fn from(error: bits::OutOfInput) -> Self {
        Self::OutOfInput(error)
    }
}

impl From<crate::error::Error> for DecodeError {
    fn from(error: crate::error::Error) -> Self {
        Self::Error(error)
    }
}

impl From<DecodeError> for crate::error::Error {
    fn from(error: DecodeError) -> Self {
        match error {
            DecodeError::OutOfInput(error) => error.into(),
            DecodeError::Error(error) => error,
        }
    }
}

#[derive(Debug)]
struct DecodeFailure {
    kind: FrameSkipReason,
}

/// The canonical, format-agnostic RTCM 3 message IR.
///
/// Each variant stores raw transmitted field integers (see the per-type docs),
/// and [`Message::encode`] is the exact inverse of [`Message::decode`].
///
/// The variant set is the codec's full supported coverage; any other message
/// number decodes to [`Message::Unsupported`], so the enum is exhaustive and a
/// caller can both build any variant from scratch and match every case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Message {
    /// An MSM4 or MSM7 multi-signal observation message.
    Msm(MsmMessage),
    /// A 1005 / 1006 station antenna reference point.
    StationCoordinates(StationCoordinates),
    /// A 1007 / 1008 / 1033 antenna or receiver descriptor.
    AntennaDescriptor(AntennaDescriptor),
    /// A 1019 GPS broadcast ephemeris.
    GpsEphemeris(GpsEphemeris),
    /// A 1020 GLONASS broadcast ephemeris.
    GlonassEphemeris(GlonassEphemeris),
    /// A 1042 BeiDou broadcast ephemeris.
    BeidouEphemeris(BeidouEphemeris),
    /// A 1044 QZSS broadcast ephemeris.
    QzssEphemeris(QzssEphemeris),
    /// A 1045 Galileo F/NAV broadcast ephemeris.
    GalileoFnavEphemeris(GalileoFnavEphemeris),
    /// A 1046 Galileo I/NAV broadcast ephemeris.
    GalileoInavEphemeris(GalileoInavEphemeris),
    /// A supported RTCM SSR correction message.
    Ssr(SsrMessage),
    /// A recognized-but-undecoded message, preserved verbatim.
    Unsupported(UnsupportedMessage),
}

/// Read the 12-bit RTCM message number from the start of a message body.
///
/// Returns [`crate::Error::Parse`] if the body is shorter than 12 bits.
pub fn message_number(body: &[u8]) -> Result<u16> {
    message_number_classified(body).map_err(Into::into)
}

fn message_number_classified(body: &[u8]) -> DecodeResult<u16> {
    let mut r = BitReader::new(body);
    Ok(r.u(12)? as u16)
}

impl Message {
    /// Decode a single RTCM 3 message body (the bytes between a frame's length
    /// word and its CRC).
    ///
    /// Never errors on an unknown message number: an unrecognized type decodes
    /// to [`Message::Unsupported`]. Errors only on a truncated body of a
    /// recognized type.
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_inner(body).map_err(Into::into)
    }

    fn decode_inner(body: &[u8]) -> DecodeResult<Self> {
        let number = message_number_classified(body)?;
        let message = match number {
            1005 | 1006 => Message::StationCoordinates(StationCoordinates::decode_inner(body)?),
            1007 | 1008 | 1033 => {
                Message::AntennaDescriptor(AntennaDescriptor::decode_inner(body)?)
            }
            1019 => Message::GpsEphemeris(GpsEphemeris::decode_inner(body)?),
            1020 => Message::GlonassEphemeris(GlonassEphemeris::decode_inner(body)?),
            1042 => Message::BeidouEphemeris(BeidouEphemeris::decode_inner(body)?),
            1044 => Message::QzssEphemeris(QzssEphemeris::decode_inner(body)?),
            1045 => Message::GalileoFnavEphemeris(GalileoFnavEphemeris::decode_inner(body)?),
            1046 => Message::GalileoInavEphemeris(GalileoInavEphemeris::decode_inner(body)?),
            n if msm::is_supported_msm(n) => Message::Msm(MsmMessage::decode_inner(body)?),
            n if ssr::is_supported_ssr(n) => Message::Ssr(SsrMessage::decode_inner(body)?),
            _ => Message::Unsupported(UnsupportedMessage {
                message_number: number,
                body: body.to_vec(),
            }),
        };
        Ok(message)
    }

    fn decode_classified(body: &[u8]) -> std::result::Result<Self, DecodeFailure> {
        Self::decode_inner(body).map_err(|error| DecodeFailure {
            kind: match error {
                DecodeError::OutOfInput(_) => FrameSkipReason::Truncated,
                DecodeError::Error(crate::error::Error::Parse(message)) => {
                    FrameSkipReason::Malformed(message)
                }
                DecodeError::Error(other) => FrameSkipReason::Malformed(other.to_string()),
            },
        })
    }

    /// Encode this message back into a body (without the transport frame).
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Message::Msm(m) => m.encode(),
            Message::StationCoordinates(s) => s.encode(),
            Message::AntennaDescriptor(a) => a.encode(),
            Message::GpsEphemeris(e) => e.encode(),
            Message::GlonassEphemeris(e) => e.encode(),
            Message::BeidouEphemeris(e) => e.encode(),
            Message::QzssEphemeris(e) => e.encode(),
            Message::GalileoFnavEphemeris(e) => e.encode(),
            Message::GalileoInavEphemeris(e) => e.encode(),
            Message::Ssr(s) => s.encode(),
            Message::Unsupported(u) => u.body.clone(),
        }
    }

    /// The RTCM message number this IR encodes to.
    pub fn message_number(&self) -> u16 {
        match self {
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
        }
    }

    /// Decode this message and wrap it in a fresh RTCM transport frame.
    ///
    /// Returns [`crate::Error::InvalidInput`] if the encoded body exceeds the frame
    /// length limit.
    pub fn to_frame(&self) -> Result<Vec<u8>> {
        encode_frame(&self.encode())
    }
}

/// Decode every CRC-valid frame in a byte buffer into the message IR.
///
/// Frames whose CRC fails, or whose body cannot be decoded, are skipped; the
/// scan resynchronizes on the next preamble. This is the forgiving stream entry
/// point for a noisy serial feed.
pub fn decode_messages(bytes: &[u8]) -> Vec<Message> {
    decode_stream(bytes).messages
}

/// Decode every CRC-valid frame while recording forgiving stream diagnostics.
///
/// Unknown message numbers decode to [`Message::Unsupported`] values and are
/// not diagnostics. CRC-valid frames for recognized message types whose body
/// cannot be decoded are skipped and recorded in [`RtcmStream::diagnostics`].
pub fn decode_stream(bytes: &[u8]) -> RtcmStream {
    let mut stream = RtcmStream {
        messages: Vec::new(),
        diagnostics: StreamDiagnostics::default(),
    };
    let mut pos = 0usize;

    while pos < bytes.len() {
        let Some(rel) = bytes[pos..].iter().position(|&b| b == PREAMBLE) else {
            stream.diagnostics.resync_bytes += bytes.len() - pos;
            break;
        };
        stream.diagnostics.resync_bytes += rel;
        pos += rel;

        if bytes.len() - pos < FRAME_OVERHEAD {
            stream.diagnostics.resync_bytes += 1;
            pos += 1;
            continue;
        }

        let body_len = ((usize::from(bytes[pos + 1] & 0x03)) << 8) | usize::from(bytes[pos + 2]);
        let frame_len = 3 + body_len + 3;
        if bytes.len() - pos < frame_len {
            stream.diagnostics.resync_bytes += 1;
            pos += 1;
            continue;
        }

        match decode_frame(&bytes[pos..pos + frame_len]) {
            Ok(frame) => {
                match Message::decode_classified(frame.body) {
                    Ok(message) => stream.messages.push(message),
                    Err(failure) => stream.diagnostics.skipped_frames.push(FrameSkip {
                        offset: pos,
                        message_number: message_number(frame.body).ok(),
                        reason: failure.kind,
                    }),
                }
                pos += frame.frame_len;
            }
            Err(_) => {
                stream.diagnostics.resync_bytes += 1;
                pos += 1;
            }
        }
    }

    stream
}

/// Owns an RTCM carry buffer for chunked stream decoding.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SsrStreamAssembler {
    buf: Vec<u8>,
}

impl SsrStreamAssembler {
    /// Build an empty assembler.
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Append bytes and drain every complete CRC-valid frame.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<Result<Message>> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        let mut pos = 0usize;

        while pos < self.buf.len() {
            let Some(rel) = self.buf[pos..].iter().position(|&b| b == PREAMBLE) else {
                pos = self.buf.len();
                break;
            };
            pos += rel;
            if self.buf.len() - pos < FRAME_OVERHEAD {
                break;
            }

            let body_len =
                ((usize::from(self.buf[pos + 1] & 0x03)) << 8) | usize::from(self.buf[pos + 2]);
            let frame_len = 3 + body_len + 3;
            if self.buf.len() - pos < frame_len {
                break;
            }

            match decode_frame(&self.buf[pos..pos + frame_len]) {
                Ok(frame) => {
                    out.push(Message::decode(frame.body));
                    pos += frame.frame_len;
                }
                Err(_) => {
                    pos += 1;
                }
            }
        }

        if pos > 0 {
            self.buf.drain(..pos);
        }
        out
    }

    /// Number of bytes retained for the next chunk.
    pub fn retained_len(&self) -> usize {
        self.buf.len()
    }
}
