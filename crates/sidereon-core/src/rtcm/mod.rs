//! RTCM 3 differential-GNSS stream decoding and encoding.
//!
//! RTCM 10403.x ("RTCM Standard for Differential GNSS Services, Version 3") is
//! the dominant wire format for real-time GNSS correction and observation
//! streams: base-station observations, reference coordinates, antenna metadata,
//! and broadcast ephemerides flow from a caster to a rover as a sequence of
//! framed binary messages. This module is a sans-I/O codec for that stream,
//! built to the same shape as the crate's RINEX / SP3 / IONEX parsers:
//!
//! 1. a forgiving byte-level frame layer ([`framing`]) that syncs on the `0xD3`
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
//!
//! Any other message number is preserved losslessly as [`Message::Unsupported`]
//! (its raw body is kept so the frame still round-trips). Deferred message types
//! include the other MSM variants (MSM1/2/3/5/6), the legacy L1/L1-L2
//! observation messages (1001-1004, 1009-1012), the network-RTK and SSR
//! correction families, and the Galileo / BeiDou / QZSS ephemerides
//! (1042-1046). They decode as `Unsupported` rather than erroring.
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

mod antenna;
mod bits;
mod crc;
mod ephemeris;
mod framing;
mod msm;
mod station;

#[cfg(test)]
mod tests;

use crate::error::Result;

use bits::BitReader;

pub use antenna::AntennaDescriptor;
pub use ephemeris::{GlonassEphemeris, GpsEphemeris};
pub use framing::{
    decode_frame, encode_frame, DecodedFrame, FrameScanner, FRAME_OVERHEAD, MAX_BODY_LEN, PREAMBLE,
};
pub use msm::{MsmHeader, MsmKind, MsmMessage, MsmSatellite, MsmSignal};
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
    /// A recognized-but-undecoded message, preserved verbatim.
    Unsupported(UnsupportedMessage),
}

/// Read the 12-bit RTCM message number from the start of a message body.
///
/// Returns [`Error::Parse`] if the body is shorter than 12 bits.
pub fn message_number(body: &[u8]) -> Result<u16> {
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
        let number = message_number(body)?;
        let message = match number {
            1005 | 1006 => Message::StationCoordinates(StationCoordinates::decode(body)?),
            1007 | 1008 | 1033 => Message::AntennaDescriptor(AntennaDescriptor::decode(body)?),
            1019 => Message::GpsEphemeris(GpsEphemeris::decode(body)?),
            1020 => Message::GlonassEphemeris(GlonassEphemeris::decode(body)?),
            n if msm::is_supported_msm(n) => Message::Msm(MsmMessage::decode(body)?),
            _ => Message::Unsupported(UnsupportedMessage {
                message_number: number,
                body: body.to_vec(),
            }),
        };
        Ok(message)
    }

    /// Encode this message back into a body (without the transport frame).
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Message::Msm(m) => m.encode(),
            Message::StationCoordinates(s) => s.encode(),
            Message::AntennaDescriptor(a) => a.encode(),
            Message::GpsEphemeris(e) => e.encode(),
            Message::GlonassEphemeris(e) => e.encode(),
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
            Message::Unsupported(u) => u.message_number,
        }
    }

    /// Decode this message and wrap it in a fresh RTCM transport frame.
    ///
    /// Returns [`Error::InvalidInput`] if the encoded body exceeds the frame
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
    FrameScanner::new(bytes)
        .filter_map(|frame| Message::decode(frame.body).ok())
        .collect()
}
