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
//! | MSM1..MSM7 observations | 1071..1077 GPS, 1081..1087 GLONASS, 1091..1097 Galileo, 1101..1107 SBAS, 1111..1117 QZSS, 1121..1127 BeiDou, 1131..1137 NavIC | [`MsmMessage`] |
//! | Station coordinates| 1005 / 1006                              | [`StationCoordinates`] |
//! | Antenna / receiver | 1007 / 1008 / 1033                       | [`AntennaDescriptor`] |
//! | GPS ephemeris      | 1019                                     | [`GpsEphemeris`] |
//! | GLONASS ephemeris  | 1020                                     | [`GlonassEphemeris`] |
//! | BeiDou ephemeris   | 1042                                     | [`BeidouEphemeris`] |
//! | QZSS ephemeris     | 1044                                     | [`QzssEphemeris`] |
//! | Galileo ephemeris  | 1045 / 1046                              | [`GalileoFnavEphemeris`] / [`GalileoInavEphemeris`] |
//! | SSR corrections    | GPS 1057-1062, 1265; GLONASS 1063-1068; Galileo 1240-1245, 1267; QZSS 1246-1251, 1268; BeiDou 1258-1263, 1270 | [`SsrMessage`] |
//!
//! Any other message number is preserved losslessly as [`Message::Unsupported`]
//! (its raw body is kept so the frame still round-trips). Deferred message types
//! include the legacy L1/L1-L2
//! observation messages (1001-1004, 1009-1012), the NavIC ephemeris 1041, the
//! GLONASS code-phase biases 1230, the IGS SSR messages 4076, the network-RTK
//! correction families and the SSR messages not listed above. They decode as
//! `Unsupported` rather than erroring.
//!
//! ## Departures and policy
//!
//! Input whose every field can be read but which departs from the format -
//! nonzero frame reserved bits, bits after a message's last field other than
//! the zero byte alignment, an MSM cell mask over 64 bits, an SSR body that ends
//! before the records its header counts - is an
//! [`RtcmDeparture`]. Under [`RtcmPolicy::Strict`], the default, it is refused
//! by name; under [`RtcmPolicy::Lenient`] it is read and reported. The encoders
//! write every field in its own width and refuse by name a value they would
//! otherwise truncate, fill or drop.
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
//!     trailing_bits: Vec::new(),
//! };
//! // A constructed message encodes either directly on the typed value or
//! // through the [`Message`] wrapper; both produce the same body bytes.
//! let body = station.encode().unwrap();
//! assert_eq!(body, Message::StationCoordinates(station).encode().unwrap());
//! let frame = rtcm::encode_frame(&body).unwrap();
//!
//! // Decode it back out of the framed stream.
//! let decoded = rtcm::decode_messages(&frame).unwrap();
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
mod encode_error;
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
pub use encode_error::{
    MsmMaskProblem, MsmOptionalField, MsmOptionalProblem, RtcmConversionError, RtcmEncodeError,
    RtcmFieldEncoding, RtcmRecordKind,
};
pub use ephemeris::{
    BeidouEphemeris, GalileoFnavEphemeris, GalileoInavEphemeris, GlonassEphemeris, GpsEphemeris,
    QzssEphemeris,
};
pub use framing::{
    decode_frame, encode_frame, encode_frame_with_reserved, DecodedFrame, FrameScanner,
    FRAME_OVERHEAD, MAX_BODY_LEN, PREAMBLE,
};
pub use lli::{
    derive_lli, minimum_lock_time_ms, msm_epoch_dt_ms, msm_signal_rinex_code, CellLli,
    LockTimeTracker, PreviousLock, LLI_HALF_CYCLE, LLI_LOSS_OF_LOCK,
};
pub use msm::{
    msm_signal_mask, MsmHeader, MsmKind, MsmMessage, MsmSatellite, MsmSignal,
    MSM4_FINE_PHASE_RANGE_INVALID, MSM4_FINE_PSEUDORANGE_INVALID, MSM7_FINE_PHASE_RANGE_INVALID,
    MSM7_FINE_PSEUDORANGE_INVALID, MSM_FINE_PHASE_RANGE_RATE_INVALID,
    MSM_ROUGH_PHASE_RANGE_RATE_INVALID, MSM_ROUGH_RANGE_INVALID,
};
pub(crate) use ssr::is_native_qzss_ssr;
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

/// How the RTCM decoders and encoders treat input that departs from the RTCM 3
/// format while every field in it can still be read.
///
/// Each departure is an [`RtcmDeparture`]. Under [`RtcmPolicy::Strict`] the
/// frame or message is refused and the departure named; under
/// [`RtcmPolicy::Lenient`] it is read or written and the departure reported.
/// A CRC-24Q mismatch and a body that ends inside a field are refused under
/// both policies: no reading of those bits is known to be the message. An SSR
/// body that ends before the records its header counts is read under
/// [`RtcmPolicy::Lenient`] up to its last complete record
/// ([`RtcmDeparture::SsrRecordsShort`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RtcmPolicy {
    /// Refuse the first departure.
    #[default]
    Strict,
    /// Read or write the input and report each departure.
    Lenient,
}

/// A departure from the RTCM 3 format, refused under [`RtcmPolicy::Strict`]
/// and reported under [`RtcmPolicy::Lenient`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RtcmDeparture {
    /// The six reserved bits between the frame preamble and the length are not
    /// zero. The value is kept in [`DecodedFrame::reserved`] and written back
    /// by [`encode_frame_with_reserved`]; RTKLIB `input_rtcm3` reads past it.
    FrameReservedBits {
        /// The six reserved bits as read.
        reserved: u8,
    },
    /// Bits follow the last field of a message other than the fewer than eight
    /// zero bits that align the body to a byte. The message is read from its
    /// fields and keeps the bits after them in its `trailing_bits` (for SSR,
    /// `padding_bits`), which its encoder writes back after the fields.
    TrailingBits {
        /// The message number.
        message_number: u16,
        /// Every bit after the last field, in order.
        bits: Vec<bool>,
    },
    /// An MSM cell mask (DF396) longer than the 64 bits RTCM 10403 allows: the
    /// product of the satellite and signal counts exceeds 64. RTKLIB
    /// `decode_msm_head` refuses such a message. The mask and every cell are
    /// read as the counts state them.
    MsmCellMaskOver64 {
        /// The message number.
        message_number: u16,
        /// Satellite count times signal count.
        cells: usize,
    },
    /// An SSR body that ends before the records its header's satellite count
    /// (DF387) states. RTKLIB `decode_ssr1`..`decode_ssr7` read the complete
    /// records. The header count is kept as transmitted and the bits of the
    /// incomplete record in [`SsrMessage::padding_bits`].
    SsrRecordsShort {
        /// The message number.
        message_number: u16,
        /// The satellite count the header states.
        declared: usize,
        /// The complete records the body holds.
        read: usize,
    },
}

impl core::fmt::Display for RtcmDeparture {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::FrameReservedBits { reserved } => {
                write!(f, "RTCM frame reserved bits are {reserved:#04x}, not zero")
            }
            Self::TrailingBits {
                message_number,
                bits,
            } => write!(
                f,
                "RTCM {message_number} carries {} bits after its last field{}",
                bits.len(),
                if bits.len() < 8 { ", not all zero" } else { "" }
            ),
            Self::MsmCellMaskOver64 {
                message_number,
                cells,
            } => write!(
                f,
                "RTCM MSM {message_number} cell mask is {cells} bits, over the 64 RTCM allows"
            ),
            Self::SsrRecordsShort {
                message_number,
                declared,
                read,
            } => write!(
                f,
                "RTCM SSR {message_number} header counts {declared} satellites, and the body holds {read} complete records"
            ),
        }
    }
}

/// A decoded RTCM byte stream plus diagnostics for skipped bytes and frames.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RtcmStream {
    /// Every message decoded from a CRC-valid frame, in stream order.
    pub messages: Vec<Message>,
    /// Stream diagnostics for skipped bytes, CRC failures, skipped frames and
    /// departures read under [`RtcmPolicy::Lenient`].
    pub diagnostics: StreamDiagnostics,
}

/// Diagnostics collected while scanning an RTCM byte stream.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StreamDiagnostics {
    /// Bytes skipped while resynchronizing on the next valid frame: bytes
    /// before a preamble, a preamble whose frame fails its CRC-24Q or runs past
    /// the buffer, and a trailing partial frame.
    pub resync_bytes: usize,
    /// Preambles whose declared frame lay wholly within the buffer but failed
    /// its CRC-24Q. Each also counts one resync byte.
    pub crc_failures: usize,
    /// CRC-valid frames whose body could not be decoded into the message IR,
    /// or that departed from the format under [`RtcmPolicy::Strict`].
    pub skipped_frames: Vec<FrameSkip>,
    /// Departures read under [`RtcmPolicy::Lenient`], in stream order.
    pub departures: Vec<StreamDeparture>,
}

impl StreamDiagnostics {
    /// True when nothing was skipped, no CRC failed and no departure was read.
    pub fn is_clean(&self) -> bool {
        self.resync_bytes == 0
            && self.crc_failures == 0
            && self.skipped_frames.is_empty()
            && self.departures.is_empty()
    }
}

/// One departure read under [`RtcmPolicy::Lenient`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamDeparture {
    /// Byte offset of the frame preamble in the scanned stream.
    pub offset: usize,
    /// The departure.
    pub departure: RtcmDeparture,
}

/// One CRC-valid frame that could not be decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameSkip {
    /// Byte offset of the frame preamble in the scanned stream.
    pub offset: usize,
    /// RTCM message number when the body was long enough to carry one.
    pub message_number: Option<u16>,
    /// Why the body did not decode.
    pub reason: FrameSkipReason,
}

impl core::fmt::Display for FrameSkip {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "RTCM frame at byte {} (message ", self.offset)?;
        match self.message_number {
            Some(number) => write!(f, "{number}")?,
            None => write!(f, "unknown")?,
        }
        write!(f, ") was not decoded: ")?;
        match &self.reason {
            FrameSkipReason::Truncated => write!(f, "body truncated"),
            FrameSkipReason::Malformed(text) => write!(f, "{text}"),
            FrameSkipReason::Departure(departure) => {
                write!(f, "{departure} (refused under the strict policy)")
            }
        }
    }
}

impl FrameSkip {
    /// The skip as a [`crate::Error::Parse`] whose text is the skip's
    /// [`Display`](core::fmt::Display): the frame offset, the message number
    /// and the reason.
    pub fn to_error(&self) -> crate::error::Error {
        crate::error::Error::Parse(self.to_string())
    }
}

/// Typed reason for a skipped CRC-valid frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameSkipReason {
    /// The body ended before all required fields of its recognized type.
    Truncated,
    /// The body is internally inconsistent for its recognized type.
    Malformed(String),
    /// The frame or body departs from the format, refused under
    /// [`RtcmPolicy::Strict`].
    Departure(RtcmDeparture),
}

pub(crate) type DecodeResult<T> = std::result::Result<T, DecodeError>;

#[derive(Debug)]
pub(crate) enum DecodeError {
    OutOfInput(bits::OutOfInput),
    Error(crate::error::Error),
    Departure(RtcmDeparture),
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
            DecodeError::Departure(departure) => {
                crate::error::Error::Parse(format!("{departure} (refused under the strict policy)"))
            }
        }
    }
}

/// The policy a decode runs under and the departures it has read.
pub(crate) struct DecodeContext {
    policy: RtcmPolicy,
    departures: Vec<RtcmDeparture>,
}

impl DecodeContext {
    pub(crate) fn new(policy: RtcmPolicy) -> Self {
        Self {
            policy,
            departures: Vec::new(),
        }
    }

    pub(crate) fn policy(&self) -> RtcmPolicy {
        self.policy
    }

    pub(crate) fn into_departures(self) -> Vec<RtcmDeparture> {
        self.departures
    }

    /// Refuse `departure` under the strict policy, record it under the lenient.
    pub(crate) fn depart(&mut self, departure: RtcmDeparture) -> DecodeResult<()> {
        match self.policy {
            RtcmPolicy::Strict => Err(DecodeError::Departure(departure)),
            RtcmPolicy::Lenient => {
                self.departures.push(departure);
                Ok(())
            }
        }
    }

    /// Read what follows a message's last field. Fewer than eight zero bits
    /// are the byte alignment and give an empty tail; anything else is a
    /// departure, refused under the strict policy and returned under the
    /// lenient one for the message to keep.
    pub(crate) fn finish(
        &mut self,
        r: &mut BitReader<'_>,
        message_number: u16,
    ) -> DecodeResult<Vec<bool>> {
        let bits = r.rest();
        if !is_departing_tail(&bits) {
            return Ok(Vec::new());
        }
        self.depart(RtcmDeparture::TrailingBits {
            message_number,
            bits: bits.clone(),
        })?;
        Ok(bits)
    }
}

/// Whether the bits after a message's last field, up to the end of its body,
/// are anything other than the fewer than eight zero bits of byte alignment.
pub(crate) fn is_departing_tail(bits: &[bool]) -> bool {
    bits.len() >= 8 || bits.iter().any(|&bit| bit)
}

/// A decoded message type that keeps the bits after its last field.
pub(crate) trait TrailingBits {
    /// The kept bits after the last field.
    fn trailing_bits_mut(&mut self) -> &mut Vec<bool>;
}

/// Decode a fixed-layout message body with `read`, then read what follows its
/// last field under `ctx` into the message's trailing bits.
pub(crate) fn decode_body<T: TrailingBits>(
    body: &[u8],
    ctx: &mut DecodeContext,
    read: impl FnOnce(&mut BitReader<'_>, &mut DecodeContext) -> DecodeResult<T>,
) -> DecodeResult<T> {
    let message_number = message_number_classified(body)?;
    let mut r = BitReader::new(body);
    let mut value = read(&mut r, ctx)?;
    *value.trailing_bits_mut() = ctx.finish(&mut r, message_number)?;
    Ok(value)
}

/// Write a message's kept trailing bits after its last field under `policy`.
///
/// An empty tail writes nothing. A nonempty tail is a
/// [`RtcmDeparture::TrailingBits`]: refused under the strict policy, written
/// and reported under the lenient one. A tail that, with the zero bits that
/// align the body, would read back as the alignment alone is refused under
/// both, since it would not be read back into `trailing_bits`.
pub(crate) fn write_trailing(
    w: &mut bits::FieldWriter,
    bits: &[bool],
    policy: RtcmPolicy,
) -> Result<Vec<RtcmDeparture>> {
    let message_number = w.message_number();
    if bits.is_empty() {
        return Ok(Vec::new());
    }
    let pad = (8 - (w.bit_len() + bits.len()) % 8) % 8;
    let mut read_back = bits.to_vec();
    read_back.extend(std::iter::repeat_n(false, pad));
    if !is_departing_tail(&read_back) {
        return Err(RtcmEncodeError::TrailingZeroBits {
            message_number,
            bits: bits.len(),
        }
        .into());
    }
    let departure = RtcmDeparture::TrailingBits {
        message_number,
        bits: read_back,
    };
    match policy {
        RtcmPolicy::Strict => Err(RtcmEncodeError::StrictDeparture(departure).into()),
        RtcmPolicy::Lenient => {
            for &bit in bits {
                w.flag(bit);
            }
            Ok(vec![departure])
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
    /// An MSM1 through MSM7 multi-signal observation message.
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
    /// word and its CRC) under [`RtcmPolicy::Strict`].
    ///
    /// Never errors on an unknown message number: an unrecognized type decodes
    /// to [`Message::Unsupported`]. Errors on a truncated body of a recognized
    /// type and on an [`RtcmDeparture`] in its body.
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_with_policy(body, RtcmPolicy::Strict).map(|(message, _)| message)
    }

    /// Decode a single RTCM 3 message body under `policy`, returning the
    /// departures read under [`RtcmPolicy::Lenient`] (always empty under
    /// [`RtcmPolicy::Strict`], which refuses them).
    pub fn decode_with_policy(
        body: &[u8],
        policy: RtcmPolicy,
    ) -> Result<(Self, Vec<RtcmDeparture>)> {
        let mut ctx = DecodeContext::new(policy);
        let message = Self::decode_inner(body, &mut ctx)?;
        Ok((message, ctx.departures))
    }

    fn decode_inner(body: &[u8], ctx: &mut DecodeContext) -> DecodeResult<Self> {
        let number = message_number_classified(body)?;
        let message = match number {
            1005 | 1006 => Message::StationCoordinates(decode_body(body, ctx, |r, _| {
                StationCoordinates::read(r)
            })?),
            1007 | 1008 | 1033 => Message::AntennaDescriptor(decode_body(body, ctx, |r, _| {
                AntennaDescriptor::read(r)
            })?),
            1019 => Message::GpsEphemeris(decode_body(body, ctx, |r, _| GpsEphemeris::read(r))?),
            1020 => {
                Message::GlonassEphemeris(decode_body(body, ctx, |r, _| GlonassEphemeris::read(r))?)
            }
            1042 => {
                Message::BeidouEphemeris(decode_body(body, ctx, |r, _| BeidouEphemeris::read(r))?)
            }
            1044 => Message::QzssEphemeris(decode_body(body, ctx, |r, _| QzssEphemeris::read(r))?),
            1045 => Message::GalileoFnavEphemeris(decode_body(body, ctx, |r, _| {
                GalileoFnavEphemeris::read(r)
            })?),
            1046 => Message::GalileoInavEphemeris(decode_body(body, ctx, |r, _| {
                GalileoInavEphemeris::read(r)
            })?),
            n if msm::is_supported_msm(n) => {
                Message::Msm(decode_body(body, ctx, MsmMessage::read)?)
            }
            n if ssr::is_supported_ssr(n) => Message::Ssr(SsrMessage::decode_inner(body, ctx)?),
            _ => Message::Unsupported(UnsupportedMessage {
                message_number: number,
                body: body.to_vec(),
            }),
        };
        Ok(message)
    }

    fn decode_classified(
        body: &[u8],
        policy: RtcmPolicy,
    ) -> std::result::Result<(Self, Vec<RtcmDeparture>), DecodeFailure> {
        let mut ctx = DecodeContext::new(policy);
        match Self::decode_inner(body, &mut ctx) {
            Ok(message) => Ok((message, ctx.departures)),
            Err(error) => Err(DecodeFailure {
                kind: match error {
                    DecodeError::OutOfInput(_) => FrameSkipReason::Truncated,
                    DecodeError::Error(crate::error::Error::Parse(message)) => {
                        FrameSkipReason::Malformed(message)
                    }
                    DecodeError::Error(other) => FrameSkipReason::Malformed(other.to_string()),
                    DecodeError::Departure(departure) => FrameSkipReason::Departure(departure),
                },
            }),
        }
    }

    /// Encode this message back into a body (without the transport frame)
    /// under [`RtcmPolicy::Strict`].
    ///
    /// # Errors
    ///
    /// [`crate::Error::RtcmEncode`], an [`RtcmEncodeError`] naming the message
    /// and the field, when the message cannot be written as its wire form states it: a field value
    /// wider than its field, a message number that does not name the variant's
    /// layout, an optional part present where the message has none or absent
    /// where it has one, or an [`RtcmDeparture`]. The encoder never truncates,
    /// fills or drops a value; see the per-type `encode` methods.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this message under `policy`, returning the departures written
    /// under [`RtcmPolicy::Lenient`]. Every refusal of [`Message::encode`]
    /// other than a departure applies under both policies.
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        match self {
            Message::Msm(m) => m.encode_with_policy(policy),
            Message::StationCoordinates(s) => s.encode_with_policy(policy),
            Message::AntennaDescriptor(a) => a.encode_with_policy(policy),
            Message::GpsEphemeris(e) => e.encode_with_policy(policy),
            Message::GlonassEphemeris(e) => e.encode_with_policy(policy),
            Message::BeidouEphemeris(e) => e.encode_with_policy(policy),
            Message::QzssEphemeris(e) => e.encode_with_policy(policy),
            Message::GalileoFnavEphemeris(e) => e.encode_with_policy(policy),
            Message::GalileoInavEphemeris(e) => e.encode_with_policy(policy),
            Message::Ssr(s) => s.encode_with_policy(policy),
            Message::Unsupported(u) => u.encode().map(|body| (body, Vec::new())),
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

    /// Encode this message and wrap it in a fresh RTCM transport frame.
    ///
    /// Returns [`crate::Error::RtcmEncode`] if the body cannot be encoded (see
    /// [`Message::encode`]) or exceeds the frame length limit.
    pub fn to_frame(&self) -> Result<Vec<u8>> {
        encode_frame(&self.encode()?)
    }
}

impl UnsupportedMessage {
    /// The preserved body, checked to decode back to this message.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RtcmEncode`] when the body is shorter than the 12-bit
    /// message number, when its first 12 bits differ from
    /// [`Self::message_number`], or when the number is one this codec decodes
    /// into a typed variant: the body would then decode as that variant, or be
    /// refused, rather than come back as this message. Frame such a body with
    /// [`encode_frame`] directly.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let carried = message_number(&self.body).map_err(|_| {
            crate::error::Error::from(RtcmEncodeError::UnsupportedBodyTooShort {
                message_number: self.message_number,
            })
        })?;
        if carried != self.message_number {
            return Err(RtcmEncodeError::UnsupportedBodyNumber {
                message_number: self.message_number,
                carried,
            }
            .into());
        }
        if is_decoded_number(self.message_number) {
            return Err(RtcmEncodeError::UnsupportedDecodedNumber {
                message_number: self.message_number,
            }
            .into());
        }
        Ok(self.body.clone())
    }
}

/// Whether `number` decodes into a typed [`Message`] variant.
fn is_decoded_number(number: u16) -> bool {
    matches!(
        number,
        1005 | 1006 | 1007 | 1008 | 1033 | 1019 | 1020 | 1042 | 1044 | 1045 | 1046
    ) || msm::is_supported_msm(number)
        || ssr::is_supported_ssr(number)
}

/// Decode every frame of a complete RTCM byte stream under
/// [`RtcmPolicy::Strict`], refusing the stream unless every byte belongs to a
/// CRC-valid frame whose body decodes.
///
/// # Errors
///
/// [`crate::Error::Parse`] naming what was not read: resynchronized bytes (a
/// stray byte, a CRC-24Q failure or a trailing partial frame) or a skipped
/// frame. [`decode_stream`] reads a noisy stream frame by frame and reports
/// every skip in [`RtcmStream::diagnostics`].
pub fn decode_messages(bytes: &[u8]) -> Result<Vec<Message>> {
    let stream = decode_stream(bytes);
    let diagnostics = &stream.diagnostics;
    if let Some(skip) = diagnostics.skipped_frames.first() {
        return Err(crate::error::Error::Parse(format!(
            "{skip}; {} frames skipped",
            diagnostics.skipped_frames.len()
        )));
    }
    if diagnostics.resync_bytes > 0 {
        return Err(crate::error::Error::Parse(format!(
            "RTCM stream has {} bytes outside CRC-valid frames ({} CRC-24Q failures)",
            diagnostics.resync_bytes, diagnostics.crc_failures
        )));
    }
    Ok(stream.messages)
}

/// Decode every CRC-valid frame under [`RtcmPolicy::Strict`] while recording
/// stream diagnostics.
///
/// Unknown message numbers decode to [`Message::Unsupported`] values and are
/// not diagnostics. CRC-valid frames for recognized message types whose body
/// cannot be decoded, and frames that depart from the format, are skipped and
/// recorded in [`RtcmStream::diagnostics`]; bytes passed over while
/// resynchronizing and CRC-24Q failures are counted there.
pub fn decode_stream(bytes: &[u8]) -> RtcmStream {
    decode_stream_with_policy(bytes, RtcmPolicy::Strict)
}

/// Decode every CRC-valid frame under `policy` while recording stream
/// diagnostics; see [`decode_stream`]. Under [`RtcmPolicy::Lenient`] a frame
/// that departs from the format is read, and each departure is recorded in
/// [`StreamDiagnostics::departures`] with its frame offset.
pub fn decode_stream_with_policy(bytes: &[u8], policy: RtcmPolicy) -> RtcmStream {
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

        match decode_frame(&bytes[pos..]) {
            Ok(frame) => {
                read_frame(&frame, pos, policy, &mut stream.diagnostics, |message| {
                    stream.messages.push(message);
                });
                pos += frame.frame_len;
            }
            Err(_) => {
                if framing::frame_fails_crc(&bytes[pos..]) {
                    stream.diagnostics.crc_failures += 1;
                }
                stream.diagnostics.resync_bytes += 1;
                pos += 1;
            }
        }
    }

    stream
}

/// Decode one CRC-valid frame under `policy`, handing the message to `emit`
/// or recording why it was skipped. `offset` is the frame's position in the
/// stream.
fn read_frame(
    frame: &DecodedFrame<'_>,
    offset: usize,
    policy: RtcmPolicy,
    diagnostics: &mut StreamDiagnostics,
    emit: impl FnOnce(Message),
) {
    let skip = |diagnostics: &mut StreamDiagnostics, reason| {
        diagnostics.skipped_frames.push(FrameSkip {
            offset,
            message_number: message_number(frame.body).ok(),
            reason,
        });
    };
    let mut departures = Vec::new();
    if frame.reserved != 0 {
        let departure = RtcmDeparture::FrameReservedBits {
            reserved: frame.reserved,
        };
        match policy {
            RtcmPolicy::Strict => {
                skip(diagnostics, FrameSkipReason::Departure(departure));
                return;
            }
            RtcmPolicy::Lenient => departures.push(departure),
        }
    }
    match Message::decode_classified(frame.body, policy) {
        Ok((message, body_departures)) => {
            departures.extend(body_departures);
            diagnostics.departures.extend(
                departures
                    .into_iter()
                    .map(|departure| StreamDeparture { offset, departure }),
            );
            emit(message);
        }
        Err(failure) => skip(diagnostics, failure.kind),
    }
}

/// Owns an RTCM carry buffer for chunked stream decoding.
///
/// Bytes passed over while resynchronizing, CRC-24Q failures and departures
/// read under [`RtcmPolicy::Lenient`] accumulate in
/// [`SsrStreamAssembler::diagnostics`], with offsets counted from the first
/// byte pushed. A frame whose body does not decode, or that departs from the
/// format under [`RtcmPolicy::Strict`], is recorded in
/// [`StreamDiagnostics::skipped_frames`] and returned from
/// [`SsrStreamAssembler::push`] as its error.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SsrStreamAssembler {
    buf: Vec<u8>,
    policy: RtcmPolicy,
    /// Stream offset of `buf[0]`.
    drained: usize,
    diagnostics: StreamDiagnostics,
}

impl SsrStreamAssembler {
    /// Build an empty assembler that decodes under [`RtcmPolicy::Strict`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Build an empty assembler that decodes under `policy`.
    pub fn with_policy(policy: RtcmPolicy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }

    /// Append bytes and drain every complete CRC-valid frame.
    ///
    /// Each frame yields its decoded message, or the error that refused it: a
    /// truncated or malformed body, or under [`RtcmPolicy::Strict`] a
    /// departure. A trailing partial frame is kept for the next chunk.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<Result<Message>> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        let mut pos = 0usize;

        while pos < self.buf.len() {
            let Some(rel) = self.buf[pos..].iter().position(|&b| b == PREAMBLE) else {
                self.diagnostics.resync_bytes += self.buf.len() - pos;
                pos = self.buf.len();
                break;
            };
            self.diagnostics.resync_bytes += rel;
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
                    out.push(Self::decode_frame_message(
                        self.policy,
                        &mut self.diagnostics,
                        &frame,
                        self.drained + pos,
                    ));
                    pos += frame.frame_len;
                }
                Err(_) => {
                    self.diagnostics.crc_failures += 1;
                    self.diagnostics.resync_bytes += 1;
                    pos += 1;
                }
            }
        }

        if pos > 0 {
            self.buf.drain(..pos);
            self.drained += pos;
        }
        out
    }

    fn decode_frame_message(
        policy: RtcmPolicy,
        diagnostics: &mut StreamDiagnostics,
        frame: &DecodedFrame<'_>,
        offset: usize,
    ) -> Result<Message> {
        let skipped = diagnostics.skipped_frames.len();
        let mut decoded = None;
        read_frame(frame, offset, policy, diagnostics, |message| {
            decoded = Some(message);
        });
        match decoded {
            Some(message) => Ok(message),
            None => Err(diagnostics.skipped_frames[skipped].to_error()),
        }
    }

    /// Read the retained bytes as the end of the stream and return their
    /// frames.
    ///
    /// [`Self::push`] keeps the bytes from a preamble whose declared frame runs
    /// past the data so far, waiting for the rest. At the end of the stream no
    /// rest comes: the preamble is either a partial frame or a byte that only
    /// looks like one, and a whole frame can follow it. This scans the retained
    /// bytes as [`decode_stream`] scans a buffer, counting every byte outside
    /// a frame in [`StreamDiagnostics::resync_bytes`], and leaves the
    /// assembler empty.
    pub fn finish(&mut self) -> Vec<Result<Message>> {
        let buf = std::mem::take(&mut self.buf);
        let mut out = Vec::new();
        let mut pos = 0usize;
        while pos < buf.len() {
            let Some(rel) = buf[pos..].iter().position(|&b| b == PREAMBLE) else {
                self.diagnostics.resync_bytes += buf.len() - pos;
                break;
            };
            self.diagnostics.resync_bytes += rel;
            pos += rel;
            match decode_frame(&buf[pos..]) {
                Ok(frame) => {
                    out.push(Self::decode_frame_message(
                        self.policy,
                        &mut self.diagnostics,
                        &frame,
                        self.drained + pos,
                    ));
                    pos += frame.frame_len;
                }
                Err(_) => {
                    if framing::frame_fails_crc(&buf[pos..]) {
                        self.diagnostics.crc_failures += 1;
                    }
                    self.diagnostics.resync_bytes += 1;
                    pos += 1;
                }
            }
        }
        self.drained += buf.len();
        out
    }

    /// Number of bytes retained for the next chunk.
    pub fn retained_len(&self) -> usize {
        self.buf.len()
    }

    /// Diagnostics accumulated over every chunk pushed so far.
    pub fn diagnostics(&self) -> &StreamDiagnostics {
        &self.diagnostics
    }
}
