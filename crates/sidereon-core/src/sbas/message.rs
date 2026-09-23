use core::fmt;

use crate::error::{Error, Result};
use crate::rtcm::bits::{BitReader, BitWriter};
use crate::rtcm::crc::crc24q_bits;

/// The six-bit SBAS message-type field carried as an unsigned byte.
pub type SbasMessageType = u8;

const FRAMED_LEN: usize = 32;
const BODY_LEN: usize = 29;
const HEADER_BITS: usize = 14;
const DATA_BITS: usize = 212;
const BODY_BITS: usize = HEADER_BITS + DATA_BITS;
const CRC_BITS: usize = 24;
const FRAMED_BITS: usize = BODY_BITS + CRC_BITS;
/// The bits completing the last byte of either form: 226 + 6 = 29 * 8 and
/// 250 + 6 = 32 * 8.
const PAD_BITS: usize = 6;
const LONG_HALF_BITS: usize = 106;
/// Bytes holding a 212-bit raw payload: 26 full bytes and the four high bits
/// of a 27th.
const RAW_PAYLOAD_LEN: usize = 27;
const PREAMBLES: [u8; 3] = [0x53, 0x9A, 0xC6];
const MAX_MESSAGE_TYPE: u8 = 63;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
/// Raw reserved segments retained from an SBAS message payload.
///
/// Each tuple stores a segment's value and width in bits, in wire order. The
/// decoder fills every reserved segment a message layout defines, and the
/// encoder writes them back in place; it refuses a collection whose widths
/// differ from the layout or whose values do not fit their widths, rather
/// than inventing or dropping bits.
pub struct SpareBits(pub Vec<(u64, u8)>);

impl SpareBits {
    /// Create an empty collection of reserved wire segments.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Append one reserved segment with its raw value and bit width.
    pub fn push(&mut self, value: u64, width: u8) {
        self.0.push((value, width));
    }

    fn widths(&self) -> Vec<u8> {
        self.0.iter().map(|&(_, width)| width).collect()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// How the SBAS block codec and log readers treat input that departs from the
/// SBAS message or log format.
///
/// The codec reads one departure, a preamble other than the three eight-bit
/// values `0x53`, `0x9A` and `0xC6` that make up the SBAS preamble sequence.
/// The log readers read one, a record whose message-type field differs from
/// the type its message carries. A CRC mismatch in a framed block is refused
/// under both policies: the 226 bits then differ from the bits the CRC was
/// computed over, so no reading of them is known to be the message.
pub enum SbasPolicy {
    /// Refuse the first departure.
    #[default]
    Strict,
    /// Read or write the input and report each departure as an
    /// [`SbasDeparture`].
    Lenient,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
/// A departure from the SBAS message format read or written under
/// [`SbasPolicy::Lenient`].
pub enum SbasDeparture {
    /// The first eight bits are none of `0x53`, `0x9A` and `0xC6`. The value is
    /// kept in the message and written back as read.
    UnrecognizedPreamble {
        /// The preamble as read.
        preamble: u8,
    },
    /// A log record's message-type field differs from the six-bit type its
    /// message carries at message bits 8 through 13 (zero-based). Both values
    /// are kept: the field in [`crate::sbas::SbasLogBlock::declared_message_type`]
    /// and the message in its bytes.
    DeclaredMessageType {
        /// The message type the record's field states.
        declared: SbasMessageType,
        /// The message type the record's message carries.
        carried: SbasMessageType,
    },
}

impl fmt::Display for SbasDeparture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnrecognizedPreamble { preamble } => {
                write!(
                    f,
                    "SBAS preamble 0x{preamble:02X} is not 0x53, 0x9A or 0xC6"
                )
            }
            Self::DeclaredMessageType { declared, carried } => write!(
                f,
                "record declares SBAS message type {declared} but its message carries type {carried}"
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
/// A value [`SbasBlock::encode`] cannot write as the SBAS wire form states it.
///
/// The encoder writes every field in its own width and position and refuses
/// a value it would otherwise have to truncate, fill or drop.
pub enum SbasEncodeError {
    /// A field value does not fit the field's wire width.
    FieldOutOfRange {
        /// Message type being encoded.
        message_type: SbasMessageType,
        /// Field name, as the Rust field is named.
        field: &'static str,
        /// Array element, for a field that repeats.
        index: Option<usize>,
        /// The value held.
        value: i128,
        /// Field width in bits.
        width: u8,
        /// Whether the field is two's-complement signed.
        signed: bool,
    },
    /// The preamble is none of `0x53`, `0x9A` and `0xC6`, refused under
    /// [`SbasPolicy::Strict`].
    UnrecognizedPreamble {
        /// The preamble held.
        preamble: u8,
    },
    /// The message type cannot be carried by the variant that holds it: a
    /// fast-correction type outside 2..=5, an unsupported-message type that
    /// the decoder reads as a typed message, or a type wider than six bits.
    MessageType {
        /// The message type held.
        message_type: SbasMessageType,
        /// Why the type cannot be written.
        reason: &'static str,
    },
    /// A raw payload is not the 212-bit data field: 27 bytes whose last four
    /// bits are zero.
    RawPayload {
        /// Message type being encoded.
        message_type: SbasMessageType,
        /// Byte count held.
        bytes: usize,
        /// Whether bits past the 212th are set.
        bits_past_payload: bool,
    },
    /// Reserved segments whose widths differ from the message layout.
    ReservedLayout {
        /// Message type being encoded.
        message_type: SbasMessageType,
        /// The part of the message the segments belong to.
        part: &'static str,
        /// Segment widths the layout defines, in wire order.
        expected: Vec<u8>,
        /// Segment widths held.
        found: Vec<u8>,
    },
    /// A long-term half holding a record count its velocity code does not
    /// carry: one record with the velocity code set, two without it.
    LongTermRecordCount {
        /// Message type being encoded.
        message_type: SbasMessageType,
        /// Zero-based half within the message.
        half: usize,
        /// The half's velocity code.
        velocity_code: bool,
        /// Record count the velocity code carries.
        expected: usize,
        /// Record count held.
        found: usize,
    },
    /// A long-term record holding a value its half's velocity code does not
    /// carry: a rate, clock-drift or time-of-day value in a half without the
    /// velocity code.
    LongTermFieldNotCarried {
        /// Message type being encoded.
        message_type: SbasMessageType,
        /// Zero-based half within the message.
        half: usize,
        /// Zero-based record within the half.
        record: usize,
        /// Field name.
        field: &'static str,
    },
    /// A record in a half with the velocity code set holds no time of day,
    /// which that layout always carries.
    LongTermMissingTimeOfDay {
        /// Message type being encoded.
        message_type: SbasMessageType,
        /// Zero-based half within the message.
        half: usize,
    },
    /// Pad bits that do not fit the six bits completing the last byte.
    PadBits {
        /// The value held.
        value: u8,
    },
}

impl fmt::Display for SbasEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FieldOutOfRange {
                message_type,
                field,
                index,
                value,
                width,
                signed,
            } => {
                let kind = if *signed { "signed" } else { "unsigned" };
                match index {
                    Some(index) => write!(
                        f,
                        "SBAS message type {message_type} {field}[{index}] value {value} does not fit {width} {kind} bits"
                    ),
                    None => write!(
                        f,
                        "SBAS message type {message_type} {field} value {value} does not fit {width} {kind} bits"
                    ),
                }
            }
            Self::UnrecognizedPreamble { preamble } => {
                write!(f, "SBAS preamble 0x{preamble:02X} is not 0x53, 0x9A or 0xC6")
            }
            Self::MessageType {
                message_type,
                reason,
            } => write!(f, "SBAS message type {message_type}: {reason}"),
            Self::RawPayload {
                message_type,
                bytes,
                bits_past_payload,
            } => {
                if *bits_past_payload {
                    write!(
                        f,
                        "SBAS message type {message_type} raw payload sets bits past the 212-bit data field"
                    )
                } else {
                    write!(
                        f,
                        "SBAS message type {message_type} raw payload holds {bytes} bytes, not the {RAW_PAYLOAD_LEN} bytes of a 212-bit data field"
                    )
                }
            }
            Self::ReservedLayout {
                message_type,
                part,
                expected,
                found,
            } => write!(
                f,
                "SBAS message type {message_type} {part} reserved segment widths {found:?} differ from the layout {expected:?}"
            ),
            Self::LongTermRecordCount {
                message_type,
                half,
                velocity_code,
                expected,
                found,
            } => write!(
                f,
                "SBAS message type {message_type} long-term half {half} with velocity code {} holds {found} records, not {expected}",
                u8::from(*velocity_code)
            ),
            Self::LongTermFieldNotCarried {
                message_type,
                half,
                record,
                field,
            } => write!(
                f,
                "SBAS message type {message_type} long-term half {half} record {record} holds {field}, which a half without the velocity code does not carry"
            ),
            Self::LongTermMissingTimeOfDay { message_type, half } => write!(
                f,
                "SBAS message type {message_type} long-term half {half} has the velocity code set but no time of day"
            ),
            Self::PadBits { value } => {
                write!(f, "SBAS pad bits value {value} does not fit six bits")
            }
        }
    }
}

impl From<SbasEncodeError> for Error {
    fn from(error: SbasEncodeError) -> Self {
        Error::SbasEncode(Box::new(error))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// A decoded SBAS payload selected by its six-bit message type.
///
/// The decoder maps recognized phase-A IDs to typed records and retains
/// unsupported IDs with their raw payload bits.
pub enum SbasMessage {
    /// Message type 0; ingesting it disables the selected GEO for 60 seconds.
    DoNotUse(SbasDoNotUse),
    /// Message type 1; supplies the active 210-position satellite mask.
    PrnMask(SbasPrnMask),
    /// Message types 2 through 5; supplies fast pseudorange corrections.
    FastCorrections(SbasFastCorrections),
    /// Message type 6; supplies integrity indices for fast corrections.
    Integrity(SbasIntegrity),
    /// Message type 7; supplies fast-correction degradation information.
    FastDegradation(SbasFastDegradation),
    /// Message type 9; supplies raw GEO navigation state coefficients.
    GeoNav(SbasGeoNav),
    /// Message type 12, retained as an uninterpreted 212-bit payload.
    NetworkTime(SbasNetworkTime),
    /// Message type 17, retained as an uninterpreted 212-bit payload.
    GeoAlmanac(SbasGeoAlmanac),
    /// Message type 18; supplies a band- and IODI-qualified IGP mask.
    IgpMask(SbasIgpMask),
    /// Message type 24; combines six fast slots with one long-term half.
    MixedCorrections(SbasMixedCorrections),
    /// Message type 25; supplies two long-term correction halves.
    LongTermCorrections(SbasLongTermCorrections),
    /// Message type 26; supplies fifteen ionospheric delay entries.
    IonoDelays(SbasIonoDelays),
    /// A message ID outside the phase-A classification, retained as raw data.
    Unsupported(SbasUnsupported),
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The raw payload carried by SBAS message type 0.
pub struct SbasDoNotUse {
    /// The eight-bit preamble as read.
    pub preamble: u8,
    /// The 212-bit data field as 27 bytes, the last four bits zero. The
    /// encoder refuses any other length or a set bit past the 212th.
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The 210-position satellite mask and issue fields from message type 1.
pub struct SbasPrnMask {
    /// The eight-bit preamble as read.
    pub preamble: u8,
    /// The two-bit issue used by the correction store to select this mask.
    pub iodp: u8,
    /// Mask flags in wire order; true positions are resolved to monitored satellites.
    pub mask: [bool; 210],
    /// Empty: the mask and IODP fill the 212 data bits.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The thirteen fast-correction slots in message types 2 through 5.
pub struct SbasFastCorrections {
    /// The eight-bit preamble as read.
    pub preamble: u8,
    /// The original message ID, in the inclusive range 2 through 5.
    pub message_type: SbasMessageType,
    /// The two-bit issue matched against integrity blocks.
    pub iodf: u8,
    /// The two-bit issue matched against the active PRN mask.
    pub iodp: u8,
    /// Signed 12-bit pseudorange correction counts, scaled by the store at 0.125 meters.
    pub prc: [i16; 13],
    /// Four-bit integrity indices for the thirteen correction slots; 14 marks
    /// a satellite not monitored and 15 one not to use.
    pub udrei: [u8; 13],
    /// Empty: the typed fields fill the 212 data bits.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The integrity blocks and UDREI values from message type 6.
pub struct SbasIntegrity {
    /// The eight-bit preamble as read.
    pub preamble: u8,
    /// Four two-bit issue values, one for each group of thirteen UDREIs.
    pub iodf: [u8; 4],
    /// Fifty-one four-bit integrity indices applied to matching fast slots.
    pub udrei: [u8; 51],
    /// Empty: the typed fields fill the 212 data bits.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The latency and degradation indicators from message type 7.
///
/// The layout is RTKLIB `decode_sbstype7`'s: latency at message bit 14, IODP
/// at bit 18, two reserved bits at bit 20, and the fifty-one indicators from
/// bit 22.
pub struct SbasFastDegradation {
    /// The eight-bit preamble as read.
    pub preamble: u8,
    /// Four-bit latency, interpreted directly as seconds by the correction store.
    pub system_latency_s: u8,
    /// The two-bit issue that gates acceptance of the latency value.
    pub iodp: u8,
    /// Fifty-one four-bit degradation indicators preserved by the codec.
    pub ai: [u8; 51],
    /// The two reserved bits between the IODP and the first indicator.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The raw GEO navigation coefficients from message type 9.
pub struct SbasGeoNav {
    /// The eight-bit preamble as read.
    pub preamble: u8,
    /// A 13-bit time-of-day count in 16-second units.
    pub time_of_day_s: u16,
    /// The four-bit URA index retained with the navigation coefficients.
    pub ura: u8,
    /// Signed 30-bit raw X ECEF coordinate, scaled by the store at 0.08 meters.
    pub x_m: i32,
    /// Signed 30-bit raw Y ECEF coordinate, scaled by the store at 0.08 meters.
    pub y_m: i32,
    /// Signed 25-bit raw Z ECEF coordinate, scaled by the store at 0.4 meters.
    pub z_m: i32,
    /// Signed 17-bit raw X ECEF velocity, scaled by the store at 0.000625 meters per second.
    pub x_rate_m_s: i32,
    /// Signed 17-bit raw Y ECEF velocity, scaled by the store at 0.000625 meters per second.
    pub y_rate_m_s: i32,
    /// Signed 18-bit raw Z ECEF velocity, scaled by the store at 0.004 meters per second.
    pub z_rate_m_s: i32,
    /// Signed 10-bit raw X ECEF acceleration, scaled by the store at 0.0000125 meters per second squared.
    pub x_accel_m_s2: i16,
    /// Signed 10-bit raw Y ECEF acceleration, scaled by the store at 0.0000125 meters per second squared.
    pub y_accel_m_s2: i16,
    /// Signed 10-bit raw Z ECEF acceleration, scaled by the store at 0.0000625 meters per second squared.
    pub z_accel_m_s2: i16,
    /// Signed 12-bit raw GEO clock offset coefficient, scaled by the store at 1/2^31 seconds.
    pub a_gf0_s: i16,
    /// Signed 8-bit raw GEO clock drift coefficient, scaled by the store at 1/2^40 seconds per second.
    pub a_gf1_s_s: i16,
    /// The eight reserved bits before the time-of-day field.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The uninterpreted raw payload carried by message type 12.
pub struct SbasNetworkTime {
    /// The eight-bit preamble as read.
    pub preamble: u8,
    /// The 212-bit data field as 27 bytes, the last four bits zero.
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The uninterpreted raw payload carried by message type 17.
pub struct SbasGeoAlmanac {
    /// The eight-bit preamble as read.
    pub preamble: u8,
    /// The 212-bit data field as 27 bytes, the last four bits zero.
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The combined fast and long-term payload from message type 24.
pub struct SbasMixedCorrections {
    /// The eight-bit preamble as read.
    pub preamble: u8,
    /// The first 106 data bits, containing six fast slots and their issues.
    pub fast: SbasMixedFastCorrections,
    /// The second 106 data bits, containing one long-term half.
    pub long_term: SbasLongTermHalf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The six-slot fast portion of message type 24.
pub struct SbasMixedFastCorrections {
    /// The two-bit issue passed to the fast-correction ingest path.
    pub iodf: u8,
    /// The two-bit issue matched against the active PRN mask.
    pub iodp: u8,
    /// The two-bit block selector mapped to message types 2 through 5.
    pub block_id: u8,
    /// Signed 12-bit pseudorange correction counts, scaled at 0.125 meters by the store.
    pub prc: [i16; 6],
    /// Four-bit integrity indices for the six correction slots.
    pub udrei: [u8; 6],
    /// The four reserved bits between the fast and long-term portions.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The two long-term halves carried by message type 25.
pub struct SbasLongTermCorrections {
    /// The eight-bit preamble as read.
    pub preamble: u8,
    /// The two 106-bit halves, visited in order by correction-store ingest.
    pub halves: [SbasLongTermHalf; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// One 106-bit long-term correction half.
///
/// With the velocity code set the half carries one record with position,
/// clock, rate and clock-drift deltas and a time of day; without it, two
/// records with position and clock deltas only, and one reserved bit.
pub struct SbasLongTermHalf {
    /// Selects one velocity record when true, or two non-velocity records when false.
    pub velocity_code: bool,
    /// The two-bit issue matched against the active PRN mask.
    pub iodp: u8,
    /// One record in velocity mode, or two records in non-velocity mode. The
    /// encoder refuses any other count.
    pub records: Vec<SbasLongTermRecord>,
    /// The trailing reserved bit in non-velocity mode; empty in velocity mode.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Raw orbit and clock deltas for one monitored satellite.
///
/// Widths follow RTKLIB `decode_longcorr0` and `decode_longcorr1`: position
/// deltas are 9 bits and the clock delta 10 bits without the velocity code,
/// 11 bits each with it; rates and the clock drift are 8 bits and the time of
/// day 13 bits, carried only with the velocity code.
pub struct SbasLongTermRecord {
    /// A six-bit one-based index into the active monitored-satellite mask; 0
    /// fills an unused record slot.
    pub monitored_index: u8,
    /// The eight-bit issue used to select the broadcast ephemeris.
    pub iode: u8,
    /// Signed raw X ECEF delta, scaled at 0.125 meters by the store.
    pub delta_x: i32,
    /// Signed raw Y ECEF delta, scaled at 0.125 meters by the store.
    pub delta_y: i32,
    /// Signed raw Z ECEF delta, scaled at 0.125 meters by the store.
    pub delta_z: i32,
    /// Signed raw X ECEF rate, scaled at 2^-11 meters per second in velocity
    /// mode; zero without the velocity code.
    pub delta_x_rate: i32,
    /// Signed raw Y ECEF rate, scaled at 2^-11 meters per second in velocity
    /// mode; zero without the velocity code.
    pub delta_y_rate: i32,
    /// Signed raw Z ECEF rate, scaled at 2^-11 meters per second in velocity
    /// mode; zero without the velocity code.
    pub delta_z_rate: i32,
    /// Signed raw clock-offset delta, scaled at 1/2^31 seconds.
    pub delta_a_f0: i32,
    /// Signed raw clock-drift delta, scaled at 1/2^39 seconds per second in
    /// velocity mode; zero without the velocity code.
    pub delta_a_f1: i32,
    /// A velocity-mode time-of-day count in 16-second units, or `None` in non-velocity mode.
    pub time_of_day_s: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The band- and IODI-qualified 201-position mask from message type 18.
pub struct SbasIgpMask {
    /// The eight-bit preamble as read.
    pub preamble: u8,
    /// The four-bit ionospheric band selector.
    pub band_number: u8,
    /// The two-bit ionospheric issue used to match delay blocks.
    pub iodi: u8,
    /// Mask flags in wire order; active positions receive delay entries.
    pub mask: [bool; 201],
    /// Two segments: the four bits before the band number, which carry the
    /// count of IGP bands being broadcast and which RTKLIB does not read, and
    /// the one reserved bit after the mask.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Fifteen ionospheric delay/GIVE entries from message type 26.
pub struct SbasIonoDelays {
    /// The eight-bit preamble as read.
    pub preamble: u8,
    /// The four-bit ionospheric band selector.
    pub band_number: u8,
    /// The four-bit block selector, with fifteen entries per block.
    pub block_id: u8,
    /// The two-bit ionospheric issue matched against the IGP mask.
    pub iodi: u8,
    /// Fifteen entries assigned to consecutive active IGP positions.
    pub entries: [SbasIgpDelay; 15],
    /// The seven trailing reserved bits.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
/// One raw vertical-delay and GIVE pair from an ionospheric delay block.
pub struct SbasIgpDelay {
    /// A nine-bit delay count scaled at 0.125 meters by the store. The
    /// all-ones value 511 is DO-229's "do not use"; the codec keeps it as
    /// read and the store lists the point as unavailable rather than as a
    /// 63.875 m delay.
    pub vertical_delay: u16,
    /// A four-bit GIVE index; 15 marks an entry that is not monitored.
    pub givei: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// A non-phase-A SBAS message retained without a typed payload decoder.
pub struct SbasUnsupported {
    /// The eight-bit preamble as read.
    pub preamble: u8,
    /// The original six-bit message ID, one the decoder does not read as a
    /// typed message.
    pub message_type: SbasMessageType,
    /// The 212-bit data field as 27 bytes, the last four bits zero.
    pub data: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// The wire representation that controls block length and CRC handling.
pub enum SbasWireForm {
    /// A 32-byte block carrying 226 body bits, 24 CRC bits, and six pad bits.
    Framed250,
    /// A 29-byte body carrying 226 bits and six pad bits.
    Body226,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The decoded message and wire representation needed for re-encoding.
///
/// [`SbasBlock::decode`] fills every field, and [`SbasBlock::encode`] writes
/// them back, so an unedited decoded block encodes to the bytes it was read
/// from.
pub struct SbasBlock {
    /// The body-only or CRC-framed representation selected for this block.
    pub form: SbasWireForm,
    /// The typed or raw message decoded from the block body.
    pub message: SbasMessage,
    /// The six bits completing the last byte, as read: after the 226-bit body
    /// in [`SbasWireForm::Body226`], after the CRC in
    /// [`SbasWireForm::Framed250`]. RTKLIB `sbsdecodemsg` and `readmsgs` store
    /// them as zero (`msg[28]&=0xC0`), so `sbsoutmsg` writes zeros; a 29-byte
    /// body cut from a framed block carries the first six CRC bits here. Only
    /// the low six bits may be set.
    pub pad_bits: u8,
}

/// A bit writer that refuses a value wider than its field instead of masking
/// it, naming the message and field.
struct FieldWriter {
    writer: BitWriter,
    nbits: usize,
    message_type: SbasMessageType,
}

impl FieldWriter {
    fn new(message_type: SbasMessageType) -> Self {
        Self {
            writer: BitWriter::new(),
            nbits: 0,
            message_type,
        }
    }

    fn out_of_range(
        &self,
        field: &'static str,
        index: Option<usize>,
        value: i128,
        width: u8,
        signed: bool,
    ) -> SbasEncodeError {
        SbasEncodeError::FieldOutOfRange {
            message_type: self.message_type,
            field,
            index,
            value,
            width,
            signed,
        }
    }

    fn u_at(
        &mut self,
        field: &'static str,
        index: Option<usize>,
        value: u64,
        width: u8,
    ) -> core::result::Result<(), SbasEncodeError> {
        if width < 64 && value >> width != 0 {
            return Err(self.out_of_range(field, index, i128::from(value), width, false));
        }
        self.writer.push_u(value, usize::from(width));
        self.nbits += usize::from(width);
        Ok(())
    }

    fn u(
        &mut self,
        field: &'static str,
        value: u64,
        width: u8,
    ) -> core::result::Result<(), SbasEncodeError> {
        self.u_at(field, None, value, width)
    }

    fn i_at(
        &mut self,
        field: &'static str,
        index: Option<usize>,
        value: i64,
        width: u8,
    ) -> core::result::Result<(), SbasEncodeError> {
        let half = 1i64 << (width - 1);
        if value < -half || value >= half {
            return Err(self.out_of_range(field, index, i128::from(value), width, true));
        }
        self.writer.push_i(value, usize::from(width));
        self.nbits += usize::from(width);
        Ok(())
    }

    fn i(
        &mut self,
        field: &'static str,
        value: i64,
        width: u8,
    ) -> core::result::Result<(), SbasEncodeError> {
        self.i_at(field, None, value, width)
    }

    fn flag(&mut self, value: bool) {
        self.writer.push_flag(value);
        self.nbits += 1;
    }

    fn raw_bits(&mut self, bytes: &[u8], nbits: usize) {
        for bit_pos in 0..nbits {
            self.writer.push_u(u64::from(bit_at(bytes, bit_pos)), 1);
        }
        self.nbits += nbits;
    }

    /// Write one reserved segment whose width was already checked against
    /// the layout.
    fn spare(&mut self, (value, width): (u64, u8)) -> core::result::Result<(), SbasEncodeError> {
        self.u("reserved", value, width)
    }

    fn into_bytes(self) -> Vec<u8> {
        self.writer.into_bytes()
    }
}

/// Check that `spare` holds exactly the reserved segment widths `expected`,
/// in order, and return its segments.
fn reserved_layout<'a>(
    message_type: SbasMessageType,
    part: &'static str,
    spare: &'a SpareBits,
    expected: &[u8],
) -> core::result::Result<&'a [(u64, u8)], SbasEncodeError> {
    let found = spare.widths();
    if found != expected {
        return Err(SbasEncodeError::ReservedLayout {
            message_type,
            part,
            expected: expected.to_vec(),
            found,
        });
    }
    Ok(&spare.0)
}

impl SbasBlock {
    /// Decode a body or CRC-framed SBAS block under [`SbasPolicy::Strict`].
    ///
    /// Body input must be 29 bytes. Framed input must be 32 bytes and must
    /// contain a CRC-24Q matching the first 226 bits. Both forms require one
    /// of the three recognized SBAS preambles; invalid lengths, CRCs,
    /// preambles, and bit reads are returned as parse errors. The six pad
    /// bits are kept in [`SbasBlock::pad_bits`].
    pub fn decode(bytes: &[u8], form: SbasWireForm) -> Result<Self> {
        Self::decode_with_policy(bytes, form, SbasPolicy::Strict).map(|(block, _)| block)
    }

    /// Decode a body or CRC-framed SBAS block under `policy`.
    ///
    /// Under [`SbasPolicy::Lenient`] a preamble other than the three
    /// SBAS values is read, kept in the message, and reported; length, CRC
    /// and bit-read failures are refused under both policies.
    pub fn decode_with_policy(
        bytes: &[u8],
        form: SbasWireForm,
        policy: SbasPolicy,
    ) -> Result<(Self, Vec<SbasDeparture>)> {
        let pad_start = match form {
            SbasWireForm::Framed250 => {
                if bytes.len() != FRAMED_LEN {
                    return Err(parse_error("SBAS framed block must be 32 bytes"));
                }
                let got = bits_as_u32(bytes, BODY_BITS, CRC_BITS);
                let want = crc24q_bits(bytes, BODY_BITS);
                if got != want {
                    return Err(Error::Parse(format!(
                        "SBAS CRC mismatch: block carries 0x{got:06X}, body gives 0x{want:06X}"
                    )));
                }
                FRAMED_BITS
            }
            SbasWireForm::Body226 => {
                if bytes.len() != BODY_LEN {
                    return Err(parse_error("SBAS body block must be 29 bytes"));
                }
                BODY_BITS
            }
        };

        let mut departures = Vec::new();
        let mut reader = BitReader::new(bytes);
        let preamble = reader.u(8)? as u8;
        if !PREAMBLES.contains(&preamble) {
            match policy {
                SbasPolicy::Strict => {
                    return Err(Error::Parse(format!(
                        "SBAS preamble 0x{preamble:02X} not recognized"
                    )));
                }
                SbasPolicy::Lenient => {
                    departures.push(SbasDeparture::UnrecognizedPreamble { preamble });
                }
            }
        }
        let message_type = reader.u(6)? as u8;
        let data = read_bits_as_bytes(&mut reader, DATA_BITS)?;
        let message = decode_message(preamble, message_type, &data)?;
        let pad_bits = bits_as_u32(bytes, pad_start, PAD_BITS) as u8;
        Ok((
            Self {
                form,
                message,
                pad_bits,
            },
            departures,
        ))
    }

    /// Encode the block under [`SbasPolicy::Strict`].
    ///
    /// The body holds the preamble, the six-bit message ID and the 212-bit
    /// data field; framed output appends the CRC-24Q over those 226 bits.
    /// Both forms end with [`SbasBlock::pad_bits`]. A value that does not fit
    /// its field, a reserved-segment layout or long-term record count that
    /// differs from the message layout, a raw payload that is not 212 bits,
    /// and an unrecognized preamble are refused with an
    /// [`Error::SbasEncode`] naming the field.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(SbasPolicy::Strict)
            .map(|(bytes, _)| bytes)
    }

    /// Encode the block under `policy`.
    ///
    /// Under [`SbasPolicy::Lenient`] a preamble other than the three
    /// SBAS values is written as held and reported; every other refusal of
    /// [`SbasBlock::encode`] applies under both policies.
    pub fn encode_with_policy(&self, policy: SbasPolicy) -> Result<(Vec<u8>, Vec<SbasDeparture>)> {
        let mut departures = Vec::new();
        let preamble = self.message.preamble();
        if !PREAMBLES.contains(&preamble) {
            match policy {
                SbasPolicy::Strict => {
                    return Err(SbasEncodeError::UnrecognizedPreamble { preamble }.into());
                }
                SbasPolicy::Lenient => {
                    departures.push(SbasDeparture::UnrecognizedPreamble { preamble });
                }
            }
        }
        if u32::from(self.pad_bits) >> PAD_BITS != 0 {
            return Err(SbasEncodeError::PadBits {
                value: self.pad_bits,
            }
            .into());
        }
        let message_type = self.message.message_type();
        let data = self.message.payload()?;

        let mut body = FieldWriter::new(message_type);
        body.u("preamble", u64::from(preamble), 8)?;
        body.u("message_type", u64::from(message_type), 6)?;
        body.raw_bits(&data, DATA_BITS);
        let body = body.into_bytes();

        let mut out = FieldWriter::new(message_type);
        out.raw_bits(&body, BODY_BITS);
        if self.form == SbasWireForm::Framed250 {
            out.u("crc", u64::from(crc24q_bits(&body, BODY_BITS)), 24)?;
        }
        out.u("pad_bits", u64::from(self.pad_bits), 6)?;
        Ok((out.into_bytes(), departures))
    }
}

impl SbasMessage {
    /// Return the six-bit SBAS message ID represented by this value.
    ///
    /// Typed variants use their protocol IDs; fast corrections and unsupported
    /// messages return the ID stored in their records.
    pub fn message_type(&self) -> SbasMessageType {
        match self {
            Self::DoNotUse(_) => 0,
            Self::PrnMask(_) => 1,
            Self::FastCorrections(m) => m.message_type,
            Self::Integrity(_) => 6,
            Self::FastDegradation(_) => 7,
            Self::GeoNav(_) => 9,
            Self::NetworkTime(_) => 12,
            Self::GeoAlmanac(_) => 17,
            Self::IgpMask(_) => 18,
            Self::MixedCorrections(_) => 24,
            Self::LongTermCorrections(_) => 25,
            Self::IonoDelays(_) => 26,
            Self::Unsupported(m) => m.message_type,
        }
    }

    fn preamble(&self) -> u8 {
        match self {
            Self::DoNotUse(m) => m.preamble,
            Self::PrnMask(m) => m.preamble,
            Self::FastCorrections(m) => m.preamble,
            Self::Integrity(m) => m.preamble,
            Self::FastDegradation(m) => m.preamble,
            Self::GeoNav(m) => m.preamble,
            Self::NetworkTime(m) => m.preamble,
            Self::GeoAlmanac(m) => m.preamble,
            Self::IgpMask(m) => m.preamble,
            Self::MixedCorrections(m) => m.preamble,
            Self::LongTermCorrections(m) => m.preamble,
            Self::IonoDelays(m) => m.preamble,
            Self::Unsupported(m) => m.preamble,
        }
    }

    /// Check that the wire form carries this message as held.
    ///
    /// This is the validation [`SbasBlock::encode`] applies to the message and
    /// [`crate::sbas::SbasCorrectionStore::ingest`] applies before storing
    /// anything: the message type suits the variant, every field fits its
    /// width, reserved segments match the layout, raw payloads are 212 bits,
    /// and each long-term half holds the record count its velocity code
    /// carries, with a time of day under the velocity code and no rate,
    /// clock drift or time of day without it. The preamble is not checked
    /// here; [`SbasPolicy`] governs it.
    pub fn validate(&self) -> core::result::Result<(), SbasEncodeError> {
        self.payload().map(|_| ())
    }

    /// The 212-bit data field, after the checks [`SbasMessage::validate`]
    /// names.
    fn payload(&self) -> core::result::Result<Vec<u8>, SbasEncodeError> {
        self.check_message_type()?;
        let mut data = FieldWriter::new(self.message_type());
        self.encode_data(&mut data)?;
        debug_assert_eq!(data.nbits, DATA_BITS);
        Ok(data.into_bytes())
    }

    /// Refuse a message type the variant cannot carry, so that decoding the
    /// written block gives back the same variant.
    fn check_message_type(&self) -> core::result::Result<(), SbasEncodeError> {
        let message_type = self.message_type();
        let reason = match self {
            Self::FastCorrections(_) if !(2..=5).contains(&message_type) => {
                Some("fast corrections are message types 2 through 5")
            }
            Self::Unsupported(_) if message_type > MAX_MESSAGE_TYPE => {
                Some("the message-type field is six bits")
            }
            Self::Unsupported(_) if is_phase_a_message(message_type) => {
                Some("the decoder reads this type as a typed message, not an unsupported one")
            }
            _ => None,
        };
        match reason {
            Some(reason) => Err(SbasEncodeError::MessageType {
                message_type,
                reason,
            }),
            None => Ok(()),
        }
    }

    fn encode_data(&self, w: &mut FieldWriter) -> core::result::Result<(), SbasEncodeError> {
        match self {
            Self::DoNotUse(m) => encode_raw(w, &m.data),
            Self::PrnMask(m) => encode_prn_mask(w, m),
            Self::FastCorrections(m) => encode_fast(w, m),
            Self::Integrity(m) => encode_integrity(w, m),
            Self::FastDegradation(m) => encode_fast_degradation(w, m),
            Self::GeoNav(m) => encode_geo_nav(w, m),
            Self::NetworkTime(m) => encode_raw(w, &m.data),
            Self::GeoAlmanac(m) => encode_raw(w, &m.data),
            Self::IgpMask(m) => encode_igp_mask(w, m),
            Self::MixedCorrections(m) => encode_mixed(w, m),
            Self::LongTermCorrections(m) => encode_long_term(w, m),
            Self::IonoDelays(m) => encode_iono_delays(w, m),
            Self::Unsupported(m) => encode_raw(w, &m.data),
        }
    }
}

pub(crate) fn is_phase_a_message(mt: SbasMessageType) -> bool {
    matches!(mt, 0..=7 | 9 | 12 | 17 | 18 | 24 | 25 | 26)
}

fn decode_message(preamble: u8, message_type: u8, data: &[u8]) -> Result<SbasMessage> {
    if !is_phase_a_message(message_type) {
        return Ok(SbasMessage::Unsupported(SbasUnsupported {
            preamble,
            message_type,
            data: data.to_vec(),
        }));
    }
    match message_type {
        0 => Ok(SbasMessage::DoNotUse(SbasDoNotUse {
            preamble,
            data: data.to_vec(),
        })),
        1 => decode_prn_mask(preamble, data).map(SbasMessage::PrnMask),
        2..=5 => decode_fast(preamble, message_type, data).map(SbasMessage::FastCorrections),
        6 => decode_integrity(preamble, data).map(SbasMessage::Integrity),
        7 => decode_fast_degradation(preamble, data).map(SbasMessage::FastDegradation),
        9 => decode_geo_nav(preamble, data).map(SbasMessage::GeoNav),
        12 => Ok(SbasMessage::NetworkTime(SbasNetworkTime {
            preamble,
            data: data.to_vec(),
        })),
        17 => Ok(SbasMessage::GeoAlmanac(SbasGeoAlmanac {
            preamble,
            data: data.to_vec(),
        })),
        18 => decode_igp_mask(preamble, data).map(SbasMessage::IgpMask),
        24 => decode_mixed(preamble, data).map(SbasMessage::MixedCorrections),
        25 => decode_long_term(preamble, data).map(SbasMessage::LongTermCorrections),
        26 => decode_iono_delays(preamble, data).map(SbasMessage::IonoDelays),
        _ => unreachable!("phase A classification and decode match are out of sync"),
    }
}

fn encode_raw(w: &mut FieldWriter, data: &[u8]) -> core::result::Result<(), SbasEncodeError> {
    if data.len() != RAW_PAYLOAD_LEN || data[RAW_PAYLOAD_LEN - 1] & 0x0F != 0 {
        return Err(SbasEncodeError::RawPayload {
            message_type: w.message_type,
            bytes: data.len(),
            bits_past_payload: data.len() == RAW_PAYLOAD_LEN,
        });
    }
    w.raw_bits(data, DATA_BITS);
    Ok(())
}

fn decode_prn_mask(preamble: u8, data: &[u8]) -> Result<SbasPrnMask> {
    let mut r = BitReader::new(data);
    let mut mask = [false; 210];
    for bit in &mut mask {
        *bit = r.flag()?;
    }
    let iodp = r.u(2)? as u8;
    Ok(SbasPrnMask {
        preamble,
        iodp,
        mask,
        reserved: SpareBits::new(),
    })
}

fn encode_prn_mask(
    w: &mut FieldWriter,
    m: &SbasPrnMask,
) -> core::result::Result<(), SbasEncodeError> {
    reserved_layout(1, "mask", &m.reserved, &[])?;
    for bit in m.mask {
        w.flag(bit);
    }
    w.u("iodp", u64::from(m.iodp), 2)
}

fn decode_fast(preamble: u8, message_type: u8, data: &[u8]) -> Result<SbasFastCorrections> {
    let mut r = BitReader::new(data);
    let iodf = r.u(2)? as u8;
    let iodp = r.u(2)? as u8;
    let mut prc = [0i16; 13];
    for value in &mut prc {
        *value = r.i(12)? as i16;
    }
    let mut udrei = [0u8; 13];
    for value in &mut udrei {
        *value = r.u(4)? as u8;
    }
    Ok(SbasFastCorrections {
        preamble,
        message_type,
        iodf,
        iodp,
        prc,
        udrei,
        reserved: SpareBits::new(),
    })
}

fn encode_fast(
    w: &mut FieldWriter,
    m: &SbasFastCorrections,
) -> core::result::Result<(), SbasEncodeError> {
    reserved_layout(m.message_type, "fast corrections", &m.reserved, &[])?;
    w.u("iodf", u64::from(m.iodf), 2)?;
    w.u("iodp", u64::from(m.iodp), 2)?;
    for (index, value) in m.prc.iter().enumerate() {
        w.i_at("prc", Some(index), i64::from(*value), 12)?;
    }
    for (index, value) in m.udrei.iter().enumerate() {
        w.u_at("udrei", Some(index), u64::from(*value), 4)?;
    }
    Ok(())
}

fn decode_integrity(preamble: u8, data: &[u8]) -> Result<SbasIntegrity> {
    let mut r = BitReader::new(data);
    let mut iodf = [0u8; 4];
    for value in &mut iodf {
        *value = r.u(2)? as u8;
    }
    let mut udrei = [0u8; 51];
    for value in &mut udrei {
        *value = r.u(4)? as u8;
    }
    Ok(SbasIntegrity {
        preamble,
        iodf,
        udrei,
        reserved: SpareBits::new(),
    })
}

fn encode_integrity(
    w: &mut FieldWriter,
    m: &SbasIntegrity,
) -> core::result::Result<(), SbasEncodeError> {
    reserved_layout(6, "integrity", &m.reserved, &[])?;
    for (index, value) in m.iodf.iter().enumerate() {
        w.u_at("iodf", Some(index), u64::from(*value), 2)?;
    }
    for (index, value) in m.udrei.iter().enumerate() {
        w.u_at("udrei", Some(index), u64::from(*value), 4)?;
    }
    Ok(())
}

fn decode_fast_degradation(preamble: u8, data: &[u8]) -> Result<SbasFastDegradation> {
    let mut r = BitReader::new(data);
    let system_latency_s = r.u(4)? as u8;
    let iodp = r.u(2)? as u8;
    let mut reserved = SpareBits::new();
    reserved.push(r.u(2)?, 2);
    let mut ai = [0u8; 51];
    for value in &mut ai {
        *value = r.u(4)? as u8;
    }
    Ok(SbasFastDegradation {
        preamble,
        system_latency_s,
        iodp,
        ai,
        reserved,
    })
}

fn encode_fast_degradation(
    w: &mut FieldWriter,
    m: &SbasFastDegradation,
) -> core::result::Result<(), SbasEncodeError> {
    let spare = reserved_layout(7, "degradation", &m.reserved, &[2])?;
    w.u("system_latency_s", u64::from(m.system_latency_s), 4)?;
    w.u("iodp", u64::from(m.iodp), 2)?;
    w.spare(spare[0])?;
    for (index, value) in m.ai.iter().enumerate() {
        w.u_at("ai", Some(index), u64::from(*value), 4)?;
    }
    Ok(())
}

fn decode_geo_nav(preamble: u8, data: &[u8]) -> Result<SbasGeoNav> {
    let mut r = BitReader::new(data);
    let mut reserved = SpareBits::new();
    reserved.push(r.u(8)?, 8);
    let time_of_day_s = r.u(13)? as u16;
    let ura = r.u(4)? as u8;
    let x_m = r.i(30)? as i32;
    let y_m = r.i(30)? as i32;
    let z_m = r.i(25)? as i32;
    let x_rate_m_s = r.i(17)? as i32;
    let y_rate_m_s = r.i(17)? as i32;
    let z_rate_m_s = r.i(18)? as i32;
    let x_accel_m_s2 = r.i(10)? as i16;
    let y_accel_m_s2 = r.i(10)? as i16;
    let z_accel_m_s2 = r.i(10)? as i16;
    let a_gf0_s = r.i(12)? as i16;
    let a_gf1_s_s = r.i(8)? as i16;
    Ok(SbasGeoNav {
        preamble,
        time_of_day_s,
        ura,
        x_m,
        y_m,
        z_m,
        x_rate_m_s,
        y_rate_m_s,
        z_rate_m_s,
        x_accel_m_s2,
        y_accel_m_s2,
        z_accel_m_s2,
        a_gf0_s,
        a_gf1_s_s,
        reserved,
    })
}

fn encode_geo_nav(
    w: &mut FieldWriter,
    m: &SbasGeoNav,
) -> core::result::Result<(), SbasEncodeError> {
    let spare = reserved_layout(9, "navigation", &m.reserved, &[8])?;
    w.spare(spare[0])?;
    w.u("time_of_day_s", u64::from(m.time_of_day_s), 13)?;
    w.u("ura", u64::from(m.ura), 4)?;
    w.i("x_m", i64::from(m.x_m), 30)?;
    w.i("y_m", i64::from(m.y_m), 30)?;
    w.i("z_m", i64::from(m.z_m), 25)?;
    w.i("x_rate_m_s", i64::from(m.x_rate_m_s), 17)?;
    w.i("y_rate_m_s", i64::from(m.y_rate_m_s), 17)?;
    w.i("z_rate_m_s", i64::from(m.z_rate_m_s), 18)?;
    w.i("x_accel_m_s2", i64::from(m.x_accel_m_s2), 10)?;
    w.i("y_accel_m_s2", i64::from(m.y_accel_m_s2), 10)?;
    w.i("z_accel_m_s2", i64::from(m.z_accel_m_s2), 10)?;
    w.i("a_gf0_s", i64::from(m.a_gf0_s), 12)?;
    w.i("a_gf1_s_s", i64::from(m.a_gf1_s_s), 8)
}

fn decode_igp_mask(preamble: u8, data: &[u8]) -> Result<SbasIgpMask> {
    let mut r = BitReader::new(data);
    let mut reserved = SpareBits::new();
    reserved.push(r.u(4)?, 4);
    let band_number = r.u(4)? as u8;
    let iodi = r.u(2)? as u8;
    let mut mask = [false; 201];
    for bit in &mut mask {
        *bit = r.flag()?;
    }
    reserved.push(r.u(1)?, 1);
    Ok(SbasIgpMask {
        preamble,
        band_number,
        iodi,
        mask,
        reserved,
    })
}

fn encode_igp_mask(
    w: &mut FieldWriter,
    m: &SbasIgpMask,
) -> core::result::Result<(), SbasEncodeError> {
    let spare = reserved_layout(18, "IGP mask", &m.reserved, &[4, 1])?;
    w.spare(spare[0])?;
    w.u("band_number", u64::from(m.band_number), 4)?;
    w.u("iodi", u64::from(m.iodi), 2)?;
    for bit in m.mask {
        w.flag(bit);
    }
    w.spare(spare[1])
}

fn decode_mixed(preamble: u8, data: &[u8]) -> Result<SbasMixedCorrections> {
    let mut r = BitReader::new(data);
    let mut prc = [0i16; 6];
    for value in &mut prc {
        *value = r.i(12)? as i16;
    }
    let mut udrei = [0u8; 6];
    for value in &mut udrei {
        *value = r.u(4)? as u8;
    }
    let iodp = r.u(2)? as u8;
    let block_id = r.u(2)? as u8;
    let iodf = r.u(2)? as u8;
    let mut reserved = SpareBits::new();
    reserved.push(r.u(4)?, 4);
    let long_term = decode_long_half(&mut r)?;
    Ok(SbasMixedCorrections {
        preamble,
        fast: SbasMixedFastCorrections {
            iodf,
            iodp,
            block_id,
            prc,
            udrei,
            reserved,
        },
        long_term,
    })
}

fn encode_mixed(
    w: &mut FieldWriter,
    m: &SbasMixedCorrections,
) -> core::result::Result<(), SbasEncodeError> {
    let spare = reserved_layout(24, "fast corrections", &m.fast.reserved, &[4])?;
    for (index, value) in m.fast.prc.iter().enumerate() {
        w.i_at("prc", Some(index), i64::from(*value), 12)?;
    }
    for (index, value) in m.fast.udrei.iter().enumerate() {
        w.u_at("udrei", Some(index), u64::from(*value), 4)?;
    }
    w.u("iodp", u64::from(m.fast.iodp), 2)?;
    w.u("block_id", u64::from(m.fast.block_id), 2)?;
    w.u("iodf", u64::from(m.fast.iodf), 2)?;
    w.spare(spare[0])?;
    encode_long_half(w, &m.long_term, 0)
}

fn decode_long_term(preamble: u8, data: &[u8]) -> Result<SbasLongTermCorrections> {
    let mut r = BitReader::new(data);
    let first = decode_long_half(&mut r)?;
    let second = decode_long_half(&mut r)?;
    Ok(SbasLongTermCorrections {
        preamble,
        halves: [first, second],
    })
}

fn encode_long_term(
    w: &mut FieldWriter,
    m: &SbasLongTermCorrections,
) -> core::result::Result<(), SbasEncodeError> {
    for (index, half) in m.halves.iter().enumerate() {
        encode_long_half(w, half, index)?;
    }
    Ok(())
}

/// Read one 106-bit long-term half at the reader's position.
///
/// Offsets are RTKLIB `decode_longcorrh`'s relative to the half start `p`:
/// the velocity code at `p`, the records from `p + 1`, and the IODP at
/// `p + 103` without the velocity code, followed by one reserved bit, or at
/// `p + 104` with it.
fn decode_long_half(r: &mut BitReader<'_>) -> Result<SbasLongTermHalf> {
    let velocity_code = r.flag()?;
    let mut records = Vec::new();
    let mut reserved = SpareBits::new();
    if velocity_code {
        records.push(SbasLongTermRecord {
            monitored_index: r.u(6)? as u8,
            iode: r.u(8)? as u8,
            delta_x: r.i(11)? as i32,
            delta_y: r.i(11)? as i32,
            delta_z: r.i(11)? as i32,
            delta_a_f0: r.i(11)? as i32,
            delta_x_rate: r.i(8)? as i32,
            delta_y_rate: r.i(8)? as i32,
            delta_z_rate: r.i(8)? as i32,
            delta_a_f1: r.i(8)? as i32,
            time_of_day_s: Some(r.u(13)? as u32),
        });
        let iodp = r.u(2)? as u8;
        Ok(SbasLongTermHalf {
            velocity_code,
            iodp,
            records,
            reserved,
        })
    } else {
        for _ in 0..2 {
            records.push(SbasLongTermRecord {
                monitored_index: r.u(6)? as u8,
                iode: r.u(8)? as u8,
                delta_x: r.i(9)? as i32,
                delta_y: r.i(9)? as i32,
                delta_z: r.i(9)? as i32,
                delta_x_rate: 0,
                delta_y_rate: 0,
                delta_z_rate: 0,
                delta_a_f0: r.i(10)? as i32,
                delta_a_f1: 0,
                time_of_day_s: None,
            });
        }
        let iodp = r.u(2)? as u8;
        reserved.push(r.u(1)?, 1);
        Ok(SbasLongTermHalf {
            velocity_code,
            iodp,
            records,
            reserved,
        })
    }
}

fn encode_long_half(
    w: &mut FieldWriter,
    m: &SbasLongTermHalf,
    half: usize,
) -> core::result::Result<(), SbasEncodeError> {
    let message_type = w.message_type;
    let start = w.nbits;
    let expected = if m.velocity_code { 1 } else { 2 };
    if m.records.len() != expected {
        return Err(SbasEncodeError::LongTermRecordCount {
            message_type,
            half,
            velocity_code: m.velocity_code,
            expected,
            found: m.records.len(),
        });
    }
    w.flag(m.velocity_code);
    if m.velocity_code {
        reserved_layout(message_type, "long-term half", &m.reserved, &[])?;
        let record = &m.records[0];
        let Some(time_of_day) = record.time_of_day_s else {
            return Err(SbasEncodeError::LongTermMissingTimeOfDay { message_type, half });
        };
        w.u("monitored_index", u64::from(record.monitored_index), 6)?;
        w.u("iode", u64::from(record.iode), 8)?;
        w.i("delta_x", i64::from(record.delta_x), 11)?;
        w.i("delta_y", i64::from(record.delta_y), 11)?;
        w.i("delta_z", i64::from(record.delta_z), 11)?;
        w.i("delta_a_f0", i64::from(record.delta_a_f0), 11)?;
        w.i("delta_x_rate", i64::from(record.delta_x_rate), 8)?;
        w.i("delta_y_rate", i64::from(record.delta_y_rate), 8)?;
        w.i("delta_z_rate", i64::from(record.delta_z_rate), 8)?;
        w.i("delta_a_f1", i64::from(record.delta_a_f1), 8)?;
        w.u("time_of_day_s", u64::from(time_of_day), 13)?;
        w.u("iodp", u64::from(m.iodp), 2)?;
    } else {
        let spare = reserved_layout(message_type, "long-term half", &m.reserved, &[1])?;
        for (index, record) in m.records.iter().enumerate() {
            let not_carried = [
                ("delta_x_rate", record.delta_x_rate != 0),
                ("delta_y_rate", record.delta_y_rate != 0),
                ("delta_z_rate", record.delta_z_rate != 0),
                ("delta_a_f1", record.delta_a_f1 != 0),
                ("time_of_day_s", record.time_of_day_s.is_some()),
            ];
            if let Some(&(field, _)) = not_carried.iter().find(|(_, held)| *held) {
                return Err(SbasEncodeError::LongTermFieldNotCarried {
                    message_type,
                    half,
                    record: index,
                    field,
                });
            }
            w.u("monitored_index", u64::from(record.monitored_index), 6)?;
            w.u("iode", u64::from(record.iode), 8)?;
            w.i("delta_x", i64::from(record.delta_x), 9)?;
            w.i("delta_y", i64::from(record.delta_y), 9)?;
            w.i("delta_z", i64::from(record.delta_z), 9)?;
            w.i("delta_a_f0", i64::from(record.delta_a_f0), 10)?;
        }
        w.u("iodp", u64::from(m.iodp), 2)?;
        w.spare(spare[0])?;
    }
    debug_assert_eq!(w.nbits - start, LONG_HALF_BITS);
    Ok(())
}

fn decode_iono_delays(preamble: u8, data: &[u8]) -> Result<SbasIonoDelays> {
    let mut r = BitReader::new(data);
    let band_number = r.u(4)? as u8;
    let block_id = r.u(4)? as u8;
    let mut entries: [SbasIgpDelay; 15] = core::array::from_fn(|_| SbasIgpDelay::default());
    for entry in &mut entries {
        entry.vertical_delay = r.u(9)? as u16;
        entry.givei = r.u(4)? as u8;
    }
    let iodi = r.u(2)? as u8;
    let mut reserved = SpareBits::new();
    reserved.push(r.u(7)?, 7);
    Ok(SbasIonoDelays {
        preamble,
        band_number,
        block_id,
        iodi,
        entries,
        reserved,
    })
}

fn encode_iono_delays(
    w: &mut FieldWriter,
    m: &SbasIonoDelays,
) -> core::result::Result<(), SbasEncodeError> {
    let spare = reserved_layout(26, "ionospheric delays", &m.reserved, &[7])?;
    w.u("band_number", u64::from(m.band_number), 4)?;
    w.u("block_id", u64::from(m.block_id), 4)?;
    for (index, entry) in m.entries.iter().enumerate() {
        w.u_at(
            "vertical_delay",
            Some(index),
            u64::from(entry.vertical_delay),
            9,
        )?;
        w.u_at("givei", Some(index), u64::from(entry.givei), 4)?;
    }
    w.u("iodi", u64::from(m.iodi), 2)?;
    w.spare(spare[0])
}

/// Read `nbits` bits into bytes, most significant bit first, with the final
/// byte's unused low bits zero.
fn read_bits_as_bytes(reader: &mut BitReader<'_>, nbits: usize) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();
    for _ in 0..nbits {
        w.push_u(reader.u(1)?, 1);
    }
    Ok(w.into_bytes())
}

fn bit_at(bytes: &[u8], bit_pos: usize) -> u8 {
    (bytes[bit_pos / 8] >> (7 - (bit_pos % 8))) & 1
}

fn bits_as_u32(bytes: &[u8], bit_pos: usize, nbits: usize) -> u32 {
    let mut value = 0u32;
    for pos in bit_pos..bit_pos + nbits {
        value = (value << 1) | u32::from(bit_at(bytes, pos));
    }
    value
}

fn parse_error(message: &str) -> Error {
    Error::Parse(message.to_string())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::rtcm::crc::crc24q;

    fn block(message: SbasMessage, form: SbasWireForm) -> Vec<u8> {
        SbasBlock {
            form,
            message,
            pad_bits: 0,
        }
        .encode()
        .expect("encodable SBAS block")
    }

    fn encode_err(message: SbasMessage) -> SbasEncodeError {
        let err = SbasBlock {
            form: SbasWireForm::Body226,
            message,
            pad_bits: 0,
        }
        .encode()
        .expect_err("the block must be refused");
        match err {
            Error::SbasEncode(err) => *err,
            other => panic!("expected Error::SbasEncode, got {other:?}"),
        }
    }

    fn bits_i(bytes: &[u8], bit_pos: usize, nbits: usize) -> i32 {
        let raw = bits_as_u32(bytes, bit_pos, nbits);
        let sign_bit = 1u32 << (nbits - 1);
        if raw & sign_bit != 0 {
            raw as i32 - (1i32 << nbits)
        } else {
            raw as i32
        }
    }

    fn hex_bytes(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|idx| u8::from_str_radix(&hex[idx..idx + 2], 16).expect("hex byte"))
            .collect()
    }

    fn fast_message() -> SbasFastCorrections {
        SbasFastCorrections {
            preamble: 0x53,
            message_type: 2,
            iodf: 1,
            iodp: 2,
            prc: [0; 13],
            udrei: [0; 13],
            reserved: SpareBits::new(),
        }
    }

    fn non_velocity_record(monitored_index: u8) -> SbasLongTermRecord {
        SbasLongTermRecord {
            monitored_index,
            iode: 9,
            delta_x: -10,
            delta_y: 11,
            delta_z: -12,
            delta_x_rate: 0,
            delta_y_rate: 0,
            delta_z_rate: 0,
            delta_a_f0: 13,
            delta_a_f1: 0,
            time_of_day_s: None,
        }
    }

    fn non_velocity_half(records: Vec<SbasLongTermRecord>) -> SbasLongTermHalf {
        SbasLongTermHalf {
            velocity_code: false,
            iodp: 2,
            records,
            reserved: SpareBits(vec![(0, 1)]),
        }
    }

    fn velocity_half(record: SbasLongTermRecord) -> SbasLongTermHalf {
        SbasLongTermHalf {
            velocity_code: true,
            iodp: 3,
            records: vec![record],
            reserved: SpareBits::new(),
        }
    }

    /// A velocity-code record holding the extreme value of every field.
    fn extreme_velocity_record() -> SbasLongTermRecord {
        SbasLongTermRecord {
            monitored_index: 63,
            iode: 255,
            delta_x: -1024,
            delta_y: 1023,
            delta_z: -1,
            delta_x_rate: -128,
            delta_y_rate: 127,
            delta_z_rate: -1,
            delta_a_f0: -1024,
            delta_a_f1: 127,
            time_of_day_s: Some(8191),
        }
    }

    #[test]
    fn crc_bits_matches_byte_crc_when_aligned() {
        let data = [0x53, 0x01, 0x23, 0x45, 0x67, 0x89];
        assert_eq!(crc24q_bits(&data, data.len() * 8), crc24q(&data));
    }

    #[test]
    fn prn_mask_decodes_and_round_trips_in_both_forms() {
        let mut mask = [false; 210];
        mask[0] = true;
        mask[119] = true;
        mask[157] = true;
        let msg = SbasMessage::PrnMask(SbasPrnMask {
            preamble: 0x53,
            iodp: 2,
            mask,
            reserved: SpareBits::new(),
        });
        for form in [SbasWireForm::Body226, SbasWireForm::Framed250] {
            let encoded = block(msg.clone(), form);
            let decoded = SbasBlock::decode(&encoded, form).expect("valid SBAS block");
            assert_eq!(decoded.message, msg);
            assert_eq!(decoded.encode().expect("re-encode"), encoded);
            assert_eq!(decoded.message.message_type(), 1);
        }
    }

    #[test]
    fn corrupted_crc_is_rejected() {
        let msg = SbasMessage::Unsupported(SbasUnsupported {
            preamble: 0x9A,
            message_type: 62,
            data: vec![0; RAW_PAYLOAD_LEN],
        });
        let mut encoded = block(msg, SbasWireForm::Framed250);
        encoded[31] ^= 0x40;
        assert!(SbasBlock::decode(&encoded, SbasWireForm::Framed250).is_err());
        assert!(SbasBlock::decode_with_policy(
            &encoded,
            SbasWireForm::Framed250,
            SbasPolicy::Lenient
        )
        .is_err());
    }

    #[test]
    fn phase_a_message_classification() {
        for mt in [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 17, 18, 24, 25, 26] {
            assert!(is_phase_a_message(mt));
        }
        assert!(!is_phase_a_message(10));
        assert!(!is_phase_a_message(63));
    }

    #[test]
    fn mt24_encode_uses_rtklib_offsets() {
        let mut second = non_velocity_record(6);
        second.iode = 10;
        second.delta_x = 14;
        second.delta_y = -15;
        second.delta_z = 16;
        second.delta_a_f0 = -17;
        let msg = SbasMessage::MixedCorrections(SbasMixedCorrections {
            preamble: 0x53,
            fast: SbasMixedFastCorrections {
                iodf: 1,
                iodp: 2,
                block_id: 3,
                prc: [-1, 2, -3, 4, -5, 6],
                udrei: [1, 2, 3, 4, 5, 6],
                reserved: SpareBits(vec![(0b1010, 4)]),
            },
            long_term: SbasLongTermHalf {
                reserved: SpareBits(vec![(1, 1)]),
                ..non_velocity_half(vec![non_velocity_record(5), second])
            },
        });
        let body = block(msg.clone(), SbasWireForm::Body226);

        assert_eq!(bits_as_u32(&body, 8, 6), 24);
        assert_eq!(bits_i(&body, 14, 12), -1);
        assert_eq!(bits_i(&body, 26, 12), 2);
        assert_eq!(bits_i(&body, 38, 12), -3);
        assert_eq!(bits_i(&body, 50, 12), 4);
        assert_eq!(bits_i(&body, 62, 12), -5);
        assert_eq!(bits_i(&body, 74, 12), 6);
        assert_eq!(bits_as_u32(&body, 86, 4), 1);
        assert_eq!(bits_as_u32(&body, 90, 4), 2);
        assert_eq!(bits_as_u32(&body, 94, 4), 3);
        assert_eq!(bits_as_u32(&body, 98, 4), 4);
        assert_eq!(bits_as_u32(&body, 102, 4), 5);
        assert_eq!(bits_as_u32(&body, 106, 4), 6);
        assert_eq!(bits_as_u32(&body, 110, 2), 2);
        assert_eq!(bits_as_u32(&body, 112, 2), 3);
        assert_eq!(bits_as_u32(&body, 114, 2), 1);
        assert_eq!(bits_as_u32(&body, 116, 4), 0b1010);
        assert_eq!(bits_as_u32(&body, 120, 1), 0);
        assert_eq!(bits_as_u32(&body, 121, 6), 5);
        assert_eq!(bits_as_u32(&body, 172, 6), 6);
        assert_eq!(bits_as_u32(&body, 223, 2), 2);
        assert_eq!(bits_as_u32(&body, 225, 1), 1);

        let decoded = SbasBlock::decode(&body, SbasWireForm::Body226).expect("decode MT24");
        assert_eq!(decoded.message, msg);
    }

    /// RTKLIB `decode_longcorrh` at `p` = 14 (first half) and 120 (second),
    /// `decode_longcorr1` from `p + 1` and `decode_longcorr0` from `p + 1` and
    /// `p + 52`: every field lands at RTKLIB's offset with its sign, and the
    /// velocity half survives the round trip.
    #[test]
    fn mt25_velocity_and_non_velocity_halves_use_rtklib_offsets() {
        let mut second = non_velocity_record(2);
        second.delta_x = -256;
        second.delta_y = 255;
        second.delta_z = -1;
        second.delta_a_f0 = -512;
        let msg = SbasMessage::LongTermCorrections(SbasLongTermCorrections {
            preamble: 0xC6,
            halves: [
                velocity_half(extreme_velocity_record()),
                SbasLongTermHalf {
                    reserved: SpareBits(vec![(1, 1)]),
                    ..non_velocity_half(vec![non_velocity_record(1), second])
                },
            ],
        });
        for form in [SbasWireForm::Body226, SbasWireForm::Framed250] {
            let body = block(msg.clone(), form);
            // First half, velocity code 1, p = 14.
            assert_eq!(bits_as_u32(&body, 14, 1), 1);
            assert_eq!(bits_as_u32(&body, 15, 6), 63);
            assert_eq!(bits_as_u32(&body, 21, 8), 255);
            assert_eq!(bits_i(&body, 29, 11), -1024);
            assert_eq!(bits_i(&body, 40, 11), 1023);
            assert_eq!(bits_i(&body, 51, 11), -1);
            assert_eq!(bits_i(&body, 62, 11), -1024);
            assert_eq!(bits_i(&body, 73, 8), -128);
            assert_eq!(bits_i(&body, 81, 8), 127);
            assert_eq!(bits_i(&body, 89, 8), -1);
            assert_eq!(bits_i(&body, 97, 8), 127);
            assert_eq!(bits_as_u32(&body, 105, 13), 8191);
            assert_eq!(bits_as_u32(&body, 118, 2), 3);
            // Second half, velocity code 0, p = 120.
            assert_eq!(bits_as_u32(&body, 120, 1), 0);
            assert_eq!(bits_as_u32(&body, 121, 6), 1);
            assert_eq!(bits_as_u32(&body, 127, 8), 9);
            assert_eq!(bits_i(&body, 135, 9), -10);
            assert_eq!(bits_i(&body, 144, 9), 11);
            assert_eq!(bits_i(&body, 153, 9), -12);
            assert_eq!(bits_i(&body, 162, 10), 13);
            assert_eq!(bits_as_u32(&body, 172, 6), 2);
            assert_eq!(bits_i(&body, 186, 9), -256);
            assert_eq!(bits_i(&body, 195, 9), 255);
            assert_eq!(bits_i(&body, 204, 9), -1);
            assert_eq!(bits_i(&body, 213, 10), -512);
            assert_eq!(bits_as_u32(&body, 223, 2), 2);
            assert_eq!(bits_as_u32(&body, 225, 1), 1);

            let decoded = SbasBlock::decode(&body, form).expect("decode MT25");
            assert_eq!(decoded.message, msg);
            assert_eq!(decoded.encode().expect("re-encode MT25"), body);
        }
    }

    #[test]
    fn mt24_velocity_half_round_trips() {
        let msg = SbasMessage::MixedCorrections(SbasMixedCorrections {
            preamble: 0x9A,
            fast: SbasMixedFastCorrections {
                iodf: 3,
                iodp: 3,
                block_id: 0,
                prc: [-2048, 2047, 0, -1, 1, 0],
                udrei: [14, 15, 0, 13, 1, 2],
                reserved: SpareBits(vec![(0, 4)]),
            },
            long_term: velocity_half(extreme_velocity_record()),
        });
        let body = block(msg.clone(), SbasWireForm::Body226);
        // RTKLIB decode_sbstype24 hands the long-term half to
        // decode_longcorrh at p = 120.
        assert_eq!(bits_as_u32(&body, 120, 1), 1);
        assert_eq!(bits_as_u32(&body, 121, 6), 63);
        assert_eq!(bits_as_u32(&body, 211, 13), 8191);
        assert_eq!(bits_as_u32(&body, 224, 2), 3);
        let decoded = SbasBlock::decode(&body, SbasWireForm::Body226).expect("decode MT24");
        assert_eq!(decoded.message, msg);
    }

    /// RTKLIB `decode_sbstype7`: latency at bit 14, IODP at 18, the first
    /// indicator at 22. The two bits at 20 are reserved.
    #[test]
    fn mt7_reads_indicators_after_the_reserved_bits() {
        let mut ai = [0u8; 51];
        for (index, value) in ai.iter_mut().enumerate() {
            *value = (index % 16) as u8;
        }
        let msg = SbasMessage::FastDegradation(SbasFastDegradation {
            preamble: 0x53,
            system_latency_s: 11,
            iodp: 2,
            ai,
            reserved: SpareBits(vec![(0b01, 2)]),
        });
        let body = block(msg.clone(), SbasWireForm::Body226);
        assert_eq!(bits_as_u32(&body, 14, 4), 11);
        assert_eq!(bits_as_u32(&body, 18, 2), 2);
        assert_eq!(bits_as_u32(&body, 20, 2), 0b01);
        for index in 0..51 {
            assert_eq!(
                bits_as_u32(&body, 22 + 4 * index, 4),
                (index % 16) as u32,
                "ai[{index}]"
            );
        }
        let decoded = SbasBlock::decode(&body, SbasWireForm::Body226).expect("decode MT7");
        assert_eq!(decoded.message, msg);
    }

    /// The second record of the gLAB EMS format example
    /// (`tests/fixtures/sbas_ems/glab_ems_format_example.ems`) is a CRC-valid
    /// framed MT7 block whose preamble is 0xA9. The expected fields were read
    /// at RTKLIB `decode_sbstype7` offsets by a script independent of this
    /// decoder.
    ///
    /// DO-229 lays message type 7 out as system latency (4 bits), IODP (2),
    /// spare (2) and 51 four-bit indicators, 212 bits in all, so the
    /// indicators start at message bit 22. Read from bit 20 instead, this
    /// record gives a tidy run of 7s and 14s and zero trailing bits, while
    /// the standard reading gives spare bits `11`. That pattern is not
    /// evidence for a bit-20 layout: the fixture is an authored example whose
    /// preamble is none of the SBAS values and whose message-type column
    /// contradicts two of its three messages, so its contents say nothing
    /// about where a receiver finds the indicators.
    #[test]
    fn glab_mt7_record_decodes_leniently_and_restates_exactly() {
        let bytes = hex_bytes("A91EEE7E7EE7777EEEE777E777777EEEE77EEE700000000000000000234C75C0");
        let err = SbasBlock::decode(&bytes, SbasWireForm::Framed250).unwrap_err();
        assert!(
            matches!(err, Error::Parse(ref msg) if msg.contains("0xA9")),
            "strict decoding names the preamble, got {err:?}"
        );

        let (decoded, departures) =
            SbasBlock::decode_with_policy(&bytes, SbasWireForm::Framed250, SbasPolicy::Lenient)
                .expect("lenient decode");
        assert_eq!(
            departures,
            vec![SbasDeparture::UnrecognizedPreamble { preamble: 0xA9 }]
        );
        let SbasMessage::FastDegradation(ref degradation) = decoded.message else {
            panic!("expected MT7, got {:?}", decoded.message);
        };
        assert_eq!(degradation.preamble, 0xA9);
        assert_eq!(degradation.system_latency_s, 11);
        assert_eq!(degradation.iodp, 2);
        assert_eq!(degradation.reserved, SpareBits(vec![(3, 2)]));
        let expected_ai: [u8; 51] = [
            9, 15, 9, 15, 11, 9, 13, 13, 13, 15, 11, 11, 11, 9, 13, 13, 15, 9, 13, 13, 13, 13, 13,
            15, 11, 11, 11, 9, 13, 15, 11, 11, 9, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0,
        ];
        assert_eq!(degradation.ai, expected_ai);

        assert_eq!(
            decoded.encode(),
            Err(Error::SbasEncode(Box::new(
                SbasEncodeError::UnrecognizedPreamble { preamble: 0xA9 }
            )))
        );
        let (encoded, departures) = decoded
            .encode_with_policy(SbasPolicy::Lenient)
            .expect("lenient encode");
        assert_eq!(encoded, bytes);
        assert_eq!(
            departures,
            vec![SbasDeparture::UnrecognizedPreamble { preamble: 0xA9 }]
        );
    }

    #[test]
    fn pad_bits_are_kept_and_restated_in_both_forms() {
        let msg = SbasMessage::FastCorrections(fast_message());
        for form in [SbasWireForm::Body226, SbasWireForm::Framed250] {
            let mut bytes = block(msg.clone(), form);
            let last = bytes.len() - 1;
            bytes[last] |= 0b10_1101;
            let decoded = SbasBlock::decode(&bytes, form).expect("pad bits are not refused");
            assert_eq!(decoded.pad_bits, 0b10_1101);
            assert_eq!(decoded.message, msg);
            assert_eq!(decoded.encode().expect("re-encode"), bytes);
        }

        let err = SbasBlock {
            form: SbasWireForm::Body226,
            message: msg,
            pad_bits: 0x40,
        }
        .encode()
        .unwrap_err();
        assert_eq!(
            err,
            Error::SbasEncode(Box::new(SbasEncodeError::PadBits { value: 0x40 }))
        );
    }

    #[test]
    fn not_available_encodings_round_trip_as_read() {
        let mut fast = fast_message();
        fast.udrei = [14, 15, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 13];
        fast.prc = [-2048, 2047, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut entries: [SbasIgpDelay; 15] = core::array::from_fn(|_| SbasIgpDelay::default());
        entries[0] = SbasIgpDelay {
            vertical_delay: 511,
            givei: 15,
        };
        entries[1] = SbasIgpDelay {
            vertical_delay: 511,
            givei: 0,
        };
        entries[2] = SbasIgpDelay {
            vertical_delay: 0,
            givei: 15,
        };
        let iono = SbasIonoDelays {
            preamble: 0x9A,
            band_number: 10,
            block_id: 13,
            iodi: 3,
            entries,
            reserved: SpareBits(vec![(0x7F, 7)]),
        };
        for msg in [
            SbasMessage::FastCorrections(fast),
            SbasMessage::IonoDelays(iono),
        ] {
            for form in [SbasWireForm::Body226, SbasWireForm::Framed250] {
                let bytes = block(msg.clone(), form);
                let decoded = SbasBlock::decode(&bytes, form).expect("decode");
                assert_eq!(decoded.message, msg);
                assert_eq!(decoded.encode().expect("re-encode"), bytes);
            }
        }
    }

    #[test]
    fn raw_payload_messages_round_trip_all_212_bits() {
        let mut data = vec![0xA5; RAW_PAYLOAD_LEN];
        data[RAW_PAYLOAD_LEN - 1] = 0xF0;
        for msg in [
            SbasMessage::DoNotUse(SbasDoNotUse {
                preamble: 0x53,
                data: data.clone(),
            }),
            SbasMessage::NetworkTime(SbasNetworkTime {
                preamble: 0x9A,
                data: data.clone(),
            }),
            SbasMessage::GeoAlmanac(SbasGeoAlmanac {
                preamble: 0xC6,
                data: data.clone(),
            }),
            SbasMessage::Unsupported(SbasUnsupported {
                preamble: 0x53,
                message_type: 63,
                data: data.clone(),
            }),
        ] {
            let bytes = block(msg.clone(), SbasWireForm::Framed250);
            let decoded = SbasBlock::decode(&bytes, SbasWireForm::Framed250).expect("decode");
            assert_eq!(decoded.message, msg);
        }
    }

    #[test]
    fn encode_refuses_a_raw_payload_that_is_not_212_bits() {
        for (data, bits_past_payload) in [
            (Vec::new(), false),
            (vec![0; 26], false),
            (vec![0; 29], false),
            (
                {
                    let mut data = vec![0; RAW_PAYLOAD_LEN];
                    data[RAW_PAYLOAD_LEN - 1] = 0x01;
                    data
                },
                true,
            ),
        ] {
            let bytes = data.len();
            let err = encode_err(SbasMessage::DoNotUse(SbasDoNotUse {
                preamble: 0x53,
                data,
            }));
            assert_eq!(
                err,
                SbasEncodeError::RawPayload {
                    message_type: 0,
                    bytes,
                    bits_past_payload,
                }
            );
        }
    }

    #[test]
    fn encode_refuses_a_value_wider_than_its_field() {
        let mut fast = fast_message();
        fast.prc[4] = 2048;
        assert_eq!(
            encode_err(SbasMessage::FastCorrections(fast)),
            SbasEncodeError::FieldOutOfRange {
                message_type: 2,
                field: "prc",
                index: Some(4),
                value: 2048,
                width: 12,
                signed: true,
            }
        );

        let mut fast = fast_message();
        fast.prc[0] = -2049;
        assert!(matches!(
            encode_err(SbasMessage::FastCorrections(fast)),
            SbasEncodeError::FieldOutOfRange {
                field: "prc",
                index: Some(0),
                value: -2049,
                ..
            }
        ));

        let mut fast = fast_message();
        fast.udrei[12] = 16;
        assert!(matches!(
            encode_err(SbasMessage::FastCorrections(fast)),
            SbasEncodeError::FieldOutOfRange {
                field: "udrei",
                index: Some(12),
                width: 4,
                signed: false,
                ..
            }
        ));

        let mut fast = fast_message();
        fast.iodp = 4;
        assert!(matches!(
            encode_err(SbasMessage::FastCorrections(fast)),
            SbasEncodeError::FieldOutOfRange {
                field: "iodp",
                value: 4,
                width: 2,
                ..
            }
        ));

        let mut record = extreme_velocity_record();
        record.time_of_day_s = Some(8192);
        let msg = SbasMessage::LongTermCorrections(SbasLongTermCorrections {
            preamble: 0x53,
            halves: [
                velocity_half(record),
                velocity_half(extreme_velocity_record()),
            ],
        });
        assert!(matches!(
            encode_err(msg),
            SbasEncodeError::FieldOutOfRange {
                message_type: 25,
                field: "time_of_day_s",
                value: 8192,
                width: 13,
                ..
            }
        ));

        let mut record = non_velocity_record(1);
        record.delta_x = 256;
        let msg = SbasMessage::LongTermCorrections(SbasLongTermCorrections {
            preamble: 0x53,
            halves: [
                non_velocity_half(vec![record, non_velocity_record(2)]),
                non_velocity_half(vec![non_velocity_record(3), non_velocity_record(4)]),
            ],
        });
        assert!(matches!(
            encode_err(msg),
            SbasEncodeError::FieldOutOfRange {
                field: "delta_x",
                value: 256,
                width: 9,
                ..
            }
        ));

        let mut igp = SbasIonoDelays {
            preamble: 0x53,
            band_number: 16,
            block_id: 0,
            iodi: 0,
            entries: core::array::from_fn(|_| SbasIgpDelay::default()),
            reserved: SpareBits(vec![(0, 7)]),
        };
        assert!(matches!(
            encode_err(SbasMessage::IonoDelays(igp.clone())),
            SbasEncodeError::FieldOutOfRange {
                field: "band_number",
                ..
            }
        ));
        igp.band_number = 0;
        igp.entries[3].vertical_delay = 512;
        assert!(matches!(
            encode_err(SbasMessage::IonoDelays(igp.clone())),
            SbasEncodeError::FieldOutOfRange {
                field: "vertical_delay",
                index: Some(3),
                ..
            }
        ));
        igp.entries[3].vertical_delay = 0;
        igp.reserved = SpareBits(vec![(0x80, 7)]);
        assert!(matches!(
            encode_err(SbasMessage::IonoDelays(igp)),
            SbasEncodeError::FieldOutOfRange {
                field: "reserved",
                value: 0x80,
                width: 7,
                ..
            }
        ));
    }

    #[test]
    fn encode_refuses_long_term_halves_the_layout_cannot_carry() {
        let long = |halves| {
            encode_err(SbasMessage::LongTermCorrections(SbasLongTermCorrections {
                preamble: 0x53,
                halves,
            }))
        };
        let good = || non_velocity_half(vec![non_velocity_record(1), non_velocity_record(2)]);

        // A missing or an extra record is refused rather than filled or dropped.
        for records in [
            vec![non_velocity_record(1)],
            vec![
                non_velocity_record(1),
                non_velocity_record(2),
                non_velocity_record(3),
            ],
        ] {
            let found = records.len();
            assert_eq!(
                long([good(), non_velocity_half(records)]),
                SbasEncodeError::LongTermRecordCount {
                    message_type: 25,
                    half: 1,
                    velocity_code: false,
                    expected: 2,
                    found,
                }
            );
        }
        for records in [
            Vec::new(),
            vec![extreme_velocity_record(), extreme_velocity_record()],
        ] {
            let found = records.len();
            let half = SbasLongTermHalf {
                records,
                ..velocity_half(extreme_velocity_record())
            };
            assert_eq!(
                long([half, good()]),
                SbasEncodeError::LongTermRecordCount {
                    message_type: 25,
                    half: 0,
                    velocity_code: true,
                    expected: 1,
                    found,
                }
            );
        }

        // Velocity terms in a half without the velocity code are refused
        // rather than dropped.
        for field in [
            "delta_x_rate",
            "delta_y_rate",
            "delta_z_rate",
            "delta_a_f1",
            "time_of_day_s",
        ] {
            let mut record = non_velocity_record(2);
            match field {
                "delta_x_rate" => record.delta_x_rate = 1,
                "delta_y_rate" => record.delta_y_rate = -1,
                "delta_z_rate" => record.delta_z_rate = 1,
                "delta_a_f1" => record.delta_a_f1 = 1,
                _ => record.time_of_day_s = Some(0),
            }
            assert_eq!(
                long([
                    good(),
                    non_velocity_half(vec![non_velocity_record(1), record])
                ]),
                SbasEncodeError::LongTermFieldNotCarried {
                    message_type: 25,
                    half: 1,
                    record: 1,
                    field,
                }
            );
        }

        // A velocity record without its time of day is refused rather than
        // written as time of day zero.
        let mut record = extreme_velocity_record();
        record.time_of_day_s = None;
        assert_eq!(
            long([velocity_half(record), good()]),
            SbasEncodeError::LongTermMissingTimeOfDay {
                message_type: 25,
                half: 0,
            }
        );

        // MT24 carries its one half as half 0.
        let err = encode_err(SbasMessage::MixedCorrections(SbasMixedCorrections {
            preamble: 0x53,
            fast: SbasMixedFastCorrections {
                iodf: 0,
                iodp: 0,
                block_id: 0,
                prc: [0; 6],
                udrei: [0; 6],
                reserved: SpareBits(vec![(0, 4)]),
            },
            long_term: non_velocity_half(Vec::new()),
        }));
        assert_eq!(
            err,
            SbasEncodeError::LongTermRecordCount {
                message_type: 24,
                half: 0,
                velocity_code: false,
                expected: 2,
                found: 0,
            }
        );
    }

    /// `validate` is the check `encode` applies, so the two refuse alike.
    #[test]
    fn validate_refuses_what_encode_refuses() {
        let mut record = extreme_velocity_record();
        record.time_of_day_s = None;
        let mut fast = fast_message();
        fast.prc[0] = 2048;
        for message in [
            SbasMessage::LongTermCorrections(SbasLongTermCorrections {
                preamble: 0x53,
                halves: [
                    velocity_half(record),
                    non_velocity_half(vec![non_velocity_record(1)]),
                ],
            }),
            SbasMessage::FastCorrections(fast),
        ] {
            assert_eq!(message.validate().unwrap_err(), encode_err(message.clone()));
        }
        assert_eq!(
            SbasMessage::FastCorrections(fast_message()).validate(),
            Ok(())
        );
    }

    #[test]
    fn encode_refuses_reserved_segments_that_differ_from_the_layout() {
        let geo = SbasGeoNav {
            preamble: 0x9A,
            time_of_day_s: 0,
            ura: 0,
            x_m: 0,
            y_m: 0,
            z_m: 0,
            x_rate_m_s: 0,
            y_rate_m_s: 0,
            z_rate_m_s: 0,
            x_accel_m_s2: 0,
            y_accel_m_s2: 0,
            z_accel_m_s2: 0,
            a_gf0_s: 0,
            a_gf1_s_s: 0,
            reserved: SpareBits::new(),
        };
        assert_eq!(
            encode_err(SbasMessage::GeoNav(geo)),
            SbasEncodeError::ReservedLayout {
                message_type: 9,
                part: "navigation",
                expected: vec![8],
                found: Vec::new(),
            }
        );

        let mut fast = fast_message();
        fast.reserved = SpareBits(vec![(1, 1)]);
        assert!(matches!(
            encode_err(SbasMessage::FastCorrections(fast)),
            SbasEncodeError::ReservedLayout {
                message_type: 2,
                ..
            }
        ));

        let igp = SbasIgpMask {
            preamble: 0x53,
            band_number: 0,
            iodi: 0,
            mask: [false; 201],
            reserved: SpareBits(vec![(0, 4)]),
        };
        assert_eq!(
            encode_err(SbasMessage::IgpMask(igp)),
            SbasEncodeError::ReservedLayout {
                message_type: 18,
                part: "IGP mask",
                expected: vec![4, 1],
                found: vec![4],
            }
        );

        let mut half = velocity_half(extreme_velocity_record());
        half.reserved = SpareBits(vec![(0, 1)]);
        assert!(matches!(
            encode_err(SbasMessage::LongTermCorrections(SbasLongTermCorrections {
                preamble: 0x53,
                halves: [half, velocity_half(extreme_velocity_record())],
            })),
            SbasEncodeError::ReservedLayout {
                message_type: 25,
                part: "long-term half",
                ..
            }
        ));
    }

    #[test]
    fn encode_refuses_a_message_type_its_variant_cannot_carry() {
        let mut fast = fast_message();
        fast.message_type = 6;
        assert!(matches!(
            encode_err(SbasMessage::FastCorrections(fast)),
            SbasEncodeError::MessageType {
                message_type: 6,
                ..
            }
        ));
        for message_type in [2, 25, 64] {
            assert!(matches!(
                encode_err(SbasMessage::Unsupported(SbasUnsupported {
                    preamble: 0x53,
                    message_type,
                    data: vec![0; RAW_PAYLOAD_LEN],
                })),
                SbasEncodeError::MessageType { message_type: got, .. } if got == message_type
            ));
        }
    }

    #[test]
    fn strict_encode_refuses_an_unrecognized_preamble() {
        let mut fast = fast_message();
        fast.preamble = 0x00;
        let message = SbasMessage::FastCorrections(fast);
        assert_eq!(
            encode_err(message.clone()),
            SbasEncodeError::UnrecognizedPreamble { preamble: 0x00 }
        );
        let (bytes, departures) = SbasBlock {
            form: SbasWireForm::Body226,
            message: message.clone(),
            pad_bits: 0,
        }
        .encode_with_policy(SbasPolicy::Lenient)
        .expect("lenient encode");
        assert_eq!(
            departures,
            vec![SbasDeparture::UnrecognizedPreamble { preamble: 0x00 }]
        );
        assert!(SbasBlock::decode(&bytes, SbasWireForm::Body226).is_err());
        let (decoded, _) =
            SbasBlock::decode_with_policy(&bytes, SbasWireForm::Body226, SbasPolicy::Lenient)
                .expect("lenient decode");
        assert_eq!(decoded.message, message);
    }
}
