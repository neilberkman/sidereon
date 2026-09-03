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
const PREAMBLES: [u8; 3] = [0x53, 0x9A, 0xC6];

#[derive(Clone, Debug, PartialEq, Eq, Default)]
/// Raw reserved segments retained from an SBAS message payload.
///
/// Each tuple stores a segment's value and width in bits, in wire order, so
/// the encoder can replay reserved fields whose layout is known.
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
    /// An SBAS preamble accepted by [`SbasBlock::decode`].
    pub preamble: u8,
    /// Raw payload bits; encoding uses at most 212 bits and pads short input.
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The 210-position satellite mask and issue fields from message type 1.
pub struct SbasPrnMask {
    /// An SBAS preamble accepted by [`SbasBlock::decode`].
    pub preamble: u8,
    /// The two-bit issue used by the correction store to select this mask.
    pub iodp: u8,
    /// Mask flags in wire order; true positions are resolved to monitored satellites.
    pub mask: [bool; 210],
    /// Empty for decoded message type 1 values because its fields fill the payload.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The thirteen fast-correction slots in message types 2 through 5.
pub struct SbasFastCorrections {
    /// An SBAS preamble accepted by [`SbasBlock::decode`].
    pub preamble: u8,
    /// The original message ID, in the inclusive range 2 through 5.
    pub message_type: SbasMessageType,
    /// The two-bit issue matched against integrity blocks.
    pub iodf: u8,
    /// The two-bit issue matched against the active PRN mask.
    pub iodp: u8,
    /// Signed 12-bit pseudorange correction counts, scaled by the store at 0.125 meters.
    pub prc: [i16; 13],
    /// Four-bit integrity indices for the thirteen correction slots.
    pub udrei: [u8; 13],
    /// Empty because the typed fields fill all 212 data bits.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The integrity blocks and UDREI values from message type 6.
pub struct SbasIntegrity {
    /// An SBAS preamble accepted by [`SbasBlock::decode`].
    pub preamble: u8,
    /// Four two-bit issue values, one for each group of thirteen UDREIs.
    pub iodf: [u8; 4],
    /// Fifty-one four-bit integrity indices applied to matching fast slots.
    pub udrei: [u8; 51],
    /// Empty because the typed fields fill all 212 data bits.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The latency and degradation indicators from message type 7.
pub struct SbasFastDegradation {
    /// An SBAS preamble accepted by [`SbasBlock::decode`].
    pub preamble: u8,
    /// Four-bit latency, interpreted directly as seconds by the correction store.
    pub system_latency_s: u8,
    /// The two-bit issue that gates acceptance of the latency value.
    pub iodp: u8,
    /// Fifty-one four-bit degradation indicators preserved by the codec.
    pub ai: [u8; 51],
    /// The two trailing reserved bits from the payload.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The raw GEO navigation coefficients from message type 9.
pub struct SbasGeoNav {
    /// An SBAS preamble accepted by [`SbasBlock::decode`].
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
    /// An SBAS preamble accepted by [`SbasBlock::decode`].
    pub preamble: u8,
    /// Raw payload bits normalized to 212 bits when encoded.
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The uninterpreted raw payload carried by message type 17.
pub struct SbasGeoAlmanac {
    /// An SBAS preamble accepted by [`SbasBlock::decode`].
    pub preamble: u8,
    /// Raw payload bits normalized to 212 bits when encoded.
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The combined fast and long-term payload from message type 24.
pub struct SbasMixedCorrections {
    /// An SBAS preamble accepted by [`SbasBlock::decode`].
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
    /// An SBAS preamble accepted by [`SbasBlock::decode`].
    pub preamble: u8,
    /// The two 106-bit halves, visited in order by correction-store ingest.
    pub halves: [SbasLongTermHalf; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// One 106-bit long-term correction half.
pub struct SbasLongTermHalf {
    /// Selects one velocity record when true, or two non-velocity records when false.
    pub velocity_code: bool,
    /// The two-bit issue matched against the active PRN mask.
    pub iodp: u8,
    /// One record in velocity mode, or two records in non-velocity mode.
    pub records: Vec<SbasLongTermRecord>,
    /// The trailing reserved bit available in non-velocity mode.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Raw orbit and clock deltas for one monitored satellite.
pub struct SbasLongTermRecord {
    /// A six-bit one-based index into the active monitored-satellite mask.
    pub monitored_index: u8,
    /// The eight-bit issue used to select the broadcast ephemeris.
    pub iode: u8,
    /// Signed raw X ECEF delta, scaled at 0.125 meters by the store.
    pub delta_x: i32,
    /// Signed raw Y ECEF delta, scaled at 0.125 meters by the store.
    pub delta_y: i32,
    /// Signed raw Z ECEF delta, scaled at 0.125 meters by the store.
    pub delta_z: i32,
    /// Signed raw X ECEF rate, scaled at 0.000625 meters per second in velocity mode.
    pub delta_x_rate: i32,
    /// Signed raw Y ECEF rate, scaled at 0.000625 meters per second in velocity mode.
    pub delta_y_rate: i32,
    /// Signed raw Z ECEF rate, scaled at 0.000625 meters per second in velocity mode.
    pub delta_z_rate: i32,
    /// Signed raw clock-offset delta, scaled at 1/2^31 seconds.
    pub delta_a_f0: i32,
    /// Signed raw clock-drift delta, scaled at 1/2^39 seconds per second in velocity mode.
    pub delta_a_f1: i32,
    /// A velocity-mode time-of-day count in 16-second units, or `None` in non-velocity mode.
    pub time_of_day_s: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The band- and IODI-qualified 201-position mask from message type 18.
pub struct SbasIgpMask {
    /// An SBAS preamble accepted by [`SbasBlock::decode`].
    pub preamble: u8,
    /// The four-bit ionospheric band selector.
    pub band_number: u8,
    /// The two-bit ionospheric issue used to match delay blocks.
    pub iodi: u8,
    /// Mask flags in wire order; active positions receive delay entries.
    pub mask: [bool; 201],
    /// The four-bit prefix and optional one-bit trailing reserved segments.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Fifteen ionospheric delay/GIVE entries from message type 26.
pub struct SbasIonoDelays {
    /// An SBAS preamble accepted by [`SbasBlock::decode`].
    pub preamble: u8,
    /// The four-bit ionospheric band selector.
    pub band_number: u8,
    /// The four-bit block selector, with fifteen entries per block.
    pub block_id: u8,
    /// The two-bit ionospheric issue matched against the IGP mask.
    pub iodi: u8,
    /// Fifteen entries assigned to consecutive active IGP positions.
    pub entries: [SbasIgpDelay; 15],
    /// The seven trailing reserved bits, when present.
    pub reserved: SpareBits,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
/// One raw vertical-delay and GIVE pair from an ionospheric delay block.
pub struct SbasIgpDelay {
    /// A nine-bit delay count scaled at 0.125 meters by the store.
    pub vertical_delay: u16,
    /// A four-bit GIVE index; 15 marks an unusable entry.
    pub givei: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// A non-phase-A SBAS message retained without a typed payload decoder.
pub struct SbasUnsupported {
    /// An SBAS preamble accepted by [`SbasBlock::decode`].
    pub preamble: u8,
    /// The original six-bit message ID.
    pub message_type: SbasMessageType,
    /// Raw payload bits normalized to 212 bits when encoded.
    pub data: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// The wire representation that controls block length and CRC handling.
pub enum SbasWireForm {
    /// A 32-byte block carrying 226 body bits, 24 CRC bits, and six pad bits.
    Framed250,
    /// A 29-byte body carrying 226 bits and six unused trailing bits.
    Body226,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The decoded message and wire representation needed for re-encoding.
///
/// [`SbasBlock::decode`] fills both fields, and [`SbasBlock::encode`] uses
/// them to select body-only or CRC-framed output.
pub struct SbasBlock {
    /// The body-only or CRC-framed representation selected for this block.
    pub form: SbasWireForm,
    /// The typed or raw message decoded from the block body.
    pub message: SbasMessage,
}

struct CountedBitWriter {
    writer: BitWriter,
    nbits: usize,
}

impl CountedBitWriter {
    fn new() -> Self {
        Self {
            writer: BitWriter::new(),
            nbits: 0,
        }
    }

    fn push_u(&mut self, value: u64, width: usize) {
        self.writer.push_u(value, width);
        self.nbits += width;
    }

    fn push_i(&mut self, value: i64, width: usize) {
        self.writer.push_i(value, width);
        self.nbits += width;
    }

    fn push_flag(&mut self, value: bool) {
        self.writer.push_flag(value);
        self.nbits += 1;
    }

    fn push_bits_from(&mut self, bytes: &[u8], nbits: usize) {
        for bit_pos in 0..nbits {
            self.push_u(u64::from(bit_at(bytes, bit_pos)), 1);
        }
    }

    fn pad_to(&mut self, nbits: usize) {
        while self.nbits < nbits {
            self.push_u(0, 1);
        }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.writer.into_bytes()
    }
}

impl SbasBlock {
    /// Decode a body or CRC-framed SBAS block.
    ///
    /// Body input must be 29 bytes. Framed input must be 32 bytes and must
    /// contain a CRC-24Q matching the first 226 bits. Both forms require one
    /// of the three recognized SBAS preambles; invalid lengths, CRCs,
    /// preambles, and bit reads are returned as parse errors.
    pub fn decode(bytes: &[u8], form: SbasWireForm) -> Result<Self> {
        match form {
            SbasWireForm::Framed250 => {
                if bytes.len() != FRAMED_LEN {
                    return Err(parse_error("SBAS framed block must be 32 bytes"));
                }
                let got = bits_as_u32(bytes, BODY_BITS, CRC_BITS);
                let want = crc24q_bits(bytes, BODY_BITS);
                if got != want {
                    return Err(parse_error("SBAS CRC mismatch"));
                }
            }
            SbasWireForm::Body226 => {
                if bytes.len() != BODY_LEN {
                    return Err(parse_error("SBAS body block must be 29 bytes"));
                }
            }
        }

        let mut reader = BitReader::new(bytes);
        let preamble = reader.u(8)? as u8;
        if !PREAMBLES.contains(&preamble) {
            return Err(parse_error("SBAS preamble not recognized"));
        }
        let message_type = reader.u(6)? as u8;
        let data = read_bits_as_bytes(&mut reader, DATA_BITS)?;
        let message = decode_message(preamble, message_type, &data)?;
        Ok(Self { form, message })
    }

    /// Encode the message as the block's selected body or framed wire form.
    ///
    /// The body contains the preamble, six-bit message ID, and a normalized
    /// 212-bit data payload. Framed output appends CRC-24Q over those 226 body
    /// bits and six zero pad bits.
    pub fn encode(&self) -> Vec<u8> {
        let mut body = CountedBitWriter::new();
        body.push_u(u64::from(self.message.preamble()), 8);
        body.push_u(u64::from(self.message.message_type()), 6);
        let data = self.message.encode_data();
        body.push_bits_from(&data, DATA_BITS);
        body.pad_to(BODY_BITS);
        let body_bytes = body.into_bytes();

        match self.form {
            SbasWireForm::Body226 => body_bytes,
            SbasWireForm::Framed250 => {
                let crc = crc24q_bits(&body_bytes, BODY_BITS);
                let mut framed = CountedBitWriter::new();
                framed.push_bits_from(&body_bytes, BODY_BITS);
                framed.push_u(u64::from(crc), CRC_BITS);
                framed.pad_to(FRAMED_BITS);
                framed.into_bytes()
            }
        }
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

    fn encode_data(&self) -> Vec<u8> {
        match self {
            Self::DoNotUse(m) => data_from_raw(&m.data),
            Self::PrnMask(m) => encode_prn_mask(m),
            Self::FastCorrections(m) => encode_fast(m),
            Self::Integrity(m) => encode_integrity(m),
            Self::FastDegradation(m) => encode_fast_degradation(m),
            Self::GeoNav(m) => encode_geo_nav(m),
            Self::NetworkTime(m) => data_from_raw(&m.data),
            Self::GeoAlmanac(m) => data_from_raw(&m.data),
            Self::IgpMask(m) => encode_igp_mask(m),
            Self::MixedCorrections(m) => encode_mixed(m),
            Self::LongTermCorrections(m) => encode_long_term(m),
            Self::IonoDelays(m) => encode_iono_delays(m),
            Self::Unsupported(m) => data_from_raw(&m.data),
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
            data: data_from_raw(data),
        }));
    }
    match message_type {
        0 => Ok(SbasMessage::DoNotUse(SbasDoNotUse {
            preamble,
            data: data_from_raw(data),
        })),
        1 => decode_prn_mask(preamble, data).map(SbasMessage::PrnMask),
        2..=5 => decode_fast(preamble, message_type, data).map(SbasMessage::FastCorrections),
        6 => decode_integrity(preamble, data).map(SbasMessage::Integrity),
        7 => decode_fast_degradation(preamble, data).map(SbasMessage::FastDegradation),
        9 => decode_geo_nav(preamble, data).map(SbasMessage::GeoNav),
        12 => Ok(SbasMessage::NetworkTime(SbasNetworkTime {
            preamble,
            data: data_from_raw(data),
        })),
        17 => Ok(SbasMessage::GeoAlmanac(SbasGeoAlmanac {
            preamble,
            data: data_from_raw(data),
        })),
        18 => decode_igp_mask(preamble, data).map(SbasMessage::IgpMask),
        24 => decode_mixed(preamble, data).map(SbasMessage::MixedCorrections),
        25 => decode_long_term(preamble, data).map(SbasMessage::LongTermCorrections),
        26 => decode_iono_delays(preamble, data).map(SbasMessage::IonoDelays),
        _ => unreachable!("phase A classification and decode match are out of sync"),
    }
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

fn encode_prn_mask(m: &SbasPrnMask) -> Vec<u8> {
    let mut w = CountedBitWriter::new();
    for bit in m.mask {
        w.push_flag(bit);
    }
    w.push_u(u64::from(m.iodp), 2);
    w.pad_to(DATA_BITS);
    w.into_bytes()
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

fn encode_fast(m: &SbasFastCorrections) -> Vec<u8> {
    let mut w = CountedBitWriter::new();
    w.push_u(u64::from(m.iodf), 2);
    w.push_u(u64::from(m.iodp), 2);
    for value in m.prc {
        w.push_i(i64::from(value), 12);
    }
    for value in m.udrei {
        w.push_u(u64::from(value), 4);
    }
    w.pad_to(DATA_BITS);
    w.into_bytes()
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

fn encode_integrity(m: &SbasIntegrity) -> Vec<u8> {
    let mut w = CountedBitWriter::new();
    for value in m.iodf {
        w.push_u(u64::from(value), 2);
    }
    for value in m.udrei {
        w.push_u(u64::from(value), 4);
    }
    w.pad_to(DATA_BITS);
    w.into_bytes()
}

fn decode_fast_degradation(preamble: u8, data: &[u8]) -> Result<SbasFastDegradation> {
    let mut r = BitReader::new(data);
    let system_latency_s = r.u(4)? as u8;
    let iodp = r.u(2)? as u8;
    let mut ai = [0u8; 51];
    for value in &mut ai {
        *value = r.u(4)? as u8;
    }
    let mut reserved = SpareBits::new();
    if r.remaining_bits() >= 2 {
        reserved.push(r.u(2)?, 2);
    }
    Ok(SbasFastDegradation {
        preamble,
        system_latency_s,
        iodp,
        ai,
        reserved,
    })
}

fn encode_fast_degradation(m: &SbasFastDegradation) -> Vec<u8> {
    let mut w = CountedBitWriter::new();
    w.push_u(u64::from(m.system_latency_s), 4);
    w.push_u(u64::from(m.iodp), 2);
    for value in m.ai {
        w.push_u(u64::from(value), 4);
    }
    push_spares(&mut w, &m.reserved);
    w.pad_to(DATA_BITS);
    w.into_bytes()
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

fn encode_geo_nav(m: &SbasGeoNav) -> Vec<u8> {
    let mut w = CountedBitWriter::new();
    let spare_start = push_prefix_spare(&mut w, &m.reserved, 8);
    w.push_u(u64::from(m.time_of_day_s), 13);
    w.push_u(u64::from(m.ura), 4);
    w.push_i(i64::from(m.x_m), 30);
    w.push_i(i64::from(m.y_m), 30);
    w.push_i(i64::from(m.z_m), 25);
    w.push_i(i64::from(m.x_rate_m_s), 17);
    w.push_i(i64::from(m.y_rate_m_s), 17);
    w.push_i(i64::from(m.z_rate_m_s), 18);
    w.push_i(i64::from(m.x_accel_m_s2), 10);
    w.push_i(i64::from(m.y_accel_m_s2), 10);
    w.push_i(i64::from(m.z_accel_m_s2), 10);
    w.push_i(i64::from(m.a_gf0_s), 12);
    w.push_i(i64::from(m.a_gf1_s_s), 8);
    push_spares_from(&mut w, &m.reserved, spare_start);
    w.pad_to(DATA_BITS);
    w.into_bytes()
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
    if r.remaining_bits() >= 1 {
        reserved.push(r.u(1)?, 1);
    }
    Ok(SbasIgpMask {
        preamble,
        band_number,
        iodi,
        mask,
        reserved,
    })
}

fn encode_igp_mask(m: &SbasIgpMask) -> Vec<u8> {
    let mut w = CountedBitWriter::new();
    let spare_start = push_prefix_spare(&mut w, &m.reserved, 4);
    w.push_u(u64::from(m.band_number), 4);
    w.push_u(u64::from(m.iodi), 2);
    for bit in m.mask {
        w.push_flag(bit);
    }
    push_spares_from(&mut w, &m.reserved, spare_start);
    w.pad_to(DATA_BITS);
    w.into_bytes()
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
    let long_raw = read_bits_as_bytes(&mut r, 106)?;
    let long_term = decode_long_half(&long_raw)?;
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

fn encode_mixed(m: &SbasMixedCorrections) -> Vec<u8> {
    let mut w = CountedBitWriter::new();
    for value in m.fast.prc {
        w.push_i(i64::from(value), 12);
    }
    for value in m.fast.udrei {
        w.push_u(u64::from(value), 4);
    }
    w.push_u(u64::from(m.fast.iodp), 2);
    w.push_u(u64::from(m.fast.block_id), 2);
    w.push_u(u64::from(m.fast.iodf), 2);
    push_spares(&mut w, &m.fast.reserved);
    w.pad_to(106);
    let long = encode_long_half(&m.long_term);
    w.push_bits_from(&long, 106);
    w.pad_to(DATA_BITS);
    w.into_bytes()
}

fn decode_long_term(preamble: u8, data: &[u8]) -> Result<SbasLongTermCorrections> {
    let mut r = BitReader::new(data);
    let first = read_bits_as_bytes(&mut r, 106)?;
    let second = read_bits_as_bytes(&mut r, 106)?;
    Ok(SbasLongTermCorrections {
        preamble,
        halves: [decode_long_half(&first)?, decode_long_half(&second)?],
    })
}

fn encode_long_term(m: &SbasLongTermCorrections) -> Vec<u8> {
    let mut w = CountedBitWriter::new();
    for half in &m.halves {
        let encoded = encode_long_half(half);
        w.push_bits_from(&encoded, 106);
    }
    w.pad_to(DATA_BITS);
    w.into_bytes()
}

fn decode_long_half(raw: &[u8]) -> Result<SbasLongTermHalf> {
    let mut r = BitReader::new(raw);
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
        if r.remaining_bits() >= 1 {
            reserved.push(r.u(1)?, 1);
        }
        Ok(SbasLongTermHalf {
            velocity_code,
            iodp,
            records,
            reserved,
        })
    }
}

fn encode_long_half(m: &SbasLongTermHalf) -> Vec<u8> {
    let mut w = CountedBitWriter::new();
    w.push_flag(m.velocity_code);
    if m.velocity_code {
        if let Some(record) = m.records.first() {
            push_long_record_with_velocity(&mut w, record);
        } else {
            push_long_record_with_velocity(&mut w, &zero_long_record());
        }
        w.push_u(u64::from(m.iodp), 2);
    } else {
        for idx in 0..2 {
            if let Some(record) = m.records.get(idx) {
                push_long_record_without_velocity(&mut w, record);
            } else {
                push_long_record_without_velocity(&mut w, &zero_long_record());
            }
        }
        w.push_u(u64::from(m.iodp), 2);
        push_spares(&mut w, &m.reserved);
    }
    w.pad_to(106);
    w.into_bytes()
}

fn push_long_record_without_velocity(w: &mut CountedBitWriter, record: &SbasLongTermRecord) {
    w.push_u(u64::from(record.monitored_index), 6);
    w.push_u(u64::from(record.iode), 8);
    w.push_i(i64::from(record.delta_x), 9);
    w.push_i(i64::from(record.delta_y), 9);
    w.push_i(i64::from(record.delta_z), 9);
    w.push_i(i64::from(record.delta_a_f0), 10);
}

fn push_long_record_with_velocity(w: &mut CountedBitWriter, record: &SbasLongTermRecord) {
    w.push_u(u64::from(record.monitored_index), 6);
    w.push_u(u64::from(record.iode), 8);
    w.push_i(i64::from(record.delta_x), 11);
    w.push_i(i64::from(record.delta_y), 11);
    w.push_i(i64::from(record.delta_z), 11);
    w.push_i(i64::from(record.delta_a_f0), 11);
    w.push_i(i64::from(record.delta_x_rate), 8);
    w.push_i(i64::from(record.delta_y_rate), 8);
    w.push_i(i64::from(record.delta_z_rate), 8);
    w.push_i(i64::from(record.delta_a_f1), 8);
    w.push_u(u64::from(record.time_of_day_s.unwrap_or(0)), 13);
}

fn zero_long_record() -> SbasLongTermRecord {
    SbasLongTermRecord {
        monitored_index: 0,
        iode: 0,
        delta_x: 0,
        delta_y: 0,
        delta_z: 0,
        delta_x_rate: 0,
        delta_y_rate: 0,
        delta_z_rate: 0,
        delta_a_f0: 0,
        delta_a_f1: 0,
        time_of_day_s: None,
    }
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
    if r.remaining_bits() >= 7 {
        reserved.push(r.u(7)?, 7);
    }
    Ok(SbasIonoDelays {
        preamble,
        band_number,
        block_id,
        iodi,
        entries,
        reserved,
    })
}

fn encode_iono_delays(m: &SbasIonoDelays) -> Vec<u8> {
    let mut w = CountedBitWriter::new();
    w.push_u(u64::from(m.band_number), 4);
    w.push_u(u64::from(m.block_id), 4);
    for entry in &m.entries {
        w.push_u(u64::from(entry.vertical_delay), 9);
        w.push_u(u64::from(entry.givei), 4);
    }
    w.push_u(u64::from(m.iodi), 2);
    push_spares(&mut w, &m.reserved);
    w.pad_to(DATA_BITS);
    w.into_bytes()
}

fn push_spares(w: &mut CountedBitWriter, spare: &SpareBits) {
    for &(value, width) in &spare.0 {
        w.push_u(value, usize::from(width));
    }
}

fn push_prefix_spare(w: &mut CountedBitWriter, spare: &SpareBits, width: u8) -> usize {
    if let Some(&(value, got_width)) = spare
        .0
        .first()
        .filter(|&&(_, got_width)| got_width == width)
    {
        w.push_u(value, usize::from(got_width));
        1
    } else {
        w.push_u(0, usize::from(width));
        0
    }
}

fn push_spares_from(w: &mut CountedBitWriter, spare: &SpareBits, start: usize) {
    for &(value, width) in spare.0.iter().skip(start) {
        w.push_u(value, usize::from(width));
    }
}

fn read_bits_as_bytes(reader: &mut BitReader<'_>, nbits: usize) -> Result<Vec<u8>> {
    let mut w = CountedBitWriter::new();
    for _ in 0..nbits {
        w.push_u(reader.u(1)?, 1);
    }
    Ok(w.into_bytes())
}

fn data_from_raw(raw: &[u8]) -> Vec<u8> {
    let mut w = CountedBitWriter::new();
    w.push_bits_from(raw, DATA_BITS.min(raw.len() * 8));
    w.pad_to(DATA_BITS);
    w.into_bytes()
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
        SbasBlock { form, message }.encode()
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
            assert_eq!(decoded.encode(), encoded);
            assert_eq!(decoded.message.message_type(), 1);
        }
    }

    #[test]
    fn corrupted_crc_is_rejected() {
        let msg = SbasMessage::Unsupported(SbasUnsupported {
            preamble: 0x9A,
            message_type: 62,
            data: vec![0; BODY_LEN],
        });
        let mut encoded = block(msg, SbasWireForm::Framed250);
        encoded[31] ^= 0x40;
        assert!(SbasBlock::decode(&encoded, SbasWireForm::Framed250).is_err());
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
        let body = block(
            SbasMessage::MixedCorrections(SbasMixedCorrections {
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
                    velocity_code: false,
                    iodp: 2,
                    records: vec![
                        SbasLongTermRecord {
                            monitored_index: 5,
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
                        },
                        SbasLongTermRecord {
                            monitored_index: 6,
                            iode: 10,
                            delta_x: 14,
                            delta_y: -15,
                            delta_z: 16,
                            delta_x_rate: 0,
                            delta_y_rate: 0,
                            delta_z_rate: 0,
                            delta_a_f0: -17,
                            delta_a_f1: 0,
                            time_of_day_s: None,
                        },
                    ],
                    reserved: SpareBits(vec![(1, 1)]),
                },
            }),
            SbasWireForm::Body226,
        );

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
    }
}
