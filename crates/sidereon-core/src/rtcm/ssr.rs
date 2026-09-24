//! RTCM SSR orbit, clock, URA, and high-rate clock messages.
//!
//! This module is the wire-level IR for the RTCM SSR Phase A messages. Values
//! are stored as the raw transmitted integers. Scaling to meters and seconds is
//! handled by the crate-level `ssr` correction store.

use crate::error::{Error, Result};
use crate::id::GnssSystem;

use super::bits::{BitReader, FieldWriter};
use super::{
    is_departing_tail, DecodeContext, DecodeError, DecodeResult, RtcmDeparture, RtcmEncodeError,
    RtcmPolicy, RtcmRecordKind,
};

/// The SSR message group derived from the message number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SsrKind {
    /// Orbit corrections.
    Orbit,
    /// Clock corrections.
    Clock,
    /// Combined orbit and clock corrections.
    CombinedOrbitClock,
    /// Code-bias corrections.
    CodeBias,
    /// Phase-bias corrections.
    PhaseBias,
    /// User range accuracy.
    Ura,
    /// High-rate clock correction.
    HighRateClock,
    /// VTEC ionosphere correction.
    Vtec,
}

/// Common header for RTCM SSR messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsrHeader {
    /// SSR epoch time. GPS, Galileo, and BeiDou use seconds of week; GLONASS
    /// uses its RTCM 17-bit day time field.
    pub epoch_time_s: u32,
    /// SSR update interval index.
    pub update_interval: u8,
    /// Multiple-message indicator.
    pub multiple_message: bool,
    /// IOD SSR.
    pub iod_ssr: u8,
    /// SSR provider identifier.
    pub provider_id: u16,
    /// SSR solution identifier.
    pub solution_id: u8,
    /// Satellite reference datum bit for orbit and combined messages.
    pub satellite_reference_datum: Option<bool>,
    /// Phase-bias dispersive-bias consistency flag.
    pub dispersive_bias_consistency: Option<bool>,
    /// Phase-bias Melbourne-Wubbena consistency flag.
    pub mw_consistency: Option<bool>,
    /// Number of satellite records.
    pub satellite_count: u8,
}

/// One satellite orbit-correction record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsrOrbitRecord {
    /// Constellation-native satellite id.
    pub satellite_id: u8,
    /// Referenced broadcast issue, with constellation-specific bit width.
    pub iode: u32,
    /// Radial delta, int22, scale 0.1 mm.
    pub delta_radial: i32,
    /// Along-track delta, int20, scale 0.4 mm.
    pub delta_along: i32,
    /// Cross-track delta, int20, scale 0.4 mm.
    pub delta_cross: i32,
    /// Radial delta rate, int21, scale 0.001 mm/s.
    pub dot_delta_radial: i32,
    /// Along-track delta rate, int19, scale 0.004 mm/s.
    pub dot_delta_along: i32,
    /// Cross-track delta rate, int19, scale 0.004 mm/s.
    pub dot_delta_cross: i32,
}

/// One satellite clock-correction record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsrClockRecord {
    /// Constellation-native satellite id.
    pub satellite_id: u8,
    /// C0 clock term, int22, scale 0.1 mm.
    pub c0: i32,
    /// C1 clock term, int21, scale 0.001 mm/s.
    pub c1: i32,
    /// C2 clock term, int27, scale 0.02 micrometer/s^2.
    pub c2: i32,
}

/// One satellite code-bias record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsrCodeBiasRecord {
    /// Constellation-native satellite id.
    pub satellite_id: u8,
    /// Raw signal and tracking-mode id plus raw bias.
    pub biases: Vec<(u8, i16)>,
}

/// One signal in a phase-bias record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsrPhaseBiasSignal {
    /// Raw signal and tracking-mode id.
    pub signal_id: u8,
    /// Signal integer indicator.
    pub integer_indicator: u8,
    /// Wide-lane integer indicator.
    pub wide_lane_integer_indicator: u8,
    /// Discontinuity counter.
    pub discontinuity_counter: u8,
    /// Raw phase bias.
    pub bias: i32,
}

/// One satellite phase-bias record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsrPhaseBiasRecord {
    /// Constellation-native satellite id.
    pub satellite_id: u8,
    /// Raw yaw angle.
    pub yaw_angle: u16,
    /// Raw yaw rate.
    pub yaw_rate: i8,
    /// Per-signal phase biases.
    pub biases: Vec<SsrPhaseBiasSignal>,
}

/// A decoded RTCM SSR message body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsrMessage {
    /// RTCM message number.
    pub message_number: u16,
    /// Constellation derived from the message number.
    pub system: GnssSystem,
    /// SSR message group.
    pub kind: SsrKind,
    /// Common SSR header.
    pub header: SsrHeader,
    /// Orbit records, present for orbit and combined messages.
    pub orbit: Vec<SsrOrbitRecord>,
    /// Clock records, present for clock, combined, and high-rate messages.
    pub clock: Vec<SsrClockRecord>,
    /// Code-bias records.
    pub code_bias: Vec<SsrCodeBiasRecord>,
    /// Phase-bias records.
    pub phase_bias: Vec<SsrPhaseBiasRecord>,
    /// URA records as `(satellite_id, ura_index)`.
    pub ura: Vec<(u8, u8)>,
    /// Every body bit after the last record, including the zero bits that
    /// align the body to a byte, written back after the records. Anything
    /// other than fewer than eight zero bits is an
    /// [`RtcmDeparture::TrailingBits`] (or, after a short body read under
    /// [`RtcmPolicy::Lenient`], the bits of its cut record).
    pub padding_bits: Vec<bool>,
}

/// Map an RTCM message number to a supported SSR Phase A type.
pub(crate) fn ssr_kind(message_number: u16) -> Option<(GnssSystem, SsrKind)> {
    match message_number {
        1057 => Some((GnssSystem::Gps, SsrKind::Orbit)),
        1058 => Some((GnssSystem::Gps, SsrKind::Clock)),
        1059 => Some((GnssSystem::Gps, SsrKind::CodeBias)),
        1060 => Some((GnssSystem::Gps, SsrKind::CombinedOrbitClock)),
        1061 => Some((GnssSystem::Gps, SsrKind::Ura)),
        1062 => Some((GnssSystem::Gps, SsrKind::HighRateClock)),
        1063 => Some((GnssSystem::Glonass, SsrKind::Orbit)),
        1064 => Some((GnssSystem::Glonass, SsrKind::Clock)),
        1065 => Some((GnssSystem::Glonass, SsrKind::CodeBias)),
        1066 => Some((GnssSystem::Glonass, SsrKind::CombinedOrbitClock)),
        1067 => Some((GnssSystem::Glonass, SsrKind::Ura)),
        1068 => Some((GnssSystem::Glonass, SsrKind::HighRateClock)),
        1265 => Some((GnssSystem::Gps, SsrKind::PhaseBias)),
        1240 => Some((GnssSystem::Galileo, SsrKind::Orbit)),
        1241 => Some((GnssSystem::Galileo, SsrKind::Clock)),
        1242 => Some((GnssSystem::Galileo, SsrKind::CodeBias)),
        1243 => Some((GnssSystem::Galileo, SsrKind::CombinedOrbitClock)),
        1244 => Some((GnssSystem::Galileo, SsrKind::Ura)),
        1245 => Some((GnssSystem::Galileo, SsrKind::HighRateClock)),
        1267 => Some((GnssSystem::Galileo, SsrKind::PhaseBias)),
        1246 => Some((GnssSystem::Qzss, SsrKind::Orbit)),
        1247 => Some((GnssSystem::Qzss, SsrKind::Clock)),
        1248 => Some((GnssSystem::Qzss, SsrKind::CodeBias)),
        1249 => Some((GnssSystem::Qzss, SsrKind::CombinedOrbitClock)),
        1250 => Some((GnssSystem::Qzss, SsrKind::Ura)),
        1251 => Some((GnssSystem::Qzss, SsrKind::HighRateClock)),
        1268 => Some((GnssSystem::Qzss, SsrKind::PhaseBias)),
        1258 => Some((GnssSystem::BeiDou, SsrKind::Orbit)),
        1259 => Some((GnssSystem::BeiDou, SsrKind::Clock)),
        1260 => Some((GnssSystem::BeiDou, SsrKind::CodeBias)),
        1261 => Some((GnssSystem::BeiDou, SsrKind::CombinedOrbitClock)),
        1262 => Some((GnssSystem::BeiDou, SsrKind::Ura)),
        1263 => Some((GnssSystem::BeiDou, SsrKind::HighRateClock)),
        1270 => Some((GnssSystem::BeiDou, SsrKind::PhaseBias)),
        _ => None,
    }
}

/// True when this module decodes `message_number`.
pub(crate) fn is_supported_ssr(message_number: u16) -> bool {
    ssr_kind(message_number).is_some()
}

impl SsrMessage {
    /// Decode one RTCM SSR body, without the transport frame, under
    /// [`RtcmPolicy::Strict`]: a body that ends before the records its header
    /// counts is refused as truncated.
    ///
    /// Every bit after the last record is kept in [`Self::padding_bits`], so
    /// the body re-encodes as read.
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_inner(body, &mut DecodeContext::new(RtcmPolicy::Strict)).map_err(Into::into)
    }

    /// Decode one RTCM SSR body under `policy`, returning the departures read
    /// under [`RtcmPolicy::Lenient`].
    ///
    /// Under [`RtcmPolicy::Lenient`] a body that ends before the records its
    /// header counts is read as RTKLIB `decode_ssr1`..`decode_ssr7` read it:
    /// every complete record, with the header count kept as transmitted, the
    /// bits of the incomplete record kept in [`Self::padding_bits`], and an
    /// [`RtcmDeparture::SsrRecordsShort`] naming both counts. RTKLIB also keeps
    /// a bias record cut inside its signal list with the signals it holds;
    /// here such a record is incomplete and is not read.
    pub fn decode_with_policy(
        body: &[u8],
        policy: RtcmPolicy,
    ) -> Result<(Self, Vec<RtcmDeparture>)> {
        let mut ctx = DecodeContext::new(policy);
        let message = Self::decode_inner(body, &mut ctx)?;
        Ok((message, ctx.into_departures()))
    }

    pub(crate) fn decode_inner(body: &[u8], ctx: &mut DecodeContext) -> DecodeResult<Self> {
        let mut r = BitReader::new(body);
        let message_number = r.u(12)? as u16;
        let (system, kind) = ssr_kind(message_number).ok_or_else(|| {
            Error::Parse(format!(
                "message {message_number} is not a supported RTCM SSR Phase A type"
            ))
        })?;
        let header = read_header(&mut r, system, kind)?;
        let sat_bits = satellite_id_bits(system, message_number);
        let count = usize::from(header.satellite_count);
        let mut orbit = Vec::new();
        let mut clock = Vec::new();
        let mut code_bias = Vec::new();
        let mut phase_bias = Vec::new();
        let mut ura = Vec::new();

        match kind {
            SsrKind::Orbit => {
                orbit = read_records(&mut r, ctx, message_number, count, &mut |r| {
                    read_orbit_record(r, system, sat_bits)
                })?;
            }
            SsrKind::Clock => {
                clock = read_records(&mut r, ctx, message_number, count, &mut |r| {
                    read_clock_record(r, sat_bits)
                })?;
            }
            SsrKind::CombinedOrbitClock => {
                let pairs = read_records(&mut r, ctx, message_number, count, &mut |r| {
                    let rec = read_orbit_record(r, system, sat_bits)?;
                    let clock = SsrClockRecord {
                        satellite_id: rec.satellite_id,
                        c0: r.i(22)? as i32,
                        c1: r.i(21)? as i32,
                        c2: r.i(27)? as i32,
                    };
                    Ok((rec, clock))
                })?;
                (orbit, clock) = pairs.into_iter().unzip();
            }
            SsrKind::Ura => {
                ura = read_records(&mut r, ctx, message_number, count, &mut |r| {
                    Ok((r.u(sat_bits)? as u8, r.u(6)? as u8))
                })?;
            }
            SsrKind::HighRateClock => {
                clock = read_records(&mut r, ctx, message_number, count, &mut |r| {
                    Ok(SsrClockRecord {
                        satellite_id: r.u(sat_bits)? as u8,
                        c0: r.i(22)? as i32,
                        c1: 0,
                        c2: 0,
                    })
                })?;
            }
            SsrKind::CodeBias => {
                code_bias = read_records(&mut r, ctx, message_number, count, &mut |r| {
                    read_code_bias_record(r, sat_bits)
                })?;
            }
            SsrKind::PhaseBias => {
                phase_bias = read_records(&mut r, ctx, message_number, count, &mut |r| {
                    read_phase_bias_record(r, sat_bits)
                })?;
            }
            SsrKind::Vtec => {
                return Err(Error::Parse(format!(
                    "message {message_number} is not enabled in RTCM SSR Phase A"
                ))
                .into());
            }
        }

        let padding_bits = r.rest();
        let records = orbit
            .len()
            .max(clock.len())
            .max(code_bias.len())
            .max(phase_bias.len())
            .max(ura.len());
        // A short body's leftover bits belong to its cut record, which
        // `read_records` has already reported.
        if records == count && is_departing_tail(&padding_bits) {
            ctx.depart(RtcmDeparture::TrailingBits {
                message_number,
                bits: padding_bits.clone(),
            })?;
        }

        Ok(Self {
            message_number,
            system,
            kind,
            header,
            orbit,
            clock,
            code_bias,
            phase_bias,
            ura,
            padding_bits,
        })
    }

    /// Encode this message back into an RTCM body.
    ///
    /// # Errors
    ///
    /// [`Error::RtcmEncode`] naming what cannot be written as the SSR wire
    /// form states it:
    ///
    /// * a message number this codec does not decode, or one whose
    ///   constellation and message group differ from [`Self::system`] and
    ///   [`Self::kind`]: the body would be written in one layout and read in
    ///   another (4076, the IGS SSR number, has its own layout);
    /// * a satellite field wider than the message's: five bits for GLONASS,
    ///   four for the native QZSS messages (1246..1251, 1268), six otherwise;
    /// * a satellite count that differs from the number of records the message
    ///   group writes, or records in a list the group does not write (an orbit
    ///   message's clock list, say), which would be dropped;
    /// * a combined orbit/clock message whose orbit and clock lists differ in
    ///   length, or whose clock record names a different satellite from the
    ///   orbit record at the same position; the frame writes each clock right
    ///   after its orbit under the orbit's satellite field;
    /// * a high-rate clock record with a nonzero `c1` or `c2`, which the
    ///   message does not carry;
    /// * a header flag present for a group that does not carry it or absent
    ///   for one that does (the satellite reference datum of orbit and
    ///   combined messages, the two consistency flags of phase-bias messages);
    /// * a code- or phase-bias record with more signals than its 5-bit count
    ///   states, or any value wider than its field.
    ///
    /// A header satellite count above the number of records written is an
    /// [`RtcmDeparture::SsrRecordsShort`], refused here and written by
    /// [`Self::encode_with_policy`] under [`RtcmPolicy::Lenient`].
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this message under `policy`. Under [`RtcmPolicy::Lenient`] a
    /// header satellite count above the number of records written is written
    /// as held and reported, which re-encodes a message read leniently from a
    /// short body; every other refusal of [`Self::encode`] applies under both
    /// policies.
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let number = self.message_number;
        if ssr_kind(number) != Some((self.system, self.kind)) {
            return Err(RtcmEncodeError::MessageNumber {
                message_number: number,
                record: RtcmRecordKind::Ssr {
                    system: self.system,
                    kind: self.kind,
                },
            }
            .into());
        }
        let departures = self.check_lists(policy)?;
        self.check_header_flags()?;
        let sat_bits = satellite_id_bits(self.system, number);
        let widest = (1u64 << sat_bits) - 1;
        if let Some(satellite_id) = self
            .satellite_fields()
            .into_iter()
            .find(|id| u64::from(*id) > widest)
        {
            return Err(RtcmEncodeError::SsrSatelliteIdOutOfRange {
                message_number: number,
                value: satellite_id,
                width: sat_bits as u8,
            }
            .into());
        }
        let mut w = FieldWriter::new(number);
        w.u("message number", u64::from(number), 12)?;
        write_header(&mut w, self.system, &self.header, self.kind)?;

        match self.kind {
            SsrKind::Orbit => {
                for rec in &self.orbit {
                    write_orbit_record(&mut w, self.system, sat_bits, rec)?;
                }
            }
            SsrKind::Clock => {
                for rec in &self.clock {
                    write_clock_record(&mut w, sat_bits, rec)?;
                }
            }
            SsrKind::CombinedOrbitClock => {
                for (orbit, clock) in self.orbit.iter().zip(&self.clock) {
                    write_orbit_record(&mut w, self.system, sat_bits, orbit)?;
                    write_clock_terms(&mut w, clock)?;
                }
            }
            SsrKind::Ura => {
                for &(satellite_id, index) in &self.ura {
                    w.u("satellite id", u64::from(satellite_id), sat_bits)?;
                    w.u(
                        format_args!("satellite {satellite_id} URA"),
                        u64::from(index),
                        6,
                    )?;
                }
            }
            SsrKind::HighRateClock => {
                for rec in &self.clock {
                    w.u("satellite id", u64::from(rec.satellite_id), sat_bits)?;
                    w.i(
                        format_args!("satellite {} high-rate clock", rec.satellite_id),
                        i64::from(rec.c0),
                        22,
                    )?;
                }
            }
            SsrKind::CodeBias => {
                for rec in &self.code_bias {
                    write_code_bias_record(&mut w, sat_bits, rec)?;
                }
            }
            SsrKind::PhaseBias => {
                for rec in &self.phase_bias {
                    write_phase_bias_record(&mut w, sat_bits, rec)?;
                }
            }
            SsrKind::Vtec => {}
        }

        let pad = (8 - (w.bit_len() + self.padding_bits.len()) % 8) % 8;
        let mut tail = self.padding_bits.clone();
        tail.extend(std::iter::repeat_n(false, pad));
        let mut departures = departures;
        if departures.is_empty() && is_departing_tail(&tail) {
            let departure = RtcmDeparture::TrailingBits {
                message_number: number,
                bits: tail,
            };
            match policy {
                RtcmPolicy::Strict => {
                    return Err(RtcmEncodeError::StrictDeparture(departure).into())
                }
                RtcmPolicy::Lenient => departures.push(departure),
            }
        }
        for &bit in &self.padding_bits {
            w.flag(bit);
        }
        Ok((w.into_bytes(), departures))
    }

    /// Refuse record lists the message group does not write, a satellite
    /// count other than the written record count, unpaired combined records,
    /// and high-rate clock terms the message does not carry.
    fn check_lists(&self, policy: RtcmPolicy) -> Result<Vec<RtcmDeparture>> {
        let number = self.message_number;
        let written = |name: &'static str, len: usize, writes: bool| -> Result<()> {
            if len > 0 && !writes {
                return Err(RtcmEncodeError::SsrRecordsNotCarried {
                    message_number: number,
                    kind: self.kind,
                    records: name,
                    count: len,
                }
                .into());
            }
            Ok(())
        };
        let kind = self.kind;
        written(
            "orbit",
            self.orbit.len(),
            matches!(kind, SsrKind::Orbit | SsrKind::CombinedOrbitClock),
        )?;
        written(
            "clock",
            self.clock.len(),
            matches!(
                kind,
                SsrKind::Clock | SsrKind::CombinedOrbitClock | SsrKind::HighRateClock
            ),
        )?;
        written("code-bias", self.code_bias.len(), kind == SsrKind::CodeBias)?;
        written(
            "phase-bias",
            self.phase_bias.len(),
            kind == SsrKind::PhaseBias,
        )?;
        written("URA", self.ura.len(), kind == SsrKind::Ura)?;

        if kind == SsrKind::CombinedOrbitClock {
            if self.orbit.len() != self.clock.len() {
                return Err(RtcmEncodeError::SsrCombinedRecordCounts {
                    message_number: number,
                    orbit: self.orbit.len(),
                    clock: self.clock.len(),
                }
                .into());
            }
            for (index, (orbit, clock)) in self.orbit.iter().zip(&self.clock).enumerate() {
                if orbit.satellite_id != clock.satellite_id {
                    return Err(RtcmEncodeError::SsrCombinedSatelliteMismatch {
                        message_number: number,
                        index,
                        orbit_satellite: orbit.satellite_id,
                        clock_satellite: clock.satellite_id,
                    }
                    .into());
                }
            }
        }
        if kind == SsrKind::HighRateClock {
            if let Some(rec) = self.clock.iter().find(|rec| rec.c1 != 0 || rec.c2 != 0) {
                return Err(RtcmEncodeError::SsrHighRateClockTerms {
                    message_number: number,
                    satellite: rec.satellite_id,
                    c1: rec.c1,
                    c2: rec.c2,
                }
                .into());
            }
        }
        let records = self.satellite_fields().len();
        let declared = usize::from(self.header.satellite_count);
        if declared > records && policy == RtcmPolicy::Lenient {
            return Ok(vec![RtcmDeparture::SsrRecordsShort {
                message_number: number,
                declared,
                read: records,
            }]);
        }
        if declared != records {
            return Err(RtcmEncodeError::SsrSatelliteCount {
                message_number: number,
                declared,
                records,
            }
            .into());
        }
        Ok(Vec::new())
    }

    /// Refuse a header flag present for a group that does not carry it, or
    /// absent for one that does.
    fn check_header_flags(&self) -> Result<()> {
        let number = self.message_number;
        let datum = matches!(self.kind, SsrKind::Orbit | SsrKind::CombinedOrbitClock);
        let phase = self.kind == SsrKind::PhaseBias;
        for (name, present, carried) in [
            (
                "satellite reference datum",
                self.header.satellite_reference_datum.is_some(),
                datum,
            ),
            (
                "dispersive bias consistency indicator",
                self.header.dispersive_bias_consistency.is_some(),
                phase,
            ),
            (
                "MW consistency indicator",
                self.header.mw_consistency.is_some(),
                phase,
            ),
        ] {
            if present != carried {
                return Err(RtcmEncodeError::FieldPresence {
                    message_number: number,
                    record: RtcmRecordKind::Ssr {
                        system: self.system,
                        kind: self.kind,
                    },
                    field: name,
                    carried,
                }
                .into());
            }
        }
        Ok(())
    }

    /// Every satellite field the encoder writes for this message's kind.
    fn satellite_fields(&self) -> Vec<u8> {
        match self.kind {
            SsrKind::Orbit => self.orbit.iter().map(|r| r.satellite_id).collect(),
            SsrKind::Clock | SsrKind::HighRateClock => {
                self.clock.iter().map(|r| r.satellite_id).collect()
            }
            SsrKind::CombinedOrbitClock => self
                .orbit
                .iter()
                .zip(&self.clock)
                .map(|(orbit, _)| orbit.satellite_id)
                .collect(),
            SsrKind::Ura => self.ura.iter().map(|&(id, _)| id).collect(),
            SsrKind::CodeBias => self.code_bias.iter().map(|r| r.satellite_id).collect(),
            SsrKind::PhaseBias => self.phase_bias.iter().map(|r| r.satellite_id).collect(),
            SsrKind::Vtec => Vec::new(),
        }
    }
}

/// Read up to `count` records with `read`. A body that ends inside a record
/// is refused as truncated under the strict policy; under the lenient policy
/// the complete records are kept, the reader is left after the last of them,
/// and an [`RtcmDeparture::SsrRecordsShort`] is recorded.
fn read_records<T>(
    r: &mut BitReader<'_>,
    ctx: &mut DecodeContext,
    message_number: u16,
    count: usize,
    read: &mut dyn FnMut(&mut BitReader<'_>) -> DecodeResult<T>,
) -> DecodeResult<Vec<T>> {
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        let mut trial = r.clone();
        match read(&mut trial) {
            Ok(record) => {
                *r = trial;
                out.push(record);
            }
            Err(DecodeError::OutOfInput(error)) if ctx.policy() == RtcmPolicy::Strict => {
                return Err(DecodeError::OutOfInput(error));
            }
            Err(DecodeError::OutOfInput(_)) => {
                ctx.depart(RtcmDeparture::SsrRecordsShort {
                    message_number,
                    declared: count,
                    read: index,
                })?;
                break;
            }
            Err(error) => return Err(error),
        }
    }
    Ok(out)
}

fn read_header(
    r: &mut BitReader<'_>,
    system: GnssSystem,
    kind: SsrKind,
) -> DecodeResult<SsrHeader> {
    let epoch_time_s = r.u(epoch_time_bits(system))? as u32;
    let update_interval = r.u(4)? as u8;
    let multiple_message = r.flag()?;
    let satellite_reference_datum = if matches!(kind, SsrKind::Orbit | SsrKind::CombinedOrbitClock)
    {
        Some(r.flag()?)
    } else {
        None
    };
    let iod_ssr = r.u(4)? as u8;
    let provider_id = r.u(16)? as u16;
    let solution_id = r.u(4)? as u8;
    let dispersive_bias_consistency = if kind == SsrKind::PhaseBias {
        Some(r.flag()?)
    } else {
        None
    };
    let mw_consistency = if kind == SsrKind::PhaseBias {
        Some(r.flag()?)
    } else {
        None
    };
    let satellite_count = r.u(SATELLITE_COUNT_BITS)? as u8;
    Ok(SsrHeader {
        epoch_time_s,
        update_interval,
        multiple_message,
        iod_ssr,
        provider_id,
        solution_id,
        satellite_reference_datum,
        dispersive_bias_consistency,
        mw_consistency,
        satellite_count,
    })
}

fn write_header(
    w: &mut FieldWriter,
    system: GnssSystem,
    header: &SsrHeader,
    kind: SsrKind,
) -> Result<()> {
    w.u(
        "epoch time",
        u64::from(header.epoch_time_s),
        epoch_time_bits(system),
    )?;
    w.u("update interval", u64::from(header.update_interval), 4)?;
    w.flag(header.multiple_message);
    if let Some(datum) = header.satellite_reference_datum {
        w.flag(datum);
    }
    w.u("IOD SSR", u64::from(header.iod_ssr), 4)?;
    w.u("provider ID", u64::from(header.provider_id), 16)?;
    w.u("solution ID", u64::from(header.solution_id), 4)?;
    if kind == SsrKind::PhaseBias {
        // `check_header_flags` has refused a phase-bias header without them.
        w.flag(header.dispersive_bias_consistency.unwrap_or_default());
        w.flag(header.mw_consistency.unwrap_or_default());
    }
    w.u(
        "satellite count",
        u64::from(header.satellite_count),
        SATELLITE_COUNT_BITS,
    )
}

fn read_orbit_record(
    r: &mut BitReader<'_>,
    system: GnssSystem,
    sat_bits: usize,
) -> DecodeResult<SsrOrbitRecord> {
    Ok(SsrOrbitRecord {
        satellite_id: r.u(sat_bits)? as u8,
        iode: r.u(iode_bits(system))? as u32,
        delta_radial: r.i(22)? as i32,
        delta_along: r.i(20)? as i32,
        delta_cross: r.i(20)? as i32,
        dot_delta_radial: r.i(21)? as i32,
        dot_delta_along: r.i(19)? as i32,
        dot_delta_cross: r.i(19)? as i32,
    })
}

fn write_orbit_record(
    w: &mut FieldWriter,
    system: GnssSystem,
    sat_bits: usize,
    rec: &SsrOrbitRecord,
) -> Result<()> {
    let id = rec.satellite_id;
    w.u("satellite id", u64::from(id), sat_bits)?;
    w.u(
        format_args!("satellite {id} IODE"),
        u64::from(rec.iode),
        iode_bits(system),
    )?;
    w.i(
        format_args!("satellite {id} radial delta"),
        i64::from(rec.delta_radial),
        22,
    )?;
    w.i(
        format_args!("satellite {id} along-track delta"),
        i64::from(rec.delta_along),
        20,
    )?;
    w.i(
        format_args!("satellite {id} cross-track delta"),
        i64::from(rec.delta_cross),
        20,
    )?;
    w.i(
        format_args!("satellite {id} radial delta rate"),
        i64::from(rec.dot_delta_radial),
        21,
    )?;
    w.i(
        format_args!("satellite {id} along-track delta rate"),
        i64::from(rec.dot_delta_along),
        19,
    )?;
    w.i(
        format_args!("satellite {id} cross-track delta rate"),
        i64::from(rec.dot_delta_cross),
        19,
    )
}

fn read_clock_record(r: &mut BitReader<'_>, sat_bits: usize) -> DecodeResult<SsrClockRecord> {
    Ok(SsrClockRecord {
        satellite_id: r.u(sat_bits)? as u8,
        c0: r.i(22)? as i32,
        c1: r.i(21)? as i32,
        c2: r.i(27)? as i32,
    })
}

fn write_clock_record(w: &mut FieldWriter, sat_bits: usize, rec: &SsrClockRecord) -> Result<()> {
    w.u("satellite id", u64::from(rec.satellite_id), sat_bits)?;
    write_clock_terms(w, rec)
}

fn write_clock_terms(w: &mut FieldWriter, rec: &SsrClockRecord) -> Result<()> {
    let id = rec.satellite_id;
    w.i(
        format_args!("satellite {id} clock C0"),
        i64::from(rec.c0),
        22,
    )?;
    w.i(
        format_args!("satellite {id} clock C1"),
        i64::from(rec.c1),
        21,
    )?;
    w.i(
        format_args!("satellite {id} clock C2"),
        i64::from(rec.c2),
        27,
    )
}

fn read_code_bias_record(
    r: &mut BitReader<'_>,
    sat_bits: usize,
) -> DecodeResult<SsrCodeBiasRecord> {
    let satellite_id = r.u(sat_bits)? as u8;
    let count = r.u(5)? as usize;
    let mut biases = Vec::with_capacity(count);
    for _ in 0..count {
        let signal_id = r.u(5)? as u8;
        let bias = r.i(14)? as i16;
        biases.push((signal_id, bias));
    }
    Ok(SsrCodeBiasRecord {
        satellite_id,
        biases,
    })
}

fn write_code_bias_record(
    w: &mut FieldWriter,
    sat_bits: usize,
    rec: &SsrCodeBiasRecord,
) -> Result<()> {
    let id = rec.satellite_id;
    w.u("satellite id", u64::from(id), sat_bits)?;
    w.u(
        format_args!("satellite {id} code-bias count"),
        rec.biases.len() as u64,
        5,
    )?;
    for &(signal_id, bias) in &rec.biases {
        w.u(
            format_args!("satellite {id} code-bias signal ID"),
            u64::from(signal_id),
            5,
        )?;
        w.i(
            format_args!("satellite {id} signal {signal_id} code bias"),
            i64::from(bias),
            14,
        )?;
    }
    Ok(())
}

/// Read one phase-bias record. Each signal is signal ID (5 bits), integer
/// indicator (1), wide-lane integer indicator (2), discontinuity counter (4)
/// and phase bias (20), as RTCM 10403.3 defines it for 1265..1270, with no
/// standard-deviation field. RTKLIB does not decode 1265..1270; the 17-bit
/// phase-bias standard deviation its `decode_ssr7` reads after each bias
/// belongs to its tentative message numbers 11..14, not to these messages.
fn read_phase_bias_record(
    r: &mut BitReader<'_>,
    sat_bits: usize,
) -> DecodeResult<SsrPhaseBiasRecord> {
    let satellite_id = r.u(sat_bits)? as u8;
    let count = r.u(5)? as usize;
    let yaw_angle = r.u(9)? as u16;
    let yaw_rate = r.i(8)? as i8;
    let mut biases = Vec::with_capacity(count);
    for _ in 0..count {
        biases.push(SsrPhaseBiasSignal {
            signal_id: r.u(5)? as u8,
            integer_indicator: r.u(1)? as u8,
            wide_lane_integer_indicator: r.u(2)? as u8,
            discontinuity_counter: r.u(4)? as u8,
            bias: r.i(20)? as i32,
        });
    }
    Ok(SsrPhaseBiasRecord {
        satellite_id,
        yaw_angle,
        yaw_rate,
        biases,
    })
}

fn write_phase_bias_record(
    w: &mut FieldWriter,
    sat_bits: usize,
    rec: &SsrPhaseBiasRecord,
) -> Result<()> {
    let id = rec.satellite_id;
    w.u("satellite id", u64::from(id), sat_bits)?;
    w.u(
        format_args!("satellite {id} phase-bias count"),
        rec.biases.len() as u64,
        5,
    )?;
    w.u(
        format_args!("satellite {id} yaw angle"),
        u64::from(rec.yaw_angle),
        9,
    )?;
    w.i(
        format_args!("satellite {id} yaw rate"),
        i64::from(rec.yaw_rate),
        8,
    )?;
    for bias in &rec.biases {
        let signal = bias.signal_id;
        w.u(
            format_args!("satellite {id} phase-bias signal ID"),
            u64::from(signal),
            5,
        )?;
        w.u(
            format_args!("satellite {id} signal {signal} integer indicator"),
            u64::from(bias.integer_indicator),
            1,
        )?;
        w.u(
            format_args!("satellite {id} signal {signal} wide-lane integer indicator"),
            u64::from(bias.wide_lane_integer_indicator),
            2,
        )?;
        w.u(
            format_args!("satellite {id} signal {signal} discontinuity counter"),
            u64::from(bias.discontinuity_counter),
            4,
        )?;
        w.i(
            format_args!("satellite {id} signal {signal} phase bias"),
            i64::from(bias.bias),
            20,
        )?;
    }
    Ok(())
}

/// Width of an SSR record's satellite field: five bits for GLONASS, four for
/// the native QZSS messages (1246..1251 and the 1268 phase bias), six
/// otherwise - the widths RTKLIB `decode_ssr1`..`decode_ssr7` read.
fn satellite_id_bits(system: GnssSystem, message_number: u16) -> usize {
    match system {
        GnssSystem::Glonass => 5,
        GnssSystem::Qzss if is_native_qzss_ssr(message_number) => 4,
        _ => 6,
    }
}

/// Width of the SSR header's satellite count (DF387): six bits for every
/// message, the uint6 RTCM 10403.3 defines and BNC reads. RTKLIB
/// `decode_ssr1_head`, `decode_ssr2_head`, `decode_ssr7_head` and
/// `encode_ssr_head` read and write four bits for QZSS, from a draft layout, and
/// RTKLIB does not decode 1268; the standard's width is kept here. The QZSS
/// satellite ID itself (DF430) is four bits; see [`satellite_id_bits`].
const SATELLITE_COUNT_BITS: usize = 6;

/// Whether a message number is a native RTCM QZSS SSR message, whose satellite
/// field is four bits: 1246..1251 and the 1268 phase bias.
pub(crate) fn is_native_qzss_ssr(message_number: u16) -> bool {
    (1246..=1251).contains(&message_number) || message_number == 1268
}

fn iode_bits(system: GnssSystem) -> usize {
    match system {
        GnssSystem::Galileo => 10,
        GnssSystem::BeiDou => 18,
        _ => 8,
    }
}

fn epoch_time_bits(system: GnssSystem) -> usize {
    match system {
        GnssSystem::Glonass => 17,
        _ => 20,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::rtcm::bits::BitWriter;
    use crate::rtcm::{
        decode_frame, encode_frame, Message, SsrStreamAssembler, UnsupportedMessage,
    };

    const REAL_SSRA02IGS0_1243_FRAME_HEX: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/ssr/SSRA02IGS0_2026181234930_1243.hex"
    ));
    const REAL_SSRA02IGS0_1060_FRAME_HEX: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/ssr/SSRA02IGS0_2026181234930_1060.hex"
    ));

    type RtklibCombinedRecord = (u8, u32, i32, i32, i32, i32, i32, i32, i32, i32, i32);

    const RTKLIB_GALILEO_1243: &[RtklibCombinedRecord] = &[
        (2, 65, 1010, 274, -80, -46, -28, 7, 1426, 0, 0),
        (3, 64, -714, 92, -83, 101, -29, 10, 1467, 0, 0),
        (4, 63, 2270, -273, -570, 62, -10, -10, -1957, 0, 0),
        (5, 65, 598, -257, -32, 85, -31, 4, -334, 0, 0),
        (6, 63, 3510, -770, -997, 44, 11, 3, -4312, 0, 0),
        (7, 61, -523, -420, 424, 8, -30, -14, 2136, 0, 0),
        (8, 65, -678, -462, 147, 26, -20, 6, 4289, 0, 0),
        (9, 65, 4049, -350, -709, 53, -25, 32, -2437, 0, 0),
        (10, 61, 2796, -279, 104, -5, -14, -22, -2916, 0, 0),
        (11, 63, 5304, -453, 225, -5, -23, -16, 4, 0, 0),
        (12, 65, -150, 129, -165, -5, -22, 5, 2686, 0, 0),
        (13, 65, -1364, -594, 186, 34, -39, -7, 1752, 0, 0),
        (15, 63, 1526, -1182, -594, 48, -15, 23, -129, 0, 0),
        (16, 63, 1103, 153, -549, -18, -22, 15, -3064, 0, 0),
        (19, 63, 1957, 1032, 379, -40, 35, 2, -3568, 0, 0),
        (21, 65, -2238, 369, -208, 12, -38, 3, 3171, 0, 0),
        (23, 65, 1153, 535, -516, -49, -22, 23, -2598, 0, 0),
        (25, 65, 98, 733, -726, -25, -15, 0, 581, 0, 0),
        (26, 64, -822, -146, 190, 23, -9, -20, 2149, 0, 0),
        (27, 64, 343, -1258, -237, 32, -24, -9, 220, 0, 0),
        (28, 65, 2459, -256, -275, -53, -16, 8, -1086, 0, 0),
        (29, 65, 1202, 228, -407, 0, -12, -16, -77, 0, 0),
        (30, 65, 1485, 157, 415, -53, -12, 0, 566, 0, 0),
        (31, 65, -563, 616, 1, -30, 4, -10, 151, 0, 0),
        (33, 65, 630, -60, 258, -87, -5, -6, 1554, 0, 0),
        (34, 58, -471, -690, -100, 20, -26, -20, 1790, 0, 0),
        (36, 49, 1519, 292, 670, -54, -15, 16, 694, 0, 0),
    ];
    const RTKLIB_GPS_1060: &[RtklibCombinedRecord] = &[
        (30, 90, 807, 621, -349, 30, -10, -8, 166, 0, 0),
        (31, 67, -227, -1752, 1423, -43, -7, 3, 4170, 0, 0),
    ];

    fn header(system: GnssSystem, kind: SsrKind, count: u8) -> SsrHeader {
        SsrHeader {
            epoch_time_s: if system == GnssSystem::Glonass {
                61_632
            } else {
                345_600
            },
            update_interval: 2,
            multiple_message: true,
            iod_ssr: 9,
            provider_id: 123,
            solution_id: 4,
            satellite_reference_datum: matches!(kind, SsrKind::Orbit | SsrKind::CombinedOrbitClock)
                .then_some(false),
            dispersive_bias_consistency: (kind == SsrKind::PhaseBias).then_some(true),
            mw_consistency: (kind == SsrKind::PhaseBias).then_some(false),
            satellite_count: count,
        }
    }

    fn orbit_record(system: GnssSystem) -> SsrOrbitRecord {
        SsrOrbitRecord {
            satellite_id: 3,
            iode: match system {
                GnssSystem::Galileo => 513,
                GnssSystem::BeiDou => 123_456,
                _ => 42,
            },
            delta_radial: -12_345,
            delta_along: 23_456,
            delta_cross: -34_567,
            dot_delta_radial: 456,
            dot_delta_along: -567,
            dot_delta_cross: 678,
        }
    }

    fn clock_record() -> SsrClockRecord {
        SsrClockRecord {
            satellite_id: 3,
            c0: -78_901,
            c1: 89_012,
            c2: -9_012_345,
        }
    }

    fn message(message_number: u16, system: GnssSystem, kind: SsrKind) -> SsrMessage {
        let mut orbit = Vec::new();
        let mut clock = Vec::new();
        let mut ura = Vec::new();
        let mut phase_bias = Vec::new();
        match kind {
            SsrKind::Orbit => orbit.push(orbit_record(system)),
            SsrKind::Clock => clock.push(clock_record()),
            SsrKind::CombinedOrbitClock => {
                orbit.push(orbit_record(system));
                clock.push(clock_record());
            }
            SsrKind::Ura => ura.push((3, 41)),
            SsrKind::HighRateClock => clock.push(SsrClockRecord {
                satellite_id: 3,
                c0: -22_222,
                c1: 0,
                c2: 0,
            }),
            SsrKind::CodeBias => {
                // RTCM SSR code bias: 5-bit signal id, int14 bias at 0.01 m.
            }
            SsrKind::PhaseBias => phase_bias.push(SsrPhaseBiasRecord {
                satellite_id: 3,
                yaw_angle: 127,
                yaw_rate: -12,
                biases: vec![
                    SsrPhaseBiasSignal {
                        signal_id: 1,
                        integer_indicator: 1,
                        wide_lane_integer_indicator: 2,
                        discontinuity_counter: 3,
                        bias: -123_456,
                    },
                    SsrPhaseBiasSignal {
                        signal_id: 9,
                        integer_indicator: 0,
                        wide_lane_integer_indicator: 1,
                        discontinuity_counter: 4,
                        bias: 234_567,
                    },
                ],
            }),
            SsrKind::Vtec => {}
        }
        let code_bias = if kind == SsrKind::CodeBias {
            vec![SsrCodeBiasRecord {
                satellite_id: 3,
                biases: vec![(1, -1234), (9, 2345)],
            }]
        } else {
            Vec::new()
        };
        SsrMessage {
            message_number,
            system,
            kind,
            header: header(system, kind, 1),
            orbit,
            clock,
            code_bias,
            phase_bias,
            ura,
            padding_bits: Vec::new(),
        }
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

    /// The native QZSS messages carry a four-bit satellite field: 15 is the
    /// widest native value and round-trips; 16 is refused rather than written
    /// as its low four bits.
    #[test]
    fn qzss_satellite_field_width_follows_the_message_number() {
        let mut native = message(1246, GnssSystem::Qzss, SsrKind::Orbit);
        native.orbit[0].satellite_id = 15;
        let body = native.encode().expect("J15 fits the native four-bit field");
        let decoded = SsrMessage::decode(&body).expect("decode native QZSS orbit");
        // The body is padded with zero bits to a whole byte, and the decoder
        // keeps that padding; every other field comes back as built, and the
        // decoded message encodes to the same bytes.
        assert!(decoded.padding_bits.iter().all(|bit| !bit));
        let mut expected = native.clone();
        expected.padding_bits = decoded.padding_bits.clone();
        assert_eq!(decoded, expected);
        assert_eq!(decoded.encode().unwrap(), body);
        native.orbit[0].satellite_id = 16;
        let err = native
            .encode()
            .expect_err("16 does not fit the native four-bit field");
        assert!(err.to_string().contains("4-bit"), "{err}");
    }

    /// The native QZSS SSR header states its satellite count in the six bits
    /// RTCM 10403.3 gives DF387 for every system, followed by the four-bit
    /// QZSS satellite ID (DF430). RTKLIB reads a four-bit count for QZSS from a
    /// draft layout; the standard's width is written and read here.
    #[test]
    fn native_qzss_ssr_satellite_count_is_six_bits_and_satellite_id_four() {
        let mut orbit = message(1246, GnssSystem::Qzss, SsrKind::Orbit);
        orbit.orbit[0].satellite_id = 7;
        let body = orbit.encode().unwrap();
        let mut r = BitReader::new(&body);
        assert_eq!(r.u(12).unwrap(), 1246);
        // Orbit header: epoch 20, update interval 4, multiple message 1, datum
        // 1, IOD SSR 4, provider 16, solution 4, then the six-bit count.
        r.u(20 + 4 + 1 + 1 + 4 + 16 + 4).unwrap();
        assert_eq!(r.u(6).unwrap(), 1, "satellite count");
        assert_eq!(r.u(4).unwrap(), 7, "first satellite field");
        assert_eq!(SsrMessage::decode(&body).unwrap().orbit, orbit.orbit);

        // The phase-bias header puts its two consistency flags before the
        // count.
        let phase = message(1268, GnssSystem::Qzss, SsrKind::PhaseBias);
        let body = phase.encode().unwrap();
        let mut r = BitReader::new(&body);
        r.u(12 + 20 + 4 + 1 + 4 + 16 + 4 + 1 + 1).unwrap();
        assert_eq!(r.u(6).unwrap(), 1, "satellite count");
        assert_eq!(r.u(4).unwrap(), 3, "first satellite field");

        // Sixty-three records fit the count; sixty-four are refused, not
        // written as the count's low bits.
        let mut many = message(1247, GnssSystem::Qzss, SsrKind::Clock);
        many.clock = (0..63)
            .map(|index| SsrClockRecord {
                satellite_id: index % 15 + 1,
                ..clock_record()
            })
            .collect();
        many.header.satellite_count = 63;
        let body = many.encode().expect("sixty-three records fit the count");
        assert_eq!(SsrMessage::decode(&body).unwrap().clock.len(), 63);
        many.clock.push(clock_record());
        many.header.satellite_count = 64;
        let err = many.encode().expect_err("sixty-four do not fit six bits");
        assert!(err.to_string().contains("satellite count 64"), "{err}");
    }

    /// A body that ends before the records its header counts is refused as
    /// truncated under the strict policy. Under the lenient policy the complete
    /// records are read, as RTKLIB `decode_ssr2` reads them, the header count
    /// and the bits of the cut record are kept, the departure names both
    /// counts, and the lenient encoder writes the body back as read.
    #[test]
    fn short_ssr_body_is_refused_strictly_and_read_leniently() {
        let mut full = message(1058, GnssSystem::Gps, SsrKind::Clock);
        full.clock.push(SsrClockRecord {
            satellite_id: 33,
            ..clock_record()
        });
        full.header.satellite_count = 2;
        let body = full.encode().unwrap();
        // Header 67 bits, then two 76-bit records: the first ends at bit 143.
        assert_eq!(body.len(), 28);
        let short = body[..18].to_vec();

        let err = SsrMessage::decode(&short).expect_err("strict refuses");
        assert!(err.to_string().contains("truncated"), "{err}");
        let frame = encode_frame(&short).unwrap();
        assert_eq!(
            crate::rtcm::decode_stream(&frame)
                .diagnostics
                .skipped_frames[0]
                .reason,
            crate::rtcm::FrameSkipReason::Truncated
        );

        let departure = RtcmDeparture::SsrRecordsShort {
            message_number: 1058,
            declared: 2,
            read: 1,
        };
        let (read, departures) =
            SsrMessage::decode_with_policy(&short, RtcmPolicy::Lenient).unwrap();
        assert_eq!(departures, vec![departure.clone()]);
        assert_eq!(read.header.satellite_count, 2);
        assert_eq!(read.clock, vec![clock_record()]);
        // Satellite 33 is 0b100001: its first bit is the one bit left over.
        assert_eq!(read.padding_bits, vec![true]);

        let err = read.encode().expect_err("strict encode refuses the count");
        assert!(
            err.to_string().contains("header satellite count 2"),
            "{err}"
        );
        let (written, departures) = read.encode_with_policy(RtcmPolicy::Lenient).unwrap();
        assert_eq!(written, short);
        assert_eq!(departures, vec![departure.clone()]);

        let stream = crate::rtcm::decode_stream_with_policy(&frame, RtcmPolicy::Lenient);
        assert_eq!(stream.messages, vec![Message::Ssr(read)]);
        assert_eq!(
            stream.diagnostics.departures,
            vec![crate::rtcm::StreamDeparture {
                offset: 0,
                departure,
            }]
        );

        // More records than the count states is refused under both policies.
        let mut over = full.clone();
        over.header.satellite_count = 1;
        assert!(over.encode_with_policy(RtcmPolicy::Lenient).is_err());

        // A combined orbit/clock body cut inside its second record reads the
        // first pair.
        let mut combined = message(1060, GnssSystem::Gps, SsrKind::CombinedOrbitClock);
        combined.orbit.push(SsrOrbitRecord {
            satellite_id: 4,
            ..orbit_record(GnssSystem::Gps)
        });
        combined.clock.push(SsrClockRecord {
            satellite_id: 4,
            ..clock_record()
        });
        combined.header.satellite_count = 2;
        let body = combined.encode().unwrap();
        let (read, departures) =
            SsrMessage::decode_with_policy(&body[..body.len() - 5], RtcmPolicy::Lenient).unwrap();
        assert_eq!(read.orbit.len(), 1);
        assert_eq!(read.clock.len(), 1);
        assert_eq!(
            departures,
            vec![RtcmDeparture::SsrRecordsShort {
                message_number: 1060,
                declared: 2,
                read: 1,
            }]
        );
    }

    /// The encoder writes the RTCM SSR layout for the message numbers this
    /// codec decodes and nothing else. 4076 (IGS SSR) carries a version and a
    /// subtype after its number; writing the RTCM layout under it would give
    /// bytes no reader takes for the fields held, so it is refused, as is a
    /// number whose system or group differs from the record's.
    #[test]
    fn ssr_encode_refuses_a_number_that_names_another_layout() {
        let igs = message(4076, GnssSystem::Qzss, SsrKind::Orbit);
        let err = igs.encode().expect_err("4076 is not an RTCM SSR layout");
        assert!(err.to_string().contains("4076"), "{err}");
        let mut crossed = message(1057, GnssSystem::Gps, SsrKind::Orbit);
        crossed.kind = SsrKind::Clock;
        crossed.clock = crossed
            .orbit
            .drain(..)
            .map(|rec| SsrClockRecord {
                satellite_id: rec.satellite_id,
                ..clock_record()
            })
            .collect();
        crossed.header.satellite_reference_datum = None;
        assert!(crossed.encode().is_err(), "1057 is the GPS orbit layout");
    }

    /// Every SSR field is written in its own width or refused by name: the
    /// header fields, each record field, the header flags and the record lists
    /// the group does not write.
    #[test]
    fn ssr_encode_refuses_values_it_would_truncate_fill_or_drop() {
        let base = message(1060, GnssSystem::Gps, SsrKind::CombinedOrbitClock);
        base.encode().expect("the base message encodes");
        let refused = |edit: &dyn Fn(&mut SsrMessage), needle: &str| {
            let mut m = base.clone();
            edit(&mut m);
            let err = m.encode().expect_err(needle);
            assert!(
                matches!(err, Error::RtcmEncode(ref e) if e.to_string().contains(needle)),
                "expected {needle:?}, got {err}"
            );
        };
        refused(&|m| m.header.epoch_time_s = 1 << 20, "epoch time 1048576");
        refused(&|m| m.header.update_interval = 16, "update interval 16");
        refused(&|m| m.header.iod_ssr = 16, "IOD SSR 16");
        refused(&|m| m.header.solution_id = 16, "solution ID 16");
        refused(
            &|m| m.header.satellite_count = 2,
            "header satellite count 2",
        );
        refused(
            &|m| m.header.satellite_reference_datum = None,
            "satellite reference datum",
        );
        refused(
            &|m| m.header.mw_consistency = Some(false),
            "MW consistency indicator",
        );
        refused(&|m| m.orbit[0].iode = 256, "IODE 256");
        refused(
            &|m| m.orbit[0].delta_radial = 1 << 21,
            "radial delta 2097152",
        );
        refused(
            &|m| m.orbit[0].dot_delta_cross = -(1 << 18) - 1,
            "cross-track delta rate",
        );
        refused(&|m| m.clock[0].c2 = 1 << 26, "clock C2 67108864");
        refused(&|m| m.ura.push((3, 1)), "writes no URA records");

        // A GLONASS epoch is the 17-bit time of day.
        let mut glonass = message(1063, GnssSystem::Glonass, SsrKind::Orbit);
        glonass.header.epoch_time_s = 1 << 17;
        assert!(glonass.encode().is_err());

        // A high-rate clock carries only C0.
        let mut high_rate = message(1062, GnssSystem::Gps, SsrKind::HighRateClock);
        high_rate.clock[0].c1 = 5;
        let err = high_rate.encode().expect_err("c1 is not carried");
        assert!(err.to_string().contains("carries only c0"), "{err}");

        // Bias counts are five bits and bias values fourteen and twenty.
        let mut code = message(1059, GnssSystem::Gps, SsrKind::CodeBias);
        code.code_bias[0].biases = (0..32).map(|signal| (signal % 32, 1)).collect();
        let err = code.encode().expect_err("32 biases do not fit five bits");
        assert!(err.to_string().contains("code-bias count 32"), "{err}");
        let mut code = message(1059, GnssSystem::Gps, SsrKind::CodeBias);
        code.code_bias[0].biases[0].1 = 1 << 13;
        assert!(code.encode().is_err(), "code bias past int14");
        let mut phase = message(1265, GnssSystem::Gps, SsrKind::PhaseBias);
        phase.phase_bias[0].biases[0].bias = 1 << 19;
        assert!(phase.encode().is_err(), "phase bias past int20");
        let mut phase = message(1265, GnssSystem::Gps, SsrKind::PhaseBias);
        phase.phase_bias[0].yaw_angle = 512;
        assert!(phase.encode().is_err(), "yaw angle past nine bits");
        let mut phase = message(1265, GnssSystem::Gps, SsrKind::PhaseBias);
        phase.phase_bias[0].biases[0].integer_indicator = 2;
        assert!(phase.encode().is_err(), "integer indicator past one bit");
        let mut phase = message(1265, GnssSystem::Gps, SsrKind::PhaseBias);
        phase.header.dispersive_bias_consistency = None;
        assert!(phase.encode().is_err(), "phase-bias header flag absent");
    }

    #[test]
    fn phase_a_messages_decode_fields_and_roundtrip() {
        for (number, system, kind) in [
            (1057, GnssSystem::Gps, SsrKind::Orbit),
            (1058, GnssSystem::Gps, SsrKind::Clock),
            (1059, GnssSystem::Gps, SsrKind::CodeBias),
            (1060, GnssSystem::Gps, SsrKind::CombinedOrbitClock),
            (1061, GnssSystem::Gps, SsrKind::Ura),
            (1062, GnssSystem::Gps, SsrKind::HighRateClock),
            (1265, GnssSystem::Gps, SsrKind::PhaseBias),
            (1063, GnssSystem::Glonass, SsrKind::Orbit),
            (1064, GnssSystem::Glonass, SsrKind::Clock),
            (1065, GnssSystem::Glonass, SsrKind::CodeBias),
            (1066, GnssSystem::Glonass, SsrKind::CombinedOrbitClock),
            (1067, GnssSystem::Glonass, SsrKind::Ura),
            (1068, GnssSystem::Glonass, SsrKind::HighRateClock),
            (1240, GnssSystem::Galileo, SsrKind::Orbit),
            (1241, GnssSystem::Galileo, SsrKind::Clock),
            (1242, GnssSystem::Galileo, SsrKind::CodeBias),
            (1243, GnssSystem::Galileo, SsrKind::CombinedOrbitClock),
            (1244, GnssSystem::Galileo, SsrKind::Ura),
            (1245, GnssSystem::Galileo, SsrKind::HighRateClock),
            (1267, GnssSystem::Galileo, SsrKind::PhaseBias),
            (1258, GnssSystem::BeiDou, SsrKind::Orbit),
            (1259, GnssSystem::BeiDou, SsrKind::Clock),
            (1260, GnssSystem::BeiDou, SsrKind::CodeBias),
            (1261, GnssSystem::BeiDou, SsrKind::CombinedOrbitClock),
            (1262, GnssSystem::BeiDou, SsrKind::Ura),
            (1263, GnssSystem::BeiDou, SsrKind::HighRateClock),
            (1270, GnssSystem::BeiDou, SsrKind::PhaseBias),
            (1246, GnssSystem::Qzss, SsrKind::Orbit),
            (1247, GnssSystem::Qzss, SsrKind::Clock),
            (1248, GnssSystem::Qzss, SsrKind::CodeBias),
            (1249, GnssSystem::Qzss, SsrKind::CombinedOrbitClock),
            (1250, GnssSystem::Qzss, SsrKind::Ura),
            (1251, GnssSystem::Qzss, SsrKind::HighRateClock),
            (1268, GnssSystem::Qzss, SsrKind::PhaseBias),
        ] {
            let expected = message(number, system, kind);
            let body = expected.encode().unwrap();
            let decoded = SsrMessage::decode(&body).unwrap();
            assert_eq!(
                decoded.message_number, expected.message_number,
                "message {number}"
            );
            assert_eq!(decoded.system, expected.system, "message {number}");
            assert_eq!(decoded.kind, expected.kind, "message {number}");
            assert_eq!(decoded.header, expected.header, "message {number}");
            assert_eq!(decoded.orbit, expected.orbit, "message {number}");
            assert_eq!(decoded.clock, expected.clock, "message {number}");
            assert_eq!(decoded.code_bias, expected.code_bias, "message {number}");
            assert_eq!(decoded.phase_bias, expected.phase_bias, "message {number}");
            assert_eq!(decoded.ura, expected.ura, "message {number}");
            assert_eq!(
                decoded.encode().unwrap(),
                body,
                "message {number} round trip"
            );
            assert!(matches!(Message::decode(&body).unwrap(), Message::Ssr(_)));
        }
    }

    #[test]
    fn real_ssr_apc_frames_match_rtklib_decode_oracle_and_roundtrip() {
        let gal_frame = hex_bytes(REAL_SSRA02IGS0_1243_FRAME_HEX);
        let gps_frame = hex_bytes(REAL_SSRA02IGS0_1060_FRAME_HEX);
        let mut stream = gal_frame.clone();
        stream.extend_from_slice(&gps_frame);
        let mut assembler = SsrStreamAssembler::new();
        let decoded = assembler.push(&stream);
        assert_eq!(decoded.len(), 2);

        let Message::Ssr(gal) = decoded[0].as_ref().unwrap() else {
            panic!("expected Galileo SSR");
        };
        assert_eq!(gal.message_number, 1243);
        assert_eq!(gal.system, GnssSystem::Galileo);
        assert_eq!(gal.kind, SsrKind::CombinedOrbitClock);
        assert_eq!(gal.header.epoch_time_s, 344_970);
        assert_eq!(gal.header.update_interval, 3);
        assert!(!gal.header.multiple_message);
        assert_eq!(gal.header.iod_ssr, 1);
        assert_eq!(gal.header.provider_id, 0);
        assert_eq!(gal.header.solution_id, 2);
        assert_eq!(gal.header.satellite_count, 27);
        assert_rtklib_combined_records(gal, RTKLIB_GALILEO_1243);
        assert_eq!(
            encode_frame(&Message::Ssr(gal.clone()).encode().unwrap()).unwrap(),
            gal_frame
        );

        let Message::Ssr(gps) = decoded[1].as_ref().unwrap() else {
            panic!("expected GPS SSR");
        };
        assert_eq!(gps.message_number, 1060);
        assert_eq!(gps.system, GnssSystem::Gps);
        assert_eq!(gps.kind, SsrKind::CombinedOrbitClock);
        assert_eq!(gps.header.epoch_time_s, 344_970);
        assert_eq!(gps.header.update_interval, 3);
        assert!(!gps.header.multiple_message);
        assert_eq!(gps.header.iod_ssr, 1);
        assert_eq!(gps.header.provider_id, 0);
        assert_eq!(gps.header.solution_id, 2);
        assert_eq!(gps.header.satellite_count, 2);
        assert_rtklib_combined_records(gps, RTKLIB_GPS_1060);
        assert_eq!(
            encode_frame(&Message::Ssr(gps.clone()).encode().unwrap()).unwrap(),
            gps_frame
        );
        assert_eq!(
            decode_frame(&gal_frame).unwrap().body,
            gal.encode().unwrap()
        );
        assert_eq!(
            decode_frame(&gps_frame).unwrap().body,
            gps.encode().unwrap()
        );
        assert_eq!(assembler.retained_len(), 0);
    }

    fn assert_rtklib_combined_records(message: &SsrMessage, expected: &[RtklibCombinedRecord]) {
        assert_eq!(message.orbit.len(), expected.len());
        assert_eq!(message.clock.len(), expected.len());
        for ((orbit, clock), expected) in message.orbit.iter().zip(&message.clock).zip(expected) {
            let (
                satellite_id,
                iode,
                delta_radial,
                delta_along,
                delta_cross,
                dot_delta_radial,
                dot_delta_along,
                dot_delta_cross,
                c0,
                c1,
                c2,
            ) = *expected;
            assert_eq!(orbit.satellite_id, satellite_id);
            assert_eq!(orbit.iode, iode, "sat {satellite_id}");
            assert_eq!(orbit.delta_radial, delta_radial, "sat {satellite_id}");
            assert_eq!(orbit.delta_along, delta_along, "sat {satellite_id}");
            assert_eq!(orbit.delta_cross, delta_cross, "sat {satellite_id}");
            assert_eq!(
                orbit.dot_delta_radial, dot_delta_radial,
                "sat {satellite_id}"
            );
            assert_eq!(orbit.dot_delta_along, dot_delta_along, "sat {satellite_id}");
            assert_eq!(orbit.dot_delta_cross, dot_delta_cross, "sat {satellite_id}");
            assert_eq!(clock.satellite_id, satellite_id);
            assert_eq!(clock.c0, c0, "sat {satellite_id}");
            assert_eq!(clock.c1, c1, "sat {satellite_id}");
            assert_eq!(clock.c2, c2, "sat {satellite_id}");
        }
    }

    #[test]
    fn truncated_supported_ssr_is_parse_error() {
        let body = message(1057, GnssSystem::Gps, SsrKind::Orbit)
            .encode()
            .unwrap();
        let err = SsrMessage::decode(&body[..body.len() - 1]).unwrap_err();
        assert!(matches!(err, Error::Parse(_)));
    }

    #[test]
    fn unsupported_ssr_bias_message_stays_unsupported() {
        let mut w = BitWriter::new();
        w.push_u(1266, 12);
        let body = w.into_bytes();
        let decoded = Message::decode(&body).unwrap();
        assert_eq!(
            decoded,
            Message::Unsupported(UnsupportedMessage {
                message_number: 1266,
                body
            })
        );
    }

    #[test]
    fn stream_assembler_keeps_trailing_partial_frame() {
        let a = Message::Ssr(message(1057, GnssSystem::Gps, SsrKind::Orbit))
            .to_frame()
            .unwrap();
        let b = Message::Ssr(message(1058, GnssSystem::Gps, SsrKind::Clock))
            .to_frame()
            .unwrap();
        let mut chunk = Vec::new();
        chunk.extend_from_slice(&[0, 1, 2]);
        chunk.extend_from_slice(&a);
        chunk.extend_from_slice(&b[..b.len() - 2]);

        let mut assembler = SsrStreamAssembler::new();
        let first = assembler.push(&chunk);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].as_ref().unwrap().message_number(), 1057);
        assert_eq!(assembler.retained_len(), b.len() - 2);

        let second = assembler.push(&b[b.len() - 2..]);
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].as_ref().unwrap().message_number(), 1058);
        assert_eq!(assembler.retained_len(), 0);
    }

    #[test]
    fn framed_ssr_roundtrips_through_message_decode() {
        let message = Message::Ssr(message(
            1243,
            GnssSystem::Galileo,
            SsrKind::CombinedOrbitClock,
        ));
        let frame = message.to_frame().unwrap();
        let mut assembler = SsrStreamAssembler::new();
        let decoded = assembler.push(&frame);
        assert_eq!(decoded.len(), 1);
        assert_eq!(
            decoded[0].as_ref().unwrap().encode().unwrap(),
            message.encode().unwrap()
        );
        assert_eq!(encode_frame(&message.encode().unwrap()).unwrap(), frame);
    }

    /// A combined orbit/clock frame writes each clock right after its orbit
    /// under the orbit's satellite field. A clock naming another satellite,
    /// or lists of different lengths, are refused by name rather than written
    /// under the wrong satellite or dropped.
    #[test]
    fn combined_orbit_clock_encode_refuses_unpaired_records() {
        let paired = message(1060, GnssSystem::Gps, SsrKind::CombinedOrbitClock);
        paired.encode().expect("matched pair encodes");

        let mut mismatched = paired.clone();
        mismatched.clock[0].satellite_id = 4;
        let err = mismatched
            .encode()
            .expect_err("clock satellite differs from orbit satellite");
        assert!(matches!(err, Error::RtcmEncode(_)), "{err}");
        assert!(
            err.to_string()
                .contains("record 0 names satellite id 3 for its orbit and 4 for its clock"),
            "{err}"
        );

        let mut extra_orbit = paired.clone();
        extra_orbit.orbit.push(SsrOrbitRecord {
            satellite_id: 5,
            ..orbit_record(GnssSystem::Gps)
        });
        extra_orbit.header.satellite_count = 2;
        let err = extra_orbit
            .encode()
            .expect_err("unequal orbit and clock record counts");
        assert!(
            err.to_string()
                .contains("2 orbit records and 1 clock records"),
            "{err}"
        );

        let mut extra_clock = paired;
        extra_clock.clock.push(SsrClockRecord {
            satellite_id: 5,
            ..clock_record()
        });
        let err = extra_clock
            .encode()
            .expect_err("unequal orbit and clock record counts");
        assert!(
            err.to_string()
                .contains("1 orbit records and 2 clock records"),
            "{err}"
        );
    }
}
