//! RTCM 3 legacy RTK observation messages: GPS 1001-1004 and GLONASS
//! 1009-1012.
//!
//! The four messages of each system carry one record per satellite (RTCM
//! 10403.3 Tables 3.5-2 to 3.5-15):
//!
//! | GPS  | GLONASS | L1 | L2 | Extended (ambiguity and CNR) |
//! |------|---------|----|----|------------------------------|
//! | 1001 | 1009    | x  |    |                              |
//! | 1002 | 1010    | x  |    | x                            |
//! | 1003 | 1011    | x  | x  |                              |
//! | 1004 | 1012    | x  | x  | x                            |
//!
//! The header states the number of satellite records (DF006, DF035). Every
//! field is stored as its raw transmitted integer, so a decode followed by an
//! encode reproduces the body bit for bit. RTKLIB decodes 1002, 1004, 1010 and
//! 1012 and reads only the header of 1001, 1003, 1009 and 1011.

use crate::error::{Error, Result};
use crate::id::GnssSystem;

use super::bits::{BitReader, FieldWriter, OutOfInput};
use super::{
    is_departing_tail, write_trailing, DecodeContext, DecodeError, DecodeResult, RtcmDeparture,
    RtcmEncodeError, RtcmPolicy, RtcmRecordKind,
};

/// DF012 / DF018 / DF042 / DF048 phase-range-minus-pseudorange invalid
/// value, `-2^19` (`0x80000`), as RTKLIB `decode_type1002`..`decode_type1012`
/// test it.
pub const LEGACY_PHASE_RANGE_INVALID: i32 = -(1 << 19);

/// DF017 / DF047 L2-minus-L1 pseudorange difference invalid value, `-2^13`
/// (`0x2000`), as RTKLIB `decode_type1004` and `decode_type1012` test it.
pub const LEGACY_PSEUDORANGE_DIFFERENCE_INVALID: i16 = -(1 << 13);

/// The layout a legacy observation message number names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Layout {
    system: GnssSystem,
    extended: bool,
    l2: bool,
}

impl Layout {
    fn of(message_number: u16) -> Option<Self> {
        let (system, index) = match message_number {
            1001..=1004 => (GnssSystem::Gps, message_number - 1001),
            1009..=1012 => (GnssSystem::Glonass, message_number - 1009),
            _ => return None,
        };
        Some(Self {
            system,
            extended: index % 2 == 1,
            l2: index >= 2,
        })
    }

    fn glonass(self) -> bool {
        self.system == GnssSystem::Glonass
    }

    /// Width of the epoch time: DF004 (GPS) or DF034 (GLONASS).
    fn epoch_bits(self) -> usize {
        if self.glonass() {
            27
        } else {
            30
        }
    }

    /// Width of the L1 pseudorange: DF011 (GPS) or DF041 (GLONASS).
    fn pseudorange_bits(self) -> usize {
        if self.glonass() {
            25
        } else {
            24
        }
    }

    /// Width of the L1 pseudorange modulus ambiguity: DF014 (GPS) or DF044
    /// (GLONASS).
    fn ambiguity_bits(self) -> usize {
        if self.glonass() {
            7
        } else {
            8
        }
    }
}

/// Whether `message_number` is a legacy RTK observation message.
pub(crate) fn is_legacy_observation(message_number: u16) -> bool {
    Layout::of(message_number).is_some()
}

/// A decoded legacy GPS (1001-1004) or GLONASS (1009-1012) RTK observation
/// message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegacyObservations {
    /// 1001..=1004 or 1009..=1012.
    pub message_number: u16,
    /// Reference station ID (DF003).
    pub reference_station_id: u16,
    /// Epoch time: for GPS the time of week in milliseconds (DF004, 30 bits);
    /// for GLONASS the time of day in milliseconds of GLONASS time (DF034,
    /// 27 bits).
    pub epoch_time: u32,
    /// Synchronous GNSS flag (DF005): more observations of this epoch follow.
    pub synchronous_gnss: bool,
    /// Number of satellite records the header states (DF006, DF035), as
    /// transmitted.
    pub satellite_count: u8,
    /// Divergence-free smoothing indicator (DF007, DF036).
    pub divergence_free_smoothing: bool,
    /// Smoothing interval (DF008, DF037, 3 bits).
    pub smoothing_interval: u8,
    /// The satellite records, in transmitted order.
    pub satellites: Vec<LegacySatellite>,
    /// Every body bit after the last record, the zeros that align the body to
    /// a byte included, kept whenever those bits are anything other than fewer
    /// than eight zeros: read under [`RtcmPolicy::Lenient`] and written back
    /// after the last record by `encode_with_policy` under that policy, so the
    /// body re-encodes byte for byte. After a body read leniently that ends
    /// before the records its header counts
    /// ([`RtcmDeparture::RecordsShort`]), the bits of the incomplete record.
    /// Empty for every body read under [`RtcmPolicy::Strict`] and for a
    /// message built by hand; `encode` refuses a nonempty value.
    pub trailing_bits: Vec<bool>,
}

/// One satellite record of a legacy observation message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LegacySatellite {
    /// Satellite ID (DF009 for GPS, DF038 for GLONASS, 6 bits). GPS values
    /// 40..=63 name SBAS satellites, broadcast PRN `value + 80`, as RTKLIB
    /// `decode_type1002` and `decode_type1004` read them.
    pub satellite_id: u8,
    /// GLONASS satellite frequency channel number plus 7 (DF040, 5 bits);
    /// `None` for GPS.
    pub frequency_channel: Option<u8>,
    /// The L1 observables.
    pub l1: LegacyL1,
    /// The L2 observables, carried by 1003, 1004, 1011 and 1012; `None` in
    /// the L1-only messages.
    pub l2: Option<LegacyL2>,
}

/// The L1 observables of one legacy satellite record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LegacyL1 {
    /// L1 code indicator (DF010, DF039): C/A code (false) or P code (true).
    pub code_indicator: bool,
    /// L1 pseudorange modulo the ambiguity unit (DF011, 24 bits; DF041,
    /// 25 bits), scale 0.02 m.
    pub pseudorange: u32,
    /// L1 phase range minus L1 pseudorange (DF012, DF042, int20), scale
    /// 0.0005 m. [`LEGACY_PHASE_RANGE_INVALID`] marks it invalid.
    pub phase_range_minus_pseudorange: i32,
    /// L1 lock-time indicator (DF013, DF043, 7 bits).
    pub lock_time_indicator: u8,
    /// Integer L1 pseudorange modulus ambiguity (DF014, 8 bits, units of
    /// 299 792.458 m; DF044, 7 bits, units of 599 584.916 m), carried by the
    /// extended messages 1002, 1004, 1010 and 1012.
    pub pseudorange_modulus_ambiguity: Option<u8>,
    /// L1 carrier-to-noise ratio (DF015, DF045, 8 bits), scale 0.25 dB-Hz,
    /// carried by the extended messages.
    pub cnr: Option<u8>,
}

/// The L2 observables of one legacy satellite record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LegacyL2 {
    /// L2 code indicator (DF016, DF046, 2 bits).
    pub code_indicator: u8,
    /// L2 minus L1 pseudorange difference (DF017, DF047, int14), scale
    /// 0.02 m. [`LEGACY_PSEUDORANGE_DIFFERENCE_INVALID`] marks it invalid.
    pub pseudorange_difference: i16,
    /// L2 phase range minus L1 pseudorange (DF018, DF048, int20), scale
    /// 0.0005 m. [`LEGACY_PHASE_RANGE_INVALID`] marks it invalid.
    pub phase_range_minus_l1_pseudorange: i32,
    /// L2 lock-time indicator (DF019, DF049, 7 bits).
    pub lock_time_indicator: u8,
    /// L2 carrier-to-noise ratio (DF020, DF050, 8 bits), scale 0.25 dB-Hz,
    /// carried by 1004 and 1012.
    pub cnr: Option<u8>,
}

impl LegacyObservations {
    /// The constellation the message number names: GPS for 1001..=1004,
    /// GLONASS for 1009..=1012, `None` for any other number.
    pub fn system(&self) -> Option<GnssSystem> {
        Layout::of(self.message_number).map(|layout| layout.system)
    }

    /// Decode a legacy observation body (without the transport frame) under
    /// [`RtcmPolicy::Strict`]: a body that ends before the records its header
    /// counts is refused as truncated, and bits after the last record other
    /// than the zero byte alignment are refused.
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_inner(body, &mut DecodeContext::new(RtcmPolicy::Strict)).map_err(Into::into)
    }

    /// Decode a legacy observation body under `policy`, returning the
    /// departures read under [`RtcmPolicy::Lenient`].
    ///
    /// Under [`RtcmPolicy::Lenient`] a body that ends before the records its
    /// header counts is read as RTKLIB `decode_type1002`..`decode_type1012`
    /// read it: every complete record, with the header count kept as
    /// transmitted, the bits of the incomplete record kept in
    /// [`Self::trailing_bits`], and an [`RtcmDeparture::RecordsShort`] naming
    /// both counts.
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
        let layout = Layout::of(message_number).ok_or_else(|| {
            Error::Parse(format!(
                "message {message_number} is not a legacy observation message 1001-1004/1009-1012"
            ))
        })?;
        let reference_station_id = r.u(12)? as u16;
        let epoch_time = r.u(layout.epoch_bits())? as u32;
        let synchronous_gnss = r.flag()?;
        let satellite_count = r.u(5)? as u8;
        let divergence_free_smoothing = r.flag()?;
        let smoothing_interval = r.u(3)? as u8;

        let declared = usize::from(satellite_count);
        let mut satellites = Vec::with_capacity(declared);
        for index in 0..declared {
            let mut trial = r.clone();
            match read_satellite(&mut trial, layout) {
                Ok(satellite) => {
                    r = trial;
                    satellites.push(satellite);
                }
                Err(error) if ctx.policy() == RtcmPolicy::Strict => {
                    return Err(DecodeError::OutOfInput(error));
                }
                Err(_) => {
                    ctx.depart(RtcmDeparture::RecordsShort {
                        message_number,
                        declared,
                        read: index,
                    })?;
                    break;
                }
            }
        }
        let rest = r.rest();
        let trailing_bits = if satellites.len() < declared {
            // The bits of the cut record, reported above.
            rest
        } else if is_departing_tail(&rest) {
            ctx.depart(RtcmDeparture::TrailingBits {
                message_number,
                bits: rest.clone(),
            })?;
            rest
        } else {
            Vec::new()
        };
        Ok(Self {
            message_number,
            reference_station_id,
            epoch_time,
            synchronous_gnss,
            satellite_count,
            divergence_free_smoothing,
            smoothing_interval,
            satellites,
            trailing_bits,
        })
    }

    /// Encode this message back into a body (without the transport frame)
    /// under [`RtcmPolicy::Strict`].
    ///
    /// # Errors
    ///
    /// [`Error::RtcmEncode`] naming what cannot be written as the message's
    /// wire form states it:
    ///
    /// * a message number outside 1001..=1004 and 1009..=1012;
    /// * a part the message number's layout carries held as `None`, or one it
    ///   does not carry held as `Some`: the GLONASS frequency channel, the L1
    ///   ambiguity and CNR of the extended messages, the L2 observables of
    ///   1003, 1004, 1011 and 1012, and the L2 CNR of 1004 and 1012;
    /// * a header satellite count other than the number of records
    ///   ([`RtcmDeparture::RecordsShort`] when it is larger);
    /// * nonempty `trailing_bits` ([`RtcmDeparture::TrailingBits`]);
    /// * a value wider than its field.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this message under `policy`. Under [`RtcmPolicy::Lenient`] a
    /// header satellite count above the number of records is written as held
    /// and reported, with `trailing_bits` after the records, which re-encodes
    /// a message read leniently from a short body; nonempty `trailing_bits`
    /// after a complete set of records are written and reported. Every other
    /// refusal of [`Self::encode`] applies under both policies.
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let number = self.message_number;
        let layout = Layout::of(number).ok_or(RtcmEncodeError::MessageNumber {
            message_number: number,
            record: RtcmRecordKind::LegacyObservations,
        })?;
        self.check_parts(layout)?;

        let declared = usize::from(self.satellite_count);
        let records = self.satellites.len();
        let mut departures = Vec::new();
        let short = declared > records;
        if short && policy == RtcmPolicy::Lenient {
            departures.push(RtcmDeparture::RecordsShort {
                message_number: number,
                declared,
                read: records,
            });
        } else if declared != records {
            return Err(RtcmEncodeError::CountMismatch {
                message_number: number,
                field: "satellite record",
                expected: declared,
                actual: records,
            }
            .into());
        }

        let mut w = FieldWriter::new(number);
        w.u("message number", u64::from(number), 12)?;
        w.u(
            "reference station ID",
            u64::from(self.reference_station_id),
            12,
        )?;
        w.u(
            "epoch time",
            u64::from(self.epoch_time),
            layout.epoch_bits(),
        )?;
        w.flag(self.synchronous_gnss);
        w.u("satellite count", u64::from(self.satellite_count), 5)?;
        w.flag(self.divergence_free_smoothing);
        w.u("smoothing interval", u64::from(self.smoothing_interval), 3)?;
        for satellite in &self.satellites {
            write_satellite(&mut w, layout, satellite)?;
        }
        if short {
            // The cut record's bits, reported with the short count.
            for &bit in &self.trailing_bits {
                w.flag(bit);
            }
        } else {
            departures.extend(write_trailing(&mut w, &self.trailing_bits, policy)?);
        }
        Ok((w.into_bytes(), departures))
    }

    /// Refuse a part the layout carries held as `None`, or one it does not
    /// carry held as `Some`.
    fn check_parts(&self, layout: Layout) -> Result<()> {
        let number = self.message_number;
        let check =
            |satellite: u8, field: &'static str, present: bool, carried: bool| -> Result<()> {
                if present == carried {
                    return Ok(());
                }
                Err(RtcmEncodeError::SatelliteFieldPresence {
                    message_number: number,
                    record: RtcmRecordKind::LegacyObservations,
                    satellite,
                    field,
                    carried,
                }
                .into())
            };
        for s in &self.satellites {
            check(
                s.satellite_id,
                "GLONASS frequency channel",
                s.frequency_channel.is_some(),
                layout.glonass(),
            )?;
            check(
                s.satellite_id,
                "L1 pseudorange modulus ambiguity",
                s.l1.pseudorange_modulus_ambiguity.is_some(),
                layout.extended,
            )?;
            check(
                s.satellite_id,
                "L1 CNR",
                s.l1.cnr.is_some(),
                layout.extended,
            )?;
            check(s.satellite_id, "L2 observables", s.l2.is_some(), layout.l2)?;
            if let Some(l2) = &s.l2 {
                check(s.satellite_id, "L2 CNR", l2.cnr.is_some(), layout.extended)?;
            }
        }
        Ok(())
    }
}

fn read_satellite(
    r: &mut BitReader<'_>,
    layout: Layout,
) -> std::result::Result<LegacySatellite, OutOfInput> {
    let satellite_id = r.u(6)? as u8;
    let code_indicator = r.flag()?;
    let frequency_channel = if layout.glonass() {
        Some(r.u(5)? as u8)
    } else {
        None
    };
    let pseudorange = r.u(layout.pseudorange_bits())? as u32;
    let phase_range_minus_pseudorange = r.i(20)? as i32;
    let lock_time_indicator = r.u(7)? as u8;
    let (pseudorange_modulus_ambiguity, cnr) = if layout.extended {
        (
            Some(r.u(layout.ambiguity_bits())? as u8),
            Some(r.u(8)? as u8),
        )
    } else {
        (None, None)
    };
    let l2 = if layout.l2 {
        Some(LegacyL2 {
            code_indicator: r.u(2)? as u8,
            pseudorange_difference: r.i(14)? as i16,
            phase_range_minus_l1_pseudorange: r.i(20)? as i32,
            lock_time_indicator: r.u(7)? as u8,
            cnr: if layout.extended {
                Some(r.u(8)? as u8)
            } else {
                None
            },
        })
    } else {
        None
    };
    Ok(LegacySatellite {
        satellite_id,
        frequency_channel,
        l1: LegacyL1 {
            code_indicator,
            pseudorange,
            phase_range_minus_pseudorange,
            lock_time_indicator,
            pseudorange_modulus_ambiguity,
            cnr,
        },
        l2,
    })
}

/// Write one satellite record. `check_parts` has refused a part the layout
/// carries held as `None` and one it does not carry held as `Some`, so each
/// `if let` writes exactly the carried fields.
fn write_satellite(w: &mut FieldWriter, layout: Layout, s: &LegacySatellite) -> Result<()> {
    let id = s.satellite_id;
    w.u("satellite ID", u64::from(id), 6)?;
    w.flag(s.l1.code_indicator);
    if let Some(channel) = s.frequency_channel {
        w.u(
            format_args!("satellite {id} frequency channel"),
            u64::from(channel),
            5,
        )?;
    }
    w.u(
        format_args!("satellite {id} L1 pseudorange"),
        u64::from(s.l1.pseudorange),
        layout.pseudorange_bits(),
    )?;
    w.i(
        format_args!("satellite {id} L1 phase range minus pseudorange"),
        i64::from(s.l1.phase_range_minus_pseudorange),
        20,
    )?;
    w.u(
        format_args!("satellite {id} L1 lock-time indicator"),
        u64::from(s.l1.lock_time_indicator),
        7,
    )?;
    if let Some(ambiguity) = s.l1.pseudorange_modulus_ambiguity {
        w.u(
            format_args!("satellite {id} L1 pseudorange modulus ambiguity"),
            u64::from(ambiguity),
            layout.ambiguity_bits(),
        )?;
    }
    if let Some(cnr) = s.l1.cnr {
        w.u(format_args!("satellite {id} L1 CNR"), u64::from(cnr), 8)?;
    }
    if let Some(l2) = &s.l2 {
        w.u(
            format_args!("satellite {id} L2 code indicator"),
            u64::from(l2.code_indicator),
            2,
        )?;
        w.i(
            format_args!("satellite {id} L2-L1 pseudorange difference"),
            i64::from(l2.pseudorange_difference),
            14,
        )?;
        w.i(
            format_args!("satellite {id} L2 phase range minus L1 pseudorange"),
            i64::from(l2.phase_range_minus_l1_pseudorange),
            20,
        )?;
        w.u(
            format_args!("satellite {id} L2 lock-time indicator"),
            u64::from(l2.lock_time_indicator),
            7,
        )?;
        if let Some(cnr) = l2.cnr {
            w.u(format_args!("satellite {id} L2 CNR"), u64::from(cnr), 8)?;
        }
    }
    Ok(())
}
