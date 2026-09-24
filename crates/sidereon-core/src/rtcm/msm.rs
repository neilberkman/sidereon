//! RTCM 3 Multiple Signal Messages (MSM), types MSM1 through MSM7.
//!
//! The MSM family carries multi-constellation, multi-signal pseudorange,
//! carrier-phase, phase-range-rate, lock-time, and carrier-to-noise observations
//! in one compact message (RTCM 10403.3, Section 3.5). The seven types share a
//! header and a satellite/signal/cell mask layout and differ in the fields they
//! carry:
//!
//! | Type | Satellite data                      | Signal data                                  |
//! |------|-------------------------------------|----------------------------------------------|
//! | MSM1 | DF398                               | DF400                                        |
//! | MSM2 | DF398                               | DF401, DF402, DF420                          |
//! | MSM3 | DF398                               | DF400, DF401, DF402, DF420                   |
//! | MSM4 | DF397, DF398                        | DF400, DF401, DF402, DF420, DF403            |
//! | MSM5 | DF397, DF419, DF398, DF399          | DF400, DF401, DF402, DF420, DF403, DF404     |
//! | MSM6 | DF397, DF398                        | DF405, DF406, DF407, DF420, DF408            |
//! | MSM7 | DF397, DF419, DF398, DF399          | DF405, DF406, DF407, DF420, DF408, DF404     |
//!
//! MSM1, MSM2 and MSM3 carry no whole-millisecond rough range (DF397): their
//! ranges are known modulo one millisecond. MSM6 and MSM7 carry the
//! extended-resolution fields.
//!
//! The message number alone fixes both the constellation and the MSM type via
//! the regular RTCM numbering (`107x` GPS, `108x` GLONASS, `109x` Galileo, `110x`
//! SBAS, `111x` QZSS, `112x` BeiDou, `113x` NavIC; the trailing digit is the MSM
//! type).
//!
//! ## Field-major packing
//!
//! MSM does not store one record per observation. The body is a common header,
//! then the satellite block with every field laid out column-first (all the
//! rough-range integers, then all the rough-range remainders, ...), then the
//! signal block laid out the same way over the active cells. The cell set is the
//! cross product of the satellite mask and signal mask, pruned by the cell mask.
//!
//! ## Canonical representation
//!
//! Field values are stored as the raw transmitted integers (the
//! `DFxxx`-numbered quantities), not pre-scaled engineering units, so the IR is
//! an exact, loss-free image of the wire bits and `decode` -> `encode`
//! round-trips byte-for-byte. A field the message's type does not carry is
//! `None`. Each accessor documents the standard scale factor so a consumer can
//! recover meters, milliseconds, or dB-Hz when needed.

use crate::error::{Error, Result};
use crate::id::GnssSystem;

use super::bits::{BitReader, FieldWriter, OutOfInput};
use super::{decode_body, write_trailing, DecodeContext, DecodeResult, RtcmDeparture, RtcmPolicy};
use super::{
    MsmMaskProblem, MsmOptionalField, MsmOptionalProblem, RtcmEncodeError, RtcmRecordKind,
};

/// DF397 rough range invalid / not available value (255), as RTKLIB
/// `decode_msm4`..`decode_msm7` test it.
pub const MSM_ROUGH_RANGE_INVALID: u8 = 255;

/// DF399 rough phase-range-rate invalid / not available sentinel (-8192 = -(1 << 13)).
pub const MSM_ROUGH_PHASE_RANGE_RATE_INVALID: i16 = -(1 << 13);

/// DF404 fine phase-range-rate invalid / not available sentinel (-16384 = -(1 << 14)).
pub const MSM_FINE_PHASE_RANGE_RATE_INVALID: i16 = -(1 << 14);

/// DF400 fine pseudorange invalid value, `-2^14`, carried by MSM1, MSM3,
/// MSM4 and MSM5.
pub const MSM4_FINE_PSEUDORANGE_INVALID: i32 = -(1 << 14);

/// DF401 fine phase range invalid value, `-2^21`, carried by MSM2, MSM3, MSM4
/// and MSM5.
pub const MSM4_FINE_PHASE_RANGE_INVALID: i32 = -(1 << 21);

/// DF405 fine pseudorange invalid value, `-2^19`, carried by MSM6 and MSM7.
pub const MSM7_FINE_PSEUDORANGE_INVALID: i32 = -(1 << 19);

/// DF406 fine phase range invalid value, `-2^23`, carried by MSM6 and MSM7.
pub const MSM7_FINE_PHASE_RANGE_INVALID: i32 = -(1 << 23);

/// Longest cell mask (DF396) RTCM 10403 allows, in bits.
const MSM_MAX_CELLS: usize = 64;

/// Which MSM type a message is, from the last digit of its message number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MsmKind {
    /// MSM1: fine pseudorange (DF400); ranges modulo one millisecond.
    Msm1,
    /// MSM2: fine phase range, lock time and half-cycle indicator; ranges
    /// modulo one millisecond.
    Msm2,
    /// MSM3: the MSM1 and MSM2 fields together; ranges modulo one millisecond.
    Msm3,
    /// MSM4: full pseudorange and phase range plus CNR, standard resolution.
    Msm4,
    /// MSM5: MSM4 plus the phase-range rate, standard resolution.
    Msm5,
    /// MSM6: full pseudorange and phase range plus CNR, extended resolution.
    Msm6,
    /// MSM7: MSM6 plus the phase-range rate, extended resolution.
    Msm7,
}

impl MsmKind {
    /// The MSM type number, `1..=7`: the last digit of the message number.
    pub const fn number(self) -> u8 {
        match self {
            Self::Msm1 => 1,
            Self::Msm2 => 2,
            Self::Msm3 => 3,
            Self::Msm4 => 4,
            Self::Msm5 => 5,
            Self::Msm6 => 6,
            Self::Msm7 => 7,
        }
    }

    /// The MSM type whose number is `number`, the last digit of an MSM
    /// message number; `None` outside `1..=7`.
    pub const fn from_number(number: u8) -> Option<Self> {
        match number {
            1 => Some(Self::Msm1),
            2 => Some(Self::Msm2),
            3 => Some(Self::Msm3),
            4 => Some(Self::Msm4),
            5 => Some(Self::Msm5),
            6 => Some(Self::Msm6),
            7 => Some(Self::Msm7),
            _ => None,
        }
    }

    /// Whether the satellite data carries the whole-millisecond rough range
    /// (DF397): MSM4 through MSM7. Without it a range is known modulo one
    /// millisecond.
    pub const fn carries_rough_range_ms(self) -> bool {
        matches!(self, Self::Msm4 | Self::Msm5 | Self::Msm6 | Self::Msm7)
    }

    /// Whether the message carries the extended satellite information (DF419)
    /// and the rough and fine phase-range rates (DF399, DF404): MSM5 and MSM7.
    pub const fn carries_phase_range_rate(self) -> bool {
        matches!(self, Self::Msm5 | Self::Msm7)
    }

    /// Whether the signal data carries a fine pseudorange (DF400 or DF405):
    /// every type but MSM2.
    pub const fn carries_pseudorange(self) -> bool {
        !matches!(self, Self::Msm2)
    }

    /// Whether the signal data carries a fine phase range, a lock-time
    /// indicator and a half-cycle ambiguity indicator: every type but MSM1.
    pub const fn carries_phase_range(self) -> bool {
        !matches!(self, Self::Msm1)
    }

    /// Whether the signal data carries the carrier-to-noise ratio (DF403 or
    /// DF408): MSM4 through MSM7.
    pub const fn carries_cnr(self) -> bool {
        self.carries_rough_range_ms()
    }

    /// Whether the signal fields have the extended resolution (DF405, DF406,
    /// DF407, DF408): MSM6 and MSM7.
    pub const fn is_extended_resolution(self) -> bool {
        matches!(self, Self::Msm6 | Self::Msm7)
    }

    /// Widths of the fine pseudorange, fine phase range, lock-time indicator
    /// and CNR fields.
    const fn signal_widths(self) -> SignalWidths {
        if self.is_extended_resolution() {
            SignalWidths {
                pseudorange: 20,
                phase_range: 24,
                lock_time: 10,
                cnr: 10,
            }
        } else {
            SignalWidths {
                pseudorange: 15,
                phase_range: 22,
                lock_time: 4,
                cnr: 6,
            }
        }
    }
}

/// Signal-field widths of one MSM type.
#[derive(Clone, Copy)]
struct SignalWidths {
    pseudorange: usize,
    phase_range: usize,
    lock_time: usize,
    cnr: usize,
}

/// The MSM message header, common to every MSM type (RTCM 10403.3 Table 3.5-78).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MsmHeader {
    /// Reference station identifier (DF003).
    pub reference_station_id: u16,
    /// GNSS epoch time, the raw 30-bit field. Its meaning is constellation
    /// specific: milliseconds of the GPS/Galileo/BeiDou week, or, for GLONASS,
    /// a 3-bit day-of-week joined with a 27-bit millisecond-of-day count.
    pub epoch_time: u32,
    /// Multiple message bit (DF393): more MSM messages share this epoch.
    pub multiple_message: bool,
    /// Issue of data station (DF409).
    pub iods: u8,
    /// Reserved field DF001 (7 bits), preserved for exact round-trip.
    pub reserved: u8,
    /// Clock steering indicator (DF411).
    pub clock_steering: u8,
    /// External clock indicator (DF412).
    pub external_clock: u8,
    /// Divergence-free smoothing indicator (DF417).
    pub divergence_free_smoothing: bool,
    /// Smoothing interval (DF418).
    pub smoothing_interval: u8,
}

/// Per-satellite data for one MSM satellite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MsmSatellite {
    /// Satellite identifier: the 1-based index of the set bit in the satellite
    /// mask (DF394). For most constellations this equals the PRN / slot number.
    pub id: u8,
    /// Rough range, whole milliseconds (DF397), carried by MSM4 through MSM7
    /// and `None` in MSM1, MSM2 and MSM3. The value 255
    /// ([`MSM_ROUGH_RANGE_INVALID`]) marks the satellite range as invalid.
    pub rough_range_ms: Option<u8>,
    /// Rough range remainder, in units of 1/1024 ms (DF398, scale 2^-10 ms).
    pub rough_range_mod1: u16,
    /// Extended satellite info (DF419), carried by MSM5 and MSM7. For GLONASS
    /// this is the frequency channel number plus 7.
    pub extended_info: Option<u8>,
    /// Rough phase-range-rate in whole m/s (DF399), carried by MSM5 and MSM7;
    /// `None` also when the field holds its invalid value
    /// ([`MSM_ROUGH_PHASE_RANGE_RATE_INVALID`]).
    pub rough_phase_range_rate_m_s: Option<i16>,
}

/// Per-cell signal data for one active (satellite, signal) pair.
///
/// A field the message's MSM type does not carry is `None`; see
/// [`MsmKind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MsmSignal {
    /// Owning satellite id (1-based satellite-mask index).
    pub satellite_id: u8,
    /// Signal id: the 1-based index of the set bit in the signal mask (DF395).
    pub signal_id: u8,
    /// Fine pseudorange (DF400 in MSM1, MSM3, MSM4 and MSM5, scale 2^-24 ms;
    /// DF405 in MSM6 and MSM7, scale 2^-29 ms); `None` in MSM2. The invalid
    /// value is [`MSM4_FINE_PSEUDORANGE_INVALID`] (`-2^14`) for DF400 and
    /// [`MSM7_FINE_PSEUDORANGE_INVALID`] (`-2^19`) for DF405.
    pub fine_pseudorange: Option<i32>,
    /// Fine phase range (DF401 in MSM2 through MSM5, scale 2^-29 ms; DF406 in
    /// MSM6 and MSM7, scale 2^-31 ms); `None` in MSM1. The invalid value is
    /// [`MSM4_FINE_PHASE_RANGE_INVALID`] (`-2^21`) for DF401 and
    /// [`MSM7_FINE_PHASE_RANGE_INVALID`] (`-2^23`) for DF406.
    pub fine_phase_range: Option<i32>,
    /// Phase-range lock-time indicator (DF402, 4-bit, in MSM2 through MSM5;
    /// DF407, 10-bit, in MSM6 and MSM7); `None` in MSM1.
    pub lock_time_indicator: Option<u16>,
    /// Half-cycle ambiguity indicator (DF420); `None` in MSM1.
    pub half_cycle_ambiguity: Option<bool>,
    /// Carrier-to-noise density ratio (DF403, 1 dB-Hz, in MSM4 and MSM5;
    /// DF408, scale 2^-4 dB-Hz, in MSM6 and MSM7); `None` in MSM1, MSM2 and
    /// MSM3.
    pub cnr: Option<u16>,
    /// Fine phase-range-rate (DF404, scale 0.0001 m/s), carried by MSM5 and
    /// MSM7; `None` also when the field holds its invalid value
    /// ([`MSM_FINE_PHASE_RANGE_RATE_INVALID`]).
    pub fine_phase_range_rate: Option<i16>,
}

impl MsmSignal {
    /// Minimum continuous-lock time encoded by this signal's lock indicator.
    ///
    /// The caller supplies the owning message kind because MSM2 through MSM5
    /// carry DF402 while MSM6 and MSM7 carry DF407. Returns `None` when the
    /// signal carries no lock indicator (MSM1) and for values outside the
    /// indicator's defined range.
    pub fn minimum_lock_time_ms(&self, kind: MsmKind) -> Option<u32> {
        super::lli::minimum_lock_time_ms(kind, self.lock_time_indicator?)
    }
}

/// A decoded MSM1 through MSM7 observation message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MsmMessage {
    /// The message number (e.g. 1077).
    pub message_number: u16,
    /// The constellation, derived from the message number.
    pub system: GnssSystem,
    /// The MSM type, from the last digit of the message number.
    pub kind: MsmKind,
    /// Common MSM header.
    pub header: MsmHeader,
    /// The signal mask (DF395) as transmitted: bit `32 - id` is set for each
    /// signal id the message lists. A listed signal may have no cell, so the
    /// mask is kept rather than rebuilt from [`Self::signals`]; every cell's
    /// signal must be set in it.
    pub signal_mask: u32,
    /// Active satellites, in ascending id order.
    pub satellites: Vec<MsmSatellite>,
    /// Active signal cells, in satellite-major then signal order.
    pub signals: Vec<MsmSignal>,
    /// Every body bit after the last field, the zeros that align the body to a
    /// byte included, kept whenever those bits are anything other than fewer
    /// than eight zeros: read under [`RtcmPolicy::Lenient`] and written back
    /// after the last field by `encode_with_policy` under that policy, so the
    /// body re-encodes byte for byte. Empty when the bits after the last field
    /// are fewer than eight zeros, for every body read under
    /// [`RtcmPolicy::Strict`], and for a message built by hand; `encode`
    /// refuses a nonempty value. A tail set by hand is zero-padded to the byte
    /// when written and reads back with that padding.
    pub trailing_bits: Vec<bool>,
}

/// Map an MSM message number to its constellation and MSM type.
///
/// Returns `None` for numbers outside the MSM ranges `1071..=1077`,
/// `1081..=1087`, ..., `1131..=1137`: the last digit 8, 9 and 0 are not MSM
/// types.
pub(crate) fn msm_kind(message_number: u16) -> Option<(GnssSystem, MsmKind)> {
    if !(1071..=1137).contains(&message_number) {
        return None;
    }
    let group = (message_number - 1071) / 10;
    let system = match group {
        0 => GnssSystem::Gps,
        1 => GnssSystem::Glonass,
        2 => GnssSystem::Galileo,
        3 => GnssSystem::Sbas,
        4 => GnssSystem::Qzss,
        5 => GnssSystem::BeiDou,
        6 => GnssSystem::Navic,
        _ => return None,
    };
    let kind = MsmKind::from_number((message_number % 10) as u8)?;
    Some((system, kind))
}

/// True if `message_number` is an MSM message number.
pub(crate) fn is_supported_msm(message_number: u16) -> bool {
    msm_kind(message_number).is_some()
}

impl MsmMessage {
    /// Decode an MSM body (without the transport frame) under
    /// [`RtcmPolicy::Strict`]: a cell mask over 64 bits and bits after the last
    /// field other than the zero byte alignment are refused.
    pub fn decode(body: &[u8]) -> Result<Self> {
        decode_body(
            body,
            &mut DecodeContext::new(RtcmPolicy::Strict),
            Self::read,
        )
        .map_err(Into::into)
    }

    pub(crate) fn read(r: &mut BitReader<'_>, ctx: &mut DecodeContext) -> DecodeResult<Self> {
        let message_number = r.u(12)? as u16;
        let (system, kind) = msm_kind(message_number)
            .ok_or_else(|| Error::Parse(format!("message {message_number} is not an MSM type")))?;

        let header = MsmHeader {
            reference_station_id: r.u(12)? as u16,
            epoch_time: r.u(30)? as u32,
            multiple_message: r.flag()?,
            iods: r.u(3)? as u8,
            reserved: r.u(7)? as u8,
            clock_steering: r.u(2)? as u8,
            external_clock: r.u(2)? as u8,
            divergence_free_smoothing: r.flag()?,
            smoothing_interval: r.u(3)? as u8,
        };

        let satellite_mask = r.u(64)?;
        let signal_mask = r.u(32)? as u32;
        let sat_ids = set_bits(satellite_mask, 64);
        let sig_ids = set_bits_u32(signal_mask);

        let nsat = sat_ids.len();
        let nsig = sig_ids.len();
        if nsat * nsig > MSM_MAX_CELLS {
            ctx.depart(RtcmDeparture::MsmCellMaskOver64 {
                message_number,
                cells: nsat * nsig,
            })?;
        }

        // Cell mask: nsat * nsig bits, satellite-major.
        let mut cell_present = Vec::with_capacity(nsat * nsig);
        for _ in 0..nsat * nsig {
            cell_present.push(r.flag()?);
        }

        // Satellite block (column-major).
        let rough_range_ms = read_if(r, kind.carries_rough_range_ms(), nsat, |rr| {
            rr.u(8).map(|v| v as u8)
        })?;
        let extended_info = read_if(r, kind.carries_phase_range_rate(), nsat, |rr| {
            rr.u(4).map(|v| v as u8)
        })?;
        let rough_range_mod1 = read_vec(r, nsat, |rr| rr.u(10).map(|v| v as u16))?;
        let rough_prr = read_if(r, kind.carries_phase_range_rate(), nsat, |rr| {
            rr.i(14).map(|v| v as i16)
        })?;

        let satellites: Vec<MsmSatellite> = (0..nsat)
            .map(|s| MsmSatellite {
                id: sat_ids[s],
                rough_range_ms: rough_range_ms[s],
                rough_range_mod1: rough_range_mod1[s],
                extended_info: extended_info[s],
                rough_phase_range_rate_m_s: rough_prr[s]
                    .filter(|&raw| raw != MSM_ROUGH_PHASE_RANGE_RATE_INVALID),
            })
            .collect();

        // The ordered list of active (satellite, signal) cells.
        let cells = active_cells(&sat_ids, &sig_ids, &cell_present);
        let ncell = cells.len();

        // Signal block (column-major over cells).
        let widths = kind.signal_widths();
        let phase = kind.carries_phase_range();
        let fine_pr = read_if(r, kind.carries_pseudorange(), ncell, |rr| {
            rr.i(widths.pseudorange).map(|v| v as i32)
        })?;
        let fine_ph = read_if(r, phase, ncell, |rr| {
            rr.i(widths.phase_range).map(|v| v as i32)
        })?;
        let lock = read_if(r, phase, ncell, |rr| {
            rr.u(widths.lock_time).map(|v| v as u16)
        })?;
        let half = read_if(r, phase, ncell, |rr| rr.flag())?;
        let cnr = read_if(r, kind.carries_cnr(), ncell, |rr| {
            rr.u(widths.cnr).map(|v| v as u16)
        })?;
        let fine_prr = read_if(r, kind.carries_phase_range_rate(), ncell, |rr| {
            rr.i(15).map(|v| v as i16)
        })?;
        let signals = cells
            .iter()
            .enumerate()
            .map(|(c, &(sat, sig))| MsmSignal {
                satellite_id: sat,
                signal_id: sig,
                fine_pseudorange: fine_pr[c],
                fine_phase_range: fine_ph[c],
                lock_time_indicator: lock[c],
                half_cycle_ambiguity: half[c],
                cnr: cnr[c],
                fine_phase_range_rate: fine_prr[c]
                    .filter(|&raw| raw != MSM_FINE_PHASE_RANGE_RATE_INVALID),
            })
            .collect();

        Ok(Self {
            message_number,
            system,
            kind,
            header,
            signal_mask,
            satellites,
            signals,
            trailing_bits: Vec::new(),
        })
    }

    /// Encode this message back into an MSM body (without the transport frame)
    /// under [`RtcmPolicy::Strict`].
    ///
    /// # Errors
    ///
    /// [`Error::RtcmEncode`] naming what cannot be written as the MSM wire
    /// form states it:
    ///
    /// * a message number whose constellation and MSM type differ from
    ///   [`Self::system`] and [`Self::kind`];
    /// * satellite and signal lists the masks cannot state: a satellite id
    ///   outside `1..=64`, a satellite or cell listed twice, a signal id
    ///   outside `1..=32` or not set in [`Self::signal_mask`], or a signal
    ///   whose satellite is not in the satellite list;
    /// * a cell mask over 64 bits ([`RtcmDeparture::MsmCellMaskOver64`]);
    /// * a field the message's MSM type carries held as `None`, or a field it
    ///   does not carry held as `Some` (see [`MsmKind`]); the rough and fine
    ///   phase-range rates, whose `None` is also the invalid value, are only
    ///   refused as `Some` where the type does not carry them;
    /// * a phase-range rate of `Some` holding its field's invalid value, which
    ///   is how `None` is written and would be read back as `None`;
    /// * a value wider than its field.
    ///
    /// Each would otherwise be shifted onto another bit, filled, or left out of
    /// the body without a trace.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this message under `policy`. Under [`RtcmPolicy::Lenient`] a cell
    /// mask over 64 bits is written and reported; every other refusal of
    /// [`Self::encode`] applies under both policies.
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let number = self.message_number;
        if msm_kind(number) != Some((self.system, self.kind)) {
            return Err(RtcmEncodeError::MessageNumber {
                message_number: number,
                record: RtcmRecordKind::Msm {
                    system: self.system,
                    kind: self.kind,
                },
            }
            .into());
        }
        self.check_masks()?;
        self.check_optional_fields()?;

        // Satellite ids (sorted) and the satellite mask.
        let mut sat_ids: Vec<u8> = self.satellites.iter().map(|s| s.id).collect();
        sat_ids.sort_unstable();
        let mut satellite_mask: u64 = 0;
        for &id in &sat_ids {
            satellite_mask |= 1u64 << (64 - u32::from(id));
        }
        let sig_ids = set_bits_u32(self.signal_mask);

        let mut departures = Vec::new();
        let cells = sat_ids.len() * sig_ids.len();
        if cells > MSM_MAX_CELLS {
            let departure = RtcmDeparture::MsmCellMaskOver64 {
                message_number: number,
                cells,
            };
            match policy {
                RtcmPolicy::Strict => {
                    return Err(RtcmEncodeError::StrictDeparture(departure).into())
                }
                RtcmPolicy::Lenient => departures.push(departure),
            }
        }

        let mut w = FieldWriter::new(number);
        w.u("message number", u64::from(number), 12)?;
        w.u(
            "reference station ID",
            u64::from(self.header.reference_station_id),
            12,
        )?;
        w.u("epoch time", u64::from(self.header.epoch_time), 30)?;
        w.flag(self.header.multiple_message);
        w.u("IODS", u64::from(self.header.iods), 3)?;
        w.u("reserved", u64::from(self.header.reserved), 7)?;
        w.u(
            "clock steering indicator",
            u64::from(self.header.clock_steering),
            2,
        )?;
        w.u(
            "external clock indicator",
            u64::from(self.header.external_clock),
            2,
        )?;
        w.flag(self.header.divergence_free_smoothing);
        w.u(
            "smoothing interval",
            u64::from(self.header.smoothing_interval),
            3,
        )?;
        w.u("satellite mask", satellite_mask, 64)?;
        w.u("signal mask", u64::from(self.signal_mask), 32)?;

        // Cell mask, satellite-major, plus the ordered active cell list.
        let mut ordered: Vec<&MsmSignal> = Vec::with_capacity(self.signals.len());
        for &sat in &sat_ids {
            for &sig in &sig_ids {
                let cell = self
                    .signals
                    .iter()
                    .find(|s| s.satellite_id == sat && s.signal_id == sig);
                w.flag(cell.is_some());
                ordered.extend(cell);
            }
        }

        // Satellite block, column-major, in the same sorted id order.
        // `check_optional_fields` has refused a `None` the type carries and a
        // `Some` it does not, so each `if let` below writes exactly the
        // carried columns.
        let mut satellites: Vec<&MsmSatellite> = self.satellites.iter().collect();
        satellites.sort_unstable_by_key(|s| s.id);
        for s in &satellites {
            if let Some(ms) = s.rough_range_ms {
                w.u(
                    format_args!("satellite {} rough range", s.id),
                    u64::from(ms),
                    8,
                )?;
            }
        }
        for s in &satellites {
            if let Some(ext) = s.extended_info {
                w.u(
                    format_args!("satellite {} extended info", s.id),
                    u64::from(ext),
                    4,
                )?;
            }
        }
        for s in &satellites {
            w.u(
                format_args!("satellite {} rough range modulo 1 ms", s.id),
                u64::from(s.rough_range_mod1),
                10,
            )?;
        }
        if self.kind.carries_phase_range_rate() {
            for s in &satellites {
                let prr = s
                    .rough_phase_range_rate_m_s
                    .unwrap_or(MSM_ROUGH_PHASE_RANGE_RATE_INVALID);
                w.i(
                    format_args!("satellite {} rough phase-range rate", s.id),
                    i64::from(prr),
                    14,
                )?;
            }
        }

        // Signal block, column-major over the ordered cells.
        let widths = self.kind.signal_widths();
        let cell_name = |s: &MsmSignal, field: &str| {
            format!(
                "satellite {} signal {} {field}",
                s.satellite_id, s.signal_id
            )
        };
        for s in &ordered {
            if let Some(value) = s.fine_pseudorange {
                w.i(
                    cell_name(s, "fine pseudorange"),
                    i64::from(value),
                    widths.pseudorange,
                )?;
            }
        }
        for s in &ordered {
            if let Some(value) = s.fine_phase_range {
                w.i(
                    cell_name(s, "fine phase range"),
                    i64::from(value),
                    widths.phase_range,
                )?;
            }
        }
        for s in &ordered {
            if let Some(value) = s.lock_time_indicator {
                w.u(
                    cell_name(s, "lock-time indicator"),
                    u64::from(value),
                    widths.lock_time,
                )?;
            }
        }
        for s in &ordered {
            if let Some(half) = s.half_cycle_ambiguity {
                w.flag(half);
            }
        }
        for s in &ordered {
            if let Some(value) = s.cnr {
                w.u(cell_name(s, "CNR"), u64::from(value), widths.cnr)?;
            }
        }
        if self.kind.carries_phase_range_rate() {
            for s in &ordered {
                w.i(
                    cell_name(s, "fine phase-range rate"),
                    i64::from(
                        s.fine_phase_range_rate
                            .unwrap_or(MSM_FINE_PHASE_RANGE_RATE_INVALID),
                    ),
                    15,
                )?;
            }
        }

        departures.extend(write_trailing(&mut w, &self.trailing_bits, policy)?);
        Ok((w.into_bytes(), departures))
    }

    /// Refuse satellite and signal lists the MSM masks cannot state exactly.
    fn check_masks(&self) -> Result<()> {
        let refuse = |problem: MsmMaskProblem| -> Result<()> {
            Err(RtcmEncodeError::MsmMask {
                message_number: self.message_number,
                problem,
            }
            .into())
        };
        let mut sat_ids = std::collections::BTreeSet::new();
        for satellite in &self.satellites {
            if !(1..=MSM_SATELLITE_MASK_BITS).contains(&satellite.id) {
                return refuse(MsmMaskProblem::SatelliteOutsideMask {
                    satellite: satellite.id,
                });
            }
            if !sat_ids.insert(satellite.id) {
                return refuse(MsmMaskProblem::SatelliteListedTwice {
                    satellite: satellite.id,
                });
            }
        }
        let mut cells = std::collections::BTreeSet::new();
        for signal in &self.signals {
            if !(1..=MSM_SIGNAL_MASK_BITS).contains(&signal.signal_id) {
                return refuse(MsmMaskProblem::SignalOutsideMask {
                    signal: signal.signal_id,
                });
            }
            if self.signal_mask & (1u32 << (32 - u32::from(signal.signal_id))) == 0 {
                return refuse(MsmMaskProblem::SignalNotInMask {
                    signal: signal.signal_id,
                    mask: self.signal_mask,
                });
            }
            if !sat_ids.contains(&signal.satellite_id) {
                return refuse(MsmMaskProblem::SignalSatelliteNotListed {
                    signal: signal.signal_id,
                    satellite: signal.satellite_id,
                });
            }
            if !cells.insert((signal.satellite_id, signal.signal_id)) {
                return refuse(MsmMaskProblem::CellListedTwice {
                    satellite: signal.satellite_id,
                    signal: signal.signal_id,
                });
            }
        }
        Ok(())
    }

    /// Refuse optional values the message's MSM type does not carry, a
    /// missing value it does, and `Some` of an invalid value (the spelling of
    /// `None`).
    fn check_optional_fields(&self) -> Result<()> {
        let number = self.message_number;
        let kind = self.kind;
        let record = RtcmRecordKind::Msm {
            system: self.system,
            kind,
        };
        let check = |field: &'static str, present: bool, carried: bool| -> Result<()> {
            match (carried, present) {
                (true, false) => Err(RtcmEncodeError::FieldPresence {
                    message_number: number,
                    record,
                    field,
                    carried: true,
                }
                .into()),
                (false, true) => Err(RtcmEncodeError::FieldPresence {
                    message_number: number,
                    record,
                    field,
                    carried: false,
                }
                .into()),
                _ => Ok(()),
            }
        };
        let rate = kind.carries_phase_range_rate();
        for s in &self.satellites {
            let id = s.id;
            check(
                "rough range",
                s.rough_range_ms.is_some(),
                kind.carries_rough_range_ms(),
            )?;
            match (rate, s.extended_info.is_some()) {
                (true, false) => {
                    return Err(RtcmEncodeError::MsmOptional {
                        message_number: number,
                        kind,
                        satellite: id,
                        signal: None,
                        field: MsmOptionalField::ExtendedInfo,
                        problem: MsmOptionalProblem::Missing,
                    }
                    .into())
                }
                (false, true) => {
                    return Err(RtcmEncodeError::MsmOptional {
                        message_number: number,
                        kind,
                        satellite: id,
                        signal: None,
                        field: MsmOptionalField::ExtendedInfo,
                        problem: MsmOptionalProblem::NotCarried,
                    }
                    .into())
                }
                _ => {}
            }
            match s.rough_phase_range_rate_m_s {
                Some(_) if !rate => {
                    return Err(RtcmEncodeError::MsmOptional {
                        message_number: number,
                        kind,
                        satellite: id,
                        signal: None,
                        field: MsmOptionalField::RoughPhaseRangeRate,
                        problem: MsmOptionalProblem::NotCarried,
                    }
                    .into())
                }
                Some(MSM_ROUGH_PHASE_RANGE_RATE_INVALID) => {
                    return Err(RtcmEncodeError::MsmOptional {
                        message_number: number,
                        kind: self.kind,
                        satellite: id,
                        signal: None,
                        field: MsmOptionalField::RoughPhaseRangeRate,
                        problem: MsmOptionalProblem::InvalidValue(i64::from(
                            MSM_ROUGH_PHASE_RANGE_RATE_INVALID,
                        )),
                    }
                    .into())
                }
                _ => {}
            }
        }
        for s in &self.signals {
            check(
                "fine pseudorange",
                s.fine_pseudorange.is_some(),
                kind.carries_pseudorange(),
            )?;
            check(
                "fine phase range",
                s.fine_phase_range.is_some(),
                kind.carries_phase_range(),
            )?;
            check(
                "lock-time indicator",
                s.lock_time_indicator.is_some(),
                kind.carries_phase_range(),
            )?;
            check(
                "half-cycle ambiguity indicator",
                s.half_cycle_ambiguity.is_some(),
                kind.carries_phase_range(),
            )?;
            check("CNR", s.cnr.is_some(), kind.carries_cnr())?;
            match s.fine_phase_range_rate {
                Some(_) if !rate => {
                    return Err(RtcmEncodeError::MsmOptional {
                        message_number: number,
                        kind,
                        satellite: s.satellite_id,
                        signal: Some(s.signal_id),
                        field: MsmOptionalField::FinePhaseRangeRate,
                        problem: MsmOptionalProblem::NotCarried,
                    }
                    .into())
                }
                Some(MSM_FINE_PHASE_RANGE_RATE_INVALID) => {
                    return Err(RtcmEncodeError::MsmOptional {
                        message_number: number,
                        kind,
                        satellite: s.satellite_id,
                        signal: Some(s.signal_id),
                        field: MsmOptionalField::FinePhaseRangeRate,
                        problem: MsmOptionalProblem::InvalidValue(i64::from(
                            MSM_FINE_PHASE_RANGE_RATE_INVALID,
                        )),
                    }
                    .into())
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// The signal mask (DF395) with a bit set for every signal id in `signals`,
/// for building a message whose every listed signal has a cell.
pub fn msm_signal_mask(signals: &[MsmSignal]) -> u32 {
    signals
        .iter()
        .filter(|s| (1..=MSM_SIGNAL_MASK_BITS).contains(&s.signal_id))
        .fold(0u32, |mask, s| {
            mask | (1u32 << (32 - u32::from(s.signal_id)))
        })
}

/// Satellite mask width (DF394): satellite ids run `1..=64`.
const MSM_SATELLITE_MASK_BITS: u8 = 64;
/// Signal mask width (DF395): signal ids run `1..=32`.
const MSM_SIGNAL_MASK_BITS: u8 = 32;

/// Read `n` values with `f` when `carried`, or give `n` `None`s when the
/// message's type does not carry the field.
fn read_if<T>(
    r: &mut BitReader<'_>,
    carried: bool,
    n: usize,
    f: impl FnMut(&mut BitReader<'_>) -> std::result::Result<T, OutOfInput>,
) -> std::result::Result<Vec<Option<T>>, OutOfInput> {
    if carried {
        Ok(read_vec(r, n, f)?.into_iter().map(Some).collect())
    } else {
        Ok(std::iter::repeat_with(|| None).take(n).collect())
    }
}

/// Read `n` values with `f`, collecting into a vector.
fn read_vec<T>(
    r: &mut BitReader<'_>,
    n: usize,
    mut f: impl FnMut(&mut BitReader<'_>) -> std::result::Result<T, OutOfInput>,
) -> std::result::Result<Vec<T>, OutOfInput> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(f(r)?);
    }
    Ok(v)
}

/// The 1-based positions of the set bits in `mask`, scanning from the MSB of an
/// `n`-bit field.
fn set_bits(mask: u64, n: u32) -> Vec<u8> {
    let mut ids = Vec::new();
    for i in 0..n {
        if (mask >> (n - 1 - i)) & 1 == 1 {
            ids.push((i + 1) as u8);
        }
    }
    ids
}

/// The 1-based positions of the set bits in a 32-bit signal mask.
fn set_bits_u32(mask: u32) -> Vec<u8> {
    let mut ids = Vec::new();
    for i in 0..32u32 {
        if (mask >> (31 - i)) & 1 == 1 {
            ids.push((i + 1) as u8);
        }
    }
    ids
}

/// Build the ordered active-cell list from the masks and the cell-present bits.
fn active_cells(sat_ids: &[u8], sig_ids: &[u8], cell_present: &[bool]) -> Vec<(u8, u8)> {
    let nsig = sig_ids.len();
    let mut cells = Vec::new();
    for (si, &sat) in sat_ids.iter().enumerate() {
        for (gi, &sig) in sig_ids.iter().enumerate() {
            if cell_present[si * nsig + gi] {
                cells.push((sat, sig));
            }
        }
    }
    cells
}

impl super::TrailingBits for MsmMessage {
    fn trailing_bits_mut(&mut self) -> &mut Vec<bool> {
        &mut self.trailing_bits
    }
}
