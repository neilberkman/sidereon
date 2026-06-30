//! RTCM 3 Multiple Signal Message (MSM) observations, types MSM4 and MSM7.
//!
//! The MSM family carries multi-constellation, multi-signal pseudorange,
//! carrier-phase, phase-range-rate, lock-time, and carrier-to-noise observations
//! in one compact message (RTCM 10403.3, Section 3.5). This module decodes and
//! re-encodes the two highest-value members:
//!
//!   * **MSM4** - full pseudoranges and phase ranges with the standard
//!     resolution (message numbers 1074 / 1084 / 1094 / 1124, and the SBAS /
//!     QZSS / NavIC siblings).
//!   * **MSM7** - full pseudoranges, phase ranges, phase-range-rates, and
//!     extended resolution (message numbers 1077 / 1087 / 1097 / 1127, and the
//!     siblings).
//!
//! The message number alone fixes both the constellation and the MSM type via
//! the regular RTCM numbering (`107x` GPS, `108x` GLONASS, `109x` Galileo, `110x`
//! SBAS, `111x` QZSS, `112x` BeiDou, `113x` NavIC; the trailing digit is the MSM
//! type). Other MSM types (1, 2, 3, 5, 6) are left to the caller as
//! [`super::Message::Unsupported`].
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
//! round-trips byte-for-byte. Each accessor documents the standard scale factor
//! so a consumer can recover meters, milliseconds, or dB-Hz when needed.

use crate::error::{Error, Result};
use crate::id::GnssSystem;

use super::bits::{BitReader, BitWriter};

/// Which MSM variant a message is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MsmKind {
    /// MSM4: full pseudorange + phase range, standard resolution.
    Msm4,
    /// MSM7: full pseudorange + phase range + phase-range-rate, extended
    /// resolution.
    Msm7,
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
    /// Rough range, whole milliseconds (DF397). The value 255 marks the
    /// satellite range as invalid.
    pub rough_range_ms: u8,
    /// Rough range remainder, in units of 1/1024 ms (DF398, scale 2^-10 ms).
    pub rough_range_mod1: u16,
    /// Extended satellite info (DF419), present only in MSM7. For GLONASS this
    /// is the frequency channel number.
    pub extended_info: Option<u8>,
    /// Rough phase-range-rate in whole m/s (DF399), present only in MSM7.
    pub rough_phase_range_rate_m_s: Option<i16>,
}

/// Per-cell signal data for one active (satellite, signal) pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MsmSignal {
    /// Owning satellite id (1-based satellite-mask index).
    pub satellite_id: u8,
    /// Signal id: the 1-based index of the set bit in the signal mask (DF395).
    pub signal_id: u8,
    /// Fine pseudorange (DF400 for MSM4, scale 2^-24 ms; DF405 for MSM7, scale
    /// 2^-29 ms). The MSM4 invalid marker is -16384.
    pub fine_pseudorange: i32,
    /// Fine phase range (DF401 for MSM4, scale 2^-29 ms; DF406 for MSM7, scale
    /// 2^-31 ms).
    pub fine_phase_range: i32,
    /// Phase-range lock-time indicator (DF402, 4-bit, for MSM4; DF407, 10-bit,
    /// for MSM7).
    pub lock_time_indicator: u16,
    /// Half-cycle ambiguity indicator (DF420).
    pub half_cycle_ambiguity: bool,
    /// Carrier-to-noise density ratio (DF403, 1 dB-Hz, for MSM4; DF408, scale
    /// 2^-4 dB-Hz, for MSM7).
    pub cnr: u16,
    /// Fine phase-range-rate (DF404, scale 0.0001 m/s), present only in MSM7.
    pub fine_phase_range_rate: Option<i16>,
}

/// A decoded MSM4 or MSM7 observation message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MsmMessage {
    /// The message number (e.g. 1077).
    pub message_number: u16,
    /// The constellation, derived from the message number.
    pub system: GnssSystem,
    /// The MSM variant (MSM4 or MSM7).
    pub kind: MsmKind,
    /// Common MSM header.
    pub header: MsmHeader,
    /// Active satellites, in ascending id order.
    pub satellites: Vec<MsmSatellite>,
    /// Active signal cells, in satellite-major then signal order.
    pub signals: Vec<MsmSignal>,
}

/// Map an MSM message number to its constellation and (supported) MSM type.
///
/// Returns `None` for numbers outside the MSM range and for MSM types this
/// module does not decode (1, 2, 3, 5, 6).
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
    let kind = match message_number % 10 {
        4 => MsmKind::Msm4,
        7 => MsmKind::Msm7,
        _ => return None,
    };
    Some((system, kind))
}

/// True if `message_number` is an MSM type decoded by this module.
pub(crate) fn is_supported_msm(message_number: u16) -> bool {
    msm_kind(message_number).is_some()
}

impl MsmMessage {
    /// Decode an MSM4 / MSM7 message body (without the transport frame).
    pub fn decode(body: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(body);
        let message_number = r.u(12)? as u16;
        let (system, kind) = msm_kind(message_number).ok_or_else(|| {
            Error::Parse(format!(
                "message {message_number} is not a supported MSM4/MSM7 type"
            ))
        })?;

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

        // Cell mask: nsat * nsig bits, satellite-major.
        let mut cell_present = Vec::with_capacity(nsat * nsig);
        for _ in 0..nsat * nsig {
            cell_present.push(r.flag()?);
        }

        // Satellite block (column-major).
        let mut rough_range_ms = Vec::with_capacity(nsat);
        for _ in 0..nsat {
            rough_range_ms.push(r.u(8)? as u8);
        }
        let extended_info = if kind == MsmKind::Msm7 {
            let mut v = Vec::with_capacity(nsat);
            for _ in 0..nsat {
                v.push(Some(r.u(4)? as u8));
            }
            v
        } else {
            vec![None; nsat]
        };
        let mut rough_range_mod1 = Vec::with_capacity(nsat);
        for _ in 0..nsat {
            rough_range_mod1.push(r.u(10)? as u16);
        }
        let rough_prr = if kind == MsmKind::Msm7 {
            let mut v = Vec::with_capacity(nsat);
            for _ in 0..nsat {
                v.push(Some(r.i(14)? as i16));
            }
            v
        } else {
            vec![None; nsat]
        };

        let satellites: Vec<MsmSatellite> = (0..nsat)
            .map(|s| MsmSatellite {
                id: sat_ids[s],
                rough_range_ms: rough_range_ms[s],
                rough_range_mod1: rough_range_mod1[s],
                extended_info: extended_info[s],
                rough_phase_range_rate_m_s: rough_prr[s],
            })
            .collect();

        // The ordered list of active (satellite, signal) cells.
        let cells = active_cells(&sat_ids, &sig_ids, &cell_present);
        let ncell = cells.len();

        // Signal block (column-major over cells).
        let signals = match kind {
            MsmKind::Msm4 => {
                let fine_pr = read_vec(&mut r, ncell, |rr| rr.i(15).map(|v| v as i32))?;
                let fine_ph = read_vec(&mut r, ncell, |rr| rr.i(22).map(|v| v as i32))?;
                let lock = read_vec(&mut r, ncell, |rr| rr.u(4).map(|v| v as u16))?;
                let half = read_vec(&mut r, ncell, |rr| rr.flag())?;
                let cnr = read_vec(&mut r, ncell, |rr| rr.u(6).map(|v| v as u16))?;
                cells
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
                        fine_phase_range_rate: None,
                    })
                    .collect()
            }
            MsmKind::Msm7 => {
                let fine_pr = read_vec(&mut r, ncell, |rr| rr.i(20).map(|v| v as i32))?;
                let fine_ph = read_vec(&mut r, ncell, |rr| rr.i(24).map(|v| v as i32))?;
                let lock = read_vec(&mut r, ncell, |rr| rr.u(10).map(|v| v as u16))?;
                let half = read_vec(&mut r, ncell, |rr| rr.flag())?;
                let cnr = read_vec(&mut r, ncell, |rr| rr.u(10).map(|v| v as u16))?;
                let fine_prr = read_vec(&mut r, ncell, |rr| rr.i(15).map(|v| v as i16))?;
                cells
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
                        fine_phase_range_rate: Some(fine_prr[c]),
                    })
                    .collect()
            }
        };

        Ok(Self {
            message_number,
            system,
            kind,
            header,
            satellites,
            signals,
        })
    }

    /// Encode this message back into an MSM body (without the transport frame).
    pub fn encode(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.push_u(u64::from(self.message_number), 12);
        w.push_u(u64::from(self.header.reference_station_id), 12);
        w.push_u(u64::from(self.header.epoch_time), 30);
        w.push_flag(self.header.multiple_message);
        w.push_u(u64::from(self.header.iods), 3);
        w.push_u(u64::from(self.header.reserved), 7);
        w.push_u(u64::from(self.header.clock_steering), 2);
        w.push_u(u64::from(self.header.external_clock), 2);
        w.push_flag(self.header.divergence_free_smoothing);
        w.push_u(u64::from(self.header.smoothing_interval), 3);

        // Reconstruct the satellite ids (sorted) and the satellite mask.
        let mut sat_ids: Vec<u8> = self.satellites.iter().map(|s| s.id).collect();
        sat_ids.sort_unstable();
        let mut satellite_mask: u64 = 0;
        for &id in &sat_ids {
            satellite_mask |= 1u64 << (64 - u32::from(id));
        }

        // Signal ids = sorted union of signals referenced by the cells.
        let mut sig_ids: Vec<u8> = self.signals.iter().map(|s| s.signal_id).collect();
        sig_ids.sort_unstable();
        sig_ids.dedup();
        let mut signal_mask: u32 = 0;
        for &id in &sig_ids {
            signal_mask |= 1u32 << (32 - u32::from(id));
        }

        w.push_u(satellite_mask, 64);
        w.push_u(u64::from(signal_mask), 32);

        // Cell mask, satellite-major, plus the ordered active cell list.
        let mut ordered_cells: Vec<(u8, u8)> = Vec::new();
        for &sat in &sat_ids {
            for &sig in &sig_ids {
                let present = self
                    .signals
                    .iter()
                    .any(|s| s.satellite_id == sat && s.signal_id == sig);
                w.push_flag(present);
                if present {
                    ordered_cells.push((sat, sig));
                }
            }
        }

        // Satellite block, column-major, in the same sorted id order.
        let sat_by_id = |id: u8| self.satellites.iter().find(|s| s.id == id);
        for &id in &sat_ids {
            let sat = sat_by_id(id);
            w.push_u(u64::from(sat.map_or(0, |s| s.rough_range_ms)), 8);
        }
        if self.kind == MsmKind::Msm7 {
            for &id in &sat_ids {
                let ext = sat_by_id(id).and_then(|s| s.extended_info).unwrap_or(0);
                w.push_u(u64::from(ext), 4);
            }
        }
        for &id in &sat_ids {
            let sat = sat_by_id(id);
            w.push_u(u64::from(sat.map_or(0, |s| s.rough_range_mod1)), 10);
        }
        if self.kind == MsmKind::Msm7 {
            for &id in &sat_ids {
                let prr = sat_by_id(id)
                    .and_then(|s| s.rough_phase_range_rate_m_s)
                    .unwrap_or(0);
                w.push_i(i64::from(prr), 14);
            }
        }

        // Signal block, column-major over the ordered cells.
        let cell_signal = |sat: u8, sig: u8| {
            self.signals
                .iter()
                .find(|s| s.satellite_id == sat && s.signal_id == sig)
                .expect("ordered cell must reference an existing signal")
        };
        let ordered: Vec<&MsmSignal> = ordered_cells
            .iter()
            .map(|&(sat, sig)| cell_signal(sat, sig))
            .collect();

        match self.kind {
            MsmKind::Msm4 => {
                for s in &ordered {
                    w.push_i(i64::from(s.fine_pseudorange), 15);
                }
                for s in &ordered {
                    w.push_i(i64::from(s.fine_phase_range), 22);
                }
                for s in &ordered {
                    w.push_u(u64::from(s.lock_time_indicator), 4);
                }
                for s in &ordered {
                    w.push_flag(s.half_cycle_ambiguity);
                }
                for s in &ordered {
                    w.push_u(u64::from(s.cnr), 6);
                }
            }
            MsmKind::Msm7 => {
                for s in &ordered {
                    w.push_i(i64::from(s.fine_pseudorange), 20);
                }
                for s in &ordered {
                    w.push_i(i64::from(s.fine_phase_range), 24);
                }
                for s in &ordered {
                    w.push_u(u64::from(s.lock_time_indicator), 10);
                }
                for s in &ordered {
                    w.push_flag(s.half_cycle_ambiguity);
                }
                for s in &ordered {
                    w.push_u(u64::from(s.cnr), 10);
                }
                for s in &ordered {
                    w.push_i(i64::from(s.fine_phase_range_rate.unwrap_or(0)), 15);
                }
            }
        }

        w.into_bytes()
    }
}

/// Read `n` values with `f`, collecting into a vector.
fn read_vec<T>(
    r: &mut BitReader<'_>,
    n: usize,
    mut f: impl FnMut(&mut BitReader<'_>) -> Result<T>,
) -> Result<Vec<T>> {
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
