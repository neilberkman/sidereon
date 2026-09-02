//! Galileo HAS MT1 correction payload parser.
//!
//! This module parses the recovered HAS Message Type 1 payload, starting at the
//! 32-bit MT1 header and ending at the byte-aligned MT1 body. E6 page recovery,
//! Reed-Solomon reconstruction, and outer transport framing are intentionally
//! outside this module.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use crate::constants::{C_M_S, F_E1_HZ, F_E5A_HZ, F_L1_HZ, F_L2_HZ};
use crate::error::{Error, Result};
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::rtcm::bits::{BitReader, BitWriter};

const HAS_ORBIT_RADIAL_SCALE_M: f64 = 0.0025;
const HAS_ORBIT_ALONG_CROSS_SCALE_M: f64 = 0.0080;
const HAS_CLOCK_SCALE_M: f64 = 0.0025;
const HAS_CODE_BIAS_SCALE_M: f64 = 0.02;
const HAS_PHASE_BIAS_SCALE_CYCLES: f64 = 0.01;

/// HAS MT1 header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HasMt1Header {
    /// Time of hour in GST seconds.
    pub toh_s: u16,
    /// Mask content block present.
    pub mask: bool,
    /// Orbit correction content block present.
    pub orbit: bool,
    /// Clock full-set content block present.
    pub clock_full_set: bool,
    /// Clock subset content block present.
    pub clock_subset: bool,
    /// Code-bias content block present.
    pub code_bias: bool,
    /// Phase-bias content block present.
    pub phase_bias: bool,
    /// Reserved 4-bit field.
    pub reserved: u8,
    /// Mask identifier.
    pub mask_id: u8,
    /// IOD set identifier.
    pub iod_set_id: u8,
}

/// One GNSS mask inside an HAS MT1 mask block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HasGnssMask {
    /// Corrected constellation.
    pub system: GnssSystem,
    /// Corrected PRNs in HAS mask order.
    pub satellites: Vec<u8>,
    /// Corrected signal indices in HAS mask order.
    pub signals: Vec<u8>,
    /// Optional row-major satellite by signal cell mask.
    pub cell_mask: Option<Vec<bool>>,
    /// HAS navigation-message index.
    pub nav_message: u8,
}

/// HAS MT1 mask block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HasMaskBlock {
    /// GNSS-specific masks in transmitted order.
    pub systems: Vec<HasGnssMask>,
}

/// HAS MT1 orbit correction block.
#[derive(Clone, Debug, PartialEq)]
pub struct HasOrbitBlock {
    /// Validity interval index.
    pub validity_interval: u8,
    /// Orbit records in mask order, excluding unavailable records.
    pub records: Vec<HasOrbitCorrection>,
}

/// HAS orbit correction for one satellite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HasOrbitCorrection {
    /// Corrected satellite.
    pub sat: GnssSatelliteId,
    /// Reference navigation issue.
    pub iode: u32,
    /// Delta radial, meters, additive per HAS ICD.
    pub radial_m: f64,
    /// Delta in-track, meters, additive per HAS ICD.
    pub along_m: f64,
    /// Delta cross-track, meters, additive per HAS ICD.
    pub cross_m: f64,
}

/// HAS MT1 clock full-set or subset correction block.
#[derive(Clone, Debug, PartialEq)]
pub struct HasClockBlock {
    /// Validity interval index.
    pub validity_interval: u8,
    /// Clock records in mask order, excluding unavailable and do-not-use records.
    pub records: Vec<HasClockCorrection>,
}

/// HAS clock correction for one satellite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HasClockCorrection {
    /// Corrected satellite.
    pub sat: GnssSatelliteId,
    /// Delta clock in meters, additive per HAS ICD.
    pub correction_m: f64,
}

/// HAS MT1 code-bias block.
#[derive(Clone, Debug, PartialEq)]
pub struct HasCodeBiasBlock {
    /// Validity interval index.
    pub validity_interval: u8,
    /// Code-bias records.
    pub records: Vec<HasCodeBias>,
}

/// HAS code bias for one satellite signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HasCodeBias {
    /// Corrected satellite.
    pub sat: GnssSatelliteId,
    /// HAS signal index.
    pub signal_id: u8,
    /// Bias in meters. Add to the pseudorange observation per HAS ICD.
    pub bias_m: f64,
}

/// HAS MT1 phase-bias block.
#[derive(Clone, Debug, PartialEq)]
pub struct HasPhaseBiasBlock {
    /// Validity interval index.
    pub validity_interval: u8,
    /// Phase-bias records.
    pub records: Vec<HasPhaseBias>,
}

/// HAS phase bias for one satellite signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HasPhaseBias {
    /// Corrected satellite.
    pub sat: GnssSatelliteId,
    /// HAS signal index.
    pub signal_id: u8,
    /// Bias in cycles.
    pub bias_cycles: f64,
    /// Bias converted to meters using the HAS signal carrier.
    pub bias_m: f64,
    /// Phase discontinuity indicator.
    pub discontinuity_indicator: u8,
}

/// Decoded HAS MT1 payload.
#[derive(Clone, Debug, PartialEq)]
pub struct HasMt1Message {
    /// MT1 header.
    pub header: HasMt1Header,
    /// Mask block, when present.
    pub mask: Option<HasMaskBlock>,
    /// Orbit correction block, when present.
    pub orbit: Option<HasOrbitBlock>,
    /// Clock full-set correction block, when present.
    pub clock_full_set: Option<HasClockBlock>,
    /// Clock subset correction block, when present.
    pub clock_subset: Option<HasClockBlock>,
    /// Code-bias block, when present.
    pub code_bias: Option<HasCodeBiasBlock>,
    /// Phase-bias block, when present.
    pub phase_bias: Option<HasPhaseBiasBlock>,
    /// Remaining non-decoded padding or future-extension bits.
    pub padding_bits: Vec<bool>,
}

impl HasMt1Message {
    /// Decode one recovered HAS MT1 payload.
    pub fn decode(body: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(body);
        let header = read_header(&mut r)?;
        let mask = if header.mask {
            Some(read_mask_block(&mut r)?)
        } else {
            None
        };
        let Some(mask_ref) = mask.as_ref() else {
            if header.orbit
                || header.clock_full_set
                || header.clock_subset
                || header.code_bias
                || header.phase_bias
            {
                return Err(Error::Parse(
                    "HAS MT1 correction blocks require a mask in this stateless decoder"
                        .to_string(),
                ));
            }
            return Ok(Self {
                header,
                mask,
                orbit: None,
                clock_full_set: None,
                clock_subset: None,
                code_bias: None,
                phase_bias: None,
                padding_bits: read_padding_bits(&mut r)?,
            });
        };

        let orbit = if header.orbit {
            Some(read_orbit_block(&mut r, mask_ref)?)
        } else {
            None
        };
        let clock_full_set = if header.clock_full_set {
            Some(read_clock_full_set_block(&mut r, mask_ref)?)
        } else {
            None
        };
        let clock_subset = if header.clock_subset {
            Some(read_clock_subset_block(&mut r, mask_ref)?)
        } else {
            None
        };
        let code_bias = if header.code_bias {
            Some(read_code_bias_block(&mut r, mask_ref)?)
        } else {
            None
        };
        let phase_bias = if header.phase_bias {
            Some(read_phase_bias_block(&mut r, mask_ref)?)
        } else {
            None
        };

        Ok(Self {
            header,
            mask,
            orbit,
            clock_full_set,
            clock_subset,
            code_bias,
            phase_bias,
            padding_bits: read_padding_bits(&mut r)?,
        })
    }

    /// Encode this message back to an MT1 payload.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        write_header(&mut w, self.header);
        if let Some(mask) = &self.mask {
            write_mask_block(&mut w, mask);
        }
        if let Some(orbit) = &self.orbit {
            write_orbit_block(&mut w, orbit);
        }
        if let (Some(mask), Some(clock)) = (self.mask.as_ref(), self.clock_full_set.as_ref()) {
            write_clock_full_set_block(&mut w, mask, clock);
        }
        if let (Some(mask), Some(clock)) = (self.mask.as_ref(), self.clock_subset.as_ref()) {
            write_clock_subset_block(&mut w, mask, clock);
        }
        if let (Some(mask), Some(code_bias)) = (self.mask.as_ref(), self.code_bias.as_ref()) {
            write_code_bias_block(&mut w, mask, code_bias);
        }
        if let (Some(mask), Some(phase_bias)) = (self.mask.as_ref(), self.phase_bias.as_ref()) {
            write_phase_bias_block(&mut w, mask, phase_bias);
        }
        for &bit in &self.padding_bits {
            w.push_flag(bit);
        }
        w.into_bytes()
    }
}

/// Resolve an HAS MT1 TOH to a J2000 epoch using the GST reception time.
pub fn has_mt1_reference_j2000_s(
    reception_gst: crate::astro::time::model::GnssWeekTow,
    toh_s: u16,
) -> Result<f64> {
    if reception_gst.system != crate::astro::time::model::TimeScale::Gst {
        return Err(Error::InvalidInput(
            "HAS reception time must use GST".to_string(),
        ));
    }
    let reception_s =
        f64::from(reception_gst.week) * crate::constants::SECONDS_PER_WEEK + reception_gst.tow_s;
    let hour = (reception_s / crate::constants::SECONDS_PER_HOUR).floor();
    let candidate = hour * crate::constants::SECONDS_PER_HOUR + f64::from(toh_s);
    let gst_s = if candidate <= reception_s {
        candidate
    } else {
        candidate - crate::constants::SECONDS_PER_HOUR
    };
    Ok(gst_s - crate::constants::GPS_EPOCH_TO_J2000_S)
}

/// HAS validity interval in seconds for a VI index.
pub const fn has_validity_interval_s(index: u8) -> Option<f64> {
    match index {
        0 => Some(5.0),
        1 => Some(10.0),
        2 => Some(15.0),
        3 => Some(20.0),
        4 => Some(30.0),
        5 => Some(60.0),
        6 => Some(90.0),
        7 => Some(120.0),
        8 => Some(180.0),
        9 => Some(240.0),
        10 => Some(300.0),
        11 => Some(600.0),
        12 => Some(900.0),
        13 => Some(1800.0),
        14 => Some(3600.0),
        _ => None,
    }
}

fn read_header(r: &mut BitReader<'_>) -> Result<HasMt1Header> {
    Ok(HasMt1Header {
        toh_s: r.u(12)? as u16,
        mask: r.flag()?,
        orbit: r.flag()?,
        clock_full_set: r.flag()?,
        clock_subset: r.flag()?,
        code_bias: r.flag()?,
        phase_bias: r.flag()?,
        reserved: r.u(4)? as u8,
        mask_id: r.u(5)? as u8,
        iod_set_id: r.u(5)? as u8,
    })
}

fn write_header(w: &mut BitWriter, header: HasMt1Header) {
    w.push_u(u64::from(header.toh_s), 12);
    w.push_flag(header.mask);
    w.push_flag(header.orbit);
    w.push_flag(header.clock_full_set);
    w.push_flag(header.clock_subset);
    w.push_flag(header.code_bias);
    w.push_flag(header.phase_bias);
    w.push_u(u64::from(header.reserved), 4);
    w.push_u(u64::from(header.mask_id), 5);
    w.push_u(u64::from(header.iod_set_id), 5);
}

fn read_mask_block(r: &mut BitReader<'_>) -> Result<HasMaskBlock> {
    let nsys = r.u(4)? as usize;
    if nsys == 0 {
        return Err(Error::Parse(
            "HAS MT1 mask Nsys is reserved value 0".to_string(),
        ));
    }
    let mut systems = Vec::with_capacity(nsys);
    for _ in 0..nsys {
        let gnss_id = r.u(4)? as u8;
        let system = has_gnss_system(gnss_id)?;
        let sat_mask = r.u(40)?;
        let signal_mask = r.u(16)? as u16;
        let cell_mask_available = r.flag()?;
        let satellites = mask_indices(sat_mask, 40)
            .into_iter()
            .map(|idx| idx + 1)
            .collect::<Vec<_>>();
        let signals = mask_indices(u64::from(signal_mask), 16);
        let cell_mask = if cell_mask_available {
            let mut cells = Vec::with_capacity(satellites.len() * signals.len());
            for _ in 0..satellites.len() * signals.len() {
                cells.push(r.flag()?);
            }
            Some(cells)
        } else {
            None
        };
        let nav_message = r.u(3)? as u8;
        systems.push(HasGnssMask {
            system,
            satellites,
            signals,
            cell_mask,
            nav_message,
        });
    }
    Ok(HasMaskBlock { systems })
}

fn write_mask_block(w: &mut BitWriter, mask: &HasMaskBlock) {
    w.push_u(mask.systems.len() as u64, 4);
    for system in &mask.systems {
        w.push_u(u64::from(has_gnss_id(system.system).unwrap_or(0)), 4);
        let sat_mask = mask_from_indices(system.satellites.iter().map(|prn| prn - 1), 40);
        let signal_mask = mask_from_indices(system.signals.iter().copied(), 16);
        w.push_u(sat_mask, 40);
        w.push_u(signal_mask, 16);
        w.push_flag(system.cell_mask.is_some());
        if let Some(cells) = &system.cell_mask {
            for &cell in cells {
                w.push_flag(cell);
            }
        }
        w.push_u(u64::from(system.nav_message), 3);
    }
}

fn read_orbit_block(r: &mut BitReader<'_>, mask: &HasMaskBlock) -> Result<HasOrbitBlock> {
    let validity_interval = r.u(4)? as u8;
    if has_validity_interval_s(validity_interval).is_none() {
        return Err(Error::Parse("HAS orbit VI is reserved".to_string()));
    }
    let mut records = Vec::new();
    for sat in mask_satellites(mask)? {
        let iode = r.u(iode_bits(sat.system))? as u32;
        let radial = r.i(13)? as i16;
        let along = r.i(12)? as i16;
        let cross = r.i(12)? as i16;
        if radial == -(1 << 12) || along == -(1 << 11) || cross == -(1 << 11) {
            continue;
        }
        records.push(HasOrbitCorrection {
            sat,
            iode,
            radial_m: f64::from(radial) * HAS_ORBIT_RADIAL_SCALE_M,
            along_m: f64::from(along) * HAS_ORBIT_ALONG_CROSS_SCALE_M,
            cross_m: f64::from(cross) * HAS_ORBIT_ALONG_CROSS_SCALE_M,
        });
    }
    Ok(HasOrbitBlock {
        validity_interval,
        records,
    })
}

fn write_orbit_block(w: &mut BitWriter, orbit: &HasOrbitBlock) {
    w.push_u(u64::from(orbit.validity_interval), 4);
    for rec in &orbit.records {
        w.push_u(u64::from(rec.iode), iode_bits(rec.sat.system));
        w.push_i((rec.radial_m / HAS_ORBIT_RADIAL_SCALE_M).round() as i64, 13);
        w.push_i(
            (rec.along_m / HAS_ORBIT_ALONG_CROSS_SCALE_M).round() as i64,
            12,
        );
        w.push_i(
            (rec.cross_m / HAS_ORBIT_ALONG_CROSS_SCALE_M).round() as i64,
            12,
        );
    }
}

fn read_clock_full_set_block(r: &mut BitReader<'_>, mask: &HasMaskBlock) -> Result<HasClockBlock> {
    let validity_interval = r.u(4)? as u8;
    if has_validity_interval_s(validity_interval).is_none() {
        return Err(Error::Parse("HAS clock VI is reserved".to_string()));
    }
    let multipliers = read_clock_multipliers(r, mask.systems.len())?;
    let mut records = Vec::new();
    for (system_idx, system_mask) in mask.systems.iter().enumerate() {
        let multiplier = multipliers[system_idx];
        for &prn in &system_mask.satellites {
            let dcc = r.i(13)? as i16;
            if let Some(correction_m) = has_clock_value_m(dcc, multiplier) {
                records.push(HasClockCorrection {
                    sat: has_satellite(system_mask.system, prn)?,
                    correction_m,
                });
            }
        }
    }
    Ok(HasClockBlock {
        validity_interval,
        records,
    })
}

fn write_clock_full_set_block(w: &mut BitWriter, mask: &HasMaskBlock, clock: &HasClockBlock) {
    w.push_u(u64::from(clock.validity_interval), 4);
    for _ in &mask.systems {
        w.push_u(0, 2);
    }
    for system_mask in &mask.systems {
        for &prn in &system_mask.satellites {
            let Ok(sat) = has_satellite(system_mask.system, prn) else {
                continue;
            };
            let raw = clock
                .records
                .iter()
                .find(|record| record.sat == sat)
                .map(|record| (record.correction_m / HAS_CLOCK_SCALE_M).round() as i64)
                .unwrap_or(-(1 << 12));
            w.push_i(raw, 13);
        }
    }
}

fn read_clock_subset_block(r: &mut BitReader<'_>, mask: &HasMaskBlock) -> Result<HasClockBlock> {
    let validity_interval = r.u(4)? as u8;
    if has_validity_interval_s(validity_interval).is_none() {
        return Err(Error::Parse("HAS clock subset VI is reserved".to_string()));
    }
    let nsys_sub = r.u(4)? as usize;
    if nsys_sub == 0 {
        return Err(Error::Parse(
            "HAS clock subset Nsys_sub is reserved value 0".to_string(),
        ));
    }
    let mut records = Vec::new();
    for _ in 0..nsys_sub {
        let system = has_gnss_system(r.u(4)? as u8)?;
        let multiplier = dcm_multiplier(r.u(2)? as u8);
        let Some(system_mask) = mask.systems.iter().find(|m| m.system == system) else {
            return Err(Error::Parse(
                "HAS clock subset GNSS not in mask".to_string(),
            ));
        };
        let mut subset = Vec::with_capacity(system_mask.satellites.len());
        for _ in 0..system_mask.satellites.len() {
            subset.push(r.flag()?);
        }
        for (&present, &prn) in subset.iter().zip(&system_mask.satellites) {
            if !present {
                continue;
            }
            let dcc = r.i(13)? as i16;
            if let Some(correction_m) = has_clock_value_m(dcc, multiplier) {
                records.push(HasClockCorrection {
                    sat: has_satellite(system, prn)?,
                    correction_m,
                });
            }
        }
    }
    Ok(HasClockBlock {
        validity_interval,
        records,
    })
}

fn write_clock_subset_block(w: &mut BitWriter, mask: &HasMaskBlock, clock: &HasClockBlock) {
    w.push_u(u64::from(clock.validity_interval), 4);
    let systems = mask
        .systems
        .iter()
        .filter(|system_mask| {
            clock
                .records
                .iter()
                .any(|record| record.sat.system == system_mask.system)
        })
        .collect::<Vec<_>>();
    w.push_u(systems.len() as u64, 4);
    for system_mask in systems {
        w.push_u(u64::from(has_gnss_id(system_mask.system).unwrap_or(0)), 4);
        w.push_u(0, 2);
        let mut selected = Vec::with_capacity(system_mask.satellites.len());
        for &prn in &system_mask.satellites {
            let Ok(sat) = has_satellite(system_mask.system, prn) else {
                continue;
            };
            let present = clock.records.iter().any(|record| record.sat == sat);
            selected.push((sat, present));
            w.push_flag(present);
        }
        for (sat, present) in selected {
            if present {
                let raw = clock
                    .records
                    .iter()
                    .find(|record| record.sat == sat)
                    .map(|record| (record.correction_m / HAS_CLOCK_SCALE_M).round() as i64)
                    .unwrap_or(0);
                w.push_i(raw, 13);
            }
        }
    }
}

fn read_code_bias_block(r: &mut BitReader<'_>, mask: &HasMaskBlock) -> Result<HasCodeBiasBlock> {
    let validity_interval = r.u(4)? as u8;
    if has_validity_interval_s(validity_interval).is_none() {
        return Err(Error::Parse("HAS code-bias VI is reserved".to_string()));
    }
    let mut records = Vec::new();
    for cell in mask_cells(mask)? {
        let raw = r.i(11)? as i16;
        if raw == -(1 << 10) {
            continue;
        }
        records.push(HasCodeBias {
            sat: cell.sat,
            signal_id: cell.signal_id,
            bias_m: f64::from(raw) * HAS_CODE_BIAS_SCALE_M,
        });
    }
    Ok(HasCodeBiasBlock {
        validity_interval,
        records,
    })
}

fn write_code_bias_block(w: &mut BitWriter, mask: &HasMaskBlock, code_bias: &HasCodeBiasBlock) {
    w.push_u(u64::from(code_bias.validity_interval), 4);
    for cell in mask_cells(mask).unwrap_or_default() {
        let raw = code_bias
            .records
            .iter()
            .find(|record| record.sat == cell.sat && record.signal_id == cell.signal_id)
            .map(|record| (record.bias_m / HAS_CODE_BIAS_SCALE_M).round() as i64)
            .unwrap_or(-(1 << 10));
        w.push_i(raw, 11);
    }
}

fn read_phase_bias_block(r: &mut BitReader<'_>, mask: &HasMaskBlock) -> Result<HasPhaseBiasBlock> {
    let validity_interval = r.u(4)? as u8;
    if has_validity_interval_s(validity_interval).is_none() {
        return Err(Error::Parse("HAS phase-bias VI is reserved".to_string()));
    }
    let mut records = Vec::new();
    for cell in mask_cells(mask)? {
        let raw = r.i(11)? as i16;
        let discontinuity_indicator = r.u(2)? as u8;
        if raw == -(1 << 10) {
            continue;
        }
        let bias_cycles = f64::from(raw) * HAS_PHASE_BIAS_SCALE_CYCLES;
        let frequency_hz = has_signal_frequency_hz(cell.sat.system, cell.signal_id)?;
        records.push(HasPhaseBias {
            sat: cell.sat,
            signal_id: cell.signal_id,
            bias_cycles,
            bias_m: bias_cycles * C_M_S / frequency_hz,
            discontinuity_indicator,
        });
    }
    Ok(HasPhaseBiasBlock {
        validity_interval,
        records,
    })
}

fn write_phase_bias_block(w: &mut BitWriter, mask: &HasMaskBlock, phase_bias: &HasPhaseBiasBlock) {
    w.push_u(u64::from(phase_bias.validity_interval), 4);
    for cell in mask_cells(mask).unwrap_or_default() {
        if let Some(record) = phase_bias
            .records
            .iter()
            .find(|record| record.sat == cell.sat && record.signal_id == cell.signal_id)
        {
            w.push_i(
                (record.bias_cycles / HAS_PHASE_BIAS_SCALE_CYCLES).round() as i64,
                11,
            );
            w.push_u(u64::from(record.discontinuity_indicator), 2);
        } else {
            w.push_i(-(1 << 10), 11);
            w.push_u(0, 2);
        }
    }
}

fn read_clock_multipliers(r: &mut BitReader<'_>, count: usize) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(dcm_multiplier(r.u(2)? as u8));
    }
    Ok(out)
}

const fn dcm_multiplier(value: u8) -> f64 {
    match value {
        0 => 1.0,
        1 => 2.0,
        2 => 3.0,
        _ => 4.0,
    }
}

fn has_clock_value_m(raw: i16, multiplier: f64) -> Option<f64> {
    if raw == -(1 << 12) || raw == (1 << 12) - 1 {
        None
    } else {
        Some(f64::from(raw) * HAS_CLOCK_SCALE_M * multiplier)
    }
}

#[derive(Clone, Copy)]
struct HasCell {
    sat: GnssSatelliteId,
    signal_id: u8,
}

fn mask_satellites(mask: &HasMaskBlock) -> Result<Vec<GnssSatelliteId>> {
    let mut out = Vec::new();
    for system in &mask.systems {
        for &prn in &system.satellites {
            out.push(has_satellite(system.system, prn)?);
        }
    }
    Ok(out)
}

fn mask_cells(mask: &HasMaskBlock) -> Result<Vec<HasCell>> {
    let mut out = Vec::new();
    for system in &mask.systems {
        for (sat_idx, &prn) in system.satellites.iter().enumerate() {
            let sat = has_satellite(system.system, prn)?;
            for (sig_idx, &signal_id) in system.signals.iter().enumerate() {
                let present = system
                    .cell_mask
                    .as_ref()
                    .map(|cells| cells[sat_idx * system.signals.len() + sig_idx])
                    .unwrap_or(true);
                if present {
                    out.push(HasCell { sat, signal_id });
                }
            }
        }
    }
    Ok(out)
}

fn read_padding_bits(r: &mut BitReader<'_>) -> Result<Vec<bool>> {
    let mut padding = Vec::with_capacity(r.remaining_bits());
    while r.remaining_bits() > 0 {
        padding.push(r.flag()?);
    }
    Ok(padding)
}

fn has_gnss_system(gnss_id: u8) -> Result<GnssSystem> {
    match gnss_id {
        0 => Ok(GnssSystem::Gps),
        2 => Ok(GnssSystem::Galileo),
        _ => Err(Error::Parse(format!("unsupported HAS GNSS id {gnss_id}"))),
    }
}

fn has_satellite(system: GnssSystem, prn: u8) -> Result<GnssSatelliteId> {
    GnssSatelliteId::new(system, prn)
        .map_err(|err| Error::Parse(format!("invalid HAS satellite {system:?} {prn}: {err}")))
}

fn has_gnss_id(system: GnssSystem) -> Option<u8> {
    match system {
        GnssSystem::Gps => Some(0),
        GnssSystem::Galileo => Some(2),
        _ => None,
    }
}

fn iode_bits(system: GnssSystem) -> usize {
    match system {
        GnssSystem::Galileo => 10,
        _ => 8,
    }
}

fn mask_indices(mask: u64, width: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for idx in 0..width {
        if ((mask >> (width - 1 - idx)) & 1) != 0 {
            out.push(idx as u8);
        }
    }
    out
}

fn mask_from_indices(indices: impl IntoIterator<Item = u8>, width: usize) -> u64 {
    let mut mask = 0_u64;
    for idx in indices {
        mask |= 1_u64 << (width - 1 - usize::from(idx));
    }
    mask
}

fn has_signal_frequency_hz(system: GnssSystem, signal_id: u8) -> Result<f64> {
    match (system, signal_id) {
        (GnssSystem::Gps, 0 | 3 | 4 | 5) => Ok(F_L1_HZ),
        (GnssSystem::Gps, 6..=9) => Ok(F_L2_HZ),
        (GnssSystem::Gps, 11..=13) => Ok(F_E5A_HZ),
        (GnssSystem::Galileo, 0..=2) => Ok(F_E1_HZ),
        (GnssSystem::Galileo, 3..=5) => Ok(F_E5A_HZ),
        (GnssSystem::Galileo, 6..=8) => Ok(1_207_140_000.0),
        (GnssSystem::Galileo, 9..=11) => Ok(1_191_795_000.0),
        (GnssSystem::Galileo, 12..=14) => Ok(1_278_750_000.0),
        _ => Err(Error::Parse(format!(
            "unsupported HAS signal {signal_id} for {system:?}"
        ))),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn mt1_mask_orbit_clock_and_bias_blocks_roundtrip() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 3).unwrap();
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: 1234,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 7,
                iod_set_id: 9,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![3],
                    signals: vec![0, 9],
                    cell_mask: None,
                    nav_message: 0,
                }],
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat,
                    iode: 42,
                    radial_m: 1.25,
                    along_m: -2.0,
                    cross_m: 3.0,
                }],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                records: vec![HasClockCorrection {
                    sat,
                    correction_m: -0.75,
                }],
            }),
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasCodeBias {
                        sat,
                        signal_id: 0,
                        bias_m: 0.24,
                    },
                    HasCodeBias {
                        sat,
                        signal_id: 9,
                        bias_m: -0.46,
                    },
                ],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasPhaseBias {
                        sat,
                        signal_id: 0,
                        bias_cycles: 1.25,
                        bias_m: 1.25 * C_M_S / F_L1_HZ,
                        discontinuity_indicator: 1,
                    },
                    HasPhaseBias {
                        sat,
                        signal_id: 9,
                        bias_cycles: -2.5,
                        bias_m: -2.5 * C_M_S / F_L2_HZ,
                        discontinuity_indicator: 2,
                    },
                ],
            }),
            padding_bits: Vec::new(),
        };

        let body = message.encode();
        let decoded = HasMt1Message::decode(&body).unwrap();
        assert_eq!(decoded, message);
    }

    #[test]
    fn toh_reference_uses_current_or_previous_gst_hour() {
        let reception = crate::astro::time::model::GnssWeekTow::new(
            crate::astro::time::model::TimeScale::Gst,
            2400,
            3605.0,
        )
        .unwrap();
        let same_hour = has_mt1_reference_j2000_s(reception, 5).unwrap();
        let expected_same = f64::from(reception.week) * crate::constants::SECONDS_PER_WEEK + 3605.0
            - crate::constants::GPS_EPOCH_TO_J2000_S;
        assert_eq!(same_hour.to_bits(), expected_same.to_bits());

        let previous_hour = has_mt1_reference_j2000_s(reception, 3599).unwrap();
        assert_eq!(previous_hour.to_bits(), (expected_same - 6.0).to_bits());
    }
}
