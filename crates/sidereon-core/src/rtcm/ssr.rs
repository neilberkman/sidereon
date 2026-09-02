//! RTCM SSR orbit, clock, URA, and high-rate clock messages.
//!
//! This module is the wire-level IR for the RTCM SSR Phase A messages. Values
//! are stored as the raw transmitted integers. Scaling to meters and seconds is
//! handled by the crate-level `ssr` correction store.

use crate::error::{Error, Result};
use crate::id::GnssSystem;

use super::bits::{BitReader, BitWriter};
use super::DecodeResult;

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
    /// Body bits after the parsed message fields.
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
    /// Decode one RTCM SSR body, without the transport frame.
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_inner(body).map_err(Into::into)
    }

    pub(crate) fn decode_inner(body: &[u8]) -> DecodeResult<Self> {
        let mut r = BitReader::new(body);
        let message_number = r.u(12)? as u16;
        let (system, kind) = ssr_kind(message_number).ok_or_else(|| {
            Error::Parse(format!(
                "message {message_number} is not a supported RTCM SSR Phase A type"
            ))
        })?;
        let header = read_header(&mut r, system, kind)?;
        let count = usize::from(header.satellite_count);
        let mut orbit = Vec::new();
        let mut clock = Vec::new();
        let mut code_bias = Vec::new();
        let mut phase_bias = Vec::new();
        let mut ura = Vec::new();

        match kind {
            SsrKind::Orbit => {
                orbit.reserve(count);
                for _ in 0..count {
                    orbit.push(read_orbit_record(&mut r, system)?);
                }
            }
            SsrKind::Clock => {
                clock.reserve(count);
                for _ in 0..count {
                    clock.push(read_clock_record(&mut r, system)?);
                }
            }
            SsrKind::CombinedOrbitClock => {
                orbit.reserve(count);
                clock.reserve(count);
                for _ in 0..count {
                    let rec = read_orbit_record(&mut r, system)?;
                    let satellite_id = rec.satellite_id;
                    orbit.push(rec);
                    clock.push(SsrClockRecord {
                        satellite_id,
                        c0: r.i(22)? as i32,
                        c1: r.i(21)? as i32,
                        c2: r.i(27)? as i32,
                    });
                }
            }
            SsrKind::Ura => {
                ura.reserve(count);
                for _ in 0..count {
                    let satellite_id = r.u(satellite_id_bits(system))? as u8;
                    let index = r.u(6)? as u8;
                    ura.push((satellite_id, index));
                }
            }
            SsrKind::HighRateClock => {
                clock.reserve(count);
                for _ in 0..count {
                    clock.push(SsrClockRecord {
                        satellite_id: r.u(satellite_id_bits(system))? as u8,
                        c0: r.i(22)? as i32,
                        c1: 0,
                        c2: 0,
                    });
                }
            }
            SsrKind::CodeBias => {
                code_bias.reserve(count);
                for _ in 0..count {
                    code_bias.push(read_code_bias_record(&mut r, system)?);
                }
            }
            SsrKind::PhaseBias => {
                phase_bias.reserve(count);
                for _ in 0..count {
                    phase_bias.push(read_phase_bias_record(&mut r, system)?);
                }
            }
            SsrKind::Vtec => {
                return Err(Error::Parse(format!(
                    "message {message_number} is not enabled in RTCM SSR Phase A"
                ))
                .into());
            }
        }

        let mut padding_bits = Vec::with_capacity(r.remaining_bits());
        while r.remaining_bits() > 0 {
            padding_bits.push(r.flag()?);
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
    pub fn encode(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.push_u(u64::from(self.message_number), 12);
        write_header(&mut w, self.system, &self.header, self.kind);

        match self.kind {
            SsrKind::Orbit => {
                for rec in &self.orbit {
                    write_orbit_record(&mut w, self.system, rec);
                }
            }
            SsrKind::Clock => {
                for rec in &self.clock {
                    write_clock_record(&mut w, self.system, rec);
                }
            }
            SsrKind::CombinedOrbitClock => {
                for (orbit, clock) in self.orbit.iter().zip(&self.clock) {
                    write_orbit_record(&mut w, self.system, orbit);
                    w.push_i(i64::from(clock.c0), 22);
                    w.push_i(i64::from(clock.c1), 21);
                    w.push_i(i64::from(clock.c2), 27);
                }
            }
            SsrKind::Ura => {
                for &(satellite_id, index) in &self.ura {
                    w.push_u(u64::from(satellite_id), satellite_id_bits(self.system));
                    w.push_u(u64::from(index), 6);
                }
            }
            SsrKind::HighRateClock => {
                for rec in &self.clock {
                    w.push_u(u64::from(rec.satellite_id), satellite_id_bits(self.system));
                    w.push_i(i64::from(rec.c0), 22);
                }
            }
            SsrKind::CodeBias => {
                for rec in &self.code_bias {
                    write_code_bias_record(&mut w, self.system, rec);
                }
            }
            SsrKind::PhaseBias => {
                for rec in &self.phase_bias {
                    write_phase_bias_record(&mut w, self.system, rec);
                }
            }
            SsrKind::Vtec => {}
        }

        for &bit in &self.padding_bits {
            w.push_flag(bit);
        }
        w.into_bytes()
    }
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
    let satellite_count = r.u(6)? as u8;
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

fn write_header(w: &mut BitWriter, system: GnssSystem, header: &SsrHeader, kind: SsrKind) {
    w.push_u(u64::from(header.epoch_time_s), epoch_time_bits(system));
    w.push_u(u64::from(header.update_interval), 4);
    w.push_flag(header.multiple_message);
    if matches!(kind, SsrKind::Orbit | SsrKind::CombinedOrbitClock) {
        w.push_flag(header.satellite_reference_datum.unwrap_or(false));
    }
    w.push_u(u64::from(header.iod_ssr), 4);
    w.push_u(u64::from(header.provider_id), 16);
    w.push_u(u64::from(header.solution_id), 4);
    if kind == SsrKind::PhaseBias {
        w.push_flag(header.dispersive_bias_consistency.unwrap_or(false));
        w.push_flag(header.mw_consistency.unwrap_or(false));
    }
    w.push_u(u64::from(header.satellite_count), 6);
}

fn read_orbit_record(r: &mut BitReader<'_>, system: GnssSystem) -> DecodeResult<SsrOrbitRecord> {
    Ok(SsrOrbitRecord {
        satellite_id: r.u(satellite_id_bits(system))? as u8,
        iode: r.u(iode_bits(system))? as u32,
        delta_radial: r.i(22)? as i32,
        delta_along: r.i(20)? as i32,
        delta_cross: r.i(20)? as i32,
        dot_delta_radial: r.i(21)? as i32,
        dot_delta_along: r.i(19)? as i32,
        dot_delta_cross: r.i(19)? as i32,
    })
}

fn write_orbit_record(w: &mut BitWriter, system: GnssSystem, rec: &SsrOrbitRecord) {
    w.push_u(u64::from(rec.satellite_id), satellite_id_bits(system));
    w.push_u(u64::from(rec.iode), iode_bits(system));
    w.push_i(i64::from(rec.delta_radial), 22);
    w.push_i(i64::from(rec.delta_along), 20);
    w.push_i(i64::from(rec.delta_cross), 20);
    w.push_i(i64::from(rec.dot_delta_radial), 21);
    w.push_i(i64::from(rec.dot_delta_along), 19);
    w.push_i(i64::from(rec.dot_delta_cross), 19);
}

fn read_clock_record(r: &mut BitReader<'_>, system: GnssSystem) -> DecodeResult<SsrClockRecord> {
    Ok(SsrClockRecord {
        satellite_id: r.u(satellite_id_bits(system))? as u8,
        c0: r.i(22)? as i32,
        c1: r.i(21)? as i32,
        c2: r.i(27)? as i32,
    })
}

fn write_clock_record(w: &mut BitWriter, system: GnssSystem, rec: &SsrClockRecord) {
    w.push_u(u64::from(rec.satellite_id), satellite_id_bits(system));
    w.push_i(i64::from(rec.c0), 22);
    w.push_i(i64::from(rec.c1), 21);
    w.push_i(i64::from(rec.c2), 27);
}

fn read_code_bias_record(
    r: &mut BitReader<'_>,
    system: GnssSystem,
) -> DecodeResult<SsrCodeBiasRecord> {
    let satellite_id = r.u(satellite_id_bits(system))? as u8;
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

fn write_code_bias_record(w: &mut BitWriter, system: GnssSystem, rec: &SsrCodeBiasRecord) {
    w.push_u(u64::from(rec.satellite_id), satellite_id_bits(system));
    w.push_u(rec.biases.len() as u64, 5);
    for &(signal_id, bias) in &rec.biases {
        w.push_u(u64::from(signal_id), 5);
        w.push_i(i64::from(bias), 14);
    }
}

fn read_phase_bias_record(
    r: &mut BitReader<'_>,
    system: GnssSystem,
) -> DecodeResult<SsrPhaseBiasRecord> {
    let satellite_id = r.u(satellite_id_bits(system))? as u8;
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

fn write_phase_bias_record(w: &mut BitWriter, system: GnssSystem, rec: &SsrPhaseBiasRecord) {
    w.push_u(u64::from(rec.satellite_id), satellite_id_bits(system));
    w.push_u(rec.biases.len() as u64, 5);
    w.push_u(u64::from(rec.yaw_angle), 9);
    w.push_i(i64::from(rec.yaw_rate), 8);
    for bias in &rec.biases {
        w.push_u(u64::from(bias.signal_id), 5);
        w.push_u(u64::from(bias.integer_indicator), 1);
        w.push_u(u64::from(bias.wide_lane_integer_indicator), 2);
        w.push_u(u64::from(bias.discontinuity_counter), 4);
        w.push_i(i64::from(bias.bias), 20);
    }
}

fn satellite_id_bits(system: GnssSystem) -> usize {
    match system {
        GnssSystem::Glonass => 5,
        _ => 6,
    }
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
        ] {
            let expected = message(number, system, kind);
            let body = expected.encode();
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
            assert_eq!(decoded.encode(), body, "message {number} round trip");
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
            encode_frame(&Message::Ssr(gal.clone()).encode()).unwrap(),
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
            encode_frame(&Message::Ssr(gps.clone()).encode()).unwrap(),
            gps_frame
        );
        assert_eq!(decode_frame(&gal_frame).unwrap().body, gal.encode());
        assert_eq!(decode_frame(&gps_frame).unwrap().body, gps.encode());
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
        let body = message(1057, GnssSystem::Gps, SsrKind::Orbit).encode();
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
        assert_eq!(decoded[0].as_ref().unwrap().encode(), message.encode());
        assert_eq!(encode_frame(&message.encode()).unwrap(), frame);
    }
}
