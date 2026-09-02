//! RINEX loss-of-lock indicator derivation from decoded RTCM MSM fields.
//!
//! RTCM MSM messages carry raw phase lock-time indicators (DF402 for MSM4,
//! DF407 for MSM7) plus the half-cycle ambiguity flag (DF420). RINEX stores
//! that state as an LLI digit attached to phase observations: bit 0 means loss
//! of lock is possible, and bit 1 means half-cycle ambiguity is possible. This
//! module keeps that mapping as a small sans-I/O layer over decoded MSM IR.

use std::collections::BTreeMap;

use crate::id::GnssSystem;

use super::{MsmKind, MsmMessage};

const WEEK_MS: u64 = 604_800_000;
const DAY_MS: u64 = 86_400_000;
const GLONASS_DAY_UNKNOWN: u32 = 7;
const GLONASS_DAY_SHIFT: u32 = 27;
const GLONASS_MS_MASK: u32 = (1 << GLONASS_DAY_SHIFT) - 1;

/// RINEX LLI bit 0: loss of lock, so a cycle slip is possible.
pub const LLI_LOSS_OF_LOCK: u8 = 0b001;

/// RINEX LLI bit 1: half-cycle ambiguity or half-cycle slip is possible.
pub const LLI_HALF_CYCLE: u8 = 0b010;

/// The tracked previous observation of one RTCM MSM signal cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreviousLock {
    /// Previous minimum continuous-lock time in milliseconds.
    ///
    /// `None` means the previous indicator was reserved or otherwise did not
    /// have a defined minimum lock time.
    pub min_lock_time_ms: Option<u32>,
    /// Elapsed milliseconds between the previous and current observation of the
    /// same `(system, satellite, signal)` cell.
    pub elapsed_ms: u64,
}

/// Derived RINEX LLI for one MSM signal cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellLli {
    /// Satellite id, as carried by [`super::MsmSignal::satellite_id`].
    pub satellite_id: u8,
    /// Signal id, as carried by [`super::MsmSignal::signal_id`].
    pub signal_id: u8,
    /// Derived RINEX LLI value. Only bits 0 and 1 are set by this module.
    pub lli: u8,
    /// Current normalized minimum lock time in milliseconds.
    ///
    /// `None` means the current indicator is reserved or outside the defined
    /// range.
    pub min_lock_time_ms: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CellKey {
    system: GnssSystem,
    satellite_id: u8,
    signal_id: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CellState {
    raw_epoch_time: u32,
    min_lock_time_ms: Option<u32>,
}

/// Tracks per-signal MSM lock history and derives RINEX LLI values.
///
/// State is keyed by `(constellation, satellite id, signal id)`, so MSM4 and
/// MSM7 observations of the same signal share continuity while different
/// constellations never collide. Call [`Self::reset`] when a stream reconnects
/// or the caller intentionally starts a new continuity arc.
#[derive(Clone, Debug, Default)]
pub struct LockTimeTracker {
    cells: BTreeMap<CellKey, CellState>,
}

impl LockTimeTracker {
    /// Build an empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Derive LLI for every signal cell in `message` and advance tracker state.
    ///
    /// Output order matches `message.signals`. The tracker stores raw MSM epoch
    /// fields and computes elapsed time pairwise, including GPS-week rollover
    /// and GLONASS day-of-week handling. If the same cell repeats at the same
    /// epoch, LLI is derived but the stored state is not advanced, so a
    /// duplicate message cannot hide a later decrease.
    pub fn observe(&mut self, message: &MsmMessage) -> Vec<CellLli> {
        let mut out = Vec::with_capacity(message.signals.len());
        for signal in &message.signals {
            let key = CellKey {
                system: message.system,
                satellite_id: signal.satellite_id,
                signal_id: signal.signal_id,
            };
            let min_lock_time_ms = signal.minimum_lock_time_ms(message.kind);
            let previous = self.cells.get(&key).map(|state| PreviousLock {
                min_lock_time_ms: state.min_lock_time_ms,
                elapsed_ms: msm_epoch_dt_ms(
                    message.system,
                    state.raw_epoch_time,
                    message.header.epoch_time,
                ),
            });
            let lli = derive_lli(previous, min_lock_time_ms, signal.half_cycle_ambiguity);
            out.push(CellLli {
                satellite_id: signal.satellite_id,
                signal_id: signal.signal_id,
                lli,
                min_lock_time_ms,
            });
            if previous.is_none_or(|prev| prev.elapsed_ms != 0) {
                self.cells.insert(
                    key,
                    CellState {
                        raw_epoch_time: message.header.epoch_time,
                        min_lock_time_ms,
                    },
                );
            }
        }
        out
    }

    /// Drop all per-cell lock history.
    pub fn reset(&mut self) {
        self.cells.clear();
    }
}

/// Minimum continuous-lock time encoded by an MSM lock-time indicator.
///
/// MSM4 uses DF402, a 4-bit coarse bucket. MSM7 uses DF407, a 10-bit extended
/// bucket. Returns `None` for indicators outside the field's bit range and for
/// DF407 reserved values 705 through 1023.
pub fn minimum_lock_time_ms(kind: MsmKind, indicator: u16) -> Option<u32> {
    match kind {
        MsmKind::Msm4 => df402_minimum_lock_time_ms(indicator),
        MsmKind::Msm7 => df407_minimum_lock_time_ms(indicator),
    }
}

/// Derive the RINEX LLI digit for one signal cell.
///
/// Bit 0 is set when loss of lock is possible under the lock-time continuity
/// rules. Bit 1 is the current half-cycle ambiguity flag, copied verbatim from
/// DF420. Bit 2 is never set here because MSM does not carry the RINEX tracking
/// mode represented by that bit.
pub fn derive_lli(
    previous: Option<PreviousLock>,
    current_min_lock_ms: Option<u32>,
    half_cycle_ambiguity: bool,
) -> u8 {
    let mut lli = if half_cycle_ambiguity {
        LLI_HALF_CYCLE
    } else {
        0
    };

    let Some(previous) = previous else {
        return lli;
    };
    let Some(current) = current_min_lock_ms else {
        return lli | LLI_LOSS_OF_LOCK;
    };

    let decreased = previous
        .min_lock_time_ms
        .is_some_and(|previous_min| current < previous_min);
    let uncovered_gap = (current as u64) < previous.elapsed_ms;
    if decreased || uncovered_gap {
        lli |= LLI_LOSS_OF_LOCK;
    }
    lli
}

/// Elapsed milliseconds between two raw MSM epoch-time fields of one system.
///
/// Non-GLONASS MSM epoch fields are milliseconds of week. GLONASS carries a
/// 3-bit day-of-week plus millisecond-of-day field; when either day is the
/// unknown value `7`, the subtraction falls back to day modulo arithmetic.
pub fn msm_epoch_dt_ms(system: GnssSystem, previous: u32, current: u32) -> u64 {
    if system == GnssSystem::Glonass {
        let prev_day = previous >> GLONASS_DAY_SHIFT;
        let now_day = current >> GLONASS_DAY_SHIFT;
        let prev_ms = u64::from(previous & GLONASS_MS_MASK);
        let now_ms = u64::from(current & GLONASS_MS_MASK);
        if prev_day == GLONASS_DAY_UNKNOWN || now_day == GLONASS_DAY_UNKNOWN {
            modulo_elapsed_ms(prev_ms, now_ms, DAY_MS)
        } else {
            let prev = u64::from(prev_day) * DAY_MS + prev_ms;
            let now = u64::from(now_day) * DAY_MS + now_ms;
            modulo_elapsed_ms(prev, now, WEEK_MS)
        }
    } else {
        modulo_elapsed_ms(u64::from(previous), u64::from(current), WEEK_MS)
    }
}

/// RINEX 3 observation-code suffix for an MSM signal-mask id.
///
/// The phase observation code is `"L" <> suffix`; for example GPS signal id 2
/// maps to suffix `"1C"` and phase observable `L1C`. Returns `None` for
/// reserved signal ids.
pub fn msm_signal_rinex_code(system: GnssSystem, signal_id: u8) -> Option<&'static str> {
    signal_table(system)
        .get(usize::from(signal_id))
        .copied()
        .flatten()
}

fn df402_minimum_lock_time_ms(indicator: u16) -> Option<u32> {
    match indicator {
        0 => Some(0),
        1..=15 => Some(1u32 << (u32::from(indicator) + 4)),
        _ => None,
    }
}

fn df407_minimum_lock_time_ms(indicator: u16) -> Option<u32> {
    let n = u32::from(indicator);
    match n {
        0..=63 => Some(n),
        64..=703 => {
            let segment = (n - 64) / 32;
            let start = 64 + segment * 32;
            let scale = 1u32 << (segment + 1);
            let start_value = scale * start - scale * 32 * (segment + 1);
            Some(start_value + scale * (n - start))
        }
        704 => Some(67_108_864),
        _ => None,
    }
}

fn modulo_elapsed_ms(previous: u64, current: u64, modulus: u64) -> u64 {
    (current + modulus - (previous % modulus)) % modulus
}

fn signal_table(system: GnssSystem) -> &'static [Option<&'static str>; 33] {
    match system {
        GnssSystem::Gps => &GPS_SIGNALS,
        GnssSystem::Glonass => &GLONASS_SIGNALS,
        GnssSystem::Galileo => &GALILEO_SIGNALS,
        GnssSystem::Sbas => &SBAS_SIGNALS,
        GnssSystem::Qzss => &QZSS_SIGNALS,
        GnssSystem::BeiDou => &BEIDOU_SIGNALS,
        GnssSystem::Navic => &NAVIC_SIGNALS,
    }
}

const N: Option<&str> = None;

#[rustfmt::skip]
const GPS_SIGNALS: [Option<&str>; 33] = [
    N,
    N, Some("1C"), Some("1P"), Some("1W"), N, N, N,
    Some("2C"), Some("2P"), Some("2W"), N, N, N, N,
    Some("2S"), Some("2L"), Some("2X"), N, N, N, N,
    Some("5I"), Some("5Q"), Some("5X"), N, N, N, N, N,
    Some("1S"), Some("1L"), Some("1X"),
];

#[rustfmt::skip]
const GLONASS_SIGNALS: [Option<&str>; 33] = [
    N,
    N, Some("1C"), Some("1P"), N, N, N, N,
    Some("2C"), Some("2P"), N, N, N, N, Some("3I"),
    Some("3Q"), Some("3X"), N, Some("4A"), Some("4B"), Some("4X"), N,
    Some("6A"), Some("6B"), Some("6X"), N, N, N, N, N,
    N, N, N,
];

#[rustfmt::skip]
const GALILEO_SIGNALS: [Option<&str>; 33] = [
    N,
    N, Some("1C"), Some("1A"), Some("1B"), Some("1X"), Some("1Z"), N,
    Some("6C"), Some("6A"), Some("6B"), Some("6X"), Some("6Z"), N,
    Some("7I"), Some("7Q"), Some("7X"), N, Some("8I"), Some("8Q"), Some("8X"), N,
    Some("5I"), Some("5Q"), Some("5X"), N, N, N, N, N,
    N, N, N,
];

#[rustfmt::skip]
const SBAS_SIGNALS: [Option<&str>; 33] = [
    N,
    N, Some("1C"), N, N, N, N, N,
    N, N, N, N, N, N, N,
    N, N, N, N, N, N, N,
    Some("5I"), Some("5Q"), Some("5X"), N, N, N, N, N,
    N, N, N,
];

#[rustfmt::skip]
const QZSS_SIGNALS: [Option<&str>; 33] = [
    N,
    N, Some("1C"), N, N, Some("1E"), Some("1Z"), Some("1B"),
    N, Some("6S"), Some("6L"), Some("6X"), Some("6E"), Some("6Z"), N,
    Some("2S"), Some("2L"), Some("2X"), N, N, N, N,
    Some("5I"), Some("5Q"), Some("5X"), Some("5D"), Some("5P"), Some("5Z"), N, N,
    Some("1S"), Some("1L"), Some("1X"),
];

#[rustfmt::skip]
const BEIDOU_SIGNALS: [Option<&str>; 33] = [
    N,
    N, Some("2I"), Some("2Q"), Some("2X"), Some("1S"), Some("1L"), Some("1Z"),
    Some("6I"), Some("6Q"), Some("6X"), Some("6D"), Some("6P"), Some("6Z"), Some("7I"),
    Some("7Q"), Some("7X"), N, Some("8D"), Some("8P"), Some("8X"), N,
    Some("5D"), Some("5P"), Some("5X"), Some("7D"), Some("7P"), Some("7Z"), N, N,
    Some("1D"), Some("1P"), Some("1X"),
];

#[rustfmt::skip]
const NAVIC_SIGNALS: [Option<&str>; 33] = [
    N,
    N, Some("1D"), Some("1P"), Some("1X"), N, N, N,
    Some("9A"), Some("9B"), Some("9C"), Some("9X"), N, N, N,
    N, N, N, N, N, N, N,
    Some("5A"), Some("5B"), Some("5C"), Some("5X"), N, N, N, N,
    N, N, N,
];

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn df407_boundaries_are_monotone_and_pinned() {
        let pinned = [
            (0, Some(0)),
            (1, Some(1)),
            (63, Some(63)),
            (64, Some(64)),
            (95, Some(126)),
            (96, Some(128)),
            (127, Some(252)),
            (128, Some(256)),
            (159, Some(504)),
            (160, Some(512)),
            (191, Some(1008)),
            (192, Some(1024)),
            (223, Some(2016)),
            (224, Some(2048)),
            (255, Some(4032)),
            (256, Some(4096)),
            (287, Some(8064)),
            (288, Some(8192)),
            (319, Some(16128)),
            (320, Some(16384)),
            (351, Some(32256)),
            (352, Some(32768)),
            (383, Some(64512)),
            (384, Some(65536)),
            (415, Some(129024)),
            (416, Some(131072)),
            (447, Some(258048)),
            (448, Some(262144)),
            (479, Some(516096)),
            (480, Some(524288)),
            (511, Some(1032192)),
            (512, Some(1048576)),
            (543, Some(2064384)),
            (544, Some(2097152)),
            (575, Some(4128768)),
            (576, Some(4194304)),
            (607, Some(8257536)),
            (608, Some(8388608)),
            (639, Some(16515072)),
            (640, Some(16777216)),
            (671, Some(33030144)),
            (672, Some(33554432)),
            (703, Some(66060288)),
            (704, Some(67108864)),
            (705, None),
            (800, None),
            (1023, None),
        ];
        for (indicator, expected) in pinned {
            assert_eq!(
                minimum_lock_time_ms(MsmKind::Msm7, indicator),
                expected,
                "DF407 {indicator}"
            );
        }
        for indicator in 1..=704 {
            let prev = minimum_lock_time_ms(MsmKind::Msm7, indicator - 1).unwrap();
            let now = minimum_lock_time_ms(MsmKind::Msm7, indicator).unwrap();
            assert!(now > prev, "DF407 must increase at {indicator}");
        }

        let segment_starts = [
            64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608,
            640, 672, 704,
        ];
        for start in segment_starts {
            let previous_step = if start == 64 {
                1
            } else {
                1u32 << ((start - 64) / 32)
            };
            let joined = minimum_lock_time_ms(MsmKind::Msm7, start - 1).unwrap() + previous_step;
            assert_eq!(
                minimum_lock_time_ms(MsmKind::Msm7, start).unwrap(),
                joined,
                "DF407 segment join at {start}"
            );
        }
    }
}
