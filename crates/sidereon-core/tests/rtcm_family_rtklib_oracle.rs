//! RTKLIB decoding oracle for the RTCM 3 message families.
//!
//! Every stream in `tests/fixtures/rtcm/families/` has a `.rtklib.jsonl`
//! companion: for each CRC-valid frame, what RTKLIB demo5 (commit
//! 75a2e56275485b21a67bd35bc94bbeb8936e1a74) `decode_rtcm3` returns and stores,
//! written by `fixtures-generators/rtklib_rtcm_oracle/generate.sh`. Each floating
//! value there is the hex of its IEEE 754 bits.
//!
//! The tests decode every frame here under the strict policy, re-encode it to
//! its exact body, and recompute from the decoded raw integers each value
//! RTKLIB stores, with RTKLIB's constants and in its order of operations, and
//! compare the bits. A wrong bit in any decoded field changes the value.
//! Where RTKLIB decodes only a header (MSM1, MSM2, MSM3), the header is compared
//! and the fields are checked against the MSM4 of the same system and epoch,
//! whose fields RTKLIB decodes.
//!
//! Streams:
//!
//! - `rtk2go_*.rtcm3`: real streams captured from the public rtk2go.com NTRIP
//!   caster on 2026-09-24 between 04:26:37 and 04:26:57 UTC (mountpoints
//!   `TiftGA`, `Ormalingen_Ribi`, `sejongnav`, `Mirmenhof`, `granthamall` and
//!   `jacksbay`), reduced to the frames of the family each file names, byte for
//!   byte.
//! - `rtklib_testglo_legacy.rtcm3`: the first 60 frames (1004 and 1012) of
//!   RTKLIB's test stream `test/data/rcvraw/testglo.rtcm3`, byte for byte.
//! - `rtklib_encoded_msm1_to_msm4.rtcm3`: RTKLIB's encoder output (`encode-msm`)
//!   from the MSM7 observations of RTKLIB's test stream
//!   `test/data/rcvraw/GMSD7_20121014.rtcm3`, first 12 epochs.
//! - `rtk2go_1013.rtcm3`: the 1013 frames of the `sejongnav` capture.
//! - `text_1029.rtcm3`: 1029 frames written by
//!   `fixtures-generators/pyrtcm_layouts/generate_network_frames.py`.
//! - `rtk2go_1230.rtcm3`: the first two 1230 frames of each capture above and of
//!   `HEYT` and `FF-Malar`, byte for byte.
//! - `bnc_encoded_ssr.rtcm3`: BNC's encoder output, written by
//!   `fixtures-generators/bnc_ssr_oracle/generate.sh`: every RTCM SSR message
//!   1057..1270 and every IGS SSR subtype once.
//! - `rtklib_encoded_1041.rtcm3`: RTKLIB's encoder output (`encode-1041`), one
//!   1041 per NavIC record of `tests/fixtures/nav/BRDM00DLR_S_20262650000_01D_MN_navic.rnx`.
//! - `rtklib_encoded_4076.rtcm3`: RTKLIB's encoder output (`encode-4076`), the
//!   42 IGS SSR satellite subtypes for each of the first two SSR epochs of
//!   `tests/fixtures/ssr/SSRA03IGS0_2026188140760_3epoch.rtcm3`, with the phase
//!   biases the generator states.
//! - `rtklib_encoded_legacy.rtcm3`: RTKLIB's encoder output
//!   (`encode-legacy`), 1001..1004 and 1009..1012 from the observations of
//!   `testglo.rtcm3`, first 12 epochs.

// The RTKLIB constants below are spelled as RTKLIB spells them; the nearest
// double to each decimal is what RTKLIB multiplies by.
#![allow(clippy::excessive_precision)]

use std::collections::{BTreeMap, HashMap};

use serde_json::Value;
use sidereon_core::rtcm::{
    self, FrameScanner, LegacyObservations, Message, MsmKind, MsmMessage,
    LEGACY_PHASE_RANGE_INVALID, LEGACY_PSEUDORANGE_DIFFERENCE_INVALID,
};
use sidereon_core::GnssSystem;

const FAMILIES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/rtcm/families/");

// RTKLIB constants, spelled as rtklib.h and rtcm3.c spell them: several of the
// `P2_*` decimals are not exactly the power of two they name.
const CLIGHT: f64 = 299792458.0;
const RANGE_MS: f64 = CLIGHT * 0.001;
const P2_10: f64 = 0.0009765625;
const P2_24: f64 = 5.960464477539063E-08;
const P2_29: f64 = 1.862645149230957E-09;
const P2_31: f64 = 4.656612873077393E-10;
const FREQL1: f64 = 1.57542E9;
const FREQL2: f64 = 1.22760E9;
const FREQE5B: f64 = 1.20714E9;
const FREQL5: f64 = 1.17645E9;
const FREQL6: f64 = 1.27875E9;
const FREQE5AB: f64 = 1.191795E9;
const FREQS: f64 = 2.492028E9;
const FREQ1_GLO: f64 = 1.60200E9;
const DFRQ1_GLO: f64 = 0.56250E6;
const FREQ2_GLO: f64 = 1.24600E9;
const DFRQ2_GLO: f64 = 0.43750E6;
const FREQ3_GLO: f64 = 1.202025E9;
const FREQ1A_GLO: f64 = 1.600995E9;
const FREQ2A_GLO: f64 = 1.248060E9;
const FREQ1_CMP: f64 = 1.561098E9;
const FREQ2_CMP: f64 = 1.20714E9;
const FREQ3_CMP: f64 = 1.26852E9;

/// One CRC-valid frame of a fixture stream, decoded, with RTKLIB's record.
struct Frame {
    offset: usize,
    message: Message,
    rtklib: Value,
}

/// Every frame of `name`, decoded strictly and checked to re-encode to its
/// body, zipped with the RTKLIB record of the same frame.
fn fixture(name: &str) -> Vec<Frame> {
    let bytes = std::fs::read(format!("{FAMILIES}{name}")).expect("read stream");
    let text = std::fs::read_to_string(format!("{FAMILIES}{name}.rtklib.jsonl"))
        .expect("read RTKLIB records");
    let records: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).expect("RTKLIB record"))
        .collect();
    let mut scanner = FrameScanner::new(&bytes);
    let frames: Vec<_> = scanner.by_ref().collect();
    assert_eq!(scanner.resync_bytes(), 0, "{name}");
    assert_eq!(
        frames.len(),
        records.len(),
        "{name}: one RTKLIB record per frame"
    );
    frames
        .into_iter()
        .zip(records)
        .map(|(frame, rtklib)| {
            let offset = frame.body.as_ptr() as usize - bytes.as_ptr() as usize - 3;
            assert_eq!(rtklib["offset"].as_u64(), Some(offset as u64), "{name}");
            let message = Message::decode(frame.body)
                .unwrap_or_else(|err| panic!("{name} frame at {offset}: {err}"));
            assert!(
                !matches!(message, Message::Unsupported(_)),
                "{name} frame at {offset}: message {} is not decoded",
                message.message_number()
            );
            assert_eq!(
                message.encode().expect("re-encode"),
                frame.body,
                "{name} frame at {offset}"
            );
            assert_eq!(
                rtklib["type"].as_u64(),
                Some(u64::from(message.message_number())),
                "{name} frame at {offset}"
            );
            Frame {
                offset,
                message,
                rtklib,
            }
        })
        .collect()
}

fn bits64(value: &Value) -> u64 {
    u64::from_str_radix(value.as_str().expect("hex bits"), 16).expect("hex bits")
}

fn bits32(value: &Value) -> u32 {
    u32::from_str_radix(value.as_str().expect("hex bits"), 16).expect("hex bits")
}

/// The time of week RTKLIB `time2gpst` gives back for a time of week `tow`
/// that `adjweek` placed in the receiver's week: the whole seconds, reduced to
/// the week `time2gpst` counts them in, plus the fraction `gpst2time` split
/// off. A BeiDou epoch 14 s before the end of the week lands on the next week.
fn rtklib_tow(tow: f64) -> f64 {
    let whole = tow.trunc();
    whole % 604_800.0 + (tow - whole)
}

/// RTKLIB `code2freq` for a RINEX 3 code suffix.
fn code2freq(system: GnssSystem, code: &str, fcn: i32) -> f64 {
    let band = code.as_bytes()[0];
    match (system, band) {
        (GnssSystem::Gps, b'1') | (GnssSystem::Qzss, b'1') => FREQL1,
        (GnssSystem::Gps, b'2') | (GnssSystem::Qzss, b'2') => FREQL2,
        (GnssSystem::Gps, b'5') | (GnssSystem::Qzss, b'5') => FREQL5,
        (GnssSystem::Qzss, b'6') => FREQL6,
        (GnssSystem::Glonass, b'1') if (-7..=6).contains(&fcn) => {
            FREQ1_GLO + DFRQ1_GLO * f64::from(fcn)
        }
        (GnssSystem::Glonass, b'2') if (-7..=6).contains(&fcn) => {
            FREQ2_GLO + DFRQ2_GLO * f64::from(fcn)
        }
        (GnssSystem::Glonass, b'3') => FREQ3_GLO,
        (GnssSystem::Glonass, b'4') => FREQ1A_GLO,
        (GnssSystem::Glonass, b'6') => FREQ2A_GLO,
        (GnssSystem::Galileo, b'1') => FREQL1,
        (GnssSystem::Galileo, b'7') => FREQE5B,
        (GnssSystem::Galileo, b'5') => FREQL5,
        (GnssSystem::Galileo, b'6') => FREQL6,
        (GnssSystem::Galileo, b'8') => FREQE5AB,
        (GnssSystem::Sbas, b'1') => FREQL1,
        (GnssSystem::Sbas, b'5') => FREQL5,
        (GnssSystem::BeiDou, b'2') => FREQ1_CMP,
        (GnssSystem::BeiDou, b'7') => FREQ2_CMP,
        (GnssSystem::BeiDou, b'5') => FREQL5,
        (GnssSystem::BeiDou, b'6') => FREQ3_CMP,
        (GnssSystem::BeiDou, b'1') => FREQL1,
        (GnssSystem::BeiDou, b'8') => FREQE5AB,
        (GnssSystem::Navic, b'5') => FREQL5,
        (GnssSystem::Navic, b'9') => FREQS,
        (GnssSystem::Navic, b'1') => FREQL1,
        _ => 0.0,
    }
}

/// RTKLIB's satellite name for an MSM satellite-mask number (`satno2id`).
fn msm_satellite_name(system: GnssSystem, id: u8) -> String {
    match system {
        GnssSystem::Gps => format!("G{id:02}"),
        GnssSystem::Glonass => format!("R{id:02}"),
        GnssSystem::Galileo => format!("E{id:02}"),
        // PRN 119 + id, named by the PRN itself.
        GnssSystem::Sbas => format!("{:03}", u16::from(id) + 119),
        GnssSystem::Qzss => format!("J{id:02}"),
        GnssSystem::BeiDou => format!("C{id:02}"),
        GnssSystem::Navic => format!("I{id:02}"),
    }
}

/// RTKLIB `lossoflock` state, keyed by satellite and observation slot.
#[derive(Default)]
struct LockState {
    lock: HashMap<(String, u64), u16>,
}

impl LockState {
    fn loss_of_lock(&mut self, satellite: &str, slot: u64, lock: u16) -> u8 {
        let previous = self
            .lock
            .insert((satellite.to_string(), slot), lock)
            .unwrap_or(0);
        u8::from((lock == 0 && previous == 0) || lock < previous)
    }
}

/// What the oracle comparison covered.
#[derive(Debug, Default)]
struct Coverage {
    frames: usize,
    header_only: usize,
    values: usize,
    not_stored: usize,
}

/// State `save_msm_obs` carries between messages: GLONASS channels learned
/// from MSM5/MSM7 extended info (`nav.glo_fcn`, channel + 8) and lock history.
#[derive(Default)]
struct MsmOracleState {
    glo_fcn: HashMap<u8, i32>,
    locks: LockState,
}

/// Compare one decoded MSM message with RTKLIB's record of the same frame.
fn check_msm(
    name: &str,
    frame: &Frame,
    msm: &MsmMessage,
    state: &mut MsmOracleState,
    coverage: &mut Coverage,
) {
    let at = format!("{name} frame at {} ({})", frame.offset, msm.message_number);
    let rtklib = &frame.rtklib;
    coverage.frames += 1;
    // The epoch RTKLIB placed: milliseconds of week (BeiDou: plus 14 s to GPS
    // time). GLONASS goes through UTC and is not compared here.
    if msm.system != GnssSystem::Glonass {
        let mut tow = f64::from(msm.header.epoch_time) * 0.001;
        if msm.system == GnssSystem::BeiDou {
            tow += 14.0;
        }
        assert_eq!(
            bits64(&rtklib["tow"]),
            rtklib_tow(tow).to_bits(),
            "{at}: epoch"
        );
    }
    assert_eq!(
        rtklib["staid"].as_u64(),
        Some(u64::from(msm.header.reference_station_id)),
        "{at}: station"
    );
    let stored = rtklib["obs"].as_array().expect("obs");
    if !msm.kind.carries_rough_range_ms() {
        // RTKLIB `decode_msm0` reads the header and stores no observation.
        assert!(stored.is_empty(), "{at}");
        coverage.header_only += 1;
        return;
    }
    let by_satellite: BTreeMap<&str, &Value> = stored
        .iter()
        .map(|entry| (entry["sat"].as_str().expect("sat"), &entry["sig"]))
        .collect();
    let extended = msm.kind.is_extended_resolution();
    let mut matched = 0usize;
    for satellite in &msm.satellites {
        let sat_name = msm_satellite_name(msm.system, satellite.id);
        let rough = satellite.rough_range_ms.expect("MSM4..MSM7 rough range");
        // decode_msm4..7: the range stays 0 for 255 and gets no remainder.
        let mut r = 0.0;
        if rough != 255 {
            r = f64::from(rough) * RANGE_MS;
        }
        if r != 0.0 {
            r += f64::from(satellite.rough_range_mod1) * P2_10 * RANGE_MS;
        }
        // `rate*1.0` in RTKLIB, exact.
        let rr = satellite.rough_phase_range_rate_m_s.map_or(0.0, f64::from);
        let mut fcn = 0;
        if msm.system == GnssSystem::Glonass {
            fcn = -8;
            match satellite.extended_info {
                Some(ex) if ex <= 13 => {
                    fcn = i32::from(ex) - 7;
                    state.glo_fcn.entry(satellite.id).or_insert(fcn + 8);
                }
                _ => {
                    if let Some(&stored_fcn) = state.glo_fcn.get(&satellite.id) {
                        fcn = stored_fcn - 8;
                    }
                }
            }
        }
        for signal in msm
            .signals
            .iter()
            .filter(|signal| signal.satellite_id == satellite.id)
        {
            let Some(code) = rtcm::msm_signal_rinex_code(msm.system, signal.signal_id) else {
                continue;
            };
            let entry = by_satellite.get(sat_name.as_str()).and_then(|sigs| {
                sigs.as_array()
                    .expect("sig")
                    .iter()
                    .find(|sig| sig["code"].as_str() == Some(code))
            });
            let Some(entry) = entry else {
                coverage.not_stored += 1;
                continue;
            };
            let fine_pr = signal.fine_pseudorange.expect("pseudorange");
            let fine_ph = signal.fine_phase_range.expect("phase range");
            let (pr, cp) = if extended {
                (
                    (fine_pr != -524288).then(|| f64::from(fine_pr) * P2_29 * RANGE_MS),
                    (fine_ph != -8388608).then(|| f64::from(fine_ph) * P2_31 * RANGE_MS),
                )
            } else {
                (
                    (fine_pr != -16384).then(|| f64::from(fine_pr) * P2_24 * RANGE_MS),
                    (fine_ph != -2097152).then(|| f64::from(fine_ph) * P2_29 * RANGE_MS),
                )
            };
            let freq = if fcn < -7 {
                0.0
            } else {
                code2freq(msm.system, code, fcn)
            };
            let p = match pr {
                Some(pr) if r != 0.0 => r + pr,
                _ => 0.0,
            };
            let l = match cp {
                Some(cp) if r != 0.0 => (r + cp) * freq / CLIGHT,
                _ => 0.0,
            };
            let d = match signal.fine_phase_range_rate {
                Some(rrv) if msm.kind.carries_phase_range_rate() => {
                    let rrf = f64::from(rrv) * 0.0001;
                    (-(rr + rrf) * freq / CLIGHT) as f32
                }
                _ => 0.0,
            };
            // MSM4/MSM5 store `cnr*1.0`, which is exact; MSM6/MSM7 `cnr*0.0625`.
            let cnr = f64::from(signal.cnr.expect("CNR"));
            let snr = (if extended { cnr * 0.0625 } else { cnr }) as f32;
            let slot = entry["slot"].as_u64().expect("slot");
            let lli = state.locks.loss_of_lock(
                &sat_name,
                slot,
                signal.lock_time_indicator.expect("lock"),
            ) + if signal.half_cycle_ambiguity.expect("half") {
                2
            } else {
                0
            };
            let cell = format!("{at} {sat_name} {code}");
            assert_eq!(bits64(&entry["P"]), p.to_bits(), "{cell}: P");
            assert_eq!(bits64(&entry["L"]), l.to_bits(), "{cell}: L");
            assert_eq!(bits32(&entry["D"]), d.to_bits(), "{cell}: D");
            assert_eq!(bits32(&entry["SNR"]), snr.to_bits(), "{cell}: SNR");
            assert_eq!(entry["LLI"].as_u64(), Some(u64::from(lli)), "{cell}: LLI");
            matched += 1;
            coverage.values += 1;
        }
    }
    let rtklib_cells: usize = stored
        .iter()
        .map(|entry| entry["sig"].as_array().expect("sig").len())
        .sum();
    assert_eq!(
        matched, rtklib_cells,
        "{at}: every value RTKLIB stored is checked"
    );
}

/// Check every MSM frame of `name` against RTKLIB, and each MSM1, MSM2 and
/// MSM3 against the MSM4 of its system and epoch in the same stream.
fn check_msm_stream(name: &str) -> Coverage {
    let frames = fixture(name);
    let mut state = MsmOracleState::default();
    let mut coverage = Coverage::default();
    let mut msm4: HashMap<(GnssSystem, u32), MsmMessage> = HashMap::new();
    for frame in &frames {
        let Message::Msm(msm) = &frame.message else {
            panic!("{name}: frame at {} is not MSM", frame.offset);
        };
        check_msm(name, frame, msm, &mut state, &mut coverage);
        if msm.kind == MsmKind::Msm4 {
            msm4.insert((msm.system, msm.header.epoch_time), msm.clone());
        }
    }
    for frame in &frames {
        let Message::Msm(msm) = &frame.message else {
            continue;
        };
        if msm.kind.carries_rough_range_ms() {
            continue;
        }
        let full = msm4
            .get(&(msm.system, msm.header.epoch_time))
            .unwrap_or_else(|| panic!("{name}: no MSM4 for frame at {}", frame.offset));
        assert_same_fields(name, frame.offset, msm, full);
    }
    coverage
}

/// An MSM1, MSM2 or MSM3 message holds the fields the MSM4 of its epoch
/// holds, where both carry them.
fn assert_same_fields(name: &str, offset: usize, msm: &MsmMessage, full: &MsmMessage) {
    let at = format!("{name} frame at {offset} ({})", msm.message_number);
    assert_eq!(msm.signal_mask, full.signal_mask, "{at}");
    assert_eq!(msm.satellites.len(), full.satellites.len(), "{at}");
    for (a, b) in msm.satellites.iter().zip(&full.satellites) {
        assert_eq!(a.id, b.id, "{at}");
        assert_eq!(
            a.rough_range_mod1, b.rough_range_mod1,
            "{at} satellite {}",
            a.id
        );
    }
    assert_eq!(msm.signals.len(), full.signals.len(), "{at}");
    for (a, b) in msm.signals.iter().zip(&full.signals) {
        let cell = format!("{at} satellite {} signal {}", a.satellite_id, a.signal_id);
        assert_eq!(
            (a.satellite_id, a.signal_id),
            (b.satellite_id, b.signal_id),
            "{cell}"
        );
        if msm.kind.carries_pseudorange() {
            assert_eq!(a.fine_pseudorange, b.fine_pseudorange, "{cell}");
        }
        if msm.kind.carries_phase_range() {
            assert_eq!(a.fine_phase_range, b.fine_phase_range, "{cell}");
            assert_eq!(a.lock_time_indicator, b.lock_time_indicator, "{cell}");
            assert_eq!(a.half_cycle_ambiguity, b.half_cycle_ambiguity, "{cell}");
        }
    }
}

/// Every family fixture decodes strictly, frame for frame, into a typed
/// message that re-encodes to the frame's body.
#[test]
fn every_family_fixture_round_trips() {
    let mut names: Vec<String> = std::fs::read_dir(FAMILIES)
        .expect("families directory")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .into_string()
                .expect("name")
        })
        .filter(|name| name.ends_with(".rtcm3"))
        .collect();
    names.sort();
    assert!(!names.is_empty());
    for name in names {
        assert!(!fixture(&name).is_empty(), "{name}");
    }
}

/// Real MSM3 and MSM4 of every system from one receiver, and RTKLIB-encoded
/// MSM1..MSM4: MSM4 values against RTKLIB bit for bit, MSM1..MSM3 headers
/// against RTKLIB and their fields against the MSM4 of the same epoch.
#[test]
fn msm1_to_msm4_match_rtklib() {
    for name in [
        "rtk2go_tiftga_msm3_msm4.rtcm3",
        "rtklib_encoded_msm1_to_msm4.rtcm3",
    ] {
        let coverage = check_msm_stream(name);
        eprintln!("{name}: {coverage:?}");
        assert!(coverage.header_only > 0, "{name}");
        assert!(coverage.values > 0, "{name}");
    }
}

/// Real MSM5 of every system RTKLIB decodes, from two receivers, bit for bit
/// against RTKLIB's pseudoranges, phases, Dopplers, CNR and LLI.
#[test]
fn msm5_matches_rtklib() {
    for name in [
        "rtk2go_ormalingen_msm5.rtcm3",
        "rtk2go_sejongnav_msm5.rtcm3",
    ] {
        let coverage = check_msm_stream(name);
        eprintln!("{name}: {coverage:?}");
        assert!(coverage.values > 0, "{name}");
    }
}

/// Real MSM6 of every system, bit for bit against RTKLIB.
#[test]
fn msm6_matches_rtklib() {
    let name = "rtk2go_mirmenhof_msm6.rtcm3";
    let coverage = check_msm_stream(name);
    eprintln!("{name}: {coverage:?}");
    assert!(coverage.values > 0, "{name}");
}

const PRUNIT_GPS: f64 = 299792.458;
const PRUNIT_GLO: f64 = 599584.916;

/// State `decode_type1002`..`decode_type1012` carry between messages: the
/// carrier-phase rollover reference (`rtcm->cp`) and lock history.
#[derive(Default)]
struct LegacyOracleState {
    cp: HashMap<(String, usize), f64>,
    locks: LockState,
}

impl LegacyOracleState {
    /// RTKLIB `adjcp`.
    fn adjcp(&mut self, satellite: &str, slot: usize, cp: f64) -> f64 {
        let previous = self
            .cp
            .get(&(satellite.to_string(), slot))
            .copied()
            .unwrap_or(0.0);
        let mut cp = cp;
        if previous != 0.0 {
            if cp < previous - 750.0 {
                cp += 1500.0;
            } else if cp > previous + 750.0 {
                cp -= 1500.0;
            }
        }
        self.cp.insert((satellite.to_string(), slot), cp);
        cp
    }
}

/// RTKLIB `snratio`.
fn snratio(snr: f64) -> f64 {
    if snr <= 0.0 || 100.0 <= snr {
        0.0
    } else {
        snr
    }
}

/// RTKLIB's satellite name for a legacy satellite ID, or `None` when RTKLIB
/// `satno` has no satellite for it and skips the record.
fn legacy_satellite_name(system: GnssSystem, id: u8) -> Option<String> {
    match system {
        GnssSystem::Gps if (1..=32).contains(&id) => Some(format!("G{id:02}")),
        // decode_type1002/1004: an ID from 40 is SBAS, PRN + 80.
        GnssSystem::Gps if (40..=58).contains(&id) => Some(format!("{:03}", u16::from(id) + 80)),
        GnssSystem::Glonass if (1..=27).contains(&id) => Some(format!("R{id:02}")),
        _ => None,
    }
}

/// Compare one decoded legacy observation message with RTKLIB's record.
fn check_legacy(
    name: &str,
    frame: &Frame,
    obs: &LegacyObservations,
    state: &mut LegacyOracleState,
    coverage: &mut Coverage,
) {
    let at = format!("{name} frame at {} ({})", frame.offset, obs.message_number);
    let rtklib = &frame.rtklib;
    let system = obs.system().expect("legacy system");
    coverage.frames += 1;
    if system == GnssSystem::Gps {
        assert_eq!(
            bits64(&rtklib["tow"]),
            rtklib_tow(f64::from(obs.epoch_time) * 0.001).to_bits(),
            "{at}: epoch"
        );
    }
    assert_eq!(
        rtklib["staid"].as_u64(),
        Some(u64::from(obs.reference_station_id)),
        "{at}: station"
    );
    let stored = rtklib["obs"].as_array().expect("obs");
    let extended = obs.satellites.iter().all(|s| s.l1.cnr.is_some());
    if !matches!(obs.message_number, 1002 | 1004 | 1010 | 1012) {
        // RTKLIB decode_type1001/1003/1009/1011 read the header only.
        assert!(stored.is_empty(), "{at}");
        coverage.header_only += 1;
        return;
    }
    assert!(extended, "{at}");
    let by_satellite: BTreeMap<&str, &Value> = stored
        .iter()
        .map(|entry| (entry["sat"].as_str().expect("sat"), &entry["sig"]))
        .collect();
    let mut matched = 0usize;
    for s in &obs.satellites {
        let Some(sat_name) = legacy_satellite_name(system, s.satellite_id) else {
            coverage.not_stored += 1;
            continue;
        };
        let sigs = by_satellite
            .get(sat_name.as_str())
            .unwrap_or_else(|| panic!("{at}: RTKLIB stored no {sat_name}"))
            .as_array()
            .expect("sig");
        let slot = |index: u64| {
            sigs.iter()
                .find(|sig| sig["slot"].as_u64() == Some(index))
                .unwrap_or_else(|| panic!("{at}: {sat_name} has no slot {index}"))
        };
        let (unit, freq1, freq2) = match system {
            GnssSystem::Glonass => {
                let fcn = i32::from(s.frequency_channel.expect("channel")) - 7;
                (
                    PRUNIT_GLO,
                    code2freq(system, "1C", fcn),
                    code2freq(system, "2C", fcn),
                )
            }
            _ => (PRUNIT_GPS, FREQL1, FREQL2),
        };
        let ambiguity = s.l1.pseudorange_modulus_ambiguity.expect("ambiguity");
        let pr1 = f64::from(s.l1.pseudorange) * 0.02 + f64::from(ambiguity) * unit;
        let l1 = if s.l1.phase_range_minus_pseudorange != LEGACY_PHASE_RANGE_INVALID {
            let cp1 = state.adjcp(
                &sat_name,
                0,
                f64::from(s.l1.phase_range_minus_pseudorange) * 0.0005 * freq1 / CLIGHT,
            );
            pr1 * freq1 / CLIGHT + cp1
        } else {
            0.0
        };
        let lli1 = state
            .locks
            .loss_of_lock(&sat_name, 0, u16::from(s.l1.lock_time_indicator));
        let snr1 = snratio(f64::from(s.l1.cnr.expect("CNR")) * 0.25) as f32;
        let code1 = if s.l1.code_indicator { "1P" } else { "1C" };
        let entry = slot(0);
        let cell = format!("{at} {sat_name} L1");
        assert_eq!(entry["code"].as_str(), Some(code1), "{cell}: code");
        assert_eq!(bits64(&entry["P"]), pr1.to_bits(), "{cell}: P");
        assert_eq!(bits64(&entry["L"]), l1.to_bits(), "{cell}: L");
        assert_eq!(bits32(&entry["SNR"]), snr1.to_bits(), "{cell}: SNR");
        assert_eq!(entry["LLI"].as_u64(), Some(u64::from(lli1)), "{cell}: LLI");
        matched += 1;
        coverage.values += 1;
        if let Some(l2) = &s.l2 {
            let p2 = if l2.pseudorange_difference != LEGACY_PSEUDORANGE_DIFFERENCE_INVALID {
                pr1 + f64::from(l2.pseudorange_difference) * 0.02
            } else {
                0.0
            };
            let phase2 = if l2.phase_range_minus_l1_pseudorange != LEGACY_PHASE_RANGE_INVALID {
                let cp2 = state.adjcp(
                    &sat_name,
                    1,
                    f64::from(l2.phase_range_minus_l1_pseudorange) * 0.0005 * freq2 / CLIGHT,
                );
                pr1 * freq2 / CLIGHT + cp2
            } else {
                0.0
            };
            let lli2 = state
                .locks
                .loss_of_lock(&sat_name, 1, u16::from(l2.lock_time_indicator));
            let snr2 = snratio(f64::from(l2.cnr.expect("CNR")) * 0.25) as f32;
            let code2 = match system {
                GnssSystem::Glonass if l2.code_indicator != 0 => "2P",
                GnssSystem::Glonass => "2C",
                _ => ["2X", "2P", "2D", "2W"][usize::from(l2.code_indicator)],
            };
            let entry = slot(1);
            let cell = format!("{at} {sat_name} L2");
            assert_eq!(entry["code"].as_str(), Some(code2), "{cell}: code");
            assert_eq!(bits64(&entry["P"]), p2.to_bits(), "{cell}: P");
            assert_eq!(bits64(&entry["L"]), phase2.to_bits(), "{cell}: L");
            assert_eq!(bits32(&entry["SNR"]), snr2.to_bits(), "{cell}: SNR");
            assert_eq!(entry["LLI"].as_u64(), Some(u64::from(lli2)), "{cell}: LLI");
            matched += 1;
            coverage.values += 1;
        }
    }
    let rtklib_cells: usize = stored
        .iter()
        .map(|entry| entry["sig"].as_array().expect("sig").len())
        .sum();
    assert_eq!(
        matched, rtklib_cells,
        "{at}: every value RTKLIB stored is checked"
    );
}

/// Check every legacy frame of `name` against RTKLIB, and each compact
/// message (1001, 1003, 1009, 1011) and 1002/1010 against the 1004 or 1012 of
/// its epoch in the same stream, where one is there.
fn check_legacy_stream(name: &str) -> (Coverage, usize) {
    let frames = fixture(name);
    let mut state = LegacyOracleState::default();
    let mut coverage = Coverage::default();
    let mut full: HashMap<(GnssSystem, u32), LegacyObservations> = HashMap::new();
    for frame in &frames {
        let Message::LegacyObservations(obs) = &frame.message else {
            panic!(
                "{name}: frame at {} is not a legacy observation",
                frame.offset
            );
        };
        check_legacy(name, frame, obs, &mut state, &mut coverage);
        if matches!(obs.message_number, 1004 | 1012) {
            full.insert((obs.system().expect("system"), obs.epoch_time), obs.clone());
        }
    }
    let mut compared = 0;
    for frame in &frames {
        let Message::LegacyObservations(obs) = &frame.message else {
            continue;
        };
        if matches!(obs.message_number, 1004 | 1012) {
            continue;
        }
        let Some(reference) = full.get(&(obs.system().expect("system"), obs.epoch_time)) else {
            continue;
        };
        let at = format!("{name} frame at {} ({})", frame.offset, obs.message_number);
        assert_eq!(obs.satellites.len(), reference.satellites.len(), "{at}");
        for (a, b) in obs.satellites.iter().zip(&reference.satellites) {
            let sat = format!("{at} satellite {}", a.satellite_id);
            assert_eq!(a.satellite_id, b.satellite_id, "{sat}");
            assert_eq!(a.frequency_channel, b.frequency_channel, "{sat}");
            assert_eq!(a.l1.code_indicator, b.l1.code_indicator, "{sat}");
            assert_eq!(a.l1.pseudorange, b.l1.pseudorange, "{sat}");
            assert_eq!(
                a.l1.phase_range_minus_pseudorange, b.l1.phase_range_minus_pseudorange,
                "{sat}"
            );
            assert_eq!(a.l1.lock_time_indicator, b.l1.lock_time_indicator, "{sat}");
            if a.l1.cnr.is_some() {
                assert_eq!(
                    a.l1.pseudorange_modulus_ambiguity, b.l1.pseudorange_modulus_ambiguity,
                    "{sat}"
                );
                assert_eq!(a.l1.cnr, b.l1.cnr, "{sat}");
            }
            if let (Some(x), Some(y)) = (&a.l2, &b.l2) {
                assert_eq!(x.code_indicator, y.code_indicator, "{sat}");
                assert_eq!(x.pseudorange_difference, y.pseudorange_difference, "{sat}");
                assert_eq!(
                    x.phase_range_minus_l1_pseudorange, y.phase_range_minus_l1_pseudorange,
                    "{sat}"
                );
                assert_eq!(x.lock_time_indicator, y.lock_time_indicator, "{sat}");
            }
        }
        compared += 1;
    }
    (coverage, compared)
}

/// Real 1004 and 1012 from three receivers and RTKLIB's own test stream, real
/// 1001, 1003 and 1009, and RTKLIB-encoded 1001..1004 and 1009..1012: every
/// value RTKLIB stores from 1002, 1004, 1010 and 1012 bit for bit, the headers
/// of the others, and their fields against the 1004 or 1012 of the same epoch.
#[test]
fn legacy_observations_match_rtklib() {
    for (name, compact) in [
        ("rtk2go_granthamall_legacy.rtcm3", true),
        ("rtk2go_jacksbay_legacy.rtcm3", true),
        ("rtk2go_mirmenhof_legacy.rtcm3", false),
        ("rtklib_testglo_legacy.rtcm3", false),
        ("rtklib_encoded_legacy.rtcm3", true),
    ] {
        let (coverage, compared) = check_legacy_stream(name);
        eprintln!("{name}: {coverage:?}, {compared} messages checked against 1004/1012");
        assert!(coverage.values > 0, "{name}");
        assert_eq!(compared > 0, compact, "{name}");
    }
}

const P2_19: f64 = 1.907348632812500E-06;
const P2_28: f64 = 3.725290298461914E-09;
const P2_33: f64 = 1.164153218269348E-10;
const P2_41: f64 = 4.547473508864641E-13;
const P2_43: f64 = 1.136868377216160E-13;
const P2_55: f64 = 2.775557561562891E-17;
/// RTKLIB's semicircle-to-radian factor, the IS-GPS-200 value, not `PI`.
#[allow(clippy::approx_constant)]
const SC2RAD: f64 = 3.1415926535898;
/// RTKLIB `gpst2time` of the GPS epoch: 1980-01-06 as a Unix time.
const GPST0_UNIX_S: i64 = 315_964_800;

/// NavIC 1041 frames RTKLIB's encoder wrote from the real NavIC records of
/// DLR's merged broadcast file: every ephemeris value RTKLIB `decode_type1041`
/// stores, bit for bit, and the broadcast record built from the message
/// against the RINEX record it came from.
#[test]
fn navic_ephemeris_matches_rtklib() {
    let name = "rtklib_encoded_1041.rtcm3";
    let frames = fixture(name);
    let rinex = sidereon_core::rinex::nav::parse_nav(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/nav/BRDM00DLR_S_20262650000_01D_MN_navic.rnx"
        ))
        .expect("read NavIC RINEX"),
    )
    .expect("parse NavIC RINEX");
    let rinex: Vec<_> = rinex
        .into_iter()
        .filter(|record| record.satellite_id.system == GnssSystem::Navic)
        .collect();
    assert_eq!(frames.len(), rinex.len(), "one 1041 per NavIC record");
    for (frame, record) in frames.iter().zip(&rinex) {
        let Message::NavicEphemeris(eph) = &frame.message else {
            panic!("{name}: frame at {} is not 1041", frame.offset);
        };
        let at = format!("{name} frame at {}", frame.offset);
        let e = &frame.rtklib["eph"];
        assert_eq!(frame.rtklib["ret"].as_i64(), Some(2), "{at}");
        assert_eq!(
            e["sat"].as_str(),
            Some(format!("I{:02}", eph.satellite_id).as_str()),
            "{at}"
        );
        assert_eq!(e["iode"].as_u64(), Some(u64::from(eph.iodec)), "{at}");
        assert_eq!(e["iodc"].as_u64(), Some(u64::from(eph.iodec)), "{at}");
        assert_eq!(e["sva"].as_u64(), Some(u64::from(eph.ura)), "{at}");
        assert_eq!(e["svh"].as_u64(), Some(u64::from(eph.health())), "{at}");
        let week = e["week"].as_i64().expect("week");
        assert_eq!(week % 1024, i64::from(eph.week_number), "{at}");
        for (key, count) in [("toe", eph.t_oe), ("toc", eph.t_oc)] {
            assert_eq!(
                e[key][0].as_i64(),
                Some(GPST0_UNIX_S + 604_800 * week + i64::from(count) * 16),
                "{at} {key}"
            );
            assert_eq!(bits64(&e[key][1]), 0.0f64.to_bits(), "{at} {key}");
        }
        let sqrt_a = eph.sqrt_a as f64 * P2_19;
        for (key, value) in [
            ("A", sqrt_a * sqrt_a),
            ("e", eph.eccentricity as f64 * P2_33),
            ("i0", eph.i0 as f64 * P2_31 * SC2RAD),
            ("OMG0", eph.omega0 as f64 * P2_31 * SC2RAD),
            ("omg", eph.omega as f64 * P2_31 * SC2RAD),
            ("M0", eph.m0 as f64 * P2_31 * SC2RAD),
            ("deln", f64::from(eph.delta_n) * P2_41 * SC2RAD),
            ("OMGd", f64::from(eph.omega_dot) * P2_41 * SC2RAD),
            ("idot", f64::from(eph.idot) * P2_43 * SC2RAD),
            ("crc", f64::from(eph.c_rc) * 0.0625),
            ("crs", f64::from(eph.c_rs) * 0.0625),
            ("cuc", f64::from(eph.c_uc) * P2_28),
            ("cus", f64::from(eph.c_us) * P2_28),
            ("cic", f64::from(eph.c_ic) * P2_28),
            ("cis", f64::from(eph.c_is) * P2_28),
            ("toes", f64::from(eph.t_oe) * 16.0),
            ("f0", f64::from(eph.a_f0) * P2_31),
            ("f1", f64::from(eph.a_f1) * P2_43),
            ("f2", f64::from(eph.a_f2) * P2_55),
            ("tgd0", f64::from(eph.t_gd) * P2_31),
        ] {
            assert_eq!(bits64(&e[key]), value.to_bits(), "{at} {key}");
        }

        // The record built from the message states what the RINEX record
        // states: the same satellite, issue, health and times, the accuracy
        // bin RTKLIB `uraindex` chose for the RINEX metres, and the orbit and
        // clock to half a unit of each field RTKLIB rounded them to.
        let built = eph
            .to_broadcast_record(u32::try_from(week).expect("week"))
            .expect("broadcast record");
        assert_eq!(built.satellite_id, record.satellite_id, "{at}");
        assert_eq!(built.message, record.message, "{at}");
        assert_eq!(built.issue_of_data, record.issue_of_data, "{at}");
        assert_eq!(built.sv_health, record.sv_health, "{at}");
        assert!(
            record.sv_accuracy_m.expect("RINEX accuracy") <= built.sv_accuracy_m.expect("URA bin"),
            "{at}"
        );
        assert_eq!(built.elements.toe_sow, record.elements.toe_sow, "{at}");
        assert_eq!(built.clock.toc_sow, record.clock.toc_sow, "{at}");
        let semicircle = std::f64::consts::PI;
        for (what, a, b, unit) in [
            (
                "sqrt_a",
                built.elements.sqrt_a,
                record.elements.sqrt_a,
                2f64.powi(-19),
            ),
            ("e", built.elements.e, record.elements.e, 2f64.powi(-33)),
            (
                "m0",
                built.elements.m0,
                record.elements.m0,
                2f64.powi(-31) * semicircle,
            ),
            (
                "omega0",
                built.elements.omega0,
                record.elements.omega0,
                2f64.powi(-31) * semicircle,
            ),
            (
                "omega_dot",
                built.elements.omega_dot,
                record.elements.omega_dot,
                2f64.powi(-41) * semicircle,
            ),
            (
                "delta_n",
                built.elements.delta_n,
                record.elements.delta_n,
                2f64.powi(-41) * semicircle,
            ),
            ("af0", built.clock.af0, record.clock.af0, 2f64.powi(-31)),
            ("af1", built.clock.af1, record.clock.af1, 2f64.powi(-43)),
        ] {
            assert!(
                (a - b).abs() <= unit * 0.500_001 + 1e-14 * b.abs(),
                "{at} {what}: {a} vs {b}"
            );
        }
    }
}

/// Real 1230 frames from seven rtk2go receivers, one with nonzero biases:
/// the alignment flag and the four biases RTKLIB `decode_type1230` stores,
/// bit for bit.
#[test]
fn glonass_code_phase_biases_match_rtklib() {
    let name = "rtk2go_1230.rtcm3";
    let frames = fixture(name);
    let mut station = 0u16;
    let mut compared = 0;
    for frame in &frames {
        let Message::GlonassCodePhaseBiases(biases) = &frame.message else {
            panic!("{name}: frame at {} is not 1230", frame.offset);
        };
        let at = format!("{name} frame at {}", frame.offset);
        // RTKLIB `test_staid` refuses a station that differs from the nonzero
        // one it holds, and forgets that one; only such a frame goes unread.
        let id = biases.reference_station_id;
        if station != 0 && station != id {
            assert_eq!(frame.rtklib["ret"].as_i64(), Some(-1), "{at}");
            station = 0;
            continue;
        }
        station = id;
        assert_eq!(frame.rtklib["ret"].as_i64(), Some(5), "{at}");
        let cp = &frame.rtklib["glo_cp"];
        assert_eq!(
            cp["align"].as_u64(),
            Some(u64::from(biases.aligned)),
            "{at}"
        );
        for (index, bias) in [biases.l1_ca, biases.l1_p, biases.l2_ca, biases.l2_p]
            .into_iter()
            .enumerate()
        {
            let expected = match bias {
                Some(raw) if raw != rtcm::GLONASS_CODE_PHASE_BIAS_INVALID => f64::from(raw) * 0.02,
                _ => 0.0,
            };
            assert_eq!(
                bits64(&cp["bias"][index]),
                expected.to_bits(),
                "{at} {index}"
            );
        }
        compared += 1;
    }
    eprintln!("{name}: {compared} of {} frames compared", frames.len());
    assert!(compared >= frames.len() - 1, "{name}");
    assert!(frames.iter().any(|frame| matches!(
        &frame.message,
        Message::GlonassCodePhaseBiases(b) if b.l1_ca.is_some_and(|raw| raw != 0)
    )));
}

/// RTKLIB `ssrudint`: the update interval of each SSR update-interval index.
const SSR_UPDATE_INTERVAL_S: [f64; 16] = [
    1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 240.0, 300.0, 600.0, 900.0, 1800.0, 3600.0,
    7200.0, 10800.0,
];

/// RTKLIB's satellite name for an IGS SSR satellite ID, or `None` when RTKLIB
/// `satno` has no satellite for it (`decode_ssr*` offsets QZSS by 192 and SBAS
/// by 119, and reads 0 as no satellite).
fn igs_ssr_satellite_name(system: GnssSystem, id: u8) -> Option<String> {
    let name = match system {
        GnssSystem::Gps if (1..=32).contains(&id) => format!("G{id:02}"),
        GnssSystem::Glonass if (1..=27).contains(&id) => format!("R{id:02}"),
        GnssSystem::Galileo if (1..=36).contains(&id) => format!("E{id:02}"),
        GnssSystem::Qzss if (1..=10).contains(&id) => format!("J{id:02}"),
        GnssSystem::BeiDou if (1..=63).contains(&id) => format!("C{id:02}"),
        GnssSystem::Sbas if (1..=39).contains(&id) => format!("{:03}", u16::from(id) + 119),
        _ => return None,
    };
    Some(name)
}

/// Every value RTKLIB `decode_ssr1`..`decode_ssr7` stored from `ssr`, bit for
/// bit, with its epoch, update interval, IOD SSR, issue and datum; the number
/// of satellites checked. `name_of` names a satellite as RTKLIB does.
fn check_rtklib_ssr(
    at: &str,
    ssr: &rtcm::SsrMessage,
    rtklib: &Value,
    name_of: fn(GnssSystem, u8) -> Option<String>,
    values: &mut usize,
) -> usize {
    let stored = rtklib["ssr"].as_array().expect("ssr");
    let by_satellite: BTreeMap<&str, &Value> = stored
        .iter()
        .map(|entry| (entry["sat"].as_str().expect("sat"), entry))
        .collect();
    let week = rtklib["week"].as_i64().expect("week");
    let epoch = if ssr.system == GnssSystem::Glonass && ssr.igs_ssr_version.is_none() {
        // The native GLONASS epoch is a GLONASS time of day (UTC + 3 h), which
        // RTKLIB `adjday_glot` places in GPS time: GPS - UTC is 18 s here.
        let tow = f64::from_bits(bits64(&rtklib["tow"]));
        assert_eq!(tow.fract(), 0.0, "{at}");
        let tow = tow as i64;
        assert_eq!(
            (tow - 18 + 10_800).rem_euclid(86_400),
            i64::from(ssr.header.epoch_time_s),
            "{at} GLONASS epoch"
        );
        GPST0_UNIX_S + 604_800 * week + tow
    } else {
        GPST0_UNIX_S + 604_800 * week + i64::from(ssr.header.epoch_time_s)
    };
    let udi = SSR_UPDATE_INTERVAL_S[usize::from(ssr.header.update_interval)];
    let mut named = 0usize;
    let check_common = |entry: &Value, slots: &[usize]| {
        for &k in slots {
            assert_eq!(entry["t0"][k][0].as_i64(), Some(epoch), "{at} t0[{k}]");
            assert_eq!(bits64(&entry["t0"][k][1]), 0f64.to_bits(), "{at} t0[{k}]");
            assert_eq!(bits64(&entry["udi"][k]), udi.to_bits(), "{at} udi[{k}]");
            assert_eq!(
                entry["iod"][k].as_u64(),
                Some(u64::from(ssr.header.iod_ssr)),
                "{at} iod[{k}]"
            );
        }
    };
    let entry_of = |id: u8| -> Option<&Value> {
        let sat = name_of(ssr.system, id)?;
        Some(
            by_satellite
                .get(sat.as_str())
                .copied()
                .unwrap_or_else(|| panic!("{at}: RTKLIB stored no {sat}")),
        )
    };
    let orbit_values = |entry: &Value, orbit: &rtcm::SsrOrbitRecord| {
        let iodcrc = match (ssr.igs_ssr_version, ssr.system) {
            (None, GnssSystem::Sbas) => orbit.iod_crc.expect("IOD CRC"),
            _ => 0,
        };
        assert_eq!(entry["iode"].as_u64(), Some(u64::from(orbit.iode)), "{at}");
        assert_eq!(entry["iodcrc"].as_u64(), Some(u64::from(iodcrc)), "{at}");
        assert_eq!(
            entry["refd"].as_u64(),
            Some(u64::from(
                ssr.header.satellite_reference_datum.expect("datum")
            )),
            "{at}"
        );
        let deph = [
            f64::from(orbit.delta_radial) * 1E-4,
            f64::from(orbit.delta_along) * 4E-4,
            f64::from(orbit.delta_cross) * 4E-4,
        ];
        let ddeph = [
            f64::from(orbit.dot_delta_radial) * 1E-6,
            f64::from(orbit.dot_delta_along) * 4E-6,
            f64::from(orbit.dot_delta_cross) * 4E-6,
        ];
        for k in 0..3 {
            assert_eq!(bits64(&entry["deph"][k]), deph[k].to_bits(), "{at} deph");
            assert_eq!(bits64(&entry["ddeph"][k]), ddeph[k].to_bits(), "{at} ddeph");
        }
    };
    let clock_values = |entry: &Value, clock: &rtcm::SsrClockRecord| {
        let dclk = [
            f64::from(clock.c0) * 1E-4,
            f64::from(clock.c1) * 1E-6,
            f64::from(clock.c2) * 2E-8,
        ];
        for (k, value) in dclk.iter().enumerate() {
            assert_eq!(bits64(&entry["dclk"][k]), value.to_bits(), "{at} dclk");
        }
    };
    match ssr.kind {
        rtcm::SsrKind::Orbit => {
            for orbit in &ssr.orbit {
                let Some(entry) = entry_of(orbit.satellite_id) else {
                    continue;
                };
                check_common(entry, &[0]);
                orbit_values(entry, orbit);
                named += 1;
                *values += 6;
            }
        }
        rtcm::SsrKind::Clock => {
            for clock in &ssr.clock {
                let Some(entry) = entry_of(clock.satellite_id) else {
                    continue;
                };
                check_common(entry, &[1]);
                clock_values(entry, clock);
                named += 1;
                *values += 3;
            }
        }
        rtcm::SsrKind::CombinedOrbitClock => {
            for (orbit, clock) in ssr.orbit.iter().zip(&ssr.clock) {
                let Some(entry) = entry_of(orbit.satellite_id) else {
                    continue;
                };
                check_common(entry, &[0, 1]);
                orbit_values(entry, orbit);
                clock_values(entry, clock);
                named += 1;
                *values += 9;
            }
        }
        rtcm::SsrKind::HighRateClock => {
            for clock in &ssr.clock {
                let Some(entry) = entry_of(clock.satellite_id) else {
                    continue;
                };
                check_common(entry, &[2]);
                let hrclk = f64::from(clock.c0) * 1E-4;
                assert_eq!(bits64(&entry["hrclk"]), hrclk.to_bits(), "{at} hrclk");
                named += 1;
                *values += 1;
            }
        }
        rtcm::SsrKind::Ura => {
            for &(id, ura) in &ssr.ura {
                let Some(entry) = entry_of(id) else { continue };
                check_common(entry, &[3]);
                assert_eq!(entry["ura"].as_u64(), Some(u64::from(ura)), "{at} ura");
                named += 1;
                *values += 1;
            }
        }
        rtcm::SsrKind::CodeBias => {
            for record in &ssr.code_bias {
                let Some(entry) = entry_of(record.satellite_id) else {
                    continue;
                };
                check_common(entry, &[4]);
                // RTKLIB reads the identifiers through its RTCM SSR tables and
                // keeps the last value of a signal.
                let mut expected = BTreeMap::new();
                for &(signal, bias) in &record.biases {
                    if let Some(physical) = sidereon_core::ssr::rtcm_ssr_signal(ssr.system, signal)
                    {
                        expected
                            .insert(physical.code().to_string(), (f64::from(bias) * 0.01) as f32);
                    }
                }
                expected.retain(|_, value| *value != 0.0);
                let stored = entry["cbias"].as_object().expect("cbias");
                assert_eq!(stored.len(), expected.len(), "{at} cbias");
                for (code, value) in &expected {
                    assert_eq!(bits32(&stored[code]), value.to_bits(), "{at} cbias {code}");
                }
                named += 1;
                *values += expected.len();
            }
        }
        rtcm::SsrKind::PhaseBias => {
            for record in &ssr.phase_bias {
                let Some(entry) = entry_of(record.satellite_id) else {
                    continue;
                };
                check_common(entry, &[5]);
                let yaw_ang = f64::from(record.yaw_angle) / 256.0 * 180.0;
                let yaw_rate = f64::from(record.yaw_rate) / 8192.0 * 180.0;
                assert_eq!(bits64(&entry["yaw_ang"]), yaw_ang.to_bits(), "{at} yaw");
                assert_eq!(
                    bits64(&entry["yaw_rate"]),
                    yaw_rate.to_bits(),
                    "{at} yaw rate"
                );
                let mut expected = BTreeMap::new();
                for bias in &record.biases {
                    if let Some(physical) =
                        sidereon_core::ssr::rtcm_ssr_signal(ssr.system, bias.signal_id)
                    {
                        expected.insert(physical.code().to_string(), f64::from(bias.bias) * 0.0001);
                    }
                }
                expected.retain(|_, value| *value != 0.0);
                let stored = entry["pbias"].as_object().expect("pbias");
                assert_eq!(stored.len(), expected.len(), "{at} pbias");
                for (code, value) in &expected {
                    assert_eq!(bits64(&stored[code]), value.to_bits(), "{at} pbias {code}");
                }
                named += 1;
                *values += 2 + expected.len();
            }
        }
    }
    assert_eq!(
        named,
        stored.len(),
        "{at}: every satellite RTKLIB stored is checked"
    );
    named
}

/// IGS SSR 4076 frames of every satellite subtype, written by RTKLIB's encoder
/// from the corrections of a real IGS RTCM SSR stream: every orbit, clock,
/// high-rate clock, URA, code-bias and phase-bias value RTKLIB `decode_ssr1`..
/// `decode_ssr7` store, bit for bit, with its epoch, update interval, IOD SSR,
/// issue and datum. RTKLIB `decode_ssr7` reads no SBAS phase bias (126); those
/// frames are checked against the standard by the unit tests.
#[test]
fn igs_ssr_matches_rtklib() {
    let name = "rtklib_encoded_4076.rtcm3";
    let frames = fixture(name);
    let mut values = 0usize;
    let mut satellites = 0usize;
    let mut unread = 0usize;
    let mut subtypes = std::collections::BTreeSet::new();
    for frame in &frames {
        let Message::Ssr(ssr) = &frame.message else {
            panic!(
                "{name}: frame at {} is not an IGS SSR message",
                frame.offset
            );
        };
        let at = format!("{name} frame at {}", frame.offset);
        let subtype = ssr.igs_ssr_subtype().expect("IGS SSR subtype");
        subtypes.insert(subtype);
        let stored = frame.rtklib["ssr"].as_array().expect("ssr");
        if ssr.system == GnssSystem::Sbas && ssr.kind == rtcm::SsrKind::PhaseBias {
            assert!(stored.is_empty(), "{at}");
            unread += 1;
            continue;
        }
        let named = check_rtklib_ssr(&at, ssr, &frame.rtklib, igs_ssr_satellite_name, &mut values);
        satellites += named;
    }
    eprintln!(
        "{name}: {} frames, {satellites} satellite records, {values} values, {unread} frames RTKLIB does not read",
        frames.len()
    );
    assert_eq!(subtypes.len(), 42, "every satellite subtype");
    assert!(values > 0);
}

/// RTKLIB's satellite name for a native RTCM SSR satellite ID of GPS, GLONASS,
/// Galileo or SBAS, or `None` when RTKLIB `satno` has no satellite for it
/// (`decode_ssr*` offsets SBAS by 120).
fn rtcm_ssr_satellite_name(system: GnssSystem, id: u8) -> Option<String> {
    let name = match system {
        GnssSystem::Gps if (1..=32).contains(&id) => format!("G{id:02}"),
        GnssSystem::Glonass if (1..=27).contains(&id) => format!("R{id:02}"),
        GnssSystem::Galileo if (1..=36).contains(&id) => format!("E{id:02}"),
        GnssSystem::Sbas if id <= 38 => format!("{:03}", u16::from(id) + 120),
        _ => return None,
    };
    Some(name)
}

/// The RTCM SSR messages BNC's encoder writes (`bnc_encoded_ssr.rtcm3`, see
/// `rtcm_ssr_bnc_oracle.rs`) for GPS, GLONASS, Galileo and SBAS (1252..1257,
/// with the 9-bit t0 modulo and the 24-bit IOD CRC): every value RTKLIB
/// `decode_ssr1`..`decode_ssr6` store, bit for bit.
///
/// RTKLIB stores nothing from the phase-bias messages 1265..1270 or the VTEC
/// message 1264. It reads the other frames in layouts that differ from the
/// one RTCM 10403.3 states and BNC reads, so they are checked against BNC
/// only: for QZSS (1246..1251) RTKLIB reads a 4-bit satellite count where the
/// header carries the 6-bit DF387 count, and stores no satellite; for BeiDou
/// (1258..1263) RTKLIB demo5 reads a 24-bit IOD CRC after the 10-bit toe
/// modulo where the frame carries the 8-bit BeiDou IOD.
#[test]
fn rtcm_ssr_matches_rtklib() {
    let name = "bnc_encoded_ssr.rtcm3";
    let mut values = 0usize;
    let mut satellites = 0usize;
    let mut numbers = std::collections::BTreeSet::new();
    for frame in fixture(name) {
        let Message::Ssr(ssr) = &frame.message else {
            continue;
        };
        if ssr.igs_ssr_version.is_some() {
            continue;
        }
        let at = format!("{name} frame at {}", frame.offset);
        if ssr.kind == rtcm::SsrKind::PhaseBias {
            assert_eq!(frame.rtklib["ret"].as_i64(), Some(0), "{at}");
            continue;
        }
        let stored = frame.rtklib["ssr"].as_array().expect("ssr");
        match ssr.system {
            GnssSystem::Qzss => {
                assert!(stored.is_empty(), "{at}");
                continue;
            }
            GnssSystem::BeiDou => continue,
            _ => {}
        }
        numbers.insert(ssr.message_number);
        let named = check_rtklib_ssr(
            &at,
            ssr,
            &frame.rtklib,
            rtcm_ssr_satellite_name,
            &mut values,
        );
        assert!(named > 0, "{at}");
        satellites += named;
    }
    eprintln!(
        "{name}: {} messages, {satellites} satellite records, {values} values",
        numbers.len()
    );
    assert_eq!(
        numbers.len(),
        24,
        "GPS, GLONASS, Galileo and SBAS, six messages each"
    );
}

/// Real 1013 frames decode strictly and round-trip; RTKLIB `decode_type1013`
/// reads nothing from them. 1029 frames with ASCII and multi-byte UTF-8 text:
/// the text is the DF139 count of code units, and RTKLIB `decode_type1029`,
/// which reads the DF138 count of characters as bytes, stores the first
/// `character_count` of those code units, so a multi-byte text reaches it cut
/// short.
#[test]
fn system_parameters_and_text_match_rtklib() {
    let frames = fixture("rtk2go_1013.rtcm3");
    assert!(!frames.is_empty());
    for frame in &frames {
        let Message::SystemParameters(parameters) = &frame.message else {
            panic!("frame at {} is not 1013", frame.offset);
        };
        assert_eq!(
            usize::from(parameters.announcement_count),
            parameters.announcements.len()
        );
    }
    let frames = fixture("text_1029.rtcm3");
    let mut cut = 0;
    for frame in &frames {
        let Message::Text(text) = &frame.message else {
            panic!("frame at {} is not 1029", frame.offset);
        };
        assert!(text.text().is_ok());
        assert_eq!(
            text.text().expect("UTF-8").chars().count(),
            usize::from(text.character_count)
        );
        let stored = frame.rtklib["text"].as_str().expect("text");
        let expected: String = text.code_units[..usize::from(text.character_count).min(126)]
            .iter()
            .map(|unit| format!("{unit:02x}"))
            .collect();
        assert_eq!(stored, expected, "frame at {}", frame.offset);
        cut += usize::from(usize::from(text.character_count) < text.code_units.len());
    }
    assert!(cut > 0, "a multi-byte text is among the frames");
}
