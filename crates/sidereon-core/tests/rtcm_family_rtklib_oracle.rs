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
//!   `TiftGA`, `Ormalingen_Ribi`, `sejongnav` and `Mirmenhof`), reduced to the
//!   frames of the family each file names, byte for byte.
//! - `rtklib_encoded_msm1_to_msm4.rtcm3`: RTKLIB's encoder output (`encode-msm`)
//!   from the MSM7 observations of RTKLIB's test stream
//!   `test/data/rcvraw/GMSD7_20121014.rtcm3`, first 12 epochs.

use std::collections::{BTreeMap, HashMap};

use serde_json::Value;
use sidereon_core::rtcm::{self, FrameScanner, Message, MsmKind, MsmMessage};
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
