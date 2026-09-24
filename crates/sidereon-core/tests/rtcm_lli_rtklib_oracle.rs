//! RTKLIB convbin oracle for RTCM MSM LLI derivation.
//!
//! Fixture: `tests/fixtures/rtcm/gmsd7_20121014.rtcm3`, copied byte-for-byte
//! from `test/data/rcvraw/GMSD7_20121014.rtcm3` in the RTKLIB demo5 tree.
//! Source SHA-256:
//! `8b466a6829122c21ee28053324811022b3e44baa6dcd878a51b9a746a10d8cb1`.
//! Source provenance: RTKLIB demo5 sample receiver stream, marker `0611`,
//! observation window 2012-10-13 23:59:58 GPST through 2012-10-14 00:04:14
//! GPST. The stream carries MSM7 message types 1077, 1087, 1117, and 1127,
//! but convbin's RINEX emission for this receiver yields only BeiDou
//! observations, so the independent cross-check exercises BeiDou LLI
//! derivation over B1I/B2I/B3I (RINEX codes L2I/L7I/L6I). GPS, GLONASS,
//! Galileo, and QZSS LLI derivation are covered by the synthetic truth-table
//! vectors in the unit tests rather than by this oracle.
//! Reference tool: RTKLIB demo5 `convbin` (built from
//! `app/consapp/convbin/gcc`), RINEX header program `CONVBIN demo5 b34L`; the
//! test runs `-v 3.05 -r rtcm3 -tr 2012/10/14 0:00:00 -od -os`. Set
//! `RTKLIB_CONVBIN` to the binary's path to run the oracle; without it the test
//! skips.
//! GPS week 1710 after the week rollover in the fixture; leap-second count
//! 16 s; BeiDou-to-GPST offset +14 s.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use sidereon_core::rtcm::{self, LockTimeTracker, Message};
use sidereon_core::GnssSystem;

const FIXTURE: &[u8] = include_bytes!("fixtures/rtcm/gmsd7_20121014.rtcm3");
const FIXTURE_FILE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtcm/gmsd7_20121014.rtcm3"
);
const WEEK_MS: i64 = 604_800_000;
const DAY_MS: i64 = 86_400_000;
const LEAP_SECONDS_MS: i64 = 16_000;
const BDT_TO_GPST_MS: i64 = 14_000;
const GLONASS_UTC_PLUS_3_MS: i64 = 10_800_000;
const GLONASS_DAY_SHIFT: u32 = 27;
const GLONASS_MS_MASK: u32 = (1 << GLONASS_DAY_SHIFT) - 1;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct ObsKey {
    epoch_ms: u64,
    satellite: String,
    phase_code: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Divergence {
    None,
    D1,
    D2,
}

#[derive(Clone, Copy, Debug)]
struct RawState {
    raw_epoch_time: u32,
    raw_lock_indicator: u16,
}

#[derive(Clone, Copy, Debug)]
struct OurObservation {
    lli: u8,
    divergence: Divergence,
}

#[derive(Debug)]
struct RtklibOracle {
    observations: BTreeMap<ObsKey, u8>,
    phase_codes_by_system: BTreeMap<char, BTreeSet<String>>,
    program_line: String,
    first_epoch_ms: u64,
    last_epoch_ms: u64,
}

#[derive(Debug)]
struct OurObservations {
    observations: BTreeMap<ObsKey, OurObservation>,
    joined_signal_codes: BTreeSet<(u8, String)>,
}

#[test]
fn rtklib_convbin_real_msm_stream_matches_lli_except_d1_d2() {
    assert_eq!(FIXTURE.len(), 262_144);

    let Some(convbin) = resolve_convbin() else {
        eprintln!(
            "SKIP rtklib_convbin_real_msm_stream_matches_lli_except_d1_d2: \
             RTKLIB_CONVBIN is unset or does not name a file. Set it to an RTKLIB \
             demo5 convbin binary to run the LLI oracle."
        );
        return;
    };

    let oracle = run_convbin_and_parse_rinex(&convbin);
    assert!(
        oracle.program_line.contains("CONVBIN demo5 b34L"),
        "unexpected convbin header: {}",
        oracle.program_line
    );
    assert_eq!(oracle.first_epoch_ms, 604_798_000);
    assert_eq!(oracle.last_epoch_ms, 254_000);
    assert_eq!(oracle.observations.len(), 6_198);

    let our = build_our_observations(&oracle.phase_codes_by_system);
    assert_eq!(
        our.joined_signal_codes,
        BTreeSet::from([
            (2, String::from("L2I")),
            (8, String::from("L6I")),
            (14, String::from("L7I")),
        ])
    );

    let mut remaining_ours = our.observations;
    let mut joined = 0usize;
    let mut d1_count = 0usize;
    let mut d2_count = 0usize;
    let mut rtklib_bit0_count = 0usize;
    let mut our_bit0_count = 0usize;
    let mut unclassified = Vec::new();

    for (key, &rtklib_lli) in &oracle.observations {
        let ours = remaining_ours
            .remove(key)
            .unwrap_or_else(|| panic!("missing sidereon observation for {key:?}"));
        joined += 1;
        rtklib_bit0_count += usize::from((rtklib_lli & rtcm::LLI_LOSS_OF_LOCK) != 0);
        our_bit0_count += usize::from((ours.lli & rtcm::LLI_LOSS_OF_LOCK) != 0);

        assert_eq!(
            ours.lli & rtcm::LLI_HALF_CYCLE,
            rtklib_lli & rtcm::LLI_HALF_CYCLE,
            "half-cycle LLI mismatch for {key:?}"
        );

        if ours.lli != rtklib_lli {
            match ours.divergence {
                Divergence::D1 => d1_count += 1,
                Divergence::D2 => d2_count += 1,
                Divergence::None => unclassified.push(format!(
                    "{key:?}: sidereon LLI {} vs RTKLIB LLI {}",
                    ours.lli, rtklib_lli
                )),
            }
        }
    }

    assert_eq!(joined, 6_198);
    assert!(
        remaining_ours.is_empty(),
        "{} sidereon observations did not join RTKLIB RINEX, first keys: {:?}",
        remaining_ours.len(),
        remaining_ours.keys().take(5).collect::<Vec<_>>()
    );
    assert!(
        unclassified.is_empty(),
        "unclassified RTKLIB LLI mismatches: {unclassified:?}"
    );
    assert_eq!(d1_count, 0);
    assert_eq!(d2_count, 3);
    assert_eq!(rtklib_bit0_count, 4);
    assert_eq!(our_bit0_count, 7);
}

/// Resolve the RTKLIB convbin binary from `RTKLIB_CONVBIN`. Returns `None`
/// when the variable is unset or does not name a file, so the oracle skips
/// rather than failing a build where RTKLIB is not provisioned.
fn resolve_convbin() -> Option<PathBuf> {
    let candidate = PathBuf::from(env::var_os("RTKLIB_CONVBIN")?);
    candidate.is_file().then_some(candidate)
}

fn run_convbin_and_parse_rinex(convbin: &PathBuf) -> RtklibOracle {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).expect("create convbin temp dir");
    let obs_path = temp_dir.join("gmsd.obs");
    let nav_path = temp_dir.join("gmsd.nav");

    let output = Command::new(convbin)
        .args([
            "-v",
            "3.05",
            "-r",
            "rtcm3",
            "-tr",
            "2012/10/14",
            "0:00:00",
            "-od",
            "-os",
            "-o",
        ])
        .arg(&obs_path)
        .arg("-n")
        .arg(&nav_path)
        .arg(FIXTURE_FILE)
        .output()
        .unwrap_or_else(|err| panic!("run convbin at {}: {err}", convbin.display()));

    assert!(
        output.status.success(),
        "convbin failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let text = fs::read_to_string(&obs_path).expect("read convbin RINEX OBS");
    let oracle = parse_rinex_obs(&text);
    let _ = fs::remove_dir_all(&temp_dir);
    oracle
}

fn unique_temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after Unix epoch")
        .as_nanos();
    env::temp_dir().join(format!(
        "sidereon-rtcm-lli-oracle-{}-{nanos}",
        std::process::id()
    ))
}

fn parse_rinex_obs(text: &str) -> RtklibOracle {
    let lines: Vec<&str> = text.lines().collect();
    let mut obs_types: BTreeMap<char, Vec<String>> = BTreeMap::new();
    let mut program_line = String::new();
    let mut data_start = None;
    let mut i = 0usize;

    while i < lines.len() {
        let line = lines[i];
        let label = rinex_label(line);
        if label == "PGM / RUN BY / DATE" {
            program_line = line.to_string();
        } else if label == "SYS / # / OBS TYPES" {
            let system = line.as_bytes()[0] as char;
            let count: usize = line[3..6].trim().parse().expect("RINEX obs type count");
            let mut types = rinex_obs_types_from_header_line(line);
            while types.len() < count {
                i += 1;
                types.extend(rinex_obs_types_from_header_line(lines[i]));
            }
            types.truncate(count);
            obs_types.insert(system, types);
        } else if label == "END OF HEADER" {
            data_start = Some(i + 1);
            break;
        }
        i += 1;
    }

    let data_start = data_start.expect("RINEX END OF HEADER");
    let mut phase_codes_by_system = BTreeMap::new();
    for (&system, types) in &obs_types {
        phase_codes_by_system.insert(
            system,
            types
                .iter()
                .filter(|code| code.starts_with('L'))
                .cloned()
                .collect::<BTreeSet<_>>(),
        );
    }

    let mut observations = BTreeMap::new();
    let mut first_epoch_ms = None;
    let mut last_epoch_ms = None;
    let mut idx = data_start;
    while idx < lines.len() {
        let line = lines[idx];
        if !line.starts_with('>') {
            idx += 1;
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        assert!(parts.len() >= 9, "bad RINEX epoch line: {line}");
        let epoch_ms = rinex_epoch_ms_of_week(
            parts[1].parse().expect("year"),
            parts[2].parse().expect("month"),
            parts[3].parse().expect("day"),
            parts[4].parse().expect("hour"),
            parts[5].parse().expect("minute"),
            parts[6],
        );
        first_epoch_ms.get_or_insert(epoch_ms);
        last_epoch_ms = Some(epoch_ms);
        let satellite_count: usize = parts[8].parse().expect("satellite count");
        idx += 1;

        for _ in 0..satellite_count {
            let obs_line = lines[idx];
            idx += 1;
            let satellite = obs_line[0..3].to_string();
            let system = satellite.as_bytes()[0] as char;
            let Some(types) = obs_types.get(&system) else {
                continue;
            };
            let body = &obs_line[3..];
            for (obs_index, obs_type) in types.iter().enumerate() {
                if !obs_type.starts_with('L') {
                    continue;
                }
                let field = slice_padded(body, obs_index * 16, 16);
                if field[0..14].trim().is_empty() {
                    continue;
                }
                let lli = field.as_bytes()[14];
                let lli = if lli.is_ascii_digit() { lli - b'0' } else { 0 };
                let key = ObsKey {
                    epoch_ms,
                    satellite: satellite.clone(),
                    phase_code: obs_type.clone(),
                };
                assert!(
                    observations.insert(key.clone(), lli).is_none(),
                    "duplicate RTKLIB RINEX observation {key:?}"
                );
            }
        }
    }

    RtklibOracle {
        observations,
        phase_codes_by_system,
        program_line,
        first_epoch_ms: first_epoch_ms.expect("first RINEX epoch"),
        last_epoch_ms: last_epoch_ms.expect("last RINEX epoch"),
    }
}

fn rinex_label(line: &str) -> &str {
    line.get(60..).unwrap_or("").trim()
}

fn rinex_obs_types_from_header_line(line: &str) -> Vec<String> {
    line.get(7..60)
        .unwrap_or("")
        .split_whitespace()
        .map(String::from)
        .collect()
}

fn slice_padded(text: &str, start: usize, len: usize) -> String {
    let mut out = String::with_capacity(len);
    if start < text.len() {
        out.push_str(&text[start..text.len().min(start + len)]);
    }
    while out.len() < len {
        out.push(' ');
    }
    out
}

fn build_our_observations(
    phase_codes_by_system: &BTreeMap<char, BTreeSet<String>>,
) -> OurObservations {
    let stream = rtcm::decode_stream(FIXTURE);
    assert!(
        stream.diagnostics.skipped_frames.is_empty(),
        "fixture has skipped RTCM frames: {:?}",
        stream.diagnostics.skipped_frames
    );

    let mut tracker = LockTimeTracker::new();
    let mut raw_states: BTreeMap<(GnssSystem, u8, u8), RawState> = BTreeMap::new();
    let mut observations = BTreeMap::new();
    let mut joined_signal_codes = BTreeSet::new();

    for message in &stream.messages {
        let Message::Msm(msm) = message else {
            continue;
        };
        let cells = tracker.observe(msm);
        assert_eq!(cells.len(), msm.signals.len());

        let system_char = rinex_system_char(msm.system);
        let Some(phase_codes) = phase_codes_by_system.get(&system_char) else {
            continue;
        };

        for (signal, cell) in msm.signals.iter().zip(cells) {
            let Some(suffix) = rtcm::msm_signal_rinex_code(msm.system, signal.signal_id) else {
                continue;
            };
            let phase_code = format!("L{suffix}");
            if !phase_codes.contains(&phase_code) {
                continue;
            }

            let state_key = (msm.system, signal.satellite_id, signal.signal_id);
            let previous = raw_states.get(&state_key).copied();
            let min_lock_time_ms = rtcm::minimum_lock_time_ms(msm.kind, signal.lock_time_indicator);
            let divergence = classify_divergence(previous, msm, signal, min_lock_time_ms);
            if previous.is_none_or(|state| {
                rtcm::msm_epoch_dt_ms(msm.system, state.raw_epoch_time, msm.header.epoch_time) != 0
            }) {
                raw_states.insert(
                    state_key,
                    RawState {
                        raw_epoch_time: msm.header.epoch_time,
                        raw_lock_indicator: signal.lock_time_indicator,
                    },
                );
            }

            let key = ObsKey {
                epoch_ms: msm_epoch_gpst_ms(msm.system, msm.header.epoch_time),
                satellite: rinex_satellite(msm.system, signal.satellite_id),
                phase_code: phase_code.clone(),
            };
            joined_signal_codes.insert((signal.signal_id, phase_code));
            assert!(
                observations
                    .insert(
                        key.clone(),
                        OurObservation {
                            lli: cell.lli & 0b11,
                            divergence,
                        },
                    )
                    .is_none(),
                "duplicate sidereon observation {key:?}"
            );
        }
    }

    OurObservations {
        observations,
        joined_signal_codes,
    }
}

fn classify_divergence(
    previous: Option<RawState>,
    msm: &rtcm::MsmMessage,
    signal: &rtcm::MsmSignal,
    min_lock_time_ms: Option<u32>,
) -> Divergence {
    let Some(previous) = previous else {
        return if signal.lock_time_indicator == 0 {
            Divergence::D1
        } else {
            Divergence::None
        };
    };
    let Some(current_min_lock_ms) = min_lock_time_ms else {
        return Divergence::None;
    };
    let dt = rtcm::msm_epoch_dt_ms(msm.system, previous.raw_epoch_time, msm.header.epoch_time);
    let no_raw_decrease = signal.lock_time_indicator >= previous.raw_lock_indicator;
    let bucket_zero_stall = signal.lock_time_indicator == 0 && previous.raw_lock_indicator == 0;
    if no_raw_decrease && !bucket_zero_stall && u64::from(current_min_lock_ms) < dt {
        Divergence::D2
    } else {
        Divergence::None
    }
}

fn rinex_system_char(system: GnssSystem) -> char {
    match system {
        GnssSystem::Gps => 'G',
        GnssSystem::Glonass => 'R',
        GnssSystem::Galileo => 'E',
        GnssSystem::Sbas => 'S',
        GnssSystem::Qzss => 'J',
        GnssSystem::BeiDou => 'C',
        GnssSystem::Navic => 'I',
    }
}

fn rinex_satellite(system: GnssSystem, satellite_id: u8) -> String {
    match system {
        GnssSystem::Gps => format!("G{satellite_id:02}"),
        GnssSystem::Glonass => format!("R{satellite_id:02}"),
        GnssSystem::Galileo => format!("E{satellite_id:02}"),
        GnssSystem::Sbas => format!("S{:02}", satellite_id + 19),
        GnssSystem::Qzss => format!("J{satellite_id:02}"),
        GnssSystem::BeiDou => format!("C{satellite_id:02}"),
        GnssSystem::Navic => format!("I{satellite_id:02}"),
    }
}

fn msm_epoch_gpst_ms(system: GnssSystem, raw_epoch_time: u32) -> u64 {
    match system {
        GnssSystem::BeiDou => wrap_week(i64::from(raw_epoch_time) + BDT_TO_GPST_MS),
        GnssSystem::Glonass => {
            let day = raw_epoch_time >> GLONASS_DAY_SHIFT;
            let ms = i64::from(raw_epoch_time & GLONASS_MS_MASK);
            let glonass_week_ms = if day == 7 {
                ms
            } else {
                i64::from(day) * DAY_MS + ms
            };
            wrap_week(glonass_week_ms - GLONASS_UTC_PLUS_3_MS + LEAP_SECONDS_MS)
        }
        GnssSystem::Gps
        | GnssSystem::Galileo
        | GnssSystem::Sbas
        | GnssSystem::Qzss
        | GnssSystem::Navic => wrap_week(i64::from(raw_epoch_time)),
    }
}

fn rinex_epoch_ms_of_week(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    seconds: &str,
) -> u64 {
    let days = days_from_civil(year, month, day) - days_from_civil(1980, 1, 6);
    let day_ms = i64::from(hour) * 3_600_000
        + i64::from(minute) * 60_000
        + i64::try_from(parse_second_ms(seconds)).expect("seconds fit in i64");
    wrap_week(days * DAY_MS + day_ms)
}

fn parse_second_ms(seconds: &str) -> u64 {
    let (whole, frac) = seconds.split_once('.').unwrap_or((seconds, ""));
    let mut frac_ms = String::from(frac);
    while frac_ms.len() < 3 {
        frac_ms.push('0');
    }
    let frac_ms = &frac_ms[..3];
    whole.parse::<u64>().expect("whole seconds") * 1000
        + frac_ms.parse::<u64>().expect("millisecond fraction")
}

fn wrap_week(ms: i64) -> u64 {
    ms.rem_euclid(WEEK_MS) as u64
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let year = year - i32::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let shifted_month = month as i32 + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day as i32 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    i64::from(era) * 146_097 + i64::from(day_of_era) - 719_468
}
