//! Independent SP3 evaluation oracle.
//!
//! Fixture: `tests/fixtures/sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3`,
//! a public IGS MGEX final orbit and clock product for 2020-06-25, committed
//! in this crate's existing SP3 fixture set. The checks below read the SP3 text
//! records directly with fixed-column parsing in the test. Expected positions
//! and clocks are not taken from `Sp3::state`, `Sp3::position_at_j2000_seconds`,
//! or the cached interpolant.

use sidereon_core::astro::time::civil::{j2000_seconds, j2000_seconds_from_split};
use sidereon_core::astro::time::model::{Instant, JulianDateSplit, TimeScale};
use sidereon_core::constants::{J2000_JD, SECONDS_PER_DAY};
use sidereon_core::ephemeris::{
    observable_states_at_j2000_s, PreciseEphemerisInterpolant, PreciseEphemerisSample,
    PreciseEphemerisSamples, Sp3,
};
use sidereon_core::{Error, GnssSatelliteId, GnssSystem};
use std::collections::BTreeMap;

const COD_5M_FIXTURE: &str = "tests/fixtures/sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3";
const GBM_5M_TRIM_FIXTURE: &str =
    "tests/fixtures/sp3/GBM0MGXRAP_20201770000_01D_05M_ORB_120epoch.sp3";
const GAP_15M_FIXTURE: &str = "tests/fixtures/sp3/GAP_G01_20201760000_15M.sp3";
const SP3_POSITION_RESOLUTION_M: f64 = 1.0e-3;
const SP3_CLOCK_RESOLUTION_S: f64 = 1.0e-12;
const DECIMATION_STRIDE: usize = 3;
const ADJUDICATION_RMS_RATIO_BOUND: f64 = 2.0;

#[derive(Debug, Clone, Copy)]
struct TextEpoch {
    j2000_s: f64,
    index: usize,
    is_boundary_node: bool,
}

#[derive(Debug, Clone, Copy)]
struct TextRecord {
    epoch: TextEpoch,
    sat: GnssSatelliteId,
    position_m: [f64; 3],
    clock_s: Option<f64>,
}

fn gps(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS satellite")
}

fn fixture_text(name: &str) -> String {
    std::fs::read_to_string(std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name))
        .expect("read SP3 fixture")
}

fn parse_epoch(line: &str, index: usize) -> TextEpoch {
    let fields: Vec<_> = line[1..].split_whitespace().collect();
    assert_eq!(fields.len(), 6, "unexpected SP3 epoch line: {line}");
    let year = fields[0].parse::<i32>().expect("year");
    let month = fields[1].parse::<i32>().expect("month");
    let day = fields[2].parse::<i32>().expect("day");
    let hour = fields[3].parse::<i32>().expect("hour");
    let minute = fields[4].parse::<i32>().expect("minute");
    let second = fields[5].parse::<f64>().expect("second");
    let j2000_s = j2000_seconds(year, month, day, hour, minute, second);
    let whole = j2000_s.round() as i64;
    TextEpoch {
        j2000_s,
        index,
        is_boundary_node: whole.rem_euclid(2700) == 1800,
    }
}

fn parse_record(line: &str, epoch: TextEpoch) -> Option<TextRecord> {
    if !line.starts_with('P') {
        return None;
    }
    let sat = line.get(1..4)?.trim().parse::<GnssSatelliteId>().ok()?;
    let x_km = line.get(4..18)?.trim().parse::<f64>().expect("x km");
    let y_km = line.get(18..32)?.trim().parse::<f64>().expect("y km");
    let z_km = line.get(32..46)?.trim().parse::<f64>().expect("z km");
    if x_km == 0.0 && y_km == 0.0 && z_km == 0.0 {
        return None;
    }
    let clock_us = line
        .get(46..60)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<f64>().expect("clock us"))
        .filter(|value| value.abs() < 999_999.999_999);
    Some(TextRecord {
        epoch,
        sat,
        position_m: [x_km * 1000.0, y_km * 1000.0, z_km * 1000.0],
        clock_s: clock_us.map(|value| value * 1.0e-6),
    })
}

fn text_records(text: &str, sat: GnssSatelliteId) -> Vec<TextRecord> {
    let mut current_epoch = None;
    let mut epoch_index = 0usize;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with('*') {
            current_epoch = Some(parse_epoch(line, epoch_index));
            epoch_index += 1;
        } else if let Some(epoch) = current_epoch {
            if let Some(record) = parse_record(line, epoch).filter(|record| record.sat == sat) {
                out.push(record);
            }
        }
    }
    out
}

fn all_text_records(text: &str) -> Vec<TextRecord> {
    let mut current_epoch = None;
    let mut epoch_index = 0usize;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with('*') {
            current_epoch = Some(parse_epoch(line, epoch_index));
            epoch_index += 1;
        } else if let Some(epoch) = current_epoch {
            if let Some(record) = parse_record(line, epoch) {
                out.push(record);
            }
        }
    }
    out
}

fn decimated_product_text(text: &str) -> String {
    let epoch_count = text.lines().filter(|line| line.starts_with('*')).count();
    let keep_count = epoch_count.div_ceil(DECIMATION_STRIDE);
    let mut out = String::with_capacity(text.len() / DECIMATION_STRIDE + 1024);
    let mut epoch_index = 0usize;
    let mut keep_current_epoch = true;
    for line in text.lines() {
        if line
            .strip_prefix("EOF")
            .is_some_and(|padding| padding.bytes().all(|byte| byte == b' '))
        {
            continue;
        }
        if line.starts_with('*') {
            keep_current_epoch = epoch_index.is_multiple_of(DECIMATION_STRIDE);
            if keep_current_epoch {
                out.push_str(line);
                out.push('\n');
            }
            epoch_index += 1;
        } else if line.starts_with('#') && !line.starts_with("##") {
            out.push_str(&replace_field(line, 32, 40, &format!("{keep_count:>8}")));
            out.push('\n');
        } else if line.starts_with("##") {
            out.push_str(&replace_field(line, 24, 38, &format!("{:14.8}", 900.0)));
            out.push('\n');
        } else if keep_current_epoch {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.push_str("EOF\n");
    out
}

fn replace_field(line: &str, start: usize, end: usize, replacement: &str) -> String {
    assert_eq!(
        replacement.len(),
        end - start,
        "replacement must match fixed-width field"
    );
    assert!(
        line.len() >= end,
        "line too short for fixed-width replacement: {line:?}"
    );
    let mut out = String::with_capacity(line.len());
    out.push_str(&line[..start]);
    out.push_str(replacement);
    out.push_str(&line[end..]);
    out
}

fn decimated_samples(source: &Sp3) -> Vec<PreciseEphemerisSample> {
    let mut out = Vec::new();
    for (idx, &epoch) in source.epochs.iter().enumerate() {
        if !idx.is_multiple_of(DECIMATION_STRIDE) {
            continue;
        }
        let states = source.states_at(idx).expect("decimated epoch in range");
        for (&sat, state) in states {
            out.push(PreciseEphemerisSample {
                sat,
                epoch,
                position_ecef_m: state.position.as_array(),
                clock_s: state.clock_s,
                clock_event: state.flags.clock_event,
            });
        }
    }
    out
}

#[derive(Debug, Clone, Copy, Default)]
struct SchemeScore {
    count: u64,
    sum_sq_3d_m2: f64,
    max_3d_m: f64,
}

impl SchemeScore {
    fn record(&mut self, error_3d_m: f64) {
        self.count += 1;
        self.sum_sq_3d_m2 += error_3d_m * error_3d_m;
        self.max_3d_m = self.max_3d_m.max(error_3d_m);
    }

    fn rms_3d_m(self) -> f64 {
        (self.sum_sq_3d_m2 / self.count as f64).sqrt()
    }
}

#[derive(Debug, Clone, Default)]
struct AdjudicationStats {
    parsed: SchemeScore,
    samples: SchemeScore,
    path_delta: SchemeScore,
    parsed_wins: u64,
    sample_wins: u64,
    ties: u64,
    /// Held-out records both schemes refuse because the run serving them holds
    /// fewer than the eleven nodes the interpolator takes.
    refused: u64,
    per_sat: BTreeMap<GnssSatelliteId, (SchemeScore, SchemeScore)>,
}

impl AdjudicationStats {
    fn record(
        &mut self,
        sat: GnssSatelliteId,
        parsed_error_3d_m: f64,
        sample_error_3d_m: f64,
        path_delta_3d_m: f64,
    ) {
        self.parsed.record(parsed_error_3d_m);
        self.samples.record(sample_error_3d_m);
        self.path_delta.record(path_delta_3d_m);
        match parsed_error_3d_m.partial_cmp(&sample_error_3d_m) {
            Some(std::cmp::Ordering::Less) => self.parsed_wins += 1,
            Some(std::cmp::Ordering::Greater) => self.sample_wins += 1,
            _ => self.ties += 1,
        }
        let entry = self.per_sat.entry(sat).or_default();
        entry.0.record(parsed_error_3d_m);
        entry.1.record(sample_error_3d_m);
    }

    fn merge(&mut self, other: &Self) {
        self.parsed.count += other.parsed.count;
        self.parsed.sum_sq_3d_m2 += other.parsed.sum_sq_3d_m2;
        self.parsed.max_3d_m = self.parsed.max_3d_m.max(other.parsed.max_3d_m);
        self.samples.count += other.samples.count;
        self.samples.sum_sq_3d_m2 += other.samples.sum_sq_3d_m2;
        self.samples.max_3d_m = self.samples.max_3d_m.max(other.samples.max_3d_m);
        self.path_delta.count += other.path_delta.count;
        self.path_delta.sum_sq_3d_m2 += other.path_delta.sum_sq_3d_m2;
        self.path_delta.max_3d_m = self.path_delta.max_3d_m.max(other.path_delta.max_3d_m);
        self.parsed_wins += other.parsed_wins;
        self.sample_wins += other.sample_wins;
        self.ties += other.ties;
        self.refused += other.refused;
        for (&sat, &(parsed, samples)) in &other.per_sat {
            let entry = self.per_sat.entry(sat).or_default();
            entry.0.count += parsed.count;
            entry.0.sum_sq_3d_m2 += parsed.sum_sq_3d_m2;
            entry.0.max_3d_m = entry.0.max_3d_m.max(parsed.max_3d_m);
            entry.1.count += samples.count;
            entry.1.sum_sq_3d_m2 += samples.sum_sq_3d_m2;
            entry.1.max_3d_m = entry.1.max_3d_m.max(samples.max_3d_m);
        }
    }
}

/// Held-out epochs near the decimated product's first and last nodes are
/// evaluated outside the interpolation window's full support, so their errors
/// (metre-scale, versus centimetre-scale in the interior) reflect boundary
/// extrapolation, not interpolation quality. Both paths behave identically
/// there; the aggregate bounds below include those epochs deliberately.
fn score_decimation_holdout(fixture: &str) -> AdjudicationStats {
    let text = fixture_text(fixture);
    let full = Sp3::parse(text.as_bytes()).expect("parse full SP3 fixture");
    let full_epoch_count = full.epoch_count();
    let decimated_text = decimated_product_text(&text);
    let decimated_parsed = Sp3::parse(decimated_text.as_bytes()).expect("parse decimated SP3");
    assert_eq!(
        decimated_parsed.header.epoch_interval_s, 900.0,
        "decimated product should declare 15 minute spacing"
    );
    let parsed_path = PreciseEphemerisInterpolant::from_sp3(&decimated_parsed);
    let sample_path = PreciseEphemerisInterpolant::from_samples(decimated_samples(&full))
        .expect("sample-backed decimated source");

    let mut stats = AdjudicationStats::default();
    for record in all_text_records(&text) {
        if record.epoch.index.is_multiple_of(DECIMATION_STRIDE) {
            continue;
        }
        let next_node =
            record.epoch.index + (DECIMATION_STRIDE - record.epoch.index % DECIMATION_STRIDE);
        if next_node >= full_epoch_count {
            continue;
        }
        let parsed = parsed_path.position_at_j2000_seconds(record.sat, record.epoch.j2000_s);
        let sample = sample_path.position_at_j2000_seconds(record.sat, record.epoch.j2000_s);
        if let Err(Error::InsufficientPreciseNodes { .. }) = parsed {
            assert_eq!(
                sample.map(|state| state.position),
                parsed.map(|state| state.position),
                "both schemes refuse a short run alike"
            );
            stats.refused += 1;
            continue;
        }
        let parsed_state = parsed.expect("parsed decimated interpolation");
        let sample_state = sample.expect("sample decimated interpolation");
        stats.record(
            record.sat,
            error_3d_m(parsed_state.position.as_array(), record.position_m),
            error_3d_m(sample_state.position.as_array(), record.position_m),
            error_3d_m(
                parsed_state.position.as_array(),
                sample_state.position.as_array(),
            ),
        );
    }

    assert_eq!(
        stats.parsed.count, stats.samples.count,
        "schemes must score the same held-out records"
    );
    stats
}

fn error_3d_m(got: [f64; 3], want: [f64; 3]) -> f64 {
    let dx = got[0] - want[0];
    let dy = got[1] - want[1];
    let dz = got[2] - want[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn assert_close(name: &str, got: f64, want: f64, tolerance: f64) {
    let delta = (got - want).abs();
    assert!(
        delta <= tolerance,
        "{name}: got {got:.15e}, want {want:.15e}, delta {delta:.3e}"
    );
}

fn assert_win_rate_close(name: &str, got_count: u64, want_count: u64, total: u64) {
    const WIN_RATE_TOL: f64 = 2.0e-3;
    let got = got_count as f64 / total as f64;
    let want = want_count as f64 / total as f64;
    let delta = (got - want).abs();
    assert!(
        delta <= WIN_RATE_TOL,
        "{name}: got {got:.6}, want {want:.6}, delta {delta:.6}"
    );
}

fn worst_satellite_rms(stats: &AdjudicationStats) -> (GnssSatelliteId, f64, f64) {
    stats
        .per_sat
        .iter()
        .map(|(&sat, &(parsed, samples))| (sat, parsed.rms_3d_m(), samples.rms_3d_m()))
        .max_by(|a, b| {
            a.1.max(a.2)
                .partial_cmp(&b.1.max(b.2))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("per-satellite stats")
}

fn worst_satellite_max(stats: &AdjudicationStats) -> (GnssSatelliteId, f64, f64) {
    stats
        .per_sat
        .iter()
        .map(|(&sat, &(parsed, samples))| (sat, parsed.max_3d_m, samples.max_3d_m))
        .max_by(|a, b| {
            a.1.max(a.2)
                .partial_cmp(&b.1.max(b.2))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("per-satellite stats")
}

#[derive(Debug, Clone, Copy)]
struct ExpectedAdjudication {
    records: u64,
    satellites: usize,
    parsed_rms_3d_m: f64,
    parsed_max_3d_m: f64,
    sample_rms_3d_m: f64,
    sample_max_3d_m: f64,
    path_delta_rms_3d_m: f64,
    path_delta_max_3d_m: f64,
    parsed_wins: u64,
    sample_wins: u64,
    ties: u64,
    worst_rms_sat: &'static str,
    worst_rms_parsed_3d_m: f64,
    worst_rms_sample_3d_m: f64,
    worst_max_sat: &'static str,
    worst_max_parsed_3d_m: f64,
    worst_max_sample_3d_m: f64,
}

fn assert_adjudication(label: &str, stats: &AdjudicationStats, expected: ExpectedAdjudication) {
    const METRIC_TOL_M: f64 = 5.0e-8;
    assert_eq!(stats.parsed.count, expected.records, "{label} record count");
    assert_eq!(
        stats.samples.count, expected.records,
        "{label} sample count"
    );
    assert_eq!(
        stats.per_sat.len(),
        expected.satellites,
        "{label} satellite count"
    );
    assert_close(
        &format!("{label} parsed RMS"),
        stats.parsed.rms_3d_m(),
        expected.parsed_rms_3d_m,
        METRIC_TOL_M,
    );
    assert_close(
        &format!("{label} parsed max"),
        stats.parsed.max_3d_m,
        expected.parsed_max_3d_m,
        METRIC_TOL_M,
    );
    assert_close(
        &format!("{label} sample RMS"),
        stats.samples.rms_3d_m(),
        expected.sample_rms_3d_m,
        METRIC_TOL_M,
    );
    assert_close(
        &format!("{label} sample max"),
        stats.samples.max_3d_m,
        expected.sample_max_3d_m,
        METRIC_TOL_M,
    );
    assert_close(
        &format!("{label} path-delta RMS"),
        stats.path_delta.rms_3d_m(),
        expected.path_delta_rms_3d_m,
        METRIC_TOL_M,
    );
    assert_close(
        &format!("{label} path-delta max"),
        stats.path_delta.max_3d_m,
        expected.path_delta_max_3d_m,
        METRIC_TOL_M,
    );
    assert_eq!(
        stats.parsed_wins + stats.sample_wins + stats.ties,
        expected.records,
        "{label} scored win/tie count"
    );
    assert_win_rate_close(
        &format!("{label} parsed win rate"),
        stats.parsed_wins,
        expected.parsed_wins,
        expected.records,
    );
    assert_win_rate_close(
        &format!("{label} sample win rate"),
        stats.sample_wins,
        expected.sample_wins,
        expected.records,
    );
    assert_win_rate_close(
        &format!("{label} tie rate"),
        stats.ties,
        expected.ties,
        expected.records,
    );

    let (worst_rms_sat, worst_rms_parsed, worst_rms_samples) = worst_satellite_rms(stats);
    assert_eq!(
        worst_rms_sat.to_string(),
        expected.worst_rms_sat,
        "{label} worst RMS satellite"
    );
    assert_close(
        &format!("{label} worst sat parsed RMS"),
        worst_rms_parsed,
        expected.worst_rms_parsed_3d_m,
        METRIC_TOL_M,
    );
    assert_close(
        &format!("{label} worst sat sample RMS"),
        worst_rms_samples,
        expected.worst_rms_sample_3d_m,
        METRIC_TOL_M,
    );

    let (worst_max_sat, worst_max_parsed, worst_max_samples) = worst_satellite_max(stats);
    assert_eq!(
        worst_max_sat.to_string(),
        expected.worst_max_sat,
        "{label} worst max satellite"
    );
    assert_close(
        &format!("{label} worst sat parsed max"),
        worst_max_parsed,
        expected.worst_max_parsed_3d_m,
        METRIC_TOL_M,
    );
    assert_close(
        &format!("{label} worst sat sample max"),
        worst_max_samples,
        expected.worst_max_sample_3d_m,
        METRIC_TOL_M,
    );
}

fn floor_sensitive_45m_fixture_text() -> String {
    let header_src = fixture_text(GAP_15M_FIXTURE);
    let epoch_start = header_src.find("\n*  ").expect("first epoch line") + 1;
    let mut text = String::from(&header_src[..epoch_start]);

    for i in 0..12 {
        let second_of_day = 6 * 3600 + 7 + i * 2700;
        let hour = second_of_day / 3600;
        let minute = (second_of_day % 3600) / 60;
        let second = second_of_day % 60;
        let x_km = 20_000.0 + 10.0 * i as f64;
        let y_km = 14_000.0 + 7.0 * i as f64;
        let z_km = 21_000.0 - 5.0 * i as f64;
        let clock_us = 100.0 + i as f64;
        text.push_str(&format!(
            "*  2000  1  1 {hour:2} {minute:2} {second:2}.00000000\n"
        ));
        text.push_str(&format!(
            "PG01{x_km:14.6}{y_km:14.6}{z_km:14.6}{clock_us:14.6}\n"
        ));
    }
    text.push_str("EOF\n");
    text
}

fn converted_epoch_one_ulp_low(scale: TimeScale, exact_j2000_s: f64) -> Instant {
    assert!(exact_j2000_s.is_sign_positive());
    assert_eq!(
        (exact_j2000_s as i64).rem_euclid(2700),
        1800,
        "fixture epoch must be on the affected 45-minute boundary"
    );
    let converted = f64::from_bits(exact_j2000_s.to_bits() - 1);
    assert_eq!(
        (converted + 2f64.powi(-24)).to_bits(),
        exact_j2000_s.to_bits(),
        "reported bisection should flip the converted epoch onto the node"
    );

    let day = (converted / SECONDS_PER_DAY).floor();
    let fraction = (converted - day * SECONDS_PER_DAY) / SECONDS_PER_DAY;
    let split =
        JulianDateSplit::new(J2000_JD + day, fraction).expect("valid converted split epoch");
    assert_eq!(
        j2000_seconds_from_split(split.jd_whole, split.fraction).to_bits(),
        converted.to_bits(),
        "test helper must exercise the same split-to-J2000 conversion as sample ingestion"
    );
    Instant::from_julian_date(scale, split)
}

fn converted_epoch_samples(records: &[TextRecord]) -> Vec<PreciseEphemerisSample> {
    records
        .iter()
        .map(|record| {
            PreciseEphemerisSample::new(
                record.sat,
                if record.epoch.is_boundary_node {
                    converted_epoch_one_ulp_low(TimeScale::Gpst, record.epoch.j2000_s)
                } else {
                    let day = (record.epoch.j2000_s / SECONDS_PER_DAY).floor();
                    let fraction = (record.epoch.j2000_s - day * SECONDS_PER_DAY) / SECONDS_PER_DAY;
                    Instant::from_julian_date(
                        TimeScale::Gpst,
                        JulianDateSplit::new(J2000_JD + day, fraction)
                            .expect("valid exact split epoch"),
                    )
                },
                record.position_m,
                record.clock_s,
            )
        })
        .collect()
}

fn assert_record_matches(
    record: TextRecord,
    state_position_m: [f64; 3],
    state_clock_s: Option<f64>,
) {
    for (axis, &got) in state_position_m.iter().enumerate() {
        let delta_m = (got - record.position_m[axis]).abs();
        assert!(
            delta_m <= 0.5 * SP3_POSITION_RESOLUTION_M,
            "axis {axis} at {} s differs from SP3 text by {delta_m:e} m",
            record.epoch.j2000_s
        );
    }
    match (state_clock_s, record.clock_s) {
        (Some(got), Some(want)) => {
            let delta_s = (got - want).abs();
            assert!(
                delta_s <= 0.5 * SP3_CLOCK_RESOLUTION_S,
                "clock at {} s differs from SP3 text by {delta_s:e} s",
                record.epoch.j2000_s
            );
        }
        (None, None) => {}
        other => panic!(
            "clock presence mismatch at {} s: {other:?}",
            record.epoch.j2000_s
        ),
    }
}

fn assert_record_matches_all_sample_paths(
    record: TextRecord,
    direct_samples: &PreciseEphemerisSamples,
    cached_samples: &PreciseEphemerisInterpolant,
) {
    let direct = direct_samples
        .position_at_j2000_seconds(record.sat, record.epoch.j2000_s)
        .expect("direct sample-backed state at record epoch");
    assert_record_matches(record, direct.position.as_array(), direct.clock_s);

    let cached = cached_samples
        .position_at_j2000_seconds(record.sat, record.epoch.j2000_s)
        .expect("cached sample-backed state at record epoch");
    assert_record_matches(record, cached.position.as_array(), cached.clock_s);
}

fn assert_state_bits_eq(
    sat: GnssSatelliteId,
    epoch_j2000_s: f64,
    parsed: sidereon_core::ephemeris::Sp3State,
    from_samples: sidereon_core::ephemeris::Sp3State,
) {
    assert_eq!(
        parsed.position.as_array().map(f64::to_bits),
        from_samples.position.as_array().map(f64::to_bits),
        "{sat} position bits differ at {epoch_j2000_s}"
    );
    assert_eq!(
        parsed.clock_s.map(f64::to_bits),
        from_samples.clock_s.map(f64::to_bits),
        "{sat} clock bits differ at {epoch_j2000_s}"
    );
}

#[test]
fn converted_sample_epochs_match_sp3_text_oracle_at_45m_boundaries() {
    let text = fixture_text(COD_5M_FIXTURE);
    let sp3 = Sp3::parse(text.as_bytes()).expect("parse SP3 fixture");
    let sat = gps(1);
    let records: Vec<_> = text_records(&text, sat).into_iter().take(144).collect();
    let boundary_count = records
        .iter()
        .filter(|record| record.epoch.is_boundary_node)
        .count();
    assert_eq!(
        boundary_count, 16,
        "fixture must retain the affected 16/144 boundary-node coverage"
    );

    let direct_samples =
        PreciseEphemerisSamples::from_samples(converted_epoch_samples(&records)).expect("source");
    let cached_samples =
        PreciseEphemerisInterpolant::from_samples(converted_epoch_samples(&records))
            .expect("sample-backed interpolant");

    for record in records {
        let parsed = sp3
            .position_at_j2000_seconds(record.sat, record.epoch.j2000_s)
            .expect("parsed state at record epoch");
        assert_record_matches(record, parsed.position.as_array(), parsed.clock_s);
        assert_record_matches_all_sample_paths(record, &direct_samples, &cached_samples);
    }
}

#[test]
fn exact_record_epochs_match_sp3_text_oracle_on_both_construction_paths() {
    let text = fixture_text(COD_5M_FIXTURE);
    let sp3 = Sp3::parse(text.as_bytes()).expect("parse SP3 fixture");
    let samples = PreciseEphemerisInterpolant::from_samples(sp3.precise_ephemeris_samples())
        .expect("sample-backed interpolant");
    let sat = gps(1);
    let records = text_records(&text, sat);
    assert!(
        records.len() >= 144,
        "fixture does not cover a 12 hour window"
    );
    assert!(
        records.iter().any(|record| record.epoch.is_boundary_node),
        "fixture does not span a 45 minute boundary node"
    );

    for &record in records.iter().take(144) {
        let parsed = sp3
            .position_at_j2000_seconds(record.sat, record.epoch.j2000_s)
            .expect("parsed state at record epoch");
        assert_record_matches(record, parsed.position.as_array(), parsed.clock_s);

        let from_samples = samples
            .position_at_j2000_seconds(record.sat, record.epoch.j2000_s)
            .expect("sample-backed state at record epoch");
        assert_record_matches(
            record,
            from_samples.position.as_array(),
            from_samples.clock_s,
        );
    }
}

#[test]
fn exact_record_epochs_match_sp3_text_oracle_on_45m_sample_backed_path() {
    let text = floor_sensitive_45m_fixture_text();
    let sp3 = Sp3::parse(text.as_bytes()).expect("parse SP3 fixture");
    let samples = PreciseEphemerisInterpolant::from_samples(sp3.precise_ephemeris_samples())
        .expect("sample-backed interpolant");
    let sat = gps(1);
    let records = text_records(&text, sat);
    assert_eq!(records.len(), 12, "fixture record coverage changed");
    for window in records.windows(2) {
        assert_eq!(
            window[1].epoch.j2000_s - window[0].epoch.j2000_s,
            2700.0,
            "fixture is no longer 45-minute cadence"
        );
    }

    for record in records {
        let parsed = sp3
            .position_at_j2000_seconds(record.sat, record.epoch.j2000_s)
            .expect("parsed state at record epoch");
        assert_record_matches(record, parsed.position.as_array(), parsed.clock_s);

        let from_samples = samples
            .position_at_j2000_seconds(record.sat, record.epoch.j2000_s)
            .expect("sample-backed state at record epoch");
        assert_record_matches(
            record,
            from_samples.position.as_array(),
            from_samples.clock_s,
        );
    }
}

#[test]
fn sample_backed_path_matches_parsed_path_bits_at_45m_midpoints() {
    let text = floor_sensitive_45m_fixture_text();
    let sp3 = Sp3::parse(text.as_bytes()).expect("parse SP3 fixture");
    let samples = PreciseEphemerisInterpolant::from_samples(sp3.precise_ephemeris_samples())
        .expect("sample-backed interpolant");
    let sat = gps(1);
    let records = text_records(&text, sat);
    assert_eq!(records.len(), 12, "fixture record coverage changed");

    for window in records.windows(2) {
        let midpoint = 0.5 * (window[0].epoch.j2000_s + window[1].epoch.j2000_s);
        let parsed = sp3
            .position_at_j2000_seconds(sat, midpoint)
            .expect("parsed midpoint state");
        let from_samples = samples
            .position_at_j2000_seconds(sat, midpoint)
            .expect("sample-backed midpoint state");
        assert_state_bits_eq(sat, midpoint, parsed, from_samples);
    }
}

#[test]
fn decimated_real_sp3_holdout_adjudicates_mid_interval_schemes() {
    let cod = score_decimation_holdout(COD_5M_FIXTURE);
    let gbm = score_decimation_holdout(GBM_5M_TRIM_FIXTURE);
    let mut combined = AdjudicationStats::default();
    combined.merge(&cod);
    combined.merge(&gbm);

    // The GBM trim holds 24 five-minute epochs, eight once decimated to fifteen
    // minutes: fewer than the eleven nodes the interpolator takes (RTKLIB
    // pephpos's NMAX + 1), so both schemes refuse every held-out record of it.
    assert_eq!(cod.refused, 0);
    assert_eq!(gbm.parsed.count, 0);
    assert_eq!(gbm.refused, 1_722);

    assert_adjudication(
        "COD",
        &cod,
        ExpectedAdjudication {
            records: 17_280,
            satellites: 90,
            parsed_rms_3d_m: 1.135_841_423_291_068e-2,
            parsed_max_3d_m: 1.280_029_161_638_427,
            sample_rms_3d_m: 1.135_841_423_538_319e-2,
            sample_max_3d_m: 1.280_029_161_638_427,
            path_delta_rms_3d_m: 1.858_826_561_149_592e-9,
            path_delta_max_3d_m: 4.020_965_667_248_365e-8,
            parsed_wins: 668,
            sample_wins: 651,
            ties: 15_961,
            worst_rms_sat: "E18",
            worst_rms_parsed_3d_m: 1.036_000_495_458_185e-1,
            worst_rms_sample_3d_m: 1.036_000_495_575_859e-1,
            worst_max_sat: "E18",
            worst_max_parsed_3d_m: 1.280_029_161_638_427,
            worst_max_sample_3d_m: 1.280_029_161_638_427,
        },
    );
    assert_eq!(combined.parsed.count, cod.parsed.count);
    assert_eq!(combined.refused, gbm.refused);

    let rms_ratio = (combined.parsed.rms_3d_m() / combined.samples.rms_3d_m())
        .max(combined.samples.rms_3d_m() / combined.parsed.rms_3d_m());
    assert!(
        rms_ratio < ADJUDICATION_RMS_RATIO_BOUND,
        "one scheme dominates the hold-out truth: RMS ratio {rms_ratio:.3}"
    );
}

#[test]
fn cached_batch_record_epochs_match_sp3_text_oracle() {
    let text = fixture_text(COD_5M_FIXTURE);
    let sp3 = Sp3::parse(text.as_bytes()).expect("parse SP3 fixture");
    let cached = PreciseEphemerisInterpolant::from_sp3(&sp3);
    let sat = gps(1);
    let records: Vec<_> = text_records(&text, sat)
        .into_iter()
        .filter(|record| record.epoch.is_boundary_node)
        .take(8)
        .collect();
    assert_eq!(records.len(), 8, "fixture boundary-node coverage changed");

    let satellites: Vec<_> = records.iter().map(|record| record.sat).collect();
    let epochs: Vec<_> = records.iter().map(|record| record.epoch.j2000_s).collect();
    let batch =
        observable_states_at_j2000_s(&cached, &satellites, &epochs).expect("batch evaluation");
    for (index, &record) in records.iter().enumerate() {
        assert_eq!(batch.element_results[index], Ok(()));
        assert_record_matches(record, batch.positions_ecef_m[index], batch.clocks_s[index]);
    }
}
