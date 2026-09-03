use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::format::fmtnum::fixed_decimals;
use crate::id::GnssSystem;
use crate::rinex_qc::Severity;

use super::{
    IntervalSource, MpStats, ObservationQcHeader, ObservationQcReport, ObservationQcTime, SnrStats,
};

const SNR_COL_WIDTH: usize = 96;

/// Renders an [`ObservationQcReport`] as a plain-text `RINEX OBSERVATION QC +QC SUMMARY` report.
/// The result contains a header, a fixed-width per-constellation table, and a findings section separated by blank lines; absent or empty header fields are omitted, SNR and multipath values use fixed-decimal formatting, and an empty findings list is printed as `NONE`.
pub fn render_text(report: &ObservationQcReport) -> String {
    let mut out = String::new();
    writeln!(&mut out, "RINEX OBSERVATION QC +QC SUMMARY").expect("write to string");
    writeln!(&mut out).expect("write to string");
    write_header(&mut out, report);
    writeln!(&mut out).expect("write to string");
    write_system_table(&mut out, report);
    writeln!(&mut out).expect("write to string");
    write_findings(&mut out, report);
    out
}

#[derive(Debug, Clone)]
pub(super) struct SystemRenderRow {
    pub system_letter: char,
    pub system_name: &'static str,
    pub satellites_seen: usize,
    pub epochs: usize,
    pub observations: usize,
    pub expected: usize,
    pub completeness: String,
    pub snr: String,
    pub mp1: String,
    pub mp2: String,
    pub slips: usize,
    pub gaps: usize,
    pub gap_s: String,
}

pub(super) fn system_rows(report: &ObservationQcReport) -> Vec<SystemRenderRow> {
    let slips_by_system = report
        .cycle_slips
        .by_system
        .iter()
        .map(|row| (row.system, row.slips))
        .collect::<BTreeMap<_, _>>();
    let mp_by_system = report
        .multipath
        .systems
        .iter()
        .map(|row| (row.system, row))
        .collect::<BTreeMap<_, _>>();

    report
        .systems
        .iter()
        .map(|row| {
            let mp = mp_by_system.get(&row.system).copied();
            SystemRenderRow {
                system_letter: row.system.letter(),
                system_name: row.system.as_str(),
                satellites_seen: row.satellites_seen,
                epochs: row.epochs_with_observations,
                observations: row.value_observations,
                expected: row.expected_observations,
                completeness: format_optional_ratio(row.completeness_ratio),
                snr: fit_ascii(&snr_by_band(report, row.system), SNR_COL_WIDTH),
                mp1: format_optional_mp(mp.and_then(|mp| mp.mp1)),
                mp2: format_optional_mp(mp.and_then(|mp| mp.mp2)),
                slips: *slips_by_system.get(&row.system).unwrap_or(&0),
                gaps: row.gap_count,
                gap_s: fixed_decimals(row.total_gap_s, 1),
            }
        })
        .collect()
}

pub(super) fn severity_label(severity: Severity) -> &'static str {
    match severity {
        Severity::Fatal => "FATAL",
        Severity::Error => "ERROR",
        Severity::Warning => "WARN",
        Severity::Info => "INFO",
    }
}

pub(super) fn format_epoch_time(time: &ObservationQcTime) -> String {
    let epoch = time.epoch;
    let mut second = fixed_decimals(epoch.second, 7);
    if epoch.second < 10.0 {
        second.insert(0, '0');
    }
    let mut text = format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{}",
        epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, second
    );
    if let Some(scale) = &time.time_scale {
        let _ = write!(text, " {scale}");
    }
    text
}

pub(super) fn format_vec3_m(values: [f64; 3]) -> String {
    format!(
        "{} {} {}",
        fixed_decimals(values[0], 4),
        fixed_decimals(values[1], 4),
        fixed_decimals(values[2], 4)
    )
}

fn write_header(out: &mut String, report: &ObservationQcReport) {
    writeln!(out, "HEADER").expect("write to string");
    let header = &report.header;
    push_header_field(out, "MARKER NAME", header.marker_name.as_deref());
    push_header_field(out, "MARKER NUMBER", header.marker_number.as_deref());
    push_header_field(out, "MARKER TYPE", header.marker_type.as_deref());
    push_header_field(out, "RECEIVER", format_receiver(header).as_deref());
    push_header_field(out, "ANTENNA", format_antenna(header).as_deref());
    push_header_field(
        out,
        "POSITION XYZ M",
        header.approx_position_m.map(format_vec3_m).as_deref(),
    );
    push_header_field(
        out,
        "ANTENNA HEN M",
        header.antenna_delta_hen_m.map(format_vec3_m).as_deref(),
    );
    push_header_field(
        out,
        "TIME FIRST",
        header
            .time_of_first_obs
            .as_ref()
            .map(format_epoch_time)
            .as_deref(),
    );
    push_header_field(
        out,
        "TIME LAST",
        header
            .time_of_last_obs
            .as_ref()
            .map(format_epoch_time)
            .as_deref(),
    );
    push_header_field(out, "INTERVAL S", format_interval(report).as_deref());
    push_header_field(
        out,
        "DURATION S",
        header
            .duration_s
            .map(|value| fixed_decimals(value, 1))
            .as_deref(),
    );
}

fn push_header_field(out: &mut String, label: &str, value: Option<&str>) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        writeln!(out, "  {label:<18} {value}").expect("write to string");
    }
}

fn format_receiver(header: &ObservationQcHeader) -> Option<String> {
    header.receiver.as_ref().and_then(|receiver| {
        join_non_empty([
            receiver.number.as_str(),
            receiver.receiver_type.as_str(),
            receiver.version.as_str(),
        ])
    })
}

fn format_antenna(header: &ObservationQcHeader) -> Option<String> {
    header.antenna.as_ref().and_then(|antenna| {
        join_non_empty([antenna.number.as_str(), antenna.antenna_type.as_str()])
    })
}

fn join_non_empty<const N: usize>(parts: [&str; N]) -> Option<String> {
    let joined = parts
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" / ");
    (!joined.is_empty()).then_some(joined)
}

fn format_interval(report: &ObservationQcReport) -> Option<String> {
    report.interval_s.map(|interval_s| {
        format!(
            "{} ({})",
            fixed_decimals(interval_s, 3),
            interval_source_label(report.interval_source)
        )
    })
}

fn interval_source_label(source: IntervalSource) -> &'static str {
    match source {
        IntervalSource::Override => "override",
        IntervalSource::Header => "header",
        IntervalSource::Inferred => "inferred",
        IntervalSource::Unresolved => "unresolved",
    }
}

fn write_system_table(out: &mut String, report: &ObservationQcReport) {
    writeln!(out, "PER-CONSTELLATION").expect("write to string");
    writeln!(
        out,
        "{:<3} {:<8} {:>4} {:>6} {:>8} {:>8} {:>8} {:<snr_width$} {:>8} {:>8} {:>6} {:>4} {:>9}",
        "SYS",
        "NAME",
        "SATS",
        "EPOCHS",
        "OBS",
        "EXPECT",
        "COMP",
        "SNR MEAN/MIN BY BAND",
        "MP1 RMS",
        "MP2 RMS",
        "SLIPS",
        "GAPS",
        "GAP S",
        snr_width = SNR_COL_WIDTH
    )
    .expect("write to string");
    writeln!(
        out,
        "{:<3} {:<8} {:>4} {:>6} {:>8} {:>8} {:>8} {:<snr_width$} {:>8} {:>8} {:>6} {:>4} {:>9}",
        "---",
        "--------",
        "----",
        "------",
        "--------",
        "--------",
        "--------",
        "-".repeat(SNR_COL_WIDTH),
        "--------",
        "--------",
        "------",
        "----",
        "---------",
        snr_width = SNR_COL_WIDTH
    )
    .expect("write to string");

    let rows = system_rows(report);
    if rows.is_empty() {
        writeln!(out, "  NONE").expect("write to string");
        return;
    }

    for row in rows {
        writeln!(
            out,
            "{:<3} {:<8} {:>4} {:>6} {:>8} {:>8} {:>8} {:<snr_width$} {:>8} {:>8} {:>6} {:>4} {:>9}",
            row.system_letter,
            row.system_name,
            row.satellites_seen,
            row.epochs,
            row.observations,
            row.expected,
            row.completeness,
            row.snr,
            row.mp1,
            row.mp2,
            row.slips,
            row.gaps,
            row.gap_s,
            snr_width = SNR_COL_WIDTH
        )
        .expect("write to string");
    }
}

fn write_findings(out: &mut String, report: &ObservationQcReport) {
    writeln!(out, "FINDINGS").expect("write to string");
    writeln!(out, "{:<8} {:<8} SPEC REF", "CODE", "SEVERITY").expect("write to string");
    writeln!(
        out,
        "-------- -------- ------------------------------------------------"
    )
    .expect("write to string");
    if report.lint_findings.is_empty() {
        writeln!(out, "NONE").expect("write to string");
        return;
    }

    for finding in &report.lint_findings {
        writeln!(
            out,
            "{:<8} {:<8} {}",
            finding.code,
            severity_label(finding.severity),
            finding.spec_ref
        )
        .expect("write to string");
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SnrBandAccum {
    n: usize,
    weighted_mean_sum: f64,
    min: Option<f64>,
}

impl SnrBandAccum {
    fn add(&mut self, stats: SnrStats) {
        self.n += stats.n;
        self.weighted_mean_sum += stats.mean * stats.n as f64;
        self.min = Some(self.min.map_or(stats.min, |min| min.min(stats.min)));
    }

    fn finish(self) -> Option<(f64, f64)> {
        let min = self.min?;
        (self.n > 0).then(|| (self.weighted_mean_sum / self.n as f64, min))
    }
}

fn snr_by_band(report: &ObservationQcReport, system: GnssSystem) -> String {
    let mut bands = BTreeMap::<char, SnrBandAccum>::new();
    for signal in report
        .system_signals
        .iter()
        .filter(|signal| signal.system == system)
    {
        if !signal.code.starts_with('S') {
            continue;
        }
        let Some(stats) = signal.snr else {
            continue;
        };
        let band = signal.code.chars().nth(1).unwrap_or('?');
        bands.entry(band).or_default().add(stats);
    }

    let text = bands
        .into_iter()
        .filter_map(|(band, accum)| {
            let (mean, min) = accum.finish()?;
            Some(format!(
                "{band}:{}/{}",
                fixed_decimals(mean, 1),
                fixed_decimals(min, 1)
            ))
        })
        .collect::<Vec<_>>()
        .join(" ");
    if text.is_empty() {
        "-".to_string()
    } else {
        text
    }
}

fn format_optional_ratio(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| fixed_decimals(value, 3))
        .unwrap_or_else(|| "-".to_string())
}

fn format_optional_mp(value: Option<MpStats>) -> String {
    value
        .map(|stats| stats.rms_m)
        .filter(|value| value.is_finite())
        .map(|value| fixed_decimals(value, 3))
        .unwrap_or_else(|| "-".to_string())
}

fn fit_ascii(value: &str, width: usize) -> String {
    if value.len() <= width {
        value.to_string()
    } else {
        value[..width].to_string()
    }
}
