use std::fmt::Write as _;

use crate::format::fmtnum::fixed_decimals;

use super::report_text::{format_epoch_time, format_vec3_m, severity_label, system_rows};
use super::{IntervalSource, ObservationQcHeader, ObservationQcReport};

/// Render an [`ObservationQcReport`] as a self-contained HTML document.
///
/// The result contains inline CSS and `Header`, `Per-Constellation`, and
/// `Findings` sections: header metadata, shared formatted constellation rows,
/// and finding code, severity, and specification-reference cells. Report
/// text is HTML-escaped, optional header fields are omitted when absent or
/// empty, and empty constellation or finding collections produce a `None` row.
pub fn render_html(report: &ObservationQcReport) -> String {
    let mut out = String::new();
    out.push_str(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>RINEX Observation QC</title>",
    );
    out.push_str("<style>");
    out.push_str(
        "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;color:#1f2933;background:#f7f9fb}\
         h1{font-size:22px;margin:0 0 18px}h2{font-size:16px;margin:22px 0 10px}\
         table{border-collapse:collapse;width:100%;background:#fff}th,td{border:1px solid #d8dee6;padding:6px 8px;text-align:right}\
         th{background:#e9eef4;font-weight:600}td:first-child,td:nth-child(2),th:first-child,th:nth-child(2),.text{text-align:left}\
         dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 14px;background:#fff;border:1px solid #d8dee6;padding:12px}\
         dt{font-weight:600;color:#52616f}dd{margin:0}.section{max-width:1180px}",
    );
    out.push_str("</style></head><body><main class=\"section\">");
    out.push_str("<h1>RINEX Observation QC +qc Summary</h1>");
    write_header(&mut out, report);
    write_system_table(&mut out, report);
    write_findings(&mut out, report);
    out.push_str("</main></body></html>");
    out
}

fn write_header(out: &mut String, report: &ObservationQcReport) {
    out.push_str("<h2>Header</h2><dl>");
    let header = &report.header;
    push_detail(out, "Marker name", header.marker_name.as_deref());
    push_detail(out, "Marker number", header.marker_number.as_deref());
    push_detail(out, "Marker type", header.marker_type.as_deref());
    push_detail(out, "Receiver", format_receiver(header).as_deref());
    push_detail(out, "Antenna", format_antenna(header).as_deref());
    push_detail(
        out,
        "Position XYZ m",
        header.approx_position_m.map(format_vec3_m).as_deref(),
    );
    push_detail(
        out,
        "Antenna HEN m",
        header.antenna_delta_hen_m.map(format_vec3_m).as_deref(),
    );
    push_detail(
        out,
        "Time first",
        header
            .time_of_first_obs
            .as_ref()
            .map(format_epoch_time)
            .as_deref(),
    );
    push_detail(
        out,
        "Time last",
        header
            .time_of_last_obs
            .as_ref()
            .map(format_epoch_time)
            .as_deref(),
    );
    push_detail(out, "Interval s", format_interval(report).as_deref());
    push_detail(
        out,
        "Duration s",
        header
            .duration_s
            .map(|value| fixed_decimals(value, 1))
            .as_deref(),
    );
    out.push_str("</dl>");
}

fn write_system_table(out: &mut String, report: &ObservationQcReport) {
    out.push_str("<h2>Per-Constellation</h2><table><thead><tr>");
    for heading in [
        "Sys",
        "Name",
        "Sats",
        "Epochs",
        "Obs",
        "Expect",
        "Comp",
        "SNR mean/min by band",
        "MP1 RMS",
        "MP2 RMS",
        "Slips",
        "Gaps",
        "Gap s",
    ] {
        let _ = write!(out, "<th>{}</th>", escape_html(heading));
    }
    out.push_str("</tr></thead><tbody>");
    let rows = system_rows(report);
    if rows.is_empty() {
        out.push_str("<tr><td class=\"text\" colspan=\"13\">None</td></tr>");
    } else {
        for row in rows {
            out.push_str("<tr>");
            cell(out, &row.system_letter.to_string(), true);
            cell(out, row.system_name, true);
            cell(out, &row.satellites_seen.to_string(), false);
            cell(out, &row.epochs.to_string(), false);
            cell(out, &row.observations.to_string(), false);
            cell(out, &row.expected.to_string(), false);
            cell(out, &row.completeness, false);
            cell(out, &row.snr, true);
            cell(out, &row.mp1, false);
            cell(out, &row.mp2, false);
            cell(out, &row.slips.to_string(), false);
            cell(out, &row.gaps.to_string(), false);
            cell(out, &row.gap_s, false);
            out.push_str("</tr>");
        }
    }
    out.push_str("</tbody></table>");
}

fn write_findings(out: &mut String, report: &ObservationQcReport) {
    out.push_str("<h2>Findings</h2><table><thead><tr>");
    for heading in ["Code", "Severity", "Spec ref"] {
        let _ = write!(out, "<th>{}</th>", escape_html(heading));
    }
    out.push_str("</tr></thead><tbody>");
    if report.lint_findings.is_empty() {
        out.push_str("<tr><td class=\"text\" colspan=\"3\">None</td></tr>");
    } else {
        for finding in &report.lint_findings {
            out.push_str("<tr>");
            cell(out, &finding.code, true);
            cell(out, severity_label(finding.severity), true);
            cell(out, &finding.spec_ref, true);
            out.push_str("</tr>");
        }
    }
    out.push_str("</tbody></table>");
}

fn push_detail(out: &mut String, label: &str, value: Option<&str>) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        let _ = write!(
            out,
            "<dt>{}</dt><dd>{}</dd>",
            escape_html(label),
            escape_html(value)
        );
    }
}

fn cell(out: &mut String, value: &str, text: bool) {
    if text {
        let _ = write!(out, "<td class=\"text\">{}</td>", escape_html(value));
    } else {
        let _ = write!(out, "<td>{}</td>", escape_html(value));
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

fn escape_html(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}
