use std::collections::BTreeSet;
use std::f64::consts::PI;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{anyhow, bail, Context, Result};
use clap::{ArgGroup, Parser, Subcommand};
use serde::Serialize;
use serde_json::Value;
use sidereon::antex::AntennaKind;
use sidereon::ephemeris::{
    check_continuity, BroadcastEphemeris, ContinuityOptions, EpochWindow, OrbitClass, Sp3,
    StencilExtent,
};
use sidereon::positioning::{ReceiverSolution, SolvePolicy};
use sidereon::qc_obs::{observation_qc, render_text as render_obs_qc_text};
use sidereon::rinex::qc::{FindingRef, LintReport, Severity};
use sidereon::rinex::ObservationFile;
use sidereon::{
    horizontal_radius_at, load_rinex_nav, load_rinex_obs, load_sp3, metrics_from_enu_covariance_m2,
    metrics_from_position_covariance, parse_antex, parse_rinex_nav, parse_rinex_obs,
    spherical_radius_at, spp_inputs_from_rinex_obs, vertical_radius_at, PercentileRadius,
    PositionErrorMetrics, RinexSppEpochInputs, RinexSppOptions, RinexSppSource,
};

mod mcp;
mod tui;

#[derive(Parser)]
#[command(name = "sidereon")]
#[command(about = "GNSS file inspection, QC, SPP solving, and covariance metrics")]
#[command(
    after_long_help = "JSON output uses stable field names. solve: source, obs, nav, sp3, epochs, summary, errors. qc: obs, lint, qc, parse_error. inspect output is human text with path, type, span, counts, systems, satellites. metrics accepts JSON input through --json-file and emits JSON with --json. tui is an interactive replay monitor for RINEX OBS plus NAV."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Solve RINEX observation epochs with broadcast NAV or SP3 precise orbits.
    Solve {
        /// RINEX observation file.
        #[arg(long)]
        obs: PathBuf,
        /// RINEX broadcast navigation file.
        #[arg(long)]
        nav: PathBuf,
        /// Optional SP3 precise-orbit file. Broadcast NAV is still used for ionosphere and GLONASS context.
        #[arg(long)]
        sp3: Option<PathBuf>,
        /// Emit JSON with stable fields: source, obs, nav, sp3, epochs, summary, errors.
        #[arg(long)]
        json: bool,
    },
    /// Lint a RINEX observation file and print observation QC rollups.
    Qc {
        /// RINEX observation file.
        #[arg(long)]
        obs: PathBuf,
        /// Emit JSON with stable fields: obs, lint, qc, parse_error.
        #[arg(long)]
        json: bool,
    },
    /// Compute position-error metrics from an ENU covariance.
    #[command(group(
        ArgGroup::new("input")
            .required(true)
            .args(["enu_cov", "json_file"])
    ))]
    Metrics {
        /// ENU covariance as nine comma-separated numbers, row-major.
        #[arg(long)]
        enu_cov: Option<String>,
        /// JSON input: a flat 9-array, 3x3 array, or object with enu_covariance_m2.
        #[arg(long)]
        json_file: Option<PathBuf>,
        /// Additional percentile probability for horizontal, vertical, and spherical bounds.
        #[arg(long, default_value_t = 0.95)]
        probability: f64,
        /// Emit JSON with stable fields for covariance and derived metrics.
        #[arg(long)]
        json: bool,
    },
    /// Detect a file type by trying the real parsers.
    Inspect {
        /// File to inspect.
        file: PathBuf,
        /// Inclusive evaluation window as FROM THROUGH, in product-scale seconds since J2000.
        #[arg(long, num_args = 2, value_names = ["FROM", "THROUGH"])]
        window: Option<Vec<f64>>,
    },
    /// Replay a RINEX OBS/NAV solve or watch a live RTCM stream.
    Tui {
        /// Replay input observation file. Required in replay mode.
        #[arg(long)]
        obs: Option<PathBuf>,
        /// Navigation file for replay mode.
        #[arg(long)]
        nav: Option<PathBuf>,
        /// Live mode navigation file when `--nav` is not used.
        #[arg(long)]
        ntrip_nav: Option<PathBuf>,
        /// NTRIP caster URL or host[:port].
        #[arg(long)]
        ntrip: Option<String>,
        /// NTRIP mountpoint.
        #[arg(long)]
        mount: Option<String>,
        /// Read-only source host[:port] for raw TCP RTCM.
        #[arg(long)]
        tcp: Option<String>,
        /// NTRIP username.
        #[arg(long)]
        user: Option<String>,
        /// NTRIP password (or `SIDEREON_NTRIP_PASSWORD`).
        #[arg(long)]
        pass: Option<String>,
        /// Replay speed multiplier.
        #[arg(long, default_value_t = 10.0)]
        speed: f64,
        /// Optional static GGA latitude for live mode.
        #[arg(long)]
        gga_lat: Option<f64>,
        /// Optional static GGA longitude for live mode.
        #[arg(long)]
        gga_lon: Option<f64>,
        /// Start paused after loading the first epoch.
        #[arg(long)]
        paused: bool,
    },
    /// Serve MCP tools over stdio for typed capabilities.
    ServeMcp {
        /// Tool profile: gnss, astro, or all.
        #[arg(long, default_value = "all")]
        profile: String,
    },
}

fn main() -> ExitCode {
    match Cli::try_parse() {
        Ok(cli) => match run(cli) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("error: {error:#}");
                ExitCode::from(1)
            }
        },
        Err(error) => {
            let code = match error.kind() {
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion => 0,
                _ => 2,
            };
            let _ = error.print();
            ExitCode::from(code)
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Solve {
            obs,
            nav,
            sp3,
            json,
        } => solve_command(&obs, &nav, sp3.as_deref(), json),
        Command::Qc { obs, json } => qc_command(&obs, json),
        Command::Metrics {
            enu_cov,
            json_file,
            probability,
            json,
        } => metrics_command(enu_cov.as_deref(), json_file.as_deref(), probability, json),
        Command::Inspect { file, window } => inspect_command(&file, window.as_deref()),
        Command::Tui {
            obs,
            nav,
            ntrip_nav,
            ntrip,
            mount,
            tcp,
            user,
            pass,
            speed,
            gga_lat,
            gga_lon,
            paused,
        } => {
            let use_live = ntrip.is_some() || tcp.is_some();
            let ntrip = validate_path_and_conflict("ntrip", ntrip)?;
            let tcp = validate_path_and_conflict("tcp", tcp)?;
            let has_ntrip = ntrip.is_some();
            let has_tcp = tcp.is_some();
            if use_live && has_ntrip == has_tcp {
                bail!("exactly one of --ntrip and --tcp is required");
            }
            let (nav, live_nav) = match (&nav, &ntrip_nav) {
                (Some(_), Some(_)) => {
                    bail!("exactly one of --nav and --ntrip-nav is allowed")
                }
                (Some(path), None) => (Some(path.clone()), None),
                (None, Some(path)) => (None, Some(path.clone())),
                (None, None) => (None, None),
            };

            if use_live {
                if ntrip.is_some() && mount.is_none() {
                    bail!("--mount is required for --ntrip");
                }
                let nav_path = match live_nav.as_deref().or(nav.as_deref()) {
                    Some(path) => path,
                    None => bail!("live mode requires --nav or --ntrip-nav"),
                };
                let mode = if has_ntrip {
                    let (host, port) = parse_host_port(&ntrip.unwrap(), "ntrip")?;
                    let user = user.unwrap_or_else(|| {
                        std::env::var("SIDEREON_NTRIP_USER")
                            .ok()
                            .filter(|value| !value.is_empty())
                            .unwrap_or_default()
                    });
                    let pass = pass.unwrap_or_else(|| {
                        std::env::var("SIDEREON_NTRIP_PASSWORD")
                            .ok()
                            .filter(|value| !value.is_empty())
                            .unwrap_or_default()
                    });
                    let mountpoint = match mount.clone() {
                        Some(mount) if !mount.is_empty() => mount,
                        _ => bail!("--mount is required for --ntrip"),
                    };
                    tui::LiveMode::Ntrip(tui::NtripConfigInput {
                        host,
                        port,
                        mount: mountpoint,
                        user,
                        pass,
                        gga_lat,
                        gga_lon,
                    })
                } else {
                    let (host, port) = parse_host_port(&tcp.unwrap(), "tcp")?;
                    tui::LiveMode::Tcp(tui::TcpConfigInput { host, port })
                };
                tui::run_tui(obs.as_deref(), nav_path, speed, paused, mode)
            } else {
                let obs_path = obs.as_ref().context("replay mode requires --obs")?;
                let nav_path = nav.clone().context("replay mode requires --nav")?;
                let mode = tui::LiveMode::Replay;
                tui::run_tui(Some(obs_path), &nav_path, speed, paused, mode)
            }
        }
        Command::ServeMcp { profile } => {
            mcp::serve_mcp_command(&profile).context("start serve-mcp stdio server")
        }
    }
}

fn validate_path_and_conflict(name: &str, value: Option<String>) -> Result<Option<String>> {
    if let Some(value) = &value {
        if value.is_empty() {
            bail!("--{name} cannot be empty");
        }
    }
    Ok(value)
}

fn parse_host_port(value: &str, label: &str) -> Result<(String, u16)> {
    let stripped = value
        .strip_prefix("https://")
        .or_else(|| value.strip_prefix("http://"))
        .map(|value| value.find('/').map_or(value, |slash| &value[..slash]))
        .unwrap_or(value);
    let (host, port) = if let Some((host, port_text)) = stripped.rsplit_once(':') {
        let port = port_text
            .parse::<u16>()
            .with_context(|| format!("{label}: invalid TCP port {port_text}"))?;
        (host.to_string(), port)
    } else {
        (stripped.to_string(), 2101)
    };
    if host.is_empty() {
        bail!("--{label} must include a host");
    }
    Ok((host, port))
}

fn solve_command(
    obs_path: &Path,
    nav_path: &Path,
    sp3_path: Option<&Path>,
    json: bool,
) -> Result<()> {
    let report = solve_rinex_report(obs_path, nav_path, sp3_path)?;
    if json {
        print_json(&report)?;
    } else {
        print_solve_human(&report);
    }
    if report.summary.failed_count > 0 {
        bail!("{} epoch solves failed", report.summary.failed_count);
    }
    Ok(())
}

pub(crate) fn solve_rinex_report(
    obs_path: &Path,
    nav_path: &Path,
    sp3_path: Option<&Path>,
) -> Result<SolveJson> {
    let obs =
        load_rinex_obs(obs_path).with_context(|| format!("load OBS {}", obs_path.display()))?;
    let nav =
        load_rinex_nav(nav_path).with_context(|| format!("load NAV {}", nav_path.display()))?;
    let options = RinexSppOptions::default_for(&obs).context("build default RINEX SPP options")?;

    let (source_label, assembled, solved) = if let Some(sp3_path) = sp3_path {
        let sp3_bytes =
            std::fs::read(sp3_path).with_context(|| format!("read SP3 {}", sp3_path.display()))?;
        let sp3 =
            load_sp3(&sp3_bytes).with_context(|| format!("parse SP3 {}", sp3_path.display()))?;
        let source = RinexSppSource::with_broadcast_context(&sp3, &nav);
        let assembled = spp_inputs_from_rinex_obs(&obs, &source, &options)
            .context("assemble RINEX SPP inputs")?;
        let solved = assembled
            .iter()
            .map(|epoch| sidereon::solve_spp(&source, &epoch.inputs, true, SolvePolicy::default()))
            .collect();
        ("sp3".to_string(), assembled, solved)
    } else {
        let assembled =
            spp_inputs_from_rinex_obs(&obs, &nav, &options).context("assemble RINEX SPP inputs")?;
        let solved = assembled
            .iter()
            .map(|epoch| sidereon::solve_spp(&nav, &epoch.inputs, true, SolvePolicy::default()))
            .collect();
        ("broadcast".to_string(), assembled, solved)
    };

    solve_report(
        source_label,
        obs_path,
        nav_path,
        sp3_path,
        &assembled,
        solved,
    )
}

fn solve_report(
    source: String,
    obs_path: &Path,
    nav_path: &Path,
    sp3_path: Option<&Path>,
    assembled: &[RinexSppEpochInputs],
    solved: Vec<sidereon::Result<ReceiverSolution>>,
) -> Result<SolveJson> {
    let mut epochs = Vec::with_capacity(assembled.len());
    let mut errors = Vec::new();
    let mut solved_count = 0usize;
    let mut nsats_total = 0usize;
    let mut metrics_count = 0usize;
    let mut cep_total = 0.0;
    let mut r95_total = 0.0;
    let mut vertical_total = 0.0;

    for (epoch_inputs, result) in assembled.iter().zip(solved) {
        match result {
            Ok(solution) => {
                solved_count += 1;
                nsats_total += solution.used_sats.len();
                let (metrics, metrics_error) = match solution_metrics(&solution) {
                    Ok(metrics) => {
                        metrics_count += 1;
                        cep_total += metrics.cep_m;
                        r95_total += metrics.r95_m;
                        vertical_total += metrics.vertical_95_m;
                        (Some(metrics), None)
                    }
                    Err(error) => (None, Some(error.to_string())),
                };
                epochs.push(SolveEpochJson {
                    epoch_index: epoch_inputs.epoch_index,
                    time: format_epoch(epoch_inputs.epoch),
                    solved: true,
                    error: None,
                    metrics_error,
                    lat_deg: solution.geodetic.map(|geo| rad_to_deg(geo.lat_rad)),
                    lon_deg: solution.geodetic.map(|geo| rad_to_deg(geo.lon_rad)),
                    height_m: solution.geodetic.map(|geo| geo.height_m),
                    ecef_m: Some(solution.position.as_array()),
                    nsats: solution.used_sats.len(),
                    satellites: solution.used_sats.iter().map(ToString::to_string).collect(),
                    systems: solution
                        .metadata
                        .systems
                        .iter()
                        .map(|system| system.to_string())
                        .collect(),
                    metrics,
                });
            }
            Err(error) => {
                let message = error.to_string();
                errors.push(SolveErrorJson {
                    epoch_index: epoch_inputs.epoch_index,
                    time: format_epoch(epoch_inputs.epoch),
                    message: message.clone(),
                });
                epochs.push(SolveEpochJson {
                    epoch_index: epoch_inputs.epoch_index,
                    time: format_epoch(epoch_inputs.epoch),
                    solved: false,
                    error: Some(message),
                    metrics_error: None,
                    lat_deg: None,
                    lon_deg: None,
                    height_m: None,
                    ecef_m: None,
                    nsats: 0,
                    satellites: Vec::new(),
                    systems: Vec::new(),
                    metrics: None,
                });
            }
        }
    }

    let summary = SolveSummaryJson {
        assembled_epochs: assembled.len(),
        solved_count,
        failed_count: errors.len(),
        mean_nsats: mean(nsats_total as f64, solved_count),
        mean_cep_m: mean(cep_total, metrics_count),
        mean_r95_m: mean(r95_total, metrics_count),
        mean_vertical_95_m: mean(vertical_total, metrics_count),
    };

    Ok(SolveJson {
        source,
        obs: obs_path.display().to_string(),
        nav: nav_path.display().to_string(),
        sp3: sp3_path.map(|path| path.display().to_string()),
        epochs,
        summary,
        errors,
    })
}

fn solution_metrics(solution: &ReceiverSolution) -> Result<SolveMetricsJson> {
    let metrics = metrics_from_position_covariance(&solution.position_covariance)
        .map_err(|err| anyhow!("compute position error metrics: {err:?}"))?;
    let vertical_95_m = vertical_radius_at(solution.position_covariance.enu_m2[2][2], 0.95)
        .map_err(|err| anyhow!("compute vertical 95 bound: {err:?}"))?;
    Ok(SolveMetricsJson {
        cep_m: metrics.cep_m.radius_m,
        r95_m: metrics.r95_m.radius_m,
        r99_m: metrics.r99_m.radius_m,
        vertical_50_m: metrics.vep_m,
        vertical_95_m,
        sigma_e_m: metrics.sigma_e_m,
        sigma_n_m: metrics.sigma_n_m,
        sigma_u_m: metrics.sigma_u_m,
    })
}

fn print_solve_human(report: &SolveJson) {
    println!("source: {}", report.source);
    println!("obs: {}", report.obs);
    println!("nav: {}", report.nav);
    if let Some(sp3) = &report.sp3 {
        println!("sp3: {sp3}");
    }
    println!(
        "{:<20} {:>11} {:>12} {:>9} {:>5} {:>9} {:>9} {:>9}",
        "time", "lat_deg", "lon_deg", "height_m", "nsat", "CEP_m", "R95_m", "V95_m"
    );
    for epoch in &report.epochs {
        if let Some(metrics) = &epoch.metrics {
            println!(
                "{:<20} {:>11.6} {:>12.6} {:>9.2} {:>5} {:>9.3} {:>9.3} {:>9.3}",
                epoch.time,
                epoch.lat_deg.unwrap_or(f64::NAN),
                epoch.lon_deg.unwrap_or(f64::NAN),
                epoch.height_m.unwrap_or(f64::NAN),
                epoch.nsats,
                metrics.cep_m,
                metrics.r95_m,
                metrics.vertical_95_m
            );
        } else if epoch.solved {
            println!(
                "{:<20} {:>11.6} {:>12.6} {:>9.2} {:>5} {:>9} {:>9} {:>9}",
                epoch.time,
                epoch.lat_deg.unwrap_or(f64::NAN),
                epoch.lon_deg.unwrap_or(f64::NAN),
                epoch.height_m.unwrap_or(f64::NAN),
                epoch.nsats,
                "ERR",
                "ERR",
                "ERR"
            );
        } else {
            println!(
                "{:<20} {:>11} {:>12} {:>9} {:>5} {:>9} {:>9} {:>9}",
                epoch.time, "ERR", "ERR", "ERR", 0, "ERR", "ERR", "ERR"
            );
        }
    }
    println!();
    println!("summary:");
    println!("  assembled epochs: {}", report.summary.assembled_epochs);
    println!("  solved epochs: {}", report.summary.solved_count);
    println!("  failed epochs: {}", report.summary.failed_count);
    if let Some(mean_nsats) = report.summary.mean_nsats {
        println!("  mean nsats: {mean_nsats:.1}");
    }
    if let Some(mean_cep) = report.summary.mean_cep_m {
        println!("  mean CEP: {mean_cep:.3} m");
    }
    if let Some(mean_r95) = report.summary.mean_r95_m {
        println!("  mean R95: {mean_r95:.3} m");
    }
    if let Some(mean_vertical) = report.summary.mean_vertical_95_m {
        println!("  mean vertical 95: {mean_vertical:.3} m");
    }
    if !report.errors.is_empty() {
        println!("errors:");
        for error in &report.errors {
            println!(
                "  epoch {} {}: {}",
                error.epoch_index, error.time, error.message
            );
        }
    }
    let metrics_errors: Vec<_> = report
        .epochs
        .iter()
        .filter_map(|epoch| {
            epoch
                .metrics_error
                .as_ref()
                .map(|error| (epoch.epoch_index, epoch.time.as_str(), error))
        })
        .collect();
    if !metrics_errors.is_empty() {
        println!("metric errors:");
        for (epoch_index, time, error) in metrics_errors {
            println!("  epoch {epoch_index} {time}: {error}");
        }
    }
}

fn qc_command(obs_path: &Path, json: bool) -> Result<()> {
    let report = qc_log_report(obs_path)?;
    if json {
        print_json(&report)?;
    } else {
        print_qc_human(&report);
    }

    if let Some(error) = &report.parse_error {
        bail!("RINEX OBS parse failed: {error}");
    }
    Ok(())
}

pub(crate) fn qc_log_report(obs_path: &Path) -> Result<QcJson> {
    let text = std::fs::read_to_string(obs_path)
        .with_context(|| format!("read OBS {}", obs_path.display()))?;
    let lint = sidereon::lint_rinex_obs(&text);
    let parsed = parse_rinex_obs(&text);
    let (qc, parse_error) = match parsed {
        Ok(obs) => (Some(observation_qc(&obs)), None),
        Err(error) => (None, Some(error.to_string())),
    };

    Ok(QcJson {
        obs: obs_path.display().to_string(),
        lint: lint_json(&lint),
        qc,
        parse_error,
    })
}

fn print_qc_human(report: &QcJson) {
    println!("obs: {}", report.obs);
    println!(
        "lint: fatal={} error={} warning={} info={} decoded_crinex={}",
        report.lint.counts.fatal,
        report.lint.counts.error,
        report.lint.counts.warning,
        report.lint.counts.info,
        report.lint.decoded_from_crinex
    );
    if report.lint.findings.is_empty() {
        println!("findings: none");
    } else {
        println!(
            "{:<8} {:<8} {:<12} {:<10} ref",
            "severity", "code", "satellite", "epoch"
        );
        for finding in &report.lint.findings {
            println!(
                "{:<8} {:<8} {:<12} {:<10} {}",
                finding.severity,
                finding.code,
                finding.at.satellite.as_deref().unwrap_or("-"),
                finding
                    .at
                    .epoch_index
                    .map(|idx| idx.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                finding.spec_ref
            );
        }
    }
    if let Some(qc) = &report.qc {
        println!();
        print!("{}", render_obs_qc_text(qc));
    }
    if let Some(error) = &report.parse_error {
        println!("parse error: {error}");
    }
}

fn metrics_command(
    enu_cov: Option<&str>,
    json_file: Option<&Path>,
    probability: f64,
    json: bool,
) -> Result<()> {
    let covariance = if let Some(text) = enu_cov {
        parse_covariance_arg(text)?
    } else if let Some(path) = json_file {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read JSON covariance {}", path.display()))?;
        parse_covariance_json(&text)?
    } else {
        bail!("missing covariance input");
    };
    let metrics = metrics_from_enu_covariance_m2(covariance)
        .map_err(|err| anyhow!("compute covariance metrics: {err:?}"))?;
    let horizontal = horizontal_radius_at(covariance, probability)
        .map_err(|err| anyhow!("compute horizontal radius: {err:?}"))?;
    let spherical = spherical_radius_at(covariance, probability)
        .map_err(|err| anyhow!("compute spherical radius: {err:?}"))?;
    let vertical = vertical_radius_at(covariance[2][2], probability)
        .map_err(|err| anyhow!("compute vertical radius: {err:?}"))?;

    let report = MetricsJson::from_parts(
        covariance,
        probability,
        &metrics,
        horizontal,
        vertical,
        spherical,
    );
    if json {
        print_json(&report)?;
    } else {
        print_metrics_human(
            covariance,
            probability,
            &metrics,
            horizontal,
            vertical,
            spherical,
        );
    }
    Ok(())
}

fn print_metrics_human(
    covariance: [[f64; 3]; 3],
    probability: f64,
    metrics: &PositionErrorMetrics,
    horizontal: PercentileRadius,
    vertical: f64,
    spherical: PercentileRadius,
) {
    println!("covariance ENU m^2:");
    for row in covariance {
        println!("  {:>12.6} {:>12.6} {:>12.6}", row[0], row[1], row[2]);
    }
    println!();
    println!("{:<28} {:>14}", "metric", "meters");
    println!("{:<28} {:>14.6}", "sigma east", metrics.sigma_e_m);
    println!("{:<28} {:>14.6}", "sigma north", metrics.sigma_n_m);
    println!("{:<28} {:>14.6}", "sigma up", metrics.sigma_u_m);
    println!(
        "{:<28} {:>14.6}",
        "ellipse semi-major", metrics.ellipse.semi_major_m
    );
    println!(
        "{:<28} {:>14.6}",
        "ellipse semi-minor", metrics.ellipse.semi_minor_m
    );
    println!(
        "{:<28} {:>14.6}",
        "ellipse orientation deg",
        rad_to_deg(metrics.ellipse.orientation_rad)
    );
    println!("{:<28} {:>14.6}", "CEP", metrics.cep_m.radius_m);
    println!("{:<28} {:>14.6}", "R95", metrics.r95_m.radius_m);
    println!("{:<28} {:>14.6}", "R99", metrics.r99_m.radius_m);
    println!("{:<28} {:>14.6}", "DRMS", metrics.drms_m);
    println!("{:<28} {:>14.6}", "2DRMS", metrics.two_drms_m);
    println!("{:<28} {:>14.6}", "VEP", metrics.vep_m);
    println!("{:<28} {:>14.6}", "SEP", metrics.sep_m.radius_m);
    println!("{:<28} {:>14.6}", "MRSE", metrics.mrse_m);
    println!(
        "{:<28} {:>14.6}",
        format!("H({probability:.3})"),
        horizontal.radius_m
    );
    println!("{:<28} {:>14.6}", format!("V({probability:.3})"), vertical);
    println!(
        "{:<28} {:>14.6}",
        format!("S({probability:.3})"),
        spherical.radius_m
    );
}

fn inspect_command(path: &Path, window: Option<&[f64]>) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let text = std::str::from_utf8(&bytes).ok();

    if let Some(endpoints) = window {
        let [from_j2000_s, through_j2000_s] = endpoints else {
            bail!("--window requires exactly FROM THROUGH");
        };
        let sp3 = load_sp3(&bytes).context("--window is available only for SP3 products")?;
        print_inspect(InspectReport::sp3(path, &sp3));
        print_window_continuity(&sp3, *from_j2000_s, *through_j2000_s)?;
        return Ok(());
    }

    if let Some(text) = text {
        if let Ok(obs) = parse_rinex_obs(text) {
            print_inspect(InspectReport::obs(path, &obs));
            return Ok(());
        }
        if let Ok(nav) = parse_rinex_nav(text) {
            print_inspect(InspectReport::nav(path, &nav));
            return Ok(());
        }
    }
    if let Ok(sp3) = load_sp3(&bytes) {
        print_inspect(InspectReport::sp3(path, &sp3));
        return Ok(());
    }
    if let Some(text) = text {
        if let Some(report) = InspectReport::tle(path, text) {
            print_inspect(report);
            return Ok(());
        }
        if let Ok(antex) = parse_antex(text) {
            if !antex.antennas.is_empty() {
                print_inspect(InspectReport::antex(path, &antex));
                return Ok(());
            }
        }
    }

    bail!("unrecognized file type: {}", path.display())
}

fn print_window_continuity(sp3: &Sp3, from_j2000_s: f64, through_j2000_s: f64) -> Result<()> {
    let report = check_continuity(
        &sp3.precise_ephemeris_samples(),
        &ContinuityOptions::for_orbit_class(OrbitClass::MeoGnss),
    );
    let window = EpochWindow::new(from_j2000_s, through_j2000_s)?;
    let stencil = StencilExtent::for_sp3(sp3)?;
    let verdict = report.verdict_for_window(window, stencil);
    println!(
        "continuity: {} (defects={}, pairs_checked={}, residuals_checked={}, residuals_skipped={})",
        if report.attested() {
            "attested"
        } else {
            "defects"
        },
        report.defects.len(),
        report.pairs_checked,
        report.residuals_checked,
        report.residuals_skipped,
    );
    println!(
        "window_continuity: {} (influencing_defects={}, stencil_before_s={}, stencil_after_s={})",
        if verdict.accepted() {
            "accept"
        } else {
            "refuse"
        },
        verdict.influencing_defects.len(),
        stencil.before_s(),
        stencil.after_s(),
    );
    Ok(())
}

fn print_inspect(report: InspectReport) {
    println!("path: {}", report.path);
    println!("type: {}", report.file_type);
    if let Some(span) = report.span {
        println!("span: {span}");
    }
    for (key, value) in report.counts {
        println!("{key}: {value}");
    }
    if !report.systems.is_empty() {
        println!("systems: {}", report.systems.join(","));
    }
    if !report.satellites.is_empty() {
        println!("satellites: {}", compact_list(&report.satellites, 32));
    }
}

#[derive(Debug)]
struct InspectReport {
    path: String,
    file_type: &'static str,
    span: Option<String>,
    counts: Vec<(&'static str, String)>,
    systems: Vec<String>,
    satellites: Vec<String>,
}

impl InspectReport {
    fn obs(path: &Path, obs: &ObservationFile) -> Self {
        let mut satellites = BTreeSet::new();
        let mut systems = BTreeSet::new();
        for epoch in obs.epochs() {
            for sat in epoch.sats.keys() {
                satellites.insert(sat.to_string());
                systems.insert(sat.system.to_string());
            }
        }
        let observation_epochs = obs.epochs().iter().filter(|epoch| epoch.flag <= 1).count();
        let event_records = obs.epochs().iter().filter(|epoch| epoch.flag > 1).count();
        Self {
            path: path.display().to_string(),
            file_type: "RINEX OBS",
            span: obs_span(obs),
            counts: vec![
                ("version", format!("{:.2}", obs.header().version)),
                ("epochs", obs.epochs().len().to_string()),
                ("observation_epochs", observation_epochs.to_string()),
                ("event_records", event_records.to_string()),
                ("skipped_records", obs.skipped_records.to_string()),
                ("satellite_count", satellites.len().to_string()),
            ],
            systems: systems.into_iter().collect(),
            satellites: satellites.into_iter().collect(),
        }
    }

    fn nav(path: &Path, nav: &BroadcastEphemeris) -> Self {
        let mut satellites = BTreeSet::new();
        let mut systems = BTreeSet::new();
        for record in nav.records() {
            satellites.insert(record.satellite_id.to_string());
            systems.insert(record.satellite_id.system.to_string());
        }
        for record in nav.glonass_records() {
            satellites.insert(record.satellite_id.to_string());
            systems.insert(record.satellite_id.system.to_string());
        }
        Self {
            path: path.display().to_string(),
            file_type: "RINEX NAV",
            span: nav_span(nav),
            counts: vec![
                ("records", nav.records().len().to_string()),
                ("glonass_records", nav.glonass_records().len().to_string()),
                ("satellite_count", satellites.len().to_string()),
            ],
            systems: systems.into_iter().collect(),
            satellites: satellites.into_iter().collect(),
        }
    }

    fn sp3(path: &Path, sp3: &Sp3) -> Self {
        let sats: Vec<_> = sp3.satellites().iter().map(ToString::to_string).collect();
        let systems = sp3
            .satellites()
            .iter()
            .map(|sat| sat.system.to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let epochs = sp3.epochs_j2000_seconds();
        Self {
            path: path.display().to_string(),
            file_type: "SP3",
            span: span_from_numbers(epochs.first().copied(), epochs.last().copied(), "j2000_s"),
            counts: vec![
                ("epochs", sp3.epoch_count().to_string()),
                ("satellite_count", sp3.satellites().len().to_string()),
                ("skipped_records", sp3.skipped_records.to_string()),
                ("coordinate_system", sp3.header.coordinate_system.clone()),
                ("agency", sp3.header.agency.clone()),
                ("time_system", format!("{:?}", sp3.header.time_system)),
            ],
            systems,
            satellites: sats,
        }
    }

    fn antex(path: &Path, antex: &sidereon::antex::Antex) -> Self {
        let mut receiver_count = 0usize;
        let mut satellite_count = 0usize;
        let mut ids = Vec::new();
        for antenna in antex.antennas.values() {
            ids.push(antenna.id.clone());
            match antenna.kind {
                AntennaKind::Receiver => receiver_count += 1,
                AntennaKind::Satellite => satellite_count += 1,
            }
        }
        Self {
            path: path.display().to_string(),
            file_type: "ANTEX",
            span: None,
            counts: vec![
                ("antennas", antex.antennas.len().to_string()),
                ("receiver_antennas", receiver_count.to_string()),
                ("satellite_antennas", satellite_count.to_string()),
                ("skipped_records", antex.skipped_records().to_string()),
            ],
            systems: Vec::new(),
            satellites: ids,
        }
    }

    fn tle(path: &Path, text: &str) -> Option<Self> {
        let lines: Vec<_> = text
            .lines()
            .map(str::trim_end)
            .filter(|line| !line.trim().is_empty())
            .collect();
        let mut catalogs = Vec::new();
        let mut checksum_warnings = 0usize;
        let mut index = 0usize;
        while index + 1 < lines.len() {
            let (line1, line2) =
                if lines[index].starts_with('1') && lines[index + 1].starts_with('2') {
                    (lines[index], lines[index + 1])
                } else if index + 2 < lines.len()
                    && lines[index + 1].starts_with('1')
                    && lines[index + 2].starts_with('2')
                {
                    index += 1;
                    (lines[index], lines[index + 1])
                } else {
                    return None;
                };
            let parsed = sidereon::tle::parse(line1, line2).ok()?;
            checksum_warnings += parsed.checksum_warnings.len();
            catalogs.push(parsed.elements.catalog_number);
            index += 2;
        }
        if catalogs.is_empty() {
            return None;
        }
        Some(Self {
            path: path.display().to_string(),
            file_type: "TLE",
            span: None,
            counts: vec![
                ("tle_pairs", catalogs.len().to_string()),
                ("checksum_warnings", checksum_warnings.to_string()),
            ],
            systems: Vec::new(),
            satellites: catalogs,
        })
    }
}

#[derive(Serialize)]
struct SolveJson {
    source: String,
    obs: String,
    nav: String,
    sp3: Option<String>,
    epochs: Vec<SolveEpochJson>,
    summary: SolveSummaryJson,
    errors: Vec<SolveErrorJson>,
}

#[derive(Serialize)]
struct SolveEpochJson {
    epoch_index: usize,
    time: String,
    solved: bool,
    error: Option<String>,
    metrics_error: Option<String>,
    lat_deg: Option<f64>,
    lon_deg: Option<f64>,
    height_m: Option<f64>,
    ecef_m: Option<[f64; 3]>,
    nsats: usize,
    satellites: Vec<String>,
    systems: Vec<String>,
    metrics: Option<SolveMetricsJson>,
}

#[derive(Serialize)]
struct SolveMetricsJson {
    cep_m: f64,
    r95_m: f64,
    r99_m: f64,
    vertical_50_m: f64,
    vertical_95_m: f64,
    sigma_e_m: f64,
    sigma_n_m: f64,
    sigma_u_m: f64,
}

#[derive(Serialize)]
struct SolveSummaryJson {
    assembled_epochs: usize,
    solved_count: usize,
    failed_count: usize,
    mean_nsats: Option<f64>,
    mean_cep_m: Option<f64>,
    mean_r95_m: Option<f64>,
    mean_vertical_95_m: Option<f64>,
}

#[derive(Serialize)]
struct SolveErrorJson {
    epoch_index: usize,
    time: String,
    message: String,
}

#[derive(Serialize)]
struct MetricsJson {
    enu_covariance_m2: [[f64; 3]; 3],
    probability: f64,
    sigma_e_m: f64,
    sigma_n_m: f64,
    sigma_u_m: f64,
    ellipse_semi_major_m: f64,
    ellipse_semi_minor_m: f64,
    ellipse_orientation_deg: f64,
    cep_m: f64,
    r95_m: f64,
    r99_m: f64,
    drms_m: f64,
    two_drms_m: f64,
    vep_m: f64,
    sep_m: f64,
    mrse_m: f64,
    horizontal_radius_m: f64,
    vertical_radius_m: f64,
    spherical_radius_m: f64,
}

impl MetricsJson {
    fn from_parts(
        enu_covariance_m2: [[f64; 3]; 3],
        probability: f64,
        metrics: &PositionErrorMetrics,
        horizontal: PercentileRadius,
        vertical: f64,
        spherical: PercentileRadius,
    ) -> Self {
        Self {
            enu_covariance_m2,
            probability,
            sigma_e_m: metrics.sigma_e_m,
            sigma_n_m: metrics.sigma_n_m,
            sigma_u_m: metrics.sigma_u_m,
            ellipse_semi_major_m: metrics.ellipse.semi_major_m,
            ellipse_semi_minor_m: metrics.ellipse.semi_minor_m,
            ellipse_orientation_deg: rad_to_deg(metrics.ellipse.orientation_rad),
            cep_m: metrics.cep_m.radius_m,
            r95_m: metrics.r95_m.radius_m,
            r99_m: metrics.r99_m.radius_m,
            drms_m: metrics.drms_m,
            two_drms_m: metrics.two_drms_m,
            vep_m: metrics.vep_m,
            sep_m: metrics.sep_m.radius_m,
            mrse_m: metrics.mrse_m,
            horizontal_radius_m: horizontal.radius_m,
            vertical_radius_m: vertical,
            spherical_radius_m: spherical.radius_m,
        }
    }
}

#[derive(Serialize)]
struct QcJson {
    obs: String,
    lint: LintJson,
    qc: Option<sidereon::qc_obs::ObservationQcReport>,
    parse_error: Option<String>,
}

#[derive(Serialize)]
struct LintJson {
    decoded_from_crinex: bool,
    clean: bool,
    counts: SeverityCountsJson,
    findings: Vec<FindingJson>,
}

#[derive(Serialize)]
struct SeverityCountsJson {
    fatal: usize,
    error: usize,
    warning: usize,
    info: usize,
}

#[derive(Serialize)]
struct FindingJson {
    code: String,
    severity: String,
    spec_ref: String,
    at: FindingRefJson,
}

#[derive(Serialize)]
struct FindingRefJson {
    epoch_index: Option<usize>,
    satellite: Option<String>,
    field: Option<String>,
}

fn lint_json(report: &LintReport) -> LintJson {
    LintJson {
        decoded_from_crinex: report.decoded_from_crinex,
        clean: report.is_clean(),
        counts: SeverityCountsJson {
            fatal: report.count(Severity::Fatal),
            error: report.count(Severity::Error),
            warning: report.count(Severity::Warning),
            info: report.count(Severity::Info),
        },
        findings: report
            .findings
            .iter()
            .map(|finding| FindingJson {
                code: finding.code().to_string(),
                severity: severity_label(finding.severity()).to_string(),
                spec_ref: finding.spec_ref().to_string(),
                at: finding_ref_json(finding.at()),
            })
            .collect(),
    }
}

fn finding_ref_json(at: &FindingRef) -> FindingRefJson {
    FindingRefJson {
        epoch_index: at.epoch_index,
        satellite: at.satellite.clone(),
        field: at.field.map(str::to_string),
    }
}

fn severity_label(severity: Severity) -> &'static str {
    match severity {
        Severity::Fatal => "fatal",
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Info => "info",
    }
}

fn parse_covariance_arg(text: &str) -> Result<[[f64; 3]; 3]> {
    let values = text
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<f64>()
                .with_context(|| format!("parse covariance value {part:?}"))
        })
        .collect::<Result<Vec<_>>>()?;
    covariance_from_flat(&values)
}

fn parse_covariance_json(text: &str) -> Result<[[f64; 3]; 3]> {
    let value: Value = serde_json::from_str(text).context("parse covariance JSON")?;
    let covariance_value = if let Some(value) = value.get("enu_covariance_m2") {
        value
    } else if let Some(value) = value.get("covariance_enu_m2") {
        value
    } else {
        &value
    };
    covariance_from_value(covariance_value)
}

fn covariance_from_value(value: &Value) -> Result<[[f64; 3]; 3]> {
    if let Some(rows) = value.as_array() {
        if rows.len() == 9 && rows.iter().all(Value::is_number) {
            let flat = rows
                .iter()
                .map(|value| value.as_f64().context("covariance entries must be numbers"))
                .collect::<Result<Vec<_>>>()?;
            return covariance_from_flat(&flat);
        }
        if rows.len() == 3 && rows.iter().all(Value::is_array) {
            let mut covariance = [[0.0; 3]; 3];
            for (row_index, row) in rows.iter().enumerate() {
                let row = row.as_array().context("covariance row must be an array")?;
                if row.len() != 3 {
                    bail!("covariance rows must have exactly three entries");
                }
                for (col_index, value) in row.iter().enumerate() {
                    covariance[row_index][col_index] = value
                        .as_f64()
                        .context("covariance entries must be numbers")?;
                }
            }
            return Ok(covariance);
        }
    }
    bail!("covariance JSON must be a flat 9-array, 3x3 array, or object with enu_covariance_m2")
}

fn covariance_from_flat(values: &[f64]) -> Result<[[f64; 3]; 3]> {
    if values.len() != 9 {
        bail!("ENU covariance requires exactly nine numbers");
    }
    Ok([
        [values[0], values[1], values[2]],
        [values[3], values[4], values[5]],
        [values[6], values[7], values[8]],
    ])
}

fn obs_span(obs: &ObservationFile) -> Option<String> {
    let first = obs
        .header()
        .time_of_first_obs
        .map(|(epoch, _)| format_epoch(epoch))
        .or_else(|| obs.epochs().first().map(|epoch| format_epoch(epoch.epoch)))?;
    let last = obs
        .header()
        .time_of_last_obs
        .map(|(epoch, _)| format_epoch(epoch))
        .or_else(|| obs.epochs().last().map(|epoch| format_epoch(epoch.epoch)))?;
    Some(format!("{first} to {last}"))
}

fn nav_span(nav: &BroadcastEphemeris) -> Option<String> {
    let mut native_first: Option<f64> = None;
    let mut native_last: Option<f64> = None;
    for record in nav.records() {
        update_min_max(
            &mut native_first,
            &mut native_last,
            f64::from(record.toe.week) * 604_800.0 + record.toe.tow_s,
        );
    }
    let mut spans = Vec::new();
    if let Some(span) = span_from_numbers(native_first, native_last, "native_week_s") {
        spans.push(span);
    }
    let mut glonass_first: Option<f64> = None;
    let mut glonass_last: Option<f64> = None;
    for record in nav.glonass_records() {
        update_min_max(
            &mut glonass_first,
            &mut glonass_last,
            record.toe_utc_j2000_s,
        );
    }
    if let Some(span) = span_from_numbers(glonass_first, glonass_last, "glonass_j2000_s") {
        spans.push(span);
    }
    if spans.is_empty() {
        None
    } else {
        Some(spans.join("; "))
    }
}

fn span_from_numbers(first: Option<f64>, last: Option<f64>, unit: &str) -> Option<String> {
    Some(format!("{:.3} to {:.3} {unit}", first?, last?))
}

fn update_min_max(first: &mut Option<f64>, last: &mut Option<f64>, value: f64) {
    *first = Some(first.map_or(value, |current| current.min(value)));
    *last = Some(last.map_or(value, |current| current.max(value)));
}

fn compact_list(items: &[String], limit: usize) -> String {
    if items.len() <= limit {
        return items.join(",");
    }
    let mut out = items[..limit].join(",");
    let _ = write!(&mut out, ",...(+{})", items.len() - limit);
    out
}

fn format_epoch(epoch: sidereon::rinex::observations::ObsEpochTime) -> String {
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:06.3}",
        epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second
    )
}

fn rad_to_deg(rad: f64) -> f64 {
    rad * 180.0 / PI
}

fn mean(total: f64, count: usize) -> Option<f64> {
    (count > 0).then_some(total / count as f64)
}

fn print_json<T: Serialize>(value: &T) -> Result<()> {
    serde_json::to_writer_pretty(std::io::stdout(), value).context("write JSON")?;
    println!();
    Ok(())
}
