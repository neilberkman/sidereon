//! Validation scoreboard harness.
//!
//! The library keeps the scoring pipeline testable without network access. The
//! binary supplies the HTTPS fetcher and file output paths.

#![deny(missing_docs)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde::Serialize;
use sidereon_core::astro::frames::transforms::FrameTransformError;
use sidereon_core::astro::propagator::ForceModelKind;
use sidereon_core::astro::time::civil::civil_from_j2000_seconds;
use sidereon_core::constants::{J2000_JD, SECONDS_PER_DAY};
use sidereon_core::data::{DataCatalogError, ProductDate};
use sidereon_core::ephemeris::{
    fit_precise_ephemeris_sample_orbit, fit_precise_ephemeris_state_sample_orbit, OrbitFitOptions,
    OrbitResidualStats, OrientedPreciseEphemerisStateSample, PreciseEphemerisSample,
    PreciseEphemerisStateSample, Sp3,
};
use sidereon_core::{
    EarthOrientation, EarthOrientationProvider, Error as CoreError, GnssSatelliteId,
    TdbEarthOrientationProvider,
};

pub mod ppc;

const UNIX_TO_J2000_S: i64 = 946_728_000;
const DEFAULT_LOOKBACK_DAYS: u32 = 4;

/// Scoreboard result emitted as JSON.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoreboardReport {
    /// Product date in UTC, formatted as `YYYY-MM-DD`.
    pub date_utc: String,
    /// Crate version used to build the scorer.
    pub sidereon_version: String,
    /// Whether the run scored an SP3 product or recorded a no-data result.
    pub status: ScoreboardStatus,
    /// Product identity and SP3 producing agency.
    pub product: Option<ProductReport>,
    /// Product URLs attempted before scoring or declaring no data.
    pub attempted_candidates: Vec<AttemptedCandidateReport>,
    /// Per-constellation aggregate residual summaries.
    pub per_constellation: BTreeMap<String, ConstellationReport>,
    /// Per-satellite best, worst, and skipped rows.
    pub per_sat: PerSatelliteReport,
    /// Caveats and method notes that affect interpretation.
    pub notes: Vec<String>,
}

/// Scoreboard run status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreboardStatus {
    /// An SP3 product was fetched and scored.
    Scored,
    /// No product was posted in the candidate window.
    NoData,
}

/// Product identity in a scoreboard report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProductReport {
    /// Canonical product filename.
    pub name: String,
    /// SP3 header agency string.
    pub agency: String,
    /// SP3 parser skips for unsupported declaration or record entries.
    pub parser_skipped_records: usize,
}

/// Product candidate attempted by the resolver.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AttemptedCandidateReport {
    /// Product date in UTC, formatted as `YYYY-MM-DD`.
    pub date_utc: String,
    /// Candidate family, for example `rapid` or `ultra_rapid`.
    pub cadence: String,
    /// Archive or analysis-center source.
    pub source: String,
    /// Canonical product filename.
    pub name: String,
    /// HTTPS archive URL.
    pub url: String,
}

/// Per-constellation scoreboard aggregate.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConstellationReport {
    /// Satellites declared for this constellation in the SP3 header.
    pub sat_count: usize,
    /// Satellites that produced a fit and residual ledger.
    pub fit_count: usize,
    /// Satellites that were skipped or failed fitting.
    pub skipped: usize,
    /// Median three-dimensional RMS residual, meters, over fitted satellites.
    pub median_rms_3d_m: Option<f64>,
    /// Largest three-dimensional RMS residual, meters, over fitted satellites.
    pub worst_rms_3d_m: Option<f64>,
}

/// Per-satellite scoreboard section.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerSatelliteReport {
    /// Three fitted satellites with the lowest RMS residuals.
    pub top: Vec<SatelliteFitReport>,
    /// Three fitted satellites with the highest RMS residuals.
    pub bottom: Vec<SatelliteFitReport>,
    /// Satellites that were not fitted, with reasons.
    pub skipped: Vec<SatelliteSkipReport>,
}

/// Residual row for one fitted satellite.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SatelliteFitReport {
    /// SP3 satellite token.
    pub satellite: String,
    /// Constellation display name.
    pub constellation: String,
    /// Three-dimensional RMS residual, meters.
    pub rms_3d_m: f64,
    /// Radial RMS residual, meters.
    pub radial_rms_m: f64,
    /// Along-track RMS residual, meters.
    pub along_rms_m: f64,
    /// Cross-track RMS residual, meters.
    pub cross_rms_m: f64,
    /// Number of residual epochs.
    pub n: usize,
    /// Whether the ledger marks this row as short.
    pub low_sample_count: bool,
}

/// Skip row for one satellite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SatelliteSkipReport {
    /// SP3 satellite token.
    pub satellite: String,
    /// Constellation display name.
    pub constellation: String,
    /// Machine-readable skip reason.
    pub reason: String,
}

/// Candidate product URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductCandidate {
    /// Product date.
    pub date: ProductDate,
    /// Candidate family.
    pub cadence: ProductCadence,
    /// Archive or analysis-center source.
    pub source: &'static str,
    /// Canonical product filename.
    pub name: String,
    /// HTTPS archive URL.
    pub url: String,
}

/// Product candidate family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductCadence {
    /// Rapid daily product.
    Rapid,
    /// Ultra-rapid issued product.
    UltraRapid,
}

impl ProductCadence {
    fn as_str(self) -> &'static str {
        match self {
            Self::Rapid => "rapid",
            Self::UltraRapid => "ultra_rapid",
        }
    }
}

/// Product bytes resolved from the latest available candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedProduct {
    /// Candidate metadata.
    pub candidate: ProductCandidate,
    /// Decompressed SP3 bytes.
    pub bytes: Vec<u8>,
}

/// Product resolution result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductResolution {
    /// Product bytes when a candidate was posted.
    pub resolved: Option<ResolvedProduct>,
    /// Candidates attempted in request order.
    pub attempted: Vec<ProductCandidate>,
}

/// Fetch result for one product candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchOutcome {
    /// Candidate was present and returned decompressed SP3 bytes.
    Available(Vec<u8>),
    /// Candidate was not posted at the archive.
    NotPosted,
}

enum SatelliteScoreRow {
    Fit(SatelliteFitReport),
    Skip(SatelliteSkipReport),
}

struct SatelliteScoreOutcome {
    row: SatelliteScoreRow,
    used_position_fallback: bool,
}

/// Minimal fetch interface used by the resolver.
pub trait ProductFetcher {
    /// Fetch one product candidate.
    fn fetch(&self, candidate: &ProductCandidate) -> Result<FetchOutcome, ScoreboardError>;
}

/// HTTPS product fetcher used by the binary.
#[derive(Debug, Clone, Copy, Default)]
pub struct HttpsFetcher;

impl ProductFetcher for HttpsFetcher {
    fn fetch(&self, candidate: &ProductCandidate) -> Result<FetchOutcome, ScoreboardError> {
        fetch_https_product(candidate)
    }
}

/// Scoring options for one SP3 product.
#[derive(Debug, Clone)]
pub struct ScoreOptions {
    /// Orbit fit options passed to the core orbit-determination fitter.
    pub fit_options: OrbitFitOptions,
    /// Whether velocity-bearing SP3 state samples should use the state path.
    pub prefer_state_samples: bool,
}

impl Default for ScoreOptions {
    fn default() -> Self {
        Self {
            fit_options: OrbitFitOptions::default(),
            prefer_state_samples: true,
        }
    }
}

/// Error returned by the scoreboard harness.
#[derive(Debug, thiserror::Error)]
pub enum ScoreboardError {
    /// Product catalog resolution failed.
    #[error("data catalog error: {0}")]
    DataCatalog(#[from] DataCatalogError),
    /// The resolved archive URL was not HTTPS.
    #[error("non-HTTPS product URL: {url}")]
    NonHttpsUrl {
        /// URL rejected by the fetcher.
        url: String,
    },
    /// An HTTP status other than a not-posted status was returned.
    #[error("HTTP status {status} while fetching {url}")]
    HttpStatus {
        /// URL requested.
        url: String,
        /// HTTP status code.
        status: u16,
    },
    /// Network transport failed.
    #[error("network error while fetching {url}: {message}")]
    Network {
        /// URL requested.
        url: String,
        /// Transport error message.
        message: String,
    },
    /// File or stream I/O failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// SP3 parsing failed.
    #[error("SP3 parse error: {0}")]
    Sp3(#[from] CoreError),
    /// Earth-orientation evaluation failed.
    #[error("frame transform error: {0}")]
    Frame(#[from] FrameTransformError),
    /// JSON serialization failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// CLI arguments were invalid.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// System time was before the Unix epoch.
    #[error("system time is before the Unix epoch")]
    SystemTimeBeforeUnixEpoch,
}

/// Return the default lookback window, in whole UTC days.
#[must_use]
pub const fn default_lookback_days() -> u32 {
    DEFAULT_LOOKBACK_DAYS
}

/// Resolve and fetch the latest available rapid multi-GNSS SP3 product.
pub fn resolve_latest_available_rapid_sp3(
    target_date: ProductDate,
    lookback_days: u32,
    fetcher: &impl ProductFetcher,
) -> Result<ProductResolution, ScoreboardError> {
    let mut attempted = Vec::new();
    for candidate in product_candidates(target_date, lookback_days)? {
        attempted.push(candidate.clone());
        match fetcher.fetch(&candidate)? {
            FetchOutcome::Available(bytes) => {
                return Ok(ProductResolution {
                    resolved: Some(ResolvedProduct { candidate, bytes }),
                    attempted,
                });
            }
            FetchOutcome::NotPosted => {}
        }
    }
    Ok(ProductResolution {
        resolved: None,
        attempted,
    })
}

/// Build a scoreboard report from SP3 bytes.
pub fn score_sp3_bytes(
    bytes: &[u8],
    product_name: &str,
    product_date: ProductDate,
    options: &ScoreOptions,
) -> Result<ScoreboardReport, ScoreboardError> {
    let product = Sp3::parse(bytes)?;
    let state_samples = product.precise_ephemeris_state_samples();
    let position_samples = product.precise_ephemeris_samples();
    let provider = TdbEarthOrientationProvider::new();
    let use_state_samples = options.prefer_state_samples && !state_samples.is_empty();
    let state_counts = state_sample_counts(&state_samples);
    let position_counts = position_sample_counts(&position_samples);
    let oriented_samples = if use_state_samples {
        orient_state_samples(&state_samples, &provider)?
    } else {
        Vec::new()
    };

    let sat_results = product
        .satellites()
        .par_iter()
        .map(|&satellite| {
            score_satellite(
                satellite,
                use_state_samples,
                &state_counts,
                &position_counts,
                &oriented_samples,
                &position_samples,
                options,
            )
        })
        .collect::<Vec<_>>();

    let mut fitted = Vec::new();
    let mut skipped = Vec::new();
    let mut used_position_fallback = false;
    for outcome in sat_results {
        used_position_fallback |= outcome.used_position_fallback;
        match outcome.row {
            SatelliteScoreRow::Fit(row) => fitted.push(row),
            SatelliteScoreRow::Skip(row) => skipped.push(row),
        }
    }

    fitted.sort_by(|a, b| {
        a.rms_3d_m
            .total_cmp(&b.rms_3d_m)
            .then_with(|| a.satellite.cmp(&b.satellite))
    });
    skipped.sort_by(|a, b| a.satellite.cmp(&b.satellite));

    let per_constellation = constellation_reports(product.satellites(), &fitted, &skipped);
    let bottom = fitted.iter().rev().take(3).cloned().collect::<Vec<_>>();
    let top = fitted.iter().take(3).cloned().collect::<Vec<_>>();

    let mut notes = vec![
        force_model_note(&options.fit_options.force_model),
        "EOP source: core time-scale and Earth-orientation tables with zero polar motion."
            .to_string(),
        "Large residuals do not affect process exit status; skipped and failed satellites are shown."
            .to_string(),
    ];
    if used_position_fallback {
        notes.push(
            "Position-only SP3 rows used the core position-sample fitter; no velocity was synthesized."
                .to_string(),
        );
    }
    if product.skipped_records > 0 {
        notes.push(format!(
            "SP3 parser skipped {} unsupported declaration or record entries; see product.parser_skipped_records.",
            product.skipped_records
        ));
    }

    Ok(ScoreboardReport {
        date_utc: product_date.to_string(),
        sidereon_version: env!("CARGO_PKG_VERSION").to_string(),
        status: ScoreboardStatus::Scored,
        product: Some(ProductReport {
            name: product_name.to_string(),
            agency: product.header.agency,
            parser_skipped_records: product.skipped_records,
        }),
        attempted_candidates: Vec::new(),
        per_constellation,
        per_sat: PerSatelliteReport {
            top,
            bottom,
            skipped,
        },
        notes,
    })
}

/// Build a no-data report from attempted product candidates.
pub fn no_data_report(
    target_date: ProductDate,
    attempted: &[ProductCandidate],
) -> ScoreboardReport {
    ScoreboardReport {
        date_utc: target_date.to_string(),
        sidereon_version: env!("CARGO_PKG_VERSION").to_string(),
        status: ScoreboardStatus::NoData,
        product: None,
        attempted_candidates: attempted_candidate_reports(attempted),
        per_constellation: BTreeMap::new(),
        per_sat: empty_per_satellite_report(),
        notes: vec![
            format!(
                "No rapid or ultra-rapid SP3 product was posted in {} attempted candidates.",
                attempted.len()
            ),
            "This is recorded as no data, not a harness failure.".to_string(),
        ],
    }
}

/// Write the latest report file and append one compact JSON line to history.
pub fn write_report_outputs(
    report: &ScoreboardReport,
    output_path: Option<&Path>,
    history_path: Option<&Path>,
) -> Result<(), ScoreboardError> {
    if let Some(path) = output_path {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, report)?;
    }
    if let Some(path) = history_path {
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        serde_json::to_writer(&mut file, report)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

fn attempted_candidate_reports(attempted: &[ProductCandidate]) -> Vec<AttemptedCandidateReport> {
    attempted
        .iter()
        .map(|candidate| AttemptedCandidateReport {
            date_utc: candidate.date.to_string(),
            cadence: candidate.cadence.as_str().to_string(),
            source: candidate.source.to_string(),
            name: candidate.name.clone(),
            url: candidate.url.clone(),
        })
        .collect()
}

fn empty_per_satellite_report() -> PerSatelliteReport {
    PerSatelliteReport {
        top: Vec::new(),
        bottom: Vec::new(),
        skipped: Vec::new(),
    }
}

/// Format a report as pretty JSON.
pub fn report_json_pretty(report: &ScoreboardReport) -> Result<String, ScoreboardError> {
    serde_json::to_string_pretty(report).map_err(ScoreboardError::from)
}

/// Current UTC product date from the system clock.
pub fn utc_today() -> Result<ProductDate, ScoreboardError> {
    let unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ScoreboardError::SystemTimeBeforeUnixEpoch)?
        .as_secs();
    let j2000 = i64::try_from(unix_seconds)
        .map_err(|_| ScoreboardError::InvalidArgument("system time out of range".to_string()))?
        - UNIX_TO_J2000_S;
    product_date_from_j2000_seconds(j2000)
}

/// Parse `YYYY-MM-DD` into a product date.
pub fn parse_product_date(value: &str) -> Result<ProductDate, ScoreboardError> {
    let mut parts = value.split('-');
    let year = parts
        .next()
        .ok_or_else(|| ScoreboardError::InvalidArgument("date missing year".to_string()))?
        .parse::<i32>()
        .map_err(|_| ScoreboardError::InvalidArgument("date year is invalid".to_string()))?;
    let month = parts
        .next()
        .ok_or_else(|| ScoreboardError::InvalidArgument("date missing month".to_string()))?
        .parse::<u8>()
        .map_err(|_| ScoreboardError::InvalidArgument("date month is invalid".to_string()))?;
    let day = parts
        .next()
        .ok_or_else(|| ScoreboardError::InvalidArgument("date missing day".to_string()))?
        .parse::<u8>()
        .map_err(|_| ScoreboardError::InvalidArgument("date day is invalid".to_string()))?;
    if parts.next().is_some() {
        return Err(ScoreboardError::InvalidArgument(
            "date has extra fields".to_string(),
        ));
    }
    ProductDate::new(year, month, day).map_err(ScoreboardError::from)
}

fn product_date_candidates(
    target: ProductDate,
    lookback_days: u32,
) -> Result<Vec<ProductDate>, ScoreboardError> {
    let start = sidereon_core::astro::time::civil::j2000_seconds(
        target.year,
        i32::from(target.month),
        i32::from(target.day),
        0,
        0,
        0.0,
    ) as i64;
    let mut out = Vec::with_capacity(usize::try_from(lookback_days).unwrap_or(usize::MAX) + 1);
    for back in 0..=lookback_days {
        out.push(product_date_from_j2000_seconds(
            start - i64::from(back) * 86_400,
        )?);
    }
    Ok(out)
}

fn product_candidates(
    target: ProductDate,
    lookback_days: u32,
) -> Result<Vec<ProductCandidate>, ScoreboardError> {
    let dates = product_date_candidates(target, lookback_days)?;
    let mut out = Vec::new();
    for &date in &dates {
        out.extend(bkg_product_candidates(date)?);
    }
    for &date in &dates {
        out.extend(esa_product_candidates(date)?);
    }
    for date in dates {
        out.extend(gfz_product_candidates(date)?);
    }
    Ok(out)
}

fn bkg_product_candidates(date: ProductDate) -> Result<Vec<ProductCandidate>, ScoreboardError> {
    let week = date.gps_week()?;
    let daily_block = date_block(date, "0000");
    let mut out = vec![product_candidate(
        date,
        ProductCadence::Rapid,
        "BKG IGS",
        &format!("IGS0OPSRAP_{daily_block}_01D_15M_ORB.SP3"),
        &format!(
            "https://igs.bkg.bund.de/root_ftp/IGS/products/{week}/IGS0OPSRAP_{daily_block}_01D_15M_ORB.SP3.gz"
        ),
    )];
    for issue in ["1800", "1200", "0600", "0000"] {
        let date_block = date_block(date, issue);
        out.push(product_candidate(
            date,
            ProductCadence::UltraRapid,
            "BKG IGS",
            &format!("IGS0OPSULT_{date_block}_02D_15M_ORB.SP3"),
            &format!(
                "https://igs.bkg.bund.de/root_ftp/IGS/products/{week}/IGS0OPSULT_{date_block}_02D_15M_ORB.SP3.gz"
            ),
        ));
    }
    Ok(out)
}

fn esa_product_candidates(date: ProductDate) -> Result<Vec<ProductCandidate>, ScoreboardError> {
    let week = date.gps_week()?;
    let daily_block = date_block(date, "0000");
    let mut out = vec![product_candidate(
        date,
        ProductCadence::Rapid,
        "ESA",
        &format!("ESA0OPSRAP_{daily_block}_01D_05M_ORB.SP3"),
        &format!(
            "https://navigation-office.esa.int/products/gnss-products/{week}/ESA0OPSRAP_{daily_block}_01D_05M_ORB.SP3.gz"
        ),
    )];
    for issue in ["1800", "1200", "0600", "0000"] {
        let date_block = date_block(date, issue);
        out.push(product_candidate(
            date,
            ProductCadence::UltraRapid,
            "ESA",
            &format!("ESA0OPSULT_{date_block}_02D_05M_ORB.SP3"),
            &format!(
                "https://navigation-office.esa.int/products/gnss-products/{week}/ESA0OPSULT_{date_block}_02D_05M_ORB.SP3.gz"
            ),
        ));
    }
    Ok(out)
}

fn gfz_product_candidates(date: ProductDate) -> Result<Vec<ProductCandidate>, ScoreboardError> {
    let week = date.gps_week()?;
    let daily_block = date_block(date, "0000");
    let mut out = vec![product_candidate(
        date,
        ProductCadence::Rapid,
        "GFZ ISDC",
        &format!("GFZ0OPSRAP_{daily_block}_01D_05M_ORB.SP3"),
        &format!(
            "https://isdc-data.gfz.de/gnss/products/rapid/w{week}/GFZ0OPSRAP_{daily_block}_01D_05M_ORB.SP3.gz"
        ),
    )];
    for issue in [
        "2100", "1800", "1500", "1200", "0900", "0600", "0300", "0000",
    ] {
        let date_block = date_block(date, issue);
        out.push(product_candidate(
            date,
            ProductCadence::UltraRapid,
            "GFZ ISDC",
            &format!("GFZ0OPSULT_{date_block}_02D_05M_ORB.SP3"),
            &format!(
                "https://isdc-data.gfz.de/gnss/products/ultra/w{week}/GFZ0OPSULT_{date_block}_02D_05M_ORB.SP3.gz"
            ),
        ));
    }
    Ok(out)
}

fn product_candidate(
    date: ProductDate,
    cadence: ProductCadence,
    source: &'static str,
    name: &str,
    url: &str,
) -> ProductCandidate {
    ProductCandidate {
        date,
        cadence,
        source,
        name: name.to_string(),
        url: url.to_string(),
    }
}

fn date_block(date: ProductDate, issue: &str) -> String {
    format!("{}{:03}{issue}", date.year, date.day_of_year())
}

fn product_date_from_j2000_seconds(seconds: i64) -> Result<ProductDate, ScoreboardError> {
    let (year, month, day, _, _, _) = civil_from_j2000_seconds(seconds);
    ProductDate::new(
        i32::try_from(year)
            .map_err(|_| ScoreboardError::InvalidArgument("year out of range".to_string()))?,
        u8::try_from(month)
            .map_err(|_| ScoreboardError::InvalidArgument("month out of range".to_string()))?,
        u8::try_from(day)
            .map_err(|_| ScoreboardError::InvalidArgument("day out of range".to_string()))?,
    )
    .map_err(ScoreboardError::from)
}

fn fetch_https_product(candidate: &ProductCandidate) -> Result<FetchOutcome, ScoreboardError> {
    if !candidate.url.starts_with("https://") {
        return Err(ScoreboardError::NonHttpsUrl {
            url: candidate.url.clone(),
        });
    }

    let status_output = Command::new("curl")
        .args([
            "--http1.1",
            "--location",
            "--head",
            "--silent",
            "--retry",
            "2",
            "--output",
            "/dev/null",
            "--write-out",
            "%{http_code}",
            &candidate.url,
        ])
        .output()?;
    let status_text = String::from_utf8_lossy(&status_output.stdout);
    let status = status_text
        .trim()
        .parse::<u16>()
        .map_err(|_| ScoreboardError::Network {
            url: candidate.url.clone(),
            message: curl_message(&status_output.stderr, &status_output.stdout),
        })?;
    if status == 0 || status == 403 || status == 404 {
        return Ok(FetchOutcome::NotPosted);
    }
    if !status_output.status.success() {
        return Err(ScoreboardError::Network {
            url: candidate.url.clone(),
            message: curl_message(&status_output.stderr, &status_output.stdout),
        });
    }
    if !(200..300).contains(&status) {
        return Err(ScoreboardError::HttpStatus {
            url: candidate.url.clone(),
            status,
        });
    }

    let response = Command::new("curl")
        .args([
            "--http1.1",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "2",
            &candidate.url,
        ])
        .output()?;
    if !response.status.success() {
        return Err(ScoreboardError::Network {
            url: candidate.url.clone(),
            message: String::from_utf8_lossy(&response.stderr).to_string(),
        });
    }
    let bytes = response.stdout;
    if candidate.url.ends_with(".gz") {
        let mut decoder = GzDecoder::new(bytes.as_slice());
        let mut decoded = Vec::new();
        decoder.read_to_end(&mut decoded)?;
        Ok(FetchOutcome::Available(decoded))
    } else {
        Ok(FetchOutcome::Available(bytes))
    }
}

fn curl_message(stderr: &[u8], stdout: &[u8]) -> String {
    let stderr = String::from_utf8_lossy(stderr).trim().to_string();
    if !stderr.is_empty() {
        return stderr;
    }
    let stdout = String::from_utf8_lossy(stdout).trim().to_string();
    if stdout.is_empty() {
        "curl failed without diagnostic output".to_string()
    } else {
        format!("curl failed with output {stdout:?}")
    }
}

fn score_satellite(
    satellite: GnssSatelliteId,
    use_state_samples: bool,
    state_counts: &BTreeMap<GnssSatelliteId, usize>,
    position_counts: &BTreeMap<GnssSatelliteId, usize>,
    oriented_samples: &[OrientedPreciseEphemerisStateSample],
    position_samples: &[PreciseEphemerisSample],
    options: &ScoreOptions,
) -> SatelliteScoreOutcome {
    let position_count = position_counts.get(&satellite).copied().unwrap_or(0);
    if position_count == 0 {
        return SatelliteScoreOutcome {
            row: SatelliteScoreRow::Skip(skip_row(satellite, "missing_position_samples")),
            used_position_fallback: false,
        };
    }

    if use_state_samples {
        let state_count = state_counts.get(&satellite).copied().unwrap_or(0);
        if state_count != position_count {
            return SatelliteScoreOutcome {
                row: SatelliteScoreRow::Skip(skip_row(
                    satellite,
                    &format!("partial_velocity_samples:{state_count}/{position_count}"),
                )),
                used_position_fallback: false,
            };
        }
        let row = match fit_precise_ephemeris_state_sample_orbit(
            oriented_samples,
            satellite,
            &options.fit_options,
        ) {
            Ok(report) => report
                .ledger
                .per_sat
                .get(&satellite)
                .map(|stats| SatelliteScoreRow::Fit(fit_row(satellite, *stats)))
                .unwrap_or_else(|| SatelliteScoreRow::Skip(skip_row(satellite, "missing_ledger"))),
            Err(error) => {
                SatelliteScoreRow::Skip(skip_row(satellite, &format!("fit_error:{error}")))
            }
        };
        return SatelliteScoreOutcome {
            row,
            used_position_fallback: false,
        };
    }

    let sat_position_samples: Vec<PreciseEphemerisSample> = position_samples
        .iter()
        .copied()
        .filter(|sample| sample.sat == satellite)
        .collect();
    let row = match fit_precise_ephemeris_sample_orbit(
        &sat_position_samples,
        satellite,
        &options.fit_options,
    ) {
        Ok(report) => report
            .ledger
            .per_sat
            .get(&satellite)
            .map(|stats| SatelliteScoreRow::Fit(fit_row(satellite, *stats)))
            .unwrap_or_else(|| SatelliteScoreRow::Skip(skip_row(satellite, "missing_ledger"))),
        Err(error) => SatelliteScoreRow::Skip(skip_row(satellite, &format!("fit_error:{error}"))),
    };
    SatelliteScoreOutcome {
        row,
        used_position_fallback: true,
    }
}

fn state_sample_counts(
    samples: &[PreciseEphemerisStateSample],
) -> BTreeMap<GnssSatelliteId, usize> {
    let mut counts = BTreeMap::new();
    for sample in samples {
        *counts.entry(sample.sat).or_insert(0) += 1;
    }
    counts
}

fn position_sample_counts(samples: &[PreciseEphemerisSample]) -> BTreeMap<GnssSatelliteId, usize> {
    let mut counts = BTreeMap::new();
    for sample in samples {
        *counts.entry(sample.sat).or_insert(0) += 1;
    }
    counts
}

fn orient_state_samples(
    samples: &[PreciseEphemerisStateSample],
    provider: &impl EarthOrientationProvider,
) -> Result<Vec<OrientedPreciseEphemerisStateSample>, ScoreboardError> {
    samples
        .iter()
        .map(|sample| {
            let seed = EarthOrientation::from_instant(sample.epoch)?;
            let tdb_seconds = (seed.time_scales().jd_tdb - J2000_JD) * SECONDS_PER_DAY;
            let orientation = provider.orientation_at_tdb_seconds(tdb_seconds)?;
            Ok(OrientedPreciseEphemerisStateSample::new(
                *sample,
                orientation,
            ))
        })
        .collect()
}

fn fit_row(satellite: GnssSatelliteId, stats: OrbitResidualStats) -> SatelliteFitReport {
    SatelliteFitReport {
        satellite: satellite.to_string(),
        constellation: satellite.system.as_str().to_string(),
        rms_3d_m: stats.rms_3d_m,
        radial_rms_m: stats.radial_rms_m,
        along_rms_m: stats.along_rms_m,
        cross_rms_m: stats.cross_rms_m,
        n: stats.n,
        low_sample_count: stats.low_sample_count,
    }
}

fn skip_row(satellite: GnssSatelliteId, reason: &str) -> SatelliteSkipReport {
    SatelliteSkipReport {
        satellite: satellite.to_string(),
        constellation: satellite.system.as_str().to_string(),
        reason: reason.to_string(),
    }
}

fn constellation_reports(
    satellites: &[GnssSatelliteId],
    fitted: &[SatelliteFitReport],
    skipped: &[SatelliteSkipReport],
) -> BTreeMap<String, ConstellationReport> {
    let mut systems = BTreeSet::new();
    for sat in satellites {
        systems.insert(sat.system);
    }

    let mut reports = BTreeMap::new();
    for system in systems {
        let name = system.as_str().to_string();
        let sat_count = satellites.iter().filter(|sat| sat.system == system).count();
        let fit_rows: Vec<&SatelliteFitReport> = fitted
            .iter()
            .filter(|row| row.constellation == name)
            .collect();
        let skipped_count = skipped
            .iter()
            .filter(|row| row.constellation == name)
            .count();
        let mut rms_values = fit_rows.iter().map(|row| row.rms_3d_m).collect::<Vec<_>>();
        rms_values.sort_by(f64::total_cmp);
        reports.insert(
            name,
            ConstellationReport {
                sat_count,
                fit_count: fit_rows.len(),
                skipped: skipped_count,
                median_rms_3d_m: median(&rms_values),
                worst_rms_3d_m: rms_values.last().copied(),
            },
        );
    }
    reports
}

fn median(sorted: &[f64]) -> Option<f64> {
    match sorted.len() {
        0 => None,
        len if len % 2 == 1 => Some(sorted[len / 2]),
        len => Some((sorted[len / 2 - 1] + sorted[len / 2]) / 2.0),
    }
}

fn force_model_note(force_model: &ForceModelKind) -> String {
    match force_model {
        ForceModelKind::Composite { .. } => {
            "Force model: core composite model, production default is Earth Phase A without spacecraft SRP parameters.".to_string()
        }
        ForceModelKind::TwoBody { .. } => "Force model: core two-body model.".to_string(),
        ForceModelKind::TwoBodyJ2 { .. } => "Force model: core two-body plus J2 model.".to_string(),
    }
}

/// Run the default network-backed scoreboard pipeline.
pub fn run_default(
    target_date: ProductDate,
    lookback_days: u32,
) -> Result<ScoreboardReport, ScoreboardError> {
    run_with_fetcher(target_date, lookback_days, &HttpsFetcher)
}

/// Run the scoreboard pipeline with a supplied product fetcher.
pub fn run_with_fetcher(
    target_date: ProductDate,
    lookback_days: u32,
    fetcher: &impl ProductFetcher,
) -> Result<ScoreboardReport, ScoreboardError> {
    let resolution = resolve_latest_available_rapid_sp3(target_date, lookback_days, fetcher)?;
    let Some(resolved) = resolution.resolved else {
        return Ok(no_data_report(target_date, &resolution.attempted));
    };
    let mut report = score_sp3_bytes(
        &resolved.bytes,
        &resolved.candidate.name,
        resolved.candidate.date,
        &ScoreOptions::default(),
    )?;
    report.attempted_candidates = attempted_candidate_reports(&resolution.attempted);
    Ok(report)
}
