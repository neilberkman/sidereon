//! Validation scoreboard harness.
//!
//! The library keeps the scoring pipeline testable without network access. The
//! binary supplies the HTTPS fetcher and file output paths.

#![deny(missing_docs)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use rayon::prelude::*;
use serde::Serialize;
use sidereon_core::astro::frames::transforms::FrameTransformError;
use sidereon_core::astro::propagator::ForceModelKind;
use sidereon_core::astro::time::civil::civil_from_j2000_seconds;
use sidereon_core::constants::{J2000_JD, SECONDS_PER_DAY};
use sidereon_core::data::{
    newest_published_product, parse_archive_listing, publication_listing_urls,
    published_issue_age_minutes, AnalysisCenter, DataCatalogError, ProductDate, ProductDateTime,
    ProductType, PublishedProduct,
};
use sidereon_core::ephemeris::{
    fit_precise_ephemeris_sample_orbit, fit_precise_ephemeris_state_sample_orbit, parse_exact_sp3,
    ExactSp3Request, ExactSp3ValidationError, OrbitFitOptions, OrbitResidualStats,
    OrientedPreciseEphemerisStateSample, PreciseEphemerisSample, PreciseEphemerisStateSample, Sp3,
};
use sidereon_core::{
    EarthOrientation, EarthOrientationProvider, Error as CoreError, GnssSatelliteId,
    TdbEarthOrientationProvider,
};

// The scoreboard is an unpublished workspace tool.  Compile the facade's
// private transport decoder from its single source file so both call sites use
// identical logic without exposing a Rust-only Sidereon API.
#[path = "../../sidereon/src/compression.rs"]
mod compression;

const UNIX_TO_J2000_S: i64 = 946_728_000;
const DEFAULT_LOOKBACK_DAYS: u32 = 4;
const IGS_LONG_FILENAME_START_GPS_WEEK: u32 = 2238;
const MAX_SCOREBOARD_ARCHIVE_BYTES: usize = 64 * 1024 * 1024;
/// AIUB's whole-tree CSV listing is ~41 MiB; directory autoindexes are far
/// smaller. One bound covers both.
const MAX_SCOREBOARD_LISTING_BYTES: usize = 64 * 1024 * 1024;
const MAX_SCOREBOARD_PRODUCT_BYTES: usize = 500 * 1024 * 1024;
const MAX_COMMAND_DIAGNOSTIC_BYTES: usize = 1024 * 1024;
const CURL_CONNECT_TIMEOUT_SECONDS: &str = "30";
const CURL_TRANSFER_TIMEOUT_SECONDS: &str = "300";
const CURL_BODY_ATTEMPTS: usize = 3;
const CURL_STATUS_FRAME_PREFIX: &[u8] = b"\nSIDEREON_HTTP_STATUS:";
const CURL_STATUS_FRAME_BYTES: usize = CURL_STATUS_FRAME_PREFIX.len() + 3;

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
    /// HTTP status proving that this candidate URL was absent, when available.
    pub http_status: Option<u16>,
    /// Source-local fetch failure, when the resolver continued with another source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
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
    /// HTTP absence statuses aligned with `attempted`; `None` means unavailable or successful.
    pub attempted_http_statuses: Vec<Option<u16>>,
    /// Source-local fetch failures aligned with `attempted`.
    pub attempted_errors: Vec<Option<String>>,
}

/// Fetch result for one product candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchOutcome {
    /// Candidate was present and returned decompressed SP3 bytes. The resolver
    /// still performs exact date, issue, span, cadence, and grid validation.
    Available(Vec<u8>),
    /// The candidate URL returned an HTTP absence status.
    NotPosted {
        /// HTTP 404 or 410 when supplied by a network fetcher.
        http_status: Option<u16>,
    },
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
    #[error("non-HTTPS product URL for {archive_source} candidate {name}: {url}")]
    NonHttpsUrl {
        /// Archive or analysis-center source.
        archive_source: String,
        /// Canonical candidate filename.
        name: String,
        /// URL rejected by the fetcher.
        url: String,
    },
    /// An HTTP status other than a not-posted status was returned.
    #[error("HTTP status {status} while fetching {archive_source} candidate {name} at {url}")]
    HttpStatus {
        /// Archive or analysis-center source.
        archive_source: String,
        /// Canonical candidate filename.
        name: String,
        /// URL requested.
        url: String,
        /// HTTP status code.
        status: u16,
    },
    /// Network transport failed.
    #[error("network error while fetching {archive_source} candidate {name} at {url}: {message}")]
    Network {
        /// Archive or analysis-center source.
        archive_source: String,
        /// Canonical candidate filename.
        name: String,
        /// URL requested.
        url: String,
        /// Transport error message.
        message: String,
    },
    /// Candidate metadata cannot describe one exact SP3 request.
    #[error("invalid SP3 candidate {name} from {archive_source}: {reason}")]
    InvalidProductCandidate {
        /// Archive or analysis-center source.
        archive_source: String,
        /// Canonical candidate filename.
        name: String,
        /// Candidate-identity validation diagnostic.
        reason: String,
    },
    /// Fetched bytes failed exact-product integrity validation.
    #[error("integrity failure for SP3 candidate {name} from {archive_source}: {error}")]
    ExactSp3Integrity {
        /// Archive or analysis-center source.
        archive_source: String,
        /// Canonical candidate filename.
        name: String,
        /// Exact-content validation diagnostic.
        #[source]
        error: ExactSp3ValidationError,
    },
    /// Fetched bytes did not match a separately declared product digest.
    #[error(
        "digest mismatch for SP3 candidate {name} from {archive_source}: expected {expected}, got {actual}"
    )]
    DigestMismatch {
        /// Archive or analysis-center source.
        archive_source: String,
        /// Canonical candidate filename.
        name: String,
        /// Expected digest text.
        expected: String,
        /// Digest computed from the fetched bytes.
        actual: String,
    },
    /// The scoreboard has no verified archive convention for this product era.
    #[error(
        "unsupported scoreboard SP3 candidate era for {date} (GPS week {gps_week}); verified long-name candidates start at GPS week {minimum_gps_week}"
    )]
    UnsupportedCandidateEra {
        /// Requested candidate date.
        date: ProductDate,
        /// GPS week containing the requested date.
        gps_week: u32,
        /// First GPS week supported by the modeled long-name candidates.
        minimum_gps_week: u32,
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
///
/// Ordinary publication absence (`404`, `410`, or an equivalent fetcher result
/// without an HTTP status) permits trying the next candidate. Present but
/// malformed or identity-inconsistent bytes are terminal. Independent-source
/// transport failures may be bypassed, but the first such failure is returned
/// if no source yields a valid product.
pub fn resolve_latest_available_rapid_sp3(
    target_date: ProductDate,
    lookback_days: u32,
    fetcher: &impl ProductFetcher,
) -> Result<ProductResolution, ScoreboardError> {
    let mut attempted = Vec::new();
    let mut attempted_http_statuses = Vec::new();
    let mut attempted_errors = Vec::new();
    let mut unavailable_sources = BTreeSet::new();
    let mut first_nonabsence_error = None;
    for candidate in product_candidates(target_date, lookback_days)? {
        if unavailable_sources.contains(candidate.source) {
            continue;
        }
        attempted.push(candidate.clone());
        let exact_request = candidate_exact_sp3_request(&candidate)?;
        match fetcher.fetch(&candidate) {
            Ok(FetchOutcome::Available(bytes)) => {
                parse_exact_sp3(&bytes, &exact_request).map_err(|error| {
                    ScoreboardError::ExactSp3Integrity {
                        archive_source: candidate.source.to_string(),
                        name: candidate.name.clone(),
                        error,
                    }
                })?;
                attempted_http_statuses.push(None);
                attempted_errors.push(None);
                return Ok(ProductResolution {
                    resolved: Some(ResolvedProduct { candidate, bytes }),
                    attempted,
                    attempted_http_statuses,
                    attempted_errors,
                });
            }
            Ok(FetchOutcome::NotPosted { http_status }) => {
                if let Some(status) = http_status {
                    if status != 404 && status != 410 {
                        return Err(ScoreboardError::HttpStatus {
                            archive_source: candidate.source.to_string(),
                            name: candidate.name.clone(),
                            url: candidate.url.clone(),
                            status,
                        });
                    }
                }
                attempted_http_statuses.push(http_status);
                attempted_errors.push(None);
            }
            Err(error) if source_local_fetch_failure(&error) => {
                attempted_http_statuses.push(fetch_failure_http_status(&error));
                attempted_errors.push(Some(error.to_string()));
                unavailable_sources.insert(candidate.source);
                if first_nonabsence_error.is_none() {
                    first_nonabsence_error = Some(error);
                }
            }
            Err(error) => return Err(error),
        }
    }
    if let Some(error) = first_nonabsence_error {
        return Err(error);
    }
    Ok(ProductResolution {
        resolved: None,
        attempted,
        attempted_http_statuses,
        attempted_errors,
    })
}

/// Listing text or authoritative absence for one archive listing URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ListingOutcome {
    /// The listing answered with a body.
    Available(String),
    /// The listing URL returned an authoritative HTTP absence status
    /// (404/410): the archive answered, and this directory does not exist.
    NotPosted(u16),
}

/// Minimal listing-fetch interface used by [`publication_status`].
pub trait ListingFetcher {
    /// Fetch one archive listing URL.
    fn fetch_listing(&self, url: &str) -> Result<ListingOutcome, ScoreboardError>;
}

/// HTTPS listing fetcher used by the binary.
#[derive(Debug, Clone, Copy, Default)]
pub struct HttpsListingFetcher;

impl ListingFetcher for HttpsListingFetcher {
    fn fetch_listing(&self, url: &str) -> Result<ListingOutcome, ScoreboardError> {
        fetch_https_listing(url)
    }
}

/// Answer of one bounded publication-status query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicationStatusOutcome {
    /// The newest object this line has actually published.
    Published {
        /// Newest published issue, with the archive's verbatim
        /// modification text where the listing exposes one.
        product: PublishedProduct,
        /// Listing URL that evidenced it.
        listing_url: String,
        /// Whole minutes from the published issue's nominal epoch to the
        /// caller's query instant - the "N hours behind nominal" number.
        behind_nominal_minutes: i64,
    },
    /// Every listing answered, and none of them holds an object of this
    /// line: the archive is reachable but has published nothing in the
    /// bounded window.
    NothingPublished {
        /// Listing URLs consulted, in order.
        listing_urls: Vec<String>,
    },
    /// The archive itself did not answer, so publication state is unknown.
    /// This is deliberately distinct from [`Self::NothingPublished`]: an
    /// unreachable archive must never be reported as "nothing published".
    Unreachable {
        /// Listing URL whose fetch failed.
        listing_url: String,
        /// Transport diagnostic.
        reason: String,
    },
}

/// One bounded query answering "what is the newest published issue for this
/// center and product line, and how far behind nominal is it?".
///
/// This is the single networked composition of the pure core pieces
/// (`publication_listing_urls` -> `parse_archive_listing` ->
/// `newest_published_product` -> `published_issue_age_minutes`); the split
/// is doctrine, documented at the core's publication-status section. The
/// query fetches at most the bounded listing URLs (current week directory
/// plus the previous week, or one whole-tree listing), never any product
/// bytes, and never loops or polls.
///
/// Two asymmetric rules are deliberate; do not "fix" them:
///
/// - An authoritative 404 on the newer directory WALKS BACK to the older
///   one, because a 404 is the archive answering: "this directory does not
///   exist", which is exactly what a late archive looks like (the recorded
///   2026-08-04 BKG state, where the current week's directory had not been
///   created yet). Walking back converts that answer into the newest issue
///   that does exist.
/// - A transport failure NEVER walks back and reports
///   [`PublicationStatusOutcome::Unreachable`] immediately, because when the
///   newer directory's state is unknown, an answer from the older directory
///   is indistinguishable from real lag - a monitoring consumer would page
///   on phantom staleness (or worse, trust a stale "current" issue).
///
/// The same asymmetry governs an unrecognizable listing body: the archive
/// answered, but the answer cannot be read, so publication state is unknown
/// and the outcome is `Unreachable`, never "nothing published".
pub fn publication_status(
    center: AnalysisCenter,
    product_type: ProductType,
    now: ProductDateTime,
    fetcher: &impl ListingFetcher,
) -> Result<PublicationStatusOutcome, ScoreboardError> {
    let listing_urls = publication_listing_urls(center, product_type, now.date)?;
    for listing_url in &listing_urls {
        match fetcher.fetch_listing(listing_url) {
            Ok(ListingOutcome::Available(body)) => {
                // A reachable archive serving a body that fits no recognized
                // listing dialect cannot answer the publication question:
                // report it as unreachable-for-this-purpose rather than
                // letting an archive format change or error page read as
                // "nothing published".
                let objects = match parse_archive_listing(&body) {
                    Ok(objects) => objects,
                    Err(error) => {
                        return Ok(PublicationStatusOutcome::Unreachable {
                            listing_url: listing_url.clone(),
                            reason: error.to_string(),
                        });
                    }
                };
                if let Some(product) = newest_published_product(center, product_type, &objects)? {
                    let behind_nominal_minutes = published_issue_age_minutes(&product, now)?;
                    return Ok(PublicationStatusOutcome::Published {
                        product,
                        listing_url: listing_url.clone(),
                        behind_nominal_minutes,
                    });
                }
            }
            Ok(ListingOutcome::NotPosted(_)) => {}
            Err(error) => {
                return Ok(PublicationStatusOutcome::Unreachable {
                    listing_url: listing_url.clone(),
                    reason: error.to_string(),
                });
            }
        }
    }
    Ok(PublicationStatusOutcome::NothingPublished { listing_urls })
}

fn fetch_https_listing(url: &str) -> Result<ListingOutcome, ScoreboardError> {
    if !url.starts_with("https://") {
        return Err(ScoreboardError::NonHttpsUrl {
            archive_source: "listing".to_string(),
            name: url.rsplit('/').next().unwrap_or(url).to_string(),
            url: url.to_string(),
        });
    }
    let candidate = ProductCandidate {
        date: ProductDate {
            year: 2000,
            month: 1,
            day: 1,
        },
        cadence: ProductCadence::UltraRapid,
        source: "listing",
        name: url.rsplit('/').next().unwrap_or(url).to_string(),
        url: url.to_string(),
    };
    let outcome = fetch_bounded_http_body(
        &candidate,
        MAX_SCOREBOARD_LISTING_BYTES,
        CURL_BODY_ATTEMPTS,
        || {
            let mut command = Command::new("curl");
            command.args([
                "--http1.1",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--connect-timeout",
                CURL_CONNECT_TIMEOUT_SECONDS,
                "--max-time",
                CURL_TRANSFER_TIMEOUT_SECONDS,
                "--write-out",
                "\\nSIDEREON_HTTP_STATUS:%{http_code}",
                url,
            ]);
            command
        },
    )?;
    Ok(match outcome {
        HttpBodyOutcome::Available(bytes) => {
            ListingOutcome::Available(String::from_utf8_lossy(&bytes).into_owned())
        }
        HttpBodyOutcome::NotPosted(status) => ListingOutcome::NotPosted(status),
    })
}

fn candidate_exact_sp3_request(
    candidate: &ProductCandidate,
) -> Result<ExactSp3Request, ScoreboardError> {
    let invalid = |reason: String| ScoreboardError::InvalidProductCandidate {
        archive_source: candidate.source.to_string(),
        name: candidate.name.clone(),
        reason,
    };
    let mut fields = candidate.name.split('_');
    let product_code = fields
        .next()
        .ok_or_else(|| invalid("missing producer field".to_string()))?;
    let date_issue = fields
        .next()
        .ok_or_else(|| invalid("missing date/issue field".to_string()))?;
    let span = fields
        .next()
        .ok_or_else(|| invalid("missing span field".to_string()))?;
    let sample = fields
        .next()
        .ok_or_else(|| invalid("missing sample field".to_string()))?;
    let content = fields
        .next()
        .ok_or_else(|| invalid("missing content field".to_string()))?;
    if content != "ORB.SP3" || fields.next().is_some() {
        return Err(invalid(
            "expected a long-form ORB.SP3 product name".to_string(),
        ));
    }
    let (expected_product_code, expected_agency) = match (candidate.source, candidate.cadence) {
        ("BKG IGS", ProductCadence::Rapid) => ("IGS0OPSRAP", "IGS"),
        ("BKG IGS", ProductCadence::UltraRapid) => ("IGS0OPSULT", "IGS"),
        ("ESA", ProductCadence::Rapid) => ("ESA0OPSRAP", "ESOC"),
        ("ESA", ProductCadence::UltraRapid) => ("ESA0OPSULT", "ESOC"),
        ("GFZ ISDC", ProductCadence::Rapid) => ("GFZ0OPSRAP", "GFZ"),
        ("GFZ ISDC", ProductCadence::UltraRapid) => ("GFZ0OPSULT", "GFZ"),
        _ => {
            return Err(invalid(format!(
                "unsupported source/cadence pair {:?}/{:?}",
                candidate.source, candidate.cadence
            )))
        }
    };
    if product_code != expected_product_code {
        return Err(invalid(format!(
            "filename product code {product_code:?} does not match expected {expected_product_code:?}"
        )));
    }
    if date_issue.len() != 11 || !date_issue.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid(
            "date/issue field must contain YYYYDDDHHMM".to_string(),
        ));
    }
    let expected_date = format!("{}{:03}", candidate.date.year, candidate.date.day_of_year());
    if date_issue[..7] != expected_date {
        return Err(invalid(format!(
            "filename date {} does not match candidate date {}",
            &date_issue[..7],
            candidate.date
        )));
    }
    ExactSp3Request::new(candidate.date, Some(&date_issue[7..]), span, sample)
        .and_then(|request| request.with_expected_agency(expected_agency))
        .map_err(|error| invalid(error.to_string()))
}

fn source_local_fetch_failure(error: &ScoreboardError) -> bool {
    match error {
        ScoreboardError::Network { .. } => true,
        ScoreboardError::HttpStatus { status, .. } => (500..600).contains(status),
        _ => false,
    }
}

fn fetch_failure_http_status(error: &ScoreboardError) -> Option<u16> {
    match error {
        ScoreboardError::HttpStatus { status, .. } => Some(*status),
        _ => None,
    }
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
    let attempted_http_statuses = vec![None; attempted.len()];
    let attempted_errors = vec![None; attempted.len()];
    no_data_report_with_statuses(
        target_date,
        attempted,
        &attempted_http_statuses,
        &attempted_errors,
    )
}

fn no_data_report_with_statuses(
    target_date: ProductDate,
    attempted: &[ProductCandidate],
    attempted_http_statuses: &[Option<u16>],
    attempted_errors: &[Option<String>],
) -> ScoreboardReport {
    let mut notes = vec![
        format!(
            "No rapid or ultra-rapid SP3 candidate URL returned data in {} attempts.",
            attempted.len()
        ),
        "HTTP 404/410 establishes absence at an attempted URL, not authoritative publication status at another provider path."
            .to_string(),
    ];
    if attempted_errors.iter().any(Option::is_some) {
        notes.push(
            "Source-local fetch failures did not stop fallback to independent archives; diagnostics are retained on attempted candidates."
                .to_string(),
        );
    }
    ScoreboardReport {
        date_utc: target_date.to_string(),
        sidereon_version: env!("CARGO_PKG_VERSION").to_string(),
        status: ScoreboardStatus::NoData,
        product: None,
        attempted_candidates: attempted_candidate_reports(
            attempted,
            attempted_http_statuses,
            attempted_errors,
        ),
        per_constellation: BTreeMap::new(),
        per_sat: empty_per_satellite_report(),
        notes,
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

fn attempted_candidate_reports(
    attempted: &[ProductCandidate],
    attempted_http_statuses: &[Option<u16>],
    attempted_errors: &[Option<String>],
) -> Vec<AttemptedCandidateReport> {
    attempted
        .iter()
        .enumerate()
        .map(|(index, candidate)| AttemptedCandidateReport {
            date_utc: candidate.date.to_string(),
            cadence: candidate.cadence.as_str().to_string(),
            source: candidate.source.to_string(),
            name: candidate.name.clone(),
            url: candidate.url.clone(),
            http_status: attempted_http_statuses.get(index).copied().flatten(),
            error: attempted_errors.get(index).cloned().flatten(),
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
    let target_gps_week = target.gps_week()?;
    if target_gps_week < IGS_LONG_FILENAME_START_GPS_WEEK {
        return Err(ScoreboardError::UnsupportedCandidateEra {
            date: target,
            gps_week: target_gps_week,
            minimum_gps_week: IGS_LONG_FILENAME_START_GPS_WEEK,
        });
    }
    let mut dates = Vec::new();
    for date in product_date_candidates(target, lookback_days)? {
        if date.gps_week()? >= IGS_LONG_FILENAME_START_GPS_WEEK {
            dates.push(date);
        }
    }
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
            archive_source: candidate.source.to_string(),
            name: candidate.name.clone(),
            url: candidate.url.clone(),
        });
    }

    let transport_limit = if candidate.url.ends_with(".gz") {
        MAX_SCOREBOARD_ARCHIVE_BYTES
    } else {
        MAX_SCOREBOARD_PRODUCT_BYTES
    };
    // One GET is authoritative for both publication state and body bytes, so
    // there is no HEAD/GET time-of-check race.  Each transport/server retry gets
    // a fresh process and buffer; failed partial bodies are discarded.
    let body_outcome =
        fetch_bounded_http_body(candidate, transport_limit, CURL_BODY_ATTEMPTS, || {
            let mut command = Command::new("curl");
            command.args([
                "--http1.1",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--connect-timeout",
                CURL_CONNECT_TIMEOUT_SECONDS,
                "--max-time",
                CURL_TRANSFER_TIMEOUT_SECONDS,
                "--write-out",
                "\\nSIDEREON_HTTP_STATUS:%{http_code}",
                &candidate.url,
            ]);
            command
        })?;
    let bytes = match body_outcome {
        HttpBodyOutcome::Available(bytes) => bytes,
        HttpBodyOutcome::NotPosted(status) => {
            return Ok(FetchOutcome::NotPosted {
                http_status: Some(status),
            });
        }
    };
    if candidate.url.ends_with(".gz") {
        Ok(FetchOutcome::Available(decode_gzip_members(&bytes)?))
    } else {
        Ok(FetchOutcome::Available(bytes))
    }
}

fn decode_gzip_members(bytes: &[u8]) -> std::io::Result<Vec<u8>> {
    compression::decode_gzip_members(
        bytes,
        compression::GzipLimits::new(MAX_SCOREBOARD_ARCHIVE_BYTES, MAX_SCOREBOARD_PRODUCT_BYTES),
    )
    .map_err(Into::into)
}

fn output_bounded(command: &mut Command, limit: usize) -> std::io::Result<Output> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("child stdout was not piped"))?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| std::io::Error::other("child stderr was not piped"))?;

    // Drain stderr concurrently so a verbose failing process cannot fill its
    // pipe and block while stdout is being bounded.
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut overflow = false;
        let mut chunk = [0u8; 8 * 1024];
        loop {
            let read = stderr.read(&mut chunk)?;
            if read == 0 {
                break;
            }
            let remaining = MAX_COMMAND_DIAGNOSTIC_BYTES.saturating_sub(bytes.len());
            let retained = read.min(remaining);
            bytes.extend_from_slice(&chunk[..retained]);
            overflow |= retained != read;
        }
        Ok::<_, std::io::Error>((bytes, overflow))
    });

    let probe_limit = limit.saturating_add(1);
    let mut stdout_bytes = Vec::with_capacity(probe_limit.min(64 * 1024));
    let stdout_result = stdout
        .by_ref()
        .take(u64::try_from(probe_limit).unwrap_or(u64::MAX))
        .read_to_end(&mut stdout_bytes);
    if stdout_result.is_err() || stdout_bytes.len() > limit {
        let _ = child.kill();
    }
    let status = child.wait()?;
    let (stderr_bytes, stderr_overflow) = stderr_reader
        .join()
        .map_err(|_| std::io::Error::other("stderr reader thread panicked"))??;
    stdout_result?;
    if stdout_bytes.len() > limit {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "response body exceeds the {limit}-byte transport limit (read {} bytes)",
                stdout_bytes.len()
            ),
        ));
    }
    if stderr_overflow {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("command diagnostics exceed the {MAX_COMMAND_DIAGNOSTIC_BYTES}-byte limit"),
        ));
    }
    Ok(Output {
        status,
        stdout: stdout_bytes,
        stderr: stderr_bytes,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HttpBodyOutcome {
    Available(Vec<u8>),
    NotPosted(u16),
}

fn fetch_bounded_http_body(
    candidate: &ProductCandidate,
    limit: usize,
    attempts: usize,
    mut command: impl FnMut() -> Command,
) -> Result<HttpBodyOutcome, ScoreboardError> {
    if attempts == 0 {
        return Err(ScoreboardError::InvalidArgument(
            "at least one HTTP body attempt is required".to_string(),
        ));
    }

    let mut last_error: Option<ScoreboardError> = None;
    for _ in 0..attempts {
        let framed_limit = limit.checked_add(CURL_STATUS_FRAME_BYTES).ok_or_else(|| {
            ScoreboardError::InvalidArgument("HTTP body limit is too large".to_string())
        })?;
        let output = match output_bounded(&mut command(), framed_limit) {
            Ok(output) => output,
            Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                return Err(ScoreboardError::Io(error));
            }
            Err(error) => {
                last_error = Some(network_error(candidate, error.to_string()));
                continue;
            }
        };
        let process_exit = output.status.code();
        let stderr = output.stderr;
        let (body, status_bytes, status) = match split_curl_body_status(output.stdout) {
            Ok(parts) => parts,
            Err(message) => {
                last_error = Some(network_error(candidate, message));
                continue;
            }
        };

        match process_exit {
            Some(0) if (200..300).contains(&status) => {
                return Ok(HttpBodyOutcome::Available(body));
            }
            Some(0) => {
                return Err(network_error(
                    candidate,
                    format!("curl succeeded with unexpected HTTP status {status}"),
                ));
            }
            // curl documents exit 22 as an HTTP response rejected by --fail.
            // Only that exit makes a framed 404/410 authoritative evidence of
            // ordinary publication absence.  Transport failures can retain a
            // partial body ending in arbitrary digits and must never authorize
            // candidate fallback.
            Some(22) if status == 404 || status == 410 => {
                return Ok(HttpBodyOutcome::NotPosted(status));
            }
            Some(22) if (500..600).contains(&status) => {
                last_error = Some(ScoreboardError::HttpStatus {
                    archive_source: candidate.source.to_string(),
                    name: candidate.name.clone(),
                    url: candidate.url.clone(),
                    status,
                });
                continue;
            }
            Some(22) if (400..500).contains(&status) => {
                return Err(ScoreboardError::HttpStatus {
                    archive_source: candidate.source.to_string(),
                    name: candidate.name.clone(),
                    url: candidate.url.clone(),
                    status,
                });
            }
            _ => {
                last_error = Some(network_error(
                    candidate,
                    curl_message(&stderr, &status_bytes),
                ));
                continue;
            }
        }
    }
    Err(last_error.unwrap_or_else(|| {
        network_error(candidate, "HTTP body fetch produced no result".to_string())
    }))
}

fn split_curl_body_status(mut stdout: Vec<u8>) -> Result<(Vec<u8>, [u8; 3], u16), String> {
    if stdout.len() < CURL_STATUS_FRAME_BYTES {
        return Err("curl output omitted its final HTTP status".to_string());
    }
    let frame_offset = stdout.len() - CURL_STATUS_FRAME_BYTES;
    if &stdout[frame_offset..frame_offset + CURL_STATUS_FRAME_PREFIX.len()]
        != CURL_STATUS_FRAME_PREFIX
    {
        return Err("curl output omitted its final HTTP status frame".to_string());
    }
    let status_offset = stdout.len() - 3;
    let status_bytes: [u8; 3] = stdout[status_offset..]
        .try_into()
        .map_err(|_| "curl emitted a malformed HTTP status".to_string())?;
    if !status_bytes.iter().all(u8::is_ascii_digit) {
        return Err("curl emitted a non-numeric HTTP status".to_string());
    }
    stdout.truncate(frame_offset);
    let status = u16::from(status_bytes[0] - b'0') * 100
        + u16::from(status_bytes[1] - b'0') * 10
        + u16::from(status_bytes[2] - b'0');
    Ok((stdout, status_bytes, status))
}

fn network_error(candidate: &ProductCandidate, message: String) -> ScoreboardError {
    ScoreboardError::Network {
        archive_source: candidate.source.to_string(),
        name: candidate.name.clone(),
        url: candidate.url.clone(),
        message,
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
        return Ok(no_data_report_with_statuses(
            target_date,
            &resolution.attempted,
            &resolution.attempted_http_statuses,
            &resolution.attempted_errors,
        ));
    };
    let mut report = score_sp3_bytes(
        &resolved.bytes,
        &resolved.candidate.name,
        resolved.candidate.date,
        &ScoreOptions::default(),
    )?;
    report.attempted_candidates = attempted_candidate_reports(
        &resolution.attempted,
        &resolution.attempted_http_statuses,
        &resolution.attempted_errors,
    );
    Ok(report)
}

#[cfg(test)]
mod fetch_tests {
    use super::*;

    fn gzip_member(content: &[u8]) -> Vec<u8> {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(content).expect("write gzip member");
        encoder.finish().expect("finish gzip member")
    }

    fn candidate() -> ProductCandidate {
        ProductCandidate {
            date: ProductDate::new(2026, 7, 14).expect("date"),
            cadence: ProductCadence::UltraRapid,
            source: "AIUB CODE",
            name: "COD0OPSULT_20261950000_01D_05M_ORB.SP3".to_string(),
            url: "https://www.aiub.unibe.ch/download/CODE/COD0OPSULT_20261950000_01D_05M_ORB.SP3"
                .to_string(),
        }
    }

    #[test]
    fn gzip_decoder_consumes_every_complete_member() {
        let mut archive = gzip_member(b"first");
        archive.extend_from_slice(&gzip_member(b"second"));
        assert_eq!(decode_gzip_members(&archive).unwrap(), b"firstsecond");

        let mut long_comment = gzip_member(b"long comment");
        long_comment[3] |= 0x10;
        long_comment.splice(10..10, vec![b'c'; 70_000].into_iter().chain([0]));
        assert_eq!(decode_gzip_members(&long_comment).unwrap(), b"long comment");

        let mut junk_tailed = archive.clone();
        junk_tailed.extend_from_slice(b"not another gzip member");
        assert!(decode_gzip_members(&junk_tailed).is_err());

        archive.pop();
        assert!(decode_gzip_members(&archive).is_err());
    }

    #[test]
    fn command_stdout_is_bounded_during_the_read() {
        let mut exact = Command::new("sh");
        exact.args(["-c", "printf 1234"]);
        assert_eq!(output_bounded(&mut exact, 4).unwrap().stdout, b"1234");

        let mut over = Command::new("sh");
        over.args(["-c", "printf 1234"]);
        let error = output_bounded(&mut over, 3).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("3-byte transport limit"));

        let mut diagnostics = Command::new("sh");
        diagnostics.args(["-c", "head -c 1048577 /dev/zero >&2"]);
        let error = output_bounded(&mut diagnostics, 0).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("diagnostics exceed"));
    }

    #[test]
    fn body_retry_discards_partial_bytes_and_starts_with_a_fresh_buffer() {
        let candidate = candidate();
        let mut built = 0usize;
        let output = fetch_bounded_http_body(&candidate, 64, 2, || {
            built += 1;
            let mut command = Command::new("sh");
            if built == 1 {
                command.args(["-c", "printf 'partial\\nSIDEREON_HTTP_STATUS:200'; exit 1"]);
            } else {
                command.args(["-c", "printf 'complete\\nSIDEREON_HTTP_STATUS:200'"]);
            }
            command
        })
        .unwrap();
        assert_eq!(built, 2);
        assert_eq!(output, HttpBodyOutcome::Available(b"complete".to_vec()));

        let mut oversized_attempts = 0usize;
        let error = fetch_bounded_http_body(&candidate, 3, 3, || {
            oversized_attempts += 1;
            let mut command = Command::new("sh");
            command.args(["-c", "printf '1234\\nSIDEREON_HTTP_STATUS:200'"]);
            command
        })
        .unwrap_err();
        assert!(matches!(
            error,
            ScoreboardError::Io(ref io) if io.kind() == std::io::ErrorKind::InvalidData
        ));
        assert_eq!(oversized_attempts, 1);
    }

    #[test]
    fn only_candidate_absence_statuses_are_not_posted() {
        let candidate = candidate();
        let mut attempts = 0usize;
        assert_eq!(
            fetch_bounded_http_body(&candidate, 16, 3, || {
                attempts += 1;
                let mut command = Command::new("sh");
                command.args(["-c", "printf '\\nSIDEREON_HTTP_STATUS:404'; exit 22"]);
                command
            })
            .expect("404 classification"),
            HttpBodyOutcome::NotPosted(404)
        );
        assert_eq!(attempts, 1, "ordinary absence must not be retried");
        assert_eq!(
            fetch_bounded_http_body(&candidate, 16, 1, || {
                let mut command = Command::new("sh");
                command.args(["-c", "printf '\\nSIDEREON_HTTP_STATUS:410'; exit 22"]);
                command
            })
            .expect("410 classification"),
            HttpBodyOutcome::NotPosted(410)
        );
    }

    #[test]
    fn access_denial_preserves_status_url_and_candidate_details() {
        let candidate = candidate();
        match fetch_bounded_http_body(&candidate, 16, 1, || {
            let mut command = Command::new("sh");
            command.args(["-c", "printf '\\nSIDEREON_HTTP_STATUS:403'; exit 22"]);
            command
        }) {
            Err(ScoreboardError::HttpStatus {
                archive_source,
                name,
                url,
                status,
            }) => {
                assert_eq!(archive_source, candidate.source);
                assert_eq!(name, candidate.name);
                assert_eq!(url, candidate.url);
                assert_eq!(status, 403);
            }
            other => panic!("expected HTTP status diagnostic, got {other:?}"),
        }
    }

    #[test]
    fn server_error_retries_fresh_and_preserves_the_final_http_status() {
        let candidate = candidate();
        let mut attempts = 0usize;
        match fetch_bounded_http_body(&candidate, 16, 3, || {
            attempts += 1;
            let mut command = Command::new("sh");
            command.args(["-c", "printf '\\nSIDEREON_HTTP_STATUS:503'; exit 22"]);
            command
        }) {
            Err(ScoreboardError::HttpStatus { status, .. }) => assert_eq!(status, 503),
            other => panic!("expected HTTP 503 diagnostic, got {other:?}"),
        }
        assert_eq!(attempts, 3);
    }

    #[test]
    fn transport_failure_is_not_classified_as_publication_absence() {
        let candidate = candidate();
        match fetch_bounded_http_body(&candidate, 16, 1, || {
            let mut command = Command::new("sh");
            command.args([
                "-c",
                "printf '\\nSIDEREON_HTTP_STATUS:000'; printf 'connection refused' >&2; exit 1",
            ]);
            command
        }) {
            Err(ScoreboardError::Network {
                archive_source,
                name,
                url,
                message,
            }) => {
                assert_eq!(archive_source, candidate.source);
                assert_eq!(name, candidate.name);
                assert_eq!(url, candidate.url);
                assert!(message.contains("connection refused"));
            }
            other => panic!("expected network diagnostic, got {other:?}"),
        }
    }

    #[test]
    fn truncated_transfer_ending_in_absence_status_never_authorizes_fallback() {
        let candidate = candidate();
        let mut attempts = 0usize;
        match fetch_bounded_http_body(&candidate, 64, 2, || {
            attempts += 1;
            let mut command = Command::new("sh");
            command.args([
                "-c",
                "printf 'partial404\\nSIDEREON_HTTP_STATUS:404'; exit 18",
            ]);
            command
        }) {
            Err(ScoreboardError::Network { .. }) => {}
            other => panic!("expected terminal transport diagnostic, got {other:?}"),
        }
        assert_eq!(
            attempts, 2,
            "transport failures should exhaust fresh retries"
        );
    }

    #[test]
    fn filename_date_mismatch_is_a_candidate_configuration_error() {
        let mut candidate = candidate();
        candidate.name = "COD0OPSULT_20261940000_01D_05M_ORB.SP3".to_string();

        assert!(matches!(
            candidate_exact_sp3_request(&candidate),
            Err(ScoreboardError::InvalidProductCandidate { .. })
        ));
    }
}
