//! Semantic validation for an exact SP3 product request.
//!
//! [`Sp3::parse`] remains a general, permissive SP3 reader. Acquisition code
//! that promised a particular date, span, and sample interval should additionally
//! use [`validate_exact_sp3`] (or [`parse_exact_sp3`]) before accepting bytes as
//! that product.

use core::fmt;

use crate::astro::time::civil::j2000_seconds;
use crate::data::{AnalysisCenter, DataCatalogError, ProductDate, ProductIdentity, ProductType};
use crate::tolerances::WHOLE_SECOND_EPS_S;

use super::{Sp3, Sp3DataType, Sp3Version};

/// Maximum legal epoch interval from the SP3-d specification, in seconds.
/// The interval must be strictly less than this value.
const SP3_MAX_EPOCH_INTERVAL_S: f64 = 100_000.0;
/// Modified Julian day containing the J2000 epoch (at 12:00).
const J2000_MJD_DAY: i64 = 51_544;
/// Modified Julian day at the GPS week-numbering origin, 1980-01-06 00:00.
const GPS_ZERO_MJD_DAY: i64 = 44_244;
const SECONDS_PER_DAY_I64: i64 = 86_400;
const SECONDS_PER_WEEK_I64: i64 = 604_800;

/// The two regular-grid representations accepted for an exact declared span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactSp3Coverage {
    /// The boundary is excluded. A 24-hour, five-minute product has 288 epochs,
    /// ending at 23:55.
    HalfOpen,
    /// The boundary is included. A 24-hour, five-minute product has 289 epochs,
    /// ending at the following midnight.
    Inclusive,
}

/// Requested identity fields needed to validate decompressed SP3 content.
///
/// This source-independent request is useful to acquisition resolvers whose
/// candidates are defined by an official archive convention but are not catalog
/// entries. [`ExactSp3Request::from_identity`] supplies the same values from a
/// full [`ProductIdentity`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactSp3Request {
    date: ProductDate,
    issue: Option<String>,
    span: String,
    sample: String,
    format_version: Option<String>,
    expected_agency: Option<String>,
    /// Whole seconds from the filename epoch to the cataloged first content
    /// epoch. Only [`Self::from_identity`] can set a nonzero value.
    content_start_offset_s: i64,
}

impl ExactSp3Request {
    /// Build and validate a source-independent exact SP3 request.
    pub fn new(
        date: ProductDate,
        issue: Option<&str>,
        span: &str,
        sample: &str,
    ) -> Result<Self, ExactSp3ValidationError> {
        // ProductDate's fields are public, so revalidate caller-built values.
        ProductDate::new(date.year, date.month, date.day)
            .map_err(ExactSp3ValidationError::Catalog)?;
        parse_issue(issue)?;
        parse_duration_token(span, DurationField::Span)?;
        let cadence_s = parse_duration_token(sample, DurationField::Sample)?;
        if cadence_s as f64 >= SP3_MAX_EPOCH_INTERVAL_S {
            return Err(ExactSp3ValidationError::UnsupportedSampleToken {
                token: sample.to_owned(),
            });
        }

        Ok(Self {
            date,
            issue: issue.map(str::to_owned),
            span: span.to_owned(),
            sample: sample.to_owned(),
            format_version: None,
            expected_agency: None,
            content_start_offset_s: 0,
        })
    }

    /// Build a request from a complete catalog identity.
    pub fn from_identity(identity: &ProductIdentity) -> Result<Self, ExactSp3ValidationError> {
        if identity.family != ProductType::Sp3 {
            return Err(ExactSp3ValidationError::WrongProductFamily {
                actual: identity.family,
            });
        }
        identity
            .validate()
            .map_err(ExactSp3ValidationError::Catalog)?;
        let mut request = Self::new(
            identity.date,
            identity.issue.as_deref(),
            &identity.span,
            &identity.sample,
        )?;
        request.format_version = identity.format_version.clone();
        request.content_start_offset_s = crate::data::exact_sp3_content_start_offset_s(identity)
            .map_err(ExactSp3ValidationError::Catalog)?;
        request.expected_agency = Some(
            match identity.analysis_center {
                AnalysisCenter::Igs => "IGS",
                AnalysisCenter::Esa | AnalysisCenter::EsaUlt => "ESOC",
                AnalysisCenter::Gfz | AnalysisCenter::GfzUlt => "GFZ",
                AnalysisCenter::Cod
                | AnalysisCenter::CodRap
                | AnalysisCenter::CodPrd1
                | AnalysisCenter::CodPrd2
                | AnalysisCenter::CodUlt => "AIUB",
                AnalysisCenter::IgsUlt => "IGS",
                AnalysisCenter::WumNrt => "WHU",
            }
            .to_owned(),
        );
        Ok(request)
    }

    /// Require the producing-agency code declared on SP3 header line 1.
    ///
    /// Agency codes are one to four upper-case ASCII alphanumeric characters,
    /// matching the SP3 `A4` field and official IGS producer codes.
    pub fn with_expected_agency(mut self, agency: &str) -> Result<Self, ExactSp3ValidationError> {
        if agency.is_empty()
            || agency.len() > 4
            || !agency
                .bytes()
                .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit())
        {
            return Err(ExactSp3ValidationError::InvalidExpectedAgency {
                agency: agency.to_owned(),
            });
        }
        self.expected_agency = Some(agency.to_owned());
        Ok(self)
    }

    /// Requested filename date.
    ///
    /// For requests built with [`Self::new`], this is also the required first
    /// content date. [`Self::from_identity`] may apply a cataloged historical
    /// content-start convention while retaining this filename date.
    pub fn date(&self) -> ProductDate {
        self.date
    }

    /// Optional requested `HHMM` filename issue/epoch token.
    ///
    /// No issue means midnight. As with [`Self::date`], a catalog-derived
    /// request may require a historical content start before this epoch.
    pub fn issue(&self) -> Option<&str> {
        self.issue.as_deref()
    }

    /// Requested coverage-period token.
    pub fn span(&self) -> &str {
        &self.span
    }

    /// Requested sample-interval token.
    pub fn sample(&self) -> &str {
        &self.sample
    }

    /// Optional content revision inherited from a resolved product identity.
    pub fn format_version(&self) -> Option<&str> {
        self.format_version.as_deref()
    }

    /// Producing-agency code required from SP3 header line 1, when constrained.
    pub fn expected_agency(&self) -> Option<&str> {
        self.expected_agency.as_deref()
    }
}

/// Integrity failure while parsing or validating an exact SP3 product.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ExactSp3ValidationError {
    /// The bytes are not a parseable SP3 product.
    Parse(crate::Error),
    /// The full catalog identity is invalid.
    Catalog(DataCatalogError),
    /// A non-SP3 identity was supplied to the SP3 validator.
    WrongProductFamily {
        /// Supplied family.
        actual: ProductType,
    },
    /// The optional start/issue token is not a valid `HHMM` value.
    InvalidIssue {
        /// Supplied token; an absent issue is represented by midnight and is
        /// never invalid.
        issue: String,
    },
    /// The requested span token is not a supported fixed-duration token.
    UnsupportedSpanToken {
        /// Supplied token.
        token: String,
    },
    /// The requested sample token is not a supported positive SP3 interval.
    UnsupportedSampleToken {
        /// Supplied token.
        token: String,
    },
    /// A valid fixed-duration span token did not use the longest exact unit.
    NonCanonicalSpanToken {
        /// Supplied token.
        token: String,
        /// Equivalent canonical token.
        canonical: String,
    },
    /// A valid fixed-duration sample token did not use the longest exact unit.
    NonCanonicalSampleToken {
        /// Supplied token.
        token: String,
        /// Equivalent canonical token.
        canonical: String,
    },
    /// An expected agency constraint is not a valid SP3 `A4` producer code.
    InvalidExpectedAgency {
        /// Supplied agency code.
        agency: String,
    },
    /// Parsed SP3 producing agency differs from the exact request.
    AgencyMismatch {
        /// Required agency code.
        expected: String,
        /// Parsed header agency code.
        actual: String,
    },
    /// The product omitted its terminal `EOF` record.
    MissingEof,
    /// The product contained an EOF-like record that violated the accepted
    /// logical-record grammar.
    MalformedEofRecord {
        /// One-based logical-record line number.
        line_number: usize,
        /// Length of the offending logical record in bytes.
        record_length: usize,
    },
    /// Nonblank records appeared after the terminal `EOF` marker.
    TrailingContentAfterEof,
    /// A mandatory SP3 header record count is incomplete or inconsistent.
    MandatoryHeaderRecordCount {
        /// Record prefix (`+`, `++`, `%c`, `%f`, or `%i`).
        record: &'static str,
        /// Required exact or minimum count.
        expected: usize,
        /// Parsed count.
        actual: usize,
    },
    /// The first `+` record does not carry a parseable declared satellite count.
    MissingDeclaredSatelliteCount,
    /// The line-3 count differs from the number of non-padding raw declaration
    /// tokens across all `+` records.
    DeclaredSatelliteCountMismatch {
        /// Count from line 3.
        declared: usize,
        /// Number of raw non-padding tokens.
        tokens: usize,
    },
    /// A satellite token appears more than once in the header declaration.
    DuplicateDeclaredSatellite {
        /// Duplicate raw satellite token.
        token: String,
        /// First declaration index.
        first_index: usize,
        /// Duplicate declaration index.
        duplicate_index: usize,
    },
    /// The header declares no satellites.
    NoDeclaredSatellites,
    /// One epoch's raw P or V records do not exactly match the declared
    /// satellite count and order.
    SatelliteRecordSequenceMismatch {
        /// Record type (`P` or `V`).
        record: &'static str,
        /// Zero-based parsed epoch index.
        epoch_index: usize,
        /// Raw declared satellite order.
        expected: Vec<String>,
        /// Raw record satellite order.
        actual: Vec<String>,
    },
    /// P/V records have the right per-type tokens but not the required body
    /// order (for a velocity product, `P(sat), V(sat)` pairs).
    BodyRecordInterleavingMismatch {
        /// Zero-based parsed epoch index.
        epoch_index: usize,
        /// Required tagged record sequence, such as `PG01`, `VG01`.
        expected: Vec<String>,
        /// Parsed tagged record sequence.
        actual: Vec<String>,
    },
    /// The SP3 header declares a non-finite epoch interval.
    NonFiniteHeaderCadence,
    /// The SP3 header declares a zero or negative epoch interval.
    NonPositiveHeaderCadence {
        /// Declared interval.
        actual_s: f64,
    },
    /// The SP3 header interval is outside the range allowed by the format.
    UnsupportedHeaderCadence {
        /// Declared interval.
        actual_s: f64,
    },
    /// Header cadence does not equal the exact requested sample interval.
    CadenceMismatch {
        /// Cadence derived from the trusted request token.
        requested_s: f64,
        /// Cadence declared on SP3 header line 2.
        header_s: f64,
    },
    /// Header line 1's declared number of epochs differs from the parsed grid.
    DeclaredEpochCountMismatch {
        /// Count from header line 1.
        declared: u64,
        /// Number of parsed epoch records.
        parsed: usize,
    },
    /// Header line 1 does not contain an interpretable start epoch.
    MissingDeclaredStart,
    /// Header line 1's start does not equal the exact requested start.
    DeclaredStartMismatch {
        /// Exact requested start, seconds since J2000.
        requested_j2000_s: f64,
        /// Header line 1 start, seconds since J2000.
        declared_j2000_s: f64,
    },
    /// The requested start predates the GPS week-numbering epoch that SP3 line
    /// 2 uses.
    RequestBeforeGpsEpoch,
    /// A floating-point SP3 line-2 start field is not finite.
    NonFiniteHeaderStartMetadata {
        /// SP3 header field name.
        field: &'static str,
    },
    /// An SP3 line-2 start field is outside its specified range.
    InvalidHeaderStartMetadata {
        /// SP3 header field name.
        field: &'static str,
        /// Parsed field value.
        actual: f64,
    },
    /// SP3 line-2 week/SOW or MJD metadata does not represent the trusted
    /// requested start.
    HeaderStartMetadataMismatch {
        /// Logical SP3 header field (`gps_week`, `seconds_of_week`, or `mjd`).
        field: &'static str,
        /// Value derived from the trusted request.
        requested: f64,
        /// Parsed header value.
        actual: f64,
    },
    /// The product contains no parsed epoch records.
    EmptyEpochGrid,
    /// The first parsed epoch does not equal the exact requested start.
    FirstEpochMismatch {
        /// Exact requested start, seconds since J2000.
        requested_j2000_s: f64,
        /// First parsed epoch, seconds since J2000.
        actual_j2000_s: f64,
    },
    /// Parsed epochs are not a strictly increasing regular requested-cadence grid.
    IrregularEpochGrid {
        /// Index of the later epoch in the failing pair.
        epoch_index: usize,
        /// Exact requested cadence.
        requested_s: f64,
        /// Difference between this epoch and its predecessor.
        actual_s: f64,
    },
    /// The declared span is not an integer multiple of the requested cadence.
    SpanNotMultipleOfCadence {
        /// Requested span.
        span_s: u64,
        /// Requested cadence.
        cadence_s: u64,
    },
    /// A regular grid does not represent either the half-open or inclusive
    /// form of the exact requested span.
    SpanMismatch {
        /// Parsed epoch count.
        parsed: usize,
        /// Required count for half-open coverage.
        half_open: usize,
        /// Required count for inclusive coverage.
        inclusive: usize,
    },
    /// A resolved identity constrained an SP3 content revision that does not
    /// match the parsed header.
    FormatVersionMismatch {
        /// Requested revision string.
        requested: String,
        /// Parsed revision string.
        actual: String,
    },
}

impl fmt::Display for ExactSp3ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => write!(f, "exact SP3 parse failed: {error}"),
            Self::Catalog(error) => write!(f, "invalid exact SP3 identity: {error}"),
            Self::WrongProductFamily { actual } => {
                write!(f, "exact SP3 validation cannot validate {actual}")
            }
            Self::InvalidIssue { issue } => {
                write!(f, "invalid exact SP3 issue time {issue:?}; expected HHMM")
            }
            Self::UnsupportedSpanToken { token } => {
                write!(f, "unsupported exact SP3 span token {token:?}")
            }
            Self::UnsupportedSampleToken { token } => {
                write!(f, "unsupported exact SP3 sample token {token:?}")
            }
            Self::NonCanonicalSpanToken { token, canonical } => write!(
                f,
                "noncanonical exact SP3 span token {token:?}; use {canonical:?}"
            ),
            Self::NonCanonicalSampleToken { token, canonical } => write!(
                f,
                "noncanonical exact SP3 sample token {token:?}; use {canonical:?}"
            ),
            Self::InvalidExpectedAgency { agency } => {
                write!(f, "invalid exact SP3 expected agency {agency:?}")
            }
            Self::AgencyMismatch { expected, actual } => write!(
                f,
                "SP3 agency mismatch: requested {expected:?}, header declares {actual:?}"
            ),
            Self::MissingEof => write!(f, "SP3 product is missing its EOF record"),
            Self::MalformedEofRecord {
                line_number,
                record_length,
            } => write!(
                f,
                "SP3 product contains a malformed EOF record at line {line_number} ({record_length} bytes)"
            ),
            Self::TrailingContentAfterEof => {
                write!(f, "SP3 product contains nonblank records after EOF")
            }
            Self::MandatoryHeaderRecordCount {
                record,
                expected,
                actual,
            } => write!(
                f,
                "SP3 header record {record} count is {actual}, expected {expected}"
            ),
            Self::MissingDeclaredSatelliteCount => {
                write!(f, "SP3 line 3 has no valid declared satellite count")
            }
            Self::DeclaredSatelliteCountMismatch { declared, tokens } => write!(
                f,
                "SP3 declared satellite-count mismatch: line 3 declares {declared}, header contains {tokens} tokens"
            ),
            Self::DuplicateDeclaredSatellite {
                token,
                first_index,
                duplicate_index,
            } => write!(
                f,
                "SP3 header satellite {token:?} is duplicated at indices {first_index} and {duplicate_index}"
            ),
            Self::NoDeclaredSatellites => {
                write!(f, "SP3 product declares no satellites")
            }
            Self::SatelliteRecordSequenceMismatch {
                record,
                epoch_index,
                expected,
                actual,
            } => write!(
                f,
                "SP3 epoch {epoch_index} {record}-record sequence mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::BodyRecordInterleavingMismatch {
                epoch_index,
                expected,
                actual,
            } => write!(
                f,
                "SP3 epoch {epoch_index} body record ordering mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::NonFiniteHeaderCadence => {
                write!(f, "SP3 header cadence is not finite")
            }
            Self::NonPositiveHeaderCadence { actual_s } => {
                write!(f, "SP3 header cadence must be positive, got {actual_s}")
            }
            Self::UnsupportedHeaderCadence { actual_s } => write!(
                f,
                "SP3 header cadence {actual_s} s is outside the supported format range"
            ),
            Self::CadenceMismatch {
                requested_s,
                header_s,
            } => write!(
                f,
                "SP3 cadence mismatch: requested {requested_s} s, header declares {header_s} s"
            ),
            Self::DeclaredEpochCountMismatch { declared, parsed } => write!(
                f,
                "SP3 epoch-count mismatch: header declares {declared}, parsed {parsed}"
            ),
            Self::MissingDeclaredStart => {
                write!(f, "SP3 header line 1 has no valid declared start epoch")
            }
            Self::DeclaredStartMismatch {
                requested_j2000_s,
                declared_j2000_s,
            } => write!(
                f,
                "SP3 declared start mismatch: requested {requested_j2000_s} J2000 s, header declares {declared_j2000_s} J2000 s"
            ),
            Self::RequestBeforeGpsEpoch => {
                write!(f, "exact SP3 request starts before the GPS week epoch")
            }
            Self::NonFiniteHeaderStartMetadata { field } => {
                write!(f, "SP3 header start field {field} is not finite")
            }
            Self::InvalidHeaderStartMetadata { field, actual } => {
                write!(f, "SP3 header start field {field} is out of range: {actual}")
            }
            Self::HeaderStartMetadataMismatch {
                field,
                requested,
                actual,
            } => write!(
                f,
                "SP3 header start mismatch in {field}: requested {requested}, header has {actual}"
            ),
            Self::EmptyEpochGrid => write!(f, "SP3 product has no epoch records"),
            Self::FirstEpochMismatch {
                requested_j2000_s,
                actual_j2000_s,
            } => write!(
                f,
                "SP3 first epoch mismatch: requested {requested_j2000_s} J2000 s, parsed {actual_j2000_s} J2000 s"
            ),
            Self::IrregularEpochGrid {
                epoch_index,
                requested_s,
                actual_s,
            } => write!(
                f,
                "SP3 epoch grid is irregular at index {epoch_index}: requested step {requested_s} s, got {actual_s} s"
            ),
            Self::SpanNotMultipleOfCadence {
                span_s,
                cadence_s,
            } => write!(
                f,
                "exact SP3 span {span_s} s is not a multiple of cadence {cadence_s} s"
            ),
            Self::SpanMismatch {
                parsed,
                half_open,
                inclusive,
            } => write!(
                f,
                "SP3 span mismatch: parsed {parsed} epochs, expected {half_open} half-open or {inclusive} inclusive"
            ),
            Self::FormatVersionMismatch { requested, actual } => write!(
                f,
                "SP3 format-version mismatch: requested {requested:?}, parsed {actual:?}"
            ),
        }
    }
}

impl std::error::Error for ExactSp3ValidationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
            Self::Catalog(error) => Some(error),
            _ => None,
        }
    }
}

/// Parse and semantically validate bytes for one exact SP3 request.
///
/// The returned coverage value states which of the two supported boundary
/// representations was present.
pub fn parse_exact_sp3(
    bytes: &[u8],
    request: &ExactSp3Request,
) -> Result<(Sp3, ExactSp3Coverage), ExactSp3ValidationError> {
    let product = Sp3::parse(bytes).map_err(ExactSp3ValidationError::Parse)?;
    let coverage = validate_exact_sp3(&product, request)?;
    Ok((product, coverage))
}

/// Validate a parsed product against an exact date, span, and sample request.
///
/// Durations and permitted epoch counts are derived from the validated request
/// tokens. Header cadence is checked independently and never becomes the source
/// of truth for the allowed coverage.
pub fn validate_exact_sp3(
    product: &Sp3,
    request: &ExactSp3Request,
) -> Result<ExactSp3Coverage, ExactSp3ValidationError> {
    let cadence_s = parse_duration_token(&request.sample, DurationField::Sample)?;
    let span_s = parse_duration_token(&request.span, DurationField::Span)?;

    validate_mandatory_structure(product)?;
    if let Some(expected) = request.expected_agency.as_deref() {
        let actual = product.header.agency.trim();
        if actual != expected {
            return Err(ExactSp3ValidationError::AgencyMismatch {
                expected: expected.to_owned(),
                actual: actual.to_owned(),
            });
        }
    }

    let header_cadence_s = product.header.epoch_interval_s;
    if !header_cadence_s.is_finite() {
        return Err(ExactSp3ValidationError::NonFiniteHeaderCadence);
    }
    if header_cadence_s <= 0.0 {
        return Err(ExactSp3ValidationError::NonPositiveHeaderCadence {
            actual_s: header_cadence_s,
        });
    }
    if header_cadence_s >= SP3_MAX_EPOCH_INTERVAL_S {
        return Err(ExactSp3ValidationError::UnsupportedHeaderCadence {
            actual_s: header_cadence_s,
        });
    }
    if !seconds_match(header_cadence_s, cadence_s as f64) {
        return Err(ExactSp3ValidationError::CadenceMismatch {
            requested_s: cadence_s as f64,
            header_s: header_cadence_s,
        });
    }

    if product.declared_num_epochs != product.epoch_j2000_s.len() as u64 {
        return Err(ExactSp3ValidationError::DeclaredEpochCountMismatch {
            declared: product.declared_num_epochs,
            parsed: product.epoch_j2000_s.len(),
        });
    }

    let requested_start_j2000_s = requested_start_j2000_s(request);
    let declared_start_j2000_s = product
        .declared_start_j2000_s
        .ok_or(ExactSp3ValidationError::MissingDeclaredStart)?;
    if !seconds_match(declared_start_j2000_s, requested_start_j2000_s) {
        return Err(ExactSp3ValidationError::DeclaredStartMismatch {
            requested_j2000_s: requested_start_j2000_s,
            declared_j2000_s: declared_start_j2000_s,
        });
    }
    validate_line2_start_metadata(product, requested_start_j2000_s)?;

    let first_j2000_s = product
        .epoch_j2000_s
        .first()
        .copied()
        .ok_or(ExactSp3ValidationError::EmptyEpochGrid)?;
    if !seconds_match(first_j2000_s, requested_start_j2000_s) {
        return Err(ExactSp3ValidationError::FirstEpochMismatch {
            requested_j2000_s: requested_start_j2000_s,
            actual_j2000_s: first_j2000_s,
        });
    }

    for (index, pair) in product.epoch_j2000_s.windows(2).enumerate() {
        let actual_s = pair[1] - pair[0];
        if !actual_s.is_finite() || !seconds_match(actual_s, cadence_s as f64) {
            return Err(ExactSp3ValidationError::IrregularEpochGrid {
                epoch_index: index + 1,
                requested_s: cadence_s as f64,
                actual_s,
            });
        }
    }

    if span_s % cadence_s != 0 {
        return Err(ExactSp3ValidationError::SpanNotMultipleOfCadence { span_s, cadence_s });
    }
    let half_open = usize::try_from(span_s / cadence_s).unwrap_or(usize::MAX);
    let inclusive = half_open.saturating_add(1);
    match product.epoch_j2000_s.len() {
        count if count == half_open => Ok(ExactSp3Coverage::HalfOpen),
        count if count == inclusive => Ok(ExactSp3Coverage::Inclusive),
        parsed => Err(ExactSp3ValidationError::SpanMismatch {
            parsed,
            half_open,
            inclusive,
        }),
    }
    .and_then(|coverage| {
        validate_format_version(product.header.version, request.format_version.as_deref())?;
        Ok(coverage)
    })
}

fn validate_mandatory_structure(product: &Sp3) -> Result<(), ExactSp3ValidationError> {
    if let Some(malformed) = product.terminal_record.first_malformed_record {
        return Err(ExactSp3ValidationError::MalformedEofRecord {
            line_number: malformed.line_number,
            record_length: malformed.record_length,
        });
    }
    if !product.terminal_record.had_valid_record {
        return Err(ExactSp3ValidationError::MissingEof);
    }
    if product.terminal_record.had_trailing_content {
        return Err(ExactSp3ValidationError::TrailingContentAfterEof);
    }
    if product.satellite_header_lines < 5 {
        return Err(ExactSp3ValidationError::MandatoryHeaderRecordCount {
            record: "+",
            expected: 5,
            actual: product.satellite_header_lines,
        });
    }
    if product.accuracy_header_lines != product.satellite_header_lines {
        return Err(ExactSp3ValidationError::MandatoryHeaderRecordCount {
            record: "++",
            expected: product.satellite_header_lines,
            actual: product.accuracy_header_lines,
        });
    }
    for (record, actual) in [
        ("%c", product.time_system_header_lines),
        ("%f", product.float_header_lines),
        ("%i", product.integer_header_lines),
    ] {
        if actual != 2 {
            return Err(ExactSp3ValidationError::MandatoryHeaderRecordCount {
                record,
                expected: 2,
                actual,
            });
        }
    }
    if product.header_comment_lines < 4 {
        return Err(ExactSp3ValidationError::MandatoryHeaderRecordCount {
            record: "/*",
            expected: 4,
            actual: product.header_comment_lines,
        });
    }
    let declared_count = product
        .declared_satellite_count
        .ok_or(ExactSp3ValidationError::MissingDeclaredSatelliteCount)?;
    if declared_count != product.declared_satellite_tokens.len() {
        return Err(ExactSp3ValidationError::DeclaredSatelliteCountMismatch {
            declared: declared_count,
            tokens: product.declared_satellite_tokens.len(),
        });
    }
    for duplicate_index in 0..product.declared_satellite_tokens.len() {
        if let Some(first_index) = product.declared_satellite_tokens[..duplicate_index]
            .iter()
            .position(|token| token == &product.declared_satellite_tokens[duplicate_index])
        {
            return Err(ExactSp3ValidationError::DuplicateDeclaredSatellite {
                token: product.declared_satellite_tokens[duplicate_index].clone(),
                first_index,
                duplicate_index,
            });
        }
    }
    if product.header.satellites.is_empty() {
        return Err(ExactSp3ValidationError::NoDeclaredSatellites);
    }
    for epoch_index in 0..product.epochs.len() {
        let positions = product
            .epoch_position_tokens
            .get(epoch_index)
            .cloned()
            .unwrap_or_default();
        if positions != product.declared_satellite_tokens {
            return Err(ExactSp3ValidationError::SatelliteRecordSequenceMismatch {
                record: "P",
                epoch_index,
                expected: product.declared_satellite_tokens.clone(),
                actual: positions,
            });
        }
        let velocities = product
            .epoch_velocity_tokens
            .get(epoch_index)
            .cloned()
            .unwrap_or_default();
        let expected_velocities = match product.header.data_type {
            Sp3DataType::Position => Vec::new(),
            Sp3DataType::Velocity => product.declared_satellite_tokens.clone(),
        };
        if velocities != expected_velocities {
            return Err(ExactSp3ValidationError::SatelliteRecordSequenceMismatch {
                record: "V",
                epoch_index,
                expected: expected_velocities,
                actual: velocities,
            });
        }
        let mut expected_body = Vec::with_capacity(match product.header.data_type {
            Sp3DataType::Position => product.declared_satellite_tokens.len(),
            Sp3DataType::Velocity => product.declared_satellite_tokens.len() * 2,
        });
        for token in &product.declared_satellite_tokens {
            expected_body.push(format!("P{token}"));
            if matches!(product.header.data_type, Sp3DataType::Velocity) {
                expected_body.push(format!("V{token}"));
            }
        }
        let actual_body = product
            .epoch_state_record_sequence
            .get(epoch_index)
            .map(|records| {
                records
                    .iter()
                    .map(|(record, token)| format!("{record}{token}"))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if actual_body != expected_body {
            return Err(ExactSp3ValidationError::BodyRecordInterleavingMismatch {
                epoch_index,
                expected: expected_body,
                actual: actual_body,
            });
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum DurationField {
    Span,
    Sample,
}

fn parse_duration_token(token: &str, field: DurationField) -> Result<u64, ExactSp3ValidationError> {
    let bytes = token.as_bytes();
    let invalid = || match field {
        DurationField::Span => ExactSp3ValidationError::UnsupportedSpanToken {
            token: token.to_owned(),
        },
        DurationField::Sample => ExactSp3ValidationError::UnsupportedSampleToken {
            token: token.to_owned(),
        },
    };
    if bytes.len() != 3 || !bytes[0].is_ascii_digit() || !bytes[1].is_ascii_digit() {
        return Err(invalid());
    }
    let amount = u64::from(bytes[0] - b'0') * 10 + u64::from(bytes[1] - b'0');
    if amount == 0 {
        return Err(invalid());
    }
    let unit_s = match bytes[2] {
        b'S' => 1,
        b'M' => 60,
        b'H' => 3_600,
        b'D' => 86_400,
        b'W' if matches!(field, DurationField::Span) => 604_800,
        // Month/year duration depends on the start calendar and `U` has no
        // declared duration; neither is an exact fixed interval here.
        _ => return Err(invalid()),
    };
    let canonical = match bytes[2] {
        b'S' if amount % 60 == 0 => Some(format!("{:02}M", amount / 60)),
        b'M' if amount % 60 == 0 => Some(format!("{:02}H", amount / 60)),
        b'H' if amount % 24 == 0 => Some(format!("{:02}D", amount / 24)),
        _ => None,
    };
    if let Some(canonical) = canonical {
        return Err(match field {
            DurationField::Span => ExactSp3ValidationError::NonCanonicalSpanToken {
                token: token.to_owned(),
                canonical,
            },
            DurationField::Sample => ExactSp3ValidationError::NonCanonicalSampleToken {
                token: token.to_owned(),
                canonical,
            },
        });
    }
    amount.checked_mul(unit_s).ok_or_else(invalid)
}

fn parse_issue(issue: Option<&str>) -> Result<(u8, u8), ExactSp3ValidationError> {
    let Some(issue) = issue else {
        return Ok((0, 0));
    };
    let bytes = issue.as_bytes();
    if bytes.len() != 4 || !bytes.iter().all(u8::is_ascii_digit) {
        return Err(ExactSp3ValidationError::InvalidIssue {
            issue: issue.to_owned(),
        });
    }
    let hour = (bytes[0] - b'0') * 10 + (bytes[1] - b'0');
    let minute = (bytes[2] - b'0') * 10 + (bytes[3] - b'0');
    if hour > 23 || minute > 59 {
        return Err(ExactSp3ValidationError::InvalidIssue {
            issue: issue.to_owned(),
        });
    }
    Ok((hour, minute))
}

fn requested_start_j2000_s(request: &ExactSp3Request) -> f64 {
    let (hour, minute) = parse_issue(request.issue.as_deref())
        .expect("ExactSp3Request construction validates its issue token");
    let filename_epoch_j2000_s = j2000_seconds(
        request.date.year,
        i32::from(request.date.month),
        i32::from(request.date.day),
        i32::from(hour),
        i32::from(minute),
        0.0,
    );
    // The exact equality is unchanged. Catalog evidence determines the
    // required instant before the product bytes are parsed; no caller-facing
    // override can move it.
    filename_epoch_j2000_s + request.content_start_offset_s as f64
}

/// Cross-check SP3 line 2 against the trusted civil start.
///
/// SP3-d states that all time fields in a file use the `%c` time system, even
/// Gregorian and Modified Julian representations. Therefore this is coordinate
/// arithmetic in the product's declared time system; applying a GPS/UTC leap
/// conversion between line 1, line 2, and epoch records would be incorrect.
fn validate_line2_start_metadata(
    product: &Sp3,
    requested_start_j2000_s: f64,
) -> Result<(), ExactSp3ValidationError> {
    // ExactSp3Request starts on a whole minute, so this conversion is exact over
    // its supported civil-year range.
    let requested_start_j2000_s = requested_start_j2000_s as i64;
    let requested_mjd_total_s =
        requested_start_j2000_s + J2000_MJD_DAY * SECONDS_PER_DAY_I64 + SECONDS_PER_DAY_I64 / 2;
    let gps_zero_mjd_total_s = GPS_ZERO_MJD_DAY * SECONDS_PER_DAY_I64;
    let since_gps_zero_s = requested_mjd_total_s - gps_zero_mjd_total_s;
    if since_gps_zero_s < 0 {
        return Err(ExactSp3ValidationError::RequestBeforeGpsEpoch);
    }

    let requested_week = since_gps_zero_s.div_euclid(SECONDS_PER_WEEK_I64);
    let requested_sow_s = since_gps_zero_s.rem_euclid(SECONDS_PER_WEEK_I64) as f64;
    if i64::from(product.header.gnss_week) != requested_week {
        return Err(ExactSp3ValidationError::HeaderStartMetadataMismatch {
            field: "gps_week",
            requested: requested_week as f64,
            actual: f64::from(product.header.gnss_week),
        });
    }

    let header_sow_s = product.header.seconds_of_week;
    if !header_sow_s.is_finite() {
        return Err(ExactSp3ValidationError::NonFiniteHeaderStartMetadata {
            field: "seconds_of_week",
        });
    }
    if !(0.0..SECONDS_PER_WEEK_I64 as f64).contains(&header_sow_s) {
        return Err(ExactSp3ValidationError::InvalidHeaderStartMetadata {
            field: "seconds_of_week",
            actual: header_sow_s,
        });
    }
    if !seconds_match(header_sow_s, requested_sow_s) {
        return Err(ExactSp3ValidationError::HeaderStartMetadataMismatch {
            field: "seconds_of_week",
            requested: requested_sow_s,
            actual: header_sow_s,
        });
    }

    let header_mjd_fraction = product.header.mjd_fraction;
    if !header_mjd_fraction.is_finite() {
        return Err(ExactSp3ValidationError::NonFiniteHeaderStartMetadata {
            field: "mjd_fraction",
        });
    }
    if !(0.0..1.0).contains(&header_mjd_fraction) {
        return Err(ExactSp3ValidationError::InvalidHeaderStartMetadata {
            field: "mjd_fraction",
            actual: header_mjd_fraction,
        });
    }
    let header_mjd_total_s =
        i64::from(product.header.mjd) as f64 * 86_400.0 + header_mjd_fraction * 86_400.0;
    if !seconds_match(header_mjd_total_s, requested_mjd_total_s as f64) {
        return Err(ExactSp3ValidationError::HeaderStartMetadataMismatch {
            field: "mjd",
            requested: requested_mjd_total_s as f64 / 86_400.0,
            actual: header_mjd_total_s / 86_400.0,
        });
    }
    Ok(())
}

fn seconds_match(left: f64, right: f64) -> bool {
    (left - right).abs() <= WHOLE_SECOND_EPS_S
}

fn validate_format_version(
    actual: Sp3Version,
    requested: Option<&str>,
) -> Result<(), ExactSp3ValidationError> {
    let Some(requested) = requested else {
        return Ok(());
    };
    let actual = match actual {
        Sp3Version::A => "SP3-a",
        Sp3Version::B => "SP3-b",
        Sp3Version::C => "SP3-c",
        Sp3Version::D => "SP3-d",
    };
    if requested.eq_ignore_ascii_case(actual) {
        Ok(())
    } else {
        Err(ExactSp3ValidationError::FormatVersionMismatch {
            requested: requested.to_owned(),
            actual: actual.to_owned(),
        })
    }
}
