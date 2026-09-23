//! RINEX clock (`.CLK`) products: lossless reading, typed views, editing and
//! writing. [`RinexClock`] describes how the source lines and the derived
//! views relate.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::astro::time::model::{Instant, TimeScale};
use crate::validate::{self, FieldError};

mod derived;
mod epoch;
mod header;
mod numeric;
mod policy;
mod record;
#[cfg(test)]
mod tests;

pub use epoch::{civil_to_clock_instant, civil_to_gps_seconds};
pub use header::{
    ClockHeaderField, ClockHeaderReading, ClockHeaderRecord, ClockLayout, ClockTimeSystem,
    ClockTimeSystemStatus,
};
pub use policy::{ClockWriteDeparture, ClockWriteLeniency, ClockWritePolicy};
pub use record::{ClockRecord, ClockRecordReading, ClockRecordType, ClockSurplusValue};

use derived::{Derived, DerivedBuilder};
use epoch::{
    civil_second_policy_for_time_scale, civil_to_instant, epoch_cmp, gps_seconds_to_instant,
    interpolate, point_gps_seconds, sample_at_gps_seconds, validate_instant, Civil, EpochSource,
};
use header::{
    constructed_layout, identify_label, is_end_of_header, label_rank, read_header,
    render_constructed_header, render_time_system_line, HeaderContext, Label, TimeResolution,
};
use record::{
    is_potential_parent_record, read_continuation, read_parent, render_record, render_record_with,
    sigma_gap_of_line, validate_name, validate_values, EpochContext, SigmaGap, TypedEpoch,
    TypedRecord,
};

/// One satellite clock-bias sample.
///
/// A sample read from a record or built from GPS seconds also keeps the civil
/// tag or GPS seconds its epoch was built from, which fixes
/// [`ClockPoint::gps_seconds`] exactly; see there. Samples compare equal when
/// their epoch, bias and additional values are equal.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ClockPoint {
    /// Scale-tagged epoch from the RINEX clock file's declared time system.
    pub epoch: Instant,
    /// Satellite clock bias in seconds.
    pub bias_s: f64,
    /// Additional declared clock values following the bias, in standard order:
    /// bias sigma (s), clock rate (dimensionless), clock rate sigma
    /// (dimensionless), clock acceleration (s^-1), and clock acceleration sigma
    /// (s^-1). Values a record carries beyond its declared count are not
    /// included; [`ClockRecord::surplus_values`] reports them.
    pub additional_values: Vec<f64>,
    /// What `epoch` was built from, when known.
    source: EpochSource,
}

impl PartialEq for ClockPoint {
    fn eq(&self, other: &Self) -> bool {
        self.epoch == other.epoch
            && self.bias_s == other.bias_s
            && self.additional_values == other.additional_values
    }
}

impl ClockPoint {
    /// A sample at a scale-tagged instant, with the additional declared
    /// values after the bias in standard order.
    pub fn new(epoch: Instant, bias_s: f64, additional_values: Vec<f64>) -> Self {
        Self::with_source(epoch, bias_s, additional_values, EpochSource::Instant)
    }

    pub(crate) fn with_source(
        epoch: Instant,
        bias_s: f64,
        additional_values: Vec<f64>,
        source: EpochSource,
    ) -> Self {
        Self {
            epoch,
            bias_s,
            additional_values,
            source,
        }
    }

    /// This sample's epoch as GPS seconds, when the sample is on the GPST
    /// timeline. GPST and QZSST samples project (QZSST shares the TAI - 19 s
    /// alignment of GPST); every other scale returns `None`.
    ///
    /// A sample read from a record, or built from a civil tag through
    /// [`ClockRecord::new`], gives the `f64` nearest to the GPS second count
    /// its tag states, the value [`civil_to_gps_seconds`] gives for that tag.
    /// A sample built from GPS seconds ([`RinexClock::from_series_rows`])
    /// gives those GPS seconds. Either holds while `epoch` is the instant the
    /// sample was built with. A sample built from an instant alone
    /// ([`ClockPoint::new`], [`RinexClock::from_instant_series_rows`]), or
    /// whose `epoch` has been replaced, gives the GPS seconds of the civil tag
    /// with at most ten fractional second digits whose reading is `epoch`,
    /// when there is one, and otherwise the `f64` nearest to the exact time
    /// `epoch` holds.
    pub fn gps_seconds(&self) -> Option<f64> {
        point_gps_seconds(&self.epoch, self.source)
    }

    /// Validate that this clock point has a valid epoch instant, finite bias,
    /// at most 5 additional values, and all additional values finite.
    pub fn validate(&self) -> Result<(), RinexClockError> {
        validate_clock_point(self)
    }
}

/// Civil epoch tag used by RINEX clock records, interpreted in the file's time scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockEpoch {
    /// Four-digit calendar year.
    pub year: i32,
    /// Calendar month, 1..=12.
    pub month: u8,
    /// Calendar day of month, 1..=31.
    pub day: u8,
    /// Hour of day, 0..=23.
    pub hour: u8,
    /// Minute of hour, 0..=59.
    pub minute: u8,
    /// Seconds of minute, including fractional seconds. A UTC product accepts
    /// `60.x` on a day that ends with a positive leap second.
    pub second: f64,
}

/// A data record read from the source that is not part of the satellite
/// series (`AR`, `CR`, `DR` and `MS` records).
///
/// The record itself is retained; [`RinexClock::records`] returns it with its
/// typed values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RinexClockSkip {
    /// One-based input line number where the record appeared.
    pub line: usize,
    /// Two-letter RINEX clock record type identifier (e.g. `"AR"`, `"CR"`, `"DR"`, `"MS"`).
    pub record_type: String,
}

impl RinexClockSkip {
    /// Create a new skipped record report entry.
    pub fn new(line: usize, record_type: impl Into<String>) -> Self {
        Self {
            line,
            record_type: record_type.into(),
        }
    }
}

impl fmt::Display for RinexClockSkip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} record at line {} is outside the satellite series",
            self.record_type, self.line
        )
    }
}

/// A diagnostic recorded when lossy parsing keeps a line it cannot read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RinexClockDiagnostic {
    /// One-based line number of the malformed or invalid record.
    pub line: usize,
    /// The underlying parse error.
    pub error: RinexClockError,
}

impl RinexClockDiagnostic {
    /// Create a new parse diagnostic entry.
    pub fn new(line: usize, error: RinexClockError) -> Self {
        Self { line, error }
    }
}

impl fmt::Display for RinexClockDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse diagnostic at line {}: {}", self.line, self.error)
    }
}

/// A finding about how a product was read that does not stop it being read.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RinexClockNotice {
    /// No `TIME SYSTEM ID` record; the default in
    /// [`ClockTimeSystemStatus::Defaulted`] applies.
    TimeSystemDefaulted {
        /// The system applied.
        system: ClockTimeSystem,
    },
    /// A version 3.04 or later file has no `TIME SYSTEM ID` record, which its
    /// version requires. The 3.00 default was applied and reported with
    /// [`RinexClockNotice::TimeSystemDefaulted`].
    TimeSystemMissing,
    /// The time system has no core time scale (`IRN`). Record epochs keep their
    /// civil fields and have no instant.
    TimeSystemWithoutScale {
        /// The declared system.
        system: ClockTimeSystem,
    },
    /// A header record read at the other layout's columns or as
    /// whitespace-separated values.
    HeaderRecordNonconforming {
        /// One-based line number.
        line: usize,
    },
    /// A header record with a known label whose fields do not read.
    HeaderRecordUninterpreted {
        /// One-based line number.
        line: usize,
    },
    /// A header line whose label is not a RINEX clock header label.
    HeaderRecordUnknownLabel {
        /// One-based line number.
        line: usize,
    },
    /// Records carrying values beyond their declared count, such as a bias
    /// sigma on a record that declares one value.
    SurplusValues {
        /// Number of records.
        records: usize,
        /// One-based line number of the first.
        first_line: usize,
    },
    /// Records read at the columns of the layout the file does not declare.
    OtherLayoutRecords {
        /// Number of records.
        records: usize,
        /// One-based line number of the first.
        first_line: usize,
    },
    /// Records read as whitespace-separated values.
    WhitespaceRecords {
        /// Number of records.
        records: usize,
        /// One-based line number of the first.
        first_line: usize,
    },
}

impl fmt::Display for RinexClockNotice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimeSystemDefaulted { system } => write!(
                f,
                "no TIME SYSTEM ID record; epochs read in the default {} time system",
                system.label()
            ),
            Self::TimeSystemMissing => f.write_str(
                "version 3.04 or later file without the TIME SYSTEM ID record its version requires",
            ),
            Self::TimeSystemWithoutScale { system } => write!(
                f,
                "time system {} has no supported time scale; epochs have no instant",
                system.label()
            ),
            Self::HeaderRecordNonconforming { line } => write!(
                f,
                "header record at line {line} does not follow its version's columns"
            ),
            Self::HeaderRecordUninterpreted { line } => {
                write!(f, "header record at line {line} has fields that do not read")
            }
            Self::HeaderRecordUnknownLabel { line } => {
                write!(f, "header line {line} has no RINEX clock header label")
            }
            Self::SurplusValues {
                records,
                first_line,
            } => write!(
                f,
                "{records} records carry values beyond their declared count, first at line {first_line}"
            ),
            Self::OtherLayoutRecords {
                records,
                first_line,
            } => write!(
                f,
                "{records} records follow the other version's columns, first at line {first_line}"
            ),
            Self::WhitespaceRecords {
                records,
                first_line,
            } => write!(
                f,
                "{records} records read as whitespace-separated values, first at line {first_line}"
            ),
        }
    }
}

/// A RINEX clock product.
///
/// A product read from text keeps the text as its authority: every header
/// line with its exact label and payload, and every body line in order,
/// including blank lines, records of every type (`AR`, `AS`, `CR`, `DR`,
/// `MS`), continuation lines and, in a lossy read, lines that do not read as a
/// record. Header fields, data records, the per-satellite [`ClockPoint`]
/// series, skipped-record reports, diagnostics and notices are derived from
/// those lines and cannot be changed independently of them. Writing an
/// unedited product restates its input byte for byte, line terminators
/// included.
///
/// Edits go through typed setters ([`RinexClock::set_time_system`],
/// [`RinexClock::set_record_values`], [`RinexClock::insert_record`],
/// [`RinexClock::remove_record`], and the batch forms
/// [`RinexClock::retain_records`] and [`RinexClock::edit_records`]) that
/// validate the whole change first and then replace the affected lines. An edited or inserted record, and every
/// record of a product built from series rows, is held as typed values and
/// written in the product's column layout; the writer refuses by name a value
/// it cannot state exactly.
///
/// Records are read at the columns of the file's declared version (before
/// 3.04: the 80-column layout; from 3.04: the 85-column layout), then at the
/// other version's columns, then as whitespace-separated values, and each
/// record reports how it was read. The strict parser fails on the first line
/// it cannot read; [`RinexClock::parse_lossy`] keeps such lines verbatim with
/// a diagnostic and reads the rest.
///
/// Equality compares the retained source text, the ordered header and body
/// entries and the typed records; two products with the same satellite series
/// but different headers are not equal.
#[derive(Clone)]
pub struct RinexClock {
    source: String,
    line_starts: Vec<usize>,
    header: Vec<HeaderEntry>,
    body: Vec<BodyEntry>,
    /// File-order key of each body entry, strictly increasing; an inserted
    /// entry takes a key between its neighbours.
    order_keys: Vec<u64>,
    /// Body position of each record, in record order.
    record_positions: Vec<usize>,
    constructed: Option<TimeScale>,
    /// Blanks before the bias sigma in 3.04-layout records this product writes.
    sigma_gap: SigmaGap,
    context: HeaderContext,
    derived: Derived,
    diagnostics: Vec<RinexClockDiagnostic>,
    header_notices: Vec<RinexClockNotice>,
    notices: Vec<RinexClockNotice>,
}

impl PartialEq for RinexClock {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
            && self.header == other.header
            && self.body == other.body
            && self.constructed == other.constructed
    }
}

/// Spacing between consecutive order keys when keys are assigned afresh.
const ORDER_KEY_GAP: u64 = 1 << 20;

/// One header line: a source line by index, or a line written by an edit.
#[derive(Debug, Clone, PartialEq)]
enum HeaderEntry {
    Source(usize),
    Written(String),
}

/// One body entry, in file order.
#[derive(Debug, Clone, PartialEq)]
enum BodyEntry {
    /// A whitespace-only source line.
    Blank(usize),
    /// A record read from source lines `first..first + count`.
    Record { first: usize, count: usize },
    /// Source lines `first..first + count` that do not read as a record.
    Unparsed {
        first: usize,
        count: usize,
        diagnostic: Box<RinexClockDiagnostic>,
    },
    /// A record held as typed values.
    Typed(Box<TypedRecord>),
}

impl fmt::Debug for RinexClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RinexClock")
            .field("version", &self.context.version)
            .field("layout", &self.context.layout)
            .field("time_system", &self.context.time.system)
            .field("time_system_status", &self.context.time.status)
            .field("time_scale", &self.context.time.scale)
            .field("header_lines", &self.header.len())
            .field("body_entries", &self.body.len())
            .field("series_satellites", &self.derived.series().len())
            .field("skipped_records", &self.derived.skipped().len())
            .field("diagnostics", &self.diagnostics)
            .field("notices", &self.notices)
            .finish()
    }
}

/// RINEX clock parse error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RinexClockError {
    /// An `AS` satellite clock row is too short to carry the required bias.
    MalformedAsRecord {
        /// One-based input line number.
        line: usize,
        /// Human-readable parse failure.
        reason: &'static str,
        /// The full record text.
        record: String,
    },
    /// A declared continuation record was missing.
    MissingContinuation {
        /// One-based input line number of the parent record.
        line: usize,
        /// Record type of the parent record.
        record_type: String,
    },
    /// A continuation record was malformed or truncated.
    MalformedContinuation {
        /// One-based input line number of the continuation record.
        line: usize,
        /// Human-readable parse failure.
        reason: &'static str,
        /// The full record text.
        record: String,
    },
    /// A record or header field could not be parsed or was out of range.
    BadField {
        /// One-based input line number.
        line: usize,
        /// Field name.
        field: &'static str,
        /// Source field value.
        value: String,
    },
    /// Public manual input or query parameter was invalid.
    InvalidInput {
        /// Field name.
        field: &'static str,
        /// Human-readable validation failure.
        reason: &'static str,
    },
    /// The clock product names a time scale RINEX clock headers cannot represent.
    UnsupportedTimeScale {
        /// Unsupported time scale.
        scale: TimeScale,
    },
}

impl fmt::Display for RinexClockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RinexClockError::MalformedAsRecord {
                line,
                reason,
                record,
            } => write!(
                f,
                "malformed RINEX AS clock record at line {line}: {reason}: {record}"
            ),
            RinexClockError::MissingContinuation { line, record_type } => write!(
                f,
                "missing required continuation record for {record_type} at line {line}"
            ),
            RinexClockError::MalformedContinuation {
                line,
                reason,
                record,
            } => write!(
                f,
                "malformed RINEX clock continuation record at line {line}: {reason}: {record}"
            ),
            RinexClockError::BadField { line, field, value } => {
                write!(f, "bad RINEX clock field at line {line}: {field}={value}")
            }
            RinexClockError::InvalidInput { field, reason } => {
                write!(f, "invalid RINEX clock input {field}: {reason}")
            }
            RinexClockError::UnsupportedTimeScale { scale } => {
                write!(f, "unsupported RINEX clock time scale {}", scale.abbrev())
            }
        }
    }
}

impl std::error::Error for RinexClockError {}

impl RinexClock {
    /// Parse RINEX clock text, failing on the first line that does not read.
    ///
    /// Every line is retained; see [`RinexClock`]. Text without an
    /// `END OF HEADER` line has no header section, and every line is then read
    /// as a data line.
    pub fn parse(text: &str) -> Result<Self, RinexClockError> {
        match Self::read(text, true) {
            (clock, None) => Ok(clock),
            (_, Some(error)) => Err(error),
        }
    }

    /// Parse RINEX clock text, keeping lines that do not read verbatim with a
    /// diagnostic.
    ///
    /// Nothing is dropped: [`RinexClock::to_rinex_string`] on the result
    /// restates the input exactly. An unrecognised `TIME SYSTEM ID` leaves
    /// the time system unresolved rather than assuming one.
    pub fn parse_lossy(text: &str) -> Self {
        Self::read(text, false).0
    }

    /// Rebuild a GPST product from the legacy public GPS-second rows.
    ///
    /// GPS seconds outside the civil years 1 through 9999 are refused with
    /// [`RinexClockError::InvalidInput`].
    pub fn from_series_rows(rows: Vec<(String, Vec<(f64, f64)>)>) -> Result<Self, RinexClockError> {
        let rows = rows
            .into_iter()
            .map(|(sat, points)| {
                validate::require_strictly_increasing(
                    points.iter().map(|&(gps_seconds, _)| gps_seconds),
                    "gps_seconds",
                )
                .map_err(map_manual_order_error)?;
                let points = points
                    .into_iter()
                    .map(|(gps_seconds, bias_s)| {
                        validate_finite(bias_s, "bias_s")?;
                        Ok(ClockPoint::with_source(
                            gps_seconds_to_instant(gps_seconds)?,
                            bias_s,
                            Vec::new(),
                            EpochSource::GpsSeconds(gps_seconds),
                        ))
                    })
                    .collect::<Result<Vec<_>, RinexClockError>>()?;
                Ok((sat, points))
            })
            .collect::<Result<Vec<_>, RinexClockError>>()?;
        Self::from_clock_points(TimeScale::Gpst, rows)
    }

    /// Build a product from scale-tagged instant rows.
    pub fn from_instant_series_rows(
        time_scale: TimeScale,
        rows: Vec<(String, Vec<(Instant, f64)>)>,
    ) -> Result<Self, RinexClockError> {
        Self::from_clock_points(
            time_scale,
            rows.into_iter()
                .map(|(sat, points)| {
                    let points = points
                        .into_iter()
                        .map(|(epoch, bias_s)| ClockPoint::new(epoch, bias_s, Vec::new()))
                        .collect();
                    (sat, points)
                })
                .collect(),
        )
    }

    /// Build a product from per-satellite clock points, keeping every declared
    /// value of each point.
    ///
    /// Each satellite's points must be strictly increasing in time. A
    /// satellite named in more than one row keeps the points of every row.
    /// The records are held as typed values; [`RinexClock::to_rinex_string`]
    /// writes them in the layout the time scale needs (3.00 for GPST, GST, UTC
    /// and TAI; 3.04 for QZSST and BDT), refuses a scale no RINEX clock time
    /// system names (GLONASS system time among them, since `GLO` names UTC
    /// hours), and refuses by name a value or epoch it cannot state exactly.
    pub fn from_clock_points(
        time_scale: TimeScale,
        rows: Vec<(String, Vec<ClockPoint>)>,
    ) -> Result<Self, RinexClockError> {
        let mut by_satellite = BTreeMap::<String, Vec<ClockPoint>>::new();
        for (sat, points) in rows {
            let indexed = points
                .into_iter()
                .enumerate()
                .map(|(idx, point)| {
                    validate_clock_point(&point)?;
                    Ok((point, idx))
                })
                .collect::<Result<Vec<_>, RinexClockError>>()?;
            validate_instant_series_order(&indexed)?;
            by_satellite
                .entry(sat)
                .or_default()
                .extend(indexed.into_iter().map(|(point, _)| point));
        }
        let body: Vec<BodyEntry> = by_satellite
            .into_iter()
            .flat_map(|(sat, points)| {
                points.into_iter().map(move |point| {
                    BodyEntry::Typed(Box::new(TypedRecord {
                        record_type: ClockRecordType::As,
                        name: sat.clone(),
                        epoch: TypedEpoch::Instant {
                            instant: point.epoch,
                            source: point.source,
                        },
                        values: std::iter::once(point.bias_s)
                            .chain(point.additional_values)
                            .collect(),
                    }))
                })
            })
            .collect();
        let entries = body.len();
        let mut clock = Self {
            source: String::new(),
            line_starts: Vec::new(),
            header: Vec::new(),
            body,
            order_keys: fresh_order_keys(entries),
            record_positions: (0..entries).collect(),
            constructed: Some(time_scale),
            sigma_gap: SigmaGap::One,
            context: HeaderContext::constructed(time_scale),
            derived: Derived::default(),
            diagnostics: Vec::new(),
            header_notices: Vec::new(),
            notices: Vec::new(),
        };
        clock.rebuild();
        Ok(clock)
    }

    /// Declared format version; for a product built from rows, the version it
    /// is written in.
    pub fn version(&self) -> Option<f64> {
        self.context.version
    }

    /// Column layout records are read and written in. `None` when the text
    /// declares no version (records are then read at the 3.00 columns first
    /// and written in them) or a built product's time scale has no RINEX
    /// clock time system.
    pub fn layout(&self) -> Option<ClockLayout> {
        self.context.layout
    }

    /// Satellite system code of the `RINEX VERSION / TYPE` record (`G`, `R`,
    /// `E`, `C`, `I`, `J`, `S` or `M`), when one is written.
    pub fn satellite_system(&self) -> Option<char> {
        self.context.satellite_system
    }

    /// The product's time system, when one is declared, defaulted or built in.
    pub fn time_system(&self) -> Option<ClockTimeSystem> {
        self.context.time.system
    }

    /// How the time system was established.
    pub fn time_system_status(&self) -> &ClockTimeSystemStatus {
        &self.context.time.status
    }

    /// The time scale record epochs are interpreted in; `None` when the time
    /// system is missing, unrecognised, conflicting or has no core scale.
    pub fn time_scale(&self) -> Option<TimeScale> {
        self.context.time.scale
    }

    /// Every header line in order with its typed reading. A product built
    /// from rows has no header lines; its header is written from its time
    /// scale.
    pub fn header_records(&self) -> Vec<ClockHeaderRecord> {
        read_header(&self.header_lines()).0
    }

    /// Every data record in order, including duplicate records for one name
    /// and epoch (RINEX clock section 4 uses two `AR` records at one epoch to
    /// state a discontinuity). Lines a lossy read could not read are not
    /// records; [`RinexClock::diagnostics`] names them.
    pub fn records(&self) -> impl Iterator<Item = ClockRecord> + '_ {
        let ctx = epoch_context(&self.context.time);
        let layout = self.context.layout;
        self.body.iter().filter_map(move |entry| match entry {
            BodyEntry::Record { first, .. } => {
                read_record_at(&self.source, &self.line_starts, *first, layout, &ctx)
                    .ok()
                    .map(|(record, _)| record)
            }
            BodyEntry::Typed(record) => Some(record.view(&ctx)),
            BodyEntry::Blank(_) | BodyEntry::Unparsed { .. } => None,
        })
    }

    /// Number of data records.
    pub fn record_count(&self) -> usize {
        self.record_positions.len()
    }

    /// One line of the text the product was read from, by one-based line
    /// number, without its terminator.
    pub fn source_line(&self, line: usize) -> Option<&str> {
        let index = line.checked_sub(1)?;
        (index < self.line_starts.len())
            .then(|| line_content(&self.source, &self.line_starts, index))
    }

    /// Per-satellite clock-bias series derived from the `AS` records whose
    /// epoch resolves to an instant, each strictly time-ordered. Where records
    /// repeat one satellite and instant, the last in file order is the sample;
    /// every such record remains in [`RinexClock::records`].
    pub fn series(&self) -> &BTreeMap<String, Vec<ClockPoint>> {
        self.derived.series()
    }

    /// Records read from the source that are not in the satellite series.
    pub fn skipped_records(&self) -> &[RinexClockSkip] {
        self.derived.skipped()
    }

    /// Lines a lossy read kept without reading them as records, and header
    /// time-system errors.
    pub fn diagnostics(&self) -> &[RinexClockDiagnostic] {
        &self.diagnostics
    }

    /// Findings about how the product was read that do not stop it being read.
    pub fn notices(&self) -> &[RinexClockNotice] {
        &self.notices
    }

    /// Export GPST and QZSST samples as `[(satellite, [(gps_seconds, bias_s), ...]), ...]`.
    ///
    /// Samples on other time scales are not coerced into GPS seconds and are
    /// omitted.
    pub fn series_rows(&self) -> Vec<(String, Vec<(f64, f64)>)> {
        self.series()
            .iter()
            .map(|(sat, points)| {
                (
                    sat.clone(),
                    points
                        .iter()
                        .filter_map(|point| Some((point.gps_seconds()?, point.bias_s)))
                        .collect(),
                )
            })
            .collect()
    }

    /// Export the product as scale-tagged instant rows.
    pub fn instant_series_rows(&self) -> Vec<(String, Vec<(Instant, f64)>)> {
        self.series()
            .iter()
            .map(|(sat, points)| {
                (
                    sat.clone(),
                    points
                        .iter()
                        .map(|point| (point.epoch, point.bias_s))
                        .collect(),
                )
            })
            .collect()
    }

    /// Interpolate one satellite clock bias at a civil epoch in this file's
    /// scale. On a UTC product a `23:59:60.x` label on a leap-second day is a
    /// valid query, and interpolation across a leap second uses elapsed time.
    pub fn clock_s(
        &self,
        satellite_id: &str,
        epoch: ClockEpoch,
    ) -> Result<Option<f64>, RinexClockError> {
        let scale = self.time_scale().ok_or_else(|| {
            invalid_input(
                "time_system",
                "the product's time system does not resolve to a time scale",
            )
        })?;
        let epoch = civil_to_clock_instant(
            scale,
            epoch.year,
            epoch.month,
            epoch.day,
            epoch.hour,
            epoch.minute,
            epoch.second,
        )
        .ok_or_else(|| invalid_input("epoch", "invalid civil clock epoch"))?;
        self.clock_s_at_instant(satellite_id, epoch)
    }

    /// Interpolate one satellite clock bias at a scale-tagged instant.
    pub fn clock_s_at_instant(
        &self,
        satellite_id: &str,
        epoch: Instant,
    ) -> Result<Option<f64>, RinexClockError> {
        validate_instant(epoch, "epoch")?;
        let Some(records) = self.series().get(satellite_id) else {
            return Ok(None);
        };
        Ok(interpolate(records, epoch))
    }

    /// Interpolate one satellite clock bias at GPS seconds. GPST and QZSST
    /// series answer; GPS seconds outside the civil years 1 through 9999 are
    /// refused with [`RinexClockError::InvalidInput`].
    ///
    /// A sample whose GPS seconds ([`ClockPoint::gps_seconds`], as
    /// [`RinexClock::series_rows`] exports them) equal the query answers with
    /// its own bias, so every exported row is answered at its own GPS seconds.
    /// GPS seconds resolve time more coarsely than a sample's epoch (about
    /// 0.24 microseconds in 2026), so the query's instant can fall just beside
    /// that sample; when two samples share the query's GPS seconds, the bias
    /// is interpolated at the query's instant.
    pub fn clock_s_at_gps_seconds(
        &self,
        satellite_id: &str,
        gps_seconds: f64,
    ) -> Result<Option<f64>, RinexClockError> {
        let epoch = gps_seconds_to_instant(gps_seconds)?;
        if let Some(bias_s) = self
            .series()
            .get(satellite_id)
            .and_then(|records| sample_at_gps_seconds(records, &epoch, gps_seconds))
        {
            return Ok(Some(bias_s));
        }
        self.clock_s_at_instant(satellite_id, epoch)
    }

    /// Write the product as RINEX clock text.
    ///
    /// A product read from text restates every retained line byte for byte,
    /// including header records, blank lines, records of every type and, after
    /// a lossy read, lines that did not read. Records held as typed values are
    /// written in the product's layout (3.00 columns when the text declared no
    /// version) with the product's line terminator. Values are written in
    /// 19-column `E19.12` fields only when they read back to the same bits;
    /// otherwise the value is refused by name with
    /// [`RinexClockError::InvalidInput`]. An epoch is written only when its
    /// microsecond text states it exactly: an edited record restates its source
    /// seconds text, and an instant is written when it is, bit for bit, the
    /// split the reader, the GPS-seconds constructor (from the correctly
    /// rounded double of the GPS seconds the text states, which is what
    /// `series_rows` exports for a record read from that text) or the
    /// whole-J2000-second constructor builds from that text; any other epoch
    /// is refused rather than rounded
    /// ([`RinexClock::to_rinex_string_with_policy`] can allow the nearest
    /// microsecond and report it). A product built from rows is written
    /// with a header stating its version, satellite system, time system and
    /// data types; a time scale no RINEX clock time system names is refused
    /// with [`RinexClockError::UnsupportedTimeScale`].
    pub fn to_rinex_string(&self) -> Result<String, RinexClockError> {
        self.to_rinex_string_with_policy(ClockWritePolicy::strict())
            .map(|(text, _)| text)
    }

    /// Write the product under `policy`, with the departures from what the
    /// product states that the policy allowed and the writer emitted.
    ///
    /// [`ClockWritePolicy::default`] allows none, which is what
    /// [`RinexClock::to_rinex_string`] does. With
    /// [`ClockWritePolicy::nearest_microsecond_epochs`] allowed, an epoch no
    /// microsecond text states exactly is written as the nearest one and
    /// reported as [`ClockWriteDeparture::EpochAtNearestMicrosecond`] with the
    /// record's index, name, epoch and the epoch text written.
    pub fn to_rinex_string_with_policy(
        &self,
        policy: ClockWritePolicy,
    ) -> Result<(String, Vec<ClockWriteDeparture>), RinexClockError> {
        if let Some(scale) = self.constructed {
            return self.write_constructed(scale, policy);
        }
        let layout = self.context.layout.unwrap_or(ClockLayout::V300);
        let scale = self.context.time.scale;
        let ctx = epoch_context(&self.context.time);
        let eol = self.line_ending();
        let mut out = String::with_capacity(self.source.len() + 128);
        let mut departures = Vec::new();
        let mut open = false;
        for entry in &self.header {
            match entry {
                HeaderEntry::Source(index) => self.push_source(&mut out, &mut open, *index, eol),
                HeaderEntry::Written(text) => push_written(&mut out, &mut open, text, eol),
            }
        }
        let mut record_index = 0;
        for entry in &self.body {
            match entry {
                BodyEntry::Blank(index) => self.push_source(&mut out, &mut open, *index, eol),
                BodyEntry::Record { first, count } | BodyEntry::Unparsed { first, count, .. } => {
                    for index in *first..*first + *count {
                        self.push_source(&mut out, &mut open, index, eol);
                    }
                }
                BodyEntry::Typed(record) => {
                    let (lines, rounded) = render_record_with(
                        record,
                        layout,
                        scale,
                        self.sigma_gap,
                        policy.nearest_microsecond_epochs,
                    )?;
                    if rounded {
                        departures.push(epoch_departure(
                            record_index,
                            record,
                            &ctx,
                            layout,
                            &lines,
                        ));
                    }
                    for line in lines {
                        push_written(&mut out, &mut open, &line, eol);
                    }
                }
            }
            if matches!(entry, BodyEntry::Record { .. } | BodyEntry::Typed(_)) {
                record_index += 1;
            }
        }
        Ok((out, departures))
    }

    /// Declare the product's time system.
    ///
    /// Replaces every `TIME SYSTEM ID` record with one written at the `3X,A3`
    /// columns of the product's layout, or inserts one before the first header
    /// record that Table A15 orders after it. Every record epoch is checked in
    /// the new time system first; if one does not convert (a `23:59:60` label
    /// in a continuous scale), nothing changes and the error is returned. A
    /// product with no header section, or built from rows, is refused.
    pub fn set_time_system(&mut self, system: ClockTimeSystem) -> Result<(), RinexClockError> {
        if self.constructed.is_some() {
            return Err(invalid_input(
                "time_system",
                "a product built from series rows states its time scale on every epoch",
            ));
        }
        if self.header.is_empty() {
            return Err(invalid_input(
                "time_system",
                "the product has no header section to declare a time system in",
            ));
        }
        let layout = self.context.layout.unwrap_or(ClockLayout::V300);
        let written = render_time_system_line(system, layout);
        let mut header = Vec::with_capacity(self.header.len() + 1);
        let mut placed = false;
        for entry in &self.header {
            let content = header_entry_content(&self.source, &self.line_starts, entry);
            let label = identify_label(content).map(|(label, _)| label);
            if label == Some(Label::TimeSystem) {
                if !placed {
                    header.push(HeaderEntry::Written(written.clone()));
                    placed = true;
                }
                continue;
            }
            if !placed
                && label.is_some_and(|label| label_rank(label) > label_rank(Label::TimeSystem))
            {
                header.push(HeaderEntry::Written(written.clone()));
                placed = true;
            }
            header.push(entry.clone());
        }
        if !placed {
            let at = header.len().saturating_sub(1);
            header.insert(at, HeaderEntry::Written(written));
        }

        let lines: Vec<(Option<usize>, &str)> = header
            .iter()
            .map(|entry| {
                (
                    header_entry_line(entry),
                    header_entry_content(&self.source, &self.line_starts, entry),
                )
            })
            .collect();
        let (_, context, diagnostics, _) = read_header(&lines);
        if let Some(diagnostic) = diagnostics.into_iter().next() {
            return Err(diagnostic.error);
        }
        let ctx = epoch_context(&context.time);
        for entry in &self.body {
            match entry {
                BodyEntry::Record { first, .. } => {
                    read_record_at(
                        &self.source,
                        &self.line_starts,
                        *first,
                        context.layout,
                        &ctx,
                    )
                    .map_err(|(diagnostic, _)| diagnostic.error)?;
                }
                BodyEntry::Typed(record) => {
                    if let TypedEpoch::Civil { civil, .. } = &record.epoch {
                        check_civil_in_context(*civil, &ctx)?;
                    }
                }
                BodyEntry::Blank(_) | BodyEntry::Unparsed { .. } => {}
            }
        }
        drop(lines);
        self.header = header;
        self.rebuild();
        Ok(())
    }

    /// Replace the declared values of the record at `index` (in
    /// [`RinexClock::records`] order), bias first.
    ///
    /// The record keeps its type, name and epoch, including the exact text of
    /// its seconds field, and is then held as typed values written in the
    /// product's layout. The edit is refused, and nothing changes, when the
    /// record could not then be written (a value no 19-column field states
    /// exactly, a name or year the layout cannot hold, an epoch the seconds
    /// field cannot state), or when the source record carries values beyond
    /// its declared count that the new value list does not restate: those
    /// values are never dropped silently.
    pub fn set_record_values(
        &mut self,
        index: usize,
        values: Vec<f64>,
    ) -> Result<(), RinexClockError> {
        let position = self.record_position(index)?;
        let current = self.view_at(position)?;
        let typed = self.edited_record(position, current.clone(), values)?;
        self.detach(position, current)?;
        self.body[position] = BodyEntry::Typed(Box::new(typed));
        self.attach(position)?;
        self.refresh_notices();
        Ok(())
    }

    /// Insert a record before the record at `index` (in
    /// [`RinexClock::records`] order), or after the last record when `index`
    /// equals [`RinexClock::record_count`].
    ///
    /// The record must be writable in the product's layout (name width, year,
    /// values stated exactly in 19-column fields, an epoch the seconds field
    /// can state) and its epoch valid in the product's time scale. A record
    /// carrying surplus values is refused: a written record states only its
    /// declared values. Appending records one by one takes time linear in
    /// their number.
    pub fn insert_record(
        &mut self,
        index: usize,
        record: ClockRecord,
    ) -> Result<(), RinexClockError> {
        if !record.surplus.is_empty() {
            return Err(invalid_input(
                "surplus_values",
                "a written record states only its declared values",
            ));
        }
        validate_values(&record.values)?;
        let count = self.record_count();
        if index > count {
            return Err(invalid_input("index", "past the end of the records"));
        }
        let name = match (&record.satellite, record.record_type) {
            (Some(satellite), ClockRecordType::As) => satellite.clone(),
            (None, ClockRecordType::As) => {
                return Err(invalid_input(
                    "satellite",
                    "not a RINEX satellite identifier",
                ));
            }
            _ => record.name.clone(),
        };
        if let Some(layout) = self.writing_layout() {
            let name_field = if record.record_type == ClockRecordType::As {
                "satellite"
            } else {
                "name"
            };
            validate_name(&name, name_field, layout)?;
        }
        let ctx = epoch_context(&self.context.time);
        check_civil_in_context(record.civil, &ctx)?;
        let epoch = match self.constructed {
            Some(scale) => TypedEpoch::Instant {
                instant: civil_to_instant(scale, record.civil)
                    .map_err(|_| invalid_input("epoch", "invalid civil clock epoch"))?,
                source: EpochSource::Civil(record.civil),
            },
            None => TypedEpoch::Civil {
                civil: record.civil,
                second_text: record.second_text.clone(),
            },
        };
        let typed = TypedRecord {
            record_type: record.record_type,
            name,
            epoch,
            values: record.values,
        };
        self.check_writable(&typed)?;
        let position = if index == count {
            self.body.len()
        } else {
            self.record_position(index)?
        };
        let key = self.order_key_before(position);
        self.body
            .insert(position, BodyEntry::Typed(Box::new(typed)));
        self.order_keys.insert(position, key);
        for later in &mut self.record_positions[index..] {
            *later += 1;
        }
        self.record_positions.insert(index, position);
        self.attach(position)?;
        self.refresh_notices();
        Ok(())
    }

    /// Keep the records `keep` accepts and remove every other, with every line
    /// it spans, in one pass; returns the number removed. Blank and unread
    /// lines stay. The derived views are rebuilt once, so removing many
    /// records, such as every `AR` record of a product, takes time linear in
    /// the product's size apart from one sort of the satellite series.
    pub fn retain_records(&mut self, mut keep: impl FnMut(&ClockRecord) -> bool) -> usize {
        let ctx = epoch_context(&self.context.time);
        let mut kept_body = Vec::with_capacity(self.body.len());
        let mut kept_keys = Vec::with_capacity(self.order_keys.len());
        let mut record_positions = Vec::with_capacity(self.record_positions.len());
        let mut removed = 0;
        let body = std::mem::take(&mut self.body);
        let keys = std::mem::take(&mut self.order_keys);
        for (entry, key) in body.into_iter().zip(keys) {
            let view = match &entry {
                BodyEntry::Record { first, .. } => read_record_at(
                    &self.source,
                    &self.line_starts,
                    *first,
                    self.context.layout,
                    &ctx,
                )
                .ok()
                .map(|(record, _)| record),
                BodyEntry::Typed(record) => Some(record.view(&ctx)),
                BodyEntry::Blank(_) | BodyEntry::Unparsed { .. } => None,
            };
            // A record entry always reads back; were one not to, it is kept
            // rather than removed unseen.
            if let Some(record) = &view {
                if !keep(record) {
                    removed += 1;
                    continue;
                }
            }
            if matches!(entry, BodyEntry::Record { .. } | BodyEntry::Typed(_)) {
                record_positions.push(kept_body.len());
            }
            kept_body.push(entry);
            kept_keys.push(key);
        }
        self.body = kept_body;
        self.order_keys = kept_keys;
        self.record_positions = record_positions;
        self.rebuild();
        removed
    }

    /// Replace the declared values of every record for which `edit` returns
    /// new values, bias first, in one pass; returns the number edited.
    ///
    /// Each edit follows [`RinexClock::set_record_values`], and the whole batch
    /// is checked before anything changes: if one edit is refused, none is
    /// applied and its error is returned. The derived views are rebuilt once.
    pub fn edit_records(
        &mut self,
        mut edit: impl FnMut(&ClockRecord) -> Option<Vec<f64>>,
    ) -> Result<usize, RinexClockError> {
        let ctx = epoch_context(&self.context.time);
        let mut replacements = Vec::new();
        for &position in &self.record_positions {
            let current = match &self.body[position] {
                BodyEntry::Record { first, .. } => read_record_at(
                    &self.source,
                    &self.line_starts,
                    *first,
                    self.context.layout,
                    &ctx,
                )
                .map(|(record, _)| record)
                .map_err(|(diagnostic, _)| diagnostic.error)?,
                BodyEntry::Typed(record) => record.view(&ctx),
                BodyEntry::Blank(_) | BodyEntry::Unparsed { .. } => continue,
            };
            let Some(values) = edit(&current) else {
                continue;
            };
            let typed = self.edited_record(position, current, values)?;
            replacements.push((position, typed));
        }
        let edited = replacements.len();
        for (position, typed) in replacements {
            self.body[position] = BodyEntry::Typed(Box::new(typed));
        }
        if edited > 0 {
            self.rebuild();
        }
        Ok(edited)
    }

    /// Remove the record at `index` (in [`RinexClock::records`] order) with
    /// every line it spans, returning it.
    pub fn remove_record(&mut self, index: usize) -> Result<ClockRecord, RinexClockError> {
        let position = self.record_position(index)?;
        let record = self.view_at(position)?;
        self.detach(position, record.clone())?;
        self.body.remove(position);
        self.order_keys.remove(position);
        self.record_positions.remove(index);
        for later in &mut self.record_positions[index..] {
            *later -= 1;
        }
        self.refresh_notices();
        Ok(record)
    }

    fn read(text: &str, strict: bool) -> (Self, Option<RinexClockError>) {
        let line_starts = split_lines(text);
        let line_total = line_starts.len();
        let header_len = (0..line_total)
            .find(|&index| is_end_of_header(line_content(text, &line_starts, index)))
            .map_or(0, |index| index + 1);
        let header_lines: Vec<(Option<usize>, &str)> = (0..header_len)
            .map(|index| (Some(index + 1), line_content(text, &line_starts, index)))
            .collect();
        let (_, context, header_diagnostics, header_notices) = read_header(&header_lines);
        let ctx = epoch_context(&context.time);

        let mut first_error = None;
        if strict {
            first_error = header_diagnostics
                .first()
                .map(|diagnostic| diagnostic.error.clone());
        }
        let mut diagnostics = header_diagnostics;
        let mut body = Vec::new();
        let mut builder = DerivedBuilder::new(context.layout);
        let mut record_positions = Vec::new();
        let mut sigma_gap = None;
        let mut index = header_len;
        while first_error.is_none() && index < line_total {
            let key = order_key_at(body.len());
            if line_content(text, &line_starts, index).trim().is_empty() {
                body.push(BodyEntry::Blank(index));
                index += 1;
                continue;
            }
            match read_record_at(text, &line_starts, index, context.layout, &ctx) {
                Ok((record, used)) => {
                    let carries_sigma = record.values.len() >= 2
                        || record.surplus.iter().any(|value| value.position == 1);
                    if sigma_gap.is_none()
                        && carries_sigma
                        && record.reading == ClockRecordReading::Columns(ClockLayout::V304)
                    {
                        sigma_gap = sigma_gap_of_line(line_content(text, &line_starts, index));
                    }
                    record_positions.push(body.len());
                    builder.add(&record, key);
                    body.push(BodyEntry::Record {
                        first: index,
                        count: used,
                    });
                    index += used;
                }
                Err((diagnostic, used)) => {
                    if strict {
                        first_error = Some(diagnostic.error);
                        break;
                    }
                    diagnostics.push(diagnostic.clone());
                    body.push(BodyEntry::Unparsed {
                        first: index,
                        count: used,
                        diagnostic: Box::new(diagnostic),
                    });
                    index += used;
                }
            }
        }
        let derived = builder.finish();
        let mut notices = header_notices.clone();
        notices.extend(derived.notices());
        let entries = body.len();
        let clock = Self {
            source: text.to_string(),
            line_starts,
            header: (0..header_len).map(HeaderEntry::Source).collect(),
            body,
            order_keys: fresh_order_keys(entries),
            record_positions,
            constructed: None,
            sigma_gap: sigma_gap.unwrap_or(SigmaGap::One),
            context,
            derived,
            diagnostics,
            header_notices,
            notices,
        };
        (clock, first_error)
    }

    /// Recompute the header context and every derived view from the entries.
    fn rebuild(&mut self) {
        let (context, header_diagnostics, header_notices) = match self.constructed {
            Some(scale) => (HeaderContext::constructed(scale), Vec::new(), Vec::new()),
            None => {
                let (_, context, diagnostics, notices) = read_header(&self.header_lines());
                (context, diagnostics, notices)
            }
        };
        let ctx = epoch_context(&context.time);
        let mut diagnostics = header_diagnostics;
        let mut builder = DerivedBuilder::new(context.layout);
        for (entry, &key) in self.body.iter().zip(&self.order_keys) {
            match entry {
                BodyEntry::Blank(_) => {}
                BodyEntry::Record { first, .. } => {
                    if let Ok((record, _)) = read_record_at(
                        &self.source,
                        &self.line_starts,
                        *first,
                        context.layout,
                        &ctx,
                    ) {
                        builder.add(&record, key);
                    }
                }
                BodyEntry::Unparsed { diagnostic, .. } => diagnostics.push((**diagnostic).clone()),
                BodyEntry::Typed(record) => builder.add(&record.view(&ctx), key),
            }
        }
        self.derived = builder.finish();
        self.context = context;
        self.diagnostics = diagnostics;
        self.header_notices = header_notices;
        self.refresh_notices();
    }

    /// The typed view of the record at a body position.
    fn view_at(&self, position: usize) -> Result<ClockRecord, RinexClockError> {
        let ctx = epoch_context(&self.context.time);
        match self.body.get(position) {
            Some(BodyEntry::Record { first, .. }) => read_record_at(
                &self.source,
                &self.line_starts,
                *first,
                self.context.layout,
                &ctx,
            )
            .map(|(record, _)| record)
            .map_err(|(diagnostic, _)| diagnostic.error),
            Some(BodyEntry::Typed(record)) => Ok(record.view(&ctx)),
            _ => Err(invalid_input("index", "no record at this index")),
        }
    }

    /// Add the record at a body position to the derived views.
    fn attach(&mut self, position: usize) -> Result<(), RinexClockError> {
        let record = self.view_at(position)?;
        let key = self.order_keys[position];
        self.derived.insert(&record, key, self.context.layout);
        Ok(())
    }

    /// Remove `record`, the record at a body position, from the derived views.
    fn detach(&mut self, position: usize, record: ClockRecord) -> Result<(), RinexClockError> {
        let key = self.order_keys[position];
        let replacement = match self.derived.removal_replacement(&record, key) {
            Some(replacement_key) => {
                let replacement_position = self
                    .order_keys
                    .binary_search(&replacement_key)
                    .map_err(|_| invalid_input("index", "no record at this index"))?;
                self.view_at(replacement_position)?.clock_point()
            }
            None => None,
        };
        self.derived
            .remove(&record, key, self.context.layout, replacement);
        Ok(())
    }

    fn refresh_notices(&mut self) {
        self.notices = self.header_notices.clone();
        self.notices.extend(self.derived.notices());
    }

    /// An order key for an entry inserted at a body position, between the keys
    /// of its neighbours. Keys are reassigned when two neighbours leave no room.
    fn order_key_before(&mut self, position: usize) -> u64 {
        let bounds = |keys: &[u64]| {
            let low = position
                .checked_sub(1)
                .and_then(|previous| keys.get(previous))
                .copied()
                .unwrap_or(0);
            let high = keys
                .get(position)
                .copied()
                .unwrap_or_else(|| low.saturating_add(2 * ORDER_KEY_GAP));
            (low, high)
        };
        let (low, high) = bounds(&self.order_keys);
        if high > low.saturating_add(1) {
            return low + (high - low) / 2;
        }
        let old_keys = std::mem::replace(&mut self.order_keys, fresh_order_keys(self.body.len()));
        self.derived.renumber(|old| {
            let index = old_keys.partition_point(|&key| key < old);
            order_key_at(index)
        });
        let (low, high) = bounds(&self.order_keys);
        low + (high - low) / 2
    }

    /// The typed record a value edit of the record at a body position makes,
    /// refused when the writer would refuse it or when it would drop values
    /// the record carries beyond its declared count.
    fn edited_record(
        &self,
        position: usize,
        current: ClockRecord,
        values: Vec<f64>,
    ) -> Result<TypedRecord, RinexClockError> {
        validate_values(&values)?;
        if let Some(last) = current.surplus.iter().map(|value| value.position).max() {
            if values.len() <= last {
                return Err(invalid_input(
                    "values",
                    "the record carries values beyond its declared count; the new values must restate them",
                ));
            }
        }
        let typed = match &self.body[position] {
            BodyEntry::Typed(record) => TypedRecord {
                values,
                ..(**record).clone()
            },
            _ => TypedRecord {
                record_type: current.record_type,
                name: current.satellite.unwrap_or(current.name),
                epoch: TypedEpoch::Civil {
                    civil: current.civil,
                    second_text: current.second_text,
                },
                values,
            },
        };
        self.check_writable(&typed)?;
        Ok(typed)
    }

    /// Refuse a typed record the writer would refuse.
    fn check_writable(&self, record: &TypedRecord) -> Result<(), RinexClockError> {
        match self.writing_layout() {
            Some(layout) => {
                render_record(record, layout, self.context.time.scale, self.sigma_gap).map(|_| ())
            }
            None => Ok(()),
        }
    }

    fn header_lines(&self) -> Vec<(Option<usize>, &str)> {
        self.header
            .iter()
            .map(|entry| {
                (
                    header_entry_line(entry),
                    header_entry_content(&self.source, &self.line_starts, entry),
                )
            })
            .collect()
    }

    fn record_position(&self, index: usize) -> Result<usize, RinexClockError> {
        self.record_positions
            .get(index)
            .copied()
            .ok_or_else(|| invalid_input("index", "no record at this index"))
    }

    fn writing_layout(&self) -> Option<ClockLayout> {
        match self.constructed {
            Some(_) => self.context.layout,
            None => Some(self.context.layout.unwrap_or(ClockLayout::V300)),
        }
    }

    fn line_ending(&self) -> &'static str {
        (0..self.line_starts.len())
            .map(|index| line_span(&self.source, &self.line_starts, index))
            .find(|span| span.ends_with('\n'))
            .map_or(
                "\n",
                |span| {
                    if span.ends_with("\r\n") {
                        "\r\n"
                    } else {
                        "\n"
                    }
                },
            )
    }

    fn push_source(&self, out: &mut String, open: &mut bool, index: usize, eol: &str) {
        if *open {
            out.push_str(eol);
        }
        let span = line_span(&self.source, &self.line_starts, index);
        out.push_str(span);
        *open = !span.ends_with('\n');
    }

    fn write_constructed(
        &self,
        scale: TimeScale,
        policy: ClockWritePolicy,
    ) -> Result<(String, Vec<ClockWriteDeparture>), RinexClockError> {
        let system = ClockTimeSystem::for_time_scale(scale)
            .ok_or(RinexClockError::UnsupportedTimeScale { scale })?;
        let (layout, _) = constructed_layout(system);
        let ctx = epoch_context(&self.context.time);
        let mut record_lines = Vec::new();
        let mut departures = Vec::new();
        let mut systems = BTreeSet::new();
        let mut types = BTreeSet::new();
        for (record_index, entry) in self.body.iter().enumerate() {
            if let BodyEntry::Typed(record) = entry {
                let (lines, rounded) = render_record_with(
                    record,
                    layout,
                    Some(scale),
                    SigmaGap::One,
                    policy.nearest_microsecond_epochs,
                )?;
                if rounded {
                    departures.push(epoch_departure(record_index, record, &ctx, layout, &lines));
                }
                record_lines.extend(lines);
                types.insert(record.record_type);
                if record.record_type == ClockRecordType::As {
                    if let Some(letter) = record.name.chars().next() {
                        systems.insert(letter);
                    }
                }
            }
        }
        let valid_letters: &[char] = match layout {
            ClockLayout::V300 => &['G', 'R', 'E', 'S'],
            ClockLayout::V304 => &['G', 'R', 'E', 'C', 'I', 'J', 'S'],
        };
        let satellite_system = match systems.iter().next() {
            None => ' ',
            Some(&letter) if systems.len() == 1 && valid_letters.contains(&letter) => letter,
            Some(_) => 'M',
        };
        let type_codes: Vec<&str> = types.iter().map(|record_type| record_type.code()).collect();
        let mut out = String::new();
        for line in render_constructed_header(system, satellite_system, &type_codes)
            .into_iter()
            .chain(record_lines)
        {
            out.push_str(&line);
            out.push('\n');
        }
        Ok((out, departures))
    }
}

/// The departure a record written at its nearest microsecond epoch makes.
fn epoch_departure(
    record_index: usize,
    record: &TypedRecord,
    ctx: &EpochContext,
    layout: ClockLayout,
    lines: &[String],
) -> ClockWriteDeparture {
    let columns = match layout {
        ClockLayout::V300 => 8..34,
        ClockLayout::V304 => 13..39,
    };
    let written = lines
        .first()
        .and_then(|line| line.get(columns))
        .map(|text| text.split_whitespace().collect::<Vec<_>>().join(" "))
        .unwrap_or_default();
    ClockWriteDeparture::EpochAtNearestMicrosecond {
        record: record_index,
        name: record.name.clone(),
        epoch: record.view(ctx).epoch,
        written,
    }
}

/// Read the record starting at source line `first`, returning it and the
/// number of lines it spans, or a diagnostic and the lines to keep with it.
fn read_record_at(
    text: &str,
    starts: &[usize],
    first: usize,
    layout: Option<ClockLayout>,
    ctx: &EpochContext,
) -> Result<(ClockRecord, usize), (RinexClockDiagnostic, usize)> {
    let line_number = first + 1;
    let content = line_content(text, starts, first);
    let parent = read_parent(line_number, content, layout, ctx)
        .map_err(|error| (RinexClockDiagnostic::new(line_number, error), 1))?;
    let record_type = parent.record_type;
    let count = parent.count;
    let mut record = ClockRecord {
        record_type,
        name: parent.name,
        satellite: parent.satellite,
        civil: parent.civil,
        second_text: Some(parent.second_text),
        epoch: parent.epoch,
        epoch_source: EpochSource::Civil(parent.civil),
        values: parent.values,
        surplus: parent.surplus,
        line: Some(line_number),
        line_count: 1,
        reading: parent.reading,
        continuation_reading: None,
    };
    if count <= 2 {
        return Ok((record, 1));
    }

    let missing = || {
        (
            RinexClockDiagnostic::new(
                line_number,
                RinexClockError::MissingContinuation {
                    line: line_number,
                    record_type: record_type.code().to_string(),
                },
            ),
            1,
        )
    };
    let mut continuation = first + 1;
    while continuation < starts.len() && line_content(text, starts, continuation).trim().is_empty()
    {
        continuation += 1;
    }
    if continuation >= starts.len() {
        return Err(missing());
    }
    let continuation_content = line_content(text, starts, continuation);
    if is_potential_parent_record(continuation_content) {
        return Err(missing());
    }
    let used = continuation - first + 1;
    let read = read_continuation(continuation + 1, continuation_content, count - 2, layout)
        .map_err(|error| (RinexClockDiagnostic::new(continuation + 1, error), used))?;
    record.values.extend(read.values);
    record.surplus.extend(read.surplus);
    record.line_count = used;
    record.continuation_reading = Some(read.reading);
    Ok((record, used))
}

fn epoch_context(time: &TimeResolution) -> EpochContext {
    let policy = match (time.scale, time.system) {
        (Some(scale), _) => civil_second_policy_for_time_scale(scale),
        (None, Some(ClockTimeSystem::Irn)) => validate::CivilSecondPolicy::Continuous,
        (None, _) => validate::CivilSecondPolicy::UtcLike,
    };
    EpochContext {
        scale: time.scale,
        policy,
    }
}

/// Check that a civil epoch names an instant in a product's time scale.
fn check_civil_in_context(civil: Civil, ctx: &EpochContext) -> Result<(), RinexClockError> {
    if civil.second == 60 && ctx.policy == validate::CivilSecondPolicy::Continuous {
        return Err(invalid_input(
            "epoch",
            "a 23:59:60 label names no epoch in a continuous time scale",
        ));
    }
    if let Some(scale) = ctx.scale {
        civil_to_instant(scale, civil)
            .map_err(|_| invalid_input("epoch", "invalid civil clock epoch"))?;
    }
    Ok(())
}

fn header_entry_line(entry: &HeaderEntry) -> Option<usize> {
    match entry {
        HeaderEntry::Source(index) => Some(index + 1),
        HeaderEntry::Written(_) => None,
    }
}

fn header_entry_content<'a>(source: &'a str, starts: &[usize], entry: &'a HeaderEntry) -> &'a str {
    match entry {
        HeaderEntry::Source(index) => line_content(source, starts, *index),
        HeaderEntry::Written(text) => text.as_str(),
    }
}

fn push_written(out: &mut String, open: &mut bool, text: &str, eol: &str) {
    if *open {
        out.push_str(eol);
    }
    out.push_str(text);
    out.push_str(eol);
    *open = false;
}

/// Start offset of every physical line. A line runs to and includes its `\n`;
/// a final line without one is still a line.
fn split_lines(text: &str) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut position = 0;
    while position < text.len() {
        starts.push(position);
        position = match text[position..].find('\n') {
            Some(offset) => position + offset + 1,
            None => text.len(),
        };
    }
    starts
}

/// A physical line including its terminator.
fn line_span<'a>(text: &'a str, starts: &[usize], index: usize) -> &'a str {
    let start = starts.get(index).copied().unwrap_or(text.len());
    let end = starts.get(index + 1).copied().unwrap_or(text.len());
    text.get(start..end).unwrap_or("")
}

/// A physical line without its `\n` or `\r\n` terminator.
fn line_content<'a>(text: &'a str, starts: &[usize], index: usize) -> &'a str {
    let span = line_span(text, starts, index);
    match span.strip_suffix('\n') {
        Some(rest) => rest.strip_suffix('\r').unwrap_or(rest),
        None => span,
    }
}

fn validate_clock_point(point: &ClockPoint) -> Result<(), RinexClockError> {
    validate_instant(point.epoch, "epoch")?;
    validate_finite(point.bias_s, "bias_s")?;
    if point.additional_values.len() > 5 {
        return Err(invalid_input(
            "additional_values",
            "cannot exceed 5 additional values (maximum count is 6)",
        ));
    }
    for (idx, &val) in point.additional_values.iter().enumerate() {
        validate_finite(val, numeric::field_name_for_value_index(idx + 1))?;
    }
    Ok(())
}

fn validate_finite(value: f64, field: &'static str) -> Result<(), RinexClockError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "must be finite"))
    }
}

fn invalid_input(field: &'static str, reason: &'static str) -> RinexClockError {
    RinexClockError::InvalidInput { field, reason }
}

fn map_manual_order_error(error: FieldError) -> RinexClockError {
    match error {
        FieldError::NonFinite { field } => invalid_input(field, "must be finite"),
        FieldError::OutOfRange { field, .. } => invalid_input(field, "must be strictly increasing"),
        _ => invalid_input(error.field(), error.reason()),
    }
}

fn validate_instant_series_order(points: &[(ClockPoint, usize)]) -> Result<(), RinexClockError> {
    if points
        .windows(2)
        .any(|pair| epoch_cmp(&pair[0].0.epoch, &pair[1].0.epoch) != Ordering::Less)
    {
        return Err(invalid_input("epoch", "must be strictly increasing"));
    }
    Ok(())
}

/// Order keys for `count` entries assigned afresh.
fn fresh_order_keys(count: usize) -> Vec<u64> {
    (0..count).map(order_key_at).collect()
}

/// The order key a fresh assignment gives the entry at a body position.
fn order_key_at(position: usize) -> u64 {
    (position as u64 + 1).saturating_mul(ORDER_KEY_GAP)
}
