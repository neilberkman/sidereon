//! SP3 serialization - the inverse of the parser ([`super::Sp3::parse`]).
//!
//! Pure and deterministic: a product that serializes at all always produces
//! byte-identical text. No I/O. A read -> (merge) -> write pipeline round-trips
//! exactly: re-parsing the output yields the same epochs, satellites, positions,
//! velocities, clocks, and clock rates the product holds, value for value.
//! Header fields are derived from the product, never hardcoded: the data-used
//! and file-type descriptors, the per-satellite accuracy codes, and the first
//! `%f` line's position/velocity and clock/rate bases are written back from the
//! parsed header. The remaining `%f`/`%i` descriptor columns carry the format's
//! standard zero values, which is all the parser models of them.
//!
//! # What the writer refuses
//!
//! [`Sp3::to_sp3_string`] reports an [`Sp3WriteError`] rather than emitting text
//! that says something other than what the product holds. Every value is checked
//! against the fixed columns that carry it: text fields must be printable ASCII
//! no wider than their columns and must survive the reader's trim unchanged, the
//! header satellite list must agree with the records stored against it, and the
//! per-epoch arrays must be parallel to the epoch list.
//!
//! Every numeric field is checked by writing it: the column is formatted, read
//! back the way [`Sp3::parse`] reads it, and the result - after the unit factor
//! the reader applies - must be the value the product holds, bit for bit,
//! including the sign of a zero. The intermediate division that forms a column
//! is not required to be exact on its own: a kilometre value one unit in the
//! last place off its parsed original prints the same six decimals and
//! multiplies back to the same metres, and that is what the check asks. A value
//! that fails is never rounded, truncated, shifted into a neighbouring column,
//! replaced by a default, or dropped from the output.
//!
//! The record columns are `F14.6`: millimetres for a position, picoseconds for
//! a clock. A product holding a finer value - the mean of several analysis
//! centers, an interpolated state written into a product - cannot be stated in
//! them, and the writer reports
//! [`Sp3WriteError::RecordValueNotRepresentable`], naming the satellite, the
//! epoch, and the value. Rounding it to fit would publish numbers the product
//! does not hold under a header that says they are the product; a caller that
//! wants SP3-resolution output rounds its own values first, deliberately. The
//! parser applies the same rule to every record value it reads, so a record
//! value read from a file always writes back. That scope is the record value,
//! not the whole product: a parsed product can still be refused on a header
//! field the parser read more loosely - the `%f` bases below are the named
//! case - or on structure no record states, such as a velocity read under a
//! `P` header.
//!
//! The format spells some absences with a value: an all-zero position is a
//! missing orbit, an all-zero velocity is a missing velocity, and a clock column
//! at or beyond `999999.999999` is "no estimate". A stored value that would land
//! on one of those is reported as [`Sp3WriteError::RecordReadsAsAbsent`] rather
//! than emitted, so a present value is never written as an absence.
//!
//! # Epochs
//!
//! An epoch record states a calendar day and a clock time whose finest digit is
//! the 10-nanosecond tick the `F11.8` seconds field resolves, and it is checked
//! the same way a numeric column is: the candidate record is formatted, read
//! back through the civil validation and the civil-to-Julian conversion
//! [`Sp3::parse`] itself runs on an epoch line, and the instant that yields must
//! be the instant the product holds. There is no tolerance on an epoch, physical
//! or numerical. The comparison is of the instant `JulianDateSplit` defines -
//! `jd_whole + fraction`, exactly - so a split that carries the same moment
//! across the day boundary differently still compares equal, while a fraction
//! one unit in the last place away does not.
//!
//! This is stricter than a bound on how far the stored epoch sits from the
//! record could be. An epoch built by arithmetic rather than read from a file -
//! a day added to a product through its day *fraction*, say - can land a few
//! units in the last place from the instant any record states, and the record
//! that would be written then means a different time than the product holds,
//! by however little. Such an epoch is reported as
//! [`Sp3WriteError::EpochNotRestatable`], naming the epoch and how far the
//! record would move it, rather than written. Adding the day on the *boundary*
//! instead (`jd_whole + 1.0`) is exact and writes.
//!
//! Everything [`Sp3::parse`] produces restates exactly: the writer recovers the
//! same calendar fields the parser was handed, so the conversion back is the
//! same call on the same arguments. That includes a UTC-like `23:59:60` leap
//! label, which `split_julian_date` folds into the following day - the writer
//! offers the label and the ordinary next-day statement as candidates and keeps
//! whichever restates the stored split, so a leap epoch is written back as the
//! leap line it was read from.
//!
//! An epoch held as [`crate::astro::time::model::InstantRepr::Nanos`] rather
//! than as a split Julian date is refused by name
//! ([`Sp3WriteError::EpochRepresentationUnsupported`]). `Sp3::parse` never
//! builds one, but [`Sp3::epochs`] is public and can be assigned one; the
//! refusal records the limit of this writer's contract, not a claim that the
//! calendar fields are unrecoverable from an exact nanosecond count. See that
//! variant for why choosing an origin here is a separate decision.
//!
//! Header numeric fields are held to the same standard, so a base or cadence
//! value re-reads as exactly the number the product holds. The parser is looser
//! than that on the `%f` bases on purpose - it keeps any finite value its source
//! field expressed - so a product can carry a base this writer declines to emit;
//! that refusal is [`Sp3WriteError::PrecisionNotRepresentable`], not a parse
//! failure.
//!
//! # What a successful write does and does not claim
//!
//! Success states that the product's modeled content is expressed exactly; it is
//! not a certificate of full SP3 conformance. Position and velocity
//! standard-deviation exponents, the `EP`/`EV` correlation records, and records
//! for satellites outside the header list are not part of the parsed product, so
//! they are absent from the output. A source file carrying them does not
//! reproduce byte for byte through a read/write cycle.
//!
//! A satellite absent at an epoch is written as the SP3 missing-orbit sentinel
//! (`0.0 0.0 0.0`, bad clock), never a fabricated position - so a quarantined
//! `(sat, epoch)` cell from [`super::merge`] re-reads as missing, not zero. For
//! velocity products the matching `V` record is still emitted, using the SP3
//! missing-velocity vector and bad clock-rate sentinel when needed.

use core::fmt::Write as _;
use std::collections::BTreeSet;

use crate::astro::time::civil::civil_from_julian_day_number as civil_from_jdn;
use crate::astro::time::model::{Instant, JulianDateSplit, TimeScale};
use crate::constants::{KM_TO_M, SECONDS_PER_DAY, US_TO_S};
use crate::frame::ItrfVelocityMS;
use crate::id::GnssSatelliteId;
use crate::validate;

use super::{
    Sp3, Sp3ClockRecord, Sp3DataType, Sp3Flags, Sp3State, Sp3TimeSystem, Sp3Version, BAD_CLOCK_US,
    CLOCK_RATE_TO_S_PER_S, DM_S_TO_M_S, EPOCH_SECONDS_DECIMALS, EPOCH_SECONDS_WIDTH,
    LINE2_INTERVAL_DECIMALS, LINE2_INTERVAL_WIDTH, LINE2_MJD_FRACTION_DECIMALS,
    LINE2_SECONDS_OF_WEEK_DECIMALS, LINE2_SECONDS_OF_WEEK_WIDTH, LINE_PF_CLOCK_RATE_BASE_DECIMALS,
    LINE_PF_CLOCK_RATE_BASE_WIDTH, LINE_PF_POS_VEL_BASE_DECIMALS, LINE_PF_POS_VEL_BASE_WIDTH,
    MISSING_POSITION_KM, MISSING_VELOCITY_DM_S, RECORD_VALUE_DECIMALS, RECORD_VALUE_WIDTH,
};

/// Maximum SP3 satellite-id slots per `+` / `++` header line.
const SATS_PER_LINE: usize = 17;
/// SP3-c fixes five `+`/`++` lines (85 slots); SP3-d may use more.
const MIN_PLUS_LINES: usize = 5;
/// SP3-d retains at least four header comment records for backward
/// compatibility, even when fewer carry semantic text.
const MIN_COMMENT_LINES: usize = 4;
const SP3_TIME_TICKS_PER_SECOND: i64 = 100_000_000;
const SP3_TIME_TICKS_PER_MINUTE: i64 = 60 * SP3_TIME_TICKS_PER_SECOND;
const SP3_TIME_TICKS_PER_HOUR: i64 = 60 * SP3_TIME_TICKS_PER_MINUTE;
const SP3_TIME_TICKS_PER_DAY: i64 = 24 * SP3_TIME_TICKS_PER_HOUR;

/// Columns each emitted field occupies, in the canonical SP3 header layout.
///
/// These are the spans this writer fills, and what a value has to fit for the
/// emitted line to keep every later field in its own columns. A column the
/// format reserves as a blank separator is *not* part of the span beside it:
/// the writer emits the blank and fits the value in the columns the layout
/// names for it. This parser also reads a field back from the separator
/// onwards, so it would accept a value that took one; a reader indexing the
/// specified columns would not, and would return a truncated label.
///
/// SP3-d layout, header line 1: epoch count `I7` in columns 33-39, data used
/// `A5` in 41-45, coordinate system `A5` in 47-51, orbit type `A3` in 53-55,
/// agency `A4` in 57-60, with columns 40, 46, 52, and 56 reserved blank.
const LINE1_EPOCH_COUNT_COLUMNS: usize = 7;
const LINE1_DATA_USED_COLUMNS: usize = 5;
const LINE1_COORDINATE_SYSTEM_COLUMNS: usize = 5;
const LINE1_ORBIT_TYPE_COLUMNS: usize = 3;
const LINE1_AGENCY_COLUMNS: usize = 4;
const LINE2_GNSS_WEEK_COLUMNS: usize = 4;
const LINE2_MJD_COLUMNS: usize = 5;
/// Line 2 fractional day: `F15.13` in columns 46-60, the last field on the line.
const LINE2_MJD_FRACTION_WIDTH: usize = 15;
/// `+` line 1 satellite count (`I3`) and every 3-column satellite-id slot.
const SATELLITE_COUNT_COLUMNS: usize = 3;
const SATELLITE_TOKEN_COLUMNS: usize = 3;
/// `++` per-satellite accuracy exponent (`I3`).
const ACCURACY_CODE_COLUMNS: usize = 3;
/// First `%c` line file-type descriptor (`A2`, columns 4-5).
const PC_FILE_TYPE_COLUMNS: usize = 2;
/// Comment text (`A77`, columns 4-80).
const COMMENT_COLUMNS: usize = 77;
/// Calendar year field (`I4`) shared by header line 1 and every epoch record.
const CALENDAR_YEAR_COLUMNS: usize = 4;

/// Largest Julian Day Number magnitude the calendar inverse is evaluated at.
///
/// Around three billion years either side of the epoch, some six hundred times
/// the four-digit year field's reach and well inside what
/// [`civil_from_julian_day_number`] multiplies an `i64` by.
///
/// [`civil_from_julian_day_number`]: crate::astro::time::civil::civil_from_julian_day_number
const JDN_CONVERSION_LIMIT: f64 = (1_i64 << 40) as f64;

/// Why a product cannot be written as canonical SP3 text.
///
/// Every variant names the field that could not be expressed and the value that
/// could not be expressed in it - for a record field, the satellite and epoch
/// holding it as well - so a caller can repair the product rather than guess.
/// See the module docs for what the writer checks and why.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Sp3WriteError {
    /// The product holds no epoch, so header line 1 has no start epoch to state.
    /// SP3 has no representation for an epoch-less product.
    NoEpochs,
    /// A text field carries a byte canonical SP3 text cannot hold: a line break,
    /// another control byte, or a non-ASCII byte.
    TextNotColumnSafe {
        /// The header field the text belongs to.
        field: &'static str,
        /// The text as the product holds it.
        value: String,
    },
    /// A text field carries leading or trailing whitespace. The reader trims the
    /// column it is written into, so the value would come back changed.
    TextNotColumnStable {
        /// The header field the text belongs to.
        field: &'static str,
        /// The text as the product holds it.
        value: String,
    },
    /// An optional descriptor holds a blank string. Its columns spell absence
    /// with a blank, so `Some("")` would come back as `None`.
    BlankDescriptor {
        /// The header field the descriptor belongs to.
        field: &'static str,
        /// The text as the product holds it.
        value: String,
    },
    /// A comment record holds no text. The reader keeps `/*` records carrying
    /// text and treats the rest as the format's structural padding, so an empty
    /// comment would not come back at all.
    EmptyComment {
        /// Index of the comment in [`Sp3::comments`].
        index: usize,
        /// The text as the product holds it.
        value: String,
    },
    /// A text field is wider than the columns reserved for it; writing it would
    /// push every later field on the line out of its own columns.
    TextTooWide {
        /// The header field the text belongs to.
        field: &'static str,
        /// Columns the field occupies.
        columns: usize,
        /// The text as the product holds it.
        value: String,
    },
    /// An integer field holds a value wider than its columns express.
    IntegerTooWide {
        /// The header field the count belongs to.
        field: &'static str,
        /// Columns the field occupies.
        columns: usize,
        /// The value as the product holds it.
        value: u64,
    },
    /// A numeric field is not finite. SP3 fixed-column text has no form for an
    /// infinity or a NaN, and the parser keeps such a header value only so exact
    /// validation can report it as a typed integrity failure.
    NonFinite {
        /// The field the value belongs to.
        field: &'static str,
    },
    /// A finite value's `F{columns}.{decimals}` form is wider than its columns.
    NumberTooWide {
        /// The field the value belongs to.
        field: &'static str,
        /// Columns the field occupies.
        columns: usize,
        /// Decimal places the field carries.
        decimals: usize,
        /// The value as the product holds it.
        value: f64,
    },
    /// A finite value carries more precision than its `F{columns}.{decimals}`
    /// field expresses, so writing it would silently change the number.
    PrecisionNotRepresentable {
        /// The field the value belongs to.
        field: &'static str,
        /// Columns the field occupies.
        columns: usize,
        /// Decimal places the field carries.
        decimals: usize,
        /// The value as the product holds it.
        value: f64,
    },
    /// A calendar year falls outside the four digits the header line-1 and epoch
    /// records reserve for it.
    YearNotRepresentable {
        /// Index of the epoch whose calendar year is out of range.
        epoch_index: usize,
        /// The year the epoch converts to.
        year: i64,
    },
    /// An epoch is held as [`InstantRepr::Nanos`] rather than as a split Julian
    /// date, and this writer states a calendar record only from the latter.
    ///
    /// This is a limit of the SP3 writer's contract, not of the representation.
    /// An integer-nanosecond instant carries an exact count, and several core
    /// adapters read one against the J2000 origin
    /// ([`crate::astro::time::civil::julian_date_from_instant`], and the RINEX
    /// clock writer's own civil decomposition, which is exact integer
    /// arithmetic). SP3's own node axis takes the opposite position: the count
    /// is "nanoseconds since an implied scale epoch"
    /// ([`InstantRepr::Nanos`]), the type names no origin, and
    /// `sp3::interp` declines the conversion for that reason rather than
    /// assuming one. Writing a calendar record here would have to settle that
    /// question for the whole SP3 module, and settling it is a separate public
    /// contract - so the writer refuses by name instead of picking an origin,
    /// and instead of reducing an `i128` through `f64` to hide the choice.
    ///
    /// [`InstantRepr::Nanos`]: crate::astro::time::model::InstantRepr::Nanos
    EpochRepresentationUnsupported {
        /// Index of the epoch in [`Sp3::epochs`].
        epoch_index: usize,
    },
    /// An epoch record cannot restate the instant the product holds: read back
    /// the way [`Sp3::parse`] reads an epoch line, the record states a
    /// different instant.
    EpochNotRestatable {
        /// Index of the epoch in [`Sp3::epochs`].
        epoch_index: usize,
        /// Seconds-of-minute the record would state for it.
        field_seconds: f64,
        /// How far the stored epoch sits from the instant that record states,
        /// `stored - stated`. `NaN` when no candidate statement could be read
        /// back at all, which leaves no instant to measure against.
        residual_s: f64,
    },
    /// An epoch is tagged with a different time scale than the header states.
    /// The epoch records carry no scale of their own - the header's `%c`
    /// descriptor names it for the whole product - so the epoch would come back
    /// tagged with the header's.
    EpochTimeScaleMismatch {
        /// Index of the epoch in [`Sp3::epochs`].
        epoch_index: usize,
        /// Scale the epoch is tagged with.
        epoch_scale: TimeScale,
        /// Scale the header states.
        header_scale: TimeScale,
    },
    /// The header's SP3 time system and core time scale disagree. One `%c`
    /// descriptor carries both, so the pair would come back reconciled.
    HeaderTimeScaleMismatch {
        /// SP3 time system the header states.
        time_system: Sp3TimeSystem,
        /// Core time scale the header states.
        time_scale: TimeScale,
    },
    /// The header's epoch count differs from the number of epochs the product
    /// holds. Header line 1 states one number, and it is the body's.
    EpochCountMismatch {
        /// Count the header states.
        declared: u64,
        /// Epochs the product holds.
        epochs: usize,
    },
    /// The per-satellite accuracy codes are not index-aligned with the header
    /// satellite list, so at least one satellite has no code of its own.
    AccuracyCodeCountMismatch {
        /// Satellites declared in the header.
        satellites: usize,
        /// Accuracy codes held against them.
        codes: usize,
    },
    /// The header satellite list names one satellite twice. The `+` lines
    /// declare each satellite once, so one of the two would be lost.
    DuplicateSatellite {
        /// The satellite declared more than once.
        sat: GnssSatelliteId,
    },
    /// A stored per-epoch array is not parallel to the epoch list.
    EpochArrayLengthMismatch {
        /// The product array that does not line up.
        field: &'static str,
        /// Epochs the product holds.
        epochs: usize,
        /// Entries the array holds.
        entries: usize,
    },
    /// A satellite holds a record at an epoch but is absent from the header
    /// satellite list, so no record line would be written for it.
    UndeclaredSatelliteRecord {
        /// The satellite holding the record.
        sat: GnssSatelliteId,
        /// Index of the epoch in [`Sp3::epochs`].
        epoch_index: usize,
    },
    /// A satellite holds both a state and a clock-only record at one epoch. One
    /// `P` record states one or the other, so writing the state would drop the
    /// clock record.
    ConflictingRecords {
        /// The satellite holding both.
        sat: GnssSatelliteId,
        /// Index of the epoch in [`Sp3::epochs`].
        epoch_index: usize,
    },
    /// A position product holds velocity or clock-rate state. A `#dP` header
    /// declares no `V` records, so the stored value would be dropped; a product
    /// carrying velocities states so in its header.
    VelocityStateInPositionProduct {
        /// The record field holding the value.
        field: &'static str,
        /// The satellite holding it.
        sat: GnssSatelliteId,
        /// Index of the epoch in [`Sp3::epochs`].
        epoch_index: usize,
    },
    /// A record field is not finite. SP3 record columns have no form for an
    /// infinity or a NaN, and the parser accepts neither.
    RecordValueNonFinite {
        /// The record field the value belongs to.
        field: &'static str,
        /// The satellite holding it.
        sat: GnssSatelliteId,
        /// Index of the epoch in [`Sp3::epochs`].
        epoch_index: usize,
    },
    /// A record field's `F{columns}.{decimals}` form is wider than its columns,
    /// so writing it would push the rest of the record out of its own columns.
    RecordValueTooWide {
        /// The record field the value belongs to.
        field: &'static str,
        /// The satellite holding it.
        sat: GnssSatelliteId,
        /// Index of the epoch in [`Sp3::epochs`].
        epoch_index: usize,
        /// Columns the field occupies.
        columns: usize,
        /// Decimal places the field carries.
        decimals: usize,
        /// The number the column would carry, in the format's own units.
        column_value: f64,
    },
    /// A record field's column cannot restate the value the product holds: read
    /// back the way the parser reads it, the column is a different value.
    RecordValueNotRepresentable {
        /// The record field the value belongs to.
        field: &'static str,
        /// The satellite holding it.
        sat: GnssSatelliteId,
        /// Index of the epoch in [`Sp3::epochs`].
        epoch_index: usize,
        /// Columns the field occupies.
        columns: usize,
        /// Decimal places the field carries.
        decimals: usize,
        /// The value as the product holds it, in the product's own units
        /// (meters, seconds, meters per second, seconds per second).
        stored: f64,
        /// The number the column would carry, in the format's own units.
        column_value: f64,
    },
    /// A record field holds a value the format spells an absence with: an
    /// all-zero position or velocity, or a clock column at or beyond the
    /// bad-clock sentinel. Writing it would make a value the product holds come
    /// back missing.
    RecordReadsAsAbsent {
        /// The record field the value belongs to.
        field: &'static str,
        /// The satellite holding it.
        sat: GnssSatelliteId,
        /// Index of the epoch in [`Sp3::epochs`].
        epoch_index: usize,
        /// The number the column would carry, in the format's own units.
        column_value: f64,
    },
    /// A record holds a native-unit value with no value in the product's own
    /// units beside it, or the other way round. One column states one or the
    /// other, so the missing half would come back filled in.
    RecordFieldsDisagree {
        /// The record field the values belong to.
        field: &'static str,
        /// The satellite holding them.
        sat: GnssSatelliteId,
        /// Index of the epoch in [`Sp3::epochs`].
        epoch_index: usize,
        /// The value in the product's own units, when it holds one.
        stored: Option<f64>,
        /// The retained native-unit value, when it holds one.
        native: Option<f64>,
    },
}

impl core::fmt::Display for Sp3WriteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoEpochs => write!(f, "SP3 product holds no epoch to write as its start epoch"),
            Self::TextNotColumnSafe { field, value } => write!(
                f,
                "SP3 {field} {value:?} carries a byte fixed-column SP3 text cannot hold"
            ),
            Self::TextNotColumnStable { field, value } => write!(
                f,
                "SP3 {field} {value:?} carries whitespace its column loses when read back"
            ),
            Self::BlankDescriptor { field, value } => write!(
                f,
                "SP3 {field} {value:?} is blank, which its column reads back as an absent field"
            ),
            Self::EmptyComment { index, value } => write!(
                f,
                "SP3 comment {index} {value:?} carries no text, which its record reads back as padding"
            ),
            Self::TextTooWide {
                field,
                columns,
                value,
            } => write!(
                f,
                "SP3 {field} {value:?} is wider than its {columns} columns"
            ),
            Self::IntegerTooWide {
                field,
                columns,
                value,
            } => write!(f, "SP3 {field} {value} exceeds its {columns} columns"),
            Self::NonFinite { field } => write!(f, "SP3 {field} is not finite"),
            Self::NumberTooWide {
                field,
                columns,
                decimals,
                value,
            } => write!(
                f,
                "SP3 {field} {value} is wider than its F{columns}.{decimals} field"
            ),
            Self::PrecisionNotRepresentable {
                field,
                columns,
                decimals,
                value,
            } => write!(
                f,
                "SP3 {field} {value} is finer than its F{columns}.{decimals} field states"
            ),
            Self::YearNotRepresentable { epoch_index, year } => write!(
                f,
                "SP3 epoch {epoch_index} falls in year {year}, outside the 4-digit year field"
            ),
            Self::EpochRepresentationUnsupported { epoch_index } => write!(
                f,
                "SP3 epoch {epoch_index} counts nanoseconds from an origin the instant does not name; this writer states a calendar record only from a split Julian date"
            ),
            Self::EpochNotRestatable {
                epoch_index,
                field_seconds,
                residual_s,
            } => write!(
                f,
                "SP3 epoch {epoch_index} sits {residual_s} s from the instant its record would state at {field_seconds} s"
            ),
            Self::EpochTimeScaleMismatch {
                epoch_index,
                epoch_scale,
                header_scale,
            } => write!(
                f,
                "SP3 epoch {epoch_index} is in {epoch_scale:?} but the header states {header_scale:?}"
            ),
            Self::HeaderTimeScaleMismatch {
                time_system,
                time_scale,
            } => write!(
                f,
                "SP3 header time system {} does not carry time scale {time_scale:?}",
                time_system.label()
            ),
            Self::EpochCountMismatch { declared, epochs } => write!(
                f,
                "SP3 header declares {declared} epochs but the product holds {epochs}"
            ),
            Self::AccuracyCodeCountMismatch { satellites, codes } => write!(
                f,
                "SP3 header declares {satellites} satellites but holds {codes} accuracy codes"
            ),
            Self::DuplicateSatellite { sat } => {
                write!(f, "SP3 header satellite list names {sat} more than once")
            }
            Self::EpochArrayLengthMismatch {
                field,
                epochs,
                entries,
            } => write!(
                f,
                "SP3 {field} holds {entries} entries for {epochs} epochs"
            ),
            Self::UndeclaredSatelliteRecord { sat, epoch_index } => write!(
                f,
                "SP3 epoch {epoch_index} holds a record for undeclared satellite {sat}"
            ),
            Self::ConflictingRecords { sat, epoch_index } => write!(
                f,
                "SP3 epoch {epoch_index} holds both a state and a clock record for {sat}"
            ),
            Self::VelocityStateInPositionProduct {
                field,
                sat,
                epoch_index,
            } => write!(
                f,
                "SP3 position product holds a {field} for {sat} at epoch {epoch_index}"
            ),
            Self::RecordValueNonFinite {
                field,
                sat,
                epoch_index,
            } => write!(
                f,
                "SP3 {field} for {sat} at epoch {epoch_index} is not finite"
            ),
            Self::RecordValueTooWide {
                field,
                sat,
                epoch_index,
                columns,
                decimals,
                column_value,
            } => write!(
                f,
                "SP3 {field} for {sat} at epoch {epoch_index} is {column_value}, wider than its F{columns}.{decimals} field"
            ),
            Self::RecordValueNotRepresentable {
                field,
                sat,
                epoch_index,
                columns,
                decimals,
                stored,
                column_value,
            } => write!(
                f,
                "SP3 {field} {stored} for {sat} at epoch {epoch_index} is finer than the {column_value} its F{columns}.{decimals} field states"
            ),
            Self::RecordReadsAsAbsent {
                field,
                sat,
                epoch_index,
                column_value,
            } => write!(
                f,
                "SP3 {field} for {sat} at epoch {epoch_index} is {column_value}, which the format reads back as an absent value"
            ),
            Self::RecordFieldsDisagree {
                field,
                sat,
                epoch_index,
                stored,
                native,
            } => write!(
                f,
                "SP3 {field} for {sat} at epoch {epoch_index} holds {stored:?} beside native {native:?}"
            ),
        }
    }
}

impl std::error::Error for Sp3WriteError {}

impl Sp3 {
    /// Serialize this product to standard SP3 text (the format named by its
    /// header version, `c` or `d`).
    ///
    /// Pure and deterministic. Returns [`Sp3WriteError`] when the product holds
    /// state the canonical fixed-column text cannot express: a field wider than
    /// its columns, a non-finite number, a record value finer than the `F14.6`
    /// column that carries it, a value that would land on one of the format's
    /// absence sentinels, an epoch off the grid its record states, or a
    /// satellite list that disagrees with the stored records. Nothing is
    /// rounded away, shifted, defaulted, or omitted to make a write succeed - a
    /// product whose values SP3 cannot state is reported, not approximated.
    ///
    /// On success every value re-reads as the one the product holds, bit for
    /// bit. See this module's docs for the checks, the round-trip and
    /// missing-satellite guarantees, and what a successful write does not
    /// claim.
    pub fn to_sp3_string(&self) -> Result<String, Sp3WriteError> {
        self.check_structure()?;
        let mut out =
            String::with_capacity(self.epochs.len() * (self.header.satellites.len() + 4) * 61);
        self.write_header(&mut out)?;
        self.write_records(&mut out)?;
        out.push_str("EOF\n");
        Ok(out)
    }

    /// Check the product's shape before any value is formatted: the header
    /// against itself, the satellite list against the codes and records held
    /// for it, and the per-epoch arrays against the epoch list. These are the
    /// conditions under which the record loop would otherwise invent an
    /// accuracy code, drop a stored record, or write a header field that says
    /// something the product does not.
    fn check_structure(&self) -> Result<(), Sp3WriteError> {
        let h = &self.header;
        // One `%c` descriptor carries both the SP3 label and the core scale the
        // parser derives from it, so a pair that disagrees cannot be written:
        // the product would come back with the pair the label implies.
        if h.time_scale != h.time_system.time_scale() {
            return Err(Sp3WriteError::HeaderTimeScaleMismatch {
                time_system: h.time_system,
                time_scale: h.time_scale,
            });
        }
        // Header line 1 states one epoch count, and the parser reads it back as
        // the number of epoch records. Writing the body count under a header
        // field that says otherwise would drop the stated one.
        if h.num_epochs != self.epochs.len() as u64 {
            return Err(Sp3WriteError::EpochCountMismatch {
                declared: h.num_epochs,
                epochs: self.epochs.len(),
            });
        }
        if h.satellite_accuracy_codes.len() != h.satellites.len() {
            return Err(Sp3WriteError::AccuracyCodeCountMismatch {
                satellites: h.satellites.len(),
                codes: h.satellite_accuracy_codes.len(),
            });
        }
        let mut declared = BTreeSet::new();
        for sat in &h.satellites {
            if !declared.insert(*sat) {
                return Err(Sp3WriteError::DuplicateSatellite { sat: *sat });
            }
            let token = sat.to_string();
            if token.len() != SATELLITE_TOKEN_COLUMNS {
                return Err(Sp3WriteError::TextTooWide {
                    field: "satellite id",
                    columns: SATELLITE_TOKEN_COLUMNS,
                    value: token,
                });
            }
        }

        let epochs = self.epochs.len();
        for (field, entries) in [
            ("satellite states", self.states.len()),
            ("clock records", self.clock_records.len()),
            ("interpolation nodes", self.interp_raw.len()),
            ("epoch seconds", self.epoch_j2000_s.len()),
        ] {
            if entries != epochs {
                return Err(Sp3WriteError::EpochArrayLengthMismatch {
                    field,
                    epochs,
                    entries,
                });
            }
        }

        let positions_only = matches!(self.header.data_type, Sp3DataType::Position);
        for (epoch_index, (states, clocks)) in
            self.states.iter().zip(&self.clock_records).enumerate()
        {
            // A record held against a satellite the header never declares would
            // simply not be written: the record loop walks the header list.
            // Report it rather than dropping it.
            for sat in states.keys().chain(clocks.keys()) {
                if !declared.contains(sat) {
                    return Err(Sp3WriteError::UndeclaredSatelliteRecord {
                        sat: *sat,
                        epoch_index,
                    });
                }
            }
            // One `P` record per satellite per epoch states either an orbit or
            // the missing-orbit sentinel with a clock. Holding both, the record
            // loop writes the state and the clock record is never emitted.
            for sat in states.keys() {
                if clocks.contains_key(sat) {
                    return Err(Sp3WriteError::ConflictingRecords {
                        sat: *sat,
                        epoch_index,
                    });
                }
            }
            // A `P` product writes no `V` records at all, so any velocity or
            // clock rate it holds would be dropped without a column to name it.
            if positions_only {
                for (sat, state) in states {
                    check_position_product_state(
                        *sat,
                        epoch_index,
                        state.velocity,
                        [state.clock_rate_s_s, None],
                    )?;
                }
                for (sat, record) in clocks {
                    check_position_product_state(
                        *sat,
                        epoch_index,
                        record.velocity,
                        [record.clock_rate_s_s, record.clock_rate_raw],
                    )?;
                }
            }
        }
        Ok(())
    }

    fn write_header(&self, out: &mut String) -> Result<(), Sp3WriteError> {
        let h = &self.header;
        let version = match h.version {
            Sp3Version::A => 'a',
            Sp3Version::B => 'b',
            Sp3Version::C => 'c',
            Sp3Version::D => 'd',
        };
        let dtype = match h.data_type {
            Sp3DataType::Position => 'P',
            Sp3DataType::Velocity => 'V',
        };

        // Line 1: version/type, first-epoch calendar (cosmetic - the parser reads
        // epochs from the `*` lines), epoch count, data descriptor, coordinate
        // system, orbit type, agency. Columns match the parser's field offsets.
        // The start epoch is the product's own first epoch; a product with no
        // epoch has no start to state and is refused rather than dated.
        let first = self.epochs.first().ok_or(Sp3WriteError::NoEpochs)?;
        let dt = self.format_calendar(first, 0)?;
        let data = optional_text(h.data_used.as_deref(), "data used", LINE1_DATA_USED_COLUMNS)?;
        check_text(
            &h.coordinate_system,
            "coordinate system",
            LINE1_COORDINATE_SYSTEM_COLUMNS,
        )?;
        check_text(&h.orbit_type, "orbit type", LINE1_ORBIT_TYPE_COLUMNS)?;
        check_text(&h.agency, "agency", LINE1_AGENCY_COLUMNS)?;
        let epoch_count = self.epochs.len() as u64;
        check_integer(epoch_count, "epoch count", LINE1_EPOCH_COUNT_COLUMNS)?;
        // Columns 40, 46, 52, and 56 are the format's reserved blanks and are
        // written as blanks; each label sits in the columns the layout gives it.
        let _ = writeln!(
            out,
            "#{version}{dtype}{dt} {n:>7} {data:<5} {coord:>5} {orbit:>3} {agency:>4}",
            n = epoch_count,
            coord = h.coordinate_system,
            orbit = h.orbit_type,
            agency = h.agency,
        );

        // Line 2 (`##`): GPS week, seconds-of-week, epoch interval, MJD, MJD frac.
        check_integer(u64::from(h.gnss_week), "GNSS week", LINE2_GNSS_WEEK_COLUMNS)?;
        check_exact(
            h.seconds_of_week,
            "seconds-of-week",
            LINE2_SECONDS_OF_WEEK_WIDTH,
            LINE2_SECONDS_OF_WEEK_DECIMALS,
        )?;
        check_exact(
            h.epoch_interval_s,
            "epoch interval",
            LINE2_INTERVAL_WIDTH,
            LINE2_INTERVAL_DECIMALS,
        )?;
        check_integer(u64::from(h.mjd), "MJD", LINE2_MJD_COLUMNS)?;
        check_exact(
            h.mjd_fraction,
            "MJD fraction",
            LINE2_MJD_FRACTION_WIDTH,
            LINE2_MJD_FRACTION_DECIMALS,
        )?;
        let _ = writeln!(
            out,
            "## {wk:>4} {sow:15.8} {interval:14.8} {mjd:>5} {frac:15.13}",
            wk = h.gnss_week,
            sow = h.seconds_of_week,
            interval = h.epoch_interval_s,
            mjd = h.mjd,
            frac = h.mjd_fraction,
        );

        // `+` satellite-id lines and `++` accuracy-exponent lines. The satellite
        // list and the codes are index-aligned by `check_structure`, so every
        // declared satellite writes its own code and none is invented.
        let sats = &h.satellites;
        check_integer(
            sats.len() as u64,
            "satellite count",
            SATELLITE_COUNT_COLUMNS,
        )?;
        let n_lines = MIN_PLUS_LINES.max(sats.len().div_ceil(SATS_PER_LINE));
        for line in 0..n_lines {
            // `+` line: first carries the count in columns 3-5; all start ids at 9.
            if line == 0 {
                let _ = write!(out, "+  {:>3}   ", sats.len());
            } else {
                out.push_str("+        ");
            }
            for slot in 0..SATS_PER_LINE {
                match sats.get(line * SATS_PER_LINE + slot) {
                    Some(sat) => {
                        let _ = write!(out, "{sat}");
                    }
                    None => out.push_str("  0"),
                }
            }
            out.push('\n');
        }
        for line in 0..n_lines {
            out.push_str("++       ");
            for slot in 0..SATS_PER_LINE {
                let idx = line * SATS_PER_LINE + slot;
                // A slot past the satellite list is the format's zero padding,
                // not a satellite whose code went missing: `check_structure`
                // has already required one code per declared satellite.
                let code = h.satellite_accuracy_codes.get(idx).copied().unwrap_or(0);
                check_integer(
                    u64::from(code),
                    "satellite accuracy code",
                    ACCURACY_CODE_COLUMNS,
                )?;
                let _ = write!(out, "{code:>3}");
            }
            out.push('\n');
        }

        // `%c` descriptors: the first carries the file type in columns 4-5 and
        // the time system in columns 10-12, both of which the parser reads back.
        // Its remaining columns, and the whole second `%c` line, are the
        // format's filler. The first `%f` line carries the position/velocity and
        // clock/rate bases the parser reads; its two trailing floats and the
        // second `%f` line are the standard zeros. `%i` is standard throughout.
        let tsys = h.time_system.label();
        let ft = optional_text(h.file_type.as_deref(), "file type", PC_FILE_TYPE_COLUMNS)?;
        let _ = writeln!(
            out,
            "%c {ft:<2} cc {tsys} ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc"
        );
        out.push_str("%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
        let pv_base = optional_base(
            h.pos_vel_base,
            "pos/vel base",
            LINE_PF_POS_VEL_BASE_WIDTH,
            LINE_PF_POS_VEL_BASE_DECIMALS,
        )?;
        let clk_base = optional_base(
            h.clock_rate_base,
            "clock/rate base",
            LINE_PF_CLOCK_RATE_BASE_WIDTH,
            LINE_PF_CLOCK_RATE_BASE_DECIMALS,
        )?;
        let _ = writeln!(
            out,
            "%f {pv_base} {clk_base}  0.00000000000  0.000000000000000"
        );
        out.push_str("%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n");
        out.push_str("%i    0    0    0    0      0      0      0      0         0\n");
        out.push_str("%i    0    0    0    0      0      0      0      0         0\n");

        // Provenance comments (e.g. merge derivation) are preserved when present.
        for (index, comment) in self.comments.iter().enumerate() {
            check_comment(comment, index)?;
            let _ = writeln!(out, "/* {comment}");
        }
        for _ in self.comments.len()..MIN_COMMENT_LINES {
            out.push_str("/*\n");
        }
        Ok(())
    }

    fn write_records(&self, out: &mut String) -> Result<(), Sp3WriteError> {
        let with_velocity = matches!(self.header.data_type, Sp3DataType::Velocity);
        for (idx, epoch) in self.epochs.iter().enumerate() {
            let dt = self.format_calendar(epoch, idx)?;
            let _ = writeln!(out, "*  {dt}");

            let states = &self.states[idx];
            let clock_records = &self.clock_records[idx];
            // Every header satellite gets a record at every epoch; an absent one
            // is the missing-orbit sentinel (so a quarantined cell is "missing",
            // never a fabricated zero position). Velocity products also get the
            // paired V record, using the missing-velocity sentinel as needed.
            for sat in &self.header.satellites {
                let site = RecordSite {
                    sat: *sat,
                    epoch_index: idx,
                };
                if let Some(state) = states.get(sat) {
                    write_state_record(out, site, state, with_velocity)?;
                } else if let Some(clock_rec) = clock_records.get(sat) {
                    write_clock_record(out, site, clock_rec, with_velocity)?;
                } else {
                    let _ = writeln!(
                        out,
                        "P{sat}{:14.6}{:14.6}{:14.6}{:14.6}",
                        MISSING_POSITION_KM, MISSING_POSITION_KM, MISSING_POSITION_KM, BAD_CLOCK_US
                    );
                    if with_velocity {
                        write_velocity_record(out, site, None, None, None)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The SP3 epoch-line / line-1 calendar fields, `YYYY MM DD HH MM
    /// SS.SSSSSSSS`.
    ///
    /// The record carries no time scale of its own, so the epoch must be in the
    /// scale the header states, and it must be a split-Julian-date instant (see
    /// [`Sp3WriteError::EpochRepresentationUnsupported`] for what that excludes
    /// and why). `epoch_index` names the epoch in any error, including for
    /// header line 1, whose calendar fields are the first epoch's.
    ///
    /// The statement is then checked by reading it: the candidate calendar
    /// fields are formatted, read back through the same civil conversion
    /// [`Sp3::parse`] runs on an epoch line, and the instant that yields must be
    /// the instant the product holds - exactly, as a value. Nothing here is a
    /// tolerance. A number of the last-place noise a physical bound would have
    /// to absorb is not roundoff at all but a different stored epoch: a
    /// fraction of `f64::from_bits(0.5f64.to_bits() + 1)` is 9.6 picoseconds
    /// past noon and states the same `12  0  0.00000000` a fraction of exactly
    /// `0.5` does, while an ordinary parsed epoch eleven seconds after J2000
    /// leaves the very same last-place residual behind in the tick count and
    /// must be written. Only the readback separates them.
    ///
    /// Equality is of the instant, not of the two `f64`s that carry it:
    /// `JulianDateSplit` defines the instant as `jd_whole + fraction`, and two
    /// splits that divide the same instant differently across the day boundary
    /// are the same epoch. See [`exact_instant`].
    fn format_calendar(
        &self,
        epoch: &Instant,
        epoch_index: usize,
    ) -> Result<String, Sp3WriteError> {
        if epoch.scale != self.header.time_scale {
            return Err(Sp3WriteError::EpochTimeScaleMismatch {
                epoch_index,
                epoch_scale: epoch.scale,
                header_scale: self.header.time_scale,
            });
        }
        let split = epoch
            .julian_date()
            .ok_or(Sp3WriteError::EpochRepresentationUnsupported { epoch_index })?;
        // `Sp3::epochs` is public and `JulianDateSplit`'s fields are public, so
        // a split this constructor never validated can arrive here. A
        // non-finite part has no calendar fields at all, and the tick and day
        // roundings below would turn it into a plausible-looking date.
        if !split.jd_whole.is_finite() || !split.fraction.is_finite() {
            return Err(Sp3WriteError::NonFinite {
                field: "epoch Julian date",
            });
        }
        let (primary, alternative) = epoch_candidates(split, self.header.time_system);
        if !(0..10i64.pow(CALENDAR_YEAR_COLUMNS as u32)).contains(&primary.year) {
            return Err(Sp3WriteError::YearNotRepresentable {
                epoch_index,
                year: primary.year,
            });
        }
        // The seconds field is formed from a whole tick count within a minute,
        // so it lands in its columns; the check states that rather than
        // assuming it.
        check_finite_width(
            primary.seconds,
            "epoch seconds",
            EPOCH_SECONDS_WIDTH,
            EPOCH_SECONDS_DECIMALS,
        )?;

        // What the refusal reports: the first statement that could be read back
        // at all, and how far the epoch it names sits from the stored one.
        let mut field_seconds = primary.seconds;
        let mut residual_s = f64::NAN;
        let mut measured = false;
        for fields in core::iter::once(primary).chain(alternative) {
            if !(0..10i64.pow(CALENDAR_YEAR_COLUMNS as u32)).contains(&fields.year) {
                continue;
            }
            let text = fields.record_text();
            let Some(restated) = restated_split(&text, self.header.time_system) else {
                // A statement this parser would not accept back - a `23:59:60`
                // label on a day that carries no leap second - is not a
                // statement of this epoch, whatever its arithmetic says.
                continue;
            };
            if exact_instant(restated) == exact_instant(split) {
                return Ok(text);
            }
            if !measured {
                field_seconds = fields.seconds;
                residual_s = elapsed_seconds(split, restated);
                measured = true;
            }
        }
        Err(Sp3WriteError::EpochNotRestatable {
            epoch_index,
            field_seconds,
            residual_s,
        })
    }
}

/// The satellite and epoch a record value belongs to, carried into any refusal
/// so a caller can find the cell rather than search for it.
#[derive(Debug, Clone, Copy)]
struct RecordSite {
    sat: GnssSatelliteId,
    epoch_index: usize,
}

/// A `P` record for a satellite the product holds a state for, plus the paired
/// `V` record for a velocity product.
fn write_state_record(
    out: &mut String,
    site: RecordSite,
    state: &Sp3State,
    with_velocity: bool,
) -> Result<(), Sp3WriteError> {
    let p = state.position;
    let x_km = p.x_m / KM_TO_M;
    let y_km = p.y_m / KM_TO_M;
    let z_km = p.z_m / KM_TO_M;
    let read_x = check_record_value(x_km, p.x_m, KM_TO_M, "position x", site)?;
    let read_y = check_record_value(y_km, p.y_m, KM_TO_M, "position y", site)?;
    let read_z = check_record_value(z_km, p.z_m, KM_TO_M, "position z", site)?;
    // An all-zero position is how SP3 says "no orbit here". A state whose
    // columns would land on it is a position the reader gives back as missing.
    if read_x == MISSING_POSITION_KM
        && read_y == MISSING_POSITION_KM
        && read_z == MISSING_POSITION_KM
    {
        return Err(Sp3WriteError::RecordReadsAsAbsent {
            field: "position",
            sat: site.sat,
            epoch_index: site.epoch_index,
            column_value: MISSING_POSITION_KM,
        });
    }
    let clk = clock_column(None, state.clock_s, US_TO_S, "clock", site)?;
    let sat = site.sat;
    let _ = write!(out, "P{sat}{x_km:14.6}{y_km:14.6}{z_km:14.6}{clk:14.6}");
    write_record_flags(out, state.flags);
    out.push('\n');
    if with_velocity {
        write_velocity_record(out, site, state.velocity, None, state.clock_rate_s_s)?;
    }
    Ok(())
}

/// A `P` record for a satellite the product holds only a clock for: the
/// missing-orbit sentinel with the record's own clock beside it.
fn write_clock_record(
    out: &mut String,
    site: RecordSite,
    record: &Sp3ClockRecord,
    with_velocity: bool,
) -> Result<(), Sp3WriteError> {
    let clk = clock_column(
        Some(record.clock_us),
        Some(record.clock_s),
        US_TO_S,
        "clock",
        site,
    )?;
    let sat = site.sat;
    let _ = write!(
        out,
        "P{sat}{:14.6}{:14.6}{:14.6}{clk:14.6}",
        MISSING_POSITION_KM, MISSING_POSITION_KM, MISSING_POSITION_KM,
    );
    write_record_flags(out, record.flags);
    out.push('\n');
    if with_velocity {
        write_velocity_record(
            out,
            site,
            record.velocity,
            record.clock_rate_raw,
            record.clock_rate_s_s,
        )?;
    }
    Ok(())
}

/// The `V` record paired with a `P` record, carrying the velocity and clock
/// rate when the product holds them and the format's sentinels when it does not.
fn write_velocity_record(
    out: &mut String,
    site: RecordSite,
    velocity: Option<ItrfVelocityMS>,
    clock_rate_raw: Option<f64>,
    clock_rate_s_s: Option<f64>,
) -> Result<(), Sp3WriteError> {
    let (vx, vy, vz) = match velocity {
        Some(v) => {
            let vx = v.vx_m_s / DM_S_TO_M_S;
            let vy = v.vy_m_s / DM_S_TO_M_S;
            let vz = v.vz_m_s / DM_S_TO_M_S;
            let read_x = check_record_value(vx, v.vx_m_s, DM_S_TO_M_S, "velocity x", site)?;
            let read_y = check_record_value(vy, v.vy_m_s, DM_S_TO_M_S, "velocity y", site)?;
            let read_z = check_record_value(vz, v.vz_m_s, DM_S_TO_M_S, "velocity z", site)?;
            // The all-zero velocity vector is the format's "no velocity here".
            if read_x == MISSING_VELOCITY_DM_S
                && read_y == MISSING_VELOCITY_DM_S
                && read_z == MISSING_VELOCITY_DM_S
            {
                return Err(Sp3WriteError::RecordReadsAsAbsent {
                    field: "velocity",
                    sat: site.sat,
                    epoch_index: site.epoch_index,
                    column_value: MISSING_VELOCITY_DM_S,
                });
            }
            (vx, vy, vz)
        }
        None => (
            MISSING_VELOCITY_DM_S,
            MISSING_VELOCITY_DM_S,
            MISSING_VELOCITY_DM_S,
        ),
    };
    let rate = clock_column(
        clock_rate_raw,
        clock_rate_s_s,
        CLOCK_RATE_TO_S_PER_S,
        "clock rate",
        site,
    )?;
    let sat = site.sat;
    let _ = writeln!(out, "V{sat}{vx:14.6}{vy:14.6}{vz:14.6}{rate:14.6}");
    Ok(())
}

/// The column a clock or clock-rate field is written in.
///
/// `stored` is the value in the product's own units (seconds, or seconds per
/// second) and `scale` is the factor the reader applies to the column to get
/// back to them. `native` is the column's own number as the product retained
/// it: a clock-only record keeps the microseconds it was read from, exactly as
/// the ASCII stated them.
///
/// The retained native value is what the column carries whenever it is both
/// restatable and consistent - it restates itself through the column, and the
/// reader's multiplication takes it back to `stored`. That is what makes a
/// product read from a file write back unchanged. When the two disagree, the
/// value in the product's own units is the one every consumer reads and the one
/// the column states; a native value derived from it rather than read can sit
/// one unit in the last place away, and that difference is not a second
/// estimate to preserve. A retained native value with no stored value beside it
/// is a different matter: no column states both a number and an absence, and
/// the mismatch is reported.
///
/// An absent value is written as the bad-clock sentinel, which is how SP3 says
/// "no estimate"; a present one that would land on it is refused instead.
fn clock_column(
    native: Option<f64>,
    stored: Option<f64>,
    scale: f64,
    field: &'static str,
    site: RecordSite,
) -> Result<f64, Sp3WriteError> {
    let Some(stored) = stored else {
        if let Some(native) = native {
            return Err(Sp3WriteError::RecordFieldsDisagree {
                field,
                sat: site.sat,
                epoch_index: site.epoch_index,
                stored: None,
                native: Some(native),
            });
        }
        return Ok(BAD_CLOCK_US);
    };
    let column = match native {
        Some(native)
            if restates_itself(native) && scaled_bits(native, scale) == stored.to_bits() =>
        {
            native
        }
        _ => stored / scale,
    };
    let read = check_record_value(column, stored, scale, field, site)?;
    if read.abs() >= BAD_CLOCK_US {
        return Err(Sp3WriteError::RecordReadsAsAbsent {
            field,
            sat: site.sat,
            epoch_index: site.epoch_index,
            column_value: column,
        });
    }
    Ok(column)
}

/// Whether a record column can restate `value` itself, before any unit factor.
fn restates_itself(value: f64) -> bool {
    column_read_back(value).is_some_and(|read| read.to_bits() == value.to_bits())
}

/// The bits of `column * scale`, the product the reader forms from a column.
fn scaled_bits(column: f64, scale: f64) -> u64 {
    (column * scale).to_bits()
}

/// The value the parser reads back from a record column holding `value`, or
/// `None` when the column cannot hold it at all.
///
/// The parser trims the fixed column and parses what is left, so the text below
/// is the column's content without its padding.
fn column_read_back(value: f64) -> Option<f64> {
    if !value.is_finite() {
        return None;
    }
    let decimals = RECORD_VALUE_DECIMALS;
    let text = format!("{value:.decimals$}");
    if text.len() > RECORD_VALUE_WIDTH {
        return None;
    }
    text.parse::<f64>().ok()
}

/// A `%f` base column: the value in its canonical `Fw.d` form, or the blank
/// field the parser reads back as "no base declared".
fn optional_base(
    base: Option<f64>,
    field: &'static str,
    columns: usize,
    decimals: usize,
) -> Result<String, Sp3WriteError> {
    match base {
        Some(base) => {
            check_exact(base, field, columns, decimals)?;
            Ok(format!("{base:columns$.decimals$}"))
        }
        None => Ok(" ".repeat(columns)),
    }
}

fn write_record_flags(out: &mut String, flags: Sp3Flags) {
    let last_col = if flags.orbit_predicted {
        Some(79)
    } else if flags.maneuver {
        Some(78)
    } else if flags.clock_predicted {
        Some(75)
    } else if flags.clock_event {
        Some(74)
    } else {
        None
    };
    let Some(last_col) = last_col else {
        return;
    };

    for col in 60..=last_col {
        out.push(match col {
            74 if flags.clock_event => 'E',
            75 if flags.clock_predicted => 'P',
            78 if flags.maneuver => 'M',
            79 if flags.orbit_predicted => 'P',
            _ => ' ',
        });
    }
}

/// A fixed-column text field must be printable ASCII, fit its columns, and
/// survive the reader.
///
/// Anything else changes what the line says: a line break or a control byte
/// turns one record into two or into something no reader can column-index, a
/// non-ASCII byte makes the byte offsets stop matching the columns, and an
/// over-wide value pushes every later field off its own columns. The reader
/// trims the column it takes a label from, so a label padded with its own
/// whitespace comes back a different string and is reported rather than
/// quietly shortened.
fn check_text(value: &str, field: &'static str, columns: usize) -> Result<(), Sp3WriteError> {
    if !value.is_ascii() || value.bytes().any(|byte| byte < b' ' || byte == 0x7f) {
        return Err(Sp3WriteError::TextNotColumnSafe {
            field,
            value: value.to_string(),
        });
    }
    if value.len() > columns {
        return Err(Sp3WriteError::TextTooWide {
            field,
            columns,
            value: value.to_string(),
        });
    }
    if value.trim() != value {
        return Err(Sp3WriteError::TextNotColumnStable {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

/// An optional descriptor's text, or the blank columns that state its absence.
///
/// The reader spells "this field is absent" with blank columns, so a descriptor
/// holding a blank string has no form of its own: writing it would come back as
/// `None` and the distinction the product draws would be gone.
fn optional_text<'a>(
    value: Option<&'a str>,
    field: &'static str,
    columns: usize,
) -> Result<&'a str, Sp3WriteError> {
    let Some(value) = value else {
        return Ok("");
    };
    check_text(value, field, columns)?;
    if value.is_empty() {
        return Err(Sp3WriteError::BlankDescriptor {
            field,
            value: value.to_string(),
        });
    }
    Ok(value)
}

/// A comment record's text.
///
/// The reader keeps the text from column 4 on with its trailing blanks removed,
/// and keeps only records that carry text - the rest are the format's
/// structural padding. So a comment that is empty, or that ends in whitespace,
/// does not come back as itself.
fn check_comment(value: &str, index: usize) -> Result<(), Sp3WriteError> {
    if !value.is_ascii() || value.bytes().any(|byte| byte < b' ' || byte == 0x7f) {
        return Err(Sp3WriteError::TextNotColumnSafe {
            field: "comment",
            value: value.to_string(),
        });
    }
    if value.len() > COMMENT_COLUMNS {
        return Err(Sp3WriteError::TextTooWide {
            field: "comment",
            columns: COMMENT_COLUMNS,
            value: value.to_string(),
        });
    }
    if value.is_empty() {
        return Err(Sp3WriteError::EmptyComment {
            index,
            value: value.to_string(),
        });
    }
    if value.trim_end() != value {
        return Err(Sp3WriteError::TextNotColumnStable {
            field: "comment",
            value: value.to_string(),
        });
    }
    Ok(())
}

/// A position product may hold no velocity and no clock rate.
///
/// `#dP` declares position records only, so the writer emits no `V` record to
/// carry them and the stored values would be dropped without a column to name
/// them. Reported per field, so the caller learns which one the product holds.
fn check_position_product_state(
    sat: GnssSatelliteId,
    epoch_index: usize,
    velocity: Option<ItrfVelocityMS>,
    rates: [Option<f64>; 2],
) -> Result<(), Sp3WriteError> {
    if velocity.is_some() {
        return Err(Sp3WriteError::VelocityStateInPositionProduct {
            field: "velocity",
            sat,
            epoch_index,
        });
    }
    if rates.iter().any(Option::is_some) {
        return Err(Sp3WriteError::VelocityStateInPositionProduct {
            field: "clock rate",
            sat,
            epoch_index,
        });
    }
    Ok(())
}

/// A fixed-column integer field must fit its columns.
fn check_integer(value: u64, field: &'static str, columns: usize) -> Result<(), Sp3WriteError> {
    let limit = 10u64.saturating_pow(columns as u32);
    if value >= limit {
        return Err(Sp3WriteError::IntegerTooWide {
            field,
            columns,
            value,
        });
    }
    Ok(())
}

/// A fixed-column numeric field must be finite and its `Fw.d` form must fit.
///
/// This is the width half of the check; [`check_exact`] adds the requirement
/// that the form state the number itself.
fn check_finite_width(
    value: f64,
    field: &'static str,
    columns: usize,
    decimals: usize,
) -> Result<(), Sp3WriteError> {
    if !value.is_finite() {
        return Err(Sp3WriteError::NonFinite { field });
    }
    if format!("{value:.decimals$}").len() > columns {
        return Err(Sp3WriteError::NumberTooWide {
            field,
            columns,
            decimals,
            value,
        });
    }
    Ok(())
}

/// A header numeric field must additionally re-read as the number it holds.
///
/// Header values name the product's cadence, start, and accuracy bases; a
/// rounded one describes a different product. The parser keeps any finite value
/// its source field expressed, so this is where a value too precise for the
/// canonical columns is reported.
fn check_exact(
    value: f64,
    field: &'static str,
    columns: usize,
    decimals: usize,
) -> Result<(), Sp3WriteError> {
    check_finite_width(value, field, columns, decimals)?;
    if !validate::representable_in_fixed_field(value, Some(columns), decimals) {
        return Err(Sp3WriteError::PrecisionNotRepresentable {
            field,
            columns,
            decimals,
            value,
        });
    }
    Ok(())
}

/// A position, velocity, clock, or clock-rate column (`F14.6`), checked against
/// the value the product stores behind it.
///
/// `column_value` is the number this writer would print, `stored` is the value
/// the product holds, and `scale` is the factor the reader applies to the
/// column to get back to the product's units (`KM_TO_M` for a position,
/// `US_TO_S` for a clock, and so on). The check is that whole product: format
/// the column, read it back the way the parser does, apply the reader's factor,
/// and require the value the product holds, bit for bit - so a signed zero
/// keeps its sign and a value the column would round is refused.
///
/// The division that formed `column_value` is deliberately not required to be
/// exact on its own. A position read from a file is kept in meters as
/// `km * 1000`, and dividing it back can land one unit in the last place from
/// the kilometres the file stated; those kilometres are still what the column
/// prints and what the reader multiplies back to the stored meters, and
/// refusing them would refuse every product this parser reads.
///
/// Returns the value the reader takes back from the column, which is what the
/// format's sentinel checks compare against.
fn check_record_value(
    column_value: f64,
    stored: f64,
    scale: f64,
    field: &'static str,
    site: RecordSite,
) -> Result<f64, Sp3WriteError> {
    if !column_value.is_finite() {
        return Err(Sp3WriteError::RecordValueNonFinite {
            field,
            sat: site.sat,
            epoch_index: site.epoch_index,
        });
    }
    let Some(read) = column_read_back(column_value) else {
        return Err(Sp3WriteError::RecordValueTooWide {
            field,
            sat: site.sat,
            epoch_index: site.epoch_index,
            columns: RECORD_VALUE_WIDTH,
            decimals: RECORD_VALUE_DECIMALS,
            column_value,
        });
    };
    if scaled_bits(read, scale) != stored.to_bits() {
        return Err(Sp3WriteError::RecordValueNotRepresentable {
            field,
            sat: site.sat,
            epoch_index: site.epoch_index,
            columns: RECORD_VALUE_WIDTH,
            decimals: RECORD_VALUE_DECIMALS,
            stored,
            column_value,
        });
    }
    Ok(read)
}

/// The civil fields one candidate epoch record states.
#[derive(Debug, Clone, Copy)]
struct EpochFields {
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    /// Seconds of the minute, formed from a whole tick count.
    seconds: f64,
}

impl EpochFields {
    /// The record text itself, `YYYY MM DD HH MM SS.SSSSSSSS`.
    ///
    /// This is the only place the epoch columns are formed, so the bytes the
    /// readback check reads are the bytes the file carries.
    fn record_text(self) -> String {
        format!(
            "{year:4} {month:>2} {day:>2} {hour:>2} {minute:>2} {seconds:11.8}",
            year = self.year,
            month = self.month,
            day = self.day,
            hour = self.hour,
            minute = self.minute,
            seconds = self.seconds,
        )
    }
}

/// The calendar statements an epoch record could make about one instant, most
/// faithful first.
///
/// The split is reduced to the civil day the instant falls in and a whole count
/// of the 10-nanosecond ticks the `F11.8` seconds field resolves within that
/// day. For the `*.5` midnight boundary `super::civil_to_julian_split`
/// produces, that reduction is exactly its inverse; it is not restricted to it.
/// [`JulianDateSplit`] accepts any finite `jd_whole` with a residual inside one
/// day, so noon held as `(2451545.0, 0.0)`, a quarter day later held as
/// `(2451545.25, 0.0)`, and a midnight held as the next boundary less a whole
/// day are all ordinary epochs a record states - and for each of them the day
/// the instant falls in is not the day `jd_whole` alone rounds to. Offering the
/// wrong date for such an epoch would make the readback refuse a record that
/// states the instant perfectly, so the date comes from both parts. See
/// [`midnight_decomposition`].
///
/// Neither rounding there is trusted: what the writer emits is decided by
/// reading the candidate back, not by how close the roundings came.
///
/// Most instants have exactly one statement. A UTC-like system has two for one
/// second of the year, because `split_julian_date` carries a `:60` leap-second
/// label past the day boundary: a fraction of exactly `1.0` is the label
/// itself, and a label with a fractional part lands in the next day's small
/// fraction. Both statements are offered in that window, ordered so the one
/// whose split the parser would rebuild unchanged is offered first - the pair
/// the parser builds for a label is the one whose own `jd_whole` boundary opens
/// the day *before* the one the instant falls in. The ordinary decomposition is
/// always one of the two. A statement the parser would not accept back - a
/// `:60` label on a day that carries no leap second - is discarded by the
/// readback rather than by a rule here.
fn epoch_candidates(
    split: JulianDateSplit,
    time_system: Sp3TimeSystem,
) -> (EpochFields, Option<EpochFields>) {
    let (day, ticks, boundary_day) = midnight_decomposition(split);
    let ordinary = civil_fields(day, ticks);
    if !is_utc_like(time_system) || ticks >= SP3_TIME_TICKS_PER_SECOND {
        return (ordinary, None);
    }
    // Less than a second into a day: the window this day's `00:00:00.xx` shares
    // with the previous day's `23:59:60.xx` label.
    let label = leap_fields(day - 1, ticks);
    if boundary_day < day {
        // The split carries the instant on the previous day's boundary, a whole
        // day's worth of residual or a little more past it. That is the pair
        // the parser built for a label line, so the label is offered before the
        // next-day statement of the same instant.
        return (label, Some(ordinary));
    }
    if ticks > 0 {
        // The ordinary split of the same window. The parser reaches one from an
        // `00:00:00.xx` line and from the previous day's `23:59:60.xx` label,
        // and the two are not the same `f64` pair: the carry that folded the
        // label into this day rounded the fraction. The ordinary form is tried
        // first, so the label is only written when it is the statement that
        // restates the stored split.
        return (ordinary, Some(label));
    }
    (ordinary, None)
}

/// The civil day a split Julian date falls in, the whole tick count within that
/// day, and the day the split's own `jd_whole` boundary opens - as
/// `(day, ticks, boundary_day)`, with both days Julian Day Numbers and `ticks`
/// in `0..SP3_TIME_TICKS_PER_DAY`.
///
/// The instant is `jd_whole + fraction`, and that sum does not fit one `f64`:
/// the last place of a single-`f64` day number is 46 microseconds around a
/// modern Julian date, thousands of the ticks the seconds field resolves.
/// Recombining the parts first would quantize every epoch onto that grid before
/// the day and the clock were read off it, so nothing here does.
/// [`exact_instant`] divides the sum into a head and a tail that together lose
/// nothing, the half day that moves the noon origin of a Julian date to civil
/// midnight is added to the head the same way, and the day number is taken from
/// the head while both tails stay with the within-day residual. The residual
/// that results resolves the instant to well under a tick across the whole
/// writable calendar, and the record it suggests is still decided by reading it
/// back.
///
/// A `JulianDateSplit` is only required to be finite, and `Sp3::epochs` is
/// public, so either day number can be astronomically larger than any calendar
/// a four-digit year states. The Fliegel-Van Flandern inverse is integer
/// arithmetic that a saturated `i64` would overflow, so both are bounded here -
/// far outside the writable calendar, which the year check then refuses by
/// name.
fn midnight_decomposition(split: JulianDateSplit) -> (i64, i64, i64) {
    let (jd_head, jd_tail) = exact_instant(split);
    let (head, half_day_tail) = two_sum(jd_head, 0.5);
    let boundary = head.floor();
    let residual = (head - boundary) + (half_day_tail + jd_tail);
    // The residual is within one day of the boundary either way, so the tick
    // count is within one day's worth of zero and the carry moves the day by at
    // most one - in either direction, which is what `div_euclid` is for.
    let ticks = (residual * SECONDS_PER_DAY * SP3_TIME_TICKS_PER_SECOND as f64).round() as i64;
    let day = clamped_day_number(boundary) + ticks.div_euclid(SP3_TIME_TICKS_PER_DAY);

    // The day the split's own boundary opens, from `jd_whole` alone and on the
    // same terms. `floor` is corrected by the tail the half-day addition
    // dropped, so a boundary that lands exactly on a day number from below
    // still names the day below it.
    let (boundary_head, boundary_tail) = two_sum(split.jd_whole, 0.5);
    let mut own_boundary = boundary_head.floor();
    if own_boundary == boundary_head && boundary_tail < 0.0 {
        own_boundary -= 1.0;
    }

    (
        day,
        ticks.rem_euclid(SP3_TIME_TICKS_PER_DAY),
        clamped_day_number(own_boundary),
    )
}

/// A whole day number bounded to the range the calendar inverse is evaluated
/// at.
fn clamped_day_number(days: f64) -> i64 {
    days.clamp(-JDN_CONVERSION_LIMIT, JDN_CONVERSION_LIMIT) as i64
}

/// Calendar fields for a Julian Day Number and a whole tick count within it.
fn civil_fields(jdn: i64, ticks: i64) -> EpochFields {
    let (year, month, day) = civil_from_jdn(jdn);
    let hour = ticks / SP3_TIME_TICKS_PER_HOUR;
    let rem = ticks % SP3_TIME_TICKS_PER_HOUR;
    let minute = rem / SP3_TIME_TICKS_PER_MINUTE;
    let seconds = (rem % SP3_TIME_TICKS_PER_MINUTE) as f64 / SP3_TIME_TICKS_PER_SECOND as f64;
    EpochFields {
        year,
        month,
        day,
        hour,
        minute,
        seconds,
    }
}

/// The `23:59:60` leap-second statement of a day, `ticks_into_leap` ticks in.
fn leap_fields(jdn: i64, ticks_into_leap: i64) -> EpochFields {
    let (year, month, day) = civil_from_jdn(jdn);
    EpochFields {
        year,
        month,
        day,
        hour: 23,
        minute: 59,
        seconds: 60.0 + ticks_into_leap as f64 / SP3_TIME_TICKS_PER_SECOND as f64,
    }
}

/// The instant an emitted epoch record states, read the way [`Sp3::parse`]
/// reads an epoch line.
///
/// The record's six fields are taken from the text this writer produced, put
/// through the same civil validation the parser applies under the header's own
/// second policy, and converted by the parser's own
/// `super::civil_to_julian_split`. `None` when the parser would not accept the
/// line back at all.
fn restated_split(text: &str, time_system: Sp3TimeSystem) -> Option<JulianDateSplit> {
    let mut fields = text.split_whitespace();
    let year = fields.next()?.parse::<i64>().ok()?;
    let month = fields.next()?.parse::<i64>().ok()?;
    let day = fields.next()?.parse::<i64>().ok()?;
    let hour = fields.next()?.parse::<i64>().ok()?;
    let minute = fields.next()?.parse::<i64>().ok()?;
    let seconds = fields.next()?.parse::<f64>().ok()?;
    if fields.next().is_some() {
        return None;
    }
    let civil = validate::civil_datetime_with_second_policy(
        year,
        month,
        day,
        hour,
        minute,
        seconds,
        time_system.civil_second_policy(),
    )
    .ok()?;
    super::civil_to_julian_split(civil).ok()
}

/// The exact value of a split Julian date, as the two `f64`s of an error-free
/// sum.
///
/// [`JulianDateSplit`] defines the instant as `jd_whole + fraction`, and that
/// sum does not fit one `f64`: the day number spends twenty-two bits before the
/// point and the fraction carries the rest of the day below it. [`two_sum`]
/// returns a head and a tail whose sum is the exact value, and the tail is
/// itself representable, so no information is lost and no tolerance is
/// introduced. The pair is canonical: two splits are equal here exactly when
/// they denote the same real instant, whichever side of the day boundary each
/// carries it on - so the `*.5` midnight convention this writer emits and a
/// split holding the same moment as the next day plus a negative residual
/// compare equal, while a fraction one unit in the last place from `0.5`
/// compares different from `0.5`.
///
/// A non-finite part yields `NaN`, which compares unequal to everything
/// including itself - the refusal such an epoch deserves.
fn exact_instant(split: JulianDateSplit) -> (f64, f64) {
    two_sum(split.jd_whole, split.fraction)
}

/// The error-free sum of two `f64`s, as a head and a tail.
///
/// Fast2Sum: the smaller magnitude is added to the larger, then what the
/// rounding dropped is recovered by subtracting the larger back out. `head`
/// is the rounded sum, `tail` is representable, and `head + tail` is the exact
/// real value of `a + b`, so nothing is lost by carrying the pair instead of
/// the sum. A non-finite input yields a `NaN` tail.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let (large, small) = if a.abs() >= b.abs() { (a, b) } else { (b, a) };
    let head = large + small;
    let tail = small - (head - large);
    (head, tail)
}

/// Seconds from the instant `stated` names to the instant `stored` names.
///
/// Both are reduced by [`exact_instant`] first and differenced head against
/// head and tail against tail, so a difference far below the resolution of
/// either Julian date still survives into the refusal a reader has to act on.
fn elapsed_seconds(stored: JulianDateSplit, stated: JulianDateSplit) -> f64 {
    let (stored_head, stored_tail) = exact_instant(stored);
    let (stated_head, stated_tail) = exact_instant(stated);
    ((stored_head - stated_head) + (stored_tail - stated_tail)) * SECONDS_PER_DAY
}

fn is_utc_like(time_system: Sp3TimeSystem) -> bool {
    matches!(time_system, Sp3TimeSystem::Glonass | Sp3TimeSystem::Utc)
}
