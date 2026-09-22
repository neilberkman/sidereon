//! ANTEX 1.4 receiver and satellite antenna parser and writer.
//!
//! The parser owns the byte/record grammar for the antenna calibration blocks
//! used by PPP and RTK correction paths. Values are stored in SI units:
//! PCO/PCV are meters, azimuth and zenith grids are degrees. A millimetre field
//! becomes meters as `mm * 1e-3`, the operation RTKLIB's `readantex` applies
//! (`decodef` in `rtkcmn.c`), so the stored meters equal that reader's bit for
//! bit. Dividing by `1000.0` instead differs by one unit in the last place for
//! about one value in seven; both keep every two-decimal millimetre value
//! recoverable.
//!
//! Every record the format defines is retained: the header's version, system,
//! PCV type and reference antenna and comments; each antenna's calibration
//! method records, `DAZI` and zenith grid, validity bounds with their exact
//! fractional seconds, SINEX code, comments and frequency sections in file
//! order, including optional RMS sections; comments outside the header and
//! the blocks; and the file order of the antenna blocks. A record the source
//! does not carry stays absent, and the writer does not invent it.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use crate::antenna;
use crate::constants::MM_PER_M;
use crate::format::columns::{field, fortran_f64, raw_field};
use crate::format::{Diagnostics, RecordRef, Skip, SkipReason};
use crate::validate::{self, FieldError};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Meters per millimetre, applied by multiplication as RTKLIB `decodef` does.
const M_PER_MM: f64 = 1e-3;

/// Parsed ANTEX antenna calibration product.
#[derive(Debug, Clone, PartialEq)]
pub struct Antex {
    /// Header records.
    pub header: AntexHeader,
    /// `COMMENT` records after `END OF HEADER` outside every antenna block, in
    /// file order, each placed by the number of blocks before it.
    pub outer_comments: Vec<OuterComment>,
    /// Latest completed antenna block for each trimmed `TYPE / SERIAL NO` id.
    /// A later block with the same id replaces this entry; all blocks remain
    /// available through [`Antex::antenna_intervals`].
    pub antennas: BTreeMap<String, Antenna>,
    antenna_intervals: BTreeMap<String, Vec<Antenna>>,
    /// File order of every antenna block, as its id and its index in that id's
    /// interval list.
    block_order: Vec<(String, usize)>,
    /// Count of records skipped or found inconsistent during a forgiving parse
    /// (a corrupt PCV grid value, an unrecognized grid-row head, a line outside
    /// any record the format defines, a `# OF FREQUENCIES` count that disagrees
    /// with the frequency sections read, a block or section not closed by its
    /// own end record); each is surfaced as a typed [`Skip`]
    /// in the parser's [`Diagnostics`]. A clean file parses with
    /// `skipped_records == 0`; a non-zero count lets a caller tell a pristine
    /// product apart from one that carried a malformed record without aborting
    /// the whole parse. No fabricated sample is emitted in place of a skipped
    /// one. Read it through [`Antex::skipped_records`]. Mirrors
    /// [`crate::atmosphere::Ionex::skipped_records`].
    skipped_records: usize,
}

/// ANTEX header records.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AntexHeader {
    /// `ANTEX VERSION / SYST`, or `None` when the source has no such record.
    pub version: Option<AntexVersion>,
    /// `PCV TYPE / REFANT`, or `None` when the source has no such record.
    pub pcv_type: Option<PcvTypeRecord>,
    /// Text of the `COMMENT` records before `END OF HEADER`, or before the
    /// first antenna block when the source has no `END OF HEADER`, in file
    /// order, trailing blanks removed.
    pub comments: Vec<String>,
    /// Whether the source carries `END OF HEADER`.
    pub end_of_header: bool,
}

/// `ANTEX VERSION / SYST` record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AntexVersion {
    /// Format version from columns 1-8 (`F8.1`).
    pub version: f64,
    /// Satellite system flag from column 21 (`G`, `R`, `E`, `C`, `J`, `S` or
    /// `M` in ANTEX 1.4), or `None` when that column is blank.
    pub system: Option<char>,
}

/// Phase center variation type from `PCV TYPE / REFANT` column 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcvType {
    /// `A`: absolute values.
    Absolute,
    /// `R`: values relative to a reference antenna.
    Relative,
}

/// `PCV TYPE / REFANT` record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcvTypeRecord {
    /// Whether the file's values are absolute or relative.
    pub pcv_type: PcvType,
    /// Reference antenna type from columns 21-40, trimmed; empty when blank.
    pub reference_antenna_type: String,
    /// Reference antenna serial number from columns 41-60, trimmed; empty when
    /// blank.
    pub reference_antenna_serial: String,
}

/// Reference antenna ANTEX 1.4 names for relative values with a blank type.
pub const DEFAULT_RELATIVE_REFERENCE_ANTENNA: &str = "AOAD/M_T";

impl PcvTypeRecord {
    /// Antenna type the values are relative to: the stated type, or
    /// [`DEFAULT_RELATIVE_REFERENCE_ANTENNA`] when a relative file leaves it
    /// blank. `None` for absolute values.
    pub fn reference_antenna(&self) -> Option<&str> {
        match self.pcv_type {
            PcvType::Absolute => None,
            PcvType::Relative if self.reference_antenna_type.is_empty() => {
                Some(DEFAULT_RELATIVE_REFERENCE_ANTENNA)
            }
            PcvType::Relative => Some(self.reference_antenna_type.as_str()),
        }
    }
}

/// Receiver or satellite antenna block.
#[derive(Debug, Clone, PartialEq)]
pub struct Antenna {
    /// Trimmed body of the `TYPE / SERIAL NO` record, also used as the key in
    /// the parsed antenna views and as the identifier in lookup errors.
    pub id: String,
    /// Classification assigned from the trimmed serial by the
    /// `is_satellite_serial` helper.
    pub kind: AntennaKind,
    /// Trimmed first 20-byte field of the `TYPE / SERIAL NO` record.
    pub antenna_type: String,
    /// Trimmed 20..40-byte field of the `TYPE / SERIAL NO` record. It drives
    /// receiver/satellite classification and satellite PRN lookup.
    pub serial: String,
    /// Text of the `COMMENT` records between `START OF ANTENNA` and
    /// `TYPE / SERIAL NO`, in file order, trailing blanks removed. The writer
    /// restates them in that place.
    pub leading_comments: Vec<String>,
    /// Every `METH / BY / # / DATE` record of the block, in file order.
    pub calibrations: Vec<Calibration>,
    /// Value of the `DAZI` record, in degrees; `None` when the block has no
    /// such record.
    pub dazi_deg: Option<f64>,
    /// The `ZEN1 / ZEN2 / DZEN` record; `None` when the block has no such
    /// record, in which case the block has no PCV values.
    pub zenith_grid: Option<ZenithGrid>,
    /// Whether the block carries a `# OF FREQUENCIES` record. The count itself
    /// is not stored: the writer states the number of frequency sections.
    pub has_frequency_count: bool,
    /// Nonblank trimmed `SINEX CODE` content, or `None` when that record is
    /// absent or blank.
    pub sinex_code: Option<String>,
    /// GPS-time instant from `VALID FROM`, used as an inclusive lower bound by
    /// [`Antenna::valid_at`].
    pub valid_from: Option<AntexDateTime>,
    /// GPS-time instant from `VALID UNTIL`, used as an inclusive upper bound by
    /// [`Antenna::valid_at`].
    pub valid_until: Option<AntexDateTime>,
    /// Text of the `COMMENT` records inside the block after `TYPE / SERIAL NO`,
    /// in file order, trailing blanks removed. The writer restates them at the
    /// position ANTEX 1.4 assigns them, after `SINEX CODE`; one read from inside
    /// a frequency section, where the format allows none, moves there.
    pub comments: Vec<String>,
    /// Completed frequency sections in file order. A label that appears in
    /// more than one section keeps every section; lookup by that label is
    /// refused as ambiguous unless the sections are identical.
    pub frequencies: Vec<Frequency>,
}

/// `ZEN1 / ZEN2 / DZEN` record, in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZenithGrid {
    /// `ZEN1`: first zenith (receiver) or nadir (satellite) angle of each row;
    /// it anchors recovered PCV sample zeniths and is the lower bound checked
    /// by [`Antenna::pcv`].
    pub start_deg: f64,
    /// `ZEN2`: last angle of the declared grid, the inclusive upper bound
    /// checked by [`Antenna::pcv`].
    pub end_deg: f64,
    /// `DZEN`: increment between successive values of a row.
    pub step_deg: f64,
}

/// A `COMMENT` record after `END OF HEADER` that lies outside every antenna
/// block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OuterComment {
    /// Number of antenna blocks, in file order, that precede the comment.
    pub blocks_before: usize,
    /// Comment text, trailing blanks removed.
    pub text: String,
}

/// `METH / BY / # / DATE` record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Calibration {
    /// Calibration method from columns 1-20 (`CHAMBER`, `FIELD`, `ROBOT`,
    /// `COPIED`, `CONVERTED` or blank), trailing blanks removed.
    pub method: String,
    /// Agency from columns 21-40, trailing blanks removed.
    pub agency: String,
    /// Number of individual antennas calibrated, columns 41-46 (`I6`); `None`
    /// when blank.
    pub antennas_calibrated: Option<u32>,
    /// Date text from columns 51-60, trailing blanks removed.
    pub date: String,
}

/// ANTEX antenna block role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntennaKind {
    /// Assigned when the trimmed serial does not match the three-byte satellite
    /// PRN pattern accepted by the `is_satellite_serial` helper.
    Receiver,
    /// Assigned for a serial consisting of one uppercase ASCII letter and two
    /// ASCII digits; satellite lookup requires this classification.
    Satellite,
}

/// Frequency-specific PCO/PCV calibration block.
#[derive(Debug, Clone, PartialEq)]
pub struct Frequency {
    /// Trimmed label from the `START OF FREQUENCY` record and the key used by
    /// antenna frequency lookup.
    pub frequency: String,
    /// Finite `NORTH / EAST / UP` PCO values converted from ANTEX millimeters
    /// to meters and kept in north/east/up order.
    pub pco_m: [f64; 3],
    /// PCV samples recovered in row/token order, with values expressed in
    /// meters; lookup partitions them by [`PcvGrid`] for interpolation.
    pub pcv_samples: Vec<PcvSample>,
    /// The `START OF FREQ RMS` section for this frequency, when the file has
    /// one.
    pub rms: Option<FrequencyRms>,
}

/// RMS values of one frequency's calibration, from a `START OF FREQ RMS`
/// section.
#[derive(Debug, Clone, PartialEq)]
pub struct FrequencyRms {
    /// RMS of the `NORTH / EAST / UP` eccentricities in meters, or `None`
    /// when the section has no such record.
    pub pco_m: Option<[f64; 3]>,
    /// RMS of the pattern values in meters, placed on the antenna's grid the
    /// same way as [`Frequency::pcv_samples`].
    pub pcv_samples: Vec<PcvSample>,
}

/// One phase-center-variation grid value.
#[derive(Debug, Clone, PartialEq)]
pub struct PcvSample {
    /// Whether the source row was headed by `NOAZI` or by a numeric azimuth.
    pub grid: PcvGrid,
    /// `None` for `NOAZI` samples; otherwise the unnormalized parsed angle
    /// from the source row head, in degrees.
    pub azimuth_deg: Option<f64>,
    /// Zenith coordinate computed from the antenna grid start, step, and this
    /// value's row position.
    pub zenith_deg: f64,
    /// PCV token converted from ANTEX millimeters to meters; malformed tokens
    /// are skipped instead of producing a sample.
    pub value_m: f64,
}

/// PCV grid type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcvGrid {
    /// A `NOAZI` row used for zenith-only interpolation or fallback.
    NoAzimuth,
    /// A numeric-azimuth row whose samples are grouped by azimuth for lookup.
    Azimuth,
}

/// GPS-time calendar instant from `VALID FROM` / `VALID UNTIL`.
///
/// ANTEX states validity bounds in GPS time, which has no leap-second label,
/// so the second is `0..=59`. The `F13.7` seconds field is kept exactly: its
/// fraction is a [`SecondFraction`], which holds every decimal the field can
/// state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AntexDateTime {
    /// Calendar year in `0..=9999`.
    pub year: i32,
    /// One-based month in `1..=12`.
    pub month: u8,
    /// Day of the month.
    pub day: u8,
    /// Hour in `0..=23`.
    pub hour: u8,
    /// Minute in `0..=59`.
    pub minute: u8,
    /// Whole second in `0..=59`.
    pub second: u8,
    /// Fraction of the second.
    pub fraction: SecondFraction,
}

/// Exact decimal fraction of a second, `digits / 10^scale`, in `[0, 1)`.
///
/// It is kept normalized, with no trailing zero digit, so equal fractions have
/// equal representations. A thirteen-column seconds field with an exponent
/// can state a fraction far below a nanosecond (`1.2345678E-9`); the digits
/// and the power of ten hold it without rounding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SecondFraction {
    digits: u64,
    scale: u64,
}

impl SecondFraction {
    /// A zero fraction.
    pub const ZERO: Self = Self {
        digits: 0,
        scale: 0,
    };

    /// `digits / 10^scale`, or `None` when that is not below one second.
    pub fn new(digits: u64, scale: u64) -> Option<Self> {
        if digits == 0 {
            return Some(Self::ZERO);
        }
        if decimal_digits(digits) > scale {
            return None;
        }
        let (mut digits, mut scale) = (digits, scale);
        while digits.is_multiple_of(10) {
            digits /= 10;
            scale -= 1;
        }
        Some(Self { digits, scale })
    }

    /// A fraction of whole nanoseconds, or `None` for `1_000_000_000` or more.
    pub fn from_nanoseconds(nanoseconds: u32) -> Option<Self> {
        if nanoseconds >= 1_000_000_000 {
            return None;
        }
        Self::new(u64::from(nanoseconds), 9)
    }

    /// Significant digits: the fraction is `digits / 10^scale`.
    pub fn digits(self) -> u64 {
        self.digits
    }

    /// Power of ten dividing [`SecondFraction::digits`].
    pub fn scale(self) -> u64 {
        self.scale
    }

    /// Whole nanoseconds, when the fraction is a whole number of them.
    pub fn nanoseconds(self) -> Option<u32> {
        let shift = 9_u64.checked_sub(self.scale)?;
        u32::try_from(self.digits * 10_u64.pow(shift as u32)).ok()
    }

    /// Number of zero digits between the decimal point and the first
    /// significant digit; `None` for zero.
    fn leading_zeros(self) -> Option<u64> {
        (self.digits != 0).then(|| self.scale - decimal_digits(self.digits))
    }
}

impl Ord for SecondFraction {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self.leading_zeros(), other.leading_zeros()) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            // More zeros after the point is a smaller fraction.
            (Some(a), Some(b)) if a != b => b.cmp(&a),
            (Some(_), Some(_)) => {
                // Same position of the first significant digit: compare the
                // digit strings left-aligned.
                let (na, nb) = (decimal_digits(self.digits), decimal_digits(other.digits));
                let width = na.max(nb);
                let a = u128::from(self.digits) * 10_u128.pow((width - na) as u32);
                let b = u128::from(other.digits) * 10_u128.pow((width - nb) as u32);
                a.cmp(&b)
            }
        }
    }
}

impl PartialOrd for SecondFraction {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Number of decimal digits of `value`; `1` for zero.
fn decimal_digits(value: u64) -> u64 {
    u64::from(value.checked_ilog10().unwrap_or(0)) + 1
}

/// ANTEX parse or lookup error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AntexError {
    /// [`AntexDateTime`] construction received a component outside the GPS
    /// calendar and clock ranges.
    InvalidDateTime,
    /// A record field holds text that is not a valid value for that field.
    InvalidField {
        /// Id of the antenna block holding the record; `None` for a header
        /// record.
        antenna_id: Option<String>,
        /// Record label.
        record: &'static str,
        /// Field name within the record.
        field: &'static str,
        /// Trimmed field text.
        value: String,
    },
    /// A record the format allows once per antenna block, frequency section or
    /// header appears again with different content.
    RepeatedRecord {
        /// Id of the antenna block; `None` for a header record.
        antenna_id: Option<String>,
        /// Record label.
        record: &'static str,
    },
    /// A PCV row cannot be placed on a grid of distinct positions: a value
    /// past the first of its row with a zenith step that is not positive, a
    /// second value at a grid position already filled in the same section, or
    /// a row read before the block's `ZEN1 / ZEN2 / DZEN` record.
    DegenerateGrid {
        /// Id of the antenna block.
        antenna_id: String,
        /// Label of the frequency section.
        frequency: String,
        /// What makes the grid degenerate.
        reason: String,
    },
    /// A public PCV input was rejected by shared validation.
    InvalidInput {
        /// Static label of the rejected PCV argument; the current path uses
        /// `"zenith_deg"`.
        field: &'static str,
        /// Validator reason, such as `"not finite"` or `"out of range"`.
        reason: &'static str,
    },
    /// The trimmed requested frequency label was absent from an antenna's
    /// frequency sections.
    UnknownFrequency {
        /// Id of the antenna whose frequency sections were queried.
        antenna_id: String,
        /// Original caller-supplied frequency argument, before lookup trims it.
        frequency: String,
    },
    /// More than one frequency section of an antenna carries the requested
    /// label and their contents differ, so no single calibration answers.
    AmbiguousFrequency {
        /// Id of the antenna whose frequency sections were queried.
        antenna_id: String,
        /// Trimmed frequency label.
        frequency: String,
        /// Number of sections carrying the label.
        sections: usize,
    },
    /// A frequency block ended without a finite three-value `NORTH / EAST / UP`
    /// record having been parsed.
    MissingPco {
        /// Id of the antenna containing the incomplete frequency block.
        antenna_id: String,
        /// Trimmed label from that block's `START OF FREQUENCY` record.
        frequency: String,
    },
    /// The selected PCV interpolation input contained no samples.
    EmptyPcvGrid {
        /// Id of the antenna requested by the PCV lookup.
        antenna_id: String,
        /// Stored frequency label used by the PCV lookup.
        frequency: String,
    },
    /// A product or record cannot be faithfully serialized to ANTEX text.
    Unwritable {
        /// Field, record, or context that failed validation.
        field: &'static str,
        /// Actionable detail explaining why serialization was refused.
        reason: String,
    },
}

impl fmt::Display for AntexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDateTime => write!(f, "invalid ANTEX datetime"),
            Self::InvalidField {
                antenna_id,
                record,
                field,
                value,
            } => {
                write!(f, "invalid ANTEX {record} {field} {value:?}")?;
                if let Some(id) = antenna_id {
                    write!(f, " on {id:?}")?;
                }
                Ok(())
            }
            Self::RepeatedRecord { antenna_id, record } => {
                write!(f, "repeated ANTEX {record} record with different content")?;
                if let Some(id) = antenna_id {
                    write!(f, " on {id:?}")?;
                }
                Ok(())
            }
            Self::DegenerateGrid {
                antenna_id,
                frequency,
                reason,
            } => write!(
                f,
                "degenerate PCV grid for frequency {frequency:?} on {antenna_id:?}: {reason}"
            ),
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid ANTEX input {field}: {reason}")
            }
            Self::UnknownFrequency {
                antenna_id,
                frequency,
            } => write!(f, "unknown frequency {frequency:?} for {antenna_id:?}"),
            Self::AmbiguousFrequency {
                antenna_id,
                frequency,
                sections,
            } => write!(
                f,
                "frequency {frequency:?} on {antenna_id:?} has {sections} differing sections"
            ),
            Self::MissingPco {
                antenna_id,
                frequency,
            } => write!(
                f,
                "missing or malformed PCO for frequency {frequency:?} on {antenna_id:?}"
            ),
            Self::EmptyPcvGrid {
                antenna_id,
                frequency,
            } => write!(
                f,
                "empty PCV grid for frequency {frequency:?} on {antenna_id:?}"
            ),
            Self::Unwritable { field, reason } => {
                write!(f, "cannot serialize ANTEX {field}: {reason}")
            }
        }
    }
}

impl std::error::Error for AntexError {}

#[derive(Debug, Clone)]
struct ParseState {
    header: AntexHeader,
    /// Bodies of the once-only header records already read.
    header_records: Vec<(&'static str, String)>,
    antennas: BTreeMap<String, Antenna>,
    antenna_intervals: BTreeMap<String, Vec<Antenna>>,
    block_order: Vec<(String, usize)>,
    outer_comments: Vec<OuterComment>,
    /// Whether an antenna block has been opened; comments outside the blocks
    /// belong to the header only before the first one.
    block_seen: bool,
    current: Option<AntennaState>,
    section: Option<SectionState>,
    /// One-based number of the line currently being processed, attached to skips.
    line: usize,
    /// Non-fatal diagnostics: typed skips for malformed or inconsistent records
    /// the forgiving parser reported rather than aborting on.
    diagnostics: Diagnostics,
}

#[derive(Debug, Clone)]
struct AntennaState {
    antenna: Antenna,
    /// Whether `TYPE / SERIAL NO` has been read, which gives the block its
    /// identity.
    identified: bool,
    /// Bodies of the once-only records of this block already read.
    records: Vec<(&'static str, String)>,
    declared_frequencies: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SectionKind {
    Values,
    Rms,
}

#[derive(Debug, Clone)]
struct SectionState {
    kind: SectionKind,
    frequency: String,
    pco_m: Option<[f64; 3]>,
    pco_body: Option<String>,
    samples: Vec<PcvSample>,
    /// Filled grid positions: the row head's azimuth bits (`None` for `NOAZI`)
    /// and the zenith index.
    occupied: BTreeSet<(Option<u64>, usize)>,
}

impl Antex {
    /// Parse ANTEX text into receiver and satellite antenna blocks.
    pub fn parse(text: &str) -> Result<Self, AntexError> {
        let mut state = ParseState {
            header: AntexHeader::default(),
            header_records: Vec::new(),
            antennas: BTreeMap::new(),
            antenna_intervals: BTreeMap::new(),
            block_order: Vec::new(),
            outer_comments: Vec::new(),
            block_seen: false,
            current: None,
            section: None,
            line: 0,
            diagnostics: Diagnostics::new(),
        };

        for (index, line) in text.lines().enumerate() {
            state.line = index + 1;
            step(line, &mut state)?;
        }
        close_open_block(&mut state)?;

        let skipped_records = state.diagnostics.skips.len();
        Ok(Self {
            header: state.header,
            outer_comments: state.outer_comments,
            antennas: state.antennas,
            antenna_intervals: state.antenna_intervals,
            block_order: state.block_order,
            skipped_records,
        })
    }

    /// Number of records skipped during a forgiving parse (see the field docs).
    pub fn skipped_records(&self) -> usize {
        self.skipped_records
    }

    /// Return an antenna by the `TYPE / SERIAL` id.
    pub fn antenna(&self, id: &str) -> Option<&Antenna> {
        self.antennas.get(id.trim())
    }

    /// Return all validity blocks for a `TYPE / SERIAL` id, in file order.
    pub fn antenna_intervals(&self, id: &str) -> impl Iterator<Item = &Antenna> {
        self.antenna_intervals.get(id.trim()).into_iter().flatten()
    }

    /// Return every antenna block in file order.
    pub fn antenna_blocks(&self) -> impl Iterator<Item = &Antenna> {
        self.block_order.iter().filter_map(|(id, index)| {
            self.antenna_intervals
                .get(id)
                .and_then(|blocks| blocks.get(*index))
        })
    }

    /// Return the antenna validity block for a `TYPE / SERIAL` id at an epoch.
    pub fn antenna_at(&self, id: &str, epoch: AntexDateTime) -> Option<&Antenna> {
        self.antenna_intervals(id)
            .find(|antenna| antenna.valid_at(epoch))
    }

    /// Return the satellite antenna block for a PRN at an epoch.
    pub fn satellite_antenna(&self, prn: &str, epoch: AntexDateTime) -> Option<&Antenna> {
        let prn = prn.trim();
        self.antenna_intervals.values().flatten().find(|antenna| {
            antenna.kind == AntennaKind::Satellite
                && antenna.serial.trim() == prn
                && antenna.valid_at(epoch)
        })
    }
}

impl Antenna {
    /// Whether this antenna block is valid at `epoch`.
    pub fn valid_at(&self, epoch: AntexDateTime) -> bool {
        self.valid_from.is_none_or(|from| epoch >= from)
            && self.valid_until.is_none_or(|until| epoch <= until)
    }

    /// The frequency section with the trimmed `frequency` label.
    ///
    /// When several sections carry the label, they must be identical;
    /// otherwise the lookup is refused with [`AntexError::AmbiguousFrequency`].
    pub fn frequency(&self, frequency: &str) -> Result<&Frequency, AntexError> {
        let label = frequency.trim();
        let mut sections = self.frequencies.iter().filter(|f| f.frequency == label);
        let Some(first) = sections.next() else {
            return Err(AntexError::UnknownFrequency {
                antenna_id: self.id.clone(),
                frequency: frequency.to_string(),
            });
        };
        let mut count = 1;
        let mut identical = true;
        for other in sections {
            count += 1;
            identical &= frequency_sync_eq(first, other);
        }
        if !identical {
            return Err(AntexError::AmbiguousFrequency {
                antenna_id: self.id.clone(),
                frequency: label.to_string(),
                sections: count,
            });
        }
        Ok(first)
    }

    /// Frequency-dependent PCO (north/east/up), meters.
    pub fn pco(&self, frequency: &str) -> Result<[f64; 3], AntexError> {
        self.frequency(frequency).map(|f| f.pco_m)
    }

    /// Frequency-dependent PCV, meters, with linear zenith/azimuth interpolation.
    pub fn pcv(
        &self,
        frequency: &str,
        zenith_deg: f64,
        azimuth_deg: Option<f64>,
    ) -> Result<f64, AntexError> {
        validate::finite(zenith_deg, "zenith_deg").map_err(map_antex_field_error)?;
        if let Some(grid) = self.zenith_grid {
            validate_pcv_zenith(zenith_deg, grid.start_deg, grid.end_deg)?;
        }
        self.frequency(frequency)?
            .pcv(self.id.as_str(), zenith_deg, azimuth_deg)
    }
}

impl Frequency {
    fn pcv(
        &self,
        antenna_id: &str,
        zenith_deg: f64,
        azimuth_deg: Option<f64>,
    ) -> Result<f64, AntexError> {
        let noazi: Vec<(f64, f64)> = self
            .pcv_samples
            .iter()
            .filter(|sample| sample.grid == PcvGrid::NoAzimuth)
            .map(|sample| (sample.zenith_deg, sample.value_m))
            .collect();

        let has_azimuth = self
            .pcv_samples
            .iter()
            .any(|sample| sample.grid == PcvGrid::Azimuth);

        if azimuth_deg.is_none() || !has_azimuth {
            return interpolate(antenna_id, &self.frequency, &noazi, zenith_deg);
        }

        let mut azimuth_samples: BTreeMap<OrderedF64, Vec<(f64, f64)>> = BTreeMap::new();
        for sample in self
            .pcv_samples
            .iter()
            .filter(|sample| sample.grid == PcvGrid::Azimuth)
        {
            if let Some(azimuth) = sample.azimuth_deg {
                azimuth_samples
                    .entry(OrderedF64(azimuth))
                    .or_default()
                    .push((sample.zenith_deg, sample.value_m));
            }
        }

        if azimuth_samples.is_empty() {
            interpolate(antenna_id, &self.frequency, &noazi, zenith_deg)
        } else {
            let Some(azimuth_deg) = azimuth_deg else {
                return interpolate(antenna_id, &self.frequency, &noazi, zenith_deg);
            };
            interpolate_azimuth(
                antenna_id,
                &self.frequency,
                &azimuth_samples,
                azimuth_deg,
                zenith_deg,
            )
        }
    }
}

fn validate_pcv_zenith(
    zenith_deg: f64,
    zenith_start_deg: f64,
    zenith_end_deg: f64,
) -> Result<(), AntexError> {
    validate::finite(zenith_deg, "zenith_deg").map_err(map_antex_field_error)?;
    if zenith_deg < zenith_start_deg || zenith_deg > zenith_end_deg {
        return Err(invalid_input("zenith_deg", "out of range"));
    }
    Ok(())
}

fn map_antex_field_error(error: validate::FieldError) -> AntexError {
    invalid_input(error.field(), error.reason())
}

fn invalid_input(field: &'static str, reason: &'static str) -> AntexError {
    AntexError::InvalidInput { field, reason }
}

impl AntexDateTime {
    /// Construct a whole-second GPS-time instant.
    ///
    /// A component outside the calendar or GPS clock ranges, including a
    /// `60` second, returns [`AntexError::InvalidDateTime`].
    pub fn new(
        year: i32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
    ) -> Result<Self, AntexError> {
        Self::new_with_nanosecond(year, month, day, hour, minute, second, 0)
    }

    /// Construct a GPS-time instant with a fractional second in whole
    /// nanoseconds (`0..1_000_000_000`).
    pub fn new_with_nanosecond(
        year: i32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
        nanosecond: u32,
    ) -> Result<Self, AntexError> {
        let fraction =
            SecondFraction::from_nanoseconds(nanosecond).ok_or(AntexError::InvalidDateTime)?;
        Self::new_with_fraction(year, month, day, hour, minute, second, fraction)
    }

    /// Construct a GPS-time instant with an exact fractional second.
    pub fn new_with_fraction(
        year: i32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
        fraction: SecondFraction,
    ) -> Result<Self, AntexError> {
        check_gps_civil([
            i64::from(year),
            i64::from(month),
            i64::from(day),
            i64::from(hour),
            i64::from(minute),
            i64::from(second),
        ])
        .map_err(|_| AntexError::InvalidDateTime)?;
        Ok(Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
            fraction,
        })
    }
}

const DATETIME_FIELDS: [&str; 6] = ["year", "month", "day", "hour", "minute", "second"];

/// Check calendar and GPS clock ranges for
/// `[year, month, day, hour, minute, second]`, returning the index of the
/// first component out of range. GPS time has no leap-second label, so the
/// second is `0..=59` on every day.
fn check_gps_civil(parts: [i64; 6]) -> Result<(), usize> {
    let [year, month, day, hour, minute, second] = parts;
    let last_day = crate::astro::time::civil::days_in_month(year, month);
    let checks = [
        (0..=9999).contains(&year),
        (1..=12).contains(&month),
        (1..=last_day).contains(&day),
        (0..=23).contains(&hour),
        (0..=59).contains(&minute),
        (0..=59).contains(&second),
    ];
    match checks.iter().position(|ok| !ok) {
        Some(index) => Err(index),
        None => Ok(()),
    }
}

/// Read an `F13.7` seconds field exactly as a whole second and a fraction.
///
/// The text is an optional sign, digits with at most one decimal point, and an
/// optional exponent `E`, `e`, `D` or `d` with an optional sign. The exponent
/// moves the decimal point in the digit string, so no value is rounded. A
/// negative zero reads as zero. `None` for any other text, for a negative
/// value, and for a whole second too large to be a clock second.
///
/// Fortran `F13.7` input and RTKLIB's `sscanf` agree on every such text but
/// two, and this reader takes a side in each. A `D` exponent is read as
/// Fortran reads it; `sscanf` stops at the `D`. A mantissa without a decimal
/// point is read as RTKLIB reads it, so `59` is 59 s; Fortran would apply the
/// implied seven decimals and read 0.0000059 s.
fn parse_gps_seconds(text: &str) -> Option<(i64, SecondFraction)> {
    let (negative, unsigned) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let (mantissa, exponent) = match unsigned.find(['e', 'E', 'd', 'D']) {
        Some(at) => (&unsigned[..at], Some(&unsigned[at + 1..])),
        None => (unsigned, None),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if (whole.is_empty() && fraction.is_empty())
        || !whole.bytes().all(|b| b.is_ascii_digit())
        || !fraction.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let exponent: i64 = match exponent {
        None => 0,
        Some(text) => {
            let digits = text.strip_prefix(['+', '-']).unwrap_or(text);
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            text.strip_prefix('+').unwrap_or(text).parse().ok()?
        }
    };

    // The value is 0.D x 10^point, D the mantissa digits without leading
    // zeros.
    let all_digits = format!("{whole}{fraction}");
    let significant = all_digits.trim_start_matches('0');
    let leading = (all_digits.len() - significant.len()) as i64;
    let significant = significant.trim_end_matches('0');
    if significant.is_empty() {
        return Some((0, SecondFraction::ZERO));
    }
    if negative {
        return None;
    }
    let point = (whole.len() as i64)
        .checked_sub(leading)?
        .checked_add(exponent)?;
    let count = significant.len() as i64;
    // A clock second has at most two whole digits.
    if point > 2 {
        return None;
    }
    let (whole_second, fraction_digits) = if point <= 0 {
        (0, significant)
    } else {
        let split = point.min(count) as usize;
        let mut whole_second: i64 = significant[..split].parse().ok()?;
        for _ in count..point {
            whole_second *= 10;
        }
        (whole_second, &significant[split..])
    };
    let fraction = if fraction_digits.is_empty() {
        SecondFraction::ZERO
    } else {
        let scale = count.checked_sub(point)?;
        SecondFraction::new(fraction_digits.parse().ok()?, u64::try_from(scale).ok()?)?
    };
    Some((whole_second, fraction))
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// ── Serialization ────────────────────────────────────────────────────
//
// The inverse of [`Antex::parse`]: a parsed product is rendered back to ANTEX
// text. The canonical [`Antex`] container is the format-agnostic IR (SI units:
// PCO/PCV in meters, angles in degrees); this writer maps it onto the ANTEX
// 1.4 record grammar in the order the format lays records out. Parsing the
// output reproduces the same [`Antex`]: header records and comments, antenna
// blocks in file order, calibration records, validity bounds with their
// fractional seconds, frequency and RMS sections in order, PCO triples and PCV
// samples. Records the format requires are written from the retained values;
// a required record the source did not carry (a header without `ANTEX VERSION
// / SYST`, a block without `METH / BY / # / DATE`, a frequency without a
// `NOAZI` row) has no retained value and is not invented. `# OF FREQUENCIES`
// is written from the number of frequency sections. The output is not a
// byte-for-byte copy of the source: comments are written at the positions the
// format assigns them, labels carry no trailing blanks, and numbers use the
// format's field layouts.

/// ANTEX label column: records carry their record-type tag at columns 60..80.
const LABEL_COLUMN: usize = 60;

fn unwritable(field: &'static str, reason: impl Into<String>) -> AntexError {
    AntexError::Unwritable {
        field,
        reason: reason.into(),
    }
}

#[inline]
fn f6_1_tenths(val: f64) -> i64 {
    (val * 10.0).round() as i64
}

fn validate_f6_1(val: f64, field: &'static str) -> Result<String, AntexError> {
    if !val.is_finite() {
        return Err(unwritable(field, "value is not finite"));
    }
    if val < 0.0 {
        return Err(unwritable(field, format!("value {val} is negative")));
    }
    let formatted = format!("{:6.1}", val);
    if formatted.len() != 6 {
        return Err(unwritable(
            field,
            format!("value {val} exceeds F6.1 field width: {formatted:?}"),
        ));
    }
    let Ok(parsed) = formatted.trim().parse::<f64>() else {
        return Err(unwritable(
            field,
            format!("failed to parse formatted F6.1 value {formatted:?}"),
        ));
    };
    if parsed.to_bits() != val.to_bits() {
        return Err(unwritable(
            field,
            format!("value {val} loses precision when formatted as F6.1 ({formatted:?})"),
        ));
    }
    Ok(formatted)
}

fn validate_version_f8_1(version: f64) -> Result<String, AntexError> {
    let formatted = format!("{version:8.1}");
    if !version.is_finite()
        || formatted.len() != 8
        || formatted.trim().parse::<f64>().map(f64::to_bits) != Ok(version.to_bits())
    {
        return Err(unwritable(
            "version",
            format!("version {version} is not representable as F8.1"),
        ));
    }
    Ok(formatted)
}

fn validate_azimuth_f8_1(azimuth: f64) -> Result<String, AntexError> {
    if !azimuth.is_finite() {
        return Err(unwritable("azimuth_deg", "azimuth is not finite"));
    }
    let formatted = format!("{:8.1}", azimuth);
    if formatted.len() != 8 {
        return Err(unwritable(
            "azimuth_deg",
            format!("azimuth {azimuth} exceeds F8.1 field width: {formatted:?}"),
        ));
    }
    let Ok(parsed) = formatted.trim().parse::<f64>() else {
        return Err(unwritable(
            "azimuth_deg",
            format!("failed to parse formatted azimuth {formatted:?}"),
        ));
    };
    if parsed.to_bits() != azimuth.to_bits() {
        return Err(unwritable(
            "azimuth_deg",
            format!("azimuth {azimuth} loses precision when formatted as F8.1 ({formatted:?})"),
        ));
    }
    Ok(formatted)
}

fn validate_pco_f10_2(val_m: f64, component: &'static str) -> Result<String, AntexError> {
    if !val_m.is_finite() {
        return Err(unwritable(
            "pco_m",
            format!("PCO {component} component is not finite"),
        ));
    }
    let val_mm = val_m * MM_PER_M;
    let formatted = format!("{:10.2}", val_mm);
    if formatted.len() != 10 {
        return Err(unwritable(
            "pco_m",
            format!("PCO {component} {val_mm} mm exceeds F10.2 field width: {formatted:?}"),
        ));
    }
    let Ok(parsed_mm) = formatted.trim().parse::<f64>() else {
        return Err(unwritable(
            "pco_m",
            format!("failed to parse PCO {component} formatted value {formatted:?}"),
        ));
    };
    let readback_m = parsed_mm * M_PER_MM;
    if readback_m.to_bits() != val_m.to_bits() {
        return Err(unwritable(
            "pco_m",
            format!("PCO {component} {val_m} m loses precision when formatted as F10.2 mm ({formatted:?})"),
        ));
    }
    Ok(formatted)
}

fn validate_pcv_f8_2(val_m: f64) -> Result<String, AntexError> {
    if !val_m.is_finite() {
        return Err(unwritable("pcv_samples", "PCV sample value is not finite"));
    }
    let val_mm = val_m * MM_PER_M;
    let formatted = format!("{:8.2}", val_mm);
    if formatted.len() != 8 {
        return Err(unwritable(
            "pcv_samples",
            format!("PCV sample {val_mm} mm exceeds F8.2 field width: {formatted:?}"),
        ));
    }
    let Ok(parsed_mm) = formatted.trim().parse::<f64>() else {
        return Err(unwritable(
            "pcv_samples",
            format!("failed to parse PCV sample formatted value {formatted:?}"),
        ));
    };
    let readback_m = parsed_mm * M_PER_MM;
    if readback_m.to_bits() != val_m.to_bits() {
        return Err(unwritable(
            "pcv_samples",
            format!(
                "PCV sample {val_m} m loses precision when formatted as F8.2 mm ({formatted:?})"
            ),
        ));
    }
    Ok(formatted)
}

/// How the parser trims a text field, which decides what the writer may hold.
#[derive(Clone, Copy)]
enum TextTrim {
    /// Leading and trailing whitespace removed.
    Both,
    /// Trailing whitespace removed.
    End,
}

/// Check that `value` reads back unchanged from a `width`-byte field that the
/// parser trims as `trim` says.
fn validate_text(
    value: &str,
    width: usize,
    trim: TextTrim,
    field: &'static str,
) -> Result<(), AntexError> {
    if value.len() > width {
        return Err(unwritable(
            field,
            format!("{value:?} is longer than its {width}-column field"),
        ));
    }
    if value.contains(['\n', '\r']) {
        return Err(unwritable(
            field,
            format!("{value:?} contains a line break"),
        ));
    }
    let surrounded = value.ends_with(char::is_whitespace)
        || (matches!(trim, TextTrim::Both) && value.starts_with(char::is_whitespace));
    if surrounded {
        return Err(unwritable(
            field,
            format!("{value:?} has whitespace the reader trims"),
        ));
    }
    Ok(())
}

fn validate_sinex_code(code: &str) -> Result<(), AntexError> {
    if code.is_empty() {
        return Err(unwritable("sinex_code", "SINEX code is empty"));
    }
    validate_text(code, LABEL_COLUMN, TextTrim::Both, "sinex_code")
}

/// `START OF FREQUENCY` and its end records are `3X,A1,I2,54X`: a label is
/// written as the system flag in column 4 and the two-column frequency number
/// in columns 5-6, so it must be exactly that. A longer label would run into
/// the blank columns, where RTKLIB, reading the number with `sscanf`, takes
/// `G01X` for `G01`.
fn validate_frequency_label(freq: &str) -> Result<(), AntexError> {
    let bytes = freq.as_bytes();
    let is_code = bytes.len() == 3
        && bytes[0].is_ascii_graphic()
        && (bytes[1].is_ascii_digit() || matches!(bytes[1], b' ' | b'+' | b'-'))
        && bytes[2].is_ascii_digit();
    if !is_code {
        return Err(unwritable(
            "frequency",
            format!("frequency label {freq:?} is not a system flag and a two-column number"),
        ));
    }
    Ok(())
}

fn validate_antenna_header(antenna: &Antenna) -> Result<(), AntexError> {
    if antenna.id.is_empty() {
        return Err(unwritable("id", "antenna id is empty"));
    }
    validate_text(&antenna.id, LABEL_COLUMN, TextTrim::Both, "id")?;
    let id_line = labeled(&antenna.id, "TYPE / SERIAL NO");
    let mut decoded = blank_antenna();
    identify_antenna(&mut decoded, &id_line);
    if decoded.id != antenna.id
        || decoded.antenna_type != antenna.antenna_type
        || decoded.serial != antenna.serial
        || decoded.kind != antenna.kind
    {
        return Err(unwritable(
            "id",
            format!(
                "antenna header fields diverge from id: id={:?}, type={:?}, serial={:?}, kind={:?}",
                antenna.id, antenna.antenna_type, antenna.serial, antenna.kind
            ),
        ));
    }
    Ok(())
}

#[inline]
fn f64_bits_eq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

#[inline]
fn opt_f64_bits_eq(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x.to_bits() == y.to_bits(),
        (None, None) => true,
        _ => false,
    }
}

fn grid_bits_eq(a: Option<ZenithGrid>, b: Option<ZenithGrid>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => {
            f64_bits_eq(x.start_deg, y.start_deg)
                && f64_bits_eq(x.end_deg, y.end_deg)
                && f64_bits_eq(x.step_deg, y.step_deg)
        }
        (None, None) => true,
        _ => false,
    }
}

fn triple_bits_eq(a: [f64; 3], b: [f64; 3]) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| f64_bits_eq(*x, *y))
}

fn pcv_sample_sync_eq(a: &PcvSample, b: &PcvSample) -> bool {
    a.grid == b.grid
        && opt_f64_bits_eq(a.azimuth_deg, b.azimuth_deg)
        && f64_bits_eq(a.zenith_deg, b.zenith_deg)
        && f64_bits_eq(a.value_m, b.value_m)
}

fn samples_sync_eq(a: &[PcvSample], b: &[PcvSample]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(s1, s2)| pcv_sample_sync_eq(s1, s2))
}

fn frequency_sync_eq(a: &Frequency, b: &Frequency) -> bool {
    let rms_eq = match (&a.rms, &b.rms) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            let pco_eq = match (x.pco_m, y.pco_m) {
                (None, None) => true,
                (Some(p), Some(q)) => triple_bits_eq(p, q),
                _ => false,
            };
            pco_eq && samples_sync_eq(&x.pcv_samples, &y.pcv_samples)
        }
        _ => false,
    };
    a.frequency == b.frequency
        && triple_bits_eq(a.pco_m, b.pco_m)
        && samples_sync_eq(&a.pcv_samples, &b.pcv_samples)
        && rms_eq
}

fn antenna_sync_eq(a: &Antenna, b: &Antenna) -> bool {
    a.id == b.id
        && a.kind == b.kind
        && a.antenna_type == b.antenna_type
        && a.serial == b.serial
        && a.calibrations == b.calibrations
        && a.sinex_code == b.sinex_code
        && a.valid_from == b.valid_from
        && a.valid_until == b.valid_until
        && a.leading_comments == b.leading_comments
        && a.comments == b.comments
        && a.has_frequency_count == b.has_frequency_count
        && opt_f64_bits_eq(a.dazi_deg, b.dazi_deg)
        && grid_bits_eq(a.zenith_grid, b.zenith_grid)
        && a.frequencies.len() == b.frequencies.len()
        && a.frequencies
            .iter()
            .zip(b.frequencies.iter())
            .all(|(f1, f2)| frequency_sync_eq(f1, f2))
}

impl Antex {
    fn validate_antennas_sync(&self) -> Result<(), AntexError> {
        if self.antennas.len() != self.antenna_intervals.len() {
            return Err(unwritable(
                "antennas",
                format!(
                    "public antennas count ({}) diverges from antenna_intervals count ({})",
                    self.antennas.len(),
                    self.antenna_intervals.len()
                ),
            ));
        }
        for (id, public_ant) in &self.antennas {
            if &public_ant.id != id {
                return Err(unwritable(
                    "antennas",
                    format!(
                        "antenna map key {id:?} does not match antenna id {:?}",
                        public_ant.id
                    ),
                ));
            }
            let Some(intervals) = self.antenna_intervals.get(id) else {
                return Err(unwritable(
                    "antennas",
                    format!("public antenna {id:?} is missing from antenna_intervals"),
                ));
            };
            let Some(latest_interval) = intervals.last() else {
                return Err(unwritable(
                    "antenna_intervals",
                    format!("antenna_intervals for {id:?} has no intervals"),
                ));
            };
            if !antenna_sync_eq(public_ant, latest_interval) {
                return Err(unwritable(
                    "antennas",
                    format!("public antenna {id:?} has diverged from latest private interval"),
                ));
            }
        }
        for (id, intervals) in &self.antenna_intervals {
            if !self.antennas.contains_key(id) {
                return Err(unwritable(
                    "antennas",
                    format!(
                        "antenna_intervals has id {id:?} which is missing from public antennas"
                    ),
                ));
            }
            for ant in intervals {
                if &ant.id != id {
                    return Err(unwritable(
                        "antenna_intervals",
                        format!(
                            "interval antenna id {:?} does not match map key {id:?}",
                            ant.id
                        ),
                    ));
                }
            }
        }
        let block_count: usize = self.antenna_intervals.values().map(Vec::len).sum();
        if self.block_order.len() != block_count {
            return Err(unwritable(
                "antenna_intervals",
                format!(
                    "block order lists {} blocks for {block_count} retained intervals",
                    self.block_order.len()
                ),
            ));
        }
        Ok(())
    }

    /// Serialize this product back to ANTEX 1.4 text.
    ///
    /// # Contract and Round-Trip Guarantee
    ///
    /// Returns `Ok(String)` containing ANTEX 1.4 formatted text when the product's
    /// contents are faithfully representable in the fixed-width record grammar.
    /// While serialized output adheres to fixed-column grammar, it intentionally
    /// preserves accepted compatibility extensions (such as valid on-grid samples
    /// extending beyond declared `ZEN2`) rather than enforcing strict compliance
    /// with every specification limit. The serializer is deterministic: identical
    /// products produce byte-identical text, and for any cleanly parsed product
    /// ([`Antex::skipped_records`] is zero) whose contents meet the serialization
    /// contract, `parse(encode(product)) == product`.
    ///
    /// Every record is written from a retained value, in the order ANTEX 1.4
    /// lays records out: `ANTEX VERSION / SYST`, `PCV TYPE / REFANT`, the
    /// header comments and `END OF HEADER`; then each antenna block in file
    /// order, with the comments between blocks where they stood. A block
    /// carries its leading comments, `TYPE / SERIAL NO`, its
    /// `METH / BY / # / DATE` records, `DAZI`, `ZEN1 / ZEN2 / DZEN`,
    /// `# OF FREQUENCIES` (stating the number of frequency sections),
    /// `VALID FROM` / `VALID UNTIL` with their exact fractional seconds,
    /// `SINEX CODE`, its comments, and its frequency sections in order, each
    /// followed by its RMS section when retained. A record the source did not
    /// carry is not written. The exceptions are the start and end records of
    /// blocks and sections, which the writer always writes; the parser reports
    /// a block or section its own end record did not close as a skipped record.
    ///
    /// The parser is intentionally forgiving, but serialization is strict: retained
    /// product data that cannot be accurately represented without loss, coordinate shift,
    /// or corruption is explicitly refused with [`AntexError::Unwritable`].
    ///
    /// # Refusal Conditions
    ///
    /// Serialization is refused if:
    /// - Public [`Antex::antennas`] has diverged from the latest validity interval in
    ///   private `antenna_intervals`, or an antenna's `id` does not match its map key.
    /// - Header fields cannot be faithfully represented in their fixed columns without
    ///   width overflow or precision loss (e.g. `DAZI` or `ZEN1/ZEN2/DZEN` in `F6.1`,
    ///   the version in `F8.1`, text longer than its field or carrying whitespace the
    ///   reader trims).
    /// - A validity bound is not a GPS calendar instant, its seconds do not fit the
    ///   13-column field, or `VALID UNTIL` precedes `VALID FROM`.
    /// - A frequency label is not a system flag and a two-column number (`A1,I2`).
    /// - Comments outside the blocks are out of file order, name a position past the
    ///   last block, or stand before the first block with no `END OF HEADER` to
    ///   separate them from the header comments.
    /// - Zenith bounds are non-finite, negative, inverted (`ZEN1 > ZEN2`), or `DZEN <= 0.0`
    ///   when PCV grid rows are present, or PCV values have no zenith grid.
    /// - PCO values cannot be represented in `3F10.2` mm without overflow or precision loss.
    /// - PCV samples contain non-finite values, coordinates below `ZEN1`, off-grid zenith
    ///   coordinates that do not match the reader reconstruction formula `ZEN1 + k * DZEN`,
    ///   duplicate positions in the same row, or values that cannot be represented in `F8.2` mm.
    /// - PCV grid sample metadata is inconsistent (e.g. [`PcvGrid::NoAzimuth`] with an
    ///   azimuth coordinate, or [`PcvGrid::Azimuth`] without one).
    /// - Sample rows cannot preserve vector order (e.g. interleaved or split azimuth rows,
    ///   or `NOAZI` rows appearing after azimuth-dependent rows).
    /// - Declared grid width is maintained through `ZEN2`. Samples extending beyond `ZEN2`
    ///   are preserved via compatible row extension only when their exact coordinates
    ///   survive on-grid and remain bounded by representable fixed-width decimal grid indices
    ///   (sample grid index `k <= (99999 - start_tenths) / step_tenths`).
    pub fn encode(&self) -> Result<String, AntexError> {
        self.validate_antennas_sync()?;
        self.validate_outer_comments()?;
        let mut out = String::new();
        encode_header(&self.header, &mut out)?;
        let mut outer = self.outer_comments.iter().peekable();
        for (position, (id, index)) in self.block_order.iter().enumerate() {
            while let Some(comment) = outer.next_if(|c| c.blocks_before == position) {
                out.push_str(&labeled(&comment.text, "COMMENT"));
            }
            let Some(antenna) = self
                .antenna_intervals
                .get(id)
                .and_then(|blocks| blocks.get(*index))
            else {
                return Err(unwritable(
                    "antenna_intervals",
                    format!("block order names interval {index} of {id:?}, which is not retained"),
                ));
            };
            encode_antenna(antenna, &mut out)?;
        }
        for comment in outer {
            out.push_str(&labeled(&comment.text, "COMMENT"));
        }
        Ok(out)
    }

    fn validate_outer_comments(&self) -> Result<(), AntexError> {
        let blocks = self.block_order.len();
        let mut previous = 0;
        for comment in &self.outer_comments {
            validate_text(&comment.text, LABEL_COLUMN, TextTrim::End, "outer_comments")?;
            if comment.blocks_before < previous || comment.blocks_before > blocks {
                return Err(unwritable(
                    "outer_comments",
                    format!(
                        "comment after {} blocks is out of file order or past the last of {blocks} blocks",
                        comment.blocks_before
                    ),
                ));
            }
            if comment.blocks_before == 0 && !self.header.end_of_header {
                return Err(unwritable(
                    "outer_comments",
                    "a comment before the first block would be read as a header comment without END OF HEADER",
                ));
            }
            previous = comment.blocks_before;
        }
        Ok(())
    }
}

fn encode_header(header: &AntexHeader, out: &mut String) -> Result<(), AntexError> {
    if let Some(version) = header.version {
        let number = validate_version_f8_1(version.version)?;
        let system = match version.system {
            None => ' ',
            Some(system) if system.is_ascii_graphic() => system,
            Some(system) => {
                return Err(unwritable(
                    "system",
                    format!("system flag {system:?} is not a printable ASCII character"),
                ));
            }
        };
        out.push_str(&labeled(
            &format!("{number}{:12}{system}", ""),
            "ANTEX VERSION / SYST",
        ));
    }
    if let Some(record) = &header.pcv_type {
        validate_text(
            &record.reference_antenna_type,
            20,
            TextTrim::Both,
            "reference_antenna_type",
        )?;
        validate_text(
            &record.reference_antenna_serial,
            20,
            TextTrim::Both,
            "reference_antenna_serial",
        )?;
        let flag = match record.pcv_type {
            PcvType::Absolute => 'A',
            PcvType::Relative => 'R',
        };
        out.push_str(&labeled(
            &format!(
                "{flag}{:19}{}{}",
                "",
                pad(&record.reference_antenna_type, 20),
                record.reference_antenna_serial
            ),
            "PCV TYPE / REFANT",
        ));
    }
    for comment in &header.comments {
        validate_text(comment, LABEL_COLUMN, TextTrim::End, "comments")?;
        out.push_str(&labeled(comment, "COMMENT"));
    }
    if header.end_of_header {
        out.push_str(&labeled("", "END OF HEADER"));
    }
    Ok(())
}

fn encode_calibration(calibration: &Calibration) -> Result<String, AntexError> {
    validate_text(&calibration.method, 20, TextTrim::End, "calibrations")?;
    validate_text(&calibration.agency, 20, TextTrim::End, "calibrations")?;
    validate_text(&calibration.date, 10, TextTrim::End, "calibrations")?;
    let count = match calibration.antennas_calibrated {
        None => String::new(),
        Some(count) if count <= 999_999 => count.to_string(),
        Some(count) => {
            return Err(unwritable(
                "calibrations",
                format!("{count} antennas calibrated exceeds the I6 field"),
            ));
        }
    };
    Ok(format!(
        "{}{}{count:>6}{:4}{}",
        pad(&calibration.method, 20),
        pad(&calibration.agency, 20),
        "",
        calibration.date
    ))
}

fn encode_antenna(antenna: &Antenna, out: &mut String) -> Result<(), AntexError> {
    validate_antenna_header(antenna)?;
    out.push_str(&labeled("", "START OF ANTENNA"));
    for comment in &antenna.leading_comments {
        validate_text(comment, LABEL_COLUMN, TextTrim::End, "leading_comments")?;
        out.push_str(&labeled(comment, "COMMENT"));
    }
    out.push_str(&labeled(&antenna.id, "TYPE / SERIAL NO"));
    for calibration in &antenna.calibrations {
        out.push_str(&labeled(
            &encode_calibration(calibration)?,
            "METH / BY / # / DATE",
        ));
    }
    if let Some(dazi) = antenna.dazi_deg {
        let dazi_str = validate_f6_1(dazi, "dazi_deg")?;
        out.push_str(&labeled(&format!("  {dazi_str}"), "DAZI"));
    }
    if let Some(grid) = antenna.zenith_grid {
        let zen_start_str = validate_f6_1(grid.start_deg, "zenith_start_deg")?;
        let zen_end_str = validate_f6_1(grid.end_deg, "zenith_end_deg")?;
        let zen_step_str = validate_f6_1(grid.step_deg, "zenith_step_deg")?;
        if grid.start_deg > grid.end_deg {
            return Err(unwritable(
                "zenith_end_deg",
                format!(
                    "inverted zenith bounds: start {} > end {}",
                    grid.start_deg, grid.end_deg
                ),
            ));
        }
        out.push_str(&labeled(
            &format!("  {zen_start_str}{zen_end_str}{zen_step_str}"),
            "ZEN1 / ZEN2 / DZEN",
        ));
    }
    if antenna.has_frequency_count {
        let frequency_count = antenna.frequencies.len();
        if frequency_count > 999_999 {
            return Err(unwritable(
                "frequencies",
                format!("{frequency_count} frequency sections exceed the I6 count field"),
            ));
        }
        out.push_str(&labeled(
            &format!("{frequency_count:6}"),
            "# OF FREQUENCIES",
        ));
    }

    if let (Some(from), Some(until)) = (antenna.valid_from, antenna.valid_until) {
        if from > until {
            return Err(unwritable(
                "valid_until",
                "valid_until is earlier than valid_from",
            ));
        }
    }
    if let Some(from) = antenna.valid_from {
        out.push_str(&labeled(&fmt_datetime(from, "valid_from")?, "VALID FROM"));
    }
    if let Some(until) = antenna.valid_until {
        out.push_str(&labeled(
            &fmt_datetime(until, "valid_until")?,
            "VALID UNTIL",
        ));
    }
    if let Some(code) = &antenna.sinex_code {
        validate_sinex_code(code)?;
        out.push_str(&labeled(code, "SINEX CODE"));
    }
    for comment in &antenna.comments {
        validate_text(comment, LABEL_COLUMN, TextTrim::End, "comments")?;
        out.push_str(&labeled(comment, "COMMENT"));
    }

    let has_pcv_samples = antenna.frequencies.iter().any(|f| {
        !f.pcv_samples.is_empty()
            || f.rms
                .as_ref()
                .is_some_and(|rms| !rms.pcv_samples.is_empty())
    });
    if has_pcv_samples {
        let Some(grid) = antenna.zenith_grid else {
            return Err(unwritable(
                "zenith_grid",
                "PCV samples are present without a ZEN1 / ZEN2 / DZEN record",
            ));
        };
        if grid.step_deg <= 0.0 {
            return Err(unwritable(
                "zenith_step_deg",
                format!(
                    "positive zenith step required when PCV samples are present, found {}",
                    grid.step_deg
                ),
            ));
        }
        let start_tenths = f6_1_tenths(grid.start_deg);
        let end_tenths = f6_1_tenths(grid.end_deg);
        let step_tenths = f6_1_tenths(grid.step_deg);
        if (end_tenths - start_tenths) % step_tenths != 0 {
            return Err(unwritable(
                "zenith_end_deg",
                format!(
                    "zenith_end_deg {} is not an integer multiple of step {} from start {}",
                    grid.end_deg, grid.step_deg, grid.start_deg
                ),
            ));
        }
    }

    for frequency in &antenna.frequencies {
        encode_frequency(antenna.zenith_grid, frequency, out)?;
    }
    out.push_str(&labeled("", "END OF ANTENNA"));
    Ok(())
}

fn encode_frequency(
    grid: Option<ZenithGrid>,
    frequency: &Frequency,
    out: &mut String,
) -> Result<(), AntexError> {
    validate_frequency_label(&frequency.frequency)?;
    let label_body = format!("   {}", frequency.frequency);
    out.push_str(&labeled(&label_body, "START OF FREQUENCY"));
    out.push_str(&pco_line(frequency.pco_m)?);
    encode_rows(grid, &frequency.pcv_samples, out)?;
    out.push_str(&labeled(&label_body, "END OF FREQUENCY"));

    if let Some(rms) = &frequency.rms {
        out.push_str(&labeled(&label_body, "START OF FREQ RMS"));
        if let Some(pco_m) = rms.pco_m {
            out.push_str(&pco_line(pco_m)?);
        }
        encode_rows(grid, &rms.pcv_samples, out)?;
        out.push_str(&labeled(&label_body, "END OF FREQ RMS"));
    }
    Ok(())
}

fn pco_line(pco_m: [f64; 3]) -> Result<String, AntexError> {
    let n_str = validate_pco_f10_2(pco_m[0], "north")?;
    let e_str = validate_pco_f10_2(pco_m[1], "east")?;
    let u_str = validate_pco_f10_2(pco_m[2], "up")?;
    Ok(labeled(
        &format!("{n_str}{e_str}{u_str}"),
        "NORTH / EAST / UP",
    ))
}

fn encode_rows(
    grid: Option<ZenithGrid>,
    samples: &[PcvSample],
    out: &mut String,
) -> Result<(), AntexError> {
    if samples.is_empty() {
        return Ok(());
    }
    let Some(grid) = grid else {
        return Err(unwritable(
            "zenith_grid",
            "PCV samples are present without a ZEN1 / ZEN2 / DZEN record",
        ));
    };
    let (noazi_samples, azimuth_rows) = validate_and_group_pcv_samples(&grid, samples)?;
    if !noazi_samples.is_empty() {
        out.push_str(&pcv_row(&grid, "   NOAZI", &noazi_samples)?);
    }
    for (azimuth_head, row_samples) in &azimuth_rows {
        out.push_str(&pcv_row(&grid, azimuth_head, row_samples)?);
    }
    Ok(())
}

type GroupedPcvRows<'a> = (Vec<&'a PcvSample>, Vec<(String, Vec<&'a PcvSample>)>);

fn validate_and_group_pcv_samples<'a>(
    grid: &ZenithGrid,
    samples: &'a [PcvSample],
) -> Result<GroupedPcvRows<'a>, AntexError> {
    let mut noazi_samples: Vec<&'a PcvSample> = Vec::new();
    let mut azimuth_rows: Vec<(String, Vec<&'a PcvSample>)> = Vec::new();
    let mut seen_azimuths: Vec<u64> = Vec::new();

    enum CurrentRow {
        None,
        NoAzimuth,
        Azimuth(u64),
    }
    let mut current_row = CurrentRow::None;
    let mut last_k: Option<usize> = None;

    if !samples.is_empty() && grid.step_deg <= 0.0 {
        return Err(unwritable(
            "zenith_step_deg",
            format!(
                "positive zenith step required when PCV samples are present, found {}",
                grid.step_deg
            ),
        ));
    }
    let start_tenths = f6_1_tenths(grid.start_deg);
    let step_tenths = f6_1_tenths(grid.step_deg);
    if !samples.is_empty() && step_tenths <= 0 {
        return Err(unwritable(
            "zenith_step_deg",
            format!(
                "positive zenith step required when PCV samples are present, found {}",
                grid.step_deg
            ),
        ));
    }
    let max_grid_k = if step_tenths > 0 && start_tenths <= 99999 {
        ((99999 - start_tenths) / step_tenths) as usize
    } else {
        0
    };

    for sample in samples {
        if !sample.zenith_deg.is_finite() {
            return Err(unwritable("pcv_samples", "sample zenith_deg is not finite"));
        }
        if !sample.value_m.is_finite() {
            return Err(unwritable("pcv_samples", "sample value_m is not finite"));
        }
        if sample.zenith_deg < 0.0
            || (sample.zenith_deg == 0.0 && sample.zenith_deg.is_sign_negative())
        {
            return Err(unwritable(
                "pcv_samples",
                format!("sample zenith_deg {} is negative", sample.zenith_deg),
            ));
        }
        if sample.zenith_deg < grid.start_deg {
            return Err(unwritable(
                "pcv_samples",
                format!(
                    "sample zenith_deg {} is below ZEN1 {}",
                    sample.zenith_deg, grid.start_deg
                ),
            ));
        }
        let _ = validate_pcv_f8_2(sample.value_m)?;

        let raw_k = (sample.zenith_deg - grid.start_deg) / grid.step_deg;
        let rounded_k = raw_k.round();
        if rounded_k > max_grid_k as f64 {
            return Err(unwritable(
                "pcv_samples",
                format!(
                    "sample zenith_deg {} exceeds maximum representable decimal grid index {max_grid_k}",
                    sample.zenith_deg
                ),
            ));
        }
        let k = rounded_k as usize;
        let reconstructed_zenith = grid.start_deg + grid.step_deg * (k as f64);
        if reconstructed_zenith.to_bits() != sample.zenith_deg.to_bits() {
            return Err(unwritable(
                "pcv_samples",
                format!(
                    "sample zenith {} is off-grid; does not match reader reconstruction formula ZEN1 + k * DZEN ({})",
                    sample.zenith_deg, reconstructed_zenith
                ),
            ));
        }

        match sample.grid {
            PcvGrid::NoAzimuth => {
                if sample.azimuth_deg.is_some() {
                    return Err(unwritable(
                        "pcv_samples",
                        "NoAzimuth sample must not have azimuth_deg set",
                    ));
                }
                match current_row {
                    CurrentRow::None => {
                        current_row = CurrentRow::NoAzimuth;
                        last_k = Some(k);
                        noazi_samples.push(sample);
                    }
                    CurrentRow::NoAzimuth => {
                        let prev_k = last_k.unwrap_or(0);
                        if k == prev_k {
                            return Err(unwritable(
                                "pcv_samples",
                                format!(
                                    "duplicate NOAZI sample at grid index {k} (zenith {})",
                                    sample.zenith_deg
                                ),
                            ));
                        }
                        if k < prev_k {
                            return Err(unwritable(
                                "pcv_samples",
                                format!(
                                    "NOAZI samples out of order: index {k} (zenith {}) after {prev_k}",
                                    sample.zenith_deg
                                ),
                            ));
                        }
                        last_k = Some(k);
                        noazi_samples.push(sample);
                    }
                    CurrentRow::Azimuth(_) => {
                        return Err(unwritable(
                            "pcv_samples",
                            "NOAZI sample appears after azimuth-dependent samples; vector order cannot be preserved",
                        ));
                    }
                }
            }
            PcvGrid::Azimuth => {
                let Some(azimuth) = sample.azimuth_deg else {
                    return Err(unwritable(
                        "pcv_samples",
                        "Azimuth sample missing azimuth_deg metadata",
                    ));
                };
                let az_head = validate_azimuth_f8_1(azimuth)?;
                let az_bits = azimuth.to_bits();

                match current_row {
                    CurrentRow::None | CurrentRow::NoAzimuth => {
                        current_row = CurrentRow::Azimuth(az_bits);
                        seen_azimuths.push(az_bits);
                        last_k = Some(k);
                        azimuth_rows.push((az_head, vec![sample]));
                    }
                    CurrentRow::Azimuth(curr_az_bits) => {
                        if az_bits == curr_az_bits {
                            let prev_k = last_k.unwrap_or(0);
                            if k == prev_k {
                                return Err(unwritable(
                                    "pcv_samples",
                                    format!(
                                        "duplicate azimuth {azimuth} sample at grid index {k} (zenith {})",
                                        sample.zenith_deg
                                    ),
                                ));
                            }
                            if k < prev_k {
                                return Err(unwritable(
                                    "pcv_samples",
                                    format!(
                                        "azimuth {azimuth} samples out of order: index {k} (zenith {}) after {prev_k}",
                                        sample.zenith_deg
                                    ),
                                ));
                            }
                            last_k = Some(k);
                            if let Some(last_row) = azimuth_rows.last_mut() {
                                last_row.1.push(sample);
                            }
                        } else {
                            if seen_azimuths.contains(&az_bits) {
                                return Err(unwritable(
                                    "pcv_samples",
                                    format!(
                                        "non-contiguous/split rows for azimuth {azimuth}; vector order cannot be preserved"
                                    ),
                                ));
                            }
                            current_row = CurrentRow::Azimuth(az_bits);
                            seen_azimuths.push(az_bits);
                            last_k = Some(k);
                            azimuth_rows.push((az_head, vec![sample]));
                        }
                    }
                }
            }
        }
    }

    Ok((noazi_samples, azimuth_rows))
}

/// Render one PCV grid row: a leading label token (`NOAZI` or an azimuth) then
/// the millimeter values in fixed 8-column fields up through declared `ZEN2` (and
/// beyond `ZEN2` when valid samples extend outside the declared grid). Empty positions
/// are emitted as blank fields ("        ").
fn pcv_row(grid: &ZenithGrid, head: &str, samples: &[&PcvSample]) -> Result<String, AntexError> {
    let mut line = format!("{head:<8}");
    if grid.step_deg <= 0.0 {
        return Err(unwritable(
            "zenith_step_deg",
            format!(
                "positive zenith step required for PCV row, found {}",
                grid.step_deg
            ),
        ));
    }
    let start_tenths = f6_1_tenths(grid.start_deg);
    let end_tenths = f6_1_tenths(grid.end_deg);
    let step_tenths = f6_1_tenths(grid.step_deg);
    let grid_k = if step_tenths > 0 && end_tenths >= start_tenths {
        ((end_tenths - start_tenths) / step_tenths) as usize
    } else {
        0
    };
    let sample_max_k = samples
        .iter()
        .map(|s| ((s.zenith_deg - grid.start_deg) / grid.step_deg).round() as usize)
        .max()
        .unwrap_or(0);
    let max_k = grid_k.max(sample_max_k);

    let mut sample_idx = 0;
    for k in 0..=max_k {
        if sample_idx < samples.len() {
            let s = samples[sample_idx];
            let s_k = ((s.zenith_deg - grid.start_deg) / grid.step_deg).round() as usize;
            if s_k == k {
                let pcv_field = validate_pcv_f8_2(s.value_m)?;
                line.push_str(&pcv_field);
                sample_idx += 1;
                continue;
            }
        }
        line.push_str("        ");
    }
    line.push('\n');
    Ok(line)
}

/// Pad `value` with blanks to `width` bytes. Columns are byte offsets, so the
/// padding counts bytes rather than characters.
fn pad(value: &str, width: usize) -> String {
    let mut padded = String::with_capacity(width.max(value.len()));
    padded.push_str(value);
    while padded.len() < width {
        padded.push(' ');
    }
    padded
}

/// A labeled fixed-column ANTEX record: the body left-justified into the tag
/// column, then the record-type label.
fn labeled(body: &str, label: &str) -> String {
    let mut line = pad(body, LABEL_COLUMN);
    line.push_str(label);
    line.push('\n');
    line
}

/// `VALID FROM` / `VALID UNTIL` value field: `5I6,F13.7`.
fn fmt_datetime(dt: AntexDateTime, field: &'static str) -> Result<String, AntexError> {
    check_gps_civil([
        i64::from(dt.year),
        i64::from(dt.month),
        i64::from(dt.day),
        i64::from(dt.hour),
        i64::from(dt.minute),
        i64::from(dt.second),
    ])
    .map_err(|index| {
        unwritable(
            field,
            format!(
                "{} of {dt:?} is outside the GPS calendar",
                DATETIME_FIELDS[index]
            ),
        )
    })?;
    let seconds = seconds_text(dt.second, dt.fraction).ok_or_else(|| {
        unwritable(
            field,
            format!("the seconds of {dt:?} do not fit the 13-column F13.7 field"),
        )
    })?;
    Ok(format!(
        "{:6}{:6}{:6}{:6}{:6}{seconds:>13}",
        dt.year, dt.month, dt.day, dt.hour, dt.minute
    ))
}

/// Width of the `F13.7` seconds field.
const SECONDS_WIDTH: usize = 13;

/// The seconds as text the `F13.7` field holds exactly: seven decimals, or
/// every decimal when there are more; with the leading zero dropped when that
/// is what makes a fraction fit; or, for a fraction too small for the field's
/// fixed decimals, with an exponent. Every form carries a decimal point, so a
/// Fortran reader, which applies the implied seven decimals to a mantissa
/// without one, and RTKLIB read the same value. `None` when no such text fits
/// the field.
fn seconds_text(second: u8, fraction: SecondFraction) -> Option<String> {
    let (digits, scale) = (fraction.digits(), fraction.scale());
    let fits = |text: String| (text.len() <= SECONDS_WIDTH).then_some(text);
    if scale <= 12 {
        let mut decimals = if digits == 0 {
            String::new()
        } else {
            format!("{digits:0>width$}", width = scale as usize)
        };
        while decimals.len() < 7 {
            decimals.push('0');
        }
        if let Some(text) = fits(format!("{second}.{decimals}")) {
            return Some(text);
        }
        if second == 0 {
            if let Some(text) = fits(format!(".{decimals}")) {
                return Some(text);
            }
        }
    }
    if second != 0 || digits == 0 {
        return None;
    }
    let significant = digits.to_string();
    // The fraction is below one, so scale >= count >= 1 and neither exponent
    // below can overflow.
    let count = significant.len() as u64;
    // digits / 10^scale = d.ddd x 10^-(scale - (count - 1))
    let scientific = format!(
        "{}.{}E-{}",
        &significant[..1],
        &significant[1..],
        scale - (count - 1)
    );
    // digits / 10^scale = 0.ddd x 10^-(scale - count)
    let leading_point = format!(".{significant}E-{}", scale - count);
    let shorter = if leading_point.len() < scientific.len() {
        leading_point
    } else {
        scientific
    };
    fits(shorter)
}

fn step(line: &str, state: &mut ParseState) -> Result<(), AntexError> {
    match tag(line) {
        "START OF ANTENNA" => open_block(state)?,
        "END OF ANTENNA" => end_block(state)?,
        "TYPE / SERIAL NO" => parse_type_serial(line, state)?,
        "METH / BY / # / DATE" | "METH / BY / DATE" => parse_calibration(line, state),
        "DAZI" => parse_dazi(line, state)?,
        "ZEN1 / ZEN2 / DZEN" => parse_zenith_grid(line, state)?,
        "# OF FREQUENCIES" => parse_frequency_count(line, state)?,
        "SINEX CODE" => parse_sinex_code(line, state)?,
        "VALID FROM" => parse_valid(line, state, ValidField::From)?,
        "VALID UNTIL" => parse_valid(line, state, ValidField::Until)?,
        "COMMENT" => parse_comment(line, state),
        "ANTEX VERSION / SYST" => parse_version(line, state)?,
        "PCV TYPE / REFANT" => parse_pcv_type(line, state)?,
        "END OF HEADER" => {
            if state.block_seen {
                push_skip(&mut state.diagnostics, state.line, MISPLACED_HEADER);
            }
            state.header.end_of_header = true;
        }
        "START OF FREQUENCY" => begin_section(line, state, SectionKind::Values)?,
        "START OF FREQ RMS" => begin_section(line, state, SectionKind::Rms)?,
        "END OF FREQUENCY" => end_section(line, state, SectionKind::Values)?,
        "END OF FREQ RMS" => end_section(line, state, SectionKind::Rms)?,
        "NORTH / EAST / UP" => parse_pco(line, state)?,
        _ => parse_row(line, state)?,
    }
    Ok(())
}

fn push_skip(diagnostics: &mut Diagnostics, line: usize, reason: SkipReason) {
    diagnostics.push_skip(Skip {
        at: RecordRef::at_line(line),
        reason,
    });
}

/// Record a once-only record's body. Returns `Ok(true)` for the first
/// occurrence, `Ok(false)` for a repeat with identical content, which carries
/// nothing new, and refuses a repeat with different content, since the file
/// then states two values and neither can be preferred.
fn claim_once(
    records: &mut Vec<(&'static str, String)>,
    record: &'static str,
    line: &str,
    antenna_id: Option<&str>,
) -> Result<bool, AntexError> {
    let body = raw_field(line, 0, LABEL_COLUMN).trim_end();
    match records.iter().find(|(seen, _)| *seen == record) {
        Some((_, seen_body)) if seen_body == body => Ok(false),
        Some(_) => Err(AntexError::RepeatedRecord {
            antenna_id: antenna_id.map(str::to_string),
            record,
        }),
        None => {
            records.push((record, body.to_string()));
            Ok(true)
        }
    }
}

fn invalid_field(
    antenna_id: Option<&str>,
    record: &'static str,
    field: &'static str,
    value: &str,
) -> AntexError {
    AntexError::InvalidField {
        antenna_id: antenna_id.map(str::to_string),
        record,
        field,
        value: value.trim().to_string(),
    }
}

const OUTSIDE_ANTENNA: SkipReason =
    SkipReason::InconsistentRecord("antex antenna record outside an antenna block");
const MISPLACED_HEADER: SkipReason =
    SkipReason::InconsistentRecord("antex header record after END OF HEADER or an antenna block");
const BLOCK_NOT_CLOSED: SkipReason =
    SkipReason::InconsistentRecord("antex antenna block not closed by END OF ANTENNA");
const SECTION_NOT_CLOSED: SkipReason =
    SkipReason::InconsistentRecord("antex frequency section not closed by its end record");

fn parse_version(line: &str, state: &mut ParseState) -> Result<(), AntexError> {
    const RECORD: &str = "ANTEX VERSION / SYST";
    if !claim_once(&mut state.header_records, RECORD, line, None)? {
        return Ok(());
    }
    if state.block_seen || state.header.end_of_header {
        push_skip(&mut state.diagnostics, state.line, MISPLACED_HEADER);
    }
    let version = fortran_f64(line, 0, 8, "antex version")
        .ok_or_else(|| invalid_field(None, RECORD, "version", raw_field(line, 0, 8)))?;
    let system = raw_field(line, 20, 21).chars().next().filter(|c| *c != ' ');
    state.header.version = Some(AntexVersion { version, system });
    Ok(())
}

fn parse_pcv_type(line: &str, state: &mut ParseState) -> Result<(), AntexError> {
    const RECORD: &str = "PCV TYPE / REFANT";
    if !claim_once(&mut state.header_records, RECORD, line, None)? {
        return Ok(());
    }
    if state.block_seen || state.header.end_of_header {
        push_skip(&mut state.diagnostics, state.line, MISPLACED_HEADER);
    }
    let flag = raw_field(line, 0, 1);
    let pcv_type = match flag {
        "A" => PcvType::Absolute,
        "R" => PcvType::Relative,
        _ => return Err(invalid_field(None, RECORD, "pcv type", flag)),
    };
    state.header.pcv_type = Some(PcvTypeRecord {
        pcv_type,
        reference_antenna_type: raw_field(line, 20, 40).trim().to_string(),
        reference_antenna_serial: raw_field(line, 40, 60).trim().to_string(),
    });
    Ok(())
}

/// Attach a comment to the entity whose records surround it: the header
/// before `END OF HEADER` (or before the first block when there is none), the
/// open block, before or after its `TYPE / SERIAL NO`, or otherwise the place
/// between blocks, counted by the blocks before it.
fn parse_comment(line: &str, state: &mut ParseState) {
    let text = raw_field(line, 0, LABEL_COLUMN).trim_end().to_string();
    match state.current.as_mut() {
        Some(current) if !current.identified => current.antenna.leading_comments.push(text),
        Some(current) => current.antenna.comments.push(text),
        None if !state.block_seen && !state.header.end_of_header => {
            state.header.comments.push(text);
        }
        None => state.outer_comments.push(OuterComment {
            blocks_before: state.block_order.len(),
            text,
        }),
    }
}

fn blank_block() -> AntennaState {
    AntennaState {
        antenna: blank_antenna(),
        identified: false,
        records: Vec::new(),
        declared_frequencies: None,
    }
}

/// `START OF ANTENNA`. A block still open is completed first and reported,
/// since only its own `END OF ANTENNA` closes it.
fn open_block(state: &mut ParseState) -> Result<(), AntexError> {
    close_open_block(state)?;
    state.block_seen = true;
    state.current = Some(blank_block());
    Ok(())
}

/// `END OF ANTENNA`. One with no open block carries no value and is passed
/// over.
fn end_block(state: &mut ParseState) -> Result<(), AntexError> {
    if state.current.is_none() {
        return Ok(());
    }
    finalize_antenna(state)
}

/// Complete a block that its own end record did not close, reporting it.
fn close_open_block(state: &mut ParseState) -> Result<(), AntexError> {
    if state.current.is_none() {
        return Ok(());
    }
    push_skip(&mut state.diagnostics, state.line, BLOCK_NOT_CLOSED);
    finalize_antenna(state)
}

/// `TYPE / SERIAL NO` gives the open block its identity. Outside a block it
/// opens one, which is reported, since only `START OF ANTENNA` opens a block.
fn parse_type_serial(line: &str, state: &mut ParseState) -> Result<(), AntexError> {
    const RECORD: &str = "TYPE / SERIAL NO";
    if state.current.is_none() {
        push_skip(&mut state.diagnostics, state.line, OUTSIDE_ANTENNA);
        state.block_seen = true;
        state.current = Some(blank_block());
    }
    let Some(current) = state.current.as_mut() else {
        return Ok(());
    };
    if current.identified {
        claim_once(
            &mut current.records,
            RECORD,
            line,
            Some(&current.antenna.id),
        )?;
        return Ok(());
    }
    claim_once(&mut current.records, RECORD, line, None)?;
    identify_antenna(&mut current.antenna, line);
    current.identified = true;
    Ok(())
}

fn parse_calibration(line: &str, state: &mut ParseState) {
    let Some(current) = state.current.as_mut() else {
        push_skip(&mut state.diagnostics, state.line, OUTSIDE_ANTENNA);
        return;
    };
    let antennas_calibrated = match field(line, 40, 46) {
        None => None,
        Some(text) => match validate::strict_int::<u32>(text, "antex antennas calibrated") {
            Ok(count) => Some(count),
            Err(error) => {
                push_skip(
                    &mut state.diagnostics,
                    state.line,
                    SkipReason::MalformedField(error),
                );
                None
            }
        },
    };
    current.antenna.calibrations.push(Calibration {
        method: raw_field(line, 0, 20).trim_end().to_string(),
        agency: raw_field(line, 20, 40).trim_end().to_string(),
        antennas_calibrated,
        date: raw_field(line, 50, 60).trim_end().to_string(),
    });
}

fn parse_dazi(line: &str, state: &mut ParseState) -> Result<(), AntexError> {
    const RECORD: &str = "DAZI";
    let Some(current) = state.current.as_mut() else {
        push_skip(&mut state.diagnostics, state.line, OUTSIDE_ANTENNA);
        return Ok(());
    };
    if !claim_once(
        &mut current.records,
        RECORD,
        line,
        Some(&current.antenna.id),
    )? {
        return Ok(());
    }
    // ANTEX 1.4 defines DAZI as 2X,F6.1,52X (columns 2..8). If the two leading
    // separator columns are blank, parse the strict slice; otherwise, parse the
    // complete 0..8 compatibility field so leading numeric characters are not
    // truncated into an unintended zero value.
    let (start, end) = if raw_field(line, 0, 2) == "  " {
        (2, 8)
    } else {
        (0, 8)
    };
    let dazi = fortran_f64(line, start, end, "antex dazi").ok_or_else(|| {
        invalid_field(
            Some(&current.antenna.id),
            RECORD,
            "dazi",
            raw_field(line, start, end),
        )
    })?;
    current.antenna.dazi_deg = Some(dazi);
    Ok(())
}

fn parse_zenith_grid(line: &str, state: &mut ParseState) -> Result<(), AntexError> {
    const RECORD: &str = "ZEN1 / ZEN2 / DZEN";
    let Some(current) = state.current.as_mut() else {
        push_skip(&mut state.diagnostics, state.line, OUTSIDE_ANTENNA);
        return Ok(());
    };
    if !claim_once(
        &mut current.records,
        RECORD,
        line,
        Some(&current.antenna.id),
    )? {
        return Ok(());
    }
    let mut values = [0.0; 3];
    for (value, (start, name)) in values
        .iter_mut()
        .zip([(2, "zen1"), (8, "zen2"), (14, "dzen")])
    {
        *value = fortran_f64(line, start, start + 6, "antex zenith grid").ok_or_else(|| {
            invalid_field(
                Some(&current.antenna.id),
                RECORD,
                name,
                raw_field(line, start, start + 6),
            )
        })?;
    }
    let [start_deg, end_deg, step_deg] = values;
    current.antenna.zenith_grid = Some(ZenithGrid {
        start_deg,
        end_deg,
        step_deg,
    });
    Ok(())
}

/// Only the presence of `# OF FREQUENCIES` is retained: the writer states the
/// number of frequency sections. A count that is not an integer, or that
/// disagrees with the sections read, is reported as a skip.
fn parse_frequency_count(line: &str, state: &mut ParseState) -> Result<(), AntexError> {
    let Some(current) = state.current.as_mut() else {
        push_skip(&mut state.diagnostics, state.line, OUTSIDE_ANTENNA);
        return Ok(());
    };
    if !claim_once(
        &mut current.records,
        "# OF FREQUENCIES",
        line,
        Some(&current.antenna.id),
    )? {
        return Ok(());
    }
    current.antenna.has_frequency_count = true;
    match validate::strict_int::<usize>(raw_field(line, 0, 6), "antex # OF FREQUENCIES") {
        Ok(count) => current.declared_frequencies = Some(count),
        Err(error) => push_skip(
            &mut state.diagnostics,
            state.line,
            SkipReason::MalformedField(error),
        ),
    }
    Ok(())
}

fn parse_sinex_code(line: &str, state: &mut ParseState) -> Result<(), AntexError> {
    let Some(current) = state.current.as_mut() else {
        push_skip(&mut state.diagnostics, state.line, OUTSIDE_ANTENNA);
        return Ok(());
    };
    if !claim_once(
        &mut current.records,
        "SINEX CODE",
        line,
        Some(&current.antenna.id),
    )? {
        return Ok(());
    }
    let code = raw_field(line, 0, LABEL_COLUMN).trim();
    if !code.is_empty() {
        current.antenna.sinex_code = Some(code.to_string());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum ValidField {
    From,
    Until,
}

impl ValidField {
    fn record(self) -> &'static str {
        match self {
            Self::From => "VALID FROM",
            Self::Until => "VALID UNTIL",
        }
    }
}

/// Read `VALID FROM` / `VALID UNTIL` by its `5I6,F13.7` columns. Every field
/// must hold a value: a blank, a non-integer in an `I6` column, a seconds
/// field that is not a number, or a component outside the GPS calendar and
/// clock is refused by field name, so a damaged bound never disappears and
/// leaves the block valid for all time on that side.
fn parse_valid(line: &str, state: &mut ParseState, which: ValidField) -> Result<(), AntexError> {
    let record = which.record();
    let Some(current) = state.current.as_mut() else {
        push_skip(&mut state.diagnostics, state.line, OUTSIDE_ANTENNA);
        return Ok(());
    };
    let id = current.antenna.id.as_str();
    if !claim_once(&mut current.records, record, line, Some(id))? {
        return Ok(());
    }
    let texts = [
        raw_field(line, 0, 6).trim(),
        raw_field(line, 6, 12).trim(),
        raw_field(line, 12, 18).trim(),
        raw_field(line, 18, 24).trim(),
        raw_field(line, 24, 30).trim(),
        raw_field(line, 30, 43).trim(),
    ];
    let mut parts = [0_i64; 6];
    for (index, (part, text)) in parts.iter_mut().zip(texts).take(5).enumerate() {
        *part = text
            .parse::<i64>()
            .map_err(|_| invalid_field(Some(id), record, DATETIME_FIELDS[index], text))?;
    }
    let (second, fraction) = parse_gps_seconds(texts[5])
        .ok_or_else(|| invalid_field(Some(id), record, "second", texts[5]))?;
    parts[5] = second;
    check_gps_civil(parts)
        .map_err(|index| invalid_field(Some(id), record, DATETIME_FIELDS[index], texts[index]))?;
    // Every component is in range, so the narrowing casts are exact.
    let dt = AntexDateTime {
        year: parts[0] as i32,
        month: parts[1] as u8,
        day: parts[2] as u8,
        hour: parts[3] as u8,
        minute: parts[4] as u8,
        second: parts[5] as u8,
        fraction,
    };
    match which {
        ValidField::From => current.antenna.valid_from = Some(dt),
        ValidField::Until => current.antenna.valid_until = Some(dt),
    }
    Ok(())
}

fn blank_antenna() -> Antenna {
    Antenna {
        id: String::new(),
        kind: AntennaKind::Receiver,
        antenna_type: String::new(),
        serial: String::new(),
        leading_comments: Vec::new(),
        calibrations: Vec::new(),
        dazi_deg: None,
        zenith_grid: None,
        has_frequency_count: false,
        sinex_code: None,
        valid_from: None,
        valid_until: None,
        comments: Vec::new(),
        frequencies: Vec::new(),
    }
}

/// Set the identity fields a `TYPE / SERIAL NO` line gives.
fn identify_antenna(antenna: &mut Antenna, line: &str) {
    antenna.id = raw_field(line, 0, 60).trim().to_string();
    antenna.antenna_type = raw_field(line, 0, 20).trim().to_string();
    antenna.serial = raw_field(line, 20, 40).trim().to_string();
    antenna.kind = if is_satellite_serial(&antenna.serial) {
        AntennaKind::Satellite
    } else {
        AntennaKind::Receiver
    };
}

fn is_satellite_serial(serial: &str) -> bool {
    let bytes = serial.as_bytes();
    bytes.len() == 3
        && bytes[0].is_ascii_uppercase()
        && bytes[1].is_ascii_digit()
        && bytes[2].is_ascii_digit()
}

/// Open a frequency or RMS section. A section still open is completed first
/// and reported, since only its own end record closes it.
fn begin_section(line: &str, state: &mut ParseState, kind: SectionKind) -> Result<(), AntexError> {
    if state.current.is_none() {
        push_skip(&mut state.diagnostics, state.line, OUTSIDE_ANTENNA);
        return Ok(());
    }
    close_open_section(state)?;
    state.section = Some(SectionState {
        kind,
        frequency: raw_field(line, 0, 20).trim().to_string(),
        pco_m: None,
        pco_body: None,
        samples: Vec::new(),
        occupied: BTreeSet::new(),
    });
    Ok(())
}

/// Close the open section. An end record naming another frequency, or ending
/// the other kind of section, is reported as inconsistent; the section's
/// values are kept either way. An end record with no open section carries no
/// value and is passed over.
fn end_section(line: &str, state: &mut ParseState, kind: SectionKind) -> Result<(), AntexError> {
    let Some(section) = state.section.as_ref() else {
        return Ok(());
    };
    let label = raw_field(line, 0, 20).trim();
    if section.kind != kind {
        push_skip(
            &mut state.diagnostics,
            state.line,
            SkipReason::InconsistentRecord(
                "antex frequency section closed by the wrong end record",
            ),
        );
    }
    if !label.is_empty() && label != section.frequency {
        push_skip(
            &mut state.diagnostics,
            state.line,
            SkipReason::InconsistentRecord(
                "antex frequency section end label differs from its start",
            ),
        );
    }
    finalize_section(state)
}

fn parse_pco(line: &str, state: &mut ParseState) -> Result<(), AntexError> {
    const RECORD: &str = "NORTH / EAST / UP";
    let Some(section) = state.section.as_mut() else {
        push_skip(
            &mut state.diagnostics,
            state.line,
            SkipReason::InconsistentRecord("antex NORTH / EAST / UP outside a frequency section"),
        );
        return Ok(());
    };
    let antenna_id = state.current.as_ref().map(|c| c.antenna.id.as_str());
    let body = raw_field(line, 0, LABEL_COLUMN).trim_end();
    if let Some(seen) = &section.pco_body {
        if seen == body {
            return Ok(());
        }
        return Err(AntexError::RepeatedRecord {
            antenna_id: antenna_id.map(str::to_string),
            record: RECORD,
        });
    }
    let mut pco_m = [0.0; 3];
    for (value, (start, name)) in pco_m
        .iter_mut()
        .zip([(0, "north"), (10, "east"), (20, "up")])
    {
        let mm = fortran_f64(line, start, start + 10, "antex pco").ok_or_else(|| {
            invalid_field(antenna_id, RECORD, name, raw_field(line, start, start + 10))
        })?;
        *value = mm * M_PER_MM;
    }
    section.pco_m = Some(pco_m);
    section.pco_body = Some(body.to_string());
    Ok(())
}

fn parse_row(line: &str, state: &mut ParseState) -> Result<(), AntexError> {
    if line.trim().is_empty() {
        return Ok(());
    }
    if state.section.is_none() {
        // A nonblank line that is neither a record the format defines nor a
        // grid row inside a frequency section is reported rather than passed
        // over.
        let label = tag(line);
        let what = if label.is_empty() {
            raw_field(line, 0, LABEL_COLUMN).trim()
        } else {
            label
        };
        push_skip(
            &mut state.diagnostics,
            state.line,
            SkipReason::UnknownBlock(what.to_string()),
        );
        return Ok(());
    }

    let head = field(line, 0, 8).unwrap_or("");
    if head == "NOAZI" {
        add_pcv_values(None, line, state)
    } else if let Some(azimuth) = fortran_f64(line, 0, 8, "antex azimuth") {
        add_pcv_values(Some(azimuth), line, state)
    } else {
        // A grid row whose head token is neither `NOAZI` nor a parseable azimuth
        // is recorded as a typed skip rather than silently dropped, consistent
        // with the rest of the sans-I/O contract. Real ANTEX rows always carry a
        // recognized head, so a clean file is unaffected.
        push_skip(
            &mut state.diagnostics,
            state.line,
            SkipReason::MalformedField(FieldError::FloatParse {
                field: "antex pcv row head",
                value: head.to_string(),
            }),
        );
        Ok(())
    }
}

/// Place one grid row's values at `ZEN1 + k * DZEN`, `k` counting the row's
/// 8-column fields. Blank fields leave their position empty. A row cannot be
/// read before the block's grid is declared, a value past the first needs a
/// positive `DZEN` to have a zenith of its own, and a grid position is filled
/// at most once per section; each of these is refused by name rather than
/// stacking several values on one zenith.
fn add_pcv_values(
    azimuth_deg: Option<f64>,
    line: &str,
    state: &mut ParseState,
) -> Result<(), AntexError> {
    let Some(current) = state.current.as_ref() else {
        return Ok(());
    };
    let Some(section) = state.section.as_mut() else {
        return Ok(());
    };
    let degenerate = |reason: String| AntexError::DegenerateGrid {
        antenna_id: current.antenna.id.clone(),
        frequency: section.frequency.clone(),
        reason,
    };
    let Some(grid) = current.antenna.zenith_grid else {
        return Err(degenerate(
            "PCV row precedes the ZEN1 / ZEN2 / DZEN record".to_string(),
        ));
    };

    let grid_start = grid.start_deg;
    let grid_step = grid.step_deg;
    let row_key = azimuth_deg.map(f64::to_bits);
    let row_name = match azimuth_deg {
        None => "NOAZI".to_string(),
        Some(azimuth) => format!("azimuth {azimuth}"),
    };

    let mut values = Vec::new();
    let mut k = 0;
    while 8 + 8 * k < line.len() {
        let start = 8 + 8 * k;
        let end = start + 8;
        if let Some(val_str) = field(line, start, end) {
            match fortran_f64(line, start, end, "antex pcv value") {
                Some(value) => {
                    if k > 0 && grid_step.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                        return Err(degenerate(format!(
                            "{row_name} row has a value at zenith index {k} but DZEN {grid_step} is not positive"
                        )));
                    }
                    if section.occupied.contains(&(row_key, k)) {
                        return Err(degenerate(format!(
                            "{row_name} row repeats zenith index {k} already filled in this section"
                        )));
                    }
                    values.push((k, value));
                }
                None => {
                    // A malformed PCV grid value is skipped with a typed reason rather
                    // than silently dropped or replaced by a fabricated default. The
                    // remaining valid samples on the row are still recovered.
                    push_skip(
                        &mut state.diagnostics,
                        state.line,
                        SkipReason::MalformedField(FieldError::FloatParse {
                            field: "antex pcv value",
                            value: val_str.to_string(),
                        }),
                    );
                }
            }
        }
        k += 1;
    }

    for (k, value) in values {
        section.occupied.insert((row_key, k));
        let zenith_deg = if grid_step == 0.0 {
            grid_start
        } else {
            grid_start + grid_step * k as f64
        };
        section.samples.push(PcvSample {
            grid: if azimuth_deg.is_some() {
                PcvGrid::Azimuth
            } else {
                PcvGrid::NoAzimuth
            },
            azimuth_deg,
            zenith_deg,
            value_m: value * M_PER_MM,
        });
    }
    Ok(())
}

fn finalize_section(state: &mut ParseState) -> Result<(), AntexError> {
    let Some(section) = state.section.take() else {
        return Ok(());
    };
    let Some(current) = state.current.as_mut() else {
        return Ok(());
    };
    match section.kind {
        SectionKind::Values => {
            let pco_m = section.pco_m.ok_or_else(|| AntexError::MissingPco {
                antenna_id: current.antenna.id.clone(),
                frequency: section.frequency.clone(),
            })?;
            current.antenna.frequencies.push(Frequency {
                frequency: section.frequency,
                pco_m,
                pcv_samples: section.samples,
                rms: None,
            });
        }
        SectionKind::Rms => {
            let target = current
                .antenna
                .frequencies
                .iter_mut()
                .rev()
                .find(|f| f.frequency == section.frequency && f.rms.is_none());
            match target {
                Some(frequency) => {
                    frequency.rms = Some(FrequencyRms {
                        pco_m: section.pco_m,
                        pcv_samples: section.samples,
                    });
                }
                None => push_skip(
                    &mut state.diagnostics,
                    state.line,
                    SkipReason::InconsistentRecord(
                        "antex FREQ RMS section has no earlier frequency section to belong to",
                    ),
                ),
            }
        }
    }
    Ok(())
}

/// Complete a section that its own end record did not close, reporting it.
fn close_open_section(state: &mut ParseState) -> Result<(), AntexError> {
    if state.section.is_none() {
        return Ok(());
    }
    push_skip(&mut state.diagnostics, state.line, SECTION_NOT_CLOSED);
    finalize_section(state)
}

fn finalize_antenna(state: &mut ParseState) -> Result<(), AntexError> {
    close_open_section(state)?;
    let Some(current) = state.current.take() else {
        return Ok(());
    };
    if !current.identified {
        // A block without `TYPE / SERIAL NO` has no id to be kept under.
        push_skip(
            &mut state.diagnostics,
            state.line,
            SkipReason::InconsistentRecord("antex antenna block without TYPE / SERIAL NO"),
        );
        return Ok(());
    }
    if current
        .declared_frequencies
        .is_some_and(|declared| declared != current.antenna.frequencies.len())
    {
        push_skip(
            &mut state.diagnostics,
            state.line,
            SkipReason::InconsistentRecord(
                "antex # OF FREQUENCIES differs from the frequency sections read",
            ),
        );
    }
    let antenna = current.antenna;
    let intervals = state
        .antenna_intervals
        .entry(antenna.id.clone())
        .or_default();
    state
        .block_order
        .push((antenna.id.clone(), intervals.len()));
    intervals.push(antenna.clone());
    state.antennas.insert(antenna.id.clone(), antenna);
    Ok(())
}

fn interpolate_azimuth(
    antenna_id: &str,
    frequency: &str,
    azimuth_samples: &BTreeMap<OrderedF64, Vec<(f64, f64)>>,
    azimuth_deg: f64,
    zenith_deg: f64,
) -> Result<f64, AntexError> {
    let azimuth = antenna::normalize_azimuth(azimuth_deg);
    let azimuths: Vec<f64> = azimuth_samples.keys().map(|az| az.0).collect();
    let (low_deg, high_deg) = antenna::azimuth_bracket(&azimuths, azimuth);

    let low_samples = &azimuth_samples[&OrderedF64(low_deg)];
    let high_samples = &azimuth_samples[&OrderedF64(high_deg)];

    let low_value = interpolate(antenna_id, frequency, low_samples, zenith_deg)?;
    let high_value = interpolate(antenna_id, frequency, high_samples, zenith_deg)?;

    Ok(antenna::blend_azimuth(
        low_deg, high_deg, azimuth, low_value, high_value,
    ))
}

fn interpolate(
    antenna_id: &str,
    frequency: &str,
    samples: &[(f64, f64)],
    zenith_deg: f64,
) -> Result<f64, AntexError> {
    if samples.is_empty() {
        return Err(AntexError::EmptyPcvGrid {
            antenna_id: antenna_id.to_string(),
            frequency: frequency.to_string(),
        });
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

    antenna::interpolate_zenith_sorted(&sorted, zenith_deg).ok_or_else(|| {
        AntexError::EmptyPcvGrid {
            antenna_id: antenna_id.to_string(),
            frequency: frequency.to_string(),
        }
    })
}

fn tag(line: &str) -> &str {
    raw_field(line, 60, 80).trim()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn test_antenna() -> Antenna {
        Antenna {
            id: "TESTANT             TESTSER".to_string(),
            kind: AntennaKind::Receiver,
            antenna_type: "TESTANT".to_string(),
            serial: "TESTSER".to_string(),
            calibrations: Vec::new(),
            leading_comments: Vec::new(),
            dazi_deg: Some(0.0),
            zenith_grid: Some(ZenithGrid {
                start_deg: 0.0,
                end_deg: 10.0,
                step_deg: 10.0,
            }),
            has_frequency_count: false,
            sinex_code: None,
            valid_from: None,
            valid_until: None,
            comments: Vec::new(),
            frequencies: vec![Frequency {
                frequency: "G01".to_string(),
                pco_m: [0.0, 0.0, 0.0],
                pcv_samples: vec![
                    PcvSample {
                        grid: PcvGrid::NoAzimuth,
                        azimuth_deg: None,
                        zenith_deg: 0.0,
                        value_m: 1.0,
                    },
                    PcvSample {
                        grid: PcvGrid::NoAzimuth,
                        azimuth_deg: None,
                        zenith_deg: 10.0,
                        value_m: 3.0,
                    },
                ],
                rms: None,
            }],
        }
    }

    fn grid(antenna: &mut Antenna) -> &mut ZenithGrid {
        antenna.zenith_grid.as_mut().expect("test antenna grid")
    }

    fn synchronized_antex(antenna: Antenna) -> Antex {
        let mut antex = Antex {
            header: AntexHeader::default(),
            antennas: BTreeMap::new(),
            antenna_intervals: BTreeMap::new(),
            block_order: vec![(antenna.id.clone(), 0)],
            outer_comments: Vec::new(),
            skipped_records: 0,
        };
        antex
            .antenna_intervals
            .insert(antenna.id.clone(), vec![antenna.clone()]);
        antex.antennas.insert(antenna.id.clone(), antenna);
        antex
    }

    #[test]
    fn pcv_rejects_nonfinite_zenith() {
        let antenna = test_antenna();
        for zenith_deg in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                antenna.pcv("G01", zenith_deg, None),
                Err(AntexError::InvalidInput {
                    field: "zenith_deg",
                    reason: "not finite"
                })
            );
        }
    }

    #[test]
    fn pcv_rejects_out_of_range_zenith() {
        let antenna = test_antenna();
        assert_eq!(
            antenna.pcv("G01", 11.0, None),
            Err(AntexError::InvalidInput {
                field: "zenith_deg",
                reason: "out of range"
            })
        );
    }

    #[test]
    fn pcv_accepts_valid_zenith_unchanged() {
        let antenna = test_antenna();
        let got = antenna.pcv("G01", 5.0, None).expect("valid PCV");
        assert_eq!(got.to_bits(), 2.0_f64.to_bits());
    }

    fn line(prefix: &str, tag: &str) -> String {
        format!("{prefix:<60}{tag}")
    }

    fn synthetic_block() -> String {
        [
            line("", "START OF ANTENNA"),
            line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
            line("     0.0", "DAZI"),
            line("     0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN"),
            line("IGS_TEST", "SINEX CODE"),
            line("  2020     1     1     0     0    0.0000000", "VALID FROM"),
            line("  2021    12    31    23    59   59.0000000", "VALID UNTIL"),
            line("G01", "START OF FREQUENCY"),
            line("      1.50      2.00      3.00", "NORTH / EAST / UP"),
            "   NOAZI    1.00    2.00    3.00".to_string(),
            "     0.0    1.00    2.00    3.00".to_string(),
            "    90.0    4.00    5.00    6.00".to_string(),
            line("", "END OF FREQUENCY"),
            line("", "END OF ANTENNA"),
        ]
        .join("\n")
    }

    #[test]
    fn encode_round_trips_synthetic_block() {
        let antex = Antex::parse(&synthetic_block()).expect("parse synthetic block");
        assert_eq!(antex.skipped_records(), 0);
        let encoded = antex.encode().expect("encode synthetic block");
        let reparsed = Antex::parse(&encoded).expect("re-parse encoded block");
        assert_eq!(antex, reparsed);
        assert_eq!(
            encoded,
            reparsed.encode().expect("re-encode synthetic block")
        );
    }

    #[test]
    fn malformed_pcv_value_is_skipped_not_silent() {
        let text = [
            line("", "START OF ANTENNA"),
            line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
            line("     0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN"),
            line("G01", "START OF FREQUENCY"),
            line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
            "   NOAZI    1.00     BAD    3.00".to_string(),
            line("", "END OF FREQUENCY"),
            line("", "END OF ANTENNA"),
        ]
        .join("\n");

        let antex = Antex::parse(&text).expect("forgiving parse");
        assert_eq!(antex.skipped_records(), 1);
        let antenna = antex
            .antenna("TESTANT             TESTSER")
            .expect("antenna");
        let frequency = antenna.frequency("G01").unwrap();
        assert_eq!(frequency.pcv_samples.len(), 2);
    }

    #[test]
    fn fixed_column_pcv_adjacent_values_and_blank_fields() {
        let text = [
            line("", "START OF ANTENNA"),
            line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
            line("     0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN"),
            line("G01", "START OF FREQUENCY"),
            line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
            "   NOAZI12345.67-1234.56    3.00".to_string(),
            "     0.0    1.00            2.00".to_string(),
            line("", "END OF FREQUENCY"),
            line("", "END OF ANTENNA"),
        ]
        .join("\n");

        let antex = Antex::parse(&text).expect("parse fixed-column PCV");
        assert_eq!(antex.skipped_records(), 0);
        let antenna = antex
            .antenna("TESTANT             TESTSER")
            .expect("antenna");
        let frequency = antenna.frequency("G01").unwrap();

        let noazi_samples: Vec<&PcvSample> = frequency
            .pcv_samples
            .iter()
            .filter(|s| s.grid == PcvGrid::NoAzimuth)
            .collect();
        assert_eq!(noazi_samples.len(), 3);
        assert_eq!(noazi_samples[0].zenith_deg, 0.0);
        assert_eq!(noazi_samples[0].value_m * MM_PER_M, 12345.67);
        assert_eq!(noazi_samples[1].zenith_deg, 5.0);
        assert_eq!(noazi_samples[1].value_m * MM_PER_M, -1234.56);
        assert_eq!(noazi_samples[2].zenith_deg, 10.0);
        assert_eq!(noazi_samples[2].value_m * MM_PER_M, 3.00);

        let azi_samples: Vec<&PcvSample> = frequency
            .pcv_samples
            .iter()
            .filter(|s| s.grid == PcvGrid::Azimuth)
            .collect();
        assert_eq!(azi_samples.len(), 2);
        assert_eq!(azi_samples[0].zenith_deg, 0.0);
        assert_eq!(azi_samples[0].value_m * MM_PER_M, 1.00);
        assert_eq!(azi_samples[1].zenith_deg, 10.0);
        assert_eq!(azi_samples[1].value_m * MM_PER_M, 2.00);

        let encoded = antex.encode().expect("encode fixed column");
        let reparsed = Antex::parse(&encoded).expect("re-parse encoded");
        assert_eq!(antex, reparsed);
    }

    #[test]
    fn antex_ignores_comments_inside_frequency_block() {
        let text = [
            line("     1.4            M", "ANTEX VERSION / SYST"),
            line("Header comment", "COMMENT"),
            line("", "END OF HEADER"),
            line("", "START OF ANTENNA"),
            line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
            line("     0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN"),
            line("G01", "START OF FREQUENCY"),
            line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
            line("Calibration performed on 2026-09-21", "COMMENT"),
            "   NOAZI    1.00    2.00    3.00".to_string(),
            line("", "END OF FREQUENCY"),
            line("", "END OF ANTENNA"),
        ]
        .join("\n");

        let antex = Antex::parse(&text).expect("parse ANTEX with comments");
        assert_eq!(antex.skipped_records(), 0);
        let antenna = antex
            .antenna("TESTANT             TESTSER")
            .expect("antenna");
        let frequency = antenna.frequency("G01").unwrap();
        assert_eq!(frequency.pcv_samples.len(), 3);
    }

    #[test]
    fn dazi_standard_layout_and_compact_compatibility_regression() {
        fn dazi_block(dazi_field: &str) -> String {
            [
                line("", "START OF ANTENNA"),
                line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
                line(dazi_field, "DAZI"),
                line("     0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN"),
                line("G01", "START OF FREQUENCY"),
                line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
                "   NOAZI    1.00    2.00    3.00".to_string(),
                line("", "END OF FREQUENCY"),
                line("", "END OF ANTENNA"),
            ]
            .join("\n")
        }

        let antex_std_5 = Antex::parse(&dazi_block("     5.0")).expect("parse standard 5.0");
        let ant_std_5 = antex_std_5.antenna("TESTANT             TESTSER").unwrap();
        assert_eq!(ant_std_5.dazi_deg, Some(5.0));

        let antex_std_10 = Antex::parse(&dazi_block("    10.0")).expect("parse standard 10.0");
        let ant_std_10 = antex_std_10.antenna("TESTANT             TESTSER").unwrap();
        assert_eq!(ant_std_10.dazi_deg, Some(10.0));

        let antex_std_0 = Antex::parse(&dazi_block("     0.0")).expect("parse standard 0.0");
        let ant_std_0 = antex_std_0.antenna("TESTANT             TESTSER").unwrap();
        assert_eq!(ant_std_0.dazi_deg, Some(0.0));

        let antex_compact_5 = Antex::parse(&dazi_block("5.0")).expect("parse compact 5.0");
        let ant_compact_5 = antex_compact_5
            .antenna("TESTANT             TESTSER")
            .unwrap();

        let antex_compact_space_5 =
            Antex::parse(&dazi_block(" 5.0")).expect("parse compact space 5.0");
        let ant_compact_space_5 = antex_compact_space_5
            .antenna("TESTANT             TESTSER")
            .unwrap();

        let antex_compact_10 = Antex::parse(&dazi_block("10.0")).expect("parse compact 10.0");
        let ant_compact_10 = antex_compact_10
            .antenna("TESTANT             TESTSER")
            .unwrap();

        let antex_compact_0 = Antex::parse(&dazi_block("0.0")).expect("parse compact 0.0");
        let ant_compact_0 = antex_compact_0
            .antenna("TESTANT             TESTSER")
            .unwrap();

        let antex_compact_space_0 =
            Antex::parse(&dazi_block(" 0.0")).expect("parse compact space 0.0");
        let ant_compact_space_0 = antex_compact_space_0
            .antenna("TESTANT             TESTSER")
            .unwrap();

        assert_eq!(ant_compact_5.dazi_deg, ant_std_5.dazi_deg);
        assert_eq!(ant_compact_5.dazi_deg, Some(5.0));
        assert_ne!(ant_compact_5.dazi_deg, Some(0.0));

        assert_eq!(ant_compact_space_5.dazi_deg, ant_std_5.dazi_deg);
        assert_eq!(ant_compact_space_5.dazi_deg, Some(5.0));
        assert_ne!(ant_compact_space_5.dazi_deg, Some(0.0));

        assert_eq!(ant_compact_10.dazi_deg, ant_std_10.dazi_deg);
        assert_eq!(ant_compact_10.dazi_deg, Some(10.0));
        assert_ne!(ant_compact_10.dazi_deg, Some(0.0));

        assert_eq!(ant_compact_0.dazi_deg, ant_std_0.dazi_deg);
        assert_eq!(ant_compact_0.dazi_deg, Some(0.0));

        assert_eq!(ant_compact_space_0.dazi_deg, ant_std_0.dazi_deg);
        assert_eq!(ant_compact_space_0.dazi_deg, Some(0.0));

        let encoded_compact = antex_compact_5.encode().expect("encode compact");
        let reparsed = Antex::parse(&encoded_compact).expect("re-parse encoded compact");
        assert_eq!(reparsed.skipped_records(), 0);
        let ant_reparsed = reparsed.antenna("TESTANT             TESTSER").unwrap();
        assert_eq!(ant_reparsed.dazi_deg, Some(5.0));
        assert!(encoded_compact
            .contains("     5.0                                                    DAZI"));
    }

    #[test]
    fn pcv_row_emits_full_declared_grid_preserving_trailing_and_interior_gaps() {
        let text = [
            line("", "START OF ANTENNA"),
            line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
            line("     0.0", "DAZI"),
            line("     0.0  90.0   5.0", "ZEN1 / ZEN2 / DZEN"),
            line("G01", "START OF FREQUENCY"),
            line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
            "   NOAZI    1.00    2.00    3.00            5.00    6.00    7.00    8.00    9.00   10.00   11.00   12.00   13.00   14.00   15.00".to_string(),
            "     0.0    1.10    2.10    3.10            5.10    6.10    7.10    8.10    9.10   10.10   11.10   12.10   13.10   14.10   15.10".to_string(),
            line("", "END OF FREQUENCY"),
            line("", "END OF ANTENNA"),
        ]
        .join("\n");

        let antex = Antex::parse(&text).expect("parse 0..90/5 grid with gaps");
        assert_eq!(antex.skipped_records(), 0);

        let encoded = antex.encode().expect("encode 0..90/5 block");

        // Inspect encoded NOAZI row
        let noazi_line = encoded
            .lines()
            .find(|l| l.starts_with("   NOAZI"))
            .expect("NOAZI line present");
        assert_eq!(&noazi_line[0..8], "   NOAZI");
        let noazi_values = &noazi_line[8..];
        assert_eq!(noazi_values.len(), 19 * 8);

        let noazi_fields: Vec<&str> = (0..19).map(|i| &noazi_values[i * 8..(i + 1) * 8]).collect();
        assert_eq!(noazi_fields[0], "    1.00");
        assert_eq!(noazi_fields[1], "    2.00");
        assert_eq!(noazi_fields[2], "    3.00");
        assert_eq!(noazi_fields[3], "        ");
        assert_eq!(noazi_fields[14], "   15.00");
        assert_eq!(noazi_fields[15], "        ");
        assert_eq!(noazi_fields[16], "        ");
        assert_eq!(noazi_fields[17], "        ");
        assert_eq!(noazi_fields[18], "        ");

        // Inspect encoded azimuth row (skip past NOAZI so DAZI header line is not matched first)
        let noazi_idx = encoded
            .lines()
            .position(|l| l.starts_with("   NOAZI"))
            .expect("NOAZI line present");
        let azi_line = encoded
            .lines()
            .skip(noazi_idx + 1)
            .find(|l| l.starts_with("     0.0"))
            .expect("azimuth line present");
        assert_eq!(&azi_line[0..8], "     0.0");
        let azi_values = &azi_line[8..];
        assert_eq!(azi_values.len(), 19 * 8);
        let azi_fields: Vec<&str> = (0..19).map(|i| &azi_values[i * 8..(i + 1) * 8]).collect();
        assert_eq!(azi_fields[3], "        ");
        assert_eq!(azi_fields[14], "   15.10");
        assert_eq!(azi_fields[15], "        ");
        assert_eq!(azi_fields[16], "        ");
        assert_eq!(azi_fields[17], "        ");
        assert_eq!(azi_fields[18], "        ");

        let reparsed = Antex::parse(&encoded).expect("re-parse encoded 0..90/5 block");
        assert_eq!(reparsed.skipped_records(), 0);
        let ant = reparsed.antenna("TESTANT             TESTSER").unwrap();
        let freq = ant.frequency("G01").unwrap();

        let noazi_samples: Vec<&PcvSample> = freq
            .pcv_samples
            .iter()
            .filter(|s| s.grid == PcvGrid::NoAzimuth)
            .collect();
        assert_eq!(noazi_samples.len(), 14);

        let zeniths: Vec<f64> = noazi_samples.iter().map(|s| s.zenith_deg).collect();
        assert_eq!(
            zeniths,
            vec![0.0, 5.0, 10.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
        );
        for gap_zenith in [15.0, 75.0, 80.0, 85.0, 90.0] {
            assert!(
                noazi_samples.iter().all(|s| s.zenith_deg != gap_zenith),
                "zenith {gap_zenith} should be absent"
            );
        }

        assert_eq!(noazi_samples[0].value_m * MM_PER_M, 1.00);
        assert_eq!(noazi_samples[13].value_m * MM_PER_M, 15.00);

        assert_eq!(antex, reparsed);
        assert_eq!(encoded, reparsed.encode().expect("re-encode 0..90/5 block"));
    }

    #[test]
    fn pcv_row_nonzero_zen1_grid_indexing_and_interior_gap() {
        let text = [
            line("", "START OF ANTENNA"),
            line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
            line("     0.0", "DAZI"),
            line("    10.0  50.0   5.0", "ZEN1 / ZEN2 / DZEN"),
            line("G01", "START OF FREQUENCY"),
            line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
            "   NOAZI    2.00    3.00            4.00    5.00".to_string(),
            "     0.0    2.50    3.50            4.50    5.50".to_string(),
            line("", "END OF FREQUENCY"),
            line("", "END OF ANTENNA"),
        ]
        .join("\n");

        let antex = Antex::parse(&text).expect("parse nonzero ZEN1 grid");
        assert_eq!(antex.skipped_records(), 0);

        let encoded = antex.encode().expect("encode nonzero ZEN1 grid");

        let noazi_line = encoded
            .lines()
            .find(|l| l.starts_with("   NOAZI"))
            .expect("NOAZI line present");
        let noazi_values = &noazi_line[8..];
        assert_eq!(noazi_values.len(), 9 * 8);
        let fields: Vec<&str> = (0..9).map(|i| &noazi_values[i * 8..(i + 1) * 8]).collect();
        assert_eq!(fields[0], "    2.00");
        assert_eq!(fields[1], "    3.00");
        assert_eq!(fields[2], "        ");
        assert_eq!(fields[3], "    4.00");
        assert_eq!(fields[4], "    5.00");
        assert_eq!(fields[5], "        ");
        assert_eq!(fields[6], "        ");
        assert_eq!(fields[7], "        ");
        assert_eq!(fields[8], "        ");

        let reparsed = Antex::parse(&encoded).expect("re-parse nonzero ZEN1 block");
        assert_eq!(reparsed.skipped_records(), 0);
        let ant = reparsed.antenna("TESTANT             TESTSER").unwrap();
        let freq = ant.frequency("G01").unwrap();

        let noazi_samples: Vec<&PcvSample> = freq
            .pcv_samples
            .iter()
            .filter(|s| s.grid == PcvGrid::NoAzimuth)
            .collect();
        assert_eq!(noazi_samples.len(), 4);

        assert_eq!(noazi_samples[0].zenith_deg, 10.0);
        assert_eq!(noazi_samples[0].value_m * MM_PER_M, 2.00);

        assert_eq!(noazi_samples[1].zenith_deg, 15.0);
        assert_eq!(noazi_samples[1].value_m * MM_PER_M, 3.00);

        assert_eq!(noazi_samples[2].zenith_deg, 25.0);
        assert_eq!(noazi_samples[2].value_m * MM_PER_M, 4.00);

        assert_eq!(noazi_samples[3].zenith_deg, 30.0);
        assert_eq!(noazi_samples[3].value_m * MM_PER_M, 5.00);

        assert_eq!(antex, reparsed);
        assert_eq!(
            encoded,
            reparsed.encode().expect("re-encode nonzero ZEN1 block")
        );
    }

    #[test]
    fn pcv_row_retains_samples_outside_declared_grid() {
        let text = [
            line("", "START OF ANTENNA"),
            line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
            line("     0.0", "DAZI"),
            line("     0.0  20.0  10.0", "ZEN1 / ZEN2 / DZEN"),
            line("G01", "START OF FREQUENCY"),
            line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
            "   NOAZI    1.00    2.00    3.00    4.00".to_string(),
            line("", "END OF FREQUENCY"),
            line("", "END OF ANTENNA"),
        ]
        .join("\n");

        let antex = Antex::parse(&text).expect("parse grid with sample beyond ZEN2");
        assert_eq!(antex.skipped_records(), 0);

        let encoded = antex.encode().expect("encode grid with sample beyond ZEN2");
        let noazi_line = encoded
            .lines()
            .find(|l| l.starts_with("   NOAZI"))
            .expect("NOAZI line present");
        let noazi_values = &noazi_line[8..];
        assert_eq!(noazi_values.len(), 4 * 8);

        let reparsed = Antex::parse(&encoded).expect("re-parse block");
        let ant = reparsed.antenna("TESTANT             TESTSER").unwrap();
        let freq = ant.frequency("G01").unwrap();
        let zeniths: Vec<f64> = freq.pcv_samples.iter().map(|s| s.zenith_deg).collect();
        assert_eq!(zeniths, vec![0.0, 10.0, 20.0, 30.0]);
        assert_eq!(antex, reparsed);
        assert_eq!(encoded, reparsed.encode().expect("re-encode block"));
    }

    #[test]
    fn encode_refuses_divergent_public_antennas_map() {
        let mut antex = synchronized_antex(test_antenna());
        antex
            .antennas
            .get_mut("TESTANT             TESTSER")
            .unwrap()
            .zenith_grid
            .as_mut()
            .unwrap()
            .step_deg = 5.0;
        let err = antex.encode().expect_err("divergent public map must error");
        assert!(matches!(
            err,
            AntexError::Unwritable {
                field: "antennas",
                ..
            }
        ));

        let mut antex_removed = synchronized_antex(test_antenna());
        antex_removed.antennas.clear();
        let err_removed = antex_removed
            .encode()
            .expect_err("empty public map must error");
        assert!(matches!(
            err_removed,
            AntexError::Unwritable {
                field: "antennas",
                ..
            }
        ));
    }

    #[test]
    fn encode_refuses_off_grid_zenith_coordinate() {
        let mut ant = test_antenna();
        ant.frequencies[0].pcv_samples.push(PcvSample {
            grid: PcvGrid::NoAzimuth,
            azimuth_deg: None,
            zenith_deg: 7.3,
            value_m: 0.001,
        });
        let antex = synchronized_antex(ant);
        let err = antex.encode().expect_err("off-grid sample must be refused");
        match err {
            AntexError::Unwritable { field, reason } => {
                assert_eq!(field, "pcv_samples");
                assert!(reason.contains("off-grid"));
            }
            other => panic!("expected Unwritable, got {other:?}"),
        }
    }

    #[test]
    fn encode_refuses_sample_below_zen1() {
        let mut ant = test_antenna();
        grid(&mut ant).start_deg = 10.0;
        grid(&mut ant).end_deg = 20.0;
        grid(&mut ant).step_deg = 10.0;
        ant.frequencies[0].pcv_samples = vec![
            PcvSample {
                grid: PcvGrid::NoAzimuth,
                azimuth_deg: None,
                zenith_deg: 5.0,
                value_m: 0.001,
            },
            PcvSample {
                grid: PcvGrid::NoAzimuth,
                azimuth_deg: None,
                zenith_deg: 10.0,
                value_m: 0.002,
            },
        ];
        let antex = synchronized_antex(ant);
        let err = antex
            .encode()
            .expect_err("sample below ZEN1 must be refused");
        match err {
            AntexError::Unwritable { field, reason } => {
                assert_eq!(field, "pcv_samples");
                assert!(reason.contains("below ZEN1"));
            }
            other => panic!("expected Unwritable, got {other:?}"),
        }
    }

    #[test]
    fn encode_refuses_negative_sample_zenith() {
        let mut ant = test_antenna();
        ant.frequencies[0].pcv_samples.insert(
            0,
            PcvSample {
                grid: PcvGrid::NoAzimuth,
                azimuth_deg: None,
                zenith_deg: -5.0,
                value_m: 0.001,
            },
        );
        let antex = synchronized_antex(ant);
        let err = antex
            .encode()
            .expect_err("negative sample zenith must be refused");
        match err {
            AntexError::Unwritable { field, reason } => {
                assert_eq!(field, "pcv_samples");
                assert!(reason.contains("negative"));
            }
            other => panic!("expected Unwritable, got {other:?}"),
        }
    }

    #[test]
    fn encode_refuses_duplicate_sample_at_same_grid_position() {
        let mut ant = test_antenna();
        ant.frequencies[0].pcv_samples.insert(
            1,
            PcvSample {
                grid: PcvGrid::NoAzimuth,
                azimuth_deg: None,
                zenith_deg: 0.0,
                value_m: 0.002,
            },
        );
        let antex = synchronized_antex(ant);
        let err = antex
            .encode()
            .expect_err("duplicate sample must be refused");
        match err {
            AntexError::Unwritable { field, reason } => {
                assert_eq!(field, "pcv_samples");
                assert!(reason.contains("duplicate"));
            }
            other => panic!("expected Unwritable, got {other:?}"),
        }
    }

    #[test]
    fn encode_refuses_wrong_sample_metadata_and_vector_order() {
        let mut ant = test_antenna();
        ant.frequencies[0].pcv_samples[0].azimuth_deg = Some(0.0);
        let err = synchronized_antex(ant)
            .encode()
            .expect_err("NoAzimuth with azimuth must fail");
        assert!(matches!(
            err,
            AntexError::Unwritable {
                field: "pcv_samples",
                ..
            }
        ));

        let mut ant2 = test_antenna();
        ant2.frequencies[0].pcv_samples[0].grid = PcvGrid::Azimuth;
        ant2.frequencies[0].pcv_samples[0].azimuth_deg = None;
        let err2 = synchronized_antex(ant2)
            .encode()
            .expect_err("Azimuth without azimuth must fail");
        assert!(matches!(
            err2,
            AntexError::Unwritable {
                field: "pcv_samples",
                ..
            }
        ));

        let mut ant3 = test_antenna();
        ant3.frequencies[0].pcv_samples.swap(0, 1);
        let err3 = synchronized_antex(ant3)
            .encode()
            .expect_err("out of order zenith must fail");
        assert!(matches!(
            err3,
            AntexError::Unwritable {
                field: "pcv_samples",
                ..
            }
        ));

        let mut ant4 = test_antenna();
        let freq = &mut ant4.frequencies[0];
        freq.pcv_samples.push(PcvSample {
            grid: PcvGrid::Azimuth,
            azimuth_deg: Some(0.0),
            zenith_deg: 0.0,
            value_m: 0.001,
        });
        freq.pcv_samples.push(PcvSample {
            grid: PcvGrid::NoAzimuth,
            azimuth_deg: None,
            zenith_deg: 20.0,
            value_m: 0.001,
        });
        grid(&mut ant4).end_deg = 20.0;
        let err4 = synchronized_antex(ant4)
            .encode()
            .expect_err("NOAZI after Azimuth must fail");
        assert!(matches!(
            err4,
            AntexError::Unwritable {
                field: "pcv_samples",
                ..
            }
        ));
    }

    #[test]
    fn encode_refuses_precision_loss_and_overflow() {
        let mut ant = test_antenna();
        ant.frequencies[0].pco_m[0] = 0.0012345;
        let err = synchronized_antex(ant)
            .encode()
            .expect_err("PCO precision loss must fail");
        assert!(matches!(err, AntexError::Unwritable { field: "pco_m", .. }));

        let mut ant_pco_overflow = test_antenna();
        ant_pco_overflow.frequencies[0].pco_m[0] = 10_000.0;
        let err_pco_overflow = synchronized_antex(ant_pco_overflow)
            .encode()
            .expect_err("PCO overflow must fail");
        assert!(matches!(
            err_pco_overflow,
            AntexError::Unwritable { field: "pco_m", .. }
        ));

        let mut ant2 = test_antenna();
        ant2.frequencies[0].pcv_samples[0].value_m = 0.0012345;
        let err2 = synchronized_antex(ant2)
            .encode()
            .expect_err("PCV precision loss must fail");
        assert!(matches!(
            err2,
            AntexError::Unwritable {
                field: "pcv_samples",
                ..
            }
        ));

        let mut ant_pcv_overflow = test_antenna();
        ant_pcv_overflow.frequencies[0].pcv_samples[0].value_m = 100.0;
        let err_pcv_overflow = synchronized_antex(ant_pcv_overflow)
            .encode()
            .expect_err("PCV overflow must fail");
        assert!(matches!(
            err_pcv_overflow,
            AntexError::Unwritable {
                field: "pcv_samples",
                ..
            }
        ));

        let mut ant3 = test_antenna();
        ant3.frequencies[0].pco_m[0] = f64::INFINITY;
        let err3 = synchronized_antex(ant3)
            .encode()
            .expect_err("Inf PCO must fail");
        assert!(matches!(
            err3,
            AntexError::Unwritable { field: "pco_m", .. }
        ));

        let mut ant4 = test_antenna();
        ant4.frequencies[0].pcv_samples[0].value_m = f64::INFINITY;
        let err4 = synchronized_antex(ant4)
            .encode()
            .expect_err("Inf PCV must fail");
        assert!(matches!(
            err4,
            AntexError::Unwritable {
                field: "pcv_samples",
                ..
            }
        ));

        let mut ant_nonfinite_zen = test_antenna();
        ant_nonfinite_zen.frequencies[0].pcv_samples[0].zenith_deg = f64::INFINITY;
        let err_nonfinite_zen = synchronized_antex(ant_nonfinite_zen)
            .encode()
            .expect_err("nonfinite sample zenith must fail");
        assert!(matches!(
            err_nonfinite_zen,
            AntexError::Unwritable {
                field: "pcv_samples",
                ..
            }
        ));

        let mut ant_nonfinite_azi = test_antenna();
        ant_nonfinite_azi.frequencies[0].pcv_samples = vec![PcvSample {
            grid: PcvGrid::Azimuth,
            azimuth_deg: Some(f64::INFINITY),
            zenith_deg: 0.0,
            value_m: 0.001,
        }];
        let err_nonfinite_azi = synchronized_antex(ant_nonfinite_azi)
            .encode()
            .expect_err("nonfinite azimuth must fail");
        assert!(matches!(
            err_nonfinite_azi,
            AntexError::Unwritable {
                field: "azimuth_deg",
                ..
            }
        ));

        let mut ant5 = test_antenna();
        grid(&mut ant5).step_deg = 0.05;
        let err5 = synchronized_antex(ant5)
            .encode()
            .expect_err("F6.1 precision loss must fail");
        assert!(matches!(
            err5,
            AntexError::Unwritable {
                field: "zenith_step_deg",
                ..
            }
        ));

        let mut ant_hdr_overflow = test_antenna();
        ant_hdr_overflow.dazi_deg = Some(10000.0);
        let err_hdr_overflow = synchronized_antex(ant_hdr_overflow)
            .encode()
            .expect_err("header F6.1 overflow must fail");
        assert!(matches!(
            err_hdr_overflow,
            AntexError::Unwritable {
                field: "dazi_deg",
                ..
            }
        ));

        let mut ant_azi_prec = test_antenna();
        ant_azi_prec.frequencies[0].pcv_samples = vec![PcvSample {
            grid: PcvGrid::Azimuth,
            azimuth_deg: Some(45.25),
            zenith_deg: 0.0,
            value_m: 0.001,
        }];
        let err_azi_prec = synchronized_antex(ant_azi_prec)
            .encode()
            .expect_err("azimuth precision loss must fail");
        assert!(matches!(
            err_azi_prec,
            AntexError::Unwritable {
                field: "azimuth_deg",
                ..
            }
        ));

        let mut ant_azi_overflow = test_antenna();
        ant_azi_overflow.frequencies[0].pcv_samples = vec![PcvSample {
            grid: PcvGrid::Azimuth,
            azimuth_deg: Some(1000000.0),
            zenith_deg: 0.0,
            value_m: 0.001,
        }];
        let err_azi_overflow = synchronized_antex(ant_azi_overflow)
            .encode()
            .expect_err("azimuth width overflow must fail");
        assert!(matches!(
            err_azi_overflow,
            AntexError::Unwritable {
                field: "azimuth_deg",
                ..
            }
        ));

        let mut ant_zen_overflow = test_antenna();
        ant_zen_overflow.frequencies[0].pcv_samples = vec![PcvSample {
            grid: PcvGrid::NoAzimuth,
            azimuth_deg: None,
            zenith_deg: 10000.0,
            value_m: 0.001,
        }];
        let err_zen_overflow = synchronized_antex(ant_zen_overflow)
            .encode()
            .expect_err("sample zenith beyond max grid index must fail");
        assert!(matches!(
            err_zen_overflow,
            AntexError::Unwritable {
                field: "pcv_samples",
                ..
            }
        ));

        let mut antex_nan = synchronized_antex(test_antenna());
        antex_nan
            .antennas
            .get_mut("TESTANT             TESTSER")
            .unwrap()
            .frequencies[0]
            .pco_m[0] = f64::NAN;
        let err_nan = antex_nan
            .encode()
            .expect_err("NaN PCO mutation must refuse encode");
        assert!(matches!(
            err_nan,
            AntexError::Unwritable {
                field: "antennas",
                ..
            }
        ));

        let mut ant_nan_sync = test_antenna();
        ant_nan_sync.frequencies[0].pco_m[0] = f64::NAN;
        let err_nan_sync = synchronized_antex(ant_nan_sync)
            .encode()
            .expect_err("synchronized NaN PCO must pass sync and get named refusal from encoder");
        assert!(matches!(
            err_nan_sync,
            AntexError::Unwritable { field: "pco_m", .. }
        ));

        let mut antex_dazi_zero = synchronized_antex(test_antenna());
        antex_dazi_zero
            .antennas
            .get_mut("TESTANT             TESTSER")
            .unwrap()
            .dazi_deg = Some(-0.0);
        let err_dazi = antex_dazi_zero
            .encode()
            .expect_err("signed zero DAZI mutation must refuse encode");
        assert!(matches!(
            err_dazi,
            AntexError::Unwritable {
                field: "antennas",
                ..
            }
        ));
    }

    #[test]
    fn encode_refuses_invalid_bounds_and_step() {
        let mut ant = test_antenna();
        grid(&mut ant).start_deg = 90.0;
        grid(&mut ant).end_deg = 0.0;
        let err = synchronized_antex(ant)
            .encode()
            .expect_err("inverted bounds must fail");
        assert!(matches!(
            err,
            AntexError::Unwritable {
                field: "zenith_end_deg",
                ..
            }
        ));

        let mut ant2 = test_antenna();
        grid(&mut ant2).step_deg = 0.0;
        let err2 = synchronized_antex(ant2)
            .encode()
            .expect_err("zero step with samples must fail");
        assert!(matches!(
            err2,
            AntexError::Unwritable {
                field: "zenith_step_deg",
                ..
            }
        ));

        let mut ant3 = test_antenna();
        grid(&mut ant3).step_deg = 1e-15;
        let err3 = synchronized_antex(ant3)
            .encode()
            .expect_err("subnormal step must fail");
        assert!(matches!(
            err3,
            AntexError::Unwritable {
                field: "zenith_step_deg",
                ..
            }
        ));
    }

    #[test]
    fn extreme_fraction_scale_is_ordered_and_refused_without_overflow() {
        // One unit at the largest scale: every exponent the writer forms is
        // computed without overflow, and none fits the 13-column field.
        let tiny = SecondFraction::new(1, u64::MAX).expect("1 / 10^u64::MAX is below one");
        assert_eq!(tiny.scale(), u64::MAX);
        assert_eq!(tiny.nanoseconds(), None);
        assert_eq!(seconds_text(0, tiny), None);
        let wide = SecondFraction::new(123_456_789_012_345_678, u64::MAX).expect("below one");
        assert_eq!(seconds_text(0, wide), None);

        assert!(tiny > SecondFraction::ZERO);
        assert!(tiny < wide);
        assert!(tiny < SecondFraction::from_nanoseconds(1).expect("one nanosecond"));

        let mut ant = test_antenna();
        ant.valid_from =
            Some(AntexDateTime::new_with_fraction(2020, 1, 1, 0, 0, 0, tiny).expect("GPS instant"));
        assert!(matches!(
            synchronized_antex(ant).encode(),
            Err(AntexError::Unwritable {
                field: "valid_from",
                ..
            })
        ));
    }

    #[test]
    fn encode_refuses_off_grid_zenith_end_header() {
        let mut ant = test_antenna();
        grid(&mut ant).start_deg = 0.0;
        grid(&mut ant).end_deg = 10.0;
        grid(&mut ant).step_deg = 3.0;
        let err = synchronized_antex(ant)
            .encode()
            .expect_err("off-grid zenith_end_deg must fail");
        match err {
            AntexError::Unwritable { field, reason } => {
                assert_eq!(field, "zenith_end_deg");
                assert!(reason.contains("not an integer multiple"));
            }
            other => panic!("expected Unwritable, got {other:?}"),
        }
    }
}
