//! CCSDS Tracking Data Message KVN reader and writer.
//!
//! This module implements the CCSDS 503.0-B-2 KVN form as a sans-IO parser and
//! serializer. Date/time fields remain raw strings, matching the other NDM
//! readers in this crate. Observable values are kept as both the parsed `f64`
//! and the exact decimal token read from the message, so frequency-domain
//! records such as `RECEIVE_FREQ` and `TRANSMIT_FREQ_n` re-emit without decimal
//! rewriting.

use std::collections::{HashMap, HashSet};
use std::fmt;

const VERSION_KEY: &str = "CCSDS_TDM_VERS";
/// The keywords CCSDS 503.0-B-2 4.2.5 c) excepts from the `keyword = value`
/// syntax. None of them can carry a value, in any section.
const EXCEPTED_KEYWORDS: [&str; 5] = [
    COMMENT_KEY,
    "META_START",
    "META_STOP",
    "DATA_START",
    "DATA_STOP",
];
const COMMENT_KEY: &str = "COMMENT";
/// The order table 3-2 fixes for header keywords, which 3.2.3 makes binding.
const HEADER_ORDER: [&str; 5] = [
    VERSION_KEY,
    COMMENT_KEY,
    "CREATION_DATE",
    "ORIGINATOR",
    "MESSAGE_ID",
];

/// The order table 3-3 fixes for metadata keywords, which 3.3.1.8 makes
/// binding, by family: an indexed keyword ranks where its base does.
const METADATA_ORDER: [&str; 33] = [
    COMMENT_KEY,
    "TRACK_ID",
    "DATA_TYPES",
    "TIME_SYSTEM",
    "START_TIME",
    "STOP_TIME",
    "PARTICIPANT",
    "MODE",
    "PATH",
    "EPHEMERIS_NAME",
    "TRANSMIT_BAND",
    "RECEIVE_BAND",
    "TURNAROUND_NUMERATOR",
    "TURNAROUND_DENOMINATOR",
    "TIMETAG_REF",
    "INTEGRATION_INTERVAL",
    "INTEGRATION_REF",
    "FREQ_OFFSET",
    "RANGE_MODE",
    "RANGE_MODULUS",
    "RANGE_UNITS",
    "ANGLE_TYPE",
    "REFERENCE_FRAME",
    "INTERPOLATION",
    "INTERPOLATION_DEGREE",
    "DOPPLER_COUNT_BIAS",
    "DOPPLER_COUNT_SCALE",
    "DOPPLER_COUNT_ROLLOVER",
    "TRANSMIT_DELAY",
    "RECEIVE_DELAY",
    "DATA_QUALITY",
    "CORRECTION",
    "CORRECTIONS_APPLIED",
];

/// The longest line CCSDS 503.0-B-2 4.2.1 allows, excluding its terminator.
const MAX_LINE_CHARACTERS: usize = 254;

/// A parsed CCSDS Tracking Data Message.
#[derive(Debug, Clone, PartialEq)]
pub struct Tdm {
    /// The `CCSDS_TDM_VERS` header value.
    pub version: String,
    /// Header comments in parse order.
    pub comments: Vec<String>,
    /// The optional `CREATION_DATE` header value.
    pub creation_date: Option<String>,
    /// The optional `ORIGINATOR` header value.
    pub originator: Option<String>,
    /// The optional `MESSAGE_ID` header value.
    pub message_id: Option<String>,
    /// Header fields that are not part of the common modeled header.
    pub header_fields: Vec<TdmField>,
    /// Metadata/data segments in message order.
    pub segments: Vec<TdmSegment>,
}

/// One TDM segment, consisting of one metadata block and one data block.
#[derive(Debug, Clone, PartialEq)]
pub struct TdmSegment {
    /// Metadata describing the records in this segment.
    pub metadata: TdmMetadata,
    /// Tracking data records in this segment.
    pub data: TdmDataSection,
}

/// A KVN key/value field preserved in parse order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdmField {
    /// The KVN keyword.
    pub key: String,
    /// The trimmed KVN value.
    pub value: String,
}

/// Metadata extracted from a TDM `META_START` / `META_STOP` block.
#[derive(Debug, Clone, PartialEq)]
pub struct TdmMetadata {
    /// Metadata comments in parse order.
    pub comments: Vec<String>,
    /// Raw metadata fields in parse order.
    pub fields: Vec<TdmField>,
    /// Parsed `PARTICIPANT_n` entries.
    pub participants: Vec<TdmParticipant>,
    /// The optional `MODE` metadata value.
    pub mode: Option<String>,
    /// Parsed `PATH`, `PATH_1`, and `PATH_2` entries.
    pub paths: Vec<TdmPath>,
    /// The optional `TIMETAG_REF` metadata value.
    pub timetag_ref: Option<String>,
    /// The optional `TIME_SYSTEM` metadata value.
    pub time_system: Option<String>,
    /// The range unit for `RANGE` records, defaulting to kilometers when absent.
    pub range_units: TdmUnit,
}

impl TdmMetadata {
    /// Return the last metadata value for `key`.
    pub fn get_last(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .rev()
            .find(|field| field.key == key)
            .map(|field| field.value.as_str())
            .filter(|value| !value.is_empty())
    }
}

/// One named tracking participant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdmParticipant {
    /// The numeric suffix from `PARTICIPANT_n`.
    pub index: u8,
    /// The participant name.
    pub name: String,
}

/// A parsed signal path from `PATH`, `PATH_1`, or `PATH_2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdmPath {
    /// The original path keyword.
    pub key: String,
    /// The path suffix for `PATH_n`, or `None` for the unindexed `PATH`.
    pub index: Option<u8>,
    /// Participant indices listed in path order.
    pub participants: Vec<u8>,
}

/// A data-section comment and where it sits among the records.
///
/// 4.5.2 c) puts a data-section comment between `DATA_START` and the first
/// record, so a conforming message gives every comment `before_record` of 0.
/// The position is kept anyway: a message read under a policy that forgives the
/// placement writes back with each comment where it was, rather than gathered
/// to the top of the block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdmComment {
    /// The comment text: everything after the keyword and the space 4.5.3
    /// requires.
    pub text: String,
    /// The index of the record this comment precedes, which is the record count
    /// for a comment after the last one.
    pub before_record: usize,
}

/// A TDM data block.
#[derive(Debug, Clone, PartialEq)]
pub struct TdmDataSection {
    /// Data-section comments in parse order, each with its position.
    pub comments: Vec<TdmComment>,
    /// Data records in parse order.
    pub records: Vec<TdmDataRecord>,
}

/// One time-tagged tracking data record.
#[derive(Debug, Clone, PartialEq)]
pub struct TdmDataRecord {
    /// The parsed observable family.
    pub observable: TdmObservable,
    /// The original data keyword.
    pub keyword: String,
    /// The raw epoch string.
    pub epoch: String,
    /// The numeric observable value.
    pub value: TdmScalar,
    /// The unit assigned by CCSDS 503.0-B-2.
    pub unit: TdmUnit,
}

/// A numeric record value plus the exact decimal token used to encode it.
#[derive(Debug, Clone, PartialEq)]
pub struct TdmScalar {
    /// The exact decimal or scientific-notation token read from the KVN record.
    pub text: String,
    /// The parsed finite `f64` value.
    pub value: f64,
}

/// Observable families used by TDM tracking data records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TdmObservable {
    /// A `RANGE` record.
    Range,
    /// A `DOPPLER_INSTANTANEOUS` record.
    DopplerInstantaneous,
    /// A `DOPPLER_INTEGRATED` record.
    DopplerIntegrated,
    /// A `RECEIVE_FREQ` or `RECEIVE_FREQ_n` record.
    ReceiveFreq {
        /// The participant suffix from `RECEIVE_FREQ_n`, if present.
        participant: Option<u8>,
    },
    /// A `TRANSMIT_FREQ` or `TRANSMIT_FREQ_n` record.
    TransmitFreq {
        /// The participant suffix from `TRANSMIT_FREQ_n`, if present.
        participant: Option<u8>,
    },
    /// A `TRANSMIT_FREQ_RATE` or `TRANSMIT_FREQ_RATE_n` record.
    TransmitFreqRate {
        /// The participant suffix from `TRANSMIT_FREQ_RATE_n`, if present.
        participant: Option<u8>,
    },
    /// An `ANGLE_1` record.
    Angle1,
    /// An `ANGLE_2` record.
    Angle2,
    /// A TDM data keyword not modeled as a dedicated enum variant.
    Other(String),
}

/// Units attached to TDM data records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TdmUnit {
    /// Kilometers.
    Kilometers,
    /// Seconds.
    Seconds,
    /// CCSDS range units.
    RangeUnits,
    /// Kilometers per second.
    KilometersPerSecond,
    /// Hertz.
    Hertz,
    /// Hertz per second.
    HertzPerSecond,
    /// Degrees.
    Degrees,
    /// Decibel watts.
    DecibelWatts,
    /// Decibel hertz.
    DecibelHertz,
    /// Square meters.
    SquareMeters,
    /// Meters.
    Meters,
    /// Seconds per second.
    SecondsPerSecond,
    /// Percent.
    Percent,
    /// Kelvin.
    Kelvin,
    /// Hectopascals.
    Hectopascals,
    /// Total electron content units.
    TotalElectronContentUnits,
    /// Dimensionless quantity.
    Dimensionless,
    /// A unit label not modeled by this enum.
    Unknown(String),
}

impl TdmUnit {
    /// Return the canonical unit label.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Kilometers => "km",
            Self::Seconds => "s",
            Self::RangeUnits => "RU",
            Self::KilometersPerSecond => "km/s",
            Self::Hertz => "Hz",
            Self::HertzPerSecond => "Hz/s",
            Self::Degrees => "deg",
            Self::DecibelWatts => "dBW",
            Self::DecibelHertz => "dBHz",
            Self::SquareMeters => "m**2",
            Self::Meters => "m",
            Self::SecondsPerSecond => "s/s",
            Self::Percent => "%",
            Self::Kelvin => "K",
            Self::Hectopascals => "hPa",
            Self::TotalElectronContentUnits => "TECU",
            Self::Dimensionless => "n/a",
            Self::Unknown(label) => label.as_str(),
        }
    }
}

/// Whether the TDM reader forgives one kind of departure from CCSDS 503.0-B-2
/// or refuses the message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TdmLeniency {
    /// Refuse the message, naming what was wrong.
    #[default]
    Strict,
    /// Read the message and report the departure as a [`TdmWarning`].
    Forgive,
}

/// The policies the TDM reader applies to departures from CCSDS 503.0-B-2.
///
/// The default refuses every departure. A field set to
/// [`TdmLeniency::Forgive`] forgives one that does not change what a value
/// means, and the read reports it as a [`TdmWarning`] naming the line.
///
/// Nothing that changes what the message means is forgivable under any policy:
/// a value that does not parse, a unit that contradicts table 3-5, a keyword
/// the standard does not define and whose meaning would have to be invented,
/// duplicate keys with conflicting values, a structural error such as a nested
/// or unmatched block, and a `COMMENT` used as an assignment key. A reader that
/// guesses a value is worse than one that refuses. `PARTICIPANT_6` and up are
/// refused under every policy for the same reason: 3.3.1.11 allows them only by
/// arrangement outside the message, so a `PATH` entry naming an index the
/// message cannot resolve would leave the reader to guess which participant a
/// measurement belongs to.
///
/// The writer stays strict whatever the reader forgave. Reading a message
/// leniently does not let [`encode_kvn`] emit a non-conforming one: it refuses
/// what the standard forbids, and the [`TdmWarning`] from the read is what says
/// why. A message read under a lenient policy either writes back unchanged or
/// is refused by name, and is never quietly repaired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct TdmPolicy {
    /// Characters outside the printable ASCII 4.2.1 allows.
    pub non_printable: TdmLeniency,
    /// Keywords tables 3-2 and 3-3 mark mandatory that the message omits.
    pub missing_keywords: TdmLeniency,
    /// Lines longer than the 254 characters 4.2.1 allows.
    pub long_lines: TdmLeniency,
    /// Data sections holding none of the records 3.1.3 requires.
    pub empty_data_sections: TdmLeniency,
    /// A keyword's records out of the chronological order 3.4.10 requires.
    pub record_order: TdmLeniency,
    /// A keyword and timetag pair repeating, which 3.4.11 forbids.
    pub duplicate_records: TdmLeniency,
    /// Keywords out of the order tables 3-2 and 3-3 fix.
    pub keyword_order: TdmLeniency,
    /// A final line with none of the terminators 4.2.11 requires.
    pub final_terminator: TdmLeniency,
}

impl TdmPolicy {
    /// This policy with `non_printable`.
    #[must_use]
    pub const fn with_non_printable(mut self, non_printable: TdmLeniency) -> Self {
        self.non_printable = non_printable;
        self
    }

    /// This policy with `missing_keywords`.
    #[must_use]
    pub const fn with_missing_keywords(mut self, missing_keywords: TdmLeniency) -> Self {
        self.missing_keywords = missing_keywords;
        self
    }

    /// This policy with `long_lines`.
    #[must_use]
    pub const fn with_long_lines(mut self, long_lines: TdmLeniency) -> Self {
        self.long_lines = long_lines;
        self
    }

    /// This policy with `empty_data_sections`.
    #[must_use]
    pub const fn with_empty_data_sections(mut self, empty_data_sections: TdmLeniency) -> Self {
        self.empty_data_sections = empty_data_sections;
        self
    }

    /// A policy that forgives nothing, the same as [`TdmPolicy::default`] but
    /// usable in a constant.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            non_printable: TdmLeniency::Strict,
            missing_keywords: TdmLeniency::Strict,
            long_lines: TdmLeniency::Strict,
            empty_data_sections: TdmLeniency::Strict,
            record_order: TdmLeniency::Strict,
            duplicate_records: TdmLeniency::Strict,
            keyword_order: TdmLeniency::Strict,
            final_terminator: TdmLeniency::Strict,
        }
    }

    /// This policy with `final_terminator`.
    #[must_use]
    pub const fn with_final_terminator(mut self, final_terminator: TdmLeniency) -> Self {
        self.final_terminator = final_terminator;
        self
    }

    /// This policy with `keyword_order`.
    #[must_use]
    pub const fn with_keyword_order(mut self, keyword_order: TdmLeniency) -> Self {
        self.keyword_order = keyword_order;
        self
    }

    /// This policy with `duplicate_records`.
    #[must_use]
    pub const fn with_duplicate_records(mut self, duplicate_records: TdmLeniency) -> Self {
        self.duplicate_records = duplicate_records;
        self
    }

    /// This policy with `record_order`.
    #[must_use]
    pub const fn with_record_order(mut self, record_order: TdmLeniency) -> Self {
        self.record_order = record_order;
        self
    }

    /// This policy with every forgivable departure forgiven.
    #[must_use]
    pub const fn lenient() -> Self {
        Self {
            non_printable: TdmLeniency::Forgive,
            missing_keywords: TdmLeniency::Forgive,
            long_lines: TdmLeniency::Forgive,
            empty_data_sections: TdmLeniency::Forgive,
            record_order: TdmLeniency::Forgive,
            duplicate_records: TdmLeniency::Forgive,
            keyword_order: TdmLeniency::Forgive,
            final_terminator: TdmLeniency::Forgive,
        }
    }
}

/// A departure from CCSDS 503.0-B-2 the reader forgave under a [`TdmPolicy`]
/// instead of refusing the message.
///
/// Each names where the departure is, so a caller can report it or decide the
/// message is not good enough. The corresponding [`TdmError`] is what a strict
/// read returns for the same input.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TdmWarning {
    /// A line held a character outside the printable ASCII 4.2.1 allows.
    NonPrintableCharacter {
        /// One-based input line number.
        line: usize,
        /// The line's first whitespace-delimited token, which is the keyword
        /// the writer names for the same departure.
        keyword: String,
        /// One-based character position within the line.
        column: usize,
        /// The offending character.
        character: char,
    },
    /// A line was longer than the 254 characters 4.2.1 allows.
    LineTooLong {
        /// One-based input line number.
        line: usize,
        /// The line's first whitespace-delimited token, which is the keyword
        /// the writer names for the same departure.
        keyword: String,
        /// The line's length in characters, excluding its terminator.
        length: usize,
    },
    /// A keyword appeared twice in one block carrying the same value, which
    /// 4.2.5 a) gives one value assignment.
    RepeatedKeyword {
        /// One-based input line number of the repeat.
        line: usize,
        /// The repeated keyword.
        keyword: String,
        /// The section it appeared in, `header` or `metadata`.
        section: &'static str,
    },
    /// A keyword CCSDS 503.0-B-2 marks mandatory was absent.
    MissingKeyword {
        /// The absent keyword.
        keyword: String,
        /// The one-based segment that required it, or `None` for the header.
        segment: Option<usize>,
    },
    /// A data section held none of the records 3.1.3 requires.
    EmptyDataSection {
        /// The one-based segment index.
        segment: usize,
    },
    /// A keyword's records were not in the chronological order 3.4.10 requires.
    RecordsOutOfOrder {
        /// One-based segment index.
        segment: usize,
        /// The keyword whose records go backwards.
        keyword: String,
        /// The timetag that goes back.
        epoch: String,
    },
    /// The last line carried none of the terminators 4.2.11 requires.
    UnterminatedFinalLine {
        /// One-based number of the unterminated line.
        line: usize,
    },
    /// A keyword appeared before one the table for its section orders earlier.
    KeywordOutOfOrder {
        /// One-based input line number.
        line: usize,
        /// The keyword out of place.
        keyword: String,
        /// The section it appeared in, `header` or `metadata`.
        section: &'static str,
    },
    /// A keyword and timetag pair repeated, which 3.4.11 forbids. Both records
    /// are kept, in the order the file gives them.
    DuplicateRecord {
        /// One-based segment index.
        segment: usize,
        /// The repeated keyword.
        keyword: String,
        /// The repeated timetag.
        epoch: String,
    },
}

impl fmt::Display for TdmWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPrintableCharacter {
                line,
                keyword: _,
                column,
                character,
            } => write!(
                f,
                "TDM line {line} column {column} holds {character:?}, which is not printable ASCII"
            ),
            Self::LineTooLong {
                line,
                keyword: _,
                length,
            } => write!(
                f,
                "TDM line {line} is {length} characters, over the {MAX_LINE_CHARACTERS} allowed"
            ),
            Self::RepeatedKeyword {
                line,
                keyword,
                section,
            } => write!(
                f,
                "TDM {section} keyword {keyword} repeats at line {line} with the same value"
            ),
            Self::MissingKeyword {
                keyword,
                segment: Some(segment),
            } => write!(f, "missing TDM {keyword} in segment {segment}"),
            Self::MissingKeyword {
                keyword,
                segment: None,
            } => write!(f, "missing TDM {keyword}"),
            Self::EmptyDataSection { segment } => {
                write!(f, "TDM segment {segment} holds no tracking data record")
            }
            Self::RecordsOutOfOrder {
                segment,
                keyword,
                epoch,
            } => write!(
                f,
                "TDM segment {segment} gives {keyword} at {epoch} after a later one"
            ),
            Self::DuplicateRecord {
                segment,
                keyword,
                epoch,
            } => write!(f, "TDM segment {segment} repeats {keyword} at {epoch}"),
            Self::KeywordOutOfOrder {
                line,
                keyword,
                section,
            } => write!(
                f,
                "TDM {section} keyword {keyword} at line {line} is out of the order its table fixes"
            ),
            Self::UnterminatedFinalLine { line } => {
                write!(f, "TDM line {line} carries no terminator")
            }
        }
    }
}

/// The policies the TDM writer applies to departures from CCSDS 503.0-B-2.
///
/// The default emits none, so [`encode_kvn`] writes a conforming message or
/// refuses the value. A field set to [`TdmLeniency::Forgive`] lets the writer
/// emit one departure that a reader under the matching [`TdmPolicy`] forgives,
/// and [`encode_kvn_with_policy`] returns each one it emitted. The two policies
/// mirror each other deliberately: a message read leniently can be written back
/// by asking for the same departures, and the list that comes back says exactly
/// what makes the file non-conforming.
///
/// Nothing outside that mirror is emittable under any policy. A field keyed
/// `COMMENT` or another keyword 4.2.5 c) excepts, a key holding an equals sign
/// or whitespace, a comment carrying a newline, a value that does not parse as
/// its keyword's type: each produces a file that reads back as something other
/// than the value written, which no policy can make correct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct TdmWritePolicy {
    /// Characters outside the printable ASCII 4.2.1 allows.
    pub non_printable: TdmLeniency,
    /// Keywords tables 3-2 and 3-3 mark mandatory that the value omits.
    pub missing_keywords: TdmLeniency,
    /// Lines longer than the 254 characters 4.2.1 allows.
    pub long_lines: TdmLeniency,
    /// Data sections holding none of the records 3.1.3 requires.
    pub empty_data_sections: TdmLeniency,
    /// A keyword's records out of the chronological order 3.4.10 requires.
    pub record_order: TdmLeniency,
    /// A keyword and timetag pair repeating, which 3.4.11 forbids.
    pub duplicate_records: TdmLeniency,
    /// Keywords out of the order tables 3-2 and 3-3 fix.
    pub keyword_order: TdmLeniency,
    /// A final line with none of the terminators 4.2.11 requires.
    pub final_terminator: TdmLeniency,
    /// A keyword written twice in one block with the same value.
    ///
    /// This axis has no counterpart on [`TdmPolicy`]. No file in the 53-file
    /// corpus repeats a keyword within a block, so a reader axis would be dead;
    /// the writer has one because it is choosing to emit the line rather than
    /// forgiving one someone else wrote. A reader meeting the repeat takes the
    /// message whatever the policy says, since both lines say the same thing,
    /// and reports [`TdmWarning::RepeatedKeyword`]. A writer is choosing to
    /// produce the line, so a conforming write does not.
    pub repeated_keywords: TdmLeniency,
}

impl TdmWritePolicy {
    /// A policy that emits no departure, the same as [`TdmWritePolicy::default`]
    /// but usable in a constant.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            non_printable: TdmLeniency::Strict,
            missing_keywords: TdmLeniency::Strict,
            long_lines: TdmLeniency::Strict,
            empty_data_sections: TdmLeniency::Strict,
            record_order: TdmLeniency::Strict,
            duplicate_records: TdmLeniency::Strict,
            keyword_order: TdmLeniency::Strict,
            final_terminator: TdmLeniency::Strict,
            repeated_keywords: TdmLeniency::Strict,
        }
    }

    /// A policy that emits every departure the mirror allows.
    #[must_use]
    pub const fn lenient() -> Self {
        Self {
            non_printable: TdmLeniency::Forgive,
            missing_keywords: TdmLeniency::Forgive,
            long_lines: TdmLeniency::Forgive,
            empty_data_sections: TdmLeniency::Forgive,
            record_order: TdmLeniency::Forgive,
            duplicate_records: TdmLeniency::Forgive,
            keyword_order: TdmLeniency::Forgive,
            final_terminator: TdmLeniency::Forgive,
            repeated_keywords: TdmLeniency::Forgive,
        }
    }

    /// The reader policy this write policy mirrors.
    ///
    /// The writer holds a value to the rules the reader holds a file to by
    /// running the reader's own checks over what it is about to write, which
    /// needs the axes in the reader's vocabulary. Every field is named here
    /// rather than copied with `..`, so an axis added to one policy and not the
    /// other stops the build.
    #[must_use]
    pub const fn as_read(self) -> TdmPolicy {
        TdmPolicy {
            non_printable: self.non_printable,
            missing_keywords: self.missing_keywords,
            long_lines: self.long_lines,
            empty_data_sections: self.empty_data_sections,
            record_order: self.record_order,
            duplicate_records: self.duplicate_records,
            keyword_order: self.keyword_order,
            final_terminator: self.final_terminator,
        }
    }

    /// This policy with `repeated_keywords`.
    #[must_use]
    pub const fn with_repeated_keywords(mut self, repeated_keywords: TdmLeniency) -> Self {
        self.repeated_keywords = repeated_keywords;
        self
    }

    /// This policy with `non_printable`.
    #[must_use]
    pub const fn with_non_printable(mut self, non_printable: TdmLeniency) -> Self {
        self.non_printable = non_printable;
        self
    }

    /// This policy with `missing_keywords`.
    #[must_use]
    pub const fn with_missing_keywords(mut self, missing_keywords: TdmLeniency) -> Self {
        self.missing_keywords = missing_keywords;
        self
    }

    /// This policy with `long_lines`.
    #[must_use]
    pub const fn with_long_lines(mut self, long_lines: TdmLeniency) -> Self {
        self.long_lines = long_lines;
        self
    }

    /// This policy with `empty_data_sections`.
    #[must_use]
    pub const fn with_empty_data_sections(mut self, empty_data_sections: TdmLeniency) -> Self {
        self.empty_data_sections = empty_data_sections;
        self
    }

    /// This policy with `record_order`.
    #[must_use]
    pub const fn with_record_order(mut self, record_order: TdmLeniency) -> Self {
        self.record_order = record_order;
        self
    }

    /// This policy with `duplicate_records`.
    #[must_use]
    pub const fn with_duplicate_records(mut self, duplicate_records: TdmLeniency) -> Self {
        self.duplicate_records = duplicate_records;
        self
    }

    /// This policy with `keyword_order`.
    #[must_use]
    pub const fn with_keyword_order(mut self, keyword_order: TdmLeniency) -> Self {
        self.keyword_order = keyword_order;
        self
    }

    /// This policy with `final_terminator`.
    #[must_use]
    pub const fn with_final_terminator(mut self, final_terminator: TdmLeniency) -> Self {
        self.final_terminator = final_terminator;
        self
    }
}

/// A departure from CCSDS 503.0-B-2 the writer emitted under a
/// [`TdmWritePolicy`].
///
/// Each names what makes the written file non-conforming, in the vocabulary
/// [`TdmWarning`] uses for the same departure on the way in. A reader under the
/// matching [`TdmPolicy`] forgives exactly these.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TdmDeparture {
    /// A written line holds a character outside printable ASCII.
    NonPrintableCharacter {
        /// The keyword whose line holds it.
        keyword: String,
        /// The offending character.
        character: char,
    },
    /// A written line is longer than the 254 characters 4.2.1 allows.
    LineTooLong {
        /// The keyword whose line is too long.
        keyword: String,
        /// The line's length in characters.
        length: usize,
    },
    /// A keyword CCSDS 503.0-B-2 marks mandatory is absent from the message.
    MissingKeyword {
        /// The absent keyword.
        keyword: String,
        /// The one-based segment that required it, or `None` for the header.
        segment: Option<usize>,
    },
    /// A data section is written holding none of the records 3.1.3 requires.
    EmptyDataSection {
        /// The one-based segment index.
        segment: usize,
    },
    /// A keyword's records are written out of chronological order.
    RecordsOutOfOrder {
        /// One-based segment index.
        segment: usize,
        /// The keyword whose records go backwards.
        keyword: String,
        /// The timetag that goes back.
        epoch: String,
    },
    /// A keyword and timetag pair is written twice.
    DuplicateRecord {
        /// One-based segment index.
        segment: usize,
        /// The repeated keyword.
        keyword: String,
        /// The repeated timetag.
        epoch: String,
    },
    /// A keyword the writer emitted twice in one block with the same value.
    RepeatedKeyword {
        /// The repeated keyword.
        keyword: String,
        /// The section it was written in, `header` or `metadata`.
        section: &'static str,
    },
    /// A keyword is written out of the order its table fixes.
    KeywordOutOfOrder {
        /// The keyword out of place.
        keyword: String,
        /// The section it is written in, `header` or `metadata`.
        section: &'static str,
    },
    /// The last line is written with no terminator.
    UnterminatedFinalLine,
}

impl fmt::Display for TdmDeparture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPrintableCharacter { keyword, character } => write!(
                f,
                "TDM {keyword} is written with {character:?}, which is not printable ASCII"
            ),
            Self::LineTooLong { keyword, length } => write!(
                f,
                "TDM {keyword} is written as {length} characters, over the {MAX_LINE_CHARACTERS} allowed"
            ),
            Self::MissingKeyword {
                keyword,
                segment: Some(segment),
            } => write!(f, "TDM segment {segment} is written without {keyword}"),
            Self::MissingKeyword {
                keyword,
                segment: None,
            } => write!(f, "TDM written without {keyword}"),
            Self::EmptyDataSection { segment } => write!(
                f,
                "TDM segment {segment} is written with no tracking data record"
            ),
            Self::RecordsOutOfOrder {
                segment,
                keyword,
                epoch,
            } => write!(
                f,
                "TDM segment {segment} writes {keyword} at {epoch} after a later one"
            ),
            Self::DuplicateRecord {
                segment,
                keyword,
                epoch,
            } => write!(f, "TDM segment {segment} writes {keyword} at {epoch} twice"),
            Self::RepeatedKeyword { keyword, section } => write!(
                f,
                "TDM {section} writes {keyword} twice with the same value"
            ),
            Self::KeywordOutOfOrder { keyword, section } => write!(
                f,
                "TDM {section} writes {keyword} out of the order its table fixes"
            ),
            Self::UnterminatedFinalLine => {
                write!(f, "TDM is written with no terminator on its last line")
            }
        }
    }
}

/// Boundary validation failure category for TDM parsing and encoding.
///
/// Marked `#[non_exhaustive]`: the CCSDS 503.0-B-2 audit adds categories as it
/// covers more of the standard, and the annotation keeps each addition additive
/// for callers, who match with a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TdmInputErrorKind {
    /// A required field or token was absent.
    Missing,
    /// A floating-point value could not be parsed.
    FloatParse,
    /// A floating-point value was NaN or infinite.
    NonFinite,
    /// A positive field was zero or negative.
    NotPositive,
    /// A numeric value was outside the CCSDS domain for that keyword.
    OutOfRange,
    /// An indexed keyword or path component did not contain a valid integer.
    InvalidIndex,
    /// A TDM data keyword is not defined by CCSDS 503.0-B-2 table 3-5.
    UnknownKeyword,
    /// A displayed unit was present even though TDM KVN units are table-defined.
    UnexpectedUnit,
    /// An integer-valued field contained a fractional value.
    NonInteger,
    /// A non-negative field contained a negative value.
    Negative,
    /// A numeric token used a negative zero form.
    NegativeZero,
    /// A record unit does not match CCSDS 503.0-B-2 table 3-5.
    UnitMismatch,
    /// The stored decimal token and `f64` value do not parse to the same bits.
    DecimalMismatch,
}

impl fmt::Display for TdmInputErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Missing => "missing",
            Self::FloatParse => "invalid float",
            Self::NonFinite => "not finite",
            Self::NotPositive => "not positive",
            Self::OutOfRange => "out of range",
            Self::InvalidIndex => "invalid index",
            Self::UnknownKeyword => "unknown keyword",
            Self::UnexpectedUnit => "unexpected unit",
            Self::NonInteger => "not an integer",
            Self::Negative => "negative",
            Self::NegativeZero => "negative zero",
            Self::UnitMismatch => "unit mismatch",
            Self::DecimalMismatch => "decimal mismatch",
        };
        f.write_str(label)
    }
}

/// Failure modes for TDM KVN parsing and encoding.
///
///
/// Payload conventions, settled across every variant: a keyword, whether this
/// crate named it or the message did, is `keyword: String`, so a caller
/// rendering an error never has to know which. A label this crate classifies
/// with rather than reads, such as `section` or `detail`, stays
/// `&'static str`. Each variant names where the problem is with the most
/// specific locator that means something in both directions: `line` for a
/// failure only the reader can raise, `line: Option<usize>` where the writer
/// raises the same failure with no input line to point at, and `segment` where
/// the problem belongs to a segment rather than to one line.
/// Marked `#[non_exhaustive]`: the CCSDS 503.0-B-2 audit adds failure modes as
/// it covers more of the standard, and the annotation keeps each addition
/// additive for callers, who match with a wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TdmError {
    /// The message contained no complete metadata/data segment.
    NoSegments,
    /// A section marker appeared in an invalid location.
    Section {
        /// One-based input line number.
        line: usize,
        /// The section validation detail.
        detail: &'static str,
    },
    /// A non-comment line was not a valid KVN assignment or section marker.
    MalformedLine {
        /// One-based input line number.
        line: usize,
        /// The offending input line.
        text: String,
    },
    /// A line held a character outside the printable ASCII set CCSDS 503.0-B-2
    /// 4.2.1 allows.
    NonPrintableCharacter {
        /// One-based input line number, or `None` for a line [`encode_kvn`]
        /// would write, which no input produced.
        line: Option<usize>,
        /// The line's first whitespace-delimited token, which is the keyword on
        /// every line the writer builds and the only locator it has.
        keyword: String,
        /// One-based character position within the line.
        column: usize,
        /// The offending character.
        character: char,
    },
    /// A line was longer than the 254 characters CCSDS 503.0-B-2 4.2.1 allows.
    LineTooLong {
        /// One-based input line number, or `None` for a line [`encode_kvn`]
        /// would write, which no input produced.
        line: Option<usize>,
        /// The line's first whitespace-delimited token, which is the keyword on
        /// every line the writer builds and the only locator it has.
        keyword: String,
        /// The line's length in characters, excluding its terminator.
        length: usize,
    },
    /// A tracking data record's timetag is not one of the two forms CCSDS
    /// 503.0-B-2 4.3.9 defines.
    MalformedEpoch {
        /// One-based input line number, or `None` for a record a caller built
        /// that no input produced.
        line: Option<usize>,
        /// The record's keyword.
        keyword: String,
        /// The offending timetag.
        text: String,
    },
    /// A keyword's records are not in the chronological order 3.4.10 requires.
    RecordsOutOfOrder {
        /// One-based segment index.
        segment: usize,
        /// The keyword whose records go backwards.
        keyword: String,
        /// The timetag that goes back.
        epoch: String,
    },
    /// A keyword and timetag pair repeats, which 3.4.11 forbids.
    DuplicateRecord {
        /// One-based segment index.
        segment: usize,
        /// The repeated keyword.
        keyword: String,
        /// The repeated timetag.
        epoch: String,
    },
    /// The last line carried none of the terminators 4.2.11 requires.
    UnterminatedFinalLine {
        /// One-based number of the unterminated line.
        line: usize,
    },
    /// A field or comment holds text the KVN form cannot carry, so writing it
    /// would give a line that reads back as something other than the value.
    Unwritable {
        /// The keyword, or `COMMENT` for a comment.
        keyword: String,
        /// What the KVN form cannot carry.
        reason: &'static str,
    },
    /// A keyword appeared before one the table for its section orders earlier.
    KeywordOutOfOrder {
        /// One-based input line number, or `None` for a line [`encode_kvn`]
        /// would write, which no input produced.
        line: Option<usize>,
        /// The keyword out of place.
        keyword: String,
        /// The section it appeared in, `header` or `metadata`.
        section: &'static str,
    },
    /// A `PATH` entry names a participant index the segment does not define.
    UndefinedParticipant {
        /// One-based segment index.
        segment: usize,
        /// The path keyword naming it.
        keyword: String,
        /// The participant index it names.
        index: u8,
    },
    /// A keyword appeared twice in one block carrying different values.
    ///
    /// 4.2.5 a) gives a header or metadata keyword "a single value
    /// assignment", and 3.2.3 and 3.3.1.8 give it one place in the order its
    /// table fixes, so it carries one value. CCSDS 503.0-B-2 3.3.1.9 similarly
    /// forbids two participants sharing an indexer (`PARTICIPANT_n`). Choosing
    /// between two that disagree is inventing one, which no policy forgives.
    ConflictingKeyword {
        /// One-based input line number of the repeat, or `None` for a
        /// caller-built value that no input produced.
        line: Option<usize>,
        /// The repeated keyword.
        keyword: String,
        /// The section it appeared in, `header` or `metadata`.
        section: &'static str,
        /// The value read first.
        first: String,
        /// The value that disagrees with it.
        second: String,
    },
    /// A keyword appeared twice in one block carrying the same value, which
    /// 4.2.5 a) gives one value assignment.
    ///
    /// Raised by [`encode_kvn`] only: a reader meeting the repeat takes the
    /// message whatever the policy says, since both lines say the same thing.
    RepeatedKeyword {
        /// One-based input line number, or `None` for a line [`encode_kvn`]
        /// would write, which no input produced.
        line: Option<usize>,
        /// The repeated keyword.
        keyword: String,
        /// The section it appeared in, `header` or `metadata`.
        section: &'static str,
    },
    /// A keyword is not one the table for its section defines: table 3-2 for a
    /// header, table 3-3 for a metadata section.
    UndefinedKeyword {
        /// One-based input line number.
        line: usize,
        /// The offending keyword.
        keyword: String,
        /// The section it appeared in, `header` or `metadata`.
        section: &'static str,
    },
    /// A keyword CCSDS 503.0-B-2 marks mandatory was absent.
    MissingKeyword {
        /// The absent keyword.
        keyword: String,
        /// The one-based segment that required it, or `None` for the header.
        segment: Option<usize>,
    },
    /// A data section held no tracking data record, which 3.1.3 requires.
    EmptyDataSection {
        /// The one-based segment index.
        segment: usize,
    },
    /// A keyword was given the empty value CCSDS 503.0-B-2 4.3.1 forbids.
    EmptyValue {
        /// One-based input line number, or `None` for a caller-built field that
        /// no input produced.
        line: Option<usize>,
        /// The keyword whose value was empty.
        keyword: String,
    },
    /// The `CCSDS_TDM_VERS` value was not the `x.y` form 3.2.5 requires.
    InvalidVersion {
        /// One-based input line number, or `None` for a caller-built value that
        /// no input produced.
        line: Option<usize>,
        /// The offending value.
        value: String,
    },
    /// A field key is a keyword CCSDS 503.0-B-2 4.2.5 c) excepts from the
    /// `keyword = value` syntax, so the field has no assignment form to write.
    ///
    /// Returned by [`encode_kvn`] for a value a caller built directly, which
    /// [`parse_kvn`] cannot produce because it refuses such a line on the way in.
    KeywordNotAssignable {
        /// The offending keyword, as the field holds it.
        keyword: String,
    },
    /// A data record did not contain `epoch value`.
    MalformedRecord {
        /// One-based input line number.
        line: usize,
        /// The offending data keyword.
        keyword: String,
    },
    /// A field failed numeric or indexed-keyword validation.
    InvalidField {
        /// The offending field name.
        keyword: String,
        /// The validation failure category.
        kind: TdmInputErrorKind,
    },
}

impl fmt::Display for TdmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSegments => write!(f, "missing TDM segment"),
            Self::Section { line, detail } => {
                write!(f, "invalid TDM section at line {line}: {detail}")
            }
            Self::MalformedLine { line, text } => {
                write!(f, "malformed TDM KVN line {line}: {text}")
            }
            Self::NonPrintableCharacter {
                line: Some(line),
                keyword: _,
                column,
                character,
            } => write!(
                f,
                "TDM line {line} column {column} holds {character:?}, which is not printable ASCII"
            ),
            Self::NonPrintableCharacter {
                line: None,
                keyword,
                column,
                character,
            } => write!(
                f,
                "the TDM {keyword} line holds {character:?} at column {column}, \
                 which is not printable ASCII"
            ),
            Self::LineTooLong {
                line: Some(line),
                keyword: _,
                length,
            } => write!(
                f,
                "TDM line {line} is {length} characters, over the {MAX_LINE_CHARACTERS} allowed"
            ),
            Self::LineTooLong {
                line: None,
                keyword,
                length,
            } => write!(
                f,
                "the TDM {keyword} line is {length} characters, \
                 over the {MAX_LINE_CHARACTERS} allowed"
            ),
            Self::MissingKeyword {
                keyword,
                segment: Some(segment),
            } => write!(f, "missing TDM {keyword} in segment {segment}"),
            Self::MissingKeyword {
                keyword,
                segment: None,
            } => write!(f, "missing TDM {keyword}"),
            Self::MalformedEpoch {
                line: Some(line),
                keyword,
                text,
            } => write!(
                f,
                "TDM record {keyword} at line {line} has the timetag {text}, which is not a form 4.3.9 defines"
            ),
            Self::MalformedEpoch {
                line: None,
                keyword,
                text,
            } => write!(
                f,
                "TDM record {keyword} has the timetag {text}, which is not a form 4.3.9 defines"
            ),
            Self::RecordsOutOfOrder {
                segment,
                keyword,
                epoch,
            } => write!(
                f,
                "TDM segment {segment} gives {keyword} at {epoch} after a later one"
            ),
            Self::DuplicateRecord {
                segment,
                keyword,
                epoch,
            } => write!(f, "TDM segment {segment} repeats {keyword} at {epoch}"),
            Self::KeywordOutOfOrder {
                line: Some(line),
                keyword,
                section,
            } => write!(
                f,
                "TDM {section} keyword {keyword} at line {line} is out of the order its table fixes"
            ),
            Self::KeywordOutOfOrder {
                line: None,
                keyword,
                section,
            } => write!(
                f,
                "TDM {section} writes {keyword} out of the order its table fixes"
            ),
            Self::RepeatedKeyword {
                line: Some(line),
                keyword,
                section,
            } => write!(
                f,
                "TDM {section} keyword {keyword} repeats at line {line} with the same value"
            ),
            Self::RepeatedKeyword {
                line: None,
                keyword,
                section,
            } => write!(f, "TDM {section} writes {keyword} twice with the same value"),
            Self::UnterminatedFinalLine { line } => {
                write!(f, "TDM line {line} carries no terminator")
            }
            Self::Unwritable { keyword, reason } => {
                write!(f, "TDM {keyword} cannot be written: {reason}")
            }
            Self::UndefinedParticipant {
                segment,
                keyword,
                index,
            } => write!(
                f,
                "TDM segment {segment} gives {keyword} naming participant {index}, which it does not define"
            ),
            Self::ConflictingKeyword {
                line: Some(line),
                keyword,
                section,
                first,
                second,
            } => write!(
                f,
                "TDM {section} keyword {keyword} at line {line} repeats with {second:?} after {first:?}"
            ),
            Self::ConflictingKeyword {
                line: None,
                keyword,
                section,
                first,
                second,
            } => write!(
                f,
                "TDM {section} keyword {keyword} carries both {first:?} and {second:?}"
            ),
            Self::UndefinedKeyword { line, keyword, section } => write!(
                f,
                "TDM {section} keyword {keyword} at line {line} is not one the standard defines"
            ),
            Self::EmptyDataSection { segment } => {
                write!(f, "TDM segment {segment} holds no tracking data record")
            }
            Self::EmptyValue {
                line: Some(line),
                keyword,
            } => write!(f, "TDM keyword {keyword} has no value at line {line}"),
            Self::EmptyValue { line: None, keyword } => {
                write!(f, "TDM keyword {keyword} has no value")
            }
            Self::InvalidVersion {
                line: Some(line),
                value,
            } => write!(f, "TDM version {value} at line {line} is not in the form x.y"),
            Self::InvalidVersion { line: None, value } => {
                write!(f, "TDM version {value} is not in the form x.y")
            }
            Self::KeywordNotAssignable { keyword } => {
                write!(f, "TDM keyword {keyword} cannot be given a value")
            }
            Self::MalformedRecord { line, keyword } => {
                write!(f, "malformed TDM data record {keyword} at line {line}")
            }
            Self::InvalidField { keyword, kind } => write!(f, "invalid TDM field {keyword}: {kind}"),
        }
    }
}

impl std::error::Error for TdmError {}

#[derive(Default)]
struct HeaderBuilder {
    /// The highest rank any keyword in this section has reached, for 3.2.3.
    highest_rank: usize,
    version: Option<String>,
    /// One-based line the `CCSDS_TDM_VERS` record was read from.
    version_line: Option<usize>,
    comments: Vec<String>,
    /// Every header assignment in the order it was read, which is what a
    /// repeated keyword is judged against; the modeled fields below are built
    /// from it.
    assignments: Vec<TdmField>,
    creation_date: Option<String>,
    originator: Option<String>,
    message_id: Option<String>,
    fields: Vec<TdmField>,
}

#[derive(Default)]
struct MetadataBuilder {
    /// The highest rank any keyword in this block has reached, for 3.3.1.8.
    highest_rank: usize,
    comments: Vec<String>,
    fields: Vec<TdmField>,
}

#[derive(Default)]
struct DataBuilder {
    comments: Vec<TdmComment>,
    records: Vec<TdmDataRecord>,
    /// Each record's timetag, read once when the record is read.
    epochs: Vec<EpochKey>,
}

/// Parse a TDM in CCSDS KVN format.
///
/// The parser accepts flexible whitespace around `=` and between the epoch and
/// value tokens. It requires complete `META_START` / `META_STOP` and
/// `DATA_START` / `DATA_STOP` blocks, and every data record with a numeric
/// keyword must contain a finite value. Frequency records are not converted to
/// range rate and keep their original decimal token for later serialization.
///
/// A line whose keyword is `COMMENT` is a comment, never an assignment: CCSDS
/// 503.0-B-2 4.2.5 c) excepts `COMMENT` from the KVN syntax, and 4.5.3 requires
/// at least one space after the keyword. `COMMENT=value` satisfies neither form
/// and is refused as a malformed line.
pub fn parse_kvn(text: &str) -> Result<Tdm, TdmError> {
    parse_kvn_with_policy(text, TdmPolicy::default()).map(|(tdm, _)| tdm)
}

/// Parse a TDM in CCSDS KVN format under `policy`, with the departures from the
/// standard it forgave.
///
/// [`TdmPolicy::default`] forgives nothing and this returns no warnings, which
/// is what [`parse_kvn`] does. A field set to [`TdmLeniency::Forgive`] reads a
/// message a strict read refuses and reports each departure as a
/// [`TdmWarning`]. What is never forgiven, and the writer staying strict
/// whatever was forgiven, are described on [`TdmPolicy`].
pub fn parse_kvn_with_policy(
    text: &str,
    policy: TdmPolicy,
) -> Result<(Tdm, Vec<TdmWarning>), TdmError> {
    let mut header = HeaderBuilder::default();
    let mut metadata: Option<MetadataBuilder> = None;
    let mut pending_metadata: Option<TdmMetadata> = None;
    let mut data: Option<DataBuilder> = None;
    let mut segments = Vec::new();
    let mut warnings = Vec::new();

    let source_lines = tdm_lines(text);
    let past_end = source_lines.len().saturating_add(1);

    for (idx, raw_line) in source_lines.iter().copied().enumerate() {
        let line_no = idx + 1;
        check_line(line_no, raw_line, policy, &mut warnings)?;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(comment) = comment_text(line) {
            if let Some(builder) = data.as_mut() {
                // 4.5.2 c) puts a data-section comment "between the
                // 'DATA_START' keyword and the first Tracking Data Record".
                if !builder.records.is_empty() {
                    match policy.keyword_order {
                        TdmLeniency::Strict => {
                            return Err(TdmError::KeywordOutOfOrder {
                                line: Some(line_no),
                                keyword: COMMENT_KEY.to_string(),
                                section: "data",
                            })
                        }
                        TdmLeniency::Forgive => {
                            warnings.push(TdmWarning::KeywordOutOfOrder {
                                line: line_no,
                                keyword: COMMENT_KEY.to_string(),
                                section: "data",
                            });
                        }
                    }
                }
                let before_record = builder.records.len();
                builder.comments.push(TdmComment {
                    text: comment,
                    before_record,
                });
            } else if let Some(builder) = metadata.as_mut() {
                check_keyword_order(
                    line_no,
                    COMMENT_KEY,
                    "metadata",
                    &METADATA_ORDER,
                    &mut builder.highest_rank,
                    policy,
                    &mut warnings,
                )?;
                builder.comments.push(comment);
            } else if pending_metadata.is_none() && segments.is_empty() {
                check_keyword_order(
                    line_no,
                    COMMENT_KEY,
                    "header",
                    &HEADER_ORDER,
                    &mut header.highest_rank,
                    policy,
                    &mut warnings,
                )?;
                header.comments.push(comment);
            } else if pending_metadata.is_none() {
                // 3.1.2 puts the header once, before the first segment, so a
                // comment after DATA_STOP belongs to no block the standard
                // defines. It used to join the header it is not part of.
                return Err(TdmError::Section {
                    line: line_no,
                    detail: "comment after a segment",
                });
            } else {
                return Err(TdmError::Section {
                    line: line_no,
                    detail: "comment between metadata and data",
                });
            }
            continue;
        }

        match line {
            "META_START" => {
                if metadata.is_some() || data.is_some() || pending_metadata.is_some() {
                    return Err(TdmError::Section {
                        line: line_no,
                        detail: "nested metadata block",
                    });
                }
                metadata = Some(MetadataBuilder::default());
                continue;
            }
            "META_STOP" => {
                let builder = metadata.take().ok_or(TdmError::Section {
                    line: line_no,
                    detail: "metadata stop without metadata start",
                })?;
                pending_metadata = Some(build_metadata(
                    builder,
                    segments.len().saturating_add(1),
                    policy,
                    &mut warnings,
                )?);
                continue;
            }
            "DATA_START" => {
                if metadata.is_some() || data.is_some() || pending_metadata.is_none() {
                    return Err(TdmError::Section {
                        line: line_no,
                        detail: "data start without completed metadata",
                    });
                }
                data = Some(DataBuilder::default());
                continue;
            }
            "DATA_STOP" => {
                let builder = data.take().ok_or(TdmError::Section {
                    line: line_no,
                    detail: "data stop without data start",
                })?;
                let metadata = pending_metadata.take().ok_or(TdmError::Section {
                    line: line_no,
                    detail: "data stop without metadata",
                })?;
                // 3.1.3: a segment's data section holds "a minimum of one
                // Tracking Data Record".
                if builder.records.is_empty() {
                    let segment = segments.len().saturating_add(1);
                    match policy.empty_data_sections {
                        TdmLeniency::Strict => return Err(TdmError::EmptyDataSection { segment }),
                        TdmLeniency::Forgive => {
                            warnings.push(TdmWarning::EmptyDataSection { segment });
                        }
                    }
                }
                check_record_order(
                    &builder,
                    segments.len().saturating_add(1),
                    policy,
                    &mut warnings,
                )?;
                segments.push(TdmSegment {
                    metadata,
                    data: TdmDataSection {
                        comments: builder.comments,
                        records: builder.records,
                    },
                });
                continue;
            }
            _ => {}
        }

        let (key, value) = parse_assignment(line).ok_or_else(|| TdmError::MalformedLine {
            line: line_no,
            text: line.to_string(),
        })?;

        // `COMMENT` is excepted from the KVN syntax (4.2.5 c)) and a comment
        // line needs a space after the keyword (4.5.3), so `COMMENT=value` is
        // neither a comment nor an assignment. Keeping it as a field produced a
        // value whose encoding, `COMMENT = value`, reparsed as a comment.
        if keyword_takes_no_value(&key) {
            return Err(TdmError::MalformedLine {
                line: line_no,
                text: line.to_string(),
            });
        }

        // 4.3.1: "A non-empty value field must be specified for each keyword
        // provided." An empty value was read as an empty string, which the
        // modeled header fields then dropped on write and the rest carried.
        if value.is_empty() {
            return Err(TdmError::EmptyValue {
                line: Some(line_no),
                keyword: key,
            });
        }

        if let Some(builder) = data.as_mut() {
            let range_units = pending_metadata
                .as_ref()
                .map(|metadata| metadata.range_units.clone())
                .unwrap_or(TdmUnit::Kilometers);
            let record = parse_record(line_no, &key, &value, &range_units)?;
            let epoch = parse_epoch_key(&record.epoch).ok_or_else(|| TdmError::MalformedEpoch {
                line: Some(line_no),
                keyword: record.keyword.clone(),
                text: record.epoch.clone(),
            })?;
            builder.epochs.push(epoch);
            builder.records.push(record);
        } else if let Some(builder) = metadata.as_mut() {
            // 3.3.1.7: "Only those keywords shown in table 3-3 shall be used in
            // a TDM Metadata Section."
            if !known_metadata_keyword(&key)? {
                return Err(TdmError::UndefinedKeyword {
                    line: line_no,
                    keyword: key,
                    section: "metadata",
                });
            }
            check_keyword_order(
                line_no,
                &key,
                "metadata",
                &METADATA_ORDER,
                &mut builder.highest_rank,
                policy,
                &mut warnings,
            )?;
            check_repeated_field(
                line_no,
                "metadata",
                &key,
                &value,
                &builder.fields,
                &mut warnings,
            )?;
            builder.fields.push(TdmField { key, value });
        } else if pending_metadata.is_none() && segments.is_empty() {
            // 3.2.3: "Only those keywords shown in table 3-2 shall be used in a
            // TDM Header."
            if !known_header_keyword(&key) {
                return Err(TdmError::UndefinedKeyword {
                    line: line_no,
                    keyword: key,
                    section: "header",
                });
            }
            check_keyword_order(
                line_no,
                &key,
                "header",
                &HEADER_ORDER,
                &mut header.highest_rank,
                policy,
                &mut warnings,
            )?;
            check_repeated_field(
                line_no,
                "header",
                &key,
                &value,
                &header.assignments,
                &mut warnings,
            )?;
            header.assignments.push(TdmField {
                key: key.clone(),
                value: value.clone(),
            });
            parse_header_field(&mut header, line_no, key, value);
        } else if pending_metadata.is_none() {
            // 3.1.2 puts the header once, before the first segment. An
            // assignment after DATA_STOP used to reopen it, so a keyword read
            // after a segment joined the header it is not part of.
            return Err(TdmError::Section {
                line: line_no,
                detail: "header field after a segment",
            });
        } else {
            return Err(TdmError::Section {
                line: line_no,
                detail: "field between metadata and data",
            });
        }
    }

    if metadata.is_some() {
        return Err(TdmError::Section {
            line: past_end,
            detail: "unclosed metadata block",
        });
    }
    if data.is_some() {
        return Err(TdmError::Section {
            line: past_end,
            detail: "unclosed data block",
        });
    }
    if pending_metadata.is_some() {
        return Err(TdmError::Section {
            line: past_end,
            detail: "metadata without data block",
        });
    }

    // An empty value is refused above, so absence is the only way this is None.
    let version = header.version.ok_or(TdmError::MissingKeyword {
        keyword: VERSION_KEY.to_string(),
        segment: None,
    })?;
    check_version(header.version_line, &version)?;
    // Table 3-2 marks CREATION_DATE and ORIGINATOR mandatory. An empty value is
    // already refused above, so absence is the only way either is None here.
    require_keyword(
        header.creation_date.is_some(),
        "CREATION_DATE",
        None,
        policy,
        &mut warnings,
    )?;
    require_keyword(
        header.originator.is_some(),
        "ORIGINATOR",
        None,
        policy,
        &mut warnings,
    )?;
    if segments.is_empty() {
        return Err(TdmError::NoSegments);
    }

    // 4.2.11 terminates every TDM line, the last one included. Thirteen of the
    // 53 public files gathered for this audit end without one, which is why it
    // is forgivable: a file's last record reads the same either way.
    //
    // It is the last check in the function, behind everything about what the
    // message says: the unclosed-block checks, the mandatory keywords, the
    // version form and the segment count. Someone handed "missing TDM
    // CCSDS_TDM_VERS" can fix the file; handed "line 1 carries no terminator"
    // for a file that is also missing its version, they get the smaller of the
    // two problems first.
    //
    // check_line keeps its place at the top of the read loop, ahead of all of
    // this. A character outside printable ASCII or a line past 254 characters
    // names a position in one line and explains how the rest of that line
    // reads: a keyword holding a non-breaking space reads as an undefined
    // keyword, and reporting the undefined keyword names the wrong problem.
    // Nothing downstream turns on whether the file's last line was terminated.
    if !text.is_empty() && !text.ends_with(['\r', '\n']) {
        match policy.final_terminator {
            TdmLeniency::Strict => {
                return Err(TdmError::UnterminatedFinalLine {
                    line: source_lines.len(),
                })
            }
            TdmLeniency::Forgive => warnings.push(TdmWarning::UnterminatedFinalLine {
                line: source_lines.len(),
            }),
        }
    }

    Ok((
        Tdm {
            version,
            comments: header.comments,
            creation_date: header.creation_date,
            originator: header.originator,
            message_id: header.message_id,
            header_fields: header.fields,
            segments,
        },
        warnings,
    ))
}

/// Encode a TDM to canonical CCSDS KVN text.
///
/// The output uses `KEY = VALUE` assignments and emits each data record as
/// `KEY = epoch decimal-token`. Record decimals are not reformatted. Encoding
/// validates that every stored decimal token parses back to the stored `f64`
/// bits, which keeps `RECEIVE_FREQ` and `TRANSMIT_FREQ_n` values lossless.
///
/// A header or metadata field keyed `COMMENT` is refused rather than written:
/// `COMMENT = value` reparses as a comment, so the encoding would not read back
/// as the value it was given.
pub fn encode_kvn(tdm: &Tdm) -> Result<String, TdmError> {
    encode_kvn_with_policy(tdm, TdmWritePolicy::default()).map(|(text, _)| text)
}

/// Encode a TDM under `policy`, with the departures from CCSDS 503.0-B-2 it
/// emitted.
///
/// [`TdmWritePolicy::default`] emits none, which is what [`encode_kvn`] does: a
/// value that cannot be written conformingly is refused instead. A field set to
/// [`TdmLeniency::Forgive`] lets the writer emit the matching departure and
/// names it in the returned list, so a caller asking for a non-conforming file
/// is told exactly what makes it one. The set mirrors what [`TdmPolicy`]
/// forgives on the way in, so a message read leniently can be written back by
/// asking for the same departures.
pub fn encode_kvn_with_policy(
    tdm: &Tdm,
    policy: TdmWritePolicy,
) -> Result<(String, Vec<TdmDeparture>), TdmError> {
    let mut departures = Vec::new();
    validate_tdm(tdm, policy, &mut departures)?;

    let mut lines = Vec::new();
    lines.push(format!("{VERSION_KEY} = {}", tdm.version));
    lines.extend(tdm.comments.iter().map(comment_line));
    if let Some(creation_date) = &tdm.creation_date {
        lines.push(format!("CREATION_DATE = {creation_date}"));
    }
    if let Some(originator) = &tdm.originator {
        lines.push(format!("ORIGINATOR = {originator}"));
    }
    if let Some(message_id) = &tdm.message_id {
        lines.push(format!("MESSAGE_ID = {message_id}"));
    }
    lines.extend(tdm.header_fields.iter().map(field_line));

    for segment in &tdm.segments {
        lines.push("META_START".to_string());
        lines.extend(segment.metadata.comments.iter().map(comment_line));
        lines.extend(segment.metadata.fields.iter().map(field_line));
        lines.push("META_STOP".to_string());
        lines.push("DATA_START".to_string());
        // Each comment goes back where it was read, so a message that carried
        // one away from the start of its block writes back unchanged rather
        // than with its comments gathered to the top.
        for (index, record) in segment.data.records.iter().enumerate() {
            for comment in &segment.data.comments {
                if comment.before_record == index {
                    lines.push(comment_line(&comment.text));
                }
            }
            lines.push(format!(
                "{} = {} {}",
                record.keyword, record.epoch, record.value.text
            ));
        }
        for comment in &segment.data.comments {
            if comment.before_record >= segment.data.records.len() {
                lines.push(comment_line(&comment.text));
            }
        }
        lines.push("DATA_STOP".to_string());
    }

    // 4.2.1 holds every line the writer built to printable ASCII and 254
    // characters, the two rules check_line holds the reader to. A value read
    // under a lenient policy carries both departures back out here, so the
    // check runs over what was built rather than over what the value came from.
    for line in &lines {
        check_written_line(line, policy, &mut departures)?;
    }

    // 4.2.11 terminates every line, so the last one carries one too unless the
    // policy asks for the departure a lenient reader forgives.
    match policy.final_terminator {
        TdmLeniency::Strict => lines.push(String::new()),
        TdmLeniency::Forgive => departures.push(TdmDeparture::UnterminatedFinalLine),
    }
    Ok((lines.join("\n"), departures))
}

fn parse_header_field(header: &mut HeaderBuilder, line: usize, key: String, value: String) {
    match key.as_str() {
        VERSION_KEY => {
            header.version = Some(value);
            header.version_line = Some(line);
        }
        "CREATION_DATE" => header.creation_date = empty_to_none(value),
        "ORIGINATOR" => header.originator = empty_to_none(value),
        "MESSAGE_ID" => header.message_id = empty_to_none(value),
        _ => header.fields.push(TdmField { key, value }),
    }
}

fn build_metadata(
    builder: MetadataBuilder,
    segment: usize,
    policy: TdmPolicy,
    warnings: &mut Vec<TdmWarning>,
) -> Result<TdmMetadata, TdmError> {
    let mut participants = Vec::new();
    let mut mode = None;
    let mut paths = Vec::new();
    let mut timetag_ref = None;
    let mut time_system = None;
    let mut range_units = TdmUnit::Kilometers;

    for field in &builder.fields {
        // Table 3-3 indexes PARTICIPANT_n with n = {1,2,3,4,5}. An index past
        // five is refused under every policy, not for the cap itself but
        // because 3.3.1.11 allows more only by arrangement outside the message:
        // a PATH entry naming an index the message cannot resolve would leave
        // the reader to guess which participant a measurement belongs to, which
        // changes what the file means.
        if let Some(index) = indexed_suffix_in_range(&field.key, "PARTICIPANT", 1, 5)? {
            // An identical repeat is reported as RepeatedKeyword; avoid
            // producing a phantom second participant for the same index.
            if !participants
                .iter()
                .any(|existing: &TdmParticipant| existing.index == index)
            {
                participants.push(TdmParticipant {
                    index,
                    name: field.value.clone(),
                });
            }
        } else if field.key == "MODE" {
            mode = empty_to_none(field.value.clone());
        } else if field.key == "PATH" || field.key.starts_with("PATH_") {
            paths.push(parse_path(field)?);
        } else if field.key == "TIMETAG_REF" {
            timetag_ref = empty_to_none(field.value.clone());
        } else if field.key == "TIME_SYSTEM" {
            time_system = empty_to_none(field.value.clone());
        } else if field.key == "RANGE_UNITS" && !field.value.is_empty() {
            range_units = range_unit_from_label(&field.value)?;
        }
    }

    // Table 3-3 marks TIME_SYSTEM mandatory and PARTICIPANT_n mandatory with
    // "at least one"; 3.3.1.7 requires every mandatory item in every metadata
    // section.
    require_keyword(
        time_system.is_some(),
        "TIME_SYSTEM",
        Some(segment),
        policy,
        warnings,
    )?;
    require_keyword(
        !participants.is_empty(),
        "PARTICIPANT_n",
        Some(segment),
        policy,
        warnings,
    )?;

    // 3.3.1.9 requires participant indices to differ, not to run consecutively,
    // so PARTICIPANT_1 beside PARTICIPANT_3 with no _2 is legal and nothing
    // points into the gap. A PATH naming an index the segment does not define
    // is a different matter: the measurements it describes would belong to
    // nobody, and choosing which participant was meant is inventing one. It is
    // refused under every policy for that reason.
    for path in &paths {
        for index in &path.participants {
            if !participants.iter().any(|held| held.index == *index) {
                return Err(TdmError::UndefinedParticipant {
                    segment,
                    keyword: path.key.clone(),
                    index: *index,
                });
            }
        }
    }

    Ok(TdmMetadata {
        comments: builder.comments,
        fields: builder.fields,
        participants,
        mode,
        paths,
        timetag_ref,
        time_system,
        range_units,
    })
}

fn parse_path(field: &TdmField) -> Result<TdmPath, TdmError> {
    let index = if field.key == "PATH" {
        None
    } else {
        Some(
            indexed_suffix_in_range(&field.key, "PATH", 1, 2)?
                .ok_or_else(|| invalid_index(&field.key))?,
        )
    };
    let mut participants = Vec::new();
    for token in field.value.split(',') {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return Err(invalid_index(&field.key));
        }
        let value = trimmed
            .parse::<u8>()
            .map_err(|_| invalid_index(&field.key))?;
        participants.push(value);
    }
    if participants.is_empty() {
        return Err(invalid_index(&field.key));
    }
    Ok(TdmPath {
        key: field.key.clone(),
        index,
        participants,
    })
}

fn parse_record(
    line: usize,
    keyword: &str,
    value: &str,
    range_units: &TdmUnit,
) -> Result<TdmDataRecord, TdmError> {
    if has_displayed_unit(keyword) || has_displayed_unit(value) {
        return Err(TdmError::InvalidField {
            keyword: keyword.to_string(),
            kind: TdmInputErrorKind::UnexpectedUnit,
        });
    }

    let mut parts = value.split_whitespace();
    let epoch = parts
        .next()
        .ok_or_else(|| malformed_record(line, keyword))?;
    let value_text = parts
        .next()
        .ok_or_else(|| malformed_record(line, keyword))?;
    if parts.next().is_some() {
        return Err(malformed_record(line, keyword));
    }

    let observable = observable_from_keyword(keyword)?;
    let scalar = parse_scalar(keyword, value_text, &observable)?;
    validate_record_value(keyword, &observable, &scalar)?;
    let unit = unit_for_keyword(keyword, &observable, range_units);

    Ok(TdmDataRecord {
        observable,
        keyword: keyword.to_string(),
        epoch: epoch.to_string(),
        value: scalar,
        unit,
    })
}

fn parse_scalar(
    field: &str,
    text: &str,
    observable: &TdmObservable,
) -> Result<TdmScalar, TdmError> {
    if is_nonfinite_float_token(text) {
        return Err(TdmError::InvalidField {
            keyword: field.to_string(),
            kind: TdmInputErrorKind::NonFinite,
        });
    }
    validate_numeric_token(field, text, observable)?;
    let value = text.parse::<f64>().map_err(|_| TdmError::InvalidField {
        keyword: field.to_string(),
        kind: TdmInputErrorKind::FloatParse,
    })?;
    if !value.is_finite() {
        return Err(TdmError::InvalidField {
            keyword: field.to_string(),
            kind: TdmInputErrorKind::NonFinite,
        });
    }
    let lexical_zero = numeric_token_is_zero(text);
    if !lexical_zero && decimal_magnitude_below_minimum_positive_double(text) {
        return Err(TdmError::InvalidField {
            keyword: field.to_string(),
            kind: TdmInputErrorKind::OutOfRange,
        });
    }
    if value == 0.0 && text.trim_start().starts_with('-') {
        return Err(TdmError::InvalidField {
            keyword: field.to_string(),
            kind: TdmInputErrorKind::NegativeZero,
        });
    }
    if value == 0.0 && !lexical_zero {
        return Err(TdmError::InvalidField {
            keyword: field.to_string(),
            kind: TdmInputErrorKind::OutOfRange,
        });
    }
    Ok(TdmScalar {
        text: text.to_string(),
        value,
    })
}

fn is_nonfinite_float_token(text: &str) -> bool {
    matches!(
        text,
        "NaN" | "+NaN" | "-NaN" | "Inf" | "+Inf" | "-Inf" | "Infinity" | "+Infinity" | "-Infinity"
    )
}

fn validate_numeric_token(
    field: &str,
    text: &str,
    observable: &TdmObservable,
) -> Result<(), TdmError> {
    if matches!(observable, TdmObservable::Other(name) if name == "DOPPLER_COUNT") {
        validate_integer_token(field, text)
    } else if phase_count_keyword(field) {
        validate_phase_count_token(field, text)
    } else if is_ccsds_double_token(text) {
        Ok(())
    } else {
        Err(TdmError::InvalidField {
            keyword: field.to_string(),
            kind: TdmInputErrorKind::FloatParse,
        })
    }
}

fn validate_integer_token(field: &str, text: &str) -> Result<(), TdmError> {
    let digits = strip_ascii_sign(text);
    if digits.is_empty() {
        return Err(TdmError::InvalidField {
            keyword: field.to_string(),
            kind: TdmInputErrorKind::NonInteger,
        });
    }
    if !digits.chars().all(|character| character.is_ascii_digit()) {
        return Err(TdmError::InvalidField {
            keyword: field.to_string(),
            kind: TdmInputErrorKind::NonInteger,
        });
    }
    let value = text.parse::<i64>().map_err(|_| TdmError::InvalidField {
        keyword: field.to_string(),
        kind: TdmInputErrorKind::OutOfRange,
    })?;
    if !(i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&value) {
        return Err(TdmError::InvalidField {
            keyword: field.to_string(),
            kind: TdmInputErrorKind::OutOfRange,
        });
    }
    Ok(())
}

fn validate_phase_count_token(field: &str, text: &str) -> Result<(), TdmError> {
    if is_phase_count_token(text) {
        Ok(())
    } else {
        Err(TdmError::InvalidField {
            keyword: field.to_string(),
            kind: TdmInputErrorKind::FloatParse,
        })
    }
}

fn is_ccsds_double_token(text: &str) -> bool {
    let Some(unsigned) = strip_optional_sign(text) else {
        return false;
    };
    is_fixed_point_token(unsigned, Some(16)) || is_floating_point_token(unsigned, Some(16))
}

fn is_phase_count_token(text: &str) -> bool {
    is_unsigned_integer(text) || is_fixed_point_token(text, None)
}

fn strip_optional_sign(text: &str) -> Option<&str> {
    let unsigned = strip_ascii_sign(text);
    (!unsigned.is_empty()).then_some(unsigned)
}

fn strip_ascii_sign(text: &str) -> &str {
    match text.as_bytes().first() {
        Some(b'+') | Some(b'-') => &text[1..],
        _ => text,
    }
}

fn is_unsigned_integer(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|character| character.is_ascii_digit())
}

fn is_fixed_point_token(text: &str, max_digits: Option<usize>) -> bool {
    let Some((integer, fraction)) = text.split_once('.') else {
        return false;
    };
    if integer.is_empty()
        || fraction.is_empty()
        || fraction.contains('.')
        || !is_unsigned_integer(integer)
        || !is_unsigned_integer(fraction)
    {
        return false;
    }
    match max_digits {
        Some(max) => integer.len() + fraction.len() <= max,
        None => true,
    }
}

fn is_floating_point_token(text: &str, max_digits: Option<usize>) -> bool {
    let Some(exponent_index) = text.find(['E', 'e']) else {
        return false;
    };
    let mantissa = &text[..exponent_index];
    let exponent = &text[exponent_index + 1..];
    if exponent.is_empty() || exponent.find(['E', 'e']).is_some() {
        return false;
    }
    let exponent_digits = strip_ascii_sign(exponent);
    if exponent_digits.is_empty()
        || !exponent_digits
            .chars()
            .all(|character| character.is_ascii_digit())
    {
        return false;
    }
    let mut mantissa_chars = mantissa.chars();
    let Some(integer) = mantissa_chars.next() else {
        return false;
    };
    if !integer.is_ascii_digit() || mantissa_chars.next() != Some('.') {
        return false;
    }
    let fraction = mantissa_chars.as_str();
    if fraction.is_empty() || !is_unsigned_integer(fraction) {
        return false;
    }
    match max_digits {
        Some(max) => fraction.len() < max,
        None => true,
    }
}

fn numeric_token_is_zero(text: &str) -> bool {
    let unsigned = strip_ascii_sign(text);
    let mantissa = unsigned
        .find(['E', 'e'])
        .map_or(unsigned, |exponent_index| &unsigned[..exponent_index]);
    !mantissa.is_empty()
        && mantissa
            .bytes()
            .filter(|byte| *byte != b'.')
            .all(|byte| byte == b'0')
}

fn decimal_magnitude_below_minimum_positive_double(text: &str) -> bool {
    const MIN_POSITIVE_EXPONENT: i32 = -324;
    const MIN_POSITIVE_SIGNIFICAND_16: &[u8; 16] = b"4940000000000000";

    let Some((exponent, significand)) = normalized_decimal_parts(text) else {
        return false;
    };
    if exponent < MIN_POSITIVE_EXPONENT {
        return true;
    }
    if exponent > MIN_POSITIVE_EXPONENT {
        return false;
    }
    let significand = significand.as_bytes();
    for (index, minimum) in MIN_POSITIVE_SIGNIFICAND_16.iter().enumerate() {
        let digit = significand.get(index).copied().unwrap_or(b'0');
        if digit != *minimum {
            return digit < *minimum;
        }
    }
    false
}

fn normalized_decimal_parts(text: &str) -> Option<(i32, String)> {
    let unsigned = strip_ascii_sign(text);
    let (mantissa, exponent_adjust) = if let Some(exponent_index) = unsigned.find(['E', 'e']) {
        (
            &unsigned[..exponent_index],
            parse_exponent_for_bound(&unsigned[exponent_index + 1..]),
        )
    } else {
        (unsigned, 0)
    };
    let (integer, fraction) = mantissa.split_once('.')?;
    let decimal_index = i32::try_from(integer.len()).ok()?;
    let mut digits = String::with_capacity(integer.len() + fraction.len());
    digits.push_str(integer);
    digits.push_str(fraction);
    let leading = digits.bytes().position(|byte| byte != b'0')?;
    let leading = i32::try_from(leading).ok()?;
    let exponent = exponent_adjust + decimal_index - leading - 1;
    Some((exponent, digits[leading as usize..].to_string()))
}

fn parse_exponent_for_bound(text: &str) -> i32 {
    let negative = text.starts_with('-');
    let digits = strip_ascii_sign(text);
    let digits = digits.trim_start_matches('0');
    if digits.len() > 4 {
        return if negative { -10_000 } else { 10_000 };
    }
    let value = digits.parse::<i32>().unwrap_or(0);
    if negative {
        -value
    } else {
        value
    }
}

fn observable_from_keyword(keyword: &str) -> Result<TdmObservable, TdmError> {
    match keyword {
        "RANGE" => Ok(TdmObservable::Range),
        "DOPPLER_INSTANTANEOUS" => Ok(TdmObservable::DopplerInstantaneous),
        "DOPPLER_INTEGRATED" => Ok(TdmObservable::DopplerIntegrated),
        "ANGLE_1" => Ok(TdmObservable::Angle1),
        "ANGLE_2" => Ok(TdmObservable::Angle2),
        "RECEIVE_FREQ" => Ok(TdmObservable::ReceiveFreq { participant: None }),
        _ => {
            if let Some(participant) = indexed_suffix_in_range(keyword, "RECEIVE_FREQ", 1, 5)? {
                Ok(TdmObservable::ReceiveFreq {
                    participant: Some(participant),
                })
            } else if let Some(participant) =
                indexed_suffix_in_range(keyword, "TRANSMIT_FREQ_RATE", 1, 5)?
            {
                Ok(TdmObservable::TransmitFreqRate {
                    participant: Some(participant),
                })
            } else if let Some(participant) =
                indexed_suffix_in_range(keyword, "TRANSMIT_FREQ", 1, 5)?
            {
                Ok(TdmObservable::TransmitFreq {
                    participant: Some(participant),
                })
            } else if known_table_3_5_other_keyword(keyword)? {
                Ok(TdmObservable::Other(keyword.to_string()))
            } else {
                Err(unknown_keyword(keyword))
            }
        }
    }
}

fn validate_record_value(
    keyword: &str,
    observable: &TdmObservable,
    scalar: &TdmScalar,
) -> Result<(), TdmError> {
    let value = scalar.value;
    if value == 0.0 && scalar.text.trim_start().starts_with('-') {
        return Err(TdmError::InvalidField {
            keyword: keyword.to_string(),
            kind: TdmInputErrorKind::NegativeZero,
        });
    }
    if matches!(observable, TdmObservable::TransmitFreq { .. }) && value <= 0.0 {
        return Err(TdmError::InvalidField {
            keyword: keyword.to_string(),
            kind: TdmInputErrorKind::NotPositive,
        });
    }
    if matches!(observable, TdmObservable::Other(name) if name == "DOPPLER_COUNT") {
        validate_doppler_count(keyword, scalar)?;
    }
    if matches!(observable, TdmObservable::Other(name) if name == "RCS" || name == "STEC" || name == "TEMPERATURE")
        && value <= 0.0
    {
        return Err(TdmError::InvalidField {
            keyword: keyword.to_string(),
            kind: TdmInputErrorKind::NotPositive,
        });
    }
    if matches!(observable, TdmObservable::Other(name) if name == "TROPO_DRY" || name == "TROPO_WET")
        && value < 0.0
    {
        return Err(TdmError::InvalidField {
            keyword: keyword.to_string(),
            kind: TdmInputErrorKind::Negative,
        });
    }
    if matches!(observable, TdmObservable::Other(name) if name == "RHUMIDITY")
        && !(0.0..=100.0).contains(&value)
    {
        return Err(TdmError::InvalidField {
            keyword: keyword.to_string(),
            kind: TdmInputErrorKind::OutOfRange,
        });
    }
    if matches!(observable, TdmObservable::Angle1 | TdmObservable::Angle2)
        && !(-180.0..360.0).contains(&value)
    {
        return Err(TdmError::InvalidField {
            keyword: keyword.to_string(),
            kind: TdmInputErrorKind::OutOfRange,
        });
    }
    Ok(())
}

fn validate_doppler_count(keyword: &str, scalar: &TdmScalar) -> Result<(), TdmError> {
    let text = scalar.text.trim_start();
    if text.starts_with('-') {
        return Err(TdmError::InvalidField {
            keyword: keyword.to_string(),
            kind: TdmInputErrorKind::Negative,
        });
    }
    let digits = text.strip_prefix('+').unwrap_or(text);
    if digits.is_empty() || !digits.chars().all(|character| character.is_ascii_digit()) {
        return Err(TdmError::InvalidField {
            keyword: keyword.to_string(),
            kind: TdmInputErrorKind::NonInteger,
        });
    }
    let count = digits.parse::<u64>().map_err(|_| TdmError::InvalidField {
        keyword: keyword.to_string(),
        kind: TdmInputErrorKind::OutOfRange,
    })?;
    if count > i32::MAX as u64 {
        return Err(TdmError::InvalidField {
            keyword: keyword.to_string(),
            kind: TdmInputErrorKind::OutOfRange,
        });
    }
    Ok(())
}

fn validate_tdm(
    tdm: &Tdm,
    policy: TdmWritePolicy,
    departures: &mut Vec<TdmDeparture>,
) -> Result<(), TdmError> {
    if tdm.version.is_empty() {
        return Err(TdmError::MissingKeyword {
            keyword: VERSION_KEY.to_string(),
            segment: None,
        });
    }
    check_version(None, &tdm.version)?;
    for keyword in ["CREATION_DATE", "ORIGINATOR"] {
        let present = match keyword {
            "CREATION_DATE" => tdm.creation_date.is_some(),
            _ => tdm.originator.is_some(),
        };
        if !present {
            emit_or_refuse(
                policy.missing_keywords,
                TdmError::MissingKeyword {
                    keyword: keyword.to_string(),
                    segment: None,
                },
                TdmDeparture::MissingKeyword {
                    keyword: keyword.to_string(),
                    segment: None,
                },
                departures,
            )?;
        }
    }
    if tdm.segments.is_empty() {
        return Err(TdmError::NoSegments);
    }
    for comment in &tdm.comments {
        check_comment(comment)?;
    }
    for field in &tdm.header_fields {
        check_field(field)?;
        if !known_header_keyword(&field.key) {
            return Err(TdmError::Unwritable {
                keyword: field.key.clone(),
                reason: "table 3-2 does not define it for a header",
            });
        }
    }
    // The header the writer will emit, in the order it will emit it, held to
    // the rules the reader holds a header to.
    check_written_section(
        "header",
        &HEADER_ORDER,
        &written_header_fields(tdm),
        policy,
        departures,
    )?;

    for (index, segment) in tdm.segments.iter().enumerate() {
        let number = index.saturating_add(1);
        for comment in &segment.metadata.comments {
            check_comment(comment)?;
        }
        for comment in &segment.data.comments {
            check_comment(&comment.text)?;
        }
        for field in &segment.metadata.fields {
            check_field(field)?;
            if !known_metadata_keyword(&field.key)? {
                return Err(TdmError::Unwritable {
                    keyword: field.key.clone(),
                    reason: "table 3-3 does not define it for a metadata section",
                });
            }
        }

        // The writer holds the value to the reader's rules by running the
        // reader's own checks over what it is about to write, so a rule cannot
        // hold in one direction and not the other. build_metadata is the
        // reader's: it bounds the indexed keywords, resolves every PATH
        // against the participants the segment defines, and requires
        // TIME_SYSTEM and PARTICIPANT_n. It reads `fields`, which is what the
        // writer emits, rather than the parsed properties beside it.
        let mut reader_warnings = Vec::new();
        build_metadata(
            MetadataBuilder {
                highest_rank: 0,
                comments: segment.metadata.comments.clone(),
                fields: segment.metadata.fields.clone(),
            },
            number,
            policy.as_read(),
            &mut reader_warnings,
        )?;
        check_written_section(
            "metadata",
            &METADATA_ORDER,
            &written_metadata_fields(segment),
            policy,
            departures,
        )?;

        // 4.5.2 c) puts a data-section comment before the first record, so a
        // comment the reader kept further in is one the writer is asked to put
        // back there.
        for comment in &segment.data.comments {
            if comment.before_record > 0 {
                emit_or_refuse(
                    policy.keyword_order,
                    TdmError::KeywordOutOfOrder {
                        line: None,
                        keyword: COMMENT_KEY.to_string(),
                        section: "data",
                    },
                    TdmDeparture::KeywordOutOfOrder {
                        keyword: COMMENT_KEY.to_string(),
                        section: "data",
                    },
                    departures,
                )?;
            }
        }

        if segment.data.records.is_empty() {
            emit_or_refuse(
                policy.empty_data_sections,
                TdmError::EmptyDataSection { segment: number },
                TdmDeparture::EmptyDataSection { segment: number },
                departures,
            )?;
        }

        // 4.3.9's timetags, 3.4.10's order and 3.4.11's uniqueness, checked by
        // the reader's own check_record_order over the records to be written.
        let mut data = DataBuilder::default();
        for record in &segment.data.records {
            let epoch = parse_epoch_key(&record.epoch).ok_or_else(|| TdmError::MalformedEpoch {
                line: None,
                keyword: record.keyword.clone(),
                text: record.epoch.clone(),
            })?;
            data.epochs.push(epoch);
            data.records.push(record.clone());
        }
        check_record_order(&data, number, policy.as_read(), &mut reader_warnings)?;
        departures.extend(reader_warnings.into_iter().map(departure_for));

        for record in &segment.data.records {
            if !record.value.value.is_finite() {
                return Err(TdmError::InvalidField {
                    keyword: record.keyword.clone(),
                    kind: TdmInputErrorKind::NonFinite,
                });
            }
            let observable = observable_from_keyword(&record.keyword)?;
            let parsed = parse_scalar(&record.keyword, &record.value.text, &observable)?;
            if parsed.value.to_bits() != record.value.value.to_bits() {
                return Err(TdmError::InvalidField {
                    keyword: record.keyword.clone(),
                    kind: TdmInputErrorKind::DecimalMismatch,
                });
            }
            if observable != record.observable {
                return Err(TdmError::InvalidField {
                    keyword: record.keyword.clone(),
                    kind: TdmInputErrorKind::UnknownKeyword,
                });
            }
            let expected_unit = unit_for_keyword(
                &record.keyword,
                &record.observable,
                &segment.metadata.range_units,
            );
            if expected_unit != record.unit {
                return Err(TdmError::InvalidField {
                    keyword: record.keyword.clone(),
                    kind: TdmInputErrorKind::UnitMismatch,
                });
            }
            validate_record_value(&record.keyword, &record.observable, &record.value)?;
        }
    }
    Ok(())
}

/// Split TDM text into lines on the terminators CCSDS 503.0-B-2 4.2.11 allows.
///
/// 4.2.11 terminates a line with "a single Carriage Return or a single Line Feed
/// or a Carriage Return/Line Feed pair or a Line Feed/Carriage Return pair", so
/// each pair ends one line rather than leaving an empty line between them.
/// `str::lines` splits on line feeds alone, which reads a file terminated with
/// carriage returns as a single line and refuses a message the standard allows.
fn tdm_lines(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut lines = Vec::new();
    let mut start = 0;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if !matches!(byte, b'\r' | b'\n') {
            index += 1;
            continue;
        }
        lines.push(&text[start..index]);
        let paired = matches!(
            (byte, bytes.get(index + 1).copied()),
            (b'\r', Some(b'\n')) | (b'\n', Some(b'\r'))
        );
        index += if paired { 2 } else { 1 };
        start = index;
    }
    if start < bytes.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// Refuse a line whose characters or length CCSDS 503.0-B-2 4.2.1 forbids.
///
/// 4.2.1: "The TDM line must contain only printable ASCII characters and
/// blanks. ASCII control characters (such as TAB, etc.) must not be used,
/// except as indicated below for the termination of the TDM line. A TDM line
/// must not exceed 254 ASCII characters and spaces (excluding line termination
/// character[s])."
/// A forgiven line reports one warning naming its first offending character,
/// matching what a strict read names, rather than one per character.
fn check_line(
    line_no: usize,
    line: &str,
    policy: TdmPolicy,
    warnings: &mut Vec<TdmWarning>,
) -> Result<(), TdmError> {
    let mut characters = 0;
    let mut reported = false;
    for (index, character) in line.chars().enumerate() {
        characters = index + 1;
        if matches!(character, ' '..='~') {
            continue;
        }
        match policy.non_printable {
            TdmLeniency::Strict => {
                return Err(TdmError::NonPrintableCharacter {
                    line: Some(line_no),
                    keyword: line_keyword(line),
                    column: index + 1,
                    character,
                })
            }
            TdmLeniency::Forgive => {
                if !reported {
                    reported = true;
                    warnings.push(TdmWarning::NonPrintableCharacter {
                        line: line_no,
                        keyword: line_keyword(line),
                        column: index + 1,
                        character,
                    });
                }
            }
        }
    }
    // 4.2.1 counts characters, which is the count above rather than the byte
    // length: a forgiven line can hold characters wider than one byte.
    if characters > MAX_LINE_CHARACTERS {
        match policy.long_lines {
            TdmLeniency::Strict => {
                return Err(TdmError::LineTooLong {
                    line: Some(line_no),
                    keyword: line_keyword(line),
                    length: characters,
                })
            }
            TdmLeniency::Forgive => warnings.push(TdmWarning::LineTooLong {
                line: line_no,
                keyword: line_keyword(line),
                length: characters,
            }),
        }
    }
    Ok(())
}

/// A line's first whitespace-delimited token.
///
/// Every line the writer builds is `KEYWORD = value`, `COMMENT text` or a
/// section marker, so this is the keyword. On the way in it is whatever the
/// line opens with, which is the keyword on a line 4.2.5 defines and the only
/// name there is on one it does not.
fn line_keyword(line: &str) -> String {
    line.split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string()
}

/// Hold a line the writer built to the two rules 4.2.1 sets for one: printable
/// ASCII and at most 254 characters.
///
/// These are the two axes `TdmWritePolicy` mirrors that the value itself can
/// carry. A tab never reaches here from a read, because 4.2.7 and 4.2.9 discard
/// the whitespace around a keyword and a value, but a character outside
/// printable ASCII inside a comment or a free-text value does, and so does a
/// value long enough to put its line over the cap. Neither can be repaired
/// without losing what the value says, so a strict write refuses and a lenient
/// one emits the departure and names the keyword.
fn check_written_line(
    line: &str,
    policy: TdmWritePolicy,
    departures: &mut Vec<TdmDeparture>,
) -> Result<(), TdmError> {
    let mut characters = 0;
    let mut reported = false;
    for (index, character) in line.chars().enumerate() {
        characters = index + 1;
        if matches!(character, ' '..='~') {
            continue;
        }
        if !reported {
            reported = true;
            emit_or_refuse(
                policy.non_printable,
                TdmError::NonPrintableCharacter {
                    line: None,
                    keyword: line_keyword(line),
                    column: index + 1,
                    character,
                },
                TdmDeparture::NonPrintableCharacter {
                    keyword: line_keyword(line),
                    character,
                },
                departures,
            )?;
        }
    }
    if characters > MAX_LINE_CHARACTERS {
        emit_or_refuse(
            policy.long_lines,
            TdmError::LineTooLong {
                line: None,
                keyword: line_keyword(line),
                length: characters,
            },
            TdmDeparture::LineTooLong {
                keyword: line_keyword(line),
                length: characters,
            },
            departures,
        )?;
    }
    Ok(())
}

/// The header lines the writer will emit, keyed and in emission order.
///
/// A comment is carried as a `COMMENT` field so the order check sees it where
/// 4.5.2 a) puts it; it is excepted from the one-value rule, since 4.2.5 c)
/// excepts `COMMENT` from the KVN syntax the rule is part of.
fn written_header_fields(tdm: &Tdm) -> Vec<TdmField> {
    let mut fields = vec![TdmField {
        key: VERSION_KEY.to_string(),
        value: tdm.version.clone(),
    }];
    fields.extend(tdm.comments.iter().map(|text| TdmField {
        key: COMMENT_KEY.to_string(),
        value: text.clone(),
    }));
    for (key, held) in [
        ("CREATION_DATE", &tdm.creation_date),
        ("ORIGINATOR", &tdm.originator),
        ("MESSAGE_ID", &tdm.message_id),
    ] {
        if let Some(value) = held {
            fields.push(TdmField {
                key: key.to_string(),
                value: value.clone(),
            });
        }
    }
    fields.extend(tdm.header_fields.iter().cloned());
    fields
}

/// The metadata lines the writer will emit for a segment, keyed and in
/// emission order.
fn written_metadata_fields(segment: &TdmSegment) -> Vec<TdmField> {
    let mut fields: Vec<TdmField> = segment
        .metadata
        .comments
        .iter()
        .map(|text| TdmField {
            key: COMMENT_KEY.to_string(),
            value: text.clone(),
        })
        .collect();
    fields.extend(segment.metadata.fields.iter().cloned());
    fields
}

/// Hold a section the writer is about to emit to the two rules the reader
/// holds the same section to: the order its table fixes, and the single value
/// assignment 4.2.5 a) gives each keyword.
fn check_written_section(
    section: &'static str,
    order: &[&str],
    fields: &[TdmField],
    policy: TdmWritePolicy,
    departures: &mut Vec<TdmDeparture>,
) -> Result<(), TdmError> {
    let mut highest = 0;
    for (index, field) in fields.iter().enumerate() {
        if let Some(rank) = keyword_rank(&field.key, order) {
            if rank < highest {
                emit_or_refuse(
                    policy.keyword_order,
                    TdmError::KeywordOutOfOrder {
                        line: None,
                        keyword: field.key.clone(),
                        section,
                    },
                    TdmDeparture::KeywordOutOfOrder {
                        keyword: field.key.clone(),
                        section,
                    },
                    departures,
                )?;
            } else {
                highest = rank;
            }
        }
        if field.key == COMMENT_KEY {
            continue;
        }
        let Some(previous) = fields[..index].iter().find(|held| held.key == field.key) else {
            continue;
        };
        if previous.value != field.value {
            return Err(TdmError::ConflictingKeyword {
                line: None,
                keyword: field.key.clone(),
                section,
                first: previous.value.clone(),
                second: field.value.clone(),
            });
        }
        emit_or_refuse(
            policy.repeated_keywords,
            TdmError::RepeatedKeyword {
                line: None,
                keyword: field.key.clone(),
                section,
            },
            TdmDeparture::RepeatedKeyword {
                keyword: field.key.clone(),
                section,
            },
            departures,
        )?;
    }
    Ok(())
}

/// Refuse a keyword repeated with a different value, and report one repeated
/// with the same value.
///
/// 4.2.5 a) makes a header or metadata line "a keyword, followed by an equals
/// sign ... followed by a single value assignment", and 3.2.3 and 3.3.1.8 give
/// each keyword one place in the order its table fixes, so a keyword carries
/// one value. Two that disagree leave the reader to choose, and choosing is
/// inventing a value, so it is refused under every policy. Two that agree say
/// the same thing twice, so both are kept where they were read and the repeat
/// is reported.
///
/// None of the 53 public files gathered for this audit repeats a keyword at
/// all, with the same value or a different one, which is why the reader has no
/// leniency axis here: there is nothing for one to forgive.
fn check_repeated_field(
    line: usize,
    section: &'static str,
    key: &str,
    value: &str,
    existing: &[TdmField],
    warnings: &mut Vec<TdmWarning>,
) -> Result<(), TdmError> {
    let Some(previous) = existing.iter().find(|field| field.key == key) else {
        return Ok(());
    };
    if previous.value != value {
        return Err(TdmError::ConflictingKeyword {
            line: Some(line),
            keyword: key.to_string(),
            section,
            first: previous.value.clone(),
            second: value.to_string(),
        });
    }
    warnings.push(TdmWarning::RepeatedKeyword {
        line,
        keyword: key.to_string(),
        section,
    });
    Ok(())
}

/// The departure a warning becomes on the way out.
///
/// The reader and the writer hold a value to the same rules, so a departure the
/// reader forgives is one the writer can be asked to emit. This match is where
/// that correspondence is written down; it carries no wildcard arm, so a
/// warning added to the reader without a departure to match it stops the build.
fn departure_for(warning: TdmWarning) -> TdmDeparture {
    match warning {
        TdmWarning::NonPrintableCharacter {
            line: _,
            keyword,
            column: _,
            character,
        } => TdmDeparture::NonPrintableCharacter { keyword, character },
        TdmWarning::LineTooLong {
            line: _,
            keyword,
            length,
        } => TdmDeparture::LineTooLong { keyword, length },
        TdmWarning::RepeatedKeyword {
            line: _,
            keyword,
            section,
        } => TdmDeparture::RepeatedKeyword { keyword, section },
        TdmWarning::MissingKeyword { keyword, segment } => {
            TdmDeparture::MissingKeyword { keyword, segment }
        }
        TdmWarning::EmptyDataSection { segment } => TdmDeparture::EmptyDataSection { segment },
        TdmWarning::RecordsOutOfOrder {
            segment,
            keyword,
            epoch,
        } => TdmDeparture::RecordsOutOfOrder {
            segment,
            keyword,
            epoch,
        },
        TdmWarning::DuplicateRecord {
            segment,
            keyword,
            epoch,
        } => TdmDeparture::DuplicateRecord {
            segment,
            keyword,
            epoch,
        },
        TdmWarning::UnterminatedFinalLine { line: _ } => TdmDeparture::UnterminatedFinalLine,
        TdmWarning::KeywordOutOfOrder {
            line: _,
            keyword,
            section,
        } => TdmDeparture::KeywordOutOfOrder { keyword, section },
    }
}

/// Refuse an absent mandatory keyword, or forgive it under `policy`.
fn require_keyword(
    present: bool,
    keyword: &str,
    segment: Option<usize>,
    policy: TdmPolicy,
    warnings: &mut Vec<TdmWarning>,
) -> Result<(), TdmError> {
    if present {
        return Ok(());
    }
    match policy.missing_keywords {
        TdmLeniency::Strict => Err(TdmError::MissingKeyword {
            keyword: keyword.to_string(),
            segment,
        }),
        TdmLeniency::Forgive => {
            warnings.push(TdmWarning::MissingKeyword {
                keyword: keyword.to_string(),
                segment,
            });
            Ok(())
        }
    }
}

fn parse_assignment(line: &str) -> Option<(String, String)> {
    let (key, raw_value) = line.split_once('=')?;
    let key = key.trim().to_string();
    Some((key, raw_value.trim().to_string()))
}

fn comment_text(line: &str) -> Option<String> {
    if line == COMMENT_KEY {
        return Some(String::new());
    }
    let rest = line.strip_prefix(COMMENT_KEY)?;
    if rest
        .chars()
        .next()
        .is_some_and(|character| character.is_ascii_whitespace())
    {
        Some(rest.trim_start().to_string())
    } else {
        None
    }
}

fn comment_line(comment: &String) -> String {
    if comment.is_empty() {
        COMMENT_KEY.to_string()
    } else {
        format!("{COMMENT_KEY} {comment}")
    }
}

fn field_line(field: &TdmField) -> String {
    format!("{} = {}", field.key, field.value)
}

/// Refuse a field whose key is a keyword that cannot carry a value.
///
/// `COMMENT` is excepted from the KVN syntax by CCSDS 503.0-B-2 4.2.5 c), and
/// 4.5.3 reads any line whose keyword is followed by a space as a comment. A
/// field keyed `COMMENT` has no assignment form: writing `COMMENT = value`
/// emits a line that reads back as a comment rather than as the field.
///
/// The key is compared trimmed, because [`field_line`] writes `"COMMENT "` and
/// `"COMMENT"` as the same line and a reader takes both back as a comment.
///
/// An empty value is refused on the same footing: 4.3.1 requires "A non-empty
/// value field must be specified for each keyword provided", and `KEY = ` reads
/// back as the empty value the standard does not allow.
fn check_field(field: &TdmField) -> Result<(), TdmError> {
    if keyword_takes_no_value(&field.key) {
        return Err(TdmError::KeywordNotAssignable {
            keyword: field.key.clone(),
        });
    }
    if field.value.is_empty() {
        return Err(TdmError::EmptyValue {
            line: None,
            keyword: field.key.clone(),
        });
    }

    // What follows is the set no reader can take back, whatever policy wrote
    // it: each gives a line that reparses as a different value, which is the
    // defect this module exists to close. parse_kvn cannot produce any of
    // them, so only a caller building a value directly reaches here.
    let unwritable = if field.key.is_empty() {
        Some("the keyword is empty")
    } else if field.key.contains('=') {
        Some("an equals sign in the keyword would split the line elsewhere")
    } else if field.key.trim() != field.key {
        Some("whitespace around the keyword, which 4.2.7 drops on the way back")
    } else if field.value.trim() != field.value {
        Some("whitespace around the value, which 4.2.8 and 4.2.9 drop on the way back")
    } else if field.key.contains(['\r', '\n']) || field.value.contains(['\r', '\n']) {
        Some("a line terminator, which would write a second line")
    } else {
        None
    };
    if let Some(reason) = unwritable {
        return Err(TdmError::Unwritable {
            keyword: field.key.clone(),
            reason,
        });
    }

    Ok(())
}

/// Emit a departure the policy allows, or refuse the value.
///
/// A refusal that no policy forgives — [`TdmError::ConflictingKeyword`] is the
/// case — does not go through `emit_or_refuse` and cannot, because there is no
/// departure to emit.
fn emit_or_refuse(
    leniency: TdmLeniency,
    error: TdmError,
    departure: TdmDeparture,
    departures: &mut Vec<TdmDeparture>,
) -> Result<(), TdmError> {
    match leniency {
        TdmLeniency::Strict => Err(error),
        TdmLeniency::Forgive => {
            departures.push(departure);
            Ok(())
        }
    }
}

/// A record's timetag reduced to something two records can be compared by.
///
/// CCSDS 503.0-B-2 4.3.9 gives two forms, `YYYY-MM-DDThh:mm:ss[.d->d][Z]` and
/// `YYYY-DDDThh:mm:ss[.d->d][Z]`. Both reduce to a day number and a second of
/// day, so a message may use either and still be ordered. Leap seconds are not
/// modeled: 3.4.10 asks only for chronological order within one time system,
/// which the reduction preserves.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct EpochKey {
    day: i64,
    second_of_day: f64,
}

/// Days from 1970-01-01 for a proleptic Gregorian date, after Howard Hinnant's
/// `days_from_civil`.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let shifted_month = (month + 9) % 12;
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// Report whether `year` is a leap year in the proleptic Gregorian calendar.
fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// The days in `month` of `year`, both one-based.
fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Read a fixed-width run of decimal digits with the leading zeros 4.3.9
/// requires, as an integer.
fn fixed_digits(text: &str, width: usize) -> Option<i64> {
    if text.len() != width || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse::<i64>().ok()
}

/// Read a timetag in either form 4.3.9 defines.
fn parse_epoch_key(text: &str) -> Option<EpochKey> {
    let body = text.strip_suffix('Z').unwrap_or(text);
    let (date, time) = body.split_once('T')?;

    let day = match date.len() {
        10 => {
            let year = fixed_digits(date.get(0..4)?, 4)?;
            if date.get(4..5)? != "-" || date.get(7..8)? != "-" {
                return None;
            }
            let month = fixed_digits(date.get(5..7)?, 2)?;
            let day_of_month = fixed_digits(date.get(8..10)?, 2)?;
            if !(1..=12).contains(&month) || day_of_month < 1 {
                return None;
            }
            if day_of_month > days_in_month(year, month) {
                return None;
            }
            days_from_civil(year, month, day_of_month)
        }
        8 => {
            let year = fixed_digits(date.get(0..4)?, 4)?;
            if date.get(4..5)? != "-" {
                return None;
            }
            let day_of_year = fixed_digits(date.get(5..8)?, 3)?;
            let length = if is_leap_year(year) { 366 } else { 365 };
            if !(1..=length).contains(&day_of_year) {
                return None;
            }
            days_from_civil(year, 1, 1) + day_of_year - 1
        }
        _ => return None,
    };

    let (clock, fraction) = match time.split_once('.') {
        Some((clock, digits)) => {
            if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
            (clock, format!("0.{digits}").parse::<f64>().ok()?)
        }
        None => (time, 0.0),
    };
    if clock.len() != 8 || clock.get(2..3)? != ":" || clock.get(5..6)? != ":" {
        return None;
    }
    let hours = fixed_digits(clock.get(0..2)?, 2)?;
    let minutes = fixed_digits(clock.get(3..5)?, 2)?;
    let seconds = fixed_digits(clock.get(6..8)?, 2)?;
    // A leap second is written as 60, so the range runs to 60 rather than 59.
    if hours > 23 || minutes > 59 || seconds > 60 {
        return None;
    }

    Some(EpochKey {
        day,
        second_of_day: (hours * 3600 + minutes * 60 + seconds) as f64 + fraction,
    })
}

/// Check a data section against the ordering rules 3.4.10 and 3.4.11 set.
///
/// 3.4.10: "in any given Data Section, the data for any given keyword shall be
/// in chronological order". 3.4.11: "Each keyword/timetag combination must be
/// unique within a given Data Section".
///
/// Both are forgivable. A data section is a sequence rather than a set of keyed
/// slots, so two records stamped at one instant can both be kept, in file
/// order, with nothing invented and nothing dropped; the caller sees what the
/// producer wrote. Order is forgivable for the same reason: each record carries
/// its own timetag, so reading them out of order changes no value. Figure E-17
/// needs the first of these, giving `RCS` twice at `2011-05-11T10:26:33.7008`
/// with different values, which looks like a typo for its neighbour's timetag.
fn check_record_order(
    builder: &DataBuilder,
    segment: usize,
    policy: TdmPolicy,
    warnings: &mut Vec<TdmWarning>,
) -> Result<(), TdmError> {
    let mut seen: HashSet<(&str, i64, u64)> = HashSet::new();
    let mut latest: HashMap<&str, EpochKey> = HashMap::new();

    for (record, epoch) in builder.records.iter().zip(&builder.epochs) {
        let keyword = record.keyword.as_str();
        if !seen.insert((keyword, epoch.day, epoch.second_of_day.to_bits())) {
            match policy.duplicate_records {
                TdmLeniency::Strict => {
                    return Err(TdmError::DuplicateRecord {
                        segment,
                        keyword: record.keyword.clone(),
                        epoch: record.epoch.clone(),
                    })
                }
                TdmLeniency::Forgive => warnings.push(TdmWarning::DuplicateRecord {
                    segment,
                    keyword: record.keyword.clone(),
                    epoch: record.epoch.clone(),
                }),
            }
        }

        match latest.get_mut(keyword) {
            Some(previous) => {
                if epoch < previous {
                    match policy.record_order {
                        TdmLeniency::Strict => {
                            return Err(TdmError::RecordsOutOfOrder {
                                segment,
                                keyword: record.keyword.clone(),
                                epoch: record.epoch.clone(),
                            })
                        }
                        TdmLeniency::Forgive => {
                            warnings.push(TdmWarning::RecordsOutOfOrder {
                                segment,
                                keyword: record.keyword.clone(),
                                epoch: record.epoch.clone(),
                            });
                        }
                    }
                } else {
                    *previous = *epoch;
                }
            }
            None => {
                latest.insert(keyword, *epoch);
            }
        }
    }
    Ok(())
}

/// The rank a keyword has in the order its table fixes, or `None` for a
/// keyword the table does not order.
///
/// An indexed keyword ranks where its base does, so `PARTICIPANT_2` ranks with
/// `PARTICIPANT_1` and neither is out of order beside the other.
fn keyword_rank(keyword: &str, order: &[&str]) -> Option<usize> {
    if let Some(index) = order.iter().position(|named| *named == keyword) {
        return Some(index);
    }
    order.iter().position(|base| {
        keyword
            .strip_prefix(*base)
            .is_some_and(|rest| rest.starts_with('_'))
    })
}

/// Check a keyword against the order its table fixes, or forgive it.
///
/// 3.2.3: "The order of occurrence of the mandatory and optional KVN
/// assignments shall be fixed as shown in table 3-2", and 3.3.1.8 says the same
/// of table 3-3. Order changes no value, so it is forgivable; each keyword
/// carries its own meaning wherever it sits.
fn check_keyword_order(
    line: usize,
    keyword: &str,
    section: &'static str,
    order: &[&str],
    highest: &mut usize,
    policy: TdmPolicy,
    warnings: &mut Vec<TdmWarning>,
) -> Result<(), TdmError> {
    let Some(rank) = keyword_rank(keyword, order) else {
        return Ok(());
    };
    if rank >= *highest {
        *highest = rank;
        return Ok(());
    }
    match policy.keyword_order {
        TdmLeniency::Strict => Err(TdmError::KeywordOutOfOrder {
            line: Some(line),
            keyword: keyword.to_string(),
            section,
        }),
        TdmLeniency::Forgive => {
            warnings.push(TdmWarning::KeywordOutOfOrder {
                line,
                keyword: keyword.to_string(),
                section,
            });
            Ok(())
        }
    }
}

/// Report whether `key` is a keyword 4.2.5 c) excepts from the KVN syntax.
///
/// The key is compared trimmed, because [`field_line`] writes `"COMMENT "` and
/// `"COMMENT"` as the same line and a reader takes both back the same way.
fn keyword_takes_no_value(key: &str) -> bool {
    EXCEPTED_KEYWORDS.contains(&key.trim())
}

/// Report whether table 3-2 defines `key` for a TDM header.
fn known_header_keyword(key: &str) -> bool {
    matches!(
        key,
        VERSION_KEY | "CREATION_DATE" | "ORIGINATOR" | "MESSAGE_ID"
    )
}

/// Report whether table 3-3 defines `key` for a TDM metadata section.
///
/// An indexed family reports its own out-of-range suffix, so `PARTICIPANT_9` is
/// an invalid index rather than an undefined keyword.
///
/// `EPHEMERIS_NAME` is taken unindexed as well as indexed. Table 3-3 lists only
/// `EPHEMERIS_NAME_n`, but Figure E-17 writes the bare keyword and the annex I
/// summary sheet lists it bare five times, so refusing it would refuse the
/// standard's own worked example. The bare form covers the single-participant
/// case, as it does for `PATH` and `RECEIVE_FREQ`, which the tables define
/// alongside their indexed forms.
fn known_metadata_keyword(key: &str) -> Result<bool, TdmError> {
    for (base, max) in [
        ("PARTICIPANT", 5u8),
        ("PATH", 2),
        ("EPHEMERIS_NAME", 5),
        ("TRANSMIT_DELAY", 5),
        ("RECEIVE_DELAY", 5),
    ] {
        if indexed_suffix_in_range(key, base, 1, max)?.is_some() {
            return Ok(true);
        }
    }

    Ok(matches!(
        key,
        "TRACK_ID"
            | "DATA_TYPES"
            | "EPHEMERIS_NAME"
            | "TIME_SYSTEM"
            | "START_TIME"
            | "STOP_TIME"
            | "MODE"
            | "PATH"
            | "TRANSMIT_BAND"
            | "RECEIVE_BAND"
            | "TURNAROUND_NUMERATOR"
            | "TURNAROUND_DENOMINATOR"
            | "TIMETAG_REF"
            | "INTEGRATION_INTERVAL"
            | "INTEGRATION_REF"
            | "FREQ_OFFSET"
            | "RANGE_MODE"
            | "RANGE_MODULUS"
            | "RANGE_UNITS"
            | "ANGLE_TYPE"
            | "REFERENCE_FRAME"
            | "INTERPOLATION"
            | "INTERPOLATION_DEGREE"
            | "DOPPLER_COUNT_BIAS"
            | "DOPPLER_COUNT_SCALE"
            | "DOPPLER_COUNT_ROLLOVER"
            | "DATA_QUALITY"
            | "CORRECTIONS_APPLIED"
            | "CORRECTION_ANGLE_1"
            | "CORRECTION_ANGLE_2"
            | "CORRECTION_DOPPLER"
            | "CORRECTION_MAG"
            | "CORRECTION_RANGE"
            | "CORRECTION_RCS"
            | "CORRECTION_RECEIVE"
            | "CORRECTION_TRANSMIT"
            | "CORRECTION_ABERRATION_YEARLY"
            | "CORRECTION_ABERRATION_DIURNAL"
    ))
}

/// Refuse a comment whose text the KVN form cannot carry.
///
/// 4.5.3 takes "the remainder of the line" as the comment value after the
/// keyword and its space, and 4.2.9 drops whitespace before the end of a line,
/// so text padded at either end does not read back as itself. A newline would
/// write a second line, which reparses as a fabricated field or a malformed
/// line rather than as part of this comment.
fn check_comment(text: &str) -> Result<(), TdmError> {
    let reason = if text.trim() != text {
        Some("whitespace around the text, which 4.2.9 drops on the way back")
    } else if text.contains(['\r', '\n']) {
        Some("a line terminator, which would write a second line")
    } else {
        None
    };
    match reason {
        Some(reason) => Err(TdmError::Unwritable {
            keyword: COMMENT_KEY.to_string(),
            reason,
        }),
        None => Ok(()),
    }
}

/// Check a `CCSDS_TDM_VERS` value is the `x.y` form CCSDS 503.0-B-2 3.2.5 sets.
///
/// 3.2.5: "the value shall have the form of x.y where y is incremented for
/// corrections and minor changes, and x is incremented for major changes."
/// Table 3-2 gives `0.12`, `1.0` and `2.0` as the values so far.
fn check_version(line: Option<usize>, value: &str) -> Result<(), TdmError> {
    let well_formed = value.split_once('.').is_some_and(|(major, minor)| {
        !major.is_empty()
            && !minor.is_empty()
            && major.bytes().all(|byte| byte.is_ascii_digit())
            && minor.bytes().all(|byte| byte.is_ascii_digit())
    });
    if well_formed {
        Ok(())
    } else {
        Err(TdmError::InvalidVersion {
            line,
            value: value.to_string(),
        })
    }
}

fn empty_to_none(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

fn malformed_record(line: usize, keyword: &str) -> TdmError {
    TdmError::MalformedRecord {
        line,
        keyword: keyword.to_string(),
    }
}

fn has_displayed_unit(value: &str) -> bool {
    let trimmed = value.trim_end();
    trimmed.ends_with(']') && trimmed.rfind('[').is_some()
}

fn indexed_suffix(key: &str, base: &str) -> Result<Option<u8>, TdmError> {
    let Some(suffix) = key
        .strip_prefix(base)
        .and_then(|rest| rest.strip_prefix('_'))
    else {
        return Ok(None);
    };
    // Table 3-3 writes the indexer as a single digit, so a padded suffix such
    // as `PARTICIPANT_01` is not one of the keywords it defines, and reading it
    // as 1 would give one index two spellings.
    if suffix.is_empty()
        || !suffix.chars().all(|character| character.is_ascii_digit())
        || (suffix.len() > 1 && suffix.starts_with('0'))
    {
        return Err(invalid_index(key));
    }
    suffix
        .parse::<u8>()
        .map(Some)
        .map_err(|_| invalid_index(key))
}

fn indexed_suffix_in_range(
    key: &str,
    base: &str,
    min: u8,
    max: u8,
) -> Result<Option<u8>, TdmError> {
    let Some(index) = indexed_suffix(key, base)? else {
        return Ok(None);
    };
    if (min..=max).contains(&index) {
        Ok(Some(index))
    } else {
        Err(invalid_index(key))
    }
}

fn invalid_index(keyword: &str) -> TdmError {
    TdmError::InvalidField {
        keyword: keyword.to_string(),
        kind: TdmInputErrorKind::InvalidIndex,
    }
}

fn unknown_keyword(keyword: &str) -> TdmError {
    TdmError::InvalidField {
        keyword: keyword.to_string(),
        kind: TdmInputErrorKind::UnknownKeyword,
    }
}

fn range_unit_from_label(label: &str) -> Result<TdmUnit, TdmError> {
    match label {
        "km" => Ok(TdmUnit::Kilometers),
        "s" => Ok(TdmUnit::Seconds),
        "RU" => Ok(TdmUnit::RangeUnits),
        _ => Err(TdmError::InvalidField {
            keyword: "RANGE_UNITS".to_string(),
            kind: TdmInputErrorKind::UnitMismatch,
        }),
    }
}

fn unit_for_keyword(keyword: &str, observable: &TdmObservable, range_units: &TdmUnit) -> TdmUnit {
    match observable {
        TdmObservable::Range => range_units.clone(),
        TdmObservable::DopplerInstantaneous | TdmObservable::DopplerIntegrated => {
            TdmUnit::KilometersPerSecond
        }
        TdmObservable::ReceiveFreq { .. } | TdmObservable::TransmitFreq { .. } => TdmUnit::Hertz,
        TdmObservable::TransmitFreqRate { .. } => TdmUnit::HertzPerSecond,
        TdmObservable::Angle1 | TdmObservable::Angle2 => TdmUnit::Degrees,
        TdmObservable::Other(_) => unit_for_other_keyword(keyword),
    }
}

fn unit_for_other_keyword(keyword: &str) -> TdmUnit {
    if indexed_suffix_in_range(keyword, "RECEIVE_PHASE_CT", 1, 5).is_ok_and(|value| value.is_some())
        || indexed_suffix_in_range(keyword, "TRANSMIT_PHASE_CT", 1, 5)
            .is_ok_and(|value| value.is_some())
    {
        return TdmUnit::Dimensionless;
    }
    match keyword {
        "CARRIER_POWER" => TdmUnit::DecibelWatts,
        "CLOCK_BIAS" | "DOR" | "VLBI_DELAY" => TdmUnit::Seconds,
        "CLOCK_DRIFT" => TdmUnit::SecondsPerSecond,
        "DOPPLER_COUNT" | "MAG" => TdmUnit::Dimensionless,
        "PC_N0" | "PR_N0" => TdmUnit::DecibelHertz,
        "PRESSURE" => TdmUnit::Hectopascals,
        "RCS" => TdmUnit::SquareMeters,
        "RHUMIDITY" => TdmUnit::Percent,
        "STEC" => TdmUnit::TotalElectronContentUnits,
        "TEMPERATURE" => TdmUnit::Kelvin,
        "TROPO_DRY" | "TROPO_WET" => TdmUnit::Meters,
        _ => unreachable!("table 3-5 keyword checked before unit lookup"),
    }
}

fn phase_count_keyword(keyword: &str) -> bool {
    keyword.starts_with("RECEIVE_PHASE_CT_") || keyword.starts_with("TRANSMIT_PHASE_CT_")
}

fn known_table_3_5_other_keyword(keyword: &str) -> Result<bool, TdmError> {
    if indexed_suffix_in_range(keyword, "RECEIVE_PHASE_CT", 1, 5)?.is_some()
        || indexed_suffix_in_range(keyword, "TRANSMIT_PHASE_CT", 1, 5)?.is_some()
    {
        return Ok(true);
    }

    Ok(matches!(
        keyword,
        "CARRIER_POWER"
            | "CLOCK_BIAS"
            | "CLOCK_DRIFT"
            | "DOPPLER_COUNT"
            | "DOR"
            | "MAG"
            | "PC_N0"
            | "PR_N0"
            | "PRESSURE"
            | "RCS"
            | "RHUMIDITY"
            | "STEC"
            | "TEMPERATURE"
            | "TROPO_DRY"
            | "TROPO_WET"
            | "VLBI_DELAY"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE: &str = "\
CCSDS_TDM_VERS = 2.0
COMMENT sample
CREATION_DATE = 2005-160T20:15:00Z
ORIGINATOR = NASA
META_START
TIME_SYSTEM = UTC
PARTICIPANT_1 = DSS-25
PARTICIPANT_2 = yyyy-nnnA
MODE = SEQUENTIAL
PATH = 2,1
RANGE_UNITS = km
META_STOP
DATA_START
TRANSMIT_FREQ_2 = 2005-159T17:41:00 32023442781.733
RECEIVE_FREQ_1 = 2005-159T17:41:00 32021034790.7265
RANGE = 2005-159T17:41:00 80452.7542
ANGLE_1 = 2005-159T17:41:00 256.64002393
ANGLE_2 = 2005-159T17:41:00 13.38100016
DATA_STOP\n";

    #[test]
    fn parses_frequency_records_without_reformatting_decimal_tokens() {
        let tdm = parse_kvn(SIMPLE).unwrap();
        let records = &tdm.segments[0].data.records;
        assert_eq!(records[0].keyword, "TRANSMIT_FREQ_2");
        assert_eq!(records[0].value.text, "32023442781.733");
        assert_eq!(records[0].value.value.to_bits(), 0x421d_d2fb_d576_ee98);
        assert_eq!(records[0].unit, TdmUnit::Hertz);
        assert_eq!(records[1].keyword, "RECEIVE_FREQ_1");
        assert_eq!(records[1].value.text, "32021034790.7265");
        assert_eq!(records[1].value.value.to_bits(), 0x421d_d268_dc9a_e7f0);
    }

    #[test]
    fn canonical_encode_is_stable() {
        let tdm = parse_kvn(SIMPLE).unwrap();
        let encoded = encode_kvn(&tdm).unwrap();
        let reparsed = parse_kvn(&encoded).unwrap();
        assert_eq!(encode_kvn(&reparsed).unwrap(), encoded);
        assert_eq!(reparsed, tdm);
    }

    #[test]
    fn malformed_data_record_is_typed_error() {
        let err = parse_kvn(
            "\
CCSDS_TDM_VERS = 2.0
CREATION_DATE = 2005-160T20:15:00Z
ORIGINATOR = NASA
META_START
TIME_SYSTEM = UTC
PARTICIPANT_1 = DSS-25
META_STOP
DATA_START
RECEIVE_FREQ_1 = 2005-159T17:41:00
DATA_STOP\n",
        )
        .unwrap_err();
        assert_eq!(
            err,
            TdmError::MalformedRecord {
                line: 9,
                keyword: "RECEIVE_FREQ_1".to_string()
            }
        );
    }

    #[test]
    fn invalid_transmit_frequency_is_rejected() {
        let err = parse_kvn(
            "\
CCSDS_TDM_VERS = 2.0
CREATION_DATE = 2005-160T20:15:00Z
ORIGINATOR = NASA
META_START
TIME_SYSTEM = UTC
PARTICIPANT_1 = DSS-25
META_STOP
DATA_START
TRANSMIT_FREQ_1 = 2005-159T17:41:00 0.0
DATA_STOP\n",
        )
        .unwrap_err();
        assert_eq!(
            err,
            TdmError::InvalidField {
                keyword: "TRANSMIT_FREQ_1".to_string(),
                kind: TdmInputErrorKind::NotPositive,
            }
        );
    }

    #[test]
    fn comment_assignment_is_refused_in_header_and_metadata() {
        let header = "\
CCSDS_TDM_VERS = 2.0
CREATION_DATE = 2005-160T20:15:00Z
ORIGINATOR = NASA
COMMENT=
META_START
TIME_SYSTEM = UTC
PARTICIPANT_1 = DSS-25
META_STOP
DATA_START
RANGE = 2005-159T17:41:00 1.0
DATA_STOP\n";
        assert_eq!(
            parse_kvn(header),
            Err(TdmError::MalformedLine {
                line: 4,
                text: "COMMENT=".to_string(),
            })
        );

        let metadata = "\
CCSDS_TDM_VERS = 2.0
CREATION_DATE = 2005-160T20:15:00Z
ORIGINATOR = NASA
META_START
TIME_SYSTEM = UTC
COMMENT=file = tdm.dat
PARTICIPANT_1 = DSS-25
META_STOP
DATA_START
RANGE = 2005-159T17:41:00 1.0
DATA_STOP\n";
        assert_eq!(
            parse_kvn(metadata),
            Err(TdmError::MalformedLine {
                line: 6,
                text: "COMMENT=file = tdm.dat".to_string(),
            })
        );
    }

    #[test]
    fn comment_keyword_followed_by_a_space_stays_a_comment() {
        // 4.5.3 takes the rest of the line as the comment value, so a comment
        // whose text opens with `=` is still a comment, not an assignment.
        let tdm = parse_kvn(
            "\
CCSDS_TDM_VERS = 2.0
COMMENT = file = tdm.dat
CREATION_DATE = 2005-160T20:15:00Z
ORIGINATOR = NASA
META_START
TIME_SYSTEM = UTC
PARTICIPANT_1 = DSS-25
META_STOP
DATA_START
RANGE = 2005-159T17:41:00 1.0
DATA_STOP\n",
        )
        .unwrap();
        assert_eq!(tdm.comments, vec!["= file = tdm.dat".to_string()]);
        assert!(tdm.header_fields.is_empty());
    }

    #[test]
    fn encode_refuses_a_comment_keyed_field() {
        let comment_field = TdmField {
            key: COMMENT_KEY.to_string(),
            value: String::new(),
        };

        let mut header_case = parse_kvn(SIMPLE).unwrap();
        header_case.header_fields.push(comment_field.clone());
        assert_eq!(
            encode_kvn(&header_case),
            Err(TdmError::KeywordNotAssignable {
                keyword: COMMENT_KEY.to_string(),
            })
        );

        let mut metadata_case = parse_kvn(SIMPLE).unwrap();
        metadata_case.segments[0]
            .metadata
            .fields
            .push(comment_field);
        assert_eq!(
            encode_kvn(&metadata_case),
            Err(TdmError::KeywordNotAssignable {
                keyword: COMMENT_KEY.to_string(),
            })
        );

        // A padded key writes the same line, so it is refused on the same
        // footing and reports the key as the field holds it.
        let mut padded_case = parse_kvn(SIMPLE).unwrap();
        padded_case.header_fields.push(TdmField {
            key: "COMMENT ".to_string(),
            value: String::new(),
        });
        assert_eq!(
            encode_kvn(&padded_case),
            Err(TdmError::KeywordNotAssignable {
                keyword: "COMMENT ".to_string(),
            })
        );
    }

    const TERMINATOR_BODY: &str = "CCSDS_TDM_VERS = 2.0|CREATION_DATE = 2005-160T20:15:00Z|\
ORIGINATOR = NASA|META_START|TIME_SYSTEM = UTC|PARTICIPANT_1 = DSS-25|META_STOP|\
DATA_START|RANGE = 2005-159T17:41:00 1.0|DATA_STOP\n";

    #[test]
    fn a_line_ends_at_every_terminator_the_standard_allows() {
        // 4.2.11: "a single Carriage Return or a single Line Feed or a Carriage
        // Return/Line Feed pair or a Line Feed/Carriage Return pair".
        //
        // Asserted on the lines the splitter produced. A pair read as two
        // terminators leaves an empty line between them, which the parser then
        // skips, so a message parses either way and parsing alone cannot tell
        // the two apart.
        for terminator in ["\n", "\r", "\r\n", "\n\r"] {
            assert_eq!(
                tdm_lines(&format!("A{terminator}B{terminator}C")),
                vec!["A", "B", "C"],
                "between lines, terminator {terminator:?}"
            );
            assert_eq!(
                tdm_lines(&format!("A{terminator}")),
                vec!["A"],
                "a trailing terminator closes the last line, {terminator:?}"
            );
        }

        // 4.2.10 allows a blank line anywhere, so two single terminators in a
        // row keep the empty line between them. That is the case a pair must
        // not produce, and the one the old assertion could not see.
        assert_eq!(tdm_lines("A\n\nB"), vec!["A", "", "B"]);
        assert_eq!(tdm_lines("A\r\rB"), vec!["A", "", "B"]);

        // And a message reads the same under each of the four.
        for terminator in ["\n", "\r", "\r\n", "\n\r"] {
            let text = TERMINATOR_BODY.replace('|', terminator);
            let tdm = parse_kvn(&text)
                .unwrap_or_else(|err| panic!("terminator {terminator:?} must parse: {err}"));
            assert_eq!(tdm.segments.len(), 1, "terminator {terminator:?} segments");
            assert_eq!(
                tdm.segments[0].data.records.len(),
                1,
                "terminator {terminator:?} records"
            );
        }
    }

    #[test]
    fn end_of_input_line_numbers_count_the_lines_the_file_has() {
        // The three end-of-input errors report the line past the last, and that
        // count comes from the terminators the file uses rather than from line
        // feeds alone. A carriage-return-terminated file used to arrive as one
        // line, so none of these could report a number that meant anything.
        const HEAD: [&str; 3] = [
            "CCSDS_TDM_VERS = 2.0",
            "CREATION_DATE = 2005-160T20:15:00Z",
            "ORIGINATOR = NASA",
        ];
        const META: [&str; 3] = ["META_START", "TIME_SYSTEM = UTC", "PARTICIPANT_1 = DSS-25"];

        for terminator in ["\n", "\r", "\r\n", "\n\r"] {
            let unclosed_metadata = [&HEAD[..], &META[..2]].concat().join(terminator);
            assert_eq!(
                parse_kvn(&unclosed_metadata),
                Err(TdmError::Section {
                    line: 6,
                    detail: "unclosed metadata block",
                }),
                "unclosed metadata, terminator {terminator:?}"
            );

            let metadata_without_data = [&HEAD[..], &META[..], &["META_STOP"][..]]
                .concat()
                .join(terminator);
            assert_eq!(
                parse_kvn(&metadata_without_data),
                Err(TdmError::Section {
                    line: 8,
                    detail: "metadata without data block",
                }),
                "metadata without data, terminator {terminator:?}"
            );

            let unclosed_data = [
                &HEAD[..],
                &META[..],
                &["META_STOP", "DATA_START", "RANGE = 2005-159T17:41:00 1.0"][..],
            ]
            .concat()
            .join(terminator);
            assert_eq!(
                parse_kvn(&unclosed_data),
                Err(TdmError::Section {
                    line: 10,
                    detail: "unclosed data block",
                }),
                "unclosed data, terminator {terminator:?}"
            );
        }
    }

    #[test]
    fn a_character_outside_printable_ascii_is_refused() {
        // 4.2.1: "The TDM line must contain only printable ASCII characters and
        // blanks. ASCII control characters (such as TAB, etc.) must not be used".
        let with_tab = "\
CCSDS_TDM_VERS = 2.0
CREATION_DATE = 2005-160T20:15:00Z
ORIGINATOR = NASA
META_START
\tTIME_SYSTEM = UTC
PARTICIPANT_1 = DSS-25
META_STOP
DATA_START
RANGE = 2005-159T17:41:00 1.0
DATA_STOP\n";
        assert_eq!(
            parse_kvn(with_tab),
            Err(TdmError::NonPrintableCharacter {
                line: Some(5),
                keyword: "TIME_SYSTEM".to_string(),
                column: 1,
                character: '\t',
            })
        );

        // A right double quotation mark, as two public TDM corpora carry in a
        // clock-offset comment. 4.5.2 a) puts a header comment between
        // CCSDS_TDM_VERS and CREATION_DATE.
        let with_non_ascii = "\
CCSDS_TDM_VERS = 2.0
COMMENT clock minus UTC\u{201d}
CREATION_DATE = 2005-160T20:15:00Z
ORIGINATOR = NASA
META_START
TIME_SYSTEM = UTC
PARTICIPANT_1 = DSS-25
META_STOP
DATA_START
RANGE = 2005-159T17:41:00 1.0
DATA_STOP\n";
        assert_eq!(
            parse_kvn(with_non_ascii),
            Err(TdmError::NonPrintableCharacter {
                line: Some(2),
                keyword: COMMENT_KEY.to_string(),
                column: 24,
                character: '\u{201d}',
            })
        );
    }

    #[test]
    fn a_line_over_the_character_limit_is_refused() {
        // 4.2.1: "A TDM line must not exceed 254 ASCII characters and spaces
        // (excluding line termination character[s])".
        let prefix = "TIME_SYSTEM = ";
        let value = "A".repeat(MAX_LINE_CHARACTERS);
        let text = format!(
            "\
CCSDS_TDM_VERS = 2.0
CREATION_DATE = 2005-160T20:15:00Z
ORIGINATOR = NASA
META_START
{prefix}{value}
PARTICIPANT_1 = DSS-25
META_STOP
DATA_START
RANGE = 2005-159T17:41:00 1.0
DATA_STOP\n"
        );
        assert_eq!(
            parse_kvn(&text),
            Err(TdmError::LineTooLong {
                line: Some(5),
                keyword: "TIME_SYSTEM".to_string(),
                length: prefix.len() + MAX_LINE_CHARACTERS,
            })
        );

        // A line of exactly the limit is accepted.
        let exact = "B".repeat(MAX_LINE_CHARACTERS - prefix.len());
        let text = format!(
            "\
CCSDS_TDM_VERS = 2.0
CREATION_DATE = 2005-160T20:15:00Z
ORIGINATOR = NASA
META_START
{prefix}{exact}
PARTICIPANT_1 = DSS-25
META_STOP
DATA_START
RANGE = 2005-159T17:41:00 1.0
DATA_STOP\n"
        );
        let tdm = parse_kvn(&text).expect("a line of exactly 254 characters is allowed");
        assert_eq!(
            tdm.segments[0].metadata.time_system.as_deref(),
            Some(exact.as_str())
        );
    }

    const CONFORMING: &str = "\
CCSDS_TDM_VERS = 2.0
CREATION_DATE = 2005-160T20:15:00Z
ORIGINATOR = NASA
META_START
TIME_SYSTEM = UTC
PARTICIPANT_1 = DSS-25
META_STOP
DATA_START
RANGE = 2005-159T17:41:00 1.0
DATA_STOP\n";

    #[test]
    fn the_keywords_the_standard_makes_mandatory_are_required() {
        parse_kvn(CONFORMING).expect("a conforming message parses");

        // Table 3-2 marks CREATION_DATE and ORIGINATOR mandatory, table 3-3
        // marks TIME_SYSTEM mandatory and PARTICIPANT_n mandatory with "at
        // least one", and 3.1.3 gives each data section "a minimum of one
        // Tracking Data Record".
        for (dropped, expected) in [
            (
                "CREATION_DATE = 2005-160T20:15:00Z\n",
                TdmError::MissingKeyword {
                    keyword: "CREATION_DATE".to_string(),
                    segment: None,
                },
            ),
            (
                "ORIGINATOR = NASA\n",
                TdmError::MissingKeyword {
                    keyword: "ORIGINATOR".to_string(),
                    segment: None,
                },
            ),
            (
                "TIME_SYSTEM = UTC\n",
                TdmError::MissingKeyword {
                    keyword: "TIME_SYSTEM".to_string(),
                    segment: Some(1),
                },
            ),
            (
                "PARTICIPANT_1 = DSS-25\n",
                TdmError::MissingKeyword {
                    keyword: "PARTICIPANT_n".to_string(),
                    segment: Some(1),
                },
            ),
            (
                "RANGE = 2005-159T17:41:00 1.0\n",
                TdmError::EmptyDataSection { segment: 1 },
            ),
        ] {
            let without = CONFORMING.replace(dropped, "");
            assert_eq!(parse_kvn(&without), Err(expected), "dropping {dropped:?}");
        }
    }

    #[test]
    fn an_empty_value_is_refused() {
        // 4.3.1: "A non-empty value field must be specified for each keyword
        // provided." An empty ORIGINATOR was read as absent and dropped on
        // write, so a message lost the keyword it declared.
        let text = CONFORMING.replace("ORIGINATOR = NASA", "ORIGINATOR =");
        assert_eq!(
            parse_kvn(&text),
            Err(TdmError::EmptyValue {
                line: Some(3),
                keyword: "ORIGINATOR".to_string(),
            })
        );
    }

    #[test]
    fn a_version_outside_the_x_y_form_is_refused() {
        // 3.2.5: "the value shall have the form of x.y where y is incremented
        // for corrections and minor changes, and x is incremented for major
        // changes."
        for value in ["2", "2.0.0", "x.y", "2.", "v2.0"] {
            let text =
                CONFORMING.replace("CCSDS_TDM_VERS = 2.0", &format!("CCSDS_TDM_VERS = {value}"));
            assert_eq!(
                parse_kvn(&text),
                Err(TdmError::InvalidVersion {
                    line: Some(1),
                    value: value.to_string(),
                }),
                "version {value:?}"
            );
        }

        // Table 3-2 gives 0.12 for testing, 1.0 for the 2007 version and 2.0
        // for this one.
        for value in ["0.12", "1.0", "2.0"] {
            let text =
                CONFORMING.replace("CCSDS_TDM_VERS = 2.0", &format!("CCSDS_TDM_VERS = {value}"));
            let tdm = parse_kvn(&text).unwrap_or_else(|err| panic!("version {value:?}: {err}"));
            assert_eq!(tdm.version, value);
        }
    }

    /// `CONFORMING` with its single data record replaced by `lines`.
    fn records(lines: &str) -> String {
        CONFORMING.replace("RANGE = 2005-159T17:41:00 1.0", lines)
    }

    #[test]
    fn a_timetag_outside_the_two_forms_is_refused() {
        // 4.3.9 gives `YYYY-MM-DDThh:mm:ss[.d->d][Z]` and
        // `YYYY-DDDThh:mm:ss[.d->d][Z]`, with leading zeros throughout.
        for bad in [
            "05-159T17:41:00",
            "2005-159T17:41",
            "2005-13-01T00:00:00",
            "2005-02-30T00:00:00",
            "2005-366T00:00:00",
            "2005-000T00:00:00",
            "2005-159T24:00:00",
            "2005-159T17:60:00",
            "2005-159T17:41:00.",
        ] {
            let text = records(&format!("RANGE = {bad} 1.0"));
            assert_eq!(
                parse_kvn(&text),
                Err(TdmError::MalformedEpoch {
                    line: Some(9),
                    keyword: "RANGE".to_string(),
                    text: bad.to_string(),
                }),
                "{bad}"
            );
        }

        for good in [
            "2005-06-08T17:41:00",
            "2005-159T17:41:00",
            "2005-159T17:41:00.25",
            "2005-159T17:41:00Z",
            "2004-366T00:00:00",
            "2005-159T17:41:60",
        ] {
            let text = records(&format!("RANGE = {good} 1.0"));
            parse_kvn(&text).unwrap_or_else(|err| panic!("{good}: {err}"));
        }
    }

    #[test]
    fn records_out_of_order_or_repeated_follow_the_policy() {
        // 3.4.10: "the data for any given keyword shall be in chronological
        // order". 3.4.11: "Each keyword/timetag combination must be unique".
        let backwards = records("RANGE = 2005-159T17:41:01 1.0\nRANGE = 2005-159T17:41:00 2.0");
        assert_eq!(
            parse_kvn(&backwards),
            Err(TdmError::RecordsOutOfOrder {
                segment: 1,
                keyword: "RANGE".to_string(),
                epoch: "2005-159T17:41:00".to_string(),
            })
        );
        let (tdm, warnings) = parse_kvn_with_policy(
            &backwards,
            TdmPolicy::strict().with_record_order(TdmLeniency::Forgive),
        )
        .expect("a forgiven order reads");
        assert_eq!(warnings.len(), 1);
        assert_eq!(tdm.segments[0].data.records.len(), 2);

        let repeated = records("RANGE = 2005-159T17:41:00 1.0\nRANGE = 2005-159T17:41:00 2.0");
        assert_eq!(
            parse_kvn(&repeated),
            Err(TdmError::DuplicateRecord {
                segment: 1,
                keyword: "RANGE".to_string(),
                epoch: "2005-159T17:41:00".to_string(),
            })
        );
        let (tdm, warnings) = parse_kvn_with_policy(
            &repeated,
            TdmPolicy::strict().with_duplicate_records(TdmLeniency::Forgive),
        )
        .expect("a forgiven repeat reads");
        assert_eq!(
            warnings,
            vec![TdmWarning::DuplicateRecord {
                segment: 1,
                keyword: "RANGE".to_string(),
                epoch: "2005-159T17:41:00".to_string(),
            }]
        );
        // Both records are kept, in the order the file gave them: a data
        // section is a sequence, so nothing has to be chosen between them.
        assert_eq!(tdm.segments[0].data.records.len(), 2);
        assert_eq!(tdm.segments[0].data.records[0].value.text, "1.0");
        assert_eq!(tdm.segments[0].data.records[1].value.text, "2.0");

        // The writer stays strict whatever the reader forgave.
        assert_eq!(
            encode_kvn(&tdm),
            Err(TdmError::DuplicateRecord {
                segment: 1,
                keyword: "RANGE".to_string(),
                epoch: "2005-159T17:41:00".to_string(),
            })
        );

        // The rule is per keyword, so two keywords may share a timetag.
        let paired = records("RANGE = 2005-159T17:41:00 1.0\nANGLE_1 = 2005-159T17:41:00 10.0");
        parse_kvn(&paired).expect("two keywords may share a timetag");
    }

    #[test]
    fn the_two_timetag_forms_order_against_each_other() {
        // Day 159 of 2005 is 2005-06-08, so a message may write either form and
        // still be ordered.
        let forward = records("RANGE = 2005-159T17:41:00 1.0\nRANGE = 2005-06-08T18:41:00 2.0");
        parse_kvn(&forward).expect("the two forms compare");

        let backward = records("RANGE = 2005-06-08T18:41:00 1.0\nRANGE = 2005-159T17:41:00 2.0");
        assert!(matches!(
            parse_kvn(&backward),
            Err(TdmError::RecordsOutOfOrder { .. })
        ));
    }

    #[test]
    fn a_keyword_the_tables_do_not_define_is_refused() {
        // 3.2.3: "Only those keywords shown in table 3-2 shall be used in a TDM
        // Header." 3.3.1.7 says the same of table 3-3 and a metadata section.
        let header =
            CONFORMING.replace("ORIGINATOR = NASA", "ORIGINATOR = NASA\nWRONG_KEYWORD = 1");
        assert_eq!(
            parse_kvn(&header),
            Err(TdmError::UndefinedKeyword {
                line: 4,
                keyword: "WRONG_KEYWORD".to_string(),
                section: "header",
            })
        );

        let metadata =
            CONFORMING.replace("TIME_SYSTEM = UTC", "TIME_SYSTEM = UTC\nWRONG_KEYWORD = 1");
        assert_eq!(
            parse_kvn(&metadata),
            Err(TdmError::UndefinedKeyword {
                line: 6,
                keyword: "WRONG_KEYWORD".to_string(),
                section: "metadata",
            })
        );

        // A public Artemis 1 file writes SYSTEM TIME where TIME_SYSTEM belongs.
        // That is an undefined keyword, reported where it is, rather than only
        // the absence it leaves behind.
        let swapped = CONFORMING.replace("TIME_SYSTEM = UTC", "SYSTEM TIME = UTC");
        assert_eq!(
            parse_kvn(&swapped),
            Err(TdmError::UndefinedKeyword {
                line: 5,
                keyword: "SYSTEM TIME".to_string(),
                section: "metadata",
            })
        );
    }

    #[test]
    fn a_keyword_that_cannot_carry_a_value_is_refused_both_ways() {
        // 4.2.5 c) excepts COMMENT, META_START, META_STOP, DATA_START and
        // DATA_STOP from the KVN syntax, so none of them is an assignment key.
        for keyword in ["META_START", "META_STOP", "DATA_START", "DATA_STOP"] {
            let text = CONFORMING.replace(
                "TIME_SYSTEM = UTC",
                &format!("TIME_SYSTEM = UTC\n{keyword} = 1"),
            );
            assert_eq!(
                parse_kvn(&text),
                Err(TdmError::MalformedLine {
                    line: 6,
                    text: format!("{keyword} = 1"),
                }),
                "{keyword}"
            );
        }

        // The writer refuses the same set, since a caller can hold it.
        let mut tdm = parse_kvn(CONFORMING).unwrap();
        tdm.segments[0].metadata.fields.push(TdmField {
            key: "DATA_START".to_string(),
            value: "1".to_string(),
        });
        assert_eq!(
            encode_kvn(&tdm),
            Err(TdmError::KeywordNotAssignable {
                keyword: "DATA_START".to_string(),
            })
        );
    }

    #[test]
    fn the_bare_ephemeris_name_the_standard_writes_is_read() {
        // Table 3-3 lists EPHEMERIS_NAME_n only, but figure E-17 writes the
        // keyword bare and the annex I summary sheet lists it bare five times.
        let text = CONFORMING.replace(
            "PARTICIPANT_1 = DSS-25",
            "PARTICIPANT_1 = DSS-25\nEPHEMERIS_NAME = 3203_2013-11-09T23-02-30",
        );
        let tdm = parse_kvn(&text).expect("the bare keyword the standard writes is read");
        assert_eq!(
            tdm.segments[0].metadata.get_last("EPHEMERIS_NAME"),
            Some("3203_2013-11-09T23-02-30")
        );
    }

    #[test]
    fn an_indexed_keyword_outside_its_table_range_is_refused() {
        // Table 3-3 indexes PARTICIPANT_n with n = {1,2,3,4,5}, and 3.3.1.11
        // caps a segment at five participants. A padded suffix is not one of
        // the keywords the table defines.
        for keyword in ["PARTICIPANT_0", "PARTICIPANT_6", "PARTICIPANT_01"] {
            let text = CONFORMING.replace("PARTICIPANT_1 = DSS-25", &format!("{keyword} = DSS-25"));
            assert_eq!(
                parse_kvn(&text),
                Err(TdmError::InvalidField {
                    keyword: keyword.to_string(),
                    kind: TdmInputErrorKind::InvalidIndex,
                }),
                "{keyword}"
            );
        }

        // Table 3-3 defines PATH, PATH_1 and PATH_2 and no other index.
        for keyword in ["PATH_0", "PATH_3", "PATH_02"] {
            let text = CONFORMING.replace(
                "PARTICIPANT_1 = DSS-25",
                &format!("PARTICIPANT_1 = DSS-25\n{keyword} = 1"),
            );
            assert_eq!(
                parse_kvn(&text),
                Err(TdmError::InvalidField {
                    keyword: keyword.to_string(),
                    kind: TdmInputErrorKind::InvalidIndex,
                }),
                "{keyword}"
            );
        }
    }

    #[test]
    fn the_refused_states_read_back_as_something_other_than_they_were() {
        // Why each state below is refused, shown on the line the writer would
        // have emitted rather than asserted as a rule: parse_assignment takes
        // that line back as a different key and value.
        for (key, value, reads_back_as) in [
            ("A=B", "x", ("A", "B = x")),
            (" A", "x", ("A", "x")),
            ("A", "x ", ("A", "x")),
        ] {
            let line = format!("{key} = {value}");
            let (read_key, read_value) =
                parse_assignment(line.trim()).expect("the emitted line parses");
            assert_eq!(
                (read_key.as_str(), read_value.as_str()),
                reads_back_as,
                "{line:?}"
            );
            assert_ne!(
                (read_key.as_str(), read_value.as_str()),
                (key, value),
                "{line:?} must not read back as the field it came from"
            );
        }
    }

    #[test]
    fn a_value_the_kvn_form_cannot_carry_is_refused_rather_than_written() {
        // States parse_kvn cannot produce but a caller can build directly.
        let base = parse_kvn(CONFORMING).unwrap();
        for (key, value, fragment) in [
            ("", "x", "empty"),
            ("A=B", "x", "equals sign"),
            (" TIME_SYSTEM", "UTC", "around the keyword"),
            ("TIME_SYSTEM ", "UTC", "around the keyword"),
            ("TIME_SYSTEM", " UTC", "around the value"),
            ("TIME_SYSTEM", "UTC ", "around the value"),
            ("TIME\nSYSTEM", "UTC", "line terminator"),
            ("TIME_SYSTEM", "U\nTC", "line terminator"),
        ] {
            let mut tdm = base.clone();
            tdm.segments[0].metadata.fields.push(TdmField {
                key: key.to_string(),
                value: value.to_string(),
            });
            match encode_kvn(&tdm) {
                Err(TdmError::Unwritable { reason, .. }) => assert!(
                    reason.contains(fragment),
                    "{key:?} = {value:?} refused for {reason:?}, wanted {fragment:?}"
                ),
                other => panic!("{key:?} = {value:?} must be refused, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_comment_the_kvn_form_cannot_carry_is_refused() {
        let base = parse_kvn(CONFORMING).unwrap();
        for (text, fragment) in [
            (" padded", "around the text"),
            ("padded ", "around the text"),
            ("two\nlines", "line terminator"),
        ] {
            let mut tdm = base.clone();
            tdm.comments.push(text.to_string());
            match encode_kvn(&tdm) {
                Err(TdmError::Unwritable { keyword, reason }) => {
                    assert_eq!(keyword, COMMENT_KEY);
                    assert!(reason.contains(fragment), "{text:?}: {reason}");
                }
                other => panic!("{text:?} must be refused, got {other:?}"),
            }
        }

        // A comment the form carries writes and reads back as itself.
        let mut tdm = base;
        tdm.comments.push("plain text".to_string());
        let encoded = encode_kvn(&tdm).expect("a plain comment writes");
        assert_eq!(parse_kvn(&encoded).unwrap().comments, tdm.comments);
    }

    #[test]
    fn the_writer_checks_keyword_membership_as_the_reader_does() {
        // The mandatory-keyword check looked for a PARTICIPANT_ prefix, so a
        // caller-built PARTICIPANT_9 satisfied it and was written out.
        let mut tdm = parse_kvn(CONFORMING).unwrap();
        tdm.segments[0].metadata.fields.push(TdmField {
            key: "PARTICIPANT_9".to_string(),
            value: "DSS-99".to_string(),
        });
        assert!(matches!(
            encode_kvn(&tdm),
            Err(TdmError::InvalidField {
                kind: TdmInputErrorKind::InvalidIndex,
                ..
            })
        ));

        let mut tdm = parse_kvn(CONFORMING).unwrap();
        tdm.header_fields.push(TdmField {
            key: "NOT_A_HEADER_KEYWORD".to_string(),
            value: "x".to_string(),
        });
        assert!(matches!(encode_kvn(&tdm), Err(TdmError::Unwritable { .. })));
    }

    #[test]
    fn a_final_line_with_no_terminator_follows_the_policy() {
        // 4.2.11 terminates every TDM line, the last one included. Thirteen
        // of the 53 public files gathered for this audit end without one, and
        // a file's last record reads the same either way, so it is forgivable.
        let unterminated = CONFORMING.trim_end_matches('\n');
        assert_eq!(
            parse_kvn(unterminated),
            Err(TdmError::UnterminatedFinalLine { line: 10 })
        );

        let (tdm, warnings) = parse_kvn_with_policy(
            unterminated,
            TdmPolicy::strict().with_final_terminator(TdmLeniency::Forgive),
        )
        .expect("a forgiven final line reads");
        assert_eq!(
            warnings,
            vec![TdmWarning::UnterminatedFinalLine { line: 10 }]
        );

        // The writer terminates its last line, so a forgiven message writes
        // back conforming rather than unterminated as it arrived.
        let encoded = encode_kvn(&tdm).expect("the message writes");
        assert!(encoded.ends_with('\n'));
        assert_eq!(parse_kvn(&encoded).unwrap(), tdm);

        // Every failure about what the message says is reported ahead of the
        // missing terminator. An unclosed block is unterminated by
        // construction, and a file with no version has a defect its author can
        // act on, which "the last line carries no terminator" is not.
        assert_eq!(
            parse_kvn("CCSDS_TDM_VERS = 2.0\nMETA_START"),
            Err(TdmError::Section {
                line: 3,
                detail: "unclosed metadata block",
            })
        );
        assert_eq!(
            parse_kvn("CREATION_DATE = 2005-160T20:15:00Z"),
            Err(TdmError::MissingKeyword {
                keyword: VERSION_KEY.to_string(),
                segment: None,
            })
        );
        assert_eq!(
            parse_kvn("CCSDS_TDM_VERS = 2.0\nCREATION_DATE = 2005-160T20:15:00Z"),
            Err(TdmError::MissingKeyword {
                keyword: "ORIGINATOR".to_string(),
                segment: None,
            })
        );

        // A character-set or line-length defect keeps its place ahead of them:
        // each names a position in a line and explains how the rest of that
        // line reads.
        assert_eq!(
            parse_kvn("CREATION\u{a0}_DATE = 2005-160T20:15:00Z"),
            Err(TdmError::NonPrintableCharacter {
                // U+00A0 is whitespace, so the token before it is all the
                // keyword there is to name.
                line: Some(1),
                keyword: "CREATION".to_string(),
                column: 9,
                character: '\u{a0}',
            })
        );
    }

    #[test]
    fn a_data_comment_keeps_its_place_through_a_round_trip() {
        // A comment at the start of the block is where 4.5.2 c) puts one.
        let at_start = records("COMMENT before any record\nRANGE = 2005-159T17:41:00 1.0");
        let tdm = parse_kvn(&at_start).expect("a comment at the start is in place");
        assert_eq!(
            tdm.segments[0].data.comments,
            vec![TdmComment {
                text: "before any record".to_string(),
                before_record: 0,
            }]
        );

        // 4.5.2 c) puts a data-section comment "between the 'DATA_START'
        // keyword and the first Tracking Data Record", so one after a record is
        // out of place and strict names it.
        let late = records(
            "RANGE = 2005-159T17:41:00 1.0\nCOMMENT after the first record\nRANGE = 2005-159T17:41:01 2.0",
        );
        assert_eq!(
            parse_kvn(&late),
            Err(TdmError::KeywordOutOfOrder {
                line: Some(10),
                keyword: COMMENT_KEY.to_string(),
                section: "data",
            })
        );

        // Forgiven, it is read where it sits and written back there. Nothing
        // moves: the message round-trips to the bytes it came from, rather than
        // returning with its comments gathered to the top of the block.
        let (tdm, warnings) = parse_kvn_with_policy(
            &late,
            TdmPolicy::strict().with_keyword_order(TdmLeniency::Forgive),
        )
        .expect("a forgiven comment reads");
        assert_eq!(
            warnings,
            vec![TdmWarning::KeywordOutOfOrder {
                line: 10,
                keyword: COMMENT_KEY.to_string(),
                section: "data",
            }]
        );
        assert_eq!(
            tdm.segments[0].data.comments,
            vec![TdmComment {
                text: "after the first record".to_string(),
                before_record: 1,
            }]
        );
        let (encoded, departures) = encode_kvn_with_policy(
            &tdm,
            TdmWritePolicy::strict().with_keyword_order(TdmLeniency::Forgive),
        )
        .expect("the message writes under a forgiving policy");
        assert_eq!(
            departures,
            vec![TdmDeparture::KeywordOutOfOrder {
                keyword: COMMENT_KEY.to_string(),
                section: "data",
            }]
        );
        assert_eq!(encoded, late);
    }

    #[test]
    fn a_keyword_out_of_the_order_its_table_fixes_follows_the_policy() {
        // 3.2.3: "The order of occurrence of the mandatory and optional KVN
        // assignments shall be fixed as shown in table 3-2", which puts COMMENT
        // between CCSDS_TDM_VERS and CREATION_DATE, as 4.5.2 a) also says.
        let late_comment = CONFORMING.replace(
            "ORIGINATOR = NASA",
            "ORIGINATOR = NASA\nCOMMENT written after the originator",
        );
        assert_eq!(
            parse_kvn(&late_comment),
            Err(TdmError::KeywordOutOfOrder {
                line: Some(4),
                keyword: COMMENT_KEY.to_string(),
                section: "header",
            })
        );
        let (tdm, warnings) = parse_kvn_with_policy(
            &late_comment,
            TdmPolicy::strict().with_keyword_order(TdmLeniency::Forgive),
        )
        .expect("a forgiven order reads");
        assert_eq!(
            warnings,
            vec![TdmWarning::KeywordOutOfOrder {
                line: 4,
                keyword: COMMENT_KEY.to_string(),
                section: "header",
            }]
        );
        // Forgiving the order keeps the comment; only its position was wrong.
        assert_eq!(
            tdm.comments,
            vec!["written after the originator".to_string()]
        );

        // 3.3.1.8 fixes the metadata order the same way, and table 3-3 puts
        // TIME_SYSTEM before PARTICIPANT_n.
        let swapped = CONFORMING.replace(
            "TIME_SYSTEM = UTC\nPARTICIPANT_1 = DSS-25",
            "PARTICIPANT_1 = DSS-25\nTIME_SYSTEM = UTC",
        );
        assert_eq!(
            parse_kvn(&swapped),
            Err(TdmError::KeywordOutOfOrder {
                line: Some(6),
                keyword: "TIME_SYSTEM".to_string(),
                section: "metadata",
            })
        );

        // An indexed keyword ranks where its base does, so two participants sit
        // in order beside each other whichever index comes first.
        let descending = CONFORMING.replace(
            "PARTICIPANT_1 = DSS-25",
            "PARTICIPANT_2 = yyyy-nnnA\nPARTICIPANT_1 = DSS-25",
        );
        parse_kvn(&descending).expect("indexed keywords rank together");
    }

    #[test]
    fn a_path_naming_a_participant_the_segment_lacks_is_refused() {
        // 3.3.1.9 requires participant indices to differ, not to run
        // consecutively, so a gap nothing points into is legal.
        let gap = CONFORMING.replace(
            "PARTICIPANT_1 = DSS-25",
            "PARTICIPANT_1 = DSS-25\nPARTICIPANT_3 = yyyy-nnnA",
        );
        parse_kvn(&gap).expect("a gap no PATH points into is legal");

        // A PATH naming an index the segment does not define is not legal: the
        // measurements it describes would belong to nobody.
        let dangling = CONFORMING.replace(
            "PARTICIPANT_1 = DSS-25",
            "PARTICIPANT_1 = DSS-25\nPATH = 1,2",
        );
        assert_eq!(
            parse_kvn(&dangling),
            Err(TdmError::UndefinedParticipant {
                segment: 1,
                keyword: "PATH".to_string(),
                index: 2,
            })
        );

        // Refused under every policy: picking which participant was meant
        // would be inventing one.
        assert!(matches!(
            parse_kvn_with_policy(&dangling, TdmPolicy::lenient()).map(|(tdm, _)| tdm),
            Err(TdmError::UndefinedParticipant { .. })
        ));

        // A path across a gap resolves when both ends are defined.
        let over_gap = CONFORMING.replace(
            "PARTICIPANT_1 = DSS-25",
            "PARTICIPANT_1 = DSS-25\nPARTICIPANT_3 = yyyy-nnnA\nPATH = 1,3,1",
        );
        parse_kvn(&over_gap).expect("a path across a gap resolves");
    }

    #[test]
    fn two_participants_sharing_an_index_are_refused() {
        // 3.3.1.9: "The indexer shall not be the same for any two participants
        // in a given Metadata Section." Both were kept, and the second silently
        // shadowed the first everywhere an index is resolved. Two participants
        // sharing an index means PARTICIPANT_1 twice with different values, so
        // it is refused as a conflicting keyword naming the line and both values.
        let text = CONFORMING.replace(
            "PARTICIPANT_1 = DSS-25",
            "PARTICIPANT_1 = DSS-25\nPARTICIPANT_1 = DSS-34",
        );
        assert_eq!(
            parse_kvn(&text),
            Err(TdmError::ConflictingKeyword {
                line: Some(7),
                keyword: "PARTICIPANT_1".to_string(),
                section: "metadata",
                first: "DSS-25".to_string(),
                second: "DSS-34".to_string(),
            })
        );

        // An identical repeat of a participant keyword does not produce a
        // phantom second participant downstream.
        let identical = CONFORMING.replace(
            "PARTICIPANT_1 = DSS-25",
            "PARTICIPANT_1 = DSS-25\nPARTICIPANT_1 = DSS-25",
        );
        let (tdm, warnings) = parse_kvn_with_policy(&identical, TdmPolicy::strict())
            .expect("an identical repeat reads under strict policy");
        assert_eq!(tdm.segments[0].metadata.participants.len(), 1);
        assert_eq!(
            warnings,
            vec![TdmWarning::RepeatedKeyword {
                line: 7,
                keyword: "PARTICIPANT_1".to_string(),
                section: "metadata",
            }]
        );

        // Two different indices are what the standard expects.
        let text = CONFORMING.replace(
            "PARTICIPANT_1 = DSS-25",
            "PARTICIPANT_1 = DSS-25\nPARTICIPANT_2 = yyyy-nnnA",
        );
        let tdm = parse_kvn(&text).expect("two indices parse");
        assert_eq!(tdm.segments[0].metadata.participants.len(), 2);
    }

    #[test]
    fn a_forgiven_departure_is_read_and_reported() {
        // Each forgivable departure twice: the strict read refuses it by name,
        // the lenient read takes the message and reports one warning.
        let tab = CONFORMING.replace("TIME_SYSTEM = UTC", "\tTIME_SYSTEM = UTC");
        assert_eq!(
            parse_kvn(&tab),
            Err(TdmError::NonPrintableCharacter {
                line: Some(5),
                keyword: "TIME_SYSTEM".to_string(),
                column: 1,
                character: '\t',
            })
        );
        let (tdm, warnings) = parse_kvn_with_policy(
            &tab,
            TdmPolicy::default().with_non_printable(TdmLeniency::Forgive),
        )
        .expect("a forgiven tab reads");
        assert_eq!(
            warnings,
            vec![TdmWarning::NonPrintableCharacter {
                line: 5,
                keyword: "TIME_SYSTEM".to_string(),
                column: 1,
                character: '\t',
            }]
        );
        assert_eq!(tdm.segments[0].metadata.time_system.as_deref(), Some("UTC"));

        let missing = CONFORMING.replace("ORIGINATOR = NASA\n", "");
        assert_eq!(
            parse_kvn(&missing),
            Err(TdmError::MissingKeyword {
                keyword: "ORIGINATOR".to_string(),
                segment: None,
            })
        );
        let (tdm, warnings) = parse_kvn_with_policy(
            &missing,
            TdmPolicy::default().with_missing_keywords(TdmLeniency::Forgive),
        )
        .expect("a forgiven absent ORIGINATOR reads");
        assert_eq!(
            warnings,
            vec![TdmWarning::MissingKeyword {
                keyword: "ORIGINATOR".to_string(),
                segment: None,
            }]
        );
        assert_eq!(tdm.originator, None);

        let long_value = "A".repeat(MAX_LINE_CHARACTERS);
        let long = CONFORMING.replace("TIME_SYSTEM = UTC", &format!("TIME_SYSTEM = {long_value}"));
        assert_eq!(
            parse_kvn(&long),
            Err(TdmError::LineTooLong {
                line: Some(5),
                keyword: "TIME_SYSTEM".to_string(),
                length: "TIME_SYSTEM = ".len() + MAX_LINE_CHARACTERS,
            })
        );
        let (_, warnings) = parse_kvn_with_policy(
            &long,
            TdmPolicy::default().with_long_lines(TdmLeniency::Forgive),
        )
        .expect("a forgiven long line reads");
        assert_eq!(
            warnings,
            vec![TdmWarning::LineTooLong {
                line: 5,
                keyword: "TIME_SYSTEM".to_string(),
                length: "TIME_SYSTEM = ".len() + MAX_LINE_CHARACTERS,
            }]
        );
    }

    #[test]
    fn an_empty_data_section_is_forgivable_and_still_unwritable() {
        // 3.1.3 gives a data section "a minimum of one Tracking Data Record".
        // Reading on invents nothing: the segment carries no measurements, and
        // a caller sees that in the records it gets. The fuzz payload that
        // started this work held six such segments.
        let empty = CONFORMING.replace("RANGE = 2005-159T17:41:00 1.0\n", "");
        assert_eq!(
            parse_kvn(&empty),
            Err(TdmError::EmptyDataSection { segment: 1 })
        );

        let (tdm, warnings) = parse_kvn_with_policy(
            &empty,
            TdmPolicy::default().with_empty_data_sections(TdmLeniency::Forgive),
        )
        .expect("a forgiven empty data section reads");
        assert_eq!(warnings, vec![TdmWarning::EmptyDataSection { segment: 1 }]);
        assert_eq!(tdm.segments.len(), 1);
        assert!(tdm.segments[0].data.records.is_empty());

        // The writer stays strict, so the forgiven message is refused by name
        // rather than written back missing what 3.1.3 requires.
        assert_eq!(
            encode_kvn(&tdm),
            Err(TdmError::EmptyDataSection { segment: 1 })
        );
    }

    #[test]
    fn a_leniently_read_message_writes_unchanged_or_is_refused_by_name() {
        // The writer is strict whatever the reader forgave. A forgiven message
        // either writes back as it was read, or is refused naming what the
        // standard forbids; it is never quietly repaired.
        let missing = CONFORMING.replace("ORIGINATOR = NASA\n", "");
        let (tdm, _) = parse_kvn_with_policy(
            &missing,
            TdmPolicy::default().with_missing_keywords(TdmLeniency::Forgive),
        )
        .unwrap();
        assert_eq!(
            encode_kvn(&tdm),
            Err(TdmError::MissingKeyword {
                keyword: "ORIGINATOR".to_string(),
                segment: None,
            })
        );

        // A forgiven tab here is whitespace before the keyword, which 4.2.7
        // makes insignificant, so this message does write back conforming.
        let tab = CONFORMING.replace("TIME_SYSTEM = UTC", "\tTIME_SYSTEM = UTC");
        let (tdm, _) = parse_kvn_with_policy(
            &tab,
            TdmPolicy::default().with_non_printable(TdmLeniency::Forgive),
        )
        .unwrap();
        let encoded = encode_kvn(&tdm).expect("insignificant whitespace leaves a writable value");
        assert_eq!(parse_kvn(&encoded).unwrap(), tdm);
    }

    #[test]
    fn the_default_policy_forgives_nothing() {
        assert_eq!(TdmPolicy::default().non_printable, TdmLeniency::Strict);
        assert_eq!(TdmPolicy::default().missing_keywords, TdmLeniency::Strict);
        assert_eq!(TdmPolicy::default().long_lines, TdmLeniency::Strict);
        assert_eq!(
            TdmPolicy::default().empty_data_sections,
            TdmLeniency::Strict
        );
        assert_eq!(TdmPolicy::default().record_order, TdmLeniency::Strict);
        assert_eq!(TdmPolicy::default().duplicate_records, TdmLeniency::Strict);
        // The constant form and the derived default agree.
        assert_eq!(TdmPolicy::default(), TdmPolicy::strict());

        let (tdm, warnings) =
            parse_kvn_with_policy(CONFORMING, TdmPolicy::default()).expect("conforming parses");
        assert!(warnings.is_empty());
        assert_eq!(tdm, parse_kvn(CONFORMING).unwrap());
    }

    #[test]
    fn encoding_refuses_a_caller_built_empty_value() {
        let mut tdm = parse_kvn(CONFORMING).unwrap();
        tdm.segments[0].metadata.fields.push(TdmField {
            key: "RANGE_MODE".to_string(),
            value: String::new(),
        });
        assert_eq!(
            encode_kvn(&tdm),
            Err(TdmError::EmptyValue {
                line: None,
                keyword: "RANGE_MODE".to_string(),
            })
        );
    }

    #[test]
    fn writer_mirror_refuses_or_forgives_keyword_order() {
        // Tables 3-2 and 3-3 fix the order of occurrence of KVN assignments.
        // Reversing metadata fields puts PARTICIPANT_1 before TIME_SYSTEM.
        let mut tdm = parse_kvn(CONFORMING).unwrap();
        tdm.segments[0].metadata.fields.reverse();
        assert_eq!(
            encode_kvn_with_policy(&tdm, TdmWritePolicy::strict()),
            Err(TdmError::KeywordOutOfOrder {
                line: None,
                keyword: "TIME_SYSTEM".to_string(),
                section: "metadata",
            })
        );
        let (encoded, departures) = encode_kvn_with_policy(
            &tdm,
            TdmWritePolicy::strict().with_keyword_order(TdmLeniency::Forgive),
        )
        .expect("keyword order is forgivable on write");
        assert_eq!(
            departures,
            vec![TdmDeparture::KeywordOutOfOrder {
                keyword: "TIME_SYSTEM".to_string(),
                section: "metadata",
            }]
        );
        assert!(encoded.contains("PARTICIPANT_1 = DSS-25\nTIME_SYSTEM = UTC"));
    }

    #[test]
    fn writer_mirror_refuses_or_forgives_record_chronological_order() {
        // CCSDS 503.0-B-2 3.4.10 requires records to be in chronological order.
        let mut tdm = parse_kvn(CONFORMING).unwrap();
        let mut earlier = tdm.segments[0].data.records[0].clone();
        earlier.epoch = "2005-159T17:40:00".to_string();
        tdm.segments[0].data.records.push(earlier);
        assert_eq!(
            encode_kvn_with_policy(&tdm, TdmWritePolicy::strict()),
            Err(TdmError::RecordsOutOfOrder {
                segment: 1,
                keyword: "RANGE".to_string(),
                epoch: "2005-159T17:40:00".to_string(),
            })
        );
        let (encoded, departures) = encode_kvn_with_policy(
            &tdm,
            TdmWritePolicy::strict().with_record_order(TdmLeniency::Forgive),
        )
        .expect("record chronological order is forgivable on write");
        assert_eq!(
            departures,
            vec![TdmDeparture::RecordsOutOfOrder {
                segment: 1,
                keyword: "RANGE".to_string(),
                epoch: "2005-159T17:40:00".to_string(),
            }]
        );
        assert!(encoded.contains("2005-159T17:40:00"));
    }

    #[test]
    fn writer_mirror_refuses_or_forgives_duplicate_record() {
        // CCSDS 503.0-B-2 3.4.11 forbids duplicate tracking data records.
        let mut tdm = parse_kvn(CONFORMING).unwrap();
        let duplicate = tdm.segments[0].data.records[0].clone();
        tdm.segments[0].data.records.push(duplicate);
        assert_eq!(
            encode_kvn_with_policy(&tdm, TdmWritePolicy::strict()),
            Err(TdmError::DuplicateRecord {
                segment: 1,
                keyword: "RANGE".to_string(),
                epoch: "2005-159T17:41:00".to_string(),
            })
        );
        let (encoded, departures) = encode_kvn_with_policy(
            &tdm,
            TdmWritePolicy::strict().with_duplicate_records(TdmLeniency::Forgive),
        )
        .expect("duplicate record is forgivable on write");
        assert_eq!(
            departures,
            vec![TdmDeparture::DuplicateRecord {
                segment: 1,
                keyword: "RANGE".to_string(),
                epoch: "2005-159T17:41:00".to_string(),
            }]
        );
        assert!(encoded.contains("2005-159T17:41:00"));
    }

    #[test]
    fn writer_mirror_refuses_path_naming_undefined_participant_under_all_policies() {
        // A PATH naming a participant index the segment does not define leaves
        // the measurements belonging to nobody, which no policy forgives.
        let mut tdm = parse_kvn(CONFORMING).unwrap();
        tdm.segments[0].metadata.fields.push(TdmField {
            key: "PATH".to_string(),
            value: "1,2".to_string(),
        });
        assert_eq!(
            encode_kvn_with_policy(&tdm, TdmWritePolicy::strict()),
            Err(TdmError::UndefinedParticipant {
                segment: 1,
                keyword: "PATH".to_string(),
                index: 2,
            })
        );
        assert_eq!(
            encode_kvn_with_policy(&tdm, TdmWritePolicy::lenient()),
            Err(TdmError::UndefinedParticipant {
                segment: 1,
                keyword: "PATH".to_string(),
                index: 2,
            })
        );
    }

    #[test]
    fn writer_mirror_refuses_indexed_keyword_outside_range_under_all_policies() {
        // Table 3-3 indexes PARTICIPANT_n with n = {1,2,3,4,5}. Indices outside
        // that range are refused under all policies.
        let mut tdm = parse_kvn(CONFORMING).unwrap();
        tdm.segments[0].metadata.fields.push(TdmField {
            key: "PARTICIPANT_6".to_string(),
            value: "DSS-65".to_string(),
        });
        assert_eq!(
            encode_kvn_with_policy(&tdm, TdmWritePolicy::strict()),
            Err(TdmError::InvalidField {
                keyword: "PARTICIPANT_6".to_string(),
                kind: TdmInputErrorKind::InvalidIndex,
            })
        );
        assert_eq!(
            encode_kvn_with_policy(&tdm, TdmWritePolicy::lenient()),
            Err(TdmError::InvalidField {
                keyword: "PARTICIPANT_6".to_string(),
                kind: TdmInputErrorKind::InvalidIndex,
            })
        );
    }

    #[test]
    fn writer_mirror_refuses_or_forgives_missing_mandatory_keyword() {
        // Table 3-3 marks TIME_SYSTEM and PARTICIPANT_n mandatory.
        let mut tdm = parse_kvn(CONFORMING).unwrap();
        tdm.segments[0]
            .metadata
            .fields
            .retain(|field| field.key != "TIME_SYSTEM");
        assert_eq!(
            encode_kvn_with_policy(&tdm, TdmWritePolicy::strict()),
            Err(TdmError::MissingKeyword {
                keyword: "TIME_SYSTEM".to_string(),
                segment: Some(1),
            })
        );
        let (encoded, departures) = encode_kvn_with_policy(
            &tdm,
            TdmWritePolicy::strict().with_missing_keywords(TdmLeniency::Forgive),
        )
        .expect("missing mandatory keyword is forgivable on write");
        assert_eq!(
            departures,
            vec![TdmDeparture::MissingKeyword {
                keyword: "TIME_SYSTEM".to_string(),
                segment: Some(1),
            }]
        );
        assert!(!encoded.contains("TIME_SYSTEM"));

        let mut tdm_no_part = parse_kvn(CONFORMING).unwrap();
        tdm_no_part.segments[0]
            .metadata
            .fields
            .retain(|field| !field.key.starts_with("PARTICIPANT_"));
        assert_eq!(
            encode_kvn_with_policy(&tdm_no_part, TdmWritePolicy::strict()),
            Err(TdmError::MissingKeyword {
                keyword: "PARTICIPANT_n".to_string(),
                segment: Some(1),
            })
        );
        let (_, departures_part) = encode_kvn_with_policy(
            &tdm_no_part,
            TdmWritePolicy::strict().with_missing_keywords(TdmLeniency::Forgive),
        )
        .expect("missing participant is forgivable on write");
        assert_eq!(
            departures_part,
            vec![TdmDeparture::MissingKeyword {
                keyword: "PARTICIPANT_n".to_string(),
                segment: Some(1),
            }]
        );
    }

    #[test]
    fn writer_mirror_refuses_empty_mandatory_value_under_all_policies() {
        // CCSDS 503.0-B-2 4.3.1 requires a non-empty value for each keyword provided.
        let mut tdm = parse_kvn(CONFORMING).unwrap();
        for field in &mut tdm.segments[0].metadata.fields {
            if field.key == "TIME_SYSTEM" {
                field.value.clear();
            }
        }
        assert_eq!(
            encode_kvn_with_policy(&tdm, TdmWritePolicy::strict()),
            Err(TdmError::EmptyValue {
                line: None,
                keyword: "TIME_SYSTEM".to_string(),
            })
        );
        assert_eq!(
            encode_kvn_with_policy(&tdm, TdmWritePolicy::lenient()),
            Err(TdmError::EmptyValue {
                line: None,
                keyword: "TIME_SYSTEM".to_string(),
            })
        );
    }

    #[test]
    fn writer_mirror_refuses_malformed_epoch_under_all_policies() {
        // CCSDS 503.0-B-2 4.3.9 defines two valid timetag forms. A record with a
        // malformed epoch is refused under all policies.
        let mut tdm = parse_kvn(CONFORMING).unwrap();
        tdm.segments[0].data.records[0].epoch = "2005-159-17:41:00".to_string();
        assert_eq!(
            encode_kvn_with_policy(&tdm, TdmWritePolicy::strict()),
            Err(TdmError::MalformedEpoch {
                line: None,
                keyword: "RANGE".to_string(),
                text: "2005-159-17:41:00".to_string(),
            })
        );
        assert_eq!(
            encode_kvn_with_policy(&tdm, TdmWritePolicy::lenient()),
            Err(TdmError::MalformedEpoch {
                line: None,
                keyword: "RANGE".to_string(),
                text: "2005-159-17:41:00".to_string(),
            })
        );
    }

    #[test]
    fn writer_mirror_refuses_or_forgives_repeated_keyword_identical_value() {
        // 4.2.5 a) gives each keyword a single value assignment. Repeating a keyword
        // with the same value is refused under strict policy and reported as a departure
        // under repeated_keywords leniency.
        let mut tdm = parse_kvn(CONFORMING).unwrap();
        tdm.segments[0].metadata.fields.insert(
            1,
            TdmField {
                key: "TIME_SYSTEM".to_string(),
                value: "UTC".to_string(),
            },
        );
        assert_eq!(
            encode_kvn_with_policy(&tdm, TdmWritePolicy::strict()),
            Err(TdmError::RepeatedKeyword {
                line: None,
                keyword: "TIME_SYSTEM".to_string(),
                section: "metadata",
            })
        );
        let (encoded, departures) = encode_kvn_with_policy(
            &tdm,
            TdmWritePolicy::strict().with_repeated_keywords(TdmLeniency::Forgive),
        )
        .expect("identical repeated keyword is forgivable on write");
        assert_eq!(
            departures,
            vec![TdmDeparture::RepeatedKeyword {
                keyword: "TIME_SYSTEM".to_string(),
                section: "metadata",
            }]
        );
        assert_eq!(encoded.matches("TIME_SYSTEM = UTC").count(), 2);
    }

    #[test]
    fn writer_mirror_refuses_repeated_keyword_conflicting_values_under_all_policies() {
        // 4.2.5 a) gives each keyword a single value assignment. Repeating a keyword
        // with different values leaves the writer to guess which value was intended,
        // which no policy forgives.
        let mut tdm = parse_kvn(CONFORMING).unwrap();
        tdm.segments[0].metadata.fields.insert(
            1,
            TdmField {
                key: "TIME_SYSTEM".to_string(),
                value: "GPS".to_string(),
            },
        );
        assert_eq!(
            encode_kvn_with_policy(&tdm, TdmWritePolicy::strict()),
            Err(TdmError::ConflictingKeyword {
                line: None,
                keyword: "TIME_SYSTEM".to_string(),
                section: "metadata",
                first: "UTC".to_string(),
                second: "GPS".to_string(),
            })
        );
        assert_eq!(
            encode_kvn_with_policy(&tdm, TdmWritePolicy::lenient()),
            Err(TdmError::ConflictingKeyword {
                line: None,
                keyword: "TIME_SYSTEM".to_string(),
                section: "metadata",
                first: "UTC".to_string(),
                second: "GPS".to_string(),
            })
        );
    }
}
