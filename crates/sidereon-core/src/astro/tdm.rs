//! CCSDS Tracking Data Message KVN reader and writer.
//!
//! This module implements the CCSDS 503.0-B-2 KVN form as a sans-IO parser and
//! serializer. Date/time fields remain raw strings, matching the other NDM
//! readers in this crate. Observable values are kept as both the parsed `f64`
//! and the exact decimal token read from the message, so frequency-domain
//! records such as `RECEIVE_FREQ` and `TRANSMIT_FREQ_n` re-emit without decimal
//! rewriting.

use std::fmt;

const VERSION_KEY: &str = "CCSDS_TDM_VERS";
const COMMENT_KEY: &str = "COMMENT";
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

/// A TDM data block.
#[derive(Debug, Clone, PartialEq)]
pub struct TdmDataSection {
    /// Data-section comments in parse order.
    pub comments: Vec<String>,
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
        /// One-based character position within the line.
        column: usize,
        /// The offending character.
        character: char,
    },
    /// A line was longer than the 254 characters 4.2.1 allows.
    LineTooLong {
        /// One-based input line number.
        line: usize,
        /// The line's length in characters, excluding its terminator.
        length: usize,
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
                column,
                character,
            } => write!(
                f,
                "TDM line {line} column {column} holds {character:?}, which is not printable ASCII"
            ),
            Self::LineTooLong { line, length } => write!(
                f,
                "TDM line {line} is {length} characters, over the {MAX_LINE_CHARACTERS} allowed"
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
        }
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
    /// Two keywords in one metadata block carry the same index, which CCSDS
    /// 503.0-B-2 3.3.1.9 forbids for participants.
    DuplicateIndex {
        /// The keyword family, such as `PARTICIPANT`.
        keyword: String,
        /// The index both carry.
        index: u8,
        /// The one-based segment they are in.
        segment: usize,
    },
    /// A tracking data record's timetag is not one of the two forms CCSDS
    /// 503.0-B-2 4.3.9 defines.
    MalformedEpoch {
        /// One-based input line number.
        line: usize,
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
        /// One-based input line number.
        line: usize,
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
            Self::DuplicateIndex {
                keyword,
                index,
                segment,
            } => write!(
                f,
                "TDM segment {segment} gives {keyword}_{index} more than once"
            ),
            Self::MalformedEpoch {
                line,
                keyword,
                text,
            } => write!(
                f,
                "TDM record {keyword} at line {line} has the timetag {text}, which is not a form 4.3.9 defines"
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
    version: Option<String>,
    comments: Vec<String>,
    creation_date: Option<String>,
    originator: Option<String>,
    message_id: Option<String>,
    fields: Vec<TdmField>,
}

#[derive(Default)]
struct MetadataBuilder {
    comments: Vec<String>,
    fields: Vec<TdmField>,
}

#[derive(Default)]
struct DataBuilder {
    comments: Vec<String>,
    records: Vec<TdmDataRecord>,
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
    let mut header = HeaderBuilder::default();
    let mut metadata: Option<MetadataBuilder> = None;
    let mut pending_metadata: Option<TdmMetadata> = None;
    let mut data: Option<DataBuilder> = None;
    let mut segments = Vec::new();

    let source_lines = tdm_lines(text);
    let past_end = source_lines.len().saturating_add(1);

    for (idx, raw_line) in source_lines.iter().copied().enumerate() {
        let line_no = idx + 1;
        check_line(line_no, raw_line)?;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(comment) = comment_text(line) {
            if let Some(builder) = data.as_mut() {
                builder.comments.push(comment);
            } else if let Some(builder) = metadata.as_mut() {
                builder.comments.push(comment);
            } else if pending_metadata.is_none() {
                header.comments.push(comment);
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
                pending_metadata = Some(build_metadata(builder)?);
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
        if key == COMMENT_KEY {
            return Err(TdmError::MalformedLine {
                line: line_no,
                text: line.to_string(),
            });
        }

        if let Some(builder) = data.as_mut() {
            let range_units = pending_metadata
                .as_ref()
                .map(|metadata| metadata.range_units.clone())
                .unwrap_or(TdmUnit::Kilometers);
            builder
                .records
                .push(parse_record(line_no, &key, &value, &range_units)?);
        } else if let Some(builder) = metadata.as_mut() {
            builder.fields.push(TdmField { key, value });
        } else if pending_metadata.is_none() {
            parse_header_field(&mut header, key, value);
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

    let version =
        header
            .version
            .filter(|value| !value.is_empty())
            .ok_or(TdmError::MissingKeyword {
                keyword: VERSION_KEY.to_string(),
                segment: None,
            })?;
    if segments.is_empty() {
        return Err(TdmError::NoSegments);
    }

    // 4.2.11 terminates every TDM line, the last one included.
    //
    // It is the last check in the function, behind everything about what the
    // message says: the unclosed-block checks, CCSDS_TDM_VERS and the segment
    // count. Someone handed "missing TDM CCSDS_TDM_VERS" can fix the file;
    // handed "line 1 carries no terminator" about a file that is also missing
    // its version, they get the smaller of the two problems first.
    //
    // check_line keeps its place at the top of the read loop, ahead of all of
    // this. A character outside printable ASCII or a line past 254 characters
    // names a position in one line and explains how the rest of that line
    // reads: a keyword holding a non-breaking space reads as an undefined
    // keyword, and reporting the undefined keyword names the wrong problem.
    // Nothing downstream turns on whether the file's last line was terminated.
    if !text.is_empty() && !text.ends_with(['\r', '\n']) {
        return Err(TdmError::UnterminatedFinalLine {
            line: source_lines.len(),
        });
    }

    Ok(Tdm {
        version,
        comments: header.comments,
        creation_date: header.creation_date,
        originator: header.originator,
        message_id: header.message_id,
        header_fields: header.fields,
        segments,
    })
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
    validate_tdm(tdm)?;

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
        lines.extend(segment.data.comments.iter().map(comment_line));
        for record in &segment.data.records {
            lines.push(format!(
                "{} = {} {}",
                record.keyword, record.epoch, record.value.text
            ));
        }
        lines.push("DATA_STOP".to_string());
    }

    // 4.2.11 terminates every line, so the last one carries one too.
    lines.push(String::new());
    Ok(lines.join("\n"))
}

fn parse_header_field(header: &mut HeaderBuilder, key: String, value: String) {
    match key.as_str() {
        VERSION_KEY => header.version = Some(value),
        "CREATION_DATE" => header.creation_date = empty_to_none(value),
        "ORIGINATOR" => header.originator = empty_to_none(value),
        "MESSAGE_ID" => header.message_id = empty_to_none(value),
        _ => header.fields.push(TdmField { key, value }),
    }
}

fn build_metadata(builder: MetadataBuilder) -> Result<TdmMetadata, TdmError> {
    let mut participants = Vec::new();
    let mut mode = None;
    let mut paths = Vec::new();
    let mut timetag_ref = None;
    let mut time_system = None;
    let mut range_units = TdmUnit::Kilometers;

    for field in &builder.fields {
        if let Some(index) = indexed_suffix(&field.key, "PARTICIPANT")? {
            participants.push(TdmParticipant {
                index,
                name: field.value.clone(),
            });
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
        Some(indexed_suffix(&field.key, "PATH")?.ok_or_else(|| invalid_index(&field.key))?)
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

fn validate_tdm(tdm: &Tdm) -> Result<(), TdmError> {
    if tdm.version.is_empty() {
        return Err(TdmError::MissingKeyword {
            keyword: VERSION_KEY.to_string(),
            segment: None,
        });
    }
    if tdm.segments.is_empty() {
        return Err(TdmError::NoSegments);
    }
    for field in &tdm.header_fields {
        check_key_assignable(field)?;
    }
    for segment in &tdm.segments {
        for field in &segment.metadata.fields {
            check_key_assignable(field)?;
        }
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
fn check_line(line_no: usize, line: &str) -> Result<(), TdmError> {
    for (index, character) in line.chars().enumerate() {
        if !matches!(character, ' '..='~') {
            return Err(TdmError::NonPrintableCharacter {
                line: Some(line_no),
                keyword: line_keyword(line),
                column: index + 1,
                character,
            });
        }
    }
    // Every character is printable ASCII by here, so one byte is one character.
    if line.len() > MAX_LINE_CHARACTERS {
        return Err(TdmError::LineTooLong {
            line: Some(line_no),
            keyword: line_keyword(line),
            length: line.len(),
        });
    }
    Ok(())
}

/// A line's first whitespace-delimited token.
///
/// On the way in this is whatever the line opens with, which is the keyword on
/// a line 4.2.5 defines and the only name there is on one it does not.
fn line_keyword(line: &str) -> String {
    line.split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string()
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

/// Append one `KEY = VALUE` line, refusing a field the KVN form cannot carry.
///
/// `COMMENT` is excepted from the KVN syntax by CCSDS 503.0-B-2 4.2.5 c), and
/// 4.5.3 reads any line whose keyword is followed by a space as a comment. A
/// field keyed `COMMENT` has no assignment form: writing `COMMENT = value`
/// emits a line that reads back as a comment rather than as the field.
fn check_key_assignable(field: &TdmField) -> Result<(), TdmError> {
    // The key is compared trimmed: a key of `"COMMENT "` writes the same
    // line, which reads back as a comment just as an untrimmed one does.
    if field.key.trim() == COMMENT_KEY {
        return Err(TdmError::KeywordNotAssignable {
            keyword: field.key.clone(),
        });
    }
    Ok(())
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
    if suffix.is_empty() || !suffix.chars().all(|character| character.is_ascii_digit()) {
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

fn invalid_index(field: &str) -> TdmError {
    TdmError::InvalidField {
        keyword: field.to_string(),
        kind: TdmInputErrorKind::InvalidIndex,
    }
}

fn unknown_keyword(field: &str) -> TdmError {
    TdmError::InvalidField {
        keyword: field.to_string(),
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
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
RECEIVE_FREQ_1 = 2005-159T17:41:00
DATA_STOP\n",
        )
        .unwrap_err();
        assert_eq!(
            err,
            TdmError::MalformedRecord {
                line: 6,
                keyword: "RECEIVE_FREQ_1".to_string()
            }
        );
    }

    #[test]
    fn invalid_transmit_frequency_is_rejected() {
        let err = parse_kvn(
            "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
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
COMMENT=
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
RANGE = 2005-159T17:41:00 1.0
DATA_STOP\n";
        assert_eq!(
            parse_kvn(header),
            Err(TdmError::MalformedLine {
                line: 2,
                text: "COMMENT=".to_string(),
            })
        );

        let metadata = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
COMMENT=file = tdm.dat
META_STOP
DATA_START
RANGE = 2005-159T17:41:00 1.0
DATA_STOP\n";
        assert_eq!(
            parse_kvn(metadata),
            Err(TdmError::MalformedLine {
                line: 4,
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
META_START
TIME_SYSTEM = UTC
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
    }

    const TERMINATOR_BODY: &str = "CCSDS_TDM_VERS = 2.0|META_START|TIME_SYSTEM = UTC|\
META_STOP|DATA_START|RANGE = 2005-159T17:41:00 1.0|DATA_STOP\n";

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
    fn a_final_line_with_no_terminator_is_refused() {
        // 4.2.11 terminates every TDM line, the last one included. Nothing
        // checked it, and the writer joined its lines with a line feed and
        // added none at the end, so every message sidereon wrote departed here.
        let terminated = TERMINATOR_BODY.replace('|', "\n");
        let tdm = parse_kvn(&terminated).expect("a terminated message reads");
        assert_eq!(
            parse_kvn(terminated.trim_end_matches('\n')),
            Err(TdmError::UnterminatedFinalLine { line: 7 })
        );

        // The writer terminates its last line, and what it writes reads back.
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
    fn a_character_outside_printable_ascii_is_refused() {
        // 4.2.1: "The TDM line must contain only printable ASCII characters and
        // blanks. ASCII control characters (such as TAB, etc.) must not be used".
        let with_tab = "\
CCSDS_TDM_VERS = 2.0
META_START
\tTIME_SYSTEM = UTC
META_STOP
DATA_START
RANGE = 2005-159T17:41:00 1.0
DATA_STOP\n";
        assert_eq!(
            parse_kvn(with_tab),
            Err(TdmError::NonPrintableCharacter {
                line: Some(3),
                keyword: "TIME_SYSTEM".to_string(),
                column: 1,
                character: '\t',
            })
        );

        // A right double quotation mark, as two public TDM corpora carry in a
        // clock-offset comment.
        let with_non_ascii = "\
CCSDS_TDM_VERS = 2.0
COMMENT clock minus UTC\u{201d}
META_START
TIME_SYSTEM = UTC
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
META_START
{prefix}{value}
META_STOP
DATA_START
RANGE = 2005-159T17:41:00 1.0
DATA_STOP\n"
        );
        assert_eq!(
            parse_kvn(&text),
            Err(TdmError::LineTooLong {
                line: Some(3),
                keyword: "TIME_SYSTEM".to_string(),
                length: prefix.len() + MAX_LINE_CHARACTERS,
            })
        );

        // A line of exactly the limit is accepted.
        let exact = "B".repeat(MAX_LINE_CHARACTERS - prefix.len());
        let text = format!(
            "\
CCSDS_TDM_VERS = 2.0
META_START
{prefix}{exact}
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
}
