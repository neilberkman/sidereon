//! Typed refusals of the RTCM encoders and of the ephemeris conversions.

use core::fmt;

use super::msm::MsmKind;
use super::ssr::SsrKind;
use super::RtcmDeparture;
use crate::ephemeris::LnavRecordError;
use crate::{GnssSatelliteId, GnssSystem, SatelliteIdError};

/// How an RTCM field states its value on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RtcmFieldEncoding {
    /// Unsigned binary, `0..=2^width - 1`.
    Unsigned,
    /// Two's complement, `-2^(width-1)..=2^(width-1) - 1`.
    TwosComplement,
    /// Sign bit and magnitude, `-(2^(width-1) - 1)..=2^(width-1) - 1`.
    SignMagnitude,
}

/// The record an encoder holds, named when the message number asked for does
/// not belong to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RtcmRecordKind {
    /// Station coordinates, 1005 or 1006.
    StationCoordinates,
    /// Antenna descriptor, 1007, 1008 or 1033.
    AntennaDescriptor,
    /// A multiple signal message of one constellation and kind.
    Msm {
        /// The constellation the record holds.
        system: GnssSystem,
        /// The MSM kind the record holds.
        kind: MsmKind,
    },
    /// An SSR message of one constellation and kind.
    Ssr {
        /// The constellation the record holds.
        system: GnssSystem,
        /// The SSR kind the record holds.
        kind: SsrKind,
    },
}

/// An optional MSM value whose presence or value the message kind decides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MsmOptionalField {
    /// The satellite's extended info (MSM7 only).
    ExtendedInfo,
    /// The satellite's rough phase-range rate (MSM7 only).
    RoughPhaseRangeRate,
    /// The signal's fine phase-range rate (MSM7 only).
    FinePhaseRangeRate,
}

/// What is wrong with an optional MSM value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MsmOptionalProblem {
    /// The message kind carries the value and none is given.
    Missing,
    /// The message kind does not carry the value and one is given.
    NotCarried,
    /// `Some` holds the invalid value, which is how `None` is written.
    InvalidValue(i64),
}

/// A satellite or signal list the MSM masks cannot state exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MsmMaskProblem {
    /// A satellite id outside `1..=64`.
    SatelliteOutsideMask {
        /// The satellite id.
        satellite: u8,
    },
    /// A satellite id listed twice.
    SatelliteListedTwice {
        /// The satellite id.
        satellite: u8,
    },
    /// A signal id outside `1..=32`.
    SignalOutsideMask {
        /// The signal id.
        signal: u8,
    },
    /// A signal id whose bit the signal mask does not set.
    SignalNotInMask {
        /// The signal id.
        signal: u8,
        /// The signal mask held.
        mask: u32,
    },
    /// A signal naming a satellite the satellite list does not hold.
    SignalSatelliteNotListed {
        /// The signal id.
        signal: u8,
        /// The satellite id the signal names.
        satellite: u8,
    },
    /// One satellite and signal cell listed twice.
    CellListedTwice {
        /// The satellite id.
        satellite: u8,
        /// The signal id.
        signal: u8,
    },
}

/// A value an RTCM encoder refuses to write.
///
/// Every encoder writes each field in its own width and refuses, with one of
/// these, a value it would otherwise have to truncate, fill, drop or write as
/// another value. The [`Display`](fmt::Display) text names the message, the
/// field and the value.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RtcmEncodeError {
    /// A value does not fit its field.
    FieldOutOfRange {
        /// Message number being encoded.
        message_number: u16,
        /// The field, with the satellite or signal it belongs to where it
        /// repeats.
        field: String,
        /// The value held.
        value: i128,
        /// Field width in bits.
        width: u8,
        /// How the field states its value.
        encoding: RtcmFieldEncoding,
    },
    /// A sign-magnitude field marked negative zero holds a nonzero value.
    NegativeZeroWithValue {
        /// Message number being encoded.
        message_number: u16,
        /// The field.
        field: String,
        /// The value held.
        value: i64,
    },
    /// The GLONASS 1020 negative-zero mask sets bits that name no
    /// sign-magnitude field.
    NegativeZeroMask {
        /// Message number being encoded.
        message_number: u16,
        /// The mask held.
        mask: u16,
    },
    /// The message number does not belong to the record being encoded.
    MessageNumber {
        /// Message number asked for.
        message_number: u16,
        /// The record held.
        record: RtcmRecordKind,
    },
    /// An optional value the message carries is missing, or one it does not
    /// carry is given.
    FieldPresence {
        /// Message number being encoded.
        message_number: u16,
        /// The record held.
        record: RtcmRecordKind,
        /// The field.
        field: &'static str,
        /// Whether the message carries the field.
        carried: bool,
    },
    /// A string character above `U+00FF`, which no 8-bit character states.
    NonLatin1Character {
        /// The string field.
        field: String,
        /// The character.
        character: char,
    },
    /// An ephemeris satellite id wider than the message's raw satellite
    /// field.
    SatelliteIdOutOfRange {
        /// Message number being encoded.
        message_number: u16,
        /// The satellite field, for example `"GPS PRN"`.
        field: &'static str,
        /// The satellite id held.
        value: u8,
        /// Field width in bits.
        width: u8,
    },
    /// An SSR satellite id wider than the message's satellite field.
    SsrSatelliteIdOutOfRange {
        /// Message number being encoded.
        message_number: u16,
        /// The satellite id held.
        value: u8,
        /// Field width in bits.
        width: u8,
    },
    /// An SSR record list the message kind does not write is not empty.
    SsrRecordsNotCarried {
        /// Message number being encoded.
        message_number: u16,
        /// The SSR kind.
        kind: SsrKind,
        /// The record list, for example `"orbit"`.
        records: &'static str,
        /// Records given.
        count: usize,
    },
    /// A combined orbit/clock message holds unequal orbit and clock lists.
    SsrCombinedRecordCounts {
        /// Message number being encoded.
        message_number: u16,
        /// Orbit records given.
        orbit: usize,
        /// Clock records given.
        clock: usize,
    },
    /// A combined orbit/clock record pair names two satellites.
    SsrCombinedSatelliteMismatch {
        /// Message number being encoded.
        message_number: u16,
        /// Zero-based record index.
        index: usize,
        /// Satellite id of the orbit record.
        orbit_satellite: u8,
        /// Satellite id of the clock record.
        clock_satellite: u8,
    },
    /// A high-rate clock record holds c1 or c2, which the message does not
    /// carry.
    SsrHighRateClockTerms {
        /// Message number being encoded.
        message_number: u16,
        /// Satellite id of the record.
        satellite: u8,
        /// c1 held.
        c1: i32,
        /// c2 held.
        c2: i32,
    },
    /// The SSR header's satellite count differs from the records written.
    SsrSatelliteCount {
        /// Message number being encoded.
        message_number: u16,
        /// Count the header declares.
        declared: usize,
        /// Records the message writes.
        records: usize,
    },
    /// An MSM satellite or signal list the masks cannot state.
    MsmMask {
        /// Message number being encoded.
        message_number: u16,
        /// What the masks cannot state.
        problem: MsmMaskProblem,
    },
    /// An optional MSM value the message kind decides is missing, not
    /// carried, or `Some` of the invalid value.
    MsmOptional {
        /// Message number being encoded.
        message_number: u16,
        /// The MSM kind.
        kind: MsmKind,
        /// Satellite id.
        satellite: u8,
        /// Signal id, for a signal field.
        signal: Option<u8>,
        /// The field.
        field: MsmOptionalField,
        /// What is wrong with it.
        problem: MsmOptionalProblem,
    },
    /// `trailing_bits` holds only zero bits, which read back as the byte
    /// alignment and so cannot be restated.
    TrailingZeroBits {
        /// Message number being encoded.
        message_number: u16,
        /// Zero bits held.
        bits: usize,
    },
    /// A departure from the format the record carries, refused under
    /// [`super::RtcmPolicy::Strict`].
    StrictDeparture(RtcmDeparture),
    /// An unsupported message's body is shorter than its 12-bit message
    /// number.
    UnsupportedBodyTooShort {
        /// Message number held.
        message_number: u16,
    },
    /// An unsupported message's body carries another message number.
    UnsupportedBodyNumber {
        /// Message number held.
        message_number: u16,
        /// Message number the body carries.
        carried: u16,
    },
    /// A message number the decoder reads into a typed variant is held as
    /// unsupported.
    UnsupportedDecodedNumber {
        /// Message number held.
        message_number: u16,
    },
    /// A frame body longer than the 1023 bytes the 10-bit length states.
    FrameBodyTooLong {
        /// Body length in bytes.
        len: usize,
    },
    /// A frame reserved value wider than its 6-bit field.
    FrameReservedOutOfRange {
        /// The reserved value.
        value: u8,
    },
}

fn msm_label(kind: MsmKind) -> &'static str {
    match kind {
        MsmKind::Msm1 => "MSM1",
        MsmKind::Msm2 => "MSM2",
        MsmKind::Msm3 => "MSM3",
        MsmKind::Msm4 => "MSM4",
        MsmKind::Msm5 => "MSM5",
        MsmKind::Msm6 => "MSM6",
        MsmKind::Msm7 => "MSM7",
    }
}

impl fmt::Display for RtcmEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FieldOutOfRange {
                message_number,
                field,
                value,
                width,
                encoding,
            } => {
                write!(
                    f,
                    "RTCM {message_number} {field} {value} does not fit its {width}-bit "
                )?;
                let widest_signed = (1i128 << (width - 1)) - 1;
                match encoding {
                    RtcmFieldEncoding::Unsigned => {
                        write!(f, "unsigned field (0..={})", (1u128 << width) - 1)
                    }
                    RtcmFieldEncoding::TwosComplement => write!(
                        f,
                        "two's-complement field ({}..={widest_signed})",
                        -widest_signed - 1
                    ),
                    RtcmFieldEncoding::SignMagnitude => write!(
                        f,
                        "sign-magnitude field (-{widest_signed}..={widest_signed})"
                    ),
                }
            }
            Self::NegativeZeroWithValue {
                message_number,
                field,
                value,
            } => write!(
                f,
                "RTCM {message_number} {field} is marked negative zero but holds {value}"
            ),
            Self::NegativeZeroMask {
                message_number,
                mask,
            } => write!(
                f,
                "RTCM {message_number} negative_zero {mask:#06x} sets bits that name no \
                 sign-magnitude field"
            ),
            Self::MessageNumber {
                message_number,
                record,
            } => {
                write!(f, "RTCM message number {message_number} is not ")?;
                match record {
                    RtcmRecordKind::StationCoordinates => {
                        write!(f, "station coordinates 1005/1006")
                    }
                    RtcmRecordKind::AntennaDescriptor => {
                        write!(f, "an antenna descriptor 1007/1008/1033")
                    }
                    RtcmRecordKind::Msm { system, kind } => {
                        write!(f, "the {system:?} {kind:?} message this MSM holds")
                    }
                    RtcmRecordKind::Ssr { system, kind } => {
                        write!(f, "the {system:?} {kind:?} SSR message this record holds")
                    }
                }
            }
            Self::FieldPresence {
                message_number,
                record,
                field,
                carried,
            } => {
                let ssr = if matches!(record, RtcmRecordKind::Ssr { .. }) {
                    "SSR "
                } else {
                    ""
                };
                match (record, carried) {
                    (RtcmRecordKind::StationCoordinates, true) => write!(
                        f,
                        "RTCM {message_number} carries an {field}, and none is given"
                    ),
                    (RtcmRecordKind::StationCoordinates, false) => write!(
                        f,
                        "RTCM {message_number} carries no {field}; a height is written as 1006"
                    ),
                    (_, true) => write!(
                        f,
                        "RTCM {ssr}{message_number} carries the {field}, and none is given"
                    ),
                    (_, false) => write!(
                        f,
                        "RTCM {ssr}{message_number} carries no {field}, and one is given"
                    ),
                }
            }
            Self::NonLatin1Character { field, character } => write!(
                f,
                "RTCM {field} character {character:?} (U+{:04X}) is not an 8-bit character",
                u32::from(*character)
            ),
            Self::SatelliteIdOutOfRange {
                message_number,
                field,
                value,
                width,
            } => write!(
                f,
                "{field} {value} in {message_number} does not fit the {width}-bit raw satellite \
                 field (0..={})",
                (1u16 << width) - 1
            ),
            Self::SsrSatelliteIdOutOfRange {
                message_number,
                value,
                width,
            } => write!(
                f,
                "RTCM SSR {message_number} satellite id {value} does not fit the {width}-bit \
                 satellite field (0..={})",
                (1u16 << width) - 1
            ),
            Self::SsrRecordsNotCarried {
                message_number,
                kind,
                records,
                count,
            } => write!(
                f,
                "RTCM SSR {message_number} ({kind:?}) writes no {records} records, and {count} \
                 are given"
            ),
            Self::SsrCombinedRecordCounts {
                message_number,
                orbit,
                clock,
            } => write!(
                f,
                "RTCM SSR {message_number} combined orbit/clock message carries {orbit} orbit \
                 records and {clock} clock records; each satellite needs one of each"
            ),
            Self::SsrCombinedSatelliteMismatch {
                message_number,
                index,
                orbit_satellite,
                clock_satellite,
            } => write!(
                f,
                "RTCM SSR {message_number} combined orbit/clock record {index} names satellite \
                 id {orbit_satellite} for its orbit and {clock_satellite} for its clock"
            ),
            Self::SsrHighRateClockTerms {
                message_number,
                satellite,
                c1,
                c2,
            } => write!(
                f,
                "RTCM SSR {message_number} high-rate clock record for satellite {satellite} \
                 holds c1 {c1} and c2 {c2}; the message carries only c0"
            ),
            Self::SsrSatelliteCount {
                message_number,
                declared,
                records,
            } => write!(
                f,
                "RTCM SSR {message_number} header satellite count {declared} differs from the \
                 {records} records the message writes"
            ),
            Self::MsmMask {
                message_number,
                problem,
            } => {
                write!(f, "RTCM MSM {message_number} cannot be encoded: ")?;
                match problem {
                    MsmMaskProblem::SatelliteOutsideMask { satellite } => write!(
                        f,
                        "satellite id {satellite} is outside the 1..=64 satellite mask"
                    ),
                    MsmMaskProblem::SatelliteListedTwice { satellite } => {
                        write!(f, "satellite id {satellite} is listed twice")
                    }
                    MsmMaskProblem::SignalOutsideMask { signal } => {
                        write!(f, "signal id {signal} is outside the 1..=32 signal mask")
                    }
                    MsmMaskProblem::SignalNotInMask { signal, mask } => write!(
                        f,
                        "signal id {signal} is not set in the signal mask {mask:#010x}"
                    ),
                    MsmMaskProblem::SignalSatelliteNotListed { signal, satellite } => write!(
                        f,
                        "signal {signal} names satellite id {satellite}, which the satellite \
                         list does not hold"
                    ),
                    MsmMaskProblem::CellListedTwice { satellite, signal } => write!(
                        f,
                        "the cell for satellite id {satellite} signal {signal} is listed twice"
                    ),
                }
            }
            Self::MsmOptional {
                message_number,
                kind,
                satellite,
                signal,
                field,
                problem,
            } => {
                write!(
                    f,
                    "RTCM {} {message_number} satellite {satellite} ",
                    msm_label(*kind)
                )?;
                if let Some(signal) = signal {
                    write!(f, "signal {signal} ")?;
                }
                let (name, article) = match field {
                    MsmOptionalField::ExtendedInfo => ("extended info", ""),
                    MsmOptionalField::RoughPhaseRangeRate => ("rough phase-range rate", "a "),
                    MsmOptionalField::FinePhaseRangeRate => ("fine phase-range rate", "a "),
                };
                match problem {
                    MsmOptionalProblem::Missing => {
                        write!(f, "has no {name}, which {} carries", msm_label(*kind))
                    }
                    MsmOptionalProblem::NotCarried => write!(
                        f,
                        "holds {article}{name}, which {} does not carry",
                        msm_label(*kind)
                    ),
                    MsmOptionalProblem::InvalidValue(value) => write!(
                        f,
                        "{name} Some({value}) is the invalid value, written for None"
                    ),
                }
            }
            Self::TrailingZeroBits {
                message_number,
                bits,
            } => write!(
                f,
                "RTCM {message_number} trailing_bits holds {bits} zero bits, which read back as \
                 the byte alignment; leave it empty"
            ),
            Self::StrictDeparture(departure) => {
                write!(f, "{departure} (refused under the strict policy)")
            }
            Self::UnsupportedBodyTooShort { message_number } => write!(
                f,
                "RTCM unsupported message {message_number} body is shorter than its 12-bit \
                 message number"
            ),
            Self::UnsupportedBodyNumber {
                message_number,
                carried,
            } => write!(
                f,
                "RTCM unsupported message {message_number} body carries message number {carried}"
            ),
            Self::UnsupportedDecodedNumber { message_number } => write!(
                f,
                "RTCM message {message_number} is decoded into its typed variant, not held as \
                 unsupported"
            ),
            Self::FrameBodyTooLong { len } => write!(
                f,
                "RTCM body of {len} bytes exceeds the 1023-byte frame limit"
            ),
            Self::FrameReservedOutOfRange { value } => write!(
                f,
                "RTCM frame reserved value {value} does not fit its 6-bit field (0..=63)"
            ),
        }
    }
}

impl std::error::Error for RtcmEncodeError {}

impl From<RtcmEncodeError> for crate::Error {
    fn from(error: RtcmEncodeError) -> Self {
        crate::Error::RtcmEncode(Box::new(error))
    }
}

/// Why a decoded RTCM ephemeris has no satellite or no broadcast record.
///
/// Returned by the ephemeris `satellite` accessors and `to_broadcast_record`
/// conversions. The raw message keeps every field it held; these say why it
/// names no satellite, or why the solver's record cannot be built from it.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RtcmConversionError {
    /// A satellite id wider than the message's raw satellite field.
    SatelliteIdOutOfRange {
        /// Message number.
        message_number: u16,
        /// The satellite field, for example `"GPS PRN"`.
        field: &'static str,
        /// The satellite id held.
        value: u8,
        /// Field width in bits.
        width: u8,
    },
    /// A satellite id the field carries that is not a satellite token.
    InvalidSatellite {
        /// Message number.
        message_number: u16,
        /// The satellite field.
        field: &'static str,
        /// The satellite id held.
        value: u8,
        /// Why the id is not a satellite token.
        error: SatelliteIdError,
    },
    /// A 1019 DF009 value in 40..=63 whose SBAS broadcast PRN lies outside the
    /// SBAS PRN window.
    SbasPrnOutsideWindow {
        /// The DF009 value held.
        value: u8,
        /// The SBAS broadcast PRN it names.
        broadcast_prn: u16,
    },
    /// A 1019 DF009 value naming an SBAS satellite, which has no GPS LNAV
    /// broadcast record.
    NoLnavRecord {
        /// The DF009 value held.
        value: u8,
        /// The SBAS satellite it names.
        satellite: GnssSatelliteId,
    },
    /// The caller's full week does not reduce to the 10-bit RTCM week.
    WeekMismatch {
        /// Message number (1019 or 1044).
        message_number: u16,
        /// The caller's full week.
        full_week: u32,
        /// The 10-bit week the message holds.
        week: u16,
    },
    /// A week and time of week that name no representable instant.
    TimeNotRepresentable {
        /// The time, for example `"GPS toe"`.
        field: &'static str,
    },
    /// The Galileo week does not fit the GPST week axis.
    GalileoWeekOverflow,
    /// A Galileo SISA index in the spare range 126..=254.
    SisaSpare {
        /// The index held.
        index: u8,
    },
    /// Galileo SISA index 255, no accuracy prediction available (NAPA).
    SisaNoPrediction,
    /// A URA index above the 4-bit domain.
    UraOutOfRange {
        /// The constellation.
        system: GnssSystem,
        /// The index held.
        index: u8,
    },
    /// A URA index with no accuracy prediction (15).
    UraNoPrediction {
        /// The constellation.
        system: GnssSystem,
        /// The index held.
        index: u8,
    },
    /// The GPS fit-interval flag, IODE and IODC name no curve-fit interval.
    FitInterval(LnavRecordError),
}

fn system_label(system: GnssSystem) -> &'static str {
    match system {
        GnssSystem::Gps => "GPS",
        GnssSystem::Glonass => "GLONASS",
        GnssSystem::Galileo => "Galileo",
        GnssSystem::BeiDou => "BeiDou",
        GnssSystem::Qzss => "QZSS",
        GnssSystem::Navic => "NavIC",
        GnssSystem::Sbas => "SBAS",
    }
}

impl fmt::Display for RtcmConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SatelliteIdOutOfRange {
                message_number,
                field,
                value,
                width,
            } => write!(
                f,
                "{field} {value} in {message_number} does not fit the {width}-bit raw satellite \
                 field (0..={})",
                (1u16 << width) - 1
            ),
            Self::InvalidSatellite {
                message_number,
                field,
                error,
                ..
            } => write!(f, "invalid {field} in {message_number}: {error}"),
            Self::SbasPrnOutsideWindow {
                value,
                broadcast_prn,
            } => write!(
                f,
                "GPS PRN {value} in 1019 names SBAS PRN {broadcast_prn}, outside the SBAS \
                 broadcast PRN window"
            ),
            Self::NoLnavRecord { value, satellite } => write!(
                f,
                "1019 satellite {value} names {satellite}, which has no GPS LNAV broadcast record"
            ),
            Self::WeekMismatch {
                message_number,
                full_week,
                week,
            } => {
                let system = if *message_number == 1044 {
                    "QZSS"
                } else {
                    "GPS"
                };
                write!(
                    f,
                    "{system} full week {full_week} disagrees with 10-bit RTCM week {week}"
                )
            }
            Self::TimeNotRepresentable { field } => {
                write!(f, "RTCM broadcast {field} is not representable")
            }
            Self::GalileoWeekOverflow => write!(f, "RTCM Galileo week overflows GPST axis"),
            Self::SisaSpare { index } => write!(
                f,
                "RTCM Galileo ephemeris SISA index {index} is spare with no defined accuracy"
            ),
            Self::SisaNoPrediction => write!(
                f,
                "RTCM Galileo ephemeris SISA index 255 indicates no accuracy prediction \
                 available (NAPA)"
            ),
            Self::UraOutOfRange { system, index } => write!(
                f,
                "RTCM {} ephemeris URA index {index} exceeds 4-bit range",
                system_label(*system)
            ),
            Self::UraNoPrediction { system, index } => write!(
                f,
                "RTCM {} ephemeris URA index {index} has no accuracy prediction",
                system_label(*system)
            ),
            Self::FitInterval(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for RtcmConversionError {}

impl From<RtcmConversionError> for crate::Error {
    fn from(error: RtcmConversionError) -> Self {
        crate::Error::RtcmConversion(Box::new(error))
    }
}
