//! Sans-I/O NMEA 0183 sentence parsing and GGA writing.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod epoch;
mod fields;
mod sentence;
#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
mod write;

pub use crate::format::{Diagnostics, Parsed, RecordRef, Skip, SkipReason, Warning, WarningKind};
pub use crate::validate::FieldError;
pub use epoch::{EpochSnapshot, GsaEntry, GsvGroup, NmeaAccumulator, NmeaChunkOutput};
pub use fields::{
    Gga, GgaQuality, Gll, Gsa, GsaFixMode, GsaSelectionMode, Gst, Gsv, GsvSatellite,
    NmeaCoordinate, NmeaDate, NmeaSatNumber, NmeaSignalId, NmeaTalker, NmeaTime, Rmc, RmcStatus,
    Vtg, Zda,
};
pub use sentence::{NmeaBody, NmeaSentence};
pub use write::write_gga;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum NmeaError {
    #[error("not an NMEA sentence: {reason}")]
    NotFramed { reason: &'static str },
    #[error("checksum mismatch: computed {computed:02X}, stated {stated:02X}")]
    ChecksumMismatch { computed: u8, stated: u8 },
    #[error("unsupported sentence type {address}")]
    UnsupportedType { address: String },
    #[error("proprietary sentence {address}")]
    Proprietary { address: String },
    #[error("malformed field: {0}")]
    MalformedField(#[from] FieldError),
    #[error("invalid input {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct NmeaLog {
    pub sentences: Vec<NmeaSentence>,
}

pub fn parse_sentence(line: &str) -> Result<Parsed<NmeaSentence>, NmeaError> {
    sentence::parse_framed(sentence::frame_sentence(line)?)
}

pub fn parse_nmea(input: &[u8]) -> Parsed<NmeaLog> {
    let mut diagnostics = Diagnostics::new();
    let mut sentences = Vec::new();
    for (index, line) in input.split(|b| *b == b'\n').enumerate() {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        let line_number = index + 1;
        match std::str::from_utf8(line) {
            Ok(line) => match parse_sentence(line) {
                Ok(mut parsed) => {
                    set_diagnostic_lines(&mut parsed.diagnostics, line_number);
                    merge_diagnostics(&mut diagnostics, parsed.diagnostics);
                    sentences.push(parsed.value);
                }
                Err(error) => push_error_skip_at_line(&mut diagnostics, error, line_number),
            },
            Err(_) => push_error_skip_at_line(
                &mut diagnostics,
                NmeaError::NotFramed {
                    reason: "non-ASCII byte",
                },
                line_number,
            ),
        }
    }
    Parsed::new(NmeaLog { sentences }, diagnostics)
}

pub fn parse_nmea_str(text: &str) -> Parsed<NmeaLog> {
    parse_nmea(text.as_bytes())
}

pub fn group_epochs(log: &NmeaLog) -> Vec<EpochSnapshot> {
    let mut accumulator = NmeaAccumulator::new();
    let mut snapshots = Vec::new();
    for sentence in &log.sentences {
        if let Some(snapshot) = accumulator.push(sentence) {
            snapshots.push(snapshot);
        }
    }
    if let Some(snapshot) = accumulator.finish() {
        snapshots.push(snapshot);
    }
    snapshots
}

pub(crate) fn merge_diagnostics(target: &mut Diagnostics, mut source: Diagnostics) {
    target.skips.append(&mut source.skips);
    target.warnings.append(&mut source.warnings);
}

fn push_error_skip_at_line(diagnostics: &mut Diagnostics, error: NmeaError, line: usize) {
    push_error_skip_at(diagnostics, error, RecordRef::at_line(line));
}

pub(crate) fn push_error_skip_at(diagnostics: &mut Diagnostics, error: NmeaError, at: RecordRef) {
    let reason = match error {
        NmeaError::NotFramed {
            reason: "non-ASCII byte",
        } => SkipReason::InconsistentRecord("non-ASCII byte"),
        NmeaError::NotFramed {
            reason: "sentence over length cap",
        } => SkipReason::InconsistentRecord("sentence over length cap"),
        NmeaError::NotFramed {
            reason: "malformed checksum",
        } => SkipReason::InconsistentRecord("malformed checksum"),
        NmeaError::NotFramed { .. } => {
            SkipReason::UnknownBlock("no NMEA start delimiter".to_string())
        }
        NmeaError::ChecksumMismatch { .. } => SkipReason::InconsistentRecord("checksum mismatch"),
        NmeaError::UnsupportedType { ref address } if address == "encapsulated sentence" => {
            SkipReason::UnsupportedRecordType("encapsulated sentence")
        }
        NmeaError::UnsupportedType { .. } => {
            SkipReason::UnsupportedRecordType("unsupported sentence type")
        }
        NmeaError::Proprietary { .. } => SkipReason::UnsupportedRecordType("proprietary sentence"),
        NmeaError::MalformedField(error) => SkipReason::MalformedField(error),
        NmeaError::InvalidInput { .. } => SkipReason::InconsistentRecord("invalid input"),
    };
    diagnostics.push_skip(Skip { at, reason });
}

pub(crate) fn set_diagnostic_lines(diagnostics: &mut Diagnostics, line: usize) {
    for skip in &mut diagnostics.skips {
        if skip.at.line.is_none() {
            skip.at.line = Some(line);
        }
    }
    for warning in &mut diagnostics.warnings {
        if warning.at.line.is_none() {
            warning.at.line = Some(line);
        }
    }
}
