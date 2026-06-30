//! ANTEX 1.4 receiver and satellite antenna parser.
//!
//! The parser owns the byte/record grammar for the antenna calibration blocks
//! used by PPP and RTK correction paths. Values are stored in SI units:
//! PCO/PCV are meters, azimuth and zenith grids are degrees.

use crate::antenna;
use crate::constants::MM_PER_M;
use crate::format::columns::{fortran_f64, raw_field};
use crate::format::{Diagnostics, RecordRef, Skip, SkipReason};
use crate::validate::{self, FieldError};
use std::collections::BTreeMap;
use std::fmt;

/// Parsed ANTEX antenna calibration product.
#[derive(Debug, Clone, PartialEq)]
pub struct Antex {
    pub antennas: BTreeMap<String, Antenna>,
    antenna_intervals: BTreeMap<String, Vec<Antenna>>,
    /// Count of malformed records skipped during a forgiving parse (a corrupt PCV
    /// grid value or an unrecognized grid-row head); each is surfaced as a typed
    /// [`Skip`] in the parser's [`Diagnostics`]. A clean file parses with
    /// `skipped_records == 0`; a non-zero count lets a caller tell a pristine
    /// product apart from one that carried a malformed record without aborting the
    /// whole parse. No fabricated sample is emitted in its place. Read it through
    /// [`Antex::skipped_records`]. Mirrors [`crate::atmosphere::Ionex::skipped_records`].
    skipped_records: usize,
}

/// Receiver or satellite antenna block.
#[derive(Debug, Clone, PartialEq)]
pub struct Antenna {
    pub id: String,
    pub kind: AntennaKind,
    pub antenna_type: String,
    pub serial: String,
    pub dazi_deg: f64,
    pub zenith_start_deg: f64,
    pub zenith_end_deg: f64,
    pub zenith_step_deg: f64,
    pub sinex_code: Option<String>,
    pub valid_from: Option<AntexDateTime>,
    pub valid_until: Option<AntexDateTime>,
    pub frequencies: BTreeMap<String, Frequency>,
}

/// ANTEX antenna block role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntennaKind {
    Receiver,
    Satellite,
}

/// Frequency-specific PCO/PCV calibration block.
#[derive(Debug, Clone, PartialEq)]
pub struct Frequency {
    pub frequency: String,
    pub pco_m: [f64; 3],
    pub pcv_samples: Vec<PcvSample>,
}

/// One phase-center-variation grid value.
#[derive(Debug, Clone, PartialEq)]
pub struct PcvSample {
    pub grid: PcvGrid,
    pub azimuth_deg: Option<f64>,
    pub zenith_deg: f64,
    pub value_m: f64,
}

/// PCV grid type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcvGrid {
    NoAzimuth,
    Azimuth,
}

/// Civil UTC-like timestamp fields from `VALID FROM` / `VALID UNTIL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AntexDateTime {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
}

/// ANTEX parse or lookup error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AntexError {
    InvalidDateTime,
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    UnknownFrequency {
        antenna_id: String,
        frequency: String,
    },
    MissingPco {
        antenna_id: String,
        frequency: String,
    },
    EmptyPcvGrid {
        antenna_id: String,
        frequency: String,
    },
}

impl fmt::Display for AntexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDateTime => write!(f, "invalid ANTEX datetime"),
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid ANTEX input {field}: {reason}")
            }
            Self::UnknownFrequency {
                antenna_id,
                frequency,
            } => write!(f, "unknown frequency {frequency:?} for {antenna_id:?}"),
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
        }
    }
}

impl std::error::Error for AntexError {}

#[derive(Debug, Clone)]
struct ParseState {
    antennas: BTreeMap<String, Antenna>,
    antenna_intervals: BTreeMap<String, Vec<Antenna>>,
    current_antenna: Option<Antenna>,
    current_frequency: Option<FrequencyState>,
    /// One-based number of the line currently being processed, attached to skips.
    line: usize,
    /// Non-fatal diagnostics: typed skips for malformed records the forgiving
    /// parser dropped rather than aborting on.
    diagnostics: Diagnostics,
}

#[derive(Debug, Clone)]
struct FrequencyState {
    frequency: String,
    phase: FrequencyPhase,
    pco_m: Option<[f64; 3]>,
    samples: Vec<PcvSample>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrequencyPhase {
    Pco,
    Pcv,
}

impl Antex {
    /// Parse ANTEX text into receiver and satellite antenna blocks.
    pub fn parse(text: &str) -> Result<Self, AntexError> {
        let mut state = ParseState {
            antennas: BTreeMap::new(),
            antenna_intervals: BTreeMap::new(),
            current_antenna: None,
            current_frequency: None,
            line: 0,
            diagnostics: Diagnostics::new(),
        };

        for (index, line) in text.lines().enumerate() {
            state.line = index + 1;
            step(line, &mut state)?;
        }
        finalize_antenna(&mut state)?;

        let skipped_records = state.diagnostics.skips.len();
        Ok(Self {
            antennas: state.antennas,
            antenna_intervals: state.antenna_intervals,
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

    /// Frequency-dependent PCO (north/east/up), meters.
    pub fn pco(&self, frequency: &str) -> Result<[f64; 3], AntexError> {
        self.frequencies
            .get(frequency.trim())
            .map(|f| f.pco_m)
            .ok_or_else(|| AntexError::UnknownFrequency {
                antenna_id: self.id.clone(),
                frequency: frequency.to_string(),
            })
    }

    /// Frequency-dependent PCV, meters, with linear zenith/azimuth interpolation.
    pub fn pcv(
        &self,
        frequency: &str,
        zenith_deg: f64,
        azimuth_deg: Option<f64>,
    ) -> Result<f64, AntexError> {
        validate_pcv_zenith(zenith_deg, self.zenith_start_deg, self.zenith_end_deg)?;

        let frequency =
            self.frequencies
                .get(frequency.trim())
                .ok_or_else(|| AntexError::UnknownFrequency {
                    antenna_id: self.id.clone(),
                    frequency: frequency.to_string(),
                })?;

        frequency.pcv(self.id.as_str(), zenith_deg, azimuth_deg)
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
            interpolate_azimuth(
                antenna_id,
                &self.frequency,
                &azimuth_samples,
                azimuth_deg.expect("checked Some"),
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
    pub fn new(
        year: i32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
    ) -> Result<Self, AntexError> {
        let civil = validate::civil_datetime_with_second_policy(
            i64::from(year),
            i64::from(month),
            i64::from(day),
            i64::from(hour),
            i64::from(minute),
            f64::from(second),
            validate::CivilSecondPolicy::UtcLike,
        )
        .map_err(|_| AntexError::InvalidDateTime)?;
        Ok(Self::from_valid_civil(civil))
    }

    fn from_valid_civil(civil: validate::ValidCivil) -> Self {
        Self {
            year: civil.year as i32,
            month: civil.month as u8,
            day: civil.day as u8,
            hour: civil.hour as u8,
            minute: civil.minute as u8,
            second: civil.second.trunc() as u8,
        }
    }
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
// PCO/PCV in meters, angles in degrees); this writer maps it onto the fixed
// `1.4` ANTEX record grammar. It is round-trippable at the IR level - parsing
// the output reproduces the same [`Antex`] (every antenna interval, frequency,
// PCO triple, and PCV sample). It is not a byte-for-byte reproduction of an
// arbitrary source file: free-text header records, comments, and `# OF
// FREQUENCIES`/method lines that the parser does not model are not carried, so
// the writer emits a minimal canonical header instead.

/// ANTEX label column: records carry their record-type tag at columns 60..80.
const LABEL_COLUMN: usize = 60;

impl Antex {
    /// Serialize this product back to ANTEX 1.4 text.
    ///
    /// Deterministic and pure: the same [`Antex`] always produces byte-identical
    /// output, and re-parsing it yields an equal [`Antex`]. Antenna validity
    /// blocks are emitted in `TYPE / SERIAL` id order, preserving the file-order
    /// sequence within each id, so both the latest-wins [`Antex::antenna`] view
    /// and the per-id [`Antex::antenna_intervals`] view round-trip.
    ///
    /// The parse -> encode -> parse equality holds for products with no skipped
    /// records ([`Antex::skipped_records`] is zero): skips are not re-emitted and
    /// the skip count is part of the derived [`PartialEq`], so a product that
    /// dropped a malformed record would not compare equal to its re-parse.
    pub fn encode(&self) -> String {
        let mut out = String::new();
        out.push_str(&labeled("     1.4            M", "ANTEX VERSION / SYST"));
        out.push_str(&labeled("", "END OF HEADER"));
        for blocks in self.antenna_intervals.values() {
            for antenna in blocks {
                encode_antenna(antenna, &mut out);
            }
        }
        out
    }
}

fn encode_antenna(antenna: &Antenna, out: &mut String) {
    out.push_str(&labeled("", "START OF ANTENNA"));
    out.push_str(&labeled(&antenna.id, "TYPE / SERIAL NO"));
    out.push_str(&labeled(&fmt_num(antenna.dazi_deg), "DAZI"));
    out.push_str(&labeled(
        &format!(
            "{} {} {}",
            fmt_num(antenna.zenith_start_deg),
            fmt_num(antenna.zenith_end_deg),
            fmt_num(antenna.zenith_step_deg),
        ),
        "ZEN1 / ZEN2 / DZEN",
    ));
    if let Some(code) = &antenna.sinex_code {
        out.push_str(&labeled(code, "SINEX CODE"));
    }
    if let Some(from) = antenna.valid_from {
        out.push_str(&labeled(&fmt_datetime(from), "VALID FROM"));
    }
    if let Some(until) = antenna.valid_until {
        out.push_str(&labeled(&fmt_datetime(until), "VALID UNTIL"));
    }
    for frequency in antenna.frequencies.values() {
        encode_frequency(frequency, out);
    }
    out.push_str(&labeled("", "END OF ANTENNA"));
}

fn encode_frequency(frequency: &Frequency, out: &mut String) {
    out.push_str(&labeled(&frequency.frequency, "START OF FREQUENCY"));
    out.push_str(&labeled(
        &format!(
            "{} {} {}",
            fmt_num(frequency.pco_m[0] * MM_PER_M),
            fmt_num(frequency.pco_m[1] * MM_PER_M),
            fmt_num(frequency.pco_m[2] * MM_PER_M),
        ),
        "NORTH / EAST / UP",
    ));

    // NOAZI grid row. Samples are emitted in their stored (zenith-index) order;
    // the reader reconstructs each zenith from the antenna grid by position, so
    // only the values are written.
    let noazi: Vec<&PcvSample> = frequency
        .pcv_samples
        .iter()
        .filter(|sample| sample.grid == PcvGrid::NoAzimuth)
        .collect();
    if !noazi.is_empty() {
        out.push_str(&pcv_row("NOAZI", &noazi));
    }

    // Azimuth-dependent rows, grouped by azimuth in first-seen order. Each
    // azimuth's samples are contiguous in the sample vector (one ANTEX row), so
    // grouping preserves the per-row zenith-index order.
    let mut azimuth_rows: Vec<(f64, Vec<&PcvSample>)> = Vec::new();
    for sample in &frequency.pcv_samples {
        if sample.grid != PcvGrid::Azimuth {
            continue;
        }
        let Some(azimuth) = sample.azimuth_deg else {
            continue;
        };
        match azimuth_rows
            .iter_mut()
            .find(|(az, _)| az.to_bits() == azimuth.to_bits())
        {
            Some((_, row)) => row.push(sample),
            None => azimuth_rows.push((azimuth, vec![sample])),
        }
    }
    for (azimuth, row) in &azimuth_rows {
        out.push_str(&pcv_row(&fmt_num(*azimuth), row));
    }

    out.push_str(&labeled("", "END OF FREQUENCY"));
}

/// Render one PCV grid row: a leading label token (`NOAZI` or an azimuth) then
/// the millimeter values. PCV rows carry no record tag, so the reader routes
/// them through its default (grid-row) arm regardless of line length.
fn pcv_row(head: &str, samples: &[&PcvSample]) -> String {
    let mut line = String::from(head);
    for sample in samples {
        line.push(' ');
        line.push_str(&fmt_num(sample.value_m * MM_PER_M));
    }
    line.push('\n');
    line
}

/// A labeled fixed-column ANTEX record: the body left-justified into the tag
/// column, then the record-type label.
fn labeled(body: &str, label: &str) -> String {
    format!("{body:<LABEL_COLUMN$}{label}\n")
}

/// `VALID FROM` / `VALID UNTIL` value field: the six civil components the reader
/// reads as floats. Seconds are emitted from the stored integer second.
fn fmt_datetime(dt: AntexDateTime) -> String {
    format!(
        "{} {} {} {} {} {}",
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
    )
}

/// Shortest decimal form of a value that round-trips back to the same `f64`.
fn fmt_num(value: f64) -> String {
    format!("{value}")
}

fn step(line: &str, state: &mut ParseState) -> Result<(), AntexError> {
    match tag(line) {
        "START OF ANTENNA" => {
            finalize_antenna(state)?;
            state.current_antenna = None;
            state.current_frequency = None;
        }
        "END OF ANTENNA" => finalize_antenna(state)?,
        "TYPE / SERIAL NO" => parse_type_serial(line, state),
        "DAZI" => parse_dazi(line, state),
        "ZEN1 / ZEN2 / DZEN" => parse_zenith_grid(line, state),
        "SINEX CODE" => parse_sinex_code(line, state),
        "VALID FROM" => parse_valid(line, state, ValidField::From)?,
        "VALID UNTIL" => parse_valid(line, state, ValidField::Until)?,
        "START OF FREQUENCY" => begin_frequency(line, state),
        "END OF FREQUENCY" => finalize_frequency(state)?,
        "NORTH / EAST / UP" => parse_pco(line, state),
        _ => parse_pcv_row(line, state),
    }
    Ok(())
}

fn parse_type_serial(line: &str, state: &mut ParseState) {
    state.current_antenna = Some(decode_antenna_header(line));
    state.current_frequency = None;
}

fn parse_dazi(line: &str, state: &mut ParseState) {
    let Some(current) = state.current_antenna.as_mut() else {
        return;
    };
    if let Some(dazi) = parse_floats_from_prefix(line).first() {
        current.dazi_deg = *dazi;
    }
}

fn parse_zenith_grid(line: &str, state: &mut ParseState) {
    let Some(current) = state.current_antenna.as_mut() else {
        return;
    };
    let values = parse_floats_from_prefix(line);
    if values.len() >= 3 {
        current.zenith_start_deg = values[0];
        current.zenith_end_deg = values[1];
        current.zenith_step_deg = values[2];
    }
}

fn parse_sinex_code(line: &str, state: &mut ParseState) {
    let Some(current) = state.current_antenna.as_mut() else {
        return;
    };
    let code = raw_field(line, 0, 60).trim();
    if !code.is_empty() {
        current.sinex_code = Some(code.to_string());
    }
}

#[derive(Debug, Clone, Copy)]
enum ValidField {
    From,
    Until,
}

fn parse_valid(line: &str, state: &mut ParseState, field: ValidField) -> Result<(), AntexError> {
    let Some(current) = state.current_antenna.as_mut() else {
        return Ok(());
    };
    let values = parse_floats_from_prefix(line);
    if values.len() >= 6 {
        let year = datetime_i32(values[0])?;
        let month = datetime_u8(values[1])?;
        let day = datetime_u8(values[2])?;
        let hour = datetime_u8(values[3])?;
        let minute = datetime_u8(values[4])?;
        let civil = validate::civil_datetime_with_second_policy(
            i64::from(year),
            i64::from(month),
            i64::from(day),
            i64::from(hour),
            i64::from(minute),
            values[5],
            validate::CivilSecondPolicy::UtcLike,
        )
        .map_err(|_| AntexError::InvalidDateTime)?;
        let dt = AntexDateTime::from_valid_civil(civil);
        match field {
            ValidField::From => current.valid_from = Some(dt),
            ValidField::Until => current.valid_until = Some(dt),
        }
    }
    Ok(())
}

fn datetime_i32(value: f64) -> Result<i32, AntexError> {
    if !value.is_finite()
        || value.fract() != 0.0
        || value < i32::MIN as f64
        || value > i32::MAX as f64
    {
        return Err(AntexError::InvalidDateTime);
    }
    Ok(value as i32)
}

fn datetime_u8(value: f64) -> Result<u8, AntexError> {
    if !value.is_finite() || value.fract() != 0.0 || value < 0.0 || value > u8::MAX as f64 {
        return Err(AntexError::InvalidDateTime);
    }
    Ok(value as u8)
}

fn decode_antenna_header(line: &str) -> Antenna {
    let id = raw_field(line, 0, 60).trim().to_string();
    let antenna_type = raw_field(line, 0, 20).trim().to_string();
    let serial = raw_field(line, 20, 40).trim().to_string();
    let kind = if is_satellite_serial(&serial) {
        AntennaKind::Satellite
    } else {
        AntennaKind::Receiver
    };

    Antenna {
        id,
        kind,
        antenna_type,
        serial,
        dazi_deg: 0.0,
        zenith_start_deg: 0.0,
        zenith_end_deg: 0.0,
        zenith_step_deg: 0.0,
        sinex_code: None,
        valid_from: None,
        valid_until: None,
        frequencies: BTreeMap::new(),
    }
}

fn is_satellite_serial(serial: &str) -> bool {
    let bytes = serial.as_bytes();
    bytes.len() == 3
        && bytes[0].is_ascii_uppercase()
        && bytes[1].is_ascii_digit()
        && bytes[2].is_ascii_digit()
}

fn begin_frequency(line: &str, state: &mut ParseState) {
    if state.current_antenna.is_none() {
        return;
    }
    state.current_frequency = Some(FrequencyState {
        frequency: raw_field(line, 0, 20).trim().to_string(),
        phase: FrequencyPhase::Pco,
        pco_m: None,
        samples: Vec::new(),
    });
}

fn parse_pco(line: &str, state: &mut ParseState) {
    let Some(current_frequency) = state.current_frequency.as_mut() else {
        return;
    };
    if current_frequency.phase != FrequencyPhase::Pco {
        return;
    }

    let values = parse_floats_from_prefix(line);
    if values.len() >= 3 && values[..3].iter().all(|value| value.is_finite()) {
        current_frequency.pco_m = Some([
            values[0] / MM_PER_M,
            values[1] / MM_PER_M,
            values[2] / MM_PER_M,
        ]);
        current_frequency.phase = FrequencyPhase::Pcv;
    }
}

fn parse_pcv_row(line: &str, state: &mut ParseState) {
    if state
        .current_frequency
        .as_ref()
        .is_none_or(|frequency| frequency.phase != FrequencyPhase::Pcv)
    {
        return;
    }

    let tokens = parse_tokens(line);
    let Some((first, values)) = tokens.split_first() else {
        return;
    };

    if *first == "NOAZI" {
        add_pcv_values(None, values, state);
    } else if let Some(azimuth) = parse_float(first) {
        add_pcv_values(Some(azimuth), values, state);
    } else {
        // A grid row whose head token is neither `NOAZI` nor a parseable azimuth
        // is recorded as a typed skip rather than silently dropped, consistent
        // with the rest of the sans-I/O contract. Real ANTEX rows always carry a
        // recognized head, so a clean file is unaffected.
        state.diagnostics.push_skip(Skip {
            at: RecordRef::at_line(state.line),
            reason: SkipReason::MalformedField(FieldError::FloatParse {
                field: "antex pcv row head",
                value: (*first).to_string(),
            }),
        });
    }
}

fn add_pcv_values(azimuth_deg: Option<f64>, values: &[&str], state: &mut ParseState) {
    let Some(current_antenna) = state.current_antenna.as_ref() else {
        return;
    };
    let Some(current_frequency) = state.current_frequency.as_mut() else {
        return;
    };

    let grid_start = current_antenna.zenith_start_deg;
    let grid_step = current_antenna.zenith_step_deg;
    let line = state.line;
    for (index, value_text) in values.iter().enumerate() {
        let Some(value) = parse_float(value_text) else {
            // A malformed PCV grid value is skipped with a typed reason rather
            // than silently dropped or replaced by a fabricated default. The
            // remaining valid samples on the row are still recovered.
            state.diagnostics.push_skip(Skip {
                at: RecordRef::at_line(line),
                reason: SkipReason::MalformedField(FieldError::FloatParse {
                    field: "antex pcv value",
                    value: (*value_text).to_string(),
                }),
            });
            continue;
        };
        let zenith_deg = if grid_step == 0.0 {
            grid_start
        } else {
            grid_start + grid_step * index as f64
        };
        current_frequency.samples.push(PcvSample {
            grid: if azimuth_deg.is_some() {
                PcvGrid::Azimuth
            } else {
                PcvGrid::NoAzimuth
            },
            azimuth_deg,
            zenith_deg,
            value_m: value / MM_PER_M,
        });
    }
}

fn finalize_frequency(state: &mut ParseState) -> Result<(), AntexError> {
    let Some(current_frequency) = state.current_frequency.take() else {
        return Ok(());
    };
    let Some(current_antenna) = state.current_antenna.as_mut() else {
        return Ok(());
    };

    let pco_m = current_frequency
        .pco_m
        .ok_or_else(|| AntexError::MissingPco {
            antenna_id: current_antenna.id.clone(),
            frequency: current_frequency.frequency.clone(),
        })?;

    let frequency = Frequency {
        frequency: current_frequency.frequency,
        pco_m,
        pcv_samples: current_frequency.samples,
    };
    current_antenna
        .frequencies
        .insert(frequency.frequency.clone(), frequency);
    Ok(())
}

fn finalize_antenna(state: &mut ParseState) -> Result<(), AntexError> {
    finalize_frequency(state)?;
    let Some(current_antenna) = state.current_antenna.take() else {
        return Ok(());
    };
    state
        .antenna_intervals
        .entry(current_antenna.id.clone())
        .or_default()
        .push(current_antenna.clone());
    state
        .antennas
        .insert(current_antenna.id.clone(), current_antenna);
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

    Ok(antenna::interpolate_zenith_sorted(&sorted, zenith_deg)
        .expect("non-empty grid yields a value"))
}

fn tag(line: &str) -> &str {
    raw_field(line, 60, 80).trim()
}

fn parse_tokens(line: &str) -> Vec<&str> {
    line.split_whitespace().collect()
}

fn parse_floats_from_prefix(line: &str) -> Vec<f64> {
    let mut values = Vec::new();
    for token in parse_tokens(line) {
        let Some(value) = parse_float(token) else {
            break;
        };
        values.push(value);
    }
    values
}

fn parse_float(token: &str) -> Option<f64> {
    fortran_f64(token, 0, token.len(), "antex numeric field")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_antenna() -> Antenna {
        Antenna {
            id: "TESTANT TESTSER".to_string(),
            kind: AntennaKind::Receiver,
            antenna_type: "TESTANT".to_string(),
            serial: "TESTSER".to_string(),
            dazi_deg: 0.0,
            zenith_start_deg: 0.0,
            zenith_end_deg: 10.0,
            zenith_step_deg: 10.0,
            sinex_code: None,
            valid_from: None,
            valid_until: None,
            frequencies: BTreeMap::from([(
                "G01".to_string(),
                Frequency {
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
                },
            )]),
        }
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
            line("0", "DAZI"),
            line("0.0 10.0 5.0", "ZEN1 / ZEN2 / DZEN"),
            line("IGS_TEST", "SINEX CODE"),
            line("2020 1 1 0 0 0", "VALID FROM"),
            line("2021 12 31 23 59 59", "VALID UNTIL"),
            line("G01", "START OF FREQUENCY"),
            line("1.5 2.0 3.0", "NORTH / EAST / UP"),
            "NOAZI 1.0 2.0 3.0".to_string(),
            "0.0 1.0 2.0 3.0".to_string(),
            "90.0 4.0 5.0 6.0".to_string(),
            line("", "END OF FREQUENCY"),
            line("", "END OF ANTENNA"),
        ]
        .join("\n")
    }

    #[test]
    fn encode_round_trips_synthetic_block() {
        let antex = Antex::parse(&synthetic_block()).expect("parse synthetic block");
        assert_eq!(antex.skipped_records(), 0);
        let encoded = antex.encode();
        let reparsed = Antex::parse(&encoded).expect("re-parse encoded block");
        assert_eq!(antex, reparsed);
        assert_eq!(encoded, reparsed.encode());
    }

    #[test]
    fn malformed_pcv_value_is_skipped_not_silent() {
        // One corrupt grid value on an otherwise valid NOAZI row: the parse
        // succeeds, the bad value is dropped with a typed skip (counted), and the
        // surrounding valid samples survive - never silently dropped, never a
        // fabricated default.
        let text = [
            line("", "START OF ANTENNA"),
            line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
            line("0.0 10.0 5.0", "ZEN1 / ZEN2 / DZEN"),
            line("G01", "START OF FREQUENCY"),
            line("0.0 0.0 0.0", "NORTH / EAST / UP"),
            "NOAZI 1.0 BAD 3.0".to_string(),
            line("", "END OF FREQUENCY"),
            line("", "END OF ANTENNA"),
        ]
        .join("\n");

        let antex = Antex::parse(&text).expect("forgiving parse");
        assert_eq!(antex.skipped_records(), 1);
        let antenna = antex
            .antenna("TESTANT             TESTSER")
            .expect("antenna");
        let frequency = &antenna.frequencies["G01"];
        // Three tokens on the row, one malformed: two valid samples recovered.
        assert_eq!(frequency.pcv_samples.len(), 2);
    }
}
