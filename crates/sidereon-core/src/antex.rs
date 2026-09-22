//! ANTEX 1.4 receiver and satellite antenna parser.
//!
//! The parser owns the byte/record grammar for the antenna calibration blocks
//! used by PPP and RTK correction paths. Values are stored in SI units:
//! PCO/PCV are meters, azimuth and zenith grids are degrees.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use crate::antenna;
use crate::constants::MM_PER_M;
use crate::format::columns::{field, fortran_f64, raw_field};
use crate::format::{Diagnostics, RecordRef, Skip, SkipReason};
use crate::validate::{self, FieldError};
use std::collections::BTreeMap;
use std::fmt;

/// Parsed ANTEX antenna calibration product.
#[derive(Debug, Clone, PartialEq)]
pub struct Antex {
    /// Latest completed antenna block for each trimmed `TYPE / SERIAL NO` id.
    /// A later block with the same id replaces this entry; all blocks remain
    /// available through [`Antex::antenna_intervals`].
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
    /// First parseable value from `DAZI`, in degrees; parsing initializes this
    /// field to `0.0` when no such value is present.
    pub dazi_deg: f64,
    /// First value from `ZEN1 / ZEN2 / DZEN`, in degrees; it anchors recovered
    /// PCV sample zeniths and is the lower bound checked by [`Antenna::pcv`].
    pub zenith_start_deg: f64,
    /// Second value from `ZEN1 / ZEN2 / DZEN`, in degrees; it is the inclusive
    /// upper bound checked by [`Antenna::pcv`].
    pub zenith_end_deg: f64,
    /// Third value from `ZEN1 / ZEN2 / DZEN`, in degrees, used to place
    /// successive PCV values; a zero value assigns every sample the start.
    pub zenith_step_deg: f64,
    /// Nonblank trimmed `SINEX CODE` content, or `None` when that record is
    /// absent or blank.
    pub sinex_code: Option<String>,
    /// Timestamp from `VALID FROM`, used as an inclusive lower bound by
    /// [`Antenna::valid_at`].
    pub valid_from: Option<AntexDateTime>,
    /// Timestamp from `VALID UNTIL`, used as an inclusive upper bound by
    /// [`Antenna::valid_at`].
    pub valid_until: Option<AntexDateTime>,
    /// Completed frequency blocks indexed by their trimmed labels; a repeated
    /// label replaces the earlier block in this antenna.
    pub frequencies: BTreeMap<String, Frequency>,
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

/// Civil UTC-like timestamp fields from `VALID FROM` / `VALID UNTIL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AntexDateTime {
    /// Validated civil calendar year, restricted by the shared validator to
    /// `0..=9999`.
    pub year: i32,
    /// Validated one-based civil month in `1..=12`.
    pub month: u8,
    /// Day accepted for the stored civil year and month.
    pub day: u8,
    /// UTC-like civil clock hour in `0..=23`.
    pub hour: u8,
    /// UTC-like civil minute in `0..=59`.
    pub minute: u8,
    /// Stored whole-second component; UTC-like validation permits ordinary
    /// values through `59` and a valid leap-second label `60`.
    pub second: u8,
}

/// ANTEX parse or lookup error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AntexError {
    /// A date/time component failed integer-range, calendar, clock, or
    /// UTC-like leap-second validation.
    InvalidDateTime,
    /// A public PCV input was rejected by shared validation.
    InvalidInput {
        /// Static label of the rejected PCV argument; the current path uses
        /// `"zenith_deg"`.
        field: &'static str,
        /// Validator reason, such as `"not finite"` or `"out of range"`.
        reason: &'static str,
    },
    /// The trimmed requested frequency label was absent from an antenna's
    /// frequency map.
    UnknownFrequency {
        /// Id of the antenna whose frequency map was queried.
        antenna_id: String,
        /// Original caller-supplied frequency argument, before lookup trims it.
        frequency: String,
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
            Self::Unwritable { field, reason } => {
                write!(f, "cannot serialize ANTEX {field}: {reason}")
            }
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
    /// Construct a timestamp after UTC-like civil-time validation.
    ///
    /// Invalid calendar or clock values, including an unsupported leap-second
    /// label, return [`AntexError::InvalidDateTime`].
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
    let readback_m = parsed_mm / MM_PER_M;
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
    let readback_m = parsed_mm / MM_PER_M;
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

fn validate_sinex_code(code: &str) -> Result<(), AntexError> {
    if code.is_empty() || code.len() > 60 || code.contains('\n') || code.contains('\r') {
        return Err(unwritable(
            "sinex_code",
            format!("invalid SINEX code: {code:?}"),
        ));
    }
    Ok(())
}

fn validate_frequency_label(freq: &str) -> Result<(), AntexError> {
    if freq.is_empty() || freq.len() > 20 || freq.contains('\n') || freq.contains('\r') {
        return Err(unwritable(
            "frequency",
            format!("invalid frequency label: {freq:?}"),
        ));
    }
    Ok(())
}

fn validate_antenna_header(antenna: &Antenna) -> Result<(), AntexError> {
    if antenna.id.is_empty()
        || antenna.id.len() > 60
        || antenna.id.contains('\n')
        || antenna.id.contains('\r')
    {
        return Err(unwritable(
            "id",
            format!("invalid antenna id: {:?}", antenna.id),
        ));
    }
    let id_line = format!("{:<60}TYPE / SERIAL NO", antenna.id);
    let decoded = decode_antenna_header(&id_line);
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

fn pcv_sample_sync_eq(a: &PcvSample, b: &PcvSample) -> bool {
    a.grid == b.grid
        && opt_f64_bits_eq(a.azimuth_deg, b.azimuth_deg)
        && f64_bits_eq(a.zenith_deg, b.zenith_deg)
        && f64_bits_eq(a.value_m, b.value_m)
}

fn frequency_sync_eq(a: &Frequency, b: &Frequency) -> bool {
    if a.frequency != b.frequency {
        return false;
    }
    if !f64_bits_eq(a.pco_m[0], b.pco_m[0])
        || !f64_bits_eq(a.pco_m[1], b.pco_m[1])
        || !f64_bits_eq(a.pco_m[2], b.pco_m[2])
    {
        return false;
    }
    if a.pcv_samples.len() != b.pcv_samples.len() {
        return false;
    }
    a.pcv_samples
        .iter()
        .zip(b.pcv_samples.iter())
        .all(|(s1, s2)| pcv_sample_sync_eq(s1, s2))
}

fn antenna_sync_eq(a: &Antenna, b: &Antenna) -> bool {
    if a.id != b.id
        || a.kind != b.kind
        || a.antenna_type != b.antenna_type
        || a.serial != b.serial
        || a.sinex_code != b.sinex_code
        || a.valid_from != b.valid_from
        || a.valid_until != b.valid_until
    {
        return false;
    }
    if !f64_bits_eq(a.dazi_deg, b.dazi_deg)
        || !f64_bits_eq(a.zenith_start_deg, b.zenith_start_deg)
        || !f64_bits_eq(a.zenith_end_deg, b.zenith_end_deg)
        || !f64_bits_eq(a.zenith_step_deg, b.zenith_step_deg)
    {
        return false;
    }
    if a.frequencies.len() != b.frequencies.len() {
        return false;
    }
    for ((k1, f1), (k2, f2)) in a.frequencies.iter().zip(b.frequencies.iter()) {
        if k1 != k2 || !frequency_sync_eq(f1, f2) {
            return false;
        }
    }
    true
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
    ///   width overflow or precision loss (e.g. `DAZI` or `ZEN1/ZEN2/DZEN` in `F6.1`).
    /// - Zenith bounds are non-finite, negative, inverted (`ZEN1 > ZEN2`), or `DZEN <= 0.0`
    ///   when PCV grid rows are present.
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
        let mut out = String::new();
        out.push_str(&labeled("     1.4            M", "ANTEX VERSION / SYST"));
        out.push_str(&labeled("", "END OF HEADER"));
        for blocks in self.antenna_intervals.values() {
            for antenna in blocks {
                encode_antenna(antenna, &mut out)?;
            }
        }
        Ok(out)
    }
}

fn encode_antenna(antenna: &Antenna, out: &mut String) -> Result<(), AntexError> {
    validate_antenna_header(antenna)?;
    out.push_str(&labeled("", "START OF ANTENNA"));
    out.push_str(&labeled(&antenna.id, "TYPE / SERIAL NO"));
    let dazi_str = validate_f6_1(antenna.dazi_deg, "dazi_deg")?;
    out.push_str(&labeled(&format!("  {dazi_str}"), "DAZI"));

    let zen_start_str = validate_f6_1(antenna.zenith_start_deg, "zenith_start_deg")?;
    let zen_end_str = validate_f6_1(antenna.zenith_end_deg, "zenith_end_deg")?;
    let zen_step_str = validate_f6_1(antenna.zenith_step_deg, "zenith_step_deg")?;
    if antenna.zenith_start_deg > antenna.zenith_end_deg {
        return Err(unwritable(
            "zenith_end_deg",
            format!(
                "inverted zenith bounds: start {} > end {}",
                antenna.zenith_start_deg, antenna.zenith_end_deg
            ),
        ));
    }
    out.push_str(&labeled(
        &format!("  {zen_start_str}{zen_end_str}{zen_step_str}"),
        "ZEN1 / ZEN2 / DZEN",
    ));

    if let Some(code) = &antenna.sinex_code {
        validate_sinex_code(code)?;
        out.push_str(&labeled(code, "SINEX CODE"));
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
        out.push_str(&labeled(&fmt_datetime(from), "VALID FROM"));
    }
    if let Some(until) = antenna.valid_until {
        out.push_str(&labeled(&fmt_datetime(until), "VALID UNTIL"));
    }

    let has_pcv_samples = antenna
        .frequencies
        .values()
        .any(|f| !f.pcv_samples.is_empty());
    if has_pcv_samples {
        if antenna.zenith_step_deg <= 0.0 {
            return Err(unwritable(
                "zenith_step_deg",
                format!(
                    "positive zenith step required when PCV samples are present, found {}",
                    antenna.zenith_step_deg
                ),
            ));
        }
        let start_tenths = f6_1_tenths(antenna.zenith_start_deg);
        let end_tenths = f6_1_tenths(antenna.zenith_end_deg);
        let step_tenths = f6_1_tenths(antenna.zenith_step_deg);
        if (end_tenths - start_tenths) % step_tenths != 0 {
            return Err(unwritable(
                "zenith_end_deg",
                format!(
                    "zenith_end_deg {} is not an integer multiple of step {} from start {}",
                    antenna.zenith_end_deg, antenna.zenith_step_deg, antenna.zenith_start_deg
                ),
            ));
        }
    }

    for (freq_key, frequency) in &antenna.frequencies {
        if freq_key != &frequency.frequency {
            return Err(unwritable(
                "frequencies",
                format!(
                    "frequency map key {freq_key:?} does not match frequency label {:?}",
                    frequency.frequency
                ),
            ));
        }
        encode_frequency(antenna, frequency, out)?;
    }
    out.push_str(&labeled("", "END OF ANTENNA"));
    Ok(())
}

fn encode_frequency(
    antenna: &Antenna,
    frequency: &Frequency,
    out: &mut String,
) -> Result<(), AntexError> {
    validate_frequency_label(&frequency.frequency)?;
    out.push_str(&labeled(&frequency.frequency, "START OF FREQUENCY"));

    let n_str = validate_pco_f10_2(frequency.pco_m[0], "north")?;
    let e_str = validate_pco_f10_2(frequency.pco_m[1], "east")?;
    let u_str = validate_pco_f10_2(frequency.pco_m[2], "up")?;
    out.push_str(&labeled(
        &format!("{n_str}{e_str}{u_str}"),
        "NORTH / EAST / UP",
    ));

    let (noazi_samples, azimuth_rows) =
        validate_and_group_pcv_samples(antenna, &frequency.pcv_samples)?;

    if !noazi_samples.is_empty() {
        out.push_str(&pcv_row(antenna, "   NOAZI", &noazi_samples)?);
    }

    for (azimuth_head, row_samples) in &azimuth_rows {
        out.push_str(&pcv_row(antenna, azimuth_head, row_samples)?);
    }

    out.push_str(&labeled("", "END OF FREQUENCY"));
    Ok(())
}

type GroupedPcvRows<'a> = (Vec<&'a PcvSample>, Vec<(String, Vec<&'a PcvSample>)>);

fn validate_and_group_pcv_samples<'a>(
    antenna: &Antenna,
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

    if !samples.is_empty() && antenna.zenith_step_deg <= 0.0 {
        return Err(unwritable(
            "zenith_step_deg",
            format!(
                "positive zenith step required when PCV samples are present, found {}",
                antenna.zenith_step_deg
            ),
        ));
    }
    let start_tenths = f6_1_tenths(antenna.zenith_start_deg);
    let step_tenths = f6_1_tenths(antenna.zenith_step_deg);
    if !samples.is_empty() && step_tenths <= 0 {
        return Err(unwritable(
            "zenith_step_deg",
            format!(
                "positive zenith step required when PCV samples are present, found {}",
                antenna.zenith_step_deg
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
        if sample.zenith_deg < antenna.zenith_start_deg {
            return Err(unwritable(
                "pcv_samples",
                format!(
                    "sample zenith_deg {} is below ZEN1 {}",
                    sample.zenith_deg, antenna.zenith_start_deg
                ),
            ));
        }
        let _ = validate_pcv_f8_2(sample.value_m)?;

        let raw_k = (sample.zenith_deg - antenna.zenith_start_deg) / antenna.zenith_step_deg;
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
        let reconstructed_zenith = antenna.zenith_start_deg + antenna.zenith_step_deg * (k as f64);
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
fn pcv_row(antenna: &Antenna, head: &str, samples: &[&PcvSample]) -> Result<String, AntexError> {
    let mut line = format!("{head:<8}");
    if antenna.zenith_step_deg <= 0.0 {
        return Err(unwritable(
            "zenith_step_deg",
            format!(
                "positive zenith step required for PCV row, found {}",
                antenna.zenith_step_deg
            ),
        ));
    }
    let start_tenths = f6_1_tenths(antenna.zenith_start_deg);
    let end_tenths = f6_1_tenths(antenna.zenith_end_deg);
    let step_tenths = f6_1_tenths(antenna.zenith_step_deg);
    let grid_k = if step_tenths > 0 && end_tenths >= start_tenths {
        ((end_tenths - start_tenths) / step_tenths) as usize
    } else {
        0
    };
    let sample_max_k = samples
        .iter()
        .map(|s| {
            ((s.zenith_deg - antenna.zenith_start_deg) / antenna.zenith_step_deg).round() as usize
        })
        .max()
        .unwrap_or(0);
    let max_k = grid_k.max(sample_max_k);

    let mut sample_idx = 0;
    for k in 0..=max_k {
        if sample_idx < samples.len() {
            let s = samples[sample_idx];
            let s_k = ((s.zenith_deg - antenna.zenith_start_deg) / antenna.zenith_step_deg).round()
                as usize;
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

/// A labeled fixed-column ANTEX record: the body left-justified into the tag
/// column, then the record-type label.
fn labeled(body: &str, label: &str) -> String {
    format!("{body:<LABEL_COLUMN$}{label}\n")
}

/// `VALID FROM` / `VALID UNTIL` value field: the six civil components in fixed columns.
/// Seconds are emitted from the stored integer second.
fn fmt_datetime(dt: AntexDateTime) -> String {
    format!(
        "{:6}{:6}{:6}{:6}{:6}{:13.7}",
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second as f64
    )
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
        "COMMENT"
        | "ANTEX VERSION / SYST"
        | "PCV TYPE / REFANT"
        | "END OF HEADER"
        | "METH / BY / # / DATE"
        | "METH / BY / DATE"
        | "# OF FREQUENCIES"
        | "START OF FREQ RMS"
        | "END OF FREQ RMS" => {}
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
    // ANTEX 1.4 defines DAZI as 2X,F6.1,52X (columns 2..8). If the two leading
    // separator columns are blank, parse the strict slice; otherwise, parse the
    // complete 0..8 compatibility field so leading numeric characters are not
    // truncated into an unintended zero value.
    let dazi = if raw_field(line, 0, 2) == "  " {
        fortran_f64(line, 2, 8, "antex dazi")
    } else {
        fortran_f64(line, 0, 8, "antex dazi")
    };
    if let Some(dazi) = dazi {
        current.dazi_deg = dazi;
    }
}

fn parse_zenith_grid(line: &str, state: &mut ParseState) {
    let Some(current) = state.current_antenna.as_mut() else {
        return;
    };
    let start = fortran_f64(line, 2, 8, "antex zenith start");
    let end = fortran_f64(line, 8, 14, "antex zenith end");
    let step = fortran_f64(line, 14, 20, "antex zenith step");
    if let (Some(start), Some(end), Some(step)) = (start, end, step) {
        current.zenith_start_deg = start;
        current.zenith_end_deg = end;
        current.zenith_step_deg = step;
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
    let year_f = fortran_f64(line, 0, 6, "antex valid year");
    let month_f = fortran_f64(line, 6, 12, "antex valid month");
    let day_f = fortran_f64(line, 12, 18, "antex valid day");
    let hour_f = fortran_f64(line, 18, 24, "antex valid hour");
    let minute_f = fortran_f64(line, 24, 30, "antex valid minute");
    let second_f = fortran_f64(line, 30, 43, "antex valid second");

    if let (Some(y), Some(m), Some(d), Some(h), Some(min), Some(sec)) =
        (year_f, month_f, day_f, hour_f, minute_f, second_f)
    {
        let year = datetime_i32(y)?;
        let month = datetime_u8(m)?;
        let day = datetime_u8(d)?;
        let hour = datetime_u8(h)?;
        let minute = datetime_u8(min)?;
        let civil = validate::civil_datetime_with_second_policy(
            i64::from(year),
            i64::from(month),
            i64::from(day),
            i64::from(hour),
            i64::from(minute),
            sec,
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

    let north = fortran_f64(line, 0, 10, "antex pco north");
    let east = fortran_f64(line, 10, 20, "antex pco east");
    let up = fortran_f64(line, 20, 30, "antex pco up");

    if let (Some(n), Some(e), Some(u)) = (north, east, up) {
        if n.is_finite() && e.is_finite() && u.is_finite() {
            current_frequency.pco_m = Some([n / MM_PER_M, e / MM_PER_M, u / MM_PER_M]);
            current_frequency.phase = FrequencyPhase::Pcv;
        }
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

    if line.trim().is_empty() {
        return;
    }

    let head = field(line, 0, 8).unwrap_or("");
    if head == "NOAZI" {
        add_pcv_values(None, line, state);
    } else if let Some(azimuth) = fortran_f64(line, 0, 8, "antex azimuth") {
        add_pcv_values(Some(azimuth), line, state);
    } else {
        // A grid row whose head token is neither `NOAZI` nor a parseable azimuth
        // is recorded as a typed skip rather than silently dropped, consistent
        // with the rest of the sans-I/O contract. Real ANTEX rows always carry a
        // recognized head, so a clean file is unaffected.
        state.diagnostics.push_skip(Skip {
            at: RecordRef::at_line(state.line),
            reason: SkipReason::MalformedField(FieldError::FloatParse {
                field: "antex pcv row head",
                value: head.to_string(),
            }),
        });
    }
}

fn add_pcv_values(azimuth_deg: Option<f64>, line: &str, state: &mut ParseState) {
    let Some(current_antenna) = state.current_antenna.as_ref() else {
        return;
    };
    let Some(current_frequency) = state.current_frequency.as_mut() else {
        return;
    };

    let grid_start = current_antenna.zenith_start_deg;
    let grid_step = current_antenna.zenith_step_deg;
    let line_num = state.line;

    let mut k = 0;
    while 8 + 8 * k < line.len() {
        let start = 8 + 8 * k;
        let end = start + 8;
        if let Some(val_str) = field(line, start, end) {
            match fortran_f64(line, start, end, "antex pcv value") {
                Some(value) => {
                    let zenith_deg = if grid_step == 0.0 {
                        grid_start
                    } else {
                        grid_start + grid_step * k as f64
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
                None => {
                    // A malformed PCV grid value is skipped with a typed reason rather
                    // than silently dropped or replaced by a fabricated default. The
                    // remaining valid samples on the row are still recovered.
                    state.diagnostics.push_skip(Skip {
                        at: RecordRef::at_line(line_num),
                        reason: SkipReason::MalformedField(FieldError::FloatParse {
                            field: "antex pcv value",
                            value: val_str.to_string(),
                        }),
                    });
                }
            }
        }
        k += 1;
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

    fn synchronized_antex(antenna: Antenna) -> Antex {
        let mut antex = Antex {
            antennas: BTreeMap::new(),
            antenna_intervals: BTreeMap::new(),
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
        let frequency = &antenna.frequencies["G01"];
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
        let frequency = &antenna.frequencies["G01"];

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
        let frequency = &antenna.frequencies["G01"];
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
        assert_eq!(ant_std_5.dazi_deg, 5.0);

        let antex_std_10 = Antex::parse(&dazi_block("    10.0")).expect("parse standard 10.0");
        let ant_std_10 = antex_std_10.antenna("TESTANT             TESTSER").unwrap();
        assert_eq!(ant_std_10.dazi_deg, 10.0);

        let antex_std_0 = Antex::parse(&dazi_block("     0.0")).expect("parse standard 0.0");
        let ant_std_0 = antex_std_0.antenna("TESTANT             TESTSER").unwrap();
        assert_eq!(ant_std_0.dazi_deg, 0.0);

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
        assert_eq!(ant_compact_5.dazi_deg, 5.0);
        assert_ne!(ant_compact_5.dazi_deg, 0.0);

        assert_eq!(ant_compact_space_5.dazi_deg, ant_std_5.dazi_deg);
        assert_eq!(ant_compact_space_5.dazi_deg, 5.0);
        assert_ne!(ant_compact_space_5.dazi_deg, 0.0);

        assert_eq!(ant_compact_10.dazi_deg, ant_std_10.dazi_deg);
        assert_eq!(ant_compact_10.dazi_deg, 10.0);
        assert_ne!(ant_compact_10.dazi_deg, 0.0);

        assert_eq!(ant_compact_0.dazi_deg, ant_std_0.dazi_deg);
        assert_eq!(ant_compact_0.dazi_deg, 0.0);

        assert_eq!(ant_compact_space_0.dazi_deg, ant_std_0.dazi_deg);
        assert_eq!(ant_compact_space_0.dazi_deg, 0.0);

        let encoded_compact = antex_compact_5.encode().expect("encode compact");
        let reparsed = Antex::parse(&encoded_compact).expect("re-parse encoded compact");
        assert_eq!(reparsed.skipped_records(), 0);
        let ant_reparsed = reparsed.antenna("TESTANT             TESTSER").unwrap();
        assert_eq!(ant_reparsed.dazi_deg, 5.0);
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
        let freq = &ant.frequencies["G01"];

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
        let freq = &ant.frequencies["G01"];

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
        let freq = &ant.frequencies["G01"];
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
            .zenith_step_deg = 5.0;
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
        ant.frequencies
            .get_mut("G01")
            .unwrap()
            .pcv_samples
            .push(PcvSample {
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
        ant.zenith_start_deg = 10.0;
        ant.zenith_end_deg = 20.0;
        ant.zenith_step_deg = 10.0;
        ant.frequencies.get_mut("G01").unwrap().pcv_samples = vec![
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
        ant.frequencies.get_mut("G01").unwrap().pcv_samples.insert(
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
        ant.frequencies.get_mut("G01").unwrap().pcv_samples.insert(
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
        ant.frequencies.get_mut("G01").unwrap().pcv_samples[0].azimuth_deg = Some(0.0);
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
        ant2.frequencies.get_mut("G01").unwrap().pcv_samples[0].grid = PcvGrid::Azimuth;
        ant2.frequencies.get_mut("G01").unwrap().pcv_samples[0].azimuth_deg = None;
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
        ant3.frequencies
            .get_mut("G01")
            .unwrap()
            .pcv_samples
            .swap(0, 1);
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
        let freq = ant4.frequencies.get_mut("G01").unwrap();
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
        ant4.zenith_end_deg = 20.0;
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
        ant.frequencies.get_mut("G01").unwrap().pco_m[0] = 0.0012345;
        let err = synchronized_antex(ant)
            .encode()
            .expect_err("PCO precision loss must fail");
        assert!(matches!(err, AntexError::Unwritable { field: "pco_m", .. }));

        let mut ant_pco_overflow = test_antenna();
        ant_pco_overflow.frequencies.get_mut("G01").unwrap().pco_m[0] = 10_000.0;
        let err_pco_overflow = synchronized_antex(ant_pco_overflow)
            .encode()
            .expect_err("PCO overflow must fail");
        assert!(matches!(
            err_pco_overflow,
            AntexError::Unwritable { field: "pco_m", .. }
        ));

        let mut ant2 = test_antenna();
        ant2.frequencies.get_mut("G01").unwrap().pcv_samples[0].value_m = 0.0012345;
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
        ant_pcv_overflow
            .frequencies
            .get_mut("G01")
            .unwrap()
            .pcv_samples[0]
            .value_m = 100.0;
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
        ant3.frequencies.get_mut("G01").unwrap().pco_m[0] = f64::INFINITY;
        let err3 = synchronized_antex(ant3)
            .encode()
            .expect_err("Inf PCO must fail");
        assert!(matches!(
            err3,
            AntexError::Unwritable { field: "pco_m", .. }
        ));

        let mut ant4 = test_antenna();
        ant4.frequencies.get_mut("G01").unwrap().pcv_samples[0].value_m = f64::INFINITY;
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
        ant_nonfinite_zen
            .frequencies
            .get_mut("G01")
            .unwrap()
            .pcv_samples[0]
            .zenith_deg = f64::INFINITY;
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
        ant_nonfinite_azi
            .frequencies
            .get_mut("G01")
            .unwrap()
            .pcv_samples = vec![PcvSample {
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
        ant5.zenith_step_deg = 0.05;
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
        ant_hdr_overflow.dazi_deg = 10000.0;
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
        ant_azi_prec.frequencies.get_mut("G01").unwrap().pcv_samples = vec![PcvSample {
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
        ant_azi_overflow
            .frequencies
            .get_mut("G01")
            .unwrap()
            .pcv_samples = vec![PcvSample {
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
        ant_zen_overflow
            .frequencies
            .get_mut("G01")
            .unwrap()
            .pcv_samples = vec![PcvSample {
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
            .frequencies
            .get_mut("G01")
            .unwrap()
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
        ant_nan_sync.frequencies.get_mut("G01").unwrap().pco_m[0] = f64::NAN;
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
            .dazi_deg = -0.0;
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
        ant.zenith_start_deg = 90.0;
        ant.zenith_end_deg = 0.0;
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
        ant2.zenith_step_deg = 0.0;
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
        ant3.zenith_step_deg = 1e-15;
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
    fn encode_refuses_off_grid_zenith_end_header() {
        let mut ant = test_antenna();
        ant.zenith_start_deg = 0.0;
        ant.zenith_end_deg = 10.0;
        ant.zenith_step_deg = 3.0;
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
