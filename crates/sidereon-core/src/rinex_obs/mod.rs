//! RINEX 2.0x/3.0x/4.0x observation-file parser and single-frequency pseudorange
//! extraction.
//!
//! Parses a RINEX observation file (`OBSERVATION DATA`) into a typed
//! [`RinexObs`] product: the header (including the surveyed
//! [`ObsHeader::approx_position_m`] a-priori receiver position and optional
//! [`ObsHeader::antenna_delta_hen_m`] antenna offset), the per-constellation
//! observation-code table, and the per-epoch
//! satellite→observation values. A pseudorange helper ([`pseudoranges`]) then
//! selects one single-frequency code per system and yields the
//! `(satellite, range_m)` pairs the single-point-positioning solver consumes.
//!
//! # Build vs adopt
//!
//! Like the SP3 and RINEX-NAV readers, this is a hand-rolled, fixed-column text
//! reader in the house style rather than an adoption of the MPL-2.0 `rinex`
//! crate (which would pull a parallel time stack and identifier set into the
//! GNSS layer). The grammar is small and fully specified.
//!
//! It is a **deterministic byte-to-record** parse of a fixed-column text format,
//! not a float recipe; there is no 0-ULP claim here. The pseudorange values are
//! the file's own ASCII decimals parsed to `f64` and carried through unchanged.
//!
//! # Layout (RINEX 3)
//!
//! - Header records are `cols 0..60` content + `cols 60..80` label. The
//!   load-bearing ones are `RINEX VERSION / TYPE` (must be observation),
//!   `APPROX POSITION XYZ`, `ANTENNA: DELTA H/E/N`, `SYS / # / OBS TYPES` (the
//!   per-system code list, order-preserving, with continuation lines),
//!   `SYS / SCALE FACTOR`, `SYS / PHASE SHIFT`, `TIME OF FIRST OBS` (+ time
//!   system), `INTERVAL`, and the optional `GLONASS SLOT / FRQ #`.
//! - The body is per-epoch: a `>`-prefixed epoch line carrying the civil time,
//!   an event flag, and the satellite count, then one logical record per
//!   satellite with each observation as a 16-column `F14.3` value + LLI + SSI
//!   field, in the order the system's `SYS / # / OBS TYPES` list declares. A
//!   logical satellite record may wrap across 80-column continuation lines.
//!
//! # Layout (RINEX 2)
//!
//! - The header uses one global `# / TYPES OF OBSERV` list. Legacy two-character
//!   codes are mapped into the same three-character code strings used by the
//!   RINEX 3 path as each satellite system is encountered.
//! - The body is per-epoch: a fixed-column epoch line with a two-digit year,
//!   event flag, satellite count, and up to twelve inline PRNs. The PRN list
//!   continues on following lines from column 32 when needed.
//! - Each satellite then contributes only observation fields, five per physical
//!   line, with no leading satellite token. Blank value fields are retained as
//!   `None`.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::astro::time::model::TimeScale;

use crate::format::columns::{fixed_record, raw_field as field, raw_field_from};
use crate::format::{Diagnostics, RecordRef, Skip, SkipReason};
use crate::frequencies::{
    rinex_band_frequency_hz, rinex_observation_frequency_hz, rinex_observation_wavelength_m,
};
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::rinex_common::time_scale_label;
use crate::rinex_nav::valid_glonass_frequency_channel;
use crate::validate::{self, FieldError};
use crate::{Error, Result};
use write::{PRN_OBS_COUNTS_COLUMN, PRN_OBS_COUNT_WIDTH, PRN_OBS_SATELLITE_COLUMN};

/// Width of one RINEX-3 observation field (`F14.3` value + LLI + SSI).
const OBS_FIELD_WIDTH: usize = 16;
/// Columns and decimals of the fixed-column numeric header fields the writer
/// re-emits: `INTERVAL` (`F10.3`), the `APPROX POSITION XYZ` and
/// `ANTENNA: DELTA H/E/N` components (`F14.4`), and the `TIME OF FIRST OBS` /
/// `TIME OF LAST OBS` seconds (`F13.7`). A header value outside what its field
/// expresses is rejected rather than written back as a different number.
/// Columns and decimals of the remaining fixed-format numbers the writer
/// re-emits: the version (two decimals in the twenty columns this writer leaves
/// ahead of the file-type field; the specification calls the field `F9.2,11X`),
/// a `GLONASS COD/PHS/BIS` bias (`F8.3`), an epoch record's seconds (`F11.7`)
/// and receiver clock offset (`F15.12`).
const VERSION_WIDTH: usize = 20;
const VERSION_DECIMALS: usize = 2;
const GLONASS_BIAS_WIDTH: usize = 8;
const GLONASS_BIAS_DECIMALS: usize = 3;
const EPOCH_SECOND_WIDTH: usize = 11;
const EPOCH_SECOND_DECIMALS: usize = 7;
const CLOCK_OFFSET_WIDTH: usize = 15;
const CLOCK_OFFSET_DECIMALS: usize = 12;
const OBS_VALUE_DECIMALS: usize = 3;
/// Columns of a RINEX 3 epoch line: the `>` marker, the six date and time
/// fields, the epoch flag, the satellite count, and the receiver clock offset.
/// The `2X` before the flag and the `6X` before the clock are the format's own
/// gaps, and are left out so that content straying into them marks the line as
/// not being in this layout.
const V3_EPOCH_COLUMNS: [(usize, usize); 10] = [
    (0, 1),
    (2, 6),
    (7, 9),
    (10, 12),
    (13, 15),
    (16, 18),
    (18, 29),
    (31, 32),
    (32, 35),
    (41, 56),
];
/// The same line carrying this writer's picosecond field, which it places after
/// the seconds and which shifts every later field six columns right.
const V3_EPOCH_PICOSECOND_COLUMNS: [(usize, usize); 11] = [
    (0, 1),
    (2, 6),
    (7, 9),
    (10, 12),
    (13, 15),
    (16, 18),
    (18, 29),
    (30, 35),
    (37, 38),
    (38, 41),
    (47, 62),
];
/// Columns of a `TIME OF FIRST OBS` / `TIME OF LAST OBS` body, `5I6,F13.7`.
/// The time scale that follows is read from its own columns already.
const TIME_OF_OBS_COLUMNS: [(usize, usize); EPOCH_TIME_TOKENS] =
    [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 43)];

/// Columns of a RINEX 2 epoch line's leading window, up to the satellite list.
/// The clock offset sits past that list and is read separately.
const V2_EPOCH_HEAD_COLUMNS: [(usize, usize); 8] = [
    (1, 3),
    (4, 6),
    (7, 9),
    (10, 12),
    (13, 15),
    (15, 26),
    (28, 29),
    (29, 32),
];

/// Year, month, day, hour, minute and seconds: the tokens every epoch line
/// opens with, before the optional picoseconds and the flag.
const EPOCH_TIME_TOKENS: usize = 6;
const HEADER_VEC3_WIDTH: usize = 14;
const HEADER_VEC3_DECIMALS: usize = 4;
const HEADER_SECOND_WIDTH: usize = 13;
const HEADER_SECOND_DECIMALS: usize = 7;
/// Width of the numeric part of one observation field (`F14.3`).
const OBS_VALUE_WIDTH: usize = 14;
/// Largest record count representable by a RINEX epoch `I3` field.
const MAX_EPOCH_RECORD_COUNT: usize = 999;
/// Width of one observation descriptor in every header code list: the `A3`
/// fields of `SYS / # / OBS TYPES` (`13(1X,A3)`), `SYS / SCALE FACTOR`, and
/// `SYS / PHASE SHIFT`. Legacy RINEX-2 `A2` codes are mapped into the same
/// three-character strings, so this bounds them too.
const OBS_CODE_FIELD_WIDTH: usize = 3;
/// Largest observation-type count representable by the `SYS / # / OBS TYPES`
/// `I3` count field. RINEX-2 `# / TYPES OF OBSERV` lists are re-emitted through
/// that same field, so they share the bound.
const MAX_OBS_TYPE_COUNT: usize = 999;
const HEADER_LABELS: &[&str] = &[
    "RINEX VERSION / TYPE",
    "PGM / RUN BY / DATE",
    "COMMENT",
    "APPROX POSITION XYZ",
    "ANTENNA: DELTA H/E/N",
    "SYS / # / OBS TYPES",
    "# / TYPES OF OBSERV",
    "SYS / SCALE FACTOR",
    "SYS / PHASE SHIFT",
    "TIME OF FIRST OBS",
    "TIME OF LAST OBS",
    "INTERVAL",
    "GLONASS SLOT / FRQ #",
    "GLONASS COD/PHS/BIS",
    "SIGNAL STRENGTH UNIT",
    "LEAP SECONDS",
    "# OF SATELLITES",
    "PRN / # OF OBS",
    "MARKER NAME",
    "MARKER NUMBER",
    "MARKER TYPE",
    "OBSERVER / AGENCY",
    "REC # / TYPE / VERS",
    "ANT # / TYPE",
    "END OF HEADER",
];

/// A civil epoch as it appears on a RINEX observation epoch line, in the file's
/// own time scale (no leap-second shifting). This is the natural boundary for
/// the solver, which derives seconds-of-J2000 / second-of-day / day-of-year
/// from the civil components.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ObsEpochTime {
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
    /// Seconds of minute (fractional), 0.0..60.0.
    pub second: f64,
}

/// One reconstructed observation: a value (or blank) with its loss-of-lock and
/// signal-strength indicators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObsValue {
    /// The observed value (meters for code/`C` observables, cycles for `L`,
    /// etc.), or `None` when the field was blank.
    pub value: Option<f64>,
    /// Loss-of-lock indicator (RINEX LLI), `None` when blank.
    pub lli: Option<u8>,
    /// Signal-strength indicator (RINEX SSI), `None` when blank.
    pub ssi: Option<u8>,
}

/// One `SYS / PHASE SHIFT` header record.
#[derive(Debug, Clone, PartialEq)]
pub struct ObsPhaseShift {
    /// Constellation the phase-shift record applies to.
    pub system: GnssSystem,
    /// RINEX carrier observable code, e.g. `L1C`.
    pub code: String,
    /// Phase correction in carrier cycles.
    pub correction_cycles: f64,
    /// Optional satellite restriction. Empty means the correction applies to
    /// all satellites of the system/code.
    pub satellites: Vec<GnssSatelliteId>,
}

/// One `SYS / SCALE FACTOR` header record.
#[derive(Debug, Clone, PartialEq)]
pub struct ObsScaleFactor {
    /// Constellation the scale-factor record applies to.
    pub system: GnssSystem,
    /// Factor to divide stored observations by before use.
    pub factor: f64,
    /// Observation codes affected. Empty means all codes for the system.
    pub codes: Vec<String>,
}

/// One `PGM / RUN BY / DATE` header record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PgmRunByDate {
    /// Program name, trimmed from A20.
    pub program: String,
    /// Run-by agency/user, trimmed from A20.
    pub run_by: String,
    /// Date string, trimmed from A20.
    pub date: String,
}

/// One `REC # / TYPE / VERS` header record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiverInfo {
    /// Receiver serial number, trimmed from A20.
    pub number: String,
    /// Receiver type, trimmed from A20.
    pub receiver_type: String,
    /// Receiver firmware/version, trimmed from A20.
    pub version: String,
}

/// One `ANT # / TYPE` header record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AntennaInfo {
    /// Antenna serial number, trimmed from A20.
    pub number: String,
    /// Antenna type, trimmed from A20.
    pub antenna_type: String,
}

/// `LEAP SECONDS` header record retained from an observation file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObsLeapSeconds {
    /// Current leap-second count.
    pub current: i64,
    /// Future/past delta field, if present.
    pub delta_future: Option<i64>,
    /// GPS week field, if present.
    pub week: Option<i64>,
    /// Day field, if present.
    pub day: Option<i64>,
}

/// One epoch record: the civil time, the event flag, and the per-satellite
/// observation values (aligned to that system's `SYS / # / OBS TYPES` order).
#[derive(Debug, Clone, PartialEq)]
pub struct ObsEpoch {
    /// Civil epoch in the header time scale.
    pub epoch: ObsEpochTime,
    /// Epoch flag: 0 = OK, 1 = power failure, >1 = an event record, whose own
    /// records are in [`ObsEpoch::special_records`].
    pub flag: u8,
    /// Optional receiver clock offset from the epoch line, seconds.
    pub rcv_clock_offset_s: Option<f64>,
    /// Optional RINEX 4 epoch picosecond extension.
    pub epoch_picoseconds: Option<u32>,
    /// Satellite/special-record count declared on the epoch line.
    pub declared_record_count: usize,
    /// The records an event epoch (flag above 1) carried, as they were written.
    ///
    /// A flag 3 epoch is followed by the header records for a new site
    /// occupation - its marker, antenna and position - and a flag 4 epoch by
    /// comments. They are kept verbatim rather than parsed, because what they
    /// mean depends on the labels they carry, and written back unchanged. Empty
    /// for an ordinary observation epoch.
    pub special_records: Vec<String>,
    /// Satellite → observation values, ascending satellite id. The value vector
    /// is index-aligned to [`ObsHeader::obs_codes`] for that satellite's system.
    pub sats: BTreeMap<GnssSatelliteId, Vec<ObsValue>>,
}

/// Parsed RINEX observation header.
#[derive(Debug, Clone, PartialEq)]
pub struct ObsHeader {
    /// The full RINEX version (e.g. `2.11`, `3.05`, or `4.02`).
    pub version: f64,
    /// The surveyed a-priori receiver position (ECEF meters), if the file
    /// carries an `APPROX POSITION XYZ` record.
    pub approx_position_m: Option<[f64; 3]>,
    /// Antenna reference-point offset from the marker in the RINEX
    /// height/east/north convention (meters), if the file carries an
    /// `ANTENNA: DELTA H/E/N` record.
    pub antenna_delta_hen_m: Option<[f64; 3]>,
    /// Per-constellation observation-code list, in declared order.
    pub obs_codes: BTreeMap<GnssSystem, Vec<String>>,
    /// Program/run-by/date header record.
    pub program_run_by_date: Option<PgmRunByDate>,
    /// Header comments retained in file order.
    pub comments: Vec<String>,
    /// Marker number, if present.
    pub marker_number: Option<String>,
    /// Marker type, if present.
    pub marker_type: Option<String>,
    /// Observer name, if present.
    pub observer: Option<String>,
    /// Agency name, if present.
    pub agency: Option<String>,
    /// Receiver information, if present.
    pub receiver: Option<ReceiverInfo>,
    /// Antenna information, if present.
    pub antenna: Option<AntennaInfo>,
    /// Nominal epoch spacing in seconds (`INTERVAL`), if present.
    ///
    /// RINEX permits zero when this optional metadata is unknown. Consumers
    /// must not use a non-positive value as cadence.
    pub interval_s: Option<f64>,
    /// First observation epoch and its time system (`TIME OF FIRST OBS`).
    pub time_of_first_obs: Option<(ObsEpochTime, TimeScale)>,
    /// Last observation epoch and its time system (`TIME OF LAST OBS`).
    pub time_of_last_obs: Option<(ObsEpochTime, TimeScale)>,
    /// Declared distinct-satellite count.
    pub n_satellites: Option<usize>,
    /// Declared per-satellite, per-code observation counts.
    pub prn_obs_counts: BTreeMap<GnssSatelliteId, Vec<Option<usize>>>,
    /// Carrier phase-shift records (`SYS / PHASE SHIFT`), in header order.
    pub phase_shifts: Vec<ObsPhaseShift>,
    /// Observation scale-factor records (`SYS / SCALE FACTOR`), in header order.
    pub scale_factors: Vec<ObsScaleFactor>,
    /// GLONASS slot → frequency channel map (`GLONASS SLOT / FRQ #`), if present.
    pub glonass_slots: BTreeMap<u8, i8>,
    /// GLONASS code-phase bias/alignment record.
    pub glonass_cod_phs_bis: Option<Vec<(String, f64)>>,
    /// Signal-strength unit, e.g. `DBHZ`.
    pub signal_strength_unit: Option<String>,
    /// Observation-header leap-second record.
    pub leap_seconds: Option<ObsLeapSeconds>,
    /// Marker (station) name, if present.
    pub marker_name: Option<String>,
    /// Header labels retained only as drop-on-rewrite disclosure.
    pub unretained_header_labels: Vec<String>,
}

/// A parsed RINEX observation product.
///
/// Construct with [`RinexObs::parse`]. Epochs are stored in file order; access
/// the header via [`RinexObs::header`], the epochs via [`RinexObs::epochs`], and
/// per-system code lists via [`RinexObs::obs_codes`].
#[derive(Debug, Clone, PartialEq)]
pub struct RinexObs {
    /// The parsed header.
    pub header: ObsHeader,
    /// Epoch records in file order. Event records (flag > 1) are retained with
    /// an empty satellite map so epoch indices stay stable.
    pub epochs: Vec<ObsEpoch>,
    /// Count of records skipped because their satellite token did not parse to a
    /// representable [`GnssSatelliteId`]: an out-of-range entry in the `GLONASS
    /// SLOT / FRQ #` header table, or an unknown/out-of-range satellite record
    /// inside an epoch (e.g. an extended GLONASS slot like `R28` beyond the
    /// engine's PRN cap). One such record is skipped rather than aborting the
    /// whole file, mirroring [`crate::astro::sgp4::TleFile::skipped`].
    pub skipped_records: usize,
}

impl RinexObs {
    /// Parse RINEX observation text into a typed product.
    ///
    /// Returns [`Error::Parse`] if the file is not observation data, is not RINEX
    /// major version 2, 3, or 4, is missing a required header record, or has a malformed
    /// epoch record.
    pub fn parse(text: &str) -> Result<Self> {
        let mut parser = Parser::new();
        let mut lines = text.lines();
        parser.parse_header(&mut lines)?;
        // A version 2 product is re-emitted through the version 3 record writer,
        // so its own output declares version 2 while carrying `>` epoch records.
        // Dispatch on the first record, which is unambiguous: a version 2 epoch
        // line begins with its two-digit year, never with `>`. Without this the
        // writer produces a file it cannot read back. Only the first record is
        // consulted, never anything past it: a version 2 event record carries
        // free text of its own, which may itself begin with `>`.
        let mut body = lines.peekable();
        while body.peek().is_some_and(|line| line.trim().is_empty()) {
            body.next();
        }
        let carries_version_three_records = body.peek().is_some_and(|line| line.starts_with('>'));
        if parser.is_rinex2() && !carries_version_three_records {
            parser.parse_body_v2(&mut body)?;
        } else {
            parser.parse_body(&mut body)?;
        }
        parser.finish()
    }

    /// The parsed header.
    pub fn header(&self) -> &ObsHeader {
        &self.header
    }

    /// The epoch records, in file order.
    pub fn epochs(&self) -> &[ObsEpoch] {
        &self.epochs
    }

    /// The observation-code list for a constellation, in declared order.
    pub fn obs_codes(&self, sys: GnssSystem) -> Option<&[String]> {
        self.header.obs_codes.get(&sys).map(Vec::as_slice)
    }
}

impl core::str::FromStr for RinexObs {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

/// Per-system single-frequency code-selection policy.
///
/// For each constellation, an ordered list of observation codes to try; the
/// first one present at an epoch is used. Build the version-aware defaults with
/// [`SignalPolicy::default_for`] and adjust per system with
/// [`SignalPolicy::with_override`].
#[derive(Debug, Clone, PartialEq)]
pub struct SignalPolicy {
    /// Ordered preference list of observation codes per constellation.
    pub codes: BTreeMap<GnssSystem, Vec<String>>,
}

impl SignalPolicy {
    /// The default single-frequency pseudorange policy:
    ///
    /// - GPS `C1C` (L1 C/A),
    /// - Galileo `C1C` then `C1X` (E1),
    /// - BeiDou `C1I` for RINEX 3.02, `C2I` for 3.01 and 3.03+ (the B1I code
    ///   label changed between minor versions),
    /// - GLONASS `C1C` (G1 C/A).
    ///
    /// `version` is the file's RINEX version, which selects the BeiDou default.
    pub fn default_for(version: f64) -> Result<Self> {
        validate_finite_input(version, "version")?;
        let mut codes = BTreeMap::new();
        codes.insert(GnssSystem::Gps, vec!["C1C".to_string()]);
        codes.insert(
            GnssSystem::Galileo,
            vec!["C1C".to_string(), "C1X".to_string()],
        );
        // BeiDou B1I label history: C2I in 3.01, relabelled band 1 (C1I) in
        // 3.02, then reverted to C2I in 3.03 and later. Only the narrow 3.02
        // window prefers C1I; every other version prefers C2I. Offer both, with
        // the version-appropriate one first.
        let beidou = if (3.015..3.025).contains(&version) {
            vec!["C1I".to_string(), "C2I".to_string()]
        } else {
            vec!["C2I".to_string(), "C1I".to_string()]
        };
        codes.insert(GnssSystem::BeiDou, beidou);
        codes.insert(GnssSystem::Glonass, vec!["C1C".to_string()]);
        Ok(Self { codes })
    }

    /// Replace the preference list for one constellation.
    pub fn with_override(mut self, sys: GnssSystem, codes: Vec<String>) -> Self {
        self.codes.insert(sys, codes);
        self
    }
}

/// Optional per-system observation-code filter.
///
/// An empty filter keeps every parsed system and code. A non-empty filter keeps
/// only listed systems; for each listed system, an empty code vector keeps every
/// code while a non-empty vector keeps only those codes, in header order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ObservationFilter {
    /// Per-constellation code allow-list.
    pub codes: BTreeMap<GnssSystem, Vec<String>>,
}

impl ObservationFilter {
    /// Construct an empty filter that keeps every parsed observation.
    pub fn all() -> Self {
        Self::default()
    }

    /// Construct a filter from `(system, codes)` entries.
    pub fn from_entries<I>(entries: I) -> Self
    where
        I: IntoIterator<Item = (GnssSystem, Vec<String>)>,
    {
        Self {
            codes: entries.into_iter().collect(),
        }
    }

    fn allowed_codes(&self, system: GnssSystem) -> Option<&[String]> {
        if self.codes.is_empty() {
            Some(&[])
        } else {
            self.codes.get(&system).map(Vec::as_slice)
        }
    }
}

/// Observation kind inferred from the RINEX observation-code leading letter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationKind {
    /// Code pseudorange (`C*`), meters.
    Pseudorange,
    /// Carrier phase (`L*`), cycles.
    CarrierPhase,
    /// Doppler (`D*`), hertz.
    Doppler,
    /// Signal strength (`S*`), dB-Hz.
    SignalStrength,
    /// Unknown or unsupported leading code letter.
    Unknown,
}

impl ObservationKind {
    /// Infer the kind from a RINEX observation code.
    pub fn from_code(code: &str) -> Self {
        match code.as_bytes().first().copied() {
            Some(b'C') => Self::Pseudorange,
            Some(b'L') => Self::CarrierPhase,
            Some(b'D') => Self::Doppler,
            Some(b'S') => Self::SignalStrength,
            _ => Self::Unknown,
        }
    }

    /// Stable lower-case API label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pseudorange => "pseudorange",
            Self::CarrierPhase => "carrier_phase",
            Self::Doppler => "doppler",
            Self::SignalStrength => "signal_strength",
            Self::Unknown => "unknown",
        }
    }

    /// Stable units label for the observation kind.
    pub fn units_str(self) -> &'static str {
        match self {
            Self::Pseudorange => "meters",
            Self::CarrierPhase => "cycles",
            Self::Doppler => "hz",
            Self::SignalStrength => "db_hz",
            Self::Unknown => "unknown",
        }
    }
}

/// One labelled raw RINEX observation value.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationValueRow {
    /// RINEX observation code, e.g. `C1C`, `L2W`, `D1C`.
    pub code: String,
    /// Kind inferred from the code's leading letter.
    pub kind: ObservationKind,
    /// Parsed observation value, or `None` for a blank field.
    pub value: Option<f64>,
    /// RINEX loss-of-lock indicator.
    pub lli: Option<u8>,
    /// RINEX signal-strength indicator.
    pub ssi: Option<u8>,
}

/// One carrier-phase observation with its carrier metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CarrierPhaseRow {
    /// RINEX carrier observation code, e.g. `L1C`.
    pub code: String,
    /// Phase in cycles as recorded in the RINEX observation body.
    pub value_cycles: Option<f64>,
    /// RINEX loss-of-lock indicator.
    pub lli: Option<u8>,
    /// RINEX signal-strength indicator.
    pub ssi: Option<u8>,
    /// Carrier frequency in hertz when known.
    pub frequency_hz: Option<f64>,
    /// Carrier wavelength in meters when known.
    pub wavelength_m: Option<f64>,
    /// Carrier phase in meters when both value and frequency are known.
    pub value_m: Option<f64>,
    /// Reported `SYS / PHASE SHIFT` correction in cycles. RINEX 3 stores
    /// already-aligned phase observations, so this correction is metadata for
    /// reconstructing originals and is not re-applied here.
    pub phase_shift_cycles: f64,
}

/// Return labelled raw observation rows for one epoch, grouped by satellite.
pub fn observation_values(
    obs: &RinexObs,
    epoch: &ObsEpoch,
    filter: &ObservationFilter,
) -> Result<Vec<(GnssSatelliteId, Vec<ObservationValueRow>)>> {
    let mut out = Vec::new();
    for (sat, values) in epoch
        .sats
        .iter()
        .filter(|(sat, _)| filter.allowed_codes(sat.system).is_some())
    {
        let Some(allowed_codes) = filter.allowed_codes(sat.system) else {
            continue;
        };
        let Some(code_list) = obs.header.obs_codes.get(&sat.system) else {
            continue;
        };
        let mut rows = Vec::new();
        for (code, value) in code_list.iter().zip(values.iter()) {
            if !allowed_codes.is_empty() && !allowed_codes.iter().any(|c| c == code) {
                continue;
            }
            if let Some(value) = value.value {
                validate_finite_input(value, "observation.value")?;
            }
            let kind = ObservationKind::from_code(code);
            rows.push(ObservationValueRow {
                code: code.clone(),
                kind,
                value: value.value,
                lli: value.lli,
                ssi: value.ssi,
            });
        }
        out.push((*sat, rows));
    }
    Ok(out)
}

/// Return carrier-phase rows for one epoch, grouped by satellite.
pub fn carrier_phase_rows(
    obs: &RinexObs,
    epoch: &ObsEpoch,
    filter: &ObservationFilter,
) -> Result<Vec<(GnssSatelliteId, Vec<CarrierPhaseRow>)>> {
    validate_finite_input(obs.header.version, "version")?;
    let mut out = Vec::new();
    for (sat, rows) in observation_values(obs, epoch, filter)? {
        let phases = rows
            .into_iter()
            .filter(|row| row.kind == ObservationKind::CarrierPhase)
            .map(|row| carrier_phase_row(obs, sat, row))
            .collect::<Result<Vec<_>>>()?;
        out.push((sat, phases));
    }
    Ok(out)
}

/// Carrier frequency in hertz for a system and RINEX band digit.
///
/// GLONASS G1/G2 carriers require the FDMA channel number from the observation
/// file's `GLONASS SLOT / FRQ #` records.
pub fn band_frequency_hz(
    system: GnssSystem,
    band: char,
    glonass_channel: Option<i8>,
) -> Option<f64> {
    rinex_band_frequency_hz(system, band, glonass_channel)
}

/// Carrier frequency in hertz for a system and full RINEX observation code.
pub fn observation_frequency_hz(
    system: GnssSystem,
    code: &str,
    rinex_version: f64,
    glonass_channel: Option<i8>,
) -> Result<Option<f64>> {
    validate_finite_input(rinex_version, "version")?;
    Ok(rinex_observation_frequency_hz(
        system,
        code,
        rinex_version,
        glonass_channel,
    ))
}

fn carrier_phase_row(
    obs: &RinexObs,
    sat: GnssSatelliteId,
    row: ObservationValueRow,
) -> Result<CarrierPhaseRow> {
    let glonass_channel = obs.header.glonass_slots.get(&sat.prn).copied();
    let frequency_hz =
        observation_frequency_hz(sat.system, &row.code, obs.header.version, glonass_channel)?;
    let phase_shift_cycles = phase_shift_cycles(obs, sat, &row.code);
    let value_cycles = row.value;
    let wavelength_m =
        rinex_observation_wavelength_m(sat.system, &row.code, obs.header.version, glonass_channel);
    let value_m = match value_cycles.zip(wavelength_m) {
        Some((cycles, lambda)) => {
            let value_m = cycles * lambda;
            validate_finite_input(value_m, "carrier_phase.value_m")?;
            Some(value_m)
        }
        None => None,
    };
    Ok(CarrierPhaseRow {
        code: row.code,
        value_cycles,
        lli: row.lli,
        ssi: row.ssi,
        frequency_hz,
        wavelength_m,
        value_m,
        phase_shift_cycles,
    })
}

fn phase_shift_cycles(obs: &RinexObs, sat: GnssSatelliteId, code: &str) -> f64 {
    let mut system_wide = None;
    for shift in obs.header.phase_shifts.iter().rev() {
        if shift.system != sat.system || shift.code != code {
            continue;
        }
        if shift.satellites.is_empty() {
            if system_wide.is_none() {
                system_wide = Some(shift.correction_cycles);
            }
        } else if shift.satellites.contains(&sat) {
            return shift.correction_cycles;
        }
    }
    system_wide.unwrap_or(0.0)
}

/// Extract single-frequency pseudoranges for one epoch under a [`SignalPolicy`].
///
/// For each satellite in the epoch, the first code in that system's preference
/// list whose value is present at the epoch is used. Satellites whose system has
/// no policy entry, or that lack every preferred code, are skipped. The result
/// is the ascending-id `(satellite, range_m)` list the solver consumes.
pub fn pseudoranges(
    obs: &RinexObs,
    epoch: &ObsEpoch,
    policy: &SignalPolicy,
) -> Result<Vec<(GnssSatelliteId, f64)>> {
    let mut out = Vec::new();
    for (sat, values) in &epoch.sats {
        let Some(prefs) = policy.codes.get(&sat.system) else {
            continue;
        };
        let Some(code_list) = obs.header.obs_codes.get(&sat.system) else {
            continue;
        };
        for code in prefs {
            if let Some(idx) = code_list.iter().position(|c| c == code) {
                if let Some(ObsValue {
                    value: Some(range_m),
                    ..
                }) = values.get(idx)
                {
                    validate_finite_input(*range_m, "pseudorange_m")?;
                    out.push((*sat, *range_m));
                    break;
                }
            }
        }
    }
    Ok(out)
}

/// Incremental RINEX 3 observation parser state.
struct Parser {
    version: Option<f64>,
    is_observation: bool,
    approx_position_m: Option<[f64; 3]>,
    antenna_delta_hen_m: Option<[f64; 3]>,
    obs_codes: BTreeMap<GnssSystem, Vec<String>>,
    interval_s: Option<f64>,
    time_of_first_obs: Option<(ObsEpochTime, TimeScale)>,
    time_of_last_obs: Option<(ObsEpochTime, TimeScale)>,
    program_run_by_date: Option<PgmRunByDate>,
    comments: Vec<String>,
    marker_number: Option<String>,
    marker_type: Option<String>,
    observer: Option<String>,
    agency: Option<String>,
    receiver: Option<ReceiverInfo>,
    antenna: Option<AntennaInfo>,
    n_satellites: Option<usize>,
    prn_obs_counts: BTreeMap<GnssSatelliteId, Vec<Option<usize>>>,
    prn_obs_counts_current: Option<GnssSatelliteId>,
    phase_shifts: Vec<ObsPhaseShift>,
    scale_factors: Vec<ObsScaleFactor>,
    scale_factor_continuation: Option<ScaleFactorContinuation>,
    glonass_slots: BTreeMap<u8, i8>,
    glonass_slots_remaining: Option<usize>,
    glonass_cod_phs_bis: Option<Vec<(String, f64)>>,
    signal_strength_unit: Option<String>,
    leap_seconds: Option<ObsLeapSeconds>,
    marker_name: Option<String>,
    unretained_header_labels: Vec<String>,
    epochs: Vec<ObsEpoch>,
    /// The constellation whose `SYS / # / OBS TYPES` list is currently being
    /// filled (for continuation lines).
    current_obs_sys: Option<GnssSystem>,
    /// Number of codes still expected for `current_obs_sys`.
    obs_codes_remaining: usize,
    /// RINEX 2 default system from the version record when it is not mixed.
    rinex2_default_system: Option<GnssSystem>,
    /// Legacy RINEX 2 global observation-code list, before per-system
    /// canonicalization.
    rinex2_obs_codes: Vec<String>,
    /// Number of global RINEX 2 observation codes still expected.
    rinex2_obs_codes_remaining: usize,
    /// Forgiving-parse diagnostics: a GLONASS-slot or epoch satellite record
    /// whose token does not parse to a representable [`GnssSatelliteId`] is
    /// pushed here as a typed [`Skip`] rather than silently dropped. The public
    /// [`RinexObs::skipped_records`] is derived from the skip count.
    diagnostics: Diagnostics,
}

#[derive(Debug, Clone, Copy)]
struct ScaleFactorContinuation {
    remaining: usize,
}

impl Parser {
    fn new() -> Self {
        Self {
            version: None,
            is_observation: false,
            approx_position_m: None,
            antenna_delta_hen_m: None,
            obs_codes: BTreeMap::new(),
            interval_s: None,
            time_of_first_obs: None,
            time_of_last_obs: None,
            program_run_by_date: None,
            comments: Vec::new(),
            marker_number: None,
            marker_type: None,
            observer: None,
            agency: None,
            receiver: None,
            antenna: None,
            n_satellites: None,
            prn_obs_counts: BTreeMap::new(),
            prn_obs_counts_current: None,
            phase_shifts: Vec::new(),
            scale_factors: Vec::new(),
            scale_factor_continuation: None,
            glonass_slots: BTreeMap::new(),
            glonass_slots_remaining: None,
            glonass_cod_phs_bis: None,
            signal_strength_unit: None,
            leap_seconds: None,
            marker_name: None,
            unretained_header_labels: Vec::new(),
            epochs: Vec::new(),
            current_obs_sys: None,
            obs_codes_remaining: 0,
            rinex2_default_system: None,
            rinex2_obs_codes: Vec::new(),
            rinex2_obs_codes_remaining: 0,
            diagnostics: Diagnostics::new(),
        }
    }

    fn is_rinex2(&self) -> bool {
        self.version
            .is_some_and(|version| version.floor() as i64 == 2)
    }

    /// Record a forgiving skip for a record whose satellite token is not a
    /// representable [`GnssSatelliteId`], carrying the raw token as its identity.
    fn push_unrepresentable_satellite_skip(&mut self, token: &str) {
        self.diagnostics.push_skip(Skip {
            at: RecordRef::default().with_satellite(token.trim()),
            reason: SkipReason::UnrepresentableSatellite,
        });
    }

    fn parse_header<'a, I: Iterator<Item = &'a str>>(&mut self, lines: &mut I) -> Result<()> {
        let mut saw_end = false;
        for raw in lines.by_ref() {
            let raw_line = raw.trim_end_matches(['\r', '\n']);
            // RINEX header fields are fixed-width printable ASCII. Keep the
            // parser forgiving, but normalize invalid Unicode and control
            // characters before interpreting byte columns. Otherwise a lossy
            // UTF-8 replacement character can straddle a 20-column boundary,
            // making parse -> write -> parse repartition retained fields.
            let ascii_line = printable_ascii_header_columns(raw_line);
            let line = normalize_header_line(&ascii_line);
            let line = line.as_ref();
            let label = raw_field_from(line, 60).trim();
            match label {
                "RINEX VERSION / TYPE" => self.parse_version(line)?,
                "PGM / RUN BY / DATE" => self.parse_pgm_run_by_date(line),
                "COMMENT" => self.comments.push(field(line, 0, 60).trim().to_string()),
                "APPROX POSITION XYZ" => self.parse_approx_position(line)?,
                "ANTENNA: DELTA H/E/N" => self.parse_antenna_delta(line)?,
                "SYS / # / OBS TYPES" => self.parse_obs_types(line)?,
                "# / TYPES OF OBSERV" => self.parse_obs_types_v2(line)?,
                "SYS / SCALE FACTOR" => self.parse_scale_factor(line)?,
                "SYS / PHASE SHIFT" => self.parse_phase_shift(line)?,
                "TIME OF FIRST OBS" => self.parse_time_of_first_obs(line)?,
                "TIME OF LAST OBS" => self.parse_time_of_last_obs(line)?,
                "INTERVAL" => {
                    self.interval_s = optional_f64_field(line, 0, 10, "interval_s")?
                        .map(|interval_s| {
                            exact_in_field(
                                interval_s,
                                crate::rinex_common::OBS_INTERVAL_WIDTH,
                                crate::rinex_common::OBS_INTERVAL_DECIMALS,
                                "interval_s",
                                line,
                            )
                        })
                        .transpose()?;
                }
                "GLONASS SLOT / FRQ #" => self.parse_glonass_slots(line)?,
                "GLONASS COD/PHS/BIS" => self.parse_glonass_cod_phs_bis(line)?,
                "SIGNAL STRENGTH UNIT" => {
                    let unit = field(line, 0, 20).trim();
                    if !unit.is_empty() {
                        self.signal_strength_unit = Some(unit.to_string());
                    }
                }
                "LEAP SECONDS" => self.parse_leap_seconds(line)?,
                "# OF SATELLITES" => {
                    self.n_satellites =
                        Some(strict_int_field::<usize>(line, 0, 6, "n_satellites")?);
                }
                "PRN / # OF OBS" => self.parse_prn_obs_counts(line)?,
                "MARKER NAME" => {
                    let name = field(line, 0, 60).trim();
                    if !name.is_empty() {
                        self.marker_name = Some(name.to_string());
                    }
                }
                "MARKER NUMBER" => {
                    self.marker_number = optional_trimmed(line, 0, 20);
                }
                "MARKER TYPE" => {
                    self.marker_type = optional_trimmed(line, 0, 20);
                }
                "OBSERVER / AGENCY" => {
                    self.observer = optional_trimmed(line, 0, 20);
                    self.agency = optional_trimmed(line, 20, 60);
                }
                "REC # / TYPE / VERS" => {
                    self.receiver = Some(ReceiverInfo {
                        number: field(line, 0, 20).trim().to_string(),
                        receiver_type: field(line, 20, 40).trim().to_string(),
                        version: field(line, 40, 60).trim().to_string(),
                    });
                }
                "ANT # / TYPE" => {
                    self.antenna = Some(AntennaInfo {
                        number: field(line, 0, 20).trim().to_string(),
                        antenna_type: field(line, 20, 40).trim().to_string(),
                    });
                }
                "END OF HEADER" => {
                    self.ensure_obs_type_count_complete(line)?;
                    self.ensure_obs_type_count_complete_v2(line)?;
                    self.ensure_scale_factor_count_complete(line)?;
                    saw_end = true;
                    break;
                }
                // Every other header record is tolerated and surfaced to QC so
                // callers know a rewrite will not carry it.
                _ => {
                    if !label.is_empty() {
                        self.unretained_header_labels.push(label.to_string());
                    }
                }
            }
        }
        if !saw_end {
            return Err(Error::Parse("RINEX OBS header has no END OF HEADER".into()));
        }
        Ok(())
    }

    fn parse_version(&mut self, line: &str) -> Result<()> {
        let version_field = field(line, 0, 20).trim();
        let version = strict_f64_token(version_field, "version", line).or_else(|_| {
            let token = field(line, 0, 60)
                .split_whitespace()
                .next()
                .ok_or_else(|| Error::Parse(format!("RINEX OBS bad version field in {line:?}")))?;
            strict_f64_token(token, "version", line)
        })?;
        let version = exact_in_field(version, VERSION_WIDTH, VERSION_DECIMALS, "version", line)?;
        // The file type letter is at column 20; observation files carry 'O'.
        let type_field = field(line, 20, 40);
        let body = field(line, 0, 60);
        self.is_observation = type_field.trim_start().starts_with('O')
            || type_field.contains("OBSERVATION")
            || body.contains("OBSERVATION")
            || body.split_whitespace().any(|token| token == "O");
        if !self.is_observation {
            return Err(Error::Parse(format!(
                "RINEX file is not observation data: {type_field:?}"
            )));
        }
        if !matches!(version.floor() as i64, 2..=4) {
            return Err(Error::Parse(format!(
                "RINEX OBS parser requires major version 2, 3, or 4, got {version}"
            )));
        }
        if version.floor() as i64 == 2 {
            let system_field = field(line, 40, 41).trim();
            if let Some(letter) = system_field.chars().next().filter(|letter| *letter != 'M') {
                self.rinex2_default_system = GnssSystem::from_letter(letter);
            }
        }
        self.version = Some(version);
        Ok(())
    }

    fn parse_approx_position(&mut self, line: &str) -> Result<()> {
        let body = field(line, 0, 60);
        self.approx_position_m = Some(strict_vec3_tokens(
            body,
            line,
            [
                "approx_position.x_m",
                "approx_position.y_m",
                "approx_position.z_m",
            ],
        )?);
        Ok(())
    }

    fn parse_antenna_delta(&mut self, line: &str) -> Result<()> {
        let body = field(line, 0, 60);
        self.antenna_delta_hen_m = Some(strict_vec3_tokens(
            body,
            line,
            [
                "antenna_delta.height_m",
                "antenna_delta.east_m",
                "antenna_delta.north_m",
            ],
        )?);
        Ok(())
    }

    fn parse_pgm_run_by_date(&mut self, line: &str) {
        self.program_run_by_date = Some(PgmRunByDate {
            program: field(line, 0, 20).trim().to_string(),
            run_by: field(line, 20, 40).trim().to_string(),
            date: field(line, 40, 60).trim().to_string(),
        });
    }

    fn parse_obs_types(&mut self, line: &str) -> Result<()> {
        // A new system line carries its letter at column 0 and the count at
        // columns 3..6; a continuation line has a blank system field and only
        // adds more codes to the current system.
        let sys_field = field(line, 0, 1).trim();
        if !sys_field.is_empty() {
            let count = match strict_int_field::<usize>(line, 3, 6, "obs_type_count") {
                Ok(count) => count,
                Err(_) => return self.parse_obs_types_whitespace(line),
            };
            self.ensure_obs_type_count_complete(line)?;
            let letter = sys_field
                .chars()
                .next()
                .ok_or_else(|| Error::Parse("RINEX OBS missing system letter".to_string()))?;
            let system = GnssSystem::from_letter(letter).ok_or_else(|| {
                Error::Parse(format!("RINEX OBS unknown system letter {letter:?}"))
            })?;
            self.ensure_obs_type_count_fits(system, count, line)?;
            self.current_obs_sys = Some(system);
            self.obs_codes_remaining = count;
            self.obs_codes.entry(system).or_default();
        }
        let Some(system) = self.current_obs_sys else {
            return Ok(());
        };
        // Codes occupy 4-wide fields (" CCC") from column 7; collect up to the
        // remaining count.
        let codes_section = field(line, 7, 60);
        let Some(list) = self.obs_codes.get_mut(&system) else {
            return Err(Error::Parse(format!(
                "RINEX OBS observation-code system {system} was not inserted"
            )));
        };
        for tok in codes_section.split_whitespace() {
            if self.obs_codes_remaining == 0 {
                return Err(Error::Parse(format!(
                    "RINEX OBS {system} SYS / # / OBS TYPES lists more codes than declared in {line:?}"
                )));
            }
            list.push(obs_code_token(tok, "SYS / # / OBS TYPES", line)?);
            self.obs_codes_remaining -= 1;
        }
        Ok(())
    }

    fn parse_obs_types_v2(&mut self, line: &str) -> Result<()> {
        if field(line, 0, 6).trim().is_empty() {
            if self.rinex2_obs_codes_remaining == 0 {
                return Ok(());
            }
        } else {
            self.ensure_obs_type_count_complete_v2(line)?;
            let count = strict_int_field::<usize>(line, 0, 6, "rinex2.obs_type_count")?;
            if count > MAX_OBS_TYPE_COUNT {
                return Err(Error::Parse(format!(
                    "RINEX OBS # / TYPES OF OBSERV declares {count} codes, exceeding the {MAX_OBS_TYPE_COUNT} the SYS / # / OBS TYPES I3 field can carry, in {line:?}"
                )));
            }
            self.rinex2_obs_codes.clear();
            self.rinex2_obs_codes_remaining = count;
        }
        // `9(4X,A2)`: a code sits in the last two of its six columns and the
        // four before it are blank. A token wider than the field means the line
        // is not in that layout, and such a code could not be written back into
        // two columns: it would read again as something else. Reading the two
        // columns and ignoring what sits beside them would instead drop the
        // rest of the token without saying so.
        let mut tokens = Vec::new();
        for token in
            field(line, OBS_TYPE_V2_COUNT_WIDTH, write::HEADER_CONTENT_WIDTH).split_whitespace()
        {
            let code = obs_code_token(token, "# / TYPES OF OBSERV", line)?;
            if code.len() > OBS_TYPE_V2_WIDTH {
                return Err(Error::Parse(format!(
                    "RINEX OBS # / TYPES OF OBSERV code {code:?} exceeds the A3 field width it is written in, in {line:?}"
                )));
            }
            tokens.push(code);
        }
        for code in tokens {
            if self.rinex2_obs_codes_remaining == 0 {
                return Err(Error::Parse(format!(
                    "RINEX OBS # / TYPES OF OBSERV lists more codes than declared in {line:?}"
                )));
            }
            self.rinex2_obs_codes.push(code);
            self.rinex2_obs_codes_remaining -= 1;
        }
        Ok(())
    }

    fn parse_obs_types_whitespace(&mut self, line: &str) -> Result<()> {
        let tokens: Vec<&str> = field(line, 0, 60).split_whitespace().collect();
        if tokens.is_empty() {
            return Err(Error::Parse(format!(
                "RINEX OBS malformed SYS / # / OBS TYPES record: {line:?}"
            )));
        }

        if tokens.len() >= 2 && tokens[0].len() == 1 {
            if let Ok(count) = strict_int_token::<usize>(tokens[1], "obs_type_count", line) {
                let letter = tokens[0]
                    .chars()
                    .next()
                    .ok_or_else(|| Error::Parse("RINEX OBS missing system letter".into()))?;
                let system = GnssSystem::from_letter(letter).ok_or_else(|| {
                    Error::Parse(format!("RINEX OBS unknown system letter {letter:?}"))
                })?;
                self.ensure_obs_type_count_complete(line)?;
                self.ensure_obs_type_count_fits(system, count, line)?;
                self.current_obs_sys = Some(system);
                self.obs_codes_remaining = count;
                self.obs_codes.entry(system).or_default();
                return self.push_obs_type_tokens(system, &tokens[2..], line);
            }
        }

        let Some(system) = self.current_obs_sys else {
            return Err(Error::Parse(format!(
                "RINEX OBS malformed SYS / # / OBS TYPES record: {line:?}"
            )));
        };
        if self.obs_codes_remaining == 0 {
            return Err(Error::Parse(format!(
                "RINEX OBS {system} SYS / # / OBS TYPES lists more codes than declared in {line:?}"
            )));
        }
        self.push_obs_type_tokens(system, &tokens, line)
    }

    fn push_obs_type_tokens(
        &mut self,
        system: GnssSystem,
        codes: &[&str],
        line: &str,
    ) -> Result<()> {
        let list = self.obs_codes.entry(system).or_default();
        for code in codes {
            if self.obs_codes_remaining == 0 {
                return Err(Error::Parse(format!(
                    "RINEX OBS {system} SYS / # / OBS TYPES lists more codes than declared in {line:?}"
                )));
            }
            list.push(obs_code_token(code, "SYS / # / OBS TYPES", line)?);
            self.obs_codes_remaining -= 1;
        }
        Ok(())
    }

    /// Reject a `SYS / # / OBS TYPES` count - including codes already collected
    /// for the system - that the record's `I3` count field cannot carry.
    fn ensure_obs_type_count_fits(
        &self,
        system: GnssSystem,
        count: usize,
        line: &str,
    ) -> Result<()> {
        let collected = self.obs_codes.get(&system).map_or(0, Vec::len);
        let total = collected.saturating_add(count);
        if total > MAX_OBS_TYPE_COUNT {
            return Err(Error::Parse(format!(
                "RINEX OBS {system} SYS / # / OBS TYPES declares {total} codes, exceeding the I3 field maximum of {MAX_OBS_TYPE_COUNT} in {line:?}"
            )));
        }
        Ok(())
    }

    fn ensure_obs_type_count_complete(&self, line: &str) -> Result<()> {
        if self.obs_codes_remaining == 0 {
            return Ok(());
        }
        let Some(system) = self.current_obs_sys else {
            return Ok(());
        };
        let supplied = self.obs_codes.get(&system).map_or(0, Vec::len);
        let declared = supplied + self.obs_codes_remaining;
        Err(Error::Parse(format!(
            "RINEX OBS {system} SYS / # / OBS TYPES declares {declared} codes but supplies {supplied} before {line:?}"
        )))
    }

    fn ensure_obs_type_count_complete_v2(&self, line: &str) -> Result<()> {
        if self.rinex2_obs_codes_remaining == 0 {
            return Ok(());
        }
        let supplied = self.rinex2_obs_codes.len();
        let declared = supplied + self.rinex2_obs_codes_remaining;
        Err(Error::Parse(format!(
            "RINEX OBS # / TYPES OF OBSERV declares {declared} codes but supplies {supplied} before {line:?}"
        )))
    }

    fn parse_phase_shift(&mut self, line: &str) -> Result<()> {
        let tokens: Vec<&str> = field(line, 0, 60).split_whitespace().collect();
        if tokens.is_empty() {
            return Ok(());
        }
        if tokens.len() < 2 {
            return Err(Error::Parse(format!(
                "RINEX OBS phase-shift header has too few fields in {line:?}"
            )));
        }

        let system = tokens[0]
            .chars()
            .next()
            .and_then(GnssSystem::from_letter)
            .ok_or_else(|| {
                Error::Parse(format!(
                    "RINEX OBS phase-shift system unparsable in {line:?}"
                ))
            })?;
        let code = obs_code_token(tokens[1], "SYS / PHASE SHIFT", line)?;
        let correction_cycles = match tokens.get(2) {
            Some(token) => strict_f64_token(token, "phase_shift.correction_cycles", line)?,
            None => 0.0,
        };

        let satellites = if let Some(count_token) = tokens.get(3) {
            let count =
                strict_int_token::<usize>(count_token, "phase_shift.satellite_count", line)?;
            let sat_tokens = &tokens[4..];
            if sat_tokens.len() != count {
                return Err(Error::Parse(format!(
                    "RINEX OBS phase-shift satellite count mismatch in {line:?}"
                )));
            }
            sat_tokens
                .iter()
                .map(|token| {
                    parse_sv_token(token).ok_or_else(|| {
                        Error::Parse(format!(
                            "RINEX OBS phase-shift satellite token {token:?} unparsable in {line:?}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

        let shift = ObsPhaseShift {
            system,
            code,
            correction_cycles,
            satellites,
        };
        // Satellite tokens may be written more compactly than the `1X,A3` fields
        // they are re-emitted in ("G1" for "G01"), so a record the parser can
        // read is not always one it can write back. Reject what would overrun
        // the content area rather than truncate it into a shorter list.
        let content_width = write::phase_shift_content(&shift).len();
        if content_width > write::HEADER_CONTENT_WIDTH {
            return Err(Error::Parse(format!(
                "RINEX OBS SYS / PHASE SHIFT record needs {content_width} columns, exceeding the {} a header record carries, in {line:?}",
                write::HEADER_CONTENT_WIDTH
            )));
        }
        self.phase_shifts.push(shift);
        Ok(())
    }

    fn parse_scale_factor(&mut self, line: &str) -> Result<()> {
        let sys_field = field(line, 0, 1).trim();
        if !sys_field.is_empty() {
            self.ensure_scale_factor_count_complete(line)?;
            let letter = sys_field
                .chars()
                .next()
                .ok_or_else(|| Error::Parse("RINEX OBS missing scale-factor system".to_string()))?;
            let system = GnssSystem::from_letter(letter).ok_or_else(|| {
                Error::Parse(format!("RINEX OBS unknown scale-factor system {letter:?}"))
            })?;
            let factor =
                scale_factor_value(strict_int_field::<u32>(line, 2, 6, "scale_factor.factor")?)?;
            let count_field = field(line, 8, 10).trim();
            let count = if count_field.is_empty() {
                0
            } else {
                strict_int_token::<usize>(count_field, "scale_factor.obs_type_count", line)?
            };
            self.scale_factors.push(ObsScaleFactor {
                system,
                factor,
                codes: Vec::new(),
            });
            if count == 0 {
                return Ok(());
            }
            self.scale_factor_continuation = Some(ScaleFactorContinuation { remaining: count });
        }

        self.collect_scale_factor_codes(line)
    }

    fn collect_scale_factor_codes(&mut self, line: &str) -> Result<()> {
        let Some(mut continuation) = self.scale_factor_continuation else {
            return Ok(());
        };
        let Some(record) = self.scale_factors.last_mut() else {
            return Err(Error::Parse(
                "RINEX OBS scale-factor continuation has no record".to_string(),
            ));
        };
        for code in field(line, 10, 60).split_whitespace() {
            if continuation.remaining == 0 {
                return Err(Error::Parse(format!(
                    "RINEX OBS SYS / SCALE FACTOR lists more codes than declared in {line:?}"
                )));
            }
            record
                .codes
                .push(obs_code_token(code, "SYS / SCALE FACTOR", line)?);
            continuation.remaining -= 1;
        }
        self.scale_factor_continuation = (continuation.remaining > 0).then_some(continuation);
        Ok(())
    }

    fn ensure_scale_factor_count_complete(&self, line: &str) -> Result<()> {
        let Some(continuation) = self.scale_factor_continuation else {
            return Ok(());
        };
        let supplied = self
            .scale_factors
            .last()
            .map_or(0, |record| record.codes.len());
        let declared = supplied + continuation.remaining;
        Err(Error::Parse(format!(
            "RINEX OBS SYS / SCALE FACTOR declares {declared} codes but supplies {supplied} before {line:?}"
        )))
    }

    fn parse_time_of_first_obs(&mut self, line: &str) -> Result<()> {
        self.time_of_first_obs = Some(self.parse_time_header(line, "time_of_first_obs")?);
        Ok(())
    }

    fn parse_time_of_last_obs(&mut self, line: &str) -> Result<()> {
        self.time_of_last_obs = Some(self.parse_time_header(line, "time_of_last_obs")?);
        Ok(())
    }

    fn parse_time_header(
        &self,
        line: &str,
        prefix: &'static str,
    ) -> Result<(ObsEpochTime, TimeScale)> {
        let body = field(line, 0, 43);
        let scale_label = field(line, 48, 51).trim();
        let scale = time_scale_from_label(scale_label, line)?;
        let year = match prefix {
            "time_of_last_obs" => "time_of_last_obs.year",
            _ => "time_of_first_obs.year",
        };
        let month = match prefix {
            "time_of_last_obs" => "time_of_last_obs.month",
            _ => "time_of_first_obs.month",
        };
        let day = match prefix {
            "time_of_last_obs" => "time_of_last_obs.day",
            _ => "time_of_first_obs.day",
        };
        let hour = match prefix {
            "time_of_last_obs" => "time_of_last_obs.hour",
            _ => "time_of_first_obs.hour",
        };
        let minute = match prefix {
            "time_of_last_obs" => "time_of_last_obs.minute",
            _ => "time_of_first_obs.minute",
        };
        let second = match prefix {
            "time_of_last_obs" => "time_of_last_obs.second",
            _ => "time_of_first_obs.second",
        };
        // `5I6,F13.7` with nothing between the fields, so they abut whenever one
        // fills its width. Read the columns when the line is in them, and keep
        // the looser reading for the files that are not.
        // This layout leaves no gaps between its fields, so the check that
        // content stays inside them cannot vouch for it on its own: the columns
        // are preferred only when all six read as the numbers they are meant to
        // be. Anything else keeps the looser reading.
        let columns = fixed_record(body, TIME_OF_OBS_COLUMNS).filter(|columns| {
            columns[..5]
                .iter()
                .all(|column| column.parse::<i64>().is_ok())
                && columns[5].parse::<f64>().is_ok()
        });
        let names = [year, month, day, hour, minute, second];
        let policy = civil_second_policy_for_time_scale(scale);
        let epoch = columns
            .and_then(|columns| parse_epoch_time_fields(columns, line, names, policy).ok())
            .map_or_else(|| parse_epoch_time_tokens(body, line, names, policy), Ok)?;
        exact_in_field(
            epoch.second,
            HEADER_SECOND_WIDTH,
            HEADER_SECOND_DECIMALS,
            second,
            line,
        )?;
        Ok((epoch, scale))
    }

    fn parse_glonass_slots(&mut self, line: &str) -> Result<()> {
        // " N R01  1 R02 -4 ...": a count then 7-wide "SVNN ±k" entries.
        let count_field = field(line, 0, 3).trim();
        if !count_field.is_empty() {
            let count = strict_int_token::<usize>(count_field, "glonass_slot.count", line)?;
            self.glonass_slots_remaining = Some(count);
        }
        let body = field(line, 4, 60);
        let tokens: Vec<&str> = body.split_whitespace().collect();
        if !tokens.len().is_multiple_of(2) {
            return Err(Error::Parse(format!(
                "RINEX OBS GLONASS slot table has an odd token count in {line:?}"
            )));
        }
        for pair in tokens.as_chunks::<2>().0 {
            // Each pair is one declared slot entry; account for it against the
            // declared count first, so a skipped (unrepresentable) slot still
            // balances the count check in `finish`.
            if let Some(remaining) = self.glonass_slots_remaining.as_mut() {
                if *remaining == 0 {
                    return Err(Error::Parse(format!(
                        "RINEX OBS GLONASS slot table has more entries than declared in {line:?}"
                    )));
                }
                *remaining -= 1;
            }
            // A slot token that does not parse to a representable
            // `GnssSatelliteId` (e.g. an extended GLONASS slot beyond the
            // engine's PRN cap, like R28 in real BKG/IGS products) must not
            // reject the whole header: skip the entry and count it, the same
            // treatment nav `parse_glonass` gives such slots.
            let Some(sat) = parse_sv_token(pair[0]) else {
                self.push_unrepresentable_satellite_skip(pair[0]);
                continue;
            };
            if sat.system != GnssSystem::Glonass {
                return Err(Error::Parse(format!(
                    "RINEX OBS GLONASS slot token {:?} is not GLONASS in {line:?}",
                    pair[0]
                )));
            }
            let channel = strict_int_token::<i8>(pair[1], "glonass_slot.channel", line)?;
            if !valid_glonass_frequency_channel(i32::from(channel)) {
                return Err(Error::Parse(format!(
                    "RINEX OBS invalid glonass_slot.channel: {channel} out of range in {line:?}"
                )));
            }
            self.glonass_slots.insert(sat.prn, channel);
        }
        Ok(())
    }

    fn parse_glonass_cod_phs_bis(&mut self, line: &str) -> Result<()> {
        let tokens: Vec<&str> = field(line, 0, 60).split_whitespace().collect();
        let mut entries = Vec::new();
        for pair in tokens.chunks(2) {
            if pair.len() != 2 {
                return Err(Error::Parse(format!(
                    "RINEX OBS GLONASS COD/PHS/BIS has an odd token count in {line:?}"
                )));
            }
            let bias = strict_f64_token(pair[1], "glonass_code_phase_bias", line)?;
            entries.push((
                pair[0].to_string(),
                exact_in_field(
                    bias,
                    GLONASS_BIAS_WIDTH,
                    GLONASS_BIAS_DECIMALS,
                    "glonass_code_phase_bias",
                    line,
                )?,
            ));
        }
        // RINEX gives this record four biases, which fit one line. More than
        // that is an extension of this crate's own: the writer continues them
        // onto another line rather than cutting them off at the sixtieth
        // column, and a further line adds to the record. A blank record still
        // means "unknown" and so clears what came before it.
        match &mut self.glonass_cod_phs_bis {
            Some(existing) if !entries.is_empty() => existing.extend(entries),
            slot => *slot = Some(entries),
        }
        Ok(())
    }

    fn parse_leap_seconds(&mut self, line: &str) -> Result<()> {
        let current = strict_int_field::<i64>(line, 0, 6, "leap_seconds.current")?;
        self.leap_seconds = Some(ObsLeapSeconds {
            current,
            delta_future: optional_i64_field(line, 6, 12, "leap_seconds.delta_future")?,
            week: optional_i64_field(line, 12, 18, "leap_seconds.week")?,
            day: optional_i64_field(line, 18, 24, "leap_seconds.day")?,
        });
        Ok(())
    }

    fn parse_prn_obs_counts(&mut self, line: &str) -> Result<()> {
        // `3X,A1,I2,9I6`: three blanks, then the satellite, then the counts.
        let token = field(line, PRN_OBS_SATELLITE_COLUMN, PRN_OBS_COUNTS_COLUMN).trim();
        let sat = if token.is_empty() {
            let Some(sat) = self.prn_obs_counts_current else {
                return Ok(());
            };
            sat
        } else {
            // A version 2 file may leave the constellation letter blank, which
            // means the one its header names.
            let parsed = if self.is_rinex2() {
                self.parse_sv_token_v2(token)
            } else {
                parse_sv_token(token)
            };
            let Some(sat) = parsed else {
                self.prn_obs_counts_current = None;
                self.push_unrepresentable_satellite_skip(token);
                return Ok(());
            };
            self.prn_obs_counts_current = Some(sat);
            sat
        };
        // A version 2 header names its codes once for the whole file, and the
        // per-constellation lists are not built until the body is read, so the
        // count comes from that one list while the header is still being read.
        let count = if self.is_rinex2() {
            self.rinex2_obs_codes.len() + self.rinex2_obs_codes_remaining
        } else {
            self.obs_codes.get(&sat.system).map_or(0, Vec::len)
        };
        let already = self.prn_obs_counts.get(&sat).map_or(0, Vec::len);
        let remaining = count.saturating_sub(already);
        let mut values = Vec::with_capacity(remaining.min(9));
        for idx in 0..remaining {
            let start = PRN_OBS_COUNTS_COLUMN + idx * PRN_OBS_COUNT_WIDTH;
            if start + PRN_OBS_COUNT_WIDTH > write::HEADER_CONTENT_WIDTH {
                break;
            }
            let raw = field(line, start, start + PRN_OBS_COUNT_WIDTH).trim();
            if raw.is_empty() {
                values.push(None);
            } else {
                values.push(Some(strict_int_token::<usize>(raw, "prn_obs_count", line)?));
            }
        }
        self.prn_obs_counts.entry(sat).or_default().extend(values);
        Ok(())
    }

    fn parse_body<'a, I: Iterator<Item = &'a str>>(
        &mut self,
        lines: &mut std::iter::Peekable<I>,
    ) -> Result<()> {
        while let Some(raw) = lines.next() {
            let line = raw.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                continue;
            }
            if !line.starts_with('>') {
                // A stray non-epoch line outside an epoch block; tolerate.
                continue;
            }
            let time_scale = self
                .time_of_first_obs
                .map_or(TimeScale::Gpst, |(_, scale)| scale);
            let (epoch_time, flag, numsat, rcv_clock_offset_s, epoch_picoseconds) =
                parse_epoch_line(line, civil_second_policy_for_time_scale(time_scale))?;

            if flag > 1 {
                // Event record: the next `numsat` lines are header or comment
                // records, not observations. They are kept as they were written
                // so the epoch can be written back whole.
                let special_records = take_special_records(lines, numsat)?;
                self.epochs.push(ObsEpoch {
                    epoch: epoch_time,
                    flag,
                    rcv_clock_offset_s,
                    epoch_picoseconds,
                    declared_record_count: numsat,
                    special_records,
                    sats: BTreeMap::new(),
                });
                continue;
            }

            let mut sats = BTreeMap::new();
            for _ in 0..numsat {
                let sat_line = lines.next().ok_or_else(|| {
                    Error::Parse("RINEX OBS epoch truncated: missing satellite line".into())
                })?;
                let sat_line = sat_line.trim_end_matches(['\r', '\n']);
                // Resolve the satellite token first: a token that does not parse
                // to a representable `GnssSatelliteId` (e.g. an extended GLONASS
                // slot like R28) is an independent record that must not reject
                // the whole epoch/file. Skip the whole record - including any
                // wrapped continuation lines so the stream stays aligned - and
                // count it. No observation values are fabricated.
                let normalized = ascii_fixed_columns(sat_line);
                if !starts_with_sat_designator(&normalized) {
                    // Not a satellite record at all (e.g. a `>` epoch header): the
                    // declared `numsat` overran this epoch's records. That is
                    // structural corruption, not a skippable unknown satellite, so
                    // fail rather than swallow the next epoch's header/records.
                    return Err(Error::Parse(
                        "RINEX OBS epoch truncated: expected satellite record".into(),
                    ));
                }
                if parse_sv_token(field(&normalized, 0, 3)).is_none() {
                    // Lexically a satellite designator but the system/PRN is not
                    // representable (e.g. extended GLONASS slot R28): skip the whole
                    // record - including wrapped continuation lines - and count it.
                    // No observation values are fabricated.
                    self.push_unrepresentable_satellite_skip(field(&normalized, 0, 3));
                    consume_skipped_sat_continuations(lines);
                    continue;
                }
                let sat_record = self.collect_sat_record(sat_line, lines)?;
                let (sat, values) = self.parse_sat_line(&sat_record)?;
                sats.insert(sat, values);
            }
            self.epochs.push(ObsEpoch {
                epoch: epoch_time,
                flag,
                rcv_clock_offset_s,
                epoch_picoseconds,
                declared_record_count: numsat,
                special_records: Vec::new(),
                sats,
            });
        }
        Ok(())
    }

    fn parse_body_v2<'a, I: Iterator<Item = &'a str>>(
        &mut self,
        lines: &mut std::iter::Peekable<I>,
    ) -> Result<()> {
        while let Some(raw) = lines.next() {
            let line = raw.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                continue;
            }
            let time_scale = self
                .time_of_first_obs
                .map_or(TimeScale::Gpst, |(_, scale)| scale);
            let (epoch_time, flag, numsat, rcv_clock_offset_s) =
                parse_epoch_line_v2(line, civil_second_policy_for_time_scale(time_scale))?;

            if flag > 1 {
                let special_records = take_special_records(lines, numsat)?;
                self.epochs.push(ObsEpoch {
                    epoch: epoch_time,
                    flag,
                    rcv_clock_offset_s,
                    epoch_picoseconds: None,
                    declared_record_count: numsat,
                    special_records,
                    sats: BTreeMap::new(),
                });
                continue;
            }

            let sv_tokens = collect_epoch_sv_tokens_v2(line, numsat, lines)?;
            let obs_lines_per_sat = self.rinex2_obs_lines_per_sat()?;
            let mut sats = BTreeMap::new();
            for token in sv_tokens {
                let mut obs_lines = Vec::with_capacity(obs_lines_per_sat);
                for _ in 0..obs_lines_per_sat {
                    let obs_line = lines.next().ok_or_else(|| {
                        Error::Parse("RINEX OBS epoch truncated: missing observation line".into())
                    })?;
                    obs_lines.push(obs_line.trim_end_matches(['\r', '\n']).to_string());
                }

                let Some(sat) = self.parse_sv_token_v2(&token) else {
                    self.push_unrepresentable_satellite_skip(&token);
                    continue;
                };
                self.ensure_rinex2_system_obs_codes(sat.system);
                let values = self.parse_sat_obs_v2(sat.system, &obs_lines)?;
                sats.insert(sat, values);
            }
            self.epochs.push(ObsEpoch {
                epoch: epoch_time,
                flag,
                rcv_clock_offset_s,
                epoch_picoseconds: None,
                declared_record_count: numsat,
                special_records: Vec::new(),
                sats,
            });
        }
        Ok(())
    }

    fn rinex2_obs_lines_per_sat(&self) -> Result<usize> {
        if self.rinex2_obs_codes.is_empty() {
            return Err(Error::Parse(
                "RINEX OBS header has no # / TYPES OF OBSERV records".into(),
            ));
        }
        Ok(self.rinex2_obs_codes.len().div_ceil(5))
    }

    fn parse_sv_token_v2(&self, token: &str) -> Option<GnssSatelliteId> {
        parse_sv_token_v2(token, self.rinex2_default_system.unwrap_or(GnssSystem::Gps))
    }

    fn ensure_rinex2_system_obs_codes(&mut self, system: GnssSystem) {
        let version = self.version.unwrap_or(2.11);
        self.obs_codes.entry(system).or_insert_with(|| {
            self.rinex2_obs_codes
                .iter()
                .map(|code| canonical_rinex2_obs_code(system, code, version))
                .collect()
        });
    }

    fn parse_sat_obs_v2(&self, system: GnssSystem, obs_lines: &[String]) -> Result<Vec<ObsValue>> {
        let code_list = self.obs_codes.get(&system).ok_or_else(|| {
            Error::Parse(format!(
                "RINEX OBS satellite system {system} has no canonical observation-code table"
            ))
        })?;
        let mut values = Vec::with_capacity(code_list.len());
        for (i, code) in code_list.iter().enumerate() {
            let line = obs_lines.get(i / 5).map_or("", String::as_str);
            let start = (i % 5) * OBS_FIELD_WIDTH;
            let value_str = field(line, start, start + OBS_VALUE_WIDTH).trim();
            let value = if value_str.is_empty() {
                None
            } else {
                let scale = self.scale_factor_for(system, code);
                let parsed = strict_f64_token(value_str, "observation.value", line)? / scale;
                if format!("{:.3}", parsed * scale).len() > OBS_VALUE_WIDTH {
                    return Err(Error::Parse(
                        "RINEX OBS observation value exceeds the F14.3 field width".into(),
                    ));
                }
                Some(parsed)
            };
            let lli = digit_at(line, start + OBS_VALUE_WIDTH);
            let ssi = digit_at(line, start + OBS_VALUE_WIDTH + 1);
            values.push(ObsValue { value, lli, ssi });
        }
        Ok(values)
    }

    fn collect_sat_record<'a, I: Iterator<Item = &'a str>>(
        &self,
        first_line: &str,
        lines: &mut std::iter::Peekable<I>,
    ) -> Result<String> {
        let first_line = ascii_fixed_columns(first_line);
        let token = field(&first_line, 0, 3);
        let sat = parse_sv_token(token).ok_or_else(|| {
            Error::Parse(format!("RINEX OBS unparsable satellite token {token:?}"))
        })?;
        let n_obs = self.obs_count_for_sat(sat)?;
        let mut record = first_line.into_owned();

        while sat_record_field_count(record.len()) < n_obs {
            let Some(raw_next) = lines.peek().copied() else {
                break;
            };
            let next = raw_next.trim_end_matches(['\r', '\n']);
            let next = ascii_fixed_columns(next);
            // Stop at the next record boundary. Use the *lexical* designator
            // check, not `parse_sv_token`: a new record whose token does not
            // resolve to a representable id (e.g. an extended GLONASS slot like
            // R28) is still a new satellite record, not continuation data. Only a
            // lexical check recognizes it; otherwise its observations would be
            // spliced onto this record and the skip would never be counted.
            if next.starts_with('>') || starts_with_sat_designator(&next) {
                break;
            }
            let Some(continuation) = lines.next() else {
                break;
            };
            let continuation = ascii_fixed_columns(continuation.trim_end_matches(['\r', '\n']));
            append_sat_continuation(&mut record, &continuation, n_obs);
        }

        Ok(record)
    }

    fn obs_count_for_sat(&self, sat: GnssSatelliteId) -> Result<usize> {
        self.obs_codes
            .get(&sat.system)
            .map(Vec::len)
            .ok_or_else(|| {
                Error::Parse(format!(
                    "RINEX OBS satellite {sat} uses undeclared observation system"
                ))
            })
    }

    fn parse_sat_line(&self, line: &str) -> Result<(GnssSatelliteId, Vec<ObsValue>)> {
        let token = field(line, 0, 3);
        let sat = parse_sv_token(token).ok_or_else(|| {
            Error::Parse(format!("RINEX OBS unparsable satellite token {token:?}"))
        })?;
        let code_list = self.obs_codes.get(&sat.system).ok_or_else(|| {
            Error::Parse(format!(
                "RINEX OBS satellite {sat} uses undeclared observation system"
            ))
        })?;
        let mut values = Vec::with_capacity(code_list.len());
        for (i, code) in code_list.iter().enumerate() {
            let start = 3 + i * OBS_FIELD_WIDTH;
            let value_str = field(line, start, start + OBS_VALUE_WIDTH).trim();
            let value = if value_str.is_empty() {
                None
            } else {
                let scale = self.scale_factor_for(sat.system, code);
                let parsed = strict_f64_token(value_str, "observation.value", line)? / scale;
                // The serializer writes this value back as `F14.3` (value * scale).
                // A value whose three-decimal form needs more than the 14-column
                // field would expand it and shift the LLI/SSI and later fields on
                // reparse; one that needs more than three decimals comes back as
                // a different number. Neither is representable in this format, so
                // reject it rather than emit text that does not read back. Real
                // F14.3 data is always in range.
                // The comparison happens after the scale is divided back out,
                // because that is the value the next read recovers. Scaling is
                // not exactly invertible in binary floating point - a file value
                // of 123456.001 at scale 10 is stored as 12345.6001 and
                // multiplies back to 123456.00099999999 - so demanding that the
                // formatted text reparse to the scaled product would reject a
                // value that round trips perfectly well.
                let formatted = format!("{:.*}", OBS_VALUE_DECIMALS, parsed * scale);
                let recovered = formatted.parse::<f64>().map(|value| value / scale);
                if formatted.len() > OBS_VALUE_WIDTH || recovered != Ok(parsed) {
                    return Err(Error::Parse(format!(
                        "RINEX OBS observation value {parsed} is not representable in its \
                         F{OBS_VALUE_WIDTH}.{OBS_VALUE_DECIMALS} field in {line:?}"
                    )));
                }
                Some(parsed)
            };
            let lli = digit_at(line, start + OBS_VALUE_WIDTH);
            let ssi = digit_at(line, start + OBS_VALUE_WIDTH + 1);
            values.push(ObsValue { value, lli, ssi });
        }
        Ok((sat, values))
    }

    fn finish(self) -> Result<RinexObs> {
        let version = self
            .version
            .ok_or_else(|| Error::Parse("RINEX OBS missing RINEX VERSION / TYPE".into()))?;
        if let Some(remaining) = self.glonass_slots_remaining {
            if remaining != 0 {
                return Err(Error::Parse(format!(
                    "RINEX OBS GLONASS slot table missing {remaining} declared entries"
                )));
            }
        }
        let mut obs_codes = self.obs_codes;
        if obs_codes.is_empty() && !self.rinex2_obs_codes.is_empty() {
            let system = self.rinex2_default_system.unwrap_or(GnssSystem::Gps);
            obs_codes.insert(
                system,
                self.rinex2_obs_codes
                    .iter()
                    .map(|code| canonical_rinex2_obs_code(system, code, version))
                    .collect(),
            );
        }
        if obs_codes.is_empty() {
            return Err(Error::Parse(
                "RINEX OBS header has no SYS / # / OBS TYPES records".into(),
            ));
        }
        let header = ObsHeader {
            version,
            approx_position_m: self.approx_position_m,
            antenna_delta_hen_m: self.antenna_delta_hen_m,
            obs_codes,
            program_run_by_date: self.program_run_by_date,
            comments: self.comments,
            marker_number: self.marker_number,
            marker_type: self.marker_type,
            observer: self.observer,
            agency: self.agency,
            receiver: self.receiver,
            antenna: self.antenna,
            interval_s: self.interval_s,
            time_of_first_obs: self.time_of_first_obs,
            time_of_last_obs: self.time_of_last_obs,
            n_satellites: self.n_satellites,
            prn_obs_counts: self.prn_obs_counts,
            phase_shifts: self.phase_shifts,
            scale_factors: self.scale_factors,
            glonass_slots: self.glonass_slots,
            glonass_cod_phs_bis: self.glonass_cod_phs_bis,
            signal_strength_unit: self.signal_strength_unit,
            leap_seconds: self.leap_seconds,
            marker_name: self.marker_name,
            unretained_header_labels: self.unretained_header_labels,
        };
        Ok(RinexObs {
            header,
            epochs: self.epochs,
            skipped_records: self.diagnostics.skips.len(),
        })
    }

    fn scale_factor_for(&self, system: GnssSystem, code: &str) -> f64 {
        self.scale_factors
            .iter()
            .rev()
            .find(|record| {
                record.system == system
                    && (record.codes.is_empty() || record.codes.iter().any(|c| c == code))
            })
            .map_or(1.0, |record| record.factor)
    }
}

fn normalize_header_line(line: &str) -> Cow<'_, str> {
    let fixed_label = raw_field_from(line, 60).trim();
    if HEADER_LABELS.contains(&fixed_label) {
        return Cow::Borrowed(line);
    }

    for &label in HEADER_LABELS {
        let Some(index) = line.rfind(label) else {
            continue;
        };
        if !line[index + label.len()..].trim().is_empty() {
            continue;
        }
        let content = line[..index].trim_end();
        let content = truncate_header_content(content);
        return Cow::Owned(format!("{content:<60}{label}"));
    }

    Cow::Borrowed(line)
}

/// Replace characters outside RINEX's printable-ASCII domain without changing
/// byte-column offsets in the forgiving UTF-8 input.
fn printable_ascii_header_columns(line: &str) -> Cow<'_, str> {
    if line
        .bytes()
        .all(|byte| byte == b' ' || byte.is_ascii_graphic())
    {
        return Cow::Borrowed(line);
    }

    let mut normalized = String::with_capacity(line.len());
    for ch in line.chars() {
        if ch == ' ' || ch.is_ascii_graphic() {
            normalized.push(ch);
        } else {
            // Fixed columns are byte columns. Preserve the byte width of a
            // lossy UTF-8 replacement so later fields stay at the offsets the
            // forgiving parser originally observed.
            for _ in 0..ch.len_utf8() {
                normalized.push(' ');
            }
        }
    }
    Cow::Owned(normalized)
}

fn truncate_header_content(content: &str) -> Cow<'_, str> {
    if content.len() <= 60 {
        return Cow::Borrowed(content);
    }
    let mut end = 60;
    while !content.is_char_boundary(end) {
        end -= 1;
    }
    Cow::Owned(content[..end].to_string())
}

/// Parse a RINEX-3 epoch line `> YYYY MM DD HH MM SS.sssssss  F NN [clock]`,
/// returning the civil time, event flag, and satellite count.
type ParsedEpochLine = (ObsEpochTime, u8, usize, Option<f64>, Option<u32>);

fn parse_epoch_line(
    line: &str,
    second_policy: validate::CivilSecondPolicy,
) -> Result<ParsedEpochLine> {
    // Read the columns the format lays the record out in. That is the only way
    // to read an epoch whose satellite count fills its `I3` field: it then abuts
    // the flag before it, leaving no space for a tokenizer to split on. A line
    // that is not in the layout, or whose columns do not parse, falls through to
    // the looser reading that has always handled files which are not
    // column-exact, so nothing this reader already accepted changes.
    if let Some(tokens) = v3_epoch_column_tokens(line) {
        if let Ok((parsed, _)) = interpret_epoch_tokens(&tokens, line, second_policy) {
            return Ok(parsed);
        }
    }
    let body = line
        .strip_prefix('>')
        .ok_or_else(|| Error::Parse(format!("RINEX OBS epoch line lacks '>': {line:?}")))?;
    let tokens: Vec<&str> = body.split_whitespace().collect();
    match interpret_epoch_tokens(&tokens, line, second_policy) {
        Ok((parsed, _)) => Ok(parsed),
        // A line that is in neither the layout nor a shape the tokenizer can
        // read may still be one whose flag and count ran together. Separating
        // them is the last thing tried, so it cannot change any other reading.
        Err(error) => {
            let Some(split) = split_merged_epoch_flag_and_count(&tokens) else {
                return Err(error);
            };
            // The separated reading is preferred only when it accounts for the
            // whole line and any clock offset it read is written the way one is
            // written. Without both, a record whose trailing field this reader
            // cannot place is accepted with that field silently dropped.
            match interpret_epoch_tokens(&split, line, second_policy) {
                Ok((parsed, consumed))
                    if consumed == split.len()
                        && clock_token_is_written_as_one(&split, &parsed) =>
                {
                    Ok(parsed)
                }
                _ => Err(error),
            }
        }
    }
}

/// Whether a clock offset the repair read is written the way one is written.
///
/// The field is `F15.12`, so a real offset carries a decimal point or a Fortran
/// exponent. A bare integer in that position is some other field - a RINEX 4
/// record puts its picoseconds after the clock, and with the clock left blank
/// they land in the same token - and reading it as an offset would invent a
/// value.
fn clock_token_is_written_as_one(tokens: &[&str], parsed: &ParsedEpochLine) -> bool {
    let (_, _, _, rcv_clock_offset_s, _) = parsed;
    if rcv_clock_offset_s.is_none() {
        return true;
    }
    tokens
        .last()
        .is_some_and(|token| token.contains(['.', 'e', 'E', 'd', 'D']))
}

/// Separate an epoch flag and satellite count that share one token.
///
/// Returns `None` unless the token is four digits whose last three are 100 or
/// more: the count is right aligned in its field, so anything below that is
/// written with a leading space and never merges.
fn split_merged_epoch_flag_and_count<'a>(tokens: &[&'a str]) -> Option<Vec<&'a str>> {
    let all_digits = |token: &&str| token.bytes().all(|byte| byte.is_ascii_digit());
    let flag_index = usize::from(
        tokens
            .get(EPOCH_TIME_TOKENS)
            .is_some_and(|token| token.len() == 5 && all_digits(token)),
    ) + EPOCH_TIME_TOKENS;
    let merged = tokens.get(flag_index)?;
    if merged.len() != 4 || !all_digits(merged) || merged[1..].starts_with('0') {
        return None;
    }
    let (flag, count) = merged.split_at(1);
    let mut split = tokens.to_vec();
    split[flag_index] = flag;
    split.insert(flag_index + 1, count);
    Some(split)
}

/// Read a RINEX 3 epoch line's fields from their columns, in the order the
/// token reader expects them.
///
/// Returns `None` when the line is not laid out that way, including when this
/// writer's picosecond field is absent from where it puts it.
fn v3_epoch_column_tokens(line: &str) -> Option<Vec<&str>> {
    if let Some([marker, year, month, day, hour, minute, second, flag, count, clock]) =
        fixed_record(line, V3_EPOCH_COLUMNS)
    {
        if marker == ">" {
            return Some(epoch_tokens(
                [year, month, day, hour, minute, second],
                "",
                flag,
                count,
                clock,
            ));
        }
    }
    let [marker, year, month, day, hour, minute, second, picoseconds, flag, count, clock] =
        fixed_record(line, V3_EPOCH_PICOSECOND_COLUMNS)?;
    (marker == ">").then(|| {
        epoch_tokens(
            [year, month, day, hour, minute, second],
            picoseconds,
            flag,
            count,
            clock,
        )
    })
}

/// Assemble the token list the epoch reader expects, leaving out the two
/// optional fields when their columns are blank.
fn epoch_tokens<'a>(
    time: [&'a str; EPOCH_TIME_TOKENS],
    picoseconds: &'a str,
    flag: &'a str,
    count: &'a str,
    clock: &'a str,
) -> Vec<&'a str> {
    let mut tokens = time.to_vec();
    if !picoseconds.is_empty() {
        tokens.push(picoseconds);
    }
    tokens.push(flag);
    tokens.push(count);
    if !clock.is_empty() {
        tokens.push(clock);
    }
    tokens
}

/// Read an epoch line's fields from its tokens, reporting how many it used.
///
/// The token count matters only to the caller's fallback: the ordinary reading
/// ignores anything trailing, as it always has.
fn interpret_epoch_tokens(
    tokens: &[&str],
    line: &str,
    second_policy: validate::CivilSecondPolicy,
) -> Result<(ParsedEpochLine, usize)> {
    if tokens.len() < 8 {
        return Err(Error::Parse(format!(
            "RINEX OBS epoch line has too few fields in {line:?}"
        )));
    }
    // Take the six fields as they were read. Re-joining them and splitting again
    // would silently drop anything inside a field that contains a space, which a
    // column read can legitimately hand over.
    let time: [&str; EPOCH_TIME_TOKENS] = core::array::from_fn(|index| tokens[index]);
    let epoch = parse_epoch_time_fields(
        time,
        line,
        [
            "epoch.year",
            "epoch.month",
            "epoch.day",
            "epoch.hour",
            "epoch.minute",
            "epoch.second",
        ],
        second_policy,
    )?;
    exact_in_field(
        epoch.second,
        EPOCH_SECOND_WIDTH,
        EPOCH_SECOND_DECIMALS,
        "epoch.second",
        line,
    )?;

    let mut index = EPOCH_TIME_TOKENS;
    let epoch_picoseconds = if tokens
        .get(index)
        .is_some_and(|token| token.len() == 5 && token.bytes().all(|b| b.is_ascii_digit()))
        && tokens.len() >= 9
    {
        let value = strict_int_token::<u32>(tokens[index], "epoch.picoseconds", line)?;
        index += 1;
        Some(value)
    } else {
        None
    };
    let flag = strict_int_token::<u8>(tokens[index], "epoch.flag", line)?;
    index += 1;
    let numsat = parse_epoch_record_count(tokens[index], line)?;
    index += 1;
    let rcv_clock_offset_s = tokens
        .get(index)
        .map(|token| {
            let offset = strict_f64_token(token, "epoch.rcv_clock_offset_s", line)?;
            exact_in_field(
                offset,
                CLOCK_OFFSET_WIDTH,
                CLOCK_OFFSET_DECIMALS,
                "epoch.rcv_clock_offset_s",
                line,
            )
        })
        .transpose()?;
    let consumed = index + usize::from(rcv_clock_offset_s.is_some());
    Ok((
        (epoch, flag, numsat, rcv_clock_offset_s, epoch_picoseconds),
        consumed,
    ))
}

type ParsedEpochLineV2 = (ObsEpochTime, u8, usize, Option<f64>);

fn parse_epoch_line_v2(
    line: &str,
    second_policy: validate::CivilSecondPolicy,
) -> Result<ParsedEpochLineV2> {
    let head = field(line, 0, 32);
    // The satellite count fills its `I3` field from a hundred satellites up and
    // then abuts the flag before it, so the columns are read first. Each looser
    // reading below it is tried in turn, so every line this reader accepted
    // before is still read the same way.
    if let Some(columns) = fixed_record(head, V2_EPOCH_HEAD_COLUMNS)
        .filter(|columns| columns.iter().all(|column| !column.is_empty()))
    {
        if let Ok(parsed) = interpret_epoch_tokens_v2(&columns, line, second_policy) {
            return Ok(parsed);
        }
    }
    let whitespace: Vec<&str> = head.split_whitespace().collect();
    match interpret_epoch_tokens_v2(&whitespace, line, second_policy) {
        Ok(parsed) => Ok(parsed),
        // The repaired head has to be exactly the eight fields the record is,
        // or the line is some other shape and keeps its rejection.
        Err(error) => match split_merged_epoch_flag_and_count(&whitespace)
            .filter(|split| split.len() == V2_EPOCH_HEAD_COLUMNS.len())
        {
            Some(split) => {
                interpret_epoch_tokens_v2(&split, line, second_policy).map_err(|_| error)
            }
            None => Err(error),
        },
    }
}

/// Read a RINEX 2 epoch line from its fields, however they were separated.
fn interpret_epoch_tokens_v2(
    tokens: &[&str],
    line: &str,
    second_policy: validate::CivilSecondPolicy,
) -> Result<ParsedEpochLineV2> {
    if tokens.len() < 8 {
        return Err(Error::Parse(format!(
            "RINEX OBS v2 epoch line has too few fields in {line:?}"
        )));
    }
    let year = strict_int_token::<i32>(tokens[0], "epoch.year", line)?;
    let year = expand_rinex2_year(year);
    let month = strict_int_token::<i64>(tokens[1], "epoch.month", line)?;
    let day = strict_int_token::<i64>(tokens[2], "epoch.day", line)?;
    let hour = strict_int_token::<i64>(tokens[3], "epoch.hour", line)?;
    let minute = strict_int_token::<i64>(tokens[4], "epoch.minute", line)?;
    let second = strict_f64_token(tokens[5], "epoch.second", line)?;
    let civil = validate::civil_datetime_with_second_policy(
        i64::from(year),
        month,
        day,
        hour,
        minute,
        second,
        second_policy,
    )
    .map_err(|error| map_field_error(error, line))?;
    // RINEX 2 epochs are re-emitted through the RINEX 3 epoch writer, so they
    // are held to that line's field geometry.
    exact_in_field(
        civil.second,
        EPOCH_SECOND_WIDTH,
        EPOCH_SECOND_DECIMALS,
        "epoch.second",
        line,
    )?;
    let flag = strict_int_token::<u8>(tokens[6], "epoch.flag", line)?;
    let numsat = parse_epoch_record_count(tokens[7], line)?;
    let clock = field(line, 68, line.len()).trim();
    let rcv_clock_offset_s = if clock.is_empty() {
        None
    } else {
        let offset = strict_f64_token(clock, "epoch.rcv_clock_offset_s", line)?;
        Some(exact_in_field(
            offset,
            CLOCK_OFFSET_WIDTH,
            CLOCK_OFFSET_DECIMALS,
            "epoch.rcv_clock_offset_s",
            line,
        )?)
    };
    Ok((
        ObsEpochTime {
            year,
            month: civil.month as u8,
            day: civil.day as u8,
            hour: civil.hour as u8,
            minute: civil.minute as u8,
            second: civil.second,
        },
        flag,
        numsat,
        rcv_clock_offset_s,
    ))
}

fn expand_rinex2_year(year: i32) -> i32 {
    if year >= 100 {
        year
    } else if year >= 80 {
        1900 + year
    } else {
        2000 + year
    }
}

/// Accept one observation descriptor from a header code list, rejecting a token
/// the format's `A3` code field cannot carry.
///
/// The header code lists are fixed-column: every writer emits a descriptor into
/// a `1X,A3` field, and the parser reads those lists out of the 60-column
/// content area. A wider token is not a RINEX descriptor, and a header carrying
/// one cannot be serialized - the record would overrun its content area and
/// re-parse with fewer codes than its own count field declares.
fn obs_code_token(token: &str, record: &str, line: &str) -> Result<String> {
    if token.len() > OBS_CODE_FIELD_WIDTH {
        return Err(Error::Parse(format!(
            "RINEX OBS {record} code {token:?} exceeds the A{OBS_CODE_FIELD_WIDTH} field width in {line:?}"
        )));
    }
    Ok(token.to_string())
}

fn parse_epoch_record_count(token: &str, line: &str) -> Result<usize> {
    let count = strict_int_token::<usize>(token, "epoch.satellite_count", line)?;
    if token.len() > 3 || count > MAX_EPOCH_RECORD_COUNT {
        return Err(Error::Parse(format!(
            "RINEX OBS epoch satellite count exceeds the I3 field maximum of {MAX_EPOCH_RECORD_COUNT} in {line:?}"
        )));
    }
    Ok(count)
}

fn collect_epoch_sv_tokens_v2<'a, I: Iterator<Item = &'a str>>(
    first_line: &str,
    count: usize,
    lines: &mut std::iter::Peekable<I>,
) -> Result<Vec<String>> {
    let mut tokens = Vec::with_capacity(count);
    append_epoch_sv_tokens_v2(first_line, count, &mut tokens);
    while tokens.len() < count {
        let continuation = lines.next().ok_or_else(|| {
            Error::Parse("RINEX OBS v2 epoch truncated: missing satellite-list line".into())
        })?;
        append_epoch_sv_tokens_v2(
            continuation.trim_end_matches(['\r', '\n']),
            count,
            &mut tokens,
        );
    }
    tokens.truncate(count);
    Ok(tokens)
}

fn append_epoch_sv_tokens_v2(line: &str, count: usize, tokens: &mut Vec<String>) {
    let remaining = count.saturating_sub(tokens.len());
    for i in 0..remaining.min(12) {
        let start = 32 + i * 3;
        let token = field(line, start, start + 3);
        if token.trim().is_empty() {
            break;
        }
        tokens.push(token.to_string());
    }
}

fn parse_sv_token_v2(token: &str, default_system: GnssSystem) -> Option<GnssSatelliteId> {
    let token = token.trim();
    if token.is_empty() {
        return None;
    }
    let mut chars = token.chars();
    let first = chars.next()?;
    let (system, prn_text) = if let Some(system) = GnssSystem::from_letter(first) {
        (system, chars.as_str().trim())
    } else {
        (default_system, token)
    };
    let prn = prn_text.parse::<u8>().ok()?;
    GnssSatelliteId::new(system, prn).ok()
}

/// Take the records an event epoch declared, as they were written.
fn take_special_records<'a, I: Iterator<Item = &'a str>>(
    lines: &mut std::iter::Peekable<I>,
    count: usize,
) -> Result<Vec<String>> {
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        let line = lines
            .next()
            .ok_or_else(|| Error::Parse("RINEX OBS event record truncated".into()))?;
        records.push(line.trim_end_matches(['\r', '\n']).to_string());
    }
    Ok(records)
}

fn canonical_rinex2_obs_code(system: GnssSystem, code: &str, version: f64) -> String {
    let code = code.trim();
    if code.len() == 3 {
        return code.to_string();
    }
    let mut chars = code.chars();
    let Some(kind) = chars.next() else {
        return code.to_string();
    };
    let Some(band) = chars.next() else {
        return code.to_string();
    };
    if chars.next().is_some() || !matches!(kind, 'C' | 'P' | 'L' | 'D' | 'S') {
        return code.to_string();
    }

    if let Some(mapped) = canonical_rinex2_code_exact(system, kind, band, version) {
        // A leading `_` in the table means "this kind, on that band and
        // tracking", so one row covers the pseudorange, phase, Doppler and
        // signal strength that share a name's letter.
        return match mapped.strip_prefix('_') {
            Some(rest) => format!("{}{rest}", if kind == 'P' { 'C' } else { kind }),
            None => mapped.to_string(),
        };
    }

    let band = rinex2_band(system, band);
    let canonical_kind = if kind == 'P' { 'C' } else { kind };
    let attr = rinex2_default_tracking_attr(system, kind, band);
    format!("{canonical_kind}{band}{attr}")
}

/// The version that gave the L1 and L2 civil signals their own letters, leaving
/// the digits to the P code.
const RINEX2_LETTERED_NAMES_VERSION: f64 = 2.12;

/// The version from which `C2` names the L2P(Y) pseudorange rather than L2C.
/// 2.12 gave L2C its own names and left `C2` to the P code.
const RINEX2_L2C_RENAMED_VERSION: f64 = 2.12;

/// Whether a constellation may carry a version 2 observation code by this name.
///
/// Version 2 gives `P` to GPS and GLONASS only: "P: Pseudorange GPS and Glonass:
/// P code". Reading someone else's `P` leniently is one thing; writing one puts
/// a code in the file that the format does not define for that constellation,
/// and a shared column has to satisfy this for every constellation in it, not
/// only the one whose candidates it came from.
fn rinex2_name_allowed(system: GnssSystem, name: &str) -> bool {
    let mut chars = name.chars();
    let (Some(kind), Some(band)) = (chars.next(), chars.next()) else {
        return false;
    };
    if kind == 'P' && !matches!(system, GnssSystem::Gps | GnssSystem::Glonass) {
        return false;
    }
    // The bands each constellation has an observable on. Version 2 shares one
    // digit space across all of them, so a name is only this constellation's
    // where it names a band this one measures. BeiDou is not in version 2 at
    // all; its digits are the frequency slots receivers wrote it into.
    let bands: &[char] = match system {
        GnssSystem::Gps => {
            if kind == 'P' {
                &['1', '2']
            } else {
                &['1', '2', '5']
            }
        }
        GnssSystem::Glonass => &['1', '2', '3'],
        GnssSystem::Galileo => &['1', '5', '6', '7', '8'],
        GnssSystem::BeiDou => &['1', '2', '6', '7'],
        GnssSystem::Qzss => &['1', '2', '5', '6'],
        GnssSystem::Sbas => &['1', '5'],
        // Version 2 has no NavIC, so no digit names one of its bands.
        GnssSystem::Navic => &[],
    };
    bands.contains(&band)
}

/// Every RINEX 2 observation code a system's canonical code was mapped from,
/// in the order a writer should prefer them.
///
/// The mapping into canonical codes is not injective - BeiDou's `C1` and `P1`
/// both become `C2I`, and Galileo's `C5` and `P2` both become `C5X` - so this
/// does not recover the text a file carried. Each name it returns maps forward
/// to the same canonical code, which is what a version 2 file has to carry for
/// the product to survive being written and read again;
/// `rinex2_code_round_trips_through_its_canonical_form` holds the two together.
///
/// A name spelling the canonical code's own kind and band comes first, so a
/// Galileo `C5X` is written `C5` rather than the alias `P2`, which no reader
/// outside this crate defines for Galileo. The rest follow as alternatives, for
/// a caller that needs one name several constellations can read.
///
/// A canonical code no version 2 name maps to - one a version 3 file named, on
/// a product whose version was then set below 3 - yields its kind and band
/// alone, losing the tracking attribute, which is the most a version 2 code can
/// say. `C1X` becomes `C1`, and reads back as this system's default tracking on
/// band 1. The list is empty only for a code that is not three characters, so
/// it did not come from RINEX at all.
fn rinex2_obs_code_candidates(system: GnssSystem, canonical: &str, version: f64) -> Vec<String> {
    let kinds: &[char] = match system {
        GnssSystem::Gps | GnssSystem::Glonass => &['C', 'P', 'L', 'D', 'S'],
        _ => &['C', 'L', 'D', 'S'],
    };
    const BANDS: [char; 9] = ['1', '2', '3', '4', '5', '6', '7', '8', '9'];
    let mut chars = canonical.chars();
    let own = match (chars.next(), chars.next(), chars.next(), chars.next()) {
        (Some(kind), Some(band), Some(_), None) => Some((kind, band)),
        _ => None,
    };
    let mut names: Vec<String> = Vec::new();
    if let Some((kind, band)) = own {
        let digit = rinex2_digit_for_band(system, band);
        // A canonical `C` may have been written `C` or `P` in version 2.
        let spellings: &[char] = if kind == 'C' && kinds.contains(&'P') {
            &['C', 'P']
        } else {
            std::slice::from_ref(&kind)
        };
        for spelling in spellings {
            let name = format!("{spelling}{digit}");
            if rinex2_name_allowed(system, &name)
                && canonical_rinex2_obs_code(system, &name, version) == canonical
                && !names.contains(&name)
            {
                names.push(name);
            }
        }
    }
    for kind in kinds {
        for band in BANDS {
            let name = format!("{kind}{band}");
            if rinex2_name_allowed(system, &name)
                && canonical_rinex2_obs_code(system, &name, version) == canonical
                && !names.contains(&name)
            {
                names.push(name);
            }
        }
    }
    if names.is_empty() {
        if let Some((kind, band)) = own {
            // The band comes back as the digit that names it, so dropping an
            // attribute version 2 cannot carry does not also move the band.
            names.push(format!("{kind}{}", rinex2_digit_for_band(system, band)));
        }
    }
    names
}

/// Width of a version 2 observation type, the `A2` of `9(4X,A2)`.
const OBS_TYPE_V2_WIDTH: usize = 2;
/// Columns the `# / TYPES OF OBSERV` count occupies before its codes, `I6`.
const OBS_TYPE_V2_COUNT_WIDTH: usize = 6;

/// The band a version 2 observation code's digit names.
///
/// For everything RINEX 2.11 covers the digit already is the band, so this is
/// the digit. BeiDou is not in version 2 at all, and the receivers that wrote it
/// there numbered the frequencies B1, B2, B3 as 1, 2, 3, while RINEX 3 numbers
/// those bands 2, 7 and 6. Without the remap a file's `C1` and `L1` would land
/// on different bands - `C1` through the table as B1I, `L1` through the digit as
/// B1C - which is the same measurement read as two signals.
fn rinex2_band(system: GnssSystem, band: char) -> char {
    match (system, band) {
        // Version 2 numbers its digits by frequency slot across every
        // constellation, not by a per-constellation count. BeiDou B1I sits in
        // slot 2, which RINEX 3 numbers band 2, and some writers put it in slot
        // 1 instead. B2I is slot 7 and B3I slot 6, which RINEX 3 numbers the
        // same, so those digits need nothing.
        (GnssSystem::BeiDou, '1' | '2') => '2',
        _ => band,
    }
}

/// The version 2 digit that names a band, the inverse of [`rinex2_band`].
///
/// A writer needs this wherever it builds a name from a canonical code's own
/// band rather than from a name it already checked. Without it a BeiDou `C2Q`
/// would be written `C2`, which reads back as B2I: the band changed to keep an
/// attribute version 2 cannot carry anyway.
fn rinex2_digit_for_band(system: GnssSystem, band: char) -> char {
    match (system, band) {
        (GnssSystem::BeiDou, '2') => '2',
        _ => band,
    }
}

fn canonical_rinex2_code_exact(
    system: GnssSystem,
    kind: char,
    band: char,
    version: f64,
) -> Option<&'static str> {
    match (system, kind, band) {
        // 2.12 gave the L1 and L2 civil signals their own letters and left the
        // digits to the P code, so `L1` there is the P(Y) phase, not C/A.
        (GnssSystem::Gps, _, 'A') if version >= RINEX2_LETTERED_NAMES_VERSION => Some("_1C"),
        (GnssSystem::Glonass, _, 'A') if version >= RINEX2_LETTERED_NAMES_VERSION => Some("_1C"),
        (GnssSystem::Qzss, _, 'A') if version >= RINEX2_LETTERED_NAMES_VERSION => Some("_1C"),
        (GnssSystem::Sbas, _, 'A') if version >= RINEX2_LETTERED_NAMES_VERSION => Some("_1C"),
        (GnssSystem::Gps, _, 'B') if version >= RINEX2_LETTERED_NAMES_VERSION => Some("_1X"),
        (GnssSystem::Qzss, _, 'B') if version >= RINEX2_LETTERED_NAMES_VERSION => Some("_1X"),
        (GnssSystem::Gps, _, 'C') if version >= RINEX2_LETTERED_NAMES_VERSION => Some("_2X"),
        (GnssSystem::Qzss, _, 'C') if version >= RINEX2_LETTERED_NAMES_VERSION => Some("_2X"),
        (GnssSystem::Glonass, _, 'D') if version >= RINEX2_LETTERED_NAMES_VERSION => Some("_2C"),
        (GnssSystem::Gps, 'L' | 'D' | 'S', '1') if version >= RINEX2_LETTERED_NAMES_VERSION => {
            Some("_1W")
        }
        (GnssSystem::Glonass, 'L' | 'D' | 'S', '1') if version >= RINEX2_LETTERED_NAMES_VERSION => {
            Some("_1P")
        }
        (GnssSystem::Gps, 'C', '1') => Some("C1C"),
        // 2.11 section 10.1.1 added "Observation code for L2C pseudorange (C2)",
        // and RINEX 3 spells L2C `C2S`, `C2L` or `C2X` by channel; `X` is both,
        // which is what the file says when it does not say which. `C2C` is L2
        // C/A, a different signal. From 2.12 the same name is L2P(Y), because
        // 2.12 gave L2C its own names and left `C2` to the P code.
        (GnssSystem::Gps, 'C', '2') if version >= RINEX2_L2C_RENAMED_VERSION => Some("C2W"),
        (GnssSystem::Gps, 'C', '2') => Some("C2X"),
        (GnssSystem::Gps, 'P', '1') => Some("C1W"),
        (GnssSystem::Gps, 'P', '2') => Some("C2W"),
        (GnssSystem::Glonass, 'C', '1') => Some("C1C"),
        (GnssSystem::Glonass, 'C', '2') => Some("C2C"),
        (GnssSystem::Glonass, 'P', '1') => Some("C1P"),
        (GnssSystem::Glonass, 'P', '2') => Some("C2P"),
        (GnssSystem::Sbas, 'C', '1') => Some("C1C"),
        _ => None,
    }
}

fn rinex2_default_tracking_attr(system: GnssSystem, kind: char, band: char) -> char {
    match system {
        GnssSystem::Gps => match band {
            '1' => 'C',
            '2' => {
                if kind == 'C' {
                    'C'
                } else {
                    'W'
                }
            }
            '5' => 'X',
            _ => 'X',
        },
        GnssSystem::Glonass => match band {
            '1' => 'C',
            '2' => 'P',
            '3' => 'X',
            _ => 'X',
        },
        // Version 2 names no channel for Galileo, so every band reads as the
        // combined one rather than claiming a single channel that the file
        // never stated.
        GnssSystem::Galileo => 'X',
        GnssSystem::BeiDou => match band {
            '2' | '6' | '7' => 'I',
            '1' => 'P',
            '5' | '8' => 'X',
            _ => 'X',
        },
        GnssSystem::Qzss => match band {
            '1' => 'C',
            // L2C, whose channel version 2 does not name.
            _ => 'X',
        },
        GnssSystem::Navic => match band {
            '5' | '9' => 'A',
            _ => 'X',
        },
        GnssSystem::Sbas => match band {
            '1' => 'C',
            '5' => 'X',
            _ => 'X',
        },
    }
}

/// Map a RINEX time-system label onto the core [`TimeScale`]. A blank label
/// defaults to GPS time, which is the scale a multi-GNSS observation file uses
/// in practice; an explicit unknown label is rejected.
fn time_scale_from_label(label: &str, line: &str) -> Result<TimeScale> {
    let label = label.trim();
    if label.is_empty() {
        Ok(TimeScale::Gpst)
    } else {
        time_scale_label(label).ok_or_else(|| {
            Error::Parse(format!(
                "RINEX OBS TIME OF FIRST OBS unknown time scale {label:?} in {line:?}"
            ))
        })
    }
}

fn civil_second_policy_for_time_scale(scale: TimeScale) -> validate::CivilSecondPolicy {
    match scale {
        // GLONASST is UTC(SU)-based, so it can carry positive-leap-second labels.
        TimeScale::Utc | TimeScale::Glonasst => validate::CivilSecondPolicy::UtcLike,
        TimeScale::Tai
        | TimeScale::Tt
        | TimeScale::Tcg
        | TimeScale::Tdb
        | TimeScale::Tcb
        | TimeScale::Gpst
        | TimeScale::Gst
        | TimeScale::Bdt
        | TimeScale::Qzsst => validate::CivilSecondPolicy::Continuous,
    }
}

fn parse_epoch_time_tokens(
    body: &str,
    line: &str,
    fields: [&'static str; 6],
    second_policy: validate::CivilSecondPolicy,
) -> Result<ObsEpochTime> {
    let tokens: Vec<&str> = body.split_whitespace().collect();
    if tokens.len() < fields.len() {
        let field = fields[tokens.len()];
        return Err(map_field_error(FieldError::Missing { field }, line));
    }
    let read: [&str; 6] = core::array::from_fn(|index| tokens[index]);
    parse_epoch_time_fields(read, line, fields, second_policy)
}

/// Validate the six date and time fields of an epoch, however they were read.
fn parse_epoch_time_fields(
    tokens: [&str; 6],
    line: &str,
    fields: [&'static str; 6],
    second_policy: validate::CivilSecondPolicy,
) -> Result<ObsEpochTime> {
    let year = strict_int_token::<i32>(tokens[0], fields[0], line)?;
    let month = strict_int_token::<i64>(tokens[1], fields[1], line)?;
    let day = strict_int_token::<i64>(tokens[2], fields[2], line)?;
    let hour = strict_int_token::<i64>(tokens[3], fields[3], line)?;
    let minute = strict_int_token::<i64>(tokens[4], fields[4], line)?;
    let second = strict_f64_token(tokens[5], fields[5], line)?;
    let civil = validate::civil_datetime_with_second_policy(
        year as i64,
        month,
        day,
        hour,
        minute,
        second,
        second_policy,
    )
    .map_err(|error| map_field_error(error, line))?;
    Ok(ObsEpochTime {
        year,
        month: civil.month as u8,
        day: civil.day as u8,
        hour: civil.hour as u8,
        minute: civil.minute as u8,
        second: civil.second,
    })
}

fn strict_vec3_tokens(body: &str, line: &str, fields: [&'static str; 3]) -> Result<[f64; 3]> {
    // The three components are adjacent `F14.4` columns with nothing between
    // them, so one that fills its field touches its neighbour: a -10,000,000 m
    // coordinate writes as `0.0000-10000000.0000`, which no tokenizer can split.
    // Read the columns when the line is laid out in them, and keep the looser
    // reading for the files that are not.
    let columns = fixed_record(
        body,
        [
            (0, HEADER_VEC3_WIDTH),
            (HEADER_VEC3_WIDTH, 2 * HEADER_VEC3_WIDTH),
            (2 * HEADER_VEC3_WIDTH, 3 * HEADER_VEC3_WIDTH),
        ],
    )
    .filter(|columns| {
        columns.iter().enumerate().all(|(index, column)| {
            !column.is_empty() && strict_f64_token(column, fields[index], line).is_ok()
        })
    });
    let parses = |token: &str, index: usize| strict_f64_token(token, fields[index], line).is_ok();
    let whitespace: Vec<&str> = body.split_whitespace().collect();
    let whitespace_reads = whitespace.len() >= fields.len()
        && whitespace
            .iter()
            .take(fields.len())
            .enumerate()
            .all(|(index, token)| parses(token, index));
    let tokens: Vec<&str> = match columns {
        Some(columns) => columns.to_vec(),
        None if whitespace_reads => whitespace,
        // Neither the layout nor the tokenizer read this line. The components
        // may still sit on their columns with something trailing them, which is
        // what this reader accepted before the layout test existed.
        None => {
            let lenient: Vec<&str> = (0..fields.len())
                .map(|index| {
                    let start = index * HEADER_VEC3_WIDTH;
                    body.get(start..start + HEADER_VEC3_WIDTH)
                        .map(str::trim)
                        .unwrap_or_default()
                })
                .collect();
            if lenient
                .iter()
                .enumerate()
                .all(|(index, column)| !column.is_empty() && parses(column, index))
            {
                lenient
            } else {
                whitespace
            }
        }
    };
    if tokens.len() < fields.len() {
        let field = fields[tokens.len()];
        return Err(map_field_error(FieldError::Missing { field }, line));
    }
    let mut values = [0.0_f64; 3];
    for (index, value) in values.iter_mut().enumerate() {
        let parsed = strict_f64_token(tokens[index], fields[index], line)?;
        *value = exact_in_field(
            parsed,
            HEADER_VEC3_WIDTH,
            HEADER_VEC3_DECIMALS,
            fields[index],
            line,
        )?;
    }
    Ok(values)
}

fn optional_f64_field(
    line: &str,
    start: usize,
    end: usize,
    field_name: &'static str,
) -> Result<Option<f64>> {
    let token = field(line, start, end).trim();
    if token.is_empty() {
        Ok(None)
    } else {
        strict_f64_token(token, field_name, line).map(Some)
    }
}

fn optional_i64_field(
    line: &str,
    start: usize,
    end: usize,
    field_name: &'static str,
) -> Result<Option<i64>> {
    let token = field(line, start, end).trim();
    if token.is_empty() {
        Ok(None)
    } else {
        strict_int_token::<i64>(token, field_name, line).map(Some)
    }
}

fn optional_trimmed(line: &str, start: usize, end: usize) -> Option<String> {
    let value = field(line, start, end).trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn strict_int_field<T>(line: &str, start: usize, end: usize, field_name: &'static str) -> Result<T>
where
    T: core::str::FromStr,
{
    strict_int_token(field(line, start, end), field_name, line)
}

fn strict_f64_token(token: &str, field_name: &'static str, line: &str) -> Result<f64> {
    validate::strict_f64(token, field_name).map_err(|error| map_field_error(error, line))
}

fn validate_finite_input(value: f64, field: &'static str) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::InvalidInput(format!(
            "RINEX OBS {field} must be finite"
        )))
    }
}

fn strict_int_token<T>(token: &str, field_name: &'static str, line: &str) -> Result<T>
where
    T: core::str::FromStr,
{
    validate::strict_int::<T>(token, field_name).map_err(|error| map_field_error(error, line))
}

fn scale_factor_value(value: u32) -> Result<f64> {
    match value {
        1 | 10 | 100 | 1000 => Ok(f64::from(value)),
        _ => Err(Error::Parse(format!(
            "RINEX OBS invalid scale_factor.factor: expected 1, 10, 100, or 1000, got {value}"
        ))),
    }
}

/// Reject a header value the fixed-column field cannot re-emit unchanged.
///
/// The writer sends every parsed header number back through the same `Fw.d`
/// format, so a value the field cannot express re-parses as a different number
/// (or, when it overruns its columns, as a line the parser cannot read at all).
/// This is the same rule the SP3 record fields apply: never accept a record the
/// writer cannot reproduce.
fn exact_in_field(
    value: f64,
    width: usize,
    decimals: usize,
    field_name: &'static str,
    line: &str,
) -> Result<f64> {
    if validate::representable_in_fixed_field(value, Some(width), decimals) {
        return Ok(value);
    }
    Err(Error::Parse(format!(
        "RINEX OBS invalid {field_name}: {value} is not representable in its \
         F{width}.{decimals} field in {line:?}"
    )))
}

fn map_field_error(error: FieldError, line: &str) -> Error {
    Error::Parse(format!(
        "RINEX OBS invalid {}: {error} in {line:?}",
        error.field()
    ))
}

fn obs_payload_field_count(payload_len: usize) -> usize {
    let full = payload_len / OBS_FIELD_WIDTH;
    let trailing = payload_len % OBS_FIELD_WIDTH;
    full + usize::from(trailing >= OBS_VALUE_WIDTH)
}

fn sat_record_field_count(record_len: usize) -> usize {
    obs_payload_field_count(record_len.saturating_sub(3))
}

fn ascii_fixed_columns(line: &str) -> Cow<'_, str> {
    if line.is_ascii() {
        Cow::Borrowed(line)
    } else {
        Cow::Owned(
            line.chars()
                .map(|ch| if ch.is_ascii() { ch } else { ' ' })
                .collect(),
        )
    }
}

fn truncate_to_char_boundary(record: &mut String, len: usize) {
    let mut end = len.min(record.len());
    while !record.is_char_boundary(end) {
        end -= 1;
    }
    record.truncate(end);
}

/// Whether `line` lexically begins with a RINEX satellite designator (a system
/// letter followed by one or two PRN digits), whether or not it parses to a
/// representable [`GnssSatelliteId`]. Used to find satellite-record boundaries
/// when skipping an unknown/out-of-range record, so that a following
/// unrepresentable record (e.g. another extended GLONASS slot) is not mistaken
/// for a wrapped continuation line. Observation continuation lines begin with a
/// right-justified numeric field, never a letter, so they never match.
fn starts_with_sat_designator(line: &str) -> bool {
    let Some(token) = line.get(0..3) else {
        return false;
    };
    let b = token.as_bytes();
    let prn = token[1..].trim();
    b[0].is_ascii_alphabetic()
        && (1..=2).contains(&prn.len())
        && prn.bytes().all(|byte| byte.is_ascii_digit())
}

/// Consume the wrapped continuation lines of a satellite record being skipped
/// (its token did not resolve), leaving the iterator positioned at the next
/// satellite record or epoch header.
fn consume_skipped_sat_continuations<'a, I: Iterator<Item = &'a str>>(
    lines: &mut std::iter::Peekable<I>,
) {
    while let Some(raw_next) = lines.peek().copied() {
        let next = ascii_fixed_columns(raw_next.trim_end_matches(['\r', '\n']));
        if next.starts_with('>') || starts_with_sat_designator(&next) {
            break;
        }
        lines.next();
    }
}

fn append_sat_continuation(record: &mut String, continuation: &str, n_obs: usize) {
    let fields_present = sat_record_field_count(record.len());
    let logical_len = 3 + fields_present * OBS_FIELD_WIDTH;
    truncate_to_char_boundary(record, logical_len);

    let remaining = n_obs.saturating_sub(fields_present);
    let payload = field(continuation, 3, continuation.len());
    let fields_available = obs_payload_field_count(payload.len());
    let fields_to_copy = remaining.min(fields_available);
    let width = fields_to_copy * OBS_FIELD_WIDTH;
    record.push_str(field(payload, 0, width));
}

/// Parse a 3-char SV token (e.g. `G01`, `C30`) into a [`GnssSatelliteId`].
fn parse_sv_token(token: &str) -> Option<GnssSatelliteId> {
    token.parse::<GnssSatelliteId>().ok()
}

/// Read a single decimal digit at byte `col`, or `None` if it is blank /
/// non-digit / past end of line.
fn digit_at(line: &str, col: usize) -> Option<u8> {
    line.as_bytes()
        .get(col)
        .filter(|b| b.is_ascii_digit())
        .map(|b| b - b'0')
}

mod write;

#[cfg(all(test, sidereon_repo_tests))]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
