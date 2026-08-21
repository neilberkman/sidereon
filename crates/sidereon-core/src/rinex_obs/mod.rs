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

use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::astro::time::model::TimeScale;

use crate::format::columns::{raw_field as field, raw_field_from};
use crate::format::{Diagnostics, RecordRef, Skip, SkipReason};
use crate::frequencies::{
    rinex_band_frequency_hz, rinex_observation_frequency_hz, rinex_observation_wavelength_m,
};
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::rinex_common::time_scale_label;
use crate::rinex_nav::valid_glonass_frequency_channel;
use crate::validate::{self, FieldError};
use crate::{Error, Result};

/// Width of one RINEX-3 observation field (`F14.3` value + LLI + SSI).
const OBS_FIELD_WIDTH: usize = 16;
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
    /// Epoch flag: 0 = OK, 1 = power failure, >1 = an event record (skipped).
    pub flag: u8,
    /// Optional receiver clock offset from the epoch line, seconds.
    pub rcv_clock_offset_s: Option<f64>,
    /// Optional RINEX 4 epoch picosecond extension.
    pub epoch_picoseconds: Option<u32>,
    /// Satellite/special-record count declared on the epoch line.
    pub declared_record_count: usize,
    /// Number of special records declared by an event epoch.
    pub special_record_count: usize,
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
        let mut body = lines.peekable();
        if parser.is_rinex2() {
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
        let allowed_codes = filter
            .allowed_codes(sat.system)
            .expect("filter presence checked");
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
                    self.interval_s = optional_f64_field(line, 0, 10, "interval_s")?;
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
            let letter = sys_field.chars().next().unwrap();
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
        let list = self.obs_codes.get_mut(&system).expect("system inserted");
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
        for code in field(line, 6, 60).split_whitespace() {
            if self.rinex2_obs_codes_remaining == 0 {
                return Err(Error::Parse(format!(
                    "RINEX OBS # / TYPES OF OBSERV lists more codes than declared in {line:?}"
                )));
            }
            self.rinex2_obs_codes
                .push(obs_code_token(code, "# / TYPES OF OBSERV", line)?);
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
            let letter = sys_field.chars().next().unwrap();
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
        let record = self
            .scale_factors
            .last_mut()
            .expect("scale factor continuation has a record");
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
        let epoch = parse_epoch_time_tokens(
            body,
            line,
            [year, month, day, hour, minute, second],
            civil_second_policy_for_time_scale(scale),
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
            entries.push((
                pair[0].to_string(),
                strict_f64_token(pair[1], "glonass_code_phase_bias", line)?,
            ));
        }
        self.glonass_cod_phs_bis = Some(entries);
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
        let token = field(line, 0, 3).trim();
        let sat = if token.is_empty() {
            let Some(sat) = self.prn_obs_counts_current else {
                return Ok(());
            };
            sat
        } else {
            let Some(sat) = parse_sv_token(token) else {
                self.prn_obs_counts_current = None;
                self.push_unrepresentable_satellite_skip(token);
                return Ok(());
            };
            self.prn_obs_counts_current = Some(sat);
            sat
        };
        let count = self.obs_codes.get(&sat.system).map_or(0, Vec::len);
        let already = self.prn_obs_counts.get(&sat).map_or(0, Vec::len);
        let remaining = count.saturating_sub(already);
        let mut values = Vec::with_capacity(remaining.min(9));
        for idx in 0..remaining {
            let start = 3 + idx * 6;
            if start + 6 > 60 {
                break;
            }
            let raw = field(line, start, start + 6).trim();
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
                // Event record: the next `numsat` lines are header/comment
                // records, not observations. Consume and skip them, keeping a
                // placeholder epoch so indices stay meaningful.
                for _ in 0..numsat {
                    lines
                        .next()
                        .ok_or_else(|| Error::Parse("RINEX OBS event record truncated".into()))?;
                }
                self.epochs.push(ObsEpoch {
                    epoch: epoch_time,
                    flag,
                    rcv_clock_offset_s,
                    epoch_picoseconds,
                    declared_record_count: numsat,
                    special_record_count: numsat,
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
                special_record_count: 0,
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
                for _ in 0..numsat {
                    lines
                        .next()
                        .ok_or_else(|| Error::Parse("RINEX OBS event record truncated".into()))?;
                }
                self.epochs.push(ObsEpoch {
                    epoch: epoch_time,
                    flag,
                    rcv_clock_offset_s,
                    epoch_picoseconds: None,
                    declared_record_count: numsat,
                    special_record_count: numsat,
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
                special_record_count: 0,
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
        self.obs_codes.entry(system).or_insert_with(|| {
            self.rinex2_obs_codes
                .iter()
                .map(|code| canonical_rinex2_obs_code(system, code))
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
            let continuation = lines.next().expect("peeked continuation line");
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
                // reparse, so it is not representable in this format - reject it
                // rather than emit ambiguous text. Real F14.3 data is always in
                // range.
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
                    .map(|code| canonical_rinex2_obs_code(system, code))
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
    let body = line
        .strip_prefix('>')
        .ok_or_else(|| Error::Parse(format!("RINEX OBS epoch line lacks '>': {line:?}")))?;
    let tokens: Vec<&str> = body.split_whitespace().collect();
    if tokens.len() < 8 {
        return Err(Error::Parse(format!(
            "RINEX OBS epoch line has too few fields in {line:?}"
        )));
    }
    let epoch = parse_epoch_time_tokens(
        &tokens[..6].join(" "),
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

    let mut index = 6;
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
        .map(|token| strict_f64_token(token, "epoch.rcv_clock_offset_s", line))
        .transpose()?;
    Ok((epoch, flag, numsat, rcv_clock_offset_s, epoch_picoseconds))
}

type ParsedEpochLineV2 = (ObsEpochTime, u8, usize, Option<f64>);

fn parse_epoch_line_v2(
    line: &str,
    second_policy: validate::CivilSecondPolicy,
) -> Result<ParsedEpochLineV2> {
    let head = field(line, 0, 32);
    let tokens: Vec<&str> = head.split_whitespace().collect();
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
    let flag = strict_int_token::<u8>(tokens[6], "epoch.flag", line)?;
    let numsat = parse_epoch_record_count(tokens[7], line)?;
    let clock = field(line, 68, line.len()).trim();
    let rcv_clock_offset_s = if clock.is_empty() {
        None
    } else {
        Some(strict_f64_token(clock, "epoch.rcv_clock_offset_s", line)?)
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

fn canonical_rinex2_obs_code(system: GnssSystem, code: &str) -> String {
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

    if let Some(mapped) = canonical_rinex2_code_exact(system, kind, band) {
        return mapped.to_string();
    }

    let canonical_kind = if kind == 'P' { 'C' } else { kind };
    let attr = rinex2_default_tracking_attr(system, kind, band);
    format!("{canonical_kind}{band}{attr}")
}

fn canonical_rinex2_code_exact(system: GnssSystem, kind: char, band: char) -> Option<&'static str> {
    match (system, kind, band) {
        (GnssSystem::Gps, 'C', '1') => Some("C1C"),
        (GnssSystem::Gps, 'C', '2') => Some("C2C"),
        (GnssSystem::Gps, 'P', '1') => Some("C1W"),
        (GnssSystem::Gps, 'P', '2') => Some("C2W"),
        (GnssSystem::Glonass, 'C', '1') => Some("C1C"),
        (GnssSystem::Glonass, 'C', '2') => Some("C2C"),
        (GnssSystem::Glonass, 'P', '1') => Some("C1P"),
        (GnssSystem::Glonass, 'P', '2') => Some("C2P"),
        (GnssSystem::Galileo, 'C', '1') => Some("C1C"),
        (GnssSystem::Galileo, 'C', '2') => Some("C5Q"),
        (GnssSystem::Galileo, 'P', '1') => Some("C1X"),
        (GnssSystem::Galileo, 'P', '2') => Some("C5X"),
        (GnssSystem::BeiDou, 'C', '1') => Some("C2I"),
        (GnssSystem::BeiDou, 'C', '2') => Some("C7I"),
        (GnssSystem::BeiDou, 'P', '1') => Some("C2I"),
        (GnssSystem::BeiDou, 'P', '2') => Some("C6I"),
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
        GnssSystem::Galileo => match band {
            '1' | '6' => 'C',
            '5' | '7' | '8' => 'X',
            _ => 'X',
        },
        GnssSystem::BeiDou => match band {
            '2' | '6' | '7' => 'I',
            '1' => 'P',
            '5' | '8' => 'X',
            _ => 'X',
        },
        GnssSystem::Qzss => match band {
            '1' => 'C',
            '2' => 'L',
            '5' | '6' => 'X',
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
    let tokens: Vec<&str> = body.split_whitespace().collect();
    if tokens.len() < fields.len() {
        let field = fields[tokens.len()];
        return Err(map_field_error(FieldError::Missing { field }, line));
    }
    Ok([
        strict_f64_token(tokens[0], fields[0], line)?,
        strict_f64_token(tokens[1], fields[1], line)?,
        strict_f64_token(tokens[2], fields[2], line)?,
    ])
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
mod tests;
