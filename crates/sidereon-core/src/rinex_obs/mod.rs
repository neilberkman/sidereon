//! RINEX 3.0x observation-file parser and single-frequency pseudorange
//! extraction.
//!
//! Parses a RINEX **version 3** observation file (`OBSERVATION DATA`) into a
//! typed [`RinexObs`] product: the header (including the surveyed
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
//!   load-bearing ones are `RINEX VERSION / TYPE` (must be observation, major 3),
//!   `APPROX POSITION XYZ`, `ANTENNA: DELTA H/E/N`, `SYS / # / OBS TYPES` (the
//!   per-system code list, order-preserving, with continuation lines),
//!   `SYS / SCALE FACTOR`, `SYS / PHASE SHIFT`, `TIME OF FIRST OBS` (+ time
//!   system), `INTERVAL`, and the optional `GLONASS SLOT / FRQ #`.
//! - The body is per-epoch: a `>`-prefixed epoch line carrying the civil time,
//!   an event flag, and the satellite count, then one logical record per
//!   satellite with each observation as a 16-column `F14.3` value + LLI + SSI
//!   field, in the order the system's `SYS / # / OBS TYPES` list declares. A
//!   logical satellite record may wrap across 80-column continuation lines.

use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::astro::time::model::TimeScale;

use crate::format::columns::raw_field as field;
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

/// A civil epoch as it appears on a RINEX observation epoch line, in the file's
/// own time scale (no leap-second shifting). This is the natural boundary for
/// the solver, which derives seconds-of-J2000 / second-of-day / day-of-year
/// from the civil components.
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// One epoch record: the civil time, the event flag, and the per-satellite
/// observation values (aligned to that system's `SYS / # / OBS TYPES` order).
#[derive(Debug, Clone, PartialEq)]
pub struct ObsEpoch {
    /// Civil epoch in the header time scale.
    pub epoch: ObsEpochTime,
    /// Epoch flag: 0 = OK, 1 = power failure, >1 = an event record (skipped).
    pub flag: u8,
    /// Satellite → observation values, ascending satellite id. The value vector
    /// is index-aligned to [`ObsHeader::obs_codes`] for that satellite's system.
    pub sats: BTreeMap<GnssSatelliteId, Vec<ObsValue>>,
}

/// Parsed RINEX 3 observation header.
#[derive(Debug, Clone, PartialEq)]
pub struct ObsHeader {
    /// The full RINEX version (e.g. `3.05`); the major must be 3.
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
    /// Nominal epoch spacing in seconds (`INTERVAL`), if present.
    pub interval_s: Option<f64>,
    /// First observation epoch and its time system (`TIME OF FIRST OBS`).
    pub time_of_first_obs: Option<(ObsEpochTime, TimeScale)>,
    /// Carrier phase-shift records (`SYS / PHASE SHIFT`), in header order.
    pub phase_shifts: Vec<ObsPhaseShift>,
    /// Observation scale-factor records (`SYS / SCALE FACTOR`), in header order.
    pub scale_factors: Vec<ObsScaleFactor>,
    /// GLONASS slot → frequency channel map (`GLONASS SLOT / FRQ #`), if present.
    pub glonass_slots: BTreeMap<u8, i8>,
    /// Marker (station) name, if present.
    pub marker_name: Option<String>,
}

/// A parsed RINEX 3 observation product.
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
    /// Parse RINEX 3 observation text into a typed product.
    ///
    /// Returns [`Error::Parse`] if the file is not observation data, is not RINEX
    /// major version 3, is missing a required header record, or has a malformed
    /// epoch record.
    pub fn parse(text: &str) -> Result<Self> {
        let mut parser = Parser::new();
        let mut lines = text.lines();
        parser.parse_header(&mut lines)?;
        parser.parse_body(&mut lines.peekable())?;
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
    phase_shifts: Vec<ObsPhaseShift>,
    scale_factors: Vec<ObsScaleFactor>,
    scale_factor_continuation: Option<ScaleFactorContinuation>,
    glonass_slots: BTreeMap<u8, i8>,
    glonass_slots_remaining: Option<usize>,
    marker_name: Option<String>,
    epochs: Vec<ObsEpoch>,
    /// The constellation whose `SYS / # / OBS TYPES` list is currently being
    /// filled (for continuation lines).
    current_obs_sys: Option<GnssSystem>,
    /// Number of codes still expected for `current_obs_sys`.
    obs_codes_remaining: usize,
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
            phase_shifts: Vec::new(),
            scale_factors: Vec::new(),
            scale_factor_continuation: None,
            glonass_slots: BTreeMap::new(),
            glonass_slots_remaining: None,
            marker_name: None,
            epochs: Vec::new(),
            current_obs_sys: None,
            obs_codes_remaining: 0,
            diagnostics: Diagnostics::new(),
        }
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
            let line = raw.trim_end_matches(['\r', '\n']);
            let label = field(line, 60, 80).trim();
            match label {
                "RINEX VERSION / TYPE" => self.parse_version(line)?,
                "APPROX POSITION XYZ" => self.parse_approx_position(line)?,
                "ANTENNA: DELTA H/E/N" => self.parse_antenna_delta(line)?,
                "SYS / # / OBS TYPES" => self.parse_obs_types(line)?,
                "SYS / SCALE FACTOR" => self.parse_scale_factor(line)?,
                "SYS / PHASE SHIFT" => self.parse_phase_shift(line)?,
                "TIME OF FIRST OBS" => self.parse_time_of_first_obs(line)?,
                "INTERVAL" => {
                    self.interval_s = Some(strict_f64_field(line, 0, 10, "interval_s")?);
                }
                "GLONASS SLOT / FRQ #" => self.parse_glonass_slots(line)?,
                "MARKER NAME" => {
                    let name = field(line, 0, 60).trim();
                    if !name.is_empty() {
                        self.marker_name = Some(name.to_string());
                    }
                }
                "END OF HEADER" => {
                    self.ensure_obs_type_count_complete(line)?;
                    self.ensure_scale_factor_count_complete(line)?;
                    saw_end = true;
                    break;
                }
                // Every other header record is tolerated and skipped.
                _ => {}
            }
        }
        if !saw_end {
            return Err(Error::Parse("RINEX OBS header has no END OF HEADER".into()));
        }
        Ok(())
    }

    fn parse_version(&mut self, line: &str) -> Result<()> {
        let version = field(line, 0, 20).trim();
        let version = strict_f64_token(version, "version", line)?;
        // The file type letter is at column 20; observation files carry 'O'.
        let type_field = field(line, 20, 40);
        self.is_observation =
            type_field.trim_start().starts_with('O') || type_field.contains("OBSERVATION");
        if !self.is_observation {
            return Err(Error::Parse(format!(
                "RINEX file is not observation data: {type_field:?}"
            )));
        }
        if version.floor() as i64 != 3 {
            return Err(Error::Parse(format!(
                "RINEX OBS parser requires major version 3, got {version}"
            )));
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

    fn parse_obs_types(&mut self, line: &str) -> Result<()> {
        // A new system line carries its letter at column 0 and the count at
        // columns 3..6; a continuation line has a blank system field and only
        // adds more codes to the current system.
        let sys_field = field(line, 0, 1).trim();
        if !sys_field.is_empty() {
            self.ensure_obs_type_count_complete(line)?;
            let letter = sys_field.chars().next().unwrap();
            let system = GnssSystem::from_letter(letter).ok_or_else(|| {
                Error::Parse(format!("RINEX OBS unknown system letter {letter:?}"))
            })?;
            let count = strict_int_field::<usize>(line, 3, 6, "obs_type_count")?;
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
            list.push(tok.to_string());
            self.obs_codes_remaining -= 1;
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
        let code = tokens[1].to_string();
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

        self.phase_shifts.push(ObsPhaseShift {
            system,
            code,
            correction_cycles,
            satellites,
        });
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
            record.codes.push(code.to_string());
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
        let body = field(line, 0, 43);
        let scale_label = field(line, 48, 51).trim();
        let scale = time_scale_from_label(scale_label, line)?;
        let epoch = parse_epoch_time_tokens(
            body,
            line,
            [
                "time_of_first_obs.year",
                "time_of_first_obs.month",
                "time_of_first_obs.day",
                "time_of_first_obs.hour",
                "time_of_first_obs.minute",
                "time_of_first_obs.second",
            ],
            civil_second_policy_for_time_scale(scale),
        )?;
        self.time_of_first_obs = Some((epoch, scale));
        Ok(())
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
        for pair in tokens.chunks_exact(2) {
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
            let (epoch_time, flag, numsat) =
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
                sats,
            });
        }
        Ok(())
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
        if self.obs_codes.is_empty() {
            return Err(Error::Parse(
                "RINEX OBS header has no SYS / # / OBS TYPES records".into(),
            ));
        }
        let header = ObsHeader {
            version,
            approx_position_m: self.approx_position_m,
            antenna_delta_hen_m: self.antenna_delta_hen_m,
            obs_codes: self.obs_codes,
            interval_s: self.interval_s,
            time_of_first_obs: self.time_of_first_obs,
            phase_shifts: self.phase_shifts,
            scale_factors: self.scale_factors,
            glonass_slots: self.glonass_slots,
            marker_name: self.marker_name,
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

/// Parse a RINEX-3 epoch line `> YYYY MM DD HH MM SS.sssssss  F NN [clock]`,
/// returning the civil time, event flag, and satellite count.
fn parse_epoch_line(
    line: &str,
    second_policy: validate::CivilSecondPolicy,
) -> Result<(ObsEpochTime, u8, usize)> {
    // The date occupies a fixed width after the leading '>'; the flag is at
    // column 31 and the satellite count at columns 32..35.
    let date_body = field(line, 1, 29);
    let epoch = parse_epoch_time_tokens(
        date_body,
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
    let flag = strict_int_field::<u8>(line, 31, 32, "epoch.flag")?;
    let numsat = strict_int_field::<usize>(line, 32, 35, "epoch.satellite_count")?;
    Ok((epoch, flag, numsat))
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
        | TimeScale::Tdb
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

fn strict_f64_field(line: &str, start: usize, end: usize, field_name: &'static str) -> Result<f64> {
    strict_f64_token(field(line, start, end), field_name, line)
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
/// letter followed by two PRN digits), whether or not it parses to a
/// representable [`GnssSatelliteId`]. Used to find satellite-record boundaries
/// when skipping an unknown/out-of-range record, so that a following
/// unrepresentable record (e.g. another extended GLONASS slot) is not mistaken
/// for a wrapped continuation line. Observation continuation lines begin with a
/// right-justified numeric field, never a letter, so they never match.
fn starts_with_sat_designator(line: &str) -> bool {
    let b = line.as_bytes();
    b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1].is_ascii_digit() && b[2].is_ascii_digit()
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
