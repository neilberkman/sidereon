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
/// Width of a `GLONASS COD/PHS/BIS` code (`A3`).
const GLONASS_BIAS_CODE_WIDTH: usize = 3;
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
/// The same line with RINEX 4.02's five further digits of the second after the
/// clock offset, `1X,I5.5`.
const V4_EPOCH_COLUMNS: [(usize, usize); 11] = [
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
    (57, 62),
];
/// The same line carrying the picosecond field where this writer used to place
/// it, after the seconds, shifting every later field six columns right. Still
/// read; no longer written.
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
/// Width and decimals of a `SYS / PHASE SHIFT` correction, `F8.5`.
const PHASE_SHIFT_CORRECTION_WIDTH: usize = 8;
const PHASE_SHIFT_CORRECTION_DECIMALS: usize = 5;
/// Largest satellite count a `SYS / PHASE SHIFT` record's `I2.2` field holds.
const MAX_PHASE_SHIFT_SATELLITES: usize = 99;
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

impl ObsHeader {
    /// The `GLONASS COD/PHS/BIS` code-phase bias the header gives a GLONASS
    /// signal, in metres, or `None` where it gives none. From RINEX 4.00, whose
    /// Table A2 says the record's "lines should be ignored by RINEX decoders and
    /// encoders", it gives none.
    ///
    /// # Errors
    ///
    /// [`CorrectionUnavailable::Unknown`] for a blank record, "If the GLONASS
    /// code phase alignment is unknown, then all fields within GLONASS
    /// COD/PHS/BIS header record are left blank" (RINEX 3.05 section 5.2.16),
    /// or a blank bias for the signal; [`CorrectionUnavailable::Ambiguous`]
    /// where one header block gives the signal different biases.
    pub fn glonass_code_phase_bias(
        &self,
        code: &str,
    ) -> core::result::Result<Option<f64>, CorrectionUnavailable> {
        if records_deprecated_in_rinex4(self.version) {
            return Ok(None);
        }
        let Some(entries) = &self.glonass_cod_phs_bis else {
            return Ok(None);
        };
        if entries.is_empty() {
            return Err(CorrectionUnavailable::Unknown);
        }
        let mut distinct: Vec<Option<f64>> = Vec::new();
        for (entry_code, bias) in entries {
            if entry_code == code && !distinct.contains(bias) {
                distinct.push(*bias);
            }
        }
        match distinct.as_slice() {
            [] => Ok(None),
            [Some(bias)] => Ok(Some(*bias)),
            [None] => Err(CorrectionUnavailable::Unknown),
            _ => Err(CorrectionUnavailable::Ambiguous {
                corrections: distinct,
            }),
        }
    }
}

/// One `SYS / PHASE SHIFT` header record.
#[derive(Debug, Clone, PartialEq)]
pub struct ObsPhaseShift {
    /// Constellation the phase-shift record applies to.
    pub system: GnssSystem,
    /// RINEX carrier observable code, e.g. `L1C`, or `None` for a record naming
    /// only its constellation. RINEX 3.05 section 5.2.12: "If the applied phase
    /// corrections or the phase alignment is unknown, then the observation code
    /// field and the rest of the SYS / PHASE SHIFT header record field of the
    /// respective satellite system(s) are left blank. This use case is intended
    /// for exceptional situations where the data is intended for special
    /// projects and analysis." Such a record gives no correction; a
    /// satellite's code no other record covers reads as
    /// [`CorrectionUnavailable::Unknown`].
    pub code: Option<String>,
    /// Phase correction in carrier cycles, or `None` where the record leaves
    /// the field blank: "Correction applied (cycles) or blank if none" (RINEX
    /// 3.05 and 4.02 Table A2).
    pub correction_cycles: Option<f64>,
    /// Optional satellite restriction. Empty, with no
    /// [`ObsPhaseShift::unrepresentable_satellites`] either, means the
    /// correction applies to all satellites of the system/code.
    pub satellites: Vec<GnssSatelliteId>,
    /// Satellites the record names by a well-formed RINEX designator that
    /// [`GnssSatelliteId`] does not hold, such as `R00`, as written. No
    /// observation of theirs is kept, so the correction applies to none of
    /// them; they are kept so the record is written back whole, and each is
    /// counted in [`RinexObs::skipped_records`].
    pub unrepresentable_satellites: Vec<String>,
}

impl ObsPhaseShift {
    /// Whether the record applies to every satellite of its system and code:
    /// it names no satellite, representable or not.
    pub fn covers_every_satellite(&self) -> bool {
        self.satellites.is_empty() && self.unrepresentable_satellites.is_empty()
    }

    /// How many satellites the record names, representable or not.
    pub fn satellite_count(&self) -> usize {
        self.satellites.len() + self.unrepresentable_satellites.len()
    }
}

/// Why a header states no one correction for a satellite's signal.
#[derive(Debug, Clone, PartialEq)]
pub enum CorrectionUnavailable {
    /// Records in one header block give the signal different corrections. RINEX
    /// gives a block's records no order to choose one by; the records are kept
    /// as written. A blank value is `None`.
    Ambiguous {
        /// The different corrections, in the order the records give them.
        corrections: Vec<Option<f64>>,
    },
    /// The header declares the correction unknown: a `SYS / PHASE SHIFT`
    /// record naming only the constellation, or a blank `GLONASS COD/PHS/BIS`
    /// record or bias.
    Unknown,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObsLeapSeconds {
    /// Current leap-second count.
    pub current: i64,
    /// Future/past delta field, if present.
    pub delta_future: Option<i64>,
    /// GPS week field, if present.
    pub week: Option<i64>,
    /// Day field, if present.
    pub day: Option<i64>,
    /// Optional time system identifier (`GPS`, `BDS`, `BDT`) from columns 25..27.
    ///
    /// RINEX 3.03 introduced `BDS` and `GPS`, with RINEX 3.05 renaming `BDS` to `BDT`.
    /// RINEX 4 permits only `GPS`. As an accepted extension, `GPS` is also accepted
    /// in pre-3.03 versions (including RINEX 2) where files populate this optional field.
    pub time_system: Option<String>,
}

/// The validation status of a `LEAP SECONDS` time system identifier token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LeapTimeSystemValidity {
    /// Token is valid and supported in the given format version.
    Valid,
    /// Token is a recognized time system identifier, but not supported in the given format version.
    UnsupportedInVersion,
    /// Token is not a recognized time system identifier (`GPS`, `BDS`, `BDT`).
    UnknownToken,
}

/// Validate a `LEAP SECONDS` time system identifier token against a format version.
///
/// Recognized identifiers are `GPS`, `BDS` (RINEX 3.03–3.04), and `BDT` (RINEX 3.05;
/// excluded in RINEX 4, which permits only `GPS`). As an accepted extension, `GPS` is
/// also accepted in pre-3.03 versions (including RINEX 2).
pub(crate) fn check_leap_seconds_time_system(token: &str, version: f64) -> LeapTimeSystemValidity {
    if !matches!(token, "GPS" | "BDS" | "BDT") {
        return LeapTimeSystemValidity::UnknownToken;
    }
    let supported = if version < 3.025 {
        token == "GPS"
    } else if (3.025..3.045).contains(&version) {
        matches!(token, "GPS" | "BDS")
    } else if (3.045..3.99).contains(&version) {
        matches!(token, "GPS" | "BDT")
    } else {
        token == "GPS"
    };
    if supported {
        LeapTimeSystemValidity::Valid
    } else {
        LeapTimeSystemValidity::UnsupportedInVersion
    }
}

/// One epoch record: the civil time, the event flag, and the per-satellite
/// observation values (aligned to that system's `SYS / # / OBS TYPES` order).
#[derive(Debug, Clone, PartialEq)]
pub struct ObsEpoch {
    /// Civil epoch in the header time scale, or `None` for an event whose epoch
    /// fields are blank.
    ///
    /// RINEX lets an event without a significant epoch leave its epoch fields
    /// blank. An observation epoch and a cycle slip epoch always carry one.
    pub epoch: Option<ObsEpochTime>,
    /// Epoch flag: 0 = OK, 1 = power failure, 6 = cycle slip records, whose
    /// slips are in [`ObsEpoch::cycle_slips`], and any other flag above 1 an
    /// event, whose own records are in [`ObsEpoch::special_records`].
    pub flag: u8,
    /// Optional receiver clock offset from the epoch line, seconds.
    pub rcv_clock_offset_s: Option<f64>,
    /// Optional RINEX 4 epoch picosecond extension.
    pub epoch_picoseconds: Option<u32>,
    /// Satellite/special-record count declared on the epoch line.
    pub declared_record_count: usize,
    /// The records an event epoch (flag above 1 other than 6) carried, as they
    /// were written.
    ///
    /// A flag 3 epoch is followed by the header records for a new site
    /// occupation - its marker, antenna and position - and a flag 4 epoch by
    /// header records or comments. They are kept verbatim rather than parsed,
    /// because what they mean depends on the labels they carry, and written back
    /// unchanged. Empty for an observation epoch and a cycle slip epoch.
    pub special_records: Vec<String>,
    /// Satellite → observation values, ascending satellite id. The value vector
    /// is index-aligned to [`ObsHeader::obs_codes`] for that satellite's system,
    /// blank under a code the list in effect at this epoch does not declare.
    /// Empty for an event epoch and a cycle slip epoch.
    pub sats: BTreeMap<GnssSatelliteId, Vec<ObsValue>>,
    /// Satellite → cycle slips a flag 6 epoch reports, ascending satellite id,
    /// index-aligned to [`ObsHeader::obs_codes`] as [`ObsEpoch::sats`] is.
    ///
    /// RINEX writes detected and repaired cycle slips in the observation record
    /// layout, with the slip in place of the observation and the loss-of-lock
    /// and signal-strength indicators blank or zero. They are slips, not
    /// measurements, so they are held apart from the observations. Empty for
    /// every epoch whose flag is not 6.
    pub cycle_slips: BTreeMap<GnssSatelliteId, Vec<ObsValue>>,
}

/// The epoch flag whose records report cycle slips.
pub const CYCLE_SLIP_FLAG: u8 = 6;

/// Whether an epoch flag marks an event, whose records are
/// [`ObsEpoch::special_records`]: every flag above 1 except the cycle slip flag.
pub(crate) fn is_event_flag(flag: u8) -> bool {
    flag > 1 && flag != CYCLE_SLIP_FLAG
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
    /// Per-constellation observation codes: the union of every list the file
    /// declares for the constellation, in its header and after its events. The
    /// file header's codes come first, in its order, then each code a later list
    /// declares that the union does not yet hold, in the order first declared.
    /// Observation values and cycle slips are index-aligned to these lists, blank
    /// under a code the list in effect at their epoch does not declare.
    pub obs_codes: BTreeMap<GnssSystem, Vec<String>>,
    /// Per-constellation code lists this header itself declares, in declared
    /// order. In the file header they are the lists its `SYS / # / OBS TYPES`
    /// records declare, or at version 2 what `rinex2_types` reads as for each
    /// constellation in `obs_codes`; in a header from [`RinexObs::header_at`]
    /// they are the lists in effect at that epoch. A product whose events
    /// declare no list holds the same lists here as in `obs_codes`.
    pub declared_obs_codes: BTreeMap<GnssSystem, Vec<String>>,
    /// The version 2 `# / TYPES OF OBSERV` names as read, in order, and empty at
    /// version 3. In a header from [`RinexObs::header_at`], the names in effect
    /// at that epoch. A version 2 file lists one set of names for every
    /// constellation, so a constellation's codes are what these names read as
    /// for it - including a constellation a `PRN / # OF OBS` count names that
    /// holds no list in [`ObsHeader::obs_codes`] because no observation names
    /// it.
    pub rinex2_types: Vec<String>,
    /// The constellation a version 2 file's version record names, and `None`
    /// for a mixed file or at version 3. With no observations, a version 2 file
    /// states the list of this constellation, or GPS for a mixed file.
    pub rinex2_system: Option<GnssSystem>,
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
    pub glonass_cod_phs_bis: Option<Vec<(String, Option<f64>)>>,
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
    /// Epoch records in file order. Event records and cycle slip records (flag
    /// above 1) are retained with an empty satellite map so epoch indices stay
    /// stable.
    pub epochs: Vec<ObsEpoch>,
    /// Count of records skipped because their satellite token did not parse to a
    /// representable [`GnssSatelliteId`]: an entry in the `GLONASS SLOT / FRQ #`
    /// header table, or a satellite record inside an epoch. The token range is
    /// `01..=99` for every constellation letter, which covers the extended
    /// slots real products carry, so what lands here is a designator naming no
    /// satellite - `R00` - or a token that is not a designator at all. One such
    /// record is skipped rather than aborting the
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
        let beidou = if crate::frequencies::is_rinex_302(version) {
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
    /// reconstructing originals and is not re-applied here. No record, or a
    /// blank correction, is 0. From RINEX 4.00 the record "should be ignored by
    /// RINEX decoders and encoders" (Table A2), so a file of that version reports
    /// 0 whatever its records say; the records are still kept and written back.
    ///
    /// [`CorrectionUnavailable::Ambiguous`] where records in one header block
    /// give the satellite's code different corrections, and
    /// [`CorrectionUnavailable::Unknown`] where the only record covering it
    /// names just the constellation, which declares the alignment unknown.
    pub phase_shift_cycles: core::result::Result<f64, CorrectionUnavailable>,
}

/// Return labelled raw observation rows for one epoch, grouped by satellite.
pub fn observation_values(
    obs: &RinexObs,
    epoch: &ObsEpoch,
    filter: &ObservationFilter,
) -> Result<Vec<(GnssSatelliteId, Vec<ObservationValueRow>)>> {
    labelled_values(&obs.header, epoch, filter)
}

/// Labelled rows for one epoch by a header's code lists. Every header in
/// effect holds the product's union, which the values are aligned to.
fn labelled_values(
    header: &ObsHeader,
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
        let Some(code_list) = header.obs_codes.get(&sat.system) else {
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
///
/// `header` is the header in effect at the epoch, from [`RinexObs::header_at`]
/// or [`RinexObs::header_timeline`]. A phase shift or GLONASS channel an event
/// declares applies to the epochs after it, where the file header still holds
/// the value from before it.
pub fn carrier_phase_rows(
    header: &ObsHeader,
    epoch: &ObsEpoch,
    filter: &ObservationFilter,
) -> Result<Vec<(GnssSatelliteId, Vec<CarrierPhaseRow>)>> {
    validate_finite_input(header.version, "version")?;
    let mut out = Vec::new();
    for (sat, rows) in labelled_values(header, epoch, filter)? {
        let phases = rows
            .into_iter()
            .filter(|row| row.kind == ObservationKind::CarrierPhase)
            .map(|row| carrier_phase_row(header, sat, row))
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
    header: &ObsHeader,
    sat: GnssSatelliteId,
    row: ObservationValueRow,
) -> Result<CarrierPhaseRow> {
    let glonass_channel = header.glonass_slots.get(&sat.prn).copied();
    let frequency_hz =
        observation_frequency_hz(sat.system, &row.code, header.version, glonass_channel)?;
    let phase_shift_cycles = phase_shift_cycles(header, sat, &row.code);
    let value_cycles = row.value;
    let wavelength_m =
        rinex_observation_wavelength_m(sat.system, &row.code, header.version, glonass_channel);
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

/// The phase shift in effect for a satellite's code: the records for the code
/// naming the satellite, else the records for every satellite of the system,
/// else a record naming only the constellation, which declares the alignment
/// unknown, whatever their order; records of one kind that give different
/// corrections are ambiguous. RINEX does not say which of the two applies; taking the
/// record naming the satellite is this reader's policy, chosen to be consistent
/// with the free ordering of header records (RINEX 3.05 section 5.2.1), under
/// which no record's position can decide it. A blank correction is none. From
/// RINEX 4.00 the records are ignored and no correction is in effect.
fn phase_shift_cycles(
    header: &ObsHeader,
    sat: GnssSatelliteId,
    code: &str,
) -> core::result::Result<f64, CorrectionUnavailable> {
    if records_deprecated_in_rinex4(header.version) {
        return Ok(0.0);
    }
    let covering = |naming: bool| {
        header
            .phase_shifts
            .iter()
            .filter(move |shift| {
                shift.system == sat.system
                    && shift.code.as_deref() == Some(code)
                    && if naming {
                        shift.satellites.contains(&sat)
                    } else {
                        shift.covers_every_satellite()
                    }
            })
            .map(|shift| shift.correction_cycles)
    };
    if let Some(correction) = one_correction(covering(true))? {
        return Ok(correction);
    }
    if let Some(correction) = one_correction(covering(false))? {
        return Ok(correction);
    }
    if header
        .phase_shifts
        .iter()
        .any(|shift| shift.system == sat.system && shift.code.is_none())
    {
        return Err(CorrectionUnavailable::Unknown);
    }
    Ok(0.0)
}

/// The one correction a set of covering records gives, a blank one being 0;
/// `None` for no record; the corrections as written where they differ.
fn one_correction(
    corrections: impl Iterator<Item = Option<f64>>,
) -> core::result::Result<Option<f64>, CorrectionUnavailable> {
    let mut distinct: Vec<Option<f64>> = Vec::new();
    for correction in corrections {
        let value = correction.unwrap_or(0.0);
        if !distinct.iter().any(|held| held.unwrap_or(0.0) == value) {
            distinct.push(correction);
        }
    }
    match distinct.as_slice() {
        [] => Ok(None),
        [one] => Ok(Some(one.unwrap_or(0.0))),
        _ => Err(CorrectionUnavailable::Ambiguous {
            corrections: distinct,
        }),
    }
}
/// Whether a version's `SYS / PHASE SHIFT` and `GLONASS COD/PHS/BIS` records are
/// deprecated. RINEX 4.00, 4.01 and 4.02 Table A2 say each "is strongly
/// deprecated. It is allowed in this version for compatibility with previous
/// RINEX versions but the lines should be ignored by RINEX decoders and
/// encoders."
pub(crate) fn records_deprecated_in_rinex4(version: f64) -> bool {
    version >= 4.0
}

/// The scale factor in effect for a code: the record naming the code, else the
/// record for every code of the system, whatever their order, or 1. As for
/// phase shifts, taking the record naming the code is this reader's policy, not
/// a rule RINEX states.
pub(crate) fn scale_factor_in(factors: &[ObsScaleFactor], system: GnssSystem, code: &str) -> f64 {
    factors
        .iter()
        .find(|record| record.system == system && record.codes.iter().any(|c| c == code))
        .or_else(|| {
            factors
                .iter()
                .find(|record| record.system == system && record.codes.is_empty())
        })
        .map_or(1.0, |record| record.factor)
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
    /// `PRN / # OF OBS` records as read, applied once the header ends: a count
    /// only means something against its constellation's types, and a header
    /// may declare those after the counts, or declare them again.
    prn_obs_count_lines: Vec<String>,
    phase_shifts: Vec<ObsPhaseShift>,
    scale_factors: Vec<ObsScaleFactor>,
    scale_factor_continuation: Option<ScaleFactorContinuation>,
    glonass_slots: BTreeMap<u8, i8>,
    glonass_slots_remaining: Option<usize>,
    glonass_cod_phs_bis: Option<Vec<(String, Option<f64>)>>,
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
    /// Whether a `# / TYPES OF OBSERV` record with a count was read.
    rinex2_obs_types_declared: bool,
    /// Satellites the last `SYS / PHASE SHIFT` record's count declares that its
    /// records have not yet listed.
    phase_shift_satellites_remaining: usize,
    /// The GLONASS slots this header block has declared, to refuse a slot it
    /// gives two channels.
    glonass_slots_in_block: BTreeMap<u8, i8>,
    /// The GLONASS code-phase biases this header block has declared, and
    /// whether it declared the biases unknown with a blank record.
    glonass_biases_in_block: (
        BTreeMap<String, Option<f64>>,
        bool,
        std::collections::BTreeSet<String>,
    ),
    /// The value each record of a label holding one value set in this header
    /// block, as its parsed value reads, to refuse a record setting another.
    single_values_in_block: BTreeMap<String, SingleValue>,
    /// The header the file declares, as it stands at `END OF HEADER`, before
    /// any event lays records over it.
    file_header: Option<ObsHeader>,
    /// The header in effect for the epochs being read: the file header with
    /// every event read so far laid over it.
    effective: Option<ObsHeader>,
    /// The code lists in effect for each stretch of epochs, the file header's
    /// first, then each an event declared.
    code_list_segments: Vec<CodeListSegment>,
    /// For each epoch read, the stretch of code lists it was read by.
    epoch_segments: Vec<usize>,
    /// The constellations a version 2 file's records name.
    rinex2_systems: std::collections::BTreeSet<GnssSystem>,
}

/// The code lists in effect for a stretch of epochs: the file header's, or the
/// ones an event declared. A version 3 file declares a list per constellation,
/// a version 2 file one list of names.
#[derive(Debug, Clone)]
struct CodeListSegment {
    lists: BTreeMap<GnssSystem, Vec<String>>,
    names: Vec<String>,
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
            prn_obs_count_lines: Vec::new(),
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
            rinex2_obs_types_declared: false,
            phase_shift_satellites_remaining: 0,
            glonass_slots_in_block: BTreeMap::new(),
            glonass_biases_in_block: (BTreeMap::new(), false, std::collections::BTreeSet::new()),
            single_values_in_block: BTreeMap::new(),
            file_header: None,
            effective: None,
            code_list_segments: Vec::new(),
            epoch_segments: Vec::new(),
            rinex2_systems: std::collections::BTreeSet::new(),
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
            // A version 2 file's observation records are laid out by its own
            // `# / TYPES OF OBSERV` list; this version 3 record does not
            // describe them, so it is reported as not retained.
            if label == "SYS / # / OBS TYPES" && self.is_rinex2() {
                self.unretained_header_labels.push(label.to_string());
                continue;
            }
            // The records an event may also carry have one reader, so an event's
            // records are read as the file header's are.
            if self.parse_layered_record(label, line)? {
                self.check_single_value(label, line)?;
                continue;
            }
            match label {
                "RINEX VERSION / TYPE" => {
                    self.parse_version(line)?;
                    self.check_single_value(label, line)?;
                }
                "PGM / RUN BY / DATE" => self.parse_pgm_run_by_date(line),
                "COMMENT" => self.comments.push(field(line, 0, 60).trim().to_string()),
                "TIME OF FIRST OBS" => {
                    self.parse_time_of_first_obs(line)?;
                    self.check_single_value(label, line)?;
                }
                "TIME OF LAST OBS" => {
                    self.parse_time_of_last_obs(line)?;
                    self.check_single_value(label, line)?;
                }
                "LEAP SECONDS" => {
                    self.parse_leap_seconds(line)?;
                    self.check_single_value(label, line)?;
                }
                "# OF SATELLITES" => {
                    self.n_satellites =
                        Some(strict_int_field::<usize>(line, 0, 6, "n_satellites")?);
                    self.check_single_value(label, line)?;
                }
                "PRN / # OF OBS" => {
                    // A count only means something against its constellation's
                    // types, which a header may declare after the counts or
                    // declare again, so every count is read against the lists
                    // the header ends with. Its fields are checked where it
                    // stands, so a malformed count is still the error reported.
                    self.check_prn_obs_count_fields(line)?;
                    self.prn_obs_count_lines.push(line.to_string());
                }
                "END OF HEADER" => {
                    // Version 3 type records read before the version record said
                    // version 2 do not describe its observation records either.
                    if self.is_rinex2()
                        && (!self.obs_codes.is_empty() || self.obs_codes_remaining > 0)
                    {
                        self.obs_codes.clear();
                        self.obs_codes_remaining = 0;
                        self.current_obs_sys = None;
                        self.unretained_header_labels
                            .push("SYS / # / OBS TYPES".to_string());
                    }
                    self.ensure_obs_type_count_complete(line)?;
                    self.ensure_obs_type_count_complete_v2(line)?;
                    self.ensure_scale_factor_count_complete(line)?;
                    self.ensure_phase_shift_count_complete(line)?;
                    for skip in phase_shift_contradictions(
                        self.version.unwrap_or_default(),
                        &self.phase_shifts,
                    ) {
                        self.diagnostics.push_skip(skip);
                    }
                    check_block_scale_factors(&self.scale_factors)?;
                    for count_line in std::mem::take(&mut self.prn_obs_count_lines) {
                        self.parse_prn_obs_counts(&count_line)?;
                    }
                    self.begin_body()?;
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

    /// Read one of the header records an event may also carry, as the file
    /// header reads it. Returns `false`, reading nothing, for any other label.
    fn parse_layered_record(&mut self, label: &str, line: &str) -> Result<bool> {
        match label {
            "APPROX POSITION XYZ" => self.parse_approx_position(line)?,
            "ANTENNA: DELTA H/E/N" => self.parse_antenna_delta(line)?,
            "SYS / # / OBS TYPES" => self.parse_obs_types(line)?,
            "# / TYPES OF OBSERV" => self.parse_obs_types_v2(line)?,
            "SYS / SCALE FACTOR" => self.parse_scale_factor(line)?,
            "SYS / PHASE SHIFT" => self.parse_phase_shift(line)?,
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
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Keep the header the file declares, as it stands at `END OF HEADER`, and
    /// start the header in effect for the body from it.
    fn begin_body(&mut self) -> Result<()> {
        let version = self
            .version
            .ok_or_else(|| Error::Parse("RINEX OBS missing RINEX VERSION / TYPE".into()))?;
        let rinex2 = version.floor() as i64 == 2;
        let header = ObsHeader {
            version,
            approx_position_m: self.approx_position_m,
            antenna_delta_hen_m: self.antenna_delta_hen_m,
            obs_codes: self.obs_codes.clone(),
            declared_obs_codes: self.obs_codes.clone(),
            rinex2_types: if rinex2 {
                self.rinex2_obs_codes.clone()
            } else {
                Vec::new()
            },
            rinex2_system: if rinex2 {
                self.rinex2_default_system
            } else {
                None
            },
            program_run_by_date: self.program_run_by_date.clone(),
            comments: self.comments.clone(),
            marker_number: self.marker_number.clone(),
            marker_type: self.marker_type.clone(),
            observer: self.observer.clone(),
            agency: self.agency.clone(),
            receiver: self.receiver.clone(),
            antenna: self.antenna.clone(),
            interval_s: self.interval_s,
            time_of_first_obs: self.time_of_first_obs,
            time_of_last_obs: self.time_of_last_obs,
            n_satellites: self.n_satellites,
            prn_obs_counts: self.prn_obs_counts.clone(),
            phase_shifts: self.phase_shifts.clone(),
            scale_factors: self.scale_factors.clone(),
            glonass_slots: self.glonass_slots.clone(),
            glonass_cod_phs_bis: self.glonass_cod_phs_bis.clone(),
            signal_strength_unit: self.signal_strength_unit.clone(),
            leap_seconds: self.leap_seconds.clone(),
            marker_name: self.marker_name.clone(),
            unretained_header_labels: self.unretained_header_labels.clone(),
        };
        self.code_list_segments.push(CodeListSegment {
            lists: self.obs_codes.clone(),
            names: self.rinex2_obs_codes.clone(),
        });
        self.effective = Some(header.clone());
        self.file_header = Some(header);
        Ok(())
    }

    /// Lay an event's header records over the header in effect, and read the
    /// epochs after it by the code lists and scale factors in effect then.
    fn apply_event(&mut self, records: &[String]) -> Result<()> {
        let epoch_index = self.epochs.len();
        let Some(effective) = self.effective.as_mut() else {
            return Ok(());
        };
        let effect = apply_event_records(effective, records)
            .map_err(|error| event_records_error(epoch_index, error))?;
        let scale_factors = effective.scale_factors.clone();
        let lists = effective.declared_obs_codes.clone();
        let names = effective.rinex2_types.clone();
        if effect.scale_factors {
            self.scale_factors = scale_factors;
        }
        if effect.lists {
            if self.is_rinex2() {
                // Each constellation's list is read from the new names when
                // its next record is.
                self.rinex2_obs_codes = names;
                self.obs_codes.clear();
            } else {
                self.obs_codes = lists;
            }
            self.code_list_segments.push(CodeListSegment {
                lists: self.obs_codes.clone(),
                names: self.rinex2_obs_codes.clone(),
            });
        }
        for skip in effect.skips {
            self.diagnostics.push_skip(skip);
        }
        Ok(())
    }

    /// Keep an epoch, with the stretch of code lists it was read by.
    fn push_epoch(&mut self, epoch: ObsEpoch) {
        self.epochs.push(epoch);
        self.epoch_segments
            .push(self.code_list_segments.len().saturating_sub(1));
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
            // A repeated complete declaration for a constellation adds its codes
            // to the list read so far. RINEX gives each constellation one
            // declaration ("In mixed files: Repeat for each satellite system");
            // adding a repeated one rather than refusing it is this reader's
            // policy.
            self.obs_codes.entry(system).or_default();
        }
        let Some(system) = self.current_obs_sys else {
            // A blank system field continues the list declared before it, and
            // before any there is none: its codes would belong to no system.
            if field(line, 7, 60).trim().is_empty() {
                return Ok(());
            }
            return Err(Error::Parse(format!(
                "RINEX OBS SYS / # / OBS TYPES continuation record continues no declared list, in {line:?}"
            )));
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
            // A blank count continues the list declared before it. With no
            // codes left to declare there is nothing to continue, and codes on
            // the record would be dropped without a word.
            if self.rinex2_obs_codes_remaining == 0 {
                if field(line, OBS_TYPE_V2_COUNT_WIDTH, write::HEADER_CONTENT_WIDTH)
                    .trim()
                    .is_empty()
                {
                    return Ok(());
                }
                return Err(Error::Parse(format!(
                    "RINEX OBS # / TYPES OF OBSERV continuation record continues no declared list with codes left, in {line:?}"
                )));
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
            self.rinex2_obs_types_declared = true;
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
        // `18X,10(1X,A3)`: a record whose first eighteen columns are blank
        // continues the satellite list of the record before it.
        if self.phase_shift_satellites_remaining > 0
            && field(line, 0, write::PHASE_SHIFT_CONTINUATION_COLUMN)
                .trim()
                .is_empty()
        {
            return self.continue_phase_shift_satellites(line);
        }
        self.ensure_phase_shift_count_complete(line)?;
        let content = field(line, 0, 60);
        if content.trim().is_empty() {
            return Ok(());
        }
        // A record naming only its constellation declares the constellation's
        // phase alignment unknown: "the observation code field and the rest of
        // the SYS / PHASE SHIFT header record field of the respective satellite
        // system(s) are left blank" (RINEX 3.05 section 5.2.12).
        let mut tokens = content.split_whitespace();
        if let (Some(only), None) = (tokens.next(), tokens.next()) {
            if let Some(system) = only
                .chars()
                .next()
                .filter(|_| only.len() == 1)
                .and_then(GnssSystem::from_letter)
            {
                self.phase_shifts.push(ObsPhaseShift {
                    system,
                    code: None,
                    correction_cycles: None,
                    satellites: Vec::new(),
                    unrepresentable_satellites: Vec::new(),
                });
                return Ok(());
            }
        }
        // `A1,1X,A3,1X,F8.5,2X,I2.2,10(1X,A3)`: read in its columns where the
        // record is laid out in them, which tells a blank correction before a
        // satellite count from a correction. A record not in its columns is
        // read by the reading its fields agree with.
        let fields = match phase_shift_columns(content) {
            Some(fields) => fields,
            None => loose_phase_shift_fields(content, line)?,
        };

        let system = fields
            .system
            .chars()
            .next()
            .filter(|_| fields.system.len() == 1)
            .and_then(GnssSystem::from_letter)
            .ok_or_else(|| {
                Error::Parse(format!(
                    "RINEX OBS phase-shift system unparsable in {line:?}"
                ))
            })?;
        let code = Some(obs_code_token(fields.code, "SYS / PHASE SHIFT", line)?);
        let correction_cycles = if fields.correction.is_empty() {
            None
        } else {
            let correction =
                strict_f64_token(fields.correction, "phase_shift.correction_cycles", line)?;
            Some(exact_in_field(
                correction,
                PHASE_SHIFT_CORRECTION_WIDTH,
                PHASE_SHIFT_CORRECTION_DECIMALS,
                "phase_shift.correction_cycles",
                line,
            )?)
        };
        let count = if fields.count.is_empty() {
            0
        } else {
            strict_int_token::<usize>(fields.count, "phase_shift.satellite_count", line)?
        };
        if count > MAX_PHASE_SHIFT_SATELLITES {
            return Err(Error::Parse(format!(
                "RINEX OBS phase-shift satellite count {count} exceeds the I2.2 field maximum of {MAX_PHASE_SHIFT_SATELLITES} in {line:?}"
            )));
        }
        // A list longer than its first record holds continues on the records
        // after it; one naming more satellites than its count does not
        // describe itself.
        if fields.satellites.len() > count {
            return Err(Error::Parse(format!(
                "RINEX OBS phase-shift satellite count mismatch in {line:?}"
            )));
        }
        let (satellites, unrepresentable_satellites) =
            phase_shift_satellites(&fields.satellites, line)?;
        for token in &unrepresentable_satellites {
            self.push_unrepresentable_satellite_skip(token);
        }
        let remaining = count - fields.satellites.len();
        self.phase_shifts.push(ObsPhaseShift {
            system,
            code,
            correction_cycles,
            satellites,
            unrepresentable_satellites,
        });
        self.phase_shift_satellites_remaining = remaining;
        Ok(())
    }

    /// Add a continuation record's satellites to the phase shift before it.
    fn continue_phase_shift_satellites(&mut self, line: &str) -> Result<()> {
        let tokens: Vec<&str> = field(line, write::PHASE_SHIFT_CONTINUATION_COLUMN, 60)
            .split_whitespace()
            .collect();
        if tokens.len() > self.phase_shift_satellites_remaining {
            return Err(Error::Parse(format!(
                "RINEX OBS phase-shift satellite count mismatch: the continuation lists more satellites than its record declares, in {line:?}"
            )));
        }
        let (satellites, unrepresentable) = phase_shift_satellites(&tokens, line)?;
        if self.phase_shifts.is_empty() {
            return Err(Error::Parse(format!(
                "RINEX OBS SYS / PHASE SHIFT continuation continues no record, in {line:?}"
            )));
        }
        for token in &unrepresentable {
            self.push_unrepresentable_satellite_skip(token);
        }
        self.phase_shift_satellites_remaining -= tokens.len();
        if let Some(shift) = self.phase_shifts.last_mut() {
            shift.satellites.extend(satellites);
            shift.unrepresentable_satellites.extend(unrepresentable);
        }
        Ok(())
    }

    /// Refuse a `SYS / PHASE SHIFT` record whose continuations ended before its
    /// count's satellites were listed.
    fn ensure_phase_shift_count_complete(&self, line: &str) -> Result<()> {
        if self.phase_shift_satellites_remaining == 0 {
            return Ok(());
        }
        let supplied = self
            .phase_shifts
            .last()
            .map_or(0, ObsPhaseShift::satellite_count);
        let declared = supplied + self.phase_shift_satellites_remaining;
        Err(Error::Parse(format!(
            "RINEX OBS phase-shift satellite count mismatch: SYS / PHASE SHIFT declares {declared} satellites but supplies {supplied} before {line:?}"
        )))
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
            // `GnssSatelliteId` (`R00`, which names no satellite) must not
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
            // The channel is kept as the file states it. `-7..=6` is the FDMA
            // allocation, not the field's syntax: real IGS headers give the
            // extended slot `R28` channel 7. A channel outside the allocation
            // resolves no carrier in SPP and is reported by `rinex_qc`, and
            // dropping it here would lose a stated value or, as a refusal,
            // the whole file.
            let channel = strict_int_token::<i8>(pair[1], "glonass_slot.channel", line)?;
            // A header block gives its records no order, so a slot given two
            // channels in one block says two things at once.
            if let Some(held) = self.glonass_slots_in_block.get(&sat.prn) {
                if *held != channel {
                    return Err(Error::Parse(format!(
                        "RINEX OBS GLONASS SLOT / FRQ # records in one header block contradict: R{:02} is given channels {held} and {channel}",
                        sat.prn
                    )));
                }
            }
            self.glonass_slots_in_block.insert(sat.prn, channel);
            self.glonass_slots.insert(sat.prn, channel);
        }
        Ok(())
    }

    fn parse_glonass_cod_phs_bis(&mut self, line: &str) -> Result<()> {
        let content = field(line, 0, 60);
        // `4(1X,A3,1X,F8.3)`: read in its columns where the record is laid out
        // in them, which is the only way to read a blank bias beside others. A
        // record not in its columns is read by its fields, a code and a bias
        // each.
        let fields = match glonass_bias_columns(content) {
            Some(fields) => fields,
            None => {
                let tokens: Vec<&str> = content.split_whitespace().collect();
                if !tokens.len().is_multiple_of(2) {
                    return Err(Error::Parse(format!(
                        "RINEX OBS GLONASS COD/PHS/BIS has an odd token count in {line:?}"
                    )));
                }
                tokens.chunks(2).map(|pair| (pair[0], pair[1])).collect()
            }
        };
        let mut entries = Vec::new();
        for (code, value) in fields {
            // The code is `A3`: a longer one would be cut when written, and read
            // back as another code.
            if code.len() > GLONASS_BIAS_CODE_WIDTH {
                return Err(Error::Parse(format!(
                    "RINEX OBS GLONASS COD/PHS/BIS code {code:?} exceeds the A3 field it is written in, in {line:?}"
                )));
            }
            let bias = if value.is_empty() {
                None
            } else {
                let bias = strict_f64_token(value, "glonass_code_phase_bias", line)?;
                Some(exact_in_field(
                    bias,
                    GLONASS_BIAS_WIDTH,
                    GLONASS_BIAS_DECIMALS,
                    "glonass_code_phase_bias",
                    line,
                )?)
            };
            entries.push((code.to_string(), bias));
        }
        // RINEX gives this record four biases, which fit one line. More than
        // that is an extension of this crate's own: the writer continues them
        // onto another line rather than cutting them off at the sixtieth
        // column, and a further line adds to the record. A blank record means
        // "unknown" and so clears what came before it.
        //
        // A blank record beside records giving biases in the same header block
        // cannot be held beside them, and is refused rather than lost.
        let (held, blank, reported) = &mut self.glonass_biases_in_block;
        if entries.is_empty() && !held.is_empty() || !entries.is_empty() && *blank {
            return Err(Error::Parse(format!(
                "RINEX OBS GLONASS COD/PHS/BIS records in one header block contradict: a blank record declares the biases unknown beside records giving them, in {line:?}"
            )));
        }
        *blank |= entries.is_empty();
        // A code given two biases in one block, which gives its records no
        // order, is kept as written and read as ambiguous; each is counted.
        // From RINEX 4.00 the record is ignored and not checked.
        let deprecated = records_deprecated_in_rinex4(self.version.unwrap_or_default());
        let mut contradicted = Vec::new();
        for (code, bias) in &entries {
            match held.get(code) {
                Some(earlier) if earlier != bias => {
                    if !deprecated && reported.insert(code.clone()) {
                        contradicted.push(code.clone());
                    }
                }
                Some(_) => {}
                None => {
                    held.insert(code.clone(), *bias);
                }
            }
        }
        for code in contradicted {
            self.diagnostics.push_skip(Skip {
                at: RecordRef::default().with_satellite(code),
                reason: SkipReason::InconsistentRecord(
                    "GLONASS COD/PHS/BIS records in one header block give a code different biases",
                ),
            });
        }
        match &mut self.glonass_cod_phs_bis {
            Some(existing) if !entries.is_empty() => existing.extend(entries),
            slot => *slot = Some(entries),
        }
        Ok(())
    }

    fn parse_leap_seconds(&mut self, line: &str) -> Result<()> {
        let current = strict_int_field::<i64>(line, 0, 6, "leap_seconds.current")?;
        let delta_future = optional_i64_field(line, 6, 12, "leap_seconds.delta_future")?;
        let week = optional_i64_field(line, 12, 18, "leap_seconds.week")?;
        let day = optional_i64_field(line, 18, 24, "leap_seconds.day")?;
        let raw_token = field(line, 24, 27).trim();
        let time_system = if raw_token.is_empty() {
            None
        } else {
            let version = self.version.unwrap_or(0.0);
            match check_leap_seconds_time_system(raw_token, version) {
                LeapTimeSystemValidity::Valid => Some(raw_token.to_string()),
                LeapTimeSystemValidity::UnknownToken => {
                    return Err(Error::Parse(format!(
                        "RINEX OBS unknown leap_seconds.time_system {raw_token:?} in {line:?}"
                    )));
                }
                LeapTimeSystemValidity::UnsupportedInVersion => {
                    return Err(Error::Parse(format!(
                        "RINEX OBS leap_seconds.time_system {raw_token:?} not supported in version {version:.2} in {line:?}"
                    )));
                }
            }
        };
        self.leap_seconds = Some(ObsLeapSeconds {
            current,
            delta_future,
            week,
            day,
            time_system,
        });
        Ok(())
    }

    /// Refuse a `PRN / # OF OBS` record whose count fields are not counts.
    /// Which fields are read depends on the lists the header ends with, so all
    /// nine are checked: `9I6` holds counts or blanks.
    fn check_prn_obs_count_fields(&self, line: &str) -> Result<()> {
        let mut start = PRN_OBS_COUNTS_COLUMN;
        while start + PRN_OBS_COUNT_WIDTH <= write::HEADER_CONTENT_WIDTH {
            let raw = field(line, start, start + PRN_OBS_COUNT_WIDTH).trim();
            if !raw.is_empty() {
                strict_int_token::<usize>(raw, "prn_obs_count", line)?;
            }
            start += PRN_OBS_COUNT_WIDTH;
        }
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

            if is_event_flag(flag) {
                // Event record: the next `numsat` lines are header or comment
                // records, not observations. They are kept as they were written
                // so the epoch can be written back whole.
                let special_records = take_special_records(lines, numsat)?;
                if applies_header_records(flag) {
                    self.apply_event(&special_records)?;
                }
                self.push_epoch(ObsEpoch {
                    epoch: epoch_time,
                    flag,
                    rcv_clock_offset_s,
                    epoch_picoseconds,
                    declared_record_count: numsat,
                    special_records,
                    sats: BTreeMap::new(),
                    cycle_slips: BTreeMap::new(),
                });
                continue;
            }

            // Cycle slip records are read exactly as observation records are,
            // and kept apart from them.
            let mut sats = BTreeMap::new();
            for _ in 0..numsat {
                let sat_line = lines.next().ok_or_else(|| {
                    Error::Parse("RINEX OBS epoch truncated: missing satellite line".into())
                })?;
                let sat_line = sat_line.trim_end_matches(['\r', '\n']);
                // Resolve the satellite token first: a token that does not parse
                // to a representable `GnssSatelliteId` (`R00`, which names no
                // satellite) is an independent record that must not reject
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
                    // Lexically a satellite designator but the PRN is not
                    // representable (`R00`): skip the whole
                    // record - including wrapped continuation lines - and count it.
                    // No observation values are fabricated.
                    self.push_unrepresentable_satellite_skip(field(&normalized, 0, 3));
                    consume_skipped_sat_continuations(lines);
                    continue;
                }
                // A version 2 file's observation records are read by its own
                // type list, epoch records in version 3 layout included.
                if self.is_rinex2() {
                    // With no list there is nothing to read them by, as for
                    // records in version 2 layout.
                    self.rinex2_obs_lines_per_sat()?;
                    if let Some(sat) = parse_sv_token(field(&normalized, 0, 3)) {
                        self.ensure_rinex2_system_obs_codes(sat.system);
                    }
                }
                let sat_record = self.collect_sat_record(sat_line, lines)?;
                let (sat, values) = self.parse_sat_line(&sat_record)?;
                sats.insert(sat, values);
            }
            let (sats, cycle_slips) = split_cycle_slips(flag, sats);
            self.push_epoch(ObsEpoch {
                epoch: epoch_time,
                flag,
                rcv_clock_offset_s,
                epoch_picoseconds,
                declared_record_count: numsat,
                special_records: Vec::new(),
                sats,
                cycle_slips,
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

            if is_event_flag(flag) {
                let special_records = take_special_records(lines, numsat)?;
                if applies_header_records(flag) {
                    self.apply_event(&special_records)?;
                }
                self.push_epoch(ObsEpoch {
                    epoch: epoch_time,
                    flag,
                    rcv_clock_offset_s,
                    epoch_picoseconds: None,
                    declared_record_count: numsat,
                    special_records,
                    sats: BTreeMap::new(),
                    cycle_slips: BTreeMap::new(),
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
            let (sats, cycle_slips) = split_cycle_slips(flag, sats);
            self.push_epoch(ObsEpoch {
                epoch: epoch_time,
                flag,
                rcv_clock_offset_s,
                epoch_picoseconds: None,
                declared_record_count: numsat,
                special_records: Vec::new(),
                sats,
                cycle_slips,
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
        self.rinex2_systems.insert(system);
        let version = self.version.unwrap_or(2.11);
        self.obs_codes
            .entry(system)
            .or_insert_with(|| rinex2_system_obs_codes(system, &self.rinex2_obs_codes, version));
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
                // As in version 3: a value wider than the field or carrying more
                // than three decimals comes back as a different number.
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
            // resolve to a representable id (`R00`) is still a new satellite
            // record, not continuation data. Only a
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

    fn finish(mut self) -> Result<RinexObs> {
        if let Some(remaining) = self.glonass_slots_remaining {
            if remaining != 0 {
                return Err(Error::Parse(format!(
                    "RINEX OBS GLONASS slot table missing {remaining} declared entries"
                )));
            }
        }
        let Some(mut header) = self.file_header.take() else {
            return Err(Error::Parse(
                "RINEX OBS missing RINEX VERSION / TYPE".into(),
            ));
        };
        let version = header.version;
        let rinex2 = version.floor() as i64 == 2;
        // Each constellation's codes are the union of every list the file
        // declares for it, the file header's first.
        let mut union: BTreeMap<GnssSystem, Vec<String>> = BTreeMap::new();
        if rinex2 {
            // A version 2 file names its codes once for every constellation,
            // and states a list for each constellation its records name, or
            // with none, for the one its version record names.
            let mut systems = self.rinex2_systems.clone();
            if systems.is_empty()
                && self
                    .code_list_segments
                    .iter()
                    .any(|segment| !segment.names.is_empty())
            {
                systems.insert(self.rinex2_default_system.unwrap_or(GnssSystem::Gps));
            }
            for system in systems {
                let list = union.entry(system).or_default();
                for segment in &self.code_list_segments {
                    extend_code_union(
                        list,
                        &rinex2_system_obs_codes(system, &segment.names, version),
                    );
                }
            }
        } else {
            for segment in &self.code_list_segments {
                for (system, list) in &segment.lists {
                    extend_code_union(union.entry(*system).or_default(), list);
                }
            }
            // A header declaring its types only as version 2 names states the
            // list those names read as, whatever its version.
            if union.is_empty() && !header_names_fallback_is_empty(&self.code_list_segments) {
                let system = self.rinex2_default_system.unwrap_or(GnssSystem::Gps);
                let names = self
                    .code_list_segments
                    .first()
                    .map(|segment| segment.names.as_slice())
                    .unwrap_or_default();
                union.insert(system, rinex2_system_obs_codes(system, names, version));
            }
        }
        if union.is_empty() {
            return Err(Error::Parse(
                "RINEX OBS header has no SYS / # / OBS TYPES records".into(),
            ));
        }
        header.declared_obs_codes = if rinex2 {
            union
                .keys()
                .map(|system| {
                    (
                        *system,
                        rinex2_system_obs_codes(*system, &header.rinex2_types, version),
                    )
                })
                .collect()
        } else if self
            .code_list_segments
            .first()
            .is_some_and(|segment| segment.lists.is_empty())
        {
            // The list the file header's names read as is the one it declares.
            union.clone()
        } else {
            self.code_list_segments
                .first()
                .map(|segment| segment.lists.clone())
                .unwrap_or_default()
        };
        // Values were read by the list in effect at their epoch, and are held
        // aligned to the union.
        if self.code_list_segments.len() > 1 {
            let mut positions: BTreeMap<(usize, GnssSystem), Option<Vec<Option<usize>>>> =
                BTreeMap::new();
            let segments = &self.code_list_segments;
            for (epoch_index, (epoch, segment)) in
                self.epochs.iter_mut().zip(&self.epoch_segments).enumerate()
            {
                for (sat, values) in epoch.sats.iter_mut().chain(epoch.cycle_slips.iter_mut()) {
                    let Some(held) = union.get(&sat.system) else {
                        continue;
                    };
                    let map = positions.entry((*segment, sat.system)).or_insert_with(|| {
                        let list = match segments.get(*segment) {
                            Some(segment) if rinex2 => {
                                rinex2_system_obs_codes(sat.system, &segment.names, version)
                            }
                            Some(segment) => {
                                segment.lists.get(&sat.system).cloned().unwrap_or_default()
                            }
                            None => Vec::new(),
                        };
                        (list != *held).then(|| union_positions(&list, held))
                    });
                    if let Some(map) = map {
                        *values = values_in_union(values, map, held.len()).ok_or_else(|| {
                            Error::Parse(format!(
                                "RINEX OBS epoch {epoch_index} {sat} holds values under codes \
                                 the union of its code lists does not hold"
                            ))
                        })?;
                    }
                }
            }
        }
        header.obs_codes = union;
        Ok(RinexObs {
            header,
            epochs: self.epochs,
            skipped_records: self.diagnostics.skips.len(),
        })
    }

    /// Refuse a record of a label holding one value whose parsed value differs
    /// from the value the same label's record before it set in this header
    /// block. A block gives its records no order, so neither could be the one
    /// in effect. Records whose text differs but which read as one value are
    /// one value. A blank `MARKER NAME` or `SIGNAL STRENGTH UNIT` record sets no
    /// value, and is not compared.
    fn check_single_value(&mut self, label: &str, line: &str) -> Result<()> {
        if !SINGLE_VALUE_LABELS.contains(&label) {
            return Ok(());
        }
        let value = match label {
            "RINEX VERSION / TYPE" => {
                SingleValue::Version(self.version, self.rinex2_default_system)
            }
            "INTERVAL" => SingleValue::Number(self.interval_s),
            "MARKER NAME" if !field(line, 0, 60).trim().is_empty() => {
                SingleValue::Text(self.marker_name.clone())
            }
            "MARKER NUMBER" => SingleValue::Text(self.marker_number.clone()),
            "MARKER TYPE" => SingleValue::Text(self.marker_type.clone()),
            "APPROX POSITION XYZ" => SingleValue::Vector(self.approx_position_m),
            "ANTENNA: DELTA H/E/N" => SingleValue::Vector(self.antenna_delta_hen_m),
            "ANT # / TYPE" => SingleValue::Antenna(self.antenna.clone()),
            "REC # / TYPE / VERS" => SingleValue::Receiver(self.receiver.clone()),
            "OBSERVER / AGENCY" => {
                SingleValue::ObserverAgency(self.observer.clone(), self.agency.clone())
            }
            "SIGNAL STRENGTH UNIT" if !field(line, 0, 20).trim().is_empty() => {
                SingleValue::Text(self.signal_strength_unit.clone())
            }
            "TIME OF FIRST OBS" => SingleValue::Time(self.time_of_first_obs),
            "TIME OF LAST OBS" => SingleValue::Time(self.time_of_last_obs),
            "LEAP SECONDS" => SingleValue::LeapSeconds(self.leap_seconds.clone()),
            "# OF SATELLITES" => SingleValue::Count(self.n_satellites),
            _ => return Ok(()),
        };
        match self.single_values_in_block.get(label) {
            Some(held) if *held != value => Err(Error::Parse(format!(
                "RINEX OBS {label} records in one header block contradict: {held:?} and {value:?}, in {line:?}"
            ))),
            Some(_) => {
                self.single_values_in_block.insert(label.to_string(), value);
                Ok(())
            }
            None => {
                self.single_values_in_block.insert(label.to_string(), value);
                Ok(())
            }
        }
    }

    fn scale_factor_for(&self, system: GnssSystem, code: &str) -> f64 {
        scale_factor_in(&self.scale_factors, system, code)
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
/// returning the civil time, event flag, and satellite count. The time is
/// `None` for an event whose epoch fields are blank.
type ParsedEpochLine = (Option<ObsEpochTime>, u8, usize, Option<f64>, Option<u32>);

fn parse_epoch_line(
    line: &str,
    second_policy: validate::CivilSecondPolicy,
) -> Result<ParsedEpochLine> {
    // Trailing whitespace is no field; left on, a tab after the last one takes
    // the line out of its layout and into a reading that places fields
    // differently.
    let line = line.trim_end_matches([' ', '\t']);
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
    let tokens = picoseconds_after_the_clock(body.split_whitespace().collect());
    // An event without a significant epoch leaves the six epoch fields blank,
    // so only its flag, its count and an optional clock offset remain; a line
    // with an epoch has at least eight fields.
    if (2..=3).contains(&tokens.len()) {
        let flag = strict_int_token::<u8>(tokens[0], "epoch.flag", line)?;
        let count = parse_epoch_record_count(tokens[1], line)?;
        let clock = tokens
            .get(2)
            .map(|token| epoch_clock_offset(token, line))
            .transpose()?;
        return blank_epoch_line(flag, line).map(|()| (None, flag, count, clock, None));
    }
    match interpret_epoch_tokens(&tokens, line, second_policy) {
        Ok((parsed, _)) => Ok(parsed),
        // A line that is in neither the layout nor a shape the tokenizer can
        // read may still be one whose flag and count ran together. Separating
        // them is the last thing tried, so it cannot change any other reading.
        Err(error) => {
            let Some(split) = split_merged_epoch_flag_and_count(&tokens) else {
                return Err(error);
            };
            // A flag and count run together only in a line written to its
            // columns, where digits straight after the count sit in reserved
            // columns rather than where RINEX 4.02 puts picoseconds; a
            // column-exact 4.02 line has already been read by its layout.
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

/// Move RINEX 4.02 picoseconds from after the clock offset to the slot before
/// the flag, where the epoch reader takes them.
///
/// Read by whitespace, a line carrying them where 4.02 puts them has them as a
/// five-digit token after a clock offset written as one, with a decimal point
/// or an exponent. Only that shape is taken: a lone five-digit token after the
/// count may as well be a clock offset written as an integer, which is how this
/// reader has always read it, and a flag that is not a single digit may be a
/// flag and count run together, whose reading this must not shift. A line whose
/// picoseconds already sit before the flag is left as it is.
fn picoseconds_after_the_clock(mut tokens: Vec<&str>) -> Vec<&str> {
    let five_digits =
        |token: &str| token.len() == 5 && token.bytes().all(|byte| byte.is_ascii_digit());
    let written_as_clock = |token: &str| token.contains(['.', 'e', 'E', 'd', 'D']);
    // Time, flag and count come first; the clock offset and the picoseconds
    // follow the count.
    let flag = EPOCH_TIME_TOKENS;
    let after_count = EPOCH_TIME_TOKENS + 2;
    let shaped = tokens.len() == after_count + 2
        && tokens[flag].len() == 1
        && tokens[flag].bytes().all(|byte| byte.is_ascii_digit())
        && written_as_clock(tokens[after_count])
        && five_digits(tokens[after_count + 1]);
    if shaped {
        if let Some(picoseconds) = tokens.pop() {
            tokens.insert(EPOCH_TIME_TOKENS, picoseconds);
        }
    }
    tokens
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
/// Returns `None` when the line is not laid out that way. Picoseconds are taken
/// from after the clock, where RINEX 4.02 puts them, or from after the seconds,
/// where this writer used to.
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
    if let Some([marker, year, month, day, hour, minute, second, flag, count, clock, picoseconds]) =
        fixed_record(line, V4_EPOCH_COLUMNS)
    {
        // Only five digits there are picoseconds; anything else in those
        // columns leaves the line to the readings below.
        if marker == ">"
            && picoseconds.len() == 5
            && picoseconds.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Some(epoch_tokens(
                [year, month, day, hour, minute, second],
                picoseconds,
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
    // A column read hands over an event's blank epoch fields as six empty ones.
    let epoch = if time.iter().all(|field| field.is_empty()) {
        None
    } else {
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
        Some(epoch)
    };

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
    if epoch.is_none() {
        blank_epoch_line(flag, line)?;
        // Picoseconds extend the epoch's second, and a blank epoch has none.
        if epoch_picoseconds.is_some() {
            return Err(Error::Parse(format!(
                "RINEX OBS epoch record carries picoseconds with blank epoch fields in {line:?}"
            )));
        }
    }
    let rcv_clock_offset_s = tokens
        .get(index)
        .map(|token| epoch_clock_offset(token, line))
        .transpose()?;
    let consumed = index + usize::from(rcv_clock_offset_s.is_some());
    Ok((
        (epoch, flag, numsat, rcv_clock_offset_s, epoch_picoseconds),
        consumed,
    ))
}

/// Read an epoch record's receiver clock offset, held to its `F15.12` field.
fn epoch_clock_offset(token: &str, line: &str) -> Result<f64> {
    let offset = strict_f64_token(token, "epoch.rcv_clock_offset_s", line)?;
    exact_in_field(
        offset,
        CLOCK_OFFSET_WIDTH,
        CLOCK_OFFSET_DECIMALS,
        "epoch.rcv_clock_offset_s",
        line,
    )
}

/// Refuse blank epoch fields on a record that is not an event. RINEX lets an
/// event without a significant epoch leave them blank; observation and cycle
/// slip records are tagged with the time they were taken at.
fn blank_epoch_line(flag: u8, line: &str) -> Result<()> {
    if is_event_flag(flag) {
        return Ok(());
    }
    Err(Error::Parse(format!(
        "RINEX OBS epoch record with flag {flag} leaves its epoch fields blank, which only an \
         event may, in {line:?}"
    )))
}

type ParsedEpochLineV2 = (Option<ObsEpochTime>, u8, usize, Option<f64>);

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
    // An event without a significant epoch leaves the six epoch fields blank,
    // and names no satellites, so its flag and count are all the window holds.
    // A count of 100 or more fills its `I3` field and abuts the flag, so the
    // two are read from their columns when the epoch columns are blank.
    if field(head, 0, V2_EPOCH_HEAD_COLUMNS[6].0).trim().is_empty() {
        let (flag_start, flag_end) = V2_EPOCH_HEAD_COLUMNS[6];
        let (count_start, count_end) = V2_EPOCH_HEAD_COLUMNS[7];
        let flag_field = field(head, flag_start, flag_end).trim();
        let count_field = field(head, count_start, count_end).trim();
        if !flag_field.is_empty() && !count_field.is_empty() {
            let flag = strict_int_token::<u8>(flag_field, "epoch.flag", line)?;
            let numsat = parse_epoch_record_count(count_field, line)?;
            blank_epoch_line(flag, line)?;
            return Ok((None, flag, numsat, rinex2_epoch_clock_offset(line)?));
        }
    }
    let whitespace: Vec<&str> = head.split_whitespace().collect();
    if whitespace.len() == 2 {
        let flag = strict_int_token::<u8>(whitespace[0], "epoch.flag", line)?;
        let numsat = parse_epoch_record_count(whitespace[1], line)?;
        blank_epoch_line(flag, line)?;
        return Ok((None, flag, numsat, rinex2_epoch_clock_offset(line)?));
    }
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
    Ok((
        Some(ObsEpochTime {
            year,
            month: civil.month as u8,
            day: civil.day as u8,
            hour: civil.hour as u8,
            minute: civil.minute as u8,
            second: civil.second,
        }),
        flag,
        numsat,
        rinex2_epoch_clock_offset(line)?,
    ))
}

/// A version 2 epoch record's receiver clock offset, from column 69, or `None`
/// when those columns are blank.
fn rinex2_epoch_clock_offset(line: &str) -> Result<Option<f64>> {
    let clock = field(line, 68, line.len()).trim();
    if clock.is_empty() {
        return Ok(None);
    }
    epoch_clock_offset(clock, line).map(Some)
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

/// The records of an epoch read in the observation record layout, as the
/// observations and cycle slips they are: a flag 6 epoch's records are slips,
/// and every other epoch read this way holds observations.
type SatRecords = BTreeMap<GnssSatelliteId, Vec<ObsValue>>;

fn split_cycle_slips(flag: u8, records: SatRecords) -> (SatRecords, SatRecords) {
    if flag == CYCLE_SLIP_FLAG {
        (BTreeMap::new(), records)
    } else {
        (records, BTreeMap::new())
    }
}

/// The codes one constellation reads from a version 2 file's single list.
///
/// Version 2 names its codes once for every constellation at once, and a
/// constellation has no observable under some of them: Galileo has no `P1`,
/// BeiDou no `P2`. Such a name stays as the file wrote it rather than being read
/// as a signal the constellation does have. Reading Galileo's `P1` as its `C1X`
/// put one code at two positions, and a writer then had to guess which held the
/// measurement. The same holds for a second name of a code already held -
/// BeiDou's `C1` and `C2` are both B1I - so one code sits at one position
/// whatever the file called it elsewhere.
///
/// Every path that builds a constellation's list goes through here, including
/// a header with no observations to read, so the rule cannot differ between
/// them.
fn rinex2_system_obs_codes(system: GnssSystem, names: &[String], version: f64) -> Vec<String> {
    let mut held: Vec<String> = Vec::with_capacity(names.len());
    let mut given: std::collections::HashSet<String> =
        std::collections::HashSet::with_capacity(names.len());
    for name in names {
        let code = rinex2_next_obs_code(system, name, version, &given);
        given.insert(code.clone());
        held.push(code);
    }
    held
}

/// Whether a code is a version 2 name kept as written rather than a code a
/// name reads as: one or two bytes, what a version 2 `A2` type field holds.
/// Every code a name reads as is three characters.
pub(crate) fn rinex2_kept_as_written(code: &str) -> bool {
    matches!(code.len(), 1 | 2)
}

/// What a constellation reads the next version 2 name as, given the codes the
/// names before it gave: the canonical code the first time a name it carries
/// reads as that code, the name as written otherwise. The one rule
/// [`rinex2_system_obs_codes`] applies name by name.
pub(crate) fn rinex2_next_obs_code(
    system: GnssSystem,
    name: &str,
    version: f64,
    given: &std::collections::HashSet<String>,
) -> String {
    let canonical = canonical_rinex2_obs_code(system, name, version);
    if rinex2_name_allowed(system, name, version) && !given.contains(&canonical) {
        canonical
    } else {
        name.to_string()
    }
}

/// Read a phase shift record's satellite tokens.
/// A phase-shift record's satellite tokens: the satellites [`GnssSatelliteId`]
/// holds, and the well-formed designators it does not hold, such as `R00`, as
/// written. A token that is not a satellite designator is refused.
fn phase_shift_satellites(
    tokens: &[&str],
    line: &str,
) -> Result<(Vec<GnssSatelliteId>, Vec<String>)> {
    let mut satellites = Vec::new();
    let mut unrepresentable = Vec::new();
    for token in tokens {
        if let Some(sat) = parse_sv_token(token) {
            satellites.push(sat);
        } else if is_satellite_designator(token) {
            unrepresentable.push((*token).to_string());
        } else {
            return Err(Error::Parse(format!(
                "RINEX OBS phase-shift satellite token {token:?} unparsable in {line:?}"
            )));
        }
    }
    Ok((satellites, unrepresentable))
}

/// Whether a token is a RINEX satellite designator as [`GnssSatelliteId`] reads
/// one, a constellation letter and a one- or two-digit number, whatever the
/// number is.
fn is_satellite_designator(token: &str) -> bool {
    let mut chars = token.chars();
    chars.next().and_then(GnssSystem::from_letter).is_some()
        && (1..=2).contains(&chars.as_str().len())
        && chars.as_str().bytes().all(|byte| byte.is_ascii_digit())
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
pub(crate) fn rinex2_name_allowed(system: GnssSystem, name: &str, version: f64) -> bool {
    let mut chars = name.chars();
    let (Some(kind), Some(band)) = (chars.next(), chars.next()) else {
        return false;
    };
    // Version 2's observation kinds are pseudorange, P code, phase, Doppler and
    // signal strength. Any other letter is not a name, whatever band follows.
    if !matches!(kind, 'C' | 'P' | 'L' | 'D' | 'S') {
        return false;
    }
    if kind == 'P' && !matches!(system, GnssSystem::Gps | GnssSystem::Glonass) {
        return false;
    }
    // 2.12's letters: `A` is L1 C/A, `B` L1C, `C` L2C and `D` GLONASS G2 C/A,
    // each for the constellations that carry that signal, and never with `P`.
    // From 2.12 the civil signals have letters, and the digits they left
    // behind are no longer names: `C1` is refused outright, and `L1`, `D1`,
    // `S1` only name a signal where a constellation has a P code on L1.
    // BeiDou has no 2.12 lettered signals and continues to use digit band 1.
    if version >= RINEX2_LETTERED_NAMES_VERSION && system != GnssSystem::BeiDou {
        if kind == 'C' && band == '1' {
            return false;
        }
        if matches!(kind, 'L' | 'D' | 'S')
            && band == '1'
            && !matches!(system, GnssSystem::Gps | GnssSystem::Glonass)
        {
            return false;
        }
    }
    if band.is_ascii_alphabetic() {
        return kind != 'P'
            && match band {
                'A' => matches!(
                    system,
                    GnssSystem::Gps | GnssSystem::Glonass | GnssSystem::Qzss | GnssSystem::Sbas
                ),
                'B' | 'C' => matches!(system, GnssSystem::Gps | GnssSystem::Qzss),
                'D' => system == GnssSystem::Glonass,
                _ => false,
            };
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
        GnssSystem::Glonass => {
            if kind == 'P' {
                &['1', '2']
            } else {
                &['1', '2', '3']
            }
        }
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
            if rinex2_name_allowed(system, &name, version)
                && canonical_rinex2_obs_code(system, &name, version) == canonical
                && !names.contains(&name)
            {
                names.push(name);
            }
        }
    }
    // From 2.12 the civil signals have letters of their own, and a product
    // holding one of those signals is written back under its letter rather
    // than under a digit that names the P code.
    let letters: &[char] = if version >= RINEX2_LETTERED_NAMES_VERSION {
        &['A', 'B', 'C', 'D']
    } else {
        &[]
    };
    for kind in kinds {
        for band in BANDS.iter().chain(letters) {
            let name = format!("{kind}{band}");
            if rinex2_name_allowed(system, &name, version)
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

/// Whether an epoch flag's special records may be header records that take
/// effect for the epochs after them. RINEX 2.10 allowed header records after
/// every event flag, which are 2 to 5; flag 6 is followed by cycle slip records,
/// and no flag above it is defined.
pub(crate) fn applies_header_records(flag: u8) -> bool {
    (2..=5).contains(&flag)
}

/// The header records an event's special records lay over the header in effect:
/// every record that changes how later epochs read or what they mean. The type
/// lists are each read at their own version only.
const EVENT_HEADER_LABELS: &[&str] = &[
    "SYS / SCALE FACTOR",
    "SYS / PHASE SHIFT",
    "GLONASS SLOT / FRQ #",
    "GLONASS COD/PHS/BIS",
    "INTERVAL",
    "MARKER NAME",
    "MARKER NUMBER",
    "MARKER TYPE",
    "APPROX POSITION XYZ",
    "ANTENNA: DELTA H/E/N",
    "ANT # / TYPE",
    "REC # / TYPE / VERS",
    "OBSERVER / AGENCY",
    "SIGNAL STRENGTH UNIT",
];

/// A special record as the header reader reads a header record: printable
/// ASCII, with a label out of its columns moved into them.
fn event_record_line(record: &str) -> String {
    let ascii = printable_ascii_header_columns(record.trim_end_matches(['\r', '\n']));
    normalize_header_line(&ascii).into_owned()
}

fn event_label_takes_effect(version: f64, label: &str) -> bool {
    let rinex2 = version.floor() as i64 == 2;
    match label {
        "SYS / # / OBS TYPES" => !rinex2,
        "# / TYPES OF OBSERV" => rinex2,
        _ => EVENT_HEADER_LABELS.contains(&label),
    }
}

/// The label a special record carries, read as the header reader reads a
/// header record's label.
pub(crate) fn event_record_label(record: &str) -> String {
    raw_field_from(&event_record_line(record), 60)
        .trim()
        .to_string()
}

/// Whether a special record following an event takes effect for the epochs
/// after it in a file of this version. Every other record - a comment, a record
/// no later epoch depends on - is kept as written and changes nothing.
pub(crate) fn event_record_takes_effect(version: f64, record: &str) -> bool {
    event_label_takes_effect(version, &event_record_label(record))
}

/// Whether an event's special records include a record with this label that
/// takes effect in a file of this version.
pub(crate) fn event_declares_label(version: f64, records: &[String], label: &str) -> bool {
    records.iter().any(|record| {
        let held = event_record_label(record);
        held == label && event_label_takes_effect(version, &held)
    })
}

/// What laying an event's records over a header changed.
#[derive(Debug, Default)]
pub(crate) struct EventEffect {
    /// Whether any record took effect.
    pub(crate) applied: bool,
    /// Whether the records declared a code list, or at version 2 a list of
    /// observation types.
    pub(crate) lists: bool,
    /// Whether the records declared a scale factor.
    pub(crate) scale_factors: bool,
    /// Records skipped as the file header skips them: a GLONASS slot the
    /// engine cannot represent.
    pub(crate) skips: Vec<Skip>,
}

/// Lay the header records an event epoch carries over the header in effect
/// before it, reading each as the file header reads it and refusing a malformed
/// one as the file header refuses it.
///
/// RINEX says "Each value remains valid until changed by an additional header
/// record" (3.05 and 4.02 section 6.5) and "allows the free ordering of the
/// header records" (section 5.2.1). It does not say how records overlapping
/// within one header block, or an event's records overlapping the header's,
/// combine. What follows is this reader's policy, chosen to be consistent with
/// those two sentences: no record's position within a block decides anything,
/// and a later block changes what it gives values to.
///
/// An event's records are one header block, laid over the header in effect.
/// Within a block, a block giving a key that changes how observations are read
/// two values is refused as contradictory: a scale factor, a GLONASS slot, or a
/// record holding one value. Phase shifts and GLONASS biases change nothing an
/// observation reads as, so a block giving a satellite's code two corrections,
/// or a GLONASS code two biases, keeps both records as written, counts the
/// contradiction as a skip, and that signal reads as
/// [`CorrectionUnavailable::Ambiguous`]. A record naming satellites or codes
/// applies to them over the block's record for every satellite or code, and a
/// phase shift record for a code over one naming only the constellation.
///
/// Across blocks: a value replaces the one in effect. A constellation's type
/// declaration replaces its list; a repeated complete declaration for the same
/// constellation within the block adds its codes to the list, as the file
/// header reads one. RINEX gives each constellation one declaration ("In mixed
/// files: Repeat for each satellite system"), so adding a repeated one is a
/// policy too. A phase shift record naming only a constellation replaces every
/// earlier record for it; one for every satellite of a code replaces the
/// default and every earlier satellite exception for the code, and one naming
/// satellites replaces those satellites only; a scale factor for every code of
/// a system and one naming codes do the same. A GLONASS slot's channel and a
/// code's GLONASS bias replace that slot's and that code's. `obs_codes`, the
/// union of every list the file declares, is not changed.
pub(crate) fn apply_event_records(
    header: &mut ObsHeader,
    records: &[String],
) -> Result<EventEffect> {
    let mut effect = EventEffect::default();
    let version = header.version;
    let lines: Vec<String> = records
        .iter()
        .map(|record| event_record_line(record))
        .filter(|line| event_label_takes_effect(version, raw_field_from(line, 60).trim()))
        .collect();
    if lines.is_empty() {
        return Ok(effect);
    }
    let mut scratch = Parser::new();
    scratch.version = Some(version);
    scratch.approx_position_m = header.approx_position_m;
    scratch.antenna_delta_hen_m = header.antenna_delta_hen_m;
    scratch.interval_s = header.interval_s;
    scratch.marker_name = header.marker_name.clone();
    scratch.marker_number = header.marker_number.clone();
    scratch.marker_type = header.marker_type.clone();
    scratch.observer = header.observer.clone();
    scratch.agency = header.agency.clone();
    scratch.receiver = header.receiver.clone();
    scratch.antenna = header.antenna.clone();
    scratch.signal_strength_unit = header.signal_strength_unit.clone();
    for line in &lines {
        let label = raw_field_from(line, 60).trim();
        scratch.parse_layered_record(label, line)?;
        scratch.check_single_value(label, line)?;
    }
    let end = "the end of the event's records";
    scratch.ensure_obs_type_count_complete(end)?;
    scratch.ensure_obs_type_count_complete_v2(end)?;
    scratch.ensure_scale_factor_count_complete(end)?;
    scratch.ensure_phase_shift_count_complete(end)?;
    for skip in phase_shift_contradictions(version, &scratch.phase_shifts) {
        scratch.diagnostics.push_skip(skip);
    }
    check_block_scale_factors(&scratch.scale_factors)?;
    if let Some(remaining) = scratch.glonass_slots_remaining.filter(|left| *left != 0) {
        return Err(Error::Parse(format!(
            "RINEX OBS GLONASS slot table missing {remaining} declared entries"
        )));
    }
    effect.applied = true;
    header.approx_position_m = scratch.approx_position_m;
    header.antenna_delta_hen_m = scratch.antenna_delta_hen_m;
    header.interval_s = scratch.interval_s;
    header.marker_name = scratch.marker_name;
    header.marker_number = scratch.marker_number;
    header.marker_type = scratch.marker_type;
    header.observer = scratch.observer;
    header.agency = scratch.agency;
    header.receiver = scratch.receiver;
    header.antenna = scratch.antenna;
    header.signal_strength_unit = scratch.signal_strength_unit;
    if version.floor() as i64 == 2 {
        if scratch.rinex2_obs_types_declared {
            header.rinex2_types = scratch.rinex2_obs_codes;
            header.declared_obs_codes = header
                .obs_codes
                .keys()
                .map(|system| {
                    (
                        *system,
                        rinex2_system_obs_codes(*system, &header.rinex2_types, version),
                    )
                })
                .collect();
            effect.lists = true;
        }
    } else if !scratch.obs_codes.is_empty() {
        header.declared_obs_codes.extend(scratch.obs_codes);
        effect.lists = true;
    }
    // The event is a block of its own, laid over the blocks before it.
    effect.scale_factors = !scratch.scale_factors.is_empty();
    lay_scale_factors_over(&mut header.scale_factors, scratch.scale_factors);
    lay_phase_shifts_over(&mut header.phase_shifts, scratch.phase_shifts);
    header.glonass_slots.extend(scratch.glonass_slots);
    if let Some(entries) = scratch.glonass_cod_phs_bis {
        lay_glonass_biases_over(&mut header.glonass_cod_phs_bis, entries);
    }
    effect.skips = scratch.diagnostics.skips;
    Ok(effect)
}

/// The value a record of a label holding one value sets, compared as parsed:
/// numbers by value, so `-0.0` and `0.0` are one value, and text as trimmed.
#[derive(Debug, Clone)]
enum SingleValue {
    Version(Option<f64>, Option<GnssSystem>),
    Number(Option<f64>),
    Text(Option<String>),
    Vector(Option<[f64; 3]>),
    Antenna(Option<AntennaInfo>),
    Receiver(Option<ReceiverInfo>),
    ObserverAgency(Option<String>, Option<String>),
    Time(Option<(ObsEpochTime, TimeScale)>),
    LeapSeconds(Option<ObsLeapSeconds>),
    Count(Option<usize>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffectiveLeapTimeSystem {
    Gps,
    BeiDou,
}

impl EffectiveLeapTimeSystem {
    fn from_token(token: Option<&str>) -> Option<Self> {
        match token {
            None | Some("GPS") => Some(Self::Gps),
            Some("BDS" | "BDT") => Some(Self::BeiDou),
            _ => None,
        }
    }
}

fn leap_seconds_semantically_equal(a: &Option<ObsLeapSeconds>, b: &Option<ObsLeapSeconds>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.current == b.current
                && a.delta_future == b.delta_future
                && a.week == b.week
                && a.day == b.day
                && EffectiveLeapTimeSystem::from_token(a.time_system.as_deref())
                    == EffectiveLeapTimeSystem::from_token(b.time_system.as_deref())
        }
        _ => false,
    }
}

impl PartialEq for SingleValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Version(a1, a2), Self::Version(b1, b2)) => a1 == b1 && a2 == b2,
            (Self::Number(a), Self::Number(b)) => a == b,
            (Self::Text(a), Self::Text(b)) => a == b,
            (Self::Vector(a), Self::Vector(b)) => a == b,
            (Self::Antenna(a), Self::Antenna(b)) => a == b,
            (Self::Receiver(a), Self::Receiver(b)) => a == b,
            (Self::ObserverAgency(a1, a2), Self::ObserverAgency(b1, b2)) => a1 == b1 && a2 == b2,
            (Self::Time(a), Self::Time(b)) => a == b,
            (Self::LeapSeconds(a), Self::LeapSeconds(b)) => leap_seconds_semantically_equal(a, b),
            (Self::Count(a), Self::Count(b)) => a == b,
            _ => false,
        }
    }
}

/// Labels whose records each set one value the product applies, and which this
/// reader's policy refuses to see given two values in one header block, where
/// no record's position could decide between them. The first ten are read in the file
/// header and after an event; the rest only in the file header. A label whose
/// records add to a list, `COMMENT` or `PRN / # OF OBS`, or whose value nothing
/// applies, `PGM / RUN BY / DATE`, is not among them.
pub(crate) const SINGLE_VALUE_LABELS: [&str; 15] = [
    "INTERVAL",
    "MARKER NAME",
    "MARKER NUMBER",
    "MARKER TYPE",
    "APPROX POSITION XYZ",
    "ANTENNA: DELTA H/E/N",
    "ANT # / TYPE",
    "REC # / TYPE / VERS",
    "OBSERVER / AGENCY",
    "SIGNAL STRENGTH UNIT",
    "RINEX VERSION / TYPE",
    "TIME OF FIRST OBS",
    "TIME OF LAST OBS",
    "LEAP SECONDS",
    "# OF SATELLITES",
];

/// The contradictions among a header block's phase shifts: a satellite's code,
/// or every satellite's, given two corrections, a blank one being 0. RINEX gives
/// a block's records no order to choose one by. Phase shift records do not
/// change how observations are read, so the records are kept as written, the
/// satellite's code reads as [`CorrectionUnavailable::Ambiguous`], and each
/// contradiction is counted as a skip. From RINEX 4.00 the records are ignored
/// and not checked.
fn phase_shift_contradictions(version: f64, shifts: &[ObsPhaseShift]) -> Vec<Skip> {
    if records_deprecated_in_rinex4(version) {
        return Vec::new();
    }
    // A satellite is keyed by its id, and one the id does not hold by its
    // designator as written.
    type Key<'a> = (GnssSystem, Option<&'a str>, Option<String>);
    let mut held: BTreeMap<Key<'_>, f64> = BTreeMap::new();
    let mut reported = std::collections::BTreeSet::new();
    let mut skips = Vec::new();
    for shift in shifts {
        let covered: Vec<Option<String>> = if shift.covers_every_satellite() {
            vec![None]
        } else {
            shift
                .satellites
                .iter()
                .map(ToString::to_string)
                .chain(shift.unrepresentable_satellites.iter().cloned())
                .map(Some)
                .collect()
        };
        let correction = shift.correction_cycles.unwrap_or(0.0);
        for satellite in covered {
            let key = (shift.system, shift.code.as_deref(), satellite);
            match held.get(&key) {
                Some(earlier) if *earlier != correction => {
                    if reported.insert(key.clone()) {
                        skips.push(Skip {
                            at: RecordRef::default().with_satellite(
                                key.2
                                    .clone()
                                    .unwrap_or_else(|| shift.system.letter().to_string()),
                            ),
                            reason: SkipReason::InconsistentRecord(
                                "SYS / PHASE SHIFT records in one header block give one code different corrections",
                            ),
                        });
                    }
                }
                Some(_) => {}
                None => {
                    held.insert(key, correction);
                }
            }
        }
    }
    skips
}

/// Refuse a header block's scale factors giving one code, or every code of a
/// system, two factors.
fn check_block_scale_factors(factors: &[ObsScaleFactor]) -> Result<()> {
    let mut held: BTreeMap<(GnssSystem, Option<&str>), f64> = BTreeMap::new();
    for record in factors {
        let covered: Vec<Option<&str>> = if record.codes.is_empty() {
            vec![None]
        } else {
            record
                .codes
                .iter()
                .map(|code| Some(code.as_str()))
                .collect()
        };
        for code in covered {
            let key = (record.system, code);
            match held.get(&key) {
                Some(factor) if *factor != record.factor => {
                    return Err(Error::Parse(format!(
                        "RINEX OBS SYS / SCALE FACTOR records in one header block contradict: {} {} is given two factors",
                        record.system,
                        code.unwrap_or("every observation type")
                    )));
                }
                _ => {
                    held.insert(key, record.factor);
                }
            }
        }
    }
    Ok(())
}

/// Lay a block's phase shifts over the ones in effect. A shift for every
/// satellite of its code replaces every earlier shift for that code, and a
/// shift naming satellites takes those satellites out of the earlier shifts
/// naming them. The block's own shifts are kept as it gives them, after the
/// earlier ones, since which of them applies does not depend on their order.
fn lay_phase_shifts_over(held: &mut Vec<ObsPhaseShift>, block: Vec<ObsPhaseShift>) {
    for shift in &block {
        let same_code =
            |earlier: &ObsPhaseShift| earlier.system == shift.system && earlier.code == shift.code;
        if shift.code.is_none() {
            // A record naming only the constellation replaces every earlier
            // record for it.
            held.retain(|earlier| earlier.system != shift.system);
        } else if shift.covers_every_satellite() {
            held.retain(|earlier| !same_code(earlier));
        } else {
            held.retain_mut(|earlier| {
                if !same_code(earlier) || earlier.covers_every_satellite() {
                    return true;
                }
                earlier
                    .satellites
                    .retain(|satellite| !shift.satellites.contains(satellite));
                earlier
                    .unrepresentable_satellites
                    .retain(|token| !shift.unrepresentable_satellites.contains(token));
                !earlier.covers_every_satellite()
            });
        }
    }
    held.extend(block);
}

/// Lay a block's scale factors over the ones in effect, as phase shifts are
/// laid: a factor for every code of a system replaces every earlier factor for
/// the system, and one naming codes takes those codes out of the earlier ones
/// naming them.
fn lay_scale_factors_over(held: &mut Vec<ObsScaleFactor>, block: Vec<ObsScaleFactor>) {
    for record in &block {
        if record.codes.is_empty() {
            held.retain(|earlier| earlier.system != record.system);
        } else {
            held.retain_mut(|earlier| {
                if earlier.system != record.system || earlier.codes.is_empty() {
                    return true;
                }
                earlier.codes.retain(|code| !record.codes.contains(code));
                !earlier.codes.is_empty()
            });
        }
    }
    held.extend(block);
}

/// Lay a block's `GLONASS COD/PHS/BIS` entries over the ones in effect: each
/// code's bias replaces that code's, and a blank record, which says the biases
/// are unknown, replaces them all.
fn lay_glonass_biases_over(
    held: &mut Option<Vec<(String, Option<f64>)>>,
    block: Vec<(String, Option<f64>)>,
) {
    match held {
        Some(entries) if !block.is_empty() => {
            entries.retain(|(code, _)| !block.iter().any(|(replacing, _)| replacing == code));
            entries.extend(block);
        }
        slot => *slot = Some(block),
    }
}

/// Satellites the first `SYS / PHASE SHIFT` record of a list holds before the
/// list continues, `10(1X,A3)`.
const PHASE_SHIFT_SATELLITES_PER_RECORD: usize = 10;

/// A `SYS / PHASE SHIFT` record not laid out in its columns, read by its fields.
/// After the system and code, a number is either a correction followed by a
/// satellite count, or a satellite count after a blank correction, and the
/// fields alone do not say which. The reading taken is the one whose count
/// agrees with the satellite tokens after it: a blank or 0 count names none, a
/// count names that many, and a count past ten names at least the ten a first
/// record holds. Where both readings agree, which happens only for a number
/// with no satellites after it, a number with a decimal point would be a
/// correction, since no count has one; a whole number is refused as ambiguous.
/// Where neither agrees, the record is read as a correction and a count, and
/// the checks that reading meets refuse it.
fn loose_phase_shift_fields<'a>(content: &'a str, line: &str) -> Result<PhaseShiftFields<'a>> {
    let tokens: Vec<&'a str> = content.split_whitespace().collect();
    if tokens.len() < 2 {
        return Err(Error::Parse(format!(
            "RINEX OBS phase-shift header has too few fields in {line:?}"
        )));
    }
    let (system, code) = (tokens[0], tokens[1]);
    let rest = &tokens[2..];
    let fields = |correction: &'a str, count: &'a str, satellites: &[&'a str]| PhaseShiftFields {
        system,
        code,
        correction,
        count,
        satellites: satellites.to_vec(),
    };
    let Some(&number) = rest.first() else {
        return Ok(fields("", "", &[]));
    };
    let agrees = |count: &str, satellites: &[&str]| {
        let count = if count.is_empty() {
            0
        } else {
            match strict_int_token::<usize>(count, "phase_shift.satellite_count", line) {
                Ok(count) => count,
                Err(_) => return false,
            }
        };
        (satellites.len() == count
            || (satellites.len() >= PHASE_SHIFT_SATELLITES_PER_RECORD && satellites.len() < count))
            && phase_shift_satellites(satellites, line).is_ok()
    };
    let (count_after, satellites_after) = (
        rest.get(1).copied().unwrap_or(""),
        rest.get(2..).unwrap_or(&[]),
    );
    let as_correction = strict_f64_token(number, "phase_shift.correction_cycles", line).is_ok()
        && agrees(count_after, satellites_after);
    let as_count = agrees(number, &rest[1..]);
    match (as_correction, as_count) {
        (false, true) => Ok(fields("", number, &rest[1..])),
        (true, true) if !number.contains('.') => Err(Error::Parse(format!(
            "RINEX OBS SYS / PHASE SHIFT record is ambiguous: {number:?} reads as a correction and as the satellite count after a blank correction, in {line:?}"
        ))),
        _ => Ok(fields(number, count_after, satellites_after)),
    }
}

/// Columns of one `GLONASS COD/PHS/BIS` entry, `1X,A3,1X,F8.3`, and the entries
/// a record holds.
const GLONASS_BIAS_ENTRY_WIDTH: usize = 13;
const GLONASS_BIAS_ENTRIES_PER_RECORD: usize = 4;

/// A `GLONASS COD/PHS/BIS` record's codes and biases read from their columns,
/// `4(1X,A3,1X,F8.3)`, a blank bias as empty text, or `None` when the record is
/// not laid out in them: a gap holding text, a code not three characters, a
/// bias that does not read as a number, a bias with no code, or text past the
/// fourth entry.
fn glonass_bias_columns(content: &str) -> Option<Vec<(&str, &str)>> {
    if !content.is_ascii() {
        return None;
    }
    let column = |start: usize, end: usize| {
        content
            .get(start.min(content.len())..end.min(content.len()))
            .unwrap_or("")
    };
    let blank = |start: usize, end: usize| column(start, end).trim().is_empty();
    let width = GLONASS_BIAS_ENTRY_WIDTH * GLONASS_BIAS_ENTRIES_PER_RECORD;
    if !blank(width, content.len()) {
        return None;
    }
    let mut entries = Vec::new();
    for entry in 0..GLONASS_BIAS_ENTRIES_PER_RECORD {
        let start = entry * GLONASS_BIAS_ENTRY_WIDTH;
        if !blank(start, start + 1) || !blank(start + 4, start + 5) {
            return None;
        }
        let code = column(start + 1, start + 4).trim();
        let bias = column(start + 5, start + GLONASS_BIAS_ENTRY_WIDTH).trim();
        if code.is_empty() {
            if !bias.is_empty() {
                return None;
            }
            continue;
        }
        if code.len() != GLONASS_BIAS_CODE_WIDTH
            || bias.contains(' ')
            || !bias.is_empty() && bias.parse::<f64>().is_err()
        {
            return None;
        }
        entries.push((code, bias));
    }
    Some(entries)
}

/// The fields of a `SYS / PHASE SHIFT` record.
struct PhaseShiftFields<'a> {
    system: &'a str,
    code: &'a str,
    correction: &'a str,
    count: &'a str,
    satellites: Vec<&'a str>,
}

/// A `SYS / PHASE SHIFT` record's fields read from their columns,
/// `A1,1X,A3,1X,F8.5,2X,I2.2,10(1X,A3)`, or `None` when the record is not laid
/// out in them: a gap holding text, or a field that does not read as what it is.
fn phase_shift_columns(content: &str) -> Option<PhaseShiftFields<'_>> {
    if !content.is_ascii() {
        return None;
    }
    let column =
        |start: usize, end: usize| content.get(start..end.min(content.len())).unwrap_or("");
    let blank = |start: usize, end: usize| column(start, end).trim().is_empty();
    let system = column(0, 1).trim();
    let code = column(2, 5).trim();
    let correction = column(6, 14).trim();
    let count = column(16, 18).trim();
    if system.is_empty()
        || !blank(1, 2)
        || code.len() != 3
        || !blank(5, 6)
        || correction.contains(' ')
        || !correction.is_empty() && correction.parse::<f64>().is_err()
        || !blank(14, 16)
        || !count.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let mut satellites = Vec::new();
    let mut start = 18;
    while start < content.len() {
        if !blank(start, start + 1) {
            return None;
        }
        let token = column(start + 1, start + 4).trim();
        if token.contains(' ') || token.is_empty() && !blank(start, content.len()) {
            return None;
        }
        if !token.is_empty() {
            satellites.push(token);
        }
        start += 4;
    }
    Some(PhaseShiftFields {
        system,
        code,
        correction,
        count,
        satellites,
    })
}

/// Add to a union of code lists the codes a list declares that it does not
/// yet hold, in the order the list declares them. A code the list declares
/// more than once is held as many times.
pub(crate) fn extend_code_union(union: &mut Vec<String>, list: &[String]) {
    let mut additions: Vec<String> = Vec::new();
    let mut declared: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for code in list {
        let wanted = declared.entry(code.as_str()).or_default();
        *wanted += 1;
        let held = union.iter().filter(|held| *held == code).count()
            + additions.iter().filter(|added| *added == code).count();
        if held < *wanted {
            additions.push(code.clone());
        }
    }
    union.extend(additions);
}

/// Where each code of a list sits in a union holding it: the `n`th time the
/// list declares a code, at the union's `n`th copy of it. `None` for a code the
/// union does not hold that many times.
pub(crate) fn union_positions(list: &[String], union: &[String]) -> Vec<Option<usize>> {
    let mut seen: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    list.iter()
        .map(|code| {
            let copy = seen.entry(code.as_str()).or_default();
            let position = union
                .iter()
                .enumerate()
                .filter(|(_, held)| *held == code)
                .nth(*copy)
                .map(|(position, _)| position);
            *copy += 1;
            position
        })
        .collect()
}

/// Whether the file header declared no version 2 names.
fn header_names_fallback_is_empty(segments: &[CodeListSegment]) -> bool {
    segments
        .first()
        .is_none_or(|segment| segment.names.is_empty())
}

/// A value read by a code list, placed at its code's position in the union,
/// blank at every position the list does not declare. `None` when a value
/// has no position.
pub(crate) fn values_in_union(
    values: &[ObsValue],
    positions: &[Option<usize>],
    width: usize,
) -> Option<Vec<ObsValue>> {
    if values.len() > positions.len() {
        return None;
    }
    let mut placed = vec![
        ObsValue {
            value: None,
            lli: None,
            ssi: None,
        };
        width
    ];
    for (value, position) in values.iter().zip(positions) {
        *placed.get_mut((*position)?)? = *value;
    }
    Some(placed)
}

/// Wrap an error laying an event's records over the header with the epoch
/// they follow.
fn event_records_error(epoch_index: usize, error: Error) -> Error {
    match error {
        Error::Parse(message) => Error::Parse(format!(
            "RINEX OBS epoch {epoch_index} event records: {message}"
        )),
        other => other,
    }
}

/// The header in effect at each epoch of a product: the file header, and from
/// each event whose records take effect, the header those records lay over the
/// one before. Built once by [`RinexObs::header_timeline`], so a loop over the
/// epochs looks each header up rather than reading the event records again.
#[derive(Debug, Clone, PartialEq)]
pub struct ObsHeaderTimeline {
    file: ObsHeader,
    later: Vec<(usize, ObsHeader)>,
}

impl ObsHeaderTimeline {
    /// The header in effect at an epoch index: the file header with every event
    /// at or before it laid over it. Past the last epoch, the header after every
    /// event.
    pub fn at(&self, epoch_index: usize) -> &ObsHeader {
        let after = self
            .later
            .partition_point(|(first, _)| *first <= epoch_index);
        match after.checked_sub(1).and_then(|index| self.later.get(index)) {
            Some((_, header)) => header,
            None => &self.file,
        }
    }

    /// The position, in [`ObsHeaderTimeline::segments`] order, of the header in
    /// effect at an epoch index.
    pub fn segment_index(&self, epoch_index: usize) -> usize {
        self.later
            .partition_point(|(first, _)| *first <= epoch_index)
    }

    /// A timeline holding the file header alone, for a consumer that cannot
    /// return an error and has reported the event records it could not read.
    pub(crate) fn file_only(file: ObsHeader) -> Self {
        Self {
            file,
            later: Vec::new(),
        }
    }

    /// Each header with the index of the first epoch it is in effect at, in
    /// file order, beginning with the file header at index 0.
    pub fn segments(&self) -> impl Iterator<Item = (usize, &ObsHeader)> + '_ {
        core::iter::once((0, &self.file))
            .chain(self.later.iter().map(|(first, header)| (*first, header)))
    }
}

impl RinexObs {
    /// The header in effect at every epoch, for a loop over the epochs.
    ///
    /// # Errors
    ///
    /// [`Error::Parse`] when an event's header record does not read, as the
    /// reader would have refused it; a product read from text never holds one.
    pub fn header_timeline(&self) -> Result<ObsHeaderTimeline> {
        let version = self.header.version;
        let mut later: Vec<(usize, ObsHeader)> = Vec::new();
        for (index, epoch) in self.epochs.iter().enumerate() {
            if !applies_header_records(epoch.flag)
                || !epoch
                    .special_records
                    .iter()
                    .any(|record| event_record_takes_effect(version, record))
            {
                continue;
            }
            let mut next = later
                .last()
                .map_or(&self.header, |(_, header)| header)
                .clone();
            apply_event_records(&mut next, &epoch.special_records)
                .map_err(|error| event_records_error(index, error))?;
            later.push((index, next));
        }
        Ok(ObsHeaderTimeline {
            file: self.header.clone(),
            later,
        })
    }

    /// The header in effect at one epoch: the file header with the header
    /// records of every event at or before `epoch_index` laid over it. Its
    /// `obs_codes` is the product's union, and its `declared_obs_codes` the
    /// lists in effect at the epoch. A loop over many epochs uses
    /// [`RinexObs::header_timeline`] instead.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when `epoch_index` is past the last epoch, and
    /// [`Error::Parse`] when an event's header record does not read.
    pub fn header_at(&self, epoch_index: usize) -> Result<ObsHeader> {
        let Some(epochs) = self.epochs.get(..=epoch_index) else {
            return Err(Error::InvalidInput(format!(
                "RINEX OBS epoch index {epoch_index} is past the product's {} epochs",
                self.epochs.len()
            )));
        };
        let mut header = self.header.clone();
        for (index, epoch) in epochs.iter().enumerate() {
            if applies_header_records(epoch.flag) {
                apply_event_records(&mut header, &epoch.special_records)
                    .map_err(|error| event_records_error(index, error))?;
            }
        }
        Ok(header)
    }
}

mod write;
pub use write::{ObsDowngradeChange, RinexObsWriteError};

#[cfg(all(test, sidereon_repo_tests))]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
