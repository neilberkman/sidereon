//! SP3-c / SP3-d precise-ephemeris parser.
//!
//! Parses the IGS SP3 precise orbit/clock format, both **SP3-c** and **SP3-d**
//! (Hilla 2016), into a typed [`Sp3`] product. The parser is multi-GNSS,
//! handles position/clock records plus optional velocity records,
//! missing-value sentinels, predicted / clock-event / maneuver flags, and a
//! system-aware [`GnssSatelliteId`]; the product's time system is read from the
//! header.
//!
//! # Build vs adopt
//!
//! The spec permits using the `sp3` crate (MPL-2.0) as a deterministic byte
//! reader, OR hand-rolling the record parsing. **This module hand-rolls it**,
//! deliberately:
//!
//! - The `refs/sp3` crate hard-depends on `hifitime` for its `Epoch`,
//!   `TimeScale`, and `Duration`, and on `flate2`. `sidereon-core` models time
//!   with the **core crate's own** [`Instant`] / [`TimeScale`]
//!   family, which is hifitime-free; adopting the `sp3` crate would
//!   invert that and pull a parallel time stack into the GNSS layer.
//! - The `sp3` crate also carries its own `SV` / `Constellation` identifiers
//!   that duplicate this crate's [`GnssSatelliteId`] / [`GnssSystem`].
//! - The SP3 record grammar is small, fixed-column, and fully specified, so a
//!   byte reader is low-risk. (Note: the `refs/sp3` velocity parser at
//!   `parsing.rs:241-245` has an axis bug - it reuses the Y component for X;
//!   this module reads each axis independently and is unit-tested for it.)
//!
//! Parsing only is adopted-grade work; it is **not** a contested float recipe.
//! The interpolation that consumes this product is built
//! separately to match the `scipy.interpolate` reference and is out of scope
//! for this module.
//!
//! # Units
//!
//! SP3 stores positions in **kilometers** and clock offsets in **microseconds**
//! (velocities in dm/s, clock-rate in 1e-4 us/s). This parser converts at parse
//! time to the crate's internal SI base units - positions in **meters**
//! (`km * 1000.0`), clocks in **seconds** (`us * 1e-6`), velocities in **m/s**
//! (`(dm/s) * 1e-1`), clock-rate in **s/s** (`(1e-4 us/s) * 1e-10`). Each scale
//! factor is applied as a single multiply so the operation order is fixed for
//! the clock-unit-conversion golden test.
//!
//! # Frames
//!
//! Positions/velocities are returned as frame-tagged [`ItrfPositionM`] /
//! [`ItrfVelocityMS`], never a bare `position_m`.

use std::collections::BTreeMap;

use crate::astro::time::civil::split_julian_date;
use crate::astro::time::model::{Instant, InstantRepr, JulianDateSplit, TimeScale};

use crate::constants::{KM_TO_M, US_TO_S};
use crate::format::columns::{
    char_at, raw_field as field, raw_field_from as field_from, strict_f64,
};
use crate::format::{Diagnostics, RecordRef, Skip, SkipReason};
use crate::frame::{ItrfPositionM, ItrfVelocityMS};
use crate::id::{is_valid_prn, GnssSatelliteId, GnssSystem};
use crate::validate;
use crate::{Error, Result};

/// SP3 missing/bad position component sentinel, in kilometers.
///
/// SP3 writes a satellite with no usable orbit as a position record of exactly
/// `0.000000 0.000000 0.000000`. We treat an all-zero position as "missing"
/// (matching the `refs/sp3` validity guard at `parsing.rs:186`): a satellite is
/// never legitimately at the geocenter.
const MISSING_POSITION_KM: f64 = 0.0;
/// SP3 missing velocity component sentinel, in decimeters per second.
///
/// Velocity products still carry a `V` record for each `P` record. When no
/// velocity estimate exists, the record uses the all-zero vector sentinel rather
/// than being omitted; do not surface that as a fabricated stationary satellite.
const MISSING_VELOCITY_DM_S: f64 = 0.0;

/// SP3 bad-clock sentinel, in microseconds: `999999.999999`.
///
/// A clock value at or above this magnitude means "no clock estimate"; it is
/// surfaced as `clock_s = None`, not converted.
const BAD_CLOCK_US: f64 = 999_999.999_999;

/// SP3 velocity records are in decimeters per second; dm/s -> m/s is `* 0.1`.
const DM_S_TO_M_S: f64 = 1.0e-1;
/// SP3 clock-rate is in 1e-4 microseconds/second; -> s/s is `* 1e-10`.
const CLOCK_RATE_TO_S_PER_S: f64 = 1.0e-10;

/// SP3 format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sp3Version {
    /// SP3-a (legacy, GPS-only).
    A,
    /// SP3-b.
    B,
    /// SP3-c.
    C,
    /// SP3-d (multi-GNSS, Hilla 2016).
    D,
}

impl Sp3Version {
    fn from_char(c: char) -> Result<Self> {
        match c {
            'a' | 'A' => Ok(Sp3Version::A),
            'b' | 'B' => Ok(Sp3Version::B),
            'c' | 'C' => Ok(Sp3Version::C),
            'd' | 'D' => Ok(Sp3Version::D),
            other => Err(Error::Parse(format!("unknown SP3 version '{other}'"))),
        }
    }
}

/// What kind of records the file carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sp3DataType {
    /// Position + clock records only (`#?P...`).
    Position,
    /// Position + velocity (+ clock + clock-rate) records (`#?V...`).
    Velocity,
}

impl Sp3DataType {
    fn from_char(c: char) -> Result<Self> {
        match c {
            'P' => Ok(Sp3DataType::Position),
            'V' => Ok(Sp3DataType::Velocity),
            other => Err(Error::Parse(format!("unknown SP3 data type '{other}'"))),
        }
    }
}

/// SP3 time-system labels from the `%c` descriptor.
///
/// The core [`TimeScale`] model does not distinguish every SP3 label as its own
/// global scale. Keep the exact SP3 label here so products using GLONASS, QZSS,
/// or IRNSS time are accepted and can be serialized without being silently
/// relabeled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sp3TimeSystem {
    /// GPS time (`GPS`).
    Gps,
    /// GLONASS UTC time system (`GLO`).
    Glonass,
    /// Galileo system time (`GAL`).
    Galileo,
    /// International Atomic Time (`TAI`).
    Tai,
    /// Coordinated Universal Time (`UTC`).
    Utc,
    /// QZSS time (`QZS`).
    Qzss,
    /// BeiDou time (`BDT`).
    Beidou,
    /// IRNSS / NavIC time (`IRN`).
    Irnss,
}

impl Sp3TimeSystem {
    /// Canonical three-character SP3 label.
    pub fn label(self) -> &'static str {
        match self {
            Sp3TimeSystem::Gps => "GPS",
            Sp3TimeSystem::Glonass => "GLO",
            Sp3TimeSystem::Galileo => "GAL",
            Sp3TimeSystem::Tai => "TAI",
            Sp3TimeSystem::Utc => "UTC",
            Sp3TimeSystem::Qzss => "QZS",
            Sp3TimeSystem::Beidou => "BDT",
            Sp3TimeSystem::Irnss => "IRN",
        }
    }

    /// Core time scale used to tag parsed [`Instant`] values.
    ///
    /// For labels the core model has exactly, this is the direct equivalent. For
    /// SP3-only labels, the exact product label remains available through
    /// [`Sp3Header::time_system`], and this value preserves the existing
    /// interpolation-axis API until the global time model grows those scales.
    pub fn time_scale(self) -> TimeScale {
        match self {
            Sp3TimeSystem::Gps | Sp3TimeSystem::Irnss => TimeScale::Gpst,
            // QZSST is the exact core scale for the SP3 "QZS" label (nominally
            // synchronous with GPST); IRNSS has no distinct core scale yet.
            Sp3TimeSystem::Qzss => TimeScale::Qzsst,
            Sp3TimeSystem::Glonass | Sp3TimeSystem::Utc => TimeScale::Utc,
            Sp3TimeSystem::Galileo => TimeScale::Gst,
            Sp3TimeSystem::Tai => TimeScale::Tai,
            Sp3TimeSystem::Beidou => TimeScale::Bdt,
        }
    }

    fn civil_second_policy(self) -> validate::CivilSecondPolicy {
        match self {
            Sp3TimeSystem::Glonass | Sp3TimeSystem::Utc => validate::CivilSecondPolicy::UtcLike,
            Sp3TimeSystem::Gps
            | Sp3TimeSystem::Galileo
            | Sp3TimeSystem::Tai
            | Sp3TimeSystem::Qzss
            | Sp3TimeSystem::Beidou
            | Sp3TimeSystem::Irnss => validate::CivilSecondPolicy::Continuous,
        }
    }
}

/// Per-record quality / status flags (SP3-c columns 75-80, SP3-d same layout).
///
/// All four flags are independent and any combination may appear (e.g. a
/// predicted orbit during a maneuver). They are surfaced verbatim from the
/// record and never alter the parsed numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Sp3Flags {
    /// `E` in the clock-event column: a clock discontinuity occurred near this
    /// epoch; clock interpolation across it is unsafe.
    pub clock_event: bool,
    /// `P` in the clock-prediction column: the clock is predicted, not fitted.
    pub clock_predicted: bool,
    /// `M` in the maneuver column: the satellite was being maneuvered; the
    /// state is not suitable for precise navigation.
    pub maneuver: bool,
    /// `P` in the orbit-prediction column: the orbit is predicted, not fitted.
    pub orbit_predicted: bool,
}

/// A single satellite state at one SP3 epoch.
///
/// This is the spec's `Sp3State { position: ItrfPositionM, clock_s, velocity?,
/// clock_rate?, flags }`. The frame/units are encoded in the
/// member types; missing optional values are `None` rather than sentinels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sp3State {
    /// Satellite position in the ITRF/IGS ECEF frame, meters.
    pub position: ItrfPositionM,
    /// Satellite clock offset in **seconds** (`None` if the bad-clock sentinel
    /// `999999.999999` us was recorded).
    pub clock_s: Option<f64>,
    /// Satellite velocity in the ITRF/IGS ECEF frame, m/s (present only for
    /// velocity products).
    pub velocity: Option<ItrfVelocityMS>,
    /// Satellite clock rate in **seconds per second** (present only for
    /// velocity products that carry a clock-rate field).
    pub clock_rate_s_s: Option<f64>,
    /// Per-record status flags.
    pub flags: Sp3Flags,
}

/// Parsed SP3 header.
#[derive(Debug, Clone, PartialEq)]
pub struct Sp3Header {
    /// SP3 format version (`a`/`b`/`c`/`d`).
    pub version: Sp3Version,
    /// Whether the file carries velocity records.
    pub data_type: Sp3DataType,
    /// Number of parsed epochs in the canonical product.
    pub num_epochs: u64,
    /// Coordinate-system / IGS-realization label (e.g. `IGS14`, `ITRF2`).
    pub coordinate_system: String,
    /// Orbit-type label (e.g. `FIT`, `BHN`).
    pub orbit_type: String,
    /// Producing agency.
    pub agency: String,
    /// GNSS week number (in the file's time system).
    pub gnss_week: u32,
    /// Seconds of week of the first epoch.
    pub seconds_of_week: f64,
    /// Nominal epoch spacing in seconds.
    pub epoch_interval_s: f64,
    /// Modified Julian Day of the first epoch (integer part).
    pub mjd: u32,
    /// Fractional day of the first epoch.
    pub mjd_fraction: f64,
    /// Time system label the epochs are expressed in. For SP3-b/c/d this is read
    /// strictly from the first `%c` descriptor (a missing/short/blank descriptor
    /// is a parse error, never a silent GPST default); SP3-a is implicitly GPST.
    pub time_system: Sp3TimeSystem,
    /// Core [`TimeScale`] used to tag parsed [`Instant`] values. See
    /// [`Sp3Header::time_system`] for the exact SP3 label when the product uses
    /// a standard SP3 time system that is not modeled as a distinct core scale.
    pub time_scale: TimeScale,
    /// The satellite list declared in the `+` header lines.
    pub satellites: Vec<GnssSatelliteId>,
    /// Per-satellite accuracy exponent codes from the `++` header lines,
    /// index-aligned with [`Sp3Header::satellites`].
    pub satellite_accuracy_codes: Vec<u16>,
}

/// A parsed SP3 precise-ephemeris product.
///
/// Construct with [`Sp3::parse`]. Epochs are stored in ascending order; each
/// epoch maps satellite -> [`Sp3State`]. Per-satellite/per-epoch access is via
/// [`Sp3::state`]; arbitrary-epoch interpolation is built separately to match
/// the parity reference and is not part of this parser.
#[derive(Debug, Clone, PartialEq)]
pub struct Sp3 {
    /// The parsed header.
    pub header: Sp3Header,
    /// Epochs in ascending time order, tagged with the header time scale.
    pub epochs: Vec<Instant>,
    /// `epoch_index -> (satellite -> state)`. Parallel to [`Sp3::epochs`].
    states: Vec<BTreeMap<GnssSatelliteId, Sp3State>>,
    /// `epoch_index -> (satellite -> native-unit node)`. Parallel to
    /// [`Sp3::epochs`]; populated **only** from genuine position records. The
    /// interpolator fits its spline over these (km/us straight from the ASCII,
    /// exactly as the `scipy`/`gnssanalysis` reference does); reconstructing km
    /// from the public meters (`km->m->km`) drifts up to 1 ULP and breaks the
    /// 0-ULP parity. See `sp3/interp.rs`.
    interp_raw: Vec<BTreeMap<GnssSatelliteId, RawNode>>,
    /// Free-form `/*` comment lines (notice retained for provenance).
    pub comments: Vec<String>,
    /// Count of entries skipped because their satellite token did not parse to a
    /// representable [`GnssSatelliteId`] (e.g. an extended GLONASS slot like `R28`
    /// beyond the engine's PRN cap): position/velocity records, plus `+`-header
    /// satellite declarations. Lets callers tell a clean file
    /// (`skipped_records == 0`) apart from one carrying unsupported satellites,
    /// without aborting the whole parse on one such entry. Mirrors
    /// [`crate::astro::sgp4::TleFile::skipped`].
    pub skipped_records: usize,
}

/// Native-unit interpolation node: the file's own km / microseconds, kept
/// verbatim from the ASCII so the spline fit is bit-identical to the reference.
/// Private - the public surface is meters/seconds via [`Sp3State`].
#[derive(Debug, Clone, Copy, PartialEq)]
struct RawNode {
    /// ECEF position in native SP3 kilometers (X/Y/Z), exact ASCII->f64.
    km: [f64; 3],
    /// Clock offset in native SP3 microseconds (`None` for the bad-clock
    /// sentinel), exact ASCII->f64.
    clock_us: Option<f64>,
    /// Whether this epoch carried the clock-event (`E`) flag (clock-arc split).
    clock_event: bool,
}

impl Sp3 {
    /// Parse an SP3-c or SP3-d byte buffer into a typed product.
    ///
    /// `bytes` is the full file content (already decompressed; this crate does
    /// not do gzip - that is a caller-layer I/O concern). Returns
    /// [`Error::Parse`] with a human-readable reason on malformed input.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| Error::Parse(format!("SP3 is not valid UTF-8: {e}")))?;
        Self::parse_str(text)
    }

    /// Parse from a `&str` (the UTF-8 fast path used by [`Sp3::parse`]).
    pub fn parse_str(text: &str) -> Result<Self> {
        if !text.is_ascii() {
            return Err(Error::Parse("SP3 product text must be ASCII".into()));
        }
        let mut parser = Parser::new();
        for (index, raw) in text.lines().enumerate() {
            parser.feed(raw, index + 1)?;
        }
        parser.finish()
    }

    /// The satellites present in this product (from the header satellite list).
    pub fn satellites(&self) -> &[GnssSatelliteId] {
        &self.header.satellites
    }

    /// Number of parsed epochs.
    pub fn epoch_count(&self) -> usize {
        self.epochs.len()
    }

    /// The state of `sat` at the parsed epoch with index `epoch_index`.
    ///
    /// Returns [`Error::EpochOutOfRange`] if the index is past the end, or
    /// [`Error::UnknownSatellite`] if the satellite has no record at that epoch.
    pub fn state(&self, sat: GnssSatelliteId, epoch_index: usize) -> Result<Sp3State> {
        let per_epoch = self.states.get(epoch_index).ok_or(Error::EpochOutOfRange)?;
        per_epoch
            .get(&sat)
            .copied()
            .ok_or(Error::UnknownSatellite(sat))
    }

    /// All `(satellite, state)` pairs recorded at `epoch_index`, in ascending
    /// satellite order.
    pub fn states_at(&self, epoch_index: usize) -> Result<&BTreeMap<GnssSatelliteId, Sp3State>> {
        self.states.get(epoch_index).ok_or(Error::EpochOutOfRange)
    }
}

impl core::str::FromStr for Sp3 {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse_str(s)
    }
}

#[cfg(test)]
impl Sp3 {}

/// Parse an SP3 time-system label.
///
/// SP3-c/-d encode the time system in the `%c` descriptor line (chars 9-12).
/// SP3-a is implicitly GPST. Unknown labels error rather than silently
/// defaulting, so a parity pipeline never mis-attributes an epoch's scale.
fn time_system_from_label(label: &str) -> Result<Sp3TimeSystem> {
    match label.trim() {
        "GPS" => Ok(Sp3TimeSystem::Gps),
        "GLO" => Ok(Sp3TimeSystem::Glonass),
        "GAL" => Ok(Sp3TimeSystem::Galileo),
        "TAI" => Ok(Sp3TimeSystem::Tai),
        "UTC" => Ok(Sp3TimeSystem::Utc),
        "QZS" => Ok(Sp3TimeSystem::Qzss),
        "BDT" | "BDS" => Ok(Sp3TimeSystem::Beidou),
        "IRN" => Ok(Sp3TimeSystem::Irnss),
        trimmed => Err(Error::Parse(format!(
            "unsupported SP3 time system '{trimmed}'"
        ))),
    }
}

/// Compute the integer-day / fraction split Julian date from a Gregorian UTC-ish
/// civil epoch, with the day fraction carried separately (Skyfield split
/// convention, matching [`JulianDateSplit`]).
///
/// SP3 epoch lines are civil dates in the file's *own* time system; we keep
/// them in that scale (no leap-second shifting here - that is a conversion
/// concern handled by the core `scales` machinery, not the parser). The
/// algorithm is the standard Fliegel-Van Flandern Gregorian-to-JDN, then the
/// time-of-day fraction. JDN is computed in integer arithmetic so the whole-day
/// boundary is exact; only the sub-day fraction is floating point.
fn civil_to_julian_split(civil: validate::ValidCivil) -> Result<JulianDateSplit> {
    // Canonical civil-to-split conversion: the integer JDN places the `*.5`
    // civil-midnight boundary and the within-day clock fields become the
    // fraction. SP3 epochs are civil days in the file's own scale (no leap
    // second). The carry below is retained for the rare epoch whose seconds
    // overflow a day.
    let (mut jd_whole, mut fraction) = split_julian_date(
        civil.year as i32,
        civil.month as i32,
        civil.day as i32,
        civil.hour as i32,
        civil.minute as i32,
        civil.second,
    );
    if fraction > 1.0 {
        let carry = fraction.floor();
        jd_whole += carry;
        fraction -= carry;
    }
    JulianDateSplit::new(jd_whole, fraction)
        .map_err(|error| Error::Parse(format!("invalid SP3 epoch Julian date: {error}")))
}

/// Incremental line-driven SP3 parser state machine.
struct Parser {
    version: Option<Sp3Version>,
    data_type: Option<Sp3DataType>,
    num_epochs: u64,
    coordinate_system: String,
    orbit_type: String,
    agency: String,
    gnss_week: u32,
    seconds_of_week: f64,
    epoch_interval_s: f64,
    mjd: u32,
    mjd_fraction: f64,
    time_system: Option<Sp3TimeSystem>,
    /// `+`-line declared satellites, in file order.
    sat_list: Vec<GnssSatelliteId>,
    /// `++`-line per-satellite accuracy codes, in satellite-list order.
    sat_accuracy_codes: Vec<u16>,
    /// Number of real (non-padding) `+`-line satellite slots seen, including any
    /// dropped because their token was unrepresentable. The `++` accuracy codes
    /// are positionally aligned with these declaration slots, so this is the axis
    /// the accuracy parser walks (not the filtered [`Self::sat_list`]).
    declared_sat_slots: usize,
    /// Declaration-slot indices (into the `declared_sat_slots` axis) whose token
    /// was unrepresentable and dropped from [`Self::sat_list`]. Their `++`
    /// accuracy columns must be skipped so the surviving satellites keep their own
    /// codes. Empty for every well-formed file, making the accuracy parse a no-op
    /// realignment in the common case.
    dropped_sat_slots: Vec<usize>,
    /// Cursor along the declaration-slot axis consumed by the `++` accuracy
    /// parser across one or more `++` lines.
    accuracy_slot_cursor: usize,
    /// `%c` descriptor lines seen so far (the first carries the time system).
    pc_count: u32,
    /// Header line 1 parsed?
    have_line1: bool,
    /// Header line 2 parsed?
    have_line2: bool,
    /// Epoch currently being filled.
    current_epoch: Option<Instant>,
    epochs: Vec<Instant>,
    states: Vec<BTreeMap<GnssSatelliteId, Sp3State>>,
    interp_raw: Vec<BTreeMap<GnssSatelliteId, RawNode>>,
    comments: Vec<String>,
    diagnostics: Diagnostics,
    done: bool,
}

impl Parser {
    fn new() -> Self {
        Self {
            version: None,
            data_type: None,
            num_epochs: 0,
            coordinate_system: String::new(),
            orbit_type: String::new(),
            agency: String::new(),
            gnss_week: 0,
            seconds_of_week: 0.0,
            epoch_interval_s: 0.0,
            mjd: 0,
            mjd_fraction: 0.0,
            time_system: None,
            sat_list: Vec::new(),
            sat_accuracy_codes: Vec::new(),
            declared_sat_slots: 0,
            dropped_sat_slots: Vec::new(),
            accuracy_slot_cursor: 0,
            pc_count: 0,
            have_line1: false,
            have_line2: false,
            current_epoch: None,
            epochs: Vec::new(),
            states: Vec::new(),
            interp_raw: Vec::new(),
            comments: Vec::new(),
            diagnostics: Diagnostics::new(),
            done: false,
        }
    }

    fn feed(&mut self, raw: &str, line_number: usize) -> Result<()> {
        if self.done {
            return Ok(());
        }
        // SP3 is fixed-column ASCII; trim only the trailing CR / newline noise,
        // never leading spaces (columns are significant).
        let line = raw.trim_end_matches(['\r', '\n']);

        if line == "EOF" {
            self.done = true;
            return Ok(());
        }
        if line.starts_with("/*") {
            // Comment line; columns 4.. are the text.
            if line.len() > 3 {
                self.comments.push(line[3..].trim_end().to_string());
            } else {
                self.comments.push(String::new());
            }
            return Ok(());
        }
        // Header line 2 (`##`) must be tested before line 1 (`#`).
        if line.starts_with("##") {
            self.parse_line2(line)?;
            return Ok(());
        }
        if line.starts_with('#') {
            self.parse_line1(line)?;
            return Ok(());
        }
        if line.starts_with('+') {
            self.parse_plus_line(line, line_number)?;
            return Ok(());
        }
        if line.starts_with("%c") {
            self.parse_pc_line(line)?;
            return Ok(());
        }
        if line.starts_with("%f") || line.starts_with("%i") {
            // Float/int accuracy descriptor lines - not needed for the typed
            // state; skipped deterministically.
            return Ok(());
        }
        if line.starts_with('*') {
            self.parse_epoch_line(line)?;
            return Ok(());
        }
        if line.starts_with('P') {
            self.parse_position_line(line, line_number)?;
            return Ok(());
        }
        if line.starts_with('V') {
            self.parse_velocity_line(line, line_number)?;
            return Ok(());
        }
        // Unknown / ignorable line (e.g. `%/`); skip without failing - SP3 has
        // optional descriptor lines a parser must tolerate.
        Ok(())
    }

    /// Header line 1: `#cP2020 ...` / `#dV...`.
    fn parse_line1(&mut self, line: &str) -> Result<()> {
        // Minimum well-formed line-1 length per the standard.
        if line.len() < 55 {
            return Err(Error::Parse(format!(
                "SP3 header line 1 too short: {line:?}"
            )));
        }
        let chars: Vec<char> = line.chars().collect();
        let version = Sp3Version::from_char(chars[1])?;
        self.version = Some(version);
        self.data_type = Some(Sp3DataType::from_char(chars[2])?);
        // SP3-a predates the %c time-system descriptor and is implicitly GPST.
        // Set it here so a (correct) SP3-a file with no %c line still resolves,
        // while SP3-b/c/d are left as None until a valid %c line proves the
        // scale (a missing %c then becomes a hard error, not a GPST default).
        if matches!(version, Sp3Version::A) {
            self.time_system = Some(Sp3TimeSystem::Gps);
        }

        // Column layout per the SP3 standard, matching the (round-trip-tested)
        // refs/sp3 line-1 reader: num_epochs 32..40, observables 40..45,
        // coord_system 45..51, orbit_type 51..55, agency 55...
        self.num_epochs = field(line, 32, 40)
            .trim()
            .parse::<u64>()
            .map_err(|_| Error::Parse(format!("SP3 num_epochs unparsable in {line:?}")))?;
        self.coordinate_system = field(line, 45, 51).trim().to_string();
        self.orbit_type = field(line, 51, 55).trim().to_string();
        self.agency = field_from(line, 55).trim().to_string();
        self.have_line1 = true;
        Ok(())
    }

    /// Header line 2: `## 2276  21600.00000000   900.00000000 60176 0.25...`.
    fn parse_line2(&mut self, line: &str) -> Result<()> {
        self.gnss_week = field(line, 3, 7)
            .trim()
            .parse::<u32>()
            .map_err(|_| Error::Parse(format!("SP3 GNSS week unparsable in {line:?}")))?;
        self.seconds_of_week = field(line, 8, 23)
            .trim()
            .parse::<f64>()
            .map_err(|_| Error::Parse(format!("SP3 seconds-of-week unparsable in {line:?}")))?;
        self.epoch_interval_s = field(line, 24, 38)
            .trim()
            .parse::<f64>()
            .map_err(|_| Error::Parse(format!("SP3 epoch interval unparsable in {line:?}")))?;
        self.mjd = field(line, 39, 44)
            .trim()
            .parse::<u32>()
            .map_err(|_| Error::Parse(format!("SP3 MJD unparsable in {line:?}")))?;
        self.mjd_fraction = strict_f64(field_from(line, 45), "mjd_fraction")
            .map_err(|error| map_field_error(error, line))?;
        self.have_line2 = true;
        Ok(())
    }

    /// `+` satellite-list line: `+   32   G01G02...` (3-char SV tokens from
    /// column 9 in groups of 17). Continuation `+` lines append more tokens.
    fn parse_plus_line(&mut self, line: &str, line_number: usize) -> Result<()> {
        if line.starts_with("++") {
            return self.parse_accuracy_line(line);
        }
        // SV tokens start at column 9 (0-based), each 3 chars, up to 17 per line.
        let mut col = 9;
        while col + 3 <= line.len() {
            let token = field(line, col, col + 3);
            let trimmed = token.trim();
            // Unused satellite slots are zero-filled, not a declaration. The
            // SP3 zero-fill varies between producers (`  0`, ` 00`, `000`), so
            // any all-zero (or blank) token is padding - never a satellite,
            // whose token is a system letter + PRN (or, in SP3-a, a non-zero
            // numeric PRN). Misreading ` 00` as an unrepresentable satellite
            // inflates `skipped_records` and breaks the parse/write/parse
            // round trip (the writer re-emits the canonical `  0`).
            if trimmed.is_empty() || trimmed.bytes().all(|b| b == b'0') {
                col += 3;
                continue;
            }
            // This is a real declaration slot; the `++` accuracy codes are aligned
            // to this axis, so track its index whether or not the token resolves.
            let slot_index = self.declared_sat_slots;
            self.declared_sat_slots += 1;
            if let Some(id) = parse_sv_token(token, self.version) {
                if !self.sat_list.contains(&id) {
                    self.sat_list.push(id);
                }
            } else {
                // A declared satellite whose token is not representable (e.g. an
                // extended GLONASS slot R28 beyond the engine's PRN cap) is
                // dropped from the satellite list, but counted rather than dropped
                // silently - consistent with the position/velocity record paths
                // (see `Sp3::skipped_records`). Record the slot so its accuracy
                // column is skipped, keeping the surviving codes aligned.
                self.push_unrepresentable_satellite_skip(line_number, token);
                self.dropped_sat_slots.push(slot_index);
            }
            col += 3;
        }
        Ok(())
    }

    /// `++` per-satellite accuracy-code line: 3-char integer fields from column
    /// 9, aligned with the `+` declaration slots.
    ///
    /// The columns track the `+` declaration order, so a column whose declaration
    /// slot was dropped (an unrepresentable satellite) is read and discarded, not
    /// pushed - otherwise the surviving satellites would inherit a neighbour's
    /// accuracy code. With no dropped slots this is exactly the 1:1 push as before.
    fn parse_accuracy_line(&mut self, line: &str) -> Result<()> {
        let mut col = 9;
        while col + 3 <= line.len() && self.accuracy_slot_cursor < self.declared_sat_slots {
            let token = field(line, col, col + 3);
            let trimmed = token.trim();
            let code = if trimmed.is_empty() {
                0
            } else {
                validate::strict_int::<u16>(trimmed, "satellite_accuracy_code")
                    .map_err(|error| map_field_error(error, line))?
            };
            if !self.dropped_sat_slots.contains(&self.accuracy_slot_cursor) {
                self.sat_accuracy_codes.push(code);
            }
            self.accuracy_slot_cursor += 1;
            col += 3;
        }
        Ok(())
    }

    /// `%c` descriptor: the first one (chars 9-12) carries the time system.
    fn parse_pc_line(&mut self, line: &str) -> Result<()> {
        if self.pc_count == 0 {
            // SP3-a is implicitly GPST regardless of descriptor content.
            if matches!(self.version, Some(Sp3Version::A)) {
                self.time_system = Some(Sp3TimeSystem::Gps);
            } else if line.len() >= 12 {
                let label = field(line, 9, 12);
                let trimmed = label.trim();
                // STRICT: a blank time-system field on the first %c is not GPST,
                // it is malformed. Reject rather than silently defaulting so a
                // precise pipeline never mis-attributes an epoch's scale.
                if trimmed.is_empty() {
                    return Err(Error::Parse(format!(
                        "SP3 %c time system is blank in {line:?}"
                    )));
                }
                self.time_system = Some(time_system_from_label(label)?);
            } else {
                // STRICT: a short %c line for SP3-b/c/d carries no time system
                // we can trust. Reject rather than defaulting to GPST.
                return Err(Error::Parse(format!(
                    "SP3 %c descriptor too short to carry a time system: {line:?}"
                )));
            }
        }
        self.pc_count += 1;
        Ok(())
    }

    /// Epoch line: `*  2020  6 24  0  0  0.00000000`.
    fn parse_epoch_line(&mut self, line: &str) -> Result<()> {
        // STRICT: by the time we reach data, the time system must be known -
        // implicitly GPST for SP3-a (set at line 1), or from a valid first %c
        // line for SP3-b/c/d. A missing/blank/short %c is an error, never GPST.
        let time_system = self.time_system.ok_or_else(|| {
            Error::Parse("SP3 epoch encountered with no time system (missing %c descriptor)".into())
        })?;
        let scale = time_system.time_scale();
        // Fields after the leading `*  ` (3 chars), then space-delimited.
        let body = &line[1..];
        let mut it = body.split_whitespace();
        let year: i64 = next_field(&mut it, "epoch year")?;
        let month: i64 = next_field(&mut it, "epoch month")?;
        let day: i64 = next_field(&mut it, "epoch day")?;
        let hour: i64 = next_field(&mut it, "epoch hour")?;
        let minute: i64 = next_field(&mut it, "epoch minute")?;
        let seconds: f64 = next_field(&mut it, "epoch seconds")?;

        let civil = validate::civil_datetime_with_second_policy(
            year,
            month,
            day,
            hour,
            minute,
            seconds,
            time_system.civil_second_policy(),
        )
        .map_err(|error| map_field_error(error, line))?;
        let split = civil_to_julian_split(civil)?;
        let epoch = Instant {
            scale,
            repr: InstantRepr::JulianDate(split),
        };
        self.epochs.push(epoch);
        self.states.push(BTreeMap::new());
        self.interp_raw.push(BTreeMap::new());
        self.current_epoch = Some(epoch);
        Ok(())
    }

    /// Position+clock record: `PG01  x  y  z  clk  ...flags`.
    fn parse_position_line(&mut self, line: &str, line_number: usize) -> Result<()> {
        if self.current_epoch.is_none() {
            return Err(Error::Parse(
                "SP3 position record before any epoch line".into(),
            ));
        }
        if line.len() < 46 {
            return Err(Error::Parse(format!(
                "SP3 position record truncated before vector fields in {line:?}"
            )));
        }
        let token = field(line, 1, 4);
        let Some(sat) = parse_sv_token(token, self.version) else {
            // A token that does not parse to a representable `GnssSatelliteId`
            // (e.g. an extended GLONASS slot like R28 beyond the engine's PRN
            // cap) is an independent, unsupported record. One such record must
            // not reject the whole file - skip and count it, mirroring nav
            // `parse_glonass` and `parse_tle_file`.
            self.push_unrepresentable_satellite_skip(line_number, token);
            return Ok(());
        };

        // The header `+` lines are the authoritative satellite declaration; a
        // position record for an undeclared satellite is malformed. Accepting it
        // would store a state the writer (which emits only declared satellites)
        // cannot reproduce, breaking parse/encode/parse round-tripping.
        if !self.sat_list.contains(&sat) {
            return Err(Error::Parse(format!(
                "SP3 position record for satellite {token:?} not in the header satellite list"
            )));
        }

        let x_km = parse_coord(line, 4, 18)?;
        let y_km = parse_coord(line, 18, 32)?;
        let z_km = parse_coord(line, 32, 46)?;

        // All-zero position is the missing-orbit sentinel: skip the record.
        if x_km == MISSING_POSITION_KM && y_km == MISSING_POSITION_KM && z_km == MISSING_POSITION_KM
        {
            return Ok(());
        }

        let clock_us = parse_clock_us(line)?;
        let clock_s = clock_us.map(|us| us * US_TO_S);

        let flags = parse_flags(line);

        let position = ItrfPositionM::new(x_km * KM_TO_M, y_km * KM_TO_M, z_km * KM_TO_M)
            .map_err(|e| Error::Parse(format!("SP3 invalid position record: {e}")))?;
        let state = Sp3State {
            position,
            clock_s,
            velocity: None,
            clock_rate_s_s: None,
            flags,
        };
        let idx = self.states.len() - 1;
        self.states[idx].insert(sat, state);
        // Keep the native-unit node for the interpolation path (see RawNode):
        // the spline must fit the file's own km/us, not the km->m->km round trip.
        self.interp_raw[idx].insert(
            sat,
            RawNode {
                km: [x_km, y_km, z_km],
                clock_us,
                clock_event: flags.clock_event,
            },
        );
        Ok(())
    }

    /// Velocity record: `VG01  vx  vy  vz  clkrate ...`. Augments the matching
    /// position record at the current epoch (must follow it).
    fn parse_velocity_line(&mut self, line: &str, line_number: usize) -> Result<()> {
        if self.current_epoch.is_none() {
            return Err(Error::Parse(
                "SP3 velocity record before any epoch line".into(),
            ));
        }
        if line.len() < 46 {
            return Err(Error::Parse(format!(
                "SP3 velocity record truncated before vector fields in {line:?}"
            )));
        }
        let token = field(line, 1, 4);
        let Some(sat) = parse_sv_token(token, self.version) else {
            // Unparsable / out-of-range satellite token: skip and count, same
            // as the position-record path above.
            self.push_unrepresentable_satellite_skip(line_number, token);
            return Ok(());
        };

        // SP3 velocity is in dm/s; read each axis independently (the refs/sp3
        // crate has a bug here that reuses Y for X - we do not).
        let vx_dm_s = parse_coord(line, 4, 18)?;
        let vy_dm_s = parse_coord(line, 18, 32)?;
        let vz_dm_s = parse_coord(line, 32, 46)?;

        let missing_velocity = vx_dm_s == MISSING_VELOCITY_DM_S
            && vy_dm_s == MISSING_VELOCITY_DM_S
            && vz_dm_s == MISSING_VELOCITY_DM_S;
        let velocity = ItrfVelocityMS::new(
            vx_dm_s * DM_S_TO_M_S,
            vy_dm_s * DM_S_TO_M_S,
            vz_dm_s * DM_S_TO_M_S,
        )
        .map_err(|e| Error::Parse(format!("SP3 invalid velocity record: {e}")))?;

        // Clock-rate field shares the clock column; bad-clock sentinel applies.
        let clock_rate_s_s = parse_clock_us(line)?.map(|rate| rate * CLOCK_RATE_TO_S_PER_S);

        let idx = self.states.len() - 1;
        match self.states[idx].get_mut(&sat) {
            Some(state) if !missing_velocity => {
                state.velocity = Some(velocity);
                state.clock_rate_s_s = clock_rate_s_s;
            }
            Some(_) => {}
            None => {
                // A V-record always follows its P-record for the same satellite
                // at the same epoch (SP3 format invariant). With no preceding
                // P-record this satellite has NO valid position at this epoch;
                // synthesizing one (e.g. the geocenter (0,0,0)) would fabricate
                // an orbit that the all-zero missing-orbit guard exists to
                // reject, and would leak through the public state()/states_at().
                // Treat it as malformed and skip - consistent with the parser's
                // tolerant skipping of other malformed records. No state is
                // inserted, so the satellite stays UnknownSatellite at this
                // epoch and no (0,0,0) position is ever exposed.
            }
        }
        Ok(())
    }

    fn push_unrepresentable_satellite_skip(&mut self, line_number: usize, token: &str) {
        self.diagnostics.push_skip(Skip {
            at: RecordRef::at_line(line_number).with_satellite(token),
            reason: SkipReason::UnrepresentableSatellite,
        });
    }

    fn finish(self) -> Result<Sp3> {
        if !self.have_line1 {
            return Err(Error::Parse("SP3 missing header line 1".into()));
        }
        if !self.have_line2 {
            return Err(Error::Parse("SP3 missing header line 2".into()));
        }
        let version = self
            .version
            .ok_or_else(|| Error::Parse("SP3 version not determined".into()))?;
        let data_type = self
            .data_type
            .ok_or_else(|| Error::Parse("SP3 data type not determined".into()))?;
        // STRICT: SP3-a is implicitly GPST (set at line 1); SP3-b/c/d must have
        // proved their scale from a valid first %c line. Never default here.
        let time_system = self.time_system.ok_or_else(|| {
            Error::Parse(
                "SP3 time system not determined (missing/short/blank %c descriptor)".into(),
            )
        })?;
        let time_scale = time_system.time_scale();

        let mut satellite_accuracy_codes = self.sat_accuracy_codes;
        satellite_accuracy_codes.truncate(self.sat_list.len());
        satellite_accuracy_codes.resize(self.sat_list.len(), 0);
        let skipped_records = self.diagnostics.skips.len();

        let header = Sp3Header {
            version,
            data_type,
            num_epochs: self.epochs.len() as u64,
            coordinate_system: self.coordinate_system,
            orbit_type: self.orbit_type,
            agency: self.agency,
            gnss_week: self.gnss_week,
            seconds_of_week: self.seconds_of_week,
            epoch_interval_s: self.epoch_interval_s,
            mjd: self.mjd,
            mjd_fraction: self.mjd_fraction,
            time_system,
            time_scale,
            satellites: self.sat_list,
            satellite_accuracy_codes,
        };

        Ok(Sp3 {
            header,
            epochs: self.epochs,
            states: self.states,
            interp_raw: self.interp_raw,
            comments: self.comments,
            skipped_records,
        })
    }
}

/// Parse a fixed-column float coordinate, mapping failures to a parse error
/// that names the offending text.
fn parse_coord(line: &str, start: usize, end: usize) -> Result<f64> {
    let raw = field(line, start, end).trim();
    strict_f64(raw, "coordinate").map_err(|error| map_field_error(error, line))
}

/// Parse the clock column (chars 46..60). Returns `None` for the bad-clock
/// sentinel `999999.999999` or an absent/blank field; `Some(us)` otherwise.
fn parse_clock_us(line: &str) -> Result<Option<f64>> {
    if line.len() <= 46 {
        return Ok(None);
    }
    let raw = field(line, 46, 60).trim();
    if raw.is_empty() {
        return Ok(None);
    }
    let value = strict_f64(raw, "clock").map_err(|error| map_field_error(error, line))?;
    // Sentinel: any value at or beyond the bad-clock magnitude is "no estimate".
    if value.abs() >= BAD_CLOCK_US {
        return Ok(None);
    }
    Ok(Some(value))
}

fn map_field_error(error: validate::FieldError, line: &str) -> Error {
    Error::Parse(format!("SP3 {error} in {line:?}"))
}

/// Parse the four status flags from their fixed columns (SP3-c/-d shared
/// layout): clock-event col 74 = `E`, clock-prediction col 75 = `P`,
/// maneuver col 78 = `M`, orbit-prediction col 79 = `P`.
fn parse_flags(line: &str) -> Sp3Flags {
    let at = |col: usize, want: char| -> bool { char_at(line, col) == Some(want) };
    Sp3Flags {
        clock_event: at(74, 'E'),
        clock_predicted: at(75, 'P'),
        maneuver: at(78, 'M'),
        orbit_predicted: at(79, 'P'),
    }
}

/// Parse a 3-char SV token (e.g. `G01`, `C30`, or a bare `  1` in SP3-a) into a
/// [`GnssSatelliteId`]. Returns `None` on an unrecognized token.
fn parse_sv_token(token: &str, version: Option<Sp3Version>) -> Option<GnssSatelliteId> {
    let token = token.trim();
    if token.is_empty() {
        return None;
    }
    let first = token.chars().next()?;
    if first.is_ascii_digit() {
        // SP3-a GPS-only: bare numeric PRN, optionally space-padded.
        if matches!(version, Some(Sp3Version::A)) || version.is_none() {
            let prn = token.parse::<u8>().ok()?;
            if !is_valid_prn(GnssSystem::Gps, prn) {
                return None;
            }
            return GnssSatelliteId::new(GnssSystem::Gps, prn).ok();
        }
        return None;
    }
    token.parse::<GnssSatelliteId>().ok()
}

/// Pull and parse the next whitespace-delimited field from an iterator.
fn next_field<T: std::str::FromStr>(
    it: &mut std::str::SplitWhitespace<'_>,
    what: &str,
) -> Result<T> {
    let tok = it
        .next()
        .ok_or_else(|| Error::Parse(format!("SP3 missing {what}")))?;
    tok.parse::<T>()
        .map_err(|_| Error::Parse(format!("SP3 {what} {tok:?} unparsable")))
}

mod combine;
mod interp;
mod samples;
mod write;

pub use combine::{
    align_clock_reference, clock_reference_offset, merge, AgreementMetric, ClockReferenceOffset,
    EpochAgreement, MergeCombine, MergeFlag, MergeOptions, MergeReport,
};
pub use samples::{PreciseEphemerisSample, PreciseEphemerisSamples, PreciseSamplesError};

#[cfg(all(test, sidereon_repo_tests))]
mod tests;
