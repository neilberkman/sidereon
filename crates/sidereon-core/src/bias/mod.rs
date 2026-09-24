//! Offline GNSS code and phase bias products.
//!
//! The parsers are sans-I/O: callers pass bytes. A Bias-SINEX product keeps
//! every line it reads, in order, as its authority: the header and footer
//! lines, comments, every block including the blocks the reader does not
//! model, and solution rows it could not read. Records, header rows and
//! lookup metadata are derived from those lines, and the writer restates them
//! byte for byte. A CODE DCB product keeps its lines the same way, and the
//! writer restates a set read from DCB byte for byte; it generates DCB text
//! only for other sets, stating their records under the product metadata.
//!
//! Lookups return a [`BiasLookup`] status that separates an available value
//! from absent coverage, a query on another time scale, and conflicting
//! records.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::str::FromStr;

use crate::astro::constants::time::SECONDS_PER_DAY_I64;
use crate::astro::time::civil::{
    day_of_year_int, exact_j2000_seconds_of_split, split_julian_date, J2000_JULIAN_DAY_NUMBER,
    J2000_NOON_OFFSET_S,
};
use crate::astro::time::exact::ExactSeconds;
use crate::astro::time::model::{Instant, InstantRepr, JulianDateSplit, TimeScale};
use crate::astro::time::scales::julian_day_number;
use crate::constants::{C_M_S, NS_TO_S};
use crate::format::columns::strict_f64;
pub use crate::format::{Diagnostics, Parsed, RecordRef, Skip, SkipReason, Warning, WarningKind};
pub use crate::validate::FieldError;
use crate::validate::{self, CivilSecondPolicy};
use crate::{frequencies, GnssSatelliteId, GnssSystem};

/// The only format version Bias-SINEX 1.00 defines (section 4.1, `F4.2`).
const BIAS_SINEX_VERSION: &str = "1.00";
/// Denominator used when converting Bias-SINEX slope values and slope
/// uncertainties to the internal per-second representation and back.
pub const SINEX_BIAS_SLOPE_DENOMINATOR_S: f64 = 1.0;
const RINEX_VERSION_FOR_BIAS_CODES: f64 = 3.04;
/// Width of the Bias-SINEX header line (section 4.1).
const SINEX_HEADER_COLUMNS: usize = 74;
/// Column ranges of the header line fields (section 4.1): marker, version,
/// file agency, creation time, data agency, start, end, mode and count.
const SINEX_HEADER_FIELDS: [(usize, usize); 9] = [
    (0, 5),
    (6, 10),
    (11, 14),
    (15, 29),
    (30, 33),
    (34, 48),
    (49, 63),
    (64, 65),
    (66, 74),
];
/// Blocks Bias-SINEX 1.00 section 2.1 allows.
const SINEX_BLOCKS: [&str; 6] = [
    "FILE/REFERENCE",
    "FILE/COMMENT",
    "INPUT/ACKNOWLEDGMENTS",
    "BIAS/DESCRIPTION",
    "BIAS/RECEIVER_INFORMATION",
    "BIAS/SOLUTION",
];
/// Blocks Bias-SINEX 1.00 section 2.1 marks mandatory.
const SINEX_MANDATORY_BLOCKS: [&str; 3] = ["FILE/REFERENCE", "BIAS/DESCRIPTION", "BIAS/SOLUTION"];
/// Width of an `E21.15` estimate or slope field (section 4.8).
const SINEX_ESTIMATE_WIDTH: usize = 21;
/// Width of an `E11.6` standard-deviation field (section 4.8).
const SINEX_SIGMA_WIDTH: usize = 11;
/// Number of ulps searched on each side of a value for a spelling that reads
/// back to it exactly.
const EXACT_SPELLING_ULPS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Bias-SINEX solution records are classified by the token parsed into this
/// enum and by whether the record carries one observable or a pair.
///
/// The parser accepts the `OSB`, `DSB`, and `ISB` labels. OSB records address
/// one observable; DSB and ISB records carry a pair.
pub enum BiasKind {
    /// The `OSB` token, which is indexed by one observable and has no second
    /// observable in a solution record.
    Osb,
    /// The `DSB` token, which is resolved as a signed difference between two
    /// observables.
    Dsb,
    /// The `ISB` token, which requires two observables when a solution line is
    /// parsed.
    Isb,
}

impl BiasKind {
    fn label(self) -> &'static str {
        match self {
            Self::Osb => "OSB",
            Self::Dsb => "DSB",
            Self::Isb => "ISB",
        }
    }
}

impl FromStr for BiasKind {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim() {
            "OSB" => Ok(Self::Osb),
            "DSB" => Ok(Self::Dsb),
            "ISB" => Ok(Self::Isb),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Observable family a bias applies to, read from its observable code.
///
/// Bias-SINEX 1.00 section 4.8 distinguishes code from phase biases by the
/// RINEX 3 observable code, not by the unit: a `C` observable is a code
/// (pseudorange) bias and an `L` observable a phase bias.
pub enum BiasObservableFamily {
    /// A code (pseudorange) observable, such as `C1W`.
    Code,
    /// A carrier-phase observable, such as `L1C`.
    Phase,
    /// A DSB or ISB between a code and a phase observable. Such a record is
    /// kept, and code and phase lookups do not use it.
    Mixed,
}

impl BiasObservableFamily {
    /// Returns the family of a RINEX 3 observable code: `C` codes are code
    /// observables and `L` codes phase observables. Any other code has no
    /// bias family and returns `None`.
    pub fn of_observable(code: &str) -> Option<Self> {
        match code.as_bytes().first() {
            Some(b'C') => Some(Self::Code),
            Some(b'L') => Some(Self::Phase),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Unit a Bias-SINEX solution row states its values in (section 4.8).
pub enum BiasUnit {
    /// `ns`: values are nanoseconds, slopes nanoseconds per second. Code
    /// biases are always stated in this unit; phase biases may be.
    Nanoseconds,
    /// `cyc`: values are carrier cycles, slopes cycles per second. Only phase
    /// biases may be stated in this unit.
    Cycles,
}

impl BiasUnit {
    /// Returns the unit token as the solution block writes it.
    pub fn label(self) -> &'static str {
        match self {
            Self::Nanoseconds => "ns",
            Self::Cycles => "cyc",
        }
    }

    fn parse(token: &str) -> Option<Self> {
        match token {
            "ns" => Some(Self::Nanoseconds),
            "cyc" => Some(Self::Cycles),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// PRN and station columns are decoded into one of four target forms in this
/// enum, which controls the lookup key used for a bias record.
///
/// Station text is kept as the file states it: the trimmed nine-column
/// Bias-SINEX station field (station name, receiver group name or legacy
/// four-character code) or the trimmed sixteen-column CODE DCB station field.
/// Only the canonical [`BiasTargetKey`] used for lookup is normalized.
pub enum BiasTarget {
    /// A one-character PRN with no station, written back as the system letter.
    System(GnssSystem),
    /// A multi-character PRN parsed as [`GnssSatelliteId`] with no station.
    Satellite(GnssSatelliteId),
    /// A one-character PRN paired with a station string.
    Receiver {
        /// GNSS system identified by the one-character PRN column.
        system: GnssSystem,
        /// Receiver station identifier as the file states it, trimmed.
        station: String,
    },
    /// A multi-character PRN paired with a station.
    SatelliteReceiver {
        /// Satellite identified by the PRN column.
        sat: GnssSatelliteId,
        /// Receiver station identifier as the file states it, trimmed.
        station: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
/// Canonical index key corresponding to a [`BiasTarget`].
///
/// Receiver station strings are trimmed and uppercased by the constructors,
/// and never shortened, so two nine-character station identifiers that share
/// their first four characters stay distinct keys.
pub struct BiasTargetKey {
    /// System component copied from a target or from its satellite identifier.
    pub system: GnssSystem,
    /// Satellite component used to distinguish satellite-specific lookup from
    /// the system fallback, when present.
    pub sat: Option<GnssSatelliteId>,
    /// Normalized station component used to distinguish receiver lookup, when
    /// present.
    pub station: Option<String>,
}

impl BiasTargetKey {
    /// Creates the key used for a system-wide target, with no satellite or
    /// station component.
    pub fn system(system: GnssSystem) -> Self {
        Self {
            system,
            sat: None,
            station: None,
        }
    }

    /// Creates a key for one satellite by copying its system and identifier;
    /// the station component is absent.
    pub fn satellite(sat: GnssSatelliteId) -> Self {
        Self {
            system: sat.system,
            sat: Some(sat),
            station: None,
        }
    }

    /// Creates a key for one receiver in `system` and stores no satellite
    /// component.
    ///
    /// The station is trimmed and uppercased before it is stored.
    pub fn receiver(system: GnssSystem, station: &str) -> Self {
        Self {
            system,
            sat: None,
            station: Some(normalize_station(station)),
        }
    }

    /// Creates a key containing the satellite's system and identifier plus a
    /// normalized receiver station.
    ///
    /// The station is trimmed and uppercased before it is stored.
    pub fn satellite_receiver(sat: GnssSatelliteId, station: &str) -> Self {
        Self {
            system: sat.system,
            sat: Some(sat),
            station: Some(normalize_station(station)),
        }
    }

    /// Returns the four-character marker of the station component: its first
    /// four characters when they are ASCII letters or digits, as in a
    /// nine-character IGS station identifier. Receiver group names, which
    /// start with `@`, and shorter names have no marker.
    pub fn station_marker(&self) -> Option<&str> {
        self.station.as_deref().and_then(station_marker)
    }
}

impl From<&BiasTarget> for BiasTargetKey {
    fn from(value: &BiasTarget) -> Self {
        match value {
            BiasTarget::System(system) => Self::system(*system),
            BiasTarget::Satellite(sat) => Self::satellite(*sat),
            BiasTarget::Receiver { system, station } => Self::receiver(*system, station),
            BiasTarget::SatelliteReceiver { sat, station } => {
                Self::satellite_receiver(*sat, station)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// Bias-SINEX validity fields store this calendar epoch as year, day of year,
/// and second of day before lookup converts it to a split Julian date.
///
/// The three components are converted to a split Julian date by
/// [`bias_epoch_instant`].
pub struct BiasEpoch {
    /// Calendar year used by [`BiasEpoch::new`] to determine the leap-year day
    /// bound and by [`BiasEpoch::format_sinex`] as the four-digit prefix.
    pub year: i32,
    /// One-based day checked against the length of `year` and formatted as a
    /// three-digit SINEX component.
    pub day_of_year: u16,
    /// Second checked against `SECONDS_PER_DAY_I64` and formatted as a
    /// five-digit SINEX component.
    pub second_of_day: u32,
}

impl BiasEpoch {
    /// Constructs an epoch after validating its calendar day and second.
    ///
    /// `day_of_year` must be within the number of days in `year`, and
    /// `second_of_day` must not exceed `SECONDS_PER_DAY_I64`; violations return
    /// [`BiasError::InvalidEpoch`].
    pub fn new(year: i32, day_of_year: u16, second_of_day: u32) -> Result<Self, BiasError> {
        if !(1..=days_in_year(year)).contains(&i32::from(day_of_year)) {
            return Err(BiasError::InvalidEpoch);
        }
        if second_of_day > SECONDS_PER_DAY_I64 as u32 {
            return Err(BiasError::InvalidEpoch);
        }
        Ok(Self {
            year,
            day_of_year,
            second_of_day,
        })
    }

    /// Parses a Bias-SINEX epoch token.
    ///
    /// An empty token and `0000:000:00000` represent an absent epoch. Every
    /// other token must contain exactly three colon-separated integer fields
    /// accepted by [`BiasEpoch::new`].
    pub fn parse_sinex(token: &str) -> Result<Option<Self>, BiasError> {
        let token = token.trim();
        if token.is_empty() || token == "0000:000:00000" {
            return Ok(None);
        }
        let mut parts = token.split(':');
        let year = parse_int::<i32>(parts.next(), "bias epoch year")?;
        let doy = parse_int::<u16>(parts.next(), "bias epoch day")?;
        let sod = parse_int::<u32>(parts.next(), "bias epoch second")?;
        if parts.next().is_some() {
            return Err(BiasError::InvalidEpoch);
        }
        Self::new(year, doy, sod).map(Some)
    }

    /// Formats the epoch as the fixed-width `YYYY:DDD:SSSSS` form used by
    /// Bias-SINEX solution records.
    pub fn format_sinex(self) -> String {
        format!(
            "{:04}:{:03}:{:05}",
            self.year, self.day_of_year, self.second_of_day
        )
    }

    fn to_split(self) -> Result<JulianDateSplit, BiasError> {
        // Second 86400 of a day is the following midnight.
        let epoch = if i64::from(self.second_of_day) >= SECONDS_PER_DAY_I64 {
            self.next_midnight()?
        } else {
            self
        };
        let jdn = epoch.day_number();
        let (year, month, day) = crate::astro::time::civil::civil_from_julian_day_number(jdn);
        let (jd_whole, fraction) = split_julian_date(
            year as i32,
            month as i32,
            day as i32,
            0,
            0,
            f64::from(epoch.second_of_day),
        );
        JulianDateSplit::new(jd_whole, fraction).map_err(|_| BiasError::InvalidEpoch)
    }

    /// Whole seconds from an arbitrary fixed origin, so two epochs naming the
    /// same instant, such as `D:86400` and `(D+1):00000`, compare equal.
    fn instant_seconds(self) -> i64 {
        self.day_number() * SECONDS_PER_DAY_I64 + i64::from(self.second_of_day)
    }

    fn day_number(self) -> i64 {
        julian_day_number(self.year, 1, 1) + i64::from(self.day_of_year) - 1
    }

    /// The exact seconds since J2000 the epoch names; second 86400 of a day
    /// is the following midnight.
    fn exact_j2000_seconds(self) -> ExactSeconds {
        ExactSeconds::from_integer(
            i128::from(self.day_number() - J2000_JULIAN_DAY_NUMBER)
                * i128::from(SECONDS_PER_DAY_I64)
                - i128::from(J2000_NOON_OFFSET_S)
                + i128::from(self.second_of_day),
        )
    }

    /// Whole seconds from `self` to `later`, exact in integer arithmetic.
    fn seconds_until(self, later: Self) -> i64 {
        (later.day_number() - self.day_number()) * SECONDS_PER_DAY_I64
            + i64::from(later.second_of_day)
            - i64::from(self.second_of_day)
    }

    fn next_midnight(self) -> Result<Self, BiasError> {
        let jdn = julian_day_number(self.year, 1, 1) + i64::from(self.day_of_year);
        let (year, _month, _day) = crate::astro::time::civil::civil_from_julian_day_number(jdn);
        let doy = day_of_year_int(year as i32, 1, 1);
        let base = julian_day_number(year as i32, 1, 1);
        let day = (jdn - base + doy) as u16;
        Self::new(year as i32, day, 0)
    }

    fn normalize_end(self) -> Result<Self, BiasError> {
        if self.second_of_day == 86_399 {
            self.next_midnight()
        } else {
            Ok(self)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// One bias record read from a Bias-SINEX solution row or a CODE DCB row.
///
/// Values stated in nanoseconds are stored in seconds and values stated in
/// cycles are stored in cycles; [`BiasRecord::unit`] says which. The
/// observable family (code or phase) comes from the observable code, so a
/// phase bias stated in nanoseconds keeps its nanosecond unit and is still a
/// phase bias.
pub struct BiasRecord {
    /// Parsed classification that selects one-observable OSB or two-observable
    /// DSB/ISB indexing and query behavior.
    pub kind: BiasKind,
    /// Parsed PRN/station target used to build the set's [`BiasTargetKey`].
    pub target: BiasTarget,
    /// Optional SVN text copied from the solution line.
    pub svn: Option<String>,
    /// First observable code used alone for OSB indexing or with `obs2` for
    /// DSB/ISB indexing.
    pub obs1: String,
    /// Second observable for DSB and ISB records; absent for OSB records.
    pub obs2: Option<String>,
    /// Inclusive validity start, when the solution line supplies one.
    pub valid_from: Option<BiasEpoch>,
    /// Exclusive validity end, when the solution line supplies one. An end
    /// written at second 86399 is read as the following midnight.
    pub valid_until: Option<BiasEpoch>,
    /// Trimmed source start and end tokens retained for serialization.
    pub raw_epochs: (String, String),
    /// Bias value: seconds when [`BiasRecord::unit`] is nanoseconds, cycles
    /// when it is cycles.
    pub value: f64,
    /// Optional uncertainty in the same units as [`BiasRecord::value`].
    pub sigma: Option<f64>,
    /// Optional slope per second, in the units of [`BiasRecord::value`],
    /// referred to the epoch [`BiasRecord::slope_reference`] gives.
    pub slope: Option<f64>,
    /// Optional slope uncertainty, kept whether or not a slope is present.
    pub slope_sigma: Option<f64>,
    /// Code or phase, from the observable code (section 4.8).
    pub family: BiasObservableFamily,
    /// Unit the source row states its values in.
    pub unit: BiasUnit,
    /// One-based source line of the row, when the record was read from text.
    pub line: Option<usize>,
}

impl BiasRecord {
    /// Returns `true` for a phase bias, one whose observable is an `L` code.
    pub fn is_phase(&self) -> bool {
        self.family == BiasObservableFamily::Phase
    }

    /// Returns the epoch a sloped value refers to, as Bias-SINEX 1.00 section
    /// 5.1 defines it: the middle of a closed validity interval, the start
    /// when the end is undefined, and the end when the start is undefined.
    ///
    /// The interval is the one lookup uses, so an end written at second 86399
    /// counts as the following midnight.
    pub fn slope_reference(&self) -> BiasSlopeReference {
        match (self.valid_from, self.valid_until) {
            (Some(start), Some(end)) => BiasSlopeReference::Midpoint { start, end },
            (Some(start), None) => BiasSlopeReference::Start(start),
            (None, Some(end)) => BiasSlopeReference::End(end),
            (None, None) => BiasSlopeReference::Undefined,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Reference epoch of a sloped bias value (Bias-SINEX 1.00 section 5.1).
pub enum BiasSlopeReference {
    /// A closed interval: the value refers to the middle of `start..end`.
    Midpoint {
        /// Interval start.
        start: BiasEpoch,
        /// Interval end.
        end: BiasEpoch,
    },
    /// The end is undefined: the value refers to the start.
    Start(BiasEpoch),
    /// The start is undefined: the value refers to the end.
    End(BiasEpoch),
    /// Both bounds are undefined, so a slope has no reference epoch.
    Undefined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Interpretation declared by the Bias-SINEX `BIAS_MODE` description.
pub enum BiasMode {
    /// Selected by `BIAS_MODE ABSOLUTE`; non-OSB solution records then produce
    /// mismatch warnings.
    Absolute,
    /// Selected by `BIAS_MODE RELATIVE` and assigned to every parsed DCB set;
    /// OSB solution records then produce mismatch warnings.
    Relative,
    /// No usable declaration: `BIAS_MODE` is missing, unrecognized or
    /// declared twice with different values. The generated SINEX writer
    /// rejects it as missing mode metadata.
    #[default]
    Unspecified,
}

#[derive(Debug, Clone, PartialEq, Default)]
/// Bias-SINEX clock-reference description lines populate this per-system map,
/// which PPP code-bias correction later uses to select a reference pair.
///
/// Each system holds the pair its first declaration states; a system whose
/// declarations disagree is left out and reported. A declaration whose
/// observable fields are blank (satellite-station link biases, section 4.6)
/// holds blank observables.
pub struct ClockReferenceObservables {
    /// Map populated by `SATELLITE_CLOCK_REFERENCE_OBSERVABLES` lines and read
    /// by PPP code-bias correction as the pair for each satellite system.
    pub per_system: BTreeMap<GnssSystem, (String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Fields of the Bias-SINEX header line (section 4.1), as the file states
/// them.
///
/// A field the header line does not carry is `None`; that happens only in a
/// lenient read, since a strict read refuses an incomplete header.
pub struct BiasSinexHeader {
    /// Format version token. A strict read accepts only `1.00`; a lenient
    /// read keeps any other version as written and reports it.
    pub version: String,
    /// Agency creating the file.
    pub file_agency: Option<String>,
    /// Creation time token, `YYYY:DDD:SSSSS`.
    pub creation_time: Option<String>,
    /// Agency providing the data.
    pub data_agency: Option<String>,
    /// Solution start time token.
    pub start: Option<String>,
    /// Solution end time token.
    pub end: Option<String>,
    /// Bias mode token, `A` or `R`.
    pub mode: Option<String>,
    /// Number-of-estimates token, eight digits.
    pub estimate_count: Option<String>,
}

impl BiasSinexHeader {
    /// Builds a header for a product written from records, such as a CODE DCB
    /// set restated as Bias-SINEX. An undefined start or end is written as
    /// `0000:000:00000`. The writer states the mode and the estimate count
    /// from the set.
    pub fn new(
        file_agency: &str,
        creation_time: BiasEpoch,
        data_agency: &str,
        start: Option<BiasEpoch>,
        end: Option<BiasEpoch>,
    ) -> Self {
        let epoch = |value: Option<BiasEpoch>| {
            value
                .map(BiasEpoch::format_sinex)
                .unwrap_or_else(|| "0000:000:00000".to_string())
        };
        Self {
            version: BIAS_SINEX_VERSION.to_string(),
            file_agency: Some(file_agency.to_string()),
            creation_time: Some(creation_time.format_sinex()),
            data_agency: Some(data_agency.to_string()),
            start: Some(epoch(start)),
            end: Some(epoch(end)),
            mode: None,
            estimate_count: None,
        }
    }

    /// Returns the declared number of estimates when the token is a decimal
    /// integer.
    pub fn estimate_count_value(&self) -> Option<u64> {
        let token = self.estimate_count.as_deref()?;
        if token.is_empty() || !token.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        token.parse().ok()
    }

    /// Returns the creation time when its token is a defined epoch.
    pub fn creation_epoch(&self) -> Option<BiasEpoch> {
        BiasEpoch::parse_sinex(self.creation_time.as_deref()?)
            .ok()
            .flatten()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// One `FILE/REFERENCE` or `BIAS/DESCRIPTION` data row.
///
/// The keyword is the row's first token and the value everything after it,
/// with leading and trailing blanks removed and internal spacing kept. The
/// row's exact text stays in the product's source lines.
pub struct BiasInfoRow {
    /// Information type or description keyword, such as `DESCRIPTION` or
    /// `TIME_SYSTEM`.
    pub keyword: String,
    /// Text after the keyword.
    pub value: String,
    /// One-based source line.
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq, Default)]
/// Header metadata of a bias product.
///
/// Rows keep file order, repeats and internal spacing; the lookup helpers
/// read them without merging or overwriting anything.
pub struct BiasSetHeader {
    /// The Bias-SINEX header line fields, for a Bias-SINEX product or one
    /// given a header by [`BiasSet::set_sinex_header`].
    pub sinex: Option<BiasSinexHeader>,
    /// `FILE/REFERENCE` rows in file order.
    pub file_reference: Vec<BiasInfoRow>,
    /// `BIAS/DESCRIPTION` rows in file order.
    pub description: Vec<BiasInfoRow>,
    /// DCB options retained by `parse_code_dcb` for `write_code_dcb`, when
    /// available.
    pub dcb_meta: Option<CodeDcbOptions>,
}

impl BiasSetHeader {
    /// Returns the value of the first `BIAS/DESCRIPTION` row with `keyword`.
    pub fn description_value(&self, keyword: &str) -> Option<&str> {
        self.description
            .iter()
            .find(|row| row.keyword == keyword)
            .map(|row| row.value.as_str())
    }

    /// Returns the values of every `BIAS/DESCRIPTION` row with `keyword`, in
    /// file order. The values borrow from the header only.
    pub fn description_values<'a, 'k>(
        &'a self,
        keyword: &'k str,
    ) -> impl Iterator<Item = &'a str> + use<'a, 'k> {
        self.description
            .iter()
            .filter(move |row| row.keyword == keyword)
            .map(|row| row.value.as_str())
    }

    /// Returns the values of every `FILE/REFERENCE` row with `keyword`, in
    /// file order. The values borrow from the header only.
    pub fn file_reference_values<'a, 'k>(
        &'a self,
        keyword: &'k str,
    ) -> impl Iterator<Item = &'a str> + use<'a, 'k> {
        self.file_reference
            .iter()
            .filter(move |row| row.keyword == keyword)
            .map(|row| row.value.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Line ending a source line had in the input.
pub enum BiasLineTerminator {
    /// `\n`.
    Lf,
    /// `\r\n`.
    CrLf,
    /// The last line of an input that does not end with a newline.
    None,
}

impl BiasLineTerminator {
    fn as_str(self) -> &'static str {
        match self {
            Self::Lf => "\n",
            Self::CrLf => "\r\n",
            Self::None => "",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// What a source line is to the reader.
pub enum BiasLineRole {
    /// The `%=BIA` header line.
    Header,
    /// The `%=ENDBIA` footer line.
    Footer,
    /// A `*` comment line.
    Comment,
    /// A line holding only blanks.
    Blank,
    /// A `+` block start line.
    BlockStart,
    /// A `-` block end line.
    BlockEnd,
    /// A `FILE/REFERENCE` row; the index is into
    /// [`BiasSetHeader::file_reference`].
    FileReference(usize),
    /// A `BIAS/DESCRIPTION` row; the index is into
    /// [`BiasSetHeader::description`].
    Description(usize),
    /// A row read as a record; the index is into [`BiasSet::records`].
    Record(usize),
    /// A solution or DCB data row that was not read as a record; the skip
    /// diagnostic at this line gives the reason.
    Skipped,
    /// A data line of a block this reader does not model, such as
    /// `FILE/COMMENT` or `BIAS/RECEIVER_INFORMATION`, kept as written.
    BlockBody,
    /// A line outside every block, or after the footer; a
    /// [`BiasDeparture`] reports it.
    Outside,
    /// A CODE DCB title, column heading or prose line.
    Text,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// One physical line of the input, kept exactly.
pub struct BiasSourceLine {
    /// One-based line number.
    pub number: usize,
    /// Line text without its terminator. A line that is not valid UTF-8 is
    /// decoded with replacement characters here and kept exactly in
    /// [`BiasSourceLine::bytes`].
    pub text: String,
    /// The exact bytes of a line that is not valid UTF-8; `None` when
    /// [`BiasSourceLine::text`] is exact.
    pub bytes: Option<Vec<u8>>,
    /// The line's ending in the input.
    pub terminator: BiasLineTerminator,
    /// How the reader used the line.
    pub role: BiasLineRole,
}

impl BiasSourceLine {
    /// The exact bytes of the line, without its terminator. Fixed columns
    /// are byte offsets into these bytes.
    pub fn raw(&self) -> &[u8] {
        self.bytes.as_deref().unwrap_or(self.text.as_bytes())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Count of every physical input line by what the reader made of it. The
/// categories are disjoint and add up to [`BiasLineCounts::lines`].
pub struct BiasLineCounts {
    /// Every physical line.
    pub lines: usize,
    /// Header and footer lines.
    pub header_footer: usize,
    /// Comment lines.
    pub comments: usize,
    /// Blank lines.
    pub blank: usize,
    /// Block start and end lines.
    pub block_delimiters: usize,
    /// `FILE/REFERENCE` and `BIAS/DESCRIPTION` rows.
    pub info_rows: usize,
    /// Rows read as records.
    pub records: usize,
    /// Solution or DCB data rows not read as records.
    pub skipped: usize,
    /// Data lines of blocks the reader keeps without modelling.
    pub block_body: usize,
    /// Lines outside every block, and DCB title, heading and prose lines.
    pub other: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// How the Bias-SINEX reader treats a file that departs from Bias-SINEX
/// 1.00.
pub enum BiasReadPolicy {
    /// Refuse the file with [`BiasError::Departure`] naming the first
    /// departure, or [`BiasError::UnsupportedVersion`] for a version other
    /// than `1.00`.
    #[default]
    Strict,
    /// Read the file and report every departure as a
    /// [`BiasNotice::Departure`].
    Lenient,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
/// A departure from something Bias-SINEX 1.00 states explicitly.
pub enum BiasDeparture {
    /// The header line is not the 74-column line section 4.1 defines.
    HeaderLayout {
        /// Which part of the layout is missing or malformed.
        reason: &'static str,
    },
    /// The header states a format version other than `1.00`, the only one
    /// section 4.1 defines.
    OtherVersion {
        /// The version token as written.
        version: String,
    },
    /// The file has no `%=ENDBIA` footer line.
    MissingFooter,
    /// A line, blank or not, follows the footer, which section 4.1 makes the
    /// last line.
    ContentAfterFooter {
        /// One-based line number.
        line: usize,
    },
    /// A `%` line other than the header and footer.
    UnexpectedControlLine {
        /// One-based line number.
        line: usize,
    },
    /// A block is still open at the footer or at the end of the input.
    UnclosedBlock {
        /// Block name.
        name: String,
        /// One-based line of the block start.
        line: usize,
    },
    /// A block end line with no open block.
    UnopenedBlockEnd {
        /// Block name on the end line.
        name: String,
        /// One-based line number.
        line: usize,
    },
    /// A block end line naming a block other than the open one.
    MismatchedBlockEnd {
        /// Open block.
        open: String,
        /// Block named by the end line.
        close: String,
        /// One-based line number.
        line: usize,
    },
    /// A block start line while another block is open.
    NestedBlock {
        /// Open block.
        open: String,
        /// Block started inside it.
        inner: String,
        /// One-based line number.
        line: usize,
    },
    /// A block section 2.1 marks mandatory is absent.
    MissingBlock {
        /// Block name.
        name: &'static str,
    },
    /// A block section 2.1 does not allow. Its lines are kept as written.
    UnknownBlock {
        /// Block name.
        name: String,
        /// One-based line of the block start.
        line: usize,
    },
    /// Text after the block name on a block start line, such as the count
    /// earlier versions of this library wrote after `+BIAS/SOLUTION`.
    BlockStartSuffix {
        /// One-based line number.
        line: usize,
    },
    /// A data line outside every block.
    DataOutsideBlock {
        /// One-based line number.
        line: usize,
    },
    /// A declaration section 4.6 marks mandatory, `BIAS_MODE` or
    /// `TIME_SYSTEM`, is absent. A missing `TIME_SYSTEM` leaves the product
    /// without a time scale.
    MissingDeclaration {
        /// Keyword.
        keyword: &'static str,
    },
    /// A `BIAS_MODE` value other than `ABSOLUTE` and `RELATIVE`.
    UnsupportedBiasMode {
        /// One-based line number.
        line: usize,
        /// The value as written.
        label: String,
    },
    /// A `TIME_SYSTEM` label that is neither a RINEX GNSS system flag nor
    /// `UTC` or `TAI` (section 4.6). A lenient read maps a label naming a
    /// scale unambiguously, such as `GPS` or `TCG`, to that scale, and leaves
    /// the product without a time scale for any other label.
    NonStandardTimeSystem {
        /// One-based line number.
        line: usize,
        /// The label as written.
        label: String,
    },
    /// The header line's bias mode does not match `BIAS_MODE` (section 4.1).
    HeaderModeMismatch {
        /// Header mode token.
        header: String,
        /// Mode `BIAS_MODE` declares.
        description: BiasMode,
    },
    /// A generated CODE DCB title (`# DCB <pair> <YYYY-MM> <label>`) whose
    /// label names no time scale this reader knows.
    UnknownDcbTimeSystem {
        /// One-based line number.
        line: usize,
        /// The label as written.
        label: String,
    },
    /// The header line's number of estimates differs from the number of
    /// solution data rows (section 4.1).
    EstimateCountMismatch {
        /// Declared count.
        declared: u64,
        /// Solution data rows in the file, read or not.
        solution_rows: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
/// A non-fatal finding about a bias product.
pub enum BiasNotice {
    /// A departure accepted by a lenient read.
    Departure(BiasDeparture),
    /// A line that is not valid UTF-8; its bytes are kept exactly, and its
    /// fields are read at their byte columns.
    InvalidUtf8 {
        /// One-based line number.
        line: usize,
    },
    /// A declaration repeated with the same meaning.
    RepeatedDeclaration {
        /// One-based line of the repeat.
        line: usize,
        /// Keyword.
        keyword: &'static str,
    },
    /// A declaration repeated with a different meaning. The declared
    /// property is left undetermined.
    ConflictingDeclaration {
        /// One-based line of the conflicting row.
        line: usize,
        /// Keyword.
        keyword: &'static str,
    },
    /// Two records for one target and observable have overlapping validity
    /// intervals. Indices are into [`BiasSet::records`].
    Overlap {
        /// Record with the earlier start.
        first: usize,
        /// Record with the later or equal start.
        second: usize,
    },
    /// A CODE DCB title states no time system and no options were given, so
    /// the product is taken to be in GPS time.
    DcbTimeSystemAssumed,
    /// A generated CODE DCB title names its time system by a constellation
    /// name, such as `GPS` or `GLO`, read as the scale the lenient
    /// Bias-SINEX reader maps it to.
    DcbTimeSystemAlias {
        /// One-based line number.
        line: usize,
        /// The label as written.
        label: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
/// Outcome of a bias lookup.
pub enum BiasLookup {
    /// The bias value, in the unit the query names.
    Available {
        /// The value.
        value: f64,
        /// Records the value comes from, as indices into
        /// [`BiasSet::records`]: the selected record, the records of a DSB
        /// path, or every record a combination uses.
        records: Vec<usize>,
        /// Records that also cover the query epoch but start earlier than a
        /// selected record, which overrides them.
        overridden: Vec<usize>,
    },
    /// No record covers the query.
    Absent,
    /// The query epoch is not on the product's time scale, or the product
    /// declares no usable time scale. No conversion is attempted.
    UnsupportedScale {
        /// Product time scale, `None` when the product has none.
        product: Option<TimeScale>,
        /// Query time scale.
        query: TimeScale,
    },
    /// Several records apply at the query epoch and give different values:
    /// records tied on the latest start, stations matching a queried
    /// identifier, or DSB paths of equal length. Indices are into
    /// [`BiasSet::records`].
    Ambiguous {
        /// The conflicting records.
        records: Vec<usize>,
    },
    /// A phase bias stated in nanoseconds was requested in cycles without a
    /// carrier frequency.
    CarrierFrequencyRequired {
        /// Index of the record stated in nanoseconds.
        record: usize,
    },
    /// A carrier frequency given or needed for a conversion is not finite
    /// and positive, or two carriers of an ionosphere-free pair are equal.
    InvalidCarrierFrequency,
    /// No carrier frequency is known for an observable, such as a GLONASS
    /// FDMA signal queried without its frequency channel.
    CarrierFrequencyUnknown {
        /// The observable code.
        observable: String,
    },
    /// A sloped record whose start and end are both undefined, so section
    /// 5.1 gives no epoch for its value.
    UndefinedSlopeReference {
        /// Index of the sloped record.
        record: usize,
    },
    /// The query epoch cannot be converted to a split Julian date.
    InvalidEpoch,
}

impl BiasLookup {
    /// Returns the value when it is available.
    pub fn value(&self) -> Option<f64> {
        match self {
            Self::Available { value, .. } => Some(*value),
            _ => None,
        }
    }

    /// Returns `true` when a value is available.
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available { .. })
    }

    /// An available value that no record supplies, such as the exact zero
    /// between an observable and itself.
    fn exact(value: f64) -> Self {
        Self::Available {
            value,
            records: Vec::new(),
            overridden: Vec::new(),
        }
    }

    fn map(self, f: impl FnOnce(f64) -> f64) -> Self {
        match self {
            Self::Available {
                value,
                records,
                overridden,
            } => Self::Available {
                value: f(value),
                records,
                overridden,
            },
            other => other,
        }
    }

    /// Combines two available values with `f`, uniting the records they come
    /// from. Any other status of `self`, then of `other`, is returned as it is.
    fn combine(self, other: Self, f: impl FnOnce(f64, f64) -> f64) -> Self {
        match (self, other) {
            (
                Self::Available {
                    value: a,
                    records: mut records_a,
                    overridden: mut overridden_a,
                },
                Self::Available {
                    value: b,
                    records: records_b,
                    overridden: overridden_b,
                },
            ) => {
                records_a.extend(records_b);
                records_a.sort_unstable();
                records_a.dedup();
                overridden_a.extend(overridden_b);
                overridden_a.sort_unstable();
                overridden_a.dedup();
                Self::Available {
                    value: f(a, b),
                    records: records_a,
                    overridden: overridden_a,
                }
            }
            (Self::Available { .. }, other) => other,
            (status, _) => status,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BiasSourceFormat {
    BiasSinex,
    CodeDcb,
}

#[derive(Debug, Clone, PartialEq)]
/// A parsed product retains its records, lookup metadata, and diagnostics so
/// code and phase bias queries can apply the product's validity rules.
///
/// Records keep file order, duplicates included. The product keeps every
/// source line with the role the reader gave it, the typed header rows, and
/// the non-fatal notices found while reading.
pub struct BiasSet {
    records: Vec<BiasRecord>,
    index: BTreeMap<(BiasTargetKey, String), Vec<usize>>,
    mode: BiasMode,
    time_scale: Option<TimeScale>,
    clock_reference: ClockReferenceObservables,
    header: BiasSetHeader,
    lines: Vec<BiasSourceLine>,
    source_format: Option<BiasSourceFormat>,
    diagnostics: Diagnostics,
    notices: Vec<BiasNotice>,
    skipped_records: usize,
}

#[derive(Debug, Clone, PartialEq)]
/// Legacy DCB parsing and writing use these fields to identify the observable
/// pair, product interval, time scale, and optional receiver constellation.
///
/// The pair and month control observable mapping and the generated validity
/// interval, while `receiver_system` supplies missing system columns.
#[non_exhaustive]
pub struct CodeDcbOptions {
    /// Legacy observable labels, such as `P1` and `C1`, to map per system.
    pub pair: (String, String),
    /// Calendar year used to construct the first day of the product interval.
    pub year: i32,
    /// Calendar month checked as 1 through 12 and used to find the next-month
    /// exclusive interval end.
    pub month: u8,
    /// Scale copied to the parsed [`BiasSet`] and emitted in the DCB title.
    pub time_scale: TimeScale,
    /// Optional system used to recognize and target receiver rows whose first
    /// column does not contain a system letter.
    pub receiver_system: Option<GnssSystem>,
}

impl CodeDcbOptions {
    /// Build DCB options from the required product identity fields.
    ///
    /// `receiver_system` defaults to `None`; assign it when the receiver rows
    /// use a system-less legacy format.
    #[must_use]
    pub fn new(pair: (String, String), year: i32, month: u8, time_scale: TimeScale) -> Self {
        Self {
            pair,
            year,
            month,
            time_scale,
            receiver_system: None,
        }
    }

    /// Set the optional receiver constellation.
    #[must_use]
    pub const fn with_receiver_system(mut self, receiver_system: GnssSystem) -> Self {
        self.receiver_system = Some(receiver_system);
        self
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
/// These variants identify the parser, time conversion, query metadata, or
/// writer condition that stopped a bias operation.
pub enum BiasError {
    #[error("invalid bias input {field}: {reason}")]
    /// A parser found a missing or invalid field, or a writer received an
    /// unsupported set component.
    InvalidInput {
        /// Name of the input or output component that failed, such as `month`
        /// or `bias target`.
        field: &'static str,
        /// Fixed explanation selected by the validation branch, such as
        /// `missing` or `out of range`.
        reason: &'static str,
    },
    #[error("invalid bias epoch")]
    /// A calendar epoch or split Julian date could not be validated.
    InvalidEpoch,
    #[error("unknown RINEX observable code {code}")]
    /// Represents an unknown RINEX observable code; the current frequency
    /// lookup returns `None` instead of constructing this variant.
    UnknownObservable {
        /// Unrecognized RINEX observable code.
        code: String,
    },
    #[error("unsupported Bias-SINEX version {version}")]
    /// The Bias-SINEX version token is not `1.00`, the only version
    /// Bias-SINEX 1.00 defines.
    UnsupportedVersion {
        /// Complete version token as written.
        version: String,
    },
    #[error("missing DCB pair or product month")]
    /// Neither caller options nor the first twelve title lines supplied DCB
    /// pair, year, and month metadata.
    MissingDcbMetadata,
    #[error("missing satellite clock reference observables")]
    /// PPP lookup found no clock-reference pair for the requested satellite's
    /// system.
    MissingClockReference,
    #[error("bias writer missing required metadata {field}")]
    /// A writer was called without metadata its output format requires.
    MissingWriterMetadata {
        /// Name of the required set component that was absent.
        field: &'static str,
    },
    #[error("input is not UTF-8")]
    /// Indicates a UTF-8 decoding failure; the byte parsers keep such lines
    /// exactly instead of constructing this variant.
    Utf8,
    #[error("Bias-SINEX departure: {departure:?}")]
    /// A strict read found a departure from Bias-SINEX 1.00.
    Departure {
        /// The first departure found.
        departure: BiasDeparture,
    },
    #[error("line {line} is not UTF-8; write the product as bytes")]
    /// A string writer met a source line that is not valid UTF-8. The byte
    /// writer restates it exactly.
    InvalidUtf8Line {
        /// One-based line number.
        line: usize,
    },
    #[error("time scale {scale:?} cannot be written in this format or under this metadata")]
    /// A writer has no time-system label that reads back as the set's scale,
    /// or the set's scale is not the one its DCB metadata states.
    UnsupportedTimeSystem {
        /// The set's time scale, `None` when the set has none.
        scale: Option<TimeScale>,
    },
    #[error("record {record} does not match the DCB metadata: {field}")]
    /// A record cannot be written under the product's DCB metadata without
    /// changing its meaning.
    DcbRecordMismatch {
        /// Index into [`BiasSet::records`].
        record: usize,
        /// The field that differs.
        field: &'static str,
    },
}

/// The unit and the stated value and slope of each record an available value
/// sums, when every such record keeps its source text.
type StatedTerms = Option<(BiasUnit, Vec<StatedAffine>)>;

/// A lookup's outcome together with what an available value states
/// exactly: the unit and the stated value and slope of each record it sums,
/// when every such record keeps its source text.
struct Answer {
    lookup: BiasLookup,
    stated: StatedTerms,
}

impl Answer {
    fn unstated(lookup: BiasLookup) -> Self {
        Self {
            lookup,
            stated: None,
        }
    }
}

/// Whether two available answers give the same bias: as exact decimals when
/// both state their terms in one unit, and otherwise by `==` on their values.
fn answers_agree(
    value_a: f64,
    stated_a: &StatedTerms,
    value_b: f64,
    stated_b: &StatedTerms,
) -> bool {
    if let (Some((unit_a, terms_a)), Some((unit_b, terms_b))) = (stated_a, stated_b) {
        if unit_a == unit_b {
            if let Some(same) = affine_sums_equal(terms_a, terms_b) {
                return same;
            }
        }
    }
    value_a == value_b
}

/// Whether two records give the same bias, each as `(unit, stated, value)`:
/// their stated values and slopes, as exact decimals, when both are known
/// and they share a unit, and otherwise their values with `==`, so `0.0` and
/// `-0.0` agree. The stated terms and values must be taken in the same
/// direction: a DSB hop against its record passes its negated terms. This is
/// the one rule for tied OSB records and parallel DSB records.
fn records_agree(
    a: (BiasUnit, Option<&StatedAffine>, f64),
    b: (BiasUnit, Option<&StatedAffine>, f64),
) -> bool {
    let ((unit_a, stated_a, value_a), (unit_b, stated_b, value_b)) = (a, b);
    if unit_a == unit_b {
        if let (Some(x), Some(y)) = (stated_a, stated_b) {
            if let Some(same) = affine_sums_equal(std::slice::from_ref(x), std::slice::from_ref(y))
            {
                return same;
            }
        }
    }
    value_a == value_b
}

/// Combines answers that each could answer the same query, such as the
/// stations a queried identifier corresponds to. One available value, or
/// several that agree as [`answers_agree`] compares them, is the answer;
/// values that differ are ambiguous. Any status other than a value, an
/// absence or an ambiguity is returned as it is.
fn combine_alternatives(answers: impl IntoIterator<Item = Answer>) -> BiasLookup {
    let mut held: Option<(f64, StatedTerms)> = None;
    let mut records = Vec::new();
    let mut overridden = Vec::new();
    let mut conflict = false;
    for answer in answers {
        match answer.lookup {
            BiasLookup::Absent => {}
            BiasLookup::Available {
                value,
                records: found_records,
                overridden: found_overridden,
            } => {
                records.extend(found_records);
                overridden.extend(found_overridden);
                let agrees = held.as_ref().map(|(held_value, held_stated)| {
                    answers_agree(*held_value, held_stated, value, &answer.stated)
                });
                match agrees {
                    None => held = Some((value, answer.stated)),
                    Some(true) => {}
                    Some(false) => conflict = true,
                }
            }
            BiasLookup::Ambiguous { records: tied } => {
                records.extend(tied);
                conflict = true;
            }
            other => return other,
        }
    }
    records.sort_unstable();
    records.dedup();
    overridden.sort_unstable();
    overridden.dedup();
    if conflict {
        return BiasLookup::Ambiguous { records };
    }
    match held {
        Some((value, _)) => BiasLookup::Available {
            value,
            records,
            overridden,
        },
        None => BiasLookup::Absent,
    }
}

impl BiasSet {
    /// Parses a Bias-SINEX byte stream under [`BiasReadPolicy::Strict`].
    ///
    /// Every line is kept, in order, with the role the reader gave it; lines
    /// that are not valid UTF-8 keep their exact bytes and are reported. Valid
    /// solution records are retained in the returned [`Parsed`] value, while
    /// malformed records, repeated declarations and overlaps are reported
    /// through [`BiasSet::diagnostics`] and [`BiasSet::notices`]. A departure
    /// from something Bias-SINEX 1.00 states explicitly (a [`BiasDeparture`])
    /// returns [`BiasError::Departure`], and a version other than `1.00`
    /// returns [`BiasError::UnsupportedVersion`];
    /// [`BiasSet::parse_bias_sinex_with_policy`] can read such a file instead.
    pub fn parse_bias_sinex(input: &[u8]) -> Result<Parsed<BiasSet>, BiasError> {
        Self::parse_bias_sinex_with_policy(input, BiasReadPolicy::Strict)
    }

    /// Parses a Bias-SINEX byte stream under `policy`.
    ///
    /// Under [`BiasReadPolicy::Lenient`] every departure, a version other than
    /// `1.00` included, is reported as a [`BiasNotice::Departure`] and the
    /// file is read. Either policy refuses input without a `%=BIA` header line
    /// and version token.
    pub fn parse_bias_sinex_with_policy(
        input: &[u8],
        policy: BiasReadPolicy,
    ) -> Result<Parsed<BiasSet>, BiasError> {
        parse_bias_sinex_input(input, policy)
    }

    /// Parses a legacy code DCB byte stream.
    ///
    /// Metadata is taken from `options` or the product title, validated, and
    /// used to convert accepted nanosecond rows into code DSB records in
    /// seconds. Rows that cannot be represented are retained as typed skips in
    /// the returned [`Parsed`] value, and every line is kept with its role.
    pub fn parse_code_dcb(
        input: &[u8],
        options: Option<CodeDcbOptions>,
    ) -> Result<Parsed<BiasSet>, BiasError> {
        Self::parse_code_dcb_with_policy(input, options, BiasReadPolicy::Strict)
    }

    /// Parses a legacy code DCB byte stream under `policy`.
    ///
    /// A generated title (`# DCB <pair> <YYYY-MM> <label>`) whose label names
    /// no time scale this reader knows, with no `options` to decide it,
    /// returns [`BiasError::Departure`] under [`BiasReadPolicy::Strict`].
    /// Under [`BiasReadPolicy::Lenient`] the rows are read, the product is
    /// left without a time scale or DCB metadata, and the departure is
    /// reported. With `options`, the options decide the scale under either
    /// policy and the unknown label is reported.
    pub fn parse_code_dcb_with_policy(
        input: &[u8],
        options: Option<CodeDcbOptions>,
        policy: BiasReadPolicy,
    ) -> Result<Parsed<BiasSet>, BiasError> {
        parse_code_dcb_input(input, options, policy)
    }

    /// Returns parser and indexing diagnostics retained by this set, including
    /// skips and overlap warnings added while building its lookup index.
    pub fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }

    /// Returns the non-fatal findings made while reading the product.
    pub fn notices(&self) -> &[BiasNotice] {
        &self.notices
    }

    /// Returns the number of entries in [`BiasSet::diagnostics`] that were
    /// skipped during parsing.
    pub fn skipped_records(&self) -> usize {
        self.skipped_records
    }

    /// Returns every physical source line in input order. Empty for a set not
    /// read from text.
    pub fn source_lines(&self) -> &[BiasSourceLine] {
        &self.lines
    }

    /// Returns the source lines of data rows that were not read as records,
    /// with their exact text.
    pub fn skipped_lines(&self) -> impl Iterator<Item = &BiasSourceLine> {
        self.lines
            .iter()
            .filter(|line| line.role == BiasLineRole::Skipped)
    }

    /// Counts every physical source line by role.
    pub fn line_counts(&self) -> BiasLineCounts {
        let mut counts = BiasLineCounts {
            lines: self.lines.len(),
            ..BiasLineCounts::default()
        };
        for line in &self.lines {
            let slot = match line.role {
                BiasLineRole::Header | BiasLineRole::Footer => &mut counts.header_footer,
                BiasLineRole::Comment => &mut counts.comments,
                BiasLineRole::Blank => &mut counts.blank,
                BiasLineRole::BlockStart | BiasLineRole::BlockEnd => &mut counts.block_delimiters,
                BiasLineRole::FileReference(_) | BiasLineRole::Description(_) => {
                    &mut counts.info_rows
                }
                BiasLineRole::Record(_) => &mut counts.records,
                BiasLineRole::Skipped => &mut counts.skipped,
                BiasLineRole::BlockBody => &mut counts.block_body,
                BiasLineRole::Outside | BiasLineRole::Text => &mut counts.other,
            };
            *slot += 1;
        }
        counts
    }

    /// Returns the header metadata.
    pub fn header(&self) -> &BiasSetHeader {
        &self.header
    }

    /// Returns the bias mode `BIAS_MODE` declares, or `Relative` for a DCB
    /// set.
    pub fn mode(&self) -> BiasMode {
        self.mode
    }

    /// Returns the product time scale: the one `TIME_SYSTEM` declares, or the
    /// DCB metadata scale. `None` when the declaration is missing,
    /// unsupported or conflicting; lookups then return
    /// [`BiasLookup::UnsupportedScale`].
    pub fn time_scale(&self) -> Option<TimeScale> {
        self.time_scale
    }

    /// Returns the first `TIME_SYSTEM` label exactly as the file writes it.
    pub fn time_system_label(&self) -> Option<&str> {
        self.header.description_value("TIME_SYSTEM")
    }

    /// Returns the satellite clock-reference observables per system.
    pub fn clock_reference(&self) -> &ClockReferenceObservables {
        &self.clock_reference
    }

    /// Returns the parsed records in their stored vector order, which is the
    /// input order for records accepted by either public parser.
    pub fn records(&self) -> &[BiasRecord] {
        &self.records
    }

    /// Gives a set that was not read from Bias-SINEX a header line, so
    /// [`write_bias_sinex`] can state it.
    ///
    /// A set read from Bias-SINEX restates its own header line, so this
    /// returns [`BiasError::InvalidInput`] for it. The header is checked
    /// against the section 4.1 layout here, so a header that cannot be
    /// written is refused now.
    pub fn set_sinex_header(&mut self, header: BiasSinexHeader) -> Result<(), BiasError> {
        if self.source_format == Some(BiasSourceFormat::BiasSinex) {
            return Err(BiasError::InvalidInput {
                field: "sinex header",
                reason: "a Bias-SINEX product restates its own header line",
            });
        }
        format_sinex_header_line(&header, 'A', 0)?;
        self.header.sinex = Some(header);
        Ok(())
    }

    /// Gives the set DCB metadata, so [`write_code_dcb`] can state its
    /// records.
    ///
    /// Every record must be a code DSB between the observables `meta.pair`
    /// maps to for its system, valid over exactly the metadata month, with no
    /// SVN, slope or slope uncertainty; otherwise the first record that
    /// differs is named in [`BiasError::DcbRecordMismatch`]. A set whose time
    /// scale is not `meta.time_scale` returns
    /// [`BiasError::UnsupportedTimeSystem`]. A set read from CODE DCB restates
    /// its own title, so for it any metadata other than its current metadata
    /// returns [`BiasError::InvalidInput`]. On any error nothing changes.
    pub fn set_dcb_meta(&mut self, meta: CodeDcbOptions) -> Result<(), BiasError> {
        if self.source_format == Some(BiasSourceFormat::CodeDcb)
            && self.header.dcb_meta.as_ref() != Some(&meta)
        {
            return Err(BiasError::InvalidInput {
                field: "dcb_meta",
                reason: "a CODE DCB product restates its own title",
            });
        }
        validate_dcb_options(&meta)?;
        check_dcb_records(self, &meta)?;
        self.header.dcb_meta = Some(meta);
        Ok(())
    }

    /// Returns a covering code OSB in seconds for `sat` and `obs`.
    ///
    /// Lookup checks the satellite target before the system target and
    /// considers only code observables. Among covering records the one with
    /// the latest start applies, and the records it overrides are named in
    /// the result. An optional record slope is evaluated at `epoch` from the
    /// section 5.1 reference epoch.
    pub fn code_osb_seconds(&self, sat: GnssSatelliteId, obs: &str, epoch: Instant) -> BiasLookup {
        self.osb_for_target_chain(sat, obs, epoch, BiasObservableFamily::Code, None)
    }

    /// Returns a covering phase OSB in cycles for `sat` and `obs`.
    ///
    /// Lookup checks the satellite target before the system target and
    /// considers only phase observables. A record stated in cycles is
    /// returned as stated. A record stated in nanoseconds is converted with
    /// `carrier_hz`, the carrier frequency of `obs` for this satellite (for a
    /// GLONASS FDMA signal, the frequency of its channel); without it the
    /// lookup returns [`BiasLookup::CarrierFrequencyRequired`].
    pub fn phase_osb_cycles(
        &self,
        sat: GnssSatelliteId,
        obs: &str,
        epoch: Instant,
        carrier_hz: Option<f64>,
    ) -> BiasLookup {
        self.osb_for_target_chain(
            sat,
            obs,
            epoch,
            BiasObservableFamily::Phase,
            Some(carrier_hz),
        )
    }

    /// Resolves a covering code DSB in seconds for a satellite or its system.
    ///
    /// The satellite key is tried before the system key. Reversing the
    /// observable arguments reverses the resolved sign, and multi-hop routes
    /// are used when the covering DSB records connect the observables only
    /// through others; the fewest hops apply. Parallel records between the
    /// same observables that differ in value make the lookup ambiguous.
    /// Routes through different observables must agree: exactly, as
    /// functions of time built from the stated values and slopes, when the
    /// rows' text is kept, and otherwise within the rounding bound of their
    /// sums. When all agree, the route first in observable order gives the
    /// value; otherwise the lookup is ambiguous and names every record on the
    /// shortest routes.
    pub fn code_dsb_seconds(
        &self,
        sat: GnssSatelliteId,
        obs1: &str,
        obs2: &str,
        epoch: Instant,
    ) -> BiasLookup {
        if obs1 == obs2 {
            return BiasLookup::exact(0.0);
        }
        let query = match self.query_seconds(epoch) {
            Ok(query) => query,
            Err(status) => return status,
        };
        for key in [
            BiasTargetKey::satellite(sat),
            BiasTargetKey::system(sat.system),
        ] {
            let answer = self.dsb_for_key(&key, obs1, obs2, &query);
            if answer.lookup != BiasLookup::Absent {
                return answer.lookup;
            }
        }
        BiasLookup::Absent
    }

    /// Returns a covering receiver code OSB in seconds.
    ///
    /// The station is matched exactly, ignoring case. When the product names
    /// no such station, a bare legacy four-character code and the
    /// nine-character identifiers or code-plus-DOMES names carrying that code
    /// answer for one another, in either direction, as do the spellings of
    /// one code and DOMES number with and without a blank. A nine-character
    /// identifier never answers for another one or for a code-plus-DOMES
    /// name. Different values among the stations that answer are ambiguous.
    pub fn receiver_code_osb_seconds(
        &self,
        system: GnssSystem,
        station: &str,
        obs: &str,
        epoch: Instant,
    ) -> BiasLookup {
        let query = match self.query_seconds(epoch) {
            Ok(query) => query,
            Err(status) => return status,
        };
        let keys = self.station_keys(&BiasTargetKey::receiver(system, station));
        combine_alternatives(
            keys.iter()
                .map(|key| self.osb_for_key(key, obs, &query, BiasObservableFamily::Code, None)),
        )
    }

    /// Resolves a covering receiver code DSB in seconds, matching the station
    /// as [`BiasSet::receiver_code_osb_seconds`] does and choosing among
    /// routes, or reporting them ambiguous, as [`BiasSet::code_dsb_seconds`]
    /// does.
    pub fn receiver_code_dsb_seconds(
        &self,
        system: GnssSystem,
        station: &str,
        obs1: &str,
        obs2: &str,
        epoch: Instant,
    ) -> BiasLookup {
        if obs1 == obs2 {
            return BiasLookup::exact(0.0);
        }
        let query = match self.query_seconds(epoch) {
            Ok(query) => query,
            Err(status) => return status,
        };
        let keys = self.station_keys(&BiasTargetKey::receiver(system, station));
        combine_alternatives(
            keys.iter()
                .map(|key| self.dsb_for_key(key, obs1, obs2, &query)),
        )
    }

    /// Returns a covering satellite-receiver code OSB in seconds, matching the
    /// station as [`BiasSet::receiver_code_osb_seconds`] does.
    pub fn sat_receiver_code_osb_seconds(
        &self,
        sat: GnssSatelliteId,
        station: &str,
        obs: &str,
        epoch: Instant,
    ) -> BiasLookup {
        let query = match self.query_seconds(epoch) {
            Ok(query) => query,
            Err(status) => return status,
        };
        let keys = self.station_keys(&BiasTargetKey::satellite_receiver(sat, station));
        combine_alternatives(
            keys.iter()
                .map(|key| self.osb_for_key(key, obs, &query, BiasObservableFamily::Code, None)),
        )
    }

    /// Resolves a covering satellite-receiver code DSB in seconds, matching
    /// the station as [`BiasSet::receiver_code_osb_seconds`] does and choosing
    /// among routes as [`BiasSet::code_dsb_seconds`] does.
    pub fn sat_receiver_code_dsb_seconds(
        &self,
        sat: GnssSatelliteId,
        station: &str,
        obs1: &str,
        obs2: &str,
        epoch: Instant,
    ) -> BiasLookup {
        if obs1 == obs2 {
            return BiasLookup::exact(0.0);
        }
        let query = match self.query_seconds(epoch) {
            Ok(query) => query,
            Err(status) => return status,
        };
        let keys = self.station_keys(&BiasTargetKey::satellite_receiver(sat, station));
        combine_alternatives(
            keys.iter()
                .map(|key| self.dsb_for_key(key, obs1, obs2, &query)),
        )
    }

    /// Computes the code-bias model relative to a satellite-clock reference in meters.
    ///
    /// Matching observable pairs return exact zero. When both ionosphere-free
    /// OSB combinations are available, the model is their difference times
    /// [`C_M_S`]; when either is absent, the corresponding code DSBs are
    /// combined with the ionosphere-free coefficients and converted to meters.
    /// Any other status of an OSB lookup, such as an ambiguity, is returned
    /// as it is rather than replaced by the DSB path. An available value
    /// names every record it uses and every record those override.
    pub fn code_bias_model_m(
        &self,
        sat: GnssSatelliteId,
        used_observables: (&str, &str),
        used_frequencies_hz: (f64, f64),
        glonass_channel: Option<i8>,
        clock_reference: (&str, &str),
        epoch: Instant,
    ) -> BiasLookup {
        if used_observables == clock_reference {
            return BiasLookup::exact(0.0);
        }
        let used_if = self.if_bias_seconds(sat, used_observables, used_frequencies_hz, epoch);
        let Some(ref_freq1) = rinex_frequency(sat, clock_reference.0, glonass_channel) else {
            return BiasLookup::CarrierFrequencyUnknown {
                observable: clock_reference.0.to_string(),
            };
        };
        let Some(ref_freq2) = rinex_frequency(sat, clock_reference.1, glonass_channel) else {
            return BiasLookup::CarrierFrequencyUnknown {
                observable: clock_reference.1.to_string(),
            };
        };
        let ref_if = self.if_bias_seconds(sat, clock_reference, (ref_freq1, ref_freq2), epoch);
        if used_if.is_available() && ref_if.is_available() {
            return used_if.combine(ref_if, |used, reference| (used - reference) * C_M_S);
        }
        for status in [&used_if, &ref_if] {
            if !matches!(status, BiasLookup::Absent | BiasLookup::Available { .. }) {
                return status.clone();
            }
        }
        self.relative_code_bias_seconds(
            sat,
            used_observables,
            used_frequencies_hz,
            clock_reference,
            epoch,
        )
        .map(|seconds| seconds * C_M_S)
    }

    fn new(
        records: Vec<BiasRecord>,
        mode: BiasMode,
        time_scale: Option<TimeScale>,
        clock_reference: ClockReferenceObservables,
        header: BiasSetHeader,
        mut diagnostics: Diagnostics,
    ) -> Self {
        let mut set = Self {
            records,
            index: BTreeMap::new(),
            mode,
            time_scale,
            clock_reference,
            header,
            lines: Vec::new(),
            source_format: None,
            diagnostics: Diagnostics::new(),
            notices: Vec::new(),
            skipped_records: 0,
        };
        set.rebuild_index(&mut diagnostics);
        set.skipped_records = diagnostics.skips.len();
        set.diagnostics = diagnostics;
        set
    }

    fn rebuild_index(&mut self, diagnostics: &mut Diagnostics) {
        let mut index: BTreeMap<(BiasTargetKey, String), Vec<usize>> = BTreeMap::new();
        for (record_index, record) in self.records.iter().enumerate() {
            let key = BiasTargetKey::from(&record.target);
            index
                .entry((key, obs_key(record)))
                .or_default()
                .push(record_index);
        }
        for indices in index.values_mut() {
            indices.sort_by(|a, b| compare_record_start(&self.records[*a], &self.records[*b]));
            // Every pair, not only neighbours in start order: a long record
            // can overlap several later ones that do not overlap each other.
            for (position, &first) in indices.iter().enumerate() {
                for &second in &indices[position + 1..] {
                    // Later records start no earlier, so once one starts at
                    // or after this record's end, none after it overlaps.
                    if let (Some(end), Some(start)) = (
                        instant_of(self.records[first].valid_until),
                        instant_of(self.records[second].valid_from),
                    ) {
                        if start >= end {
                            break;
                        }
                    }
                    if intervals_overlap(&self.records[first], &self.records[second]) {
                        diagnostics.push_warning(Warning {
                            at: RecordRef::at_record(second),
                            kind: WarningKind::Overlap,
                        });
                        self.notices.push(BiasNotice::Overlap { first, second });
                    }
                }
            }
        }
        self.index = index;
    }

    /// Checks the query scale against the product scale and takes the
    /// epoch's exact seconds since J2000 for coverage tests and slopes.
    fn query_seconds(&self, epoch: Instant) -> Result<ExactSeconds, BiasLookup> {
        match self.time_scale {
            Some(scale) if scale == epoch.scale => {}
            product => {
                return Err(BiasLookup::UnsupportedScale {
                    product,
                    query: epoch.scale,
                })
            }
        }
        instant_exact_seconds(epoch).ok_or(BiasLookup::InvalidEpoch)
    }

    fn osb_for_target_chain(
        &self,
        sat: GnssSatelliteId,
        obs: &str,
        epoch: Instant,
        family: BiasObservableFamily,
        carrier_hz: Option<Option<f64>>,
    ) -> BiasLookup {
        let query = match self.query_seconds(epoch) {
            Ok(query) => query,
            Err(status) => return status,
        };
        for key in [
            BiasTargetKey::satellite(sat),
            BiasTargetKey::system(sat.system),
        ] {
            let answer = self.osb_for_key(&key, obs, &query, family, carrier_hz);
            if answer.lookup != BiasLookup::Absent {
                return answer.lookup;
            }
        }
        BiasLookup::Absent
    }

    /// Resolves an OSB for one key. `carrier_hz` is `Some` for a phase query
    /// in cycles and carries the caller's carrier frequency.
    fn osb_for_key(
        &self,
        key: &BiasTargetKey,
        obs: &str,
        query: &ExactSeconds,
        family: BiasObservableFamily,
        carrier_hz: Option<Option<f64>>,
    ) -> Answer {
        let selection =
            self.select_record(key, &index_key(BiasKind::Osb, obs, None), query, |record| {
                record.kind == BiasKind::Osb && record.family == family
            });
        let first = selection.latest.first().copied();
        let lookup = self.evaluate(selection, query, |index, value| {
            let record = &self.records[index];
            match (carrier_hz, record.unit) {
                (None, _) | (Some(_), BiasUnit::Cycles) => Ok(value),
                (Some(None), BiasUnit::Nanoseconds) => {
                    Err(BiasLookup::CarrierFrequencyRequired { record: index })
                }
                (Some(Some(hz)), BiasUnit::Nanoseconds) => {
                    if hz.is_finite() && hz > 0.0 {
                        Ok(value * hz)
                    } else {
                        Err(BiasLookup::InvalidCarrierFrequency)
                    }
                }
            }
        });
        let stated = match (&lookup, first) {
            (BiasLookup::Available { .. }, Some(index)) => self
                .stated_affine(index)
                .map(|affine| (self.records[index].unit, vec![affine])),
            _ => None,
        };
        Answer { lookup, stated }
    }

    fn if_bias_seconds(
        &self,
        sat: GnssSatelliteId,
        observables: (&str, &str),
        frequencies_hz: (f64, f64),
        epoch: Instant,
    ) -> BiasLookup {
        let obs1 = self.code_osb_seconds(sat, observables.0, epoch);
        if !obs1.is_available() {
            return obs1;
        }
        let obs2 = self.code_osb_seconds(sat, observables.1, epoch);
        if !obs2.is_available() {
            return obs2;
        }
        let Some((alpha, beta)) = ionosphere_free_coefficients(frequencies_hz.0, frequencies_hz.1)
        else {
            return BiasLookup::InvalidCarrierFrequency;
        };
        obs1.combine(obs2, |b1, b2| alpha * b1 + beta * b2)
    }

    fn relative_code_bias_seconds(
        &self,
        sat: GnssSatelliteId,
        used_observables: (&str, &str),
        used_frequencies_hz: (f64, f64),
        clock_reference: (&str, &str),
        epoch: Instant,
    ) -> BiasLookup {
        let d1 = if used_observables.0 == clock_reference.0 {
            BiasLookup::exact(0.0)
        } else {
            self.code_dsb_seconds(sat, used_observables.0, clock_reference.0, epoch)
        };
        if !d1.is_available() {
            return d1;
        }
        let d2 = if used_observables.1 == clock_reference.1 {
            BiasLookup::exact(0.0)
        } else {
            self.code_dsb_seconds(sat, used_observables.1, clock_reference.1, epoch)
        };
        if !d2.is_available() {
            return d2;
        }
        let Some((alpha, beta)) =
            ionosphere_free_coefficients(used_frequencies_hz.0, used_frequencies_hz.1)
        else {
            return BiasLookup::InvalidCarrierFrequency;
        };
        d1.combine(d2, |d1, d2| alpha * d1 + beta * d2)
    }

    /// Resolves a code DSB between two observables for one key, through a
    /// chain of DSB records when no record joins them directly.
    fn dsb_for_key(
        &self,
        target_key: &BiasTargetKey,
        obs1: &str,
        obs2: &str,
        query: &ExactSeconds,
    ) -> Answer {
        let mut graph = DsbGraph::new();
        let mut blocked: Option<BiasLookup> = None;
        let start = (target_key.clone(), String::new());
        for ((key, _), indices) in self.index.range(start..) {
            if key != target_key {
                break;
            }
            let selection = select_covering_latest(&self.records, indices, query, |record| {
                record.kind == BiasKind::Dsb && record.family == BiasObservableFamily::Code
            });
            for &index in &selection.latest {
                let record = &self.records[index];
                let Some(record_obs2) = record.obs2.as_ref() else {
                    continue;
                };
                let value = match self.value_at(index, query) {
                    Ok(value) => value,
                    Err(status) => {
                        if blocked.is_none() {
                            blocked = Some(status);
                        }
                        continue;
                    }
                };
                let affine = self.stated_affine(index);
                graph
                    .entry(record.obs1.clone())
                    .or_default()
                    .entry(record_obs2.clone())
                    .or_default()
                    .push(DsbEdge {
                        value,
                        record: index,
                        affine: affine.clone(),
                        overridden: selection.overridden.clone(),
                    });
                graph
                    .entry(record_obs2.clone())
                    .or_default()
                    .entry(record.obs1.clone())
                    .or_default()
                    .push(DsbEdge {
                        value: -value,
                        record: index,
                        affine: affine.map(StatedAffine::negated),
                        overridden: selection.overridden.clone(),
                    });
            }
        }
        for group in graph.values_mut().flat_map(BTreeMap::values_mut) {
            group.sort_by_key(|edge| edge.record);
        }
        // Each edge carries its terms in its own direction, so a hop stated
        // against the other record compares with its negated terms.
        let hops_agree = |a: &DsbEdge, b: &DsbEdge| {
            records_agree(
                (self.records[a.record].unit, a.affine.as_ref(), a.value),
                (self.records[b.record].unit, b.affine.as_ref(), b.value),
            )
        };
        match resolve_dsb_path(&graph, obs1, obs2, &hops_agree) {
            DsbPath::Resolved {
                value,
                records,
                overridden,
                stated,
            } => Answer {
                lookup: BiasLookup::Available {
                    value,
                    records,
                    overridden,
                },
                // DSB records are code biases, stated in nanoseconds.
                stated: stated.map(|terms| (BiasUnit::Nanoseconds, terms)),
            },
            DsbPath::Conflict { records } => Answer::unstated(BiasLookup::Ambiguous { records }),
            DsbPath::None => Answer::unstated(blocked.unwrap_or(BiasLookup::Absent)),
        }
    }

    /// A record's value and slope exactly as its source row states them: the
    /// Bias-SINEX estimate (columns 70..91) and slope (columns 104..125), or
    /// the CODE DCB value (columns 24..38), with twice the slope's section
    /// 5.1 reference epoch in whole seconds. `None` for a record not read
    /// from text, or a sloped record without a reference epoch.
    fn stated_affine(&self, index: usize) -> Option<StatedAffine> {
        let record = &self.records[index];
        let line = self.lines.get(record.line?.checked_sub(1)?)?;
        let raw = line.raw().trim_ascii_end();
        let (value_text, slope_text) = match self.source_format? {
            BiasSourceFormat::BiasSinex => (byte_field(raw, 70, 91)?, byte_field(raw, 104, 125)),
            BiasSourceFormat::CodeDcb => (byte_field(raw, 24, 38)?, None),
        };
        let value = ExactDecimal::parse(&value_text)?;
        let slope = match record.slope {
            None => None,
            Some(_) => {
                let slope = ExactDecimal::parse(&slope_text?)?;
                let twice_reference_s = match record.slope_reference() {
                    BiasSlopeReference::Midpoint { start, end } => {
                        start.instant_seconds().checked_add(end.instant_seconds())?
                    }
                    BiasSlopeReference::Start(epoch) | BiasSlopeReference::End(epoch) => {
                        epoch.instant_seconds().checked_mul(2)?
                    }
                    BiasSlopeReference::Undefined => return None,
                };
                Some((slope, twice_reference_s))
            }
        };
        Some(StatedAffine { value, slope })
    }

    /// Keys a station query consults: the exact normalized station when the
    /// product names it. Otherwise, for the same system and satellite, the
    /// stations whose identifier corresponds to the query's: a bare legacy
    /// four-character code and a nine-character identifier or code-plus-DOMES
    /// name carrying that code, in either direction, and two spellings of one
    /// code and DOMES number. A nine-character identifier and a
    /// code-plus-DOMES name never correspond, nor do two different
    /// nine-character identifiers or DOMES numbers.
    fn station_keys(&self, exact: &BiasTargetKey) -> Vec<BiasTargetKey> {
        let first = self
            .index
            .range((exact.clone(), String::new())..)
            .next()
            .map(|((key, _), _)| key);
        if first == Some(exact) {
            return vec![exact.clone()];
        }
        let Some(query_station) = exact.station.as_deref() else {
            return Vec::new();
        };
        let query_form = station_form(query_station);
        let mut keys: Vec<BiasTargetKey> = Vec::new();
        for (key, _) in self.index.keys() {
            let Some(station) = key.station.as_deref() else {
                continue;
            };
            if key.system == exact.system
                && key.sat == exact.sat
                && stations_correspond(query_form, station_form(station))
                && keys.last() != Some(key)
            {
                keys.push(key.clone());
            }
        }
        keys
    }

    fn select_record(
        &self,
        target_key: &BiasTargetKey,
        obs_key: &str,
        query: &ExactSeconds,
        predicate: impl Fn(&BiasRecord) -> bool,
    ) -> Selection {
        match self.index.get(&(target_key.clone(), obs_key.to_string())) {
            Some(indices) => select_covering_latest(&self.records, indices, query, predicate),
            None => Selection::default(),
        }
    }

    /// Evaluates the records a selection returned. Records that agree, as
    /// [`records_agree`] compares them, are one answer, naming the
    /// records the selection overrides; records that differ are ambiguous.
    fn evaluate(
        &self,
        selection: Selection,
        query: &ExactSeconds,
        convert: impl Fn(usize, f64) -> Result<f64, BiasLookup>,
    ) -> BiasLookup {
        let mut value: Option<(usize, f64)> = None;
        let mut conflict = false;
        for &index in &selection.latest {
            let found = match self
                .value_at(index, query)
                .and_then(|value| convert(index, value))
            {
                Ok(found) => found,
                Err(status) => return status,
            };
            match value {
                None => value = Some((index, found)),
                Some((first, held)) => {
                    let first_stated = self.stated_affine(first);
                    let stated = self.stated_affine(index);
                    if !records_agree(
                        (self.records[first].unit, first_stated.as_ref(), held),
                        (self.records[index].unit, stated.as_ref(), found),
                    ) {
                        conflict = true;
                    }
                }
            }
        }
        if conflict {
            return BiasLookup::Ambiguous {
                records: selection.latest,
            };
        }
        match value {
            Some((_, value)) => BiasLookup::Available {
                value,
                records: selection.latest,
                overridden: selection.overridden,
            },
            None => BiasLookup::Absent,
        }
    }

    /// Value of a covering record at the query, with its slope applied from
    /// the section 5.1 reference epoch.
    fn value_at(&self, index: usize, query: &ExactSeconds) -> Result<f64, BiasLookup> {
        let record = &self.records[index];
        let Some(slope) = record.slope else {
            return Ok(record.value);
        };
        // The time from the reference epoch, exact and rounded once.
        let since = |reference: ExactSeconds| query.sub(&reference).to_f64();
        let dt_s = match record.slope_reference() {
            BiasSlopeReference::Midpoint { start, end } => {
                // Half the interval is a whole or half second.
                let half = ExactSeconds::from_decimal(i128::from(start.seconds_until(end)) * 5, 1);
                since(start.exact_j2000_seconds().add(&half))
            }
            BiasSlopeReference::Start(start) => since(start.exact_j2000_seconds()),
            BiasSlopeReference::End(end) => since(end.exact_j2000_seconds()),
            BiasSlopeReference::Undefined => {
                return Err(BiasLookup::UndefinedSlopeReference { record: index })
            }
        };
        Ok(record.value + slope * dt_s)
    }
}

/// Covering records a lookup selects, and the covering records they
/// override.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Selection {
    /// Records sharing the latest start among the covering ones.
    latest: Vec<usize>,
    /// Covering records that start earlier.
    overridden: Vec<usize>,
}

/// Records among `indices` that satisfy `predicate` and cover `query`, split
/// into those sharing the latest start and those starting earlier.
fn select_covering_latest(
    records: &[BiasRecord],
    indices: &[usize],
    query: &ExactSeconds,
    predicate: impl Fn(&BiasRecord) -> bool,
) -> Selection {
    let covering: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|&index| predicate(&records[index]) && record_covers(&records[index], query))
        .collect();
    let Some(latest_start) = covering
        .iter()
        .map(|&index| instant_of(records[index].valid_from))
        .max()
    else {
        return Selection::default();
    };
    let (latest, overridden): (Vec<usize>, Vec<usize>) = covering
        .into_iter()
        .partition(|&index| instant_of(records[index].valid_from) == latest_start);
    Selection { latest, overridden }
}

/// Whether a record's validity window `[from, until)` holds the query,
/// compared exactly.
fn record_covers(record: &BiasRecord, query: &ExactSeconds) -> bool {
    if let Some(from) = record.valid_from {
        if query.sub(&from.exact_j2000_seconds()).sign() == Ordering::Less {
            return false;
        }
    }
    if let Some(until) = record.valid_until {
        if query.sub(&until.exact_j2000_seconds()).sign() != Ordering::Less {
            return false;
        }
    }
    true
}

/// Form of a station identifier, for matching a query against the stations
/// a product names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StationForm<'a> {
    /// A legacy four-character code on its own.
    Code(&'a str),
    /// A legacy code and the DOMES number after it, as CODE DCB files name
    /// stations, with or without a blank between them.
    CodeWithDomes(&'a str, &'a str),
    /// A nine-character identifier: code, monument and receiver digits and
    /// country code.
    NineCharacter(&'a str),
    /// Anything else, such as a receiver group name; matched exactly only.
    Other,
}

fn station_form(normalized: &str) -> StationForm<'_> {
    let Some(code) = station_marker(normalized) else {
        return StationForm::Other;
    };
    let rest = &normalized[code.len()..];
    if rest.is_empty() {
        StationForm::Code(code)
    } else if is_domes_number(rest.trim_start()) {
        StationForm::CodeWithDomes(code, rest.trim_start())
    } else if normalized.len() == 9 {
        StationForm::NineCharacter(code)
    } else {
        StationForm::Other
    }
}

/// A DOMES number: five digits, a letter and three digits, such as
/// `97103M001`.
fn is_domes_number(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() == 9
        && bytes[..5].iter().all(u8::is_ascii_digit)
        && bytes[5].is_ascii_alphabetic()
        && bytes[6..].iter().all(u8::is_ascii_digit)
}

/// Whether a queried station and a product station, not equal as written,
/// name the same station: a bare legacy code corresponds to any identifier
/// carrying it, and two code-plus-DOMES spellings with the same code and
/// DOMES number, one with a blank between them and one without, correspond.
/// A nine-character identifier and a code-plus-DOMES name carry different
/// information about the station, so neither answers for the other; nor do
/// two different nine-character identifiers or DOMES numbers.
fn stations_correspond(query: StationForm<'_>, station: StationForm<'_>) -> bool {
    use StationForm::{Code, CodeWithDomes, NineCharacter};
    match (query, station) {
        (Code(a), CodeWithDomes(b, _) | NineCharacter(b))
        | (CodeWithDomes(a, _) | NineCharacter(a), Code(b)) => a == b,
        (CodeWithDomes(a, a_domes), CodeWithDomes(b, b_domes)) => a == b && a_domes == b_domes,
        _ => false,
    }
}

#[derive(Debug, Clone, PartialEq)]
struct DsbEdge {
    /// The hop's value in seconds, negated when the hop runs against the
    /// record.
    value: f64,
    record: usize,
    /// The record's stated value and slope, negated like `value`; `None`
    /// when the set keeps no source text for it.
    affine: Option<StatedAffine>,
    /// Covering records the edge's record overrides.
    overridden: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
enum DsbPath {
    None,
    Resolved {
        value: f64,
        records: Vec<usize>,
        overridden: Vec<usize>,
        /// The stated terms of the representative route, when all are kept.
        stated: Option<Vec<StatedAffine>>,
    },
    Conflict {
        records: Vec<usize>,
    },
}

/// Writes a [`BiasSet`] as Bias-SINEX text.
///
/// A set read from Bias-SINEX restates every source line exactly, in order,
/// with its line ending: the header and footer lines, comments, every block
/// including blocks the reader does not model, and rows it did not read as
/// records. A line that is not valid UTF-8 cannot be held in a `String`, so
/// such a set returns [`BiasError::InvalidUtf8Line`]; [`write_bias_sinex_bytes`]
/// restates it.
///
/// Any other set, such as a CODE DCB set, is written from its records under
/// the header line given by [`BiasSet::set_sinex_header`] (without one,
/// [`BiasError::MissingWriterMetadata`]), with the header count and mode, the
/// `%=ENDBIA` footer, the set's `FILE/REFERENCE` and `BIAS/DESCRIPTION` rows,
/// and `BIAS_MODE` and `TIME_SYSTEM` rows when those rows lack them. Every
/// field is checked to read back exactly: a numeric field is written only in a
/// spelling within its section 4.8 width that reads back to the stored `f64`,
/// and anything else is refused with a named [`BiasError`]. A satellite clock
/// reference is not required, since section 4.6 requires one only for
/// products consistent with the ionosphere-free combination.
pub fn write_bias_sinex(set: &BiasSet) -> Result<String, BiasError> {
    if set.source_format == Some(BiasSourceFormat::BiasSinex) {
        return restate_source_text(&set.lines);
    }
    write_generated_bias_sinex(set)
}

/// Writes a [`BiasSet`] as Bias-SINEX bytes, as [`write_bias_sinex`] does,
/// restating lines that are not valid UTF-8 with their exact bytes.
pub fn write_bias_sinex_bytes(set: &BiasSet) -> Result<Vec<u8>, BiasError> {
    if set.source_format == Some(BiasSourceFormat::BiasSinex) {
        return Ok(restate_source_bytes(&set.lines));
    }
    write_generated_bias_sinex(set).map(String::into_bytes)
}

/// The source lines as text, each with its line ending. A line that is not
/// valid UTF-8 is refused by number rather than replaced.
fn restate_source_text(lines: &[BiasSourceLine]) -> Result<String, BiasError> {
    let mut out = String::new();
    for line in lines {
        if line.bytes.is_some() {
            return Err(BiasError::InvalidUtf8Line { line: line.number });
        }
        out.push_str(&line.text);
        out.push_str(line.terminator.as_str());
    }
    Ok(out)
}

/// The exact source bytes, line endings included.
fn restate_source_bytes(lines: &[BiasSourceLine]) -> Vec<u8> {
    let mut out = Vec::new();
    for line in lines {
        out.extend_from_slice(line.raw());
        out.extend_from_slice(line.terminator.as_str().as_bytes());
    }
    out
}

fn write_generated_bias_sinex(set: &BiasSet) -> Result<String, BiasError> {
    let header = set
        .header
        .sinex
        .as_ref()
        .ok_or(BiasError::MissingWriterMetadata {
            field: "sinex header",
        })?;
    let (mode_char, mode_label) = match set.mode {
        BiasMode::Absolute => ('A', "ABSOLUTE"),
        BiasMode::Relative => ('R', "RELATIVE"),
        BiasMode::Unspecified => return Err(BiasError::MissingWriterMetadata { field: "mode" }),
    };
    let scale = set.time_scale.ok_or(BiasError::MissingWriterMetadata {
        field: "time system",
    })?;
    let time_label = sinex_time_system_label(scale)?;

    let mut out = format_sinex_header_line(header, mode_char, set.records.len())?;
    out.push('\n');
    out.push_str("+FILE/REFERENCE\n");
    for row in &set.header.file_reference {
        out.push_str(&format_info_row(row, 18)?);
        out.push('\n');
    }
    out.push_str("-FILE/REFERENCE\n");
    out.push_str("+BIAS/DESCRIPTION\n");
    for row in &set.header.description {
        out.push_str(&format_info_row(row, 39)?);
        out.push('\n');
    }
    if set.header.description_value("BIAS_MODE").is_none() {
        out.push_str(&format!(" {:<39} {mode_label}\n", "BIAS_MODE"));
    }
    if set.header.description_value("TIME_SYSTEM").is_none() {
        out.push_str(&format!(" {:<39} {time_label}\n", "TIME_SYSTEM"));
    }
    out.push_str("-BIAS/DESCRIPTION\n");
    out.push_str("+BIAS/SOLUTION\n");
    out.push_str(
        "*BIAS SVN_ PRN STATION__ OBS1 OBS2 BIAS_START____ BIAS_END______ UNIT __ESTIMATED_VALUE____ _STD_DEV___ __ESTIMATED_SLOPE____ _STD_DEV___\n",
    );
    for record in &set.records {
        out.push_str(&format_sinex_solution_record(record)?);
        out.push('\n');
    }
    out.push_str("-BIAS/SOLUTION\n");
    out.push_str("%=ENDBIA\n");
    Ok(out)
}

/// Formats the 74-column header line of section 4.1, checking each field.
fn format_sinex_header_line(
    header: &BiasSinexHeader,
    mode: char,
    count: usize,
) -> Result<String, BiasError> {
    if header.version != BIAS_SINEX_VERSION {
        return Err(BiasError::UnsupportedVersion {
            version: header.version.clone(),
        });
    }
    let agency = |value: &Option<String>, name: &'static str| -> Result<String, BiasError> {
        let value = value
            .as_deref()
            .ok_or(BiasError::MissingWriterMetadata { field: name })?;
        if value.is_empty() || value.len() > 3 || !value.bytes().all(|b| b.is_ascii_graphic()) {
            return Err(BiasError::InvalidInput {
                field: name,
                reason: "cannot fit format",
            });
        }
        Ok(value.to_string())
    };
    let epoch = |value: &Option<String>, name: &'static str| -> Result<String, BiasError> {
        let value = value
            .as_deref()
            .ok_or(BiasError::MissingWriterMetadata { field: name })?;
        if value.len() != 14 || BiasEpoch::parse_sinex(value).is_err() {
            return Err(BiasError::InvalidInput {
                field: name,
                reason: "cannot fit format",
            });
        }
        Ok(value.to_string())
    };
    if count > 99_999_999 {
        return Err(BiasError::InvalidInput {
            field: "estimate count",
            reason: "cannot fit format",
        });
    }
    Ok(format!(
        "%=BIA {} {:<3} {} {:<3} {} {} {mode} {count:08}",
        BIAS_SINEX_VERSION,
        agency(&header.file_agency, "file agency")?,
        epoch(&header.creation_time, "creation time")?,
        agency(&header.data_agency, "data agency")?,
        epoch(&header.start, "solution start")?,
        epoch(&header.end, "solution end")?,
    ))
}

fn format_info_row(row: &BiasInfoRow, keyword_width: usize) -> Result<String, BiasError> {
    for text in [&row.keyword, &row.value] {
        if text.chars().any(|c| c.is_control()) {
            return Err(BiasError::InvalidInput {
                field: "header row",
                reason: "contains control character",
            });
        }
    }
    if row.keyword.is_empty() || row.keyword.contains(char::is_whitespace) {
        return Err(BiasError::InvalidInput {
            field: "header row",
            reason: "keyword is not one token",
        });
    }
    if row.keyword.starts_with(['*', '+', '-', '%']) {
        return Err(BiasError::InvalidInput {
            field: "header row",
            reason: "keyword reads as a control line",
        });
    }
    if row.value.is_empty() {
        return Ok(format!(" {}", row.keyword));
    }
    Ok(format!(
        " {:<width$} {}",
        row.keyword,
        row.value,
        width = keyword_width
    ))
}

/// Label [`write_bias_sinex`] writes for a scale: a section 4.6 label, which
/// this reader reads back as the same scale. TT, TDB, TCG and TCB have no
/// section 4.6 label, and GLONASS time has none either, since the flag `R`
/// reads as UTC.
fn sinex_time_system_label(scale: TimeScale) -> Result<&'static str, BiasError> {
    let label = match scale {
        TimeScale::Gpst => "G",
        TimeScale::Gst => "E",
        TimeScale::Bdt => "C",
        TimeScale::Qzsst => "J",
        TimeScale::Utc => "UTC",
        TimeScale::Tai => "TAI",
        TimeScale::Glonasst | TimeScale::Tt | TimeScale::Tdb | TimeScale::Tcg | TimeScale::Tcb => {
            return Err(BiasError::UnsupportedTimeSystem { scale: Some(scale) })
        }
    };
    debug_assert_eq!(
        read_time_system_label(label),
        TimeSystemLabel::Standard(scale)
    );
    Ok(label)
}

/// Label a generated DCB title states for a scale: one the DCB title reader
/// reads back as the same scale.
fn dcb_time_system_label(scale: TimeScale) -> Result<&'static str, BiasError> {
    let label = match scale {
        TimeScale::Gpst => "G",
        TimeScale::Utc => "UTC",
        TimeScale::Gst => "E",
        TimeScale::Bdt => "C",
        TimeScale::Qzsst => "J",
        TimeScale::Tai => "TAI",
        TimeScale::Tcg => "TCG",
        TimeScale::Tcb => "TCB",
        TimeScale::Tt => "TT",
        TimeScale::Tdb => "TDB",
        // `R` reads back as UTC, so GLONASS time has no label.
        TimeScale::Glonasst => return Err(BiasError::UnsupportedTimeSystem { scale: Some(scale) }),
    };
    debug_assert_eq!(dcb_title_time_scale(label), Some(scale));
    Ok(label)
}

/// How a `TIME_SYSTEM` label reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimeSystemLabel {
    /// A section 4.6 label: a RINEX GNSS system flag, `UTC` or `TAI`.
    Standard(TimeScale),
    /// Any other label, with the scale it names unambiguously, if any.
    NonStandard(Option<TimeScale>),
}

/// Reads a `TIME_SYSTEM` label. Section 4.6 allows a RINEX GNSS system flag,
/// `UTC` and `TAI`. `R` reads as UTC, as RINEX reads GLONASS time tags and as
/// the RINEX clock reader does. `S` reads as GPS time, since RINEX tags SBAS
/// data in GPS time. `I` (IRNSS time) reads as GPS time, as the SP3 reader
/// of this library and RTKLIB treat IRNSS system time.
fn read_time_system_label(label: &str) -> TimeSystemLabel {
    use TimeSystemLabel::{NonStandard, Standard};
    match label {
        "G" | "S" | "I" => Standard(TimeScale::Gpst),
        "R" | "UTC" => Standard(TimeScale::Utc),
        "E" => Standard(TimeScale::Gst),
        "C" => Standard(TimeScale::Bdt),
        "J" => Standard(TimeScale::Qzsst),
        "TAI" => Standard(TimeScale::Tai),
        "GPS" | "GPST" => NonStandard(Some(TimeScale::Gpst)),
        "GLO" => NonStandard(Some(TimeScale::Utc)),
        "GAL" | "GST" => NonStandard(Some(TimeScale::Gst)),
        "BDT" => NonStandard(Some(TimeScale::Bdt)),
        "QZS" | "QZSST" => NonStandard(Some(TimeScale::Qzsst)),
        "TCG" => NonStandard(Some(TimeScale::Tcg)),
        "TCB" => NonStandard(Some(TimeScale::Tcb)),
        "TT" => NonStandard(Some(TimeScale::Tt)),
        "TDB" => NonStandard(Some(TimeScale::Tdb)),
        _ => NonStandard(None),
    }
}

/// Scale the label of a generated DCB title names: the labels the writer
/// states, `G`, `E`, `C`, `J`, `UTC`, `TAI`, `TT`, `TDB`, `TCG` and `TCB`,
/// and `R`, which version 2.1.1 of this library wrote for UTC.
fn dcb_title_time_scale(token: &str) -> Option<TimeScale> {
    match token {
        "G" => Some(TimeScale::Gpst),
        "E" => Some(TimeScale::Gst),
        "C" => Some(TimeScale::Bdt),
        "J" => Some(TimeScale::Qzsst),
        "UTC" | "R" => Some(TimeScale::Utc),
        "TT" => Some(TimeScale::Tt),
        "TDB" => Some(TimeScale::Tdb),
        "TAI" => Some(TimeScale::Tai),
        "TCG" => Some(TimeScale::Tcg),
        "TCB" => Some(TimeScale::Tcb),
        _ => None,
    }
}

/// Writes a [`BiasSet`] as CODE DCB text.
///
/// A set read from CODE DCB restates every source line exactly, with its line
/// ending, whatever its metadata, since the output is its input; a line that
/// is not valid UTF-8 returns [`BiasError::InvalidUtf8Line`], and
/// [`write_code_dcb_bytes`] restates it.
///
/// Any other set is written as a generated title, column headings and one
/// row per record. It must contain [`CodeDcbOptions`] in its header, and the metadata
/// must describe every record exactly: a code DSB targeting a satellite or
/// receiver, between the observables the metadata pair maps to for its
/// system, valid over the metadata month, with no SVN, slope or slope
/// uncertainty. A record the metadata does not describe is refused with
/// [`BiasError::DcbRecordMismatch`], and a set on another time scale than the
/// metadata's with [`BiasError::UnsupportedTimeSystem`], since writing either
/// would change what it reads back as. Code seconds and uncertainties are
/// stated in nanoseconds, each checked to read back exactly, and the title's
/// time system as a label the DCB title reader reads back as the same scale.
pub fn write_code_dcb(set: &BiasSet) -> Result<String, BiasError> {
    if set.source_format == Some(BiasSourceFormat::CodeDcb) {
        return restate_source_text(&set.lines);
    }
    let meta = checked_dcb_meta(set)?;
    write_generated_code_dcb(set, meta)
}

/// Writes a [`BiasSet`] as CODE DCB bytes, as [`write_code_dcb`] does,
/// restating source lines that are not valid UTF-8 with their exact bytes.
pub fn write_code_dcb_bytes(set: &BiasSet) -> Result<Vec<u8>, BiasError> {
    if set.source_format == Some(BiasSourceFormat::CodeDcb) {
        return Ok(restate_source_bytes(&set.lines));
    }
    let meta = checked_dcb_meta(set)?;
    write_generated_code_dcb(set, meta).map(String::into_bytes)
}

fn checked_dcb_meta(set: &BiasSet) -> Result<&CodeDcbOptions, BiasError> {
    let meta = set
        .header
        .dcb_meta
        .as_ref()
        .ok_or(BiasError::MissingWriterMetadata { field: "dcb_meta" })?;
    validate_dcb_options(meta)?;
    check_dcb_records(set, meta)?;
    Ok(meta)
}

fn write_generated_code_dcb(set: &BiasSet, meta: &CodeDcbOptions) -> Result<String, BiasError> {
    let label = dcb_time_system_label(meta.time_scale)?;
    let mut out = String::new();
    out.push_str(&format!(
        "# DCB {}-{} {:04}-{:02} {}\n",
        meta.pair.0, meta.pair.1, meta.year, meta.month, label
    ));
    out.push_str(" PRN / STATION NAME        VALUE (ns)  RMS (ns)\n");
    out.push_str("***   ****************    *****.***   *****.***\n");
    for record in &set.records {
        let value_str = validate_dcb_numeric(record.value, "value")?;
        let sigma_str = match record.sigma {
            Some(sigma) => {
                let s = validate_dcb_sigma(sigma)?;
                format!("   {s}")
            }
            None => String::new(),
        };
        match &record.target {
            BiasTarget::Satellite(sat) => {
                out.push_str(&format!(
                    "{sat:<3}                       {value_str}{sigma_str}\n"
                ));
            }
            BiasTarget::Receiver { system, station } => {
                validate_dcb_station(station)?;
                out.push_str(&format!(
                    "{:<6}{station:<16}    {value_str}{sigma_str}\n",
                    system.letter()
                ));
            }
            _ => {
                return Err(BiasError::InvalidInput {
                    field: "bias target",
                    reason: "CODE DCB writer supports satellite and receiver records",
                });
            }
        }
    }
    Ok(out)
}

/// Checks that the DCB metadata describes every record exactly, so a DCB
/// file written under it reads back as the same records.
fn check_dcb_records(set: &BiasSet, meta: &CodeDcbOptions) -> Result<(), BiasError> {
    if set.time_scale != Some(meta.time_scale) {
        return Err(BiasError::UnsupportedTimeSystem {
            scale: set.time_scale,
        });
    }
    let (start, end) = dcb_month_interval(meta.year, meta.month)?;
    for (index, record) in set.records.iter().enumerate() {
        if record.kind != BiasKind::Dsb
            || record.family != BiasObservableFamily::Code
            || record.unit != BiasUnit::Nanoseconds
        {
            return Err(BiasError::InvalidInput {
                field: "bias set",
                reason: "CODE DCB writer requires code DSB records",
            });
        }
        let system = match &record.target {
            BiasTarget::Satellite(sat) => sat.system,
            BiasTarget::Receiver { system, .. } => *system,
            _ => {
                return Err(BiasError::InvalidInput {
                    field: "bias target",
                    reason: "CODE DCB writer supports satellite and receiver records",
                })
            }
        };
        let mismatch = |field: &'static str| BiasError::DcbRecordMismatch {
            record: index,
            field,
        };
        let Some((obs1, obs2)) = map_legacy_dcb_pair(system, &meta.pair.0, &meta.pair.1) else {
            return Err(mismatch("observables"));
        };
        if record.obs1 != obs1 || record.obs2.as_deref() != Some(obs2.as_str()) {
            return Err(mismatch("observables"));
        }
        if !same_instant(record.valid_from, Some(start)) || !same_instant(record.valid_until, end) {
            return Err(mismatch("validity interval"));
        }
        if record.svn.is_some() {
            return Err(mismatch("svn"));
        }
        if record.slope.is_some() {
            return Err(mismatch("slope"));
        }
        if record.slope_sigma.is_some() {
            return Err(mismatch("slope sigma"));
        }
    }
    Ok(())
}

fn validate_dcb_numeric(value_s: f64, field_name: &'static str) -> Result<String, BiasError> {
    if !value_s.is_finite() {
        return Err(BiasError::InvalidInput {
            field: field_name,
            reason: "not finite",
        });
    }
    let value_ns = value_s / NS_TO_S;
    if !value_ns.is_finite() {
        return Err(BiasError::InvalidInput {
            field: field_name,
            reason: "cannot fit format",
        });
    }
    let formatted = format!("{value_ns:9.3}");
    if formatted.len() > 9 {
        return Err(BiasError::InvalidInput {
            field: field_name,
            reason: "cannot fit format",
        });
    }
    let parsed_ns =
        strict_f64(formatted.trim(), field_name).map_err(|_| BiasError::InvalidInput {
            field: field_name,
            reason: "cannot fit format",
        })?;
    if !parsed_ns.is_finite() {
        return Err(BiasError::InvalidInput {
            field: field_name,
            reason: "cannot fit format",
        });
    }
    let recovered_s = parsed_ns * NS_TO_S;
    if recovered_s.to_bits() != value_s.to_bits() {
        return Err(BiasError::InvalidInput {
            field: field_name,
            reason: "excessive precision",
        });
    }
    Ok(formatted)
}

fn validate_dcb_sigma(sigma_s: f64) -> Result<String, BiasError> {
    if !sigma_s.is_finite() {
        return Err(BiasError::InvalidInput {
            field: "sigma",
            reason: "not finite",
        });
    }
    validate_dcb_numeric(sigma_s, "sigma")
}

fn validate_dcb_station(station: &str) -> Result<(), BiasError> {
    if station.is_empty() || station.trim().is_empty() {
        return Err(BiasError::InvalidInput {
            field: "station",
            reason: "missing",
        });
    }
    if station.contains(['\n', '\r']) {
        return Err(BiasError::InvalidInput {
            field: "station",
            reason: "line injection",
        });
    }
    if station.chars().any(|c| c.is_ascii_control()) {
        return Err(BiasError::InvalidInput {
            field: "station",
            reason: "contains control character",
        });
    }
    if !station.is_ascii() {
        return Err(BiasError::InvalidInput {
            field: "station",
            reason: "cannot fit format",
        });
    }
    if station.starts_with(' ') || station.ends_with(' ') {
        return Err(BiasError::InvalidInput {
            field: "station",
            reason: "shifts fixed columns",
        });
    }
    if station.len() > 16 {
        return Err(BiasError::InvalidInput {
            field: "station",
            reason: "cannot fit format",
        });
    }
    Ok(())
}

/// Numeric fields of a Bias-SINEX solution row, which differ in how the
/// stated number maps to the stored value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SinexNumericField {
    Value,
    Sigma,
    Slope,
    SlopeSigma,
}

impl SinexNumericField {
    fn name(self) -> &'static str {
        match self {
            Self::Value => "bias value",
            Self::Sigma => "bias sigma",
            Self::Slope => "bias slope",
            Self::SlopeSigma => "bias slope sigma",
        }
    }

    fn width(self) -> usize {
        match self {
            Self::Value | Self::Slope => SINEX_ESTIMATE_WIDTH,
            Self::Sigma | Self::SlopeSigma => SINEX_SIGMA_WIDTH,
        }
    }

    /// The stored value of a stated number. The reader uses exactly this
    /// arithmetic, so the writer's readback check is the reader's.
    fn decode(self, unit: BiasUnit, stated: f64) -> f64 {
        match (self, unit) {
            (Self::Value | Self::Sigma, BiasUnit::Nanoseconds) => stated * NS_TO_S,
            (Self::Value | Self::Sigma, BiasUnit::Cycles) => stated,
            (Self::Slope | Self::SlopeSigma, BiasUnit::Nanoseconds) => {
                stated * NS_TO_S / SINEX_BIAS_SLOPE_DENOMINATOR_S
            }
            (Self::Slope | Self::SlopeSigma, BiasUnit::Cycles) => {
                stated / SINEX_BIAS_SLOPE_DENOMINATOR_S
            }
        }
    }

    /// A first guess at the stated number for a stored value; the search
    /// around it finds the spelling that decodes exactly.
    fn encode(self, unit: BiasUnit, stored: f64) -> f64 {
        match (self, unit) {
            (Self::Value | Self::Sigma, BiasUnit::Nanoseconds) => stored / NS_TO_S,
            (Self::Value | Self::Sigma, BiasUnit::Cycles) => stored,
            (Self::Slope | Self::SlopeSigma, BiasUnit::Nanoseconds) => {
                stored * SINEX_BIAS_SLOPE_DENOMINATOR_S / NS_TO_S
            }
            (Self::Slope | Self::SlopeSigma, BiasUnit::Cycles) => {
                stored * SINEX_BIAS_SLOPE_DENOMINATOR_S
            }
        }
    }
}

/// The shortest spelling within the field width whose reading decodes to
/// exactly `stored`, or a named refusal.
///
/// The stated numbers tried are the encoded value and its neighbours a few
/// ulps away, each spelled as its shortest round-trip decimal and in
/// exponent form, with and without a leading digit, always with a decimal
/// point. Signed zero keeps its sign.
fn spell_sinex_numeric(
    stored: f64,
    unit: BiasUnit,
    field: SinexNumericField,
) -> Result<String, BiasError> {
    let name = field.name();
    if !stored.is_finite() {
        return Err(BiasError::InvalidInput {
            field: name,
            reason: "not finite",
        });
    }
    let seed = field.encode(unit, stored);
    if !seed.is_finite() {
        return Err(BiasError::InvalidInput {
            field: name,
            reason: "cannot fit format",
        });
    }
    let mut candidates = vec![seed];
    let (mut up, mut down) = (seed, seed);
    for _ in 0..EXACT_SPELLING_ULPS {
        up = up.next_up();
        down = down.next_down();
        candidates.push(up);
        candidates.push(down);
    }
    let mut best: Option<String> = None;
    let mut exact_but_wide = false;
    for candidate in candidates {
        if !candidate.is_finite() {
            continue;
        }
        for spelling in candidate_spellings(candidate) {
            let Ok(read) = strict_f64(&spelling, name) else {
                continue;
            };
            if field.decode(unit, read).to_bits() != stored.to_bits() {
                continue;
            }
            if spelling.len() > field.width() {
                exact_but_wide = true;
                continue;
            }
            if best.as_ref().is_none_or(|held| spelling.len() < held.len()) {
                best = Some(spelling);
            }
        }
    }
    best.ok_or(BiasError::InvalidInput {
        field: name,
        reason: if exact_but_wide {
            "cannot fit format"
        } else {
            "excessive precision"
        },
    })
}

/// Spellings of one number, each with a decimal point: its shortest
/// round-trip decimal, its exponent form, and both in the leading-dot form of
/// a Fortran E edit shown in the section 4.7 and 5.2 examples
/// (`.398201E-01`), such as `.5` and `-.5E-9`.
fn candidate_spellings(value: f64) -> Vec<String> {
    let plain = with_decimal_point(format!("{value}"));
    let exponent = with_decimal_point(format!("{value:E}"));
    let mut spellings = vec![plain.clone(), exponent.clone()];
    // 0.0046 as .0046, and -0.5 as -.5.
    if let Some(rest) = plain.strip_prefix("0.") {
        spellings.push(format!(".{rest}"));
    } else if let Some(rest) = plain.strip_prefix("-0.") {
        spellings.push(format!("-.{rest}"));
    }
    // d.ddd E n as .dddd E n+1.
    let (sign, unsigned) = match exponent.strip_prefix('-') {
        Some(unsigned) => ("-", unsigned),
        None => ("", exponent.as_str()),
    };
    if let Some((mantissa, power)) = unsigned.split_once('E') {
        if let Ok(power) = power.parse::<i32>() {
            let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
            spellings.push(format!("{sign}.{digits}E{}", power + 1));
        }
    }
    spellings
}

/// A spelling with a decimal point. A Fortran `E` or `F` edit descriptor, as
/// the Bernese reader of CODE products uses, scales a number written without
/// a decimal point by the descriptor's decimal count, so `1` in an `E21.15`
/// field reads as `1E-15`. `1` becomes `1.`, `-0` becomes `-0.` and `1E-9`
/// becomes `1.E-9`.
fn with_decimal_point(spelling: String) -> String {
    if spelling.contains('.') {
        return spelling;
    }
    match spelling.find(['E', 'e']) {
        Some(exponent) => format!("{}.{}", &spelling[..exponent], &spelling[exponent..]),
        None => format!("{spelling}."),
    }
}

/// Formats one solution row at the section 4.8 columns, checking that every
/// field fits and reads back.
fn format_sinex_solution_record(record: &BiasRecord) -> Result<String, BiasError> {
    let (svn, prn, station) = target_fields(record);
    let text_field = |value: &str, width: usize, name: &'static str| -> Result<(), BiasError> {
        if !value.is_ascii() || value.len() > width {
            return Err(BiasError::InvalidInput {
                field: name,
                reason: "cannot fit format",
            });
        }
        if value.chars().any(|c| c.is_ascii_control()) {
            return Err(BiasError::InvalidInput {
                field: name,
                reason: "contains control character",
            });
        }
        if value.starts_with(' ') || value.ends_with(' ') {
            return Err(BiasError::InvalidInput {
                field: name,
                reason: "shifts fixed columns",
            });
        }
        Ok(())
    };
    let obs2 = record.obs2.as_deref().unwrap_or("");
    text_field(svn.as_str(), 4, "svn")?;
    text_field(prn.as_str(), 3, "prn")?;
    text_field(station.as_str(), 9, "station")?;
    text_field(record.obs1.as_str(), 4, "obs1")?;
    text_field(obs2, 4, "obs2")?;
    let epoch = |raw: &str, parsed: Option<BiasEpoch>| -> String {
        if raw.is_empty() {
            parsed
                .map(BiasEpoch::format_sinex)
                .unwrap_or_else(|| "0000:000:00000".to_string())
        } else {
            raw.to_string()
        }
    };
    let start = epoch(record.raw_epochs.0.as_str(), record.valid_from);
    let end = epoch(record.raw_epochs.1.as_str(), record.valid_until);
    for (text, name) in [(&start, "bias start"), (&end, "bias end")] {
        if text.len() != 14 || BiasEpoch::parse_sinex(text).is_err() {
            return Err(BiasError::InvalidInput {
                field: name,
                reason: "cannot fit format",
            });
        }
    }
    let spell = |value: Option<f64>, field: SinexNumericField| -> Result<String, BiasError> {
        match value {
            Some(value) => spell_sinex_numeric(value, record.unit, field),
            None => Ok(String::new()),
        }
    };
    let value = spell(Some(record.value), SinexNumericField::Value)?;
    let sigma = spell(record.sigma, SinexNumericField::Sigma)?;
    let slope = spell(record.slope, SinexNumericField::Slope)?;
    let slope_sigma = spell(record.slope_sigma, SinexNumericField::SlopeSigma)?;
    let line = format!(
        " {:<4} {:<4} {:<3} {:<9} {:<4} {:<4} {:<14} {:<14} {:<4} {:>21} {:>11} {:>21} {:>11}",
        record.kind.label(),
        svn,
        prn,
        station,
        record.obs1,
        obs2,
        start,
        end,
        record.unit.label(),
        value,
        sigma,
        slope,
        slope_sigma,
    );
    Ok(line.trim_end().to_string())
}

fn target_fields(record: &BiasRecord) -> (String, String, String) {
    let svn = record.svn.clone().unwrap_or_default();
    match &record.target {
        BiasTarget::System(system) => (svn, system.letter().to_string(), String::new()),
        BiasTarget::Satellite(sat) => (svn, sat.to_string(), String::new()),
        BiasTarget::Receiver { system, station } => {
            (svn, system.letter().to_string(), station.clone())
        }
        BiasTarget::SatelliteReceiver { sat, station } => (svn, sat.to_string(), station.clone()),
    }
}

/// Returns the ionosphere-free coefficients for two carrier frequencies.
///
/// For finite positive unequal frequencies, the result is
/// `(f1² / (f1² - f2²), -f2² / (f1² - f2²))`. Invalid frequencies and an equal
/// frequency pair return `None`.
pub fn ionosphere_free_coefficients(f1_hz: f64, f2_hz: f64) -> Option<(f64, f64)> {
    validate::finite_positive(f1_hz, "f1_hz").ok()?;
    validate::finite_positive(f2_hz, "f2_hz").ok()?;
    let f1_2 = f1_hz * f1_hz;
    let f2_2 = f2_hz * f2_hz;
    let denom = f1_2 - f2_2;
    if denom == 0.0 {
        return None;
    }
    let alpha = f1_2 / denom;
    let beta = -f2_2 / denom;
    Some((alpha, beta))
}

/// Splits input into physical lines, keeping each line's exact bytes when it
/// is not valid UTF-8 and its line ending.
fn split_source_lines(input: &[u8]) -> Vec<BiasSourceLine> {
    let mut lines = Vec::new();
    let mut start = 0;
    while start < input.len() {
        let (end, next, terminator) = match input[start..].iter().position(|&b| b == b'\n') {
            Some(offset) => {
                let newline = start + offset;
                if newline > start && input[newline - 1] == b'\r' {
                    (newline - 1, newline + 1, BiasLineTerminator::CrLf)
                } else {
                    (newline, newline + 1, BiasLineTerminator::Lf)
                }
            }
            None => (input.len(), input.len(), BiasLineTerminator::None),
        };
        let raw = &input[start..end];
        let (text, bytes) = match std::str::from_utf8(raw) {
            Ok(text) => (text.to_string(), None),
            Err(_) => (
                String::from_utf8_lossy(raw).into_owned(),
                Some(raw.to_vec()),
            ),
        };
        lines.push(BiasSourceLine {
            number: lines.len() + 1,
            text,
            bytes,
            terminator,
            role: BiasLineRole::Blank,
        });
        start = next;
    }
    lines
}

fn invalid_utf8_notices(lines: &[BiasSourceLine]) -> Vec<BiasNotice> {
    lines
        .iter()
        .filter(|line| line.bytes.is_some())
        .map(|line| BiasNotice::InvalidUtf8 { line: line.number })
        .collect()
}

/// Reads the header line from its exact bytes, at the section 4.1 byte
/// columns when the line keeps to them. The marker and a version token are
/// required under either policy; the returned departures name a version
/// other than `1.00` and the first departure from the section 4.1 layout, if
/// any. A byte that is not UTF-8 decodes to a replacement character in its
/// own field only.
fn read_sinex_header_line(raw: &[u8]) -> Result<(BiasSinexHeader, Vec<BiasDeparture>), BiasError> {
    let trimmed = raw.trim_ascii_end();
    let mut tokens = trimmed
        .split(u8::is_ascii_whitespace)
        .filter(|token| !token.is_empty())
        .map(|token| String::from_utf8_lossy(token).into_owned());
    let Some(marker) = tokens.next() else {
        return Err(BiasError::InvalidInput {
            field: "header",
            reason: "missing marker",
        });
    };
    if marker != "%=BIA" {
        return Err(BiasError::InvalidInput {
            field: "header",
            reason: "missing %=BIA marker",
        });
    }
    let version = tokens.next().ok_or(BiasError::InvalidInput {
        field: "version",
        reason: "missing",
    })?;
    let mut departures = Vec::new();
    if version != BIAS_SINEX_VERSION {
        departures.push(BiasDeparture::OtherVersion {
            version: version.clone(),
        });
    }
    let in_columns = header_in_columns(trimmed);
    let fields: Vec<Option<String>> = if in_columns {
        SINEX_HEADER_FIELDS[2..]
            .iter()
            .map(|&(start, end)| byte_field(trimmed, start, end))
            .collect()
    } else {
        tokens.map(Some).collect()
    };
    let field_at = |index: usize| fields.get(index).cloned().flatten();
    let header = BiasSinexHeader {
        version,
        file_agency: field_at(0),
        creation_time: field_at(1),
        data_agency: field_at(2),
        start: field_at(3),
        end: field_at(4),
        mode: field_at(5),
        estimate_count: field_at(6),
    };
    if let Some(reason) = sinex_header_departure(trimmed, in_columns, &header) {
        departures.push(BiasDeparture::HeaderLayout { reason });
    }
    Ok((header, departures))
}

/// Whether every non-blank byte of the header line falls inside one of the
/// section 4.1 fields.
fn header_in_columns(line: &[u8]) -> bool {
    line.iter().enumerate().all(|(index, &byte)| {
        byte == b' '
            || SINEX_HEADER_FIELDS
                .iter()
                .any(|&(start, end)| (start..end).contains(&index))
    })
}

fn sinex_header_departure(
    trimmed: &[u8],
    in_columns: bool,
    header: &BiasSinexHeader,
) -> Option<&'static str> {
    if trimmed.len() != SINEX_HEADER_COLUMNS {
        return Some("header line is not 74 columns");
    }
    if !in_columns || trimmed.get(6..10) != Some(header.version.as_bytes()) {
        return Some("header fields are not in their columns");
    }
    let agency_ok = |value: &Option<String>| value.as_deref().is_some_and(|v| !v.is_empty());
    if !agency_ok(&header.file_agency) || !agency_ok(&header.data_agency) {
        return Some("missing agency code");
    }
    let epoch_ok = |value: &Option<String>| {
        value
            .as_deref()
            .is_some_and(|v| v.len() == 14 && BiasEpoch::parse_sinex(v).is_ok())
    };
    if !epoch_ok(&header.creation_time) || !epoch_ok(&header.start) || !epoch_ok(&header.end) {
        return Some("malformed time field");
    }
    if !matches!(header.mode.as_deref(), Some("A" | "R")) {
        return Some("bias mode is not A or R");
    }
    let count_ok = header
        .estimate_count
        .as_deref()
        .is_some_and(|v| v.len() == 8 && v.bytes().all(|b| b.is_ascii_digit()));
    if !count_ok {
        return Some("number of estimates is not eight digits");
    }
    None
}

/// Reads the body of a Bias-SINEX file one line at a time.
struct SinexBodyReader {
    departures: Vec<BiasDeparture>,
    diagnostics: Diagnostics,
    header: BiasSetHeader,
    records: Vec<BiasRecord>,
    open_block: Option<(String, usize)>,
    seen_blocks: BTreeSet<String>,
    footer_line: Option<usize>,
    solution_rows: usize,
    solution_suffix: Option<(usize, usize)>,
}

impl SinexBodyReader {
    /// Reads one line. `text` is the decoded line and `raw` its exact bytes;
    /// data rows are read from `raw`, at their byte columns.
    fn read_line(&mut self, number: usize, text: &str, raw: &[u8]) -> BiasLineRole {
        let trimmed = text.trim_end();
        // Section 4.1: the footer is the last line, so any line after it, a
        // blank one included, departs from the format.
        if self.footer_line.is_some() {
            self.departures
                .push(BiasDeparture::ContentAfterFooter { line: number });
            return BiasLineRole::Outside;
        }
        if trimmed.trim_start().is_empty() {
            return BiasLineRole::Blank;
        }
        if trimmed == "%=ENDBIA" {
            if let Some((name, line)) = self.open_block.take() {
                self.departures
                    .push(BiasDeparture::UnclosedBlock { name, line });
            }
            self.footer_line = Some(number);
            return BiasLineRole::Footer;
        }
        if trimmed.starts_with('%') {
            self.departures
                .push(BiasDeparture::UnexpectedControlLine { line: number });
            return BiasLineRole::Outside;
        }
        if trimmed.starts_with('*') {
            return BiasLineRole::Comment;
        }
        if let Some(rest) = trimmed.strip_prefix('+') {
            let mut parts = rest.split_whitespace();
            let name = parts.next().unwrap_or("").to_string();
            if let Some((open, _)) = &self.open_block {
                self.departures.push(BiasDeparture::NestedBlock {
                    open: open.clone(),
                    inner: name.clone(),
                    line: number,
                });
            }
            if let Some(suffix) = parts.next() {
                self.departures
                    .push(BiasDeparture::BlockStartSuffix { line: number });
                if name == "BIAS/SOLUTION" {
                    // Earlier versions of this library wrote the record count
                    // here; a lenient read still checks it against the rows.
                    if let Ok(count) = suffix.parse::<usize>() {
                        self.solution_suffix = Some((number, count));
                    }
                }
            }
            if !SINEX_BLOCKS.contains(&name.as_str()) {
                self.diagnostics.push_skip(Skip {
                    at: RecordRef::at_line(number),
                    reason: SkipReason::UnknownBlock(name.clone()),
                });
                self.departures.push(BiasDeparture::UnknownBlock {
                    name: name.clone(),
                    line: number,
                });
            }
            self.seen_blocks.insert(name.clone());
            self.open_block = Some((name, number));
            return BiasLineRole::BlockStart;
        }
        if let Some(rest) = trimmed.strip_prefix('-') {
            let name = rest.split_whitespace().next().unwrap_or("").to_string();
            match self.open_block.take() {
                None => self
                    .departures
                    .push(BiasDeparture::UnopenedBlockEnd { name, line: number }),
                Some((open, _)) if open != name => {
                    self.departures.push(BiasDeparture::MismatchedBlockEnd {
                        open,
                        close: name,
                        line: number,
                    })
                }
                Some(_) => {}
            }
            return BiasLineRole::BlockEnd;
        }
        let Some((block, _)) = &self.open_block else {
            self.departures
                .push(BiasDeparture::DataOutsideBlock { line: number });
            return BiasLineRole::Outside;
        };
        match block.as_str() {
            "FILE/REFERENCE" => {
                self.header.file_reference.push(parse_info_row(raw, number));
                BiasLineRole::FileReference(self.header.file_reference.len() - 1)
            }
            "BIAS/DESCRIPTION" => {
                self.header.description.push(parse_info_row(raw, number));
                BiasLineRole::Description(self.header.description.len() - 1)
            }
            "BIAS/SOLUTION" => {
                self.solution_rows += 1;
                match parse_solution_line(raw) {
                    Ok(mut record) => {
                        record.line = Some(number);
                        self.records.push(record);
                        BiasLineRole::Record(self.records.len() - 1)
                    }
                    Err(reason) => {
                        self.diagnostics.push_skip(Skip {
                            at: RecordRef::at_line(number),
                            reason,
                        });
                        BiasLineRole::Skipped
                    }
                }
            }
            _ => BiasLineRole::BlockBody,
        }
    }
}

/// Reads a `FILE/REFERENCE` or `BIAS/DESCRIPTION` row from its bytes. Such a
/// row is a keyword and a value separated by blanks, so it is split at
/// whitespace; a byte that is not UTF-8 decodes to a replacement character
/// where it stands.
fn parse_info_row(raw: &[u8], line: usize) -> BiasInfoRow {
    let text = String::from_utf8_lossy(raw);
    let body = text.trim();
    let (keyword, value) = match body.find(char::is_whitespace) {
        Some(split) => (&body[..split], body[split..].trim()),
        None => (body, ""),
    };
    BiasInfoRow {
        keyword: keyword.to_string(),
        value: value.to_string(),
        line,
    }
}

/// Typed metadata `BIAS/DESCRIPTION` rows declare.
struct DescriptionView {
    mode: BiasMode,
    time_scale: Option<TimeScale>,
    clock_reference: ClockReferenceObservables,
}

/// Reads the mode, time scale and clock references the description rows
/// declare, without overwriting: a repeat with the same meaning is noted, and
/// a repeat with another meaning leaves the property undetermined. A missing
/// mandatory declaration, a mode other than `ABSOLUTE` or `RELATIVE`, and a
/// time label section 4.6 does not define are departures.
fn derive_description(
    rows: &[BiasInfoRow],
    diagnostics: &mut Diagnostics,
    notices: &mut Vec<BiasNotice>,
    departures: &mut Vec<BiasDeparture>,
) -> DescriptionView {
    let mut mode: Option<BiasMode> = None;
    let mut mode_undetermined = false;
    let mut mode_rows = 0;
    let mut scale: Option<TimeScale> = None;
    let mut scale_undetermined = false;
    let mut time_rows = 0;
    let mut clock_reference = ClockReferenceObservables::default();
    let mut clock_conflicts: BTreeSet<GnssSystem> = BTreeSet::new();

    for row in rows {
        match row.keyword.as_str() {
            "BIAS_MODE" => {
                mode_rows += 1;
                let token = row.value.split_whitespace().next().unwrap_or("");
                let parsed = match token {
                    "ABSOLUTE" => BiasMode::Absolute,
                    "RELATIVE" => BiasMode::Relative,
                    _ => {
                        diagnostics.push_skip(Skip {
                            at: RecordRef::at_line(row.line),
                            reason: SkipReason::UnsupportedRecordType("BIAS_MODE"),
                        });
                        departures.push(BiasDeparture::UnsupportedBiasMode {
                            line: row.line,
                            label: row.value.clone(),
                        });
                        mode_undetermined = true;
                        continue;
                    }
                };
                match mode {
                    None => mode = Some(parsed),
                    Some(first) if first == parsed => {
                        notices.push(BiasNotice::RepeatedDeclaration {
                            line: row.line,
                            keyword: "BIAS_MODE",
                        })
                    }
                    Some(_) => {
                        notices.push(BiasNotice::ConflictingDeclaration {
                            line: row.line,
                            keyword: "BIAS_MODE",
                        });
                        mode_undetermined = true;
                    }
                }
            }
            "TIME_SYSTEM" => {
                time_rows += 1;
                let parsed = match read_time_system_label(&row.value) {
                    TimeSystemLabel::Standard(parsed) => parsed,
                    TimeSystemLabel::NonStandard(named) => {
                        departures.push(BiasDeparture::NonStandardTimeSystem {
                            line: row.line,
                            label: row.value.clone(),
                        });
                        let Some(parsed) = named else {
                            diagnostics.push_skip(Skip {
                                at: RecordRef::at_line(row.line),
                                reason: SkipReason::UnsupportedRecordType("TIME_SYSTEM"),
                            });
                            scale_undetermined = true;
                            continue;
                        };
                        parsed
                    }
                };
                match scale {
                    None => scale = Some(parsed),
                    Some(first) if first == parsed => {
                        notices.push(BiasNotice::RepeatedDeclaration {
                            line: row.line,
                            keyword: "TIME_SYSTEM",
                        })
                    }
                    Some(_) => {
                        notices.push(BiasNotice::ConflictingDeclaration {
                            line: row.line,
                            keyword: "TIME_SYSTEM",
                        });
                        scale_undetermined = true;
                    }
                }
            }
            "SATELLITE_CLOCK_REFERENCE_OBSERVABLES" => {
                let mut parts = row.value.split_whitespace();
                let system_token = parts.next().unwrap_or("");
                let obs1 = parts.next().unwrap_or("").to_string();
                let obs2 = parts.next().unwrap_or("").to_string();
                let Some(system) = system_token
                    .chars()
                    .next()
                    .and_then(GnssSystem::from_letter)
                else {
                    diagnostics.push_skip(Skip {
                        at: RecordRef::at_line(row.line),
                        reason: SkipReason::MalformedField(FieldError::IntParse {
                            field: "system",
                            value: system_token.to_string(),
                        }),
                    });
                    continue;
                };
                if clock_conflicts.contains(&system) {
                    notices.push(BiasNotice::ConflictingDeclaration {
                        line: row.line,
                        keyword: "SATELLITE_CLOCK_REFERENCE_OBSERVABLES",
                    });
                    continue;
                }
                let declared = (obs1, obs2);
                let existing = clock_reference.per_system.get(&system).cloned();
                match existing {
                    None => {
                        clock_reference.per_system.insert(system, declared);
                    }
                    Some(first) if first == declared => {
                        notices.push(BiasNotice::RepeatedDeclaration {
                            line: row.line,
                            keyword: "SATELLITE_CLOCK_REFERENCE_OBSERVABLES",
                        })
                    }
                    Some(_) => {
                        notices.push(BiasNotice::ConflictingDeclaration {
                            line: row.line,
                            keyword: "SATELLITE_CLOCK_REFERENCE_OBSERVABLES",
                        });
                        clock_reference.per_system.remove(&system);
                        clock_conflicts.insert(system);
                    }
                }
            }
            _ => {}
        }
    }
    if mode_rows == 0 {
        departures.push(BiasDeparture::MissingDeclaration {
            keyword: "BIAS_MODE",
        });
    }
    if time_rows == 0 {
        departures.push(BiasDeparture::MissingDeclaration {
            keyword: "TIME_SYSTEM",
        });
    }
    DescriptionView {
        mode: if mode_undetermined {
            BiasMode::Unspecified
        } else {
            mode.unwrap_or_default()
        },
        time_scale: if scale_undetermined { None } else { scale },
        clock_reference,
    }
}

fn parse_bias_sinex_input(
    input: &[u8],
    policy: BiasReadPolicy,
) -> Result<Parsed<BiasSet>, BiasError> {
    let mut lines = split_source_lines(input);
    let Some(first) = lines.first_mut() else {
        return Err(BiasError::InvalidInput {
            field: "input",
            reason: "empty",
        });
    };
    let (sinex_header, header_departures) = read_sinex_header_line(first.raw())?;
    first.role = BiasLineRole::Header;

    let mut reader = SinexBodyReader {
        departures: header_departures,
        diagnostics: Diagnostics::new(),
        header: BiasSetHeader {
            sinex: Some(sinex_header),
            ..BiasSetHeader::default()
        },
        records: Vec::new(),
        open_block: None,
        seen_blocks: BTreeSet::new(),
        footer_line: None,
        solution_rows: 0,
        solution_suffix: None,
    };
    for line in lines.iter_mut().skip(1) {
        let role = reader.read_line(line.number, &line.text, line.raw());
        line.role = role;
    }
    if let Some((name, line)) = reader.open_block.take() {
        reader
            .departures
            .push(BiasDeparture::UnclosedBlock { name, line });
    }
    if reader.footer_line.is_none() {
        reader.departures.push(BiasDeparture::MissingFooter);
    }
    for name in SINEX_MANDATORY_BLOCKS {
        if !reader.seen_blocks.contains(name) {
            reader.departures.push(BiasDeparture::MissingBlock { name });
        }
    }

    let SinexBodyReader {
        mut departures,
        mut diagnostics,
        header,
        records,
        solution_rows,
        solution_suffix,
        ..
    } = reader;
    let mut notices = invalid_utf8_notices(&lines);
    let view = derive_description(
        &header.description,
        &mut diagnostics,
        &mut notices,
        &mut departures,
    );
    if let Some(sinex) = header.sinex.as_ref() {
        if let Some(declared) = sinex.estimate_count_value() {
            if usize::try_from(declared).ok() != Some(solution_rows) {
                departures.push(BiasDeparture::EstimateCountMismatch {
                    declared,
                    solution_rows,
                });
            }
        }
        let header_mode = match sinex.mode.as_deref() {
            Some("A") => Some(BiasMode::Absolute),
            Some("R") => Some(BiasMode::Relative),
            _ => None,
        };
        if let (Some(header_mode), Some(token)) = (header_mode, sinex.mode.as_ref()) {
            if view.mode != BiasMode::Unspecified && header_mode != view.mode {
                departures.push(BiasDeparture::HeaderModeMismatch {
                    header: token.clone(),
                    description: view.mode,
                });
            }
        }
    }

    if policy == BiasReadPolicy::Strict {
        if let Some(departure) = departures.first() {
            return Err(match departure {
                BiasDeparture::OtherVersion { version } => BiasError::UnsupportedVersion {
                    version: version.clone(),
                },
                other => BiasError::Departure {
                    departure: other.clone(),
                },
            });
        }
    }
    for departure in departures {
        diagnostics.push_warning(Warning {
            at: RecordRef::at_line(departure_line(&departure)),
            kind: match departure {
                BiasDeparture::MissingBlock { .. }
                | BiasDeparture::MissingFooter
                | BiasDeparture::MissingDeclaration { .. } => WarningKind::MissingMetadata,
                _ => WarningKind::Mismatch,
            },
        });
        notices.push(BiasNotice::Departure(departure));
    }
    if let Some((line, declared)) = solution_suffix {
        if declared != records.len() {
            diagnostics.push_warning(Warning {
                at: RecordRef::at_line(line),
                kind: WarningKind::Mismatch,
            });
        }
    }
    for record in &records {
        let mismatched = match view.mode {
            BiasMode::Absolute => record.kind != BiasKind::Osb,
            BiasMode::Relative => record.kind == BiasKind::Osb,
            BiasMode::Unspecified => false,
        };
        if mismatched {
            diagnostics.push_warning(Warning {
                at: RecordRef::at_line(record.line.unwrap_or(1)),
                kind: WarningKind::Mismatch,
            });
        }
    }

    let mut set = BiasSet::new(
        records,
        view.mode,
        view.time_scale,
        view.clock_reference,
        header,
        diagnostics,
    );
    notices.append(&mut set.notices);
    set.notices = notices;
    set.lines = lines;
    set.source_format = Some(BiasSourceFormat::BiasSinex);
    let diagnostics = set.diagnostics.clone();
    Ok(Parsed::new(set, diagnostics))
}

fn departure_line(departure: &BiasDeparture) -> usize {
    match departure {
        BiasDeparture::ContentAfterFooter { line }
        | BiasDeparture::UnexpectedControlLine { line }
        | BiasDeparture::UnclosedBlock { line, .. }
        | BiasDeparture::UnopenedBlockEnd { line, .. }
        | BiasDeparture::MismatchedBlockEnd { line, .. }
        | BiasDeparture::NestedBlock { line, .. }
        | BiasDeparture::UnknownBlock { line, .. }
        | BiasDeparture::BlockStartSuffix { line }
        | BiasDeparture::DataOutsideBlock { line }
        | BiasDeparture::UnsupportedBiasMode { line, .. }
        | BiasDeparture::UnknownDcbTimeSystem { line, .. }
        | BiasDeparture::NonStandardTimeSystem { line, .. } => *line,
        BiasDeparture::HeaderLayout { .. }
        | BiasDeparture::OtherVersion { .. }
        | BiasDeparture::MissingFooter
        | BiasDeparture::MissingBlock { .. }
        | BiasDeparture::MissingDeclaration { .. }
        | BiasDeparture::HeaderModeMismatch { .. }
        | BiasDeparture::EstimateCountMismatch { .. } => 1,
    }
}

fn parse_code_dcb_input(
    input: &[u8],
    options: Option<CodeDcbOptions>,
    policy: BiasReadPolicy,
) -> Result<Parsed<BiasSet>, BiasError> {
    let mut lines = split_source_lines(input);
    let title = parse_dcb_title_metadata(&lines);
    let mut notices = invalid_utf8_notices(&lines);
    let mismatch = BiasError::InvalidInput {
        field: "CodeDcbOptions",
        reason: "does not match title metadata",
    };
    // The options when given, or the title's pair and month; and the time
    // scale, `None` when a lenient read meets a label naming none.
    let (options, time_scale) = match (options, title) {
        (Some(options), Some(title)) => {
            if options.pair != title.options.pair
                || options.year != title.options.year
                || options.month != title.options.month
            {
                return Err(mismatch);
            }
            match title.scale {
                // Only a label the writer states contradicts the options.
                DcbTitleScale::Stated(scale) if scale != options.time_scale => {
                    return Err(mismatch);
                }
                // A constellation name is not a statement of the scale, so
                // the options decide, agreeing or not.
                DcbTitleScale::Alias(_, label) => notices.push(BiasNotice::DcbTimeSystemAlias {
                    line: title.line,
                    label,
                }),
                // The title states no scale this reader knows, so the
                // options decide.
                DcbTitleScale::Unknown(label) => {
                    notices.push(BiasNotice::Departure(BiasDeparture::UnknownDcbTimeSystem {
                        line: title.line,
                        label,
                    }))
                }
                DcbTitleScale::Stated(_) | DcbTitleScale::Unstated => {}
            }
            let scale = options.time_scale;
            (options, Some(scale))
        }
        (Some(options), None) => {
            let scale = options.time_scale;
            (options, Some(scale))
        }
        (None, Some(title)) => {
            let scale = match title.scale {
                DcbTitleScale::Stated(scale) => Some(scale),
                DcbTitleScale::Alias(scale, label) => {
                    notices.push(BiasNotice::DcbTimeSystemAlias {
                        line: title.line,
                        label,
                    });
                    Some(scale)
                }
                DcbTitleScale::Unstated => {
                    notices.push(BiasNotice::DcbTimeSystemAssumed);
                    Some(TimeScale::Gpst)
                }
                DcbTitleScale::Unknown(label) => {
                    let departure = BiasDeparture::UnknownDcbTimeSystem {
                        line: title.line,
                        label,
                    };
                    if policy == BiasReadPolicy::Strict {
                        return Err(BiasError::Departure { departure });
                    }
                    notices.push(BiasNotice::Departure(departure));
                    None
                }
            };
            let options = CodeDcbOptions {
                time_scale: scale.unwrap_or(TimeScale::Gpst),
                ..title.options
            };
            (options, scale)
        }
        (None, None) => return Err(BiasError::MissingDcbMetadata),
    };
    validate_dcb_options(&options)?;

    let mut diagnostics = Diagnostics::new();
    let (valid_from, valid_until) = dcb_month_interval(options.year, options.month)?;
    let raw_epochs = (
        valid_from.format_sinex(),
        valid_until
            .map(BiasEpoch::format_sinex)
            .unwrap_or_else(|| "0000:000:00000".to_string()),
    );
    let mut records = Vec::new();
    for line in &mut lines {
        let line_number = line.number;
        if line.text.trim().is_empty() {
            line.role = BiasLineRole::Blank;
            continue;
        }
        line.role = BiasLineRole::Text;
        // Columns are byte offsets, so rows are read from the exact bytes.
        let raw = line.raw().to_vec();
        if is_known_dcb_header_or_comment(&raw) || !looks_like_dcb_row(&raw, &options) {
            continue;
        }
        let row = match parse_dcb_row(&raw, &options) {
            Ok(Some(row)) => row,
            Ok(None) => continue,
            Err(reason) => {
                diagnostics.push_skip(Skip {
                    at: RecordRef::at_line(line_number),
                    reason,
                });
                line.role = BiasLineRole::Skipped;
                continue;
            }
        };
        let Some((obs1, obs2)) = map_legacy_dcb_pair(row.system, &options.pair.0, &options.pair.1)
        else {
            diagnostics.push_skip(Skip {
                at: RecordRef::at_line(line_number),
                reason: SkipReason::UnsupportedRecordType("DCB_PAIR"),
            });
            line.role = BiasLineRole::Skipped;
            continue;
        };
        records.push(BiasRecord {
            kind: BiasKind::Dsb,
            target: row.target,
            svn: None,
            obs1,
            obs2: Some(obs2),
            valid_from: Some(valid_from),
            valid_until,
            raw_epochs: raw_epochs.clone(),
            value: row.value_ns * NS_TO_S,
            sigma: row.sigma_ns.map(|sigma| sigma * NS_TO_S),
            slope: None,
            slope_sigma: None,
            family: BiasObservableFamily::Code,
            unit: BiasUnit::Nanoseconds,
            line: Some(line_number),
        });
        line.role = BiasLineRole::Record(records.len() - 1);
    }

    // Without a time scale the product has no complete DCB metadata.
    let header = BiasSetHeader {
        dcb_meta: time_scale.map(|_| options.clone()),
        ..BiasSetHeader::default()
    };
    let mut set = BiasSet::new(
        records,
        BiasMode::Relative,
        time_scale,
        ClockReferenceObservables::default(),
        header,
        diagnostics,
    );
    notices.append(&mut set.notices);
    set.notices = notices;
    set.lines = lines;
    set.source_format = Some(BiasSourceFormat::CodeDcb);
    let diagnostics = set.diagnostics.clone();
    Ok(Parsed::new(set, diagnostics))
}

struct DcbRow {
    target: BiasTarget,
    system: GnssSystem,
    value_ns: f64,
    sigma_ns: Option<f64>,
}

/// Bytes of a fixed-column field, clamped to the line. Columns are byte
/// offsets, so a byte that is not UTF-8 shifts no other field.
fn column_bytes(line: &[u8], start: usize, end: usize) -> &[u8] {
    let end = end.min(line.len());
    let start = start.min(end);
    &line[start..end]
}

/// A fixed-column field read at its byte columns, decoded and trimmed;
/// `None` when blank. A byte that is not UTF-8 decodes to a replacement
/// character in this field only.
fn byte_field(line: &[u8], start: usize, end: usize) -> Option<String> {
    let text = String::from_utf8_lossy(column_bytes(line, start, end));
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// A fixed-column field read at its byte columns, decoded but not trimmed.
fn byte_raw_field(line: &[u8], start: usize, end: usize) -> String {
    String::from_utf8_lossy(column_bytes(line, start, end)).into_owned()
}

fn system_letter_field(line: &[u8]) -> Option<GnssSystem> {
    byte_field(line, 0, 1).and_then(|token| token.chars().next().and_then(GnssSystem::from_letter))
}

fn satellite_field(line: &[u8]) -> Option<String> {
    byte_field(line, 0, 4).filter(|token| looks_like_satellite_token(token))
}

fn dcb_numeric(line: &[u8]) -> Result<(f64, Option<f64>), SkipReason> {
    let value_token =
        byte_field(line, 24, 38).ok_or(SkipReason::MalformedField(FieldError::Missing {
            field: "dcb value",
        }))?;
    let value_ns = strict_f64(&value_token, "dcb value").map_err(|_| {
        SkipReason::MalformedField(FieldError::FloatParse {
            field: "dcb value",
            value: value_token.clone(),
        })
    })?;
    let sigma_ns = match byte_field(line, 38, 50) {
        Some(token) => Some(strict_f64(&token, "dcb sigma").map_err(SkipReason::MalformedField)?),
        None => None,
    };
    Ok((value_ns, sigma_ns))
}

fn parse_dcb_row(line: &[u8], options: &CodeDcbOptions) -> Result<Option<DcbRow>, SkipReason> {
    if is_known_dcb_header_or_comment(line) {
        return Ok(None);
    }

    let sat_candidate = satellite_field(line);
    let explicit_system =
        system_letter_field(line).filter(|_| byte_raw_field(line, 1, 6).trim().is_empty());

    if sat_candidate.is_none() && explicit_system.is_none() {
        if let Some(system) = options.receiver_system {
            let Some(station) =
                receiver_station_candidate(line).filter(|s| looks_like_station_target(s))
            else {
                return Ok(None);
            };
            let (value_ns, sigma_ns) = dcb_numeric(line)?;
            return Ok(Some(DcbRow {
                target: BiasTarget::Receiver { system, station },
                system,
                value_ns,
                sigma_ns,
            }));
        }
        return Ok(None);
    }

    let (value_ns, sigma_ns) = dcb_numeric(line)?;

    if let Some(sat_token) = sat_candidate {
        let sat = sat_token
            .parse::<GnssSatelliteId>()
            .map_err(|_| SkipReason::UnrepresentableSatellite)?;
        return Ok(Some(DcbRow {
            target: BiasTarget::Satellite(sat),
            system: sat.system,
            value_ns,
            sigma_ns,
        }));
    }

    if let Some(system) = explicit_system {
        let station = byte_field(line, 6, 22).ok_or(SkipReason::InconsistentRecord(
            "receiver DCB record lacks station",
        ))?;
        return Ok(Some(DcbRow {
            target: BiasTarget::Receiver { system, station },
            system,
            value_ns,
            sigma_ns,
        }));
    }

    Err(SkipReason::InconsistentRecord(
        "receiver DCB record lacks system",
    ))
}

fn is_known_dcb_header_or_comment(line: &[u8]) -> bool {
    let text = String::from_utf8_lossy(line);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed
        .chars()
        .all(|c| matches!(c, '-' | '*' | '=' | '.' | ' '))
    {
        return true;
    }
    let upper_0_6 = byte_raw_field(line, 0, 6).to_ascii_uppercase();
    let upper_6_24 = byte_raw_field(line, 6, 24).to_ascii_uppercase();
    if upper_0_6.contains("PRN") && upper_6_24.contains("STATION NAME") {
        return true;
    }
    let has_numeric =
        byte_field(line, 24, 38).is_some_and(|token| strict_f64(&token, "dcb value").is_ok());
    if (trimmed.starts_with('#')
        || trimmed.starts_with('*')
        || trimmed.starts_with('=')
        || trimmed.starts_with('-'))
        && !has_numeric
        && !byte_raw_field(line, 0, 6).trim().is_empty()
    {
        return true;
    }
    let upper = trimmed.to_ascii_uppercase();
    if upper_0_6.starts_with("CODE'S") && upper.contains("SOLUTION") && !has_numeric {
        return true;
    }
    if byte_raw_field(line, 0, 12)
        .to_ascii_uppercase()
        .starts_with("DIFFERENTIAL")
        && (upper.contains("BIAS") || upper.contains("CODE"))
        && !has_numeric
    {
        return true;
    }
    false
}

fn receiver_station_candidate(line: &[u8]) -> Option<String> {
    if byte_raw_field(line, 0, 6).trim().is_empty() {
        byte_field(line, 6, 22)
    } else if satellite_field(line).is_none()
        && !(system_letter_field(line).is_some() && byte_raw_field(line, 1, 6).trim().is_empty())
        && byte_field(line, 24, 38).is_some_and(|token| strict_f64(&token, "dcb value").is_ok())
    {
        byte_field(line, 0, 22)
    } else {
        None
    }
}

fn looks_like_station_target(s: &str) -> bool {
    let trimmed = s.trim();
    if trimmed.is_empty() || trimmed.len() > 16 {
        return false;
    }
    trimmed.is_ascii() && trimmed.chars().all(|c| !c.is_ascii_control())
}

fn looks_like_dcb_row(line: &[u8], options: &CodeDcbOptions) -> bool {
    if is_known_dcb_header_or_comment(line) {
        return false;
    }
    if satellite_field(line).is_some() {
        return true;
    }
    if system_letter_field(line).is_some() && byte_raw_field(line, 1, 6).trim().is_empty() {
        return true;
    }
    if options.receiver_system.is_some() {
        if let Some(candidate) = receiver_station_candidate(line) {
            return looks_like_station_target(&candidate);
        }
    }
    false
}

/// Reads one `BIAS/SOLUTION` data row at the section 4.8 byte columns.
fn parse_solution_line(line: &[u8]) -> Result<BiasRecord, SkipReason> {
    let line = line.trim_ascii_end();
    if line.len() < 91 {
        return Err(SkipReason::Truncated);
    }
    let kind = byte_field(line, 1, 5)
        .ok_or(SkipReason::Truncated)?
        .parse::<BiasKind>()
        .map_err(|_| SkipReason::UnsupportedRecordType("BIAS"))?;
    let svn = byte_field(line, 6, 10);
    let prn = byte_field(line, 11, 14);
    let station = byte_field(line, 15, 24);
    let obs1 = byte_field(line, 25, 29).ok_or(SkipReason::Truncated)?;
    let obs2 = byte_field(line, 30, 34);
    let raw_start = byte_raw_field(line, 35, 49).trim().to_string();
    let raw_end = byte_raw_field(line, 50, 64).trim().to_string();
    let unit_token = byte_field(line, 65, 69).ok_or(SkipReason::Truncated)?;
    let value = byte_field(line, 70, 91)
        .and_then(|token| strict_f64(&token, "bias value").ok())
        .ok_or_else(|| {
            SkipReason::MalformedField(FieldError::FloatParse {
                field: "bias value",
                value: byte_raw_field(line, 70, 91).trim().to_string(),
            })
        })?;
    let parse_optional_f64 =
        |start: usize, end: usize, name: &'static str| -> Result<Option<f64>, SkipReason> {
            match byte_field(line, start, end) {
                Some(s) => strict_f64(&s, name)
                    .map(Some)
                    .map_err(SkipReason::MalformedField),
                None => Ok(None),
            }
        };
    let sigma = parse_optional_f64(92, 103, "bias sigma")?;
    let slope = parse_optional_f64(104, 125, "bias slope")?;
    let slope_sigma = parse_optional_f64(126, 137, "bias slope sigma")?;
    let unit = BiasUnit::parse(&unit_token)
        .ok_or_else(|| SkipReason::UnsupportedUnit(unit_token.clone()))?;
    if kind == BiasKind::Osb && obs2.is_some() {
        return Err(SkipReason::InconsistentRecord("OSB must not carry OBS2"));
    }
    if kind != BiasKind::Osb && obs2.is_none() {
        return Err(SkipReason::InconsistentRecord("DSB or ISB requires OBS2"));
    }
    // Section 4.8: the observable code, not the unit, says whether a bias is
    // a code or a phase bias. A DSB or ISB between a code and a phase
    // observable is kept as a mixed record.
    let family = BiasObservableFamily::of_observable(&obs1)
        .ok_or(SkipReason::UnsupportedRecordType("OBSERVABLE"))?;
    let family = match &obs2 {
        Some(obs2) => {
            let second = BiasObservableFamily::of_observable(obs2)
                .ok_or(SkipReason::UnsupportedRecordType("OBSERVABLE"))?;
            if second == family {
                family
            } else {
                BiasObservableFamily::Mixed
            }
        }
        None => family,
    };
    // Section 4.8: the unit has to be ns for code biases; only phase biases
    // may be given in cycles.
    if family == BiasObservableFamily::Code && unit != BiasUnit::Nanoseconds {
        return Err(SkipReason::InconsistentRecord(
            "code bias is not stated in ns",
        ));
    }
    let valid_from = BiasEpoch::parse_sinex(&raw_start).map_err(|_| {
        SkipReason::MalformedField(FieldError::IntParse {
            field: "bias start",
            value: raw_start.clone(),
        })
    })?;
    let valid_until = BiasEpoch::parse_sinex(&raw_end)
        .map_err(|_| {
            SkipReason::MalformedField(FieldError::IntParse {
                field: "bias end",
                value: raw_end.clone(),
            })
        })?
        .map(|epoch| epoch.normalize_end())
        .transpose()
        .map_err(|_| {
            SkipReason::MalformedField(FieldError::IntParse {
                field: "bias end",
                value: raw_end.clone(),
            })
        })?;
    let target = parse_bias_target(prn.as_deref(), station.as_deref())?;
    let decode = |field: SinexNumericField, stated: f64| field.decode(unit, stated);
    Ok(BiasRecord {
        kind,
        target,
        svn,
        obs1,
        obs2,
        valid_from,
        valid_until,
        raw_epochs: (raw_start, raw_end),
        value: decode(SinexNumericField::Value, value),
        sigma: sigma.map(|sigma| decode(SinexNumericField::Sigma, sigma)),
        slope: slope.map(|slope| decode(SinexNumericField::Slope, slope)),
        slope_sigma: slope_sigma
            .map(|slope_sigma| decode(SinexNumericField::SlopeSigma, slope_sigma)),
        family,
        unit,
        line: None,
    })
}

/// Builds the target of a solution row. The station field is kept as the
/// file states it.
fn parse_bias_target(prn: Option<&str>, station: Option<&str>) -> Result<BiasTarget, SkipReason> {
    let system_of = |prn: &str| {
        prn.chars()
            .next()
            .and_then(GnssSystem::from_letter)
            .ok_or(SkipReason::MalformedField(FieldError::IntParse {
                field: "system",
                value: prn.to_string(),
            }))
    };
    match (prn, station) {
        (Some(prn), None) if prn.len() == 1 => Ok(BiasTarget::System(system_of(prn)?)),
        (Some(prn), Some(station)) if prn.len() == 1 => Ok(BiasTarget::Receiver {
            system: system_of(prn)?,
            station: station.to_string(),
        }),
        (Some(prn), None) => prn
            .parse::<GnssSatelliteId>()
            .map(BiasTarget::Satellite)
            .map_err(|_| SkipReason::UnrepresentableSatellite),
        (Some(prn), Some(station)) => prn
            .parse::<GnssSatelliteId>()
            .map(|sat| BiasTarget::SatelliteReceiver {
                sat,
                station: station.to_string(),
            })
            .map_err(|_| SkipReason::UnrepresentableSatellite),
        (None, Some(_)) | (None, None) => Err(SkipReason::InconsistentRecord("missing PRN")),
    }
}

/// What a CODE DCB title states about its time system.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DcbTitleScale {
    /// A label a generated title states.
    Stated(TimeScale),
    /// A constellation name that names a scale unambiguously, such as `GPS`.
    Alias(TimeScale, String),
    /// A label naming no scale this reader knows.
    Unknown(String),
    /// No label: a generated title without one, or a prose title.
    Unstated,
}

/// A CODE DCB title: its pair and month (with a placeholder time scale),
/// what it states about its time system, and its line.
struct DcbTitle {
    options: CodeDcbOptions,
    scale: DcbTitleScale,
    line: usize,
}

/// Reads the DCB title.
///
/// A title in the shape the writer generates, `# DCB <pair> <YYYY-M[M]>
/// <label>`, states its time system in the label. A prose title, such as
/// CODE's, states no time system, so no token of it is read as one.
fn parse_dcb_title_metadata(lines: &[BiasSourceLine]) -> Option<DcbTitle> {
    for line in lines.iter().take(12) {
        if let Some((options, scale)) = generated_dcb_title(&line.text) {
            return Some(DcbTitle {
                options,
                scale,
                line: line.number,
            });
        }
        let mut pair = None;
        let mut year = None;
        let mut month = None;
        let tokens = line
            .text
            .split_whitespace()
            .map(|token| token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-'))
            .collect::<Vec<_>>();
        for (index, token) in tokens.iter().enumerate() {
            if let Some((a, b)) = token.split_once('-') {
                if is_legacy_dcb_label(a) && is_legacy_dcb_label(b) {
                    pair = Some((a.to_string(), b.to_string()));
                }
            }
            if let Some((y, m)) = token.split_once('-') {
                if let (Ok(y), Ok(m)) = (y.parse::<i32>(), m.parse::<u8>()) {
                    year = Some(y);
                    month = Some(m);
                }
            }
            if token.eq_ignore_ascii_case("YEAR") {
                year = tokens
                    .get(index + 1)
                    .and_then(|value| value.parse::<i32>().ok());
            }
            if token.eq_ignore_ascii_case("MONTH") {
                month = tokens
                    .get(index + 1)
                    .and_then(|value| value.parse::<u8>().ok());
            }
        }
        if let (Some(pair), Some(year), Some(month)) = (pair, year, month) {
            return Some(DcbTitle {
                options: CodeDcbOptions {
                    pair,
                    year,
                    month,
                    time_scale: TimeScale::Gpst,
                    receiver_system: None,
                },
                scale: DcbTitleScale::Unstated,
                line: line.number,
            });
        }
    }
    None
}

/// Reads a title in the generated shape `# DCB <pair> <YYYY-M[M]> [<label>]`,
/// or `None` for a line of any other shape.
fn generated_dcb_title(text: &str) -> Option<(CodeDcbOptions, DcbTitleScale)> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let (pair_token, month_token, label) = match tokens.as_slice() {
        ["#", "DCB", pair, month] => (*pair, *month, None),
        ["#", "DCB", pair, month, label] => (*pair, *month, Some(*label)),
        _ => return None,
    };
    let (left, right) = pair_token.split_once('-')?;
    if !is_legacy_dcb_label(left) || !is_legacy_dcb_label(right) {
        return None;
    }
    let (year, month) = month_token.split_once('-')?;
    let digits = |text: &str| text.bytes().all(|byte| byte.is_ascii_digit());
    if year.len() != 4 || !digits(year) || !(1..=2).contains(&month.len()) || !digits(month) {
        return None;
    }
    let (Ok(year), Ok(month)) = (year.parse::<i32>(), month.parse::<u8>()) else {
        return None;
    };
    let scale = match label {
        None => DcbTitleScale::Unstated,
        Some(label) => dcb_title_label(label),
    };
    Some((
        CodeDcbOptions {
            pair: (left.to_string(), right.to_string()),
            year,
            month,
            time_scale: TimeScale::Gpst,
            receiver_system: None,
        },
        scale,
    ))
}

/// Reads the label of a generated DCB title: a label the writer states, a
/// constellation name that names a scale as the lenient Bias-SINEX reader
/// maps it, or an unknown label.
fn dcb_title_label(label: &str) -> DcbTitleScale {
    if let Some(scale) = dcb_title_time_scale(label) {
        return DcbTitleScale::Stated(scale);
    }
    let alias = match label {
        "GPS" | "GPST" => TimeScale::Gpst,
        "GLO" => TimeScale::Utc,
        "GAL" | "GST" => TimeScale::Gst,
        "BDT" => TimeScale::Bdt,
        "QZS" | "QZSST" => TimeScale::Qzsst,
        _ => return DcbTitleScale::Unknown(label.to_string()),
    };
    DcbTitleScale::Alias(alias, label.to_string())
}

fn validate_dcb_options(options: &CodeDcbOptions) -> Result<(), BiasError> {
    if !is_legacy_dcb_label(&options.pair.0) || !is_legacy_dcb_label(&options.pair.1) {
        return Err(BiasError::InvalidInput {
            field: "pair",
            reason: "unknown legacy DCB label",
        });
    }
    if !(1..=12).contains(&options.month) {
        return Err(BiasError::InvalidInput {
            field: "month",
            reason: "out of range",
        });
    }
    // A DCB title states a four-digit year.
    if !(0..=9999).contains(&options.year) {
        return Err(BiasError::InvalidInput {
            field: "year",
            reason: "out of range",
        });
    }
    Ok(())
}

fn dcb_month_interval(year: i32, month: u8) -> Result<(BiasEpoch, Option<BiasEpoch>), BiasError> {
    let start = month_epoch(year, month)?;
    let (next_year, next_month) = if month == 12 {
        let next_year = year.checked_add(1).ok_or(BiasError::InvalidInput {
            field: "year",
            reason: "out of range",
        })?;
        (next_year, 1)
    } else {
        (year, month + 1)
    };
    Ok((start, Some(month_epoch(next_year, next_month)?)))
}

fn month_epoch(year: i32, month: u8) -> Result<BiasEpoch, BiasError> {
    let doy = day_of_year_int(year, i32::from(month), 1);
    BiasEpoch::new(year, doy as u16, 0)
}

fn map_legacy_dcb_pair(system: GnssSystem, left: &str, right: &str) -> Option<(String, String)> {
    let left = map_legacy_dcb_label(system, left)?;
    let right = map_legacy_dcb_label(system, right)?;
    Some((left.to_string(), right.to_string()))
}

fn map_legacy_dcb_label(system: GnssSystem, label: &str) -> Option<&'static str> {
    match (system, label) {
        (GnssSystem::Gps, "P1") => Some("C1W"),
        (GnssSystem::Gps, "P2") => Some("C2W"),
        (GnssSystem::Gps, "C1") => Some("C1C"),
        (GnssSystem::Gps, "C2") => Some("C2C"),
        (GnssSystem::Glonass, "P1") => Some("C1P"),
        (GnssSystem::Glonass, "P2") => Some("C2P"),
        (GnssSystem::Glonass, "C1") => Some("C1C"),
        (GnssSystem::Glonass, "C2") => Some("C2C"),
        (GnssSystem::Galileo, "P1") => Some("C1X"),
        (GnssSystem::Galileo, "P2") => Some("C5X"),
        (GnssSystem::Galileo, "C1") => Some("C1C"),
        (GnssSystem::Galileo, "C2") => Some("C5Q"),
        (GnssSystem::BeiDou, "P1") => Some("C2I"),
        (GnssSystem::BeiDou, "P2") => Some("C6I"),
        (GnssSystem::BeiDou, "C1") => Some("C2I"),
        (GnssSystem::BeiDou, "C2") => Some("C7I"),
        _ => None,
    }
}

fn is_legacy_dcb_label(label: &str) -> bool {
    matches!(label, "P1" | "P2" | "C1" | "C2")
}

/// DSB hops from one observable to another: the parallel records joining
/// each ordered pair of observables.
type DsbGraph = BTreeMap<String, BTreeMap<String, Vec<DsbEdge>>>;

/// Resolves the value between two observables through the fewest DSB hops.
///
/// The shortest routes form a directed acyclic graph over observables,
/// found from the distances to both ends. Parallel records joining two
/// observables of that graph form one hop and must agree, as `hops_agree`
/// compares them. Each observable then gets a representative route from the
/// start: the one first in observable order, found layer by layer. Every
/// route agrees exactly when, for every hop `u -> v` of the graph, the
/// representative route to `u` plus the hop equals the representative route
/// to `v`; with every record's text kept, that is checked as exact decimal
/// sums of stated values and of `value - slope * t_ref` with the stated
/// slopes. For a set without the text, the largest and smallest route sums
/// are carried through the graph and must differ by no more than the
/// rounding bound `(n + 2) * u * 2 * max(sum |terms|)`, where `n` is the
/// number of hops and `u` is 2^-53. On agreement the representative route to
/// the end gives the value, naming every record of its hops; otherwise every
/// record on the graph is named in the conflict. The work is linear in the
/// graph apart from one exact check per hop off a representative route, and
/// nothing recurses.
fn resolve_dsb_path(
    graph: &DsbGraph,
    start: &str,
    end: &str,
    hops_agree: &dyn Fn(&DsbEdge, &DsbEdge) -> bool,
) -> DsbPath {
    let from_start = dsb_distances(graph, start);
    let Some(&length) = from_start.get(end) else {
        return DsbPath::None;
    };
    // Hops are stored in both directions, so distances to the end are
    // distances from it.
    let to_end = dsb_distances(graph, end);
    let on_graph = |node: &str, depth: usize| {
        from_start.get(node) == Some(&depth)
            && to_end.get(node).is_some_and(|rest| depth + rest == length)
    };

    // Layers of the graph and the hops into each observable.
    let mut layers: Vec<Vec<&str>> = vec![Vec::new(); length + 1];
    let mut into: BTreeMap<&str, Vec<(&str, &[DsbEdge])>> = BTreeMap::new();
    for (node, &depth) in &from_start {
        if !on_graph(node.as_str(), depth) {
            continue;
        }
        layers[depth].push(node.as_str());
        if depth == length {
            continue;
        }
        for (next, group) in graph.get(node).into_iter().flatten() {
            if on_graph(next.as_str(), depth + 1) && !group.is_empty() {
                into.entry(next.as_str())
                    .or_default()
                    .push((node.as_str(), group.as_slice()));
            }
        }
    }
    let hops: Vec<&[DsbEdge]> = into
        .values()
        .flat_map(|incoming| incoming.iter().map(|&(_, group)| group))
        .collect();
    let mut all_records: Vec<usize> = hops
        .iter()
        .flat_map(|group| group.iter().map(|edge| edge.record))
        .collect();
    all_records.sort_unstable();
    all_records.dedup();
    let conflict = || DsbPath::Conflict {
        records: all_records.clone(),
    };
    if hops.iter().any(|group| {
        group
            .split_first()
            .is_some_and(|(first, rest)| rest.iter().any(|edge| !hops_agree(first, edge)))
    }) {
        return conflict();
    }
    let exact = hops
        .iter()
        .all(|group| group.iter().all(|edge| edge.affine.is_some()));

    // Representative routes, layer by layer: each observable takes the
    // predecessor whose representative route is first in observable order.
    let mut node_state: BTreeMap<&str, DsbNode> = BTreeMap::new();
    node_state.insert(
        start,
        DsbNode {
            rank: 0,
            previous: None,
            value: 0.0,
            largest: 0.0,
            smallest: 0.0,
            magnitude: 0.0,
        },
    );
    for layer in layers.iter().skip(1) {
        let mut ranked: Vec<(usize, &str)> = Vec::new();
        for &node in layer {
            let incoming = into.get(node).map_or(&[][..], Vec::as_slice);
            let mut best: Option<(usize, &str, &DsbEdge)> = None;
            let mut largest = f64::NEG_INFINITY;
            let mut smallest = f64::INFINITY;
            let mut magnitude: f64 = 0.0;
            for &(from, group) in incoming {
                let (Some(state), Some(edge)) = (node_state.get(from), group.first()) else {
                    continue;
                };
                largest = largest.max(state.largest + edge.value);
                smallest = smallest.min(state.smallest + edge.value);
                magnitude = magnitude.max(state.magnitude + edge.value.abs());
                if best.is_none_or(|(rank, _, _)| state.rank < rank) {
                    best = Some((state.rank, from, edge));
                }
            }
            let Some((rank, from, edge)) = best else {
                continue;
            };
            let value = node_state.get(from).map_or(0.0, |state| state.value) + edge.value;
            node_state.insert(
                node,
                DsbNode {
                    rank: 0,
                    previous: Some((from, edge)),
                    value,
                    largest,
                    smallest,
                    magnitude,
                },
            );
            ranked.push((rank, node));
        }
        // Routes of one layer are ordered by their predecessors' routes,
        // then by their last observable.
        ranked.sort_unstable();
        for (position, (_, node)) in ranked.into_iter().enumerate() {
            if let Some(state) = node_state.get_mut(node) {
                state.rank = position;
            }
        }
    }

    // The representative route to an observable, as its hops from the start.
    let route_to = |node: &str| {
        let mut hops = Vec::new();
        let mut current = node_state.get_key_value(node).map(|(key, _)| *key);
        while let Some(state) = current.and_then(|key| node_state.get(key)) {
            let Some((from, edge)) = state.previous else {
                break;
            };
            hops.push(edge);
            current = Some(from);
        }
        hops.reverse();
        hops
    };
    let stated_route = |node: &str| -> Option<Vec<StatedAffine>> {
        route_to(node)
            .into_iter()
            .map(|edge| edge.affine.clone())
            .collect()
    };

    if exact {
        for (&to, incoming) in &into {
            let Some(representative) = node_state.get(to).and_then(|state| state.previous) else {
                continue;
            };
            for &(from, group) in incoming {
                let Some(edge) = group.first() else {
                    continue;
                };
                if from == representative.0 && std::ptr::eq(edge, representative.1) {
                    continue;
                }
                let (Some(mut through), Some(target), Some(hop)) =
                    (stated_route(from), stated_route(to), edge.affine.clone())
                else {
                    return conflict();
                };
                through.push(hop);
                if affine_sums_equal(&through, &target) != Some(true) {
                    return conflict();
                }
            }
        }
    } else if let Some(state) = node_state.get(end) {
        let unit_roundoff = f64::EPSILON / 2.0;
        let bound = (length as f64 + 2.0) * unit_roundoff * 2.0 * state.magnitude;
        if state.largest - state.smallest > bound {
            return conflict();
        }
    }

    let Some(state) = node_state.get(end) else {
        return DsbPath::None;
    };
    let mut records = Vec::new();
    let mut overridden = Vec::new();
    let mut current = end;
    while let Some((from, _)) = node_state.get(current).and_then(|state| state.previous) {
        if let Some(group) = into
            .get(current)
            .and_then(|incoming| incoming.iter().find(|&&(node, _)| node == from))
            .map(|&(_, group)| group)
        {
            records.extend(group.iter().map(|edge| edge.record));
            overridden.extend(
                group
                    .iter()
                    .flat_map(|edge| edge.overridden.iter().copied()),
            );
        }
        current = from;
    }
    records.sort_unstable();
    records.dedup();
    overridden.sort_unstable();
    overridden.dedup();
    DsbPath::Resolved {
        value: state.value,
        records,
        overridden,
        stated: stated_route(end),
    }
}

/// The state of one observable of the shortest-route graph.
#[derive(Debug, Clone, Copy)]
struct DsbNode<'g> {
    /// Position of its representative route among its layer's, in observable
    /// order.
    rank: usize,
    /// The observable before it on its representative route, and the hop's
    /// representative record.
    previous: Option<(&'g str, &'g DsbEdge)>,
    /// Sum of its representative route's hops, in hop order.
    value: f64,
    /// Largest and smallest sums over all routes reaching it.
    largest: f64,
    smallest: f64,
    /// Largest sum of hop magnitudes over all routes reaching it.
    magnitude: f64,
}

/// Hop counts from one observable to every observable it reaches.
fn dsb_distances(graph: &DsbGraph, from: &str) -> BTreeMap<String, usize> {
    let mut distance = BTreeMap::new();
    let mut queue = VecDeque::new();
    distance.insert(from.to_string(), 0_usize);
    queue.push_back(from.to_string());
    while let Some(node) = queue.pop_front() {
        let next_distance = distance.get(&node).copied().unwrap_or(0) + 1;
        for next in graph.get(&node).into_iter().flat_map(BTreeMap::keys) {
            if !distance.contains_key(next) {
                distance.insert(next.clone(), next_distance);
                queue.push_back(next.clone());
            }
        }
    }
    distance
}

/// A record's bias exactly as its row states it: a value, and for a sloped
/// record its slope and twice its reference epoch in whole seconds.
#[derive(Debug, Clone, PartialEq, Eq)]
struct StatedAffine {
    value: ExactDecimal,
    slope: Option<(ExactDecimal, i64)>,
}

impl StatedAffine {
    fn negated(self) -> Self {
        Self {
            value: self.value.negated(),
            slope: self
                .slope
                .map(|(slope, twice_reference_s)| (slope.negated(), twice_reference_s)),
        }
    }

    /// The constant terms of the bias as a function of time: the value and
    /// `-slope * t_ref`. `None` when the product does not fit.
    fn offset_terms(&self) -> Option<Vec<ExactDecimal>> {
        let mut terms = vec![self.value.clone()];
        if let Some((slope, twice_reference_s)) = &self.slope {
            terms.push(slope.times_half(*twice_reference_s)?.negated());
        }
        Some(terms)
    }
}

/// Whether two sums of stated affine biases are the same function of time,
/// exactly: equal slopes and equal constant terms. `None` when a product does
/// not fit.
fn affine_sums_equal(a: &[StatedAffine], b: &[StatedAffine]) -> Option<bool> {
    let mut slopes = Vec::new();
    let mut offsets = Vec::new();
    for (route, sign) in [(a, false), (b, true)] {
        for hop in route {
            let apply = |term: ExactDecimal| if sign { term.negated() } else { term };
            if let Some((slope, _)) = &hop.slope {
                slopes.push(apply(slope.clone()));
            }
            offsets.extend(hop.offset_terms()?.into_iter().map(apply));
        }
    }
    Some(ExactDecimal::sum_is_zero(&slopes) && ExactDecimal::sum_is_zero(&offsets))
}

/// A decimal number exactly as a row states it: `mantissa * 10^exponent`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ExactDecimal {
    negative: bool,
    mantissa: u128,
    exponent: i64,
}

impl ExactDecimal {
    /// Reads a stated number: an optional sign, digits with an optional
    /// decimal point, and an optional `E` or `D` exponent. `None` for any
    /// other text or more digits than fit.
    fn parse(text: &str) -> Option<Self> {
        let text = text.trim();
        let (negative, unsigned) = match text.as_bytes().first()? {
            b'-' => (true, &text[1..]),
            b'+' => (false, &text[1..]),
            _ => (false, text),
        };
        let (number, power) = match unsigned.find(['E', 'e', 'D', 'd']) {
            Some(at) => (&unsigned[..at], unsigned[at + 1..].parse::<i64>().ok()?),
            None => (unsigned, 0),
        };
        let (whole, fraction) = number.split_once('.').unwrap_or((number, ""));
        if whole.is_empty() && fraction.is_empty() {
            return None;
        }
        let mut mantissa: u128 = 0;
        for byte in whole.bytes().chain(fraction.bytes()) {
            if !byte.is_ascii_digit() {
                return None;
            }
            mantissa = mantissa
                .checked_mul(10)?
                .checked_add(u128::from(byte - b'0'))?;
        }
        let fraction_digits = i64::try_from(fraction.len()).ok()?;
        Some(Self {
            negative,
            mantissa,
            exponent: power.checked_sub(fraction_digits)?,
        })
    }

    fn negated(self) -> Self {
        Self {
            negative: !self.negative,
            ..self
        }
    }

    /// `self * twice / 2`, exactly: `self * twice * 5 * 10^-1`. `None` when
    /// the product does not fit.
    fn times_half(&self, twice: i64) -> Option<Self> {
        let mantissa = self
            .mantissa
            .checked_mul(u128::from(twice.unsigned_abs()))?
            .checked_mul(5)?;
        Some(Self {
            negative: self.negative != (twice < 0),
            mantissa,
            exponent: self.exponent.checked_sub(1)?,
        })
    }

    /// Whether the terms sum to exactly zero.
    ///
    /// Zero terms are dropped and trailing zeros moved into the exponent,
    /// kept in `i128` so nothing saturates. The terms are then added in
    /// ascending exponent order into a signed accumulator counted in units
    /// of the current exponent. Every term at a higher exponent is a multiple
    /// of that exponent's power of ten, so before moving from exponent `e` to
    /// `e'` the accumulator must be divisible by `10^(e' - e)`, or the total
    /// cannot be zero; it is then divided. The total is zero when the
    /// accumulator ends at zero. The accumulator never holds more digits than
    /// one mantissa and the number of terms allow.
    fn sum_is_zero(terms: &[Self]) -> bool {
        let mut terms: Vec<(bool, u128, i128)> = terms
            .iter()
            .filter(|term| term.mantissa != 0)
            .map(|term| {
                let (mut mantissa, mut exponent) = (term.mantissa, i128::from(term.exponent));
                while mantissa.is_multiple_of(10) {
                    mantissa /= 10;
                    exponent += 1;
                }
                (term.negative, mantissa, exponent)
            })
            .collect();
        terms.sort_by_key(|&(_, _, exponent)| exponent);
        let mut accumulator = BigSigned::default();
        let mut current: Option<i128> = None;
        for (negative, mantissa, exponent) in terms {
            if let Some(from) = current {
                if exponent > from && !accumulator.divide_pow10(exponent - from) {
                    return false;
                }
            }
            current = Some(exponent);
            accumulator.add(negative, &BigNatural::from_u128(mantissa));
        }
        accumulator.is_zero()
    }
}

/// A natural number in base 10^9 limbs, least significant first, with no
/// high zero limbs, for exact decimal sums.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct BigNatural {
    limbs: Vec<u32>,
}

impl BigNatural {
    const BASE: u64 = 1_000_000_000;

    fn from_u128(mut value: u128) -> Self {
        let mut limbs = Vec::new();
        while value > 0 {
            limbs.push((value % u128::from(Self::BASE)) as u32);
            value /= u128::from(Self::BASE);
        }
        Self { limbs }
    }

    fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    fn trim(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    fn add(&mut self, other: &Self) {
        let mut carry = 0_u64;
        for index in 0..self.limbs.len().max(other.limbs.len()) {
            let left = self.limbs.get(index).copied().map_or(0, u64::from);
            let right = other.limbs.get(index).copied().map_or(0, u64::from);
            let sum = left + right + carry;
            let digit = (sum % Self::BASE) as u32;
            carry = sum / Self::BASE;
            match self.limbs.get_mut(index) {
                Some(limb) => *limb = digit,
                None => self.limbs.push(digit),
            }
        }
        if carry > 0 {
            self.limbs.push(carry as u32);
        }
    }

    /// `self - other`, for `self >= other`.
    fn sub(&mut self, other: &Self) {
        let mut borrow = 0_i64;
        for (index, limb) in self.limbs.iter_mut().enumerate() {
            let right = other.limbs.get(index).copied().map_or(0, i64::from);
            let mut difference = i64::from(*limb) - right - borrow;
            borrow = 0;
            if difference < 0 {
                difference += Self::BASE as i64;
                borrow = 1;
            }
            *limb = difference as u32;
        }
        self.trim();
    }

    fn cmp_magnitude(&self, other: &Self) -> Ordering {
        self.limbs
            .len()
            .cmp(&other.limbs.len())
            .then_with(|| self.limbs.iter().rev().cmp(other.limbs.iter().rev()))
    }

    /// Divides by `10^power` when it divides exactly, and returns whether it
    /// did. A nonzero number has fewer than `9 * limbs` digits, so a larger
    /// power never divides it.
    fn divide_pow10(&mut self, power: i128) -> bool {
        if self.is_zero() {
            return true;
        }
        let Ok(power) = usize::try_from(power) else {
            return false;
        };
        let whole_limbs = power / 9;
        if whole_limbs >= self.limbs.len() {
            return false;
        }
        if self.limbs[..whole_limbs].iter().any(|&limb| limb != 0) {
            return false;
        }
        self.limbs.drain(..whole_limbs);
        // power % 9 is below 9, so the divisor fits in a u32 and divides
        // 10^9; the lowest limb decides divisibility.
        let divisor = 10_u32.pow((power % 9) as u32);
        if !self.limbs[0].is_multiple_of(divisor) {
            return false;
        }
        let mut remainder = 0_u64;
        for limb in self.limbs.iter_mut().rev() {
            let current = remainder * Self::BASE + u64::from(*limb);
            *limb = (current / u64::from(divisor)) as u32;
            remainder = current % u64::from(divisor);
        }
        self.trim();
        true
    }
}

/// A signed integer over [`BigNatural`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct BigSigned {
    negative: bool,
    magnitude: BigNatural,
}

impl BigSigned {
    fn is_zero(&self) -> bool {
        self.magnitude.is_zero()
    }

    fn add(&mut self, negative: bool, magnitude: &BigNatural) {
        if self.is_zero() || self.negative == negative {
            if self.is_zero() {
                self.negative = negative;
            }
            self.magnitude.add(magnitude);
            return;
        }
        match self.magnitude.cmp_magnitude(magnitude) {
            Ordering::Greater | Ordering::Equal => self.magnitude.sub(magnitude),
            Ordering::Less => {
                let mut larger = magnitude.clone();
                larger.sub(&self.magnitude);
                self.magnitude = larger;
                self.negative = negative;
            }
        }
    }

    fn divide_pow10(&mut self, power: i128) -> bool {
        self.magnitude.divide_pow10(power)
    }
}

/// Epoch as an instant for ordering; an undefined bound is `None`, which
/// orders first.
fn instant_of(epoch: Option<BiasEpoch>) -> Option<i64> {
    epoch.map(BiasEpoch::instant_seconds)
}

fn same_instant(a: Option<BiasEpoch>, b: Option<BiasEpoch>) -> bool {
    instant_of(a) == instant_of(b)
}

/// Orders records by the instant their validity starts. Starts naming the
/// same instant compare equal however they are written.
fn compare_record_start(a: &BiasRecord, b: &BiasRecord) -> Ordering {
    instant_of(a.valid_from).cmp(&instant_of(b.valid_from))
}

fn intervals_overlap(a: &BiasRecord, b: &BiasRecord) -> bool {
    let a_start = instant_of(a.valid_from);
    let b_start = instant_of(b.valid_from);
    let a_end = instant_of(a.valid_until);
    let b_end = instant_of(b.valid_until);
    let a_before_b_end = b_end.is_none_or(|end| a_start.is_none_or(|start| start < end));
    let b_before_a_end = a_end.is_none_or(|end| b_start.is_none_or(|start| start < end));
    a_before_b_end && b_before_a_end
}

/// Index key of a record's bias kind and observables. The kind is part of
/// the key, so an ISB and a DSB on the same pair, which are different
/// quantities, never overlap each other.
fn obs_key(record: &BiasRecord) -> String {
    index_key(record.kind, &record.obs1, record.obs2.as_deref())
}

fn index_key(kind: BiasKind, obs1: &str, obs2: Option<&str>) -> String {
    match kind {
        BiasKind::Osb => format!("{} {obs1}", kind.label()),
        BiasKind::Dsb | BiasKind::Isb => {
            format!("{} {obs1}-{}", kind.label(), obs2.unwrap_or(""))
        }
    }
}

/// Lookup form of a station identifier: trimmed and uppercased, never
/// shortened.
fn normalize_station(station: &str) -> String {
    station.trim().to_ascii_uppercase()
}

/// The first four characters of a station identifier when they are ASCII
/// letters or digits.
fn station_marker(station: &str) -> Option<&str> {
    let bytes = station.as_bytes();
    if bytes.len() >= 4 && bytes[..4].iter().all(u8::is_ascii_alphanumeric) {
        station.get(..4)
    } else {
        None
    }
}

/// The exact seconds since J2000 an instant stands for: a split Julian date
/// as [`exact_j2000_seconds_of_split`] takes it (the label it is the reading
/// of, or the exact time its parts hold), and a nanosecond count from J2000 as
/// that count.
fn instant_exact_seconds(epoch: Instant) -> Option<ExactSeconds> {
    match epoch.repr {
        InstantRepr::JulianDate(split) => {
            exact_j2000_seconds_of_split(split.jd_whole, split.fraction)
        }
        InstantRepr::Nanos(nanos) => Some(ExactSeconds::from_decimal(nanos, 9)),
    }
}

/// Converts a [`BiasEpoch`] to an [`Instant`] on `scale`.
///
/// The calendar components are converted through a validated split Julian
/// date; an invalid conversion returns [`BiasError::InvalidEpoch`].
pub fn bias_epoch_instant(epoch: BiasEpoch, scale: TimeScale) -> Result<Instant, BiasError> {
    Ok(Instant::from_julian_date(scale, epoch.to_split()?))
}

/// Converts a civil date-time to an [`Instant`] on `scale`.
///
/// UTC and GLONASST use UTC-like second validation; all other scales use
/// continuous-second validation. Invalid calendar or second values return
/// [`BiasError::InvalidEpoch`].
pub fn civil_datetime_instant(
    epoch: crate::ppp_corrections::CivilDateTime,
    scale: TimeScale,
) -> Result<Instant, BiasError> {
    let second_policy = match scale {
        TimeScale::Utc | TimeScale::Glonasst => CivilSecondPolicy::UtcLike,
        _ => CivilSecondPolicy::Continuous,
    };
    validate::civil_datetime_with_second_policy(
        i64::from(epoch.year),
        i64::from(epoch.month),
        i64::from(epoch.day),
        i64::from(epoch.hour),
        i64::from(epoch.minute),
        epoch.second,
        second_policy,
    )
    .map_err(|_| BiasError::InvalidEpoch)?;
    let (jd_whole, fraction) = split_julian_date(
        epoch.year,
        i32::from(epoch.month),
        i32::from(epoch.day),
        i32::from(epoch.hour),
        i32::from(epoch.minute),
        epoch.second,
    );
    Ok(Instant::from_julian_date(
        scale,
        JulianDateSplit::new(jd_whole, fraction).map_err(|_| BiasError::InvalidEpoch)?,
    ))
}

fn parse_int<T>(token: Option<&str>, field: &'static str) -> Result<T, BiasError>
where
    T: FromStr,
{
    let token = token.ok_or(BiasError::InvalidInput {
        field,
        reason: "missing",
    })?;
    token.parse::<T>().map_err(|_| BiasError::InvalidInput {
        field,
        reason: "invalid integer",
    })
}

fn days_in_year(year: i32) -> i32 {
    if is_leap_year(year) {
        366
    } else {
        365
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn looks_like_satellite_token(token: &str) -> bool {
    let mut chars = token.chars();
    let Some(system) = chars.next() else {
        return false;
    };
    let Some(first_digit) = chars.next() else {
        return false;
    };
    GnssSystem::from_letter(system).is_some()
        && first_digit.is_ascii_digit()
        && chars.all(|c| c.is_ascii_digit())
}

fn rinex_frequency(sat: GnssSatelliteId, obs: &str, glonass_channel: Option<i8>) -> Option<f64> {
    frequencies::rinex_observation_frequency_hz(
        sat.system,
        obs,
        RINEX_VERSION_FOR_BIAS_CODES,
        glonass_channel,
    )
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::constants::{F_L1_HZ, F_L2_HZ};

    fn sat() -> GnssSatelliteId {
        GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap()
    }

    fn epoch(year: i32, doy: u16, sod: u32) -> Instant {
        bias_epoch_instant(BiasEpoch::new(year, doy, sod).unwrap(), TimeScale::Gpst).unwrap()
    }

    #[test]
    fn bias_epoch_normalizes_inclusive_end_of_day() {
        let end = BiasEpoch::new(2020, 366, 86_399)
            .unwrap()
            .normalize_end()
            .unwrap();
        assert_eq!(end, BiasEpoch::new(2021, 1, 0).unwrap());
    }

    #[test]
    fn dsb_path_resolution_uses_fewest_lexicographic_path() {
        let edge = |value: f64, record: usize| DsbEdge {
            value,
            record,
            affine: None,
            overridden: Vec::new(),
        };
        let mut graph = DsbGraph::new();
        let mut join = |from: &str, to: &str, value: f64, record: usize| {
            graph
                .entry(from.to_string())
                .or_default()
                .entry(to.to_string())
                .or_default()
                .push(edge(value, record));
            graph
                .entry(to.to_string())
                .or_default()
                .entry(from.to_string())
                .or_default()
                .push(edge(-value, record));
        };
        join("C1C", "C1W", 1.0, 0);
        join("C1C", "C1P", 2.0, 1);
        join("C1P", "C1W", 3.0, 2);
        assert_eq!(
            resolve_dsb_path(&graph, "C1C", "C1W", &|a: &DsbEdge, b: &DsbEdge| a.value
                == b.value),
            DsbPath::Resolved {
                value: 1.0,
                records: vec![0],
                overridden: vec![],
                stated: None,
            }
        );
    }

    #[test]
    fn stated_decimals_sum_exactly() {
        let parse = |text: &str| ExactDecimal::parse(text).unwrap();
        assert_eq!(
            parse("-.5E-9"),
            ExactDecimal {
                negative: true,
                mantissa: 5,
                exponent: -10,
            }
        );
        assert_eq!(parse("1.D3").mantissa, 1);
        assert_eq!(parse("1.D3").exponent, 3);
        assert_eq!(ExactDecimal::parse("."), None);
        assert_eq!(ExactDecimal::parse("1.2.3"), None);
        // 0.1 + 0.2 - 0.3 closes exactly; 0.1 + 0.2 - 0.30000000000000004
        // does not.
        let closes = |terms: &[&str]| {
            let terms: Vec<ExactDecimal> = terms.iter().map(|text| parse(text)).collect();
            ExactDecimal::sum_is_zero(&terms)
        };
        assert!(closes(&["0.1", "0.2", "-0.3"]));
        assert!(!closes(&["0.1", "0.2", "-0.30000000000000004"]));
        // Exponents far apart are aligned exactly.
        assert!(closes(&["1.E300", "1.E-300", "-1.E300", "-1.E-300"]));
        assert!(!closes(&["1.E300", "1.E-300", "-1.E300"]));
        // Signed zeros close.
        assert!(closes(&["-0.0", "0"]));
        // A zero term with an extreme exponent costs nothing.
        assert!(closes(&["0E-2147483648", "1.0", "-1.0"]));
        // Exponents at the i32 limits.
        assert!(closes(&["1.E2147483647", "-1.E2147483647"]));
        assert!(!closes(&["1.E2147483647", "1.E-2147483648"]));
        // Quadratically spaced exponents, cancelled in full and short of one.
        let spaced: Vec<String> = (0..60).map(|k| format!("1E{}", 46 * k * k)).collect();
        let negated: Vec<String> = spaced.iter().map(|term| format!("-{term}")).collect();
        let mut all: Vec<&str> = spaced.iter().map(String::as_str).collect();
        all.extend(negated.iter().map(String::as_str));
        assert!(closes(&all[..]));
        all.pop();
        assert!(!closes(&all[..]));
        // Stripping a trailing zero moves the exponent past i64::MAX without
        // saturating, so these differ by a factor of ten.
        assert!(!closes(&[
            "10E9223372036854775807",
            "-1E9223372036854775807"
        ]));
        assert!(closes(&[
            "10E9223372036854775807",
            "-100E9223372036854775806"
        ]));
        assert!(closes(&[
            "1.E2147483647",
            "1.E-2147483648",
            "-1.E2147483647",
            "-1.E-2147483648",
        ]));
    }

    #[test]
    fn a_set_built_in_code_compares_dsb_routes_within_the_rounding_bound() {
        let dsb = |obs1: &str, obs2: &str, value_ns: f64| BiasRecord {
            kind: BiasKind::Dsb,
            target: BiasTarget::Satellite(sat()),
            svn: None,
            obs1: obs1.to_string(),
            obs2: Some(obs2.to_string()),
            valid_from: Some(BiasEpoch::new(2020, 1, 0).unwrap()),
            valid_until: Some(BiasEpoch::new(2020, 2, 0).unwrap()),
            raw_epochs: (String::new(), String::new()),
            value: value_ns * NS_TO_S,
            sigma: None,
            slope: None,
            slope_sigma: None,
            family: BiasObservableFamily::Code,
            unit: BiasUnit::Nanoseconds,
            line: None,
        };
        let set_of = |records: Vec<BiasRecord>| {
            BiasSet::new(
                records,
                BiasMode::Relative,
                Some(TimeScale::Gpst),
                ClockReferenceObservables::default(),
                BiasSetHeader::default(),
                Diagnostics::new(),
            )
        };
        // No source text, so the routes' sums are compared within
        // (n + 2) * u * (sum |terms|): 0.1 + 0.2 and 0.15 + 0.15 differ by
        // one rounding and agree.
        let close = set_of(vec![
            dsb("C1C", "C1P", 0.1),
            dsb("C1P", "C1W", 0.2),
            dsb("C1C", "C1X", 0.15),
            dsb("C1X", "C1W", 0.15),
        ]);
        let t = epoch(2020, 1, 0);
        assert_eq!(
            close.code_dsb_seconds(sat(), "C1C", "C1W", t),
            BiasLookup::Available {
                value: 0.0 + 0.1 * NS_TO_S + 0.2 * NS_TO_S,
                records: vec![0, 1],
                overridden: vec![],
            }
        );
        let apart = set_of(vec![
            dsb("C1C", "C1P", 0.1),
            dsb("C1P", "C1W", 0.2),
            dsb("C1C", "C1X", 0.15),
            dsb("C1X", "C1W", 0.16),
        ]);
        assert_eq!(
            apart.code_dsb_seconds(sat(), "C1C", "C1W", t),
            BiasLookup::Ambiguous {
                records: vec![0, 1, 2, 3]
            }
        );
    }

    #[test]
    fn dcb_tgd_bridge_matches_pinned_operation_order() {
        let dcb_s = 4.2e-9_f64;
        let gamma = (F_L1_HZ / F_L2_HZ) * (F_L1_HZ / F_L2_HZ);
        let tgd_s = dcb_s / (1.0 - gamma);
        assert_eq!(tgd_s.to_bits(), 0xbe3be217807ad49e);
    }

    #[test]
    fn ionosphere_free_bias_model_matches_closed_form_bits() {
        let (alpha, beta) = ionosphere_free_coefficients(F_L1_HZ, F_L2_HZ).unwrap();
        let used = alpha * 1.25e-9 + beta * -0.5e-9;
        let reference = alpha * 0.25e-9 + beta * -0.5e-9;
        let model = (used - reference) * C_M_S;
        assert_eq!(model.to_bits(), 0x3fe86c0d69376a57);
    }

    #[test]
    fn code_query_ignores_phase_record() {
        let record = BiasRecord {
            kind: BiasKind::Osb,
            target: BiasTarget::Satellite(sat()),
            svn: None,
            obs1: "L1C".to_string(),
            obs2: None,
            valid_from: Some(BiasEpoch::new(2020, 1, 0).unwrap()),
            valid_until: Some(BiasEpoch::new(2020, 2, 0).unwrap()),
            raw_epochs: ("2020:001:00000".to_string(), "2020:002:00000".to_string()),
            value: -0.25,
            sigma: None,
            slope: None,
            slope_sigma: None,
            family: BiasObservableFamily::Phase,
            unit: BiasUnit::Cycles,
            line: None,
        };
        let set = BiasSet::new(
            vec![record],
            BiasMode::Absolute,
            Some(TimeScale::Gpst),
            ClockReferenceObservables::default(),
            BiasSetHeader::default(),
            Diagnostics::new(),
        );
        assert_eq!(
            set.code_osb_seconds(sat(), "L1C", epoch(2020, 1, 0)),
            BiasLookup::Absent
        );
        // A phase bias stated in cycles is returned without a carrier
        // frequency.
        assert_eq!(
            set.phase_osb_cycles(sat(), "L1C", epoch(2020, 1, 0), None),
            BiasLookup::Available {
                value: -0.25,
                records: vec![0],
                overridden: vec![],
            }
        );
    }

    #[test]
    fn dcb_parse_preserves_wide_station_identifiers() {
        let options = CodeDcbOptions {
            pair: ("P1".to_string(), "C1".to_string()),
            year: 2026,
            month: 1,
            time_scale: TimeScale::Gpst,
            receiver_system: None,
        };
        let line = "G     AB-1 12345M001        1.234       0.050";
        let row = parse_dcb_row(line.as_bytes(), &options).unwrap().unwrap();
        match row.target {
            BiasTarget::Receiver { system, station } => {
                assert_eq!(system, GnssSystem::Gps);
                assert_eq!(station, "AB-1 12345M001");
            }
            _ => panic!("expected Receiver target"),
        }
    }

    #[test]
    fn write_code_dcb_preserves_absent_and_zero_sigma() {
        let options = CodeDcbOptions {
            pair: ("P1".to_string(), "C1".to_string()),
            year: 2026,
            month: 1,
            time_scale: TimeScale::Gpst,
            receiver_system: None,
        };
        let (start, end) = dcb_month_interval(2026, 1).unwrap();
        let record_no_sigma = BiasRecord {
            kind: BiasKind::Dsb,
            target: BiasTarget::Satellite(sat()),
            svn: None,
            obs1: "C1W".to_string(),
            obs2: Some("C1C".to_string()),
            valid_from: Some(start),
            valid_until: end,
            raw_epochs: (String::new(), String::new()),
            value: 1.234e-9,
            sigma: None,
            slope: None,
            slope_sigma: None,
            family: BiasObservableFamily::Code,
            unit: BiasUnit::Nanoseconds,
            line: None,
        };
        let header = BiasSetHeader {
            dcb_meta: Some(options.clone()),
            ..Default::default()
        };
        let set_no_sigma = BiasSet::new(
            vec![record_no_sigma.clone()],
            BiasMode::Relative,
            Some(TimeScale::Gpst),
            ClockReferenceObservables::default(),
            header.clone(),
            Diagnostics::new(),
        );
        let text_no_sigma = write_code_dcb(&set_no_sigma).unwrap();
        assert!(
            text_no_sigma
                .lines()
                .any(|l| l.contains("G01") && l.ends_with("    1.234")),
            "expected line without RMS column when sigma is None: {text_no_sigma}"
        );
        assert!(!text_no_sigma.contains("0.000"));

        let record_zero_sigma = BiasRecord {
            sigma: Some(0.0),
            ..record_no_sigma
        };
        let set_zero_sigma = BiasSet::new(
            vec![record_zero_sigma],
            BiasMode::Relative,
            Some(TimeScale::Gpst),
            ClockReferenceObservables::default(),
            header,
            Diagnostics::new(),
        );
        let text_zero_sigma = write_code_dcb(&set_zero_sigma).unwrap();
        assert!(
            text_zero_sigma
                .lines()
                .any(|l| l.contains("G01") && l.ends_with("    0.000")),
            "expected line with 0.000 RMS when sigma is Some(0.0): {text_zero_sigma}"
        );
    }

    #[test]
    fn bias_sinex_solution_line_rejects_corrupted_optional_floats() {
        let line_sigma = format!(
            " {:<4} {:<4} {:<3} {:<9} {:<4} {:<4} {:<14} {:<14} {:<4} {:>21.12E} {:>11}",
            "OSB",
            "",
            "G01",
            "",
            "C1C",
            "",
            "2020:001:00000",
            "2020:002:00000",
            "ns",
            0.1,
            "BAD_SIGMA"
        );
        let err = parse_solution_line(line_sigma.as_bytes()).unwrap_err();
        assert!(
            matches!(
                err,
                SkipReason::MalformedField(FieldError::FloatParse {
                    field: "bias sigma",
                    ..
                })
            ),
            "expected MalformedField on corrupt sigma, got {err:?}"
        );

        let line_slope = format!(
            " {:<4} {:<4} {:<3} {:<9} {:<4} {:<4} {:<14} {:<14} {:<4} {:>21.12E} {:>11.5E} {:>21}",
            "OSB",
            "",
            "G01",
            "",
            "C1C",
            "",
            "2020:001:00000",
            "2020:002:00000",
            "ns",
            0.1,
            0.01,
            "BAD_SLOPE"
        );
        let err_slope = parse_solution_line(line_slope.as_bytes()).unwrap_err();
        assert!(
            matches!(
                err_slope,
                SkipReason::MalformedField(FieldError::FloatParse {
                    field: "bias slope",
                    ..
                })
            ),
            "expected MalformedField on corrupt slope, got {err_slope:?}"
        );
    }

    #[test]
    fn bias_sinex_description_rejects_unrecognized_bias_mode() {
        let rows = vec![parse_info_row(b" BIAS_MODE INVALID_MODE", 10)];
        let mut diagnostics = Diagnostics::new();
        let mut notices = Vec::new();
        let mut departures = Vec::new();
        let view = derive_description(&rows, &mut diagnostics, &mut notices, &mut departures);
        assert_eq!(view.mode, BiasMode::Unspecified);
        assert_eq!(diagnostics.skips.len(), 1);
        assert!(
            matches!(
                diagnostics.skips[0].reason,
                SkipReason::UnsupportedRecordType("BIAS_MODE")
            ),
            "expected UnsupportedRecordType(\"BIAS_MODE\"), got {:?}",
            diagnostics.skips[0].reason
        );
        assert!(departures.contains(&BiasDeparture::UnsupportedBiasMode {
            line: 10,
            label: "INVALID_MODE".to_string(),
        }));
    }

    #[test]
    fn code_dcb_parse_write_parse_roundtrip_and_canonical_lookup() {
        let dcb_text = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
G01                           0.626       0.000
G02                          -2.069
G     ABMF 97103M001         -1.365       0.050
G     ALGO 40104M002          1.234       0.000
G     VALUE00USA             -0.500
";
        let parsed = BiasSet::parse_code_dcb(dcb_text.as_bytes(), None).unwrap();
        let set = parsed.value;
        assert_eq!(set.skipped_records(), 0);
        assert_eq!(set.records().len(), 5);

        let rec_abmf = set
            .records()
            .iter()
            .find(|r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "ABMF 97103M001"))
            .unwrap();
        match &rec_abmf.target {
            BiasTarget::Receiver { system, station } => {
                assert_eq!(*system, GnssSystem::Gps);
                assert_eq!(station, "ABMF 97103M001");
            }
            _ => panic!("expected Receiver target"),
        }
        assert_eq!(rec_abmf.value.to_bits(), (-1.365 * NS_TO_S).to_bits());
        assert_eq!(
            rec_abmf.sigma.unwrap().to_bits(),
            (0.050 * NS_TO_S).to_bits()
        );

        let rec_algo = set
            .records()
            .iter()
            .find(|r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "ALGO 40104M002"))
            .unwrap();
        assert_eq!(rec_algo.value.to_bits(), (1.234 * NS_TO_S).to_bits());
        assert_eq!(rec_algo.sigma, Some(0.0));

        let rec_value = set
            .records()
            .iter()
            .find(|r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "VALUE00USA"))
            .unwrap();
        assert_eq!(rec_value.value.to_bits(), (-0.500 * NS_TO_S).to_bits());
        assert_eq!(rec_value.sigma, None);

        let t0 = epoch(2026, 153, 0);
        assert_eq!(
            set.receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-1.365 * NS_TO_S).to_bits()
        );
        assert_eq!(
            set.receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF 97103M001", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-1.365 * NS_TO_S).to_bits()
        );

        let sat_g01 = sat();
        assert_eq!(
            set.code_dsb_seconds(sat_g01, "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (0.626 * NS_TO_S).to_bits()
        );

        let written = write_code_dcb(&set).unwrap();
        assert!(written
            .lines()
            .any(|l| l.starts_with("G     ABMF 97103M001") && l.ends_with("    0.050")));
        assert!(written
            .lines()
            .any(|l| l.starts_with("G     ALGO 40104M002") && l.ends_with("    0.000")));
        assert!(written
            .lines()
            .any(|l| l.starts_with("G     VALUE00USA") && l.ends_with("   -0.500")));
        assert!(written
            .lines()
            .any(|l| l.starts_with("G01") && l.ends_with("    0.000")));
        assert!(written
            .lines()
            .any(|l| l.starts_with("G02") && l.ends_with("   -2.069")));

        let reparsed = BiasSet::parse_code_dcb(written.as_bytes(), None)
            .unwrap()
            .value;
        assert_eq!(reparsed.skipped_records(), 0);
        assert_eq!(reparsed.records().len(), 5);

        let rep_abmf = reparsed
            .records()
            .iter()
            .find(|r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "ABMF 97103M001"))
            .unwrap();
        assert_eq!(rep_abmf.value.to_bits(), rec_abmf.value.to_bits());
        assert_eq!(
            rep_abmf.sigma.unwrap().to_bits(),
            rec_abmf.sigma.unwrap().to_bits()
        );
        assert_eq!(
            reparsed
                .receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-1.365 * NS_TO_S).to_bits()
        );
        assert_eq!(
            reparsed
                .receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF 97103M001", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-1.365 * NS_TO_S).to_bits()
        );

        let rep_algo = reparsed
            .records()
            .iter()
            .find(|r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "ALGO 40104M002"))
            .unwrap();
        assert_eq!(rep_algo.value.to_bits(), rec_algo.value.to_bits());
        assert_eq!(rep_algo.sigma, Some(0.0));

        let rep_value = reparsed
            .records()
            .iter()
            .find(|r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "VALUE00USA"))
            .unwrap();
        assert_eq!(rep_value.value.to_bits(), rec_value.value.to_bits());
        assert_eq!(rep_value.sigma, None);

        let rep_g01 = reparsed
            .records()
            .iter()
            .find(|r| r.target == BiasTarget::Satellite(sat_g01))
            .unwrap();
        assert_eq!(rep_g01.value.to_bits(), (0.626 * NS_TO_S).to_bits());
        assert_eq!(rep_g01.sigma, Some(0.0));

        let sat_g02 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let rep_g02 = reparsed
            .records()
            .iter()
            .find(|r| r.target == BiasTarget::Satellite(sat_g02))
            .unwrap();
        assert_eq!(rep_g02.value.to_bits(), (-2.069 * NS_TO_S).to_bits());
        assert_eq!(rep_g02.sigma, None);
    }

    #[test]
    fn code_dcb_parse_reports_diagnostics_for_malformed_sigma_and_missing_value() {
        let text_bad_sigma = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
G     ABMF 97103M001         -1.360       BAD_SIGMA
";
        let parsed = BiasSet::parse_code_dcb(text_bad_sigma.as_bytes(), None).unwrap();
        assert_eq!(parsed.value.records().len(), 0);
        assert_eq!(parsed.value.skipped_records(), 1);
        assert!(matches!(
            parsed.value.diagnostics().skips[0].reason,
            SkipReason::MalformedField(FieldError::FloatParse {
                field: "dcb sigma",
                ..
            })
        ));

        let text_missing_val_rec = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
G     ABMF 97103M001
";
        let parsed = BiasSet::parse_code_dcb(text_missing_val_rec.as_bytes(), None).unwrap();
        assert_eq!(parsed.value.records().len(), 0);
        assert_eq!(parsed.value.skipped_records(), 1);
        assert!(matches!(
            parsed.value.diagnostics().skips[0].reason,
            SkipReason::MalformedField(FieldError::Missing { field: "dcb value" })
        ));

        let text_missing_val_sat = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
G01
";
        let parsed = BiasSet::parse_code_dcb(text_missing_val_sat.as_bytes(), None).unwrap();
        assert_eq!(parsed.value.records().len(), 0);
        assert_eq!(parsed.value.skipped_records(), 1);
        assert!(matches!(
            parsed.value.diagnostics().skips[0].reason,
            SkipReason::MalformedField(FieldError::Missing { field: "dcb value" })
        ));

        let text_missing_val_sysless = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
      ABMF 97103M001
";
        let opts = CodeDcbOptions::new(
            ("P1".to_string(), "C1".to_string()),
            2026,
            6,
            TimeScale::Gpst,
        )
        .with_receiver_system(GnssSystem::Gps);
        let parsed =
            BiasSet::parse_code_dcb(text_missing_val_sysless.as_bytes(), Some(opts.clone()))
                .unwrap();
        assert_eq!(parsed.value.records().len(), 0);
        assert_eq!(parsed.value.skipped_records(), 1);
        assert!(matches!(
            parsed.value.diagnostics().skips[0].reason,
            SkipReason::MalformedField(FieldError::Missing { field: "dcb value" })
        ));

        let text_malformed_val_sysless = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
      ABMF 97103M001         NOT_A_FLOAT
";
        let parsed =
            BiasSet::parse_code_dcb(text_malformed_val_sysless.as_bytes(), Some(opts.clone()))
                .unwrap();
        assert_eq!(parsed.value.records().len(), 0);
        assert_eq!(parsed.value.skipped_records(), 1);
        assert!(matches!(
            parsed.value.diagnostics().skips[0].reason,
            SkipReason::MalformedField(FieldError::FloatParse {
                field: "dcb value",
                ..
            })
        ));

        let text_headers_only = "\
# DCB P1-C1 2026-06 G
CODE'S MONTHLY GNSS P1-C1 DCB SOLUTION, YEAR 2026, MONTH 06      01-JUL-26 08:42
--------------------------------------------------------------------------------
DIFFERENTIAL (P1-C1) CODE BIASES FOR SATELLITES AND RECEIVERS:
 PRN / STATION NAME        VALUE (NS)  RMS (NS)
***   ****************    *****.***   *****.***
# Comment line
";
        let parsed = BiasSet::parse_code_dcb(text_headers_only.as_bytes(), None).unwrap();
        assert_eq!(parsed.value.records().len(), 0);
        assert_eq!(parsed.value.skipped_records(), 0);

        let parsed_opts =
            BiasSet::parse_code_dcb(text_headers_only.as_bytes(), Some(opts.clone())).unwrap();
        assert_eq!(parsed_opts.value.records().len(), 0);
        assert_eq!(parsed_opts.value.skipped_records(), 0);

        let text_arbitrary_prose = "\
# DCB P1-C1 2026-06 G
--------------------------------------------------------------------------------
This is an arbitrary prose line that must not be declared a receiver record.
Another prose paragraph explaining operational caveats and references.
";
        let parsed =
            BiasSet::parse_code_dcb(text_arbitrary_prose.as_bytes(), Some(opts.clone())).unwrap();
        assert_eq!(parsed.value.records().len(), 0);
        assert_eq!(parsed.value.skipped_records(), 0);

        let text_malformed_val_sat = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
G01                           VALUE       0.000
";
        let parsed = BiasSet::parse_code_dcb(text_malformed_val_sat.as_bytes(), None).unwrap();
        assert_eq!(parsed.value.records().len(), 0);
        assert_eq!(parsed.value.skipped_records(), 1);
        assert!(matches!(
            parsed.value.diagnostics().skips[0].reason,
            SkipReason::MalformedField(FieldError::FloatParse {
                field: "dcb value",
                ref value,
            }) if value == "VALUE"
        ));

        let text_malformed_val_rec_explicit = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
G     ABMF 97103M001          VALUE       0.050
";
        let parsed =
            BiasSet::parse_code_dcb(text_malformed_val_rec_explicit.as_bytes(), None).unwrap();
        assert_eq!(parsed.value.records().len(), 0);
        assert_eq!(parsed.value.skipped_records(), 1);
        assert!(matches!(
            parsed.value.diagnostics().skips[0].reason,
            SkipReason::MalformedField(FieldError::FloatParse {
                field: "dcb value",
                ref value,
            }) if value == "VALUE"
        ));

        let text_malformed_val_rec_sysless = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
      ABMF 97103M001          VALUE       0.050
";
        let parsed = BiasSet::parse_code_dcb(
            text_malformed_val_rec_sysless.as_bytes(),
            Some(opts.clone()),
        )
        .unwrap();
        assert_eq!(parsed.value.records().len(), 0);
        assert_eq!(parsed.value.skipped_records(), 1);
        assert!(matches!(
            parsed.value.diagnostics().skips[0].reason,
            SkipReason::MalformedField(FieldError::FloatParse {
                field: "dcb value",
                ref value,
            }) if value == "VALUE"
        ));

        let text_value_station = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
G     VALUE00USA             -1.365       0.050
      VALUE00USA              0.500       0.010
G     PRN00USA               -1.365       0.050
      PRN00USA                0.500       0.010
G     DIFF00USA              -1.365       0.050
      DIFF00USA               0.500       0.010
G     CODES00USA             -1.365       0.050
      CODES00USA              0.500       0.010
";
        let parsed = BiasSet::parse_code_dcb(text_value_station.as_bytes(), Some(opts)).unwrap();
        assert_eq!(parsed.value.skipped_records(), 0);
        assert_eq!(parsed.value.records().len(), 8);
        assert_eq!(
            parsed.value.records()[0].target,
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "VALUE00USA".to_string(),
            }
        );
        assert_eq!(
            parsed.value.records()[1].target,
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "VALUE00USA".to_string(),
            }
        );
        assert_eq!(
            parsed.value.records()[2].target,
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "PRN00USA".to_string(),
            }
        );
        assert_eq!(
            parsed.value.records()[3].target,
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "PRN00USA".to_string(),
            }
        );
        assert_eq!(
            parsed.value.records()[4].target,
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "DIFF00USA".to_string(),
            }
        );
        assert_eq!(
            parsed.value.records()[5].target,
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "DIFF00USA".to_string(),
            }
        );
        assert_eq!(
            parsed.value.records()[6].target,
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "CODES00USA".to_string(),
            }
        );
        assert_eq!(
            parsed.value.records()[7].target,
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "CODES00USA".to_string(),
            }
        );
    }

    #[test]
    fn code_dcb_parse_write_parse_systemless_stations_regression() {
        let dcb_text = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
      abmf                   -1.365       0.050
      ab-1                    2.500       0.010
      st_01                  -0.750       0.020
      in sp 01                1.000       0.000
";
        let opts = CodeDcbOptions::new(
            ("P1".to_string(), "C1".to_string()),
            2026,
            6,
            TimeScale::Gpst,
        )
        .with_receiver_system(GnssSystem::Gps);

        let parsed = BiasSet::parse_code_dcb(dcb_text.as_bytes(), Some(opts.clone())).unwrap();
        let set = parsed.value;
        assert_eq!(set.skipped_records(), 0);
        assert_eq!(set.records().len(), 4);

        let rec_abmf = set
            .records()
            .iter()
            .find(
                |r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "abmf"),
            )
            .expect("find abmf");
        assert_eq!(rec_abmf.value.to_bits(), (-1.365 * NS_TO_S).to_bits());
        assert_eq!(
            rec_abmf.sigma.unwrap().to_bits(),
            (0.050 * NS_TO_S).to_bits()
        );

        let rec_hyphen = set
            .records()
            .iter()
            .find(
                |r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "ab-1"),
            )
            .expect("find ab-1");
        assert_eq!(rec_hyphen.value.to_bits(), (2.500 * NS_TO_S).to_bits());
        assert_eq!(
            rec_hyphen.sigma.unwrap().to_bits(),
            (0.010 * NS_TO_S).to_bits()
        );

        let rec_under = set
            .records()
            .iter()
            .find(
                |r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "st_01"),
            )
            .expect("find st_01");
        assert_eq!(rec_under.value.to_bits(), (-0.750 * NS_TO_S).to_bits());
        assert_eq!(
            rec_under.sigma.unwrap().to_bits(),
            (0.020 * NS_TO_S).to_bits()
        );

        let rec_space = set
            .records()
            .iter()
            .find(|r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "in sp 01"))
            .expect("find in sp 01");
        assert_eq!(rec_space.value.to_bits(), (1.000 * NS_TO_S).to_bits());
        assert_eq!(
            rec_space.sigma.unwrap().to_bits(),
            (0.000 * NS_TO_S).to_bits()
        );

        let t0 = epoch(2026, 153, 0);
        // Canonical lookup succeeds with exact case and uppercase
        assert_eq!(
            set.receiver_code_dsb_seconds(GnssSystem::Gps, "abmf", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-1.365 * NS_TO_S).to_bits()
        );
        assert_eq!(
            set.receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-1.365 * NS_TO_S).to_bits()
        );
        assert_eq!(
            set.receiver_code_dsb_seconds(GnssSystem::Gps, "ab-1", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (2.500 * NS_TO_S).to_bits()
        );
        assert_eq!(
            set.receiver_code_dsb_seconds(GnssSystem::Gps, "AB-1", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (2.500 * NS_TO_S).to_bits()
        );
        assert_eq!(
            set.receiver_code_dsb_seconds(GnssSystem::Gps, "st_01", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-0.750 * NS_TO_S).to_bits()
        );
        assert_eq!(
            set.receiver_code_dsb_seconds(GnssSystem::Gps, "ST_01", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-0.750 * NS_TO_S).to_bits()
        );
        assert_eq!(
            set.receiver_code_dsb_seconds(GnssSystem::Gps, "in sp 01", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (1.000 * NS_TO_S).to_bits()
        );
        assert_eq!(
            set.receiver_code_dsb_seconds(GnssSystem::Gps, "IN SP 01", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (1.000 * NS_TO_S).to_bits()
        );

        // A set read from DCB is written back as its source lines, so the
        // system-less rows read back with the same options.
        let written = write_code_dcb(&set).unwrap();
        assert_eq!(written, dcb_text);
        let reparsed = BiasSet::parse_code_dcb(written.as_bytes(), Some(opts))
            .unwrap()
            .value;
        assert_eq!(reparsed.skipped_records(), 0);
        assert_eq!(reparsed.records().len(), 4);

        // Full field text survives
        let rep_abmf = reparsed
            .records()
            .iter()
            .find(
                |r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "abmf"),
            )
            .expect("find abmf in reparsed");
        assert_eq!(rep_abmf.value.to_bits(), rec_abmf.value.to_bits());
        assert_eq!(
            rep_abmf.sigma.unwrap().to_bits(),
            rec_abmf.sigma.unwrap().to_bits()
        );

        let rep_hyphen = reparsed
            .records()
            .iter()
            .find(
                |r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "ab-1"),
            )
            .expect("find ab-1 in reparsed");
        assert_eq!(rep_hyphen.value.to_bits(), rec_hyphen.value.to_bits());
        assert_eq!(
            rep_hyphen.sigma.unwrap().to_bits(),
            rec_hyphen.sigma.unwrap().to_bits()
        );

        let rep_under = reparsed
            .records()
            .iter()
            .find(
                |r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "st_01"),
            )
            .expect("find st_01 in reparsed");
        assert_eq!(rep_under.value.to_bits(), rec_under.value.to_bits());
        assert_eq!(
            rep_under.sigma.unwrap().to_bits(),
            rec_under.sigma.unwrap().to_bits()
        );

        let rep_space = reparsed
            .records()
            .iter()
            .find(|r| matches!(&r.target, BiasTarget::Receiver { station, .. } if station == "in sp 01"))
            .expect("find in sp 01 in reparsed");
        assert_eq!(rep_space.value.to_bits(), rec_space.value.to_bits());
        assert_eq!(
            rep_space.sigma.unwrap().to_bits(),
            rec_space.sigma.unwrap().to_bits()
        );

        // Canonical lookup survives on reparsed set
        assert_eq!(
            reparsed
                .receiver_code_dsb_seconds(GnssSystem::Gps, "abmf", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-1.365 * NS_TO_S).to_bits()
        );
        assert_eq!(
            reparsed
                .receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-1.365 * NS_TO_S).to_bits()
        );
        assert_eq!(
            reparsed
                .receiver_code_dsb_seconds(GnssSystem::Gps, "ab-1", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (2.500 * NS_TO_S).to_bits()
        );
        assert_eq!(
            reparsed
                .receiver_code_dsb_seconds(GnssSystem::Gps, "AB-1", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (2.500 * NS_TO_S).to_bits()
        );
        assert_eq!(
            reparsed
                .receiver_code_dsb_seconds(GnssSystem::Gps, "st_01", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-0.750 * NS_TO_S).to_bits()
        );
        assert_eq!(
            reparsed
                .receiver_code_dsb_seconds(GnssSystem::Gps, "ST_01", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-0.750 * NS_TO_S).to_bits()
        );
        assert_eq!(
            reparsed
                .receiver_code_dsb_seconds(GnssSystem::Gps, "in sp 01", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (1.000 * NS_TO_S).to_bits()
        );
        assert_eq!(
            reparsed
                .receiver_code_dsb_seconds(GnssSystem::Gps, "IN SP 01", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (1.000 * NS_TO_S).to_bits()
        );
    }

    #[test]
    fn code_dcb_unusual_station_names_and_unpadded_regression() {
        let mut opts = CodeDcbOptions::new(
            ("P1".to_string(), "C1".to_string()),
            2026,
            6,
            TimeScale::Gpst,
        );
        opts.receiver_system = Some(GnssSystem::Gps);
        let t0 = epoch(2026, 153, 0);

        let cases = [
            (
                "explicit STATION NAME RMS",
                format!(
                    "G     {:<16}    {:9.3}   {:9.3}",
                    "STATION NAME RMS", -1.365, 0.050
                ),
                "STATION NAME RMS",
            ),
            (
                "implicit STATION NAME RMS",
                "      STATION NAME RMS        -1.365       0.050".to_string(),
                "STATION NAME RMS",
            ),
            (
                "implicit CODE'S SOLUTION",
                "      CODE'S SOLUTION         -1.365       0.050".to_string(),
                "CODE'S SOLUTION",
            ),
            (
                "unpadded long station ABMF97103M001",
                "ABMF97103M001                 -1.365       0.050".to_string(),
                "ABMF97103M001",
            ),
            (
                "explicit -abmf",
                "G     -abmf                   -1.365       0.050".to_string(),
                "-abmf",
            ),
            (
                "implicit -abmf",
                "      -abmf                   -1.365       0.050".to_string(),
                "-abmf",
            ),
            (
                "unpadded -abmf",
                "-abmf                         -1.365       0.050".to_string(),
                "-abmf",
            ),
            (
                "explicit #abmf",
                "G     #abmf                   -1.365       0.050".to_string(),
                "#abmf",
            ),
            (
                "implicit #abmf",
                "      #abmf                   -1.365       0.050".to_string(),
                "#abmf",
            ),
            (
                "unpadded #abmf",
                "#abmf                         -1.365       0.050".to_string(),
                "#abmf",
            ),
        ];

        for (name, line, expected_station) in &cases {
            let text = format!(
                "# DCB P1-C1 2026-06 G\n\
                 PRN / STATION NAME        VALUE (ns)  RMS (ns)\n\
                ***   ****************    *****.***   *****.***\n\
                # Comment line\n\
                --------------------------------------------------------------------------------\n\
                {line}\n"
            );
            let parsed = BiasSet::parse_code_dcb(text.as_bytes(), Some(opts.clone()))
                .unwrap_or_else(|e| panic!("{name}: parse error: {e:?}"));
            let set = parsed.value;
            assert_eq!(set.skipped_records(), 0, "{name}: expected 0 skips");
            assert_eq!(set.records().len(), 1, "{name}: expected 1 record");

            let rec = &set.records()[0];
            match &rec.target {
                BiasTarget::Receiver { system, station } => {
                    assert_eq!(*system, GnssSystem::Gps, "{name}: expected GPS system");
                    assert_eq!(station, expected_station, "{name}: station mismatch");
                }
                other => panic!("{name}: expected Receiver target, got {other:?}"),
            }
            assert_eq!(
                rec.value.to_bits(),
                (-1.365 * NS_TO_S).to_bits(),
                "{name}: value mismatch"
            );
            assert_eq!(
                rec.sigma.unwrap().to_bits(),
                (0.050 * NS_TO_S).to_bits(),
                "{name}: sigma mismatch"
            );

            let written =
                write_code_dcb(&set).unwrap_or_else(|e| panic!("{name}: write error: {e:?}"));
            let reparsed = BiasSet::parse_code_dcb(written.as_bytes(), Some(opts.clone()))
                .unwrap_or_else(|e| panic!("{name}: reparse error: {e:?}"))
                .value;
            assert_eq!(
                reparsed.skipped_records(),
                0,
                "{name}: expected 0 skips on reparse"
            );
            assert_eq!(
                reparsed.records().len(),
                1,
                "{name}: expected 1 record on reparse"
            );

            let rep_rec = &reparsed.records()[0];
            assert_eq!(
                rep_rec.target, rec.target,
                "{name}: target preserved on reparse"
            );
            assert_eq!(
                rep_rec.value.to_bits(),
                rec.value.to_bits(),
                "{name}: value preserved"
            );
            assert_eq!(
                rep_rec.sigma.unwrap().to_bits(),
                rec.sigma.unwrap().to_bits(),
                "{name}: sigma preserved"
            );
        }

        // Concrete long station ABMF97103M001: verify full station and canonical ABMF lookup survive parse/write/parse
        let unpadded_text =
            "# DCB P1-C1 2026-06 G\nABMF97103M001                 -1.365       0.050\n";
        let set_abmf = BiasSet::parse_code_dcb(unpadded_text.as_bytes(), Some(opts.clone()))
            .unwrap()
            .value;
        assert_eq!(set_abmf.skipped_records(), 0);
        assert_eq!(set_abmf.records().len(), 1);
        assert_eq!(
            set_abmf.records()[0].target,
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF97103M001".to_string(),
            }
        );
        assert_eq!(
            set_abmf
                .receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF97103M001", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-1.365 * NS_TO_S).to_bits()
        );
        assert_eq!(
            set_abmf
                .receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-1.365 * NS_TO_S).to_bits()
        );

        let written_abmf = write_code_dcb(&set_abmf).unwrap();
        assert_eq!(written_abmf, unpadded_text);
        let reparsed_abmf = BiasSet::parse_code_dcb(written_abmf.as_bytes(), Some(opts.clone()))
            .unwrap()
            .value;
        assert_eq!(reparsed_abmf.skipped_records(), 0);
        assert_eq!(reparsed_abmf.records().len(), 1);
        assert_eq!(
            reparsed_abmf.records()[0].target,
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF97103M001".to_string(),
            }
        );
        assert_eq!(
            reparsed_abmf
                .receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF97103M001", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-1.365 * NS_TO_S).to_bits()
        );
        assert_eq!(
            reparsed_abmf
                .receiver_code_dsb_seconds(GnssSystem::Gps, "ABMF", "C1W", "C1C", t0)
                .value()
                .unwrap()
                .to_bits(),
            (-1.365 * NS_TO_S).to_bits()
        );

        // Data candidates with malformed VALUE still emit typed diagnostics
        let malformed_cases = [
            "G     STATION NAME RMS        VALUE       0.050",
            "      STATION NAME RMS        VALUE       0.050",
            "      CODE'S SOLUTION         VALUE       0.050",
            "      -abmf                   VALUE       0.050",
            "      #abmf                   VALUE       0.050",
        ];
        for line in malformed_cases {
            let text = format!("# DCB P1-C1 2026-06 G\n{line}\n");
            let parsed = BiasSet::parse_code_dcb(text.as_bytes(), Some(opts.clone())).unwrap();
            assert_eq!(parsed.value.records().len(), 0);
            assert_eq!(parsed.value.skipped_records(), 1);
            assert!(matches!(
                parsed.value.diagnostics().skips[0].reason,
                SkipReason::MalformedField(FieldError::FloatParse {
                    field: "dcb value",
                    ref value,
                }) if value == "VALUE"
            ));
        }
    }

    #[test]
    fn code_dcb_negative_zero_sigma_bit_preserving_roundtrip() {
        let dcb_text = "\
# DCB P1-C1 2026-06 G
 PRN / STATION NAME        VALUE (ns)  RMS (ns)
***   ****************    *****.***   *****.***
G01                           0.626      -0.000
G     ABMF 97103M001         -1.365      -0.000
";
        let parsed = BiasSet::parse_code_dcb(dcb_text.as_bytes(), None).unwrap();
        let set = parsed.value;
        assert_eq!(set.skipped_records(), 0);
        assert_eq!(set.records().len(), 2);

        let rec_g01 = &set.records()[0];
        assert_eq!(rec_g01.value.to_bits(), (0.626 * NS_TO_S).to_bits());
        assert_eq!(rec_g01.sigma.unwrap().to_bits(), (-0.0_f64).to_bits());

        let rec_abmf = &set.records()[1];
        assert_eq!(rec_abmf.value.to_bits(), (-1.365 * NS_TO_S).to_bits());
        assert_eq!(rec_abmf.sigma.unwrap().to_bits(), (-0.0_f64).to_bits());

        let written = write_code_dcb(&set).unwrap();
        assert!(written
            .lines()
            .any(|l| l.starts_with("G01") && l.ends_with("   -0.000")));
        assert!(written
            .lines()
            .any(|l| l.starts_with("G     ABMF 97103M001") && l.ends_with("   -0.000")));

        let reparsed = BiasSet::parse_code_dcb(written.as_bytes(), None)
            .unwrap()
            .value;
        assert_eq!(reparsed.skipped_records(), 0);
        assert_eq!(reparsed.records().len(), 2);

        let rep_g01 = &reparsed.records()[0];
        assert_eq!(rep_g01.value.to_bits(), rec_g01.value.to_bits());
        assert_eq!(rep_g01.sigma.unwrap().to_bits(), (-0.0_f64).to_bits());

        let rep_abmf = &reparsed.records()[1];
        assert_eq!(rep_abmf.value.to_bits(), rec_abmf.value.to_bits());
        assert_eq!(rep_abmf.sigma.unwrap().to_bits(), (-0.0_f64).to_bits());
    }

    #[test]
    fn code_dcb_writer_refusals_for_unrepresentable_values() {
        let make_set = |target: BiasTarget, value: f64, sigma: Option<f64>| -> BiasSet {
            let options = CodeDcbOptions::new(
                ("P1".to_string(), "C1".to_string()),
                2026,
                6,
                TimeScale::Gpst,
            );
            let (start, end) = dcb_month_interval(2026, 6).unwrap();
            let record = BiasRecord {
                kind: BiasKind::Dsb,
                target,
                svn: None,
                obs1: "C1W".to_string(),
                obs2: Some("C1C".to_string()),
                valid_from: Some(start),
                valid_until: end,
                raw_epochs: (String::new(), String::new()),
                value,
                sigma,
                slope: None,
                slope_sigma: None,
                family: BiasObservableFamily::Code,
                unit: BiasUnit::Nanoseconds,
                line: None,
            };
            let header = BiasSetHeader {
                dcb_meta: Some(options),
                ..Default::default()
            };
            BiasSet::new(
                vec![record],
                BiasMode::Relative,
                Some(TimeScale::Gpst),
                ClockReferenceObservables::default(),
                header,
                Diagnostics::new(),
            )
        };

        // Station line injection
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF\nINJECT".to_string(),
            },
            1.0e-9,
            None,
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "station",
                reason: "line injection",
            }
        );

        // Station control character
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF\t001".to_string(),
            },
            1.0e-9,
            None,
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "station",
                reason: "contains control character",
            }
        );

        // Station exceeds maximum width (> 16 chars)
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF 97103M001 EXTRA".to_string(),
            },
            1.0e-9,
            None,
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "station",
                reason: "cannot fit format",
            }
        );

        // Station empty
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: String::new(),
            },
            1.0e-9,
            None,
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "station",
                reason: "missing",
            }
        );

        // Station leading whitespace
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: " ABMF 97103M001".to_string(),
            },
            1.0e-9,
            None,
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "station",
                reason: "shifts fixed columns",
            }
        );

        // Value excessive precision
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF 97103M001".to_string(),
            },
            -1.3654e-9,
            None,
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "value",
                reason: "excessive precision",
            }
        );

        // Value unrepresentable sub-nanosecond precision
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF 97103M001".to_string(),
            },
            1.23400001e-9,
            None,
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "value",
                reason: "excessive precision",
            }
        );

        // Value tiny unrepresentable precision
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF 97103M001".to_string(),
            },
            1.0e-15,
            None,
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "value",
                reason: "excessive precision",
            }
        );

        // Value exceeds format width
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF 97103M001".to_string(),
            },
            100_000.0e-9,
            None,
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "value",
                reason: "cannot fit format",
            }
        );

        // Value huge finite (f64::MAX)
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF 97103M001".to_string(),
            },
            f64::MAX,
            None,
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "value",
                reason: "cannot fit format",
            }
        );

        // Value huge negative finite (-f64::MAX)
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF 97103M001".to_string(),
            },
            -f64::MAX,
            None,
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "value",
                reason: "cannot fit format",
            }
        );

        // Value non-finite NaN
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF 97103M001".to_string(),
            },
            f64::NAN,
            None,
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "value",
                reason: "not finite",
            }
        );

        // Value non-finite infinity
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF 97103M001".to_string(),
            },
            f64::INFINITY,
            None,
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "value",
                reason: "not finite",
            }
        );

        // Sigma negative is accepted and preserves format/bit pattern
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF 97103M001".to_string(),
            },
            1.0e-9,
            Some(-0.001 * NS_TO_S),
        );
        let written = write_code_dcb(&set).unwrap();
        assert!(written.ends_with("   -0.001\n"));
        let reparsed = BiasSet::parse_code_dcb(written.as_bytes(), None)
            .unwrap()
            .value;
        assert_eq!(reparsed.skipped_records(), 0);
        assert_eq!(reparsed.records().len(), 1);
        let rep_record = &reparsed.records()[0];
        assert_eq!(rep_record.target, set.records()[0].target);
        assert_eq!(rep_record.value.to_bits(), set.records()[0].value.to_bits());
        assert_eq!(
            rep_record.sigma.unwrap().to_bits(),
            (-0.001 * NS_TO_S).to_bits()
        );
        assert!(rep_record.sigma.unwrap().is_sign_negative());

        // Sigma negative zero is accepted and preserves format/bit pattern
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF 97103M001".to_string(),
            },
            1.0e-9,
            Some(-0.0),
        );
        let written = write_code_dcb(&set).unwrap();
        assert!(written.ends_with("   -0.000\n"));

        // Sigma excessive precision
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF 97103M001".to_string(),
            },
            1.0e-9,
            Some(0.00123e-9),
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "sigma",
                reason: "excessive precision",
            }
        );

        // Sigma unrepresentable sub-nanosecond precision
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF 97103M001".to_string(),
            },
            1.0e-9,
            Some(1.23400001e-9),
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "sigma",
                reason: "excessive precision",
            }
        );

        // Sigma tiny unrepresentable precision
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF 97103M001".to_string(),
            },
            1.0e-9,
            Some(1.0e-15),
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "sigma",
                reason: "excessive precision",
            }
        );

        // Sigma huge finite (f64::MAX)
        let set = make_set(
            BiasTarget::Receiver {
                system: GnssSystem::Gps,
                station: "ABMF 97103M001".to_string(),
            },
            1.0e-9,
            Some(f64::MAX),
        );
        let err = write_code_dcb(&set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "sigma",
                reason: "cannot fit format",
            }
        );

        // Satellite target with unrepresentable numeric value
        let sat_set = make_set(BiasTarget::Satellite(sat()), 1.23456e-9, None);
        let err = write_code_dcb(&sat_set).unwrap_err();
        assert_eq!(
            err,
            BiasError::InvalidInput {
                field: "value",
                reason: "excessive precision",
            }
        );
    }

    fn generated_record(value: f64, unit: BiasUnit) -> BiasRecord {
        BiasRecord {
            kind: BiasKind::Osb,
            target: BiasTarget::Satellite(sat()),
            svn: Some("G080".to_string()),
            obs1: if unit == BiasUnit::Cycles {
                "L1C"
            } else {
                "C1C"
            }
            .to_string(),
            obs2: None,
            valid_from: Some(BiasEpoch::new(2020, 1, 0).unwrap()),
            valid_until: Some(BiasEpoch::new(2020, 2, 0).unwrap()),
            raw_epochs: ("2020:001:00000".to_string(), "2020:002:00000".to_string()),
            value,
            sigma: None,
            slope: None,
            slope_sigma: None,
            family: if unit == BiasUnit::Cycles {
                BiasObservableFamily::Phase
            } else {
                BiasObservableFamily::Code
            },
            unit,
            line: None,
        }
    }

    /// Writes a record with the generated writer and reads the row back.
    fn write_and_read(record: &BiasRecord) -> BiasRecord {
        let line = format_sinex_solution_record(record).unwrap();
        assert!(line.len() <= 137, "{line:?}");
        parse_solution_line(line.as_bytes()).unwrap()
    }

    #[test]
    fn generated_rows_read_back_to_the_same_values() {
        let cases = [
            (1.234567890123456, BiasUnit::Cycles),
            (1.234567890123456 * NS_TO_S, BiasUnit::Nanoseconds),
            (-1.2345678901234567 * NS_TO_S, BiasUnit::Nanoseconds),
            (1.0e-300, BiasUnit::Cycles),
            (1.0e300 * NS_TO_S, BiasUnit::Nanoseconds),
            (-0.0, BiasUnit::Nanoseconds),
            (0.0, BiasUnit::Cycles),
            (-6.2069 * NS_TO_S, BiasUnit::Nanoseconds),
        ];
        for (value, unit) in cases {
            let record = generated_record(value, unit);
            let read = write_and_read(&record);
            assert_eq!(read.value.to_bits(), value.to_bits(), "{value:e} {unit:?}");
            assert_eq!(read.unit, unit);
        }
        // The shortest spelling that reads back is chosen.
        assert_eq!(
            spell_sinex_numeric(
                -6.2069 * NS_TO_S,
                BiasUnit::Nanoseconds,
                SinexNumericField::Value
            )
            .unwrap(),
            "-6.2069"
        );
        // Every spelling has a decimal point. A Fortran E-edit read scales
        // digits written without one by the field's decimal count.
        for (value, unit, spelled) in [
            (0.5, BiasUnit::Cycles, ".5"),
            (-0.5e-9, BiasUnit::Cycles, "-.5E-9"),
            (-0.5 * NS_TO_S, BiasUnit::Nanoseconds, "-.5"),
            (0.0046, BiasUnit::Cycles, ".0046"),
            (-0.0, BiasUnit::Nanoseconds, "-0."),
            (1.0, BiasUnit::Cycles, "1."),
            (1.0e-9, BiasUnit::Cycles, "1.E-9"),
            (1.0e300, BiasUnit::Cycles, "1.E300"),
        ] {
            assert_eq!(
                spell_sinex_numeric(value, unit, SinexNumericField::Value).unwrap(),
                spelled
            );
        }
        for (value, unit) in cases {
            let spelled = spell_sinex_numeric(value, unit, SinexNumericField::Value).unwrap();
            assert!(spelled.contains('.'), "{spelled}");
        }
    }

    #[test]
    fn generated_rows_refuse_what_their_fields_cannot_hold() {
        // Seventeen significant digits do not fit the 11-column E11.6 field.
        assert_eq!(
            spell_sinex_numeric(
                0.12345678901234568,
                BiasUnit::Cycles,
                SinexNumericField::Sigma
            ),
            Err(BiasError::InvalidInput {
                field: "bias sigma",
                reason: "cannot fit format",
            })
        );
        // The largest finite value is 22 columns wide in exponent form.
        assert_eq!(
            spell_sinex_numeric(f64::MAX, BiasUnit::Cycles, SinexNumericField::Value),
            Err(BiasError::InvalidInput {
                field: "bias value",
                reason: "cannot fit format",
            })
        );
        assert_eq!(
            spell_sinex_numeric(f64::NAN, BiasUnit::Cycles, SinexNumericField::Value),
            Err(BiasError::InvalidInput {
                field: "bias value",
                reason: "not finite",
            })
        );
        let mut record = generated_record(1.0e-9, BiasUnit::Nanoseconds);
        record.target = BiasTarget::Receiver {
            system: GnssSystem::Gps,
            station: "ABMF 97103M001".to_string(),
        };
        assert_eq!(
            format_sinex_solution_record(&record),
            Err(BiasError::InvalidInput {
                field: "station",
                reason: "cannot fit format",
            })
        );
    }

    #[test]
    fn generated_rows_keep_a_slope_uncertainty_without_a_slope() {
        let mut record = generated_record(1.0e-9, BiasUnit::Nanoseconds);
        record.sigma = Some(0.0);
        record.slope_sigma = Some(0.0001 * NS_TO_S / SINEX_BIAS_SLOPE_DENOMINATOR_S);
        let line = format_sinex_solution_record(&record).unwrap();
        // Columns 104..125 (slope) stay blank; the uncertainty ends at 137.
        assert_eq!(byte_raw_field(line.as_bytes(), 104, 125).trim(), "");
        assert_eq!(line.len(), 137);
        let read = parse_solution_line(line.as_bytes()).unwrap();
        assert_eq!(read.slope, None);
        assert_eq!(
            read.slope_sigma.map(f64::to_bits),
            record.slope_sigma.map(f64::to_bits)
        );
        assert_eq!(read.sigma.map(f64::to_bits), Some(0.0_f64.to_bits()));
    }
}
