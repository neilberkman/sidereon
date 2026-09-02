//! Offline GNSS code and phase bias products.
//!
//! The parsers are sans-I/O: callers pass bytes, and the returned bias set
//! carries typed diagnostics for records that were skipped during forgiving
//! parsing.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write as _;
use std::str::FromStr;

use crate::astro::constants::time::SECONDS_PER_DAY_I64;
use crate::astro::time::civil::{day_of_year_int, seconds_between_splits, split_julian_date};
use crate::astro::time::model::{Instant, InstantRepr, JulianDateSplit, TimeScale};
use crate::astro::time::scales::julian_day_number;
use crate::constants::{C_M_S, NS_TO_S, SECONDS_PER_DAY};
use crate::format::columns::{field, fortran_f64, raw_field};
pub use crate::format::{Diagnostics, Parsed, RecordRef, Skip, SkipReason, Warning, WarningKind};
pub use crate::validate::FieldError;
use crate::validate::{self, CivilSecondPolicy};
use crate::{frequencies, GnssSatelliteId, GnssSystem};

const BIAS_SINEX_MAJOR_VERSION: &str = "1";
pub const SINEX_BIAS_SLOPE_DENOMINATOR_S: f64 = 1.0;
const DSB_INCONSISTENCY_TOL_S: f64 = 1.0e-15;
const RINEX_VERSION_FOR_BIAS_CODES: f64 = 3.04;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiasKind {
    Osb,
    Dsb,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BiasTarget {
    System(GnssSystem),
    Satellite(GnssSatelliteId),
    Receiver {
        system: GnssSystem,
        station: String,
    },
    SatelliteReceiver {
        sat: GnssSatelliteId,
        station: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BiasTargetKey {
    pub system: GnssSystem,
    pub sat: Option<GnssSatelliteId>,
    pub station: Option<String>,
}

impl BiasTargetKey {
    pub fn system(system: GnssSystem) -> Self {
        Self {
            system,
            sat: None,
            station: None,
        }
    }

    pub fn satellite(sat: GnssSatelliteId) -> Self {
        Self {
            system: sat.system,
            sat: Some(sat),
            station: None,
        }
    }

    pub fn receiver(system: GnssSystem, station: &str) -> Self {
        Self {
            system,
            sat: None,
            station: Some(normalize_station(station)),
        }
    }

    pub fn satellite_receiver(sat: GnssSatelliteId, station: &str) -> Self {
        Self {
            system: sat.system,
            sat: Some(sat),
            station: Some(normalize_station(station)),
        }
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
pub struct BiasEpoch {
    pub year: i32,
    pub day_of_year: u16,
    pub second_of_day: u32,
}

impl BiasEpoch {
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

    pub fn format_sinex(self) -> String {
        format!(
            "{:04}:{:03}:{:05}",
            self.year, self.day_of_year, self.second_of_day
        )
    }

    fn to_split(self) -> Result<JulianDateSplit, BiasError> {
        let jdn = julian_day_number(self.year, 1, 1) + i64::from(self.day_of_year) - 1;
        let (year, month, day) = crate::astro::time::civil::civil_from_julian_day_number(jdn);
        let (jd_whole, fraction) = split_julian_date(
            year as i32,
            month as i32,
            day as i32,
            0,
            0,
            f64::from(self.second_of_day),
        );
        JulianDateSplit::new(jd_whole, fraction).map_err(|_| BiasError::InvalidEpoch)
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
pub struct BiasRecord {
    pub kind: BiasKind,
    pub target: BiasTarget,
    pub svn: Option<String>,
    pub obs1: String,
    pub obs2: Option<String>,
    pub valid_from: Option<BiasEpoch>,
    pub valid_until: Option<BiasEpoch>,
    pub raw_epochs: (String, String),
    pub value: f64,
    pub sigma: Option<f64>,
    pub slope: Option<f64>,
    pub slope_sigma: Option<f64>,
    pub is_phase: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BiasMode {
    Absolute,
    Relative,
    #[default]
    Unspecified,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ClockReferenceObservables {
    pub per_system: BTreeMap<GnssSystem, (String, String)>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BiasSetHeader {
    pub agency: Option<String>,
    pub file_reference: Vec<(String, String)>,
    pub description: BTreeMap<String, String>,
    pub declared_bias_count: Option<usize>,
    pub dcb_meta: Option<CodeDcbOptions>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BiasSet {
    records: Vec<BiasRecord>,
    index: BTreeMap<(BiasTargetKey, String), Vec<usize>>,
    pub mode: BiasMode,
    pub time_scale: TimeScale,
    pub clock_reference: ClockReferenceObservables,
    pub header: BiasSetHeader,
    diagnostics: Diagnostics,
    skipped_records: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodeDcbOptions {
    pub pair: (String, String),
    pub year: i32,
    pub month: u8,
    pub time_scale: TimeScale,
    pub receiver_system: Option<GnssSystem>,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum BiasError {
    #[error("invalid bias input {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    #[error("invalid bias epoch")]
    InvalidEpoch,
    #[error("unknown RINEX observable code {code}")]
    UnknownObservable { code: String },
    #[error("unsupported Bias-SINEX version {version}")]
    UnsupportedVersion { version: String },
    #[error("missing DCB pair or product month")]
    MissingDcbMetadata,
    #[error("missing satellite clock reference observables")]
    MissingClockReference,
    #[error("bias writer missing required metadata {field}")]
    MissingWriterMetadata { field: &'static str },
    #[error("input is not UTF-8")]
    Utf8,
}

impl BiasSet {
    pub fn parse_bias_sinex(input: &[u8]) -> Result<Parsed<BiasSet>, BiasError> {
        let text = String::from_utf8_lossy(input);
        parse_bias_sinex_text(text.as_ref())
    }

    pub fn parse_code_dcb(
        input: &[u8],
        options: Option<CodeDcbOptions>,
    ) -> Result<Parsed<BiasSet>, BiasError> {
        let text = String::from_utf8_lossy(input);
        parse_code_dcb_text(text.as_ref(), options)
    }

    pub fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }

    pub fn skipped_records(&self) -> usize {
        self.skipped_records
    }

    pub fn code_osb_seconds(&self, sat: GnssSatelliteId, obs: &str, epoch: Instant) -> Option<f64> {
        self.osb_for_target_chain(sat, obs, epoch, false)
    }

    pub fn phase_osb_cycles(&self, sat: GnssSatelliteId, obs: &str, epoch: Instant) -> Option<f64> {
        self.osb_for_target_chain(sat, obs, epoch, true)
    }

    pub fn code_dsb_seconds(
        &self,
        sat: GnssSatelliteId,
        obs1: &str,
        obs2: &str,
        epoch: Instant,
    ) -> Option<f64> {
        self.dsb_for_keys(
            [
                BiasTargetKey::satellite(sat),
                BiasTargetKey::system(sat.system),
            ],
            obs1,
            obs2,
            epoch,
        )
    }

    pub fn receiver_code_osb_seconds(
        &self,
        system: GnssSystem,
        station: &str,
        obs: &str,
        epoch: Instant,
    ) -> Option<f64> {
        self.select_record(
            &BiasTargetKey::receiver(system, station),
            obs,
            epoch,
            |record| record.kind == BiasKind::Osb && !record.is_phase,
        )
        .and_then(|record| self.record_value_at(record, epoch))
    }

    pub fn receiver_code_dsb_seconds(
        &self,
        system: GnssSystem,
        station: &str,
        obs1: &str,
        obs2: &str,
        epoch: Instant,
    ) -> Option<f64> {
        self.dsb_for_key(&BiasTargetKey::receiver(system, station), obs1, obs2, epoch)
    }

    pub fn sat_receiver_code_osb_seconds(
        &self,
        sat: GnssSatelliteId,
        station: &str,
        obs: &str,
        epoch: Instant,
    ) -> Option<f64> {
        self.select_record(
            &BiasTargetKey::satellite_receiver(sat, station),
            obs,
            epoch,
            |record| record.kind == BiasKind::Osb && !record.is_phase,
        )
        .and_then(|record| self.record_value_at(record, epoch))
    }

    pub fn sat_receiver_code_dsb_seconds(
        &self,
        sat: GnssSatelliteId,
        station: &str,
        obs1: &str,
        obs2: &str,
        epoch: Instant,
    ) -> Option<f64> {
        self.dsb_for_key(
            &BiasTargetKey::satellite_receiver(sat, station),
            obs1,
            obs2,
            epoch,
        )
    }

    pub fn records(&self) -> &[BiasRecord] {
        &self.records
    }

    pub fn code_bias_model_m(
        &self,
        sat: GnssSatelliteId,
        used_observables: (&str, &str),
        used_frequencies_hz: (f64, f64),
        glonass_channel: Option<i8>,
        clock_reference: (&str, &str),
        epoch: Instant,
    ) -> Option<f64> {
        if used_observables == clock_reference {
            return Some(0.0);
        }
        let used_if = self.if_bias_seconds(sat, used_observables, used_frequencies_hz, epoch);
        let ref_freq1 = rinex_frequency(sat, clock_reference.0, glonass_channel)?;
        let ref_freq2 = rinex_frequency(sat, clock_reference.1, glonass_channel)?;
        let ref_if = self.if_bias_seconds(sat, clock_reference, (ref_freq1, ref_freq2), epoch);
        match (used_if, ref_if) {
            (Some(used), Some(reference)) => Some((used - reference) * C_M_S),
            _ => self
                .relative_code_bias_seconds(
                    sat,
                    used_observables,
                    used_frequencies_hz,
                    clock_reference,
                    epoch,
                )
                .map(|seconds| seconds * C_M_S),
        }
    }

    fn new(
        records: Vec<BiasRecord>,
        mode: BiasMode,
        time_scale: TimeScale,
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
            diagnostics: Diagnostics::new(),
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
            for pair in indices.windows(2) {
                let lhs = &self.records[pair[0]];
                let rhs = &self.records[pair[1]];
                if intervals_overlap(lhs, rhs) {
                    diagnostics.push_warning(Warning {
                        at: RecordRef::at_record(pair[1]),
                        kind: WarningKind::Overlap,
                    });
                }
            }
        }
        self.index = index;
    }

    fn osb_for_target_chain(
        &self,
        sat: GnssSatelliteId,
        obs: &str,
        epoch: Instant,
        phase: bool,
    ) -> Option<f64> {
        for key in [
            BiasTargetKey::satellite(sat),
            BiasTargetKey::system(sat.system),
        ] {
            if let Some(value) = self
                .select_record(&key, obs, epoch, |record| {
                    record.kind == BiasKind::Osb && record.is_phase == phase
                })
                .and_then(|record| self.record_value_at(record, epoch))
            {
                return Some(value);
            }
        }
        None
    }

    fn if_bias_seconds(
        &self,
        sat: GnssSatelliteId,
        observables: (&str, &str),
        frequencies_hz: (f64, f64),
        epoch: Instant,
    ) -> Option<f64> {
        let obs1 = self.code_osb_seconds(sat, observables.0, epoch)?;
        let obs2 = self.code_osb_seconds(sat, observables.1, epoch)?;
        let (alpha, beta) = ionosphere_free_coefficients(frequencies_hz.0, frequencies_hz.1)?;
        Some(alpha * obs1 + beta * obs2)
    }

    fn relative_code_bias_seconds(
        &self,
        sat: GnssSatelliteId,
        used_observables: (&str, &str),
        used_frequencies_hz: (f64, f64),
        clock_reference: (&str, &str),
        epoch: Instant,
    ) -> Option<f64> {
        let d1 = if used_observables.0 == clock_reference.0 {
            0.0
        } else {
            self.code_dsb_seconds(sat, used_observables.0, clock_reference.0, epoch)?
        };
        let d2 = if used_observables.1 == clock_reference.1 {
            0.0
        } else {
            self.code_dsb_seconds(sat, used_observables.1, clock_reference.1, epoch)?
        };
        let (alpha, beta) =
            ionosphere_free_coefficients(used_frequencies_hz.0, used_frequencies_hz.1)?;
        Some(alpha * d1 + beta * d2)
    }

    fn dsb_for_keys<const N: usize>(
        &self,
        keys: [BiasTargetKey; N],
        obs1: &str,
        obs2: &str,
        epoch: Instant,
    ) -> Option<f64> {
        for key in keys {
            if let Some(value) = self.dsb_for_key(&key, obs1, obs2, epoch) {
                return Some(value);
            }
        }
        None
    }

    fn dsb_for_key(
        &self,
        target_key: &BiasTargetKey,
        obs1: &str,
        obs2: &str,
        epoch: Instant,
    ) -> Option<f64> {
        if obs1 == obs2 {
            return Some(0.0);
        }
        let graph = self.dsb_graph(target_key, epoch);
        resolve_dsb_path(&graph, obs1, obs2)
    }

    fn dsb_graph(
        &self,
        target_key: &BiasTargetKey,
        epoch: Instant,
    ) -> BTreeMap<String, Vec<(String, f64)>> {
        let mut graph: BTreeMap<String, Vec<(String, f64)>> = BTreeMap::new();
        for record in &self.records {
            if BiasTargetKey::from(&record.target) != *target_key
                || record.kind != BiasKind::Dsb
                || record.is_phase
                || !self.record_covers_epoch(record, epoch)
            {
                continue;
            }
            let Some(value) = self.record_value_at(record, epoch) else {
                continue;
            };
            let Some(obs2) = record.obs2.as_ref() else {
                continue;
            };
            graph
                .entry(record.obs1.clone())
                .or_default()
                .push((obs2.clone(), value));
            graph
                .entry(obs2.clone())
                .or_default()
                .push((record.obs1.clone(), -value));
        }
        for edges in graph.values_mut() {
            edges.sort_by(|a, b| a.0.cmp(&b.0));
        }
        graph
    }

    fn select_record(
        &self,
        target_key: &BiasTargetKey,
        obs_key: &str,
        epoch: Instant,
        predicate: impl Fn(&BiasRecord) -> bool,
    ) -> Option<&BiasRecord> {
        let mut selected = None;
        let indices = self.index.get(&(target_key.clone(), obs_key.to_string()))?;
        for &index in indices {
            let record = &self.records[index];
            if predicate(record) && self.record_covers_epoch(record, epoch) {
                selected = Some(record);
            }
        }
        selected
    }

    fn record_covers_epoch(&self, record: &BiasRecord, epoch: Instant) -> bool {
        if epoch.scale != self.time_scale {
            return false;
        }
        let Some(query) = instant_split(epoch) else {
            return false;
        };
        if let Some(from) = record.valid_from {
            let Ok(from) = from.to_split() else {
                return false;
            };
            if seconds_between_splits(query.jd_whole, query.fraction, from.jd_whole, from.fraction)
                < 0.0
            {
                return false;
            }
        }
        if let Some(until) = record.valid_until {
            let Ok(until) = until.to_split() else {
                return false;
            };
            if seconds_between_splits(
                query.jd_whole,
                query.fraction,
                until.jd_whole,
                until.fraction,
            ) >= 0.0
            {
                return false;
            }
        }
        true
    }

    fn record_value_at(&self, record: &BiasRecord, epoch: Instant) -> Option<f64> {
        if !self.record_covers_epoch(record, epoch) {
            return None;
        }
        let Some(slope) = record.slope else {
            return Some(record.value);
        };
        let from = record.valid_from?.to_split().ok()?;
        let query = instant_split(epoch)?;
        let dt_s =
            seconds_between_splits(query.jd_whole, query.fraction, from.jd_whole, from.fraction);
        Some(record.value + slope * dt_s)
    }
}

// invariant: formatting into a String uses infallible fmt::Write operations.
#[allow(clippy::expect_used)]
pub fn write_bias_sinex(set: &BiasSet) -> Result<String, BiasError> {
    let agency = set
        .header
        .agency
        .as_deref()
        .ok_or(BiasError::MissingWriterMetadata { field: "agency" })?;
    if set.mode == BiasMode::Unspecified {
        return Err(BiasError::MissingWriterMetadata { field: "mode" });
    }
    if set.clock_reference.per_system.is_empty() {
        return Err(BiasError::MissingWriterMetadata {
            field: "clock_reference",
        });
    }

    let mut out = String::new();
    writeln!(&mut out, "%=BIA 1.00 {agency}").expect("write string");
    writeln!(&mut out, "+FILE/REFERENCE").expect("write string");
    if set.header.file_reference.is_empty() {
        writeln!(&mut out, " DESCRIPTION GENERATED").expect("write string");
    } else {
        for (key, value) in &set.header.file_reference {
            writeln!(&mut out, " {key} {value}").expect("write string");
        }
    }
    writeln!(&mut out, "-FILE/REFERENCE").expect("write string");
    writeln!(&mut out, "+BIAS/DESCRIPTION").expect("write string");
    writeln!(
        &mut out,
        " BIAS_MODE {}",
        match set.mode {
            BiasMode::Absolute => "ABSOLUTE",
            BiasMode::Relative => "RELATIVE",
            BiasMode::Unspecified => "UNSPECIFIED",
        }
    )
    .expect("write string");
    writeln!(
        &mut out,
        " TIME_SYSTEM {}",
        time_scale_sinex_label(set.time_scale)
    )
    .expect("write string");
    for (system, (obs1, obs2)) in &set.clock_reference.per_system {
        writeln!(
            &mut out,
            " SATELLITE_CLOCK_REFERENCE_OBSERVABLES {} {obs1} {obs2}",
            system.letter()
        )
        .expect("write string");
    }
    for (key, value) in &set.header.description {
        if matches!(
            key.as_str(),
            "BIAS_MODE" | "TIME_SYSTEM" | "SATELLITE_CLOCK_REFERENCE_OBSERVABLES"
        ) {
            continue;
        }
        writeln!(&mut out, " {key} {value}").expect("write string");
    }
    writeln!(&mut out, "-BIAS/DESCRIPTION").expect("write string");
    writeln!(&mut out, "+BIAS/SOLUTION {}", set.records.len()).expect("write string");
    writeln!(
        &mut out,
        "*BIAS SVN_ PRN STATION__ OBS1 OBS2 BIAS_START____ BIAS_END______ UNIT __ESTIMATED_VALUE____ _STD_DEV___"
    )
    .expect("write string");
    for record in &set.records {
        writeln!(&mut out, "{}", format_sinex_solution_record(record)).expect("write string");
    }
    writeln!(&mut out, "-BIAS/SOLUTION").expect("write string");
    Ok(out)
}

// invariant: formatting into a String uses infallible fmt::Write operations.
#[allow(clippy::expect_used)]
pub fn write_code_dcb(set: &BiasSet) -> Result<String, BiasError> {
    let meta = set
        .header
        .dcb_meta
        .as_ref()
        .ok_or(BiasError::MissingWriterMetadata { field: "dcb_meta" })?;
    let mut out = String::new();
    writeln!(
        &mut out,
        "# DCB {}-{} {:04}-{:02} {}",
        meta.pair.0,
        meta.pair.1,
        meta.year,
        meta.month,
        time_scale_sinex_label(meta.time_scale)
    )
    .expect("write string");
    writeln!(&mut out, " PRN / STATION NAME        VALUE (ns)  RMS (ns)").expect("write string");
    writeln!(
        &mut out,
        "***   ****************      *********.***  *****.***"
    )
    .expect("write string");
    for record in &set.records {
        if record.kind != BiasKind::Dsb || record.is_phase {
            return Err(BiasError::InvalidInput {
                field: "bias set",
                reason: "CODE DCB writer requires code DSB records",
            });
        }
        let value_ns = record.value / NS_TO_S;
        let sigma_ns = record.sigma.unwrap_or(0.0) / NS_TO_S;
        match &record.target {
            BiasTarget::Satellite(sat) => {
                writeln!(
                    &mut out,
                    " {sat:<3}                       {value_ns:10.3} {sigma_ns:10.3}"
                )
                .expect("write string");
            }
            BiasTarget::Receiver { station, .. } => {
                writeln!(
                    &mut out,
                    "      {station:<16}       {value_ns:10.3} {sigma_ns:10.3}"
                )
                .expect("write string");
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

fn parse_bias_sinex_text(text: &str) -> Result<Parsed<BiasSet>, BiasError> {
    let mut diagnostics = Diagnostics::new();
    let mut header = BiasSetHeader::default();
    let mut mode = BiasMode::Unspecified;
    let mut time_scale = TimeScale::Gpst;
    let mut clock_reference = ClockReferenceObservables::default();
    let mut records = Vec::new();
    let mut current_block: Option<String> = None;
    let mut saw_file_reference = false;
    let mut solution_count_line = None;

    let first = text.lines().next().ok_or(BiasError::InvalidInput {
        field: "input",
        reason: "empty",
    })?;
    parse_bias_sinex_header(first, &mut header)?;

    for (line_index, line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let trimmed = line.trim_end();
        if line_number == 1 || trimmed.is_empty() {
            continue;
        }
        if let Some(block) = trimmed.strip_prefix('+') {
            let mut parts = block.split_whitespace();
            let name = parts.next().unwrap_or("").to_string();
            if name == "BIAS/SOLUTION" {
                solution_count_line = Some(line_number);
                if let Some(count) = parts.next() {
                    if let Ok(count) = count.parse::<usize>() {
                        header.declared_bias_count = Some(count);
                    }
                }
            } else if !matches!(name.as_str(), "FILE/REFERENCE" | "BIAS/DESCRIPTION") {
                diagnostics.push_skip(Skip {
                    at: RecordRef::at_line(line_number),
                    reason: SkipReason::UnknownBlock(name.clone()),
                });
            }
            if name == "FILE/REFERENCE" {
                saw_file_reference = true;
            }
            current_block = Some(name);
            continue;
        }
        if trimmed.starts_with('-') {
            current_block = None;
            continue;
        }
        if trimmed.starts_with('*') {
            continue;
        }

        match current_block.as_deref() {
            Some("FILE/REFERENCE") => parse_file_reference_line(trimmed, &mut header),
            Some("BIAS/DESCRIPTION") => parse_description_line(
                trimmed,
                &mut header,
                &mut mode,
                &mut time_scale,
                &mut clock_reference,
                &mut diagnostics,
                line_number,
            ),
            Some("BIAS/SOLUTION") => match parse_solution_line(trimmed) {
                Ok(record) => {
                    if mode == BiasMode::Absolute && record.kind != BiasKind::Osb {
                        diagnostics.push_warning(Warning {
                            at: RecordRef::at_line(line_number),
                            kind: WarningKind::Mismatch,
                        });
                    }
                    if mode == BiasMode::Relative && record.kind == BiasKind::Osb {
                        diagnostics.push_warning(Warning {
                            at: RecordRef::at_line(line_number),
                            kind: WarningKind::Mismatch,
                        });
                    }
                    records.push(record);
                }
                Err(reason) => diagnostics.push_skip(Skip {
                    at: RecordRef::at_line(line_number),
                    reason,
                }),
            },
            Some(_) | None => {}
        }
    }

    if !saw_file_reference {
        diagnostics.push_warning(Warning {
            at: RecordRef::at_line(1),
            kind: WarningKind::MissingMetadata,
        });
    }
    if let Some(declared) = header.declared_bias_count {
        if declared != records.len() {
            diagnostics.push_warning(Warning {
                at: RecordRef::at_line(solution_count_line.unwrap_or(1)),
                kind: WarningKind::Mismatch,
            });
        }
    }

    let set = BiasSet::new(
        records,
        mode,
        time_scale,
        clock_reference,
        header,
        diagnostics,
    );
    let diagnostics = set.diagnostics.clone();
    Ok(Parsed::new(set, diagnostics))
}

fn parse_code_dcb_text(
    text: &str,
    options: Option<CodeDcbOptions>,
) -> Result<Parsed<BiasSet>, BiasError> {
    let title_meta = parse_dcb_title_metadata(text);
    let options = match (options, title_meta) {
        (Some(options), Some(title)) => {
            if options.pair != title.pair
                || options.year != title.year
                || options.month != title.month
                || options.time_scale != title.time_scale
            {
                return Err(BiasError::InvalidInput {
                    field: "CodeDcbOptions",
                    reason: "does not match title metadata",
                });
            }
            options
        }
        (Some(options), None) => options,
        (None, Some(title)) => title,
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
    for (line_index, line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with('*')
            || trimmed.starts_with("PRN")
            || trimmed.contains("VALUE")
            || !looks_like_dcb_row(line, &options)
        {
            continue;
        }
        let row = match parse_dcb_row(line, &options) {
            Ok(Some(row)) => row,
            Ok(None) => continue,
            Err(reason) => {
                diagnostics.push_skip(Skip {
                    at: RecordRef::at_line(line_number),
                    reason,
                });
                continue;
            }
        };
        let Some((obs1, obs2)) = map_legacy_dcb_pair(row.system, &options.pair.0, &options.pair.1)
        else {
            diagnostics.push_skip(Skip {
                at: RecordRef::at_line(line_number),
                reason: SkipReason::UnsupportedRecordType("DCB_PAIR"),
            });
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
            is_phase: false,
        });
    }

    let mut header = BiasSetHeader {
        dcb_meta: Some(options.clone()),
        ..BiasSetHeader::default()
    };
    header.description.insert(
        "DCB_PAIR".to_string(),
        format!("{} {}", options.pair.0, options.pair.1),
    );
    let set = BiasSet::new(
        records,
        BiasMode::Relative,
        options.time_scale,
        ClockReferenceObservables::default(),
        header,
        diagnostics,
    );
    let diagnostics = set.diagnostics.clone();
    Ok(Parsed::new(set, diagnostics))
}

struct DcbRow {
    target: BiasTarget,
    system: GnssSystem,
    value_ns: f64,
    sigma_ns: Option<f64>,
}

fn parse_dcb_row(line: &str, options: &CodeDcbOptions) -> Result<Option<DcbRow>, SkipReason> {
    let Some(value_token) = field(line, 24, 38) else {
        return Ok(None);
    };
    let value_ns = fortran_f64(line, 24, 38, "dcb value").ok_or_else(|| {
        SkipReason::MalformedField(FieldError::FloatParse {
            field: "dcb value",
            value: value_token.to_string(),
        })
    })?;
    let sigma_ns = fortran_f64(line, 38, 50, "dcb sigma");

    if let Some(sat_token) = field(line, 0, 4).filter(|token| looks_like_satellite_token(token)) {
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

    if let Some(system) =
        field(line, 0, 1).and_then(|token| token.chars().next().and_then(GnssSystem::from_letter))
    {
        let station = field(line, 6, 10).or_else(|| field(line, 6, 22)).ok_or(
            SkipReason::InconsistentRecord("receiver DCB record lacks station"),
        )?;
        return Ok(Some(DcbRow {
            target: BiasTarget::Receiver {
                system,
                station: normalize_station(station),
            },
            system,
            value_ns,
            sigma_ns,
        }));
    }

    if let Some(system) = options.receiver_system {
        let station = field(line, 6, 22).or_else(|| field(line, 0, 22)).ok_or(
            SkipReason::InconsistentRecord("receiver DCB record lacks station"),
        )?;
        return Ok(Some(DcbRow {
            target: BiasTarget::Receiver {
                system,
                station: normalize_station(station),
            },
            system,
            value_ns,
            sigma_ns,
        }));
    }

    Err(SkipReason::InconsistentRecord(
        "receiver DCB record lacks system",
    ))
}

fn looks_like_dcb_row(line: &str, options: &CodeDcbOptions) -> bool {
    if field(line, 0, 4).is_some_and(looks_like_satellite_token) {
        return true;
    }
    if field(line, 0, 1)
        .and_then(|token| token.chars().next().and_then(GnssSystem::from_letter))
        .is_some()
        && raw_field(line, 1, 6).trim().is_empty()
    {
        return true;
    }
    options.receiver_system.is_some()
        && field(line, 24, 38).is_some_and(|token| validate::strict_f64(token, "dcb value").is_ok())
}

fn parse_bias_sinex_header(line: &str, header: &mut BiasSetHeader) -> Result<(), BiasError> {
    let mut tokens = line.split_whitespace();
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
    if !version.starts_with(BIAS_SINEX_MAJOR_VERSION) {
        return Err(BiasError::UnsupportedVersion {
            version: version.to_string(),
        });
    }
    header.agency = tokens.next().map(str::to_string);
    Ok(())
}

fn parse_file_reference_line(line: &str, header: &mut BiasSetHeader) {
    let mut parts = line.split_whitespace();
    if let Some(key) = parts.next() {
        let value = parts.collect::<Vec<_>>().join(" ");
        header.file_reference.push((key.to_string(), value));
    }
}

fn parse_description_line(
    line: &str,
    header: &mut BiasSetHeader,
    mode: &mut BiasMode,
    time_scale: &mut TimeScale,
    clock_reference: &mut ClockReferenceObservables,
    diagnostics: &mut Diagnostics,
    line_number: usize,
) {
    let mut parts = line.split_whitespace();
    let Some(key) = parts.next() else {
        return;
    };
    match key {
        "BIAS_MODE" => {
            let value = parts.next().unwrap_or("");
            *mode = match value {
                "ABSOLUTE" => BiasMode::Absolute,
                "RELATIVE" => BiasMode::Relative,
                _ => BiasMode::Unspecified,
            };
            header
                .description
                .insert(key.to_string(), value.to_string());
        }
        "TIME_SYSTEM" => {
            let value = parts.next().unwrap_or("");
            match parse_sinex_time_scale(value) {
                Some(scale) => {
                    *time_scale = scale;
                    header
                        .description
                        .insert(key.to_string(), value.to_string());
                }
                None => diagnostics.push_skip(Skip {
                    at: RecordRef::at_line(line_number),
                    reason: SkipReason::MalformedField(FieldError::FloatParse {
                        field: "time system",
                        value: value.to_string(),
                    }),
                }),
            }
        }
        "SATELLITE_CLOCK_REFERENCE_OBSERVABLES" => {
            let system_token = parts.next().unwrap_or("");
            let obs1 = parts.next().unwrap_or("");
            let obs2 = parts.next().unwrap_or("");
            if let Some(system) = system_token
                .chars()
                .next()
                .and_then(GnssSystem::from_letter)
            {
                clock_reference
                    .per_system
                    .insert(system, (obs1.to_string(), obs2.to_string()));
                header.description.insert(
                    format!("{key}_{}", system.letter()),
                    format!("{obs1} {obs2}"),
                );
            } else {
                diagnostics.push_skip(Skip {
                    at: RecordRef::at_line(line_number),
                    reason: SkipReason::MalformedField(FieldError::IntParse {
                        field: "system",
                        value: system_token.to_string(),
                    }),
                });
            }
        }
        _ => {
            let value = parts.collect::<Vec<_>>().join(" ");
            header.description.insert(key.to_string(), value);
        }
    }
}

fn parse_solution_line(line: &str) -> Result<BiasRecord, SkipReason> {
    if line.len() < 91 {
        return Err(SkipReason::Truncated);
    }
    let kind = field(line, 1, 5)
        .ok_or(SkipReason::Truncated)?
        .parse::<BiasKind>()
        .map_err(|_| SkipReason::UnsupportedRecordType("BIAS"))?;
    let svn = field(line, 6, 10).map(str::to_string);
    let prn = field(line, 11, 14);
    let station = field(line, 15, 24).map(normalize_station);
    let obs1 = field(line, 25, 29)
        .ok_or(SkipReason::Truncated)?
        .to_string();
    let obs2 = field(line, 30, 34).map(str::to_string);
    let raw_start = raw_field(line, 35, 49).trim().to_string();
    let raw_end = raw_field(line, 50, 64).trim().to_string();
    let unit = field(line, 65, 69).ok_or(SkipReason::Truncated)?;
    let value = fortran_f64(line, 70, 91, "bias value").ok_or_else(|| {
        SkipReason::MalformedField(FieldError::FloatParse {
            field: "bias value",
            value: raw_field(line, 70, 91).trim().to_string(),
        })
    })?;
    let sigma = fortran_f64(line, 92, 103, "bias sigma");
    let slope = fortran_f64(line, 104, 125, "bias slope");
    let slope_sigma = fortran_f64(line, 126, 137, "bias slope sigma");
    let is_phase = match unit {
        "ns" => false,
        "cyc" => true,
        other => return Err(SkipReason::UnsupportedUnit(other.to_string())),
    };
    if kind == BiasKind::Osb && obs2.is_some() {
        return Err(SkipReason::InconsistentRecord("OSB must not carry OBS2"));
    }
    if kind != BiasKind::Osb && obs2.is_none() {
        return Err(SkipReason::InconsistentRecord("DSB or ISB requires OBS2"));
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
    let target = parse_bias_target(svn.as_deref(), prn, station.as_deref())?;
    let value = if is_phase { value } else { value * NS_TO_S };
    let sigma = sigma.map(|sigma| if is_phase { sigma } else { sigma * NS_TO_S });
    let slope = slope.map(|slope| {
        if is_phase {
            slope / SINEX_BIAS_SLOPE_DENOMINATOR_S
        } else {
            slope * NS_TO_S / SINEX_BIAS_SLOPE_DENOMINATOR_S
        }
    });
    let slope_sigma = slope_sigma.map(|slope_sigma| {
        if is_phase {
            slope_sigma / SINEX_BIAS_SLOPE_DENOMINATOR_S
        } else {
            slope_sigma * NS_TO_S / SINEX_BIAS_SLOPE_DENOMINATOR_S
        }
    });
    Ok(BiasRecord {
        kind,
        target,
        svn,
        obs1,
        obs2,
        valid_from,
        valid_until,
        raw_epochs: (raw_start, raw_end),
        value,
        sigma,
        slope,
        slope_sigma,
        is_phase,
    })
}

fn parse_bias_target(
    _svn: Option<&str>,
    prn: Option<&str>,
    station: Option<&str>,
) -> Result<BiasTarget, SkipReason> {
    match (prn, station) {
        (Some(prn), None) if prn.len() == 1 => {
            let system = prn.chars().next().and_then(GnssSystem::from_letter).ok_or(
                SkipReason::MalformedField(FieldError::IntParse {
                    field: "system",
                    value: prn.to_string(),
                }),
            )?;
            Ok(BiasTarget::System(system))
        }
        (Some(prn), Some(station)) if prn.len() == 1 => {
            let system = prn.chars().next().and_then(GnssSystem::from_letter).ok_or(
                SkipReason::MalformedField(FieldError::IntParse {
                    field: "system",
                    value: prn.to_string(),
                }),
            )?;
            Ok(BiasTarget::Receiver {
                system,
                station: normalize_station(station),
            })
        }
        (Some(prn), None) => prn
            .parse::<GnssSatelliteId>()
            .map(BiasTarget::Satellite)
            .map_err(|_| SkipReason::UnrepresentableSatellite),
        (Some(prn), Some(station)) => prn
            .parse::<GnssSatelliteId>()
            .map(|sat| BiasTarget::SatelliteReceiver {
                sat,
                station: normalize_station(station),
            })
            .map_err(|_| SkipReason::UnrepresentableSatellite),
        (None, Some(_)) | (None, None) => Err(SkipReason::InconsistentRecord("missing PRN")),
    }
}

fn parse_dcb_title_metadata(text: &str) -> Option<CodeDcbOptions> {
    for line in text.lines().take(12) {
        let mut pair = None;
        let mut year = None;
        let mut month = None;
        let mut scale = None;
        let tokens = line
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
            scale = scale.or_else(|| parse_sinex_time_scale(token));
        }
        if let (Some(pair), Some(year), Some(month)) = (pair, year, month) {
            return Some(CodeDcbOptions {
                pair,
                year,
                month,
                time_scale: scale.unwrap_or(TimeScale::Gpst),
                receiver_system: None,
            });
        }
    }
    None
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
    Ok(())
}

fn dcb_month_interval(year: i32, month: u8) -> Result<(BiasEpoch, Option<BiasEpoch>), BiasError> {
    let start = month_epoch(year, month)?;
    let (next_year, next_month) = if month == 12 {
        (year + 1, 1)
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

// invariant: formatting into a String uses infallible fmt::Write operations.
#[allow(clippy::expect_used)]
fn format_sinex_solution_record(record: &BiasRecord) -> String {
    let (svn, prn, station) = target_fields(record);
    let obs2 = record.obs2.as_deref().unwrap_or("");
    let (unit, value, sigma, slope, slope_sigma) = if record.is_phase {
        (
            "cyc",
            record.value,
            record.sigma,
            record.slope.map(|s| s * SINEX_BIAS_SLOPE_DENOMINATOR_S),
            record
                .slope_sigma
                .map(|s| s * SINEX_BIAS_SLOPE_DENOMINATOR_S),
        )
    } else {
        (
            "ns",
            record.value / NS_TO_S,
            record.sigma.map(|s| s / NS_TO_S),
            record
                .slope
                .map(|s| s / NS_TO_S * SINEX_BIAS_SLOPE_DENOMINATOR_S),
            record
                .slope_sigma
                .map(|s| s / NS_TO_S * SINEX_BIAS_SLOPE_DENOMINATOR_S),
        )
    };
    let start = if record.raw_epochs.0.is_empty() {
        record
            .valid_from
            .map(BiasEpoch::format_sinex)
            .unwrap_or_else(|| "0000:000:00000".to_string())
    } else {
        record.raw_epochs.0.clone()
    };
    let end = if record.raw_epochs.1.is_empty() {
        record
            .valid_until
            .map(BiasEpoch::format_sinex)
            .unwrap_or_else(|| "0000:000:00000".to_string())
    } else {
        record.raw_epochs.1.clone()
    };
    let mut line = format!(
        " {:<4} {:<4} {:<3} {:<9} {:<4} {:<4} {:<14} {:<14} {:<4} {:>21.12E}",
        record.kind.label(),
        svn,
        prn,
        station,
        record.obs1,
        obs2,
        start,
        end,
        unit,
        value
    );
    if let Some(sigma) = sigma {
        write!(&mut line, " {sigma:>11.5E}").expect("write string");
    } else {
        line.push_str(&" ".repeat(12));
    }
    if let Some(slope) = slope {
        write!(&mut line, " {slope:>21.12E}").expect("write string");
        if let Some(slope_sigma) = slope_sigma {
            write!(&mut line, " {slope_sigma:>11.5E}").expect("write string");
        }
    }
    line
}

fn target_fields(record: &BiasRecord) -> (String, String, String) {
    match &record.target {
        BiasTarget::System(system) => (
            record.svn.clone().unwrap_or_default(),
            system.letter().to_string(),
            String::new(),
        ),
        BiasTarget::Satellite(sat) => (
            record.svn.clone().unwrap_or_default(),
            sat.to_string(),
            String::new(),
        ),
        BiasTarget::Receiver { system, station } => (
            record.svn.clone().unwrap_or_default(),
            system.letter().to_string(),
            station.clone(),
        ),
        BiasTarget::SatelliteReceiver { sat, station } => (
            record.svn.clone().unwrap_or_default(),
            sat.to_string(),
            station.clone(),
        ),
    }
}

fn resolve_dsb_path(
    graph: &BTreeMap<String, Vec<(String, f64)>>,
    start: &str,
    end: &str,
) -> Option<f64> {
    let mut queue = VecDeque::new();
    let mut best_depth = None;
    let mut candidates = Vec::new();
    queue.push_back((vec![start.to_string()], 0.0));

    while let Some((path, value)) = queue.pop_front() {
        let depth = path.len() - 1;
        if best_depth.is_some_and(|best| depth > best) {
            continue;
        }
        let Some(node) = path.last() else {
            continue;
        };
        if node == end {
            best_depth = Some(depth);
            candidates.push((path, value));
            continue;
        }
        for (next, edge_value) in graph.get(node).into_iter().flatten() {
            if path.iter().any(|seen| seen == next) {
                continue;
            }
            let mut next_path = path.clone();
            next_path.push(next.clone());
            queue.push_back((next_path, value + edge_value));
        }
    }

    candidates.sort_by(|a, b| a.0.cmp(&b.0));
    let (_, first) = candidates.first()?;
    if candidates
        .iter()
        .any(|(_, value)| (value - first).abs() > DSB_INCONSISTENCY_TOL_S)
    {
        return None;
    }
    Some(*first)
}

fn compare_record_start(a: &BiasRecord, b: &BiasRecord) -> Ordering {
    a.valid_from
        .cmp(&b.valid_from)
        .then_with(|| a.raw_epochs.0.cmp(&b.raw_epochs.0))
}

fn intervals_overlap(a: &BiasRecord, b: &BiasRecord) -> bool {
    let a_start = a.valid_from;
    let b_start = b.valid_from;
    let a_end = a.valid_until;
    let b_end = b.valid_until;
    let a_before_b_end = b_end.is_none_or(|end| a_start.is_none_or(|start| start < end));
    let b_before_a_end = a_end.is_none_or(|end| b_start.is_none_or(|start| start < end));
    a_before_b_end && b_before_a_end
}

fn obs_key(record: &BiasRecord) -> String {
    match record.kind {
        BiasKind::Osb => record.obs1.clone(),
        BiasKind::Dsb | BiasKind::Isb => {
            format!("{}-{}", record.obs1, record.obs2.as_deref().unwrap_or(""))
        }
    }
}

fn normalize_station(station: &str) -> String {
    let upper = station.trim().to_ascii_uppercase();
    if upper.len() > 4 && upper.as_bytes()[..4].iter().all(u8::is_ascii_alphanumeric) {
        upper[..4].to_string()
    } else {
        upper
    }
}

fn instant_split(epoch: Instant) -> Option<JulianDateSplit> {
    match epoch.repr {
        InstantRepr::JulianDate(split) => Some(split),
        InstantRepr::Nanos(nanos) => {
            let seconds = nanos as f64 * NS_TO_S;
            let days = seconds.div_euclid(SECONDS_PER_DAY);
            let rem = seconds.rem_euclid(SECONDS_PER_DAY);
            JulianDateSplit::new(crate::constants::J2000_JD + days, rem / SECONDS_PER_DAY).ok()
        }
    }
}

pub fn bias_epoch_instant(epoch: BiasEpoch, scale: TimeScale) -> Result<Instant, BiasError> {
    Ok(Instant::from_julian_date(scale, epoch.to_split()?))
}

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

fn parse_sinex_time_scale(label: &str) -> Option<TimeScale> {
    match label.trim() {
        "G" | "GPS" | "GPST" => Some(TimeScale::Gpst),
        "R" | "GLO" | "UTC" => Some(TimeScale::Utc),
        "E" | "GAL" | "GST" => Some(TimeScale::Gst),
        "C" | "BDT" => Some(TimeScale::Bdt),
        "J" | "QZS" | "QZSST" => Some(TimeScale::Qzsst),
        "TAI" => Some(TimeScale::Tai),
        "TCG" => Some(TimeScale::Tcg),
        "TCB" => Some(TimeScale::Tcb),
        _ => None,
    }
}

fn time_scale_sinex_label(scale: TimeScale) -> &'static str {
    match scale {
        TimeScale::Gpst => "G",
        TimeScale::Utc | TimeScale::Glonasst => "R",
        TimeScale::Gst => "E",
        TimeScale::Bdt => "C",
        TimeScale::Qzsst => "J",
        TimeScale::Tai => "TAI",
        TimeScale::Tt => "TT",
        TimeScale::Tcg => "TCG",
        TimeScale::Tdb => "TDB",
        TimeScale::Tcb => "TCB",
    }
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
        let mut graph = BTreeMap::new();
        graph.insert(
            "C1C".to_string(),
            vec![("C1W".to_string(), 1.0), ("C1P".to_string(), 2.0)],
        );
        graph.insert("C1P".to_string(), vec![("C1W".to_string(), 3.0)]);
        assert_eq!(resolve_dsb_path(&graph, "C1C", "C1W"), Some(1.0));
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
            is_phase: true,
        };
        let set = BiasSet::new(
            vec![record],
            BiasMode::Absolute,
            TimeScale::Gpst,
            ClockReferenceObservables::default(),
            BiasSetHeader::default(),
            Diagnostics::new(),
        );
        assert_eq!(set.code_osb_seconds(sat(), "L1C", epoch(2020, 1, 0)), None);
        assert_eq!(
            set.phase_osb_cycles(sat(), "L1C", epoch(2020, 1, 0)),
            Some(-0.25)
        );
    }
}
