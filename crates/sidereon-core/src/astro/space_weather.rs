//! CelesTrak CSSI space-weather parsing and NRLMSISE-00 lookup.
//!
//! The parser is sans-IO: callers supply bytes or text from the CelesTrak
//! CSV or fixed-width product. Lookups return the existing drag input type used
//! by the NRLMSISE-00 drag path.
//!
//! References:
//! - Picone, Hedin, Drob, and Aikin, "NRLMSISE-00 empirical model of the
//!   atmosphere", JGR 107(A12), 2002.
//! - CelesTrak CSSI space-weather format:
//!   <https://celestrak.org/SpaceData/SpaceWx-format.php>.

use crate::astro::atmosphere::{ApArray, DEFAULT_AP};
use crate::astro::constants::time::SECONDS_PER_DAY_I64;
use crate::astro::forces::SpaceWeather;
use crate::astro::time::civil::{
    civil_from_julian_day_number, days_in_month, j2000_seconds, J2000_JULIAN_DAY_NUMBER,
    J2000_NOON_OFFSET_S,
};
use crate::astro::time::scales::julian_day_number;
use crate::format::columns;
pub use crate::format::{Diagnostics, Parsed, RecordRef, Skip, SkipReason, Warning, WarningKind};
use crate::validate;
use crate::validate::FieldError;
use std::fmt::Write as _;

const CSV_HEADER: &str = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81";
const TXT_HEADER_COMMENTS: &str = "\
# --------------------------------------------------------------------------------------------------------------------------------\n\
#                              SPACE WEATHER DATA\n\
# --------------------------------------------------------------------------------------------------------------------------------\n\
#\n\
# See https://celestrak.org/SpaceData/SpaceWx-format.php for format details.\n\
#\n\
# FORMAT(I4,I3,I3,I5,I3,8I3,I4,8I4,I4,F4.1,I2,I4,F6.1,I2,5F6.1)\n\
# --------------------------------------------------------------------------------------------------------------------------------\n\
#                                                                                             Adj     Adj   Adj   Obs   Obs   Obs \n\
# yy mm dd BSRN ND Kp Kp Kp Kp Kp Kp Kp Kp Sum Ap  Ap  Ap  Ap  Ap  Ap  Ap  Ap  Avg Cp C9 ISN F10.7 Q Ctr81 Lst81 F10.7 Ctr81 Lst81\n\
# --------------------------------------------------------------------------------------------------------------------------------\n\
#";

/// Provenance class of one daily CelesTrak space-weather row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ObservationClass {
    /// Observed row (`OBS` in CSV, `OBSERVED` txt section with qualifier other than 4).
    Observed,
    /// Source-interpolated observed row (`INT` in CSV, `OBSERVED` txt with qualifier 4).
    Interpolated,
    /// Daily predicted row (`PRD` in CSV, `DAILY_PREDICTED` txt section).
    DailyPredicted,
    /// Monthly predicted row (`PRM` in CSV, `MONTHLY_PREDICTED` txt section).
    MonthlyPredicted,
}

/// One CelesTrak CSSI daily space-weather record.
///
/// Fields are stored at source quantization so serializers can reproduce the
/// product widths and decimal precision. Missing source fields are `None`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpaceWeatherDay {
    /// UTC civil year.
    pub year: i32,
    /// UTC civil month in `1..=12`.
    pub month: u8,
    /// UTC civil day of month.
    pub day: u8,
    /// Row provenance and prediction class.
    pub class: ObservationClass,
    /// Bartels solar rotation number.
    pub bsrn: Option<u16>,
    /// Day within the 27-day Bartels rotation.
    pub nd: Option<u8>,
    /// Kp per 3-hour UT bin, stored as source Kp times 10.
    pub kp_10: [Option<u16>; 8],
    /// Daily sum of the eight stored Kp values.
    pub kp_sum_10: Option<u16>,
    /// Planetary equivalent amplitude `ap` per 3-hour UT bin.
    pub ap: [Option<u16>; 8],
    /// Daily Ap index (`AP_AVG`).
    pub ap_avg: Option<u16>,
    /// Planetary daily character figure Cp, stored times 10.
    pub cp_10: Option<u8>,
    /// C9 index.
    pub c9: Option<u8>,
    /// International sunspot number.
    pub isn: Option<u16>,
    /// Fixed-width text flux qualifier; absent in CSV.
    pub flux_qualifier: Option<u8>,
    /// Observed F10.7 flux at actual Earth-Sun distance, sfu.
    pub f107_obs: Option<f64>,
    /// F10.7 flux adjusted to 1 AU, sfu.
    pub f107_adj: Option<f64>,
    /// Centered 81-day average of observed F10.7, sfu.
    pub f107_obs_center81: Option<f64>,
    /// Trailing 81-day average of observed F10.7, sfu.
    pub f107_obs_last81: Option<f64>,
    /// Centered 81-day average of adjusted F10.7, sfu.
    pub f107_adj_center81: Option<f64>,
    /// Trailing 81-day average of adjusted F10.7, sfu.
    pub f107_adj_last81: Option<f64>,
}

impl SpaceWeatherDay {
    /// Kp for one 3-hour bin as the physical Kp value, not times 10.
    pub fn kp(&self, bin: usize) -> Option<f64> {
        self.kp_10
            .get(bin)
            .and_then(|v| v.map(|v| f64::from(v) / 10.0))
    }

    /// Cp as the physical one-decimal value.
    pub fn cp(&self) -> Option<f64> {
        self.cp_10.map(|v| f64::from(v) / 10.0)
    }

    fn jdn(&self) -> i64 {
        julian_day_number(self.year, i32::from(self.month), i32::from(self.day))
    }
}

/// Time-indexed CelesTrak CSSI space-weather table.
#[derive(Debug, Clone, PartialEq)]
pub struct SpaceWeatherTable {
    days: Vec<SpaceWeatherDay>,
    monthly: Vec<SpaceWeatherDay>,
    txt_updated: Option<String>,
}

/// Coverage summary for a parsed space-weather table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpaceWeatherCoverage {
    /// Start of the first covered UTC day, as J2000 seconds.
    pub first_j2000_s: f64,
    /// Start of the last observed daily row, as J2000 seconds.
    pub last_observed_j2000_s: Option<f64>,
    /// Start of the last daily-predicted row, as J2000 seconds.
    pub last_daily_predicted_j2000_s: Option<f64>,
    /// End of the last covered day overall, as an exclusive J2000-second bound.
    pub end_j2000_s: f64,
}

/// Errors from parsing CelesTrak space-weather products or looking up samples.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SpaceWeatherError {
    /// Input is neither the CSV schema nor the CSSI fixed-width text schema.
    #[error("unrecognized space-weather format")]
    UnrecognizedFormat,
    /// Structural failure that prevents a table from being built.
    #[error("malformed space-weather input at line {line}: {reason}")]
    Malformed {
        /// Source line associated with the structural failure. CSV header failures and an empty table use `1`; fixed-width section errors use the current one-based line, while an unterminated section uses `text.lines().count()`.
        line: usize,
        /// Owned description of the structural cause, including missing or unexpected CSV headers, nested or mismatched fixed-width sections, an unterminated section, or no parseable rows. The error message places it after the line number and a colon.
        reason: String,
    },
    /// Input bytes are not UTF-8 text.
    #[error("space-weather input is not valid UTF-8")]
    NotText,
    /// Lookup before the table can supply required rows.
    #[error("space-weather lookup before coverage")]
    BeforeCoverage {
        /// Requested epoch as J2000 seconds.
        requested_j2000_s: f64,
        /// Start of table coverage as J2000 seconds.
        first_j2000_s: f64,
    },
    /// Lookup at or beyond the table's exclusive coverage end.
    #[error("space-weather lookup after coverage")]
    AfterCoverage {
        /// Requested epoch as J2000 seconds.
        requested_j2000_s: f64,
        /// End of table coverage as J2000 seconds.
        end_j2000_s: f64,
    },
    /// A covered day or required field is missing.
    #[error("space-weather data missing {field} on {year:04}-{month:02}-{day:02}")]
    MissingData {
        /// UTC civil year.
        year: i32,
        /// UTC civil month.
        month: u8,
        /// UTC civil day.
        day: u8,
        /// Missing field name.
        field: &'static str,
    },
    /// Lookup policy rejected a row class.
    #[error("space-weather row class rejected by policy")]
    RejectedByPolicy {
        /// Rejected row class.
        class: ObservationClass,
        /// UTC civil year.
        year: i32,
        /// UTC civil month.
        month: u8,
        /// UTC civil day.
        day: u8,
    },
    /// Lookup epoch was non-finite or outside the representable civil range.
    #[error("invalid space-weather epoch")]
    InvalidEpoch {
        /// `f64::to_bits()` of the rejected epoch.
        epoch_j2000_s_bits: u64,
    },
}

/// NRLMSISE-00 drag input plus metadata about the consulted rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpaceWeatherSample {
    /// Space-weather values to feed the drag path.
    pub space_weather: SpaceWeather,
    /// Least-trusted class among the rows consulted.
    pub class: ObservationClass,
    /// True when monthly-predicted geomagnetic data defaulted to quiet Ap.
    pub ap_defaulted: bool,
}

/// Lookup policy for rejecting lower-trust rows or monthly Ap defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpaceWeatherPolicy {
    /// Permit source-interpolated observed rows.
    pub allow_interpolated: bool,
    /// Permit daily predicted rows.
    pub allow_daily_predicted: bool,
    /// Permit monthly predicted rows.
    pub allow_monthly_predicted: bool,
    /// Reject monthly rows instead of using the quiet geomagnetic default.
    pub require_geomagnetic: bool,
}

impl Default for SpaceWeatherPolicy {
    fn default() -> Self {
        Self {
            allow_interpolated: true,
            allow_daily_predicted: true,
            allow_monthly_predicted: true,
            require_geomagnetic: false,
        }
    }
}

impl SpaceWeatherTable {
    /// All daily-resolution rows, sorted by date.
    pub fn days(&self) -> &[SpaceWeatherDay] {
        &self.days
    }

    /// All monthly-predicted rows, sorted by date.
    pub fn monthly(&self) -> &[SpaceWeatherDay] {
        &self.monthly
    }

    /// Row for a UTC civil date, including monthly holdover when applicable.
    pub fn day(&self, year: i32, month: u8, day: u8) -> Option<&SpaceWeatherDay> {
        let jdn = julian_day_number(year, i32::from(month), i32::from(day));
        self.day_by_jdn(jdn)
    }

    /// Summary of the dates covered by this table.
    pub fn coverage(&self) -> SpaceWeatherCoverage {
        let first_jdn = self.first_jdn().expect("nonempty table");
        let end_jdn = self.end_jdn().expect("nonempty table");
        SpaceWeatherCoverage {
            first_j2000_s: day_start_j2000_s(first_jdn),
            last_observed_j2000_s: self
                .days
                .iter()
                .rfind(|row| matches!(row.class, ObservationClass::Observed))
                .map(|row| day_start_j2000_s(row.jdn())),
            last_daily_predicted_j2000_s: self
                .days
                .iter()
                .rfind(|row| matches!(row.class, ObservationClass::DailyPredicted))
                .map(|row| day_start_j2000_s(row.jdn())),
            end_j2000_s: day_start_j2000_s(end_jdn),
        }
    }

    /// NRLMSISE-00 conventional space-weather inputs at an epoch.
    ///
    /// `epoch_j2000_s` is treated as UT seconds on the same convention used by
    /// the drag module for `CartesianState::epoch_tdb_seconds`; the TDB-UT
    /// difference is below one minute against daily and 3-hour source data.
    pub fn space_weather_at(&self, epoch_j2000_s: f64) -> Result<SpaceWeather, SpaceWeatherError> {
        self.sample_at(epoch_j2000_s)
            .map(|sample| sample.space_weather)
    }

    /// Lookup a sample with default policy, returning values plus row metadata.
    pub fn sample_at(&self, epoch_j2000_s: f64) -> Result<SpaceWeatherSample, SpaceWeatherError> {
        self.sample_at_with_policy(epoch_j2000_s, SpaceWeatherPolicy::default())
    }

    /// Lookup a sample with explicit row-class and geomagnetic-default policy.
    pub fn sample_at_with_policy(
        &self,
        epoch_j2000_s: f64,
        policy: SpaceWeatherPolicy,
    ) -> Result<SpaceWeatherSample, SpaceWeatherError> {
        let jdn = epoch_day_jdn(epoch_j2000_s)?;
        self.check_epoch_coverage(epoch_j2000_s, jdn, true)?;

        let today = self.required_day(jdn, epoch_j2000_s)?;
        let previous = self.required_day(jdn - 1, epoch_j2000_s)?;
        enforce_policy(today, policy)?;
        enforce_policy(previous, policy)?;

        let f107 = previous
            .f107_obs
            .ok_or_else(|| missing(previous, "F10.7_OBS"))?;
        let f107a = today
            .f107_obs_center81
            .ok_or_else(|| missing(today, "F10.7_OBS_CENTER81"))?;
        let (ap, ap_defaulted) = daily_ap(today, policy)?;

        Ok(SpaceWeatherSample {
            space_weather: SpaceWeather { f107, f107a, ap },
            class: today.class.max(previous.class),
            ap_defaulted,
        })
    }

    /// Build the NRLMSISE-00 seven-element Ap history array at an epoch.
    pub fn ap_array_at(&self, epoch_j2000_s: f64) -> Result<ApArray, SpaceWeatherError> {
        let (jdn, bin) = epoch_day_and_ap_bin(epoch_j2000_s)?;
        self.check_epoch_coverage(epoch_j2000_s, jdn, false)?;
        let today = self.required_day(jdn, epoch_j2000_s)?;
        let (daily, _) = daily_ap(today, SpaceWeatherPolicy::default())?;
        let slot = jdn * 8 + i64::from(bin);

        Ok([
            daily,
            self.ap_slot(slot, epoch_j2000_s)?,
            self.ap_slot(slot - 1, epoch_j2000_s)?,
            self.ap_slot(slot - 2, epoch_j2000_s)?,
            self.ap_slot(slot - 3, epoch_j2000_s)?,
            self.mean_ap_slots(slot - 11, slot - 4, epoch_j2000_s)?,
            self.mean_ap_slots(slot - 19, slot - 12, epoch_j2000_s)?,
        ])
    }

    fn day_by_jdn(&self, jdn: i64) -> Option<&SpaceWeatherDay> {
        if let Ok(index) = self.days.binary_search_by_key(&jdn, SpaceWeatherDay::jdn) {
            return self.days.get(index);
        }
        let index = self
            .monthly
            .binary_search_by_key(&jdn, SpaceWeatherDay::jdn)
            .unwrap_or_else(|index| index.saturating_sub(1));
        let row = self.monthly.get(index)?;
        let (year, month, _day) = civil_from_julian_day_number(jdn);
        if row.year == year as i32 && row.month == month as u8 {
            Some(row)
        } else {
            None
        }
    }

    fn required_day(
        &self,
        jdn: i64,
        requested_j2000_s: f64,
    ) -> Result<&SpaceWeatherDay, SpaceWeatherError> {
        self.day_by_jdn(jdn).ok_or_else(|| {
            if jdn < self.first_jdn().expect("nonempty table") {
                SpaceWeatherError::BeforeCoverage {
                    requested_j2000_s,
                    first_j2000_s: self.coverage().first_j2000_s,
                }
            } else if jdn >= self.end_jdn().expect("nonempty table") {
                SpaceWeatherError::AfterCoverage {
                    requested_j2000_s,
                    end_j2000_s: self.coverage().end_j2000_s,
                }
            } else {
                let (year, month, day) = civil_from_julian_day_number(jdn);
                SpaceWeatherError::MissingData {
                    year: year as i32,
                    month: month as u8,
                    day: day as u8,
                    field: "record",
                }
            }
        })
    }

    fn check_epoch_coverage(
        &self,
        requested_j2000_s: f64,
        jdn: i64,
        needs_previous_day: bool,
    ) -> Result<(), SpaceWeatherError> {
        let first_jdn = self.first_jdn().expect("nonempty table");
        let end_jdn = self.end_jdn().expect("nonempty table");
        let required_first = if needs_previous_day {
            first_jdn + 1
        } else {
            first_jdn
        };
        if jdn < required_first {
            return Err(SpaceWeatherError::BeforeCoverage {
                requested_j2000_s,
                first_j2000_s: self.coverage().first_j2000_s,
            });
        }
        if jdn >= end_jdn {
            return Err(SpaceWeatherError::AfterCoverage {
                requested_j2000_s,
                end_j2000_s: day_start_j2000_s(end_jdn),
            });
        }
        Ok(())
    }

    fn ap_slot(&self, slot: i64, requested_j2000_s: f64) -> Result<f64, SpaceWeatherError> {
        let jdn = slot.div_euclid(8);
        let bin = slot.rem_euclid(8) as usize;
        let row = self.required_day(jdn, requested_j2000_s)?;
        if let Some(ap) = row.ap[bin] {
            return Ok(f64::from(ap));
        }
        daily_ap(row, SpaceWeatherPolicy::default()).map(|(ap, _)| ap)
    }

    fn mean_ap_slots(
        &self,
        first_slot: i64,
        last_slot: i64,
        requested_j2000_s: f64,
    ) -> Result<f64, SpaceWeatherError> {
        let mut sum = 0.0;
        let mut count = 0.0;
        for slot in first_slot..=last_slot {
            sum += self.ap_slot(slot, requested_j2000_s)?;
            count += 1.0;
        }
        Ok(sum / count)
    }

    fn first_jdn(&self) -> Option<i64> {
        match (self.days.first(), self.monthly.first()) {
            (Some(a), Some(b)) => Some(a.jdn().min(b.jdn())),
            (Some(a), None) => Some(a.jdn()),
            (None, Some(b)) => Some(b.jdn()),
            (None, None) => None,
        }
    }

    fn end_jdn(&self) -> Option<i64> {
        let day_end = self.days.last().map(|row| row.jdn() + 1);
        let monthly_end = self.monthly.last().map(|row| {
            let next_month = if row.month == 12 {
                (row.year + 1, 1)
            } else {
                (row.year, i32::from(row.month) + 1)
            };
            julian_day_number(next_month.0, next_month.1, 1)
        });
        match (day_end, monthly_end) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }
}

/// Parse the CelesTrak CSV encoding.
pub fn parse_csv(text: &str) -> Result<Parsed<SpaceWeatherTable>, SpaceWeatherError> {
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| SpaceWeatherError::Malformed {
        line: 1,
        reason: "missing CSV header".to_string(),
    })?;
    if header.trim_end_matches('\r') != CSV_HEADER {
        return Err(SpaceWeatherError::Malformed {
            line: 1,
            reason: "unexpected CSV header".to_string(),
        });
    }

    let mut records = Vec::new();
    let mut diagnostics = Diagnostics::new();
    let mut previous_jdn = None;
    for (zero_index, raw_line) in lines.enumerate() {
        let line_no = zero_index + 2;
        let line = raw_line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        match parse_csv_record(line) {
            Ok(row) => {
                let jdn = row.jdn();
                if previous_jdn.is_some_and(|previous| jdn < previous) {
                    diagnostics.push_skip(skip_line(
                        line_no,
                        SkipReason::InconsistentRecord("out-of-order date"),
                    ));
                    continue;
                }
                previous_jdn = Some(jdn);
                records.push((line_no, row));
            }
            Err(reason) => diagnostics.push_skip(skip_line(line_no, reason)),
        }
    }
    build_table(records, diagnostics, None)
}

/// Parse the CelesTrak CSSI fixed-width text encoding.
pub fn parse_txt(text: &str) -> Result<Parsed<SpaceWeatherTable>, SpaceWeatherError> {
    let mut saw_datatype = false;
    let mut section = None;
    let mut txt_updated = None;
    let mut records = Vec::new();
    let mut diagnostics = Diagnostics::new();
    let mut observed_count = None;
    let mut daily_count = None;
    let mut monthly_count = None;
    let mut parsed_observed = 0usize;
    let mut parsed_daily = 0usize;
    let mut parsed_monthly = 0usize;
    let mut previous_jdn = None;

    for (zero_index, raw_line) in text.lines().enumerate() {
        let line_no = zero_index + 1;
        let line = raw_line.trim_end_matches('\r');
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed == "DATATYPE CssiSpaceWeather" {
            saw_datatype = true;
            continue;
        }
        if trimmed.starts_with("VERSION ") {
            continue;
        }
        if let Some(updated) = trimmed.strip_prefix("UPDATED ") {
            txt_updated = Some(updated.to_string());
            continue;
        }
        if let Some(count) = trimmed.strip_prefix("NUM_OBSERVED_POINTS ") {
            observed_count = parse_count(count).map(|count| (line_no, count));
            continue;
        }
        if let Some(count) = trimmed.strip_prefix("NUM_DAILY_PREDICTED_POINTS ") {
            daily_count = parse_count(count).map(|count| (line_no, count));
            continue;
        }
        if let Some(count) = trimmed.strip_prefix("NUM_MONTHLY_PREDICTED_POINTS ") {
            monthly_count = parse_count(count).map(|count| (line_no, count));
            continue;
        }
        match trimmed {
            "BEGIN OBSERVED" => {
                if section.is_some() {
                    return Err(SpaceWeatherError::Malformed {
                        line: line_no,
                        reason: "nested fixed-width section".to_string(),
                    });
                }
                section = Some(TxtSection::Observed);
                continue;
            }
            "END OBSERVED" => {
                if section != Some(TxtSection::Observed) {
                    return Err(SpaceWeatherError::Malformed {
                        line: line_no,
                        reason: "END OBSERVED without matching BEGIN".to_string(),
                    });
                }
                section = None;
                continue;
            }
            "BEGIN DAILY_PREDICTED" => {
                if section.is_some() {
                    return Err(SpaceWeatherError::Malformed {
                        line: line_no,
                        reason: "nested fixed-width section".to_string(),
                    });
                }
                section = Some(TxtSection::DailyPredicted);
                continue;
            }
            "END DAILY_PREDICTED" => {
                if section != Some(TxtSection::DailyPredicted) {
                    return Err(SpaceWeatherError::Malformed {
                        line: line_no,
                        reason: "END DAILY_PREDICTED without matching BEGIN".to_string(),
                    });
                }
                section = None;
                continue;
            }
            "BEGIN MONTHLY_PREDICTED" => {
                if section.is_some() {
                    return Err(SpaceWeatherError::Malformed {
                        line: line_no,
                        reason: "nested fixed-width section".to_string(),
                    });
                }
                section = Some(TxtSection::MonthlyPredicted);
                continue;
            }
            "END MONTHLY_PREDICTED" => {
                if section != Some(TxtSection::MonthlyPredicted) {
                    return Err(SpaceWeatherError::Malformed {
                        line: line_no,
                        reason: "END MONTHLY_PREDICTED without matching BEGIN".to_string(),
                    });
                }
                section = None;
                continue;
            }
            _ => {}
        }

        let Some(active_section) = section else {
            continue;
        };
        match parse_txt_record(line, active_section) {
            Ok(row) => {
                let jdn = row.jdn();
                if previous_jdn.is_some_and(|previous| jdn < previous) {
                    diagnostics.push_skip(skip_line(
                        line_no,
                        SkipReason::InconsistentRecord("out-of-order date"),
                    ));
                    continue;
                }
                previous_jdn = Some(jdn);
                match active_section {
                    TxtSection::Observed => parsed_observed += 1,
                    TxtSection::DailyPredicted => parsed_daily += 1,
                    TxtSection::MonthlyPredicted => parsed_monthly += 1,
                }
                records.push((line_no, row));
            }
            Err(error) => {
                diagnostics.push_skip(skip_line(line_no, SkipReason::MalformedField(error)))
            }
        }
    }

    if !saw_datatype {
        return Err(SpaceWeatherError::UnrecognizedFormat);
    }
    if section.is_some() {
        return Err(SpaceWeatherError::Malformed {
            line: text.lines().count(),
            reason: "unterminated fixed-width section".to_string(),
        });
    }
    warn_count_mismatch(observed_count, parsed_observed, &mut diagnostics);
    warn_count_mismatch(daily_count, parsed_daily, &mut diagnostics);
    warn_count_mismatch(monthly_count, parsed_monthly, &mut diagnostics);
    build_table(records, diagnostics, txt_updated)
}

/// Sniff and parse either CelesTrak space-weather encoding from UTF-8 bytes.
pub fn parse(data: &[u8]) -> Result<Parsed<SpaceWeatherTable>, SpaceWeatherError> {
    let text = std::str::from_utf8(data).map_err(|_| SpaceWeatherError::NotText)?;
    let first_content = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .ok_or(SpaceWeatherError::UnrecognizedFormat)?;
    if first_content == CSV_HEADER {
        parse_csv(text)
    } else if first_content == "DATATYPE CssiSpaceWeather" {
        parse_txt(text)
    } else {
        Err(SpaceWeatherError::UnrecognizedFormat)
    }
}

fn parse_csv_record(line: &str) -> Result<SpaceWeatherDay, SkipReason> {
    let fields: Vec<_> = line.split(',').collect();
    if fields.len() != 31 {
        return Err(if fields.len() < 31 {
            SkipReason::Truncated
        } else {
            SkipReason::InconsistentRecord("CSV column count")
        });
    }
    let (year, month, day) = parse_csv_date(fields[0]).map_err(SkipReason::MalformedField)?;
    let class = parse_csv_class(fields[26]).map_err(SkipReason::MalformedField)?;
    let mut kp_10 = [None; 8];
    for (idx, slot) in kp_10.iter_mut().enumerate() {
        *slot = opt_u16(fields[3 + idx], "KP").map_err(SkipReason::MalformedField)?;
    }
    let mut ap = [None; 8];
    for (idx, slot) in ap.iter_mut().enumerate() {
        *slot = opt_u16(fields[12 + idx], "AP").map_err(SkipReason::MalformedField)?;
    }
    Ok(SpaceWeatherDay {
        year,
        month,
        day,
        class,
        bsrn: opt_u16(fields[1], "BSRN").map_err(SkipReason::MalformedField)?,
        nd: opt_u8(fields[2], "ND").map_err(SkipReason::MalformedField)?,
        kp_10,
        kp_sum_10: opt_u16(fields[11], "KP_SUM").map_err(SkipReason::MalformedField)?,
        ap,
        ap_avg: opt_u16(fields[20], "AP_AVG").map_err(SkipReason::MalformedField)?,
        cp_10: opt_cp_10(fields[21]).map_err(SkipReason::MalformedField)?,
        c9: opt_u8(fields[22], "C9").map_err(SkipReason::MalformedField)?,
        isn: opt_u16(fields[23], "ISN").map_err(SkipReason::MalformedField)?,
        flux_qualifier: None,
        f107_obs: opt_f64(fields[24], "F10.7_OBS").map_err(SkipReason::MalformedField)?,
        f107_adj: opt_f64(fields[25], "F10.7_ADJ").map_err(SkipReason::MalformedField)?,
        f107_obs_center81: opt_f64(fields[27], "F10.7_OBS_CENTER81")
            .map_err(SkipReason::MalformedField)?,
        f107_obs_last81: opt_f64(fields[28], "F10.7_OBS_LAST81")
            .map_err(SkipReason::MalformedField)?,
        f107_adj_center81: opt_f64(fields[29], "F10.7_ADJ_CENTER81")
            .map_err(SkipReason::MalformedField)?,
        f107_adj_last81: opt_f64(fields[30], "F10.7_ADJ_LAST81")
            .map_err(SkipReason::MalformedField)?,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TxtSection {
    Observed,
    DailyPredicted,
    MonthlyPredicted,
}

fn parse_txt_record(line: &str, section: TxtSection) -> Result<SpaceWeatherDay, FieldError> {
    let year = req_i32_col(line, 0, 4, "year")?;
    let month = req_u8_col(line, 4, 7, "month")?;
    let day = req_u8_col(line, 7, 10, "day")?;
    validate_date(year, month, day)?;
    let mut pos = 10;
    let bsrn = opt_u16_col(line, pos, pos + 5, "BSRN")?;
    pos += 5;
    let nd = opt_u8_col(line, pos, pos + 3, "ND")?;
    pos += 3;
    let mut kp_10 = [None; 8];
    for slot in &mut kp_10 {
        *slot = opt_u16_col(line, pos, pos + 3, "KP")?;
        pos += 3;
    }
    let kp_sum_10 = opt_u16_col(line, pos, pos + 4, "KP_SUM")?;
    pos += 4;
    let mut ap = [None; 8];
    for slot in &mut ap {
        *slot = opt_u16_col(line, pos, pos + 4, "AP")?;
        pos += 4;
    }
    let ap_avg = opt_u16_col(line, pos, pos + 4, "AP_AVG")?;
    pos += 4;
    let cp_10 = opt_cp_10_col(line, pos, pos + 4)?;
    pos += 4;
    let c9 = opt_u8_col(line, pos, pos + 2, "C9")?;
    pos += 2;
    let isn = opt_u16_col(line, pos, pos + 4, "ISN")?;
    pos += 4;
    let f107_adj = opt_f64_col(line, pos, pos + 6, "F10.7_ADJ")?;
    pos += 6;
    let flux_qualifier = opt_u8_col(line, pos, pos + 2, "Q")?;
    pos += 2;
    let f107_adj_center81 = opt_f64_col(line, pos, pos + 6, "F10.7_ADJ_CENTER81")?;
    pos += 6;
    let f107_adj_last81 = opt_f64_col(line, pos, pos + 6, "F10.7_ADJ_LAST81")?;
    pos += 6;
    let f107_obs = opt_f64_col(line, pos, pos + 6, "F10.7_OBS")?;
    pos += 6;
    let f107_obs_center81 = opt_f64_col(line, pos, pos + 6, "F10.7_OBS_CENTER81")?;
    pos += 6;
    let f107_obs_last81 = opt_f64_col(line, pos, pos + 6, "F10.7_OBS_LAST81")?;

    let class = match section {
        TxtSection::Observed if flux_qualifier == Some(4) => ObservationClass::Interpolated,
        TxtSection::Observed => ObservationClass::Observed,
        TxtSection::DailyPredicted => ObservationClass::DailyPredicted,
        TxtSection::MonthlyPredicted => ObservationClass::MonthlyPredicted,
    };

    Ok(SpaceWeatherDay {
        year,
        month,
        day,
        class,
        bsrn,
        nd,
        kp_10,
        kp_sum_10,
        ap,
        ap_avg,
        cp_10,
        c9,
        isn,
        flux_qualifier,
        f107_obs,
        f107_adj,
        f107_obs_center81,
        f107_obs_last81,
        f107_adj_center81,
        f107_adj_last81,
    })
}

fn build_table(
    records: Vec<(usize, SpaceWeatherDay)>,
    mut diagnostics: Diagnostics,
    txt_updated: Option<String>,
) -> Result<Parsed<SpaceWeatherTable>, SpaceWeatherError> {
    let mut days = Vec::new();
    let mut monthly = Vec::new();
    for (line, row) in records {
        let target = if row.class == ObservationClass::MonthlyPredicted {
            &mut monthly
        } else {
            &mut days
        };
        if target
            .last()
            .is_some_and(|existing: &SpaceWeatherDay| existing.jdn() == row.jdn())
        {
            diagnostics.push_skip(skip_line(
                line,
                SkipReason::InconsistentRecord("duplicate date"),
            ));
            continue;
        }
        target.push(row);
    }
    days.sort_by_key(SpaceWeatherDay::jdn);
    monthly.sort_by_key(SpaceWeatherDay::jdn);
    if days.is_empty() && monthly.is_empty() {
        return Err(SpaceWeatherError::Malformed {
            line: 1,
            reason: "no parseable space-weather rows".to_string(),
        });
    }
    Ok(Parsed::new(
        SpaceWeatherTable {
            days,
            monthly,
            txt_updated,
        },
        diagnostics,
    ))
}

/// Serialize a table to the CelesTrak CSV encoding.
pub fn encode_csv(table: &SpaceWeatherTable) -> String {
    let mut out = String::new();
    out.push_str(CSV_HEADER);
    out.push('\n');
    for row in table.days.iter().chain(table.monthly.iter()) {
        push_csv_row(&mut out, row);
    }
    out
}

/// Serialize a table to the CelesTrak CSSI fixed-width text encoding.
///
/// When the table came from CSV, the text-only flux qualifier field is blank.
/// When the table came from txt and had an `UPDATED` header, that header is
/// reproduced.
pub fn encode_txt(table: &SpaceWeatherTable) -> String {
    let mut out = String::new();
    out.push_str("DATATYPE CssiSpaceWeather\n");
    out.push_str("VERSION 1.2\n");
    if let Some(updated) = &table.txt_updated {
        let _ = writeln!(out, "UPDATED {updated}");
    }
    out.push_str(TXT_HEADER_COMMENTS);
    out.push('\n');

    let observed_count = table
        .days
        .iter()
        .filter(|row| {
            matches!(
                row.class,
                ObservationClass::Observed | ObservationClass::Interpolated
            )
        })
        .count();
    let daily_count = table
        .days
        .iter()
        .filter(|row| row.class == ObservationClass::DailyPredicted)
        .count();
    write_txt_section(
        &mut out,
        "OBSERVED",
        observed_count,
        table.days.iter().filter(|row| {
            matches!(
                row.class,
                ObservationClass::Observed | ObservationClass::Interpolated
            )
        }),
    );
    out.push('\n');
    write_txt_section(
        &mut out,
        "DAILY_PREDICTED",
        daily_count,
        table
            .days
            .iter()
            .filter(|row| row.class == ObservationClass::DailyPredicted),
    );
    out.push('\n');
    write_txt_section(
        &mut out,
        "MONTHLY_PREDICTED",
        table.monthly.len(),
        table.monthly.iter(),
    );
    out
}

fn push_csv_row(out: &mut String, row: &SpaceWeatherDay) {
    let class = match row.class {
        ObservationClass::Observed => "OBS",
        ObservationClass::Interpolated => "INT",
        ObservationClass::DailyPredicted => "PRD",
        ObservationClass::MonthlyPredicted => "PRM",
    };
    let _ = write!(out, "{:04}-{:02}-{:02}", row.year, row.month, row.day);
    push_csv_opt_u16(out, row.bsrn);
    push_csv_opt_u8(out, row.nd);
    for value in row.kp_10 {
        push_csv_opt_u16(out, value);
    }
    push_csv_opt_u16(out, row.kp_sum_10);
    for value in row.ap {
        push_csv_opt_u16(out, value);
    }
    push_csv_opt_u16(out, row.ap_avg);
    push_csv_opt_cp(out, row.cp_10);
    push_csv_opt_u8(out, row.c9);
    push_csv_opt_u16(out, row.isn);
    push_csv_opt_f64(out, row.f107_obs);
    push_csv_opt_f64(out, row.f107_adj);
    out.push(',');
    out.push_str(class);
    push_csv_opt_f64(out, row.f107_obs_center81);
    push_csv_opt_f64(out, row.f107_obs_last81);
    push_csv_opt_f64(out, row.f107_adj_center81);
    push_csv_opt_f64(out, row.f107_adj_last81);
    out.push('\n');
}

fn push_csv_opt_u16(out: &mut String, value: Option<u16>) {
    out.push(',');
    if let Some(value) = value {
        let _ = write!(out, "{value}");
    }
}

fn push_csv_opt_u8(out: &mut String, value: Option<u8>) {
    out.push(',');
    if let Some(value) = value {
        let _ = write!(out, "{value}");
    }
}

fn push_csv_opt_cp(out: &mut String, value: Option<u8>) {
    out.push(',');
    if let Some(value) = value {
        let _ = write!(out, "{:.1}", f64::from(value) / 10.0);
    }
}

fn push_csv_opt_f64(out: &mut String, value: Option<f64>) {
    out.push(',');
    if let Some(value) = value {
        let _ = write!(out, "{value:.1}");
    }
}

fn write_txt_section<'a, I>(out: &mut String, name: &str, count: usize, rows: I)
where
    I: IntoIterator<Item = &'a SpaceWeatherDay>,
{
    let _ = writeln!(out, "NUM_{name}_POINTS {count}");
    let _ = writeln!(out, "BEGIN {name}");
    for row in rows {
        push_txt_row(out, row);
    }
    let _ = writeln!(out, "END {name}");
}

fn push_txt_row(out: &mut String, row: &SpaceWeatherDay) {
    let _ = write!(out, "{:4} {:02} {:02}", row.year, row.month, row.day);
    push_txt_opt_u16(out, row.bsrn, 5);
    push_txt_opt_u8(out, row.nd, 3);
    for value in row.kp_10 {
        push_txt_opt_u16(out, value, 3);
    }
    push_txt_opt_u16(out, row.kp_sum_10, 4);
    for value in row.ap {
        push_txt_opt_u16(out, value, 4);
    }
    push_txt_opt_u16(out, row.ap_avg, 4);
    push_txt_opt_cp(out, row.cp_10);
    push_txt_opt_u8(out, row.c9, 2);
    push_txt_opt_u16(out, row.isn, 4);
    push_txt_opt_f64(out, row.f107_adj, 6);
    push_txt_opt_u8(out, row.flux_qualifier, 2);
    push_txt_opt_f64(out, row.f107_adj_center81, 6);
    push_txt_opt_f64(out, row.f107_adj_last81, 6);
    push_txt_opt_f64(out, row.f107_obs, 6);
    push_txt_opt_f64(out, row.f107_obs_center81, 6);
    push_txt_opt_f64(out, row.f107_obs_last81, 6);
    out.push('\n');
}

fn push_txt_opt_u16(out: &mut String, value: Option<u16>, width: usize) {
    if let Some(value) = value {
        let _ = write!(out, "{value:width$}");
    } else {
        push_spaces(out, width);
    }
}

fn push_txt_opt_u8(out: &mut String, value: Option<u8>, width: usize) {
    if let Some(value) = value {
        let _ = write!(out, "{value:width$}");
    } else {
        push_spaces(out, width);
    }
}

fn push_txt_opt_cp(out: &mut String, value: Option<u8>) {
    if let Some(value) = value {
        let _ = write!(out, "{:4.1}", f64::from(value) / 10.0);
    } else {
        push_spaces(out, 4);
    }
}

fn push_txt_opt_f64(out: &mut String, value: Option<f64>, width: usize) {
    if let Some(value) = value {
        let formatted = format!("{value:.1}");
        let _ = write!(out, "{formatted:>width$}");
    } else {
        push_spaces(out, width);
    }
}

fn push_spaces(out: &mut String, count: usize) {
    for _ in 0..count {
        out.push(' ');
    }
}

fn parse_csv_date(text: &str) -> Result<(i32, u8, u8), FieldError> {
    let mut parts = text.split('-');
    let year = parse_required(parts.next(), "year")?;
    let month = parse_required(parts.next(), "month")?;
    let day = parse_required(parts.next(), "day")?;
    if parts.next().is_some() {
        return Err(FieldError::InvalidCivilDate {
            field: "DATE",
            year: i64::from(year),
            month: i64::from(month),
            day: i64::from(day),
        });
    }
    validate_date(year, month, day)?;
    Ok((year, month, day))
}

fn parse_csv_class(text: &str) -> Result<ObservationClass, FieldError> {
    match text.trim() {
        "OBS" => Ok(ObservationClass::Observed),
        "INT" => Ok(ObservationClass::Interpolated),
        "PRD" => Ok(ObservationClass::DailyPredicted),
        "PRM" => Ok(ObservationClass::MonthlyPredicted),
        value => Err(FieldError::IntParse {
            field: "F10.7_DATA_TYPE",
            value: value.to_string(),
        }),
    }
}

fn validate_date(year: i32, month: u8, day: u8) -> Result<(), FieldError> {
    let days = days_in_month(i64::from(year), i64::from(month));
    if days == 0 || day == 0 || i64::from(day) > days {
        return Err(FieldError::InvalidCivilDate {
            field: "DATE",
            year: i64::from(year),
            month: i64::from(month),
            day: i64::from(day),
        });
    }
    Ok(())
}

fn opt_u16(text: &str, field: &'static str) -> Result<Option<u16>, FieldError> {
    opt_parse(text, field)
}

fn opt_u8(text: &str, field: &'static str) -> Result<Option<u8>, FieldError> {
    opt_parse(text, field)
}

fn opt_f64(text: &str, field: &'static str) -> Result<Option<f64>, FieldError> {
    let value = text.trim();
    if value.is_empty() {
        Ok(None)
    } else {
        validate::strict_f64(value, field).map(Some)
    }
}

fn opt_cp_10(text: &str) -> Result<Option<u8>, FieldError> {
    opt_f64(text, "CP").map(|value| value.map(|cp| (cp * 10.0).round() as u8))
}

fn opt_parse<T>(text: &str, field: &'static str) -> Result<Option<T>, FieldError>
where
    T: std::str::FromStr,
{
    let value = text.trim();
    if value.is_empty() {
        Ok(None)
    } else {
        validate::strict_int(value, field).map(Some)
    }
}

fn parse_required<T>(value: Option<&str>, field: &'static str) -> Result<T, FieldError>
where
    T: std::str::FromStr,
{
    validate::strict_int(value.unwrap_or_default(), field)
}

fn req_i32_col(
    line: &str,
    start: usize,
    end: usize,
    field: &'static str,
) -> Result<i32, FieldError> {
    validate::strict_int(columns::field(line, start, end).unwrap_or_default(), field)
}

fn req_u8_col(line: &str, start: usize, end: usize, field: &'static str) -> Result<u8, FieldError> {
    validate::strict_int(columns::field(line, start, end).unwrap_or_default(), field)
}

fn opt_u16_col(
    line: &str,
    start: usize,
    end: usize,
    field: &'static str,
) -> Result<Option<u16>, FieldError> {
    columns::field(line, start, end).map_or(Ok(None), |value| opt_u16(value, field))
}

fn opt_u8_col(
    line: &str,
    start: usize,
    end: usize,
    field: &'static str,
) -> Result<Option<u8>, FieldError> {
    columns::field(line, start, end).map_or(Ok(None), |value| opt_u8(value, field))
}

fn opt_f64_col(
    line: &str,
    start: usize,
    end: usize,
    field: &'static str,
) -> Result<Option<f64>, FieldError> {
    columns::field(line, start, end).map_or(Ok(None), |value| opt_f64(value, field))
}

fn opt_cp_10_col(line: &str, start: usize, end: usize) -> Result<Option<u8>, FieldError> {
    columns::field(line, start, end).map_or(Ok(None), opt_cp_10)
}

fn parse_count(text: &str) -> Option<usize> {
    text.trim().parse::<usize>().ok()
}

fn warn_count_mismatch(
    declared: Option<(usize, usize)>,
    actual: usize,
    diagnostics: &mut Diagnostics,
) {
    if let Some((line, _)) = declared.filter(|(_, declared)| *declared != actual) {
        diagnostics.push_warning(Warning {
            at: RecordRef::at_line(line),
            kind: WarningKind::Mismatch,
        });
    }
}

fn skip_line(line: usize, reason: SkipReason) -> Skip {
    Skip {
        at: RecordRef::at_line(line),
        reason,
    }
}

fn enforce_policy(
    row: &SpaceWeatherDay,
    policy: SpaceWeatherPolicy,
) -> Result<(), SpaceWeatherError> {
    let allowed = match row.class {
        ObservationClass::Observed => true,
        ObservationClass::Interpolated => policy.allow_interpolated,
        ObservationClass::DailyPredicted => policy.allow_daily_predicted,
        ObservationClass::MonthlyPredicted => policy.allow_monthly_predicted,
    };
    if allowed {
        Ok(())
    } else {
        Err(SpaceWeatherError::RejectedByPolicy {
            class: row.class,
            year: row.year,
            month: row.month,
            day: row.day,
        })
    }
}

fn daily_ap(
    row: &SpaceWeatherDay,
    policy: SpaceWeatherPolicy,
) -> Result<(f64, bool), SpaceWeatherError> {
    if let Some(ap) = row.ap_avg {
        return Ok((f64::from(ap), false));
    }
    if row.class == ObservationClass::MonthlyPredicted {
        if policy.require_geomagnetic {
            return Err(SpaceWeatherError::RejectedByPolicy {
                class: row.class,
                year: row.year,
                month: row.month,
                day: row.day,
            });
        }
        return Ok((DEFAULT_AP, true));
    }
    Err(missing(row, "AP_AVG"))
}

fn missing(row: &SpaceWeatherDay, field: &'static str) -> SpaceWeatherError {
    SpaceWeatherError::MissingData {
        year: row.year,
        month: row.month,
        day: row.day,
        field,
    }
}

fn epoch_day_jdn(epoch_j2000_s: f64) -> Result<i64, SpaceWeatherError> {
    if !epoch_j2000_s.is_finite() {
        return Err(SpaceWeatherError::InvalidEpoch {
            epoch_j2000_s_bits: epoch_j2000_s.to_bits(),
        });
    }
    let floor_second = epoch_j2000_s.floor();
    if floor_second < i64::MIN as f64 || floor_second > i64::MAX as f64 {
        return Err(SpaceWeatherError::InvalidEpoch {
            epoch_j2000_s_bits: epoch_j2000_s.to_bits(),
        });
    }
    let from_midnight = floor_second as i64 + J2000_NOON_OFFSET_S;
    let day_index = from_midnight.div_euclid(SECONDS_PER_DAY_I64);
    let jdn = day_index + J2000_JULIAN_DAY_NUMBER;
    let min_jdn = julian_day_number(0, 1, 1);
    let max_jdn = julian_day_number(9999, 12, 31);
    if !(min_jdn..=max_jdn).contains(&jdn) {
        return Err(SpaceWeatherError::InvalidEpoch {
            epoch_j2000_s_bits: epoch_j2000_s.to_bits(),
        });
    }
    Ok(jdn)
}

fn epoch_day_and_ap_bin(epoch_j2000_s: f64) -> Result<(i64, u8), SpaceWeatherError> {
    let jdn = epoch_day_jdn(epoch_j2000_s)?;
    let floor_second = epoch_j2000_s.floor() as i64;
    let from_midnight = floor_second + J2000_NOON_OFFSET_S;
    let second_of_day = from_midnight.rem_euclid(SECONDS_PER_DAY_I64);
    Ok((jdn, (second_of_day / (3 * 3600)) as u8))
}

fn day_start_j2000_s(jdn: i64) -> f64 {
    let (year, month, day) = civil_from_julian_day_number(jdn);
    j2000_seconds(year as i32, month as i32, day as i32, 0, 0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CSV: &str = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81\n\
2024-05-09,2556,1,23,27,30,33,40,50,47,37,287,9,12,15,18,27,48,39,22,24,1.2,5,120,165.1,162.0,OBS,150.1,149.8,147.0,146.6\n\
2024-05-10,2556,2,40,50,60,70,67,57,47,37,428,27,48,80,132,111,67,39,22,66,1.8,7,121,190.2,187.1,OBS,151.2,150.9,148.0,147.6\n\
2024-05-11,2556,3,33,30,27,23,20,17,13,10,173,18,15,12,9,7,6,5,4,10,0.8,3,119,176.3,173.0,OBS,152.3,151.1,149.0,148.2\n\
2024-06-01,2557,24,,,,,,,,,,,,,,,,,,,,,118,171.0,168.0,PRM,153.0,152.0,150.0,149.0\n";

    #[test]
    fn parses_csv_and_serves_drag_weather() {
        let parsed = parse_csv(CSV).expect("csv parses");
        assert!(parsed.diagnostics.is_empty());
        let table = parsed.value;
        assert_eq!(table.days().len(), 3);
        assert_eq!(table.monthly().len(), 1);
        assert_eq!(
            table.day(2024, 6, 15).unwrap().class,
            ObservationClass::MonthlyPredicted
        );

        let epoch = j2000_seconds(2024, 5, 10, 12, 0, 0.0);
        let sample = table.sample_at(epoch).expect("sample");
        assert_eq!(
            sample.space_weather,
            SpaceWeather {
                f107: 165.1,
                f107a: 151.2,
                ap: 66.0,
            }
        );
        assert_eq!(sample.class, ObservationClass::Observed);
        assert!(!sample.ap_defaulted);
    }

    #[test]
    fn monthly_region_defaults_ap_and_can_be_rejected() {
        let table = parse_csv(CSV).unwrap().value;
        let epoch = j2000_seconds(2024, 6, 15, 0, 0, 0.0);
        let sample = table.sample_at(epoch).expect("monthly sample");
        assert_eq!(sample.space_weather.f107, 171.0);
        assert_eq!(sample.space_weather.f107a, 153.0);
        assert_eq!(sample.space_weather.ap, DEFAULT_AP);
        assert!(sample.ap_defaulted);

        let policy = SpaceWeatherPolicy {
            require_geomagnetic: true,
            ..SpaceWeatherPolicy::default()
        };
        assert!(matches!(
            table.sample_at_with_policy(epoch, policy),
            Err(SpaceWeatherError::RejectedByPolicy {
                class: ObservationClass::MonthlyPredicted,
                ..
            })
        ));
    }

    #[test]
    fn ap_array_crosses_day_boundaries() {
        let table = parse_csv(CSV).unwrap().value;
        let epoch = j2000_seconds(2024, 5, 11, 13, 0, 0.0);
        let ap = table.ap_array_at(epoch).expect("ap array");
        assert_eq!(ap[0], 10.0);
        assert_eq!(ap[1], 7.0);
        assert_eq!(ap[2], 9.0);
        assert_eq!(ap[3], 12.0);
        assert_eq!(ap[4], 15.0);
        assert_eq!(
            ap[5],
            (18.0 + 22.0 + 39.0 + 67.0 + 111.0 + 132.0 + 80.0 + 48.0) / 8.0
        );
        assert_eq!(
            ap[6],
            (12.0 + 15.0 + 18.0 + 27.0 + 48.0 + 39.0 + 22.0 + 27.0) / 8.0
        );
    }

    #[test]
    fn parse_sniffs_utf8_and_format() {
        assert!(matches!(parse(b"\xff"), Err(SpaceWeatherError::NotText)));
        assert!(matches!(
            parse(b"not cssi"),
            Err(SpaceWeatherError::UnrecognizedFormat)
        ));
        assert!(matches!(
            parse(b"not a header\nCssiSpaceWeather"),
            Err(SpaceWeatherError::UnrecognizedFormat)
        ));
        assert_eq!(parse(CSV.as_bytes()).unwrap().value.days().len(), 3);
    }

    #[test]
    fn parser_diagnostics_cover_bad_duplicate_and_ordered_rows() {
        let truncated = format!("{CSV}2024-05-12,bad\n");
        let parsed = parse_csv(&truncated).expect("forgiving parse");
        assert_eq!(parsed.value.days().len(), 3);
        assert!(matches!(
            parsed.diagnostics.skips[0].reason,
            SkipReason::Truncated
        ));

        let bad_kp = format!(
            "{CSV}{}",
            "2024-05-12,2556,4,bad,27,30,33,40,50,47,37,287,9,12,15,18,27,48,39,22,24,1.2,5,120,165.1,162.0,OBS,150.1,149.8,147.0,146.6\n"
        );
        let parsed = parse_csv(&bad_kp).expect("forgiving parse");
        assert_eq!(parsed.value.days().len(), 3);
        assert!(matches!(
            parsed.diagnostics.skips[0].reason,
            SkipReason::MalformedField(FieldError::IntParse { field: "KP", .. })
        ));

        let duplicate = format!(
            "{CSV_HEADER}\n{}{}",
            "2024-05-09,2556,1,23,27,30,33,40,50,47,37,287,9,12,15,18,27,48,39,22,24,1.2,5,120,165.1,162.0,OBS,150.1,149.8,147.0,146.6\n",
            "2024-05-09,2556,1,23,27,30,33,40,50,47,37,287,9,12,15,18,27,48,39,22,24,1.2,5,120,165.1,162.0,OBS,150.1,149.8,147.0,146.6\n"
        );
        let parsed = parse_csv(&duplicate).expect("forgiving duplicate");
        assert_eq!(parsed.value.days().len(), 1);
        assert!(matches!(
            parsed.diagnostics.skips[0].reason,
            SkipReason::InconsistentRecord("duplicate date")
        ));

        let out_of_order = format!(
            "{CSV_HEADER}\n{}{}",
            "2024-05-10,2556,2,40,50,60,70,67,57,47,37,428,27,48,80,132,111,67,39,22,66,1.8,7,121,190.2,187.1,OBS,151.2,150.9,148.0,147.6\n",
            "2024-05-09,2556,1,23,27,30,33,40,50,47,37,287,9,12,15,18,27,48,39,22,24,1.2,5,120,165.1,162.0,OBS,150.1,149.8,147.0,146.6\n"
        );
        let parsed = parse_csv(&out_of_order).expect("forgiving order check");
        assert_eq!(parsed.value.days().len(), 1);
        assert!(matches!(
            parsed.diagnostics.skips[0].reason,
            SkipReason::InconsistentRecord("out-of-order date")
        ));
    }

    #[test]
    fn txt_count_mismatch_warns_without_rejecting_observed_only() {
        let text = format!(
            "DATATYPE CssiSpaceWeather\nVERSION 1.2\nNUM_OBSERVED_POINTS 2\nBEGIN OBSERVED\n{}\nEND OBSERVED\n",
            txt_row_observed()
        );
        let parsed = parse_txt(&text).expect("txt parses with warning");
        assert_eq!(parsed.value.days().len(), 1);
        assert_eq!(parsed.value.monthly().len(), 0);
        assert_eq!(parsed.diagnostics.warnings.len(), 1);
        assert_eq!(parsed.value.coverage().last_daily_predicted_j2000_s, None);
    }

    #[test]
    fn parses_fixed_width_sections() {
        let text = format!(
            "DATATYPE CssiSpaceWeather\nVERSION 1.2\nNUM_OBSERVED_POINTS 1\nBEGIN OBSERVED\n{}\nEND OBSERVED\n",
            txt_row_observed()
        );
        let parsed = parse_txt(&text).expect("txt parses");
        assert!(parsed.diagnostics.is_empty());
        let row = parsed.value.day(2024, 5, 9).unwrap();
        assert_eq!(row.class, ObservationClass::Observed);
        assert_eq!(row.flux_qualifier, Some(0));
        assert_eq!(row.f107_obs, Some(165.1));
        assert_eq!(row.f107_obs_center81, Some(150.1));
        assert_eq!(row.f107_adj, Some(162.0));
        assert_eq!(row.f107_adj_center81, Some(147.0));
    }

    #[test]
    fn observed_only_file_has_exclusive_end_no_holdover() {
        let input = format!(
            "{CSV_HEADER}\n{}{}",
            "2024-05-09,2556,1,23,27,30,33,40,50,47,37,287,9,12,15,18,27,48,39,22,24,1.2,5,120,165.1,162.0,OBS,150.1,149.8,147.0,146.6\n",
            "2024-05-10,2556,2,40,50,60,70,67,57,47,37,428,27,48,80,132,111,67,39,22,66,1.8,7,121,190.2,187.1,OBS,151.2,150.9,148.0,147.6\n"
        );
        let table = parse_csv(&input).unwrap().value;
        assert_eq!(table.coverage().last_daily_predicted_j2000_s, None);
        assert!(matches!(
            table.sample_at(j2000_seconds(2024, 5, 11, 0, 0, 0.0)),
            Err(SpaceWeatherError::AfterCoverage { .. })
        ));
    }

    #[test]
    fn gap_days_and_boundaries_report_typed_errors() {
        let input = format!(
            "{CSV_HEADER}\n{}{}",
            "2024-05-09,2556,1,23,27,30,33,40,50,47,37,287,9,12,15,18,27,48,39,22,24,1.2,5,120,165.1,162.0,OBS,150.1,149.8,147.0,146.6\n",
            "2024-05-11,2556,3,33,30,27,23,20,17,13,10,173,18,15,12,9,7,6,5,4,10,0.8,3,119,176.3,173.0,OBS,152.3,151.1,149.0,148.2\n"
        );
        let table = parse_csv(&input).unwrap().value;
        assert!(matches!(
            table.sample_at(j2000_seconds(2024, 5, 10, 12, 0, 0.0)),
            Err(SpaceWeatherError::MissingData {
                year: 2024,
                month: 5,
                day: 10,
                field: "record"
            })
        ));
        assert!(matches!(
            table.sample_at(j2000_seconds(2024, 5, 11, 12, 0, 0.0)),
            Err(SpaceWeatherError::MissingData {
                year: 2024,
                month: 5,
                day: 10,
                field: "record"
            })
        ));

        let table = parse_csv(CSV).unwrap().value;
        let boundary = table
            .sample_at(j2000_seconds(2024, 5, 10, 0, 0, 0.0))
            .expect("day boundary belongs to starting day");
        assert_eq!(boundary.space_weather.f107, 165.1);
        assert_eq!(boundary.space_weather.ap, 66.0);
        assert!(matches!(
            table.sample_at(j2000_seconds(2024, 5, 9, 0, 0, 0.0)),
            Err(SpaceWeatherError::BeforeCoverage { .. })
        ));
        assert!(matches!(
            table.sample_at(f64::NAN),
            Err(SpaceWeatherError::InvalidEpoch { .. })
        ));
    }

    #[test]
    fn policy_rejects_each_non_observed_class_and_geomagnetic_default() {
        let input = format!(
            "{CSV_HEADER}\n{}{}{}{}",
            "2024-05-09,2556,1,23,27,30,33,40,50,47,37,287,9,12,15,18,27,48,39,22,24,1.2,5,120,165.1,162.0,OBS,150.1,149.8,147.0,146.6\n",
            "2024-05-10,2556,2,40,50,60,70,67,57,47,37,428,27,48,80,132,111,67,39,22,66,1.8,7,121,190.2,187.1,INT,151.2,150.9,148.0,147.6\n",
            "2024-05-11,2556,3,33,30,27,23,20,17,13,10,173,18,15,12,9,7,6,5,4,10,0.8,3,119,176.3,173.0,PRD,152.3,151.1,149.0,148.2\n",
            "2024-06-01,2557,24,,,,,,,,,,,,,,,,,,,,,118,171.0,168.0,PRM,153.0,152.0,150.0,149.0\n"
        );
        let table = parse_csv(&input).unwrap().value;
        assert!(matches!(
            table.sample_at_with_policy(
                j2000_seconds(2024, 5, 10, 12, 0, 0.0),
                SpaceWeatherPolicy {
                    allow_interpolated: false,
                    ..SpaceWeatherPolicy::default()
                }
            ),
            Err(SpaceWeatherError::RejectedByPolicy {
                class: ObservationClass::Interpolated,
                ..
            })
        ));
        assert!(matches!(
            table.sample_at_with_policy(
                j2000_seconds(2024, 5, 11, 12, 0, 0.0),
                SpaceWeatherPolicy {
                    allow_daily_predicted: false,
                    ..SpaceWeatherPolicy::default()
                }
            ),
            Err(SpaceWeatherError::RejectedByPolicy {
                class: ObservationClass::DailyPredicted,
                ..
            })
        ));
        assert!(matches!(
            table.sample_at_with_policy(
                j2000_seconds(2024, 6, 15, 12, 0, 0.0),
                SpaceWeatherPolicy {
                    allow_monthly_predicted: false,
                    ..SpaceWeatherPolicy::default()
                }
            ),
            Err(SpaceWeatherError::RejectedByPolicy {
                class: ObservationClass::MonthlyPredicted,
                ..
            })
        ));
        assert!(matches!(
            table.sample_at_with_policy(
                j2000_seconds(2024, 6, 15, 12, 0, 0.0),
                SpaceWeatherPolicy {
                    require_geomagnetic: true,
                    ..SpaceWeatherPolicy::default()
                }
            ),
            Err(SpaceWeatherError::RejectedByPolicy {
                class: ObservationClass::MonthlyPredicted,
                ..
            })
        ));
    }

    #[test]
    fn monthly_prediction_is_piecewise_constant_and_ap_slots_fallback_to_daily() {
        let monthly_input = format!(
            "{CSV}{}",
            "2024-07-01,2558,27,,,,,,,,,,,,,,,,,,,,,116,160.0,157.0,PRM,140.0,151.0,147.0,148.0\n"
        );
        let table = parse_csv(&monthly_input).unwrap().value;
        let june_15 = table
            .sample_at(j2000_seconds(2024, 6, 15, 12, 0, 0.0))
            .expect("June monthly sample");
        let june_20 = table
            .sample_at(j2000_seconds(2024, 6, 20, 12, 0, 0.0))
            .expect("same monthly sample");
        let july_15 = table
            .sample_at(j2000_seconds(2024, 7, 15, 12, 0, 0.0))
            .expect("next monthly sample");
        assert_eq!(june_15, june_20);
        assert_ne!(june_15.space_weather.f107a, july_15.space_weather.f107a);

        let input = format!(
            "{CSV_HEADER}\n{}{}{}",
            "2024-05-09,2556,1,23,27,30,33,40,50,47,37,287,9,12,15,18,27,48,39,22,24,1.2,5,120,165.1,162.0,OBS,150.1,149.8,147.0,146.6\n",
            "2024-05-10,2556,2,40,50,60,70,67,57,47,37,428,27,48,80,132,111,67,39,22,66,1.8,7,121,190.2,187.1,OBS,151.2,150.9,148.0,147.6\n",
            "2024-05-11,2556,3,33,30,27,23,20,17,13,10,173,18,15,12,9,,6,5,4,10,0.8,3,119,176.3,173.0,OBS,152.3,151.1,149.0,148.2\n"
        );
        let table = parse_csv(&input).unwrap().value;
        let ap = table
            .ap_array_at(j2000_seconds(2024, 5, 11, 13, 0, 0.0))
            .expect("AP fallback");
        assert_eq!(ap[1], 10.0);
    }

    fn txt_row_observed() -> String {
        let kp = [23, 27, 30, 33, 40, 50, 47, 37];
        let ap = [9, 12, 15, 18, 27, 48, 39, 22];
        let mut row = format!("{:4}{:3}{:3}{:5}{:3}", 2024, 5, 9, 2556, 1);
        for value in kp {
            row.push_str(&format!("{value:3}"));
        }
        row.push_str(&format!("{:4}", 287));
        for value in ap {
            row.push_str(&format!("{value:4}"));
        }
        row.push_str(&format!(
            "{:4}{:4.1}{:2}{:4}{:6.1}{:2}{:6.1}{:6.1}{:6.1}{:6.1}{:6.1}",
            24, 1.2, 5, 120, 162.0, 0, 147.0, 146.6, 165.1, 150.1, 149.8
        ));
        row
    }
}
