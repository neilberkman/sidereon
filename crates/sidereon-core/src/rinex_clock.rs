//! RINEX clock (`.CLK`) satellite-clock parser and interpolation.
//!
//! The parser owns the product grammar for `AS` satellite clock-bias records.
//! The strict parser reports malformed `AS` rows. Use
//! [`RinexClock::parse_lossy`] only when best-effort input recovery is intended.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt::{self, Write as _};

use crate::astro::constants::time::SECONDS_PER_DAY_I64;
use crate::astro::math::interp::lerp_ratio;
use crate::astro::time::civil::{
    civil_from_julian_day_number, j2000_seconds_from_split, seconds_between_splits,
    J2000_JULIAN_DAY_NUMBER, J2000_NOON_OFFSET_S,
};
use crate::astro::time::model::{Instant, InstantRepr, JulianDateSplit, TimeScale};
use crate::astro::time::scales::julian_day_number;
use crate::constants::{
    GPS_EPOCH_TO_J2000_S, J2000_JD, MICROSECONDS_PER_SECOND, SECONDS_PER_DAY, SECONDS_PER_HOUR,
};
use crate::format::columns::fixed_record;
use crate::validate::{self, FieldError};

const INSTANT_SCALE_ORDER_STRIDE_S: f64 = 1.0e15;

/// One satellite clock-bias sample.
#[derive(Debug, Clone, PartialEq)]
pub struct ClockPoint {
    /// Scale-tagged epoch from the RINEX clock file's declared time system.
    pub epoch: Instant,
    /// Satellite clock bias in seconds.
    pub bias_s: f64,
    /// Additional numeric clock values following the bias, in standard order:
    /// bias sigma (s), clock rate (dimensionless), clock rate sigma (dimensionless),
    /// clock acceleration (s^-1), and clock acceleration sigma (s^-1).
    pub additional_values: Vec<f64>,
}

impl ClockPoint {
    /// This sample's epoch as GPS seconds, when the sample is actually GPST.
    pub fn gps_seconds(&self) -> Option<f64> {
        instant_to_gps_seconds(&self.epoch)
    }

    /// Validate that this clock point has a valid epoch instant, finite bias,
    /// at most 5 additional values, and all additional values finite.
    pub fn validate(&self) -> Result<(), RinexClockError> {
        validate_clock_point(self)
    }
}

/// Civil epoch tag used by RINEX clock records, interpreted in the file's time scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockEpoch {
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
    /// Seconds of minute, including fractional seconds.
    pub second: f64,
}

/// An unmodelled record skipped during RINEX clock parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RinexClockSkip {
    /// One-based input line number where the record appeared.
    pub line: usize,
    /// Two-letter RINEX clock record type identifier (e.g. `"AR"`, `"CR"`, `"DR"`, `"MS"`).
    pub record_type: String,
}

impl RinexClockSkip {
    /// Create a new skipped record report entry.
    pub fn new(line: usize, record_type: impl Into<String>) -> Self {
        Self {
            line,
            record_type: record_type.into(),
        }
    }
}

impl fmt::Display for RinexClockSkip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "skipped unmodelled {} record at line {}",
            self.record_type, self.line
        )
    }
}

/// A diagnostic recorded when lossy parsing skips a malformed record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RinexClockDiagnostic {
    /// One-based line number of the malformed or invalid record.
    pub line: usize,
    /// The underlying parse error that caused the record to be skipped.
    pub error: RinexClockError,
}

impl RinexClockDiagnostic {
    /// Create a new parse diagnostic entry.
    pub fn new(line: usize, error: RinexClockError) -> Self {
        Self { line, error }
    }
}

impl fmt::Display for RinexClockDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse diagnostic at line {}: {}", self.line, self.error)
    }
}

/// Parsed RINEX clock product.
#[derive(Debug, Clone, PartialEq)]
pub struct RinexClock {
    /// Time scale declared by the RINEX clock header. Missing headers default to GPST.
    pub time_scale: TimeScale,
    /// Per-satellite, strictly time-ordered clock-bias series.
    pub series: BTreeMap<String, Vec<ClockPoint>>,
    /// Unmodelled records skipped during parsing (e.g. receiver clock records `AR`, `CR`, `DR`, `MS`).
    pub skipped_records: Vec<RinexClockSkip>,
    /// Diagnostics for malformed records skipped during lossy parsing.
    pub diagnostics: Vec<RinexClockDiagnostic>,
}

/// RINEX clock parse error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RinexClockError {
    /// An `AS` satellite clock row is too short to carry the required bias.
    MalformedAsRecord {
        /// One-based input line number.
        line: usize,
        /// Human-readable parse failure.
        reason: &'static str,
        /// The full record text.
        record: String,
    },
    /// A declared continuation record was missing.
    MissingContinuation {
        /// One-based input line number of the parent record.
        line: usize,
        /// Record type of the parent record.
        record_type: String,
    },
    /// A continuation record was malformed or truncated.
    MalformedContinuation {
        /// One-based input line number of the continuation record.
        line: usize,
        /// Human-readable parse failure.
        reason: &'static str,
        /// The full record text.
        record: String,
    },
    /// A required `AS` field could not be parsed or was out of range.
    BadField {
        /// One-based input line number.
        line: usize,
        /// Field name.
        field: &'static str,
        /// Source field value.
        value: String,
    },
    /// Public manual input or query parameter was invalid.
    InvalidInput {
        /// Field name.
        field: &'static str,
        /// Human-readable validation failure.
        reason: &'static str,
    },
    /// The clock product names a time scale RINEX clock headers cannot represent.
    UnsupportedTimeScale {
        /// Unsupported time scale.
        scale: TimeScale,
    },
}

impl fmt::Display for RinexClockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RinexClockError::MalformedAsRecord {
                line,
                reason,
                record,
            } => write!(
                f,
                "malformed RINEX AS clock record at line {line}: {reason}: {record}"
            ),
            RinexClockError::MissingContinuation { line, record_type } => write!(
                f,
                "missing required continuation record for {record_type} at line {line}"
            ),
            RinexClockError::MalformedContinuation {
                line,
                reason,
                record,
            } => write!(
                f,
                "malformed RINEX clock continuation record at line {line}: {reason}: {record}"
            ),
            RinexClockError::BadField { line, field, value } => write!(
                f,
                "bad RINEX AS clock field at line {line}: {field}={value}"
            ),
            RinexClockError::InvalidInput { field, reason } => {
                write!(f, "invalid RINEX clock input {field}: {reason}")
            }
            RinexClockError::UnsupportedTimeScale { scale } => {
                write!(f, "unsupported RINEX clock time scale {}", scale.abbrev())
            }
        }
    }
}

impl std::error::Error for RinexClockError {}

impl RinexClock {
    /// Parse a RINEX clock text into per-satellite `AS` records.
    pub fn parse(text: &str) -> Result<Self, RinexClockError> {
        let time_scale = parse_time_scale(text)?;
        let lines = data_lines(text);
        let mut by_sat = BTreeMap::<String, Vec<(ClockPoint, usize)>>::new();
        let mut skipped_records = Vec::new();
        let mut diagnostics = Vec::new();

        parse_logical_records(
            lines,
            time_scale,
            false,
            &mut by_sat,
            &mut skipped_records,
            &mut diagnostics,
        )?;

        Ok(Self {
            time_scale,
            series: build_series(by_sat),
            skipped_records,
            diagnostics: Vec::new(),
        })
    }

    /// Parse a RINEX clock text while skipping malformed and non-`AS` records.
    pub fn parse_lossy(text: &str) -> Self {
        let time_scale = parse_time_scale(text).unwrap_or(TimeScale::Gpst);
        let lines = data_lines(text);
        let mut by_sat = BTreeMap::<String, Vec<(ClockPoint, usize)>>::new();
        let mut skipped_records = Vec::new();
        let mut diagnostics = Vec::new();

        let _ = parse_logical_records(
            lines,
            time_scale,
            true,
            &mut by_sat,
            &mut skipped_records,
            &mut diagnostics,
        );

        Self {
            time_scale,
            series: build_series(by_sat),
            skipped_records,
            diagnostics,
        }
    }

    /// Rebuild a GPST product from the legacy public GPS-second rows.
    pub fn from_series_rows(rows: Vec<(String, Vec<(f64, f64)>)>) -> Result<Self, RinexClockError> {
        let rows = rows
            .into_iter()
            .map(|(sat, points)| {
                validate::require_strictly_increasing(
                    points.iter().map(|&(gps_seconds, _)| gps_seconds),
                    "gps_seconds",
                )
                .map_err(map_manual_order_error)?;
                let points = points
                    .into_iter()
                    .map(|(gps_seconds, bias_s)| {
                        validate_finite(bias_s, "bias_s")?;
                        Ok((gps_seconds_to_instant(gps_seconds), bias_s))
                    })
                    .collect::<Result<Vec<_>, RinexClockError>>()?;
                Ok((sat, points))
            })
            .collect::<Result<Vec<_>, RinexClockError>>()?;
        Self::from_instant_series_rows(TimeScale::Gpst, rows)
    }

    /// Rebuild a parsed product from scale-tagged instant rows.
    pub fn from_instant_series_rows(
        time_scale: TimeScale,
        rows: Vec<(String, Vec<(Instant, f64)>)>,
    ) -> Result<Self, RinexClockError> {
        let mut series = BTreeMap::new();
        for (sat, points) in rows {
            let mut indexed = points
                .into_iter()
                .enumerate()
                .map(|(idx, (epoch, bias_s))| {
                    let point = ClockPoint {
                        epoch,
                        bias_s,
                        additional_values: Vec::new(),
                    };
                    validate_clock_point(&point)?;
                    Ok((point, idx))
                })
                .collect::<Result<Vec<_>, RinexClockError>>()?;
            validate_instant_series_order(&indexed)?;
            indexed.sort_by(|(a, ai), (b, bi)| {
                compare_instants(&a.epoch, &b.epoch).then_with(|| ai.cmp(bi))
            });
            series.insert(sat, dedup_by_time(indexed));
        }
        Ok(Self {
            time_scale,
            series,
            skipped_records: Vec::new(),
            diagnostics: Vec::new(),
        })
    }

    /// Export GPST samples as `[(satellite, [(gps_seconds, bias_s), ...]), ...]`.
    ///
    /// Non-GPST samples are not coerced into GPS seconds and are omitted.
    pub fn series_rows(&self) -> Vec<(String, Vec<(f64, f64)>)> {
        self.series
            .iter()
            .map(|(sat, points)| {
                (
                    sat.clone(),
                    points
                        .iter()
                        .filter_map(|point| Some((point.gps_seconds()?, point.bias_s)))
                        .collect(),
                )
            })
            .collect()
    }

    /// Export the product as scale-tagged instant rows.
    pub fn instant_series_rows(&self) -> Vec<(String, Vec<(Instant, f64)>)> {
        self.series
            .iter()
            .map(|(sat, points)| {
                (
                    sat.clone(),
                    points
                        .iter()
                        .map(|point| (point.epoch, point.bias_s))
                        .collect(),
                )
            })
            .collect()
    }

    /// Interpolate one satellite clock bias at a civil epoch in this file's scale.
    pub fn clock_s(
        &self,
        satellite_id: &str,
        epoch: ClockEpoch,
    ) -> Result<Option<f64>, RinexClockError> {
        let epoch = civil_to_clock_instant(
            self.time_scale,
            epoch.year,
            epoch.month,
            epoch.day,
            epoch.hour,
            epoch.minute,
            epoch.second,
        )
        .ok_or_else(|| invalid_input("epoch", "invalid civil clock epoch"))?;
        self.clock_s_at_instant(satellite_id, epoch)
    }

    /// Interpolate one satellite clock bias at a scale-tagged instant.
    pub fn clock_s_at_instant(
        &self,
        satellite_id: &str,
        epoch: Instant,
    ) -> Result<Option<f64>, RinexClockError> {
        validate_instant(epoch, "epoch")?;
        let Some(records) = self.series.get(satellite_id) else {
            return Ok(None);
        };
        Ok(interpolate(records, epoch))
    }

    /// Interpolate one satellite clock bias at GPS seconds.
    pub fn clock_s_at_gps_seconds(
        &self,
        satellite_id: &str,
        gps_seconds: f64,
    ) -> Result<Option<f64>, RinexClockError> {
        validate_finite(gps_seconds, "gps_seconds")?;
        self.clock_s_at_instant(satellite_id, gps_seconds_to_instant(gps_seconds))
    }

    /// Serialize this product to standard RINEX 3.00 clock text.
    ///
    /// Pure and deterministic: the same product always produces byte-identical
    /// text and no I/O is performed. The minimal header declares the product
    /// time scale, and each sample is written as a fixed-column `AS` satellite
    /// clock-bias record (with continuation records when more than two values are
    /// present). Epoch components are written on the civil microsecond grid.
    /// Numeric values are formatted into fixed 19-column scientific fields
    /// (`E19.12`) guaranteeing exact bit readback upon re-parsing; values that
    /// cannot fit within the field width or cannot be represented without loss
    /// of precision are refused with a named [`RinexClockError::InvalidInput`]
    /// error. Unsupported epoch time scales return
    /// [`RinexClockError::UnsupportedTimeScale`].
    pub fn to_rinex_string(&self) -> Result<String, RinexClockError> {
        let mut out = String::new();
        let label = crate::rinex_common::time_scale_rinex_label(self.time_scale).ok_or(
            RinexClockError::UnsupportedTimeScale {
                scale: self.time_scale,
            },
        )?;
        let _ = writeln!(out, "{:<60}RINEX VERSION / TYPE", "     3.00           C");
        let _ = writeln!(out, "{label:<60}TIME SYSTEM ID");
        let _ = writeln!(out, "{:<60}END OF HEADER", "");
        for (satellite, points) in &self.series {
            for point in points {
                validate_serializable_clock_point(self.time_scale, point)?;
                write_as_record(&mut out, satellite, point)?;
            }
        }
        Ok(out)
    }
}

/// Append one `AS` satellite clock-bias record for a sample.
fn write_as_record(
    out: &mut String,
    satellite: &str,
    point: &ClockPoint,
) -> Result<(), RinexClockError> {
    let (year, month, day, hour, minute, second_us) = instant_civil_microsecond(&point.epoch);
    let second = second_us / 1_000_000;
    let microsecond = second_us % 1_000_000;
    let count = 1 + point.additional_values.len();
    if count > 6 {
        return Err(invalid_input(
            "additional_values",
            "at most 5 additional values are supported",
        ));
    }

    let bias_str = format_e19_12(point.bias_s, "bias_s")?;

    if count == 1 {
        let _ = writeln!(
            out,
            "AS {satellite:<4} {year:04} {month:02} {day:02} {hour:02} {minute:02} {second:>2}.{microsecond:06}  1   {bias_str}",
        );
    } else {
        let sigma_str = format_e19_12(point.additional_values[0], "additional_values")?;
        let _ = writeln!(
            out,
            "AS {satellite:<4} {year:04} {month:02} {day:02} {hour:02} {minute:02} {second:>2}.{microsecond:06}  {count}   {bias_str} {sigma_str}",
        );
    }

    if count > 2 {
        let mut cont = String::with_capacity(80);
        for (i, &val) in point.additional_values[1..].iter().enumerate() {
            let val_str = format_e19_12(val, "additional_values")?;
            if i > 0 {
                cont.push(' ');
            }
            cont.push_str(&val_str);
        }
        let _ = writeln!(out, "{cont}");
    }

    Ok(())
}

/// Decompose a clock-sample instant into civil `(year, month, day, hour, minute,
/// total-microseconds-of-minute)` on the microsecond grid the parser reads.
///
/// This inverts [`civil_microsecond_to_julian_split`]: the standard epoch grid
/// from its split Julian date, a UTC `:60` leap-second epoch from its stored
/// sub-midnight fraction, and a nanosecond-repr instant from its J2000 offset.
fn instant_civil_microsecond(epoch: &Instant) -> (i64, i64, i64, i64, i64, i64) {
    let (day_number, total_us) = match epoch.repr {
        InstantRepr::JulianDate(split) => {
            // A UTC leap-second epoch is stored by the parser as `remaining_s`
            // seconds before the next day's midnight (see
            // civil_microsecond_to_julian_split): a small negative fraction on the
            // next day's whole JD. Rebuild the `23:59:60.xxxxxx` label on the
            // previous civil day so it round-trips, rather than emitting a wrong
            // time from a negative time-of-day.
            if (-1.0 / SECONDS_PER_DAY..0.0).contains(&split.fraction) {
                return leap_second_civil(split);
            }
            // The parser stores `jd_whole = JDN - 0.5` (civil-day midnight
            // boundary) and carries the time-of-day as `fraction`. Read the day
            // number and the time-of-day from each part separately: recombining
            // into a single JD and subtracting the seven-digit day number would
            // lose microsecond precision to catastrophic cancellation.
            let day_number = (split.jd_whole + 0.5).round() as i64;
            let total_us =
                (split.fraction * SECONDS_PER_DAY * MICROSECONDS_PER_SECOND).round() as i64;
            (day_number, total_us)
        }
        // Nanoseconds count from J2000 (2000-01-01 12:00:00) in the instant's own
        // scale, matching the IONEX/SP3 convention. Convert the actual epoch
        // rather than fabricating J2000.
        InstantRepr::Nanos(nanos) => nanos_civil_day_microsecond(nanos),
    };
    let (year, month, day) = civil_from_julian_day_number(day_number);
    let hour = total_us / 3_600_000_000;
    let rem = total_us % 3_600_000_000;
    let minute = rem / 60_000_000;
    let second_us = rem % 60_000_000;
    (year, month, day, hour, minute, second_us)
}

/// Civil decomposition of a UTC leap-second instant whose `fraction` lies in
/// `[-1/86400, 0)` on the next day's whole JD. The instant sits `remaining_s`
/// seconds before the next day's midnight - inside the `23:59:60` leap second of
/// the previous civil day - so rebuild that label on the microsecond grid.
fn leap_second_civil(split: JulianDateSplit) -> (i64, i64, i64, i64, i64, i64) {
    let next_day_number = (split.jd_whole + 0.5).round() as i64;
    let (year, month, day) = civil_from_julian_day_number(next_day_number - 1);
    let remaining_s = -split.fraction * SECONDS_PER_DAY; // in (0, 1]
    let microsecond = ((1.0 - remaining_s) * 1_000_000.0).round() as i64;
    // Encode the `:60` second as total microseconds of minute so the shared
    // `write_as_record` split (`second_us / 1_000_000`) yields `second == 60`.
    (year, month, day, 23, 59, 60 * 1_000_000 + microsecond)
}

/// Decompose a J2000-nanosecond instant into the civil-midnight `(day number,
/// microseconds of day)` the shared decomposition consumes. Nanoseconds are
/// rounded to the microsecond grid the RINEX clock epoch field carries.
fn nanos_civil_day_microsecond(nanos: i128) -> (i64, i64) {
    const US_PER_DAY: i128 = SECONDS_PER_DAY_I64 as i128 * 1_000_000;
    // J2000 is noon (12:00:00) of 2000-01-01, whose civil-midnight day number is
    // JD 2_451_545 (jd_whole 2_451_544.5 + 0.5).
    const J2000_NOON_US: i128 = J2000_NOON_OFFSET_S as i128 * 1_000_000;
    const J2000_DAY_NUMBER: i128 = J2000_JULIAN_DAY_NUMBER as i128;
    let micros = (nanos + nanos.signum() * 500) / 1_000; // round to nearest us
    let from_midnight = J2000_NOON_US + micros;
    let day_offset = from_midnight.div_euclid(US_PER_DAY);
    let us_of_day = from_midnight.rem_euclid(US_PER_DAY);
    ((J2000_DAY_NUMBER + day_offset) as i64, us_of_day as i64)
}

/// Convert a civil clock tag in the given scale into a scale-tagged instant.
pub fn civil_to_clock_instant(
    scale: TimeScale,
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
) -> Option<Instant> {
    let civil = validate::civil_datetime_with_fractional_second_policy(
        i64::from(year),
        i64::from(month),
        i64::from(day),
        i64::from(hour),
        i64::from(minute),
        second,
        civil_second_policy_for_time_scale(scale),
    )
    .ok()?;
    civil_microsecond_to_instant(scale, civil).ok()
}

/// Convert a civil GPS-time tag into seconds since 1980-01-06 00:00:00.
pub fn civil_to_gps_seconds(
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
) -> Option<f64> {
    let civil = validate::civil_datetime_with_fractional_second_policy(
        i64::from(year),
        i64::from(month),
        i64::from(day),
        i64::from(hour),
        i64::from(minute),
        second,
        validate::CivilSecondPolicy::Continuous,
    )
    .ok()?;
    gps_seconds_from_civil(civil)
}

fn parse_time_scale(text: &str) -> Result<TimeScale, RinexClockError> {
    let mut time_scale = TimeScale::Gpst;
    for (idx, line) in text.lines().enumerate() {
        if line.contains("END OF HEADER") {
            break;
        }
        if line.contains("TIME SYSTEM ID") {
            let label = line
                .split("TIME SYSTEM ID")
                .next()
                .unwrap_or(line)
                .split_whitespace()
                .next()
                .unwrap_or("");
            if label.is_empty() {
                time_scale = TimeScale::Gpst;
            } else {
                time_scale = crate::rinex_common::time_scale_label(label).ok_or_else(|| {
                    RinexClockError::BadField {
                        line: idx + 1,
                        field: "time_system",
                        value: label.to_string(),
                    }
                })?;
            }
        }
    }
    Ok(time_scale)
}

// invariant: the parser validates GPS seconds before constructing its split JD.
#[allow(clippy::expect_used)]
fn gps_seconds_to_instant(gps_seconds: f64) -> Instant {
    let gps_epoch_jd = J2000_JD - GPS_EPOCH_TO_J2000_S / SECONDS_PER_DAY;
    let days = (gps_seconds / SECONDS_PER_DAY).floor();
    let seconds_of_day = gps_seconds - days * SECONDS_PER_DAY;
    Instant::from_julian_date(
        TimeScale::Gpst,
        JulianDateSplit::new(gps_epoch_jd + days, seconds_of_day / SECONDS_PER_DAY)
            .expect("valid split Julian date"),
    )
}

fn validate_clock_point(point: &ClockPoint) -> Result<(), RinexClockError> {
    validate_instant(point.epoch, "epoch")?;
    validate_finite(point.bias_s, "bias_s")?;
    if point.additional_values.len() > 5 {
        return Err(invalid_input(
            "additional_values",
            "cannot exceed 5 additional values (maximum count is 6)",
        ));
    }
    for (idx, &val) in point.additional_values.iter().enumerate() {
        validate_finite(val, field_name_for_value_index(idx + 1))?;
    }
    Ok(())
}

fn validate_serializable_clock_point(
    product_scale: TimeScale,
    point: &ClockPoint,
) -> Result<(), RinexClockError> {
    validate_clock_point(point)?;
    if crate::rinex_common::time_scale_rinex_label(point.epoch.scale).is_none() {
        return Err(RinexClockError::UnsupportedTimeScale {
            scale: point.epoch.scale,
        });
    }
    if point.epoch.scale != product_scale {
        return Err(invalid_input(
            "epoch",
            "epoch scale does not match clock time scale",
        ));
    }
    format_e19_12(point.bias_s, "bias")?;
    for (idx, &val) in point.additional_values.iter().enumerate() {
        format_e19_12(val, field_name_for_value_index(idx + 1))?;
    }
    Ok(())
}

fn field_name_for_value_index(idx: usize) -> &'static str {
    match idx {
        0 => "bias",
        1 => "sigma",
        2 => "rate",
        3 => "rate_sigma",
        4 => "acceleration",
        5 => "acceleration_sigma",
        _ => "additional_values",
    }
}

fn format_e19_12(value: f64, field: &'static str) -> Result<String, RinexClockError> {
    if !value.is_finite() {
        return Err(invalid_input(field, "must be finite"));
    }
    if value == 0.0 {
        let sign = if value.is_sign_negative() { '-' } else { ' ' };
        return Ok(format!("{sign}0.000000000000E+00"));
    }

    let sign = if value.is_sign_negative() { '-' } else { ' ' };
    let abs_val = value.abs();

    if let Some(formatted) = try_format_leading_zero(sign, abs_val) {
        if formatted.len() == 19 {
            if let Ok(reparsed) = formatted.trim().parse::<f64>() {
                if reparsed.to_bits() == value.to_bits() {
                    return Ok(formatted);
                }
            }
        }
    }

    if let Some(formatted) = try_format_nonzero_leading(sign, abs_val) {
        if formatted.len() == 19 {
            if let Ok(reparsed) = formatted.trim().parse::<f64>() {
                if reparsed.to_bits() == value.to_bits() {
                    return Ok(formatted);
                }
            }
        }
    }

    if let Some(formatted) = try_format_3digit_exp(sign, abs_val) {
        if formatted.len() == 19 {
            if let Ok(reparsed) = formatted.trim().parse::<f64>() {
                if reparsed.to_bits() == value.to_bits() {
                    return Ok(formatted);
                }
            }
        }
    }

    if let Some(formatted) = try_format_canonical_scientific(sign, abs_val) {
        if formatted.len() == 19 {
            if let Ok(reparsed) = formatted.trim().parse::<f64>() {
                if reparsed.to_bits() == value.to_bits() {
                    return Ok(formatted);
                }
            }
        }
    }

    if let Some(formatted) = try_format_scientific_fallback(value) {
        return Ok(formatted);
    }

    Err(invalid_input(
        field,
        "value cannot be represented in Fortran E19.12 format without loss of precision",
    ))
}

fn try_format_leading_zero(sign: char, abs_val: f64) -> Option<String> {
    let s = format!("{abs_val:.11e}");
    let (mantissa_str, exp_str) = s.split_once('e')?;
    let rust_exp: i32 = exp_str.parse().ok()?;
    let (d0, rest) = mantissa_str.split_once('.')?;
    let new_exp = rust_exp + 1;
    if !(-99..=99).contains(&new_exp) {
        return None;
    }
    let formatted_exp = if new_exp >= 0 {
        format!("E+{new_exp:02}")
    } else {
        format!("E-{:02}", new_exp.abs())
    };
    Some(format!("{sign}0.{d0}{rest}{formatted_exp}"))
}

fn try_format_nonzero_leading(sign: char, abs_val: f64) -> Option<String> {
    let s = format!("{abs_val:.12e}");
    let (mantissa_str, exp_str) = s.split_once('e')?;
    let rust_exp: i32 = exp_str.parse().ok()?;
    if !(-99..=99).contains(&rust_exp) {
        return None;
    }
    let formatted_exp = if rust_exp >= 0 {
        format!("E+{rust_exp:02}")
    } else {
        format!("E-{:02}", rust_exp.abs())
    };
    Some(format!("{sign}{mantissa_str}{formatted_exp}"))
}

fn try_format_3digit_exp(sign: char, abs_val: f64) -> Option<String> {
    let s = format!("{abs_val:.11e}");
    let (mantissa_str, exp_str) = s.split_once('e')?;
    let rust_exp: i32 = exp_str.parse().ok()?;
    let (d0, rest) = mantissa_str.split_once('.')?;
    let new_exp = rust_exp + 1;
    if !(-999..=-100).contains(&new_exp) && !(100..=999).contains(&new_exp) {
        return None;
    }
    let formatted_exp = if new_exp >= 0 {
        format!("E+{new_exp:03}")
    } else {
        format!("E-{:03}", new_exp.abs())
    };
    Some(format!("{sign}.{d0}{rest}{formatted_exp}"))
}

fn try_format_canonical_scientific(sign: char, abs_val: f64) -> Option<String> {
    let s = format!("{abs_val:.12e}");
    let (mantissa_str, exp_str) = s.split_once('e')?;
    let rust_exp: i32 = exp_str.parse().ok()?;
    let formatted_exp = if (-99..=99).contains(&rust_exp) {
        if rust_exp >= 0 {
            format!("E+{rust_exp:02}")
        } else {
            format!("E-{:02}", rust_exp.abs())
        }
    } else if (-999..=999).contains(&rust_exp) {
        if rust_exp >= 0 {
            format!("E+{rust_exp:03}")
        } else {
            format!("E-{:03}", rust_exp.abs())
        }
    } else {
        return None;
    };

    let raw = if sign == '-' {
        format!("-{mantissa_str}{formatted_exp}")
    } else {
        format!("{mantissa_str}{formatted_exp}")
    };

    if raw.len() > 19 {
        return None;
    }

    Some(format!("{raw:>19}"))
}

/// Formats a finite non-zero floating-point value into an exact 19-column
/// scientific representation when standard preferred formatters cannot fit within 19 bytes.
/// Evaluates finite 1..=17 significant digit candidates strictly containing an
/// explicit decimal point and 'E', testing finite point placement and exponent
/// adjustments alongside optional positive plus signs for input compatibility rather
/// than canonical Fortran output. Candidates of length <= 19 bytes are left-padded with
/// spaces to exactly 19 bytes and accepted only on strict bit readback (`to_bits()`).
///
/// This does not guarantee that all legally representable mathematical values fit
/// within the 19-column budget; values exceeding candidate width limits are refused.
fn try_format_scientific_fallback(value: f64) -> Option<String> {
    let abs_val = value.abs();
    let sign_prefix = if value.is_sign_negative() { "-" } else { "" };

    for sig_digits in (1..=17).rev() {
        let prec = sig_digits - 1;
        let s = format!("{abs_val:.prec$e}");
        let Some((mantissa_part, exp_part)) = s.split_once('e') else {
            continue;
        };
        let Ok(rust_exp) = exp_part.parse::<i32>() else {
            continue;
        };
        let digits: String = mantissa_part
            .chars()
            .filter(|c| c.is_ascii_digit())
            .collect();
        if digits.len() != sig_digits {
            continue;
        }

        let mut mantissa_candidates = Vec::with_capacity(sig_digits + 2);

        // 1. Standard normalized form (decimal point after first digit).
        if sig_digits == 1 {
            mantissa_candidates.push((format!("{sign_prefix}{digits}."), 0));
        } else {
            mantissa_candidates
                .push((format!("{sign_prefix}{}.{}", &digits[..1], &digits[1..]), 0));
        }

        // 2. Leading zero form (0.dddd...).
        mantissa_candidates.push((format!("{sign_prefix}0.{digits}"), 1));

        // 3. Leading dot form (.dddd...).
        mantissa_candidates.push((format!("{sign_prefix}.{digits}"), 1));

        // 4. Shift decimal point to the right across the remaining positions.
        if sig_digits > 1 {
            for k in 2..=sig_digits {
                let exp_delta = -(k as i32 - 1);
                if k < sig_digits {
                    mantissa_candidates.push((
                        format!("{sign_prefix}{}.{}", &digits[..k], &digits[k..]),
                        exp_delta,
                    ));
                } else {
                    mantissa_candidates.push((format!("{sign_prefix}{digits}."), exp_delta));
                }
            }
        }

        for (mantissa, exp_delta) in mantissa_candidates {
            let adj_exp = rust_exp + exp_delta;
            if !(-999..=999).contains(&adj_exp) {
                continue;
            }

            let mut exp_spellings = Vec::with_capacity(4);
            if adj_exp >= 0 {
                if adj_exp <= 99 {
                    exp_spellings.push(format!("E+{adj_exp:02}"));
                    exp_spellings.push(format!("E{adj_exp:02}"));
                    exp_spellings.push(format!("E+{adj_exp:03}"));
                    exp_spellings.push(format!("E{adj_exp:03}"));
                } else {
                    exp_spellings.push(format!("E+{adj_exp:03}"));
                    exp_spellings.push(format!("E{adj_exp:03}"));
                }
            } else {
                let abs_exp = adj_exp.unsigned_abs();
                if abs_exp <= 99 {
                    exp_spellings.push(format!("E-{abs_exp:02}"));
                    exp_spellings.push(format!("E-{abs_exp:03}"));
                } else {
                    exp_spellings.push(format!("E-{abs_exp:03}"));
                }
            }

            for exp_spelling in exp_spellings {
                let candidate_raw = format!("{mantissa}{exp_spelling}");
                if candidate_raw.len() > 19 {
                    continue;
                }
                let candidate = format!("{candidate_raw:>19}");
                if let Ok(reparsed) = validate::strict_f64(&candidate, "bias") {
                    if reparsed.to_bits() == value.to_bits() {
                        return Some(candidate);
                    }
                }
            }
        }
    }

    None
}

fn validate_instant(epoch: Instant, field: &'static str) -> Result<(), RinexClockError> {
    match epoch.repr {
        InstantRepr::JulianDate(split) => {
            validate_finite(split.jd_whole, field)?;
            validate_finite(split.fraction, field)?;
            if !(-1.0..=1.0).contains(&split.fraction) {
                return Err(invalid_input(field, "Julian-date fraction out of range"));
            }
            Ok(())
        }
        InstantRepr::Nanos(_) => Ok(()),
    }
}

fn validate_finite(value: f64, field: &'static str) -> Result<(), RinexClockError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "must be finite"))
    }
}

fn invalid_input(field: &'static str, reason: &'static str) -> RinexClockError {
    RinexClockError::InvalidInput { field, reason }
}

fn map_manual_order_error(error: FieldError) -> RinexClockError {
    match error {
        FieldError::NonFinite { field } => invalid_input(field, "must be finite"),
        FieldError::OutOfRange { field, .. } => invalid_input(field, "must be strictly increasing"),
        _ => invalid_input(error.field(), error.reason()),
    }
}

fn validate_instant_series_order(points: &[(ClockPoint, usize)]) -> Result<(), RinexClockError> {
    validate::require_strictly_increasing(
        points
            .iter()
            .map(|(point, _)| instant_order_key(&point.epoch)),
        "epoch",
    )
    .map_err(map_manual_order_error)
}

fn instant_order_key(epoch: &Instant) -> f64 {
    let offset_s = time_scale_rank(epoch.scale) as f64 * INSTANT_SCALE_ORDER_STRIDE_S;
    let instant_s = match epoch.repr {
        InstantRepr::JulianDate(split) => {
            split.jd_whole * SECONDS_PER_DAY + split.fraction * SECONDS_PER_DAY
        }
        InstantRepr::Nanos(nanos) => nanos as f64 / 1.0e9,
    };
    offset_s + instant_s
}

fn instant_to_gps_seconds(epoch: &Instant) -> Option<f64> {
    if epoch.scale != TimeScale::Gpst {
        return None;
    }
    instant_to_j2000_seconds(epoch).map(|seconds| seconds + GPS_EPOCH_TO_J2000_S)
}

fn instant_to_j2000_seconds(epoch: &Instant) -> Option<f64> {
    match epoch.repr {
        InstantRepr::JulianDate(split) => {
            Some(j2000_seconds_from_split(split.jd_whole, split.fraction))
        }
        InstantRepr::Nanos(_) => None,
    }
}

fn data_lines(text: &str) -> Vec<(usize, &str)> {
    drop_header(
        text.lines()
            .enumerate()
            .map(|(idx, line)| (idx + 1, line))
            .collect(),
    )
}

fn drop_header(lines: Vec<(usize, &str)>) -> Vec<(usize, &str)> {
    match lines
        .iter()
        .position(|(_, line)| line.contains("END OF HEADER"))
    {
        Some(idx) => lines.into_iter().skip(idx + 1).collect(),
        None => lines,
    }
}

#[derive(Debug, Clone, Copy)]
struct ClockEpochFields<'a> {
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: &'a str,
}

const AS_RECORD_COLUMNS: [(usize, usize); 11] = [
    (0, 2),   // Record type "AS" (cols 1..=2, A2)
    (3, 7),   // Satellite "G01 " or "G  1" (cols 4..=7, A4)
    (8, 12),  // Year "2026" (cols 9..=12, I4)
    (12, 15), // Month " 05" (cols 13..=15, I3)
    (15, 18), // Day " 13" (cols 16..=18, I3)
    (18, 21), // Hour " 00" (cols 19..=21, I3)
    (21, 24), // Minute " 00" (cols 22..=24, I3)
    (24, 34), // Second "  0.000000" (cols 25..=34, F10.6)
    (34, 37), // Count "  1" or "  2" (cols 35..=37, I3)
    (40, 59), // Bias (cols 41..=59, E19.12)
    (60, 79), // Sigma (optional, cols 61..=79, E19.12)
];

const AS_RECORD_304_COLUMNS: [(usize, usize); 11] = [
    (0, 2),   // Record type "AS", "AR", etc. (cols 1..=2, A2)
    (3, 12),  // Satellite/receiver "G16       " or "AREQ00USA" (cols 4..=12, A9)
    (13, 17), // Year "1994" (cols 14..=17, I4)
    (18, 20), // Month "07" (cols 19..=20, I2)
    (21, 23), // Day "14" (cols 22..=23, I2)
    (24, 26), // Hour "20" (cols 25..=26, I2)
    (27, 29), // Minute "59" (cols 28..=29, I2)
    (30, 39), // Second " 0.000000" (cols 31..=39, F9.6)
    (40, 42), // Count " 6" (cols 41..=42, I2)
    (45, 64), // Bias (cols 46..=64, E19.12)
    (66, 85), // Sigma (optional, cols 67..=85, E19.12)
];

const CONT_RECORD_300_COLUMNS: [(usize, usize); 4] = [
    (0, 19),  // Value 3 (cols 1..=19, E19.12)
    (20, 39), // Value 4 (cols 21..=39, E19.12)
    (40, 59), // Value 5 (cols 41..=59, E19.12)
    (60, 79), // Value 6 (cols 61..=79, E19.12)
];

const CONT_RECORD_304_COLUMNS: [(usize, usize); 4] = [
    (3, 22),  // Value 3 (cols 4..=22, 3X, E19.12)
    (24, 43), // Value 4 (cols 25..=43, 2X, E19.12)
    (45, 64), // Value 5 (cols 46..=64, 2X, E19.12)
    (66, 85), // Value 6 (cols 67..=85, 2X, E19.12)
];

fn is_known_clock_record_type(token: &str) -> bool {
    matches!(token, "AR" | "CR" | "DR" | "MS")
}

fn is_potential_parent_record(line: &str) -> bool {
    let mut tokens = line.split_whitespace();
    matches!(tokens.next(), Some("AS" | "AR" | "CR" | "DR" | "MS"))
}

enum RawParent {
    Satellite {
        sat: String,
        epoch: Instant,
        bias_s: f64,
        count: usize,
        sigma: Option<f64>,
    },
    Unsupported {
        line: usize,
        record_type: String,
        count: usize,
    },
}

fn parse_raw_parent(
    line_number: usize,
    line: &str,
    time_scale: TimeScale,
) -> Result<RawParent, RinexClockError> {
    if let Some(
        [record_type, sat_field, year_field, month_field, day_field, hour_field, minute_field, second_field, count_field, bias_field, sigma_field],
    ) = fixed_record(line, AS_RECORD_COLUMNS)
    {
        if record_type.len() == 2 && record_type.chars().all(|c| c.is_ascii_alphabetic()) {
            if record_type == "AS" {
                if bias_field.is_empty() {
                    return Err(RinexClockError::MalformedAsRecord {
                        line: line_number,
                        reason: "expected at least 10 fields",
                        record: line.trim().to_string(),
                    });
                }
                let sat = validate::strict_gnss_satellite_id(sat_field, "satellite")
                    .map_err(|error| map_field_error(line_number, error, sat_field))?
                    .to_string();
                let year = parse_int_field::<i32>(line_number, "year", year_field)?;
                let month = parse_int_field::<u8>(line_number, "month", month_field)?;
                let day = parse_int_field::<u8>(line_number, "day", day_field)?;
                let hour = parse_int_field::<u8>(line_number, "hour", hour_field)?;
                let minute = parse_int_field::<u8>(line_number, "minute", minute_field)?;
                let epoch = ClockEpochFields {
                    year,
                    month,
                    day,
                    hour,
                    minute,
                    second: second_field,
                };
                let bias_s = parse_f64_field(line_number, "bias", bias_field)?;
                let epoch = civil_decimal_second_to_instant(time_scale, epoch)
                    .map_err(|error| map_epoch_error(line_number, error, epoch))?;

                let count = parse_int_field::<usize>(line_number, "count", count_field)?;
                if !(1..=6).contains(&count) {
                    return Err(RinexClockError::BadField {
                        line: line_number,
                        field: "count",
                        value: count_field.to_string(),
                    });
                }

                let sigma = if count == 1 {
                    if !sigma_field.is_empty() {
                        return Err(RinexClockError::BadField {
                            line: line_number,
                            field: "sigma",
                            value: sigma_field.to_string(),
                        });
                    }
                    None
                } else {
                    if sigma_field.is_empty() {
                        return Err(RinexClockError::BadField {
                            line: line_number,
                            field: "sigma",
                            value: "".to_string(),
                        });
                    }
                    Some(parse_f64_field(line_number, "sigma", sigma_field)?)
                };

                return Ok(RawParent::Satellite {
                    sat,
                    epoch,
                    bias_s,
                    count,
                    sigma,
                });
            } else if is_known_clock_record_type(record_type) {
                let count = parse_int_field::<usize>(line_number, "count", count_field)?;
                if !(1..=6).contains(&count) {
                    return Err(RinexClockError::BadField {
                        line: line_number,
                        field: "count",
                        value: count_field.to_string(),
                    });
                }
                if count == 1 && !sigma_field.is_empty() {
                    return Err(RinexClockError::BadField {
                        line: line_number,
                        field: "sigma",
                        value: sigma_field.to_string(),
                    });
                }
                if count >= 2 && sigma_field.is_empty() {
                    return Err(RinexClockError::BadField {
                        line: line_number,
                        field: "sigma",
                        value: "".to_string(),
                    });
                }
                return Ok(RawParent::Unsupported {
                    line: line_number,
                    record_type: record_type.to_string(),
                    count,
                });
            } else {
                return Err(RinexClockError::BadField {
                    line: line_number,
                    field: "record_type",
                    value: record_type.to_string(),
                });
            }
        }
    }

    if let Some(
        [record_type, sat_field, year_field, month_field, day_field, hour_field, minute_field, second_field, count_field, bias_field, sigma_field],
    ) = fixed_record(line, AS_RECORD_304_COLUMNS)
    {
        if record_type.len() == 2 && record_type.chars().all(|c| c.is_ascii_alphabetic()) {
            if record_type == "AS" {
                if bias_field.is_empty() {
                    return Err(RinexClockError::MalformedAsRecord {
                        line: line_number,
                        reason: "expected at least 10 fields",
                        record: line.trim().to_string(),
                    });
                }
                let sat = validate::strict_gnss_satellite_id(sat_field, "satellite")
                    .map_err(|error| map_field_error(line_number, error, sat_field))?
                    .to_string();
                let year = parse_int_field::<i32>(line_number, "year", year_field)?;
                let month = parse_int_field::<u8>(line_number, "month", month_field)?;
                let day = parse_int_field::<u8>(line_number, "day", day_field)?;
                let hour = parse_int_field::<u8>(line_number, "hour", hour_field)?;
                let minute = parse_int_field::<u8>(line_number, "minute", minute_field)?;
                let epoch = ClockEpochFields {
                    year,
                    month,
                    day,
                    hour,
                    minute,
                    second: second_field,
                };
                let bias_s = parse_f64_field(line_number, "bias", bias_field)?;
                let epoch = civil_decimal_second_to_instant(time_scale, epoch)
                    .map_err(|error| map_epoch_error(line_number, error, epoch))?;

                let count = parse_int_field::<usize>(line_number, "count", count_field)?;
                if !(1..=6).contains(&count) {
                    return Err(RinexClockError::BadField {
                        line: line_number,
                        field: "count",
                        value: count_field.to_string(),
                    });
                }

                let sigma = if count == 1 {
                    if !sigma_field.is_empty() {
                        return Err(RinexClockError::BadField {
                            line: line_number,
                            field: "sigma",
                            value: sigma_field.to_string(),
                        });
                    }
                    None
                } else {
                    if sigma_field.is_empty() {
                        return Err(RinexClockError::BadField {
                            line: line_number,
                            field: "sigma",
                            value: "".to_string(),
                        });
                    }
                    Some(parse_f64_field(line_number, "sigma", sigma_field)?)
                };

                return Ok(RawParent::Satellite {
                    sat,
                    epoch,
                    bias_s,
                    count,
                    sigma,
                });
            } else if is_known_clock_record_type(record_type) {
                let count = parse_int_field::<usize>(line_number, "count", count_field)?;
                if !(1..=6).contains(&count) {
                    return Err(RinexClockError::BadField {
                        line: line_number,
                        field: "count",
                        value: count_field.to_string(),
                    });
                }
                if count == 1 && !sigma_field.is_empty() {
                    return Err(RinexClockError::BadField {
                        line: line_number,
                        field: "sigma",
                        value: sigma_field.to_string(),
                    });
                }
                if count >= 2 && sigma_field.is_empty() {
                    return Err(RinexClockError::BadField {
                        line: line_number,
                        field: "sigma",
                        value: "".to_string(),
                    });
                }
                return Ok(RawParent::Unsupported {
                    line: line_number,
                    record_type: record_type.to_string(),
                    count,
                });
            } else {
                return Err(RinexClockError::BadField {
                    line: line_number,
                    field: "record_type",
                    value: record_type.to_string(),
                });
            }
        }
    }

    let mut fields = line.split_whitespace();
    let Some(first) = fields.next() else {
        return Err(RinexClockError::BadField {
            line: line_number,
            field: "record_type",
            value: "".to_string(),
        });
    };

    if first == "AS" {
        let sat_field = next_as_field(&mut fields, line_number, line)?;
        let year_field = next_as_field(&mut fields, line_number, line)?;
        let month_field = next_as_field(&mut fields, line_number, line)?;
        let day_field = next_as_field(&mut fields, line_number, line)?;
        let hour_field = next_as_field(&mut fields, line_number, line)?;
        let minute_field = next_as_field(&mut fields, line_number, line)?;
        let second_field = next_as_field(&mut fields, line_number, line)?;
        let count_field = next_as_field(&mut fields, line_number, line)?;
        let bias_field = next_as_field(&mut fields, line_number, line)?;

        let sat = validate::strict_gnss_satellite_id(sat_field, "satellite")
            .map_err(|error| map_field_error(line_number, error, sat_field))?
            .to_string();
        let year = parse_int_field::<i32>(line_number, "year", year_field)?;
        let month = parse_int_field::<u8>(line_number, "month", month_field)?;
        let day = parse_int_field::<u8>(line_number, "day", day_field)?;
        let hour = parse_int_field::<u8>(line_number, "hour", hour_field)?;
        let minute = parse_int_field::<u8>(line_number, "minute", minute_field)?;
        let epoch = ClockEpochFields {
            year,
            month,
            day,
            hour,
            minute,
            second: second_field,
        };
        let bias_s = parse_f64_field(line_number, "bias", bias_field)?;
        let epoch = civil_decimal_second_to_instant(time_scale, epoch)
            .map_err(|error| map_epoch_error(line_number, error, epoch))?;

        let count = parse_int_field::<usize>(line_number, "count", count_field)?;
        if !(1..=6).contains(&count) {
            return Err(RinexClockError::BadField {
                line: line_number,
                field: "count",
                value: count_field.to_string(),
            });
        }

        let sigma = if count == 1 {
            if fields.next().is_some() {
                return Err(RinexClockError::MalformedAsRecord {
                    line: line_number,
                    reason: "excess values in parent record",
                    record: line.trim().to_string(),
                });
            }
            None
        } else {
            let sigma_field = fields.next().ok_or_else(|| RinexClockError::BadField {
                line: line_number,
                field: "sigma",
                value: "".to_string(),
            })?;
            let s = parse_f64_field(line_number, "sigma", sigma_field)?;
            if fields.next().is_some() {
                return Err(RinexClockError::MalformedAsRecord {
                    line: line_number,
                    reason: "excess values in parent record",
                    record: line.trim().to_string(),
                });
            }
            Some(s)
        };

        Ok(RawParent::Satellite {
            sat,
            epoch,
            bias_s,
            count,
            sigma,
        })
    } else if is_known_clock_record_type(first) {
        for _ in 0..7 {
            if fields.next().is_none() {
                return Err(RinexClockError::BadField {
                    line: line_number,
                    field: "count",
                    value: "".to_string(),
                });
            }
        }
        let count_field = fields.next().ok_or_else(|| RinexClockError::BadField {
            line: line_number,
            field: "count",
            value: "".to_string(),
        })?;
        let count = parse_int_field::<usize>(line_number, "count", count_field)?;
        if !(1..=6).contains(&count) {
            return Err(RinexClockError::BadField {
                line: line_number,
                field: "count",
                value: count_field.to_string(),
            });
        }
        if fields.next().is_none() {
            return Err(RinexClockError::BadField {
                line: line_number,
                field: "bias",
                value: "".to_string(),
            });
        }
        if count == 1 {
            if fields.next().is_some() {
                return Err(RinexClockError::BadField {
                    line: line_number,
                    field: "sigma",
                    value: "excess value in parent record".to_string(),
                });
            }
        } else {
            if fields.next().is_none() {
                return Err(RinexClockError::BadField {
                    line: line_number,
                    field: "sigma",
                    value: "".to_string(),
                });
            }
            if fields.next().is_some() {
                return Err(RinexClockError::BadField {
                    line: line_number,
                    field: "sigma",
                    value: "excess value in parent record".to_string(),
                });
            }
        }
        Ok(RawParent::Unsupported {
            line: line_number,
            record_type: first.to_string(),
            count,
        })
    } else {
        Err(RinexClockError::BadField {
            line: line_number,
            field: "record_type",
            value: first.to_string(),
        })
    }
}

fn parse_continuation_line(
    line_number: usize,
    line: &str,
    needed: usize,
) -> Result<Vec<f64>, RinexClockError> {
    if needed == 0 || needed > 4 {
        return Err(RinexClockError::MalformedContinuation {
            line: line_number,
            reason: "invalid needed value count",
            record: line.trim().to_string(),
        });
    }

    if line.starts_with("   ") {
        if let Some(fields) = fixed_record(line, CONT_RECORD_304_COLUMNS) {
            if fields.iter().any(|f| !f.is_empty())
                && fields.iter().all(|f| f.split_whitespace().count() <= 1)
            {
                for &f in &fields[..needed] {
                    if f.is_empty() {
                        return Err(RinexClockError::MalformedContinuation {
                            line: line_number,
                            reason: "missing required continuation value",
                            record: line.trim().to_string(),
                        });
                    }
                }
                for &f in &fields[needed..] {
                    if !f.is_empty() {
                        return Err(RinexClockError::MalformedContinuation {
                            line: line_number,
                            reason: "excess values in continuation line",
                            record: line.trim().to_string(),
                        });
                    }
                }
                let mut vals = Vec::with_capacity(needed);
                for (i, &f) in fields[..needed].iter().enumerate() {
                    let val = parse_f64_field(line_number, field_name_for_value_index(i + 2), f)
                        .map_err(|_| RinexClockError::MalformedContinuation {
                            line: line_number,
                            reason: "invalid numeric field",
                            record: line.trim().to_string(),
                        })?;
                    vals.push(val);
                }
                return Ok(vals);
            }
        }
    }

    if let Some(fields) = fixed_record(line, CONT_RECORD_300_COLUMNS) {
        if fields.iter().any(|f| !f.is_empty())
            && fields.iter().all(|f| f.split_whitespace().count() <= 1)
        {
            for &f in &fields[..needed] {
                if f.is_empty() {
                    return Err(RinexClockError::MalformedContinuation {
                        line: line_number,
                        reason: "missing required continuation value",
                        record: line.trim().to_string(),
                    });
                }
            }
            for &f in &fields[needed..] {
                if !f.is_empty() {
                    return Err(RinexClockError::MalformedContinuation {
                        line: line_number,
                        reason: "excess values in continuation line",
                        record: line.trim().to_string(),
                    });
                }
            }
            let mut vals = Vec::with_capacity(needed);
            for (i, &f) in fields[..needed].iter().enumerate() {
                let val = parse_f64_field(line_number, field_name_for_value_index(i + 2), f)
                    .map_err(|_| RinexClockError::MalformedContinuation {
                        line: line_number,
                        reason: "invalid numeric field",
                        record: line.trim().to_string(),
                    })?;
                vals.push(val);
            }
            return Ok(vals);
        }
    }

    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() < needed {
        return Err(RinexClockError::MalformedContinuation {
            line: line_number,
            reason: "too few values in continuation line",
            record: line.trim().to_string(),
        });
    }
    if tokens.len() > needed {
        return Err(RinexClockError::MalformedContinuation {
            line: line_number,
            reason: "excess values in continuation line",
            record: line.trim().to_string(),
        });
    }

    let mut vals = Vec::with_capacity(needed);
    for (i, &tok) in tokens.iter().enumerate() {
        let val =
            parse_f64_field(line_number, field_name_for_value_index(i + 2), tok).map_err(|_| {
                RinexClockError::MalformedContinuation {
                    line: line_number,
                    reason: "invalid numeric field",
                    record: line.trim().to_string(),
                }
            })?;
        vals.push(val);
    }
    Ok(vals)
}

fn parse_logical_records(
    lines: Vec<(usize, &str)>,
    time_scale: TimeScale,
    lossy: bool,
    by_sat: &mut BTreeMap<String, Vec<(ClockPoint, usize)>>,
    skipped_records: &mut Vec<RinexClockSkip>,
    diagnostics: &mut Vec<RinexClockDiagnostic>,
) -> Result<(), RinexClockError> {
    let mut i = 0;
    let mut sample_index = 0usize;

    while i < lines.len() {
        let (line_number, line) = lines[i];
        if line.trim().is_empty() {
            i += 1;
            continue;
        }

        let parent_result = parse_raw_parent(line_number, line, time_scale);

        let raw_parent = match parent_result {
            Ok(p) => p,
            Err(err) => {
                if lossy {
                    diagnostics.push(RinexClockDiagnostic::new(line_number, err));
                    i += 1;
                    continue;
                } else {
                    return Err(err);
                }
            }
        };

        match raw_parent {
            RawParent::Satellite {
                sat,
                epoch,
                bias_s,
                count,
                sigma,
            } => {
                let mut additional_values = Vec::new();
                if let Some(s) = sigma {
                    additional_values.push(s);
                }

                if count > 2 {
                    let needed = count - 2;
                    let mut cont_idx = i + 1;
                    while cont_idx < lines.len() && lines[cont_idx].1.trim().is_empty() {
                        cont_idx += 1;
                    }

                    if cont_idx >= lines.len() {
                        let err = RinexClockError::MissingContinuation {
                            line: line_number,
                            record_type: "AS".to_string(),
                        };
                        if lossy {
                            diagnostics.push(RinexClockDiagnostic::new(line_number, err));
                            i = cont_idx;
                            continue;
                        } else {
                            return Err(err);
                        }
                    }

                    let (cont_line_num, cont_line) = lines[cont_idx];
                    if is_potential_parent_record(cont_line) {
                        let err = RinexClockError::MissingContinuation {
                            line: line_number,
                            record_type: "AS".to_string(),
                        };
                        if lossy {
                            diagnostics.push(RinexClockDiagnostic::new(line_number, err));
                            i = cont_idx;
                            continue;
                        } else {
                            return Err(err);
                        }
                    }

                    match parse_continuation_line(cont_line_num, cont_line, needed) {
                        Ok(vals) => {
                            additional_values.extend(vals);
                            i = cont_idx + 1;
                        }
                        Err(err) => {
                            if lossy {
                                diagnostics.push(RinexClockDiagnostic::new(cont_line_num, err));
                                i = cont_idx + 1;
                                continue;
                            } else {
                                return Err(err);
                            }
                        }
                    }
                } else {
                    i += 1;
                }

                let point = ClockPoint {
                    epoch,
                    bias_s,
                    additional_values,
                };
                by_sat.entry(sat).or_default().push((point, sample_index));
                sample_index += 1;
            }
            RawParent::Unsupported {
                line: p_line,
                record_type,
                count,
            } => {
                if count > 2 {
                    let needed = count - 2;
                    let mut cont_idx = i + 1;
                    while cont_idx < lines.len() && lines[cont_idx].1.trim().is_empty() {
                        cont_idx += 1;
                    }

                    if cont_idx >= lines.len() {
                        let err = RinexClockError::MissingContinuation {
                            line: p_line,
                            record_type,
                        };
                        if lossy {
                            diagnostics.push(RinexClockDiagnostic::new(p_line, err));
                            i = cont_idx;
                            continue;
                        } else {
                            return Err(err);
                        }
                    }

                    let (cont_line_num, cont_line) = lines[cont_idx];
                    if is_potential_parent_record(cont_line) {
                        let err = RinexClockError::MissingContinuation {
                            line: p_line,
                            record_type,
                        };
                        if lossy {
                            diagnostics.push(RinexClockDiagnostic::new(p_line, err));
                            i = cont_idx;
                            continue;
                        } else {
                            return Err(err);
                        }
                    }

                    match parse_continuation_line(cont_line_num, cont_line, needed) {
                        Ok(_) => {
                            i = cont_idx + 1;
                        }
                        Err(err) => {
                            if lossy {
                                diagnostics.push(RinexClockDiagnostic::new(cont_line_num, err));
                                i = cont_idx + 1;
                                continue;
                            } else {
                                return Err(err);
                            }
                        }
                    }
                } else {
                    i += 1;
                }

                skipped_records.push(RinexClockSkip {
                    line: p_line,
                    record_type,
                });
            }
        }
    }

    Ok(())
}

fn next_as_field<'a, I>(
    fields: &mut I,
    line_number: usize,
    line: &str,
) -> Result<&'a str, RinexClockError>
where
    I: Iterator<Item = &'a str>,
{
    fields
        .next()
        .ok_or_else(|| RinexClockError::MalformedAsRecord {
            line: line_number,
            reason: "expected at least 10 fields",
            record: line.trim().to_string(),
        })
}

fn parse_int_field<T>(
    line_number: usize,
    field: &'static str,
    value: &str,
) -> Result<T, RinexClockError>
where
    T: std::str::FromStr,
{
    validate::strict_int(value, field).map_err(|error| map_field_error(line_number, error, value))
}

fn parse_f64_field(
    line_number: usize,
    field: &'static str,
    value: &str,
) -> Result<f64, RinexClockError> {
    validate::strict_f64(value, field).map_err(|error| map_field_error(line_number, error, value))
}

fn civil_decimal_second_to_instant(
    scale: TimeScale,
    epoch: ClockEpochFields<'_>,
) -> Result<Instant, FieldError> {
    let civil = validate::civil_datetime_with_decimal_second_policy(
        i64::from(epoch.year),
        i64::from(epoch.month),
        i64::from(epoch.day),
        i64::from(epoch.hour),
        i64::from(epoch.minute),
        epoch.second,
        civil_second_policy_for_time_scale(scale),
    )?;
    civil_microsecond_to_instant(scale, civil)
}

fn civil_microsecond_to_instant(
    scale: TimeScale,
    civil: validate::ValidCivilMicrosecond,
) -> Result<Instant, FieldError> {
    let split = civil_microsecond_to_julian_split(scale, civil)?;
    Ok(Instant::from_julian_date(scale, split))
}

// invariant: the civil fields have passed range validation before split-JD construction.
#[allow(clippy::expect_used)]
fn civil_microsecond_to_julian_split(
    scale: TimeScale,
    civil: validate::ValidCivilMicrosecond,
) -> Result<JulianDateSplit, FieldError> {
    if civil.year < 1 {
        return Err(FieldError::InvalidCivilDate {
            field: "civil datetime",
            year: civil.year,
            month: i64::from(civil.month),
            day: i64::from(civil.day),
        });
    }

    let jdn = julian_day_number(civil.year as i32, civil.month as i32, civil.day as i32);
    let jd_whole = jdn as f64 - 0.5;
    if scale == TimeScale::Utc && civil.second == 60 {
        let remaining_s = 1.0 - civil.microsecond as f64 / 1_000_000.0;
        return Ok(
            JulianDateSplit::new(jd_whole + 1.0, -remaining_s / SECONDS_PER_DAY)
                .expect("valid leap-second split Julian date"),
        );
    }

    let day_seconds = civil.hour as f64 * SECONDS_PER_HOUR
        + civil.minute as f64 * 60.0
        + civil.second as f64
        + civil.microsecond as f64 / 1_000_000.0;
    Ok(
        JulianDateSplit::new(jd_whole, day_seconds / SECONDS_PER_DAY)
            .expect("valid split Julian date"),
    )
}

fn civil_second_policy_for_time_scale(scale: TimeScale) -> validate::CivilSecondPolicy {
    match scale {
        TimeScale::Utc => validate::CivilSecondPolicy::UtcLike,
        // GLONASST is UTC(SU)-based, but a civil GLONASST leap-second (:60) label
        // is not a supported civil input: no time-system label parses to
        // GLONASST (RINEX/SP3 "GLO" is UTC), and GLONASST is reached numerically
        // via `timescale_offset_at_s`. Treat it as Continuous so a stray :60
        // GLONASST label is rejected, not silently rolled into the next minute.
        TimeScale::Glonasst
        | TimeScale::Tai
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

fn gps_seconds_from_civil(civil: validate::ValidCivilMicrosecond) -> Option<f64> {
    if civil.year < 1 {
        return None;
    }

    let days = days_since_gps_epoch(civil.year as i32, civil.month as u8, civil.day as u8);
    let whole = days as f64 * SECONDS_PER_DAY
        + (i64::from(civil.hour) * 3_600 + i64::from(civil.minute) * 60 + i64::from(civil.second))
            as f64;
    Some(whole + f64::from(civil.microsecond) / 1_000_000.0)
}

fn map_field_error(line_number: usize, error: FieldError, value: &str) -> RinexClockError {
    RinexClockError::BadField {
        line: line_number,
        field: error.field(),
        value: value.to_string(),
    }
}

fn map_epoch_error(
    line_number: usize,
    error: FieldError,
    epoch: ClockEpochFields<'_>,
) -> RinexClockError {
    match error {
        FieldError::FloatParse { .. }
        | FieldError::Missing { .. }
        | FieldError::NonFinite { .. } => RinexClockError::BadField {
            line: line_number,
            field: "second",
            value: epoch.second.to_string(),
        },
        _ => RinexClockError::BadField {
            line: line_number,
            field: "epoch",
            value: format!(
                "{} {} {} {} {} {}",
                epoch.year,
                epoch.month,
                epoch.day,
                epoch.hour,
                epoch.minute,
                normalized_second_text(epoch.second)
            ),
        },
    }
}

fn normalized_second_text(second: &str) -> String {
    validate::strict_f64(second, "second")
        .map_or_else(|_| second.to_string(), |value| value.to_string())
}

fn build_series(
    by_sat: BTreeMap<String, Vec<(ClockPoint, usize)>>,
) -> BTreeMap<String, Vec<ClockPoint>> {
    by_sat
        .into_iter()
        .map(|(sat, mut points)| {
            points.sort_by(|(a, ai), (b, bi)| {
                compare_instants(&a.epoch, &b.epoch).then_with(|| ai.cmp(bi))
            });
            (sat, dedup_by_time(points))
        })
        .collect()
}

fn dedup_by_time(points: Vec<(ClockPoint, usize)>) -> Vec<ClockPoint> {
    let mut deduped = Vec::<ClockPoint>::new();
    for (point, _) in points {
        match deduped.last_mut() {
            Some(prev) if prev.epoch == point.epoch => *prev = point,
            _ => deduped.push(point),
        }
    }
    deduped
}

fn interpolate(records: &[ClockPoint], epoch: Instant) -> Option<f64> {
    let mut prev: Option<&ClockPoint> = None;
    for point in records {
        match compare_instants_same_scale(&point.epoch, &epoch)? {
            Ordering::Equal => return Some(point.bias_s),
            Ordering::Greater => {
                let p0 = prev?;
                let p1 = point;
                let span_s = seconds_between(&p1.epoch, &p0.epoch)?;
                if span_s <= 0.0 {
                    return None;
                }
                let query_s = seconds_between(&epoch, &p0.epoch)?;
                if query_s < 0.0 {
                    return None;
                }
                return Some(lerp_ratio(p0.bias_s, p1.bias_s, query_s, span_s));
            }
            Ordering::Less => prev = Some(point),
        }
    }
    None
}

fn compare_instants(a: &Instant, b: &Instant) -> Ordering {
    time_scale_rank(a.scale)
        .cmp(&time_scale_rank(b.scale))
        .then_with(|| match (a.julian_date(), b.julian_date()) {
            (Some(a), Some(b)) => compare_julian_splits(a, b),
            _ => Ordering::Equal,
        })
}

/// Canonical clock timeline for a scale.
///
/// QZSST is synchronous with GPST (IS-QZSS-PNT sec. 3.2.2; both read TAI - 19 s),
/// so a clock file whose header tags it QZSST lives on the GPST timeline. Mapping
/// QZSST -> GPST here lets a GPST-built query instant (e.g. from
/// [`RinexClock::clock_s_at_gps_seconds`]) interpolate QZSST rows, which an
/// exact-scale match would otherwise reject. No other scale is collapsed: GST
/// carries a broadcast GGTO and the leap-second scales are genuinely distinct.
fn clock_timeline(scale: TimeScale) -> TimeScale {
    match scale {
        TimeScale::Qzsst => TimeScale::Gpst,
        other => other,
    }
}

fn compare_instants_same_scale(a: &Instant, b: &Instant) -> Option<Ordering> {
    if clock_timeline(a.scale) != clock_timeline(b.scale) {
        return None;
    }
    Some(compare_julian_splits(a.julian_date()?, b.julian_date()?))
}

fn compare_julian_splits(a: JulianDateSplit, b: JulianDateSplit) -> Ordering {
    a.jd_whole
        .partial_cmp(&b.jd_whole)
        .unwrap_or(Ordering::Equal)
        .then_with(|| {
            a.fraction
                .partial_cmp(&b.fraction)
                .unwrap_or(Ordering::Equal)
        })
}

fn seconds_between(later: &Instant, earlier: &Instant) -> Option<f64> {
    if clock_timeline(later.scale) != clock_timeline(earlier.scale) {
        return None;
    }
    let later = later.julian_date()?;
    let earlier = earlier.julian_date()?;
    let seconds = seconds_between_splits(
        later.jd_whole,
        later.fraction,
        earlier.jd_whole,
        earlier.fraction,
    );
    seconds.is_finite().then_some(seconds)
}

fn time_scale_rank(scale: TimeScale) -> u8 {
    match scale {
        TimeScale::Utc => 0,
        TimeScale::Tai => 1,
        TimeScale::Tt => 2,
        TimeScale::Tcg => 3,
        TimeScale::Tdb => 4,
        TimeScale::Tcb => 5,
        TimeScale::Gpst => 6,
        TimeScale::Gst => 7,
        TimeScale::Bdt => 8,
        TimeScale::Glonasst => 9,
        TimeScale::Qzsst => 10,
    }
}

fn days_since_gps_epoch(year: i32, month: u8, day: u8) -> i64 {
    julian_day_number(year, i32::from(month), i32::from(day)) - julian_day_number(1980, 1, 6)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn as_record(satellite: &str, bias: &str) -> String {
        format!("AS {satellite} 2020 01 01 00 00 00.000000 1 {bias}")
    }

    #[test]
    fn parse_rejects_non_finite_as_bias() {
        let err = RinexClock::parse(&as_record("G01", "NaN")).unwrap_err();
        assert_eq!(
            err,
            RinexClockError::BadField {
                line: 1,
                field: "bias",
                value: "NaN".to_string(),
            }
        );
    }

    #[test]
    fn parse_rejects_malformed_as_satellite_token() {
        let err = RinexClock::parse(&as_record("X01", "1.0e-9")).unwrap_err();
        assert_eq!(
            err,
            RinexClockError::BadField {
                line: 1,
                field: "satellite",
                value: "X01".to_string(),
            }
        );
    }

    #[test]
    fn explicit_utc_time_system_preserves_clock_epoch_scale() {
        let text = " 3.00           C                                       RINEX VERSION / TYPE\n\
                    UTC                                                     TIME SYSTEM ID\n\
                                                                        END OF HEADER\n\
                    AS G05  2017 01 01 00 00  0.000000  1   1.0e-04\n\
                    AS G05  2017 01 01 00 00 30.000000  1   2.0e-04\n";
        let clock = RinexClock::parse(text).expect("UTC RINEX clock");

        assert_eq!(clock.time_scale, TimeScale::Utc);
        assert_eq!(clock.series["G05"][0].epoch.scale, TimeScale::Utc);
        let interpolated = clock
            .clock_s(
                "G05",
                ClockEpoch {
                    year: 2017,
                    month: 1,
                    day: 1,
                    hour: 0,
                    minute: 0,
                    second: 15.0,
                },
            )
            .expect("valid clock query")
            .expect("UTC interpolated clock");
        assert!((interpolated - 1.5e-4).abs() < 1.0e-18);

        let gpst_query =
            civil_to_clock_instant(TimeScale::Gpst, 2017, 1, 1, 0, 0, 15.0).expect("GPST instant");
        assert_eq!(
            clock
                .clock_s_at_instant("G05", gpst_query)
                .expect("valid clock query"),
            None
        );

        let rows = clock.instant_series_rows();
        assert_eq!(rows[0].1[0].0.scale, TimeScale::Utc);
        let rebuilt = RinexClock::from_instant_series_rows(clock.time_scale, rows)
            .expect("valid manual RINEX clock rows");
        assert_eq!(rebuilt, clock);
    }

    #[test]
    fn manual_series_rows_reject_non_finite_inputs() {
        assert_eq!(
            RinexClock::from_series_rows(vec![("G05".to_string(), vec![(f64::NAN, 1.0e-4)])])
                .unwrap_err(),
            RinexClockError::InvalidInput {
                field: "gps_seconds",
                reason: "must be finite",
            }
        );
        assert_eq!(
            RinexClock::from_series_rows(vec![(
                "G05".to_string(),
                vec![(1_463_904_000.0, f64::INFINITY)]
            )])
            .unwrap_err(),
            RinexClockError::InvalidInput {
                field: "bias_s",
                reason: "must be finite",
            }
        );
    }

    #[test]
    fn manual_series_rows_reject_unsorted_gps_seconds() {
        assert_eq!(
            RinexClock::from_series_rows(vec![(
                "G05".to_string(),
                vec![(1_463_904_030.0, 1.0e-4), (1_463_904_000.0, 2.0e-4)]
            )])
            .unwrap_err(),
            RinexClockError::InvalidInput {
                field: "gps_seconds",
                reason: "must be strictly increasing",
            }
        );
    }

    #[test]
    fn manual_instant_rows_reject_non_finite_inputs() {
        let bad_epoch = Instant::from_julian_date(
            TimeScale::Gpst,
            JulianDateSplit {
                jd_whole: f64::NAN,
                fraction: 0.0,
            },
        );
        assert_eq!(
            RinexClock::from_instant_series_rows(
                TimeScale::Gpst,
                vec![("G05".to_string(), vec![(bad_epoch, 1.0e-4)])],
            )
            .unwrap_err(),
            RinexClockError::InvalidInput {
                field: "epoch",
                reason: "must be finite",
            }
        );

        let good_epoch =
            civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 0.0).expect("GPST instant");
        assert_eq!(
            RinexClock::from_instant_series_rows(
                TimeScale::Gpst,
                vec![("G05".to_string(), vec![(good_epoch, f64::NAN)])],
            )
            .unwrap_err(),
            RinexClockError::InvalidInput {
                field: "bias_s",
                reason: "must be finite",
            }
        );
    }

    #[test]
    fn manual_instant_rows_reject_unsorted_epochs() {
        let later =
            civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 30.0).expect("later epoch");
        let earlier =
            civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 0.0).expect("earlier epoch");

        assert_eq!(
            RinexClock::from_instant_series_rows(
                TimeScale::Gpst,
                vec![("G05".to_string(), vec![(later, 1.0e-4), (earlier, 2.0e-4)])],
            )
            .unwrap_err(),
            RinexClockError::InvalidInput {
                field: "epoch",
                reason: "must be strictly increasing",
            }
        );
    }

    #[test]
    fn rinex_clock_queries_reject_non_finite_inputs() {
        let clock = RinexClock::from_series_rows(vec![(
            "G05".to_string(),
            vec![(1_463_904_000.0, 1.0e-4)],
        )])
        .expect("valid manual RINEX clock rows");
        let bad_epoch = Instant::from_julian_date(
            TimeScale::Gpst,
            JulianDateSplit {
                jd_whole: f64::INFINITY,
                fraction: 0.0,
            },
        );
        assert_eq!(
            clock.clock_s_at_instant("G05", bad_epoch).unwrap_err(),
            RinexClockError::InvalidInput {
                field: "epoch",
                reason: "must be finite",
            }
        );
        assert_eq!(
            clock.clock_s_at_gps_seconds("G05", f64::NAN).unwrap_err(),
            RinexClockError::InvalidInput {
                field: "gps_seconds",
                reason: "must be finite",
            }
        );
        assert_eq!(
            clock
                .clock_s(
                    "G05",
                    ClockEpoch {
                        year: 2026,
                        month: 5,
                        day: 13,
                        hour: 0,
                        minute: 0,
                        second: f64::NAN,
                    },
                )
                .unwrap_err(),
            RinexClockError::InvalidInput {
                field: "epoch",
                reason: "invalid civil clock epoch",
            }
        );
    }

    #[test]
    fn interpolation_rejects_non_positive_bracket_span() {
        let day = 2_457_753.5;
        let p0 = Instant::from_julian_date(
            TimeScale::Utc,
            JulianDateSplit::new(day, 1.0).expect("valid split Julian date"),
        );
        let p1 = Instant::from_julian_date(
            TimeScale::Utc,
            JulianDateSplit::new(day + 1.0, 0.0).expect("valid split Julian date"),
        );
        let query = Instant::from_julian_date(
            TimeScale::Utc,
            JulianDateSplit::new(day + 1.0, 0.5 / SECONDS_PER_DAY)
                .expect("valid split Julian date"),
        );
        let records = [
            ClockPoint {
                epoch: p0,
                bias_s: 1.0e-4,
                additional_values: Vec::new(),
            },
            ClockPoint {
                epoch: p1,
                bias_s: 2.0e-4,
                additional_values: Vec::new(),
            },
        ];

        assert_eq!(interpolate(&records, query), None);
    }

    #[test]
    fn qzsst_rows_are_queryable_on_the_gpst_timeline() {
        // A QZSS clock file is tagged QZSST, which is synchronous with GPST. A
        // GPST-built query (clock_s_at_gps_seconds) must interpolate those rows;
        // an exact-scale match previously rejected them, returning None.
        let p0 = civil_to_clock_instant(TimeScale::Qzsst, 2026, 5, 13, 0, 0, 0.0)
            .expect("QZSST instant");
        let p1 = civil_to_clock_instant(TimeScale::Qzsst, 2026, 5, 13, 0, 0, 30.0)
            .expect("QZSST instant");
        let clock = RinexClock::from_instant_series_rows(
            TimeScale::Qzsst,
            vec![("J02".to_string(), vec![(p0, 1.0e-4), (p1, 3.0e-4)])],
        )
        .expect("QZSST clock builds");

        // QZSST civil time equals GPST civil time, so this is the GPS-seconds tag
        // of the bracket midpoint (00:00:15).
        let mid = civil_to_gps_seconds(2026, 5, 13, 0, 0, 15.0).expect("gps seconds");
        let bias = clock
            .clock_s_at_gps_seconds("J02", mid)
            .expect("query succeeds")
            .expect("QZSST row interpolates on the GPST timeline");
        assert!(
            (bias - 2.0e-4).abs() < 1.0e-12,
            "expected midpoint interpolation 2.0e-4, got {bias}"
        );

        // An exact-epoch GPST query returns the stored bias.
        let start = civil_to_gps_seconds(2026, 5, 13, 0, 0, 0.0).expect("gps seconds");
        assert_eq!(
            clock
                .clock_s_at_gps_seconds("J02", start)
                .expect("query succeeds"),
            Some(1.0e-4)
        );
    }

    #[test]
    fn to_rinex_string_round_trips_through_parse() {
        // The canonical IR is the parsed product (time scale + per-satellite
        // series). Serializing it and re-parsing must reproduce both, across
        // multiple satellites and epochs with fractional seconds.
        let text =
            "     3.00           C                                       RINEX VERSION / TYPE\n\
                    GPS                                                         TIME SYSTEM ID\n\
                                                                        END OF HEADER\n\
                    AS G05  2026 05 13 00 00  0.000000  1   -2.000000000000e-04\n\
                    AS G05  2026 05 13 00 00 30.500000  1   -2.000000600000e-04\n\
                    AS G24  2026 05 13 00 01  0.000000  1    5.000000000000e-05\n\
                    AS E11  2026 05 13 00 00  0.000000  1    1.234500000000e-09\n";
        let clock = RinexClock::parse(text).expect("parse GPST RINEX clock");
        let serialized = clock.to_rinex_string().expect("serialize RINEX clock");
        let reparsed = RinexClock::parse(&serialized).expect("re-parse serialized");
        assert_eq!(reparsed, clock, "serializer must round-trip through parse");
        // Deterministic output.
        assert_eq!(
            reparsed
                .to_rinex_string()
                .expect("serialize reparsed clock"),
            serialized
        );
    }

    #[test]
    fn to_rinex_string_round_trips_utc_time_scale() {
        // The time-system label round-trips: a UTC product re-parses as UTC.
        let text =
            "     3.00           C                                       RINEX VERSION / TYPE\n\
                    UTC                                                         TIME SYSTEM ID\n\
                                                                        END OF HEADER\n\
                    AS G05  2017 01 01 00 00  0.000000  1    1.000000000000e-04\n\
                    AS G05  2017 01 01 00 00 30.000000  1    2.000000000000e-04\n";
        let clock = RinexClock::parse(text).expect("parse UTC RINEX clock");
        assert_eq!(clock.time_scale, TimeScale::Utc);
        let serialized = clock.to_rinex_string().expect("serialize RINEX clock");
        let reparsed = RinexClock::parse(&serialized).expect("re-parse serialized");
        assert_eq!(reparsed.time_scale, TimeScale::Utc);
        assert_eq!(reparsed, clock);
    }

    #[test]
    fn to_rinex_string_rejects_unsupported_time_scale() {
        let epoch =
            civil_to_clock_instant(TimeScale::Tcg, 2026, 5, 13, 0, 0, 0.0).expect("TCG instant");
        let clock = RinexClock::from_instant_series_rows(
            TimeScale::Tcg,
            vec![("G05".to_string(), vec![(epoch, 1.0e-4)])],
        )
        .expect("TCG clock builds");

        assert_eq!(
            clock.to_rinex_string(),
            Err(RinexClockError::UnsupportedTimeScale {
                scale: TimeScale::Tcg
            })
        );
    }

    #[test]
    fn to_rinex_string_rejects_unsupported_row_time_scale() {
        let epoch =
            civil_to_clock_instant(TimeScale::Tcg, 2026, 5, 13, 0, 0, 0.0).expect("TCG instant");
        let clock = RinexClock::from_instant_series_rows(
            TimeScale::Gpst,
            vec![("G05".to_string(), vec![(epoch, 1.0e-4)])],
        )
        .expect("mixed-scale clock builds");

        assert_eq!(
            clock.to_rinex_string(),
            Err(RinexClockError::UnsupportedTimeScale {
                scale: TimeScale::Tcg
            })
        );
    }

    #[test]
    fn nanos_repr_epoch_serializes_to_true_civil_time() {
        // A `Nanos`-repr instant counts from J2000 in its own scale. The
        // serializer must render its actual civil time, not a fabricated J2000
        // (2000-01-01 12:00:00). Build the same epoch in both reprs and confirm
        // they serialize identically and the Nanos product re-parses to the
        // (Julian-date) parsed product.
        let jd_epoch =
            civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 30.0).expect("GPST instant");
        let j2000_s = instant_to_j2000_seconds(&jd_epoch).expect("J2000 seconds");
        let nanos = (j2000_s * 1.0e9).round() as i128;
        let nanos_epoch = Instant::from_nanos(TimeScale::Gpst, nanos);

        let nanos_clock = RinexClock::from_instant_series_rows(
            TimeScale::Gpst,
            vec![("G05".to_string(), vec![(nanos_epoch, 1.0e-4)])],
        )
        .expect("nanos clock builds");
        let jd_clock = RinexClock::from_instant_series_rows(
            TimeScale::Gpst,
            vec![("G05".to_string(), vec![(jd_epoch, 1.0e-4)])],
        )
        .expect("jd clock builds");

        let serialized = nanos_clock
            .to_rinex_string()
            .expect("serialize nanos RINEX clock");
        assert!(
            serialized.contains("2026 05 13 00 00 30.000000"),
            "Nanos epoch must serialize to its true civil time, got:\n{serialized}"
        );
        assert_eq!(
            serialized,
            jd_clock
                .to_rinex_string()
                .expect("serialize JD RINEX clock"),
            "Nanos- and Julian-date-repr epochs of the same instant must serialize identically"
        );

        let reparsed = RinexClock::parse(&serialized).expect("re-parse serialized Nanos product");
        assert_eq!(reparsed, jd_clock);
    }

    #[test]
    fn to_rinex_string_round_trips_utc_leap_second_epoch() {
        // The parser accepts a UTC `23:59:60.x` leap-second label, storing it as a
        // sub-midnight fraction on the next day's whole JD. The serializer must
        // reproduce that `:60` label exactly, not a wrong time from the negative
        // time-of-day.
        let text =
            "     3.00           C                                       RINEX VERSION / TYPE\n\
                    UTC                                                         TIME SYSTEM ID\n\
                                                                        END OF HEADER\n\
                    AS G05  2016 12 31 23 59 60.000000  1    1.000000000000e-04\n\
                    AS G05  2016 12 31 23 59 60.500000  1    2.000000000000e-04\n";
        let clock = RinexClock::parse(text).expect("parse UTC leap-second RINEX clock");
        let serialized = clock.to_rinex_string().expect("serialize RINEX clock");
        assert!(
            serialized.contains("23 59 60.000000"),
            "leap-second label must round-trip, got:\n{serialized}"
        );
        assert!(
            serialized.contains("23 59 60.500000"),
            "fractional leap second must round-trip, got:\n{serialized}"
        );
        let reparsed = RinexClock::parse(&serialized).expect("re-parse serialized leap second");
        assert_eq!(
            reparsed, clock,
            "leap-second epoch must round-trip bit-exact"
        );
    }

    #[test]
    fn parse_fixed_column_satellite_with_internal_space() {
        let text = "AS G  1 2026 05 13 00 00  0.000000  1   1.000000000000e-04\n";
        let clock = RinexClock::parse(text).expect("parse satellite with internal space");
        assert!(clock.series.contains_key("G01"));
        assert_eq!(clock.series["G01"].len(), 1);
    }

    #[test]
    fn parse_fixed_column_abutting_fields() {
        let text =
            "AS G01  2026 05 13 00 00  0.000000  2    2.761547232975e-04 4.197517456140e-11\n";
        let clock = RinexClock::parse(text).expect("parse abutting bias and sigma");
        assert!(clock.series.contains_key("G01"));
        let point = &clock.series["G01"][0];
        assert_eq!(point.bias_s.to_bits(), (2.761547232975e-4_f64).to_bits());
    }

    #[test]
    fn strict_parse_rejects_unrecognized_record_type() {
        let text = "XX G01  2026 05 13 00 00  0.000000  1   1.0e-04\n";
        let err =
            RinexClock::parse(text).expect_err("strict parse must reject unknown record type");
        assert_eq!(
            err,
            RinexClockError::BadField {
                line: 1,
                field: "record_type",
                value: "XX".to_string(),
            }
        );
        let lossy = RinexClock::parse_lossy(text);
        assert!(lossy.series.is_empty());
    }

    #[test]
    fn write_as_record_formats_fixed_columns_single_digit_seconds() {
        let instant = civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 5.123456).unwrap();
        let point = ClockPoint {
            epoch: instant,
            bias_s: 2.761547232975e-4,
            additional_values: Vec::new(),
        };
        let mut out = String::new();
        write_as_record(&mut out, "G01", &point).unwrap();
        assert!(
            out.starts_with("AS G01  2026 05 13 00 00  5.123456  1"),
            "expected fixed-column layout without space split in seconds: {out}"
        );
        assert!(
            fixed_record(out.trim_end(), AS_RECORD_COLUMNS).is_some(),
            "formatted record must match AS_RECORD_COLUMNS: {out}"
        );
    }

    #[test]
    fn write_as_record_formats_exact_19_column_fields_and_continuation() {
        let instant = civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 0.0).unwrap();
        let point = ClockPoint {
            epoch: instant,
            bias_s: 1.234567890123e100,
            additional_values: vec![2.761547232975e-4, -0.0, 1.234567890123e-100],
        };
        let mut out = String::new();
        write_as_record(&mut out, "G01", &point).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2);

        let parent = lines[0];
        assert_eq!(parent.len(), 79);
        assert_eq!(&parent[40..59], "1.234567890123E+100");
        assert_eq!(&parent[59..60], " ");
        assert_eq!(&parent[60..79], " 2.761547232975E-04");

        let cont = lines[1];
        assert_eq!(cont.len(), 39);
        assert_eq!(&cont[0..19], "-0.000000000000E+00");
        assert_eq!(&cont[19..20], " ");
        assert_eq!(&cont[20..39], "1.234567890123E-100");
    }

    #[test]
    fn parse_reports_unmodelled_clock_records_while_returning_modelled_satellite_series() {
        let text = "\
     3.00           C                                       RINEX VERSION / TYPE
     2    AR    AS                                          # / TYPES OF DATA
                                                            END OF HEADER
AS G01  2026 05 13 00 00  0.000000  1   1.000000000000e-04
AR ALIC 2026 05 13 00 00  0.000000  2   2.000000000000e-04 1.0e-10
AS G02  2026 05 13 00 00  0.000000  1   3.000000000000e-04
CR ALGO 2026 05 13 00 00  0.000000  2   4.000000000000e-04 1.0e-10
DR AREQ 2026 05 13 00 00  0.000000  2   5.000000000000e-04 1.0e-10
MS ASCG 2026 05 13 00 00  0.000000  2   6.000000000000e-04 1.0e-10
AS G01  2026 05 13 00 00 30.000000  1   1.500000000000e-04
";
        let clock = RinexClock::parse(text).expect("parse clock file with mixed records");
        assert_eq!(clock.series.len(), 2);
        assert_eq!(clock.series["G01"].len(), 2);
        assert_eq!(clock.series["G02"].len(), 1);
        assert_eq!(clock.series["G01"][0].bias_s, 1.0e-4);
        assert_eq!(clock.series["G01"][1].bias_s, 1.5e-4);
        assert_eq!(clock.series["G02"][0].bias_s, 3.0e-4);

        assert_eq!(clock.skipped_records.len(), 4);
        assert_eq!(
            clock.skipped_records[0],
            RinexClockSkip {
                line: 5,
                record_type: "AR".to_string(),
            }
        );
        assert_eq!(
            clock.skipped_records[1],
            RinexClockSkip {
                line: 7,
                record_type: "CR".to_string(),
            }
        );
        assert_eq!(
            clock.skipped_records[2],
            RinexClockSkip {
                line: 8,
                record_type: "DR".to_string(),
            }
        );
        assert_eq!(
            clock.skipped_records[3],
            RinexClockSkip {
                line: 9,
                record_type: "MS".to_string(),
            }
        );

        let lossy = RinexClock::parse_lossy(text);
        assert_eq!(lossy.series.len(), 2);
        assert_eq!(lossy.skipped_records, clock.skipped_records);
        assert!(lossy.diagnostics.is_empty());
    }

    #[test]
    fn format_e19_12_formats_standard_examples_and_rejects_unrepresentable() {
        assert_eq!(
            format_e19_12(-0.123456789012, "bias").unwrap(),
            "-0.123456789012E+00"
        );
        assert_eq!(
            format_e19_12(-1.23456789012, "bias").unwrap(),
            "-0.123456789012E+01"
        );
        assert_eq!(
            format_e19_12(-12.3456789012, "bias").unwrap(),
            "-0.123456789012E+02"
        );
        assert_eq!(format_e19_12(0.0, "bias").unwrap(), " 0.000000000000E+00");
        assert_eq!(format_e19_12(-0.0, "bias").unwrap(), "-0.000000000000E+00");
        assert_eq!(
            format_e19_12(1.0e-4, "bias").unwrap(),
            " 0.100000000000E-03"
        );
        assert_eq!(
            format_e19_12(2.761547232975e-4, "bias").unwrap(),
            " 2.761547232975E-04"
        );
        assert_eq!(
            format_e19_12(1.0e-105, "bias").unwrap(),
            " .100000000000E-104"
        );
        assert_eq!(
            format_e19_12(1.234567890123e100, "bias").unwrap(),
            "1.234567890123E+100"
        );
        assert_eq!(
            format_e19_12(1.234567890123e-100, "bias").unwrap(),
            "1.234567890123E-100"
        );

        assert!(format_e19_12(f64::NAN, "bias").is_err());
        assert!(format_e19_12(f64::INFINITY, "bias").is_err());
        assert!(format_e19_12(f64::NEG_INFINITY, "bias").is_err());

        let fb_neg_e100 = format_e19_12(-1.234567890123e100, "bias").unwrap();
        assert_eq!(fb_neg_e100.len(), 19);
        assert_eq!(
            fb_neg_e100.trim().parse::<f64>().unwrap().to_bits(),
            (-1.234567890123e100_f64).to_bits()
        );

        let fb_neg_em100 = format_e19_12(-1.234567890123e-100, "bias").unwrap();
        assert_eq!(fb_neg_em100.len(), 19);
        assert_eq!(
            fb_neg_em100.trim().parse::<f64>().unwrap().to_bits(),
            (-1.234567890123e-100_f64).to_bits()
        );

        assert_eq!(
            format_e19_12(-1.2345678901234e100, "bias").unwrap(),
            "-12.345678901234E99"
        );

        assert!(format_e19_12(1.23456789012345e-4, "bias").is_err());
        assert!(format_e19_12(-1.2345678901234e-100, "bias").is_err());
    }

    #[test]
    fn validate_clock_point_bounds_and_finite_checks() {
        let epoch = civil_to_clock_instant(TimeScale::Gpst, 2026, 5, 13, 0, 0, 0.0).unwrap();
        let valid = ClockPoint {
            epoch,
            bias_s: 1.0e-4,
            additional_values: vec![1.0e-5, 2.0e-6, 3.0e-7, 4.0e-8, 5.0e-9],
        };
        assert!(valid.validate().is_ok());

        let mut too_many = valid.clone();
        too_many.additional_values.push(6.0e-10);
        assert_eq!(
            too_many.validate(),
            Err(RinexClockError::InvalidInput {
                field: "additional_values",
                reason: "cannot exceed 5 additional values (maximum count is 6)",
            })
        );

        let mut non_finite = valid;
        non_finite.additional_values[2] = f64::NAN;
        assert_eq!(
            non_finite.validate(),
            Err(RinexClockError::InvalidInput {
                field: "rate_sigma",
                reason: "must be finite",
            })
        );
    }
}
