//! RINEX clock (`.CLK`) satellite-clock parser and interpolation.
//!
//! The parser owns the product grammar for `AS` satellite clock-bias records.
//! The strict parser reports malformed `AS` rows. Use
//! [`RinexClock::parse_lossy`] only when best-effort input recovery is intended.

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
use crate::constants::{GPS_EPOCH_TO_J2000_S, J2000_JD, SECONDS_PER_DAY};
use crate::validate::{self, FieldError};

const INSTANT_SCALE_ORDER_STRIDE_S: f64 = 1.0e15;

/// One satellite clock-bias sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockPoint {
    /// Scale-tagged epoch from the RINEX clock file's declared time system.
    pub epoch: Instant,
    /// Satellite clock bias in seconds.
    pub bias_s: f64,
}

impl ClockPoint {
    /// This sample's epoch as GPS seconds, when the sample is actually GPST.
    pub fn gps_seconds(&self) -> Option<f64> {
        instant_to_gps_seconds(&self.epoch)
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

/// Parsed RINEX clock product.
#[derive(Debug, Clone, PartialEq)]
pub struct RinexClock {
    /// Time scale declared by the RINEX clock header. Missing headers default to GPST.
    pub time_scale: TimeScale,
    /// Per-satellite, strictly time-ordered clock-bias series.
    pub series: BTreeMap<String, Vec<ClockPoint>>,
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
            RinexClockError::BadField { line, field, value } => write!(
                f,
                "bad RINEX AS clock field at line {line}: {field}={value}"
            ),
            RinexClockError::InvalidInput { field, reason } => {
                write!(f, "invalid RINEX clock input {field}: {reason}")
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

        for (line_number, line) in lines {
            if let Some((sat, point)) = parse_record(line_number, line, time_scale)? {
                by_sat.entry(sat).or_default().push((point, line_number));
            }
        }

        Ok(Self {
            time_scale,
            series: build_series(by_sat),
        })
    }

    /// Parse a RINEX clock text while skipping malformed and non-`AS` records.
    pub fn parse_lossy(text: &str) -> Self {
        let time_scale = parse_time_scale(text).unwrap_or(TimeScale::Gpst);
        let lines = data_lines(text);
        let mut by_sat = BTreeMap::<String, Vec<(ClockPoint, usize)>>::new();

        for (line_number, line) in lines {
            if let Ok(Some((sat, point))) = parse_record(line_number, line, time_scale) {
                by_sat.entry(sat).or_default().push((point, line_number));
            }
        }

        Self {
            time_scale,
            series: build_series(by_sat),
        }
    }

    /// Rebuild a GPST product from the legacy public GPS-second row shape.
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
                    let point = ClockPoint { epoch, bias_s };
                    validate_clock_point(point)?;
                    Ok((point, idx))
                })
                .collect::<Result<Vec<_>, RinexClockError>>()?;
            validate_instant_series_order(&indexed)?;
            indexed.sort_by(|(a, ai), (b, bi)| {
                compare_instants(&a.epoch, &b.epoch).then_with(|| ai.cmp(bi))
            });
            series.insert(sat, dedup_by_time(indexed));
        }
        Ok(Self { time_scale, series })
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

    /// Serialize this product to standard RINEX clock text - the inverse of
    /// [`RinexClock::parse`].
    ///
    /// Pure and deterministic: the same product always produces byte-identical
    /// text and no I/O is performed. The header declares the product time system
    /// and each sample is written as an `AS` satellite clock-bias record, so
    /// re-parsing the output reproduces the same time scale and per-satellite
    /// series. Epoch components are written on the microsecond civil grid the
    /// parser reads, and bias values use their shortest round-tripping decimal,
    /// so a parsed product re-encodes to the same `f64`s.
    pub fn to_rinex_string(&self) -> String {
        let mut out = String::new();
        let label = crate::rinex_common::time_scale_rinex_label(self.time_scale);
        let _ = writeln!(out, "{:<60}RINEX VERSION / TYPE", "     3.00           C");
        let _ = writeln!(out, "{label:<60}TIME SYSTEM ID");
        let _ = writeln!(out, "{:<60}END OF HEADER", "");
        for (satellite, points) in &self.series {
            for point in points {
                write_as_record(&mut out, satellite, point);
            }
        }
        out
    }
}

/// Append one `AS` satellite clock-bias record for a sample.
fn write_as_record(out: &mut String, satellite: &str, point: &ClockPoint) {
    let (year, month, day, hour, minute, second_us) = instant_civil_microsecond(&point.epoch);
    let second = second_us / 1_000_000;
    let microsecond = second_us % 1_000_000;
    // RINEX clock epochs are space-delimited (the parser splits on whitespace),
    // and one data value (the bias) is written.
    let _ = writeln!(
        out,
        "AS {satellite:<3} {year:04} {month:02} {day:02} {hour:02} {minute:02} {second:2}.{microsecond:06}  1  {bias}",
        bias = point.bias_s,
    );
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
            let total_us = (split.fraction * 86_400.0 * 1_000_000.0).round() as i64;
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

fn validate_clock_point(point: ClockPoint) -> Result<(), RinexClockError> {
    validate_instant(point.epoch, "epoch")?;
    validate_finite(point.bias_s, "bias_s")
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

fn parse_record(
    line_number: usize,
    line: &str,
    time_scale: TimeScale,
) -> Result<Option<(String, ClockPoint)>, RinexClockError> {
    let mut fields = line.split_whitespace();
    if fields.next() != Some("AS") {
        return Ok(None);
    }

    let sat_field = next_as_field(&mut fields, line_number, line)?;
    let year_field = next_as_field(&mut fields, line_number, line)?;
    let month_field = next_as_field(&mut fields, line_number, line)?;
    let day_field = next_as_field(&mut fields, line_number, line)?;
    let hour_field = next_as_field(&mut fields, line_number, line)?;
    let minute_field = next_as_field(&mut fields, line_number, line)?;
    let second_field = next_as_field(&mut fields, line_number, line)?;
    let _value_count_field = next_as_field(&mut fields, line_number, line)?;
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

    Ok(Some((sat, ClockPoint { epoch, bias_s })))
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

    let day_seconds = civil.hour as f64 * 3600.0
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
        | TimeScale::Tdb
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
    let mut prev: Option<ClockPoint> = None;
    for point in records {
        match compare_instants_same_scale(&point.epoch, &epoch)? {
            Ordering::Equal => return Some(point.bias_s),
            Ordering::Greater => {
                let p0 = prev?;
                let p1 = *point;
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
            Ordering::Less => prev = Some(*point),
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
        TimeScale::Tdb => 3,
        TimeScale::Gpst => 4,
        TimeScale::Gst => 5,
        TimeScale::Bdt => 6,
        TimeScale::Glonasst => 7,
        TimeScale::Qzsst => 8,
    }
}

fn days_since_gps_epoch(year: i32, month: u8, day: u8) -> i64 {
    julian_day_number(year, i32::from(month), i32::from(day)) - julian_day_number(1980, 1, 6)
}

#[cfg(test)]
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
            },
            ClockPoint {
                epoch: p1,
                bias_s: 2.0e-4,
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
        let reparsed = RinexClock::parse(&clock.to_rinex_string()).expect("re-parse serialized");
        assert_eq!(reparsed, clock, "serializer must round-trip through parse");
        // Deterministic output.
        assert_eq!(reparsed.to_rinex_string(), clock.to_rinex_string());
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
        let reparsed = RinexClock::parse(&clock.to_rinex_string()).expect("re-parse serialized");
        assert_eq!(reparsed.time_scale, TimeScale::Utc);
        assert_eq!(reparsed, clock);
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

        let serialized = nanos_clock.to_rinex_string();
        assert!(
            serialized.contains("2026 05 13 00 00 30.000000"),
            "Nanos epoch must serialize to its true civil time, got:\n{serialized}"
        );
        assert_eq!(
            serialized,
            jd_clock.to_rinex_string(),
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
        let serialized = clock.to_rinex_string();
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
}
