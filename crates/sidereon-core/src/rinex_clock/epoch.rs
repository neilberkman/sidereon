//! Civil clock epochs, scale-tagged instants and clock-bias interpolation.

use std::cmp::Ordering;

use crate::astro::constants::time::SECONDS_PER_DAY_I64;
use crate::astro::math::interp::lerp_ratio;
use crate::astro::time::civil::{
    civil_from_julian_day_number, j2000_seconds_from_split, seconds_from_femtoseconds,
    seconds_from_split_exact, split_julian_date_from_j2000_seconds, J2000_JULIAN_DAY_NUMBER,
    J2000_NOON_OFFSET_S,
};
use crate::astro::time::exact::{nearest_ratio, ExactSeconds};
use crate::astro::time::model::{Instant, InstantRepr, JulianDateSplit, TimeScale};
use crate::astro::time::scales::{find_leap_seconds, julian_day_number};
use crate::constants::{
    GPS_EPOCH_TO_J2000_S, J2000_JD, MICROSECONDS_PER_SECOND, SECONDS_PER_DAY, SECONDS_PER_HOUR,
};
use crate::validate::{self, FieldError};

use super::{invalid_input, ClockEpoch, ClockPoint, RinexClockError};

const GPS_SECONDS_RANGE_REASON: &str =
    "outside the civil years 1 through 9999 that a clock epoch can name";

/// Convert a civil clock tag in the given scale into a scale-tagged instant.
///
/// The second is taken as the shortest decimal that reads back to the given
/// `f64`, with every digit kept, exactly as a record's seconds field is read,
/// so a query at a record's stated epoch names that record's instant. A
/// `23:59:60` label is accepted for UTC on a day that ends with a positive
/// leap second; every other scale is continuous and refuses it.
pub fn civil_to_clock_instant(
    scale: TimeScale,
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
) -> Option<Instant> {
    let civil = clock_epoch_to_civil(
        ClockEpoch {
            year,
            month,
            day,
            hour,
            minute,
            second,
        },
        civil_second_policy_for_time_scale(scale),
    )?;
    civil_to_instant(scale, civil).ok()
}

/// Convert a civil GPS-time tag into seconds since 1980-01-06 00:00:00.
///
/// The second is read as [`civil_to_clock_instant`] reads it, and the result
/// is the `f64` nearest to the exact GPS second count the tag states. A clock
/// record read from the same tag, or built from it with
/// [`super::ClockRecord::new`], reports the same value from
/// [`super::ClockPoint::gps_seconds`] and [`super::RinexClock::series_rows`].
/// The instant [`civil_to_clock_instant`] returns does not carry the tag: a
/// sample built from that instant alone reports the same value when the tag
/// has at most ten fractional second digits, and otherwise the value nearest
/// to the instant, which can be one unit in the last place away.
pub fn civil_to_gps_seconds(
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
) -> Option<f64> {
    let civil = clock_epoch_to_civil(
        ClockEpoch {
            year,
            month,
            day,
            hour,
            minute,
            second,
        },
        validate::CivilSecondPolicy::Continuous,
    )?;
    gps_seconds_from_civil(civil)
}

/// Convert GPS seconds since 1980-01-06 into a GPST instant.
///
/// Values outside the civil years 1 through 9999 are refused: no clock epoch
/// field can name them, and past that range the day split no longer carries a
/// residual day fraction.
pub(super) fn gps_seconds_to_instant(gps_seconds: f64) -> Result<Instant, RinexClockError> {
    if !gps_seconds.is_finite() {
        return Err(invalid_input("gps_seconds", "must be finite"));
    }
    let min_s = days_since_gps_epoch(1, 1, 1) as f64 * SECONDS_PER_DAY;
    let end_s = days_since_gps_epoch(10_000, 1, 1) as f64 * SECONDS_PER_DAY;
    if !(min_s..end_s).contains(&gps_seconds) {
        return Err(invalid_input("gps_seconds", GPS_SECONDS_RANGE_REASON));
    }
    let split = gps_seconds_split(gps_seconds)
        .ok_or_else(|| invalid_input("gps_seconds", GPS_SECONDS_RANGE_REASON))?;
    Ok(Instant::from_julian_date(TimeScale::Gpst, split))
}

/// The split Julian date [`gps_seconds_to_instant`] builds from GPS seconds:
/// the civil-midnight day boundary and the seconds of that day as a fraction.
fn gps_seconds_split(gps_seconds: f64) -> Option<JulianDateSplit> {
    let gps_epoch_jd = J2000_JD - GPS_EPOCH_TO_J2000_S / SECONDS_PER_DAY;
    let days = (gps_seconds / SECONDS_PER_DAY).floor();
    let seconds_of_day = gps_seconds - days * SECONDS_PER_DAY;
    JulianDateSplit::new(gps_epoch_jd + days, seconds_of_day / SECONDS_PER_DAY).ok()
}

/// What a clock sample's instant was built from. A civil tag or GPS seconds
/// fixes the sample's GPS seconds exactly, where the instant's split Julian
/// date resolves time only to about 1e-11 s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum EpochSource {
    /// The civil tag the instant is the reader's conversion of.
    Civil(Civil),
    /// The GPS seconds [`gps_seconds_to_instant`] built the instant from.
    GpsSeconds(f64),
    /// Nothing but the instant itself.
    Instant,
}

/// GPS seconds of a sample at `epoch` built from `source`: the source's own
/// value while `epoch` is still the instant built from it, otherwise
/// [`instant_to_gps_seconds`].
pub(super) fn point_gps_seconds(epoch: &Instant, source: EpochSource) -> Option<f64> {
    if clock_timeline(epoch.scale) != TimeScale::Gpst {
        return None;
    }
    let split = epoch.julian_date()?;
    let same = |other: JulianDateSplit| {
        other.jd_whole.to_bits() == split.jd_whole.to_bits()
            && other.fraction.to_bits() == split.fraction.to_bits()
    };
    match source {
        EpochSource::Civil(civil) if civil_to_julian_split(epoch.scale, civil).is_ok_and(same) => {
            gps_seconds_from_civil(civil)
        }
        EpochSource::GpsSeconds(gps_seconds)
            if epoch.scale == TimeScale::Gpst
                && gps_seconds_split(gps_seconds).is_some_and(same) =>
        {
            Some(gps_seconds)
        }
        _ => instant_to_gps_seconds(epoch),
    }
}

/// GPS seconds of an instant on the GPST timeline.
///
/// QZSST reads the same TAI - 19 s alignment as GPST (RINEX clock 3.04 Table
/// A15; IS-QZSS-PNT sec. 3.2.2), so a QZSST epoch projects to the same GPS
/// seconds. This is the same timeline [`clock_timeline`] uses to answer a
/// GPS-seconds query against QZSST rows, so the projection and the query agree.
///
/// This is the conversion for an instant with nothing known about what it was
/// built from ([`point_gps_seconds`] uses a sample's source first). A split
/// Julian date that is the reader's conversion of a civil tag with at most
/// ten fractional second digits ([`civil_read_as_split`]) takes the GPS
/// seconds of that tag from [`gps_seconds_from_civil`]. Any other split takes
/// the `f64` nearest to the exact time its two parts hold
/// ([`split_gps_seconds`]).
pub(super) fn instant_to_gps_seconds(epoch: &Instant) -> Option<f64> {
    if clock_timeline(epoch.scale) != TimeScale::Gpst {
        return None;
    }
    let split = epoch.julian_date()?;
    match civil_read_as_split(epoch.scale, split) {
        Some(civil) => gps_seconds_from_civil(civil),
        None => Some(split_gps_seconds(split)),
    }
}

/// Julian date of the GPS epoch, 1980-01-06 00:00:00, times 86400.
const GPS_EPOCH_JD_SECONDS: i64 = 211_182_724_800;

/// GPS seconds of a split Julian date: the `f64` nearest to the exact time
/// its two parts hold. A split the fixed-point sum cannot hold (see
/// [`seconds_from_split_exact`]; no split of the civil years 1 through 9999
/// whose `jd_whole` is a whole or half day is one) takes the J2000-seconds
/// arithmetic.
fn split_gps_seconds(split: JulianDateSplit) -> f64 {
    seconds_from_split_exact(split.jd_whole, split.fraction, GPS_EPOCH_JD_SECONDS).unwrap_or_else(
        || j2000_seconds_from_split(split.jd_whole, split.fraction) + GPS_EPOCH_TO_J2000_S,
    )
}

/// The civil tag with at most ten fractional second digits whose reading
/// ([`civil_to_julian_split`] in `scale`) is `split`, bit for bit.
///
/// The reader's split of a tag lies within 5e-12 s of the tag, and the splits
/// of two tags 1e-10 s apart differ, so the nearest tag on the 1e-10 s grid to
/// the time the split holds is the only candidate, and it is accepted only
/// when its reading reproduces the split. A split read from a tag with more
/// digits can share its bits with a ten-digit tag; that tag is returned, and
/// its GPS seconds differ from the longer tag's only when a rounding boundary
/// of the `f64` grid falls between the two.
fn civil_read_as_split(scale: TimeScale, split: JulianDateSplit) -> Option<Civil> {
    const UNITS_PER_SECOND: i64 = 10_000_000_000;
    const UNITS_PER_DAY: i64 = SECONDS_PER_DAY_I64 * UNITS_PER_SECOND;
    let day_boundary = split.jd_whole + 0.5;
    if !(0.0..1.0e9).contains(&day_boundary)
        || day_boundary.fract() != 0.0
        || !(0.0..1.0).contains(&split.fraction)
    {
        return None;
    }
    let units = (split.fraction * UNITS_PER_DAY as f64).round() as i64;
    if !(0..UNITS_PER_DAY).contains(&units) {
        return None;
    }
    let (year, month, day) = civil_from_julian_day_number(day_boundary as i64);
    if !(1..=9999).contains(&year) {
        return None;
    }
    let second_of_day = units / UNITS_PER_SECOND;
    let subsecond = units % UNITS_PER_SECOND;
    let civil = Civil {
        year,
        month: month as u32,
        day: day as u32,
        hour: (second_of_day / 3_600) as u32,
        minute: (second_of_day % 3_600 / 60) as u32,
        second: (second_of_day % 60) as u32,
        microsecond: (subsecond / 10_000) as u32,
        femtosecond: (subsecond % 10_000 * 100_000) as u32,
    };
    let read = civil_to_julian_split(scale, civil).ok()?;
    (read.jd_whole.to_bits() == split.jd_whole.to_bits()
        && read.fraction.to_bits() == split.fraction.to_bits())
    .then_some(civil)
}

#[cfg(test)]
pub(super) fn instant_to_j2000_seconds(epoch: &Instant) -> Option<f64> {
    match epoch.repr {
        InstantRepr::JulianDate(split) => {
            Some(j2000_seconds_from_split(split.jd_whole, split.fraction))
        }
        InstantRepr::Nanos(_) => None,
    }
}

pub(super) fn civil_second_policy_for_time_scale(scale: TimeScale) -> validate::CivilSecondPolicy {
    match scale {
        TimeScale::Utc => validate::CivilSecondPolicy::UtcLike,
        // No RINEX clock label reads as GLONASST: GLO epochs are UTC. A
        // GLONASST leap-second label falls at 02:59:60 Moscow time, which the
        // shared civil validation does not model, so a stray :60 is refused
        // rather than rolled into the next minute.
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

/// A validated civil epoch whose second keeps up to fifteen fractional digits.
pub(super) type Civil = validate::ValidCivilFemtosecond;

/// Convert a civil epoch into an instant in `scale`, keeping every digit the
/// civil second carries that the split Julian date can hold.
pub(super) fn civil_to_instant(scale: TimeScale, civil: Civil) -> Result<Instant, FieldError> {
    let split = civil_to_julian_split(scale, civil)?;
    Ok(Instant::from_julian_date(scale, split))
}

/// The split Julian date the reader builds for a civil tag in `scale`.
///
/// `jd_whole` is the civil midnight that opens the tag's day and the fraction
/// is the `f64` nearest to the exact fraction of the day the tag states, every
/// stated digit kept, ties to even. A UTC `23:59:60.x` label is held on the
/// next day's boundary with the negative fraction of the time remaining to it,
/// likewise rounded once.
fn civil_to_julian_split(scale: TimeScale, civil: Civil) -> Result<JulianDateSplit, FieldError> {
    let invalid = || FieldError::InvalidCivilDate {
        field: "civil datetime",
        year: civil.year,
        month: i64::from(civil.month),
        day: i64::from(civil.day),
    };
    if civil.year < 1 {
        return Err(invalid());
    }

    let jdn = julian_day_number(civil.year as i32, civil.month as i32, civil.day as i32);
    let jd_whole = jdn as f64 - 0.5;
    let per_day = i128::from(SECONDS_PER_DAY_I64) * FEMTOSECONDS_PER_SECOND;
    let subminute = civil_subminute_femtoseconds(civil);
    if scale == TimeScale::Utc && civil.second == 60 {
        let remaining = 61 * FEMTOSECONDS_PER_SECOND - subminute;
        return JulianDateSplit::new(
            jd_whole + 1.0,
            nearest_ratio(-remaining, per_day.unsigned_abs()),
        )
        .map_err(|_| invalid());
    }

    let within_day = (i128::from(civil.hour) * 3_600 + i128::from(civil.minute) * 60)
        * FEMTOSECONDS_PER_SECOND
        + subminute;
    JulianDateSplit::new(jd_whole, nearest_ratio(within_day, per_day.unsigned_abs()))
        .map_err(|_| invalid())
}

/// The split Julian date the reader built for a civil tag before 3.0.0: the
/// clock fields summed in `f64` and divided by the day, which rounds more than
/// once. Kept only so the writer can restate an instant a 2.x reader built,
/// and products rebuilt from the GPS seconds exported for those splits
/// ([`gps_seconds_2x_export`]).
fn civil_to_julian_split_2x(scale: TimeScale, civil: Civil) -> Option<JulianDateSplit> {
    if civil.year < 1 {
        return None;
    }
    let jdn = julian_day_number(civil.year as i32, civil.month as i32, civil.day as i32);
    let jd_whole = jdn as f64 - 0.5;
    let microseconds_s = civil.microsecond as f64 / 1_000_000.0;
    let femtoseconds_s = f64::from(civil.femtosecond) / 1.0e15;
    if scale == TimeScale::Utc && civil.second == 60 {
        let remaining_s = if civil.femtosecond == 0 {
            1.0 - microseconds_s
        } else {
            1.0 - (microseconds_s + femtoseconds_s)
        };
        return JulianDateSplit::new(jd_whole + 1.0, -remaining_s / SECONDS_PER_DAY).ok();
    }
    let mut day_seconds = civil.hour as f64 * SECONDS_PER_HOUR
        + civil.minute as f64 * 60.0
        + civil.second as f64
        + microseconds_s;
    if civil.femtosecond != 0 {
        day_seconds += femtoseconds_s;
    }
    JulianDateSplit::new(jd_whole, day_seconds / SECONDS_PER_DAY).ok()
}

/// A validated civil epoch as the public [`ClockEpoch`]. The `f64` second is
/// the nearest double to the stated second.
pub(super) fn valid_civil_to_clock_epoch(civil: Civil) -> ClockEpoch {
    let second = seconds_from_femtoseconds(civil_subminute_femtoseconds(civil));
    ClockEpoch {
        year: civil.year as i32,
        month: civil.month as u8,
        day: civil.day as u8,
        hour: civil.hour as u8,
        minute: civil.minute as u8,
        second,
    }
}

/// Validate a public [`ClockEpoch`] under a policy. The second is taken as the
/// shortest decimal that reads back to the given `f64`, so `5.123456` is
/// 5.123456 s and `59.9999995` keeps its seventh digit.
pub(super) fn clock_epoch_to_civil(
    epoch: ClockEpoch,
    policy: validate::CivilSecondPolicy,
) -> Option<Civil> {
    if !epoch.second.is_finite() {
        return None;
    }
    let second = format!("{}", epoch.second);
    let civil = validate::civil_datetime_with_femtosecond_policy(
        i64::from(epoch.year),
        i64::from(epoch.month),
        i64::from(epoch.day),
        i64::from(epoch.hour),
        i64::from(epoch.minute),
        &second,
        policy,
    )
    .ok()?;
    (civil.year >= 1).then_some(civil)
}

/// Decompose a clock-sample instant into a civil epoch on the microsecond grid
/// the epoch field carries.
pub(super) fn instant_to_valid_civil(epoch: &Instant) -> Civil {
    let (year, month, day, hour, minute, second_us) = instant_civil_microsecond(epoch);
    Civil {
        year,
        month: month.clamp(0, i64::from(u32::MAX)) as u32,
        day: day.clamp(0, i64::from(u32::MAX)) as u32,
        hour: hour.clamp(0, i64::from(u32::MAX)) as u32,
        minute: minute.clamp(0, i64::from(u32::MAX)) as u32,
        second: (second_us / 1_000_000).clamp(0, i64::from(u32::MAX)) as u32,
        microsecond: (second_us % 1_000_000).clamp(0, i64::from(u32::MAX)) as u32,
        femtosecond: 0,
    }
}

/// Whether the microsecond epoch `civil` restates `epoch` exactly.
///
/// A nanosecond instant is restated only when it is a whole number of
/// microseconds. A split Julian date is restated when its two parts are, bit
/// for bit, the split one of the crate's conversions builds from what `civil`
/// states:
///
/// - the reader's conversion of `civil` itself;
/// - the 2.x reader's conversion of `civil` ([`civil_to_julian_split_2x`]),
///   which rounded the clock-field sum and the division by the day separately
///   and lies within about 1.2e-11 s of the tag, far inside the half
///   microsecond that separates it from every other tag;
/// - the GPS-seconds conversion ([`RinexClock::from_series_rows`]) of the
///   correctly rounded double of the GPS second count `civil` states
///   ([`gps_seconds_from_civil`]). That double is what `series_rows` exports
///   for a record read from `civil` and what GPS seconds a caller writes as
///   decimal text read to, so a product rebuilt from either is restated;
/// - the same conversion of the GPS seconds `series_rows` exported for the
///   reader's instant of `civil` before 3.0.0 ([`civil_to_julian_split_2x`]),
///   which summed the split's two parts in `f64` ([`gps_seconds_2x_export`])
///   and misses the correctly rounded double by one unit in the last place for
///   some microsecond epochs. Its two roundings (half a unit in the last
///   place of the J2000 seconds, then of the GPS seconds) keep it within 0.18
///   microseconds of the tag from 1980 through 2047, so there it lies nearer
///   that microsecond tag than any other; a product rebuilt from it is written
///   as that tag, and reading the text back gives the correctly rounded
///   double;
/// - the whole-J2000-second conversion
///   ([`crate::astro::time::split_julian_date_from_j2000_seconds`], noon day
///   boundary) of the whole J2000 second `civil` states, counted in integer
///   arithmetic from the text itself.
///
/// Every other split is not the representation of any microsecond text: it
/// may hold a sub-microsecond epoch exactly, or carry rounding from some other
/// arithmetic, and the text cannot say which. It is refused rather than
/// rounded; [`super::ClockWritePolicy`] can allow the nearest microsecond text
/// and report the departure.
///
/// [`RinexClock::from_series_rows`]: super::RinexClock::from_series_rows
pub(super) fn civil_restates_instant(civil: Civil, epoch: &Instant) -> bool {
    match epoch.repr {
        InstantRepr::JulianDate(split) => {
            let Some(read_back) = civil_to_instant(epoch.scale, civil)
                .ok()
                .and_then(|read_back| read_back.julian_date())
            else {
                return false;
            };
            let same = |other: JulianDateSplit| {
                other.jd_whole.to_bits() == split.jd_whole.to_bits()
                    && other.fraction.to_bits() == split.fraction.to_bits()
            };
            if same(read_back) {
                return true;
            }
            // The split the 2.x reader built for `civil`, persisted from a
            // 2.x read: within about 1.2e-11 s of the tag, so no other
            // microsecond tag reads to it.
            if civil_to_julian_split_2x(epoch.scale, civil).is_some_and(same) {
                return true;
            }
            if civil.femtosecond != 0 || civil.second >= 60 {
                return false;
            }
            // The GPS second count `civil` states, correctly rounded: what
            // `series_rows` exports for a record read from `civil`, and what a
            // caller's GPS seconds written as decimal text read to.
            if gps_seconds_from_civil(civil)
                .and_then(gps_seconds_split)
                .is_some_and(same)
            {
                return true;
            }
            // The GPS seconds the 2.x export gave for the 2.x reader's instant.
            if civil_to_julian_split_2x(epoch.scale, civil).is_some_and(|split_2x| {
                gps_seconds_split(gps_seconds_2x_export(split_2x)).is_some_and(same)
            }) {
                return true;
            }
            let second_of_day = i64::from(civil.hour) * 3_600
                + i64::from(civil.minute) * 60
                + i64::from(civil.second);
            // The whole J2000 second `civil` states, in exact integer arithmetic.
            if civil.microsecond == 0 {
                let days =
                    julian_day_number(civil.year as i32, civil.month as i32, civil.day as i32)
                        - J2000_JULIAN_DAY_NUMBER;
                let seconds = days * SECONDS_PER_DAY_I64 - J2000_NOON_OFFSET_S + second_of_day;
                let (jd_whole, fraction) = split_julian_date_from_j2000_seconds(seconds);
                return same(JulianDateSplit { jd_whole, fraction });
            }
            false
        }
        InstantRepr::Nanos(nanos) => nanos.rem_euclid(1_000) == 0,
    }
}

/// The GPS seconds `series_rows` exported for a split Julian date before
/// 3.0.0: the J2000 seconds of the split's two parts summed in `f64`, plus the
/// GPS epoch offset. Kept only so the writer can restate products rebuilt
/// from those exports.
fn gps_seconds_2x_export(split: JulianDateSplit) -> f64 {
    j2000_seconds_from_split(split.jd_whole, split.fraction) + GPS_EPOCH_TO_J2000_S
}

/// The civil epoch on the microsecond grid nearest to `civil`, for a writer
/// allowed to round. The stated digits are rounded as decimal text, half a
/// microsecond up, so no binary rounding of the second moves a tie or a
/// near-tie to the other side.
pub(super) fn nearest_microsecond_civil(civil: Civil) -> Option<Civil> {
    let second = format!(
        "{}.{:06}{:09}",
        civil.second, civil.microsecond, civil.femtosecond
    );
    let rounded = validate::civil_datetime_with_decimal_second_policy(
        civil.year,
        i64::from(civil.month),
        i64::from(civil.day),
        i64::from(civil.hour),
        i64::from(civil.minute),
        &second,
        validate::CivilSecondPolicy::UtcLike,
    )
    .ok()?;
    Some(Civil {
        year: rounded.year,
        month: rounded.month,
        day: rounded.day,
        hour: rounded.hour,
        minute: rounded.minute,
        second: rounded.second,
        microsecond: rounded.microsecond,
        femtosecond: 0,
    })
}

/// Decompose a clock-sample instant into civil `(year, month, day, hour, minute,
/// total-microseconds-of-minute)` on the microsecond grid the parser reads.
///
/// This inverts [`civil_to_julian_split`] on that grid: the standard epoch grid
/// from its split Julian date, a UTC `:60` leap-second epoch from its stored
/// sub-midnight fraction, and a nanosecond-repr instant from its J2000 offset.
pub(super) fn instant_civil_microsecond(epoch: &Instant) -> (i64, i64, i64, i64, i64, i64) {
    const MICROSECONDS_PER_DAY: i64 = SECONDS_PER_DAY_I64 * 1_000_000;
    let (day_number, total_us) = match epoch.repr {
        InstantRepr::JulianDate(split) => {
            // The civil midnight at or before `jd_whole`: the parser's own
            // boundary (`JDN - 0.5`) is kept as it is, and a noon boundary
            // (`*.0`) or any other is measured from the midnight before it.
            let boundary = (split.jd_whole - 0.5).floor() + 0.5;
            let offset_days = split.jd_whole - boundary;
            // A UTC leap-second epoch is stored as `remaining_s` seconds before
            // the next day's midnight: a small negative fraction on the next
            // day's boundary. Rebuild the `23:59:60.xxxxxx` label on the
            // previous civil day when that day ends with a leap second.
            if epoch.scale == TimeScale::Utc
                && offset_days == 0.0
                && (-1.0 / SECONDS_PER_DAY..0.0).contains(&split.fraction)
            {
                if let Some(leap) = leap_second_civil(split) {
                    return leap;
                }
            }
            // Read the day number and the time of day from each part
            // separately: recombining into a single JD would lose microseconds
            // to cancellation. On the parser's boundary the offset is zero and
            // this is the arithmetic the writer has always used.
            let total_us = (offset_days * SECONDS_PER_DAY * MICROSECONDS_PER_SECOND
                + split.fraction * SECONDS_PER_DAY * MICROSECONDS_PER_SECOND)
                .round() as i64;
            let day_number =
                (boundary + 0.5).round() as i64 + total_us.div_euclid(MICROSECONDS_PER_DAY);
            (day_number, total_us.rem_euclid(MICROSECONDS_PER_DAY))
        }
        // Nanoseconds count from J2000 (2000-01-01 12:00:00) in the instant's
        // own scale, matching the IONEX/SP3 convention.
        InstantRepr::Nanos(nanos) => nanos_civil_day_microsecond(nanos),
    };
    let (year, month, day) = civil_from_julian_day_number(day_number);
    let hour = total_us / 3_600_000_000;
    let rem = total_us % 3_600_000_000;
    let minute = rem / 60_000_000;
    let second_us = rem % 60_000_000;
    (year, month, day, hour, minute, second_us)
}

fn leap_second_civil(split: JulianDateSplit) -> Option<(i64, i64, i64, i64, i64, i64)> {
    let next_day_number = (split.jd_whole + 0.5).round() as i64;
    let (year, month, day) = civil_from_julian_day_number(next_day_number - 1);
    let leap_day = crate::astro::time::scales::is_positive_leap_second_label(
        i32::try_from(year).ok()?,
        i32::try_from(month).ok()?,
        i32::try_from(day).ok()?,
        23,
        59,
    );
    if !leap_day {
        return None;
    }
    let remaining_s = -split.fraction * SECONDS_PER_DAY; // in (0, 1]
    let microsecond = ((1.0 - remaining_s) * 1_000_000.0).round() as i64;
    if microsecond >= 1_000_000 {
        // Less than half a microsecond before the midnight that ends the leap
        // second: the nearest microsecond is that midnight, not a second 61.
        let (year, month, day) = civil_from_julian_day_number(next_day_number);
        return Some((year, month, day, 0, 0, 0));
    }
    Some((year, month, day, 23, 59, 60 * 1_000_000 + microsecond))
}

fn nanos_civil_day_microsecond(nanos: i128) -> (i64, i64) {
    const US_PER_DAY: i128 = SECONDS_PER_DAY_I64 as i128 * 1_000_000;
    const J2000_NOON_US: i128 = J2000_NOON_OFFSET_S as i128 * 1_000_000;
    const J2000_DAY_NUMBER: i128 = J2000_JULIAN_DAY_NUMBER as i128;
    let micros = (nanos + nanos.signum() * 500) / 1_000; // round to nearest us
    let from_midnight = J2000_NOON_US + micros;
    let day_offset = from_midnight.div_euclid(US_PER_DAY);
    let us_of_day = from_midnight.rem_euclid(US_PER_DAY);
    ((J2000_DAY_NUMBER + day_offset) as i64, us_of_day as i64)
}

pub(super) fn validate_instant(epoch: Instant, field: &'static str) -> Result<(), RinexClockError> {
    match epoch.repr {
        InstantRepr::JulianDate(split) => {
            if !split.jd_whole.is_finite() || !split.fraction.is_finite() {
                return Err(invalid_input(field, "must be finite"));
            }
            if !(-1.0..=1.0).contains(&split.fraction) {
                return Err(invalid_input(field, "Julian-date fraction out of range"));
            }
            Ok(())
        }
        InstantRepr::Nanos(_) => Ok(()),
    }
}

/// GPS seconds of a civil tag: the `f64` nearest to the exact count, formed in
/// whole femtoseconds and rounded once. Every civil-to-GPS-seconds conversion
/// of the clock module goes through here.
pub(super) fn gps_seconds_from_civil(civil: Civil) -> Option<f64> {
    if !(1..=9999).contains(&civil.year) {
        return None;
    }
    let days = days_since_gps_epoch(civil.year as i32, civil.month as u8, civil.day as u8);
    let whole_seconds = i128::from(days) * i128::from(SECONDS_PER_DAY_I64)
        + i128::from(civil.hour) * 3_600
        + i128::from(civil.minute) * 60;
    Some(seconds_from_femtoseconds(
        whole_seconds * FEMTOSECONDS_PER_SECOND + civil_subminute_femtoseconds(civil),
    ))
}

const FEMTOSECONDS_PER_SECOND: i128 = 1_000_000_000_000_000;

/// The exact seconds since J2000 a civil tag states, on the split's own terms:
/// a UTC `23:59:60.x` label, which [`civil_to_julian_split`] holds on the next
/// day's boundary less the time remaining to it, counts from that boundary.
fn civil_exact_j2000_seconds(scale: TimeScale, civil: Civil) -> ExactSeconds {
    let days = julian_day_number(civil.year as i32, civil.month as i32, civil.day as i32)
        - J2000_JULIAN_DAY_NUMBER;
    let mut whole_seconds = i128::from(days) * i128::from(SECONDS_PER_DAY_I64)
        - i128::from(J2000_NOON_OFFSET_S)
        + i128::from(civil.hour) * 3_600
        + i128::from(civil.minute) * 60;
    if scale == TimeScale::Utc && civil.second == 60 {
        // Held as the next midnight less `61 s - second`: one second earlier
        // than the label's clock fields count.
        whole_seconds -= 1;
    }
    ExactSeconds::from_decimal(
        whole_seconds * FEMTOSECONDS_PER_SECOND + civil_subminute_femtoseconds(civil),
        15,
    )
}

/// The exact seconds since J2000 a split Julian date holds.
fn split_exact_j2000_seconds(split: JulianDateSplit) -> Option<ExactSeconds> {
    // J2000 is JD 2451545.0.
    let days = ExactSeconds::from_f64(split.jd_whole)?.sub(&ExactSeconds::from_integer(
        i128::from(J2000_JULIAN_DAY_NUMBER),
    ));
    Some(
        days.mul_integer(SECONDS_PER_DAY_I64)
            .add(&ExactSeconds::from_f64(split.fraction)?.mul_integer(SECONDS_PER_DAY_I64)),
    )
}

/// The exact seconds since J2000 of GPS seconds, taken at their exact value.
fn gps_exact_j2000_seconds(gps_seconds: f64) -> Option<ExactSeconds> {
    Some(ExactSeconds::from_f64(gps_seconds)?.sub(&ExactSeconds::from_f64(GPS_EPOCH_TO_J2000_S)?))
}

/// The GPS seconds whose GPST instant ([`gps_seconds_to_instant`]) is
/// `split`, bit for bit: the `f64` nearest to the time the split holds is
/// the only candidate, accepted when its split reproduces `split`.
fn gps_seconds_read_as_split(scale: TimeScale, split: JulianDateSplit) -> Option<f64> {
    if scale != TimeScale::Gpst {
        return None;
    }
    let gps_seconds = split_gps_seconds(split);
    let read = gps_seconds_split(gps_seconds)?;
    (read.jd_whole.to_bits() == split.jd_whole.to_bits()
        && read.fraction.to_bits() == split.fraction.to_bits())
    .then_some(gps_seconds)
}

/// The exact seconds since J2000 of a clock-sample instant built from
/// `source`: the civil tag or GPS seconds it was built from while `epoch` is
/// still the instant built from them; otherwise the tag with at most ten
/// fractional second digits whose reading is `epoch`
/// ([`civil_read_as_split`]), or else the GPS seconds whose GPST instant it
/// is ([`gps_seconds_read_as_split`]), when there is one; otherwise the exact
/// time the split holds. `None` for a nanosecond instant.
fn exact_j2000_seconds(epoch: &Instant, source: EpochSource) -> Option<ExactSeconds> {
    let split = epoch.julian_date()?;
    let same = |other: JulianDateSplit| {
        other.jd_whole.to_bits() == split.jd_whole.to_bits()
            && other.fraction.to_bits() == split.fraction.to_bits()
    };
    match source {
        EpochSource::Civil(civil) if civil_to_julian_split(epoch.scale, civil).is_ok_and(same) => {
            Some(civil_exact_j2000_seconds(epoch.scale, civil))
        }
        EpochSource::GpsSeconds(gps_seconds)
            if epoch.scale == TimeScale::Gpst
                && gps_seconds_split(gps_seconds).is_some_and(same) =>
        {
            gps_exact_j2000_seconds(gps_seconds)
        }
        _ => match civil_read_as_split(epoch.scale, split) {
            Some(civil) => Some(civil_exact_j2000_seconds(epoch.scale, civil)),
            None => match gps_seconds_read_as_split(epoch.scale, split) {
                Some(gps_seconds) => gps_exact_j2000_seconds(gps_seconds),
                None => split_exact_j2000_seconds(split),
            },
        },
    }
}

/// The second of the minute `civil` states, in whole femtoseconds.
fn civil_subminute_femtoseconds(civil: Civil) -> i128 {
    i128::from(civil.second) * FEMTOSECONDS_PER_SECOND
        + i128::from(civil.microsecond) * 1_000_000_000
        + i128::from(civil.femtosecond)
}

fn days_since_gps_epoch(year: i32, month: u8, day: u8) -> i64 {
    julian_day_number(year, i32::from(month), i32::from(day)) - julian_day_number(1980, 1, 6)
}

/// The bias of `records` at `epoch`, built from `source`: a sample's own bias
/// at its epoch, otherwise interpolated linearly between the samples on either
/// side over intervals measured by [`seconds_between`].
pub(super) fn interpolate(
    records: &[ClockPoint],
    epoch: Instant,
    source: EpochSource,
) -> Option<f64> {
    let mut prev: Option<&ClockPoint> = None;
    for point in records {
        match compare_elapsed((&point.epoch, point.source), (&epoch, source))? {
            Ordering::Equal => return Some(point.bias_s),
            Ordering::Greater => {
                let p0 = prev?;
                let p1 = point;
                let span_s = seconds_between((&p1.epoch, p1.source), (&p0.epoch, p0.source))?;
                if span_s <= 0.0 {
                    return None;
                }
                let query_s = seconds_between((&epoch, source), (&p0.epoch, p0.source))?;
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

/// The bias of the one sample of `records` whose GPS seconds
/// ([`ClockPoint::gps_seconds`], as `series_rows` exports them) are
/// `gps_seconds`; `query` is the GPST instant [`gps_seconds_to_instant`]
/// builds from them.
///
/// Every sample whose GPS seconds are `gps_seconds` lies within half a unit in
/// the last place of them (plus the 5e-12 s by which a split can miss the tag
/// it was read from), and so does `query`; the samples between any of them
/// and `query` round to the same GPS seconds. The matching samples are
/// therefore the run on either side of where `query` falls. `None` when no
/// sample, or more than one, has those GPS seconds.
pub(super) fn sample_at_gps_seconds(
    records: &[ClockPoint],
    query: &Instant,
    gps_seconds: f64,
) -> Option<f64> {
    let after = records
        .iter()
        .position(|point| {
            compare_elapsed(
                (&point.epoch, point.source),
                (query, EpochSource::GpsSeconds(gps_seconds)),
            ) == Some(Ordering::Greater)
        })
        .unwrap_or(records.len());
    let at = |index: usize| {
        records
            .get(index)
            .is_some_and(|point| point.gps_seconds() == Some(gps_seconds))
    };
    let mut first = after;
    while first > 0 && at(first - 1) {
        first -= 1;
    }
    let mut end = after;
    while at(end) {
        end += 1;
    }
    if end - first == 1 {
        records.get(first).map(|point| point.bias_s)
    } else {
        None
    }
}

/// Total order of clock epochs: by time scale, then by time. Split Julian
/// dates are ordered by the elapsed time [`compare_elapsed`] orders them by,
/// whatever day boundary each is held on, so a `23:59:60` epoch (stored on
/// the next day's boundary) follows `23:59:59` of its day; nanosecond
/// instants compare as integers and follow every split Julian date of their
/// scale.
pub(super) fn epoch_cmp(a: &Instant, b: &Instant) -> Ordering {
    time_scale_rank(a.scale)
        .cmp(&time_scale_rank(b.scale))
        .then_with(|| match (a.repr, b.repr) {
            (InstantRepr::JulianDate(x), InstantRepr::JulianDate(y)) => {
                compare_elapsed((a, EpochSource::Instant), (b, EpochSource::Instant))
                    .unwrap_or_else(|| compare_julian_splits(x, y))
            }
            (InstantRepr::Nanos(x), InstantRepr::Nanos(y)) => x.cmp(&y),
            (InstantRepr::JulianDate(_), InstantRepr::Nanos(_)) => Ordering::Less,
            (InstantRepr::Nanos(_), InstantRepr::JulianDate(_)) => Ordering::Greater,
        })
}

/// Canonical clock timeline for a scale.
///
/// QZSST is synchronous with GPST (IS-QZSS-PNT sec. 3.2.2; RINEX clock 3.04
/// Table A15 aligns both to TAI - 19 s), so a clock file whose header tags it
/// QZSST lives on the GPST timeline. Mapping QZSST -> GPST lets a GPST-built
/// query instant (for example from [`super::RinexClock::clock_s_at_gps_seconds`])
/// interpolate QZSST rows, and [`instant_to_gps_seconds`] projects QZSST rows
/// to GPS seconds on the same ground. No other scale is collapsed: GST carries
/// a broadcast GGTO and the leap-second scales are distinct.
pub(super) fn clock_timeline(scale: TimeScale) -> TimeScale {
    match scale {
        TimeScale::Qzsst => TimeScale::Gpst,
        other => other,
    }
}

/// Instants whose `f64` J2000 seconds lie farther apart than this are
/// ordered by them; the `f64` misses the exact value by less than 1e-4 s
/// across the civil years 1 through 9999.
const EXACT_ORDER_WINDOW_S: f64 = 1.0e-3;

/// Time order of two instants on one clock timeline, each with what it was
/// built from: the order of the elapsed values [`seconds_between`] measures,
/// so interpolation brackets a query by the same times it measures. Two
/// identical splits of one scale are one instant. `None` across timelines or
/// for a nanosecond instant.
fn compare_elapsed(
    (a, a_source): (&Instant, EpochSource),
    (b, b_source): (&Instant, EpochSource),
) -> Option<Ordering> {
    if clock_timeline(a.scale) != clock_timeline(b.scale) {
        return None;
    }
    let (x, y) = (a.julian_date()?, b.julian_date()?);
    if a.scale == b.scale
        && x.jd_whole.to_bits() == y.jd_whole.to_bits()
        && x.fraction.to_bits() == y.fraction.to_bits()
    {
        return Some(Ordering::Equal);
    }
    let approximate = |epoch: &Instant, split: JulianDateSplit| {
        let seconds = j2000_seconds_from_split(split.jd_whole, split.fraction);
        if epoch.scale == TimeScale::Utc {
            seconds + utc_leap_count(split)
        } else {
            seconds
        }
    };
    let (ax, bx) = (approximate(a, x), approximate(b, y));
    if (ax - bx).abs() > EXACT_ORDER_WINDOW_S {
        return ax.partial_cmp(&bx);
    }
    Some(
        elapsed_seconds(a, a_source)?
            .sub(&elapsed_seconds(b, b_source)?)
            .sign(),
    )
}

/// The elapsed-time value of an instant built from `source`: its exact
/// seconds since J2000 ([`exact_j2000_seconds`]) plus, on UTC, the TAI - UTC
/// count it takes, so the value runs with elapsed time across a leap second.
fn elapsed_seconds(epoch: &Instant, source: EpochSource) -> Option<ExactSeconds> {
    let value = exact_j2000_seconds(epoch, source)?;
    if epoch.scale != TimeScale::Utc {
        return Some(value);
    }
    Some(value.add(&ExactSeconds::from_f64(utc_leap_count(
        epoch.julian_date()?,
    ))?))
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

/// Elapsed seconds between two instants on one clock timeline, each with what
/// it was built from.
///
/// Each instant is taken at the exact time [`exact_j2000_seconds`] gives it,
/// so two tags a whole number of seconds apart are exactly that far apart, and
/// the difference is rounded once, to the nearest `f64`.
///
/// UTC labels skip or repeat seconds at a leap second, so the label difference
/// is corrected by the change in TAI - UTC between the two epochs. A
/// `23:59:60.x` epoch is stored on the next day's whole Julian date and so
/// takes the post-leap count, which makes `23:59:59 -> 23:59:60` one second and
/// `23:59:60 -> 00:00:00` one second, as elapsed time runs.
pub(super) fn seconds_between(
    (later, later_source): (&Instant, EpochSource),
    (earlier, earlier_source): (&Instant, EpochSource),
) -> Option<f64> {
    if clock_timeline(later.scale) != clock_timeline(earlier.scale) {
        return None;
    }
    let seconds = elapsed_seconds(later, later_source)?
        .sub(&elapsed_seconds(earlier, earlier_source)?)
        .to_f64();
    seconds.is_finite().then_some(seconds)
}

fn utc_leap_count(split: JulianDateSplit) -> f64 {
    find_leap_seconds(split.jd_whole + split.fraction.max(0.0))
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
