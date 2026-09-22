//! GNSS week / time-of-week calendar conversions.
//!
//! The single home for the constellation week-numbering epochs (GPS, Galileo,
//! QZSS share the 1980-01-06 GPS epoch; BeiDou counts from 2006-01-01) and the
//! calendar <-> week/TOW arithmetic the RINEX navigation reader and writer each
//! carried a private copy of. All calendar steps delegate to
//! [`super::scales::julian_day_number`] / [`super::civil`] so there is one
//! Gregorian conversion under everything here.

use super::model::TimeScale;
use super::scales::julian_day_number;
use crate::constants::{SECONDS_PER_DAY, SECONDS_PER_WEEK};

/// Integer Julian Day Number of a constellation's week-numbering epoch, or
/// `None` for a scale that does not use continuous GNSS weeks (GLONASS/UTC and
/// the atomic scales).
///
/// QZSST shares the GPS epoch (it is steered synchronous with GPST).
#[must_use]
pub fn week_epoch_julian_day_number(system: TimeScale) -> Option<i64> {
    match system {
        TimeScale::Gpst | TimeScale::Gst | TimeScale::Qzsst => Some(julian_day_number(1980, 1, 6)),
        TimeScale::Bdt => Some(julian_day_number(2006, 1, 1)),
        TimeScale::Glonasst
        | TimeScale::Utc
        | TimeScale::Tai
        | TimeScale::Tt
        | TimeScale::Tcg
        | TimeScale::Tdb
        | TimeScale::Tcb => None,
    }
}

/// GNSS week number for a calendar date in `system`'s week numbering, or `None`
/// for a date before the epoch, an invalid calendar date, or a scale without GNSS weeks.
#[must_use]
pub fn week_from_calendar(system: TimeScale, year: i64, month: i64, day: i64) -> Option<u32> {
    if !(1..=12).contains(&month) {
        return None;
    }
    let last_day = super::civil::days_in_month(year, month);
    if !(1..=last_day).contains(&day) {
        return None;
    }
    let year_i32 = i32::try_from(year).ok()?;
    let month_i32 = i32::try_from(month).ok()?;
    let day_i32 = i32::try_from(day).ok()?;
    let epoch_jdn = week_epoch_julian_day_number(system)?;
    let elapsed_days = julian_day_number(year_i32, month_i32, day_i32).checked_sub(epoch_jdn)?;
    if elapsed_days < 0 {
        return None;
    }
    u32::try_from(elapsed_days / 7).ok()
}

/// Seconds-of-week of a calendar epoch in its own system time, with the GNSS
/// Sunday-00:00 origin. Sakamoto's day-of-week gives 0 = Sunday.
///
/// Returns `None` if any calendar date or time field is out of range:
/// `month` not in `1..=12`, `day` not in `1..=days_in_month(year, month)`,
/// `hour` not in `0..=23`, `minute` not in `0..=59`, or `second` not in `0..=60`.
#[must_use]
pub fn seconds_of_week_from_calendar(
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    second: i64,
) -> Option<f64> {
    if !(1..=12).contains(&month) {
        return None;
    }
    let last_day = super::civil::days_in_month(year, month);
    if !(1..=last_day).contains(&day) {
        return None;
    }
    if !(0..=23).contains(&hour) || !(0..=59).contains(&minute) || !(0..=60).contains(&second) {
        return None;
    }
    const T: [i64; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let normalized_year = year.rem_euclid(400) + 400;
    let y = if month < 3 {
        normalized_year - 1
    } else {
        normalized_year
    };
    let dow = (y + y / 4 - y / 100 + y / 400 + T[(month - 1) as usize] + day).rem_euclid(7);
    Some(dow as f64 * SECONDS_PER_DAY + (hour * 3600 + minute * 60 + second) as f64)
}

/// Decompose continuous seconds since a constellation's week epoch into an
/// integer week count and seconds-of-week, both as `f64`.
///
/// The week is `floor(seconds / 604800)` and the seconds-of-week is the residual
/// `seconds - week * 604800`. Both are returned as `f64` so a caller range-checks
/// the week against its own target integer width before narrowing; this is the
/// single home for the week/seconds-of-week split the SP3 combiner open-coded.
#[must_use]
pub fn week_and_seconds_of_week(continuous_seconds: f64) -> (f64, f64) {
    let week = (continuous_seconds / SECONDS_PER_WEEK).floor();
    let seconds_of_week = continuous_seconds - week * SECONDS_PER_WEEK;
    (week, seconds_of_week)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn week_and_seconds_of_week_splits_continuous_seconds() {
        assert_eq!(week_and_seconds_of_week(0.0), (0.0, 0.0));
        assert_eq!(week_and_seconds_of_week(SECONDS_PER_WEEK), (1.0, 0.0));
        let (week, sow) = week_and_seconds_of_week(SECONDS_PER_WEEK * 3.0 + 123.5);
        assert_eq!(week, 3.0);
        assert!((sow - 123.5).abs() < 1e-9);
    }

    #[test]
    fn week_epoch_metadata_is_centralized() {
        assert_eq!(
            week_epoch_julian_day_number(TimeScale::Gpst),
            Some(julian_day_number(1980, 1, 6))
        );
        assert_eq!(
            week_epoch_julian_day_number(TimeScale::Qzsst),
            week_epoch_julian_day_number(TimeScale::Gpst)
        );
        assert_eq!(
            week_epoch_julian_day_number(TimeScale::Bdt),
            Some(julian_day_number(2006, 1, 1))
        );
        assert_eq!(week_epoch_julian_day_number(TimeScale::Glonasst), None);
    }

    #[test]
    fn week_from_calendar_known_values() {
        // GPS week 0 starts 1980-01-06; week 1024 rollover is 1999-08-22.
        assert_eq!(week_from_calendar(TimeScale::Gpst, 1980, 1, 6), Some(0));
        assert_eq!(week_from_calendar(TimeScale::Gpst, 1980, 1, 13), Some(1));
        assert_eq!(week_from_calendar(TimeScale::Gpst, 1999, 8, 22), Some(1024));
        // BeiDou week 0 starts 2006-01-01.
        assert_eq!(week_from_calendar(TimeScale::Bdt, 2006, 1, 1), Some(0));
        // Before-epoch and non-week scales yield None.
        assert_eq!(week_from_calendar(TimeScale::Gpst, 1979, 1, 1), None);
        assert_eq!(week_from_calendar(TimeScale::Glonasst, 2020, 1, 1), None);
        // Out-of-range calendar fields yield None.
        assert_eq!(week_from_calendar(TimeScale::Gpst, 2026, 0, 1), None);
        assert_eq!(week_from_calendar(TimeScale::Gpst, 2026, 13, 1), None);
        assert_eq!(week_from_calendar(TimeScale::Gpst, 2026, 2, 29), None);
        assert_eq!(week_from_calendar(TimeScale::Gpst, 2024, 2, 29), Some(2303));
    }

    #[test]
    fn seconds_of_week_sunday_origin() {
        // 1980-01-06 was a Sunday -> dow 0, sow = time of day.
        assert_eq!(
            seconds_of_week_from_calendar(1980, 1, 6, 0, 0, 0),
            Some(0.0)
        );
        assert_eq!(
            seconds_of_week_from_calendar(1980, 1, 6, 1, 2, 3),
            Some(3723.0)
        );
        // The following Monday is dow 1.
        assert_eq!(
            seconds_of_week_from_calendar(1980, 1, 7, 0, 0, 0),
            Some(SECONDS_PER_DAY)
        );
        // Out-of-range calendar and time fields return None.
        assert_eq!(seconds_of_week_from_calendar(1980, 0, 6, 0, 0, 0), None);
        assert_eq!(seconds_of_week_from_calendar(1980, 13, 6, 0, 0, 0), None);
        assert_eq!(seconds_of_week_from_calendar(1980, 1, 0, 0, 0, 0), None);
        assert_eq!(seconds_of_week_from_calendar(1980, 1, 32, 0, 0, 0), None);
        assert_eq!(seconds_of_week_from_calendar(1980, 1, 6, 24, 0, 0), None);
        assert_eq!(seconds_of_week_from_calendar(1980, 1, 6, 0, 60, 0), None);
        assert_eq!(seconds_of_week_from_calendar(1980, 1, 6, 0, 0, 61), None);
        assert_eq!(seconds_of_week_from_calendar(1980, 1, 6, -1, 0, 0), None);
    }

    #[test]
    fn seconds_of_week_known_weekday_anchors() {
        // Leap year 2000 anchors:
        // 2000-01-01 Saturday (dow 6)
        assert_eq!(
            seconds_of_week_from_calendar(2000, 1, 1, 0, 0, 0),
            Some(6.0 * SECONDS_PER_DAY)
        );
        // 2000-02-29 Tuesday (dow 2), leap day accepted
        assert_eq!(
            seconds_of_week_from_calendar(2000, 2, 29, 0, 0, 0),
            Some(2.0 * SECONDS_PER_DAY)
        );
        // 2000-03-01 Wednesday (dow 3)
        assert_eq!(
            seconds_of_week_from_calendar(2000, 3, 1, 0, 0, 0),
            Some(3.0 * SECONDS_PER_DAY)
        );

        // Non-leap century year 1900 anchors:
        // 1900-01-01 Monday (dow 1)
        assert_eq!(
            seconds_of_week_from_calendar(1900, 1, 1, 0, 0, 0),
            Some(1.0 * SECONDS_PER_DAY)
        );
        // 1900-02-29 is refused (not a leap year)
        assert_eq!(seconds_of_week_from_calendar(1900, 2, 29, 0, 0, 0), None);
        // 1900-03-01 Thursday (dow 4)
        assert_eq!(
            seconds_of_week_from_calendar(1900, 3, 1, 0, 0, 0),
            Some(4.0 * SECONDS_PER_DAY)
        );

        // Year 0 belongs to the same Gregorian cycle as 2000:
        // 0000-01-01 Saturday (dow 6)
        assert_eq!(
            seconds_of_week_from_calendar(0, 1, 1, 0, 0, 0),
            Some(6.0 * SECONDS_PER_DAY)
        );
        // 0000-02-29 Tuesday (dow 2), leap day accepted
        assert_eq!(
            seconds_of_week_from_calendar(0, 2, 29, 0, 0, 0),
            Some(2.0 * SECONDS_PER_DAY)
        );
        // 0000-03-01 Wednesday (dow 3)
        assert_eq!(
            seconds_of_week_from_calendar(0, 3, 1, 0, 0, 0),
            Some(3.0 * SECONDS_PER_DAY)
        );
    }

    #[test]
    fn seconds_of_week_negative_years_and_400_year_equivalence() {
        // Year -400 belongs to the 2000/0 cycle (leap year):
        assert_eq!(
            seconds_of_week_from_calendar(-400, 1, 1, 0, 0, 0),
            Some(6.0 * SECONDS_PER_DAY)
        );
        assert_eq!(
            seconds_of_week_from_calendar(-400, 2, 29, 0, 0, 0),
            Some(2.0 * SECONDS_PER_DAY)
        );
        assert_eq!(
            seconds_of_week_from_calendar(-400, 3, 1, 0, 0, 0),
            Some(3.0 * SECONDS_PER_DAY)
        );

        // Year -100 belongs to the 1900 cycle (non-leap):
        assert_eq!(
            seconds_of_week_from_calendar(-100, 1, 1, 0, 0, 0),
            Some(1.0 * SECONDS_PER_DAY)
        );
        assert_eq!(seconds_of_week_from_calendar(-100, 2, 29, 0, 0, 0), None);
        assert_eq!(
            seconds_of_week_from_calendar(-100, 3, 1, 0, 0, 0),
            Some(4.0 * SECONDS_PER_DAY)
        );

        // Explicit 400-year periodicity checks across negative, zero, and positive years
        let sow_anchor = seconds_of_week_from_calendar(2000, 2, 29, 12, 34, 56);
        assert_eq!(
            seconds_of_week_from_calendar(-800, 2, 29, 12, 34, 56),
            sow_anchor
        );
        assert_eq!(
            seconds_of_week_from_calendar(-400, 2, 29, 12, 34, 56),
            sow_anchor
        );
        assert_eq!(
            seconds_of_week_from_calendar(0, 2, 29, 12, 34, 56),
            sow_anchor
        );
        assert_eq!(
            seconds_of_week_from_calendar(400, 2, 29, 12, 34, 56),
            sow_anchor
        );
        assert_eq!(
            seconds_of_week_from_calendar(2400, 2, 29, 12, 34, 56),
            sow_anchor
        );

        let sow_1900_mar1 = seconds_of_week_from_calendar(1900, 3, 1, 8, 15, 30);
        assert_eq!(
            seconds_of_week_from_calendar(-100, 3, 1, 8, 15, 30),
            sow_1900_mar1
        );
        assert_eq!(
            seconds_of_week_from_calendar(300, 3, 1, 8, 15, 30),
            sow_1900_mar1
        );
        assert_eq!(
            seconds_of_week_from_calendar(2300, 3, 1, 8, 15, 30),
            sow_1900_mar1
        );
    }

    #[test]
    fn seconds_of_week_i64_extremes() {
        // i64::MIN = -9223372036854775808 = 192 (mod 400), a leap year.
        // January: previously overflowed on year - 1.
        assert_eq!(
            seconds_of_week_from_calendar(i64::MIN, 1, 1, 0, 0, 0),
            Some(0.0)
        );
        assert_eq!(
            seconds_of_week_from_calendar(i64::MIN, 1, 1, 0, 0, 0),
            seconds_of_week_from_calendar(i64::MIN + 400, 1, 1, 0, 0, 0)
        );

        // February leap day accepted for i64::MIN:
        assert_eq!(
            seconds_of_week_from_calendar(i64::MIN, 2, 29, 0, 0, 0),
            Some(3.0 * SECONDS_PER_DAY)
        );

        // March:
        assert_eq!(
            seconds_of_week_from_calendar(i64::MIN, 3, 1, 0, 0, 0),
            Some(4.0 * SECONDS_PER_DAY)
        );
        assert_eq!(
            seconds_of_week_from_calendar(i64::MIN, 3, 1, 0, 0, 0),
            seconds_of_week_from_calendar(i64::MIN + 400, 3, 1, 0, 0, 0)
        );

        // i64::MAX = 9223372036854775807 = 207 (mod 400), non-leap year.
        // January:
        assert_eq!(
            seconds_of_week_from_calendar(i64::MAX, 1, 1, 0, 0, 0),
            Some(4.0 * SECONDS_PER_DAY)
        );
        assert_eq!(
            seconds_of_week_from_calendar(i64::MAX, 1, 1, 0, 0, 0),
            seconds_of_week_from_calendar(i64::MAX - 400, 1, 1, 0, 0, 0)
        );

        // February leap day refused for i64::MAX:
        assert_eq!(
            seconds_of_week_from_calendar(i64::MAX, 2, 29, 0, 0, 0),
            None
        );

        // March: previously overflowed y + y / 4.
        assert_eq!(
            seconds_of_week_from_calendar(i64::MAX, 3, 1, 0, 0, 0),
            Some(0.0)
        );
        assert_eq!(
            seconds_of_week_from_calendar(i64::MAX, 3, 1, 0, 0, 0),
            seconds_of_week_from_calendar(i64::MAX - 400, 3, 1, 0, 0, 0)
        );
    }
}
