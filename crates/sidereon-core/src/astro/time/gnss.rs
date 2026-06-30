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
        TimeScale::Glonasst | TimeScale::Utc | TimeScale::Tai | TimeScale::Tt | TimeScale::Tdb => {
            None
        }
    }
}

/// GNSS week number for a calendar date in `system`'s week numbering, or `None`
/// for a date before the epoch or a scale without GNSS weeks.
#[must_use]
pub fn week_from_calendar(system: TimeScale, year: i64, month: i64, day: i64) -> Option<u32> {
    let epoch_jdn = week_epoch_julian_day_number(system)?;
    let elapsed_days =
        julian_day_number(year as i32, month as i32, day as i32).checked_sub(epoch_jdn)?;
    if elapsed_days < 0 {
        return None;
    }
    u32::try_from(elapsed_days / 7).ok()
}

/// Seconds-of-week of a calendar epoch in its own system time, with the GNSS
/// Sunday-00:00 origin. Sakamoto's day-of-week gives 0 = Sunday.
#[must_use]
pub fn seconds_of_week_from_calendar(
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    second: i64,
) -> f64 {
    const T: [i64; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if month < 3 { year - 1 } else { year };
    let dow = (y + y / 4 - y / 100 + y / 400 + T[(month - 1) as usize] + day).rem_euclid(7);
    dow as f64 * SECONDS_PER_DAY + (hour * 3600 + minute * 60 + second) as f64
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
    }

    #[test]
    fn seconds_of_week_sunday_origin() {
        // 1980-01-06 was a Sunday -> dow 0, sow = time of day.
        assert_eq!(seconds_of_week_from_calendar(1980, 1, 6, 0, 0, 0), 0.0);
        assert_eq!(seconds_of_week_from_calendar(1980, 1, 6, 1, 2, 3), 3723.0);
        // The following Monday is dow 1.
        assert_eq!(
            seconds_of_week_from_calendar(1980, 1, 7, 0, 0, 0),
            SECONDS_PER_DAY
        );
    }
}
