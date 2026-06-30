//! Time-scale bridge for reduced-orbit fitting/evaluation.

use crate::astro::time::civil;
use crate::astro::time::model::TimeScale;
use crate::astro::time::scales::TimeScales;

/// A UTC calendar instant `(year, month, day, hour, minute, second)`, the form
/// the core [`TimeScales::from_utc`] consumes. The Elixir layer produces these
/// from each sample/query epoch; no `Instant`->`TimeScales` bridge exists in the
/// core crate, so the calendar tuple is carried explicitly to the boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalendarEpoch {
    /// Calendar year.
    pub year: i32,
    /// Calendar month, 1-12.
    pub month: i32,
    /// Calendar day of month, 1-31.
    pub day: i32,
    /// Hour of day, 0-23.
    pub hour: i32,
    /// Minute of hour, 0-59.
    pub minute: i32,
    /// Second of minute, fractional.
    pub second: f64,
}

impl CalendarEpoch {
    /// Construct a calendar epoch from its components.
    pub const fn new(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: f64) -> Self {
        Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
        }
    }

    /// Build the core [`TimeScales`] for this instant, interpreted in `scale`.
    ///
    /// Delegates to the canonical [`TimeScales::from_scale`]: non-UTC scales are
    /// converted to the UTC calendar label before the Skyfield split is built,
    /// so the Earth orientation used by the frame transforms is correct rather
    /// than offset by the scale's leap-second gap.
    pub(crate) fn time_scales(self, scale: TimeScale) -> TimeScales {
        TimeScales::from_scale(
            scale,
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
        )
        .expect("calendar epoch has a finite second")
    }
}

/// Seconds between two calendar epochs via their J2000-TT split day numbers.
pub(crate) fn dt_seconds(t0: &TimeScales, t: &TimeScales) -> f64 {
    civil::seconds_between_splits(t.jd_whole, t.tt_fraction, t0.jd_whole, t0.tt_fraction)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GLONASST = UTC(SU) + 3 h: a GLONASST calendar instant resolves to the
    /// same TT scales as the UTC instant three hours earlier (no leap term in
    /// the 3 h shift).
    #[test]
    fn glonasst_resolves_as_utc_plus_three_hours() {
        // 2020-06-15 03:00:00 GLONASST == 2020-06-15 00:00:00 UTC.
        let glo = CalendarEpoch::new(2020, 6, 15, 3, 0, 0.0).time_scales(TimeScale::Glonasst);
        let utc = CalendarEpoch::new(2020, 6, 15, 0, 0, 0.0).time_scales(TimeScale::Utc);
        assert_eq!(glo, utc);
    }

    /// QZSST is synchronous with GPST, so a QZSST calendar instant resolves to
    /// the same scales as the identically-labelled GPST instant.
    #[test]
    fn qzsst_resolves_identically_to_gpst() {
        let qzs = CalendarEpoch::new(2020, 6, 15, 12, 0, 0.0).time_scales(TimeScale::Qzsst);
        let gps = CalendarEpoch::new(2020, 6, 15, 12, 0, 0.0).time_scales(TimeScale::Gpst);
        assert_eq!(qzs, gps);
    }

    /// The 3 h GLONASST->UTC shift correctly crosses the day/year boundary
    /// around the 2017 leap. (Inputs are regular seconds; positive-leap `:60`
    /// labels in GLONASST are not a bridge input - the leap-aware reasoning lives
    /// in the offset helpers, which key off the UTC leap table.)
    #[test]
    fn glonasst_three_hour_shift_crosses_2017_boundary() {
        // 2017-01-01 03:00:00 GLONASST == 2017-01-01 00:00:00 UTC (post-leap).
        let post = CalendarEpoch::new(2017, 1, 1, 3, 0, 0.0).time_scales(TimeScale::Glonasst);
        let post_utc = CalendarEpoch::new(2017, 1, 1, 0, 0, 0.0).time_scales(TimeScale::Utc);
        assert_eq!(post, post_utc);

        // 2017-01-01 02:59:59 GLONASST == 2016-12-31 23:59:59 UTC (pre-leap).
        let pre = CalendarEpoch::new(2017, 1, 1, 2, 59, 59.0).time_scales(TimeScale::Glonasst);
        let pre_utc = CalendarEpoch::new(2016, 12, 31, 23, 59, 59.0).time_scales(TimeScale::Utc);
        assert_eq!(pre, pre_utc);
    }
}
