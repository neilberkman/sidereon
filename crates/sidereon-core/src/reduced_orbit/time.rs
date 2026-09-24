//! Time-scale bridge for reduced-orbit fitting/evaluation.

use crate::astro::time::civil;
use crate::astro::time::exact::{ExactEpoch, ExactSeconds};
use crate::astro::time::model::TimeScale;
use crate::astro::time::scales::{label_tai_minus_utc, TimeScales};

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

/// A calendar epoch labelled in `scale` as exact TT seconds, up to an offset
/// that depends on `scale` alone, so the difference of two epochs in one
/// scale is their exact TT interval.
///
/// A TAI, TT, GPST, Galileo, BeiDou or QZSS label is a fixed offset from TT,
/// so the label itself, read exactly ([`ExactEpoch::from_civil`]), is used. A
/// UTC or GLONASST label adds the TAI - UTC that [`TimeScales::from_scale`]
/// applies to it, so an interval across a leap second counts the leap second.
/// A TCG, TDB or TCB label, whose offset from TT varies, takes the exact TT
/// that `ts`, its [`TimeScales`], holds.
pub(crate) fn exact_tt_seconds(
    epoch: CalendarEpoch,
    ts: &TimeScales,
    scale: TimeScale,
) -> ExactSeconds {
    let split_tt = || {
        civil::exact_seconds_of_split_parts(ts.jd_whole, ts.tt_fraction)
            .expect("time scales have finite parts")
    };
    let label = || {
        ExactEpoch::from_civil(
            epoch.year,
            epoch.month,
            epoch.day,
            epoch.hour,
            epoch.minute,
            epoch.second,
        )
    };
    match scale {
        TimeScale::Tcg | TimeScale::Tdb | TimeScale::Tcb => split_tt(),
        TimeScale::Utc | TimeScale::Glonasst => {
            let leap = label_tai_minus_utc(
                scale,
                epoch.year,
                epoch.month,
                epoch.day,
                epoch.hour,
                epoch.minute,
                epoch.second,
            )
            .and_then(ExactSeconds::from_f64);
            match (label(), leap) {
                (Some(label), Some(leap)) => label.exact_seconds().add(&leap),
                _ => split_tt(),
            }
        }
        TimeScale::Tai
        | TimeScale::Tt
        | TimeScale::Gpst
        | TimeScale::Gst
        | TimeScale::Bdt
        | TimeScale::Qzsst => label().map_or_else(split_tt, ExactEpoch::exact_seconds),
    }
}

/// TT seconds from `t0` to `t`, the exact interval rounded once.
pub(crate) fn dt_seconds(t0: &ExactSeconds, t: &ExactSeconds) -> f64 {
    t.sub(t0).to_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dt(t0: CalendarEpoch, t: CalendarEpoch, scale: TimeScale) -> f64 {
        dt_seconds(
            &exact_tt_seconds(t0, &t0.time_scales(scale), scale),
            &exact_tt_seconds(t, &t.time_scales(scale), scale),
        )
    }

    #[test]
    fn intervals_are_the_exact_tt_difference_of_the_labels() {
        // Whole and decimal labels on an atomic scale: the label difference.
        let base = CalendarEpoch::new(2020, 6, 24, 0, 0, 0.0);
        assert_eq!(
            dt(
                base,
                CalendarEpoch::new(2020, 6, 24, 1, 0, 0.0),
                TimeScale::Gpst
            ),
            3_600.0
        );
        assert_eq!(
            dt(
                CalendarEpoch::new(2020, 6, 24, 0, 0, 0.2),
                CalendarEpoch::new(2020, 6, 24, 0, 0, 0.3),
                TimeScale::Gpst
            ),
            0.1
        );
        // UTC across the 2016 leap second counts it, and the leap-second
        // label sits one second after 23:59:59.
        let before = CalendarEpoch::new(2016, 12, 31, 23, 59, 59.5);
        assert_eq!(
            dt(
                before,
                CalendarEpoch::new(2017, 1, 1, 0, 0, 0.5),
                TimeScale::Utc
            ),
            2.0
        );
        assert_eq!(
            dt(
                before,
                CalendarEpoch::new(2016, 12, 31, 23, 59, 60.5),
                TimeScale::Utc
            ),
            1.0
        );
        // GLONASST is UTC three hours ahead: the same leap second, at 03:00.
        assert_eq!(
            dt(
                CalendarEpoch::new(2017, 1, 1, 2, 59, 59.5),
                CalendarEpoch::new(2017, 1, 1, 3, 0, 0.5),
                TimeScale::Glonasst
            ),
            2.0
        );
        // A TDB label takes the exact difference of its TT splits.
        let t0 = CalendarEpoch::new(2020, 6, 24, 0, 0, 0.0).time_scales(TimeScale::Tdb);
        let t1 = CalendarEpoch::new(2020, 6, 24, 1, 0, 0.0).time_scales(TimeScale::Tdb);
        let exact = civil::exact_seconds_of_split_parts(t1.jd_whole, t1.tt_fraction)
            .unwrap()
            .sub(&civil::exact_seconds_of_split_parts(t0.jd_whole, t0.tt_fraction).unwrap())
            .to_f64();
        assert_eq!(
            dt(
                CalendarEpoch::new(2020, 6, 24, 0, 0, 0.0),
                CalendarEpoch::new(2020, 6, 24, 1, 0, 0.0),
                TimeScale::Tdb
            ),
            exact
        );
        assert!((exact - 3_600.0).abs() < 1.0e-6);
    }

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
