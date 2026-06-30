//! Public time model type family.
//!
//! A bare `Epoch` is ambiguous (element epoch vs observation epoch vs product
//! epoch), so the public surface is a concrete family of types that always name
//! their scale and use a precision-preserving representation:
//!
//! - [`TimeScale`] - the named time scale of an instant.
//! - [`Instant`] - a scale + a split-Julian-date / integer-nanosecond repr.
//! - [`Duration`] - an integer-nanosecond elapsed interval.
//! - [`JulianDateSplit`] - the two-part (whole + fraction) Julian date used to
//!   avoid catastrophic cancellation, matching the Skyfield split convention.
//! - [`GnssWeekTow`] - a GNSS week number + time-of-week with rollover handling.
//!
//! These are representation/value types only. The parity-critical conversion
//! numerics live in [`crate::astro::time::scales`]; this module deliberately holds no
//! transcendental math so it cannot perturb the 0-ULP contract.

pub use crate::astro::constants::time::SECONDS_PER_WEEK;

/// Error returned when constructing public time model values from invalid input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TimeModelError {
    /// A public constructor received a non-finite or out-of-domain input.
    #[error("invalid time model {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

fn invalid_input(field: &'static str, reason: &'static str) -> TimeModelError {
    TimeModelError::InvalidInput { field, reason }
}

/// Named time scales supported by the time model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeScale {
    /// Coordinated Universal Time.
    Utc,
    /// International Atomic Time.
    Tai,
    /// Terrestrial Time.
    Tt,
    /// Barycentric Dynamical Time.
    Tdb,
    /// GPS time.
    Gpst,
    /// Galileo System Time.
    Gst,
    /// BeiDou Time.
    Bdt,
    /// GLONASS system time. UTC(SU)-based: GLONASST = UTC(SU) + 3 h, so it
    /// carries UTC leap seconds and its offset to the atomic scales is
    /// epoch-dependent (ICD GLONASS, Edition 5.1, 2008, sec. 3.3.3).
    Glonasst,
    /// QZSS system time. Steered to be synchronous with GPST (IS-QZSS-PNT,
    /// sec. 3.2.2), so the nominal QZSST-GPST offset is zero.
    Qzsst,
}

impl TimeScale {
    /// Short uppercase identifier (`"UTC"`, `"TAI"`, ...).
    pub fn abbrev(self) -> &'static str {
        match self {
            TimeScale::Utc => "UTC",
            TimeScale::Tai => "TAI",
            TimeScale::Tt => "TT",
            TimeScale::Tdb => "TDB",
            TimeScale::Gpst => "GPST",
            TimeScale::Gst => "GST",
            TimeScale::Bdt => "BDT",
            TimeScale::Glonasst => "GLONASST",
            TimeScale::Qzsst => "QZSST",
        }
    }
}

/// Two-part Julian date (whole day boundary + day fraction).
///
/// Carrying the integer day separately from the fraction preserves
/// sub-microsecond precision across the full Julian-date range, and matches the
/// Skyfield split that [`crate::astro::time::scales::TimeScales`] produces.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct JulianDateSplit {
    /// Integer day boundary (typically `*.0` or `*.5`).
    pub jd_whole: f64,
    /// Residual day fraction relative to `jd_whole`.
    pub fraction: f64,
}

impl JulianDateSplit {
    /// Construct a split Julian date.
    pub fn new(jd_whole: f64, fraction: f64) -> Result<Self, TimeModelError> {
        if !jd_whole.is_finite() {
            return Err(invalid_input("jd_whole", "must be finite"));
        }
        if !fraction.is_finite() {
            return Err(invalid_input("fraction", "must be finite"));
        }
        if !(-1.0..=1.0).contains(&fraction) {
            return Err(invalid_input("fraction", "must be within one residual day"));
        }
        Ok(Self { jd_whole, fraction })
    }

    /// Recombine into a single `f64` Julian date.
    ///
    /// Note: recombination is itself a float operation and is NOT guaranteed to
    /// be 0-ULP against a reference that consumes the split form directly; keep
    /// the split form when feeding a parity-matched recipe.
    pub fn to_jd(self) -> f64 {
        self.jd_whole + self.fraction
    }
}

/// Internal representation backing an [`Instant`].
///
/// Two reprs are offered to avoid precision loss for different consumers:
/// integer nanoseconds for exact arithmetic (hifitime-style), and the split
/// Julian date for the astronomy/Skyfield path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InstantRepr {
    /// Integer nanoseconds since an implied scale epoch (exact arithmetic).
    Nanos(i128),
    /// Two-part Julian date in the instant's own scale.
    JulianDate(JulianDateSplit),
}

/// A point in time, always tagged with its [`TimeScale`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Instant {
    /// The time scale this instant is expressed in.
    pub scale: TimeScale,
    /// The precision-preserving representation.
    pub repr: InstantRepr,
}

impl Instant {
    /// An instant from a split Julian date in the given scale.
    pub fn from_julian_date(scale: TimeScale, jd: JulianDateSplit) -> Self {
        Self {
            scale,
            repr: InstantRepr::JulianDate(jd),
        }
    }

    /// An instant from integer nanoseconds in the given scale.
    pub fn from_nanos(scale: TimeScale, nanos: i128) -> Self {
        Self {
            scale,
            repr: InstantRepr::Nanos(nanos),
        }
    }

    /// The split Julian date, if this instant is stored in that form.
    pub fn julian_date(&self) -> Option<JulianDateSplit> {
        match self.repr {
            InstantRepr::JulianDate(jd) => Some(jd),
            InstantRepr::Nanos(_) => None,
        }
    }
}

/// An elapsed interval, stored as exact integer nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Duration {
    /// Signed elapsed nanoseconds.
    pub nanos: i128,
}

impl Duration {
    /// Zero duration.
    pub const ZERO: Duration = Duration { nanos: 0 };

    /// Construct from integer nanoseconds.
    pub fn from_nanos(nanos: i128) -> Self {
        Self { nanos }
    }

    /// Construct from seconds. Sub-nanosecond input is truncated toward zero.
    pub fn from_seconds(seconds: f64) -> Result<Self, TimeModelError> {
        if !seconds.is_finite() {
            return Err(invalid_input("seconds", "must be finite"));
        }
        let nanos = seconds * 1e9;
        if !nanos.is_finite() || nanos <= i128::MIN as f64 || nanos >= i128::MAX as f64 {
            return Err(invalid_input(
                "seconds",
                "must convert to an i128 nanosecond count",
            ));
        }
        Ok(Self {
            nanos: nanos as i128,
        })
    }

    /// Convert to floating-point seconds.
    pub fn as_seconds(self) -> f64 {
        self.nanos as f64 / 1e9
    }
}

/// A GNSS week number + time-of-week, tagged by constellation.
///
/// `week` is the constellation's native (rolled-over) week count; `tow_s` is
/// seconds into that week in `[0, 604800)`. Rollover handling is provided by
/// [`GnssWeekTow::normalized`] and [`GnssWeekTow::unrolled_week`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GnssWeekTow {
    /// Which constellation's week/TOW convention this uses.
    pub system: TimeScale,
    /// Week number (constellation-native, may have rolled over).
    pub week: u32,
    /// Time of week in seconds, nominally `[0, 604800)`.
    pub tow_s: f64,
}

impl GnssWeekTow {
    /// Construct a week/TOW value.
    pub fn new(system: TimeScale, week: u32, tow_s: f64) -> Result<Self, TimeModelError> {
        if !tow_s.is_finite() {
            return Err(invalid_input("tow_s", "must be finite"));
        }
        Ok(Self {
            system,
            week,
            tow_s,
        })
    }

    /// Normalize so `tow_s` lands in `[0, 604800)`, carrying whole weeks into
    /// `week`. Negative `tow_s` borrows from the week count.
    pub fn normalized(self) -> Result<Self, TimeModelError> {
        if !self.tow_s.is_finite() {
            return Err(invalid_input("tow_s", "must be finite"));
        }
        let mut week = self.week as i64;
        let mut tow = self.tow_s;
        let weeks_carry = (tow / SECONDS_PER_WEEK).floor();
        if !weeks_carry.is_finite()
            || weeks_carry <= i64::MIN as f64
            || weeks_carry >= i64::MAX as f64
        {
            return Err(invalid_input("tow_s", "week carry is out of range"));
        }
        week = week
            .checked_add(weeks_carry as i64)
            .ok_or_else(|| invalid_input("tow_s", "week carry is out of range"))?;
        tow -= weeks_carry * SECONDS_PER_WEEK;
        if week < 0 {
            week = 0;
            tow = 0.0;
        }
        if week > u32::MAX as i64 {
            return Err(invalid_input("tow_s", "normalized week is out of range"));
        }
        if !tow.is_finite() {
            return Err(invalid_input("tow_s", "normalized TOW must be finite"));
        }
        Ok(Self {
            system: self.system,
            week: week as u32,
            tow_s: tow,
        })
    }

    /// Apply a 1024-week rollover count to recover the continuous week number
    /// (GPS legacy 10-bit week). `rollovers` is the number of completed
    /// 1024-week eras since the system's epoch.
    pub fn unrolled_week(self, rollovers: u32) -> Result<u32, TimeModelError> {
        let rollover_weeks = rollovers
            .checked_mul(1024)
            .ok_or_else(|| invalid_input("rollovers", "unrolled week is out of range"))?;
        self.week
            .checked_add(rollover_weeks)
            .ok_or_else(|| invalid_input("rollovers", "unrolled week is out of range"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_julian_date_rejects_nonfinite_parts() {
        assert!(JulianDateSplit::new(f64::NAN, 0.0).is_err());
        assert!(JulianDateSplit::new(f64::INFINITY, 0.0).is_err());
        assert!(JulianDateSplit::new(2_451_545.0, f64::NAN).is_err());
        assert!(JulianDateSplit::new(2_451_545.0, f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn split_julian_date_rejects_out_of_range_fraction() {
        assert!(JulianDateSplit::new(2_451_545.0, 1.0 + f64::EPSILON).is_err());
        assert!(JulianDateSplit::new(2_451_545.0, -1.0 - f64::EPSILON).is_err());
    }

    #[test]
    fn split_julian_date_valid_parts_are_unchanged() {
        let jd = JulianDateSplit::new(2_451_545.0, -0.25).expect("valid split Julian date");
        assert_eq!(jd.jd_whole, 2_451_545.0);
        assert_eq!(jd.fraction, -0.25);
        assert_eq!(jd.to_jd(), 2_451_544.75);
    }

    #[test]
    fn duration_from_seconds_rejects_nonfinite_seconds() {
        assert!(Duration::from_seconds(f64::NAN).is_err());
        assert!(Duration::from_seconds(f64::INFINITY).is_err());
        assert!(Duration::from_seconds(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn duration_from_seconds_rejects_unrepresentable_nanoseconds() {
        assert!(Duration::from_seconds(f64::MAX).is_err());
        assert!(Duration::from_seconds(-f64::MAX).is_err());
    }

    #[test]
    fn duration_from_seconds_valid_input_is_truncated_toward_zero() {
        assert_eq!(
            Duration::from_seconds(1.234_567_890_9)
                .expect("valid duration")
                .nanos,
            1_234_567_890
        );
        assert_eq!(
            Duration::from_seconds(-1.234_567_890_9)
                .expect("valid duration")
                .nanos,
            -1_234_567_890
        );
    }

    #[test]
    fn gnss_week_tow_rejects_nonfinite_tow() {
        assert!(GnssWeekTow::new(TimeScale::Gpst, 100, f64::NAN).is_err());
        assert!(GnssWeekTow::new(TimeScale::Gpst, 100, f64::INFINITY).is_err());
        assert!(GnssWeekTow::new(TimeScale::Gpst, 100, f64::NEG_INFINITY).is_err());
        assert!(GnssWeekTow {
            system: TimeScale::Gpst,
            week: 100,
            tow_s: f64::NAN,
        }
        .normalized()
        .is_err());
    }

    #[test]
    fn gnss_week_tow_rejects_out_of_range_week_carry() {
        let err = GnssWeekTow::new(TimeScale::Gpst, u32::MAX, SECONDS_PER_WEEK)
            .expect("finite TOW")
            .normalized();
        assert!(err.is_err());
    }

    #[test]
    fn gnss_week_tow_valid_rollover_is_unchanged() {
        let wt = GnssWeekTow::new(TimeScale::Gpst, 100, SECONDS_PER_WEEK + 5.0)
            .expect("valid week/TOW")
            .normalized()
            .expect("valid normalized week/TOW");
        assert_eq!(wt.week, 101);
        assert_eq!(wt.tow_s, 5.0);
    }

    #[test]
    fn gnss_week_tow_unrolled_week_rejects_overflow() {
        let wt = GnssWeekTow::new(TimeScale::Gpst, u32::MAX, 0.0).expect("valid week/TOW");
        let result = std::panic::catch_unwind(|| wt.unrolled_week(1));
        assert!(result.is_ok(), "overflowing unrolled week must not panic");
        assert_eq!(
            result.expect("overflowing unrolled week should not unwind"),
            Err(TimeModelError::InvalidInput {
                field: "rollovers",
                reason: "unrolled week is out of range",
            })
        );
    }

    #[test]
    fn gnss_week_tow_unrolled_week_valid_input_is_unchanged() {
        let wt = GnssWeekTow::new(TimeScale::Gpst, 10, 0.0).expect("valid week/TOW");
        assert_eq!(wt.unrolled_week(2).expect("valid unrolled week"), 10 + 2048);
    }
}
