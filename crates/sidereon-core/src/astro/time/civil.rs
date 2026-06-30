//! Civil-calendar conversions the GNSS bindings consume directly.
//!
//! These are the single source for the four civil-time conversions every
//! language binding previously reimplemented (the Elixir `Sidereon.GNSS.Time`
//! helpers, and their Python/C/WASM equivalents): split Julian date, continuous
//! seconds since J2000, second-of-day, and fractional day-of-year. A thin
//! binding marshals its native datetime into the `(year, month, day, hour,
//! minute, second)` civil fields and calls these, instead of carrying its own
//! calendar arithmetic.
//!
//! No leap-second shifting is applied: the epoch stays in whatever time scale
//! the caller supplied it in (typically GPS time for the broadcast correction
//! models). The full UTC -> TAI/TT/TDB/UT1 leap-aware conversion is the separate
//! [`super::scales::TimeScales::from_utc`] path.
//!
//! The integer day-number step delegates to [`super::scales::julian_day_number`]
//! (the existing Fliegel-style core primitive); the remaining arithmetic mirrors
//! the binding reference operation order exactly, so a binding that switches to
//! these helpers reproduces its previous values bit-for-bit.

use super::model::{Instant, InstantRepr, JulianDateSplit, TimeModelError, TimeScale};
use super::scales::julian_day_number;
use crate::astro::constants::time::SECONDS_PER_DAY_I64;
use crate::constants::{J2000_JD, SECONDS_PER_DAY};

/// Julian Date of the Modified Julian Date origin (`MJD = JD - 2_400_000.5`).
pub const MJD_JD_OFFSET: f64 = 2_400_000.5;

/// Integer Julian Day Number of the J2000 calendar day (2000-01-01).
pub const J2000_JULIAN_DAY_NUMBER: i64 = 2_451_545;

/// Seconds from civil midnight of the J2000 day to the J2000 epoch (noon).
pub const J2000_NOON_OFFSET_S: i64 = 43_200;

/// Split Julian date `(jd_whole, fraction)` for a civil instant.
///
/// `jd_whole` is the `*.5` civil-midnight boundary of the day (the integer
/// `julian_day_number` minus `0.5`) and `fraction` is the within-day part, the
/// convention the SP3 reader and the RTK epoch axis use. The instant is
/// `jd_whole + fraction` in the caller's own time scale; no leap second is
/// applied.
#[must_use]
pub fn split_julian_date(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: f64,
) -> (f64, f64) {
    let jd_whole = julian_day_number(year, month, day) as f64 - 0.5;
    let day_seconds = hour as f64 * 3600.0 + minute as f64 * 60.0 + second;
    let fraction = day_seconds / SECONDS_PER_DAY;
    (jd_whole, fraction)
}

impl Instant {
    /// A UTC [`Instant`] from civil-calendar fields.
    ///
    /// Marshals `(year, month, day, hour, minute, second)` through
    /// [`split_julian_date`] into the split-Julian-date representation, tagged
    /// [`TimeScale::Utc`]. This is the public entry a thin binding calls to build
    /// the `epoch` argument for the ionosphere/troposphere dispatchers (for
    /// example [`crate::atmosphere::ionosphere::ionosphere_delay`] and
    /// [`crate::atmosphere::ionosphere::klobuchar`]) without reaching into the
    /// crate-private split internals; it produces the same [`Instant`] as the
    /// open-coded `split_julian_date` + [`Instant::from_julian_date`] path.
    ///
    /// No leap second is applied: the instant carries the civil fields as given,
    /// the same no-leap contract as [`split_julian_date`]. An out-of-day clock
    /// field whose residual leaves the one-day fraction window is rejected by
    /// [`JulianDateSplit::new`].
    pub fn from_utc_civil(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
    ) -> Result<Self, TimeModelError> {
        let (jd_whole, fraction) = split_julian_date(year, month, day, hour, minute, second);
        Ok(Self::from_julian_date(
            TimeScale::Utc,
            JulianDateSplit::new(jd_whole, fraction)?,
        ))
    }
}

/// Continuous seconds since the J2000 epoch (JD 2451545.0) for a civil instant.
///
/// This is the value the SPP solve consumes as
/// [`crate::positioning::SolveInputs::t_rx_j2000_s`]. The whole-second part is
/// formed in integer arithmetic (so a whole-second epoch is exact) and the
/// sub-second remainder is added back, matching the binding reference order. The
/// epoch stays in the caller's time scale; no leap second is applied.
#[must_use]
pub fn j2000_seconds(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: f64) -> f64 {
    // A non-finite second has no integer whole part; propagate NaN rather than
    // letting the saturating float-to-int cast below fabricate a finite result.
    if !second.is_finite() {
        return f64::NAN;
    }
    let whole_second = second.trunc();
    let fraction = second - whole_second;
    // J2000 (JD 2451545.0) sits at this day's noon plus an integer day count, so
    // the day's noon is `(jdn - 2451545)` whole days from J2000; civil midnight
    // is 43200 s earlier. All integer, then the within-day clock fields are added.
    // Saturating arithmetic: for any real civil epoch these never saturate (so the
    // whole-second result stays exact), and an absurd out-of-range second can no
    // longer overflow-panic.
    let noon_seconds_from_j2000 = (julian_day_number(year, month, day) - J2000_JD as i64)
        .saturating_mul(SECONDS_PER_DAY as i64);
    let day_seconds = (i64::from(hour) * 3600)
        .saturating_add(i64::from(minute) * 60)
        .saturating_add(whole_second as i64);
    (noon_seconds_from_j2000
        .saturating_sub(43_200)
        .saturating_add(day_seconds)) as f64
        + fraction
}

/// Second-of-day in `[0, 86400)` formed from the clock fields.
///
/// This is the value the SPP solve consumes as
/// [`crate::positioning::SolveInputs::t_rx_second_of_day_s`] (the Klobuchar
/// diurnal argument). Formed straight from `hour`, `minute`, and `second` so it
/// is exact, with no split-Julian-date round trip.
#[must_use]
pub fn second_of_day(hour: i32, minute: i32, second: f64) -> f64 {
    hour as f64 * 3600.0 + minute as f64 * 60.0 + second
}

/// Fractional day-of-year for a civil instant; January 1 00:00 is `1.0`.
///
/// This is the value the SPP solve consumes as
/// [`crate::positioning::SolveInputs::day_of_year`] (the Niell troposphere
/// seasonal argument). The integer day-of-year is the day-number difference from
/// January 1 of the same year (exact integer arithmetic, leap-year independent),
/// plus the within-day fraction.
#[must_use]
pub fn day_of_year(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: f64) -> f64 {
    let integer_day_of_year =
        (julian_day_number(year, month, day) - julian_day_number(year, 1, 1) + 1) as f64;
    let sod = second_of_day(hour, minute, second);
    integer_day_of_year + sod / SECONDS_PER_DAY
}

/// Integer day-of-year (January 1 is `1`) for a civil date.
///
/// The integer companion of [`day_of_year`]: it is the same day-number
/// difference from January 1 (exact integer arithmetic, leap-year independent),
/// without the within-day fraction. Replaces the cumulative-month-table copies
/// in the IONEX and troposphere readers (verified bit-identical to that table).
#[must_use]
pub fn day_of_year_int(year: i32, month: i32, day: i32) -> i64 {
    julian_day_number(year, month, day) - julian_day_number(year, 1, 1) + 1
}

/// Civil `(year, month, day)` from an integer Julian Day Number.
///
/// The Fliegel-Van Flandern inverse, the single home for the calendar-from-JDN
/// step the SP3, IONEX, RINEX clock/nav, TCA, troposphere, and reduced-orbit
/// writers each carried a private copy of. The two algebraic forms found in the
/// tree (the `+68569` and `+32044` variants) are bit-identical across the
/// non-negative JDN range; this is the `+68569` form.
///
/// Domain: JDN `>= 0` (every civil date from -4712 onward, i.e. every epoch any
/// real ephemeris can occupy, since [`super::scales::julian_day_number`] never
/// emits a negative JDN). Exact inverse of `julian_day_number` across that
/// range. Negative JDN is non-physical and not supported: the Fliegel inverse's
/// truncating integer division would diverge from the floor convention there.
#[must_use]
pub fn civil_from_julian_day_number(jdn: i64) -> (i64, i64, i64) {
    let l = jdn + 68_569;
    let n = 4 * l / 146_097;
    let l = l - (146_097 * n + 3) / 4;
    let i = 4_000 * (l + 1) / 1_461_001;
    let l = l - 1_461 * i / 4 + 31;
    let j = 80 * l / 2_447;
    let day = l - 2_447 * j / 80;
    let l = j / 11;
    let month = j + 2 - 12 * l;
    let year = 100 * (n - 49) + i + l;
    (year, month, day)
}

/// Civil `(year, month, day, hour, minute, second)` from continuous integer
/// seconds since the J2000 epoch.
///
/// Exact inverse of [`j2000_seconds`] for a whole-second epoch: the J2000 noon
/// origin is shifted to civil midnight, the calendar date follows from
/// [`civil_from_julian_day_number`], and the within-day clock fields are read
/// from the floored second-of-day. No leap second is applied (the epoch stays in
/// the caller's own time scale, the same no-leap contract as the forward
/// conversions).
#[must_use]
pub fn civil_from_j2000_seconds(seconds: i64) -> (i64, i64, i64, i64, i64, i64) {
    let from_midnight = seconds + J2000_NOON_OFFSET_S;
    let day_index = from_midnight.div_euclid(SECONDS_PER_DAY_I64);
    let second_of_day = from_midnight.rem_euclid(SECONDS_PER_DAY_I64);
    let (year, month, day) = civil_from_julian_day_number(day_index + J2000_JULIAN_DAY_NUMBER);
    let hour = second_of_day / 3_600;
    let minute = (second_of_day % 3_600) / 60;
    let second = second_of_day % 60;
    (year, month, day, hour, minute, second)
}

/// Modified Julian Date from a Julian Date (`MJD = JD - 2_400_000.5`).
#[must_use]
pub fn mjd_from_jd(jd: f64) -> f64 {
    jd - MJD_JD_OFFSET
}

/// Continuous seconds since J2000 from a split Julian date `(jd_whole,
/// fraction)`.
///
/// The pure arithmetic core behind [`crate::observables::j2000_seconds_from_split`]
/// (which keeps the public-facing finiteness validation). Whole-day and
/// fractional parts are each scaled to seconds and summed, matching the split
/// the SP3 reader and RTK epoch axis carry.
#[must_use]
pub fn j2000_seconds_from_split(jd_whole: f64, fraction: f64) -> f64 {
    (jd_whole - J2000_JD) * SECONDS_PER_DAY + fraction * SECONDS_PER_DAY
}

/// Elapsed seconds between two split Julian dates `later - earlier`.
///
/// The whole-day and fractional differences are summed first and scaled once
/// (`(dwhole + dfrac) * 86400`), the policy the RINEX clock interpolation and
/// reduced-orbit fit duration share. (The J2000-seconds conversion
/// [`j2000_seconds_from_split`] scales each part separately; that ordering is
/// kept distinct because the two are not bit-identical in the last place.)
#[must_use]
pub fn seconds_between_splits(
    later_whole: f64,
    later_fraction: f64,
    earlier_whole: f64,
    earlier_fraction: f64,
) -> f64 {
    ((later_whole - earlier_whole) + (later_fraction - earlier_fraction)) * SECONDS_PER_DAY
}

/// Split Julian date `(jd_whole, fraction)` from continuous integer seconds
/// since the J2000 epoch.
///
/// The inverse of the J2000-seconds direction expressed in the split form the
/// SP3 reader and RTK epoch axis carry: the day count places the integer JD
/// boundary and the residual within-day seconds become the fraction. `jd_whole`
/// is a `*.0` boundary here (J2000 noon is JD 2451545.0), matching the IONEX
/// reader's epoch reconstruction. No leap second is applied.
#[must_use]
pub fn split_julian_date_from_j2000_seconds(seconds: i64) -> (f64, f64) {
    let days = seconds.div_euclid(SECONDS_PER_DAY_I64);
    let rem_s = seconds.rem_euclid(SECONDS_PER_DAY_I64);
    (J2000_JD + days as f64, rem_s as f64 / SECONDS_PER_DAY)
}

/// Civil `(year, month, day, hour, minute, second)` from a split Julian date.
///
/// `jd_whole` is the `*.5` civil-midnight boundary and `fraction` is the
/// within-day part (the [`split_julian_date`] convention). The integer Julian
/// Day Number is recovered from the boundary, the calendar date from
/// [`civil_from_julian_day_number`], and the clock fields from the floored
/// within-day seconds (so the fractional second is carried in `second`). This is
/// the single home for the floor-based split-to-civil decomposition the SGP4 TCA
/// path and reduced-orbit fit each open-coded. No leap second is applied.
#[must_use]
pub fn civil_from_split_julian_date(
    jd_whole: f64,
    fraction: f64,
) -> (i64, i64, i64, i64, i64, f64) {
    // Precondition: `jd_whole` is the `*.5` civil-midnight boundary (the
    // `split_julian_date` convention). Do NOT pass the `*.0` noon boundary
    // produced by `split_julian_date_from_j2000_seconds` (the IONEX reader's
    // convention): `(jd_whole + 0.5).round()` would land on the wrong day. The
    // guard catches that accidental composition in debug/test builds.
    debug_assert!(
        (jd_whole.fract().abs() - 0.5).abs() < 1e-6,
        "civil_from_split_julian_date expects a *.5 civil-midnight jd_whole, not a *.0 noon boundary"
    );
    let jdn = (jd_whole + 0.5).round() as i64;
    let (year, month, day) = civil_from_julian_day_number(jdn);
    let seconds_of_day = fraction * SECONDS_PER_DAY;
    let hour = (seconds_of_day / 3600.0).floor() as i64;
    let minute = ((seconds_of_day - hour as f64 * 3600.0) / 60.0).floor() as i64;
    let second = seconds_of_day - hour as f64 * 3600.0 - minute as f64 * 60.0;
    (year, month, day, hour, minute, second)
}

/// Advance a split Julian date `(jd_whole, fraction)` by `seconds`, renormalizing
/// the carry back onto the whole-day boundary.
///
/// The seconds are scaled to a day fraction, added to the residual, and the
/// integer-day carry floored back into `jd_whole`. This is the policy the SGP4
/// TCA frame-derivative step uses; the arithmetic order is preserved so the
/// stepped epoch is bit-identical to the open-coded form.
#[must_use]
pub fn split_julian_date_add_seconds(jd_whole: f64, fraction: f64, seconds: f64) -> (f64, f64) {
    let mut whole = jd_whole;
    let mut fraction = fraction + seconds / SECONDS_PER_DAY;
    let carry = fraction.floor();
    whole += carry;
    fraction -= carry;
    (whole, fraction)
}

/// Single-`f64` Julian date carried by an [`Instant`], in the instant's own
/// scale.
///
/// A split-Julian-date instant recombines its two parts; an integer-nanosecond
/// instant counts nanoseconds from the J2000 Julian-date origin (the
/// IONEX/SP3/RINEX-clock nanosecond convention). This is the single home for the
/// instant-to-Julian-date reduction the troposphere and IONEX seasonal terms
/// each open-coded.
#[must_use]
pub fn julian_date_from_instant(epoch: Instant) -> f64 {
    match epoch.repr {
        InstantRepr::JulianDate(split) => split.jd_whole + split.fraction,
        InstantRepr::Nanos(nanos) => (nanos as f64) / 1.0e9 / SECONDS_PER_DAY + J2000_JD,
    }
}

/// Fractional day-of-year carried by an [`Instant`]; January 1 00:00 is `1.0`.
///
/// The noon Julian-date origin is shifted to midnight, split into an integer day
/// and the within-day fraction, and the integer day-of-year recovered from the
/// calendar date. This is the single home for the seasonal day-of-year argument
/// the Niell troposphere and IONEX diurnal terms each open-coded.
#[must_use]
pub fn fractional_day_of_year_from_instant(epoch: Instant) -> f64 {
    let jd = julian_date_from_instant(epoch);
    let jd_midnight = jd + 0.5;
    let day_floor = jd_midnight.floor();
    let day_fraction = jd_midnight - day_floor;

    let (year, month, day) = civil_from_julian_day_number(day_floor as i64);
    let doy_integer = day_of_year_int(year as i32, month as i32, day as i32);
    doy_integer as f64 + day_fraction
}

/// Second-of-day in `[0, 86400)` carried by an [`Instant`], in its own scale.
///
/// A split-Julian-date instant shifts the noon day origin to midnight and keeps
/// the within-day part; an integer-nanosecond instant reduces by the
/// seconds-per-day modulus (exact). This is the single home for the diurnal
/// second-of-day argument the IONEX Klobuchar term open-coded.
#[must_use]
pub fn second_of_day_from_instant(epoch: Instant) -> f64 {
    match epoch.repr {
        InstantRepr::JulianDate(jd) => {
            let from_midnight = jd.jd_whole + 0.5 + jd.fraction;
            let day_fraction = from_midnight - from_midnight.floor();
            day_fraction * SECONDS_PER_DAY
        }
        InstantRepr::Nanos(nanos) => {
            let ns_per_day: i128 = 86_400 * 1_000_000_000;
            let mut rem = nanos % ns_per_day;
            if rem < 0 {
                rem += ns_per_day;
            }
            rem as f64 / 1.0e9
        }
    }
}

/// Gregorian leap-year test (proleptic, `%4`/`%100`/`%400` rule).
#[must_use]
pub const fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Days in a civil month; `0` for an out-of-range month index.
#[must_use]
pub const fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_jd_j2000_noon_is_exact() {
        // 2000-01-01 12:00:00 -> JD 2451545.0 -> whole 2451544.5, fraction 0.5.
        let (whole, frac) = split_julian_date(2000, 1, 1, 12, 0, 0.0);
        assert_eq!(whole, 2_451_544.5);
        assert_eq!(frac, 0.5);
        assert_eq!(whole + frac, J2000_JD);
    }

    #[test]
    fn j2000_seconds_epoch_is_zero() {
        assert_eq!(j2000_seconds(2000, 1, 1, 12, 0, 0.0), 0.0);
        // Whole-second epochs land on integers.
        assert_eq!(j2000_seconds(2000, 1, 2, 12, 0, 0.0), 86_400.0);
        // Sub-second remainder is carried.
        assert_eq!(j2000_seconds(2000, 1, 1, 12, 0, 0.25), 0.25);
    }

    #[test]
    fn second_of_day_is_clock_arithmetic() {
        assert_eq!(second_of_day(0, 0, 0.0), 0.0);
        assert_eq!(second_of_day(1, 2, 3.5), 3723.5);
    }

    #[test]
    fn day_of_year_jan1_midnight_is_one() {
        assert_eq!(day_of_year(2021, 1, 1, 0, 0, 0.0), 1.0);
        // Noon on Jan 1 is 1.5 day-of-year.
        assert_eq!(day_of_year(2021, 1, 1, 12, 0, 0.0), 1.5);
        // March 1 in a leap year is day 61.
        assert_eq!(day_of_year(2020, 3, 1, 0, 0, 0.0), 61.0);
        // March 1 in a common year is day 60.
        assert_eq!(day_of_year(2021, 3, 1, 0, 0, 0.0), 60.0);
    }

    #[test]
    fn day_of_year_int_matches_fractional_and_cumulative_table() {
        // Cumulative-month-day reference (the table the IONEX/tropo copies used).
        fn cum_table(year: i64, month: i64, day: i64) -> i64 {
            const CUM: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
            let leap = is_leap_year(year);
            let mut doy = CUM[(month - 1) as usize] + day;
            if leap && month > 2 {
                doy += 1;
            }
            doy
        }
        let cases: [(i64, i64, i64); 7] = [
            (2000, 1, 1),
            (2000, 3, 1),
            (2001, 3, 1),
            (2020, 12, 31),
            (1999, 6, 15),
            (2100, 3, 1),
            (2400, 2, 29),
        ];
        for (y, m, d) in cases {
            assert_eq!(
                day_of_year_int(y as i32, m as i32, d as i32),
                cum_table(y, m, d)
            );
            assert_eq!(
                day_of_year_int(y as i32, m as i32, d as i32) as f64,
                day_of_year(y as i32, m as i32, d as i32, 0, 0, 0.0)
            );
        }
    }

    #[test]
    fn civil_from_jdn_inverts_julian_day_number() {
        let cases: [(i32, i64, i64); 7] = [
            (2000, 1, 1),
            (1980, 1, 6),
            (2006, 1, 1),
            (1970, 1, 1),
            (2024, 2, 29),
            (1, 1, 1),
            (2099, 12, 31),
        ];
        for (y, m, d) in cases {
            let jdn = julian_day_number(y, m as i32, d as i32);
            assert_eq!(civil_from_julian_day_number(jdn), (i64::from(y), m, d));
        }
        // The +32044 algebraic variant (rinex_clock) agrees bit-for-bit.
        for jdn in (2_400_000..2_500_000).step_by(37) {
            let a = jdn + 32_044;
            let b = (4 * a + 3) / 146_097;
            let c = a - (146_097 * b) / 4;
            let dd = (4 * c + 3) / 1461;
            let e = c - (1461 * dd) / 4;
            let mm = (5 * e + 2) / 153;
            let day = e - (153 * mm + 2) / 5 + 1;
            let month = mm + 3 - 12 * (mm / 10);
            let year = 100 * b + dd - 4800 + mm / 10;
            assert_eq!(civil_from_julian_day_number(jdn), (year, month, day));
        }
    }

    #[test]
    fn civil_from_j2000_seconds_inverts_j2000_seconds() {
        // J2000 epoch itself, a day later, and a pre-J2000 negative count.
        assert_eq!(civil_from_j2000_seconds(0), (2000, 1, 1, 12, 0, 0));
        assert_eq!(civil_from_j2000_seconds(86_400), (2000, 1, 2, 12, 0, 0));
        // 2020-06-25 00:00:00 UTC is 646_315_200 J2000 seconds (IONEX parser pin).
        assert_eq!(
            civil_from_j2000_seconds(646_315_200),
            (2020, 6, 25, 0, 0, 0)
        );
        // Round-trip whole-second civil instants through the forward conversion.
        let cases: [(i32, i64, i64, i64, i64, i64); 4] = [
            (2000, 1, 1, 12, 0, 0),
            (1999, 12, 31, 23, 59, 59),
            (2024, 2, 29, 6, 30, 15),
            (2030, 7, 4, 0, 0, 1),
        ];
        for (y, m, d, h, mi, s) in cases {
            let secs = j2000_seconds(y, m as i32, d as i32, h as i32, mi as i32, s as f64) as i64;
            assert_eq!(
                civil_from_j2000_seconds(secs),
                (i64::from(y), m, d, h, mi, s)
            );
        }
    }

    #[test]
    fn mjd_and_leap_and_month_helpers() {
        assert_eq!(mjd_from_jd(J2000_JD), 51_544.5);
        assert!(is_leap_year(2000) && is_leap_year(2024) && !is_leap_year(2100));
        assert!(!is_leap_year(2023));
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(2024, 4), 30);
        assert_eq!(days_in_month(2024, 13), 0);
    }

    #[test]
    fn split_from_j2000_seconds_inverts_field_form() {
        // J2000 epoch (noon) and a day later land on *.0 boundaries.
        assert_eq!(split_julian_date_from_j2000_seconds(0), (J2000_JD, 0.0));
        assert_eq!(
            split_julian_date_from_j2000_seconds(86_400),
            (J2000_JD + 1.0, 0.0)
        );
        // 2020-06-25 00:00:00 (646_315_200 s) sits half a day before its noon JD.
        let (whole, frac) = split_julian_date_from_j2000_seconds(646_315_200);
        assert_eq!(j2000_seconds_from_split(whole, frac), 646_315_200.0);
    }

    #[test]
    fn civil_from_split_round_trips_split_julian_date() {
        let cases: [(i32, i32, i32, i32, i32, f64); 4] = [
            (2000, 1, 1, 12, 0, 0.0),
            (2020, 6, 25, 0, 0, 0.0),
            (2024, 2, 29, 6, 30, 15.0),
            (1999, 12, 31, 23, 59, 59.0),
        ];
        for (y, mo, d, h, mi, s) in cases {
            let (whole, frac) = split_julian_date(y, mo, d, h, mi, s);
            let civil = civil_from_split_julian_date(whole, frac);
            assert_eq!(
                civil,
                (
                    i64::from(y),
                    i64::from(mo),
                    i64::from(d),
                    i64::from(h),
                    i64::from(mi),
                    s
                )
            );
        }
    }

    #[test]
    fn split_add_seconds_carries_across_midnight() {
        // Add 6 hours within a day: no carry.
        let (w, f) = split_julian_date_add_seconds(2_451_544.5, 0.25, 6.0 * 3600.0);
        assert_eq!((w, f), (2_451_544.5, 0.5));
        // Add 18 hours from 12:00: carry one whole day.
        let (w, f) = split_julian_date_add_seconds(2_451_544.5, 0.5, 18.0 * 3600.0);
        assert_eq!(w, 2_451_545.5);
        assert!((f - 0.25).abs() < 1e-12);
    }

    #[test]
    fn instant_julian_date_and_doy_and_sod() {
        use super::super::model::{JulianDateSplit, TimeScale};
        // 2020-06-25 06:00:00, stored as a split Julian date.
        let (whole, frac) = split_julian_date(2020, 6, 25, 6, 0, 0.0);
        let epoch = Instant::from_julian_date(
            TimeScale::Gpst,
            JulianDateSplit::new(whole, frac).expect("valid split"),
        );
        assert_eq!(julian_date_from_instant(epoch), whole + frac);
        // 2020 is a leap year; June 25 is day 177, plus 6h = 0.25 day fraction.
        assert!((fractional_day_of_year_from_instant(epoch) - (177.0 + 0.25)).abs() < 1e-9);
        assert!((second_of_day_from_instant(epoch) - 6.0 * 3600.0).abs() < 1e-6);
    }

    #[test]
    fn from_utc_civil_matches_open_coded_path_and_accessors() {
        use super::super::model::{JulianDateSplit, TimeScale};
        let cases: [(i32, i32, i32, i32, i32, f64); 4] = [
            (2000, 1, 1, 12, 0, 0.0),
            (2020, 6, 25, 6, 0, 0.0),
            (2024, 2, 29, 6, 30, 15.0),
            (1999, 12, 31, 23, 59, 59.0),
        ];
        for (y, mo, d, h, mi, s) in cases {
            let instant = Instant::from_utc_civil(y, mo, d, h, mi, s).expect("valid civil instant");
            // Identical to the open-coded split + from_julian_date path.
            let (whole, frac) = split_julian_date(y, mo, d, h, mi, s);
            let expected = Instant::from_julian_date(
                TimeScale::Utc,
                JulianDateSplit::new(whole, frac).expect("valid split"),
            );
            assert_eq!(instant, expected);
            assert_eq!(instant.scale, TimeScale::Utc);
            // Round-trips through the existing instant accessors.
            assert_eq!(
                instant.julian_date(),
                Some(JulianDateSplit {
                    jd_whole: whole,
                    fraction: frac
                })
            );
            assert_eq!(julian_date_from_instant(instant), whole + frac);
            let civil = civil_from_split_julian_date(whole, frac);
            assert_eq!(
                civil,
                (
                    i64::from(y),
                    i64::from(mo),
                    i64::from(d),
                    i64::from(h),
                    i64::from(mi),
                    s
                )
            );
        }
    }

    #[test]
    fn from_utc_civil_rejects_out_of_day_clock_field() {
        // hour 25 pushes the day fraction past the one-day residual window.
        assert!(Instant::from_utc_civil(2020, 6, 25, 25, 0, 0.0).is_err());
    }

    #[test]
    fn j2000_seconds_from_split_matches_field_form() {
        // A split built from a civil instant scales back to the same seconds the
        // civil-fields conversion produces.
        let (whole, frac) = split_julian_date(2020, 6, 25, 0, 0, 0.0);
        assert_eq!(
            j2000_seconds_from_split(whole, frac),
            j2000_seconds(2020, 6, 25, 0, 0, 0.0)
        );
    }
}
