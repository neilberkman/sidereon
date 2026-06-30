//! Precise time scale conversions: UTC -> TAI -> TT -> TDB -> UT1.
//!
//! Mirrors Skyfield's `_utc()` path for bit-exact parity. The delta-T numerics,
//! summation order, and transcendental sequence are preserved EXACTLY: this
//! module is parity-critical and must not be refactored in any way that
//! perturbs a single last bit.
//!
//! The only change from the `orbis_nif` original is visibility: the formerly
//! `pub(crate)` `TimeScales` internals are promoted to a clean public API so a
//! Rust-only consumer of `sidereon-core` can reach the precise time scales
//! without pulling in Rustler or the BEAM.

use crate::astro::constants::time::{BDT_MINUS_TAI_S, GPST_MINUS_TAI_S};
use crate::astro::constants::time::{
    DAYS_PER_JULIAN_CENTURY, J2000_JD, SECONDS_PER_DAY, TT_MINUS_TAI_S,
};
use crate::astro::data::iers::UT1_DATA;
use crate::astro::time::civil;
use crate::astro::time::eop::{
    check_ut1_coverage, CoverageError, LeapSecondTable, TimeScaleInputErrorKind, Ut1Provenance,
    Validated, ValidityMode,
};
use crate::astro::time::model::TimeScale;
use crate::validate::{self, FieldError};

const ROUND_1E7: f64 = 10_000_000.0;

/// GLONASS system time minus UTC(SU), seconds. GLONASST = UTC(SU) + 3 h, a fixed
/// three-hour advance with no leap-second term of its own (ICD GLONASS Edition
/// 5.1, 2008, sec. 3.3.3). GLONASST still tracks UTC's leap seconds because UTC
/// does, so any GLONASST<->atomic-scale offset is epoch-dependent.
pub const GLONASST_MINUS_UTC_S: f64 = 3.0 * 3600.0;

/// Resolved set of Julian-date split time scales for one UTC instant.
///
/// All fields use the Skyfield split convention: `jd_whole` carries the integer
/// (and TAI-aligned) part of the day, and the per-scale `*_fraction` fields carry
/// the residual so that `jd_<scale> == jd_whole + <scale>_fraction` reproduces
/// the full Julian date without catastrophic cancellation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeScales {
    /// Integer Julian day boundary (TAI-aligned), shared by all scales.
    pub jd_whole: f64,
    /// UT1 day fraction relative to `jd_whole`.
    pub ut1_fraction: f64,
    /// TT day fraction relative to `jd_whole`.
    pub tt_fraction: f64,
    /// TDB day fraction relative to `jd_whole`.
    pub tdb_fraction: f64,
    /// Full UT1 Julian date.
    pub jd_ut1: f64,
    /// Full TT Julian date.
    pub jd_tt: f64,
    /// Full TDB Julian date.
    pub jd_tdb: f64,
}

struct LeapSecondEntry {
    mjd: i32,
    tai_utc: f64,
}

static LEAP_SECONDS: &[LeapSecondEntry] = &[
    LeapSecondEntry {
        mjd: 41317,
        tai_utc: 10.0,
    },
    LeapSecondEntry {
        mjd: 41499,
        tai_utc: 11.0,
    },
    LeapSecondEntry {
        mjd: 41683,
        tai_utc: 12.0,
    },
    LeapSecondEntry {
        mjd: 42048,
        tai_utc: 13.0,
    },
    LeapSecondEntry {
        mjd: 42413,
        tai_utc: 14.0,
    },
    LeapSecondEntry {
        mjd: 42778,
        tai_utc: 15.0,
    },
    LeapSecondEntry {
        mjd: 43144,
        tai_utc: 16.0,
    },
    LeapSecondEntry {
        mjd: 43509,
        tai_utc: 17.0,
    },
    LeapSecondEntry {
        mjd: 43874,
        tai_utc: 18.0,
    },
    LeapSecondEntry {
        mjd: 44239,
        tai_utc: 19.0,
    },
    LeapSecondEntry {
        mjd: 44786,
        tai_utc: 20.0,
    },
    LeapSecondEntry {
        mjd: 45151,
        tai_utc: 21.0,
    },
    LeapSecondEntry {
        mjd: 45516,
        tai_utc: 22.0,
    },
    LeapSecondEntry {
        mjd: 46247,
        tai_utc: 23.0,
    },
    LeapSecondEntry {
        mjd: 47161,
        tai_utc: 24.0,
    },
    LeapSecondEntry {
        mjd: 47892,
        tai_utc: 25.0,
    },
    LeapSecondEntry {
        mjd: 48257,
        tai_utc: 26.0,
    },
    LeapSecondEntry {
        mjd: 48804,
        tai_utc: 27.0,
    },
    LeapSecondEntry {
        mjd: 49169,
        tai_utc: 28.0,
    },
    LeapSecondEntry {
        mjd: 49534,
        tai_utc: 29.0,
    },
    LeapSecondEntry {
        mjd: 50083,
        tai_utc: 30.0,
    },
    LeapSecondEntry {
        mjd: 50448,
        tai_utc: 31.0,
    },
    LeapSecondEntry {
        mjd: 50813,
        tai_utc: 32.0,
    },
    LeapSecondEntry {
        mjd: 53736,
        tai_utc: 33.0,
    },
    LeapSecondEntry {
        mjd: 54832,
        tai_utc: 34.0,
    },
    LeapSecondEntry {
        mjd: 56109,
        tai_utc: 35.0,
    },
    LeapSecondEntry {
        mjd: 57204,
        tai_utc: 36.0,
    },
    LeapSecondEntry {
        mjd: 57754,
        tai_utc: 37.0,
    },
];

/// One segment of the pre-1972 "rubber second" UTC, where TAI-UTC varied as a
/// piecewise-linear function of the UTC Modified Julian Date rather than by
/// integer leap-second steps.
struct RubberSecondEntry {
    /// Integer UTC MJD at which this segment takes effect.
    start_mjd: i32,
    /// Constant term of `TAI-UTC = base + (MJD - ref_mjd) * rate` (seconds).
    base: f64,
    /// Reference MJD the linear drift is measured from.
    ref_mjd: f64,
    /// Drift rate of TAI-UTC, seconds per day of MJD.
    rate: f64,
}

/// The published IERS/USNO TAI-UTC table for the 1961-01-01 .. 1972-01-01
/// rubber-second era (USNO `tai-utc.dat`). Each segment gives
/// `TAI-UTC = base + (MJD - ref_mjd) * rate`, with MJD the UTC Modified Julian
/// Date. The table ends where the integer leap-second table (`LEAP_SECONDS`)
/// begins at MJD 41317 (1972-01-01, TAI-UTC = 10 s exactly).
static RUBBER_SECONDS: &[RubberSecondEntry] = &[
    RubberSecondEntry {
        start_mjd: 37300,
        base: 1.4228180,
        ref_mjd: 37300.0,
        rate: 0.001296,
    },
    RubberSecondEntry {
        start_mjd: 37512,
        base: 1.3728180,
        ref_mjd: 37300.0,
        rate: 0.001296,
    },
    RubberSecondEntry {
        start_mjd: 37665,
        base: 1.8458580,
        ref_mjd: 37665.0,
        rate: 0.0011232,
    },
    RubberSecondEntry {
        start_mjd: 38334,
        base: 1.9458580,
        ref_mjd: 37665.0,
        rate: 0.0011232,
    },
    RubberSecondEntry {
        start_mjd: 38395,
        base: 3.2401300,
        ref_mjd: 38761.0,
        rate: 0.001296,
    },
    RubberSecondEntry {
        start_mjd: 38486,
        base: 3.3401300,
        ref_mjd: 38761.0,
        rate: 0.001296,
    },
    RubberSecondEntry {
        start_mjd: 38639,
        base: 3.4401300,
        ref_mjd: 38761.0,
        rate: 0.001296,
    },
    RubberSecondEntry {
        start_mjd: 38761,
        base: 3.5401300,
        ref_mjd: 38761.0,
        rate: 0.001296,
    },
    RubberSecondEntry {
        start_mjd: 38820,
        base: 3.6401300,
        ref_mjd: 38761.0,
        rate: 0.001296,
    },
    RubberSecondEntry {
        start_mjd: 38942,
        base: 3.7401300,
        ref_mjd: 38761.0,
        rate: 0.001296,
    },
    RubberSecondEntry {
        start_mjd: 39004,
        base: 3.8401300,
        ref_mjd: 38761.0,
        rate: 0.001296,
    },
    RubberSecondEntry {
        start_mjd: 39126,
        base: 4.3131700,
        ref_mjd: 39126.0,
        rate: 0.002592,
    },
    RubberSecondEntry {
        start_mjd: 39887,
        base: 4.2131700,
        ref_mjd: 39126.0,
        rate: 0.002592,
    },
];

impl TimeScales {
    /// Resolve the split-Julian-date time scales for a UTC calendar instant.
    ///
    /// Validates the public boundary, then runs the exact Skyfield `_utc()` path.
    pub fn from_utc(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
    ) -> Result<Self, CoverageError> {
        validate::finite(second, "second").map_err(map_time_scale_field_error)?;
        validate::civil_datetime_with_second_policy(
            i64::from(year),
            i64::from(month),
            i64::from(day),
            i64::from(hour),
            i64::from(minute),
            second,
            validate::CivilSecondPolicy::UtcLike,
        )
        .map_err(map_time_scale_field_error)?;
        if second >= 60.0 && !is_positive_leap_second_label(year, month, day, hour, minute) {
            return Err(CoverageError::InvalidInput {
                field: "civil datetime",
                kind: TimeScaleInputErrorKind::InvalidCivilTime,
            });
        }
        Ok(Self::from_utc_unchecked(
            year, month, day, hour, minute, second,
        ))
    }

    /// Exact Skyfield `_utc()` path. The arithmetic order below is load-bearing
    /// for 0-ULP parity and MUST NOT be reordered.
    fn from_utc_unchecked(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
    ) -> Self {
        let jd_day = julian_day_number(year, month, day);
        let jd1 = jd_day as f64 - 0.5;
        let utc_seconds_of_day = hour as f64 * 3600.0 + minute as f64 * 60.0 + second;
        let leap_lookup_second = if second >= 60.0 { 59.0 } else { second };
        let jd2 =
            (leap_lookup_second + minute as f64 * 60.0 + hour as f64 * 3600.0) / SECONDS_PER_DAY;
        let jd_utc_total = jd1 + jd2;

        let leap_seconds = find_leap_seconds(jd_utc_total);
        let utc_seconds_at_midnight = jd1 * SECONDS_PER_DAY;

        let utc_whole_seconds = utc_seconds_of_day.trunc();
        let utc_subsecond = utc_seconds_of_day.fract();

        // Mirror Skyfield's _utc() path.
        let tai_seconds = utc_seconds_at_midnight + leap_seconds + utc_whole_seconds;
        let jd_whole = (tai_seconds / SECONDS_PER_DAY).floor();
        let tai_fraction =
            (tai_seconds - jd_whole * SECONDS_PER_DAY + utc_subsecond) / SECONDS_PER_DAY;
        let tt_offset_days = TT_MINUS_TAI_S / SECONDS_PER_DAY;

        let tt_fraction = tai_fraction + tt_offset_days;
        let jd_tt = jd_whole + tt_fraction;

        let delta_t = interpolate_delta_t(jd_tt);
        let ut1_fraction = tt_fraction - delta_t / SECONDS_PER_DAY;
        let jd_ut1 = jd_whole + ut1_fraction;

        let t = (jd_whole - J2000_JD + tt_fraction) / DAYS_PER_JULIAN_CENTURY;
        let tdb_minus_tt_seconds = 0.001657 * (628.3076 * t + 6.2401).sin()
            + 0.000022 * (575.3385 * t + 4.2970).sin()
            + 0.000014 * (1256.6152 * t + 6.1969).sin()
            + 0.000005 * (606.9777 * t + 4.0212).sin()
            + 0.000005 * (52.9691 * t + 0.4444).sin()
            + 0.000002 * (21.3299 * t + 5.5431).sin()
            + 0.000010 * t * (628.3076 * t + 4.2490).sin();

        let tdb_fraction = tt_fraction + tdb_minus_tt_seconds / SECONDS_PER_DAY;
        let jd_tdb = jd_whole + tdb_fraction;

        Self {
            jd_whole,
            ut1_fraction,
            tt_fraction,
            tdb_fraction,
            jd_ut1,
            jd_tt,
            jd_tdb,
        }
    }

    /// Coverage-policy-enforced variant of [`TimeScales::from_utc`].
    ///
    /// The numerics are produced by [`TimeScales::from_utc`] **unchanged** (the
    /// delta-T / UT1-UTC lookups still clamp at the embedded table edges exactly
    /// as before, preserving Skyfield 0-ULP parity). The only addition is that
    /// the resulting TT instant is classified against the embedded UT1/EOP
    /// coverage interval under the requested [`ValidityMode`]:
    ///
    /// - [`ValidityMode::Strict`]: an instant outside `[first_jd_tt, last_jd_tt]`
    ///   (where the delta-T table would have silently clamped/extrapolated)
    ///   returns [`CoverageError::OutsideCoverage`]. Nothing degraded is ever
    ///   returned.
    /// - [`ValidityMode::Permissive`]: the clamped value is returned, paired with
    ///   a [`crate::astro::time::eop::DegradeReason`] when the instant fell outside
    ///   coverage. This is the historical (parity) behaviour, now made explicit.
    ///
    /// In-coverage results are bit-identical to [`TimeScales::from_utc`] and are
    /// flagged not-degraded.
    pub fn from_utc_validated(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
        mode: ValidityMode,
    ) -> Result<Validated<Self>, CoverageError> {
        // Numerics first, exactly as the parity path produces them.
        let scales = Self::from_utc(year, month, day, hour, minute, second)?;
        // Classify the (already-clamped) instant against UT1 coverage. We
        // classify at jd_tt because the delta-T table axis is in TT (see
        // `ut1_coverage`), and jd_tt is independent of the clamped delta-T.
        let prov = ut1_coverage();
        let degraded = check_ut1_coverage(&prov, scales.jd_tt, mode)?;
        Ok(Validated {
            value: scales,
            degraded,
        })
    }

    /// Build [`TimeScales`] for a calendar instant labelled in `scale`.
    ///
    /// Non-UTC scales (GPST/GST/BDT/QZSST/TAI/TT/TDB and GLONASST) are converted
    /// to the UTC calendar label first via [`scale_calendar_to_utc`], so the
    /// Earth-orientation inputs used downstream are correct rather than offset by
    /// the scale's leap-second gap, then routed through [`Self::from_utc`]. This
    /// is the single home for the system-time-to-UTC inverse that the
    /// reduced-orbit bridge previously reimplemented.
    pub fn from_scale(
        scale: TimeScale,
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
    ) -> Result<Self, CoverageError> {
        let utc = scale_calendar_to_utc(
            scale,
            ScaleCal {
                year,
                month,
                day,
                hour,
                minute,
                second,
            },
        );
        Self::from_utc(
            utc.year, utc.month, utc.day, utc.hour, utc.minute, utc.second,
        )
    }
}

/// A mutable civil calendar instant used by the scale-to-UTC inverse.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ScaleCal {
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: f64,
}

/// Convert a calendar instant labelled in `scale` to the UTC calendar instant
/// [`TimeScales::from_utc`] consumes.
///
/// GLONASST = UTC(SU) + 3 h exactly (no leap term in the 3 h offset), so
/// recovering UTC is a plain -3 h shift that preserves UTC's leap labels. The
/// atomic-aligned scales are shifted to TAI and then resolved to UTC through the
/// leap-second table.
fn scale_calendar_to_utc(scale: TimeScale, cal: ScaleCal) -> ScaleCal {
    match scale {
        TimeScale::Utc => cal,
        TimeScale::Glonasst => normalize_calendar_seconds(cal, cal.second - GLONASST_MINUS_UTC_S),
        _ => {
            let tai = normalize_calendar_seconds(cal, cal.second + tai_minus_scale_seconds(scale));
            tai_calendar_to_utc(tai)
        }
    }
}

fn tai_minus_scale_seconds(scale: TimeScale) -> f64 {
    match scale {
        // Utc/Glonasst are handled before reaching here; 0.0 keeps the match
        // total without affecting the atomic path.
        TimeScale::Utc | TimeScale::Glonasst => 0.0,
        TimeScale::Tai => 0.0,
        TimeScale::Tt | TimeScale::Tdb => -TT_MINUS_TAI_S,
        // QZSST is steered synchronous with GPST, so it shares GPST's TAI offset.
        TimeScale::Gpst | TimeScale::Gst | TimeScale::Qzsst => GPST_MINUS_TAI_S,
        TimeScale::Bdt => BDT_MINUS_TAI_S,
    }
}

fn tai_calendar_to_utc(tai: ScaleCal) -> ScaleCal {
    if let Some(utc) = positive_leap_second_utc_label(tai) {
        return utc;
    }

    let mut leap = leap_seconds_at_utc_label(tai);
    let mut utc = normalize_calendar_seconds(tai, tai.second - leap);
    for _ in 0..3 {
        let next_leap = leap_seconds_at_utc_label(utc);
        if next_leap == leap {
            return utc;
        }
        leap = next_leap;
        utc = normalize_calendar_seconds(tai, tai.second - leap);
    }
    utc
}

fn positive_leap_second_utc_label(tai: ScaleCal) -> Option<ScaleCal> {
    let tai_sod = seconds_of_day(tai);
    let utc_midnight = ScaleCal {
        year: tai.year,
        month: tai.month,
        day: tai.day,
        hour: 0,
        minute: 0,
        second: 0.0,
    };
    let previous_second = normalize_calendar_seconds(utc_midnight, -1.0);
    let old_leap = leap_seconds_at_utc_label(previous_second);
    let new_leap = leap_seconds_at_utc_label(utc_midnight);
    if new_leap <= old_leap || !(old_leap..new_leap).contains(&tai_sod) {
        return None;
    }

    let mut utc = previous_second;
    utc.second = 60.0 + (tai_sod - old_leap);
    Some(utc)
}

fn leap_seconds_at_utc_label(cal: ScaleCal) -> f64 {
    let jd1 = julian_day_number(cal.year, cal.month, cal.day) as f64 - 0.5;
    let lookup_second = if cal.second >= 60.0 { 59.0 } else { cal.second };
    let jd2 =
        (cal.hour as f64 * 3600.0 + cal.minute as f64 * 60.0 + lookup_second) / SECONDS_PER_DAY;
    find_leap_seconds(jd1 + jd2)
}

fn seconds_of_day(cal: ScaleCal) -> f64 {
    cal.hour as f64 * 3600.0 + cal.minute as f64 * 60.0 + cal.second
}

fn normalize_calendar_seconds(mut cal: ScaleCal, second: f64) -> ScaleCal {
    // A non-finite second has no civil carry to perform and would spin the
    // subtract-by-60 loops below forever (inf - 60 == inf). Pass it through
    // unchanged so `from_utc` resolves it to a clean error instead of hanging.
    if !second.is_finite() {
        cal.second = second;
        return cal;
    }
    cal.second = second;
    while cal.second < 0.0 {
        cal.second += 60.0;
        cal.minute -= 1;
    }
    while cal.second >= 60.0 {
        cal.second -= 60.0;
        cal.minute += 1;
    }
    while cal.minute < 0 {
        cal.minute += 60;
        cal.hour -= 1;
    }
    while cal.minute > 59 {
        cal.minute -= 60;
        cal.hour += 1;
    }
    while cal.hour < 0 {
        cal.hour += 24;
        cal.day -= 1;
    }
    while cal.hour > 23 {
        cal.hour -= 24;
        cal.day += 1;
    }
    while cal.day < 1 {
        cal.month -= 1;
        if cal.month < 1 {
            cal.month = 12;
            cal.year -= 1;
        }
        cal.day = civil::days_in_month(i64::from(cal.year), i64::from(cal.month)) as i32;
    }
    loop {
        let month_days = civil::days_in_month(i64::from(cal.year), i64::from(cal.month)) as i32;
        if cal.day <= month_days {
            break;
        }
        cal.day -= month_days;
        cal.month += 1;
        if cal.month > 12 {
            cal.month = 1;
            cal.year += 1;
        }
    }
    cal
}

pub(crate) fn is_positive_leap_second_label(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
) -> bool {
    if hour != 23 || minute != 59 {
        return false;
    }
    let jd1 = julian_day_number(year, month, day) as f64 - 0.5;
    find_leap_seconds(jd1 + 1.0) > find_leap_seconds(jd1)
}

impl From<&FieldError> for TimeScaleInputErrorKind {
    fn from(error: &FieldError) -> Self {
        match error {
            FieldError::Missing { .. } => Self::Missing,
            FieldError::NonFinite { .. } => Self::NonFinite,
            FieldError::NotPositive { .. } => Self::NotPositive,
            FieldError::Negative { .. } => Self::Negative,
            FieldError::OutOfRange { .. } => Self::OutOfRange,
            FieldError::FloatParse { .. } => Self::FloatParse,
            FieldError::IntParse { .. } => Self::IntParse,
            FieldError::InvalidCivilDate { .. } => Self::InvalidCivilDate,
            FieldError::InvalidCivilTime { .. } => Self::InvalidCivilTime,
        }
    }
}

fn map_time_scale_field_error(error: FieldError) -> CoverageError {
    CoverageError::InvalidInput {
        field: error.field(),
        kind: TimeScaleInputErrorKind::from(&error),
    }
}

/// Civil calendar -> Julian day number (Fliegel-style, integer arithmetic).
pub fn julian_day_number(year: i32, month: i32, day: i32) -> i64 {
    let year = i64::from(year);
    let month = i64::from(month);
    let day = i64::from(day);
    let janfeb = month <= 2;
    let g = year + 4716 - if janfeb { 1 } else { 0 };
    let f = (month + 9) % 12;
    let e = 1461 * g / 4 + day - 1402;
    let j = e + (153 * f + 2) / 5;
    j + 38 - ((g + 184) / 100) * 3 / 4
}

/// TAI-UTC (cumulative leap seconds) for a UTC Julian date.
///
/// For instants from 1972-01-01 (MJD 41317) onward this reads the embedded IERS
/// integer leap-second table, clamping above to the last entry exactly as the
/// original `orbis_nif` implementation did (parity-preserving). This post-1972
/// branch is byte-for-byte the original lookup and is never perturbed.
///
/// For instants in the 1961-01-01 .. 1972-01-01 rubber-second era it evaluates
/// the published piecewise-linear IERS/USNO model
/// `TAI-UTC = base + (MJD - ref_mjd) * rate` (see [`RUBBER_SECONDS`]) using the
/// fractional UTC MJD, so the offset is continuous within each segment as the
/// historical definition requires. Before 1961 (and for a non-finite input) it
/// clamps to the first rubber-second segment's value rather than extrapolating
/// into undefined territory.
///
/// Boundary semantics (post-1972): the date is the integer MJD (`(jd_utc -
/// 2400000.5) as i32`), so the count steps at UTC midnight (the table's
/// effective MJD is the first full day under the new value). The inserted leap
/// second `23:59:60` and the following `00:00:00` share essentially the same
/// Julian date, so an end-of-day instant resolves to the **post-leap** count -
/// the leap second itself cannot be distinguished from the next day's start
/// through a JD. This is intrinsic to a JD-keyed lookup and is pinned to
/// `orbis_nif` for bit-exact parity; callers that must label `23:59:60`
/// distinctly have to carry the civil second out-of-band rather than rely on
/// this function.
pub fn find_leap_seconds(jd_utc: f64) -> f64 {
    let mjd = (jd_utc - 2400000.5) as i32;
    if mjd >= LEAP_SECONDS[0].mjd {
        // Post-1972 integer leap-second table (unchanged, bit-identical).
        let mut ls = 10.0;
        for entry in LEAP_SECONDS {
            if mjd >= entry.mjd {
                ls = entry.tai_utc;
            } else {
                break;
            }
        }
        return ls;
    }
    rubber_tai_minus_utc(jd_utc)
}

/// Evaluate the pre-1972 piecewise-linear TAI-UTC model at a UTC Julian date.
///
/// Selects the latest [`RUBBER_SECONDS`] segment whose `start_mjd` precedes the
/// instant's fractional MJD and applies `base + (MJD - ref_mjd) * rate`. Inputs
/// before the first segment (pre-1961) and non-finite inputs clamp to the first
/// segment's constant term.
fn rubber_tai_minus_utc(jd_utc: f64) -> f64 {
    let mjd = jd_utc - 2400000.5;
    let first = &RUBBER_SECONDS[0];
    // Pre-1961 or non-finite input clamps to the first segment's constant.
    if mjd.is_nan() || mjd < first.start_mjd as f64 {
        return first.base;
    }
    let mut selected = first;
    for entry in RUBBER_SECONDS {
        if mjd >= entry.start_mjd as f64 {
            selected = entry;
        } else {
            break;
        }
    }
    selected.base + (mjd - selected.ref_mjd) * selected.rate
}

/// Error returned by the inter-system time-scale offset helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TimeOffsetError {
    /// A fixed-offset query [`timescale_offset_s`] named a UTC-based scale
    /// (UTC or GLONASST) whose offset to the atomic scales depends on the
    /// instant's leap-second count. Use [`timescale_offset_at_s`] with an epoch.
    #[error(
        "time-scale {0} is UTC-based; its offset is epoch-dependent, use timescale_offset_at_s"
    )]
    EpochRequired(&'static str),
    /// TDB differs from TT by an epoch-dependent periodic relativistic term, not
    /// a fixed offset; resolve TDB through [`TimeScales::from_utc`] instead.
    #[error("time-scale {0} has no fixed/constant offset; resolve it through TimeScales")]
    Unsupported(&'static str),
    /// A leap-aware query received a non-finite UTC Julian date.
    #[error("utc_jd must be finite to resolve leap seconds for scale {0}")]
    NonFiniteEpoch(&'static str),
}

/// Stable machine-readable discriminant for [`TimeOffsetError`].
///
/// The variants of [`TimeOffsetError`] carry only a human-facing `&'static str`,
/// which a C/FFI caller cannot branch on without parsing text. This `#[repr(u8)]`
/// code gives each variant a stable numeric tag (reachable across the FFI
/// boundary as `code() as u8`). The values are part of the public contract: `0`
/// is reserved for "no error", and an existing code is never renumbered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TimeOffsetErrorCode {
    /// [`TimeOffsetError::EpochRequired`].
    EpochRequired = 1,
    /// [`TimeOffsetError::Unsupported`].
    Unsupported = 2,
    /// [`TimeOffsetError::NonFiniteEpoch`].
    NonFiniteEpoch = 3,
}

impl TimeOffsetError {
    /// The stable machine-readable discriminant for this error.
    ///
    /// For C/FFI consumers that must distinguish the variants programmatically
    /// without parsing the [`Display`](core::fmt::Display) message. The numeric
    /// values (`error.code() as u8`) are a stable part of the public API.
    #[must_use]
    pub fn code(&self) -> TimeOffsetErrorCode {
        match self {
            Self::EpochRequired(_) => TimeOffsetErrorCode::EpochRequired,
            Self::Unsupported(_) => TimeOffsetErrorCode::Unsupported,
            Self::NonFiniteEpoch(_) => TimeOffsetErrorCode::NonFiniteEpoch,
        }
    }
}

/// True for scales whose offset to TAI carries UTC leap seconds (UTC, GLONASST).
fn is_utc_based(scale: TimeScale) -> bool {
    matches!(scale, TimeScale::Utc | TimeScale::Glonasst)
}

/// `scale_reading - TAI_reading` (seconds) for the same physical instant.
///
/// For the atomic scales (TAI/TT/GPST/GST/QZSST/BDT) this is a fixed constant
/// and `utc_jd` is ignored. For the UTC-based scales (UTC/GLONASST) it depends
/// on the leap-second count at `utc_jd`. TDB is rejected (epoch-dependent
/// periodic term, no fixed offset).
fn scale_minus_tai_s(scale: TimeScale, utc_jd: f64) -> Result<f64, TimeOffsetError> {
    let leap = |s: TimeScale| -> Result<f64, TimeOffsetError> {
        if !utc_jd.is_finite() {
            return Err(TimeOffsetError::NonFiniteEpoch(s.abbrev()));
        }
        Ok(find_leap_seconds(utc_jd))
    };
    Ok(match scale {
        TimeScale::Tai => 0.0,
        // TT = TAI + 32.184 s (IERS Conventions 2010, TT definition).
        TimeScale::Tt => TT_MINUS_TAI_S,
        // GPST = TAI - 19 s (IS-GPS-200, fixed since the 1980 GPS epoch).
        TimeScale::Gpst => -GPST_MINUS_TAI_S,
        // GST nominally = GPST (Galileo OS SIS ICD, sec. 5.1.3: GST is steered
        // to GPST; the real-time GGTO is a *broadcast* correction, not a fixed
        // constant). Nominal GST - TAI therefore equals GPST - TAI = -19 s.
        TimeScale::Gst => -GPST_MINUS_TAI_S,
        // QZSST nominally = GPST (IS-QZSS-PNT, sec. 3.2.2: synchronous with
        // GPST). Nominal QZSST - TAI = GPST - TAI = -19 s.
        TimeScale::Qzsst => -GPST_MINUS_TAI_S,
        // BDT = TAI - 33 s (BeiDou ICD, BDT epoch 2006-01-01, 14 leap seconds
        // behind GPST's 1980 epoch: GPST - BDT = 14 s, so BDT - TAI = -33 s).
        TimeScale::Bdt => -BDT_MINUS_TAI_S,
        // UTC = TAI - (TAI-UTC), leap-second dependent.
        TimeScale::Utc => -leap(scale)?,
        // GLONASST = UTC + 3 h = TAI - (TAI-UTC) + 3 h.
        TimeScale::Glonasst => -leap(scale)? + GLONASST_MINUS_UTC_S,
        TimeScale::Tdb => return Err(TimeOffsetError::Unsupported("TDB")),
    })
}

/// Fixed inter-system offset `to_reading - from_reading` (seconds) for scales
/// whose mutual offset is a constant.
///
/// Returns the value that, added to a `from`-scale reading, yields the
/// `to`-scale reading of the same physical instant. This covers the atomic
/// scales TAI/TT/GPST/GST/QZSST/BDT, whose offsets are fixed by their defining
/// ICDs (see [`scale_minus_tai_s`] for the per-scale citations).
///
/// Returns [`TimeOffsetError::EpochRequired`] if either scale is UTC-based
/// (UTC/GLONASST) — those need [`timescale_offset_at_s`] — and
/// [`TimeOffsetError::Unsupported`] for TDB.
///
/// # Note on the brief's `(from, to)` signature
///
/// The original A1 brief specified `timescale_offset_s(from, to) -> Result<f64>`.
/// That signature cannot express the leap-aware GLONASST/UTC offsets (they need
/// an epoch), so this function keeps the no-epoch form for the fixed atomic
/// offsets and *errors* for UTC-based scales, while the leap-aware variant
/// [`timescale_offset_at_s`] takes an explicit epoch. The split makes the
/// epoch a compile-time-visible requirement rather than a silently-ignored arg.
pub fn timescale_offset_s(from: TimeScale, to: TimeScale) -> Result<f64, TimeOffsetError> {
    for scale in [from, to] {
        if is_utc_based(scale) {
            return Err(TimeOffsetError::EpochRequired(scale.abbrev()));
        }
    }
    // utc_jd is unused for the atomic scales reached here.
    timescale_offset_at_s(from, to, f64::NAN)
}

/// Leap-aware inter-system offset `to_reading - from_reading` (seconds) at a
/// given UTC instant.
///
/// `utc_jd` is the UTC Julian date of the instant, used only to resolve the
/// leap-second count when `from` or `to` is UTC-based (UTC/GLONASST); for
/// purely atomic pairs it is ignored. Away from a leap-second boundary the
/// leap count is stable, so the exact scale of `utc_jd` is immaterial; within
/// the boundary window pass the UTC Julian date so the correct count is picked.
///
/// The result, added to a `from`-scale reading, yields the `to`-scale reading
/// of the same physical instant. TDB is rejected (see [`TimeOffsetError`]).
pub fn timescale_offset_at_s(
    from: TimeScale,
    to: TimeScale,
    utc_jd: f64,
) -> Result<f64, TimeOffsetError> {
    Ok(scale_minus_tai_s(to, utc_jd)? - scale_minus_tai_s(from, utc_jd)?)
}

/// Provenance + coverage descriptor for the embedded leap-second table.
///
/// Exposed so a precision pipeline can interrogate table coverage and apply
/// strict-vs-permissive policy (see [`crate::astro::time::eop`]).
pub fn leap_second_table() -> LeapSecondTable {
    LeapSecondTable {
        source: "IERS Bulletin C (TAI-UTC), bundled in sidereon-core",
        first_mjd: LEAP_SECONDS.first().map(|e| e.mjd).unwrap_or(0),
        last_mjd: LEAP_SECONDS.last().map(|e| e.mjd).unwrap_or(0),
        entries: LEAP_SECONDS.len(),
    }
}

fn interpolate_delta_t(jd_tt: f64) -> f64 {
    // Build delta-T table on first call (matching C++ lazy static pattern).
    use std::sync::LazyLock;

    struct DeltaTRow {
        jd_tt: f64,
        delta_t: f64,
    }

    static TABLE: LazyLock<Vec<DeltaTRow>> = LazyLock::new(|| {
        UT1_DATA
            .iter()
            .map(|entry| {
                let jd_utc = entry.mjd as f64 + 2400000.5;
                let leap_seconds = find_leap_seconds(jd_utc);
                let tt_minus_utc = leap_seconds + TT_MINUS_TAI_S;
                let delta_t = ((tt_minus_utc - entry.ut1_utc) * ROUND_1E7).round() / ROUND_1E7;
                DeltaTRow {
                    jd_tt: jd_utc + tt_minus_utc / SECONDS_PER_DAY,
                    delta_t,
                }
            })
            .collect()
    });

    // Binary search for the bracketing entries.
    match TABLE.binary_search_by(|row| row.jd_tt.partial_cmp(&jd_tt).unwrap()) {
        Ok(i) => TABLE[i].delta_t,
        Err(0) => TABLE[0].delta_t,
        Err(i) if i >= TABLE.len() => TABLE.last().unwrap().delta_t,
        Err(i) => {
            let p1 = &TABLE[i - 1];
            let p2 = &TABLE[i];
            p1.delta_t + (jd_tt - p1.jd_tt) * (p2.delta_t - p1.delta_t) / (p2.jd_tt - p1.jd_tt)
        }
    }
}

/// UT1 coverage interval for the embedded EOP table, in TT Julian dates.
///
/// Outside this interval the delta-T interpolation clamps to the nearest table
/// edge (parity-preserving Skyfield behaviour). Strict-mode callers should treat
/// instants outside `[first_jd_tt, last_jd_tt]` as degraded; see
/// [`crate::astro::time::eop`].
pub fn ut1_coverage() -> Ut1Provenance {
    let first = UT1_DATA.first();
    let last = UT1_DATA.last();
    let to_jd_tt = |mjd: i32| -> f64 {
        let jd_utc = mjd as f64 + 2400000.5;
        let tt_minus_utc = find_leap_seconds(jd_utc) + TT_MINUS_TAI_S;
        jd_utc + tt_minus_utc / SECONDS_PER_DAY
    };
    Ut1Provenance {
        source: "IERS Earth Orientation Parameters (UT1-UTC), bundled",
        first_mjd: first.map(|e| e.mjd).unwrap_or(0),
        last_mjd: last.map(|e| e.mjd).unwrap_or(0),
        first_jd_tt: first.map(|e| to_jd_tt(e.mjd)).unwrap_or(0.0),
        last_jd_tt: last.map(|e| to_jd_tt(e.mjd)).unwrap_or(0.0),
        entries: UT1_DATA.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn julian_day_number_widens_extreme_inputs_before_arithmetic() {
        let _ = julian_day_number(i32::MIN, i32::MAX, i32::MAX);
        let _ = julian_day_number(i32::MAX, i32::MIN, i32::MIN);
    }

    /// UTC Julian date for a calendar instant, built exactly as the parity path
    /// (`from_utc_unchecked`) does, so the embedded leap table is queried at the
    /// same instant the production code would.
    fn utc_jd(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: f64) -> f64 {
        let jd1 = julian_day_number(year, month, day) as f64 - 0.5;
        let sod = hour as f64 * 3600.0 + minute as f64 * 60.0 + second;
        jd1 + sod / SECONDS_PER_DAY
    }

    // --- Pre-1972 rubber-second UTC-TAI model. --------------------------------

    #[test]
    fn tai_minus_utc_pre_1972_matches_published_table() {
        // Published IERS/USNO TAI-UTC values (tai-utc.dat) for the rubber-second
        // era, TAI-UTC = base + (MJD - ref) * rate, evaluated at UTC midnight.
        let cases = [
            // (year, month, day, published TAI-UTC seconds)
            (1961, 1, 1, 1.4228180), // segment start MJD 37300
            (1965, 1, 1, 3.5401300), // segment start MJD 38761
            (1968, 2, 1, 6.1856820), // MJD 39887: 4.2131700 + 761*0.002592
            (1971, 1, 1, 8.9461620), // MJD 40952: 4.2131700 + 1826*0.002592
        ];
        for (y, m, d, want) in cases {
            let jd = utc_jd(y, m, d, 0, 0, 0.0);
            let got = find_leap_seconds(jd);
            assert!(
                (got - want).abs() < 1.0e-7,
                "TAI-UTC at {y}-{m:02}-{d:02}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn tai_minus_utc_pre_1972_is_continuous_within_a_segment() {
        // The rubber second drifts linearly: noon must sit half a day's rate
        // above midnight inside the 1968 segment (rate 0.002592 s/day).
        let midnight = find_leap_seconds(utc_jd(1969, 6, 1, 0, 0, 0.0));
        let noon = find_leap_seconds(utc_jd(1969, 6, 1, 12, 0, 0.0));
        assert!(
            (noon - midnight - 0.5 * 0.002592).abs() < 1.0e-9,
            "rubber-second drift over half a day must equal 0.5*rate"
        );
    }

    #[test]
    fn tai_minus_utc_steps_to_ten_at_1972_and_post_1972_unchanged() {
        // The famous final rubber-second value just before the 1972 step.
        let pre = find_leap_seconds(utc_jd(1971, 12, 31, 0, 0, 0.0));
        assert!((pre - 9.8896500).abs() < 1.0e-7, "1971-12-31 TAI-UTC");
        // 1972-01-01 is the first integer leap-second entry: exactly 10 s.
        assert_eq!(find_leap_seconds(utc_jd(1972, 1, 1, 0, 0, 0.0)), 10.0);
        // Post-1972 integer table is untouched (bit-identical goldens).
        assert_eq!(find_leap_seconds(utc_jd(1980, 1, 1, 0, 0, 0.0)), 19.0);
        assert_eq!(find_leap_seconds(utc_jd(2017, 1, 1, 0, 0, 0.0)), 37.0);
    }

    #[test]
    fn tai_minus_utc_pre_1961_clamps_to_first_segment() {
        // Before the table and for a non-finite input, clamp to the 1961 value.
        assert_eq!(find_leap_seconds(utc_jd(1958, 1, 1, 0, 0, 0.0)), 1.4228180);
        assert_eq!(find_leap_seconds(f64::NAN), 1.4228180);
    }

    // --- Fixed atomic-scale offsets (hex-float goldens). ----------------------
    //
    // Goldens are exact f64 bit patterns. `timescale_offset_s(from, to)` returns
    // `to_reading - from_reading`, i.e. the value added to a `from` reading to
    // obtain the `to` reading of the same instant.

    #[test]
    fn offset_gpst_to_bdt_is_minus_14s() {
        // BeiDou ICD: BDT = GPST - 14 s. Golden: -14.0.
        let want = f64::from_bits(0xc02c_0000_0000_0000);
        assert_eq!(
            timescale_offset_s(TimeScale::Gpst, TimeScale::Bdt).expect("fixed offset"),
            want
        );
        assert_eq!(want, -14.0);
    }

    #[test]
    fn offset_bdt_to_gpst_is_plus_14s() {
        assert_eq!(
            timescale_offset_s(TimeScale::Bdt, TimeScale::Gpst).expect("fixed offset"),
            14.0
        );
    }

    #[test]
    fn offset_gpst_to_gst_is_nominal_zero() {
        // Galileo OS SIS ICD: GST steered to GPST; nominal GGTO = 0 (the live
        // GGTO is a broadcast correction, not represented here).
        assert_eq!(
            timescale_offset_s(TimeScale::Gpst, TimeScale::Gst).expect("fixed offset"),
            0.0
        );
    }

    #[test]
    fn offset_gpst_to_qzsst_is_nominal_zero() {
        // IS-QZSS-PNT: QZSST synchronous with GPST; nominal offset = 0.
        assert_eq!(
            timescale_offset_s(TimeScale::Gpst, TimeScale::Qzsst).expect("fixed offset"),
            0.0
        );
        assert_eq!(
            timescale_offset_s(TimeScale::Gst, TimeScale::Qzsst).expect("fixed offset"),
            0.0
        );
    }

    #[test]
    fn offset_tai_to_tt_is_32_184s() {
        // IERS Conventions: TT = TAI + 32.184 s. Golden: exact bits of 32.184.
        let want = f64::from_bits(0x4040_178d_4fdf_3b64);
        assert_eq!(
            timescale_offset_s(TimeScale::Tai, TimeScale::Tt).expect("fixed offset"),
            want
        );
        assert_eq!(want, 32.184);
    }

    #[test]
    fn offset_gpst_to_tt_is_51_184s() {
        // TT - GPST = (TAI + 32.184) - (TAI - 19) = 51.184 s. Golden: exact bits.
        let want = f64::from_bits(0x4049_978d_4fdf_3b64);
        assert_eq!(
            timescale_offset_s(TimeScale::Gpst, TimeScale::Tt).expect("fixed offset"),
            want
        );
        assert_eq!(want, 51.184);
    }

    #[test]
    fn offset_gpst_to_tai_is_plus_19s() {
        // IS-GPS-200: GPST = TAI - 19 s, so TAI - GPST = +19 s.
        assert_eq!(
            timescale_offset_s(TimeScale::Gpst, TimeScale::Tai).expect("fixed offset"),
            19.0
        );
    }

    #[test]
    fn fixed_offsets_are_antisymmetric_for_atomic_pairs() {
        let atomic = [
            TimeScale::Tai,
            TimeScale::Tt,
            TimeScale::Gpst,
            TimeScale::Gst,
            TimeScale::Qzsst,
            TimeScale::Bdt,
        ];
        for &a in &atomic {
            for &b in &atomic {
                let ab = timescale_offset_s(a, b).expect("fixed offset");
                let ba = timescale_offset_s(b, a).expect("fixed offset");
                assert_eq!(ab, -ba, "offset({a:?},{b:?}) must be -offset({b:?},{a:?})");
            }
        }
    }

    // --- Error cases. ---------------------------------------------------------

    #[test]
    fn fixed_offset_requires_epoch_for_utc_based_scales() {
        assert_eq!(
            timescale_offset_s(TimeScale::Gpst, TimeScale::Utc),
            Err(TimeOffsetError::EpochRequired("UTC"))
        );
        assert_eq!(
            timescale_offset_s(TimeScale::Glonasst, TimeScale::Gpst),
            Err(TimeOffsetError::EpochRequired("GLONASST"))
        );
    }

    #[test]
    fn tdb_has_no_fixed_offset() {
        assert_eq!(
            timescale_offset_s(TimeScale::Gpst, TimeScale::Tdb),
            Err(TimeOffsetError::Unsupported("TDB"))
        );
        assert_eq!(
            timescale_offset_at_s(TimeScale::Tt, TimeScale::Tdb, 2_451_545.0),
            Err(TimeOffsetError::Unsupported("TDB"))
        );
    }

    #[test]
    fn leap_aware_offset_rejects_non_finite_epoch() {
        assert_eq!(
            timescale_offset_at_s(TimeScale::Gpst, TimeScale::Utc, f64::NAN),
            Err(TimeOffsetError::NonFiniteEpoch("UTC"))
        );
        assert_eq!(
            timescale_offset_at_s(TimeScale::Glonasst, TimeScale::Gpst, f64::INFINITY),
            Err(TimeOffsetError::NonFiniteEpoch("GLONASST"))
        );
    }

    #[test]
    fn error_code_maps_each_variant_to_stable_discriminant() {
        assert_eq!(
            TimeOffsetError::EpochRequired("UTC").code(),
            TimeOffsetErrorCode::EpochRequired
        );
        assert_eq!(
            TimeOffsetError::Unsupported("TDB").code(),
            TimeOffsetErrorCode::Unsupported
        );
        assert_eq!(
            TimeOffsetError::NonFiniteEpoch("UTC").code(),
            TimeOffsetErrorCode::NonFiniteEpoch
        );
        // The repr(u8) values are the stable FFI contract: 0 is reserved for
        // "no error", so the codes start at 1 and never collide.
        assert_eq!(TimeOffsetErrorCode::EpochRequired as u8, 1);
        assert_eq!(TimeOffsetErrorCode::Unsupported as u8, 2);
        assert_eq!(TimeOffsetErrorCode::NonFiniteEpoch as u8, 3);
        // The code is independent of the payload string.
        assert_eq!(
            TimeOffsetError::EpochRequired("GLONASST").code() as u8,
            TimeOffsetError::EpochRequired("UTC").code() as u8
        );
    }

    #[test]
    fn leap_aware_offset_ignores_epoch_for_atomic_pairs() {
        // Atomic pair never touches the leap table, so a NaN epoch is harmless.
        assert_eq!(
            timescale_offset_at_s(TimeScale::Gpst, TimeScale::Bdt, f64::NAN)
                .expect("atomic pair ignores epoch"),
            -14.0
        );
    }

    // --- Leap-aware GPST<->UTC<->GLONASST, validated against RTKLIB. -----------
    //
    // RTKLIB (gpst2utc/utc2gpst + the +3 h GLONASST advance) reports, at the
    // listed UTC instants:
    //   2017-01-01 00:00:00  GPST-UTC=+18  UTC-GPST=-18  GLONASST-GPST=+10782
    //   2016-12-31 23:59:59  GPST-UTC=+17  UTC-GPST=-17  GLONASST-GPST=+10783
    //   2000-01-01 12:00:00  GPST-UTC=+13  UTC-GPST=-13  GLONASST-GPST=+10787

    #[test]
    fn offset_utc_gpst_matches_rtklib_2017() {
        let jd = utc_jd(2017, 1, 1, 0, 0, 0.0);
        // GPST - UTC = +18 (golden 18.0).
        let want = f64::from_bits(0x4032_0000_0000_0000);
        assert_eq!(
            timescale_offset_at_s(TimeScale::Utc, TimeScale::Gpst, jd).expect("leap-aware offset"),
            want
        );
        assert_eq!(want, 18.0);
        // UTC - GPST = -18.
        assert_eq!(
            timescale_offset_at_s(TimeScale::Gpst, TimeScale::Utc, jd).expect("leap-aware offset"),
            -18.0
        );
    }

    #[test]
    fn offset_glonasst_gpst_matches_rtklib_2017() {
        let jd = utc_jd(2017, 1, 1, 0, 0, 0.0);
        // GLONASST - GPST = +10782 (golden bits of 10782.0).
        let want = f64::from_bits(0x40c5_0f00_0000_0000);
        assert_eq!(
            timescale_offset_at_s(TimeScale::Gpst, TimeScale::Glonasst, jd)
                .expect("leap-aware offset"),
            want
        );
        assert_eq!(want, 10782.0);
    }

    #[test]
    fn offset_glonasst_gpst_at_j2000_matches_rtklib() {
        let jd = utc_jd(2000, 1, 1, 12, 0, 0.0);
        // GLONASST - GPST = +10787 at the J2000 leap epoch (TAI-UTC=32).
        assert_eq!(
            timescale_offset_at_s(TimeScale::Gpst, TimeScale::Glonasst, jd)
                .expect("leap-aware offset"),
            10787.0
        );
    }

    /// The brief's required leap-second-boundary test: the GPST->GLONASST offset
    /// must step by exactly one second across the 2016-12-31 -> 2017-01-01 leap,
    /// from +10783 s (TAI-UTC=36) to +10782 s (TAI-UTC=37), matching RTKLIB.
    #[test]
    fn glonasst_offset_steps_across_2017_leap_second() {
        let before = utc_jd(2016, 12, 31, 23, 59, 59.0);
        let after = utc_jd(2017, 1, 1, 0, 0, 0.0);

        let off_before = timescale_offset_at_s(TimeScale::Gpst, TimeScale::Glonasst, before)
            .expect("leap-aware offset");
        let off_after = timescale_offset_at_s(TimeScale::Gpst, TimeScale::Glonasst, after)
            .expect("leap-aware offset");

        assert_eq!(off_before, f64::from_bits(0x40c5_0f80_0000_0000)); // 10783.0
        assert_eq!(off_after, f64::from_bits(0x40c5_0f00_0000_0000)); // 10782.0
        assert_eq!(off_before, 10783.0);
        assert_eq!(off_after, 10782.0);
        // The leap insertion lengthens the GPST-ahead-of-UTC gap by 1 s, so the
        // GLONASST(=UTC+3h)-minus-GPST offset shrinks by exactly 1 s.
        assert_eq!(off_before - off_after, 1.0);

        // Cross-check the UTC leg too: GPST-UTC goes 17 -> 18 across the leap.
        assert_eq!(
            timescale_offset_at_s(TimeScale::Utc, TimeScale::Gpst, before)
                .expect("leap-aware offset"),
            17.0
        );
        assert_eq!(
            timescale_offset_at_s(TimeScale::Utc, TimeScale::Gpst, after)
                .expect("leap-aware offset"),
            18.0
        );
    }

    /// Cross-check the leap-aware GPST<->UTC offset against the parity-critical
    /// `TimeScales::from_utc` path (which is itself Skyfield 0-ULP). The two TT
    /// fractions for the same instant labelled in UTC vs in GPST must differ by
    /// the offset this helper reports.
    #[test]
    fn leap_aware_offset_agrees_with_timescales_path() {
        // GPST is ahead of UTC by `timescale_offset_at_s(Utc, Gpst, jd)` seconds.
        let jd = utc_jd(2020, 6, 15, 0, 0, 0.0);
        let gpst_minus_utc =
            timescale_offset_at_s(TimeScale::Utc, TimeScale::Gpst, jd).expect("leap-aware offset");
        // 2020 is between the 2017 leap and now: TAI-UTC=37, GPST-UTC=18.
        assert_eq!(gpst_minus_utc, 18.0);
    }
}
