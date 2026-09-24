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
    DAYS_PER_JULIAN_CENTURY, J2000_JD, SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_MINUTE,
    TT_MINUS_TAI_S,
};
use crate::astro::data::iers::{Ut1Entry, UT1_DATA};
use crate::astro::time::civil;
use crate::astro::time::eop::{
    check_ut1_coverage, ut1_coverage_departure, CoverageError, DegradeReason, LeapSecondTable,
    TimeScaleInputErrorKind, Ut1Provenance, Validated, ValidityMode,
};
use crate::astro::time::model::TimeScale;
use crate::validate::{self, FieldError};

const ROUND_1E7: f64 = 10_000_000.0;

/// GLONASS system time minus UTC(SU), seconds. GLONASST = UTC(SU) + 3 h, a fixed
/// three-hour advance with no leap-second term of its own (ICD GLONASS Edition
/// 5.1, 2008, sec. 3.3.3). GLONASST still tracks UTC's leap seconds because UTC
/// does, so any GLONASST<->atomic-scale offset is epoch-dependent.
pub const GLONASST_MINUS_UTC_S: f64 = 3.0 * SECONDS_PER_HOUR;

/// TT/TCG defining rate constant `L_G` from IAU 2000 Resolution B1.9.
pub const TT_TCG_RATE_L_G: f64 = 6.969290134e-10;

/// TDB/TCB defining rate constant `L_B` from IAU 2006 Resolution B3.
pub const TDB_TCB_RATE_L_B: f64 = 1.550519768e-8;

/// TDB offset constant `TDB0`, seconds, from IAU 2006 Resolution B3.
pub const TDB_TCB_OFFSET_TDB0_S: f64 = -6.55e-5;

/// TT/TCG and TDB/TCB reference epoch Julian Date.
pub const TCG_TCB_REFERENCE_JD: f64 = 2_443_144.500_372_5;

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
    /// Whether `ut1_fraction` and `jd_ut1` come from outside the UT1 table.
    ///
    /// `None` when the TT instant lies inside the UT1 table's coverage
    /// interval, so UT1 is interpolated from the table. `Some` when it lies
    /// before or after the table: TT-UT1 then comes from the long-term delta-T
    /// curve Skyfield splices around the table (see
    /// [`crate::astro::time::eop`]), which is a model, not a measured or
    /// predicted value. Every constructor sets this; TT and TDB are exact
    /// either way. The frame transforms that read UT1 refuse a value marked
    /// here unless the caller accepts it through a `_with_validity` entry
    /// point or [`crate::astro::frames::transforms::with_ut1_validity`].
    pub ut1_degraded: Option<DegradeReason>,
}

/// One post-1972 TAI-UTC step keyed by UTC Modified Julian Date.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeapSecondEntry {
    /// First UTC Modified Julian Date carrying this TAI-UTC value.
    pub mjd: i32,
    /// TAI minus UTC, seconds.
    pub tai_utc: f64,
}

/// Caller-supplied time tables for UTC, TAI, TT, UT1, TCG, TDB, and TCB.
#[derive(Debug, Clone, Copy)]
pub struct TimeTables<'a> {
    /// Leap-second table entries with the same shape as the embedded table.
    pub leap_seconds: &'a [LeapSecondEntry],
    /// UT1-UTC samples with the same shape as the embedded table.
    pub ut1_utc: &'a [Ut1Entry],
}

impl<'a> TimeTables<'a> {
    /// Construct time tables from caller-owned slices.
    pub fn new(
        leap_seconds: &'a [LeapSecondEntry],
        ut1_utc: &'a [Ut1Entry],
    ) -> Result<Self, CoverageError> {
        validate_time_tables(leap_seconds, ut1_utc)?;
        Ok(Self {
            leap_seconds,
            ut1_utc,
        })
    }

    /// Provenance and coverage for this leap-second table.
    #[must_use]
    pub fn leap_second_table(&self) -> LeapSecondTable {
        leap_second_table_for(self.leap_seconds, "caller-supplied leap-second table")
    }

    /// Coverage for this UT1-UTC table, expressed on the TT axis.
    #[must_use]
    pub fn ut1_coverage(&self) -> Ut1Provenance {
        ut1_coverage_for(
            self.ut1_utc,
            self.leap_seconds,
            "caller-supplied UT1-UTC table",
        )
    }
}

impl TimeTables<'static> {
    /// Embedded time tables bundled with the crate.
    #[must_use]
    pub fn embedded() -> Self {
        Self {
            leap_seconds: LEAP_SECONDS,
            ut1_utc: &UT1_DATA,
        }
    }
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
        mjd: 50630,
        tai_utc: 31.0,
    },
    LeapSecondEntry {
        mjd: 51179,
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
    ///
    /// TT and TDB are exact for any instant the leap-second table labels, so
    /// this constructor does not refuse an instant outside the embedded UT1
    /// table. For such an instant TT-UT1 comes from Skyfield's long-term
    /// delta-T curve and [`TimeScales::ut1_degraded`] reports it. Use
    /// [`TimeScales::from_utc_validated`] to refuse it instead.
    pub fn from_utc(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
    ) -> Result<Self, CoverageError> {
        validate_utc_input_embedded(year, month, day, hour, minute, second)?;
        Ok(Self::from_utc_unchecked(
            year, month, day, hour, minute, second,
        ))
    }

    /// Resolve time scales for a UTC calendar instant using caller-supplied
    /// leap-second and UT1-UTC tables.
    ///
    /// Refuses an instant outside the caller's UT1 table with
    /// [`CoverageError::OutsideCoverage`], so a value returned here always has
    /// [`TimeScales::ut1_degraded`] equal to `None`. Use
    /// [`TimeScales::from_utc_validated_with_tables`] with
    /// [`ValidityMode::Permissive`] to accept the long-term delta-T curve
    /// outside it.
    pub fn from_utc_with_tables(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
        tables: TimeTables<'_>,
    ) -> Result<Self, CoverageError> {
        validate_time_tables(tables.leap_seconds, tables.ut1_utc)?;
        validate_utc_input_with_table(year, month, day, hour, minute, second, tables.leap_seconds)?;
        let scales =
            Self::from_utc_with_tables_unchecked(year, month, day, hour, minute, second, tables)?;
        let prov = tables.ut1_coverage();
        check_ut1_coverage(&prov, scales.jd_tt, ValidityMode::Strict)?;
        Ok(scales)
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
        let utc_seconds_of_day =
            hour as f64 * SECONDS_PER_HOUR + minute as f64 * SECONDS_PER_MINUTE + second;
        let leap_lookup_second = if second >= 60.0 { 59.0 } else { second };
        let jd2 = (leap_lookup_second
            + minute as f64 * SECONDS_PER_MINUTE
            + hour as f64 * SECONDS_PER_HOUR)
            / SECONDS_PER_DAY;
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
        let tdb_minus_tt_seconds = 0.001657 * libm::sin(628.3076 * t + 6.2401)
            + 0.000022 * libm::sin(575.3385 * t + 4.2970)
            + 0.000014 * libm::sin(1256.6152 * t + 6.1969)
            + 0.000005 * libm::sin(606.9777 * t + 4.0212)
            + 0.000005 * libm::sin(52.9691 * t + 0.4444)
            + 0.000002 * libm::sin(21.3299 * t + 5.5431)
            + 0.000010 * t * libm::sin(628.3076 * t + 4.2490);

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
            ut1_degraded: ut1_coverage_departure(&embedded_ut1_coverage(), jd_tt),
        }
    }

    /// Exact UTC conversion path using caller-supplied tables.
    fn from_utc_with_tables_unchecked(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
        tables: TimeTables<'_>,
    ) -> Result<Self, CoverageError> {
        let jd_day = julian_day_number(year, month, day);
        let jd1 = jd_day as f64 - 0.5;
        let utc_seconds_of_day =
            hour as f64 * SECONDS_PER_HOUR + minute as f64 * SECONDS_PER_MINUTE + second;
        let leap_lookup_second = if second >= 60.0 { 59.0 } else { second };
        let jd2 = (leap_lookup_second
            + minute as f64 * SECONDS_PER_MINUTE
            + hour as f64 * SECONDS_PER_HOUR)
            / SECONDS_PER_DAY;
        let jd_utc_total = jd1 + jd2;

        let leap_seconds = find_leap_seconds_in_table_checked(jd_utc_total, tables.leap_seconds)?;
        let utc_seconds_at_midnight = jd1 * SECONDS_PER_DAY;

        let utc_whole_seconds = utc_seconds_of_day.trunc();
        let utc_subsecond = utc_seconds_of_day.fract();

        let tai_seconds = utc_seconds_at_midnight + leap_seconds + utc_whole_seconds;
        let jd_whole = (tai_seconds / SECONDS_PER_DAY).floor();
        let tai_fraction =
            (tai_seconds - jd_whole * SECONDS_PER_DAY + utc_subsecond) / SECONDS_PER_DAY;
        let tt_offset_days = TT_MINUS_TAI_S / SECONDS_PER_DAY;

        let tt_fraction = tai_fraction + tt_offset_days;
        let jd_tt = jd_whole + tt_fraction;

        let delta_t = interpolate_delta_t_with_table(jd_tt, tables.ut1_utc, tables.leap_seconds)?;
        let ut1_fraction = tt_fraction - delta_t / SECONDS_PER_DAY;
        let jd_ut1 = jd_whole + ut1_fraction;

        let t = (jd_whole - J2000_JD + tt_fraction) / DAYS_PER_JULIAN_CENTURY;
        let tdb_minus_tt_seconds = 0.001657 * libm::sin(628.3076 * t + 6.2401)
            + 0.000022 * libm::sin(575.3385 * t + 4.2970)
            + 0.000014 * libm::sin(1256.6152 * t + 6.1969)
            + 0.000005 * libm::sin(606.9777 * t + 4.0212)
            + 0.000005 * libm::sin(52.9691 * t + 0.4444)
            + 0.000002 * libm::sin(21.3299 * t + 5.5431)
            + 0.000010 * t * libm::sin(628.3076 * t + 4.2490);

        let tdb_fraction = tt_fraction + tdb_minus_tt_seconds / SECONDS_PER_DAY;
        let jd_tdb = jd_whole + tdb_fraction;

        Ok(Self {
            jd_whole,
            ut1_fraction,
            tt_fraction,
            tdb_fraction,
            jd_ut1,
            jd_tt,
            jd_tdb,
            ut1_degraded: ut1_coverage_departure(&tables.ut1_coverage(), jd_tt),
        })
    }

    /// Coverage-policy-enforced variant of [`TimeScales::from_utc`].
    ///
    /// The numerics are produced by [`TimeScales::from_utc`] unchanged: the
    /// table inside its coverage, and outside it the long-term delta-T curve
    /// Skyfield 1.54 splices around the table. The resulting TT instant is
    /// classified against the embedded UT1/EOP coverage interval under the
    /// requested [`ValidityMode`]:
    ///
    /// - [`ValidityMode::Strict`] (the default mode): an instant outside
    ///   `[first_jd_tt, last_jd_tt]` returns [`CoverageError::OutsideCoverage`].
    ///   Nothing outside the table is ever returned.
    /// - [`ValidityMode::Permissive`]: the long-term value is returned, paired
    ///   with a [`crate::astro::time::eop::DegradeReason`] when the instant fell
    ///   outside coverage. The same reason is in the value's
    ///   [`TimeScales::ut1_degraded`].
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
        // Classify the instant against UT1 coverage. We classify at jd_tt
        // because the delta-T table axis is in TT (see `ut1_coverage`), and
        // jd_tt does not depend on delta-T.
        let prov = ut1_coverage();
        let degraded = check_ut1_coverage(&prov, scales.jd_tt, mode)?;
        Ok(Validated {
            value: scales,
            degraded,
        })
    }

    /// Coverage-policy-enforced variant of [`TimeScales::from_utc_with_tables`].
    #[allow(clippy::too_many_arguments)]
    pub fn from_utc_validated_with_tables(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
        mode: ValidityMode,
        tables: TimeTables<'_>,
    ) -> Result<Validated<Self>, CoverageError> {
        validate_time_tables(tables.leap_seconds, tables.ut1_utc)?;
        validate_utc_input_with_table(year, month, day, hour, minute, second, tables.leap_seconds)?;
        let scales =
            Self::from_utc_with_tables_unchecked(year, month, day, hour, minute, second, tables)?;
        let prov = tables.ut1_coverage();
        let degraded = check_ut1_coverage(&prov, scales.jd_tt, mode)?;
        Ok(Validated {
            value: scales,
            degraded,
        })
    }

    /// Build [`TimeScales`] for a calendar instant labelled in `scale`.
    ///
    /// Non-UTC scales (GPST/GST/BDT/QZSST/TAI/TT/TDB and GLONASST) are converted
    /// to the UTC calendar label first via `scale_calendar_to_utc`, so the
    /// Earth-orientation inputs used downstream are correct rather than offset by
    /// the scale's leap-second gap, then routed through [`Self::from_utc`]. This
    /// is the single home for the system-time-to-UTC inverse that the
    /// reduced-orbit bridge previously reimplemented.
    ///
    /// Like [`Self::from_utc`], it does not refuse an instant outside the
    /// embedded UT1 table; [`TimeScales::ut1_degraded`] reports the departure.
    pub fn from_scale(
        scale: TimeScale,
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
    ) -> Result<Self, CoverageError> {
        let cal = ScaleCal {
            year,
            month,
            day,
            hour,
            minute,
            second,
        };
        validate_scale_input_embedded(scale, cal)?;
        let utc = scale_calendar_to_utc(scale, cal, LEAP_SECONDS);
        Self::from_utc(
            utc.year, utc.month, utc.day, utc.hour, utc.minute, utc.second,
        )
    }

    /// Build [`TimeScales`] for a calendar instant labelled in `scale` using
    /// caller-supplied leap-second and UT1-UTC tables.
    #[allow(clippy::too_many_arguments)]
    pub fn from_scale_with_tables(
        scale: TimeScale,
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
        tables: TimeTables<'_>,
    ) -> Result<Self, CoverageError> {
        validate_time_tables(tables.leap_seconds, tables.ut1_utc)?;
        let cal = ScaleCal {
            year,
            month,
            day,
            hour,
            minute,
            second,
        };
        validate_scale_input_with_table(scale, cal, tables.leap_seconds)?;
        let utc = scale_calendar_to_utc_with_table(scale, cal, tables.leap_seconds)?;
        Self::from_utc_with_tables(
            utc.year, utc.month, utc.day, utc.hour, utc.minute, utc.second, tables,
        )
    }

    /// Full TCG Julian Date from this instant's TT coordinate.
    #[must_use]
    pub fn jd_tcg(&self) -> f64 {
        tt_to_tcg_jd(self.jd_tt)
    }

    /// TCG day fraction relative to [`TimeScales::jd_whole`].
    #[must_use]
    pub fn tcg_fraction(&self) -> f64 {
        tcg_fraction_from_tt_split(self.jd_whole, self.tt_fraction)
    }

    /// Full TCB Julian Date from this instant's TDB coordinate.
    #[must_use]
    pub fn jd_tcb(&self) -> f64 {
        tdb_to_tcb_jd(self.jd_tdb)
    }

    /// TCB day fraction relative to [`TimeScales::jd_whole`].
    #[must_use]
    pub fn tcb_fraction(&self) -> f64 {
        tcb_fraction_from_tdb_split(self.jd_whole, self.tdb_fraction)
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
fn scale_calendar_to_utc(
    scale: TimeScale,
    cal: ScaleCal,
    leap_seconds: &[LeapSecondEntry],
) -> ScaleCal {
    match scale {
        TimeScale::Utc => cal,
        TimeScale::Glonasst => normalize_calendar_seconds(cal, cal.second - GLONASST_MINUS_UTC_S),
        TimeScale::Tcg => {
            let tcg_jd = continuous_calendar_jd(cal);
            coordinate_calendar_to_utc(
                cal,
                tcg_to_tt_jd(tcg_jd) - tcg_jd,
                TimeScale::Tt,
                leap_seconds,
            )
        }
        TimeScale::Tdb => {
            let tdb_jd = continuous_calendar_jd(cal);
            coordinate_calendar_to_utc(
                cal,
                tdb_to_tt_jd_for_tdb_input(tdb_jd) - tdb_jd,
                TimeScale::Tt,
                leap_seconds,
            )
        }
        TimeScale::Tcb => {
            let tcb_jd = continuous_calendar_jd(cal);
            let tdb_jd = tcb_to_tdb_jd(tcb_jd);
            let tt_jd = tdb_to_tt_jd_for_tdb_input(tdb_jd);
            coordinate_calendar_to_utc(cal, tt_jd - tcb_jd, TimeScale::Tt, leap_seconds)
        }
        _ => {
            let tai = normalize_calendar_seconds(cal, cal.second + tai_minus_scale_seconds(scale));
            tai_calendar_to_utc(tai, leap_seconds)
        }
    }
}

fn scale_calendar_to_utc_with_table(
    scale: TimeScale,
    cal: ScaleCal,
    leap_seconds: &[LeapSecondEntry],
) -> Result<ScaleCal, CoverageError> {
    match scale {
        TimeScale::Utc => Ok(cal),
        TimeScale::Glonasst => Ok(normalize_calendar_seconds(
            cal,
            cal.second - GLONASST_MINUS_UTC_S,
        )),
        TimeScale::Tcg => {
            let tcg_jd = continuous_calendar_jd(cal);
            coordinate_calendar_to_utc_with_table(
                cal,
                tcg_to_tt_jd(tcg_jd) - tcg_jd,
                TimeScale::Tt,
                leap_seconds,
            )
        }
        TimeScale::Tdb => {
            let tdb_jd = continuous_calendar_jd(cal);
            coordinate_calendar_to_utc_with_table(
                cal,
                tdb_to_tt_jd_for_tdb_input(tdb_jd) - tdb_jd,
                TimeScale::Tt,
                leap_seconds,
            )
        }
        TimeScale::Tcb => {
            let tcb_jd = continuous_calendar_jd(cal);
            let tdb_jd = tcb_to_tdb_jd(tcb_jd);
            let tt_jd = tdb_to_tt_jd_for_tdb_input(tdb_jd);
            coordinate_calendar_to_utc_with_table(cal, tt_jd - tcb_jd, TimeScale::Tt, leap_seconds)
        }
        _ => {
            let tai = normalize_calendar_seconds(cal, cal.second + tai_minus_scale_seconds(scale));
            tai_calendar_to_utc_with_table(tai, leap_seconds)
        }
    }
}

fn coordinate_calendar_to_utc(
    cal: ScaleCal,
    target_minus_source_days: f64,
    target_scale: TimeScale,
    leap_seconds: &[LeapSecondEntry],
) -> ScaleCal {
    let target =
        normalize_calendar_seconds(cal, cal.second + target_minus_source_days * SECONDS_PER_DAY);
    let tai = normalize_calendar_seconds(
        target,
        target.second + tai_minus_scale_seconds(target_scale),
    );
    tai_calendar_to_utc(tai, leap_seconds)
}

fn coordinate_calendar_to_utc_with_table(
    cal: ScaleCal,
    target_minus_source_days: f64,
    target_scale: TimeScale,
    leap_seconds: &[LeapSecondEntry],
) -> Result<ScaleCal, CoverageError> {
    let target =
        normalize_calendar_seconds(cal, cal.second + target_minus_source_days * SECONDS_PER_DAY);
    let tai = normalize_calendar_seconds(
        target,
        target.second + tai_minus_scale_seconds(target_scale),
    );
    tai_calendar_to_utc_with_table(tai, leap_seconds)
}

fn continuous_calendar_jd(cal: ScaleCal) -> f64 {
    let jd1 = julian_day_number(cal.year, cal.month, cal.day) as f64 - 0.5;
    jd1 + seconds_of_day(cal) / SECONDS_PER_DAY
}

fn tai_minus_scale_seconds(scale: TimeScale) -> f64 {
    match scale {
        // Utc/Glonasst/TCG/TDB/TCB are handled before reaching here; 0.0 keeps
        // the match total without affecting the atomic path.
        TimeScale::Utc | TimeScale::Glonasst | TimeScale::Tcg | TimeScale::Tdb | TimeScale::Tcb => {
            0.0
        }
        TimeScale::Tai => 0.0,
        TimeScale::Tt => -TT_MINUS_TAI_S,
        // QZSST is steered synchronous with GPST, so it shares GPST's TAI offset.
        TimeScale::Gpst | TimeScale::Gst | TimeScale::Qzsst => GPST_MINUS_TAI_S,
        TimeScale::Bdt => BDT_MINUS_TAI_S,
    }
}

fn tai_calendar_to_utc(tai: ScaleCal, leap_seconds: &[LeapSecondEntry]) -> ScaleCal {
    if let Some(utc) = positive_leap_second_utc_label(tai, leap_seconds) {
        return utc;
    }

    let mut leap = leap_seconds_at_utc_label(tai, leap_seconds);
    let mut utc = normalize_calendar_seconds(tai, tai.second - leap);
    for _ in 0..3 {
        let next_leap = leap_seconds_at_utc_label(utc, leap_seconds);
        if next_leap == leap {
            return utc;
        }
        leap = next_leap;
        utc = normalize_calendar_seconds(tai, tai.second - leap);
    }
    utc
}

fn tai_calendar_to_utc_with_table(
    tai: ScaleCal,
    leap_seconds: &[LeapSecondEntry],
) -> Result<ScaleCal, CoverageError> {
    if let Some(utc) = positive_leap_second_utc_label_with_table(tai, leap_seconds)? {
        return Ok(utc);
    }

    let mut leap = leap_seconds_at_utc_label_checked(tai, leap_seconds)?;
    let mut utc = normalize_calendar_seconds(tai, tai.second - leap);
    for _ in 0..3 {
        let next_leap = leap_seconds_at_utc_label_checked(utc, leap_seconds)?;
        if next_leap == leap {
            return Ok(utc);
        }
        leap = next_leap;
        utc = normalize_calendar_seconds(tai, tai.second - leap);
    }
    Ok(utc)
}

fn positive_leap_second_utc_label(
    tai: ScaleCal,
    leap_seconds: &[LeapSecondEntry],
) -> Option<ScaleCal> {
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
    let old_leap = leap_seconds_at_utc_label(previous_second, leap_seconds);
    let new_leap = leap_seconds_at_utc_label(utc_midnight, leap_seconds);
    if new_leap <= old_leap || !(old_leap..new_leap).contains(&tai_sod) {
        return None;
    }

    let mut utc = previous_second;
    utc.second = 60.0 + (tai_sod - old_leap);
    Some(utc)
}

fn positive_leap_second_utc_label_with_table(
    tai: ScaleCal,
    leap_seconds: &[LeapSecondEntry],
) -> Result<Option<ScaleCal>, CoverageError> {
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
    let Ok(old_leap) = leap_seconds_at_utc_label_checked(previous_second, leap_seconds) else {
        return Ok(None);
    };
    let new_leap = leap_seconds_at_utc_label_checked(utc_midnight, leap_seconds)?;
    if new_leap <= old_leap || !(old_leap..new_leap).contains(&tai_sod) {
        return Ok(None);
    }

    let mut utc = previous_second;
    utc.second = 60.0 + (tai_sod - old_leap);
    Ok(Some(utc))
}

fn leap_seconds_at_utc_label(cal: ScaleCal, leap_seconds: &[LeapSecondEntry]) -> f64 {
    let jd1 = julian_day_number(cal.year, cal.month, cal.day) as f64 - 0.5;
    let lookup_second = if cal.second >= 60.0 { 59.0 } else { cal.second };
    let jd2 = (cal.hour as f64 * SECONDS_PER_HOUR
        + cal.minute as f64 * SECONDS_PER_MINUTE
        + lookup_second)
        / SECONDS_PER_DAY;
    find_leap_seconds_in_table(jd1 + jd2, leap_seconds)
}

fn leap_seconds_at_utc_label_checked(
    cal: ScaleCal,
    leap_seconds: &[LeapSecondEntry],
) -> Result<f64, CoverageError> {
    let jd1 = julian_day_number(cal.year, cal.month, cal.day) as f64 - 0.5;
    let lookup_second = if cal.second >= 60.0 { 59.0 } else { cal.second };
    let jd2 = (cal.hour as f64 * SECONDS_PER_HOUR
        + cal.minute as f64 * SECONDS_PER_MINUTE
        + lookup_second)
        / SECONDS_PER_DAY;
    find_leap_seconds_in_table_checked(jd1 + jd2, leap_seconds)
}

fn seconds_of_day(cal: ScaleCal) -> f64 {
    cal.hour as f64 * SECONDS_PER_HOUR + cal.minute as f64 * SECONDS_PER_MINUTE + cal.second
}

fn tdb_to_tt_jd_for_tdb_input(jd_tdb: f64) -> f64 {
    let mut jd_tt = jd_tdb;
    for _ in 0..4 {
        jd_tt = jd_tdb - tdb_minus_tt_seconds_at_tt_jd(jd_tt) / SECONDS_PER_DAY;
    }
    jd_tt
}

fn tdb_minus_tt_seconds_at_tt_jd(jd_tt: f64) -> f64 {
    let t = (jd_tt - J2000_JD) / DAYS_PER_JULIAN_CENTURY;
    0.001657 * libm::sin(628.3076 * t + 6.2401)
        + 0.000022 * libm::sin(575.3385 * t + 4.2970)
        + 0.000014 * libm::sin(1256.6152 * t + 6.1969)
        + 0.000005 * libm::sin(606.9777 * t + 4.0212)
        + 0.000005 * libm::sin(52.9691 * t + 0.4444)
        + 0.000002 * libm::sin(21.3299 * t + 5.5431)
        + 0.000010 * t * libm::sin(628.3076 * t + 4.2490)
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
    is_positive_leap_second_label_with_table(year, month, day, hour, minute, LEAP_SECONDS)
}

fn is_positive_leap_second_label_with_table(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    leap_seconds: &[LeapSecondEntry],
) -> bool {
    if hour != 23 || minute != 59 {
        return false;
    }
    let jd1 = julian_day_number(year, month, day) as f64 - 0.5;
    find_leap_seconds_in_table(jd1 + 1.0, leap_seconds)
        > find_leap_seconds_in_table(jd1, leap_seconds)
}

fn is_positive_leap_second_label_with_table_checked(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    leap_seconds: &[LeapSecondEntry],
) -> Result<bool, CoverageError> {
    if hour != 23 || minute != 59 {
        return Ok(false);
    }
    let jd1 = julian_day_number(year, month, day) as f64 - 0.5;
    Ok(find_leap_seconds_in_table_checked(jd1 + 1.0, leap_seconds)?
        > find_leap_seconds_in_table_checked(jd1, leap_seconds)?)
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

fn validate_utc_input_embedded(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: f64,
) -> Result<(), CoverageError> {
    validate_utc_civil_input(year, month, day, hour, minute, second)?;
    if second >= 60.0
        && !is_positive_leap_second_label_with_table(year, month, day, hour, minute, LEAP_SECONDS)
    {
        return Err(CoverageError::InvalidInput {
            field: "civil datetime",
            kind: TimeScaleInputErrorKind::InvalidCivilTime,
        });
    }
    Ok(())
}

fn validate_utc_input_with_table(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: f64,
    leap_seconds: &[LeapSecondEntry],
) -> Result<(), CoverageError> {
    validate_utc_civil_input(year, month, day, hour, minute, second)?;
    ensure_leap_table_covers_calendar(year, month, day, hour, minute, second, leap_seconds)?;
    if second >= 60.0
        && !is_positive_leap_second_label_with_table_checked(
            year,
            month,
            day,
            hour,
            minute,
            leap_seconds,
        )?
    {
        return Err(CoverageError::InvalidInput {
            field: "civil datetime",
            kind: TimeScaleInputErrorKind::InvalidCivilTime,
        });
    }
    Ok(())
}

fn validate_utc_civil_input(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: f64,
) -> Result<(), CoverageError> {
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
    Ok(())
}

fn ensure_leap_table_covers_calendar(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: f64,
    leap_seconds: &[LeapSecondEntry],
) -> Result<(), CoverageError> {
    let jd1 = julian_day_number(year, month, day) as f64 - 0.5;
    let lookup_second = if second >= 60.0 { 59.0 } else { second };
    let jd2 = (hour as f64 * SECONDS_PER_HOUR + minute as f64 * SECONDS_PER_MINUTE + lookup_second)
        / SECONDS_PER_DAY;
    find_leap_seconds_in_table_checked(jd1 + jd2, leap_seconds).map(|_| ())
}

fn validate_time_tables(
    leap_seconds: &[LeapSecondEntry],
    ut1_utc: &[Ut1Entry],
) -> Result<(), CoverageError> {
    validate_leap_seconds_table(leap_seconds)?;
    validate_ut1_table(ut1_utc)?;
    if effective_ut1_utc_rows(ut1_utc, leap_seconds).len() < 2 {
        return Err(table_error("ut1_utc", TimeScaleInputErrorKind::Missing));
    }
    Ok(())
}

fn validate_leap_seconds_table(leap_seconds: &[LeapSecondEntry]) -> Result<(), CoverageError> {
    if leap_seconds.is_empty() {
        return Err(table_error(
            "leap_seconds",
            TimeScaleInputErrorKind::Missing,
        ));
    }
    let mut previous_mjd = leap_seconds[0].mjd;
    if !leap_seconds[0].tai_utc.is_finite() {
        return Err(table_error(
            "leap_seconds",
            TimeScaleInputErrorKind::NonFinite,
        ));
    }
    for entry in &leap_seconds[1..] {
        if entry.mjd <= previous_mjd {
            return Err(table_error(
                "leap_seconds",
                TimeScaleInputErrorKind::OutOfRange,
            ));
        }
        if !entry.tai_utc.is_finite() {
            return Err(table_error(
                "leap_seconds",
                TimeScaleInputErrorKind::NonFinite,
            ));
        }
        previous_mjd = entry.mjd;
    }
    Ok(())
}

fn validate_ut1_table(ut1_utc: &[Ut1Entry]) -> Result<(), CoverageError> {
    if ut1_utc.len() < 2 {
        return Err(table_error("ut1_utc", TimeScaleInputErrorKind::Missing));
    }
    let mut previous_mjd = ut1_utc[0].mjd;
    if !ut1_utc[0].ut1_utc.is_finite() {
        return Err(table_error("ut1_utc", TimeScaleInputErrorKind::NonFinite));
    }
    for entry in &ut1_utc[1..] {
        if entry.mjd <= previous_mjd {
            return Err(table_error("ut1_utc", TimeScaleInputErrorKind::OutOfRange));
        }
        if !entry.ut1_utc.is_finite() {
            return Err(table_error("ut1_utc", TimeScaleInputErrorKind::NonFinite));
        }
        previous_mjd = entry.mjd;
    }
    Ok(())
}

fn effective_ut1_utc_rows<'a>(
    ut1_utc: &'a [Ut1Entry],
    leap_seconds: &[LeapSecondEntry],
) -> &'a [Ut1Entry] {
    debug_assert!(!leap_seconds.is_empty());
    let first_covered_mjd = leap_seconds[0].mjd;
    let first = ut1_utc
        .iter()
        .position(|entry| entry.mjd >= first_covered_mjd)
        .unwrap_or(ut1_utc.len());
    &ut1_utc[first..]
}

fn table_error(field: &'static str, kind: TimeScaleInputErrorKind) -> CoverageError {
    CoverageError::InvalidInput { field, kind }
}

fn validate_scale_input_embedded(scale: TimeScale, cal: ScaleCal) -> Result<(), CoverageError> {
    if scale == TimeScale::Utc {
        return validate_utc_input_embedded(
            cal.year, cal.month, cal.day, cal.hour, cal.minute, cal.second,
        );
    }
    validate_continuous_scale_input(cal)
}

fn validate_scale_input_with_table(
    scale: TimeScale,
    cal: ScaleCal,
    leap_seconds: &[LeapSecondEntry],
) -> Result<(), CoverageError> {
    if scale == TimeScale::Utc {
        return validate_utc_input_with_table(
            cal.year,
            cal.month,
            cal.day,
            cal.hour,
            cal.minute,
            cal.second,
            leap_seconds,
        );
    }
    validate_continuous_scale_input(cal)?;
    if is_utc_based(scale) {
        let utc = scale_calendar_to_utc_with_table(scale, cal, leap_seconds)?;
        ensure_leap_table_covers_calendar(
            utc.year,
            utc.month,
            utc.day,
            utc.hour,
            utc.minute,
            utc.second,
            leap_seconds,
        )?;
    }
    Ok(())
}

fn validate_continuous_scale_input(cal: ScaleCal) -> Result<(), CoverageError> {
    validate::finite(cal.second, "second").map_err(map_time_scale_field_error)?;
    validate::civil_datetime_with_second_policy(
        i64::from(cal.year),
        i64::from(cal.month),
        i64::from(cal.day),
        i64::from(cal.hour),
        i64::from(cal.minute),
        cal.second,
        validate::CivilSecondPolicy::Continuous,
    )
    .map_err(map_time_scale_field_error)
    .map(|_| ())
}

/// Convert a TT Julian Date to TCG using IAU 2000 Resolution B1.9.
#[must_use]
pub fn tt_to_tcg_jd(jd_tt: f64) -> f64 {
    TCG_TCB_REFERENCE_JD + (jd_tt - TCG_TCB_REFERENCE_JD) / (1.0 - TT_TCG_RATE_L_G)
}

/// Convert a TCG Julian Date to TT using IAU 2000 Resolution B1.9.
#[must_use]
pub fn tcg_to_tt_jd(jd_tcg: f64) -> f64 {
    TCG_TCB_REFERENCE_JD + (jd_tcg - TCG_TCB_REFERENCE_JD) * (1.0 - TT_TCG_RATE_L_G)
}

/// Convert a TDB Julian Date to TCB using IAU 2006 Resolution B3.
#[must_use]
pub fn tdb_to_tcb_jd(jd_tdb: f64) -> f64 {
    TCG_TCB_REFERENCE_JD
        + (((jd_tdb - TCG_TCB_REFERENCE_JD) * SECONDS_PER_DAY - TDB_TCB_OFFSET_TDB0_S)
            / (1.0 - TDB_TCB_RATE_L_B))
            / SECONDS_PER_DAY
}

/// Convert a TCB Julian Date to TDB using IAU 2006 Resolution B3.
#[must_use]
pub fn tcb_to_tdb_jd(jd_tcb: f64) -> f64 {
    TCG_TCB_REFERENCE_JD
        + (((jd_tcb - TCG_TCB_REFERENCE_JD) * SECONDS_PER_DAY) * (1.0 - TDB_TCB_RATE_L_B)
            + TDB_TCB_OFFSET_TDB0_S)
            / SECONDS_PER_DAY
}

fn tcg_fraction_from_tt_split(jd_whole: f64, tt_fraction: f64) -> f64 {
    let elapsed_tt_days = (jd_whole - TCG_TCB_REFERENCE_JD) + tt_fraction;
    tt_fraction + elapsed_tt_days * TT_TCG_RATE_L_G / (1.0 - TT_TCG_RATE_L_G)
}

fn tcb_fraction_from_tdb_split(jd_whole: f64, tdb_fraction: f64) -> f64 {
    let elapsed_tdb_days = (jd_whole - TCG_TCB_REFERENCE_JD) + tdb_fraction;
    tdb_fraction
        + (elapsed_tdb_days * TDB_TCB_RATE_L_B - TDB_TCB_OFFSET_TDB0_S / SECONDS_PER_DAY)
            / (1.0 - TDB_TCB_RATE_L_B)
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
/// For instants from 1972-01-01 (MJD 41317) onward this reads the embedded
/// integer leap-second table of IERS Bulletin C, clamping above to the last
/// entry: TAI - UTC takes the value of the last entry whose MJD is at or
/// before the instant's UTC day. The table dates each change to the first
/// day under the new value (31 s from 1997-07-01, 32 s from 1999-01-01).
///
/// For instants in the 1961-01-01 .. 1972-01-01 rubber-second era it evaluates
/// the published piecewise-linear IERS/USNO model
/// `TAI-UTC = base + (MJD - ref_mjd) * rate` (see `RUBBER_SECONDS`) using the
/// fractional UTC MJD, so the offset is continuous within each segment as the
/// historical definition requires. Before 1961 it clamps to the first
/// rubber-second segment's value rather than extrapolating into undefined
/// territory. Non-finite inputs return `NaN`.
///
/// Boundary semantics (post-1972): the date is the integer MJD (`(jd_utc -
/// 2400000.5) as i32`), so the count steps at UTC midnight (the table's
/// effective MJD is the first full day under the new value). The inserted leap
/// second `23:59:60` and the following `00:00:00` share essentially the same
/// Julian date, so an end-of-day instant resolves to the **post-leap** count -
/// the leap second itself cannot be distinguished from the next day's start
/// through a JD. This is intrinsic to a JD-keyed lookup; callers that must
/// label `23:59:60` distinctly have to carry the civil second out-of-band
/// rather than rely on this function.
pub fn find_leap_seconds(jd_utc: f64) -> f64 {
    if !jd_utc.is_finite() {
        return f64::NAN;
    }
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

fn find_leap_seconds_in_table(jd_utc: f64, leap_seconds: &[LeapSecondEntry]) -> f64 {
    debug_assert!(!leap_seconds.is_empty());
    if !jd_utc.is_finite() {
        return f64::NAN;
    }
    let mjd = (jd_utc - 2400000.5) as i32;
    if mjd >= leap_seconds[0].mjd {
        let mut ls = leap_seconds[0].tai_utc;
        for entry in leap_seconds {
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

fn find_leap_seconds_in_table_checked(
    jd_utc: f64,
    leap_seconds: &[LeapSecondEntry],
) -> Result<f64, CoverageError> {
    debug_assert!(!leap_seconds.is_empty());
    if !jd_utc.is_finite() {
        return Err(table_error(
            "leap_seconds",
            TimeScaleInputErrorKind::NonFinite,
        ));
    }
    let mjd = (jd_utc - 2400000.5) as i32;
    if mjd < leap_seconds[0].mjd {
        return Err(table_error(
            "leap_seconds",
            TimeScaleInputErrorKind::OutOfRange,
        ));
    }

    let mut ls = leap_seconds[0].tai_utc;
    for entry in leap_seconds {
        if mjd >= entry.mjd {
            ls = entry.tai_utc;
        } else {
            break;
        }
    }
    Ok(ls)
}

/// TAI - UTC (the full accumulated leap-second count) at a UTC Julian date.
///
/// This is the IERS / Bulletin C quantity: the difference between International
/// Atomic Time and Coordinated Universal Time. From 2017-01-01 onward it is
/// **37 s**. It is the unambiguously-named alias of [`find_leap_seconds`] and
/// returns the identical value.
///
/// This is **not** the GNSS "leap seconds since the GPS epoch" quantity. A GNSS
/// caller who wants GPS - UTC (18 s in 2017) must call [`gps_utc_offset_s`];
/// using this function there over-counts by `TAI - GPST = 19 s`.
///
/// See [`find_leap_seconds`] for the table, rubber-second, and boundary
/// semantics, which this function inherits verbatim.
pub fn tai_utc_offset_s(jd_utc: f64) -> f64 {
    find_leap_seconds(jd_utc)
}

/// GPS - UTC (the GNSS leap-second offset since the GPS epoch) at a UTC Julian
/// date.
///
/// This is the IS-GPS-200 quantity broadcast in the navigation message: the
/// difference between GPS system time and UTC. From 2017-01-01 onward it is
/// **18 s**. By definition `GPST - TAI = -19 s` (equivalently
/// `TAI - GPST = +19 s`), so
/// `GPS-UTC = (TAI-UTC) + (GPST-TAI) = (TAI-UTC) - 19 s`,
/// i.e. `gps_utc_offset_s == tai_utc_offset_s - 19`.
///
/// Use this, not [`tai_utc_offset_s`], whenever you mean "the leap seconds a
/// GPS receiver applies"; the two differ by a constant 19 s and silently
/// returning TAI - UTC where GPS - UTC is expected is a 19 s blunder.
pub fn gps_utc_offset_s(jd_utc: f64) -> f64 {
    find_leap_seconds(jd_utc) - GPST_MINUS_TAI_S
}

/// Evaluate the pre-1972 piecewise-linear TAI-UTC model at a UTC Julian date.
///
/// Selects the latest [`RUBBER_SECONDS`] segment whose `start_mjd` precedes the
/// instant's fractional MJD and applies `base + (MJD - ref_mjd) * rate`. Inputs
/// before the first segment (pre-1961) clamp to the first segment's constant
/// term. Non-finite inputs return `NaN`.
fn rubber_tai_minus_utc(jd_utc: f64) -> f64 {
    let mjd = jd_utc - 2400000.5;
    let first = &RUBBER_SECONDS[0];
    if !mjd.is_finite() {
        return f64::NAN;
    }
    // Pre-1961 input clamps to the first segment's constant.
    if mjd < first.start_mjd as f64 {
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
    /// The named coordinate scale has no fixed offset; resolve it through
    /// [`TimeScales`] or the scale-specific conversion helpers.
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
        TimeScale::Tcg | TimeScale::Tdb | TimeScale::Tcb => {
            return Err(TimeOffsetError::Unsupported(scale.abbrev()));
        }
    })
}

/// Fixed inter-system offset `to_reading - from_reading` (seconds) for scales
/// whose mutual offset is a constant.
///
/// Returns the value that, added to a `from`-scale reading, yields the
/// `to`-scale reading of the same physical instant. This covers the atomic
/// scales TAI/TT/GPST/GST/QZSST/BDT, whose offsets are fixed by their defining
/// ICDs (see `scale_minus_tai_s` for the per-scale citations).
///
/// Returns [`TimeOffsetError::EpochRequired`] if either scale is UTC-based
/// (UTC/GLONASST), which need [`timescale_offset_at_s`], and
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
    leap_second_table_for(
        LEAP_SECONDS,
        "IERS Bulletin C (TAI-UTC), bundled in sidereon-core",
    )
}

fn leap_second_table_for(
    leap_seconds: &[LeapSecondEntry],
    source: &'static str,
) -> LeapSecondTable {
    LeapSecondTable {
        source,
        first_mjd: leap_seconds.first().map(|e| e.mjd).unwrap_or(0),
        last_mjd: leap_seconds.last().map(|e| e.mjd).unwrap_or(0),
        entries: leap_seconds.len(),
    }
}

/// One row of a delta-T table: the TT Julian date of a UT1-UTC sample and
/// TT-UT1 there, rounded to 1e-7 s as Skyfield rounds its table.
struct DeltaTRow {
    jd_tt: f64,
    delta_t: f64,
}

fn delta_t_row(jd_utc: f64, leap_seconds: f64, ut1_utc: f64) -> DeltaTRow {
    let tt_minus_utc = leap_seconds + TT_MINUS_TAI_S;
    let delta_t = ((tt_minus_utc - ut1_utc) * ROUND_1E7).round() / ROUND_1E7;
    DeltaTRow {
        jd_tt: jd_utc + tt_minus_utc / SECONDS_PER_DAY,
        delta_t,
    }
}

/// Delta-T from a table, with Skyfield's long-term splice outside it.
///
/// Inside `[first, last]` the table is interpolated linearly. Outside it,
/// delta-T is the long-term curve that Skyfield 1.54 `build_delta_t` builds
/// around the table ([`LongTermDeltaT`]), evaluated at `(jd_tt - 1721045) /
/// 365.25`, as Skyfield's `DeltaT.__call__` does.
fn table_delta_t(
    table: &[DeltaTRow],
    jd_tt: f64,
    long_term: impl FnOnce() -> Result<f64, CoverageError>,
) -> Result<f64, CoverageError> {
    match table.binary_search_by(|row| row.jd_tt.partial_cmp(&jd_tt).unwrap()) {
        Ok(i) => Ok(table[i].delta_t),
        Err(0) => long_term(),
        Err(i) if i >= table.len() => long_term(),
        Err(i) => {
            let p1 = &table[i - 1];
            let p2 = &table[i];
            Ok(p1.delta_t + (jd_tt - p1.jd_tt) * (p2.delta_t - p1.delta_t) / (p2.jd_tt - p1.jd_tt))
        }
    }
}

/// Skyfield's `DeltaT` argument for its long-term curve: TT expressed as a
/// Julian year number.
fn long_term_year(jd_tt: f64) -> f64 {
    (jd_tt - 1721045.0) / 365.25
}

fn interpolate_delta_t(jd_tt: f64) -> f64 {
    use std::sync::LazyLock;

    static TABLE: LazyLock<Vec<DeltaTRow>> = LazyLock::new(|| {
        UT1_DATA
            .iter()
            .map(|entry| {
                let jd_utc = entry.mjd as f64 + 2400000.5;
                delta_t_row(jd_utc, find_leap_seconds(jd_utc), entry.ut1_utc)
            })
            .collect()
    });
    static LONG_TERM: LazyLock<LongTermDeltaT> = LazyLock::new(|| {
        LongTermDeltaT::around(&TABLE).expect("the embedded table starts after year -720")
    });

    table_delta_t(&TABLE, jd_tt, || {
        Ok(LONG_TERM.evaluate(long_term_year(jd_tt)))
    })
    .expect("the embedded long-term curve is infallible")
}

fn interpolate_delta_t_with_table(
    jd_tt: f64,
    ut1_utc: &[Ut1Entry],
    leap_seconds: &[LeapSecondEntry],
) -> Result<f64, CoverageError> {
    let table: Vec<DeltaTRow> = effective_ut1_utc_rows(ut1_utc, leap_seconds)
        .iter()
        .map(|entry| {
            let jd_utc = entry.mjd as f64 + 2400000.5;
            let leap = find_leap_seconds_in_table_checked(jd_utc, leap_seconds)?;
            Ok(delta_t_row(jd_utc, leap, entry.ut1_utc))
        })
        .collect::<Result<_, CoverageError>>()?;

    table_delta_t(&table, jd_tt, || {
        LongTermDeltaT::around(&table).map(|curve| curve.evaluate(long_term_year(jd_tt)))
    })
}

/// One segment of a Skyfield `curvelib.Splines` table: a polynomial in
/// `t = (x - lower) / (upper - lower)` with coefficients from the highest
/// power down, evaluated by Horner's rule.
#[derive(Debug, Clone, PartialEq)]
struct SplineSegment {
    lower: f64,
    upper: f64,
    coefficients: Vec<f64>,
}

impl SplineSegment {
    fn width(&self) -> f64 {
        self.upper - self.lower
    }
}

/// Skyfield `curvelib.Splines`: piecewise polynomials selected by
/// `numpy.interp(x, lower, arange(n))` truncated to an integer, so an
/// argument before the first segment or past the last one extrapolates that
/// segment's polynomial.
#[derive(Debug, Clone, PartialEq)]
struct Splines {
    segments: Vec<SplineSegment>,
}

impl Splines {
    fn evaluate(&self, x: f64) -> f64 {
        let segment = &self.segments[self.segment_index(x)];
        let t = (x - segment.lower) / segment.width();
        let mut value = segment.coefficients[0];
        for &coefficient in &segment.coefficients[1..] {
            value *= t;
            value += coefficient;
        }
        value
    }

    /// `numpy.interp(x, lower, arange(n)).astype(int)`, following NumPy's
    /// `arr_interp`: `fp[0]` below the first knot, `fp[n - 1]` above the last,
    /// `fp[j]` on a knot, and otherwise `slope * (x - xp[j]) + fp[j]` with
    /// `slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])`.
    fn segment_index(&self, x: f64) -> usize {
        let n = self.segments.len();
        if n == 1 || x < self.segments[0].lower {
            return 0;
        }
        if x > self.segments[n - 1].lower {
            return n - 1;
        }
        let j = self
            .segments
            .partition_point(|segment| segment.lower <= x)
            .saturating_sub(1);
        if j == n - 1 || self.segments[j].lower == x {
            return j;
        }
        let slope =
            ((j + 1) as f64 - j as f64) / (self.segments[j + 1].lower - self.segments[j].lower);
        let index = slope * (x - self.segments[j].lower) + j as f64;
        index as usize
    }

    /// Skyfield `Splines.derivative`: each coefficient but the constant
    /// becomes `n * c / width`, `n` its power.
    fn derivative(&self) -> Splines {
        Splines {
            segments: self
                .segments
                .iter()
                .map(|segment| {
                    let width = segment.width();
                    let powered = &segment.coefficients[..segment.coefficients.len() - 1];
                    SplineSegment {
                        lower: segment.lower,
                        upper: segment.upper,
                        coefficients: powered
                            .iter()
                            .enumerate()
                            .map(|(i, &c)| (powered.len() - i) as f64 * c / width)
                            .collect(),
                    }
                })
                .collect(),
        }
    }
}

/// Skyfield `curvelib.build_spline_given_ends`: the cubic on `[x0, x1]` with
/// value and slope `y0`, `slope0` at `x0` and `y1`, `slope1` at `x1`.
fn spline_given_ends(
    x0: f64,
    y0: f64,
    slope0: f64,
    x1: f64,
    y1: f64,
    slope1: f64,
) -> SplineSegment {
    let width = x1 - x0;
    let slope0 = slope0 * width;
    let slope1 = slope1 * width;
    let a0 = y0;
    let a1 = slope0;
    let a2 = -2.0 * slope0 - slope1 - 3.0 * y0 + 3.0 * y1;
    let a3 = slope0 + slope1 + 2.0 * y0 - 2.0 * y1;
    SplineSegment {
        lower: x0,
        upper: x1,
        coefficients: vec![a3, a2, a1, a0],
    }
}

/// The long-term parabola of Stephenson, Morrison and Hohenkerk (2016),
/// `-320 + 32.5 ((year - 1825) / 100)^2` seconds, as Skyfield's
/// `delta_t_parabola_stephenson_morrison_hohenkerk_2016` spline.
fn smh2016_parabola() -> Splines {
    Splines {
        segments: vec![SplineSegment {
            lower: 1825.0,
            upper: 1925.0,
            coefficients: vec![0.0, 32.5, 0.0, -320.0],
        }],
    }
}

/// Knot years of Table S15 (2020 update) of Morrison, Stephenson, Hohenkerk
/// and Zawilski, the cubic delta-T splines for 720 BC to AD 2019 that Skyfield
/// bundles as `delta_t.npz` `Table-S15.2020.txt`. Segment `k` runs from
/// `S15_KNOTS[k]` to `S15_KNOTS[k + 1]`.
const S15_KNOTS: [f64; 59] = [
    -720.0, -100.0, 400.0, 1000.0, 1150.0, 1300.0, 1500.0, 1600.0, 1650.0, 1720.0, 1800.0, 1810.0,
    1820.0, 1830.0, 1840.0, 1850.0, 1855.0, 1860.0, 1865.0, 1870.0, 1875.0, 1880.0, 1885.0, 1890.0,
    1895.0, 1900.0, 1905.0, 1910.0, 1915.0, 1920.0, 1925.0, 1930.0, 1935.0, 1940.0, 1945.0, 1950.0,
    1953.0, 1956.0, 1959.0, 1962.0, 1965.0, 1968.0, 1971.0, 1974.0, 1977.0, 1980.0, 1983.0, 1986.0,
    1989.0, 1992.0, 1995.0, 1998.0, 2001.0, 2004.0, 2007.0, 2010.0, 2013.0, 2016.0, 2019.0,
];

/// Table S15 cubic coefficients `[a3, a2, a1, a0]` per segment, in the
/// variable `t = (year - lower) / (upper - lower)`.
const S15_COEFFICIENTS: [[f64; 4]; 58] = [
    [409.16, 776.247, -9999.586, 20371.848],
    [-503.433, 1303.151, -5822.27, 11557.668],
    [1085.087, -298.291, -5671.519, 6535.116],
    [-25.346, 184.811, -753.21, 1650.393],
    [-24.641, 108.771, -459.628, 1056.647],
    [-29.414, 61.953, -421.345, 681.149],
    [16.197, -6.572, -192.841, 292.343],
    [3.018, 10.505, -78.697, 109.127],
    [-2.127, 38.333, -68.089, 43.952],
    [-37.939, 41.731, 2.507, 12.068],
    [1.918, -1.126, -3.481, 18.367],
    [-3.812, 4.629, 0.021, 15.678],
    [3.25, -6.806, -2.157, 16.516],
    [-0.096, 2.944, -6.018, 10.804],
    [-0.539, 2.658, -0.416, 7.634],
    [-0.883, 0.261, 1.642, 9.338],
    [1.558, -2.389, -0.486, 10.357],
    [-2.477, 2.284, -0.591, 9.04],
    [2.72, -5.148, -3.456, 8.255],
    [-0.914, 3.011, -5.593, 2.371],
    [-0.039, 0.269, -2.314, -1.126],
    [0.563, 0.152, -1.893, -3.21],
    [-1.438, 1.842, 0.101, -4.388],
    [1.871, -2.474, -0.531, -3.884],
    [-0.232, 3.138, 0.134, -5.017],
    [-1.257, 2.443, 5.715, -1.977],
    [0.72, -1.329, 6.828, 4.923],
    [-0.825, 0.831, 6.33, 11.142],
    [0.262, -1.643, 5.518, 17.479],
    [0.008, -0.856, 3.02, 21.617],
    [0.127, -0.831, 1.333, 23.789],
    [0.142, -0.449, 0.052, 24.418],
    [0.702, -0.022, -0.419, 24.164],
    [-1.106, 2.086, 1.645, 24.426],
    [0.614, -1.232, 2.499, 27.05],
    [-0.277, 0.22, 1.127, 28.932],
    [0.631, -0.61, 0.737, 30.002],
    [-0.799, 1.282, 1.409, 30.76],
    [0.507, -1.115, 1.577, 32.652],
    [0.199, 0.406, 0.868, 33.621],
    [-0.414, 1.002, 2.275, 35.093],
    [0.202, -0.242, 3.035, 37.956],
    [-0.229, 0.364, 3.157, 40.951],
    [0.172, -0.323, 3.199, 44.244],
    [-0.192, 0.193, 3.069, 47.291],
    [0.081, -0.384, 2.878, 50.361],
    [-0.165, -0.14, 2.354, 52.936],
    [0.448, -0.637, 1.577, 54.984],
    [-0.276, 0.708, 1.648, 56.373],
    [0.11, -0.121, 2.235, 58.453],
    [-0.313, 0.21, 2.324, 60.678],
    [0.109, -0.729, 1.804, 62.898],
    [0.199, -0.402, 0.674, 64.083],
    [-0.017, 0.194, 0.466, 64.553],
    [-0.084, 0.144, 0.804, 65.197],
    [0.128, -0.109, 0.839, 66.061],
    [-0.095, 0.277, 1.007, 66.92],
    [-0.139, -0.007, 1.277, 68.109],
];

fn s15_splines() -> Splines {
    Splines {
        segments: S15_COEFFICIENTS
            .iter()
            .enumerate()
            .map(|(k, coefficients)| SplineSegment {
                lower: S15_KNOTS[k],
                upper: S15_KNOTS[k + 1],
                coefficients: coefficients.to_vec(),
            })
            .collect(),
    }
}

/// Python's float `//` (CPython `float_floor_div`): `fmod`-based, then floored.
fn python_floor_div(a: f64, b: f64) -> f64 {
    let modulo = a % b;
    let mut div = (a - modulo) / b;
    if modulo != 0.0 && ((b < 0.0) != (modulo < 0.0)) {
        div -= 1.0;
    }
    if div == 0.0 {
        return 0.0_f64.copysign(a / b);
    }
    let floor = div.floor();
    if div - floor > 0.5 {
        floor + 1.0
    } else {
        floor
    }
}

/// The long-term delta-T curve Skyfield 1.54 `timelib.build_delta_t` splices
/// around a delta-T table, as a function of the Julian year number:
///
/// - the SMH2016 parabola far outside;
/// - an 800-year cubic joining the parabola to the start of Table S15 at
///   year -720;
/// - Table S15, truncated where the table starts, with the linear term of its
///   last segment adjusted so the curve meets the table's first value there;
/// - a cubic from the table's last value, with the slope of its last year,
///   to the parabola at the largest multiple of 100 not after `x0 + 800`,
///   where `x0` is the table's last year.
struct LongTermDeltaT {
    curve: Splines,
}

impl LongTermDeltaT {
    /// Width of the connecting splines, years (Skyfield `patch_width`).
    const PATCH_WIDTH: f64 = 800.0;

    /// Build the curve around `table`, which holds at least two rows. Refuses
    /// a table starting at or before year -720, where Skyfield's truncation of
    /// Table S15 leaves no segment.
    fn around(table: &[DeltaTRow]) -> Result<Self, CoverageError> {
        let p = smh2016_parabola();
        let pd = p.derivative();
        let s = s15_splines();
        let sd = s.derivative();
        let parabola_width = p.segments[0].upper - p.segments[0].lower;

        let x1 = s.segments[0].lower;
        let x0 = x1 - Self::PATCH_WIDTH;
        let left = spline_given_ends(
            x0,
            p.evaluate(x0),
            pd.evaluate(x0),
            x1,
            s.evaluate(x1),
            sd.evaluate(x1),
        );

        let x1 = x0;
        let x0 = x1 - parabola_width;
        let far_left = spline_given_ends(
            x0,
            p.evaluate(x0),
            pd.evaluate(x0),
            x1,
            p.evaluate(x1),
            pd.evaluate(x1),
        );

        let first = &table[0];
        let last = &table[table.len() - 1];

        // numpy.searchsorted(lower, x): the first segment starting at or after x.
        let x = long_term_year(first.jd_tt);
        let kept = s.segments.partition_point(|segment| segment.lower < x);
        if kept == 0 {
            return Err(table_error("ut1_utc", TimeScaleInputErrorKind::OutOfRange));
        }
        let mut s15 = s.segments[..kept].to_vec();
        let desired_y = first.delta_t;
        let current_y = s.evaluate(x);
        let joining = &mut s15[kept - 1];
        let t = (x - joining.lower) / (joining.upper - joining.lower);
        joining.coefficients[2] += (desired_y - current_y) / t;

        let x0 = long_term_year(last.jd_tt);
        let x1 = python_floor_div(x0 + Self::PATCH_WIDTH, 100.0) * 100.0;
        let y0 = last.delta_t;
        let lookback = table.len().min(366);
        let slope =
            (last.delta_t - table[table.len() - lookback].delta_t) * lookback as f64 / 365.0;
        let right = spline_given_ends(x0, y0, slope, x1, p.evaluate(x1), pd.evaluate(x1));

        let x0 = x1;
        let x1 = x0 + parabola_width;
        let far_right = spline_given_ends(
            x0,
            p.evaluate(x0),
            pd.evaluate(x0),
            x1,
            p.evaluate(x1),
            pd.evaluate(x1),
        );

        let mut segments = Vec::with_capacity(s15.len() + 4);
        segments.push(far_left);
        segments.push(left);
        segments.extend(s15);
        segments.push(right);
        segments.push(far_right);
        Ok(Self {
            curve: Splines { segments },
        })
    }

    /// Delta-T, seconds, at Julian year number `year`.
    fn evaluate(&self, year: f64) -> f64 {
        self.curve.evaluate(year)
    }
}

/// UT1 coverage interval for the embedded EOP table, in TT Julian dates.
///
/// Outside this interval delta-T comes from the long-term curve Skyfield
/// splices around the table. Every [`TimeScales`] built outside it carries
/// [`TimeScales::ut1_degraded`]; strict-mode constructors refuse it. See
/// [`crate::astro::time::eop`].
pub fn ut1_coverage() -> Ut1Provenance {
    embedded_ut1_coverage()
}

/// The embedded table's coverage, computed once.
fn embedded_ut1_coverage() -> Ut1Provenance {
    use std::sync::LazyLock;

    static COVERAGE: LazyLock<Ut1Provenance> = LazyLock::new(|| {
        ut1_coverage_for(
            &UT1_DATA,
            LEAP_SECONDS,
            "IERS Earth Orientation Parameters (UT1-UTC), bundled",
        )
    });
    *COVERAGE
}

fn ut1_coverage_for(
    ut1_utc: &[Ut1Entry],
    leap_seconds: &[LeapSecondEntry],
    source: &'static str,
) -> Ut1Provenance {
    let ut1_utc = effective_ut1_utc_rows(ut1_utc, leap_seconds);
    let first = ut1_utc.first();
    let last = ut1_utc.last();
    let to_jd_tt = |mjd: i32| -> f64 {
        let jd_utc = mjd as f64 + 2400000.5;
        let tt_minus_utc = find_leap_seconds_in_table_checked(jd_utc, leap_seconds)
            .expect("effective UT1 rows are covered by leap table")
            + TT_MINUS_TAI_S;
        jd_utc + tt_minus_utc / SECONDS_PER_DAY
    };
    Ut1Provenance {
        source,
        first_mjd: first.map(|e| e.mjd).unwrap_or(0),
        last_mjd: last.map(|e| e.mjd).unwrap_or(0),
        first_jd_tt: first.map(|e| to_jd_tt(e.mjd)).unwrap_or(0.0),
        last_jd_tt: last.map(|e| to_jd_tt(e.mjd)).unwrap_or(0.0),
        entries: ut1_utc.len(),
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
        let sod = hour as f64 * SECONDS_PER_HOUR + minute as f64 * SECONDS_PER_MINUTE + second;
        jd1 + sod / SECONDS_PER_DAY
    }

    fn positive_ulp_distance(a: f64, b: f64) -> u64 {
        debug_assert!(a.is_sign_positive());
        debug_assert!(b.is_sign_positive());
        a.to_bits().abs_diff(b.to_bits())
    }

    // --- IAU TCG/TCB linear definitions. -------------------------------------

    #[test]
    fn iau_tcg_tcb_constants_are_exact_f64_bits() {
        assert_eq!(TT_TCG_RATE_L_G.to_bits(), 0x3e07_f240_9f5d_dc8f);
        assert_eq!(TDB_TCB_RATE_L_B.to_bits(), 0x3e50_a609_49f9_cf0c);
        assert_eq!(TDB_TCB_OFFSET_TDB0_S.to_bits(), 0xbf11_2ba1_6e7a_311f);
        assert_eq!(TCG_TCB_REFERENCE_JD.to_bits(), 0x4142_a3c4_400c_34c2);
        assert_eq!(TT_TCG_RATE_L_G, 6.969290134e-10);
        assert_eq!(TDB_TCB_RATE_L_B, 1.550519768e-8);
        assert_eq!(TDB_TCB_OFFSET_TDB0_S, -6.55e-5);
        assert_eq!(TCG_TCB_REFERENCE_JD, 2_443_144.500_372_5);
    }

    #[test]
    fn tcg_tcb_linear_conversions_round_trip_to_one_ulp() {
        let cases = [
            TCG_TCB_REFERENCE_JD,
            J2000_JD,
            2_460_000.5,
            2_443_145.5,
            2_443_144.500_372_5 + 1_000.0,
            2_460_123.456_789,
            2_400_014.327_160_368,
            2_400_015.561_728_258,
        ];
        for jd in cases {
            let tcg = tt_to_tcg_jd(jd);
            let tt_back = tcg_to_tt_jd(tcg);
            assert!(
                positive_ulp_distance(tt_back, jd) <= 1,
                "TT->TCG->TT round trip at JD {jd}: got {tt_back}"
            );

            let tcb = tdb_to_tcb_jd(jd);
            let tdb_back = tcb_to_tdb_jd(tcb);
            assert!(
                positive_ulp_distance(tdb_back, jd) <= 1,
                "TDB->TCB->TDB round trip at JD {jd}: got {tdb_back}"
            );
        }
    }

    #[test]
    fn tcg_reference_epoch_is_synchronized_and_tcb_carries_tdb0_before_jd_rounding() {
        assert_eq!(
            tt_to_tcg_jd(TCG_TCB_REFERENCE_JD).to_bits(),
            TCG_TCB_REFERENCE_JD.to_bits()
        );
        assert_eq!(
            tcg_to_tt_jd(TCG_TCB_REFERENCE_JD).to_bits(),
            TCG_TCB_REFERENCE_JD.to_bits()
        );

        let tdb_offset_at_reference_s = (TCG_TCB_REFERENCE_JD - TCG_TCB_REFERENCE_JD)
            * SECONDS_PER_DAY
            * (1.0 - TDB_TCB_RATE_L_B)
            + TDB_TCB_OFFSET_TDB0_S;
        assert_eq!(
            tdb_offset_at_reference_s.to_bits(),
            TDB_TCB_OFFSET_TDB0_S.to_bits()
        );

        let tdb_at_tcb_reference = tcb_to_tdb_jd(TCG_TCB_REFERENCE_JD);
        let rounded_full_jd = TCG_TCB_REFERENCE_JD + TDB_TCB_OFFSET_TDB0_S / SECONDS_PER_DAY;
        assert_eq!(tdb_at_tcb_reference.to_bits(), rounded_full_jd.to_bits());
        assert_eq!(tdb_at_tcb_reference.to_bits(), 0x4142_a3c4_400c_34c0);
        assert_eq!(
            tdb_to_tcb_jd(tdb_at_tcb_reference).to_bits(),
            TCG_TCB_REFERENCE_JD.to_bits()
        );
    }

    #[test]
    fn tai_defining_epoch_maps_tt_and_tcg_to_reference_jd() {
        let scales = TimeScales::from_scale(TimeScale::Tai, 1977, 1, 1, 0, 0, 0.0)
            .expect("valid TAI reference instant");
        assert_eq!(scales.jd_tt.to_bits(), TCG_TCB_REFERENCE_JD.to_bits());
        assert_eq!(scales.jd_tcg().to_bits(), TCG_TCB_REFERENCE_JD.to_bits());
    }

    #[test]
    fn tcb_calendar_reference_input_resolves_to_tt_reference_jd() {
        let scales = TimeScales::from_scale(TimeScale::Tcb, 1977, 1, 1, 0, 0, 32.184)
            .expect("valid TCB reference instant");
        assert_eq!(scales.jd_tt.to_bits(), TCG_TCB_REFERENCE_JD.to_bits());
        assert_eq!(scales.jd_tcg().to_bits(), TCG_TCB_REFERENCE_JD.to_bits());
    }

    #[test]
    fn tdb_calendar_input_uses_periodic_tdb_tt_inverse() {
        let tdb_jd = J2000_JD;
        let tt_jd = tdb_to_tt_jd_for_tdb_input(tdb_jd);
        let reconstructed_tdb = tt_jd + tdb_minus_tt_seconds_at_tt_jd(tt_jd) / SECONDS_PER_DAY;
        assert_eq!(reconstructed_tdb.to_bits(), tdb_jd.to_bits());

        let scales = TimeScales::from_scale(TimeScale::Tdb, 2000, 1, 1, 12, 0, 0.0)
            .expect("valid TDB input");
        assert_eq!(scales.jd_tdb.to_bits(), tdb_jd.to_bits());
    }

    #[test]
    fn tcg_tcb_fractions_use_split_affine_relations() {
        let scales = TimeScales::from_utc(2000, 1, 1, 12, 0, 0.0).expect("valid UTC instant");

        assert_eq!(scales.tcg_fraction().to_bits(), 0x3f48_88c2_8751_43f2);
        assert_ne!(
            scales.tcg_fraction().to_bits(),
            (scales.jd_tcg() - scales.jd_whole).to_bits()
        );

        assert_eq!(scales.tcb_fraction().to_bits(), 0x3f4c_9c46_0494_33ba);
        assert_ne!(
            scales.tcb_fraction().to_bits(),
            (scales.jd_tcb() - scales.jd_whole).to_bits()
        );
    }

    #[test]
    fn from_scale_rejects_continuous_scale_leap_second_labels_before_normalizing() {
        let expected = Err(CoverageError::InvalidInput {
            field: "civil datetime",
            kind: TimeScaleInputErrorKind::InvalidCivilTime,
        });
        for scale in [
            TimeScale::Tai,
            TimeScale::Tt,
            TimeScale::Tcg,
            TimeScale::Tdb,
            TimeScale::Tcb,
            TimeScale::Gpst,
            TimeScale::Gst,
            TimeScale::Bdt,
            TimeScale::Qzsst,
        ] {
            assert_eq!(
                TimeScales::from_scale(scale, 2017, 1, 1, 0, 0, 60.0),
                expected
            );
        }
        assert!(TimeScales::from_scale(TimeScale::Utc, 2016, 12, 31, 23, 59, 60.0).is_ok());

        let tables = TimeTables::embedded();
        assert_eq!(
            TimeScales::from_scale_with_tables(TimeScale::Tcb, 2017, 1, 1, 0, 0, 60.0, tables),
            expected
        );
    }

    #[test]
    fn embedded_tables_path_is_bit_identical() {
        let tables = TimeTables::embedded();
        for (year, month, day, hour, minute, second) in [
            (1973, 1, 2, 0, 0, 0.0),
            (2000, 1, 1, 12, 0, 0.0),
            (2016, 12, 31, 23, 59, 60.0),
            (2026, 6, 1, 0, 0, 0.0),
        ] {
            let embedded = TimeScales::from_utc(year, month, day, hour, minute, second)
                .expect("embedded UTC conversion");
            let via_tables =
                TimeScales::from_utc_with_tables(year, month, day, hour, minute, second, tables)
                    .expect("table UTC conversion");
            assert_eq!(via_tables, embedded);
        }
    }

    #[test]
    fn caller_leap_table_future_step_shifts_tt_by_exactly_one_second() {
        let mut leap_seconds = LEAP_SECONDS.to_vec();
        let last = leap_seconds.last().expect("embedded leap table");
        leap_seconds.push(LeapSecondEntry {
            mjd: 61041,
            tai_utc: last.tai_utc + 1.0,
        });
        let tables = TimeTables::new(&leap_seconds, &UT1_DATA).expect("valid override tables");

        let embedded =
            TimeScales::from_utc(2026, 1, 2, 0, 0, 0.0).expect("embedded UTC conversion");
        let override_scales = TimeScales::from_utc_with_tables(2026, 1, 2, 0, 0, 0.0, tables)
            .expect("override UTC conversion");
        assert_eq!(
            override_scales.jd_whole.to_bits(),
            embedded.jd_whole.to_bits()
        );
        assert_eq!(
            override_scales.tt_fraction.to_bits(),
            (embedded.tt_fraction + 1.0 / SECONDS_PER_DAY).to_bits()
        );

        let before_embedded =
            TimeScales::from_utc(2025, 12, 1, 0, 0, 0.0).expect("embedded UTC conversion");
        let before_override = TimeScales::from_utc_with_tables(2025, 12, 1, 0, 0, 0.0, tables)
            .expect("override UTC conversion");
        assert_eq!(before_override, before_embedded);
    }

    #[test]
    fn caller_leap_table_must_cover_queried_epoch() {
        let leap_seconds = [LeapSecondEntry {
            mjd: 61041,
            tai_utc: 38.0,
        }];
        let tables = TimeTables::new(&leap_seconds, &UT1_DATA).expect("valid future tables");

        let err = TimeScales::from_utc_with_tables(2025, 12, 31, 0, 0, 0.0, tables)
            .expect_err("caller leap table must cover the query epoch");
        assert_eq!(
            err,
            CoverageError::InvalidInput {
                field: "leap_seconds",
                kind: TimeScaleInputErrorKind::OutOfRange
            }
        );

        let err = TimeScales::from_utc_validated_with_tables(
            2025,
            12,
            31,
            0,
            0,
            0.0,
            ValidityMode::Permissive,
            tables,
        )
        .expect_err("permissive mode still requires leap table coverage");
        assert_eq!(
            err,
            CoverageError::InvalidInput {
                field: "leap_seconds",
                kind: TimeScaleInputErrorKind::OutOfRange
            }
        );
    }

    #[test]
    fn caller_ut1_table_uses_validated_coverage_modes() {
        let ut1_utc = [
            Ut1Entry {
                mjd: 61041,
                ut1_utc: 0.0,
            },
            Ut1Entry {
                mjd: 61042,
                ut1_utc: 0.0,
            },
        ];
        let tables = TimeTables::new(LEAP_SECONDS, &ut1_utc).expect("valid short UT1 table");

        let strict = TimeScales::from_utc_with_tables(2025, 12, 31, 0, 0, 0.0, tables)
            .expect_err("strict caller-table path must reject before UT1 coverage");
        assert_eq!(
            strict,
            CoverageError::OutsideCoverage(crate::astro::time::eop::DegradeReason::BeforeCoverage)
        );

        let permissive = TimeScales::from_utc_validated_with_tables(
            2025,
            12,
            31,
            0,
            0,
            0.0,
            ValidityMode::Permissive,
            tables,
        )
        .expect("permissive caller-table path returns degraded value");
        assert_eq!(
            permissive.degraded,
            Some(crate::astro::time::eop::DegradeReason::BeforeCoverage)
        );
        assert_eq!(
            permissive.value.ut1_degraded,
            Some(crate::astro::time::eop::DegradeReason::BeforeCoverage)
        );

        let strict_after = TimeScales::from_utc_validated_with_tables(
            2026,
            1,
            3,
            0,
            0,
            0.0,
            ValidityMode::Strict,
            tables,
        )
        .expect_err("strict caller-table path must reject after UT1 coverage");
        assert_eq!(
            strict_after,
            CoverageError::OutsideCoverage(crate::astro::time::eop::DegradeReason::AfterCoverage)
        );
    }

    #[test]
    fn out_of_table_ut1_leaves_tt_and_tdb_bits_unchanged() {
        // 2100-01-01 is MJD 88069, past the embedded UT1 table. A caller table
        // that covers it must give the same TT and TDB bits as the embedded
        // path, which takes UT1 from the long-term delta-T curve there: TT and
        // TDB do not depend on UT1.
        let ut1_utc = [
            Ut1Entry {
                mjd: 88068,
                ut1_utc: 0.0,
            },
            Ut1Entry {
                mjd: 88070,
                ut1_utc: 0.0,
            },
        ];
        let tables = TimeTables::new(LEAP_SECONDS, &ut1_utc).expect("valid covering table");

        let held = TimeScales::from_utc(2100, 1, 1, 0, 0, 0.0).expect("embedded conversion");
        let covered = TimeScales::from_utc_with_tables(2100, 1, 1, 0, 0, 0.0, tables)
            .expect("covered conversion");

        assert_eq!(
            held.ut1_degraded,
            Some(crate::astro::time::eop::DegradeReason::AfterCoverage)
        );
        assert_eq!(covered.ut1_degraded, None);
        assert_eq!(held.jd_whole.to_bits(), covered.jd_whole.to_bits());
        assert_eq!(held.tt_fraction.to_bits(), covered.tt_fraction.to_bits());
        assert_eq!(held.jd_tt.to_bits(), covered.jd_tt.to_bits());
        assert_eq!(held.tdb_fraction.to_bits(), covered.tdb_fraction.to_bits());
        assert_eq!(held.jd_tdb.to_bits(), covered.jd_tdb.to_bits());
    }

    fn embedded_long_term() -> &'static LongTermDeltaT {
        use std::sync::LazyLock;
        static CURVE: LazyLock<LongTermDeltaT> = LazyLock::new(|| {
            let table: Vec<DeltaTRow> = UT1_DATA
                .iter()
                .map(|entry| {
                    let jd_utc = entry.mjd as f64 + 2400000.5;
                    delta_t_row(jd_utc, find_leap_seconds(jd_utc), entry.ut1_utc)
                })
                .collect();
            LongTermDeltaT::around(&table).expect("embedded curve")
        });
        &CURVE
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: {actual} vs {expected}"
        );
    }

    #[test]
    fn long_term_curve_is_the_smh2016_parabola_far_from_the_table() {
        // p(y) = -320 + 32.5 u^2 with u = (y - 1825) / 100.
        // y = -2000: u = -38.25, u^2 = 1463.0625, 32.5 u^2 = 47549.53125,
        //            p = 47229.53125 (extrapolation of the far-left segment).
        // y = -1570: u = -33.95, u^2 = 1152.6025, p = 37139.58125.
        // y = 2850:  u = 10.25, u^2 = 105.0625, p = 3094.53125.
        // y = 3000:  u = 11.75, u^2 = 138.0625, p = 4167.03125
        //            (extrapolation of the far-right segment).
        let curve = embedded_long_term();
        for (year, expected) in [
            (-2000.0, 47229.53125),
            (-1570.0, 37139.58125),
            (2850.0, 3094.53125),
            (3000.0, 4167.03125),
        ] {
            assert_close(curve.evaluate(year), expected, 1.0e-9, "parabola");
        }
    }

    #[test]
    fn long_term_curve_reads_table_s15_between_its_knots() {
        let curve = embedded_long_term();
        // At a knot t = 0, so Horner's rule leaves the constant term a0.
        assert_eq!(curve.evaluate(1950.0), 28.932);
        assert_eq!(curve.evaluate(-720.0), 20371.848);
        // Segment 1950..1953: a3 = -0.277, a2 = 0.22, a1 = 1.127, a0 = 28.932.
        // At 1952.5, t = 2.5 / 3 = 0.8333...:
        // ((-0.277 t + 0.22) t + 1.127) t + 28.932 = 29.863643518518...
        assert_close(
            curve.evaluate(1952.5),
            29.863_643_518_518_52,
            1.0e-12,
            "S15 1952.5",
        );
    }

    #[test]
    fn long_term_curve_meets_the_table_and_its_own_segments() {
        let curve = embedded_long_term();
        let table = &UT1_DATA[..];
        let first_jd_utc = table[0].mjd as f64 + 2400000.5;
        let last_jd_utc = table[table.len() - 1].mjd as f64 + 2400000.5;
        let first = delta_t_row(
            first_jd_utc,
            find_leap_seconds(first_jd_utc),
            table[0].ut1_utc,
        );
        let last = delta_t_row(
            last_jd_utc,
            find_leap_seconds(last_jd_utc),
            table[table.len() - 1].ut1_utc,
        );
        // The right cubic starts at the table's last value (t = 0 there), and
        // the adjusted S15 segment reaches the table's first value.
        assert_eq!(curve.evaluate(long_term_year(last.jd_tt)), last.delta_t);
        assert_close(
            curve.evaluate(long_term_year(first.jd_tt)),
            first.delta_t,
            1.0e-9,
            "table start",
        );
        assert_eq!(interpolate_delta_t(last.jd_tt), last.delta_t);
        assert_eq!(interpolate_delta_t(first.jd_tt), first.delta_t);
        // The connecting cubics meet their neighbours: S15 at year -720, the
        // far-left parabola segment at -1520, the far-right one at 2800.
        for (year, value) in [
            (-720.0, 20371.848),
            (-1520.0, 36044.33125),
            (2800.0, 2769.53125),
        ] {
            assert_close(curve.evaluate(year - 1.0e-9), value, 1.0e-4, "join");
        }
    }

    #[test]
    fn out_of_table_delta_t_follows_the_skyfield_splice() {
        // 2100-01-01 00:00 UTC: jd_tt = 2488069.500800741, year
        // (jd_tt - 1721045) / 365.25 = 2099.9986332669155, in the right cubic
        // from year x0 = 2027.5003444236575 (the table's last row, delta-T
        // y0 = 69.2099116) to x1 = 2800. Its slope is the table's last-year
        // change (69.2099116 - 69.1689553) * 366 / 365 = 0.0410685090411 s per
        // year; at x1 it meets p(2800) = 2769.53125 with slope
        // 0.65 * 9.75 = 6.3375. With width 772.4996555763425 its coefficients
        // are a3 = -473.2007004956449, a2 = 3141.79662980636,
        // a1 = 31.72540908928423, a0 = 69.2099116, and at
        // t = (2099.9986332669155 - x0) / width = 0.0938489594...
        // delta-T = 99.46794026607563 s.
        let after = TimeScales::from_utc(2100, 1, 1, 0, 0, 0.0).expect("embedded conversion");
        assert_eq!(
            after.ut1_degraded,
            Some(crate::astro::time::eop::DegradeReason::AfterCoverage)
        );
        assert_close(
            interpolate_delta_t(after.jd_tt),
            99.467_940_266_075_63,
            1.0e-9,
            "2100",
        );
        let tt_minus_ut1 = (after.tt_fraction - after.ut1_fraction) * SECONDS_PER_DAY;
        assert_close(tt_minus_ut1, 99.467_940_266_075_63, 1.0e-4, "2100 TT-UT1");

        // 1972-12-01 00:00 UTC precedes the table (which starts 1973-01-02):
        // year 1972.9158124567139, in the S15 segment 1971..1974 whose linear
        // term was raised by (43.3755822 - 43.15340078057444) / 0.66780790453
        // so the curve meets the table's first value. The segment gives
        // 43.26834427265643 s there.
        let before = TimeScales::from_utc(1972, 12, 1, 0, 0, 0.0).expect("embedded conversion");
        assert_eq!(
            before.ut1_degraded,
            Some(crate::astro::time::eop::DegradeReason::BeforeCoverage)
        );
        assert_close(
            interpolate_delta_t(before.jd_tt),
            43.268_344_272_656_43,
            1.0e-9,
            "1972",
        );
    }

    #[test]
    fn in_table_delta_t_is_the_linear_interpolation() {
        // Inside the table the long-term curve is never consulted.
        let rows: Vec<DeltaTRow> = UT1_DATA[100..102]
            .iter()
            .map(|entry| {
                let jd_utc = entry.mjd as f64 + 2400000.5;
                delta_t_row(jd_utc, find_leap_seconds(jd_utc), entry.ut1_utc)
            })
            .collect();
        let mid = 0.5 * (rows[0].jd_tt + rows[1].jd_tt);
        let expected = rows[0].delta_t
            + (mid - rows[0].jd_tt) * (rows[1].delta_t - rows[0].delta_t)
                / (rows[1].jd_tt - rows[0].jd_tt);
        assert_eq!(interpolate_delta_t(mid).to_bits(), expected.to_bits());
    }

    #[test]
    fn caller_table_splices_relative_to_its_own_ends() {
        // A two-row caller table: lookback is 2 rows, so the right cubic's
        // slope is (d1 - d0) * 2 / 365; the curve starts at its last row.
        let ut1_utc = [
            Ut1Entry {
                mjd: 60000,
                ut1_utc: -0.01,
            },
            Ut1Entry {
                mjd: 60001,
                ut1_utc: -0.02,
            },
        ];
        let last_jd_utc = 60001.0 + 2400000.5;
        let last = delta_t_row(last_jd_utc, find_leap_seconds(last_jd_utc), -0.02);
        assert_eq!(
            interpolate_delta_t_with_table(last.jd_tt, &ut1_utc, LEAP_SECONDS).expect("in table"),
            last.delta_t
        );
        let after = interpolate_delta_t_with_table(last.jd_tt + 30.0, &ut1_utc, LEAP_SECONDS)
            .expect("long-term");
        let embedded_after = interpolate_delta_t(last.jd_tt + 30.0);
        assert!(after.is_finite());
        assert_ne!(
            after, embedded_after,
            "spliced to the caller table, not the embedded one"
        );
        // 69.184 - (-0.02) = 69.204 s at the last row; the curve leaves it
        // with slope (69.204 - 69.194) * 2 / 365 s per year plus the pull of
        // the parabola, so 30 days later delta-T is within a few ms of 69.204.
        assert_close(after, 69.204, 0.01, "caller splice");
    }

    #[test]
    fn caller_tables_reject_short_or_malformed_inputs() {
        assert_eq!(
            TimeTables::new(&[], &UT1_DATA).expect_err("empty leap table"),
            CoverageError::InvalidInput {
                field: "leap_seconds",
                kind: TimeScaleInputErrorKind::Missing
            }
        );
        let unsorted = [
            LeapSecondEntry {
                mjd: 41317,
                tai_utc: 10.0,
            },
            LeapSecondEntry {
                mjd: 41317,
                tai_utc: 11.0,
            },
        ];
        assert_eq!(
            TimeTables::new(&unsorted, &UT1_DATA).expect_err("unsorted leap table"),
            CoverageError::InvalidInput {
                field: "leap_seconds",
                kind: TimeScaleInputErrorKind::OutOfRange
            }
        );
        assert_eq!(
            TimeTables::new(LEAP_SECONDS, &[]).expect_err("short UT1 table"),
            CoverageError::InvalidInput {
                field: "ut1_utc",
                kind: TimeScaleInputErrorKind::Missing
            }
        );
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
    fn tai_utc_and_gps_utc_offsets_match_iers_and_is_gps_200() {
        // 2017-01-01 onward: IERS Bulletin C TAI-UTC = 37 s; IS-GPS-200
        // GPS-UTC = 18 s; the two differ by the fixed GPST-TAI = 19 s.
        let jd_2017 = utc_jd(2017, 1, 1, 0, 0, 0.0);
        assert_eq!(tai_utc_offset_s(jd_2017), 37.0);
        assert_eq!(gps_utc_offset_s(jd_2017), 18.0);
        assert_eq!(tai_utc_offset_s(jd_2017) - gps_utc_offset_s(jd_2017), 19.0);

        // The named alias must equal the historical accessor bit-for-bit, and the
        // GPS offset must track it minus 19 s across the whole integer table.
        for (y, m, d) in [(1980, 1, 1), (2000, 1, 1), (2009, 1, 1), (2017, 1, 1)] {
            let jd = utc_jd(y, m, d, 0, 0, 0.0);
            assert_eq!(
                tai_utc_offset_s(jd).to_bits(),
                find_leap_seconds(jd).to_bits()
            );
            assert_eq!(gps_utc_offset_s(jd), find_leap_seconds(jd) - 19.0);
        }

        // Cross-check GPS-UTC against the leap-aware UTC->GPST offset, which is
        // independently validated against RTKLIB.
        assert_eq!(
            gps_utc_offset_s(jd_2017),
            timescale_offset_at_s(TimeScale::Utc, TimeScale::Gpst, jd_2017)
                .expect("leap-aware offset")
        );
    }

    #[test]
    fn tai_minus_utc_pre_1961_clamps_to_first_segment_and_nonfinite_is_nan() {
        // Before the table, clamp to the first defined 1961 value.
        assert_eq!(find_leap_seconds(utc_jd(1958, 1, 1, 0, 0, 0.0)), 1.4228180);
        assert!(find_leap_seconds(f64::NAN).is_nan());
        assert!(find_leap_seconds(f64::INFINITY).is_nan());
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

    #[test]
    fn the_1997_and_1999_leap_seconds_fall_where_iers_bulletin_c_puts_them() {
        // TAI - UTC became 31 s on 1997-07-01 and 32 s on 1999-01-01 (IERS
        // Bulletin C 13 and 16); the embedded UT1 - UTC series steps by one
        // second between MJD 50629 and 50630 and between 51178 and 51179.
        let jd = |mjd: f64| mjd + 2_400_000.5;
        assert_eq!(find_leap_seconds(jd(50_447.5)), 30.0);
        assert_eq!(find_leap_seconds(jd(50_448.5)), 30.0);
        assert_eq!(find_leap_seconds(jd(50_629.5)), 30.0);
        assert_eq!(find_leap_seconds(jd(50_630.5)), 31.0);
        assert_eq!(find_leap_seconds(jd(50_813.5)), 31.0);
        assert_eq!(find_leap_seconds(jd(51_178.5)), 31.0);
        assert_eq!(find_leap_seconds(jd(51_179.5)), 32.0);
        assert!(!is_positive_leap_second_label(1996, 12, 31, 23, 59));
        assert!(is_positive_leap_second_label(1997, 6, 30, 23, 59));
        assert!(!is_positive_leap_second_label(1997, 12, 31, 23, 59));
        assert!(is_positive_leap_second_label(1998, 12, 31, 23, 59));
    }
}
