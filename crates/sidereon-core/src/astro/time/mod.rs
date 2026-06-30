//! Time scales and the public time model.
//!
//! The precise time-scale machinery, which used to be `pub(crate)` inside
//! `orbis_nif` and is now public in the core crate. It exposes three layers:
//!
//! - [`scales`] - the parity-critical UTC->TAI->TT->TDB->UT1 conversion, moved
//!   verbatim from `orbis_nif/src/time_scales.rs`. The numerics are byte-for-byte
//!   identical so the existing Skyfield 0-ULP parity holds.
//! - [`civil`] - the no-leap-second civil-calendar conversions (split Julian
//!   date, seconds since J2000, second-of-day, fractional day-of-year) that the
//!   GNSS bindings consume directly, so each interface stops reimplementing them.
//! - [`model`] - the public time model type family ([`TimeScale`], [`Instant`],
//!   [`Duration`], [`JulianDateSplit`], [`GnssWeekTow`]).
//! - [`eop`] - time/EOP validity + provenance API with strict-vs-permissive
//!   policy hooks.
//!
//! The legacy thin [`Time`] (seconds since J2000, used by the propagator) is
//! retained unchanged for backward compatibility.

pub mod civil;
pub mod eop;
pub mod gnss;
pub mod model;
pub mod scales;

pub use civil::{
    civil_from_j2000_seconds, civil_from_julian_day_number, civil_from_split_julian_date,
    day_of_year, day_of_year_int, days_in_month, fractional_day_of_year_from_instant, is_leap_year,
    j2000_seconds, j2000_seconds_from_split, julian_date_from_instant, mjd_from_jd, second_of_day,
    second_of_day_from_instant, split_julian_date, split_julian_date_add_seconds,
    split_julian_date_from_j2000_seconds,
};
pub use eop::{
    CoverageError, DegradeReason, LeapSecondTable, TimeScaleInputErrorKind, Ut1Provenance,
    Validated, ValidityMode,
};
pub use model::{
    Duration, GnssWeekTow, Instant, InstantRepr, JulianDateSplit, TimeModelError, TimeScale,
    SECONDS_PER_WEEK,
};
pub use scales::{
    find_leap_seconds, gps_utc_offset_s, leap_second_table, tai_utc_offset_s,
    timescale_offset_at_s, timescale_offset_s, TimeOffsetError, TimeOffsetErrorCode, TimeScales,
    GLONASST_MINUS_UTC_S,
};

/// Legacy lightweight epoch: seconds since the J2000 TDB epoch.
///
/// Kept for the propagator/force-model API surface; the richer public time
/// model lives in [`model`].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Time {
    pub seconds_since_j2000: f64,
}

impl Time {
    pub fn new(seconds_since_j2000: f64) -> Result<Self, TimeModelError> {
        if !seconds_since_j2000.is_finite() {
            return Err(TimeModelError::InvalidInput {
                field: "seconds_since_j2000",
                reason: "must be finite",
            });
        }
        Ok(Self {
            seconds_since_j2000,
        })
    }

    pub fn tdb(&self) -> f64 {
        self.seconds_since_j2000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn week_tow_normalizes_overflow() {
        let wt = GnssWeekTow::new(TimeScale::Gpst, 100, SECONDS_PER_WEEK + 5.0)
            .expect("valid week/TOW")
            .normalized()
            .expect("valid normalized week/TOW");
        assert_eq!(wt.week, 101);
        assert!((wt.tow_s - 5.0).abs() < 1e-9);
    }

    #[test]
    fn week_tow_borrows_negative() {
        let wt = GnssWeekTow::new(TimeScale::Gpst, 100, -10.0)
            .expect("valid week/TOW")
            .normalized()
            .expect("valid normalized week/TOW");
        assert_eq!(wt.week, 99);
        assert!((wt.tow_s - (SECONDS_PER_WEEK - 10.0)).abs() < 1e-6);
    }

    #[test]
    fn week_rollover_unrolls() {
        let wt = GnssWeekTow::new(TimeScale::Gpst, 10, 0.0).expect("valid week/TOW");
        assert_eq!(wt.unrolled_week(2).expect("valid unrolled week"), 10 + 2048);
    }

    #[test]
    fn scalar_time_rejects_nonfinite_epoch_seconds() {
        assert!(Time::new(f64::NAN).is_err());
        assert!(Time::new(f64::INFINITY).is_err());
        assert!(Time::new(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn scalar_time_valid_epoch_is_unchanged() {
        let t = Time::new(123.25).expect("valid scalar time");
        assert_eq!(t.seconds_since_j2000, 123.25);
        assert_eq!(t.tdb(), 123.25);
    }

    #[test]
    fn ut1_coverage_strict_vs_permissive() {
        let prov = scales::ut1_coverage();
        // Inside coverage: ok, not degraded.
        let mid = (prov.first_jd_tt + prov.last_jd_tt) / 2.0;
        assert_eq!(
            eop::check_ut1_coverage(&prov, mid, ValidityMode::Strict),
            Ok(None)
        );
        // Before coverage: strict errors, permissive degrades.
        let before = prov.first_jd_tt - 1.0;
        assert!(eop::check_ut1_coverage(&prov, before, ValidityMode::Strict).is_err());
        assert_eq!(
            eop::check_ut1_coverage(&prov, before, ValidityMode::Permissive),
            Ok(Some(DegradeReason::BeforeCoverage))
        );
    }

    #[test]
    fn time_scales_from_utc_unchanged_shape() {
        // J2000 epoch sanity: 2000-01-01 12:00:00 UTC.
        let ts = TimeScales::from_utc(2000, 1, 1, 12, 0, 0.0).expect("valid UTC instant");
        assert!((ts.jd_tt - 2451545.0).abs() < 1e-3);
    }

    #[test]
    fn time_scales_from_utc_rejects_non_finite_seconds() {
        let err = TimeScales::from_utc(2000, 1, 1, 12, 0, f64::NAN)
            .expect_err("non-finite second must error before time-scale arithmetic");
        assert_eq!(
            err,
            CoverageError::InvalidInput {
                field: "second",
                kind: TimeScaleInputErrorKind::NonFinite
            }
        );

        let err =
            TimeScales::from_utc_validated(2000, 1, 1, 12, 0, f64::INFINITY, ValidityMode::Strict)
                .expect_err("validated path must reject non-finite seconds before coverage checks");
        assert_eq!(
            err,
            CoverageError::InvalidInput {
                field: "second",
                kind: TimeScaleInputErrorKind::NonFinite
            }
        );
    }

    #[test]
    fn time_scales_from_utc_rejects_invalid_civil_datetime() {
        let err = TimeScales::from_utc(2001, 2, 29, 12, 0, 0.0)
            .expect_err("invalid civil date must error before time-scale arithmetic");
        assert_eq!(
            err,
            CoverageError::InvalidInput {
                field: "civil datetime",
                kind: TimeScaleInputErrorKind::InvalidCivilDate,
            }
        );

        let err = TimeScales::from_utc_validated(2000, 1, 1, 24, 0, 0.0, ValidityMode::Strict)
            .expect_err("invalid civil time must error before coverage checks");
        assert_eq!(
            err,
            CoverageError::InvalidInput {
                field: "civil datetime",
                kind: TimeScaleInputErrorKind::InvalidCivilTime,
            }
        );
    }

    #[test]
    fn time_scales_from_utc_maps_positive_leap_second_between_neighbors() {
        fn tt_delta_seconds(later: TimeScales, earlier: TimeScales) -> f64 {
            (later.jd_whole - earlier.jd_whole) * 86_400.0
                + (later.tt_fraction - earlier.tt_fraction) * 86_400.0
        }

        let before = TimeScales::from_utc(2016, 12, 31, 23, 59, 59.0).expect("leap eve second");
        let leap = TimeScales::from_utc(2016, 12, 31, 23, 59, 60.0).expect("inserted leap second");
        let after = TimeScales::from_utc(2017, 1, 1, 0, 0, 0.0).expect("post-leap midnight");

        assert!(
            (tt_delta_seconds(leap, before) - 1.0).abs() < 1.0e-8,
            "leap label must be one SI second after :59"
        );
        assert!(
            (tt_delta_seconds(after, leap) - 1.0).abs() < 1.0e-8,
            "post-leap midnight must be one SI second after :60"
        );
        assert_ne!(leap, after, "leap second must not collapse onto midnight");

        assert!(TimeScales::from_utc(2017, 1, 1, 0, 0, 60.0).is_err());
        assert!(TimeScales::from_utc(2016, 12, 31, 23, 59, 61.0).is_err());
        assert!(TimeScales::from_utc(2016, 12, 31, 23, 59, -1.0).is_err());
    }

    #[test]
    fn from_utc_validated_in_coverage_is_bit_identical_and_not_degraded() {
        // J2000 is comfortably inside the embedded UT1/EOP coverage interval.
        let plain = TimeScales::from_utc(2000, 1, 1, 12, 0, 0.0).expect("valid UTC instant");
        for mode in [ValidityMode::Strict, ValidityMode::Permissive] {
            let v = TimeScales::from_utc_validated(2000, 1, 1, 12, 0, 0.0, mode)
                .expect("in-coverage instant must not error in either mode");
            assert_eq!(v.degraded, None, "in-coverage must not be degraded");
            // The numerics must be the EXACT same bits as the parity path.
            assert_eq!(
                v.value, plain,
                "validated numerics must equal from_utc bit-for-bit"
            );
        }
    }

    #[test]
    fn from_utc_validated_strict_errors_before_coverage() {
        let prov = scales::ut1_coverage();
        // The UT1 table starts at MJD 41684 (1973); pick an instant safely before.
        let (y, m, d) = (1960, 1, 1);
        let plain = TimeScales::from_utc(y, m, d, 0, 0, 0.0).expect("valid UTC instant");
        assert!(
            plain.jd_tt < prov.first_jd_tt,
            "fixture must be before coverage"
        );

        let err = TimeScales::from_utc_validated(y, m, d, 0, 0, 0.0, ValidityMode::Strict)
            .expect_err("strict mode must error before coverage");
        assert_eq!(
            err,
            CoverageError::OutsideCoverage(DegradeReason::BeforeCoverage)
        );
    }

    #[test]
    fn from_utc_validated_strict_errors_after_coverage() {
        let prov = scales::ut1_coverage();
        // The UT1 table ends at MJD 61239 (~2026); pick an instant safely after.
        let (y, m, d) = (2100, 1, 1);
        let plain = TimeScales::from_utc(y, m, d, 0, 0, 0.0).expect("valid UTC instant");
        assert!(
            plain.jd_tt > prov.last_jd_tt,
            "fixture must be after coverage"
        );

        let err = TimeScales::from_utc_validated(y, m, d, 0, 0, 0.0, ValidityMode::Strict)
            .expect_err("strict mode must error after coverage");
        assert_eq!(
            err,
            CoverageError::OutsideCoverage(DegradeReason::AfterCoverage)
        );
    }

    #[test]
    fn from_utc_validated_permissive_clamps_and_marks_degraded() {
        // Before coverage: permissive returns the clamped value, marked degraded,
        // and the clamped numerics equal the parity path exactly.
        let plain_before = TimeScales::from_utc(1960, 1, 1, 0, 0, 0.0).expect("valid UTC instant");
        let before =
            TimeScales::from_utc_validated(1960, 1, 1, 0, 0, 0.0, ValidityMode::Permissive)
                .expect("permissive must not error");
        assert_eq!(before.degraded, Some(DegradeReason::BeforeCoverage));
        assert_eq!(before.value, plain_before);

        // After coverage: permissive returns the clamped value, marked degraded.
        let plain_after = TimeScales::from_utc(2100, 1, 1, 0, 0, 0.0).expect("valid UTC instant");
        let after = TimeScales::from_utc_validated(2100, 1, 1, 0, 0, 0.0, ValidityMode::Permissive)
            .expect("permissive must not error");
        assert_eq!(after.degraded, Some(DegradeReason::AfterCoverage));
        assert_eq!(after.value, plain_after);
    }
}
