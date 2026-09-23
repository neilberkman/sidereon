//! Time / EOP validity + provenance API.
//!
//! Leap-second and UT1/EOP tables carry source + effective date + coverage
//! interval, and the library exposes a strict vs permissive mode. Strict mode
//! (the default) errors outside table coverage; permissive mode returns the
//! long-term value and reports the departure.
//!
//! Outside the UT1 table, [`crate::astro::time::scales`] takes delta-T (TT-UT1)
//! from the curve Skyfield 1.54 `timelib.build_delta_t` splices around its
//! table, reproduced arithmetic for arithmetic:
//!
//! - before the table, Table S15 (2020) of Morrison, Stephenson, Hohenkerk
//!   and Zawilski, the cubic splines for 720 BC to AD 2019, truncated where
//!   the table starts and with the linear term of its last segment adjusted
//!   so the curve meets the table's first value;
//! - after the table, a cubic from the table's last value, with the slope of
//!   its last year, to the long-term parabola of Stephenson, Morrison and
//!   Hohenkerk (2016), `-320 + 32.5 ((year - 1825) / 100)^2` s, at the
//!   largest multiple of 100 not after `x0 + 800`, where `x0` is the table's
//!   last year;
//! - before year -720 and beyond that join, the parabola, with an 800-year
//!   cubic joining it to Table S15.
//!
//! This is Skyfield's value for any instant its own table does not cover. It
//! keeps UT1 continuous at the table edges and across a leap second announced
//! after the table ends, where setting UT1-UTC to zero outside the EOP range
//! (as Orekit does) would make UT1 jump by up to 0.9 s at the edge.
//!
//! A long-term value is never silent:
//!
//! - every [`crate::astro::time::scales::TimeScales`] carries
//!   [`crate::astro::time::scales::TimeScales::ut1_degraded`], which is `Some`
//!   exactly when its UT1 fields come from outside the table. TT and TDB do not
//!   depend on UT1 and are exact in either case.
//! - the frame transforms that read UT1, and every entry point built on them,
//!   refuse such a value with
//!   [`crate::astro::frames::transforms::FrameTransformError::Ut1OutsideCoverage`]
//!   by default. Each has a permissive route that returns the result with the
//!   departure in a [`Validated`]: an entry point that builds its own time
//!   scales has a `_with_validity` variant taking a [`ValidityMode`] (for
//!   example [`crate::astro::passes::find_passes_with_validity`]), and one that
//!   takes caller time scales runs under
//!   [`crate::astro::frames::transforms::with_ut1_validity`]. A search over a
//!   window checks every instant it evaluates, so under Strict a window
//!   reaching past the table is refused rather than cut short.
//! - malformed or non-finite inputs return [`CoverageError::InvalidInput`]
//!   before the parity-critical arithmetic runs,
//! - [`ValidityMode::Strict`] returns [`CoverageError`] outside coverage (the
//!   long-term value is never handed back), and
//! - [`ValidityMode::Permissive`] returns the long-term value paired with a
//!   [`DegradeReason`] marker.
//!
//! [`check_ut1_coverage`] is the pure policy hook both modes share; it does not
//! touch any delta-T value, so the parity-critical math is unaffected.

/// Provenance + coverage of the embedded leap-second (TAI-UTC) table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeapSecondTable {
    /// Human-readable source / bulletin identifier.
    pub source: &'static str,
    /// First Modified Julian Date covered by the table.
    pub first_mjd: i32,
    /// Last (most recent) Modified Julian Date with a leap-second step.
    pub last_mjd: i32,
    /// Number of table entries.
    pub entries: usize,
}

/// Provenance + coverage of the embedded UT1-UTC / delta-T table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ut1Provenance {
    /// Human-readable source identifier.
    pub source: &'static str,
    /// First Modified Julian Date in the UT1 table.
    pub first_mjd: i32,
    /// Last Modified Julian Date in the UT1 table.
    pub last_mjd: i32,
    /// First covered instant, expressed as a TT Julian date.
    pub first_jd_tt: f64,
    /// Last covered instant, expressed as a TT Julian date.
    pub last_jd_tt: f64,
    /// Number of table entries.
    pub entries: usize,
}

impl Ut1Provenance {
    /// True if `jd_tt` falls inside the table's covered interval (inclusive).
    pub fn covers_jd_tt(&self, jd_tt: f64) -> bool {
        jd_tt.is_finite() && jd_tt >= self.first_jd_tt && jd_tt <= self.last_jd_tt
    }
}

/// Validity policy applied when an instant falls outside table coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidityMode {
    /// Errors outside table coverage. This is the default: a value the table
    /// does not cover is refused unless the caller opts in to the long-term
    /// value.
    #[default]
    Strict,
    /// Returns the long-term delta-T value outside coverage and marks the
    /// result degraded with a [`DegradeReason`].
    Permissive,
}

/// Reason a result was marked degraded under [`ValidityMode::Permissive`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DegradeReason {
    /// Instant precedes the first covered table entry; TT-UT1 comes from the
    /// long-term curve joined to the first entry.
    BeforeCoverage,
    /// Instant follows the last covered table entry; TT-UT1 comes from the
    /// long-term curve joined to the last entry.
    AfterCoverage,
}

impl core::fmt::Display for DegradeReason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            DegradeReason::BeforeCoverage => "instant precedes the UT1 table coverage",
            DegradeReason::AfterCoverage => "instant follows the UT1 table coverage",
        })
    }
}

/// Invalid civil-time input kind for time-scale conversion boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeScaleInputErrorKind {
    /// A required input was absent; table validation uses this for empty leap-second tables,
    /// UT1 tables with fewer than two rows, or too few effective UT1 rows.
    Missing,
    /// A numeric input was NaN or infinite; time-scale validation uses this for date seconds,
    /// table offsets, checked UTC lookups, and coverage queries.
    NonFinite,
    /// A mapped validation failure required the value to be strictly greater than zero; the
    /// generic field-error mapper preserves this classification for downstream tide errors.
    NotPositive,
    /// A mapped validation failure rejected a negative value where a non-negative value was
    /// required; the generic field-error mapper preserves this classification for downstream
    /// tide errors.
    Negative,
    /// An input fell outside an accepted range or table ordering; table validation uses this for
    /// non-increasing MJD entries, and checked leap lookup uses it before the first table entry.
    OutOfRange,
    /// A textual floating-point field could not be parsed; the generic field-error mapper
    /// preserves this classification for downstream tide errors.
    FloatParse,
    /// A textual integer field could not be parsed; the generic field-error mapper preserves
    /// this classification for downstream tide errors.
    IntParse,
    /// Calendar fields did not form a valid civil date; UTC conversion reports this before
    /// time-scale arithmetic, including for an invalid date such as 2001-02-29.
    InvalidCivilDate,
    /// Clock fields did not form a valid civil time; UTC validation reports this for an
    /// out-of-range clock or a second value that is not an allowed leap-second label.
    InvalidCivilTime,
}

/// A value paired with whether it was produced inside valid table coverage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Validated<T> {
    /// The computed value (UT1 from the long-term delta-T curve if `degraded`
    /// is `Some`).
    pub value: T,
    /// `None` if produced inside coverage; otherwise why it is degraded.
    pub degraded: Option<DegradeReason>,
}

impl<T> Validated<T> {
    /// A value produced inside valid coverage.
    pub fn ok(value: T) -> Self {
        Self {
            value,
            degraded: None,
        }
    }

    /// A value produced outside coverage (UT1 from the long-term curve).
    pub fn degraded(value: T, reason: DegradeReason) -> Self {
        Self {
            value,
            degraded: Some(reason),
        }
    }

    /// True if the value was produced inside valid coverage.
    pub fn is_valid(&self) -> bool {
        self.degraded.is_none()
    }
}

impl<T> Validated<Option<T>> {
    /// `Some` of the value with its departure when there is a value, `None`
    /// otherwise; the departure goes with the value it was accepted for.
    pub fn transpose(self) -> Option<Validated<T>> {
        let degraded = self.degraded;
        self.value.map(|value| Validated { value, degraded })
    }
}

/// Error returned by strict-mode coverage checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageError {
    /// Time-scale conversion input is malformed or outside its accepted domain.
    InvalidInput {
        /// Stable label for the rejected input. Civil validation copies it from the field
        /// validator, while table and coverage checks use labels such as `leap_seconds`,
        /// `ut1_utc`, and `jd_tt`; formatting this error includes the label.
        field: &'static str,
        /// Classification selected by civil or table validation, or by a non-finite coverage
        /// query; downstream tide errors translate this classification one-for-one.
        kind: TimeScaleInputErrorKind,
    },
    /// Instant is outside the table's covered interval.
    OutsideCoverage(DegradeReason),
}

impl core::fmt::Display for CoverageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CoverageError::InvalidInput { field, kind } => {
                write!(f, "invalid time-scale input {field}: {kind:?}")
            }
            CoverageError::OutsideCoverage(DegradeReason::BeforeCoverage) => {
                write!(f, "instant precedes EOP/UT1 table coverage")
            }
            CoverageError::OutsideCoverage(DegradeReason::AfterCoverage) => {
                write!(f, "instant follows EOP/UT1 table coverage")
            }
        }
    }
}

impl std::error::Error for CoverageError {}

/// Classify `jd_tt` against UT1 coverage under the given [`ValidityMode`].
///
/// In [`ValidityMode::Strict`] this returns `Err` outside coverage. In
/// [`ValidityMode::Permissive`] it returns `Ok` with a [`DegradeReason`] flag
/// when outside coverage. This is a pure policy hook: it does NOT change any
/// delta-T value, preserving Skyfield parity.
pub fn check_ut1_coverage(
    prov: &Ut1Provenance,
    jd_tt: f64,
    mode: ValidityMode,
) -> Result<Option<DegradeReason>, CoverageError> {
    if !jd_tt.is_finite() {
        return Err(CoverageError::InvalidInput {
            field: "jd_tt",
            kind: TimeScaleInputErrorKind::NonFinite,
        });
    }

    let reason = ut1_coverage_departure(prov, jd_tt);

    match (mode, reason) {
        (ValidityMode::Strict, Some(r)) => Err(CoverageError::OutsideCoverage(r)),
        (_, r) => Ok(r),
    }
}

/// The first UT1 departure recorded by a long-lived source (a propagation
/// context, an SSR-corrected ephemeris), shared by its clones and safe to
/// record from several threads.
#[derive(Debug, Clone, Default)]
pub(crate) struct Ut1DepartureRecord(std::sync::Arc<std::sync::atomic::AtomicU8>);

impl Ut1DepartureRecord {
    const NONE: u8 = 0;
    const BEFORE: u8 = 1;
    const AFTER: u8 = 2;

    /// Record `departure` unless an earlier one is already recorded.
    pub(crate) fn record(&self, departure: Option<DegradeReason>) {
        let code = match departure {
            None => return,
            Some(DegradeReason::BeforeCoverage) => Self::BEFORE,
            Some(DegradeReason::AfterCoverage) => Self::AFTER,
        };
        let _ = self.0.compare_exchange(
            Self::NONE,
            code,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        );
    }

    /// The first recorded departure.
    pub(crate) fn first(&self) -> Option<DegradeReason> {
        match self.0.load(std::sync::atomic::Ordering::Acquire) {
            Self::BEFORE => Some(DegradeReason::BeforeCoverage),
            Self::AFTER => Some(DegradeReason::AfterCoverage),
            _ => None,
        }
    }
}

/// Where `jd_tt` lies relative to the UT1 coverage interval, without a policy.
///
/// `None` inside `[first_jd_tt, last_jd_tt]` (inclusive, the same bounds the
/// delta-T interpolation uses), otherwise the side of the table on which the
/// long-term curve supplies delta-T. A non-finite `jd_tt` compares false on both sides and yields
/// `None`; callers validate finiteness first.
pub(crate) fn ut1_coverage_departure(prov: &Ut1Provenance, jd_tt: f64) -> Option<DegradeReason> {
    if jd_tt < prov.first_jd_tt {
        Some(DegradeReason::BeforeCoverage)
    } else if jd_tt > prov.last_jd_tt {
        Some(DegradeReason::AfterCoverage)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> Ut1Provenance {
        Ut1Provenance {
            source: "test",
            first_mjd: 0,
            last_mjd: 1,
            first_jd_tt: 2_451_545.0,
            last_jd_tt: 2_451_546.0,
            entries: 2,
        }
    }

    #[test]
    fn ut1_coverage_rejects_nonfinite_query() {
        let prov = provenance();
        let expected = Err(CoverageError::InvalidInput {
            field: "jd_tt",
            kind: TimeScaleInputErrorKind::NonFinite,
        });
        assert_eq!(
            check_ut1_coverage(&prov, f64::NAN, ValidityMode::Strict),
            expected
        );
        assert_eq!(
            check_ut1_coverage(&prov, f64::INFINITY, ValidityMode::Permissive),
            expected
        );
        assert!(!prov.covers_jd_tt(f64::NAN));
    }

    #[test]
    fn ut1_coverage_valid_query_is_unchanged() {
        let prov = provenance();
        assert!(prov.covers_jd_tt(2_451_545.5));
        assert_eq!(
            check_ut1_coverage(&prov, 2_451_545.5, ValidityMode::Strict),
            Ok(None)
        );
    }

    #[test]
    fn validity_mode_defaults_to_strict() {
        assert_eq!(ValidityMode::default(), ValidityMode::Strict);
        let prov = provenance();
        assert_eq!(
            check_ut1_coverage(&prov, 2_451_547.0, ValidityMode::default()),
            Err(CoverageError::OutsideCoverage(DegradeReason::AfterCoverage))
        );
    }

    #[test]
    fn coverage_departure_is_inclusive_at_both_edges() {
        let prov = provenance();
        assert_eq!(ut1_coverage_departure(&prov, 2_451_545.0), None);
        assert_eq!(ut1_coverage_departure(&prov, 2_451_546.0), None);
        assert_eq!(
            ut1_coverage_departure(&prov, 2_451_544.999),
            Some(DegradeReason::BeforeCoverage)
        );
        assert_eq!(
            ut1_coverage_departure(&prov, 2_451_546.001),
            Some(DegradeReason::AfterCoverage)
        );
    }
}
