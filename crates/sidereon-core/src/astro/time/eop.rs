//! Time / EOP validity + provenance API.
//!
//! Leap-second and UT1/EOP tables carry source + effective date + coverage
//! interval, and the library exposes a strict vs permissive mode. Strict mode
//! errors outside table coverage; permissive mode may clamp/extrapolate and
//! marks the result degraded.
//!
//! The delta-T / UT1-UTC interpolation in [`crate::astro::time::scales`] clamps to the
//! embedded table edges outside its range. That clamp is required (and bit-exact)
//! for Skyfield parity, so the numerics are never altered. Instead, the policy is
//! ENFORCED on the public conversion path
//! [`crate::astro::time::scales::TimeScales::from_utc_validated`], which classifies the
//! result against coverage under a [`ValidityMode`]:
//!
//! - malformed or non-finite inputs return [`CoverageError::InvalidInput`]
//!   before the parity-critical arithmetic runs,
//! - [`ValidityMode::Strict`] returns [`CoverageError`] outside coverage (the
//!   clamped value is never handed back), and
//! - [`ValidityMode::Permissive`] returns the clamped value paired with a
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
    /// Errors outside table coverage. Use for precision GNSS pipelines.
    Strict,
    /// Clamps/extrapolates outside coverage and marks the result degraded.
    /// This is the historical Skyfield-parity behaviour and the default.
    #[default]
    Permissive,
}

/// Reason a result was marked degraded under [`ValidityMode::Permissive`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradeReason {
    /// Instant precedes the first covered table entry; value clamped to the edge.
    BeforeCoverage,
    /// Instant follows the last covered table entry; value clamped to the edge.
    AfterCoverage,
}

/// Invalid civil-time input kind for time-scale conversion boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeScaleInputErrorKind {
    Missing,
    NonFinite,
    NotPositive,
    Negative,
    OutOfRange,
    FloatParse,
    IntParse,
    InvalidCivilDate,
    InvalidCivilTime,
}

/// A value paired with whether it was produced inside valid table coverage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Validated<T> {
    /// The computed value (clamped/extrapolated if `degraded` is `Some`).
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

    /// A value produced outside coverage (clamped/extrapolated).
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

/// Error returned by strict-mode coverage checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageError {
    /// Time-scale conversion input is malformed or outside its accepted domain.
    InvalidInput {
        field: &'static str,
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

    let reason = if jd_tt < prov.first_jd_tt {
        Some(DegradeReason::BeforeCoverage)
    } else if jd_tt > prov.last_jd_tt {
        Some(DegradeReason::AfterCoverage)
    } else {
        None
    };

    match (mode, reason) {
        (ValidityMode::Strict, Some(r)) => Err(CoverageError::OutsideCoverage(r)),
        (_, r) => Ok(r),
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
}
