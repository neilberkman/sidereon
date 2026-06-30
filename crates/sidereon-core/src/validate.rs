//! Shared validation helpers for parser and solver boundaries.
//!
//! These functions keep malformed text, non-finite numbers, and invalid civil
//! timestamps out of typed product records and public solver entry points. The
//! computational kernels stay focused on their numerical recipes; callers map
//! [`FieldError`] into their local error enums at the boundary.

#![allow(dead_code)]

use core::str::FromStr;

use crate::id::GnssSatelliteId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("{field} length is {actual}, expected {expected}")]
pub(crate) struct LengthError {
    pub(crate) field: &'static str,
    pub(crate) expected: usize,
    pub(crate) actual: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("{field} integer arithmetic overflow")]
pub(crate) struct ArithmeticError {
    pub(crate) field: &'static str,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub(crate) enum FieldError {
    #[error("{field} is missing")]
    Missing { field: &'static str },
    #[error("{field} is not a finite number")]
    NonFinite { field: &'static str },
    #[error("{field} must be positive")]
    NotPositive { field: &'static str },
    #[error("{field} must be non-negative")]
    Negative { field: &'static str },
    #[error("{field} is out of range")]
    OutOfRange {
        field: &'static str,
        min: f64,
        max: f64,
        upper_inclusive: bool,
    },
    #[error("{field} is not a valid float: {value:?}")]
    FloatParse { field: &'static str, value: String },
    #[error("{field} is not a valid integer: {value:?}")]
    IntParse { field: &'static str, value: String },
    #[error("{field} is not a valid civil date: {year:04}-{month:02}-{day:02}")]
    InvalidCivilDate {
        field: &'static str,
        year: i64,
        month: i64,
        day: i64,
    },
    #[error("{field} is not a valid civil time: {hour:02}:{minute:02}:{second}")]
    InvalidCivilTime {
        field: &'static str,
        hour: i64,
        minute: i64,
        second: f64,
    },
}

impl FieldError {
    pub(crate) const fn field(&self) -> &'static str {
        match self {
            Self::Missing { field }
            | Self::NonFinite { field }
            | Self::NotPositive { field }
            | Self::Negative { field }
            | Self::OutOfRange { field, .. }
            | Self::FloatParse { field, .. }
            | Self::IntParse { field, .. }
            | Self::InvalidCivilDate { field, .. }
            | Self::InvalidCivilTime { field, .. } => field,
        }
    }

    pub(crate) const fn reason(&self) -> &'static str {
        match self {
            Self::Missing { .. } => "missing",
            Self::NonFinite { .. } => "not finite",
            Self::NotPositive { .. } => "not positive",
            Self::Negative { .. } => "negative",
            Self::OutOfRange { .. } => "out of range",
            Self::FloatParse { .. } => "invalid float",
            Self::IntParse { .. } => "invalid integer",
            Self::InvalidCivilDate { .. } => "invalid civil date",
            Self::InvalidCivilTime { .. } => "invalid civil time",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ValidCivil {
    pub(crate) year: i64,
    pub(crate) month: u32,
    pub(crate) day: u32,
    pub(crate) hour: u32,
    pub(crate) minute: u32,
    pub(crate) second: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ValidCivilMicrosecond {
    pub(crate) year: i64,
    pub(crate) month: u32,
    pub(crate) day: u32,
    pub(crate) hour: u32,
    pub(crate) minute: u32,
    pub(crate) second: u32,
    pub(crate) microsecond: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CivilSecondPolicy {
    /// UTC-like labels, including GLONASS UTC, permit a `:60` leap-second label.
    UtcLike,
    /// Continuous system times such as GPS, Galileo, BeiDou, QZSS, IRNSS, TAI,
    /// TT, and TDB do not carry civil leap-second labels.
    Continuous,
}

#[derive(Debug, Clone, Copy)]
struct CivilMinute {
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
}

const CIVIL_YEAR_MIN: i64 = 0;
const CIVIL_YEAR_MAX: i64 = 9999;

impl CivilSecondPolicy {
    const fn allows_leap_second_label(self) -> bool {
        match self {
            Self::UtcLike => true,
            Self::Continuous => false,
        }
    }
}

pub(crate) fn finite(x: f64, field: &'static str) -> Result<f64, FieldError> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(FieldError::NonFinite { field })
    }
}

pub(crate) fn finite_positive(x: f64, field: &'static str) -> Result<f64, FieldError> {
    finite(x, field).and_then(|x| {
        if x > 0.0 {
            Ok(x)
        } else {
            Err(FieldError::NotPositive { field })
        }
    })
}

pub(crate) fn finite_nonneg(x: f64, field: &'static str) -> Result<f64, FieldError> {
    finite(x, field).and_then(|x| {
        if x >= 0.0 {
            Ok(x)
        } else {
            Err(FieldError::Negative { field })
        }
    })
}

pub(crate) fn finite_in_range(
    x: f64,
    min: f64,
    max: f64,
    field: &'static str,
) -> Result<f64, FieldError> {
    finite_in_range_impl(x, min, max, true, field)
}

pub(crate) fn finite_in_range_exclusive_upper(
    x: f64,
    min: f64,
    max: f64,
    field: &'static str,
) -> Result<f64, FieldError> {
    finite_in_range_impl(x, min, max, false, field)
}

pub(crate) fn fraction(x: f64, field: &'static str) -> Result<f64, FieldError> {
    finite_in_range(x, 0.0, 1.0, field)
}

pub(crate) fn second_of_day(x: f64, field: &'static str) -> Result<f64, FieldError> {
    finite_in_range_exclusive_upper(x, 0.0, crate::constants::SECONDS_PER_DAY, field)
}

pub(crate) fn positive_step(x: f64, field: &'static str) -> Result<f64, FieldError> {
    finite_positive(x, field)
}

pub(crate) fn range_order(lo: f64, hi: f64, field: &'static str) -> Result<(), FieldError> {
    if lo <= hi {
        Ok(())
    } else {
        Err(FieldError::OutOfRange {
            field,
            min: lo,
            max: hi,
            upper_inclusive: true,
        })
    }
}

pub(crate) fn clamp_magnitude(x: f64, max_magnitude: f64) -> f64 {
    debug_assert!(max_magnitude.is_finite());
    debug_assert!(max_magnitude > 0.0);

    x.clamp(-max_magnitude, max_magnitude)
}

fn finite_in_range_impl(
    x: f64,
    min: f64,
    max: f64,
    upper_inclusive: bool,
    field: &'static str,
) -> Result<f64, FieldError> {
    debug_assert!(min.is_finite());
    debug_assert!(max.is_finite());
    debug_assert!(min <= max);

    let x = finite(x, field)?;
    let upper_ok = if upper_inclusive { x <= max } else { x < max };
    if x >= min && upper_ok {
        Ok(x)
    } else {
        Err(FieldError::OutOfRange {
            field,
            min,
            max,
            upper_inclusive,
        })
    }
}

pub(crate) fn finite_vec3(v: [f64; 3], field: &'static str) -> Result<[f64; 3], FieldError> {
    finite_slice(&v, field).map(|()| v)
}

pub(crate) fn finite_slice(xs: &[f64], field: &'static str) -> Result<(), FieldError> {
    if xs.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(FieldError::NonFinite { field })
    }
}

pub(crate) fn require_strictly_increasing<I>(
    values: I,
    field: &'static str,
) -> Result<(), FieldError>
where
    I: IntoIterator<Item = f64>,
{
    let mut previous = None;
    for value in values {
        finite(value, field)?;
        if let Some(previous) = previous {
            if value <= previous {
                return Err(FieldError::OutOfRange {
                    field,
                    min: previous,
                    max: value,
                    upper_inclusive: false,
                });
            }
        }
        previous = Some(value);
    }
    Ok(())
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn validate_covariance_psd<const N: usize>(
    m: &[[f64; N]; N],
    field: &'static str,
) -> Result<(), FieldError> {
    for row in m {
        finite_slice(row, field)?;
    }

    let scale = matrix_scale(m);
    let tol = covariance_matrix_tolerance(N, scale);
    for i in 0..N {
        for j in (i + 1)..N {
            if (m[i][j] - m[j][i]).abs() > tol {
                return Err(FieldError::NotPositive { field });
            }
        }
    }

    let mut symmetric_part = [[0.0_f64; N]; N];
    for i in 0..N {
        symmetric_part[i][i] = m[i][i];
        for j in (i + 1)..N {
            let value = 0.5 * (m[i][j] + m[j][i]);
            symmetric_part[i][j] = value;
            symmetric_part[j][i] = value;
        }
    }

    if symmetric_min_eigenvalue(&mut symmetric_part, tol) < -tol {
        return Err(FieldError::NotPositive { field });
    }

    Ok(())
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn validate_covariance_psd_rows(
    rows: &[&[f64]],
    field: &'static str,
) -> Result<(), FieldError> {
    let n = rows.len();
    for row in rows {
        finite_slice(row, field)?;
        debug_assert_eq!(row.len(), n);
    }

    let scale = matrix_rows_scale(rows);
    let tol = covariance_matrix_tolerance(n, scale);
    for i in 0..n {
        for j in (i + 1)..n {
            if (rows[i][j] - rows[j][i]).abs() > tol {
                return Err(FieldError::NotPositive { field });
            }
        }
    }

    let mut symmetric_part = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        symmetric_part[i][i] = rows[i][i];
        for j in (i + 1)..n {
            let value = 0.5 * (rows[i][j] + rows[j][i]);
            symmetric_part[i][j] = value;
            symmetric_part[j][i] = value;
        }
    }

    if symmetric_rows_min_eigenvalue(&mut symmetric_part, tol) < -tol {
        return Err(FieldError::NotPositive { field });
    }

    Ok(())
}

fn matrix_scale<const N: usize>(m: &[[f64; N]; N]) -> f64 {
    let mut scale = 1.0_f64;
    for row in m {
        for value in row {
            scale = scale.max(value.abs());
        }
    }
    scale
}

fn matrix_rows_scale(rows: &[&[f64]]) -> f64 {
    let mut scale = 1.0_f64;
    for row in rows {
        for value in *row {
            scale = scale.max(value.abs());
        }
    }
    scale
}

fn covariance_matrix_tolerance(n: usize, scale: f64) -> f64 {
    let scale = scale.max(1.0);
    (128.0 * f64::EPSILON * (n.max(1) as f64) * scale).max(1.0e-9 * scale)
}

#[allow(clippy::needless_range_loop)]
fn symmetric_min_eigenvalue<const N: usize>(a: &mut [[f64; N]; N], tol: f64) -> f64 {
    let max_sweeps = (16 * N * N).max(32);
    for _ in 0..max_sweeps {
        let mut p = 0usize;
        let mut q = 0usize;
        let mut max_offdiag = 0.0_f64;
        for i in 0..N {
            for j in (i + 1)..N {
                let offdiag = a[i][j].abs();
                if offdiag > max_offdiag {
                    max_offdiag = offdiag;
                    p = i;
                    q = j;
                }
            }
        }

        if max_offdiag <= tol {
            break;
        }

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        if apq == 0.0 {
            break;
        }

        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        for k in 0..N {
            if k != p && k != q {
                let akp = a[k][p];
                let akq = a[k][q];
                let new_kp = c * akp - s * akq;
                let new_kq = s * akp + c * akq;
                a[k][p] = new_kp;
                a[p][k] = new_kp;
                a[k][q] = new_kq;
                a[q][k] = new_kq;
            }
        }

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
    }

    let mut min = f64::INFINITY;
    for (i, row) in a.iter().enumerate() {
        min = min.min(row[i]);
    }
    min
}

#[allow(clippy::needless_range_loop)]
fn symmetric_rows_min_eigenvalue(a: &mut [Vec<f64>], tol: f64) -> f64 {
    let n = a.len();
    let max_sweeps = (16 * n * n).max(32);
    for _ in 0..max_sweeps {
        let mut p = 0usize;
        let mut q = 0usize;
        let mut max_offdiag = 0.0_f64;
        for (i, row) in a.iter().enumerate() {
            for (j, value) in row.iter().enumerate().skip(i + 1) {
                let offdiag = value.abs();
                if offdiag > max_offdiag {
                    max_offdiag = offdiag;
                    p = i;
                    q = j;
                }
            }
        }

        if max_offdiag <= tol {
            break;
        }

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        if apq == 0.0 {
            break;
        }

        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        for k in 0..n {
            if k != p && k != q {
                let akp = a[k][p];
                let akq = a[k][q];
                let new_kp = c * akp - s * akq;
                let new_kq = s * akp + c * akq;
                a[k][p] = new_kp;
                a[p][k] = new_kp;
                a[k][q] = new_kq;
                a[q][k] = new_kq;
            }
        }

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
    }

    let mut min = f64::INFINITY;
    for (i, row) in a.iter().enumerate() {
        min = min.min(row[i]);
    }
    min
}

pub(crate) fn present<T>(value: Option<T>, field: &'static str) -> Result<T, FieldError> {
    value.ok_or(FieldError::Missing { field })
}

pub(crate) fn exact_len<T>(
    xs: &[T],
    expected: usize,
    field: &'static str,
) -> Result<(), LengthError> {
    if xs.len() == expected {
        Ok(())
    } else {
        Err(LengthError {
            field,
            expected,
            actual: xs.len(),
        })
    }
}

pub(crate) fn checked_i64_add(
    lhs: i64,
    rhs: i64,
    field: &'static str,
) -> Result<i64, ArithmeticError> {
    lhs.checked_add(rhs).ok_or(ArithmeticError { field })
}

pub(crate) fn checked_i64_sub(
    lhs: i64,
    rhs: i64,
    field: &'static str,
) -> Result<i64, ArithmeticError> {
    lhs.checked_sub(rhs).ok_or(ArithmeticError { field })
}

pub(crate) fn checked_i64_mul(
    lhs: i64,
    rhs: i64,
    field: &'static str,
) -> Result<i64, ArithmeticError> {
    lhs.checked_mul(rhs).ok_or(ArithmeticError { field })
}

pub(crate) fn strict_f64(s: &str, field: &'static str) -> Result<f64, FieldError> {
    let value = s.trim();
    if value.is_empty() {
        return Err(FieldError::Missing { field });
    }
    let normalized = value.replace(['D', 'd'], "e");
    let parsed = normalized
        .parse::<f64>()
        .map_err(|_| FieldError::FloatParse {
            field,
            value: value.to_string(),
        })?;
    finite(parsed, field)
}

pub(crate) fn strict_int<T>(s: &str, field: &'static str) -> Result<T, FieldError>
where
    T: FromStr,
{
    let value = s.trim();
    if value.is_empty() {
        return Err(FieldError::Missing { field });
    }
    value.parse::<T>().map_err(|_| FieldError::IntParse {
        field,
        value: value.to_string(),
    })
}

pub(crate) fn strict_gnss_satellite_id(
    s: &str,
    field: &'static str,
) -> Result<GnssSatelliteId, FieldError> {
    let value = s.trim();
    if value.is_empty() {
        return Err(FieldError::Missing { field });
    }
    value
        .parse::<GnssSatelliteId>()
        .map_err(|_| FieldError::IntParse {
            field,
            value: value.to_string(),
        })
}

pub(crate) fn civil_datetime_with_second_policy(
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    second: f64,
    second_policy: CivilSecondPolicy,
) -> Result<ValidCivil, FieldError> {
    const FIELD: &str = "civil datetime";
    finite(second, FIELD)?;

    validate_civil_year(year, month, day)?;
    if !(1..=12).contains(&month) {
        return Err(FieldError::InvalidCivilDate {
            field: FIELD,
            year,
            month,
            day,
        });
    }
    let last_day = crate::astro::time::civil::days_in_month(year, month);
    if !(1..=last_day).contains(&day) {
        return Err(FieldError::InvalidCivilDate {
            field: FIELD,
            year,
            month,
            day,
        });
    }
    let time_fields_valid = (0..=23).contains(&hour) && (0..=59).contains(&minute);
    let seconds_valid = (0.0..60.0).contains(&second)
        || (second_policy.allows_leap_second_label()
            && (60.0..61.0).contains(&second)
            && time_fields_valid
            && is_positive_utc_leap_second_label(year, month, day, hour, minute));
    if !time_fields_valid || !seconds_valid {
        return Err(FieldError::InvalidCivilTime {
            field: FIELD,
            hour,
            minute,
            second,
        });
    }

    Ok(ValidCivil {
        year,
        month: month as u32,
        day: day as u32,
        hour: hour as u32,
        minute: minute as u32,
        second,
    })
}

fn validate_civil_year(year: i64, month: i64, day: i64) -> Result<(), FieldError> {
    if (CIVIL_YEAR_MIN..=CIVIL_YEAR_MAX).contains(&year) {
        Ok(())
    } else {
        Err(FieldError::InvalidCivilDate {
            field: "civil datetime",
            year,
            month,
            day,
        })
    }
}

pub(crate) fn civil_datetime_with_decimal_second_policy(
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    second: &str,
    second_policy: CivilSecondPolicy,
) -> Result<ValidCivilMicrosecond, FieldError> {
    const FIELD: &str = "civil datetime";
    let second = second.trim();
    if second.is_empty() {
        return Err(FieldError::Missing { field: FIELD });
    }

    let (whole_second, microsecond) = decimal_second_to_microseconds(second, FIELD)?;
    civil_datetime_with_whole_microsecond_policy(
        CivilMinute {
            year,
            month,
            day,
            hour,
            minute,
        },
        whole_second,
        microsecond,
        second_policy,
    )
}

pub(crate) fn civil_datetime_with_fractional_second_policy(
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    second: f64,
    second_policy: CivilSecondPolicy,
) -> Result<ValidCivilMicrosecond, FieldError> {
    let civil =
        civil_datetime_with_second_policy(year, month, day, hour, minute, second, second_policy)?;

    let whole_second = civil.second.trunc() as i64;
    let microsecond = ((civil.second - whole_second as f64) * 1_000_000.0).round() as i64;
    civil_datetime_with_whole_microsecond_policy(
        CivilMinute {
            year: civil.year,
            month: i64::from(civil.month),
            day: i64::from(civil.day),
            hour: i64::from(civil.hour),
            minute: i64::from(civil.minute),
        },
        whole_second,
        microsecond,
        second_policy,
    )
}

fn civil_datetime_with_whole_microsecond_policy(
    minute_parts: CivilMinute,
    whole_second: i64,
    mut microsecond: i64,
    second_policy: CivilSecondPolicy,
) -> Result<ValidCivilMicrosecond, FieldError> {
    const FIELD: &str = "civil datetime";
    if !(0..=1_000_000).contains(&microsecond) {
        return Err(FieldError::InvalidCivilTime {
            field: FIELD,
            hour: minute_parts.hour,
            minute: minute_parts.minute,
            second: whole_second as f64 + microsecond as f64 / 1_000_000.0,
        });
    }

    let civil = civil_datetime_with_second_policy(
        minute_parts.year,
        minute_parts.month,
        minute_parts.day,
        minute_parts.hour,
        minute_parts.minute,
        whole_second as f64,
        second_policy,
    )?;

    let mut rounded_second = whole_second;
    let rounded_from_subsecond = microsecond == 1_000_000;
    if rounded_from_subsecond {
        rounded_second += 1;
        microsecond = 0;
    }

    let leap_second_label_allowed = second_policy.allows_leap_second_label()
        && is_positive_utc_leap_second_label(
            minute_parts.year,
            minute_parts.month,
            minute_parts.day,
            minute_parts.hour,
            minute_parts.minute,
        );
    if rounded_second <= 59 || (rounded_second == 60 && leap_second_label_allowed) {
        return Ok(ValidCivilMicrosecond {
            year: civil.year,
            month: civil.month,
            day: civil.day,
            hour: civil.hour,
            minute: civil.minute,
            second: rounded_second as u32,
            microsecond: microsecond as u32,
        });
    }

    if rounded_second == 60
        || (rounded_second == 61 && rounded_from_subsecond && leap_second_label_allowed)
    {
        let (year, month, day, hour, minute) =
            carry_to_next_minute(civil.year, civil.month, civil.day, civil.hour, civil.minute)?;
        return Ok(ValidCivilMicrosecond {
            year,
            month,
            day,
            hour,
            minute,
            second: 0,
            microsecond: microsecond as u32,
        });
    }

    Err(FieldError::InvalidCivilTime {
        field: FIELD,
        hour: minute_parts.hour,
        minute: minute_parts.minute,
        second: rounded_second as f64 + microsecond as f64 / 1_000_000.0,
    })
}

fn is_positive_utc_leap_second_label(
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
) -> bool {
    let (Ok(year), Ok(month), Ok(day), Ok(hour), Ok(minute)) = (
        i32::try_from(year),
        i32::try_from(month),
        i32::try_from(day),
        i32::try_from(hour),
        i32::try_from(minute),
    ) else {
        return false;
    };
    crate::astro::time::scales::is_positive_leap_second_label(year, month, day, hour, minute)
}

fn decimal_second_to_microseconds(
    second: &str,
    field: &'static str,
) -> Result<(i64, i64), FieldError> {
    let (whole, fraction) = second
        .split_once('.')
        .map_or((second, None), |(whole, fraction)| (whole, Some(fraction)));
    if whole.is_empty() {
        return Err(FieldError::FloatParse {
            field,
            value: second.to_string(),
        });
    }
    let negative_zero_whole = whole.starts_with('-');
    let whole = whole.parse::<i64>().map_err(|_| FieldError::FloatParse {
        field,
        value: second.to_string(),
    })?;
    let negative_zero_whole = negative_zero_whole && whole == 0;

    let Some(fraction) = fraction else {
        return Ok((whole, 0));
    };
    if fraction.is_empty() || !fraction.bytes().all(|b| b.is_ascii_digit()) {
        return Err(FieldError::FloatParse {
            field,
            value: second.to_string(),
        });
    }

    let mut microsecond = 0i64;
    for i in 0..6 {
        microsecond *= 10;
        microsecond += fraction
            .as_bytes()
            .get(i)
            .map_or(0, |b| i64::from(b - b'0'));
    }
    if fraction.as_bytes().get(6).is_some_and(|b| b - b'0' >= 5) {
        microsecond += 1;
    }
    if negative_zero_whole {
        microsecond = -microsecond.max(1);
    }
    Ok((whole, microsecond))
}

fn carry_to_next_minute(
    mut year: i64,
    mut month: u32,
    mut day: u32,
    mut hour: u32,
    mut minute: u32,
) -> Result<(i64, u32, u32, u32, u32), FieldError> {
    minute += 1;
    if minute < 60 {
        return Ok((year, month, day, hour, minute));
    }
    minute = 0;
    hour += 1;
    if hour < 24 {
        return Ok((year, month, day, hour, minute));
    }
    hour = 0;
    day += 1;
    if i64::from(day) <= crate::astro::time::civil::days_in_month(year, i64::from(month)) {
        return Ok((year, month, day, hour, minute));
    }
    day = 1;
    month += 1;
    if month <= 12 {
        return Ok((year, month, day, hour, minute));
    }
    month = 1;
    year = year
        .checked_add(1)
        .ok_or_else(|| FieldError::InvalidCivilDate {
            field: "civil datetime",
            year,
            month: i64::from(month),
            day: i64::from(day),
        })?;
    validate_civil_year(year, i64::from(month), i64::from(day))?;
    Ok((year, month, day, hour, minute))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_quantity_helpers_accept_valid_inputs() {
        assert_eq!(finite_in_range(1.0, -1.0, 1.0, "bounded"), Ok(1.0));
        assert_eq!(fraction(0.0, "fraction"), Ok(0.0));
        assert_eq!(fraction(1.0, "fraction"), Ok(1.0));
        assert_eq!(second_of_day(86_399.999, "second_of_day"), Ok(86_399.999));
        assert_eq!(positive_step(0.25, "step"), Ok(0.25));
        assert_eq!(range_order(-1.0, 1.0, "range"), Ok(()));
        assert_eq!(range_order(1.0, 1.0, "range"), Ok(()));
        assert_eq!(clamp_magnitude(3.0, 2.0), 2.0);
        assert_eq!(clamp_magnitude(-3.0, 2.0), -2.0);
        assert_eq!(clamp_magnitude(1.5, 2.0), 1.5);
    }

    #[test]
    fn bounded_quantity_helpers_reject_bad_inputs() {
        assert!(matches!(
            fraction(50.0, "fraction"),
            Err(FieldError::OutOfRange {
                field: "fraction",
                min: 0.0,
                max: 1.0,
                upper_inclusive: true
            })
        ));
        assert!(matches!(
            second_of_day(86_400.0, "second_of_day"),
            Err(FieldError::OutOfRange {
                field: "second_of_day",
                min: 0.0,
                max: 86_400.0,
                upper_inclusive: false
            })
        ));
        assert!(matches!(
            positive_step(0.0, "step"),
            Err(FieldError::NotPositive { field: "step" })
        ));
        assert!(matches!(
            range_order(2.0, 1.0, "range"),
            Err(FieldError::OutOfRange {
                field: "range",
                min: 2.0,
                max: 1.0,
                upper_inclusive: true
            })
        ));
        assert!(matches!(
            finite_in_range(f64::NAN, -1.0, 1.0, "bounded"),
            Err(FieldError::NonFinite { field: "bounded" })
        ));
    }

    #[test]
    fn covariance_psd_helper_accepts_positive_semidefinite_matrices() {
        let semidefinite = [[1.0, 1.0], [1.0, 1.0]];
        assert_eq!(validate_covariance_psd(&semidefinite, "covariance"), Ok(()));

        let scaled = [
            [4.0e12, 2.0e12, 0.0],
            [2.0e12, 1.0e12, 0.0],
            [0.0, 0.0, 3.0],
        ];
        assert_eq!(validate_covariance_psd(&scaled, "covariance"), Ok(()));
    }

    #[test]
    fn covariance_psd_helper_rejects_malformed_matrices() {
        let non_finite = [[1.0, f64::NAN], [f64::NAN, 1.0]];
        assert!(matches!(
            validate_covariance_psd(&non_finite, "covariance"),
            Err(FieldError::NonFinite {
                field: "covariance"
            })
        ));

        let asymmetric = [[1.0, 1.0e-5], [0.0, 1.0]];
        assert!(matches!(
            validate_covariance_psd(&asymmetric, "covariance"),
            Err(FieldError::NotPositive {
                field: "covariance"
            })
        ));

        let indefinite = [[1.0, 2.0], [2.0, 1.0]];
        assert!(matches!(
            validate_covariance_psd(&indefinite, "covariance"),
            Err(FieldError::NotPositive {
                field: "covariance"
            })
        ));
    }

    #[test]
    fn covariance_psd_rows_helper_matches_array_helper() {
        let valid = [vec![2.0, 0.25], vec![0.25, 1.0]];
        let valid_rows: Vec<&[f64]> = valid.iter().map(Vec::as_slice).collect();
        assert_eq!(
            validate_covariance_psd_rows(&valid_rows, "covariance"),
            Ok(())
        );

        let indefinite = [vec![1.0, 2.0], vec![2.0, 1.0]];
        let indefinite_rows: Vec<&[f64]> = indefinite.iter().map(Vec::as_slice).collect();
        assert!(matches!(
            validate_covariance_psd_rows(&indefinite_rows, "covariance"),
            Err(FieldError::NotPositive {
                field: "covariance"
            })
        ));
    }

    #[test]
    fn strictly_increasing_helper_rejects_nonfinite_and_nonmonotonic_values() {
        assert_eq!(require_strictly_increasing([1.0, 2.0, 3.0], "time"), Ok(()));
        assert!(matches!(
            require_strictly_increasing([1.0, 1.0], "time"),
            Err(FieldError::OutOfRange {
                field: "time",
                min: 1.0,
                max: 1.0,
                upper_inclusive: false
            })
        ));
        assert!(matches!(
            require_strictly_increasing([1.0, f64::NAN], "time"),
            Err(FieldError::NonFinite { field: "time" })
        ));
    }

    #[test]
    fn civil_datetime_uses_time_system_leap_second_policy() {
        let utc = civil_datetime_with_second_policy(
            2016,
            12,
            31,
            23,
            59,
            60.0,
            CivilSecondPolicy::UtcLike,
        )
        .expect("UTC-like civil time accepts a leap second label");
        assert_eq!(utc.second, 60.0);

        let utc_ordinary = civil_datetime_with_second_policy(
            2016,
            12,
            30,
            23,
            59,
            59.0,
            CivilSecondPolicy::UtcLike,
        )
        .expect("UTC-like civil time accepts an ordinary final minute second");
        assert_eq!(utc_ordinary.second, 59.0);

        assert!(matches!(
            civil_datetime_with_second_policy(
                2016,
                12,
                30,
                23,
                59,
                60.0,
                CivilSecondPolicy::UtcLike
            ),
            Err(FieldError::InvalidCivilTime {
                field: "civil datetime",
                hour: 23,
                minute: 59,
                second: 60.0
            })
        ));

        let gps = civil_datetime_with_second_policy(
            2016,
            12,
            31,
            23,
            59,
            59.0,
            CivilSecondPolicy::Continuous,
        )
        .expect("GPS-like civil time accepts an ordinary final minute second");
        assert_eq!(gps.second, 59.0);

        assert!(matches!(
            civil_datetime_with_second_policy(
                2016,
                12,
                31,
                23,
                59,
                60.0,
                CivilSecondPolicy::Continuous
            ),
            Err(FieldError::InvalidCivilTime {
                field: "civil datetime",
                hour: 23,
                minute: 59,
                second: 60.0
            })
        ));
        assert!(matches!(
            civil_datetime_with_second_policy(
                2016,
                12,
                31,
                23,
                59,
                61.0,
                CivilSecondPolicy::UtcLike
            ),
            Err(FieldError::InvalidCivilTime {
                field: "civil datetime",
                hour: 23,
                minute: 59,
                second: 61.0
            })
        ));
    }

    #[test]
    fn decimal_second_parser_carries_rounded_microseconds() {
        let ordinary = civil_datetime_with_decimal_second_policy(
            2026,
            6,
            17,
            4,
            32,
            "52.9999995",
            CivilSecondPolicy::Continuous,
        )
        .expect("rounded fractional second carries into second");
        assert_eq!(
            ordinary,
            ValidCivilMicrosecond {
                year: 2026,
                month: 6,
                day: 17,
                hour: 4,
                minute: 32,
                second: 53,
                microsecond: 0,
            }
        );

        let day_boundary = civil_datetime_with_decimal_second_policy(
            2026,
            6,
            17,
            23,
            59,
            "59.9999995",
            CivilSecondPolicy::Continuous,
        )
        .expect("rounded fractional second carries across day boundary");
        assert_eq!(
            day_boundary,
            ValidCivilMicrosecond {
                year: 2026,
                month: 6,
                day: 18,
                hour: 0,
                minute: 0,
                second: 0,
                microsecond: 0,
            }
        );

        let utc_ordinary_boundary = civil_datetime_with_decimal_second_policy(
            2026,
            6,
            17,
            23,
            59,
            "59.9999995",
            CivilSecondPolicy::UtcLike,
        )
        .expect("rounded UTC ordinary second carries across day boundary");
        assert_eq!(
            utc_ordinary_boundary,
            ValidCivilMicrosecond {
                year: 2026,
                month: 6,
                day: 18,
                hour: 0,
                minute: 0,
                second: 0,
                microsecond: 0,
            }
        );

        let utc_leap_boundary = civil_datetime_with_decimal_second_policy(
            2016,
            12,
            31,
            23,
            59,
            "59.9999995",
            CivilSecondPolicy::UtcLike,
        )
        .expect("rounded UTC leap-second label stays in the leap minute");
        assert_eq!(
            utc_leap_boundary,
            ValidCivilMicrosecond {
                year: 2016,
                month: 12,
                day: 31,
                hour: 23,
                minute: 59,
                second: 60,
                microsecond: 0,
            }
        );
    }

    #[test]
    fn decimal_second_parser_rejects_bad_fraction() {
        assert!(matches!(
            civil_datetime_with_decimal_second_policy(
                2026,
                6,
                17,
                4,
                32,
                "52.x",
                CivilSecondPolicy::Continuous
            ),
            Err(FieldError::FloatParse {
                field: "civil datetime",
                ..
            })
        ));
    }

    #[test]
    fn decimal_second_parser_rejects_negative_fractional_zero_second() {
        assert!(matches!(
            civil_datetime_with_decimal_second_policy(
                2026,
                6,
                17,
                4,
                32,
                "-0.1",
                CivilSecondPolicy::Continuous
            ),
            Err(FieldError::InvalidCivilTime {
                field: "civil datetime",
                hour: 4,
                minute: 32,
                second
            }) if (second + 0.1).abs() < f64::EPSILON
        ));

        let fractional_zero_second = civil_datetime_with_decimal_second_policy(
            2026,
            6,
            17,
            4,
            32,
            "0.1",
            CivilSecondPolicy::Continuous,
        )
        .expect("positive fractional zero second parses");
        assert_eq!(
            fractional_zero_second,
            ValidCivilMicrosecond {
                year: 2026,
                month: 6,
                day: 17,
                hour: 4,
                minute: 32,
                second: 0,
                microsecond: 100_000,
            }
        );

        let positive_second = civil_datetime_with_decimal_second_policy(
            2026,
            6,
            17,
            4,
            32,
            "52.1",
            CivilSecondPolicy::Continuous,
        )
        .expect("positive decimal second parses");
        assert_eq!(
            positive_second,
            ValidCivilMicrosecond {
                year: 2026,
                month: 6,
                day: 17,
                hour: 4,
                minute: 32,
                second: 52,
                microsecond: 100_000,
            }
        );
    }
}
