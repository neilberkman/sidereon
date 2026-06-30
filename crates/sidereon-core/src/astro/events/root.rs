//! Shared scalar-crossing refinement primitives.
//!
//! These helpers own the bisection mechanics used by event-style code: keep a
//! bracket whose endpoint scalar values are on opposite sides of zero, choose a
//! midpoint, and retain the half bracket containing the crossing.

/// Return true when two scalar samples bracket a zero crossing.
///
/// Zero is treated as the non-negative side. This matches the legacy pass
/// finder semantics, where a sample exactly on the mask is considered visible.
pub fn sign_change_bracketed(a: f64, b: f64) -> Result<bool, RootError> {
    validate_finite("bracket.low_value", a)?;
    validate_finite("bracket.high_value", b)?;
    Ok(!same_sign(a, b))
}

/// Error returned when scalar root-bracketing inputs leave the finite domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RootError<E = core::convert::Infallible> {
    /// A scalar endpoint, sample, or midpoint value was non-finite.
    InvalidInput {
        /// Name of the malformed field.
        field: &'static str,
        /// Stable validation reason.
        reason: &'static str,
    },
    /// The caller-provided fallible predicate returned an error.
    Predicate(E),
}

impl<E: core::fmt::Display> core::fmt::Display for RootError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid root input {field}: {reason}")
            }
            Self::Predicate(error) => write!(f, "root predicate failed: {error}"),
        }
    }
}

impl<E: core::fmt::Debug + core::fmt::Display> std::error::Error for RootError<E> {}

fn invalid_input<E>(field: &'static str, reason: &'static str) -> RootError<E> {
    RootError::InvalidInput { field, reason }
}

fn validate_finite<E>(field: &'static str, value: f64) -> Result<f64, RootError<E>> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid_input(field, "not finite"))
    }
}

/// Refine a zero crossing for a fixed number of bisection iterations.
///
/// `value_at` returns the signed scalar value at an epoch-like point, and
/// `midpoint` supplies the midpoint arithmetic for that point type.
pub fn bisect_crossing_by_iterations<T, F, M>(
    low: T,
    high: T,
    iterations: usize,
    value_at: F,
    midpoint: M,
) -> Result<T, RootError>
where
    T: Copy + PartialEq,
    F: FnMut(T) -> f64,
    M: FnMut(T, T) -> T,
{
    let mut remaining = iterations;
    bisect_crossing_while(low, high, value_at, midpoint, |_, _| {
        if remaining == 0 {
            false
        } else {
            remaining -= 1;
            true
        }
    })
}

/// Refine a zero crossing until `within_tolerance` accepts the active bracket.
///
/// The predicate receives `(low, high)` and should return true once no more
/// refinement is required.
pub fn bisect_crossing_until<T, F, M, W>(
    low: T,
    high: T,
    value_at: F,
    midpoint: M,
    mut within_tolerance: W,
) -> Result<T, RootError>
where
    T: Copy + PartialEq,
    F: FnMut(T) -> f64,
    M: FnMut(T, T) -> T,
    W: FnMut(T, T) -> bool,
{
    bisect_crossing_while(low, high, value_at, midpoint, |lo, hi| {
        !within_tolerance(lo, hi)
    })
}

/// Refine a zero crossing until `within_tolerance` accepts the active bracket,
/// allowing predicate evaluation to abort the refinement.
pub fn try_bisect_crossing_until<T, F, M, W, E>(
    low: T,
    high: T,
    value_at: F,
    midpoint: M,
    mut within_tolerance: W,
) -> Result<T, RootError<E>>
where
    T: Copy + PartialEq,
    F: FnMut(T) -> Result<f64, E>,
    M: FnMut(T, T) -> T,
    W: FnMut(T, T) -> bool,
{
    try_bisect_crossing_while(low, high, value_at, midpoint, |lo, hi| {
        !within_tolerance(lo, hi)
    })
}

fn bisect_crossing_while<T, F, M, C>(
    low: T,
    high: T,
    mut value_at: F,
    mut midpoint: M,
    mut keep_refining: C,
) -> Result<T, RootError>
where
    T: Copy + PartialEq,
    F: FnMut(T) -> f64,
    M: FnMut(T, T) -> T,
    C: FnMut(T, T) -> bool,
{
    let mut lo = low;
    let mut hi = high;
    let mut value_lo = validate_finite("bracket.low_value", value_at(lo))?;
    validate_finite("bracket.high_value", value_at(hi))?;

    while keep_refining(lo, hi) {
        let mid = midpoint(lo, hi);
        if mid == lo || mid == hi {
            validate_finite("bracket.mid_value", value_at(mid))?;
            return Ok(mid);
        }
        let value_mid = validate_finite("bracket.mid_value", value_at(mid))?;
        if value_mid == 0.0 {
            return Ok(mid);
        }
        if same_sign(value_lo, value_mid) {
            lo = mid;
            value_lo = value_mid;
        } else {
            hi = mid;
        }
    }

    let mid = midpoint(lo, hi);
    validate_finite("bracket.mid_value", value_at(mid))?;
    Ok(mid)
}

fn try_bisect_crossing_while<T, F, M, C, E>(
    low: T,
    high: T,
    mut value_at: F,
    mut midpoint: M,
    mut keep_refining: C,
) -> Result<T, RootError<E>>
where
    T: Copy + PartialEq,
    F: FnMut(T) -> Result<f64, E>,
    M: FnMut(T, T) -> T,
    C: FnMut(T, T) -> bool,
{
    let mut lo = low;
    let mut hi = high;
    let mut value_lo = validate_finite(
        "bracket.low_value",
        value_at(lo).map_err(RootError::Predicate)?,
    )?;
    validate_finite(
        "bracket.high_value",
        value_at(hi).map_err(RootError::Predicate)?,
    )?;

    while keep_refining(lo, hi) {
        let mid = midpoint(lo, hi);
        if mid == lo || mid == hi {
            validate_finite(
                "bracket.mid_value",
                value_at(mid).map_err(RootError::Predicate)?,
            )?;
            return Ok(mid);
        }
        let value_mid = validate_finite(
            "bracket.mid_value",
            value_at(mid).map_err(RootError::Predicate)?,
        )?;
        if value_mid == 0.0 {
            return Ok(mid);
        }
        if same_sign(value_lo, value_mid) {
            lo = mid;
            value_lo = value_mid;
        } else {
            hi = mid;
        }
    }

    let mid = midpoint(lo, hi);
    validate_finite(
        "bracket.mid_value",
        value_at(mid).map_err(RootError::Predicate)?,
    )?;
    Ok(mid)
}

fn same_sign(a: f64, b: f64) -> bool {
    (a >= 0.0 && b >= 0.0) || (a < 0.0 && b < 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn midpoint(a: f64, b: f64) -> f64 {
        (a + b) * 0.5
    }

    #[test]
    fn sign_change_bracket_uses_zero_as_non_negative_side() {
        assert!(sign_change_bracketed(-1.0, 1.0).expect("finite bracket"));
        assert!(sign_change_bracketed(-1.0, 0.0).expect("finite bracket"));
        assert!(sign_change_bracketed(0.0, -1.0).expect("finite bracket"));
        assert!(!sign_change_bracketed(0.0, 1.0).expect("finite bracket"));
        assert!(!sign_change_bracketed(1.0, 0.0).expect("finite bracket"));
    }

    #[test]
    fn fixed_iteration_bisection_refines_crossing() {
        let crossing = bisect_crossing_by_iterations(0.0, 1.0, 4, |x| x - 0.3, midpoint)
            .expect("finite bisection");

        assert_eq!(crossing.to_bits(), 0.28125_f64.to_bits());
    }

    #[test]
    fn tolerance_bisection_refines_to_requested_bracket_width() {
        let crossing = bisect_crossing_until(
            1.0,
            2.0,
            |x| x * x - 2.0,
            midpoint,
            |lo, hi| (hi - lo).abs() <= 1.0e-12,
        )
        .expect("finite bisection");

        assert!((crossing - 2.0_f64.sqrt()).abs() <= 5.0e-13);
    }

    #[test]
    fn bisection_returns_exact_midpoint_root() {
        let crossing = bisect_crossing_by_iterations(0.0, 2.0, 8, |x| x - 1.0, midpoint)
            .expect("finite bisection");

        assert_eq!(crossing.to_bits(), 1.0_f64.to_bits());

        let crossing = try_bisect_crossing_until(
            0.0,
            2.0,
            |x| Ok::<f64, ()>(x - 1.0),
            midpoint,
            |lo, hi| (hi - lo).abs() <= 1.0e-12,
        )
        .expect("exact midpoint root should resolve");

        assert_eq!(crossing.to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn bisection_stops_when_midpoint_cannot_shrink_bracket() {
        let high = 1.0_f64;
        let low = f64::from_bits(high.to_bits() - 1);
        let max_iterations = 64;
        let mut value_calls = 0;

        let crossing = bisect_crossing_by_iterations(
            low,
            high,
            max_iterations,
            |x| {
                value_calls += 1;
                x - high
            },
            midpoint,
        )
        .expect("finite bisection");

        assert_eq!(crossing.to_bits(), high.to_bits());
        assert!(value_calls < max_iterations);
    }

    #[test]
    fn fallible_bisection_returns_predicate_errors() {
        let err = try_bisect_crossing_until(
            0.0,
            2.0,
            |x| {
                if x == 1.0 {
                    Err("predicate")
                } else {
                    Ok(x - 1.0)
                }
            },
            midpoint,
            |lo, hi| (hi - lo).abs() <= 0.25,
        )
        .expect_err("midpoint error must abort refinement");

        assert_eq!(err, RootError::Predicate("predicate"));
    }
}
