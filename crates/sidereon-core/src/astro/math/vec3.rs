//! Small fixed-size 3D vector helpers.
//!
//! These helpers intentionally keep simple, explicit operation order. Callers
//! that need a parity-specific order should use the named variants rather than
//! copy-pasting a local helper.

/// Error returned by checked 3D vector helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Vec3Error {
    /// A vector input or output contained NaN or infinity.
    #[error("invalid vec3 {field}: {reason}")]
    InvalidInput {
        /// Label of the operand or computed result rejected by [`checked_add3`]:
        /// `"a"`, `"b"`, or `"sum"`.
        field: &'static str,
        /// Validation reason emitted by [`checked_add3`]. The current reason is
        /// `"not finite"` for a NaN or infinity in an input or computed sum.
        reason: &'static str,
    },
}

/// Add two finite 3D vectors.
///
/// This infallible primitive is intended for internal parity-sensitive math
/// after public callers have validated inputs. Use [`checked_add3`] at public
/// boundaries or fuzz entry points.
#[inline]
pub fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    debug_assert!(finite3(&a));
    debug_assert!(finite3(&b));
    let out = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    debug_assert!(finite3(&out));
    out
}

/// Checked addition for public/fuzz entry points.
#[inline]
pub fn checked_add3(a: [f64; 3], b: [f64; 3]) -> Result<[f64; 3], Vec3Error> {
    validate_finite3(&a, "a")?;
    validate_finite3(&b, "b")?;
    let out = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    validate_finite3(&out, "sum")?;
    Ok(out)
}

#[inline]
/// Subtract corresponding components in the explicit order used by the
/// position, baseline, relative-state, and finite-difference callers.
///
/// The result is `[a[0] - b[0], a[1] - b[1], a[2] - b[2]]`; this helper does not
/// validate the operands or the result for finiteness.
pub fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
/// Negate each component, returning `[-v[0], -v[1], -v[2]]`.
///
/// The angle and PPP correction callers use this to reverse a position or a
/// local basis direction; no finiteness check is performed.
pub fn neg3(v: [f64; 3]) -> [f64; 3] {
    [-v[0], -v[1], -v[2]]
}

#[inline]
/// Multiply every component by `s` in index order.
///
/// The result is `[v[0] * s, v[1] * s, v[2] * s]`; callers use this for unit
/// vector scaling, time-step integration, and finite differences, and the
/// helper does not validate the operands or result.
pub fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

#[inline]
/// Compute the explicit dot-product reduction
/// `a[0] * b[0] + a[1] * b[1] + a[2] * b[2]`.
///
/// The addition order is retained for floating-point parity in projection,
/// angle, and orbit-geometry calculations.
pub fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
/// Compute the borrowed-vector dot product using the explicit reduction
/// `a[0] * b[0] + a[1] * b[1] + a[2] * b[2]`.
///
/// This preserves the same addition order as [`dot3`] for IOD, Lambert, tide,
/// and reduced-orbit calculations.
pub fn dot3_ref(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
/// Compute the alternate borrowed-vector reduction
/// `a[2] * b[2] + (a[1] * b[1] + a[0] * b[0])`.
///
/// The z term is added to the grouped y/x subtotal, which gives callers a
/// separately selectable floating-point operation order.
pub fn dot3_z_yx_ref(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[2] * b[2] + (a[1] * b[1] + a[0] * b[0])
}

#[inline]
/// Compute the fused alternate reduction
/// `fma(a[2], b[2], fma(a[1], b[1], a[0] * b[0]))`.
///
/// The inner x product seeds the y fusion, and the resulting subtotal is fused
/// with the z product through [`libm::fma`].
pub fn dot3_fused_z_yx_ref(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    libm::fma(a[2], b[2], libm::fma(a[1], b[1], a[0] * b[0]))
}

#[inline]
/// Compute the Euclidean magnitude as `dot3(v, v).sqrt()`.
///
/// No finiteness or zero check is performed; callers that need those checks
/// apply them around this parity-sensitive reduction.
pub fn norm3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}

#[inline]
/// Compute the borrowed-vector Euclidean magnitude as `dot3_ref(v, v).sqrt()`.
///
/// Non-finite and zero results are left for the caller to handle.
pub fn norm3_ref(v: &[f64; 3]) -> f64 {
    dot3_ref(v, v).sqrt()
}

#[inline]
/// Normalize `v` with `scale3(v, 1.0 / n)` when `n = norm3(v)` is positive.
///
/// Returns `None` when `n` is not greater than `0.0`, including a zero or NaN
/// norm. The helper does not make a separate finiteness check before scaling.
pub fn unit3(v: [f64; 3]) -> Option<[f64; 3]> {
    match norm3(v) {
        n if n > 0.0 => Some(scale3(v, 1.0 / n)),
        _ => None,
    }
}

#[inline]
/// Divide each component by the borrowed vector's `norm3_ref` result.
///
/// This helper performs no zero or finiteness check; its IOD, Lambert, and RTN
/// callers establish a usable norm before invoking it.
pub fn unit3_ref_unchecked(v: &[f64; 3]) -> [f64; 3] {
    let n = norm3_ref(v);
    [v[0] / n, v[1] / n, v[2] / n]
}

#[inline]
/// Compute the explicit right-handed cross product in component order.
///
/// The result is `[a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
/// a[0] * b[1] - a[1] * b[0]]`.
pub fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
/// Compute the borrowed-vector cross product with the same component order as
/// [`cross3`].
///
/// The result is `[a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
/// a[0] * b[1] - a[1] * b[0]]`.
pub fn cross3_ref(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn finite3(v: &[f64; 3]) -> bool {
    v.iter().all(|value| value.is_finite())
}

#[inline]
fn validate_finite3(v: &[f64; 3], field: &'static str) -> Result<(), Vec3Error> {
    if finite3(v) {
        Ok(())
    } else {
        Err(Vec3Error::InvalidInput {
            field,
            reason: "not finite",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_dot_orders_match_their_explicit_recipes() {
        let a = [1.25, -2.5, 3.75];
        let b = [-4.0, 5.5, -6.25];

        assert_eq!(
            dot3(a, b).to_bits(),
            (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).to_bits()
        );
        assert_eq!(
            dot3_z_yx_ref(&a, &b).to_bits(),
            (a[2] * b[2] + (a[1] * b[1] + a[0] * b[0])).to_bits()
        );
        assert_eq!(
            dot3_fused_z_yx_ref(&a, &b).to_bits(),
            libm::fma(a[2], b[2], libm::fma(a[1], b[1], a[0] * b[0])).to_bits()
        );
    }

    #[test]
    fn unit3_zero_vector_returns_none() {
        assert_eq!(unit3([0.0, 0.0, 0.0]), None);
    }

    #[test]
    fn checked_add3_rejects_non_finite_inputs_and_outputs() {
        assert_eq!(
            checked_add3([f64::NAN, 0.0, 0.0], [1.0, 2.0, 3.0]),
            Err(Vec3Error::InvalidInput {
                field: "a",
                reason: "not finite"
            })
        );
        assert_eq!(
            checked_add3([1.0, 2.0, 3.0], [f64::INFINITY, 0.0, 0.0]),
            Err(Vec3Error::InvalidInput {
                field: "b",
                reason: "not finite"
            })
        );
        assert_eq!(
            checked_add3([f64::MAX, 0.0, 0.0], [f64::MAX, 0.0, 0.0]),
            Err(Vec3Error::InvalidInput {
                field: "sum",
                reason: "not finite"
            })
        );
    }
}
