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
        field: &'static str,
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
pub fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn neg3(v: [f64; 3]) -> [f64; 3] {
    [-v[0], -v[1], -v[2]]
}

#[inline]
pub fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

#[inline]
pub fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn dot3_ref(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn dot3_z_yx_ref(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[2] * b[2] + (a[1] * b[1] + a[0] * b[0])
}

#[inline]
pub fn dot3_fused_z_yx_ref(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[2].mul_add(b[2], a[1].mul_add(b[1], a[0] * b[0]))
}

#[inline]
pub fn norm3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}

#[inline]
pub fn norm3_ref(v: &[f64; 3]) -> f64 {
    dot3_ref(v, v).sqrt()
}

#[inline]
pub fn unit3(v: [f64; 3]) -> Option<[f64; 3]> {
    match norm3(v) {
        n if n > 0.0 => Some(scale3(v, 1.0 / n)),
        _ => None,
    }
}

#[inline]
pub fn unit3_ref_unchecked(v: &[f64; 3]) -> [f64; 3] {
    let n = norm3_ref(v);
    [v[0] / n, v[1] / n, v[2] / n]
}

#[inline]
pub fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
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
            a[2].mul_add(b[2], a[1].mul_add(b[1], a[0] * b[0]))
                .to_bits()
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
