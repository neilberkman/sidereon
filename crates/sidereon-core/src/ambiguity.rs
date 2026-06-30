//! Shared ambiguity-resolution preparation primitives.
//!
//! RTK (`crate::rtk`) and PPP (`crate::precise_positioning`) both prepare
//! dual-frequency carrier-phase arcs for integer ambiguity resolution: they
//! apply the same cycle-slip policy, round the same Melbourne-Wubbena wide-lane
//! sample mean to an integer, derive the same narrow-lane wavelength/offset
//! algebra, and compare carrier frequencies with the same tolerance. Those
//! shared pieces live here so the two solvers cannot drift apart.
//!
//! The pieces that legitimately differ between the two stay in their own
//! modules: RTK is receiver-aware (base/rover), tags reacquired arcs with a
//! `~raN` suffix, and builds single-/double-difference ambiguity ids, whereas
//! PPP is satellite-only with `sat#N` segment ids. Those id schemes encode
//! different semantics and are not unified here.

use core::fmt;

use crate::combinations::{self, IonosphereFreeError};
use crate::constants::C_M_S;
use crate::tolerances::FREQUENCY_MATCH_EPS_HZ;

/// A carrier-phase ambiguity identifier.
///
/// Ambiguity ids label a continuous integer-ambiguity arc. Unlike a
/// [`GnssSatelliteId`](crate::id::GnssSatelliteId) they are not a fixed grammar:
/// RTK composes single-/double-difference ids that fold the base/rover arc ids
/// and reference satellite (`G01:base=...,rover=...`, `...|ref=...`), while PPP
/// uses `sat#N` segment ids. The type is therefore an opaque ordered string
/// wrapper; it exists so the solvers cannot confuse an ambiguity id with a raw
/// satellite token, and so the per-arc maps are keyed by an intent-revealing
/// type. The wrapped string is preserved byte-for-byte, so ordering, equality,
/// and the [`Display`](fmt::Display) rendering match the underlying token
/// exactly (the value the NIF boundary marshals).
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct AmbiguityId(String);

impl AmbiguityId {
    /// Wrap an already-formatted ambiguity-id token.
    pub(crate) fn new(token: impl Into<String>) -> Self {
        Self(token.into())
    }

    /// Borrow the underlying token, e.g. for comparison against a satellite id.
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the wrapper, yielding the token for the NIF I/O boundary.
    pub(crate) fn into_string(self) -> String {
        self.0
    }

    /// Replace the token in place, reusing the existing heap buffer. The hot
    /// double-difference row builders reuse a pooled [`AmbiguityId`] per scratch
    /// row, so the per-solve allocation budget depends on this not reallocating.
    pub(crate) fn assign(&mut self, token: &str) {
        self.0.clear();
        self.0.push_str(token);
    }

    /// Empty the token in place, reusing the heap buffer, before composing a
    /// multi-part id with [`push_str`](Self::push_str).
    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }

    /// Append a fragment to the token in place (after [`clear`](Self::clear)),
    /// so a composed double-difference id is built without a fresh allocation.
    pub(crate) fn push_str(&mut self, fragment: &str) {
        self.0.push_str(fragment);
    }
}

impl fmt::Display for AmbiguityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Policy applied when a cycle slip is detected while preparing an ambiguity arc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleSlipPolicy {
    /// Fail the whole solve when any slip is detected.
    Error,
    /// Drop the affected satellite from the solve.
    DropSatellite,
    /// Split the arc into a fresh ambiguity at the slip.
    SplitArc,
}

/// Narrow-lane wavelength/offset parameters for a fixed wide-lane integer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NarrowLaneParams {
    pub(crate) wavelength_m: f64,
    pub(crate) offset_m: f64,
    pub(crate) f1_hz: f64,
    pub(crate) f2_hz: f64,
}

/// Narrow-lane wavelength/offset for a fixed wide-lane integer, in the exact
/// operation order both solvers' frozen-bits goldens were captured with.
pub(crate) fn narrow_lane_params(
    f1_hz: f64,
    f2_hz: f64,
    wide_lane_cycles: f64,
) -> Result<NarrowLaneParams, IonosphereFreeError> {
    let gamma = combinations::gamma(f1_hz, f2_hz)?;
    let beta = gamma - 1.0;
    let lambda2 = C_M_S / f2_hz;
    Ok(NarrowLaneParams {
        wavelength_m: C_M_S / (f1_hz + f2_hz),
        offset_m: beta * lambda2 * wide_lane_cycles,
        f1_hz,
        f2_hz,
    })
}

/// Why a wide-lane cycle-sample mean could not be fixed to an integer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum WideLaneEstimateError {
    /// Fewer usable cycle samples than the configured minimum.
    TooFewEpochs { count: usize, minimum: usize },
    /// The sample mean did not round within tolerance of an integer.
    NotInteger { mean_cycles: f64, fixed_cycles: i64 },
}

/// Mean-to-integer wide-lane ambiguity estimation shared by RTK and PPP. The
/// running sum is a left fold from `0.0` to match both callers' captured bits.
pub(crate) fn estimate_wide_lane_integer(
    cycles: &[f64],
    min_epochs: usize,
    tolerance_cycles: f64,
) -> Result<i64, WideLaneEstimateError> {
    if cycles.len() < min_epochs {
        return Err(WideLaneEstimateError::TooFewEpochs {
            count: cycles.len(),
            minimum: min_epochs,
        });
    }

    let mut sum = 0.0;
    for &cycle in cycles {
        sum += cycle;
    }
    let mean = sum / cycles.len() as f64;
    let fixed = mean.round() as i64;

    if (mean - fixed as f64).abs() <= tolerance_cycles {
        Ok(fixed)
    } else {
        Err(WideLaneEstimateError::NotInteger {
            mean_cycles: mean,
            fixed_cycles: fixed,
        })
    }
}

/// Whether two carrier frequencies agree within the GNSS frequency-match tolerance.
pub(crate) fn frequencies_match(a: f64, b: f64) -> bool {
    (a - b).abs() <= FREQUENCY_MATCH_EPS_HZ
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrow_lane_params_match_reference_bits() {
        let f1 = f64::from_bits(0x41d779c018000000);
        let f2 = f64::from_bits(0x41d24aec20000000);
        let params = narrow_lane_params(f1, f2, 3.0).unwrap();
        assert_eq!(params.wavelength_m.to_bits(), 0x3fbb614bed5136b9);
        assert_eq!(params.offset_m.to_bits(), 0x3ff21e814dfd4618);
    }

    #[test]
    fn wide_lane_estimate_rounds_and_gates() {
        assert_eq!(
            estimate_wide_lane_integer(&[2.95, 3.02, 3.0], 2, 0.25),
            Ok(3)
        );
        assert_eq!(
            estimate_wide_lane_integer(&[3.0], 2, 0.25),
            Err(WideLaneEstimateError::TooFewEpochs {
                count: 1,
                minimum: 2,
            })
        );
        assert!(matches!(
            estimate_wide_lane_integer(&[3.4, 3.5, 3.6], 2, 0.05),
            Err(WideLaneEstimateError::NotInteger { .. })
        ));
    }

    #[test]
    fn ambiguity_id_preserves_token_bytes() {
        let id = AmbiguityId::new("G01:base=G01~ra1,rover=G01");
        // Display and as_str render the wrapped token verbatim.
        assert_eq!(id.as_str(), "G01:base=G01~ra1,rover=G01");
        assert_eq!(id.to_string(), "G01:base=G01~ra1,rover=G01");
        // into_string yields the exact bytes the NIF boundary marshals.
        assert_eq!(id.clone().into_string(), "G01:base=G01~ra1,rover=G01");
        // Ordering and equality follow the underlying string byte order.
        assert!(AmbiguityId::new("G01") < AmbiguityId::new("G02"));
        assert_eq!(AmbiguityId::new("E12"), AmbiguityId::new("E12"));
    }

    #[test]
    fn ambiguity_id_in_place_assign_matches_owned_construction() {
        // The hot row builders reuse one id per scratch row: assign and the
        // clear/push_str compose must produce the exact same token bytes as a
        // freshly-allocated id.
        let mut reused = AmbiguityId::default();
        reused.assign("G07");
        assert_eq!(reused, AmbiguityId::new("G07"));

        // Reassigning a longer token then a shorter one tracks the content
        // exactly (the retained buffer capacity does not leak into the value).
        reused.assign("G07:base=G07~ra1,rover=G07");
        assert_eq!(reused.as_str(), "G07:base=G07~ra1,rover=G07");
        reused.assign("E05");
        assert_eq!(reused.as_str(), "E05");

        let mut composed = AmbiguityId::default();
        composed.clear();
        composed.push_str("G07");
        composed.push_str("|ref=");
        composed.push_str("G04");
        assert_eq!(composed, AmbiguityId::new("G07|ref=G04"));
    }

    #[test]
    fn frequency_match_uses_named_tolerance() {
        assert!(frequencies_match(1.0e9, 1.0e9 + 5.0e-7));
        assert!(!frequencies_match(1.0e9, 1.0e9 + 1.0e-3));
    }
}
