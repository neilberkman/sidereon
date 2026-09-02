//! Precise-ephemeris interpolation contract, verification, and policy.
//!
//! # Interpolation contract (for consumers migrating from another interpolator)
//!
//! Sidereon interpolates a precise-ephemeris product with two independent,
//! separately-referenced channels (full detail on
//! [`Sp3::position_at_j2000_seconds`](crate::sp3::Sp3::position_at_j2000_seconds)):
//!
//! - **Position**: a sliding-window Lagrange (Neville) polynomial matching the
//!   RTKLIB `preceph.c` recipe. The window is up to **11 nodes (degree 10)**
//!   centred on the query and clamped to the contiguous run of nodes bracketing
//!   it (never interpolating across a coverage gap). Each node's ECEF position
//!   is rotated about `+z` by `OMEGA_E_DOT * (t_node - query)` into the
//!   query-epoch earth-fixed frame before evaluation.
//! - **Clock**: a not-a-knot cubic spline matching
//!   `scipy.interpolate.CubicSpline(x, y)` with `bc_type="not-a-knot"`,
//!   `extrapolate=True`, fit on the contiguous clock sub-arc containing the
//!   query (the arc is split at each `E` clock-event epoch).
//!
//! **Node axis**: integer seconds since J2000, snapped to whole seconds when
//! within the split-JD roundoff bound and otherwise truncated with the reference
//! policy (the query epoch is **not** quantized). **Units**: the fit is in the
//! file-native units the references consume, kilometers for position and
//! microseconds for clock, and the single unit multiply to SI (meters `* 1000`,
//! seconds `* 1e-6`) happens **after** evaluation. **Coverage**: a query more
//! than one nominal node spacing beyond the node span, or deep inside an
//! interior gap, is rejected
//! ([`crate::Error::EpochOutOfRange`]) rather than extrapolated.
//!
//! A consumer migrating from a global cubic spline over SP3 (for example scipy
//! `CubicSpline` over the whole day) will see the **largest divergence
//! mid-interval** and near-zero divergence at the nodes: a global degree-3 spline
//! and a local degree-10 Lagrange window agree at the sample points but differ by
//! up to hundreds of meters between them, especially near day boundaries and
//! gaps. [`compare_position_series`] quantifies this per epoch so a migration is
//! measured, not guessed.
//!
//! # Recommendation: assert the canonical interpolation; do not add a
//! configurable spline mode
//!
//! A configurable interpolation policy (e.g. a switchable "global cubic-spline
//! position mode") is **not recommended** for sidereon:
//!
//! - **The position recipe is a validated capability, not a free parameter.** The
//!   Neville window is pinned to the RTKLIB reference and validated end-to-end
//!   against PPP truth; a global cubic-spline alternative is a *worse* orbit
//!   interpolator (~200 m error at day boundaries and across gaps) that sidereon
//!   deliberately replaced. Exposing it as a mode would re-introduce a known-bad
//!   result as a supported option.
//! - **It conflicts with the bit-exact philosophy.** Sidereon's contract is one
//!   canonical, byte-reproducible answer per query (the clock channel is a
//!   0-ULP `scipy.CubicSpline` target; the position channel is bit-stable against
//!   RTKLIB). A per-call mode flag multiplies the outputs a downstream must pin
//!   and undermines "there is exactly one number here".
//! - **The real migration need is a bridge, not a mode.** Consumers coming from a
//!   different interpolator do not need sidereon to *emulate* it; they need to
//!   quantify the difference and satisfy themselves the canonical recipe is at
//!   least as good. That is a verification tool ([`compare_position_series`]),
//!   which this module provides, rather than a configuration surface.
//!
//! If a genuinely distinct product (e.g. a documented cubic-spline field for a
//! non-IGS use case) is ever required, it should be a **separate, explicitly
//! named source type** with its own validated reference and its own bit-exact
//! contract, never a mode flag mutating the meaning of the existing
//! `position_at_j2000_seconds`.

use crate::id::GnssSatelliteId;
use crate::observables::{ObservableEphemerisSource, ObservablesError};

/// One epoch of a supplied reference series to compare against.
///
/// `position_ecef_m` is the reference ITRF/IGS ECEF position in meters at
/// `query_j2000_s` (seconds since J2000, in the source's time scale); `clock_s`
/// is the reference satellite clock in seconds, `None` if the reference carries
/// no clock at this epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceState {
    /// Query epoch, seconds since J2000 (source time scale).
    pub query_j2000_s: f64,
    /// Reference ECEF position, meters.
    pub position_ecef_m: [f64; 3],
    /// Reference satellite clock offset, seconds (`None` if absent).
    pub clock_s: Option<f64>,
}

/// Per-epoch divergence between sidereon's interpolation and a reference state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterpolationDivergence {
    /// Query epoch, seconds since J2000.
    pub query_j2000_s: f64,
    /// Sidereon's interpolated ECEF position, meters.
    pub interpolated_position_m: [f64; 3],
    /// The supplied reference ECEF position, meters.
    pub reference_position_m: [f64; 3],
    /// `interpolated - reference` per axis, meters.
    pub position_diff_m: [f64; 3],
    /// Euclidean norm of `position_diff_m`, meters.
    pub position_norm_m: f64,
    /// Sidereon's interpolated clock, seconds (`None` if the source had none).
    pub interpolated_clock_s: Option<f64>,
    /// The supplied reference clock, seconds (`None` if the reference had none).
    pub reference_clock_s: Option<f64>,
    /// `interpolated - reference` clock, seconds; `Some` only when both clocks
    /// are present.
    pub clock_diff_s: Option<f64>,
}

/// Aggregate report over a compared reference series.
#[derive(Debug, Clone, PartialEq)]
pub struct InterpolationComparison {
    /// Per-epoch divergence, in the order of the supplied reference series.
    pub per_epoch: Vec<InterpolationDivergence>,
    /// Largest per-epoch position-difference norm, meters (`0.0` if empty).
    pub max_position_norm_m: f64,
    /// Root-mean-square of the per-epoch position-difference norms, meters
    /// (`0.0` if empty).
    pub rms_position_norm_m: f64,
    /// Largest absolute clock difference over epochs where both clocks exist,
    /// seconds; `None` if no epoch had both.
    pub max_abs_clock_diff_s: Option<f64>,
}

/// Compare sidereon's interpolation of `sat` against a supplied `reference`
/// series, reporting per-epoch and aggregate divergence.
///
/// The source is any [`ObservableEphemerisSource`] (a parsed [`crate::sp3::Sp3`]
/// or a sample-backed [`crate::sp3::PreciseEphemerisSamples`]); it is evaluated
/// at each `reference[i].query_j2000_s` and differenced against
/// `reference[i]`. This lets a consumer migrating from another SP3 interpolator
/// validate the divergence per epoch (largest mid-interval, ~0 at nodes) instead
/// of guessing it.
///
/// The first epoch the source cannot evaluate (out of coverage, unknown
/// satellite, non-finite query) aborts with that [`ObservablesError`]; supply
/// in-coverage epochs. An empty `reference` yields an empty, zero-valued report.
pub fn compare_position_series(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    reference: &[ReferenceState],
) -> Result<InterpolationComparison, ObservablesError> {
    let mut per_epoch = Vec::with_capacity(reference.len());
    let mut max_position_norm_m = 0.0f64;
    let mut sum_sq_norm = 0.0f64;
    let mut max_abs_clock_diff_s: Option<f64> = None;

    for reference_state in reference {
        let state = source.observable_state_at_j2000_s(sat, reference_state.query_j2000_s)?;
        let interpolated_position_m = state.position_ecef_m;
        let reference_position_m = reference_state.position_ecef_m;
        let position_diff_m = [
            interpolated_position_m[0] - reference_position_m[0],
            interpolated_position_m[1] - reference_position_m[1],
            interpolated_position_m[2] - reference_position_m[2],
        ];
        let position_norm_m = (position_diff_m[0] * position_diff_m[0]
            + position_diff_m[1] * position_diff_m[1]
            + position_diff_m[2] * position_diff_m[2])
            .sqrt();

        let clock_diff_s = match (state.clock_s, reference_state.clock_s) {
            (Some(a), Some(b)) => {
                let diff = a - b;
                let abs = diff.abs();
                max_abs_clock_diff_s = Some(max_abs_clock_diff_s.map_or(abs, |m| m.max(abs)));
                Some(diff)
            }
            _ => None,
        };

        max_position_norm_m = max_position_norm_m.max(position_norm_m);
        sum_sq_norm += position_norm_m * position_norm_m;

        per_epoch.push(InterpolationDivergence {
            query_j2000_s: reference_state.query_j2000_s,
            interpolated_position_m,
            reference_position_m,
            position_diff_m,
            position_norm_m,
            interpolated_clock_s: state.clock_s,
            reference_clock_s: reference_state.clock_s,
            clock_diff_s,
        });
    }

    let rms_position_norm_m = if per_epoch.is_empty() {
        0.0
    } else {
        (sum_sq_norm / per_epoch.len() as f64).sqrt()
    };

    Ok(InterpolationComparison {
        per_epoch,
        max_position_norm_m,
        rms_position_norm_m,
        max_abs_clock_diff_s,
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::astro::time::model::{Instant, InstantRepr, JulianDateSplit, TimeScale};
    use crate::sp3::{PreciseEphemerisSample, PreciseEphemerisSamples};
    use crate::GnssSystem;

    const J2000_JD_WHOLE: f64 = 2_451_545.0;
    const SECONDS_PER_DAY: f64 = 86_400.0;

    fn gps(prn: u8) -> GnssSatelliteId {
        GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite id")
    }

    fn sample(j2000_s: f64, pos: [f64; 3], clk: Option<f64>) -> PreciseEphemerisSample {
        let split =
            JulianDateSplit::new(J2000_JD_WHOLE, j2000_s / SECONDS_PER_DAY).expect("valid split");
        PreciseEphemerisSample::new(
            gps(21),
            Instant {
                scale: TimeScale::Gpst,
                repr: InstantRepr::JulianDate(split),
            },
            pos,
            clk,
        )
    }

    fn source() -> PreciseEphemerisSamples {
        let samples: Vec<_> = (0..12)
            .map(|i| {
                let t = i as f64 * 900.0;
                sample(
                    t,
                    [
                        26_000_000.0 - 5.0 * t,
                        1_000_000.0 + 7.0 * t,
                        -3_000_000.0 - 2.0 * t,
                    ],
                    Some(1.0e-6 + 1.0e-10 * t),
                )
            })
            .collect();
        PreciseEphemerisSamples::from_samples(samples).expect("valid source")
    }

    #[test]
    fn self_comparison_is_zero_divergence() {
        let source = source();
        // Reference = the source's own interpolation at a set of covered epochs.
        let epochs = [0.0, 450.0, 900.0, 1_350.0, 4_500.0, 9_900.0];
        let reference: Vec<ReferenceState> = epochs
            .iter()
            .map(|&q| {
                let state = source
                    .observable_state_at_j2000_s(gps(21), q)
                    .expect("covered query");
                ReferenceState {
                    query_j2000_s: q,
                    position_ecef_m: state.position_ecef_m,
                    clock_s: state.clock_s,
                }
            })
            .collect();

        let report = compare_position_series(&source, gps(21), &reference).expect("comparison");
        assert_eq!(report.per_epoch.len(), epochs.len());
        assert_eq!(report.max_position_norm_m.to_bits(), 0.0f64.to_bits());
        assert_eq!(report.rms_position_norm_m.to_bits(), 0.0f64.to_bits());
        assert_eq!(report.max_abs_clock_diff_s, Some(0.0));
        for d in &report.per_epoch {
            assert_eq!(d.position_norm_m.to_bits(), 0.0f64.to_bits());
            assert_eq!(d.clock_diff_s, Some(0.0));
        }
    }

    #[test]
    fn perturbed_reference_reports_the_offset() {
        let source = source();
        let q = 450.0;
        let state = source
            .observable_state_at_j2000_s(gps(21), q)
            .expect("covered query");
        // Offset the reference by a known vector; the norm must come back exact.
        let offset = [3.0, -4.0, 12.0]; // norm 13
        let reference = [ReferenceState {
            query_j2000_s: q,
            position_ecef_m: [
                state.position_ecef_m[0] - offset[0],
                state.position_ecef_m[1] - offset[1],
                state.position_ecef_m[2] - offset[2],
            ],
            clock_s: state.clock_s.map(|c| c - 5.0e-9),
        }];

        let report = compare_position_series(&source, gps(21), &reference).expect("comparison");
        let d = report.per_epoch[0];
        assert_eq!(d.position_diff_m, offset);
        assert!((d.position_norm_m - 13.0).abs() < 1e-9);
        assert_eq!(report.max_position_norm_m, d.position_norm_m);
        assert!((report.max_abs_clock_diff_s.unwrap() - 5.0e-9).abs() < 1e-18);
    }

    #[test]
    fn empty_reference_is_zeroed_report() {
        let source = source();
        let report = compare_position_series(&source, gps(21), &[]).expect("comparison");
        assert!(report.per_epoch.is_empty());
        assert_eq!(report.max_position_norm_m, 0.0);
        assert_eq!(report.rms_position_norm_m, 0.0);
        assert_eq!(report.max_abs_clock_diff_s, None);
    }

    #[test]
    fn out_of_coverage_reference_epoch_errors() {
        let source = source();
        let reference = [ReferenceState {
            query_j2000_s: 1_000_000.0,
            position_ecef_m: [0.0; 3],
            clock_s: None,
        }];
        assert!(compare_position_series(&source, gps(21), &reference).is_err());
    }
}
