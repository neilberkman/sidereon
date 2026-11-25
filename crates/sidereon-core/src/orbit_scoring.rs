//! Orbit accuracy against a final precise product, by prediction horizon.
//!
//! [`score_orbit`] differences an orbit source against a reference SP3 product
//! issued later (a rapid or final product) and reports the differences per
//! satellite and per constellation as a function of prediction horizon: the
//! time elapsed since the source's prediction origin, the last instant its orbit
//! rests on observations. The source is anything that implements
//! [`OrbitSource`]. Three are provided: a predicted SP3 product such as the
//! predicted half of an ultra-rapid ([`PredictedSp3`]), a broadcast ephemeris
//! ([`BroadcastOrbit`]), and a caller's own trajectory ([`OrbitTrajectory`]).
//!
//! # Orbit frame
//!
//! Each position difference `source - reference` is resolved into radial,
//! transverse and normal components of the reference orbit:
//!
//! - radial `R` along the reference position,
//! - normal `N` along the orbital angular momentum `r x v`,
//! - transverse `T = N x R`, in the orbit plane and within the flight-path
//!   angle of the velocity.
//!
//! This is the frame [`crate::astro::covariance::rtn_to_eci_rotation`] builds,
//! and Orekit's `LOFType.QSW` (`Q = R`, `S = T`, `W = N`). Orekit's
//! `LOFType.TNW` is a different frame: its first axis is the velocity itself and
//! its second axis is `W x T`, so for a circular orbit `T_TNW = T` and
//! `N_TNW = -R`, and for an eccentric orbit the two in-plane axes are rotated
//! from `(R, T)` by the flight-path angle. Along-track and cross-track are other
//! names for `T` and `N` here.
//!
//! The reference velocity is a centered difference of the reference position
//! (one-sided where only one neighbour exists). An SP3 position is Earth-fixed,
//! so that difference is the velocity relative to the rotating Earth. With
//! [`TriadVelocity::Inertial`] (the default) the transport term `omega x r` of
//! the nominal Earth rotation is added before the frame is built, which gives
//! the orbital frame; this neglects polar motion and the excess length of day,
//! a tilt of the frame below 3e-6 rad. With [`TriadVelocity::EarthFixed`] the
//! frame is built from the Earth-fixed velocity as it stands, which is what
//! [`crate::broadcast_comparison`] has always done: radial and the 3D magnitude
//! are the same, and `T` and `N` are rotated about the radial axis by the angle
//! between the two velocities.
//!
//! # Signal-in-space range error
//!
//! The orbit-only and combined signal-in-space range errors follow Montenbruck,
//! Steigenberger and Hauschild (2015, 2018):
//!
//! ```text
//! SISRE(orb) = sqrt( w_R^2 R^2 + w_AC^2 (T^2 + N^2) )
//! SISRE      = sqrt( (w_R R - c dt)^2 + w_AC^2 (T^2 + N^2) )
//! ```
//!
//! with `c dt` the clock difference `source - reference` in meters. The weights
//! are the global averages of the line-of-sight projection for the altitude of
//! the constellation, as tabulated by Montenbruck et al. (2015) and used in the
//! 2018 assessment ([`SisreWeights::montenbruck_2018`]), or caller-supplied.
//! Both errors depend on `T` and `N` only through `T^2 + N^2`, so they are the
//! same for either [`TriadVelocity`].
//!
//! A clock difference between two products carries the offset between their
//! reference clocks, which a receiver clock absorbs. With
//! [`ClockDatum::EpochSystemMedian`] (the default) the median of the clock
//! differences of a constellation's satellites at an epoch is removed from each
//! of them, and an epoch with a single clocked satellite of a constellation has
//! no clock error to report. A source clock that already carries the periodic
//! relativistic term, as a broadcast clock does, is compared with the reference
//! clock plus the same term.
//!
//! # Reference points
//!
//! An SP3 position is the satellite's centre of mass. A broadcast position is
//! the antenna phase centre, up to a few meters nearer the Earth. The comparison applies no antenna offset, so for a
//! broadcast source the radial mean carries that offset;
//! [`OrbitScoreReport::source_reference_point`] states which point the source
//! refers to.
//!
//! # Coverage
//!
//! Every satellite and epoch asked for is accounted for. A pair that is not
//! scored is counted under the reason in [`ScoreCoverage`], or, with
//! [`GapPolicy::Refuse`], returned as the typed error naming the satellite, the
//! epoch and the product. A source in a different time system or coordinate
//! frame than the reference is refused with a typed error.
//!
//! # References
//!
//! - O. Montenbruck, P. Steigenberger, A. Hauschild, "Multi-GNSS signal-in-space
//!   range error assessment - Methodology and results", Advances in Space
//!   Research 61(12), 2018, 3020-3038, doi:10.1016/j.asr.2018.03.041.
//! - O. Montenbruck, P. Steigenberger, A. Hauschild, "Broadcast versus precise
//!   ephemerides: a multi-GNSS perspective", GPS Solutions 19, 2015, 321-333,
//!   doi:10.1007/s10291-014-0390-8.
//! - Orekit 13.1.10 `org.orekit.frames.LOFType` (`QSW`, `TNW`), the reference the
//!   frame projection is tested against; the fixture is written by
//!   `fixtures-generators/orekit_lof_oracle`.

use std::collections::BTreeMap;

use crate::astro::constants::earth::WGS84_A_M;
use crate::astro::constants::MU_EARTH;
use crate::astro::math::vec3::{cross3, dot3, norm3, scale3, sub3, unit3};
use crate::astro::time::model::{Instant, TimeScale};
use crate::constants::{C_M_S, OMEGA_E_DOT_RAD_S};
use crate::ephemeris::{BroadcastEphemeris, Sp3};
use crate::frame_catalog::{self, FrameCatalogError, TerrestrialFrame, TerrestrialPositionM};
use crate::observables::ObservableEphemerisSource;
use crate::sp3::{
    instant_to_j2000_seconds, record_epoch_j2000_seconds, sp3_coordinate_label_frame,
    sp3_decimal_year, Sp3FrameReconciliationOptions, Sp3TimeSystem,
};
use crate::{Error, GnssSatelliteId, GnssSystem};

// ---------------------------------------------------------------------------
// Difference kernel shared with `broadcast_comparison`.
// ---------------------------------------------------------------------------

/// Combine the neighbour positions into a velocity: a centered difference when
/// both neighbours are available, a one-sided difference (using the epoch position
/// `r0`) when only one is, and `None` when neither is. Pure arithmetic, split out
/// from the evaluation so the operation order can be pinned independently.
pub(crate) fn finite_difference_velocity(
    rp: Option<[f64; 3]>,
    rm: Option<[f64; 3]>,
    r0: [f64; 3],
    half_s: f64,
) -> Option<[f64; 3]> {
    match (rp, rm) {
        (Some(rp), Some(rm)) => Some(scale3(sub3(rp, rm), 1.0 / (2.0 * half_s))),
        (Some(rp), None) => Some(scale3(sub3(rp, r0), 1.0 / half_s)),
        (None, Some(rm)) => Some(scale3(sub3(r0, rm), 1.0 / half_s)),
        (None, None) => None,
    }
}

/// The largest magnitude in `values`, `0.0` for none; a NaN is passed over.
fn largest_magnitude(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |largest, x| largest.max(x.abs()))
}

/// The exponent `e` with `2^e <= largest < 2^(e + 1)`, for a positive finite
/// `largest`.
fn binary_exponent(largest: f64) -> Option<i32> {
    (largest.is_finite() && largest > 0.0).then(|| libm::ilogb(largest))
}

/// `v` scaled by the power of two `2^-e` that brings its largest component
/// into `[1, 2)`, with `e`; `None` for a zero or non-finite vector.
///
/// Scaling by a power of two is exact, so a direction formed from the result
/// rounds exactly as the direction of `v` does wherever the squares and
/// products of `v` neither overflow nor underflow, and stays finite where
/// they would. A component more than `2^1022` times smaller than the largest
/// loses low bits to the subnormal range, far below the last place of the
/// direction or the length.
fn exponent_split(v: [f64; 3]) -> Option<([f64; 3], i32)> {
    if !finite3(v) {
        return None;
    }
    let exponent = binary_exponent(largest_magnitude(&v))?;
    Some((v.map(|c| libm::scalbn(c, -exponent)), exponent))
}

/// The vector of [`exponent_split`] without its exponent.
fn exponent_normalized(v: [f64; 3]) -> Option<[f64; 3]> {
    exponent_split(v).map(|(scaled, _)| scaled)
}

/// `|v|` with no overflow or underflow in the squares: [`norm3`] of `v` scaled
/// by a power of two, scaled back. It equals [`norm3`] bit for bit wherever
/// that neither overflows nor underflows.
pub(crate) fn scaled_norm3(v: [f64; 3]) -> f64 {
    match binary_exponent(largest_magnitude(&v)) {
        Some(exponent) => libm::scalbn(norm3(v.map(|c| libm::scalbn(c, -exponent))), exponent),
        None => norm3(v),
    }
}

/// `a b - c d` within about a unit in the last place of the result however
/// far the two products cancel: the rounding error of `c d` is recovered
/// exactly with a fused multiply-add and added back (Kahan). That holds where
/// neither product overflows or underflows.
fn difference_of_products(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let cd = c * d;
    let cd_error = libm::fma(-c, d, cd);
    libm::fma(a, b, -cd) + cd_error
}

/// `a b - c d` of four finite factors of any magnitude, as `m` and `e` with
/// `a b - c d = m 2^e` to about a unit in the last place of `m`; `(0.0, 0)`
/// where both products are zero.
///
/// Each factor is scaled by a power of two so that the larger product falls
/// in `[1, 4)` and both products keep their exponents relative to each other,
/// and [`difference_of_products`] of the scaled factors gives `m`. Neither
/// scaled product overflows. The smaller one reaches the subnormal range only
/// where it is more than `2^1020` times smaller than the larger, far below the
/// last place of `m`; and two scaled products that cancel are multiples of
/// `2^-105`, so a difference that is not zero is at least that and `m` never
/// underflows.
fn scaled_difference_of_products(a: f64, b: f64, c: f64, d: f64) -> (f64, i32) {
    let product_exponent =
        |x: f64, y: f64| (x != 0.0 && y != 0.0).then(|| libm::ilogb(x) + libm::ilogb(y));
    let Some(exponent) = product_exponent(a, b).max(product_exponent(c, d)) else {
        return (0.0, 0);
    };
    let scaled = |x: f64, y: f64| {
        if x == 0.0 || y == 0.0 {
            (0.0, 0.0)
        } else {
            let x_exponent = libm::ilogb(x);
            (
                libm::scalbn(x, -x_exponent),
                libm::scalbn(y, x_exponent - exponent),
            )
        }
    };
    let (a, b) = scaled(a, b);
    let (c, d) = scaled(c, d);
    (difference_of_products(a, b, c, d), exponent)
}

/// `r x v` of two finite vectors as `m` and `e` with `r x v = m 2^e` and the
/// largest component of `m` in `[1, 2)`; `None` where `r x v` is the zero
/// vector, a zero vector or two parallel ones.
///
/// Each component is a [`scaled_difference_of_products`] of the components of
/// `r` and `v` as given, so it is within about a unit in its last place
/// however near parallel the two are and however far their magnitudes, or the
/// magnitudes of their components, spread over the exponent range.
fn scaled_cross(r: [f64; 3], v: [f64; 3]) -> Option<([f64; 3], i32)> {
    let components = [
        scaled_difference_of_products(r[1], v[2], r[2], v[1]),
        scaled_difference_of_products(r[2], v[0], r[0], v[2]),
        scaled_difference_of_products(r[0], v[1], r[1], v[0]),
    ];
    let exponent = components
        .iter()
        .filter(|(m, _)| *m != 0.0)
        .map(|(m, e)| libm::ilogb(*m) + e)
        .max()?;
    Some((
        components.map(|(m, e)| libm::scalbn(m, e - exponent)),
        exponent,
    ))
}

/// The unit normal `(r x v) / |r x v|` of the plane of two vectors; `None`
/// for a zero or non-finite vector, or two parallel ones.
///
/// Where the two are more than 30 degrees from parallel this is the plain
/// cross product of the vectors scaled by powers of two ([`exponent_split`]),
/// whose rounding is a few units in the last place of the result. Nearer to
/// parallel the plain products cancel and the rounding turns the direction,
/// so the normal is then the direction of [`scaled_cross`] of the vectors as
/// given: each component is rounded once, which keeps the normal at right
/// angles to `r` to the last place, and no component underflows or is lost to
/// the scaling however small the angle is.
fn orbit_normal_direction(r: [f64; 3], v: [f64; 3]) -> Option<[f64; 3]> {
    let r_scaled = exponent_normalized(r)?;
    let v_scaled = exponent_normalized(v)?;
    let plain = cross3(r_scaled, v_scaled);
    let normal = if norm3(plain) >= 0.5 * norm3(r_scaled) * norm3(v_scaled) {
        unit3(plain)
    } else {
        unit3(scaled_cross(r, v)?.0)
    }?;
    finite3(normal).then_some(normal)
}

/// The unit orbit normal of `(r, v)` with the sine of the angle between the
/// two; `None` for a zero or non-finite vector, or two parallel ones.
///
/// Both come from [`scaled_cross`], and the sine is
/// `|r x v| / (|r| |v|)` formed from the scaled vectors with the powers of two
/// applied once at the end, so it holds its last places for vectors of any
/// magnitude however near parallel they are. It underflows to zero only
/// below `2^-1074`.
fn orbit_normal(r: [f64; 3], v: [f64; 3]) -> Option<([f64; 3], f64)> {
    let (r_scaled, r_exponent) = exponent_split(r)?;
    let (v_scaled, v_exponent) = exponent_split(v)?;
    let (momentum, momentum_exponent) = scaled_cross(r, v)?;
    let normal = unit3(momentum)?;
    let sine = libm::scalbn(
        norm3(momentum) / (norm3(r_scaled) * norm3(v_scaled)),
        momentum_exponent - r_exponent - v_exponent,
    );
    Some((normal, sine))
}

/// The unit vectors of the orbit frame of `(r, v)`: the radial direction
/// wherever `r` is a non-zero finite vector, and the along-track and
/// cross-track directions where `v` is one too and is not parallel to `r`.
///
/// The radial direction is formed from `r` scaled by a power of two and the
/// cross-track one by [`orbit_normal_direction`], which leaves every rounding
/// of a navigation state as it is and keeps a state of any finite magnitude,
/// or a velocity a hair from the position, from losing its frame.
#[allow(clippy::type_complexity)]
fn orbit_frame(r: [f64; 3], v: [f64; 3]) -> (Option<[f64; 3]>, Option<([f64; 3], [f64; 3])>) {
    let radial_hat = exponent_normalized(r).and_then(unit3);
    let plane = orbit_normal_direction(r, v)
        .zip(radial_hat)
        .map(|(cross_hat, radial_hat)| (cross3(cross_hat, radial_hat), cross_hat));
    (radial_hat, plane)
}

/// Project a difference vector onto the radial/along-track/cross-track triad of
/// the orbit defined by position `r` and velocity `v`. Radial is along `r`,
/// cross-track along the angular momentum `r x v`, along-track completes the
/// right-handed set; an axis `(r, v)` does not define ([`orbit_frame`]) is the
/// zero vector.
///
/// `d` is scaled by a power of two before it is projected and the components
/// are scaled back, so a difference whose length is finite has finite
/// components; for any other the scaling is exact and changes nothing.
pub(crate) fn project_rac(d: [f64; 3], r: [f64; 3], v: [f64; 3]) -> (f64, f64, f64) {
    let (radial_hat, plane) = orbit_frame(r, v);
    let radial_hat = radial_hat.unwrap_or([0.0, 0.0, 0.0]);
    let (along_hat, cross_hat) = plane.unwrap_or(([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]));
    let exponent = binary_exponent(largest_magnitude(&d)).unwrap_or(0);
    let d_scaled = d.map(|c| libm::scalbn(c, -exponent));
    let component = |axis: [f64; 3]| libm::scalbn(dot3(d_scaled, axis), exponent);
    (
        component(radial_hat),
        component(along_hat),
        component(cross_hat),
    )
}

/// One source-minus-reference difference: the position difference resolved on
/// the reference triad, its magnitude, and the clock difference in meters where
/// both clocks exist.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OrbitDifference {
    pub(crate) orbit_3d: f64,
    pub(crate) radial: f64,
    pub(crate) along: f64,
    pub(crate) cross: f64,
    pub(crate) clock_m: Option<f64>,
}

/// Difference a source state against a reference state on the triad of the
/// reference position and `triad_velocity`.
pub(crate) fn orbit_difference(
    source_position_m: [f64; 3],
    source_clock_s: Option<f64>,
    reference_position_m: [f64; 3],
    reference_clock_s: Option<f64>,
    triad_velocity_m_s: [f64; 3],
) -> OrbitDifference {
    let d = sub3(source_position_m, reference_position_m);
    let (radial, along, cross) = project_rac(d, reference_position_m, triad_velocity_m_s);

    let clock_m = match (source_clock_s, reference_clock_s) {
        (Some(sc), Some(rc)) => Some((sc - rc) * C_M_S),
        _ => None,
    };

    OrbitDifference {
        orbit_3d: scaled_norm3(d),
        radial,
        along,
        cross,
        clock_m,
    }
}

/// Reference position and clock of `sat` at `t_j2000_s`, seconds since J2000 in
/// the reference product's time scale.
///
/// With `with_relativity` the clock carries the relativistic term RTKLIB
/// `peph2pos` applies to a precise clock, for comparison with a source clock that
/// carries its own; without the term (within 1 ms of the end of coverage) the
/// epoch then has no reference clock.
pub(crate) fn reference_state_at(
    reference: &Sp3,
    sat: GnssSatelliteId,
    t_j2000_s: f64,
    with_relativity: bool,
) -> Result<([f64; 3], Option<f64>), Error> {
    let state = reference.position_at_j2000_seconds(sat, t_j2000_s)?;
    let clock_s = if with_relativity {
        state.clock_s.and_then(|clock_s| {
            match ObservableEphemerisSource::clock_relativity_s(reference, sat, t_j2000_s) {
                crate::spp::ClockRelativity::Term(term_s) => Some(clock_s + term_s),
                crate::spp::ClockRelativity::NotApplicable => Some(clock_s),
                crate::spp::ClockRelativity::Unavailable => None,
            }
        })
    } else {
        state.clock_s
    };
    Ok((state.position.as_array(), clock_s))
}

/// Reference position only, or `None` where the product serves none.
pub(crate) fn reference_position_at(
    reference: &Sp3,
    sat: GnssSatelliteId,
    t_j2000_s: f64,
) -> Option<[f64; 3]> {
    reference
        .position_at_j2000_seconds(sat, t_j2000_s)
        .ok()
        .map(|state| state.position.as_array())
}

/// Root-mean-square over a non-empty slice, folding left from `0.0`.
///
/// The values are scaled by the power of two of the largest magnitude before
/// they are squared and the result is scaled back, so a series whose squares
/// overflow or underflow keeps its RMS; elsewhere the scaling is exact and
/// the result is that of the plain fold bit for bit.
pub(crate) fn rms(values: &[f64]) -> f64 {
    let exponent = binary_exponent(largest_magnitude(values)).unwrap_or(0);
    let sum_sq = values.iter().fold(0.0, |acc, &x| {
        let scaled = libm::scalbn(x, -exponent);
        acc + scaled * scaled
    });
    libm::scalbn((sum_sq / values.len() as f64).sqrt(), exponent)
}

/// Mean over a non-empty slice, folding left from `0.0`, with the same exact
/// power-of-two scaling as [`rms`] so a sum that would overflow does not.
fn mean(values: &[f64]) -> f64 {
    let exponent = binary_exponent(largest_magnitude(values)).unwrap_or(0);
    let sum = values
        .iter()
        .fold(0.0, |acc, &x| acc + libm::scalbn(x, -exponent));
    libm::scalbn(sum / values.len() as f64, exponent)
}

/// Maximum absolute value over a non-empty slice.
pub(crate) fn max_abs(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|v| v.abs())
        .reduce(f64::max)
        .expect("max_abs requires a non-empty slice")
}

/// Median of a slice (`None` if empty); even counts average the two middles.
pub(crate) fn median(values: &[f64]) -> Option<f64> {
    let mut sorted = values.to_vec();
    crate::astro::math::robust::median_sorting_in_place(&mut sorted)
}

/// Per-epoch median of `(epoch_index, value)` entries: `None` for an epoch with
/// fewer than `min_count` entries.
pub(crate) fn median_by_epoch(
    entries: impl IntoIterator<Item = (usize, f64)>,
    n_epochs: usize,
    min_count: usize,
) -> Vec<Option<f64>> {
    let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); n_epochs];
    for (epoch_index, value) in entries {
        buckets[epoch_index].push(value);
    }
    buckets
        .iter()
        .map(|values| {
            if values.len() < min_count {
                None
            } else {
                median(values)
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Errors.
// ---------------------------------------------------------------------------

/// Which of the two compared products a finding concerns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoredProduct {
    /// The orbit source being scored.
    Source,
    /// The reference SP3 product.
    Reference,
}

impl core::fmt::Display for ScoredProduct {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            ScoredProduct::Source => "orbit source",
            ScoredProduct::Reference => "reference product",
        })
    }
}

/// Why an orbit could not be scored.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum OrbitScoreError {
    /// An argument is non-finite or outside its domain.
    #[error("invalid orbit scoring {field}: {reason}")]
    InvalidInput {
        /// The rejected argument.
        field: &'static str,
        /// Why it was rejected.
        reason: &'static str,
    },
    /// The orbit source and the reference product are in different time systems.
    #[error(
        "orbit source time system {} does not match reference time system {}",
        .orbit_source.abbrev(),
        .reference.abbrev()
    )]
    TimeSystemMismatch {
        /// Time scale of the orbit source.
        orbit_source: TimeScale,
        /// Time scale of the reference product.
        reference: TimeScale,
    },
    /// The orbit source and the reference product are read on the same time
    /// axis but state different SP3 time systems, such as an `IRN` product
    /// against a `GPS` one.
    #[error(
        "orbit source time system {} does not match reference time system {}",
        .orbit_source.label(),
        .reference.label()
    )]
    TimeSystemLabelMismatch {
        /// SP3 time system of the orbit source.
        orbit_source: Sp3TimeSystem,
        /// SP3 time system of the reference product.
        reference: Sp3TimeSystem,
    },
    /// An instant is tagged with a time scale other than the one it is read in.
    #[error(
        "{field} is in {}, expected {}",
        .found.abbrev(),
        .expected.abbrev()
    )]
    InstantTimeSystemMismatch {
        /// The instant: `"epochs"`, `"cutoff"`, `"prediction_origin"` or
        /// `"samples"`.
        field: &'static str,
        /// Index of the instant in its list, zero for a single instant.
        index: usize,
        /// Time scale the instant is tagged with.
        found: TimeScale,
        /// Time scale it has to be in.
        expected: TimeScale,
    },
    /// The orbit source and the reference product state different coordinate
    /// frames, and the frame options name no way to reconcile them.
    #[error("orbit source frame {orbit_source:?} does not match reference frame {reference:?}")]
    FrameMismatch {
        /// Frame label of the orbit source.
        orbit_source: String,
        /// Frame label of the reference product.
        reference: String,
    },
    /// A catalog transformation between the two frames failed.
    #[error("frame transformation failed: {0}")]
    FrameTransform(FrameCatalogError),
    /// A requested satellite has no data at all in one of the products.
    #[error("satellite {satellite} is not in the {product}")]
    MissingSatellite {
        /// The product that lacks the satellite.
        product: ScoredProduct,
        /// The satellite.
        satellite: GnssSatelliteId,
    },
    /// One of the products has no state for a satellite at an epoch.
    #[error("the {product} has no state for {satellite} at epoch {epoch_index}: {error}")]
    MissingEpoch {
        /// The product that lacks the state.
        product: ScoredProduct,
        /// The satellite.
        satellite: GnssSatelliteId,
        /// Index of the epoch in the evaluation epochs.
        epoch_index: usize,
        /// What the product reported.
        error: Error,
    },
    /// The orbit source states no prediction origin for a satellite.
    #[error("the orbit source has no prediction origin for satellite {satellite}")]
    NoPredictionOrigin {
        /// The satellite.
        satellite: GnssSatelliteId,
    },
    /// The published weight table has no row for a satellite's constellation and
    /// orbit regime.
    #[error("no published SISRE weights for satellite {satellite} in a {regime} orbit")]
    NoSisreWeights {
        /// The satellite.
        satellite: GnssSatelliteId,
        /// The orbit regime found for it.
        regime: OrbitRegime,
    },
    /// The reference state of a satellite is not a bound orbit, so it has no
    /// orbit regime.
    #[error("the reference state of satellite {satellite} gives no orbit regime")]
    UndeterminedOrbitRegime {
        /// The satellite.
        satellite: GnssSatelliteId,
    },
    /// An orbit source returned a non-finite position or clock.
    #[error("the orbit source state of {satellite} at epoch {epoch_index} is not finite")]
    NonFiniteSourceState {
        /// The satellite.
        satellite: GnssSatelliteId,
        /// Index of the epoch in the evaluation epochs.
        epoch_index: usize,
    },
    /// The reference position and velocity of a satellite define no orbit frame
    /// (a zero position, or a velocity parallel to it).
    #[error("the reference state of {satellite} at epoch {epoch_index} defines no orbit frame")]
    DegenerateReferenceState {
        /// The satellite.
        satellite: GnssSatelliteId,
        /// Index of the epoch in the evaluation epochs.
        epoch_index: usize,
    },
    /// A difference or a statistic overflowed.
    #[error("an orbit difference of {satellite} is not finite")]
    NonFiniteResult {
        /// The satellite whose differences overflowed.
        satellite: GnssSatelliteId,
    },
}

fn invalid_input(field: &'static str, reason: &'static str) -> OrbitScoreError {
    OrbitScoreError::InvalidInput { field, reason }
}

fn finite3(v: [f64; 3]) -> bool {
    v.iter().all(|c| c.is_finite())
}

// ---------------------------------------------------------------------------
// Orbit frame.
// ---------------------------------------------------------------------------

/// The velocity the radial/transverse/normal frame is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TriadVelocity {
    /// The inertial velocity: the Earth-fixed velocity of the reference product
    /// plus the transport term `omega x r` of the nominal Earth rotation. The
    /// frame is the orbital frame, Orekit's `LOFType.QSW` of the inertial state.
    #[default]
    Inertial,
    /// The Earth-fixed velocity as the reference product gives it. `T` and `N`
    /// are then aligned with the ground track, not the orbit plane. This is the
    /// frame of [`crate::broadcast_comparison`].
    EarthFixed,
}

/// A position difference resolved on the orbit frame, meters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtnComponents {
    /// Component along the position vector.
    pub radial_m: f64,
    /// Component along `N x R`, in the orbit plane (along-track).
    pub transverse_m: f64,
    /// Component along the angular momentum `r x v` (cross-track).
    pub normal_m: f64,
}

/// Resolve `difference_m` on the radial/transverse/normal frame of the orbit
/// state `(position_m, velocity_m_s)`.
///
/// The three vectors are in one frame. For the orbital frame the velocity is the
/// inertial one; [`inertial_velocity_from_earth_fixed`] forms it from an
/// Earth-fixed state. The result equals Orekit's
/// `LOFType.QSW.rotationFromInertial(pv).applyTo(difference)` component for
/// component.
///
/// The frame depends only on the directions of the position and velocity, and
/// is formed from them scaled by powers of two, so a state of any finite
/// magnitude has its frame.
///
/// Errors with [`OrbitScoreError::InvalidInput`] for a non-finite input, a zero
/// position, a zero velocity or one parallel to the position, or a result that
/// overflows.
pub fn rtn_components(
    difference_m: [f64; 3],
    position_m: [f64; 3],
    velocity_m_s: [f64; 3],
) -> Result<RtnComponents, OrbitScoreError> {
    if !finite3(difference_m) {
        return Err(invalid_input("difference_m", "not finite"));
    }
    if !finite3(position_m) {
        return Err(invalid_input("position_m", "not finite"));
    }
    if !finite3(velocity_m_s) {
        return Err(invalid_input("velocity_m_s", "not finite"));
    }
    if !triad_is_defined(position_m, velocity_m_s) {
        return Err(invalid_input(
            "position_m",
            "zero, or parallel to the velocity",
        ));
    }
    let (radial_m, transverse_m, normal_m) = project_rac(difference_m, position_m, velocity_m_s);
    if !(radial_m.is_finite() && transverse_m.is_finite() && normal_m.is_finite()) {
        return Err(invalid_input("difference_m", "components overflow"));
    }
    Ok(RtnComponents {
        radial_m,
        transverse_m,
        normal_m,
    })
}

/// Whether `(r, v)` has a radial direction and an orbit normal, as
/// [`project_rac`] forms them.
fn triad_is_defined(r: [f64; 3], v: [f64; 3]) -> bool {
    orbit_frame(r, v).1.is_some()
}

/// The inertial velocity of an Earth-fixed state, in the Earth-fixed axes at
/// that instant: `v + omega x r` with `omega` the nominal Earth rotation rate
/// about the Earth-fixed z-axis ([`crate::constants::OMEGA_E_DOT_RAD_S`]).
///
/// Polar motion (the rotation axis is within about 3e-6 rad of the Earth-fixed
/// z-axis) and the excess length of day (parts in 1e8 of the rate) are not
/// modelled.
///
/// Errors with [`OrbitScoreError::InvalidInput`] for a non-finite input or a
/// result that overflows.
pub fn inertial_velocity_from_earth_fixed(
    position_ecef_m: [f64; 3],
    velocity_ecef_m_s: [f64; 3],
) -> Result<[f64; 3], OrbitScoreError> {
    if !finite3(position_ecef_m) {
        return Err(invalid_input("position_ecef_m", "not finite"));
    }
    if !finite3(velocity_ecef_m_s) {
        return Err(invalid_input("velocity_ecef_m_s", "not finite"));
    }
    let velocity = add_earth_rotation(position_ecef_m, velocity_ecef_m_s);
    if !finite3(velocity) {
        return Err(invalid_input(
            "velocity_ecef_m_s",
            "inertial velocity overflows",
        ));
    }
    Ok(velocity)
}

/// `v + omega x r` about the Earth-fixed z-axis, unchecked.
fn add_earth_rotation(position_ecef_m: [f64; 3], velocity_ecef_m_s: [f64; 3]) -> [f64; 3] {
    [
        velocity_ecef_m_s[0] - OMEGA_E_DOT_RAD_S * position_ecef_m[1],
        velocity_ecef_m_s[1] + OMEGA_E_DOT_RAD_S * position_ecef_m[0],
        velocity_ecef_m_s[2],
    ]
}

// ---------------------------------------------------------------------------
// SISRE weights.
// ---------------------------------------------------------------------------

/// Orbit regime of a navigation satellite, as far as the SISRE weights
/// distinguish it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrbitRegime {
    /// Medium Earth orbit.
    Meo,
    /// Inclined geosynchronous orbit.
    Igso,
    /// Geostationary orbit.
    Geo,
}

impl core::fmt::Display for OrbitRegime {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            OrbitRegime::Meo => "MEO",
            OrbitRegime::Igso => "IGSO",
            OrbitRegime::Geo => "GEO",
        })
    }
}

/// Semi-major axis above which an orbit is geosynchronous for the weight table,
/// meters: between the highest navigation MEO (Galileo, 29 600 km) and the
/// geosynchronous radius (42 164 km).
const GEOSYNCHRONOUS_MIN_SEMI_MAJOR_AXIS_M: f64 = 35_000_000.0;

/// Cosine of the inclination below which a geosynchronous orbit is
/// geostationary for the weight table: 15 degrees, between the few degrees the
/// geostationary navigation satellites reach and the 27 degrees and more of the
/// inclined ones.
const GEOSTATIONARY_MIN_COS_INCLINATION: f64 = 0.965_925_826_289_068_3;

impl OrbitRegime {
    /// The regime of the orbit through an inertial state, from its semi-major
    /// axis (vis-viva) and its inclination to the Earth-fixed z-axis.
    ///
    /// `None` for a state that is not an orbit about the Earth: a zero or
    /// non-finite position, a zero velocity or one along the position (no
    /// angular momentum, so no orbit plane), a state that is not bound, or a
    /// bound one whose perigee is below the Earth's equatorial radius, which
    /// meets the Earth before it completes a revolution.
    ///
    /// A state of any finite magnitude is classified by these rules: the
    /// radius, speed, semi-major axis and perigee are formed from the vectors
    /// scaled by powers of two, so none of them overflows or underflows on the
    /// way, and a semi-major axis or perigee beyond the largest `f64` compares
    /// as the infinity it rounds to.
    pub fn from_inertial_state(
        position_m: [f64; 3],
        inertial_velocity_m_s: [f64; 3],
    ) -> Option<Self> {
        let mu_m3_s2 = MU_EARTH * 1.0e9;
        let (normal, sine) = orbit_normal(position_m, inertial_velocity_m_s)?;
        let (position, position_exponent) = exponent_split(position_m)?;
        let (velocity, velocity_exponent) = exponent_split(inertial_velocity_m_s)?;
        let radius = norm3(position);
        let speed = norm3(velocity);
        // `nu = r v^2 / mu`: vis-viva gives `a = r / (2 - nu)`, and the
        // semi-latus rectum is `p = a nu (2 - nu) sin^2(angle of r and v)`.
        let nu = libm::scalbn(
            radius / mu_m3_s2 * speed * speed,
            position_exponent + 2 * velocity_exponent,
        );
        if !(nu.is_finite() && nu < 2.0) {
            return None;
        }
        let semi_major_axis_m = libm::scalbn(radius / (2.0 - nu), position_exponent);
        // The perigee `a (1 - e)` is `p / (1 + e)`, so `r nu sin^2 / (1 + e)`:
        // no `1 - e` to cancel on a very eccentric orbit, and a factor of `r`
        // that is at most one, since the perigee is no farther than `r`.
        let eccentricity = (1.0 - nu * (2.0 - nu) * sine * sine).max(0.0).sqrt();
        let perigee_m = libm::scalbn(
            radius * (nu * sine * sine / (1.0 + eccentricity)),
            position_exponent,
        );
        if perigee_m.is_nan() || perigee_m < WGS84_A_M {
            return None;
        }
        if semi_major_axis_m < GEOSYNCHRONOUS_MIN_SEMI_MAJOR_AXIS_M {
            Some(OrbitRegime::Meo)
        } else if normal[2] >= GEOSTATIONARY_MIN_COS_INCLINATION {
            Some(OrbitRegime::Geo)
        } else {
            Some(OrbitRegime::Igso)
        }
    }
}

/// Weights of the signal-in-space range error: the global-average projections
/// of a radial and of an along-track or cross-track orbit error on the line of
/// sight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SisreWeights {
    radial: f64,
    transverse_normal_squared: f64,
}

impl SisreWeights {
    /// Weights from the radial factor `w_R` and the squared along-track and
    /// cross-track factor `w_AC^2`.
    ///
    /// Both are projections, so each has to be finite and within `0..=1`.
    pub fn new(radial: f64, transverse_normal_squared: f64) -> Result<Self, OrbitScoreError> {
        if !(radial.is_finite() && (0.0..=1.0).contains(&radial)) {
            return Err(invalid_input("sisre_weights.radial", "not within 0..=1"));
        }
        if !(transverse_normal_squared.is_finite()
            && (0.0..=1.0).contains(&transverse_normal_squared))
        {
            return Err(invalid_input(
                "sisre_weights.transverse_normal_squared",
                "not within 0..=1",
            ));
        }
        Ok(Self {
            radial,
            transverse_normal_squared,
        })
    }

    /// The radial weight `w_R`.
    pub fn radial(&self) -> f64 {
        self.radial
    }

    /// The squared along-track and cross-track weight `w_AC^2`.
    pub fn transverse_normal_squared(&self) -> f64 {
        self.transverse_normal_squared
    }

    /// The global-average weights tabulated by Montenbruck, Steigenberger and
    /// Hauschild (2015) and used in their 2018 assessment, for a constellation
    /// and orbit regime; `None` where there is no row.
    ///
    /// | Constellation | Regime | `w_R` | `w_AC^2` |
    /// |---|---|---|---|
    /// | GPS | MEO | 0.98 | 1/49 |
    /// | GLONASS | MEO | 0.98 | 1/45 |
    /// | Galileo | MEO | 0.98 | 1/61 |
    /// | BeiDou | MEO | 0.98 | 1/54 |
    /// | BeiDou, QZSS | IGSO, GEO | 0.99 | 1/126 |
    /// | NavIC, SBAS | IGSO, GEO | 0.99 | 1/126 |
    ///
    /// The weights are averages over the Earth visible from the satellite, so
    /// they depend on the orbit radius alone. With users spread uniformly by
    /// area over the cap that sees the satellite above a 0 degree elevation
    /// mask, and `eta` the nadir angle at which the satellite sees a user,
    /// `w_R^2` is the mean of `cos^2(eta)` and `w_AC^2` the mean of
    /// `sin^2(eta) / 2` (the along-track and cross-track axes share the
    /// off-nadir part equally over azimuth). For a mean Earth radius of
    /// 6371 km that gives, at each constellation's orbit radius:
    ///
    /// | Orbit radius | `w_R` | `1 / w_AC^2` | Row |
    /// |---|---|---|---|
    /// | 25 510 km | 0.9776 | 45.10 | GLONASS |
    /// | 26 560 km | 0.9794 | 49.02 | GPS |
    /// | 27 910 km | 0.9814 | 54.30 | BeiDou MEO |
    /// | 29 600 km | 0.9835 | 61.28 | Galileo |
    /// | 42 164 km | 0.9921 | 126.45 | geosynchronous |
    ///
    /// Rounded to the two decimals and the whole number the table is published
    /// to, these are its rows; `sisre_weights_follow_from_the_orbit_radius`
    /// checks the derivation. The published geosynchronous row covers BeiDou
    /// and QZSS; NavIC and SBAS satellites fly at the same radius and take the
    /// same row.
    pub fn montenbruck_2018(system: GnssSystem, regime: OrbitRegime) -> Option<Self> {
        let (radial, inverse_transverse_normal_squared) = match (system, regime) {
            (GnssSystem::Gps, OrbitRegime::Meo) => (0.98, 49.0),
            (GnssSystem::Glonass, OrbitRegime::Meo) => (0.98, 45.0),
            (GnssSystem::Galileo, OrbitRegime::Meo) => (0.98, 61.0),
            (GnssSystem::BeiDou, OrbitRegime::Meo) => (0.98, 54.0),
            (
                GnssSystem::BeiDou | GnssSystem::Qzss | GnssSystem::Navic | GnssSystem::Sbas,
                OrbitRegime::Igso | OrbitRegime::Geo,
            ) => (0.99, 126.0),
            _ => return None,
        };
        Some(Self {
            radial,
            transverse_normal_squared: 1.0 / inverse_transverse_normal_squared,
        })
    }

    /// Orbit-only signal-in-space range error, meters:
    /// `sqrt(w_R^2 R^2 + w_AC^2 (T^2 + N^2))`.
    ///
    /// Errors with [`OrbitScoreError::InvalidInput`] for a non-finite component
    /// or an error larger than the largest `f64`.
    pub fn orbit_only_m(&self, components: RtnComponents) -> Result<f64, OrbitScoreError> {
        check_components(components)?;
        finite_sisre(self.orbit_only_unchecked(components))
    }

    /// Signal-in-space range error with the clock, meters:
    /// `sqrt((w_R R - c dt)^2 + w_AC^2 (T^2 + N^2))`, with `clock_m` the clock
    /// difference `c dt` in meters, in the same sense as the orbit difference.
    ///
    /// Errors with [`OrbitScoreError::InvalidInput`] for a non-finite component
    /// or clock difference, or an error larger than the largest `f64`.
    pub fn with_clock_m(
        &self,
        components: RtnComponents,
        clock_m: f64,
    ) -> Result<f64, OrbitScoreError> {
        check_components(components)?;
        if !clock_m.is_finite() {
            return Err(invalid_input("clock_m", "not finite"));
        }
        finite_sisre(self.with_clock_unchecked(components, clock_m))
    }

    fn orbit_only_unchecked(&self, components: RtnComponents) -> f64 {
        self.range_error(self.radial * components.radial_m, components)
    }

    fn with_clock_unchecked(&self, components: RtnComponents, clock_m: f64) -> f64 {
        self.range_error(self.radial * components.radial_m - clock_m, components)
    }

    /// `sqrt(range^2 + w_AC^2 (T^2 + N^2))`, as the length of
    /// `(range, w_AC T, w_AC N)` with the three scaled by a power of two first,
    /// so no square overflows or underflows: the result is finite exactly when
    /// the error is, and a zero weight takes no part however large `T` and `N`
    /// are.
    fn range_error(&self, range_m: f64, components: RtnComponents) -> f64 {
        let weight = self.transverse_normal_squared.sqrt();
        scaled_norm3([
            range_m,
            weight * components.transverse_m,
            weight * components.normal_m,
        ])
    }
}

fn check_components(components: RtnComponents) -> Result<(), OrbitScoreError> {
    if finite3([
        components.radial_m,
        components.transverse_m,
        components.normal_m,
    ]) {
        Ok(())
    } else {
        Err(invalid_input("components", "not finite"))
    }
}

fn finite_sisre(value: f64) -> Result<f64, OrbitScoreError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid_input("components", "SISRE overflows"))
    }
}

/// Which SISRE weights a satellite is scored with.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SisreWeightSelection {
    /// [`SisreWeights::montenbruck_2018`] for the satellite's constellation and
    /// the orbit regime of its reference state
    /// ([`OrbitRegime::from_inertial_state`]). A satellite the table has no row
    /// for is refused with [`OrbitScoreError::NoSisreWeights`].
    #[default]
    Montenbruck2018,
    /// The same caller-supplied weights for every satellite.
    Fixed(SisreWeights),
}

// ---------------------------------------------------------------------------
// Orbit sources.
// ---------------------------------------------------------------------------

/// The coordinate frame an orbit source states its positions in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrbitFrame {
    /// A terrestrial frame named as an SP3 header names it, such as `IGS20`.
    Label(String),
    /// The broadcast datum of each constellation (WGS 84, PZ-90.11, GTRF, BDCS
    /// and their kin), each aligned with the ITRF at the centimeter level. It is
    /// compared with any reference frame as it stands; the difference between
    /// the datums is part of what the comparison measures.
    BroadcastDatums,
}

/// The point of the satellite an orbit source's positions refer to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrbitReferencePoint {
    /// The centre of mass, as in an SP3 product.
    CenterOfMass,
    /// The antenna phase centre, as in a broadcast ephemeris.
    AntennaPhaseCenter,
}

/// One state of an orbit source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitSourceState {
    /// Earth-fixed position, meters.
    pub position_ecef_m: [f64; 3],
    /// Satellite clock offset, seconds, where the source has one.
    pub clock_s: Option<f64>,
}

/// An orbit that can be scored against a reference product.
///
/// Time is seconds since J2000 in [`Self::time_scale`], the axis
/// [`Sp3::position_at_j2000_seconds`] is queried on.
pub trait OrbitSource {
    /// The time scale of the source's time axis.
    fn time_scale(&self) -> TimeScale;

    /// The SP3 time-system label of the source, where it has one that
    /// [`Self::time_scale`] does not name exactly. An SP3 product in IRNSS
    /// time (`IRN`) is read on the GPS time axis but is not in GPS time, and
    /// says so here. `None` (the default) is the label of
    /// [`Self::time_scale`].
    fn sp3_time_system(&self) -> Option<Sp3TimeSystem> {
        None
    }

    /// The coordinate frame of the positions.
    fn frame(&self) -> OrbitFrame;

    /// The point of the satellite the positions refer to.
    fn reference_point(&self) -> OrbitReferencePoint {
        OrbitReferencePoint::CenterOfMass
    }

    /// Whether the clock already carries the periodic relativistic term. A
    /// product clock (SP3, RINEX clock) does not; a broadcast clock does.
    fn clock_includes_relativity(&self) -> bool {
        false
    }

    /// The state of `satellite` at `t_j2000_s`.
    ///
    /// [`Error::UnknownSatellite`] says the source has nothing for the
    /// satellite; any other error says it has no state at that instant.
    fn state_at_j2000_s(
        &self,
        satellite: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<OrbitSourceState, Error>;

    /// The instant the prediction of `satellite` starts from, or `None` where
    /// the source states none. The prediction horizon of an epoch is its time
    /// past this instant.
    fn prediction_origin_j2000_s(&self, satellite: GnssSatelliteId) -> Option<f64>;
}

/// The SP3 time-system label that names `scale` exactly, where there is one.
fn sp3_time_system_of(scale: TimeScale) -> Option<Sp3TimeSystem> {
    match scale {
        TimeScale::Gpst => Some(Sp3TimeSystem::Gps),
        TimeScale::Glonasst => Some(Sp3TimeSystem::Glonass),
        TimeScale::Gst => Some(Sp3TimeSystem::Galileo),
        TimeScale::Tai => Some(Sp3TimeSystem::Tai),
        TimeScale::Utc => Some(Sp3TimeSystem::Utc),
        TimeScale::Qzsst => Some(Sp3TimeSystem::Qzss),
        TimeScale::Bdt => Some(Sp3TimeSystem::Beidou),
        _ => None,
    }
}

/// Seconds since J2000 of an instant on the time axis the reference product's
/// record epochs are on.
///
/// An instant an SP3 epoch record states exactly, as every parsed record epoch
/// is, is placed where the SP3 reader places that record
/// ([`Sp3::epochs_j2000_seconds`]): an instant held as a split Julian date can
/// otherwise reduce to a unit in the last place away from its whole second, and
/// a UTC `23:59:60` label to the second before it. Any other instant is reduced
/// as [`Sp3::position`] reduces a query.
fn axis_seconds(instant: &Instant) -> Option<f64> {
    sp3_time_system_of(instant.scale)
        .and_then(|system| record_epoch_j2000_seconds(instant, system))
        .or_else(|| instant_to_j2000_seconds(instant))
}

/// Seconds since J2000 of an instant that has to be in `expected`, on the
/// axis of [`axis_seconds`].
fn seconds_in_scale(
    instant: &Instant,
    expected: TimeScale,
    field: &'static str,
    index: usize,
) -> Result<f64, OrbitScoreError> {
    if instant.scale != expected {
        return Err(OrbitScoreError::InstantTimeSystemMismatch {
            field,
            index,
            found: instant.scale,
            expected,
        });
    }
    match axis_seconds(instant) {
        Some(seconds) if seconds.is_finite() => Ok(seconds),
        _ => Err(invalid_input(
            field,
            "not representable as seconds since J2000",
        )),
    }
}

/// The parsed epochs of an SP3 product as seconds since J2000, one per epoch.
fn sp3_epoch_seconds(product: &Sp3, field: &'static str) -> Result<Vec<f64>, OrbitScoreError> {
    let seconds = product.epochs_j2000_seconds();
    if seconds.len() != product.epoch_count() || seconds.iter().any(|t| !t.is_finite()) {
        return Err(invalid_input(
            field,
            "an epoch is not representable as seconds since J2000",
        ));
    }
    Ok(seconds)
}

/// Where the prediction of an SP3 product starts.
#[derive(Debug, Clone, PartialEq)]
enum Sp3Origin {
    /// Per satellite, from the orbit-prediction flags of its records.
    Flags(BTreeMap<GnssSatelliteId, f64>),
    /// One instant for every satellite.
    Cutoff(f64),
}

/// An SP3 product scored as a prediction.
///
/// The prediction origin is either read from the product's orbit-prediction
/// flags ([`Self::from_prediction_flags`]) or given by the caller
/// ([`Self::with_cutoff`]). States are the product's interpolated positions and
/// clocks.
#[derive(Debug, Clone)]
pub struct PredictedSp3<'a> {
    product: &'a Sp3,
    origin: Sp3Origin,
}

impl<'a> PredictedSp3<'a> {
    /// Take each satellite's prediction origin from the SP3 orbit-prediction
    /// flag: the last record of the satellite not flagged predicted that
    /// precedes its first predicted record.
    ///
    /// A satellite with no predicted record has its last record as origin, so
    /// none of its epochs is at a positive horizon. A satellite whose first
    /// record is already predicted has no origin; its epochs are counted under
    /// [`ScoreCoverage::no_prediction_origin`].
    pub fn from_prediction_flags(product: &'a Sp3) -> Result<Self, OrbitScoreError> {
        let seconds = sp3_epoch_seconds(product, "source.epochs")?;
        let mut first_predicted: BTreeMap<GnssSatelliteId, f64> = BTreeMap::new();
        for (index, &t) in seconds.iter().enumerate() {
            let states = product
                .states_at(index)
                .map_err(|_| invalid_input("source.epochs", "epoch index out of range"))?;
            for (&sat, state) in states {
                if state.flags.orbit_predicted {
                    let first = first_predicted.entry(sat).or_insert(t);
                    if t < *first {
                        *first = t;
                    }
                }
            }
        }
        let mut origins: BTreeMap<GnssSatelliteId, f64> = BTreeMap::new();
        for (index, &t) in seconds.iter().enumerate() {
            let states = product
                .states_at(index)
                .map_err(|_| invalid_input("source.epochs", "epoch index out of range"))?;
            for (&sat, state) in states {
                if state.flags.orbit_predicted {
                    continue;
                }
                if first_predicted.get(&sat).is_some_and(|&first| t >= first) {
                    continue;
                }
                let origin = origins.entry(sat).or_insert(t);
                if t > *origin {
                    *origin = t;
                }
            }
        }
        Ok(Self {
            product,
            origin: Sp3Origin::Flags(origins),
        })
    }

    /// Take `cutoff` as the prediction origin of every satellite, whatever the
    /// product's flags say. `cutoff` is in the product's time scale.
    pub fn with_cutoff(product: &'a Sp3, cutoff: Instant) -> Result<Self, OrbitScoreError> {
        let cutoff_s = seconds_in_scale(&cutoff, product.header.time_scale, "cutoff", 0)?;
        Ok(Self {
            product,
            origin: Sp3Origin::Cutoff(cutoff_s),
        })
    }

    /// The SP3 product.
    pub fn product(&self) -> &'a Sp3 {
        self.product
    }
}

impl OrbitSource for PredictedSp3<'_> {
    fn time_scale(&self) -> TimeScale {
        self.product.header.time_scale
    }

    fn sp3_time_system(&self) -> Option<Sp3TimeSystem> {
        Some(self.product.header.time_system)
    }

    fn frame(&self) -> OrbitFrame {
        OrbitFrame::Label(self.product.header.coordinate_system.clone())
    }

    fn state_at_j2000_s(
        &self,
        satellite: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<OrbitSourceState, Error> {
        let state = self
            .product
            .position_at_j2000_seconds(satellite, t_j2000_s)?;
        Ok(OrbitSourceState {
            position_ecef_m: state.position.as_array(),
            clock_s: state.clock_s,
        })
    }

    fn prediction_origin_j2000_s(&self, satellite: GnssSatelliteId) -> Option<f64> {
        match &self.origin {
            Sp3Origin::Flags(origins) => origins.get(&satellite).copied(),
            Sp3Origin::Cutoff(cutoff_s) => Some(*cutoff_s),
        }
    }
}

/// A broadcast ephemeris scored as a prediction from a caller-given origin.
///
/// The time axis is GPS time, the axis the broadcast store is queried on. The
/// positions are antenna phase centres in the broadcast datums, and the clock
/// carries the broadcast relativistic term and no group delay.
#[derive(Clone)]
pub struct BroadcastOrbit<'a> {
    ephemeris: &'a BroadcastEphemeris,
    origin_j2000_s: f64,
}

impl core::fmt::Debug for BroadcastOrbit<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BroadcastOrbit")
            .field("origin_j2000_s", &self.origin_j2000_s)
            .finish_non_exhaustive()
    }
}

impl<'a> BroadcastOrbit<'a> {
    /// Score `ephemeris` with horizons counted from `prediction_origin`, a GPS
    /// time instant such as the time the navigation data were collected.
    pub fn new(
        ephemeris: &'a BroadcastEphemeris,
        prediction_origin: Instant,
    ) -> Result<Self, OrbitScoreError> {
        let origin_j2000_s =
            seconds_in_scale(&prediction_origin, TimeScale::Gpst, "prediction_origin", 0)?;
        Ok(Self {
            ephemeris,
            origin_j2000_s,
        })
    }

    fn has_satellite(&self, satellite: GnssSatelliteId) -> bool {
        self.ephemeris
            .records()
            .iter()
            .any(|record| record.satellite_id == satellite)
            || self
                .ephemeris
                .glonass_records()
                .iter()
                .any(|record| record.satellite_id == satellite)
            || self
                .ephemeris
                .sbas_records()
                .iter()
                .any(|record| record.satellite_id == satellite)
    }
}

impl OrbitSource for BroadcastOrbit<'_> {
    fn time_scale(&self) -> TimeScale {
        TimeScale::Gpst
    }

    fn frame(&self) -> OrbitFrame {
        OrbitFrame::BroadcastDatums
    }

    fn reference_point(&self) -> OrbitReferencePoint {
        OrbitReferencePoint::AntennaPhaseCenter
    }

    fn clock_includes_relativity(&self) -> bool {
        true
    }

    fn state_at_j2000_s(
        &self,
        satellite: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<OrbitSourceState, Error> {
        match self
            .ephemeris
            .observable_state_at_j2000_s(satellite, t_j2000_s)
        {
            Ok(state) => Ok(OrbitSourceState {
                position_ecef_m: state.position_ecef_m,
                clock_s: state.clock_s,
            }),
            Err(_) if !self.has_satellite(satellite) => Err(Error::UnknownSatellite(satellite)),
            Err(_) => Err(Error::EpochOutOfRange),
        }
    }

    fn prediction_origin_j2000_s(&self, _satellite: GnssSatelliteId) -> Option<f64> {
        Some(self.origin_j2000_s)
    }
}

/// One state of a caller's trajectory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectorySample {
    /// The satellite.
    pub satellite: GnssSatelliteId,
    /// The instant of the state.
    pub epoch: Instant,
    /// Earth-fixed position of the centre of mass, meters.
    pub position_ecef_m: [f64; 3],
    /// Satellite clock offset, seconds, where the trajectory has one.
    pub clock_s: Option<f64>,
}

/// A caller's trajectory: states of one or more satellites at the instants the
/// caller propagated them to, scored at exactly those instants.
///
/// The trajectory is not interpolated. It has a state at an evaluation epoch
/// only where a sample was given for that instant, so score it at its own
/// instants ([`Self::epochs`] with [`ScoreEpochs::Instants`]) or sample it at
/// the reference product's record epochs (`Sp3::epochs`). A sample instant
/// that an SP3 record states is placed where the SP3 reader places the record,
/// so a sample at a reference record epoch is found at that record under
/// [`ScoreEpochs::ReferenceRecords`].
#[derive(Debug, Clone, PartialEq)]
pub struct OrbitTrajectory {
    time_scale: TimeScale,
    frame_label: String,
    origin_j2000_s: f64,
    clock_includes_relativity: bool,
    nodes: BTreeMap<GnssSatelliteId, Vec<(f64, OrbitSourceState)>>,
    epochs: Vec<(f64, Instant)>,
}

impl OrbitTrajectory {
    /// Build a trajectory from samples.
    ///
    /// `frame_label` names the terrestrial frame of the positions as an SP3
    /// header would (`IGS20`, ...). `prediction_origin` is the instant the
    /// propagation starts from, the last instant the orbit rests on
    /// observations. Every sample epoch is in the time scale of
    /// `prediction_origin`.
    ///
    /// Errors with [`OrbitScoreError::InvalidInput`] for a blank frame label, no
    /// samples, a non-finite position or clock, an epoch that is not
    /// representable, or two samples of one satellite at one instant, and with
    /// [`OrbitScoreError::InstantTimeSystemMismatch`] for a sample in another
    /// time scale.
    pub fn new(
        frame_label: impl Into<String>,
        prediction_origin: Instant,
        samples: impl IntoIterator<Item = TrajectorySample>,
    ) -> Result<Self, OrbitScoreError> {
        let frame_label = frame_label.into();
        if frame_label.trim().is_empty() {
            return Err(invalid_input("frame_label", "blank"));
        }
        let time_scale = prediction_origin.scale;
        let origin_j2000_s =
            seconds_in_scale(&prediction_origin, time_scale, "prediction_origin", 0)?;

        let mut nodes: BTreeMap<GnssSatelliteId, Vec<(f64, OrbitSourceState)>> = BTreeMap::new();
        let mut epochs: Vec<(f64, Instant)> = Vec::new();
        for (index, sample) in samples.into_iter().enumerate() {
            let t = seconds_in_scale(&sample.epoch, time_scale, "samples", index)?;
            if !finite3(sample.position_ecef_m) {
                return Err(invalid_input("samples.position_ecef_m", "not finite"));
            }
            if sample.clock_s.is_some_and(|clock_s| !clock_s.is_finite()) {
                return Err(invalid_input("samples.clock_s", "not finite"));
            }
            nodes.entry(sample.satellite).or_default().push((
                t,
                OrbitSourceState {
                    position_ecef_m: sample.position_ecef_m,
                    clock_s: sample.clock_s,
                },
            ));
            epochs.push((t, sample.epoch));
        }
        if nodes.is_empty() {
            return Err(invalid_input("samples", "empty"));
        }
        for series in nodes.values_mut() {
            series.sort_by(|a, b| a.0.total_cmp(&b.0));
            if series.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return Err(invalid_input(
                    "samples",
                    "two samples of one satellite at one instant",
                ));
            }
        }
        epochs.sort_by(|a, b| a.0.total_cmp(&b.0));
        epochs.dedup_by(|later, earlier| later.0 == earlier.0);

        Ok(Self {
            time_scale,
            frame_label,
            origin_j2000_s,
            clock_includes_relativity: false,
            nodes,
            epochs,
        })
    }

    /// State whether the sample clocks already carry the periodic relativistic
    /// term (`false` unless set, as for a product clock).
    #[must_use]
    pub fn with_clock_relativity_included(mut self, included: bool) -> Self {
        self.clock_includes_relativity = included;
        self
    }

    /// The distinct sample instants, ascending.
    pub fn epochs(&self) -> Vec<Instant> {
        self.epochs.iter().map(|(_, instant)| *instant).collect()
    }

    /// The satellites the trajectory has samples for, ascending.
    pub fn satellites(&self) -> impl Iterator<Item = GnssSatelliteId> + '_ {
        self.nodes.keys().copied()
    }
}

impl OrbitSource for OrbitTrajectory {
    fn time_scale(&self) -> TimeScale {
        self.time_scale
    }

    fn frame(&self) -> OrbitFrame {
        OrbitFrame::Label(self.frame_label.clone())
    }

    fn clock_includes_relativity(&self) -> bool {
        self.clock_includes_relativity
    }

    fn state_at_j2000_s(
        &self,
        satellite: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<OrbitSourceState, Error> {
        let series = self
            .nodes
            .get(&satellite)
            .ok_or(Error::UnknownSatellite(satellite))?;
        series
            .binary_search_by(|node| node.0.total_cmp(&t_j2000_s))
            .map(|index| series[index].1)
            .map_err(|_| Error::EpochOutOfRange)
    }

    fn prediction_origin_j2000_s(&self, _satellite: GnssSatelliteId) -> Option<f64> {
        Some(self.origin_j2000_s)
    }
}

// ---------------------------------------------------------------------------
// Options.
// ---------------------------------------------------------------------------

/// The instants the source is scored at.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ScoreEpochs {
    /// Every record epoch of the reference product.
    #[default]
    ReferenceRecords,
    /// Caller-chosen instants in the reference product's time scale, strictly
    /// increasing.
    Instants(Vec<Instant>),
}

/// How the clock differences of an epoch are referred to a common datum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClockDatum {
    /// Remove, per constellation and epoch, the median clock difference of the
    /// requested satellites that have one. An epoch with fewer than two such
    /// satellites of a constellation yields no clock error for it.
    #[default]
    EpochSystemMedian,
    /// Use the clock differences as they stand, for products known to share a
    /// clock datum.
    Unadjusted,
}

/// What happens to a satellite and epoch that cannot be scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GapPolicy {
    /// Count it in [`ScoreCoverage`] under its reason and score the rest.
    #[default]
    Count,
    /// Return the typed error for the first one: [`OrbitScoreError::MissingSatellite`],
    /// [`OrbitScoreError::MissingEpoch`] or [`OrbitScoreError::NoPredictionOrigin`].
    /// An epoch outside the horizon bins is not a gap and is counted either
    /// way; a satellite one of the products has nothing for, or the source
    /// states no prediction origin for, is a gap whatever its horizons.
    Refuse,
}

/// Default half step of the reference velocity difference, seconds.
///
/// A centered difference over one second of a navigation orbit is within
/// 1e-5 m/s of the velocity (its error is `h^2 / 6` times the third derivative,
/// about `1e-4 m/s^3` at these altitudes), a rotation of the orbit frame below
/// 1e-8 rad.
pub const DEFAULT_VELOCITY_HALF_STEP_S: f64 = 0.5;

/// Smallest half step of the reference velocity difference, seconds: the
/// 1e-8 s to which SP3 states its epochs.
///
/// Far shorter steps no longer resolve the motion of a navigation orbit in the
/// interpolated reference positions, which round at about 4e-9 m: at 1e-11 s
/// the rounding turns the orbit frame by degrees. A step is also refused where
/// `t +- h` rounds to `t` at an evaluation epoch (1.2e-7 s apart in 2026),
/// which is the larger bound for epochs more than 2^27 s (4.25 years) from
/// J2000.
pub const MIN_VELOCITY_HALF_STEP_S: f64 = 1.0e-8;

/// Options of [`score_orbit`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct OrbitScoreOptions {
    /// The instants scored.
    pub epochs: ScoreEpochs,
    /// Edges of the prediction-horizon bins, seconds, strictly increasing, at
    /// least two. Bin `i` holds the horizons `h` with
    /// `edges[i] < h <= edges[i + 1]`, so the prediction origin itself is not in
    /// a bin that starts at zero. A negative edge takes in the fitted part of a
    /// source.
    pub horizon_bin_edges_s: Vec<f64>,
    /// The velocity the orbit frame is built from.
    pub triad: TriadVelocity,
    /// Half step of the reference velocity difference, seconds; at least
    /// [`MIN_VELOCITY_HALF_STEP_S`], and long enough that `t +- h` does not
    /// round to `t` at an evaluation epoch.
    pub velocity_half_step_s: f64,
    /// The SISRE weights.
    pub sisre_weights: SisreWeightSelection,
    /// The clock datum.
    pub clock_datum: ClockDatum,
    /// What happens to a satellite and epoch that cannot be scored.
    pub gaps: GapPolicy,
    /// How a source whose frame label differs from the reference's is
    /// reconciled: not at all by default, by a caller's assertion that the
    /// labels name one frame, or by the catalog Helmert transformation between
    /// known ITRF realizations. The same options as [`crate::ephemeris::merge`].
    pub frame_reconciliation: Sp3FrameReconciliationOptions,
}

impl OrbitScoreOptions {
    /// Options with the given horizon bin edges and every default: the
    /// reference record epochs, the inertial orbit frame, the default velocity
    /// half step, the published SISRE weights, the per-constellation clock
    /// datum, gaps counted, and no frame reconciliation.
    pub fn new(horizon_bin_edges_s: Vec<f64>) -> Self {
        Self {
            epochs: ScoreEpochs::ReferenceRecords,
            horizon_bin_edges_s,
            triad: TriadVelocity::Inertial,
            velocity_half_step_s: DEFAULT_VELOCITY_HALF_STEP_S,
            sisre_weights: SisreWeightSelection::Montenbruck2018,
            clock_datum: ClockDatum::EpochSystemMedian,
            gaps: GapPolicy::Count,
            frame_reconciliation: Sp3FrameReconciliationOptions::disabled(),
        }
    }
}

// ---------------------------------------------------------------------------
// Report.
// ---------------------------------------------------------------------------

/// RMS, mean and largest magnitude of one error series, meters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorStats {
    /// Root mean square.
    pub rms_m: f64,
    /// Mean, signed.
    pub mean_m: f64,
    /// Largest absolute value.
    pub max_abs_m: f64,
}

/// Statistics of the differences in one prediction-horizon bin.
///
/// The orbit fields are `None` when `count` is zero and the clock fields when
/// `clock_count` is zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HorizonBinStats {
    /// Lower horizon edge of the bin, seconds, exclusive.
    pub horizon_start_s: f64,
    /// Upper horizon edge of the bin, seconds, inclusive.
    pub horizon_end_s: f64,
    /// Number of scored epochs.
    pub count: usize,
    /// Radial difference.
    pub radial: Option<ErrorStats>,
    /// Transverse (along-track) difference.
    pub transverse: Option<ErrorStats>,
    /// Normal (cross-track) difference.
    pub normal: Option<ErrorStats>,
    /// Magnitude of the position difference.
    pub orbit_3d: Option<ErrorStats>,
    /// Orbit-only signal-in-space range error.
    pub sisre_orbit: Option<ErrorStats>,
    /// Number of scored epochs with a clock error.
    pub clock_count: usize,
    /// Clock difference under the selected [`ClockDatum`].
    pub clock: Option<ErrorStats>,
    /// Signal-in-space range error with the clock.
    pub sisre: Option<ErrorStats>,
}

/// What became of the satellite and epoch pairs asked for.
///
/// `epochs` equals `scored` plus the six reason counts. A pair is counted
/// under the first reason that applies, checked in this order: the reference
/// has nothing for the satellite, the source has nothing for it, the source
/// states no prediction origin for it, the horizon is outside the bins, the
/// reference has no state at the epoch, the source has none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ScoreCoverage {
    /// Satellite and epoch pairs considered.
    pub epochs: usize,
    /// Pairs scored.
    pub scored: usize,
    /// Scored pairs that also have a clock error.
    pub clocked: usize,
    /// Pairs whose prediction horizon is in no bin.
    pub outside_horizon_bins: usize,
    /// Pairs of a satellite the source states no prediction origin for.
    pub no_prediction_origin: usize,
    /// Pairs of a satellite the source has nothing for.
    pub source_satellite_missing: usize,
    /// Pairs where the source has the satellite but no state at the epoch.
    pub source_epoch_missing: usize,
    /// Pairs of a satellite the reference has nothing for.
    pub reference_satellite_missing: usize,
    /// Pairs where the reference has the satellite but no state, or no
    /// velocity, at the epoch.
    pub reference_epoch_missing: usize,
}

impl ScoreCoverage {
    fn add(&mut self, other: &ScoreCoverage) {
        self.epochs += other.epochs;
        self.scored += other.scored;
        self.clocked += other.clocked;
        self.outside_horizon_bins += other.outside_horizon_bins;
        self.no_prediction_origin += other.no_prediction_origin;
        self.source_satellite_missing += other.source_satellite_missing;
        self.source_epoch_missing += other.source_epoch_missing;
        self.reference_satellite_missing += other.reference_satellite_missing;
        self.reference_epoch_missing += other.reference_epoch_missing;
    }
}

/// The scores of one satellite.
#[derive(Debug, Clone, PartialEq)]
pub struct SatelliteScore {
    /// The satellite.
    pub satellite: GnssSatelliteId,
    /// The prediction origin of the satellite, seconds since J2000 in the
    /// reference time scale.
    pub prediction_origin_j2000_s: Option<f64>,
    /// The orbit regime found for the satellite, where the weights were looked
    /// up by regime and an epoch was scored.
    pub orbit_regime: Option<OrbitRegime>,
    /// The SISRE weights the satellite was scored with, where an epoch was
    /// scored.
    pub sisre_weights: Option<SisreWeights>,
    /// One entry per horizon bin.
    pub bins: Vec<HorizonBinStats>,
    /// What became of the satellite's epochs.
    pub coverage: ScoreCoverage,
}

/// The scores of one constellation, over its requested satellites.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemScore {
    /// The constellation.
    pub system: GnssSystem,
    /// One entry per horizon bin.
    pub bins: Vec<HorizonBinStats>,
    /// What became of the epochs of the constellation's satellites.
    pub coverage: ScoreCoverage,
}

/// How the frame of the source was matched to the frame of the reference.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameAgreement {
    /// Both state the same frame label.
    SameLabel,
    /// The caller asserted that the two labels name one frame; no coordinate
    /// was changed.
    AssertedEquivalent,
    /// Source positions were taken from `from` to `to` with the catalog Helmert
    /// transformation at each epoch (the identity when the two are equal).
    Helmert {
        /// Realization the source label names.
        from: TerrestrialFrame,
        /// Realization the reference label names.
        to: TerrestrialFrame,
    },
    /// The source is in the broadcast datums, compared as it stands.
    BroadcastDatums,
}

/// The result of [`score_orbit`].
#[derive(Debug, Clone, PartialEq)]
pub struct OrbitScoreReport {
    /// The horizon bin edges, seconds.
    pub horizon_bin_edges_s: Vec<f64>,
    /// How the two frames were matched.
    pub frame: FrameAgreement,
    /// The point of the satellite the source positions refer to. The reference
    /// is a centre-of-mass product and no antenna offset is applied.
    pub source_reference_point: OrbitReferencePoint,
    /// Scores per satellite, in the requested order.
    pub per_satellite: Vec<SatelliteScore>,
    /// Scores per constellation, in the order the constellations first appear in
    /// the requested satellites.
    pub per_system: Vec<SystemScore>,
    /// What became of every satellite and epoch pair asked for.
    pub coverage: ScoreCoverage,
}

// ---------------------------------------------------------------------------
// Scoring.
// ---------------------------------------------------------------------------

/// One scored satellite and epoch.
#[derive(Debug, Clone, Copy)]
struct Sample {
    epoch_index: usize,
    bin: usize,
    components: RtnComponents,
    orbit_3d_m: f64,
    clock_m: Option<f64>,
}

/// Why a satellite and epoch pair was not scored.
#[derive(Debug, Clone, PartialEq)]
enum Gap {
    OutsideHorizonBins,
    NoPredictionOrigin,
    Missing {
        product: ScoredProduct,
        error: Error,
    },
}

/// One evaluation epoch on the reference time axis.
#[derive(Debug, Clone, Copy)]
struct EvalEpoch {
    instant: Instant,
    t_j2000_s: f64,
}

/// The frame treatment of source positions resolved before scoring.
#[derive(Debug, Clone, Copy, PartialEq)]
enum FrameTreatment {
    AsStated,
    Helmert {
        from: TerrestrialFrame,
        to: TerrestrialFrame,
    },
}

/// Score an orbit source against a reference SP3 product.
///
/// `satellites` are scored at the instants of [`OrbitScoreOptions::epochs`]. For
/// each pair the reference state and velocity are read from `reference`, the
/// source state from `source`, and the position difference `source - reference`
/// is resolved on the reference orbit frame; the pair goes to the horizon bin of
/// its time past the source's prediction origin. See the [module
/// documentation](self) for the frame, the SISRE formula and the clock datum.
///
/// # Errors
///
/// - [`OrbitScoreError::InvalidInput`]: bin edges that are fewer than two,
///   non-finite or not strictly increasing; a velocity half step that is not
///   finite, is below [`MIN_VELOCITY_HALF_STEP_S`], or rounds away at an
///   evaluation epoch; an
///   asserted frame label set that names fewer than two distinct labels or a
///   blank one; a satellite listed twice; evaluation instants that are not
///   strictly increasing.
/// - [`OrbitScoreError::TimeSystemMismatch`],
///   [`OrbitScoreError::TimeSystemLabelMismatch`],
///   [`OrbitScoreError::InstantTimeSystemMismatch`]: the source, or an evaluation
///   instant, is not in the reference product's time system.
/// - [`OrbitScoreError::FrameMismatch`], [`OrbitScoreError::FrameTransform`]: the
///   source frame differs from the reference frame and the options do not
///   reconcile them, or either states a blank frame label.
/// - [`OrbitScoreError::NoSisreWeights`],
///   [`OrbitScoreError::UndeterminedOrbitRegime`]: no SISRE weights for a
///   scored satellite.
/// - [`OrbitScoreError::NonFiniteSourceState`],
///   [`OrbitScoreError::DegenerateReferenceState`],
///   [`OrbitScoreError::NonFiniteResult`]: a state that cannot be differenced.
/// - With [`GapPolicy::Refuse`], the first gap as
///   [`OrbitScoreError::MissingSatellite`], [`OrbitScoreError::MissingEpoch`] or
///   [`OrbitScoreError::NoPredictionOrigin`].
pub fn score_orbit(
    source: &dyn OrbitSource,
    reference: &Sp3,
    satellites: &[GnssSatelliteId],
    options: &OrbitScoreOptions,
) -> Result<OrbitScoreReport, OrbitScoreError> {
    let edges = &options.horizon_bin_edges_s;
    validate_options(options)?;
    for (index, sat) in satellites.iter().enumerate() {
        if satellites[..index].contains(sat) {
            return Err(invalid_input("satellites", "a satellite is listed twice"));
        }
    }

    let reference_scale = reference.header.time_scale;
    if source.time_scale() != reference_scale {
        return Err(OrbitScoreError::TimeSystemMismatch {
            orbit_source: source.time_scale(),
            reference: reference_scale,
        });
    }
    let reference_system = reference.header.time_system;
    if let Some(source_system) = source
        .sp3_time_system()
        .or_else(|| sp3_time_system_of(source.time_scale()))
    {
        if source_system != reference_system {
            return Err(OrbitScoreError::TimeSystemLabelMismatch {
                orbit_source: source_system,
                reference: reference_system,
            });
        }
    }
    let (frame, treatment) = reconcile_frames(
        &source.frame(),
        &reference.header.coordinate_system,
        &options.frame_reconciliation,
    )?;
    let epochs = evaluation_epochs(reference, &options.epochs)?;
    check_velocity_steps(&epochs, options.velocity_half_step_s)?;
    let with_relativity = source.clock_includes_relativity();

    let mut scored: Vec<ScoredSatellite> = Vec::with_capacity(satellites.len());
    for &sat in satellites {
        scored.push(score_satellite(
            source,
            reference,
            sat,
            &epochs,
            options,
            treatment,
            with_relativity,
        )?);
    }

    // Clock datum: per constellation, the median over its satellites at each
    // epoch of the clock differences.
    let systems = systems_in_order(satellites);
    let mut datum_by_system: BTreeMap<GnssSystem, Vec<Option<f64>>> = BTreeMap::new();
    if options.clock_datum == ClockDatum::EpochSystemMedian {
        for &system in &systems {
            let entries = scored
                .iter()
                .filter(|sat| sat.satellite.system == system)
                .flat_map(|sat| sat.samples.iter())
                .filter_map(|sample| sample.clock_m.map(|clock| (sample.epoch_index, clock)));
            datum_by_system.insert(system, median_by_epoch(entries, epochs.len(), 2));
        }
    }

    let mut per_satellite = Vec::with_capacity(scored.len());
    let mut per_system_series: BTreeMap<GnssSystem, Vec<BinSeries>> = BTreeMap::new();
    let mut per_system_coverage: BTreeMap<GnssSystem, ScoreCoverage> = BTreeMap::new();
    let mut coverage = ScoreCoverage::default();
    for sat in &scored {
        let system = sat.satellite.system;
        let datum = datum_by_system.get(&system);
        let mut series = vec![BinSeries::default(); edges.len() - 1];
        let mut clocked = 0usize;
        for sample in &sat.samples {
            let clock_m = match (options.clock_datum, sample.clock_m) {
                (ClockDatum::Unadjusted, clock) => clock,
                (ClockDatum::EpochSystemMedian, Some(clock)) => datum
                    .and_then(|datum| datum[sample.epoch_index])
                    .map(|datum| clock - datum),
                (ClockDatum::EpochSystemMedian, None) => None,
            };
            // A scored sample has weights: they are resolved at the satellite's
            // first scored epoch.
            let Some(weights) = sat.sisre_weights else {
                continue;
            };
            let overflow = |_: OrbitScoreError| OrbitScoreError::NonFiniteResult {
                satellite: sat.satellite,
            };
            let bin = &mut series[sample.bin];
            bin.radial.push(sample.components.radial_m);
            bin.transverse.push(sample.components.transverse_m);
            bin.normal.push(sample.components.normal_m);
            bin.orbit_3d.push(sample.orbit_3d_m);
            bin.sisre_orbit
                .push(weights.orbit_only_m(sample.components).map_err(overflow)?);
            if let Some(clock_m) = clock_m {
                bin.clock.push(clock_m);
                bin.sisre.push(
                    weights
                        .with_clock_m(sample.components, clock_m)
                        .map_err(overflow)?,
                );
                clocked += 1;
            }
        }

        let mut sat_coverage = sat.coverage;
        sat_coverage.clocked = clocked;
        let bins = bin_stats(edges, &series).ok_or(OrbitScoreError::NonFiniteResult {
            satellite: sat.satellite,
        })?;

        let system_series = per_system_series
            .entry(system)
            .or_insert_with(|| vec![BinSeries::default(); edges.len() - 1]);
        for (total, bin) in system_series.iter_mut().zip(&series) {
            total.extend(bin);
        }
        per_system_coverage
            .entry(system)
            .or_default()
            .add(&sat_coverage);
        coverage.add(&sat_coverage);

        per_satellite.push(SatelliteScore {
            satellite: sat.satellite,
            prediction_origin_j2000_s: sat.origin_j2000_s,
            orbit_regime: sat.orbit_regime,
            sisre_weights: sat.sisre_weights,
            bins,
            coverage: sat_coverage,
        });
    }

    // The series of a constellation are its satellites' series laid end to end
    // in the requested order, so a statistic that overflows only there is
    // reported against the constellation's first satellite.
    let mut per_system = Vec::with_capacity(systems.len());
    for &sat in satellites {
        let system = sat.system;
        if per_system
            .iter()
            .any(|score: &SystemScore| score.system == system)
        {
            continue;
        }
        let Some(series) = per_system_series.get(&system) else {
            continue;
        };
        let bins =
            bin_stats(edges, series).ok_or(OrbitScoreError::NonFiniteResult { satellite: sat })?;
        per_system.push(SystemScore {
            system,
            bins,
            coverage: per_system_coverage
                .get(&system)
                .copied()
                .unwrap_or_default(),
        });
    }

    Ok(OrbitScoreReport {
        horizon_bin_edges_s: edges.clone(),
        frame,
        source_reference_point: source.reference_point(),
        per_satellite,
        per_system,
        coverage,
    })
}

fn validate_options(options: &OrbitScoreOptions) -> Result<(), OrbitScoreError> {
    let edges = &options.horizon_bin_edges_s;
    if edges.len() < 2 {
        return Err(invalid_input("horizon_bin_edges_s", "fewer than two edges"));
    }
    if edges.iter().any(|edge| !edge.is_finite()) {
        return Err(invalid_input("horizon_bin_edges_s", "not finite"));
    }
    if edges.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(invalid_input(
            "horizon_bin_edges_s",
            "not strictly increasing",
        ));
    }
    validate_velocity_half_step(options.velocity_half_step_s)?;
    for set in &options.frame_reconciliation.asserted_equivalent_label_sets {
        let mut labels = set.labels.iter().map(|label| label.trim());
        let blank = labels.clone().any(str::is_empty);
        let first = labels.next();
        if blank || !labels.any(|label| Some(label) != first) {
            return Err(invalid_input(
                "frame_reconciliation.asserted_equivalent_label_sets",
                "a set names fewer than two distinct labels, or a blank one",
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_velocity_half_step(half_step_s: f64) -> Result<(), OrbitScoreError> {
    if !(half_step_s.is_finite() && half_step_s > 0.0) {
        return Err(invalid_input(
            "velocity_half_step_s",
            "not positive and finite",
        ));
    }
    if half_step_s < MIN_VELOCITY_HALF_STEP_S {
        return Err(invalid_input(
            "velocity_half_step_s",
            "below the 1e-8 s resolution of SP3 epochs",
        ));
    }
    Ok(())
}

/// The instants `(t - h, t + h)` of the reference velocity difference at `t`,
/// as they round.
pub(crate) fn velocity_instants(t_j2000_s: f64, half_step_s: f64) -> (f64, f64) {
    (t_j2000_s - half_step_s, t_j2000_s + half_step_s)
}

pub(crate) fn velocity_instants_are_valid(t: f64, (minus, plus): (f64, f64)) -> bool {
    minus.is_finite() && plus.is_finite() && minus < t && t < plus
}

/// Refuse a velocity half step that rounds away at an evaluation epoch: `t - h`
/// and `t + h` have to be finite and distinct from `t`, or the difference has
/// no step and the frame would be built from no velocity at all.
fn check_velocity_steps(epochs: &[EvalEpoch], half_step_s: f64) -> Result<(), OrbitScoreError> {
    for epoch in epochs {
        let t = epoch.t_j2000_s;
        let (minus, plus) = velocity_instants(t, half_step_s);
        if !velocity_instants_are_valid(t, (minus, plus)) {
            return Err(invalid_input(
                "velocity_half_step_s",
                "rounds away at an evaluation epoch",
            ));
        }
    }
    Ok(())
}

/// The reference velocity from the positions at the rounded instants of
/// [`velocity_instants`]: centered where both neighbours exist, one-sided from
/// the epoch position `r0` where one does, each divided by the step between
/// the instants as they round. That is `2h` and `h` exactly wherever `t +- h`
/// are exact, as on any whole-second epoch with a binary-fraction half step.
pub(crate) fn reference_velocity(
    rp: Option<[f64; 3]>,
    rm: Option<[f64; 3]>,
    r0: [f64; 3],
    t_j2000_s: f64,
    (minus, plus): (f64, f64),
) -> Option<[f64; 3]> {
    match (rp, rm) {
        (Some(rp), Some(rm)) => Some(scale3(sub3(rp, rm), 1.0 / (plus - minus))),
        (Some(rp), None) => Some(scale3(sub3(rp, r0), 1.0 / (plus - t_j2000_s))),
        (None, Some(rm)) => Some(scale3(sub3(r0, rm), 1.0 / (t_j2000_s - minus))),
        (None, None) => None,
    }
}

/// The constellations of `satellites`, in order of first appearance.
fn systems_in_order(satellites: &[GnssSatelliteId]) -> Vec<GnssSystem> {
    let mut systems = Vec::new();
    for sat in satellites {
        if !systems.contains(&sat.system) {
            systems.push(sat.system);
        }
    }
    systems
}

/// Match the source frame to the reference frame label under the caller's
/// reconciliation options.
fn reconcile_frames(
    source_frame: &OrbitFrame,
    reference_label: &str,
    options: &Sp3FrameReconciliationOptions,
) -> Result<(FrameAgreement, FrameTreatment), OrbitScoreError> {
    let source_label = match source_frame {
        OrbitFrame::BroadcastDatums => {
            return Ok((FrameAgreement::BroadcastDatums, FrameTreatment::AsStated));
        }
        OrbitFrame::Label(label) => label.trim(),
    };
    let reference_label = reference_label.trim();
    // A blank label states no frame, so it matches nothing, not even another
    // blank label.
    if source_label.is_empty() || reference_label.is_empty() {
        return Err(OrbitScoreError::FrameMismatch {
            orbit_source: source_label.to_string(),
            reference: reference_label.to_string(),
        });
    }
    if source_label == reference_label {
        return Ok((FrameAgreement::SameLabel, FrameTreatment::AsStated));
    }
    let names = |set: &crate::sp3::Sp3FrameLabelSet, label: &str| {
        set.labels.iter().any(|named| named.trim() == label)
    };
    let asserted = options
        .asserted_equivalent_label_sets
        .iter()
        .any(|set| names(set, source_label) && names(set, reference_label));
    if asserted {
        return Ok((FrameAgreement::AssertedEquivalent, FrameTreatment::AsStated));
    }
    let mismatch = || OrbitScoreError::FrameMismatch {
        orbit_source: source_label.to_string(),
        reference: reference_label.to_string(),
    };
    if !options.helmert {
        return Err(mismatch());
    }
    let from = sp3_coordinate_label_frame(source_label).ok_or_else(mismatch)?;
    let to = sp3_coordinate_label_frame(reference_label).ok_or_else(mismatch)?;
    Ok((
        FrameAgreement::Helmert { from, to },
        FrameTreatment::Helmert { from, to },
    ))
}

/// The evaluation epochs on the reference time axis, strictly increasing.
fn evaluation_epochs(
    reference: &Sp3,
    epochs: &ScoreEpochs,
) -> Result<Vec<EvalEpoch>, OrbitScoreError> {
    let scale = reference.header.time_scale;
    let (out, field): (Vec<EvalEpoch>, &'static str) = match epochs {
        ScoreEpochs::ReferenceRecords => (
            reference
                .epochs
                .iter()
                .copied()
                .zip(sp3_epoch_seconds(reference, "reference.epochs")?)
                .map(|(instant, t_j2000_s)| EvalEpoch { instant, t_j2000_s })
                .collect(),
            "reference.epochs",
        ),
        ScoreEpochs::Instants(instants) => {
            let mut out = Vec::with_capacity(instants.len());
            for (index, instant) in instants.iter().enumerate() {
                out.push(EvalEpoch {
                    instant: *instant,
                    t_j2000_s: seconds_in_scale(instant, scale, "epochs", index)?,
                });
            }
            (out, "epochs")
        }
    };
    if out
        .windows(2)
        .any(|pair| pair[1].t_j2000_s <= pair[0].t_j2000_s)
    {
        return Err(invalid_input(field, "not strictly increasing"));
    }
    Ok(out)
}

/// The scored epochs of one satellite before the clock datum is applied.
struct ScoredSatellite {
    satellite: GnssSatelliteId,
    origin_j2000_s: Option<f64>,
    orbit_regime: Option<OrbitRegime>,
    sisre_weights: Option<SisreWeights>,
    samples: Vec<Sample>,
    coverage: ScoreCoverage,
}

fn score_satellite(
    source: &dyn OrbitSource,
    reference: &Sp3,
    sat: GnssSatelliteId,
    epochs: &[EvalEpoch],
    options: &OrbitScoreOptions,
    treatment: FrameTreatment,
    with_relativity: bool,
) -> Result<ScoredSatellite, OrbitScoreError> {
    let origin_j2000_s = source.prediction_origin_j2000_s(sat);
    if origin_j2000_s.is_some_and(|origin| !origin.is_finite()) {
        return Err(invalid_input("prediction_origin", "not finite"));
    }

    let mut out = ScoredSatellite {
        satellite: sat,
        origin_j2000_s,
        orbit_regime: None,
        sisre_weights: None,
        samples: Vec::new(),
        coverage: ScoreCoverage {
            epochs: epochs.len(),
            ..ScoreCoverage::default()
        },
    };

    // Reasons that hold for the satellite as a whole come before any epoch's:
    // a product with nothing for it, then a source with no prediction origin
    // for it. Whether a product has the satellite at all does not depend on the
    // instant it is asked at.
    let satellite_gap = epochs.first().and_then(|first| {
        let t = first.t_j2000_s;
        if let Err(error @ Error::UnknownSatellite(_)) = reference.position_at_j2000_seconds(sat, t)
        {
            Some(Gap::Missing {
                product: ScoredProduct::Reference,
                error,
            })
        } else if let Err(error @ Error::UnknownSatellite(_)) = source.state_at_j2000_s(sat, t) {
            Some(Gap::Missing {
                product: ScoredProduct::Source,
                error,
            })
        } else if origin_j2000_s.is_none() {
            Some(Gap::NoPredictionOrigin)
        } else {
            None
        }
    });
    if let Some(gap) = satellite_gap {
        if options.gaps == GapPolicy::Refuse {
            if let Some(error) = gap_error(&gap, sat, 0) {
                return Err(error);
            }
        }
        for _ in epochs {
            count_gap(&mut out.coverage, &gap);
        }
        return Ok(out);
    }
    let Some(origin_j2000_s) = origin_j2000_s else {
        return Ok(out);
    };

    for (epoch_index, epoch) in epochs.iter().enumerate() {
        let evaluated = evaluate(
            source,
            reference,
            sat,
            epoch_index,
            epoch,
            origin_j2000_s,
            options,
            treatment,
            with_relativity,
        )?;
        match evaluated {
            Ok((sample, inertial_velocity_m_s, reference_position_m)) => {
                if out.sisre_weights.is_none() {
                    let (regime, weights) = resolve_weights(
                        sat,
                        options.sisre_weights,
                        reference_position_m,
                        inertial_velocity_m_s,
                    )?;
                    out.orbit_regime = regime;
                    out.sisre_weights = Some(weights);
                }
                out.samples.push(sample);
                out.coverage.scored += 1;
            }
            Err(gap) => {
                if options.gaps == GapPolicy::Refuse {
                    if let Some(error) = gap_error(&gap, sat, epoch_index) {
                        return Err(error);
                    }
                }
                count_gap(&mut out.coverage, &gap);
            }
        }
    }
    Ok(out)
}

/// The SISRE weights of a satellite, with the orbit regime where the weights
/// are looked up by it.
fn resolve_weights(
    sat: GnssSatelliteId,
    selection: SisreWeightSelection,
    reference_position_m: [f64; 3],
    inertial_velocity_m_s: [f64; 3],
) -> Result<(Option<OrbitRegime>, SisreWeights), OrbitScoreError> {
    match selection {
        SisreWeightSelection::Fixed(weights) => Ok((None, weights)),
        SisreWeightSelection::Montenbruck2018 => {
            let regime =
                OrbitRegime::from_inertial_state(reference_position_m, inertial_velocity_m_s)
                    .ok_or(OrbitScoreError::UndeterminedOrbitRegime { satellite: sat })?;
            let weights = SisreWeights::montenbruck_2018(sat.system, regime).ok_or(
                OrbitScoreError::NoSisreWeights {
                    satellite: sat,
                    regime,
                },
            )?;
            Ok((Some(regime), weights))
        }
    }
}

fn count_gap(coverage: &mut ScoreCoverage, gap: &Gap) {
    match gap {
        Gap::OutsideHorizonBins => coverage.outside_horizon_bins += 1,
        Gap::NoPredictionOrigin => coverage.no_prediction_origin += 1,
        Gap::Missing { product, error } => {
            let satellite_missing = matches!(error, Error::UnknownSatellite(_));
            match (product, satellite_missing) {
                (ScoredProduct::Source, true) => coverage.source_satellite_missing += 1,
                (ScoredProduct::Source, false) => coverage.source_epoch_missing += 1,
                (ScoredProduct::Reference, true) => coverage.reference_satellite_missing += 1,
                (ScoredProduct::Reference, false) => coverage.reference_epoch_missing += 1,
            }
        }
    }
}

/// The typed error a gap is refused with, or `None` for an epoch outside the
/// horizon bins.
fn gap_error(gap: &Gap, sat: GnssSatelliteId, epoch_index: usize) -> Option<OrbitScoreError> {
    match gap {
        Gap::OutsideHorizonBins => None,
        Gap::NoPredictionOrigin => Some(OrbitScoreError::NoPredictionOrigin { satellite: sat }),
        Gap::Missing { product, error } => Some(match error {
            Error::UnknownSatellite(_) => OrbitScoreError::MissingSatellite {
                product: *product,
                satellite: sat,
            },
            other => OrbitScoreError::MissingEpoch {
                product: *product,
                satellite: sat,
                epoch_index,
                error: other.clone(),
            },
        }),
    }
}

/// The bin of a prediction horizon: `i` with `edges[i] < horizon <= edges[i + 1]`.
fn bin_index(edges: &[f64], horizon_s: f64) -> Option<usize> {
    let below = edges.partition_point(|&edge| edge < horizon_s);
    if below == 0 || below == edges.len() {
        None
    } else {
        Some(below - 1)
    }
}

/// Score one satellite at one epoch: the sample with the inertial reference
/// velocity and reference position it was formed from, or the reason there is
/// none.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn evaluate(
    source: &dyn OrbitSource,
    reference: &Sp3,
    sat: GnssSatelliteId,
    epoch_index: usize,
    epoch: &EvalEpoch,
    origin_j2000_s: f64,
    options: &OrbitScoreOptions,
    treatment: FrameTreatment,
    with_relativity: bool,
) -> Result<Result<(Sample, [f64; 3], [f64; 3]), Gap>, OrbitScoreError> {
    let t = epoch.t_j2000_s;
    let Some(bin) = bin_index(&options.horizon_bin_edges_s, t - origin_j2000_s) else {
        return Ok(Err(Gap::OutsideHorizonBins));
    };

    let (reference_position_m, reference_clock_s) =
        match reference_state_at(reference, sat, t, with_relativity) {
            Ok(state) => state,
            Err(error) => {
                return Ok(Err(Gap::Missing {
                    product: ScoredProduct::Reference,
                    error,
                }));
            }
        };
    let (minus, plus) = velocity_instants(t, options.velocity_half_step_s);
    let Some(earth_fixed_velocity_m_s) = reference_velocity(
        reference_position_at(reference, sat, plus),
        reference_position_at(reference, sat, minus),
        reference_position_m,
        t,
        (minus, plus),
    ) else {
        return Ok(Err(Gap::Missing {
            product: ScoredProduct::Reference,
            error: Error::EpochOutOfRange,
        }));
    };

    let state = match source.state_at_j2000_s(sat, t) {
        Ok(state) => state,
        Err(error) => {
            return Ok(Err(Gap::Missing {
                product: ScoredProduct::Source,
                error,
            }));
        }
    };
    if !finite3(state.position_ecef_m) || state.clock_s.is_some_and(|clock| !clock.is_finite()) {
        return Err(OrbitScoreError::NonFiniteSourceState {
            satellite: sat,
            epoch_index,
        });
    }
    let source_position_m = match treatment {
        FrameTreatment::AsStated => state.position_ecef_m,
        FrameTreatment::Helmert { from, to } => {
            let position = TerrestrialPositionM::from_array(state.position_ecef_m)
                .map_err(OrbitScoreError::FrameTransform)?;
            frame_catalog::transform(position, None, from, to, sp3_decimal_year(epoch.instant))
                .map_err(OrbitScoreError::FrameTransform)?
                .position
                .as_array()
        }
    };

    let inertial_velocity_m_s = add_earth_rotation(reference_position_m, earth_fixed_velocity_m_s);
    let triad_velocity_m_s = match options.triad {
        TriadVelocity::Inertial => inertial_velocity_m_s,
        TriadVelocity::EarthFixed => earth_fixed_velocity_m_s,
    };
    if !finite3(triad_velocity_m_s) || !triad_is_defined(reference_position_m, triad_velocity_m_s) {
        return Err(OrbitScoreError::DegenerateReferenceState {
            satellite: sat,
            epoch_index,
        });
    }

    let difference = orbit_difference(
        source_position_m,
        state.clock_s,
        reference_position_m,
        reference_clock_s,
        triad_velocity_m_s,
    );
    let finite = difference.orbit_3d.is_finite()
        && difference.radial.is_finite()
        && difference.along.is_finite()
        && difference.cross.is_finite()
        && difference.clock_m.is_none_or(f64::is_finite);
    if !finite {
        return Err(OrbitScoreError::NonFiniteResult { satellite: sat });
    }

    Ok(Ok((
        Sample {
            epoch_index,
            bin,
            components: RtnComponents {
                radial_m: difference.radial,
                transverse_m: difference.along,
                normal_m: difference.cross,
            },
            orbit_3d_m: difference.orbit_3d,
            clock_m: difference.clock_m,
        },
        inertial_velocity_m_s,
        reference_position_m,
    )))
}

/// The error series of one horizon bin, in the order the samples were scored.
#[derive(Debug, Clone, Default)]
struct BinSeries {
    radial: Vec<f64>,
    transverse: Vec<f64>,
    normal: Vec<f64>,
    orbit_3d: Vec<f64>,
    sisre_orbit: Vec<f64>,
    clock: Vec<f64>,
    sisre: Vec<f64>,
}

impl BinSeries {
    fn extend(&mut self, other: &BinSeries) {
        self.radial.extend_from_slice(&other.radial);
        self.transverse.extend_from_slice(&other.transverse);
        self.normal.extend_from_slice(&other.normal);
        self.orbit_3d.extend_from_slice(&other.orbit_3d);
        self.sisre_orbit.extend_from_slice(&other.sisre_orbit);
        self.clock.extend_from_slice(&other.clock);
        self.sisre.extend_from_slice(&other.sisre);
    }
}

/// RMS, mean and largest magnitude of a series; `None` for an empty one, as
/// [`rms`], [`mean`] and [`max_abs`] fold them.
fn error_stats(values: &[f64]) -> Option<ErrorStats> {
    if values.is_empty() {
        return None;
    }
    Some(ErrorStats {
        rms_m: rms(values),
        mean_m: mean(values),
        max_abs_m: max_abs(values),
    })
}

/// The statistics of every bin, or `None` when one of them is not finite.
fn bin_stats(edges: &[f64], series: &[BinSeries]) -> Option<Vec<HorizonBinStats>> {
    let mut out = Vec::with_capacity(series.len());
    for (index, bin) in series.iter().enumerate() {
        let stats = HorizonBinStats {
            horizon_start_s: edges[index],
            horizon_end_s: edges[index + 1],
            count: bin.radial.len(),
            radial: error_stats(&bin.radial),
            transverse: error_stats(&bin.transverse),
            normal: error_stats(&bin.normal),
            orbit_3d: error_stats(&bin.orbit_3d),
            sisre_orbit: error_stats(&bin.sisre_orbit),
            clock_count: bin.clock.len(),
            clock: error_stats(&bin.clock),
            sisre: error_stats(&bin.sisre),
        };
        let all = [
            stats.radial,
            stats.transverse,
            stats.normal,
            stats.orbit_3d,
            stats.sisre_orbit,
            stats.clock,
            stats.sisre,
        ];
        let finite = all.iter().flatten().all(|stats| {
            stats.rms_m.is_finite() && stats.mean_m.is_finite() && stats.max_abs_m.is_finite()
        });
        if !finite {
            return None;
        }
        out.push(stats);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gps(prn: u8) -> GnssSatelliteId {
        GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite")
    }

    /// A circular orbit of radius `radius_m` inclined by `inclination_rad`,
    /// at the ascending node: position on the x-axis, velocity in the y-z plane.
    fn circular_state(radius_m: f64, inclination_rad: f64) -> ([f64; 3], [f64; 3]) {
        let speed = (MU_EARTH * 1.0e9 / radius_m).sqrt();
        (
            [radius_m, 0.0, 0.0],
            [
                0.0,
                speed * libm::cos(inclination_rad),
                speed * libm::sin(inclination_rad),
            ],
        )
    }

    #[test]
    fn horizon_bins_are_open_below_and_closed_above() {
        let edges = [0.0, 10.0, 20.0];
        assert_eq!(bin_index(&edges, -1.0), None);
        assert_eq!(bin_index(&edges, 0.0), None);
        assert_eq!(bin_index(&edges, 1.0e-9), Some(0));
        assert_eq!(bin_index(&edges, 10.0), Some(0));
        assert_eq!(bin_index(&edges, 10.000_001), Some(1));
        assert_eq!(bin_index(&edges, 20.0), Some(1));
        assert_eq!(bin_index(&edges, 20.000_001), None);
    }

    #[test]
    fn error_stats_are_rms_mean_and_largest_magnitude() {
        assert_eq!(error_stats(&[]), None);
        let stats = error_stats(&[3.0, -4.0]).expect("stats");
        assert_eq!(stats.rms_m, 12.5_f64.sqrt());
        assert_eq!(stats.mean_m, -0.5);
        assert_eq!(stats.max_abs_m, 4.0);
    }

    #[test]
    fn per_epoch_median_needs_the_minimum_count() {
        let entries = [(0, 1.0), (0, 3.0), (1, 5.0), (2, 4.0), (2, 2.0), (2, 9.0)];
        assert_eq!(
            median_by_epoch(entries, 4, 2),
            vec![Some(2.0), None, Some(4.0), None]
        );
        assert_eq!(
            median_by_epoch(entries, 4, 1),
            vec![Some(2.0), Some(5.0), Some(4.0), None]
        );
    }

    #[test]
    fn sisre_weights_are_projections() {
        assert!(SisreWeights::new(0.98, 1.0 / 49.0).is_ok());
        assert!(SisreWeights::new(0.0, 0.0).is_ok());
        assert!(SisreWeights::new(1.0, 1.0).is_ok());
        for (radial, transverse_normal_squared) in [
            (f64::NAN, 0.02),
            (-0.1, 0.02),
            (1.1, 0.02),
            (0.98, f64::INFINITY),
            (0.98, -0.02),
            (0.98, 1.5),
        ] {
            assert!(matches!(
                SisreWeights::new(radial, transverse_normal_squared),
                Err(OrbitScoreError::InvalidInput { .. })
            ));
        }
    }

    #[test]
    fn published_weight_rows() {
        let row = |system, regime| {
            SisreWeights::montenbruck_2018(system, regime)
                .map(|w| (w.radial(), w.transverse_normal_squared()))
        };
        assert_eq!(
            row(GnssSystem::Gps, OrbitRegime::Meo),
            Some((0.98, 1.0 / 49.0))
        );
        assert_eq!(
            row(GnssSystem::Glonass, OrbitRegime::Meo),
            Some((0.98, 1.0 / 45.0))
        );
        assert_eq!(
            row(GnssSystem::Galileo, OrbitRegime::Meo),
            Some((0.98, 1.0 / 61.0))
        );
        assert_eq!(
            row(GnssSystem::BeiDou, OrbitRegime::Meo),
            Some((0.98, 1.0 / 54.0))
        );
        for system in [
            GnssSystem::BeiDou,
            GnssSystem::Qzss,
            GnssSystem::Navic,
            GnssSystem::Sbas,
        ] {
            for regime in [OrbitRegime::Igso, OrbitRegime::Geo] {
                assert_eq!(row(system, regime), Some((0.99, 1.0 / 126.0)), "{system:?}");
            }
        }
        assert_eq!(row(GnssSystem::Gps, OrbitRegime::Geo), None);
        assert_eq!(row(GnssSystem::Qzss, OrbitRegime::Meo), None);
        assert_eq!(row(GnssSystem::Sbas, OrbitRegime::Meo), None);
    }

    // SISRE(orb) = sqrt(w_R^2 R^2 + w_AC^2 (T^2 + N^2)) and
    // SISRE = sqrt((w_R R - c dt)^2 + w_AC^2 (T^2 + N^2)).
    #[test]
    fn sisre_follows_the_published_formula() {
        let weights = SisreWeights::new(0.98, 1.0 / 49.0).expect("weights");
        let components = |radial_m, transverse_m, normal_m| RtnComponents {
            radial_m,
            transverse_m,
            normal_m,
        };
        // `w_AC^2 = 1/49` is not exact in binary, so `7^2 / 49` is within an
        // ulp of one.
        let close = |a: f64, b: f64| (a - b).abs() <= 2.0 * f64::EPSILON;
        let orbit_only = |c| weights.orbit_only_m(c).expect("finite");
        let with_clock = |c, clock| weights.with_clock_m(c, clock).expect("finite");
        assert_eq!(orbit_only(components(1.0, 0.0, 0.0)), 0.98);
        assert!(close(orbit_only(components(0.0, 7.0, 0.0)), 1.0));
        assert!(close(orbit_only(components(0.0, 0.0, -7.0)), 1.0));
        assert_eq!(with_clock(components(1.0, 0.0, 0.0), 0.98), 0.0);
        assert_eq!(with_clock(components(0.0, 0.0, 0.0), -1.5), 1.5);
        let (r, t, n, clock) = (0.3, -1.2, 0.8, 0.5);
        let orbit = ((0.98 * r) * (0.98 * r) + (t * t + n * n) / 49.0_f64).sqrt();
        let expected =
            ((0.98 * r - clock) * (0.98 * r - clock) + (t * t + n * n) / 49.0_f64).sqrt();
        assert!((orbit_only(components(r, t, n)) - orbit).abs() < 1.0e-15);
        assert!((with_clock(components(r, t, n), clock) - expected).abs() < 1.0e-15);

        let refused = |result: Result<f64, OrbitScoreError>| {
            matches!(result, Err(OrbitScoreError::InvalidInput { .. }))
        };
        assert!(refused(weights.orbit_only_m(components(
            f64::NAN,
            0.0,
            0.0
        ))));
        assert!(refused(
            weights.with_clock_m(components(0.0, 0.0, 0.0), f64::INFINITY)
        ));
        // Only an error beyond the largest `f64` is refused.
        let unit = SisreWeights::new(1.0, 1.0).expect("weights");
        assert!(refused(
            unit.orbit_only_m(components(1.5e308, 1.5e308, 0.0))
        ));
        assert!(refused(
            unit.with_clock_m(components(1.5e308, 0.0, 0.0), -1.5e308)
        ));
    }

    // Every error that is a finite number is returned, whatever the squares of
    // its components do: a zero weight drops its components however large,
    // and neither a huge nor a tiny error is lost to overflow or underflow.
    #[test]
    fn sisre_keeps_every_finite_error() {
        let components = |radial_m, transverse_m, normal_m| RtnComponents {
            radial_m,
            transverse_m,
            normal_m,
        };
        let radial_only = SisreWeights::new(0.98, 0.0).expect("weights");
        assert_eq!(
            radial_only.orbit_only_m(components(1.0, 1.0e200, 0.0)),
            Ok(0.98)
        );
        assert_eq!(
            radial_only.with_clock_m(components(1.0, 0.0, -1.0e300), 0.5),
            Ok(0.98 - 0.5)
        );

        let weights = SisreWeights::new(0.98, 1.0 / 49.0).expect("weights");
        for radial_m in [1.0e300, 1.0e-200, -3.0e-310] {
            assert_eq!(
                weights.orbit_only_m(components(radial_m, 0.0, 0.0)),
                Ok((0.98 * radial_m).abs()),
                "{radial_m:e}"
            );
        }
        assert_eq!(
            weights.with_clock_m(components(1.0e200, 0.0, 0.0), 0.0),
            Ok(0.98e200)
        );
        // `T = 7e200` with `w_AC = 1/7`: one meter per 1e-200 meters, to the
        // rounding of `sqrt(1/49)`.
        let transverse = weights
            .orbit_only_m(components(0.0, 7.0e200, 0.0))
            .expect("finite");
        assert!((transverse / 1.0e200 - 1.0).abs() <= 4.0 * f64::EPSILON);
        let tiny = weights
            .orbit_only_m(components(0.0, 0.0, 7.0e-200))
            .expect("finite");
        assert!((tiny / 1.0e-200 - 1.0).abs() <= 4.0 * f64::EPSILON);
    }

    #[test]
    fn orbit_regimes_of_navigation_orbits() {
        let degree = core::f64::consts::PI / 180.0;
        let (r, v) = circular_state(26_560_000.0, 55.0 * degree);
        assert_eq!(
            OrbitRegime::from_inertial_state(r, v),
            Some(OrbitRegime::Meo)
        );
        let (r, v) = circular_state(29_600_000.0, 56.0 * degree);
        assert_eq!(
            OrbitRegime::from_inertial_state(r, v),
            Some(OrbitRegime::Meo)
        );
        let (r, v) = circular_state(42_164_000.0, 1.5 * degree);
        assert_eq!(
            OrbitRegime::from_inertial_state(r, v),
            Some(OrbitRegime::Geo)
        );
        let (r, v) = circular_state(42_164_000.0, 55.0 * degree);
        assert_eq!(
            OrbitRegime::from_inertial_state(r, v),
            Some(OrbitRegime::Igso)
        );
        // Retrograde equatorial is not geostationary.
        let (r, v) = circular_state(42_164_000.0, 179.0 * degree);
        assert_eq!(
            OrbitRegime::from_inertial_state(r, v),
            Some(OrbitRegime::Igso)
        );
        // Escape speed and beyond: no bound orbit.
        let (r, v) = circular_state(26_560_000.0, 55.0 * degree);
        let escape = [v[0] * 1.5, v[1] * 1.5, v[2] * 1.5];
        assert_eq!(OrbitRegime::from_inertial_state(r, escape), None);
        assert_eq!(OrbitRegime::from_inertial_state([0.0; 3], v), None);
        assert_eq!(
            OrbitRegime::from_inertial_state([f64::NAN, 0.0, 0.0], v),
            None
        );
    }

    #[test]
    fn weights_resolve_by_regime_or_refuse() {
        let degree = core::f64::consts::PI / 180.0;
        let (r, v) = circular_state(26_560_000.0, 55.0 * degree);
        let (regime, weights) =
            resolve_weights(gps(1), SisreWeightSelection::Montenbruck2018, r, v).expect("GPS");
        assert_eq!(regime, Some(OrbitRegime::Meo));
        assert_eq!(weights.radial(), 0.98);

        let qzss = GnssSatelliteId::new(GnssSystem::Qzss, 1).expect("valid satellite");
        assert_eq!(
            resolve_weights(qzss, SisreWeightSelection::Montenbruck2018, r, v),
            Err(OrbitScoreError::NoSisreWeights {
                satellite: qzss,
                regime: OrbitRegime::Meo,
            })
        );
        let escape = [v[0] * 2.0, v[1] * 2.0, v[2] * 2.0];
        assert_eq!(
            resolve_weights(gps(1), SisreWeightSelection::Montenbruck2018, r, escape),
            Err(OrbitScoreError::UndeterminedOrbitRegime { satellite: gps(1) })
        );
        let fixed = SisreWeights::new(0.5, 0.25).expect("weights");
        assert_eq!(
            resolve_weights(qzss, SisreWeightSelection::Fixed(fixed), r, escape),
            Ok((None, fixed))
        );
    }

    #[test]
    fn rtn_components_of_a_circular_orbit() {
        let (r, v) = circular_state(26_560_000.0, 0.0);
        // Radial is +x, transverse +y (the velocity), normal +z.
        let rtn = rtn_components([1.0, 2.0, 3.0], r, v).expect("frame");
        assert_eq!(
            rtn,
            RtnComponents {
                radial_m: 1.0,
                transverse_m: 2.0,
                normal_m: 3.0,
            }
        );
    }

    #[test]
    fn rtn_components_refuse_what_defines_no_frame() {
        let (r, v) = circular_state(26_560_000.0, 0.0);
        let field = |result: Result<RtnComponents, OrbitScoreError>| match result {
            Err(OrbitScoreError::InvalidInput { field, .. }) => field,
            other => panic!("expected InvalidInput, got {other:?}"),
        };
        assert_eq!(
            field(rtn_components([f64::NAN, 0.0, 0.0], r, v)),
            "difference_m"
        );
        assert_eq!(
            field(rtn_components([0.0; 3], [f64::INFINITY, 0.0, 0.0], v)),
            "position_m"
        );
        assert_eq!(
            field(rtn_components([0.0; 3], r, [0.0, f64::NAN, 0.0])),
            "velocity_m_s"
        );
        assert_eq!(field(rtn_components([0.0; 3], [0.0; 3], v)), "position_m");
        assert_eq!(
            field(rtn_components([0.0; 3], r, [3.0, 0.0, 0.0])),
            "position_m"
        );
        assert_eq!(field(rtn_components([0.0; 3], r, [0.0; 3])), "position_m");
        // Components of the diagonal of a huge difference overflow.
        let diagonal = [1.0e7, 1.0e7, 0.0];
        assert_eq!(
            field(rtn_components([1.5e308, 1.5e308, 0.0], diagonal, v)),
            "difference_m"
        );
    }

    // The frame depends on the directions of `r` and `v` alone. A state whose
    // squares or cross product overflow or underflow has the frame of the same
    // directions at ordinary magnitudes, bit for bit, since the vectors are
    // scaled by powers of two.
    #[test]
    fn rtn_components_hold_for_states_of_any_magnitude() {
        let d = [1.0, 2.0, 3.0];
        let close = |rtn: RtnComponents| {
            [
                (rtn.radial_m, 1.0),
                (rtn.transverse_m, 2.0),
                (rtn.normal_m, 3.0),
            ]
            .iter()
            .all(|(got, want)| (got - want).abs() <= 4.0 * f64::EPSILON * want)
        };
        for (r, v) in [
            ([1.0e80, 0.0, 0.0], [0.0, 1.0e80, 0.0]),
            ([1.0e200, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([1.0e-200, 0.0, 0.0], [0.0, 1.0e-200, 0.0]),
            ([1.0e300, 0.0, 0.0], [0.0, 1.0e-300, 0.0]),
        ] {
            let rtn = rtn_components(d, r, v).expect("defined frame");
            assert!(close(rtn), "r {r:?}, v {v:?}: {rtn:?}");
        }

        let r = [2.0e7, -1.5e7, 0.9e7];
        let v = [1.2e3, 2.9e3, -0.4e3];
        let d = [0.31, -1.07, 2.25];
        let expected = rtn_components(d, r, v).expect("defined frame");
        for (r_exponent, v_exponent, d_exponent) in [(800, -900, 0), (-900, 800, 0), (0, 0, 1000)] {
            let scaled = |x: [f64; 3], exponent| x.map(|c| libm::scalbn(c, exponent));
            let rtn = rtn_components(
                scaled(d, d_exponent),
                scaled(r, r_exponent),
                scaled(v, v_exponent),
            )
            .expect("defined frame");
            let unscaled = RtnComponents {
                radial_m: libm::scalbn(rtn.radial_m, -d_exponent),
                transverse_m: libm::scalbn(rtn.transverse_m, -d_exponent),
                normal_m: libm::scalbn(rtn.normal_m, -d_exponent),
            };
            assert_eq!(unscaled, expected, "{r_exponent} {v_exponent} {d_exponent}");
        }
    }

    // A velocity within a hair of the position still defines a plane; the
    // frame stays orthonormal, so the difference keeps its length.
    #[test]
    fn rtn_components_stay_a_rotation_near_a_radial_velocity() {
        let r = [1.3e7, 1.7e7, 1.1e7];
        let d = [0.7, -0.2, 1.9];
        let length = |x: [f64; 3]| libm::hypot(libm::hypot(x[0], x[1]), x[2]);
        for tilt in [1.0e-3, 1.0e-8, 1.0e-12, 1.0e-15] {
            let v = [1.3e3 * (1.0 + tilt), 1.7e3, 1.1e3 * (1.0 - tilt)];
            let rtn = rtn_components(d, r, v).expect("defined frame");
            let kept = length([rtn.radial_m, rtn.transverse_m, rtn.normal_m]);
            assert!(
                (kept - length(d)).abs() <= 8.0 * f64::EPSILON * length(d),
                "tilt {tilt:e}: {kept} vs {}",
                length(d)
            );
        }
    }

    // A velocity a hair from the position still defines the plane of the two,
    // however small the angle: `r x v` is formed from the vectors as given, so
    // it neither underflows nor loses the components a scaled vector drops.
    #[test]
    fn rtn_components_hold_a_hair_from_parallel() {
        let d = [1.0, 2.0, 3.0];
        let v = [1.0, 0.0, 0.0];
        // Within `ulps` units in the last place of each expected component (the
        // unit vectors round to a unit in their last place).
        let close = |r: [f64; 3], expected: [f64; 3], ulps: f64| {
            let rtn = rtn_components(d, r, v).expect("defined frame");
            for (got, want) in [rtn.radial_m, rtn.transverse_m, rtn.normal_m]
                .into_iter()
                .zip(expected)
            {
                assert!(
                    (got - want).abs() <= ulps * f64::EPSILON * want.abs(),
                    "r {r:?}: {got} vs {want}"
                );
            }
        };
        // `r = (1, y, 0)`: radial is +x, `r x v = (0, 0, -y)` puts normal along
        // -z, and transverse is -y, whatever the size of `y`.
        for y in [2.7e-162, 1.0e-200, 1.0e-300, 5.0e-324] {
            close([1.0, y, 0.0], [1.0, -2.0, -3.0], 2.0);
        }
        // Components 2^1992 apart: scaling `r` to its largest component drops
        // the small one, the cross product keeps it.
        close(
            [libm::scalbn(1.0, 996), libm::scalbn(1.0, -996), 0.0],
            [1.0, -2.0, -3.0],
            2.0,
        );
        // Two small components that scaling `r` leaves in the subnormal range,
        // where they keep 45 of their 53 bits: `r x v = (0, 3e-10, -1e-10)`,
        // so normal is `(0, 3, -1) / sqrt(10)` and transverse
        // `(0, -1, -3) / sqrt(10)`.
        let root_ten = libm::sqrt(10.0);
        close(
            [libm::scalbn(1.0, 996), 1.0e-10, 3.0e-10],
            [1.0, -11.0 / root_ten, 3.0 / root_ten],
            4.0,
        );

        // Near parallel at ordinary magnitudes the frame is that of the same
        // directions at any other magnitudes, bit for bit.
        let r = [1.3e7, 1.7e7, 1.1e7];
        let v = [1.3e3 * (1.0 + 1.0e-12), 1.7e3, 1.1e3 * (1.0 - 1.0e-12)];
        let expected = rtn_components(d, r, v).expect("defined frame");
        for (r_exponent, v_exponent) in [(800, -900), (-900, 800), (-1000, -60)] {
            let scaled = |x: [f64; 3], exponent| x.map(|c| libm::scalbn(c, exponent));
            assert_eq!(
                rtn_components(d, scaled(r, r_exponent), scaled(v, v_exponent)),
                Ok(expected),
                "{r_exponent} {v_exponent}"
            );
        }
    }

    // A half step that rounds away at an epoch is refused; one that rounds to
    // another step is divided by the step it rounds to.
    #[test]
    fn velocity_steps_are_the_steps_the_instants_round_to() {
        let t = 831_902_400.0;
        let epoch = |t_j2000_s| EvalEpoch {
            instant: Instant::from_nanos(TimeScale::Gpst, 0),
            t_j2000_s,
        };
        assert!(check_velocity_steps(&[epoch(t)], 0.5).is_ok());
        for half_step_s in [1.0e-8, 5.0e-8] {
            assert!(matches!(
                check_velocity_steps(&[epoch(0.0), epoch(t)], half_step_s),
                Err(OrbitScoreError::InvalidInput {
                    field: "velocity_half_step_s",
                    ..
                })
            ));
        }

        // 0.3 s is not a whole number of the 2^-23 s spacing at `t`, so the
        // instants are 0.29999995... s from it. A uniform motion sampled there
        // gives back its velocity; dividing by `2h` would be off by 1.6e-7.
        let (minus, plus) = velocity_instants(t, 0.3);
        assert_ne!(plus - minus, 0.6);
        let velocity = [3_100.0, -1_250.0, 640.0];
        let r0 = [1.5e7, 2.0e7, -0.8e7];
        let at = |instant: f64| {
            let dt = instant - t;
            [
                r0[0] + velocity[0] * dt,
                r0[1] + velocity[1] * dt,
                r0[2] + velocity[2] * dt,
            ]
        };
        for (rp, rm) in [
            (Some(at(plus)), Some(at(minus))),
            (Some(at(plus)), None),
            (None, Some(at(minus))),
        ] {
            let estimate = reference_velocity(rp, rm, r0, t, (minus, plus)).expect("velocity");
            for (got, want) in estimate.iter().zip(velocity) {
                // The positions round at 3.7e-9 m over a 0.3 s step.
                assert!((got - want).abs() <= 3.0e-8, "{got} vs {want}");
            }
        }
        assert_eq!(reference_velocity(None, None, r0, t, (minus, plus)), None);
    }

    // No angular momentum, or so little that the orbit meets the Earth, is no
    // orbit regime; an eccentric orbit that clears the Earth has one.
    #[test]
    fn orbit_regime_needs_an_orbit_that_clears_the_earth() {
        let mu = MU_EARTH * 1.0e9;
        for radius_m in [26_560_000.0, 42_164_000.0, 80_000_000.0] {
            let r = [radius_m, 0.0, 0.0];
            assert_eq!(OrbitRegime::from_inertial_state(r, [0.0; 3]), None);
            assert_eq!(
                OrbitRegime::from_inertial_state(r, [1_000.0, 0.0, 0.0]),
                None
            );
            assert_eq!(
                OrbitRegime::from_inertial_state(r, [1_000.0, 50.0, 0.0]),
                None
            );
        }
        // At apogee 26 000 km: perigee 7 000 km clears the Earth, 6 000 km
        // does not.
        let apogee_m = 26_000_000.0;
        let at_apogee = |perigee_m: f64| {
            let semi_major_axis_m = (apogee_m + perigee_m) / 2.0;
            let speed = (mu * (2.0 / apogee_m - 1.0 / semi_major_axis_m)).sqrt();
            OrbitRegime::from_inertial_state([apogee_m, 0.0, 0.0], [0.0, speed, 0.0])
        };
        assert_eq!(at_apogee(7_000_000.0), Some(OrbitRegime::Meo));
        assert_eq!(at_apogee(6_000_000.0), None);
    }

    // An orbit far larger than a real one keeps its regime: its perigee is
    // formed with no `1 - e` to cancel, and a radius, semi-major axis or
    // perigee beyond the largest `f64` is compared without overflowing first.
    #[test]
    fn orbit_regime_of_orbits_of_any_size() {
        let mu = MU_EARTH * 1.0e9;
        // At 1e25 m, 2e-14 m/s across the radius: `1 - e` is 2.0e-17, below the
        // rounding of `e`; the perigee is `r nu / 2 = 50 175 km`.
        assert_eq!(
            OrbitRegime::from_inertial_state([1.0e25, 0.0, 0.0], [0.0, 2.0e-14, 0.0]),
            Some(OrbitRegime::Geo)
        );
        // `nu = 1.5` at 1e308 m: the semi-major axis is 2e308 m, the perigee
        // the radius.
        let speed = libm::sqrt(1.5 * mu / 1.0e308);
        assert_eq!(
            OrbitRegime::from_inertial_state([1.0e308, 0.0, 0.0], [0.0, speed, 0.0]),
            Some(OrbitRegime::Geo)
        );
        // Circular at `|r| = 1.5e308 sqrt(2)`, which overflows: in the equator,
        // and inclined 45 degrees.
        let r = [1.5e308, 1.5e308, 0.0];
        let speed = libm::sqrt(mu / 1.5e308) / libm::sqrt(libm::sqrt(2.0));
        let half = core::f64::consts::FRAC_1_SQRT_2;
        let equatorial = [-speed * half, speed * half, 0.0];
        let inclined = [-speed * 0.5, speed * 0.5, speed * half];
        assert_eq!(
            OrbitRegime::from_inertial_state(r, equatorial),
            Some(OrbitRegime::Geo)
        );
        assert_eq!(
            OrbitRegime::from_inertial_state(r, inclined),
            Some(OrbitRegime::Igso)
        );
        // The same state at 2^-520 of the speed falls to a perigee of 1e-5 m,
        // and at 2^20 of it is not bound at all.
        for exponent in [-520, 20] {
            let v = equatorial.map(|c| libm::scalbn(c, exponent));
            assert_eq!(OrbitRegime::from_inertial_state(r, v), None, "{exponent}");
        }
    }

    // The published weights are the average projections over the users that
    // see the satellite: spread uniformly by area over the cap above a 0 degree
    // elevation mask, `w_R^2` is the mean of `cos^2` of the nadir angle and
    // `w_AC^2` the mean of `sin^2 / 2`. Area on a sphere is uniform in the
    // cosine of the Earth-central angle, so the mean is a Simpson integral
    // over that cosine from `R / a` (the 0 degree horizon) to 1.
    #[test]
    fn sisre_weights_follow_from_the_orbit_radius() {
        use crate::astro::constants::earth::MEAN_EARTH_RADIUS_KM;
        let derived = |orbit_radius_km: f64| {
            let earth = MEAN_EARTH_RADIUS_KM;
            let horizon = earth / orbit_radius_km;
            let intervals = 2_000;
            let step = (1.0 - horizon) / intervals as f64;
            let mut mean_cos_sq = 0.0;
            for k in 0..=intervals {
                let cos_central = horizon + step * k as f64;
                let toward = orbit_radius_km - earth * cos_central;
                let distance_sq = orbit_radius_km * orbit_radius_km + earth * earth
                    - 2.0 * orbit_radius_km * earth * cos_central;
                let cos_sq_nadir = toward * toward / distance_sq;
                let simpson = if k == 0 || k == intervals {
                    1.0
                } else if k % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                mean_cos_sq += simpson * cos_sq_nadir;
            }
            mean_cos_sq *= step / 3.0 / (1.0 - horizon);
            (mean_cos_sq.sqrt(), (1.0 - mean_cos_sq) / 2.0)
        };
        let rounded = |value: f64, places: i32| {
            let scale = 10f64.powi(places);
            (value * scale).round() / scale
        };
        for (system, regime, orbit_radius_km) in [
            (GnssSystem::Glonass, OrbitRegime::Meo, 25_510.0),
            (GnssSystem::Gps, OrbitRegime::Meo, 26_560.0),
            (GnssSystem::BeiDou, OrbitRegime::Meo, 27_910.0),
            (GnssSystem::Galileo, OrbitRegime::Meo, 29_600.0),
            (GnssSystem::BeiDou, OrbitRegime::Igso, 42_164.0),
            (GnssSystem::BeiDou, OrbitRegime::Geo, 42_164.0),
        ] {
            let (radial, transverse_normal_squared) = derived(orbit_radius_km);
            let published = SisreWeights::montenbruck_2018(system, regime).expect("row");
            assert_eq!(
                rounded(radial, 2),
                published.radial(),
                "{system:?} {regime} w_R {radial}"
            );
            assert_eq!(
                (1.0 / transverse_normal_squared).round(),
                (1.0 / published.transverse_normal_squared()).round(),
                "{system:?} {regime} 1/w_AC^2 {}",
                1.0 / transverse_normal_squared
            );
        }
    }

    #[test]
    fn transport_velocity_of_the_earth_rotation() {
        let velocity = inertial_velocity_from_earth_fixed([7.0e6, 0.0, 1.0e6], [0.0, 0.0, 0.0]);
        assert_eq!(velocity, Ok([0.0, OMEGA_E_DOT_RAD_S * 7.0e6, 0.0]));
        let velocity = inertial_velocity_from_earth_fixed([0.0, 7.0e6, 0.0], [1.0, 2.0, 3.0]);
        assert_eq!(velocity, Ok([1.0 - OMEGA_E_DOT_RAD_S * 7.0e6, 2.0, 3.0]));
        for (position, velocity) in [
            ([f64::NAN, 0.0, 0.0], [0.0; 3]),
            ([7.0e6, 0.0, 0.0], [0.0, f64::INFINITY, 0.0]),
            ([1.0e308, 0.0, 0.0], [0.0, f64::MAX, 0.0]),
        ] {
            assert!(matches!(
                inertial_velocity_from_earth_fixed(position, velocity),
                Err(OrbitScoreError::InvalidInput { .. })
            ));
        }
    }

    #[test]
    fn trajectory_states_are_looked_up_exactly() {
        let origin = Instant::from_nanos(TimeScale::Gpst, 0);
        let sample = |seconds: i128, x: f64| TrajectorySample {
            satellite: gps(3),
            epoch: Instant::from_nanos(TimeScale::Gpst, seconds * 1_000_000_000),
            position_ecef_m: [x, 1.0e7, 1.0e7],
            clock_s: Some(1.0e-4),
        };
        let trajectory = OrbitTrajectory::new("IGS20", origin, [sample(60, 2.0), sample(0, 1.0)])
            .expect("trajectory");
        assert_eq!(
            trajectory
                .state_at_j2000_s(gps(3), 60.0)
                .expect("state")
                .position_ecef_m[0],
            2.0
        );
        assert!(matches!(
            trajectory.state_at_j2000_s(gps(3), 30.0),
            Err(Error::EpochOutOfRange)
        ));
        assert!(matches!(
            trajectory.state_at_j2000_s(gps(4), 0.0),
            Err(Error::UnknownSatellite(_))
        ));
        assert_eq!(trajectory.prediction_origin_j2000_s(gps(4)), Some(0.0));
        assert_eq!(
            trajectory.epochs(),
            vec![
                Instant::from_nanos(TimeScale::Gpst, 0),
                Instant::from_nanos(TimeScale::Gpst, 60_000_000_000)
            ]
        );
        assert!(!trajectory.clock_includes_relativity());
        assert!(trajectory
            .with_clock_relativity_included(true)
            .clock_includes_relativity());

        let utc_sample = TrajectorySample {
            epoch: Instant::from_nanos(TimeScale::Utc, 0),
            ..sample(0, 1.0)
        };
        assert!(matches!(
            OrbitTrajectory::new("IGS20", origin, [sample(0, 1.0), utc_sample]),
            Err(OrbitScoreError::InstantTimeSystemMismatch {
                field: "samples",
                index: 1,
                found: TimeScale::Utc,
                expected: TimeScale::Gpst,
            })
        ));
    }

    // An asserted label set names two distinct labels, none blank, as the merge
    // requires; a blank frame label matches nothing; a set label is matched
    // after trimming.
    #[test]
    fn frame_label_sets_and_blank_labels_are_refused() {
        use crate::sp3::Sp3FrameLabelSet;
        use std::collections::BTreeSet;
        let refused = |labels: &[&str]| {
            let mut options = OrbitScoreOptions::new(vec![0.0, 1.0]);
            options.frame_reconciliation.asserted_equivalent_label_sets = vec![Sp3FrameLabelSet {
                labels: labels.iter().map(|label| label.to_string()).collect(),
            }];
            matches!(
                validate_options(&options),
                Err(OrbitScoreError::InvalidInput {
                    field: "frame_reconciliation.asserted_equivalent_label_sets",
                    ..
                })
            )
        };
        assert!(refused(&["", "IGc20"]));
        assert!(refused(&["  ", "IGc20"]));
        assert!(refused(&["IGc20"]));
        assert!(refused(&[]));
        assert!(refused(&["IGS20", " IGS20 "]));
        assert!(!refused(&["IGS20", "IGc20"]));

        let disabled = Sp3FrameReconciliationOptions::disabled();
        for (source, reference) in [("", "IGS20"), ("IGS20", " "), ("", "")] {
            assert!(matches!(
                reconcile_frames(&OrbitFrame::Label(source.into()), reference, &disabled),
                Err(OrbitScoreError::FrameMismatch { .. })
            ));
        }

        let mut untrimmed = Sp3FrameReconciliationOptions::disabled();
        untrimmed.asserted_equivalent_label_sets = vec![Sp3FrameLabelSet {
            labels: BTreeSet::from([" IGb14 ".to_string(), "IGS20".to_string()]),
        }];
        assert_eq!(
            reconcile_frames(&OrbitFrame::Label("IGb14".into()), "IGS20", &untrimmed),
            Ok((FrameAgreement::AssertedEquivalent, FrameTreatment::AsStated))
        );
    }

    #[test]
    fn frame_labels_reconcile_only_as_asked() {
        let disabled = Sp3FrameReconciliationOptions::disabled();
        assert_eq!(
            reconcile_frames(&OrbitFrame::Label(" IGS20 ".into()), "IGS20", &disabled),
            Ok((FrameAgreement::SameLabel, FrameTreatment::AsStated))
        );
        assert_eq!(
            reconcile_frames(&OrbitFrame::BroadcastDatums, "IGS20", &disabled),
            Ok((FrameAgreement::BroadcastDatums, FrameTreatment::AsStated))
        );
        assert!(matches!(
            reconcile_frames(&OrbitFrame::Label("IGb14".into()), "IGS20", &disabled),
            Err(OrbitScoreError::FrameMismatch { .. })
        ));
        assert_eq!(
            reconcile_frames(
                &OrbitFrame::Label("IGb14".into()),
                "IGS20",
                &Sp3FrameReconciliationOptions::helmert()
            ),
            Ok((
                FrameAgreement::Helmert {
                    from: TerrestrialFrame::Itrf2014,
                    to: TerrestrialFrame::Itrf2020,
                },
                FrameTreatment::Helmert {
                    from: TerrestrialFrame::Itrf2014,
                    to: TerrestrialFrame::Itrf2020,
                }
            ))
        );
    }
}
