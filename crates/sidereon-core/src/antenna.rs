//! Shared antenna-calibration interpolation kernels (ANTEX PCV/PCO).
//!
//! ANTEX phase-center variation (PCV) grids are interpolated the same way
//! wherever they are consumed: clamp-and-linear in zenith, and a wrap-aware
//! linear blend between the two bracketing azimuth nodes. Before this module the
//! arithmetic was copy-pasted across the ANTEX reader, the PPP correction
//! tables, the static PPP solver, and the sequential RTK filter, each wrapped in
//! a different container (`Result`/`Option`/`f64`, owned `Vec`/borrowed slice/
//! reused scratch buffer). The kernels here own the floating-point operation
//! order so every caller stays bit-identical; the caller keeps only its own
//! sample extraction, sort comparator, and empty-grid policy.
//!
//! Callers must pass a slice already sorted ascending by zenith to
//! [`interpolate_zenith_sorted`]; the sort comparator stays caller-side because
//! it is part of each caller's frozen behavior.

use crate::astro::constants::units::DEGREES_PER_CIRCLE;

/// Linearly interpolate a PCV value from a zenith-sorted `(zenith_deg, value_m)`
/// grid, clamping to the first node below the grid and the last node above it.
///
/// `sorted` must be ascending by zenith. Returns `None` only when the grid is
/// empty, leaving the empty-grid policy (error, zero, skip) to the caller.
pub(crate) fn interpolate_zenith_sorted(sorted: &[(f64, f64)], zenith_deg: f64) -> Option<f64> {
    let first = *sorted.first()?;
    let last = *sorted.last()?;

    if zenith_deg <= first.0 {
        return Some(first.1);
    }
    if zenith_deg >= last.0 {
        return Some(last.1);
    }

    let low = sorted
        .iter()
        .rev()
        .find(|&&(z, _)| z <= zenith_deg)
        .copied()?;
    let high = sorted.iter().find(|&&(z, _)| z >= zenith_deg).copied()?;

    if high.0 == low.0 {
        Some(low.1)
    } else {
        let t = (zenith_deg - low.0) / (high.0 - low.0);
        Some(low.1 + (high.1 - low.1) * t)
    }
}

/// Fold an azimuth in degrees into `[0, 360)`.
pub(crate) fn normalize_azimuth(azimuth_deg: f64) -> f64 {
    let wrapped = azimuth_deg % DEGREES_PER_CIRCLE;
    if wrapped < 0.0 {
        wrapped + DEGREES_PER_CIRCLE
    } else {
        wrapped
    }
}

/// Bracket a normalized azimuth between the nearest grid nodes at or below and
/// at or above it. When the azimuth falls outside the grid (only the wrap gap
/// remains) the bracket is the last and first nodes, which the blend closes
/// across the 360-degree seam.
pub(crate) fn azimuth_bracket(azimuths: &[f64], azimuth: f64) -> (f64, f64) {
    let first = azimuths[0];
    let last = azimuths[azimuths.len() - 1];
    let low = azimuths.iter().rev().find(|&&a| a <= azimuth).copied();
    let high = azimuths.iter().find(|&&a| a >= azimuth).copied();

    match (low, high) {
        (Some(low), Some(high)) => (low, high),
        _ => (last, first),
    }
}

/// Wrap-aware linear blend between two azimuth nodes whose PCV has already been
/// interpolated in zenith. `low_deg`/`high_deg` are the bracket from
/// [`azimuth_bracket`] and `azimuth` is the normalized target; when the bracket
/// straddles the 360-degree seam the high node and target are unwrapped by one
/// full turn so the fraction is monotone.
pub(crate) fn blend_azimuth(
    low_deg: f64,
    high_deg: f64,
    azimuth: f64,
    low_value: f64,
    high_value: f64,
) -> f64 {
    let low = low_deg;
    let high = if high_deg < low {
        high_deg + DEGREES_PER_CIRCLE
    } else {
        high_deg
    };
    let target = if azimuth < low {
        azimuth + DEGREES_PER_CIRCLE
    } else {
        azimuth
    };

    if high == low {
        low_value
    } else {
        let t = (target - low) / (high - low);
        low_value + (high_value - low_value) * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zenith_clamps_below_and_above_the_grid() {
        let grid = [(0.0, 1.0), (5.0, 2.0), (10.0, 4.0)];
        assert_eq!(interpolate_zenith_sorted(&grid, -1.0), Some(1.0));
        assert_eq!(interpolate_zenith_sorted(&grid, 11.0), Some(4.0));
    }

    #[test]
    fn zenith_interpolates_interior_with_frozen_bits() {
        let grid = [(0.0, 1.0), (5.0, 2.0), (10.0, 4.0)];
        // 2.0 + (4.0 - 2.0) * ((7.0 - 5.0) / (10.0 - 5.0))
        let value = interpolate_zenith_sorted(&grid, 7.0).expect("interior");
        assert_eq!(value.to_bits(), (2.8_f64).to_bits());
    }

    #[test]
    fn zenith_empty_grid_is_none() {
        assert_eq!(interpolate_zenith_sorted(&[], 5.0), None);
    }

    #[test]
    fn azimuth_normalizes_into_unit_circle() {
        assert_eq!(normalize_azimuth(-10.0).to_bits(), (350.0_f64).to_bits());
        assert_eq!(normalize_azimuth(370.0).to_bits(), (10.0_f64).to_bits());
    }

    #[test]
    fn azimuth_blend_interpolates_and_wraps_seam() {
        // Interior: 1.0 + (3.0 - 1.0) * ((45.0 - 0.0) / (90.0 - 0.0)).
        let interior = blend_azimuth(0.0, 90.0, 45.0, 1.0, 3.0);
        assert_eq!(interior.to_bits(), (2.0_f64).to_bits());

        // Seam: bracket (270, 0) wraps to span [270, 360); target 315 -> t = 0.5.
        let seam = blend_azimuth(270.0, 0.0, 315.0, 2.0, 4.0);
        assert_eq!(seam.to_bits(), (3.0_f64).to_bits());
    }

    #[test]
    fn azimuth_bracket_handles_interior_and_wrap_gap() {
        let azimuths = [0.0, 90.0, 180.0, 270.0];
        assert_eq!(azimuth_bracket(&azimuths, 135.0), (90.0, 180.0));
        // Above the last node only the wrap gap remains: (last, first).
        assert_eq!(azimuth_bracket(&azimuths, 315.0), (270.0, 0.0));
    }
}
