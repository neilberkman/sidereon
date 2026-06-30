//! Range / Sagnac substrate: recipe-selected Earth-rotation corrections.
//!
//! The geometric-range step needs different floating-point operation orders for
//! different references: SPP and the observable predictor rotate the transmit-time
//! satellite ECEF position with a closed-form Z rotation, while the RTK baseline
//! filter applies RTKLIB's first-order scalar Sagnac term to the range itself.
//! Both orders are already pinned in [`crate::geometry::range`]; this substrate is
//! the recipe-keyed entry the strategies route through, selecting the op-order by
//! [`SagnacRecipe`] value instead of each caller owning a copy of the choice.
//!
//! Routing a caller through here is behavior-preserving by construction: each
//! recipe arm delegates to the exact helper (and therefore the exact op-order)
//! the caller used before.

use crate::astro::math::vec3::{norm3, sub3};

use crate::estimation::recipe::SagnacRecipe;
use crate::geometry::range::{sagnac_range_first_order, sagnac_rotate_exact};

/// Apply the Sagnac (Earth-rotation) correction to a transmit-time satellite
/// ECEF position, selected by recipe.
///
/// [`SagnacRecipe::ClosedFormZRotation`] rotates the position by `+omega*tau`
/// about Z (the SPP / observable order). The first-order scalar and `Off`
/// recipes leave the satellite vector unrotated: the RTKLIB recipe corrects the
/// scalar range in [`geometric_range`] rather than rotating the vector, and
/// `Off` applies no correction.
#[inline]
pub(crate) fn rotate_transmit_satellite(
    sagnac: SagnacRecipe,
    pos: [f64; 3],
    tau_s: f64,
    omega_rad_s: f64,
) -> [f64; 3] {
    match sagnac {
        SagnacRecipe::ClosedFormZRotation => sagnac_rotate_exact(pos, tau_s, omega_rad_s),
        SagnacRecipe::RtklibFirstOrderScalar | SagnacRecipe::Off => pos,
    }
}

/// Geometric range between a transmit-time satellite position and a receiver,
/// selected by recipe.
///
/// [`SagnacRecipe::RtklibFirstOrderScalar`] adds RTKLIB's first-order scalar
/// Sagnac term to the Euclidean range (the RTK baseline order). The closed-form
/// and `Off` recipes return the plain Euclidean range: the closed-form recipe
/// rotates the satellite vector upstream via [`rotate_transmit_satellite`] and
/// computes its own post-rotation range, and `Off` applies no correction.
#[inline]
pub(crate) fn geometric_range(
    sagnac: SagnacRecipe,
    sat: [f64; 3],
    recv: [f64; 3],
    omega_rad_s: f64,
    c_m_s: f64,
) -> f64 {
    match sagnac {
        SagnacRecipe::RtklibFirstOrderScalar => {
            sagnac_range_first_order(sat, recv, omega_rad_s, c_m_s)
        }
        SagnacRecipe::ClosedFormZRotation | SagnacRecipe::Off => norm3(sub3(sat, recv)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAT: [f64; 3] = [15_600_000.0, -20_400_000.0, 9_800_000.0];
    const RECV: [f64; 3] = [4_027_894.0, 307_046.0, 4_919_474.0];
    const TAU_S: f64 = 0.072_345;
    const OMEGA: f64 = 7.292_115_146_7e-5;
    const C: f64 = 299_792_458.0;

    fn bits3(v: [f64; 3]) -> [u64; 3] {
        [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()]
    }

    #[test]
    fn closed_form_rotation_matches_geometry_helper_bits() {
        assert_eq!(
            bits3(rotate_transmit_satellite(
                SagnacRecipe::ClosedFormZRotation,
                SAT,
                TAU_S,
                OMEGA
            )),
            bits3(sagnac_rotate_exact(SAT, TAU_S, OMEGA))
        );
    }

    #[test]
    fn non_rotating_recipes_leave_satellite_vector_unchanged() {
        for recipe in [SagnacRecipe::RtklibFirstOrderScalar, SagnacRecipe::Off] {
            assert_eq!(
                bits3(rotate_transmit_satellite(recipe, SAT, TAU_S, OMEGA)),
                bits3(SAT)
            );
        }
    }

    #[test]
    fn first_order_range_matches_geometry_helper_bits() {
        assert_eq!(
            geometric_range(SagnacRecipe::RtklibFirstOrderScalar, SAT, RECV, OMEGA, C).to_bits(),
            sagnac_range_first_order(SAT, RECV, OMEGA, C).to_bits()
        );
    }

    #[test]
    fn non_scalar_recipes_return_plain_euclidean_range() {
        let euclid = norm3(sub3(SAT, RECV));
        for recipe in [SagnacRecipe::ClosedFormZRotation, SagnacRecipe::Off] {
            assert_eq!(
                geometric_range(recipe, SAT, RECV, OMEGA, C).to_bits(),
                euclid.to_bits()
            );
        }
    }
}
