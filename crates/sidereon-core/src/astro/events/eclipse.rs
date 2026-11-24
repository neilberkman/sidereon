//! Conical Earth and Moon shadow (eclipse) geometry.
//!
//! Determines whether a satellite is in full sunlight, penumbra, or umbra.
//!
//! Two formulations are provided:
//!
//! - [`shadow_fraction`] and [`shadow_fraction_with_model`] evaluate Earth-only
//!   conical shadow in `[0.0, 1.0]` using linear penumbra/antumbra
//!   interpolation between the umbra and penumbra cone radii in a single shared
//!   exponent-separated evaluation (shared with
//!   [`shadow_fraction_with_gradient`],
//!   [`shadow_fraction_with_model_and_gradient`], and the conical lighting
//!   ratio and gradient of `ShadowFunction::Conical`). The cross product
//!   `c = sun x sat` is formed with the compensated scaled cross product (with
//!   polar scaling `k = 1 / (1 - WGS84_FLATTENING)` factored out of the exact
//!   products under [`EarthShadowModel::Wgs84Oblate`]), so `c` is zero iff the
//!   inputs are exactly collinear; `rho = |c| / |sun|`,
//!   `u_perp = normalize(c x sun)`, and `proj = (sat . sun) / |sun|` are
//!   evaluated in the scaled frame. Both `shadow` and the lighting ratio `nu`
//!   are evaluated directly from `rho` (`shadow = (r_pen - rho) / (r_pen - r_umb)`
//!   and `nu = (rho - r_umb) / (r_pen - r_umb)` in penumbra;
//!   `shadow = (r_pen - rho) / r_pen` and `nu = rho / r_pen` in antumbra), and
//!   `d(nu) = -d(shadow)` is formed with the quotient rule of whichever of `nu`
//!   or `shadow` is at most `1/2`. Orekit has no linear conical counterpart.
//! - [`moon_shadow_fraction`], [`shadow_fraction_with_moon`], and
//!   [`shadow_fraction_with_model_and_moon`] port Orekit 13.1.10's
//!   `OccultationEngine` (`angles`, `Ellipsoid.pointOnLimb`, `Vector3D.angle`)
//!   and `SolarRadiationPressure` (`maskingRatio` and `getLightingRatio`, with
//!   `ANGULAR_MARGIN = 1e-10`, `Precision.SAFE_MIN` / `FastMath.nextDown(1.0)`
//!   clamps, inside-Sun dummy angles `(PI, 0, 0)`, and Hipparchus 4.0.3
//!   `FastMath` / `MathArrays.linearCombination`), using
//!   [`IAU_2015_SOLAR_RADIUS_KM`](crate::astro::constants::astro::IAU_2015_SOLAR_RADIUS_KM),
//!   [`MEAN_EARTH_RADIUS_KM`], [`WGS84_FLATTENING`], and
//!   [`MOON_RADIUS_KM`](crate::astro::constants::astro::MOON_RADIUS_KM).
//!   Because Orekit's `maskingRatio` has no outer `[0, 1]` clamp, both
//!   single-body and combined two-body apparent-disc shadow fractions can
//!   overshoot `[0.0, 1.0]` just outside `ANGULAR_MARGIN` (up to `1.435e-6`
//!   measured over the oracle fixtures and property sweeps, `1.423e-6` on the
//!   144-case Orekit oracle): on these grazing cases `alpha2 / alphaCentral` is
//!   near `±1` (not clamped), so rounding in `alpha2 / alphaCentral` is
//!   amplified by `acos` near `±1` and does not cancel against
//!   `sqrt(aE2 - alpha2^2)`, which has lost most of its digits to cancellation.
//!   The `±nextDown(1.0)` clamp acts on annular and small-disc cases (such as
//!   the `8.408e-10` annular Moon difference against the 40-digit Montenbruck &
//!   Gill (§3.4.2) apparent-disc reference; largest Earth difference
//!   `1.163e-9`).
//!
//! Documented departures from Orekit 13.1.10:
//!
//! - **Simultaneous Earth-Moon overlap:** Orekit 13.1.10's `getLightingRatio`
//!   subtracts the unclipped Earth-Moon mutual overlap `|E ∩ M| / |S|` rather
//!   than adding back the triple intersection `|S ∩ E ∩ M| / |S|` and does not
//!   check whether Earth occults the Sun, producing negative lighting ratios
//!   (down to `-10.0527` on the 207-case fixture). Here `getLightingRatio`
//!   follows Orekit's exact operation order (`0.0 + l_earth`, mutual slot when
//!   `l_moon != 1`, `+ l_moon - 1.0`, no outer `[0, 1]` clamp) with only the
//!   mutual-eclipse term replaced by adding `triple_area / (pi * alpha_sun^2)`.
//! - **Occulting body behind the Sun:** Orekit does not check whether the
//!   occulting body lies in front of the Sun; here, whenever Orekit's own
//!   `OccultationEngine.angles` computes real angles (i.e. outside the Sun), a
//!   body at distance `d_body >= d_sun` from the satellite casts no shadow.
//!   Satellites inside the Sun go through Orekit's dummy angles `(PI, 0, 0)`
//!   and `maskingRatio`.
//! - **Sun-centred frame guard:** Orekit's `getLightingRatio` returns lighting
//!   `1.0` whenever `|r_sun| < 2 * R_sun` as a Sun-centred frame check. Because
//!   this crate's positions are relative to Earth's centre (or arbitrary
//!   inertial origin for `moon_shadow_fraction`), that test does not identify a
//!   Sun-centred frame and the geometry is evaluated instead.
//! - **Exact collinearity (including underflow and frame noise):** Orekit's
//!   `Ellipsoid.pointOnLimb` calls `cross(psat, pted).orthogonal()`, which
//!   throws `ZERO_NORM` whenever the cross product's norm is zero—either
//!   because the vectors are exactly collinear or because the norm underflows
//!   to `0.0`—and its double `maskingRatio` divides by `0.0` (`NaN`) at zero
//!   separation in annular eclipse. Here `pointOnLimb` falls back to
//!   `psat.orthogonal()` when `|cross(psat, pted)| == 0.0`, and
//!   double-precision annular collinearity evaluates `1 - (ro / rs)^2`, while
//!   forward-mode field evaluation on-axis (or when the cross-product norm
//!   underflows onto `angle = 0.0`) produces `NaN` and returns
//!   [`ShadowGradientError::NumericalFailure`]. In
//!   [`crate::astro::forces::ecom`], exact exponent-separated collinearity and
//!   orthogonality predicates (`sun x sat == 0` and, with `r x v != 0`,
//!   `r . sun == 0 && v . sun == 0`) similarly guard the ECOM2 fast path so
//!   rounding noise in `e_D x sat` or `Z x sun` never produces a finite frame
//!   where the geometry is mathematically undefined.
//! - **Overflowing vector operations and distant-body translation cancellation:**
//!   Hipparchus 4.0.3's `Vector3D.normalize`, `FieldVector3D.normalize`,
//!   `orthogonal`, and `angle` return zero or `NaN` when a finite vector's
//!   squared norm or cross/dot product overflows to `+Inf`. Whenever the
//!   unscaled real value is finite and nonzero, `normalize`, `orthogonal`, and
//!   `angle` return Hipparchus's exact result (including non-finite gradient
//!   entries). Only when the unscaled norm or product overflows while every
//!   component value is finite are the components prescaled by an exact power of
//!   two before evaluating, leaving every input with a finite nonzero unscaled
//!   norm bit-identical. Likewise, in `pointOnLimb`, overflowing
//!   `observer × outside` or limb-distance comparisons fall back to scaled
//!   arithmetic, `getPlaneSection` falls back to `tan(theta) = 0` (with zero
//!   gradient in the field path, Orekit's own `|gamma| < SAFE_MIN` branch)
//!   when Orekit's `tan(theta)` value or any derivative entry is non-finite
//!   (where `|beta - alpha|` and `|gamma|` are below about `2^-512`, making
//!   the section circular far below binary64 resolution), and when translating
//!   both satellite and Sun into a distant Moon's frame cancels
//!   `(sat - moon) - (sun - moon)` to zero while `sat != sun`, `sat - sun` is
//!   used for `ps`.
//! - **Non-finite field evaluation:** Where Orekit's `Field<Gradient>`
//!   evaluation produces `NaN` or an infinity, the gradient functions return
//!   [`ShadowGradientError::NumericalFailure`].
//!
//! Each shadow fraction has a counterpart returning its forward-mode derivative
//! with respect to the satellite position ([`shadow_fraction_with_gradient`],
//! [`shadow_fraction_with_model_and_gradient`],
//! [`shadow_fraction_with_model_moon_and_gradient`]). The conical gradient
//! functions share the exponent-separated conical geometry evaluation with
//! [`shadow_fraction`] and [`shadow_fraction_with_model`] (returning
//! bit-identical values and [`ShadowGradientError::NumericalFailure`] on the
//! exact antumbra axis); the Earth-Moon apparent-disc gradient function
//! reproduces Hipparchus 4.0.3 `Gradient` arithmetic and Orekit 13.1.10's
//! `FieldSpacecraftState` branch decisions.

use crate::astro::constants::earth::WGS84_F;
use crate::astro::math::vec3;

pub(crate) mod ad;
pub(crate) mod fast_math;
mod gradient;
pub(crate) mod orekit;

pub use gradient::{
    shadow_fraction_with_gradient, shadow_fraction_with_model_and_gradient,
    shadow_fraction_with_model_moon_and_gradient, ShadowGradientError,
};

pub(crate) use gradient::{
    conical_lighting_ratio_double_scaled, conical_lighting_ratio_field_scaled,
};
pub(crate) use orekit::{
    earth_lighting_ratio_double, earth_lighting_ratio_field, earth_moon_lighting_ratio_double,
    earth_moon_lighting_ratio_field,
};

/// Public eclipse alias for WGS84 flattening `1 / 298.257223563`.
pub const WGS84_FLATTENING: f64 = WGS84_F;

/// Earth figure used for conical eclipse geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarthShadowModel {
    /// Spherical Earth with [`MEAN_EARTH_RADIUS_KM`].
    Spherical,
    /// WGS84 oblate Earth model.
    ///
    /// In the Earth-only conical functions ([`shadow_fraction_with_model`],
    /// [`status_with_model`]), the z components of the satellite and Sun
    /// vectors are scaled by `1 / (1 - WGS84_FLATTENING)` before evaluating
    /// the conical shadow geometry.
    ///
    /// In the apparent-disc functions ([`shadow_fraction_with_model_and_moon`],
    /// [`status_with_model_and_moon`]), Earth is modeled as a `OneAxisEllipsoid`
    /// with equatorial radius [`MEAN_EARTH_RADIUS_KM`] and flattening
    /// [`WGS84_FLATTENING`] whose limb is computed by Orekit's `pointOnLimb`.
    Wgs84Oblate,
}

/// Shadow function used to compute the solar lighting ratio `nu` for solar
/// radiation pressure force models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShadowFunction {
    /// Apparent-disc occultation lighting ratio porting Orekit 13.1.10's
    /// `OccultationEngine` and `SolarRadiationPressure.getLightingRatio`,
    /// using the selected [`EarthShadowModel`] and, when lunar occultation is
    /// enabled, the Moon as a second occulting body.
    ApparentDisc,
    /// Earth-only linear conical lighting ratio using the selected
    /// [`EarthShadowModel`], sharing the exponent-separated conical geometry
    /// with [`shadow_fraction_with_model`] and evaluating
    /// `nu = (rho - r_umbra) / (r_penumbra - r_umbra)` in penumbra and
    /// `nu = rho / r_penumbra` in antumbra directly from `rho` (rather than as
    /// `1 - shadow_fraction_with_model`, which loses precision near the umbra
    /// boundary and antumbra axis).
    Conical,
}

/// Illumination state of a satellite relative to Earth's conical shadow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EclipseStatus {
    /// Full sunlight.
    Sunlit,
    /// Partial shadow between the umbra and penumbra cones.
    Penumbra,
    /// Full shadow inside the umbra cone.
    Umbra,
}

/// Error while computing conical eclipse geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EclipseError {
    #[error("invalid eclipse input {field}: {reason}")]
    /// A satellite, Sun, or Moon position failed validation before eclipse
    /// geometry was evaluated.
    InvalidInput {
        /// Name of the position argument that failed validation: `"sat_pos"`,
        /// `"sun_pos"`, or `"moon_pos"`.
        field: &'static str,
        /// Validation failure: `"not finite"`, `"zero vector"`, or
        /// `"out of range"`.
        reason: &'static str,
    },
}

/// Shadow fraction in `[0.0, 1.0]`: `0.0` is full sunlight, `1.0` is full
/// umbra, intermediate values are penumbra.
///
/// `sat_pos` is the satellite GCRS position (km); `sun_pos` is the vector from
/// Earth center to the Sun (km).
pub fn shadow_fraction(sat_pos: [f64; 3], sun_pos: [f64; 3]) -> Result<f64, EclipseError> {
    gradient::conical_shadow_value(sat_pos, sun_pos, EarthShadowModel::Spherical)
}

/// Shadow fraction with an explicit Earth figure model.
///
/// [`EarthShadowModel::Spherical`] delegates to the same shared conical
/// geometry evaluation as [`shadow_fraction`] and is bit-identical to it.
/// [`EarthShadowModel::Wgs84Oblate`] evaluates the conical geometry in the
/// WGS84 polar-scaled frame.
pub fn shadow_fraction_with_model(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    model: EarthShadowModel,
) -> Result<f64, EclipseError> {
    gradient::conical_shadow_value(sat_pos, sun_pos, model)
}

/// Classify the satellite's eclipse status from its [`shadow_fraction`].
pub fn status(sat_pos: [f64; 3], sun_pos: [f64; 3]) -> Result<EclipseStatus, EclipseError> {
    let fraction = shadow_fraction(sat_pos, sun_pos)?;
    if fraction >= 1.0 {
        Ok(EclipseStatus::Umbra)
    } else if fraction > 0.0 {
        Ok(EclipseStatus::Penumbra)
    } else {
        Ok(EclipseStatus::Sunlit)
    }
}

/// Classify eclipse status with an explicit Earth figure model.
pub fn status_with_model(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    model: EarthShadowModel,
) -> Result<EclipseStatus, EclipseError> {
    let fraction = shadow_fraction_with_model(sat_pos, sun_pos, model)?;
    Ok(classify_fraction(fraction))
}

/// Shadow fraction cast by the Moon alone, porting Orekit 13.1.10's
/// `OccultationEngine` and `SolarRadiationPressure.maskingRatio`.
///
/// Nominally in `[0.0, 1.0]` (`0.0` full sunlight, `1.0` full umbra), but
/// unclamped like Orekit: just outside `ANGULAR_MARGIN = 1e-10`, rounding in
/// `alpha2 / alphaCentral` is amplified by `acos` near `±1` (unclamped) and
/// does not cancel against `sqrt(aE2 - alpha2^2)` after cancellation in
/// `aE2 - alpha2^2`, which can overshoot `[0.0, 1.0]` (up to `1.435e-6`
/// measured across the oracle fixtures and property sweeps).
///
/// `sat_pos`, `sun_pos`, and `moon_pos` are inertial positions in the same
/// frame (km). Uses [`IAU_2015_SOLAR_RADIUS_KM`](crate::astro::constants::astro::IAU_2015_SOLAR_RADIUS_KM)
/// and [`MOON_RADIUS_KM`](crate::astro::constants::astro::MOON_RADIUS_KM).
/// Returns `0.0` in full sunlight, when the Moon is not closer to the observer
/// than the Sun, or when the observer is inside the Sun (not at its center).
pub fn moon_shadow_fraction(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    moon_pos: [f64; 3],
) -> Result<f64, EclipseError> {
    let lighting = orekit::moon_lighting_ratio_double(sat_pos, sun_pos, moon_pos)?;
    Ok(1.0 - lighting)
}

/// Combined spherical-Earth and Moon apparent-disc shadow fraction.
///
/// Nominally in `[0.0, 1.0]`, but unclamped like Orekit (can overshoot
/// `[0.0, 1.0]` by up to `1.435e-6` measured across the oracle fixtures and
/// property sweeps; see [`shadow_fraction_with_model_and_moon`]).
///
/// Delegates to [`shadow_fraction_with_model_and_moon`] with
/// [`EarthShadowModel::Spherical`].
pub fn shadow_fraction_with_moon(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    moon_pos: [f64; 3],
) -> Result<f64, EclipseError> {
    shadow_fraction_with_model_and_moon(sat_pos, sun_pos, moon_pos, EarthShadowModel::Spherical)
}

/// Combined Earth and Moon apparent-disc shadow fraction with an explicit
/// Earth figure model.
///
/// Ports Orekit 13.1.10's `OccultationEngine.angles`, `Ellipsoid.pointOnLimb`,
/// `SolarRadiationPressure.maskingRatio`, and `getLightingRatio` operation
/// order for Earth and the Moon, replacing only Orekit's defective
/// mutual-eclipse term with adding the exact three-disc intersection area
/// `triple_area / (pi * alpha_sun^2)`.
///
/// Nominally in `[0.0, 1.0]`, but unclamped like Orekit: just outside
/// `ANGULAR_MARGIN = 1e-10`, rounding in `alpha2 / alphaCentral` is amplified
/// by `acos` near `±1` (unclamped) and does not cancel against
/// `sqrt(aE2 - alpha2^2)` after cancellation in `aE2 - alpha2^2`, which can
/// overshoot `[0.0, 1.0]` in both single-body and two-body evaluations (up to
/// `1.435e-6` measured across the oracle fixtures and property sweeps).
///
/// `sat_pos`, `sun_pos`, and `moon_pos` are Earth-centered inertial positions
/// (km). Returns [`EclipseError::InvalidInput`] if any vector is non-finite,
/// zero, overflows its Euclidean norm, or places the satellite on or inside
/// Earth or the Moon or at the center of the Sun.
pub fn shadow_fraction_with_model_and_moon(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    moon_pos: [f64; 3],
    model: EarthShadowModel,
) -> Result<f64, EclipseError> {
    let lighting = orekit::earth_moon_lighting_ratio_double(sat_pos, sun_pos, moon_pos, model)?;
    Ok(1.0 - lighting)
}

/// Classify eclipse status with spherical Earth and Moon occultation.
pub fn status_with_moon(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    moon_pos: [f64; 3],
) -> Result<EclipseStatus, EclipseError> {
    let fraction = shadow_fraction_with_moon(sat_pos, sun_pos, moon_pos)?;
    Ok(classify_fraction(fraction))
}

/// Classify eclipse status with an explicit Earth figure model and Moon
/// occultation.
pub fn status_with_model_and_moon(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    moon_pos: [f64; 3],
    model: EarthShadowModel,
) -> Result<EclipseStatus, EclipseError> {
    let fraction = shadow_fraction_with_model_and_moon(sat_pos, sun_pos, moon_pos, model)?;
    Ok(classify_fraction(fraction))
}

fn classify_fraction(fraction: f64) -> EclipseStatus {
    if fraction >= 1.0 {
        EclipseStatus::Umbra
    } else if fraction > 0.0 {
        EclipseStatus::Penumbra
    } else {
        EclipseStatus::Sunlit
    }
}

pub(crate) fn scale_polar_axis(v: [f64; 3]) -> [f64; 3] {
    let z_scale = 1.0 / (1.0 - WGS84_FLATTENING);
    [v[0], v[1], v[2] * z_scale]
}

pub(crate) fn validate_nonzero_vec3(v: [f64; 3], field: &'static str) -> Result<(), EclipseError> {
    if !v.iter().all(|value| value.is_finite()) {
        return Err(invalid_eclipse_input(field, "not finite"));
    }
    let norm = vec3::norm3(v);
    if norm == 0.0 {
        return Err(invalid_eclipse_input(field, "zero vector"));
    }
    if !norm.is_finite() {
        return Err(invalid_eclipse_input(field, "out of range"));
    }
    Ok(())
}

pub(crate) fn invalid_eclipse_input(field: &'static str, reason: &'static str) -> EclipseError {
    EclipseError::InvalidInput { field, reason }
}

#[cfg(test)]
mod tests {
    use crate::astro::constants::{
        astro::{IAU_2015_SOLAR_RADIUS_KM, MOON_RADIUS_KM, SOLAR_RADIUS_KM},
        earth::MEAN_EARTH_RADIUS_KM,
    };
    use crate::astro::events::{CrossingDirection, EventFinder};

    use super::*;

    const AU_KM: f64 = 149_597_870.7;

    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn unit(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
        }
        fn log_uniform(&mut self, lo_exp: f64, hi_exp: f64) -> f64 {
            libm::pow(10.0, lo_exp + (hi_exp - lo_exp) * self.unit())
        }
        fn unit_vec3(&mut self) -> [f64; 3] {
            let z = 2.0 * self.unit() - 1.0;
            let phi = std::f64::consts::TAU * self.unit();
            let r = (1.0 - z * z).max(0.0).sqrt();
            let (sin_p, cos_p) = libm::sincos(phi);
            [r * cos_p, r * sin_p, z]
        }
    }

    #[test]
    fn shadow_fraction_matches_reference_bits() {
        let sun = [AU_KM, 0.0, 0.0];

        assert_eq!(
            shadow_fraction([7000.0, 0.0, 0.0], sun)
                .expect("valid eclipse geometry")
                .to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            shadow_fraction([0.0, 7000.0, 0.0], sun)
                .expect("valid eclipse geometry")
                .to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            shadow_fraction([-7000.0, 0.0, 0.0], sun)
                .expect("valid eclipse geometry")
                .to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(
            shadow_fraction([-7000.0, 6370.0, 0.0], sun)
                .expect("valid eclipse geometry")
                .to_bits(),
            0x3fe0_a32f_08e7_fb1f
        );
        assert_eq!(
            shadow_fraction([-7000.0, 6410.0, 0.0], sun)
                .expect("valid eclipse geometry")
                .to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            shadow_fraction([-7000.0, 6330.0, 0.0], sun)
                .expect("valid eclipse geometry")
                .to_bits(),
            1.0_f64.to_bits()
        );
    }

    #[test]
    fn oblate_shadow_model_preserves_equatorial_bits_and_changes_polar_grazing() {
        assert_eq!(WGS84_FLATTENING.to_bits(), 0x3f6b_775a_84f3_e128);

        let sun = [AU_KM, 0.0, 0.0];
        for sat in [
            [7000.0, 0.0, 0.0],
            [-7000.0, 0.0, 0.0],
            [-7000.0, 6370.0, 0.0],
        ] {
            let default = shadow_fraction(sat, sun).expect("valid eclipse geometry");
            let spherical = shadow_fraction_with_model(sat, sun, EarthShadowModel::Spherical)
                .expect("valid spherical geometry");
            let oblate = shadow_fraction_with_model(sat, sun, EarthShadowModel::Wgs84Oblate)
                .expect("valid oblate geometry");
            assert_eq!(spherical.to_bits(), default.to_bits());
            assert_eq!(oblate.to_bits(), default.to_bits());
        }

        let polar_grazing = [-7000.0, 0.0, 6370.0];
        let spherical = shadow_fraction_with_model(polar_grazing, sun, EarthShadowModel::Spherical)
            .expect("valid spherical geometry");
        let oblate = shadow_fraction_with_model(polar_grazing, sun, EarthShadowModel::Wgs84Oblate)
            .expect("valid oblate geometry");
        assert_eq!(spherical.to_bits(), 0x3fe0_a32f_08e7_fb1f);
        assert_eq!(oblate.to_bits(), 0x3fc8_7576_4827_2b98);
        assert!(
            oblate < spherical,
            "polar-axis flattening must reduce the grazing shadow fraction"
        );
    }

    #[test]
    fn status_classifies_cones() {
        let sun = [AU_KM, 0.0, 0.0];

        assert_eq!(
            status([7000.0, 0.0, 0.0], sun).expect("valid eclipse geometry"),
            EclipseStatus::Sunlit
        );
        assert_eq!(
            status([0.0, 7000.0, 0.0], sun).expect("valid eclipse geometry"),
            EclipseStatus::Sunlit
        );
        assert_eq!(
            status([-7000.0, 0.0, 0.0], sun).expect("valid eclipse geometry"),
            EclipseStatus::Umbra
        );
        assert_eq!(
            status([-7000.0, 6370.0, 0.0], sun).expect("valid eclipse geometry"),
            EclipseStatus::Penumbra
        );
    }

    #[test]
    fn shadow_fraction_increases_toward_axis() {
        let sun = [AU_KM, 0.0, 0.0];
        let outside = shadow_fraction([-7000.0, 6410.0, 0.0], sun).expect("valid eclipse geometry");
        let penumbra =
            shadow_fraction([-7000.0, 6370.0, 0.0], sun).expect("valid eclipse geometry");
        let umbra = shadow_fraction([-7000.0, 6330.0, 0.0], sun).expect("valid eclipse geometry");
        let center = shadow_fraction([-7000.0, 0.0, 0.0], sun).expect("valid eclipse geometry");

        assert_eq!(outside, 0.0);
        assert!(penumbra > 0.0 && penumbra < 1.0);
        assert_eq!(umbra, 1.0);
        assert_eq!(center, 1.0);
    }

    #[test]
    fn event_finder_refines_eclipse_fraction_crossings() {
        let sun = [AU_KM, 0.0, 0.0];
        let x_km = -7000.0;
        let start_y_km = -7000.0;
        let end_y_km = 7000.0;
        let duration_seconds = 1200.0;
        let threshold = 0.5;
        let y_at = |time_seconds: f64| {
            start_y_km + (end_y_km - start_y_km) * time_seconds / duration_seconds
        };
        let fraction_at = |time_seconds: f64| {
            shadow_fraction([x_km, y_at(time_seconds), 0.0], sun)
                .expect("valid synthetic eclipse geometry")
        };

        let events = EventFinder::new(0.0, duration_seconds, 60.0, 1.0e-7)
            .expect("valid eclipse finder")
            .find_crossings(fraction_at, threshold)
            .expect("finite eclipse predicate");

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].direction, CrossingDirection::Rising);
        assert_eq!(events[1].direction, CrossingDirection::Falling);
        assert_eq!(events[0].threshold, threshold);
        assert_eq!(events[1].threshold, threshold);

        let half_shadow_radius = shadow_radius_for_fraction(threshold, -x_km);
        let expected_enter =
            time_for_y(-half_shadow_radius, start_y_km, end_y_km, duration_seconds);
        let expected_exit = time_for_y(half_shadow_radius, start_y_km, end_y_km, duration_seconds);

        assert_close(events[0].time_seconds, expected_enter, 1.0e-6);
        assert_close(events[1].time_seconds, expected_exit, 1.0e-6);
        assert_close(events[0].value, threshold, 1.0e-7);
        assert_close(events[1].value, threshold, 1.0e-7);
        assert!(fraction_at(0.0) < threshold);
        assert!(fraction_at(duration_seconds * 0.5) > threshold);
        assert!(fraction_at(duration_seconds) < threshold);
    }

    fn shadow_radius_for_fraction(fraction: f64, dist_behind_km: f64) -> f64 {
        let alpha_umbra = libm::asin((SOLAR_RADIUS_KM - MEAN_EARTH_RADIUS_KM) / AU_KM);
        let alpha_penumbra = libm::asin((SOLAR_RADIUS_KM + MEAN_EARTH_RADIUS_KM) / AU_KM);
        let r_umbra = MEAN_EARTH_RADIUS_KM - dist_behind_km * libm::tan(alpha_umbra);
        let r_penumbra = MEAN_EARTH_RADIUS_KM + dist_behind_km * libm::tan(alpha_penumbra);
        r_penumbra - fraction * (r_penumbra - r_umbra)
    }

    fn time_for_y(y_km: f64, start_y_km: f64, end_y_km: f64, duration_seconds: f64) -> f64 {
        (y_km - start_y_km) * duration_seconds / (end_y_km - start_y_km)
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} differs from {expected} by more than {tolerance}"
        );
    }

    #[test]
    fn eclipse_helpers_reject_invalid_vectors() {
        assert_invalid_eclipse_field(
            shadow_fraction([0.0, 0.0, 0.0], [AU_KM, 0.0, 0.0]).unwrap_err(),
            "sat_pos",
            "zero vector",
        );
        assert_invalid_eclipse_field(
            status([7000.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0]).unwrap_err(),
            "sun_pos",
            "not finite",
        );
    }

    #[test]
    fn moon_eclipse_helpers_cover_clear_partial_annular_total_and_errors() {
        let sat = [7_000.0, 0.0, 0.0];
        let sun = [AU_KM, 0.0, 0.0];

        // Clear: Moon far off-axis.
        let moon_clear = [0.0, 384_400.0, 0.0];
        assert_eq!(
            moon_shadow_fraction(sat, sun, moon_clear).expect("clear moon"),
            0.0
        );
        assert_eq!(
            shadow_fraction_with_moon(sat, sun, moon_clear).expect("clear combined"),
            0.0
        );
        assert_eq!(
            status_with_moon(sat, sun, moon_clear).expect("clear status"),
            EclipseStatus::Sunlit
        );

        // Total lunar umbra: Moon at perigee directly between sat and Sun.
        let moon_perigee = [sat[0] + 356_000.0, 0.0, 0.0];
        assert_eq!(
            moon_shadow_fraction(sat, sun, moon_perigee).expect("total moon"),
            1.0
        );
        assert_eq!(
            shadow_fraction_with_moon(sat, sun, moon_perigee).expect("total combined"),
            1.0
        );
        assert_eq!(
            status_with_moon(sat, sun, moon_perigee).expect("total status"),
            EclipseStatus::Umbra
        );

        // Annular lunar eclipse: Moon at apogee directly between sat and Sun.
        let d_moon = 405_000.0;
        let moon_apogee = [sat[0] + d_moon, 0.0, 0.0];
        let annular = moon_shadow_fraction(sat, sun, moon_apogee).expect("annular moon");
        let a = libm::asin(IAU_2015_SOLAR_RADIUS_KM / (AU_KM - sat[0]));
        let b = libm::asin(MOON_RADIUS_KM / d_moon);
        let expected_annular = (b / a) * (b / a);
        // On-axis annular departure evaluates 1 - (ro/rs)^2, matching (b/a)^2 to 0 ULP here.
        assert_eq!(annular.to_bits(), expected_annular.to_bits());
        assert_eq!(
            status_with_moon(sat, sun, moon_apogee).expect("annular status"),
            EclipseStatus::Penumbra
        );

        // Moon behind the Sun cannot occult the Sun.
        let moon_behind_sun = [2.0 * AU_KM, 0.0, 0.0];
        assert_eq!(
            moon_shadow_fraction(sat, sun, moon_behind_sun).expect("moon behind sun"),
            0.0
        );

        // Inside the Sun (not at its center): Orekit dummy angles give full lighting (shadow 0.0).
        assert_eq!(
            shadow_fraction_with_moon([AU_KM - 100_000.0, 0.0, 0.0], sun, moon_clear)
                .expect("inside sun dummy angles"),
            0.0
        );

        // Validation errors: non-finite, zero, inside Earth/Moon, norm overflow.
        assert_invalid_eclipse_field(
            shadow_fraction_with_moon(sat, sun, [f64::NAN, 0.0, 0.0]).unwrap_err(),
            "moon_pos",
            "not finite",
        );
        assert_invalid_eclipse_field(
            shadow_fraction_with_moon(sat, sun, [0.0, 0.0, 0.0]).unwrap_err(),
            "moon_pos",
            "zero vector",
        );
        assert_invalid_eclipse_field(
            shadow_fraction_with_moon([6_000.0, 0.0, 0.0], sun, moon_clear).unwrap_err(),
            "sat_pos",
            "out of range",
        );
        assert_invalid_eclipse_field(
            shadow_fraction_with_moon([0.0, 0.0, 6_350.0], sun, moon_clear).unwrap_err(),
            "sat_pos",
            "out of range",
        );
        assert_eq!(
            shadow_fraction_with_model_and_moon(
                [0.0, 0.0, 6_350.0],
                sun,
                moon_clear,
                EarthShadowModel::Wgs84Oblate,
            )
            .expect("above oblate pole"),
            0.0
        );
        assert_invalid_eclipse_field(
            shadow_fraction_with_model_and_moon(
                [0.0, 0.0, 6_349.0],
                sun,
                moon_clear,
                EarthShadowModel::Wgs84Oblate,
            )
            .unwrap_err(),
            "sat_pos",
            "out of range",
        );
        assert_invalid_eclipse_field(
            shadow_fraction_with_moon([0.0, 384_400.0, 1_000.0], sun, moon_clear).unwrap_err(),
            "sat_pos",
            "out of range",
        );
        assert_invalid_eclipse_field(
            shadow_fraction_with_moon([1.0e200, 0.0, 0.0], sun, moon_clear).unwrap_err(),
            "sat_pos",
            "out of range",
        );
    }

    #[test]
    fn triple_disc_overlap_matches_analytic_limits() {
        // Equal overlapping circles: when two identical discs coincide with the
        // Sun's right half, their union equals a single disc's overlap rather
        // than doubling it or multiplying lighting fractions.
        let a = 0.005;
        let b = 0.005;
        let c = 0.005;
        let single = orekit::single_body_disc_shadow_orekit(a, b, c);
        let identical = orekit::combined_two_body_disc_shadow_orekit(a, b, c, b, c, 1.0, 0.0);
        assert_close(identical, single, 1.0e-14);

        // Two equal discs placed on opposite sides of the Sun (psi = pi) are
        // disjoint inside the Sun when c > b, so their combined shadow is
        // strictly additive (2 * single), not 1 - (1 - single)^2.
        let c_disjoint = 0.006;
        let one_side = orekit::single_body_disc_shadow_orekit(a, b, c_disjoint);
        let opposite = orekit::combined_two_body_disc_shadow_orekit(
            a, b, c_disjoint, b, c_disjoint, -1.0, 0.0,
        );
        assert_close(opposite, 2.0 * one_side, 1.0e-14);
        let product_formula = 1.0 - (1.0 - one_side) * (1.0 - one_side);
        assert!((opposite - product_formula).abs() > 1.0e-3);
    }

    #[test]
    fn equal_apparent_radii_near_coincident_earth_and_moon_preserve_shadow() {
        let sat = [-637_100.0, 0.0, 0.0];
        let angle = 0.008_f64;
        let sun = [
            sat[0] + AU_KM * libm::cos(angle),
            sat[1] + AU_KM * libm::sin(angle),
            sat[2],
        ];
        let moon = [sat[0] + 173_740.0, sat[1], sat[2] + 1.7374e-14];

        let f_moon = moon_shadow_fraction(sat, sun, moon).expect("valid moon geometry");
        let far_moon = [0.0, 384_400.0, 0.0];
        let f_earth =
            shadow_fraction_with_moon(sat, sun, far_moon).expect("valid earth-only geometry");
        let f_both = shadow_fraction_with_moon(sat, sun, moon).expect("valid combined geometry");

        assert!(
            f_earth > 0.65 && f_earth < 0.80,
            "expected partial earth shadow, got {f_earth}"
        );
        assert_close(f_moon, f_earth, 1.0e-12);
        assert!(
            f_both + 1.0e-14 >= f_earth && f_both + 1.0e-14 >= f_moon,
            "combined shadow {f_both} fell below single-body shadows ({f_earth}, {f_moon})"
        );
        assert_close(f_both, f_earth, 1.0e-11);

        // Three equal-radius discs separated by sub-ULP distances must not eliminate all discs.
        let r = 0.01_f64;
        let eps = 1.0e-19_f64;
        let area_all_equal = orekit::triple_disc_intersection_area_double([
            (0.0, 0.0, r),
            (eps, 0.0, r),
            (0.0, eps, r),
        ]);
        assert_close(area_all_equal, std::f64::consts::PI * r * r, 1.0e-14);

        // A small disc centered on the boundary of a much larger disc is only half inside it.
        let r_big = 1.0_f64;
        let r_small = 1.0e-17_f64;
        let area_boundary = orekit::triple_disc_intersection_area_double([
            (0.0, 0.0, r_small),
            (r_big, 0.0, r_big),
            (0.0, 0.0, 2.0 * r_small),
        ]);
        let expected_half = 0.5 * std::f64::consts::PI * r_small * r_small;
        assert!(
            (area_boundary - expected_half).abs() < 1.0e-3 * expected_half,
            "boundary disc intersection {area_boundary:e} should be ~half disc {expected_half:e}"
        );
    }

    #[test]
    fn tiny_grazing_moon_arc_on_earth_limb_preserves_triple_area() {
        // Sub-ULP grazing configuration (d_EM = b_earth + 1 ULP(b_earth),
        // alpha_EM < 0.5 ULP(pi)): verify the triple-disc intersection kernel
        // itself stays strictly positive and bounded by both pairwise overlaps.
        let d_earth = 6_500.0_f64;
        let b_earth = libm::asin(MEAN_EARTH_RADIUS_KM / d_earth);
        let d_sun = 1.0e20_f64;
        let a = libm::asin(IAU_2015_SOLAR_RADIUS_KM / d_sun);
        let d_moon = 5.1e18_f64;
        let b_moon = libm::asin(MOON_RADIUS_KM / d_moon);
        let c_e = b_earth - 0.5 * a;
        let dx_p = (c_e - b_earth) - 2.220_446_049_250_313e-16;
        let dy_p = 0.8 * a;
        let c_m = libm::hypot(dx_p, dy_p);

        let raw_triple = orekit::triple_disc_intersection_area_double([
            (0.0, 0.0, a),
            (c_e, 0.0, b_earth),
            (dx_p, dy_p, b_moon),
        ]);
        let area_m_direct = orekit::two_disc_intersection_area_orekit(a, b_moon, c_m);
        let area_em = orekit::two_disc_intersection_area_orekit(
            b_earth,
            b_moon,
            libm::hypot(dx_p - c_e, dy_p),
        );
        assert!(
            raw_triple > 0.0 && raw_triple <= area_m_direct && raw_triple <= area_em * (1.0 + 1.0e-12),
            "raw triple intersection {raw_triple:e} out of physical bounds (area_m={area_m_direct:e}, area_em={area_em:e})"
        );
    }

    #[test]
    fn combined_eclipse_fraction_property_monotonic_and_bounded() {
        let mut rng = SplitMix64(0x5eed_cafe_1234_5678);

        // 1. Planar apparent-disc sweep (6,000 cases): random scales, equal
        //    apparent radii (down to sub-ULP offsets), external/internal
        //    tangency, and tiny grazing discs on large limbs.
        let mut max_planar_overshoot = 0.0_f64;
        let mut max_planar_mono_viol = 0.0_f64;
        let mut max_planar_subadd_viol = 0.0_f64;
        let mut max_planar_triple_excess = 0.0_f64;
        for case in 0..6_000 {
            let (a, b_e, c_e, b_m, c_m, psi) = match case % 6 {
                0 => {
                    let a = rng.log_uniform(-15.0, -0.5);
                    let b_e = rng.log_uniform(-15.0, 0.1);
                    let b_m = rng.log_uniform(-15.0, 0.1);
                    let c_e = rng.unit() * (a + b_e) * 1.1;
                    let c_m = rng.unit() * (a + b_m) * 1.1;
                    let psi = std::f64::consts::TAU * rng.unit();
                    (a, b_e, c_e, b_m, c_m, psi)
                }
                1 => {
                    let a = rng.log_uniform(-8.0, -1.0);
                    let b = if rng.unit() < 0.3 {
                        a
                    } else {
                        rng.log_uniform(-8.0, 0.0)
                    };
                    let c_e = rng.unit() * (a + b);
                    let perturb = match (case / 6) % 5 {
                        0 => 0.0,
                        1 => b * 1.0e-18,
                        2 => b * 1.0e-16,
                        3 => b * 1.0e-12,
                        _ => b * rng.unit() * 0.5,
                    };
                    let theta = std::f64::consts::TAU * rng.unit();
                    let (sin_t, cos_t) = libm::sincos(theta);
                    let mx = c_e + perturb * cos_t;
                    let my = perturb * sin_t;
                    let c_m = libm::hypot(mx, my);
                    let psi = libm::atan2(my, mx);
                    (a, b, c_e, b, c_m, psi)
                }
                2 => {
                    let a = rng.log_uniform(-10.0, -1.0);
                    let b_e = rng.log_uniform(-10.0, 0.1);
                    let b_m = rng.log_uniform(-10.0, 0.1);
                    let shift = |x: f64, mode: usize| match mode % 3 {
                        0 => x,
                        1 => x.next_down(),
                        _ => x.next_up(),
                    };
                    let c_e = shift(a + b_e, case);
                    let c_m = if rng.unit() < 0.5 {
                        shift(a + b_m, case / 3)
                    } else {
                        rng.unit() * (a + b_m)
                    };
                    let psi = std::f64::consts::TAU * rng.unit();
                    (a, b_e, c_e, b_m, c_m, psi)
                }
                3 => {
                    let a = rng.log_uniform(-10.0, -1.0);
                    let b_e = rng.log_uniform(-10.0, 0.1);
                    let b_m = rng.log_uniform(-10.0, 0.1);
                    let c_e = (a - b_e).abs();
                    let c_e = match (case / 6) % 3 {
                        0 => c_e,
                        1 => c_e.next_down().max(0.0),
                        _ => c_e.next_up(),
                    };
                    let touch_dist = if rng.unit() < 0.5 {
                        b_e + b_m
                    } else {
                        (b_e - b_m).abs()
                    };
                    let touch_dist = match (case / 18) % 3 {
                        0 => touch_dist,
                        1 => touch_dist.next_down().max(0.0),
                        _ => touch_dist.next_up(),
                    };
                    let angle = std::f64::consts::TAU * rng.unit();
                    let (sin_a, cos_a) = libm::sincos(angle);
                    let mx = c_e + touch_dist * cos_a;
                    let my = touch_dist * sin_a;
                    let c_m = libm::hypot(mx, my);
                    let psi = libm::atan2(my, mx);
                    (a, b_e, c_e, b_m, c_m, psi)
                }
                4 => {
                    let b_e = 0.5 + 0.9 * rng.unit();
                    let a = rng.log_uniform(-15.0, -4.0);
                    let b_m = a * rng.log_uniform(-2.0, 0.5);
                    let c_e = b_e + (2.0 * rng.unit() - 1.0) * a;
                    let offset = (2.0 * rng.unit() - 1.0) * b_m;
                    let mx = (c_e - b_e) + offset;
                    let my = (2.0 * rng.unit() - 1.0) * a;
                    let c_m = libm::hypot(mx, my);
                    let psi = libm::atan2(my, mx);
                    (a, b_e, c_e, b_m, c_m, psi)
                }
                _ => {
                    let a = rng.log_uniform(-8.0, -1.0);
                    let b_e = rng.log_uniform(-8.0, 0.1);
                    let b_m = rng.log_uniform(-8.0, 0.1);
                    let c_e = rng.unit() * (a + b_e);
                    let c_m = rng.unit() * (a + b_m);
                    let psi = if rng.unit() < 0.5 {
                        0.0
                    } else {
                        std::f64::consts::PI
                    };
                    (a, b_e, c_e, b_m, c_m, psi)
                }
            };

            let (sin_psi, cos_psi) = libm::sincos(psi);
            let area_e = orekit::two_disc_intersection_area_orekit(a, b_e, c_e);
            let area_m = orekit::two_disc_intersection_area_orekit(a, b_m, c_m);
            let raw_triple = orekit::triple_disc_intersection_area_double([
                (0.0, 0.0, a),
                (c_e, 0.0, b_e),
                (c_m * cos_psi, c_m * sin_psi, b_m),
            ]);
            let sun_area = std::f64::consts::PI * a * a;
            let triple_excess = (raw_triple - area_e.min(area_m)).max(0.0) / sun_area;
            max_planar_triple_excess = max_planar_triple_excess.max(triple_excess);
            assert!(
                raw_triple.is_finite()
                    && raw_triple >= 0.0
                    && raw_triple <= area_e.min(area_m) + 2.1e-15 * sun_area,
                "case {case}: raw triple {raw_triple:e} out of [0, min({area_e:e}, {area_m:e})]"
            );

            // Check exact-kernel inclusion-exclusion union on all 6,000 cases.
            let f_e_exact = (area_e / sun_area).clamp(0.0, 1.0);
            let f_m_exact = (area_m / sun_area).clamp(0.0, 1.0);
            let f_both_exact = ((area_e + area_m - raw_triple) / sun_area).clamp(0.0, 1.0);
            assert!(
                f_both_exact + 1.0e-14 >= f_e_exact.max(f_m_exact),
                "planar exact case {case}: mono violated"
            );

            // Also check Orekit's maskingRatio + triple-disc union on planar
            // cases within the physical solar-system ratio regime (a >= 1e-3,
            // b/a <= 300, covering LEO Earth b/a ~ 255 through distant).
            if a >= 1.0e-3 && b_e <= 300.0 * a && b_m <= 300.0 * a {
                let f_e = orekit::single_body_disc_shadow_orekit(a, b_e, c_e);
                let f_m = orekit::single_body_disc_shadow_orekit(a, b_m, c_m);
                let f_both = orekit::combined_two_body_disc_shadow_orekit(
                    a, b_e, c_e, b_m, c_m, cos_psi, sin_psi,
                );
                assert!(f_both.is_finite(), "planar case {case}: non-finite f_both");
                max_planar_overshoot = max_planar_overshoot
                    .max((-f_both).max(f_both - 1.0))
                    .max((-f_e).max(f_e - 1.0))
                    .max((-f_m).max(f_m - 1.0));
                max_planar_mono_viol = max_planar_mono_viol.max((f_e.max(f_m) - f_both).max(0.0));
                max_planar_subadd_viol =
                    max_planar_subadd_viol.max((f_both - (f_e + f_m).min(1.0)).max(0.0));
            }
        }

        // 2. End-to-end 3D geometries (2,000 cases), each evaluated under both
        //    Spherical and Wgs84Oblate Earth models.
        let mut max_3d_overshoot = 0.0_f64;
        let mut max_3d_mono_viol = 0.0_f64;
        let mut max_3d_subadd_viol = 0.0_f64;
        for case in 0..2_000 {
            let u_sat = rng.unit_vec3();
            let d_earth = rng.log_uniform(3.85, 6.8);
            let sat = vec3::scale3(u_sat, d_earth);
            let u_earth = vec3::neg3(u_sat);
            let b_earth = libm::asin(MEAN_EARTH_RADIUS_KM / d_earth);

            let ref_v = if u_earth[0].abs() < 0.8 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let t1_raw = vec3::cross3(u_earth, ref_v);
            let t1_n = vec3::norm3(t1_raw);
            let t1 = [t1_raw[0] / t1_n, t1_raw[1] / t1_n, t1_raw[2] / t1_n];
            let t2 = vec3::cross3(u_earth, t1);

            let d_sun = rng.log_uniform(7.8, 9.0);
            let a = libm::asin(IAU_2015_SOLAR_RADIUS_KM / d_sun);

            let c_sun_offset = match (case / 2) % 6 {
                0 => rng.unit() * (a + b_earth) * 1.05,
                1 => a + b_earth,
                2 => (b_earth - a).abs(),
                3 => 0.0,
                4 => rng.log_uniform(-16.0, -9.0),
                _ => rng.unit() * (a + b_earth),
            };
            let az_sun = std::f64::consts::TAU * rng.unit();
            let (s_az, c_az) = libm::sincos(az_sun);
            let (s_cs, c_cs) = libm::sincos(c_sun_offset);
            let u_sun = [
                c_cs * u_earth[0] + s_cs * (c_az * t1[0] + s_az * t2[0]),
                c_cs * u_earth[1] + s_cs * (c_az * t1[1] + s_az * t2[1]),
                c_cs * u_earth[2] + s_cs * (c_az * t1[2] + s_az * t2[2]),
            ];
            let sun = [
                sat[0] + d_sun * u_sun[0],
                sat[1] + d_sun * u_sun[1],
                sat[2] + d_sun * u_sun[2],
            ];

            let d_moon = if (case / 2) % 3 == 0 {
                d_earth * (MOON_RADIUS_KM / MEAN_EARTH_RADIUS_KM)
            } else {
                rng.log_uniform(3.5, 7.5).min(0.95 * d_sun)
            };
            let b_moon = libm::asin(MOON_RADIUS_KM / d_moon);
            let moon_offset = match (case / 2) % 5 {
                0 => 0.0,
                1 => 1.0e-18,
                2 => a + b_moon,
                3 => b_earth + b_moon,
                _ => rng.unit() * (a + b_moon) * 1.1,
            };
            let az_m = std::f64::consts::TAU * rng.unit();
            let (sm_az, cm_az) = libm::sincos(az_m);
            let (s_mo, c_mo) = libm::sincos(moon_offset);
            let anchor = if (case / 2) % 2 == 0 { u_earth } else { u_sun };
            let m_ref = if anchor[0].abs() < 0.8 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let m1_raw = vec3::cross3(anchor, m_ref);
            let m1_n = vec3::norm3(m1_raw);
            let m1 = [m1_raw[0] / m1_n, m1_raw[1] / m1_n, m1_raw[2] / m1_n];
            let m2 = vec3::cross3(anchor, m1);
            let u_moon = [
                c_mo * anchor[0] + s_mo * (cm_az * m1[0] + sm_az * m2[0]),
                c_mo * anchor[1] + s_mo * (cm_az * m1[1] + sm_az * m2[1]),
                c_mo * anchor[2] + s_mo * (cm_az * m1[2] + sm_az * m2[2]),
            ];
            let moon = [
                sat[0] + d_moon * u_moon[0],
                sat[1] + d_moon * u_moon[1],
                sat[2] + d_moon * u_moon[2],
            ];
            if vec3::norm3(moon) == 0.0 {
                continue;
            }
            let far_moon = [
                sat[0] + 2.0 * d_sun * u_sun[0],
                sat[1] + 2.0 * d_sun * u_sun[1],
                sat[2] + 2.0 * d_sun * u_sun[2],
            ];

            let f_moon = moon_shadow_fraction(sat, sun, moon).expect("valid moon-only 3D case");
            let b_earth_polar =
                libm::asin(MEAN_EARTH_RADIUS_KM * (1.0 - WGS84_FLATTENING) / d_earth);
            for model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
                let f_earth = shadow_fraction_with_model_and_moon(sat, sun, far_moon, model)
                    .expect("valid earth-only 3D case");
                let f_both = shadow_fraction_with_model_and_moon(sat, sun, moon, model)
                    .expect("valid combined 3D case");

                assert!(f_both.is_finite(), "3D case {case} {model:?}: non-finite");
                max_3d_overshoot = max_3d_overshoot
                    .max((-f_both).max(f_both - 1.0))
                    .max((-f_earth).max(f_earth - 1.0))
                    .max((-f_moon).max(f_moon - 1.0));
                max_3d_mono_viol = max_3d_mono_viol.max((f_earth.max(f_moon) - f_both).max(0.0));
                max_3d_subadd_viol =
                    max_3d_subadd_viol.max((f_both - (f_earth + f_moon).min(1.0)).max(0.0));
                if c_sun_offset <= 1.0e-9 && b_earth_polar > a + 1.0e-6 {
                    assert_eq!(
                        f_earth, 1.0,
                        "3D case {case} {model:?}: Earth alone must cover the Sun on the axis"
                    );
                }
            }
        }
        // Measured bounds over the 6,000-case planar sweep (solar-system ratio
        // subset a >= 1e-3, b <= 300 a): overshoot 1.556e-9, monotonicity
        // residual 1.570e-9, subadditivity residual 1.110e-16 (0.5 * EPSILON),
        // and triple-disc kernel excess 2.049e-15 across all 6,000 cases.
        // Cause: Orekit's unclamped maskingRatio (aE2 - alpha2^2 cancellation
        // with acos amplification near ±1 on grazing cases, and ±nextDown(1.0)
        // clamping on annular/small-disc cases) vs the exact triple-disc area.
        assert!(
            max_planar_triple_excess <= 2.1e-15,
            "planar triple excess {max_planar_triple_excess:e}"
        );
        assert!(
            max_planar_overshoot <= 1.6e-9,
            "planar overshoot {max_planar_overshoot:e}"
        );
        assert!(
            max_planar_mono_viol <= 1.6e-9,
            "planar mono {max_planar_mono_viol:e}"
        );
        assert!(
            max_planar_subadd_viol <= f64::EPSILON,
            "planar subadd {max_planar_subadd_viol:e}"
        );

        // Measured bounds over the 2,000-case 3D sweep (both Spherical and
        // Wgs84Oblate): overshoot 1.435e-6, monotonicity residual 7.026e-11,
        // subadditivity residual 1.435e-6. Cause: Orekit's unclamped
        // maskingRatio just outside ANGULAR_MARGIN = 1e-10, where rounding in
        // alpha2 / alphaCentral (near ±1, not clamped) is amplified by acos and
        // does not cancel against sqrt(aE2 - alpha2^2) after cancellation in
        // aE2 - alpha2^2 (the ±nextDown(1.0) clamp acts on annular/small-disc
        // cases, not on these grazing overshoots).
        assert!(
            max_3d_overshoot <= 1.44e-6,
            "3d overshoot {max_3d_overshoot:e}"
        );
        assert!(max_3d_mono_viol <= 7.1e-11, "3d mono {max_3d_mono_viol:e}");
        assert!(
            max_3d_subadd_viol <= 1.44e-6,
            "3d subadd {max_3d_subadd_viol:e}"
        );
    }

    #[test]
    fn oblate_earth_limb_angle_stays_accurate_at_large_distance() {
        use crate::astro::constants::units::M_PER_KM;

        let r_eq_m = MEAN_EARTH_RADIUS_KM * M_PER_KM;
        let r_polar_m = r_eq_m * (1.0 - WGS84_FLATTENING);

        // At large distances where the polar/equatorial limb difference
        // (R_eq - R_polar) / d still exceeds Vec3::angle(psat - plimb, psat)
        // rounding (~2e-16 rad), check the limb radius directly against
        // fast_math::asin(R / d).
        for scale_km in [1.0e7, 1.0e9, 1.0e11, 1.0e12, 1.0e14] {
            let sat = [scale_km, scale_km, scale_km];
            let sun = [-1.0e3 * scale_km, -1.0e3 * scale_km, -1.0e3 * scale_km];
            let d_m = ad::Vec3::new(sat[0] * M_PER_KM, sat[1] * M_PER_KM, sat[2] * M_PER_KM).norm();
            let b_eq = fast_math::asin(r_eq_m / d_m);
            let b_polar = fast_math::asin(r_polar_m / d_m);
            assert!(
                b_eq - b_polar > 1.0e-13,
                "scale={scale_km:e}: polar/equatorial gap must exceed angle rounding"
            );

            let (_, limb_sph, _) =
                orekit::earth_occultation_angles_double(sat, sun, EarthShadowModel::Spherical)
                    .expect("spherical far-field angles");
            let (_, limb_obl, _) =
                orekit::earth_occultation_angles_double(sat, sun, EarthShadowModel::Wgs84Oblate)
                    .expect("oblate far-field angles");

            let sph_err = (limb_sph - b_eq).abs();
            assert!(
                sph_err <= 2.0e-15,
                "scale={scale_km:e}: spherical limb {limb_sph:.17e} != asin(R/d) {b_eq:.17e} (err={sph_err:e})"
            );
            assert!(
                limb_obl > b_polar && limb_obl < b_eq,
                "scale={scale_km:e}: oblate limb {limb_obl:.17e} not strictly between polar {b_polar:.17e} and equatorial {b_eq:.17e}"
            );
        }

        // Shadow assertion at a large distance where a_sun >> ANGULAR_MARGIN
        // (1e-10), so f == 1.0 fails if the limb radius collapses toward 0.
        let sat = [1.0e10, 1.0e10, 1.0e10];
        let sun = [-1.0e13, -1.0e13, -1.0e13];
        let moon = [0.0, 384_400.0, 0.0];
        let (_, _, a_sun) =
            orekit::earth_occultation_angles_double(sat, sun, EarthShadowModel::Wgs84Oblate)
                .expect("far-field sun angle");
        assert!(
            a_sun > 100.0 * 1.0e-10,
            "a_sun ({a_sun:e}) must exceed ANGULAR_MARGIN so shadow == 1.0 is non-vacuous"
        );
        let f_sph =
            shadow_fraction_with_model_and_moon(sat, sun, moon, EarthShadowModel::Spherical)
                .expect("spherical far-field shadow");
        let f_obl =
            shadow_fraction_with_model_and_moon(sat, sun, moon, EarthShadowModel::Wgs84Oblate)
                .expect("oblate far-field shadow");
        assert_eq!(f_sph, 1.0);
        assert_eq!(f_obl, 1.0);
    }

    #[test]
    fn conical_shadow_rejects_sun_overlapping_earth() {
        let sat = [-10.0, 6_370.0, 0.0];
        let sun = [700_000.0, 0.0, 0.0];

        assert_invalid_eclipse_field(
            shadow_fraction(sat, sun).unwrap_err(),
            "sun_pos",
            "out of range",
        );
        assert_invalid_eclipse_field(
            shadow_fraction_with_model(sat, sun, EarthShadowModel::Spherical).unwrap_err(),
            "sun_pos",
            "out of range",
        );
        assert_invalid_eclipse_field(
            shadow_fraction_with_model(sat, sun, EarthShadowModel::Wgs84Oblate).unwrap_err(),
            "sun_pos",
            "out of range",
        );
    }

    #[test]
    fn coincident_and_overflowing_relative_positions_report_consistent_errors() {
        let sun = [AU_KM, 0.0, 0.0];
        let moon = [0.0, 384_400.0, 0.0];

        assert_invalid_eclipse_field(
            moon_shadow_fraction(sun, sun, moon).unwrap_err(),
            "sat_pos",
            "out of range",
        );
        assert_invalid_eclipse_field(
            shadow_fraction_with_moon(sun, sun, moon).unwrap_err(),
            "sat_pos",
            "out of range",
        );
        assert_invalid_eclipse_field(
            moon_shadow_fraction(moon, sun, moon).unwrap_err(),
            "sat_pos",
            "out of range",
        );
        assert_invalid_eclipse_field(
            shadow_fraction_with_moon(moon, sun, moon).unwrap_err(),
            "sat_pos",
            "out of range",
        );

        let err =
            moon_shadow_fraction([-1.0e308, 0.0, 0.0], [1.0e308, 0.0, 0.0], moon).unwrap_err();
        let EclipseError::InvalidInput { reason, .. } = err;
        assert_eq!(reason, "out of range");
    }

    #[test]
    fn oblate_limb_gives_total_umbra_on_the_anti_sun_axis_at_geo() {
        let sat = [30_000.0, 30_000.0, 0.0];
        let sun = [-1.0e8, -1.0e8, 0.0];
        let moon = [0.0, 384_400.0, 0.0];

        for model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
            assert_eq!(
                shadow_fraction_with_model_and_moon(sat, sun, moon, model)
                    .expect("valid GEO umbra geometry"),
                1.0,
                "{model:?}"
            );
            assert_eq!(
                status_with_model_and_moon(sat, sun, moon, model).expect("valid GEO umbra status"),
                EclipseStatus::Umbra,
                "{model:?}"
            );
        }
    }

    #[test]
    fn every_earth_model_gives_total_umbra_on_and_near_the_anti_sun_axis() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let t = 1.0 / 3.0_f64.sqrt();
        let fixed_dirs = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [s, s, 0.0],
            [-s, -s, 0.0],
            [s, -s, 0.0],
            [0.0, s, s],
            [s, 0.0, -s],
            [t, t, t],
            [-t, t, -t],
        ];
        let bands = [
            (6_771.0, 8_371.0),
            (20_000.0, 30_000.0),
            (42_164.0, 42_164.0),
            (356_500.0, 406_700.0),
        ];
        let offsets = [0.0, 1.0e-12, 1.0e-6, 1.0, 100.0];
        let models = [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate];

        let mut rng = SplitMix64(0x0a11_5ade_1234_0001);
        for case in 0..500 {
            let dir = if case < fixed_dirs.len() {
                fixed_dirs[case]
            } else {
                rng.unit_vec3()
            };
            let ref_v = if dir[0].abs() < 0.8 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let perp_raw = vec3::cross3(dir, ref_v);
            let perp = vec3::scale3(perp_raw, 1.0 / vec3::norm3(perp_raw));
            let d_sun = AU_KM * (0.983 + 0.034 * rng.unit());
            let sun = vec3::scale3(dir, d_sun);
            let moon = vec3::scale3(dir, 2.0 * d_sun);

            for &(lo, hi) in &bands {
                let d = lo + (hi - lo) * rng.unit();
                for &offset in &offsets {
                    let sat = [
                        -d * dir[0] + offset * perp[0],
                        -d * dir[1] + offset * perp[1],
                        -d * dir[2] + offset * perp[2],
                    ];
                    for model in models {
                        let context = format!("case {case} d={d} offset={offset} {model:?}");
                        assert_eq!(
                            shadow_fraction_with_model(sat, sun, model).expect("valid conical"),
                            1.0,
                            "conical {context}"
                        );
                        assert_eq!(
                            status_with_model(sat, sun, model).expect("valid conical status"),
                            EclipseStatus::Umbra,
                            "conical status {context}"
                        );
                        assert_eq!(
                            shadow_fraction_with_model_and_moon(sat, sun, moon, model)
                                .expect("valid apparent disc"),
                            1.0,
                            "apparent disc {context}"
                        );
                        assert_eq!(
                            status_with_model_and_moon(sat, sun, moon, model)
                                .expect("valid apparent disc status"),
                            EclipseStatus::Umbra,
                            "apparent disc status {context}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn oblate_disc_shadow_stays_between_polar_and_equatorial_sphere_shadows() {
        use crate::astro::constants::units::M_PER_KM;
        let re = MEAN_EARTH_RADIUS_KM;
        let rc = re * (1.0 - WGS84_FLATTENING);
        let re_m = re * M_PER_KM;
        let rc_m = rc * M_PER_KM;
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let fixed_dirs = [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [s, s, 0.0],
            [-s, 0.0, s],
            [0.0, -s, -s],
        ];

        let mut max_lo_viol = 0.0_f64;
        let mut max_hi_viol = 0.0_f64;
        let mut max_overshoot = 0.0_f64;
        let mut rng = SplitMix64(0x0b1a_7e00_5eed_0002);
        for case in 0..6_000 {
            let u_sat = if case % 4 == 0 {
                fixed_dirs[(case / 4) % fixed_dirs.len()]
            } else {
                rng.unit_vec3()
            };
            let d_earth = re * 1.001 * libm::pow(2.0e6 / (re * 1.001), rng.unit());
            let sat = vec3::scale3(u_sat, d_earth);
            let u_earth = vec3::neg3(u_sat);

            let ref_v = if u_earth[0].abs() < 0.8 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let t1_raw = vec3::cross3(u_earth, ref_v);
            let t1 = vec3::scale3(t1_raw, 1.0 / vec3::norm3(t1_raw));
            let t2 = vec3::cross3(u_earth, t1);
            let rotate = |center: [f64; 3], offset: f64, azimuth: f64| {
                let (s_az, c_az) = libm::sincos(azimuth);
                let (s_off, c_off) = libm::sincos(offset);
                [
                    c_off * center[0] + s_off * (c_az * t1[0] + s_az * t2[0]),
                    c_off * center[1] + s_off * (c_az * t1[1] + s_az * t2[1]),
                    c_off * center[2] + s_off * (c_az * t1[2] + s_az * t2[2]),
                ]
            };

            let d_sun = AU_KM * (0.983 + 0.034 * rng.unit());
            let a = libm::asin(IAU_2015_SOLAR_RADIUS_KM / d_sun);
            let b_lo = libm::asin(rc / d_earth);
            let b_hi = libm::asin(re / d_earth);
            let c_sun = match case % 5 {
                0 => 0.0,
                1 => rng.log_uniform(-16.0, -9.0),
                2 => (b_lo - a).abs() + (b_hi - b_lo + 2.0 * a) * rng.unit(),
                3 => b_lo + a + (b_hi - b_lo) * rng.unit(),
                _ => rng.unit() * (b_hi + a) * 1.2,
            };
            let u_sun = rotate(u_earth, c_sun, std::f64::consts::TAU * rng.unit());
            let sun = [
                sat[0] + d_sun * u_sun[0],
                sat[1] + d_sun * u_sun[1],
                sat[2] + d_sun * u_sun[2],
            ];

            let moon = if (case / 5) % 2 == 0 {
                [
                    sat[0] + 2.0 * d_sun * u_sun[0],
                    sat[1] + 2.0 * d_sun * u_sun[1],
                    sat[2] + 2.0 * d_sun * u_sun[2],
                ]
            } else {
                let d_moon = 300_000.0 + 100_000.0 * rng.unit();
                let b_moon = libm::asin(MOON_RADIUS_KM / d_moon);
                let u_moon = rotate(
                    u_sun,
                    rng.unit() * (a + b_moon) * 1.2,
                    std::f64::consts::TAU * rng.unit(),
                );
                [
                    sat[0] + d_moon * u_moon[0],
                    sat[1] + d_moon * u_moon[1],
                    sat[2] + d_moon * u_moon[2],
                ]
            };

            let f_lo = 1.0
                - orekit::earth_moon_lighting_ratio_with_ellipsoid_double(
                    sat, sun, moon, rc_m, 0.0,
                )
                .expect("polar sphere");
            let f_oblate = 1.0
                - orekit::earth_moon_lighting_ratio_with_ellipsoid_double(
                    sat,
                    sun,
                    moon,
                    re_m,
                    WGS84_FLATTENING,
                )
                .expect("oblate spheroid");
            let f_hi = 1.0
                - orekit::earth_moon_lighting_ratio_with_ellipsoid_double(
                    sat, sun, moon, re_m, 0.0,
                )
                .expect("equatorial sphere");
            max_lo_viol = max_lo_viol.max((f_lo - f_oblate).max(0.0));
            max_hi_viol = max_hi_viol.max((f_oblate - f_hi).max(0.0));
            for f in [f_lo, f_oblate, f_hi] {
                max_overshoot = max_overshoot.max((-f).max(f - 1.0));
            }
        }
        // Measured bounds over the 6,000-case oblate-between-spheres sweep:
        // lo_viol 2.624e-10, hi_viol 9.622e-10, overshoot 2.919e-11.
        // Cause: Orekit's maskingRatio aE2 - alpha2^2 cancellation with acos
        // amplification near ±1 (and ±nextDown(1.0) clamping on annular cases)
        // differs slightly across the polar sphere, WGS84 oblate spheroid, and
        // equatorial sphere.
        assert!(max_lo_viol <= 2.7e-10, "oblate lo_viol {max_lo_viol:e}");
        assert!(max_hi_viol <= 9.7e-10, "oblate hi_viol {max_hi_viol:e}");
        assert!(
            max_overshoot <= 3.0e-11,
            "oblate overshoot {max_overshoot:e}"
        );
    }

    /// Adaptive Simpson quadrature of `f` over `[lo, hi]`.
    fn adaptive_simpson(f: &dyn Fn(f64) -> f64, lo: f64, hi: f64, eps: f64) -> f64 {
        fn step(
            f: &dyn Fn(f64) -> f64,
            (lo, mid, hi): (f64, f64, f64),
            (f_lo, f_mid, f_hi): (f64, f64, f64),
            whole: f64,
            eps: f64,
            depth: u32,
        ) -> f64 {
            let (left_mid, right_mid) = (0.5 * (lo + mid), 0.5 * (mid + hi));
            let (f_left_mid, f_right_mid) = (f(left_mid), f(right_mid));
            let left = (mid - lo) / 6.0 * (f_lo + 4.0 * f_left_mid + f_mid);
            let right = (hi - mid) / 6.0 * (f_mid + 4.0 * f_right_mid + f_hi);
            let delta = left + right - whole;
            if depth == 0 || delta.abs() <= 15.0 * eps {
                return left + right + delta / 15.0;
            }
            step(
                f,
                (lo, left_mid, mid),
                (f_lo, f_left_mid, f_mid),
                left,
                0.5 * eps,
                depth - 1,
            ) + step(
                f,
                (mid, right_mid, hi),
                (f_mid, f_right_mid, f_hi),
                right,
                0.5 * eps,
                depth - 1,
            )
        }
        let mid = 0.5 * (lo + hi);
        let (f_lo, f_mid, f_hi) = (f(lo), f(mid), f(hi));
        let whole = (hi - lo) / 6.0 * (f_lo + 4.0 * f_mid + f_hi);
        step(f, (lo, mid, hi), (f_lo, f_mid, f_hi), whole, eps, 32)
    }

    /// Area of `{|p| <= a} ∩ {|p - (c_e, 0)| <= b_e} ∩ {p · e >= delta}` by
    /// quadrature over `t = p · e`, in coordinates scaled by the power of two
    /// `scale` so every length is of order one.
    fn sun_earth_area_beyond_line(
        a: f64,
        c_e: f64,
        b_e: f64,
        e: (f64, f64),
        delta: f64,
        scale: f64,
    ) -> f64 {
        let (a_s, c_s, b_s, delta_s) = (a * scale, c_e * scale, b_e * scale, delta * scale);
        let t_e = c_s * e.0;
        let w_e = -c_s * e.1;
        let chord = |t: f64| {
            let h_sun2 = a_s * a_s - t * t;
            let h_earth2 = b_s * b_s - (t - t_e) * (t - t_e);
            if h_sun2 <= 0.0 || h_earth2 <= 0.0 {
                return 0.0;
            }
            let (h_sun, h_earth) = (h_sun2.sqrt(), h_earth2.sqrt());
            (h_sun.min(w_e + h_earth) - (-h_sun).max(w_e - h_earth)).max(0.0)
        };
        adaptive_simpson(&chord, delta_s, a_s, 1.0e-13) / (scale * scale)
    }

    #[test]
    fn large_moon_limb_keeps_sub_ulp_earth_offset_in_triple_area() {
        let a = 3.0e-16;
        let c_e = 2.5e-17;
        let delta = f64::EPSILON / 2.0;
        let b_m = 0.625 - delta;
        assert_eq!(0.625 - b_m, delta);
        let scale = (1u64 << 53) as f64;

        for (mx, my) in [
            (-0.375, 0.5),
            (0.375, 0.5),
            (0.375, -0.5),
            (-0.375, -0.5),
            (0.5, 0.375),
            (-0.5, 0.375),
            (0.5, -0.375),
            (-0.5, -0.375),
        ] {
            assert_eq!(libm::hypot(mx, my), 0.625);
            assert_eq!(
                c_e - mx,
                -mx,
                "Earth's offset must vanish next to the Moon's x"
            );
            let e = (mx / 0.625, my / 0.625);
            for b_e in [a + 0.5 * c_e, a - 0.5 * c_e] {
                let area = orekit::triple_disc_intersection_area_double([
                    (0.0, 0.0, a),
                    (c_e, 0.0, b_e),
                    (mx, my, b_m),
                ]);
                let reference = sun_earth_area_beyond_line(a, c_e, b_e, e, delta, scale);
                assert!(
                    (area - reference).abs() <= 1.0e-9 * reference,
                    "moon=({mx}, {my}) b_e={b_e:e}: triple area {area:e} vs quadrature \
                     {reference:e} (relative error {:e})",
                    (area - reference) / reference
                );
            }
        }
    }

    #[test]
    fn near_coincident_large_discs_crossing_inside_a_small_sun_keep_their_vertex() {
        let n = 1_000_003.0_f64;
        let ulp = 1.0 / (1u64 << 41) as f64;
        let mx = 2.0 * n * (n + 1.0) * ulp;
        let my = (2.0 * n + 1.0) * ulp;
        let c = (2.0 * n * (n + 1.0) + 1.0) * ulp;
        let delta = 1.0 / (1u64 << 28) as f64;
        let r = c - delta;
        assert_eq!(libm::hypot(mx, my), c);
        assert_eq!(c - r, delta);
        assert_eq!(c - mx, ulp);
        let a = 1.0e-8;

        let area =
            orekit::triple_disc_intersection_area_double([(0.0, 0.0, a), (c, 0.0, r), (mx, my, r)]);

        let scale = (1u64 << 27) as f64;
        let moon_left = delta - ulp;
        let chord = |xs: f64| {
            let x = xs / scale;
            let h_sun2 = a * a - x * x;
            let h_earth2 = (x - delta) * (2.0 * c - delta - x);
            let h_moon2 = (x - moon_left) * (mx + r - x);
            if h_sun2 <= 0.0 || h_earth2 <= 0.0 || h_moon2 <= 0.0 {
                return 0.0;
            }
            let (h_sun, h_earth, h_moon) = (h_sun2.sqrt(), h_earth2.sqrt(), h_moon2.sqrt());
            let top = h_sun.min(h_earth).min(my + h_moon);
            let bottom = (-h_sun).max(-h_earth).max(my - h_moon);
            (top - bottom).max(0.0) * scale
        };
        let reference =
            adaptive_simpson(&chord, delta * scale, a * scale, 1.0e-13) / (scale * scale);

        assert!(
            (area - reference).abs() <= 1.0e-7 * reference,
            "triple area {area:e} vs quadrature {reference:e} (relative error {:e})",
            (area - reference) / reference
        );

        for angle in [0.3_f64, 0.7, 2.0, -1.1] {
            let (sin_t, cos_t) = libm::sincos(angle);
            let turn = |x: f64, y: f64| (x * cos_t - y * sin_t, x * sin_t + y * cos_t);
            let (ex, ey) = turn(c, 0.0);
            let (qx, qy) = turn(mx, my);
            let turned = orekit::triple_disc_intersection_area_double([
                (0.0, 0.0, a),
                (ex, ey, r),
                (qx, qy, r),
            ]);
            assert!(
                (turned - reference).abs() <= 1.0e-6 * reference,
                "turned by {angle}: triple area {turned:e} vs quadrature {reference:e} \
                 (relative error {:e})",
                (turned - reference) / reference
            );
        }
    }

    #[test]
    fn coplanar_earth_and_moon_on_opposite_sides_of_the_sun_get_position_angle_pi() {
        // Every position lies in the `z = 0` plane, with Earth and the Moon on
        // opposite sides of the Sun direction, so the position angle between
        // their offsets from the Sun must be exactly `pi` (`cos_psi == -1.0,
        // sin_psi == 0.0`) even when the offsets are only a few ULP of a unit vector.
        let sat = [-1.5e19 + 2048.0, -1.5e19 - 2048.0, 0.0];
        let sun = [1.0e21 + sat[0], 1.0e21 + sat[1], 0.0];
        let moon = [sat[0] + 1.2e19 + 2048.0, sat[1] + 1.2e19 - 2048.0, 0.0];
        let (cos_psi, sin_psi) = orekit::position_angle_for_vectors_double(sat, sun, moon);
        assert!(
            (cos_psi + 1.0).abs() <= f64::EPSILON,
            "expected cos_psi ≈ -1.0 (within 1 ULP from z * (1/z)), got {cos_psi:.17e}"
        );
        assert_eq!(sin_psi, 0.0);

        // At 1 AU, coplanar opposite-side Earth and Moon discs that are disjoint
        // on the Sun add their shadows directly.
        let sat_geo = [-42_164.0, 0.0, 0.0];
        let b_e = libm::asin(MEAN_EARTH_RADIUS_KM / 42_164.0);
        let a = libm::asin(IAU_2015_SOLAR_RADIUS_KM / AU_KM);
        let c_e = b_e + 0.3 * a;
        let sun_geo = [
            sat_geo[0] + AU_KM * libm::cos(c_e),
            sat_geo[1] + AU_KM * libm::sin(c_e),
            0.0,
        ];
        let c_m = 1.4 * a;
        let angle_m = c_e + c_m;
        let moon_geo = [
            sat_geo[0] + 384_400.0 * libm::cos(angle_m),
            sat_geo[1] + 384_400.0 * libm::sin(angle_m),
            0.0,
        ];
        let (cos_g, sin_g) = orekit::position_angle_for_vectors_double(sat_geo, sun_geo, moon_geo);
        assert!(
            (cos_g + 1.0).abs() <= f64::EPSILON,
            "expected cos_g ≈ -1.0 (within 1 ULP from z * (1/z)), got {cos_g:.17e}"
        );
        assert_eq!(sin_g, 0.0);
        let far_moon = [-sun_geo[0], -sun_geo[1], 0.0];
        let f_e = shadow_fraction_with_moon(sat_geo, sun_geo, far_moon).expect("earth");
        let f_m = moon_shadow_fraction(sat_geo, sun_geo, moon_geo).expect("moon");
        let f_both = shadow_fraction_with_moon(sat_geo, sun_geo, moon_geo).expect("both");
        assert!(f_e > 0.0 && f_e < 1.0 && f_m > 0.0 && f_m < 1.0);
        assert_close(f_both, f_e + f_m, 1.0e-15);
    }

    #[test]
    fn departures_behind_sun_and_sun_within_two_solar_radii_behave_as_documented() {
        // 1. Sun within 2 * R_sun of Earth's center (Orekit's isSunCenteredFrame
        //    guard returns lighting 1.0; here the geometry is evaluated).
        let sat = [-7_000.0, 0.0, 0.0];
        let close_sun = [1_000_000.0, 0.0, 0.0]; // < 2 * 695_700 = 1_391_400 km
        let moon = [0.0, 384_400.0, 0.0];
        assert_eq!(
            shadow_fraction_with_moon(sat, close_sun, moon).expect("close sun"),
            1.0
        );

        // 2. Occulting body behind the Sun (d_body >= d_sun): casts no shadow.
        let sat_far = [2.0 * AU_KM, 0.0, 0.0];
        let sun = [AU_KM, 0.0, 0.0];
        assert_eq!(
            shadow_fraction_with_moon(sat_far, sun, moon).expect("earth behind sun"),
            0.0
        );
        let (f_grad, g_grad) = shadow_fraction_with_model_moon_and_gradient(
            sat_far,
            sun,
            moon,
            EarthShadowModel::Spherical,
        )
        .expect("earth behind sun grad");
        assert_eq!(f_grad, 0.0);
        assert_eq!(g_grad, [0.0; 3]);
    }

    fn assert_invalid_eclipse_field(
        error: EclipseError,
        expected: &'static str,
        expected_reason: &'static str,
    ) {
        let EclipseError::InvalidInput { field, reason } = error;
        assert_eq!(field, expected);
        assert_eq!(reason, expected_reason);
    }
}
