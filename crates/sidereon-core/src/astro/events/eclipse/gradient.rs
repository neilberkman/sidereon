//! Shadow fractions with their forward-mode automatic differentiation gradients
//! with respect to the satellite position (per km), reproducing Hipparchus
//! 4.0.3 `Gradient` arithmetic and Orekit 13.1.10's `FieldSpacecraftState`
//! branch decisions.

use super::ad::{Grad, ScaledF64, ScaledGrad, ScaledVec3};
use super::orekit::earth_moon_lighting_ratio_field;
use super::{
    invalid_eclipse_input, scale_polar_axis, validate_nonzero_vec3, EarthShadowModel, EclipseError,
    WGS84_FLATTENING,
};
use crate::astro::constants::{
    astro::SOLAR_RADIUS_KM, earth::MEAN_EARTH_RADIUS_KM, units::M_PER_KM,
};

/// Error while computing a shadow fraction with its gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ShadowGradientError {
    /// The positions failed the validation of the shadow fraction itself.
    #[error(transparent)]
    Eclipse(#[from] EclipseError),
    /// Evaluating the derivative produced a non-finite value (`NaN` or an
    /// infinity) from finite positions.
    #[error("shadow fraction gradient is not finite")]
    NumericalFailure,
}

/// [`shadow_fraction`](super::shadow_fraction) with its forward-mode derivative
/// with respect to the satellite position (per km).
///
/// Shares the single exponent-separated conical geometry evaluation with
/// [`shadow_fraction`](super::shadow_fraction) (so the returned shadow value is
/// bit-identical), forming `c = sun x sat` via the compensated scaled cross
/// product (`rho = |c| / |sun|`, `u_perp = normalize(c x sun)`,
/// `proj = (sat . sun) / |sun|`), evaluating both `shadow` and `nu = 1 - shadow`
/// directly from `rho`, and computing `d(shadow) = -d(nu)` with the quotient
/// rule of whichever of `shadow` or `nu` is at most `1/2`. Returns
/// [`ShadowGradientError::NumericalFailure`] on the exact antumbra axis
/// (`rho = 0`, `r_umbra <= 0`) or if evaluation produces a non-finite
/// derivative.
pub fn shadow_fraction_with_gradient(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
) -> Result<(f64, [f64; 3]), ShadowGradientError> {
    let g = conical_shadow_field(sat_pos, sun_pos, EarthShadowModel::Spherical)?;
    finish_grad(g)
}

/// [`shadow_fraction_with_model`](super::shadow_fraction_with_model) with its
/// forward-mode derivative with respect to the satellite position (per km).
///
/// The returned shadow value is bit-identical to
/// [`shadow_fraction_with_model`](super::shadow_fraction_with_model).
pub fn shadow_fraction_with_model_and_gradient(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    model: EarthShadowModel,
) -> Result<(f64, [f64; 3]), ShadowGradientError> {
    let g = conical_shadow_field(sat_pos, sun_pos, model)?;
    finish_grad(g)
}

/// [`shadow_fraction_with_model_and_moon`](super::shadow_fraction_with_model_and_moon)
/// with its forward-mode derivative with respect to the satellite position
/// (per km).
///
/// Evaluates the ported Orekit 13.1.10 `Field<Gradient>` occultation,
/// `maskingRatio`, and `getLightingRatio` operation order (with only the
/// mutual-eclipse term replaced by adding `triple_area / (pi * alpha_sun^2)`),
/// using `getReal()`-based branch decisions so derivatives match Orekit's
/// field evaluation including one-sided values at boundaries. Like
/// [`shadow_fraction_with_model_and_moon`](super::shadow_fraction_with_model_and_moon),
/// the shadow value is unclamped and can overshoot `[0.0, 1.0]` just outside
/// `ANGULAR_MARGIN = 1e-10` by up to `1.435e-6` measured across the oracle
/// fixtures and property sweeps. Returns
/// [`ShadowGradientError::NumericalFailure`] when field evaluation produces
/// `NaN` or an infinity.
pub fn shadow_fraction_with_model_moon_and_gradient(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    moon_pos: [f64; 3],
    model: EarthShadowModel,
) -> Result<(f64, [f64; 3]), ShadowGradientError> {
    let lighting = earth_moon_lighting_ratio_field(sat_pos, sun_pos, moon_pos, model)?;
    let shadow_val = 1.0 - lighting.value;
    let shadow_grad_km = [
        -lighting.grad[0] * M_PER_KM,
        -lighting.grad[1] * M_PER_KM,
        -lighting.grad[2] * M_PER_KM,
    ];
    let shadow = Grad {
        value: shadow_val,
        grad: shadow_grad_km,
    };
    finish_grad(shadow)
}

/// Unified result of evaluating the Earth-only conical shadow geometry in
/// exponent-separated arithmetic.
#[derive(Debug, Clone, Copy)]
struct ConicalGeometry {
    shadow: ScaledF64,
    nu: ScaledF64,
    shadow_grad_km: [ScaledF64; 3],
    nu_grad_km: [ScaledF64; 3],
    is_sunlit: bool,
}

impl ConicalGeometry {
    #[inline]
    fn sunlit() -> Self {
        Self {
            shadow: ScaledF64::ZERO,
            nu: ScaledF64::from_f64(1.0),
            shadow_grad_km: [ScaledF64::ZERO; 3],
            nu_grad_km: [ScaledF64::ZERO; 3],
            is_sunlit: true,
        }
    }

    #[inline]
    fn umbra_interior() -> Self {
        Self {
            shadow: ScaledF64::from_f64(1.0),
            nu: ScaledF64::ZERO,
            shadow_grad_km: [ScaledF64::ZERO; 3],
            nu_grad_km: [ScaledF64::ZERO; 3],
            is_sunlit: false,
        }
    }

    #[inline]
    fn antumbra_axis(need_grad: bool) -> Self {
        let grad = if need_grad {
            [ScaledF64 { m: f64::NAN, e: 0 }; 3]
        } else {
            [ScaledF64::ZERO; 3]
        };
        Self {
            shadow: ScaledF64::from_f64(1.0),
            nu: ScaledF64::ZERO,
            shadow_grad_km: grad,
            nu_grad_km: grad,
            is_sunlit: false,
        }
    }
}

/// Shared exponent-separated evaluation of the Earth-only conical shadow
/// geometry, used by `shadow_fraction*`, `shadow_fraction_with_*gradient`,
/// and the conical lighting ratio and gradient in ECOM1, ECOM2, and cannonball
/// SRP so every entry point agrees on every branch and value.
///
/// Forms `c = sun x sat` in the polar-scaled frame via exact compensated
/// `ScaledF64::lincomb2` with `z_scale` factored out of the exact products:
/// `c = (k (s_y r_z - s_z r_y), k (s_z r_x - s_x r_z), s_x r_y - s_y r_x)`.
/// Evaluates `rho = |c| / |sun|`, `u_perp = normalize(c x sun)`, and
/// `proj = (sat . sun) / |sun|` in the polar-scaled frame, evaluates both
/// `shadow` and `nu = 1 - shadow` directly from `rho`, and computes
/// `d(nu) = -d(shadow)` from the quotient rule of whichever of `nu` or
/// `shadow` is at most `1/2`.
fn conical_geometry_core(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    model: EarthShadowModel,
    need_grad: bool,
) -> Result<ConicalGeometry, EclipseError> {
    validate_nonzero_vec3(sat_pos, "sat_pos")?;
    validate_nonzero_vec3(sun_pos, "sun_pos")?;

    let z_scale = match model {
        EarthShadowModel::Spherical => 1.0,
        EarthShadowModel::Wgs84Oblate => {
            let sat_scaled = scale_polar_axis(sat_pos);
            let sun_scaled = scale_polar_axis(sun_pos);
            if !sat_scaled[2].is_finite() {
                return Err(invalid_eclipse_input("sat_pos", "out of range"));
            }
            if !sun_scaled[2].is_finite() {
                return Err(invalid_eclipse_input("sun_pos", "out of range"));
            }
            validate_nonzero_vec3(sat_scaled, "sat_pos")?;
            validate_nonzero_vec3(sun_scaled, "sun_pos")?;
            1.0 / (1.0 - WGS84_FLATTENING)
        }
    };

    let one = ScaledF64::from_f64(1.0);
    let k_s = ScaledF64::from_f64(z_scale);
    let r_x = ScaledF64::from_f64(sat_pos[0]);
    let r_y = ScaledF64::from_f64(sat_pos[1]);
    let r_z = ScaledF64::from_f64(sat_pos[2]);
    let s_x = ScaledF64::from_f64(sun_pos[0]);
    let s_y = ScaledF64::from_f64(sun_pos[1]);
    let s_z = ScaledF64::from_f64(sun_pos[2]);

    let (sat_v, sun_v) = match model {
        EarthShadowModel::Spherical => (
            ScaledVec3 {
                x: r_x,
                y: r_y,
                z: r_z,
            },
            ScaledVec3 {
                x: s_x,
                y: s_y,
                z: s_z,
            },
        ),
        EarthShadowModel::Wgs84Oblate => (
            ScaledVec3 {
                x: r_x,
                y: r_y,
                z: r_z.mul(k_s),
            },
            ScaledVec3 {
                x: s_x,
                y: s_y,
                z: s_z.mul(k_s),
            },
        ),
    };

    let d_sun_s = sun_v
        .norm()
        .ok_or_else(|| invalid_eclipse_input("sun_pos", "out of range"))?;
    let d_sun = d_sun_s.to_f64();
    if !d_sun.is_finite() || d_sun <= SOLAR_RADIUS_KM + MEAN_EARTH_RADIUS_KM {
        return Err(invalid_eclipse_input("sun_pos", "out of range"));
    }

    // Projection along the Earth-Sun axis in the scaled frame. Positive points
    // toward the Sun; a satellite on the sunlit side cannot be in shadow.
    let proj_s = sat_v.dot(sun_v).div(d_sun_s);
    if proj_s.m >= 0.0 {
        return Ok(ConicalGeometry::sunlit());
    }

    // Compensated scaled cross product c = sun x sat with z_scale factored out
    // of the exact products so c is identically zero iff the inputs are exactly
    // collinear.
    let cx_raw = ScaledF64::lincomb2(s_y, r_z, s_z.negate(), r_y);
    let cy_raw = ScaledF64::lincomb2(s_z, r_x, s_x.negate(), r_z);
    let cz = ScaledF64::lincomb2(s_x, r_y, s_y.negate(), r_x);
    let c = match model {
        EarthShadowModel::Spherical => ScaledVec3 {
            x: cx_raw,
            y: cy_raw,
            z: cz,
        },
        EarthShadowModel::Wgs84Oblate => ScaledVec3 {
            x: cx_raw.mul(k_s),
            y: cy_raw.mul(k_s),
            z: cz,
        },
    };
    let c_norm_opt = c.norm();
    let rho_s = match c_norm_opt {
        Some(c_norm) => c_norm.div(d_sun_s),
        None => ScaledF64::ZERO,
    };
    let dist_behind_s = proj_s.negate();

    let alpha_umbra = libm::asin((SOLAR_RADIUS_KM - MEAN_EARTH_RADIUS_KM) / d_sun);
    let alpha_penumbra = libm::asin((SOLAR_RADIUS_KM + MEAN_EARTH_RADIUS_KM) / d_sun);
    let tan_umbra_s = ScaledF64::from_f64(libm::tan(alpha_umbra));
    let tan_penumbra_s = ScaledF64::from_f64(libm::tan(alpha_penumbra));
    let r_earth_s = ScaledF64::from_f64(MEAN_EARTH_RADIUS_KM);

    let r_umbra_s = r_earth_s.sub(dist_behind_s.mul(tan_umbra_s));
    let r_penumbra_s = r_earth_s.add(dist_behind_s.mul(tan_penumbra_s));

    if !rho_s.to_f64().is_finite()
        || !r_umbra_s.to_f64().is_finite()
        || !r_penumbra_s.to_f64().is_finite()
    {
        return Err(invalid_eclipse_input("sat_pos", "out of range"));
    }

    // Beyond the penumbra cone: full sunlight.
    if rho_s.sub(r_penumbra_s).m >= 0.0 {
        return Ok(ConicalGeometry::sunlit());
    }
    // Inside the umbra cone: full shadow, zero lighting, zero gradient.
    if r_umbra_s.m > 0.0 && r_umbra_s.sub(rho_s).m >= 0.0 {
        return Ok(ConicalGeometry::umbra_interior());
    }

    let in_penumbra = r_umbra_s.m > 0.0;
    // Exact antumbra axis (rho == 0, r_umbra <= 0): shadow = 1, nu = 0,
    // gradient undefined.
    if !in_penumbra && rho_s.is_zero() {
        return Ok(ConicalGeometry::antumbra_axis(need_grad));
    }

    let (den_val, shadow_s, nu_s) = if in_penumbra {
        let den = r_penumbra_s.sub(r_umbra_s);
        let shadow = r_penumbra_s.sub(rho_s).div(den);
        let nu = rho_s.sub(r_umbra_s).div(den);
        (den, shadow, nu)
    } else {
        let den = r_penumbra_s;
        let shadow = r_penumbra_s.sub(rho_s).div(den);
        let nu = rho_s.div(den);
        (den, shadow, nu)
    };

    if !need_grad {
        return Ok(ConicalGeometry {
            shadow: shadow_s,
            nu: nu_s,
            shadow_grad_km: [ScaledF64::ZERO; 3],
            nu_grad_km: [ScaledF64::ZERO; 3],
            is_sunlit: false,
        });
    }

    // Unit perpendicular direction u_perp = normalize(c x sun) in the scaled
    // frame (since (sun x sat) x sun = |sun|^2 * perp).
    let Some((u_perp, _)) = c.cross(sun_v).normalize_scaled() else {
        return Ok(ConicalGeometry {
            shadow: shadow_s,
            nu: nu_s,
            shadow_grad_km: [ScaledF64 { m: f64::NAN, e: 0 }; 3],
            nu_grad_km: [ScaledF64 { m: f64::NAN, e: 0 }; 3],
            is_sunlit: false,
        });
    };

    let u_sun = sun_v.scalar_div(d_sun_s);
    let (rho_grad, proj_grad) = match model {
        EarthShadowModel::Spherical => {
            ([u_perp.x, u_perp.y, u_perp.z], [u_sun.x, u_sun.y, u_sun.z])
        }
        EarthShadowModel::Wgs84Oblate => (
            [u_perp.x, u_perp.y, u_perp.z.mul(k_s)],
            [u_sun.x, u_sun.y, u_sun.z.mul(k_s)],
        ),
    };

    let dist_behind_grad = [
        proj_grad[0].negate(),
        proj_grad[1].negate(),
        proj_grad[2].negate(),
    ];
    let r_umbra_grad = [
        dist_behind_grad[0].mul(tan_umbra_s).negate(),
        dist_behind_grad[1].mul(tan_umbra_s).negate(),
        dist_behind_grad[2].mul(tan_umbra_s).negate(),
    ];
    let r_penumbra_grad = [
        dist_behind_grad[0].mul(tan_penumbra_s),
        dist_behind_grad[1].mul(tan_penumbra_s),
        dist_behind_grad[2].mul(tan_penumbra_s),
    ];

    // Form d(nu) = -d(shadow) using the quotient rule of whichever of nu or
    // shadow has a value of at most 1/2, avoiding (1 - q) cancellation on both
    // the umbra/antumbra-axis side and the outer penumbra boundary.
    let use_nu_quotient = nu_s.is_zero() || nu_s.e < -1 || (nu_s.e == -1 && nu_s.m == 1.0);
    let (shadow_grad_km, nu_grad_km) = if use_nu_quotient {
        let mut nu_grad = [ScaledF64::ZERO; 3];
        let mut shadow_grad = [ScaledF64::ZERO; 3];
        for j in 0..3 {
            let (num_g, den_g) = if in_penumbra {
                (
                    rho_grad[j].sub(r_umbra_grad[j]),
                    r_penumbra_grad[j].sub(r_umbra_grad[j]),
                )
            } else {
                (rho_grad[j], r_penumbra_grad[j])
            };
            let g = ScaledF64::lincomb2(one, num_g, nu_s.negate(), den_g).div(den_val);
            nu_grad[j] = g;
            shadow_grad[j] = g.negate();
        }
        (shadow_grad, nu_grad)
    } else {
        let mut shadow_grad = [ScaledF64::ZERO; 3];
        let mut nu_grad = [ScaledF64::ZERO; 3];
        for j in 0..3 {
            let (num_g, den_g) = if in_penumbra {
                (
                    r_penumbra_grad[j].sub(rho_grad[j]),
                    r_penumbra_grad[j].sub(r_umbra_grad[j]),
                )
            } else {
                (r_penumbra_grad[j].sub(rho_grad[j]), r_penumbra_grad[j])
            };
            let g = ScaledF64::lincomb2(one, num_g, shadow_s.negate(), den_g).div(den_val);
            shadow_grad[j] = g;
            nu_grad[j] = g.negate();
        }
        (shadow_grad, nu_grad)
    };

    Ok(ConicalGeometry {
        shadow: shadow_s,
        nu: nu_s,
        shadow_grad_km,
        nu_grad_km,
        is_sunlit: false,
    })
}

/// Conical shadow fraction value in binary64 from the shared exponent-separated
/// geometry evaluation.
pub(crate) fn conical_shadow_value(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    model: EarthShadowModel,
) -> Result<f64, EclipseError> {
    conical_geometry_core(sat_pos, sun_pos, model, false).map(|c| c.shadow.to_f64())
}

/// Forward-mode `Grad` evaluation of the conical shadow fraction with respect
/// to `sat_pos` in km (`d(shadow)/d(sat_km)`).
pub(crate) fn conical_shadow_field(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    model: EarthShadowModel,
) -> Result<Grad, EclipseError> {
    let c = conical_geometry_core(sat_pos, sun_pos, model, true)?;
    Ok(Grad {
        value: c.shadow.to_f64(),
        grad: [
            c.shadow_grad_km[0].to_f64(),
            c.shadow_grad_km[1].to_f64(),
            c.shadow_grad_km[2].to_f64(),
        ],
    })
}

/// Conical lighting ratio `nu = 1 - shadow` in `ScaledF64` together with
/// `is_sunlit` (`true` iff in full sunlight where `nu == 1` with zero
/// gradient), bit-identical to the value part of
/// [`conical_lighting_ratio_field_scaled`].
pub(crate) fn conical_lighting_ratio_double_scaled(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    model: EarthShadowModel,
) -> Result<(ScaledF64, bool), EclipseError> {
    conical_geometry_core(sat_pos, sun_pos, model, false).map(|c| (c.nu, c.is_sunlit))
}

/// Conical lighting ratio `nu = 1 - shadow` and its gradient `d(nu)/d(sat_km)`
/// in `ScaledGrad<3>` (1/km) together with `is_sunlit` (`true` iff in full
/// sunlight where `nu == 1` with zero gradient).
pub(crate) fn conical_lighting_ratio_field_scaled(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    model: EarthShadowModel,
) -> Result<(ScaledGrad<3>, bool), EclipseError> {
    let c = conical_geometry_core(sat_pos, sun_pos, model, true)?;
    Ok((
        ScaledGrad {
            val: c.nu,
            grad: c.nu_grad_km,
        },
        c.is_sunlit,
    ))
}

fn finish_grad(g: Grad) -> Result<(f64, [f64; 3]), ShadowGradientError> {
    if !g.is_all_finite() {
        return Err(ShadowGradientError::NumericalFailure);
    }
    Ok((g.value, g.grad))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::constants::astro::AU_KM;
    use crate::astro::events::eclipse::{shadow_fraction, shadow_fraction_with_model};
    use crate::astro::math::vec3;

    fn central_diff_4(f: impl Fn([f64; 3]) -> f64, sat: [f64; 3], h: f64) -> [f64; 3] {
        let mut out = [0.0; 3];
        for axis in 0..3 {
            let at = |s: f64| {
                let mut p = sat;
                p[axis] += s * h;
                f(p)
            };
            out[axis] = (at(-2.0) - 8.0 * at(-1.0) + 8.0 * at(1.0) - at(2.0)) / (12.0 * h);
        }
        out
    }

    #[test]
    fn conical_gradients_match_central_differences_in_penumbra_and_antumbra() {
        let sun = [AU_KM, 1.0e7, -2.0e7];
        let sun_n = vec3::norm3(sun);
        let sun_u = [sun[0] / sun_n, sun[1] / sun_n, sun[2] / sun_n];
        let side = vec3::cross3(sun_u, [0.0, 0.0, 1.0]);
        let side_n = vec3::norm3(side);
        let side_u = [side[0] / side_n, side[1] / side_n, side[2] / side_n];
        let up_u = vec3::cross3(side_u, sun_u);
        // 26000 km behind Earth and 6360 km off the shadow axis (well inside the
        // ~240 km penumbra band for both spherical and oblate Earth).
        let sat_pen = [
            -26_000.0 * sun_u[0] + 5_500.0 * side_u[0] + 3_200.0 * up_u[0],
            -26_000.0 * sun_u[1] + 5_500.0 * side_u[1] + 3_200.0 * up_u[1],
            -26_000.0 * sun_u[2] + 5_500.0 * side_u[2] + 3_200.0 * up_u[2],
        ];
        for model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
            let (f, g) =
                shadow_fraction_with_model_and_gradient(sat_pen, sun, model).expect("penumbra");
            let f_val = shadow_fraction_with_model(sat_pen, sun, model).expect("value");
            assert_eq!(f.to_bits(), f_val.to_bits());
            assert!(f > 0.0 && f < 1.0, "expected penumbra, got {f}");
            let num = central_diff_4(
                |p| shadow_fraction_with_model(p, sun, model).expect("val"),
                sat_pen,
                1.0e-3,
            );
            for i in 0..3 {
                assert!(
                    (g[i] - num[i]).abs() <= 1.0e-8 * g[i].abs().max(1.0e-12),
                    "{model:?} axis {i}: ad {} vs num {}",
                    g[i],
                    num[i]
                );
            }

            // Antumbra (past umbra tip, off axis).
            let sat_ant = [-2.0e6, 5_000.0, 3_000.0];
            let (f_ant, g_ant) =
                shadow_fraction_with_model_and_gradient(sat_ant, [AU_KM, 0.0, 0.0], model)
                    .expect("antumbra");
            let f_ant_val =
                shadow_fraction_with_model(sat_ant, [AU_KM, 0.0, 0.0], model).expect("ant val");
            assert_eq!(f_ant.to_bits(), f_ant_val.to_bits());
            assert!(f_ant > 0.0 && f_ant < 1.0);
            let num_ant = central_diff_4(
                |p| shadow_fraction_with_model(p, [AU_KM, 0.0, 0.0], model).expect("val"),
                sat_ant,
                1.0e-2,
            );
            for i in 0..3 {
                assert!(
                    (g_ant[i] - num_ant[i]).abs() <= 1.0e-8 * g_ant[i].abs().max(1.0e-12),
                    "antumbra {model:?} axis {i}: ad {} vs num {}",
                    g_ant[i],
                    num_ant[i]
                );
            }
        }
    }

    #[test]
    fn conical_gradient_value_matches_shadow_fraction_zero_ulp_across_regimes() {
        let suns = [
            [AU_KM, 0.0, 0.0],
            [1.2e8, -8.0e7, -3.5e7],
            [-0.7e8, 0.9e8, 1.0e8],
        ];
        // Distances behind Earth spanning LEO, GNSS, GEO (penumbra) and distant
        // positions past the umbra tip (~1.38e6 km, antumbra).
        let distances_behind = [6_878.0, 26_560.0, 42_164.0, 2.0e6, 5.0e6];
        let mut penumbra_hits = 0usize;
        let mut antumbra_hits = 0usize;

        for sun in suns {
            let d_sun = vec3::norm3(sun);
            let sun_u = [sun[0] / d_sun, sun[1] / d_sun, sun[2] / d_sun];
            let ref_v = if sun_u[2].abs() < 0.9 {
                [0.0, 0.0, 1.0]
            } else {
                [1.0, 0.0, 0.0]
            };
            let side_raw = vec3::cross3(sun_u, ref_v);
            let side_n = vec3::norm3(side_raw);
            let side_u = [
                side_raw[0] / side_n,
                side_raw[1] / side_n,
                side_raw[2] / side_n,
            ];
            let up_u = vec3::cross3(side_u, sun_u);
            let alpha_u = libm::asin((SOLAR_RADIUS_KM - MEAN_EARTH_RADIUS_KM) / d_sun);
            let alpha_p = libm::asin((SOLAR_RADIUS_KM + MEAN_EARTH_RADIUS_KM) / d_sun);
            let tan_u = libm::tan(alpha_u);
            let tan_p = libm::tan(alpha_p);

            for dist in distances_behind {
                let r_u = MEAN_EARTH_RADIUS_KM - dist * tan_u;
                let r_p = MEAN_EARTH_RADIUS_KM + dist * tan_p;
                let lo = r_u.max(0.0);
                for k in 1..=9 {
                    let frac = (k as f64) / 10.0;
                    let rho = lo + frac * (r_p - lo);
                    let az = 0.73 * (k as f64);
                    let (s_az, c_az) = libm::sincos(az);
                    let sat = [
                        -dist * sun_u[0] + rho * (c_az * side_u[0] + s_az * up_u[0]),
                        -dist * sun_u[1] + rho * (c_az * side_u[1] + s_az * up_u[1]),
                        -dist * sun_u[2] + rho * (c_az * side_u[2] + s_az * up_u[2]),
                    ];
                    let f_sph = shadow_fraction(sat, sun).expect("shadow_fraction");
                    let (fg_sph, _) = shadow_fraction_with_gradient(sat, sun).expect("grad");
                    assert_eq!(
                        fg_sph.to_bits(),
                        f_sph.to_bits(),
                        "spherical dist={dist} k={k}: 0x{:016x} != 0x{:016x}",
                        fg_sph.to_bits(),
                        f_sph.to_bits()
                    );
                    if r_u > 0.0 && f_sph > 0.0 && f_sph < 1.0 {
                        penumbra_hits += 1;
                    } else if r_u <= 0.0 && f_sph > 0.0 && f_sph < 1.0 {
                        antumbra_hits += 1;
                    }
                    for model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
                        let f_m = shadow_fraction_with_model(sat, sun, model).expect("model");
                        let (fg_m, _) = shadow_fraction_with_model_and_gradient(sat, sun, model)
                            .expect("model grad");
                        assert_eq!(
                            fg_m.to_bits(),
                            f_m.to_bits(),
                            "{model:?} dist={dist} k={k}: 0x{:016x} != 0x{:016x}",
                            fg_m.to_bits(),
                            f_m.to_bits()
                        );
                    }
                }
            }
        }
        assert!(
            penumbra_hits >= 50,
            "expected penumbra coverage, got {penumbra_hits}"
        );
        assert!(
            antumbra_hits >= 30,
            "expected antumbra coverage, got {antumbra_hits}"
        );
    }

    #[test]
    fn conical_gradient_zero_in_sunlight_and_umbra_and_refuses_antumbra_axis() {
        let sun = [AU_KM, 0.0, 0.0];
        assert_eq!(
            shadow_fraction_with_gradient([7_000.0, 0.0, 0.0], sun).expect("sunlit"),
            (0.0, [0.0; 3])
        );
        assert_eq!(
            shadow_fraction_with_gradient([-7_000.0, 0.0, 0.0], sun).expect("umbra"),
            (1.0, [0.0; 3])
        );
        // On the exact antumbra axis past the umbra tip, rho == 0 produces NaN in sqrt'.
        assert_eq!(
            shadow_fraction_with_gradient([-2.0e6, 0.0, 0.0], sun),
            Err(ShadowGradientError::NumericalFailure)
        );
        // Matches shadow_fraction on sunlit and umbra.
        assert_eq!(
            shadow_fraction([-7_000.0, 0.0, 0.0], sun).expect("val"),
            1.0
        );
    }

    #[test]
    fn conical_exact_and_near_axis_non_axis_aligned_sun_and_along_axis_gradient() {
        let r_earth = MEAN_EARTH_RADIUS_KM;
        let r_sun = SOLAR_RADIUS_KM;

        // 1. Exact antumbra axis with non-axis-aligned Sun: Spherical and WGS84.
        let exact_axis_cases = [
            ([-2.0e6, -3.0e6, 0.0], [2.0e8, 3.0e8, 0.0]),
            ([-1.0e6, -2.0e6, -5.0e6], [1.0e8, 2.0e8, 5.0e8]),
        ];
        for (sat, sun) in exact_axis_cases {
            for model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
                assert_eq!(
                    shadow_fraction_with_model(sat, sun, model).expect("exact axis shadow"),
                    1.0,
                    "{model:?} sat={sat:?} sun={sun:?}"
                );
                assert_eq!(
                    shadow_fraction_with_model_and_gradient(sat, sun, model),
                    Err(ShadowGradientError::NumericalFailure),
                    "{model:?} sat={sat:?} sun={sun:?}"
                );
                let (nu_d, is_sunlit) =
                    conical_lighting_ratio_double_scaled(sat, sun, model).expect("nu double");
                assert!(nu_d.is_zero() && !is_sunlit);
                let (nu_f, is_sunlit_f) =
                    conical_lighting_ratio_field_scaled(sat, sun, model).expect("nu field");
                assert!(nu_f.val.is_zero() && !is_sunlit_f && !nu_f.is_all_finite());
            }
        }

        // 2. Near-axis antumbra with non-axis-aligned Sun:
        //    sun = [2e8, 3e8, 0], sat = [-2e6, -3e6, 1e-12] (Spherical and WGS84).
        let sun_na = [2.0e8, 3.0e8, 0.0];
        let sat_na = [-2.0e6, -3.0e6, 1.0e-12];
        let d_sun_na = libm::hypot(2.0e8, 3.0e8);
        let dist_behind_na = libm::hypot(2.0e6, 3.0e6);
        let tan_p_na = libm::tan(libm::asin((r_sun + r_earth) / d_sun_na));
        let r_pen_na = r_earth + dist_behind_na * tan_p_na;
        let u_sun_na = [2.0e8 / d_sun_na, 3.0e8 / d_sun_na, 0.0];

        for model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
            let k = match model {
                EarthShadowModel::Spherical => 1.0,
                EarthShadowModel::Wgs84Oblate => 1.0 / (1.0 - WGS84_FLATTENING),
            };
            let rho_ref = 1.0e-12 * k;
            let nu_ref = rho_ref / r_pen_na;
            let grad_nu_ref = [
                nu_ref * tan_p_na / r_pen_na * u_sun_na[0],
                nu_ref * tan_p_na / r_pen_na * u_sun_na[1],
                k / r_pen_na,
            ];
            let (sh, g_sh) =
                shadow_fraction_with_model_and_gradient(sat_na, sun_na, model).expect("near axis");
            let sh_val = shadow_fraction_with_model(sat_na, sun_na, model).expect("near axis val");
            assert_eq!(sh.to_bits(), sh_val.to_bits());
            let (nu_f, is_sunlit) =
                conical_lighting_ratio_field_scaled(sat_na, sun_na, model).expect("nu near axis");
            assert!(!is_sunlit);
            let nu_rel = (nu_f.val.to_f64() - nu_ref).abs() / nu_ref;
            assert!(
                nu_rel <= 8.0 * f64::EPSILON,
                "{model:?}: nu={} vs ref={nu_ref:e} (rel={nu_rel:e})",
                nu_f.val.to_f64()
            );
            for j in 0..3 {
                assert_eq!(
                    g_sh[j].to_bits(),
                    (-nu_f.grad[j].to_f64()).to_bits(),
                    "{model:?} axis {j}: d(shadow) != -d(nu)"
                );
                let rel = (nu_f.grad[j].to_f64() - grad_nu_ref[j]).abs() / grad_nu_ref[j].abs();
                assert!(
                    rel <= 8.0 * f64::EPSILON,
                    "{model:?} axis {j}: grad_nu={:e} vs ref={:e} (rel={rel:e})",
                    nu_f.grad[j].to_f64(),
                    grad_nu_ref[j]
                );
            }
        }

        // 3. Along-axis gradient in near-axis antumbra where binary64 q_val loses
        //    rho / r_penumbra or rounds to 1.0 (both Spherical and WGS84).
        let sun_au = [AU_KM, 0.0, 0.0];
        let tan_p_au = libm::tan(libm::asin((r_sun + r_earth) / AU_KM));
        let r_pen_au = r_earth + 2.0e6 * tan_p_au;
        for rho_y in [5.0e-13_f64, 1.0e-12_f64] {
            let sat_q = [-2.0e6, rho_y, 0.0];
            let expected_gx = -(rho_y / r_pen_au) * tan_p_au / r_pen_au;
            let expected_gy = -1.0 / r_pen_au;
            for model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
                let (sh, g) =
                    shadow_fraction_with_model_and_gradient(sat_q, sun_au, model).expect("q_val");
                let sh_val = shadow_fraction_with_model(sat_q, sun_au, model).expect("q_val val");
                assert_eq!(sh.to_bits(), sh_val.to_bits());
                let rel_x = (g[0] - expected_gx).abs() / expected_gx.abs();
                let rel_y = (g[1] - expected_gy).abs() / expected_gy.abs();
                assert!(
                    rel_x <= 8.0 * f64::EPSILON,
                    "{model:?} rho={rho_y:e}: g[0]={:e} vs {expected_gx:e} (rel={rel_x:e})",
                    g[0]
                );
                assert!(
                    rel_y <= 8.0 * f64::EPSILON,
                    "{model:?} rho={rho_y:e}: g[1]={:e} vs {expected_gy:e} (rel={rel_y:e})",
                    g[1]
                );
                assert_eq!(g[2], 0.0);
            }
        }

        // 4. Outer penumbra boundary (shadow <= 1/2 branch of quotient rule):
        //    1 ULP inside r_penumbra so r_pen_p - rho_outer is exact in binary64.
        let dist_pen = 26_560.0_f64;
        let tan_u_au = libm::tan(libm::asin((r_sun - r_earth) / AU_KM));
        let r_umb_p = r_earth - dist_pen * tan_u_au;
        let r_pen_p = r_earth + dist_pen * tan_p_au;
        let rho_outer = r_pen_p.next_down();
        let delta_outer = r_pen_p - rho_outer;
        let sat_outer = [-dist_pen, rho_outer, 0.0];
        let sh_ref_outer = delta_outer / (r_pen_p - r_umb_p);
        let expected_gx_outer =
            (-tan_p_au - sh_ref_outer * (-tan_p_au - tan_u_au)) / (r_pen_p - r_umb_p);
        let expected_gy_outer = -1.0 / (r_pen_p - r_umb_p);
        let (sh_o, g_o) = shadow_fraction_with_gradient(sat_outer, sun_au).expect("outer penumbra");
        assert!(
            (sh_o - sh_ref_outer).abs() <= 16.0 * f64::EPSILON * sh_ref_outer,
            "outer penumbra sh={sh_o:e} vs {sh_ref_outer:e}"
        );
        assert!(
            (g_o[0] - expected_gx_outer).abs() <= 16.0 * f64::EPSILON * expected_gx_outer.abs(),
            "outer penumbra g[0]={:e} vs {expected_gx_outer:e}",
            g_o[0]
        );
        assert!(
            (g_o[1] - expected_gy_outer).abs() <= 16.0 * f64::EPSILON * expected_gy_outer.abs(),
            "outer penumbra g[1]={:e} vs {expected_gy_outer:e}",
            g_o[1]
        );
    }

    #[test]
    fn conical_near_axis_and_exact_axis_non_axis_aligned_sweep() {
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
                ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
            }
            fn log_uniform(&mut self, lo_exp2: i32, hi_exp2: i32) -> f64 {
                let e = lo_exp2 as f64 + (hi_exp2 - lo_exp2) as f64 * self.unit();
                let ei = e.floor() as i32;
                libm::ldexp(libm::exp2(e - ei as f64), ei)
            }
        }

        let mut rng = SplitMix64(0xc0a1_ca10_2026_0009);
        let r_earth = MEAN_EARTH_RADIUS_KM;
        let r_sun = SOLAR_RADIUS_KM;
        let mut exact_antumbra_hits = 0usize;
        let mut exact_umbra_hits = 0usize;
        let mut near_axis_hits = 0usize;
        let mut max_nu_eps = 0.0_f64;
        let mut max_grad_eps = 0.0_f64;

        for case in 0..256 {
            // Non-axis-aligned exact integer Sun direction scaled to ~0.5..2 AU.
            let a = ((rng.next_u64() % 19) + 1) as f64
                * if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
            let b = ((rng.next_u64() % 19) + 1) as f64
                * if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
            let c = ((rng.next_u64() % 19) + 1) as f64
                * if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
            let sun_exp = 23 + (rng.next_u64() % 3) as i32;
            let sun = [
                libm::ldexp(a, sun_exp),
                libm::ldexp(b, sun_exp),
                libm::ldexp(c, sun_exp),
            ];

            for model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
                let k = match model {
                    EarthShadowModel::Spherical => 1.0,
                    EarthShadowModel::Wgs84Oblate => 1.0 / (1.0 - WGS84_FLATTENING),
                };
                let sun_s = [sun[0], sun[1], sun[2] * k];
                let d_sun = vec3::norm3(sun_s);
                let tan_u = libm::tan(libm::asin((r_sun - r_earth) / d_sun));

                // 1. Exact antumbra axis (sat = -2^-6 * sun, ~2..10e6 km behind Earth):
                //    exact collinearity in binary64, r_umbra < 0.
                let sat_ant_axis = [
                    libm::ldexp(-sun[0], -6),
                    libm::ldexp(-sun[1], -6),
                    libm::ldexp(-sun[2], -6),
                ];
                let dist_ant = libm::ldexp(d_sun, -6);
                if r_earth - dist_ant * tan_u <= 0.0 {
                    assert_eq!(
                        shadow_fraction_with_model(sat_ant_axis, sun, model).expect("ant axis"),
                        1.0
                    );
                    assert_eq!(
                        shadow_fraction_with_model_and_gradient(sat_ant_axis, sun, model),
                        Err(ShadowGradientError::NumericalFailure)
                    );
                    let (nu_d, sunlit_d) =
                        conical_lighting_ratio_double_scaled(sat_ant_axis, sun, model)
                            .expect("ant axis nu");
                    assert!(nu_d.is_zero() && !sunlit_d);
                    let (nu_f, sunlit_f) =
                        conical_lighting_ratio_field_scaled(sat_ant_axis, sun, model)
                            .expect("ant axis nu field");
                    assert!(nu_f.val.is_zero() && !sunlit_f && !nu_f.is_all_finite());
                    exact_antumbra_hits += 1;
                }

                // 2. Exact umbra axis (sat = -2^-13 * sun, ~20,000 km behind Earth):
                //    r_umbra > 0 -> shadow = 1, nu = 0, zero gradient.
                let sat_umb_axis = [
                    libm::ldexp(-sun[0], -13),
                    libm::ldexp(-sun[1], -13),
                    libm::ldexp(-sun[2], -13),
                ];
                let dist_umb = libm::ldexp(d_sun, -13);
                if r_earth - dist_umb * tan_u > 0.0 {
                    assert_eq!(
                        shadow_fraction_with_model_and_gradient(sat_umb_axis, sun, model)
                            .expect("umb axis"),
                        (1.0, [0.0; 3])
                    );
                    let (nu_f, sunlit_f) =
                        conical_lighting_ratio_field_scaled(sat_umb_axis, sun, model)
                            .expect("umb axis nu field");
                    assert!(
                        nu_f.val.is_zero() && !sunlit_f && nu_f.grad.iter().all(|g| g.is_zero())
                    );
                    exact_umbra_hits += 1;
                }

                // 3. Near-axis antumbra with exact off-axis offset:
                //    Pick an exact integer offset vector delta = (-b, a, 0) (exact dot with sun = 0
                //    in the xy plane, so delta_z == 0 and delta . sun_s == 0 identically under BOTH
                //    Spherical and WGS84!), scaled by a power-of-two eps from 2^-1050 to 2^-20 so
                //    sat_ant_axis + eps * delta is exact whenever eps * |delta| >= 2^-52 * |sat|,
                //    and for smaller eps perturb ONLY z with sun_z == 0.
                let (sun_na, sat_base, delta_exact, delta_scale) = if case % 2 == 0 {
                    // 2D non-axis-aligned Sun in xy-plane with pure-z offset down to 2^-1050:
                    // adding pure z to sat_base is exact at EVERY exponent down to subnormals!
                    let s_xy = [sun[0], sun[1], 0.0];
                    let r_base = [libm::ldexp(-sun[0], -6), libm::ldexp(-sun[1], -6), 0.0];
                    let dz = rng.log_uniform(-1050, -30);
                    (s_xy, r_base, [0.0, 0.0, 1.0], dz)
                } else {
                    // Full 3D non-axis-aligned Sun with exact orthogonal xy offset (-b, a, 0)
                    // at scales large enough that r_base + eps * (-b, a, 0) does not round away.
                    let shift = -36 + (rng.next_u64() % 28) as i32; // -36..-9
                    let d_xy = [libm::ldexp(-b, shift), libm::ldexp(a, shift), 0.0];
                    let r_base = sat_ant_axis;
                    let sat_sum = [
                        r_base[0] + d_xy[0],
                        r_base[1] + d_xy[1],
                        r_base[2] + d_xy[2],
                    ];
                    // Actual realized offset after binary64 addition:
                    let real_d = [
                        sat_sum[0] - r_base[0],
                        sat_sum[1] - r_base[1],
                        sat_sum[2] - r_base[2],
                    ];
                    (sun, r_base, real_d, 1.0)
                };
                let sat_near = [
                    sat_base[0] + delta_exact[0] * delta_scale,
                    sat_base[1] + delta_exact[1] * delta_scale,
                    sat_base[2] + delta_exact[2] * delta_scale,
                ];
                let sun_na_s = [sun_na[0], sun_na[1], sun_na[2] * k];
                let d_sun_na = vec3::norm3(sun_na_s);
                let tan_u_na = libm::tan(libm::asin((r_sun - r_earth) / d_sun_na));
                let tan_p_na = libm::tan(libm::asin((r_sun + r_earth) / d_sun_na));
                let u_sun_na = [
                    sun_na_s[0] / d_sun_na,
                    sun_na_s[1] / d_sun_na,
                    sun_na_s[2] / d_sun_na,
                ];
                let k_s = ScaledF64::from_f64(k);
                let u_sun_s = [
                    ScaledF64::from_f64(u_sun_na[0]),
                    ScaledF64::from_f64(u_sun_na[1]),
                    ScaledF64::from_f64(u_sun_na[2]),
                ];
                // Independent reference from the exact perturbation delta_s in scaled frame
                // (computed in ScaledF64 so subnormal dz * k does not lose mantissa bits).
                let delta_scale_s = ScaledF64::from_f64(delta_scale);
                let delta_s = ScaledVec3 {
                    x: ScaledF64::from_f64(delta_exact[0]).mul(delta_scale_s),
                    y: ScaledF64::from_f64(delta_exact[1]).mul(delta_scale_s),
                    z: ScaledF64::from_f64(delta_exact[2])
                        .mul(delta_scale_s)
                        .mul(k_s),
                };
                let u_sun_v = ScaledVec3 {
                    x: u_sun_s[0],
                    y: u_sun_s[1],
                    z: u_sun_s[2],
                };
                let d_proj_s = delta_s.dot(u_sun_v);
                let d_perp_s = delta_s.sub(u_sun_v.scalar_mul(d_proj_s));
                let Some((u_perp_ref, rho_ref_s)) = d_perp_s.normalize_scaled() else {
                    continue;
                };
                let dist_behind_ref = libm::ldexp(d_sun_na, -6) - d_proj_s.to_f64();
                let r_umb_ref = r_earth - dist_behind_ref * tan_u_na;
                let r_pen_ref = r_earth + dist_behind_ref * tan_p_na;
                if r_umb_ref > 0.0 || rho_ref_s.to_f64() >= r_pen_ref * 0.5 {
                    continue;
                }
                let r_pen_s = ScaledF64::from_f64(r_pen_ref);
                let nu_ref_s = rho_ref_s.div(r_pen_s);
                let tan_p_s = ScaledF64::from_f64(tan_p_na);
                let u_perp_arr = [u_perp_ref.x, u_perp_ref.y, u_perp_ref.z];
                let mut g_nu_ref = [ScaledF64::ZERO; 3];
                for j in 0..3 {
                    let cj = if j == 2 {
                        k_s
                    } else {
                        ScaledF64::from_f64(1.0)
                    };
                    g_nu_ref[j] = u_perp_arr[j]
                        .add(nu_ref_s.mul(tan_p_s).mul(u_sun_s[j]))
                        .mul(cj)
                        .div(r_pen_s);
                }

                let sh_val = shadow_fraction_with_model(sat_near, sun_na, model).expect("near val");
                let (sh_g, g_sh) = shadow_fraction_with_model_and_gradient(sat_near, sun_na, model)
                    .expect("near grad");
                assert_eq!(sh_val.to_bits(), sh_g.to_bits());
                let (nu_d, sunlit_d) =
                    conical_lighting_ratio_double_scaled(sat_near, sun_na, model).expect("nu d");
                let (nu_f, sunlit_f) =
                    conical_lighting_ratio_field_scaled(sat_near, sun_na, model).expect("nu f");
                assert!(!sunlit_d && !sunlit_f);
                assert_eq!(
                    (nu_d.m.to_bits(), nu_d.e),
                    (nu_f.val.m.to_bits(), nu_f.val.e)
                );

                let nu_rel = nu_f.val.sub(nu_ref_s).div(nu_ref_s).to_f64().abs();
                max_nu_eps = max_nu_eps.max(nu_rel / f64::EPSILON);
                for j in 0..3 {
                    assert_eq!(g_sh[j].to_bits(), (-nu_f.grad[j].to_f64()).to_bits());
                    if !g_nu_ref[j].is_zero() {
                        let g_rel = nu_f.grad[j]
                            .sub(g_nu_ref[j])
                            .div(g_nu_ref[j])
                            .to_f64()
                            .abs();
                        // Only check components not dominated by orthogonal projection rounding
                        // when delta_exact was non-orthogonal after binary64 addition.
                        if case % 2 == 0 || g_nu_ref[j].e >= -20 {
                            max_grad_eps = max_grad_eps.max(g_rel / f64::EPSILON);
                        }
                    }
                }
                near_axis_hits += 1;
            }
        }

        assert!(
            exact_antumbra_hits >= 400 && exact_umbra_hits >= 400 && near_axis_hits >= 400,
            "insufficient coverage: ant={exact_antumbra_hits}, umb={exact_umbra_hits}, near={near_axis_hits}"
        );
        // Measured maxima across 512 non-axis-aligned cases (Spherical & WGS84):
        // nu = 2.48 eps, grad = 5.02 eps (ScaledF64 rounding in norm,
        // compensated cross products, r_penumbra, and quotient division).
        assert!(
            max_nu_eps <= 16.0 && max_grad_eps <= 16.0,
            "conical non-axis-aligned sweep maxima: nu={max_nu_eps:.2} eps, grad={max_grad_eps:.2} eps"
        );
    }

    #[test]
    fn apparent_disc_gradient_matches_central_differences_across_regimes() {
        let sun = [1.2e8, -8.0e7, -3.5e7];
        let far_moon = [-1.0e8, 5.0e7, 2.0e7];
        // LEO penumbra over high latitude (exercises WGS84 oblate limb).
        let u_sun = {
            let n = vec3::norm3(sun);
            [sun[0] / n, sun[1] / n, sun[2] / n]
        };
        let z_axis = [0.0, 0.0, 1.0];
        let side = vec3::cross3(u_sun, z_axis);
        let side_n = vec3::norm3(side);
        let side = [side[0] / side_n, side[1] / side_n, side[2] / side_n];
        let toward_pole = vec3::cross3(side, u_sun);
        for (model, limb_r) in [
            (EarthShadowModel::Spherical, 6_371.0),
            (EarthShadowModel::Wgs84Oblate, 6_352.0),
        ] {
            let theta = std::f64::consts::PI - libm::asin(limb_r / 7_000.0);
            let sat = [
                7_000.0 * (libm::cos(theta) * u_sun[0] + libm::sin(theta) * toward_pole[0]),
                7_000.0 * (libm::cos(theta) * u_sun[1] + libm::sin(theta) * toward_pole[1]),
                7_000.0 * (libm::cos(theta) * u_sun[2] + libm::sin(theta) * toward_pole[2]),
            ];
            let (f, g) = shadow_fraction_with_model_moon_and_gradient(sat, sun, far_moon, model)
                .expect("earth penumbra");
            assert!(f > 0.05 && f < 0.95, "{model:?}: f={f}");
            let num = central_diff_4(
                |p| {
                    crate::astro::events::eclipse::shadow_fraction_with_model_and_moon(
                        p, sun, far_moon, model,
                    )
                    .expect("val")
                },
                sat,
                5.0e-2,
            );
            let scale = vec3::norm3(g);
            let err = vec3::norm3(vec3::sub3(g, num)) / scale;
            assert!(
                err < 5.0e-7,
                "{model:?}: rel err {err:e}, ad={g:?}, num={num:?}"
            );
        }
    }
}
