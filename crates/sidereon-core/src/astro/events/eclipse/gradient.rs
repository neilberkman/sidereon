//! Shadow fractions with their forward-mode automatic differentiation gradients
//! with respect to the satellite position (per km), reproducing Hipparchus
//! 4.0.3 `Gradient` arithmetic and Orekit 13.1.10's `FieldSpacecraftState`
//! branch decisions.

use super::ad::Grad;
use super::orekit::earth_moon_lighting_ratio_field;
use super::{
    invalid_eclipse_input, scale_polar_axis, validate_nonzero_vec3, EarthShadowModel, EclipseError,
    WGS84_FLATTENING,
};
use crate::astro::constants::{
    astro::SOLAR_RADIUS_KM, earth::MEAN_EARTH_RADIUS_KM, units::M_PER_KM,
};
use crate::astro::math::vec3;

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
/// Differentiates the established linear conical shadow arithmetic directly
/// (using IEEE-754 division for the value with the quotient rule for the
/// derivative, so the returned shadow value is bit-identical to
/// [`shadow_fraction`](super::shadow_fraction)), choosing branches on the real
/// value as `shadow_fraction` does (yielding the active branch's one-sided
/// derivative at cone boundaries). Returns
/// [`ShadowGradientError::NumericalFailure`] if evaluation produces a
/// non-finite derivative (such as on the antumbra axis past the umbra tip).
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

/// Forward-mode `Grad` evaluation of the conical shadow fraction with respect
/// to `sat_pos` in km (`d(shadow)/d(sat_km)`).
pub(crate) fn conical_shadow_field(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    model: EarthShadowModel,
) -> Result<Grad, EclipseError> {
    validate_nonzero_vec3(sat_pos, "sat_pos")?;
    validate_nonzero_vec3(sun_pos, "sun_pos")?;

    let (sat_g, sun_scaled) = match model {
        EarthShadowModel::Spherical => (
            [
                Grad::variable(0, sat_pos[0]),
                Grad::variable(1, sat_pos[1]),
                Grad::variable(2, sat_pos[2]),
            ],
            sun_pos,
        ),
        EarthShadowModel::Wgs84Oblate => {
            let z_scale = 1.0 / (1.0 - WGS84_FLATTENING);
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
            let sat_z = Grad::variable(2, sat_pos[2]).mul_f64(z_scale);
            (
                [
                    Grad::variable(0, sat_pos[0]),
                    Grad::variable(1, sat_pos[1]),
                    sat_z,
                ],
                sun_scaled,
            )
        }
    };

    let d_sun = vec3::norm3(sun_scaled);
    if d_sun <= SOLAR_RADIUS_KM + MEAN_EARTH_RADIUS_KM {
        return Err(invalid_eclipse_input("sun_pos", "out of range"));
    }

    let sun_u = [
        sun_scaled[0] / d_sun,
        sun_scaled[1] / d_sun,
        sun_scaled[2] / d_sun,
    ];

    let proj = sat_g[0]
        .mul_f64(sun_u[0])
        .add(sat_g[1].mul_f64(sun_u[1]))
        .add(sat_g[2].mul_f64(sun_u[2]));
    if proj.value >= 0.0 {
        return Ok(Grad::ZERO);
    }

    let perp = [
        sat_g[0].sub(proj.mul_f64(sun_u[0])),
        sat_g[1].sub(proj.mul_f64(sun_u[1])),
        sat_g[2].sub(proj.mul_f64(sun_u[2])),
    ];
    let rho = perp[0]
        .square()
        .add(perp[1].square())
        .add(perp[2].square())
        .sqrt();

    let dist_behind = proj.negate();

    let alpha_umbra = libm::asin((SOLAR_RADIUS_KM - MEAN_EARTH_RADIUS_KM) / d_sun);
    let alpha_penumbra = libm::asin((SOLAR_RADIUS_KM + MEAN_EARTH_RADIUS_KM) / d_sun);
    let tan_umbra = libm::tan(alpha_umbra);
    let tan_penumbra = libm::tan(alpha_penumbra);

    let r_umbra = Grad::constant(MEAN_EARTH_RADIUS_KM).sub(dist_behind.mul_f64(tan_umbra));
    let r_penumbra = Grad::constant(MEAN_EARTH_RADIUS_KM).add(dist_behind.mul_f64(tan_penumbra));

    if !rho.value.is_finite() || !r_umbra.value.is_finite() || !r_penumbra.value.is_finite() {
        return Err(invalid_eclipse_input("sat_pos", "out of range"));
    }

    if rho.value >= r_penumbra.value {
        Ok(Grad::ZERO)
    } else if r_umbra.value > 0.0 && rho.value <= r_umbra.value {
        Ok(Grad::ONE)
    } else if r_umbra.value > 0.0 {
        Ok(ieee_div(r_penumbra.sub(rho), r_penumbra.sub(r_umbra)))
    } else {
        let frac = ieee_div(r_penumbra.sub(rho), r_penumbra);
        if frac.value <= 0.0 {
            Ok(Grad::ZERO)
        } else {
            Ok(frac)
        }
    }
}

/// Quotient rule using direct IEEE-754 division `num / den` for the value so
/// conical shadow values are bit-identical to [`super::shadow_fraction`] and
/// [`super::shadow_fraction_with_model`].
#[inline]
fn ieee_div(num: Grad, den: Grad) -> Grad {
    let q = num.value / den.value;
    Grad {
        value: q,
        grad: [
            (num.grad[0] - q * den.grad[0]) / den.value,
            (num.grad[1] - q * den.grad[1]) / den.value,
            (num.grad[2] - q * den.grad[2]) / den.value,
        ],
    }
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
