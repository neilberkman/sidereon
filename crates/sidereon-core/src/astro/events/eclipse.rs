//! Conical Earth and Moon shadow (eclipse) geometry.
//!
//! Determines whether a satellite is in full sunlight, penumbra, or umbra.
//!
//! Two formulations are provided:
//!
//! - [`shadow_fraction`] and [`shadow_fraction_with_model`] evaluate Earth-only
//!   conical shadow using linear penumbra interpolation between the umbra and
//!   penumbra cone radii. They remain unchanged so existing Earth-only callers
//!   keep bit-identical outputs.
//! - [`moon_shadow_fraction`], [`shadow_fraction_with_moon`], and
//!   [`shadow_fraction_with_model_and_moon`] evaluate the apparent-disc
//!   occultation fraction of Montenbruck & Gill, *Satellite Orbits* (2000),
//!   section 3.4.2, using the IAU 2015 Resolution B3 nominal solar radius
//!   ([`IAU_2015_SOLAR_RADIUS_KM`]), mean spherical Earth radius
//!   ([`MEAN_EARTH_RADIUS_KM`]) or WGS84 oblate Earth limb tangent, and mean
//!   lunar radius ([`MOON_RADIUS_KM`]). When both Earth and the Moon occult the
//!   Sun simultaneously, the occluded fraction is the exact area of the union
//!   of the two apparent occulting discs intersected with the solar disc,
//!   computed via inclusion-exclusion and circular-arc boundary integration
//!   (Green's theorem) so overlapping discs are never double-counted and the
//!   result always remains in `[0.0, 1.0]`.
//!
//! Cross-checks against Orekit 13.1.10 (`SolarRadiationPressure` with
//! `addOccultingBody` and `OccultationEngine`) and 40-digit Montenbruck & Gill
//! reference evaluations live in `tests/eclipse_moon_orekit_oracle.rs`.

use crate::astro::constants::{
    astro::{IAU_2015_SOLAR_RADIUS_KM, MOON_RADIUS_KM, SOLAR_RADIUS_KM},
    earth::{MEAN_EARTH_RADIUS_KM, WGS84_F},
};
use crate::astro::math::vec3;

/// Public eclipse alias for WGS84 flattening `1 / 298.257223563`.
pub const WGS84_FLATTENING: f64 = WGS84_F;

/// Earth figure used for conical eclipse geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarthShadowModel {
    /// Spherical Earth with [`MEAN_EARTH_RADIUS_KM`].
    Spherical,
    /// WGS84 oblate Earth approximation by polar-axis scaling.
    ///
    /// The z components of the satellite and Sun vectors are scaled by
    /// `1 / (1 - WGS84_FLATTENING)` before evaluating the same conical shadow
    /// geometry. Equatorial geometries are therefore bit-identical to the
    /// spherical model, while polar grazing geometries see the smaller polar
    /// radius implied by the flattening.
    ///
    /// The apparent-disc functions ([`shadow_fraction_with_model_and_moon`],
    /// [`status_with_model_and_moon`]) use the exact limb tangent of the same
    /// spheroid: equatorial radius [`MEAN_EARTH_RADIUS_KM`] and polar radius
    /// `MEAN_EARTH_RADIUS_KM * (1 - WGS84_FLATTENING)`.
    Wgs84Oblate,
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
    validate_nonzero_vec3(sat_pos, "sat_pos")?;
    validate_nonzero_vec3(sun_pos, "sun_pos")?;

    let d_sun = vec3::norm3(sun_pos);
    if d_sun <= SOLAR_RADIUS_KM + MEAN_EARTH_RADIUS_KM {
        return Err(invalid_eclipse_input("sun_pos", "out of range"));
    }

    // Unit vector from Earth to Sun. The division form (not reciprocal-multiply
    // via `vec3::unit3`) is required to stay bit-exact with the reference.
    let sun_u = [sun_pos[0] / d_sun, sun_pos[1] / d_sun, sun_pos[2] / d_sun];

    // Project the satellite onto the Earth-Sun line. Positive points toward the
    // Sun; a satellite on the sunlit side cannot be in shadow.
    let proj = vec3::dot3(sat_pos, sun_u);
    if proj >= 0.0 {
        return Ok(0.0);
    }

    // Perpendicular distance from the satellite to the shadow axis.
    let perp = vec3::sub3(sat_pos, vec3::scale3(sun_u, proj));
    let rho = vec3::norm3(perp);

    // Distance behind Earth along the shadow axis (positive).
    let dist_behind = -proj;

    // Umbra cone (converging): sin(alpha) = (R_sun - R_earth) / d_sun.
    let alpha_umbra = libm::asin((SOLAR_RADIUS_KM - MEAN_EARTH_RADIUS_KM) / d_sun);
    // Penumbra cone (diverging): sin(alpha) = (R_sun + R_earth) / d_sun.
    let alpha_penumbra = libm::asin((SOLAR_RADIUS_KM + MEAN_EARTH_RADIUS_KM) / d_sun);

    // Cone radii at the satellite's distance behind Earth.
    let r_umbra = MEAN_EARTH_RADIUS_KM - dist_behind * libm::tan(alpha_umbra);
    let r_penumbra = MEAN_EARTH_RADIUS_KM + dist_behind * libm::tan(alpha_penumbra);

    if !rho.is_finite() || !r_umbra.is_finite() || !r_penumbra.is_finite() {
        return Err(invalid_eclipse_input("sat_pos", "out of range"));
    }

    if rho >= r_penumbra {
        // Beyond the penumbra cone: full sunlight.
        Ok(0.0)
    } else if r_umbra > 0.0 && rho <= r_umbra {
        // Inside the umbra cone: full shadow.
        Ok(1.0)
    } else if r_umbra > 0.0 {
        // Penumbra region: linear interpolation between the cone radii.
        Ok((r_penumbra - rho) / (r_penumbra - r_umbra))
    } else {
        // Past the umbra tip (antumbra): penumbra still applies, fraction
        // decreasing from the axis.
        Ok(0.0_f64.max((r_penumbra - rho) / r_penumbra))
    }
}

/// Shadow fraction with an explicit Earth figure model.
///
/// [`EarthShadowModel::Spherical`] delegates to [`shadow_fraction`] and is
/// bit-identical to the default API. [`EarthShadowModel::Wgs84Oblate`] applies
/// the WGS84 polar-axis scaling before evaluating the same conical model.
pub fn shadow_fraction_with_model(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    model: EarthShadowModel,
) -> Result<f64, EclipseError> {
    match model {
        EarthShadowModel::Spherical => shadow_fraction(sat_pos, sun_pos),
        EarthShadowModel::Wgs84Oblate => {
            validate_nonzero_vec3(sat_pos, "sat_pos")?;
            validate_nonzero_vec3(sun_pos, "sun_pos")?;
            let sat_pos = scale_polar_axis(sat_pos);
            let sun_pos = scale_polar_axis(sun_pos);
            if !sat_pos[2].is_finite() {
                return Err(invalid_eclipse_input("sat_pos", "out of range"));
            }
            if !sun_pos[2].is_finite() {
                return Err(invalid_eclipse_input("sun_pos", "out of range"));
            }
            shadow_fraction(sat_pos, sun_pos)
        }
    }
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

/// Shadow fraction in `[0.0, 1.0]` cast by the Moon alone using the
/// apparent-disc occultation model of Montenbruck & Gill (2000), section 3.4.2.
///
/// `sat_pos`, `sun_pos`, and `moon_pos` are inertial positions in the same
/// frame (km); `sat_pos` may be any point outside the Moon and the Sun
/// (including an Earth surface point for albedo shadowing). Uses
/// [`IAU_2015_SOLAR_RADIUS_KM`] and [`MOON_RADIUS_KM`]. Returns `0.0` in full
/// sunlight or when the Moon is not closer to the observer than the Sun,
/// `(b / a)^2` in annular eclipse, and `1.0` in total lunar umbra.
pub fn moon_shadow_fraction(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    moon_pos: [f64; 3],
) -> Result<f64, EclipseError> {
    validate_finite_norm_vec3(sat_pos, "sat_pos")?;
    validate_finite_norm_vec3(sun_pos, "sun_pos")?;
    validate_finite_norm_vec3(moon_pos, "moon_pos")?;

    let (u_sun, d_sun) =
        relative_unit_and_norm(sun_pos, sat_pos, IAU_2015_SOLAR_RADIUS_KM, "sun_pos")?;
    let (u_moon, d_moon) = relative_unit_and_norm(moon_pos, sat_pos, MOON_RADIUS_KM, "moon_pos")?;

    if d_moon >= d_sun {
        return Ok(0.0);
    }

    let a = libm::asin(IAU_2015_SOLAR_RADIUS_KM / d_sun);
    let b_moon = libm::asin(MOON_RADIUS_KM / d_moon);
    let c_moon = angle_between_units(u_sun, u_moon);

    Ok(single_body_disc_shadow_fraction(a, b_moon, c_moon))
}

/// Combined spherical-Earth and Moon shadow fraction in `[0.0, 1.0]`.
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

/// Combined Earth and Moon shadow fraction in `[0.0, 1.0]` with an explicit
/// Earth figure model.
///
/// Uses the apparent-disc occultation geometry of Montenbruck & Gill (2000),
/// section 3.4.2, with [`IAU_2015_SOLAR_RADIUS_KM`], [`MEAN_EARTH_RADIUS_KM`],
/// [`WGS84_FLATTENING`], and [`MOON_RADIUS_KM`]. Under
/// [`EarthShadowModel::Wgs84Oblate`], Earth's apparent limb radius toward the
/// Sun is the exact tangent angle to the oblate spheroid with equatorial radius
/// [`MEAN_EARTH_RADIUS_KM`], polar radius
/// `MEAN_EARTH_RADIUS_KM * (1 - WGS84_FLATTENING)`, and polar axis along the
/// frame `z` axis (the same figure as the polar-axis scaling of
/// [`shadow_fraction_with_model`]).
///
/// When both Earth and the Moon occult the Sun simultaneously, the combined
/// shadow fraction is the exact area of `D_S ∩ (D_E ∪ D_M)` divided by the
/// apparent solar disc area `pi * a^2`, so overlapping occulting discs are not
/// double-counted and the result always lies in `[0.0, 1.0]`.
///
/// `sat_pos`, `sun_pos`, and `moon_pos` are Earth-centered inertial positions
/// (km). Returns [`EclipseError::InvalidInput`] if any vector is non-finite,
/// zero, overflows its Euclidean norm, or places the satellite on or inside
/// Earth, the Moon, or the Sun.
pub fn shadow_fraction_with_model_and_moon(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    moon_pos: [f64; 3],
    model: EarthShadowModel,
) -> Result<f64, EclipseError> {
    let figure = match model {
        EarthShadowModel::Spherical => EarthFigure::Sphere {
            radius_km: MEAN_EARTH_RADIUS_KM,
        },
        EarthShadowModel::Wgs84Oblate => EarthFigure::Spheroid {
            equatorial_radius_km: MEAN_EARTH_RADIUS_KM,
            flattening: WGS84_FLATTENING,
        },
    };
    shadow_fraction_with_figure_and_moon(sat_pos, sun_pos, moon_pos, figure)
}

/// Earth figure whose apparent limb bounds the Earth disc in the
/// apparent-disc model.
#[derive(Debug, Clone, Copy)]
enum EarthFigure {
    /// Sphere of the given radius (km).
    Sphere { radius_km: f64 },
    /// Oblate spheroid with polar axis along the frame `z` axis.
    Spheroid {
        equatorial_radius_km: f64,
        flattening: f64,
    },
}

fn shadow_fraction_with_figure_and_moon(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    moon_pos: [f64; 3],
    figure: EarthFigure,
) -> Result<f64, EclipseError> {
    validate_nonzero_vec3(sat_pos, "sat_pos")?;
    validate_nonzero_vec3(sun_pos, "sun_pos")?;
    validate_nonzero_vec3(moon_pos, "moon_pos")?;

    let to_earth = vec3::neg3(sat_pos);
    let (u_earth, d_earth) = unit_and_norm(to_earth, "sat_pos")?;

    let (u_sun, d_sun) =
        relative_unit_and_norm(sun_pos, sat_pos, IAU_2015_SOLAR_RADIUS_KM, "sun_pos")?;
    let (u_moon, d_moon) = relative_unit_and_norm(moon_pos, sat_pos, MOON_RADIUS_KM, "moon_pos")?;

    let b_earth = match figure {
        EarthFigure::Sphere { radius_km } => {
            if d_earth <= radius_km {
                return Err(invalid_eclipse_input("sat_pos", "out of range"));
            }
            libm::asin(radius_km / d_earth)
        }
        EarthFigure::Spheroid {
            equatorial_radius_km,
            flattening,
        } => ellipsoid_limb_angle(
            sat_pos,
            u_earth,
            d_earth,
            u_sun,
            equatorial_radius_km,
            flattening,
        )?,
    };

    let a = libm::asin(IAU_2015_SOLAR_RADIUS_KM / d_sun);
    let earth_in_front = d_earth < d_sun;
    let moon_in_front = d_moon < d_sun;

    let c_earth = angle_between_units(u_sun, u_earth);
    let b_moon = libm::asin(MOON_RADIUS_KM / d_moon);
    let c_moon = angle_between_units(u_sun, u_moon);

    let earth_occults = earth_in_front && !discs_disjoint(a, b_earth, c_earth);
    let moon_occults = moon_in_front && !discs_disjoint(a, b_moon, c_moon);

    if !earth_occults && !moon_occults {
        return Ok(0.0);
    }
    if !moon_occults {
        return Ok(single_body_disc_shadow_fraction(a, b_earth, c_earth));
    }
    if !earth_occults {
        return Ok(single_body_disc_shadow_fraction(a, b_moon, c_moon));
    }

    // Either body alone completely covers the Sun.
    if disc_contains(b_earth, a, c_earth) || disc_contains(b_moon, a, c_moon) {
        return Ok(1.0);
    }

    // Position angle psi between the Earth and Moon offsets around the Sun.
    // `u_sun × u_body` is perpendicular to `u_sun` by construction, so its
    // direction is the body's offset direction rotated a quarter turn about
    // `u_sun`, and its length is the same `sin(c_body)` that gives `c_body`.
    // Projecting `u_body - (u_body · u_sun) u_sun` instead leaves a residual
    // along `u_sun` from the unit vectors' rounding, which turns psi when
    // `c_body` is only a few ULP.
    let (cos_psi, sin_psi) = match (
        unit_direction(vec3::cross3(u_sun, u_earth)),
        unit_direction(vec3::cross3(u_sun, u_moon)),
    ) {
        (Some(w_earth), Some(w_moon)) => {
            let cos_p = vec3::dot3(w_earth, w_moon);
            let sin_p = vec3::norm3(vec3::cross3(w_earth, w_moon));
            let n = libm::hypot(cos_p, sin_p);
            (cos_p / n, sin_p / n)
        }
        _ => (1.0, 0.0),
    };

    Ok(combined_two_body_disc_shadow_fraction(
        a, b_earth, c_earth, b_moon, c_moon, cos_psi, sin_psi,
    ))
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

/// Exact apparent limb angular radius of an oblate ellipsoid (equatorial radius
/// `re`, flattening `f`, polar axis along `z`) toward `u_sun` from `sat_pos`.
///
/// Rescaling the `z` axis by `1 / (1 - f)` maps the ellipsoid to a unit sphere
/// while preserving coplanarity and tangency. The tangent point on the unit
/// sphere in the plane of the origin, scaled satellite, and scaled Sun direction
/// maps back to the exact limb point on the oblate ellipsoid.
///
/// When the Sun lies on the satellite's anti-Earth axis to within rounding, that
/// plane is undefined and any plane through the axis gives a valid limb
/// tangent; the result then lies between the polar and equatorial limb angles.
fn ellipsoid_limb_angle(
    sat_pos: [f64; 3],
    u_earth: [f64; 3],
    d_earth: f64,
    u_sun: [f64; 3],
    re: f64,
    flattening: f64,
) -> Result<f64, EclipseError> {
    let rc = re * (1.0 - flattening);
    let sat_s = [sat_pos[0] / re, sat_pos[1] / re, sat_pos[2] / rc];
    let r2 = vec3::dot3(sat_s, sat_s);
    if !r2.is_finite() || r2 <= 1.0 {
        return Err(invalid_eclipse_input("sat_pos", "out of range"));
    }
    let r = r2.sqrt();
    let u = [sat_s[0] / r, sat_s[1] / r, sat_s[2] / r];

    let to_sun_s = [u_sun[0], u_sun[1], u_sun[2] / (1.0 - flattening)];
    let (to_sun_s_u, _) = unit_and_norm(to_sun_s, "sun_pos")?;

    // Unit vector `v` in the tangent plane, perpendicular to `u`, toward the
    // Sun. With `t = to_sun_s_u`, `(u × t) × u` is perpendicular to `u` to
    // working precision for every `t`. The projection `t - (t · u) u` is not: when `t` is anti-parallel to `u`
    // its rounding residual lies along `u`, which would tilt `v` onto `±u` and
    // move the tangent point off the limb.
    let v = unit_direction(vec3::cross3(u, to_sun_s_u))
        .and_then(|axis| unit_direction(vec3::cross3(axis, u)))
        .or_else(|| {
            let fallback = if u[0].abs() < 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            unit_direction(vec3::cross3(u, fallback))
        })
        .ok_or_else(|| invalid_eclipse_input("sat_pos", "out of range"))?;

    let c = 1.0 / r;
    let s = (r2 - 1.0).sqrt() / r;
    let p_s = [
        c * u[0] + s * v[0],
        c * u[1] + s * v[1],
        c * u[2] + s * v[2],
    ];
    let p = [p_s[0] * re, p_s[1] * re, p_s[2] * rc];

    // `(p - sat_pos) × u_earth = p × u_earth` and
    // `(p - sat_pos) · u_earth = d_earth + p · u_earth`, avoiding direct
    // subtraction of `sat_pos` from `p` at large satellite distances.
    let sin_term = vec3::norm3(vec3::cross3(p, u_earth));
    let cos_term = d_earth + vec3::dot3(p, u_earth);
    Ok(libm::atan2(sin_term, cos_term))
}

/// Exact IEEE-754 predicate for `d >= r1 + r2` (discs have empty interior
/// intersection) without rounding-induced false disjointness when `r1 >> r2`.
fn discs_disjoint(r1: f64, r2: f64, d: f64) -> bool {
    let (r_max, r_min) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };
    if d < r_max {
        false
    } else if d > 2.0 * r_max {
        true
    } else {
        // Sterbenz: `d - r_max` is exact for `r_max <= d <= 2 * r_max`.
        d - r_max >= r_min
    }
}

/// Exact IEEE-754 predicate for `d + r_inner <= r_outer` (disc of radius
/// `r_inner` at center distance `d` lies entirely inside disc of radius
/// `r_outer`).
fn disc_contains(r_outer: f64, r_inner: f64, d: f64) -> bool {
    if r_outer < r_inner || d > r_outer {
        return false;
    }
    if r_inner >= 0.5 * r_outer {
        // Sterbenz: `r_outer - r_inner` is exact.
        d <= r_outer - r_inner
    } else if d >= 0.5 * r_outer {
        // Sterbenz: `r_outer - d` is exact.
        r_inner <= r_outer - d
    } else {
        true
    }
}

/// Pairwise center distance `d` and cancellation-free `d - max(r1, r2)` between
/// two discs `(x1, y1, r1)` and `(x2, y2, r2)`.
///
/// When one disc has a large radius `r_b` with center distance `c_b ≈ r_b` from
/// the origin while the other disc lies close to the origin (`c_s << r_b`),
/// directly forming `x_s - x_b` quantizes `x_s` to `ULP(r_b)`. Projecting
/// `(x_s, y_s)` onto the large disc's radial unit vector and evaluating
/// `d - r_b = (2 r_b q + q^2 + s_perp^2) / (d + r_b)` with exact Sterbenz
/// `delta_b = c_b - r_b` and `q = delta_b - s_par` preserves full precision.
fn disc_pair_dist_and_delta(d1: (f64, f64, f64), d2: (f64, f64, f64)) -> (f64, f64) {
    let d = libm::hypot(d2.0 - d1.0, d2.1 - d1.1);
    let ((xs, ys, _rs), (xb, yb, rb)) = if d2.2 >= d1.2 { (d1, d2) } else { (d2, d1) };
    let cs = libm::hypot(xs, ys);
    if cs > 0.0 && rb > 0.0 && cs < 0.25 * rb {
        let cb = libm::hypot(xb, yb);
        if cb >= 0.5 * rb && cb <= 2.0 * rb {
            let delta_b = cb - rb;
            let ux = xb / cb;
            let uy = yb / cb;
            let s_par = xs * ux + ys * uy;
            let s_perp = -xs * uy + ys * ux;
            let q = delta_b - s_par;
            let d_minus_rb = (2.0 * rb * q + (q * q + s_perp * s_perp)) / (d + rb);
            return (d, d_minus_rb);
        }
    }
    (d, d - rb)
}

/// Cancellation-free Heron half-angle factors `(p1, p2, p3, p4)` using a
/// precomputed `d_minus_rmax = d - max(r1, r2)`.
fn heron_factors_with_delta(r1: f64, r2: f64, d: f64, d_minus_rmax: f64) -> (f64, f64, f64, f64) {
    if r2 >= r1 {
        let p1 = r1 - d_minus_rmax;
        let p2 = if r1 >= 0.5 * r2 {
            d - (r2 - r1)
        } else {
            r1 + d_minus_rmax
        };
        let p3 = d + (r2 - r1);
        let p4 = (d + r2) + r1;
        (p1, p2, p3, p4)
    } else {
        let p1 = r2 - d_minus_rmax;
        let p2 = d + (r1 - r2);
        let p3 = if r2 >= 0.5 * r1 {
            d - (r1 - r2)
        } else {
            r2 + d_minus_rmax
        };
        let p4 = (d + r1) + r2;
        (p1, p2, p3, p4)
    }
}

/// Cancellation-free evaluation of `theta - sin(theta)`.
fn theta_minus_sin(theta: f64) -> f64 {
    if theta.abs() < 0.1 {
        let t2 = theta * theta;
        theta
            * t2
            * (1.0 / 6.0
                + t2 * (-1.0 / 120.0
                    + t2 * (1.0 / 5_040.0 + t2 * (-1.0 / 362_880.0 + t2 / 39_916_800.0))))
    } else {
        theta - libm::sin(theta)
    }
}

/// Exact area of the intersection of two circular discs of radii `r1` and `r2`
/// separated by center distance `d`.
///
/// Uses the half-angle circular-segment decomposition
/// `0.5 * r1^2 * (2*alpha - sin(2*alpha)) + 0.5 * r2^2 * (2*beta - sin(2*beta))`
/// with Heron's half-angle formulas grouped to avoid cancellation when one disc
/// is much larger than the other (such as Earth seen from LEO against the Sun).
fn two_disc_intersection_area(r1: f64, r2: f64, d: f64) -> f64 {
    two_disc_intersection_area_with_delta(r1, r2, d, d - r1.max(r2))
}

fn two_disc_intersection_area_with_delta(r1: f64, r2: f64, d: f64, d_minus_rmax: f64) -> f64 {
    let r_min = r1.min(r2);
    if d_minus_rmax >= r_min {
        return 0.0;
    }
    if r2 >= r1 && (r1 >= 0.5 * r2 && d <= r2 - r1 || r1 < 0.5 * r2 && d_minus_rmax <= -r1) {
        return std::f64::consts::PI * r1 * r1;
    }
    if r1 > r2 && (r2 >= 0.5 * r1 && d <= r1 - r2 || r2 < 0.5 * r1 && d_minus_rmax <= -r2) {
        return std::f64::consts::PI * r2 * r2;
    }
    let (p1, p2, p3, p4) = heron_factors_with_delta(r1, r2, d, d_minus_rmax);
    if p1 <= 0.0 {
        return 0.0;
    }
    if p2 <= 0.0 {
        return std::f64::consts::PI * r1 * r1;
    }
    if p3 <= 0.0 {
        return std::f64::consts::PI * r2 * r2;
    }
    let s1 = p1.sqrt();
    let s2 = p2.sqrt();
    let s3 = p3.sqrt();
    let s4 = p4.sqrt();
    let alpha = 2.0 * libm::atan2(s1 * s3, s2 * s4);
    let beta = 2.0 * libm::atan2(s1 * s2, s3 * s4);
    0.5 * r1 * r1 * theta_minus_sin(2.0 * alpha) + 0.5 * r2 * r2 * theta_minus_sin(2.0 * beta)
}

fn single_body_disc_shadow_fraction(a: f64, b: f64, c: f64) -> f64 {
    if discs_disjoint(a, b, c) {
        return 0.0;
    }
    if disc_contains(b, a, c) {
        return 1.0;
    }
    if disc_contains(a, b, c) {
        let ratio = b / a;
        return (ratio * ratio).clamp(0.0, 1.0);
    }
    let sun_area = std::f64::consts::PI * a * a;
    if sun_area <= 0.0 {
        return 0.0;
    }
    (two_disc_intersection_area(a, b, c) / sun_area).clamp(0.0, 1.0)
}

/// Combined shadow fraction `Area(D_S ∩ (D_E ∪ D_M)) / (pi * a^2)` when both
/// Earth and Moon partially or annularly occult the Sun.
fn combined_two_body_disc_shadow_fraction(
    a: f64,
    b_earth: f64,
    c_earth: f64,
    b_moon: f64,
    c_moon: f64,
    cos_psi: f64,
    sin_psi: f64,
) -> f64 {
    let f_earth = single_body_disc_shadow_fraction(a, b_earth, c_earth);
    let f_moon = single_body_disc_shadow_fraction(a, b_moon, c_moon);
    let sun_area = std::f64::consts::PI * a * a;
    if sun_area <= 0.0 {
        return f_earth.max(f_moon);
    }
    let area_earth = two_disc_intersection_area(a, b_earth, c_earth);
    let area_moon = two_disc_intersection_area(a, b_moon, c_moon);
    let discs = [
        (0.0, 0.0, a),
        (c_earth, 0.0, b_earth),
        (c_moon * cos_psi, c_moon * sin_psi, b_moon),
    ];
    let (area_hi, area_lo) = if area_earth >= area_moon {
        (area_earth, area_moon)
    } else {
        (area_moon, area_earth)
    };
    let area_triple = triple_disc_intersection_area(discs).clamp(0.0, area_lo);
    let union_area = area_hi + (area_lo - area_triple);
    (union_area / sun_area)
        .clamp(0.0, 1.0)
        .max(f_earth)
        .max(f_moon)
}

/// Represent a cyclic arc `[center - half_width, center + half_width]` with
/// `center ∈ [-pi, pi]` as 0, 1, or 2 linear intervals inside `[-pi, pi]`.
///
/// Using `[-pi, pi]` places the wrap seam opposite local angle `0`, so tiny
/// grazing arcs near `center ≈ 0` never wrap or lose precision against `2*pi`.
fn split_signed_cyclic_arc(center: f64, half_width: f64) -> ([(f64, f64); 2], usize) {
    let pi = std::f64::consts::PI;
    let tau = std::f64::consts::TAU;
    let mut out = [(0.0, 0.0); 2];
    if half_width <= 0.0 {
        return (out, 0);
    }
    if half_width >= pi {
        out[0] = (-pi, pi);
        return (out, 1);
    }
    let lo = center - half_width;
    let hi = center + half_width;
    if lo >= hi {
        return (out, 0);
    }
    if lo >= -pi && hi <= pi {
        out[0] = (lo, hi);
        return (out, 1);
    }
    if lo < -pi {
        let lo_wrap = lo + tau;
        if lo_wrap <= hi {
            out[0] = (-pi, pi);
            return (out, 1);
        }
        let mut count = 0;
        if hi > -pi {
            out[count] = (-pi, hi);
            count += 1;
        }
        if lo_wrap < pi {
            out[count] = (lo_wrap, pi);
            count += 1;
        }
        (out, count)
    } else {
        let hi_wrap = hi - tau;
        if hi_wrap >= lo {
            out[0] = (-pi, pi);
            return (out, 1);
        }
        let mut count = 0;
        if hi_wrap > -pi {
            out[count] = (-pi, hi_wrap);
            count += 1;
        }
        if lo < pi {
            out[count] = (lo, pi);
            count += 1;
        }
        (out, count)
    }
}

fn intersect_arc_sets(a: &[(f64, f64)], b: &[(f64, f64)], out: &mut [(f64, f64); 4]) -> usize {
    let mut count = 0;
    for &(a0, a1) in a {
        for &(b0, b1) in b {
            let lo = a0.max(b0);
            let hi = a1.min(b1);
            if hi > lo && count < out.len() {
                out[count] = (lo, hi);
                count += 1;
            }
        }
    }
    count
}

/// Exact area of the common intersection `D_0 ∩ D_1 ∩ D_2` of three circular
/// discs `(cx, cy, r)` in the plane.
///
/// Reduces redundant containing discs using a strict partial order on
/// `(radius, index)` so equal or near-coincident discs can never eliminate each
/// other, then integrates the positively oriented boundary arcs of
/// `∂(D_0 ∩ D_1 ∩ D_2)` via Green's theorem in the segment-plus-chord-triangle
/// form `0.5 * r^2 * (dt - sin(dt)) + 0.5 * (p1 × p2)`.
///
/// Endpoint coordinates and local angles on each circle are measured relative
/// to the center of the smallest kept disc so every boundary point lies within
/// `r_min` of the reference origin and tiny grazing arcs on much larger discs
/// stay centered near angle `0` without precision loss. Centers near that
/// origin are placed on a large circle's frame through their offset from the
/// origin, so offsets below an ULP of the large circle's center coordinates
/// still move the arc endpoints.
fn triple_disc_intersection_area(discs: [(f64, f64, f64); 3]) -> f64 {
    for i in 0..3 {
        for j in (i + 1)..3 {
            let (_d, d_minus_rmax) = disc_pair_dist_and_delta(discs[i], discs[j]);
            if d_minus_rmax >= discs[i].2.min(discs[j].2) {
                return 0.0;
            }
        }
    }

    let mut kept = [(0.0, 0.0, 0.0); 3];
    let mut kept_len = 0;
    for i in 0..3 {
        let mut redundant = false;
        for j in 0..3 {
            if i == j {
                continue;
            }
            let (d, d_minus_rmax) = disc_pair_dist_and_delta(discs[i], discs[j]);
            let ri = discs[i].2;
            let rj = discs[j].2;
            let contains = if ri < rj {
                false
            } else if rj >= 0.5 * ri {
                d <= ri - rj
            } else {
                d_minus_rmax <= -rj
            };
            if contains && (ri > rj || i > j) {
                redundant = true;
                break;
            }
        }
        if !redundant {
            kept[kept_len] = discs[i];
            kept_len += 1;
        }
    }

    if kept_len == 1 {
        return std::f64::consts::PI * kept[0].2 * kept[0].2;
    }
    if kept_len == 2 {
        let (d, d_minus_rmax) = disc_pair_dist_and_delta(kept[0], kept[1]);
        return two_disc_intersection_area_with_delta(kept[0].2, kept[1].2, d, d_minus_rmax);
    }

    let mut min_idx = 0;
    for k in 1..3 {
        if kept[k].2 < kept[min_idx].2 {
            min_idx = k;
        }
    }
    let origin_disc = kept[min_idx];
    let (ox, oy, _) = origin_disc;

    let pi = std::f64::consts::PI;
    let mut total = 0.0;
    for i in 0..3 {
        let (xi, yi, ri) = kept[i];
        let vx = ox - xi;
        let vy = oy - yi;
        let (ci, ci_minus_ri) = if i == min_idx {
            (0.0, -ri)
        } else {
            // `ri >= kept[min_idx].2`, so `disc_pair_dist_and_delta` returns `ci - ri`.
            disc_pair_dist_and_delta(kept[i], origin_disc)
        };
        let ri_minus_ci = -ci_minus_ri;
        let (e_par, e_perp) = if ci > 0.0 {
            let px = vx / ci;
            let py = vy / ci;
            ((px, py), (-py, px))
        } else {
            ((1.0, 0.0), (0.0, 1.0))
        };

        let mut arcs = [(-pi, pi), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        let mut arcs_len = 1;

        for (j, &(xj, yj, rj)) in kept.iter().enumerate() {
            if i == j {
                continue;
            }
            let (d, d_minus_rmax) = disc_pair_dist_and_delta(kept[i], kept[j]);
            let (p1, p2, p3, p4) = heron_factors_with_delta(ri, rj, d, d_minus_rmax);
            if p1 <= 0.0 || p3 <= 0.0 {
                arcs_len = 0;
                break;
            }
            if p2 <= 0.0 {
                // Full circle `i` lies inside disc `j`.
                continue;
            }
            let alpha = 2.0 * libm::atan2(p1.sqrt() * p3.sqrt(), p2.sqrt() * p4.sqrt());
            let s_phi = if j == min_idx && ci > 0.0 {
                0.0
            } else {
                // Offset `C_j - C_i` in circle `i`'s frame, in which
                // `C_i = O - c_i e_par`. When `C_j` is no farther from `O` than
                // from `C_i`, `(C_j - O) + c_i e_par` keeps every bit of the
                // short vector `C_j - O`, which differencing coordinates
                // directly would round to an ULP of a large circle's center.
                // For near-coincident large circles (`C_j` far from `O` but
                // close to `C_i`), the direct difference is the accurate one.
                let wx = xj - ox;
                let wy = yj - oy;
                let (u_par, u_perp) = if libm::hypot(wx, wy) <= d {
                    (
                        ci + (wx * e_par.0 + wy * e_par.1),
                        wx * e_perp.0 + wy * e_perp.1,
                    )
                } else {
                    let dx = xj - xi;
                    let dy = yj - yi;
                    (dx * e_par.0 + dy * e_par.1, dx * e_perp.0 + dy * e_perp.1)
                };
                libm::atan2(u_perp, u_par)
            };
            let (other, other_len) = split_signed_cyclic_arc(s_phi, alpha);
            let mut next = [(0.0, 0.0); 4];
            let next_len = intersect_arc_sets(&arcs[..arcs_len], &other[..other_len], &mut next);
            arcs = next;
            arcs_len = next_len;
            if arcs_len == 0 {
                break;
            }
        }

        for &(s1, s2) in &arcs[..arcs_len] {
            let dt = s2 - s1;
            let (a1, b1) = circle_point_local(ri, ci, ri_minus_ci, s1);
            let (a2, b2) = circle_point_local(ri, ci, ri_minus_ci, s2);
            total += 0.5 * ri * ri * theta_minus_sin(dt) + 0.5 * (a1 * b2 - b1 * a2);
        }
    }

    total.max(0.0)
}

/// Evaluate the point on circle `(C_i, r_i)` at local angle `s` measured CCW
/// from `e_par = (O - C_i) / |O - C_i|`, expressed in the circle's right-handed
/// local frame `(e_par, e_perp)` relative to reference origin `O`.
///
/// Using `cos(s) - 1 = -2 sin^2(s / 2)` and the precomputed cancellation-free
/// `r_minus_c = r - c` avoids catastrophic cancellation when `c ≈ r` and
/// `|s| << 1`.
fn circle_point_local(r: f64, c: f64, r_minus_c: f64, s: f64) -> (f64, f64) {
    if c == 0.0 {
        let (sin_s, cos_s) = libm::sincos(s);
        return (r * cos_s, r * sin_s);
    }
    let s_half = libm::sin(0.5 * s);
    let p_par = r_minus_c - 2.0 * r * s_half * s_half;
    let p_perp = r * libm::sin(s);
    (p_par, p_perp)
}

/// Unit vector along `v`, or `None` when `v` is zero or not finite.
///
/// Dividing by the largest component magnitude first keeps the result exact to
/// rounding when the components of `v` are small enough for their squares to
/// underflow.
fn unit_direction(v: [f64; 3]) -> Option<[f64; 3]> {
    let scale = v[0].abs().max(v[1].abs()).max(v[2].abs());
    if !(scale > 0.0 && scale.is_finite()) {
        return None;
    }
    let w = [v[0] / scale, v[1] / scale, v[2] / scale];
    let n = vec3::norm3(w);
    Some([w[0] / n, w[1] / n, w[2] / n])
}

fn angle_between_units(u: [f64; 3], v: [f64; 3]) -> f64 {
    libm::atan2(vec3::norm3(vec3::cross3(u, v)), vec3::dot3(u, v))
}

fn validate_finite_norm_vec3(v: [f64; 3], field: &'static str) -> Result<(), EclipseError> {
    if !v.iter().all(|value| value.is_finite()) {
        return Err(invalid_eclipse_input(field, "not finite"));
    }
    if !vec3::norm3(v).is_finite() {
        return Err(invalid_eclipse_input(field, "out of range"));
    }
    Ok(())
}

fn relative_unit_and_norm(
    target: [f64; 3],
    sat: [f64; 3],
    body_radius_km: f64,
    target_field: &'static str,
) -> Result<([f64; 3], f64), EclipseError> {
    let rel = vec3::sub3(target, sat);
    if !rel.iter().all(|value| value.is_finite()) {
        return Err(invalid_eclipse_input(target_field, "out of range"));
    }
    let norm = vec3::norm3(rel);
    if !norm.is_finite() {
        return Err(invalid_eclipse_input(target_field, "out of range"));
    }
    if norm <= body_radius_km {
        return Err(invalid_eclipse_input("sat_pos", "out of range"));
    }
    Ok(([rel[0] / norm, rel[1] / norm, rel[2] / norm], norm))
}

fn unit_and_norm(v: [f64; 3], field: &'static str) -> Result<([f64; 3], f64), EclipseError> {
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
    Ok(([v[0] / norm, v[1] / norm, v[2] / norm], norm))
}

fn scale_polar_axis(v: [f64; 3]) -> [f64; 3] {
    let z_scale = 1.0 / (1.0 - WGS84_FLATTENING);
    [v[0], v[1], v[2] * z_scale]
}

fn validate_nonzero_vec3(v: [f64; 3], field: &'static str) -> Result<(), EclipseError> {
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

fn invalid_eclipse_input(field: &'static str, reason: &'static str) -> EclipseError {
    EclipseError::InvalidInput { field, reason }
}

#[cfg(test)]
mod tests {
    use crate::astro::events::{CrossingDirection, EventFinder};

    use super::*;

    // Approximate Sun-Earth distance, ~1 AU in km, matching the reference
    // oracle inputs.
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
        // Frozen bits captured from the reference (Elixir) conical-shadow
        // implementation. Cross-language 0-ULP equality.
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

    /// Oracle provenance: WGS84 flattening is the published inverse flattening
    /// `1 / 298.257223563`; the oblate expected value is the same conical
    /// formula as `shadow_fraction` after applying the documented polar-axis
    /// scaling.
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

        // Annular lunar eclipse: Moon at apogee directly between sat and Sun
        // has apparent radius smaller than the Sun's.
        let d_moon = 405_000.0;
        let moon_apogee = [sat[0] + d_moon, 0.0, 0.0];
        let annular = moon_shadow_fraction(sat, sun, moon_apogee).expect("annular moon");
        let a = libm::asin(IAU_2015_SOLAR_RADIUS_KM / (AU_KM - sat[0]));
        let b = libm::asin(MOON_RADIUS_KM / d_moon);
        let expected_annular = (b / a) * (b / a);
        assert_close(annular, expected_annular, 1.0e-15);
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

        // Validation errors: non-finite, zero, inside Earth/Moon/Sun, norm overflow.
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
        // Polar radius for MEAN_EARTH_RADIUS_KM with WGS84 flattening is ~6349.64 km:
        // 6350 km is outside the oblate pole (accepted) yet inside the 6371 km sphere
        // (rejected), while 6349 km is inside both.
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
            shadow_fraction_with_moon([AU_KM - 100_000.0, 0.0, 0.0], sun, moon_clear).unwrap_err(),
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
        let single = single_body_disc_shadow_fraction(a, b, c);
        let identical = combined_two_body_disc_shadow_fraction(a, b, c, b, c, 1.0, 0.0);
        assert_close(identical, single, 1.0e-14);

        // Two equal discs placed on opposite sides of the Sun (psi = pi) are
        // disjoint inside the Sun when c > b, so their combined shadow is
        // strictly additive (2 * single), not 1 - (1 - single)^2.
        let c_disjoint = 0.006;
        let one_side = single_body_disc_shadow_fraction(a, b, c_disjoint);
        let opposite =
            combined_two_body_disc_shadow_fraction(a, b, c_disjoint, b, c_disjoint, -1.0, 0.0);
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
            f_both >= f_earth && f_both >= f_moon,
            "combined shadow {f_both} fell below single-body shadows ({f_earth}, {f_moon})"
        );
        assert_close(f_both, f_earth, 1.0e-11);

        // Three equal-radius discs separated by sub-ULP distances must not eliminate all discs.
        let r = 0.01_f64;
        let eps = 1.0e-19_f64;
        let area_all_equal =
            triple_disc_intersection_area([(0.0, 0.0, r), (eps, 0.0, r), (0.0, eps, r)]);
        assert_close(area_all_equal, std::f64::consts::PI * r * r, 1.0e-14);

        // A small disc centered on the boundary of a much larger disc is only half inside it.
        let r_big = 1.0_f64;
        let r_small = 1.0e-17_f64;
        let area_boundary = triple_disc_intersection_area([
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
    fn tiny_grazing_moon_arc_on_earth_limb_preserves_monotonicity_and_area() {
        let sat = [6_500.0, 0.0, 0.0];
        let d_earth = 6_500.0_f64;
        let b_earth = libm::asin(MEAN_EARTH_RADIUS_KM / d_earth);

        let d_sun = 1.0e20_f64;
        let a = libm::asin(IAU_2015_SOLAR_RADIUS_KM / d_sun);
        let c_earth = b_earth - 0.5 * a;
        let u_sun = [-libm::cos(c_earth), libm::sin(c_earth), 0.0];
        let sun = [
            sat[0] + d_sun * u_sun[0],
            sat[1] + d_sun * u_sun[1],
            sat[2] + d_sun * u_sun[2],
        ];
        let (u_sun_actual, _) = unit_and_norm(vec3::sub3(sun, sat), "sun_pos").unwrap();
        let c_earth_actual = angle_between_units(u_sun_actual, [-1.0, 0.0, 0.0]);

        // Place the Moon so Earth's limb grazes it by 1 ULP(b_earth), giving a sub-ULP(pi)
        // half-angle on Earth's circle.
        let e1 = [-u_sun_actual[1], u_sun_actual[0], 0.0];
        let e2 = [0.0, 0.0, 1.0];
        let dx_m = (c_earth_actual - b_earth) - 2.220_446_049_250_313e-16;
        let dy_m = 0.8 * a;
        let c_moon = libm::hypot(dx_m, dy_m);
        let u_moon = [
            libm::cos(c_moon) * u_sun_actual[0]
                + (libm::sin(c_moon) / c_moon) * (dx_m * e1[0] + dy_m * e2[0]),
            libm::cos(c_moon) * u_sun_actual[1]
                + (libm::sin(c_moon) / c_moon) * (dx_m * e1[1] + dy_m * e2[1]),
            libm::cos(c_moon) * u_sun_actual[2]
                + (libm::sin(c_moon) / c_moon) * (dx_m * e1[2] + dy_m * e2[2]),
        ];
        let d_moon = 5.1e18_f64;
        let moon = [
            sat[0] + d_moon * u_moon[0],
            sat[1] + d_moon * u_moon[1],
            sat[2] + d_moon * u_moon[2],
        ];

        let far_moon = [sat[0] + 2.0 * d_sun, sat[1], sat[2]];
        let f_earth = shadow_fraction_with_moon(sat, sun, far_moon).expect("earth-only shadow");
        let f_moon = moon_shadow_fraction(sat, sun, moon).expect("moon-only shadow");
        let f_both = shadow_fraction_with_moon(sat, sun, moon).expect("combined shadow");

        assert!(f_earth > 0.75 && f_earth < 0.85, "f_earth={f_earth}");
        assert!(f_moon > 0.001 && f_moon < 0.004, "f_moon={f_moon}");
        assert!(
            f_both >= f_earth && f_both >= f_moon,
            "combined shadow {f_both} < max({f_earth}, {f_moon})"
        );
        assert!(
            f_both <= f_earth + f_moon + 1.0e-12,
            "combined shadow {f_both} > f_earth + f_moon ({})",
            f_earth + f_moon
        );

        // Direct planar check of the same sub-ULP grazing configuration (where
        // d_EM = b_earth + 1 ULP(b_earth) and alpha_EM < 0.5 ULP(pi)).
        let b_moon = libm::asin(MOON_RADIUS_KM / d_moon);
        let c_e = b_earth - 0.5 * a;
        let dx_p = (c_e - b_earth) - 2.220_446_049_250_313e-16;
        let dy_p = 0.8 * a;
        let c_m = libm::hypot(dx_p, dy_p);
        let f_e_direct = single_body_disc_shadow_fraction(a, b_earth, c_e);
        let f_m_direct = single_body_disc_shadow_fraction(a, b_moon, c_m);
        let f_both_direct = combined_two_body_disc_shadow_fraction(
            a,
            b_earth,
            c_e,
            b_moon,
            c_m,
            dx_p / c_m,
            dy_p / c_m,
        );
        assert!(
            f_both_direct >= f_e_direct && f_both_direct >= f_m_direct,
            "direct combined {f_both_direct} < max({f_e_direct}, {f_m_direct})"
        );
        assert!(
            f_both_direct <= f_e_direct + f_m_direct + 1.0e-12,
            "direct combined {f_both_direct} > sum {}",
            f_e_direct + f_m_direct
        );

        // Verify the unclamped triple intersection area itself is strictly
        // positive and bounded by the Moon-Earth overlap (not corrupted by an
        // unclipped full-circle arc).
        let raw_triple = triple_disc_intersection_area([
            (0.0, 0.0, a),
            (c_e, 0.0, b_earth),
            (dx_p, dy_p, b_moon),
        ]);
        let area_m_direct = two_disc_intersection_area(a, b_moon, c_m);
        let area_em = two_disc_intersection_area(b_earth, b_moon, libm::hypot(dx_p - c_e, dy_p));
        assert!(
            raw_triple > 0.0 && raw_triple <= area_m_direct && raw_triple <= area_em * (1.0 + 1.0e-12),
            "raw triple intersection {raw_triple:e} out of physical bounds (area_m={area_m_direct:e}, area_em={area_em:e})"
        );
    }

    #[test]
    fn combined_eclipse_fraction_property_monotonic_and_bounded() {
        let mut rng = SplitMix64(0x5eed_cafe_1234_5678);

        // 1. Planar apparent-disc sweep: random scales, equal apparent radii
        //    (down to sub-ULP offsets), external/internal tangency, and tiny
        //    grazing discs on large limbs.
        for case in 0..6_000 {
            let (a, b_e, c_e, b_m, c_m, psi) = match case % 6 {
                0 => {
                    // General random overlapping radii across many decades.
                    let a = rng.log_uniform(-15.0, -0.5);
                    let b_e = rng.log_uniform(-15.0, 0.1);
                    let b_m = rng.log_uniform(-15.0, 0.1);
                    let c_e = rng.unit() * (a + b_e) * 1.1;
                    let c_m = rng.unit() * (a + b_m) * 1.1;
                    let psi = std::f64::consts::TAU * rng.unit();
                    (a, b_e, c_e, b_m, c_m, psi)
                }
                1 => {
                    // Equal apparent radii (b_e == b_m) with separations from
                    // exact coincidence and sub-ULP up to overlapping discs.
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
                    // External tangency (exact and ±1 ULP) on Earth/Moon vs Sun.
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
                    // Internal tangency (exact and ±1 ULP) and mutually tangent
                    // Earth/Moon discs inside the Sun.
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
                    // Large Earth limb (~1 rad) with tiny Sun and grazing Moon.
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
                    // Opposite-side or collinear discs (`psi = 0` or `pi`).
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
            let f_e = single_body_disc_shadow_fraction(a, b_e, c_e);
            let f_m = single_body_disc_shadow_fraction(a, b_m, c_m);
            let f_both =
                combined_two_body_disc_shadow_fraction(a, b_e, c_e, b_m, c_m, cos_psi, sin_psi);

            assert!(
                f_both.is_finite() && (0.0..=1.0).contains(&f_both),
                "case {case}: out of [0, 1]: f_both={f_both}"
            );
            assert!(
                f_both >= f_e && f_both >= f_m,
                "case {case}: monotonicity violated: f_both={f_both} < max({f_e}, {f_m})"
            );
            assert!(
                f_both <= (f_e + f_m).min(1.0) + 1.0e-12,
                "case {case}: subadditivity violated: f_both={f_both} > min(1, {f_e} + {f_m})"
            );

            let area_e = two_disc_intersection_area(a, b_e, c_e);
            let area_m = two_disc_intersection_area(a, b_m, c_m);
            let raw_triple = triple_disc_intersection_area([
                (0.0, 0.0, a),
                (c_e, 0.0, b_e),
                (c_m * cos_psi, c_m * sin_psi, b_m),
            ]);
            let sun_area = std::f64::consts::PI * a * a;
            assert!(
                raw_triple.is_finite()
                    && raw_triple >= 0.0
                    && raw_triple <= area_e.min(area_m) + 1.0e-11 * sun_area,
                "case {case}: raw triple {raw_triple:e} out of [0, min({area_e:e}, {area_m:e})]"
            );
        }

        // 2. End-to-end 3D geometries, each evaluated under both the Spherical
        //    and Wgs84Oblate Earth models, including equal apparent radii,
        //    tangent configurations, and Suns on or within 1e-9 rad of the
        //    anti-Sun axis.
        for case in 0..2_000 {
            let u_sat = rng.unit_vec3();
            let d_earth = rng.log_uniform(3.85, 6.8); // ~7,079 km to ~6.3e6 km
            let sat = vec3::scale3(u_sat, d_earth);
            let u_earth = vec3::neg3(u_sat);
            let b_earth = libm::asin(MEAN_EARTH_RADIUS_KM / d_earth);

            // Build orthonormal basis (t1, t2) perpendicular to u_earth.
            let ref_v = if u_earth[0].abs() < 0.8 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let t1_raw = vec3::cross3(u_earth, ref_v);
            let t1_n = vec3::norm3(t1_raw);
            let t1 = [t1_raw[0] / t1_n, t1_raw[1] / t1_n, t1_raw[2] / t1_n];
            let t2 = vec3::cross3(u_earth, t1);

            let d_sun = rng.log_uniform(7.8, 9.0); // ~6.3e7 km to 1e9 km
            let a = libm::asin(IAU_2015_SOLAR_RADIUS_KM / d_sun);

            let c_sun_offset = match (case / 2) % 6 {
                0 => rng.unit() * (a + b_earth) * 1.05,
                1 => a + b_earth,         // tangent Sun-Earth
                2 => (b_earth - a).abs(), // internal tangent
                3 => 0.0,                 // anti-Sun axis
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

            // Choose Moon distance: equal apparent radius to Earth in 1/3 of cases.
            let d_moon = if (case / 2) % 3 == 0 {
                d_earth * (MOON_RADIUS_KM / MEAN_EARTH_RADIUS_KM)
            } else {
                rng.log_uniform(3.5, 7.5).min(0.95 * d_sun)
            };
            let b_moon = libm::asin(MOON_RADIUS_KM / d_moon);

            let moon_offset = match (case / 2) % 5 {
                0 => 0.0,
                1 => 1.0e-18,
                2 => a + b_moon,       // tangent Moon-Sun
                3 => b_earth + b_moon, // tangent Earth-Moon
                _ => rng.unit() * (a + b_moon) * 1.1,
            };
            let az_m = std::f64::consts::TAU * rng.unit();
            let (sm_az, cm_az) = libm::sincos(az_m);
            let (s_mo, c_mo) = libm::sincos(moon_offset);
            // Offset Moon relative to either u_earth or u_sun.
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
            // Place a non-occulting Moon behind the Sun to evaluate Earth alone.
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

                assert!(
                    f_both.is_finite() && (0.0..=1.0).contains(&f_both),
                    "3D case {case} {model:?}: f_both out of [0, 1]: {f_both}"
                );
                assert!(
                    f_both >= f_earth && f_both >= f_moon,
                    "3D case {case} {model:?}: f_both={f_both} < max({f_earth}, {f_moon})"
                );
                assert!(
                    f_both <= (f_earth + f_moon).min(1.0) + 1.0e-12,
                    "3D case {case} {model:?}: f_both={f_both} > min(1, {f_earth} + {f_moon})"
                );
                if c_sun_offset <= 1.0e-9 && b_earth_polar > a + 1.0e-6 {
                    // The Sun sits on or next to the anti-Sun axis and Earth's
                    // apparent disc is larger than the Sun's under either figure.
                    assert_eq!(
                        f_earth, 1.0,
                        "3D case {case} {model:?}: Earth alone must cover the Sun on the axis"
                    );
                }
            }
        }
    }

    #[test]
    fn oblate_earth_limb_angle_stays_accurate_at_large_distance() {
        let sat = [1.0e20, 1.0e20, 1.0e20];
        let sun = [-1.0e23, -1.0e23, -1.0e23];
        let moon = [0.0, 384_400.0, 0.0];

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
        // Equatorial GEO satellite on the anti-Sun axis with `x == y`, where
        // the normalized satellite direction is not exactly unit length.
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
        // Distance bands from Earth's center (km): LEO, MEO, GEO, and the
        // range of the lunar orbit.
        let bands = [
            (6_771.0, 8_371.0),
            (20_000.0, 30_000.0),
            (42_164.0, 42_164.0),
            (356_500.0, 406_700.0),
        ];
        // Perpendicular offsets from the axis (km), all well inside the umbra
        // of both figures in every band.
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
            // Behind the Sun as seen from the satellite: never occults.
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
        let re = MEAN_EARTH_RADIUS_KM;
        let rc = re * (1.0 - WGS84_FLATTENING);
        let oblate = EarthFigure::Spheroid {
            equatorial_radius_km: re,
            flattening: WGS84_FLATTENING,
        };
        let polar = EarthFigure::Sphere { radius_km: rc };
        let equatorial = EarthFigure::Sphere { radius_km: re };
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let fixed_dirs = [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [s, s, 0.0],
            [-s, 0.0, s],
            [0.0, -s, -s],
        ];

        let mut rng = SplitMix64(0x0b1a_7e00_5eed_0002);
        for case in 0..6_000 {
            let u_sat = if case % 4 == 0 {
                fixed_dirs[(case / 4) % fixed_dirs.len()]
            } else {
                rng.unit_vec3()
            };
            // LEO just above the equatorial radius out to well beyond the Moon.
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
            // Sun offsets on the axis, next to it, and across the band where
            // the polar and equatorial spheres give different shadows.
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
                // Behind the Sun: never occults.
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

            let fraction = |figure: EarthFigure| {
                shadow_fraction_with_figure_and_moon(sat, sun, moon, figure)
                    .expect("valid bound geometry")
            };
            let f_lo = fraction(polar);
            let f_oblate = fraction(oblate);
            let f_hi = fraction(equatorial);
            assert!(
                f_lo <= f_oblate + 1.0e-12 && f_oblate <= f_hi + 1.0e-12,
                "case {case} d_earth={d_earth} c_sun={c_sun}: oblate {f_oblate} outside \
                 [{f_lo}, {f_hi}]"
            );
        }
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
        // Earth's center in the `(t, w)` frame with `w = p · (-e.1, e.0)`.
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
        // A large Moon disc whose limb passes 2^-53 from the center of a
        // 3e-16 Sun disc, with an Earth disc whose center is 2.5e-17 from the
        // Sun's: below half an ULP of the Moon center's x coordinate. Every
        // Moon center below lies exactly 0.625 from the origin, so the Moon's
        // limb near the Sun is the line `p · e = 2^-53` to within 1e-15
        // relative.
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
                let area =
                    triple_disc_intersection_area([(0.0, 0.0, a), (c_e, 0.0, b_e), (mx, my, b_m)]);
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
        // Two large discs 9e-7 apart whose limbs pass 2^-28 from the center
        // of a 1e-8 Sun disc and cross inside it. The second center comes from
        // the Pythagorean triple (2n + 1, 2n(n + 1), 2n(n + 1) + 1) with
        // n = 1_000_003 scaled by 2^-41, so both centers lie exactly `c` from
        // the origin and differ by exactly 2^-41 in x.
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

        let area = triple_disc_intersection_area([(0.0, 0.0, a), (c, 0.0, r), (mx, my, r)]);

        // Each disc's chord at abscissa `x` from its circle equation factored
        // as `(y - y_c)^2 <= (x - x_left) (x_right - x)`, with every factor
        // formed without cancellation.
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

        // The same configuration turned about the Sun's center. Rounding the
        // turned centers moves each by about 1e-16, which changes the area by
        // about 1e-7 of itself at most.
        for angle in [0.3_f64, 0.7, 2.0, -1.1] {
            let (sin_t, cos_t) = libm::sincos(angle);
            let turn = |x: f64, y: f64| (x * cos_t - y * sin_t, x * sin_t + y * cos_t);
            let (ex, ey) = turn(c, 0.0);
            let (qx, qy) = turn(mx, my);
            let turned = triple_disc_intersection_area([(0.0, 0.0, a), (ex, ey, r), (qx, qy, r)]);
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
        // their offsets from the Sun must be exactly `pi` whatever the offsets'
        // size. The offsets here are a few ULP of a unit vector.
        let sat = [-1.5e19 + 2048.0, -1.5e19 - 2048.0, 0.0];
        let sun = [1.0e21 + sat[0], 1.0e21 + sat[1], 0.0];
        let moon = [sat[0] + 1.2e19 + 2048.0, sat[1] + 1.2e19 - 2048.0, 0.0];

        let (u_earth, d_earth) = unit_and_norm(vec3::neg3(sat), "sat_pos").unwrap();
        let (u_sun, d_sun) =
            relative_unit_and_norm(sun, sat, IAU_2015_SOLAR_RADIUS_KM, "sun_pos").unwrap();
        let (u_moon, d_moon) =
            relative_unit_and_norm(moon, sat, MOON_RADIUS_KM, "moon_pos").unwrap();
        let side_earth = vec3::cross3(u_sun, u_earth)[2];
        let side_moon = vec3::cross3(u_sun, u_moon)[2];
        assert!(
            side_earth * side_moon < 0.0,
            "Earth ({side_earth:e}) and Moon ({side_moon:e}) must lie on opposite sides"
        );

        let a = libm::asin(IAU_2015_SOLAR_RADIUS_KM / d_sun);
        let b_earth = libm::asin(MEAN_EARTH_RADIUS_KM / d_earth);
        let b_moon = libm::asin(MOON_RADIUS_KM / d_moon);
        let c_earth = angle_between_units(u_sun, u_earth);
        let c_moon = angle_between_units(u_sun, u_moon);
        let expected =
            combined_two_body_disc_shadow_fraction(a, b_earth, c_earth, b_moon, c_moon, -1.0, 0.0);
        // At these angles the two occulting discs are disjoint, so the shadows add.
        let f_earth = single_body_disc_shadow_fraction(a, b_earth, c_earth);
        let f_moon = single_body_disc_shadow_fraction(a, b_moon, c_moon);
        assert!(c_earth + c_moon > b_earth + b_moon);
        assert!(f_earth > 0.0 && f_earth < 1.0 && f_moon > 0.0 && f_moon < 1.0);
        assert_close(expected, f_earth + f_moon, 1.0e-15);

        let actual = shadow_fraction_with_moon(sat, sun, moon).expect("valid coplanar geometry");
        assert_close(actual, expected, 1.0e-15);
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
