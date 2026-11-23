//! Port of Orekit 13.1.10's `Ellipsoid` (`pointOnLimb`, `getPlaneSection`),
//! `OccultationEngine` (`angles`), and `SolarRadiationPressure`
//! (`maskingRatio`, `getLightingRatio`) in both `f64` and forward-mode `Grad`,
//! with the corrected exact three-disc union for simultaneous Earth-Moon
//! occultations.

use super::ad::{FieldVec3, Grad, Vec3};
use super::fast_math;
use super::{invalid_eclipse_input, EarthShadowModel, EclipseError, WGS84_FLATTENING};
use crate::astro::constants::{
    astro::{IAU_2015_SOLAR_RADIUS_KM, MOON_RADIUS_KM},
    earth::MEAN_EARTH_RADIUS_KM,
    units::M_PER_KM,
};

/// Orekit `SolarRadiationPressure.ANGULAR_MARGIN`.
const ANGULAR_MARGIN: f64 = 1.0e-10;

/// Solar radius in metres (`Constants.SUN_RADIUS = 6.957e8`).
const SUN_RADIUS_M: f64 = IAU_2015_SOLAR_RADIUS_KM * M_PER_KM;

/// Mean spherical / equatorial Earth radius in metres (`6_371_000.0`).
const EARTH_RADIUS_M: f64 = MEAN_EARTH_RADIUS_KM * M_PER_KM;

/// Mean lunar radius in metres (`1_737_400.0`).
const MOON_RADIUS_M: f64 = MOON_RADIUS_KM * M_PER_KM;

#[derive(Debug, Clone, Copy)]
struct Ellipsoid {
    a: f64,
    b: f64,
    c: f64,
}

impl Ellipsoid {
    /// Orekit `OneAxisEllipsoid(ae, f, frame)` -> `Ellipsoid(frame, ae, ae, ae * (1.0 - f))`.
    #[inline]
    fn one_axis(ae: f64, flattening: f64) -> Self {
        Self {
            a: ae,
            b: ae,
            c: ae * (1.0 - flattening),
        }
    }

    #[inline]
    fn is_inside(self, point: Vec3) -> bool {
        let sx = point.x / self.a;
        let sy = point.y / self.b;
        let sz = point.z / self.c;
        sx * sx + sy * sy + sz * sz <= 1.0
    }

    #[inline]
    fn is_inside_field(self, point: FieldVec3) -> bool {
        let sx = point.x.div_f64(self.a);
        let sy = point.y.div_f64(self.b);
        let sz = point.z.div_f64(self.c);
        let d2 = sx.mul(sx).add(sy.mul(sy)).add(sz.mul(sz));
        d2.sub_f64(1.0).value <= 0.0
    }

    /// Orekit `Ellipsoid.getPlaneSection(planePoint, planeNormal)`.
    fn plane_section(self, plane_point: Vec3, plane_normal: Vec3) -> Option<Ellipse2D> {
        let u = plane_normal.orthogonal()?;
        let v = plane_normal.cross(u).normalize()?;
        let x_u_o_a = u.x / self.a;
        let y_u_o_b = u.y / self.b;
        let z_u_o_c = u.z / self.c;
        let x_v_o_a = v.x / self.a;
        let y_v_o_b = v.y / self.b;
        let z_v_o_c = v.z / self.c;
        let x_p_o_a = plane_point.x / self.a;
        let y_p_o_b = plane_point.y / self.b;
        let z_p_o_c = plane_point.z / self.c;

        let alpha = x_u_o_a * x_u_o_a + y_u_o_b * y_u_o_b + z_u_o_c * z_u_o_c;
        let beta = x_v_o_a * x_v_o_a + y_v_o_b * y_v_o_b + z_v_o_c * z_v_o_c;
        let gamma =
            fast_math::linear_combination_3(x_u_o_a, x_v_o_a, y_u_o_b, y_v_o_b, z_u_o_c, z_v_o_c);
        let delta =
            fast_math::linear_combination_3(x_p_o_a, x_u_o_a, y_p_o_b, y_u_o_b, z_p_o_c, z_u_o_c);
        let epsilon =
            fast_math::linear_combination_3(x_p_o_a, x_v_o_a, y_p_o_b, y_v_o_b, z_p_o_c, z_v_o_c);
        let zeta = fast_math::linear_combination_4(
            x_p_o_a, x_p_o_a, y_p_o_b, y_p_o_b, z_p_o_c, z_p_o_c, 1.0, -1.0,
        );

        let tan_theta = if fast_math::abs(gamma) < fast_math::SAFE_MIN {
            0.0
        } else {
            let b_m_a = beta - alpha;
            if b_m_a >= 0.0 {
                -2.0 * gamma / (b_m_a + fast_math::sqrt(b_m_a * b_m_a + 4.0 * gamma * gamma))
            } else {
                -2.0 * gamma / (b_m_a - fast_math::sqrt(b_m_a * b_m_a + 4.0 * gamma * gamma))
            }
        };
        let tan2 = tan_theta * tan_theta;
        let cos2 = 1.0 / (1.0 + tan2);
        let sin2 = tan2 * cos2;
        let cos_sin = tan_theta * cos2;
        let cos = fast_math::sqrt(cos2);
        let sin = tan_theta * cos;

        let denom = fast_math::linear_combination_2(gamma, gamma, -alpha, beta);
        let tau_c = fast_math::linear_combination_2(beta, delta, -gamma, epsilon) / denom;
        let nu_c = fast_math::linear_combination_2(alpha, epsilon, -gamma, delta) / denom;

        let two_gcs = 2.0 * gamma * cos_sin;
        let big_a = alpha * cos2 + beta * sin2 + two_gcs;
        let big_b = alpha * sin2 + beta * cos2 - two_gcs;
        let big_f = (alpha * tau_c + 2.0 * (gamma * nu_c + delta)) * tau_c
            + (beta * nu_c + 2.0 * epsilon) * nu_c
            + zeta;
        let l = fast_math::sqrt(-big_f / big_a);
        let m = fast_math::sqrt(-big_f / big_b);
        if (l + m).is_nan() {
            return None;
        }

        let center = Vec3::lincomb3(1.0, plane_point, tau_c, u, nu_c, v);
        if l > m {
            Some(Ellipse2D {
                center,
                u: Vec3::lincomb2(cos, u, sin, v),
                v: Vec3::lincomb2(-sin, u, cos, v),
                a: l,
                b: m,
            })
        } else {
            Some(Ellipse2D {
                center,
                u: Vec3::lincomb2(sin, u, -cos, v),
                v: Vec3::lincomb2(cos, u, sin, v),
                a: m,
                b: l,
            })
        }
    }

    /// Orekit `Ellipsoid.getPlaneSection(FieldVector3D, FieldVector3D)`.
    fn plane_section_field(
        self,
        plane_point: FieldVec3,
        plane_normal: FieldVec3,
    ) -> Option<FieldEllipse2D> {
        let u = plane_normal.orthogonal()?;
        let v = plane_normal.cross(u).normalize()?;
        let x_u_o_a = u.x.div_f64(self.a);
        let y_u_o_b = u.y.div_f64(self.b);
        let z_u_o_c = u.z.div_f64(self.c);
        let x_v_o_a = v.x.div_f64(self.a);
        let y_v_o_b = v.y.div_f64(self.b);
        let z_v_o_c = v.z.div_f64(self.c);
        let x_p_o_a = plane_point.x.div_f64(self.a);
        let y_p_o_b = plane_point.y.div_f64(self.b);
        let z_p_o_c = plane_point.z.div_f64(self.c);

        let alpha = x_u_o_a
            .mul(x_u_o_a)
            .add(y_u_o_b.mul(y_u_o_b))
            .add(z_u_o_c.mul(z_u_o_c));
        let beta = x_v_o_a
            .mul(x_v_o_a)
            .add(y_v_o_b.mul(y_v_o_b))
            .add(z_v_o_c.mul(z_v_o_c));
        let gamma =
            Grad::linear_combination_3(x_u_o_a, x_v_o_a, y_u_o_b, y_v_o_b, z_u_o_c, z_v_o_c);
        let delta =
            Grad::linear_combination_3(x_p_o_a, x_u_o_a, y_p_o_b, y_u_o_b, z_p_o_c, z_u_o_c);
        let epsilon =
            Grad::linear_combination_3(x_p_o_a, x_v_o_a, y_p_o_b, y_v_o_b, z_p_o_c, z_v_o_c);
        let zeta = Grad::linear_combination_4(
            x_p_o_a,
            x_p_o_a,
            y_p_o_b,
            y_p_o_b,
            z_p_o_c,
            z_p_o_c,
            Grad::ONE,
            Grad::ONE.negate(),
        );

        let tan_theta = if fast_math::abs(gamma.value) < fast_math::SAFE_MIN {
            Grad::ZERO
        } else {
            let b_m_a = beta.sub(alpha);
            let disc = b_m_a.mul(b_m_a).add(gamma.mul(gamma).mul_f64(4.0)).sqrt();
            if b_m_a.value >= 0.0 {
                gamma.mul_f64(-2.0).div(b_m_a.add(disc))
            } else {
                gamma.mul_f64(-2.0).div(b_m_a.sub(disc))
            }
        };
        let tan2 = tan_theta.mul(tan_theta);
        let cos2 = tan2.add_f64(1.0).reciprocal();
        let sin2 = tan2.mul(cos2);
        let cos_sin = tan_theta.mul(cos2);
        let cos = cos2.sqrt();
        let sin = tan_theta.mul(cos);

        let inv_denom = Grad::linear_combination_2(gamma, gamma, alpha.negate(), beta).reciprocal();
        let tau_c = Grad::linear_combination_2(beta, delta, gamma.negate(), epsilon).mul(inv_denom);
        let nu_c = Grad::linear_combination_2(alpha, epsilon, gamma.negate(), delta).mul(inv_denom);

        let two_gcs = gamma.mul(cos_sin).mul_f64(2.0);
        let big_a = alpha.mul(cos2).add(beta.mul(sin2)).add(two_gcs);
        let big_b = alpha.mul(sin2).add(beta.mul(cos2)).sub(two_gcs);
        let big_f_n = alpha
            .mul(tau_c)
            .add(gamma.mul(nu_c).add(delta).mul_f64(2.0))
            .mul(tau_c)
            .add(beta.mul(nu_c).add(epsilon.mul_f64(2.0)).mul(nu_c))
            .add(zeta)
            .negate();
        let l = big_f_n.div(big_a).sqrt();
        let m = big_f_n.div(big_b).sqrt();
        if l.add(m).is_nan() {
            return None;
        }

        let center = FieldVec3::lincomb3(Grad::ONE, plane_point, tau_c, u, nu_c, v);
        if l.sub(m).value > 0.0 {
            Some(FieldEllipse2D {
                center,
                u: FieldVec3::lincomb2(cos, u, sin, v),
                v: FieldVec3::lincomb2(sin.negate(), u, cos, v),
                a: l,
                b: m,
            })
        } else {
            Some(FieldEllipse2D {
                center,
                u: FieldVec3::lincomb2(sin, u, cos.negate(), v),
                v: FieldVec3::lincomb2(cos, u, sin, v),
                a: m,
                b: l,
            })
        }
    }

    /// Orekit `Ellipsoid.pointOnLimb(observer, outside)` with collinearity fallback
    /// when `cross(observer, outside)` has zero norm (exact or by underflow).
    fn point_on_limb(self, observer: Vec3, outside: Vec3) -> Result<Vec3, EclipseError> {
        if self.is_inside(observer) {
            return Err(invalid_eclipse_input("sat_pos", "out of range"));
        }
        let raw_normal = observer.cross(outside);
        let normal = if raw_normal.norm() == 0.0 {
            observer
                .orthogonal()
                .ok_or_else(|| invalid_eclipse_input("sat_pos", "out of range"))?
        } else {
            raw_normal
        };
        let section = self
            .plane_section(Vec3::ZERO, normal)
            .ok_or_else(|| invalid_eclipse_input("sat_pos", "out of range"))?;

        let (obs_x, obs_y) = section.to_plane(observer);
        let ap = section.a;
        let bp = section.b;
        let xpo = obs_x / ap;
        let ypo = obs_y / bp;
        let xpo2 = xpo * xpo;
        let ypo2 = ypo * ypo;
        let alphap = ypo2 + xpo2;
        let gammap = 1.0 - ypo2;

        let sqrt = fast_math::sqrt(alphap - 1.0);
        let sqrtp = fast_math::abs(ypo) * sqrt;
        let sqrt_signed = fast_math::copy_sign(sqrt, ypo);

        let (xpt1, xpt2, ypt1, ypt2) = if xpo > 0.0 {
            let s = xpo + sqrtp;
            (
                s / alphap,
                gammap / s,
                (ypo - xpo * sqrt_signed) / alphap,
                (xpo * ypo + sqrt_signed) / s,
            )
        } else {
            let s = xpo - sqrtp;
            (
                gammap / s,
                s / alphap,
                (xpo * ypo - sqrt_signed) / s,
                (ypo + xpo * sqrt_signed) / alphap,
            )
        };

        let tp1 = section.to_space(ap * xpt1, bp * ypt1);
        let tp2 = section.to_space(ap * xpt2, bp * ypt2);
        if tp1.distance(outside) <= tp2.distance(outside) {
            Ok(tp1)
        } else {
            Ok(tp2)
        }
    }

    /// Orekit `Ellipsoid.pointOnLimb(FieldVector3D, FieldVector3D)` with
    /// collinearity fallback when `cross(observer, outside)` has zero real norm
    /// (exact or by underflow).
    fn point_on_limb_field(
        self,
        observer: FieldVec3,
        outside: FieldVec3,
    ) -> Result<FieldVec3, EclipseError> {
        if self.is_inside_field(observer) {
            return Err(invalid_eclipse_input("sat_pos", "out of range"));
        }
        let raw_normal = observer.cross(outside);
        let normal = if raw_normal.norm().value == 0.0 {
            observer
                .orthogonal()
                .ok_or_else(|| invalid_eclipse_input("sat_pos", "out of range"))?
        } else {
            raw_normal
        };
        let section = self
            .plane_section_field(FieldVec3::ZERO, normal)
            .ok_or_else(|| invalid_eclipse_input("sat_pos", "out of range"))?;

        let (obs_x, obs_y) = section.to_plane(observer);
        let ap = section.a;
        let bp = section.b;
        let xpo = obs_x.div(ap);
        let ypo = obs_y.div(bp);
        let xpo2 = xpo.mul(xpo);
        let ypo2 = ypo.mul(ypo);
        let alphap = ypo2.add(xpo2);
        let gammap = ypo2.negate().add_f64(1.0);

        let sqrt = alphap.sub_f64(1.0).sqrt();
        let sqrtp = ypo.abs().mul(sqrt);
        let sqrt_signed = sqrt.copy_sign(ypo);

        let (xpt1, xpt2, ypt1, ypt2) = if xpo.value > 0.0 {
            let s = xpo.add(sqrtp);
            (
                s.div(alphap),
                gammap.div(s),
                ypo.sub(xpo.mul(sqrt_signed)).div(alphap),
                xpo.mul(ypo).add(sqrt_signed).div(s),
            )
        } else {
            let s = xpo.sub(sqrtp);
            (
                gammap.div(s),
                s.div(alphap),
                xpo.mul(ypo).sub(sqrt_signed).div(s),
                ypo.add(xpo.mul(sqrt_signed)).div(alphap),
            )
        };

        let tp1 = section.to_space(ap.mul(xpt1), bp.mul(ypt1));
        let tp2 = section.to_space(ap.mul(xpt2), bp.mul(ypt2));
        if tp1.distance(outside).sub(tp2.distance(outside)).value <= 0.0 {
            Ok(tp1)
        } else {
            Ok(tp2)
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Ellipse2D {
    center: Vec3,
    u: Vec3,
    v: Vec3,
    a: f64,
    b: f64,
}

impl Ellipse2D {
    #[inline]
    fn to_plane(self, p: Vec3) -> (f64, f64) {
        let delta = p.sub(self.center);
        (delta.dot(self.u), delta.dot(self.v))
    }

    #[inline]
    fn to_space(self, x: f64, y: f64) -> Vec3 {
        Vec3::lincomb3(1.0, self.center, x, self.u, y, self.v)
    }
}

#[derive(Debug, Clone, Copy)]
struct FieldEllipse2D {
    center: FieldVec3,
    u: FieldVec3,
    v: FieldVec3,
    a: Grad,
    b: Grad,
}

impl FieldEllipse2D {
    #[inline]
    fn to_plane(self, p: FieldVec3) -> (Grad, Grad) {
        let delta = p.sub(self.center);
        (delta.dot(self.u), delta.dot(self.v))
    }

    #[inline]
    fn to_space(self, x: Grad, y: Grad) -> FieldVec3 {
        FieldVec3::lincomb3(Grad::ONE, self.center, x, self.u, y, self.v)
    }
}

/// Orekit `OccultationEngine.OccultationAngles`.
#[derive(Debug, Clone, Copy)]
struct OccultationAngles {
    separation: f64,
    limb_radius: f64,
    occulted_radius: f64,
}

/// Orekit `OccultationEngine.FieldOccultationAngles<Gradient>`.
#[derive(Debug, Clone, Copy)]
struct FieldOccultationAngles {
    separation: Grad,
    limb_radius: Grad,
    occulted_radius: Grad,
}

/// Orekit `OccultationEngine.angles(SpacecraftState)` in the occulting body frame.
///
/// Returns `(angles, ps, inside_sun)` where `inside_sun` is `true` when
/// `rs.is_nan()` triggered Orekit's dummy angles `(PI, 0, 0)`.
fn occultation_angles(
    psat: Vec3,
    pted: Vec3,
    occulting: Ellipsoid,
    occulted_radius: f64,
) -> Result<(OccultationAngles, Vec3, bool), EclipseError> {
    let plimb = occulting.point_on_limb(psat, pted)?;
    let ps = psat.sub(pted);
    let pi = psat.sub(plimb);
    let angle =
        Vec3::angle(ps, psat).ok_or_else(|| invalid_eclipse_input("sat_pos", "out of range"))?;
    let rs = fast_math::asin(occulted_radius / ps.norm());
    let ro =
        Vec3::angle(pi, psat).ok_or_else(|| invalid_eclipse_input("sat_pos", "out of range"))?;
    let inside_sun = rs.is_nan();
    let angles = if inside_sun {
        OccultationAngles {
            separation: fast_math::PI,
            limb_radius: 0.0,
            occulted_radius: 0.0,
        }
    } else {
        OccultationAngles {
            separation: angle,
            limb_radius: ro,
            occulted_radius: rs,
        }
    };
    Ok((angles, ps, inside_sun))
}

/// Orekit `OccultationEngine.angles(FieldSpacecraftState<Gradient>)` in the occulting body frame.
///
/// Returns `(angles, ps, inside_sun)` where `inside_sun` is `true` when
/// `rs.is_nan()` triggered Orekit's dummy angles `(PI, 0, 0)`.
fn occultation_angles_field(
    psat: FieldVec3,
    pted: FieldVec3,
    occulting: Ellipsoid,
    occulted_radius: f64,
) -> Result<(FieldOccultationAngles, FieldVec3, bool), EclipseError> {
    let plimb = occulting.point_on_limb_field(psat, pted)?;
    let ps = psat.sub(pted);
    let pi = psat.sub(plimb);
    let angle = FieldVec3::angle(ps, psat)
        .ok_or_else(|| invalid_eclipse_input("sat_pos", "out of range"))?;
    let rs = ps.norm().reciprocal().mul_f64(occulted_radius).asin();
    let ro = FieldVec3::angle(pi, psat)
        .ok_or_else(|| invalid_eclipse_input("sat_pos", "out of range"))?;
    let inside_sun = rs.is_nan();
    let angles = if inside_sun {
        FieldOccultationAngles {
            separation: Grad::PI,
            limb_radius: Grad::ZERO,
            occulted_radius: Grad::ZERO,
        }
    } else {
        FieldOccultationAngles {
            separation: angle,
            limb_radius: ro,
            occulted_radius: rs,
        }
    };
    Ok((angles, ps, inside_sun))
}

/// Orekit `SolarRadiationPressure.maskingRatio(OccultationAngles)`.
///
/// Diverges from Orekit only at exact collinearity (`separation == 0.0`) inside
/// the annular region where Orekit divides by `0.0` and returns `NaN`; there we
/// return `1.0 - aE2 / aS2`.
fn masking_ratio(angles: OccultationAngles) -> f64 {
    let sun_sat_central_body_angle = angles.separation;
    let alpha_central = angles.limb_radius;
    let alpha_sun = angles.occulted_radius;

    if sun_sat_central_body_angle - alpha_central + alpha_sun <= ANGULAR_MARGIN {
        0.0
    } else if sun_sat_central_body_angle - alpha_central - alpha_sun < -ANGULAR_MARGIN {
        if sun_sat_central_body_angle == 0.0 {
            let a_s2 = alpha_sun * alpha_sun;
            let a_e2 = alpha_central * alpha_central;
            return 1.0 - a_e2 / a_s2;
        }
        let s_ea2 = sun_sat_central_body_angle * sun_sat_central_body_angle;
        let oo2s_ea = 1.0 / (2.0 * sun_sat_central_body_angle);
        let a_s2 = alpha_sun * alpha_sun;
        let a_e2 = alpha_central * alpha_central;
        let a_e2_m_a_s2 = a_e2 - a_s2;

        let alpha1 = (s_ea2 - a_e2_m_a_s2) * oo2s_ea;
        let alpha2 = (s_ea2 + a_e2_m_a_s2) * oo2s_ea;

        let almost0 = fast_math::SAFE_MIN;
        let almost1 = fast_math::next_down(1.0);
        let a1_o_a_s = fast_math::min(almost1, fast_math::max(-almost1, alpha1 / alpha_sun));
        let a_s2_m_a12 = fast_math::max(almost0, a_s2 - alpha1 * alpha1);
        let a2_o_a_e = fast_math::min(almost1, fast_math::max(-almost1, alpha2 / alpha_central));
        let a_e2_m_a22 = fast_math::max(almost0, a_e2 - alpha2 * alpha2);

        let p1 = a_s2 * fast_math::acos(a1_o_a_s) - alpha1 * fast_math::sqrt(a_s2_m_a12);
        let p2 = a_e2 * fast_math::acos(a2_o_a_e) - alpha2 * fast_math::sqrt(a_e2_m_a22);

        1.0 - (p1 + p2) / (fast_math::PI * a_s2)
    } else {
        1.0
    }
}

/// Orekit `SolarRadiationPressure.min(double d, T f)`.
#[inline]
fn orekit_min_const(d: f64, f: Grad) -> Grad {
    if f.value > d {
        Grad::constant(d)
    } else {
        f
    }
}

/// Orekit `SolarRadiationPressure.max(double d, T f)`.
#[inline]
fn orekit_max_const(d: f64, f: Grad) -> Grad {
    if f.value <= d {
        Grad::constant(d)
    } else {
        f
    }
}

/// Orekit `SolarRadiationPressure.maskingRatio(FieldOccultationAngles<Gradient>)`.
fn masking_ratio_field(angles: FieldOccultationAngles) -> Grad {
    let occulted_sat_occulting = angles.separation;
    let alpha_occulting = angles.limb_radius;
    let alpha_occulted = angles.occulted_radius;

    if occulted_sat_occulting.value - alpha_occulting.value + alpha_occulted.value <= ANGULAR_MARGIN
    {
        Grad::ZERO
    } else if occulted_sat_occulting.value - alpha_occulting.value - alpha_occulted.value
        < -ANGULAR_MARGIN
    {
        let s_ea2 = occulted_sat_occulting.mul(occulted_sat_occulting);
        let oo2s_ea = occulted_sat_occulting.mul_f64(2.0).reciprocal();
        let a_s2 = alpha_occulted.mul(alpha_occulted);
        let a_e2 = alpha_occulting.mul(alpha_occulting);
        let a_e2_m_a_s2 = a_e2.sub(a_s2);

        let alpha1 = s_ea2.sub(a_e2_m_a_s2).mul(oo2s_ea);
        let alpha2 = s_ea2.add(a_e2_m_a_s2).mul(oo2s_ea);

        let almost0 = fast_math::SAFE_MIN;
        let almost1 = fast_math::next_down(1.0);
        let a1_o_a_s = orekit_min_const(
            almost1,
            orekit_max_const(-almost1, alpha1.div(alpha_occulted)),
        );
        let a_s2_m_a12 = orekit_max_const(almost0, a_s2.sub(alpha1.mul(alpha1)));
        let a2_o_a_e = orekit_min_const(
            almost1,
            orekit_max_const(-almost1, alpha2.div(alpha_occulting)),
        );
        let a_e2_m_a22 = orekit_max_const(almost0, a_e2.sub(alpha2.mul(alpha2)));

        let p1 = a_s2.mul(a1_o_a_s.acos()).sub(alpha1.mul(a_s2_m_a12.sqrt()));
        let p2 = a_e2.mul(a2_o_a_e.acos()).sub(alpha2.mul(a_e2_m_a22.sqrt()));

        Grad::ONE.sub(p1.add(p2).div(a_s2.mul(Grad::PI)))
    } else {
        Grad::ONE
    }
}

/// Transform a GCRF position into Orekit's `ExtendedPositionProviderAdapter`
/// frame centered on `body` (pure translation by `-body` followed by
/// `Rotation.IDENTITY`).
#[inline]
fn to_body_frame(pos: Vec3, body: Vec3) -> Vec3 {
    body.negate().add(pos).apply_identity_rotation()
}

/// Field counterpart of [`to_body_frame`].
#[inline]
fn to_body_frame_field(pos: FieldVec3, body: FieldVec3) -> FieldVec3 {
    body.negate().add(pos).apply_identity_rotation()
}

fn validate_finite_vec3_m(v: [f64; 3], field: &'static str) -> Result<Vec3, EclipseError> {
    if !v.iter().all(|x| x.is_finite()) {
        return Err(invalid_eclipse_input(field, "not finite"));
    }
    let vm = Vec3::new(v[0] * M_PER_KM, v[1] * M_PER_KM, v[2] * M_PER_KM);
    if !vm.x.is_finite() || !vm.y.is_finite() || !vm.z.is_finite() || !vm.norm().is_finite() {
        return Err(invalid_eclipse_input(field, "out of range"));
    }
    Ok(vm)
}

fn validate_nonzero_vec3_m(v: [f64; 3], field: &'static str) -> Result<Vec3, EclipseError> {
    if !v.iter().all(|x| x.is_finite()) {
        return Err(invalid_eclipse_input(field, "not finite"));
    }
    if v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0 {
        return Err(invalid_eclipse_input(field, "zero vector"));
    }
    let vm = Vec3::new(v[0] * M_PER_KM, v[1] * M_PER_KM, v[2] * M_PER_KM);
    let n = vm.norm();
    if n == 0.0 {
        return Err(invalid_eclipse_input(field, "zero vector"));
    }
    if !n.is_finite() {
        return Err(invalid_eclipse_input(field, "out of range"));
    }
    Ok(vm)
}

#[inline]
fn earth_ellipsoid(model: EarthShadowModel) -> Ellipsoid {
    match model {
        EarthShadowModel::Spherical => Ellipsoid::one_axis(EARTH_RADIUS_M, 0.0),
        EarthShadowModel::Wgs84Oblate => Ellipsoid::one_axis(EARTH_RADIUS_M, WGS84_FLATTENING),
    }
}

/// Moon-only lighting ratio (`double`), matching Orekit 13.1.10's
/// `SolarRadiationPressure.getLightingRatio` with a spherical Moon central body.
pub(crate) fn moon_lighting_ratio_double(
    sat_km: [f64; 3],
    sun_km: [f64; 3],
    moon_km: [f64; 3],
) -> Result<f64, EclipseError> {
    let sat_m = validate_finite_vec3_m(sat_km, "sat_pos")?;
    let sun_m = validate_finite_vec3_m(sun_km, "sun_pos")?;
    let moon_m = validate_finite_vec3_m(moon_km, "moon_pos")?;
    moon_lighting_ratio_m_double(sat_m, sun_m, moon_m)
}

/// Moon-only lighting ratio (`double`), inputs already in metres.
pub(crate) fn moon_lighting_ratio_m_double(
    sat_m: Vec3,
    sun_m: Vec3,
    moon_m: Vec3,
) -> Result<f64, EclipseError> {
    let psat = to_body_frame(sat_m, moon_m);
    let pted = to_body_frame(sun_m, moon_m);
    if !psat.norm().is_finite() {
        return Err(invalid_eclipse_input("moon_pos", "out of range"));
    }
    if !pted.norm().is_finite() {
        return Err(invalid_eclipse_input("sun_pos", "out of range"));
    }
    let moon_shape = Ellipsoid::one_axis(MOON_RADIUS_M, 0.0);
    let (angles, ps, inside_sun) = occultation_angles(psat, pted, moon_shape, SUN_RADIUS_M)?;
    if !ps.norm().is_finite() {
        return Err(invalid_eclipse_input("sun_pos", "out of range"));
    }
    let lighting = if !inside_sun && psat.norm() >= ps.norm() {
        1.0
    } else {
        masking_ratio(angles)
    };
    if !lighting.is_finite() {
        return Err(invalid_eclipse_input("sat_pos", "out of range"));
    }
    Ok(lighting)
}

/// Earth-only lighting ratio (`double`), inputs already in metres.
#[cfg(all(test, sidereon_repo_tests))]
pub(crate) fn earth_lighting_ratio_m_double(
    sat_m: Vec3,
    sun_m: Vec3,
    model: EarthShadowModel,
) -> Result<f64, EclipseError> {
    let earth_shape = earth_ellipsoid(model);
    let (angles, ps, inside_sun) = occultation_angles(sat_m, sun_m, earth_shape, SUN_RADIUS_M)?;
    if !ps.norm().is_finite() {
        return Err(invalid_eclipse_input("sun_pos", "out of range"));
    }
    let lighting = if !inside_sun && sat_m.norm() >= ps.norm() {
        1.0
    } else {
        masking_ratio(angles)
    };
    if !lighting.is_finite() {
        return Err(invalid_eclipse_input("sat_pos", "out of range"));
    }
    Ok(lighting)
}

/// Earth-only lighting ratio (`Field<Gradient>`), inputs already in metres.
#[cfg(all(test, sidereon_repo_tests))]
pub(crate) fn earth_lighting_ratio_m_field(
    sat_m: Vec3,
    sun_m: Vec3,
    model: EarthShadowModel,
) -> Result<Grad, EclipseError> {
    let sat_f = FieldVec3::new(
        Grad::variable(0, sat_m.x),
        Grad::variable(1, sat_m.y),
        Grad::variable(2, sat_m.z),
    );
    let sun_f = FieldVec3::constant(sun_m);
    let earth_shape = earth_ellipsoid(model);
    let (angles, ps, inside_sun) =
        occultation_angles_field(sat_f, sun_f, earth_shape, SUN_RADIUS_M)?;
    if !ps.norm().value.is_finite() {
        return Err(invalid_eclipse_input("sun_pos", "out of range"));
    }
    if !inside_sun && sat_f.norm().value >= ps.norm().value {
        return Ok(Grad::ONE);
    }
    Ok(masking_ratio_field(angles))
}

/// Moon-only lighting ratio (`Field<Gradient>`), inputs already in metres.
#[cfg(all(test, sidereon_repo_tests))]
pub(crate) fn moon_lighting_ratio_m_field(
    sat_m: Vec3,
    sun_m: Vec3,
    moon_m: Vec3,
) -> Result<Grad, EclipseError> {
    let sat_f = FieldVec3::new(
        Grad::variable(0, sat_m.x),
        Grad::variable(1, sat_m.y),
        Grad::variable(2, sat_m.z),
    );
    let sun_f = FieldVec3::constant(sun_m);
    let moon_f = FieldVec3::constant(moon_m);
    let psat = to_body_frame_field(sat_f, moon_f);
    let pted = to_body_frame_field(sun_f, moon_f);
    let moon_shape = Ellipsoid::one_axis(MOON_RADIUS_M, 0.0);
    let (angles, ps, inside_sun) = occultation_angles_field(psat, pted, moon_shape, SUN_RADIUS_M)?;
    if !ps.norm().value.is_finite() {
        return Err(invalid_eclipse_input("sun_pos", "out of range"));
    }
    if !inside_sun && psat.norm().value >= ps.norm().value {
        return Ok(Grad::ONE);
    }
    Ok(masking_ratio_field(angles))
}

/// Combined Earth + Moon lighting ratio (`double`), taking km inputs.
pub(crate) fn earth_moon_lighting_ratio_double(
    sat_km: [f64; 3],
    sun_km: [f64; 3],
    moon_km: [f64; 3],
    model: EarthShadowModel,
) -> Result<f64, EclipseError> {
    let sat_m = validate_nonzero_vec3_m(sat_km, "sat_pos")?;
    let sun_m = validate_nonzero_vec3_m(sun_km, "sun_pos")?;
    let moon_m = validate_nonzero_vec3_m(moon_km, "moon_pos")?;
    earth_moon_lighting_ratio_m_double(sat_m, sun_m, moon_m, model)
}

/// Combined Earth + Moon lighting ratio (`double`) with an explicit Earth
/// `OneAxisEllipsoid` (`ae_m`, `flattening`), taking km inputs (used by
/// oblate-between-spheres property tests).
#[cfg(test)]
pub(crate) fn earth_moon_lighting_ratio_with_ellipsoid_double(
    sat_km: [f64; 3],
    sun_km: [f64; 3],
    moon_km: [f64; 3],
    ae_m: f64,
    flattening: f64,
) -> Result<f64, EclipseError> {
    let sat_m = validate_nonzero_vec3_m(sat_km, "sat_pos")?;
    let sun_m = validate_nonzero_vec3_m(sun_km, "sun_pos")?;
    let moon_m = validate_nonzero_vec3_m(moon_km, "moon_pos")?;
    earth_moon_lighting_ratio_with_shape_m_double(
        sat_m,
        sun_m,
        moon_m,
        Ellipsoid::one_axis(ae_m, flattening),
    )
}

/// Combined Earth + Moon lighting ratio (`double`), inputs in metres.
///
/// Follows Orekit 13.1.10's `SolarRadiationPressure.getLightingRatio`
/// operation-for-operation (`result = 0.0 + l_earth`, mutual slot when
/// `l_moon != 1`, `result += l_moon`, `result -= 1.0`), replacing only
/// Orekit's defective mutual-eclipse subtraction with adding
/// `triple_area / (PI * alpha_sun_sq)`.
pub(crate) fn earth_moon_lighting_ratio_m_double(
    sat_m: Vec3,
    sun_m: Vec3,
    moon_m: Vec3,
    model: EarthShadowModel,
) -> Result<f64, EclipseError> {
    earth_moon_lighting_ratio_with_shape_m_double(sat_m, sun_m, moon_m, earth_ellipsoid(model))
}

fn earth_moon_lighting_ratio_with_shape_m_double(
    sat_m: Vec3,
    sun_m: Vec3,
    moon_m: Vec3,
    earth_shape: Ellipsoid,
) -> Result<f64, EclipseError> {
    let moon_shape = Ellipsoid::one_axis(MOON_RADIUS_M, 0.0);

    let (angles_earth, ps_earth, earth_inside_sun) =
        occultation_angles(sat_m, sun_m, earth_shape, SUN_RADIUS_M)?;
    if !ps_earth.norm().is_finite() {
        return Err(invalid_eclipse_input("sun_pos", "out of range"));
    }
    let psat_moon = to_body_frame(sat_m, moon_m);
    let pted_moon = to_body_frame(sun_m, moon_m);
    if !psat_moon.norm().is_finite() {
        return Err(invalid_eclipse_input("moon_pos", "out of range"));
    }
    if !pted_moon.norm().is_finite() {
        return Err(invalid_eclipse_input("sun_pos", "out of range"));
    }
    let (angles_moon, ps_moon, moon_inside_sun) =
        occultation_angles(psat_moon, pted_moon, moon_shape, SUN_RADIUS_M)?;
    if !ps_moon.norm().is_finite() {
        return Err(invalid_eclipse_input("sun_pos", "out of range"));
    }

    let alpha_sun_sq = angles_earth.occulted_radius * angles_earth.occulted_radius;

    let mut result = 0.0;

    // i = 0 (Earth)
    let earth_behind_sun = !earth_inside_sun && sat_m.norm() >= ps_earth.norm();
    let lighting_earth = if earth_behind_sun {
        1.0
    } else {
        masking_ratio(angles_earth)
    };
    if lighting_earth == 0.0 {
        return Ok(0.0);
    }
    result += lighting_earth;

    // j = 1 (Moon)
    let moon_behind_sun = !moon_inside_sun && psat_moon.norm() >= ps_moon.norm();
    let lighting_moon = if moon_behind_sun {
        1.0
    } else {
        masking_ratio(angles_moon)
    };
    if lighting_moon == 0.0 {
        return Ok(0.0);
    } else if lighting_moon != 1.0 && lighting_earth != 1.0 {
        let a = angles_earth.occulted_radius;
        let b_e = angles_earth.limb_radius;
        let c_e = angles_earth.separation;
        let b_m = angles_moon.limb_radius;
        let c_m = angles_moon.separation;
        let (cos_psi, sin_psi) = position_angle_double(ps_earth, sat_m, psat_moon);
        let triple_area = triple_disc_intersection_area_double([
            (0.0, 0.0, a),
            (c_e, 0.0, b_e),
            (c_m * cos_psi, c_m * sin_psi, b_m),
        ]);
        let correction = triple_area / (fast_math::PI * alpha_sun_sq);
        result += correction;
    }

    // i = 1 (Moon)
    result += lighting_moon;

    // Final term: result -= n - 1
    result -= 1.0;

    Ok(result)
}

/// Combined Earth + Moon lighting ratio (`Field<Gradient>`), taking km inputs
/// and returning value plus partials with respect to satellite position in
/// metres (`d(lighting)/d(pos_m)`).
pub(crate) fn earth_moon_lighting_ratio_field(
    sat_km: [f64; 3],
    sun_km: [f64; 3],
    moon_km: [f64; 3],
    model: EarthShadowModel,
) -> Result<Grad, EclipseError> {
    let sat_m = validate_nonzero_vec3_m(sat_km, "sat_pos")?;
    let sun_m = validate_nonzero_vec3_m(sun_km, "sun_pos")?;
    let moon_m = validate_nonzero_vec3_m(moon_km, "moon_pos")?;
    earth_moon_lighting_ratio_m_field(sat_m, sun_m, moon_m, model)
}

/// Combined Earth + Moon lighting ratio (`Field<Gradient>`), inputs in metres.
pub(crate) fn earth_moon_lighting_ratio_m_field(
    sat_m: Vec3,
    sun_m: Vec3,
    moon_m: Vec3,
    model: EarthShadowModel,
) -> Result<Grad, EclipseError> {
    let sat_f = FieldVec3::new(
        Grad::variable(0, sat_m.x),
        Grad::variable(1, sat_m.y),
        Grad::variable(2, sat_m.z),
    );
    let sun_f = FieldVec3::constant(sun_m);
    let moon_f = FieldVec3::constant(moon_m);

    let earth_shape = earth_ellipsoid(model);
    let moon_shape = Ellipsoid::one_axis(MOON_RADIUS_M, 0.0);

    let (angles_earth, ps_earth, earth_inside_sun) =
        occultation_angles_field(sat_f, sun_f, earth_shape, SUN_RADIUS_M)?;
    if !ps_earth.norm().value.is_finite() {
        return Err(invalid_eclipse_input("sun_pos", "out of range"));
    }
    let psat_moon = to_body_frame_field(sat_f, moon_f);
    let pted_moon = to_body_frame_field(sun_f, moon_f);
    if !psat_moon.norm().value.is_finite() {
        return Err(invalid_eclipse_input("moon_pos", "out of range"));
    }
    if !pted_moon.norm().value.is_finite() {
        return Err(invalid_eclipse_input("sun_pos", "out of range"));
    }
    let (angles_moon, ps_moon, moon_inside_sun) =
        occultation_angles_field(psat_moon, pted_moon, moon_shape, SUN_RADIUS_M)?;
    if !ps_moon.norm().value.is_finite() {
        return Err(invalid_eclipse_input("sun_pos", "out of range"));
    }

    let alpha_sun_sq = angles_earth
        .occulted_radius
        .mul(angles_earth.occulted_radius);

    let mut result = Grad::ZERO;

    // i = 0 (Earth)
    let earth_behind_sun = !earth_inside_sun && sat_f.norm().value >= ps_earth.norm().value;
    let lighting_earth = if earth_behind_sun {
        Grad::ONE
    } else {
        masking_ratio_field(angles_earth)
    };
    if lighting_earth.is_zero() {
        return Ok(Grad::ZERO);
    }
    result = result.add(lighting_earth);

    // j = 1 (Moon)
    let moon_behind_sun = !moon_inside_sun && psat_moon.norm().value >= ps_moon.norm().value;
    let lighting_moon = if moon_behind_sun {
        Grad::ONE
    } else {
        masking_ratio_field(angles_moon)
    };
    if lighting_moon.is_zero() {
        return Ok(Grad::ZERO);
    } else if lighting_moon.value != 1.0 && lighting_earth.value != 1.0 {
        let a = angles_earth.occulted_radius;
        let b_e = angles_earth.limb_radius;
        let c_e = angles_earth.separation;
        let b_m = angles_moon.limb_radius;
        let c_m = angles_moon.separation;
        let (cos_psi, sin_psi) = position_angle_field(ps_earth, sat_f, psat_moon);
        let triple_area = triple_disc_intersection_area_field([
            (Grad::ZERO, Grad::ZERO, a),
            (c_e, Grad::ZERO, b_e),
            (c_m.mul(cos_psi), c_m.mul(sin_psi), b_m),
        ]);
        let correction = triple_area.div(alpha_sun_sq.mul(Grad::PI));
        result = result.add(correction);
    }

    // i = 1 (Moon)
    result = result.add(lighting_moon);

    // Final term: result -= n - 1
    result = result.sub_f64(1.0);

    Ok(result)
}

#[cfg(test)]
pub(crate) fn earth_occultation_angles_double(
    sat_km: [f64; 3],
    sun_km: [f64; 3],
    model: EarthShadowModel,
) -> Result<(f64, f64, f64), EclipseError> {
    let sat_m = validate_nonzero_vec3_m(sat_km, "sat_pos")?;
    let sun_m = validate_nonzero_vec3_m(sun_km, "sun_pos")?;
    let (angles, _, _) = occultation_angles(sat_m, sun_m, earth_ellipsoid(model), SUN_RADIUS_M)?;
    Ok((
        angles.separation,
        angles.limb_radius,
        angles.occulted_radius,
    ))
}

#[cfg(test)]
pub(crate) fn single_body_disc_shadow_orekit(a: f64, b: f64, c: f64) -> f64 {
    1.0 - masking_ratio(OccultationAngles {
        separation: c,
        limb_radius: b,
        occulted_radius: a,
    })
}

#[cfg(test)]
pub(crate) fn combined_two_body_disc_shadow_orekit(
    a: f64,
    b_e: f64,
    c_e: f64,
    b_m: f64,
    c_m: f64,
    cos_psi: f64,
    sin_psi: f64,
) -> f64 {
    let alpha_sun_sq = a * a;
    let mut result = 0.0;
    let l_e = masking_ratio(OccultationAngles {
        separation: c_e,
        limb_radius: b_e,
        occulted_radius: a,
    });
    if l_e == 0.0 {
        return 1.0;
    }
    result += l_e;
    let l_m = masking_ratio(OccultationAngles {
        separation: c_m,
        limb_radius: b_m,
        occulted_radius: a,
    });
    if l_m == 0.0 {
        return 1.0;
    } else if l_m != 1.0 && l_e != 1.0 {
        let triple_area = triple_disc_intersection_area_double([
            (0.0, 0.0, a),
            (c_e, 0.0, b_e),
            (c_m * cos_psi, c_m * sin_psi, b_m),
        ]);
        result += triple_area / (fast_math::PI * alpha_sun_sq);
    }
    result += l_m;
    result -= 1.0;
    1.0 - result
}

#[cfg(test)]
pub(crate) fn two_disc_intersection_area_orekit(r1: f64, r2: f64, d: f64) -> f64 {
    let rmax = fast_math::max(r1, r2);
    two_disc_area_with_delta_double(r1, r2, d, d - rmax)
}

#[cfg(test)]
pub(crate) fn position_angle_for_vectors_double(
    sat_km: [f64; 3],
    sun_km: [f64; 3],
    moon_km: [f64; 3],
) -> (f64, f64) {
    let sat_m = Vec3::new(
        sat_km[0] * M_PER_KM,
        sat_km[1] * M_PER_KM,
        sat_km[2] * M_PER_KM,
    );
    let sun_m = Vec3::new(
        sun_km[0] * M_PER_KM,
        sun_km[1] * M_PER_KM,
        sun_km[2] * M_PER_KM,
    );
    let moon_m = Vec3::new(
        moon_km[0] * M_PER_KM,
        moon_km[1] * M_PER_KM,
        moon_km[2] * M_PER_KM,
    );
    let ps_earth = sat_m.sub(sun_m);
    let psat_moon = to_body_frame(sat_m, moon_m);
    position_angle_double(ps_earth, sat_m, psat_moon)
}

/// Smooth signed `(cos psi, sin psi)` between the Earth and Moon directions
/// around the Sun (`ps = sat - sun`, `psat_e = sat`, `psat_m = sat - moon`).
///
/// Using the scalar triple product `dot(w_e_hat × w_m_hat, u_sun)` gives a
/// signed `sin(psi)` with no `sqrt(0)` singularity on coplanar geometries.
fn position_angle_double(ps: Vec3, psat_e: Vec3, psat_m: Vec3) -> (f64, f64) {
    let w_e = ps.cross(psat_e);
    let w_m = ps.cross(psat_m);
    match (w_e.normalize(), w_m.normalize(), ps.negate().normalize()) {
        (Some(we), Some(wm), Some(u_sun)) => {
            let cos_p = we.dot(wm);
            let sin_p = we.cross(wm).dot(u_sun);
            (cos_p, sin_p)
        }
        _ => (1.0, 0.0),
    }
}

fn position_angle_field(ps: FieldVec3, psat_e: FieldVec3, psat_m: FieldVec3) -> (Grad, Grad) {
    let w_e = ps.cross(psat_e);
    let w_m = ps.cross(psat_m);
    match (w_e.normalize(), w_m.normalize(), ps.negate().normalize()) {
        (Some(we), Some(wm), Some(u_sun)) => {
            let cos_p = we.dot(wm);
            let sin_p = we.cross(wm).dot(u_sun);
            (cos_p, sin_p)
        }
        _ => (Grad::ONE, Grad::ZERO),
    }
}

// ---------------------------------------------------------------------------
// Exact three-disc intersection area (double & Field<Gradient>, FastMath)
// ---------------------------------------------------------------------------

fn disc_pair_dist_and_delta_double(d1: (f64, f64, f64), d2: (f64, f64, f64)) -> (f64, f64) {
    let d = fast_math::hypot(d2.0 - d1.0, d2.1 - d1.1);
    let ((xs, ys, _rs), (xb, yb, rb)) = if d2.2 >= d1.2 { (d1, d2) } else { (d2, d1) };
    let cs = fast_math::hypot(xs, ys);
    if cs > 0.0 && rb > 0.0 && cs < 0.25 * rb {
        let cb = fast_math::hypot(xb, yb);
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

fn disc_pair_dist_and_delta_field(d1: (Grad, Grad, Grad), d2: (Grad, Grad, Grad)) -> (Grad, Grad) {
    let d = d2.0.sub(d1.0).hypot(d2.1.sub(d1.1));
    let ((xs, ys, _rs), (xb, yb, rb)) = if d2.2.value >= d1.2.value {
        (d1, d2)
    } else {
        (d2, d1)
    };
    let cs = xs.hypot(ys);
    if cs.value > 0.0 && rb.value > 0.0 && cs.value < 0.25 * rb.value {
        let cb = xb.hypot(yb);
        if cb.value >= 0.5 * rb.value && cb.value <= 2.0 * rb.value {
            let delta_b = cb.sub(rb);
            let ux = xb.div(cb);
            let uy = yb.div(cb);
            let s_par = xs.mul(ux).add(ys.mul(uy));
            let s_perp = xs.negate().mul(uy).add(ys.mul(ux));
            let q = delta_b.sub(s_par);
            let d_minus_rb = rb
                .mul_f64(2.0)
                .mul(q)
                .add(q.mul(q).add(s_perp.mul(s_perp)))
                .div(d.add(rb));
            return (d, d_minus_rb);
        }
    }
    (d, d.sub(rb))
}

fn heron_factors_double(r1: f64, r2: f64, d: f64, d_minus_rmax: f64) -> (f64, f64, f64, f64) {
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

fn heron_factors_field(
    r1: Grad,
    r2: Grad,
    d: Grad,
    d_minus_rmax: Grad,
) -> (Grad, Grad, Grad, Grad) {
    if r2.value >= r1.value {
        let p1 = r1.sub(d_minus_rmax);
        let p2 = if r1.value >= 0.5 * r2.value {
            d.sub(r2.sub(r1))
        } else {
            r1.add(d_minus_rmax)
        };
        let p3 = d.add(r2.sub(r1));
        let p4 = d.add(r2).add(r1);
        (p1, p2, p3, p4)
    } else {
        let p1 = r2.sub(d_minus_rmax);
        let p2 = d.add(r1.sub(r2));
        let p3 = if r2.value >= 0.5 * r1.value {
            d.sub(r1.sub(r2))
        } else {
            r2.add(d_minus_rmax)
        };
        let p4 = d.add(r1).add(r2);
        (p1, p2, p3, p4)
    }
}

fn theta_minus_sin_double(theta: f64) -> f64 {
    if fast_math::abs(theta) < 0.1 {
        let t2 = theta * theta;
        theta
            * t2
            * (1.0 / 6.0
                + t2 * (-1.0 / 120.0
                    + t2 * (1.0 / 5_040.0 + t2 * (-1.0 / 362_880.0 + t2 / 39_916_800.0))))
    } else {
        theta - fast_math::sin(theta)
    }
}

fn theta_minus_sin_field(theta: Grad) -> Grad {
    if fast_math::abs(theta.value) < 0.1 {
        let t2 = theta.mul(theta);
        let poly = t2
            .div_f64(39_916_800.0)
            .add_f64(-1.0 / 362_880.0)
            .mul(t2)
            .add_f64(1.0 / 5_040.0)
            .mul(t2)
            .add_f64(-1.0 / 120.0)
            .mul(t2)
            .add_f64(1.0 / 6.0);
        theta.mul(t2).mul(poly)
    } else {
        theta.sub(theta.sin())
    }
}

fn two_disc_area_with_delta_double(r1: f64, r2: f64, d: f64, d_minus_rmax: f64) -> f64 {
    let r_min = fast_math::min(r1, r2);
    if d_minus_rmax >= r_min {
        return 0.0;
    }
    if r2 >= r1 && (r1 >= 0.5 * r2 && d <= r2 - r1 || r1 < 0.5 * r2 && d_minus_rmax <= -r1) {
        return fast_math::PI * r1 * r1;
    }
    if r1 > r2 && (r2 >= 0.5 * r1 && d <= r1 - r2 || r2 < 0.5 * r1 && d_minus_rmax <= -r2) {
        return fast_math::PI * r2 * r2;
    }
    let (p1, p2, p3, p4) = heron_factors_double(r1, r2, d, d_minus_rmax);
    if p1 <= 0.0 {
        return 0.0;
    }
    if p2 <= 0.0 {
        return fast_math::PI * r1 * r1;
    }
    if p3 <= 0.0 {
        return fast_math::PI * r2 * r2;
    }
    let s1 = fast_math::sqrt(p1);
    let s2 = fast_math::sqrt(p2);
    let s3 = fast_math::sqrt(p3);
    let s4 = fast_math::sqrt(p4);
    let alpha = 2.0 * fast_math::atan2(s1 * s3, s2 * s4);
    let beta = 2.0 * fast_math::atan2(s1 * s2, s3 * s4);
    0.5 * r1 * r1 * theta_minus_sin_double(2.0 * alpha)
        + 0.5 * r2 * r2 * theta_minus_sin_double(2.0 * beta)
}

fn two_disc_area_with_delta_field(r1: Grad, r2: Grad, d: Grad, d_minus_rmax: Grad) -> Grad {
    let r_min = fast_math::min(r1.value, r2.value);
    if d_minus_rmax.value >= r_min {
        return Grad::ZERO;
    }
    if r2.value >= r1.value
        && (r1.value >= 0.5 * r2.value && d.value <= r2.value - r1.value
            || r1.value < 0.5 * r2.value && d_minus_rmax.value <= -r1.value)
    {
        return Grad::PI.mul(r1).mul(r1);
    }
    if r1.value > r2.value
        && (r2.value >= 0.5 * r1.value && d.value <= r1.value - r2.value
            || r2.value < 0.5 * r1.value && d_minus_rmax.value <= -r2.value)
    {
        return Grad::PI.mul(r2).mul(r2);
    }
    let (p1, p2, p3, p4) = heron_factors_field(r1, r2, d, d_minus_rmax);
    if p1.value <= 0.0 {
        return Grad::ZERO;
    }
    if p2.value <= 0.0 {
        return Grad::PI.mul(r1).mul(r1);
    }
    if p3.value <= 0.0 {
        return Grad::PI.mul(r2).mul(r2);
    }
    let s1 = p1.sqrt();
    let s2 = p2.sqrt();
    let s3 = p3.sqrt();
    let s4 = p4.sqrt();
    let alpha = s1.mul(s3).atan2(s2.mul(s4)).mul_f64(2.0);
    let beta = s1.mul(s2).atan2(s3.mul(s4)).mul_f64(2.0);
    r1.mul(r1)
        .mul_f64(0.5)
        .mul(theta_minus_sin_field(alpha.mul_f64(2.0)))
        .add(
            r2.mul(r2)
                .mul_f64(0.5)
                .mul(theta_minus_sin_field(beta.mul_f64(2.0))),
        )
}

fn split_signed_cyclic_arc_double(center: f64, half_width: f64) -> ([(f64, f64); 2], usize) {
    let pi = fast_math::PI;
    let tau = 2.0 * pi;
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

fn split_signed_cyclic_arc_field(center: Grad, half_width: Grad) -> ([(Grad, Grad); 2], usize) {
    let pi = fast_math::PI;
    let tau = 2.0 * pi;
    let neg_pi = Grad::constant(-pi);
    let pos_pi = Grad::constant(pi);
    let mut out = [(Grad::ZERO, Grad::ZERO); 2];
    if half_width.value <= 0.0 {
        return (out, 0);
    }
    if half_width.value >= pi {
        out[0] = (neg_pi, pos_pi);
        return (out, 1);
    }
    let lo = center.sub(half_width);
    let hi = center.add(half_width);
    if lo.value >= hi.value {
        return (out, 0);
    }
    if lo.value >= -pi && hi.value <= pi {
        out[0] = (lo, hi);
        return (out, 1);
    }
    if lo.value < -pi {
        let lo_wrap = lo.add_f64(tau);
        if lo_wrap.value <= hi.value {
            out[0] = (neg_pi, pos_pi);
            return (out, 1);
        }
        let mut count = 0;
        if hi.value > -pi {
            out[count] = (neg_pi, hi);
            count += 1;
        }
        if lo_wrap.value < pi {
            out[count] = (lo_wrap, pos_pi);
            count += 1;
        }
        (out, count)
    } else {
        let hi_wrap = hi.sub_f64(tau);
        if hi_wrap.value >= lo.value {
            out[0] = (neg_pi, pos_pi);
            return (out, 1);
        }
        let mut count = 0;
        if hi_wrap.value > -pi {
            out[count] = (neg_pi, hi_wrap);
            count += 1;
        }
        if lo.value < pi {
            out[count] = (lo, pos_pi);
            count += 1;
        }
        (out, count)
    }
}

fn intersect_arc_sets_double(
    a: &[(f64, f64)],
    b: &[(f64, f64)],
    out: &mut [(f64, f64); 4],
) -> usize {
    let mut count = 0;
    for &(a0, a1) in a {
        for &(b0, b1) in b {
            let lo = fast_math::max(a0, b0);
            let hi = fast_math::min(a1, b1);
            if hi > lo && count < out.len() {
                out[count] = (lo, hi);
                count += 1;
            }
        }
    }
    count
}

fn intersect_arc_sets_field(
    a: &[(Grad, Grad)],
    b: &[(Grad, Grad)],
    out: &mut [(Grad, Grad); 4],
) -> usize {
    let mut count = 0;
    for &(a0, a1) in a {
        for &(b0, b1) in b {
            let lo = if a0.value >= b0.value { a0 } else { b0 };
            let hi = if a1.value <= b1.value { a1 } else { b1 };
            if hi.value > lo.value && count < out.len() {
                out[count] = (lo, hi);
                count += 1;
            }
        }
    }
    count
}

fn circle_point_local_double(r: f64, c: f64, r_minus_c: f64, s: f64) -> (f64, f64) {
    if c == 0.0 {
        let (sin_s, cos_s) = fast_math::sin_cos(s);
        return (r * cos_s, r * sin_s);
    }
    let s_half = fast_math::sin(0.5 * s);
    let p_par = r_minus_c - 2.0 * r * s_half * s_half;
    let p_perp = r * fast_math::sin(s);
    (p_par, p_perp)
}

fn circle_point_local_field(r: Grad, c: Grad, r_minus_c: Grad, s: Grad) -> (Grad, Grad) {
    if c.value == 0.0 {
        let (sin_s, cos_s) = s.sin_cos();
        return (r.mul(cos_s), r.mul(sin_s));
    }
    let s_half = s.mul_f64(0.5).sin();
    let p_par = r_minus_c.sub(r.mul_f64(2.0).mul(s_half).mul(s_half));
    let p_perp = r.mul(s.sin());
    (p_par, p_perp)
}

pub(crate) fn triple_disc_intersection_area_double(discs: [(f64, f64, f64); 3]) -> f64 {
    for i in 0..3 {
        for j in (i + 1)..3 {
            let (_d, d_minus_rmax) = disc_pair_dist_and_delta_double(discs[i], discs[j]);
            if d_minus_rmax >= fast_math::min(discs[i].2, discs[j].2) {
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
            let (d, d_minus_rmax) = disc_pair_dist_and_delta_double(discs[i], discs[j]);
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
        return fast_math::PI * kept[0].2 * kept[0].2;
    }
    if kept_len == 2 {
        let (d, d_minus_rmax) = disc_pair_dist_and_delta_double(kept[0], kept[1]);
        return two_disc_area_with_delta_double(kept[0].2, kept[1].2, d, d_minus_rmax);
    }

    let mut min_idx = 0;
    for k in 1..3 {
        if kept[k].2 < kept[min_idx].2 {
            min_idx = k;
        }
    }
    let origin_disc = kept[min_idx];
    let (ox, oy, _) = origin_disc;

    let pi = fast_math::PI;
    let mut total = 0.0;
    for i in 0..3 {
        let (xi, yi, ri) = kept[i];
        let vx = ox - xi;
        let vy = oy - yi;
        let (ci, ci_minus_ri) = if i == min_idx {
            (0.0, -ri)
        } else {
            disc_pair_dist_and_delta_double(kept[i], origin_disc)
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
            let (d, d_minus_rmax) = disc_pair_dist_and_delta_double(kept[i], kept[j]);
            let (p1, p2, p3, p4) = heron_factors_double(ri, rj, d, d_minus_rmax);
            if p1 <= 0.0 || p3 <= 0.0 {
                arcs_len = 0;
                break;
            }
            if p2 <= 0.0 {
                continue;
            }
            let alpha = 2.0
                * fast_math::atan2(
                    fast_math::sqrt(p1) * fast_math::sqrt(p3),
                    fast_math::sqrt(p2) * fast_math::sqrt(p4),
                );
            let s_phi = if j == min_idx && ci > 0.0 {
                0.0
            } else {
                let wx = xj - ox;
                let wy = yj - oy;
                let (u_par, u_perp) = if fast_math::hypot(wx, wy) <= d {
                    (
                        ci + (wx * e_par.0 + wy * e_par.1),
                        wx * e_perp.0 + wy * e_perp.1,
                    )
                } else {
                    let dx = xj - xi;
                    let dy = yj - yi;
                    (dx * e_par.0 + dy * e_par.1, dx * e_perp.0 + dy * e_perp.1)
                };
                fast_math::atan2(u_perp, u_par)
            };
            let (other, other_len) = split_signed_cyclic_arc_double(s_phi, alpha);
            let mut next = [(0.0, 0.0); 4];
            let next_len =
                intersect_arc_sets_double(&arcs[..arcs_len], &other[..other_len], &mut next);
            arcs = next;
            arcs_len = next_len;
            if arcs_len == 0 {
                break;
            }
        }

        for &(s1, s2) in &arcs[..arcs_len] {
            let dt = s2 - s1;
            let (a1, b1) = circle_point_local_double(ri, ci, ri_minus_ci, s1);
            let (a2, b2) = circle_point_local_double(ri, ci, ri_minus_ci, s2);
            total += 0.5 * ri * ri * theta_minus_sin_double(dt) + 0.5 * (a1 * b2 - b1 * a2);
        }
    }

    fast_math::max(0.0, total)
}

fn triple_disc_intersection_area_field(discs: [(Grad, Grad, Grad); 3]) -> Grad {
    for i in 0..3 {
        for j in (i + 1)..3 {
            let (_d, d_minus_rmax) = disc_pair_dist_and_delta_field(discs[i], discs[j]);
            if d_minus_rmax.value >= fast_math::min(discs[i].2.value, discs[j].2.value) {
                return Grad::ZERO;
            }
        }
    }

    let mut kept = [(Grad::ZERO, Grad::ZERO, Grad::ZERO); 3];
    let mut kept_len = 0;
    for i in 0..3 {
        let mut redundant = false;
        for j in 0..3 {
            if i == j {
                continue;
            }
            let (d, d_minus_rmax) = disc_pair_dist_and_delta_field(discs[i], discs[j]);
            let ri = discs[i].2.value;
            let rj = discs[j].2.value;
            let contains = if ri < rj {
                false
            } else if rj >= 0.5 * ri {
                d.value <= ri - rj
            } else {
                d_minus_rmax.value <= -rj
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
        return Grad::PI.mul(kept[0].2).mul(kept[0].2);
    }
    if kept_len == 2 {
        let (d, d_minus_rmax) = disc_pair_dist_and_delta_field(kept[0], kept[1]);
        return two_disc_area_with_delta_field(kept[0].2, kept[1].2, d, d_minus_rmax);
    }

    let mut min_idx = 0;
    for k in 1..3 {
        if kept[k].2.value < kept[min_idx].2.value {
            min_idx = k;
        }
    }
    let origin_disc = kept[min_idx];
    let (ox, oy, _) = origin_disc;

    let neg_pi = Grad::constant(-fast_math::PI);
    let pos_pi = Grad::constant(fast_math::PI);
    let mut total = Grad::ZERO;
    for i in 0..3 {
        let (xi, yi, ri) = kept[i];
        let vx = ox.sub(xi);
        let vy = oy.sub(yi);
        let (ci, ci_minus_ri) = if i == min_idx {
            (Grad::ZERO, ri.negate())
        } else {
            disc_pair_dist_and_delta_field(kept[i], origin_disc)
        };
        let ri_minus_ci = ci_minus_ri.negate();
        let (e_par, e_perp) = if ci.value > 0.0 {
            let px = vx.div(ci);
            let py = vy.div(ci);
            ((px, py), (py.negate(), px))
        } else {
            ((Grad::ONE, Grad::ZERO), (Grad::ZERO, Grad::ONE))
        };

        let mut arcs = [
            (neg_pi, pos_pi),
            (Grad::ZERO, Grad::ZERO),
            (Grad::ZERO, Grad::ZERO),
            (Grad::ZERO, Grad::ZERO),
        ];
        let mut arcs_len = 1;

        for (j, &(xj, yj, rj)) in kept.iter().enumerate() {
            if i == j {
                continue;
            }
            let (d, d_minus_rmax) = disc_pair_dist_and_delta_field(kept[i], kept[j]);
            let (p1, p2, p3, p4) = heron_factors_field(ri, rj, d, d_minus_rmax);
            if p1.value <= 0.0 || p3.value <= 0.0 {
                arcs_len = 0;
                break;
            }
            if p2.value <= 0.0 {
                continue;
            }
            let alpha = p1
                .sqrt()
                .mul(p3.sqrt())
                .atan2(p2.sqrt().mul(p4.sqrt()))
                .mul_f64(2.0);
            let s_phi = if j == min_idx && ci.value > 0.0 {
                Grad::ZERO
            } else {
                let wx = xj.sub(ox);
                let wy = yj.sub(oy);
                let (u_par, u_perp) = if wx.hypot(wy).value <= d.value {
                    (
                        ci.add(wx.mul(e_par.0).add(wy.mul(e_par.1))),
                        wx.mul(e_perp.0).add(wy.mul(e_perp.1)),
                    )
                } else {
                    let dx = xj.sub(xi);
                    let dy = yj.sub(yi);
                    (
                        dx.mul(e_par.0).add(dy.mul(e_par.1)),
                        dx.mul(e_perp.0).add(dy.mul(e_perp.1)),
                    )
                };
                u_perp.atan2(u_par)
            };
            let (other, other_len) = split_signed_cyclic_arc_field(s_phi, alpha);
            let mut next = [(Grad::ZERO, Grad::ZERO); 4];
            let next_len =
                intersect_arc_sets_field(&arcs[..arcs_len], &other[..other_len], &mut next);
            arcs = next;
            arcs_len = next_len;
            if arcs_len == 0 {
                break;
            }
        }

        for &(s1, s2) in &arcs[..arcs_len] {
            let dt = s2.sub(s1);
            let (a1, b1) = circle_point_local_field(ri, ci, ri_minus_ci, s1);
            let (a2, b2) = circle_point_local_field(ri, ci, ri_minus_ci, s2);
            let term = ri
                .mul(ri)
                .mul_f64(0.5)
                .mul(theta_minus_sin_field(dt))
                .add(a1.mul(b2).sub(b1.mul(a2)).mul_f64(0.5));
            total = total.add(term);
        }
    }

    orekit_max_const(0.0, total)
}
