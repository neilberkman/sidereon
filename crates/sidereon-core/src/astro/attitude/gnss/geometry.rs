//! Satellite, Sun and orbit-plane geometry shared by the GNSS attitude laws.

use super::{AttitudeSample, AttitudeTrajectory, GnssAttitudeError};
use crate::astro::angles::beta_angle_from_cos_rad;
use crate::astro::math::vec3::{add3, cross3, dot3, neg3, norm3, scale3, sub3, unit3};
use crate::tolerances::YAW_SINGULARITY_EPS_RAD;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// Nominal yaw steering formed with RTKLIB's `sat_yaw` and `yaw_angle`
/// arithmetic, in the same operation order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RtklibNominalYaw {
    /// Yaw as RTKLIB forms it, `atan2(-tan(beta), sin(mu)) + pi`, in `[0, 2 pi]`.
    pub(crate) yaw_rad: f64,
    /// Sun elevation above the orbit plane.
    pub(crate) beta_rad: f64,
    /// Unit body X axis.
    pub(crate) x: [f64; 3],
    /// Unit body Y axis.
    pub(crate) y: [f64; 3],
    /// Unit position vector; body Z is its negative.
    pub(crate) e_r: [f64; 3],
}

/// RTKLIB nominal yaw for a satellite at `position` moving with `velocity`
/// (both in one frame in which `velocity` is the inertial velocity direction)
/// and the Sun at `sun`, all in that frame.
///
/// Returns `None` where RTKLIB's unit vectors are undefined: a zero position,
/// Sun vector, orbit normal (`position x velocity`) or `sun x normal`.
///
/// The arithmetic and its order are RTKLIB's: the PPP wind-up correction calls
/// this and must keep its bits.
pub(crate) fn rtklib_nominal_yaw(
    position: [f64; 3],
    velocity: [f64; 3],
    sun: [f64; 3],
) -> Option<RtklibNominalYaw> {
    let n = cross3(position, velocity);
    let p = cross3(sun, n);

    let es = unit3(position)?;
    let esun = unit3(sun)?;
    let en = unit3(n)?;
    let ep = unit3(p)?;

    let beta = beta_angle_from_cos_rad(dot3(esun, en));
    let ee = libm::acos(dot3(es, ep).clamp(-1.0, 1.0));
    let mut mu = PI / 2.0 + if dot3(es, esun) <= 0.0 { -ee } else { ee };

    if mu < -PI / 2.0 {
        mu += TAU;
    } else if mu >= PI / 2.0 {
        mu -= TAU;
    }

    let yaw = rtklib_yaw_angle(beta, mu);
    let ex = cross3(en, es);
    let cosy = libm::cos(yaw);
    let siny = libm::sin(yaw);
    let x = add3(scale3(en, -siny), scale3(ex, cosy));
    let y = add3(scale3(en, -cosy), scale3(ex, -siny));
    Some(RtklibNominalYaw {
        yaw_rad: yaw,
        beta_rad: beta,
        x,
        y,
        e_r: es,
    })
}

/// RTKLIB `yaw_angle` for the nominal law, including its singular branch.
fn rtklib_yaw_angle(beta: f64, mu: f64) -> f64 {
    if beta.abs() < YAW_SINGULARITY_EPS_RAD && mu.abs() < YAW_SINGULARITY_EPS_RAD {
        PI
    } else {
        libm::atan2(-libm::tan(beta), libm::sin(mu)) + PI
    }
}

/// Wrap an angle into `(-pi, pi]`.
pub(super) fn wrap_pi(angle: f64) -> f64 {
    let wrapped = angle - TAU * libm::floor((angle + PI) / TAU);
    // `wrapped` is in [-pi, pi); move -pi onto +pi.
    if wrapped <= -PI {
        wrapped + TAU
    } else {
        wrapped
    }
}

/// `acos` of a cosine formed from rounded values, which rounding can push
/// just outside [-1, 1].
pub(super) fn acos_clamped(cosine: f64) -> f64 {
    libm::acos(cosine.clamp(-1.0, 1.0))
}

/// `asin` of a sine formed from rounded values.
fn asin_clamped(sine: f64) -> f64 {
    libm::asin(sine.clamp(-1.0, 1.0))
}

/// Check a bound on the norm of a vector the laws form from the inputs.
/// Norms are taken as `sqrt(x . x)`, so the bound's square must be finite;
/// a bound whose square underflows leaves no direction to normalise.
pub(super) fn check_product(field: &'static str, bound: f64) -> Result<(), GnssAttitudeError> {
    let square = bound * bound;
    if !square.is_finite() {
        Err(GnssAttitudeError::NonFinite { field })
    } else if square < f64::MIN_POSITIVE {
        Err(GnssAttitudeError::ZeroLength { field })
    } else {
        Ok(())
    }
}

/// Hipparchus `Vector3D.angle`: `acos` of the normalised dot product, switching
/// to `asin` of the cross product norm within 1e-4 of parallel.
pub(super) fn vector_angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    let norm_product = norm3(a) * norm3(b);
    let dot = dot3(a, b);
    let threshold = norm_product * 0.9999;
    if dot < -threshold || dot > threshold {
        let cross = norm3(cross3(a, b));
        if dot >= 0.0 {
            asin_clamped(cross / norm_product)
        } else {
            PI - asin_clamped(cross / norm_product)
        }
    } else {
        acos_clamped(dot / norm_product)
    }
}

/// Validate a caller-supplied attitude sample for nominal yaw steering:
/// finite components, finite and non-zero norms of the position, Sun position
/// and orbit normal, and a representable `sun x (position x velocity)`.
pub(super) fn validate_sample(sample: &AttitudeSample) -> Result<(), GnssAttitudeError> {
    for (field, vector) in [
        ("position_m", sample.position_m),
        ("velocity_m_s", sample.velocity_m_s),
        ("sun_position_m", sample.sun_position_m),
    ] {
        if !vector.iter().all(|c| c.is_finite()) || !norm3(vector).is_finite() {
            return Err(GnssAttitudeError::NonFinite { field });
        }
    }
    for (field, vector) in [
        ("position_m", sample.position_m),
        ("sun_position_m", sample.sun_position_m),
        (
            "position_m x velocity_m_s",
            cross3(sample.position_m, sample.velocity_m_s),
        ),
    ] {
        let norm = norm3(vector);
        if !norm.is_finite() {
            return Err(GnssAttitudeError::NonFinite { field });
        }
        if norm == 0.0 {
            return Err(GnssAttitudeError::ZeroLength { field });
        }
    }
    check_product(
        "sun_position_m x (position_m x velocity_m_s)",
        norm3(sample.sun_position_m) * norm3(cross3(sample.position_m, sample.velocity_m_s)),
    )
}

/// Validate a trajectory sample for the attitude laws: [`validate_sample`]
/// and the further products [`Context`] forms (`sun x position` and the
/// orbital rate `|r x v| / |r|^2`).
fn validate_law_sample(sample: &AttitudeSample) -> Result<(), GnssAttitudeError> {
    validate_sample(sample)?;
    let r = norm3(sample.position_m);
    let s = norm3(sample.sun_position_m);
    // The orbital rate divides by |r|^2.
    check_product("position_m", r)?;
    check_product("sun_position_m x position_m", s * r)?;
    let relative: [f64; 3] =
        std::array::from_fn(|k| sample.sun_position_m[k] - sample.position_m[k]);
    if !norm3(relative).is_finite() {
        return Err(GnssAttitudeError::NonFinite {
            field: "sun_position_m - position_m",
        });
    }
    let mu_rate = norm3(cross3(sample.position_m, sample.velocity_m_s)) / (r * r);
    if !mu_rate.is_finite() || mu_rate == 0.0 {
        return Err(GnssAttitudeError::NonFinite {
            field: "|position_m x velocity_m_s| / |position_m|^2",
        });
    }
    Ok(())
}

/// Read and validate the trajectory at `t_s`.
pub(super) fn sample_at<T: AttitudeTrajectory + ?Sized>(
    trajectory: &T,
    t_s: f64,
) -> Result<AttitudeSample, GnssAttitudeError> {
    if !t_s.is_finite() {
        return Err(GnssAttitudeError::NonFinite { field: "t_s" });
    }
    let sample = trajectory
        .sample(t_s)
        .ok_or(GnssAttitudeError::TrajectoryUnavailable { t_s })?;
    validate_law_sample(&sample)?;
    Ok(sample)
}

/// Orbit and Sun geometry at one instant, in the quantities Orekit's
/// `GNSSAttitudeContext` forms.
#[derive(Debug, Clone, Copy)]
pub(super) struct Context {
    pub(super) t_s: f64,
    pub(super) position: [f64; 3],
    pub(super) sun: [f64; 3],
    /// Unit position.
    pub(super) e_r: [f64; 3],
    /// Unit orbit normal, `position x velocity`.
    pub(super) e_n: [f64; 3],
    /// Orbital angular rate `|r x v| / |r|^2`.
    pub(super) mu_rate: f64,
    /// Cosine of the Sun-Earth-satellite angle.
    pub(super) svb_cos: f64,
    /// Sun elevation above the orbit plane.
    pub(super) beta: f64,
}

impl Context {
    pub(super) fn at<T: AttitudeTrajectory + ?Sized>(
        trajectory: &T,
        t_s: f64,
    ) -> Result<Self, GnssAttitudeError> {
        let sample = sample_at(trajectory, t_s)?;
        Self::from_sample(t_s, &sample)
    }

    /// The geometry of a sample that passed [`sample_at`]'s validation.
    /// Errors when the orbital rate, a quotient by `|r|^2`, overflows;
    /// validation keeps `|r|^2` normal.
    pub(super) fn from_sample(
        t_s: f64,
        sample: &AttitudeSample,
    ) -> Result<Self, GnssAttitudeError> {
        let r = sample.position_m;
        let v = sample.velocity_m_s;
        let s = sample.sun_position_m;
        let h = cross3(r, v);
        let r_norm = norm3(r);
        let r2 = dot3(r, r);
        let mu_rate = norm3(h) / r2;
        if !mu_rate.is_finite() || mu_rate == 0.0 {
            return Err(GnssAttitudeError::NonFinite {
                field: "|position_m x velocity_m_s| / |position_m|^2",
            });
        }
        let s_norm = norm3(s);
        Ok(Self {
            t_s,
            position: r,
            sun: s,
            e_r: scale3(r, 1.0 / r_norm),
            e_n: scale3(h, 1.0 / norm3(h)),
            mu_rate,
            svb_cos: dot3(s, r) / (s_norm * r_norm),
            beta: FRAC_PI_2 - vector_angle(s, h),
        })
    }

    pub(super) fn sun_side(&self) -> bool {
        self.svb_cos > 0.0
    }

    /// Orekit `inOrbitPlaneAbsoluteAngle`: the in-plane angle whose
    /// Sun-satellite angle is `angle` at this instant's beta.
    pub(super) fn in_orbit_plane_absolute_angle(&self, angle: f64) -> f64 {
        acos_clamped(libm::cos(angle) / libm::cos(self.beta))
    }

    /// Along-track unit vector `e_n x e_r`.
    pub(super) fn e_along(&self) -> [f64; 3] {
        cross3(self.e_n, self.e_r)
    }

    /// Body axes for a yaw angle about the radial axis: Z toward the Earth,
    /// `X . e_n = -sin(yaw)` and `Y . e_n = -cos(yaw)`.
    pub(super) fn axes_for_yaw(&self, yaw: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let along = self.e_along();
        let (sin_y, cos_y) = (libm::sin(yaw), libm::cos(yaw));
        let x = add3(scale3(self.e_n, -sin_y), scale3(along, cos_y));
        let y = add3(scale3(self.e_n, -cos_y), scale3(along, -sin_y));
        (x, y, neg3(self.e_r))
    }

    /// Nominal yaw from the yaw-steering axes: body Y along `sun x r`.
    pub(super) fn geometric_nominal_yaw(&self) -> f64 {
        let y = unit3(cross3(self.sun, self.position)).unwrap_or([0.0; 3]);
        let x = cross3(self.e_r, y);
        libm::atan2(-dot3(x, self.e_n), -dot3(y, self.e_n))
    }

    /// In-plane argument of the satellite measured from noon (the Sun's
    /// projection on the orbit plane) in the direction of motion.
    pub(super) fn argument_from_noon(&self) -> f64 {
        let sun_in_plane = sub3(self.sun, scale3(self.e_n, dot3(self.sun, self.e_n)));
        libm::atan2(
            dot3(cross3(sun_in_plane, self.e_r), self.e_n),
            dot3(sun_in_plane, self.e_r),
        )
    }
}

/// Orekit `computePhi`: yaw `atan2(-tan(beta), sin(angle))`.
pub(super) fn compute_phi(beta: f64, in_orbit_plane_angle: f64) -> f64 {
    libm::atan2(-libm::tan(beta), libm::sin(in_orbit_plane_angle))
}

/// Sun elevation above the orbit plane at `t_s`, as Orekit's
/// `GNSSAttitudeContext.beta` forms it.
pub(super) fn beta_at<T: AttitudeTrajectory + ?Sized>(
    trajectory: &T,
    t_s: f64,
) -> Result<f64, GnssAttitudeError> {
    let sample = sample_at(trajectory, t_s)?;
    Ok(FRAC_PI_2
        - vector_angle(
            sample.sun_position_m,
            cross3(sample.position_m, sample.velocity_m_s),
        ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_pi_lands_in_the_half_open_interval() {
        assert_eq!(wrap_pi(PI), PI);
        assert_eq!(wrap_pi(-PI), PI);
        assert!((wrap_pi(3.0 * PI) - PI).abs() < 1e-15);
        assert!((wrap_pi(-0.5) + 0.5).abs() == 0.0);
        assert!((wrap_pi(TAU + 0.25) - 0.25).abs() < 1e-15);
    }
}
