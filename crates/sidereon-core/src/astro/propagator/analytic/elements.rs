//! Circular, Keplerian and equinoctial element sets used by the analytic
//! theories, and the two-body conversions between them and Cartesian states.
//!
//! The circular parameters follow Eckstein and Hechler: `ex = e cos(w)`,
//! `ey = e sin(w)` and the argument of latitude `alpha = w + anomaly`, which
//! stay defined as the eccentricity goes to zero. The equinoctial elements
//! (`ex = e cos(w + raan)`, `ey = e sin(w + raan)`, `hx = tan(i/2) cos(raan)`,
//! `hy = tan(i/2) sin(raan)`, true longitude `lv = raan + w + v`) are the
//! variables of the osculating-to-mean fixed point. The formulas are the
//! standard two-body relations (Vallado, "Fundamentals of Astrodynamics and
//! Applications", 4th ed., section 2.4; Broucke and Cefola, "On the
//! equinoctial orbit elements", Celestial Mechanics 5 (1972) 303-310), in the
//! arrangement Orekit's `CircularOrbit`, `KeplerianOrbit` and
//! `CircularLatitudeArgumentUtility` use.

use super::{finite, normalize_angle, AnalyticPropagationError};
use core::f64::consts::PI;

/// Circular orbital parameters with the mean argument of latitude.
///
/// The element set of the Eckstein-Hechler theory: well defined for circular
/// orbits, singular only for equatorial ones (where the node is undefined).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircularElements {
    /// Semi-major axis, km.
    pub a_km: f64,
    /// `e cos(w)`, eccentricity vector component along the ascending node.
    pub ex: f64,
    /// `e sin(w)`, eccentricity vector component normal to the node in the
    /// orbit plane.
    pub ey: f64,
    /// Inclination, radians. Any finite value: one outside `[0, pi]` means
    /// the orbit [`Self::to_cartesian`] gives, and the theories use that orbit
    /// with the inclination in `[0, pi]` (see the
    /// [module documentation](super)).
    pub i_rad: f64,
    /// Right ascension of the ascending node, radians.
    pub raan_rad: f64,
    /// Mean argument of latitude `w + M`, radians.
    pub mean_latitude_argument_rad: f64,
}

/// Keplerian orbital elements with the mean anomaly.
///
/// The element set of the Brouwer-Lyddane theory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeplerianElements {
    /// Semi-major axis, km.
    pub a_km: f64,
    /// Eccentricity, in `[0, 1)`.
    pub e: f64,
    /// Inclination, radians. Any finite value: one outside `[0, pi]` means
    /// the orbit [`Self::to_cartesian`] gives, and the theories use that orbit
    /// with the inclination in `[0, pi]` (see the
    /// [module documentation](super)).
    pub i_rad: f64,
    /// Argument of perigee, radians.
    pub arg_perigee_rad: f64,
    /// Right ascension of the ascending node, radians.
    pub raan_rad: f64,
    /// Mean anomaly, radians.
    pub mean_anomaly_rad: f64,
}

/// Equinoctial elements with the true longitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Equinoctial {
    pub a: f64,
    pub ex: f64,
    pub ey: f64,
    pub hx: f64,
    pub hy: f64,
    pub lv: f64,
}

type Pv = ([f64; 3], [f64; 3]);

/// Retrograde equatorial limit of the equinoctial elements, as in Orekit.
const RETROGRADE_EQUATORIAL_LIMIT: f64 = 1.0e-10;

/// The inclination brought into `[0, pi]`, and whether the argument of
/// perigee and the node move by `pi` with it.
///
/// An inclination already in `[0, pi]` is returned unchanged. Any other is
/// reduced modulo `2 pi` into `[-pi, pi)`, and a negative one `i` replaced by
/// `-i`: the orbit `(i, w, raan)` is the orbit `(-i, w + pi, raan + pi)`,
/// because `sin i`, `sin w`, `cos w`, `sin raan` and `cos raan` all change
/// sign and `cos i` does not, which leaves both in-plane unit vectors of
/// [`orbit_plane_to_inertial`] unchanged.
fn canonical_inclination(i: f64) -> (f64, bool) {
    if (0.0..=PI).contains(&i) {
        return (i, false);
    }
    let reduced = normalize_angle(i, 0.0);
    if reduced < 0.0 {
        (-reduced, true)
    } else {
        (reduced, false)
    }
}

/// `angle + pi` in `[-pi, pi)`, reduced first so that `pi` is not rounded
/// away far from zero.
fn plus_pi(angle: f64) -> f64 {
    normalize_angle(normalize_angle(angle, 0.0) + PI, 0.0)
}

impl CircularElements {
    /// Circular parameters of a two-body Cartesian state.
    ///
    /// `position_km` and `velocity_km_s` are inertial; `mu_km3_s2` is the
    /// gravitational parameter. Returns [`AnalyticPropagationError::NotElliptic`]
    /// for a parabolic, hyperbolic or rectilinear state.
    pub fn from_cartesian(
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        mu_km3_s2: f64,
    ) -> Result<Self, AnalyticPropagationError> {
        let [x, y, z] = position_km;
        let [vx, vy, vz] = velocity_km_s;
        for (value, field) in [
            (x, "position_km"),
            (y, "position_km"),
            (z, "position_km"),
            (vx, "velocity_km_s"),
            (vy, "velocity_km_s"),
            (vz, "velocity_km_s"),
            (mu_km3_s2, "mu_km3_s2"),
        ] {
            finite(value, field)?;
        }
        if mu_km3_s2 <= 0.0 {
            return Err(AnalyticPropagationError::InvalidParameter {
                field: "mu_km3_s2",
                reason: "must be positive",
            });
        }
        // The conversion is homogeneous in a length and a time scale, so
        // scaling by powers of two changes no rounding: positions are brought
        // to order 1 and the time unit chosen so that mu is of order 1. That
        // keeps every square and product below in range for any finite state
        // whose elements are representable.
        let p_max = x.abs().max(y.abs()).max(z.abs());
        if p_max == 0.0 {
            return Err(AnalyticPropagationError::InvalidParameter {
                field: "position_km",
                reason: "must be nonzero",
            });
        }
        let length_exp = libm::ilogb(p_max);
        let time_exp = (libm::ilogb(mu_km3_s2) - 3 * length_exp).div_euclid(2);
        let [x, y, z] = [x, y, z].map(|c| libm::scalbn(c, -length_exp));
        let [vx, vy, vz] = [vx, vy, vz].map(|c| libm::scalbn(c, -length_exp - time_exp));
        let mu = libm::scalbn(mu_km3_s2, -3 * length_exp - 2 * time_exp);

        let r2 = x * x + y * y + z * z;
        let r = libm::sqrt(r2);
        let v2 = vx * vx + vy * vy + vz * vz;
        let rv2_on_mu = r * v2 / mu;
        let denominator = 2.0 - rv2_on_mu;
        if denominator.is_nan() || denominator <= 0.0 {
            return Err(AnalyticPropagationError::NotElliptic { field: "state" });
        }
        let a = r / denominator;

        let hx = y * vz - z * vy;
        let hy = z * vx - x * vz;
        let hz = x * vy - y * vx;
        let h_xy = libm::sqrt(hx * hx + hy * hy);
        if h_xy == 0.0 && hz == 0.0 {
            return Err(AnalyticPropagationError::NotElliptic { field: "state" });
        }
        let i = libm::atan2(h_xy, hz);
        // Node direction K x h = (-hy, hx, 0).
        let raan = libm::atan2(hx, -hy);
        let (sin_raan, cos_raan) = (libm::sin(raan), libm::cos(raan));
        let (sin_i, cos_i) = (libm::sin(i), libm::cos(i));
        let x2 = (x * cos_raan + y * sin_raan) / a;
        let y2 = ((y * cos_raan - x * sin_raan) * cos_i + z * sin_i) / a;

        let e_se = (x * vx + y * vy + z * vz) / libm::sqrt(mu * a);
        let e_ce = rv2_on_mu - 1.0;
        let e2 = e_ce * e_ce + e_se * e_se;
        // A state whose eccentricity rounds to 1 or more (a nearly
        // rectilinear one) has no elliptic elements.
        if e2.is_nan() || e2 >= 1.0 {
            return Err(AnalyticPropagationError::NotElliptic { field: "state" });
        }
        let f = e_ce - e2;
        let g = libm::sqrt(1.0 - e2) * e_se;
        let a_on_r = a / r;
        let a2_on_r2 = a_on_r * a_on_r;
        let ex = a2_on_r2 * (f * x2 + g * y2);
        let ey = a2_on_r2 * (f * y2 - g * x2);

        let beta = 1.0 / (1.0 + libm::sqrt(1.0 - ex * ex - ey * ey));
        let alpha_e = libm::atan2(y2 + ey + e_se * beta * ex, x2 + ex - e_se * beta * ey);
        let alpha_m = circular_eccentric_to_mean(ex, ey, alpha_e);
        let a_km = libm::scalbn(a, length_exp);
        if !a_km.is_finite() {
            return Err(AnalyticPropagationError::InvalidParameter {
                field: "state",
                reason: "the semi-major axis overflows",
            });
        }
        let elements = Self {
            a_km,
            ex,
            ey,
            i_rad: i,
            raan_rad: raan,
            mean_latitude_argument_rad: alpha_m,
        };
        // A nearly rectilinear state can still round the eccentricity vector
        // to unit length.
        elements
            .validate("state")
            .map_err(|_| AnalyticPropagationError::NotElliptic { field: "state" })?;
        Ok(elements)
    }

    /// Two-body Cartesian state of these elements: position km, velocity km/s.
    pub fn to_cartesian(self, mu_km3_s2: f64) -> Result<Pv, AnalyticPropagationError> {
        self.validate("elements")?;
        finite(mu_km3_s2, "mu_km3_s2")?;
        if mu_km3_s2 <= 0.0 {
            return Err(AnalyticPropagationError::InvalidParameter {
                field: "mu_km3_s2",
                reason: "must be positive",
            });
        }
        let alpha_e = circular_mean_to_eccentric(
            self.ex,
            self.ey,
            normalize_angle(self.mean_latitude_argument_rad, 0.0),
        )?;
        let (sin_ae, cos_ae) = (libm::sin(alpha_e), libm::cos(alpha_e));
        let (ex, ey) = (self.ex, self.ey);
        let beta = 1.0 / (1.0 + libm::sqrt(1.0 - ex * ex - ey * ey));
        let u = beta * ex * ey * sin_ae - ex + (1.0 - beta * ey * ey) * cos_ae;
        let v = beta * ex * ey * cos_ae - ey + (1.0 - beta * ex * ex) * sin_ae;
        // d(alpha_e)/dt = n / (1 - ex cos(alpha_e) - ey sin(alpha_e)) and
        // a n = sqrt(mu / a).
        let factor = sqrt_ratio(mu_km3_s2, self.a_km) / (1.0 - ex * cos_ae - ey * sin_ae);
        let u_dot = factor * (beta * ex * ey * cos_ae - (1.0 - beta * ey * ey) * sin_ae);
        let v_dot = factor * ((1.0 - beta * ex * ex) * cos_ae - beta * ex * ey * sin_ae);
        check_state(orbit_plane_to_inertial(
            self.a_km * u,
            self.a_km * v,
            u_dot,
            v_dot,
            self.i_rad,
            self.raan_rad,
            0.0,
        ))
    }

    /// Eccentricity `sqrt(ex^2 + ey^2)`.
    pub fn eccentricity(&self) -> f64 {
        libm::sqrt(self.ex * self.ex + self.ey * self.ey)
    }

    /// The same orbit with the inclination in `[0, pi]`, returned unchanged
    /// if it already is. Moving `w` by `pi` negates `(ex, ey)` and moves the
    /// argument of latitude by `pi`, which negates the in-plane position and
    /// velocity; with the node moved by `pi` and the inclination negated, the
    /// in-plane unit vectors (`w = 0` in [`orbit_plane_to_inertial`]) are
    /// negated too, so the Cartesian state is unchanged.
    pub(crate) fn canonical(&self) -> Self {
        let (i_rad, flipped) = canonical_inclination(self.i_rad);
        if flipped {
            Self {
                ex: -self.ex,
                ey: -self.ey,
                i_rad,
                raan_rad: plus_pi(self.raan_rad),
                mean_latitude_argument_rad: plus_pi(self.mean_latitude_argument_rad),
                ..*self
            }
        } else {
            Self { i_rad, ..*self }
        }
    }

    pub(crate) fn validate(&self, field: &'static str) -> Result<(), AnalyticPropagationError> {
        for value in [
            self.a_km,
            self.ex,
            self.ey,
            self.i_rad,
            self.raan_rad,
            self.mean_latitude_argument_rad,
        ] {
            finite(value, field)?;
        }
        if self.a_km <= 0.0 || self.ex * self.ex + self.ey * self.ey >= 1.0 {
            return Err(AnalyticPropagationError::NotElliptic { field });
        }
        Ok(())
    }

    pub(crate) fn to_equinoctial(self) -> Result<Equinoctial, AnalyticPropagationError> {
        // In [0, pi], the one retrograde equatorial inclination is pi.
        let c = self.canonical();
        check_not_retrograde_equatorial(c.i_rad)?;
        let (sin_raan, cos_raan) = (libm::sin(c.raan_rad), libm::cos(c.raan_rad));
        let t = libm::tan(0.5 * c.i_rad);
        let raan = normalize_angle(c.raan_rad, 0.0);
        let alpha_v = circular_mean_to_true(
            c.ex,
            c.ey,
            normalize_angle(c.mean_latitude_argument_rad, 0.0),
        )?;
        Ok(Equinoctial {
            a: c.a_km,
            ex: c.ex * cos_raan - c.ey * sin_raan,
            ey: c.ey * cos_raan + c.ex * sin_raan,
            hx: cos_raan * t,
            hy: sin_raan * t,
            lv: alpha_v + raan,
        })
    }

    pub(crate) fn from_equinoctial(eq: &Equinoctial) -> Self {
        let h = libm::sqrt(eq.hx * eq.hx + eq.hy * eq.hy);
        let i = 2.0 * libm::atan(h);
        let raan = libm::atan2(eq.hy, eq.hx);
        let (cos_raan, sin_raan) = if h == 0.0 {
            (libm::cos(raan), libm::sin(raan))
        } else {
            (eq.hx / h, eq.hy / h)
        };
        let ex = eq.ex * cos_raan + eq.ey * sin_raan;
        let ey = eq.ey * cos_raan - eq.ex * sin_raan;
        let alpha_m = circular_true_to_mean(ex, ey, eq.lv - raan);
        Self {
            a_km: eq.a,
            ex,
            ey,
            i_rad: i,
            raan_rad: raan,
            mean_latitude_argument_rad: alpha_m,
        }
    }
}

impl KeplerianElements {
    /// Keplerian elements of a two-body Cartesian state.
    ///
    /// The argument of perigee and the mean anomaly are measured from the
    /// circular parameters, so a circular orbit gets `w = 0` and an equatorial
    /// one `raan = 0` or `pi`. Returns [`AnalyticPropagationError::NotElliptic`]
    /// for a parabolic, hyperbolic or rectilinear state.
    pub fn from_cartesian(
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        mu_km3_s2: f64,
    ) -> Result<Self, AnalyticPropagationError> {
        let c = CircularElements::from_cartesian(position_km, velocity_km_s, mu_km3_s2)?;
        let pa = libm::atan2(c.ey, c.ex);
        Ok(Self {
            a_km: c.a_km,
            e: c.eccentricity(),
            i_rad: c.i_rad,
            arg_perigee_rad: pa,
            raan_rad: c.raan_rad,
            mean_anomaly_rad: normalize_angle(c.mean_latitude_argument_rad - pa, 0.0),
        })
    }

    /// Two-body Cartesian state of these elements: position km, velocity km/s.
    pub fn to_cartesian(self, mu_km3_s2: f64) -> Result<Pv, AnalyticPropagationError> {
        self.validate("elements")?;
        finite(mu_km3_s2, "mu_km3_s2")?;
        if mu_km3_s2 <= 0.0 {
            return Err(AnalyticPropagationError::InvalidParameter {
                field: "mu_km3_s2",
                reason: "must be positive",
            });
        }
        self.to_cartesian_unchecked(mu_km3_s2)
    }

    pub(crate) fn to_cartesian_unchecked(self, mu: f64) -> Result<Pv, AnalyticPropagationError> {
        let e = self.e;
        let big_e = elliptic_mean_to_eccentric(normalize_angle(self.mean_anomaly_rad, 0.0), e)?;
        let (sin_e, cos_e) = (libm::sin(big_e), libm::cos(big_e));
        let s = libm::sqrt(1.0 - e * e);
        let x = self.a_km * (cos_e - e);
        let y = self.a_km * s * sin_e;
        let factor = sqrt_ratio(mu, self.a_km) / (1.0 - e * cos_e);
        check_state(orbit_plane_to_inertial(
            x,
            y,
            -sin_e * factor,
            s * cos_e * factor,
            self.i_rad,
            self.raan_rad,
            self.arg_perigee_rad,
        ))
    }

    /// The same orbit with the inclination in `[0, pi]`, returned unchanged
    /// if it already is (see [`canonical_inclination`]).
    pub(crate) fn canonical(&self) -> Self {
        let (i_rad, flipped) = canonical_inclination(self.i_rad);
        if flipped {
            Self {
                i_rad,
                arg_perigee_rad: plus_pi(self.arg_perigee_rad),
                raan_rad: plus_pi(self.raan_rad),
                ..*self
            }
        } else {
            Self { i_rad, ..*self }
        }
    }

    pub(crate) fn validate(&self, field: &'static str) -> Result<(), AnalyticPropagationError> {
        for value in [
            self.a_km,
            self.e,
            self.i_rad,
            self.arg_perigee_rad,
            self.raan_rad,
            self.mean_anomaly_rad,
        ] {
            finite(value, field)?;
        }
        if self.a_km <= 0.0 || self.e < 0.0 || self.e >= 1.0 {
            return Err(AnalyticPropagationError::NotElliptic { field });
        }
        Ok(())
    }

    pub(crate) fn to_equinoctial(self) -> Result<Equinoctial, AnalyticPropagationError> {
        // In [0, pi], the one retrograde equatorial inclination is pi.
        let k = self.canonical();
        check_not_retrograde_equatorial(k.i_rad)?;
        let (sin_raan, cos_raan) = (libm::sin(k.raan_rad), libm::cos(k.raan_rad));
        let t = libm::tan(0.5 * k.i_rad);
        let pa_raan = normalize_angle(k.arg_perigee_rad, 0.0) + normalize_angle(k.raan_rad, 0.0);
        let v = elliptic_mean_to_true(normalize_angle(k.mean_anomaly_rad, 0.0), k.e)?;
        Ok(Equinoctial {
            a: k.a_km,
            ex: k.e * libm::cos(pa_raan),
            ey: k.e * libm::sin(pa_raan),
            hx: cos_raan * t,
            hy: sin_raan * t,
            lv: pa_raan + v,
        })
    }

    pub(crate) fn from_equinoctial(eq: &Equinoctial) -> Self {
        let e = libm::sqrt(eq.ex * eq.ex + eq.ey * eq.ey);
        let i = 2.0 * libm::atan(libm::sqrt(eq.hx * eq.hx + eq.hy * eq.hy));
        let raan = libm::atan2(eq.hy, eq.hx);
        let pa = libm::atan2(eq.ey, eq.ex) - raan;
        let v = eq.lv - raan - pa;
        Self {
            a_km: eq.a,
            e,
            i_rad: i,
            arg_perigee_rad: pa,
            raan_rad: raan,
            mean_anomaly_rad: elliptic_true_to_mean(v, e),
        }
    }
}

impl Equinoctial {
    /// Equinoctial elements of a Cartesian state, through its circular
    /// parameters.
    pub(crate) fn from_cartesian(
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        mu: f64,
    ) -> Result<Self, AnalyticPropagationError> {
        CircularElements::from_cartesian(position_km, velocity_km_s, mu)?.to_equinoctial()
    }
}

/// `sqrt(mu / a)`, without overflow or underflow of the quotient.
fn sqrt_ratio(mu: f64, a: f64) -> f64 {
    let q = mu / a;
    if q.is_normal() {
        libm::sqrt(q)
    } else {
        libm::sqrt(mu) / libm::sqrt(a)
    }
}

/// A two-body state from finite elements, refused if it overflows.
fn check_state(pv: Pv) -> Result<Pv, AnalyticPropagationError> {
    if pv.0.iter().chain(pv.1.iter()).all(|x| x.is_finite()) {
        Ok(pv)
    } else {
        Err(AnalyticPropagationError::InvalidParameter {
            field: "elements",
            reason: "the Cartesian state overflows",
        })
    }
}

fn check_not_retrograde_equatorial(i: f64) -> Result<(), AnalyticPropagationError> {
    if (i - PI).abs() < RETROGRADE_EQUATORIAL_LIMIT {
        Err(AnalyticPropagationError::RetrogradeEquatorial { inclination_rad: i })
    } else {
        Ok(())
    }
}

/// Rotate an orbit-plane position `(x, y)` and velocity `(xd, yd)`, with `x`
/// along the direction at angle `w` from the ascending node, into the
/// inertial frame.
fn orbit_plane_to_inertial(x: f64, y: f64, xd: f64, yd: f64, i: f64, raan: f64, w: f64) -> Pv {
    let (sin_raan, cos_raan) = (libm::sin(raan), libm::cos(raan));
    let (sin_i, cos_i) = (libm::sin(i), libm::cos(i));
    let (sin_w, cos_w) = (libm::sin(w), libm::cos(w));
    // Unit vectors P (towards w) and Q (90 degrees ahead) in the inertial frame.
    let p = [
        cos_raan * cos_w - cos_i * sin_raan * sin_w,
        sin_raan * cos_w + cos_i * cos_raan * sin_w,
        sin_i * sin_w,
    ];
    let q = [
        -cos_raan * sin_w - cos_i * sin_raan * cos_w,
        -sin_raan * sin_w + cos_i * cos_raan * cos_w,
        sin_i * cos_w,
    ];
    (
        [
            x * p[0] + y * q[0],
            x * p[1] + y * q[1],
            x * p[2] + y * q[2],
        ],
        [
            xd * p[0] + yd * q[0],
            xd * p[1] + yd * q[1],
            xd * p[2] + yd * q[2],
        ],
    )
}

// ---- Argument of latitude (circular parameters) ----

pub(crate) fn circular_eccentric_to_true(ex: f64, ey: f64, alpha_e: f64) -> f64 {
    let epsilon = libm::sqrt(1.0 - ex * ex - ey * ey);
    let (s, c) = (libm::sin(alpha_e), libm::cos(alpha_e));
    let num = ex * s - ey * c;
    let den = epsilon + 1.0 - ex * c - ey * s;
    alpha_e + 2.0 * libm::atan(num / den)
}

pub(crate) fn circular_true_to_eccentric(ex: f64, ey: f64, alpha_v: f64) -> f64 {
    let epsilon = libm::sqrt(1.0 - ex * ex - ey * ey);
    let (s, c) = (libm::sin(alpha_v), libm::cos(alpha_v));
    let num = ey * c - ex * s;
    let den = epsilon + 1.0 + ex * c + ey * s;
    alpha_v + 2.0 * libm::atan(num / den)
}

pub(crate) fn circular_eccentric_to_mean(ex: f64, ey: f64, alpha_e: f64) -> f64 {
    alpha_e + (ey * libm::cos(alpha_e) - ex * libm::sin(alpha_e))
}

pub(crate) fn circular_true_to_mean(ex: f64, ey: f64, alpha_v: f64) -> f64 {
    circular_eccentric_to_mean(ex, ey, circular_true_to_eccentric(ex, ey, alpha_v))
}

/// Solve `alpha_m = alpha_e - ex sin(alpha_e) + ey cos(alpha_e)` for
/// `alpha_e` with Halley's iteration, the scheme of Orekit's
/// `CircularLatitudeArgumentUtility`: stop once a correction is at most
/// 1e-11 rad, which with cubic convergence leaves an error far below the
/// rounding of the result.
///
/// As for Kepler's equation, `alpha_m` is reduced to `[-pi, pi)` for the
/// solve and the removed multiple of `2 pi` added back, so the result lies
/// near `alpha_m`. Callers that need its sine and cosine pass a reduced
/// `alpha_m`: far from zero the sum rounds away the correction.
pub(crate) fn circular_mean_to_eccentric(
    ex: f64,
    ey: f64,
    alpha_m: f64,
) -> Result<f64, AnalyticPropagationError> {
    let unreduced = alpha_m;
    let alpha_m = normalize_angle(alpha_m, 0.0);
    let mut alpha_e = alpha_m;
    let mut alpha_e_minus_alpha_m = 0.0;
    for _ in 0..50 {
        let (s, c) = (libm::sin(alpha_e), libm::cos(alpha_e));
        let f2 = ex * s - ey * c;
        let f1 = 1.0 - ex * c - ey * s;
        let f0 = alpha_e_minus_alpha_m - f2;
        let f12 = 2.0 * f1;
        let shift = f0 * f12 / (f1 * f12 - f0 * f2);
        alpha_e_minus_alpha_m -= shift;
        alpha_e = alpha_m + alpha_e_minus_alpha_m;
        if shift.abs() <= 1.0e-11 {
            return Ok(alpha_e + (unreduced - alpha_m));
        }
    }
    Err(AnalyticPropagationError::KeplerNotConverged {
        field: "eccentric argument of latitude",
    })
}

pub(crate) fn circular_mean_to_true(
    ex: f64,
    ey: f64,
    alpha_m: f64,
) -> Result<f64, AnalyticPropagationError> {
    Ok(circular_eccentric_to_true(
        ex,
        ey,
        circular_mean_to_eccentric(ex, ey, alpha_m)?,
    ))
}

// ---- Elliptic anomalies ----

/// Solve Kepler's equation `M = E - e sin(E)` for `0 <= e < 1`.
///
/// `M` is reduced to `[-pi, pi)`, solved by Halley's iteration from Danby's
/// starter `E0 = M + 0.85 e sign(M)`, and the removed multiple of `2 pi` is
/// added back, so `E` lies near `M`. Callers that need the sine and cosine of
/// `E` pass a reduced `M`: far from zero the sum rounds away the solution.
pub(crate) fn elliptic_mean_to_eccentric(m: f64, e: f64) -> Result<f64, AnalyticPropagationError> {
    let reduced = normalize_angle(m, 0.0);
    if e == 0.0 {
        return Ok(m);
    }
    let mut big_e = if reduced >= 0.0 {
        reduced + 0.85 * e
    } else {
        reduced - 0.85 * e
    };
    for _ in 0..64 {
        let (s, c) = (libm::sin(big_e), libm::cos(big_e));
        let f = big_e - e * s - reduced;
        let f1 = 1.0 - e * c;
        let f2 = e * s;
        let step = f / (f1 - 0.5 * f * f2 / f1);
        big_e -= step;
        if step.abs() <= 1.0e-15 {
            return Ok(big_e + (m - reduced));
        }
    }
    Err(AnalyticPropagationError::KeplerNotConverged {
        field: "eccentric anomaly",
    })
}

pub(crate) fn elliptic_eccentric_to_true(big_e: f64, e: f64) -> f64 {
    let beta = e / (1.0 + libm::sqrt((1.0 - e) * (1.0 + e)));
    let (s, c) = (libm::sin(big_e), libm::cos(big_e));
    big_e + 2.0 * libm::atan(beta * s / (1.0 - beta * c))
}

pub(crate) fn elliptic_true_to_eccentric(v: f64, e: f64) -> f64 {
    let beta = e / (1.0 + libm::sqrt((1.0 - e) * (1.0 + e)));
    let (s, c) = (libm::sin(v), libm::cos(v));
    v - 2.0 * libm::atan(beta * s / (1.0 + beta * c))
}

pub(crate) fn elliptic_mean_to_true(m: f64, e: f64) -> Result<f64, AnalyticPropagationError> {
    Ok(elliptic_eccentric_to_true(
        elliptic_mean_to_eccentric(m, e)?,
        e,
    ))
}

pub(crate) fn elliptic_true_to_mean(v: f64, e: f64) -> f64 {
    let big_e = elliptic_true_to_eccentric(v, e);
    big_e - e * libm::sin(big_e)
}
