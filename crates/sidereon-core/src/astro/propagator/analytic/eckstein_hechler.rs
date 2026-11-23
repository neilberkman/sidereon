//! Eckstein-Hechler analytic propagator, J2 through J6, for near-circular
//! orbits.
//!
//! M. C. Eckstein and F. Hechler, "A reliable derivation of the perturbations
//! due to any zonal and tesseral harmonics of the geopotential for
//! nearly-circular satellite orbits", ESRO Scientific Report SR-13 (1970).
//! The mean circular parameters drift secularly (node and argument of latitude
//! at constant rates, the eccentricity vector rotating about its frozen value
//! `(0, eps2)`), and first-order short-period terms in harmonics of the mean
//! argument of latitude up to `6 alpha` are added to give the osculating
//! parameters. The coefficient arrangement follows Orekit's
//! `EcksteinHechlerPropagator`, the oracle of the tests.

use super::dual::Dual;
use super::elements::{CircularElements, Equinoctial};
use super::fixed_point::{osculating_to_mean, MeanTheory};
use super::{
    finite, finite_result, normalize_angle, AnalyticPropagationError, AnalyticTheory,
    MeanConversionOptions, MeanElements, ZonalHarmonics,
};
use crate::astro::state::CartesianState;
use core::f64::consts::PI;

const THEORY: AnalyticTheory = AnalyticTheory::EcksteinHechler;

/// Largest mean eccentricity the theory accepts.
const MAX_ECCENTRICITY: f64 = 0.1;
/// `sin^2 i` below this is treated as equatorial.
const EQUATORIAL_SIN2_LIMIT: f64 = 1.0e-10;
/// Half-width of the band of `sin^2 i` around 4/5 treated as critical.
const CRITICAL_SIN2_HALF_WIDTH: f64 = 1.0e-3;

/// Mean elements and the coefficients of the secular rates and periodic terms.
#[derive(Debug, Clone, Copy)]
struct Model {
    mean: CircularElements,
    xnot_dot: f64,
    rdpom: f64,
    rdpomp: f64,
    eps1: f64,
    eps2: f64,
    omm_d: f64,
    a_m_d: f64,
    kh: f64,
    kl: f64,
    ax1: f64,
    ay1: f64,
    as1: f64,
    ac2: f64,
    axy3: f64,
    as3: f64,
    ac4: f64,
    as5: f64,
    ac6: f64,
    ex1: f64,
    exx2: f64,
    exy2: f64,
    ex3: f64,
    ex4: f64,
    ey1: f64,
    eyx2: f64,
    eyy2: f64,
    ey3: f64,
    ey4: f64,
    rx1: f64,
    ry1: f64,
    r2: f64,
    r3: f64,
    rl: f64,
    iy1: f64,
    ix1: f64,
    i2: f64,
    i3: f64,
    ih: f64,
    lx1: f64,
    ly1: f64,
    l2: f64,
    l3: f64,
    ll: f64,
}

impl Model {
    fn new(mean: &CircularElements, g: &ZonalHarmonics) -> Result<Self, AnalyticPropagationError> {
        mean.validate("mean elements")?;
        let mean = &mean.canonical();
        // The theory expands in powers of re / a.
        if mean.a_km < g.re_km {
            return Err(AnalyticPropagationError::InsideReferenceSphere {
                semi_major_axis_km: mean.a_km,
                reference_radius_km: g.re_km,
            });
        }

        let mut q = g.re_km / mean.a_km;
        let mut ql = q * q;
        let g2 = g.c20 * ql;
        ql *= q;
        let g3 = g.c30 * ql;
        ql *= q;
        let g4 = g.c40 * ql;
        ql *= q;
        let g5 = g.c50 * ql;
        ql *= q;
        let g6 = g.c60 * ql;

        let cos_i1 = libm::cos(mean.i_rad);
        let sin_i1 = libm::sin(mean.i_rad);
        let sin_i2 = sin_i1 * sin_i1;
        let sin_i4 = sin_i2 * sin_i2;
        let sin_i6 = sin_i2 * sin_i4;

        if sin_i2 < EQUATORIAL_SIN2_LIMIT {
            return Err(AnalyticPropagationError::NearlyEquatorial {
                inclination_rad: mean.i_rad,
            });
        }
        if (sin_i2 - 4.0 / 5.0).abs() < CRITICAL_SIN2_HALF_WIDTH {
            return Err(AnalyticPropagationError::NearCriticalInclination {
                inclination_rad: mean.i_rad,
            });
        }
        let e = mean.eccentricity();
        if e > MAX_ECCENTRICITY {
            return Err(AnalyticPropagationError::EccentricityTooLarge {
                eccentricity: e,
                limit: MAX_ECCENTRICITY,
            });
        }

        let xnot_dot = libm::sqrt(g.mu_km3_s2 / mean.a_km) / mean.a_km;

        let rdpom = -0.75 * g2 * (4.0 - 5.0 * sin_i2);
        let rdpomp = 7.5 * g4 * (1.0 - 31.0 / 8.0 * sin_i2 + 49.0 / 16.0 * sin_i4)
            - 13.125 * g6 * (1.0 - 8.0 * sin_i2 + 129.0 / 8.0 * sin_i4 - 297.0 / 32.0 * sin_i6);

        q = 3.0 / (32.0 * rdpom);
        let eps1 = q * g4 * sin_i2 * (30.0 - 35.0 * sin_i2)
            - 175.0 * q * g6 * sin_i2 * (1.0 - 3.0 * sin_i2 + 2.0625 * sin_i4);
        q = 3.0 * sin_i1 / (8.0 * rdpom);
        let eps2 = q * g3 * (4.0 - 5.0 * sin_i2) - q * g5 * (10.0 - 35.0 * sin_i2 + 26.25 * sin_i4);

        let omm_d = cos_i1
            * (1.50 * g2 - 2.25 * g2 * g2 * (2.5 - 19.0 / 6.0 * sin_i2)
                + 0.9375 * g4 * (7.0 * sin_i2 - 4.0)
                + 3.28125 * g6 * (2.0 - 9.0 * sin_i2 + 8.25 * sin_i4));

        let rdl = 1.0 - 1.50 * g2 * (3.0 - 4.0 * sin_i2);
        let a_m_d = rdl
            + 2.25 * g2 * g2 * (9.0 - 263.0 / 12.0 * sin_i2 + 341.0 / 24.0 * sin_i4)
            + 15.0 / 16.0 * g4 * (8.0 - 31.0 * sin_i2 + 24.5 * sin_i4)
            + 105.0 / 32.0 * g6 * (-10.0 / 3.0 + 25.0 * sin_i2 - 48.75 * sin_i4 + 27.5 * sin_i6);

        let qq = -1.5 * g2 / rdl;
        let q_a = 0.75 * g2 * g2 * sin_i2;
        let q_b = 0.25 * g4 * sin_i2;
        let q_c = 105.0 / 16.0 * g6 * sin_i2;
        let q_d = -0.75 * g3 * sin_i1;
        let q_e = 3.75 * g5 * sin_i1;
        let kh = 0.375 / rdpom;
        let kl = kh / sin_i1;

        let ax1 = qq * (2.0 - 3.5 * sin_i2);
        let ay1 = qq * (2.0 - 2.5 * sin_i2);
        let as1 = q_d * (4.0 - 5.0 * sin_i2) + q_e * (2.625 * sin_i4 - 3.5 * sin_i2 + 1.0);
        let ac2 = qq * sin_i2
            + q_a * 7.0 * (2.0 - 3.0 * sin_i2)
            + q_b * (15.0 - 17.5 * sin_i2)
            + q_c * (3.0 * sin_i2 - 1.0 - 33.0 / 16.0 * sin_i4);
        let axy3 = qq * 3.5 * sin_i2;
        let as3 = q_d * 5.0 / 3.0 * sin_i2 + q_e * 7.0 / 6.0 * sin_i2 * (1.0 - 1.125 * sin_i2);
        let ac4 = q_a * sin_i2 + q_b * 4.375 * sin_i2 + q_c * 0.75 * (1.1 * sin_i4 - sin_i2);
        let as5 = q_e * 21.0 / 80.0 * sin_i4;
        let ac6 = q_c * -11.0 / 80.0 * sin_i4;

        let ex1 = qq * (1.0 - 1.25 * sin_i2);
        let exx2 = qq * 0.5 * (3.0 - 5.0 * sin_i2);
        let exy2 = qq * (2.0 - 1.5 * sin_i2);
        let ex3 = qq * 7.0 / 12.0 * sin_i2;
        let ex4 = qq * 17.0 / 8.0 * sin_i2;

        let ey1 = qq * (1.0 - 1.75 * sin_i2);
        let eyx2 = qq * (1.0 - 3.0 * sin_i2);
        let eyy2 = qq * (2.0 * sin_i2 - 1.5);
        let ey3 = qq * 7.0 / 12.0 * sin_i2;
        let ey4 = qq * 17.0 / 8.0 * sin_i2;

        q = -qq * cos_i1;
        let rx1 = 3.5 * q;
        let ry1 = -2.5 * q;
        let r2 = -0.5 * q;
        let r3 = 7.0 / 6.0 * q;
        let rl = g3 * cos_i1 * (4.0 - 15.0 * sin_i2)
            - 2.5 * g5 * cos_i1 * (4.0 - 42.0 * sin_i2 + 52.5 * sin_i4);

        q = 0.5 * qq * sin_i1 * cos_i1;
        let iy1 = q;
        let ix1 = -q;
        let i2 = q;
        let i3 = q * 7.0 / 3.0;
        let ih = -g3 * cos_i1 * (4.0 - 5.0 * sin_i2)
            + 2.5 * g5 * cos_i1 * (4.0 - 14.0 * sin_i2 + 10.5 * sin_i4);

        let lx1 = qq * (7.0 - 77.0 / 8.0 * sin_i2);
        let ly1 = qq * (55.0 / 8.0 * sin_i2 - 7.50);
        let l2 = qq * (1.25 * sin_i2 - 0.5);
        let l3 = qq * (77.0 / 24.0 * sin_i2 - 7.0 / 6.0);
        let ll = g3 * (53.0 * sin_i2 - 4.0 - 57.5 * sin_i4)
            + 2.5 * g5 * (4.0 - 96.0 * sin_i2 + 269.5 * sin_i4 - 183.75 * sin_i6);

        Ok(Self {
            mean: *mean,
            xnot_dot,
            rdpom,
            rdpomp,
            eps1,
            eps2,
            omm_d,
            a_m_d,
            kh,
            kl,
            ax1,
            ay1,
            as1,
            ac2,
            axy3,
            as3,
            ac4,
            as5,
            ac6,
            ex1,
            exx2,
            exy2,
            ex3,
            ex4,
            ey1,
            eyx2,
            eyy2,
            ey3,
            ey4,
            rx1,
            ry1,
            r2,
            r3,
            rl,
            iy1,
            ix1,
            i2,
            i3,
            ih,
            lx1,
            ly1,
            l2,
            l3,
            ll,
        })
    }

    /// Osculating circular parameters `dt` seconds after the mean elements'
    /// epoch, with their time derivatives: `[a, ex, ey, i, raan, alpha_m]`.
    fn parameters(&self, dt: f64) -> [Dual; 6] {
        let m = &self.mean;
        let dt = Dual::new(dt, 1.0);
        let xnot = dt * self.xnot_dot;

        // Secular rotation of the eccentricity vector about (0, eps2).
        let x = xnot * (self.rdpom + self.rdpomp);
        let cx = x.cos();
        let sx = x.sin();
        let exm = cx * m.ex + sx * (self.eps2 - (1.0 - self.eps1) * m.ey);
        let eym = (sx * ((1.0 + self.eps1) * m.ex) + cx * (m.ey - self.eps2)).offset(self.eps2);

        // Secular node and argument of latitude; no secular inclination change.
        let omm = Dual::new(
            normalize_angle(normalize_angle(m.raan_rad, 0.0) + self.omm_d * xnot.v, PI),
            self.omm_d * self.xnot_dot,
        );
        let xlm = Dual::new(
            normalize_angle(
                normalize_angle(m.mean_latitude_argument_rad, 0.0) + self.a_m_d * xnot.v,
                PI,
            ),
            self.a_m_d * self.xnot_dot,
        );

        // Harmonics of the mean argument of latitude.
        let cl1 = xlm.cos();
        let sl1 = xlm.sin();
        let cl2 = cl1 * cl1 - sl1 * sl1;
        let sl2 = cl1 * sl1 + sl1 * cl1;
        let cl3 = cl2 * cl1 - sl2 * sl1;
        let sl3 = cl2 * sl1 + sl2 * cl1;
        let cl4 = cl3 * cl1 - sl3 * sl1;
        let sl4 = cl3 * sl1 + sl3 * cl1;
        let cl5 = cl4 * cl1 - sl4 * sl1;
        let sl5 = cl4 * sl1 + sl4 * cl1;
        let cl6 = cl5 * cl1 - sl5 * sl1;

        let qh = (eym.offset(-self.eps2)) * self.kh;
        let ql = exm * self.kl;

        let exm_cl1 = exm * cl1;
        let exm_sl1 = exm * sl1;
        let eym_cl1 = eym * cl1;
        let eym_sl1 = eym * sl1;
        let exm_cl2 = exm * cl2;
        let exm_sl2 = exm * sl2;
        let eym_cl2 = eym * cl2;
        let eym_sl2 = eym * sl2;
        let exm_cl3 = exm * cl3;
        let exm_sl3 = exm * sl3;
        let eym_cl3 = eym * cl3;
        let eym_sl3 = eym * sl3;
        let exm_cl4 = exm * cl4;
        let exm_sl4 = exm * sl4;
        let eym_cl4 = eym * cl4;
        let eym_sl4 = eym * sl4;

        let rda = exm_cl1 * self.ax1
            + eym_sl1 * self.ay1
            + sl1 * self.as1
            + cl2 * self.ac2
            + (exm_cl3 + eym_sl3) * self.axy3
            + sl3 * self.as3
            + cl4 * self.ac4
            + sl5 * self.as5
            + cl6 * self.ac6;

        let rdex = cl1 * self.ex1
            + exm_cl2 * self.exx2
            + eym_sl2 * self.exy2
            + cl3 * self.ex3
            + (exm_cl4 + eym_sl4) * self.ex4;

        let rdey = sl1 * self.ey1
            + exm_sl2 * self.eyx2
            + eym_cl2 * self.eyy2
            + sl3 * self.ey3
            + (exm_sl4 - eym_cl4) * self.ey4;

        let rdom = exm_sl1 * self.rx1
            + eym_cl1 * self.ry1
            + sl2 * self.r2
            + (eym_cl3 - exm_sl3) * self.r3
            + ql * self.rl;

        let rdxi = eym_sl1 * self.iy1
            + exm_cl1 * self.ix1
            + cl2 * self.i2
            + (exm_cl3 + eym_sl3) * self.i3
            + qh * self.ih;

        let rdxl = exm_sl1 * self.lx1
            + eym_cl1 * self.ly1
            + sl2 * self.l2
            + (exm_sl3 - eym_cl3) * self.l3
            + ql * self.ll;

        [
            rda.offset(1.0) * m.a_km,
            rdex + exm,
            rdey + eym,
            rdxi.offset(m.i_rad),
            rdom + omm,
            rdxl + xlm,
        ]
    }

    /// Osculating circular elements `dt` seconds after the mean elements'
    /// epoch: `NonFiniteResult` if the theory overflows, `NotElliptic` if
    /// the periodic terms leave no elliptic orbit.
    fn osculating_elements(&self, dt: f64) -> Result<CircularElements, AnalyticPropagationError> {
        elements_of(&self.parameters(dt))
    }
}

/// The osculating elements of [`Model::parameters`], checked as in
/// [`Model::osculating_elements`], with the inclination in `[0, pi]`: near the
/// equator the J3 and J5 inclination terms, which do not vanish with `sin i`,
/// can carry the mean inclination plus its periodic terms out of that range.
fn elements_of(p: &[Dual; 6]) -> Result<CircularElements, AnalyticPropagationError> {
    let osc = CircularElements {
        a_km: p[0].v,
        ex: p[1].v,
        ey: p[2].v,
        i_rad: p[3].v,
        raan_rad: p[4].v,
        mean_latitude_argument_rad: p[5].v,
    };
    check_elements(&osc)?;
    Ok(osc.canonical())
}

/// Position from osculating circular parameters, and its time derivative.
fn to_cartesian(p: &[Dual; 6]) -> Result<([f64; 3], [f64; 3]), AnalyticPropagationError> {
    let [a, ex, ey, inc, raan, alpha_m] = *p;
    let cos_omega = raan.cos();
    let sin_omega = raan.sin();
    let cos_i = inc.cos();
    let sin_i = inc.sin();
    let alpha_e = mean_to_eccentric(alpha_m, ex, ey)?;
    let cos_ae = alpha_e.cos();
    let sin_ae = alpha_e.sin();
    let ex2 = ex * ex;
    let ey2 = ey * ey;
    let exy = ex * ey;
    let q = (Dual::constant(1.0) - ex2 - ey2).sqrt();
    let beta = q.offset(1.0).recip();
    let bx2 = beta * ex2;
    let by2 = beta * ey2;
    let bxy = beta * exy;
    let u = bxy * sin_ae - (ex + by2.offset(-1.0) * cos_ae);
    let v = bxy * cos_ae - (ey + bx2.offset(-1.0) * sin_ae);
    let x = a * u;
    let y = a * v;
    let px = x * cos_omega - y * (cos_i * sin_omega);
    let py = x * sin_omega + y * (cos_i * cos_omega);
    let pz = y * sin_i;
    Ok(([px.v, py.v, pz.v], [px.d, py.d, pz.d]))
}

/// Solve `alpha_m = alpha_e - ex sin(alpha_e) + ey cos(alpha_e)` carrying the
/// time derivatives, by the Halley iteration Orekit applies to its derivative
/// type.
///
/// Only the sine and cosine of the result are used, so `alpha_m` is first
/// reduced into `[0, 2 pi)`, the range of the secular argument of latitude,
/// where it is returned unchanged. The periodic terms added to it are of order
/// J_n, but a field with very large coefficients can put it far from zero,
/// where adding the eccentric correction would round the correction away.
fn mean_to_eccentric(alpha_m: Dual, ex: Dual, ey: Dual) -> Result<Dual, AnalyticPropagationError> {
    let alpha_m = Dual::new(normalize_angle(alpha_m.v, PI), alpha_m.d);
    let mut alpha_e = alpha_m;
    let mut alpha_e_minus_alpha_m = Dual::constant(0.0);
    let mut cos_alpha_e = alpha_e.cos();
    let mut sin_alpha_e = alpha_e.sin();
    for _ in 0..50 {
        let f2 = ex * sin_alpha_e - ey * cos_alpha_e;
        let f1 = Dual::constant(1.0) - ex * cos_alpha_e - ey * sin_alpha_e;
        let f0 = alpha_e_minus_alpha_m - f2;
        let f12 = f1.scale(2.0);
        let shift = f0 * f12 / (f1 * f12 - f0 * f2);
        alpha_e_minus_alpha_m = alpha_e_minus_alpha_m - shift;
        alpha_e = alpha_m + alpha_e_minus_alpha_m;
        cos_alpha_e = alpha_e.cos();
        sin_alpha_e = alpha_e.sin();
        if shift.v.abs() <= 1.0e-12 {
            return Ok(alpha_e);
        }
    }
    Err(AnalyticPropagationError::KeplerNotConverged {
        field: "eccentric argument of latitude",
    })
}

struct Theory<'a>(&'a ZonalHarmonics);

impl MeanTheory for Theory<'_> {
    fn theory(&self) -> AnalyticTheory {
        THEORY
    }

    fn reference_radius_km(&self) -> f64 {
        self.0.re_km
    }

    fn mean_to_osculating(
        &self,
        mean: &Equinoctial,
    ) -> Result<Equinoctial, AnalyticPropagationError> {
        let model = Model::new(&CircularElements::from_equinoctial(mean), self.0)?;
        model.osculating_elements(0.0)?.to_equinoctial()
    }
}

/// Eckstein-Hechler mean circular elements of an osculating Cartesian state.
///
/// The state's epoch is not used. `options` sets the fixed-point tolerance and
/// iteration limit; [`MeanConversionOptions::ECKSTEIN_HECHLER_DEFAULT`] is
/// Orekit's default.
///
/// # Errors
///
/// [`AnalyticPropagationError::NonFinite`] or `InvalidParameter` for invalid
/// inputs, `NotElliptic` for an open orbit, `InsideReferenceSphere` when the
/// osculating semi-major axis is below the reference radius, the theory's
/// domain errors (`EccentricityTooLarge`, `NearlyEquatorial`,
/// `NearCriticalInclination`) for any iterate outside it, and
/// `MeanElementsNotConverged` when the tolerance is not met.
pub fn eckstein_hechler_mean_elements(
    osculating: &CartesianState,
    gravity: &ZonalHarmonics,
    options: MeanConversionOptions,
) -> Result<MeanElements<CircularElements>, AnalyticPropagationError> {
    gravity.validate()?;
    options.validate()?;
    let osc = Equinoctial::from_cartesian(
        osculating.position_array(),
        osculating.velocity_array(),
        gravity.mu_km3_s2,
    )?;
    let (mean, iterations) = osculating_to_mean(&Theory(gravity), &osc, &options)?;
    let elements = CircularElements::from_equinoctial(&mean);
    // The converged iterate passed the model's domain checks.
    Model::new(&elements, gravity)?;
    Ok(MeanElements {
        elements,
        iterations,
    })
}

/// Eckstein-Hechler osculating circular elements of mean elements, at the mean
/// elements' own epoch. The mean inclination may be any finite angle and the
/// returned one is in `[0, pi]` (see the [module documentation](super)).
///
/// # Errors
///
/// `NonFinite`, `InvalidParameter` or `NotElliptic` for invalid inputs,
/// `InsideReferenceSphere` for a mean semi-major axis below the reference
/// radius, and the theory's domain errors for mean elements outside it.
pub fn eckstein_hechler_osculating_elements(
    mean: &CircularElements,
    gravity: &ZonalHarmonics,
) -> Result<CircularElements, AnalyticPropagationError> {
    gravity.validate()?;
    Model::new(mean, gravity)?.osculating_elements(0.0)
}

fn check_elements(e: &CircularElements) -> Result<(), AnalyticPropagationError> {
    for (value, field) in [
        (e.a_km, "a_km"),
        (e.ex, "ex"),
        (e.ey, "ey"),
        (e.i_rad, "i_rad"),
        (e.raan_rad, "raan_rad"),
        (e.mean_latitude_argument_rad, "mean_latitude_argument_rad"),
    ] {
        finite_result(value, THEORY, field)?;
    }
    e.validate("osculating elements")
}

/// Eckstein-Hechler analytic propagator.
///
/// Holds mean circular elements at an epoch and evaluates the osculating state
/// at any other epoch in closed form. See the [module documentation](super)
/// for the theory, its domain and the oracle.
///
/// ```
/// use sidereon_core::astro::propagator::analytic::{
///     EcksteinHechlerPropagator, MeanConversionOptions, ZonalHarmonics,
/// };
/// use sidereon_core::astro::state::CartesianState;
///
/// // A 700 km sun-synchronous orbit.
/// let initial = CartesianState::new(0.0, [7078.137, 0.0, 0.0], [0.0, -1.0791, 7.4258]);
/// let propagator = EcksteinHechlerPropagator::new(
///     &initial,
///     &ZonalHarmonics::earth(),
///     MeanConversionOptions::ECKSTEIN_HECHLER_DEFAULT,
/// )
/// .unwrap();
/// let one_day_later = propagator.propagate(86_400.0).unwrap();
/// assert_eq!(one_day_later.epoch_tdb_seconds, 86_400.0);
/// ```
#[derive(Debug, Clone)]
pub struct EcksteinHechlerPropagator {
    epoch_tdb_seconds: f64,
    gravity: ZonalHarmonics,
    model: Model,
}

impl EcksteinHechlerPropagator {
    /// Start from an osculating Cartesian state, converting it to mean elements
    /// with [`eckstein_hechler_mean_elements`].
    ///
    /// # Errors
    ///
    /// As [`eckstein_hechler_mean_elements`], and `NonFinite` for a non-finite
    /// epoch.
    pub fn new(
        initial: &CartesianState,
        gravity: &ZonalHarmonics,
        options: MeanConversionOptions,
    ) -> Result<Self, AnalyticPropagationError> {
        finite(initial.epoch_tdb_seconds, "epoch_tdb_seconds")?;
        let mean = eckstein_hechler_mean_elements(initial, gravity, options)?;
        Self::from_mean_elements(initial.epoch_tdb_seconds, mean.elements, gravity)
    }

    /// Start from mean circular elements at `epoch_tdb_seconds`.
    ///
    /// # Errors
    ///
    /// `NonFinite`, `InvalidParameter` or `NotElliptic` for invalid inputs,
    /// `InsideReferenceSphere` for a mean semi-major axis below the reference
    /// radius, and the theory's domain errors for mean elements outside it.
    pub fn from_mean_elements(
        epoch_tdb_seconds: f64,
        mean: CircularElements,
        gravity: &ZonalHarmonics,
    ) -> Result<Self, AnalyticPropagationError> {
        finite(epoch_tdb_seconds, "epoch_tdb_seconds")?;
        gravity.validate()?;
        Ok(Self {
            epoch_tdb_seconds,
            gravity: *gravity,
            model: Model::new(&mean, gravity)?,
        })
    }

    /// Epoch of the mean elements, TDB seconds from J2000.
    pub fn epoch_tdb_seconds(&self) -> f64 {
        self.epoch_tdb_seconds
    }

    /// Mean circular elements at [`Self::epoch_tdb_seconds`], with the
    /// inclination in `[0, pi]`: elements given outside that range are
    /// returned as the same orbit inside it (see the
    /// [module documentation](super)).
    pub fn mean_elements(&self) -> CircularElements {
        self.model.mean
    }

    /// Gravity field of the theory.
    pub fn gravity(&self) -> &ZonalHarmonics {
        &self.gravity
    }

    /// Osculating circular elements at `epoch_tdb_seconds`.
    ///
    /// # Errors
    ///
    /// `NonFinite` for a non-finite epoch, `NonFiniteResult` if the theory
    /// overflows (only for spans far beyond its accuracy), `NotElliptic` if
    /// the osculating elements are not elliptic.
    pub fn osculating_elements(
        &self,
        epoch_tdb_seconds: f64,
    ) -> Result<CircularElements, AnalyticPropagationError> {
        let dt = finite(epoch_tdb_seconds, "epoch_tdb_seconds")? - self.epoch_tdb_seconds;
        self.model.osculating_elements(dt)
    }

    /// Osculating Cartesian state at `epoch_tdb_seconds`: position km and its
    /// time derivative, km/s.
    ///
    /// # Errors
    ///
    /// As [`Self::osculating_elements`], and `KeplerNotConverged` if the
    /// theory fails numerically.
    pub fn propagate(
        &self,
        epoch_tdb_seconds: f64,
    ) -> Result<CartesianState, AnalyticPropagationError> {
        let dt = finite(epoch_tdb_seconds, "epoch_tdb_seconds")? - self.epoch_tdb_seconds;
        let parameters = self.model.parameters(dt);
        elements_of(&parameters)?;
        let (position, velocity) = to_cartesian(&parameters)?;
        for value in position {
            finite_result(value, THEORY, "position_km")?;
        }
        for value in velocity {
            finite_result(value, THEORY, "velocity_km_s")?;
        }
        Ok(CartesianState::new(epoch_tdb_seconds, position, velocity))
    }

    /// [`Self::propagate`] at each epoch, in order.
    ///
    /// # Errors
    ///
    /// The first error of [`Self::propagate`].
    pub fn ephemeris(
        &self,
        epochs_tdb_seconds: &[f64],
    ) -> Result<Vec<CartesianState>, AnalyticPropagationError> {
        epochs_tdb_seconds
            .iter()
            .map(|&epoch| self.propagate(epoch))
            .collect()
    }
}
