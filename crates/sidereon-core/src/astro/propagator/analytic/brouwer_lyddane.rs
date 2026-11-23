//! Brouwer-Lyddane analytic propagator, J2 through J5, with an optional
//! secular along-track term.
//!
//! D. Brouwer, "Solution of the problem of artificial satellite theory
//! without drag", Astronomical Journal 64 (1959) 378-397, gives the secular
//! rates of the mean argument of perigee, node and mean anomaly to second
//! order in J2 (with J4), the long-period terms (J2 to J5) and the
//! short-period terms (J2). R. H. Lyddane, "Small eccentricities or
//! inclinations in the Brouwer theory of the artificial satellite",
//! Astronomical Journal 68 (1963) 555-558, recombines the corrections in the
//! variables `e cos l`, `e sin l`, `sin(i/2) cos h`, `sin(i/2) sin h` and
//! `l + g + h`, which removes the divisions by `e` and `sin i`. The
//! `1 / (1 - 5 cos^2 i)` factor of the long-period terms, singular at the
//! critical inclination, is replaced by the smooth function of W. E. Phipps
//! Jr., "Parallelization of the Navy Space Surveillance Center (NAVSPASUR)
//! Satellite Model", Naval Postgraduate School thesis (1992), Eq. 2.47-2.48,
//! and the optional drag term `M2` adds the secular mean-anomaly, semi-major
//! axis and eccentricity rates of Phipps' Eq. 2.38, 2.41 and 2.45.
//!
//! The arrangement of the terms follows Orekit's `BrouwerLyddanePropagator`,
//! the oracle of the tests, except where Orekit 13.1 departs from Brouwer:
//! the short-period eccentricity term, the `sin f` part of the short-period
//! `e dl` term and the long-period inclination are Brouwer's as published (the
//! same terms NASA GMAT's Brouwer-Lyddane conversions evaluate). The tests
//! check them against the published form, against Brouwer's relation between
//! the long-period inclination and eccentricity terms, and against numerical
//! integration.

use super::elements::{elliptic_mean_to_true, Equinoctial, KeplerianElements};
use super::fixed_point::{osculating_to_mean, MeanTheory};
use super::{
    finite, finite_result, normalize_angle, AnalyticPropagationError, AnalyticTheory,
    MeanConversionOptions, MeanElements, ZonalHarmonics,
};
use crate::astro::state::CartesianState;

const THEORY: AnalyticTheory = AnalyticTheory::BrouwerLyddane;

/// Upper clamp of the drag-decayed mean eccentricity.
const MAX_ECC: f64 = 0.999_999;
/// Phipps' beta in the critical-inclination factor, `100 * 2^-11`.
const BETA: f64 = 100.0 / 2048.0;

/// Which arrangement of the periodic terms to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShortPeriodTerms {
    /// Brouwer (1959) as published.
    Brouwer,
    /// Orekit 13.1's arrangement, which differs from Brouwer in three places:
    /// the short-period eccentricity term (`w20 + e cos(2g + 2f)` for
    /// `(w20 + e) cos(2g + 2f)`, and an extra factor `e` on
    /// `3 cos(2g + f) + cos(2g + 3f)`), the `sin f` part of the short-period
    /// `e dl` (`2 sin^2 i` for `2 (3 cos^2 i - 1)`), and the long-period
    /// inclination (`-e q1 / sin i` for `-e^2 cos i q1 / sin i` on `cos 2g`,
    /// and the sign of the `sin 3g` term). Only the test-only replay of the
    /// Orekit fixture uses it.
    #[cfg_attr(not(feature = "test-replays"), allow(dead_code))]
    OrekitReplay,
}

/// Short-period corrections: `da`, `de`, `di`, `e dl`, `sin(i) dh` and
/// `d(l + g + h)`.
#[derive(Debug, Clone, Copy)]
struct ShortPeriod {
    d2a: f64,
    d2e: f64,
    d2i: f64,
    eppd2l: f64,
    s_ippd2h: f64,
    d2z: f64,
}

#[derive(Debug, Clone, Copy)]
struct Model {
    terms: ShortPeriodTerms,
    mean: KeplerianElements,
    n0: f64,
    n: f64,
    n2: f64,
    n3: f64,
    t8: f64,
    dsl: f64,
    dsg: f64,
    dsh: f64,
    a_rate: f64,
    e_rate: f64,
    yp2: f64,
    ci: f64,
    si: f64,
    one_m_ci2: f64,
    ci2_x3_m1: f64,
    vle1: f64,
    vle2: f64,
    vle3: f64,
    vli1: f64,
    vli2: f64,
    vli3: f64,
    vll2: f64,
    vlh1i: f64,
    vlh2i: f64,
    vlh3i: f64,
    vls1: f64,
    vls2: f64,
    vls3: f64,
}

/// Phipps' replacement for `1 / (1 - 5 cos^2 i)`, Eq. 2.47-2.48: equal to it
/// away from the critical inclination and bounded through it. With
/// `x = 1 - 5 cos^2 i` and `y = beta x^2`, the series is
/// `(1 - exp(-y)) / y` and the product telescopes to
/// `(1 - exp(-2048 y)) / (1 - exp(-y))`, so the function is
/// `(1 - exp(-100 x^2)) / x`, evaluated here in that closed form.
fn t2(cos_i: f64) -> f64 {
    let x = 1.0 - 5.0 * cos_i * cos_i;
    if x == 0.0 {
        0.0
    } else {
        -libm::expm1(-100.0 * x * x) / x
    }
}

/// [`t2`] as Phipps writes it and Orekit evaluates it: the series of
/// `(1 - exp(-y)) / y` cut after 13 terms, which is off by up to 1e-12
/// relative near the poles. Only the replay of Orekit's arrangement uses it.
fn t2_series(cos_i: f64) -> f64 {
    let x = 1.0 - 5.0 * cos_i * cos_i;
    let x2 = x * x;
    let mut sum = 0.0;
    let mut factorial = 1.0;
    for i in 0..=12 {
        factorial *= (i + 1) as f64;
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * libm::pow(BETA, i as f64) * libm::pow(x2, i as f64) / factorial;
    }
    let mut product = 1.0;
    for i in 0..=10 {
        product *= 1.0 + libm::exp(libm::ldexp(-1.0, i) * BETA * x2);
    }
    BETA * x * sum * product
}

impl Model {
    fn new(
        mean: &KeplerianElements,
        g: &ZonalHarmonics,
        terms: ShortPeriodTerms,
    ) -> Result<Self, AnalyticPropagationError> {
        mean.validate("mean elements")?;
        // The theory expands in powers of re / a.
        if mean.a_km < g.re_km {
            return Err(AnalyticPropagationError::InsideReferenceSphere {
                semi_major_axis_km: mean.a_km,
                reference_radius_km: g.re_km,
            });
        }
        let epp = mean.e;
        let epp2 = epp * epp;
        let n2 = 1.0 - epp2;
        let n = libm::sqrt(n2);
        let n3 = n2 * n;
        let t8 = n + 1.0 / (1.0 + n);

        let app = mean.a_km;
        let n0 = libm::sqrt(g.mu_km3_s2 / app) / app;
        let q = g.re_km / app;
        let mut ql = q * q;
        let mut nl = n2 * n2;
        let yp2 = -0.5 * g.c20 * ql / nl;
        let yp22 = yp2 * yp2;
        ql *= q;
        nl *= n2;
        let yp3 = g.c30 * ql / nl;
        ql *= q;
        nl *= n2;
        let yp4 = 0.375 * g.c40 * ql / nl;
        ql *= q;
        nl *= n2;
        let yp5 = g.c50 * ql / nl;

        let si = libm::sin(mean.i_rad);
        let ci = libm::cos(mean.i_rad);
        let ci2 = ci * ci;
        let one_m_ci2 = 1.0 - ci2;
        let ci2_x3_m1 = 3.0 * ci2 - 1.0;
        let ci2_x5_m1 = 5.0 * ci2 - 1.0;

        // Secular rates (Brouwer 1959, with Lyddane's arrangement).
        let dsl = 1.5
            * yp2
            * n
            * (ci2_x3_m1
                + 0.0625
                    * yp2
                    * (-15.0
                        + n * (16.0 + 25.0 * n)
                        + ci2
                            * (30.0 - n * (96.0 + 90.0 * n)
                                + ci2 * (105.0 + n * (144.0 + 25.0 * n)))))
            + 0.9375 * yp4 * n * epp2 * (3.0 - ci2 * (30.0 - 35.0 * ci2));
        let dsg = 1.5 * yp2 * ci2_x5_m1
            + 0.09375
                * yp22
                * (-35.0
                    + n * (24.0 + 25.0 * n)
                    + ci2
                        * (90.0 - n * (192.0 + 126.0 * n)
                            + ci2 * (385.0 + n * (360.0 + 45.0 * n))))
            + 0.3125
                * yp4
                * (21.0 - 9.0 * n2 + ci2 * (-270.0 + 126.0 * n2 + ci2 * (385.0 - 189.0 * n2)));
        let dsh = (-3.0 * yp2
            + 0.375 * yp22 * (-5.0 + n * (12.0 + 9.0 * n) - ci2 * (35.0 + n * (36.0 + 5.0 * n)))
            + 1.25 * yp4 * (5.0 - 3.0 * n2) * (3.0 - 7.0 * ci2))
            * ci;

        // Drag rates of the mean semi-major axis and eccentricity, Phipps
        // Eq. 2.41 and 2.45, per unit M2 and second.
        let coef = -4.0 / (3.0 * n0 * (1.0 + dsl));
        let a_rate = coef * app;
        let e_rate = coef * epp * n2;

        let t2 = match terms {
            ShortPeriodTerms::Brouwer => t2(ci),
            ShortPeriodTerms::OrekitReplay => t2_series(ci),
        };
        let fs12 = yp3 / yp2;
        let fs13 = 10.0 * yp4 / (3.0 * yp2);
        let fs14 = yp5 / yp2;

        let ci2_xt2 = ci2 * t2;
        // With t2 = (1 - E) / x, x = 1 - 5 cos^2 i and E = exp(-100 x^2),
        //   c_a = (sin^2 i (1 - 15 cos^2 i) + 40 cos^4 i E) / x,
        //   c_b = (sin^2 i (1 - 7 cos^2 i) + 8 cos^4 i E) / x.
        // E is below 4e-44 for x <= -1 (cos^2 i >= 0.4) and zero in double
        // precision for cos^2 i > 0.75, so both vanish as sin^2 i towards the
        // poles. Evaluated in this form there, c_a / sin i and c_b / sin i
        // (in the long-period inclination) are exact through i = 0 and pi,
        // where the form below leaves the rounding of 1 - cos^2 i (11 + ...)
        // divided by sin i.
        let x = 1.0 - 5.0 * ci2;
        let (c_a, c_b, c_a_o_si, c_b_o_si) = if terms == ShortPeriodTerms::Brouwer && x <= -1.0 {
            let ci4_e = ci2 * ci2 * libm::exp(-100.0 * x * x);
            let ci4_e_o_si = if ci4_e == 0.0 { 0.0 } else { ci4_e / si };
            (
                (si * si * (1.0 - 15.0 * ci2) + 40.0 * ci4_e) / x,
                (si * si * (1.0 - 7.0 * ci2) + 8.0 * ci4_e) / x,
                (si * (1.0 - 15.0 * ci2) + 40.0 * ci4_e_o_si) / x,
                (si * (1.0 - 7.0 * ci2) + 8.0 * ci4_e_o_si) / x,
            )
        } else {
            let c_a = 1.0 - ci2 * (11.0 + 40.0 * ci2_xt2);
            let c_b = 1.0 - ci2 * (3.0 + 8.0 * ci2_xt2);
            (c_a, c_b, c_a / si, c_b / si)
        };
        let c_c = 1.0 - ci2 * (9.0 + 24.0 * ci2_xt2);
        let c_d = 1.0 - ci2 * (5.0 + 16.0 * ci2_xt2);
        let c_e = 1.0 - ci2 * (33.0 + 200.0 * ci2_xt2);
        let c_f = 1.0 - ci2 * (9.0 + 40.0 * ci2_xt2);

        let p5p = 1.0 + ci2_xt2 * (8.0 + 20.0 * ci2_xt2);
        let p5p2 = 1.0 + 2.0 * p5p;
        let p5p4 = 1.0 + 4.0 * p5p;
        let p5p10 = 1.0 + 10.0 * p5p;

        let e2_x3_p4 = 4.0 + 3.0 * epp2;
        let ci_o1_pci = ci / (1.0 + ci);

        let q1 = 0.125 * (yp2 * c_a - fs13 * c_b);
        let q1_o_si = 0.125 * (yp2 * c_a_o_si - fs13 * c_b_o_si);
        let q2 = 0.125 * epp2 * ci * (yp2 * p5p10 - fs13 * p5p2);
        let q5 = 0.25 * (fs12 + 0.3125 * e2_x3_p4 * fs14 * c_c);
        let p2 = 0.46875 * p5p2 * epp * ci * si * e2_x3_p4 * fs14;
        let p3 = 0.15625 * epp * si * fs14 * c_c;
        let kf = 35.0 / 1152.0;
        let p4 = kf * epp * fs14 * c_d;
        let p5 = 2.0 * kf * epp * epp2 * ci * si * fs14 * p5p4;

        let vle1 = epp * n2 * q1;
        let vle2 = n2 * si * q5;
        let vle3 = -3.0 * epp * n2 * si * p4;

        // Long-period inclination: the Delaunay momentum H = G cos(i) has no
        // long-period part and neither does a, so each harmonic of g gives
        // di = -e cos(i) de / (eta^2 sin(i)) (Brouwer 1959), from the
        // eccentricity coefficients above.
        let (vli1, vli3) = match terms {
            ShortPeriodTerms::Brouwer => (-epp2 * ci * q1_o_si, 3.0 * epp2 * ci * p4),
            ShortPeriodTerms::OrekitReplay => (-epp * q1 / si, -3.0 * epp2 * ci * p4),
        };
        let vli2 = -epp * ci * q5;

        let vll2 = vle2 + 3.0 * epp * n2 * p3;

        let vlh1i = -si * q2;
        let vlh2i = epp * ci * q5 + si * p2;
        let vlh3i = -epp2 * ci * p4 - si * p5;

        let vls1 = (n3 - 1.0) * q1 - q2
            + 25.0 * epp2 * ci2 * ci2_xt2 * ci2_xt2 * (yp2 - 0.2 * fs13)
            - 0.0625 * epp2 * (yp2 * c_e - fs13 * c_f);
        let vls2 =
            epp * si * (t8 + ci_o1_pci) * q5 + (11.0 + 3.0 * (epp2 - n3)) * p3 + (1.0 - ci) * p2;
        let vls3 = si * p4 * (3.0 * (n3 - 1.0) - epp2 * (2.0 + ci_o1_pci)) - (1.0 - ci) * p5;

        Ok(Self {
            terms,
            mean: *mean,
            n0,
            n,
            n2,
            n3,
            t8,
            dsl,
            dsg,
            dsh,
            a_rate,
            e_rate,
            yp2,
            ci,
            si,
            one_m_ci2,
            ci2_x3_m1,
            vle1,
            vle2,
            vle3,
            vli1,
            vli2,
            vli3,
            vll2,
            vlh1i,
            vlh2i,
            vlh3i,
            vls1,
            vls2,
            vls3,
        })
    }

    /// Osculating Keplerian elements `dt` seconds after the mean elements'
    /// epoch, with drag coefficient `m2` (rad/s^2).
    fn osculating(&self, dt: f64, m2: f64) -> Result<KeplerianElements, AnalyticPropagationError> {
        let mean = &self.mean;
        let not = dt * self.n0;
        let dt_m2 = dt * m2;
        let dt2_m2 = dt * dt_m2;

        // Secular terms.
        let app = dt_m2 * self.a_rate + mean.a_km;
        let epp = (dt_m2 * self.e_rate + mean.e).clamp(0.0, MAX_ECC);
        let gpp = normalize_angle(
            normalize_angle(mean.arg_perigee_rad, 0.0) + self.dsg * not,
            0.0,
        );
        let hpp = normalize_angle(normalize_angle(mean.raan_rad, 0.0) + self.dsh * not, 0.0);
        let lpp = normalize_angle(
            normalize_angle(mean.mean_anomaly_rad, 0.0) + (1.0 + self.dsl) * not + dt2_m2,
            0.0,
        );

        // Long-period terms.
        let (sgpp, cgpp) = (libm::sin(gpp), libm::cos(gpp));
        let (s2gpp, c2gpp) = (libm::sin(2.0 * gpp), libm::cos(2.0 * gpp));
        let (s3gpp, c3gpp) = (libm::sin(3.0 * gpp), libm::cos(3.0 * gpp));

        let d1e = c2gpp * self.vle1 + sgpp * self.vle2 + s3gpp * self.vle3;
        let mut d1i = sgpp * self.vli2 + s3gpp * self.vli3;
        // Orekit's arrangement divides by sin(i) and leaves 0 / 0 at i = 0;
        // Brouwer's is finite everywhere.
        if self.vli1.is_finite() {
            d1i += c2gpp * self.vli1;
        }
        let eppd1l = (s2gpp * self.vle1 - cgpp * self.vll2 - c3gpp * self.vle3) * self.n;
        let s_ippd1h = s2gpp * self.vlh1i + cgpp * self.vlh2i + c3gpp * self.vlh3i;
        let d1z = s2gpp * self.vls1 + cgpp * self.vls2 + c3gpp * self.vls3;

        // Short-period terms.
        let ShortPeriod {
            d2a,
            d2e,
            d2i,
            eppd2l,
            s_ippd2h,
            d2z,
        } = self.short_period(app, epp, gpp, lpp)?;
        let half_i = 0.5 * mean.i_rad;
        let (si_o2, ci_o2) = (libm::sin(half_i), libm::cos(half_i));

        // Lyddane's non-singular recombination.
        let de = epp + d1e + d2e;
        let dl = eppd1l + eppd2l;
        let dh = (s_ippd1h + s_ippd2h) / (2.0 * ci_o2);
        let di = (d1i + d2i) * (0.5 * ci_o2) + si_o2;
        let z = lpp + gpp + hpp + d1z + d2z;

        let a = app + d2a;
        let e = libm::sqrt(de * de + dl * dl);
        let (slpp, clpp) = (libm::sin(lpp), libm::cos(lpp));
        let l = libm::atan2(de * slpp + dl * clpp, de * clpp - dl * slpp);
        let i = libm::acos(1.0 - 2.0 * (di * di + dh * dh));
        let (shpp, chpp) = (libm::sin(hpp), libm::cos(hpp));
        let h = libm::atan2(di * shpp + dh * chpp, di * chpp - dh * shpp);
        let g = z - l - h;

        let elements = KeplerianElements {
            a_km: finite_result(a, THEORY, "a_km")?,
            e: finite_result(e, THEORY, "e")?,
            i_rad: finite_result(i, THEORY, "i_rad")?,
            arg_perigee_rad: finite_result(g, THEORY, "arg_perigee_rad")?,
            raan_rad: finite_result(h, THEORY, "raan_rad")?,
            mean_anomaly_rad: finite_result(l, THEORY, "mean_anomaly_rad")?,
        };
        if !(elements.a_km > 0.0 && elements.e < 1.0) {
            return Err(AnalyticPropagationError::NotElliptic {
                field: "osculating elements",
            });
        }
        Ok(elements)
    }

    /// Brouwer's J2 short-period terms in Lyddane's variables, at mean
    /// semi-major axis `app`, eccentricity `epp`, argument of perigee `gpp` and
    /// mean anomaly `lpp`.
    fn short_period(
        &self,
        app: f64,
        epp: f64,
        gpp: f64,
        lpp: f64,
    ) -> Result<ShortPeriod, AnalyticPropagationError> {
        let fpp = elliptic_mean_to_true(lpp, epp)?;
        let (sfpp, cfpp) = (libm::sin(fpp), libm::cos(fpp));
        let eppsfpp = epp * sfpp;
        let eppcfpp = epp * cfpp;
        let eppcfpp_p1 = eppcfpp + 1.0;
        let eppcfpp_p2 = eppcfpp + 2.0;
        let eppcfpp_p3 = eppcfpp + 3.0;
        let eppcfpp_p1_3 = eppcfpp_p1 * eppcfpp_p1 * eppcfpp_p1;

        let g2 = 2.0 * gpp;
        let g2f = g2 + fpp;
        let (s2gf, c2gf) = (libm::sin(g2f), libm::cos(g2f));
        let eppc2gf = epp * c2gf;
        let epps2gf = epp * s2gf;
        let g2f2 = g2 + 2.0 * fpp;
        let (s2g2f, c2g2f) = (libm::sin(g2f2), libm::cos(g2f2));
        let g2f3 = g2 + 3.0 * fpp;
        let (s2g3f, c2g3f) = (libm::sin(g2f3), libm::cos(g2f3));
        let eppc2g3f = epp * c2g3f;
        let epps2g3f = epp * s2g3f;

        let w17 = fpp + eppsfpp - lpp;
        let w20 = cfpp * (eppcfpp_p3 * eppcfpp + 3.0);
        let w21 = (s2g2f + epps2gf) * 3.0 + epps2g3f;
        let w22 = eppcfpp_p1 * eppcfpp_p2 / self.n2;

        let (yp2, ci, si, n2, n3) = (self.yp2, self.ci, self.si, self.n2, self.n3);
        let one_m_ci2 = self.one_m_ci2;
        let ci2_x3_m1 = self.ci2_x3_m1;

        let d2a = app
            * (yp2 / n2)
            * ((eppcfpp_p1_3 - n3) * ci2_x3_m1 + eppcfpp_p1_3 * c2g2f * (3.0 * one_m_ci2));
        // Brouwer (1959) short-period eccentricity term, multiplied out of its
        // eta^2 / (2 e) factor:
        //   (yp2 / 2) [ (3 cos^2 i - 1) (w20 + e t8)
        //             + 3 sin^2 i (w20 + e) cos(2g + 2f)
        //             - eta^2 sin^2 i (3 cos(2g + f) + cos(2g + 3f)) ].
        let d2e = match self.terms {
            ShortPeriodTerms::Brouwer => {
                ((w20 + epp * self.t8) * ci2_x3_m1 + (w20 + epp) * c2g2f * (3.0 * one_m_ci2)
                    - (c2gf * 3.0 + c2g3f) * (n2 * one_m_ci2))
                    * (0.5 * yp2)
            }
            ShortPeriodTerms::OrekitReplay => {
                ((w20 + epp * self.t8) * ci2_x3_m1 + (w20 + epp * c2g2f) * (3.0 * one_m_ci2)
                    - (eppc2gf * 3.0 + eppc2g3f) * (n2 * one_m_ci2))
                    * (0.5 * yp2)
            }
        };
        let d2i = ((c2g2f + eppc2gf) * 3.0 + eppc2g3f) * (0.5 * yp2 * ci * si);
        // Brouwer's e dl short-period term:
        //   -(eta^3 yp2 / 4) [ 2 (3 cos^2 i - 1) (w22 + 1) sin f
        //     + 3 sin^2 i ((1 - w22) sin(2g + f) + (w22 + 1/3) sin(2g + 3f)) ].
        let sin_f_factor = match self.terms {
            ShortPeriodTerms::Brouwer => 2.0 * ci2_x3_m1,
            ShortPeriodTerms::OrekitReplay => 2.0 * one_m_ci2,
        };
        let eppd2l = -(((w22 + 1.0) * sfpp * sin_f_factor
            + (-(w22 - 1.0) * s2gf + (w22 + 1.0 / 3.0) * s2g3f) * (3.0 * one_m_ci2))
            * (0.25 * yp2 * n3));
        let s_ippd2h = (w21 - w17 * 6.0) * (0.5 * yp2 * ci * si);
        let k = ci * (2.0 - 5.0 * ci);
        let d2z = -(epp * eppd2l * (self.t8 - 1.0) / n3
            + (w17 * (6.0 * (1.0 + k)) - w21 * (3.0 + k)) * (0.25 * yp2));

        Ok(ShortPeriod {
            d2a,
            d2e,
            d2i,
            eppd2l,
            s_ippd2h,
            d2z,
        })
    }
}

/// The theory for mean elements of any inclination.
///
/// Lyddane's variables `sin(i/2) e^(ih)` and `l + g + h` are regular at
/// `i = 0` but not at `i = pi`, where the long-period node and longitude
/// terms divide by `cos(i/2)` and `1 + cos i`. The zonal problem is
/// invariant under time reversal, which maps a state `(r, v)` at `t` to
/// `(r, -v)` at `-t`: in elements `i -> pi - i`, `w -> pi - w`,
/// `raan -> raan + pi`, `M -> -M`, and the drag term `M2 -> -M2`. Brouwer's
/// secular rates, generating functions and Phipps' factor are covariant under
/// it, so a retrograde orbit (`cos i < 0`) is evaluated as its prograde image
/// at `-dt` and mapped back, which is regular through `i = pi`. The two
/// evaluations agree to first order; they differ only in how the recombination
/// folds the inclination and node corrections `di` and `dh` back into
/// elements, by `(di^2 + dh^2) tan(i/2) / 4` for the direct one and
/// `(di^2 + dh^2) cot(i/2) / 4` for the reversed one. Both corrections are
/// proportional to `cos i`, so the two agree at `i = pi / 2`, where the
/// switch is made, and each side uses the smaller of the two. Orekit's
/// arrangement is replayed as Orekit evaluates it, without the reflection.
///
/// Lyddane's recombination takes `sin(i/2)` and `cos(i/2)` of the evaluated
/// mean inclination as the nonnegative components of a point on the unit
/// circle, which holds for `i` in `[0, pi]` only. The mean elements are
/// therefore first brought into that range as the same orbit
/// ([`KeplerianElements::canonical`]), so the evaluated inclination is in
/// `[0, pi/2]` with the reflection and in `[0, pi]` without it.
#[derive(Debug, Clone, Copy)]
struct Propagation {
    mean: KeplerianElements,
    model: Model,
    reflected: bool,
}

impl Propagation {
    fn new(
        mean: &KeplerianElements,
        g: &ZonalHarmonics,
        terms: ShortPeriodTerms,
    ) -> Result<Self, AnalyticPropagationError> {
        mean.validate("mean elements")?;
        let mean = mean.canonical();
        let reflected = terms == ShortPeriodTerms::Brouwer && libm::cos(mean.i_rad) < 0.0;
        let model = if reflected {
            Model::new(&time_reversed(&mean), g, terms)?
        } else {
            Model::new(&mean, g, terms)?
        };
        Ok(Self {
            mean,
            model,
            reflected,
        })
    }

    fn osculating(&self, dt: f64, m2: f64) -> Result<KeplerianElements, AnalyticPropagationError> {
        if self.reflected {
            Ok(time_reversed(&self.model.osculating(-dt, -m2)?))
        } else {
            self.model.osculating(dt, m2)
        }
    }
}

/// The elements of the time-reversed orbit, `(r, v) -> (r, -v)`; an
/// involution up to the wrapping of the angles into `[-pi, pi)`.
fn time_reversed(k: &KeplerianElements) -> KeplerianElements {
    use core::f64::consts::PI;
    KeplerianElements {
        a_km: k.a_km,
        e: k.e,
        i_rad: PI - k.i_rad,
        arg_perigee_rad: normalize_angle(PI - normalize_angle(k.arg_perigee_rad, 0.0), 0.0),
        raan_rad: normalize_angle(normalize_angle(k.raan_rad, 0.0) + PI, 0.0),
        mean_anomaly_rad: normalize_angle(-k.mean_anomaly_rad, 0.0),
    }
}

struct Theory<'a>(&'a ZonalHarmonics, ShortPeriodTerms);

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
        let theory = Propagation::new(&KeplerianElements::from_equinoctial(mean), self.0, self.1)?;
        // The drag term vanishes at the mean elements' own epoch.
        theory.osculating(0.0, 0.0)?.to_equinoctial()
    }
}

/// Brouwer-Lyddane mean Keplerian elements of an osculating Cartesian state.
///
/// The state's epoch is not used, and neither is a drag coefficient: the `M2`
/// terms vanish at the conversion epoch. `options` sets the fixed-point
/// tolerance and iteration limit; [`MeanConversionOptions::BROUWER_LYDDANE_DEFAULT`]
/// is Orekit's default.
///
/// # Errors
///
/// [`AnalyticPropagationError::NonFinite`] or `InvalidParameter` for invalid
/// inputs, `NotElliptic` for an open orbit, `RetrogradeEquatorial` within
/// 1e-10 rad of `i = pi`, `InsideReferenceSphere` when the osculating
/// semi-major axis is below the reference radius, and
/// `MeanElementsNotConverged` when the tolerance is not met.
pub fn brouwer_lyddane_mean_elements(
    osculating: &CartesianState,
    gravity: &ZonalHarmonics,
    options: MeanConversionOptions,
) -> Result<MeanElements<KeplerianElements>, AnalyticPropagationError> {
    mean_elements(osculating, gravity, options, ShortPeriodTerms::Brouwer)
}

fn mean_elements(
    osculating: &CartesianState,
    gravity: &ZonalHarmonics,
    options: MeanConversionOptions,
    terms: ShortPeriodTerms,
) -> Result<MeanElements<KeplerianElements>, AnalyticPropagationError> {
    gravity.validate()?;
    options.validate()?;
    let osc = Equinoctial::from_cartesian(
        osculating.position_array(),
        osculating.velocity_array(),
        gravity.mu_km3_s2,
    )?;
    let (mean, iterations) = osculating_to_mean(&Theory(gravity, terms), &osc, &options)?;
    let elements = KeplerianElements::from_equinoctial(&mean);
    Propagation::new(&elements, gravity, terms)?;
    Ok(MeanElements {
        elements,
        iterations,
    })
}

/// Brouwer-Lyddane osculating Keplerian elements of mean elements, at the mean
/// elements' own epoch. The mean inclination may be any finite angle and the
/// returned one is in `[0, pi]` (see the [module documentation](super)).
///
/// # Errors
///
/// `NonFinite` or `NotElliptic` for invalid mean elements,
/// `InsideReferenceSphere` for a mean semi-major axis below the reference
/// radius, `NotElliptic` if the osculating elements are not elliptic.
pub fn brouwer_lyddane_osculating_elements(
    mean: &KeplerianElements,
    gravity: &ZonalHarmonics,
) -> Result<KeplerianElements, AnalyticPropagationError> {
    gravity.validate()?;
    Propagation::new(mean, gravity, ShortPeriodTerms::Brouwer)?.osculating(0.0, 0.0)
}

/// Brouwer-Lyddane analytic propagator.
///
/// Holds mean Keplerian elements at an epoch and evaluates the osculating state
/// at any other epoch in closed form. `m2_rad_s2` is the coefficient of the
/// secular along-track term: the mean anomaly gains `M2 dt^2` and the mean
/// semi-major axis and eccentricity decay accordingly. Zero turns it off;
/// fitted values are typically of order 1e-15 to 1e-14 rad/s^2. See the
/// [module documentation](super) for the theory, its domain and the oracle.
///
/// ```
/// use sidereon_core::astro::propagator::analytic::{
///     BrouwerLyddanePropagator, KeplerianElements, ZonalHarmonics,
/// };
///
/// // A GPS-like orbit given by its mean elements.
/// let mean = KeplerianElements {
///     a_km: 26_559.7,
///     e: 0.01,
///     i_rad: 55f64.to_radians(),
///     arg_perigee_rad: 0.5,
///     raan_rad: 4.4,
///     mean_anomaly_rad: 1.7,
/// };
/// let propagator =
///     BrouwerLyddanePropagator::from_mean_elements(0.0, mean, &ZonalHarmonics::earth(), 0.0)
///         .unwrap();
/// let state = propagator.propagate(3.0 * 86_400.0).unwrap();
/// assert_eq!(state.epoch_tdb_seconds, 3.0 * 86_400.0);
/// ```
#[derive(Debug, Clone)]
pub struct BrouwerLyddanePropagator {
    epoch_tdb_seconds: f64,
    gravity: ZonalHarmonics,
    m2_rad_s2: f64,
    model: Propagation,
}

impl BrouwerLyddanePropagator {
    /// Start from an osculating Cartesian state, converting it to mean elements
    /// with [`brouwer_lyddane_mean_elements`].
    ///
    /// # Errors
    ///
    /// As [`brouwer_lyddane_mean_elements`], and `NonFinite` for a non-finite
    /// epoch or `m2_rad_s2`.
    pub fn new(
        initial: &CartesianState,
        gravity: &ZonalHarmonics,
        m2_rad_s2: f64,
        options: MeanConversionOptions,
    ) -> Result<Self, AnalyticPropagationError> {
        Self::new_with(
            initial,
            gravity,
            m2_rad_s2,
            options,
            ShortPeriodTerms::Brouwer,
        )
    }

    fn new_with(
        initial: &CartesianState,
        gravity: &ZonalHarmonics,
        m2_rad_s2: f64,
        options: MeanConversionOptions,
        terms: ShortPeriodTerms,
    ) -> Result<Self, AnalyticPropagationError> {
        finite(initial.epoch_tdb_seconds, "epoch_tdb_seconds")?;
        finite(m2_rad_s2, "m2_rad_s2")?;
        let mean = mean_elements(initial, gravity, options, terms)?;
        Self::from_mean_elements_with(
            initial.epoch_tdb_seconds,
            mean.elements,
            gravity,
            m2_rad_s2,
            terms,
        )
    }

    /// Start from mean Keplerian elements at `epoch_tdb_seconds`.
    ///
    /// # Errors
    ///
    /// `NonFinite`, `InvalidParameter` or `NotElliptic` for invalid inputs,
    /// `InsideReferenceSphere` for a mean semi-major axis below the reference
    /// radius.
    pub fn from_mean_elements(
        epoch_tdb_seconds: f64,
        mean: KeplerianElements,
        gravity: &ZonalHarmonics,
        m2_rad_s2: f64,
    ) -> Result<Self, AnalyticPropagationError> {
        Self::from_mean_elements_with(
            epoch_tdb_seconds,
            mean,
            gravity,
            m2_rad_s2,
            ShortPeriodTerms::Brouwer,
        )
    }

    fn from_mean_elements_with(
        epoch_tdb_seconds: f64,
        mean: KeplerianElements,
        gravity: &ZonalHarmonics,
        m2_rad_s2: f64,
        terms: ShortPeriodTerms,
    ) -> Result<Self, AnalyticPropagationError> {
        finite(epoch_tdb_seconds, "epoch_tdb_seconds")?;
        finite(m2_rad_s2, "m2_rad_s2")?;
        gravity.validate()?;
        Ok(Self {
            epoch_tdb_seconds,
            gravity: *gravity,
            m2_rad_s2,
            model: Propagation::new(&mean, gravity, terms)?,
        })
    }

    /// Epoch of the mean elements, TDB seconds from J2000.
    pub fn epoch_tdb_seconds(&self) -> f64 {
        self.epoch_tdb_seconds
    }

    /// Mean Keplerian elements at [`Self::epoch_tdb_seconds`], with the
    /// inclination in `[0, pi]`: elements given outside that range are
    /// returned as the same orbit inside it (see the
    /// [module documentation](super)).
    pub fn mean_elements(&self) -> KeplerianElements {
        self.model.mean
    }

    /// Gravity field of the theory.
    pub fn gravity(&self) -> &ZonalHarmonics {
        &self.gravity
    }

    /// Secular along-track coefficient `M2`, rad/s^2.
    pub fn m2_rad_s2(&self) -> f64 {
        self.m2_rad_s2
    }

    /// Osculating Keplerian elements at `epoch_tdb_seconds`.
    ///
    /// # Errors
    ///
    /// `NonFinite` for a non-finite epoch, `NotElliptic` if the osculating
    /// elements are not elliptic, `NonFiniteResult` if the evaluation
    /// overflows (only for spans far beyond the theory's accuracy).
    pub fn osculating_elements(
        &self,
        epoch_tdb_seconds: f64,
    ) -> Result<KeplerianElements, AnalyticPropagationError> {
        let dt = finite(epoch_tdb_seconds, "epoch_tdb_seconds")? - self.epoch_tdb_seconds;
        self.model.osculating(dt, self.m2_rad_s2)
    }

    /// Osculating Cartesian state at `epoch_tdb_seconds`: the two-body state of
    /// the osculating elements, km and km/s.
    ///
    /// # Errors
    ///
    /// As [`Self::osculating_elements`].
    pub fn propagate(
        &self,
        epoch_tdb_seconds: f64,
    ) -> Result<CartesianState, AnalyticPropagationError> {
        let elements = self.osculating_elements(epoch_tdb_seconds)?;
        let (position, velocity) = elements
            .to_cartesian_unchecked(self.gravity.mu_km3_s2)
            .map_err(|error| match error {
                AnalyticPropagationError::InvalidParameter { .. } => {
                    AnalyticPropagationError::NonFiniteResult {
                        theory: THEORY,
                        field: "position_km",
                    }
                }
                other => other,
            })?;
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

/// The Brouwer-Lyddane propagator and conversion with Orekit 13.1's
/// arrangement of the periodic terms, which differs from Brouwer (1959) in
/// the short-period eccentricity term, the `sin f` part of the short-period
/// `e dl` and the long-period inclination. Exists only under the test-only
/// `test-replays` feature, so the repository's tests can replay the Orekit
/// fixture to rounding and show that the public propagator differs from it by
/// those terms alone.
#[cfg(feature = "test-replays")]
#[doc(hidden)]
pub mod orekit_replay {
    use super::*;

    /// [`super::brouwer_lyddane_mean_elements`] with Orekit's terms.
    pub fn mean_elements(
        osculating: &CartesianState,
        gravity: &ZonalHarmonics,
        options: MeanConversionOptions,
    ) -> Result<MeanElements<KeplerianElements>, AnalyticPropagationError> {
        super::mean_elements(osculating, gravity, options, ShortPeriodTerms::OrekitReplay)
    }

    /// [`BrouwerLyddanePropagator::new`] with Orekit's terms.
    pub fn new(
        initial: &CartesianState,
        gravity: &ZonalHarmonics,
        m2_rad_s2: f64,
        options: MeanConversionOptions,
    ) -> Result<BrouwerLyddanePropagator, AnalyticPropagationError> {
        BrouwerLyddanePropagator::new_with(
            initial,
            gravity,
            m2_rad_s2,
            options,
            ShortPeriodTerms::OrekitReplay,
        )
    }

    /// [`BrouwerLyddanePropagator::from_mean_elements`] with Orekit's terms.
    pub fn from_mean_elements(
        epoch_tdb_seconds: f64,
        mean: KeplerianElements,
        gravity: &ZonalHarmonics,
        m2_rad_s2: f64,
    ) -> Result<BrouwerLyddanePropagator, AnalyticPropagationError> {
        BrouwerLyddanePropagator::from_mean_elements_with(
            epoch_tdb_seconds,
            mean,
            gravity,
            m2_rad_s2,
            ShortPeriodTerms::OrekitReplay,
        )
    }
}

#[cfg(test)]
pub(super) fn t2_series_for_tests(cos_i: f64) -> f64 {
    t2_series(cos_i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::propagator::analytic::elements::elliptic_mean_to_true;

    fn model(a: f64, e: f64, i_deg: f64, terms: ShortPeriodTerms) -> Model {
        let mean = KeplerianElements {
            a_km: a,
            e,
            i_rad: i_deg.to_radians(),
            arg_perigee_rad: 0.0,
            raan_rad: 0.0,
            mean_anomaly_rad: 0.0,
        };
        Model::new(&mean, &ZonalHarmonics::earth(), terms).unwrap()
    }

    const ORBITS: [(f64, f64, f64); 5] = [
        (7500.0, 0.02, 30.0),
        (7078.137, 0.0012, 98.19),
        (26_600.0, 0.3, 40.0),
        (26_600.0, 0.72, 63.43),
        (42_164.0, 0.01, 120.0),
    ];

    #[test]
    fn long_period_inclination_follows_from_the_eccentricity_terms() {
        // Brouwer (1959): neither a nor the Delaunay H = sqrt(mu a (1 - e^2))
        // cos(i) has a long-period part, so each harmonic of g in the
        // long-period inclination is -e cos(i) / (eta^2 sin(i)) times that
        // harmonic in the eccentricity.
        for (a, e, i) in ORBITS {
            let m = model(a, e, i, ShortPeriodTerms::Brouwer);
            let k = -e * m.ci / (m.n2 * m.si);
            for (di, de) in [(m.vli1, m.vle1), (m.vli2, m.vle2), (m.vli3, m.vle3)] {
                assert!(
                    (di - k * de).abs() <= 1.0e-14 * de.abs().max(1.0e-12),
                    "a {a} e {e} i {i}: {di} vs {}",
                    k * de
                );
            }
        }
    }

    /// Brouwer's short-period terms in the form of his 1959 paper (Eqs. for
    /// da, de, di, e dl, dh and d(l + g + h) with gamma2 = J2 Re^2 / (2 a^2),
    /// gamma2' = gamma2 / eta^4 and a / r = (1 + e cos f) / eta^2), the form
    /// also implemented in NASA GMAT's `BrouwerMeanShortToOsculatingElements`.
    fn published(m: &Model, gpp: f64, lpp: f64) -> [f64; 6] {
        let e = m.mean.e;
        let a = m.mean.a_km;
        let eta2 = 1.0 - e * e;
        let eta = eta2.sqrt();
        let theta = m.ci;
        let th2 = theta * theta;
        let g = ZonalHarmonics::earth();
        let gm2 = -0.5 * g.c20 * (g.re_km / a).powi(2);
        let gm2p = gm2 / (eta2 * eta2);
        let f = elliptic_mean_to_true(lpp, e).unwrap();
        let adr = (1.0 + e * libm::cos(f)) / eta2;
        let c = libm::cos;
        let s = libm::sin;
        let da = a
            * gm2
            * ((adr.powi(3) - eta.powi(-3)) * (3.0 * th2 - 1.0)
                + 3.0 * (1.0 - th2) * adr.powi(3) * c(2.0 * gpp + 2.0 * f));
        let de = eta2 / (2.0 * e)
            * (gm2
                * ((3.0 * th2 - 1.0) * (adr.powi(3) - eta.powi(-3))
                    + 3.0 * (1.0 - th2) * (adr.powi(3) - eta.powi(-4)) * c(2.0 * gpp + 2.0 * f))
                - gm2p * (1.0 - th2) * (3.0 * e * c(2.0 * gpp + f) + e * c(2.0 * gpp + 3.0 * f)));
        let di = 0.5
            * gm2p
            * theta
            * m.si
            * (3.0 * c(2.0 * gpp + 2.0 * f)
                + 3.0 * e * c(2.0 * gpp + f)
                + e * c(2.0 * gpp + 3.0 * f));
        let w = adr * adr * eta2 + adr;
        let edl = -eta.powi(3) / 4.0
            * gm2p
            * (2.0 * (3.0 * th2 - 1.0) * (w + 1.0) * s(f)
                + 3.0
                    * (1.0 - th2)
                    * ((1.0 - w) * s(2.0 * gpp + f) + (w + 1.0 / 3.0) * s(2.0 * gpp + 3.0 * f)));
        let w17 = f - lpp + e * s(f);
        let w21 =
            3.0 * s(2.0 * gpp + 2.0 * f) + 3.0 * e * s(2.0 * gpp + f) + e * s(2.0 * gpp + 3.0 * f);
        let dh = -0.5 * gm2p * theta * (6.0 * w17 - w21);
        let dz = gm2p / 4.0
            * (6.0 * (-1.0 - 2.0 * theta + 5.0 * th2) * w17
                + (3.0 + 2.0 * theta - 5.0 * th2) * w21)
            + gm2p / 4.0 * eta2 / (eta + 1.0)
                * e
                * (3.0
                    * (1.0 - th2)
                    * (s(2.0 * gpp + 3.0 * f) * (1.0 / 3.0 + w) + s(2.0 * gpp + f) * (1.0 - w))
                    + 2.0 * s(f) * (3.0 * th2 - 1.0) * (1.0 + w));
        [da, de, di, edl, m.si * dh, dz]
    }

    #[test]
    fn short_period_terms_match_the_published_form() {
        // The Lyddane arrangement multiplies the 1 / e of the published
        // eccentricity term through; both are the same function, so they
        // agree to rounding (relative 1e-12 of the largest term covers the
        // cancellation in de at e = 0.0012).
        for (a, e, i) in ORBITS {
            let m = model(a, e, i, ShortPeriodTerms::Brouwer);
            let r = model(a, e, i, ShortPeriodTerms::OrekitReplay);
            let mut replay_differs = false;
            for k in 0..24 {
                let gpp = 0.37 * k as f64;
                let lpp = normalize_angle(1.13 * k as f64 + 0.2, 0.0);
                let sp = m.short_period(a, e, gpp, lpp).unwrap();
                let ours = [sp.d2a, sp.d2e, sp.d2i, sp.eppd2l, sp.s_ippd2h, sp.d2z];
                let theirs = published(&m, gpp, lpp);
                let scale = theirs.iter().skip(1).fold(0.0f64, |x, y| x.max(y.abs()));
                for q in 0..6 {
                    let tol = if q == 0 {
                        1.0e-12 * a * scale
                    } else {
                        1.0e-11 * scale
                    };
                    assert!(
                        (ours[q] - theirs[q]).abs() <= tol,
                        "a {a} e {e} i {i} g {gpp} l {lpp} term {q}: {} vs {}",
                        ours[q],
                        theirs[q]
                    );
                }
                let rp = r.short_period(a, e, gpp, lpp).unwrap();
                replay_differs |= (rp.d2e - sp.d2e).abs() > 1.0e-3 * scale;
            }
            assert!(replay_differs, "a {a} e {e} i {i}");
        }
    }

    #[test]
    fn time_reversed_evaluation_agrees_with_the_direct_one() {
        // Away from both poles the direct and the time-reversed evaluations
        // are both regular and agree to first order. They differ only in how
        // Lyddane's recombination folds the inclination and node corrections
        // di and dh back into elements, at second order: (di^2 + dh^2)
        // (tan(i/2) + cot(i/2)) / 4 = (di^2 + dh^2) / (2 sin i) in angle. Near
        // the critical inclination the J5 long-period node term carries
        // 1 / (1 - 5 cos^2 i)^2, so dh reaches 2e-3 rad at 120 degrees.
        let g = ZonalHarmonics::earth();
        let mut largest_ratio = 0.0f64;
        for (a, e, i_deg) in [
            (7078.137, 0.0012, 98.19),
            (7500.0, 0.02, 91.0),
            (8000.0, 0.1, 120.0),
            (26_600.0, 0.3, 140.0),
            (42_164.0, 0.01, 100.0),
            (7500.0, 0.02, 175.0),
        ] {
            let mean = KeplerianElements {
                a_km: a,
                e,
                i_rad: f64::to_radians(i_deg),
                arg_perigee_rad: 0.4,
                raan_rad: 1.1,
                mean_anomaly_rad: 0.3,
            };
            let reversed = Propagation::new(&mean, &g, ShortPeriodTerms::Brouwer).unwrap();
            assert!(reversed.reflected);
            let direct = Model::new(&mean, &g, ShortPeriodTerms::Brouwer).unwrap();
            for (dt, m2) in [
                (0.0, 0.0),
                (3_600.0, 0.0),
                (86_400.0, 1.0e-14),
                (-86_400.0, 1.0e-14),
            ] {
                let x = reversed.osculating(dt, m2).unwrap();
                let y = direct.osculating(dt, m2).unwrap();
                // First-order corrections of the direct evaluation.
                let di = y.i_rad - mean.i_rad;
                let secular_raan = mean.raan_rad + direct.dsh * direct.n0 * dt;
                let dh = normalize_angle(y.raan_rad - secular_raan, 0.0);
                let second_order = (di * di + dh * dh) / (2.0 * libm::sin(mean.i_rad));
                let (px, _) = x.to_cartesian(g.mu_km3_s2).unwrap();
                let (py, _) = y.to_cartesian(g.mu_km3_s2).unwrap();
                let d = libm::sqrt((0..3).map(|k| (px[k] - py[k]).powi(2)).sum());
                let r_max = a * (1.0 + e);
                let ratio = d / (r_max * second_order + 1.0e-12);
                largest_ratio = largest_ratio.max(ratio);
                assert!(
                    ratio <= 4.0,
                    "a {a} e {e} i {i_deg} dt {dt}: {d} km against r (di^2 + dh^2) / (2 sin i) = {} km",
                    r_max * second_order
                );
            }
        }
        // The bound is met, and not vacuously.
        assert!(largest_ratio > 0.05, "{largest_ratio}");
    }
}
