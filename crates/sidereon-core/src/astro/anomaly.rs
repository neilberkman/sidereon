//! Anomaly conversions and analytic two-body propagation for classical elements.
//!
//! The scalar conversion functions use the same names across all conic regimes.
//! The middle anomaly is eccentric anomaly `E` for elliptic orbits, hyperbolic
//! anomaly `H` for hyperbolic orbits, and Barker's parabolic anomaly `D` for the
//! parabolic boundary selected by eccentricity.

use crate::astro::elements::{ClassicalElements, OrbitType};
use crate::astro::math::{normalize_angle, wrap_to_pi, SMALL};

const PI: f64 = std::f64::consts::PI;
const TOL_ABS: f64 = 1.0e-12;
const TOL_REL: f64 = 1.0e-14;
const MAX_ITER: usize = 50;

/// Errors returned by the anomaly conversions, [`solve_kepler`], and
/// [`propagate_kepler`].
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum AnomalyError {
    /// A checked floating-point input was not finite.
    #[error("non-finite input {field}")]
    NonFinite {
        /// Names the input that was NaN or infinite.
        field: &'static str,
    },
    /// The supplied eccentricity was negative.
    #[error("eccentricity must be non-negative")]
    NegativeEccentricity,
    /// Propagation received a non-positive gravitational parameter.
    #[error("mu must be positive")]
    NonPositiveMu,
    /// Propagation received a non-positive semi-latus rectum.
    #[error("semi-latus rectum must be positive")]
    NonPositiveSemiLatus,
    /// A true anomaly was at or beyond the admissible open-orbit limit.
    #[error("true anomaly {nu} is beyond the open-orbit asymptote {limit}")]
    BeyondAsymptote {
        /// The original, unwrapped true anomaly that failed the domain check.
        nu: f64,
        /// The parabolic or hyperbolic open-orbit limit used by the check.
        limit: f64,
    },
    /// The supplied classical elements are inconsistent with their conic or orbit type.
    #[error("inconsistent element {field}")]
    InconsistentElements {
        /// Names the invalid semi-major-axis, eccentricity, or inclination field.
        field: &'static str,
    },
    /// The iterative elliptic or hyperbolic solve exhausted its iteration limit.
    #[error("Kepler solve did not converge in {iterations} iters (residual {residual})")]
    NonConvergent {
        /// The number of iterations attempted, which is `MAX_ITER` on this error.
        iterations: usize,
        /// The residual from the final iteration.
        residual: f64,
    },
}

/// Middle-anomaly result and iteration count returned by [`solve_kepler`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeplerSolution {
    /// Eccentric anomaly `E`, hyperbolic anomaly `H`, or Barker anomaly `D`.
    /// Elliptic results are normalized to `[0, 2*pi)`.
    pub anomaly: f64,
    /// The converged 1-based iteration count, or zero for the direct parabolic inversion.
    pub iterations: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Regime {
    Elliptic,
    Parabolic,
    Hyperbolic,
}

/// Solve Kepler's equation for the conic regime selected by `ecc`.
///
/// Mean, eccentric, hyperbolic, and Barker anomalies are in radians; `ecc` is
/// dimensionless. Elliptic mean anomaly is reduced to `[0, 2*pi)` before the
/// iterative solve, while the parabolic branch uses Barker's direct inversion.
///
/// # Errors
///
/// Returns [`AnomalyError::NonFinite`] for a non-finite input,
/// [`AnomalyError::NegativeEccentricity`] for negative `ecc`, or
/// [`AnomalyError::NonConvergent`] when an elliptic or hyperbolic iteration
/// does not reach its residual tolerance.
pub fn solve_kepler(mean_anom: f64, ecc: f64) -> Result<KeplerSolution, AnomalyError> {
    check_finite(mean_anom, "mean_anom")?;
    validate_ecc(ecc)?;

    match regime(ecc) {
        Regime::Elliptic => solve_iterative(
            normalize_angle(mean_anom),
            ecc,
            Regime::Elliptic,
            elliptic_seed(mean_anom, ecc),
        ),
        Regime::Parabolic => Ok(KeplerSolution {
            anomaly: barker_d_from_mean(mean_anom),
            iterations: 0,
        }),
        Regime::Hyperbolic => solve_iterative(
            mean_anom,
            ecc,
            Regime::Hyperbolic,
            libm::asinh(mean_anom / ecc),
        ),
    }
}

/// Convert mean anomaly to the middle anomaly for the selected conic regime.
///
/// The returned value is the `anomaly` field from [`solve_kepler`]: `E` for an
/// elliptic orbit, `H` for a hyperbolic orbit, or Barker's `D` at the
/// parabolic boundary. Angles are in radians and eccentricity is dimensionless.
///
/// # Errors
///
/// Propagates validation and convergence errors from [`solve_kepler`].
pub fn mean_to_eccentric(mean_anom: f64, ecc: f64) -> Result<f64, AnomalyError> {
    Ok(solve_kepler(mean_anom, ecc)?.anomaly)
}

/// Convert eccentric, hyperbolic, or Barker anomaly to mean anomaly.
///
/// The elliptic branch computes `E - ecc*sin(E)` and normalizes the result to
/// `[0, 2*pi)`. The parabolic branch computes `D + D^3/3`, and the hyperbolic
/// branch computes `ecc*sinh(H) - H`.
///
/// # Errors
///
/// Returns [`AnomalyError::NonFinite`] if `ecc_anom` or `ecc` is not finite,
/// or [`AnomalyError::NegativeEccentricity`] if `ecc` is negative.
pub fn eccentric_to_mean(ecc_anom: f64, ecc: f64) -> Result<f64, AnomalyError> {
    check_finite(ecc_anom, "ecc_anom")?;
    validate_ecc(ecc)?;

    match regime(ecc) {
        Regime::Elliptic => {
            let anomaly = normalize_angle(ecc_anom);
            Ok(normalize_angle(anomaly - ecc * libm::sin(anomaly)))
        }
        Regime::Parabolic => Ok(ecc_anom + ecc_anom.powi(3) / 3.0),
        Regime::Hyperbolic => Ok(ecc * libm::sinh(ecc_anom) - ecc_anom),
    }
}

/// Convert eccentric, Barker, or hyperbolic anomaly to true anomaly.
///
/// The elliptic and hyperbolic branches use the corresponding sine and cosine
/// relations, while the parabolic branch uses `nu = 2*atan(D)`. The returned
/// true anomaly is normalized to `[0, 2*pi)`.
///
/// # Errors
///
/// Returns [`AnomalyError::NonFinite`] if `ecc_anom` or `ecc` is not finite,
/// or [`AnomalyError::NegativeEccentricity`] if `ecc` is negative.
pub fn eccentric_to_true(ecc_anom: f64, ecc: f64) -> Result<f64, AnomalyError> {
    check_finite(ecc_anom, "ecc_anom")?;
    validate_ecc(ecc)?;

    match regime(ecc) {
        Regime::Elliptic => {
            let anomaly = normalize_angle(ecc_anom);
            let sin_nu = ((1.0 - ecc) * (1.0 + ecc)).sqrt() * libm::sin(anomaly);
            let cos_nu = libm::cos(anomaly) - ecc;
            Ok(normalize_angle(libm::atan2(sin_nu, cos_nu)))
        }
        Regime::Parabolic => Ok(normalize_angle(2.0 * libm::atan(ecc_anom))),
        Regime::Hyperbolic => {
            let sin_nu = ((ecc - 1.0) * (ecc + 1.0)).sqrt() * libm::sinh(ecc_anom);
            let cos_nu = ecc - libm::cosh(ecc_anom);
            Ok(normalize_angle(libm::atan2(sin_nu, cos_nu)))
        }
    }
}

/// Convert true anomaly to eccentric, Barker, or hyperbolic anomaly.
///
/// Elliptic input is converted with the half-angle relation and normalized to
/// `[0, 2*pi)`. The parabolic result is `D = tan(nu/2)`, and the hyperbolic
/// result is obtained with `asinh` after wrapping `nu` to `(-pi, pi]`.
///
/// # Errors
///
/// Returns [`AnomalyError::NonFinite`] if `true_anom` or `ecc` is not finite,
/// [`AnomalyError::NegativeEccentricity`] if `ecc` is negative, or
/// [`AnomalyError::BeyondAsymptote`] when an open-orbit input is at or beyond
/// its parabolic or hyperbolic limit.
pub fn true_to_eccentric(true_anom: f64, ecc: f64) -> Result<f64, AnomalyError> {
    check_finite(true_anom, "true_anom")?;
    validate_ecc(ecc)?;

    match regime(ecc) {
        Regime::Elliptic => {
            let nu = normalize_angle(true_anom);
            let half_nu = 0.5 * nu;
            let s = (1.0 - ecc).sqrt() * libm::sin(half_nu);
            let c = (1.0 + ecc).sqrt() * libm::cos(half_nu);
            let sin_e = 2.0 * s * c;
            let cos_e = c * c - s * s;
            Ok(normalize_angle(libm::atan2(sin_e, cos_e)))
        }
        Regime::Parabolic => {
            check_open_true_anomaly(true_anom, PI)?;
            Ok(libm::tan(0.5 * wrap_to_pi(true_anom)))
        }
        Regime::Hyperbolic => {
            let limit = libm::acos(-1.0 / ecc);
            check_open_true_anomaly(true_anom, limit)?;
            let nu = wrap_to_pi(true_anom);
            Ok(libm::asinh(
                libm::sin(nu) * ((ecc - 1.0) * (ecc + 1.0)).sqrt() / (1.0 + ecc * libm::cos(nu)),
            ))
        }
    }
}

/// Convert mean anomaly directly to normalized true anomaly.
///
/// This composes [`mean_to_eccentric`] with [`eccentric_to_true`], selecting
/// `E`, `H`, or `D` from `ecc` before returning the true anomaly in
/// `[0, 2*pi)`.
///
/// # Errors
///
/// Returns the first validation, convergence, or open-orbit-domain error from
/// the two conversions.
pub fn mean_to_true(mean_anom: f64, ecc: f64) -> Result<f64, AnomalyError> {
    let ecc_anom = mean_to_eccentric(mean_anom, ecc)?;
    eccentric_to_true(ecc_anom, ecc)
}

/// Convert true anomaly directly to mean anomaly.
///
/// This composes [`true_to_eccentric`] with [`eccentric_to_mean`]. Open-orbit
/// domain checks therefore occur before the mean anomaly is evaluated.
///
/// # Errors
///
/// Returns the first validation or open-orbit-domain error from the two
/// conversions.
pub fn true_to_mean(true_anom: f64, ecc: f64) -> Result<f64, AnomalyError> {
    let ecc_anom = true_to_eccentric(true_anom, ecc)?;
    eccentric_to_mean(ecc_anom, ecc)
}

/// Propagate classical elements for a two-body Keplerian orbit by `dt`.
///
/// The input is copied before propagation. Circular inclined and circular
/// equatorial elements advance `arglat` and `truelon`, respectively; eccentric
/// orbit types advance `nu` through mean anomaly using `mu` and the relevant
/// `a` or `p`. All other fields are preserved.
///
/// # Errors
///
/// Returns [`AnomalyError::NonFinite`], [`AnomalyError::NonPositiveMu`],
/// [`AnomalyError::NonPositiveSemiLatus`], or
/// [`AnomalyError::InconsistentElements`] when the inputs fail validation, or
/// an anomaly-conversion error when the propagated anomaly cannot be computed.
pub fn propagate_kepler(
    elements: &ClassicalElements,
    mu: f64,
    dt: f64,
) -> Result<ClassicalElements, AnomalyError> {
    validate_propagation_inputs(elements, mu, dt)?;

    let mut out = *elements;
    let conic = regime(elements.ecc);

    match elements.orbit_type {
        OrbitType::CircularInclined => {
            let mean_motion = (mu / elements.a.powi(3)).sqrt();
            out.arglat = normalize_angle(elements.arglat + mean_motion * dt);
        }
        OrbitType::CircularEquatorial => {
            let mean_motion = (mu / elements.a.powi(3)).sqrt();
            out.truelon = normalize_angle(elements.truelon + mean_motion * dt);
        }
        OrbitType::EllipticalInclined | OrbitType::EllipticalEquatorial => {
            let mean0 = true_to_mean(elements.nu, elements.ecc)?;
            let mean1 = match conic {
                Regime::Elliptic => {
                    let mean_motion = (mu / elements.a.powi(3)).sqrt();
                    normalize_angle(mean0 + mean_motion * dt)
                }
                Regime::Parabolic => {
                    let mean_motion = (mu / elements.p.powi(3)).sqrt();
                    mean0 + mean_motion * dt
                }
                Regime::Hyperbolic => {
                    let mean_motion = (mu / (-elements.a).powi(3)).sqrt();
                    mean0 + mean_motion * dt
                }
            };
            out.nu = mean_to_true(mean1, elements.ecc)?;
        }
    }

    Ok(out)
}

fn elliptic_seed(mean_anom: f64, ecc: f64) -> f64 {
    let mean = normalize_angle(mean_anom);
    if ecc < 0.8 {
        mean + ecc * libm::sin(mean)
    } else {
        PI
    }
}

fn solve_iterative(
    mean_anom: f64,
    ecc: f64,
    conic: Regime,
    seed: f64,
) -> Result<KeplerSolution, AnomalyError> {
    let mut anomaly = seed;
    let tolerance = TOL_ABS + TOL_REL * mean_anom.abs();
    let mut residual = f64::NAN;

    for iterations in 1..=MAX_ITER {
        let (f, fp, fpp) = residuals(anomaly, mean_anom, ecc, conic);
        residual = f;
        if f.abs() <= tolerance {
            return Ok(KeplerSolution {
                anomaly: if conic == Regime::Elliptic {
                    normalize_angle(anomaly)
                } else {
                    anomaly
                },
                iterations,
            });
        }

        let denom = 2.0 * fp * fp - f * fpp;
        let step = if denom.is_finite() && denom.abs() > f64::EPSILON {
            2.0 * f * fp / denom
        } else {
            f / fp
        };
        anomaly -= step;
    }

    Err(AnomalyError::NonConvergent {
        iterations: MAX_ITER,
        residual,
    })
}

fn residuals(anomaly: f64, mean_anom: f64, ecc: f64, conic: Regime) -> (f64, f64, f64) {
    match conic {
        Regime::Elliptic => (
            anomaly - ecc * libm::sin(anomaly) - mean_anom,
            1.0 - ecc * libm::cos(anomaly),
            ecc * libm::sin(anomaly),
        ),
        Regime::Hyperbolic => (
            ecc * libm::sinh(anomaly) - anomaly - mean_anom,
            ecc * libm::cosh(anomaly) - 1.0,
            ecc * libm::sinh(anomaly),
        ),
        Regime::Parabolic => unreachable!(),
    }
}

fn barker_d_from_mean(mean_anom: f64) -> f64 {
    2.0 * libm::sinh(libm::asinh(1.5 * mean_anom) / 3.0)
}

fn validate_propagation_inputs(
    elements: &ClassicalElements,
    mu: f64,
    dt: f64,
) -> Result<(), AnomalyError> {
    check_finite(mu, "mu")?;
    if mu <= 0.0 {
        return Err(AnomalyError::NonPositiveMu);
    }
    check_finite(dt, "dt")?;
    validate_ecc(elements.ecc)?;
    check_finite(elements.p, "p")?;
    if elements.p <= 0.0 {
        return Err(AnomalyError::NonPositiveSemiLatus);
    }
    check_finite(elements.incl, "incl")?;

    match regime(elements.ecc) {
        Regime::Elliptic => {
            if !elements.a.is_finite() || elements.a <= 0.0 {
                return Err(AnomalyError::InconsistentElements { field: "a" });
            }
        }
        Regime::Parabolic => {}
        Regime::Hyperbolic => {
            if !elements.a.is_finite() || elements.a >= 0.0 {
                return Err(AnomalyError::InconsistentElements { field: "a" });
            }
        }
    }

    validate_orbit_type_fields(elements)
}

fn validate_orbit_type_fields(elements: &ClassicalElements) -> Result<(), AnomalyError> {
    let equatorial = is_equatorial(elements.incl);

    match elements.orbit_type {
        OrbitType::EllipticalInclined => {
            check_finite(elements.raan, "raan")?;
            check_finite(elements.argp, "argp")?;
            check_finite(elements.nu, "nu")?;
            require_eccentric(elements.ecc)?;
            require_inclined(equatorial)?;
        }
        OrbitType::EllipticalEquatorial => {
            check_finite(elements.lonper, "lonper")?;
            check_finite(elements.nu, "nu")?;
            require_eccentric(elements.ecc)?;
            require_equatorial(equatorial)?;
        }
        OrbitType::CircularInclined => {
            check_finite(elements.raan, "raan")?;
            check_finite(elements.arglat, "arglat")?;
            require_circular(elements.ecc)?;
            require_inclined(equatorial)?;
        }
        OrbitType::CircularEquatorial => {
            check_finite(elements.truelon, "truelon")?;
            require_circular(elements.ecc)?;
            require_equatorial(equatorial)?;
        }
    }

    Ok(())
}

fn require_eccentric(ecc: f64) -> Result<(), AnomalyError> {
    if ecc >= SMALL {
        Ok(())
    } else {
        Err(AnomalyError::InconsistentElements { field: "ecc" })
    }
}

fn require_circular(ecc: f64) -> Result<(), AnomalyError> {
    if ecc < SMALL {
        Ok(())
    } else {
        Err(AnomalyError::InconsistentElements { field: "ecc" })
    }
}

fn require_equatorial(equatorial: bool) -> Result<(), AnomalyError> {
    if equatorial {
        Ok(())
    } else {
        Err(AnomalyError::InconsistentElements { field: "incl" })
    }
}

fn require_inclined(equatorial: bool) -> Result<(), AnomalyError> {
    if !equatorial {
        Ok(())
    } else {
        Err(AnomalyError::InconsistentElements { field: "incl" })
    }
}

fn is_equatorial(incl: f64) -> bool {
    incl < SMALL || (incl - PI).abs() < SMALL
}

fn validate_ecc(ecc: f64) -> Result<(), AnomalyError> {
    check_finite(ecc, "ecc")?;
    if ecc < 0.0 {
        Err(AnomalyError::NegativeEccentricity)
    } else {
        Ok(())
    }
}

fn check_open_true_anomaly(true_anom: f64, limit: f64) -> Result<(), AnomalyError> {
    if wrap_to_pi(true_anom).abs() >= limit {
        Err(AnomalyError::BeyondAsymptote {
            nu: true_anom,
            limit,
        })
    } else {
        Ok(())
    }
}

fn check_finite(value: f64, field: &'static str) -> Result<(), AnomalyError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(AnomalyError::NonFinite { field })
    }
}

fn regime(ecc: f64) -> Regime {
    if ecc <= 1.0 - SMALL {
        Regime::Elliptic
    } else if ecc < 1.0 + SMALL {
        Regime::Parabolic
    } else {
        Regime::Hyperbolic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEG: f64 = std::f64::consts::PI / 180.0;

    fn assert_close(got: f64, want: f64, tol: f64, label: &str) {
        assert!(
            (got - want).abs() <= tol,
            "{label}: got {got}, want {want}, diff {}",
            (got - want).abs()
        );
    }

    fn assert_angle_close(got: f64, want: f64, tol: f64, label: &str) {
        let diff = wrap_to_pi(got - want).abs();
        assert!(diff <= tol, "{label}: got {got}, want {want}, diff {diff}");
    }

    #[test]
    fn vallado_example_2_1_elliptic_kepler() {
        // Vallado 2022 Example 2-1, KepEqtnE: M = 235.4 deg, e = 0.4,
        // E = 220.512074 deg.
        let eccentric = mean_to_eccentric(235.4 * DEG, 0.4).unwrap();
        assert_close(eccentric, 220.512074 * DEG, 1.0e-6, "E");
    }

    #[test]
    fn elliptic_round_trips_across_grid() {
        let eccs = [
            0.0,
            1.0e-9,
            0.01,
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
            0.99,
            1.0 - 1.0e-6,
        ];

        for ecc in eccs {
            for step in 0..24 {
                let mean = step as f64 * crate::astro::math::TWO_PI / 24.0;
                let eccentric = mean_to_eccentric(mean, ecc).unwrap();
                let true_anom = eccentric_to_true(eccentric, ecc).unwrap();

                assert_angle_close(
                    eccentric_to_mean(eccentric, ecc).unwrap(),
                    mean,
                    1.0e-11,
                    "M",
                );
                assert_angle_close(
                    true_to_eccentric(true_anom, ecc).unwrap(),
                    eccentric,
                    1.0e-11,
                    "E",
                );
                assert_angle_close(
                    true_to_mean(true_anom, ecc).unwrap(),
                    mean,
                    1.0e-11,
                    "M true",
                );

                let solution = solve_kepler(mean, ecc).unwrap();
                let residual = wrap_to_pi(eccentric_to_mean(solution.anomaly, ecc).unwrap() - mean);
                assert!(solution.iterations <= MAX_ITER);
                assert!(residual.abs() <= TOL_ABS + TOL_REL * mean.abs());
            }
        }
    }

    #[test]
    fn hyperbolic_round_trips_across_grid() {
        let eccs = [1.0 + 1.0e-4, 1.1, 1.5, 2.4, 5.0];
        let means = [-10.0, -3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0, 10.0];

        for ecc in eccs {
            for mean in means {
                let hyper = mean_to_eccentric(mean, ecc).unwrap();
                let true_anom = eccentric_to_true(hyper, ecc).unwrap();
                let mean_back = eccentric_to_mean(hyper, ecc).unwrap();
                let hyper_back = true_to_eccentric(true_anom, ecc).unwrap();
                let mean_from_true = true_to_mean(true_anom, ecc).unwrap();

                let scale = 1.0_f64.max(mean.abs());
                assert!((mean_back - mean).abs() <= 1.0e-9 * scale);
                assert!((hyper_back - hyper).abs() <= 1.0e-9 * 1.0_f64.max(hyper.abs()));
                assert!((mean_from_true - mean).abs() <= 1.0e-9 * scale);

                let solution = solve_kepler(mean, ecc).unwrap();
                let residual = ecc * libm::sinh(solution.anomaly) - solution.anomaly - mean;
                assert!(solution.iterations <= MAX_ITER);
                assert!(residual.abs() <= TOL_ABS + TOL_REL * mean.abs());
            }
        }
    }

    #[test]
    fn parabolic_barker_round_trips_wide_signed_range() {
        for step in 0..24 {
            if step == 12 {
                continue;
            }
            let true_anom = step as f64 * crate::astro::math::TWO_PI / 24.0;
            let d = true_to_eccentric(true_anom, 1.0).unwrap();
            let mean = eccentric_to_mean(d, 1.0).unwrap();
            let d_back = mean_to_eccentric(mean, 1.0).unwrap();
            let true_back = mean_to_true(mean, 1.0).unwrap();

            assert!((d_back - d).abs() <= 1.0e-11 * 1.0_f64.max(d.abs()));
            assert_angle_close(true_back, true_anom, 1.0e-11, "parabolic true");
        }

        let means = [-1.0e6, -10.0, -0.1, 0.0, 0.1, 10.0, 1.0e6];

        for mean in means {
            let d = mean_to_eccentric(mean, 1.0).unwrap();
            let true_anom = eccentric_to_true(d, 1.0).unwrap();
            let d_back = true_to_eccentric(true_anom, 1.0).unwrap();
            let mean_back = true_to_mean(true_anom, 1.0).unwrap();

            assert!((d_back - d).abs() <= 1.0e-11 * 1.0_f64.max(d.abs()));
            assert!((mean_back - mean).abs() <= 1.0e-9 * 1.0_f64.max(mean.abs()));
            assert_eq!(solve_kepler(mean, 1.0).unwrap().iterations, 0);
        }
    }

    #[test]
    fn large_elliptic_mean_anomaly_is_reduced() {
        let baseline = mean_to_eccentric(1.3, 0.7).unwrap();
        let many_revs = mean_to_eccentric(100.0 * crate::astro::math::TWO_PI + 1.3, 0.7).unwrap();
        assert_angle_close(many_revs, baseline, 1.0e-12, "large M");
    }

    #[test]
    fn degenerate_circular_anomalies_match() {
        let mean = 5.8;
        assert_angle_close(mean_to_eccentric(mean, 0.0).unwrap(), mean, 1.0e-14, "E");
        assert_angle_close(mean_to_true(mean, 0.0).unwrap(), mean, 1.0e-14, "nu");
        assert_angle_close(true_to_mean(mean, 0.0).unwrap(), mean, 1.0e-14, "M");
    }

    #[test]
    fn true_anomaly_at_or_beyond_open_asymptote_is_rejected() {
        assert!(matches!(
            true_to_eccentric(PI, 1.0),
            Err(AnomalyError::BeyondAsymptote { .. })
        ));

        let ecc = 1.5;
        let limit = libm::acos(-1.0_f64 / ecc);
        assert!(matches!(
            true_to_eccentric(limit, ecc),
            Err(AnomalyError::BeyondAsymptote { .. })
        ));
        assert!(matches!(
            true_to_mean(limit + 1.0e-3, ecc),
            Err(AnomalyError::BeyondAsymptote { .. })
        ));
    }

    #[test]
    fn scalar_error_paths_are_distinct() {
        assert_eq!(
            mean_to_eccentric(f64::NAN, 0.1),
            Err(AnomalyError::NonFinite { field: "mean_anom" })
        );
        assert_eq!(
            mean_to_eccentric(0.0, f64::INFINITY),
            Err(AnomalyError::NonFinite { field: "ecc" })
        );
        assert_eq!(
            mean_to_eccentric(0.0, -0.1),
            Err(AnomalyError::NegativeEccentricity)
        );
    }
}
