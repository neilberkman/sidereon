//! Analytic zonal-harmonic propagators for Earth orbits.
//!
//! Two closed-form theories that advance a set of mean elements under the
//! zonal part of the geopotential and add the periodic terms back to give the
//! osculating state at any epoch, with no numerical integration:
//!
//! - [`EcksteinHechlerPropagator`]: the Eckstein-Hechler theory, J2 through J6,
//!   for near-circular orbits, in the circular parameters `a`,
//!   `ex = e cos(w)`, `ey = e sin(w)`, `i`, `raan` and the mean argument of
//!   latitude `alpha = w + M` ([`CircularElements`]). M. C. Eckstein and
//!   F. Hechler, "A reliable derivation of the perturbations due to any zonal
//!   and tesseral harmonics of the geopotential for nearly-circular satellite
//!   orbits", ESRO Scientific Report SR-13 (1970).
//! - [`BrouwerLyddanePropagator`]: Brouwer's theory, J2 through J5, in the
//!   Keplerian elements ([`KeplerianElements`]), with Lyddane's modification
//!   for small eccentricity and inclination, the critical-inclination
//!   treatment of W. E. Phipps Jr. (Naval Postgraduate School thesis, 1992),
//!   and Phipps' optional secular along-track term `M2`. D. Brouwer, "Solution
//!   of the problem of artificial satellite theory without drag", Astronomical
//!   Journal 64 (1959) 378-397; R. H. Lyddane, "Small eccentricities or
//!   inclinations in the Brouwer theory of the artificial satellite",
//!   Astronomical Journal 68 (1963) 555-558.
//!
//! Both theories also convert between mean and osculating elements:
//! mean-to-osculating is the theory evaluated at the elements' own epoch, and
//! osculating-to-mean ([`eckstein_hechler_mean_elements`],
//! [`brouwer_lyddane_mean_elements`]) is the fixed-point iteration both
//! references use, with the tolerance and iteration limit given explicitly in
//! [`MeanConversionOptions`].
//!
//! ## Conventions
//!
//! Positions are km, velocities km/s, angles radians and epochs TDB seconds
//! from J2000, as elsewhere in [`crate::astro::propagator`]. The theories are
//! frame-agnostic: the zonal field is symmetric about the +Z axis of whatever
//! inertial frame the initial state is given in, and the output states are in
//! that frame. Propagated time is the difference of the two epochs.
//!
//! The gravity field is passed explicitly as [`ZonalHarmonics`]: the
//! gravitational parameter, the reference radius and the unnormalized zonal
//! coefficients `C20` through `C60` (`C_n0 = -J_n`).
//!
//! Element sets may carry any finite inclination; one outside `[0, pi]` means
//! the orbit `to_cartesian` gives for it. Both propagators, their
//! constructors and both mean-to-osculating conversions first reduce such an
//! inclination modulo `2 pi` into `[-pi, pi)` and replace a negative one `i`
//! by `-i`, with `w -> w + pi` and `raan -> raan + pi` (in the circular
//! parameters `(ex, ey) -> (-ex, -ey)` and `alpha -> alpha + pi`), which is
//! the same orbit: the Cartesian state is unchanged to rounding. Elements
//! already in `[0, pi]` are used as given. Every element set returned,
//! including the propagators' mean elements, has its inclination in
//! `[0, pi]`.
//!
//! The Eckstein-Hechler output velocity is the time derivative of the
//! analytic position, as in Orekit, so position and velocity are consistent
//! along the analytic trajectory. The Brouwer-Lyddane output state is the
//! two-body state of the osculating Keplerian elements, as in Orekit.
//!
//! ## Validity
//!
//! Eckstein-Hechler refuses mean elements with eccentricity above 0.1
//! ([`AnalyticPropagationError::EccentricityTooLarge`]; accuracy already
//! degrades above about 0.005), with `sin^2 i < 1e-10`
//! ([`AnalyticPropagationError::NearlyEquatorial`]), and with
//! `|sin^2 i - 4/5| < 1e-3`, within about 0.07 degrees of the critical
//! inclinations 63.43 and 116.57 degrees
//! ([`AnalyticPropagationError::NearCriticalInclination`]).
//!
//! Brouwer-Lyddane accepts any elliptic mean orbit, including circular,
//! equatorial (prograde or retrograde) and critically inclined ones.
//! Lyddane's variables are regular at `i = 0` only, so retrograde mean
//! orbits are evaluated through the time-reversal symmetry of the zonal
//! problem as prograde ones, which keeps the theory regular through
//! `i = pi`.
//! It carries only the J2 short-period terms, so in LEO its along-track error
//! grows by one to three kilometres per day against an integration of the
//! same zonal field, where Eckstein-Hechler stays within tens of metres for
//! eccentricities near 1e-3.
//!
//! Both theories expand in powers of the reference radius over the
//! semi-major axis and refuse mean elements, and both osculating-to-mean
//! conversions an osculating state, whose semi-major axis is below the
//! reference radius ([`AnalyticPropagationError::InsideReferenceSphere`]).
//! The conversions also refuse orbits within 1e-10 rad of `i = pi`
//! ([`AnalyticPropagationError::RetrogradeEquatorial`]), where the
//! equinoctial elements they iterate in are undefined.
//!
//! ## Oracle
//!
//! The propagated states and both conversions are checked against Orekit 13.1
//! (`EcksteinHechlerPropagator`, `BrouwerLyddanePropagator` and
//! `FixedPointConverter` with their mean theories) for LEO, MEO, GNSS, IGSO,
//! GEO, GTO and Molniya-like orbits over seven days, with the same gravity
//! coefficients on both sides; see `tests/analytic_propagators_orekit.rs` and
//! the generator in `fixtures-generators/orekit_analytic_oracle`. Both
//! theories are also checked against numerical integration of the same zonal
//! field.
//!
//! The Brouwer-Lyddane terms follow Brouwer (1959) where Orekit 13.1 departs
//! from it in three places: the short-period eccentricity term, the `sin f`
//! part of the short-period `e dl` term, and the long-period inclination. The
//! published terms are tested directly, and against numerical integration,
//! where Orekit's are kilometres off within one revolution and make its
//! osculating-to-mean iteration fail for some near-circular LEO orbits that
//! converge here in a few iterations. The comparison with Orekit replays its
//! arrangement of those terms and agrees to rounding.

mod brouwer_lyddane;
mod dual;
mod eckstein_hechler;
mod elements;
mod fixed_point;

#[cfg(feature = "test-replays")]
#[doc(hidden)]
pub use brouwer_lyddane::orekit_replay as brouwer_lyddane_orekit_replay;
pub use brouwer_lyddane::{
    brouwer_lyddane_mean_elements, brouwer_lyddane_osculating_elements, BrouwerLyddanePropagator,
};
pub use eckstein_hechler::{
    eckstein_hechler_mean_elements, eckstein_hechler_osculating_elements, EcksteinHechlerPropagator,
};
pub use elements::{CircularElements, KeplerianElements};

use crate::astro::constants::{
    J2_EARTH, J3_EARTH, J4_EARTH, J5_EARTH, J6_EARTH, MU_EARTH, RE_EARTH,
};
use crate::astro::error::PropagationError;

/// The analytic theory an error or result refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalyticTheory {
    /// The Eckstein-Hechler theory.
    EcksteinHechler,
    /// The Brouwer-Lyddane theory.
    BrouwerLyddane,
}

impl core::fmt::Display for AnalyticTheory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::EcksteinHechler => "Eckstein-Hechler",
            Self::BrouwerLyddane => "Brouwer-Lyddane",
        })
    }
}

/// Failure returned by the analytic propagators and element conversions.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum AnalyticPropagationError {
    /// An input was NaN or infinite.
    #[error("non-finite input {field}")]
    NonFinite {
        /// The offending input.
        field: &'static str,
    },
    /// A finite input was outside the values the model accepts.
    #[error("invalid {field}: {reason}")]
    InvalidParameter {
        /// The offending input.
        field: &'static str,
        /// What the input must satisfy.
        reason: &'static str,
    },
    /// The state or elements do not describe an elliptic orbit (the
    /// semi-major axis is not positive, or the eccentricity is not below 1).
    #[error("{field} is not an elliptic orbit")]
    NotElliptic {
        /// Which state or element set.
        field: &'static str,
    },
    /// The orbit is equatorial and retrograde (inclination within 1e-10 rad of
    /// pi), where the equinoctial elements used by the conversions are
    /// undefined.
    #[error("retrograde equatorial orbit (i = {inclination_rad} rad)")]
    RetrogradeEquatorial {
        /// Inclination, radians.
        inclination_rad: f64,
    },
    /// The semi-major axis (osculating for a conversion, mean for the
    /// theories) is below the reference radius of the gravity field.
    #[error("semi-major axis {semi_major_axis_km} km is inside the reference radius {reference_radius_km} km")]
    InsideReferenceSphere {
        /// Semi-major axis, km.
        semi_major_axis_km: f64,
        /// Reference radius, km.
        reference_radius_km: f64,
    },
    /// Eckstein-Hechler: the mean eccentricity is above its limit.
    #[error(
        "eccentricity {eccentricity} is above the {limit} limit of the Eckstein-Hechler theory"
    )]
    EccentricityTooLarge {
        /// Mean eccentricity.
        eccentricity: f64,
        /// The model's limit.
        limit: f64,
    },
    /// Eckstein-Hechler: the mean orbit is (nearly) equatorial,
    /// `sin^2 i < 1e-10`.
    #[error("nearly equatorial orbit (i = {inclination_rad} rad) for the Eckstein-Hechler theory")]
    NearlyEquatorial {
        /// Mean inclination, radians.
        inclination_rad: f64,
    },
    /// Eckstein-Hechler: the mean inclination is within the
    /// `|sin^2 i - 4/5| < 1e-3` band around a critical inclination.
    #[error(
        "nearly critical inclination (i = {inclination_rad} rad) for the Eckstein-Hechler theory"
    )]
    NearCriticalInclination {
        /// Mean inclination, radians.
        inclination_rad: f64,
    },
    /// The osculating-to-mean fixed point did not meet the tolerance within the
    /// iteration limit.
    #[error("{theory} mean elements not converged after {iterations} iterations")]
    MeanElementsNotConverged {
        /// Theory whose mean elements were sought.
        theory: AnalyticTheory,
        /// Iterations performed.
        iterations: u32,
    },
    /// A Kepler-equation solve did not converge.
    #[error("Kepler equation not converged for {field}")]
    KeplerNotConverged {
        /// Which anomaly conversion.
        field: &'static str,
    },
    /// The theory produced a non-finite value, which happens only when its
    /// evaluation overflows, for spans far beyond its accuracy.
    #[error("non-finite {field} from the {theory} theory")]
    NonFiniteResult {
        /// Theory that produced it.
        theory: AnalyticTheory,
        /// The quantity.
        field: &'static str,
    },
}

impl From<AnalyticPropagationError> for PropagationError {
    fn from(error: AnalyticPropagationError) -> Self {
        match error {
            AnalyticPropagationError::MeanElementsNotConverged { .. }
            | AnalyticPropagationError::KeplerNotConverged { .. }
            | AnalyticPropagationError::NonFiniteResult { .. } => {
                PropagationError::NumericalFailure(error.to_string())
            }
            _ => PropagationError::InvalidInput(error.to_string()),
        }
    }
}

/// Zonal gravity field for the analytic theories.
///
/// The coefficients are unnormalized, `C_n0 = -J_n`. Brouwer-Lyddane reads
/// `C20` through `C50`; Eckstein-Hechler reads `C20` through `C60`. Both
/// theories are expansions in J2 and divide by it, so `C20` must be nonzero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZonalHarmonics {
    /// Gravitational parameter, km^3/s^2.
    pub mu_km3_s2: f64,
    /// Reference radius of the coefficients, km.
    pub re_km: f64,
    /// Unnormalized C20.
    pub c20: f64,
    /// Unnormalized C30.
    pub c30: f64,
    /// Unnormalized C40.
    pub c40: f64,
    /// Unnormalized C50.
    pub c50: f64,
    /// Unnormalized C60.
    pub c60: f64,
}

impl ZonalHarmonics {
    /// The crate's Earth field: [`MU_EARTH`], [`RE_EARTH`] and `C_n0 = -J_n`
    /// for the [`J2_EARTH`] through [`J6_EARTH`] constants (EGM96).
    pub fn earth() -> Self {
        Self {
            mu_km3_s2: MU_EARTH,
            re_km: RE_EARTH,
            c20: -J2_EARTH,
            c30: -J3_EARTH,
            c40: -J4_EARTH,
            c50: -J5_EARTH,
            c60: -J6_EARTH,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), AnalyticPropagationError> {
        for (value, field) in [
            (self.mu_km3_s2, "mu_km3_s2"),
            (self.re_km, "re_km"),
            (self.c20, "c20"),
            (self.c30, "c30"),
            (self.c40, "c40"),
            (self.c50, "c50"),
            (self.c60, "c60"),
        ] {
            finite(value, field)?;
        }
        if self.mu_km3_s2 <= 0.0 {
            return Err(AnalyticPropagationError::InvalidParameter {
                field: "mu_km3_s2",
                reason: "must be positive",
            });
        }
        if self.re_km <= 0.0 {
            return Err(AnalyticPropagationError::InvalidParameter {
                field: "re_km",
                reason: "must be positive",
            });
        }
        if self.c20 == 0.0 {
            return Err(AnalyticPropagationError::InvalidParameter {
                field: "c20",
                reason: "must be nonzero: both theories expand in J2",
            });
        }
        Ok(())
    }
}

impl Default for ZonalHarmonics {
    fn default() -> Self {
        Self::earth()
    }
}

/// Tolerance and iteration limit of the osculating-to-mean fixed point.
///
/// Each iteration maps the current mean elements to osculating ones and
/// corrects the mean elements by the residual against the given osculating
/// elements, in equinoctial form (`a`, `ex`, `ey`, `hx`, `hy`, true longitude).
/// The iteration stops when every residual is below its threshold:
/// `tolerance * a` for the semi-major axis, `tolerance * (1 + e)` for the two
/// eccentricity components, `tolerance * (1 + tan(i/2))` for the two
/// inclination components and `tolerance * pi` for the longitude, all taken
/// from the osculating elements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanConversionOptions {
    /// Relative convergence threshold; finite and positive.
    pub tolerance: f64,
    /// Largest number of iterations; at least 1.
    pub max_iterations: u32,
}

impl MeanConversionOptions {
    /// Orekit's default for Eckstein-Hechler: tolerance 1e-11, 100 iterations.
    pub const ECKSTEIN_HECHLER_DEFAULT: Self = Self {
        tolerance: 1.0e-11,
        max_iterations: 100,
    };

    /// Orekit's default for Brouwer-Lyddane: tolerance 1e-13, 200 iterations.
    pub const BROUWER_LYDDANE_DEFAULT: Self = Self {
        tolerance: 1.0e-13,
        max_iterations: 200,
    };

    /// Options with the given tolerance and iteration limit.
    pub fn new(tolerance: f64, max_iterations: u32) -> Self {
        Self {
            tolerance,
            max_iterations,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), AnalyticPropagationError> {
        finite(self.tolerance, "tolerance")?;
        if self.tolerance <= 0.0 {
            return Err(AnalyticPropagationError::InvalidParameter {
                field: "tolerance",
                reason: "must be positive",
            });
        }
        if self.max_iterations == 0 {
            return Err(AnalyticPropagationError::InvalidParameter {
                field: "max_iterations",
                reason: "must be at least 1",
            });
        }
        Ok(())
    }
}

/// Mean elements found by an osculating-to-mean conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanElements<E> {
    /// The mean elements.
    pub elements: E,
    /// Fixed-point iterations performed, counting the one that met the
    /// tolerance.
    pub iterations: u32,
}

pub(crate) fn finite(value: f64, field: &'static str) -> Result<f64, AnalyticPropagationError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AnalyticPropagationError::NonFinite { field })
    }
}

pub(crate) fn finite_result(
    value: f64,
    theory: AnalyticTheory,
    field: &'static str,
) -> Result<f64, AnalyticPropagationError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AnalyticPropagationError::NonFiniteResult { theory, field })
    }
}

/// `a` wrapped into `[center - pi, center + pi)`.
///
/// An angle already in range is returned unchanged. Otherwise the reduction
/// goes through `atan2(sin a, cos a)`, whose argument reduction is exact, so
/// the result is within a few units in the last place of the true remainder
/// for any finite `a`; subtracting a rounded multiple of `2 pi` instead loses
/// `|a| * 4e-17` rad, a whole radian near `1e16`. Non-finite `a` gives NaN.
pub(crate) fn normalize_angle(a: f64, center: f64) -> f64 {
    use core::f64::consts::PI;
    let (low, high) = (center - PI, center + PI);
    if a >= low && a < high {
        return a;
    }
    if !a.is_finite() {
        return f64::NAN;
    }
    let two_pi = 2.0 * PI;
    let mut r = libm::atan2(libm::sin(a), libm::cos(a));
    r -= two_pi * libm::floor((r - low) / two_pi);
    if r >= high {
        r -= two_pi;
    }
    r.max(low)
}

#[cfg(test)]
mod tests;
