//! Osculating-to-mean conversion by fixed-point iteration.
//!
//! The mean elements `m` that a theory maps to given osculating elements `o`
//! solve `T(m) = o`. Starting from `m_0 = o`, each step evaluates the theory's
//! mean-to-osculating map and corrects the mean elements by the residual,
//! `m_{k+1} = m_k + (o - T(m_k))`, in equinoctial elements so the iteration is
//! regular for circular and equatorial orbits. This is the iteration of
//! Orekit's `FixedPointConverter` with unit damping, used there for both
//! theories; it converges when the periodic terms change slowly with the mean
//! elements, which holds wherever the theory itself is regular.

use super::elements::Equinoctial;
use super::{normalize_angle, AnalyticPropagationError, AnalyticTheory, MeanConversionOptions};

/// A theory's mean-to-osculating map at the elements' own epoch.
pub(crate) trait MeanTheory {
    fn theory(&self) -> AnalyticTheory;
    fn reference_radius_km(&self) -> f64;
    fn mean_to_osculating(
        &self,
        mean: &Equinoctial,
    ) -> Result<Equinoctial, AnalyticPropagationError>;
}

/// Mean equinoctial elements whose osculating image is `osculating`, and the
/// number of iterations performed.
pub(crate) fn osculating_to_mean<T: MeanTheory>(
    theory: &T,
    osculating: &Equinoctial,
    options: &MeanConversionOptions,
) -> Result<(Equinoctial, u32), AnalyticPropagationError> {
    options.validate()?;
    if osculating.a < theory.reference_radius_km() {
        return Err(AnalyticPropagationError::InsideReferenceSphere {
            semi_major_axis_km: osculating.a,
            reference_radius_km: theory.reference_radius_km(),
        });
    }
    let eps = options.tolerance;
    let threshold_a = eps * osculating.a.abs();
    let threshold_e = eps * (1.0 + libm::hypot(osculating.ex, osculating.ey));
    let threshold_h = eps * (1.0 + libm::hypot(osculating.hx, osculating.hy));
    let threshold_lv = eps * core::f64::consts::PI;

    let mut mean = *osculating;
    for iteration in 1..=options.max_iterations {
        let updated = theory.mean_to_osculating(&mean)?;
        let delta_a = osculating.a - updated.a;
        let delta_ex = osculating.ex - updated.ex;
        let delta_ey = osculating.ey - updated.ey;
        let delta_hx = osculating.hx - updated.hx;
        let delta_hy = osculating.hy - updated.hy;
        let delta_lv = normalize_angle(osculating.lv - updated.lv, 0.0);
        if delta_a.abs() < threshold_a
            && delta_ex.abs() < threshold_e
            && delta_ey.abs() < threshold_e
            && delta_hx.abs() < threshold_h
            && delta_hy.abs() < threshold_h
            && delta_lv.abs() < threshold_lv
        {
            return Ok((mean, iteration));
        }
        if ![delta_a, delta_ex, delta_ey, delta_hx, delta_hy, delta_lv]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(AnalyticPropagationError::NonFiniteResult {
                theory: theory.theory(),
                field: "osculating elements",
            });
        }
        mean.a += delta_a;
        mean.ex += delta_ex;
        mean.ey += delta_ey;
        mean.hx += delta_hx;
        mean.hy += delta_hy;
        mean.lv += delta_lv;
    }
    Err(AnalyticPropagationError::MeanElementsNotConverged {
        theory: theory.theory(),
        iterations: options.max_iterations,
    })
}
