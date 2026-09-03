//! Inverse SGP4 fitting from a TEME state arc.
//!
//! The fitter turns target TEME samples into SGP4 mean elements by minimizing
//! position and optional velocity residuals with the workspace trust-region
//! least-squares solver. The optimization variables are mean equinoctial
//! coordinates: mean motion, the eccentricity vector projected through longitude
//! of perigee, the inclination vector projected through RAAN, mean longitude,
//! and optionally scaled B*. This avoids the circular-orbit and equatorial-orbit
//! singularities of direct TLE angles while still mapping algebraically back to
//! the SGP4 `ElementSet`.
//!
//! The initial point comes from an osculating `rv2coe` state at the selected fit
//! epoch, followed by two guarded fixed-point passes through SGP4 at zero
//! elapsed time. Infeasible trial points do not throw from the residual callback:
//! they fill the fixed residual length with a large finite penalty so the trust
//! region can reject the step and continue. B* is scaled by a typical LEO
//! magnitude before optimization so finite differences can see the drag column.
//!
//! The returned elements are full precision. The TLE lines are fixed-width and
//! therefore quantized; `tle_rms_position_km` reports the downstream effect of
//! re-parsing those lines. The OMM carries the same full-precision fields as
//! plain data. Epochs are treated as UTC split Julian dates and propagated with
//! the same split-JD subtraction used by operational SGP4 consumers.

use std::cell::Cell;

use thiserror::Error;
pub use trust_region_least_squares::loss::Loss;
use trust_region_least_squares::model::{solve_model_with, ResidualModel};
pub use trust_region_least_squares::trf::XScale;
use trust_region_least_squares::trf::{TrfError, TrfOptions, TrfResult};

use crate::astro::anomaly::true_to_mean;
use crate::astro::elements::{rv2coe, ClassicalElements, OrbitType};
use crate::astro::math::portable::PortableNumerics;
use crate::astro::math::vec3;
use crate::astro::omm::Omm;
use crate::astro::sgp4::{ElementSet, Error as Sgp4Error, JulianDate, OpsMode, Satellite};
use crate::astro::time::civil::{civil_from_split_julian_date, day_of_year};
use crate::astro::tle::{self, TleElements};

use super::MAX_MINUTES_SINCE_EPOCH;

const BSTAR_SCALE: f64 = 1.0e-4;
const ECC_MAX: f64 = 0.999;
const PENALTY_KM: f64 = 1.0e6;
const SEED_REFINE_PASSES: usize = 2;
/// Observability floor for the fitted B\* Jacobian column, as a fraction of
/// the largest column norm. This is a diagnostic threshold feeding
/// [`FitStatistics::bstar_observable`], not a solver parameter: it never
/// changes the fit, only whether the recovered B\* is reported as trustworthy.
/// 1.0e-5 is calibrated on the regression arcs: at that column-norm ratio and
/// above, B\* recovery is demonstrably reliable (the high-drag decay arc
/// recovers B\* to within 1e-6 of truth), while arcs whose ratio falls below
/// it (the short GEO arc) carry no usable drag signal.
const BSTAR_OBSERVABLE_REL: f64 = 1.0e-5;
const MU_WGS72_KM3_S2: f64 = 398_600.8;
const SECONDS_PER_DAY: f64 = 86_400.0;
const TAU: f64 = std::f64::consts::TAU;

/// One target state on the arc: UTC epoch, TEME position, and optional TEME
/// velocity.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FitSample {
    pub epoch: JulianDate,
    pub position_teme_km: [f64; 3],
    pub velocity_teme_km_s: Option<[f64; 3]>,
}

/// Which sample epoch becomes the TLE epoch.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum FitEpoch {
    /// The sample nearest the arc midpoint.
    #[default]
    Midpoint,
    First,
    Last,
    /// A specific sample index.
    Sample(usize),
    /// An explicit epoch inside the sample arc.
    Jd(JulianDate),
}

/// Pass-through TLE/OMM bookkeeping.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TleMetadata {
    pub catalog_number: u32,
    pub classification: String,
    pub international_designator: String,
    pub element_set_number: i32,
    pub rev_at_epoch: i64,
    pub object_name: String,
}

impl Default for TleMetadata {
    fn default() -> Self {
        Self {
            catalog_number: 0,
            classification: "U".to_string(),
            international_designator: String::new(),
            element_set_number: 999,
            rev_at_epoch: 0,
            object_name: String::new(),
        }
    }
}

/// Configuration for [`fit_tle`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FitConfig {
    pub epoch: FitEpoch,
    pub fit_bstar: bool,
    pub bstar_seed: f64,
    pub use_velocity: bool,
    pub velocity_weight_s: Option<f64>,
    pub weights: Option<Vec<f64>>,
    pub opsmode: OpsMode,
    pub ftol: Option<f64>,
    pub xtol: Option<f64>,
    pub gtol: Option<f64>,
    pub max_nfev: Option<usize>,
    pub x_scale: Option<XScale>,
    pub loss: Loss,
    pub f_scale: f64,
    pub metadata: TleMetadata,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            epoch: FitEpoch::default(),
            fit_bstar: true,
            bstar_seed: 0.0,
            use_velocity: true,
            velocity_weight_s: None,
            weights: None,
            opsmode: OpsMode::Improved,
            ftol: None,
            xtol: None,
            gtol: None,
            max_nfev: None,
            x_scale: None,
            loss: Loss::Linear,
            f_scale: 1.0,
            metadata: TleMetadata::default(),
        }
    }
}

/// Fit diagnostics computed from the returned elements and encoded TLE.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FitStatistics {
    pub rms_position_km: f64,
    pub max_position_km: f64,
    pub rms_position_axes_km: [f64; 3],
    pub rms_velocity_km_s: Option<f64>,
    pub tle_rms_position_km: f64,
    pub status: i32,
    pub nfev: usize,
    pub njev: usize,
    pub cost: f64,
    pub optimality: f64,
    pub bstar_observable: bool,
    pub seed_refine_passes: usize,
}

/// Fitted SGP4 elements plus compatibility encodings.
///
/// Cross-format epoch fidelity: `elements.epoch` is the exact split JD the
/// fit solved at. `omm` carries it twice: as femtosecond-precision `EPOCH`
/// calendar text, and through the in-memory `exact_sgp4_epoch` side channel
/// (with `quantize_tle_derived_fields` disabled), so
/// `Satellite::from_omm(&fit.omm)` is bit-identical to
/// `Satellite::from_elements(&fit.elements)`. Encoding the OMM to text and
/// reparsing rebuilds the epoch from the calendar form: the same instant to
/// femtosecond precision, bit-identical when the sample epochs were
/// midnight-anchored splits (see [`crate::astro::omm::Omm::exact_sgp4_epoch`]).
/// The TLE lines are coarser by format: the TLE epoch field holds 8
/// fractional day-of-year digits, a grid of about 0.86 ms, so `line1`/`line2`
/// reproduce the fit epoch only to that resolution (reflected in
/// `stats.tle_rms_position_km`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TleFit {
    pub elements: ElementSet,
    pub line1: String,
    pub line2: String,
    pub omm: Omm,
    pub stats: FitStatistics,
}

/// Fitting failures.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum TleFitError {
    #[error("fit arc has {samples} samples; need at least {needed}")]
    ArcTooShort { samples: usize, needed: usize },
    #[error("fit input invalid: {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    #[error("sample epochs must be strictly increasing (violation at index {index})")]
    EpochsNotIncreasing { index: usize },
    #[error("requested epoch lies outside the sample arc")]
    EpochOutsideArc,
    #[error("velocities must be present on every sample or on none")]
    MixedVelocityPresence,
    #[error("target arc is not elliptical (ecc >= 1 at the seed epoch)")]
    NotElliptical,
    #[error("seed inclination {inclination_deg} deg is too close to retrograde-equatorial")]
    InclinationNearRetrograde { inclination_deg: f64 },
    #[error("seed elements cannot propagate the arc at sample {epoch_index}: {source}")]
    SeedPropagation {
        epoch_index: usize,
        source: Sgp4Error,
    },
    #[error("solver error: {0}")]
    Solver(TrfError),
    #[error("solver stopped at an infeasible point")]
    SolutionInfeasible,
    #[error("evaluation budget exhausted before convergence (best-effort result attached)")]
    DidNotConverge { result: Box<TleFit> },
    #[error("fitted elements failed final validation: {0}")]
    FinalElements(Sgp4Error),
    /// The final fitted TLE elements could not be encoded as TLE text.
    #[error("fitted TLE failed encoding: {0}")]
    TleEncode(tle::TleError),
}

/// Fit SGP4 mean elements and optional B* to a TEME ephemeris arc.
pub fn fit_tle(samples: &[FitSample], config: &FitConfig) -> Result<TleFit, TleFitError> {
    let resolved = validate_and_resolve(samples, config)?;
    let (x0, seed_refine_passes) = initial_guess(samples, config, &resolved)?;
    let seed_elements = chart_to_elements(
        &x0,
        resolved.epoch,
        config.fit_bstar,
        config.bstar_seed,
        config.metadata.catalog_number,
    )
    .ok_or(TleFitError::NotElliptical)?;
    validate_seed_propagates(samples, &seed_elements, config.opsmode)?;

    let w_vel = config.velocity_weight_s.unwrap_or_else(|| {
        let n_rad_s = x0[0] * TAU / SECONDS_PER_DAY;
        1.0 / n_rad_s
    });

    let problem = Sgp4FitProblem {
        samples,
        epoch: resolved.epoch,
        opsmode: config.opsmode,
        fit_bstar: config.fit_bstar,
        fixed_bstar: config.bstar_seed,
        catalog_number: config.metadata.catalog_number,
        weights: resolved.weights,
        w_vel,
        use_velocity: resolved.use_velocity,
        penalty_hit_at_solution: Cell::new(false),
    };

    let options = solver_options(config);
    let result = solve_model_with(&problem, &x0, &PortableNumerics, &options)
        .map_err(TleFitError::Solver)?;

    let mut final_rows = Vec::new();
    problem.penalty_hit_at_solution.set(false);
    problem.residual(&result.x, &mut final_rows);
    if problem.penalty_hit_at_solution.get() {
        return Err(TleFitError::SolutionInfeasible);
    }

    let fit = build_fit(&problem, config, result, seed_refine_passes)?;
    if fit.stats.status == 0 {
        Err(TleFitError::DidNotConverge {
            result: Box::new(fit),
        })
    } else {
        Ok(fit)
    }
}

struct ResolvedFit {
    epoch: JulianDate,
    epoch_index: usize,
    use_velocity: bool,
    weights: Vec<f64>,
}

struct Sgp4FitProblem<'a> {
    samples: &'a [FitSample],
    epoch: JulianDate,
    opsmode: OpsMode,
    fit_bstar: bool,
    fixed_bstar: f64,
    catalog_number: u32,
    weights: Vec<f64>,
    w_vel: f64,
    use_velocity: bool,
    penalty_hit_at_solution: Cell<bool>,
}

impl Sgp4FitProblem<'_> {
    fn residual_len(&self) -> usize {
        self.samples.len() * 3 * if self.use_velocity { 2 } else { 1 }
    }

    fn penalty(&self, out: &mut Vec<f64>) {
        self.penalty_hit_at_solution.set(true);
        out.clear();
        out.resize(self.residual_len(), PENALTY_KM);
    }
}

impl ResidualModel for Sgp4FitProblem<'_> {
    fn residual(&self, x: &[f64], out: &mut Vec<f64>) {
        let Some(elements) = chart_to_elements(
            x,
            self.epoch,
            self.fit_bstar,
            self.fixed_bstar,
            self.catalog_number,
        ) else {
            self.penalty(out);
            return;
        };

        let Ok(satellite) = Satellite::from_elements_with_opsmode(&elements, self.opsmode) else {
            self.penalty(out);
            return;
        };

        out.clear();
        for (sample, &weight) in self.samples.iter().zip(&self.weights) {
            let Ok(prediction) = satellite.propagate_jd(sample.epoch) else {
                self.penalty(out);
                return;
            };
            for axis in 0..3 {
                out.push(weight * (prediction.position[axis] - sample.position_teme_km[axis]));
            }
            if self.use_velocity {
                let velocity = sample
                    .velocity_teme_km_s
                    .expect("validated velocity presence");
                for (axis, target) in velocity.iter().enumerate() {
                    out.push(weight * self.w_vel * (prediction.velocity[axis] - *target));
                }
            }
        }
        self.penalty_hit_at_solution.set(false);
    }
}

#[derive(Debug, Clone, Copy)]
struct MeanChart {
    n_rev_day: f64,
    af: f64,
    ag: f64,
    chi: f64,
    psi: f64,
    lam: f64,
}

impl MeanChart {
    fn to_vec(self, fit_bstar: bool, bstar: f64) -> Vec<f64> {
        let mut x = vec![
            self.n_rev_day,
            self.af,
            self.ag,
            self.chi,
            self.psi,
            self.lam,
        ];
        if fit_bstar {
            x.push(bstar / BSTAR_SCALE);
        }
        x
    }

    #[cfg(test)]
    fn from_elements(elements: &ElementSet) -> Self {
        let ecc = elements.eccentricity;
        let argp = deg_to_rad(elements.argument_of_perigee_deg);
        let raan = deg_to_rad(elements.right_ascension_deg);
        let mean = deg_to_rad(elements.mean_anomaly_deg);
        let incl = deg_to_rad(elements.inclination_deg);
        let varpi = argp + raan;
        let half_tan = libm::tan(incl * 0.5);
        Self {
            n_rev_day: elements.mean_motion_rev_per_day,
            af: ecc * libm::cos(varpi),
            ag: ecc * libm::sin(varpi),
            chi: half_tan * libm::sin(raan),
            psi: half_tan * libm::cos(raan),
            lam: normalize_angle(mean + varpi),
        }
    }
}

fn validate_and_resolve(
    samples: &[FitSample],
    config: &FitConfig,
) -> Result<ResolvedFit, TleFitError> {
    if samples.len() < 3 {
        return Err(TleFitError::ArcTooShort {
            samples: samples.len(),
            needed: 3,
        });
    }

    for (i, sample) in samples.iter().enumerate() {
        validate_epoch(sample.epoch)?;
        if i > 0 && seconds_between(sample.epoch, samples[i - 1].epoch) <= 0.0 {
            return Err(TleFitError::EpochsNotIncreasing { index: i });
        }
        validate_vec3("position_teme_km", &sample.position_teme_km)?;
        if vec3::norm3(sample.position_teme_km) <= 0.0 {
            return Err(TleFitError::InvalidInput {
                field: "position_teme_km",
                reason: "magnitude must be positive",
            });
        }
        if let Some(velocity) = &sample.velocity_teme_km_s {
            validate_vec3("velocity_teme_km_s", velocity)?;
        }
    }

    let velocity_count = samples
        .iter()
        .filter(|sample| sample.velocity_teme_km_s.is_some())
        .count();
    let use_velocity = if config.use_velocity {
        if velocity_count == 0 {
            false
        } else if velocity_count == samples.len() {
            true
        } else {
            return Err(TleFitError::MixedVelocityPresence);
        }
    } else {
        false
    };

    let n_params = 6 + usize::from(config.fit_bstar);
    let rows_per_sample = 3 * if use_velocity { 2 } else { 1 };
    if samples.len() * rows_per_sample < n_params {
        let needed = n_params.div_ceil(rows_per_sample).max(3);
        return Err(TleFitError::ArcTooShort {
            samples: samples.len(),
            needed,
        });
    }

    let weights = if let Some(weights) = &config.weights {
        if weights.len() != samples.len() {
            return Err(TleFitError::InvalidInput {
                field: "weights",
                reason: "length must match samples",
            });
        }
        for &weight in weights {
            validate_positive("weights", weight)?;
        }
        weights.clone()
    } else {
        vec![1.0; samples.len()]
    };

    validate_optional_positive("velocity_weight_s", config.velocity_weight_s)?;
    validate_finite("bstar_seed", config.bstar_seed)?;
    validate_optional_positive("ftol", config.ftol)?;
    validate_optional_positive("xtol", config.xtol)?;
    validate_optional_positive("gtol", config.gtol)?;
    validate_positive("f_scale", config.f_scale)?;
    if config.max_nfev == Some(0) {
        return Err(TleFitError::InvalidInput {
            field: "max_nfev",
            reason: "must be positive",
        });
    }
    if config.metadata.catalog_number > 99_999 {
        return Err(TleFitError::InvalidInput {
            field: "metadata.catalog_number",
            reason: "must be <= 99999",
        });
    }
    if !matches!(config.metadata.classification.as_str(), "U" | "C" | "S") {
        return Err(TleFitError::InvalidInput {
            field: "metadata.classification",
            reason: "must be U, C, or S",
        });
    }
    if i32::try_from(config.metadata.rev_at_epoch).is_err() {
        return Err(TleFitError::InvalidInput {
            field: "metadata.rev_at_epoch",
            reason: "must fit i32",
        });
    }

    if let Some(XScale::Values(values)) = &config.x_scale {
        if values.len() != n_params {
            return Err(TleFitError::InvalidInput {
                field: "x_scale",
                reason: "length must match parameter count",
            });
        }
        for &value in values {
            validate_positive("x_scale", value)?;
        }
    }

    let (epoch, epoch_index) = resolve_epoch(samples, config.epoch)?;
    for sample in samples {
        let minutes = seconds_between(sample.epoch, epoch).abs() / 60.0;
        if minutes > MAX_MINUTES_SINCE_EPOCH {
            return Err(TleFitError::InvalidInput {
                field: "arc_span",
                reason: "outside SGP4 time domain",
            });
        }
    }

    Ok(ResolvedFit {
        epoch,
        epoch_index,
        use_velocity,
        weights,
    })
}

fn initial_guess(
    samples: &[FitSample],
    config: &FitConfig,
    resolved: &ResolvedFit,
) -> Result<(Vec<f64>, usize), TleFitError> {
    let sample = samples[resolved.epoch_index];
    let velocity = sample
        .velocity_teme_km_s
        .unwrap_or_else(|| finite_difference_velocity(samples, resolved.epoch_index));
    let mut target = chart_from_state(sample.position_teme_km, velocity)?;
    let dt = seconds_between(resolved.epoch, sample.epoch);
    target.lam = normalize_angle(target.lam + target.n_rev_day * TAU / SECONDS_PER_DAY * dt);

    if rad_to_deg(2.0 * libm::atan(libm::hypot(target.chi, target.psi))) >= 179.5 {
        return Err(TleFitError::InclinationNearRetrograde {
            inclination_deg: rad_to_deg(2.0 * libm::atan(libm::hypot(target.chi, target.psi))),
        });
    }

    let x0 = target.to_vec(config.fit_bstar, config.bstar_seed);
    let (x, passes) = refine_seed(
        x0,
        target,
        resolved.epoch,
        config.opsmode,
        config.fit_bstar,
        config.bstar_seed,
        config.metadata.catalog_number,
    );
    Ok((x, passes))
}

fn refine_seed(
    mut x: Vec<f64>,
    target: MeanChart,
    epoch: JulianDate,
    opsmode: OpsMode,
    fit_bstar: bool,
    fixed_bstar: f64,
    catalog_number: u32,
) -> (Vec<f64>, usize) {
    let mut passes = 0;
    let mut previous_norm = f64::INFINITY;
    for _ in 0..SEED_REFINE_PASSES {
        let Some(elements) = chart_to_elements(&x, epoch, fit_bstar, fixed_bstar, catalog_number)
        else {
            break;
        };
        let Ok(satellite) = Satellite::from_elements_with_opsmode(&elements, opsmode) else {
            break;
        };
        let Ok(prediction) = satellite.propagate_jd(epoch) else {
            break;
        };
        let Ok(osc) = chart_from_state(prediction.position, prediction.velocity) else {
            break;
        };

        let correction = [
            target.n_rev_day - osc.n_rev_day,
            target.af - osc.af,
            target.ag - osc.ag,
            target.chi - osc.chi,
            target.psi - osc.psi,
            wrap_to_pi(target.lam - osc.lam),
        ];
        let correction_norm = correction.iter().map(|v| v * v).sum::<f64>().sqrt();
        if correction_norm > previous_norm {
            break;
        }

        let mut candidate = x.clone();
        for i in 0..6 {
            candidate[i] += correction[i];
        }
        if chart_to_elements(&candidate, epoch, fit_bstar, fixed_bstar, catalog_number).is_none() {
            break;
        }

        x = candidate;
        previous_norm = correction_norm;
        passes += 1;
    }
    (x, passes)
}

fn validate_seed_propagates(
    samples: &[FitSample],
    elements: &ElementSet,
    opsmode: OpsMode,
) -> Result<(), TleFitError> {
    let satellite = Satellite::from_elements_with_opsmode(elements, opsmode).map_err(|source| {
        TleFitError::SeedPropagation {
            epoch_index: 0,
            source,
        }
    })?;
    for (i, sample) in samples.iter().enumerate() {
        satellite
            .propagate_jd(sample.epoch)
            .map_err(|source| TleFitError::SeedPropagation {
                epoch_index: i,
                source,
            })?;
    }
    Ok(())
}

fn build_fit(
    problem: &Sgp4FitProblem<'_>,
    config: &FitConfig,
    result: TrfResult,
    seed_refine_passes: usize,
) -> Result<TleFit, TleFitError> {
    let elements = chart_to_elements(
        &result.x,
        problem.epoch,
        problem.fit_bstar,
        problem.fixed_bstar,
        problem.catalog_number,
    )
    .ok_or(TleFitError::SolutionInfeasible)?;
    let satellite = Satellite::from_elements_with_opsmode(&elements, problem.opsmode)
        .map_err(TleFitError::FinalElements)?;

    let tle_elements = tle_elements_from_fit(&elements, &config.metadata)?;
    let (line1, line2) = tle::encode(&tle_elements).map_err(TleFitError::TleEncode)?;
    let tle_satellite = Satellite::from_tle_with_opsmode(&line1, &line2, problem.opsmode)
        .map_err(TleFitError::FinalElements)?;

    let arc_stats = compute_arc_stats(problem, &satellite).map_err(TleFitError::FinalElements)?;
    let tle_rms_position_km =
        compute_tle_rms(problem, &tle_satellite).map_err(TleFitError::FinalElements)?;
    let bstar_observable = bstar_observable(&result, problem.fit_bstar);
    let omm = omm_from_fit(&elements, &config.metadata)?;

    Ok(TleFit {
        elements,
        line1,
        line2,
        omm,
        stats: FitStatistics {
            rms_position_km: arc_stats.rms_position_km,
            max_position_km: arc_stats.max_position_km,
            rms_position_axes_km: arc_stats.rms_position_axes_km,
            rms_velocity_km_s: arc_stats.rms_velocity_km_s,
            tle_rms_position_km,
            status: result.status,
            nfev: result.nfev,
            njev: result.njev,
            cost: result.cost,
            optimality: result.optimality,
            bstar_observable,
            seed_refine_passes,
        },
    })
}

#[derive(Debug, Clone, Copy)]
struct ArcStats {
    rms_position_km: f64,
    max_position_km: f64,
    rms_position_axes_km: [f64; 3],
    rms_velocity_km_s: Option<f64>,
}

fn compute_arc_stats(
    problem: &Sgp4FitProblem<'_>,
    satellite: &Satellite,
) -> Result<ArcStats, Sgp4Error> {
    let mut sum_r2 = 0.0;
    let mut max_r = 0.0;
    let mut axis_sum = [0.0; 3];
    let mut sum_v2 = 0.0;
    for sample in problem.samples {
        let prediction = satellite.propagate_jd(sample.epoch)?;
        let mut r2 = 0.0;
        for (axis, axis_total) in axis_sum.iter_mut().enumerate() {
            let residual = prediction.position[axis] - sample.position_teme_km[axis];
            r2 += residual * residual;
            *axis_total += residual * residual;
        }
        let r = r2.sqrt();
        sum_r2 += r2;
        if r > max_r {
            max_r = r;
        }

        if problem.use_velocity {
            let velocity = sample
                .velocity_teme_km_s
                .expect("validated velocity presence");
            let mut v2 = 0.0;
            for (pred, target) in prediction.velocity.iter().zip(velocity) {
                let residual = pred - target;
                v2 += residual * residual;
            }
            sum_v2 += v2;
        }
    }
    let n = problem.samples.len() as f64;
    Ok(ArcStats {
        rms_position_km: (sum_r2 / n).sqrt(),
        max_position_km: max_r,
        rms_position_axes_km: [
            (axis_sum[0] / n).sqrt(),
            (axis_sum[1] / n).sqrt(),
            (axis_sum[2] / n).sqrt(),
        ],
        rms_velocity_km_s: if problem.use_velocity {
            Some((sum_v2 / n).sqrt())
        } else {
            None
        },
    })
}

fn compute_tle_rms(problem: &Sgp4FitProblem<'_>, satellite: &Satellite) -> Result<f64, Sgp4Error> {
    let mut sum_r2 = 0.0;
    for sample in problem.samples {
        let prediction = satellite.propagate_jd(sample.epoch)?;
        let mut r2 = 0.0;
        for axis in 0..3 {
            let residual = prediction.position[axis] - sample.position_teme_km[axis];
            r2 += residual * residual;
        }
        sum_r2 += r2;
    }
    Ok((sum_r2 / problem.samples.len() as f64).sqrt())
}

fn bstar_observable(result: &TrfResult, fit_bstar: bool) -> bool {
    if !fit_bstar {
        return false;
    }
    let n = result.x.len();
    if n == 0 || result.jac.is_empty() || !result.jac.len().is_multiple_of(n) {
        return false;
    }
    let m = result.jac.len() / n;
    let mut largest = 0.0_f64;
    let mut bstar = 0.0_f64;
    for col in 0..n {
        let mut norm2 = 0.0;
        for row in 0..m {
            let value = result.jac[row * n + col];
            norm2 += value * value;
        }
        let norm = norm2.sqrt();
        if norm > largest {
            largest = norm;
        }
        if col == n - 1 {
            bstar = norm;
        }
    }
    largest > 0.0 && bstar / largest >= BSTAR_OBSERVABLE_REL
}

fn chart_from_state(position: [f64; 3], velocity: [f64; 3]) -> Result<MeanChart, TleFitError> {
    let coe =
        rv2coe(position, velocity, MU_WGS72_KM3_S2).map_err(|_| TleFitError::NotElliptical)?;
    chart_from_coe(&coe)
}

fn chart_from_coe(coe: &ClassicalElements) -> Result<MeanChart, TleFitError> {
    if !coe.ecc.is_finite() || coe.ecc >= 1.0 || coe.ecc < 0.0 {
        return Err(TleFitError::NotElliptical);
    }
    if !coe.a.is_finite() || coe.a <= 0.0 {
        return Err(TleFitError::NotElliptical);
    }

    let (raan, argp, nu) = match coe.orbit_type {
        OrbitType::EllipticalInclined => (coe.raan, coe.argp, coe.nu),
        OrbitType::EllipticalEquatorial => (0.0, coe.lonper, coe.nu),
        OrbitType::CircularInclined => (coe.raan, 0.0, coe.arglat),
        OrbitType::CircularEquatorial => (0.0, 0.0, coe.truelon),
    };
    if !raan.is_finite() || !argp.is_finite() || !nu.is_finite() || !coe.incl.is_finite() {
        return Err(TleFitError::NotElliptical);
    }

    let mean = true_to_mean(nu, coe.ecc).map_err(|_| TleFitError::NotElliptical)?;
    let a3 = coe.a * coe.a * coe.a;
    let n_rad_s = (MU_WGS72_KM3_S2 / a3).sqrt();
    let varpi = argp + raan;
    let half_tan = libm::tan(coe.incl * 0.5);
    Ok(MeanChart {
        n_rev_day: n_rad_s * SECONDS_PER_DAY / TAU,
        af: coe.ecc * libm::cos(varpi),
        ag: coe.ecc * libm::sin(varpi),
        chi: half_tan * libm::sin(raan),
        psi: half_tan * libm::cos(raan),
        lam: normalize_angle(mean + varpi),
    })
}

fn chart_to_elements(
    x: &[f64],
    epoch: JulianDate,
    fit_bstar: bool,
    fixed_bstar: f64,
    catalog_number: u32,
) -> Option<ElementSet> {
    let expected = 6 + usize::from(fit_bstar);
    if x.len() != expected || x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let n_rev_day = x[0];
    let af = x[1];
    let ag = x[2];
    let chi = x[3];
    let psi = x[4];
    let lam = x[5];
    let ecc = libm::hypot(af, ag);
    if n_rev_day <= 0.0 || ecc >= ECC_MAX {
        return None;
    }
    let bstar = if fit_bstar {
        x[6] * BSTAR_SCALE
    } else {
        fixed_bstar
    };
    if !bstar.is_finite() {
        return None;
    }

    let varpi = libm::atan2(ag, af);
    let incl = 2.0 * libm::atan(libm::hypot(chi, psi));
    let raan = libm::atan2(chi, psi);
    let argp = varpi - raan;
    let mean = lam - varpi;

    Some(ElementSet {
        epoch,
        bstar,
        mean_motion_dot: 0.0,
        mean_motion_double_dot: 0.0,
        eccentricity: ecc,
        argument_of_perigee_deg: normalize_degrees(rad_to_deg(argp)),
        inclination_deg: rad_to_deg(incl),
        mean_anomaly_deg: normalize_degrees(rad_to_deg(mean)),
        mean_motion_rev_per_day: n_rev_day,
        right_ascension_deg: normalize_degrees(rad_to_deg(raan)),
        catalog_number,
    })
}

fn finite_difference_velocity(samples: &[FitSample], index: usize) -> [f64; 3] {
    let (lo, hi) = if index == 0 {
        (0, 1)
    } else if index + 1 == samples.len() {
        (samples.len() - 2, samples.len() - 1)
    } else {
        (index - 1, index + 1)
    };
    let dt = seconds_between(samples[hi].epoch, samples[lo].epoch);
    let mut velocity = [0.0; 3];
    for (axis, value) in velocity.iter_mut().enumerate() {
        *value = (samples[hi].position_teme_km[axis] - samples[lo].position_teme_km[axis]) / dt;
    }
    velocity
}

fn tle_elements_from_fit(
    elements: &ElementSet,
    metadata: &TleMetadata,
) -> Result<TleElements, TleFitError> {
    let (year, month, day, hour, minute, second) = civil_from_jd(elements.epoch);
    let rev_number =
        i32::try_from(metadata.rev_at_epoch).map_err(|_| TleFitError::InvalidInput {
            field: "metadata.rev_at_epoch",
            reason: "must fit i32",
        })?;
    Ok(TleElements {
        catalog_number: format!("{:05}", metadata.catalog_number),
        classification: metadata.classification.clone(),
        international_designator: metadata.international_designator.clone(),
        epoch_year: year as i32,
        epoch_day_of_year: day_of_year(
            year as i32,
            month as i32,
            day as i32,
            hour as i32,
            minute as i32,
            second,
        ),
        mean_motion_dot: 0.0,
        mean_motion_double_dot: 0.0,
        bstar: elements.bstar,
        ephemeris_type: 0,
        elset_number: metadata.element_set_number,
        inclination_deg: elements.inclination_deg,
        raan_deg: elements.right_ascension_deg,
        eccentricity: elements.eccentricity,
        arg_perigee_deg: elements.argument_of_perigee_deg,
        mean_anomaly_deg: elements.mean_anomaly_deg,
        mean_motion: elements.mean_motion_rev_per_day,
        rev_number,
    })
}

fn omm_from_fit(elements: &ElementSet, metadata: &TleMetadata) -> Result<Omm, TleFitError> {
    Ok(Omm {
        ccsds_omm_vers: "2.0".to_string(),
        creation_date: None,
        originator: None,
        object_name: Some(metadata.object_name.clone()),
        object_id: object_id_from_designator(&metadata.international_designator),
        center_name: Some("EARTH".to_string()),
        ref_frame: Some("TEME".to_string()),
        time_system: Some("UTC".to_string()),
        mean_element_theory: Some("SGP4".to_string()),
        epoch: crate::astro::omm::OmmEpoch::from_sgp4_julian_date(elements.epoch),
        mean_motion: elements.mean_motion_rev_per_day,
        eccentricity: elements.eccentricity,
        inclination_deg: elements.inclination_deg,
        ra_of_asc_node_deg: elements.right_ascension_deg,
        arg_of_pericenter_deg: elements.argument_of_perigee_deg,
        mean_anomaly_deg: elements.mean_anomaly_deg,
        ephemeris_type: 0,
        classification_type: metadata.classification.clone(),
        norad_cat_id: metadata.catalog_number,
        element_set_no: metadata.element_set_number,
        rev_at_epoch: metadata.rev_at_epoch,
        bstar: elements.bstar,
        mean_motion_dot: 0.0,
        mean_motion_ddot: 0.0,
        exact_sgp4_epoch: Some(elements.epoch),
        quantize_tle_derived_fields: false,
    })
}

fn object_id_from_designator(designator: &str) -> Option<String> {
    let trimmed = designator.trim();
    if trimmed.is_empty() {
        return None;
    }
    let bytes = trimmed.as_bytes();
    if bytes.len() >= 5 && bytes[..5].iter().all(u8::is_ascii_digit) {
        let yy: i32 = std::str::from_utf8(&bytes[..2]).ok()?.parse().ok()?;
        let year = if yy < 57 { 2000 + yy } else { 1900 + yy };
        let number = std::str::from_utf8(&bytes[2..5]).ok()?;
        let piece = std::str::from_utf8(&bytes[5..]).ok()?;
        Some(format!("{year:04}-{number}{piece}"))
    } else {
        Some(trimmed.to_string())
    }
}

fn civil_from_jd(epoch: JulianDate) -> (i64, i64, i64, i64, i64, f64) {
    let (midnight, fraction) = if (epoch.0.fract().abs() - 0.5).abs() < 1.0e-9 {
        (epoch.0, epoch.1)
    } else if epoch.1 >= 0.5 {
        (epoch.0 + 0.5, epoch.1 - 0.5)
    } else {
        (epoch.0 - 0.5, epoch.1 + 0.5)
    };
    civil_from_split_julian_date(midnight, fraction)
}

fn solver_options(config: &FitConfig) -> TrfOptions {
    let mut options = TrfOptions::default();
    if let Some(ftol) = config.ftol {
        options.ftol = ftol;
    }
    if let Some(xtol) = config.xtol {
        options.xtol = xtol;
    }
    if let Some(gtol) = config.gtol {
        options.gtol = gtol;
    }
    options.max_nfev = config.max_nfev;
    options.x_scale = config.x_scale.clone().unwrap_or(XScale::Jac);
    options.loss = config.loss;
    options.f_scale = config.f_scale;
    options
}

fn resolve_epoch(
    samples: &[FitSample],
    epoch: FitEpoch,
) -> Result<(JulianDate, usize), TleFitError> {
    match epoch {
        FitEpoch::Midpoint => {
            let first = samples[0].epoch;
            let last = samples[samples.len() - 1].epoch;
            let mid_seconds = seconds_between(last, first) * 0.5;
            let target = add_seconds(first, mid_seconds);
            let index = nearest_sample(samples, target);
            Ok((samples[index].epoch, index))
        }
        FitEpoch::First => Ok((samples[0].epoch, 0)),
        FitEpoch::Last => {
            let index = samples.len() - 1;
            Ok((samples[index].epoch, index))
        }
        FitEpoch::Sample(index) => samples
            .get(index)
            .map(|sample| (sample.epoch, index))
            .ok_or(TleFitError::InvalidInput {
                field: "epoch",
                reason: "sample index out of bounds",
            }),
        FitEpoch::Jd(jd) => {
            validate_epoch(jd)?;
            if seconds_between(jd, samples[0].epoch) < 0.0
                || seconds_between(samples[samples.len() - 1].epoch, jd) < 0.0
            {
                return Err(TleFitError::EpochOutsideArc);
            }
            Ok((jd, nearest_sample(samples, jd)))
        }
    }
}

fn nearest_sample(samples: &[FitSample], target: JulianDate) -> usize {
    let mut best = 0;
    let mut best_dt = seconds_between(samples[0].epoch, target).abs();
    for (i, sample) in samples.iter().enumerate().skip(1) {
        let dt = seconds_between(sample.epoch, target).abs();
        if dt < best_dt {
            best = i;
            best_dt = dt;
        }
    }
    best
}

fn add_seconds(epoch: JulianDate, seconds: f64) -> JulianDate {
    let mut whole = epoch.0;
    let mut fraction = epoch.1 + seconds / SECONDS_PER_DAY;
    let carry = fraction.floor();
    whole += carry;
    fraction -= carry;
    JulianDate(whole, fraction)
}

fn validate_epoch(epoch: JulianDate) -> Result<(), TleFitError> {
    validate_finite("epoch.whole", epoch.0)?;
    validate_finite("epoch.fraction", epoch.1)?;
    if !(0.0..1.0).contains(&epoch.1) {
        return Err(TleFitError::InvalidInput {
            field: "epoch.fraction",
            reason: "must be in [0, 1)",
        });
    }
    Ok(())
}

fn validate_vec3(field: &'static str, values: &[f64; 3]) -> Result<(), TleFitError> {
    for value in values {
        validate_finite(field, *value)?;
    }
    Ok(())
}

fn validate_optional_positive(field: &'static str, value: Option<f64>) -> Result<(), TleFitError> {
    if let Some(value) = value {
        validate_positive(field, value)?;
    }
    Ok(())
}

fn validate_positive(field: &'static str, value: f64) -> Result<(), TleFitError> {
    validate_finite(field, value)?;
    if value <= 0.0 {
        return Err(TleFitError::InvalidInput {
            field,
            reason: "must be > 0",
        });
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), TleFitError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(TleFitError::InvalidInput {
            field,
            reason: "must be finite",
        })
    }
}

fn seconds_between(later: JulianDate, earlier: JulianDate) -> f64 {
    ((later.0 - earlier.0) + (later.1 - earlier.1)) * SECONDS_PER_DAY
}

#[cfg(test)]
fn deg_to_rad(deg: f64) -> f64 {
    deg * std::f64::consts::PI / 180.0
}

fn rad_to_deg(rad: f64) -> f64 {
    rad * 180.0 / std::f64::consts::PI
}

fn normalize_angle(angle: f64) -> f64 {
    let mut out = angle % TAU;
    if out < 0.0 {
        out += TAU;
    }
    out
}

fn wrap_to_pi(angle: f64) -> f64 {
    let mut out = (angle + std::f64::consts::PI) % TAU;
    if out <= 0.0 {
        out += TAU;
    }
    out - std::f64::consts::PI
}

fn normalize_degrees(degrees: f64) -> f64 {
    let mut out = degrees % 360.0;
    if out < 0.0 {
        out += 360.0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::frames::transforms::{gcrs_to_teme_compute, TemeStateKm};
    use crate::astro::omm::{encode_kvn, parse_kvn};
    use crate::astro::propagator::{propagate_states, PropagationConfig, PropagationForceModel};
    use crate::astro::time::civil::split_julian_date;
    use crate::astro::time::scales::TimeScales;
    use crate::astro::tle;
    use crate::constants::J2000_JD;
    use trust_region_least_squares::model::ResidualModel;

    const ISS_L1: &str = "1 25544U 98067A   26168.18949189  .00009113  00000+0  17172-3 0  9996";
    const ISS_L2: &str = "2 25544  51.6332 300.0813 0004737 195.1146 164.9702 15.49273435571752";
    const GEO_L1: &str = "1 28884U 05041A   26167.71607684 -.00000267  00000+0  00000+0 0  9995";
    const GEO_L2: &str = "2 28884   3.5359  77.2731 0014354 137.8081 105.3728  0.98943614 75438";
    const DECAY_L1: &str = "1 28872U 05037B   05333.02012661  .25992681  00000-0  24476-3 0  1534";
    const DECAY_L2: &str = "2 28872  96.4736 157.9986 0303955 244.0492 110.6523 16.46015938 10708";
    const SSO_L1: &str = "1 28057U 03049A   06177.78615833  .00000060  00000-0  35970-4 0  1836";
    const SSO_L2: &str = "2 28057  98.4283 247.6961 0000884  88.1964 271.9322 14.35478080140550";

    fn arc_from_tle(line1: &str, line2: &str, offsets_min: &[f64]) -> Vec<FitSample> {
        let satellite = Satellite::from_tle(line1, line2).expect("valid TLE");
        let epoch = satellite.epoch_jd();
        offsets_min
            .iter()
            .map(|&minutes| {
                let prediction = satellite
                    .propagate(super::super::MinutesSinceEpoch(minutes))
                    .expect("propagates");
                FitSample {
                    epoch: add_seconds(epoch, minutes * 60.0),
                    position_teme_km: prediction.position,
                    velocity_teme_km_s: Some(prediction.velocity),
                }
            })
            .collect()
    }

    fn arc_from_elements(elements: &ElementSet, offsets_min: &[f64]) -> Vec<FitSample> {
        let satellite = Satellite::from_elements(elements).expect("valid elements");
        offsets_min
            .iter()
            .map(|&minutes| {
                let prediction = satellite
                    .propagate(super::super::MinutesSinceEpoch(minutes))
                    .expect("propagates");
                FitSample {
                    epoch: add_seconds(elements.epoch, minutes * 60.0),
                    position_teme_km: prediction.position,
                    velocity_teme_km_s: Some(prediction.velocity),
                }
            })
            .collect()
    }

    fn metadata_from_tle(line1: &str, line2: &str) -> TleMetadata {
        let parsed = tle::parse(line1, line2).expect("parse").elements;
        TleMetadata {
            catalog_number: parsed.catalog_number.parse().unwrap(),
            classification: parsed.classification,
            international_designator: parsed.international_designator,
            element_set_number: parsed.elset_number,
            rev_at_epoch: parsed.rev_number as i64,
            object_name: String::new(),
        }
    }

    fn metadata_for_catalog(catalog_number: u32) -> TleMetadata {
        TleMetadata {
            catalog_number,
            classification: "U".to_string(),
            international_designator: String::new(),
            element_set_number: 999,
            rev_at_epoch: 0,
            object_name: String::new(),
        }
    }

    fn chart_delta(a: &ElementSet, b: &ElementSet) -> [f64; 6] {
        let ca = MeanChart::from_elements(a);
        let cb = MeanChart::from_elements(b);
        [
            ca.n_rev_day - cb.n_rev_day,
            ca.af - cb.af,
            ca.ag - cb.ag,
            ca.chi - cb.chi,
            ca.psi - cb.psi,
            wrap_to_pi(ca.lam - cb.lam),
        ]
    }

    fn assert_element_sets_bit_identical(a: &ElementSet, b: &ElementSet) {
        assert_eq!(a.epoch.0.to_bits(), b.epoch.0.to_bits(), "epoch whole");
        assert_eq!(a.epoch.1.to_bits(), b.epoch.1.to_bits(), "epoch frac");
        assert_eq!(a.bstar.to_bits(), b.bstar.to_bits(), "bstar");
        assert_eq!(
            a.mean_motion_dot.to_bits(),
            b.mean_motion_dot.to_bits(),
            "mean motion dot"
        );
        assert_eq!(
            a.mean_motion_double_dot.to_bits(),
            b.mean_motion_double_dot.to_bits(),
            "mean motion ddot"
        );
        assert_eq!(a.eccentricity.to_bits(), b.eccentricity.to_bits(), "ecc");
        assert_eq!(
            a.argument_of_perigee_deg.to_bits(),
            b.argument_of_perigee_deg.to_bits(),
            "argp"
        );
        assert_eq!(
            a.inclination_deg.to_bits(),
            b.inclination_deg.to_bits(),
            "incl"
        );
        assert_eq!(
            a.mean_anomaly_deg.to_bits(),
            b.mean_anomaly_deg.to_bits(),
            "mean anomaly"
        );
        assert_eq!(
            a.mean_motion_rev_per_day.to_bits(),
            b.mean_motion_rev_per_day.to_bits(),
            "mean motion"
        );
        assert_eq!(
            a.right_ascension_deg.to_bits(),
            b.right_ascension_deg.to_bits(),
            "raan"
        );
        assert_eq!(a.catalog_number, b.catalog_number, "catalog");
    }

    fn assert_satellites_bit_identical(a: &Satellite, b: &Satellite) {
        let ea = a.epoch_jd();
        let eb = b.epoch_jd();
        assert_eq!(ea.0.to_bits(), eb.0.to_bits(), "sat epoch whole");
        assert_eq!(ea.1.to_bits(), eb.1.to_bits(), "sat epoch fraction");
        for minutes in [-120.0, 0.0, 120.0] {
            let pa = a
                .propagate(super::super::MinutesSinceEpoch(minutes))
                .expect("left propagates");
            let pb = b
                .propagate(super::super::MinutesSinceEpoch(minutes))
                .expect("right propagates");
            for axis in 0..3 {
                assert_eq!(
                    pa.position[axis].to_bits(),
                    pb.position[axis].to_bits(),
                    "position axis {axis} at {minutes} min"
                );
                assert_eq!(
                    pa.velocity[axis].to_bits(),
                    pb.velocity[axis].to_bits(),
                    "velocity axis {axis} at {minutes} min"
                );
            }
        }
    }

    fn assert_fit_bit_identical(a: &TleFit, b: &TleFit) {
        assert_element_sets_bit_identical(&a.elements, &b.elements);
        assert_eq!(a.line1, b.line1);
        assert_eq!(a.line2, b.line2);
        assert_eq!(a.omm, b.omm);
        assert_eq!(a.stats, b.stats);
    }

    fn rms_against_samples(samples: &[FitSample], satellite: &Satellite) -> f64 {
        let mut sum = 0.0;
        for sample in samples {
            let prediction = satellite
                .propagate_jd(sample.epoch)
                .expect("sample propagates");
            let mut r2 = 0.0;
            for axis in 0..3 {
                let residual = prediction.position[axis] - sample.position_teme_km[axis];
                r2 += residual * residual;
            }
            sum += r2;
        }
        (sum / samples.len() as f64).sqrt()
    }

    fn seed_rms(samples: &[FitSample], config: &FitConfig) -> f64 {
        let resolved = validate_and_resolve(samples, config).expect("valid fit input");
        let (x0, _) = initial_guess(samples, config, &resolved).expect("seed");
        let seed = chart_to_elements(
            &x0,
            resolved.epoch,
            config.fit_bstar,
            config.bstar_seed,
            config.metadata.catalog_number,
        )
        .expect("seed elements");
        let satellite =
            Satellite::from_elements_with_opsmode(&seed, config.opsmode).expect("seed satellite");
        rms_against_samples(samples, &satellite)
    }

    fn clean_position_rms(samples: &[FitSample], satellite: &Satellite) -> f64 {
        rms_against_samples(samples, satellite)
    }

    fn fit_round_trip_case(
        name: &str,
        truth: &ElementSet,
        samples: &[FitSample],
        max_rms_km: f64,
        max_chart_delta: f64,
    ) {
        let config = FitConfig {
            epoch: FitEpoch::Jd(truth.epoch),
            fit_bstar: false,
            metadata: metadata_for_catalog(truth.catalog_number),
            max_nfev: Some(120),
            ..FitConfig::default()
        };
        let fit = fit_tle(samples, &config).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            fit.stats.rms_position_km <= max_rms_km,
            "{name} rms {}",
            fit.stats.rms_position_km
        );
        let delta = chart_delta(&fit.elements, truth);
        for value in delta {
            assert!(value.abs() <= max_chart_delta, "{name} chart delta {value}");
        }
    }

    fn utc_time_scales_with_offset(offset_s: f64) -> TimeScales {
        let hour = (offset_s / 3600.0).floor() as i32;
        let rem = offset_s - hour as f64 * 3600.0;
        let minute = (rem / 60.0).floor() as i32;
        let second = rem - minute as f64 * 60.0;
        TimeScales::from_utc(2026, 6, 17, hour, minute, second).expect("time scales")
    }

    fn numerical_teme_arc(offsets_s: &[f64], include_velocity: bool) -> Vec<FitSample> {
        let start = TimeScales::from_utc(2026, 6, 17, 0, 0, 0.0).expect("start time");
        let start_tdb_s = (start.jd_tdb - J2000_JD) * SECONDS_PER_DAY;
        let config = PropagationConfig {
            force_model: PropagationForceModel::TwoBodyJ2,
            ..PropagationConfig::new(start_tdb_s, [7000.0, -1210.0, 1300.0], [1.2, 7.35, 0.92])
        };
        let epochs: Vec<f64> = offsets_s
            .iter()
            .map(|offset| start_tdb_s + offset)
            .collect();
        let states = propagate_states(&config, &epochs).expect("numerical propagation");
        states
            .iter()
            .zip(offsets_s)
            .map(|(state, &offset_s)| {
                let ts = utc_time_scales_with_offset(offset_s);
                let gcrs = TemeStateKm {
                    position_km: state.position_array(),
                    velocity_km_s: state.velocity_array(),
                };
                let (position, velocity) =
                    gcrs_to_teme_compute(&gcrs, &ts, false).expect("GCRS to TEME");
                let (jd_whole, fraction) = split_julian_date(2026, 6, 17, 0, 0, offset_s);
                FitSample {
                    epoch: JulianDate(jd_whole, fraction),
                    position_teme_km: [position.0, position.1, position.2],
                    velocity_teme_km_s: include_velocity
                        .then_some([velocity.0, velocity.1, velocity.2]),
                }
            })
            .collect()
    }

    #[test]
    fn chart_round_trips_degenerate_angles() {
        let cases = [
            ElementSet {
                epoch: JulianDate(2_460_000.5, 0.0),
                bstar: 0.0,
                mean_motion_dot: 0.0,
                mean_motion_double_dot: 0.0,
                eccentricity: 0.0,
                argument_of_perigee_deg: 0.0,
                inclination_deg: 0.0,
                mean_anomaly_deg: 42.0,
                mean_motion_rev_per_day: 1.0027,
                right_ascension_deg: 0.0,
                catalog_number: 1,
            },
            tle::parse(ISS_L1, ISS_L2)
                .unwrap()
                .elements
                .to_element_set()
                .unwrap(),
            tle::parse(GEO_L1, GEO_L2)
                .unwrap()
                .elements
                .to_element_set()
                .unwrap(),
        ];

        for elements in cases {
            let chart = MeanChart::from_elements(&elements).to_vec(true, elements.bstar);
            let round =
                chart_to_elements(&chart, elements.epoch, true, 0.0, elements.catalog_number)
                    .expect("chart domain");
            let delta = chart_delta(&elements, &round);
            for value in delta {
                assert!(value.abs() <= 1.0e-12, "delta {value}");
            }
            assert!((elements.bstar - round.bstar).abs() <= 1.0e-15);
        }
    }

    #[test]
    fn penalty_path_keeps_residual_length() {
        let samples = arc_from_tle(ISS_L1, ISS_L2, &[-20.0, 0.0, 20.0]);
        let problem = Sgp4FitProblem {
            samples: &samples,
            epoch: samples[1].epoch,
            opsmode: OpsMode::Improved,
            fit_bstar: true,
            fixed_bstar: 0.0,
            catalog_number: 25544,
            weights: vec![1.0; samples.len()],
            w_vel: 900.0,
            use_velocity: true,
            penalty_hit_at_solution: Cell::new(false),
        };
        let mut out = Vec::new();
        problem.residual(&[15.0, ECC_MAX, 0.0, 0.0, 0.0, 0.0, 0.0], &mut out);
        assert_eq!(out.len(), samples.len() * 6);
        assert!(out.iter().all(|&value| value == PENALTY_KM));
        assert!(problem.penalty_hit_at_solution.get());
    }

    #[test]
    fn round_trip_recovers_iss_elements() {
        let offsets: Vec<f64> = (-36..=36).map(|i| i as f64 * 10.0).collect();
        let samples = arc_from_tle(ISS_L1, ISS_L2, &offsets);
        let truth = tle::parse(ISS_L1, ISS_L2)
            .unwrap()
            .elements
            .to_element_set()
            .unwrap();
        let config = FitConfig {
            epoch: FitEpoch::Jd(truth.epoch),
            metadata: metadata_from_tle(ISS_L1, ISS_L2),
            max_nfev: Some(80),
            ..FitConfig::default()
        };

        let fit = fit_tle(&samples, &config).expect("fit converges");
        assert!(fit.stats.status >= 1);
        assert!(
            fit.stats.rms_position_km <= 1.0e-3,
            "rms {}",
            fit.stats.rms_position_km
        );
        let delta = chart_delta(&fit.elements, &truth);
        assert!(delta[0].abs() <= 1.0e-8, "n {}", delta[0]);
        for value in &delta[1..5] {
            assert!(value.abs() <= 1.0e-7, "chart {value}");
        }
        assert!(delta[5].abs() <= 1.0e-6, "lam {}", delta[5]);
        assert!(fit.stats.tle_rms_position_km <= 0.1);
        assert_eq!(fit.elements.mean_motion_dot, 0.0);
        assert_eq!(fit.elements.mean_motion_double_dot, 0.0);
        assert_eq!(fit.omm.mean_motion_dot, 0.0);
        assert_eq!(fit.omm.mean_motion_ddot, 0.0);

        let omm_elements = fit.omm.to_element_set().expect("fitted OMM bridge");
        assert_element_sets_bit_identical(&omm_elements, &fit.elements);
        let from_omm = Satellite::from_omm(&fit.omm).expect("fitted OMM satellite");
        let from_elements =
            Satellite::from_elements(&fit.elements).expect("fitted element satellite");
        assert_satellites_bit_identical(&from_omm, &from_elements);

        let encoded_omm = encode_kvn(&fit.omm);
        let reparsed_omm = parse_kvn(&encoded_omm).expect("encoded fitted OMM reparses");
        let _ = reparsed_omm
            .to_element_set()
            .expect("encoded fitted OMM bridges");
    }

    #[test]
    fn geo_short_arc_marks_bstar_unobservable() {
        let offsets: Vec<f64> = (-12..=12).map(|i| i as f64 * 30.0).collect();
        let samples = arc_from_tle(GEO_L1, GEO_L2, &offsets);
        let truth = tle::parse(GEO_L1, GEO_L2)
            .unwrap()
            .elements
            .to_element_set()
            .unwrap();
        let config = FitConfig {
            epoch: FitEpoch::Jd(truth.epoch),
            metadata: metadata_from_tle(GEO_L1, GEO_L2),
            max_nfev: Some(80),
            ..FitConfig::default()
        };

        let fit = fit_tle(&samples, &config).expect("fit converges");
        assert!(fit.stats.rms_position_km <= 1.0e-3);
        assert!(!fit.stats.bstar_observable);
    }

    #[test]
    fn high_drag_bstar_recovery_is_observable() {
        let offsets: Vec<f64> = (0..=10).map(|i| i as f64 * 5.0).collect();
        let samples = arc_from_tle(DECAY_L1, DECAY_L2, &offsets);
        let truth = tle::parse(DECAY_L1, DECAY_L2)
            .unwrap()
            .elements
            .to_element_set()
            .unwrap();
        let config = FitConfig {
            epoch: FitEpoch::Jd(truth.epoch),
            metadata: metadata_from_tle(DECAY_L1, DECAY_L2),
            max_nfev: Some(120),
            ..FitConfig::default()
        };

        let fit = fit_tle(&samples, &config).expect("high-drag fit converges");
        assert!(fit.stats.bstar_observable);
        assert!(
            (fit.elements.bstar - truth.bstar).abs() <= 1.0e-6,
            "bstar fit {} truth {}",
            fit.elements.bstar,
            truth.bstar
        );
    }

    #[test]
    fn fitted_omm_epoch_survives_text_round_trip() {
        let truth = tle::parse(ISS_L1, ISS_L2)
            .unwrap()
            .elements
            .to_element_set()
            .unwrap();
        let metadata = metadata_from_tle(ISS_L1, ISS_L2);

        // Midnight-anchored split (what calendar/TLE producers emit): the
        // femtosecond epoch text carries the full split JD, so a text round
        // trip rebuilds it bit-identically and propagation matches exactly.
        let omm = omm_from_fit(&truth, &metadata).expect("fitted OMM");
        let mut reparsed = parse_kvn(&encode_kvn(&omm)).expect("fitted OMM reparses");
        assert_eq!(reparsed.epoch, omm.epoch);
        assert_eq!(reparsed.exact_sgp4_epoch, None);
        // The quantize flag is in-memory fit state, not carried by the text.
        reparsed.quantize_tle_derived_fields = false;

        let in_memory_elements = omm.to_element_set().expect("in-memory bridge");
        let rebuilt = reparsed.to_element_set().expect("reparsed OMM bridges");
        assert_element_sets_bit_identical(&rebuilt, &in_memory_elements);

        let direct = Satellite::from_omm(&omm).expect("direct satellite");
        let round_tripped = Satellite::from_omm(&reparsed).expect("round-tripped satellite");
        for minutes in [0.0, 45.0, 720.0] {
            let jd = add_seconds(truth.epoch, minutes * 60.0);
            let a = direct.propagate_jd(jd).expect("direct propagates");
            let b = round_tripped.propagate_jd(jd).expect("reparsed propagates");
            assert_eq!(a.position, b.position, "position at {minutes} min");
            assert_eq!(a.velocity, b.velocity, "velocity at {minutes} min");
        }

        // Non-canonical (noon-anchored) split: the side channel preserves the
        // producer's representation in memory, while the text round trip
        // renormalizes to the midnight-anchored split of the same instant.
        let JulianDate(whole, fraction) = truth.epoch;
        let shifted = if fraction >= 0.5 {
            JulianDate(whole + 0.5, fraction - 0.5)
        } else {
            JulianDate(whole - 0.5, fraction + 0.5)
        };
        assert_eq!(shifted.0 + shifted.1, whole + fraction);
        let mut shifted_elements = truth.clone();
        shifted_elements.epoch = shifted;
        let omm2 = omm_from_fit(&shifted_elements, &metadata).expect("shifted fitted OMM");
        let in_memory = omm2.to_element_set().expect("in-memory bridge").epoch;
        assert_eq!(in_memory.0.to_bits(), shifted.0.to_bits());
        assert_eq!(in_memory.1.to_bits(), shifted.1.to_bits());

        let mut reparsed2 = parse_kvn(&encode_kvn(&omm2)).expect("shifted OMM reparses");
        reparsed2.quantize_tle_derived_fields = false;
        let rebuilt2 = reparsed2.to_element_set().expect("shifted bridge").epoch;
        assert_eq!(rebuilt2.0.to_bits(), whole.to_bits());
        assert_eq!(rebuilt2.1.to_bits(), fraction.to_bits());
        assert_eq!(rebuilt2.0 + rebuilt2.1, shifted.0 + shifted.1);

        // Propagation error bound between the in-memory representation and
        // the renormalized text round trip: the instant is exact, only the
        // split representation differs, so any drift stays at the last-ULP
        // tsince level (well under a micrometer).
        let in_memory_sat = Satellite::from_omm(&omm2).expect("in-memory satellite");
        let round_tripped2 = Satellite::from_omm(&reparsed2).expect("round-tripped satellite");
        for minutes in [0.0, 45.0, 720.0] {
            let jd = add_seconds(truth.epoch, minutes * 60.0);
            let a = in_memory_sat
                .propagate_jd(jd)
                .expect("in-memory propagates");
            let b = round_tripped2
                .propagate_jd(jd)
                .expect("reparsed propagates");
            for axis in 0..3 {
                assert!(
                    (a.position[axis] - b.position[axis]).abs() <= 1.0e-9,
                    "axis {axis} at {minutes} min: {} vs {}",
                    a.position[axis],
                    b.position[axis]
                );
            }
        }
    }

    #[test]
    fn molniya_and_sun_sync_round_trip_cases_converge() {
        let molniya = ElementSet {
            epoch: JulianDate(2_461_208.5, 0.317_123_456_789_012),
            bstar: 0.0,
            mean_motion_dot: 0.0,
            mean_motion_double_dot: 0.0,
            eccentricity: 0.742_8,
            argument_of_perigee_deg: 270.5,
            inclination_deg: 63.4,
            mean_anomaly_deg: 15.8,
            mean_motion_rev_per_day: 2.006_1,
            right_ascension_deg: 90.16,
            catalog_number: 28163,
        };
        let molniya_offsets: Vec<f64> = (-12..=12).map(|i| i as f64 * 120.0).collect();
        let molniya_samples = arc_from_elements(&molniya, &molniya_offsets);
        fit_round_trip_case("molniya", &molniya, &molniya_samples, 1.0e-3, 2.0e-6);

        let sso_truth = tle::parse(SSO_L1, SSO_L2)
            .unwrap()
            .elements
            .to_element_set()
            .unwrap();
        let sso_offsets: Vec<f64> = (-24..=24).map(|i| i as f64 * 10.0).collect();
        let sso_samples = arc_from_tle(SSO_L1, SSO_L2, &sso_offsets);
        fit_round_trip_case("sun-sync", &sso_truth, &sso_samples, 1.0e-3, 2.0e-6);
    }

    #[test]
    fn raan_singular_geo_round_trip_converges() {
        let truth = ElementSet {
            epoch: JulianDate(2_461_208.5, 0.625_987_654_321_098),
            bstar: 0.0,
            mean_motion_dot: 0.0,
            mean_motion_double_dot: 0.0,
            eccentricity: 0.0012,
            argument_of_perigee_deg: 137.8,
            inclination_deg: 0.05,
            mean_anomaly_deg: 105.4,
            mean_motion_rev_per_day: 1.0027,
            right_ascension_deg: 77.3,
            catalog_number: 39000,
        };
        let offsets: Vec<f64> = (-12..=12).map(|i| i as f64 * 120.0).collect();
        let samples = arc_from_elements(&truth, &offsets);
        fit_round_trip_case("raan-singular geo", &truth, &samples, 1.0e-3, 2.0e-6);
    }

    #[test]
    fn numerical_cross_source_oracle_improves_seed_position_only() {
        let offsets_s: Vec<f64> = (0..=18).map(|i| i as f64 * 300.0).collect();
        let samples = numerical_teme_arc(&offsets_s, false);
        let config = FitConfig {
            epoch: FitEpoch::Midpoint,
            fit_bstar: false,
            use_velocity: false,
            metadata: metadata_for_catalog(60001),
            max_nfev: Some(120),
            ..FitConfig::default()
        };

        let seed = seed_rms(&samples, &config);
        let fit = fit_tle(&samples, &config).expect("numerical position fit");
        assert!(
            fit.stats.rms_position_km < seed,
            "fit rms {} seed rms {}",
            fit.stats.rms_position_km,
            seed
        );
        assert!(!fit.stats.bstar_observable);
    }

    #[test]
    fn numerical_cross_source_oracle_improves_seed_with_velocity() {
        let offsets_s: Vec<f64> = (0..=18).map(|i| i as f64 * 300.0).collect();
        let samples = numerical_teme_arc(&offsets_s, true);
        let config = FitConfig {
            epoch: FitEpoch::Midpoint,
            fit_bstar: false,
            use_velocity: true,
            metadata: metadata_for_catalog(60002),
            max_nfev: Some(120),
            ..FitConfig::default()
        };

        let seed = seed_rms(&samples, &config);
        let fit = fit_tle(&samples, &config).expect("numerical position+velocity fit");
        assert!(
            fit.stats.rms_position_km < seed,
            "fit rms {} seed rms {}",
            fit.stats.rms_position_km,
            seed
        );
        assert!(fit.stats.rms_velocity_km_s.is_some());
        assert!(!fit.stats.bstar_observable);
    }

    #[test]
    fn seed_refinement_improves_initial_guess() {
        let samples = arc_from_tle(ISS_L1, ISS_L2, &[-30.0, -15.0, 0.0, 15.0, 30.0]);
        let config = FitConfig {
            epoch: FitEpoch::Sample(2),
            metadata: metadata_from_tle(ISS_L1, ISS_L2),
            ..FitConfig::default()
        };
        let resolved = validate_and_resolve(&samples, &config).expect("valid fit input");
        let sample = samples[resolved.epoch_index];
        let target = chart_from_state(sample.position_teme_km, sample.velocity_teme_km_s.unwrap())
            .expect("target chart");
        let raw = target.to_vec(config.fit_bstar, config.bstar_seed);
        let raw_elements = chart_to_elements(
            &raw,
            resolved.epoch,
            config.fit_bstar,
            config.bstar_seed,
            config.metadata.catalog_number,
        )
        .expect("raw seed");
        let raw_satellite = Satellite::from_elements(&raw_elements).expect("raw seed satellite");
        let raw_rms = rms_against_samples(&samples, &raw_satellite);

        let (refined, passes) = refine_seed(
            raw,
            target,
            resolved.epoch,
            config.opsmode,
            config.fit_bstar,
            config.bstar_seed,
            config.metadata.catalog_number,
        );
        let refined_elements = chart_to_elements(
            &refined,
            resolved.epoch,
            config.fit_bstar,
            config.bstar_seed,
            config.metadata.catalog_number,
        )
        .expect("refined seed");
        let refined_satellite =
            Satellite::from_elements(&refined_elements).expect("refined seed satellite");
        let refined_rms = rms_against_samples(&samples, &refined_satellite);

        assert!(passes > 0);
        assert!(
            refined_rms < raw_rms,
            "refined rms {refined_rms} raw rms {raw_rms}"
        );
    }

    #[test]
    fn seed_propagation_reports_typed_error() {
        let elements = tle::parse(DECAY_L1, DECAY_L2)
            .unwrap()
            .elements
            .to_element_set()
            .unwrap();
        let samples = [
            FitSample {
                epoch: elements.epoch,
                position_teme_km: [7000.0, 0.0, 0.0],
                velocity_teme_km_s: Some([0.0, 7.5, 0.0]),
            },
            FitSample {
                epoch: add_seconds(elements.epoch, 1440.0 * 60.0),
                position_teme_km: [7000.0, 0.0, 0.0],
                velocity_teme_km_s: Some([0.0, 7.5, 0.0]),
            },
        ];
        let err = validate_seed_propagates(&samples, &elements, OpsMode::Improved)
            .expect_err("decayed seed must error");
        assert!(matches!(
            err,
            TleFitError::SeedPropagation { epoch_index: 1, .. }
        ));
    }

    #[test]
    fn epoch_selection_modes_resolve_expected_epochs() {
        let samples = arc_from_tle(ISS_L1, ISS_L2, &[-20.0, -10.0, 0.0, 10.0, 20.0]);
        for (mode, expected_index) in [
            (FitEpoch::First, 0),
            (FitEpoch::Midpoint, 2),
            (FitEpoch::Last, 4),
            (FitEpoch::Sample(3), 3),
        ] {
            let config = FitConfig {
                epoch: mode,
                ..FitConfig::default()
            };
            let resolved = validate_and_resolve(&samples, &config).expect("resolve epoch");
            assert_eq!(resolved.epoch_index, expected_index);
            assert_eq!(resolved.epoch, samples[expected_index].epoch);
        }

        let explicit = add_seconds(samples[0].epoch, 15.0 * 60.0);
        let config = FitConfig {
            epoch: FitEpoch::Jd(explicit),
            ..FitConfig::default()
        };
        let resolved = validate_and_resolve(&samples, &config).expect("resolve JD epoch");
        assert_eq!(resolved.epoch, explicit);
        assert_eq!(resolved.epoch_index, 1);

        let outside = add_seconds(samples[0].epoch, -1.0);
        let config = FitConfig {
            epoch: FitEpoch::Jd(outside),
            ..FitConfig::default()
        };
        assert!(matches!(
            validate_and_resolve(&samples, &config),
            Err(TleFitError::EpochOutsideArc)
        ));
    }

    #[test]
    fn sample_weights_reduce_outlier_leverage() {
        let offsets: Vec<f64> = (-12..=12).map(|i| i as f64 * 5.0).collect();
        let clean = arc_from_tle(ISS_L1, ISS_L2, &offsets);
        let mut corrupted = clean.clone();
        let outlier = corrupted.len() / 2;
        corrupted[outlier].position_teme_km[0] += 8.0;
        corrupted[outlier].position_teme_km[1] -= 5.0;

        let base_config = FitConfig {
            epoch: FitEpoch::Jd(clean[outlier].epoch),
            metadata: metadata_from_tle(ISS_L1, ISS_L2),
            max_nfev: Some(100),
            ..FitConfig::default()
        };
        let unweighted = fit_tle(&corrupted, &base_config).expect("unweighted fit");
        let unweighted_sat =
            Satellite::from_elements(&unweighted.elements).expect("unweighted satellite");
        let unweighted_clean_rms = clean_position_rms(&clean, &unweighted_sat);

        let mut weights = vec![1.0; corrupted.len()];
        weights[outlier] = 0.01;
        let weighted_config = FitConfig {
            weights: Some(weights),
            ..base_config
        };
        let weighted = fit_tle(&corrupted, &weighted_config).expect("weighted fit");
        let weighted_sat =
            Satellite::from_elements(&weighted.elements).expect("weighted satellite");
        let weighted_clean_rms = clean_position_rms(&clean, &weighted_sat);

        assert!(
            weighted_clean_rms < unweighted_clean_rms,
            "weighted clean rms {weighted_clean_rms} unweighted {unweighted_clean_rms}"
        );
    }

    #[test]
    fn deterministic_fit_repeats_bit_identically() {
        let samples = arc_from_tle(ISS_L1, ISS_L2, &[-30.0, -15.0, 0.0, 15.0, 30.0]);
        let config = FitConfig {
            epoch: FitEpoch::Sample(2),
            metadata: metadata_from_tle(ISS_L1, ISS_L2),
            max_nfev: Some(80),
            ..FitConfig::default()
        };

        let a = fit_tle(&samples, &config).expect("first fit");
        let b = fit_tle(&samples, &config).expect("second fit");
        assert_fit_bit_identical(&a, &b);
    }

    #[test]
    fn iss_ninety_minute_arc_marks_bstar_unobservable() {
        let offsets: Vec<f64> = (-9..=9).map(|i| i as f64 * 5.0).collect();
        let samples = arc_from_tle(ISS_L1, ISS_L2, &offsets);
        let truth = tle::parse(ISS_L1, ISS_L2)
            .unwrap()
            .elements
            .to_element_set()
            .unwrap();
        let config = FitConfig {
            epoch: FitEpoch::Jd(truth.epoch),
            metadata: metadata_from_tle(ISS_L1, ISS_L2),
            max_nfev: Some(80),
            ..FitConfig::default()
        };

        let fit = fit_tle(&samples, &config).expect("ISS short arc fit");
        assert!(!fit.stats.bstar_observable);
    }

    #[test]
    fn validation_reports_typed_errors() {
        let mut samples = arc_from_tle(ISS_L1, ISS_L2, &[-10.0, 0.0, 10.0]);
        samples[1].epoch = samples[0].epoch;
        assert!(matches!(
            fit_tle(&samples, &FitConfig::default()),
            Err(TleFitError::EpochsNotIncreasing { index: 1 })
        ));

        let mut samples = arc_from_tle(ISS_L1, ISS_L2, &[-10.0, 0.0, 10.0]);
        samples[1].velocity_teme_km_s = None;
        assert!(matches!(
            fit_tle(&samples, &FitConfig::default()),
            Err(TleFitError::MixedVelocityPresence)
        ));

        let samples = arc_from_tle(ISS_L1, ISS_L2, &[-10.0, 0.0, 10.0]);
        let mut config = FitConfig::default();
        config.metadata.classification = "X".to_string();
        assert!(matches!(
            fit_tle(&samples, &config),
            Err(TleFitError::InvalidInput {
                field: "metadata.classification",
                ..
            })
        ));
    }

    #[test]
    fn did_not_converge_carries_best_effort_fit() {
        let offsets: Vec<f64> = (-6..=6).map(|i| i as f64 * 10.0).collect();
        let samples = arc_from_tle(ISS_L1, ISS_L2, &offsets);
        let truth = tle::parse(ISS_L1, ISS_L2)
            .unwrap()
            .elements
            .to_element_set()
            .unwrap();
        let config = FitConfig {
            epoch: FitEpoch::Jd(truth.epoch),
            metadata: metadata_from_tle(ISS_L1, ISS_L2),
            max_nfev: Some(2),
            ..FitConfig::default()
        };

        let err = fit_tle(&samples, &config).expect_err("budget stops");
        match err {
            TleFitError::DidNotConverge { result } => {
                assert_eq!(result.stats.status, 0);
                assert!(!result.line1.is_empty());
                assert!(!result.line2.is_empty());
            }
            other => panic!("unexpected error {other:?}"),
        }
    }
}
