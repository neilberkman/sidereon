//! Time of closest approach search between two TLE-backed satellites.
//!
//! The search predicate is relative range in the TEME frame. Sampling and
//! refinement are delegated to the shared event finder; this module only owns
//! the SGP4 propagation glue and the relative-state result packaging.

use crate::astro::conjunction::{
    collision_probability, CollisionPc, ConjunctionError, ConjunctionState, PcMethod,
};
use crate::astro::covariance::{Covariance6, Covariance6Error, Mat6};
use crate::astro::events::{
    EventFinder, EventFinderError, ExtremumEvent, ExtremumKind, ScalarEventPredicate,
};
use crate::astro::frames::transforms::{mat3_vec3_mul_unchecked, teme_to_gcrs_matrix};
use crate::astro::math::mat3::Mat3;
use crate::astro::math::vec3;
use crate::astro::propagator::api::IntegratorOptions;
use crate::astro::propagator::{ForceModelKind, IntegratorKind, StatePropagator};
use crate::astro::sgp4::{Error as Sgp4Error, JulianDate, MinutesSinceEpoch, Satellite};
use crate::astro::state::CartesianState;
use crate::astro::time::civil::{civil_from_split_julian_date, split_julian_date_add_seconds};
use crate::astro::time::scales::{julian_day_number, TimeScales};
use crate::constants::SECONDS_PER_DAY;
use crate::validate;
use std::sync::Mutex;
const FRAME_DERIVATIVE_STEP_SECONDS: f64 = 0.5;
const BOUNDARY_RANGE_ABS_TOL_KM: f64 = 1.0e-9;
const BOUNDARY_RANGE_REL_TOL: f64 = 1.0e-12;
const MAX_TCA_RELATIVE_POSITION_KM: f64 = 1.0e12;
const MAX_TCA_RELATIVE_VELOCITY_KM_S: f64 = 1.0e9;
/// Fallback per-object position covariance for Pc screening, km^2.
///
/// This is a convenience covariance for catalog screening when no object
/// covariance is available. Callers with conjunction data should supply their
/// own object covariances through [`TcaPcOptions::with_covariances`].
pub const DEFAULT_TCA_POSITION_COVARIANCE_KM2: [[f64; 3]; 3] =
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Options for [`find_tca_candidates`] and [`find_tca_candidates_from_tles`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcaFinderOptions {
    /// Coarse sampling step used to bracket local range minima.
    pub coarse_step_seconds: f64,
    /// Time tolerance to which each local range minimum is refined.
    pub time_tolerance_seconds: f64,
}

impl Default for TcaFinderOptions {
    fn default() -> Self {
        Self {
            coarse_step_seconds: 60.0,
            time_tolerance_seconds: 1.0e-3,
        }
    }
}

/// One local time of closest approach candidate.
///
/// The relative state is `primary - secondary`, in the same TEME frame that
/// SGP4 returns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcaCandidate {
    /// Refined absolute TCA as a split Julian date.
    pub tca_time: JulianDate,
    /// Refined seconds since the search window start.
    pub tca_seconds_since_window_start: f64,
    /// Norm of [`TcaCandidate::relative_position_km`].
    pub miss_distance_km: f64,
    /// Primary minus secondary TEME position, km.
    pub relative_position_km: [f64; 3],
    /// Primary minus secondary TEME velocity, km/s.
    pub relative_velocity_km_s: [f64; 3],
}

/// One threshold-screening result for a primary against a secondary catalog.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcaScreeningHit {
    /// Index of the secondary satellite in the caller-supplied catalog slice.
    pub secondary_index: usize,
    /// Refined TCA candidate whose miss distance is at or below the threshold.
    pub candidate: TcaCandidate,
}

/// Borrowed two-line element set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TcaTle<'a> {
    /// TLE line 1.
    pub line1: &'a str,
    /// TLE line 2.
    pub line2: &'a str,
}

impl<'a> TcaTle<'a> {
    /// Build a borrowed TLE pair.
    pub const fn new(line1: &'a str, line2: &'a str) -> Self {
        Self { line1, line2 }
    }
}

/// Borrowed TLE plus its 6x6 state covariance at the TLE epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcaTleWithCovariance<'a> {
    /// Borrowed two-line element set.
    pub tle: TcaTle<'a>,
    /// State covariance at the satellite's TLE epoch.
    pub covariance0: Covariance6,
}

impl<'a> TcaTleWithCovariance<'a> {
    /// Build a borrowed TLE pair with its initial state covariance.
    pub const fn new(line1: &'a str, line2: &'a str, covariance0: Covariance6) -> Self {
        Self {
            tle: TcaTle::new(line1, line2),
            covariance0,
        }
    }

    /// Attach an initial state covariance to an existing borrowed TLE.
    pub const fn from_tle(tle: TcaTle<'a>, covariance0: Covariance6) -> Self {
        Self { tle, covariance0 }
    }
}

/// Absolute time window searched for TCA events.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcaWindow {
    /// Start of the search window.
    pub start: JulianDate,
    /// End of the search window.
    pub end: JulianDate,
}

impl TcaWindow {
    /// Build a TCA search window from absolute start and end Julian dates.
    pub const fn new(start: JulianDate, end: JulianDate) -> Self {
        Self { start, end }
    }

    /// Build a TCA search window from a start Julian date and duration.
    pub fn from_start_and_duration_seconds(
        start: JulianDate,
        duration_seconds: f64,
    ) -> Result<Self, TcaError> {
        validate::finite_nonneg(duration_seconds, "duration_seconds").map_err(map_input)?;
        Ok(Self {
            start,
            end: add_seconds_to_julian_date(start, duration_seconds),
        })
    }
}

/// Per-object position covariances used when computing Pc at a TCA.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcaPcCovariances {
    /// Primary-object GCRS position covariance, km^2.
    pub primary_covariance_km2: [[f64; 3]; 3],
    /// Secondary-object GCRS position covariance, km^2.
    pub secondary_covariance_km2: [[f64; 3]; 3],
}

impl Default for TcaPcCovariances {
    fn default() -> Self {
        Self {
            primary_covariance_km2: DEFAULT_TCA_POSITION_COVARIANCE_KM2,
            secondary_covariance_km2: DEFAULT_TCA_POSITION_COVARIANCE_KM2,
        }
    }
}

/// Collision-probability options for evaluating a TCA candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcaPcOptions {
    /// Hard-body radius passed to the conjunction Pc module, km.
    pub hard_body_radius_km: f64,
    /// Collision-probability method from the conjunction Pc module.
    pub method: PcMethod,
    /// Per-object position covariances, km^2.
    pub covariances: TcaPcCovariances,
}

impl TcaPcOptions {
    /// Build Pc options using the fallback TCA position covariance.
    pub fn with_default_covariance(hard_body_radius_km: f64, method: PcMethod) -> Self {
        Self {
            hard_body_radius_km,
            method,
            covariances: TcaPcCovariances::default(),
        }
    }

    /// Build Pc options with caller-supplied object position covariances.
    pub fn with_covariances(
        hard_body_radius_km: f64,
        method: PcMethod,
        primary_covariance_km2: [[f64; 3]; 3],
        secondary_covariance_km2: [[f64; 3]; 3],
    ) -> Self {
        Self {
            hard_body_radius_km,
            method,
            covariances: TcaPcCovariances {
                primary_covariance_km2,
                secondary_covariance_km2,
            },
        }
    }
}

/// Collision-probability and covariance-transport settings for screening
/// helpers whose primary/secondary covariances vary by catalog object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcaPropagatedCovarianceOptions {
    /// Hard-body radius passed to the conjunction Pc module, km.
    pub hard_body_radius_km: f64,
    /// Collision-probability method from the conjunction Pc module.
    pub method: PcMethod,
    /// Numerical force model used only for covariance transport.
    pub force_model: ForceModelKind,
    /// Numerical integrator used only for covariance transport.
    pub integrator: IntegratorKind,
    /// Step-size / tolerance controls for covariance transport.
    pub integrator_options: IntegratorOptions,
}

impl TcaPropagatedCovarianceOptions {
    /// Build propagated-covariance screening settings using Earth two-body + J2
    /// and adaptive Dormand-Prince propagation for covariance transport.
    pub fn new(hard_body_radius_km: f64, method: PcMethod) -> Self {
        Self {
            hard_body_radius_km,
            method,
            force_model: ForceModelKind::two_body_j2(),
            integrator: IntegratorKind::Dp54,
            integrator_options: IntegratorOptions::default(),
        }
    }

    /// Replace the numerical covariance-transport settings.
    pub fn with_covariance_propagator(
        mut self,
        force_model: ForceModelKind,
        integrator: IntegratorKind,
        integrator_options: IntegratorOptions,
    ) -> Self {
        self.force_model = force_model;
        self.integrator = integrator;
        self.integrator_options = integrator_options;
        self
    }

    fn for_initial_covariances(
        self,
        primary_covariance0: Covariance6,
        secondary_covariance0: Covariance6,
    ) -> TcaPropagatedCovariancePcOptions {
        TcaPropagatedCovariancePcOptions {
            hard_body_radius_km: self.hard_body_radius_km,
            method: self.method,
            primary_covariance0,
            secondary_covariance0,
            force_model: self.force_model,
            integrator: self.integrator,
            integrator_options: self.integrator_options,
        }
    }
}

/// Collision-probability options for propagating state covariances to a TCA.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcaPropagatedCovariancePcOptions {
    /// Hard-body radius passed to the conjunction Pc module, km.
    pub hard_body_radius_km: f64,
    /// Collision-probability method from the conjunction Pc module.
    pub method: PcMethod,
    /// Primary-object initial 6x6 state covariance at the primary TLE epoch.
    pub primary_covariance0: Covariance6,
    /// Secondary-object initial 6x6 state covariance at the secondary TLE epoch.
    pub secondary_covariance0: Covariance6,
    /// Numerical force model used only for covariance transport.
    pub force_model: ForceModelKind,
    /// Numerical integrator used only for covariance transport.
    pub integrator: IntegratorKind,
    /// Step-size / tolerance controls for covariance transport.
    pub integrator_options: IntegratorOptions,
}

impl TcaPropagatedCovariancePcOptions {
    /// Build propagated-covariance Pc options using Earth two-body + J2 and
    /// adaptive Dormand-Prince propagation for the covariance transport.
    pub fn new(
        hard_body_radius_km: f64,
        method: PcMethod,
        primary_covariance0: Covariance6,
        secondary_covariance0: Covariance6,
    ) -> Self {
        Self {
            hard_body_radius_km,
            method,
            primary_covariance0,
            secondary_covariance0,
            force_model: ForceModelKind::two_body_j2(),
            integrator: IntegratorKind::Dp54,
            integrator_options: IntegratorOptions::default(),
        }
    }

    /// Replace the numerical covariance-transport settings.
    pub fn with_covariance_propagator(
        mut self,
        force_model: ForceModelKind,
        integrator: IntegratorKind,
        integrator_options: IntegratorOptions,
    ) -> Self {
        self.force_model = force_model;
        self.integrator = integrator;
        self.integrator_options = integrator_options;
        self
    }
}

/// A TCA candidate with the existing conjunction-module Pc result at that TCA.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcaConjunction {
    /// Refined TCA candidate and miss-distance summary.
    pub candidate: TcaCandidate,
    /// Collision probability computed by [`crate::astro::conjunction`].
    pub collision_probability: CollisionPc,
}

/// A threshold-screening hit with Pc evaluated at the returned TCA.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcaScreeningConjunctionHit {
    /// Index of the secondary satellite in the caller-supplied catalog slice.
    pub secondary_index: usize,
    /// TCA and Pc result for this threshold breach.
    pub conjunction: TcaConjunction,
}

/// One object in a materialized state-vector screening catalog.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogStateVector {
    /// Optional caller-supplied object identifier.
    pub id: Option<String>,
    /// ECI position, km.
    pub position_km: [f64; 3],
    /// ECI velocity, km/s.
    pub velocity_km_s: [f64; 3],
    /// Position covariance, km^2.
    pub covariance_km2: [[f64; 3]; 3],
    /// Object hard-body radius, km.
    pub hard_body_radius_km: f64,
}

/// Options for [`screen_state_vector_catalog`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CatalogScreeningOptions {
    /// Coarse position-pair filter threshold, km.
    pub miss_threshold_km: f64,
    /// Minimum successful collision probability retained in the result set.
    pub pc_threshold: f64,
    /// Collision-probability method.
    pub method: PcMethod,
}

impl Default for CatalogScreeningOptions {
    fn default() -> Self {
        Self {
            miss_threshold_km: 50.0,
            pc_threshold: 0.0,
            method: PcMethod::FosterEqualArea,
        }
    }
}

/// A prefiltered catalog pair.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogScreeningCandidate {
    pub i: usize,
    pub j: usize,
    pub id1: Option<String>,
    pub id2: Option<String>,
    pub miss_km: f64,
}

/// Successful collision-probability evaluation for a catalog candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CatalogCollision {
    pub probability: CollisionPc,
    pub method: PcMethod,
}

/// One catalog screening result row.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogScreeningResult {
    pub candidate: CatalogScreeningCandidate,
    pub collision: Option<CatalogCollision>,
    pub error: Option<ConjunctionError>,
}

/// Which object failed during TCA setup or propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcaObject {
    /// The first satellite supplied by the caller.
    Primary,
    /// The second satellite supplied by the caller.
    Secondary,
}

/// Error while finding TCA candidates.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TcaError {
    /// Invalid TCA input.
    #[error("invalid TCA input {field}: {reason}")]
    InvalidInput {
        /// Field or input source that failed validation.
        field: &'static str,
        /// Stable reason string.
        reason: &'static str,
    },
    /// TLE parsing or SGP4 initialization failed.
    #[error("{object:?} satellite initialization failed: {source}")]
    Init {
        /// Object that failed initialization.
        object: TcaObject,
        /// SGP4 initialization error.
        source: Sgp4Error,
    },
    /// SGP4 propagation failed during the search.
    #[error("{object:?} satellite propagation failed: {source}")]
    Propagate {
        /// Object that failed propagation.
        object: TcaObject,
        /// SGP4 propagation error.
        source: Sgp4Error,
    },
    /// Numerical state-covariance propagation failed while transporting Pc inputs.
    #[error("{object:?} covariance propagation failed: {reason}")]
    CovariancePropagation {
        /// Object whose covariance propagation failed.
        object: TcaObject,
        /// Stable diagnostic from the numerical propagator.
        reason: String,
    },
    /// The shared event finder rejected configuration or predicate values.
    #[error(transparent)]
    EventFinder(#[from] EventFinderError),
    /// The existing conjunction Pc module rejected the TCA relative state.
    #[error(transparent)]
    Conjunction(#[from] ConjunctionError),
}

/// Find local TCA candidates between two TLE strings over an absolute JD window.
pub fn find_tca_candidates_from_tles(
    primary_line1: &str,
    primary_line2: &str,
    secondary_line1: &str,
    secondary_line2: &str,
    window_start: JulianDate,
    window_end: JulianDate,
    options: TcaFinderOptions,
) -> Result<Vec<TcaCandidate>, TcaError> {
    let primary =
        Satellite::from_tle(primary_line1, primary_line2).map_err(|source| TcaError::Init {
            object: TcaObject::Primary,
            source,
        })?;
    let secondary =
        Satellite::from_tle(secondary_line1, secondary_line2).map_err(|source| TcaError::Init {
            object: TcaObject::Secondary,
            source,
        })?;

    find_tca_candidates(&primary, &secondary, window_start, window_end, options)
}

/// Find local TCA candidates between two borrowed TLEs.
pub fn find_tca_candidates_between_tles(
    primary_tle: TcaTle<'_>,
    secondary_tle: TcaTle<'_>,
    window: TcaWindow,
    options: TcaFinderOptions,
) -> Result<Vec<TcaCandidate>, TcaError> {
    find_tca_candidates_from_tles(
        primary_tle.line1,
        primary_tle.line2,
        secondary_tle.line1,
        secondary_tle.line2,
        window.start,
        window.end,
        options,
    )
}

/// Find local TCA candidates between two initialized SGP4 satellites.
pub fn find_tca_candidates(
    primary: &Satellite,
    secondary: &Satellite,
    window_start: JulianDate,
    window_end: JulianDate,
    options: TcaFinderOptions,
) -> Result<Vec<TcaCandidate>, TcaError> {
    let span_seconds = validate_window(window_start, window_end)?;
    let options = validate_options(options)?;
    if span_seconds <= 0.0 {
        return Ok(Vec::new());
    }

    let finder = EventFinder::new(
        0.0,
        span_seconds,
        options.coarse_step_seconds,
        options.time_tolerance_seconds,
    )
    .map_err(TcaError::EventFinder)?;
    let predicate = RelativeRange::new(primary, secondary, window_start);
    let extrema = finder.find_extrema(&predicate).map_err(|error| {
        predicate
            .take_error()
            .unwrap_or(TcaError::EventFinder(error))
    })?;
    let minima = minimum_extrema_including_boundaries(
        &predicate,
        extrema,
        span_seconds,
        options.coarse_step_seconds,
    )?;

    minima
        .into_iter()
        .map(|event| tca_candidate_from_extremum(primary, secondary, window_start, event))
        .collect()
}

/// Find TCA candidates between two TLE strings and compute Pc at each TCA.
pub fn find_tca_conjunctions_from_tles(
    primary_tle: TcaTle<'_>,
    secondary_tle: TcaTle<'_>,
    window_start: JulianDate,
    window_end: JulianDate,
    tca_options: TcaFinderOptions,
    pc_options: TcaPcOptions,
) -> Result<Vec<TcaConjunction>, TcaError> {
    find_tca_candidates_from_tles(
        primary_tle.line1,
        primary_tle.line2,
        secondary_tle.line1,
        secondary_tle.line2,
        window_start,
        window_end,
        tca_options,
    )?
    .into_iter()
    .map(|candidate| tca_collision_probability(candidate, pc_options))
    .collect()
}

/// Find TCA candidates between two borrowed TLEs and compute Pc at each TCA.
pub fn find_tca_conjunctions_between_tles(
    primary_tle: TcaTle<'_>,
    secondary_tle: TcaTle<'_>,
    window: TcaWindow,
    tca_options: TcaFinderOptions,
    pc_options: TcaPcOptions,
) -> Result<Vec<TcaConjunction>, TcaError> {
    find_tca_conjunctions_from_tles(
        primary_tle,
        secondary_tle,
        window.start,
        window.end,
        tca_options,
        pc_options,
    )
}

/// Find TCA candidates between two initialized SGP4 satellites and compute Pc.
pub fn find_tca_conjunctions(
    primary: &Satellite,
    secondary: &Satellite,
    window_start: JulianDate,
    window_end: JulianDate,
    tca_options: TcaFinderOptions,
    pc_options: TcaPcOptions,
) -> Result<Vec<TcaConjunction>, TcaError> {
    find_tca_candidates(primary, secondary, window_start, window_end, tca_options)?
        .into_iter()
        .map(|candidate| tca_collision_probability(candidate, pc_options))
        .collect()
}

/// Find TCA candidates between two borrowed TLEs and compute Pc after propagating
/// each object's initial state covariance to the candidate TCA.
pub fn find_tca_conjunctions_with_propagated_covariance_from_tles(
    primary_tle: TcaTle<'_>,
    secondary_tle: TcaTle<'_>,
    window_start: JulianDate,
    window_end: JulianDate,
    tca_options: TcaFinderOptions,
    pc_options: TcaPropagatedCovariancePcOptions,
) -> Result<Vec<TcaConjunction>, TcaError> {
    let primary = satellite_from_tle(primary_tle, TcaObject::Primary)?;
    let secondary = satellite_from_tle(secondary_tle, TcaObject::Secondary)?;
    find_tca_conjunctions_with_propagated_covariance(
        &primary,
        &secondary,
        window_start,
        window_end,
        tca_options,
        pc_options,
    )
}

/// Find TCA candidates between two borrowed TLEs and compute Pc after
/// propagating each object's initial state covariance to TCA.
pub fn find_tca_conjunctions_with_propagated_covariance_between_tles(
    primary_tle: TcaTle<'_>,
    secondary_tle: TcaTle<'_>,
    window: TcaWindow,
    tca_options: TcaFinderOptions,
    pc_options: TcaPropagatedCovariancePcOptions,
) -> Result<Vec<TcaConjunction>, TcaError> {
    find_tca_conjunctions_with_propagated_covariance_from_tles(
        primary_tle,
        secondary_tle,
        window.start,
        window.end,
        tca_options,
        pc_options,
    )
}

/// Find TCA candidates between two initialized SGP4 satellites and compute Pc
/// after propagating each object's initial state covariance to each TCA.
pub fn find_tca_conjunctions_with_propagated_covariance(
    primary: &Satellite,
    secondary: &Satellite,
    window_start: JulianDate,
    window_end: JulianDate,
    tca_options: TcaFinderOptions,
    pc_options: TcaPropagatedCovariancePcOptions,
) -> Result<Vec<TcaConjunction>, TcaError> {
    find_tca_candidates(primary, secondary, window_start, window_end, tca_options)?
        .into_iter()
        .map(|candidate| {
            tca_collision_probability_with_propagated_covariance(
                primary, secondary, candidate, pc_options,
            )
        })
        .collect()
}

/// Serially screen a primary against a secondary catalog for threshold TCAs.
///
/// Returns one hit per local TCA whose miss distance is at or below
/// `miss_distance_threshold_km`, preserving the caller's secondary indices.
pub fn screen_tca_candidates_serial(
    primary: &Satellite,
    secondaries: &[Satellite],
    window_start: JulianDate,
    window_end: JulianDate,
    miss_distance_threshold_km: f64,
    options: TcaFinderOptions,
) -> Result<Vec<TcaScreeningHit>, TcaError> {
    screen_tca_candidates_batched(
        primary,
        secondaries,
        window_start,
        window_end,
        miss_distance_threshold_km,
        options,
        BatchMode::Serial,
    )
}

/// Parallel-screen a primary against a secondary catalog for threshold TCAs.
///
/// Results are in the same deterministic order as
/// [`screen_tca_candidates_serial`]: secondary catalog order, then TCA time
/// order within each secondary.
pub fn screen_tca_candidates_parallel(
    primary: &Satellite,
    secondaries: &[Satellite],
    window_start: JulianDate,
    window_end: JulianDate,
    miss_distance_threshold_km: f64,
    options: TcaFinderOptions,
) -> Result<Vec<TcaScreeningHit>, TcaError> {
    screen_tca_candidates_batched(
        primary,
        secondaries,
        window_start,
        window_end,
        miss_distance_threshold_km,
        options,
        BatchMode::Parallel,
    )
}

/// Serially screen a primary TLE against a borrowed secondary TLE catalog.
pub fn screen_tca_candidates_from_tle_catalog_serial(
    primary_tle: TcaTle<'_>,
    secondary_tles: &[TcaTle<'_>],
    window: TcaWindow,
    miss_distance_threshold_km: f64,
    options: TcaFinderOptions,
) -> Result<Vec<TcaScreeningHit>, TcaError> {
    let primary = satellite_from_tle(primary_tle, TcaObject::Primary)?;
    let secondaries = satellites_from_tles(secondary_tles)?;
    screen_tca_candidates_serial(
        &primary,
        &secondaries,
        window.start,
        window.end,
        miss_distance_threshold_km,
        options,
    )
}

/// Parallel-screen a primary TLE against a borrowed secondary TLE catalog.
pub fn screen_tca_candidates_from_tle_catalog_parallel(
    primary_tle: TcaTle<'_>,
    secondary_tles: &[TcaTle<'_>],
    window: TcaWindow,
    miss_distance_threshold_km: f64,
    options: TcaFinderOptions,
) -> Result<Vec<TcaScreeningHit>, TcaError> {
    let primary = satellite_from_tle(primary_tle, TcaObject::Primary)?;
    let secondaries = satellites_from_tles(secondary_tles)?;
    screen_tca_candidates_parallel(
        &primary,
        &secondaries,
        window.start,
        window.end,
        miss_distance_threshold_km,
        options,
    )
}

/// Serially screen a catalog and compute Pc for each threshold TCA.
pub fn screen_tca_conjunctions_serial(
    primary: &Satellite,
    secondaries: &[Satellite],
    window_start: JulianDate,
    window_end: JulianDate,
    miss_distance_threshold_km: f64,
    tca_options: TcaFinderOptions,
    pc_options: TcaPcOptions,
) -> Result<Vec<TcaScreeningConjunctionHit>, TcaError> {
    let hits = screen_tca_candidates_serial(
        primary,
        secondaries,
        window_start,
        window_end,
        miss_distance_threshold_km,
        tca_options,
    )?;
    screening_hits_to_conjunctions(hits, pc_options)
}

/// Serially screen a borrowed TLE catalog and compute Pc for each threshold TCA.
pub fn screen_tca_conjunctions_from_tle_catalog_serial(
    primary_tle: TcaTle<'_>,
    secondary_tles: &[TcaTle<'_>],
    window: TcaWindow,
    miss_distance_threshold_km: f64,
    tca_options: TcaFinderOptions,
    pc_options: TcaPcOptions,
) -> Result<Vec<TcaScreeningConjunctionHit>, TcaError> {
    let hits = screen_tca_candidates_from_tle_catalog_serial(
        primary_tle,
        secondary_tles,
        window,
        miss_distance_threshold_km,
        tca_options,
    )?;
    screening_hits_to_conjunctions(hits, pc_options)
}

/// Parallel-screen a catalog and compute Pc for each threshold TCA.
pub fn screen_tca_conjunctions_parallel(
    primary: &Satellite,
    secondaries: &[Satellite],
    window_start: JulianDate,
    window_end: JulianDate,
    miss_distance_threshold_km: f64,
    tca_options: TcaFinderOptions,
    pc_options: TcaPcOptions,
) -> Result<Vec<TcaScreeningConjunctionHit>, TcaError> {
    let hits = screen_tca_candidates_parallel(
        primary,
        secondaries,
        window_start,
        window_end,
        miss_distance_threshold_km,
        tca_options,
    )?;
    screening_hits_to_conjunctions(hits, pc_options)
}

/// Parallel-screen a borrowed TLE catalog and compute Pc for each threshold TCA.
pub fn screen_tca_conjunctions_from_tle_catalog_parallel(
    primary_tle: TcaTle<'_>,
    secondary_tles: &[TcaTle<'_>],
    window: TcaWindow,
    miss_distance_threshold_km: f64,
    tca_options: TcaFinderOptions,
    pc_options: TcaPcOptions,
) -> Result<Vec<TcaScreeningConjunctionHit>, TcaError> {
    let hits = screen_tca_candidates_from_tle_catalog_parallel(
        primary_tle,
        secondary_tles,
        window,
        miss_distance_threshold_km,
        tca_options,
    )?;
    screening_hits_to_conjunctions(hits, pc_options)
}

/// Serially screen a borrowed TLE catalog and compute Pc for each threshold TCA
/// after propagating each object's initial covariance to the TCA.
pub fn screen_tca_conjunctions_with_propagated_covariance_from_tle_catalog_serial(
    primary: TcaTleWithCovariance<'_>,
    secondaries: &[TcaTleWithCovariance<'_>],
    window: TcaWindow,
    miss_distance_threshold_km: f64,
    tca_options: TcaFinderOptions,
    pc_options: TcaPropagatedCovarianceOptions,
) -> Result<Vec<TcaScreeningConjunctionHit>, TcaError> {
    screen_tca_conjunctions_with_propagated_covariance_from_tle_catalog(
        primary,
        secondaries,
        window,
        miss_distance_threshold_km,
        tca_options,
        pc_options,
        BatchMode::Serial,
    )
}

/// Parallel-screen a borrowed TLE catalog and compute Pc for each threshold TCA
/// after propagating each object's initial covariance to the TCA.
pub fn screen_tca_conjunctions_with_propagated_covariance_from_tle_catalog_parallel(
    primary: TcaTleWithCovariance<'_>,
    secondaries: &[TcaTleWithCovariance<'_>],
    window: TcaWindow,
    miss_distance_threshold_km: f64,
    tca_options: TcaFinderOptions,
    pc_options: TcaPropagatedCovarianceOptions,
) -> Result<Vec<TcaScreeningConjunctionHit>, TcaError> {
    screen_tca_conjunctions_with_propagated_covariance_from_tle_catalog(
        primary,
        secondaries,
        window,
        miss_distance_threshold_km,
        tca_options,
        pc_options,
        BatchMode::Parallel,
    )
}

fn screen_tca_conjunctions_with_propagated_covariance_from_tle_catalog(
    primary: TcaTleWithCovariance<'_>,
    secondaries: &[TcaTleWithCovariance<'_>],
    window: TcaWindow,
    miss_distance_threshold_km: f64,
    tca_options: TcaFinderOptions,
    pc_options: TcaPropagatedCovarianceOptions,
    mode: BatchMode,
) -> Result<Vec<TcaScreeningConjunctionHit>, TcaError> {
    let primary_satellite = satellite_from_tle(primary.tle, TcaObject::Primary)?;
    let secondary_satellites = satellites_from_tles_with_covariance(secondaries)?;
    let hits = screen_tca_candidates_batched(
        &primary_satellite,
        &secondary_satellites,
        window.start,
        window.end,
        miss_distance_threshold_km,
        tca_options,
        mode,
    )?;
    screening_hits_to_propagated_covariance_conjunctions(
        hits,
        &primary_satellite,
        primary.covariance0,
        &secondary_satellites,
        secondaries,
        pc_options,
    )
}

/// Compute Pc for an already-refined TEME TCA candidate.
///
/// The candidate relative state is converted to GCRS before invoking the
/// conjunction Pc solver. Supplied position covariances must be in that same
/// GCRS frame.
pub fn tca_collision_probability(
    candidate: TcaCandidate,
    options: TcaPcOptions,
) -> Result<TcaConjunction, TcaError> {
    validate_tca_candidate_for_pc(candidate)?;
    let pc_state = tca_candidate_relative_state_for_pc(candidate)?;
    let primary_state = ConjunctionState {
        position_km: pc_state.relative_position_km,
        velocity_km_s: pc_state.relative_velocity_km_s,
        covariance_km2: options.covariances.primary_covariance_km2,
    };
    let secondary_state = ConjunctionState {
        position_km: [0.0; 3],
        velocity_km_s: [0.0; 3],
        covariance_km2: options.covariances.secondary_covariance_km2,
    };
    let collision_probability = collision_probability(
        &primary_state,
        &secondary_state,
        options.hard_body_radius_km,
        options.method,
    )?;

    Ok(TcaConjunction {
        candidate,
        collision_probability,
    })
}

/// Compute Pc for an already-refined TCA candidate after propagating each
/// object's initial state covariance to the TCA epoch.
pub fn tca_collision_probability_with_propagated_covariance(
    primary: &Satellite,
    secondary: &Satellite,
    candidate: TcaCandidate,
    options: TcaPropagatedCovariancePcOptions,
) -> Result<TcaConjunction, TcaError> {
    validate_tca_candidate_for_pc(candidate)?;
    let primary_covariance_km2 = propagate_position_covariance_to_tca(
        primary,
        TcaObject::Primary,
        options.primary_covariance0,
        candidate.tca_time,
        options,
    )?;
    let secondary_covariance_km2 = propagate_position_covariance_to_tca(
        secondary,
        TcaObject::Secondary,
        options.secondary_covariance0,
        candidate.tca_time,
        options,
    )?;
    let at_tca_options = TcaPcOptions::with_covariances(
        options.hard_body_radius_km,
        options.method,
        primary_covariance_km2,
        secondary_covariance_km2,
    );
    tca_collision_probability(candidate, at_tca_options)
}

#[derive(Debug, Clone, Copy)]
enum BatchMode {
    Serial,
    Parallel,
}

fn screen_tca_candidates_batched(
    primary: &Satellite,
    secondaries: &[Satellite],
    window_start: JulianDate,
    window_end: JulianDate,
    miss_distance_threshold_km: f64,
    options: TcaFinderOptions,
    mode: BatchMode,
) -> Result<Vec<TcaScreeningHit>, TcaError> {
    let span_seconds = validate_window(window_start, window_end)?;
    let options = validate_options(options)?;
    let miss_distance_threshold_km = validate_miss_distance_threshold(miss_distance_threshold_km)?;
    if span_seconds <= 0.0 || secondaries.is_empty() {
        return Ok(Vec::new());
    }

    let finder = EventFinder::new(
        0.0,
        span_seconds,
        options.coarse_step_seconds,
        options.time_tolerance_seconds,
    )
    .map_err(TcaError::EventFinder)?;
    let ranges = secondaries
        .iter()
        .map(|secondary| RelativeRange::new(primary, secondary, window_start))
        .collect::<Vec<_>>();
    let predicates = ranges.iter().collect::<Vec<_>>();
    let extrema_by_secondary = match mode {
        BatchMode::Serial => finder.find_extrema_batch_serial(&predicates),
        BatchMode::Parallel => finder.find_extrema_batch_parallel(&predicates),
    };

    let mut hits = Vec::new();
    for ((secondary_index, secondary), (range, extrema_result)) in secondaries
        .iter()
        .enumerate()
        .zip(ranges.iter().zip(extrema_by_secondary))
    {
        let extrema = extrema_result
            .map_err(|error| range.take_error().unwrap_or(TcaError::EventFinder(error)))?;
        let minima = minimum_extrema_including_boundaries(
            range,
            extrema,
            span_seconds,
            options.coarse_step_seconds,
        )?;
        for event in minima {
            let candidate = tca_candidate_from_extremum(primary, secondary, window_start, event)?;
            if candidate.miss_distance_km <= miss_distance_threshold_km {
                hits.push(TcaScreeningHit {
                    secondary_index,
                    candidate,
                });
            }
        }
    }

    Ok(hits)
}

fn minimum_extrema_including_boundaries(
    range: &RelativeRange<'_>,
    extrema: Vec<ExtremumEvent>,
    span_seconds: f64,
    coarse_step_seconds: f64,
) -> Result<Vec<ExtremumEvent>, TcaError> {
    minimum_extrema_including_boundaries_from_values(
        extrema,
        span_seconds,
        coarse_step_seconds,
        |time_seconds| finite_range_km_at(range, time_seconds),
    )
}

fn minimum_extrema_including_boundaries_from_values(
    extrema: Vec<ExtremumEvent>,
    span_seconds: f64,
    coarse_step_seconds: f64,
    mut value_at: impl FnMut(f64) -> Result<f64, TcaError>,
) -> Result<Vec<ExtremumEvent>, TcaError> {
    let (start_neighbor_time, end_neighbor_time) =
        extrema_boundary_neighbor_times(span_seconds, coarse_step_seconds);
    let start_value = value_at(0.0)?;
    let start_neighbor_value = value_at(start_neighbor_time)?;
    let end_neighbor_value = value_at(end_neighbor_time)?;
    let end_value = value_at(span_seconds)?;

    let mut minima = Vec::with_capacity(extrema.len() + 2);
    if is_strict_boundary_minimum(start_value, start_neighbor_value) {
        minima.push(ExtremumEvent {
            time_seconds: 0.0,
            value: start_value,
            kind: ExtremumKind::Minimum,
        });
    }
    minima.extend(
        extrema
            .into_iter()
            .filter(|event| event.kind == ExtremumKind::Minimum),
    );
    if is_strict_boundary_minimum(end_value, end_neighbor_value) {
        minima.push(ExtremumEvent {
            time_seconds: span_seconds,
            value: end_value,
            kind: ExtremumKind::Minimum,
        });
    }
    Ok(minima)
}

fn extrema_boundary_neighbor_times(span_seconds: f64, coarse_step_seconds: f64) -> (f64, f64) {
    debug_assert!(span_seconds > 0.0);
    debug_assert!(coarse_step_seconds > 0.0);

    let sample_iterations =
        ((span_seconds / coarse_step_seconds).ceil() as usize).saturating_add(1);
    let mut offset_seconds = 0.0;
    let mut sample_count = 0_usize;
    let mut start_neighbor_time = None;
    let mut end_neighbor_time = 0.0;

    for _ in 0..sample_iterations {
        if offset_seconds >= span_seconds {
            break;
        }
        let time_seconds = offset_seconds;
        if time_seconds >= span_seconds {
            break;
        }

        sample_count += 1;
        if sample_count == 2 {
            start_neighbor_time = Some(time_seconds);
        }
        end_neighbor_time = time_seconds;

        let next_offset_seconds = offset_seconds + coarse_step_seconds;
        debug_assert!(next_offset_seconds > offset_seconds);
        offset_seconds = next_offset_seconds;
    }

    if sample_count == 1 {
        let midpoint = span_seconds * 0.5;
        return (midpoint, midpoint);
    }

    (
        start_neighbor_time.expect("multi-sample boundary has a start neighbor"),
        end_neighbor_time,
    )
}

fn is_strict_boundary_minimum(boundary_value: f64, neighbor_value: f64) -> bool {
    neighbor_value - boundary_value > boundary_range_tolerance(boundary_value, neighbor_value)
}

fn boundary_range_tolerance(a: f64, b: f64) -> f64 {
    BOUNDARY_RANGE_ABS_TOL_KM.max(BOUNDARY_RANGE_REL_TOL * a.abs().max(b.abs()).max(1.0))
}

fn finite_range_km_at(range: &RelativeRange<'_>, time_seconds: f64) -> Result<f64, TcaError> {
    let value = range.range_km_at(time_seconds);
    if value.is_finite() {
        Ok(value)
    } else if let Some(error) = range.take_error() {
        Err(error)
    } else {
        Err(TcaError::EventFinder(EventFinderError::InvalidInput {
            field: "predicate",
            reason: "not finite",
        }))
    }
}

fn satellites_from_tles(tles: &[TcaTle<'_>]) -> Result<Vec<Satellite>, TcaError> {
    tles.iter()
        .map(|tle| satellite_from_tle(*tle, TcaObject::Secondary))
        .collect()
}

fn satellites_from_tles_with_covariance(
    tles: &[TcaTleWithCovariance<'_>],
) -> Result<Vec<Satellite>, TcaError> {
    tles.iter()
        .map(|tle| satellite_from_tle(tle.tle, TcaObject::Secondary))
        .collect()
}

fn satellite_from_tle(tle: TcaTle<'_>, object: TcaObject) -> Result<Satellite, TcaError> {
    Satellite::from_tle(tle.line1, tle.line2).map_err(|source| TcaError::Init { object, source })
}

fn screening_hits_to_conjunctions(
    hits: Vec<TcaScreeningHit>,
    pc_options: TcaPcOptions,
) -> Result<Vec<TcaScreeningConjunctionHit>, TcaError> {
    hits.into_iter()
        .map(|hit| {
            Ok(TcaScreeningConjunctionHit {
                secondary_index: hit.secondary_index,
                conjunction: tca_collision_probability(hit.candidate, pc_options)?,
            })
        })
        .collect()
}

fn screening_hits_to_propagated_covariance_conjunctions(
    hits: Vec<TcaScreeningHit>,
    primary: &Satellite,
    primary_covariance0: Covariance6,
    secondaries: &[Satellite],
    secondary_objects: &[TcaTleWithCovariance<'_>],
    pc_options: TcaPropagatedCovarianceOptions,
) -> Result<Vec<TcaScreeningConjunctionHit>, TcaError> {
    hits.into_iter()
        .map(|hit| {
            let secondary = &secondaries[hit.secondary_index];
            let secondary_covariance0 = secondary_objects[hit.secondary_index].covariance0;
            let conjunction = tca_collision_probability_with_propagated_covariance(
                primary,
                secondary,
                hit.candidate,
                pc_options.for_initial_covariances(primary_covariance0, secondary_covariance0),
            )?;
            Ok(TcaScreeningConjunctionHit {
                secondary_index: hit.secondary_index,
                conjunction,
            })
        })
        .collect()
}

fn tca_candidate_from_extremum(
    primary: &Satellite,
    secondary: &Satellite,
    window_start: JulianDate,
    event: ExtremumEvent,
) -> Result<TcaCandidate, TcaError> {
    let tca_time = add_seconds_to_julian_date(window_start, event.time_seconds);
    let state = relative_state_at(primary, secondary, tca_time)?;
    Ok(TcaCandidate {
        tca_time,
        tca_seconds_since_window_start: event.time_seconds,
        miss_distance_km: vec3::norm3(state.relative_position_km),
        relative_position_km: state.relative_position_km,
        relative_velocity_km_s: state.relative_velocity_km_s,
    })
}

#[derive(Debug, Clone, Copy)]
struct RelativeState {
    relative_position_km: [f64; 3],
    relative_velocity_km_s: [f64; 3],
}

fn relative_state_at(
    primary: &Satellite,
    secondary: &Satellite,
    time: JulianDate,
) -> Result<RelativeState, TcaError> {
    let primary_prediction = primary
        .propagate_jd(time)
        .map_err(|source| TcaError::Propagate {
            object: TcaObject::Primary,
            source,
        })?;
    let secondary_prediction =
        secondary
            .propagate_jd(time)
            .map_err(|source| TcaError::Propagate {
                object: TcaObject::Secondary,
                source,
            })?;

    Ok(RelativeState {
        relative_position_km: vec3::sub3(
            primary_prediction.position,
            secondary_prediction.position,
        ),
        relative_velocity_km_s: vec3::sub3(
            primary_prediction.velocity,
            secondary_prediction.velocity,
        ),
    })
}

fn tca_candidate_relative_state_for_pc(candidate: TcaCandidate) -> Result<RelativeState, TcaError> {
    let transform = teme_to_gcrs_state_transform_at(candidate.tca_time, TcaObject::Primary)?;
    let state = transform.transform_state(CartesianState::new(
        0.0,
        candidate.relative_position_km,
        candidate.relative_velocity_km_s,
    ));
    let state = RelativeState {
        relative_position_km: state.position_array(),
        relative_velocity_km_s: state.velocity_array(),
    };
    validate_relative_state_for_pc(state)?;
    Ok(state)
}

fn propagate_position_covariance_to_tca(
    satellite: &Satellite,
    object: TcaObject,
    covariance0: Covariance6,
    tca_time: JulianDate,
    options: TcaPropagatedCovariancePcOptions,
) -> Result<[[f64; 3]; 3], TcaError> {
    validate_julian_date_fields(tca_time, "tca_time.whole", "tca_time.fraction")?;
    let epoch_time = satellite.epoch_jd();
    let span_seconds = seconds_between_julian_dates(epoch_time, tca_time);
    validate::finite(span_seconds, "tca_time").map_err(map_input)?;

    let epoch_teme_to_gcrs = teme_to_gcrs_state_transform_at(epoch_time, object)?;
    let initial_state =
        epoch_teme_to_gcrs.transform_state(satellite_epoch_state(satellite, object)?);
    let covariance0 = epoch_teme_to_gcrs
        .transform_covariance(covariance0)
        .map_err(|source| TcaError::CovariancePropagation {
            object,
            reason: format!(
                "initial TEME->GCRS covariance transform failed: {}",
                covariance_error_reason(source)
            ),
        })?;
    let propagator = StatePropagator {
        initial: initial_state,
        force_model: options.force_model,
        integrator: options.integrator,
        options: options.integrator_options,
    };
    let (_, covariance_f) = propagator
        .propagate_state_with_covariance(covariance0, span_seconds)
        .map_err(|source| TcaError::CovariancePropagation {
            object,
            reason: source.to_string(),
        })?;

    Ok(covariance_f.position_covariance_km2())
}

fn satellite_epoch_state(
    satellite: &Satellite,
    object: TcaObject,
) -> Result<CartesianState, TcaError> {
    let prediction = satellite
        .propagate(MinutesSinceEpoch(0.0))
        .map_err(|source| TcaError::Propagate { object, source })?;
    Ok(CartesianState::new(
        0.0,
        prediction.position,
        prediction.velocity,
    ))
}

fn teme_to_gcrs_rotation_at(time: JulianDate, _object: TcaObject) -> Result<Mat3, TcaError> {
    validate_julian_date_fields(time, "frame_time.whole", "frame_time.fraction")?;
    let time_scales = time_scales_from_sgp4_julian_date(time)?;
    Ok(teme_to_gcrs_matrix(&time_scales, false))
}

fn teme_to_gcrs_state_transform_at(
    time: JulianDate,
    object: TcaObject,
) -> Result<TemeToGcrsStateTransform, TcaError> {
    let rotation = teme_to_gcrs_rotation_at(time, object)?;
    let before = teme_to_gcrs_rotation_at(
        add_seconds_to_julian_date(time, -FRAME_DERIVATIVE_STEP_SECONDS),
        object,
    )?;
    let after = teme_to_gcrs_rotation_at(
        add_seconds_to_julian_date(time, FRAME_DERIVATIVE_STEP_SECONDS),
        object,
    )?;

    Ok(TemeToGcrsStateTransform {
        rotation,
        rotation_derivative: centered_rotation_derivative(
            &before,
            &after,
            FRAME_DERIVATIVE_STEP_SECONDS,
        ),
    })
}

fn time_scales_from_sgp4_julian_date(time: JulianDate) -> Result<TimeScales, TcaError> {
    validate::finite(time.0, "julian_date").map_err(map_input)?;
    validate::finite(time.1, "julian_date").map_err(map_input)?;

    let mut jd_whole = time.0.floor();
    let mut jd_fraction = (time.0 - jd_whole) + time.1;
    let fraction_day_offset = jd_fraction.floor();
    jd_whole += fraction_day_offset;
    jd_fraction -= fraction_day_offset;

    let (jd_midnight, day_fraction) = if jd_fraction >= 0.5 {
        (jd_whole + 0.5, jd_fraction - 0.5)
    } else {
        (jd_whole - 0.5, jd_fraction + 0.5)
    };

    // Range-gate the calendar day (the canonical split-to-civil helper assumes a
    // valid domain); the discarded JDN equals the one the helper recomputes.
    julian_day_number_from_jd_midnight(jd_midnight)?;
    let (year, month, day, hour, minute, second) =
        civil_from_split_julian_date(jd_midnight, day_fraction);

    TimeScales::from_utc(
        year as i32,
        month as i32,
        day as i32,
        hour as i32,
        minute as i32,
        second,
    )
    .map_err(|_| TcaError::InvalidInput {
        field: "julian_date",
        reason: "out of range",
    })
}

fn julian_day_number_from_jd_midnight(jd_midnight: f64) -> Result<i64, TcaError> {
    let jdn = (jd_midnight + 0.5).round();
    let min_jdn = julian_day_number(0, 1, 1) as f64;
    let max_jdn = julian_day_number(9999, 12, 31) as f64;
    if jdn.is_finite() && (min_jdn..=max_jdn).contains(&jdn) {
        Ok(jdn as i64)
    } else {
        Err(TcaError::InvalidInput {
            field: "julian_date",
            reason: "out of range",
        })
    }
}

fn centered_rotation_derivative(before: &Mat3, after: &Mat3, step_seconds: f64) -> Mat3 {
    let scale = 1.0 / (2.0 * step_seconds);
    let mut derivative = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            derivative[i][j] = (after[i][j] - before[i][j]) * scale;
        }
    }
    derivative
}

#[derive(Clone, Copy)]
struct TemeToGcrsStateTransform {
    rotation: Mat3,
    rotation_derivative: Mat3,
}

impl TemeToGcrsStateTransform {
    fn transform_state(self, state: CartesianState) -> CartesianState {
        let position_teme = state.position_array();
        let velocity_teme = state.velocity_array();
        let position_gcrs = mat3_vec3_mul_unchecked(&self.rotation, &position_teme);
        let velocity_rotated = mat3_vec3_mul_unchecked(&self.rotation, &velocity_teme);
        let velocity_coupled = mat3_vec3_mul_unchecked(&self.rotation_derivative, &position_teme);
        CartesianState::new(
            state.epoch_tdb_seconds,
            position_gcrs,
            vec3::add3(velocity_rotated, velocity_coupled),
        )
    }

    fn transform_covariance(
        self,
        covariance: Covariance6,
    ) -> Result<Covariance6, Covariance6Error> {
        covariance.propagate_with_stm(&self.state_jacobian())
    }

    fn state_jacobian(self) -> Mat6 {
        let mut jacobian = [[0.0_f64; 6]; 6];
        for row in 0..3 {
            for col in 0..3 {
                jacobian[row][col] = self.rotation[row][col];
                jacobian[row + 3][col] = self.rotation_derivative[row][col];
                jacobian[row + 3][col + 3] = self.rotation[row][col];
            }
        }
        jacobian
    }
}

fn covariance_error_reason(error: Covariance6Error) -> &'static str {
    match error {
        Covariance6Error::NonFinite => "not finite",
        Covariance6Error::Asymmetric => "not symmetric",
        Covariance6Error::NotPositiveSemidefinite => "not positive semidefinite",
    }
}

struct RelativeRange<'a> {
    primary: &'a Satellite,
    secondary: &'a Satellite,
    window_start: JulianDate,
    first_error: Mutex<Option<TcaError>>,
}

impl<'a> RelativeRange<'a> {
    fn new(primary: &'a Satellite, secondary: &'a Satellite, window_start: JulianDate) -> Self {
        Self {
            primary,
            secondary,
            window_start,
            first_error: Mutex::new(None),
        }
    }

    fn range_km_at(&self, time_seconds: f64) -> f64 {
        let time = add_seconds_to_julian_date(self.window_start, time_seconds);
        match relative_state_at(self.primary, self.secondary, time) {
            Ok(state) => vec3::norm3(state.relative_position_km),
            Err(error) => {
                self.record_error(error);
                f64::NAN
            }
        }
    }

    fn record_error(&self, error: TcaError) {
        if let Ok(mut first_error) = self.first_error.lock() {
            if first_error.is_none() {
                *first_error = Some(error);
            }
        }
    }

    fn take_error(&self) -> Option<TcaError> {
        self.first_error
            .lock()
            .ok()
            .and_then(|mut first_error| first_error.take())
    }
}

impl ScalarEventPredicate for &RelativeRange<'_> {
    fn value_at(&self, time_seconds: f64) -> f64 {
        self.range_km_at(time_seconds)
    }
}

fn validate_options(options: TcaFinderOptions) -> Result<TcaFinderOptions, TcaError> {
    validate::positive_step(options.coarse_step_seconds, "coarse_step_seconds")
        .map_err(map_input)?;
    validate::positive_step(options.time_tolerance_seconds, "time_tolerance_seconds")
        .map_err(map_input)?;
    Ok(options)
}

fn validate_miss_distance_threshold(miss_distance_threshold_km: f64) -> Result<f64, TcaError> {
    validate::finite_nonneg(miss_distance_threshold_km, "miss_distance_threshold_km")
        .map_err(map_input)
}

fn validate_tca_candidate_for_pc(candidate: TcaCandidate) -> Result<(), TcaError> {
    validate_julian_date_fields(
        candidate.tca_time,
        "candidate.tca_time.whole",
        "candidate.tca_time.fraction",
    )?;
    validate::finite(
        candidate.tca_seconds_since_window_start,
        "candidate.tca_seconds_since_window_start",
    )
    .map_err(map_input)?;
    validate::finite_nonneg(candidate.miss_distance_km, "candidate.miss_distance_km")
        .map_err(map_input)?;
    validate_bounded_vec3(
        candidate.relative_position_km,
        MAX_TCA_RELATIVE_POSITION_KM,
        "candidate.relative_position_km",
    )?;
    validate_bounded_vec3(
        candidate.relative_velocity_km_s,
        MAX_TCA_RELATIVE_VELOCITY_KM_S,
        "candidate.relative_velocity_km_s",
    )?;
    Ok(())
}

fn validate_relative_state_for_pc(state: RelativeState) -> Result<(), TcaError> {
    validate_bounded_vec3(
        state.relative_position_km,
        MAX_TCA_RELATIVE_POSITION_KM,
        "relative_position_km",
    )?;
    validate_bounded_vec3(
        state.relative_velocity_km_s,
        MAX_TCA_RELATIVE_VELOCITY_KM_S,
        "relative_velocity_km_s",
    )?;
    Ok(())
}

fn validate_bounded_vec3(
    value: [f64; 3],
    max_abs: f64,
    field: &'static str,
) -> Result<(), TcaError> {
    validate::finite_vec3(value, field).map_err(map_input)?;
    if value.iter().any(|component| component.abs() > max_abs) {
        return Err(TcaError::InvalidInput {
            field,
            reason: "out of range",
        });
    }
    Ok(())
}

fn validate_window(window_start: JulianDate, window_end: JulianDate) -> Result<f64, TcaError> {
    validate_julian_date_fields(window_start, "window_start.whole", "window_start.fraction")?;
    validate_julian_date_fields(window_end, "window_end.whole", "window_end.fraction")?;
    let span_seconds = seconds_between_julian_dates(window_start, window_end);
    validate::finite_nonneg(span_seconds, "window_end").map_err(map_input)
}

fn validate_julian_date_fields(
    time: JulianDate,
    whole_field: &'static str,
    fraction_field: &'static str,
) -> Result<(), TcaError> {
    validate::finite(time.0, whole_field).map_err(map_input)?;
    validate::finite(time.1, fraction_field).map_err(map_input)?;
    Ok(())
}

fn seconds_between_julian_dates(start: JulianDate, end: JulianDate) -> f64 {
    (end.0 - start.0) * SECONDS_PER_DAY + (end.1 - start.1) * SECONDS_PER_DAY
}

fn add_seconds_to_julian_date(start: JulianDate, seconds: f64) -> JulianDate {
    let (whole, fraction) = split_julian_date_add_seconds(start.0, start.1, seconds);
    JulianDate(whole, fraction)
}

fn map_input(error: validate::FieldError) -> TcaError {
    TcaError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

/// Single-epoch catalog coarse screen.
///
/// Given object positions at one common epoch, return every unordered pair
/// `(i, j)` with `i < j` whose Euclidean separation is at or below
/// `miss_threshold_km`, paired with that separation `miss_km`. Results are
/// ordered by `i` ascending, then `j` ascending. This is the cheap geometric
/// pre-filter a catalog screener runs before evaluating collision probability
/// on the survivors; the distance uses [`vec3`] so callers (e.g. language
/// bindings) never reimplement it.
pub fn screen_catalog_pairs(
    positions_km: &[[f64; 3]],
    miss_threshold_km: f64,
) -> Vec<(usize, usize, f64)> {
    let mut pairs = Vec::new();
    for i in 0..positions_km.len() {
        for j in (i + 1)..positions_km.len() {
            let miss_km = vec3::norm3(vec3::sub3(positions_km[i], positions_km[j]));
            if miss_km <= miss_threshold_km {
                pairs.push((i, j, miss_km));
            }
        }
    }
    pairs
}

/// Screen a materialized state-vector catalog and evaluate Pc on survivors.
///
/// Results with a successful Pc below [`CatalogScreeningOptions::pc_threshold`]
/// are dropped. Candidate rows whose Pc evaluation fails are retained with the
/// error populated and sort as zero probability.
pub fn screen_state_vector_catalog(
    objects: &[CatalogStateVector],
    options: CatalogScreeningOptions,
) -> Vec<CatalogScreeningResult> {
    let positions = objects
        .iter()
        .map(|object| object.position_km)
        .collect::<Vec<_>>();
    let mut results = screen_catalog_pairs(&positions, options.miss_threshold_km)
        .into_iter()
        .map(|(i, j, miss_km)| {
            let obj1 = &objects[i];
            let obj2 = &objects[j];
            let candidate = CatalogScreeningCandidate {
                i,
                j,
                id1: obj1.id.clone(),
                id2: obj2.id.clone(),
                miss_km,
            };
            let state1 = ConjunctionState {
                position_km: obj1.position_km,
                velocity_km_s: obj1.velocity_km_s,
                covariance_km2: obj1.covariance_km2,
            };
            let state2 = ConjunctionState {
                position_km: obj2.position_km,
                velocity_km_s: obj2.velocity_km_s,
                covariance_km2: obj2.covariance_km2,
            };
            match collision_probability(
                &state1,
                &state2,
                obj1.hard_body_radius_km + obj2.hard_body_radius_km,
                options.method,
            ) {
                Ok(probability) => CatalogScreeningResult {
                    candidate,
                    collision: Some(CatalogCollision {
                        probability,
                        method: options.method,
                    }),
                    error: None,
                },
                Err(error) => CatalogScreeningResult {
                    candidate,
                    collision: None,
                    error: Some(error),
                },
            }
        })
        .filter(|result| {
            result
                .collision
                .as_ref()
                .is_none_or(|collision| collision.probability.pc >= options.pc_threshold)
        })
        .collect::<Vec<_>>();

    results.sort_by(|a, b| catalog_result_pc(b).total_cmp(&catalog_result_pc(a)));
    results
}

fn catalog_result_pc(result: &CatalogScreeningResult) -> f64 {
    result
        .collision
        .as_ref()
        .map_or(0.0, |collision| collision.probability.pc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::time::civil::civil_from_julian_day_number;

    #[test]
    fn screen_catalog_pairs_thresholds_and_orders() {
        // A-B are 0.1 km apart; C is ~1000 km away. Only A-B survives at 1 km.
        let positions = [[7000.0, 0.0, 0.0], [7000.1, 0.0, 0.0], [8000.0, 0.0, 0.0]];

        let pairs = screen_catalog_pairs(&positions, 1.0);
        assert_eq!(pairs.len(), 1);
        let (i, j, miss) = pairs[0];
        assert_eq!((i, j), (0, 1));
        assert!((miss - 0.1).abs() < 1.0e-9);

        // With a huge threshold every i<j pair is returned, i asc then j asc.
        let all = screen_catalog_pairs(&positions, 1.0e9);
        assert_eq!(
            all.iter().map(|(i, j, _)| (*i, *j)).collect::<Vec<_>>(),
            vec![(0, 1), (0, 2), (1, 2)]
        );

        // Degenerate inputs.
        assert!(screen_catalog_pairs(&[], 1.0).is_empty());
        assert!(screen_catalog_pairs(&[[0.0, 0.0, 0.0]], 1.0).is_empty());
    }

    #[test]
    fn state_vector_catalog_driver_matches_manual_prefilter_and_pc() {
        let cov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let objects = vec![
            CatalogStateVector {
                id: Some("A".to_string()),
                position_km: [7000.0, 0.0, 0.0],
                velocity_km_s: [0.0, 0.0, 0.0],
                covariance_km2: cov,
                hard_body_radius_km: 0.01,
            },
            CatalogStateVector {
                id: Some("B".to_string()),
                position_km: [7000.02, 0.0, 0.0],
                velocity_km_s: [0.0, 7.5, 0.0],
                covariance_km2: cov,
                hard_body_radius_km: 0.01,
            },
            CatalogStateVector {
                id: Some("C".to_string()),
                position_km: [7000.04, 0.0, 0.0],
                velocity_km_s: [0.0, 0.0, 0.0],
                covariance_km2: cov,
                hard_body_radius_km: 0.02,
            },
        ];
        let options = CatalogScreeningOptions {
            miss_threshold_km: 0.05,
            pc_threshold: 0.0,
            method: PcMethod::FosterEqualArea,
        };

        let driver = screen_state_vector_catalog(&objects, options);
        let positions = objects
            .iter()
            .map(|object| object.position_km)
            .collect::<Vec<_>>();
        let mut manual = screen_catalog_pairs(&positions, options.miss_threshold_km)
            .into_iter()
            .map(|(i, j, miss_km)| {
                let obj1 = &objects[i];
                let obj2 = &objects[j];
                let candidate = CatalogScreeningCandidate {
                    i,
                    j,
                    id1: obj1.id.clone(),
                    id2: obj2.id.clone(),
                    miss_km,
                };
                let state1 = ConjunctionState {
                    position_km: obj1.position_km,
                    velocity_km_s: obj1.velocity_km_s,
                    covariance_km2: obj1.covariance_km2,
                };
                let state2 = ConjunctionState {
                    position_km: obj2.position_km,
                    velocity_km_s: obj2.velocity_km_s,
                    covariance_km2: obj2.covariance_km2,
                };
                match collision_probability(
                    &state1,
                    &state2,
                    obj1.hard_body_radius_km + obj2.hard_body_radius_km,
                    options.method,
                ) {
                    Ok(probability) => CatalogScreeningResult {
                        candidate,
                        collision: Some(CatalogCollision {
                            probability,
                            method: options.method,
                        }),
                        error: None,
                    },
                    Err(error) => CatalogScreeningResult {
                        candidate,
                        collision: None,
                        error: Some(error),
                    },
                }
            })
            .collect::<Vec<_>>();
        manual.sort_by(|a, b| catalog_result_pc(b).total_cmp(&catalog_result_pc(a)));

        assert_eq!(driver, manual);
        assert!(driver
            .iter()
            .any(|result| result.error == Some(ConjunctionError::UndefinedFrame)));
    }

    const ISS_L1: &str = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993";
    const ISS_L2: &str = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106";
    const ISS_FAST_L1: &str =
        "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993";
    const ISS_FAST_L2: &str =
        "2 25544  51.6414 295.8524 0003435 262.6267 202.7868 15.64005638121106";
    const ISS_OPPOSITE_L2: &str =
        "2 25544  51.6414 295.8524 0003435 262.6267  24.2868 15.54005638121106";

    #[test]
    fn constructed_tles_find_expected_close_approach() {
        let primary = Satellite::from_tle(ISS_L1, ISS_L2).expect("primary TLE parses");
        let start = primary.epoch_jd();
        let end = add_seconds_to_julian_date(start, 12_000.0);
        let candidates = find_tca_candidates_from_tles(
            ISS_L1,
            ISS_L2,
            ISS_FAST_L1,
            ISS_FAST_L2,
            start,
            end,
            TcaFinderOptions {
                coarse_step_seconds: 30.0,
                time_tolerance_seconds: 1.0e-4,
            },
        )
        .expect("TCA search succeeds");

        assert_eq!(candidates.len(), 1);
        let best = candidates[0];

        assert_close(
            best.tca_seconds_since_window_start,
            3_599.834_762_793_019_3,
            2.0e-4,
        );
        assert_close(best.miss_distance_km, 28.942_032_135_766_88, 1.0e-9);
        assert_close(
            vec3::norm3(best.relative_position_km),
            best.miss_distance_km,
            1.0e-12,
        );
        assert!(vec3::norm3(best.relative_velocity_km_s) > 0.0);
    }

    #[test]
    fn tca_candidates_include_window_boundary_minima() {
        let primary = Satellite::from_tle(ISS_L1, ISS_L2).expect("primary TLE parses");
        let secondary =
            Satellite::from_tle(ISS_FAST_L1, ISS_FAST_L2).expect("secondary TLE parses");
        let options = TcaFinderOptions {
            coarse_step_seconds: 30.0,
            time_tolerance_seconds: 1.0e-4,
        };
        let full_start = primary.epoch_jd();
        let full_end = add_seconds_to_julian_date(full_start, 12_000.0);
        let interior = find_tca_candidates(&primary, &secondary, full_start, full_end, options)
            .expect("interior TCA search succeeds");
        assert_eq!(interior.len(), 1);
        let tca = interior[0];

        let start_boundary_end = add_seconds_to_julian_date(tca.tca_time, 600.0);
        let start_boundary = find_tca_candidates(
            &primary,
            &secondary,
            tca.tca_time,
            start_boundary_end,
            options,
        )
        .expect("start-boundary TCA search succeeds");

        assert_eq!(start_boundary.len(), 1);
        assert_close(start_boundary[0].tca_seconds_since_window_start, 0.0, 0.0);
        assert_close(
            start_boundary[0].miss_distance_km,
            tca.miss_distance_km,
            1.0e-9,
        );

        let end_boundary_start = add_seconds_to_julian_date(tca.tca_time, -600.0);
        let end_boundary = find_tca_candidates(
            &primary,
            &secondary,
            end_boundary_start,
            tca.tca_time,
            options,
        )
        .expect("end-boundary TCA search succeeds");

        assert_eq!(end_boundary.len(), 1);
        assert_close(
            end_boundary[0].tca_seconds_since_window_start,
            600.0,
            1.0e-8,
        );
        assert_close(
            end_boundary[0].miss_distance_km,
            tca.miss_distance_km,
            1.0e-9,
        );
    }

    #[test]
    fn tca_candidates_suppress_constant_range_boundary_minima() {
        let primary = Satellite::from_tle(ISS_L1, ISS_L2).expect("primary TLE parses");
        let secondary = Satellite::from_tle(ISS_L1, ISS_L2).expect("secondary TLE parses");
        let start = primary.epoch_jd();
        let end = add_seconds_to_julian_date(start, 900.0);
        let options = TcaFinderOptions {
            coarse_step_seconds: 30.0,
            time_tolerance_seconds: 1.0e-4,
        };

        let candidates = find_tca_candidates(&primary, &secondary, start, end, options)
            .expect("constant-range TCA search succeeds");

        assert!(
            candidates.is_empty(),
            "constant relative range should not emit boundary TCAs: {candidates:?}"
        );
    }

    #[test]
    fn tca_boundary_minimum_uses_last_partial_coarse_sample() {
        let minima = minimum_extrema_including_boundaries_from_values(
            Vec::new(),
            95.0,
            30.0,
            |time_seconds| {
                Ok(sampled_range_value(
                    time_seconds,
                    &[
                        (0.0, 10.0),
                        (30.0, 9.0),
                        (65.0, 9.0),
                        (90.0, 11.0),
                        (95.0, 10.0),
                    ],
                ))
            },
        )
        .expect("boundary classification succeeds");

        assert_eq!(minima.len(), 1);
        assert_close(minima[0].time_seconds, 95.0, 0.0);
        assert_close(minima[0].value, 10.0, 0.0);
        assert_eq!(minima[0].kind, ExtremumKind::Minimum);
    }

    #[test]
    fn tca_boundary_minimum_uses_repeated_addition_penultimate_sample() {
        let minima = minimum_extrema_including_boundaries_from_values(
            Vec::new(),
            1.05,
            0.1,
            |time_seconds| {
                Ok(sampled_range_value(
                    time_seconds,
                    &[
                        (0.0, 10.0),
                        (0.1, 9.0),
                        (0.999_999_999_999_999_9, 11.0),
                        (1.0, 9.0),
                        (1.05, 10.0),
                    ],
                ))
            },
        )
        .expect("boundary classification succeeds");

        assert_eq!(minima.len(), 1);
        assert_close(minima[0].time_seconds, 1.05, 0.0);
        assert_close(minima[0].value, 10.0, 0.0);
        assert_eq!(minima[0].kind, ExtremumKind::Minimum);
    }

    #[test]
    fn tca_boundary_minimum_uses_inserted_midpoint_sample() {
        let midpoint_minimum = ExtremumEvent {
            time_seconds: 5.0,
            value: 8.0,
            kind: ExtremumKind::Minimum,
        };
        let minima = minimum_extrema_including_boundaries_from_values(
            vec![midpoint_minimum],
            10.0,
            30.0,
            |time_seconds| {
                Ok(sampled_range_value(
                    time_seconds,
                    &[(0.0, 9.0), (5.0, 8.0), (10.0, 10.0)],
                ))
            },
        )
        .expect("boundary classification succeeds");

        assert_eq!(minima, vec![midpoint_minimum]);
    }

    #[test]
    fn tca_boundary_neighbor_times_keep_exact_multiple_window() {
        assert_eq!(extrema_boundary_neighbor_times(120.0, 30.0), (30.0, 90.0));
    }

    #[test]
    fn tca_candidate_pipeline_deduplicates_flat_bottom_close_approach() {
        let start = JulianDate(2_458_303.0, 0.0);
        let extrema = EventFinder::new(0.0, 3.0, 1.0, 1.0e-12)
            .expect("valid finder")
            .find_extrema(flat_bottom_range_km)
            .expect("finite flat-bottom range");
        let candidates: Vec<_> = extrema
            .into_iter()
            .filter(|event| event.kind == ExtremumKind::Minimum)
            .map(|event| TcaCandidate {
                tca_time: add_seconds_to_julian_date(start, event.time_seconds),
                tca_seconds_since_window_start: event.time_seconds,
                miss_distance_km: event.value,
                relative_position_km: [event.value, 0.0, 0.0],
                relative_velocity_km_s: [0.0; 3],
            })
            .collect();

        assert_eq!(candidates.len(), 1);
        assert!((1.0..=2.0).contains(&candidates[0].tca_seconds_since_window_start));
        assert_close(candidates[0].miss_distance_km, 1.0, 1.0e-12);
    }

    #[test]
    fn teme_to_gcrs_covariance_transport_uses_full_state_jacobian() {
        let transform = teme_to_gcrs_state_transform_at(
            JulianDate(2_458_303.0, 0.809_691_02),
            TcaObject::Primary,
        )
        .expect("TEME->GCRS state transform is valid");
        let covariance = Covariance6::from_diagonal([100.0, 144.0, 196.0, 1.0e-4, 1.5e-4, 2.0e-4])
            .expect("test covariance is valid");

        let actual = transform
            .transform_covariance(covariance)
            .expect("full-jacobian transform preserves covariance validity");
        let expected =
            manual_full_jacobian_covariance(covariance.as_matrix(), &transform.state_jacobian())
                .expect("manual full-jacobian covariance is valid");
        assert_covariance_close(actual.as_matrix(), expected.as_matrix(), 1.0e-10);

        let position_only = covariance
            .propagate_with_stm(&position_only_state_jacobian(&transform.rotation))
            .expect("position-only transform preserves covariance validity");
        assert_covariance_diff_exceeds(actual.as_matrix(), position_only.as_matrix(), 1.0e-12);
        assert!(actual.is_symmetric());
        assert!(actual.is_positive_semidefinite());
    }

    #[test]
    fn split_jd_preserves_civil_day_just_before_midnight() {
        let epsilon_days = 2.0e-10;
        let tca_time = JulianDate(2_458_303.0, 0.5 - epsilon_days);
        assert_eq!(tca_time.0 + tca_time.1, 2_458_303.5);

        let day_fraction = tca_time.1 + 0.5;
        let seconds_of_day = day_fraction * SECONDS_PER_DAY;
        let expected_second = seconds_of_day - 23.0 * 3600.0 - 59.0 * 60.0;
        let expected =
            TimeScales::from_utc(2018, 7, 3, 23, 59, expected_second).expect("valid UTC instant");
        let actual =
            time_scales_from_sgp4_julian_date(tca_time).expect("split JD converts to time scales");
        assert_eq!(actual, expected);

        let one_day_late =
            TimeScales::from_utc(2018, 7, 4, 23, 59, expected_second).expect("valid UTC instant");
        assert_ne!(actual, one_day_late);

        let actual_rotation =
            teme_to_gcrs_rotation_at(tca_time, TcaObject::Primary).expect("split rotation");
        let expected_rotation = teme_to_gcrs_matrix(&expected, false);
        assert_eq!(actual_rotation, expected_rotation);
        let one_day_late_rotation = teme_to_gcrs_matrix(&one_day_late, false);
        assert_matrix_diff_exceeds(&actual_rotation, &one_day_late_rotation, 1.0e-8);

        let candidate = TcaCandidate {
            tca_time,
            tca_seconds_since_window_start: 0.0,
            miss_distance_km: vec3::norm3([0.012, -0.017, 0.006]),
            relative_position_km: [0.012, -0.017, 0.006],
            relative_velocity_km_s: [0.09, 7.4, -1.2],
        };
        let pc_options = TcaPcOptions::with_covariances(
            0.010,
            PcMethod::Alfano2005,
            [
                [4.0e-4, 1.0e-5, -2.0e-5],
                [1.0e-5, 2.5e-5, 3.0e-6],
                [-2.0e-5, 3.0e-6, 9.0e-5],
            ],
            [[1.0e-5, 0.0, 0.0], [0.0, 6.4e-5, 0.0], [0.0, 0.0, 2.5e-5]],
        );
        let actual_pc = tca_collision_probability(candidate, pc_options)
            .expect("split-JD Pc")
            .collision_probability;
        let one_day_late_pc = tca_collision_probability(
            TcaCandidate {
                tca_time: JulianDate(tca_time.0 + 1.0, tca_time.1),
                ..candidate
            },
            pc_options,
        )
        .expect("one-day-late Pc")
        .collision_probability;
        assert_probability_diff_exceeds(actual_pc.pc, one_day_late_pc.pc, 1.0e-10);
    }

    #[test]
    fn split_jd_matches_summed_path_away_from_midnight() {
        let midday = JulianDate(2_458_303.0, 0.0);

        let actual =
            time_scales_from_sgp4_julian_date(midday).expect("midday converts to time scales");
        let summed = summed_time_scales_from_sgp4_julian_date(midday);
        assert_eq!(actual, summed);

        let actual_rotation =
            teme_to_gcrs_rotation_at(midday, TcaObject::Primary).expect("midday rotation");
        let summed_rotation = summed_teme_to_gcrs_rotation_at(midday);
        assert_eq!(actual_rotation, summed_rotation);
    }

    #[test]
    fn tca_pc_rejects_unconvertible_julian_dates_as_invalid_input() {
        let options = TcaPcOptions::with_default_covariance(0.010, PcMethod::Alfano2005);

        assert_eq!(
            tca_collision_probability(tca_pc_test_candidate(JulianDate(1.0e20, 0.0)), options),
            Err(TcaError::InvalidInput {
                field: "julian_date",
                reason: "out of range",
            })
        );
        assert!(matches!(
            tca_collision_probability(
                tca_pc_test_candidate(JulianDate(f64::INFINITY, 0.0)),
                options
            ),
            Err(TcaError::InvalidInput {
                reason: "not finite",
                ..
            })
        ));
    }

    #[test]
    fn tca_pc_rejects_invalid_candidate_relative_state_before_transform() {
        let options = TcaPcOptions::with_default_covariance(0.010, PcMethod::Alfano2005);
        let mut candidate = tca_pc_test_candidate(JulianDate(2_458_303.0, 0.0));
        candidate.relative_position_km[0] = f64::NAN;

        assert_eq!(
            tca_collision_probability(candidate, options),
            Err(TcaError::InvalidInput {
                field: "candidate.relative_position_km",
                reason: "not finite",
            })
        );

        candidate.relative_position_km = [MAX_TCA_RELATIVE_POSITION_KM * 2.0, 0.0, 0.0];
        assert_eq!(
            tca_collision_probability(candidate, options),
            Err(TcaError::InvalidInput {
                field: "candidate.relative_position_km",
                reason: "out of range",
            })
        );
    }

    #[test]
    fn tca_pc_in_range_julian_date_matches_summed_reference_path() {
        let candidate = tca_pc_test_candidate(JulianDate(2_458_303.0, 0.0));
        let options = TcaPcOptions::with_default_covariance(0.010, PcMethod::Alfano2005);

        let actual_state =
            tca_candidate_relative_state_for_pc(candidate).expect("in-range frame transform");
        let expected_transform = summed_teme_to_gcrs_state_transform_at(candidate.tca_time);
        let expected_state = expected_transform.transform_state(CartesianState::new(
            0.0,
            candidate.relative_position_km,
            candidate.relative_velocity_km_s,
        ));
        assert_eq!(
            actual_state.relative_position_km,
            expected_state.position_array()
        );
        assert_eq!(
            actual_state.relative_velocity_km_s,
            expected_state.velocity_array()
        );

        let actual = tca_collision_probability(candidate, options).expect("in-range TCA Pc");
        let expected_primary = ConjunctionState {
            position_km: expected_state.position_array(),
            velocity_km_s: expected_state.velocity_array(),
            covariance_km2: options.covariances.primary_covariance_km2,
        };
        let expected_secondary = ConjunctionState {
            position_km: [0.0; 3],
            velocity_km_s: [0.0; 3],
            covariance_km2: options.covariances.secondary_covariance_km2,
        };
        let expected = collision_probability(
            &expected_primary,
            &expected_secondary,
            options.hard_body_radius_km,
            options.method,
        )
        .expect("summed-reference TCA Pc");
        assert_eq!(actual.candidate, candidate);
        assert_eq!(actual.collision_probability, expected);
    }

    #[test]
    fn invalid_window_is_rejected() {
        let start = JulianDate(2_458_303.0, 0.5);
        let end = JulianDate(2_458_303.0, 0.4);
        assert_eq!(
            find_tca_candidates_from_tles(
                ISS_L1,
                ISS_L2,
                ISS_FAST_L1,
                ISS_FAST_L2,
                start,
                end,
                TcaFinderOptions::default(),
            ),
            Err(TcaError::InvalidInput {
                field: "window_end",
                reason: "negative",
            })
        );
    }

    #[test]
    fn screening_returns_only_threshold_breaches_and_parallel_matches_serial() {
        let primary = Satellite::from_tle(ISS_L1, ISS_L2).expect("primary TLE parses");
        let far = Satellite::from_tle(ISS_FAST_L1, ISS_OPPOSITE_L2).expect("far TLE parses");
        let close = Satellite::from_tle(ISS_FAST_L1, ISS_FAST_L2).expect("close TLE parses");
        let secondaries = [far, close];
        let start = primary.epoch_jd();
        let end = add_seconds_to_julian_date(start, 12_000.0);
        let options = TcaFinderOptions {
            coarse_step_seconds: 30.0,
            time_tolerance_seconds: 1.0e-4,
        };

        let serial =
            screen_tca_candidates_serial(&primary, &secondaries, start, end, 30.0, options)
                .expect("serial screening succeeds");
        let parallel =
            screen_tca_candidates_parallel(&primary, &secondaries, start, end, 30.0, options)
                .expect("parallel screening succeeds");

        assert_eq!(serial, parallel);
        assert_eq!(serial.len(), 1);

        let hit = serial[0];
        assert_eq!(hit.secondary_index, 1);
        assert_close(
            hit.candidate.tca_seconds_since_window_start,
            3_599.834_762_793_019_3,
            2.0e-4,
        );
        assert_close(
            hit.candidate.miss_distance_km,
            28.942_032_135_766_88,
            1.0e-9,
        );
        assert!(hit.candidate.miss_distance_km <= 30.0);
    }

    #[test]
    fn tca_pc_matches_direct_conjunction_call_with_supplied_covariance() {
        let primary = Satellite::from_tle(ISS_L1, ISS_L2).expect("primary TLE parses");
        let secondary =
            Satellite::from_tle(ISS_FAST_L1, ISS_FAST_L2).expect("secondary TLE parses");
        let start = primary.epoch_jd();
        let end = add_seconds_to_julian_date(start, 12_000.0);
        let tca_options = TcaFinderOptions {
            coarse_step_seconds: 30.0,
            time_tolerance_seconds: 1.0e-4,
        };
        let candidates = find_tca_candidates(&primary, &secondary, start, end, tca_options)
            .expect("TCA search succeeds");
        assert_eq!(candidates.len(), 1);

        let candidate = candidates[0];
        let primary_covariance_km2 = [[0.04, 0.001, 0.0], [0.001, 0.09, 0.002], [0.0, 0.002, 0.16]];
        let secondary_covariance_km2 = [
            [0.01, 0.0, 0.0],
            [0.0, 0.0225, 0.0005],
            [0.0, 0.0005, 0.0625],
        ];
        let pc_options = TcaPcOptions::with_covariances(
            0.020,
            PcMethod::Alfano2005,
            primary_covariance_km2,
            secondary_covariance_km2,
        );

        let conjunction =
            tca_collision_probability(candidate, pc_options).expect("Pc is defined at TCA");
        let pc_state =
            tca_candidate_relative_state_for_pc(candidate).expect("candidate converts to GCRS");
        let direct_primary = ConjunctionState {
            position_km: pc_state.relative_position_km,
            velocity_km_s: pc_state.relative_velocity_km_s,
            covariance_km2: primary_covariance_km2,
        };
        let direct_secondary = ConjunctionState {
            position_km: [0.0; 3],
            velocity_km_s: [0.0; 3],
            covariance_km2: secondary_covariance_km2,
        };
        let direct = collision_probability(
            &direct_primary,
            &direct_secondary,
            pc_options.hard_body_radius_km,
            pc_options.method,
        )
        .expect("direct Pc is defined");

        assert_eq!(conjunction.candidate, candidate);
        assert_eq!(conjunction.collision_probability, direct);

        let from_finder =
            find_tca_conjunctions(&primary, &secondary, start, end, tca_options, pc_options)
                .expect("TCA Pc search succeeds");
        assert_eq!(from_finder, vec![conjunction]);
    }

    #[test]
    fn tca_pc_with_initial_covariances_matches_direct_pc_after_manual_propagation() {
        let primary = Satellite::from_tle(ISS_L1, ISS_L2).expect("primary TLE parses");
        let secondary =
            Satellite::from_tle(ISS_FAST_L1, ISS_FAST_L2).expect("secondary TLE parses");
        let start = primary.epoch_jd();
        let end = add_seconds_to_julian_date(start, 12_000.0);
        let tca_options = TcaFinderOptions {
            coarse_step_seconds: 30.0,
            time_tolerance_seconds: 1.0e-4,
        };
        let candidates = find_tca_candidates(&primary, &secondary, start, end, tca_options)
            .expect("TCA search succeeds");
        assert_eq!(candidates.len(), 1);

        let candidate = candidates[0];
        let primary_covariance0 =
            Covariance6::from_diagonal([100.0, 144.0, 196.0, 1.0e-4, 1.5e-4, 2.0e-4]).unwrap();
        let secondary_covariance0 =
            Covariance6::from_diagonal([64.0, 81.0, 121.0, 8.0e-5, 9.0e-5, 1.0e-4]).unwrap();
        let pc_options = TcaPropagatedCovariancePcOptions::new(
            0.020,
            PcMethod::Alfano2005,
            primary_covariance0,
            secondary_covariance0,
        )
        .with_covariance_propagator(
            ForceModelKind::two_body_j2(),
            IntegratorKind::Rk4,
            IntegratorOptions {
                initial_step: 30.0,
                ..IntegratorOptions::default()
            },
        );

        let conjunction = tca_collision_probability_with_propagated_covariance(
            &primary, &secondary, candidate, pc_options,
        )
        .expect("propagated covariance Pc is defined");

        let primary_covariance_km2 =
            manual_position_covariance_at_tca(&primary, primary_covariance0, candidate, pc_options);
        let secondary_covariance_km2 = manual_position_covariance_at_tca(
            &secondary,
            secondary_covariance0,
            candidate,
            pc_options,
        );
        assert_ne!(
            primary_covariance_km2,
            primary_covariance0.position_covariance_km2()
        );
        assert_ne!(
            secondary_covariance_km2,
            secondary_covariance0.position_covariance_km2()
        );
        assert_matrix_diff_exceeds(
            &primary_covariance_km2,
            &pre_fix_position_covariance_at_tca(
                &primary,
                primary_covariance0,
                candidate,
                pc_options,
            ),
            1.0e-6,
        );
        assert_matrix_diff_exceeds(
            &secondary_covariance_km2,
            &pre_fix_position_covariance_at_tca(
                &secondary,
                secondary_covariance0,
                candidate,
                pc_options,
            ),
            1.0e-6,
        );

        let pc_state =
            tca_candidate_relative_state_for_pc(candidate).expect("candidate converts to GCRS");
        let direct_primary = ConjunctionState {
            position_km: pc_state.relative_position_km,
            velocity_km_s: pc_state.relative_velocity_km_s,
            covariance_km2: primary_covariance_km2,
        };
        let direct_secondary = ConjunctionState {
            position_km: [0.0; 3],
            velocity_km_s: [0.0; 3],
            covariance_km2: secondary_covariance_km2,
        };
        let direct = collision_probability(
            &direct_primary,
            &direct_secondary,
            pc_options.hard_body_radius_km,
            pc_options.method,
        )
        .expect("direct propagated-covariance Pc is defined");

        assert_eq!(conjunction.candidate, candidate);
        assert_eq!(conjunction.collision_probability, direct);

        let unconverted_primary = ConjunctionState {
            position_km: candidate.relative_position_km,
            velocity_km_s: candidate.relative_velocity_km_s,
            covariance_km2: primary_covariance_km2,
        };
        let unconverted = collision_probability(
            &unconverted_primary,
            &direct_secondary,
            pc_options.hard_body_radius_km,
            pc_options.method,
        )
        .expect("unconverted TEME candidate still produces a Pc");
        assert_probability_diff_exceeds(direct.pc, unconverted.pc, 1.0e-20);

        let from_finder = find_tca_conjunctions_with_propagated_covariance(
            &primary,
            &secondary,
            start,
            end,
            tca_options,
            pc_options,
        )
        .expect("propagated covariance TCA Pc search succeeds");
        assert_eq!(from_finder, vec![conjunction]);
    }

    #[test]
    fn tle_catalog_screening_propagates_initial_covariances_to_pc() {
        let primary_satellite = Satellite::from_tle(ISS_L1, ISS_L2).expect("primary TLE parses");
        let start = primary_satellite.epoch_jd();
        let window = TcaWindow::from_start_and_duration_seconds(start, 12_000.0)
            .expect("window duration is valid");
        let tca_options = TcaFinderOptions {
            coarse_step_seconds: 30.0,
            time_tolerance_seconds: 1.0e-4,
        };
        let primary_covariance0 =
            Covariance6::from_diagonal([100.0, 144.0, 196.0, 1.0e-4, 1.5e-4, 2.0e-4]).unwrap();
        let secondary_covariance0 =
            Covariance6::from_diagonal([64.0, 81.0, 121.0, 8.0e-5, 9.0e-5, 1.0e-4]).unwrap();
        let primary = TcaTleWithCovariance::new(ISS_L1, ISS_L2, primary_covariance0);
        let secondary = TcaTleWithCovariance::new(ISS_FAST_L1, ISS_FAST_L2, secondary_covariance0);
        let pc_options = TcaPropagatedCovarianceOptions::new(0.020, PcMethod::Alfano2005)
            .with_covariance_propagator(
                ForceModelKind::two_body_j2(),
                IntegratorKind::Rk4,
                IntegratorOptions {
                    initial_step: 30.0,
                    ..IntegratorOptions::default()
                },
            );

        let secondaries = [secondary];
        let serial = screen_tca_conjunctions_with_propagated_covariance_from_tle_catalog_serial(
            primary,
            &secondaries,
            window,
            30.0,
            tca_options,
            pc_options,
        )
        .expect("serial propagated-covariance screening succeeds");
        let parallel =
            screen_tca_conjunctions_with_propagated_covariance_from_tle_catalog_parallel(
                primary,
                &secondaries,
                window,
                30.0,
                tca_options,
                pc_options,
            )
            .expect("parallel propagated-covariance screening succeeds");

        assert_eq!(serial, parallel);
        assert_eq!(serial.len(), 1);
        assert_eq!(serial[0].secondary_index, 0);
        assert!(serial[0].conjunction.candidate.miss_distance_km <= 30.0);
        assert!(serial[0].conjunction.collision_probability.pc.is_finite());
        assert!((0.0..=1.0).contains(&serial[0].conjunction.collision_probability.pc));

        let pairwise = find_tca_conjunctions_with_propagated_covariance_between_tles(
            primary.tle,
            secondary.tle,
            window,
            tca_options,
            pc_options.for_initial_covariances(primary_covariance0, secondary_covariance0),
        )
        .expect("pairwise propagated-covariance TCA Pc search succeeds");
        assert_eq!(serial[0].conjunction, pairwise[0]);
    }

    fn flat_bottom_range_km(time_seconds: f64) -> f64 {
        if time_seconds < 1.0 {
            2.0 - time_seconds
        } else if time_seconds <= 2.0 {
            1.0
        } else {
            time_seconds - 1.0
        }
    }

    fn sampled_range_value(time_seconds: f64, samples: &[(f64, f64)]) -> f64 {
        samples
            .iter()
            .find_map(|(sample_time, value)| {
                (time_seconds.to_bits() == sample_time.to_bits()).then_some(*value)
            })
            .unwrap_or_else(|| panic!("unexpected boundary sample time {time_seconds}"))
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} differs from {expected} by more than {tolerance}"
        );
    }

    fn assert_matrix_diff_exceeds(actual: &Mat3, expected: &Mat3, threshold: f64) {
        let mut max_diff = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                max_diff = max_diff.max((actual[i][j] - expected[i][j]).abs());
            }
        }
        assert!(
            max_diff > threshold,
            "matrix diff {max_diff} did not exceed {threshold}"
        );
    }

    fn assert_probability_diff_exceeds(actual: f64, expected: f64, threshold: f64) {
        let diff = (actual - expected).abs();
        assert!(
            diff > threshold,
            "probability diff {diff} did not exceed {threshold}: {actual} vs {expected}"
        );
    }

    fn assert_covariance_close(actual: &Mat6, expected: &Mat6, tolerance: f64) {
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (actual[i][j] - expected[i][j]).abs() <= tolerance,
                    "covariance[{i}][{j}] = {} differs from {} by more than {tolerance}",
                    actual[i][j],
                    expected[i][j]
                );
            }
        }
    }

    fn assert_covariance_diff_exceeds(actual: &Mat6, expected: &Mat6, threshold: f64) {
        let mut max_diff = 0.0_f64;
        for i in 0..6 {
            for j in 0..6 {
                max_diff = max_diff.max((actual[i][j] - expected[i][j]).abs());
            }
        }
        assert!(
            max_diff > threshold,
            "covariance diff {max_diff} did not exceed {threshold}"
        );
    }

    fn summed_time_scales_from_sgp4_julian_date(time: JulianDate) -> TimeScales {
        let jd_total = time.0 + time.1;
        let mut jd_midnight = (jd_total - 0.5).floor() + 0.5;
        let mut day_fraction = (time.0 - jd_midnight) + time.1;
        if !(0.0..1.0).contains(&day_fraction) {
            let day_offset = day_fraction.floor();
            jd_midnight += day_offset;
            day_fraction -= day_offset;
        }

        let jdn = (jd_midnight + 0.5).round() as i64;
        let (year, month, day) = civil_from_julian_day_number(jdn);
        let seconds_of_day = day_fraction * SECONDS_PER_DAY;
        let hour = (seconds_of_day / 3600.0).floor() as i32;
        let minute = ((seconds_of_day - f64::from(hour) * 3600.0) / 60.0).floor() as i32;
        let second = seconds_of_day - f64::from(hour) * 3600.0 - f64::from(minute) * 60.0;

        TimeScales::from_utc(year as i32, month as i32, day as i32, hour, minute, second)
            .expect("summed JD test instant is valid")
    }

    fn summed_teme_to_gcrs_rotation_at(time: JulianDate) -> Mat3 {
        let time_scales = summed_time_scales_from_sgp4_julian_date(time);
        teme_to_gcrs_matrix(&time_scales, false)
    }

    fn summed_teme_to_gcrs_state_transform_at(time: JulianDate) -> TemeToGcrsStateTransform {
        let rotation = summed_teme_to_gcrs_rotation_at(time);
        let before = summed_teme_to_gcrs_rotation_at(add_seconds_to_julian_date(
            time,
            -FRAME_DERIVATIVE_STEP_SECONDS,
        ));
        let after = summed_teme_to_gcrs_rotation_at(add_seconds_to_julian_date(
            time,
            FRAME_DERIVATIVE_STEP_SECONDS,
        ));

        TemeToGcrsStateTransform {
            rotation,
            rotation_derivative: centered_rotation_derivative(
                &before,
                &after,
                FRAME_DERIVATIVE_STEP_SECONDS,
            ),
        }
    }

    fn tca_pc_test_candidate(tca_time: JulianDate) -> TcaCandidate {
        TcaCandidate {
            tca_time,
            tca_seconds_since_window_start: 0.0,
            miss_distance_km: vec3::norm3([0.012, -0.017, 0.006]),
            relative_position_km: [0.012, -0.017, 0.006],
            relative_velocity_km_s: [0.09, 7.4, -1.2],
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn manual_full_jacobian_covariance(
        covariance: &Mat6,
        jacobian: &Mat6,
    ) -> Result<Covariance6, Covariance6Error> {
        let mut transformed = [[0.0_f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..6 {
                    for l in 0..6 {
                        transformed[i][j] += jacobian[i][k] * covariance[k][l] * jacobian[j][l];
                    }
                }
            }
        }
        symmetrize6_for_test(&mut transformed);
        Covariance6::try_from_matrix(transformed)
    }

    fn position_only_state_jacobian(rotation: &Mat3) -> Mat6 {
        let mut jacobian = [[0.0_f64; 6]; 6];
        for row in 0..3 {
            for col in 0..3 {
                jacobian[row][col] = rotation[row][col];
                jacobian[row + 3][col + 3] = rotation[row][col];
            }
        }
        jacobian
    }

    #[allow(clippy::needless_range_loop)]
    fn symmetrize6_for_test(matrix: &mut Mat6) {
        for i in 0..6 {
            for j in (i + 1)..6 {
                let avg = 0.5 * (matrix[i][j] + matrix[j][i]);
                matrix[i][j] = avg;
                matrix[j][i] = avg;
            }
        }
    }

    fn manual_position_covariance_at_tca(
        satellite: &Satellite,
        covariance0: Covariance6,
        candidate: TcaCandidate,
        options: TcaPropagatedCovariancePcOptions,
    ) -> [[f64; 3]; 3] {
        let epoch_state = satellite
            .propagate(MinutesSinceEpoch(0.0))
            .expect("satellite propagates at its epoch");
        let epoch = satellite.epoch_jd();
        let span_seconds = seconds_between_julian_dates(epoch, candidate.tca_time);
        let epoch_teme_to_gcrs = teme_to_gcrs_state_transform_at(epoch, TcaObject::Primary)
            .expect("epoch frame transform");
        let initial = epoch_teme_to_gcrs.transform_state(CartesianState::new(
            0.0,
            epoch_state.position,
            epoch_state.velocity,
        ));
        let covariance0 = epoch_teme_to_gcrs
            .transform_covariance(covariance0)
            .expect("initial covariance rotates into GCRS");
        let propagator = StatePropagator {
            initial,
            force_model: options.force_model,
            integrator: options.integrator,
            options: options.integrator_options,
        };
        let (_, covariance_f) = propagator
            .propagate_state_with_covariance(covariance0, span_seconds)
            .expect("manual covariance propagation succeeds");
        covariance_f.position_covariance_km2()
    }

    fn pre_fix_position_covariance_at_tca(
        satellite: &Satellite,
        covariance0: Covariance6,
        candidate: TcaCandidate,
        options: TcaPropagatedCovariancePcOptions,
    ) -> [[f64; 3]; 3] {
        let epoch_state = satellite
            .propagate(MinutesSinceEpoch(0.0))
            .expect("satellite propagates at its epoch");
        let epoch = satellite.epoch_jd();
        let span_seconds = seconds_between_julian_dates(epoch, candidate.tca_time);
        let propagator = StatePropagator {
            initial: CartesianState::new(0.0, epoch_state.position, epoch_state.velocity),
            force_model: options.force_model,
            integrator: options.integrator,
            options: options.integrator_options,
        };
        let (_, covariance_f) = propagator
            .propagate_state_with_covariance(covariance0, span_seconds)
            .expect("pre-fix covariance propagation succeeds");
        covariance_f.position_covariance_km2()
    }
}
