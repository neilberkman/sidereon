//! Code-differential GNSS (DGPS) pseudorange corrections.
//!
//! This module owns the language-independent DGNSS modeling that used to live
//! in Sidereon: base-station pseudorange correction generation, rover observation
//! pairing, and the direct corrected-observation SPP orchestration.

use std::collections::BTreeMap;

use crate::astro::math::vec3;
use crate::constants::C_M_S;
use crate::id::GnssSatelliteId;
use crate::observables::{
    predict, ObservableEphemerisSource, ObservablesError, ObservablesInputErrorKind, PredictOptions,
};
use crate::spp::{self, EphemerisSource, Observation, ReceiverSolution, SolveInputs, SppError};
use crate::validate;

/// A single code pseudorange observation keyed by its RINEX/SP3 satellite token.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeObservation {
    /// Satellite token, e.g. `"G21"`.
    pub satellite_id: String,
    /// Measured pseudorange in meters.
    pub pseudorange_m: f64,
}

impl CodeObservation {
    /// Construct a pseudorange observation from a satellite token and range.
    pub fn new(satellite_id: impl Into<String>, pseudorange_m: f64) -> Self {
        Self {
            satellite_id: satellite_id.into(),
            pseudorange_m,
        }
    }
}

/// Result of applying base corrections to rover observations.
#[derive(Debug, Clone, PartialEq)]
pub struct AppliedCorrections {
    /// Corrected rover pseudoranges, in rover-observation order.
    pub corrected: Vec<CodeObservation>,
    /// Rover satellite tokens that had no matching correction, in rover order.
    pub dropped: Vec<String>,
}

/// DGNSS rover solve output.
#[derive(Debug, Clone)]
pub struct PositionSolution {
    /// Corrected rover SPP solution.
    pub solution: ReceiverSolution,
    /// Rover minus base ECEF vector in meters.
    pub baseline_vector_m: [f64; 3],
    /// Baseline length in meters.
    pub baseline_m: f64,
    /// Rover satellite tokens without matching base correction.
    pub dropped_sats: Vec<String>,
}

/// Error from the DGNSS position orchestration.
#[derive(Debug, Clone)]
pub enum DgnssError {
    /// A public DGNSS input was malformed, non-finite, or outside its physical
    /// domain.
    InvalidInput {
        /// The invalid input field.
        field: &'static str,
        /// The validation failure category.
        reason: &'static str,
    },
    /// Corrected-observation SPP solve failed.
    Spp(SppError),
}

impl core::fmt::Display for DgnssError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid DGNSS input {field}: {reason}")
            }
            Self::Spp(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for DgnssError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spp(err) => Some(err),
            Self::InvalidInput { .. } => None,
        }
    }
}

impl From<SppError> for DgnssError {
    fn from(value: SppError) -> Self {
        Self::Spp(value)
    }
}

/// Compute per-satellite pseudorange corrections from a surveyed base station.
///
/// The correction is `PRC = pr_base - (range_base - c * sat_clock)`, with the
/// range and satellite clock coming from the same light-time/Sagnac observable
/// predictor used by the SPP pipeline. Observations with malformed satellite
/// tokens or unavailable orbit/clock data are skipped, matching Sidereon'
/// historical "cannot correct this satellite" behavior.
pub fn pseudorange_corrections(
    source: &dyn ObservableEphemerisSource,
    base_position_m: [f64; 3],
    base_observations: &[CodeObservation],
    t_rx_j2000_s: f64,
) -> Result<BTreeMap<String, f64>, DgnssError> {
    validate_base_position(base_position_m)?;
    validate::finite(t_rx_j2000_s, "t_rx_j2000_s").map_err(dgnss_invalid_input)?;

    let mut corrections = BTreeMap::new();
    for obs in base_observations {
        let pseudorange_m =
            validate::finite_positive(obs.pseudorange_m, "base_observation.pseudorange_m")
                .map_err(dgnss_invalid_input)?;
        let Some(sat) = sat_from_token(&obs.satellite_id) else {
            continue;
        };
        let pred = match predict(
            source,
            sat,
            base_position_m,
            t_rx_j2000_s,
            PredictOptions::default(),
        ) {
            Ok(pred) => pred,
            Err(ObservablesError::InvalidInput { field, kind }) => {
                return Err(invalid_observable_input(field, kind));
            }
            Err(_) => continue,
        };
        let Some(sat_clock_s) = pred.sat_clock_s else {
            continue;
        };
        let geometric_range_m =
            validate::finite(pred.geometric_range_m, "predicted.geometric_range_m")
                .map_err(dgnss_invalid_input)?;
        let sat_clock_s =
            validate::finite(sat_clock_s, "predicted.sat_clock_s").map_err(dgnss_invalid_input)?;
        let modeled_base_m =
            validate::finite(geometric_range_m - C_M_S * sat_clock_s, "modeled_base_m")
                .map_err(dgnss_invalid_input)?;
        let correction_m =
            validate::finite(pseudorange_m - modeled_base_m, "pseudorange_correction_m")
                .map_err(dgnss_invalid_input)?;
        corrections.insert(obs.satellite_id.clone(), correction_m);
    }
    Ok(corrections)
}

/// Apply base pseudorange corrections to rover observations by satellite token.
///
/// The output order follows the rover observation order. Corrections without a
/// rover observation are ignored; rover observations without a correction are
/// reported in `dropped`.
pub fn apply_corrections(
    rover_observations: &[CodeObservation],
    corrections: &BTreeMap<String, f64>,
) -> Result<AppliedCorrections, DgnssError> {
    for prc_m in corrections.values() {
        validate::finite(*prc_m, "pseudorange_correction_m").map_err(dgnss_invalid_input)?;
    }

    let mut corrected = Vec::with_capacity(rover_observations.len());
    let mut dropped = Vec::new();
    for obs in rover_observations {
        let pseudorange_m =
            validate::finite_positive(obs.pseudorange_m, "rover_observation.pseudorange_m")
                .map_err(dgnss_invalid_input)?;
        match corrections.get(&obs.satellite_id) {
            Some(prc_m) => {
                let corrected_pseudorange_m =
                    validate::finite_positive(pseudorange_m - prc_m, "corrected_pseudorange_m")
                        .map_err(dgnss_invalid_input)?;
                corrected.push(CodeObservation::new(
                    obs.satellite_id.clone(),
                    corrected_pseudorange_m,
                ));
            }
            None => dropped.push(obs.satellite_id.clone()),
        }
    }
    Ok(AppliedCorrections { corrected, dropped })
}

/// Compute DGNSS corrections, apply them to rover observations, and solve SPP.
///
/// `solve_inputs` supplies the receive-time scalars, initial guess, meteorology,
/// Klobuchar coefficients, and optional Huber configuration. Its observations
/// and atmospheric-correction flags are replaced: DGNSS solves the corrected
/// rover pseudoranges with ionosphere/troposphere disabled because the
/// differential already removed common path delays.
pub fn solve_position<S>(
    source: &S,
    base_position_m: [f64; 3],
    base_observations: &[CodeObservation],
    rover_observations: &[CodeObservation],
    mut solve_inputs: SolveInputs,
    with_geodetic: bool,
) -> Result<PositionSolution, DgnssError>
where
    S: ObservableEphemerisSource + EphemerisSource,
{
    let corrections = pseudorange_corrections(
        source,
        base_position_m,
        base_observations,
        solve_inputs.t_rx_j2000_s,
    )?;
    let applied = apply_corrections(rover_observations, &corrections)?;
    solve_inputs.observations = applied
        .corrected
        .iter()
        .filter_map(|obs| {
            sat_from_token(&obs.satellite_id).map(|satellite_id| Observation {
                satellite_id,
                pseudorange_m: obs.pseudorange_m,
            })
        })
        .collect();
    solve_inputs.corrections = spp::Corrections::NONE;

    let solution = spp::solve(source, &solve_inputs, with_geodetic)?;
    let pos = solution.position.as_array();
    let baseline_vector_m = vec3::sub3(pos, base_position_m);
    let baseline_m = vec3::norm3(baseline_vector_m);

    Ok(PositionSolution {
        solution,
        baseline_vector_m,
        baseline_m,
        dropped_sats: applied.dropped,
    })
}

fn sat_from_token(token: &str) -> Option<GnssSatelliteId> {
    token.parse::<GnssSatelliteId>().ok()
}

fn validate_base_position(base_position_m: [f64; 3]) -> Result<(), DgnssError> {
    const FIELDS: [&str; 3] = [
        "base_position_m[0]",
        "base_position_m[1]",
        "base_position_m[2]",
    ];
    for (value, field) in base_position_m.into_iter().zip(FIELDS) {
        validate::finite(value, field).map_err(dgnss_invalid_input)?;
    }
    Ok(())
}

fn dgnss_invalid_input(error: validate::FieldError) -> DgnssError {
    DgnssError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

fn invalid_observable_input(field: &'static str, kind: ObservablesInputErrorKind) -> DgnssError {
    DgnssError::InvalidInput {
        field,
        reason: observable_input_reason(kind),
    }
}

fn observable_input_reason(kind: ObservablesInputErrorKind) -> &'static str {
    match kind {
        ObservablesInputErrorKind::NonFinite => "not finite",
        ObservablesInputErrorKind::NotPositive => "not positive",
        ObservablesInputErrorKind::Negative => "negative",
        ObservablesInputErrorKind::OutOfRange => "out of range",
        ObservablesInputErrorKind::Missing => "missing",
        ObservablesInputErrorKind::FloatParse => "invalid float",
        ObservablesInputErrorKind::IntParse => "invalid integer",
        ObservablesInputErrorKind::InvalidCivilDate => "invalid civil date",
        ObservablesInputErrorKind::InvalidCivilTime => "invalid civil time",
    }
}
