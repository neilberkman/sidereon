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
    pseudorange_transmit_epoch_j2000_s, pseudorange_transmit_geometry, ObservableEphemerisSource,
    ObservablesError, ObservablesInputErrorKind, TransmitGeometry,
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
    /// The ephemeris source refused a base satellite's state because producing
    /// it reads UT1 outside the UT1 table under a strict UT1 policy. The
    /// corrections fail rather than skipping that satellite.
    Ut1OutsideCoverage(crate::astro::time::DegradeReason),
}

impl core::fmt::Display for DgnssError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid DGNSS input {field}: {reason}")
            }
            Self::Spp(err) => write!(f, "{err}"),
            Self::Ut1OutsideCoverage(reason) => {
                write!(
                    f,
                    "the ephemeris source refused a base satellite state: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for DgnssError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spp(err) => Some(err),
            Self::InvalidInput { .. } | Self::Ut1OutsideCoverage(_) => None,
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
/// range and satellite clock formed as the SPP model forms them and as RTKLIB
/// `zdres` forms them for a base station: the transmission epoch placed from the
/// base pseudorange as RTKLIB `satposs` places it, the `geodist` range with the
/// Sagnac term, and the satellite clock at that epoch less the source's
/// single-frequency group delay. Observations with malformed satellite
/// tokens or unavailable orbit/clock data are skipped, matching Sidereon'
/// historical "cannot correct this satellite" behavior. A state the source
/// refuses because producing it reads UT1 outside the UT1 table under a strict
/// UT1 policy is not skipped: it fails with [`DgnssError::Ut1OutsideCoverage`].
/// [`pseudorange_corrections_validated`] also reports a departure accepted
/// under a permissive policy.
pub fn pseudorange_corrections(
    source: &dyn ObservableEphemerisSource,
    base_position_m: [f64; 3],
    base_observations: &[CodeObservation],
    t_rx_j2000_s: f64,
) -> Result<BTreeMap<String, f64>, DgnssError> {
    pseudorange_corrections_validated(source, base_position_m, base_observations, t_rx_j2000_s)
        .map(|corrections| corrections.value)
}

/// [`pseudorange_corrections`] with the first UT1 departure the source
/// accepted, under a permissive UT1 policy, while producing a base satellite
/// state, in [`Validated::degraded`](crate::astro::time::Validated::degraded).
pub fn pseudorange_corrections_validated(
    source: &dyn ObservableEphemerisSource,
    base_position_m: [f64; 3],
    base_observations: &[CodeObservation],
    t_rx_j2000_s: f64,
) -> Result<crate::astro::time::Validated<BTreeMap<String, f64>>, DgnssError> {
    let tracked = spp::Ut1Tracked::new(source);
    let corrections =
        tracked_pseudorange_corrections(&tracked, base_position_m, base_observations, t_rx_j2000_s);
    if let Some(reason) = tracked.refusal() {
        return Err(DgnssError::Ut1OutsideCoverage(reason));
    }
    Ok(crate::astro::time::Validated {
        value: corrections?,
        degraded: tracked.departure(),
    })
}

fn tracked_pseudorange_corrections(
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
        // The base pseudorange places its transmission epoch as RTKLIB `satposs` places
        // it, and the state there is ranged with `geodist`, as RTKLIB `zdres` models a
        // base station.
        let (pred, group_delay) =
            match base_transmit_geometry(source, sat, base_position_m, t_rx_j2000_s, pseudorange_m)
            {
                Ok(placed) => placed,
                Err(ObservablesError::InvalidInput { field, kind }) => {
                    return Err(invalid_observable_input(field, kind));
                }
                Err(ObservablesError::Ephemeris(crate::Error::Ut1OutsideCoverage(reason))) => {
                    return Err(DgnssError::Ut1OutsideCoverage(reason));
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
        // The predicted clock is RTKLIB's `satposs` clock, without the broadcast group
        // delay. The base pseudorange is single-frequency, and the rover is solved by the
        // SPP model, which takes the group delay from the clock; the base model takes it
        // the same way, so the correction carries none and the two stay consistent.
        // A product clock (SP3, RINEX CLK) takes the relativistic term RTKLIB `peph2pos`
        // applies for positioning, as the rover's SPP model applies it.
        // Where the term cannot be formed the satellite has no usable clock, as
        // `peph2pos` returns no state, and gets no correction.
        let sat_clock_s = match source.clock_relativity_s(sat, pred.transmit_time_j2000_s) {
            crate::spp::ClockRelativity::NotApplicable => sat_clock_s,
            crate::spp::ClockRelativity::Term(relativity_s) => sat_clock_s + relativity_s,
            crate::spp::ClockRelativity::Unavailable => continue,
        };
        let sat_clock_s = match group_delay {
            Some(group_delay_s) => sat_clock_s - group_delay_s,
            None => sat_clock_s,
        };
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

/// Transmit-time geometry of a base pseudorange, with the single-frequency group delay of
/// the record it comes from: the transmission epoch placed from the pseudorange
/// ([`pseudorange_transmit_epoch_j2000_s`]) and the `geodist` range there with the Sagnac
/// term ([`pseudorange_transmit_geometry`]), the record selected at the reception epoch as
/// RTKLIB `satposs` selects it.
fn base_transmit_geometry(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    base_position_m: [f64; 3],
    t_rx_j2000_s: f64,
    pseudorange_m: f64,
) -> Result<(TransmitGeometry, Option<f64>), ObservablesError> {
    let t_tx = pseudorange_transmit_epoch_j2000_s(source, sat, t_rx_j2000_s, pseudorange_m)?;
    let geometry =
        pseudorange_transmit_geometry(source, sat, base_position_m, t_rx_j2000_s, t_tx, true)?;
    let group_delay = source
        .try_observable_state_group_delay_selected_at_j2000_s(sat, t_tx, t_rx_j2000_s)?
        .value
        .1;
    Ok((geometry, group_delay))
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
///
/// Each rover satellite's transmission epoch is placed from the rover's raw
/// pseudorange, as RTKLIB `rtkpos` calls `satposs` with the rover's own
/// observations; the corrected pseudorange forms only the residual. The correction
/// carries the base receiver clock, so placing from the corrected code would put each
/// satellite that clock's worth of its motion away from where the rover's signal left it.
///
/// A UT1 refusal fails the solve: [`DgnssError::Ut1OutsideCoverage`] for a
/// base satellite, [`SppError::Ut1OutsideCoverage`] for a rover satellite. A
/// departure accepted under a permissive UT1 policy, on either side, is
/// reported in the solution's
/// [`SolutionMetadata::ut1_degraded`](crate::spp::SolutionMetadata::ut1_degraded).
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
    let corrections = pseudorange_corrections_validated(
        source,
        base_position_m,
        base_observations,
        solve_inputs.t_rx_j2000_s,
    )?;
    let applied = apply_corrections(rover_observations, &corrections.value)?;
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
    let placement: BTreeMap<GnssSatelliteId, f64> = rover_observations
        .iter()
        .filter_map(|obs| {
            sat_from_token(&obs.satellite_id).map(|satellite_id| (satellite_id, obs.pseudorange_m))
        })
        .collect();

    let mut solution = spp::solve_placed(source, &solve_inputs, &placement, with_geodetic)?;
    // The rover solve reports its own departure; a base-correction departure
    // also shaped this position.
    solution.metadata.ut1_degraded = solution.metadata.ut1_degraded.or(corrections.degraded);
    scale_position_covariance(&mut solution.position_covariance, 2.0);
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

fn scale_position_covariance(covariance: &mut crate::dop::PositionCovariance, scale: f64) {
    for row in 0..3 {
        for col in 0..3 {
            covariance.ecef_m2[row][col] *= scale;
            covariance.enu_m2[row][col] *= scale;
        }
    }
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
