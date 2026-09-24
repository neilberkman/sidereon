use crate::astro::math::special::{normal_q, normal_q_inv};
use crate::id::GnssSystem;
use crate::integrity::{gain_matrix_enu_from_design_rows, IntegrityError};
pub(crate) use crate::integrity::{metric_sigma, GainMatrix};

use super::{clock_system_for_row, validate_probability, AraimError, AraimGeometry};

/// Supplies per-row range sigmas for ARAIM protection and reliability designs.
///
/// Implementations must return externally supplied, positive finite values.
/// These values are treated as model input. They are never estimated from
/// residuals or measured ranges by the reliability design path.
pub trait ProtectionModel {
    /// Integrity one-sigma range error for a geometry row, meters.
    fn sigma_int_m(&self, row: &super::AraimRow) -> Result<f64, AraimError>;

    /// Accuracy one-sigma range error for a geometry row, meters.
    fn sigma_acc_m(&self, row: &super::AraimRow) -> Result<f64, AraimError>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ProtectionEquationTerm {
    pub prior: f64,
    pub sigma_m: f64,
    pub bias_m: f64,
    pub threshold_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ProtectionLevelSolution {
    pub value_m: f64,
    pub converged: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FalseAlertAxis {
    Horizontal,
    Vertical,
}

pub(crate) fn gain_matrix_enu(
    geometry: &AraimGeometry,
    weights: &[f64],
) -> Result<GainMatrix, AraimError> {
    gain_matrix_enu_for_clock_systems(geometry, weights, &geometry.clock_systems)
}

pub(crate) fn gain_matrix_enu_for_clock_systems(
    geometry: &AraimGeometry,
    weights: &[f64],
    clock_systems: &[GnssSystem],
) -> Result<GainMatrix, AraimError> {
    if weights.len() != geometry.rows.len() || clock_systems.is_empty() {
        return Err(AraimError::InsufficientGeometry);
    }
    let n_state = 3 + clock_systems.len();
    if weights.iter().filter(|&&weight| weight > 0.0).count() < n_state {
        return Err(AraimError::InsufficientGeometry);
    }
    if weights
        .iter()
        .any(|&weight| !weight.is_finite() || weight < 0.0)
    {
        return Err(AraimError::NumericalFailure);
    }

    let mut design_rows: Vec<Vec<f64>> = Vec::with_capacity(geometry.rows.len());
    for (row, &weight) in geometry.rows.iter().zip(weights) {
        let design = if weight > 0.0 {
            design_row(clock_systems, row)?
        } else {
            vec![0.0; n_state]
        };
        design_rows.push(design);
    }

    gain_matrix_enu_from_design_rows(&design_rows, weights, geometry.receiver, n_state)
        .map_err(map_integrity_error)
}

pub(crate) fn metric_bias(gain_row: &[f64], biases_m: &[f64]) -> f64 {
    gain_row
        .iter()
        .zip(biases_m)
        .map(|(&s, &bias)| s.abs() * bias)
        .sum()
}

pub(crate) fn separation_sigma(
    gain_row: &[f64],
    fault_free_gain_row: &[f64],
    sigmas_m: &[f64],
) -> f64 {
    gain_row
        .iter()
        .zip(fault_free_gain_row)
        .zip(sigmas_m)
        .map(|((&sk, &s0), &sigma)| {
            let ds = sk - s0;
            ds * ds * sigma * sigma
        })
        .sum::<f64>()
        .sqrt()
}

pub(crate) fn k_false_alert(
    pfa: f64,
    n_fault_modes: usize,
    axis: FalseAlertAxis,
) -> Result<f64, AraimError> {
    if n_fault_modes == 0 {
        return Ok(0.0);
    }
    if !validate_probability(pfa, false) {
        return Err(AraimError::InvalidAllocation);
    }
    let denominator_per_mode = match axis {
        // WG-C Reference ADD v3.0 Eq. (26).
        FalseAlertAxis::Horizontal => 4.0,
        // WG-C Reference ADD v3.0 Eq. (27).
        FalseAlertAxis::Vertical => 2.0,
    };
    normal_q_inv(pfa / (denominator_per_mode * n_fault_modes as f64))
        .ok_or(AraimError::InvalidAllocation)
}

pub(crate) fn solve_protection_level(
    fault_free: ProtectionEquationTerm,
    fault_terms: &[ProtectionEquationTerm],
    integrity_target: f64,
    tolerance_m: f64,
) -> Result<ProtectionLevelSolution, AraimError> {
    if !validate_probability(integrity_target, false)
        || tolerance_m <= 0.0
        || !tolerance_m.is_finite()
    {
        return Err(AraimError::InvalidAllocation);
    }
    validate_term(fault_free)?;
    for term in fault_terms {
        validate_term(*term)?;
    }

    if fault_terms.is_empty() {
        let value_m = normal_q_inv(integrity_target * 0.5).ok_or(AraimError::InvalidAllocation)?
            * fault_free.sigma_m
            + fault_free.bias_m;
        return Ok(ProtectionLevelSolution {
            value_m,
            converged: true,
        });
    }

    let mut lo = 0.0;
    if protection_lhs(lo, fault_free, fault_terms) <= integrity_target {
        return Ok(ProtectionLevelSolution {
            value_m: 0.0,
            converged: true,
        });
    }

    let mut hi = normal_q_inv(integrity_target * 0.5).ok_or(AraimError::InvalidAllocation)?
        * fault_free.sigma_m
        + fault_free.bias_m;
    if hi <= lo || !hi.is_finite() {
        hi = 1.0;
    }
    let mut expanded = 0usize;
    while protection_lhs(hi, fault_free, fault_terms) > integrity_target {
        hi *= 2.0;
        expanded += 1;
        if !hi.is_finite() || expanded > 100 {
            return Err(AraimError::NumericalFailure);
        }
    }

    let mut converged = false;
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if protection_lhs(mid, fault_free, fault_terms) > integrity_target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= tolerance_m {
            converged = true;
            break;
        }
    }
    let value_m = if converged {
        // Return a conservative PL that is still within the configured root
        // tolerance. This follows the ADD requirement to output an upper PL
        // within TOLPL of the equation solution.
        lo + tolerance_m
    } else {
        hi
    };
    Ok(ProtectionLevelSolution { value_m, converged })
}

fn protection_lhs(
    y_m: f64,
    fault_free: ProtectionEquationTerm,
    fault_terms: &[ProtectionEquationTerm],
) -> f64 {
    // WG-C Reference ADD v3.0 Eqs. (31)-(32): the fault-free term is two-sided,
    // and monitored fault terms use the modified Q of Eqs. (7)-(8).
    let mut value = 2.0 * normal_q((y_m - fault_free.bias_m) / fault_free.sigma_m);
    for term in fault_terms {
        value += term.prior * modified_q((y_m - term.threshold_m - term.bias_m) / term.sigma_m);
    }
    value
}

fn modified_q(argument: f64) -> f64 {
    if argument <= 0.0 {
        1.0
    } else {
        normal_q(argument)
    }
}

fn validate_term(term: ProtectionEquationTerm) -> Result<(), AraimError> {
    if term.prior.is_finite()
        && term.prior >= 0.0
        && term.sigma_m.is_finite()
        && term.sigma_m > 0.0
        && term.bias_m.is_finite()
        && term.bias_m >= 0.0
        && term.threshold_m.is_finite()
        && term.threshold_m >= 0.0
    {
        Ok(())
    } else {
        Err(AraimError::NumericalFailure)
    }
}

pub(crate) fn design_row(
    clock_systems: &[GnssSystem],
    row: &super::AraimRow,
) -> Result<Vec<f64>, AraimError> {
    let mut design = vec![0.0_f64; 3 + clock_systems.len()];
    let los = row.line_of_sight;
    if !los.e_x.is_finite()
        || !los.e_y.is_finite()
        || !los.e_z.is_finite()
        || !row.elevation_rad.is_finite()
    {
        return Err(AraimError::InsufficientGeometry);
    }
    design[0] = -los.e_x;
    design[1] = -los.e_y;
    design[2] = -los.e_z;
    let clock_system = clock_system_for_row(row.system, clock_systems);
    let clock_idx = clock_systems
        .iter()
        .position(|&system| system == clock_system)
        .ok_or(AraimError::InsufficientGeometry)?;
    design[3 + clock_idx] = 1.0;
    Ok(design)
}

fn map_integrity_error(error: IntegrityError) -> AraimError {
    match error {
        IntegrityError::Singular => AraimError::InsufficientGeometry,
        IntegrityError::InvalidInput { .. }
        | IntegrityError::NonFinite
        | IntegrityError::NotPositiveSemidefinite
        | IntegrityError::InvalidProbability { .. } => AraimError::NumericalFailure,
    }
}
