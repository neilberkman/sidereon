use std::collections::BTreeSet;

use super::fault_modes::{enumerate_fault_modes_checked, FaultHypothesis};
use super::ism::Ism;
use super::protection::{
    gain_matrix_enu, gain_matrix_enu_for_clock_systems, k_false_alert, metric_bias, metric_sigma,
    separation_sigma, solve_protection_level, FalseAlertAxis, ProtectionEquationTerm,
    ProtectionLevelSolution,
};
use super::{
    clock_system_for_row, validate_probability, AraimError, AraimGeometry, IntegrityAllocation,
};
use crate::id::{GnssSatelliteId, GnssSystem};

// WG-C Reference ADD v3.0 Table 3 and Appendix B, Eq. (68): TOLPL is 5e-2 m.
const HORIZONTAL_PL_TOL_M: f64 = 5.0e-2;
const VERTICAL_PL_TOL_M: f64 = 1.0e-4;

/// Per-hypothesis ARAIM monitor data.
#[derive(Debug, Clone, PartialEq)]
pub struct FaultMode {
    /// Satellites excluded by this mode.
    pub excluded: Vec<GnssSatelliteId>,
    /// Constellation excluded by this mode, if any.
    pub excluded_constellation: Option<GnssSystem>,
    /// Fault prior probability for this mode.
    pub prior: f64,
    /// Integrity sigma in local `[east, north, up]`, meters.
    pub sigma_int_enu_m: [f64; 3],
    /// Nominal bias bound in local `[east, north, up]`, meters.
    pub bias_enu_m: [f64; 3],
    /// Separation monitor threshold in local `[east, north, up]`, meters.
    pub threshold_enu_m: [f64; 3],
    /// True when the subset geometry is full-rank.
    pub monitorable: bool,
}

/// ARAIM protection-level result.
#[derive(Debug, Clone, PartialEq)]
pub struct AraimResult {
    /// True when ARAIM met the allocation and all PL roots converged.
    pub available: bool,
    /// Horizontal protection level, meters.
    pub hpl_m: f64,
    /// Vertical protection level, meters.
    pub vpl_m: f64,
    /// All-in-view horizontal accuracy sigma, meters.
    pub sigma_acc_h_m: f64,
    /// All-in-view vertical accuracy sigma, meters.
    pub sigma_acc_v_m: f64,
    /// Effective monitor threshold, meters.
    pub emt_m: f64,
    /// Per-mode monitor data, including the fault-free mode first.
    pub fault_modes: Vec<FaultMode>,
    /// Unenumerated plus unmonitorable fault probability mass.
    pub p_unmonitored: f64,
    /// True when ARAIM met the allocation and all PL roots converged.
    ///
    /// Prefer [`AraimResult::available`]. This field is kept as an alias for
    /// existing callers.
    pub availability: bool,
}

/// Run the ARAIM MHSS protection-level algorithm.
pub fn araim(
    geometry: &AraimGeometry,
    ism: &Ism,
    allocation: &IntegrityAllocation,
) -> Result<AraimResult, AraimError> {
    validate_geometry(geometry)?;
    validate_allocation(allocation)?;
    ism.validate()?;

    let effective = geometry
        .rows
        .iter()
        .map(|row| ism.effective_for(row))
        .collect::<Result<Vec<_>, _>>()?;
    let sigma_int_m: Vec<f64> = effective.iter().map(|model| model.sigma_int_m).collect();
    let sigma_acc_m: Vec<f64> = effective.iter().map(|model| model.sigma_acc_m).collect();
    let bias_m: Vec<f64> = effective.iter().map(|model| model.b_nom_m).collect();
    let weights_int: Vec<f64> = sigma_int_m
        .iter()
        .map(|sigma| 1.0 / (sigma * sigma))
        .collect();

    let enumeration = enumerate_fault_modes_checked(geometry, ism, allocation)?;
    let n_fault_modes = enumeration.modes.len().saturating_sub(1);
    let k_h = k_false_alert(
        allocation.pfa_hor,
        n_fault_modes,
        FalseAlertAxis::Horizontal,
    )?;
    let k_v = k_false_alert(allocation.pfa_vert, n_fault_modes, FalseAlertAxis::Vertical)?;

    let fault_free_int = gain_matrix_enu(geometry, &weights_int)?;
    let sigma_acc_e = metric_sigma(&fault_free_int.enu_rows[0], &sigma_acc_m);
    let sigma_acc_n = metric_sigma(&fault_free_int.enu_rows[1], &sigma_acc_m);
    let sigma_acc_u = metric_sigma(&fault_free_int.enu_rows[2], &sigma_acc_m);

    let mut p_unmonitored = enumeration.p_unenumerated;
    let mut fault_modes = Vec::with_capacity(enumeration.modes.len());
    for (mode_idx, hypothesis) in enumeration.modes.iter().enumerate() {
        let mode = if mode_idx == 0 {
            compute_monitorable_mode(
                hypothesis,
                MonitorInputs {
                    gain_int: &fault_free_int,
                    fault_free_int: &fault_free_int,
                    sigma_int_m: &sigma_int_m,
                    sigma_acc_m: &sigma_acc_m,
                    bias_m: &bias_m,
                    k: [0.0, 0.0, 0.0],
                },
            )
        } else {
            let weights_int_k = zeroed_weights(geometry, &weights_int, hypothesis);
            let clock_systems_k = active_clock_systems(geometry, hypothesis);
            match gain_matrix_enu_for_clock_systems(geometry, &weights_int_k, &clock_systems_k) {
                Ok(gain_int) => compute_monitorable_mode(
                    hypothesis,
                    MonitorInputs {
                        gain_int: &gain_int,
                        fault_free_int: &fault_free_int,
                        sigma_int_m: &sigma_int_m,
                        sigma_acc_m: &sigma_acc_m,
                        bias_m: &bias_m,
                        k: [k_h, k_h, k_v],
                    },
                ),
                _ => {
                    p_unmonitored += hypothesis.prior;
                    unmonitorable_mode(hypothesis)
                }
            }
        };
        fault_modes.push(mode);
    }

    if p_unmonitored > allocation.p_threshold_unmonitored {
        return Ok(unavailable_result(
            fault_modes,
            p_unmonitored,
            sigma_acc_e,
            sigma_acc_n,
            sigma_acc_u,
            allocation,
        ));
    }

    let budget_scale = match integrity_budget_scale(allocation, p_unmonitored) {
        Ok(scale) => scale,
        Err(AraimError::UnmonitorableFaultMass) => {
            return Ok(unavailable_result(
                fault_modes,
                p_unmonitored,
                sigma_acc_e,
                sigma_acc_n,
                sigma_acc_u,
                allocation,
            ));
        }
        Err(error) => return Err(error),
    };
    let pl_e = solve_coord_pl(
        &fault_modes,
        0,
        0.5 * allocation.phmi_hor * budget_scale,
        HORIZONTAL_PL_TOL_M,
    )?;
    let pl_n = solve_coord_pl(
        &fault_modes,
        1,
        0.5 * allocation.phmi_hor * budget_scale,
        HORIZONTAL_PL_TOL_M,
    )?;
    let pl_u = solve_coord_pl(
        &fault_modes,
        2,
        allocation.phmi_vert * budget_scale,
        VERTICAL_PL_TOL_M,
    )?;
    let roots_converged = pl_e.converged && pl_n.converged && pl_u.converged;
    let emt_m = fault_modes
        .iter()
        .filter(|mode| mode.monitorable)
        .filter(|mode| mode.prior >= allocation.p_emt)
        .map(|mode| mode.threshold_enu_m[2])
        .fold(0.0_f64, f64::max);
    let fault_free_full_rank = fault_modes
        .first()
        .map(|mode| mode.monitorable)
        .unwrap_or(false);

    let available = fault_free_full_rank && roots_converged;
    Ok(AraimResult {
        available,
        hpl_m: (pl_e.value_m * pl_e.value_m + pl_n.value_m * pl_n.value_m).sqrt(),
        vpl_m: pl_u.value_m,
        sigma_acc_h_m: (sigma_acc_e * sigma_acc_e + sigma_acc_n * sigma_acc_n).sqrt(),
        sigma_acc_v_m: sigma_acc_u,
        emt_m,
        fault_modes,
        p_unmonitored,
        availability: available,
    })
}

fn unavailable_result(
    fault_modes: Vec<FaultMode>,
    p_unmonitored: f64,
    sigma_acc_e: f64,
    sigma_acc_n: f64,
    sigma_acc_u: f64,
    allocation: &IntegrityAllocation,
) -> AraimResult {
    let emt_m = fault_modes
        .iter()
        .filter(|mode| mode.monitorable)
        .filter(|mode| mode.prior >= allocation.p_emt)
        .map(|mode| mode.threshold_enu_m[2])
        .fold(0.0_f64, f64::max);
    AraimResult {
        available: false,
        hpl_m: f64::INFINITY,
        vpl_m: f64::INFINITY,
        sigma_acc_h_m: (sigma_acc_e * sigma_acc_e + sigma_acc_n * sigma_acc_n).sqrt(),
        sigma_acc_v_m: sigma_acc_u,
        emt_m,
        fault_modes,
        p_unmonitored,
        availability: false,
    }
}

struct MonitorInputs<'a> {
    gain_int: &'a super::protection::GainMatrix,
    fault_free_int: &'a super::protection::GainMatrix,
    sigma_int_m: &'a [f64],
    sigma_acc_m: &'a [f64],
    bias_m: &'a [f64],
    k: [f64; 3],
}

fn compute_monitorable_mode(hypothesis: &FaultHypothesis, inputs: MonitorInputs<'_>) -> FaultMode {
    let mut sigma_int_enu_m = [0.0_f64; 3];
    let mut bias_enu_m = [0.0_f64; 3];
    let mut threshold_enu_m = [0.0_f64; 3];
    for coord in 0..3 {
        sigma_int_enu_m[coord] = metric_sigma(&inputs.gain_int.enu_rows[coord], inputs.sigma_int_m);
        bias_enu_m[coord] = metric_bias(&inputs.gain_int.enu_rows[coord], inputs.bias_m);
        threshold_enu_m[coord] = inputs.k[coord]
            * separation_sigma(
                &inputs.gain_int.enu_rows[coord],
                &inputs.fault_free_int.enu_rows[coord],
                inputs.sigma_acc_m,
            );
    }

    FaultMode {
        excluded: hypothesis.excluded.clone(),
        excluded_constellation: hypothesis.excluded_constellation,
        prior: hypothesis.prior,
        sigma_int_enu_m,
        bias_enu_m,
        threshold_enu_m,
        monitorable: true,
    }
}

fn unmonitorable_mode(hypothesis: &FaultHypothesis) -> FaultMode {
    FaultMode {
        excluded: hypothesis.excluded.clone(),
        excluded_constellation: hypothesis.excluded_constellation,
        prior: hypothesis.prior,
        sigma_int_enu_m: [f64::INFINITY; 3],
        bias_enu_m: [f64::INFINITY; 3],
        threshold_enu_m: [f64::INFINITY; 3],
        monitorable: false,
    }
}

fn zeroed_weights(
    geometry: &AraimGeometry,
    weights: &[f64],
    hypothesis: &FaultHypothesis,
) -> Vec<f64> {
    geometry
        .rows
        .iter()
        .zip(weights)
        .map(|(row, &weight)| {
            if hypothesis.excludes_satellite(row.id, row.system) {
                0.0
            } else {
                weight
            }
        })
        .collect()
}

fn active_clock_systems(geometry: &AraimGeometry, hypothesis: &FaultHypothesis) -> Vec<GnssSystem> {
    geometry
        .clock_systems
        .iter()
        .copied()
        .filter(|&clock_system| {
            geometry.rows.iter().any(|row| {
                !hypothesis.excludes_satellite(row.id, row.system)
                    && clock_system_for_row(row.system, &geometry.clock_systems) == clock_system
            })
        })
        .collect()
}

fn solve_coord_pl(
    modes: &[FaultMode],
    coord: usize,
    integrity_target: f64,
    tolerance_m: f64,
) -> Result<ProtectionLevelSolution, AraimError> {
    let fault_free = modes.first().ok_or(AraimError::NumericalFailure)?;
    if !fault_free.monitorable {
        return Err(AraimError::InsufficientGeometry);
    }
    let fault_free_term = ProtectionEquationTerm {
        prior: 1.0,
        sigma_m: fault_free.sigma_int_enu_m[coord],
        bias_m: fault_free.bias_enu_m[coord],
        threshold_m: 0.0,
    };
    let fault_terms: Vec<ProtectionEquationTerm> = modes
        .iter()
        .skip(1)
        .filter(|mode| mode.monitorable)
        .map(|mode| ProtectionEquationTerm {
            prior: mode.prior,
            sigma_m: mode.sigma_int_enu_m[coord],
            bias_m: mode.bias_enu_m[coord],
            threshold_m: mode.threshold_enu_m[coord],
        })
        .collect();
    solve_protection_level(fault_free_term, &fault_terms, integrity_target, tolerance_m)
}

fn integrity_budget_scale(
    allocation: &IntegrityAllocation,
    p_unmonitored: f64,
) -> Result<f64, AraimError> {
    let phmi_split = allocation.phmi_vert + allocation.phmi_hor;
    let scale = 1.0 - p_unmonitored / phmi_split;
    if scale > 0.0 && scale.is_finite() {
        Ok(scale)
    } else {
        Err(AraimError::UnmonitorableFaultMass)
    }
}

fn validate_geometry(geometry: &AraimGeometry) -> Result<(), AraimError> {
    if geometry.clock_systems.is_empty() {
        return Err(AraimError::InsufficientGeometry);
    }
    let n_state = 3 + geometry.clock_systems.len();
    if geometry.rows.len() < n_state {
        return Err(AraimError::InsufficientGeometry);
    }
    let mut clock_systems = BTreeSet::new();
    for &system in &geometry.clock_systems {
        if !clock_systems.insert(system) {
            return Err(AraimError::InsufficientGeometry);
        }
    }

    let mut ids = BTreeSet::new();
    for row in &geometry.rows {
        if row.id.system != row.system || !ids.insert(row.id) {
            return Err(AraimError::InsufficientGeometry);
        }
        if !(-core::f64::consts::FRAC_PI_2..=core::f64::consts::FRAC_PI_2)
            .contains(&row.elevation_rad)
        {
            return Err(AraimError::InsufficientGeometry);
        }
        let los = row.line_of_sight;
        let norm = (los.e_x * los.e_x + los.e_y * los.e_y + los.e_z * los.e_z).sqrt();
        if !norm.is_finite() || (norm - 1.0).abs() > 1.0e-3 {
            return Err(AraimError::InsufficientGeometry);
        }
    }
    Ok(())
}

fn validate_allocation(allocation: &IntegrityAllocation) -> Result<(), AraimError> {
    let phmi_split = allocation.phmi_vert + allocation.phmi_hor;
    let phmi_split_tolerance = allocation.phmi_total * 16.0 * f64::EPSILON;
    let valid = validate_probability(allocation.phmi_total, false)
        && validate_probability(allocation.phmi_vert, false)
        && validate_probability(allocation.phmi_hor, false)
        && validate_probability(allocation.pfa_vert, false)
        && validate_probability(allocation.pfa_hor, false)
        && validate_probability(allocation.p_threshold_unmonitored, true)
        && validate_probability(allocation.p_emt, false)
        && phmi_split <= allocation.phmi_total + phmi_split_tolerance;
    if valid {
        Ok(())
    } else {
        Err(AraimError::InvalidAllocation)
    }
}
