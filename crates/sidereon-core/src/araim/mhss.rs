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

    // ADD section 4.6 first stops selection on the residual Eq. (15) mass.
    // The later "Filtering the subsets" step then adds unobservable selected
    // modes once under Eq. (18), as MAAST `determine_subsets_v4` does.
    let enumeration = enumerate_fault_modes_checked(geometry, ism, allocation)?;
    let fault_free_int = gain_matrix_enu(geometry, &weights_int)?;
    let sigma_acc_e = metric_sigma(&fault_free_int.enu_rows[0], &sigma_acc_m);
    let sigma_acc_n = metric_sigma(&fault_free_int.enu_rows[1], &sigma_acc_m);
    let sigma_acc_u = metric_sigma(&fault_free_int.enu_rows[2], &sigma_acc_m);

    let observable = enumeration
        .modes
        .iter()
        .skip(1)
        .filter(|hypothesis| {
            let weights = zeroed_weights(geometry, &weights_int, hypothesis);
            let clocks = active_clock_systems(geometry, hypothesis);
            gain_matrix_enu_for_clock_systems(geometry, &weights, &clocks).is_ok()
        })
        .count();
    let k_h = if allocation.pfa_hor == 0.0 || observable == 0 {
        0.0
    } else {
        k_false_alert(allocation.pfa_hor, observable, FalseAlertAxis::Horizontal)?
    };
    let k_v = if allocation.pfa_vert == 0.0 || observable == 0 {
        0.0
    } else {
        k_false_alert(allocation.pfa_vert, observable, FalseAlertAxis::Vertical)?
    };

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
    let pl_e = solve_optional_coord_pl(
        &fault_modes,
        0,
        0.5 * allocation.phmi_hor * budget_scale,
        HORIZONTAL_PL_TOL_M,
    )?;
    let pl_n = solve_optional_coord_pl(
        &fault_modes,
        1,
        0.5 * allocation.phmi_hor * budget_scale,
        HORIZONTAL_PL_TOL_M,
    )?;
    let pl_u = solve_optional_coord_pl(
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

fn solve_optional_coord_pl(
    modes: &[FaultMode],
    coord: usize,
    integrity_target: f64,
    tolerance_m: f64,
) -> Result<ProtectionLevelSolution, AraimError> {
    if integrity_target == 0.0 {
        Ok(ProtectionLevelSolution {
            value_m: 0.0,
            converged: true,
        })
    } else {
        solve_coord_pl(modes, coord, integrity_target, tolerance_m)
    }
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
        && validate_probability(allocation.phmi_vert, true)
        && validate_probability(allocation.phmi_hor, true)
        && validate_probability(allocation.pfa_vert, true)
        && validate_probability(allocation.pfa_hor, true)
        && validate_probability(allocation.p_threshold_unmonitored, true)
        && validate_probability(allocation.p_emt, false)
        && allocation.p_threshold_unmonitored < allocation.phmi_total
        && (allocation.phmi_vert == 0.0) == (allocation.pfa_vert == 0.0)
        && (allocation.phmi_hor == 0.0) == (allocation.pfa_hor == 0.0)
        && phmi_split <= allocation.phmi_total + phmi_split_tolerance;
    if valid {
        Ok(())
    } else {
        Err(AraimError::InvalidAllocation)
    }
}

#[cfg(test)]
mod departure_tests {
    use super::*;
    use crate::araim::{AraimRow, ConstellationIsm, SatelliteIsm, SatelliteIsmModel};
    use crate::dop::LineOfSight;
    use crate::frame::Wgs84Geodetic;
    use crate::GnssSatelliteId;

    #[test]
    fn maast_departures_are_measured_against_finely_resolved_engine_roots() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/araim_maast_matrix.json"))
                .unwrap();
        for case in fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|case| !case["add_reference"].is_null())
        {
            let n = case["n_sat"].as_u64().unwrap() as usize;
            let nc = case["n_const"].as_u64().unwrap() as usize;
            let systems = [GnssSystem::Gps, GnssSystem::Galileo, GnssSystem::BeiDou];
            let rows: Vec<_> = (0..n)
                .map(|i| {
                    let az = (137.50776 * i as f64).to_radians();
                    let el = (15.0 + 60.0 * (0.61803 * i as f64).fract()).to_radians();
                    let system = systems[i % nc];
                    AraimRow {
                        id: GnssSatelliteId::new(system, (i / nc + 1) as u8).unwrap(),
                        system,
                        line_of_sight: LineOfSight::new(
                            libm::sin(el),
                            libm::cos(el) * libm::sin(az),
                            libm::cos(el) * libm::cos(az),
                        ),
                        elevation_rad: el,
                    }
                })
                .collect();
            let geometry = AraimGeometry {
                rows,
                receiver: Wgs84Geodetic::new(0.0, 0.0, 0.0).unwrap(),
                clock_systems: systems[..nc].to_vec(),
                ut1_degraded: None,
            };
            let (ura, ure, bias, p_sat) = (
                case["ura_m"].as_f64().unwrap(),
                case["ure_m"].as_f64().unwrap(),
                case["bias_m"].as_f64().unwrap(),
                case["p_sat"].as_f64().unwrap(),
            );
            let defaults = systems[..nc]
                .iter()
                .map(|&system| {
                    ConstellationIsm::new(
                        system,
                        case["p_const"].as_f64().unwrap(),
                        SatelliteIsmModel::new(ura, ure, bias, p_sat),
                    )
                })
                .collect();
            let satellites = geometry
                .rows
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    SatelliteIsm::new_with_effective_sigmas(
                        row.id,
                        ura,
                        ure,
                        bias,
                        p_sat,
                        libm::sqrt(case["cint_m2"][i].as_f64().unwrap()),
                        libm::sqrt(case["cacc_m2"][i].as_f64().unwrap()),
                    )
                })
                .collect();
            let mut allocation = IntegrityAllocation::lpv_200();
            allocation.max_fault_order = case["max_fault_order"].as_u64().unwrap() as usize;
            allocation.p_threshold_unmonitored = case["p_threshold_unmonitored"].as_f64().unwrap();
            let result = araim(&geometry, &Ism::new(defaults, satellites), &allocation).unwrap();
            let scale = integrity_budget_scale(&allocation, result.p_unmonitored).unwrap();
            // Resolve the same selected-mode equations more finely for this
            // comparison, without changing the public default's PL tolerance.
            // An upper scalar root is at most TOL above its exact ADD root;
            // hypot is 1-Lipschitz, giving sqrt(2)*TOL horizontally. These
            // windows are much smaller than the recorded MAAST departures.
            let tolerance = 1e-5;
            let east = solve_coord_pl(
                &result.fault_modes,
                0,
                0.5 * allocation.phmi_hor * scale,
                tolerance,
            )
            .unwrap()
            .value_m;
            let north = solve_coord_pl(
                &result.fault_modes,
                1,
                0.5 * allocation.phmi_hor * scale,
                tolerance,
            )
            .unwrap()
            .value_m;
            let vertical = solve_coord_pl(
                &result.fault_modes,
                2,
                allocation.phmi_vert * scale,
                tolerance,
            )
            .unwrap()
            .value_m;
            let horizontal = libm::hypot(east, north);
            for (engine, axis, bound) in [
                (vertical, "vpl", tolerance),
                (horizontal, "hpl", std::f64::consts::SQRT_2 * tolerance),
            ] {
                let add = &case["add_reference"];
                let root = add[format!("{axis}_root_m")].as_f64().unwrap();
                assert!(
                    (0.0..=bound).contains(&(engine - root)),
                    "{} {axis}: {engine} root {root}",
                    case["name"]
                );
                let departure = case[format!("{axis}_m")].as_f64().unwrap() - engine;
                let recorded = add[format!("maast_{axis}_departure_m")].as_f64().unwrap();
                assert!((0.0..=bound).contains(&(recorded - departure)));
            }
        }
    }
}
