//! Shared RTK-filter fixture for the allocation gate and the timing bench.

use sidereon_core::rtk_filter::{
    DynamicsModel, Epoch, MeasModel, SatMeas, SearchOpts, StochasticModel, UpdateOpts,
};
use std::collections::BTreeMap;

pub fn range(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// A representative 6-satellite epoch (1 reference + 5 non-reference) plus the
/// matching model/wavelength/offset/opts inputs for `update_epoch`.
#[allow(clippy::type_complexity)]
pub fn inputs() -> (
    Epoch,
    [f64; 3],
    MeasModel,
    BTreeMap<String, f64>,
    BTreeMap<String, f64>,
    UpdateOpts,
) {
    let base = [4_075_580.0, 931_854.0, 4_801_568.0];
    let truth = [1.2, -0.85, 0.91];
    let rover = [base[0] + truth[0], base[1] + truth[1], base[2] + truth[2]];
    let lambda = 299_792_458.0 / 1_575_420_000.0;
    let sats: [(&str, [f64; 3], i64); 6] = [
        ("G01", [15_000_000.0, 7_000_000.0, 21_000_000.0], 0),
        ("G02", [-12_000_000.0, 18_000_000.0, 19_000_000.0], 3),
        ("G03", [20_000_000.0, -10_000_000.0, 17_000_000.0], -7),
        ("G04", [-19_000_000.0, -13_000_000.0, 20_000_000.0], 12),
        ("G05", [9_000_000.0, 22_000_000.0, 16_000_000.0], -4),
        ("G06", [-6_000_000.0, -20_000_000.0, 18_000_000.0], 8),
    ];
    let mk = |pos: [f64; 3], id: &str, ncyc: i64| SatMeas {
        sat: id.into(),
        sd_ambiguity_id: id.into(),
        base_code_m: range(pos, base),
        base_phase_m: range(pos, base),
        rover_code_m: range(pos, rover),
        rover_phase_m: range(pos, rover) + (ncyc as f64) * lambda,
        base_tx_pos: pos,
        rover_tx_pos: pos,
        pos,
    };
    let epoch = Epoch {
        references: vec![mk(sats[0].1, sats[0].0, sats[0].2)],
        nonref: sats[1..].iter().map(|&(id, p, n)| mk(p, id, n)).collect(),
        velocity_mps: None,
        dt_s: 0.0,
    };
    let wl: BTreeMap<String, f64> = sats[1..]
        .iter()
        .map(|&(id, _, _)| (id.to_string(), lambda))
        .collect();
    let off: BTreeMap<String, f64> = sats[1..]
        .iter()
        .map(|&(id, _, _)| (id.to_string(), 0.0))
        .collect();
    let model = MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: true,
        stochastic: StochasticModel::Rtklib,
    };
    let opts = UpdateOpts {
        hold_sigma_m: 1.0e-4,
        position_tol_m: 1.0e-4,
        ambiguity_tol_m: 1.0e-6,
        max_iterations: 10,
        process_noise_baseline_sigma_m: 0.0,
        dynamics_model: DynamicsModel::ConstantPosition,
        float_only_systems: vec![],
        innovation_screen: None,
        report_residuals: false,
        receiver_antenna_corrections: None,
        ar_arming_sigma_m: None,
        search: SearchOpts {
            ratio_threshold: 3.0,
        },
    };
    (epoch, base, model, wl, off, opts)
}
