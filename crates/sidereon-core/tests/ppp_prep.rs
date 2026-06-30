use sidereon_core::carrier_phase::{CycleSlipOptions, SlipReason};
use sidereon_core::constants::{C_M_S, F_L1_HZ};
use sidereon_core::precise_positioning::{
    prepare_widelane_fixed_epochs, split_float_cycle_slip_epochs, CycleSlipPolicy,
    DualFrequencyEpoch, DualFrequencyObservation, FloatCycleSlipEpoch, FloatCycleSlipObservation,
    PppSplitArc, WideLanePrepError, WideLanePrepOptions,
};
use std::collections::BTreeMap;

const F_L2_HZ: f64 = 1_227_600_000.0;

fn dual_epochs(slip: bool) -> Vec<DualFrequencyEpoch> {
    (0..3)
        .map(|epoch_idx| DualFrequencyEpoch {
            gap_time_s: Some(epoch_idx as f64 * 30.0),
            observations: (0..4)
                .map(|sat_idx| {
                    let slip_cycles = if slip && sat_idx == 0 && epoch_idx >= 1 {
                        8.0
                    } else {
                        0.0
                    };
                    let lli1 = if slip && sat_idx == 0 && epoch_idx == 1 {
                        Some(1)
                    } else {
                        None
                    };
                    dual_observation(sat_idx, epoch_idx, slip_cycles, lli1)
                })
                .collect(),
        })
        .collect()
}

fn dual_observation(
    sat_idx: usize,
    epoch_idx: usize,
    slip_cycles: f64,
    lli1: Option<i64>,
) -> DualFrequencyObservation {
    let satellite_id = format!("G{:02}", sat_idx + 1);
    let base = 23_000_000.0 + epoch_idx as f64 * 200.0 + sat_idx as f64 * 500.0;
    let n1 = 80_000.0 + sat_idx as f64 * 37.0 + slip_cycles;
    let nw = 5.0 + sat_idx as f64;
    let n2 = 80_000.0 + sat_idx as f64 * 37.0 - nw;
    let lambda1 = C_M_S / F_L1_HZ;
    let lambda2 = C_M_S / F_L2_HZ;

    DualFrequencyObservation {
        satellite_id: satellite_id.clone(),
        ambiguity_id: satellite_id,
        p1_m: base,
        p2_m: base,
        phi1_cyc: (base + n1 * lambda1) / lambda1,
        phi2_cyc: (base + n2 * lambda2) / lambda2,
        f1_hz: F_L1_HZ,
        f2_hz: F_L2_HZ,
        lli1,
        lli2: None,
    }
}

fn wide_lane_options() -> WideLanePrepOptions {
    WideLanePrepOptions {
        min_epochs: 2,
        tolerance_cycles: 0.01,
    }
}

fn slip_options() -> CycleSlipOptions {
    CycleSlipOptions {
        gf_threshold_m: 0.05,
        mw_threshold_cycles: 4.0,
        min_arc_gap_s: 1_000.0,
    }
}

#[test]
fn widelane_fixed_prep_splits_arcs_with_frozen_bits() {
    let result = prepare_widelane_fixed_epochs(
        &dual_epochs(true),
        wide_lane_options(),
        CycleSlipPolicy::SplitArc,
        slip_options(),
    )
    .unwrap();

    assert_eq!(
        result.wide_lane_cycles,
        BTreeMap::from([
            ("G01#2".to_string(), 13),
            ("G02".to_string(), 6),
            ("G03".to_string(), 7),
            ("G04".to_string(), 8),
        ])
    );
    assert_eq!(
        result.split_arcs,
        vec![PppSplitArc {
            satellite_id: "G01".to_string(),
            ambiguity_id: "G01#2".to_string(),
            start_epoch_index: 1,
            end_epoch_index: 2,
            n_epochs: 2,
        }]
    );
    assert_eq!(
        result
            .wavelengths_m
            .iter()
            .map(|(sat, value)| (sat.as_str(), value.to_bits()))
            .collect::<Vec<_>>(),
        vec![
            ("G01#2", 0x3fbb614bed5136b9),
            ("G02", 0x3fbb614bed5136b9),
            ("G03", 0x3fbb614bed5136b9),
            ("G04", 0x3fbb614bed5136b9),
        ]
    );
    assert_eq!(
        result
            .offsets_m
            .iter()
            .map(|(sat, value)| (sat.as_str(), value.to_bits()))
            .collect::<Vec<_>>(),
        vec![
            ("G01#2", 0x4013a10c147d0bf0),
            ("G02", 0x40021e814dfd4618),
            ("G03", 0x40052396dafcd1c7),
            ("G04", 0x400828ac67fc5d76),
        ]
    );
    assert_eq!(
        result
            .epochs
            .iter()
            .flat_map(|epoch| {
                epoch.observations.iter().map(move |obs| {
                    (
                        epoch.epoch_index,
                        obs.satellite_id.as_str(),
                        obs.ambiguity_id.as_str(),
                        obs.code_m.to_bits(),
                        obs.phase_m.to_bits(),
                    )
                })
            })
            .collect::<Vec<_>>(),
        vec![
            (0, "G02", "G02", 0x4175ef5b40000000, 0x4175f17267e0f54a),
            (0, "G03", "G03", 0x4175ef7a80000000, 0x4175f191ed3c1ffa),
            (0, "G04", "G04", 0x4175ef99c0000000, 0x4175f1b172974aa8),
            (1, "G01", "G01#2", 0x4175ef4880000000, 0x4175f15fa087c962),
            (1, "G02", "G02", 0x4175ef67c0000000, 0x4175f17ee7e0f54a),
            (1, "G03", "G03", 0x4175ef8700000000, 0x4175f19e6d3c1ffa),
            (1, "G04", "G04", 0x4175efa640000000, 0x4175f1bdf2974aa8),
            (2, "G01", "G01#2", 0x4175ef5500000000, 0x4175f16c2087c962),
            (2, "G02", "G02", 0x4175ef7440000000, 0x4175f18b67e0f54a),
            (2, "G03", "G03", 0x4175ef9380000000, 0x4175f1aaed3c1ffa),
            (2, "G04", "G04", 0x4175efb2c0000000, 0x4175f1ca72974aa8),
        ]
    );
}

#[test]
fn widelane_fixed_prep_exposes_cycle_slip_policies() {
    let epochs = dual_epochs(true);

    assert_eq!(
        prepare_widelane_fixed_epochs(
            &epochs,
            wide_lane_options(),
            CycleSlipPolicy::Error,
            slip_options(),
        ),
        Err(WideLanePrepError::CycleSlipDetected {
            satellite_id: "G01".to_string(),
            epoch_index: 1,
            reasons: vec![
                SlipReason::Lli,
                SlipReason::GeometryFree,
                SlipReason::MelbourneWubbena,
            ],
        })
    );

    let dropped = prepare_widelane_fixed_epochs(
        &epochs,
        wide_lane_options(),
        CycleSlipPolicy::DropSatellite,
        slip_options(),
    )
    .unwrap();

    assert_eq!(dropped.dropped_sats, vec!["G01".to_string()]);
    assert_eq!(
        dropped.wide_lane_cycles,
        BTreeMap::from([
            ("G02".to_string(), 6),
            ("G03".to_string(), 7),
            ("G04".to_string(), 8),
        ])
    );
}

#[test]
fn float_cycle_slip_split_tags_are_core_owned() {
    let epochs = dual_epochs(true)
        .into_iter()
        .map(|epoch| FloatCycleSlipEpoch {
            gap_time_s: epoch.gap_time_s,
            observations: epoch
                .observations
                .into_iter()
                .map(|raw| FloatCycleSlipObservation {
                    satellite_id: raw.satellite_id.clone(),
                    ambiguity_id: raw.satellite_id.clone(),
                    raw: Some(raw),
                })
                .collect(),
        })
        .collect::<Vec<_>>();

    let tagged = split_float_cycle_slip_epochs(&epochs, slip_options());

    assert_eq!(
        tagged
            .iter()
            .map(|epoch| {
                epoch
                    .observations
                    .iter()
                    .map(|obs| (obs.satellite_id.as_str(), obs.ambiguity_id.as_str()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        vec![
            vec![
                ("G01", "G01#1"),
                ("G02", "G02"),
                ("G03", "G03"),
                ("G04", "G04"),
            ],
            vec![
                ("G01", "G01#2"),
                ("G02", "G02"),
                ("G03", "G03"),
                ("G04", "G04"),
            ],
            vec![
                ("G01", "G01#2"),
                ("G02", "G02"),
                ("G03", "G03"),
                ("G04", "G04"),
            ],
        ]
    );
}

#[test]
fn float_cycle_slip_split_skips_only_epochs_missing_raw_dual_frequency_data() {
    let mut epochs = dual_epochs(true)
        .into_iter()
        .map(|epoch| FloatCycleSlipEpoch {
            gap_time_s: epoch.gap_time_s,
            observations: epoch
                .observations
                .into_iter()
                .map(|raw| FloatCycleSlipObservation {
                    satellite_id: raw.satellite_id.clone(),
                    ambiguity_id: raw.satellite_id.clone(),
                    raw: Some(raw),
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    epochs[1].observations[0].raw = None;

    let tagged = split_float_cycle_slip_epochs(&epochs, slip_options());

    assert_eq!(
        tagged
            .iter()
            .map(|epoch| {
                epoch
                    .observations
                    .iter()
                    .find(|obs| obs.satellite_id == "G01")
                    .expect("G01 observation")
                    .ambiguity_id
                    .as_str()
            })
            .collect::<Vec<_>>(),
        vec!["G01#1", "G01#1", "G01#2"]
    );
}
