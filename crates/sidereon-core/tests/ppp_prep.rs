use sidereon_core::carrier_phase::{CycleSlipOptions, SlipReason};
use sidereon_core::constants::{C_M_S, F_L1_HZ};
use sidereon_core::precise_positioning::{
    prepare_widelane_fixed_epochs, split_float_cycle_slip_epochs, CycleSlipPolicy,
    DualFrequencyEpoch, DualFrequencyObservation, FloatCycleSlipEpoch, FloatCycleSlipObservation,
    PppSplitArc, WideLanePrepError, WideLanePrepOptions,
};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

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

fn dual_epochs_with_l1_slip(
    epoch_count: usize,
    slip_epoch: Option<usize>,
    persistent: bool,
) -> Vec<DualFrequencyEpoch> {
    (0..epoch_count)
        .map(|epoch_idx| DualFrequencyEpoch {
            gap_time_s: Some(epoch_idx as f64 * 30.0),
            observations: (0..4)
                .map(|sat_idx| {
                    let slipped = sat_idx == 0
                        && slip_epoch.is_some_and(|epoch| {
                            if persistent {
                                epoch_idx >= epoch
                            } else {
                                epoch_idx == epoch
                            }
                        });
                    dual_observation_with_code_noise(
                        sat_idx,
                        epoch_idx,
                        if slipped { 1.0 } else { 0.0 },
                        None,
                        0.0,
                    )
                })
                .collect(),
        })
        .collect()
}

fn day_length_mw_spike_epochs(epoch_count: usize, sat_count: usize) -> Vec<FloatCycleSlipEpoch> {
    (0..epoch_count)
        .map(|epoch_idx| FloatCycleSlipEpoch {
            gap_time_s: Some(epoch_idx as f64 * 30.0),
            observations: (0..sat_count)
                .map(|sat_idx| {
                    let code_noise_m = if epoch_idx > 0
                        && epoch_idx + 1 < epoch_count
                        && epoch_idx % 97 == sat_idx
                    {
                        4.5
                    } else {
                        0.0
                    };
                    let raw = dual_observation_with_code_noise(
                        sat_idx,
                        epoch_idx,
                        0.0,
                        None,
                        code_noise_m,
                    );
                    FloatCycleSlipObservation {
                        satellite_id: raw.satellite_id.clone(),
                        ambiguity_id: raw.satellite_id.clone(),
                        raw: Some(raw),
                    }
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
    dual_observation_with_code_noise(sat_idx, epoch_idx, slip_cycles, lli1, 0.0)
}

fn dual_observation_with_code_noise(
    sat_idx: usize,
    epoch_idx: usize,
    slip_cycles: f64,
    lli1: Option<i64>,
    code_noise_m: f64,
) -> DualFrequencyObservation {
    let satellite_id = format!("G{:02}", sat_idx + 1);
    let base = 23_000_000.0 + epoch_idx as f64 * 200.0 + sat_idx as f64 * 500.0;
    let code = base + code_noise_m;
    let n1 = 80_000.0 + sat_idx as f64 * 37.0 + slip_cycles;
    let nw = 5.0 + sat_idx as f64;
    let n2 = 80_000.0 + sat_idx as f64 * 37.0 - nw;
    let lambda1 = C_M_S / F_L1_HZ;
    let lambda2 = C_M_S / F_L2_HZ;

    DualFrequencyObservation {
        satellite_id: satellite_id.clone(),
        ambiguity_id: satellite_id,
        p1_m: code,
        p2_m: code,
        phi1_cyc: (base + n1 * lambda1) / lambda1,
        phi2_cyc: (base + n2 * lambda2) / lambda2,
        f1_hz: F_L1_HZ,
        f2_hz: F_L2_HZ,
        lli1,
        lli2: None,
    }
}

fn wide_lane_options() -> WideLanePrepOptions {
    WideLanePrepOptions::new(2, 0.01)
}

fn slip_options() -> CycleSlipOptions {
    let mut options = CycleSlipOptions::default();
    options.gf_threshold_m = 0.05;
    options.mw_threshold_cycles = 4.0;
    options.min_arc_gap_s = 1_000.0;
    options
}

fn float_epochs_from_dual(epochs: Vec<DualFrequencyEpoch>) -> Vec<FloatCycleSlipEpoch> {
    epochs
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
        .collect()
}

fn ambiguity_ids_for_sat<'a>(
    epochs: &'a [sidereon_core::precise_positioning::FloatCycleSlipTaggedEpoch],
    satellite_id: &str,
) -> Vec<&'a str> {
    epochs
        .iter()
        .map(|epoch| {
            epoch
                .observations
                .iter()
                .find(|obs| obs.satellite_id == satellite_id)
                .expect("satellite observation")
                .ambiguity_id
                .as_str()
        })
        .collect()
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
fn float_cycle_slip_split_confirms_one_cycle_gf_jump_and_leaves_clean_arc_unsplit() {
    let slipped = split_float_cycle_slip_epochs(
        &float_epochs_from_dual(dual_epochs_with_l1_slip(6, Some(3), true)),
        slip_options(),
    );
    assert_eq!(
        ambiguity_ids_for_sat(&slipped, "G01"),
        vec!["G01#1", "G01#1", "G01#1", "G01#2", "G01#2", "G01#2"]
    );

    let clean = split_float_cycle_slip_epochs(
        &float_epochs_from_dual(dual_epochs_with_l1_slip(6, None, true)),
        slip_options(),
    );
    assert_eq!(
        ambiguity_ids_for_sat(&clean, "G01"),
        vec!["G01", "G01", "G01", "G01", "G01", "G01"]
    );
}

#[test]
fn float_cycle_slip_split_ignores_single_epoch_mw_excursion() {
    let tagged = split_float_cycle_slip_epochs(&day_length_mw_spike_epochs(12, 2), slip_options());
    for sat in ["G01", "G02"] {
        assert!(
            ambiguity_ids_for_sat(&tagged, sat)
                .into_iter()
                .all(|ambiguity_id| ambiguity_id == sat),
            "{sat} should not split on a one-epoch code-only MW excursion"
        );
    }
}

#[test]
fn day_length_float_cycle_slip_split_with_gf_mw_enabled_is_tractable() {
    let epoch_count = 2_880;
    let sat_count = 8;
    let epochs = day_length_mw_spike_epochs(epoch_count, sat_count);
    let start = std::time::Instant::now();
    let tagged = split_float_cycle_slip_epochs(&epochs, CycleSlipOptions::default());
    let elapsed = start.elapsed();
    eprintln!(
        "PPP GF/MW split prep {epoch_count} epochs x {sat_count} sats completed in {elapsed:?}"
    );

    assert!(
        elapsed < Duration::from_secs(10),
        "GF/MW split prep took {elapsed:?}"
    );
    let ambiguity_ids = tagged
        .iter()
        .flat_map(|epoch| {
            epoch
                .observations
                .iter()
                .map(|obs| obs.ambiguity_id.clone())
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(ambiguity_ids.len(), sat_count);
    assert!(
        ambiguity_ids.iter().all(|id| !id.contains('#')),
        "clean day-length arc created split ambiguity ids: {ambiguity_ids:?}"
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
