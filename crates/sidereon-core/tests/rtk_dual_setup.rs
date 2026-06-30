use sidereon_core::rtk::{
    prepare_ionosphere_free_baseline_epochs, DualIonosphereFreeSetupEpoch, DualObservation,
    DualSatelliteObservation,
};
use std::collections::BTreeMap;

const F1: f64 = 1_575_420_000.0;
const F2: f64 = 1_227_600_000.0;

fn setup_observation(
    ambiguity_id: &str,
    p1_m: f64,
    p2_m: f64,
    phi1_cycles: f64,
    phi2_cycles: f64,
) -> DualObservation {
    DualObservation {
        ambiguity_id: ambiguity_id.to_string(),
        p1_m,
        p2_m,
        phi1_cycles,
        phi2_cycles,
        f1_hz: F1,
        f2_hz: F2,
    }
}

fn setup_pair(
    sat: &str,
    base_code: f64,
    rover_code: f64,
    base_phase_cycles: f64,
    rover_phase_cycles: f64,
) -> DualSatelliteObservation {
    DualSatelliteObservation {
        satellite_id: sat.to_string(),
        base: setup_observation(
            sat,
            base_code,
            base_code + 2.0,
            base_phase_cycles,
            base_phase_cycles - 4.0,
        ),
        rover: setup_observation(
            sat,
            rover_code,
            rover_code + 2.5,
            rover_phase_cycles,
            rover_phase_cycles - 3.0,
        ),
    }
}

fn position_map(entries: &[(&str, [f64; 3])]) -> BTreeMap<String, [f64; 3]> {
    entries
        .iter()
        .map(|(sat, position)| (sat.to_string(), *position))
        .collect()
}

#[test]
fn dual_frequency_if_setup_has_frozen_troposphere_bits() {
    let base_m = [4_078_500.0, 931_000.0, 4_801_500.0];
    let initial_baseline_m = [7.25, -3.5, 1.75];
    let satellite_positions = position_map(&[
        ("G01", [16_314_000.0, 3_724_000.0, 19_206_000.0]),
        ("G02", [15_682_600.0, 5_351_600.0, 18_285_400.0]),
        ("G03", [17_036_200.0, 1_951_700.0, 19_966_300.0]),
    ]);
    let epochs = vec![
        DualIonosphereFreeSetupEpoch {
            jd_whole: 2_460_100.5,
            jd_fraction: 0.375,
            observations: vec![
                setup_pair(
                    "G01",
                    20_000_000.0,
                    20_000_020.0,
                    105_100_000.0,
                    105_100_040.0,
                ),
                setup_pair(
                    "G02",
                    21_000_000.0,
                    21_000_035.0,
                    110_200_000.0,
                    110_200_090.0,
                ),
                setup_pair(
                    "G03",
                    22_000_000.0,
                    22_000_055.0,
                    115_300_000.0,
                    115_300_120.0,
                ),
            ],
            base_satellite_positions_m: satellite_positions.clone(),
            rover_satellite_positions_m: satellite_positions.clone(),
        },
        DualIonosphereFreeSetupEpoch {
            jd_whole: 2_460_100.5,
            jd_fraction: 0.375_347_222_222_222_2,
            observations: vec![
                setup_pair(
                    "G01",
                    20_000_100.0,
                    20_000_120.0,
                    105_100_500.0,
                    105_100_540.0,
                ),
                setup_pair(
                    "G02",
                    21_000_100.0,
                    21_000_135.0,
                    110_200_500.0,
                    110_200_590.0,
                ),
                setup_pair(
                    "G03",
                    22_000_100.0,
                    22_000_155.0,
                    115_300_500.0,
                    115_300_620.0,
                ),
            ],
            base_satellite_positions_m: satellite_positions.clone(),
            rover_satellite_positions_m: satellite_positions,
        },
    ];
    let wide_lanes = BTreeMap::from([("G02".to_string(), 3), ("G03".to_string(), -5)]);

    let off = prepare_ionosphere_free_baseline_epochs(
        base_m,
        initial_baseline_m,
        &epochs,
        "G01",
        &wide_lanes,
        false,
    )
    .unwrap();
    let on = prepare_ionosphere_free_baseline_epochs(
        base_m,
        initial_baseline_m,
        &epochs,
        "G01",
        &wide_lanes,
        true,
    )
    .unwrap();

    assert_eq!(
        off.epochs
            .iter()
            .map(|epoch| (
                epoch.epoch_index,
                epoch
                    .base_observations
                    .iter()
                    .map(|obs| (obs.satellite_id.as_str(), obs.code_m.to_bits()))
                    .collect::<Vec<_>>(),
                epoch
                    .rover_observations
                    .iter()
                    .map(|obs| (obs.satellite_id.as_str(), obs.phase_m.to_bits()))
                    .collect::<Vec<_>>()
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                0,
                vec![
                    ("G01", 0x4173_12cf_ce89_65e4),
                    ("G02", 0x4174_06f3_ce89_65e4),
                    ("G03", 0x4174_fb17_ce89_65e4),
                ],
                vec![
                    ("G01", 0x4165_70ac_ae81_9db4),
                    ("G02", 0x4166_7b04_20f1_cbd0),
                    ("G03", 0x4167_855b_4eee_bc18),
                ],
            ),
            (
                1,
                vec![
                    ("G01", 0x4173_12d6_0e89_65e4),
                    ("G02", 0x4174_06fa_0e89_65e4),
                    ("G03", 0x4174_fb1e_0e89_65e4),
                ],
                vec![
                    ("G01", 0x4165_70b3_5dc2_a728),
                    ("G02", 0x4166_7b0a_d032_d540),
                    ("G03", 0x4167_8561_fe2f_c588),
                ],
            ),
        ]
    );
    assert_eq!(
        on.epochs
            .iter()
            .map(|epoch| (
                epoch.epoch_index,
                epoch
                    .base_observations
                    .iter()
                    .map(|obs| (obs.satellite_id.as_str(), obs.code_m.to_bits()))
                    .collect::<Vec<_>>(),
                epoch
                    .rover_observations
                    .iter()
                    .map(|obs| (obs.satellite_id.as_str(), obs.phase_m.to_bits()))
                    .collect::<Vec<_>>()
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                0,
                vec![
                    ("G01", 0x4173_12cf_b2d0_af82),
                    ("G02", 0x4174_06f3_b2ae_adba),
                    ("G03", 0x4174_fb17_b2b0_8066),
                ],
                vec![
                    ("G01", 0x4165_70ac_7719_d56e),
                    ("G02", 0x4166_7b03_e946_0b7d),
                    ("G03", 0x4167_855b_1746_a11b),
                ],
            ),
            (
                1,
                vec![
                    ("G01", 0x4173_12d5_f2d0_af82),
                    ("G02", 0x4174_06f9_f2ae_adba),
                    ("G03", 0x4174_fb1d_f2b0_8066),
                ],
                vec![
                    ("G01", 0x4165_70b3_265a_dee2),
                    ("G02", 0x4166_7b0a_9887_14ed),
                    ("G03", 0x4167_8561_c687_aa8b),
                ],
            ),
        ]
    );
}
