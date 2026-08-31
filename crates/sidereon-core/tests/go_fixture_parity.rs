use std::collections::BTreeMap;

use sidereon_core::ephemeris::Sp3;
use sidereon_core::positioning::{
    Corrections, KlobucharCoeffs, Observation, SolveInputs, SurfaceMet,
};
use sidereon_core::static_positioning::{solve_static, StaticEpoch, StaticSolveOptions};
use sidereon_core::{GnssSatelliteId, GnssSystem};

fn fixture_sp3() -> Sp3 {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sp3/trimmed_go_static.sp3"
    );
    let bytes = std::fs::read(path).expect("read SP3 fixture");
    Sp3::parse(&bytes).expect("parse SP3 fixture")
}

fn gps(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS PRN")
}

fn go_fixture_inputs() -> SolveInputs {
    let observations = [
        (8, 23_825_519.844459895),
        (10, 22_717_690.10174763),
        (16, 20_478_653.376262885),
        (18, 21_768_335.23365917),
        (20, 21_248_327.738292538),
        (21, 20_808_709.800933376),
        (26, 21_126_481.58786735),
        (27, 21_341_367.541037586),
    ]
    .into_iter()
    .map(|(prn, pseudorange_m)| Observation {
        satellite_id: gps(prn),
        pseudorange_m,
    })
    .collect();
    SolveInputs {
        observations,
        t_rx_j2000_s: 646_272_000.0,
        t_rx_second_of_day_s: 43_200.0,
        day_of_year: 176.5,
        initial_guess: [4.5e6, 0.5e6, 4.5e6, 0.0],
        corrections: Corrections::NONE,
        klobuchar: KlobucharCoeffs {
            alpha: [0.0; 4],
            beta: [0.0; 4],
        },
        beidou_klobuchar: None,
        galileo_nequick: None,
        sbas_iono: None,
        glonass_channels: BTreeMap::new(),
        met: SurfaceMet::default(),
        robust: None,
    }
}

fn bit_pattern(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn go_fixture_static_portable_bits() {
    let source = fixture_sp3();
    let inputs = go_fixture_inputs();
    let first = StaticEpoch::from_solve_inputs(inputs.clone());
    let second = StaticEpoch::from_solve_inputs(inputs);
    let static_result = solve_static(&source, &[first, second], StaticSolveOptions::default())
        .expect("static solve");

    assert_eq!(
        bit_pattern(&static_result.position.as_array()),
        vec![0x41511b07ff6d7461, 0x4120cd6b5f0f3fb6, 0x41511e622290ed5b,]
    );
    assert_eq!(
        static_result
            .per_epoch_clock
            .iter()
            .map(|clock| clock.clock_s.to_bits())
            .collect::<Vec<_>>(),
        vec![0x3f1a3b88234bcff9, 0x3f1a3b88234bcff9]
    );
    let ecef = static_result
        .covariance
        .position_ecef_m2
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(
        bit_pattern(&ecef),
        vec![
            0x401782d26e36cfd9,
            0x3faa3d846f455d90,
            0x4004fb98e648c404,
            0x3faa3d846f455d90,
            0x3ff6b38b995266fb,
            0x3febe4866fbce509,
            0x4004fb98e648c404,
            0x3febe4866fbce509,
            0x4008045be20a8ea6,
        ]
    );
    let state = static_result
        .covariance
        .state_m2
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(
        bit_pattern(&state),
        vec![
            0x401782d26e36cfd9,
            0x3faa3d846f455d90,
            0x4004fb98e648c404,
            0x40150690b6fe685a,
            0x40150690b6fe685a,
            0x3faa3d846f455d90,
            0x3ff6b38b995266fb,
            0x3febe4866fbce509,
            0x3fe31f309a58b4e9,
            0x3fe31f309a58b4e7,
            0x4004fb98e648c404,
            0x3febe4866fbce509,
            0x4008045be20a8ea6,
            0x4008f69cda22e0da,
            0x4008f69cda22e0da,
            0x40150690b6fe685a,
            0x3fe31f309a58b4e9,
            0x4008f69cda22e0da,
            0x40156e6be7e8618f,
            0x40144247fc8a61fe,
            0x40150690b6fe685a,
            0x3fe31f309a58b4e7,
            0x4008f69cda22e0da,
            0x40144247fc8a61fe,
            0x40156e6be7e8618f,
        ]
    );
    assert_eq!(static_result.metadata.iterations, 9);
    assert_eq!(
        static_result.geometry_quality.condition_number.to_bits(),
        0x402846c3b2c6388d
    );
    assert_eq!(
        static_result.geometry_quality.gdop.to_bits(),
        0x4012562a1c8a19f4
    );
    assert_eq!(
        static_result
            .residuals_m
            .iter()
            .map(|row| row.residual_m.to_bits())
            .collect::<Vec<_>>(),
        vec![
            0x3f40cd1000000000,
            0xbf378e0000000000,
            0xbf38c9e000000000,
            0x3f5161ec00000000,
            0x3f2a2b0000000000,
            0xbf30255000000000,
            0x3f3c5ed000000000,
            0xbefdad0000000000,
            0x3f40cd1000000000,
            0xbf378e0000000000,
            0xbf38c9e000000000,
            0x3f5161ec00000000,
            0x3f2a2b0000000000,
            0xbf30255000000000,
            0x3f3c5ed000000000,
            0xbefdad0000000000,
        ]
    );
    assert_eq!(
        static_result
            .residuals_m
            .iter()
            .map(|row| row.base_weight.to_bits())
            .collect::<Vec<_>>(),
        vec![
            0x3fb439f6b0724321,
            0x3fe9b0257691870c,
            0x3fe006d19c286312,
            0x3fb028d0c17f3f79,
            0x3fdd0ad788b89a4f,
            0x3fd93783778cfc91,
            0x3fec59af7f4c7937,
            0x3fcdbb1a1c321f3e,
            0x3fb439f6b0724321,
            0x3fe9b0257691870c,
            0x3fe006d19c286312,
            0x3fb028d0c17f3f79,
            0x3fdd0ad788b89a4f,
            0x3fd93783778cfc91,
            0x3fec59af7f4c7937,
            0x3fcdbb1a1c321f3e,
        ]
    );
}
