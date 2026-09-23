#![cfg(sidereon_repo_tests)]

use std::collections::BTreeMap;

use sidereon_core::ephemeris::Sp3;
use sidereon_core::positioning::{
    Corrections, EphemerisSource, KlobucharCoeffs, Observation, SolveInputs, SurfaceMet,
};
use sidereon_core::static_positioning::{
    solve_static, solve_static_geometric_light_time_replay, StaticEpoch, StaticSolveOptions,
};
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
        pseudorange_code: sidereon_core::positioning::PseudorangeCode::SingleFrequency,
    }
}

fn bit_pattern(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The SP3 source through the no-term path: its clock as written, with no `peph2pos`
/// relativistic term. The Go fixture's pseudoranges come from a model without the term,
/// so the frozen bits are those of the no-term path; positioning applies the term, and
/// the SPP trace tests check that the term is the only difference between the paths.
struct NoRelativityTerm<'a>(&'a Sp3);

impl EphemerisSource for NoRelativityTerm<'_> {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        EphemerisSource::position_clock_at_j2000_s(self.0, sat, t_j2000_s)
    }
}

/// The SP3 source with the `peph2pos` term folded into its clock, and no term method:
/// a source whose clock already carries the term. The clock that places the
/// transmission epoch stays the product's, without the term, as RTKLIB `satposs` places
/// it with `ephclk`, which applies no `peph2pos` term.
struct ClockWithTerm<'a>(&'a Sp3);

impl EphemerisSource for ClockWithTerm<'_> {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        let (position, clock) = EphemerisSource::position_clock_at_j2000_s(self.0, sat, t_j2000_s)?;
        let term = EphemerisSource::clock_relativity_s(self.0, sat, t_j2000_s).term()?;
        Some((position, clock + term))
    }

    fn try_transmit_epoch_clock_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<Option<sidereon_core::astro::time::Validated<f64>>, sidereon_core::Error> {
        EphemerisSource::try_transmit_epoch_clock_s(self.0, sat, t_j2000_s, selection_j2000_s)
    }
}

/// Applying the term through the source's `clock_relativity_s` and folding it into the
/// clock give the same static solution bit for bit: the model adds the term once, at the
/// epoch the clock came from.
#[test]
fn go_fixture_static_term_through_the_source_equals_the_folded_clock() {
    let sp3 = fixture_sp3();
    let inputs = go_fixture_inputs();
    let epochs = || {
        [
            StaticEpoch::from_solve_inputs(inputs.clone()),
            StaticEpoch::from_solve_inputs(inputs.clone()),
        ]
    };
    let through_source =
        solve_static(&sp3, &epochs(), StaticSolveOptions::default()).expect("static solve");
    let folded = solve_static(
        &ClockWithTerm(&sp3),
        &epochs(),
        StaticSolveOptions::default(),
    )
    .expect("static solve with the folded clock");
    assert_eq!(
        through_source.position.as_array().map(f64::to_bits),
        folded.position.as_array().map(f64::to_bits)
    );
    assert_eq!(
        through_source
            .per_epoch_clock
            .iter()
            .map(|clock| clock.clock_s.to_bits())
            .collect::<Vec<_>>(),
        folded
            .per_epoch_clock
            .iter()
            .map(|clock| clock.clock_s.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        through_source
            .residuals_m
            .iter()
            .map(|row| row.residual_m.to_bits())
            .collect::<Vec<_>>(),
        folded
            .residuals_m
            .iter()
            .map(|row| row.residual_m.to_bits())
            .collect::<Vec<_>>()
    );
}

/// The Go fixture's pseudoranges come from a geometric light-time model, which iterates
/// the transmission epoch from the receiver's time tag and leaves out the receiver clock
/// (about 0.1 ms here). The frozen bits are that model's, replayed through
/// [`solve_static_geometric_light_time_replay`]; the static solve places each epoch from
/// the pseudorange as RTKLIB `satposs` does, and the in-crate static tests check that the
/// two differ through the transmission epoch alone.
#[test]
fn go_fixture_static_portable_bits() {
    let sp3 = fixture_sp3();
    let source = NoRelativityTerm(&sp3);
    let inputs = go_fixture_inputs();
    let first = StaticEpoch::from_solve_inputs(inputs.clone());
    let second = StaticEpoch::from_solve_inputs(inputs.clone());
    let static_result = solve_static_geometric_light_time_replay(
        &source,
        &[first, second],
        StaticSolveOptions::default(),
    )
    .expect("static solve");

    // With the term the same pseudoranges solve elsewhere: every satellite has a nonzero
    // term, and the solution moves by metres, not by rounding.
    for observation in &inputs.observations {
        let term =
            EphemerisSource::clock_relativity_s(&sp3, observation.satellite_id, 646_272_000.0)
                .term()
                .expect("peph2pos term");
        assert!(
            term != 0.0 && term.abs() < 1.0e-7,
            "{}: {term}",
            observation.satellite_id
        );
    }
    let with_term = solve_static_geometric_light_time_replay(
        &sp3,
        &[
            StaticEpoch::from_solve_inputs(inputs.clone()),
            StaticEpoch::from_solve_inputs(inputs),
        ],
        StaticSolveOptions::default(),
    )
    .expect("static solve with the term");
    let moved = with_term
        .position
        .as_array()
        .iter()
        .zip(static_result.position.as_array())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    assert!(
        moved > 0.1 && moved < 100.0,
        "the term moves the solution by {moved} m"
    );

    let ecef = static_result
        .covariance
        .position_ecef_m2
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let state = static_result
        .covariance
        .state_m2
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    // Every frozen value is compared at once and all are printed on a mismatch.
    let got: Vec<(&str, Vec<u64>)> = vec![
        ("position", bit_pattern(&static_result.position.as_array())),
        (
            "per_epoch_clock",
            static_result
                .per_epoch_clock
                .iter()
                .map(|clock| clock.clock_s.to_bits())
                .collect(),
        ),
        ("covariance_ecef", bit_pattern(&ecef)),
        ("covariance_state", bit_pattern(&state)),
        ("iterations", vec![static_result.metadata.iterations as u64]),
        (
            "condition_number",
            vec![static_result.geometry_quality.condition_number.to_bits()],
        ),
        ("gdop", vec![static_result.geometry_quality.gdop.to_bits()]),
        (
            "residuals",
            static_result
                .residuals_m
                .iter()
                .map(|row| row.residual_m.to_bits())
                .collect(),
        ),
        (
            "base_weights",
            static_result
                .residuals_m
                .iter()
                .map(|row| row.base_weight.to_bits())
                .collect(),
        ),
    ];
    let frozen: Vec<(&str, Vec<u64>)> = vec![
        (
            "position",
            vec![0x41511b07ff6d7461, 0x4120cd6b5f0f3fb6, 0x41511e622290ed5b],
        ),
        (
            "per_epoch_clock",
            vec![0x3f1a3b88234bcff9, 0x3f1a3b88234bcff9],
        ),
        (
            "covariance_ecef",
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
            ],
        ),
        (
            "covariance_state",
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
            ],
        ),
        ("iterations", vec![9]),
        ("condition_number", vec![0x402846c3b2c6388d]),
        ("gdop", vec![0x4012562a1c8a19f4]),
        (
            "residuals",
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
            ],
        ),
        (
            "base_weights",
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
            ],
        ),
    ];
    assert_eq!(
        got.iter()
            .map(|(label, bits)| format!("{label}: {bits:#x?}"))
            .collect::<Vec<_>>(),
        frozen
            .iter()
            .map(|(label, bits)| format!("{label}: {bits:#x?}"))
            .collect::<Vec<_>>(),
        "Go fixture static frozen bits"
    );
}
