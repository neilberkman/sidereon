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
    // Re-frozen when the selection, elevation mask and weights moved to the current
    // iterate, as RTKLIB `estpos` re-runs `rescode`. The solve starts at the geocentre,
    // the default initial position, where the weights had stayed for the whole solve;
    // they are now the elevation weights at the solution, which moves the covariance and
    // the base weights, and the position by 1.6 mm. The solve also ends with RTKLIB's
    // least-squares step, counted as an iteration, whose rounding leaves the clocks of
    // the two identical epochs 24 ulp apart. The condition number is that of the
    // design RTKLIB steps with, `sqrt(W) [-e, 1]` at the solution.
    //
    // Re-frozen again when the fixture grew from five epochs to thirteen, because the
    // position interpolator refuses a run shorter than the eleven nodes RTKLIB pephpos
    // takes. With five nodes every satellite was a degree-4 fit; with eleven the model
    // reproduces the Go fixture's pseudoranges to within 3.7e-9 m (every residual is
    // 0 or 2^-28 m), where the degree-4 fit left residuals up to 7e-4 m. The solution
    // moves by 1.2 mm, the two identical epochs now share one clock, and the solve
    // takes nine iterations rather than ten.
    let frozen: Vec<(&str, Vec<u64>)> = vec![
        (
            "position",
            vec![0x41511b07ff82440b, 0x4120cd6b5f861f74, 0x41511e62229e1c36],
        ),
        (
            "per_epoch_clock",
            vec![0x3f1a3b884188ea82, 0x3f1a3b884188ea82],
        ),
        (
            "covariance_ecef",
            vec![
                0x400988cb35cb6e11,
                0x3fd48b9a0d51f1e2,
                0x4000274f931ac7c4,
                0x3fd48b9a0d51f1e2,
                0x3fe7a0ee62d73fe8,
                0x3fe106d7a566227f,
                0x4000274f931ac7c4,
                0x3fe106d7a566227f,
                0x40065c549e8921f4,
            ],
        ),
        (
            "covariance_state",
            vec![
                0x400988cb35cb6e11,
                0x3fd48b9a0d51f1e2,
                0x4000274f931ac7c4,
                0x4008c021d0623ceb,
                0x4008c021d0623cec,
                0x3fd48b9a0d51f1e2,
                0x3fe7a0ee62d73fe8,
                0x3fe106d7a566227f,
                0x3fe263bc6a6c1354,
                0x3fe263bc6a6c1357,
                0x4000274f931ac7c4,
                0x3fe106d7a566227f,
                0x40065c549e8921f4,
                0x4006a259132f61ab,
                0x4006a259132f61af,
                0x4008c021d0623ceb,
                0x3fe263bc6a6c1354,
                0x4006a259132f61ab,
                0x400dccf820cc54a6,
                0x400c11bbda6ca6ae,
                0x4008c021d0623cec,
                0x3fe263bc6a6c1357,
                0x4006a259132f61af,
                0x400c11bbda6ca6ae,
                0x400dccf820cc54ac,
            ],
        ),
        ("iterations", vec![0x9]),
        ("condition_number", vec![0x40274ea1c23d4ba0]),
        ("gdop", vec![0x400e1ec71a66a2e7]),
        (
            "residuals",
            vec![
                0x0,
                0x0,
                0x3e30000000000000,
                0x3e30000000000000,
                0xbe30000000000000,
                0x3e30000000000000,
                0x3e30000000000000,
                0xbe30000000000000,
                0x0,
                0x0,
                0x3e30000000000000,
                0x3e30000000000000,
                0xbe30000000000000,
                0x3e30000000000000,
                0x3e30000000000000,
                0xbe30000000000000,
            ],
        ),
        (
            "base_weights",
            vec![
                0x3fb8cb465b587ec3,
                0x3fd44d0c2b9a2f6a,
                0x3fed27a1c8576fdb,
                0x3fddb593331039dd,
                0x3fe3378d7e2df301,
                0x3fef228ba75cba43,
                0x3fe5ac56ce184d4e,
                0x3fe292e96d2254b2,
                0x3fb8cb465b587ec3,
                0x3fd44d0c2b9a2f6a,
                0x3fed27a1c8576fdb,
                0x3fddb593331039dd,
                0x3fe3378d7e2df301,
                0x3fef228ba75cba43,
                0x3fe5ac56ce184d4e,
                0x3fe292e96d2254b2,
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
