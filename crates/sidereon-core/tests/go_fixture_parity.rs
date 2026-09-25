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
        qzss_clock: sidereon_core::positioning::QzssClock::Gps,
        troposphere_model: sidereon_core::positioning::TroposphereModel::Rtklib,
    }
}

fn bit_pattern(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn static_result_bit_fields(
    result: &sidereon_core::static_positioning::StaticSolution,
) -> Vec<(&'static str, Vec<u64>)> {
    let covariance_ecef = result
        .covariance
        .position_ecef_m2
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let covariance_state = result
        .covariance
        .state_m2
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    vec![
        ("position", bit_pattern(&result.position.as_array())),
        (
            "per_epoch_clock",
            result
                .per_epoch_clock
                .iter()
                .map(|clock| clock.clock_s.to_bits())
                .collect(),
        ),
        ("covariance_ecef", bit_pattern(&covariance_ecef)),
        ("covariance_state", bit_pattern(&covariance_state)),
        ("iterations", vec![result.metadata.iterations as u64]),
        (
            "condition_number",
            vec![result.geometry_quality.condition_number.to_bits()],
        ),
        ("gdop", vec![result.geometry_quality.gdop.to_bits()]),
        (
            "residuals",
            result
                .residuals_m
                .iter()
                .map(|row| row.residual_m.to_bits())
                .collect(),
        ),
        (
            "base_weights",
            result
                .residuals_m
                .iter()
                .map(|row| row.base_weight.to_bits())
                .collect(),
        ),
    ]
}

/// The SP3 source through the no-term path: its clock as written, with no `peph2pos`
/// relativistic term. The Go fixture's pseudoranges come from a model without the term;
/// positioning applies it through the source callback.
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

    fn try_transmit_epoch_clock_at_epoch_query(
        &self,
        sat: GnssSatelliteId,
        epoch: &sidereon_core::astro::time::ExactEpochQuery,
        selection_epoch: &sidereon_core::astro::time::ExactEpochQuery,
    ) -> Result<Option<sidereon_core::astro::time::Validated<f64>>, sidereon_core::Error> {
        EphemerisSource::try_transmit_epoch_clock_at_epoch_query(
            self.0,
            sat,
            epoch,
            selection_epoch,
        )
    }

    fn ephemeris_variance_m2(
        &self,
        sat: GnssSatelliteId,
        state_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> f64 {
        EphemerisSource::ephemeris_variance_m2(self.0, sat, state_j2000_s, selection_j2000_s)
    }

    fn ephemeris_variance_at_epoch_query(
        &self,
        sat: GnssSatelliteId,
        state_epoch: &sidereon_core::astro::time::ExactEpochQuery,
        selection_epoch: &sidereon_core::astro::time::ExactEpochQuery,
    ) -> f64 {
        EphemerisSource::ephemeris_variance_at_epoch_query(
            self.0,
            sat,
            state_epoch,
            selection_epoch,
        )
    }

    fn try_position_clock_group_delay_selected_at_epoch_query(
        &self,
        sat: GnssSatelliteId,
        epoch: &sidereon_core::astro::time::ExactEpochQuery,
        selection_epoch: &sidereon_core::astro::time::ExactEpochQuery,
    ) -> Result<
        Option<sidereon_core::astro::time::Validated<([f64; 3], f64, Option<f64>)>>,
        sidereon_core::Error,
    > {
        let Some(state) = EphemerisSource::try_position_clock_group_delay_selected_at_epoch_query(
            self.0,
            sat,
            epoch,
            selection_epoch,
        )?
        else {
            return Ok(None);
        };
        let (position, clock, group_delay) = state.value;
        let clock = match EphemerisSource::clock_relativity_for_state_at_epoch_query(
            self.0, sat, epoch, position,
        ) {
            sidereon_core::positioning::ClockRelativity::NotApplicable => clock,
            sidereon_core::positioning::ClockRelativity::Term(term) => clock + term,
            sidereon_core::positioning::ClockRelativity::Unavailable => return Ok(None),
        };
        Ok(Some(sidereon_core::astro::time::Validated {
            value: (position, clock, group_delay),
            degraded: state.degraded,
        }))
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
        static_result_bit_fields(&through_source),
        static_result_bit_fields(&folded),
        "source-applied and folded relativity static-result fields differ"
    );
}

/// The Go fixture's pseudoranges come from a geometric light-time model, which iterates
/// the transmission epoch from the receiver's time tag and leaves out the receiver clock
/// (about 0.1 ms here). This test checks repeated-run bit determinism through
/// [`solve_static_geometric_light_time_replay`], not numerical accuracy; the independent
/// precise oracle is separate.
#[test]
fn go_fixture_static_repeated_run_bits_are_deterministic() {
    let sp3 = fixture_sp3();
    let source = NoRelativityTerm(&sp3);
    let inputs = go_fixture_inputs();
    let make_epochs = || {
        [
            StaticEpoch::from_solve_inputs(inputs.clone()),
            StaticEpoch::from_solve_inputs(inputs.clone()),
        ]
    };
    let static_result = solve_static_geometric_light_time_replay(
        &source,
        &make_epochs(),
        StaticSolveOptions::default(),
    )
    .expect("static solve");
    let repeated_result = solve_static_geometric_light_time_replay(
        &source,
        &make_epochs(),
        StaticSolveOptions::default(),
    )
    .expect("repeated static solve");

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
            StaticEpoch::from_solve_inputs(inputs.clone()),
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

    assert_eq!(
        static_result_bit_fields(&static_result),
        static_result_bit_fields(&repeated_result),
        "repeated Go fixture static solves are deterministic"
    );
}
