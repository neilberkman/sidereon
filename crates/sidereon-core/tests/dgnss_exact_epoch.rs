#![cfg(sidereon_repo_tests)]

use std::cell::RefCell;
use std::collections::BTreeMap;

use sidereon_core::astro::time::{DegradeReason, ExactEpoch, ExactEpochQuery, Validated};
use sidereon_core::constants::{C_M_S, OMEGA_E_DOT_RAD_S};
use sidereon_core::dgnss::{solve_position, CodeObservation, DgnssError};
use sidereon_core::observables::{ObservableEphemerisSource, ObservableState, ObservablesError};
use sidereon_core::positioning::{
    ClockRelativity, Corrections, EphemerisSource, KlobucharCoeffs, Observation, PseudorangeCode,
    QzssClock, SolveInputs, SurfaceMet, TroposphereModel,
};
use sidereon_core::{Error, GnssSatelliteId};

const RECEIVE_EPOCH_J2000_S: f64 = 900_000_000.0;
const RAW_PSEUDORANGE_M: f64 = 22_123_456.789_123;
const PLACEMENT_CLOCK_S: f64 = 0.000_123_456_789;
const SATELLITE_TOKEN: &str = "G01";

#[derive(Clone, Copy)]
enum RefusalPoint {
    PlacementClock,
    SelectedState,
}

struct ExactEpochSource {
    refusal_point: RefusalPoint,
    placement_clock_queries: RefCell<Vec<(ExactEpochQuery, ExactEpochQuery)>>,
    selected_state_queries: RefCell<Vec<(ExactEpochQuery, ExactEpochQuery)>>,
}

impl ExactEpochSource {
    fn new(refusal_point: RefusalPoint) -> Self {
        Self {
            refusal_point,
            placement_clock_queries: RefCell::new(Vec::new()),
            selected_state_queries: RefCell::new(Vec::new()),
        }
    }

    fn refusal(&self) -> Error {
        Error::Ut1OutsideCoverage(DegradeReason::BeforeCoverage)
    }
}

impl ObservableEphemerisSource for ExactEpochSource {
    fn observable_state_at_j2000_s(
        &self,
        _satellite: GnssSatelliteId,
        _epoch_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        panic!("DGNSS exact-epoch route must not use the scalar observable callback")
    }
}

impl EphemerisSource for ExactEpochSource {
    fn position_clock_at_j2000_s(
        &self,
        _satellite: GnssSatelliteId,
        _epoch_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        panic!("DGNSS exact-epoch route must not use the scalar state callback")
    }

    fn try_transmit_epoch_clock_at_epoch_query(
        &self,
        _satellite: GnssSatelliteId,
        epoch: &ExactEpochQuery,
        selection_epoch: &ExactEpochQuery,
    ) -> Result<Option<Validated<f64>>, Error> {
        self.placement_clock_queries
            .borrow_mut()
            .push((epoch.clone(), selection_epoch.clone()));
        match self.refusal_point {
            RefusalPoint::PlacementClock => Err(self.refusal()),
            RefusalPoint::SelectedState => Ok(Some(Validated::ok(PLACEMENT_CLOCK_S))),
        }
    }

    fn try_position_clock_group_delay_selected_at_epoch_query(
        &self,
        _satellite: GnssSatelliteId,
        epoch: &ExactEpochQuery,
        selection_epoch: &ExactEpochQuery,
    ) -> Result<Option<Validated<([f64; 3], f64, Option<f64>)>>, Error> {
        self.selected_state_queries
            .borrow_mut()
            .push((epoch.clone(), selection_epoch.clone()));
        Err(self.refusal())
    }

    fn clock_relativity_s(
        &self,
        _satellite: GnssSatelliteId,
        _epoch_j2000_s: f64,
    ) -> ClockRelativity {
        panic!("DGNSS exact-epoch refusal must occur before clock relativity")
    }
}

fn solve_inputs() -> SolveInputs {
    SolveInputs {
        observations: Vec::<Observation>::new(),
        t_rx_j2000_s: RECEIVE_EPOCH_J2000_S,
        t_rx_second_of_day_s: 43_200.0,
        day_of_year: 176.5,
        initial_guess: [0.0; 4],
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
        pseudorange_code: PseudorangeCode::SingleFrequency,
        qzss_clock: QzssClock::Gps,
        troposphere_model: TroposphereModel::Rtklib,
    }
}

fn assert_exact_queries(
    actual: &(ExactEpochQuery, ExactEpochQuery),
    expected_transmit: &ExactEpochQuery,
    expected_receive: &ExactEpochQuery,
) {
    assert_eq!(&actual.0, expected_transmit);
    assert_eq!(&actual.1, expected_receive);
}

fn assert_ut1_refusal(refusal_point: RefusalPoint) {
    let source = ExactEpochSource::new(refusal_point);
    let base_observations = [CodeObservation::new(SATELLITE_TOKEN, RAW_PSEUDORANGE_M)];
    let rover_observations = [CodeObservation::new(SATELLITE_TOKEN, RAW_PSEUDORANGE_M)];
    let result = solve_position(
        &source,
        [6_378_137.0, 0.0, 0.0],
        &base_observations,
        &rover_observations,
        solve_inputs(),
        false,
    );
    assert!(matches!(
        result,
        Err(DgnssError::Ut1OutsideCoverage(
            DegradeReason::BeforeCoverage
        ))
    ));

    let receive_epoch =
        ExactEpoch::from_binary_j2000_seconds(RECEIVE_EPOCH_J2000_S).expect("finite receive epoch");
    let expected_transmit = receive_epoch
        .clone()
        .checked_sub_binary_seconds(RAW_PSEUDORANGE_M / C_M_S)
        .expect("finite pseudorange offset");
    assert_ne!(
        expected_transmit,
        ExactEpoch::from_binary_j2000_seconds(expected_transmit.j2000_seconds())
            .expect("rounded epoch remains finite"),
        "fixture must exercise a non-grid exact offset"
    );
    let placement_queries = source.placement_clock_queries.borrow();
    assert_eq!(placement_queries.len(), 1);
    assert_exact_queries(&placement_queries[0], &expected_transmit, &receive_epoch);

    let selected_queries = source.selected_state_queries.borrow();
    match refusal_point {
        RefusalPoint::PlacementClock => assert!(selected_queries.is_empty()),
        RefusalPoint::SelectedState => {
            assert_eq!(selected_queries.len(), 1);
            let expected_state = expected_transmit
                .checked_sub_binary_seconds(PLACEMENT_CLOCK_S)
                .expect("finite placement clock offset");
            assert_exact_queries(&selected_queries[0], &expected_state, &receive_epoch);
        }
    }
}

#[test]
fn exact_placement_clock_ut1_refusal_is_reported_without_scalar_fallback() {
    assert_ut1_refusal(RefusalPoint::PlacementClock);
}

#[test]
fn exact_selected_state_ut1_refusal_is_reported_without_scalar_fallback() {
    assert_ut1_refusal(RefusalPoint::SelectedState);
}

struct ConstantSatelliteSource {
    positions_m: BTreeMap<u8, [f64; 3]>,
    group_delay_scale_s: Option<f64>,
}

impl ObservableEphemerisSource for ConstantSatelliteSource {
    fn observable_state_at_j2000_s(
        &self,
        _satellite: GnssSatelliteId,
        _epoch_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        panic!("DGNSS exact solver must use the ephemeris state callbacks")
    }
}

impl EphemerisSource for ConstantSatelliteSource {
    fn position_clock_at_j2000_s(
        &self,
        satellite: GnssSatelliteId,
        _epoch_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        self.positions_m
            .get(&satellite.prn)
            .copied()
            .map(|position_m| (position_m, 0.0))
    }

    fn single_frequency_group_delay_s(
        &self,
        satellite: GnssSatelliteId,
        _epoch_j2000_s: f64,
    ) -> Option<f64> {
        self.group_delay_scale_s
            .map(|scale_s| scale_s * f64::from(satellite.prn))
    }

    fn clock_relativity_s(
        &self,
        _satellite: GnssSatelliteId,
        _epoch_j2000_s: f64,
    ) -> ClockRelativity {
        ClockRelativity::NotApplicable
    }
}

fn assert_float_bits_equal(actual: f64, expected: f64) {
    assert_eq!(actual.to_bits(), expected.to_bits());
}

#[test]
fn ionosphere_free_dgnss_ignores_base_and_rover_group_delay() {
    let base_position_m = [6_378_137.0, 0.0, 0.0];
    let offsets_m = [
        [20_000_000.0, 0.0, 0.0],
        [20_000_000.0, 10_000_000.0, 0.0],
        [20_000_000.0, -10_000_000.0, 0.0],
        [20_000_000.0, 0.0, 10_000_000.0],
        [20_000_000.0, 0.0, -10_000_000.0],
    ];
    let positions_m: BTreeMap<u8, [f64; 3]> = offsets_m
        .iter()
        .enumerate()
        .map(|(index, offset_m)| {
            (
                u8::try_from(index + 1).expect("five fixture satellites fit in a PRN"),
                [
                    base_position_m[0] + offset_m[0],
                    base_position_m[1] + offset_m[1],
                    base_position_m[2] + offset_m[2],
                ],
            )
        })
        .collect();
    let observations: Vec<CodeObservation> = positions_m
        .iter()
        .map(|(prn, satellite_position_m)| {
            let difference_m = [
                satellite_position_m[0] - base_position_m[0],
                satellite_position_m[1] - base_position_m[1],
                satellite_position_m[2] - base_position_m[2],
            ];
            let euclidean_range_m = ((difference_m[0] * difference_m[0]
                + difference_m[1] * difference_m[1])
                + difference_m[2] * difference_m[2])
                .sqrt();
            let sagnac_range_m = OMEGA_E_DOT_RAD_S
                * (satellite_position_m[0] * base_position_m[1]
                    - satellite_position_m[1] * base_position_m[0])
                / C_M_S;
            CodeObservation::new(format!("G{prn:02}"), euclidean_range_m + sagnac_range_m)
        })
        .collect();

    let solve_with_group_delay = |group_delay_scale_s| {
        let source = ConstantSatelliteSource {
            positions_m: positions_m.clone(),
            group_delay_scale_s,
        };
        let mut inputs = solve_inputs();
        inputs.initial_guess = [
            base_position_m[0],
            base_position_m[1],
            base_position_m[2],
            0.0,
        ];
        inputs.pseudorange_code = PseudorangeCode::IonosphereFree;
        solve_position(
            &source,
            base_position_m,
            &observations,
            &observations,
            inputs,
            false,
        )
        .expect("full-rank constant satellite geometry solves")
    };

    let no_group_delay = solve_with_group_delay(None);
    let per_satellite_delay = solve_with_group_delay(Some(1.0e-8));
    let nonfinite_group_delay = solve_with_group_delay(Some(f64::NAN));
    assert_eq!(no_group_delay.solution.used_sats.len(), offsets_m.len());

    for compared in [&per_satellite_delay, &nonfinite_group_delay] {
        for (actual, expected) in compared
            .solution
            .position
            .as_array()
            .into_iter()
            .zip(no_group_delay.solution.position.as_array())
        {
            assert_float_bits_equal(actual, expected);
        }
        assert_float_bits_equal(
            compared.solution.rx_clock_s,
            no_group_delay.solution.rx_clock_s,
        );
        assert_eq!(
            compared.solution.system_clocks_s.len(),
            no_group_delay.solution.system_clocks_s.len()
        );
        for ((actual_system, actual_clock), (expected_system, expected_clock)) in compared
            .solution
            .system_clocks_s
            .iter()
            .zip(&no_group_delay.solution.system_clocks_s)
        {
            assert_eq!(actual_system, expected_system);
            assert_float_bits_equal(*actual_clock, *expected_clock);
        }
        assert_eq!(
            compared.solution.residuals_m.len(),
            no_group_delay.solution.residuals_m.len()
        );
        for (actual, expected) in compared
            .solution
            .residuals_m
            .iter()
            .zip(&no_group_delay.solution.residuals_m)
        {
            assert_float_bits_equal(*actual, *expected);
        }
        for (actual, expected) in compared
            .baseline_vector_m
            .iter()
            .zip(&no_group_delay.baseline_vector_m)
        {
            assert_float_bits_equal(*actual, *expected);
        }
        assert_float_bits_equal(compared.baseline_m, no_group_delay.baseline_m);
    }
}
