use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::astro::time::ExactEpochQuery;
use crate::id::GnssSatelliteId;

use super::atmosphere_certificate::{
    self, CenterGeodeticEnclosure, CenterSatelliteAngles, KlobucharCenterError,
};
use super::broadcast_certificate::GpsStateEnclosure;
use super::centre_certificate::{self, CenterCertificateError, CenterInputs, CenterIntervals};
use super::covariance_certificate;
use super::interval_certificate::Interval;
use super::regional_certificate::{self, SatelliteRegion};
use super::{
    ClockRelativity, Corrections, EphemerisSource, GnssSystem, PseudorangeCode, Selection,
    SolveInputs, SppModelRecipe, TroposphereModel, C_M_S, ELEVATION_MASK_RAD,
};
use crate::constants::{WGS84_A_M, WGS84_F};

const REGION_RADIUS_M: f64 = 1.0;

#[cfg(test)]
mod tests {
    use super::{klobuchar_error_for_corrections, midpoint, radius, Interval};
    use crate::spp::Corrections;

    #[test]
    fn radius_encloses_both_endpoints_after_midpoint_rounding() {
        for interval in [
            Interval::new(1.0, f64::from_bits(1.0_f64.to_bits() + 1)),
            Interval::new(f64::from_bits((-1.0_f64).to_bits() + 1), -1.0),
            Interval::new(0.0, f64::from_bits(1)),
        ] {
            let center = midpoint(interval);
            let enclosure = radius(interval);
            assert!(enclosure >= (interval.lower() - center).abs());
            assert!(enclosure >= (interval.upper() - center).abs());
        }
    }

    #[test]
    fn fixture_correction_configurations_only_require_active_ionosphere() {
        let fixtures = [
            ("esbc_iono_tropo", true, true),
            ("esbc_tropo", false, true),
            ("esbc_iono", true, false),
            ("wtzr_iono_tropo", true, true),
            ("wtzr_iono", true, false),
        ];
        for (label, ionosphere, troposphere) in fixtures {
            let result = klobuchar_error_for_corrections(
                Corrections {
                    ionosphere,
                    troposphere,
                },
                || Err(super::EndpointCertificateError::UnsupportedInputs),
            );
            if ionosphere {
                assert!(
                    matches!(
                        result,
                        Err(super::EndpointCertificateError::UnsupportedInputs)
                    ),
                    "{label}: enabled ionosphere requires its center enclosure"
                );
            } else {
                let error = result.expect("disabled ionosphere has no center discrepancy");
                assert_eq!(error.raw_phi_i_semicircles, 0.0);
                assert_eq!(error.phi_m_semicircles, 0.0);
                assert_eq!(error.local_time_seconds, 0.0);
                assert_eq!(error.phase_radians, 0.0);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct IndependentSatelliteRow {
    pub satellite_id: GnssSatelliteId,
    pub design_row: [f64; 4],
    pub residual_m: f64,
    pub total_variance_m2: f64,
}

pub(super) struct OracleSatelliteState {
    pub transmit_epoch: ExactEpochQuery,
    pub position_ecef_m: [f64; 3],
    pub clock_s: f64,
    pub group_delay_s: f64,
    pub ephemeris_variance_m2: f64,
    pub orbit_evidence: IndependentStateEnclosure,
}

pub(super) struct IndependentStateEnclosure {
    pub transmit_epoch: ExactEpochQuery,
    pub orbit: GpsStateEnclosure,
}

pub(super) struct IndependentEndpoint {
    pub position_ecef_m: [f64; 3],
    pub receiver_clock_m: f64,
    pub c_geodetic_rad_m: [f64; 3],
    pub satellite_rows: Vec<IndependentSatelliteRow>,
    pub oracle_states: BTreeMap<GnssSatelliteId, OracleSatelliteState>,
    pub native_state_enclosures: BTreeMap<GnssSatelliteId, IndependentStateEnclosure>,
}

pub(super) struct IndependentLsqSnapshot {
    pub receiver_state: [f64; 4],
    pub step: [f64; 4],
    pub c_geodetic_rad_m: [f64; 3],
    pub weighted_design_columns: Vec<[f64; 4]>,
    pub covariance: [[f64; 4]; 4],
    pub reported_position_covariance_f32: [f32; 6],
}

pub(super) struct NativeEndpoint {
    pub position_ecef_m: [f64; 3],
    pub receiver_clock_m: f64,
    pub position_covariance_ecef_m2: [[f64; 3]; 3],
}

#[derive(Clone, Copy)]
struct ReceiverCenter {
    position_ecef_m: [f64; 3],
    receiver_clock_m: f64,
    finite_inverse_candidates_m: [[f64; 3]; 2],
}

#[derive(Debug, Clone)]
struct CandidateState {
    transmit_epoch: ExactEpochQuery,
    position_ecef_m: [f64; 3],
    clock_s: f64,
    group_delay_s: f64,
    ephemeris_variance_m2: f64,
}

#[derive(Debug)]
pub(super) enum EndpointCertificateError {
    UnsupportedInputs,
    InvalidEndpoint,
    SourceRefused,
    ArithmeticOutOfRange,
    MissingCandidateState(GnssSatelliteId),
    MissingNativeOrbitEnclosure(GnssSatelliteId),
    NativeStateEpochMismatch(GnssSatelliteId),
    NativeStateOutsideOrbitEnclosure(GnssSatelliteId),
    OracleStateOutsideOrbitEnclosure(GnssSatelliteId),
    CandidateClassification,
    SelectionChanged,
    ReferenceStateSet,
    NativeDesignOutsideInterval(GnssSatelliteId),
    NativeWeightOutsideInterval(GnssSatelliteId),
    NativeResidualOutsideInterval(GnssSatelliteId),
    OracleDesignOutsideInterval(GnssSatelliteId),
    OracleWeightOutsideInterval(GnssSatelliteId),
    OracleResidualOutsideInterval(GnssSatelliteId),
    Center(CenterCertificateError),
    Atmosphere(atmosphere_certificate::AtmosphereBoundError),
    RegionNotCertified,
    EndpointOutsideFixedBall,
    EndpointDistanceExceedsDerivedBound,
    FinalLsqReplayMismatch,
    FinalLsqColumnCount,
    FinalLsqDesignOutsideInterval,
    FinalLsqCovarianceOutsideInterval,
    FinalLsqPositionCovarianceOutsideCell,
    CPointCovarianceUnavailable,
    CPointCovarianceOutsideInterval,
    NativeCovarianceOutsideInterval,
}

impl core::fmt::Display for EndpointCertificateError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedInputs => formatter.write_str("unsupported inputs"),
            Self::InvalidEndpoint => formatter.write_str("invalid endpoint"),
            Self::SourceRefused => formatter.write_str("source refused"),
            Self::ArithmeticOutOfRange => formatter.write_str("arithmetic out of range"),
            Self::MissingCandidateState(satellite) => {
                write!(formatter, "missing candidate state: {satellite}")
            }
            Self::MissingNativeOrbitEnclosure(satellite) => {
                write!(formatter, "missing native orbit enclosure: {satellite}")
            }
            Self::NativeStateEpochMismatch(satellite) => {
                write!(formatter, "native state epoch mismatch: {satellite}")
            }
            Self::NativeStateOutsideOrbitEnclosure(satellite) => write!(
                formatter,
                "native state outside orbit enclosure: {satellite}"
            ),
            Self::OracleStateOutsideOrbitEnclosure(satellite) => write!(
                formatter,
                "oracle state outside orbit enclosure: {satellite}"
            ),
            Self::CandidateClassification => formatter.write_str("candidate classification"),
            Self::SelectionChanged => formatter.write_str("selection changed"),
            Self::ReferenceStateSet => formatter.write_str("reference state set"),
            Self::NativeDesignOutsideInterval(satellite) => {
                write!(formatter, "native design outside interval: {satellite}")
            }
            Self::NativeWeightOutsideInterval(satellite) => {
                write!(formatter, "native weight outside interval: {satellite}")
            }
            Self::NativeResidualOutsideInterval(satellite) => {
                write!(formatter, "native residual outside interval: {satellite}")
            }
            Self::OracleDesignOutsideInterval(satellite) => {
                write!(formatter, "oracle design outside interval: {satellite}")
            }
            Self::OracleWeightOutsideInterval(satellite) => {
                write!(formatter, "oracle weight outside interval: {satellite}")
            }
            Self::OracleResidualOutsideInterval(satellite) => {
                write!(formatter, "oracle residual outside interval: {satellite}")
            }
            Self::Center(cause) => write!(formatter, "center: {cause:?}"),
            Self::Atmosphere(cause) => write!(formatter, "atmosphere: {cause:?}"),
            Self::RegionNotCertified => formatter.write_str("region not certified"),
            Self::EndpointOutsideFixedBall => formatter.write_str("endpoint outside fixed ball"),
            Self::EndpointDistanceExceedsDerivedBound => {
                formatter.write_str("endpoint distance exceeds derived bound")
            }
            Self::FinalLsqReplayMismatch => formatter.write_str("final lsq replay mismatch"),
            Self::FinalLsqColumnCount => formatter.write_str("final lsq column count"),
            Self::FinalLsqDesignOutsideInterval => {
                formatter.write_str("final lsq design outside interval")
            }
            Self::FinalLsqCovarianceOutsideInterval => {
                formatter.write_str("final lsq covariance outside interval")
            }
            Self::FinalLsqPositionCovarianceOutsideCell => {
                formatter.write_str("final lsq position covariance outside cell")
            }
            Self::CPointCovarianceUnavailable => {
                formatter.write_str("cpoint covariance unavailable")
            }
            Self::CPointCovarianceOutsideInterval => {
                formatter.write_str("cpoint covariance outside interval")
            }
            Self::NativeCovarianceOutsideInterval => {
                formatter.write_str("native covariance outside interval")
            }
        }
    }
}

pub(super) struct EndpointCertificate {
    pub membership_distance_m: f64,
    pub endpoint_distance_bound_m: f64,
    pub contraction: f64,
    pub candidate_satellites: usize,
    pub used_satellites: usize,
}

pub(super) fn certify(
    source: &dyn EphemerisSource,
    inputs: &SolveInputs,
    receive_epoch: &ExactEpochQuery,
    reference: &IndependentEndpoint,
    native_endpoint: &NativeEndpoint,
    final_lsq: &IndependentLsqSnapshot,
) -> Result<EndpointCertificate, EndpointCertificateError> {
    validate_inputs(inputs, receive_epoch, reference, native_endpoint, final_lsq)?;
    let reference_clock_m = reference.receiver_clock_m;
    let reference_center = reference_center(reference, reference_clock_m);
    let solution_center = native_center(
        native_endpoint.position_ecef_m,
        native_endpoint.receiver_clock_m,
    )?;
    let lsq_center = oracle_prestate_center(final_lsq);
    let candidates = candidate_states(source, inputs, receive_epoch)?;
    check_native_orbit_enclosures(reference, &candidates)?;
    let candidate_positions: Vec<_> = candidates
        .iter()
        .map(|(satellite, state)| (*satellite, state.position_ecef_m))
        .collect();
    let model = SppModelRecipe::reference();
    let reference_selection = super::select_at_with_epoch(
        source,
        inputs,
        model,
        None,
        reference.position_ecef_m,
        &|_| reference_clock_m,
        Some(receive_epoch.clone()),
    );
    let solution_selection = super::select_at_with_epoch(
        source,
        inputs,
        model,
        None,
        native_endpoint.position_ecef_m,
        &|_| native_endpoint.receiver_clock_m,
        Some(receive_epoch.clone()),
    );
    if reference_selection.used != solution_selection.used {
        return Err(EndpointCertificateError::SelectionChanged);
    }
    check_reference_rows(reference, &reference_selection)?;

    let reference_centres =
        centre_intervals(inputs, &reference_center, &reference_selection, &candidates)?;
    let oracle_states = oracle_states(reference)?;
    let oracle_centres = centre_intervals(
        inputs,
        &reference_center,
        &reference_selection,
        &oracle_states,
    )?;
    check_oracle_intervals(reference, &oracle_centres)?;
    let solution_centres =
        centre_intervals(inputs, &solution_center, &solution_selection, &candidates)?;
    check_selection_intervals(&reference_selection, &reference_centres)?;
    check_selection_intervals(&solution_selection, &solution_centres)?;
    let c_endpoint_covariance = super::spp_position_covariance(
        &reference_selection.lines_of_sight,
        &vec![3; reference_selection.used.len()],
        4,
        &reference_selection.weights,
        super::geodetic_from_ecef(SppModelRecipe::reference().frame, reference.position_ecef_m),
    )
    .ok_or(EndpointCertificateError::CPointCovarianceUnavailable)?
    .ecef_m2;
    check_position_covariance(
        &reference_selection,
        &reference_centres,
        c_endpoint_covariance,
        CovarianceEndpoint::CPoint,
    )?;
    check_position_covariance(
        &solution_selection,
        &solution_centres,
        native_endpoint.position_covariance_ecef_m2,
        CovarianceEndpoint::Native,
    )?;
    let prestate_selection = super::select_at_with_epoch(
        source,
        inputs,
        model,
        None,
        lsq_center.position_ecef_m,
        &|_| lsq_center.receiver_clock_m,
        Some(receive_epoch.clone()),
    );
    if prestate_selection.used != reference_selection.used {
        return Err(EndpointCertificateError::SelectionChanged);
    }
    let prestate_oracle_centres =
        centre_intervals(inputs, &lsq_center, &prestate_selection, &oracle_states)?;
    check_final_lsq(
        reference,
        final_lsq,
        &prestate_selection,
        &prestate_oracle_centres,
    )?;

    let membership_distance_m = regional_certificate::endpoint_distance(
        state(reference.position_ecef_m, reference_clock_m),
        state(
            native_endpoint.position_ecef_m,
            native_endpoint.receiver_clock_m,
        ),
    );
    if membership_distance_m > REGION_RADIUS_M {
        return Err(EndpointCertificateError::EndpointOutsideFixedBall);
    }

    let center_geodetic = geodetic_enclosure(&reference_centres)?;
    let center_angles = candidate_angles(
        inputs,
        &reference_center,
        &candidates,
        &reference_selection,
        &reference_centres,
    )?;
    let atmosphere = atmosphere_certificate::satellite_regions(
        inputs,
        reference.position_ecef_m,
        REGION_RADIUS_M,
        &reference_selection,
        &candidate_positions,
        &center_geodetic,
        &center_angles,
    )
    .map_err(EndpointCertificateError::Atmosphere)?;
    let reference_regions = merge_regions(&reference_selection, &reference_centres, &atmosphere)?;
    let solution_regions = merge_regions(&solution_selection, &solution_centres, &atmosphere)?;

    let reference_step = selection_step(&reference_selection)?;
    let solution_step = selection_step(&solution_selection)?;
    let regional = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        regional_certificate::certify(
            &reference_selection,
            reference_step,
            &reference_regions,
            REGION_RADIUS_M,
        )
    }))
    .map_err(|_| EndpointCertificateError::RegionNotCertified)?;
    let reference_step_error =
        step_error(&reference_selection, reference_step, &reference_regions)?;
    let solution_step_error = step_error(&solution_selection, solution_step, &solution_regions)?;
    let endpoint_distance_bound_m = regional_certificate::endpoint_distance_bound(
        reference_step,
        solution_step,
        reference_step_error,
        solution_step_error,
        regional.contraction,
    );
    if membership_distance_m > endpoint_distance_bound_m {
        return Err(EndpointCertificateError::EndpointDistanceExceedsDerivedBound);
    }
    Ok(EndpointCertificate {
        membership_distance_m,
        endpoint_distance_bound_m,
        contraction: regional.contraction,
        candidate_satellites: candidates.len(),
        used_satellites: reference_selection.used.len(),
    })
}

fn reference_center(endpoint: &IndependentEndpoint, receiver_clock_m: f64) -> ReceiverCenter {
    let skyfield =
        super::geodetic_from_ecef(SppModelRecipe::reference().frame, endpoint.position_ecef_m);
    ReceiverCenter {
        position_ecef_m: endpoint.position_ecef_m,
        receiver_clock_m,
        finite_inverse_candidates_m: [
            [skyfield.lat_rad, skyfield.lon_rad, skyfield.height_m],
            endpoint.c_geodetic_rad_m,
        ],
    }
}

fn native_center(
    position_ecef_m: [f64; 3],
    receiver_clock_m: f64,
) -> Result<ReceiverCenter, EndpointCertificateError> {
    let skyfield = super::geodetic_from_ecef(SppModelRecipe::reference().frame, position_ecef_m);
    let rtklib = rtklib_geodetic_candidate(position_ecef_m)?;
    Ok(ReceiverCenter {
        position_ecef_m,
        receiver_clock_m,
        finite_inverse_candidates_m: [
            [skyfield.lat_rad, skyfield.lon_rad, skyfield.height_m],
            rtklib,
        ],
    })
}

fn oracle_prestate_center(snapshot: &IndependentLsqSnapshot) -> ReceiverCenter {
    let position_ecef_m = [
        snapshot.receiver_state[0],
        snapshot.receiver_state[1],
        snapshot.receiver_state[2],
    ];
    let skyfield = super::geodetic_from_ecef(SppModelRecipe::reference().frame, position_ecef_m);
    ReceiverCenter {
        position_ecef_m,
        receiver_clock_m: snapshot.receiver_state[3],
        finite_inverse_candidates_m: [
            [skyfield.lat_rad, skyfield.lon_rad, skyfield.height_m],
            snapshot.c_geodetic_rad_m,
        ],
    }
}

fn rtklib_geodetic_candidate(
    position_ecef_m: [f64; 3],
) -> Result<[f64; 3], EndpointCertificateError> {
    let [x_m, y_m, z_m] = position_ecef_m;
    let horizontal_squared_m2 = x_m * x_m + y_m * y_m;
    if !horizontal_squared_m2.is_finite() || horizontal_squared_m2 <= 1.0e-12 {
        return Err(EndpointCertificateError::ArithmeticOutOfRange);
    }
    let eccentricity_squared = WGS84_F * (2.0 - WGS84_F);
    let mut latitude_height_m = z_m;
    let mut previous_height_m = 0.0;
    let mut prime_vertical_radius_m = WGS84_A_M;
    for _ in 0..32 {
        if (latitude_height_m - previous_height_m).abs() < 1.0e-4 {
            break;
        }
        previous_height_m = latitude_height_m;
        let sine_latitude = latitude_height_m
            / libm::sqrt(horizontal_squared_m2 + latitude_height_m * latitude_height_m);
        prime_vertical_radius_m =
            WGS84_A_M / libm::sqrt(1.0 - eccentricity_squared * sine_latitude * sine_latitude);
        latitude_height_m = z_m + prime_vertical_radius_m * eccentricity_squared * sine_latitude;
    }
    let latitude_rad = libm::atan(latitude_height_m / libm::sqrt(horizontal_squared_m2));
    let longitude_rad = libm::atan2(y_m, x_m);
    let height_m = libm::sqrt(horizontal_squared_m2 + latitude_height_m * latitude_height_m)
        - prime_vertical_radius_m;
    let result = [latitude_rad, longitude_rad, height_m];
    result
        .iter()
        .all(|value| value.is_finite())
        .then_some(result)
        .ok_or(EndpointCertificateError::ArithmeticOutOfRange)
}

fn check_final_lsq(
    reference: &IndependentEndpoint,
    snapshot: &IndependentLsqSnapshot,
    selection: &Selection,
    centers: &BTreeMap<GnssSatelliteId, CenterIntervals>,
) -> Result<(), EndpointCertificateError> {
    for component in 0..3 {
        let replayed = snapshot.receiver_state[component] + snapshot.step[component];
        if replayed.to_bits() != reference.position_ecef_m[component].to_bits() {
            return Err(EndpointCertificateError::FinalLsqReplayMismatch);
        }
    }
    if snapshot.weighted_design_columns.len() != selection.used.len() {
        return Err(EndpointCertificateError::FinalLsqColumnCount);
    }
    let expected_columns: Vec<[Interval; 4]> = selection
        .used
        .iter()
        .map(|satellite| {
            let center = centers
                .get(satellite)
                .ok_or(EndpointCertificateError::CandidateClassification)?;
            let weighted_scale = center.weight.sqrt();
            Ok(center
                .design_row
                .map(|component| component.mul(weighted_scale)))
        })
        .collect::<Result<_, EndpointCertificateError>>()?;
    if !columns_have_perfect_interval_match(&snapshot.weighted_design_columns, &expected_columns) {
        return Err(EndpointCertificateError::FinalLsqDesignOutsideInterval);
    }

    let h_rows: Vec<_> = selection
        .used
        .iter()
        .map(|satellite| {
            centers
                .get(satellite)
                .map(|center| center.design_row)
                .ok_or(EndpointCertificateError::CandidateClassification)
        })
        .collect::<Result<_, _>>()?;
    let weights: Vec<_> = selection
        .used
        .iter()
        .map(|satellite| {
            centers
                .get(satellite)
                .map(|center| center.weight)
                .ok_or(EndpointCertificateError::CandidateClassification)
        })
        .collect::<Result<_, _>>()?;
    let covariance = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        covariance_certificate::inverse_interval(&h_rows, &weights)
    }))
    .map_err(|_| EndpointCertificateError::FinalLsqCovarianceOutsideInterval)?;
    if (0..4).any(|row| {
        (0..4).any(|column| !covariance[row][column].contains(snapshot.covariance[row][column]))
    }) {
        return Err(EndpointCertificateError::FinalLsqCovarianceOutsideInterval);
    }
    let covariance_indices = [(0, 0), (1, 1), (2, 2), (1, 0), (2, 1), (2, 0)];
    if covariance_indices
        .iter()
        .zip(snapshot.reported_position_covariance_f32)
        .any(|((row, column), value)| {
            let captured = snapshot.covariance[*row][*column];
            !covariance_certificate::f32_quantization_cell(value).contains(captured)
                || (captured as f32).to_bits() != value.to_bits()
        })
    {
        return Err(EndpointCertificateError::FinalLsqPositionCovarianceOutsideCell);
    }
    Ok(())
}

fn columns_have_perfect_interval_match(observed: &[[f64; 4]], expected: &[[Interval; 4]]) -> bool {
    fn assign(
        observed_index: usize,
        observed: &[[f64; 4]],
        expected: &[[Interval; 4]],
        matched_observed: &mut [Option<usize>],
        visited: &mut [bool],
    ) -> bool {
        for expected_index in 0..expected.len() {
            if visited[expected_index]
                || !observed[observed_index]
                    .iter()
                    .zip(expected[expected_index])
                    .all(|(value, interval)| interval.contains(*value))
            {
                continue;
            }
            visited[expected_index] = true;
            if matched_observed[expected_index].is_none()
                || assign(
                    matched_observed[expected_index].unwrap(),
                    observed,
                    expected,
                    matched_observed,
                    visited,
                )
            {
                matched_observed[expected_index] = Some(observed_index);
                return true;
            }
        }
        false
    }

    if observed.len() != expected.len() {
        return false;
    }
    let mut matched_observed = vec![None; expected.len()];
    for observed_index in 0..observed.len() {
        if !assign(
            observed_index,
            observed,
            expected,
            &mut matched_observed,
            &mut vec![false; expected.len()],
        ) {
            return false;
        }
    }
    true
}

fn validate_inputs(
    inputs: &SolveInputs,
    receive_epoch: &ExactEpochQuery,
    reference: &IndependentEndpoint,
    native_endpoint: &NativeEndpoint,
    final_lsq: &IndependentLsqSnapshot,
) -> Result<(), EndpointCertificateError> {
    if inputs.pseudorange_code != PseudorangeCode::SingleFrequency
        || inputs.troposphere_model != TroposphereModel::Rtklib
        || inputs.robust.is_some()
        || inputs.sbas_iono.is_some()
        || inputs
            .observations
            .iter()
            .any(|observation| observation.satellite_id.system != GnssSystem::Gps)
    {
        return Err(EndpointCertificateError::UnsupportedInputs);
    }
    let finite = reference
        .position_ecef_m
        .iter()
        .chain(native_endpoint.position_ecef_m.iter())
        .chain(reference.c_geodetic_rad_m.iter())
        .chain([
            &reference.receiver_clock_m,
            &native_endpoint.receiver_clock_m,
        ])
        .chain(final_lsq.receiver_state.iter())
        .chain(final_lsq.step.iter())
        .chain(final_lsq.c_geodetic_rad_m.iter())
        .all(|value| value.is_finite());
    if !finite
        || !inputs.t_rx_j2000_s.is_finite()
        || receive_epoch.j2000_seconds().to_bits() != inputs.t_rx_j2000_s.to_bits()
        || inputs
            .observations
            .iter()
            .map(|row| row.satellite_id)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != inputs.observations.len()
        || reference.oracle_states.values().any(|state| {
            !state.transmit_epoch.j2000_seconds().is_finite()
                || state.position_ecef_m.iter().any(|value| !value.is_finite())
                || !state.clock_s.is_finite()
                || !state.group_delay_s.is_finite()
                || !state.ephemeris_variance_m2.is_finite()
                || state.ephemeris_variance_m2 < 0.0
                || !state
                    .orbit_evidence
                    .transmit_epoch
                    .j2000_seconds()
                    .is_finite()
                || state
                    .orbit_evidence
                    .orbit
                    .position_m
                    .iter()
                    .any(|interval| !interval.lower().is_finite() || !interval.upper().is_finite())
                || !state.orbit_evidence.orbit.clock_s.lower().is_finite()
                || !state.orbit_evidence.orbit.clock_s.upper().is_finite()
        })
        || reference.native_state_enclosures.values().any(|evidence| {
            !evidence.transmit_epoch.j2000_seconds().is_finite()
                || evidence
                    .orbit
                    .position_m
                    .iter()
                    .any(|interval| !interval.lower().is_finite() || !interval.upper().is_finite())
                || !evidence.orbit.clock_s.lower().is_finite()
                || !evidence.orbit.clock_s.upper().is_finite()
        })
        || final_lsq
            .weighted_design_columns
            .iter()
            .flatten()
            .chain(final_lsq.covariance.iter().flatten())
            .chain(native_endpoint.position_covariance_ecef_m2.iter().flatten())
            .any(|value| !value.is_finite())
        || final_lsq
            .reported_position_covariance_f32
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(EndpointCertificateError::InvalidEndpoint);
    }
    Ok(())
}

fn candidate_states(
    source: &dyn EphemerisSource,
    inputs: &SolveInputs,
    receive_epoch: &ExactEpochQuery,
) -> Result<BTreeMap<GnssSatelliteId, CandidateState>, EndpointCertificateError> {
    let mut states = BTreeMap::new();
    for observation in &inputs.observations {
        let satellite = observation.satellite_id;
        let pseudorange_m = observation.pseudorange_m;
        if !pseudorange_m.is_finite() || pseudorange_m <= 0.0 {
            continue;
        }
        let clock_epoch = receive_epoch
            .clone()
            .checked_sub_binary_seconds(pseudorange_m / C_M_S)
            .ok_or(EndpointCertificateError::ArithmeticOutOfRange)?;
        let placement_clock = source
            .try_transmit_epoch_clock_at_epoch_query(satellite, &clock_epoch, receive_epoch)
            .map_err(|_| EndpointCertificateError::SourceRefused)?;
        let Some(placement_clock) = placement_clock else {
            continue;
        };
        let transmit_epoch = clock_epoch
            .checked_sub_binary_seconds(placement_clock.value)
            .ok_or(EndpointCertificateError::ArithmeticOutOfRange)?;
        let selected = source
            .try_position_clock_group_delay_selected_at_epoch_query(
                satellite,
                &transmit_epoch,
                receive_epoch,
            )
            .map_err(|_| EndpointCertificateError::SourceRefused)?;
        let Some(selected) = selected else {
            continue;
        };
        let (position_ecef_m, mut clock_s, group_delay) = selected.value;
        match source.clock_relativity_for_state_at_epoch_query(
            satellite,
            &transmit_epoch,
            position_ecef_m,
        ) {
            ClockRelativity::NotApplicable => {}
            ClockRelativity::Term(term_s) if term_s.is_finite() => clock_s += term_s,
            ClockRelativity::Term(_) | ClockRelativity::Unavailable => {
                return Err(EndpointCertificateError::SourceRefused)
            }
        }
        let ephemeris_variance_m2 =
            source.ephemeris_variance_at_epoch_query(satellite, &transmit_epoch, receive_epoch);
        if position_ecef_m.iter().any(|value| !value.is_finite())
            || !clock_s.is_finite()
            || !group_delay.unwrap_or(0.0).is_finite()
            || !ephemeris_variance_m2.is_finite()
            || ephemeris_variance_m2 < 0.0
        {
            return Err(EndpointCertificateError::SourceRefused);
        }
        if states
            .insert(
                satellite,
                CandidateState {
                    transmit_epoch: transmit_epoch.clone(),
                    position_ecef_m,
                    clock_s,
                    group_delay_s: group_delay.unwrap_or(0.0),
                    ephemeris_variance_m2,
                },
            )
            .is_some()
        {
            return Err(EndpointCertificateError::InvalidEndpoint);
        }
    }
    Ok(states)
}

fn centre_intervals(
    inputs: &SolveInputs,
    endpoint: &ReceiverCenter,
    selection: &Selection,
    candidates: &BTreeMap<GnssSatelliteId, CandidateState>,
) -> Result<BTreeMap<GnssSatelliteId, CenterIntervals>, EndpointCertificateError> {
    let observations: BTreeMap<_, _> = inputs
        .observations
        .iter()
        .map(|observation| (observation.satellite_id, observation.pseudorange_m))
        .collect();
    let mut output = BTreeMap::new();
    for satellite in &selection.used {
        let candidate = candidates
            .get(satellite)
            .ok_or(EndpointCertificateError::MissingCandidateState(*satellite))?;
        let pseudorange_m = *observations
            .get(satellite)
            .ok_or(EndpointCertificateError::CandidateClassification)?;
        let center_inputs = CenterInputs {
            receiver_ecef_m: endpoint.position_ecef_m,
            finite_inverse_candidates: endpoint.finite_inverse_candidates_m,
            receiver_clock_m: endpoint.receiver_clock_m,
            satellite_ecef_m: candidate.position_ecef_m,
            satellite_clock_s: candidate.clock_s,
            group_delay_s: candidate.group_delay_s,
            pseudorange_m,
            ephemeris_variance_m2: candidate.ephemeris_variance_m2,
            second_of_day_s: inputs.t_rx_second_of_day_s,
            alpha: inputs.klobuchar.alpha,
            beta: inputs.klobuchar.beta,
            apply_ionosphere: inputs.corrections.ionosphere,
            apply_troposphere: inputs.corrections.troposphere,
        };
        output.insert(
            *satellite,
            centre_certificate::evaluate(&center_inputs)
                .map_err(EndpointCertificateError::Center)?,
        );
    }
    Ok(output)
}

fn check_reference_rows(
    reference: &IndependentEndpoint,
    selection: &Selection,
) -> Result<(), EndpointCertificateError> {
    let rows: BTreeMap<_, _> = reference
        .satellite_rows
        .iter()
        .map(|row| (row.satellite_id, row))
        .collect();
    let oracle_satellites: std::collections::BTreeSet<_> =
        reference.oracle_states.keys().copied().collect();
    if rows.len() != reference.satellite_rows.len()
        || rows.len() != selection.used.len()
        || oracle_satellites != rows.keys().copied().collect()
        || reference.satellite_rows.iter().any(|row| {
            !row.residual_m.is_finite()
                || !row.total_variance_m2.is_finite()
                || row.total_variance_m2 <= 0.0
                || row.design_row.iter().any(|value| !value.is_finite())
        })
        || selection
            .used
            .iter()
            .any(|satellite| !rows.contains_key(satellite))
    {
        return Err(EndpointCertificateError::ReferenceStateSet);
    }
    Ok(())
}

fn check_native_orbit_enclosures(
    reference: &IndependentEndpoint,
    candidates: &BTreeMap<GnssSatelliteId, CandidateState>,
) -> Result<(), EndpointCertificateError> {
    if reference.native_state_enclosures.len() != candidates.len() {
        return Err(EndpointCertificateError::ReferenceStateSet);
    }
    for (satellite, state) in candidates {
        let evidence = reference.native_state_enclosures.get(satellite).ok_or(
            EndpointCertificateError::MissingNativeOrbitEnclosure(*satellite),
        )?;
        if evidence
            .transmit_epoch
            .compare_interval_query(&state.transmit_epoch, 0.0)
            != Some(Ordering::Equal)
        {
            return Err(EndpointCertificateError::NativeStateEpochMismatch(
                *satellite,
            ));
        }
        if !state_is_enclosed(state.position_ecef_m, state.clock_s, evidence.orbit) {
            return Err(EndpointCertificateError::NativeStateOutsideOrbitEnclosure(
                *satellite,
            ));
        }
    }
    for satellite in reference.oracle_states.keys() {
        if !candidates.contains_key(satellite) {
            return Err(EndpointCertificateError::ReferenceStateSet);
        }
    }
    for (satellite, state) in &reference.oracle_states {
        if state
            .transmit_epoch
            .compare_interval_query(&state.orbit_evidence.transmit_epoch, 0.0)
            != Some(Ordering::Equal)
        {
            return Err(EndpointCertificateError::NativeStateEpochMismatch(
                *satellite,
            ));
        }
        if !state_is_enclosed(
            state.position_ecef_m,
            state.clock_s,
            state.orbit_evidence.orbit,
        ) {
            return Err(EndpointCertificateError::OracleStateOutsideOrbitEnclosure(
                *satellite,
            ));
        }
    }
    Ok(())
}

fn state_is_enclosed(
    position_ecef_m: [f64; 3],
    clock_s: f64,
    enclosure: GpsStateEnclosure,
) -> bool {
    position_ecef_m
        .into_iter()
        .zip(enclosure.position_m)
        .all(|(value, interval)| interval.contains(value))
        && enclosure.clock_s.contains(clock_s)
}

fn check_selection_intervals(
    selection: &Selection,
    centers: &BTreeMap<GnssSatelliteId, CenterIntervals>,
) -> Result<(), EndpointCertificateError> {
    for (index, satellite) in selection.used.iter().enumerate() {
        let center = centers
            .get(satellite)
            .ok_or(EndpointCertificateError::CandidateClassification)?;
        if !center
            .design_row
            .iter()
            .zip(selection_row(selection, index))
            .all(|(interval, value)| interval.contains(value))
        {
            return Err(EndpointCertificateError::NativeDesignOutsideInterval(
                *satellite,
            ));
        }
        if !center.weight.contains(selection.weights[index]) {
            return Err(EndpointCertificateError::NativeWeightOutsideInterval(
                *satellite,
            ));
        }
        if !center.residual_m.contains(selection.residuals_m[index]) {
            return Err(EndpointCertificateError::NativeResidualOutsideInterval(
                *satellite,
            ));
        }
    }
    Ok(())
}

fn check_position_covariance(
    selection: &Selection,
    centers: &BTreeMap<GnssSatelliteId, CenterIntervals>,
    actual: [[f64; 3]; 3],
    endpoint: CovarianceEndpoint,
) -> Result<(), EndpointCertificateError> {
    let h_rows: Vec<_> = selection
        .used
        .iter()
        .map(|satellite| {
            centers
                .get(satellite)
                .map(|center| center.design_row)
                .ok_or(EndpointCertificateError::CandidateClassification)
        })
        .collect::<Result<_, _>>()?;
    let weights: Vec<_> = selection
        .used
        .iter()
        .map(|satellite| {
            centers
                .get(satellite)
                .map(|center| center.weight)
                .ok_or(EndpointCertificateError::CandidateClassification)
        })
        .collect::<Result<_, _>>()?;
    let covariance = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        covariance_certificate::inverse_interval(&h_rows, &weights)
    }))
    .map_err(|_| endpoint.outside_error())?;
    if (0..3).any(|row| (0..3).any(|column| !covariance[row][column].contains(actual[row][column])))
    {
        return Err(endpoint.outside_error());
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum CovarianceEndpoint {
    CPoint,
    Native,
}

impl CovarianceEndpoint {
    fn outside_error(self) -> EndpointCertificateError {
        match self {
            Self::CPoint => EndpointCertificateError::CPointCovarianceOutsideInterval,
            Self::Native => EndpointCertificateError::NativeCovarianceOutsideInterval,
        }
    }
}

fn selection_row(selection: &Selection, index: usize) -> [f64; 4] {
    let line = selection.lines_of_sight[index];
    [-line.e_x, -line.e_y, -line.e_z, 1.0]
}

fn check_oracle_intervals(
    reference: &IndependentEndpoint,
    centers: &BTreeMap<GnssSatelliteId, CenterIntervals>,
) -> Result<(), EndpointCertificateError> {
    let rows: BTreeMap<_, _> = reference
        .satellite_rows
        .iter()
        .map(|row| (row.satellite_id, row))
        .collect();
    for (satellite, row) in rows {
        let center = centers
            .get(&satellite)
            .ok_or(EndpointCertificateError::ReferenceStateSet)?;
        if !center
            .design_row
            .iter()
            .zip(row.design_row)
            .all(|(interval, value)| interval.contains(value))
        {
            return Err(EndpointCertificateError::OracleDesignOutsideInterval(
                satellite,
            ));
        }
        let oracle_weight = Interval::point(1.0).div(Interval::point(row.total_variance_m2));
        if !center.weight.contains_interval(oracle_weight) {
            return Err(EndpointCertificateError::OracleWeightOutsideInterval(
                satellite,
            ));
        }
        if !center.residual_m.contains(row.residual_m) {
            return Err(EndpointCertificateError::OracleResidualOutsideInterval(
                satellite,
            ));
        }
    }
    Ok(())
}

fn oracle_states(
    reference: &IndependentEndpoint,
) -> Result<BTreeMap<GnssSatelliteId, CandidateState>, EndpointCertificateError> {
    let mut states = BTreeMap::new();
    for (satellite, oracle_state) in &reference.oracle_states {
        let state = CandidateState {
            transmit_epoch: oracle_state.transmit_epoch.clone(),
            position_ecef_m: oracle_state.position_ecef_m,
            clock_s: oracle_state.clock_s,
            group_delay_s: oracle_state.group_delay_s,
            ephemeris_variance_m2: oracle_state.ephemeris_variance_m2,
        };
        if states.insert(*satellite, state).is_some() {
            return Err(EndpointCertificateError::ReferenceStateSet);
        }
    }
    Ok(states)
}

fn geodetic_enclosure(
    centers: &BTreeMap<GnssSatelliteId, CenterIntervals>,
) -> Result<CenterGeodeticEnclosure, EndpointCertificateError> {
    let center = centers
        .values()
        .next()
        .ok_or(EndpointCertificateError::CandidateClassification)?;
    Ok(CenterGeodeticEnclosure {
        latitude_rad: midpoint(center.ideal_latitude_rad),
        latitude_error_rad: radius(center.ideal_latitude_rad),
        longitude_rad: midpoint(center.ideal_longitude_rad),
        longitude_error_rad: radius(center.ideal_longitude_rad),
        height_m: midpoint(center.ideal_height_m),
        height_error_m: radius(center.ideal_height_m),
    })
}

fn candidate_angles(
    inputs: &SolveInputs,
    endpoint: &ReceiverCenter,
    candidates: &BTreeMap<GnssSatelliteId, CandidateState>,
    selection: &Selection,
    centers: &BTreeMap<GnssSatelliteId, CenterIntervals>,
) -> Result<Vec<CenterSatelliteAngles>, EndpointCertificateError> {
    let mut result = Vec::with_capacity(candidates.len());
    for (satellite, candidate) in candidates {
        let center = centers.get(satellite);
        let geometry = if let Some(center) = center {
            super::centre_certificate::CenterGeometry {
                ideal_latitude_rad: center.ideal_latitude_rad,
                ideal_longitude_rad: center.ideal_longitude_rad,
                ideal_height_m: center.ideal_height_m,
                ideal_azimuth_rad: center.ideal_azimuth_rad,
                ideal_elevation_rad: center.ideal_elevation_rad,
                latitude_rad: center.latitude_rad,
                longitude_rad: center.longitude_rad,
                height_m: center.height_m,
                azimuth_rad: center.azimuth_rad,
                elevation_rad: center.elevation_rad,
                design_row: center.design_row,
                finite_inverse_forward_residuals_m: center.finite_inverse_forward_residuals_m,
            }
        } else {
            let center_inputs = CenterInputs {
                receiver_ecef_m: endpoint.position_ecef_m,
                finite_inverse_candidates: endpoint.finite_inverse_candidates_m,
                receiver_clock_m: endpoint.receiver_clock_m,
                satellite_ecef_m: candidate.position_ecef_m,
                satellite_clock_s: candidate.clock_s,
                group_delay_s: candidate.group_delay_s,
                pseudorange_m: 1.0,
                ephemeris_variance_m2: candidate.ephemeris_variance_m2,
                second_of_day_s: inputs.t_rx_second_of_day_s,
                alpha: inputs.klobuchar.alpha,
                beta: inputs.klobuchar.beta,
                apply_ionosphere: false,
                apply_troposphere: false,
            };
            centre_certificate::evaluate_geometry(&center_inputs)
                .map_err(EndpointCertificateError::Center)?
        };
        let selected_index = selection.used.iter().position(|id| id == satellite);
        let (weight_error, klobuchar_error) = match (selected_index, center) {
            (Some(index), Some(center)) => (
                distance_to_interval(selection.weights[index], center.weight),
                source_klobuchar_error(inputs, endpoint, candidate, center)?,
            ),
            _ => (
                0.0,
                KlobucharCenterError {
                    raw_phi_i_semicircles: 0.0,
                    phi_m_semicircles: 0.0,
                    local_time_seconds: 0.0,
                    phase_radians: 0.0,
                },
            ),
        };
        if geometry.ideal_elevation_rad.lower() <= ELEVATION_MASK_RAD && selected_index.is_some() {
            return Err(EndpointCertificateError::CandidateClassification);
        }
        let azimuth_error_rad = radius(geometry.ideal_azimuth_rad).max(interval_separation(
            geometry.azimuth_rad,
            geometry.ideal_azimuth_rad,
        ));
        let elevation_error_rad = radius(geometry.ideal_elevation_rad).max(interval_separation(
            geometry.elevation_rad,
            geometry.ideal_elevation_rad,
        ));
        result.push(CenterSatelliteAngles {
            satellite_id: *satellite,
            azimuth_rad: midpoint(geometry.ideal_azimuth_rad),
            azimuth_error_rad,
            elevation_rad: midpoint(geometry.ideal_elevation_rad),
            elevation_error_rad,
            sin_elevation_lower: geometry.ideal_elevation_rad.sin().lower(),
            weight_error,
            klobuchar_error,
        });
    }
    Ok(result)
}

fn source_klobuchar_error(
    inputs: &SolveInputs,
    endpoint: &ReceiverCenter,
    candidate: &CandidateState,
    center: &CenterIntervals,
) -> Result<KlobucharCenterError, EndpointCertificateError> {
    klobuchar_error_for_corrections(inputs.corrections, || {
        let ideal = center
            .ideal_klobuchar
            .as_ref()
            .ok_or(EndpointCertificateError::UnsupportedInputs)?;
        let geometry = super::az_el_from_ecef(
            SppModelRecipe::reference().frame,
            endpoint.position_ecef_m,
            candidate.position_ecef_m,
        );
        let source = crate::ionex::klobuchar_l1_components(
            geometry.geodetic.lat_rad.to_degrees(),
            geometry.geodetic.lon_rad.to_degrees(),
            geometry.az_rad.to_degrees(),
            geometry.el_rad.to_degrees(),
            inputs.t_rx_second_of_day_s,
            inputs.klobuchar.alpha,
            inputs.klobuchar.beta,
        );
        if ![source.phi_i, source.phi_m, source.t, source.x]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(EndpointCertificateError::InvalidEndpoint);
        }
        Ok(KlobucharCenterError {
            raw_phi_i_semicircles: distance_to_interval(source.phi_i, ideal.raw_phi_i_semicircles),
            phi_m_semicircles: distance_to_interval(source.phi_m, ideal.phi_m_semicircles),
            local_time_seconds: distance_to_interval(source.t, ideal.local_time_seconds),
            phase_radians: distance_to_interval(source.x, ideal.phase_radians),
        })
    })
}

fn klobuchar_error_for_corrections(
    corrections: Corrections,
    calculate: impl FnOnce() -> Result<KlobucharCenterError, EndpointCertificateError>,
) -> Result<KlobucharCenterError, EndpointCertificateError> {
    if corrections.ionosphere {
        calculate()
    } else {
        Ok(KlobucharCenterError {
            raw_phi_i_semicircles: 0.0,
            phi_m_semicircles: 0.0,
            local_time_seconds: 0.0,
            phase_radians: 0.0,
        })
    }
}

fn merge_regions(
    selection: &Selection,
    centers: &BTreeMap<GnssSatelliteId, CenterIntervals>,
    atmosphere: &[atmosphere_certificate::SatelliteAtmosphereBounds],
) -> Result<Vec<SatelliteRegion>, EndpointCertificateError> {
    if selection.used.len() != atmosphere.len() {
        return Err(EndpointCertificateError::CandidateClassification);
    }
    selection
        .used
        .iter()
        .enumerate()
        .map(|(index, satellite)| {
            let center = centers
                .get(satellite)
                .ok_or(EndpointCertificateError::CandidateClassification)?;
            let native_row = selection_row(selection, index);
            Ok(SatelliteRegion {
                design_derivative: atmosphere[index].design_derivative,
                correction_gradient: atmosphere[index].correction_gradient,
                weight_max: atmosphere[index].weight_max,
                weight_gradient: atmosphere[index].weight_gradient,
                centre_design_error: vector_interval_error(native_row, center.design_row),
                centre_weight_error: distance_to_interval(selection.weights[index], center.weight),
                centre_residual_error: distance_to_interval(
                    selection.residuals_m[index],
                    center.residual_m,
                ),
            })
        })
        .collect()
}

fn selection_step(selection: &Selection) -> Result<[f64; 4], EndpointCertificateError> {
    if selection
        .used
        .iter()
        .any(|satellite| satellite.system != GnssSystem::Gps)
    {
        return Err(EndpointCertificateError::UnsupportedInputs);
    }
    let step = super::rtklib_step(
        &selection.lines_of_sight,
        &vec![3; selection.used.len()],
        4,
        &selection.weights,
        &selection.residuals_m,
    )
    .ok_or(EndpointCertificateError::RegionNotCertified)?;
    if step.len() != 4 || step.iter().any(|value| !value.is_finite()) {
        return Err(EndpointCertificateError::RegionNotCertified);
    }
    Ok([step[0], step[1], step[2], step[3]])
}

fn step_error(
    selection: &Selection,
    step: [f64; 4],
    regions: &[SatelliteRegion],
) -> Result<f64, EndpointCertificateError> {
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        regional_certificate::step_evaluation_error(selection, step, regions)
    }))
    .map_err(|_| EndpointCertificateError::RegionNotCertified)?;
    if !error.is_finite() || error < 0.0 {
        return Err(EndpointCertificateError::RegionNotCertified);
    }
    Ok(error)
}

fn state(position_ecef_m: [f64; 3], clock_m: f64) -> [f64; 4] {
    [
        position_ecef_m[0],
        position_ecef_m[1],
        position_ecef_m[2],
        clock_m,
    ]
}

fn midpoint(interval: Interval) -> f64 {
    interval.lower() + (interval.upper() - interval.lower()) * 0.5
}

fn radius(interval: Interval) -> f64 {
    distance_to_interval(midpoint(interval), interval)
}

fn distance_to_interval(value: f64, interval: Interval) -> f64 {
    let difference = Interval::point(value).sub(interval);
    difference.lower().abs().max(difference.upper().abs())
}

fn interval_separation(left: Interval, right: Interval) -> f64 {
    let difference = left.sub(right);
    difference.lower().abs().max(difference.upper().abs())
}

fn vector_interval_error(values: [f64; 4], intervals: [Interval; 4]) -> f64 {
    let squares = values.into_iter().zip(intervals).map(|(value, interval)| {
        let error = distance_to_interval(value, interval);
        Interval::point(error).square()
    });
    squares
        .fold(Interval::point(0.0), |sum, square| sum.add(square))
        .sqrt()
        .upper()
}
