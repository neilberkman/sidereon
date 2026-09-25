use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

use serde_json::Value;

use crate::astro::time::{ExactEpoch, ExactEpochQuery};
use crate::id::GnssSatelliteId;
use crate::rinex_nav::{BroadcastRecord, BroadcastStore, NavMessage};

use super::broadcast_certificate::{self, GpsStateEnclosure};
use super::endpoint_certificate::{
    self, EndpointCertificate, EndpointCertificateError, IndependentEndpoint,
    IndependentLsqSnapshot, IndependentSatelliteRow, IndependentStateEnclosure, NativeEndpoint,
    OracleSatelliteState,
};
use super::{ClockRelativity, EphemerisSource, GnssSystem, ReceiverSolution, SolveInputs, C_M_S};

const RTKLIB_POSITION_STATES: usize = 9;
const RECEIVER_STATES: usize = 4;

#[cfg(test)]
mod tests {
    use super::{lsq_snapshot, OracleCertificateError, RECEIVER_STATES, RTKLIB_POSITION_STATES};

    #[test]
    fn gps_projection_validates_then_removes_only_decoupled_bias_columns() {
        let identity: Vec<Vec<f64>> = (0..RTKLIB_POSITION_STATES)
            .map(|row| {
                (0..RTKLIB_POSITION_STATES)
                    .map(|column| if row == column { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let mut case = serde_json::json!({
            "lsq_weighted_design_columns": identity,
            "lsq_covariance": identity,
            "satellite_states": [null, null, null, null],
            "qr_m2": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        });
        let snapshot = lsq_snapshot(
            &case,
            [0.0; RTKLIB_POSITION_STATES],
            [0.0; RTKLIB_POSITION_STATES],
            [0.0; 3],
        )
        .unwrap_or_else(|error| panic!("decoupled GPS projection: {error}"));
        assert_eq!(snapshot.weighted_design_columns.len(), RECEIVER_STATES);
        assert_eq!(snapshot.weighted_design_columns[0], [1.0, 0.0, 0.0, 0.0]);
        case["lsq_weighted_design_columns"][RECEIVER_STATES][0] = 1.0.into();
        assert!(matches!(
            lsq_snapshot(
                &case,
                [0.0; RTKLIB_POSITION_STATES],
                [0.0; RTKLIB_POSITION_STATES],
                [0.0; 3],
            ),
            Err(OracleCertificateError::InvalidField(
                "lsq_weighted_design_columns block coupling"
            ))
        ));
    }
}

pub(super) struct OracleDiagnostics {
    pub endpoint: EndpointCertificate,
    pub native_state_count: usize,
    pub oracle_state_count: usize,
}

#[derive(Debug)]
pub(super) enum OracleCertificateError {
    MissingField(&'static str),
    InvalidField(&'static str),
    UnsupportedCase,
    MissingSelectedRecord(GnssSatelliteId),
    NotGpsLnav(GnssSatelliteId),
    NativeStateRefused(GnssSatelliteId),
    NativeStateOutsideOrbit(GnssSatelliteId),
    OracleStateOutsideOrbit(GnssSatelliteId),
    ExactTimeUnavailable,
    RecordTimeUnavailable(GnssSatelliteId),
    VarianceMismatch(GnssSatelliteId),
    PassiveStepMismatch,
    ReferenceStateSet,
    Endpoint(EndpointCertificateError),
}

impl core::fmt::Display for OracleCertificateError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingField(field) => write!(formatter, "missing field: {field}"),
            Self::InvalidField(field) => write!(formatter, "invalid field: {field}"),
            Self::UnsupportedCase => formatter.write_str("unsupported case"),
            Self::MissingSelectedRecord(satellite) => {
                write!(formatter, "missing selected record: {satellite}")
            }
            Self::NotGpsLnav(satellite) => write!(formatter, "not gps lnav: {satellite}"),
            Self::NativeStateRefused(satellite) => {
                write!(formatter, "native state refused: {satellite}")
            }
            Self::NativeStateOutsideOrbit(satellite) => {
                write!(formatter, "native state outside orbit: {satellite}")
            }
            Self::OracleStateOutsideOrbit(satellite) => {
                write!(formatter, "oracle state outside orbit: {satellite}")
            }
            Self::ExactTimeUnavailable => formatter.write_str("exact time unavailable"),
            Self::RecordTimeUnavailable(satellite) => {
                write!(formatter, "record time unavailable: {satellite}")
            }
            Self::VarianceMismatch(satellite) => {
                write!(formatter, "variance mismatch: {satellite}")
            }
            Self::PassiveStepMismatch => formatter.write_str("passive step mismatch"),
            Self::ReferenceStateSet => formatter.write_str("reference state set"),
            Self::Endpoint(cause) => write!(formatter, "endpoint: {cause}"),
        }
    }
}

pub(super) fn verify_case(
    store: &BroadcastStore,
    inputs: &SolveInputs,
    case: &Value,
    solution: &ReceiverSolution,
) -> Result<OracleDiagnostics, OracleCertificateError> {
    if inputs
        .observations
        .iter()
        .any(|observation| observation.satellite_id.system != GnssSystem::Gps)
    {
        return Err(OracleCertificateError::UnsupportedCase);
    }
    let receive_epoch = ExactEpoch::from_binary_j2000_seconds(inputs.t_rx_j2000_s)
        .ok_or(OracleCertificateError::ExactTimeUnavailable)?;
    let prestate = array::<RTKLIB_POSITION_STATES>(field(case, "lsq_receiver_state")?)?;
    let step = array::<RTKLIB_POSITION_STATES>(field(case, "lsq_step")?)?;
    let endpoint_position = array::<3>(field(case, "position_m")?)?;
    let endpoint_clock_m = number(field(case, "clock_m")?)?;
    verify_passive_step(prestate, step, endpoint_position, endpoint_clock_m)?;

    let endpoint_geodetic = array::<3>(field(case, "geodetic_rad_m")?)?;
    let prestate_geodetic = array::<3>(field(case, "lsq_geodetic_rad_m")?)?;
    let satellite_states = field(case, "satellite_states")?
        .as_array()
        .ok_or(OracleCertificateError::InvalidField("satellite_states"))?;
    let expected_used: BTreeSet<_> = field(case, "used")?
        .as_array()
        .ok_or(OracleCertificateError::InvalidField("used"))?
        .iter()
        .map(satellite_id)
        .collect::<Result<_, _>>()?;
    let solution_used: BTreeSet<_> = solution.used_sats.iter().copied().collect();
    if expected_used.len() != solution.used_sats.len() || expected_used != solution_used {
        return Err(OracleCertificateError::ReferenceStateSet);
    }
    let mut oracle_states = BTreeMap::new();
    let mut satellite_rows = Vec::with_capacity(satellite_states.len());
    for state in satellite_states {
        let satellite = satellite_id(field(state, "sat")?)?;
        if satellite.system != GnssSystem::Gps || oracle_states.contains_key(&satellite) {
            return Err(OracleCertificateError::InvalidField("satellite_states.sat"));
        }
        let record = selected_lnav_record(store, satellite, &receive_epoch)?;
        let transmit_epoch = broadcast_certificate::query_from_rtklib_time(
            integer(field(state, "transmit_j2000_whole_s")?)?,
            string(field(state, "transmit_fraction_bits")?)?,
        )
        .ok_or(OracleCertificateError::ExactTimeUnavailable)?;
        let orbit = enclose_record(record, &transmit_epoch)
            .ok_or(OracleCertificateError::RecordTimeUnavailable(satellite))?;
        let position_ecef_m = array::<3>(field(state, "position_m")?)?;
        let clock_s = number(field(state, "clock_s")?)?;
        if !state_is_enclosed(position_ecef_m, clock_s, orbit) {
            return Err(OracleCertificateError::OracleStateOutsideOrbit(satellite));
        }
        let ephemeris_variance_m2 = number(field(state, "variance_m2")?)?;
        let selected_variance =
            store.ephemeris_variance_at_epoch_query(satellite, &transmit_epoch, &receive_epoch);
        if ephemeris_variance_m2.to_bits() != selected_variance.to_bits() {
            return Err(OracleCertificateError::VarianceMismatch(satellite));
        }
        oracle_states.insert(
            satellite,
            OracleSatelliteState {
                transmit_epoch: transmit_epoch.clone(),
                position_ecef_m,
                clock_s,
                group_delay_s: record.broadcast_clock_group_delay_s(),
                ephemeris_variance_m2,
                orbit_evidence: IndependentStateEnclosure {
                    transmit_epoch,
                    orbit,
                },
            },
        );
        satellite_rows.push(IndependentSatelliteRow {
            satellite_id: satellite,
            design_row: array::<4>(field(state, "design_row")?)?,
            residual_m: number(field(state, "residual_m")?)?,
            total_variance_m2: number(field(state, "reference_variance_m2")?)?,
        });
    }
    if oracle_states.keys().copied().collect::<BTreeSet<_>>() != expected_used {
        return Err(OracleCertificateError::ReferenceStateSet);
    }

    let native_state_enclosures = native_orbit_enclosures(store, inputs, &receive_epoch)?;
    let final_lsq = lsq_snapshot(case, prestate, step, prestate_geodetic)?;
    let reference = IndependentEndpoint {
        position_ecef_m: endpoint_position,
        receiver_clock_m: endpoint_clock_m,
        c_geodetic_rad_m: endpoint_geodetic,
        satellite_rows,
        oracle_states,
        native_state_enclosures,
    };
    let solution_clock_m = solution.rx_clock_s * C_M_S;
    let native_endpoint = NativeEndpoint {
        position_ecef_m: solution.position.as_array(),
        receiver_clock_m: solution_clock_m,
        position_covariance_ecef_m2: solution.position_covariance.ecef_m2,
    };
    let endpoint = endpoint_certificate::certify(
        store,
        inputs,
        &receive_epoch,
        &reference,
        &native_endpoint,
        &final_lsq,
    )
    .map_err(OracleCertificateError::Endpoint)?;
    Ok(OracleDiagnostics {
        endpoint,
        native_state_count: reference.native_state_enclosures.len(),
        oracle_state_count: reference.oracle_states.len(),
    })
}

fn verify_passive_step(
    prestate: [f64; RTKLIB_POSITION_STATES],
    step: [f64; RTKLIB_POSITION_STATES],
    endpoint_position: [f64; 3],
    endpoint_clock_m: f64,
) -> Result<(), OracleCertificateError> {
    if prestate[RECEIVER_STATES..]
        .iter()
        .any(|value| *value != 0.0)
        || step[RECEIVER_STATES..].iter().any(|value| *value != 0.0)
    {
        return Err(OracleCertificateError::PassiveStepMismatch);
    }
    let mut replayed = [0.0; RTKLIB_POSITION_STATES];
    for state_index in 0..RTKLIB_POSITION_STATES {
        replayed[state_index] = prestate[state_index] + step[state_index];
    }
    if (0..3).any(|axis| replayed[axis].to_bits() != endpoint_position[axis].to_bits())
        || (replayed[3] / C_M_S * C_M_S).to_bits() != endpoint_clock_m.to_bits()
    {
        return Err(OracleCertificateError::PassiveStepMismatch);
    }
    Ok(())
}

fn lsq_snapshot(
    case: &Value,
    prestate: [f64; RTKLIB_POSITION_STATES],
    step: [f64; RTKLIB_POSITION_STATES],
    c_geodetic_rad_m: [f64; 3],
) -> Result<IndependentLsqSnapshot, OracleCertificateError> {
    let columns = field(case, "lsq_weighted_design_columns")?
        .as_array()
        .ok_or(OracleCertificateError::InvalidField(
            "lsq_weighted_design_columns",
        ))?;
    let satellite_count = field(case, "satellite_states")?
        .as_array()
        .ok_or(OracleCertificateError::InvalidField("satellite_states"))?
        .len();
    if columns.len() != satellite_count + RTKLIB_POSITION_STATES - RECEIVER_STATES {
        return Err(OracleCertificateError::InvalidField(
            "lsq_weighted_design_columns length",
        ));
    }
    let mut weighted_design_columns = columns
        .iter()
        .enumerate()
        .map(|(row_index, column)| {
            let values = array::<RTKLIB_POSITION_STATES>(column)?;
            let satellite_row = row_index < satellite_count;
            if (satellite_row && values[RECEIVER_STATES..].iter().any(|value| *value != 0.0))
                || (!satellite_row && values[..RECEIVER_STATES].iter().any(|value| *value != 0.0))
            {
                return Err(OracleCertificateError::InvalidField(
                    "lsq_weighted_design_columns block coupling",
                ));
            }
            Ok([values[0], values[1], values[2], values[3]])
        })
        .collect::<Result<Vec<_>, OracleCertificateError>>()?;
    weighted_design_columns.truncate(satellite_count);
    let matrix = field(case, "lsq_covariance")?
        .as_array()
        .ok_or(OracleCertificateError::InvalidField("lsq_covariance"))?;
    if matrix.len() != RTKLIB_POSITION_STATES {
        return Err(OracleCertificateError::InvalidField("lsq_covariance"));
    }
    let mut full_covariance = [[0.0; RTKLIB_POSITION_STATES]; RTKLIB_POSITION_STATES];
    for row in 0..RTKLIB_POSITION_STATES {
        let values = matrix[row]
            .as_array()
            .ok_or(OracleCertificateError::InvalidField("lsq_covariance"))?;
        if values.len() != RTKLIB_POSITION_STATES {
            return Err(OracleCertificateError::InvalidField("lsq_covariance"));
        }
        for column in 0..RTKLIB_POSITION_STATES {
            full_covariance[row][column] = number(&values[column])?;
            if (row < RECEIVER_STATES) != (column < RECEIVER_STATES)
                && full_covariance[row][column] != 0.0
            {
                return Err(OracleCertificateError::InvalidField(
                    "lsq_covariance block coupling",
                ));
            }
        }
    }
    let mut covariance = [[0.0; RECEIVER_STATES]; RECEIVER_STATES];
    for row in 0..RECEIVER_STATES {
        covariance[row].copy_from_slice(&full_covariance[row][..RECEIVER_STATES]);
    }
    let qr = array::<6>(field(case, "qr_m2")?)?;
    let reported_position_covariance_f32 = qr.map(|value| value as f32);
    if reported_position_covariance_f32
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(OracleCertificateError::InvalidField("qr_m2"));
    }
    Ok(IndependentLsqSnapshot {
        receiver_state: [prestate[0], prestate[1], prestate[2], prestate[3]],
        step: [step[0], step[1], step[2], step[3]],
        c_geodetic_rad_m,
        weighted_design_columns,
        covariance,
        reported_position_covariance_f32,
    })
}

fn native_orbit_enclosures(
    store: &BroadcastStore,
    inputs: &SolveInputs,
    receive_epoch: &ExactEpochQuery,
) -> Result<BTreeMap<GnssSatelliteId, IndependentStateEnclosure>, OracleCertificateError> {
    let mut enclosures = BTreeMap::new();
    for observation in &inputs.observations {
        let satellite = observation.satellite_id;
        let pseudorange_m = observation.pseudorange_m;
        if !pseudorange_m.is_finite() || pseudorange_m <= 0.0 {
            continue;
        }
        let clock_epoch = receive_epoch
            .clone()
            .checked_sub_binary_seconds(pseudorange_m / C_M_S)
            .ok_or(OracleCertificateError::ExactTimeUnavailable)?;
        let placement_clock = store
            .try_transmit_epoch_clock_at_epoch_query(satellite, &clock_epoch, receive_epoch)
            .map_err(|_| OracleCertificateError::NativeStateRefused(satellite))?;
        let Some(placement_clock) = placement_clock else {
            continue;
        };
        let transmit_epoch = clock_epoch
            .checked_sub_binary_seconds(placement_clock.value)
            .ok_or(OracleCertificateError::ExactTimeUnavailable)?;
        let record = require_lnav_record(
            store.select_record_at_epoch_query(satellite, receive_epoch),
            satellite,
        )?;
        let selected_state = store
            .try_position_clock_group_delay_selected_at_epoch_query(
                satellite,
                &transmit_epoch,
                receive_epoch,
            )
            .map_err(|_| OracleCertificateError::NativeStateRefused(satellite))?;
        let Some(selected_state) = selected_state else {
            continue;
        };
        let (position_ecef_m, mut clock_s, group_delay_s) = selected_state.value;
        match store.clock_relativity_for_state_at_epoch_query(
            satellite,
            &transmit_epoch,
            position_ecef_m,
        ) {
            ClockRelativity::NotApplicable => {}
            ClockRelativity::Term(term_s) if term_s.is_finite() => clock_s += term_s,
            ClockRelativity::Term(_) | ClockRelativity::Unavailable => {
                return Err(OracleCertificateError::NativeStateRefused(satellite));
            }
        }
        let record_group_delay = record.broadcast_clock_group_delay_s();
        if group_delay_s.unwrap_or(0.0).to_bits() != record_group_delay.to_bits() {
            return Err(OracleCertificateError::NativeStateRefused(satellite));
        }
        let orbit = enclose_record(record, &transmit_epoch)
            .ok_or(OracleCertificateError::RecordTimeUnavailable(satellite))?;
        if !state_is_enclosed(position_ecef_m, clock_s, orbit) {
            return Err(OracleCertificateError::NativeStateOutsideOrbit(satellite));
        }
        if enclosures
            .insert(
                satellite,
                IndependentStateEnclosure {
                    transmit_epoch: transmit_epoch.clone(),
                    orbit,
                },
            )
            .is_some()
        {
            return Err(OracleCertificateError::InvalidField(
                "duplicate native satellite state",
            ));
        }
    }
    Ok(enclosures)
}

fn selected_lnav_record<'a>(
    store: &'a BroadcastStore,
    satellite: GnssSatelliteId,
    selection_epoch: &ExactEpochQuery,
) -> Result<&'a BroadcastRecord, OracleCertificateError> {
    let selection_time = selection_epoch.j2000_seconds();
    let roundtrip = ExactEpoch::from_binary_j2000_seconds(selection_time)
        .ok_or(OracleCertificateError::ExactTimeUnavailable)?;
    if selection_epoch.compare_interval_query(&roundtrip, 0.0) != Some(std::cmp::Ordering::Equal) {
        return Err(OracleCertificateError::ExactTimeUnavailable);
    }
    require_lnav_record(store.select_record_at(satellite, selection_time), satellite)
}

fn require_lnav_record(
    record: Option<&BroadcastRecord>,
    satellite: GnssSatelliteId,
) -> Result<&BroadcastRecord, OracleCertificateError> {
    let record = record.ok_or(OracleCertificateError::MissingSelectedRecord(satellite))?;
    if record.message != NavMessage::GpsLnav
        || record.toe.system != crate::astro::time::TimeScale::Gpst
    {
        return Err(OracleCertificateError::NotGpsLnav(satellite));
    }
    Ok(record)
}

fn enclose_record(record: &BroadcastRecord, epoch: &ExactEpochQuery) -> Option<GpsStateEnclosure> {
    if record.satellite_id.system != GnssSystem::Gps
        || record.message != NavMessage::GpsLnav
        || record.toe.system != crate::astro::time::TimeScale::Gpst
        || record.toc.system != crate::astro::time::TimeScale::Gpst
    {
        return None;
    }
    let (tk, toc_delta) = broadcast_certificate::gps_time_deltas_at_query(
        record.toe.week,
        record.toe.tow_s,
        record.toc.week,
        record.toc.tow_s,
        epoch,
    )?;
    broadcast_certificate::enclose_gps_lnav_state(&record.elements, &record.clock, tk, toc_delta)
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

fn field<'a>(value: &'a Value, name: &'static str) -> Result<&'a Value, OracleCertificateError> {
    value
        .get(name)
        .ok_or(OracleCertificateError::MissingField(name))
}

fn number(value: &Value) -> Result<f64, OracleCertificateError> {
    value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or(OracleCertificateError::InvalidField("number"))
}

fn integer(value: &Value) -> Result<i64, OracleCertificateError> {
    value
        .as_i64()
        .ok_or(OracleCertificateError::InvalidField("integer"))
}

fn string(value: &Value) -> Result<&str, OracleCertificateError> {
    value
        .as_str()
        .ok_or(OracleCertificateError::InvalidField("string"))
}

fn array<const SIZE: usize>(value: &Value) -> Result<[f64; SIZE], OracleCertificateError> {
    let values = value
        .as_array()
        .ok_or(OracleCertificateError::InvalidField("array"))?;
    if values.len() != SIZE {
        return Err(OracleCertificateError::InvalidField("array length"));
    }
    let mut parsed = [0.0; SIZE];
    for (index, item) in values.iter().enumerate() {
        parsed[index] = number(item)?;
    }
    Ok(parsed)
}

fn satellite_id(value: &Value) -> Result<GnssSatelliteId, OracleCertificateError> {
    GnssSatelliteId::from_str(string(value)?)
        .map_err(|_| OracleCertificateError::InvalidField("satellite id"))
}
