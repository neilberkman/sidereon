#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::antex::{Antenna, Antex, PcvGrid};
use sidereon_core::astro::time::model::{Instant, JulianDateSplit, TimeScale};
use sidereon_core::astro::time::split_julian_date;
use sidereon_core::carrier_phase::CycleSlipOptions;
use sidereon_core::constants::{C_M_S, F_L1_HZ, F_L2_HZ};
use sidereon_core::ephemeris::Sp3;
use sidereon_core::observables::j2000_seconds_from_split;
use sidereon_core::rinex::observations::{
    band_frequency_hz, carrier_phase_rows, observation_values, ObsEpoch, ObsEpochTime,
    ObservationFilter, ObservationValueRow, RinexObs,
};
use sidereon_core::rtk::{
    apply_elevation_mask, baseline_reference_satellites, estimate_wide_lane_ambiguities,
    hatch_smooth_baseline_code_epochs, prepare_cycle_slip_baseline_epochs,
    prepare_dual_cycle_slip_baseline_epochs, prepare_ionosphere_free_baseline_epochs,
    BaselineReferenceEpoch, BaselineReferenceSelection, CodeSmoothingEpoch,
    CodeSmoothingObservation, CycleSlipPolicy, DualCycleSlipEpoch, DualCycleSlipObservation,
    DualEpoch, DualIonosphereFreeSetupEpoch, DualObservation, DualSatelliteObservation,
    ElevationMaskEpoch, IonosphereFreeBaselineResult, Observation, WideLaneOptions,
};
use sidereon_core::rtk_filter::{
    solve_fixed_baseline, solve_float_baseline, update_epoch, AmbiguityScale, AmbiguitySet,
    DynamicsModel, Epoch, EpochUpdate, FilterState, FixedSolveOpts, FloatPrior, FloatSolveOpts,
    MeasModel, ReceiverAntennaCalibration, ReceiverAntennaCorrections, SatMeas, SearchOpts,
    StochasticModel, UpdateOpts,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

const WTZR_MARKER_M: [f64; 3] = [4075580.3111, 931854.0543, 4801568.2808];
const WTZZ_MARKER_M: [f64; 3] = [4075579.1913, 931853.3696, 4801569.1897];

const MULTIGNSS_L1_CODES: &[(GnssSystem, &[(&str, &str)])] = &[
    (GnssSystem::Gps, &[("C1C", "L1C")]),
    (GnssSystem::Glonass, &[("C1C", "L1C")]),
    (GnssSystem::Galileo, &[("C1C", "L1C"), ("C1X", "L1X")]),
    (GnssSystem::BeiDou, &[("C2I", "L2I")]),
];

#[derive(Clone)]
struct RawEpoch {
    epoch: ObsEpochTime,
    satellite_positions_m: BTreeMap<String, [f64; 3]>,
    base_satellite_positions_m: BTreeMap<String, [f64; 3]>,
    rover_satellite_positions_m: BTreeMap<String, [f64; 3]>,
    base_observations: BTreeMap<String, ArcObservation>,
    rover_observations: BTreeMap<String, ArcObservation>,
}

#[derive(Clone)]
struct ArcObservation {
    ambiguity_id: String,
    code_m: f64,
    phase_m: f64,
    lli: Option<i64>,
}

#[derive(Clone)]
struct RawDualEpoch {
    epoch: ObsEpochTime,
    satellite_positions_m: BTreeMap<String, [f64; 3]>,
    base_satellite_positions_m: BTreeMap<String, [f64; 3]>,
    rover_satellite_positions_m: BTreeMap<String, [f64; 3]>,
    base_observations: BTreeMap<String, DualArcObservation>,
    rover_observations: BTreeMap<String, DualArcObservation>,
}

#[derive(Clone)]
struct DualArcObservation {
    ambiguity_id: String,
    p1_m: f64,
    p2_m: f64,
    phi1_cycles: f64,
    phi2_cycles: f64,
    f1_hz: f64,
    f2_hz: f64,
    lli1: Option<i64>,
    lli2: Option<i64>,
}

fn fixture_path(parts: &[&str]) -> PathBuf {
    parts.iter().fold(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
        |path, part| path.join(part),
    )
}

fn load_text(parts: &[&str]) -> String {
    let path = fixture_path(parts);
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read fixture {path:?}: {err}"))
}

fn load_sp3(parts: &[&str]) -> Sp3 {
    let path = fixture_path(parts);
    let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("read fixture {path:?}: {err}"));
    Sp3::parse(&bytes).unwrap_or_else(|err| panic!("parse SP3 {path:?}: {err}"))
}

fn load_obs(parts: &[&str]) -> RinexObs {
    RinexObs::parse(&load_text(parts))
        .unwrap_or_else(|err| panic!("parse RINEX obs {parts:?}: {err}"))
}

fn load_antex(parts: &[&str]) -> Antex {
    Antex::parse(&load_text(parts)).unwrap_or_else(|err| panic!("parse ANTEX {parts:?}: {err}"))
}

fn load_oracle(name: &str) -> Value {
    let text = load_text(&["rtk", name]);
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("parse oracle {name}: {err}"))
}

fn satellite_id(token: &str) -> Option<GnssSatelliteId> {
    let mut chars = token.chars();
    let system = GnssSystem::from_letter(chars.next()?)?;
    let prn = chars.as_str().parse::<u8>().ok()?;
    Some(GnssSatelliteId::new(system, prn).expect("valid satellite id"))
}

fn civil_to_julian_split(epoch: ObsEpochTime) -> JulianDateSplit {
    let (jd_whole, fraction) = split_julian_date(
        epoch.year,
        i32::from(epoch.month),
        i32::from(epoch.day),
        i32::from(epoch.hour),
        i32::from(epoch.minute),
        epoch.second,
    );
    JulianDateSplit::new(jd_whole, fraction).expect("valid split Julian date")
}

fn instant(epoch: ObsEpochTime) -> Instant {
    Instant::from_julian_date(TimeScale::Gpst, civil_to_julian_split(epoch))
}

fn j2000_seconds(epoch: ObsEpochTime) -> f64 {
    let split = civil_to_julian_split(epoch);
    j2000_seconds_from_split(split.jd_whole, split.fraction).expect("valid split Julian date")
}

fn epoch_key(epoch: ObsEpochTime) -> (i32, u8, u8, u8, u8, u64) {
    (
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second.to_bits(),
    )
}

fn position_at(sp3: &Sp3, sat: &str, epoch: ObsEpochTime) -> Option<[f64; 3]> {
    let state = sp3.position(satellite_id(sat)?, instant(epoch)).ok()?;
    Some(state.position.as_array())
}

fn transmit_position_at(
    sp3: &Sp3,
    sat: &str,
    receive_epoch: ObsEpochTime,
    code_m: f64,
    c_m_s: f64,
) -> Option<[f64; 3]> {
    let transmit_offset_us = (code_m / c_m_s * 1_000_000.0).round();
    let t_tx = j2000_seconds(receive_epoch) - transmit_offset_us / 1_000_000.0;
    let state = sp3
        .position_at_j2000_seconds(satellite_id(sat)?, t_tx)
        .ok()?;
    Some(state.position.as_array())
}

fn l1_filter() -> ObservationFilter {
    ObservationFilter::from_entries([(GnssSystem::Gps, vec!["C1C".to_string(), "L1C".to_string()])])
}

fn l1_l2_filter() -> ObservationFilter {
    ObservationFilter::from_entries([(
        GnssSystem::Gps,
        vec![
            "C1C".to_string(),
            "C2W".to_string(),
            "L1C".to_string(),
            "L2W".to_string(),
        ],
    )])
}

fn multignss_l1_filter(systems: &[GnssSystem]) -> ObservationFilter {
    ObservationFilter::from_entries(
        MULTIGNSS_L1_CODES
            .iter()
            .filter(|(system, _)| systems.contains(system))
            .map(|(system, pairs)| {
                let codes = pairs
                    .iter()
                    .flat_map(|(code, phase)| [(*code).to_string(), (*phase).to_string()])
                    .collect::<Vec<_>>();
                (*system, codes)
            }),
    )
}

fn gps_l1_constants(obs: &RinexObs) -> (f64, f64) {
    let filter = l1_filter();
    obs.epochs()
        .iter()
        .find_map(|epoch| {
            carrier_phase_rows(obs, epoch, &filter)
                .expect("valid carrier-phase rows")
                .into_iter()
                .flat_map(|(_, rows)| rows)
                .find(|row| row.code == "L1C")
                .and_then(|row| Some((row.frequency_hz?, row.wavelength_m?)))
        })
        .expect("GPS L1 carrier constants from RINEX fixture")
}

fn l1_observations(obs: &RinexObs, epoch: &ObsEpoch) -> BTreeMap<String, ArcObservation> {
    let filter = l1_filter();
    let (_frequency_hz, wavelength_m) = gps_l1_constants(obs);

    observation_values(obs, epoch, &filter)
        .expect("valid observation values")
        .into_iter()
        .filter_map(|(sat, rows)| {
            let mut code_m = None;
            let mut phase_cycles = None;
            let mut lli = None;
            for row in rows {
                match row.code.as_str() {
                    "C1C" => code_m = row.value,
                    "L1C" => {
                        phase_cycles = row.value;
                        lli = row.lli.map(i64::from);
                    }
                    _ => {}
                }
            }
            Some((
                sat.to_string(),
                ArcObservation {
                    ambiguity_id: sat.to_string(),
                    code_m: code_m?,
                    phase_m: phase_cycles? * wavelength_m,
                    lli,
                },
            ))
        })
        .collect()
}

fn multignss_l1_observations(
    obs: &RinexObs,
    epoch: &ObsEpoch,
    systems: &[GnssSystem],
) -> BTreeMap<String, ArcObservation> {
    let filter = multignss_l1_filter(systems);
    observation_values(obs, epoch, &filter)
        .expect("valid observation values")
        .into_iter()
        .filter_map(|(sat, rows)| {
            let pairs = MULTIGNSS_L1_CODES
                .iter()
                .find(|(system, _)| *system == sat.system)?
                .1;
            let mut values_by_code = BTreeMap::new();
            for row in rows {
                values_by_code.insert(row.code.clone(), row);
            }
            let (phase_code, code_m, phase_cycles, lli) =
                first_complete_code_phase_pair(&values_by_code, pairs)?;
            let band = phase_code.chars().nth(1)?;
            let channel = (sat.system == GnssSystem::Glonass)
                .then(|| obs.header().glonass_slots.get(&sat.prn).copied())
                .flatten();
            let frequency_hz = band_frequency_hz(sat.system, band, channel)?;
            let wavelength_m = C_M_S / frequency_hz;
            let sat = sat.to_string();
            Some((
                sat.clone(),
                ArcObservation {
                    ambiguity_id: sat,
                    code_m,
                    phase_m: phase_cycles * wavelength_m,
                    lli,
                },
            ))
        })
        .collect()
}

fn first_complete_code_phase_pair(
    values_by_code: &BTreeMap<String, ObservationValueRow>,
    pairs: &[(&str, &str)],
) -> Option<(String, f64, f64, Option<i64>)> {
    for (code, phase) in pairs {
        let Some(code_m) = values_by_code.get(*code).and_then(|row| row.value) else {
            continue;
        };
        let Some(phase_row) = values_by_code.get(*phase) else {
            continue;
        };
        let Some(phase_cycles) = phase_row.value else {
            continue;
        };
        return Some((
            (*phase).to_string(),
            code_m,
            phase_cycles,
            phase_row.lli.map(i64::from),
        ));
    }
    None
}

fn l1_l2_observations(obs: &RinexObs, epoch: &ObsEpoch) -> BTreeMap<String, DualArcObservation> {
    let filter = l1_l2_filter();
    observation_values(obs, epoch, &filter)
        .expect("valid observation values")
        .into_iter()
        .filter_map(|(sat, rows)| {
            let mut c1 = None;
            let mut c2 = None;
            let mut l1 = None;
            let mut l2 = None;
            let mut lli1 = None;
            let mut lli2 = None;
            for row in rows {
                match row.code.as_str() {
                    "C1C" => c1 = row.value,
                    "C2W" => c2 = row.value,
                    "L1C" => {
                        l1 = row.value;
                        lli1 = row.lli.map(i64::from);
                    }
                    "L2W" => {
                        l2 = row.value;
                        lli2 = row.lli.map(i64::from);
                    }
                    _ => {}
                }
            }
            Some((
                sat.to_string(),
                DualArcObservation {
                    ambiguity_id: sat.to_string(),
                    p1_m: c1?,
                    p2_m: c2?,
                    phi1_cycles: l1?,
                    phi2_cycles: l2?,
                    f1_hz: F_L1_HZ,
                    f2_hz: F_L2_HZ,
                    lli1,
                    lli2,
                },
            ))
        })
        .collect()
}

fn real_gps_l1_epochs(
    sp3: &Sp3,
    base_obs: &RinexObs,
    rover_obs: &RinexObs,
    count: usize,
) -> Vec<RawEpoch> {
    let (frequency_hz, wavelength_m) = gps_l1_constants(base_obs);
    let c_m_s = frequency_hz * wavelength_m;
    let rover_by_epoch: BTreeMap<_, _> = rover_obs
        .epochs()
        .iter()
        .map(|epoch| (epoch_key(epoch.epoch), epoch))
        .collect();
    let mut out = Vec::new();

    for base_epoch in base_obs.epochs().iter().take(count) {
        let Some(rover_epoch) = rover_by_epoch.get(&epoch_key(base_epoch.epoch)).copied() else {
            continue;
        };
        let base_values = l1_observations(base_obs, base_epoch);
        let rover_values = l1_observations(rover_obs, rover_epoch);
        let base_sats = base_values.keys().cloned().collect::<BTreeSet<_>>();
        let rover_sats = rover_values.keys().cloned().collect::<BTreeSet<_>>();
        let common = base_sats
            .intersection(&rover_sats)
            .cloned()
            .collect::<Vec<_>>();

        let mut satellite_positions_m = BTreeMap::new();
        let mut base_satellite_positions_m = BTreeMap::new();
        let mut rover_satellite_positions_m = BTreeMap::new();
        let mut usable = Vec::new();

        for sat in common {
            let Some(position) = position_at(sp3, &sat, base_epoch.epoch) else {
                continue;
            };
            let Some(base_tx) =
                transmit_position_at(sp3, &sat, base_epoch.epoch, base_values[&sat].code_m, c_m_s)
            else {
                continue;
            };
            let Some(rover_tx) = transmit_position_at(
                sp3,
                &sat,
                base_epoch.epoch,
                rover_values[&sat].code_m,
                c_m_s,
            ) else {
                continue;
            };
            satellite_positions_m.insert(sat.clone(), position);
            base_satellite_positions_m.insert(sat.clone(), base_tx);
            rover_satellite_positions_m.insert(sat.clone(), rover_tx);
            usable.push(sat);
        }

        if usable.len() >= 4 {
            out.push(RawEpoch {
                epoch: base_epoch.epoch,
                satellite_positions_m,
                base_satellite_positions_m,
                rover_satellite_positions_m,
                base_observations: base_values
                    .into_iter()
                    .filter(|(sat, _)| usable.binary_search(sat).is_ok())
                    .collect(),
                rover_observations: rover_values
                    .into_iter()
                    .filter(|(sat, _)| usable.binary_search(sat).is_ok())
                    .collect(),
            });
        }
    }

    out
}

fn real_multignss_l1_epochs(
    sp3: &Sp3,
    base_obs: &RinexObs,
    rover_obs: &RinexObs,
    count: usize,
    systems: &[GnssSystem],
) -> Vec<RawEpoch> {
    let rover_by_epoch: BTreeMap<_, _> = rover_obs
        .epochs()
        .iter()
        .map(|epoch| (epoch_key(epoch.epoch), epoch))
        .collect();
    let mut out = Vec::new();

    for base_epoch in base_obs.epochs().iter().take(count) {
        let Some(rover_epoch) = rover_by_epoch.get(&epoch_key(base_epoch.epoch)).copied() else {
            continue;
        };
        let base_values = multignss_l1_observations(base_obs, base_epoch, systems);
        let rover_values = multignss_l1_observations(rover_obs, rover_epoch, systems);
        let base_sats = base_values.keys().cloned().collect::<BTreeSet<_>>();
        let rover_sats = rover_values.keys().cloned().collect::<BTreeSet<_>>();
        let common = base_sats
            .intersection(&rover_sats)
            .cloned()
            .collect::<Vec<_>>();

        let mut satellite_positions_m = BTreeMap::new();
        let mut base_satellite_positions_m = BTreeMap::new();
        let mut rover_satellite_positions_m = BTreeMap::new();
        let mut usable = Vec::new();

        for sat in common {
            let Some(position) = position_at(sp3, &sat, base_epoch.epoch) else {
                continue;
            };
            let Some(base_tx) =
                transmit_position_at(sp3, &sat, base_epoch.epoch, base_values[&sat].code_m, C_M_S)
            else {
                continue;
            };
            let Some(rover_tx) = transmit_position_at(
                sp3,
                &sat,
                base_epoch.epoch,
                rover_values[&sat].code_m,
                C_M_S,
            ) else {
                continue;
            };
            satellite_positions_m.insert(sat.clone(), position);
            base_satellite_positions_m.insert(sat.clone(), base_tx);
            rover_satellite_positions_m.insert(sat.clone(), rover_tx);
            usable.push(sat);
        }

        if usable.len() >= 4 {
            out.push(RawEpoch {
                epoch: base_epoch.epoch,
                satellite_positions_m,
                base_satellite_positions_m,
                rover_satellite_positions_m,
                base_observations: base_values
                    .into_iter()
                    .filter(|(sat, _)| usable.binary_search(sat).is_ok())
                    .collect(),
                rover_observations: rover_values
                    .into_iter()
                    .filter(|(sat, _)| usable.binary_search(sat).is_ok())
                    .collect(),
            });
        }
    }

    out
}

fn real_gps_l1_l2_epochs(
    sp3: &Sp3,
    base_obs: &RinexObs,
    rover_obs: &RinexObs,
    count: usize,
) -> Vec<RawDualEpoch> {
    let rover_by_epoch: BTreeMap<_, _> = rover_obs
        .epochs()
        .iter()
        .map(|epoch| (epoch_key(epoch.epoch), epoch))
        .collect();
    let mut out = Vec::new();

    for base_epoch in base_obs.epochs().iter().take(count) {
        let Some(rover_epoch) = rover_by_epoch.get(&epoch_key(base_epoch.epoch)).copied() else {
            continue;
        };
        let base_values = l1_l2_observations(base_obs, base_epoch);
        let rover_values = l1_l2_observations(rover_obs, rover_epoch);
        let base_sats = base_values.keys().cloned().collect::<BTreeSet<_>>();
        let rover_sats = rover_values.keys().cloned().collect::<BTreeSet<_>>();
        let common = base_sats
            .intersection(&rover_sats)
            .cloned()
            .collect::<Vec<_>>();

        let mut satellite_positions_m = BTreeMap::new();
        let mut base_satellite_positions_m = BTreeMap::new();
        let mut rover_satellite_positions_m = BTreeMap::new();
        let mut usable = Vec::new();

        for sat in common {
            let Some(position) = position_at(sp3, &sat, base_epoch.epoch) else {
                continue;
            };
            let Some(base_tx) =
                transmit_position_at(sp3, &sat, base_epoch.epoch, base_values[&sat].p1_m, C_M_S)
            else {
                continue;
            };
            let Some(rover_tx) =
                transmit_position_at(sp3, &sat, base_epoch.epoch, rover_values[&sat].p1_m, C_M_S)
            else {
                continue;
            };
            satellite_positions_m.insert(sat.clone(), position);
            base_satellite_positions_m.insert(sat.clone(), base_tx);
            rover_satellite_positions_m.insert(sat.clone(), rover_tx);
            usable.push(sat);
        }

        if usable.len() >= 4 {
            out.push(RawDualEpoch {
                epoch: base_epoch.epoch,
                satellite_positions_m,
                base_satellite_positions_m,
                rover_satellite_positions_m,
                base_observations: base_values
                    .into_iter()
                    .filter(|(sat, _)| usable.binary_search(sat).is_ok())
                    .collect(),
                rover_observations: rover_values
                    .into_iter()
                    .filter(|(sat, _)| usable.binary_search(sat).is_ok())
                    .collect(),
            });
        }
    }

    out
}

fn apply_mask(
    base_m: [f64; 3],
    epochs: &[RawEpoch],
    mask_deg: f64,
) -> (Vec<RawEpoch>, Vec<String>) {
    let mask_epochs = epochs
        .iter()
        .map(|epoch| ElevationMaskEpoch {
            satellite_positions_m: epoch.satellite_positions_m.clone(),
        })
        .collect::<Vec<_>>();
    let mask = apply_elevation_mask(base_m, &mask_epochs, mask_deg)
        .expect("valid elevation mask geometry");

    let masked_epochs = epochs
        .iter()
        .zip(mask.epochs.iter())
        .map(|(epoch, keep)| {
            let keep = keep
                .kept_satellite_ids
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            RawEpoch {
                epoch: epoch.epoch,
                satellite_positions_m: retain_keys(&epoch.satellite_positions_m, &keep),
                base_satellite_positions_m: retain_keys(&epoch.base_satellite_positions_m, &keep),
                rover_satellite_positions_m: retain_keys(&epoch.rover_satellite_positions_m, &keep),
                base_observations: retain_keys(&epoch.base_observations, &keep),
                rover_observations: retain_keys(&epoch.rover_observations, &keep),
            }
        })
        .collect();

    (masked_epochs, mask.masked_satellite_ids)
}

fn retain_keys<T: Clone>(
    input: &BTreeMap<String, T>,
    keep: &BTreeSet<String>,
) -> BTreeMap<String, T> {
    input
        .iter()
        .filter(|(sat, _)| keep.contains(*sat))
        .map(|(sat, value)| (sat.clone(), value.clone()))
        .collect()
}

fn single_frequency_prep_epochs(epochs: &[RawEpoch]) -> Vec<CodeSmoothingEpoch> {
    epochs
        .iter()
        .map(|epoch| CodeSmoothingEpoch {
            base_observations: code_smoothing_observations(&epoch.base_observations),
            rover_observations: code_smoothing_observations(&epoch.rover_observations),
        })
        .collect()
}

fn code_smoothing_observations(
    observations: &BTreeMap<String, ArcObservation>,
) -> Vec<CodeSmoothingObservation> {
    observations
        .iter()
        .map(|(sat, obs)| CodeSmoothingObservation {
            satellite_id: sat.clone(),
            ambiguity_id: obs.ambiguity_id.clone(),
            code_m: obs.code_m,
            phase_m: obs.phase_m,
            lli: obs.lli,
        })
        .collect()
}

fn apply_single_frequency_prep(
    raw_epochs: &[RawEpoch],
    prepared_epochs: &[CodeSmoothingEpoch],
) -> Vec<RawEpoch> {
    raw_epochs
        .iter()
        .zip(prepared_epochs)
        .map(|(raw, prepared)| {
            let base_observations = arc_observation_map(&prepared.base_observations);
            let rover_observations = arc_observation_map(&prepared.rover_observations);
            let keep = base_observations
                .keys()
                .filter(|sat| rover_observations.contains_key(*sat))
                .cloned()
                .collect::<BTreeSet<_>>();
            RawEpoch {
                epoch: raw.epoch,
                satellite_positions_m: retain_keys(&raw.satellite_positions_m, &keep),
                base_satellite_positions_m: retain_keys(&raw.base_satellite_positions_m, &keep),
                rover_satellite_positions_m: retain_keys(&raw.rover_satellite_positions_m, &keep),
                base_observations: retain_keys(&base_observations, &keep),
                rover_observations: retain_keys(&rover_observations, &keep),
            }
        })
        .collect()
}

fn arc_observation_map(
    observations: &[CodeSmoothingObservation],
) -> BTreeMap<String, ArcObservation> {
    observations
        .iter()
        .map(|obs| {
            (
                obs.satellite_id.clone(),
                ArcObservation {
                    ambiguity_id: obs.ambiguity_id.clone(),
                    code_m: obs.code_m,
                    phase_m: obs.phase_m,
                    lli: obs.lli,
                },
            )
        })
        .collect()
}

fn cycle_slip_split_epochs(epochs: &[RawEpoch]) -> (Vec<RawEpoch>, usize) {
    let prepared = prepare_cycle_slip_baseline_epochs(
        &single_frequency_prep_epochs(epochs),
        CycleSlipPolicy::SplitArc,
    )
    .expect("single-frequency cycle-slip split prep");
    (
        apply_single_frequency_prep(epochs, &prepared.epochs),
        prepared.split_arcs.len(),
    )
}

fn hatch_smoothed_epochs(epochs: &[RawEpoch], hatch_window_cap: usize) -> Vec<RawEpoch> {
    let smoothed =
        hatch_smooth_baseline_code_epochs(&single_frequency_prep_epochs(epochs), hatch_window_cap)
            .expect("Hatch-smoothed RTK epochs");
    apply_single_frequency_prep(epochs, &smoothed)
}

fn dual_cycle_slip_epochs(epochs: &[RawDualEpoch]) -> Vec<DualCycleSlipEpoch> {
    epochs
        .iter()
        .map(|epoch| DualCycleSlipEpoch {
            epoch_sort_key: format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:.9}",
                epoch.epoch.year,
                epoch.epoch.month,
                epoch.epoch.day,
                epoch.epoch.hour,
                epoch.epoch.minute,
                epoch.epoch.second
            ),
            gap_time_s: Some(j2000_seconds(epoch.epoch)),
            base_observations: dual_cycle_slip_observations(&epoch.base_observations),
            rover_observations: dual_cycle_slip_observations(&epoch.rover_observations),
        })
        .collect()
}

fn dual_cycle_slip_observations(
    observations: &BTreeMap<String, DualArcObservation>,
) -> Vec<DualCycleSlipObservation> {
    observations
        .iter()
        .map(|(sat, obs)| DualCycleSlipObservation {
            satellite_id: sat.clone(),
            ambiguity_id: obs.ambiguity_id.clone(),
            p1_m: obs.p1_m,
            p2_m: obs.p2_m,
            phi1_cycles: obs.phi1_cycles,
            phi2_cycles: obs.phi2_cycles,
            f1_hz: obs.f1_hz,
            f2_hz: obs.f2_hz,
            lli1: obs.lli1,
            lli2: obs.lli2,
        })
        .collect()
}

fn dual_epoch_observation_map(
    observations: &[DualCycleSlipObservation],
) -> BTreeMap<String, DualObservation> {
    observations
        .iter()
        .map(|obs| {
            (
                obs.satellite_id.clone(),
                DualObservation {
                    ambiguity_id: obs.ambiguity_id.clone(),
                    p1_m: obs.p1_m,
                    p2_m: obs.p2_m,
                    phi1_cycles: obs.phi1_cycles,
                    phi2_cycles: obs.phi2_cycles,
                    f1_hz: obs.f1_hz,
                    f2_hz: obs.f2_hz,
                },
            )
        })
        .collect()
}

fn prepared_dual_common_sats(epoch: &DualCycleSlipEpoch) -> Vec<String> {
    let base = epoch
        .base_observations
        .iter()
        .map(|obs| obs.satellite_id.clone())
        .collect::<BTreeSet<_>>();
    let rover = epoch
        .rover_observations
        .iter()
        .map(|obs| obs.satellite_id.clone())
        .collect::<BTreeSet<_>>();
    base.intersection(&rover).cloned().collect()
}

fn dual_epochs(prepared_epochs: &[DualCycleSlipEpoch]) -> Vec<DualEpoch> {
    prepared_epochs
        .iter()
        .map(|epoch| {
            let base = dual_epoch_observation_map(&epoch.base_observations);
            let rover = dual_epoch_observation_map(&epoch.rover_observations);
            DualEpoch {
                observations: prepared_dual_common_sats(epoch)
                    .into_iter()
                    .map(|sat| DualSatelliteObservation {
                        satellite_id: sat.clone(),
                        base: base[&sat].clone(),
                        rover: rover[&sat].clone(),
                    })
                    .collect(),
            }
        })
        .collect()
}

fn dual_setup_epochs(
    raw_epochs: &[RawDualEpoch],
    prepared_epochs: &[DualCycleSlipEpoch],
) -> Vec<DualIonosphereFreeSetupEpoch> {
    raw_epochs
        .iter()
        .zip(prepared_epochs)
        .map(|(raw, prepared)| {
            let split = civil_to_julian_split(raw.epoch);
            let base = dual_epoch_observation_map(&prepared.base_observations);
            let rover = dual_epoch_observation_map(&prepared.rover_observations);
            let keep = prepared_dual_common_sats(prepared)
                .into_iter()
                .collect::<BTreeSet<_>>();
            DualIonosphereFreeSetupEpoch {
                jd_whole: split.jd_whole,
                jd_fraction: split.fraction,
                observations: keep
                    .iter()
                    .map(|sat| DualSatelliteObservation {
                        satellite_id: sat.clone(),
                        base: base[sat].clone(),
                        rover: rover[sat].clone(),
                    })
                    .collect(),
                base_satellite_positions_m: retain_keys(&raw.base_satellite_positions_m, &keep),
                rover_satellite_positions_m: retain_keys(&raw.rover_satellite_positions_m, &keep),
            }
        })
        .collect()
}

fn dual_reference_sats(
    base_m: [f64; 3],
    raw_epochs: &[RawDualEpoch],
    prepared_epochs: &[DualCycleSlipEpoch],
) -> BTreeMap<String, String> {
    let reference_epochs = raw_epochs
        .iter()
        .zip(prepared_epochs)
        .map(|(raw, prepared)| {
            let keep = prepared_dual_common_sats(prepared)
                .into_iter()
                .collect::<BTreeSet<_>>();
            BaselineReferenceEpoch {
                available_satellite_ids: keep.iter().cloned().collect(),
                satellite_positions_m: retain_keys(&raw.satellite_positions_m, &keep),
            }
        })
        .collect::<Vec<_>>();
    baseline_reference_satellites(base_m, &reference_epochs, BaselineReferenceSelection::Auto)
        .expect("select dual-frequency reference")
}

fn raw_epochs_from_if_result(
    raw_dual_epochs: &[RawDualEpoch],
    result: &IonosphereFreeBaselineResult,
) -> Vec<RawEpoch> {
    result
        .epochs
        .iter()
        .map(|epoch| {
            let raw = &raw_dual_epochs[epoch.epoch_index];
            let keep = epoch.satellite_ids.iter().cloned().collect::<BTreeSet<_>>();
            RawEpoch {
                epoch: raw.epoch,
                satellite_positions_m: retain_keys(&raw.satellite_positions_m, &keep),
                base_satellite_positions_m: retain_keys(&raw.base_satellite_positions_m, &keep),
                rover_satellite_positions_m: retain_keys(&raw.rover_satellite_positions_m, &keep),
                base_observations: observations_to_arc_map(&epoch.base_observations),
                rover_observations: observations_to_arc_map(&epoch.rover_observations),
            }
        })
        .collect()
}

fn observations_to_arc_map(observations: &[Observation]) -> BTreeMap<String, ArcObservation> {
    observations
        .iter()
        .map(|obs| {
            (
                obs.satellite_id.clone(),
                ArcObservation {
                    ambiguity_id: obs.ambiguity_id.clone(),
                    code_m: obs.code_m,
                    phase_m: obs.phase_m,
                    lli: None,
                },
            )
        })
        .collect()
}

fn reference_sats(base_m: [f64; 3], epochs: &[RawEpoch]) -> BTreeMap<String, String> {
    let reference_epochs = epochs
        .iter()
        .map(|epoch| BaselineReferenceEpoch {
            available_satellite_ids: epoch.satellite_positions_m.keys().cloned().collect(),
            satellite_positions_m: epoch.satellite_positions_m.clone(),
        })
        .collect::<Vec<_>>();
    baseline_reference_satellites(base_m, &reference_epochs, BaselineReferenceSelection::Auto)
        .expect("select baseline reference")
}

fn sat_meas(epoch: &RawEpoch, sat: &str) -> SatMeas {
    let base = &epoch.base_observations[sat];
    let rover = &epoch.rover_observations[sat];
    SatMeas {
        sat: sat.to_string(),
        sd_ambiguity_id: single_difference_ambiguity_id(sat, base, rover),
        base_code_m: base.code_m,
        base_phase_m: base.phase_m,
        rover_code_m: rover.code_m,
        rover_phase_m: rover.phase_m,
        base_tx_pos: epoch.base_satellite_positions_m[sat],
        rover_tx_pos: epoch.rover_satellite_positions_m[sat],
        pos: epoch.satellite_positions_m[sat],
    }
}

fn single_difference_ambiguity_id(
    sat: &str,
    base_obs: &ArcObservation,
    rover_obs: &ArcObservation,
) -> String {
    let base_id = base_obs.ambiguity_id.as_str();
    let rover_id = rover_obs.ambiguity_id.as_str();
    if base_id == sat && rover_id == sat {
        sat.to_string()
    } else if base_id == sat {
        rover_id.to_string()
    } else if rover_id == sat || base_id == rover_id {
        base_id.to_string()
    } else {
        format!("{sat}:base={base_id},rover={rover_id}")
    }
}

fn satellite_system(sat: &str) -> &str {
    &sat[..1]
}

fn double_difference_ambiguity_id(
    sat: &str,
    sat_sd_id: &str,
    ref_sat: &str,
    ref_sd_id: &str,
) -> String {
    if sat_sd_id == sat && ref_sd_id == ref_sat {
        sat.to_string()
    } else {
        format!("{sat_sd_id}|ref={ref_sd_id}")
    }
}

fn core_epochs(epochs: &[RawEpoch], refs: &BTreeMap<String, String>) -> Vec<Epoch> {
    let ref_set = refs.values().cloned().collect::<BTreeSet<_>>();
    epochs
        .iter()
        .map(|epoch| Epoch {
            references: refs
                .values()
                .filter(|sat| epoch.satellite_positions_m.contains_key(*sat))
                .map(|sat| sat_meas(epoch, sat))
                .collect(),
            nonref: epoch
                .satellite_positions_m
                .keys()
                .filter(|sat| !ref_set.contains(*sat))
                .map(|sat| sat_meas(epoch, sat))
                .collect(),
            velocity_mps: None,
            dt_s: 0.0,
        })
        .collect()
}

fn all_dd_ambiguity_ids(epochs: &[Epoch]) -> Vec<String> {
    dd_ambiguity_satellites(epochs)
        .into_keys()
        .collect::<Vec<_>>()
}

fn dd_ambiguity_satellites(epochs: &[Epoch]) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for epoch in epochs {
        for meas in &epoch.nonref {
            let Some(reference) = epoch
                .references
                .iter()
                .find(|reference| satellite_system(&reference.sat) == satellite_system(&meas.sat))
            else {
                continue;
            };
            let ambiguity_id = double_difference_ambiguity_id(
                &meas.sat,
                &meas.sd_ambiguity_id,
                &reference.sat,
                &reference.sd_ambiguity_id,
            );
            out.insert(ambiguity_id, meas.sat.clone());
        }
    }
    out
}

fn all_nonreference_sats(epochs: &[Epoch], refs: &BTreeMap<String, String>) -> Vec<String> {
    let ref_set = refs.values().cloned().collect::<BTreeSet<_>>();
    epochs
        .iter()
        .flat_map(|epoch| {
            epoch
                .references
                .iter()
                .chain(epoch.nonref.iter())
                .map(|m| m.sat.clone())
        })
        .filter(|sat| !ref_set.contains(sat))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn all_sd_ids(epochs: &[Epoch]) -> Vec<String> {
    epochs
        .iter()
        .flat_map(|epoch| {
            epoch
                .references
                .iter()
                .chain(epoch.nonref.iter())
                .map(|m| m.sd_ambiguity_id.clone())
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn sd_ambiguity_satellites(epochs: &[Epoch]) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for epoch in epochs {
        for meas in epoch.references.iter().chain(epoch.nonref.iter()) {
            out.insert(meas.sd_ambiguity_id.clone(), meas.sat.clone());
        }
    }
    out
}

fn ambiguity_satellites(ids: &[String]) -> BTreeMap<String, String> {
    ids.iter().map(|id| (id.clone(), id.clone())).collect()
}

fn scalar_map(ids: &[String], value: f64) -> BTreeMap<String, f64> {
    ids.iter().map(|id| (id.clone(), value)).collect()
}

fn multignss_wavelengths_m(
    ambiguity_satellites: &BTreeMap<String, String>,
    glonass_slots: &BTreeMap<u8, i8>,
) -> BTreeMap<String, f64> {
    ambiguity_satellites
        .iter()
        .map(|(ambiguity_id, sat)| {
            let id = satellite_id(sat).expect("known GNSS satellite id");
            let band = match id.system {
                GnssSystem::BeiDou => '2',
                _ => '1',
            };
            let channel = (id.system == GnssSystem::Glonass)
                .then(|| glonass_slots.get(&id.prn).copied())
                .flatten();
            let frequency_hz =
                band_frequency_hz(id.system, band, channel).expect("L1-band carrier frequency");
            (ambiguity_id.clone(), C_M_S / frequency_hz)
        })
        .collect()
}

fn simple_model() -> MeasModel {
    MeasModel {
        code_sigma_m: 2.0,
        phase_sigma_m: 0.01,
        sagnac: true,
        stochastic: StochasticModel::Simple {
            elevation_weighting: true,
        },
    }
}

fn receiver_antenna_corrections(
    antex: &Antex,
    base_name: &str,
    rover_name: &str,
) -> ReceiverAntennaCorrections {
    let base = antex
        .antenna(base_name)
        .unwrap_or_else(|| panic!("ANTEX missing {base_name}"));
    let rover = antex
        .antenna(rover_name)
        .unwrap_or_else(|| panic!("ANTEX missing {rover_name}"));
    ReceiverAntennaCorrections {
        base: receiver_antenna_calibration(base, "G01"),
        rover: receiver_antenna_calibration(rover, "G01"),
    }
}

fn receiver_antenna_calibration(antenna: &Antenna, frequency: &str) -> ReceiverAntennaCalibration {
    let frequency = antenna
        .frequencies
        .get(frequency)
        .unwrap_or_else(|| panic!("ANTEX missing frequency {frequency} for {}", antenna.id));
    ReceiverAntennaCalibration {
        pco_neu_m: frequency.pco_m,
        noazi_pcv_m: frequency
            .pcv_samples
            .iter()
            .filter(|sample| sample.grid == PcvGrid::NoAzimuth)
            .map(|sample| (sample.zenith_deg, sample.value_m))
            .collect(),
        azi_pcv_m: frequency
            .pcv_samples
            .iter()
            .filter(|sample| sample.grid == PcvGrid::Azimuth)
            .map(|sample| {
                (
                    sample.azimuth_deg.expect("azimuth grid sample"),
                    sample.zenith_deg,
                    sample.value_m,
                )
            })
            .collect(),
    }
}

fn rtklib_model() -> MeasModel {
    MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: true,
        stochastic: StochasticModel::Rtklib,
    }
}

fn arp_position(marker_m: [f64; 3], obs: &RinexObs) -> [f64; 3] {
    let [height_m, east_m, north_m] = obs
        .header()
        .antenna_delta_hen_m
        .expect("RINEX antenna delta H/E/N");
    assert_eq!(east_m, 0.0);
    assert_eq!(north_m, 0.0);
    let inv_norm = 1.0
        / (marker_m[0] * marker_m[0] + marker_m[1] * marker_m[1] + marker_m[2] * marker_m[2])
            .sqrt();
    [
        marker_m[0] + marker_m[0] * inv_norm * height_m,
        marker_m[1] + marker_m[1] * inv_norm * height_m,
        marker_m[2] + marker_m[2] * inv_norm * height_m,
    ]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = sub3(a, b);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn json_vec3(value: &Value) -> [f64; 3] {
    [
        value["x"].as_f64().expect("x coordinate"),
        value["y"].as_f64().expect("y coordinate"),
        value["z"].as_f64().expect("z coordinate"),
    ]
}

fn json_usize(value: &Value) -> usize {
    usize::try_from(value.as_u64().expect("unsigned integer")).expect("usize value")
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[derive(Clone)]
struct SequentialRunOptions {
    initial_baseline_m: [f64; 3],
    baseline_prior_sigma_m: f64,
    ambiguity_prior_sigma_m: f64,
    hold_sigma_m: f64,
    process_noise_baseline_sigma_m: f64,
    float_only_systems: Vec<String>,
    receiver_antenna_corrections: Option<ReceiverAntennaCorrections>,
    ar_arming_sigma_m: Option<f64>,
}

impl Default for SequentialRunOptions {
    fn default() -> Self {
        Self {
            initial_baseline_m: [0.0, 0.0, 0.0],
            baseline_prior_sigma_m: 100.0,
            ambiguity_prior_sigma_m: 1_000.0,
            hold_sigma_m: 1.0e-4,
            process_noise_baseline_sigma_m: 0.0,
            float_only_systems: Vec::new(),
            receiver_antenna_corrections: None,
            ar_arming_sigma_m: None,
        }
    }
}

fn sequential_updates(
    epochs: &[Epoch],
    refs: &BTreeMap<String, String>,
    base_m: [f64; 3],
    model: &MeasModel,
    wavelengths_m: &BTreeMap<String, f64>,
    offsets_m: &BTreeMap<String, f64>,
    process_noise_baseline_sigma_m: f64,
) -> Vec<EpochUpdate> {
    sequential_updates_with_options(
        epochs,
        refs,
        base_m,
        model,
        wavelengths_m,
        offsets_m,
        SequentialRunOptions {
            process_noise_baseline_sigma_m,
            ..SequentialRunOptions::default()
        },
    )
}

fn sequential_updates_with_options(
    epochs: &[Epoch],
    refs: &BTreeMap<String, String>,
    base_m: [f64; 3],
    model: &MeasModel,
    wavelengths_m: &BTreeMap<String, f64>,
    offsets_m: &BTreeMap<String, f64>,
    run_opts: SequentialRunOptions,
) -> Vec<EpochUpdate> {
    let mut state = FilterState::new(
        refs.clone(),
        run_opts.initial_baseline_m,
        run_opts.baseline_prior_sigma_m,
        run_opts.ambiguity_prior_sigma_m,
    )
    .expect("valid sequential RTK filter state");
    let opts = UpdateOpts {
        hold_sigma_m: run_opts.hold_sigma_m,
        position_tol_m: 1.0e-4,
        ambiguity_tol_m: 1.0e-4,
        max_iterations: 10,
        process_noise_baseline_sigma_m: run_opts.process_noise_baseline_sigma_m,
        dynamics_model: DynamicsModel::ConstantPosition,
        float_only_systems: run_opts.float_only_systems,
        innovation_screen: None,
        report_residuals: true,
        receiver_antenna_corrections: run_opts.receiver_antenna_corrections,
        ar_arming_sigma_m: run_opts.ar_arming_sigma_m,
        search: SearchOpts {
            ratio_threshold: 3.0,
        },
    };
    let mut updates = Vec::with_capacity(epochs.len());
    for epoch in epochs {
        let update = update_epoch(state, epoch, base_m, model, wavelengths_m, offsets_m, &opts)
            .expect("sequential RTK update");
        state = update.state.clone();
        updates.push(update);
    }
    updates
}

#[test]
fn wettzell_static_gps_rtk_real_arc_self_validates_batch_paths() {
    let sp3 = load_sp3(&["sp3", "GBM0MGXRAP_20201770000_01D_05M_ORB_120epoch.sp3"]);
    let base_obs = RinexObs::parse(&load_text(&[
        "obs",
        "WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx",
    ]))
    .expect("parse WTZR RINEX observation fixture");
    let rover_obs = RinexObs::parse(&load_text(&[
        "obs",
        "WTZZ00DEU_R_20201770000_01D_30S_MO_120epoch.rnx",
    ]))
    .expect("parse WTZZ RINEX observation fixture");

    let base_arp_m = arp_position(WTZR_MARKER_M, &base_obs);
    let rover_arp_m = arp_position(WTZZ_MARKER_M, &rover_obs);
    let marker_baseline_m = sub3(WTZZ_MARKER_M, WTZR_MARKER_M);
    let antenna_baseline_m = sub3(rover_arp_m, base_arp_m);
    let raw_epochs = real_gps_l1_epochs(&sp3, &base_obs, &rover_obs, 120);
    assert_eq!(raw_epochs.len(), 120);

    let (split_epochs, split_count) = cycle_slip_split_epochs(&raw_epochs);
    assert_eq!(split_count, 4);
    let refs = reference_sats(base_arp_m, &split_epochs);
    assert_eq!(refs, BTreeMap::from([("G".to_string(), "G30".to_string())]));
    let epochs = core_epochs(&split_epochs, &refs);
    let ambiguity_ids = all_dd_ambiguity_ids(&epochs);
    let ambiguity_satellites = dd_ambiguity_satellites(&epochs);
    let (_frequency_hz, l1_wavelength_m) = gps_l1_constants(&base_obs);
    let wavelengths_m = scalar_map(&ambiguity_ids, l1_wavelength_m);
    let offsets_m = scalar_map(&ambiguity_ids, 0.0);

    if std::env::var("SIDEREON_DUMP_FIXTURES").is_ok() {
        dump_rtk_fixture(
            &epochs,
            base_arp_m,
            &ambiguity_ids,
            &ambiguity_satellites,
            &wavelengths_m,
            &offsets_m,
        );
    }

    let float = solve_float_baseline(
        &epochs,
        base_arp_m,
        &ambiguity_ids,
        [0.0, 0.0, 0.0],
        &simple_model(),
        FloatSolveOpts {
            position_tol_m: 1.0e-4,
            ambiguity_tol_m: 1.0e-4,
            max_iterations: 10,
        },
        None,
    )
    .expect("static float real-arc solve");

    let smoothed_epochs = hatch_smoothed_epochs(&split_epochs, 100);
    let smoothed_core_epochs = core_epochs(&smoothed_epochs, &refs);
    let smoothed_float = solve_float_baseline(
        &smoothed_core_epochs,
        base_arp_m,
        &ambiguity_ids,
        [0.0, 0.0, 0.0],
        &simple_model(),
        FloatSolveOpts {
            position_tol_m: 1.0e-4,
            ambiguity_tol_m: 1.0e-4,
            max_iterations: 10,
        },
        None,
    )
    .expect("Hatch-smoothed static float real-arc solve");

    let float_antenna_error_m = distance(float.baseline_m, antenna_baseline_m);
    let float_marker_error_m = distance(float.baseline_m, marker_baseline_m);
    assert!(float.phase_rms_m < 0.03);
    assert!(smoothed_float.code_rms_m < float.code_rms_m * 0.5);
    assert!(distance(smoothed_float.baseline_m, antenna_baseline_m) < 0.08);
    assert!(float_marker_error_m > 0.15);
    assert!(float_antenna_error_m < 0.08);

    let fixed = solve_fixed_baseline(
        &epochs,
        base_arp_m,
        AmbiguitySet {
            ids: &ambiguity_ids,
            satellites: &ambiguity_satellites,
            scale: AmbiguityScale {
                wavelengths_m: &wavelengths_m,
                offsets_m: &offsets_m,
            },
            float_only_systems: &[],
        },
        FloatPrior {
            baseline_m: float.baseline_m,
            ambiguities_m: &float.ambiguities_m,
            covariance_m: &float.ambiguity_covariance_m,
        },
        &simple_model(),
        FixedSolveOpts {
            position_tol_m: 1.0e-4,
            ambiguity_tol_m: 1.0e-4,
            max_iterations: 10,
            ratio_threshold: 3.0,
            partial_ambiguity_resolution: false,
            partial_min_ambiguities: 4,
        },
        None,
    )
    .expect("static fixed real-arc solve");

    assert_eq!(
        fixed.search.integer_status,
        sidereon_core::rtk_filter::IntegerStatus::NotFixed
    );
    assert!(fixed.search.integer_ratio.unwrap() < 3.0);
    assert!(fixed.search.integer_candidates > 0);
    let fixed_antenna_error_m = distance(fixed.baseline_m, antenna_baseline_m);
    assert!(fixed_antenna_error_m < float_antenna_error_m);
    assert!(fixed_antenna_error_m < 0.01);

    let partial_fixed = solve_fixed_baseline(
        &epochs,
        base_arp_m,
        AmbiguitySet {
            ids: &ambiguity_ids,
            satellites: &ambiguity_satellites,
            scale: AmbiguityScale {
                wavelengths_m: &wavelengths_m,
                offsets_m: &offsets_m,
            },
            float_only_systems: &[],
        },
        FloatPrior {
            baseline_m: float.baseline_m,
            ambiguities_m: &float.ambiguities_m,
            covariance_m: &float.ambiguity_covariance_m,
        },
        &simple_model(),
        FixedSolveOpts {
            position_tol_m: 1.0e-4,
            ambiguity_tol_m: 1.0e-4,
            max_iterations: 10,
            ratio_threshold: 3.0,
            partial_ambiguity_resolution: true,
            partial_min_ambiguities: 4,
        },
        None,
    )
    .expect("static partial fixed real-arc solve");

    assert_eq!(
        partial_fixed.search.integer_status,
        sidereon_core::rtk_filter::IntegerStatus::Fixed
    );
    assert!(partial_fixed.search.integer_ratio.unwrap() > 3.0);
    assert!(partial_fixed.search.partial.enabled);
    assert!(partial_fixed.search.partial.fixed);
    assert_eq!(
        partial_fixed.search.partial.fixed_ambiguities,
        vec![
            "G05",
            "G07",
            "G08",
            "G09",
            "G13",
            "G15@rover#2|ref=G30",
            "G18",
            "G27@rover#1|ref=G30",
            "G28",
        ]
    );
    assert!(!partial_fixed.search.partial.free_ambiguities.is_empty());
    let partial_antenna_error_m = distance(partial_fixed.baseline_m, antenna_baseline_m);
    assert!(partial_antenna_error_m < float_antenna_error_m);
    assert!(partial_antenna_error_m < 0.06);

    let raw_dual_epochs = real_gps_l1_l2_epochs(&sp3, &base_obs, &rover_obs, 120);
    assert_eq!(raw_dual_epochs.len(), 120);
    let prepared_dual = prepare_dual_cycle_slip_baseline_epochs(
        &dual_cycle_slip_epochs(&raw_dual_epochs),
        CycleSlipPolicy::DropSatellite,
        CycleSlipOptions::default(),
    )
    .expect("dual-frequency cycle-slip drop prep");
    let dual_refs = dual_reference_sats(base_arp_m, &raw_dual_epochs, &prepared_dual.epochs);
    let reference_sat = dual_refs
        .values()
        .next()
        .expect("single GPS reference")
        .clone();
    assert_eq!(reference_sat, "G30");
    let wide_lane_cycles = estimate_wide_lane_ambiguities(
        &dual_epochs(&prepared_dual.epochs),
        &reference_sat,
        WideLaneOptions {
            min_epochs: 2,
            tolerance_cycles: 0.5,
            skip_short_fragments: false,
        },
    )
    .expect("wide-lane integer estimates");
    assert!(!wide_lane_cycles.is_empty());
    let if_result = prepare_ionosphere_free_baseline_epochs(
        base_arp_m,
        [0.0, 0.0, 0.0],
        &dual_setup_epochs(&raw_dual_epochs, &prepared_dual.epochs),
        &reference_sat,
        &wide_lane_cycles,
        true,
    )
    .expect("dual-frequency IF baseline epochs");
    let if_raw_epochs = raw_epochs_from_if_result(&raw_dual_epochs, &if_result);
    let if_epochs = core_epochs(&if_raw_epochs, &dual_refs);
    let if_ambiguity_ids = all_dd_ambiguity_ids(&if_epochs);
    let if_ambiguity_satellites = dd_ambiguity_satellites(&if_epochs);
    let if_float = solve_float_baseline(
        &if_epochs,
        base_arp_m,
        &if_ambiguity_ids,
        [0.0, 0.0, 0.0],
        &simple_model(),
        FloatSolveOpts {
            position_tol_m: 1.0e-4,
            ambiguity_tol_m: 1.0e-4,
            max_iterations: 10,
        },
        None,
    )
    .expect("wide-lane IF float solve");
    let wide_lane_fixed = solve_fixed_baseline(
        &if_epochs,
        base_arp_m,
        AmbiguitySet {
            ids: &if_ambiguity_ids,
            satellites: &if_ambiguity_satellites,
            scale: AmbiguityScale {
                wavelengths_m: &if_result.wavelengths_m,
                offsets_m: &if_result.offsets_m,
            },
            float_only_systems: &[],
        },
        FloatPrior {
            baseline_m: if_float.baseline_m,
            ambiguities_m: &if_float.ambiguities_m,
            covariance_m: &if_float.ambiguity_covariance_m,
        },
        &simple_model(),
        FixedSolveOpts {
            position_tol_m: 1.0e-4,
            ambiguity_tol_m: 1.0e-4,
            max_iterations: 10,
            ratio_threshold: 3.0,
            partial_ambiguity_resolution: false,
            partial_min_ambiguities: 4,
        },
        None,
    )
    .expect("wide-lane/narrow-lane fixed solve");

    assert_eq!(
        wide_lane_fixed.search.integer_status,
        sidereon_core::rtk_filter::IntegerStatus::Fixed
    );
    assert!(wide_lane_fixed.search.integer_ratio.unwrap() > 3.0);
    assert!(distance(wide_lane_fixed.baseline_m, antenna_baseline_m) < 0.01);
}

/// Bounded-tolerance band (m) for canonical RTK vs the RTKLIB-faithful reference
/// RTK float baseline on the shared WTZR/WTZZ arc. Canonical and reference solve
/// the SAME SPD double-difference information system and differ ONLY in the
/// linear-algebra op-order: the reference factors it by general first-tie
/// Gaussian elimination, canonical by the owned Cholesky (square-root-information)
/// factorization. Two factorizations of one well-conditioned SPD system agree to
/// roundoff, so the converged baselines can only cluster at the level of f64
/// rounding (the observed separation on this arc is ~7e-14 m). The band is held
/// at 1 nanometre, ~1e4x above that roundoff floor; a divergence beyond it is a
/// canonical bug to root-cause, not a tolerance to widen.
const CANONICAL_VS_REFERENCE_RTK_TOL_M: f64 = 1.0e-9;

/// Surveyed-truth bound (m): the canonical RTK float baseline vs the surveyed
/// WTZR/WTZZ EPN antenna baseline (the ARP-to-ARP vector from the marker XYZ).
/// This is the same physical-truth bound the reference float baseline holds on
/// this fixture (an L1 code+phase float solve), not a bit-exact gate.
const CANONICAL_RTK_TRUTH_BOUND_M: f64 = 0.08;

/// P6 increment 2: the canonical RTK strategy, an ADDITIVE selectable strategy
/// whose canonical divergence is the numerically rigorous square-root-information
/// solve (owned Cholesky factorization of the SPD double-difference normal matrix
/// on the [`OwnedDeterministicCholesky`] kernel), not the reference's general
/// first-tie Gaussian elimination. It changes nothing about the RTKLIB-faithful
/// RTK path. Both canonical bars are checked on the real WTZR/WTZZ EPN arc:
///
///   1. DETERMINISM: canonical is bit-reproducible run-to-run (the frozen-bits
///      baseline golden below, re-asserted on a second canonical solve). The
///      whole RTK canonical path is owned scalar arithmetic (the shared block
///      fold plus the owned Cholesky square-root) with no nalgebra and no BLAS,
///      and f64 sqrt is IEEE-754 correctly rounded, so unlike canonical SPP these
///      bits are portable across platforms, not merely run-to-run on this build.
///   2. BOUNDED-TOLERANCE + TRUTH: canonical lands within
///      [`CANONICAL_VS_REFERENCE_RTK_TOL_M`] of the RTKLIB-faithful reference RTK
///      float baseline on the shared case, and within
///      [`CANONICAL_RTK_TRUTH_BOUND_M`] of the surveyed EPN antenna baseline.
#[test]
fn canonical_rtk_is_deterministic_bounded_and_truthful() {
    use sidereon_core::estimation::{
        estimate, EstimateInput, EstimateOptions, EstimateOutput, StrategyId, Technique,
    };

    let sp3 = load_sp3(&["sp3", "GBM0MGXRAP_20201770000_01D_05M_ORB_120epoch.sp3"]);
    let base_obs = RinexObs::parse(&load_text(&[
        "obs",
        "WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx",
    ]))
    .expect("parse WTZR RINEX observation fixture");
    let rover_obs = RinexObs::parse(&load_text(&[
        "obs",
        "WTZZ00DEU_R_20201770000_01D_30S_MO_120epoch.rnx",
    ]))
    .expect("parse WTZZ RINEX observation fixture");

    let base_arp_m = arp_position(WTZR_MARKER_M, &base_obs);
    let rover_arp_m = arp_position(WTZZ_MARKER_M, &rover_obs);
    let antenna_baseline_m = sub3(rover_arp_m, base_arp_m);
    let raw_epochs = real_gps_l1_epochs(&sp3, &base_obs, &rover_obs, 120);
    let (split_epochs, _split_count) = cycle_slip_split_epochs(&raw_epochs);
    let refs = reference_sats(base_arp_m, &split_epochs);
    let epochs = core_epochs(&split_epochs, &refs);
    let ambiguity_ids = all_dd_ambiguity_ids(&epochs);
    let opts = FloatSolveOpts {
        position_tol_m: 1.0e-4,
        ambiguity_tol_m: 1.0e-4,
        max_iterations: 10,
    };

    // Reference RTK float (RTKLIB-faithful), the unchanged reference path.
    let reference = solve_float_baseline(
        &epochs,
        base_arp_m,
        &ambiguity_ids,
        [0.0, 0.0, 0.0],
        &simple_model(),
        opts,
        None,
    )
    .expect("reference RTK float solve");

    let model = simple_model();
    let run_canonical = || -> sidereon_core::rtk_filter::FloatBaselineSolution {
        match estimate(
            EstimateInput::RtkFloat {
                epochs: &epochs,
                base: base_arp_m,
                ambiguity_ids: &ambiguity_ids,
                initial_baseline_m: [0.0, 0.0, 0.0],
                model: &model,
                opts,
                receiver_antenna_corrections: None,
            },
            EstimateOptions::new(StrategyId::Canonical {
                technique: Technique::Rtk,
            }),
        )
        .expect("canonical RTK float solves")
        {
            EstimateOutput::RtkFloat(solution) => *solution,
            other => panic!("canonical RTK must yield an RTK float solution, got {other:?}"),
        }
    };
    let canonical = run_canonical();

    let dpos = distance(canonical.baseline_m, reference.baseline_m);
    let terr = distance(canonical.baseline_m, antenna_baseline_m);

    // BAR 2a: bounded tolerance vs the RTKLIB-faithful reference.
    assert!(
        dpos < CANONICAL_VS_REFERENCE_RTK_TOL_M,
        "canonical RTK diverged from reference by {dpos} m (> {CANONICAL_VS_REFERENCE_RTK_TOL_M} m); root-cause, do not widen"
    );
    // BAR 2b: surveyed-truth bound.
    assert!(
        terr < CANONICAL_RTK_TRUTH_BOUND_M,
        "canonical RTK truth error was {terr} m (> {CANONICAL_RTK_TRUTH_BOUND_M} m)"
    );

    // BAR 1: frozen-bits determinism golden (portable: owned scalar + IEEE sqrt).
    assert_eq!(canonical.baseline_m[0].to_bits(), 0xbfef8e410f517d27);
    assert_eq!(canonical.baseline_m[1].to_bits(), 0xbfe5295e574d4787);
    assert_eq!(canonical.baseline_m[2].to_bits(), 0x3ff117cc1610b9af);

    // Determinism: a second canonical solve is bit-identical.
    let again = run_canonical();
    assert_eq!(
        canonical.baseline_m[0].to_bits(),
        again.baseline_m[0].to_bits()
    );
    assert_eq!(
        canonical.baseline_m[1].to_bits(),
        again.baseline_m[1].to_bits()
    );
    assert_eq!(
        canonical.baseline_m[2].to_bits(),
        again.baseline_m[2].to_bits()
    );
}

#[test]
fn wettzell_two_epoch_rtk_real_arc_solves_rtklib_prefix_target() {
    let oracle = load_oracle("wtzr_wtzz_rtklib_oracle.json");
    assert_eq!(oracle["reference"]["first_fixed_index"], 1);
    assert_eq!(oracle["per_epoch"][1]["fix_status"], "fixed");
    assert!(oracle["per_epoch"][1]["ratio"].as_f64().unwrap() >= 3.0);

    let sp3 = load_sp3(&["sp3", "COD0MGXFIN_20201770000_01D_05M_ORB.SP3"]);
    let base_obs = RinexObs::parse(&load_text(&[
        "obs",
        "WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx",
    ]))
    .expect("parse WTZR RINEX observation fixture");
    let rover_obs = RinexObs::parse(&load_text(&[
        "obs",
        "WTZZ00DEU_R_20201770000_01D_30S_MO_120epoch.rnx",
    ]))
    .expect("parse WTZZ RINEX observation fixture");

    let base_arp_m = arp_position(WTZR_MARKER_M, &base_obs);
    let rover_arp_m = arp_position(WTZZ_MARKER_M, &rover_obs);
    let antenna_baseline_m = sub3(rover_arp_m, base_arp_m);
    let raw_epochs = real_gps_l1_epochs(&sp3, &base_obs, &rover_obs, 2);
    assert_eq!(raw_epochs.len(), 2);

    let (masked_epochs, masked_sats) = apply_mask(base_arp_m, &raw_epochs, 10.0);
    assert_eq!(masked_sats, vec!["G08", "G18", "G27"]);

    let refs = reference_sats(base_arp_m, &masked_epochs);
    assert_eq!(refs, BTreeMap::from([("G".to_string(), "G30".to_string())]));

    let epochs = core_epochs(&masked_epochs, &refs);
    let (_frequency_hz, wavelength_m) = gps_l1_constants(&base_obs);
    let l1_wavelength_m = wavelength_m;
    let ambiguity_ids = all_nonreference_sats(&epochs, &refs);
    let ambiguity_satellites = ambiguity_satellites(&ambiguity_ids);
    let wavelengths_m = scalar_map(&ambiguity_ids, l1_wavelength_m);
    let offsets_m = scalar_map(&ambiguity_ids, 0.0);

    let float = solve_float_baseline(
        &epochs,
        base_arp_m,
        &ambiguity_ids,
        [0.0, 0.0, 0.0],
        &simple_model(),
        FloatSolveOpts {
            position_tol_m: 1.0e-4,
            ambiguity_tol_m: 1.0e-4,
            max_iterations: 10,
        },
        None,
    )
    .expect("float real-arc solve");

    let fixed = solve_fixed_baseline(
        &epochs,
        base_arp_m,
        AmbiguitySet {
            ids: &ambiguity_ids,
            satellites: &ambiguity_satellites,
            scale: AmbiguityScale {
                wavelengths_m: &wavelengths_m,
                offsets_m: &offsets_m,
            },
            float_only_systems: &[],
        },
        FloatPrior {
            baseline_m: float.baseline_m,
            ambiguities_m: &float.ambiguities_m,
            covariance_m: &float.ambiguity_covariance_m,
        },
        &simple_model(),
        FixedSolveOpts {
            position_tol_m: 1.0e-4,
            ambiguity_tol_m: 1.0e-4,
            max_iterations: 10,
            ratio_threshold: 3.0,
            partial_ambiguity_resolution: true,
            partial_min_ambiguities: 4,
        },
        None,
    )
    .expect("fixed real-arc solve");

    assert_eq!(
        fixed.search.integer_status,
        sidereon_core::rtk_filter::IntegerStatus::Fixed
    );
    assert!(fixed.search.integer_ratio.unwrap() >= 3.0);
    assert!(distance(fixed.baseline_m, antenna_baseline_m) < 0.01);

    let sd_ids = all_sd_ids(&epochs);
    let sd_wavelengths_m = scalar_map(&sd_ids, l1_wavelength_m);
    let sd_offsets_m = scalar_map(&sd_ids, 0.0);
    let updates = sequential_updates(
        &epochs,
        &refs,
        base_arp_m,
        &rtklib_model(),
        &sd_wavelengths_m,
        &sd_offsets_m,
        0.0,
    );
    assert!(updates.iter().any(|update| update.integer_fixed));
    assert!(
        distance(
            updates.last().unwrap().reported_baseline_m,
            antenna_baseline_m
        ) < 0.01
    );
}

#[test]
fn wettzell_kinematic_rtk_filter_tracks_rtklib_truth_class() {
    let oracle = load_oracle("wtzr_wtzz_kinematic_gps_rtklib_oracle.json");
    let epoch_count = oracle["reference"]["epochs"].as_u64().unwrap() as usize;
    let oracle_fixed_epochs = oracle["reference"]["fixed_epochs"].as_u64().unwrap() as usize;
    assert_eq!(epoch_count, 120);

    let sp3 = load_sp3(&["sp3", "COD0MGXFIN_20201770000_01D_05M_ORB.SP3"]);
    let base_obs = RinexObs::parse(&load_text(&[
        "obs",
        "WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx",
    ]))
    .expect("parse WTZR RINEX observation fixture");
    let rover_obs = RinexObs::parse(&load_text(&[
        "obs",
        "WTZZ00DEU_R_20201770000_01D_30S_MO_120epoch.rnx",
    ]))
    .expect("parse WTZZ RINEX observation fixture");

    let base_arp_m = arp_position(WTZR_MARKER_M, &base_obs);
    let rover_arp_m = arp_position(WTZZ_MARKER_M, &rover_obs);
    let antenna_baseline_m = sub3(rover_arp_m, base_arp_m);
    let raw_epochs = real_gps_l1_epochs(&sp3, &base_obs, &rover_obs, epoch_count);
    assert_eq!(raw_epochs.len(), epoch_count);

    let (masked_epochs, _masked_sats) = apply_mask(base_arp_m, &raw_epochs, 10.0);
    let refs = reference_sats(base_arp_m, &masked_epochs);
    let epochs = core_epochs(&masked_epochs, &refs);
    let (_frequency_hz, wavelength_m) = gps_l1_constants(&base_obs);
    let sd_ids = all_sd_ids(&epochs);
    let wavelengths_m = scalar_map(&sd_ids, wavelength_m);
    let offsets_m = scalar_map(&sd_ids, 0.0);

    let static_updates = sequential_updates(
        &epochs,
        &refs,
        base_arp_m,
        &rtklib_model(),
        &wavelengths_m,
        &offsets_m,
        0.0,
    );
    let kinematic_updates = sequential_updates(
        &epochs,
        &refs,
        base_arp_m,
        &rtklib_model(),
        &wavelengths_m,
        &offsets_m,
        30.0,
    );

    let static_baseline_m = static_updates.last().unwrap().reported_baseline_m;
    let kinematic_baseline_m = kinematic_updates.last().unwrap().reported_baseline_m;
    assert!(distance(kinematic_baseline_m, static_baseline_m) > 0.0);

    let kinematic_fixed = kinematic_updates
        .iter()
        .filter(|update| update.integer_fixed)
        .count();
    assert!(kinematic_fixed >= oracle_fixed_epochs - 5);
    assert!(distance(kinematic_baseline_m, antenna_baseline_m) < 0.011);

    let fixed_errors = kinematic_updates
        .iter()
        .filter(|update| update.integer_fixed)
        .map(|update| distance(update.reported_baseline_m, antenna_baseline_m))
        .collect::<Vec<_>>();
    let mean_fixed_error_m = fixed_errors.iter().sum::<f64>() / fixed_errors.len() as f64;
    assert!(fixed_errors.iter().copied().fold(0.0, f64::max) < 0.02);
    assert!(mean_fixed_error_m < 0.01);

    assert_eq!(kinematic_fixed, 120);
    assert_eq!(
        kinematic_baseline_m.map(f64::to_bits),
        [
            0xbfef_8873_53a6_52a3,
            0xbfe4_e3eb_2e47_f1d6,
            0x3ff0_fd89_2e64_d93d,
        ]
    );

    for process_sigma_m in [0.1, 1.0, 3.0, 10.0, 30.0, 100.0] {
        let updates = sequential_updates(
            &epochs,
            &refs,
            base_arp_m,
            &rtklib_model(),
            &wavelengths_m,
            &offsets_m,
            process_sigma_m,
        );
        let final_error_m = distance(
            updates.last().unwrap().reported_baseline_m,
            antenna_baseline_m,
        );
        assert!(
            final_error_m.is_finite() && final_error_m < 0.5,
            "process sigma {process_sigma_m} m ended with {final_error_m} m error"
        );
    }
}

#[test]
fn pasa_scoa_receiver_antenna_corrections_are_core_validated() {
    let oracle = load_oracle("pasa_scoa_2026_120_l1_static_fixhold_rtklib_oracle.json");
    let epoch_count = json_usize(&oracle["reference"]["epochs"]);
    let truth = &oracle["truth"];
    let base_ecef_m = json_vec3(&truth["base_station"]["marker_ecef_m"]);
    let rover_ecef_m = json_vec3(&truth["rover_station"]["marker_ecef_m"]);
    let truth_baseline_m = sub3(rover_ecef_m, base_ecef_m);

    let sp3 = load_sp3(&["sp3", "IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"]);
    let base_obs = load_obs(&["obs", "SCOA00FRA_R_20261201000_02H_30S_MO.rnx"]);
    let rover_obs = load_obs(&["obs", "PASA00ESP_R_20261201000_02H_30S_MO.rnx"]);
    let antex = load_antex(&["antex", "igs20_pasa_scoa_gps.atx"]);
    let corrections = receiver_antenna_corrections(
        &antex,
        truth["base_station"]["antenna"]
            .as_str()
            .expect("base antenna name"),
        truth["rover_station"]["antenna"]
            .as_str()
            .expect("rover antenna name"),
    );

    let initial_baseline_m = sub3(
        rover_obs
            .header()
            .approx_position_m
            .expect("rover approximate position"),
        base_ecef_m,
    );
    let raw_epochs = real_gps_l1_epochs(&sp3, &base_obs, &rover_obs, epoch_count);
    assert_eq!(raw_epochs.len(), epoch_count);
    let (masked_epochs, _masked_sats) = apply_mask(base_ecef_m, &raw_epochs, 10.0);
    let refs = reference_sats(base_ecef_m, &masked_epochs);
    let epochs = core_epochs(&masked_epochs, &refs);
    let sd_ids = all_sd_ids(&epochs);
    let (_frequency_hz, wavelength_m) = gps_l1_constants(&base_obs);
    let wavelengths_m = scalar_map(&sd_ids, wavelength_m);
    let offsets_m = scalar_map(&sd_ids, 0.0);

    let updates = sequential_updates_with_options(
        &epochs,
        &refs,
        base_ecef_m,
        &rtklib_model(),
        &wavelengths_m,
        &offsets_m,
        SequentialRunOptions {
            initial_baseline_m,
            receiver_antenna_corrections: Some(corrections),
            ..SequentialRunOptions::default()
        },
    );

    assert_eq!(updates.len(), epoch_count);
    let final_baseline_m = updates.last().unwrap().reported_baseline_m;
    assert_eq!(
        final_baseline_m.map(f64::to_bits),
        [
            0x40b3_681d_8ac5_fa51,
            0xc0d3_f0dd_6cff_76ca,
            0xc0b7_2944_95dc_3a83,
        ]
    );
    assert!(distance(final_baseline_m, truth_baseline_m) < 1.0);
}

#[test]
fn pasa_scoa_ar_arming_and_single_system_gauge_protect_real_arc() {
    let oracle = load_oracle("pasa_scoa_2026_120_l1_static_fixhold_rtklib_oracle.json");
    let epoch_count = json_usize(&oracle["reference"]["epochs"]);
    let truth = &oracle["truth"];
    let base_ecef_m = json_vec3(&truth["base_station"]["marker_ecef_m"]);
    let rover_ecef_m = json_vec3(&truth["rover_station"]["marker_ecef_m"]);
    let truth_baseline_m = sub3(rover_ecef_m, base_ecef_m);

    let sp3 = load_sp3(&["sp3", "IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"]);
    let base_obs = load_obs(&["obs", "SCOA00FRA_R_20261201000_02H_30S_MO.rnx"]);
    let rover_obs = load_obs(&["obs", "PASA00ESP_R_20261201000_02H_30S_MO.rnx"]);
    let initial_baseline_m = sub3(
        rover_obs
            .header()
            .approx_position_m
            .expect("rover approximate position"),
        base_ecef_m,
    );

    let raw_epochs = real_gps_l1_epochs(&sp3, &base_obs, &rover_obs, epoch_count);
    assert_eq!(raw_epochs.len(), epoch_count);
    let (masked_epochs, _masked_sats) = apply_mask(base_ecef_m, &raw_epochs, 15.0);
    let refs = reference_sats(base_ecef_m, &masked_epochs);
    let epochs = core_epochs(&masked_epochs, &refs);
    let sd_ids = all_sd_ids(&epochs);
    let (_frequency_hz, wavelength_m) = gps_l1_constants(&base_obs);
    let wavelengths_m = scalar_map(&sd_ids, wavelength_m);
    let offsets_m = scalar_map(&sd_ids, 0.0);

    let default_updates = sequential_updates_with_options(
        &epochs,
        &refs,
        base_ecef_m,
        &rtklib_model(),
        &wavelengths_m,
        &offsets_m,
        SequentialRunOptions {
            initial_baseline_m,
            ..SequentialRunOptions::default()
        },
    );
    assert_eq!(default_updates.len(), epoch_count);
    assert_eq!(
        default_updates
            .last()
            .unwrap()
            .reported_baseline_m
            .map(f64::to_bits),
        [
            0x40b3_6899_e001_df89,
            0xc0d3_f108_03a6_31df,
            0xc0b7_294b_cad3_6552,
        ]
    );

    let armed_updates = sequential_updates_with_options(
        &epochs,
        &refs,
        base_ecef_m,
        &rtklib_model(),
        &wavelengths_m,
        &offsets_m,
        SequentialRunOptions {
            initial_baseline_m,
            ar_arming_sigma_m: Some(0.05),
            ..SequentialRunOptions::default()
        },
    );
    assert_eq!(armed_updates.len(), epoch_count);
    assert_eq!(
        armed_updates
            .last()
            .unwrap()
            .reported_baseline_m
            .map(f64::to_bits),
        [
            0x40b3_6899_e002_14c1,
            0xc0d3_f108_03a6_8c8d,
            0xc0b7_294b_cad2_f507,
        ]
    );

    let fixed_count = armed_updates
        .iter()
        .filter(|update| update.integer_fixed)
        .count();
    assert_eq!(fixed_count, 206);
    assert!(fixed_count >= 20);

    let mut fixed_errors = armed_updates
        .iter()
        .filter(|update| update.integer_fixed)
        .map(|update| distance(update.reported_baseline_m, truth_baseline_m))
        .collect::<Vec<_>>();
    let fixed_median_m = median(&mut fixed_errors);
    assert!(fixed_median_m <= 2.0 * oracle["reference"]["mean_truth_error_m"].as_f64().unwrap());
}

#[test]
fn multignss_static_rtk_filter_reproduces_track_b_truth_gate() {
    let oracle = load_oracle("wtzr_wtzz_multignss_static_rtklib_oracle.json");
    let epoch_count = json_usize(&oracle["reference"]["epochs"]);
    let oracle_fixed_epochs = json_usize(&oracle["reference"]["fixed_epochs"]);

    let sp3 = load_sp3(&["sp3", "COD0MGXFIN_20201770000_01D_05M_ORB.SP3"]);
    let base_obs = load_obs(&["obs", "WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx"]);
    let rover_obs = load_obs(&["obs", "WTZZ00DEU_R_20201770000_01D_30S_MO_120epoch.rnx"]);
    let base_arp_m = arp_position(WTZR_MARKER_M, &base_obs);
    let rover_arp_m = arp_position(WTZZ_MARKER_M, &rover_obs);
    let antenna_baseline_m = sub3(rover_arp_m, base_arp_m);

    let raw_epochs = real_multignss_l1_epochs(
        &sp3,
        &base_obs,
        &rover_obs,
        epoch_count,
        &[
            GnssSystem::Gps,
            GnssSystem::Glonass,
            GnssSystem::Galileo,
            GnssSystem::BeiDou,
        ],
    );
    assert_eq!(raw_epochs.len(), epoch_count);
    let (masked_epochs, _masked_sats) = apply_mask(base_arp_m, &raw_epochs, 10.0);
    let refs = reference_sats(base_arp_m, &masked_epochs);
    assert_eq!(refs.keys().collect::<Vec<_>>(), vec!["C", "E", "G", "R"]);

    let epochs = core_epochs(&masked_epochs, &refs);
    let ambiguity_satellites = sd_ambiguity_satellites(&epochs);
    let wavelengths_m =
        multignss_wavelengths_m(&ambiguity_satellites, &base_obs.header().glonass_slots);
    let offsets_m = scalar_map(&all_sd_ids(&epochs), 0.0);
    let updates = sequential_updates_with_options(
        &epochs,
        &refs,
        base_arp_m,
        &rtklib_model(),
        &wavelengths_m,
        &offsets_m,
        SequentialRunOptions {
            float_only_systems: vec!["R".to_string()],
            ..SequentialRunOptions::default()
        },
    );

    assert_eq!(updates.len(), epoch_count);
    for update in &updates {
        assert!(!update.fixed_ids.iter().any(|id| id.starts_with('R')));
    }
    let fixed_count = updates.iter().filter(|update| update.integer_fixed).count();
    assert_eq!(fixed_count, oracle_fixed_epochs);

    let final_baseline_m = updates.last().unwrap().reported_baseline_m;
    assert_eq!(
        final_baseline_m.map(f64::to_bits),
        [
            0xbfef_90a0_d505_a577,
            0xbfe4_e420_4fc3_7928,
            0x3ff1_1582_e7fc_143c,
        ]
    );
    assert!(distance(final_baseline_m, antenna_baseline_m) < 0.01);

    let mut fixed_errors = updates
        .iter()
        .filter(|update| update.integer_fixed)
        .map(|update| distance(update.reported_baseline_m, antenna_baseline_m))
        .collect::<Vec<_>>();
    assert!(fixed_errors.len() >= 20);
    let fixed_median_m = median(&mut fixed_errors);
    assert!(fixed_median_m <= 2.0 * oracle["reference"]["mean_truth_error_m"].as_f64().unwrap());

    let oracle_sat_counts = oracle["per_epoch"]
        .as_array()
        .expect("oracle epochs")
        .iter()
        .map(|epoch| epoch["satellites"].as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    assert_eq!(
        (
            *oracle_sat_counts.iter().min().unwrap(),
            *oracle_sat_counts.iter().max().unwrap()
        ),
        (14, 17)
    );
    let min_core_sat_count = updates
        .iter()
        .map(|update| {
            update
                .residuals
                .iter()
                .flat_map(|residual| [&residual.satellite_id, &residual.reference_satellite_id])
                .collect::<BTreeSet<_>>()
                .len()
        })
        .min()
        .unwrap();
    assert!(min_core_sat_count >= 14);
}

/// Env-gated emitter (`SIDEREON_DUMP_FIXTURES=1`) that serializes the fully
/// built WTZR/WTZZ static-arc RTK inputs plus the engine's float and validated
/// fixed reference baselines to a JSON fixture consumed by the Python binding's
/// pytest. It reuses this validated harness verbatim; it changes no assertion
/// and never runs in a normal `cargo test`.
fn dump_rtk_fixture(
    epochs: &[Epoch],
    base_arp_m: [f64; 3],
    ambiguity_ids: &[String],
    ambiguity_satellites: &BTreeMap<String, String>,
    wavelengths_m: &BTreeMap<String, f64>,
    offsets_m: &BTreeMap<String, f64>,
) {
    use serde_json::{json, Value};

    let model = simple_model();
    let float_opts = FloatSolveOpts {
        position_tol_m: 1.0e-4,
        ambiguity_tol_m: 1.0e-4,
        max_iterations: 10,
    };
    let fixed_opts = FixedSolveOpts {
        position_tol_m: 1.0e-4,
        ambiguity_tol_m: 1.0e-4,
        max_iterations: 10,
        ratio_threshold: 3.0,
        partial_ambiguity_resolution: false,
        partial_min_ambiguities: 4,
    };
    let residual_opts = sidereon_core::rtk_filter::ResidualValidationOpts {
        threshold_sigma: None,
        max_exclusions: 0,
    };

    let float = solve_float_baseline(
        epochs,
        base_arp_m,
        ambiguity_ids,
        [0.0, 0.0, 0.0],
        &model,
        float_opts,
        None,
    )
    .expect("dump: float solve");

    let validated = sidereon_core::rtk_filter::solve_fixed_baseline_validated(
        epochs,
        base_arp_m,
        AmbiguitySet {
            ids: ambiguity_ids,
            satellites: ambiguity_satellites,
            scale: AmbiguityScale {
                wavelengths_m,
                offsets_m,
            },
            float_only_systems: &[],
        },
        [0.0, 0.0, 0.0],
        &model,
        sidereon_core::rtk_filter::ValidatedFixedSolveOpts {
            float: float_opts,
            fixed: fixed_opts,
            residual: residual_opts,
        },
        None,
    )
    .expect("dump: validated fixed solve");

    let sat_meas = |m: &SatMeas| -> Value {
        json!({
            "sat": m.sat,
            "sd_ambiguity_id": m.sd_ambiguity_id,
            "base_code_m": m.base_code_m,
            "base_phase_m": m.base_phase_m,
            "rover_code_m": m.rover_code_m,
            "rover_phase_m": m.rover_phase_m,
            "base_tx_pos": m.base_tx_pos,
            "rover_tx_pos": m.rover_tx_pos,
            "pos": m.pos,
        })
    };
    let epochs_json: Vec<Value> = epochs
        .iter()
        .map(|e| {
            json!({
                "references": e.references.iter().map(sat_meas).collect::<Vec<_>>(),
                "nonref": e.nonref.iter().map(sat_meas).collect::<Vec<_>>(),
                "velocity_mps": e.velocity_mps,
                "dt_s": e.dt_s,
            })
        })
        .collect();
    let stochastic = match model.stochastic {
        StochasticModel::Simple {
            elevation_weighting,
        } => json!({"kind": "simple", "elevation_weighting": elevation_weighting}),
        StochasticModel::Rtklib => json!({"kind": "rtklib", "elevation_weighting": false}),
    };

    let doc = json!({
        "source": "wettzell_static_gps_rtk_real_arc_self_validates_batch_paths",
        "base_arp_m": base_arp_m,
        "ambiguity_ids": ambiguity_ids,
        "ambiguity_satellites": ambiguity_satellites,
        "wavelengths_m": wavelengths_m,
        "offsets_m": offsets_m,
        "float_only_systems": Vec::<String>::new(),
        "initial_baseline_m": [0.0, 0.0, 0.0],
        "model": {
            "code_sigma_m": model.code_sigma_m,
            "phase_sigma_m": model.phase_sigma_m,
            "sagnac": model.sagnac,
            "stochastic": stochastic,
        },
        "float_opts": {
            "position_tol_m": float_opts.position_tol_m,
            "ambiguity_tol_m": float_opts.ambiguity_tol_m,
            "max_iterations": float_opts.max_iterations,
        },
        "fixed_opts": {
            "position_tol_m": fixed_opts.position_tol_m,
            "ambiguity_tol_m": fixed_opts.ambiguity_tol_m,
            "max_iterations": fixed_opts.max_iterations,
            "ratio_threshold": fixed_opts.ratio_threshold,
            "partial_ambiguity_resolution": fixed_opts.partial_ambiguity_resolution,
            "partial_min_ambiguities": fixed_opts.partial_min_ambiguities,
        },
        "residual_opts": {
            "threshold_sigma": Option::<f64>::None,
            "max_exclusions": 0,
        },
        "epochs": epochs_json,
        "expected": {
            "float_baseline_m": float.baseline_m,
            "validated_float_baseline_m": validated.float_solution.baseline_m,
            "fixed_baseline_m": validated.fixed_solution.baseline_m,
            "fixed_integer_status": format!("{:?}", validated.fixed_solution.search.integer_status),
        },
    });

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/tests/fixtures/rtk_wtzr.json");
    std::fs::create_dir_all(out.parent().unwrap()).expect("dump: create fixture dir");
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).expect("dump: write fixture");
    eprintln!("dumped RTK fixture to {out:?}");
}
