#![no_main]

mod compute_common;

use std::collections::BTreeMap;

use arbitrary::Arbitrary;
use compute_common::*;
use libfuzzer_sys::fuzz_target;
use sidereon_core::{
    carrier_phase::CycleSlipOptions,
    dgnss::{self, CodeObservation},
    observables::{ObservableEphemerisSource, ObservableState, ObservablesError},
    positioning::{
        Corrections, EphemerisSource, KlobucharCoeffs, Observation as SppObservation, SolveInputs,
        SurfaceMet,
    },
    precise_positioning::CycleSlipPolicy,
    rtk::{
        self, BaselineReferenceEpoch, BaselineReferenceSelection, CodeSmoothingEpoch,
        CodeSmoothingObservation, DualCycleSlipEpoch, DualCycleSlipObservation, DualEpoch,
        DualIonosphereFreeEpoch, DualIonosphereFreeObservation,
        DualIonosphereFreeSatelliteObservation, DualObservation, DualSatelliteObservation,
        ElevationMaskEpoch, IonosphereFreeBaselineEpoch, Observation, ReferenceSelection,
        WideLaneOptions,
    },
    rtk_filter::{
        self, AmbiguityScale, AmbiguitySet, DynamicsModel, Epoch, FilterState, FixedSolveOpts,
        FloatPrior, FloatSolveOpts, MeasModel, SatMeas, SearchOpts, StochasticModel, UpdateOpts,
    },
    GnssSatelliteId, GnssSystem,
};

#[derive(Debug, Arbitrary)]
struct Input {
    positions: [[f64; 3]; 8],
    velocities: [[f64; 3]; 8],
    baseline: [f64; 3],
    base: [f64; 3],
    values: Vec<f64>,
    scalars: [f64; 16],
    bits: [u8; 8],
}

struct FuzzSource {
    positions: [[f64; 3]; 8],
    velocities: [[f64; 3]; 8],
    clocks: [f64; 8],
}

impl FuzzSource {
    fn idx(sat: GnssSatelliteId) -> usize {
        usize::from(sat.prn.saturating_sub(1)) % 8
    }
}

impl ObservableEphemerisSource for FuzzSource {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        let idx = Self::idx(sat);
        Ok(ObservableState {
            position_ecef_m: [
                self.positions[idx][0] + self.velocities[idx][0] * t_j2000_s,
                self.positions[idx][1] + self.velocities[idx][1] * t_j2000_s,
                self.positions[idx][2] + self.velocities[idx][2] * t_j2000_s,
            ],
            clock_s: Some(self.clocks[idx]),
        })
    }
}

impl EphemerisSource for FuzzSource {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        let state = self.observable_state_at_j2000_s(sat, t_j2000_s).ok()?;
        Some((state.position_ecef_m, state.clock_s.unwrap_or(0.0)))
    }
}

fn sat(idx: usize) -> String {
    format!("G{:02}", idx + 1)
}

fn obs(input: &Input, idx: usize) -> Observation {
    Observation {
        satellite_id: sat(idx),
        ambiguity_id: format!("{}#{}", sat(idx), idx),
        code_m: input
            .values
            .get(idx)
            .copied()
            .unwrap_or(input.scalars[idx % 16]),
        phase_m: input
            .values
            .get(idx + 8)
            .copied()
            .unwrap_or(input.scalars[(idx + 1) % 16]),
    }
}

fn code_obs(input: &Input, idx: usize) -> CodeSmoothingObservation {
    CodeSmoothingObservation {
        satellite_id: sat(idx),
        ambiguity_id: format!("{}#{}", sat(idx), idx),
        code_m: input
            .values
            .get(idx)
            .copied()
            .unwrap_or(input.scalars[idx % 16]),
        phase_m: input
            .values
            .get(idx + 8)
            .copied()
            .unwrap_or(input.scalars[(idx + 1) % 16]),
        lli: Some(input.bits[idx % 8] as i64),
    }
}

fn dual(input: &Input, idx: usize) -> DualObservation {
    DualObservation {
        ambiguity_id: format!("{}#{}", sat(idx), idx),
        p1_m: input.values.get(idx).copied().unwrap_or(input.scalars[0]),
        p2_m: input
            .values
            .get(idx + 1)
            .copied()
            .unwrap_or(input.scalars[1]),
        phi1_cycles: input
            .values
            .get(idx + 2)
            .copied()
            .unwrap_or(input.scalars[2]),
        phi2_cycles: input
            .values
            .get(idx + 3)
            .copied()
            .unwrap_or(input.scalars[3]),
        f1_hz: input.scalars[4],
        f2_hz: input.scalars[5],
    }
}

fn sat_meas(input: &Input, idx: usize) -> SatMeas {
    SatMeas {
        sat: sat(idx),
        sd_ambiguity_id: format!("{}#{}", sat(idx), idx),
        base_code_m: input.values.get(idx).copied().unwrap_or(input.scalars[0]),
        base_phase_m: input
            .values
            .get(idx + 1)
            .copied()
            .unwrap_or(input.scalars[1]),
        rover_code_m: input
            .values
            .get(idx + 2)
            .copied()
            .unwrap_or(input.scalars[2]),
        rover_phase_m: input
            .values
            .get(idx + 3)
            .copied()
            .unwrap_or(input.scalars[3]),
        base_tx_pos: input.positions[idx % 8],
        rover_tx_pos: input.positions[(idx + 1) % 8],
        pos: input.positions[idx % 8],
    }
}

fn rtk_epoch(input: &Input) -> Epoch {
    let count = bounded_usize(input.bits[0], 0, 6);
    Epoch {
        references: vec![sat_meas(input, 0)],
        nonref: (1..=count).map(|idx| sat_meas(input, idx)).collect(),
        velocity_mps: Some(input.velocities[0]),
        dt_s: input.scalars[6],
    }
}

fn model(input: &Input) -> MeasModel {
    MeasModel {
        code_sigma_m: input.scalars[7],
        phase_sigma_m: input.scalars[8],
        sagnac: input.bits[1] & 1 == 1,
        stochastic: if input.bits[2] & 1 == 0 {
            StochasticModel::Simple {
                elevation_weighting: input.bits[3] & 1 == 1,
            }
        } else {
            StochasticModel::Rtklib
        },
    }
}

fn scale_maps(input: &Input, epoch: &Epoch) -> (BTreeMap<String, f64>, BTreeMap<String, f64>) {
    let mut wavelengths = BTreeMap::new();
    let mut offsets = BTreeMap::new();
    for meas in epoch.references.iter().chain(epoch.nonref.iter()) {
        wavelengths.insert(meas.sd_ambiguity_id.clone(), input.scalars[4]);
        offsets.insert(meas.sd_ambiguity_id.clone(), input.scalars[5]);
    }
    (wavelengths, offsets)
}

fuzz_target!(|data: &[u8]| {
    let Some(input) = fuzz_input::<Input>(data) else {
        return;
    };

    let count = bounded_usize(input.bits[0], 0, MAX_OBS);
    let base_obs: Vec<Observation> = (0..count).map(|idx| obs(&input, idx)).collect();
    let rover_obs: Vec<Observation> = (0..count).map(|idx| obs(&input, idx)).collect();
    if let Ok(dd) = rtk::double_differences(&base_obs, &rover_obs, ReferenceSelection::Auto) {
        for row in dd.double_differences {
            assert_all_finite("rtk::double_differences", [row.code_m, row.phase_m]);
        }
    }

    let smoothing_epoch = CodeSmoothingEpoch {
        base_observations: (0..count).map(|idx| code_obs(&input, idx)).collect(),
        rover_observations: (0..count).map(|idx| code_obs(&input, idx)).collect(),
    };
    assert_ok_or_err(
        "rtk::hatch_smooth_baseline_code_epochs",
        rtk::hatch_smooth_baseline_code_epochs(
            std::slice::from_ref(&smoothing_epoch),
            bounded_usize(input.bits[1], 1, 8),
        ),
    );
    assert_ok_or_err(
        "rtk::prepare_cycle_slip_baseline_epochs",
        rtk::prepare_cycle_slip_baseline_epochs(
            std::slice::from_ref(&smoothing_epoch),
            CycleSlipPolicy::SplitArc,
        ),
    );

    let dual_slip_epoch = DualCycleSlipEpoch {
        epoch_sort_key: "0".to_string(),
        gap_time_s: Some(input.scalars[6]),
        base_observations: (0..count)
            .map(|idx| DualCycleSlipObservation {
                satellite_id: sat(idx),
                ambiguity_id: format!("{}#{}", sat(idx), idx),
                p1_m: input.scalars[0],
                p2_m: input.scalars[1],
                phi1_cycles: input.scalars[2],
                phi2_cycles: input.scalars[3],
                f1_hz: input.scalars[4],
                f2_hz: input.scalars[5],
                lli1: Some(input.bits[2] as i64),
                lli2: Some(input.bits[3] as i64),
            })
            .collect(),
        rover_observations: (0..count)
            .map(|idx| DualCycleSlipObservation {
                satellite_id: sat(idx),
                ambiguity_id: format!("{}#{}", sat(idx), idx),
                p1_m: input.scalars[1],
                p2_m: input.scalars[2],
                phi1_cycles: input.scalars[3],
                phi2_cycles: input.scalars[4],
                f1_hz: input.scalars[5],
                f2_hz: input.scalars[6],
                lli1: Some(input.bits[4] as i64),
                lli2: Some(input.bits[5] as i64),
            })
            .collect(),
    };
    assert_ok_or_err(
        "rtk::prepare_dual_cycle_slip_baseline_epochs",
        rtk::prepare_dual_cycle_slip_baseline_epochs(
            std::slice::from_ref(&dual_slip_epoch),
            CycleSlipPolicy::DropSatellite,
            CycleSlipOptions::default(),
        ),
    );

    let positions: BTreeMap<String, [f64; 3]> = (0..count)
        .map(|idx| (sat(idx), input.positions[idx % 8]))
        .collect();
    let baseline_epoch = BaselineReferenceEpoch {
        available_satellite_ids: (0..count).map(sat).collect(),
        satellite_positions_m: positions.clone(),
    };
    assert_ok_or_err(
        "rtk::baseline_reference_satellites",
        rtk::baseline_reference_satellites(
            input.base,
            std::slice::from_ref(&baseline_epoch),
            BaselineReferenceSelection::Auto,
        ),
    );
    let mask = rtk::apply_elevation_mask(
        input.base,
        std::slice::from_ref(&ElevationMaskEpoch {
            satellite_positions_m: positions,
        }),
        input.scalars[9],
    );
    if let Ok(mask) = mask {
        let _ = mask.masked_satellite_ids;
    }

    let dual_epoch = DualEpoch {
        observations: (0..count)
            .map(|idx| DualSatelliteObservation {
                satellite_id: sat(idx),
                base: dual(&input, idx),
                rover: dual(&input, idx + 1),
            })
            .collect(),
    };
    assert_ok_or_err(
        "rtk::estimate_wide_lane_ambiguities",
        rtk::estimate_wide_lane_ambiguities(
            std::slice::from_ref(&dual_epoch),
            "G01",
            WideLaneOptions {
                min_epochs: bounded_usize(input.bits[2], 1, 4),
                tolerance_cycles: input.scalars[10],
                skip_short_fragments: input.bits[3] & 1 == 1,
            },
        ),
    );
    let if_epoch = DualIonosphereFreeEpoch {
        observations: (0..count)
            .map(|idx| DualIonosphereFreeSatelliteObservation {
                satellite_id: sat(idx),
                base: DualIonosphereFreeObservation {
                    ambiguity_id: format!("{}#{}", sat(idx), idx),
                    p1_m: input.scalars[0],
                    p2_m: input.scalars[1],
                    phi1_cycles: input.scalars[2],
                    phi2_cycles: input.scalars[3],
                    f1_hz: input.scalars[4],
                    f2_hz: input.scalars[5],
                    tropo_m: input.scalars[6],
                },
                rover: DualIonosphereFreeObservation {
                    ambiguity_id: format!("{}#{}", sat(idx), idx),
                    p1_m: input.scalars[1],
                    p2_m: input.scalars[2],
                    phi1_cycles: input.scalars[3],
                    phi2_cycles: input.scalars[4],
                    f1_hz: input.scalars[5],
                    f2_hz: input.scalars[6],
                    tropo_m: input.scalars[7],
                },
            })
            .collect(),
    };
    let wide_lane_cycles: BTreeMap<String, i64> = (0..count)
        .map(|idx| (format!("{}#{}", sat(idx), idx), idx as i64))
        .collect();
    if let Ok(result) = rtk::build_ionosphere_free_baseline_epochs(
        std::slice::from_ref(&if_epoch),
        "G01",
        &wide_lane_cycles,
    ) {
        assert_success(
            "rtk::build_ionosphere_free_baseline_epochs wavelengths",
            result.wavelengths_m.values().copied().collect::<Vec<_>>(),
        );
        assert_success(
            "rtk::build_ionosphere_free_baseline_epochs offsets",
            result.offsets_m.values().copied().collect::<Vec<_>>(),
        );
        for IonosphereFreeBaselineEpoch {
            base_observations,
            rover_observations,
            ..
        } in result.epochs
        {
            for observation in base_observations.iter().chain(rover_observations.iter()) {
                assert_all_finite(
                    "rtk::build_ionosphere_free_baseline_epochs",
                    [observation.code_m, observation.phase_m],
                );
            }
        }
    }

    let source = FuzzSource {
        positions: input.positions,
        velocities: input.velocities,
        clocks: [input.scalars[0]; 8],
    };
    let base_code: Vec<CodeObservation> = (0..count)
        .map(|idx| CodeObservation::new(sat(idx), input.values.get(idx).copied().unwrap_or(0.0)))
        .collect();
    let corrections =
        dgnss::pseudorange_corrections(&source, input.base, &base_code, input.scalars[11]);
    assert_ok_or_err("dgnss::pseudorange_corrections", corrections.as_ref());
    if let Ok(corrections) = &corrections {
        assert_ok_finite_or_err(
            "dgnss::apply_corrections",
            dgnss::apply_corrections(&base_code, corrections),
        );
    }
    let spp_inputs = SolveInputs {
        observations: (0..count)
            .filter_map(|idx| {
                Some(SppObservation {
                    satellite_id: GnssSatelliteId::new(GnssSystem::Gps, (idx as u8) + 1).ok()?,
                    pseudorange_m: input.values.get(idx).copied().unwrap_or(0.0),
                })
            })
            .collect(),
        t_rx_j2000_s: input.scalars[11],
        t_rx_second_of_day_s: input.scalars[12],
        day_of_year: input.scalars[13],
        initial_guess: [
            input.base[0],
            input.base[1],
            input.base[2],
            input.scalars[14],
        ],
        corrections: Corrections::NONE,
        klobuchar: KlobucharCoeffs {
            alpha: [input.scalars[0]; 4],
            beta: [input.scalars[1]; 4],
        },
        galileo_nequick: None,
        beidou_klobuchar: None,
        glonass_channels: BTreeMap::new(),
        met: SurfaceMet {
            pressure_hpa: input.scalars[2],
            temperature_k: input.scalars[3],
            relative_humidity: input.scalars[4],
        },
        robust: None,
    };
    assert_ok_finite_or_err(
        "dgnss::solve_position",
        dgnss::solve_position(
            &source, input.base, &base_code, &base_code, spp_inputs, false,
        ),
    );

    let float_cycles: Vec<f64> = (0..bounded_usize(input.bits[4], 0, MAX_MATRIX_DIM))
        .map(|idx| input.values.get(idx).copied().unwrap_or(0.0))
        .collect();
    let covariance = square_from_flat(&input.values, float_cycles.len().max(1));
    assert_ok_finite_or_err(
        "ils::bounded_ils_search",
        sidereon_core::ils::bounded_ils_search(
            &float_cycles,
            &covariance,
            i64::from(input.bits[5] % 2),
            bounded_usize(input.bits[6], 1, 64),
            input.scalars[15],
        ),
    );
    assert_ok_finite_or_err(
        "ils::lambda_ils_search",
        sidereon_core::ils::lambda_ils_search(&float_cycles, &covariance, input.scalars[15]),
    );

    let epoch = rtk_epoch(&input);
    let model = model(&input);
    let ambiguity_ids: Vec<String> = epoch
        .nonref
        .iter()
        .map(|meas| meas.sd_ambiguity_id.clone())
        .collect();
    assert_ok_finite_or_err(
        "rtk_filter::solve_float_baseline",
        rtk_filter::solve_float_baseline(
            std::slice::from_ref(&epoch),
            input.base,
            &ambiguity_ids,
            input.baseline,
            &model,
            FloatSolveOpts {
                position_tol_m: input.scalars[0],
                ambiguity_tol_m: input.scalars[1],
                max_iterations: bounded_usize(input.bits[0], 1, 5),
            },
            None,
        ),
    );
    let (wavelengths, offsets) = scale_maps(&input, &epoch);
    let mut references = BTreeMap::new();
    references.insert("G".to_string(), "G01#0".to_string());
    let state = FilterState::new(
        references,
        input.baseline,
        input.scalars[2],
        input.scalars[3],
    );
    assert_ok_or_err("FilterState::new", state.as_ref());
    let opts = UpdateOpts {
        hold_sigma_m: input.scalars[4],
        position_tol_m: input.scalars[5],
        ambiguity_tol_m: input.scalars[6],
        max_iterations: bounded_usize(input.bits[1], 1, 5),
        process_noise_baseline_sigma_m: input.scalars[7],
        dynamics_model: DynamicsModel::VelocityPropagated,
        float_only_systems: Vec::new(),
        innovation_screen: None,
        report_residuals: input.bits[2] & 1 == 1,
        receiver_antenna_corrections: None,
        ar_arming_sigma_m: Some(input.scalars[8]),
        search: SearchOpts {
            ratio_threshold: input.scalars[9],
        },
    };
    if let Ok(state) = state {
        assert_ok_finite_or_err(
            "rtk_filter::update_epoch",
            rtk_filter::update_epoch(
                state,
                &epoch,
                input.base,
                &model,
                &wavelengths,
                &offsets,
                &opts,
            ),
        );
    }
    let ambiguity_satellites: BTreeMap<String, String> = ambiguity_ids
        .iter()
        .map(|id| (id.clone(), id.chars().take(3).collect()))
        .collect();
    let ambiguities = AmbiguitySet {
        ids: &ambiguity_ids,
        satellites: &ambiguity_satellites,
        scale: AmbiguityScale {
            wavelengths_m: &wavelengths,
            offsets_m: &offsets,
        },
        float_only_systems: &[],
    };
    let prior_pairs: Vec<(String, f64)> = ambiguity_ids
        .iter()
        .map(|id| (id.clone(), input.scalars[10]))
        .collect();
    let prior_cov = flat_square(&input.values, ambiguity_ids.len().max(1));
    assert_ok_finite_or_err(
        "rtk_filter::solve_fixed_baseline",
        rtk_filter::solve_fixed_baseline(
            std::slice::from_ref(&epoch),
            input.base,
            ambiguities,
            FloatPrior {
                baseline_m: input.baseline,
                ambiguities_m: &prior_pairs,
                covariance_m: &prior_cov,
            },
            &model,
            FixedSolveOpts {
                position_tol_m: input.scalars[11],
                ambiguity_tol_m: input.scalars[12],
                max_iterations: bounded_usize(input.bits[3], 1, 5),
                ratio_threshold: input.scalars[13],
                partial_ambiguity_resolution: input.bits[4] & 1 == 1,
                partial_min_ambiguities: bounded_usize(input.bits[5], 1, 4),
            },
            None,
        ),
    );
});
