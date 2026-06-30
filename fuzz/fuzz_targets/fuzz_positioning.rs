#![no_main]

mod compute_common;

use std::collections::BTreeMap;

use arbitrary::Arbitrary;
use compute_common::*;
use libfuzzer_sys::fuzz_target;
use sidereon_core::{
    estimation::recipe::SolverRecipe,
    geometry::line_of_sight_from_az_el_deg,
    observables::{ObservableEphemerisSource, ObservableState, ObservablesError},
    positioning::{
        self, Corrections, EphemerisSource, KlobucharCoeffs, LineOfSight, Observation,
        RobustConfig, SolveInputs, SurfaceMet,
    },
    ppp_corrections::CivilDateTime,
    precise_positioning::raim::{
        global_test, global_test_with_geometry, per_satellite_statistics, protection_levels,
    },
    precise_positioning::{
        self, FixedAmbiguityOptions, FixedSolveConfig, FloatEpoch, FloatObservation, FloatResidual,
        FloatSolution, FloatSolveConfig, FloatSolveOptions, FloatState, FloatStatus,
        KinematicConfig, KinematicMotionModel, KinematicState, MeasurementWeights, RaimConfig,
        RaimGeometryRow, RangeCorrections, ReceiverVelocityState, TroposphereOptions,
        VelocityConfig, VelocityObservation as PppVelocityObservation, VelocityRobustConfig,
    },
    velocity::{self, VelocityObservable, VelocityObservation, VelocitySolveOptions},
    GnssSatelliteId, GnssSystem, Wgs84Geodetic,
};

#[derive(Debug, Arbitrary)]
struct Input {
    positions: [[f64; 3]; 8],
    velocities: [[f64; 3]; 8],
    clocks: [f64; 8],
    clock_rates: [f64; 8],
    receiver: [f64; 4],
    values: Vec<f64>,
    bits: [u8; 8],
    scalars: [f64; 16],
}

struct FuzzSource {
    positions: [[f64; 3]; 8],
    velocities: [[f64; 3]; 8],
    clocks: [f64; 8],
    clock_rates: [f64; 8],
}

impl FuzzSource {
    fn idx(sat: GnssSatelliteId) -> usize {
        usize::from(sat.prn.saturating_sub(1)) % 8
    }

    fn state(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> ObservableState {
        let idx = Self::idx(sat);
        let position_ecef_m = [
            self.positions[idx][0] + self.velocities[idx][0] * t_j2000_s,
            self.positions[idx][1] + self.velocities[idx][1] * t_j2000_s,
            self.positions[idx][2] + self.velocities[idx][2] * t_j2000_s,
        ];
        ObservableState {
            position_ecef_m,
            clock_s: Some(self.clocks[idx] + self.clock_rates[idx] * t_j2000_s),
        }
    }
}

impl ObservableEphemerisSource for FuzzSource {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        Ok(self.state(sat, t_j2000_s))
    }
}

impl EphemerisSource for FuzzSource {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        let state = self.state(sat, t_j2000_s);
        Some((state.position_ecef_m, state.clock_s.unwrap_or(0.0)))
    }
}

fn gps_sat(idx: usize) -> Option<GnssSatelliteId> {
    GnssSatelliteId::new(GnssSystem::Gps, (idx as u8 % 32) + 1).ok()
}

fn observations(input: &Input) -> Vec<Observation> {
    let count = bounded_usize(input.bits[0], 0, MAX_OBS);
    (0..count)
        .filter_map(|idx| {
            Some(Observation {
                satellite_id: gps_sat(idx)?,
                pseudorange_m: input
                    .values
                    .get(idx)
                    .copied()
                    .unwrap_or(input.scalars[idx % 16]),
            })
        })
        .collect()
}

fn float_observations(input: &Input) -> Vec<FloatObservation> {
    let count = bounded_usize(input.bits[1], 0, MAX_OBS);
    (0..count)
        .filter_map(|idx| {
            let sat = gps_sat(idx)?;
            let id = sat.to_string();
            Some(FloatObservation {
                sat,
                satellite_id: id.clone(),
                ambiguity_id: id,
                code_m: input
                    .values
                    .get(idx)
                    .copied()
                    .unwrap_or(input.scalars[idx % 16]),
                phase_m: input
                    .values
                    .get(idx + MAX_OBS)
                    .copied()
                    .unwrap_or(input.scalars[(idx + 1) % 16]),
                freq1_hz: input.scalars[6],
                freq2_hz: input.scalars[7],
            })
        })
        .collect()
}

fn float_epoch(input: &Input) -> FloatEpoch {
    FloatEpoch {
        epoch: CivilDateTime {
            year: 2020,
            month: 1,
            day: 1,
            hour: 0,
            minute: 0,
            second: input.scalars[8],
        },
        jd_whole: input.scalars[9],
        jd_fraction: input.scalars[10],
        t_rx_j2000_s: input.scalars[11],
        observations: float_observations(input),
    }
}

fn ppp_state(input: &Input, epoch: &FloatEpoch) -> FloatState {
    FloatState {
        position_m: [input.receiver[0], input.receiver[1], input.receiver[2]],
        clocks_m: vec![input.receiver[3]],
        ambiguities_m: epoch
            .observations
            .iter()
            .map(|obs| (obs.ambiguity_id.clone(), input.scalars[12]))
            .collect(),
        ztd_m: input.scalars[13],
    }
}

fn float_config(input: &Input) -> FloatSolveConfig {
    FloatSolveConfig {
        weights: MeasurementWeights {
            code: input.scalars[0],
            phase: input.scalars[1],
            elevation_weighting: input.bits[2] & 1 == 1,
        },
        tropo: TroposphereOptions::disabled(),
        corrections: RangeCorrections::disabled(),
        opts: FloatSolveOptions {
            max_iterations: bounded_usize(input.bits[3], 1, 5),
            position_tolerance_m: input.scalars[2],
            clock_tolerance_m: input.scalars[3],
            ambiguity_tolerance_m: input.scalars[4],
            ztd_tolerance_m: input.scalars[5],
        },
        residual_screen: input.bits[4] & 1 == 1,
    }
}

fn raim_residuals(epoch: &FloatEpoch, input: &Input) -> Vec<FloatResidual> {
    epoch
        .observations
        .iter()
        .enumerate()
        .map(|(idx, obs)| FloatResidual {
            epoch_index: 0,
            satellite_id: obs.satellite_id.clone(),
            code_m: input.values.get(idx).copied().unwrap_or(0.0),
            phase_m: input.values.get(idx + 1).copied().unwrap_or(0.0),
            code_weight: input.scalars[0],
            phase_weight: input.scalars[1],
        })
        .collect()
}

fn finite_square(dim: usize, value: f64) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; dim]; dim];
    for (idx, row) in m.iter_mut().enumerate() {
        row[idx] = value;
    }
    m
}

fuzz_target!(|data: &[u8]| {
    let Some(input) = fuzz_input::<Input>(data) else {
        return;
    };
    let source = FuzzSource {
        positions: input.positions,
        velocities: input.velocities,
        clocks: input.clocks,
        clock_rates: input.clock_rates,
    };
    let receiver = [input.receiver[0], input.receiver[1], input.receiver[2]];
    let geodetic = Wgs84Geodetic::new(input.scalars[14], input.scalars[15], input.receiver[3]);
    assert_ok_finite_or_err("Wgs84Geodetic::new", geodetic.as_ref());
    let geodetic = geodetic.ok();

    let solve_inputs = SolveInputs {
        observations: observations(&input),
        t_rx_j2000_s: input.scalars[11],
        t_rx_second_of_day_s: input.scalars[8],
        day_of_year: input.scalars[9],
        initial_guess: input.receiver,
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
        robust: Some(RobustConfig {
            huber_k: input.scalars[5],
            scale_floor_m: input.scalars[6],
            max_outer: bounded_usize(input.bits[5], 1, 4),
            outer_tol_m: input.scalars[7],
        }),
    };
    assert_ok_finite_or_err(
        "positioning::solve",
        positioning::solve(&source, &solve_inputs, input.bits[6] & 1 == 1),
    );
    assert_ok_finite_or_err(
        "positioning::solve_with_solver",
        positioning::solve_with_solver(
            &source,
            &solve_inputs,
            input.bits[6] & 1 == 1,
            SolverRecipe::OwnedDeterministicTrf,
        ),
    );

    let los: Vec<LineOfSight> = cap_vec(input.positions.to_vec(), MAX_OBS)
        .into_iter()
        .map(|p| LineOfSight::new(p[0], p[1], p[2]))
        .collect();
    let weights: Vec<f64> = (0..los.len())
        .map(|idx| input.values.get(idx).copied().unwrap_or(1.0))
        .collect();
    if let Some(geodetic) = geodetic {
        assert_ok_finite_or_err(
            "positioning::dop",
            positioning::dop(&los, &weights, geodetic),
        );
        assert_ok_finite_or_err(
            "positioning::line_of_sight_from_az_el_deg",
            line_of_sight_from_az_el_deg(
                input.scalars[0],
                input.scalars[1],
                geodetic,
            ),
        );
    }

    let epoch = float_epoch(&input);
    let state = ppp_state(&input, &epoch);
    let config = float_config(&input);
    assert_ok_finite_or_err(
        "precise_positioning::solve_float_epoch",
        precise_positioning::solve_float_epoch(
            &source,
            epoch.clone(),
            state.clone(),
            config.clone(),
        ),
    );
    assert_ok_finite_or_err(
        "precise_positioning::solve_float_epochs",
        precise_positioning::solve_float_epochs(
            &source,
            std::slice::from_ref(&epoch),
            state.clone(),
            config.clone(),
        ),
    );
    let residuals = raim_residuals(&epoch, &input);
    let geometry: Vec<RaimGeometryRow> = residuals
        .iter()
        .enumerate()
        .map(|(idx, residual)| RaimGeometryRow {
            satellite_id: residual.satellite_id.clone(),
            line_of_sight: los
                .get(idx)
                .copied()
                .unwrap_or_else(|| LineOfSight::new(1.0, 0.0, 0.0)),
        })
        .collect();
    let raim_config = RaimConfig {
        false_alarm_probability: input.scalars[0],
        missed_detection_probability: input.scalars[1],
        measurement_sigma_m: input.scalars[2],
        chi_square_threshold: Some(input.scalars[3]),
    };
    assert_ok_finite_or_err(
        "precise_positioning::global_test",
        global_test(&residuals, bounded_usize(input.bits[7], 1, 16), raim_config),
    );
    assert_ok_finite_or_err(
        "precise_positioning::global_test_with_geometry",
        global_test_with_geometry(&residuals, &geometry, raim_config),
    );
    assert_ok_finite_or_err(
        "precise_positioning::per_satellite_statistics",
        per_satellite_statistics(&residuals, &geometry, raim_config),
    );
    if let Some(geodetic) = geodetic {
        assert_ok_finite_or_err(
            "precise_positioning::protection_levels",
            protection_levels(&geometry, geodetic, raim_config),
        );
    }
    let float_solution = FloatSolution {
        position_m: receiver,
        epoch_clocks_m: vec![input.receiver[3]],
        ambiguities_m: state.ambiguities_m.clone(),
        ztd_residual_m: Some(input.scalars[13]),
        residuals_m: residuals.clone(),
        used_sats: epoch
            .observations
            .iter()
            .map(|obs| obs.satellite_id.clone())
            .collect(),
        iterations: bounded_usize(input.bits[3], 1, 5),
        converged: input.bits[0] & 1 == 1,
        status: FloatStatus::MaxIterations,
        code_rms_m: input.scalars[4],
        phase_rms_m: input.scalars[5],
        weighted_rms_m: input.scalars[6],
    };
    let fixed_config = FixedSolveConfig {
        weights: config.weights,
        tropo: config.tropo,
        corrections: config.corrections.clone(),
        opts: config.opts,
        ambiguity: FixedAmbiguityOptions {
            wavelengths_m: epoch
                .observations
                .iter()
                .map(|obs| (obs.ambiguity_id.clone(), input.scalars[7]))
                .collect(),
            offsets_m: BTreeMap::new(),
            ratio_threshold: input.scalars[8],
        },
    };
    assert_ok_finite_or_err(
        "precise_positioning::solve_fixed_from_float",
        precise_positioning::solve_fixed_from_float(
            &source,
            std::slice::from_ref(&epoch),
            float_solution,
            fixed_config,
        ),
    );
    assert_ok_finite_or_err(
        "precise_positioning::solve_float_epoch_with_raim",
        precise_positioning::solve_float_epoch_with_raim(
            &source,
            epoch.clone(),
            state.clone(),
            config.clone(),
            raim_config,
        ),
    );

    let mut kin_state = KinematicState {
        position_m: receiver,
        clock_m: input.receiver[3],
        ztd_residual_m: input.scalars[13],
        ambiguities_m: state.ambiguities_m.clone(),
    };
    let mut kin_cov = finite_square(kin_state.dimension(), input.scalars[10]);
    let kin_config = KinematicConfig {
        initial_state: kin_state.clone(),
        initial_covariance_m2: kin_cov.clone(),
        motion: KinematicMotionModel::ConstantVelocity {
            velocity_m_s: input.velocities[0],
        },
        ..KinematicConfig::default()
    };
    let active: Vec<String> = epoch
        .observations
        .iter()
        .map(|obs| obs.ambiguity_id.clone())
        .collect();
    assert_ok_or_err(
        "precise_positioning::predict_kinematic_state",
        precise_positioning::predict_kinematic_state(
            &mut kin_state,
            &mut kin_cov,
            input.scalars[11],
            &active,
            &kin_config,
        ),
    );
    assert_ok_finite_or_err(
        "precise_positioning::correct_kinematic_state",
        precise_positioning::correct_kinematic_state(
            &source,
            &epoch,
            &mut kin_state,
            &mut kin_cov,
            &kin_config,
        ),
    );
    assert_ok_finite_or_err(
        "precise_positioning::solve_kinematic_ppp",
        precise_positioning::solve_kinematic_ppp(&source, std::slice::from_ref(&epoch), kin_config),
    );

    let ppp_velocity_obs: Vec<PppVelocityObservation> =
        (0..bounded_usize(input.bits[1], 0, MAX_OBS))
            .filter_map(|idx| {
                Some(PppVelocityObservation {
                    sat: gps_sat(idx)?,
                    satellite_position_m: input.positions[idx % 8],
                    satellite_velocity_m_s: input.velocities[idx % 8],
                    measured_range_rate_m_s: input.values.get(idx).copied().unwrap_or(0.0),
                    sigma_m_s: input.scalars[12],
                    satellite_clock_drift_m_s: input.clock_rates[idx % 8],
                })
            })
            .collect();
    if let Some(first) = ppp_velocity_obs.first() {
        assert_option_finite(
            "precise_positioning::predict_range_rate_m_s",
            precise_positioning::predict_range_rate_m_s(
                first,
                ReceiverVelocityState::stationary_at(receiver),
            ),
        );
    }
    assert_ok_finite_or_err(
        "precise_positioning::solve_velocity",
        precise_positioning::solve_velocity(
            &ppp_velocity_obs,
            receiver,
            VelocityConfig {
                minimum_observations: bounded_usize(input.bits[0], 1, 8),
                robust: Some(VelocityRobustConfig {
                    huber_k: input.scalars[0],
                    scale_floor_m_s: input.scalars[1],
                    max_outer: bounded_usize(input.bits[2], 1, 4),
                    outer_tol_m_s: input.scalars[2],
                }),
            },
        ),
    );

    let velocity_obs: Vec<VelocityObservation> = (0..bounded_usize(input.bits[1], 0, MAX_OBS))
        .filter_map(|idx| {
            Some(VelocityObservation {
                satellite_id: gps_sat(idx)?,
                value: input.values.get(idx).copied().unwrap_or(0.0),
                carrier_hz: input.scalars[7],
                sat_clock_drift_s_s: input.clock_rates[idx % 8],
            })
        })
        .collect();
    assert_ok_finite_or_err(
        "velocity::doppler_to_range_rate",
        velocity::doppler_to_range_rate(input.scalars[0], input.scalars[1]),
    );
    assert_ok_finite_or_err(
        "velocity::range_rate_to_doppler",
        velocity::range_rate_to_doppler(input.scalars[2], input.scalars[3]),
    );
    assert_ok_finite_or_err(
        "velocity::solve",
        velocity::solve(
            &source,
            &velocity_obs,
            receiver,
            input.scalars[11],
            VelocitySolveOptions {
                observable: if input.bits[0] & 1 == 0 {
                    VelocityObservable::RangeRate
                } else {
                    VelocityObservable::Doppler
                },
                light_time: input.bits[1] & 1 == 1,
                sagnac: input.bits[2] & 1 == 1,
            },
        ),
    );
});
