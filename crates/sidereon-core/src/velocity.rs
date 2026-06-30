//! Receiver velocity and clock-drift solve from GNSS range-rate observations.
//!
//! This module owns the language-independent inverse of the observable range
//! rate model. The caller supplies a known receiver position plus one epoch of
//! pseudorange-rate or Doppler observations; the core builds the deterministic
//! normal equations and returns receiver velocity, clock drift, residuals, and
//! used-satellite ordering.

use std::collections::BTreeSet;

use crate::astro::math::linear::{
    dot4, invert_4x4_cofactor, mat4_vec4, normal_matrix_4_unweighted_row_outer,
};
use crate::astro::math::vec3;

use crate::constants::{C_M_S, F_L1_HZ};
use crate::id::GnssSatelliteId;
use crate::observables::{predict, ObservableEphemerisSource, PredictOptions};

/// Observation value convention for [`solve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelocityObservable {
    /// Observation values are pseudorange rates in meters per second.
    RangeRate,
    /// Observation values are Doppler shifts in hertz and will be converted
    /// with the observation's `carrier_hz`.
    Doppler,
}

/// One satellite observation for the velocity solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityObservation {
    /// Satellite identifier.
    pub satellite_id: GnssSatelliteId,
    /// Pseudorange rate in m/s or Doppler in Hz, depending on
    /// [`VelocitySolveOptions::observable`].
    pub value: f64,
    /// Carrier frequency in hertz. Used only for Doppler observations.
    pub carrier_hz: f64,
    /// Satellite clock drift in seconds per second.
    pub sat_clock_drift_s_s: f64,
}

/// Options controlling the velocity solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocitySolveOptions {
    /// Observation value convention.
    pub observable: VelocityObservable,
    /// Apply fixed-point light-time correction in the geometry substrate.
    pub light_time: bool,
    /// Apply Earth-rotation Sagnac correction in the geometry substrate.
    pub sagnac: bool,
}

impl Default for VelocitySolveOptions {
    fn default() -> Self {
        Self {
            observable: VelocityObservable::RangeRate,
            light_time: true,
            sagnac: true,
        }
    }
}

/// Receiver velocity solve result.
#[derive(Debug, Clone, PartialEq)]
pub struct VelocitySolution {
    /// Receiver ECEF velocity in meters per second.
    pub velocity_m_s: [f64; 3],
    /// Receiver speed in meters per second.
    pub speed_m_s: f64,
    /// Receiver clock drift in seconds per second.
    pub clock_drift_s_s: f64,
    /// Post-fit range-rate residuals in meters per second, in `used_sats` order.
    pub residuals_m_s: Vec<(GnssSatelliteId, f64)>,
    /// Satellites contributing rows, in input order after unusable geometry is
    /// dropped.
    pub used_sats: Vec<GnssSatelliteId>,
}

/// Error returned by the velocity solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelocityError {
    /// No observation entries were supplied.
    NoObservations,
    /// Fewer than four usable satellites remained after geometry lookup.
    TooFewSatellites { used: usize, required: usize },
    /// The 4x4 normal matrix is singular.
    SingularGeometry,
    /// A satellite appears more than once in the input observations.
    DuplicateObservation { satellite_id: GnssSatelliteId },
    /// Doppler conversion needs a positive finite carrier frequency.
    InvalidCarrier { satellite_id: GnssSatelliteId },
    /// A scalar conversion helper received a malformed input.
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    /// An observation carries a non-finite measurement or satellite-clock drift.
    InvalidObservation { satellite_id: GnssSatelliteId },
    /// The receiver state or receive epoch is non-finite.
    InvalidReceiverState,
}

impl core::fmt::Display for VelocityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoObservations => write!(f, "no observations"),
            Self::TooFewSatellites { used, required } => {
                write!(f, "too few satellites: {used}, required {required}")
            }
            Self::SingularGeometry => write!(f, "singular geometry"),
            Self::DuplicateObservation { satellite_id } => {
                write!(f, "duplicate observation for {satellite_id}")
            }
            Self::InvalidCarrier { satellite_id } => {
                write!(f, "invalid carrier for {satellite_id}")
            }
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid velocity input {field}: {reason}")
            }
            Self::InvalidObservation { satellite_id } => {
                write!(f, "invalid observation for {satellite_id}")
            }
            Self::InvalidReceiverState => write!(f, "invalid receiver state"),
        }
    }
}

impl std::error::Error for VelocityError {}

#[derive(Debug, Clone, Copy)]
struct Row {
    sat: GnssSatelliteId,
    h: [f64; 4],
    y: f64,
}

/// Convert a Doppler shift in hertz to pseudorange rate in meters per second.
pub fn doppler_to_range_rate(doppler_hz: f64, carrier_hz: f64) -> Result<f64, VelocityError> {
    let doppler_hz = velocity_finite(doppler_hz, "doppler_hz")?;
    let carrier_hz = velocity_positive(carrier_hz, "carrier_hz")?;
    velocity_finite_output(-doppler_hz * C_M_S / carrier_hz, "range_rate_m_s")
}

/// Convert a pseudorange rate in meters per second to Doppler shift in hertz.
pub fn range_rate_to_doppler(range_rate_m_s: f64, carrier_hz: f64) -> Result<f64, VelocityError> {
    let range_rate_m_s = velocity_finite(range_rate_m_s, "range_rate_m_s")?;
    let carrier_hz = velocity_positive(carrier_hz, "carrier_hz")?;
    velocity_finite_output(-range_rate_m_s * carrier_hz / C_M_S, "doppler_hz")
}

/// Solve receiver velocity and clock drift from one epoch of observations.
pub fn solve(
    source: &dyn ObservableEphemerisSource,
    observations: &[VelocityObservation],
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: VelocitySolveOptions,
) -> Result<VelocitySolution, VelocityError> {
    if observations.is_empty() {
        return Err(VelocityError::NoObservations);
    }

    validate_receiver_state(receiver_ecef_m, t_rx_j2000_s)?;
    ensure_no_duplicates(observations)?;
    validate_observations(observations)?;
    let rows = build_rows(source, observations, receiver_ecef_m, t_rx_j2000_s, options)?;
    if rows.len() < 4 {
        return Err(VelocityError::TooFewSatellites {
            used: rows.len(),
            required: 4,
        });
    }

    let x = solve_normal_equations(&rows)?;
    assemble_solution(x, &rows)
}

fn validate_receiver_state(
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
) -> Result<(), VelocityError> {
    if receiver_ecef_m.iter().all(|value| value.is_finite()) && t_rx_j2000_s.is_finite() {
        Ok(())
    } else {
        Err(VelocityError::InvalidReceiverState)
    }
}

fn ensure_no_duplicates(observations: &[VelocityObservation]) -> Result<(), VelocityError> {
    let mut seen = BTreeSet::new();
    for obs in observations {
        if !seen.insert(obs.satellite_id) {
            return Err(VelocityError::DuplicateObservation {
                satellite_id: obs.satellite_id,
            });
        }
    }
    Ok(())
}

fn validate_observations(observations: &[VelocityObservation]) -> Result<(), VelocityError> {
    for obs in observations {
        if !(obs.value.is_finite() && obs.sat_clock_drift_s_s.is_finite()) {
            return Err(VelocityError::InvalidObservation {
                satellite_id: obs.satellite_id,
            });
        }
    }
    Ok(())
}

fn velocity_finite(x: f64, field: &'static str) -> Result<f64, VelocityError> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(VelocityError::InvalidInput {
            field,
            reason: "not finite",
        })
    }
}

fn velocity_positive(x: f64, field: &'static str) -> Result<f64, VelocityError> {
    let x = velocity_finite(x, field)?;
    if x > 0.0 {
        Ok(x)
    } else {
        Err(VelocityError::InvalidInput {
            field,
            reason: "not positive",
        })
    }
}

fn velocity_finite_output(value: f64, field: &'static str) -> Result<f64, VelocityError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(VelocityError::InvalidInput {
            field,
            reason: "out of range",
        })
    }
}

fn build_rows(
    source: &dyn ObservableEphemerisSource,
    observations: &[VelocityObservation],
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: VelocitySolveOptions,
) -> Result<Vec<Row>, VelocityError> {
    let predict_options = PredictOptions {
        carrier_hz: F_L1_HZ,
        light_time: options.light_time,
        sagnac: options.sagnac,
    };
    let mut rows = Vec::with_capacity(observations.len());

    for obs in observations {
        let rho_dot_m_s = match options.observable {
            VelocityObservable::RangeRate => obs.value,
            VelocityObservable::Doppler => {
                if !(obs.carrier_hz.is_finite() && obs.carrier_hz > 0.0) {
                    return Err(VelocityError::InvalidCarrier {
                        satellite_id: obs.satellite_id,
                    });
                }
                doppler_to_range_rate(obs.value, obs.carrier_hz).map_err(|error| match error {
                    VelocityError::InvalidInput {
                        field: "carrier_hz",
                        ..
                    } => VelocityError::InvalidCarrier {
                        satellite_id: obs.satellite_id,
                    },
                    _ => VelocityError::InvalidObservation {
                        satellite_id: obs.satellite_id,
                    },
                })?
            }
        };

        let Ok(predicted) = predict(
            source,
            obs.satellite_id,
            receiver_ecef_m,
            t_rx_j2000_s,
            predict_options,
        ) else {
            continue;
        };

        let [ex, ey, ez] = predicted.los_unit;
        let y = rho_dot_m_s - predicted.range_rate_m_s + C_M_S * obs.sat_clock_drift_s_s;
        if ![ex, ey, ez, predicted.range_rate_m_s, y]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(VelocityError::InvalidInput {
                field: "velocity row",
                reason: "out of range",
            });
        }
        rows.push(Row {
            sat: obs.satellite_id,
            h: [-ex, -ey, -ez, 1.0],
            y,
        });
    }

    Ok(rows)
}

#[allow(clippy::needless_range_loop)] // Index loops pin the normal-equation accumulation order.
fn solve_normal_equations(rows: &[Row]) -> Result<[f64; 4], VelocityError> {
    let mut aty = [0.0_f64; 4];

    for row in rows {
        for i in 0..4 {
            aty[i] += row.h[i] * row.y;
        }
    }
    let row_h: Vec<[f64; 4]> = rows.iter().map(|row| row.h).collect();
    let ata = normal_matrix_4_unweighted_row_outer(&row_h);

    let inv = invert_4x4_cofactor(&ata).ok_or(VelocityError::SingularGeometry)?;
    let solution = mat4_vec4(&inv, &aty);
    if solution.iter().all(|value| value.is_finite()) {
        Ok(solution)
    } else {
        Err(VelocityError::InvalidInput {
            field: "velocity solution",
            reason: "out of range",
        })
    }
}

fn assemble_solution(x: [f64; 4], rows: &[Row]) -> Result<VelocitySolution, VelocityError> {
    let velocity_m_s = [x[0], x[1], x[2]];
    let speed_m_s = vec3::norm3(velocity_m_s);
    let clock_drift_s_s = x[3] / C_M_S;
    let residuals_m_s: Vec<_> = rows
        .iter()
        .map(|row| (row.sat, row.y - hx(&row.h, &x)))
        .collect();
    if !velocity_m_s.iter().all(|value| value.is_finite())
        || !speed_m_s.is_finite()
        || !clock_drift_s_s.is_finite()
        || !residuals_m_s
            .iter()
            .all(|(_, residual)| residual.is_finite())
    {
        return Err(VelocityError::InvalidInput {
            field: "velocity solution",
            reason: "out of range",
        });
    }
    let used_sats = rows.iter().map(|row| row.sat).collect();
    Ok(VelocitySolution {
        velocity_m_s,
        speed_m_s,
        clock_drift_s_s,
        residuals_m_s,
        used_sats,
    })
}

fn hx(h: &[f64; 4], x: &[f64; 4]) -> f64 {
    dot4(h, x)
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests {
    use super::*;
    use crate::ephemeris::Sp3;
    use crate::observables::{
        j2000_seconds_from_split, predict, ObservableState, ObservablesError,
    };
    use crate::{GnssSatelliteId, GnssSystem};

    const T_RX_J2000_S: f64 = 646_272_000.0;
    const RECEIVER: [f64; 3] = [4_500_000.0, 500_000.0, 4_500_000.0];
    const V_TRUE: [f64; 3] = [12.0, -7.0, 3.0];
    const DRIFT_TRUE: f64 = 1.0e-9;

    fn sp3_fixture() -> Sp3 {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read SP3 fixture {path}: {e}"));
        Sp3::parse(&bytes).expect("parse SP3 fixture")
    }

    fn visible_gps(sp3: &Sp3) -> Vec<GnssSatelliteId> {
        let planning = PredictOptions {
            light_time: false,
            ..PredictOptions::default()
        };
        sp3.satellites()
            .iter()
            .copied()
            .filter(|sat| sat.system == GnssSystem::Gps)
            .filter(|sat| {
                predict(sp3, *sat, RECEIVER, T_RX_J2000_S, planning)
                    .map(|obs| obs.elevation_deg >= 5.0)
                    .unwrap_or(false)
            })
            .collect()
    }

    fn synth_range_rate(sp3: &Sp3, sat: GnssSatelliteId, v_true: [f64; 3], drift: f64) -> f64 {
        let obs = predict(sp3, sat, RECEIVER, T_RX_J2000_S, PredictOptions::default())
            .expect("predict synthetic observation");
        let e_dot_vtrue =
            obs.los_unit[0] * v_true[0] + obs.los_unit[1] * v_true[1] + obs.los_unit[2] * v_true[2];
        obs.range_rate_m_s - e_dot_vtrue + C_M_S * drift
    }

    fn synth_observations(sp3: &Sp3, sats: &[GnssSatelliteId]) -> Vec<VelocityObservation> {
        sats.iter()
            .map(|&sat| VelocityObservation {
                satellite_id: sat,
                value: synth_range_rate(sp3, sat, V_TRUE, DRIFT_TRUE),
                carrier_hz: F_L1_HZ,
                sat_clock_drift_s_s: 0.0,
            })
            .collect()
    }

    #[derive(Debug, Clone, Copy)]
    struct StaticVelocitySource {
        state: ObservableState,
    }

    impl ObservableEphemerisSource for StaticVelocitySource {
        fn observable_state_at_j2000_s(
            &self,
            _sat: GnssSatelliteId,
            _t_j2000_s: f64,
        ) -> Result<ObservableState, ObservablesError> {
            Ok(self.state)
        }
    }

    fn static_velocity_source(position_ecef_m: [f64; 3]) -> StaticVelocitySource {
        StaticVelocitySource {
            state: ObservableState {
                position_ecef_m,
                clock_s: Some(0.0),
            },
        }
    }

    #[test]
    fn split_epoch_constant_matches_orbis_velocity_fixture() {
        assert_eq!(
            j2000_seconds_from_split(2_459_024.5, 0.5).expect("valid split Julian date"),
            T_RX_J2000_S
        );
    }

    #[test]
    fn range_rate_solve_has_frozen_bits_golden() {
        let sp3 = sp3_fixture();
        let sats = visible_gps(&sp3);
        assert!(sats.len() >= 4);
        let observations = synth_observations(&sp3, &sats);

        let solution = solve(
            &sp3,
            &observations,
            RECEIVER,
            T_RX_J2000_S,
            VelocitySolveOptions::default(),
        )
        .expect("solve velocity");

        assert_eq!(
            solution.velocity_m_s.map(f64::to_bits),
            [0x4028000000000000, 0xc01c000000000016, 0x4007ffffffffff00]
        );
        assert_eq!(solution.speed_m_s.to_bits(), 0x402c6ce322982a37);
        assert_eq!(solution.clock_drift_s_s.to_bits(), 0x3e112e0be826d2ee);
        assert_eq!(
            solution
                .used_sats
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            ["G07", "G08", "G10", "G16", "G18", "G20", "G21", "G26", "G27"]
        );
        assert_eq!(
            solution
                .residuals_m_s
                .iter()
                .map(|(_, residual)| residual.to_bits())
                .collect::<Vec<_>>(),
            [
                0xbd01000000000000,
                0xbd24000000000000,
                0x3cfc000000000000,
                0xbd16000000000000,
                0xbd1a800000000000,
                0x3cf0000000000000,
                0xbd14000000000000,
                0x3d31800000000000,
                0x3d18000000000000,
            ]
        );
    }

    #[test]
    fn doppler_path_has_frozen_bits_with_per_sat_carriers() {
        let sp3 = sp3_fixture();
        let sats = visible_gps(&sp3);
        let range_rate_observations = synth_observations(&sp3, &sats);
        let doppler_observations: Vec<_> = range_rate_observations
            .iter()
            .enumerate()
            .map(|(idx, obs)| {
                let k = (idx % 14) as i8 - 7;
                let carrier_hz =
                    crate::frequencies::rinex_band_frequency_hz(GnssSystem::Glonass, '1', Some(k))
                        .expect("canonical GLONASS G1 channel carrier exists");
                VelocityObservation {
                    value: range_rate_to_doppler(obs.value, carrier_hz)
                        .expect("valid range-rate conversion"),
                    carrier_hz,
                    ..*obs
                }
            })
            .collect();

        let range_rate = solve(
            &sp3,
            &range_rate_observations,
            RECEIVER,
            T_RX_J2000_S,
            VelocitySolveOptions::default(),
        )
        .expect("range-rate solve");
        let doppler = solve(
            &sp3,
            &doppler_observations,
            RECEIVER,
            T_RX_J2000_S,
            VelocitySolveOptions {
                observable: VelocityObservable::Doppler,
                ..VelocitySolveOptions::default()
            },
        )
        .expect("doppler solve");

        assert_eq!(
            range_rate.velocity_m_s.map(f64::to_bits),
            [0x4028000000000000, 0xc01c000000000016, 0x4007ffffffffff00]
        );
        assert_eq!(
            doppler.velocity_m_s.map(f64::to_bits),
            [0x402800000000000c, 0xc01c00000000000f, 0x4007ffffffffff60]
        );
        assert_eq!(doppler.speed_m_s.to_bits(), 0x402c6ce322982a44);
        assert_eq!(doppler.clock_drift_s_s.to_bits(), 0x3e112e0be826d4b8);
        assert_eq!(
            doppler
                .residuals_m_s
                .iter()
                .map(|(_, residual)| residual.to_bits())
                .collect::<Vec<_>>(),
            [
                0x3d24c00000000000,
                0xbd2b000000000000,
                0xbd00000000000000,
                0xbd00000000000000,
                0xbd0b000000000000,
                0x3d06000000000000,
                0x0000000000000000,
                0x3d40c00000000000,
                0x3d22000000000000,
            ]
        );
    }

    #[test]
    fn validates_core_error_cases() {
        let sp3 = sp3_fixture();
        let sats = visible_gps(&sp3);
        let mut observations = synth_observations(&sp3, &sats);
        let first = observations[0].satellite_id;

        assert_eq!(
            solve(
                &sp3,
                &[],
                RECEIVER,
                T_RX_J2000_S,
                VelocitySolveOptions::default()
            ),
            Err(VelocityError::NoObservations)
        );

        assert_eq!(
            solve(
                &sp3,
                &observations[..3],
                RECEIVER,
                T_RX_J2000_S,
                VelocitySolveOptions::default()
            ),
            Err(VelocityError::TooFewSatellites {
                used: 3,
                required: 4
            })
        );

        observations[1].satellite_id = first;
        assert_eq!(
            solve(
                &sp3,
                &observations,
                RECEIVER,
                T_RX_J2000_S,
                VelocitySolveOptions::default()
            ),
            Err(VelocityError::DuplicateObservation {
                satellite_id: first
            })
        );

        let invalid_carrier = [VelocityObservation {
            satellite_id: first,
            value: 1.0,
            carrier_hz: -1.0,
            sat_clock_drift_s_s: 0.0,
        }];
        assert_eq!(
            solve(
                &sp3,
                &invalid_carrier,
                RECEIVER,
                T_RX_J2000_S,
                VelocitySolveOptions {
                    observable: VelocityObservable::Doppler,
                    ..VelocitySolveOptions::default()
                }
            ),
            Err(VelocityError::InvalidCarrier {
                satellite_id: first
            })
        );
    }

    #[test]
    fn rejects_non_finite_velocity_inputs() {
        let sp3 = sp3_fixture();
        let sats = visible_gps(&sp3);
        let mut observations = synth_observations(&sp3, &sats);
        let first = observations[0].satellite_id;

        observations[0].value = f64::NAN;
        assert_eq!(
            solve(
                &sp3,
                &observations,
                RECEIVER,
                T_RX_J2000_S,
                VelocitySolveOptions::default()
            ),
            Err(VelocityError::InvalidObservation {
                satellite_id: first
            })
        );

        observations[0].value = 0.0;
        observations[0].sat_clock_drift_s_s = f64::NAN;
        assert_eq!(
            solve(
                &sp3,
                &observations,
                RECEIVER,
                T_RX_J2000_S,
                VelocitySolveOptions::default()
            ),
            Err(VelocityError::InvalidObservation {
                satellite_id: first
            })
        );

        observations[0].sat_clock_drift_s_s = 0.0;
        let mut bad_receiver = RECEIVER;
        bad_receiver[0] = f64::NAN;
        assert_eq!(
            solve(
                &sp3,
                &observations,
                bad_receiver,
                T_RX_J2000_S,
                VelocitySolveOptions::default()
            ),
            Err(VelocityError::InvalidReceiverState)
        );

        assert_eq!(
            solve(
                &sp3,
                &observations,
                RECEIVER,
                f64::NAN,
                VelocitySolveOptions::default()
            ),
            Err(VelocityError::InvalidReceiverState)
        );
    }

    #[test]
    fn conversion_helpers_reject_invalid_domains() {
        assert_eq!(
            doppler_to_range_rate(f64::NAN, F_L1_HZ),
            Err(VelocityError::InvalidInput {
                field: "doppler_hz",
                reason: "not finite"
            })
        );
        assert_eq!(
            range_rate_to_doppler(f64::INFINITY, F_L1_HZ),
            Err(VelocityError::InvalidInput {
                field: "range_rate_m_s",
                reason: "not finite"
            })
        );

        for carrier_hz in [f64::NAN, f64::INFINITY] {
            assert_eq!(
                doppler_to_range_rate(1.0, carrier_hz),
                Err(VelocityError::InvalidInput {
                    field: "carrier_hz",
                    reason: "not finite"
                })
            );
            assert_eq!(
                range_rate_to_doppler(1.0, carrier_hz),
                Err(VelocityError::InvalidInput {
                    field: "carrier_hz",
                    reason: "not finite"
                })
            );
        }

        for carrier_hz in [0.0, -1.0] {
            assert_eq!(
                doppler_to_range_rate(1.0, carrier_hz),
                Err(VelocityError::InvalidInput {
                    field: "carrier_hz",
                    reason: "not positive"
                })
            );
            assert_eq!(
                range_rate_to_doppler(1.0, carrier_hz),
                Err(VelocityError::InvalidInput {
                    field: "carrier_hz",
                    reason: "not positive"
                })
            );
        }
    }

    #[test]
    fn solve_rejects_non_finite_internal_rows() {
        let source = static_velocity_source([20_200_000.0, 14_000_000.0, 21_700_000.0]);
        let observations: Vec<_> = (1..=4)
            .map(|prn| VelocityObservation {
                satellite_id: GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid sat"),
                value: 0.0,
                carrier_hz: F_L1_HZ,
                sat_clock_drift_s_s: f64::MAX,
            })
            .collect();

        assert_eq!(
            solve(
                &source,
                &observations,
                [0.0, 0.0, 0.0],
                646_272_000.0,
                VelocitySolveOptions {
                    light_time: false,
                    sagnac: false,
                    ..VelocitySolveOptions::default()
                }
            ),
            Err(VelocityError::InvalidInput {
                field: "velocity row",
                reason: "out of range",
            })
        );
    }
}
