#![no_main]

mod compute_common;

use arbitrary::Arbitrary;
use compute_common::*;
use libfuzzer_sys::fuzz_target;
use sidereon_core::{
    carrier_phase::{self, ArcEpoch, CycleSlipOptions},
    combinations,
    observables::{
        self, ObservableEphemerisSource, ObservableState, ObservablesError, PredictOptions,
    },
    signal::{self, AcquisitionOptions, CorrelateOptions, IqSample, ReplicaOptions},
    velocity, GnssSatelliteId, GnssSystem,
};

#[derive(Debug, Arbitrary)]
struct Input {
    positions: [[f64; 3]; 8],
    velocities: [[f64; 3]; 8],
    receiver: [f64; 3],
    samples: Vec<[f64; 2]>,
    chips: Vec<i8>,
    values: Vec<f64>,
    params: [f64; 16],
    ints: [i64; 4],
    bits: [u8; 4],
}

struct Source {
    positions: [[f64; 3]; 8],
    velocities: [[f64; 3]; 8],
}

impl ObservableEphemerisSource for Source {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        let idx = usize::from(sat.prn.saturating_sub(1)) % 8;
        Ok(ObservableState {
            position_ecef_m: [
                self.positions[idx][0] + self.velocities[idx][0] * t_j2000_s,
                self.positions[idx][1] + self.velocities[idx][1] * t_j2000_s,
                self.positions[idx][2] + self.velocities[idx][2] * t_j2000_s,
            ],
            clock_s: Some(self.values_clock(idx)),
        })
    }
}

impl Source {
    fn values_clock(&self, idx: usize) -> f64 {
        self.velocities[idx][0]
    }
}

fn iq(input: &Input) -> Vec<IqSample> {
    cap_vec(input.samples.clone(), 256)
        .into_iter()
        .map(|sample| IqSample::new(sample[0], sample[1]))
        .collect()
}

fn arc(input: &Input) -> Vec<ArcEpoch> {
    (0..bounded_usize(input.bits[0], 0, MAX_OBS))
        .map(|idx| ArcEpoch {
            phi1_cycles: Some(input.values.get(idx).copied().unwrap_or(input.params[0])),
            phi2_cycles: Some(
                input
                    .values
                    .get(idx + 1)
                    .copied()
                    .unwrap_or(input.params[1]),
            ),
            p1_m: Some(
                input
                    .values
                    .get(idx + 2)
                    .copied()
                    .unwrap_or(input.params[2]),
            ),
            p2_m: Some(
                input
                    .values
                    .get(idx + 3)
                    .copied()
                    .unwrap_or(input.params[3]),
            ),
            lli1: Some(input.ints[0]),
            lli2: Some(input.ints[1]),
            f1_hz: Some(input.params[4]),
            f2_hz: Some(input.params[5]),
            gap_time_s: Some(input.params[6] + idx as f64),
        })
        .collect()
}

fuzz_target!(|data: &[u8]| {
    let Some(input) = fuzz_input::<Input>(data) else {
        return;
    };
    let prn = input.ints[0];
    assert_ok_or_err("signal::ca_code", signal::ca_code(prn));
    assert_ok_or_err("signal::ca_chip", signal::ca_chip(prn, input.ints[1]));
    let chips = cap_vec(input.chips.clone(), 128);
    let other = cap_vec(
        input.chips.iter().rev().copied().collect::<Vec<_>>(),
        chips.len(),
    );
    assert_ok_or_err::<_, ()>(
        "signal::autocorrelation",
        Ok(signal::autocorrelation(&chips)),
    );
    if chips.len() == other.len() {
        assert_ok_or_err::<_, ()>(
            "signal::cross_correlation",
            Ok(signal::cross_correlation(&chips, &other)),
        );
        assert_ok_finite_or_err(
            "signal::correlation_at",
            signal::correlation_at(&chips, &other, input.ints[2]).map(f64::from),
        );
    }
    assert_ok_or_err(
        "signal::replica",
        signal::replica(
            prn,
            ReplicaOptions {
                sample_rate_hz: input.params[0],
                num_samples: bounded_usize(input.bits[1], 0, 256),
                code_phase_chips: input.params[1],
                code_doppler_hz: input.params[2],
            },
        ),
    );
    let iq = iq(&input);
    assert_ok_finite_or_err(
        "signal::correlate",
        signal::correlate(
            &iq,
            prn,
            CorrelateOptions {
                sample_rate_hz: input.params[0],
                doppler_hz: input.params[3],
                code_phase_chips: input.params[1],
                code_doppler_hz: input.params[2],
            },
        ),
    );
    assert_ok_finite_or_err(
        "signal::correlate_against",
        signal::correlate_against(&iq, &chips, input.params[0], input.params[3]),
    );
    assert_ok_finite_or_err(
        "signal::acquire",
        signal::acquire(
            &iq,
            prn,
            AcquisitionOptions {
                sample_rate_hz: input.params[0],
                doppler_min_hz: bounded_abs_or_raw(input.params[4], 5_000.0),
                doppler_max_hz: bounded_abs_or_raw(input.params[5], 5_000.0),
                doppler_step_hz: bounded_positive_or_raw(input.params[6], 1.0, 1_000.0),
            },
        ),
    );
    assert_ok_finite_or_err(
        "signal::coherent_loss",
        signal::coherent_loss(input.params[7], input.params[8]),
    );
    assert_ok_finite_or_err(
        "signal::coherent_loss_db",
        signal::coherent_loss_db(input.params[7], input.params[8]),
    );
    assert_ok_finite_or_err(
        "signal::snr_post_db",
        signal::snr_post_db(input.params[9], input.params[8]),
    );

    assert_ok_finite_or_err(
        "carrier_phase::phase_meters",
        carrier_phase::phase_meters(input.params[0], input.params[4]),
    );
    assert_ok_finite_or_err(
        "carrier_phase::geometry_free",
        carrier_phase::geometry_free(input.params[0], input.params[1]),
    );
    assert_ok_finite_or_err(
        "carrier_phase::wide_lane_wavelength",
        carrier_phase::wide_lane_wavelength(input.params[4], input.params[5]),
    );
    assert_ok_finite_or_err(
        "carrier_phase::narrow_lane_code",
        carrier_phase::narrow_lane_code(
            input.params[0],
            input.params[1],
            input.params[4],
            input.params[5],
        ),
    );
    assert_ok_finite_or_err(
        "carrier_phase::melbourne_wubbena",
        carrier_phase::melbourne_wubbena(
            input.params[0],
            input.params[1],
            input.params[2],
            input.params[3],
            input.params[4],
            input.params[5],
        ),
    );
    assert_ok_finite_or_err(
        "carrier_phase::code_minus_carrier",
        carrier_phase::code_minus_carrier(input.params[2], input.params[0], input.params[4]),
    );
    assert_ok_finite_or_err(
        "carrier_phase::wide_lane_cycles",
        carrier_phase::wide_lane_cycles(
            input.params[0],
            input.params[1],
            input.params[2],
            input.params[3],
            input.params[4],
            input.params[5],
        ),
    );
    let arc = arc(&input);
    assert_ok_or_err::<_, ()>(
        "carrier_phase::detect_cycle_slips",
        Ok(carrier_phase::detect_cycle_slips(
            &arc,
            CycleSlipOptions::default(),
        )),
    );
    assert_ok_or_err::<_, ()>(
        "carrier_phase::smooth_code",
        Ok(carrier_phase::smooth_code(
            &arc,
            CycleSlipOptions::default(),
            bounded_usize(input.bits[2], 1, 16),
        )),
    );
    assert_ok_or_err::<_, ()>(
        "carrier_phase::smooth_iono_free_code",
        Ok(carrier_phase::smooth_iono_free_code(
            &arc,
            CycleSlipOptions::default(),
            bounded_usize(input.bits[2], 1, 16),
        )),
    );

    assert_ok_finite_or_err(
        "combinations::frequency_hz",
        combinations::frequency_hz('G', "l1"),
    );
    assert_ok_finite_or_err(
        "combinations::default_pair",
        combinations::default_pair('G').map(|_| 0.0),
    );
    assert_ok_finite_or_err(
        "combinations::gamma",
        combinations::gamma(input.params[4], input.params[5]),
    );
    assert_ok_finite_or_err(
        "combinations::noise_amplification",
        combinations::noise_amplification(input.params[4], input.params[5]),
    );
    assert_ok_finite_or_err(
        "combinations::ionosphere_free",
        combinations::ionosphere_free(
            input.params[0],
            input.params[1],
            input.params[4],
            input.params[5],
        ),
    );
    assert_ok_finite_or_err(
        "combinations::ionosphere_free_phase_m",
        combinations::ionosphere_free_phase_m(
            input.params[0],
            input.params[1],
            input.params[4],
            input.params[5],
        ),
    );
    assert_ok_finite_or_err(
        "combinations::ionosphere_free_phase_cycles",
        combinations::ionosphere_free_phase_cycles(
            input.params[0],
            input.params[1],
            input.params[4],
            input.params[5],
        ),
    );
    let band1 = vec![("G01".to_string(), input.params[0])];
    let band2 = vec![("G01".to_string(), input.params[1])];
    if let Ok((combined, _)) = combinations::ionosphere_free_pseudoranges(&band1, &band2, &[]) {
        assert_success(
            "combinations::ionosphere_free_pseudoranges",
            combined
                .into_iter()
                .map(|(_, value)| value)
                .collect::<Vec<_>>(),
        );
    }

    let source = Source {
        positions: input.positions,
        velocities: input.velocities,
    };
    assert_ok_finite_or_err(
        "observables::j2000_seconds_from_split",
        observables::j2000_seconds_from_split(input.params[10], input.params[11]),
    );
    if let Ok(sat) = GnssSatelliteId::new(GnssSystem::Gps, 1) {
        assert_ok_finite_or_err(
            "observables::predict",
            observables::predict(
                &source,
                sat,
                input.receiver,
                input.params[12],
                PredictOptions {
                    carrier_hz: input.params[4],
                    light_time: input.bits[0] & 1 == 1,
                    sagnac: input.bits[1] & 1 == 1,
                },
            ),
        );
    }
    assert_ok_finite_or_err(
        "velocity::doppler_to_range_rate",
        velocity::doppler_to_range_rate(input.params[13], input.params[4]),
    );
    assert_ok_finite_or_err(
        "velocity::range_rate_to_doppler",
        velocity::range_rate_to_doppler(input.params[14], input.params[4]),
    );
});
