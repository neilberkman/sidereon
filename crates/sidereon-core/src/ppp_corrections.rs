//! Static-arc PPP correction precomputation.
//!
//! This module owns the language-independent correction algebra that used to sit
//! in Sidereon' `PPPCorrections` wrapper: per-epoch Sun/Moon and solid-earth tide
//! evaluation, per-satellite carrier-phase wind-up continuity, and satellite
//! antenna PCO/PCV projection in the satellite body frame.

use crate::astro::bodies::{sun_moon_ecef, SunMoon};
use crate::astro::math::vec3::{add3, cross3, dot3, neg3, norm3, scale3, sub3, unit3};
use crate::astro::time::{CoverageError, TimeScaleInputErrorKind, TimeScales, ValidityMode};
use crate::validate;
use std::collections::BTreeMap;
use std::f64::consts::PI;

use crate::antenna;
use crate::constants::{C_M_S, F_L1_HZ, OMEGA_E_DOT_RAD_S, RAD_TO_DEG};
use crate::ephemeris::Sp3;
use crate::observables::{
    predict, ObservablesError, ObservablesInputErrorKind, PredictOptions, PredictedObservables,
};
use crate::tides::{ocean_tide_loading, solid_earth_pole_tide, solid_earth_tide, TideError};

// The ocean-loading types live in `tides` (the displacement math owns them), but
// `PppCorrectionsOptions::ocean_loading` is the public entry point that consumes
// them. Re-export them here so a caller configuring PPP corrections can name and
// build the option's type — and size the BLQ block with `NUM_OCEAN_CONSTITUENTS`
// rather than a hardcoded `11` — without reaching into `tides`. The pole-tide
// option (`PoleTideOptions`) is defined in this module because it is a
// PPP-correction switch with no role in the tide math itself; this keeps the
// `PppCorrectionsOptions` surface coherent from one module.
pub use crate::tides::{OceanLoadingBlq, NUM_OCEAN_CONSTITUENTS};
use crate::tolerances::{FREQUENCY_DENOMINATOR_EPS_HZ, YAW_SINGULARITY_EPS_RAD};
use crate::GnssSatelliteId;

const TWO_PI: f64 = 2.0 * PI;

/// Civil date/time fields used by Sidereon PPP correction tables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CivilDateTime {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: f64,
}

/// One satellite observation row needed by the static correction precompute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PppCorrectionObservation {
    pub sat: GnssSatelliteId,
    pub freq1_hz: f64,
    pub freq2_hz: f64,
}

/// One receiver epoch and its visible satellite rows.
#[derive(Debug, Clone, PartialEq)]
pub struct PppCorrectionEpoch {
    pub epoch: CivilDateTime,
    pub t_rx_j2000_s: f64,
    pub observations: Vec<PppCorrectionObservation>,
}

/// Frequency-dependent satellite antenna calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct SatelliteAntennaFrequency {
    pub label: String,
    pub pco_m: [f64; 3],
    pub noazi_pcv_m: Vec<(f64, f64)>,
}

/// Satellite antenna block selected by PRN and validity window.
#[derive(Debug, Clone, PartialEq)]
pub struct SatelliteAntenna {
    pub sat: GnssSatelliteId,
    pub valid_from: Option<CivilDateTime>,
    pub valid_until: Option<CivilDateTime>,
    pub frequencies: Vec<SatelliteAntennaFrequency>,
}

/// Satellite antenna correction options.
#[derive(Debug, Clone, PartialEq)]
pub struct SatelliteAntennaOptions {
    pub freq1_label: String,
    pub freq1_hz: f64,
    pub freq2_label: String,
    pub freq2_hz: f64,
    pub antennas: Vec<SatelliteAntenna>,
}

/// Solid-Earth pole tide correction options.
///
/// The pole tide needs the epoch's IERS polar motion, which the engine's
/// embedded EOP table does not carry (it holds UT1-UTC only). The caller
/// supplies it in arcseconds, sourced from IERS EOP exactly like the other
/// Earth-orientation inputs. Polar motion drifts only a few mas/day, so a single
/// daily value is representative across a static arc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoleTideOptions {
    /// IERS polar motion x of the date (arcsec).
    pub xp_arcsec: f64,
    /// IERS polar motion y of the date (arcsec).
    pub yp_arcsec: f64,
}

/// PPP correction precompute switches.
#[derive(Debug, Clone, PartialEq)]
pub struct PppCorrectionsOptions {
    pub solid_earth_tide: bool,
    pub pole_tide: Option<PoleTideOptions>,
    /// Ocean tide loading: the station's BLQ coefficients. The engine does not
    /// embed ocean-loading models, so the caller supplies the per-station BLQ
    /// block (Bos-Scherneck / OSO Chalmers or equivalent), exactly like the
    /// polar-motion data dependency of the pole tide.
    pub ocean_loading: Option<OceanLoadingBlq>,
    pub phase_windup: bool,
    pub satellite_antenna: Option<SatelliteAntennaOptions>,
}

/// Indexed vector result. The epoch index refers to the input epoch slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochVectorCorrection {
    pub epoch_index: usize,
    pub vector_m: [f64; 3],
}

/// Indexed satellite scalar result. The epoch index refers to the input epoch slice.
#[derive(Debug, Clone, PartialEq)]
pub struct SatScalarCorrection {
    pub sat: GnssSatelliteId,
    pub epoch_index: usize,
    pub value_m: f64,
}

/// Indexed satellite vector result. The epoch index refers to the input epoch slice.
#[derive(Debug, Clone, PartialEq)]
pub struct SatVectorCorrection {
    pub sat: GnssSatelliteId,
    pub epoch_index: usize,
    pub vector_m: [f64; 3],
}

/// Precomputed PPP correction tables.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PppCorrections {
    pub tide: Vec<EpochVectorCorrection>,
    pub pole_tide: Vec<EpochVectorCorrection>,
    pub ocean_loading: Vec<EpochVectorCorrection>,
    pub windup_m: Vec<SatScalarCorrection>,
    pub sat_pco_ecef: Vec<SatVectorCorrection>,
    pub sat_pcv_m: Vec<SatScalarCorrection>,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PppCorrectionsError {
    #[error("invalid PPP correction input {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    #[error("invalid PPP correction epoch at epoch {epoch_index}: {source}")]
    Epoch {
        epoch_index: usize,
        #[source]
        source: CoverageError,
    },
    #[error("solid Earth tide correction failed at epoch {epoch_index}: {source}")]
    Tide {
        epoch_index: usize,
        #[source]
        source: TideError,
    },
    #[error("solid Earth pole tide correction failed at epoch {epoch_index}: {source}")]
    PoleTide {
        epoch_index: usize,
        #[source]
        source: TideError,
    },
    #[error("ocean tide loading correction failed at epoch {epoch_index}: {source}")]
    OceanLoading {
        epoch_index: usize,
        #[source]
        source: TideError,
    },
    #[error(
        "invalid phase wind-up carrier frequencies at epoch {epoch_index} for {sat}: {field} {reason}"
    )]
    WindupFrequency {
        epoch_index: usize,
        sat: GnssSatelliteId,
        field: &'static str,
        reason: &'static str,
    },
    #[error("invalid satellite antenna carrier frequencies: {field} {reason}")]
    SatelliteAntennaFrequency {
        field: &'static str,
        reason: &'static str,
    },
}

/// Build static PPP correction tables for a precise-orbit arc.
pub fn build(
    sp3: &Sp3,
    epochs: &[PppCorrectionEpoch],
    receiver_ecef_m: [f64; 3],
    options: &PppCorrectionsOptions,
) -> Result<PppCorrections, PppCorrectionsError> {
    validate_receiver_state(receiver_ecef_m)?;

    let mut corrections = PppCorrections::default();
    if !options.solid_earth_tide
        && options.pole_tide.is_none()
        && options.ocean_loading.is_none()
        && !options.phase_windup
        && options.satellite_antenna.is_none()
    {
        return Ok(corrections);
    }

    let satellite_antenna_frequencies = options
        .satellite_antenna
        .as_ref()
        .map(validate_satellite_antenna_options)
        .transpose()?;

    let mut previous_windup_cycles: BTreeMap<GnssSatelliteId, f64> = BTreeMap::new();

    // Sun/Moon is needed only by the solid-earth tide and the satellite-yaw
    // corrections (phase wind-up + satellite antenna). Pole tide and ocean
    // loading are pure station displacements, so a pole/ocean-only config must
    // not be coupled to the Sun/Moon (and the EOP/SP3 time paths behind them).
    let need_sun_moon =
        options.solid_earth_tide || options.phase_windup || options.satellite_antenna.is_some();
    // The per-observation predict() loop only feeds the wind-up and satellite
    // antenna corrections; skip it entirely when neither is enabled.
    let need_obs_loop = options.phase_windup || options.satellite_antenna.is_some();

    for (epoch_index, epoch_row) in epochs.iter().enumerate() {
        let sun_moon = if need_sun_moon {
            Some(
                sun_moon_at(epoch_row.epoch).map_err(|source| PppCorrectionsError::Epoch {
                    epoch_index,
                    source,
                })?,
            )
        } else {
            None
        };

        if options.solid_earth_tide {
            let sun_moon = sun_moon.expect("Sun/Moon computed when solid-earth tide is enabled");
            let d = tide_at(
                receiver_ecef_m,
                epoch_row.epoch,
                sun_moon.sun,
                sun_moon.moon,
            )
            .map_err(|source| PppCorrectionsError::Tide {
                epoch_index,
                source,
            })?;
            corrections.tide.push(EpochVectorCorrection {
                epoch_index,
                vector_m: d,
            });
        }

        if let Some(pole) = options.pole_tide {
            let d = pole_tide_at(receiver_ecef_m, epoch_row.epoch, pole).map_err(|source| {
                PppCorrectionsError::PoleTide {
                    epoch_index,
                    source,
                }
            })?;
            corrections.pole_tide.push(EpochVectorCorrection {
                epoch_index,
                vector_m: d,
            });
        }

        if let Some(blq) = options.ocean_loading.as_ref() {
            let d = ocean_loading_at(receiver_ecef_m, epoch_row.epoch, blq).map_err(|source| {
                PppCorrectionsError::OceanLoading {
                    epoch_index,
                    source,
                }
            })?;
            corrections.ocean_loading.push(EpochVectorCorrection {
                epoch_index,
                vector_m: d,
            });
        }

        if !need_obs_loop {
            continue;
        }
        let sun_moon = sun_moon.expect("Sun/Moon computed when the observation loop runs");

        for observation in &epoch_row.observations {
            let obs = match predict(
                sp3,
                observation.sat,
                receiver_ecef_m,
                epoch_row.t_rx_j2000_s,
                PredictOptions {
                    carrier_hz: F_L1_HZ,
                    light_time: true,
                    sagnac: true,
                },
            ) {
                Ok(obs) => obs,
                Err(ObservablesError::InvalidInput { field, kind }) => {
                    return Err(PppCorrectionsError::InvalidInput {
                        field,
                        reason: observables_input_reason(kind),
                    });
                }
                Err(ObservablesError::NoEphemeris | ObservablesError::Ephemeris(_)) => continue,
            };

            if options.phase_windup {
                let prev = previous_windup_cycles.get(&observation.sat).copied();
                if let Some(phw) = windup_cycles(&obs, receiver_ecef_m, sun_moon.sun, prev) {
                    let (f1, f2) = windup_frequency_pair(options, observation, epoch_index)?;
                    corrections.windup_m.push(SatScalarCorrection {
                        sat: observation.sat,
                        epoch_index,
                        value_m: windup_metres(phw, f1, f2),
                    });
                    previous_windup_cycles.insert(observation.sat, phw);
                }
            }

            if let Some(sat_ant) = &options.satellite_antenna {
                if let Some((pco_ecef, pcv_m)) = satellite_antenna_correction(
                    &obs,
                    sun_moon.sun,
                    observation.sat,
                    epoch_row.epoch,
                    sat_ant,
                    satellite_antenna_frequencies
                        .expect("satellite antenna frequencies are validated when enabled"),
                ) {
                    corrections.sat_pco_ecef.push(SatVectorCorrection {
                        sat: observation.sat,
                        epoch_index,
                        vector_m: pco_ecef,
                    });
                    corrections.sat_pcv_m.push(SatScalarCorrection {
                        sat: observation.sat,
                        epoch_index,
                        value_m: pcv_m,
                    });
                }
            }
        }
    }

    Ok(corrections)
}

fn validate_receiver_state(receiver_ecef_m: [f64; 3]) -> Result<(), PppCorrectionsError> {
    validate::finite_vec3(receiver_ecef_m, "receiver_ecef_m").map_err(ppp_invalid_input)?;
    validate::finite_positive(norm3(receiver_ecef_m), "receiver radius_m")
        .map_err(ppp_invalid_input)?;
    Ok(())
}

fn ppp_invalid_input(error: validate::FieldError) -> PppCorrectionsError {
    PppCorrectionsError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

fn observables_input_reason(kind: ObservablesInputErrorKind) -> &'static str {
    match kind {
        ObservablesInputErrorKind::NonFinite => "not finite",
        ObservablesInputErrorKind::NotPositive => "not positive",
        ObservablesInputErrorKind::Negative => "negative",
        ObservablesInputErrorKind::OutOfRange => "out of range",
        ObservablesInputErrorKind::Missing => "missing",
        ObservablesInputErrorKind::FloatParse => "invalid float",
        ObservablesInputErrorKind::IntParse => "invalid integer",
        ObservablesInputErrorKind::InvalidCivilDate => "invalid civil date",
        ObservablesInputErrorKind::InvalidCivilTime => "invalid civil time",
    }
}

fn windup_frequency_pair(
    options: &PppCorrectionsOptions,
    observation: &PppCorrectionObservation,
    epoch_index: usize,
) -> Result<(f64, f64), PppCorrectionsError> {
    let (f1_hz, f2_hz) = options
        .satellite_antenna
        .as_ref()
        .map(|a| (a.freq1_hz, a.freq2_hz))
        .unwrap_or((observation.freq1_hz, observation.freq2_hz));
    validate_frequency_pair(
        f1_hz,
        f2_hz,
        FrequencyPairFields {
            freq1: "phase wind-up freq1_hz",
            freq2: "phase wind-up freq2_hz",
            pair: "phase wind-up frequency pair",
        },
        |field, reason| PppCorrectionsError::WindupFrequency {
            epoch_index,
            sat: observation.sat,
            field,
            reason,
        },
    )
}

fn validate_satellite_antenna_frequency_pair(
    options: &SatelliteAntennaOptions,
) -> Result<(f64, f64), PppCorrectionsError> {
    validate_frequency_pair(
        options.freq1_hz,
        options.freq2_hz,
        FrequencyPairFields {
            freq1: "satellite antenna freq1_hz",
            freq2: "satellite antenna freq2_hz",
            pair: "satellite antenna frequency pair",
        },
        |field, reason| PppCorrectionsError::SatelliteAntennaFrequency { field, reason },
    )
}

fn validate_satellite_antenna_options(
    options: &SatelliteAntennaOptions,
) -> Result<(f64, f64), PppCorrectionsError> {
    let frequencies_hz = validate_satellite_antenna_frequency_pair(options)?;
    validate_satellite_antenna_pcv_samples(options)?;
    Ok(frequencies_hz)
}

fn validate_satellite_antenna_pcv_samples(
    options: &SatelliteAntennaOptions,
) -> Result<(), PppCorrectionsError> {
    for antenna in &options.antennas {
        for frequency in &antenna.frequencies {
            for &(nadir_deg, pcv_m) in &frequency.noazi_pcv_m {
                validate::finite(nadir_deg, "satellite antenna noazi_pcv_m")
                    .map_err(ppp_invalid_input)?;
                validate::finite(pcv_m, "satellite antenna noazi_pcv_m")
                    .map_err(ppp_invalid_input)?;
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct FrequencyPairFields {
    freq1: &'static str,
    freq2: &'static str,
    pair: &'static str,
}

fn validate_frequency_pair(
    f1_hz: f64,
    f2_hz: f64,
    fields: FrequencyPairFields,
    invalid: impl Fn(&'static str, &'static str) -> PppCorrectionsError,
) -> Result<(f64, f64), PppCorrectionsError> {
    let f1_hz = validate::finite_positive(f1_hz, fields.freq1)
        .map_err(|e| invalid(e.field(), e.reason()))?;
    let f2_hz = validate::finite_positive(f2_hz, fields.freq2)
        .map_err(|e| invalid(e.field(), e.reason()))?;
    if (f1_hz - f2_hz).abs() < FREQUENCY_DENOMINATOR_EPS_HZ {
        Err(invalid(fields.pair, "must differ"))
    } else {
        Ok((f1_hz, f2_hz))
    }
}

fn sun_moon_at(epoch: CivilDateTime) -> Result<SunMoon, CoverageError> {
    let ts = time_scales_at(epoch)?;
    Ok(sun_moon_ecef(&ts).expect("validated time scales produce Sun/Moon vectors"))
}

fn time_scales_at(epoch: CivilDateTime) -> Result<TimeScales, CoverageError> {
    let civil = validate::civil_datetime_with_second_policy(
        i64::from(epoch.year),
        i64::from(epoch.month),
        i64::from(epoch.day),
        i64::from(epoch.hour),
        i64::from(epoch.minute),
        epoch.second,
        validate::CivilSecondPolicy::UtcLike,
    )
    .map_err(|error| CoverageError::InvalidInput {
        field: error.field(),
        kind: TimeScaleInputErrorKind::from(&error),
    })?;

    TimeScales::from_utc_validated(
        civil.year as i32,
        civil.month as i32,
        civil.day as i32,
        civil.hour as i32,
        civil.minute as i32,
        civil.second,
        ValidityMode::Strict,
    )
    .map(|validated| validated.value)
}

fn tide_at(
    receiver_ecef_m: [f64; 3],
    epoch: CivilDateTime,
    sun_ecef_m: [f64; 3],
    moon_ecef_m: [f64; 3],
) -> Result<[f64; 3], TideError> {
    let fhr = epoch.hour as f64 + epoch.minute as f64 / 60.0 + epoch.second / 3600.0;
    solid_earth_tide(
        &receiver_ecef_m,
        epoch.year,
        epoch.month as i32,
        epoch.day as i32,
        fhr,
        &sun_ecef_m,
        &moon_ecef_m,
    )
}

fn pole_tide_at(
    receiver_ecef_m: [f64; 3],
    epoch: CivilDateTime,
    pole: PoleTideOptions,
) -> Result<[f64; 3], TideError> {
    let fhr = epoch.hour as f64 + epoch.minute as f64 / 60.0 + epoch.second / 3600.0;
    solid_earth_pole_tide(
        &receiver_ecef_m,
        epoch.year,
        epoch.month as i32,
        epoch.day as i32,
        fhr,
        pole.xp_arcsec,
        pole.yp_arcsec,
    )
}

fn ocean_loading_at(
    receiver_ecef_m: [f64; 3],
    epoch: CivilDateTime,
    blq: &OceanLoadingBlq,
) -> Result<[f64; 3], TideError> {
    let fhr = epoch.hour as f64 + epoch.minute as f64 / 60.0 + epoch.second / 3600.0;
    ocean_tide_loading(
        &receiver_ecef_m,
        epoch.year,
        epoch.month as i32,
        epoch.day as i32,
        fhr,
        blq,
    )
}

fn windup_metres(phw_cycles: f64, f1_hz: f64, f2_hz: f64) -> f64 {
    let lam1 = C_M_S / f1_hz;
    let lam2 = C_M_S / f2_hz;
    let gamma = ionosphere_free_gamma(f1_hz, f2_hz);
    (gamma * lam1 - (gamma - 1.0) * lam2) * phw_cycles
}

fn windup_cycles(
    pred: &PredictedObservables,
    receiver_ecef_m: [f64; 3],
    sun_ecef_m: [f64; 3],
    prev_phw: Option<f64>,
) -> Option<f64> {
    let rs = pred.sat_pos_ecef_m;
    let vs = pred.sat_velocity_m_s;
    let (exs, eys) = sat_yaw(rs, vs, sun_ecef_m)?;
    let ek = unit3(sub3(receiver_ecef_m, rs))?;

    let (n, e, _u) = crate::estimation::substrate::frames::local_neu_basis(
        crate::estimation::recipe::FrameRecipe::GeodeticNeuCrossProduct,
        receiver_ecef_m,
    );
    let exr = n;
    let eyr = neg3(e);

    let eks = cross3(ek, eys);
    let ekr = cross3(ek, eyr);
    let ds = sub3(exs, add3(scale3(ek, dot3(ek, exs)), eks));
    let dr = sub3(exr, sub3(scale3(ek, dot3(ek, exr)), ekr));

    let nds = norm3(ds);
    let ndr = norm3(dr);
    if nds == 0.0 || ndr == 0.0 {
        return None;
    }

    let cosp = clamp(dot3(ds, dr) / nds / ndr);
    let mut ph = cosp.acos() / TWO_PI;
    let drs = cross3(ds, dr);
    if dot3(ek, drs) < 0.0 {
        ph = -ph;
    }

    Some(match prev_phw {
        None => ph,
        Some(prev) => ph + (prev - ph + 0.5).floor(),
    })
}

fn sat_yaw(rs: [f64; 3], vs: [f64; 3], sun_ecef_m: [f64; 3]) -> Option<([f64; 3], [f64; 3])> {
    let ri_v = [
        vs[0] - OMEGA_E_DOT_RAD_S * rs[1],
        vs[1] + OMEGA_E_DOT_RAD_S * rs[0],
        vs[2],
    ];
    let n = cross3(rs, ri_v);
    let p = cross3(sun_ecef_m, n);

    let es = unit3(rs)?;
    let esun = unit3(sun_ecef_m)?;
    let en = unit3(n)?;
    let ep = unit3(p)?;

    let beta = PI / 2.0 - clamp(dot3(esun, en)).acos();
    let ee = clamp(dot3(es, ep)).acos();
    let mut mu = PI / 2.0 + if dot3(es, esun) <= 0.0 { -ee } else { ee };

    if mu < -PI / 2.0 {
        mu += TWO_PI;
    } else if mu >= PI / 2.0 {
        mu -= TWO_PI;
    }

    let yaw = yaw_nominal(beta, mu);
    let ex = cross3(en, es);
    let cosy = yaw.cos();
    let siny = yaw.sin();
    let exs = add3(scale3(en, -siny), scale3(ex, cosy));
    let eys = add3(scale3(en, -cosy), scale3(ex, -siny));
    Some((exs, eys))
}

fn yaw_nominal(beta: f64, mu: f64) -> f64 {
    if beta.abs() < YAW_SINGULARITY_EPS_RAD && mu.abs() < YAW_SINGULARITY_EPS_RAD {
        PI
    } else {
        (-beta.tan()).atan2(mu.sin()) + PI
    }
}

fn satellite_antenna_correction(
    pred: &PredictedObservables,
    sun_ecef_m: [f64; 3],
    sat: GnssSatelliteId,
    epoch: CivilDateTime,
    options: &SatelliteAntennaOptions,
    frequencies_hz: (f64, f64),
) -> Option<([f64; 3], f64)> {
    let rs = pred.sat_pos_ecef_m;
    let ant = options.antenna_for(sat, epoch)?;

    let ez = unit3(neg3(rs))?;
    let es = unit3(sub3(sun_ecef_m, rs))?;
    let ey = unit3(cross3(ez, es))?;
    let ex = cross3(ey, ez);

    let off1 = ant.pco(&options.freq1_label)?;
    let off2 = ant.pco(&options.freq2_label)?;
    let gamma = ionosphere_free_gamma(frequencies_hz.0, frequencies_hz.1);

    let dant1 = body_to_ecef(off1, ex, ey, ez);
    let dant2 = body_to_ecef(off2, ex, ey, ez);
    let dant_ecef = sub3(scale3(dant1, gamma), scale3(dant2, gamma - 1.0));
    let pcv_m = nadir_pcv_if(ant, pred, options, gamma)?;

    Some((dant_ecef, pcv_m))
}

fn body_to_ecef(pco_body_m: [f64; 3], ex: [f64; 3], ey: [f64; 3], ez: [f64; 3]) -> [f64; 3] {
    add3(
        add3(scale3(ex, pco_body_m[0]), scale3(ey, pco_body_m[1])),
        scale3(ez, pco_body_m[2]),
    )
}

fn ionosphere_free_gamma(f1_hz: f64, f2_hz: f64) -> f64 {
    let f1_sq = f1_hz * f1_hz;
    f1_sq / (f1_sq - f2_hz * f2_hz)
}

fn nadir_pcv_if(
    ant: &SatelliteAntenna,
    pred: &PredictedObservables,
    options: &SatelliteAntennaOptions,
    gamma: f64,
) -> Option<f64> {
    let eu = unit3(neg3(pred.los_unit))?;
    let ez = unit3(neg3(pred.sat_pos_ecef_m))?;
    let nadir_deg = clamp(dot3(eu, ez)).acos() * RAD_TO_DEG;
    let p1 = ant.pcv_noazi(&options.freq1_label, nadir_deg)?;
    let p2 = ant.pcv_noazi(&options.freq2_label, nadir_deg)?;
    Some(gamma * p1 - (gamma - 1.0) * p2)
}

impl SatelliteAntennaOptions {
    fn antenna_for(&self, sat: GnssSatelliteId, epoch: CivilDateTime) -> Option<&SatelliteAntenna> {
        self.antennas
            .iter()
            .find(|ant| ant.sat == sat && ant.valid_at(epoch))
    }
}

impl SatelliteAntenna {
    fn valid_at(&self, epoch: CivilDateTime) -> bool {
        let after_from = self
            .valid_from
            .is_none_or(|from| civil_cmp(epoch, from) != std::cmp::Ordering::Less);
        let before_until = self
            .valid_until
            .is_none_or(|until| civil_cmp(epoch, until) != std::cmp::Ordering::Greater);
        after_from && before_until
    }

    fn frequency(&self, label: &str) -> Option<&SatelliteAntennaFrequency> {
        self.frequencies
            .iter()
            .find(|f| f.label.trim() == label.trim())
    }

    fn pco(&self, label: &str) -> Option<[f64; 3]> {
        self.frequency(label).map(|f| f.pco_m)
    }

    fn pcv_noazi(&self, label: &str, zenith_deg: f64) -> Option<f64> {
        let frequency = self.frequency(label)?;
        interpolate_samples(&frequency.noazi_pcv_m, zenith_deg)
    }
}

fn civil_cmp(a: CivilDateTime, b: CivilDateTime) -> std::cmp::Ordering {
    (
        a.year,
        a.month,
        a.day,
        a.hour,
        a.minute,
        ordered_seconds(a.second),
    )
        .cmp(&(
            b.year,
            b.month,
            b.day,
            b.hour,
            b.minute,
            ordered_seconds(b.second),
        ))
}

fn ordered_seconds(second: f64) -> i64 {
    (second * 1_000_000.0).round() as i64
}

fn interpolate_samples(samples: &[(f64, f64)], zenith_deg: f64) -> Option<f64> {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    antenna::interpolate_zenith_sorted(&sorted, zenith_deg)
}

fn clamp(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests {
    use super::*;
    use crate::astro::time::split_julian_date;
    use crate::constants::F_L2_HZ;
    use crate::observables::j2000_seconds_from_split;
    use crate::GnssSystem;

    fn sp3_fixture() -> Sp3 {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read SP3 fixture {path}: {e}"));
        Sp3::parse(&bytes).expect("parse SP3 fixture")
    }

    fn civil(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: f64) -> CivilDateTime {
        CivilDateTime {
            year,
            month,
            day,
            hour,
            minute,
            second,
        }
    }

    fn split_jd(epoch: CivilDateTime) -> (f64, f64) {
        split_julian_date(
            epoch.year,
            i32::from(epoch.month),
            i32::from(epoch.day),
            i32::from(epoch.hour),
            i32::from(epoch.minute),
            epoch.second,
        )
    }

    fn fake_antenna_options(sat: GnssSatelliteId) -> SatelliteAntennaOptions {
        SatelliteAntennaOptions {
            freq1_label: "G01".to_string(),
            freq1_hz: F_L1_HZ,
            freq2_label: "G02".to_string(),
            freq2_hz: F_L2_HZ,
            antennas: vec![SatelliteAntenna {
                sat,
                valid_from: Some(civil(2020, 1, 1, 0, 0, 0.0)),
                valid_until: Some(civil(2021, 1, 1, 0, 0, 0.0)),
                frequencies: vec![
                    SatelliteAntennaFrequency {
                        label: "G01".to_string(),
                        pco_m: [0.1, -0.2, 1.0],
                        noazi_pcv_m: vec![(0.0, 0.001), (5.0, 0.002), (10.0, 0.004)],
                    },
                    SatelliteAntennaFrequency {
                        label: "G02".to_string(),
                        pco_m: [-0.1, 0.3, 0.5],
                        noazi_pcv_m: vec![(0.0, -0.001), (5.0, -0.002), (10.0, -0.003)],
                    },
                ],
            }],
        }
    }

    fn windup_epoch(sat: GnssSatelliteId, freq1_hz: f64, freq2_hz: f64) -> PppCorrectionEpoch {
        let epoch = civil(2020, 6, 24, 12, 0, 0.0);
        let (jd_whole, jd_fraction) = split_jd(epoch);
        PppCorrectionEpoch {
            epoch,
            t_rx_j2000_s: j2000_seconds_from_split(jd_whole, jd_fraction)
                .expect("valid split Julian date"),
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz,
                freq2_hz,
            }],
        }
    }

    #[test]
    fn ppp_corrections_match_elixir_reference_fixture() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let epoch = civil(2020, 6, 24, 12, 0, 0.0);
        let (jd_whole, jd_fraction) = split_jd(epoch);
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let epochs = vec![PppCorrectionEpoch {
            epoch,
            t_rx_j2000_s: j2000_seconds_from_split(jd_whole, jd_fraction)
                .expect("valid split Julian date"),
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz: F_L1_HZ,
                freq2_hz: F_L2_HZ,
            }],
        }];
        let options = PppCorrectionsOptions {
            solid_earth_tide: true,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: true,
            satellite_antenna: Some(fake_antenna_options(sat)),
        };

        let got = build(&sp3, &epochs, receiver, &options).expect("valid PPP corrections");

        assert_eq!(got.tide.len(), 1);
        assert_eq!(
            got.tide[0].vector_m.map(f64::to_bits),
            [0x3FB8BC98E788ED00, 0x3FAA54D8C1097508, 0x3FB03498C46B3B50]
        );
        assert_eq!(got.windup_m.len(), 1);
        assert_eq!(got.windup_m[0].value_m.to_bits(), 0xBF808DE79DBD2C16);
        assert_eq!(got.sat_pco_ecef.len(), 1);
        assert_eq!(
            got.sat_pco_ecef[0].vector_m.map(f64::to_bits),
            [0xBFE58ED947570048, 0x3FDEDBB280CEB1BE, 0xBFFE3BCA6A354E4A]
        );
        assert_eq!(got.sat_pcv_m.len(), 1);
        assert_eq!(got.sat_pcv_m[0].value_m.to_bits(), 0x3F77617E95BD232C);
    }

    #[test]
    fn pole_tide_correction_is_emitted_and_matches_standalone() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let epoch = civil(2020, 6, 24, 12, 0, 0.0);
        let (jd_whole, jd_fraction) = split_jd(epoch);
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let epochs = vec![PppCorrectionEpoch {
            epoch,
            t_rx_j2000_s: j2000_seconds_from_split(jd_whole, jd_fraction)
                .expect("valid split Julian date"),
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz: F_L1_HZ,
                freq2_hz: F_L2_HZ,
            }],
        }];
        let pole = PoleTideOptions {
            xp_arcsec: 0.169_051,
            yp_arcsec: 0.411_760,
        };
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: Some(pole),
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
        };

        let got = build(&sp3, &epochs, receiver, &options).expect("valid PPP corrections");

        assert_eq!(got.pole_tide.len(), 1);
        assert_eq!(got.pole_tide[0].epoch_index, 0);
        let expected = crate::tides::solid_earth_pole_tide(
            &receiver,
            2020,
            6,
            24,
            12.0,
            pole.xp_arcsec,
            pole.yp_arcsec,
        )
        .expect("valid pole tide");
        assert_eq!(got.pole_tide[0].vector_m, expected);
        // Pole tide is opt-in and independent of the solid-earth tide.
        assert!(got.tide.is_empty());
    }

    // ZIM2 ocean-loading BLQ (GOT4.7), OLFG/Scherneck Onsala 2020-Jun-25,
    // holt.oso.chalmers.se; used here purely as a finite, real-valued BLQ to
    // exercise the precompute plumbing (the receiver below is not ZIM2).
    fn zim2_blq() -> OceanLoadingBlq {
        OceanLoadingBlq {
            amplitude_m: [
                [
                    0.00693, 0.00228, 0.00148, 0.00061, 0.00220, 0.00094, 0.00070, 0.00001,
                    0.00047, 0.00025, 0.00019,
                ],
                [
                    0.00272, 0.00076, 0.00061, 0.00020, 0.00036, 0.00025, 0.00011, 0.00005,
                    0.00004, 0.00001, 0.00002,
                ],
                [
                    0.00061, 0.00026, 0.00010, 0.00009, 0.00025, 0.00002, 0.00008, 0.00003,
                    0.00002, 0.00000, 0.00001,
                ],
            ],
            phase_deg: [
                [
                    -72.3, -44.2, -90.8, -44.1, -62.9, -94.5, -64.3, 171.0, 3.4, 3.6, 1.1,
                ],
                [
                    84.3, 115.4, 63.3, 113.7, 98.6, 20.7, 94.2, -44.5, -170.0, -162.7, -177.8,
                ],
                [
                    -29.3, 1.7, -44.0, -4.2, 44.2, -39.1, 43.7, 170.1, -93.3, -118.3, -176.4,
                ],
            ],
        }
    }

    #[test]
    fn ocean_loading_correction_is_emitted_and_matches_standalone() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let epoch = civil(2020, 6, 24, 12, 0, 0.0);
        let (jd_whole, jd_fraction) = split_jd(epoch);
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let epochs = vec![PppCorrectionEpoch {
            epoch,
            t_rx_j2000_s: j2000_seconds_from_split(jd_whole, jd_fraction)
                .expect("valid split Julian date"),
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz: F_L1_HZ,
                freq2_hz: F_L2_HZ,
            }],
        }];
        let blq = zim2_blq();
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: Some(blq),
            phase_windup: false,
            satellite_antenna: None,
        };

        let got = build(&sp3, &epochs, receiver, &options).expect("valid PPP corrections");

        assert_eq!(got.ocean_loading.len(), 1);
        assert_eq!(got.ocean_loading[0].epoch_index, 0);
        let expected = crate::tides::ocean_tide_loading(&receiver, 2020, 6, 24, 12.0, &blq)
            .expect("valid ocean loading");
        assert_eq!(got.ocean_loading[0].vector_m, expected);
        // Ocean loading is opt-in and independent of the other corrections.
        assert!(got.tide.is_empty());
        assert!(got.pole_tide.is_empty());
    }

    #[test]
    fn pole_or_ocean_only_skips_sun_moon_and_prediction() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];

        // An epoch crafted so the Sun/Moon and the per-observation predict path
        // would BOTH fail if they ran: the date is past the embedded EOP
        // coverage (sun_moon_at -> Epoch::OutsideCoverage) and t_rx is non-finite
        // (predict -> InvalidInput). Pole tide and ocean loading are pure station
        // displacements needing neither, so a pole/ocean-only build must still
        // succeed. Their only date requirement is a valid civil date, which
        // 2100-01-01 satisfies.
        let epochs = vec![PppCorrectionEpoch {
            epoch: civil(2100, 1, 1, 12, 0, 0.0),
            t_rx_j2000_s: f64::NAN,
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz: F_L1_HZ,
                freq2_hz: F_L2_HZ,
            }],
        }];

        // Pole tide only.
        let pole = PoleTideOptions {
            xp_arcsec: 0.169_051,
            yp_arcsec: 0.411_760,
        };
        let got = build(
            &sp3,
            &epochs,
            receiver,
            &PppCorrectionsOptions {
                solid_earth_tide: false,
                pole_tide: Some(pole),
                ocean_loading: None,
                phase_windup: false,
                satellite_antenna: None,
            },
        )
        .expect("pole-only build must not touch the Sun/Moon or predict paths");
        assert_eq!(got.pole_tide.len(), 1);
        assert!(got.tide.is_empty());
        assert!(got.ocean_loading.is_empty());
        assert!(got.windup_m.is_empty());
        assert!(got.sat_pco_ecef.is_empty());
        assert!(got.sat_pcv_m.is_empty());

        // Ocean loading only.
        let blq = zim2_blq();
        let got = build(
            &sp3,
            &epochs,
            receiver,
            &PppCorrectionsOptions {
                solid_earth_tide: false,
                pole_tide: None,
                ocean_loading: Some(blq),
                phase_windup: false,
                satellite_antenna: None,
            },
        )
        .expect("ocean-only build must not touch the Sun/Moon or predict paths");
        assert_eq!(got.ocean_loading.len(), 1);
        assert!(got.tide.is_empty());
        assert!(got.pole_tide.is_empty());
        assert!(got.windup_m.is_empty());
        assert!(got.sat_pco_ecef.is_empty());
        assert!(got.sat_pcv_m.is_empty());
    }

    #[test]
    fn phase_windup_rejects_invalid_observation_frequency_pairs() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: true,
            satellite_antenna: None,
        };
        let cases = [
            (0.0, F_L2_HZ, "phase wind-up freq1_hz", "not positive"),
            (-F_L1_HZ, F_L2_HZ, "phase wind-up freq1_hz", "not positive"),
            (
                F_L1_HZ,
                F_L1_HZ,
                "phase wind-up frequency pair",
                "must differ",
            ),
        ];

        for (freq1_hz, freq2_hz, field, reason) in cases {
            let epochs = vec![windup_epoch(sat, freq1_hz, freq2_hz)];
            let err = build(&sp3, &epochs, receiver, &options)
                .expect_err("invalid phase wind-up frequencies must error");

            assert_eq!(
                err,
                PppCorrectionsError::WindupFrequency {
                    epoch_index: 0,
                    sat,
                    field,
                    reason,
                }
            );
        }
    }

    #[test]
    fn phase_windup_observation_frequency_pair_computes_finite_correction() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: true,
            satellite_antenna: None,
        };
        let epochs = vec![windup_epoch(sat, F_L1_HZ, F_L2_HZ)];

        let got =
            build(&sp3, &epochs, receiver, &options).expect("valid phase wind-up frequencies");

        assert_eq!(got.windup_m.len(), 1);
        assert!(got.windup_m[0].value_m.is_finite());
    }

    #[test]
    fn satellite_antenna_rejects_invalid_frequency_pairs_without_windup() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let cases = [
            (0.0, F_L2_HZ, "satellite antenna freq1_hz", "not positive"),
            (
                F_L1_HZ,
                f64::INFINITY,
                "satellite antenna freq2_hz",
                "not finite",
            ),
            (
                f64::NAN,
                F_L2_HZ,
                "satellite antenna freq1_hz",
                "not finite",
            ),
            (
                F_L1_HZ,
                F_L1_HZ,
                "satellite antenna frequency pair",
                "must differ",
            ),
        ];

        for (freq1_hz, freq2_hz, field, reason) in cases {
            let mut antenna = fake_antenna_options(sat);
            antenna.freq1_hz = freq1_hz;
            antenna.freq2_hz = freq2_hz;
            let options = PppCorrectionsOptions {
                solid_earth_tide: false,
                pole_tide: None,
                ocean_loading: None,
                phase_windup: false,
                satellite_antenna: Some(antenna),
            };
            let epochs = vec![windup_epoch(sat, F_L1_HZ, F_L2_HZ)];

            let err = build(&sp3, &epochs, receiver, &options)
                .expect_err("invalid satellite antenna frequencies must error");

            assert_eq!(
                err,
                PppCorrectionsError::SatelliteAntennaFrequency { field, reason }
            );
        }
    }

    #[test]
    fn satellite_antenna_frequency_pair_computes_finite_corrections_without_windup() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: Some(fake_antenna_options(sat)),
        };
        let epochs = vec![windup_epoch(sat, F_L1_HZ, F_L2_HZ)];

        let got =
            build(&sp3, &epochs, receiver, &options).expect("valid satellite antenna frequencies");

        assert!(got.windup_m.is_empty());
        assert_eq!(got.sat_pco_ecef.len(), 1);
        assert!(got.sat_pco_ecef[0]
            .vector_m
            .iter()
            .all(|value| value.is_finite()));
        assert_eq!(got.sat_pcv_m.len(), 1);
        assert!(got.sat_pcv_m[0].value_m.is_finite());
    }

    #[test]
    fn satellite_antenna_rejects_non_finite_pcv_samples() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let mut antenna = fake_antenna_options(sat);
        antenna.antennas[0].frequencies[0].noazi_pcv_m[1] = (5.0, f64::NAN);
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: Some(antenna),
        };
        let epochs = vec![windup_epoch(sat, F_L1_HZ, F_L2_HZ)];

        let err = build(&sp3, &epochs, receiver, &options)
            .expect_err("non-finite satellite PCV samples must error");

        assert_eq!(
            err,
            PppCorrectionsError::InvalidInput {
                field: "satellite antenna noazi_pcv_m",
                reason: "not finite",
            }
        );
    }

    #[test]
    fn satellite_antenna_empty_pcv_grid_is_not_materialized_as_zero() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let epoch = civil(2020, 6, 24, 12, 0, 0.0);
        let (jd_whole, jd_fraction) = split_jd(epoch);
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let epochs = vec![PppCorrectionEpoch {
            epoch,
            t_rx_j2000_s: j2000_seconds_from_split(jd_whole, jd_fraction)
                .expect("valid split Julian date"),
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz: F_L1_HZ,
                freq2_hz: F_L2_HZ,
            }],
        }];
        let mut antenna = fake_antenna_options(sat);
        antenna.antennas[0].frequencies[0].noazi_pcv_m.clear();
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: Some(antenna),
        };

        let got = build(&sp3, &epochs, receiver, &options).expect("valid PPP corrections");

        assert!(got.sat_pco_ecef.is_empty());
        assert!(got.sat_pcv_m.is_empty());
    }

    #[test]
    fn build_rejects_non_finite_receive_time_for_satellite_corrections() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: true,
            satellite_antenna: None,
        };
        let epochs = vec![PppCorrectionEpoch {
            epoch: civil(2020, 6, 24, 12, 0, 0.0),
            t_rx_j2000_s: f64::NAN,
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz: F_L1_HZ,
                freq2_hz: F_L2_HZ,
            }],
        }];

        let err = build(
            &sp3,
            &epochs,
            [3_512_900.0, 780_500.0, 5_248_700.0],
            &options,
        )
        .expect_err("non-finite receive time must be reported");

        assert_eq!(
            err,
            PppCorrectionsError::InvalidInput {
                field: "t_rx_j2000_s",
                reason: "not finite",
            }
        );
    }

    #[test]
    fn noazi_pcv_interpolation_clamps_and_interpolates() {
        let samples = vec![(10.0, 4.0), (0.0, 1.0), (5.0, 2.0)];

        assert_eq!(interpolate_samples(&samples, -1.0), Some(1.0));
        assert_eq!(interpolate_samples(&samples, 2.5), Some(1.5));
        assert_eq!(interpolate_samples(&samples, 99.0), Some(4.0));
    }

    #[test]
    fn build_rejects_invalid_receiver_state_before_disabled_short_circuit() {
        let sp3 = sp3_fixture();
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
        };

        for (receiver, field, reason) in [
            (
                [f64::NAN, 780_500.0, 5_248_700.0],
                "receiver_ecef_m",
                "not finite",
            ),
            ([0.0, 0.0, 0.0], "receiver radius_m", "not positive"),
        ] {
            let err = build(&sp3, &[], receiver, &options)
                .expect_err("invalid receiver state must error before empty success");

            assert_eq!(err, PppCorrectionsError::InvalidInput { field, reason });
        }
    }

    #[test]
    fn build_rejects_invalid_correction_epoch_without_panicking() {
        let sp3 = sp3_fixture();
        let options = PppCorrectionsOptions {
            solid_earth_tide: true,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
        };
        let epochs = vec![PppCorrectionEpoch {
            epoch: civil(2021, 2, 29, 12, 0, 0.0),
            t_rx_j2000_s: 0.0,
            observations: Vec::new(),
        }];

        let err = build(
            &sp3,
            &epochs,
            [3_512_900.0, 780_500.0, 5_248_700.0],
            &options,
        )
        .expect_err("invalid PPP correction epoch must return an error");

        assert_eq!(
            err,
            PppCorrectionsError::Epoch {
                epoch_index: 0,
                source: CoverageError::InvalidInput {
                    field: "civil datetime",
                    kind: TimeScaleInputErrorKind::InvalidCivilDate,
                },
            }
        );
    }

    #[test]
    fn build_rejects_non_finite_correction_epoch_without_panicking() {
        let sp3 = sp3_fixture();
        let options = PppCorrectionsOptions {
            solid_earth_tide: true,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
        };
        let epochs = vec![PppCorrectionEpoch {
            epoch: civil(2020, 6, 24, 12, 0, f64::NAN),
            t_rx_j2000_s: 0.0,
            observations: Vec::new(),
        }];

        let err = build(
            &sp3,
            &epochs,
            [3_512_900.0, 780_500.0, 5_248_700.0],
            &options,
        )
        .expect_err("non-finite PPP correction epoch must return an error");

        assert_eq!(
            err,
            PppCorrectionsError::Epoch {
                epoch_index: 0,
                source: CoverageError::InvalidInput {
                    field: "civil datetime",
                    kind: TimeScaleInputErrorKind::NonFinite,
                },
            }
        );
    }

    #[test]
    fn build_rejects_correction_epoch_after_eop_coverage() {
        let sp3 = sp3_fixture();
        let options = PppCorrectionsOptions {
            solid_earth_tide: true,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
        };
        let epochs = vec![PppCorrectionEpoch {
            epoch: civil(2100, 1, 1, 0, 0, 0.0),
            t_rx_j2000_s: 0.0,
            observations: Vec::new(),
        }];

        let err = build(
            &sp3,
            &epochs,
            [3_512_900.0, 780_500.0, 5_248_700.0],
            &options,
        )
        .expect_err("post-coverage PPP correction epoch must return an error");

        assert_eq!(
            err,
            PppCorrectionsError::Epoch {
                epoch_index: 0,
                source: CoverageError::OutsideCoverage(
                    crate::astro::time::DegradeReason::AfterCoverage
                ),
            }
        );
    }

    #[test]
    fn build_accepts_valid_correction_epoch() {
        let sp3 = sp3_fixture();
        let options = PppCorrectionsOptions {
            solid_earth_tide: true,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
        };
        let epochs = vec![PppCorrectionEpoch {
            epoch: civil(2020, 6, 24, 12, 0, 0.0),
            t_rx_j2000_s: 0.0,
            observations: Vec::new(),
        }];

        let got = build(
            &sp3,
            &epochs,
            [3_512_900.0, 780_500.0, 5_248_700.0],
            &options,
        )
        .expect("valid PPP correction epoch must build");

        assert_eq!(got.tide.len(), 1);
    }

    #[test]
    fn build_rejects_degenerate_receiver_state_before_tide() {
        let sp3 = sp3_fixture();
        let epoch = civil(2020, 6, 24, 12, 0, 0.0);
        let options = PppCorrectionsOptions {
            solid_earth_tide: true,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
        };
        let epochs = vec![PppCorrectionEpoch {
            epoch,
            t_rx_j2000_s: 0.0,
            observations: Vec::new(),
        }];

        let err = build(&sp3, &epochs, [0.0, 0.0, 0.0], &options)
            .expect_err("degenerate tide geometry must error");

        assert_eq!(
            err,
            PppCorrectionsError::InvalidInput {
                field: "receiver radius_m",
                reason: "not positive",
            }
        );
    }
}
