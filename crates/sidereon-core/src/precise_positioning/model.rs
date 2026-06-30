//! Static PPP measurement-model leaf: range-correction algebra, receiver and
//! satellite antenna PCO/PCV projection, troposphere slant and mapping,
//! satellite-clock and relativity corrections, elevation weighting, and the
//! local north/east/up frame geometry the antenna projection depends on.
//!
//! These are pure modeling helpers shared by the float and fixed solve clusters
//! in the parent module; they hold no solve state of their own.

use std::collections::BTreeMap;

use crate::astro::angles::{normalize_geodetic_lon_rad, rad_to_deg_ref};
use crate::astro::frames::transforms::itrs_to_geodetic_compute;
use crate::astro::math::interp::lerp_ratio;
use crate::astro::math::vec3::{add3, dot3, scale3, sub3, unit3};
use crate::astro::time::civil::mjd_from_jd;
use crate::astro::time::model::{Instant, JulianDateSplit, TimeModelError, TimeScale};

use crate::antenna;
use crate::constants::{C_M_S, DEG_TO_RAD, GPS_EPOCH_TO_J2000_S, KM_TO_M};
use crate::observables::PredictedObservables;
use crate::tropo::{tropo_mapping_unchecked, tropo_slant_with_mapping_unchecked, MappingModel};
use crate::{GnssSatelliteId, Wgs84Geodetic};

use super::{
    estimates_ztd, missing_correction, missing_satellite_clock, FloatEpoch, FloatObservation,
    FloatSolveError, FloatState, MeasurementWeights, MissingCorrection, PppCorrectionLookup,
    RangeCorrections, ReceiverAntennaFrequency, ReceiverAntennaOptions, SatelliteClockCorrections,
    TropoMapping, TroposphereOptions,
};

const MIN_ELEVATION_WEIGHT_SCALE: f64 = 1.0e-3;

#[derive(Debug, Clone, Copy)]
pub(super) struct TropoModelState {
    slant_m: f64,
    pub(super) ztd_mapping: f64,
}

pub(super) fn model_troposphere(
    pred: &PredictedObservables,
    receiver_m: [f64; 3],
    epoch: &FloatEpoch,
    tropo: TroposphereOptions,
) -> Result<TropoModelState, FloatSolveError> {
    if !tropo.enabled {
        return Ok(TropoModelState {
            slant_m: 0.0,
            ztd_mapping: 0.0,
        });
    }

    let (lat_deg, lon_deg, height_km) = itrs_to_geodetic_compute(
        receiver_m[0] / KM_TO_M,
        receiver_m[1] / KM_TO_M,
        receiver_m[2] / KM_TO_M,
    )
    .expect("valid receiver ITRS coordinates");
    let receiver = Wgs84Geodetic::new(
        lat_deg.to_radians(),
        normalize_geodetic_lon_rad(lon_deg.to_radians()),
        height_km * KM_TO_M,
    )
    .expect("valid WGS84 geodetic position");
    let jd = JulianDateSplit::new(epoch.jd_whole, epoch.jd_fraction)
        .map_err(ppp_tropo_invalid_julian_split)?;
    let instant = Instant::from_julian_date(TimeScale::Gpst, jd);
    let elevation_rad = pred.elevation_deg.to_radians();
    // VMF nodes are tabulated in UTC; the ~18 s GPST-UTC offset is far below the
    // 6-hourly sampling and shifts the interpolated `a` coefficients by ~1e-9, so
    // the GPST MJD is used directly.
    let mapping_model = match tropo.mapping {
        TropoMapping::Niell => MappingModel::Niell,
        TropoMapping::Vmf1(series) => {
            let mjd = mjd_from_jd(epoch.jd_whole + epoch.jd_fraction);
            // Flag an epoch outside the VMF site series rather than reusing a
            // stale endpoint coefficient for every later epoch (the unbounded
            // clamp). A short clamp past each endpoint is still honored.
            let (ah, aw) = series.interpolate_checked(mjd).ok_or({
                FloatSolveError::InvalidInput {
                    field: "ppp tropo vmf epoch",
                    reason: "epoch is outside the VMF site-coefficient series span",
                }
            })?;
            MappingModel::Vmf1 { ah, aw }
        }
    };
    let slant_m = tropo_slant_with_mapping_unchecked(
        mapping_model,
        elevation_rad,
        receiver,
        tropo.met,
        instant,
    );
    let ztd_mapping = if estimates_ztd(tropo) {
        tropo_mapping_unchecked(mapping_model, elevation_rad, receiver, instant).wet
    } else {
        0.0
    };
    Ok(TropoModelState {
        slant_m,
        ztd_mapping,
    })
}

fn ppp_tropo_invalid_julian_split(error: TimeModelError) -> FloatSolveError {
    let TimeModelError::InvalidInput { field, reason } = error;
    let field = match field {
        "jd_whole" => "ppp epoch jd_whole",
        "fraction" => "ppp epoch jd_fraction",
        _ => "ppp epoch Julian date split",
    };
    FloatSolveError::InvalidInput { field, reason }
}

fn applied_troposphere_m(tropo_model: &TropoModelState, state: &FloatState) -> f64 {
    tropo_model.slant_m + state.ztd_m * tropo_model.ztd_mapping
}

pub(super) fn range_corrections_m(
    pred: &PredictedObservables,
    rx_pos: [f64; 3],
    epoch_idx: usize,
    obs: &FloatObservation,
    tropo_model: &TropoModelState,
    state: &FloatState,
    corrections: &RangeCorrections,
) -> Result<f64, FloatSolveError> {
    let receiver_antenna_m =
        receiver_antenna_correction_m(pred, rx_pos, obs, corrections.receiver_antenna.as_ref())?;
    let satellite_clock_m =
        satellite_clock_correction_m(pred, obs, corrections.satellite_clock.as_ref())?;
    let tide_m = solid_earth_tide_correction_m(pred, obs, epoch_idx, &corrections.ppp)?;
    let pole_tide_m = pole_tide_correction_m(pred, obs, epoch_idx, &corrections.ppp)?;
    let ocean_loading_m = ocean_loading_correction_m(pred, obs, epoch_idx, &corrections.ppp)?;
    let satellite_antenna_m =
        satellite_antenna_correction_m(pred, obs, epoch_idx, &corrections.ppp)?;
    Ok(applied_troposphere_m(tropo_model, state)
        + receiver_antenna_m
        + sat_clock_relativity_correction_m(pred, corrections.sat_clock_relativity)
        + satellite_clock_m
        + tide_m
        + pole_tide_m
        + ocean_loading_m
        + satellite_antenna_m)
}

pub(super) fn satellite_clock_m(
    pred: &PredictedObservables,
    obs: &FloatObservation,
    clock: Option<&SatelliteClockCorrections>,
) -> Result<f64, FloatSolveError> {
    if let Some(sat_clock_s) = pred.sat_clock_s {
        return Ok(C_M_S * sat_clock_s);
    }
    let Some(clock) = clock else {
        return Err(missing_satellite_clock(obs));
    };
    clock
        .clock_s(obs.sat, pred.transmit_time_j2000_s)
        .map(|sat_clock_s| C_M_S * sat_clock_s)
        .ok_or_else(|| missing_satellite_clock(obs))
}

fn solid_earth_tide_correction_m(
    pred: &PredictedObservables,
    obs: &FloatObservation,
    epoch_idx: usize,
    corrections: &PppCorrectionLookup,
) -> Result<f64, FloatSolveError> {
    if !corrections.tide_enabled {
        return Ok(0.0);
    }
    corrections
        .tide
        .get(&epoch_idx)
        .map(|d| -dot3(*d, pred.los_unit))
        .ok_or_else(|| missing_correction(obs, MissingCorrection::SolidEarthTide))
}

fn pole_tide_correction_m(
    pred: &PredictedObservables,
    obs: &FloatObservation,
    epoch_idx: usize,
    corrections: &PppCorrectionLookup,
) -> Result<f64, FloatSolveError> {
    if !corrections.pole_tide_enabled {
        return Ok(0.0);
    }
    corrections
        .pole_tide
        .get(&epoch_idx)
        .map(|d| -dot3(*d, pred.los_unit))
        .ok_or_else(|| missing_correction(obs, MissingCorrection::PoleTide))
}

fn ocean_loading_correction_m(
    pred: &PredictedObservables,
    obs: &FloatObservation,
    epoch_idx: usize,
    corrections: &PppCorrectionLookup,
) -> Result<f64, FloatSolveError> {
    if !corrections.ocean_loading_enabled {
        return Ok(0.0);
    }
    corrections
        .ocean_loading
        .get(&epoch_idx)
        .map(|d| -dot3(*d, pred.los_unit))
        .ok_or_else(|| missing_correction(obs, MissingCorrection::OceanLoading))
}

fn satellite_antenna_correction_m(
    pred: &PredictedObservables,
    obs: &FloatObservation,
    epoch_idx: usize,
    corrections: &PppCorrectionLookup,
) -> Result<f64, FloatSolveError> {
    if !corrections.satellite_antenna_enabled {
        return Ok(0.0);
    }
    let key = (obs.sat, epoch_idx);
    let pco = corrections
        .sat_pco_ecef
        .get(&key)
        .ok_or_else(|| missing_correction(obs, MissingCorrection::SatelliteAntennaPco))?;
    let pcv_m = corrections
        .sat_pcv_m
        .get(&key)
        .copied()
        .ok_or_else(|| missing_correction(obs, MissingCorrection::SatelliteAntennaPcv))?;
    Ok(dot3(*pco, pred.los_unit) + pcv_m)
}

pub(super) fn phase_windup_m(
    obs: &FloatObservation,
    epoch_idx: usize,
    corrections: &RangeCorrections,
) -> Result<f64, FloatSolveError> {
    if !corrections.ppp.windup_enabled {
        return Ok(0.0);
    }
    corrections
        .ppp
        .windup_m
        .get(&(obs.sat, epoch_idx))
        .copied()
        .ok_or_else(|| missing_correction(obs, MissingCorrection::PhaseWindup))
}

fn satellite_clock_correction_m(
    pred: &PredictedObservables,
    obs: &FloatObservation,
    clock: Option<&SatelliteClockCorrections>,
) -> Result<f64, FloatSolveError> {
    let Some(clock) = clock else {
        return Ok(0.0);
    };
    match clock.clock_s(obs.sat, pred.transmit_time_j2000_s) {
        Some(rinex_clock_s) => Ok(pred
            .sat_clock_s
            .map(|sat_clock_s| C_M_S * (sat_clock_s - rinex_clock_s))
            .unwrap_or(0.0)),
        None => Err(missing_satellite_clock(obs)),
    }
}

impl SatelliteClockCorrections {
    fn clock_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        let gps_s = t_j2000_s + GPS_EPOCH_TO_J2000_S;
        let records = self.series.get(&sat)?;
        interpolate_clock(records, gps_s)
    }
}

fn interpolate_clock(records: &[(f64, f64)], t: f64) -> Option<f64> {
    let &(t_first, _) = records.first()?;
    let &(t_last, _) = records.last()?;

    // Clamped endpoint extrapolation at the arc boundaries. A transmit time can
    // land just outside the CLK node span: signal travel time pushes the first
    // epoch's transmit time ~0.07 s before the first node, and the last epoch
    // sits just past the last node. Without this, an interior-only lookup
    // returns `None` at the very first/last epoch of a 30 s-CLK arc and the
    // whole solve fails with "satellite clock unavailable".
    if t < t_first {
        return clamped_extrapolation(records.first()?, records.get(1), t);
    }
    if t > t_last {
        let inner = records.len().checked_sub(2).and_then(|i| records.get(i));
        return clamped_extrapolation(records.last()?, inner, t);
    }

    let mut prev: Option<(f64, f64)> = None;
    for &(ti, bi) in records {
        if ti == t {
            return Some(bi);
        }
        if ti > t {
            let (t0, b0) = prev?;
            return Some(lerp_ratio(b0, bi, t - t0, ti - t0));
        }
        prev = Some((ti, bi));
    }
    None
}

/// Clamped linear extrapolation past a CLK series endpoint, used only for a
/// transmit time just outside the node span. `edge` is the nearest endpoint
/// node and `inner` its neighbour; extrapolation along the boundary segment is
/// permitted within one node interval of the edge, beyond which the satellite
/// clock is genuinely unavailable (`None`).
fn clamped_extrapolation(edge: &(f64, f64), inner: Option<&(f64, f64)>, t: f64) -> Option<f64> {
    let &(t_edge, b_edge) = edge;
    let Some(&(t_inner, b_inner)) = inner else {
        // Single-node series: with no neighbour there is no segment to define a
        // slope or bound the extrapolation, so a clock is available only AT the
        // node itself - any other transmit time is genuinely unavailable, never a
        // silently held stale value.
        return (t == t_edge).then_some(b_edge);
    };
    // Nodes are validated finite upstream, so the interval is finite and the
    // ordering is total here.
    let interval = (t_edge - t_inner).abs();
    if interval <= 0.0 || (t - t_edge).abs() > interval {
        return None;
    }
    let slope = (b_edge - b_inner) / (t_edge - t_inner);
    Some(b_edge + slope * (t - t_edge))
}

fn receiver_antenna_correction_m(
    pred: &PredictedObservables,
    rx_pos: [f64; 3],
    obs: &FloatObservation,
    receiver_antenna: Option<&ReceiverAntennaOptions>,
) -> Result<f64, FloatSolveError> {
    let Some(antenna) = receiver_antenna else {
        return Ok(0.0);
    };
    let c1 = single_freq_receiver_antenna_m(pred, rx_pos, obs, antenna, &antenna.freq1_label)?;
    let c2 = single_freq_receiver_antenna_m(pred, rx_pos, obs, antenna, &antenna.freq2_label)?;
    let gamma = antenna.freq1_hz * antenna.freq1_hz
        / (antenna.freq1_hz * antenna.freq1_hz - antenna.freq2_hz * antenna.freq2_hz);
    Ok(-(gamma * c1 - (gamma - 1.0) * c2))
}

fn single_freq_receiver_antenna_m(
    pred: &PredictedObservables,
    rx_pos: [f64; 3],
    obs: &FloatObservation,
    antenna: &ReceiverAntennaOptions,
    frequency: &str,
) -> Result<f64, FloatSolveError> {
    let Some(freq) = antenna.frequencies.iter().find(|f| f.label == frequency) else {
        return Err(missing_correction(
            obs,
            MissingCorrection::ReceiverAntennaFrequency(frequency.to_string()),
        ));
    };
    let Some(los) = unit3(sub3(pred.sat_pos_ecef_m, rx_pos)) else {
        return Err(missing_correction(
            obs,
            MissingCorrection::ReceiverAntennaGeometry,
        ));
    };
    let (north, east, up) = crate::estimation::substrate::frames::local_neu_basis(
        crate::estimation::recipe::FrameRecipe::GeodeticNeuCrossProduct,
        rx_pos,
    );
    let pco_projection = los_projection(freq.pco_m, north, east, up, los);
    let (zenith_deg, azimuth_deg) = los_zenith_azimuth_deg(los, up, north, east);
    let pcv_m = pcv(freq, zenith_deg, Some(azimuth_deg)).ok_or_else(|| {
        missing_correction(
            obs,
            MissingCorrection::ReceiverAntennaPcv(frequency.to_string()),
        )
    })?;
    Ok(pco_projection + pcv_m)
}

fn pcv(freq: &ReceiverAntennaFrequency, zenith_deg: f64, azimuth_deg: Option<f64>) -> Option<f64> {
    let noazi: Vec<(f64, f64)> = freq
        .pcv_samples
        .iter()
        .filter(|s| s.azimuth_deg.is_none())
        .map(|s| (s.zenith_deg, s.value_m))
        .collect();
    let has_azi = freq.pcv_samples.iter().any(|s| s.azimuth_deg.is_some());
    if azimuth_deg.is_none() || !has_azi {
        return interpolate_samples(noazi, zenith_deg);
    }

    let mut by_az: BTreeMap<i64, Vec<(f64, f64)>> = BTreeMap::new();
    for sample in freq.pcv_samples.iter().filter(|s| s.azimuth_deg.is_some()) {
        let az_key = (sample.azimuth_deg.unwrap() * 1.0e9).round() as i64;
        by_az
            .entry(az_key)
            .or_default()
            .push((sample.zenith_deg, sample.value_m));
    }
    if by_az.is_empty() {
        return interpolate_samples(noazi, zenith_deg);
    }
    interpolate_azimuth(by_az, azimuth_deg.unwrap(), zenith_deg)
}

fn interpolate_azimuth(
    azimuth_samples: BTreeMap<i64, Vec<(f64, f64)>>,
    azimuth_deg: f64,
    zenith_deg: f64,
) -> Option<f64> {
    let azimuths: Vec<f64> = azimuth_samples.keys().map(|k| *k as f64 / 1.0e9).collect();
    let azimuth = antenna::normalize_azimuth(azimuth_deg);
    let (low_deg, high_deg) = antenna::azimuth_bracket(&azimuths, azimuth);
    let low_samples = azimuth_samples
        .get(&((low_deg * 1.0e9).round() as i64))
        .cloned()
        .unwrap_or_default();
    let high_samples = azimuth_samples
        .get(&((high_deg * 1.0e9).round() as i64))
        .cloned()
        .unwrap_or_default();
    let low_value = interpolate_samples(low_samples, zenith_deg)?;
    let high_value = interpolate_samples(high_samples, zenith_deg)?;
    Some(antenna::blend_azimuth(
        low_deg, high_deg, azimuth, low_value, high_value,
    ))
}

fn interpolate_samples(mut samples: Vec<(f64, f64)>, zenith_deg: f64) -> Option<f64> {
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    antenna::interpolate_zenith_sorted(&samples, zenith_deg)
}

fn sat_clock_relativity_correction_m(pred: &PredictedObservables, enabled: bool) -> f64 {
    if !enabled {
        return 0.0;
    }
    2.0 * dot3(pred.sat_pos_ecef_m, pred.sat_velocity_m_s) / C_M_S
}

pub(super) fn measurement_weight(
    weights: MeasurementWeights,
    code: bool,
    elevation_deg: f64,
) -> f64 {
    let base = if code { weights.code } else { weights.phase };
    if weights.elevation_weighting {
        base * elevation_weight_scale(elevation_deg)
    } else {
        base
    }
}

fn elevation_weight_scale(elevation_deg: f64) -> f64 {
    let sin_el = (elevation_deg * DEG_TO_RAD).sin();
    if !sin_el.is_finite() || sin_el < MIN_ELEVATION_WEIGHT_SCALE {
        MIN_ELEVATION_WEIGHT_SCALE
    } else {
        sin_el
    }
}

fn los_projection(
    neu_offset: [f64; 3],
    north: [f64; 3],
    east: [f64; 3],
    up: [f64; 3],
    los: [f64; 3],
) -> f64 {
    let pco_ecef = add3(
        add3(scale3(north, neu_offset[0]), scale3(east, neu_offset[1])),
        scale3(up, neu_offset[2]),
    );
    dot3(pco_ecef, los)
}

fn los_zenith_azimuth_deg(
    los: [f64; 3],
    up: [f64; 3],
    north: [f64; 3],
    east: [f64; 3],
) -> (f64, f64) {
    let elevation_sin = dot3(los, up).clamp(-1.0, 1.0);
    let zenith_deg = rad_to_deg_ref(elevation_sin.acos());
    let mut azimuth_deg = rad_to_deg_ref(dot3(los, east).atan2(dot3(los, north)));
    if azimuth_deg < 0.0 {
        azimuth_deg += 360.0;
    }
    (zenith_deg, azimuth_deg)
}

#[cfg(test)]
mod clock_boundary_tests {
    use super::{clamped_extrapolation, interpolate_clock};

    // A 30 s CLK arc: three nodes at 0/30/60 s with a linear ramp.
    const RECORDS: &[(f64, f64)] = &[(0.0, 1.0e-6), (30.0, 1.3e-6), (60.0, 1.5e-6)];

    #[test]
    fn interior_query_is_unchanged_linear_interpolation() {
        let got = interpolate_clock(RECORDS, 15.0).expect("interior resolves");
        assert!((got - 1.15e-6).abs() < 1.0e-18, "got {got}");
        // Exact node hit returns the node value.
        assert_eq!(interpolate_clock(RECORDS, 30.0), Some(1.3e-6));
    }

    #[test]
    fn transmit_time_just_before_first_node_resolves() {
        // ~0.07 s before the first node (the signal-travel-time case that used to
        // return None and fail the whole first-epoch solve).
        let got = interpolate_clock(RECORDS, -0.07).expect("pre-first-node resolves");
        let slope = (1.3e-6 - 1.0e-6) / 30.0;
        assert!(
            (got - (1.0e-6 + slope * -0.07)).abs() < 1.0e-18,
            "got {got}"
        );
    }

    #[test]
    fn transmit_time_just_after_last_node_resolves() {
        let got = interpolate_clock(RECORDS, 60.05).expect("post-last-node resolves");
        let slope = (1.5e-6 - 1.3e-6) / 30.0;
        assert!((got - (1.5e-6 + slope * 0.05)).abs() < 1.0e-18, "got {got}");
    }

    #[test]
    fn extrapolation_beyond_one_node_interval_is_unavailable() {
        // More than one 30 s interval before/after the edge is genuinely missing.
        assert_eq!(interpolate_clock(RECORDS, -31.0), None);
        assert_eq!(interpolate_clock(RECORDS, 91.0), None);
    }

    #[test]
    fn single_node_series_resolves_only_at_the_node() {
        // A lone CLK node has no segment to extrapolate along, so a clock is
        // available only AT the node; any other time is unavailable rather than a
        // silently held stale value.
        let single = [(10.0, 4.2e-6)];
        assert_eq!(interpolate_clock(&single, 10.0), Some(4.2e-6));
        assert_eq!(interpolate_clock(&single, 9.0), None);
        assert_eq!(interpolate_clock(&single, 11.0), None);
        assert_eq!(clamped_extrapolation(&single[0], None, 9.0), None);
        assert_eq!(clamped_extrapolation(&single[0], None, 10.0), Some(4.2e-6));
    }

    #[test]
    fn empty_series_is_unavailable() {
        assert_eq!(interpolate_clock(&[], 0.0), None);
    }
}
