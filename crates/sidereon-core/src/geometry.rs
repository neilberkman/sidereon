//! GNSS geometry primitives.

pub(crate) mod range;

use crate::astro::frames::transforms::itrs_to_geodetic_compute;
use std::collections::BTreeSet;
use std::f64::consts::PI;

use crate::astro::angles::normalize_geodetic_lon_rad;

use crate::constants::{C_M_S, F_L1_HZ, KM_TO_M, OMEGA_E_DOT_RAD_S};
pub use crate::dop::{
    dop, dop_with_convention, error_ellipse_2x2, error_ellipse_from_geometry, geometry_cofactor,
    geometry_cofactor_with_convention, horizontal_error_ellipse, line_of_sight_from_az_el_deg,
    position_covariance_from_geometry_m2, Dop, DopError, EnuConvention, ErrorEllipse2,
    GeometryCofactor, HorizontalErrorEllipse, LineOfSight, PositionCovariance,
};
pub use crate::frame::{ItrfPositionM, ItrfVelocityMS, Wgs84Geodetic};
use crate::observables::{predict, PredictOptions};
pub use crate::observables::{
    transmit_time_satellite_state, ObservableEphemerisSource, ObservableState, ObservablesError,
    TransmitTimeOptions, TransmitTimeSatelliteState,
};
use crate::validate;
use crate::{GnssSatelliteId, GnssSystem};

/// Error type returned by DOP calculations.
pub type Error = DopError;

const DEFAULT_ELEVATION_MASK_DEG: f64 = 5.0;
const DEG_TO_RAD: f64 = PI / 180.0;

/// Closed-form Sagnac/Earth-rotation transport of a transmit-time ECEF satellite
/// position into the receive-time ECEF frame.
///
/// Uses the canonical WGS84 Earth sidereal rate [`OMEGA_E_DOT_RAD_S`] and the
/// same `+omega*tau` Z rotation used by SPP and observable prediction.
pub fn sagnac_rotate_ecef_m(position_ecef_m: [f64; 3], signal_flight_time_s: f64) -> [f64; 3] {
    sagnac_rotate_ecef_m_with_rate(position_ecef_m, signal_flight_time_s, OMEGA_E_DOT_RAD_S)
}

/// Closed-form Sagnac/Earth-rotation transport with an explicit Earth rotation
/// rate in radians per second.
pub fn sagnac_rotate_ecef_m_with_rate(
    position_ecef_m: [f64; 3],
    signal_flight_time_s: f64,
    omega_rad_s: f64,
) -> [f64; 3] {
    range::sagnac_rotate_exact(position_ecef_m, signal_flight_time_s, omega_rad_s)
}

/// First-order RTKLIB-style scalar Sagnac range correction.
///
/// Returns the Euclidean satellite-receiver range plus
/// `omega * (sat_x * recv_y - sat_y * recv_x) / c`, using
/// [`OMEGA_E_DOT_RAD_S`] and [`C_M_S`].
pub fn sagnac_range_first_order_m(satellite_ecef_m: [f64; 3], receiver_ecef_m: [f64; 3]) -> f64 {
    sagnac_range_first_order_m_with_rate(
        satellite_ecef_m,
        receiver_ecef_m,
        OMEGA_E_DOT_RAD_S,
        C_M_S,
    )
}

/// First-order RTKLIB-style scalar Sagnac range correction with explicit
/// rotation rate and light speed.
pub fn sagnac_range_first_order_m_with_rate(
    satellite_ecef_m: [f64; 3],
    receiver_ecef_m: [f64; 3],
    omega_rad_s: f64,
    c_m_s: f64,
) -> f64 {
    range::sagnac_range_first_order(satellite_ecef_m, receiver_ecef_m, omega_rad_s, c_m_s)
}

/// Visibility planning options for SP3-derived GNSS geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct VisibilityOptions {
    /// Minimum topocentric elevation, degrees.
    pub elevation_mask_deg: f64,
    /// Optional constellation filter. `None` keeps all systems.
    pub systems: Option<BTreeSet<GnssSystem>>,
}

impl Default for VisibilityOptions {
    fn default() -> Self {
        Self {
            elevation_mask_deg: DEFAULT_ELEVATION_MASK_DEG,
            systems: None,
        }
    }
}

/// DOP weighting policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DopWeighting {
    /// Unweighted geometric DOP.
    #[default]
    Unit,
    /// Elevation weighting with `sin(elevation)^2`.
    Elevation,
}

/// DOP planning options.
#[derive(Debug, Clone, PartialEq)]
pub struct DopOptions {
    /// Visibility scan options used when no explicit satellite list is supplied.
    pub visibility: VisibilityOptions,
    /// DOP row weighting policy.
    pub weighting: DopWeighting,
    /// Apply light-time and Sagnac corrections when forming the line of sight.
    pub light_time: bool,
}

impl Default for DopOptions {
    fn default() -> Self {
        Self {
            visibility: VisibilityOptions::default(),
            weighting: DopWeighting::Unit,
            light_time: false,
        }
    }
}

/// One visible satellite row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisibleSatellite {
    /// Satellite identifier.
    pub satellite: GnssSatelliteId,
    /// Topocentric elevation, degrees.
    pub elevation_deg: f64,
    /// Topocentric azimuth, degrees in `[0, 360)`.
    pub azimuth_deg: f64,
}

/// DOP result plus the exact satellites that contributed rows.
#[derive(Debug, Clone, PartialEq)]
pub struct DopAtEpoch {
    /// DOP scalars.
    pub dop: Dop,
    /// Satellites with successful predicted line-of-sight rows.
    pub satellites: Vec<GnssSatelliteId>,
}

/// DOP result for one sampled epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct DopSeriesPoint {
    /// Zero-based sample index from the series start.
    pub step_index: usize,
    /// DOP result at this sample.
    pub geometry: DopAtEpoch,
}

/// Visible-satellite count for one sampled epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisibilitySeriesPoint {
    /// Zero-based sample index from the series start.
    pub step_index: usize,
    /// Number of satellites visible at this sample.
    pub n_visible: usize,
}

/// One sampled visibility pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisibilityPass {
    /// Satellite identifier.
    pub satellite: GnssSatelliteId,
    /// Zero-based sample index of the first above-mask sample.
    pub rise_step_index: usize,
    /// Zero-based sample index of the last above-mask sample.
    pub set_step_index: usize,
    /// Maximum sampled elevation in the pass, degrees.
    pub peak_elevation_deg: f64,
    /// Zero-based sample index of the maximum sampled elevation.
    pub peak_step_index: usize,
}

#[derive(Debug, Clone, Copy)]
struct VisibilitySample {
    step_index: usize,
    elevation_deg: f64,
}

/// List satellites visible from a static receiver at one epoch.
pub fn visible(
    source: &dyn ObservableEphemerisSource,
    satellites: &[GnssSatelliteId],
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: &VisibilityOptions,
) -> Result<Vec<VisibleSatellite>, DopError> {
    validate_visibility_options(options)?;

    let mut visible = Vec::new();
    for &sat in satellites {
        if !system_allowed(sat, options.systems.as_ref()) {
            continue;
        }

        let prediction = predict(
            source,
            sat,
            receiver_ecef_m,
            t_rx_j2000_s,
            PredictOptions {
                carrier_hz: F_L1_HZ,
                light_time: false,
                sagnac: true,
            },
        );
        let Ok(obs) = prediction else {
            continue;
        };
        if obs.elevation_deg >= options.elevation_mask_deg {
            visible.push(VisibleSatellite {
                satellite: sat,
                elevation_deg: obs.elevation_deg,
                azimuth_deg: obs.azimuth_deg,
            });
        }
    }

    visible.sort_by(|a, b| b.elevation_deg.total_cmp(&a.elevation_deg));
    Ok(visible)
}

/// Compute DOP at one epoch from either an explicit satellite set or a visibility scan.
pub fn dop_at_epoch(
    source: &dyn ObservableEphemerisSource,
    all_satellites: &[GnssSatelliteId],
    explicit_satellites: Option<&[GnssSatelliteId]>,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: &DopOptions,
) -> Result<DopAtEpoch, DopError> {
    validate::finite_vec3(receiver_ecef_m, "receiver_ecef_m").map_err(map_geometry_input)?;
    validate_visibility_options(&options.visibility)?;

    let selected: Vec<GnssSatelliteId> = match explicit_satellites {
        Some(satellites) => satellites.to_vec(),
        None => visible(
            source,
            all_satellites,
            receiver_ecef_m,
            t_rx_j2000_s,
            &options.visibility,
        )?
        .into_iter()
        .map(|sat| sat.satellite)
        .collect(),
    };

    let mut line_of_sight = Vec::new();
    let mut weights = Vec::new();
    let mut used = Vec::new();
    for sat in selected {
        let prediction = predict(
            source,
            sat,
            receiver_ecef_m,
            t_rx_j2000_s,
            PredictOptions {
                carrier_hz: F_L1_HZ,
                light_time: options.light_time,
                sagnac: options.light_time,
            },
        );
        let Ok(obs) = prediction else {
            continue;
        };
        line_of_sight.push(LineOfSight::new(
            obs.los_unit[0],
            obs.los_unit[1],
            obs.los_unit[2],
        ));
        weights.push(weight_for(options.weighting, obs.elevation_deg));
        used.push(sat);
    }

    let receiver = receiver_geodetic(receiver_ecef_m)?;
    let dop = dop(&line_of_sight, &weights, receiver)?;
    Ok(DopAtEpoch {
        dop,
        satellites: used,
    })
}

/// Sample DOP over an inclusive time window, skipping singular or underdetermined samples.
pub fn dop_series(
    source: &dyn ObservableEphemerisSource,
    all_satellites: &[GnssSatelliteId],
    explicit_satellites: Option<&[GnssSatelliteId]>,
    receiver_ecef_m: [f64; 3],
    window_j2000_s: (f64, f64),
    step_seconds: u64,
    options: &DopOptions,
) -> Result<Vec<DopSeriesPoint>, DopError> {
    validate::finite_vec3(receiver_ecef_m, "receiver_ecef_m").map_err(map_geometry_input)?;
    validate_visibility_options(&options.visibility)?;

    let mut out = Vec::new();
    for (step_index, t_rx_j2000_s) in sample_times(window_j2000_s, step_seconds)? {
        if let Ok(geometry) = dop_at_epoch(
            source,
            all_satellites,
            explicit_satellites,
            receiver_ecef_m,
            t_rx_j2000_s,
            options,
        ) {
            out.push(DopSeriesPoint {
                step_index,
                geometry,
            });
        }
    }
    Ok(out)
}

/// Count visible satellites over an inclusive sampled time window.
pub fn visibility_series(
    source: &dyn ObservableEphemerisSource,
    satellites: &[GnssSatelliteId],
    receiver_ecef_m: [f64; 3],
    window_j2000_s: (f64, f64),
    step_seconds: u64,
    options: &VisibilityOptions,
) -> Result<Vec<VisibilitySeriesPoint>, DopError> {
    validate_visibility_options(options)?;

    sample_times(window_j2000_s, step_seconds)?
        .into_iter()
        .map(|(step_index, t_rx_j2000_s)| {
            visible(source, satellites, receiver_ecef_m, t_rx_j2000_s, options).map(|visible| {
                VisibilitySeriesPoint {
                    step_index,
                    n_visible: visible.len(),
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()
}

/// Build sampled rise/set/peak visibility passes over an inclusive time window.
pub fn passes(
    source: &dyn ObservableEphemerisSource,
    satellites: &[GnssSatelliteId],
    receiver_ecef_m: [f64; 3],
    window_j2000_s: (f64, f64),
    step_seconds: u64,
    options: &VisibilityOptions,
) -> Result<Vec<VisibilityPass>, DopError> {
    validate_visibility_options(options)?;

    let samples = sample_times(window_j2000_s, step_seconds)?;
    let mut out = Vec::new();

    for &sat in satellites {
        if !system_allowed(sat, options.systems.as_ref()) {
            continue;
        }

        let mut current_run: Vec<VisibilitySample> = Vec::new();
        for &(step_index, t_rx_j2000_s) in &samples {
            let prediction = predict(
                source,
                sat,
                receiver_ecef_m,
                t_rx_j2000_s,
                PredictOptions {
                    carrier_hz: F_L1_HZ,
                    light_time: false,
                    sagnac: true,
                },
            );
            let above = match prediction {
                Ok(obs) if obs.elevation_deg >= options.elevation_mask_deg => {
                    Some(VisibilitySample {
                        step_index,
                        elevation_deg: obs.elevation_deg,
                    })
                }
                Ok(_) | Err(_) => None,
            };

            match above {
                Some(sample) => current_run.push(sample),
                None if !current_run.is_empty() => {
                    out.push(pass_from_run(sat, &current_run));
                    current_run.clear();
                }
                None => {}
            }
        }

        if !current_run.is_empty() {
            out.push(pass_from_run(sat, &current_run));
        }
    }

    out.sort_by_key(|pass| pass.rise_step_index);
    Ok(out)
}

fn system_allowed(sat: GnssSatelliteId, systems: Option<&BTreeSet<GnssSystem>>) -> bool {
    systems.is_none_or(|systems| systems.contains(&sat.system))
}

fn weight_for(weighting: DopWeighting, elevation_deg: f64) -> f64 {
    match weighting {
        DopWeighting::Unit => 1.0,
        DopWeighting::Elevation => {
            let s = (elevation_deg * DEG_TO_RAD).sin();
            s * s
        }
    }
}

fn validate_visibility_options(options: &VisibilityOptions) -> Result<(), DopError> {
    validate::finite_in_range(
        options.elevation_mask_deg,
        -90.0,
        90.0,
        "elevation_mask_deg",
    )
    .map(|_| ())
    .map_err(map_geometry_input)
}

fn receiver_geodetic(receiver_ecef_m: [f64; 3]) -> Result<Wgs84Geodetic, DopError> {
    let (lat_deg, lon_deg, _height_km) = itrs_to_geodetic_compute(
        receiver_ecef_m[0] / KM_TO_M,
        receiver_ecef_m[1] / KM_TO_M,
        receiver_ecef_m[2] / KM_TO_M,
    )
    .map_err(|_| invalid_receiver_geodetic())?;
    let lon_rad = normalize_geodetic_lon_rad(lon_deg * DEG_TO_RAD);
    Wgs84Geodetic::new(lat_deg * DEG_TO_RAD, lon_rad, 0.0).map_err(|_| invalid_receiver_geodetic())
}

fn invalid_receiver_geodetic() -> DopError {
    DopError::InvalidInput {
        field: "receiver_ecef_m",
        reason: "invalid geodetic",
    }
}

fn sample_times(
    window_j2000_s: (f64, f64),
    step_seconds: u64,
) -> Result<Vec<(usize, f64)>, DopError> {
    validate::positive_step(step_seconds as f64, "step_seconds").map_err(map_geometry_input)?;

    let (t0, t1) = window_j2000_s;
    validate::finite(t0, "window_j2000_s.0").map_err(map_geometry_input)?;
    validate::finite(t1, "window_j2000_s.1").map_err(map_geometry_input)?;
    if t0 > t1 {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    let step = step_seconds as f64;
    let mut step_index = 0usize;
    loop {
        let t = t0 + step * step_index as f64;
        if t > t1 {
            break;
        }
        out.push((step_index, t));
        step_index += 1;
    }
    if let Some((_, last_t)) = out.last() {
        if *last_t < t1 {
            out.push((step_index, t1));
        }
    }
    Ok(out)
}

fn map_geometry_input(error: validate::FieldError) -> DopError {
    DopError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

fn pass_from_run(sat: GnssSatelliteId, run: &[VisibilitySample]) -> VisibilityPass {
    let rise = run[0];
    let set = run[run.len() - 1];
    let mut peak = run[0];
    for &sample in &run[1..] {
        if sample.elevation_deg > peak.elevation_deg {
            peak = sample;
        }
    }

    VisibilityPass {
        satellite: sat,
        rise_step_index: rise.step_index,
        set_step_index: set.step_index,
        peak_elevation_deg: peak.elevation_deg,
        peak_step_index: peak.step_index,
    }
}

#[cfg(test)]
mod sampling_tests {
    use super::*;
    use crate::observables::{ObservableState, ObservablesError};

    const RECEIVER_ECEF_M: [f64; 3] = [6_378_137.0, 0.0, 0.0];
    const ANTI_MERIDIAN_RECEIVER_ECEF_M: [f64; 3] = [-6_378_137.0, 0.0, 0.0];
    const RANGE_M: f64 = 20_200_000.0;

    #[test]
    fn public_sagnac_helpers_match_explicit_formulas() {
        let sat = [15_600_000.0, -20_400_000.0, 9_800_000.0];
        let recv = [4_027_894.0, 307_046.0, 4_919_474.0];
        let tau = 0.072_345;
        let theta = OMEGA_E_DOT_RAD_S * tau;
        let c = theta.cos();
        let s = theta.sin();
        let rotated = sagnac_rotate_ecef_m(sat, tau);
        assert_eq!(
            rotated.map(f64::to_bits),
            [
                (c * sat[0] + s * sat[1]).to_bits(),
                (-s * sat[0] + c * sat[1]).to_bits(),
                sat[2].to_bits(),
            ]
        );

        let dx = sat[0] - recv[0];
        let dy = sat[1] - recv[1];
        let dz = sat[2] - recv[2];
        let euclid = (dx * dx + dy * dy + dz * dz).sqrt();
        let want = euclid + OMEGA_E_DOT_RAD_S * (sat[0] * recv[1] - sat[1] * recv[0]) / C_M_S;
        assert_eq!(
            sagnac_range_first_order_m(sat, recv).to_bits(),
            want.to_bits()
        );
    }

    struct FinalOnlySource {
        visible_from_s: f64,
    }

    impl ObservableEphemerisSource for FinalOnlySource {
        fn observable_state_at_j2000_s(
            &self,
            sat: GnssSatelliteId,
            t_j2000_s: f64,
        ) -> Result<ObservableState, ObservablesError> {
            let los = if t_j2000_s >= self.visible_from_s {
                final_los(sat)
            } else {
                [-1.0, 0.0, 0.0]
            };
            Ok(ObservableState {
                position_ecef_m: [
                    RECEIVER_ECEF_M[0] + RANGE_M * los[0],
                    RECEIVER_ECEF_M[1] + RANGE_M * los[1],
                    RECEIVER_ECEF_M[2] + RANGE_M * los[2],
                ],
                clock_s: Some(0.0),
            })
        }
    }

    struct ReceiverRelativeSource {
        receiver_ecef_m: [f64; 3],
    }

    impl ObservableEphemerisSource for ReceiverRelativeSource {
        fn observable_state_at_j2000_s(
            &self,
            sat: GnssSatelliteId,
            _t_j2000_s: f64,
        ) -> Result<ObservableState, ObservablesError> {
            let los = final_los(sat);
            Ok(ObservableState {
                position_ecef_m: [
                    self.receiver_ecef_m[0] + RANGE_M * los[0],
                    self.receiver_ecef_m[1] + RANGE_M * los[1],
                    self.receiver_ecef_m[2] + RANGE_M * los[2],
                ],
                clock_s: Some(0.0),
            })
        }
    }

    #[test]
    fn sample_times_includes_partial_end_without_duplicating_exact_end() {
        assert_eq!(
            sample_times((0.0, 25.0), 10).expect("partial window"),
            vec![(0, 0.0), (1, 10.0), (2, 20.0), (3, 25.0)]
        );
        assert_eq!(
            sample_times((0.0, 20.0), 10).expect("exact window"),
            vec![(0, 0.0), (1, 10.0), (2, 20.0)]
        );
    }

    #[test]
    fn partial_window_end_sample_feeds_all_geometry_series() {
        let source = FinalOnlySource {
            visible_from_s: 25.0,
        };
        let sats = [sat(1), sat(2), sat(3), sat(4)];
        let window = (0.0, 25.0);

        let visibility = visibility_series(
            &source,
            &sats,
            RECEIVER_ECEF_M,
            window,
            10,
            &VisibilityOptions::default(),
        )
        .expect("visibility series");
        assert_eq!(
            visibility
                .iter()
                .map(|sample| (sample.step_index, sample.n_visible))
                .collect::<Vec<_>>(),
            [(0, 0), (1, 0), (2, 0), (3, 4)]
        );

        let passes = passes(
            &source,
            &sats,
            RECEIVER_ECEF_M,
            window,
            10,
            &VisibilityOptions::default(),
        )
        .expect("passes");
        assert_eq!(passes.len(), sats.len());
        for pass in &passes {
            assert_eq!(pass.rise_step_index, 3);
            assert_eq!(pass.set_step_index, 3);
            assert_eq!(pass.peak_step_index, 3);
        }

        let dop = dop_series(
            &source,
            &sats,
            None,
            RECEIVER_ECEF_M,
            window,
            10,
            &DopOptions::default(),
        )
        .expect("DOP series");
        assert_eq!(dop.len(), 1);
        assert_eq!(dop[0].step_index, 3);
        assert_eq!(dop[0].geometry.satellites.len(), sats.len());
    }

    #[test]
    fn dop_rejects_non_finite_receiver_coordinates() {
        let source = FinalOnlySource {
            visible_from_s: 0.0,
        };
        let sats = [sat(1), sat(2), sat(3), sat(4)];
        let cases = [
            [f64::NAN, RECEIVER_ECEF_M[1], RECEIVER_ECEF_M[2]],
            [RECEIVER_ECEF_M[0], f64::INFINITY, RECEIVER_ECEF_M[2]],
            [RECEIVER_ECEF_M[0], RECEIVER_ECEF_M[1], f64::NEG_INFINITY],
        ];

        for receiver in cases {
            assert_invalid_receiver(dop_at_epoch(
                &source,
                &sats,
                Some(&sats),
                receiver,
                0.0,
                &DopOptions::default(),
            ));
            assert_invalid_receiver(dop_series(
                &source,
                &sats,
                Some(&sats),
                receiver,
                (0.0, 10.0),
                10,
                &DopOptions::default(),
            ));
        }
    }

    #[test]
    fn dop_handles_antimeridian_receiver_coordinates() {
        let source = ReceiverRelativeSource {
            receiver_ecef_m: ANTI_MERIDIAN_RECEIVER_ECEF_M,
        };
        let sats = [sat(1), sat(2), sat(3), sat(4)];

        let epoch = dop_at_epoch(
            &source,
            &sats,
            Some(&sats),
            ANTI_MERIDIAN_RECEIVER_ECEF_M,
            0.0,
            &DopOptions::default(),
        )
        .expect("antimeridian receiver should produce DOP");
        assert_eq!(epoch.satellites, sats);

        let series = dop_series(
            &source,
            &sats,
            Some(&sats),
            ANTI_MERIDIAN_RECEIVER_ECEF_M,
            (0.0, 0.0),
            10,
            &DopOptions::default(),
        )
        .expect("antimeridian receiver DOP series");
        assert_eq!(series.len(), 1);
    }

    #[test]
    fn geometry_apis_reject_invalid_elevation_masks() {
        let source = FinalOnlySource {
            visible_from_s: 0.0,
        };
        let sats = [sat(1), sat(2), sat(3), sat(4)];
        let invalid_masks = [
            (f64::NAN, "not finite"),
            (f64::INFINITY, "not finite"),
            (-91.0, "out of range"),
            (91.0, "out of range"),
        ];

        for (mask, reason) in invalid_masks {
            let visibility = VisibilityOptions {
                elevation_mask_deg: mask,
                systems: None,
            };
            let dop_options = DopOptions {
                visibility: visibility.clone(),
                weighting: DopWeighting::Unit,
                light_time: false,
            };

            assert_invalid_elevation_mask(
                visible(&source, &sats, RECEIVER_ECEF_M, 0.0, &visibility),
                reason,
            );
            assert_invalid_elevation_mask(
                visibility_series(
                    &source,
                    &sats,
                    RECEIVER_ECEF_M,
                    (0.0, 10.0),
                    10,
                    &visibility,
                ),
                reason,
            );
            assert_invalid_elevation_mask(
                passes(
                    &source,
                    &sats,
                    RECEIVER_ECEF_M,
                    (0.0, 10.0),
                    10,
                    &visibility,
                ),
                reason,
            );
            assert_invalid_elevation_mask(
                dop_at_epoch(
                    &source,
                    &sats,
                    Some(&sats),
                    RECEIVER_ECEF_M,
                    0.0,
                    &dop_options,
                ),
                reason,
            );
            assert_invalid_elevation_mask(
                dop_series(
                    &source,
                    &sats,
                    Some(&sats),
                    RECEIVER_ECEF_M,
                    (0.0, 10.0),
                    10,
                    &dop_options,
                ),
                reason,
            );
        }
    }

    fn sat(prn: u8) -> GnssSatelliteId {
        GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite id")
    }

    fn final_los(sat: GnssSatelliteId) -> [f64; 3] {
        let a = std::f64::consts::FRAC_1_SQRT_2;
        match sat.prn {
            1 => [1.0, 0.0, 0.0],
            2 => [a, a, 0.0],
            3 => [a, -a, 0.0],
            4 => [a, 0.0, a],
            _ => [-1.0, 0.0, 0.0],
        }
    }

    fn assert_invalid_receiver<T>(result: Result<T, DopError>) {
        match result {
            Err(DopError::InvalidInput { field, reason }) => {
                assert_eq!(field, "receiver_ecef_m");
                assert_eq!(reason, "not finite");
            }
            Err(other) => panic!("expected invalid receiver input, got {other:?}"),
            Ok(_) => panic!("expected invalid receiver input"),
        }
    }

    fn assert_invalid_elevation_mask<T>(result: Result<T, DopError>, expected_reason: &str) {
        match result {
            Err(DopError::InvalidInput { field, reason }) => {
                assert_eq!(field, "elevation_mask_deg");
                assert_eq!(reason, expected_reason);
            }
            Err(other) => panic!("expected invalid elevation mask input, got {other:?}"),
            Ok(_) => panic!("expected invalid elevation mask input"),
        }
    }
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests {
    use super::*;
    use crate::observables::j2000_seconds_from_split;
    use crate::sp3::Sp3;
    use serde_json::Value;

    const APPLICATION_GOLDEN: &str =
        include_str!("../tests/fixtures/orbis_gnss_application_golden.json");
    const SPP_TRACE: &str = include_str!("../tests/fixtures/spp_trace_L2_tropo.json");

    fn sp3_fixture() -> Sp3 {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read SP3 fixture {path}: {e}"));
        Sp3::parse(&bytes).expect("parse SP3 fixture")
    }

    fn application_case() -> Value {
        let doc: Value = serde_json::from_str(APPLICATION_GOLDEN).expect("parse golden");
        doc["sp3_application"].clone()
    }

    fn parse_hex_float(s: &str) -> f64 {
        let s = s.strip_prefix("0x").unwrap_or(s);
        let (mantissa, exp_part) = s.split_once('p').expect("hex float exponent");
        let exp: i32 = exp_part.parse().expect("hex float exponent integer");
        let (whole, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        let mut value = u64::from_str_radix(whole, 16).expect("hex float whole") as f64;
        let mut scale = 1.0 / 16.0;
        for c in frac.chars() {
            let digit = c.to_digit(16).expect("hex float fraction digit") as f64;
            value += digit * scale;
            scale /= 16.0;
        }
        value * 2.0_f64.powi(exp)
    }

    fn hexf(value: &Value) -> f64 {
        parse_hex_float(value.as_str().expect("hex float string"))
    }

    fn hex_bits(value: &Value) -> f64 {
        let raw = value.as_str().expect("hex bits string");
        let hex = raw.strip_prefix("0x").unwrap_or(raw);
        f64::from_bits(u64::from_str_radix(hex, 16).expect("hex bits"))
    }

    fn receiver(case: &Value) -> [f64; 3] {
        [
            hexf(&case["receiver_ecef_m"][0]),
            hexf(&case["receiver_ecef_m"][1]),
            hexf(&case["receiver_ecef_m"][2]),
        ]
    }

    fn trace_receiver() -> [f64; 3] {
        let doc: Value = serde_json::from_str(SPP_TRACE).expect("parse SPP trace");
        let truth = &doc["fixture"]["final_solution"]["truth_x"];
        [
            hex_bits(&truth[0]),
            hex_bits(&truth[1]),
            hex_bits(&truth[2]),
        ]
    }

    fn gps_options(mask: f64) -> VisibilityOptions {
        VisibilityOptions {
            elevation_mask_deg: mask,
            systems: Some(BTreeSet::from([GnssSystem::Gps])),
        }
    }

    fn sat(system: GnssSystem, prn: u8) -> GnssSatelliteId {
        GnssSatelliteId::new(system, prn).expect("valid satellite id")
    }

    fn j2000(jd_whole: f64, jd_fraction: f64) -> f64 {
        j2000_seconds_from_split(jd_whole, jd_fraction).expect("valid split Julian date")
    }

    #[test]
    fn visible_gps_mask10_matches_application_golden_bits() {
        let sp3 = sp3_fixture();
        let case = application_case();
        let rx = receiver(&case);
        let t = j2000(2_459_024.5, 0.5);
        let got = visible(&sp3, sp3.satellites(), rx, t, &gps_options(10.0))
            .expect("valid visibility mask");
        let expected = case["visible_gps_mask10"].as_array().expect("visible rows");

        assert_eq!(got.len(), expected.len());
        for (got, want) in got.iter().zip(expected) {
            assert_eq!(got.satellite.to_string(), want["satellite_id"]);
            assert_eq!(
                got.elevation_deg.to_bits(),
                hexf(&want["elevation_deg"]).to_bits()
            );
            assert_eq!(
                got.azimuth_deg.to_bits(),
                hexf(&want["azimuth_deg"]).to_bits()
            );
        }
    }

    #[test]
    fn weighted_dop_matches_application_golden_bits() {
        let sp3 = sp3_fixture();
        let case = application_case();
        let rx = receiver(&case);
        let t = j2000(2_459_024.5, 0.5);
        let dop_case = &case["dop_weighted"];
        let satellites = dop_case["satellites"]
            .as_array()
            .expect("satellites")
            .iter()
            .map(|value| {
                let token = value.as_str().expect("satellite token");
                let prn: u8 = token[1..].parse().expect("satellite PRN");
                sat(GnssSystem::Gps, prn)
            })
            .collect::<Vec<_>>();

        let got = dop_at_epoch(
            &sp3,
            sp3.satellites(),
            Some(&satellites),
            rx,
            t,
            &DopOptions {
                visibility: gps_options(10.0),
                weighting: DopWeighting::Elevation,
                light_time: true,
            },
        )
        .expect("weighted DOP");

        assert_eq!(got.satellites, satellites);
        assert_eq!(got.dop.gdop.to_bits(), hexf(&dop_case["gdop"]).to_bits());
        assert_eq!(got.dop.pdop.to_bits(), hexf(&dop_case["pdop"]).to_bits());
        assert_eq!(got.dop.hdop.to_bits(), hexf(&dop_case["hdop"]).to_bits());
        assert_eq!(got.dop.vdop.to_bits(), hexf(&dop_case["vdop"]).to_bits());
        assert_eq!(got.dop.tdop.to_bits(), hexf(&dop_case["tdop"]).to_bits());
    }

    #[test]
    fn visibility_series_matches_orbis_sampling_counts() {
        let sp3 = sp3_fixture();
        let rx = trace_receiver();
        let window = (j2000(2_459_024.5, 0.5), j2000(2_459_024.5, 0.5) + 3_600.0);

        let got = visibility_series(&sp3, sp3.satellites(), rx, window, 300, &gps_options(5.0))
            .expect("valid visibility step");
        let counts: Vec<usize> = got.iter().map(|sample| sample.n_visible).collect();
        assert_eq!(counts, [9, 9, 9, 9, 10, 11, 11, 11, 11, 11, 11, 11, 11]);
        assert_eq!(
            got.iter()
                .map(|sample| sample.step_index)
                .collect::<Vec<_>>(),
            (0..13).collect::<Vec<_>>()
        );
    }

    #[test]
    fn dop_series_matches_orbis_first_sample_bits() {
        let sp3 = sp3_fixture();
        let rx = trace_receiver();
        let window = (j2000(2_459_024.5, 0.5), j2000(2_459_024.5, 0.5) + 3_600.0);

        let got = dop_series(
            &sp3,
            sp3.satellites(),
            None,
            rx,
            window,
            300,
            &DopOptions {
                visibility: gps_options(5.0),
                weighting: DopWeighting::Unit,
                light_time: false,
            },
        )
        .expect("valid DOP step");

        assert_eq!(got.len(), 13);
        let first = &got[0];
        assert_eq!(first.step_index, 0);
        assert_eq!(
            first
                .geometry
                .satellites
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            ["G21", "G16", "G26", "G20", "G27", "G18", "G10", "G08", "G07"]
        );
        assert_eq!(first.geometry.dop.gdop.to_bits(), 0x4000c042642e3cbc);
        assert_eq!(first.geometry.dop.pdop.to_bits(), 0x3ffd34cde2c7e400);
        assert_eq!(first.geometry.dop.hdop.to_bits(), 0x3ff257e7df379517);
        assert_eq!(first.geometry.dop.vdop.to_bits(), 0x3ff6ba2ad4e284af);
        assert_eq!(first.geometry.dop.tdop.to_bits(), 0x3ff069acbf06750f);
    }

    #[test]
    fn passes_match_orbis_sampled_rise_set_peak_rows() {
        let sp3 = sp3_fixture();
        let rx = trace_receiver();
        let window = (
            j2000(2_459_024.5, 0.0),
            j2000(2_459_024.5, 0.9895833333333334),
        );

        let got = passes(&sp3, sp3.satellites(), rx, window, 900, &gps_options(10.0))
            .expect("valid pass step");
        assert_eq!(got.len(), 51);
        let expected = [
            ("G02", 0, 0, 0, 0x4024d260407442fe),
            ("G05", 0, 10, 0, 0x40513cd3dd1f7866),
            ("G07", 0, 6, 0, 0x4046e04ff1c2a900),
            ("G09", 0, 1, 0, 0x402fdced3853f1fb),
            ("G13", 0, 19, 8, 0x4054b61de01a5608),
            ("G15", 0, 22, 11, 0x4053483acdeec548),
            ("G28", 0, 16, 7, 0x404d9cd49009957c),
            ("G30", 0, 11, 0, 0x4053eb9157f4b766),
        ];
        for (got, (satellite, rise, set, peak, elevation_bits)) in got.iter().zip(expected) {
            assert_eq!(got.satellite.to_string(), satellite);
            assert_eq!(got.rise_step_index, rise);
            assert_eq!(got.set_step_index, set);
            assert_eq!(got.peak_step_index, peak);
            assert_eq!(got.peak_elevation_deg.to_bits(), elevation_bits);
        }
    }

    #[test]
    fn sampled_geometry_rejects_zero_step() {
        let sp3 = sp3_fixture();
        let rx = trace_receiver();
        let window = (j2000(2_459_024.5, 0.5), j2000(2_459_024.5, 0.5) + 3_600.0);

        assert_invalid_geometry_field(
            visibility_series(&sp3, sp3.satellites(), rx, window, 0, &gps_options(5.0))
                .unwrap_err(),
            "step_seconds",
            "not positive",
        );
        assert_invalid_geometry_field(
            dop_series(
                &sp3,
                sp3.satellites(),
                None,
                rx,
                window,
                0,
                &DopOptions::default(),
            )
            .unwrap_err(),
            "step_seconds",
            "not positive",
        );
        assert_invalid_geometry_field(
            passes(&sp3, sp3.satellites(), rx, window, 0, &gps_options(10.0)).unwrap_err(),
            "step_seconds",
            "not positive",
        );
    }

    #[test]
    fn sampled_geometry_rejects_non_finite_window_bounds() {
        let sp3 = sp3_fixture();
        let rx = trace_receiver();
        let t = j2000(2_459_024.5, 0.5);
        let cases = [
            ((f64::NAN, t + 300.0), "window_j2000_s.0"),
            ((f64::NEG_INFINITY, t + 300.0), "window_j2000_s.0"),
            ((t, f64::INFINITY), "window_j2000_s.1"),
        ];

        for (window, field) in cases {
            assert_invalid_geometry_field(
                visibility_series(&sp3, sp3.satellites(), rx, window, 300, &gps_options(5.0))
                    .unwrap_err(),
                field,
                "not finite",
            );
            assert_invalid_geometry_field(
                dop_series(
                    &sp3,
                    sp3.satellites(),
                    None,
                    rx,
                    window,
                    300,
                    &DopOptions::default(),
                )
                .unwrap_err(),
                field,
                "not finite",
            );
            assert_invalid_geometry_field(
                passes(&sp3, sp3.satellites(), rx, window, 300, &gps_options(10.0)).unwrap_err(),
                field,
                "not finite",
            );
        }
    }

    fn assert_invalid_geometry_field(
        error: DopError,
        expected: &'static str,
        expected_reason: &'static str,
    ) {
        match error {
            DopError::InvalidInput { field, reason } => {
                assert_eq!(field, expected);
                assert_eq!(reason, expected_reason);
            }
            other => panic!("expected invalid geometry input for {expected}, got {other:?}"),
        }
    }
}
