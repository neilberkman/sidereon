//! TLE pass prediction over a ground station.
//!
//! This module owns the legacy Sidereon pass orchestration ([`predict_passes`]:
//! coarse elevation sampling, horizon-crossing bisection, peak-elevation search)
//! and the event-finder-backed pass finder ([`find_passes`]: elevation-mask
//! crossings refined by the shared event finder, and culmination through the
//! shared extrema finder). The
//! low-level propagation and frame transforms stay delegated to the core SGP4
//! and frames modules.

use crate::astro::constants::units::MICROSECONDS_PER_SECOND_I64;
use crate::astro::constants::{
    earth::{GM_EARTH_KM3_S2, OMEGA_E_DOT_RAD_S},
    time::{MICROSECONDS_PER_DAY_I64, SECONDS_PER_DAY, SECONDS_PER_DAY_I64},
};
use crate::astro::events::root::{
    bisect_crossing_by_iterations, bisect_crossing_until, sign_change_bracketed,
};
use crate::astro::events::{
    CrossingDirection, CrossingEvent, EventFinder, EventFinderError, ExtremumEvent, ExtremumKind,
    ScalarEventPredicate,
};
use crate::astro::frames::transforms::{
    gcrs_to_topocentric_compute, geodetic_to_itrs, teme_to_gcrs_compute, FrameTransformError,
    GeodeticStationKm, TemeStateKm,
};
use crate::astro::sgp4::{
    ElementSet, Error as Sgp4Error, JulianDate, OpsMode, Prediction, Satellite,
};
use crate::astro::time::civil::civil_from_julian_day_number;
use crate::astro::time::scales::{julian_day_number, TimeScales};
use crate::validate;
use rayon::prelude::*;

const UNIX_EPOCH_JDN: i64 = 2_440_588;

const BISECT_ITERATIONS: usize = 20;
const GOLDEN_ITERATIONS: usize = 30;
const GOLDEN_RESPHI: f64 = 0.381_966_011_250_105_1;
const ROBUST_CROSSING_FAST_SAMPLES_PER_FASTEST_REV: f64 = 360.0;
const ROBUST_CROSSING_SLOW_SAMPLES_PER_FASTEST_REV: f64 = 48.0;
const ROBUST_CROSSING_SLOW_ORBIT_SECONDS: f64 = 12.0 * 60.0 * 60.0;
const ROBUST_CROSSING_MASK_DWELL_SAMPLES: f64 = 4.0;
const EVENT_FINDER_COARSE_SAMPLE_LIMIT: i64 = 1_000_000;

/// UTC instant represented as unix microseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct UtcInstant {
    unix_microseconds: i64,
}

impl UtcInstant {
    /// Construct from unix microseconds.
    pub fn from_unix_microseconds(unix_microseconds: i64) -> Self {
        Self { unix_microseconds }
    }

    /// Construct from UTC calendar fields.
    ///
    /// This type stores a POSIX-like Unix microsecond count, so leap-second
    /// labels cannot be represented distinctly from the following midnight.
    pub fn from_utc(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: i32,
        microsecond: i32,
    ) -> Option<Self> {
        if !(0..=999_999).contains(&microsecond) {
            return None;
        }
        validate::civil_datetime_with_second_policy(
            i64::from(year),
            i64::from(month),
            i64::from(day),
            i64::from(hour),
            i64::from(minute),
            f64::from(second),
            validate::CivilSecondPolicy::UtcLike,
        )
        .ok()?;
        if second == 60 {
            return None;
        }

        let days = julian_day_number(year, month, day) - UNIX_EPOCH_JDN;
        let seconds_of_day = hour as i64 * 3600 + minute as i64 * 60 + second as i64;
        Some(Self {
            unix_microseconds: days * MICROSECONDS_PER_DAY_I64
                + seconds_of_day * MICROSECONDS_PER_SECOND_I64
                + microsecond as i64,
        })
    }

    /// Unix microseconds.
    pub fn unix_microseconds(self) -> i64 {
        self.unix_microseconds
    }

    // Time arithmetic saturates rather than panicking (debug) or wrapping
    // (release). `UtcInstant` is public and constructible at the i64 extremes, so
    // an out-of-range window must degrade gracefully; the public entry points
    // additionally reject an unrepresentable span up front via
    // [`validate_pass_window`]. For every in-range window these are exact, so no
    // golden trajectory changes.
    fn add_microseconds(self, delta: i64) -> Self {
        Self {
            unix_microseconds: self.unix_microseconds.saturating_add(delta),
        }
    }

    fn diff_microseconds(self, earlier: Self) -> i64 {
        self.unix_microseconds
            .saturating_sub(earlier.unix_microseconds)
    }

    /// Microsecond span `self - earlier`, or `None` if it overflows `i64`.
    fn checked_diff_microseconds(self, earlier: Self) -> Option<i64> {
        self.unix_microseconds
            .checked_sub(earlier.unix_microseconds)
    }

    fn diff_seconds(self, earlier: Self) -> i64 {
        self.diff_microseconds(earlier) / MICROSECONDS_PER_SECOND_I64
    }

    fn components(self) -> UtcComponents {
        let seconds = div_floor(self.unix_microseconds, MICROSECONDS_PER_SECOND_I64);
        let microsecond = rem_floor(self.unix_microseconds, MICROSECONDS_PER_SECOND_I64);
        let days = div_floor(seconds, SECONDS_PER_DAY_I64);
        let second_of_day = seconds - days * SECONDS_PER_DAY_I64;
        let (year, month, day) = civil_from_days(days);

        UtcComponents {
            year,
            month,
            day,
            hour: (second_of_day / 3600) as i32,
            minute: ((second_of_day % 3600) / 60) as i32,
            second: (second_of_day % 60) as i32,
            microsecond: microsecond as i32,
        }
    }

    /// Resolve the precise split-Julian-date time scales (UT1/TT/TDB) for this
    /// UTC instant, via the parity-critical [`TimeScales::from_utc`] path.
    ///
    /// Exposed so a Rust or Python consumer can reach the precise time scales for
    /// an instant given only its unix-microsecond UTC stamp, without restating the
    /// calendar breakdown. The numerics are unchanged from the internal pass-finder
    /// use.
    pub fn time_scales(self) -> TimeScales {
        let c = self.components();
        TimeScales::from_utc(
            c.year,
            c.month,
            c.day,
            c.hour,
            c.minute,
            c.second as f64 + c.microsecond as f64 / 1_000_000.0,
        )
        .expect("UtcInstant components produce a finite UTC second")
    }

    fn sgp4_julian_date(self) -> JulianDate {
        let c = self.components();
        let jdn = julian_day_number(c.year, c.month, c.day);
        let jd_midnight = jdn as f64 - 0.5;
        let frac = (c.hour as f64) / 24.0
            + (c.minute as f64) / 1440.0
            + (c.second as f64) / SECONDS_PER_DAY
            + (c.microsecond as f64) / MICROSECONDS_PER_DAY_I64 as f64;
        JulianDate(jd_midnight, frac)
    }
}

#[derive(Debug, Clone, Copy)]
struct UtcComponents {
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: i32,
    microsecond: i32,
}

/// Ground-station geodetic coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundStation {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub altitude_m: f64,
}

/// Pass-prediction options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassPredictionOptions {
    pub min_elevation_deg: f64,
    pub step_seconds: i64,
}

impl Default for PassPredictionOptions {
    fn default() -> Self {
        Self {
            min_elevation_deg: 0.0,
            step_seconds: 60,
        }
    }
}

/// Predicted visible pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictedPass {
    pub rise: UtcInstant,
    pub set: UtcInstant,
    pub max_elevation_deg: f64,
    pub max_elevation_time: UtcInstant,
}

/// Error while planning a TLE-backed pass arc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PassError {
    #[error("invalid pass input {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

/// Topocentric look angle from a ground station to a TLE satellite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LookAngle {
    /// Topocentric azimuth in `[0, 360)`, degrees.
    ///
    /// At (and arbitrarily near) the station zenith the azimuth is geometrically
    /// undefined; it is defined here to be exactly `0.0` once the horizontal
    /// line-of-sight projection falls below
    /// [`crate::constants::AZIMUTH_ZENITH_EPS`], rather than returning rounding
    /// noise or erroring.
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    pub range_km: f64,
}

/// One member of a TLE-backed constellation.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstellationMember {
    pub catalog_number: String,
    pub elements: ElementSet,
}

/// One satellite visible from a ground station at an instant.
#[derive(Debug, Clone, PartialEq)]
pub struct VisibleSatellite {
    pub catalog_number: String,
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    pub range_km: f64,
    pub position_km: [f64; 3],
}

/// Error while computing a TLE look angle.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LookAngleError {
    #[error("invalid look-angle input {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    #[error("SGP4 initialization failed: {0}")]
    Init(crate::astro::sgp4::Error),
    #[error("SGP4 propagation failed: {0}")]
    Propagate(crate::astro::sgp4::Error),
    #[error("look-angle frame transform failed: {0}")]
    FrameTransform(#[from] FrameTransformError),
}

/// Propagate a pre-parsed SGP4 element set and compute its topocentric look angle.
pub fn look_angle(
    elements: &ElementSet,
    ground_station: GroundStation,
    datetime: UtcInstant,
) -> Result<LookAngle, LookAngleError> {
    validate_ground_station(ground_station).map_err(map_look_angle_input)?;
    let ts = time_scales_for_look_angle(datetime)?;
    let satellite = Satellite::from_elements_with_opsmode(elements, OpsMode::Afspc)
        .map_err(LookAngleError::Init)?;
    let pred = satellite
        .propagate_jd(datetime.sgp4_julian_date())
        .map_err(LookAngleError::Propagate)?;
    look_angle_from_teme_prediction(&pred, &ts, ground_station)
}

/// Propagate one already-initialized SGP4 satellite to TEME position/velocity at
/// each UTC instant.
///
/// The satellite (its `satrec`) is built once by the caller, so this steps the
/// pure propagation kernel over the epoch grid without re-running `sgp4init` per
/// epoch. Each [`Prediction`] is TEME position (km) and velocity (km/s), bit-for-bit
/// identical to calling [`Satellite::propagate_jd`] at the same instant. The first
/// propagation error aborts the arc.
pub fn propagate_teme_arc(
    satellite: &Satellite,
    datetimes: &[UtcInstant],
) -> Result<Vec<Prediction>, Sgp4Error> {
    datetimes
        .iter()
        .map(|&datetime| satellite.propagate_jd(datetime.sgp4_julian_date()))
        .collect()
}

/// Topocentric look angle from a ground station to one already-initialized SGP4
/// satellite at each UTC instant.
///
/// The satellite is built once by the caller; this steps it over the epoch grid
/// and runs the same TEME-to-topocentric path as [`look_angle`], so each
/// [`LookAngle`] is bit-for-bit identical to a per-epoch [`look_angle`] call. The
/// first propagation error aborts the arc.
pub fn look_angle_arc(
    satellite: &Satellite,
    ground_station: GroundStation,
    datetimes: &[UtcInstant],
) -> Result<Vec<LookAngle>, LookAngleError> {
    validate_ground_station(ground_station).map_err(map_look_angle_input)?;
    datetimes
        .iter()
        .map(|&datetime| {
            let ts = time_scales_for_look_angle(datetime)?;
            let pred = satellite
                .propagate_jd(datetime.sgp4_julian_date())
                .map_err(LookAngleError::Propagate)?;
            look_angle_from_teme_prediction(&pred, &ts, ground_station)
        })
        .collect()
}

/// Propagate many already-initialized SGP4 satellites over a shared epoch grid,
/// serially.
///
/// Element `i` of the result is the TEME arc for `satellites[i]` over
/// `datetimes`, exactly as [`propagate_teme_arc`] would produce it on its own
/// (the first propagation error in a satellite's arc becomes that element's
/// `Err`). This is the single-threaded reference the parallel
/// [`propagate_teme_batch_parallel`] is proven bit-identical against.
pub fn propagate_teme_batch_serial(
    satellites: &[Satellite],
    datetimes: &[UtcInstant],
) -> Vec<Result<Vec<Prediction>, Sgp4Error>> {
    satellites
        .iter()
        .map(|satellite| propagate_teme_arc(satellite, datetimes))
        .collect()
}

/// Propagate many already-initialized SGP4 satellites over a shared epoch grid,
/// fanning the independent per-satellite arcs across a rayon thread pool.
///
/// Each satellite's arc is computed by the same serial [`propagate_teme_arc`]
/// kernel and the indexed parallel collect preserves input order, so element `i`
/// is byte-for-byte identical to element `i` of
/// [`propagate_teme_batch_serial`]: no reduction, no cross-satellite state, and
/// no reordering inside a single SGP4 propagation. The work is embarrassingly
/// parallel (satellites are independent), so throughput scales with cores while
/// every value stays bit-exact.
pub fn propagate_teme_batch_parallel(
    satellites: &[Satellite],
    datetimes: &[UtcInstant],
) -> Vec<Result<Vec<Prediction>, Sgp4Error>> {
    satellites
        .par_iter()
        .map(|satellite| propagate_teme_arc(satellite, datetimes))
        .collect()
}

/// Topocentric look angles for many already-initialized SGP4 satellites over a
/// shared epoch grid, serially.
///
/// Element `i` is the look-angle arc for `satellites[i]` from `ground_station`,
/// exactly as [`look_angle_arc`] would produce it. The serial reference for
/// [`look_angle_batch_parallel`].
pub fn look_angle_batch_serial(
    satellites: &[Satellite],
    ground_station: GroundStation,
    datetimes: &[UtcInstant],
) -> Vec<Result<Vec<LookAngle>, LookAngleError>> {
    satellites
        .iter()
        .map(|satellite| look_angle_arc(satellite, ground_station, datetimes))
        .collect()
}

/// Topocentric look angles for many already-initialized SGP4 satellites over a
/// shared epoch grid, fanned across a rayon thread pool.
///
/// Each satellite's arc is computed by the same serial [`look_angle_arc`] kernel
/// and the indexed parallel collect preserves input order, so element `i` is
/// byte-for-byte identical to element `i` of [`look_angle_batch_serial`].
pub fn look_angle_batch_parallel(
    satellites: &[Satellite],
    ground_station: GroundStation,
    datetimes: &[UtcInstant],
) -> Vec<Result<Vec<LookAngle>, LookAngleError>> {
    satellites
        .par_iter()
        .map(|satellite| look_angle_arc(satellite, ground_station, datetimes))
        .collect()
}

/// Sub-satellite (ground-track) geodetic points for one already-initialized
/// satellite over a time grid.
///
/// For each instant this composes the existing core transforms - propagate to
/// TEME ([`Satellite::propagate_jd`]), TEME->GCRS ([`teme_to_gcrs_compute`]),
/// GCRS->ITRS/ECEF ([`gcrs_to_itrs_compute`]), then ECEF->geodetic
/// ([`itrs_to_geodetic_compute`]) - and returns the WGS84 sub-point (geodetic
/// latitude/longitude and ellipsoidal height). No geometry is reinvented; the
/// same TEME->GCRS path [`look_angle`] uses feeds an ECEF step and the shared
/// geodetic reduction. Like [`look_angle_arc`], the first propagation or frame
/// error aborts the whole arc.
pub fn ground_track(
    satellite: &Satellite,
    datetimes: &[UtcInstant],
) -> Result<Vec<crate::frame::Wgs84Geodetic>, LookAngleError> {
    use crate::astro::frames::transforms::{gcrs_to_itrs_compute, itrs_to_geodetic_compute};
    use crate::frame::Wgs84Geodetic;

    datetimes
        .iter()
        .map(|&datetime| {
            let ts = time_scales_for_look_angle(datetime)?;
            let pred = satellite
                .propagate_jd(datetime.sgp4_julian_date())
                .map_err(LookAngleError::Propagate)?;
            let (gcrs_position, _) = teme_to_gcrs_compute(
                &TemeStateKm {
                    position_km: pred.position,
                    velocity_km_s: pred.velocity,
                },
                &ts,
                false,
            )?;
            let (x_km, y_km, z_km) = gcrs_to_itrs_compute(
                gcrs_position.0,
                gcrs_position.1,
                gcrs_position.2,
                &ts,
                false,
            )?;
            let (lat_deg, lon_deg, alt_km) = itrs_to_geodetic_compute(x_km, y_km, z_km)?;
            Wgs84Geodetic::new(lat_deg.to_radians(), lon_deg.to_radians(), alt_km * 1000.0)
                .map_err(map_frame_value_to_look_angle)
        })
        .collect()
}

fn map_frame_value_to_look_angle(error: crate::frame::FrameValueError) -> LookAngleError {
    match error {
        crate::frame::FrameValueError::InvalidInput { field, reason } => {
            LookAngleError::InvalidInput { field, reason }
        }
    }
}

/// Find constellation members above an elevation threshold at one instant.
///
/// Invalid element sets or per-satellite propagation failures are skipped,
/// matching the legacy Sidereon `visible_from/4` behavior through
/// `propagate_all/2`.
pub fn visible_from_constellation(
    members: &[ConstellationMember],
    ground_station: GroundStation,
    datetime: UtcInstant,
    min_elevation_deg: f64,
) -> Result<Vec<VisibleSatellite>, PassError> {
    validate_ground_station(ground_station).map_err(map_pass_input)?;
    validate_elevation_threshold(min_elevation_deg, "min_elevation_deg").map_err(map_pass_input)?;
    let ts = time_scales_for_pass(datetime)?;

    let mut visible = Vec::new();

    for member in members {
        let satellite =
            match Satellite::from_elements_with_opsmode(&member.elements, OpsMode::Afspc) {
                Ok(satellite) => satellite,
                Err(_) => continue,
            };
        let pred = match satellite.propagate_jd(datetime.sgp4_julian_date()) {
            Ok(pred) => pred,
            Err(_) => continue,
        };
        let look = match look_angle_from_teme_prediction(&pred, &ts, ground_station) {
            Ok(look) => look,
            Err(_) => continue,
        };

        if look.elevation_deg >= min_elevation_deg {
            visible.push(VisibleSatellite {
                catalog_number: member.catalog_number.clone(),
                azimuth_deg: look.azimuth_deg,
                elevation_deg: look.elevation_deg,
                range_km: look.range_km,
                position_km: pred.position,
            });
        }
    }

    visible.sort_by(|a, b| {
        b.elevation_deg
            .partial_cmp(&a.elevation_deg)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(visible)
}

/// Find visible satellites above an elevation threshold at one instant, from
/// already-initialized [`Satellite`]s.
///
/// Unlike [`visible_from_constellation`], which rebuilds each satellite with a
/// hardcoded [`OpsMode::Afspc`] from raw element sets, this variant operates on
/// satellites the caller built (e.g. via [`crate::astro::sgp4::parse_tle_file`]),
/// so each satellite's own opsmode is honored end-to-end: the result for a
/// deep-space / opsmode-sensitive object differs between an `Afspc`-built and an
/// `Improved`-built satellite, exactly as their propagation does.
///
/// Identity is supplied out-of-band as a parallel `ids` slice (one id per
/// satellite, same order); `ids[i]` becomes [`VisibleSatellite::catalog_number`]
/// for `satellites[i]`. This keeps [`VisibleSatellite`] faithful (the caller
/// chooses whether the id is a NORAD number, a name, or anything else) and
/// mirrors the `&[Satellite]` convention of [`look_angle_batch_serial`]. The
/// slices must have equal length.
///
/// The geometry reuses the exact single-instant TEME-to-topocentric path of
/// [`look_angle`] / [`look_angle_arc`]. Per-satellite propagation or frame
/// failures are skipped (matching [`visible_from_constellation`]); the result is
/// filtered by `min_elevation_deg` and sorted by elevation descending.
pub fn visible_from_satellites(
    satellites: &[Satellite],
    ids: &[String],
    ground_station: GroundStation,
    datetime: UtcInstant,
    min_elevation_deg: f64,
) -> Result<Vec<VisibleSatellite>, PassError> {
    validate_ground_station(ground_station).map_err(map_pass_input)?;
    validate_elevation_threshold(min_elevation_deg, "min_elevation_deg").map_err(map_pass_input)?;
    if ids.len() != satellites.len() {
        return Err(invalid_pass_input("ids", "must have one id per satellite"));
    }
    let ts = time_scales_for_pass(datetime)?;

    let mut visible = Vec::new();

    for (satellite, id) in satellites.iter().zip(ids) {
        let pred = match satellite.propagate_jd(datetime.sgp4_julian_date()) {
            Ok(pred) => pred,
            Err(_) => continue,
        };
        let look = match look_angle_from_teme_prediction(&pred, &ts, ground_station) {
            Ok(look) => look,
            Err(_) => continue,
        };

        if look.elevation_deg >= min_elevation_deg {
            visible.push(VisibleSatellite {
                catalog_number: id.clone(),
                azimuth_deg: look.azimuth_deg,
                elevation_deg: look.elevation_deg,
                range_km: look.range_km,
                position_km: pred.position,
            });
        }
    }

    visible.sort_by(|a, b| {
        b.elevation_deg
            .partial_cmp(&a.elevation_deg)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(visible)
}

/// Predict visible passes for a pre-parsed SGP4 element set.
///
/// Invalid element sets or per-sample propagation failures are treated as
/// below-horizon samples, matching the legacy Sidereon public behavior.
pub fn predict_passes(
    elements: &ElementSet,
    ground_station: GroundStation,
    start_time: UtcInstant,
    end_time: UtcInstant,
    options: PassPredictionOptions,
) -> Result<Vec<PredictedPass>, PassError> {
    predict_passes_with_opsmode(
        elements,
        ground_station,
        start_time,
        end_time,
        options,
        OpsMode::Afspc,
    )
}

/// [`predict_passes`] with an explicit SGP4 [`OpsMode`].
///
/// [`predict_passes`] initializes SGP4 with [`OpsMode::Afspc`] (its established
/// default, which the committed pass goldens are pinned to). The rest of the
/// crate (`parse_tle_file`, [`Satellite::from_elements`], `propagate_elements`)
/// defaults to [`OpsMode::Improved`], so the same TLE can otherwise yield a
/// different trajectory by path. Pass [`OpsMode::Improved`] here to make a pass
/// prediction consistent with those paths.
pub fn predict_passes_with_opsmode(
    elements: &ElementSet,
    ground_station: GroundStation,
    start_time: UtcInstant,
    end_time: UtcInstant,
    options: PassPredictionOptions,
    opsmode: OpsMode,
) -> Result<Vec<PredictedPass>, PassError> {
    validate_ground_station(ground_station).map_err(map_pass_input)?;
    let step_seconds = validate_pass_prediction_options(options)?;
    validate_pass_window(start_time, end_time)?;

    let satellite = match Satellite::from_elements_with_opsmode(elements, opsmode) {
        Ok(satellite) => satellite,
        Err(_) => return Ok(Vec::new()),
    };

    let samples = coarse_scan(
        &satellite,
        ground_station,
        start_time,
        end_time,
        step_seconds,
    );

    Ok(extract_passes(&samples, &satellite, ground_station)
        .into_iter()
        .filter(|pass| pass.max_elevation_deg >= options.min_elevation_deg)
        .collect())
}

fn coarse_scan(
    satellite: &Satellite,
    ground_station: GroundStation,
    start_time: UtcInstant,
    end_time: UtcInstant,
    step_seconds: i64,
) -> Vec<(UtcInstant, f64)> {
    let total_seconds = end_time.diff_seconds(start_time);
    let num_steps = (total_seconds / step_seconds).max(0);

    let mut samples: Vec<(UtcInstant, f64)> = (0..=num_steps)
        .map(|i| {
            let dt = start_time.add_microseconds(i * step_seconds * MICROSECONDS_PER_SECOND_I64);
            (dt, elevation_at(satellite, dt, ground_station))
        })
        .collect();

    if let Some(&(last_dt, _)) = samples.last() {
        if end_time.diff_microseconds(last_dt) > 0 {
            samples.push((end_time, elevation_at(satellite, end_time, ground_station)));
        }
    }

    samples
}

fn extract_passes(
    samples: &[(UtcInstant, f64)],
    satellite: &Satellite,
    ground_station: GroundStation,
) -> Vec<PredictedPass> {
    let mut rise_time = match samples.first() {
        Some((dt, el)) if *el >= 0.0 => Some(*dt),
        _ => None,
    };
    let mut passes = Vec::new();

    for pair in samples.windows(2) {
        let (dt_a, el_a) = pair[0];
        let (dt_b, el_b) = pair[1];

        if rise_time.is_none() && el_a < 0.0 && el_b >= 0.0 {
            rise_time = Some(bisect_crossing(satellite, ground_station, dt_a, dt_b));
        } else if let Some(rise) = rise_time {
            if el_a >= 0.0 && el_b < 0.0 {
                let set = bisect_crossing(satellite, ground_station, dt_a, dt_b);
                passes.push(build_pass(satellite, ground_station, rise, set));
                rise_time = None;
            }
        }
    }

    passes
}

fn bisect_crossing(
    satellite: &Satellite,
    ground_station: GroundStation,
    dt_low: UtcInstant,
    dt_high: UtcInstant,
) -> UtcInstant {
    bisect_crossing_by_iterations(
        dt_low,
        dt_high,
        BISECT_ITERATIONS,
        |dt| elevation_at(satellite, dt, ground_station),
        midpoint_instant,
    )
    .unwrap_or(dt_low)
}

fn midpoint_instant(a: UtcInstant, b: UtcInstant) -> UtcInstant {
    a.add_microseconds(b.diff_microseconds(a) / 2)
}

fn build_pass(
    satellite: &Satellite,
    ground_station: GroundStation,
    rise: UtcInstant,
    set: UtcInstant,
) -> PredictedPass {
    let (max_elevation_deg, max_elevation_time) =
        find_max_elevation(satellite, ground_station, rise, set);

    PredictedPass {
        rise,
        set,
        max_elevation_deg,
        max_elevation_time,
    }
}

fn find_max_elevation(
    satellite: &Satellite,
    ground_station: GroundStation,
    rise: UtcInstant,
    set: UtcInstant,
) -> (f64, UtcInstant) {
    let total_us = set.diff_microseconds(rise);
    let mut a = 0_i64;
    let mut b = total_us;

    for _ in 0..GOLDEN_ITERATIONS {
        let span = b - a;
        let x1 = (a as f64 + GOLDEN_RESPHI * span as f64).round() as i64;
        let x2 = (b as f64 - GOLDEN_RESPHI * span as f64).round() as i64;

        let dt1 = rise.add_microseconds(x1);
        let dt2 = rise.add_microseconds(x2);
        let el1 = elevation_at(satellite, dt1, ground_station);
        let el2 = elevation_at(satellite, dt2, ground_station);

        if el1 > el2 {
            b = x2;
        } else {
            a = x1;
        }
    }

    let best_us = (a + b) / 2;
    let best_dt = rise.add_microseconds(best_us);
    let best_el = elevation_at(satellite, best_dt, ground_station);
    (best_el, best_dt)
}

fn elevation_at(satellite: &Satellite, datetime: UtcInstant, ground_station: GroundStation) -> f64 {
    let pred = match satellite.propagate_jd(datetime.sgp4_julian_date()) {
        Ok(pred) => pred,
        Err(_) => return -90.0,
    };
    let ts = match time_scales_for_look_angle(datetime) {
        Ok(ts) => ts,
        Err(_) => return -90.0,
    };
    match look_angle_from_teme_prediction(&pred, &ts, ground_station) {
        Ok(look) => look.elevation_deg,
        Err(_) => -90.0,
    }
}

fn look_angle_from_teme_prediction(
    pred: &Prediction,
    ts: &TimeScales,
    ground_station: GroundStation,
) -> Result<LookAngle, LookAngleError> {
    let (gcrs_position, _) = teme_to_gcrs_compute(
        &TemeStateKm {
            position_km: [pred.position[0], pred.position[1], pred.position[2]],
            velocity_km_s: [pred.velocity[0], pred.velocity[1], pred.velocity[2]],
        },
        ts,
        false,
    )?;
    let (azimuth, elevation, range) = gcrs_to_topocentric_compute(
        [gcrs_position.0, gcrs_position.1, gcrs_position.2],
        &GeodeticStationKm {
            latitude_deg: ground_station.latitude_deg,
            longitude_deg: ground_station.longitude_deg,
            altitude_km: ground_station.altitude_m / 1000.0,
        },
        ts,
        false,
    )?;
    validate_look_angle(LookAngle {
        azimuth_deg: azimuth,
        elevation_deg: elevation,
        range_km: range,
    })
}

/// Civil `(year, month, day)` from a day count relative to 1970-01-01, via the
/// canonical inverse Julian Day Number (the Unix epoch is JDN
/// [`UNIX_EPOCH_JDN`]).
fn civil_from_days(days_since_unix_epoch: i64) -> (i32, i32, i32) {
    let (year, month, day) = civil_from_julian_day_number(days_since_unix_epoch + UNIX_EPOCH_JDN);
    (year as i32, month as i32, day as i32)
}

fn div_floor(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

fn rem_floor(a: i64, b: i64) -> i64 {
    a.rem_euclid(b)
}

/// Half-step (microseconds) of the central difference used to snap an extrema
/// culmination to the existing pass-finder microsecond convention.
const CULMINATION_RATE_HALF_STEP_US: i64 = 1_000_000;

/// Options for the event-finder-backed pass finder [`find_passes`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassFinderOptions {
    /// Elevation mask (degrees). AOS/LOS are the times the satellite crosses
    /// this elevation, not merely the geometric horizon.
    pub elevation_mask_deg: f64,
    /// Requested maximum sampling step (seconds) for bracketing mask crossings.
    /// The pass finder may sub-step below this based on orbital geometry so
    /// short or grazing passes are not skipped by a too-coarse caller step.
    pub coarse_step_seconds: f64,
    /// Time tolerance (seconds) to which each crossing and the culmination are
    /// refined by bisection.
    pub time_tolerance_seconds: f64,
}

impl Default for PassFinderOptions {
    fn default() -> Self {
        Self {
            elevation_mask_deg: 0.0,
            coarse_step_seconds: 30.0,
            time_tolerance_seconds: 1.0e-3,
        }
    }
}

/// A satellite pass over a ground station: acquisition of signal (rise above the
/// mask), loss of signal (set below the mask), the culmination (max-elevation)
/// time, and the elevation at culmination.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SatellitePass {
    /// Acquisition of signal: the satellite rises above the elevation mask.
    pub aos: UtcInstant,
    /// Loss of signal: the satellite sets below the elevation mask.
    pub los: UtcInstant,
    /// Culmination: the time of maximum elevation within the pass.
    pub culmination: UtcInstant,
    /// Elevation at culmination, degrees.
    pub max_elevation_deg: f64,
}

/// Find satellite passes over a ground station through the shared event finder.
///
/// The elevation function is passed to [`EventFinder::find_crossings`] across
/// the window, and each AOS/LOS crossing is refined by the shared event finder
/// to [`PassFinderOptions::time_tolerance_seconds`]. The culmination is the
/// maximum elevation returned by [`EventFinder::find_extrema`] between AOS and
/// LOS, with endpoints considered for partial passes.
///
/// A pass is emitted only when both its AOS and LOS fall inside the window; a
/// satellite already above the mask at the window start has its AOS clamped to
/// the start, mirroring [`predict_passes`]. Invalid element sets or per-sample
/// propagation failures are treated as below-mask samples.
pub fn find_passes(
    elements: &ElementSet,
    ground_station: GroundStation,
    start_time: UtcInstant,
    end_time: UtcInstant,
    options: PassFinderOptions,
) -> Result<Vec<SatellitePass>, PassError> {
    find_passes_with_opsmode(
        elements,
        ground_station,
        start_time,
        end_time,
        options,
        OpsMode::Afspc,
    )
}

/// [`find_passes`] with an explicit SGP4 [`OpsMode`].
///
/// [`find_passes`] initializes SGP4 with [`OpsMode::Afspc`] (the established
/// default the committed pass goldens are pinned to); the rest of the crate
/// defaults to [`OpsMode::Improved`]. Pass [`OpsMode::Improved`] here to make a
/// pass search consistent with `parse_tle_file` /
/// [`Satellite::from_elements`]. To reuse an already-initialized satellite (any
/// opsmode), use [`find_passes_for_satellite`].
pub fn find_passes_with_opsmode(
    elements: &ElementSet,
    ground_station: GroundStation,
    start_time: UtcInstant,
    end_time: UtcInstant,
    options: PassFinderOptions,
    opsmode: OpsMode,
) -> Result<Vec<SatellitePass>, PassError> {
    validate_ground_station(ground_station).map_err(map_pass_input)?;
    let validated_options = validate_pass_finder_options(options)?;
    // Validate the window before SGP4 init so an unrepresentable window is a typed
    // error regardless of element validity, matching `predict_passes_with_opsmode`
    // (an invalid element set otherwise short-circuits to Ok(empty) first).
    validate_pass_window(start_time, end_time)?;
    let satellite = match Satellite::from_elements_with_opsmode(elements, opsmode) {
        Ok(satellite) => satellite,
        Err(_) => return Ok(Vec::new()),
    };

    find_passes_for_satellite_validated(
        &satellite,
        ground_station,
        start_time,
        end_time,
        validated_options,
    )
}

/// Find satellite passes for many pre-parsed SGP4 element sets, serially.
///
/// Result element `i` belongs to `elements[i]`. Valid satellites are scanned
/// independently with the same per-satellite robust step used by
/// [`find_passes`]. Invalid element sets produce `Ok(Vec::new())`, matching
/// [`find_passes`].
pub fn find_passes_batch_serial(
    elements: &[ElementSet],
    ground_station: GroundStation,
    start_time: UtcInstant,
    end_time: UtcInstant,
    options: PassFinderOptions,
) -> Vec<Result<Vec<SatellitePass>, PassError>> {
    find_passes_batch_serial_with_opsmode(
        elements,
        ground_station,
        start_time,
        end_time,
        options,
        OpsMode::Afspc,
    )
}

/// [`find_passes_batch_serial`] with an explicit SGP4 [`OpsMode`] (see
/// [`find_passes_with_opsmode`] for why the bare function uses
/// [`OpsMode::Afspc`]).
pub fn find_passes_batch_serial_with_opsmode(
    elements: &[ElementSet],
    ground_station: GroundStation,
    start_time: UtcInstant,
    end_time: UtcInstant,
    options: PassFinderOptions,
    opsmode: OpsMode,
) -> Vec<Result<Vec<SatellitePass>, PassError>> {
    find_passes_batch(
        elements,
        ground_station,
        start_time,
        end_time,
        options,
        PassBatchMode::Serial,
        opsmode,
    )
}

/// Find satellite passes for many pre-parsed SGP4 element sets in parallel.
///
/// This parallelizes independent per-satellite scans. Indexed collection
/// preserves input order, so result element `i` belongs to `elements[i]` and is
/// bit-identical to [`find_passes_batch_serial`] for the same inputs.
pub fn find_passes_batch_parallel(
    elements: &[ElementSet],
    ground_station: GroundStation,
    start_time: UtcInstant,
    end_time: UtcInstant,
    options: PassFinderOptions,
) -> Vec<Result<Vec<SatellitePass>, PassError>> {
    find_passes_batch_parallel_with_opsmode(
        elements,
        ground_station,
        start_time,
        end_time,
        options,
        OpsMode::Afspc,
    )
}

/// [`find_passes_batch_parallel`] with an explicit SGP4 [`OpsMode`] (see
/// [`find_passes_with_opsmode`] for why the bare function uses
/// [`OpsMode::Afspc`]).
pub fn find_passes_batch_parallel_with_opsmode(
    elements: &[ElementSet],
    ground_station: GroundStation,
    start_time: UtcInstant,
    end_time: UtcInstant,
    options: PassFinderOptions,
    opsmode: OpsMode,
) -> Vec<Result<Vec<SatellitePass>, PassError>> {
    find_passes_batch(
        elements,
        ground_station,
        start_time,
        end_time,
        options,
        PassBatchMode::Parallel,
        opsmode,
    )
}

/// Find satellite passes for an already-initialized SGP4 satellite.
///
/// This is the same event-finder-backed pass finder as [`find_passes`], but the
/// caller owns SGP4 initialization. Use this when the satellite was built with
/// an explicit [`OpsMode`] or from a handle that must preserve its initialized
/// state.
pub fn find_passes_for_satellite(
    satellite: &Satellite,
    ground_station: GroundStation,
    start_time: UtcInstant,
    end_time: UtcInstant,
    options: PassFinderOptions,
) -> Result<Vec<SatellitePass>, PassError> {
    validate_ground_station(ground_station).map_err(map_pass_input)?;
    let validated_options = validate_pass_finder_options(options)?;
    find_passes_for_satellite_validated(
        satellite,
        ground_station,
        start_time,
        end_time,
        validated_options,
    )
}

fn find_passes_for_satellite_validated(
    satellite: &Satellite,
    ground_station: GroundStation,
    start_time: UtcInstant,
    end_time: UtcInstant,
    options: ValidatedPassFinderOptions,
) -> Result<Vec<SatellitePass>, PassError> {
    if end_time <= start_time {
        return Ok(Vec::new());
    }
    validate_pass_window(start_time, end_time)?;
    let search_step_us = robust_crossing_step_us(
        satellite,
        ground_station,
        options.raw.elevation_mask_deg,
        options.step_us,
    );
    let step_seconds = search_step_us as f64 / MICROSECONDS_PER_SECOND_I64 as f64;
    let time_tolerance_seconds = options.tol_us as f64 / MICROSECONDS_PER_SECOND_I64 as f64;
    let search = PassSearch {
        satellite,
        ground_station,
        start_time,
        end_time,
        step_us: search_step_us,
        tol_us: options.tol_us,
        mask: options.raw.elevation_mask_deg,
    };

    let crossings = find_mask_crossings(search, step_seconds, time_tolerance_seconds)?;

    assemble_passes_from_crossings(search, crossings)
}

fn find_mask_crossings(
    search: PassSearch<'_>,
    step_seconds: f64,
    time_tolerance_seconds: f64,
) -> Result<Vec<CrossingEvent>, PassError> {
    let total_us = search.end_time.diff_microseconds(search.start_time);
    find_segmented_crossings(
        search,
        total_us,
        search.step_us,
        step_seconds,
        time_tolerance_seconds,
    )
    .map_err(map_event_finder_input)
}

fn find_segmented_crossings<P>(
    predicate: P,
    total_us: i64,
    step_us: i64,
    step_seconds: f64,
    time_tolerance_seconds: f64,
) -> Result<Vec<CrossingEvent>, EventFinderError>
where
    P: ScalarEventPredicate + Copy,
{
    if total_us <= 0 {
        return Ok(Vec::new());
    }

    let max_segment_us = max_event_finder_segment_us(step_us);
    if total_us <= max_segment_us {
        return find_crossings_in_offsets(
            predicate,
            0,
            total_us,
            step_seconds,
            time_tolerance_seconds,
        );
    }

    let stride_us = event_finder_segment_stride_us(step_us);
    let mut crossings = Vec::new();
    let mut segment_start_us = 0;

    while segment_start_us < total_us {
        let owned_end_us = segment_start_us.saturating_add(stride_us).min(total_us);
        // Scan one step past the owned range so exact samples at artificial
        // segment boundaries keep the same left/right context as an unsegmented
        // EventFinder scan.
        let finder_end_us = if owned_end_us < total_us {
            owned_end_us.saturating_add(step_us).min(total_us)
        } else {
            owned_end_us
        };
        let segment_crossings = find_crossings_in_offsets(
            predicate,
            segment_start_us,
            finder_end_us,
            step_seconds,
            time_tolerance_seconds,
        )?;

        for crossing in segment_crossings {
            if crossing_is_in_owned_segment(crossing, segment_start_us, owned_end_us)
                && crossing_preserves_truncated_plateau(
                    predicate,
                    crossing,
                    finder_end_us,
                    total_us,
                    step_us,
                )
            {
                crossings.push(crossing);
            }
        }

        if owned_end_us >= total_us {
            break;
        }
        segment_start_us = owned_end_us;
    }

    Ok(crossings)
}

fn find_crossings_in_offsets<P>(
    predicate: P,
    start_offset_us: i64,
    end_offset_us: i64,
    step_seconds: f64,
    time_tolerance_seconds: f64,
) -> Result<Vec<CrossingEvent>, EventFinderError>
where
    P: ScalarEventPredicate,
{
    EventFinder::new(
        offset_seconds(start_offset_us),
        offset_seconds(end_offset_us),
        step_seconds,
        time_tolerance_seconds,
    )
    .and_then(|finder| finder.find_crossings(predicate, 0.0))
}

fn find_segmented_extrema<P>(
    predicate: P,
    total_us: i64,
    step_us: i64,
    step_seconds: f64,
    time_tolerance_seconds: f64,
) -> Result<Vec<ExtremumEvent>, EventFinderError>
where
    P: ScalarEventPredicate + Copy,
{
    if total_us <= 0 {
        return Ok(Vec::new());
    }

    let max_segment_us = max_event_finder_segment_us(step_us);
    if total_us <= max_segment_us {
        return find_extrema_in_offsets(
            predicate,
            0,
            total_us,
            step_seconds,
            time_tolerance_seconds,
        );
    }

    let stride_us = event_finder_segment_stride_us(step_us);
    let mut extrema = Vec::new();
    let mut segment_start_us = 0;

    while segment_start_us < total_us {
        let owned_end_us = segment_start_us.saturating_add(stride_us).min(total_us);
        let finder_end_us = if owned_end_us < total_us {
            owned_end_us.saturating_add(step_us).min(total_us)
        } else {
            owned_end_us
        };
        let segment_extrema = find_extrema_in_offsets(
            predicate,
            segment_start_us,
            finder_end_us,
            step_seconds,
            time_tolerance_seconds,
        )?;

        for extremum in segment_extrema {
            if extremum_is_in_owned_segment(extremum, segment_start_us, owned_end_us) {
                extrema.push(extremum);
            }
        }

        if owned_end_us >= total_us {
            break;
        }
        segment_start_us = owned_end_us;
    }

    Ok(extrema)
}

fn find_extrema_in_offsets<P>(
    predicate: P,
    start_offset_us: i64,
    end_offset_us: i64,
    step_seconds: f64,
    time_tolerance_seconds: f64,
) -> Result<Vec<ExtremumEvent>, EventFinderError>
where
    P: ScalarEventPredicate,
{
    EventFinder::new(
        offset_seconds(start_offset_us),
        offset_seconds(end_offset_us),
        step_seconds,
        time_tolerance_seconds,
    )
    .and_then(|finder| finder.find_extrema(predicate))
}

fn max_event_finder_segment_us(step_us: i64) -> i64 {
    step_us.saturating_mul(EVENT_FINDER_COARSE_SAMPLE_LIMIT)
}

fn event_finder_segment_stride_us(step_us: i64) -> i64 {
    step_us
        .saturating_mul(EVENT_FINDER_COARSE_SAMPLE_LIMIT - 1)
        .max(1)
}

fn crossing_is_in_owned_segment(
    crossing: CrossingEvent,
    segment_start_us: i64,
    owned_end_us: i64,
) -> bool {
    let crossing_us = (crossing.time_seconds * MICROSECONDS_PER_SECOND_I64 as f64).floor() as i64;
    let after_start = segment_start_us == 0 || crossing_us > segment_start_us;
    after_start && crossing_us <= owned_end_us
}

fn crossing_preserves_truncated_plateau<P>(
    predicate: P,
    crossing: CrossingEvent,
    finder_end_us: i64,
    total_us: i64,
    step_us: i64,
) -> bool
where
    P: ScalarEventPredicate + Copy,
{
    if crossing.value - crossing.threshold != 0.0 || finder_end_us >= total_us {
        return true;
    }

    let crossing_us = (crossing.time_seconds * MICROSECONDS_PER_SECOND_I64 as f64).round() as i64;
    if crossing_us > finder_end_us
        || !coarse_zero_run_reaches_offset(
            predicate,
            crossing_us,
            finder_end_us,
            step_us,
            crossing.threshold,
        )
    {
        return true;
    }

    match first_nonzero_coarse_value_after(
        predicate,
        finder_end_us,
        total_us,
        step_us,
        crossing.threshold,
    ) {
        Some(right) => plateau_direction_reaches_opposite_side(crossing.direction, right),
        None => true,
    }
}

fn coarse_zero_run_reaches_offset<P>(
    predicate: P,
    start_us: i64,
    end_us: i64,
    step_us: i64,
    threshold: f64,
) -> bool
where
    P: ScalarEventPredicate + Copy,
{
    let mut offset_us = start_us;
    loop {
        if predicate.value_at(offset_seconds(offset_us)) - threshold != 0.0 {
            return false;
        }
        if offset_us >= end_us {
            return true;
        }
        let Some(next_offset_us) = next_coarse_sample_offset_us(offset_us, end_us, step_us) else {
            return true;
        };
        offset_us = next_offset_us;
    }
}

fn first_nonzero_coarse_value_after<P>(
    predicate: P,
    start_us: i64,
    end_us: i64,
    step_us: i64,
    threshold: f64,
) -> Option<f64>
where
    P: ScalarEventPredicate + Copy,
{
    let mut offset_us = start_us;
    while let Some(next_offset_us) = next_coarse_sample_offset_us(offset_us, end_us, step_us) {
        let value = predicate.value_at(offset_seconds(next_offset_us)) - threshold;
        if value != 0.0 {
            return Some(value);
        }
        offset_us = next_offset_us;
    }
    None
}

fn next_coarse_sample_offset_us(offset_us: i64, end_us: i64, step_us: i64) -> Option<i64> {
    if offset_us >= end_us {
        return None;
    }
    let next_offset_us = offset_us.saturating_add(step_us).min(end_us);
    (next_offset_us > offset_us).then_some(next_offset_us)
}

fn plateau_direction_reaches_opposite_side(direction: CrossingDirection, right_value: f64) -> bool {
    match direction {
        CrossingDirection::Rising => right_value > 0.0,
        CrossingDirection::Falling => right_value < 0.0,
    }
}

fn extremum_is_in_owned_segment(
    extremum: ExtremumEvent,
    segment_start_us: i64,
    owned_end_us: i64,
) -> bool {
    let extremum_us = (extremum.time_seconds * MICROSECONDS_PER_SECOND_I64 as f64).floor() as i64;
    let after_start = segment_start_us == 0 || extremum_us > segment_start_us;
    after_start && extremum_us <= owned_end_us
}

fn offset_seconds(offset_us: i64) -> f64 {
    offset_us as f64 / MICROSECONDS_PER_SECOND_I64 as f64
}

#[derive(Debug, Clone, Copy)]
enum PassBatchMode {
    Serial,
    Parallel,
}

#[derive(Debug, Clone, Copy)]
struct ValidatedPassFinderOptions {
    raw: PassFinderOptions,
    step_us: i64,
    tol_us: i64,
}

fn find_passes_batch(
    elements: &[ElementSet],
    ground_station: GroundStation,
    start_time: UtcInstant,
    end_time: UtcInstant,
    options: PassFinderOptions,
    mode: PassBatchMode,
    opsmode: OpsMode,
) -> Vec<Result<Vec<SatellitePass>, PassError>> {
    if elements.is_empty() {
        return Vec::new();
    }
    if let Err(error) = validate_ground_station(ground_station).map_err(map_pass_input) {
        return vec![Err(error); elements.len()];
    }
    let validated_options = match validate_pass_finder_options(options) {
        Ok(validated) => validated,
        Err(error) => return vec![Err(error); elements.len()],
    };
    // Reject an unrepresentable window up front (before SGP4 init), so the result
    // does not depend on which element sets happen to initialize - consistent with
    // the single-satellite entry points.
    if let Err(error) = validate_pass_window(start_time, end_time) {
        return vec![Err(error); elements.len()];
    }

    let mut results = vec![Ok(Vec::new()); elements.len()];
    let mut valid_indices = Vec::new();
    let mut satellites = Vec::new();
    for (index, element_set) in elements.iter().enumerate() {
        if let Ok(satellite) = Satellite::from_elements_with_opsmode(element_set, opsmode) {
            valid_indices.push(index);
            satellites.push(satellite);
        }
    }
    if satellites.is_empty() {
        return results;
    }

    let valid_results = find_passes_batch_for_satellites_validated(
        &satellites,
        ground_station,
        start_time,
        end_time,
        validated_options,
        mode,
    );
    for (index, result) in valid_indices.into_iter().zip(valid_results) {
        results[index] = result;
    }

    results
}

fn find_passes_batch_for_satellites_validated(
    satellites: &[Satellite],
    ground_station: GroundStation,
    start_time: UtcInstant,
    end_time: UtcInstant,
    options: ValidatedPassFinderOptions,
    mode: PassBatchMode,
) -> Vec<Result<Vec<SatellitePass>, PassError>> {
    if end_time <= start_time {
        return vec![Ok(Vec::new()); satellites.len()];
    }

    match mode {
        PassBatchMode::Serial => satellites
            .iter()
            .map(|satellite| {
                find_passes_for_satellite_validated(
                    satellite,
                    ground_station,
                    start_time,
                    end_time,
                    options,
                )
            })
            .collect(),
        PassBatchMode::Parallel => satellites
            .par_iter()
            .map(|satellite| {
                find_passes_for_satellite_validated(
                    satellite,
                    ground_station,
                    start_time,
                    end_time,
                    options,
                )
            })
            .collect(),
    }
}

fn assemble_passes_from_crossings(
    search: PassSearch<'_>,
    crossings: Vec<CrossingEvent>,
) -> Result<Vec<SatellitePass>, PassError> {
    let mut aos = if search.window_start_is_inside_pass(&crossings) {
        Some(search.start_time)
    } else {
        None
    };
    let mut passes = Vec::new();

    for crossing in crossings {
        let crossing_time = search.refine_event_mask_crossing_to_utc(crossing.time_seconds);
        match crossing.direction {
            CrossingDirection::Rising => {
                if aos.is_none() {
                    aos = Some(crossing_time);
                }
            }
            CrossingDirection::Falling => {
                if let Some(rise) = aos.take() {
                    let set = crossing_time;
                    let (culmination, max_elevation_deg) = find_culmination(
                        search.satellite,
                        search.ground_station,
                        rise,
                        set,
                        search.step_us,
                        search.tol_us,
                    )?;
                    passes.push(SatellitePass {
                        aos: rise,
                        los: set,
                        culmination,
                        max_elevation_deg,
                    });
                }
            }
        }
    }

    Ok(passes)
}

fn validate_pass_prediction_options(options: PassPredictionOptions) -> Result<i64, PassError> {
    validate_elevation_threshold(options.min_elevation_deg, "min_elevation_deg")
        .map_err(map_pass_input)?;
    validate_pass_step_seconds(options.step_seconds)
}

fn validate_pass_step_seconds(step_seconds: i64) -> Result<i64, PassError> {
    validate::positive_step(step_seconds as f64, "step_seconds").map_err(map_pass_input)?;
    Ok(step_seconds)
}

fn validate_pass_finder_options(
    options: PassFinderOptions,
) -> Result<ValidatedPassFinderOptions, PassError> {
    validate_elevation_threshold(options.elevation_mask_deg, "elevation_mask_deg")
        .map_err(map_pass_input)?;
    let step_us = positive_seconds_to_microseconds(
        options.coarse_step_seconds,
        "coarse_step_seconds",
        false,
    )?;
    let tol_us = positive_seconds_to_microseconds(
        options.time_tolerance_seconds,
        "time_tolerance_seconds",
        true,
    )?;
    Ok(ValidatedPassFinderOptions {
        raw: options,
        step_us,
        tol_us,
    })
}

fn robust_crossing_step_us(
    satellite: &Satellite,
    ground_station: GroundStation,
    elevation_mask_deg: f64,
    requested_step_us: i64,
) -> i64 {
    let orbit_step_us = fastest_orbit_fraction_step_us(satellite).unwrap_or(requested_step_us);
    let geometry_step_us = mask_geometry_step_us(satellite, ground_station, elevation_mask_deg)
        .unwrap_or(requested_step_us);
    requested_step_us
        .min(orbit_step_us)
        .min(geometry_step_us)
        .max(1)
}

fn fastest_orbit_fraction_step_us(satellite: &Satellite) -> Option<i64> {
    let perigee_rate_rad_s = perigee_angular_rate_rad_s(satellite)?;
    let fastest_rev_seconds = std::f64::consts::TAU / perigee_rate_rad_s;
    let samples_per_fastest_rev = if fastest_rev_seconds >= ROBUST_CROSSING_SLOW_ORBIT_SECONDS {
        ROBUST_CROSSING_SLOW_SAMPLES_PER_FASTEST_REV
    } else {
        ROBUST_CROSSING_FAST_SAMPLES_PER_FASTEST_REV
    };
    let step_seconds = fastest_rev_seconds / samples_per_fastest_rev;
    if !(step_seconds.is_finite() && step_seconds > 0.0) {
        return None;
    }

    Some((step_seconds * MICROSECONDS_PER_SECOND_I64 as f64).round() as i64)
}

fn perigee_angular_rate_rad_s(satellite: &Satellite) -> Option<f64> {
    let mean_motion_rad_s = satellite.mean_motion_rad_per_min() / 60.0;
    let eccentricity = satellite.eccentricity();
    if !(mean_motion_rad_s.is_finite()
        && mean_motion_rad_s > 0.0
        && eccentricity.is_finite()
        && (0.0..1.0).contains(&eccentricity))
    {
        return None;
    }

    let one_minus_e = 1.0 - eccentricity;
    let perigee_rate_rad_s =
        mean_motion_rad_s * (1.0 + eccentricity).sqrt() / one_minus_e.powf(1.5);
    if !(perigee_rate_rad_s.is_finite() && perigee_rate_rad_s > 0.0) {
        return None;
    }

    Some(perigee_rate_rad_s)
}

fn mask_geometry_step_us(
    satellite: &Satellite,
    ground_station: GroundStation,
    elevation_mask_deg: f64,
) -> Option<i64> {
    let station_radius_km = station_geocentric_radius_km(ground_station)?;
    let perigee_radius_km = orbit_perigee_radius_km(satellite)?;
    if perigee_radius_km <= station_radius_km {
        return None;
    }

    let mask_rad = elevation_mask_deg.to_radians();
    if !(mask_rad.is_finite() && mask_rad < std::f64::consts::FRAC_PI_2) {
        return None;
    }
    let access_half_angle_rad =
        elevation_mask_central_angle_rad(station_radius_km, perigee_radius_km, mask_rad)?;
    let perigee_rate_rad_s = perigee_angular_rate_rad_s(satellite)?;
    let station_lat_rad = ground_station.latitude_deg.to_radians();
    if !station_lat_rad.is_finite() {
        return None;
    }
    let station_spin_rad_s = OMEGA_E_DOT_RAD_S * station_lat_rad.cos().abs();
    let relative_rate_rad_s = perigee_rate_rad_s + station_spin_rad_s;
    if !(relative_rate_rad_s.is_finite() && relative_rate_rad_s > 0.0) {
        return None;
    }

    let dwell_seconds = 2.0 * access_half_angle_rad / relative_rate_rad_s;
    let step_seconds = dwell_seconds / ROBUST_CROSSING_MASK_DWELL_SAMPLES;
    if !(step_seconds.is_finite() && step_seconds > 0.0) {
        return None;
    }

    Some(((step_seconds * MICROSECONDS_PER_SECOND_I64 as f64).floor() as i64).max(1))
}

fn orbit_perigee_radius_km(satellite: &Satellite) -> Option<f64> {
    let mean_motion_rad_s = satellite.mean_motion_rad_per_min() / 60.0;
    let eccentricity = satellite.eccentricity();
    if !(mean_motion_rad_s.is_finite()
        && mean_motion_rad_s > 0.0
        && eccentricity.is_finite()
        && (0.0..1.0).contains(&eccentricity))
    {
        return None;
    }

    let semi_major_axis_km = (GM_EARTH_KM3_S2 / (mean_motion_rad_s * mean_motion_rad_s)).cbrt();
    let perigee_radius_km = semi_major_axis_km * (1.0 - eccentricity);
    if !(perigee_radius_km.is_finite() && perigee_radius_km > 0.0) {
        return None;
    }

    Some(perigee_radius_km)
}

fn station_geocentric_radius_km(ground_station: GroundStation) -> Option<f64> {
    if !(ground_station.latitude_deg.is_finite()
        && ground_station.longitude_deg.is_finite()
        && ground_station.altitude_m.is_finite())
    {
        return None;
    }

    let (x, y, z) = geodetic_to_itrs(
        ground_station.latitude_deg,
        ground_station.longitude_deg,
        ground_station.altitude_m / 1000.0,
    )
    .ok()?;
    let radius_km = (x * x + y * y + z * z).sqrt();
    if !(radius_km.is_finite() && radius_km > 0.0) {
        return None;
    }

    Some(radius_km)
}

fn elevation_mask_central_angle_rad(
    station_radius_km: f64,
    satellite_radius_km: f64,
    elevation_mask_rad: f64,
) -> Option<f64> {
    let radius_ratio = station_radius_km / satellite_radius_km;
    if !(radius_ratio.is_finite() && (0.0..1.0).contains(&radius_ratio)) {
        return None;
    }

    let sin_mask = elevation_mask_rad.sin();
    let cos_mask = elevation_mask_rad.cos();
    let cos_mask_squared = cos_mask * cos_mask;
    let horizon_term = 1.0 - radius_ratio * radius_ratio * cos_mask_squared;
    if horizon_term < 0.0 {
        return None;
    }

    let cos_central_angle = radius_ratio * cos_mask_squared + sin_mask * horizon_term.sqrt();
    let central_angle = cos_central_angle.clamp(-1.0, 1.0).acos();
    if !(central_angle.is_finite() && central_angle > 0.0) {
        return None;
    }

    Some(central_angle)
}

fn positive_seconds_to_microseconds(
    seconds: f64,
    field: &'static str,
    round_submicrosecond_to_one: bool,
) -> Result<i64, PassError> {
    let seconds = validate::positive_step(seconds, field).map_err(map_pass_input)?;
    let max_seconds = (i64::MAX / MICROSECONDS_PER_SECOND_I64) as f64;
    if seconds > max_seconds {
        return Err(invalid_pass_input(field, "out of range"));
    }

    let microseconds = (seconds * MICROSECONDS_PER_SECOND_I64 as f64).round();
    if round_submicrosecond_to_one && microseconds < 1.0 {
        return Ok(1);
    }
    validate::positive_step(microseconds, field).map_err(map_pass_input)?;
    Ok(microseconds as i64)
}

fn validate_ground_station(ground_station: GroundStation) -> Result<(), validate::FieldError> {
    validate::finite_in_range(
        ground_station.latitude_deg,
        -90.0,
        90.0,
        "ground_station.latitude_deg",
    )?;
    validate::finite_in_range(
        ground_station.longitude_deg,
        -180.0,
        180.0,
        "ground_station.longitude_deg",
    )?;
    validate::finite(ground_station.altitude_m, "ground_station.altitude_m")?;
    Ok(())
}

fn validate_elevation_threshold(
    elevation_deg: f64,
    field: &'static str,
) -> Result<(), validate::FieldError> {
    validate::finite_in_range(elevation_deg, -90.0, 90.0, field)?;
    Ok(())
}

fn time_scales_for_look_angle(datetime: UtcInstant) -> Result<TimeScales, LookAngleError> {
    time_scales_from_instant(datetime).ok_or(LookAngleError::InvalidInput {
        field: "datetime",
        reason: "invalid UTC instant",
    })
}

fn time_scales_for_pass(datetime: UtcInstant) -> Result<TimeScales, PassError> {
    time_scales_from_instant(datetime).ok_or(PassError::InvalidInput {
        field: "datetime",
        reason: "invalid UTC instant",
    })
}

fn time_scales_from_instant(datetime: UtcInstant) -> Option<TimeScales> {
    let c = datetime.components();
    TimeScales::from_utc(
        c.year,
        c.month,
        c.day,
        c.hour,
        c.minute,
        c.second as f64 + c.microsecond as f64 / 1_000_000.0,
    )
    .ok()
}

fn validate_look_angle(look: LookAngle) -> Result<LookAngle, LookAngleError> {
    validate::finite(look.azimuth_deg, "azimuth_deg").map_err(map_look_angle_input)?;
    validate::finite(look.elevation_deg, "elevation_deg").map_err(map_look_angle_input)?;
    validate::finite(look.range_km, "range_km").map_err(map_look_angle_input)?;
    Ok(look)
}

fn map_look_angle_input(error: validate::FieldError) -> LookAngleError {
    LookAngleError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

fn map_pass_input(error: validate::FieldError) -> PassError {
    invalid_pass_input(error.field(), error.reason())
}

fn invalid_pass_input(field: &'static str, reason: &'static str) -> PassError {
    PassError::InvalidInput { field, reason }
}

/// Reject a `[start, end]` window whose microsecond span overflows `i64`.
///
/// `UtcInstant` is public and can be built at the i64 extremes, so a span of
/// `end - start` can overflow. The sampling/bisection arithmetic assumes a
/// representable span; rather than rely on that arithmetic saturating silently,
/// the public entry points reject an unrepresentable window with a typed error.
/// Every realistic window (hours to years) is far inside the representable range
/// (~292,000 years of microseconds), so this never fires for valid inputs.
fn validate_pass_window(start_time: UtcInstant, end_time: UtcInstant) -> Result<(), PassError> {
    if end_time.checked_diff_microseconds(start_time).is_none() {
        return Err(invalid_pass_input(
            "time window",
            "span between start_time and end_time exceeds the representable range",
        ));
    }
    Ok(())
}

fn map_event_finder_input(error: EventFinderError) -> PassError {
    match error {
        EventFinderError::InvalidInput { field, reason } => {
            PassError::InvalidInput { field, reason }
        }
    }
}

fn instant_at_offset_seconds(start_time: UtcInstant, offset_seconds: f64) -> UtcInstant {
    start_time
        .add_microseconds((offset_seconds * MICROSECONDS_PER_SECOND_I64 as f64).floor() as i64)
}

#[derive(Debug, Clone, Copy)]
struct PassSearch<'a> {
    satellite: &'a Satellite,
    ground_station: GroundStation,
    start_time: UtcInstant,
    end_time: UtcInstant,
    step_us: i64,
    tol_us: i64,
    mask: f64,
}

impl ScalarEventPredicate for PassSearch<'_> {
    fn value_at(&self, offset_seconds: f64) -> f64 {
        self.masked_elevation_at(instant_at_offset_seconds(self.start_time, offset_seconds))
    }
}

impl PassSearch<'_> {
    fn window_start_is_inside_pass(&self, crossings: &[CrossingEvent]) -> bool {
        match self.masked_elevation_at(self.start_time).partial_cmp(&0.0) {
            Some(core::cmp::Ordering::Greater) => true,
            Some(core::cmp::Ordering::Equal) => crossings.first().is_some_and(|crossing| {
                if crossing.time_seconds == 0.0 {
                    crossing.direction == CrossingDirection::Rising
                } else {
                    crossing.direction == CrossingDirection::Falling
                }
            }),
            _ => false,
        }
    }

    fn masked_elevation_at(&self, datetime: UtcInstant) -> f64 {
        elevation_at(self.satellite, datetime, self.ground_station) - self.mask
    }

    fn refine_event_mask_crossing_to_utc(&self, crossing_time_seconds: f64) -> UtcInstant {
        let total_us = self.end_time.diff_microseconds(self.start_time);
        let crossing_us = ((crossing_time_seconds * MICROSECONDS_PER_SECOND_I64 as f64).floor()
            as i64)
            .clamp(0, total_us);
        let mut low_offset = (crossing_us / self.step_us) * self.step_us;
        if low_offset >= total_us && total_us > 0 {
            low_offset = ((total_us - 1) / self.step_us) * self.step_us;
        }
        let mut high_offset = (low_offset + self.step_us).min(total_us);
        if high_offset == low_offset && low_offset > 0 {
            low_offset = low_offset.saturating_sub(self.step_us);
            high_offset = (low_offset + self.step_us).min(total_us);
        }

        if let Some(refined) = self.refine_mask_crossing_in_offsets(low_offset, high_offset) {
            return refined;
        }

        let previous_low = low_offset.saturating_sub(self.step_us);
        if previous_low < low_offset {
            if let Some(refined) = self.refine_mask_crossing_in_offsets(previous_low, low_offset) {
                return refined;
            }
        }

        instant_at_offset_seconds(self.start_time, crossing_time_seconds)
    }

    fn refine_mask_crossing_in_offsets(
        &self,
        low_offset_us: i64,
        high_offset_us: i64,
    ) -> Option<UtcInstant> {
        if low_offset_us == high_offset_us {
            return None;
        }

        let low = self.start_time.add_microseconds(low_offset_us);
        let high = self.start_time.add_microseconds(high_offset_us);
        let value_low = self.masked_elevation_at(low);
        let value_high = self.masked_elevation_at(high);
        if !sign_change_bracketed(value_low, value_high).unwrap_or(false) {
            return None;
        }

        bisect_crossing_until(
            low,
            high,
            |dt| self.masked_elevation_at(dt),
            midpoint_instant,
            |lo, hi| hi.diff_microseconds(lo) <= self.tol_us,
        )
        .ok()
    }
}

/// Elevation rate (deg/s) at `dt` by central difference with half-step `h_us`.
fn elevation_rate<F>(elevation_at_time: &F, dt: UtcInstant, h_us: i64) -> f64
where
    F: Fn(UtcInstant) -> f64 + ?Sized,
{
    let plus = elevation_at_time(dt.add_microseconds(h_us));
    let minus = elevation_at_time(dt.add_microseconds(-h_us));
    (plus - minus) / (2.0 * h_us as f64 / MICROSECONDS_PER_SECOND_I64 as f64)
}

/// Find the maximum elevation between AOS and LOS through the shared extrema
/// finder. For a partial pass with no interior peak, return the higher endpoint.
fn find_culmination(
    satellite: &Satellite,
    ground_station: GroundStation,
    aos: UtcInstant,
    los: UtcInstant,
    step_us: i64,
    tol_us: i64,
) -> Result<(UtcInstant, f64), PassError> {
    find_culmination_with(aos, los, step_us, tol_us, |dt| {
        elevation_at(satellite, dt, ground_station)
    })
}

fn find_culmination_with<F>(
    aos: UtcInstant,
    los: UtcInstant,
    step_us: i64,
    tol_us: i64,
    elevation_at_time: F,
) -> Result<(UtcInstant, f64), PassError>
where
    F: Fn(UtcInstant) -> f64,
{
    let span_us = los.diff_microseconds(aos);
    let aos_elevation = elevation_at_time(aos);
    if span_us <= 0 {
        return Ok((aos, aos_elevation));
    }

    let los_elevation = elevation_at_time(los);
    let mut best = if aos_elevation >= los_elevation {
        (aos, aos_elevation)
    } else {
        (los, los_elevation)
    };
    let mut selected_maximum_bracket = None;
    let extremum_step_us = step_us.min((span_us / 4).max(1)).max(1);
    let extremum_step_seconds = extremum_step_us as f64 / MICROSECONDS_PER_SECOND_I64 as f64;
    let time_tolerance_seconds = tol_us as f64 / MICROSECONDS_PER_SECOND_I64 as f64;

    let extrema = find_segmented_extrema(
        |offset_seconds| elevation_at_time(instant_at_offset_seconds(aos, offset_seconds)),
        span_us,
        extremum_step_us,
        extremum_step_seconds,
        time_tolerance_seconds,
    )
    .map_err(map_event_finder_input)?;

    for extremum in extrema {
        if extremum.kind != ExtremumKind::Maximum {
            continue;
        }
        let candidate_time = instant_at_offset_seconds(aos, extremum.time_seconds);
        let candidate_elevation = elevation_at_time(candidate_time);
        if candidate_elevation > best.1 {
            best = (candidate_time, candidate_elevation);
            selected_maximum_bracket =
                extremum_sample_bracket(aos, span_us, extremum_step_us, extremum.time_seconds);
        }
    }

    if let Some((low, high)) = selected_maximum_bracket {
        if let Some(refined) = refine_culmination_rate_zero(&elevation_at_time, aos, los, tol_us) {
            if low <= refined.0 && refined.0 <= high {
                return Ok(refined);
            }
        }
        if let Some(refined) = refine_culmination_rate_zero(&elevation_at_time, low, high, tol_us) {
            best = refined;
        }
    }

    Ok(best)
}

fn extremum_sample_bracket(
    aos: UtcInstant,
    span_us: i64,
    step_us: i64,
    time_seconds: f64,
) -> Option<(UtcInstant, UtcInstant)> {
    if !(span_us > 0 && step_us > 0 && time_seconds.is_finite()) {
        return None;
    }

    let step_seconds = step_us as f64 / MICROSECONDS_PER_SECOND_I64 as f64;
    let center_index = (time_seconds / step_seconds).round() as i64;
    let center_offset_us = center_index.saturating_mul(step_us).clamp(0, span_us);
    if center_offset_us == 0 || center_offset_us == span_us {
        return None;
    }

    let low_offset_us = center_offset_us.saturating_sub(step_us);
    let high_offset_us = center_offset_us.saturating_add(step_us).min(span_us);
    if low_offset_us >= high_offset_us {
        return None;
    }

    Some((
        aos.add_microseconds(low_offset_us),
        aos.add_microseconds(high_offset_us),
    ))
}

fn refine_culmination_rate_zero<F>(
    elevation_at_time: &F,
    low: UtcInstant,
    high: UtcInstant,
    tol_us: i64,
) -> Option<(UtcInstant, f64)>
where
    F: Fn(UtcInstant) -> f64 + ?Sized,
{
    let span = high.diff_microseconds(low);
    if span <= 0 {
        return None;
    }

    let h_us = CULMINATION_RATE_HALF_STEP_US.min((span / 4).max(1));
    let rate_low = elevation_rate(elevation_at_time, low, h_us);
    let rate_high = elevation_rate(elevation_at_time, high, h_us);
    if !(rate_low.is_finite() && rate_high.is_finite() && rate_low > 0.0 && rate_high < 0.0) {
        return None;
    }

    let culmination = bisect_crossing_until(
        low,
        high,
        |dt| elevation_rate(elevation_at_time, dt, h_us),
        midpoint_instant,
        |lo, hi| hi.diff_microseconds(lo) <= tol_us,
    )
    .ok()?;
    Some((culmination, elevation_at_time(culmination)))
}

#[cfg(test)]
mod tests {
    use crate::astro::events::{
        CrossingDirection, CrossingEvent, EventFinder, EventFinderError, ExtremumKind,
    };
    use crate::astro::frames::transforms::{
        gcrs_to_itrs_compute, itrs_to_geodetic_compute, teme_to_gcrs_compute, TemeStateKm,
    };

    use super::*;

    fn iss_2024_12_19_elements() -> ElementSet {
        ElementSet {
            epoch: crate::astro::sgp4::sgp4_julian_date_from_day_of_year(2024, 354.52609954),
            bstar: 0.000_370_420_000_000_000_05,
            mean_motion_dot: 0.00020888,
            mean_motion_double_dot: 0.0,
            eccentricity: 0.0006955,
            argument_of_perigee_deg: 37.7614,
            inclination_deg: 51.6393,
            mean_anomaly_deg: 87.9783,
            mean_motion_rev_per_day: 15.49970085,
            right_ascension_deg: 213.2584,
            catalog_number: 0,
        }
    }

    fn iss_2024_01_01_elements() -> ElementSet {
        ElementSet {
            epoch: crate::astro::sgp4::sgp4_julian_date_from_day_of_year(2024, 1.5),
            bstar: 0.000_102_70,
            mean_motion_dot: 0.000_167_17,
            mean_motion_double_dot: 0.0,
            eccentricity: 0.000_264_4,
            argument_of_perigee_deg: 250.3037,
            inclination_deg: 51.6400,
            mean_anomaly_deg: 109.7782,
            mean_motion_rev_per_day: 15.49560812,
            right_ascension_deg: 208.8657,
            catalog_number: 25_544,
        }
    }

    fn iss_fixture_elements() -> ElementSet {
        ElementSet {
            epoch: crate::astro::sgp4::sgp4_julian_date_from_day_of_year(2026, 95.55331950),
            bstar: 0.000_164_20,
            mean_motion_dot: 0.000_085_43,
            mean_motion_double_dot: 0.0,
            eccentricity: 0.000_635_1,
            argument_of_perigee_deg: 274.8255,
            inclination_deg: 51.6328,
            mean_anomaly_deg: 85.2008,
            mean_motion_rev_per_day: 15.4878698,
            right_ascension_deg: 299.5432,
            catalog_number: 25_544,
        }
    }

    fn css_fixture_elements() -> ElementSet {
        ElementSet {
            epoch: crate::astro::sgp4::sgp4_julian_date_from_day_of_year(2026, 95.32454765),
            bstar: 0.000_372_23,
            mean_motion_dot: 0.000_331_73,
            mean_motion_double_dot: 0.0,
            eccentricity: 0.000_355_7,
            argument_of_perigee_deg: 129.2727,
            inclination_deg: 41.4682,
            mean_anomaly_deg: 230.8429,
            mean_motion_rev_per_day: 15.6194274,
            right_ascension_deg: 45.9319,
            catalog_number: 48_274,
        }
    }

    fn fregat_fixture_elements() -> ElementSet {
        ElementSet {
            epoch: crate::astro::sgp4::sgp4_julian_date_from_day_of_year(2026, 95.51225242),
            bstar: 0.012_423,
            mean_motion_dot: 0.000_085_41,
            mean_motion_double_dot: 0.0,
            eccentricity: 0.095_504_7,
            argument_of_perigee_deg: 120.8974,
            inclination_deg: 51.6426,
            mean_anomaly_deg: 248.9327,
            mean_motion_rev_per_day: 12.40936816,
            right_ascension_deg: 220.2066,
            catalog_number: 49_271,
        }
    }

    fn geo_like_fixture_elements() -> ElementSet {
        ElementSet {
            epoch: crate::astro::sgp4::sgp4_julian_date_from_day_of_year(2026, 95.0),
            bstar: 0.0,
            mean_motion_dot: 0.0,
            mean_motion_double_dot: 0.0,
            eccentricity: 0.000_1,
            argument_of_perigee_deg: 0.0,
            inclination_deg: 0.1,
            mean_anomaly_deg: 0.0,
            mean_motion_rev_per_day: 1.002_7,
            right_ascension_deg: 0.0,
            catalog_number: 99_001,
        }
    }

    fn station_under_satellite(satellite: &Satellite, datetime: UtcInstant) -> GroundStation {
        let pred = satellite
            .propagate_jd(datetime.sgp4_julian_date())
            .expect("satellite propagates at subpoint instant");
        let ts = datetime.time_scales();
        let (gcrs_position, _) = teme_to_gcrs_compute(
            &TemeStateKm {
                position_km: pred.position,
                velocity_km_s: pred.velocity,
            },
            &ts,
            false,
        )
        .expect("valid TEME to GCRS transform");
        let (x, y, z) = gcrs_to_itrs_compute(
            gcrs_position.0,
            gcrs_position.1,
            gcrs_position.2,
            &ts,
            false,
        )
        .expect("valid GCRS to ITRS transform");
        let (latitude_deg, longitude_deg, _) =
            itrs_to_geodetic_compute(x, y, z).expect("valid ITRS geodetic coordinates");

        GroundStation {
            latitude_deg,
            longitude_deg,
            altitude_m: 0.0,
        }
    }

    fn instant_at_offset_seconds(start: UtcInstant, offset_seconds: f64) -> UtcInstant {
        start.add_microseconds((offset_seconds * MICROSECONDS_PER_SECOND_I64 as f64).floor() as i64)
    }

    fn multi_stationary_elevation(dt: UtcInstant) -> f64 {
        let t = dt.unix_microseconds() as f64 / MICROSECONDS_PER_SECOND_I64 as f64;
        if t <= 1.5 {
            hermite(t, 0.0, 0.0, 8.0, 1.5, 10.0, 0.0)
        } else if t <= 6.0 {
            hermite(t, 1.5, 10.0, 0.0, 6.0, -4.0, 0.0)
        } else if t <= 8.5 {
            hermite(t, 6.0, -4.0, 0.0, 8.5, 30.0, 0.0)
        } else {
            hermite(t, 8.5, 30.0, 0.0, 10.0, 0.0, -20.0)
        }
    }

    fn hermite(t: f64, t0: f64, y0: f64, m0: f64, t1: f64, y1: f64, m1: f64) -> f64 {
        let span = t1 - t0;
        let u = (t - t0) / span;
        let u2 = u * u;
        let u3 = u2 * u;
        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = u3 - u2;
        h00 * y0 + h10 * span * m0 + h01 * y1 + h11 * span * m1
    }

    fn synthetic_pass_wave(time_seconds: f64) -> f64 {
        (time_seconds * std::f64::consts::TAU / 5_400.0).sin()
    }

    fn crossing_offset_us(crossing: CrossingEvent) -> i64 {
        (crossing.time_seconds * MICROSECONDS_PER_SECOND_I64 as f64).floor() as i64
    }

    #[test]
    fn utc_instant_round_trips_calendar_fields() {
        let instant = UtcInstant::from_utc(2024, 12, 19, 7, 3, 11, 825_435).unwrap();
        assert_eq!(instant.unix_microseconds(), 1_734_591_791_825_435);
        let c = instant.components();
        assert_eq!(
            (
                c.year,
                c.month,
                c.day,
                c.hour,
                c.minute,
                c.second,
                c.microsecond
            ),
            (2024, 12, 19, 7, 3, 11, 825_435)
        );
    }

    #[test]
    fn utc_instant_rejects_invalid_civil_dates_and_non_leap_seconds() {
        assert!(UtcInstant::from_utc(2024, 2, 31, 0, 0, 0, 0).is_none());
        assert!(UtcInstant::from_utc(2023, 2, 29, 0, 0, 0, 0).is_none());
        assert!(UtcInstant::from_utc(2024, 12, 31, 23, 59, 60, 0).is_none());
    }

    #[test]
    fn utc_instant_rejects_leap_second_labels_to_avoid_midnight_collision() {
        assert!(UtcInstant::from_utc(2024, 2, 29, 0, 0, 0, 0).is_some());
        assert!(UtcInstant::from_utc(2023, 2, 28, 23, 59, 59, 999_999).is_some());

        assert!(UtcInstant::from_utc(2016, 12, 31, 23, 59, 60, 0).is_none());
        assert_eq!(
            UtcInstant::from_utc(2017, 1, 1, 0, 0, 0, 0)
                .expect("following midnight is a normal instant")
                .unix_microseconds(),
            1_483_228_800_000_000
        );
    }

    // Cross-validated against Skyfield 1.54 (sgp4 2.25, WGS72, AFSPC opsmode
    // 'a', skyfield load.timescale(builtin=False) = current IERS
    // finals2000A.all). The same ISS element set this fixture builds is fed to
    // sgp4.Satrec.sgp4init and wrapped in skyfield EarthSatellite.from_satrec;
    // the pass events come from
    // EarthSatellite.find_events(station, t0, t1, altitude_degrees=0.0).
    // Skyfield reference (microseconds since the unix epoch, and peak elevation):
    //   rise = 1_734_604_991_843_315, culmination = 1_734_605_261_880_434,
    //   set  = 1_734_605_533_416_157, peak elevation = 12.547260311655 deg.
    // Measured sidereon-vs-Skyfield residuals: rise 17.88 ms, set 15.79 ms,
    // culmination 12.15 ms (Skyfield find_events carries its own root-finding
    // tolerance), peak elevation 1.14e-7 deg. The tolerances below sit a few
    // factors above those residuals.
    #[test]
    fn iss_london_pass_matches_skyfield() {
        const SKYFIELD_RISE_US: i64 = 1_734_604_991_843_315;
        const SKYFIELD_CULMINATION_US: i64 = 1_734_605_261_880_434;
        const SKYFIELD_SET_US: i64 = 1_734_605_533_416_157;
        const SKYFIELD_PEAK_ELEVATION_DEG: f64 = 12.547_260_311_655;
        const TIME_TOL_US: i64 = 50_000; // 0.05 s
        const ELEVATION_TOL_DEG: f64 = 1.0e-5;

        let start = UtcInstant::from_utc(2024, 12, 19, 0, 0, 0, 0).unwrap();
        let end = UtcInstant::from_utc(2024, 12, 19, 12, 0, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };

        let passes = predict_passes(
            &iss_2024_12_19_elements(),
            station,
            start,
            end,
            PassPredictionOptions::default(),
        )
        .expect("valid pass-prediction step");

        assert_eq!(passes.len(), 1);
        let pass = passes[0];
        assert!(
            (pass.rise.unix_microseconds() - SKYFIELD_RISE_US).abs() <= TIME_TOL_US,
            "rise {} vs skyfield {SKYFIELD_RISE_US}",
            pass.rise.unix_microseconds()
        );
        assert!(
            (pass.set.unix_microseconds() - SKYFIELD_SET_US).abs() <= TIME_TOL_US,
            "set {} vs skyfield {SKYFIELD_SET_US}",
            pass.set.unix_microseconds()
        );
        assert!(
            (pass.max_elevation_time.unix_microseconds() - SKYFIELD_CULMINATION_US).abs()
                <= TIME_TOL_US,
            "culmination {} vs skyfield {SKYFIELD_CULMINATION_US}",
            pass.max_elevation_time.unix_microseconds()
        );
        assert!(
            (pass.max_elevation_deg - SKYFIELD_PEAK_ELEVATION_DEG).abs() <= ELEVATION_TOL_DEG,
            "peak elevation {} vs skyfield {SKYFIELD_PEAK_ELEVATION_DEG}",
            pass.max_elevation_deg
        );

        let high = predict_passes(
            &iss_2024_12_19_elements(),
            station,
            start,
            end,
            PassPredictionOptions {
                min_elevation_deg: 30.0,
                step_seconds: 60,
            },
        )
        .expect("valid pass-prediction step");
        assert!(high.is_empty());
    }

    #[test]
    fn predict_passes_includes_set_in_final_partial_interval() {
        let start = UtcInstant::from_utc(2024, 12, 19, 10, 42, 0, 0).unwrap();
        let end = UtcInstant::from_utc(2024, 12, 19, 10, 52, 30, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };

        let passes = predict_passes(
            &iss_2024_12_19_elements(),
            station,
            start,
            end,
            PassPredictionOptions {
                min_elevation_deg: 0.0,
                step_seconds: 60,
            },
        )
        .expect("valid pass-prediction step");

        assert_eq!(passes.len(), 1);
        assert!(passes[0].rise.unix_microseconds() >= start.unix_microseconds());
        assert!(passes[0].set.unix_microseconds() <= end.unix_microseconds());
    }

    #[test]
    fn predict_passes_rejects_unrepresentable_window_instead_of_panicking() {
        // A window spanning the full i64 microsecond range overflows the internal
        // span arithmetic. It must surface as a typed error, never a debug panic
        // or a release wrap.
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let start = UtcInstant::from_unix_microseconds(i64::MIN);
        let end = UtcInstant::from_unix_microseconds(i64::MAX);

        let predict_err = predict_passes(
            &iss_2024_12_19_elements(),
            station,
            start,
            end,
            PassPredictionOptions::default(),
        )
        .expect_err("an unrepresentable window must be a typed error");
        assert!(matches!(
            predict_err,
            PassError::InvalidInput {
                field: "time window",
                ..
            }
        ));

        let find_err = find_passes(
            &iss_2024_12_19_elements(),
            station,
            start,
            end,
            PassFinderOptions::default(),
        )
        .expect_err("an unrepresentable window must be a typed error");
        assert!(matches!(
            find_err,
            PassError::InvalidInput {
                field: "time window",
                ..
            }
        ));
    }

    #[test]
    fn find_passes_opsmode_is_threaded_and_consistent_across_paths() {
        // The element-based pass APIs default to OpsMode::Afspc (golden-pinned),
        // while the rest of the crate defaults to Improved. The _with_opsmode
        // variants thread the choice through so the same TLE no longer yields a
        // different trajectory by path.
        let elements = iss_2024_12_19_elements();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let start = UtcInstant::from_utc(2024, 12, 19, 0, 0, 0, 0).unwrap();
        let end = UtcInstant::from_utc(2024, 12, 19, 12, 0, 0, 0).unwrap();
        let opts = PassFinderOptions::default();

        // The bare function and the explicit-Afspc variant are identical.
        let bare = find_passes(&elements, station, start, end, opts).unwrap();
        let afspc =
            find_passes_with_opsmode(&elements, station, start, end, opts, OpsMode::Afspc).unwrap();
        assert_eq!(bare.len(), afspc.len());
        assert!(!bare.is_empty(), "expected at least one pass in the window");
        for (a, b) in bare.iter().zip(&afspc) {
            assert_eq!(a.aos.unix_microseconds(), b.aos.unix_microseconds());
            assert_eq!(a.los.unix_microseconds(), b.los.unix_microseconds());
        }

        // The Improved variant matches the caller-owns-init path built with the
        // crate-default (Improved) opsmode - the consistency #5 is about.
        let improved_sat = Satellite::from_elements(&elements).expect("Improved-built satellite");
        let via_sat = find_passes_for_satellite(&improved_sat, station, start, end, opts).unwrap();
        let improved =
            find_passes_with_opsmode(&elements, station, start, end, opts, OpsMode::Improved)
                .unwrap();
        assert_eq!(improved.len(), via_sat.len());
        for (a, b) in improved.iter().zip(&via_sat) {
            assert_eq!(a.aos.unix_microseconds(), b.aos.unix_microseconds());
            assert_eq!(a.los.unix_microseconds(), b.los.unix_microseconds());
            assert_eq!(
                a.culmination.unix_microseconds(),
                b.culmination.unix_microseconds()
            );
        }
    }

    #[test]
    fn coarse_scan_does_not_duplicate_exact_end_sample() {
        let start = UtcInstant::from_utc(2024, 12, 19, 10, 42, 0, 0).unwrap();
        let end = UtcInstant::from_utc(2024, 12, 19, 10, 52, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let satellite =
            Satellite::from_elements_with_opsmode(&iss_2024_12_19_elements(), OpsMode::Afspc)
                .unwrap();

        let samples = coarse_scan(&satellite, station, start, end, 60);

        assert_eq!(samples.len(), 11);
        assert_eq!(samples.last().expect("end sample").0, end);
        assert_eq!(samples.iter().filter(|(dt, _)| *dt == end).count(), 1);
    }

    // Cross-validated against Skyfield 1.54 (sgp4 2.25, WGS72, AFSPC opsmode
    // 'a', current IERS finals2000A.all). The ISS element set is fed to
    // sgp4.Satrec.sgp4init, wrapped in skyfield EarthSatellite.from_satrec, and
    // the look angle is (sat - wgs84.latlon(51.5, -0.1, 11)).at(t).altaz() at
    // 2024-01-01T12:00:00Z. Skyfield reference: az = 255.645831085991 deg,
    // el = -37.079388752166 deg, range = 8348.734510155984 km. Measured
    // sidereon-vs-Skyfield residuals: az and el below 1e-12 deg, range
    // 1.6e-11 km. sidereon mirrors Skyfield's altaz path and this epoch's EOP
    // is final, so they agree to float noise; the tolerances stay well above
    // that while catching any real regression.
    #[test]
    fn iss_london_look_angle_matches_skyfield() {
        const SKYFIELD_AZIMUTH_DEG: f64 = 255.645_831_085_991;
        const SKYFIELD_ELEVATION_DEG: f64 = -37.079_388_752_166;
        const SKYFIELD_RANGE_KM: f64 = 8348.734510155984;
        const ANGLE_TOL_DEG: f64 = 1.0e-9;
        const RANGE_TOL_KM: f64 = 1.0e-6;

        let datetime = UtcInstant::from_utc(2024, 1, 1, 12, 0, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5,
            longitude_deg: -0.1,
            altitude_m: 11.0,
        };

        let look = look_angle(&iss_2024_01_01_elements(), station, datetime).unwrap();

        assert!(
            (look.azimuth_deg - SKYFIELD_AZIMUTH_DEG).abs() <= ANGLE_TOL_DEG,
            "az {} vs skyfield {SKYFIELD_AZIMUTH_DEG}",
            look.azimuth_deg
        );
        assert!(
            (look.elevation_deg - SKYFIELD_ELEVATION_DEG).abs() <= ANGLE_TOL_DEG,
            "el {} vs skyfield {SKYFIELD_ELEVATION_DEG}",
            look.elevation_deg
        );
        assert!(
            (look.range_km - SKYFIELD_RANGE_KM).abs() <= RANGE_TOL_KM,
            "range {} vs skyfield {SKYFIELD_RANGE_KM}",
            look.range_km
        );
    }

    #[test]
    fn look_angle_rejects_out_of_range_datetime_without_panicking() {
        let station = GroundStation {
            latitude_deg: 51.5,
            longitude_deg: -0.1,
            altitude_m: 11.0,
        };

        for unix_microseconds in [i64::MIN, i64::MAX] {
            assert_eq!(
                look_angle(
                    &iss_2024_01_01_elements(),
                    station,
                    UtcInstant::from_unix_microseconds(unix_microseconds),
                ),
                Err(LookAngleError::InvalidInput {
                    field: "datetime",
                    reason: "invalid UTC instant"
                })
            );
        }
    }

    #[test]
    fn look_angle_apis_reject_invalid_station() {
        let datetime = UtcInstant::from_utc(2024, 1, 1, 12, 0, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: f64::NAN,
            longitude_deg: -0.1,
            altitude_m: 11.0,
        };
        let elements = iss_2024_01_01_elements();

        assert_invalid_look_angle_field(
            look_angle(&elements, station, datetime).unwrap_err(),
            "ground_station.latitude_deg",
            "not finite",
        );

        let satellite = Satellite::from_elements_with_opsmode(&elements, OpsMode::Afspc).unwrap();
        assert_invalid_look_angle_field(
            look_angle_arc(&satellite, station, &[datetime]).unwrap_err(),
            "ground_station.latitude_deg",
            "not finite",
        );
    }

    struct SkyfieldVis {
        catalog: &'static str,
        azimuth_deg: f64,
        elevation_deg: f64,
        range_km: f64,
        teme_position_km: Option<[f64; 3]>,
    }

    fn assert_visible_matches_skyfield(got: &VisibleSatellite, want: &SkyfieldVis) {
        const ANGLE_TOL_DEG: f64 = 5.0e-6;
        const RANGE_TOL_KM: f64 = 5.0e-4;
        const POSITION_TOL_KM: f64 = 5.0e-4;

        assert_eq!(got.catalog_number, want.catalog);
        assert!(
            (got.azimuth_deg - want.azimuth_deg).abs() <= ANGLE_TOL_DEG,
            "{} az {} vs skyfield {}",
            want.catalog,
            got.azimuth_deg,
            want.azimuth_deg
        );
        assert!(
            (got.elevation_deg - want.elevation_deg).abs() <= ANGLE_TOL_DEG,
            "{} el {} vs skyfield {}",
            want.catalog,
            got.elevation_deg,
            want.elevation_deg
        );
        assert!(
            (got.range_km - want.range_km).abs() <= RANGE_TOL_KM,
            "{} range {} vs skyfield {}",
            want.catalog,
            got.range_km,
            want.range_km
        );
        if let Some(teme) = want.teme_position_km {
            for (axis, &component) in teme.iter().enumerate() {
                assert!(
                    (got.position_km[axis] - component).abs() <= POSITION_TOL_KM,
                    "{} teme[{axis}] {} vs skyfield {component}",
                    want.catalog,
                    got.position_km[axis]
                );
            }
        }
    }

    // Cross-validated against Skyfield 1.54 (sgp4 2.25, WGS72, AFSPC opsmode
    // 'a', current IERS finals2000A.all) at 2026-04-05T13:16:46.804800Z from
    // London (51.5074, -0.1278, 11 m). Each element set is fed to
    // sgp4.Satrec.sgp4init and wrapped in skyfield EarthSatellite.from_satrec;
    // az/el/range come from (sat - station).at(t).altaz(); the TEME position
    // comes from sgp4.Satrec.sgp4 at the matching UTC Julian date (the frame
    // VisibleSatellite.position_km reports). Skyfield references (deg / km):
    //   49271: az 157.417716366442  el -29.164801401540  range 8438.433859058045
    //          teme [4122.533671989, 6040.546926999, -2484.416093432]
    //   25544: az 272.823261379335  el -44.228568303544  range 9483.053279562850
    //   48274: az 309.090826171057  el -68.048200809472  range 12241.755488755362
    // Measured sidereon-vs-Skyfield residuals across the three: az <= 1.6e-6 deg,
    // el <= 6.0e-7 deg, range <= 9.0e-5 km, TEME position <= 1.1e-4 km. The
    // sub-arcsec / sub-decimeter spread is dominated by polar motion (sidereon's
    // look-angle path omits it) at a predicted-EOP epoch; the tolerances below
    // sit a few factors above those residuals.
    #[test]
    fn constellation_visible_from_matches_skyfield() {
        let datetime = UtcInstant::from_utc(2026, 4, 5, 13, 16, 46, 804_800).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let members = vec![
            ConstellationMember {
                catalog_number: "25544".to_string(),
                elements: iss_fixture_elements(),
            },
            ConstellationMember {
                catalog_number: "48274".to_string(),
                elements: css_fixture_elements(),
            },
            ConstellationMember {
                catalog_number: "49271".to_string(),
                elements: fregat_fixture_elements(),
            },
        ];

        let visible =
            visible_from_constellation(&members, station, datetime, -90.0).expect("valid station");

        assert_eq!(
            visible
                .iter()
                .map(|sat| sat.catalog_number.as_str())
                .collect::<Vec<_>>(),
            ["49271", "25544", "48274"]
        );

        assert_visible_matches_skyfield(
            &visible[0],
            &SkyfieldVis {
                catalog: "49271",
                azimuth_deg: 157.417716366442,
                elevation_deg: -29.16480140154,
                range_km: 8438.433859058045,
                teme_position_km: Some([4122.533671989, 6040.546926999, -2484.416093432]),
            },
        );
        assert_visible_matches_skyfield(
            &visible[1],
            &SkyfieldVis {
                catalog: "25544",
                azimuth_deg: 272.823_261_379_335,
                elevation_deg: -44.228_568_303_544,
                range_km: 9483.05327956285,
                teme_position_km: None,
            },
        );
        assert_visible_matches_skyfield(
            &visible[2],
            &SkyfieldVis {
                catalog: "48274",
                azimuth_deg: 309.090_826_171_057,
                elevation_deg: -68.048_200_809_472,
                range_km: 12241.755488755362,
                teme_position_km: None,
            },
        );

        assert!(
            visible_from_constellation(&members, station, datetime, -20.0)
                .expect("valid station")
                .is_empty()
        );
    }

    #[test]
    fn visible_from_satellites_threads_opsmode() {
        let datetime = UtcInstant::from_utc(2026, 4, 5, 13, 16, 46, 804_800).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        // A deep-space (near-GEO) object: the only path on which Afspc vs Improved
        // opsmode actually changes the propagated TEME state.
        let elements = geo_like_fixture_elements();
        let afspc = Satellite::from_elements_with_opsmode(&elements, OpsMode::Afspc).unwrap();
        let improved = Satellite::from_elements_with_opsmode(&elements, OpsMode::Improved).unwrap();
        let ids = vec!["99001".to_string()];

        let vis_afspc =
            visible_from_satellites(std::slice::from_ref(&afspc), &ids, station, datetime, -90.0)
                .unwrap();
        let vis_improved = visible_from_satellites(
            std::slice::from_ref(&improved),
            &ids,
            station,
            datetime,
            -90.0,
        )
        .unwrap();

        assert_eq!(vis_afspc.len(), 1);
        assert_eq!(vis_improved.len(), 1);
        // Opsmode is honored end-to-end: the look angles differ between modes.
        assert_ne!(vis_afspc[0], vis_improved[0]);
        assert!(
            (vis_afspc[0].azimuth_deg - vis_improved[0].azimuth_deg).abs()
                + (vis_afspc[0].elevation_deg - vis_improved[0].elevation_deg).abs()
                > 1.0e-9,
            "afspc {:?} vs improved {:?}",
            vis_afspc[0],
            vis_improved[0]
        );

        // The Afspc satellite path reproduces the element-based (Afspc-hardcoded)
        // path bit-for-bit.
        let members = vec![ConstellationMember {
            catalog_number: "99001".to_string(),
            elements: elements.clone(),
        }];
        let element_based = visible_from_constellation(&members, station, datetime, -90.0).unwrap();
        assert_eq!(vis_afspc, element_based);
    }

    #[test]
    fn visible_from_satellites_filters_and_sorts() {
        let datetime = UtcInstant::from_utc(2026, 4, 5, 13, 16, 46, 804_800).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let sats = vec![
            Satellite::from_elements_with_opsmode(&iss_fixture_elements(), OpsMode::Afspc).unwrap(),
            Satellite::from_elements_with_opsmode(&css_fixture_elements(), OpsMode::Afspc).unwrap(),
            Satellite::from_elements_with_opsmode(&fregat_fixture_elements(), OpsMode::Afspc)
                .unwrap(),
        ];
        let ids = vec![
            "25544".to_string(),
            "48274".to_string(),
            "49271".to_string(),
        ];

        let visible = visible_from_satellites(&sats, &ids, station, datetime, -90.0).unwrap();
        // Sorted by elevation descending (same order the element-based path gives
        // for these cross-validated fixtures).
        assert_eq!(
            visible
                .iter()
                .map(|s| s.catalog_number.as_str())
                .collect::<Vec<_>>(),
            ["49271", "25544", "48274"]
        );
        for pair in visible.windows(2) {
            assert!(pair[0].elevation_deg >= pair[1].elevation_deg);
        }

        // All three sit well below the horizon here, so a -20 deg mask drops them.
        assert!(
            visible_from_satellites(&sats, &ids, station, datetime, -20.0)
                .unwrap()
                .is_empty()
        );

        // A mismatched id/satellite count is rejected.
        assert!(matches!(
            visible_from_satellites(&sats, &ids[..2], station, datetime, -90.0),
            Err(PassError::InvalidInput { field: "ids", .. })
        ));
    }

    #[test]
    fn ground_track_subpoint_is_consistent() {
        let elements = iss_2024_01_01_elements();
        let satellite =
            Satellite::from_elements_with_opsmode(&elements, OpsMode::Improved).unwrap();
        let epochs: Vec<UtcInstant> = (0..8)
            .map(|i| UtcInstant::from_utc(2024, 1, 1, 12, i, 0, 0).unwrap())
            .collect();

        let track = ground_track(&satellite, &epochs).unwrap();
        assert_eq!(track.len(), epochs.len());

        let inclination_rad = elements.inclination_deg.to_radians();
        for point in &track {
            // Sub-point latitude is bounded by the orbital inclination (a small
            // margin covers nodal/oblateness wobble).
            assert!(
                point.lat_rad.abs() <= inclination_rad + 1.0e-3,
                "lat {} exceeds inclination {}",
                point.lat_rad,
                inclination_rad
            );
            // ISS-class altitude.
            let alt_km = point.height_m / 1000.0;
            assert!(
                (300.0..=600.0).contains(&alt_km),
                "alt {alt_km} km implausible"
            );
        }

        // Longitude advances monotonically (single direction) over the short arc,
        // accounting for the +-pi wrap.
        let mut deltas = Vec::new();
        for pair in track.windows(2) {
            let mut dlon = pair[1].lon_rad - pair[0].lon_rad;
            if dlon > std::f64::consts::PI {
                dlon -= std::f64::consts::TAU;
            } else if dlon < -std::f64::consts::PI {
                dlon += std::f64::consts::TAU;
            }
            deltas.push(dlon);
        }
        assert!(
            deltas.iter().all(|&d| d > 0.0) || deltas.iter().all(|&d| d < 0.0),
            "longitude not monotonic: {deltas:?}"
        );

        // One epoch matches a manual propagate -> TEME->GCRS -> GCRS->ITRS ->
        // geodetic composition bit-for-bit.
        let dt = epochs[3];
        let ts = time_scales_for_look_angle(dt).unwrap();
        let pred = satellite.propagate_jd(dt.sgp4_julian_date()).unwrap();
        let (gcrs, _) = teme_to_gcrs_compute(
            &TemeStateKm {
                position_km: pred.position,
                velocity_km_s: pred.velocity,
            },
            &ts,
            false,
        )
        .unwrap();
        let (x_km, y_km, z_km) = gcrs_to_itrs_compute(gcrs.0, gcrs.1, gcrs.2, &ts, false).unwrap();
        let (lat_deg, lon_deg, alt_km) = itrs_to_geodetic_compute(x_km, y_km, z_km).unwrap();
        assert_eq!(track[3].lat_rad.to_bits(), lat_deg.to_radians().to_bits());
        assert_eq!(track[3].lon_rad.to_bits(), lon_deg.to_radians().to_bits());
        assert_eq!(track[3].height_m.to_bits(), (alt_km * 1000.0).to_bits());
    }

    #[test]
    fn look_angle_arc_reproduces_single_london_golden_bits() {
        // The arc walker, fed the same AFSPC satellite the single-shot
        // `look_angle` builds internally, must reproduce the frozen London golden
        // bit-for-bit at the matching instant.
        let datetime = UtcInstant::from_utc(2024, 1, 1, 12, 0, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5,
            longitude_deg: -0.1,
            altitude_m: 11.0,
        };
        let satellite =
            Satellite::from_elements_with_opsmode(&iss_2024_01_01_elements(), OpsMode::Afspc)
                .unwrap();

        let arc = look_angle_arc(&satellite, station, &[datetime]).unwrap();

        assert_eq!(arc.len(), 1);
        assert_eq!(arc[0].azimuth_deg.to_bits(), 0x406f_f4aa_a5f4_2254);
        assert_eq!(arc[0].elevation_deg.to_bits(), 0xc042_8a29_691f_1ca2);
        assert_eq!(arc[0].range_km.to_bits(), 0x40c0_4e5e_046d_c53b);
    }

    #[test]
    fn arc_walkers_match_per_epoch_calls_bit_for_bit() {
        // Building the satellite once and stepping it over the grid must equal
        // the per-epoch single-shot calls exactly (no satrec drift).
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let elements = iss_2024_01_01_elements();
        let satellite = Satellite::from_elements_with_opsmode(&elements, OpsMode::Afspc).unwrap();
        let epochs: Vec<UtcInstant> = (0..6)
            .map(|i| UtcInstant::from_utc(2024, 1, 1, 12, 10 * i, 0, 0).unwrap())
            .collect();

        let pos_arc = propagate_teme_arc(&satellite, &epochs).unwrap();
        let look_arc = look_angle_arc(&satellite, station, &epochs).unwrap();

        for (i, &datetime) in epochs.iter().enumerate() {
            let single_pred = satellite.propagate_jd(datetime.sgp4_julian_date()).unwrap();
            let single_look = look_angle(&elements, station, datetime).unwrap();
            for axis in 0..3 {
                assert_eq!(
                    pos_arc[i].position[axis].to_bits(),
                    single_pred.position[axis].to_bits()
                );
                assert_eq!(
                    pos_arc[i].velocity[axis].to_bits(),
                    single_pred.velocity[axis].to_bits()
                );
            }
            assert_eq!(
                look_arc[i].azimuth_deg.to_bits(),
                single_look.azimuth_deg.to_bits()
            );
            assert_eq!(
                look_arc[i].elevation_deg.to_bits(),
                single_look.elevation_deg.to_bits()
            );
            assert_eq!(
                look_arc[i].range_km.to_bits(),
                single_look.range_km.to_bits()
            );
        }
    }

    #[test]
    fn parallel_batch_is_bit_identical_to_serial_multi_sat() {
        // A real multi-satellite fixture (LEO + Tiangong + a high-eccentricity
        // Fregat upper stage, plus the two ISS epochs) propagated over a full
        // day of epochs. The rayon-parallel batch must equal the single-threaded
        // batch to_bits, element by element, for both the TEME states and the
        // topocentric look angles.
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let element_sets = [
            iss_fixture_elements(),
            css_fixture_elements(),
            fregat_fixture_elements(),
            iss_2024_01_01_elements(),
            iss_2024_12_19_elements(),
        ];
        let satellites: Vec<Satellite> = element_sets
            .iter()
            .map(|e| Satellite::from_elements_with_opsmode(e, OpsMode::Afspc).unwrap())
            .collect();
        // 145 epochs spaced 10 minutes apart (a full day) so the parallel pool
        // has many independent arcs to interleave.
        let base = UtcInstant::from_utc(2026, 4, 5, 0, 0, 0, 0).unwrap();
        let epochs: Vec<UtcInstant> = (0..145)
            .map(|i| base.add_microseconds(i * 600 * MICROSECONDS_PER_SECOND_I64))
            .collect();

        let pos_serial = propagate_teme_batch_serial(&satellites, &epochs);
        let pos_parallel = propagate_teme_batch_parallel(&satellites, &epochs);
        let look_serial = look_angle_batch_serial(&satellites, station, &epochs);
        let look_parallel = look_angle_batch_parallel(&satellites, station, &epochs);

        assert_eq!(pos_serial.len(), satellites.len());
        assert_eq!(pos_parallel.len(), satellites.len());
        assert_eq!(look_serial.len(), satellites.len());
        assert_eq!(look_parallel.len(), satellites.len());

        for sat_idx in 0..satellites.len() {
            let serial_arc = pos_serial[sat_idx].as_ref().expect("serial arc ok");
            let parallel_arc = pos_parallel[sat_idx].as_ref().expect("parallel arc ok");
            // Tie the batch back to the per-satellite arc walker (which is itself
            // pinned to the single-shot SGP4 goldens) so this is anchored to the
            // existing 0-ULP reference, not just serial==parallel.
            let reference_arc = propagate_teme_arc(&satellites[sat_idx], &epochs).unwrap();
            assert_eq!(serial_arc.len(), epochs.len());
            assert_eq!(parallel_arc.len(), epochs.len());
            for epoch_idx in 0..epochs.len() {
                for axis in 0..3 {
                    let s = serial_arc[epoch_idx].position[axis].to_bits();
                    let p = parallel_arc[epoch_idx].position[axis].to_bits();
                    let r = reference_arc[epoch_idx].position[axis].to_bits();
                    assert_eq!(
                        s, p,
                        "position bits sat {sat_idx} epoch {epoch_idx} axis {axis}"
                    );
                    assert_eq!(
                        s, r,
                        "position vs arc reference sat {sat_idx} epoch {epoch_idx}"
                    );
                    let sv = serial_arc[epoch_idx].velocity[axis].to_bits();
                    let pv = parallel_arc[epoch_idx].velocity[axis].to_bits();
                    let rv = reference_arc[epoch_idx].velocity[axis].to_bits();
                    assert_eq!(
                        sv, pv,
                        "velocity bits sat {sat_idx} epoch {epoch_idx} axis {axis}"
                    );
                    assert_eq!(
                        sv, rv,
                        "velocity vs arc reference sat {sat_idx} epoch {epoch_idx}"
                    );
                }
            }

            let look_s = look_serial[sat_idx].as_ref().expect("serial look ok");
            let look_p = look_parallel[sat_idx].as_ref().expect("parallel look ok");
            assert_eq!(look_s.len(), epochs.len());
            assert_eq!(look_p.len(), epochs.len());
            for epoch_idx in 0..epochs.len() {
                assert_eq!(
                    look_s[epoch_idx].azimuth_deg.to_bits(),
                    look_p[epoch_idx].azimuth_deg.to_bits(),
                    "azimuth bits sat {sat_idx} epoch {epoch_idx}"
                );
                assert_eq!(
                    look_s[epoch_idx].elevation_deg.to_bits(),
                    look_p[epoch_idx].elevation_deg.to_bits(),
                    "elevation bits sat {sat_idx} epoch {epoch_idx}"
                );
                assert_eq!(
                    look_s[epoch_idx].range_km.to_bits(),
                    look_p[epoch_idx].range_km.to_bits(),
                    "range bits sat {sat_idx} epoch {epoch_idx}"
                );
            }
        }
    }

    #[test]
    fn find_passes_batch_parallel_matches_serial_multi_sat() {
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let element_sets = [
            iss_fixture_elements(),
            css_fixture_elements(),
            fregat_fixture_elements(),
            iss_2024_12_19_elements(),
        ];
        let start = UtcInstant::from_utc(2026, 4, 5, 0, 0, 0, 0).unwrap();
        let end = UtcInstant::from_utc(2026, 4, 6, 0, 0, 0, 0).unwrap();
        let options = PassFinderOptions {
            elevation_mask_deg: 0.0,
            coarse_step_seconds: 600.0,
            time_tolerance_seconds: 1.0e-3,
        };

        let serial = find_passes_batch_serial(&element_sets, station, start, end, options);
        let parallel = find_passes_batch_parallel(&element_sets, station, start, end, options);

        assert_eq!(parallel, serial);
        assert_eq!(serial.len(), element_sets.len());
        assert!(serial
            .iter()
            .all(|result| result.as_ref().is_ok_and(|passes| {
                passes
                    .iter()
                    .all(|pass| pass.aos <= pass.culmination && pass.culmination <= pass.los)
            })));
        assert!(serial
            .iter()
            .any(|result| result.as_ref().is_ok_and(|passes| !passes.is_empty())));

        let single = find_passes(&element_sets[0], station, start, end, options)
            .expect("single pass finder succeeds");
        let batch_first = serial[0].as_ref().expect("serial batch element succeeds");
        assert_eq!(batch_first.len(), single.len());
        for (batch_pass, single_pass) in batch_first.iter().zip(&single) {
            assert_pass_close(batch_pass, single_pass, 1_000, 1.0e-6);
        }
    }

    #[test]
    fn segmented_crossings_long_window_match_under_cap_event_finder() {
        let total_us = 181 * SECONDS_PER_DAY_I64 * MICROSECONDS_PER_SECOND_I64;
        let step_us = 15 * MICROSECONDS_PER_SECOND_I64;
        let step_seconds = step_us as f64 / MICROSECONDS_PER_SECOND_I64 as f64;
        let time_tolerance_seconds = 1.0e-3;
        let total_seconds = offset_seconds(total_us);
        assert!(
            (total_seconds / step_seconds).ceil() > EVENT_FINDER_COARSE_SAMPLE_LIMIT as f64,
            "181-day LEO-rate scan should exceed one EventFinder coarse budget"
        );
        assert_eq!(
            EventFinder::new(0.0, total_seconds, step_seconds, time_tolerance_seconds)
                .expect("valid long event window")
                .find_crossings(synthetic_pass_wave, 0.0)
                .unwrap_err(),
            EventFinderError::InvalidInput {
                field: "step_seconds",
                reason: "too many samples",
            }
        );

        let segmented = find_segmented_crossings(
            synthetic_pass_wave,
            total_us,
            step_us,
            step_seconds,
            time_tolerance_seconds,
        )
        .expect("segmented long-window crossings succeed");
        assert!(!segmented.is_empty());

        let boundary_us = event_finder_segment_stride_us(step_us);
        assert!(0 < boundary_us && boundary_us < total_us);
        let reference_start_us = boundary_us - 12 * 60 * 60 * MICROSECONDS_PER_SECOND_I64;
        let reference_end_us = boundary_us + 12 * 60 * 60 * MICROSECONDS_PER_SECOND_I64;
        let reference = EventFinder::new(
            offset_seconds(reference_start_us),
            offset_seconds(reference_end_us),
            step_seconds,
            time_tolerance_seconds,
        )
        .expect("valid under-cap reference window")
        .find_crossings(synthetic_pass_wave, 0.0)
        .expect("under-cap reference crossings succeed");
        let segmented_boundary: Vec<_> = segmented
            .iter()
            .copied()
            .filter(|crossing| {
                let crossing_us = crossing_offset_us(*crossing);
                reference_start_us <= crossing_us && crossing_us <= reference_end_us
            })
            .collect();

        assert!(!reference.is_empty());
        assert_eq!(segmented_boundary.len(), reference.len());
        for (actual, expected) in segmented_boundary.iter().zip(&reference) {
            assert_eq!(actual.direction, expected.direction);
            assert!(
                (actual.time_seconds - expected.time_seconds).abs() <= time_tolerance_seconds,
                "segmented event time diverged from under-cap EventFinder reference"
            );
        }
    }

    #[test]
    fn segmented_crossings_boundary_plateaus_match_under_cap_event_finder() {
        let step_us = 10;
        let step_seconds = offset_seconds(step_us);
        let time_tolerance_seconds = 1.0e-12;
        let total_us = max_event_finder_segment_us(step_us) + 1_000;
        let boundary_us = event_finder_segment_stride_us(step_us);
        let reference_start_us = boundary_us - 2 * step_us;
        let reference_end_us = boundary_us + 4 * step_us;
        let plateau = |right_value: f64| {
            move |time_seconds: f64| {
                let offset_us = (time_seconds * MICROSECONDS_PER_SECOND_I64 as f64).round() as i64;
                if boundary_us <= offset_us && offset_us <= boundary_us + 2 * step_us {
                    0.0
                } else if offset_us < boundary_us {
                    -1.0
                } else {
                    right_value
                }
            }
        };

        let same_side = plateau(-1.0);
        let same_reference = EventFinder::new(
            offset_seconds(reference_start_us),
            offset_seconds(reference_end_us),
            step_seconds,
            time_tolerance_seconds,
        )
        .expect("valid same-side reference window")
        .find_crossings(same_side, 0.0)
        .expect("same-side reference scan succeeds");
        let same_segmented = find_segmented_crossings(
            same_side,
            total_us,
            step_us,
            step_seconds,
            time_tolerance_seconds,
        )
        .expect("same-side segmented scan succeeds");
        let same_segmented_boundary: Vec<_> = same_segmented
            .iter()
            .copied()
            .filter(|crossing| {
                let crossing_us = crossing_offset_us(*crossing);
                reference_start_us <= crossing_us && crossing_us <= reference_end_us
            })
            .collect();

        assert!(same_reference.is_empty());
        assert_eq!(same_segmented_boundary, same_reference);

        let opposite_side = plateau(1.0);
        let opposite_reference = EventFinder::new(
            offset_seconds(reference_start_us),
            offset_seconds(reference_end_us),
            step_seconds,
            time_tolerance_seconds,
        )
        .expect("valid opposite-side reference window")
        .find_crossings(opposite_side, 0.0)
        .expect("opposite-side reference scan succeeds");
        let opposite_segmented = find_segmented_crossings(
            opposite_side,
            total_us,
            step_us,
            step_seconds,
            time_tolerance_seconds,
        )
        .expect("opposite-side segmented scan succeeds");
        let opposite_segmented_boundary: Vec<_> = opposite_segmented
            .iter()
            .copied()
            .filter(|crossing| {
                let crossing_us = crossing_offset_us(*crossing);
                reference_start_us <= crossing_us && crossing_us <= reference_end_us
            })
            .collect();

        assert_eq!(opposite_reference.len(), 1);
        assert_eq!(opposite_segmented_boundary.len(), opposite_reference.len());
        for (actual, expected) in opposite_segmented_boundary.iter().zip(&opposite_reference) {
            assert_eq!(actual.direction, expected.direction);
            assert!(
                (crossing_offset_us(*actual) - crossing_offset_us(*expected)).abs() <= 1,
                "segmented plateau crossing time diverged from under-cap reference"
            );
        }
    }

    #[test]
    fn segmented_culmination_long_window_stays_under_event_sample_cap() {
        let start = UtcInstant::from_unix_microseconds(0);
        let step_us = 10;
        let total_us = max_event_finder_segment_us(step_us) + 1_000;
        let peak_offset_us = event_finder_segment_stride_us(step_us) + 500;
        let end = start.add_microseconds(total_us);
        let peak_time = start.add_microseconds(peak_offset_us);
        let step_seconds = offset_seconds(step_us);
        let time_tolerance_seconds = 1.0e-6;
        let elevation = |dt: UtcInstant| {
            let centered_us = (dt.diff_microseconds(start) - peak_offset_us) as f64;
            42.0 - (centered_us / 10_000.0).powi(2)
        };

        assert!(peak_offset_us > max_event_finder_segment_us(step_us));
        assert_eq!(
            EventFinder::new(
                0.0,
                offset_seconds(total_us),
                step_seconds,
                time_tolerance_seconds
            )
            .expect("valid long event window")
            .find_extrema(|offset_seconds| elevation(instant_at_offset_seconds(
                start,
                offset_seconds
            )))
            .unwrap_err(),
            EventFinderError::InvalidInput {
                field: "step_seconds",
                reason: "too many samples",
            }
        );

        let (culmination, max_elevation) = find_culmination_with(start, end, step_us, 1, elevation)
            .expect("segmented long-window culmination succeeds");

        assert!(
            (culmination.unix_microseconds() - peak_time.unix_microseconds()).abs() <= 1,
            "culmination should stay on the synthetic peak"
        );
        assert!(
            (max_elevation - elevation(peak_time)).abs() <= 1.0e-12,
            "max elevation should come from the synthetic peak"
        );
    }

    #[test]
    fn segmented_extrema_short_window_matches_direct_event_finder() {
        let start = UtcInstant::from_unix_microseconds(0);
        let total_us = 10 * MICROSECONDS_PER_SECOND_I64;
        let step_us = 500_000;
        let step_seconds = offset_seconds(step_us);
        let time_tolerance_seconds = 1.0e-4;
        let elevation = |offset_seconds| {
            multi_stationary_elevation(instant_at_offset_seconds(start, offset_seconds))
        };

        let direct = EventFinder::new(
            0.0,
            offset_seconds(total_us),
            step_seconds,
            time_tolerance_seconds,
        )
        .expect("valid short event window")
        .find_extrema(elevation)
        .expect("direct short-window extrema succeeds");
        let segmented = find_segmented_extrema(
            elevation,
            total_us,
            step_us,
            step_seconds,
            time_tolerance_seconds,
        )
        .expect("segmented short-window extrema succeeds");

        assert_eq!(segmented, direct);
    }

    #[test]
    fn find_passes_matches_predict_passes_and_resists_coarse_drop() {
        let start = UtcInstant::from_utc(2024, 12, 19, 0, 0, 0, 0).unwrap();
        let end = UtcInstant::from_utc(2024, 12, 20, 0, 0, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let elements = iss_2024_12_19_elements();

        // Existing reference path at a fine step.
        let reference = predict_passes(
            &elements,
            station,
            start,
            end,
            PassPredictionOptions {
                min_elevation_deg: 0.0,
                step_seconds: 10,
            },
        )
        .expect("valid pass-prediction step");
        // Dense finder at mask 0 must reproduce it.
        let mine = find_passes(
            &elements,
            station,
            start,
            end,
            PassFinderOptions {
                elevation_mask_deg: 0.0,
                coarse_step_seconds: 10.0,
                time_tolerance_seconds: 1.0e-3,
            },
        )
        .expect("valid pass-finder step");
        // The same reference path at a coarse step drops passes.
        let coarse = predict_passes(
            &elements,
            station,
            start,
            end,
            PassPredictionOptions {
                min_elevation_deg: 0.0,
                step_seconds: 900,
            },
        )
        .expect("valid pass-prediction step");

        assert!(reference.len() >= 2);
        assert_eq!(mine.len(), reference.len());
        assert!(coarse.len() < reference.len());

        for (m, r) in mine.iter().zip(reference.iter()) {
            assert!(
                (m.aos.unix_microseconds() - r.rise.unix_microseconds()).abs() < 1_000,
                "AOS within 1 ms of reference rise"
            );
            assert!(
                (m.los.unix_microseconds() - r.set.unix_microseconds()).abs() < 1_000,
                "LOS within 1 ms of reference set"
            );
            assert!(
                (m.culmination.unix_microseconds() - r.max_elevation_time.unix_microseconds())
                    .abs()
                    < 1_000,
                "culmination within 1 ms of reference peak time"
            );
            assert!(
                (m.max_elevation_deg - r.max_elevation_deg).abs() < 1.0e-7,
                "max elevation within 1e-7 deg of reference"
            );
        }
    }

    #[test]
    fn find_passes_extrema_culmination_matches_predict_passes_peak() {
        let start = UtcInstant::from_utc(2024, 12, 19, 0, 0, 0, 0).unwrap();
        let end = UtcInstant::from_utc(2024, 12, 19, 12, 0, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let elements = iss_2024_12_19_elements();
        let reference = predict_passes(
            &elements,
            station,
            start,
            end,
            PassPredictionOptions {
                min_elevation_deg: 0.0,
                step_seconds: 10,
            },
        )
        .expect("valid pass-prediction step");
        let found = find_passes(
            &elements,
            station,
            start,
            end,
            PassFinderOptions {
                elevation_mask_deg: 0.0,
                coarse_step_seconds: 10.0,
                time_tolerance_seconds: 1.0e-4,
            },
        )
        .expect("valid pass-finder step");

        assert_eq!(reference.len(), 1);
        assert_eq!(found.len(), reference.len());
        let expected = reference[0];
        let actual = found[0];
        assert!(
            (actual.culmination.unix_microseconds()
                - expected.max_elevation_time.unix_microseconds())
            .abs()
                < 1_000,
            "culmination within 1 ms of legacy peak time"
        );
        assert!(
            (actual.max_elevation_deg - expected.max_elevation_deg).abs() < 1.0e-7,
            "max elevation within 1e-7 deg of legacy peak"
        );

        let satellite = Satellite::from_elements_with_opsmode(&elements, OpsMode::Afspc).unwrap();
        let span_seconds =
            actual.los.diff_microseconds(actual.aos) as f64 / MICROSECONDS_PER_SECOND_I64 as f64;
        let extrema = EventFinder::new(0.0, span_seconds, 10.0, 1.0e-4)
            .expect("valid event-finder window")
            .find_extrema(|offset_seconds| {
                elevation_at(
                    &satellite,
                    instant_at_offset_seconds(actual.aos, offset_seconds),
                    station,
                )
            })
            .expect("finite elevation predicate");
        let peak = extrema
            .iter()
            .filter(|event| event.kind == ExtremumKind::Maximum)
            .max_by(|a, b| {
                a.value
                    .partial_cmp(&b.value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("pass has an interior maximum");
        let peak_time = instant_at_offset_seconds(actual.aos, peak.time_seconds);

        assert!(
            (actual.culmination.unix_microseconds() - peak_time.unix_microseconds()).abs() <= 1_000,
            "pass culmination follows the shared extrema finder"
        );
        assert!(
            (actual.max_elevation_deg - elevation_at(&satellite, peak_time, station)).abs()
                < 1.0e-7,
            "pass max elevation follows the shared extrema finder"
        );
    }

    #[test]
    fn robust_crossing_step_honors_requested_step_for_geo_like_orbit() {
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let satellite =
            Satellite::from_elements_with_opsmode(&geo_like_fixture_elements(), OpsMode::Afspc)
                .expect("GEO-like fixture initializes");
        let requested_step_us = 900 * MICROSECONDS_PER_SECOND_I64;
        let orbit_step_us = fastest_orbit_fraction_step_us(&satellite)
            .expect("GEO-like orbit has a finite safe step");

        assert!(orbit_step_us >= requested_step_us);
        assert_eq!(
            robust_crossing_step_us(&satellite, station, 0.0, requested_step_us),
            requested_step_us
        );
    }

    #[test]
    fn robust_crossing_step_substeps_fast_orbit_by_orbit_bound() {
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let satellite =
            Satellite::from_elements_with_opsmode(&iss_2024_12_19_elements(), OpsMode::Afspc)
                .expect("ISS fixture initializes");
        let requested_step_us = 900 * MICROSECONDS_PER_SECOND_I64;
        let orbit_step_us =
            fastest_orbit_fraction_step_us(&satellite).expect("ISS orbit has a finite safe step");

        assert!(orbit_step_us < requested_step_us);
        assert_eq!(
            robust_crossing_step_us(&satellite, station, 0.0, requested_step_us),
            orbit_step_us
        );
    }

    #[test]
    fn find_passes_geometry_step_finds_high_mask_pass_period_step_skips() {
        let elements = iss_2024_12_19_elements();
        let satellite = Satellite::from_elements_with_opsmode(&elements, OpsMode::Afspc)
            .expect("ISS fixture initializes");
        let peak_seed = UtcInstant::from_utc(2024, 12, 19, 12, 37, 35, 0).unwrap();
        let station = station_under_satellite(&satellite, peak_seed);
        let orbit_step_us =
            fastest_orbit_fraction_step_us(&satellite).expect("ISS orbit has finite step");
        let requested_step_us = 900 * MICROSECONDS_PER_SECOND_I64;

        let reference = find_passes_for_satellite(
            &satellite,
            station,
            peak_seed.add_microseconds(-30 * 60 * MICROSECONDS_PER_SECOND_I64),
            peak_seed.add_microseconds(30 * 60 * MICROSECONDS_PER_SECOND_I64),
            PassFinderOptions {
                elevation_mask_deg: 0.0,
                coarse_step_seconds: 1.0,
                time_tolerance_seconds: 1.0e-3,
            },
        )
        .expect("fine overhead reference succeeds");
        let reference_pass = reference
            .iter()
            .max_by(|a, b| {
                a.max_elevation_deg
                    .partial_cmp(&b.max_elevation_deg)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("overhead station has a visible ISS pass");
        let mask = 85.0;
        assert!(
            reference_pass.max_elevation_deg > mask,
            "subpoint pass should clear the high mask"
        );

        let half_phase_us = orbit_step_us / 2;
        let side_us = 20 * orbit_step_us + half_phase_us;
        let start = reference_pass.culmination.add_microseconds(-side_us);
        let end = reference_pass.culmination.add_microseconds(side_us);
        let span_seconds = end.diff_microseconds(start) as f64 / MICROSECONDS_PER_SECOND_I64 as f64;
        let orbit_step_seconds = orbit_step_us as f64 / MICROSECONDS_PER_SECOND_I64 as f64;

        let period_only_crossings = EventFinder::new(0.0, span_seconds, orbit_step_seconds, 1.0e-3)
            .expect("valid period-only window")
            .find_crossings(
                PassSearch {
                    satellite: &satellite,
                    ground_station: station,
                    start_time: start,
                    end_time: end,
                    step_us: orbit_step_us,
                    tol_us: 1_000,
                    mask,
                },
                0.0,
            )
            .expect("finite period-only masked elevation predicate");
        assert!(
            period_only_crossings.is_empty(),
            "period-only robust step should skip this narrow high-mask pass"
        );

        let geometry_step_us =
            robust_crossing_step_us(&satellite, station, mask, requested_step_us);
        assert!(
            geometry_step_us < orbit_step_us,
            "high-mask geometry step should refine below the orbital step"
        );

        let fine = find_passes_for_satellite(
            &satellite,
            station,
            start,
            end,
            PassFinderOptions {
                elevation_mask_deg: mask,
                coarse_step_seconds: 0.5,
                time_tolerance_seconds: 1.0e-3,
            },
        )
        .expect("fine high-mask pass search succeeds");
        let robust = find_passes_for_satellite(
            &satellite,
            station,
            start,
            end,
            PassFinderOptions {
                elevation_mask_deg: mask,
                coarse_step_seconds: 900.0,
                time_tolerance_seconds: 1.0e-3,
            },
        )
        .expect("geometry-aware high-mask pass search succeeds");

        assert_eq!(fine.len(), 1);
        assert_eq!(robust.len(), fine.len());
        assert_pass_close(&robust[0], &fine[0], 1_000, 1.0e-3);
        assert!(robust[0].max_elevation_deg > mask);
    }

    #[test]
    fn culmination_refines_selected_maximum_not_first_rate_zero() {
        let start = UtcInstant::from_unix_microseconds(0);
        let end = start.add_microseconds(10 * MICROSECONDS_PER_SECOND_I64);

        let whole_window =
            refine_culmination_rate_zero(&multi_stationary_elevation, start, end, 100)
                .expect("whole-window rate signs bracket some stationary point");
        let (culmination, max_elevation) =
            find_culmination_with(start, end, 500_000, 100, multi_stationary_elevation)
                .expect("synthetic pass culmination should resolve");

        let whole_window_offset =
            whole_window.0.diff_microseconds(start) as f64 / MICROSECONDS_PER_SECOND_I64 as f64;
        let selected_offset =
            culmination.diff_microseconds(start) as f64 / MICROSECONDS_PER_SECOND_I64 as f64;

        assert!(
            whole_window_offset < 3.0,
            "whole-window rate bisection locked onto offset {whole_window_offset}"
        );
        assert!(
            (selected_offset - 8.5).abs() < 0.05,
            "selected maximum should refine near 8.5 s, got {selected_offset}"
        );
        assert!(
            max_elevation > whole_window.1 + 15.0,
            "selected culmination should beat the earlier stationary point"
        );
        assert!(
            max_elevation > 25.0,
            "selected maximum elevation should stay on the high synthetic peak"
        );
    }

    #[test]
    fn event_finder_elevation_crossings_match_predict_passes() {
        let start = UtcInstant::from_utc(2024, 12, 19, 0, 0, 0, 0).unwrap();
        let end = UtcInstant::from_utc(2024, 12, 20, 0, 0, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let elements = iss_2024_12_19_elements();
        let satellite = Satellite::from_elements_with_opsmode(&elements, OpsMode::Afspc).unwrap();

        let reference = predict_passes(
            &elements,
            station,
            start,
            end,
            PassPredictionOptions {
                min_elevation_deg: 0.0,
                step_seconds: 10,
            },
        )
        .expect("valid pass-prediction step");
        assert!(reference.len() >= 2);

        let span_seconds = end.diff_microseconds(start) as f64 / MICROSECONDS_PER_SECOND_I64 as f64;
        let crossings = EventFinder::new(0.0, span_seconds, 10.0, 1.0e-4)
            .expect("valid event-finder window")
            .find_crossings(
                |offset_seconds| {
                    elevation_at(
                        &satellite,
                        instant_at_offset_seconds(start, offset_seconds),
                        station,
                    )
                },
                0.0,
            )
            .expect("finite elevation predicate");
        let found = find_passes(
            &elements,
            station,
            start,
            end,
            PassFinderOptions {
                elevation_mask_deg: 0.0,
                coarse_step_seconds: 10.0,
                time_tolerance_seconds: 1.0e-4,
            },
        )
        .expect("valid pass-finder step");

        assert_eq!(crossings.len(), reference.len() * 2);
        assert_eq!(found.len(), reference.len());

        for (pass_index, pass) in reference.iter().enumerate() {
            let rise = crossings[pass_index * 2];
            let set = crossings[pass_index * 2 + 1];
            assert_eq!(rise.direction, CrossingDirection::Rising);
            assert_eq!(set.direction, CrossingDirection::Falling);

            let rise_time = instant_at_offset_seconds(start, rise.time_seconds);
            let set_time = instant_at_offset_seconds(start, set.time_seconds);
            assert!(
                (rise_time.unix_microseconds() - pass.rise.unix_microseconds()).abs() < 1_000,
                "event finder rise within 1 ms of predict_passes"
            );
            assert!(
                (set_time.unix_microseconds() - pass.set.unix_microseconds()).abs() < 1_000,
                "event finder set within 1 ms of predict_passes"
            );
            assert!(
                (found[pass_index].aos.unix_microseconds() - rise_time.unix_microseconds()).abs()
                    <= 100,
                "pass finder AOS follows event-finder rise within snap tolerance"
            );
            assert!(
                (found[pass_index].los.unix_microseconds() - set_time.unix_microseconds()).abs()
                    <= 100,
                "pass finder LOS follows event-finder set within snap tolerance"
            );
        }
    }

    #[test]
    fn event_finder_pass_peak_equal_to_mask_emits_no_crossing() {
        let day_start = UtcInstant::from_utc(2024, 12, 19, 0, 0, 0, 0).unwrap();
        let day_end = UtcInstant::from_utc(2024, 12, 20, 0, 0, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let elements = iss_2024_12_19_elements();
        let satellite = Satellite::from_elements_with_opsmode(&elements, OpsMode::Afspc).unwrap();
        let reference = find_passes(
            &elements,
            station,
            day_start,
            day_end,
            PassFinderOptions {
                elevation_mask_deg: 0.0,
                coarse_step_seconds: 10.0,
                time_tolerance_seconds: 1.0e-3,
            },
        )
        .expect("valid pass-finder step");
        assert!(!reference.is_empty());

        let peak_time = reference[0].culmination;
        let mask = elevation_at(&satellite, peak_time, station);
        let step_us = 10 * MICROSECONDS_PER_SECOND_I64;
        let start = peak_time.add_microseconds(-step_us);
        let end = peak_time.add_microseconds(step_us);
        let search = PassSearch {
            satellite: &satellite,
            ground_station: station,
            start_time: start,
            end_time: end,
            step_us,
            tol_us: 1_000,
            mask,
        };
        assert!(search.masked_elevation_at(start) < 0.0);
        assert_eq!(
            search.masked_elevation_at(peak_time).to_bits(),
            0.0_f64.to_bits()
        );
        assert!(search.masked_elevation_at(end) < 0.0);

        let crossings = EventFinder::new(0.0, 20.0, 10.0, 1.0e-3)
            .expect("valid event-finder window")
            .find_crossings(search, 0.0)
            .expect("finite masked elevation predicate");
        assert!(
            crossings.is_empty(),
            "peak tangent at the mask must not emit AOS/LOS crossings"
        );

        let found = find_passes(
            &elements,
            station,
            start,
            end,
            PassFinderOptions {
                elevation_mask_deg: mask,
                coarse_step_seconds: 10.0,
                time_tolerance_seconds: 1.0e-3,
            },
        )
        .expect("valid pass-finder step");
        assert!(found.is_empty());
    }

    #[test]
    fn assemble_passes_seeds_exact_aos_but_not_exact_los_window_start() {
        let start = UtcInstant::from_utc(2024, 12, 19, 0, 0, 0, 0).unwrap();
        let end = UtcInstant::from_utc(2024, 12, 20, 0, 0, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let elements = iss_2024_12_19_elements();
        let satellite = Satellite::from_elements_with_opsmode(&elements, OpsMode::Afspc).unwrap();
        let options = PassFinderOptions {
            elevation_mask_deg: 0.0,
            coarse_step_seconds: 10.0,
            time_tolerance_seconds: 1.0e-3,
        };

        let reference =
            find_passes(&elements, station, start, end, options).expect("valid pass-finder step");
        assert!(reference.len() >= 2);
        let search_step_us =
            robust_crossing_step_us(&satellite, station, 0.0, 10 * MICROSECONDS_PER_SECOND_I64);

        let aos_at_window_start = reference[0].aos;
        let aos_mask = elevation_at(&satellite, aos_at_window_start, station);
        assert!(
            elevation_at(
                &satellite,
                aos_at_window_start.add_microseconds(-MICROSECONDS_PER_SECOND_I64),
                station,
            ) < aos_mask
        );
        assert!(
            elevation_at(
                &satellite,
                aos_at_window_start.add_microseconds(MICROSECONDS_PER_SECOND_I64),
                station,
            ) > aos_mask
        );

        let aos_search = PassSearch {
            satellite: &satellite,
            ground_station: station,
            start_time: aos_at_window_start,
            end_time: end,
            step_us: search_step_us,
            tol_us: 1_000,
            mask: aos_mask,
        };
        let first_los_offset_seconds = reference[0].los.diff_microseconds(aos_at_window_start)
            as f64
            / MICROSECONDS_PER_SECOND_I64 as f64;
        let aos_found = assemble_passes_from_crossings(
            aos_search,
            vec![CrossingEvent {
                time_seconds: first_los_offset_seconds,
                value: 0.0,
                threshold: 0.0,
                direction: CrossingDirection::Falling,
            }],
        )
        .expect("exact AOS assembly succeeds");

        assert_eq!(aos_found.len(), 1);
        assert_eq!(aos_found[0].aos, aos_at_window_start);
        assert!(aos_found[0].los > aos_at_window_start);

        let los_at_window_start = reference[0].los;
        let los_mask = elevation_at(&satellite, los_at_window_start, station);
        assert!(
            elevation_at(
                &satellite,
                los_at_window_start.add_microseconds(-MICROSECONDS_PER_SECOND_I64),
                station,
            ) > los_mask
        );
        assert!(
            elevation_at(
                &satellite,
                los_at_window_start.add_microseconds(MICROSECONDS_PER_SECOND_I64),
                station,
            ) < los_mask
        );

        let los_search = PassSearch {
            satellite: &satellite,
            ground_station: station,
            start_time: los_at_window_start,
            end_time: end,
            step_us: search_step_us,
            tol_us: 1_000,
            mask: los_mask,
        };
        let next_aos_offset_seconds = reference[1].aos.diff_microseconds(los_at_window_start)
            as f64
            / MICROSECONDS_PER_SECOND_I64 as f64;
        let next_los_offset_seconds = reference[1].los.diff_microseconds(los_at_window_start)
            as f64
            / MICROSECONDS_PER_SECOND_I64 as f64;
        let los_found = assemble_passes_from_crossings(
            los_search,
            vec![
                CrossingEvent {
                    time_seconds: next_aos_offset_seconds,
                    value: 0.0,
                    threshold: 0.0,
                    direction: CrossingDirection::Rising,
                },
                CrossingEvent {
                    time_seconds: next_los_offset_seconds,
                    value: 0.0,
                    threshold: 0.0,
                    direction: CrossingDirection::Falling,
                },
            ],
        )
        .expect("exact LOS assembly succeeds");

        assert_eq!(los_found.len(), 1);
        assert!(los_found[0].aos > los_at_window_start);
        assert!(los_found[0].aos < los_found[0].los);
    }

    #[test]
    fn find_passes_does_not_seed_at_exact_los_window_start() {
        let start = UtcInstant::from_utc(2024, 12, 19, 0, 0, 0, 0).unwrap();
        let end = UtcInstant::from_utc(2024, 12, 20, 0, 0, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let elements = iss_2024_12_19_elements();
        let satellite = Satellite::from_elements_with_opsmode(&elements, OpsMode::Afspc).unwrap();
        let options = PassFinderOptions {
            elevation_mask_deg: 0.0,
            coarse_step_seconds: 10.0,
            time_tolerance_seconds: 1.0e-3,
        };

        let reference =
            find_passes(&elements, station, start, end, options).expect("valid pass-finder step");
        assert!(reference.len() >= 2);
        let los_at_window_start = reference[0].los;
        let mask = elevation_at(&satellite, los_at_window_start, station);
        assert!(
            elevation_at(
                &satellite,
                los_at_window_start.add_microseconds(-MICROSECONDS_PER_SECOND_I64),
                station,
            ) > mask
        );
        assert!(
            elevation_at(
                &satellite,
                los_at_window_start.add_microseconds(MICROSECONDS_PER_SECOND_I64),
                station,
            ) < mask
        );

        let found = find_passes(
            &elements,
            station,
            los_at_window_start,
            end,
            PassFinderOptions {
                elevation_mask_deg: mask,
                ..options
            },
        )
        .expect("valid pass-finder step");

        assert!(
            !found.is_empty(),
            "later pass after a real rising crossing should still be found"
        );
        assert!(
            found.iter().all(|pass| pass.aos > los_at_window_start),
            "window-start LOS must not be seeded as a phantom AOS"
        );
        assert!(found.iter().all(|pass| pass.aos < pass.los));
    }

    #[test]
    fn find_passes_substeps_coarse_scan_for_short_masked_pass() {
        let start = UtcInstant::from_utc(2024, 12, 19, 0, 0, 0, 0).unwrap();
        let end = UtcInstant::from_utc(2024, 12, 19, 12, 0, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let elements = iss_2024_12_19_elements();
        let satellite = Satellite::from_elements_with_opsmode(&elements, OpsMode::Afspc).unwrap();
        let mask = 12.0;
        let span_seconds = end.diff_microseconds(start) as f64 / MICROSECONDS_PER_SECOND_I64 as f64;

        let naive_crossings = EventFinder::new(0.0, span_seconds, 900.0, 1.0e-3)
            .expect("valid event-finder window")
            .find_crossings(
                |offset_seconds| {
                    elevation_at(
                        &satellite,
                        instant_at_offset_seconds(start, offset_seconds),
                        station,
                    ) - mask
                },
                0.0,
            )
            .expect("finite masked elevation predicate");
        assert!(
            naive_crossings.is_empty(),
            "900-second naive scan should skip this short masked pass"
        );

        let reference = find_passes(
            &elements,
            station,
            start,
            end,
            PassFinderOptions {
                elevation_mask_deg: mask,
                coarse_step_seconds: 10.0,
                time_tolerance_seconds: 1.0e-3,
            },
        )
        .expect("valid pass-finder step");
        let robust = find_passes(
            &elements,
            station,
            start,
            end,
            PassFinderOptions {
                elevation_mask_deg: mask,
                coarse_step_seconds: 900.0,
                time_tolerance_seconds: 1.0e-3,
            },
        )
        .expect("valid pass-finder step");

        assert_eq!(reference.len(), 1);
        assert_eq!(robust.len(), reference.len());
        assert_pass_close(&robust[0], &reference[0], 1_000, 1.0e-7);
        assert!(robust[0].max_elevation_deg > mask);
    }

    #[test]
    fn find_passes_applies_elevation_mask() {
        let start = UtcInstant::from_utc(2024, 12, 19, 0, 0, 0, 0).unwrap();
        let end = UtcInstant::from_utc(2024, 12, 20, 0, 0, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let elements = iss_2024_12_19_elements();

        let opts = |mask: f64| PassFinderOptions {
            elevation_mask_deg: mask,
            coarse_step_seconds: 10.0,
            time_tolerance_seconds: 1.0e-3,
        };

        let all =
            find_passes(&elements, station, start, end, opts(0.0)).expect("valid pass-finder step");
        let high = find_passes(&elements, station, start, end, opts(40.0))
            .expect("valid pass-finder step");

        // A higher mask admits no more passes than a lower one, and every pass
        // it keeps culminates above the mask.
        assert!(high.len() <= all.len());
        for pass in &high {
            assert!(pass.max_elevation_deg >= 40.0);
            // AOS precedes culmination precedes LOS.
            assert!(pass.aos.unix_microseconds() <= pass.culmination.unix_microseconds());
            assert!(pass.culmination.unix_microseconds() <= pass.los.unix_microseconds());
        }
    }

    #[test]
    fn pass_and_visibility_apis_reject_invalid_station_and_thresholds() {
        let start = UtcInstant::from_utc(2024, 12, 19, 0, 0, 0, 0).unwrap();
        let end = UtcInstant::from_utc(2024, 12, 19, 1, 0, 0, 0).unwrap();
        let datetime = UtcInstant::from_utc(2024, 1, 1, 12, 0, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let invalid_station = GroundStation {
            latitude_deg: 91.0,
            ..station
        };
        let elements = iss_2024_12_19_elements();

        assert_invalid_pass_field_with_reason(
            predict_passes(
                &elements,
                invalid_station,
                start,
                end,
                PassPredictionOptions::default(),
            )
            .unwrap_err(),
            "ground_station.latitude_deg",
            "out of range",
        );
        assert_invalid_pass_field_with_reason(
            predict_passes(
                &elements,
                station,
                start,
                end,
                PassPredictionOptions {
                    min_elevation_deg: f64::INFINITY,
                    step_seconds: 60,
                },
            )
            .unwrap_err(),
            "min_elevation_deg",
            "not finite",
        );
        assert_invalid_pass_field_with_reason(
            find_passes(
                &elements,
                station,
                start,
                end,
                PassFinderOptions {
                    elevation_mask_deg: 91.0,
                    coarse_step_seconds: 10.0,
                    time_tolerance_seconds: 1.0e-3,
                },
            )
            .unwrap_err(),
            "elevation_mask_deg",
            "out of range",
        );
        assert_invalid_pass_field_with_reason(
            visible_from_constellation(&[], station, datetime, f64::NAN).unwrap_err(),
            "min_elevation_deg",
            "not finite",
        );
    }

    #[test]
    fn pass_planners_reject_zero_steps() {
        let start = UtcInstant::from_utc(2024, 12, 19, 0, 0, 0, 0).unwrap();
        let end = UtcInstant::from_utc(2024, 12, 19, 1, 0, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let elements = iss_2024_12_19_elements();

        assert_invalid_pass_field(
            predict_passes(
                &elements,
                station,
                start,
                end,
                PassPredictionOptions {
                    min_elevation_deg: 0.0,
                    step_seconds: 0,
                },
            )
            .unwrap_err(),
            "step_seconds",
        );
        assert_invalid_pass_field(
            find_passes(
                &elements,
                station,
                start,
                end,
                PassFinderOptions {
                    elevation_mask_deg: 0.0,
                    coarse_step_seconds: 0.0,
                    time_tolerance_seconds: 1.0e-3,
                },
            )
            .unwrap_err(),
            "coarse_step_seconds",
        );
        assert_invalid_pass_field(
            find_passes(
                &elements,
                station,
                start,
                end,
                PassFinderOptions {
                    elevation_mask_deg: 0.0,
                    coarse_step_seconds: 10.0,
                    time_tolerance_seconds: 0.0,
                },
            )
            .unwrap_err(),
            "time_tolerance_seconds",
        );
    }

    fn assert_pass_close(
        actual: &SatellitePass,
        expected: &SatellitePass,
        time_tolerance_us: i64,
        elevation_tolerance_deg: f64,
    ) {
        assert!(
            (actual.aos.unix_microseconds() - expected.aos.unix_microseconds()).abs()
                <= time_tolerance_us,
            "AOS differs by more than {time_tolerance_us} us"
        );
        assert!(
            (actual.los.unix_microseconds() - expected.los.unix_microseconds()).abs()
                <= time_tolerance_us,
            "LOS differs by more than {time_tolerance_us} us"
        );
        assert!(
            (actual.culmination.unix_microseconds() - expected.culmination.unix_microseconds())
                .abs()
                <= time_tolerance_us,
            "culmination differs by more than {time_tolerance_us} us"
        );
        assert!(
            (actual.max_elevation_deg - expected.max_elevation_deg).abs()
                <= elevation_tolerance_deg,
            "max elevation differs by more than {elevation_tolerance_deg} deg"
        );
    }

    #[test]
    fn pass_finder_rejects_overflowing_coarse_step() {
        let start = UtcInstant::from_utc(2024, 12, 19, 0, 0, 0, 0).unwrap();
        let end = UtcInstant::from_utc(2024, 12, 19, 1, 0, 0, 0).unwrap();
        let station = GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        };
        let elements = iss_2024_12_19_elements();

        assert_invalid_pass_field_with_reason(
            find_passes(
                &elements,
                station,
                start,
                end,
                PassFinderOptions {
                    elevation_mask_deg: 0.0,
                    coarse_step_seconds: 1.0e20,
                    time_tolerance_seconds: 1.0e-3,
                },
            )
            .unwrap_err(),
            "coarse_step_seconds",
            "out of range",
        );
    }

    fn assert_invalid_pass_field(error: PassError, expected: &'static str) {
        assert_invalid_pass_field_with_reason(error, expected, "not positive");
    }

    fn assert_invalid_look_angle_field(
        error: LookAngleError,
        expected: &'static str,
        expected_reason: &'static str,
    ) {
        match error {
            LookAngleError::InvalidInput { field, reason } => {
                assert_eq!(field, expected);
                assert_eq!(reason, expected_reason);
            }
            other => panic!("expected invalid look-angle input for {expected}, got {other:?}"),
        }
    }

    fn assert_invalid_pass_field_with_reason(
        error: PassError,
        expected: &'static str,
        expected_reason: &'static str,
    ) {
        let PassError::InvalidInput { field, reason } = error;
        assert_eq!(field, expected);
        assert_eq!(reason, expected_reason);
    }
}
