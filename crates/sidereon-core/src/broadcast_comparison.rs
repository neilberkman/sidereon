//! Broadcast-ephemeris accuracy: compare a broadcast navigation product against
//! a precise SP3 product over a window (the orbit/clock pieces of the
//! signal-in-space range error, SISRE).
//!
//! For each satellite at each epoch this differences the broadcast-evaluated ECEF
//! position and clock against the precise SP3 values, decomposes the position
//! error into radial / along-track / cross-track (RAC) components built from the
//! precise state, and summarizes the differences as RMS and maximum statistics per
//! satellite and overall. Only epochs where **both** sources return a valid state
//! contribute; an epoch missing from either product is skipped, never
//! extrapolated.
//!
//! The caller supplies the per-epoch evaluation keys ([`EpochInputs`]): the
//! continuous J2000 second the broadcast product is evaluated at, and the SP3
//! split Julian dates for the epoch and its `+/-` velocity-finite-difference
//! neighbours. The SP3 velocity is a centered finite difference of the precise
//! position (one-sided at a window edge), since SP3 interpolation exposes position
//! only. Time-scale and calendar handling stay at the (Elixir) interface boundary;
//! this module owns the evaluation orchestration and all of the difference algebra.

use crate::astro::math::vec3::{cross3, dot3, norm3, scale3, sub3, unit3};
use crate::astro::time::model::{Instant, JulianDateSplit, TimeScale};
use crate::constants::C_M_S;
use crate::ephemeris::{BroadcastEphemeris, Sp3};
use crate::error::{Error, Result};
use crate::observables::ObservableEphemerisSource;
use crate::GnssSatelliteId;

/// The per-epoch evaluation keys for one sample, marshaled by the interface.
///
/// `broadcast_t_j2000_s` is the continuous second-of-J2000 the broadcast product
/// is queried at; the three split Julian dates query the precise product at the
/// epoch and at the velocity finite-difference neighbours (`epoch +/- half`),
/// each in the precise product's own header time scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochInputs {
    /// Broadcast query instant, continuous seconds since J2000 (GPST-aligned).
    pub broadcast_t_j2000_s: f64,
    /// Precise query at the epoch (split Julian date).
    pub precise: JulianDateSplit,
    /// Precise query at `epoch + half` for the centered velocity difference.
    pub precise_plus: JulianDateSplit,
    /// Precise query at `epoch - half` for the centered velocity difference.
    pub precise_minus: JulianDateSplit,
}

/// Orbit and clock difference statistics for one satellite (or the overall set).
///
/// All values are meters except `count` (the number of compared epochs). The
/// float fields are `None` when no compared epoch populated them (an empty set,
/// or no clocked epoch). `orbit_3d_*` are the Euclidean position-difference
/// magnitudes; `radial_*`/`along_*`/`cross_*` summarize the signed RAC components
/// of the position difference (`broadcast - precise`). `clock_*` are the raw
/// satellite-clock differences scaled to meters; `clock_datum_removed_*` are the
/// same after the per-epoch common reference-clock offset (the median over all
/// satellites at the epoch) is removed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompareStats {
    /// Number of compared epochs contributing to the statistics.
    pub count: usize,
    /// RMS of the 3D position-difference magnitude, meters.
    pub orbit_3d_rms_m: Option<f64>,
    /// Maximum 3D position-difference magnitude, meters.
    pub orbit_3d_max_m: Option<f64>,
    /// RMS of the radial position-difference component, meters.
    pub radial_rms_m: Option<f64>,
    /// Maximum absolute radial component, meters.
    pub radial_max_m: Option<f64>,
    /// RMS of the along-track position-difference component, meters.
    pub along_rms_m: Option<f64>,
    /// Maximum absolute along-track component, meters.
    pub along_max_m: Option<f64>,
    /// RMS of the cross-track position-difference component, meters.
    pub cross_rms_m: Option<f64>,
    /// Maximum absolute cross-track component, meters.
    pub cross_max_m: Option<f64>,
    /// RMS of the raw satellite-clock difference, meters.
    pub clock_rms_m: Option<f64>,
    /// Maximum absolute raw satellite-clock difference, meters.
    pub clock_max_m: Option<f64>,
    /// RMS of the datum-removed (true SIS) clock difference, meters.
    pub clock_datum_removed_rms_m: Option<f64>,
    /// Maximum absolute datum-removed clock difference, meters.
    pub clock_datum_removed_max_m: Option<f64>,
}

impl CompareStats {
    /// The statistics for an empty set of compared epochs.
    fn empty() -> Self {
        Self {
            count: 0,
            orbit_3d_rms_m: None,
            orbit_3d_max_m: None,
            radial_rms_m: None,
            radial_max_m: None,
            along_rms_m: None,
            along_max_m: None,
            cross_rms_m: None,
            cross_max_m: None,
            clock_rms_m: None,
            clock_max_m: None,
            clock_datum_removed_rms_m: None,
            clock_datum_removed_max_m: None,
        }
    }
}

/// The result of a broadcast-vs-precise comparison.
///
/// `per_satellite` pairs each satellite with its statistics (in the input
/// satellite order); `overall` aggregates every compared epoch across all
/// satellites; `missing` lists `(satellite, count)` for satellites with skipped
/// epochs, sorted by satellite token then count.
#[derive(Debug, Clone, PartialEq)]
pub struct CompareReport {
    /// Per-satellite statistics, in the input satellite order.
    pub per_satellite: Vec<(GnssSatelliteId, CompareStats)>,
    /// Statistics over every compared epoch across all satellites.
    pub overall: CompareStats,
    /// Satellites with one or more skipped epochs and their skip counts.
    pub missing: Vec<(GnssSatelliteId, usize)>,
}

/// One compared satellite-epoch difference. `clock_residual_m` is filled by the
/// datum-removal pass before aggregation.
#[derive(Debug, Clone, Copy)]
struct Diff {
    epoch_index: usize,
    orbit_3d: f64,
    radial: f64,
    along: f64,
    cross: f64,
    clock_m: Option<f64>,
    clock_residual_m: Option<f64>,
}

/// One satellite's compared differences plus the count of skipped epochs.
struct SatelliteDiffs {
    satellite: GnssSatelliteId,
    diffs: Vec<Diff>,
    missing: usize,
}

/// Compare a broadcast product against a precise SP3 product over `epochs`.
///
/// `velocity_half_s` is the finite-difference half-step in seconds (the caller's
/// `round(step_s / 2)`), used both to form the neighbour queries and to scale the
/// difference into a velocity.
pub fn compare(
    broadcast: &BroadcastEphemeris,
    precise: &Sp3,
    satellites: &[GnssSatelliteId],
    epochs: &[EpochInputs],
    velocity_half_s: f64,
) -> Result<CompareReport> {
    validate_compare_inputs(epochs, velocity_half_s)?;

    let scale = precise.header.time_scale;

    // Per-satellite compared diffs (in epoch order) plus the skipped-epoch count.
    let mut per_sat: Vec<SatelliteDiffs> = Vec::with_capacity(satellites.len());
    for &sat in satellites {
        let mut diffs = Vec::new();
        let mut missing = 0usize;
        for (idx, ep) in epochs.iter().enumerate() {
            match diff_at(broadcast, precise, scale, sat, idx, ep, velocity_half_s) {
                Some(diff) => diffs.push(diff),
                None => missing += 1,
            }
        }
        per_sat.push(SatelliteDiffs {
            satellite: sat,
            diffs,
            missing,
        });
    }

    // Flatten in satellite-major, epoch order (the order the statistics fold in).
    let mut all: Vec<Diff> = Vec::new();
    for sat in &per_sat {
        all.extend(sat.diffs.iter().copied());
    }

    // The raw clock difference carries a per-epoch common reference-clock offset
    // between the two products' datums; estimate it (median over all satellites
    // at the epoch) and remove it so each diff also carries the true clock error.
    let datum = clock_datum_by_epoch(&all, epochs.len());

    let overall = aggregate(&enrich(&all, &datum));
    let per_satellite = per_sat
        .iter()
        .map(|sat| (sat.satellite, aggregate(&enrich(&sat.diffs, &datum))))
        .collect();

    let mut missing: Vec<(GnssSatelliteId, usize)> = per_sat
        .iter()
        .filter(|sat| sat.missing > 0)
        .map(|sat| (sat.satellite, sat.missing))
        .collect();
    missing.sort_by(|a, b| a.0.to_string().cmp(&b.0.to_string()).then(a.1.cmp(&b.1)));

    Ok(CompareReport {
        per_satellite,
        overall,
        missing,
    })
}

/// Seconds in one Julian day, the scale between a window's second-of-J2000 axis
/// and the split-Julian-date day fraction.
const SECONDS_PER_DAY: f64 = 86_400.0;

/// A regularly sampled comparison window, the window-form input to
/// [`compare_window`].
///
/// It mirrors how the geometry series functions take a window (an inclusive
/// `(t0, t1)` second-of-J2000 span plus a step), extended with the second time
/// axis this comparison needs: the broadcast product is queried on the
/// continuous J2000 second axis (`broadcast_window_j2000_s`), while the precise
/// product is queried by split Julian date in its own header time scale. The
/// caller supplies the precise split Julian date for the window start
/// (`precise_start`); the driver advances it in lockstep with the broadcast
/// axis, so the per-epoch broadcast-to-precise time-scale offset stays fixed at
/// the value baked into the two anchors. Time-scale and calendar handling thus
/// stay at the interface boundary (the caller picks the two start anchors); the
/// driver only builds the regular per-epoch grid and the velocity neighbours.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompareWindow {
    /// Inclusive broadcast query window, continuous seconds since J2000
    /// (`(t0, t1)`); `t0` is the instant of the first epoch.
    pub broadcast_window_j2000_s: (f64, f64),
    /// Precise query for the window start `t0`, split Julian date in the precise
    /// product's header time scale.
    pub precise_start: JulianDateSplit,
    /// Sampling step between consecutive epochs, seconds.
    pub step_s: f64,
    /// Velocity finite-difference half step, seconds (the precise `+/- half`
    /// neighbour queries); also the velocity scale passed to [`compare`].
    pub velocity_half_s: f64,
}

/// Build the per-epoch evaluation keys for a [`CompareWindow`] without running the
/// comparison.
///
/// Exposed so a caller can inspect or cache the grid; [`compare_window`] builds
/// the same grid and delegates to [`compare`]. The sampling matches the geometry
/// series convention: epochs land at `t0, t0 + step, ...` up to and including the
/// window end `t1`, with a final sample snapped to `t1` when the last stepped
/// sample falls short. An empty grid is returned when `t0 > t1`.
pub fn compare_window_epochs(window: &CompareWindow) -> Result<Vec<EpochInputs>> {
    validate_window(window)?;

    let (t0, _t1) = window.broadcast_window_j2000_s;
    let times = sample_times(window.broadcast_window_j2000_s, window.step_s);
    let mut epochs = Vec::with_capacity(times.len());
    for t in times {
        let dt = t - t0;
        epochs.push(EpochInputs {
            broadcast_t_j2000_s: t,
            precise: advance_split(window.precise_start, dt),
            precise_plus: advance_split(window.precise_start, dt + window.velocity_half_s),
            precise_minus: advance_split(window.precise_start, dt - window.velocity_half_s),
        });
    }
    Ok(epochs)
}

/// Compare a broadcast product against a precise SP3 product over a sampled
/// window.
///
/// Builds the per-epoch grid from `window` (see [`compare_window_epochs`]) and
/// delegates to [`compare`] with the window's `velocity_half_s`. This is the
/// window-form sibling of [`compare`] for callers that hold a regular sampling
/// window rather than precomputed per-epoch keys.
pub fn compare_window(
    broadcast: &BroadcastEphemeris,
    precise: &Sp3,
    satellites: &[GnssSatelliteId],
    window: &CompareWindow,
) -> Result<CompareReport> {
    let epochs = compare_window_epochs(window)?;
    compare(
        broadcast,
        precise,
        satellites,
        &epochs,
        window.velocity_half_s,
    )
}

/// Advance a split Julian date by `delta_s` seconds, carrying whole days into the
/// integer part so the residual fraction stays within one day.
fn advance_split(start: JulianDateSplit, delta_s: f64) -> JulianDateSplit {
    let total_fraction = start.fraction + delta_s / SECONDS_PER_DAY;
    let whole_days = total_fraction.trunc();
    JulianDateSplit {
        jd_whole: start.jd_whole + whole_days,
        fraction: total_fraction - whole_days,
    }
}

/// The broadcast query instants for an inclusive `(t0, t1)` window at `step_s`,
/// mirroring the geometry series sampling: regular steps from `t0`, plus a final
/// snap to `t1` when the last step falls short. Empty when `t0 > t1`.
fn sample_times(window_j2000_s: (f64, f64), step_s: f64) -> Vec<f64> {
    let (t0, t1) = window_j2000_s;
    if t0 > t1 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut step_index = 0usize;
    loop {
        let t = t0 + step_s * step_index as f64;
        if t > t1 {
            break;
        }
        out.push(t);
        step_index += 1;
    }
    if let Some(&last) = out.last() {
        if last < t1 {
            out.push(t1);
        }
    }
    out
}

fn validate_window(window: &CompareWindow) -> Result<()> {
    let (t0, t1) = window.broadcast_window_j2000_s;
    validate_finite(t0, "window.broadcast_window_j2000_s.0")?;
    validate_finite(t1, "window.broadcast_window_j2000_s.1")?;
    validate_finite(window.step_s, "window.step_s")?;
    if window.step_s <= 0.0 {
        return Err(invalid_input("window.step_s", "not positive"));
    }
    validate_finite(window.velocity_half_s, "window.velocity_half_s")?;
    if window.velocity_half_s <= 0.0 {
        return Err(invalid_input("window.velocity_half_s", "not positive"));
    }
    validate_finite(
        window.precise_start.jd_whole,
        "window.precise_start.jd_whole",
    )?;
    validate_finite(
        window.precise_start.fraction,
        "window.precise_start.fraction",
    )?;
    Ok(())
}

fn validate_compare_inputs(epochs: &[EpochInputs], velocity_half_s: f64) -> Result<()> {
    validate_finite(velocity_half_s, "velocity_half_s")?;
    if velocity_half_s <= 0.0 {
        return Err(invalid_input("velocity_half_s", "not positive"));
    }
    for (index, epoch) in epochs.iter().enumerate() {
        validate_finite(epoch.broadcast_t_j2000_s, "epochs.broadcast_t_j2000_s")?;
        validate_split(epoch.precise, index, "precise")?;
        validate_split(epoch.precise_plus, index, "precise_plus")?;
        validate_split(epoch.precise_minus, index, "precise_minus")?;
    }
    Ok(())
}

fn validate_split(split: JulianDateSplit, _index: usize, field: &'static str) -> Result<()> {
    validate_finite(split.jd_whole, field)?;
    validate_finite(split.fraction, field)?;
    if !(-1.0..=1.0).contains(&split.fraction) {
        return Err(invalid_input(field, "fraction out of range"));
    }
    Ok(())
}

fn validate_finite(value: f64, field: &'static str) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "not finite"))
    }
}

fn invalid_input(field: &'static str, reason: &'static str) -> Error {
    Error::InvalidInput(format!("{field} {reason}"))
}

/// A single epoch's broadcast-minus-precise difference decomposed into RAC and a
/// clock difference, or `None` when either product lacks a valid position (or the
/// precise velocity finite difference is unavailable).
fn diff_at(
    broadcast: &BroadcastEphemeris,
    precise: &Sp3,
    scale: TimeScale,
    sat: GnssSatelliteId,
    epoch_index: usize,
    ep: &EpochInputs,
    velocity_half_s: f64,
) -> Option<Diff> {
    let (b_pos, b_clock) = broadcast_state(broadcast, sat, ep.broadcast_t_j2000_s)?;
    let (p_pos, p_clock) = precise_state(precise, scale, sat, ep.precise)?;
    let vel = precise_velocity(precise, scale, sat, ep, p_pos, velocity_half_s)?;

    let d = sub3(b_pos, p_pos);
    let (radial, along, cross) = project_rac(d, p_pos, vel);

    let clock_m = match (b_clock, p_clock) {
        (Some(bc), Some(pc)) => Some((bc - pc) * C_M_S),
        _ => None,
    };

    Some(Diff {
        epoch_index,
        orbit_3d: norm3(d),
        radial,
        along,
        cross,
        clock_m,
        clock_residual_m: None,
    })
}

/// Broadcast ECEF position and clock at the query second, or `None` on a miss.
fn broadcast_state(
    broadcast: &BroadcastEphemeris,
    sat: GnssSatelliteId,
    t_j2000_s: f64,
) -> Option<([f64; 3], Option<f64>)> {
    let state = broadcast.observable_state_at_j2000_s(sat, t_j2000_s).ok()?;
    Some((state.position_ecef_m, state.clock_s))
}

/// Precise ECEF position and clock at the split-Julian-date epoch, or `None`.
fn precise_state(
    precise: &Sp3,
    scale: TimeScale,
    sat: GnssSatelliteId,
    split: JulianDateSplit,
) -> Option<([f64; 3], Option<f64>)> {
    let state = precise
        .position(sat, Instant::from_julian_date(scale, split))
        .ok()?;
    Some((state.position.as_array(), state.clock_s))
}

/// Precise ECEF position only at the split-Julian-date epoch, or `None`.
fn precise_position(
    precise: &Sp3,
    scale: TimeScale,
    sat: GnssSatelliteId,
    split: JulianDateSplit,
) -> Option<[f64; 3]> {
    precise_state(precise, scale, sat, split).map(|(pos, _clock)| pos)
}

/// Centered finite-difference velocity of the precise position, falling back to a
/// one-sided difference when a neighbour epoch is outside the SP3 span. `r0` is the
/// precise position already evaluated at the epoch (reused for the one-sided case).
fn precise_velocity(
    precise: &Sp3,
    scale: TimeScale,
    sat: GnssSatelliteId,
    ep: &EpochInputs,
    r0: [f64; 3],
    half_s: f64,
) -> Option<[f64; 3]> {
    let rp = precise_position(precise, scale, sat, ep.precise_plus);
    let rm = precise_position(precise, scale, sat, ep.precise_minus);
    finite_difference_velocity(rp, rm, r0, half_s)
}

/// Combine the neighbour positions into a velocity: a centered difference when
/// both neighbours are available, a one-sided difference (using the epoch position
/// `r0`) when only one is, and `None` when neither is. Pure arithmetic, split out
/// from the evaluation so the operation order can be pinned independently.
fn finite_difference_velocity(
    rp: Option<[f64; 3]>,
    rm: Option<[f64; 3]>,
    r0: [f64; 3],
    half_s: f64,
) -> Option<[f64; 3]> {
    match (rp, rm) {
        (Some(rp), Some(rm)) => Some(scale3(sub3(rp, rm), 1.0 / (2.0 * half_s))),
        (Some(rp), None) => Some(scale3(sub3(rp, r0), 1.0 / half_s)),
        (None, Some(rm)) => Some(scale3(sub3(r0, rm), 1.0 / half_s)),
        (None, None) => None,
    }
}

/// Project a difference vector onto the radial/along-track/cross-track triad of
/// the orbit defined by position `r` and velocity `v`. Radial is along `r`,
/// cross-track along the angular momentum `r x v`, along-track completes the
/// right-handed set.
fn project_rac(d: [f64; 3], r: [f64; 3], v: [f64; 3]) -> (f64, f64, f64) {
    let radial_hat = unit3(r).unwrap_or([0.0, 0.0, 0.0]);
    let cross_hat = unit3(cross3(r, v)).unwrap_or([0.0, 0.0, 0.0]);
    let along_hat = cross3(cross_hat, radial_hat);
    (dot3(d, radial_hat), dot3(d, along_hat), dot3(d, cross_hat))
}

/// Per-epoch common reference-clock offset (meters): the median over all
/// satellites at the epoch of the raw clock difference. `None` for an epoch with
/// no clocked satellite.
fn clock_datum_by_epoch(all: &[Diff], n_epochs: usize) -> Vec<Option<f64>> {
    let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); n_epochs];
    for diff in all {
        if let Some(clock) = diff.clock_m {
            buckets[diff.epoch_index].push(clock);
        }
    }
    buckets.iter().map(|clocks| median(clocks)).collect()
}

/// Attach the datum-removed clock residual to each diff: raw clock difference
/// minus the epoch's common offset, when both are present.
fn enrich(diffs: &[Diff], datum: &[Option<f64>]) -> Vec<Diff> {
    diffs
        .iter()
        .map(|diff| {
            let clock_residual_m = match (diff.clock_m, datum[diff.epoch_index]) {
                (Some(clock), Some(d)) => Some(clock - d),
                _ => None,
            };
            Diff {
                clock_residual_m,
                ..*diff
            }
        })
        .collect()
}

/// Summarize a set of compared diffs into RMS/max statistics.
fn aggregate(diffs: &[Diff]) -> CompareStats {
    if diffs.is_empty() {
        return CompareStats::empty();
    }

    let orbit: Vec<f64> = diffs.iter().map(|d| d.orbit_3d).collect();
    let radial: Vec<f64> = diffs.iter().map(|d| d.radial).collect();
    let along: Vec<f64> = diffs.iter().map(|d| d.along).collect();
    let cross: Vec<f64> = diffs.iter().map(|d| d.cross).collect();
    let clocks: Vec<f64> = diffs.iter().filter_map(|d| d.clock_m).collect();
    let clock_resids: Vec<f64> = diffs.iter().filter_map(|d| d.clock_residual_m).collect();

    CompareStats {
        count: diffs.len(),
        orbit_3d_rms_m: Some(rms(&orbit)),
        orbit_3d_max_m: Some(max_abs(&orbit)),
        radial_rms_m: Some(rms(&radial)),
        radial_max_m: Some(max_abs(&radial)),
        along_rms_m: Some(rms(&along)),
        along_max_m: Some(max_abs(&along)),
        cross_rms_m: Some(rms(&cross)),
        cross_max_m: Some(max_abs(&cross)),
        clock_rms_m: rms_or_none(&clocks),
        clock_max_m: max_abs_or_none(&clocks),
        clock_datum_removed_rms_m: rms_or_none(&clock_resids),
        clock_datum_removed_max_m: max_abs_or_none(&clock_resids),
    }
}

/// Root-mean-square over a non-empty slice, folding left from `0.0`.
fn rms(values: &[f64]) -> f64 {
    let sum_sq = values.iter().fold(0.0, |acc, &x| acc + x * x);
    (sum_sq / values.len() as f64).sqrt()
}

fn rms_or_none(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(rms(values))
    }
}

/// Maximum absolute value over a non-empty slice.
fn max_abs(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|v| v.abs())
        .reduce(f64::max)
        .expect("max_abs requires a non-empty slice")
}

fn max_abs_or_none(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(max_abs(values))
    }
}

/// Median of a slice (`None` if empty); even counts average the two middles.
fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let mid = n / 2;
    if n % 2 == 1 {
        Some(sorted[mid])
    } else {
        Some((sorted[mid - 1] + sorted[mid]) / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(epoch_index: usize, orbit: f64, r: f64, a: f64, c: f64, clock: Option<f64>) -> Diff {
        Diff {
            epoch_index,
            orbit_3d: orbit,
            radial: r,
            along: a,
            cross: c,
            clock_m: clock,
            clock_residual_m: None,
        }
    }

    #[test]
    fn window_grid_matches_independent_construction() {
        let window = CompareWindow {
            broadcast_window_j2000_s: (1_000.0, 1_000.0 + 3.0 * 900.0),
            precise_start: JulianDateSplit::new(2_451_545.0, 0.25).expect("valid split"),
            step_s: 900.0,
            velocity_half_s: 450.0,
        };

        let grid = compare_window_epochs(&window).expect("window grid");

        // Independent reconstruction of the same per-epoch keys: epochs land at
        // t0, t0 + step, ... up to t1, with precise queries advanced from the
        // start anchor by the elapsed seconds (and +/- the velocity half step).
        let (t0, _t1) = window.broadcast_window_j2000_s;
        let advance = |seconds: f64| {
            let total = window.precise_start.fraction + seconds / 86_400.0;
            let days = total.trunc();
            JulianDateSplit {
                jd_whole: window.precise_start.jd_whole + days,
                fraction: total - days,
            }
        };
        let mut expected = Vec::new();
        for index in 0..4 {
            let dt = 900.0 * index as f64;
            expected.push(EpochInputs {
                broadcast_t_j2000_s: t0 + dt,
                precise: advance(dt),
                precise_plus: advance(dt + 450.0),
                precise_minus: advance(dt - 450.0),
            });
        }

        assert_eq!(grid, expected);
    }

    #[test]
    fn window_grid_snaps_final_sample_to_window_end() {
        // A step that does not divide the span exactly still includes t1 as a
        // final snapped epoch, mirroring the geometry series sampling.
        let window = CompareWindow {
            broadcast_window_j2000_s: (0.0, 1_000.0),
            precise_start: JulianDateSplit::new(2_451_545.0, 0.0).expect("valid split"),
            step_s: 400.0,
            velocity_half_s: 200.0,
        };
        let grid = compare_window_epochs(&window).expect("window grid");
        let times: Vec<f64> = grid.iter().map(|e| e.broadcast_t_j2000_s).collect();
        assert_eq!(times, vec![0.0, 400.0, 800.0, 1_000.0]);
    }

    #[test]
    fn window_grid_is_empty_when_start_after_end() {
        let window = CompareWindow {
            broadcast_window_j2000_s: (2_000.0, 1_000.0),
            precise_start: JulianDateSplit::new(2_451_545.0, 0.0).expect("valid split"),
            step_s: 100.0,
            velocity_half_s: 50.0,
        };
        assert!(compare_window_epochs(&window)
            .expect("empty grid")
            .is_empty());
    }

    #[test]
    fn window_rejects_non_positive_step() {
        let window = CompareWindow {
            broadcast_window_j2000_s: (0.0, 1_000.0),
            precise_start: JulianDateSplit::new(2_451_545.0, 0.0).expect("valid split"),
            step_s: 0.0,
            velocity_half_s: 50.0,
        };
        assert!(matches!(
            compare_window_epochs(&window),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn median_odd_and_even_match_reference() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&[]), None);
    }

    #[test]
    fn rms_folds_left_from_zero() {
        let values = [1.0, 2.0, 2.0];
        let expected = ((1.0_f64 + 4.0 + 4.0) / 3.0).sqrt();
        assert_eq!(rms(&values).to_bits(), expected.to_bits());
        assert_eq!(rms_or_none(&[]), None);
        assert_eq!(max_abs(&[-3.0, 2.0, -1.0]), 3.0);
    }

    // The RAC projection is pure arithmetic (no transcendental, no eval), so its
    // bits are frozen here as an operation-order regression pin.
    #[test]
    fn rac_projection_has_frozen_bits() {
        let r = [7.0e6, 1.0e6, 2.0e6];
        let v = [-500.0, 7000.0, 1000.0];
        let d = [3.0, -4.0, 5.0];
        let (radial, along, cross) = project_rac(d, r, v);

        assert_eq!(radial.to_bits(), 0x400d64d51e0db1c6);
        assert_eq!(along.to_bits(), 0xc00eed09ea935852);
        assert_eq!(cross.to_bits(), 0x40129246dff98f29);

        // Orthonormal rotation preserves the norm.
        let quad = (radial * radial + along * along + cross * cross).sqrt();
        assert!((quad - norm3(d)).abs() < 1.0e-9);
    }

    #[test]
    fn finite_difference_velocity_has_frozen_bits() {
        let rp = [1.0e3, 2.0e3, 3.0e3];
        let rm = [-1.0e3, 1.0e3, 2.5e3];
        let r0 = [100.0, 1600.0, 2700.0];
        let half = 450.0;

        let centered = finite_difference_velocity(Some(rp), Some(rm), r0, half).unwrap();
        assert_eq!(centered[0].to_bits(), 0x4001c71c71c71c72);
        assert_eq!(centered[1].to_bits(), 0x3ff1c71c71c71c72);
        assert_eq!(centered[2].to_bits(), 0x3fe1c71c71c71c72);

        let one_sided = finite_difference_velocity(Some(rp), None, r0, half).unwrap();
        assert_eq!(one_sided[0].to_bits(), 0x4000000000000000);
        assert_eq!(one_sided[1].to_bits(), 0x3fec71c71c71c71c);
        assert_eq!(one_sided[2].to_bits(), 0x3fe5555555555555);

        assert_eq!(finite_difference_velocity(None, None, r0, half), None);
    }

    // The full aggregation (satellite-major RMS fold, even/odd median, per-epoch
    // clock-datum removal across satellites) frozen as a 0-ULP regression. Pure
    // arithmetic on synthetic per-cell differences, independent of the ephemeris
    // evaluation, so it is bit-stable across optimization levels.
    #[test]
    fn aggregation_and_datum_have_frozen_bits() {
        let g01 = [
            cell(0, 1.0, 0.6, 0.8, 0.0, Some(10.0)),
            cell(1, 2.0, 1.2, 1.6, 0.0, Some(12.0)),
            cell(2, 3.0, 1.8, 2.4, 0.0, None),
        ];
        let g02 = [
            cell(0, 4.0, 2.4, 3.2, 0.0, Some(20.0)),
            cell(1, 5.0, 3.0, 4.0, 0.0, Some(8.0)),
        ];

        let mut all = Vec::new();
        all.extend(g01.iter().copied());
        all.extend(g02.iter().copied());
        let datum = clock_datum_by_epoch(&all, 3);
        // Even-median datum per epoch: e0 = median(10,20) = 15, e1 = median(12,8) = 10.
        assert_eq!(datum, vec![Some(15.0), Some(10.0), None]);

        let overall = aggregate(&enrich(&all, &datum));
        assert_eq!(overall.count, 5);
        assert_eq!(
            overall.orbit_3d_rms_m.unwrap().to_bits(),
            0x400a887293fd6f34
        );
        assert_eq!(
            overall.orbit_3d_max_m.unwrap().to_bits(),
            0x4014000000000000
        );
        assert_eq!(overall.clock_rms_m.unwrap().to_bits(), 0x402a9bb78af6cabc);
        assert_eq!(
            overall.clock_datum_removed_rms_m.unwrap().to_bits(),
            0x400e768d399dc470
        );
        assert_eq!(
            overall.clock_datum_removed_max_m.unwrap().to_bits(),
            0x4014000000000000
        );

        let g01_stats = aggregate(&enrich(&g01, &datum));
        assert_eq!(g01_stats.count, 3);
        assert_eq!(
            g01_stats.orbit_3d_rms_m.unwrap().to_bits(),
            0x4001482f86c40c43
        );
        // Only two of G01's cells carry a clock; the third is dropped from clock stats.
        assert_eq!(g01_stats.clock_rms_m.unwrap().to_bits(), 0x402617398f2aaa48);
        assert_eq!(
            g01_stats.clock_datum_removed_rms_m.unwrap().to_bits(),
            0x400e768d399dc470
        );
    }
}
