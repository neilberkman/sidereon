//! SP3 arbitrary-epoch position/clock interpolation.
//!
//! For a consumer-facing summary of the interpolation contract, a utility to
//! verify sidereon's interpolation against a supplied reference series
//! ([`compare_position_series`](crate::sp3::compare_position_series)), and the
//! recommendation on configurable interpolation modes, see that function's
//! module documentation.
//!
//! Two channels with different recipes, each validated against its correct
//! external reference (the two-bars doctrine: capability vs the deployed
//! reference, not a bit-exact port of a convenient primitive):
//!
//! # Position channel: sliding-window Lagrange/Neville (RTKLIB recipe)
//!
//! The satellite position is interpolated with a sliding-window high-degree
//! Lagrange (Neville) polynomial matching RTKLIB `preceph.c` pephpos/interppol:
//! the contiguous run of nodes bracketing the query, the RTKLIB window of up to
//! 11 nodes (degree 10) centred on the query, an `OMEGA_E_DOT` per-node
//! earth-rotation correction into the query-epoch frame, then Neville evaluation
//! per axis. This is the IGS-standard orbit interpolation. It replaced a global
//! not-a-knot cubic spline, which is only degree 3 over the whole day and erred
//! ~200 m at the day boundary and across coverage gaps (a query deep inside a
//! coverage gap is now rejected, never interpolated across). Validated against
//! the RTKLIB reference (`interp_tests`) and end-to-end against ZIM2 PPP truth.
//!
//! # Clock channel: not-a-knot cubic spline (gnssanalysis recipe)
//!
//! The clock is locally smooth, so it keeps the not-a-knot cubic spline matching
//! `scipy.interpolate.CubicSpline(x, y)` with gnssanalysis defaults
//! (`bc_type="not-a-knot"`, `extrapolate=true`), evaluated at the query, with
//! clock-event (`E`) arc splitting. BLAS-free (the not-a-knot solve dispatches
//! to LAPACK `dgtsv`), a legitimate 0-ULP target against scipy.
//!
//! # Node substrate (load-bearing for 0-ULP)
//!
//! Nodes are **integer seconds since J2000** (2000-01-01 12:00:00 in the file's
//! own time scale), exactly as gnssanalysis builds them in `datetime2j2000`
//! (`gn_datetime.py:286-288`): epochs quantized to whole seconds, differenced
//! against the J2000 origin, kept as `i64`, then promoted to `f64` on entry to
//! the spline. The parser stores that seconds axis from the epoch record's civil
//! fields with integer whole-second arithmetic, so node bucketing does not
//! depend on a split-Julian floating round trip. If a caller mutates the public
//! [`Sp3::epochs`] vector, interpolation falls back to the public epoch when it
//! no longer matches the parsed axis within the shared whole-second tolerance.
//!
//! J2000 = JD 2451545.0. Query seconds from [`Sp3::position`] are computed from
//! the supplied [`Instant`] in a cancellation-safe way and are never quantized.
//!
//! # Units
//!
//! The spline is fit in the SP3-native units the reference carries -
//! **kilometers** for position, **microseconds** for clock - and the evaluated
//! result is converted to the public API boundary (**meters**, **seconds**) by a
//! **single final multiply** (`* 1000.0`, `* 1e-6`). The conversion happens
//! AFTER evaluation, never before the fit; this operation order is pinned.
//!
//! # Clock interpolation near gaps / discontinuities
//!
//! gnssanalysis defines none, so the policy is authored in the canonical recipe
//! and matched here:
//!
//! - Clock uses the **same** `CubicSpline` construction over the nodes that have
//!   a clock estimate (the bad-clock sentinel yields no clock node).
//! - Position and clock node sets are independent.
//! - Position is never split (orbits are continuous through clock resets).
//! - Clock interpolation does **not** cross a clock-event (`E`) epoch: the arc
//!   is split at each `E`-flagged epoch and the clock spline is fit on the
//!   contiguous sub-arc containing the query epoch.

use crate::astro::time::model::{Instant, InstantRepr, JulianDateSplit};

use crate::astro::time::civil::j2000_seconds_from_split;
use crate::constants::{J2000_JD, KM_TO_M, OMEGA_E_DOT_RAD_S, SECONDS_PER_DAY, US_TO_S};
use crate::frame::ItrfPositionM;
use crate::id::GnssSatelliteId;
use crate::sp3::{Sp3, Sp3State};
use crate::tolerances::WHOLE_SECOND_EPS_S;
use crate::validate;
use crate::{Error, Result};

/// Per-satellite precise node series in native fit units.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PreciseSatSeries {
    /// Whole-second J2000 node axis.
    pub(super) x: Vec<f64>,
    /// X position nodes in SP3-native kilometers.
    pub(super) kx: Vec<f64>,
    /// Y position nodes in SP3-native kilometers.
    pub(super) ky: Vec<f64>,
    /// Z position nodes in SP3-native kilometers.
    pub(super) kz: Vec<f64>,
    /// Clock nodes as `(x_seconds, clock_us, clock_event)`.
    pub(super) clk: Vec<(f64, f64, bool)>,
}

impl PreciseSatSeries {
    pub(super) fn new() -> Self {
        Self {
            x: Vec::new(),
            kx: Vec::new(),
            ky: Vec::new(),
            kz: Vec::new(),
            clk: Vec::new(),
        }
    }
}

/// One clock sub-arc with precomputed not-a-knot cubic coefficients.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ClockSplineArc {
    /// Clock node axis for this sub-arc.
    pub(super) x: Vec<f64>,
    /// Cubic coefficient for `(query - x[i])^3`, in native microseconds.
    pub(super) c0: Vec<f64>,
    /// Cubic coefficient for `(query - x[i])^2`, in native microseconds.
    pub(super) c1: Vec<f64>,
    /// Cubic coefficient for `(query - x[i])`, in native microseconds.
    pub(super) c2: Vec<f64>,
    /// Constant coefficient, in native microseconds.
    pub(super) c3: Vec<f64>,
}

impl ClockSplineArc {
    fn from_nodes(nodes: &[(f64, f64, bool)]) -> Self {
        let x: Vec<_> = nodes.iter().map(|node| node.0).collect();
        if nodes.len() < 2 {
            return Self {
                x,
                c0: Vec::new(),
                c1: Vec::new(),
                c2: Vec::new(),
                c3: Vec::new(),
            };
        }

        let y: Vec<_> = nodes.iter().map(|node| node.1).collect();
        let dydx = solve_not_a_knot_slopes(&x, &y);
        let (c0, c1, c2, c3) = hermite_segment_coeffs(&x, &y, &dydx);
        Self { x, c0, c1, c2, c3 }
    }
}

impl Sp3 {
    /// The product's parsed epochs as seconds since J2000, in the file's own time
    /// scale, ascending.
    ///
    /// This is the exact query axis [`Sp3::position_at_j2000_seconds`] uses for
    /// parsed record nodes, so a caller can read the grid here, form query times
    /// on it, and feed them straight back without a Julian-date round trip. An
    /// unmodified parsed product returns one value per epoch.
    pub fn epochs_j2000_seconds(&self) -> Vec<f64> {
        self.epochs
            .iter()
            .enumerate()
            .filter_map(|(idx, epoch)| sp3_epoch_j2000_seconds(self, idx, epoch))
            .collect()
    }

    /// Interpolate the state of `sat` at an arbitrary `epoch`.
    ///
    /// Reproduces the pinned `scipy.interpolate.CubicSpline` recipe (see module
    /// docs) bit-for-bit: a per-axis not-a-knot cubic spline over the
    /// J2000-integer-second node axis, evaluated at `epoch`, with the unit
    /// conversion as a single final multiply.
    ///
    /// - `position` is always returned (interpolated from all position nodes of
    ///   `sat`), in meters, ITRF/IGS ECEF.
    /// - `clock_s` is `Some` when `sat` has at least two clock nodes in the
    ///   clock sub-arc containing `epoch` (after clock-event splitting); `None`
    ///   otherwise.
    /// - `velocity` / `clock_rate_s_s` are `None` (this API interpolates the
    ///   position/clock product; velocity products are a separate concern).
    /// - `flags` are defaulted (an interpolated state is synthetic, not a record).
    ///
    /// Errors:
    /// - [`Error::UnknownSatellite`] if `sat` has no position nodes.
    /// - [`Error::EpochOutOfRange`] if fewer than two position nodes exist (a
    ///   spline needs at least two points) or the epoch is not representable.
    /// - [`Error::InvalidInput`] if `epoch` is tagged with a different time
    ///   scale than the SP3 product.
    pub fn position(&self, sat: GnssSatelliteId, epoch: Instant) -> Result<Sp3State> {
        if epoch.scale != self.header.time_scale {
            return Err(Error::InvalidInput(format!(
                "SP3 query time scale {} does not match product time scale {}",
                epoch.scale.abbrev(),
                self.header.time_scale.abbrev()
            )));
        }
        let query = instant_to_j2000_seconds(&epoch).ok_or(Error::EpochOutOfRange)?;
        self.position_at_j2000_seconds(sat, query)
    }

    /// Interpolate the state of `sat` at an arbitrary J2000-second epoch
    /// supplied directly as an `f64`.
    ///
    /// Identical to [`Sp3::position`] except the query is the seconds-since-J2000
    /// value as already computed by the caller, rather than derived from an
    /// [`Instant`]. The transmit-time iteration of the SPP residual carries the
    /// epoch as a J2000-second `f64` (`t_tx = t_rx - rho/c`) and must feed that
    /// exact value to the spline, with no Julian-date round-trip in the loop, so
    /// the interpolated position/clock match the reference recipe bit-for-bit.
    /// The real-product decimation hold-out oracle pins the parsed-text versus
    /// sample-backed 3D difference at no more than `4.020965667248365e-8` m
    /// over interior held-out 5-minute records bracketed by 15-minute nodes.
    ///
    /// Errors:
    /// - [`Error::InvalidInput`] if `query` is NaN or infinite.
    pub fn position_at_j2000_seconds(&self, sat: GnssSatelliteId, query: f64) -> Result<Sp3State> {
        let series = gather_sp3_precise_series(self, sat);
        interpolate_precise_state(
            sat,
            &series.x,
            &series.kx,
            &series.ky,
            &series.kz,
            &series.clk,
            query,
        )
    }
}

/// Gather one satellite's SP3-native node series in ascending epoch order.
pub(super) fn gather_sp3_precise_series(source: &Sp3, sat: GnssSatelliteId) -> PreciseSatSeries {
    let mut series = PreciseSatSeries::new();

    for (idx, ep) in source.epochs.iter().enumerate() {
        // Node axis: shared whole-second construction. The query is never
        // quantized.
        let xi = match sp3_epoch_j2000_seconds(source, idx, ep) {
            Some(v) => precise_node_j2000_seconds(v),
            None => continue,
        };
        // Use the parser's native km/us node values. Reconstructing km from the
        // public meters can drift by 1 ULP; unit conversion stays after eval.
        let Some(raw) = source.interp_raw[idx].get(&sat) else {
            continue;
        };
        series.x.push(xi);
        series.kx.push(raw.km[0]);
        series.ky.push(raw.km[1]);
        series.kz.push(raw.km[2]);

        if let Some(clk_us) = raw.clock_us {
            series.clk.push((xi, clk_us, raw.clock_event));
        }
    }

    series
}

pub(super) fn sp3_epoch_j2000_seconds(source: &Sp3, idx: usize, epoch: &Instant) -> Option<f64> {
    let public_seconds = instant_to_j2000_seconds(epoch)?;
    let parsed_seconds = *source.epoch_j2000_s.get(idx)?;
    if (public_seconds - parsed_seconds).abs() <= WHOLE_SECOND_EPS_S {
        Some(parsed_seconds)
    } else {
        Some(public_seconds)
    }
}

/// Convert already-reduced J2000 seconds onto the SP3 interpolation node axis.
///
/// This keeps the gnssanalysis truncation policy for genuinely fractional node
/// epochs. Callers that still have a split-Julian instant should use
/// [`precise_node_j2000_seconds_from_instant`] so whole-second record nodes can
/// be reconstructed before reducing the epoch to one large `f64`.
pub(super) fn precise_node_j2000_seconds(seconds: f64) -> f64 {
    seconds.floor()
}

/// Convert a split-aware instant onto the SP3 interpolation node axis.
///
/// SP3 record epochs are whole-second in normal products. A split-Julian sample
/// can reduce to the adjacent `f64` below an integer J2000 second when the day
/// term is large. Recover the whole-second candidate from the day term and
/// within-day second first, then accept it only when the residual is within two
/// ULPs at the candidate epoch. Genuinely fractional epochs still use the
/// truncating policy.
pub(super) fn precise_node_j2000_seconds_from_instant(instant: &Instant) -> Option<f64> {
    match instant.repr {
        InstantRepr::JulianDate(split) => Some(precise_node_j2000_seconds_from_split(split)),
        InstantRepr::Nanos(_) => instant_to_j2000_seconds(instant).map(precise_node_j2000_seconds),
    }
}

fn precise_node_j2000_seconds_from_split(split: JulianDateSplit) -> f64 {
    let seconds = j2000_seconds_from_split(split.jd_whole, split.fraction);
    let day_seconds = (split.jd_whole - J2000_JD) * SECONDS_PER_DAY;
    let within_day_seconds = split.fraction * SECONDS_PER_DAY;
    let nearest_within_day = within_day_seconds.round();
    let candidate = day_seconds + nearest_within_day;
    if candidate.is_finite()
        && (within_day_seconds - nearest_within_day).abs() <= 2.0 * epoch_ulp(candidate)
    {
        candidate
    } else {
        precise_node_j2000_seconds(seconds)
    }
}

fn epoch_ulp(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    (next_up(value) - value)
        .abs()
        .max((value - next_down(value)).abs())
}

fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == -0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value.is_sign_negative() {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

fn next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value.is_sign_negative() {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// Interpolate a satellite state from already-gathered native-unit nodes.
///
/// This is the shared interpolation substrate. Both the SP3-parsed source
/// ([`Sp3::position_at_j2000_seconds`]) and the sample-backed source
/// ([`crate::sp3::PreciseEphemerisSamples`]) gather the same node vectors
/// (ascending whole-second J2000 seconds `x`; native km `kx/ky/kz`; native
/// `(x, clock_us, clock_event)` clock nodes) and drive this one function, so a
/// source built from samples produces byte-identical states to the SP3 source
/// those samples serialize to.
///
/// Inputs are the file-native units the reference recipes consume (km for
/// position, microseconds for clock); the single final unit multiply to meters /
/// seconds happens inside the position/clock evaluators, exactly as the SP3 path
/// requires for 0-ULP parity.
pub(super) fn interpolate_precise_state(
    sat: GnssSatelliteId,
    pos_x: &[f64],
    pos_kx: &[f64],
    pos_ky: &[f64],
    pos_kz: &[f64],
    clk_nodes: &[(f64, f64, bool)],
    query: f64,
) -> Result<Sp3State> {
    let (x_m, y_m, z_m) = interpolate_precise_position(sat, pos_x, pos_kx, pos_ky, pos_kz, query)?;
    let clock_s = interpolate_clock(clk_nodes, query);

    Ok(Sp3State {
        position: ItrfPositionM::new(x_m, y_m, z_m).expect("valid ITRF position"),
        clock_s,
        velocity: None,
        clock_rate_s_s: None,
        flags: crate::sp3::Sp3Flags::default(),
    })
}

pub(super) fn interpolate_precise_state_with_clock_arcs(
    sat: GnssSatelliteId,
    pos_x: &[f64],
    pos_kx: &[f64],
    pos_ky: &[f64],
    pos_kz: &[f64],
    clock_arcs: &[ClockSplineArc],
    query: f64,
) -> Result<Sp3State> {
    let (x_m, y_m, z_m) = interpolate_precise_position(sat, pos_x, pos_kx, pos_ky, pos_kz, query)?;
    let clock_s = interpolate_fitted_clock(clock_arcs, query);

    Ok(Sp3State {
        position: ItrfPositionM::new(x_m, y_m, z_m).expect("valid ITRF position"),
        clock_s,
        velocity: None,
        clock_rate_s_s: None,
        flags: crate::sp3::Sp3Flags::default(),
    })
}

fn interpolate_precise_position(
    sat: GnssSatelliteId,
    pos_x: &[f64],
    pos_kx: &[f64],
    pos_ky: &[f64],
    pos_kz: &[f64],
    query: f64,
) -> Result<(f64, f64, f64)> {
    let query = validate::finite(query, "query_j2000_s").map_err(map_query_input)?;

    if pos_x.is_empty() {
        return Err(Error::UnknownSatellite(sat));
    }
    if pos_x.len() < 2 {
        // A cubic spline needs >= 2 points; a single node cannot define one.
        return Err(Error::EpochOutOfRange);
    }
    validate_strictly_increasing_nodes(pos_x)?;

    // Refuse grossly out-of-coverage queries instead of silently returning a
    // diverging extrapolation. The underlying cubic spline mirrors scipy
    // CubicSpline(extrapolate=True): a query well past the node span runs off
    // to nonsense (megametres and worse). We allow up to one node spacing of
    // edge extrapolation (the end cubic is still physically reasonable that
    // close to the data) and reject anything beyond. In-coverage interpolation
    // is bit-for-bit unchanged, so 0-ULP parity is preserved. Nodes are in
    // ascending epoch order.
    // Reject a query that lands deep inside an interior coverage gap rather
    // than interpolating across it. Nominal spacing is the smallest
    // consecutive node gap; a bracketing interval far larger than that is a
    // gap. One nominal spacing of interpolation past either edge node is
    // allowed (the near-gap edge stays usable); beyond that the query is in
    // the gap and is refused.
    let nominal = nominal_positive_spacing(pos_x).ok_or(Error::EpochOutOfRange)?;
    let first = pos_x[0];
    let last = pos_x[pos_x.len() - 1];
    if query < first - nominal || query > last + nominal {
        return Err(Error::EpochOutOfRange);
    }

    let gap_thresh = 1.5 * nominal;
    let mut bi = 0usize;
    while bi + 1 < pos_x.len() && pos_x[bi + 1] <= query {
        bi += 1;
    }
    if bi + 1 < pos_x.len() {
        let (lo, hi) = (pos_x[bi], pos_x[bi + 1]);
        if hi - lo > gap_thresh && query > lo + nominal && query < hi - nominal {
            return Err(Error::EpochOutOfRange);
        }
    }

    Ok(interpolate_position_neville(
        pos_x, pos_kx, pos_ky, pos_kz, query,
    ))
}

fn map_query_input(error: validate::FieldError) -> Error {
    Error::InvalidInput(format!("{} {}", error.field(), error.reason()))
}

fn nominal_positive_spacing(x: &[f64]) -> Option<f64> {
    let nominal = x
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&d| d > 0.0)
        .fold(f64::INFINITY, f64::min);
    if nominal.is_finite() {
        Some(nominal)
    } else {
        None
    }
}

fn validate_strictly_increasing_nodes(x: &[f64]) -> Result<()> {
    for window in x.windows(2) {
        if window[1] <= window[0] {
            return Err(Error::InvalidInput(
                "SP3 interpolation epochs must be strictly increasing".to_string(),
            ));
        }
    }
    Ok(())
}

/// Interpolate the clock channel with the clock-event-split policy.
///
/// Splits the clock node arc at each clock-event (`E`) epoch and fits the
/// not-a-knot spline on the contiguous sub-arc containing `query`. Returns
/// `None` if that sub-arc has fewer than two nodes.
fn interpolate_clock(clk_nodes: &[(f64, f64, bool)], query: f64) -> Option<f64> {
    let arcs = fit_clock_spline_arcs(clk_nodes);
    interpolate_fitted_clock(&arcs, query)
}

pub(super) fn fit_clock_spline_arcs(clk_nodes: &[(f64, f64, bool)]) -> Vec<ClockSplineArc> {
    if clk_nodes.is_empty() {
        return Vec::new();
    }

    // Partition into contiguous sub-arcs split at clock-event epochs. A
    // clock-event epoch marks a discontinuity *at* that epoch, so it ends the
    // sub-arc before it and starts a new one (the flagged node belongs to the
    // new sub-arc, since the reset takes effect there).
    let mut sub_start = 0usize;
    let mut arcs = Vec::new();
    for i in 0..clk_nodes.len() {
        let is_break = clk_nodes[i].2 && i > sub_start;
        if is_break {
            // Sub-arc [sub_start, i) ends here.
            arcs.push(ClockSplineArc::from_nodes(&clk_nodes[sub_start..i]));
            sub_start = i;
        }
    }
    // Trailing sub-arc [sub_start, len).
    arcs.push(ClockSplineArc::from_nodes(&clk_nodes[sub_start..]));
    arcs
}

/// Interpolate the clock channel from precomputed clock spline sub-arcs.
pub(super) fn interpolate_fitted_clock(arcs: &[ClockSplineArc], query: f64) -> Option<f64> {
    let mut chosen: Option<usize> = None;
    for (idx, arc) in arcs.iter().enumerate() {
        if arc_contains_query(arc, query) {
            chosen = Some(idx);
            break;
        }
    }
    // If the query is outside every sub-arc span (extrapolation), use the
    // sub-arc nearest the query so the default extrapolate=True behavior holds
    // within the contiguous piece on that side.
    let arc = match chosen {
        Some(idx) => &arcs[idx],
        None => nearest_fitted_subarc(arcs, query)?,
    };

    if arc.x.len() < 2 {
        return None;
    }
    Some(evaluate_ppoly(&arc.x, &arc.c0, &arc.c1, &arc.c2, &arc.c3, query) * US_TO_S)
}

/// Whether `query` lies within the closed node-span of a fitted sub-arc.
fn arc_contains_query(arc: &ClockSplineArc, query: f64) -> bool {
    if arc.x.is_empty() {
        return false;
    }
    let lo = arc.x[0];
    let hi = arc.x[arc.x.len() - 1];
    query >= lo && query <= hi
}

/// Find the sub-arc (split at clock-event epochs) whose node-span is nearest to
/// `query` for extrapolation. Returns `[start, end)` or `None` if empty.
fn nearest_fitted_subarc(arcs: &[ClockSplineArc], query: f64) -> Option<&ClockSplineArc> {
    arcs.iter()
        .filter(|arc| arc.x.len() >= 2)
        .min_by(|arc1, arc2| {
            let d1 = fitted_span_distance(arc1, query);
            let d2 = fitted_span_distance(arc2, query);
            d1.partial_cmp(&d2).unwrap_or(core::cmp::Ordering::Equal)
        })
}

fn fitted_span_distance(arc: &ClockSplineArc, query: f64) -> f64 {
    let lo = arc.x[0];
    let hi = arc.x[arc.x.len() - 1];
    if query < lo {
        lo - query
    } else if query > hi {
        query - hi
    } else {
        0.0
    }
}

/// Convert a parser [`Instant`] to seconds since J2000, as `f64`, **exact**
/// (not quantized).
///
/// The split-JD difference is taken whole-part first to avoid cancellation.
/// This returns the precise instant; whole-second quantization belongs to the
/// *node axis* only:
///
/// - **Node epochs** are quantized to whole seconds at the call site to mirror
///   gnssanalysis `datetime2j2000` (`datetime64[s]` truncation). Parsed SP3
///   nodes use the parser's exact civil-second axis, and sample nodes with a
///   split-Julian representation reconstruct whole-second candidates before
///   reducing the epoch to this continuous `f64` value.
/// - The **query** is evaluated at this exact value, never quantized: truncating
///   a sub-second query epoch would discard up to ~1 s, a kilometre-scale
///   position error at orbital speed (this was a real bug - the node and query
///   conversions must NOT share quantization).
pub(super) fn instant_to_j2000_seconds(instant: &Instant) -> Option<f64> {
    match instant.repr {
        InstantRepr::JulianDate(split) => {
            // (jd - J2000_JD) days -> seconds, whole/fraction kept separate to
            // avoid cancellation (canonical split-to-J2000-seconds reduction).
            Some(j2000_seconds_from_split(split.jd_whole, split.fraction))
        }
        InstantRepr::Nanos(ns) => {
            // Integer ns since the scale epoch - but the parser stores SP3
            // epochs as JulianDate, so this path is not exercised by SP3.
            // J2000 is JD 2451545.0; without a fixed ns-origin convention here
            // we cannot map ns->J2000-seconds unambiguously, so decline.
            let _ = ns;
            None
        }
    }
}

/// Number of nodes in the sliding interpolation window (RTKLIB `NMAX`=10 ->
/// degree-10 polynomial, 11 nodes).
pub(super) const NEVILLE_POINTS: usize = 11;

/// Sliding-window Lagrange (Neville) satellite-POSITION interpolation, matching
/// RTKLIB `preceph.c` pephpos/interppol. Replaces the global not-a-knot cubic
/// spline, which is degree-3 over the whole day and errs ~200 m at the day
/// boundary and across coverage gaps; SP3 15-minute orbit nodes need local
/// ~degree-10 interpolation for sub-cm accuracy. Validated against the external
/// RTKLIB reference and the ZIM2 PPP truth (two-bars doctrine: this channel is a
/// capability gated on the deployed reference, not a bit-exact port of a scipy
/// primitive). The CLOCK channel keeps its cubic spline (locally smooth, matched
/// to the 30 s clock product at the cm level).
///
/// Recipe: restrict to the contiguous run of nodes bracketing `query` (never
/// interpolate across a coverage gap), take the RTKLIB window of up to
/// `NEVILLE_POINTS` nodes centred on the query (shifted inward at run edges),
/// rotate each node's ECEF position about +z by `OMEGA_E_DOT * (t_node - query)`
/// into the query-epoch earth-fixed frame, then Neville-interpolate each axis at
/// the query. Inputs are ascending J2000 seconds (`x`) and km (`kx/ky/kz`).
fn interpolate_position_neville(
    x: &[f64],
    kx: &[f64],
    ky: &[f64],
    kz: &[f64],
    query: f64,
) -> (f64, f64, f64) {
    let n = x.len();

    // Nominal node spacing = smallest positive consecutive gap (robust to one
    // large coverage gap); the gap threshold marks a non-contiguous jump.
    let nominal = nominal_positive_spacing(x).unwrap_or(1.0);
    let gap_thresh = 1.5 * nominal;

    // Last node at or before the query (clamped into range).
    let mut pivot = 0usize;
    while pivot + 1 < n && x[pivot + 1] <= query {
        pivot += 1;
    }
    // The gap policy admits one nominal spacing of extrapolation from either
    // arc. Near the next arc, anchor the window there instead of extrapolating
    // the previous arc across the whole gap.
    if pivot + 1 < n && (x[pivot + 1] - x[pivot]) > gap_thresh && query >= x[pivot + 1] - nominal {
        pivot += 1;
    }

    // Contiguous run [run_lo, run_hi) around the pivot: extend while the
    // neighbour gap stays within the threshold (do not cross a coverage gap).
    let mut run_lo = pivot;
    while run_lo > 0 && (x[run_lo] - x[run_lo - 1]) <= gap_thresh {
        run_lo -= 1;
    }
    let mut run_hi = pivot + 1;
    while run_hi < n && (x[run_hi] - x[run_hi - 1]) <= gap_thresh {
        run_hi += 1;
    }
    let run_len = run_hi - run_lo;

    // RTKLIB window: centre on the pivot, width = min(NEVILLE_POINTS, run_len),
    // clamped to the run.
    let win = NEVILLE_POINTS.min(run_len);
    let half = (NEVILLE_POINTS / 2) as isize;
    let mut start = pivot as isize - half;
    if start < run_lo as isize {
        start = run_lo as isize;
    }
    if start + win as isize > run_hi as isize {
        start = run_hi as isize - win as isize;
    }
    let start = start as usize;

    // Windowed nodes on the (t = node - query) abscissa, earth-rotation-corrected
    // into the query-epoch frame; query is t = 0.
    let mut t = [0.0f64; NEVILLE_POINTS];
    let mut px = [0.0f64; NEVILLE_POINTS];
    let mut py = [0.0f64; NEVILLE_POINTS];
    let mut pz = [0.0f64; NEVILLE_POINTS];
    for j in 0..win {
        let k = start + j;
        let tj = x[k] - query;
        let theta = OMEGA_E_DOT_RAD_S * tj;
        let s = libm::sin(theta);
        let c = libm::cos(theta);
        t[j] = tj;
        px[j] = c * kx[k] - s * ky[k];
        py[j] = s * kx[k] + c * ky[k];
        pz[j] = kz[k];
    }

    let x_km = neville(&t[..win], &px[..win]);
    let y_km = neville(&t[..win], &py[..win]);
    let z_km = neville(&t[..win], &pz[..win]);
    (x_km * KM_TO_M, y_km * KM_TO_M, z_km * KM_TO_M)
}

/// Neville's algorithm evaluated at 0, reproducing RTKLIB `rtkcmn.c` interppol
/// (the abscissa `x` carries node-minus-query offsets, so the query is 0).
pub(super) fn neville(x: &[f64], y: &[f64]) -> f64 {
    let n = y.len();
    let mut c: [f64; NEVILLE_POINTS] = [0.0; NEVILLE_POINTS];
    c[..n].copy_from_slice(&y[..n]);
    for j in 1..n {
        for i in 0..(n - j) {
            c[i] = (x[i + j] * c[i] - x[i] * c[i + 1]) / (x[i + j] - x[i]);
        }
    }
    c[0]
}

/// Evaluate a not-a-knot cubic spline at `query`, reproducing
/// `scipy.interpolate.CubicSpline(x, y)(query)` bit-for-bit.
///
/// `x` must be strictly increasing with `x.len() == y.len() >= 2`.
#[cfg(all(test, sidereon_repo_tests))]
fn eval_cubic_spline(x: &[f64], y: &[f64], query: f64) -> f64 {
    let n = x.len();
    debug_assert_eq!(n, y.len());
    debug_assert!(n >= 2);

    let dydx = solve_not_a_knot_slopes(x, y);
    let (c0, c1, c2, c3) = hermite_segment_coeffs(x, y, &dydx);
    evaluate_ppoly(x, &c0, &c1, &c2, &c3, query)
}

/// Solve the not-a-knot tridiagonal system for the derivative values `s[i]` at
/// each node, exactly as `scipy.interpolate.CubicSpline.__init__` assembles it
/// (`_cubic.py`, scipy 1.17.1) and `scipy.linalg.solve_banded((1,1), ...)`
/// solves it via LAPACK `dgtsv`.
///
/// Banded layout mirrors scipy's `A` of shape `(3, n)`:
/// - `A[1, :]` diagonal `d`
/// - `A[0, 1:]` upper diagonal `du` (i.e. `du[j]` couples row `j` to `j+1`)
/// - `A[2, :-1]` lower diagonal `dl` (i.e. `dl[j]` couples row `j+1` to `j`)
fn solve_not_a_knot_slopes(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();

    // dx[i] = x[i+1]-x[i]; slope[i] = (y[i+1]-y[i])/dx[i]. (scipy: np.diff / dxr)
    let mut dx = vec![0.0; n - 1];
    let mut slope = vec![0.0; n - 1];
    for i in 0..n - 1 {
        dx[i] = x[i + 1] - x[i];
        slope[i] = (y[i + 1] - y[i]) / dx[i];
    }

    // Special case n == 2: not-a-knot is replaced by clamped to the secant
    // slope on both ends (scipy `_cubic.py`: bc -> (1, slope[0])), giving the
    // straight-line Hermite - both derivatives equal slope[0].
    if n == 2 {
        return vec![slope[0], slope[0]];
    }

    // Special case n == 3 with not-a-knot on both ends: scipy builds a 3x3 dense
    // system (a parabola through the points) and solves with LAPACK `gesv`.
    if n == 3 {
        return solve_n3_parabola(&dx, &slope, y);
    }

    // General n >= 4: tridiagonal banded system.
    // Diagonal/off-diagonals as scipy fills them.
    // Interior rows i=1..n-2:
    //   d[i]   = 2*(dx[i-1]+dx[i])
    //   du[i]  (A[0, i+1]) = dx[i-1]
    //   dl[i-1](A[2, i-1]) = dx[i]
    //   b[i]   = 3*(dx[i]*slope[i-1] + dx[i-1]*slope[i])
    let mut d = vec![0.0; n];
    // upper diagonal du[j] for j in 0..n-1 couples row j -> j+1 (A[0, j+1]).
    let mut du = vec![0.0; n - 1];
    // lower diagonal dl[j] for j in 0..n-1 couples row j+1 -> j (A[2, j]).
    let mut dl = vec![0.0; n - 1];
    let mut b = vec![0.0; n];

    for i in 1..n - 1 {
        d[i] = 2.0 * (dx[i - 1] + dx[i]); // A[1, i]
        du[i] = dx[i - 1]; // A[0, i+1] -> our du index i (couples i->i+1)
        dl[i - 1] = dx[i]; // A[2, i-1] -> our dl index i-1 (couples i->i-1)
        b[i] = 3.0 * (dx[i] * slope[i - 1] + dx[i - 1] * slope[i]);
    }

    // not-a-knot start (scipy):
    //   A[1,0]=dx[1]; A[0,1]=x[2]-x[0]; d=x[2]-x[0];
    //   b[0]=((dx[0]+2*d)*dx[1]*slope[0] + dx[0]^2*slope[1]) / d
    {
        let dd = x[2] - x[0];
        d[0] = dx[1]; // A[1,0]
        du[0] = dd; // A[0,1] couples row 0->1
        b[0] = ((dx[0] + 2.0 * dd) * dx[1] * slope[0] + dx[0] * dx[0] * slope[1]) / dd;
    }
    // not-a-knot end (scipy):
    //   A[1,-1]=dx[-2]; A[-1,-2]=x[-1]-x[-3]; d=x[-1]-x[-3];
    //   b[-1]=(dx[-1]^2*slope[-2] + (2*d+dx[-1])*dx[-2]*slope[-1]) / d
    {
        let dd = x[n - 1] - x[n - 3];
        d[n - 1] = dx[n - 2]; // A[1,-1]
        dl[n - 2] = dd; // A[-1,-2] couples row n-1 -> n-2
        b[n - 1] = (dx[n - 2] * dx[n - 2] * slope[n - 3]
            + (2.0 * dd + dx[n - 2]) * dx[n - 3] * slope[n - 2])
            / dd;
    }

    dgtsv(dl, d, du, b)
}

/// n == 3 not-a-knot special case: scipy solves a dense 3x3 `A s = b` via
/// LAPACK `gesv` (partial-pivot LU). Reproduced with the same partial-pivoting
/// Gaussian elimination operation order.
fn solve_n3_parabola(dx: &[f64], slope: &[f64], _y: &[f64]) -> Vec<f64> {
    // A (scipy `_cubic.py` n==3 branch):
    //   A[0,0]=1 A[0,1]=1
    //   A[1,0]=dx[1] A[1,1]=2*(dx[0]+dx[1]) A[1,2]=dx[0]
    //   A[2,1]=1 A[2,2]=1
    // b:
    //   b[0]=2*slope[0]
    //   b[1]=3*(dx[0]*slope[1] + dx[1]*slope[0])
    //   b[2]=2*slope[1]
    let mut a = [
        [1.0, 1.0, 0.0],
        [dx[1], 2.0 * (dx[0] + dx[1]), dx[0]],
        [0.0, 1.0, 1.0],
    ];
    let mut b = [
        2.0 * slope[0],
        3.0 * (dx[0] * slope[1] + dx[1] * slope[0]),
        2.0 * slope[1],
    ];
    gesv3(&mut a, &mut b);
    b.to_vec()
}

/// LAPACK `dgtsv`-equivalent tridiagonal solve (scipy `solve_banded((1,1),...)`
/// dispatch). Partial pivoting, scalar arithmetic, NRHS=1.
///
/// `dl[i]` = sub-diagonal coupling row `i+1`->`i`; `d[i]` = diagonal; `du[i]` =
/// super-diagonal coupling row `i`->`i+1`. Reproduces the Reference-LAPACK
/// `dgtsv.f` operation order, **with one pinned-environment subtlety**: the
/// certified parity target's LAPACK is **Apple Accelerate** (macOS arm64; scipy
/// 1.17.1, `detection method: extraframeworks`), whose `dgtsv` contracts each
/// `acc - fact*x` update into a **fused multiply-add**. So every `y - a*x`
/// elimination/back-substitution update here uses [`libm::fma`]
/// (`libm::fma(-a, x, y)`), NOT a separate multiply then subtract - the
/// per-function FMA-contraction discipline the parity contract requires.
/// Verified 0-ULP against `scipy.linalg.lapack.dgtsv` on this target; on a
/// non-FMA LAPACK build the last bits differ (the portable-mode reality, where
/// 0 ULP is not promised across platforms).
fn dgtsv(mut dl: Vec<f64>, mut d: Vec<f64>, mut du: Vec<f64>, mut b: Vec<f64>) -> Vec<f64> {
    let n = d.len();

    if n == 1 {
        b[0] /= d[0];
        return b;
    }

    // Forward elimination, rows i = 0 .. n-3 (Fortran 1..N-2). On a pivot, the
    // fill-in second super-diagonal is stored back into `dl[i]` (NOT a separate
    // du2 array) - exactly as Reference-LAPACK dgtsv.f does; the back
    // substitution reads it as the B(I+2) coefficient.
    for i in 0..n.saturating_sub(2) {
        if d[i].abs() >= dl[i].abs() {
            // No pivot.
            let fact = dl[i] / d[i];
            d[i + 1] = libm::fma(-fact, du[i], d[i + 1]);
            b[i + 1] = libm::fma(-fact, b[i], b[i + 1]);
            dl[i] = 0.0;
        } else {
            // Pivot (swap rows i and i+1). Note `dl[i] = du[i+1]` happens
            // BEFORE `du[i+1] = -fact*dl[i]`, so the latter uses the new dl[i]
            // (= old du[i+1]).
            let fact = d[i] / dl[i];
            d[i] = dl[i];
            let temp = d[i + 1];
            d[i + 1] = libm::fma(-fact, temp, du[i]);
            dl[i] = du[i + 1];
            du[i + 1] = -fact * dl[i];
            du[i] = temp;
            let tb = b[i];
            b[i] = b[i + 1];
            b[i + 1] = libm::fma(-fact, b[i + 1], tb);
        }
    }

    // Row i = n-2 (Fortran I = N-1) - no du2 fill-in.
    if n > 1 {
        let i = n - 2;
        if d[i].abs() >= dl[i].abs() {
            let fact = dl[i] / d[i];
            d[i + 1] = libm::fma(-fact, du[i], d[i + 1]);
            b[i + 1] = libm::fma(-fact, b[i], b[i + 1]);
        } else {
            let fact = d[i] / dl[i];
            d[i] = dl[i];
            let temp = d[i + 1];
            d[i + 1] = libm::fma(-fact, temp, du[i]);
            du[i] = temp;
            let tb = b[i];
            b[i] = b[i + 1];
            b[i + 1] = libm::fma(-fact, b[i + 1], tb);
        }
    }

    // Back substitution (dgtsv), FMA-contracted as above.
    b[n - 1] /= d[n - 1];
    if n > 1 {
        b[n - 2] = libm::fma(-du[n - 2], b[n - 1], b[n - 2]) / d[n - 2];
    }
    for i in (0..n.saturating_sub(2)).rev() {
        // (b[i] - du[i]*b[i+1] - dl[i]*b[i+2]) / d[i], each subtraction fused.
        let t = libm::fma(-du[i], b[i + 1], b[i]);
        b[i] = libm::fma(-dl[i], b[i + 2], t) / d[i];
    }

    b
}

/// 3x3 dense solve with partial-pivot LU, matching LAPACK `gesv` (`dgesv`) for
/// the n==3 not-a-knot parabola case. As with [`dgtsv`], the certified parity
/// target is Apple Accelerate, whose `dgesv` contracts the `acc - factor*x`
/// elimination and substitution updates into fused multiply-adds; this routine
/// uses [`libm::fma`] to match it bit-for-bit.
fn gesv3(a: &mut [[f64; 3]; 3], b: &mut [f64; 3]) {
    let mut perm = [0usize, 1, 2];
    // LU with partial pivoting (column-major in LAPACK; we keep row-major but
    // pivot by largest |a[col]| in the column, matching the same pivot choice).
    for k in 0..3 {
        // Find pivot row in column k at or below k.
        let mut piv = k;
        let mut best = a[k][k].abs();
        for r in (k + 1)..3 {
            let v = a[r][k].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if piv != k {
            a.swap(k, piv);
            perm.swap(k, piv);
        }
        for r in (k + 1)..3 {
            let factor = a[r][k] / a[k][k];
            a[r][k] = factor;
            for c in (k + 1)..3 {
                a[r][c] = libm::fma(-factor, a[k][c], a[r][c]);
            }
        }
    }
    // Apply row permutation to b.
    let pb = [b[perm[0]], b[perm[1]], b[perm[2]]];
    // Forward solve Ly = Pb (unit lower).
    let mut yv = [0.0; 3];
    for r in 0..3 {
        let mut s = pb[r];
        for c in 0..r {
            s = libm::fma(-a[r][c], yv[c], s);
        }
        yv[r] = s;
    }
    // Back solve Ux = y.
    for r in (0..3).rev() {
        let mut s = yv[r];
        for c in (r + 1)..3 {
            s = libm::fma(-a[r][c], b[c], s);
        }
        b[r] = s / a[r][r];
    }
}

/// Build the per-segment PPoly coefficients exactly as
/// `scipy.interpolate.CubicHermiteSpline.__init__` (scipy 1.17.1):
///
/// ```text
/// dxr   = x[i+1]-x[i]
/// slope = (y[i+1]-y[i])/dxr
/// t     = (dydx[i] + dydx[i+1] - 2*slope)/dxr
/// c0 = t/dxr
/// c1 = (slope - dydx[i])/dxr - t
/// c2 = dydx[i]
/// c3 = y[i]
/// ```
///
/// for segment `i` between `x[i]` and `x[i+1]`, with local variable
/// `s = xval - x[i]`.
fn hermite_segment_coeffs(
    x: &[f64],
    y: &[f64],
    dydx: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut c0 = vec![0.0; n - 1];
    let mut c1 = vec![0.0; n - 1];
    let mut c2 = vec![0.0; n - 1];
    let mut c3 = vec![0.0; n - 1];
    for i in 0..n - 1 {
        let dxr = x[i + 1] - x[i];
        let slope = (y[i + 1] - y[i]) / dxr;
        let t = (dydx[i] + dydx[i + 1] - 2.0 * slope) / dxr;
        c0[i] = t / dxr;
        c1[i] = (slope - dydx[i]) / dxr - t;
        c2[i] = dydx[i];
        c3[i] = y[i];
    }
    (c0, c1, c2, c3)
}

/// Evaluate the PPoly at `query`, reproducing scipy `_ppoly.evaluate` /
/// `find_interval_ascending` (extrapolate=True) and `evaluate_poly1` (dx=0).
///
/// Interval selection: the largest `i` with `x[i] <= query`, clamped to
/// `[0, n-2]`; `query == x[n-1]` maps to interval `n-2` (right-closed); out of
/// bounds extrapolates from interval 0 (below) or `n-2` (above).
///
/// Evaluation order (`evaluate_poly1`, dx=0): with `s = query - x[i]` and
/// `z` accumulating powers via repeated `z *= s`,
/// `res = c3 + c2*s + c1*s^2 + c0*s^3` summed low-power-first.
fn evaluate_ppoly(x: &[f64], c0: &[f64], c1: &[f64], c2: &[f64], c3: &[f64], query: f64) -> f64 {
    let n = x.len();
    let last = n - 2; // last interval index

    // find_interval_ascending with extrapolate=True.
    let interval = if query.is_nan() {
        // scipy returns -1 -> NaN out; propagate NaN.
        return f64::NAN;
    } else if query < x[0] {
        0
    } else if query > x[n - 1] {
        last
    } else {
        // x[0] <= query <= x[n-1]: binary search for i with x[i] <= query < x[i+1];
        // query == x[n-1] -> n-2.
        if query == x[n - 1] {
            last
        } else {
            let mut lo = 0usize;
            let mut hi = n - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if x[mid] <= query {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            lo
        }
    };

    // evaluate_poly1 (dx=0): res = sum_{kp} c[K-kp-1] * z, z = s^kp built by *=.
    let s = query - x[interval];
    let mut res = 0.0;
    let mut z = 1.0;
    // kp = 0 -> coefficient c3 (lowest power), kp=1 -> c2, kp=2 -> c1, kp=3 -> c0.
    res += c3[interval] * z;
    z *= s;
    res += c2[interval] * z;
    z *= s;
    res += c1[interval] * z;
    z *= s;
    res += c0[interval] * z;
    res
}

/// Test-only re-export of the core spline evaluator so the parity test can
/// drive it directly against the scipy golden fixture.
#[cfg(all(test, sidereon_repo_tests))]
pub(super) fn eval_cubic_spline_for_test(x: &[f64], y: &[f64], query: f64) -> f64 {
    eval_cubic_spline(x, y, query)
}

#[cfg(all(test, sidereon_repo_tests))]
mod interp_tests;
