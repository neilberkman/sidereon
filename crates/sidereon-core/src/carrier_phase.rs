//! Carrier-phase combinations, cycle-slip detection, and Hatch smoothing.
//!
//! This module owns the language-independent modeling behind Sidereon'
//! `CarrierPhase` API: phase-cycle conversion, geometry-free and
//! Melbourne-Wubbena combinations, loss-of-lock/GF/MW/data-gap cycle-slip
//! classification, and single/dual-frequency Hatch smoothing.

use crate::combinations;
use crate::constants::C_M_S;
use crate::tolerances::FREQUENCY_DENOMINATOR_EPS_HZ;
use crate::validate;

/// Frequency separation below which wide-lane / narrow-lane denominators are
/// treated as degenerate, hertz.
pub use crate::tolerances::FREQUENCY_DENOMINATOR_EPS_HZ as FREQ_EPSILON_HZ;

/// Default geometry-free cycle-slip threshold, meters.
pub const DEFAULT_GF_THRESHOLD_M: f64 = 0.05;

/// Default Melbourne-Wubbena cycle-slip threshold, wide-lane cycles.
pub const DEFAULT_MW_THRESHOLD_CYCLES: f64 = 4.0;

/// Default maximum gap between consecutive usable arc samples, seconds.
pub const DEFAULT_MIN_ARC_GAP_S: f64 = 300.0;

pub(crate) const MIN_HATCH_WINDOW_CAP: usize = 1;

/// Default Hatch carrier-smoothing window cap, in epochs.
///
/// The smoothing window length is held to at most this many epochs so the code
/// noise keeps averaging down while the carrier-phase ambiguity contribution
/// stays bounded. This is the value every binding hardcodes and the core
/// smoothing goldens run with (the `smooth_code` / `smooth_iono_free_code`
/// tests pass `100`). Feeds the `hatch_window_cap` argument of
/// [`smooth_code`] and [`smooth_iono_free_code`]; the floor enforced internally
/// is [`MIN_HATCH_WINDOW_CAP`].
pub const DEFAULT_HATCH_WINDOW_CAP: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InvalidHatchWindowCap;

/// Error produced by carrier-phase scalar combinations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarrierPhaseError {
    /// The denominator for a frequency combination is degenerate.
    EqualFrequencies,
    /// Cycle-to-meter conversion requires a positive carrier frequency.
    InvalidFrequency,
    /// Observation values must be finite and produce finite combinations.
    InvalidObservation,
    /// Cycle-slip and smoothing thresholds must be finite and in range.
    InvalidThreshold,
}

impl core::fmt::Display for CarrierPhaseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EqualFrequencies => write!(f, "equal carrier frequencies"),
            Self::InvalidFrequency => write!(f, "carrier frequency must be positive"),
            Self::InvalidObservation => write!(f, "carrier observations must be finite"),
            Self::InvalidThreshold => write!(f, "carrier thresholds must be finite and sane"),
        }
    }
}

impl std::error::Error for CarrierPhaseError {}

/// One epoch in a single-satellite carrier-phase arc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArcEpoch {
    /// Band-1 carrier phase, cycles.
    pub phi1_cycles: Option<f64>,
    /// Band-2 carrier phase, cycles.
    pub phi2_cycles: Option<f64>,
    /// Band-1 code pseudorange, meters.
    pub p1_m: Option<f64>,
    /// Band-2 code pseudorange, meters.
    pub p2_m: Option<f64>,
    /// Band-1 loss-of-lock indicator.
    pub lli1: Option<i64>,
    /// Band-2 loss-of-lock indicator.
    pub lli2: Option<i64>,
    /// Band-1 carrier frequency, hertz. `None` means the epoch is skipped.
    pub f1_hz: Option<f64>,
    /// Band-2 carrier frequency, hertz. `None` means the epoch is skipped.
    pub f2_hz: Option<f64>,
    /// Comparable epoch coordinate in seconds, when the caller can supply one.
    pub gap_time_s: Option<f64>,
}

/// Options controlling cycle-slip classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CycleSlipOptions {
    /// Geometry-free step threshold, meters.
    pub gf_threshold_m: f64,
    /// Melbourne-Wubbena step threshold, wide-lane cycles.
    pub mw_threshold_cycles: f64,
    /// Data-gap threshold, seconds.
    pub min_arc_gap_s: f64,
}

impl Default for CycleSlipOptions {
    fn default() -> Self {
        Self {
            gf_threshold_m: DEFAULT_GF_THRESHOLD_M,
            mw_threshold_cycles: DEFAULT_MW_THRESHOLD_CYCLES,
            min_arc_gap_s: DEFAULT_MIN_ARC_GAP_S,
        }
    }
}

/// Reason a carrier arc was split at an epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlipReason {
    /// Loss-of-lock indicator bit 0 was set on either band.
    Lli,
    /// Gap to the previous usable sample exceeded the configured threshold.
    DataGap,
    /// Geometry-free phase step exceeded the configured threshold.
    GeometryFree,
    /// Melbourne-Wubbena step exceeded the configured wide-lane-cycle threshold.
    MelbourneWubbena,
}

/// Cycle-slip classification for one input epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct SlipResult {
    /// Whether any slip reason was flagged.
    pub slip: bool,
    /// Slip reasons in deterministic Sidereon API order.
    pub reasons: Vec<SlipReason>,
    /// Current geometry-free phase, meters, when computable.
    pub gf_m: Option<f64>,
    /// Current Melbourne-Wubbena combination, meters, when computable.
    pub mw_m: Option<f64>,
    /// Whether the epoch was skipped because a frequency was unavailable.
    pub skipped: bool,
}

/// Hatch-smoothed single-frequency code output for one epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothCodeResult {
    /// Smoothed code pseudorange, meters, when computable.
    pub p_smooth_m: Option<f64>,
    /// Hatch window length used at this epoch.
    pub window: usize,
    /// True when a prior running window was reset by a slip at this epoch.
    pub reset: bool,
}

/// Hatch-smoothed ionosphere-free code output for one epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IonoFreeSmoothResult {
    /// Smoothed ionosphere-free code pseudorange, meters, when computable.
    pub p_smooth_m: Option<f64>,
    /// Instantaneous ionosphere-free code, meters.
    pub p_if_m: Option<f64>,
    /// Instantaneous ionosphere-free carrier phase, meters.
    pub l_if_m: Option<f64>,
    /// Hatch window length used at this epoch.
    pub window: usize,
    /// True when a prior running window was reset by a slip at this epoch.
    pub reset: bool,
}

/// Carrier phase in meters, `L = c / f * phi`.
pub fn phase_meters(phi_cycles: f64, f_hz: f64) -> Result<f64, CarrierPhaseError> {
    let f_hz = validate_frequency(f_hz, "f_hz")?;
    let phi_cycles = validate_observation(phi_cycles, "phi_cycles")?;
    validate_observation(C_M_S / f_hz * phi_cycles, "phase_m")
}

/// Geometry-free phase combination `L_GF = L1 - L2`, meters.
pub fn geometry_free(l1_m: f64, l2_m: f64) -> Result<f64, CarrierPhaseError> {
    let l1_m = validate_observation(l1_m, "l1_m")?;
    let l2_m = validate_observation(l2_m, "l2_m")?;
    validate_observation(l1_m - l2_m, "geometry_free_m")
}

/// Wide-lane wavelength `lambda_WL = c / (f1 - f2)`, meters.
pub fn wide_lane_wavelength(f1_hz: f64, f2_hz: f64) -> Result<f64, CarrierPhaseError> {
    let f1_hz = validate_frequency(f1_hz, "f1_hz")?;
    let f2_hz = validate_frequency(f2_hz, "f2_hz")?;
    if (f1_hz - f2_hz).abs() < FREQUENCY_DENOMINATOR_EPS_HZ {
        Err(CarrierPhaseError::EqualFrequencies)
    } else {
        Ok(C_M_S / (f1_hz - f2_hz))
    }
}

/// Narrow-lane code `P_NL = (f1*P1 + f2*P2) / (f1 + f2)`, meters.
pub fn narrow_lane_code(
    p1_m: f64,
    p2_m: f64,
    f1_hz: f64,
    f2_hz: f64,
) -> Result<f64, CarrierPhaseError> {
    let f1_hz = validate_frequency(f1_hz, "f1_hz")?;
    let f2_hz = validate_frequency(f2_hz, "f2_hz")?;
    let p1_m = validate_observation(p1_m, "p1_m")?;
    let p2_m = validate_observation(p2_m, "p2_m")?;
    if (f1_hz + f2_hz).abs() < FREQUENCY_DENOMINATOR_EPS_HZ {
        Err(CarrierPhaseError::EqualFrequencies)
    } else {
        validate_observation((f1_hz * p1_m + f2_hz * p2_m) / (f1_hz + f2_hz), "p_nl_m")
    }
}

/// Melbourne-Wubbena combination, meters.
pub fn melbourne_wubbena(
    phi1_cycles: f64,
    phi2_cycles: f64,
    p1_m: f64,
    p2_m: f64,
    f1_hz: f64,
    f2_hz: f64,
) -> Result<f64, CarrierPhaseError> {
    let lambda_wl = wide_lane_wavelength(f1_hz, f2_hz)?;
    let p_nl = narrow_lane_code(p1_m, p2_m, f1_hz, f2_hz)?;
    let phi1_cycles = validate_observation(phi1_cycles, "phi1_cycles")?;
    let phi2_cycles = validate_observation(phi2_cycles, "phi2_cycles")?;
    let l_wl = lambda_wl * (phi1_cycles - phi2_cycles);
    let l_wl = validate_observation(l_wl, "wide_lane_phase_m")?;
    validate_observation(l_wl - p_nl, "melbourne_wubbena_m")
}

/// Code-minus-carrier diagnostic `CMC = P - L`, meters.
pub fn code_minus_carrier(p_m: f64, phi_cycles: f64, f_hz: f64) -> Result<f64, CarrierPhaseError> {
    let p_m = validate_observation(p_m, "p_m")?;
    let l_m = phase_meters(phi_cycles, f_hz)?;
    validate_observation(p_m - l_m, "code_minus_carrier_m")
}

/// Melbourne-Wubbena wide-lane ambiguity estimate in wide-lane cycles,
/// `(MW combination) / lambda_WL`.
pub fn wide_lane_cycles(
    phi1_cycles: f64,
    phi2_cycles: f64,
    p1_m: f64,
    p2_m: f64,
    f1_hz: f64,
    f2_hz: f64,
) -> Result<f64, CarrierPhaseError> {
    let mw_m = melbourne_wubbena(phi1_cycles, phi2_cycles, p1_m, p2_m, f1_hz, f2_hz)?;
    let lambda_wl = wide_lane_wavelength(f1_hz, f2_hz)?;
    validate_observation(mw_m / lambda_wl, "wide_lane_cycles")
}

fn validate_frequency(f_hz: f64, field: &'static str) -> Result<f64, CarrierPhaseError> {
    validate::finite_positive(f_hz, field).map_err(|_| CarrierPhaseError::InvalidFrequency)
}

fn validate_observation(value: f64, field: &'static str) -> Result<f64, CarrierPhaseError> {
    validate::finite(value, field).map_err(|_| CarrierPhaseError::InvalidObservation)
}

fn validate_optional_observation(
    value: Option<f64>,
    field: &'static str,
) -> Result<(), CarrierPhaseError> {
    match value {
        Some(value) => validate_observation(value, field).map(|_| ()),
        None => Ok(()),
    }
}

fn validate_optional_frequency(
    value: Option<f64>,
    field: &'static str,
) -> Result<(), CarrierPhaseError> {
    match value {
        Some(value) => validate_frequency(value, field).map(|_| ()),
        None => Ok(()),
    }
}

fn validate_options(options: CycleSlipOptions) -> Result<(), CarrierPhaseError> {
    validate::finite_nonneg(options.gf_threshold_m, "gf_threshold_m")
        .map_err(|_| CarrierPhaseError::InvalidThreshold)?;
    validate::finite_nonneg(options.mw_threshold_cycles, "mw_threshold_cycles")
        .map_err(|_| CarrierPhaseError::InvalidThreshold)?;
    validate::finite_positive(options.min_arc_gap_s, "min_arc_gap_s")
        .map_err(|_| CarrierPhaseError::InvalidThreshold)?;
    Ok(())
}

fn validate_arc_epoch(ep: &ArcEpoch) -> Result<(), CarrierPhaseError> {
    validate_optional_observation(ep.phi1_cycles, "phi1_cycles")?;
    validate_optional_observation(ep.phi2_cycles, "phi2_cycles")?;
    validate_optional_observation(ep.p1_m, "p1_m")?;
    validate_optional_observation(ep.p2_m, "p2_m")?;
    validate_optional_frequency(ep.f1_hz, "f1_hz")?;
    validate_optional_frequency(ep.f2_hz, "f2_hz")?;
    validate_optional_observation(ep.gap_time_s, "gap_time_s")
}

pub(crate) fn validate_hatch_window_cap(
    hatch_window_cap: usize,
) -> Result<usize, InvalidHatchWindowCap> {
    if hatch_window_cap < MIN_HATCH_WINDOW_CAP {
        Err(InvalidHatchWindowCap)
    } else {
        Ok(hatch_window_cap)
    }
}

fn clamp_hatch_window_cap(hatch_window_cap: usize) -> usize {
    validate_hatch_window_cap(hatch_window_cap).unwrap_or(MIN_HATCH_WINDOW_CAP)
}

/// Detect cycle slips on a time-ordered single-satellite arc.
pub fn detect_cycle_slips(
    arc: &[ArcEpoch],
    options: CycleSlipOptions,
) -> Result<Vec<SlipResult>, CarrierPhaseError> {
    validate_options(options)?;
    let mut results = Vec::with_capacity(arc.len());
    let mut prev = None;

    for ep in arc {
        validate_arc_epoch(ep)?;
        let result = classify_epoch(ep, prev, options)?;
        if dual_frequency_reference_usable(&result) {
            prev = Some(PreviousEpoch {
                gf_m: result.gf_m,
                mw_m: result.mw_m,
                gap_time_s: ep.gap_time_s,
            });
        }
        results.push(result);
    }

    Ok(results)
}

fn dual_frequency_reference_usable(result: &SlipResult) -> bool {
    !result.skipped && (result.gf_m.is_some() || result.mw_m.is_some())
}

/// Single-frequency Hatch carrier-smoothed code on band 1.
pub fn smooth_code(
    arc: &[ArcEpoch],
    options: CycleSlipOptions,
    hatch_window_cap: usize,
) -> Result<Vec<SmoothCodeResult>, CarrierPhaseError> {
    let hatch_window_cap = clamp_hatch_window_cap(hatch_window_cap);
    let slips = detect_band1_hatch_slips(arc, options)?;
    let mut results = Vec::with_capacity(arc.len());
    let mut state = None;

    for (ep, slip) in arc.iter().zip(&slips) {
        let (result, next_state) = hatch_epoch(ep, slip, state, hatch_window_cap);
        validate_smooth_code_result(result)?;
        results.push(result);
        state = next_state;
    }

    Ok(results)
}

fn detect_band1_hatch_slips(
    arc: &[ArcEpoch],
    options: CycleSlipOptions,
) -> Result<Vec<SlipResult>, CarrierPhaseError> {
    validate_options(options)?;
    let mut results = Vec::with_capacity(arc.len());
    let mut prev_band1 = None;
    let mut prev_dual = None;

    for ep in arc {
        validate_arc_epoch(ep)?;
        let band1 = classify_single_frequency_hatch_epoch(ep, prev_band1, options);
        let dual = if ep.f2_hz.is_some() {
            Some(classify_epoch(ep, prev_dual, options)?)
        } else {
            None
        };

        let result = match dual.as_ref() {
            Some(dual) if !band1.skipped => combine_band1_hatch_slip(&band1, dual),
            _ => band1,
        };

        if band1_hatch_usable(ep) {
            prev_band1 = Some(PreviousEpoch {
                gf_m: None,
                mw_m: None,
                gap_time_s: ep.gap_time_s,
            });
        }
        if let Some(dual) = dual {
            if dual_frequency_reference_usable(&dual) {
                prev_dual = Some(PreviousEpoch {
                    gf_m: dual.gf_m,
                    mw_m: dual.mw_m,
                    gap_time_s: ep.gap_time_s,
                });
            }
        }
        results.push(result);
    }

    Ok(results)
}

fn combine_band1_hatch_slip(band1: &SlipResult, dual: &SlipResult) -> SlipResult {
    let mut reasons = Vec::new();
    if dual.reasons.contains(&SlipReason::Lli) || band1.reasons.contains(&SlipReason::Lli) {
        reasons.push(SlipReason::Lli);
    }
    if band1.reasons.contains(&SlipReason::DataGap) {
        reasons.push(SlipReason::DataGap);
    }
    if dual.reasons.contains(&SlipReason::GeometryFree) {
        reasons.push(SlipReason::GeometryFree);
    }
    if dual.reasons.contains(&SlipReason::MelbourneWubbena) {
        reasons.push(SlipReason::MelbourneWubbena);
    }

    SlipResult {
        slip: !reasons.is_empty(),
        reasons,
        gf_m: dual.gf_m,
        mw_m: dual.mw_m,
        skipped: false,
    }
}

/// Dual-frequency ionosphere-free Hatch carrier-smoothed code.
pub fn smooth_iono_free_code(
    arc: &[ArcEpoch],
    options: CycleSlipOptions,
    hatch_window_cap: usize,
) -> Result<Vec<IonoFreeSmoothResult>, CarrierPhaseError> {
    let hatch_window_cap = clamp_hatch_window_cap(hatch_window_cap);
    let slips = detect_cycle_slips(arc, options)?;
    let mut results = Vec::with_capacity(arc.len());
    let mut state = None;

    for (ep, slip) in arc.iter().zip(&slips) {
        let (result, next_state) = iono_free_hatch_epoch(ep, slip, state, hatch_window_cap)?;
        validate_iono_free_smooth_result(result)?;
        results.push(result);
        state = next_state;
    }

    Ok(results)
}

#[derive(Debug, Clone, Copy)]
struct PreviousEpoch {
    gf_m: Option<f64>,
    mw_m: Option<f64>,
    gap_time_s: Option<f64>,
}

fn classify_epoch(
    ep: &ArcEpoch,
    prev: Option<PreviousEpoch>,
    options: CycleSlipOptions,
) -> Result<SlipResult, CarrierPhaseError> {
    let (Some(f1), Some(f2)) = (ep.f1_hz, ep.f2_hz) else {
        return Ok(SlipResult {
            slip: false,
            reasons: Vec::new(),
            gf_m: None,
            mw_m: None,
            skipped: true,
        });
    };

    let gf = current_gf(ep.phi1_cycles, ep.phi2_cycles, f1, f2)?;
    let mw = current_mw(ep.phi1_cycles, ep.phi2_cycles, ep.p1_m, ep.p2_m, f1, f2)?;

    let mut reasons = Vec::new();
    if loss_of_lock(ep) {
        reasons.push(SlipReason::Lli);
    }
    if gap_reason(ep.gap_time_s, prev, options.min_arc_gap_s) {
        reasons.push(SlipReason::DataGap);
    }
    if gf_reason(gf, prev, options.gf_threshold_m) {
        reasons.push(SlipReason::GeometryFree);
    }
    if mw_reason(mw, prev, f1, f2, options.mw_threshold_cycles) {
        reasons.push(SlipReason::MelbourneWubbena);
    }

    Ok(SlipResult {
        slip: !reasons.is_empty(),
        reasons,
        gf_m: gf,
        mw_m: mw,
        skipped: false,
    })
}

fn classify_single_frequency_hatch_epoch(
    ep: &ArcEpoch,
    prev: Option<PreviousEpoch>,
    options: CycleSlipOptions,
) -> SlipResult {
    if !band1_hatch_usable(ep) {
        return SlipResult {
            slip: false,
            reasons: Vec::new(),
            gf_m: None,
            mw_m: None,
            skipped: true,
        };
    }

    let mut reasons = Vec::new();
    if lli_set(ep.lli1) {
        reasons.push(SlipReason::Lli);
    }
    if gap_reason(ep.gap_time_s, prev, options.min_arc_gap_s) {
        reasons.push(SlipReason::DataGap);
    }

    SlipResult {
        slip: !reasons.is_empty(),
        reasons,
        gf_m: None,
        mw_m: None,
        skipped: false,
    }
}

fn band1_hatch_usable(ep: &ArcEpoch) -> bool {
    ep.f1_hz.is_some() && ep.p1_m.is_some() && ep.phi1_cycles.is_some()
}

fn current_gf(
    phi1: Option<f64>,
    phi2: Option<f64>,
    f1: f64,
    f2: f64,
) -> Result<Option<f64>, CarrierPhaseError> {
    let (Some(phi1), Some(phi2)) = (phi1, phi2) else {
        return Ok(None);
    };
    let l1 = phase_meters(phi1, f1)?;
    let l2 = phase_meters(phi2, f2)?;
    geometry_free(l1, l2).map(Some)
}

fn current_mw(
    phi1: Option<f64>,
    phi2: Option<f64>,
    p1: Option<f64>,
    p2: Option<f64>,
    f1: f64,
    f2: f64,
) -> Result<Option<f64>, CarrierPhaseError> {
    let (Some(phi1), Some(phi2), Some(p1), Some(p2)) = (phi1, phi2, p1, p2) else {
        return Ok(None);
    };
    melbourne_wubbena(phi1, phi2, p1, p2, f1, f2).map(Some)
}

fn gf_reason(gf: Option<f64>, prev: Option<PreviousEpoch>, threshold_m: f64) -> bool {
    match (gf, prev.and_then(|p| p.gf_m)) {
        (Some(gf), Some(prev_gf)) => (gf - prev_gf).abs() > threshold_m,
        _ => false,
    }
}

fn mw_reason(
    mw: Option<f64>,
    prev: Option<PreviousEpoch>,
    f1: f64,
    f2: f64,
    threshold_cycles: f64,
) -> bool {
    let (Some(mw), Some(prev_mw), Ok(lambda_wl)) =
        (mw, prev.and_then(|p| p.mw_m), wide_lane_wavelength(f1, f2))
    else {
        return false;
    };
    ((mw - prev_mw).abs() / lambda_wl.abs()) > threshold_cycles
}

fn gap_reason(time_s: Option<f64>, prev: Option<PreviousEpoch>, min_arc_gap_s: f64) -> bool {
    match (time_s, prev.and_then(|p| p.gap_time_s)) {
        (Some(time_s), Some(prev_time_s)) => (time_s - prev_time_s).abs() > min_arc_gap_s,
        _ => false,
    }
}

fn loss_of_lock(ep: &ArcEpoch) -> bool {
    lli_set(ep.lli1) || lli_set(ep.lli2)
}

fn lli_set(lli: Option<i64>) -> bool {
    lli.is_some_and(|value| (value & 1) == 1)
}

#[derive(Debug, Clone, Copy)]
struct HatchState {
    p_smooth_m: f64,
    l1_m: f64,
    window: usize,
}

fn hatch_epoch(
    ep: &ArcEpoch,
    slip: &SlipResult,
    state: Option<HatchState>,
    cap: usize,
) -> (SmoothCodeResult, Option<HatchState>) {
    if slip.skipped || !band1_hatch_usable(ep) {
        return (
            SmoothCodeResult {
                p_smooth_m: None,
                window: 0,
                reset: false,
            },
            None,
        );
    }

    let (Some(f1), Some(p1), Some(phi1)) = (ep.f1_hz, ep.p1_m, ep.phi1_cycles) else {
        unreachable!("presence checked above")
    };
    match phase_meters(phi1, f1) {
        Ok(l1) => do_hatch(p1, l1, slip.slip, state, cap),
        Err(_) => (
            SmoothCodeResult {
                p_smooth_m: None,
                window: 0,
                reset: false,
            },
            None,
        ),
    }
}

fn do_hatch(
    p1_m: f64,
    l1_m: f64,
    slip: bool,
    state: Option<HatchState>,
    cap: usize,
) -> (SmoothCodeResult, Option<HatchState>) {
    if slip || state.is_none() {
        let result = SmoothCodeResult {
            p_smooth_m: Some(p1_m),
            window: 1,
            reset: state.is_some() && slip,
        };
        let next = HatchState {
            p_smooth_m: p1_m,
            l1_m,
            window: 1,
        };
        return (result, Some(next));
    }

    let state = state.expect("checked above");
    let window = (state.window + 1).min(cap);
    let n = window as f64;
    let p_smooth_m = p1_m / n + (n - 1.0) / n * (state.p_smooth_m + (l1_m - state.l1_m));
    let result = SmoothCodeResult {
        p_smooth_m: Some(p_smooth_m),
        window,
        reset: false,
    };
    let next = HatchState {
        p_smooth_m,
        l1_m,
        window,
    };
    (result, Some(next))
}

#[derive(Debug, Clone, Copy)]
struct IonoFreeHatchState {
    p_smooth_m: f64,
    l_if_m: f64,
    window: usize,
}

fn iono_free_hatch_epoch(
    ep: &ArcEpoch,
    slip: &SlipResult,
    state: Option<IonoFreeHatchState>,
    cap: usize,
) -> Result<(IonoFreeSmoothResult, Option<IonoFreeHatchState>), CarrierPhaseError> {
    let Some((p_if_m, l_if_m)) = current_iono_free_code_phase(ep)? else {
        return Ok((
            IonoFreeSmoothResult {
                p_smooth_m: None,
                p_if_m: None,
                l_if_m: None,
                window: 0,
                reset: false,
            },
            None,
        ));
    };

    Ok(do_iono_free_hatch(p_if_m, l_if_m, slip.slip, state, cap))
}

fn current_iono_free_code_phase(ep: &ArcEpoch) -> Result<Option<(f64, f64)>, CarrierPhaseError> {
    let (Some(f1), Some(f2), Some(p1), Some(p2), Some(phi1), Some(phi2)) = (
        ep.f1_hz,
        ep.f2_hz,
        ep.p1_m,
        ep.p2_m,
        ep.phi1_cycles,
        ep.phi2_cycles,
    ) else {
        return Ok(None);
    };

    let p_if = combinations::ionosphere_free(p1, p2, f1, f2).map_err(map_combination_error)?;
    let l_if = combinations::ionosphere_free_phase_cycles(phi1, phi2, f1, f2)
        .map_err(map_combination_error)?;
    Ok(Some((p_if, l_if)))
}

fn map_combination_error(error: combinations::IonosphereFreeError) -> CarrierPhaseError {
    match error {
        combinations::IonosphereFreeError::EqualFrequencies => CarrierPhaseError::EqualFrequencies,
        combinations::IonosphereFreeError::InvalidFrequency => CarrierPhaseError::InvalidFrequency,
        combinations::IonosphereFreeError::InvalidObservation => {
            CarrierPhaseError::InvalidObservation
        }
        combinations::IonosphereFreeError::UnknownSystem(_)
        | combinations::IonosphereFreeError::UnknownBand { .. } => {
            CarrierPhaseError::InvalidFrequency
        }
    }
}

fn validate_smooth_code_result(result: SmoothCodeResult) -> Result<(), CarrierPhaseError> {
    validate_optional_observation(result.p_smooth_m, "p_smooth_m")
}

fn validate_iono_free_smooth_result(result: IonoFreeSmoothResult) -> Result<(), CarrierPhaseError> {
    validate_optional_observation(result.p_smooth_m, "p_smooth_m")?;
    validate_optional_observation(result.p_if_m, "p_if_m")?;
    validate_optional_observation(result.l_if_m, "l_if_m")
}

fn do_iono_free_hatch(
    p_if_m: f64,
    l_if_m: f64,
    slip: bool,
    state: Option<IonoFreeHatchState>,
    cap: usize,
) -> (IonoFreeSmoothResult, Option<IonoFreeHatchState>) {
    if slip || state.is_none() {
        let result = IonoFreeSmoothResult {
            p_smooth_m: Some(p_if_m),
            p_if_m: Some(p_if_m),
            l_if_m: Some(l_if_m),
            window: 1,
            reset: state.is_some() && slip,
        };
        let next = IonoFreeHatchState {
            p_smooth_m: p_if_m,
            l_if_m,
            window: 1,
        };
        return (result, Some(next));
    }

    let state = state.expect("checked above");
    let window = (state.window + 1).min(cap);
    let n = window as f64;
    let p_smooth_m = p_if_m / n + (n - 1.0) / n * (state.p_smooth_m + (l_if_m - state.l_if_m));
    let result = IonoFreeSmoothResult {
        p_smooth_m: Some(p_smooth_m),
        p_if_m: Some(p_if_m),
        l_if_m: Some(l_if_m),
        window,
        reset: false,
    };
    let next = IonoFreeHatchState {
        p_smooth_m,
        l_if_m,
        window,
    };
    (result, Some(next))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(bits: u64) -> f64 {
        f64::from_bits(bits)
    }

    fn b(value: Option<f64>) -> Option<u64> {
        value.map(f64::to_bits)
    }

    fn oracle_arc() -> Vec<ArcEpoch> {
        let rows = [
            (
                0_u64,
                0x419ad7697cf35157,
                0x4194f2cad78dd8ca,
                0x4174689c023d70a4,
                0x4174689bfd06a506,
                0,
                0,
                Some(0x41d779c018000000),
                Some(0x41d24aec20000000),
            ),
            (
                1,
                0x419ad771514355cd,
                0x4194f2d0f16095a4,
                0x417468a1f420c49c,
                0x417468a1f5c5f3ee,
                0,
                0,
                Some(0x41d779c018000000),
                Some(0x41d24aec20000000),
            ),
            (
                2,
                0x419ad779344a2977,
                0x4194f2d716aa7f95,
                0x417468a7f9374bc6,
                0x417468a7f499bdb8,
                0,
                0,
                Some(0x41d779c018000000),
                Some(0x41d24aec20000000),
            ),
            (
                3,
                0x419ad7812607cc59,
                0x4194f2dd476b96a2,
                0x417468adffe76c8b,
                0x417468ae036d8781,
                0,
                0,
                Some(0x41d779c018000000),
                Some(0x41d24aec20000000),
            ),
            (
                4,
                0x419ad7893e7c3e72,
                0x4194f2e383a3dac9,
                0x417468b41a45a1cb,
                0x417468b41574847e,
                0,
                0,
                Some(0x41d779c018000000),
                Some(0x41d24aec20000000),
            ),
            (
                5,
                0x419ad7914da77fc0,
                0x4194f2e9cb534c08,
                0x417468ba38a3d70a,
                0x417468ba3ba4773d,
                0,
                0,
                Some(0x41d779c018000000),
                Some(0x41d24aec20000000),
            ),
            (
                6,
                0x419ad7996b899045,
                0x4194f2f01e79ea62,
                0x417468c06ad91687,
                0x417468c066ca2c8c,
                0,
                1,
                Some(0x41d779c018000000),
                Some(0x41d24aec20000000),
            ),
            (
                7,
                0x419ad7a198226fff,
                0x4194f2f67d17b5d4,
                0x417468c6a06a7ef9,
                0x417468c6a2bcaea7,
                0,
                0,
                Some(0x41d779c018000000),
                Some(0x41d24aec20000000),
            ),
            (
                8,
                0x419ad7a9d3721ef1,
                0x4194f2fce72cae62,
                0x417468cce5d2f1a9,
                0x417468cce471c01e,
                0,
                0,
                None,
                Some(0x41d24aec20000000),
            ),
        ];

        rows.into_iter()
            .map(|(epoch, phi1, phi2, p1, p2, lli1, lli2, f1, f2)| ArcEpoch {
                phi1_cycles: Some(f(phi1)),
                phi2_cycles: Some(f(phi2)),
                p1_m: Some(f(p1)),
                p2_m: Some(f(p2)),
                lli1: Some(lli1),
                lli2: Some(lli2),
                f1_hz: f1.map(f),
                f2_hz: f2.map(f),
                gap_time_s: Some(epoch as f64),
            })
            .collect()
    }

    #[test]
    fn scalar_combinations_match_python_oracle_bits() {
        let f1 = f(0x41d779c018000000);
        let f2 = f(0x41d24aec20000000);

        assert_eq!(
            phase_meters(123_456_789.25, f1).unwrap().to_bits(),
            0x4176679b5dbb7fd0
        );
        assert_eq!(
            code_minus_carrier(23_000_010.25, 123_456_789.25, f1)
                .unwrap()
                .to_bits(),
            0xc11e17ae6edff400
        );
        assert_eq!(
            geometry_free(100.0, 60.0).unwrap().to_bits(),
            0x4044000000000000
        );
        assert_eq!(
            wide_lane_wavelength(f1, f2).unwrap().to_bits(),
            0x3feb94d5e5a6844d
        );
        assert_eq!(
            narrow_lane_code(10.0, 12.0, f1, f2).unwrap().to_bits(),
            0x4025c077975b8fe2
        );
        assert_eq!(
            melbourne_wubbena(5.0, 3.0, 10.0, 12.0, f1, f2)
                .unwrap()
                .to_bits(),
            0xc0224ddcdaa6bf58
        );
    }

    #[test]
    fn invalid_frequency_modes_are_tagged() {
        let f1 = f(0x41d779c018000000);
        let f2 = f(0x41d24aec20000000);
        assert_eq!(
            phase_meters(100.0, 0.0),
            Err(CarrierPhaseError::InvalidFrequency)
        );
        assert_eq!(
            phase_meters(100.0, f64::NAN),
            Err(CarrierPhaseError::InvalidFrequency)
        );
        assert_eq!(
            phase_meters(100.0, f64::INFINITY),
            Err(CarrierPhaseError::InvalidFrequency)
        );
        assert_eq!(
            wide_lane_wavelength(f1, f1),
            Err(CarrierPhaseError::EqualFrequencies)
        );
        assert_eq!(
            wide_lane_wavelength(f1, f64::INFINITY),
            Err(CarrierPhaseError::InvalidFrequency)
        );
        assert_eq!(
            narrow_lane_code(10.0, 12.0, f1, -f2),
            Err(CarrierPhaseError::InvalidFrequency)
        );
        assert_eq!(
            melbourne_wubbena(1.0, 2.0, 3.0, 4.0, f1, f1),
            Err(CarrierPhaseError::EqualFrequencies)
        );
    }

    #[test]
    fn invalid_observations_and_thresholds_are_tagged() {
        let f1 = f(0x41d779c018000000);
        let f2 = f(0x41d24aec20000000);
        assert_eq!(
            phase_meters(f64::NAN, f1),
            Err(CarrierPhaseError::InvalidObservation)
        );
        assert_eq!(
            geometry_free(f64::INFINITY, 1.0),
            Err(CarrierPhaseError::InvalidObservation)
        );
        assert_eq!(
            narrow_lane_code(10.0, f64::NAN, f1, f2),
            Err(CarrierPhaseError::InvalidObservation)
        );
        assert_eq!(
            melbourne_wubbena(f64::NAN, 2.0, 3.0, 4.0, f1, f2),
            Err(CarrierPhaseError::InvalidObservation)
        );
        assert_eq!(
            wide_lane_cycles(1.0, 2.0, f64::INFINITY, 4.0, f1, f2),
            Err(CarrierPhaseError::InvalidObservation)
        );

        let options = CycleSlipOptions {
            gf_threshold_m: f64::NAN,
            ..CycleSlipOptions::default()
        };
        assert_eq!(
            detect_cycle_slips(&oracle_arc(), options),
            Err(CarrierPhaseError::InvalidThreshold)
        );

        let mut arc = oracle_arc();
        arc[0].p1_m = Some(f64::NAN);
        assert_eq!(
            smooth_code(&arc, CycleSlipOptions::default(), 100),
            Err(CarrierPhaseError::InvalidObservation)
        );
    }

    #[test]
    fn cycle_slip_classification_matches_python_oracle_bits() {
        let actual = detect_cycle_slips(&oracle_arc(), CycleSlipOptions::default())
            .expect("valid cycle-slip arc");
        let expected = [
            (
                false,
                vec![],
                Some(0xc0e07fd931e60e00),
                Some(0xc0f7618a9fb55c00),
                false,
            ),
            (
                false,
                vec![],
                Some(0xc0e07fd93c7f8a00),
                Some(0xc0f76189f4fdf000),
                false,
            ),
            (
                false,
                vec![],
                Some(0xc0e07fd947190400),
                Some(0xc0f7618b8b4d9c00),
                false,
            ),
            (
                false,
                vec![],
                Some(0xc0e07fd951b28000),
                Some(0xc0f76189d67f0600),
                false,
            ),
            (
                true,
                vec![SlipReason::GeometryFree, SlipReason::MelbourneWubbena],
                Some(0xc0e07fb4d2fb7200),
                Some(0xc0f76136a660be00),
                false,
            ),
            (
                false,
                vec![],
                Some(0xc0e07fb4dd94ec00),
                Some(0xc0f76136155f9c00),
                false,
            ),
            (
                true,
                vec![SlipReason::Lli],
                Some(0xc0e07fb4e82e6800),
                Some(0xc0f76137a3e94c00),
                false,
            ),
            (
                false,
                vec![],
                Some(0xc0e07fb4f2c7e200),
                Some(0xc0f761373e423d00),
                false,
            ),
            (false, vec![], None, None, true),
        ];

        for (got, (slip, reasons, gf, mw, skipped)) in actual.iter().zip(expected) {
            assert_eq!(got.slip, slip);
            assert_eq!(got.reasons, reasons);
            assert_eq!(b(got.gf_m), gf);
            assert_eq!(b(got.mw_m), mw);
            assert_eq!(got.skipped, skipped);
        }
    }

    #[test]
    fn data_gap_uses_previous_usable_epoch() {
        let mut arc = oracle_arc();
        arc.truncate(3);
        arc[1].f1_hz = None;
        arc[1].gap_time_s = Some(1_000.0);
        arc[2].gap_time_s = Some(30.0);

        let actual =
            detect_cycle_slips(&arc, CycleSlipOptions::default()).expect("valid cycle-slip arc");
        assert!(actual[1].skipped);
        assert!(!actual[2].reasons.contains(&SlipReason::DataGap));

        arc[2].gap_time_s = Some(301.0);
        let actual =
            detect_cycle_slips(&arc, CycleSlipOptions::default()).expect("valid cycle-slip arc");
        assert!(actual[2].reasons.contains(&SlipReason::DataGap));
    }

    #[test]
    fn unusable_dual_frequency_row_does_not_hide_later_data_gap() {
        let mut arc = oracle_arc();
        arc.truncate(3);
        arc[0].gap_time_s = Some(0.0);
        arc[1].phi1_cycles = None;
        arc[1].phi2_cycles = None;
        arc[1].p1_m = None;
        arc[1].p2_m = None;
        arc[1].gap_time_s = Some(100.0);
        arc[2].gap_time_s = Some(350.0);

        let actual =
            detect_cycle_slips(&arc, CycleSlipOptions::default()).expect("valid cycle-slip arc");
        assert!(!actual[1].skipped);
        assert_eq!(actual[1].gf_m, None);
        assert_eq!(actual[1].mw_m, None);
        assert!(actual[2].reasons.contains(&SlipReason::DataGap));
    }

    #[test]
    fn hatch_smoothing_matches_python_oracle_bits() {
        let actual =
            smooth_code(&oracle_arc(), CycleSlipOptions::default(), 100).expect("valid smoothing");
        let expected = [
            (Some(0x4174689c023d70a4), 1, false),
            (Some(0x417468a1f6000000), 2, false),
            (Some(0x417468a7f7a06d39), 3, false),
            (Some(0x417468ae02b851eb), 4, false),
            (Some(0x417468b41a45a1cb), 1, true),
            (Some(0x417468ba3aac0831), 2, false),
            (Some(0x417468c06ad91687), 1, true),
            (Some(0x417468c6a20c49ba), 2, false),
            (None, 0, false),
        ];

        for (got, (p_smooth, window, reset)) in actual.iter().zip(expected) {
            assert_eq!(b(got.p_smooth_m), p_smooth);
            assert_eq!(got.window, window);
            assert_eq!(got.reset, reset);
        }
    }

    #[test]
    fn hatch_smoothing_accepts_l1_only_arc() {
        let arc: Vec<_> = [(0.0, 10.0, 100.0), (30.0, 12.0, 101.0), (60.0, 15.0, 103.0)]
            .into_iter()
            .map(|(gap_time_s, p1_m, phi1_cycles)| ArcEpoch {
                phi1_cycles: Some(phi1_cycles),
                phi2_cycles: None,
                p1_m: Some(p1_m),
                p2_m: None,
                lli1: Some(0),
                lli2: None,
                f1_hz: Some(C_M_S),
                f2_hz: None,
                gap_time_s: Some(gap_time_s),
            })
            .collect();

        let actual =
            smooth_code(&arc, CycleSlipOptions::default(), 100).expect("valid L1-only smoothing");
        let expected = [
            (Some(10.0), 1, false),
            (Some(11.5), 2, false),
            (Some(14.0), 3, false),
        ];

        for (got, (p_smooth_m, window, reset)) in actual.iter().zip(expected) {
            assert_eq!(got.p_smooth_m, p_smooth_m);
            assert_eq!(got.window, window);
            assert_eq!(got.reset, reset);
        }
    }

    #[test]
    fn hatch_smoothing_keeps_band1_state_across_sparse_l2_arc() {
        let f1 = C_M_S;
        let f2 = C_M_S / 2.0;
        let rows = [
            (0.0, 10.0, 100.0, Some(10.0), Some(50.0), Some(f2)),
            (30.0, 12.0, 101.0, None, None, None),
            (60.0, 15.0, 103.0, Some(15.0), Some(51.5), Some(f2)),
        ];
        let arc: Vec<_> = rows
            .into_iter()
            .map(
                |(gap_time_s, p1_m, phi1_cycles, p2_m, phi2_cycles, f2_hz)| ArcEpoch {
                    phi1_cycles: Some(phi1_cycles),
                    phi2_cycles,
                    p1_m: Some(p1_m),
                    p2_m,
                    lli1: Some(0),
                    lli2: Some(0),
                    f1_hz: Some(f1),
                    f2_hz,
                    gap_time_s: Some(gap_time_s),
                },
            )
            .collect();

        let actual =
            smooth_code(&arc, CycleSlipOptions::default(), 100).expect("valid mixed smoothing");
        let expected = [
            (Some(10.0), 1, false),
            (Some(11.5), 2, false),
            (Some(14.0), 3, false),
        ];

        for (got, (p_smooth_m, window, reset)) in actual.iter().zip(expected) {
            assert_eq!(got.p_smooth_m, p_smooth_m);
            assert_eq!(got.window, window);
            assert_eq!(got.reset, reset);
        }
    }

    #[test]
    fn hatch_window_cap_zero_clamps_to_minimum() {
        let arc = oracle_arc();

        let single_zero =
            smooth_code(&arc, CycleSlipOptions::default(), 0).expect("valid smoothing");
        let single_one =
            smooth_code(&arc, CycleSlipOptions::default(), 1).expect("valid smoothing");
        assert_eq!(single_zero, single_one);
        for result in &single_zero {
            if let Some(p_smooth_m) = result.p_smooth_m {
                assert!(p_smooth_m.is_finite());
            }
            assert!(result.window <= MIN_HATCH_WINDOW_CAP);
        }

        let if_zero =
            smooth_iono_free_code(&arc, CycleSlipOptions::default(), 0).expect("valid smoothing");
        let if_one =
            smooth_iono_free_code(&arc, CycleSlipOptions::default(), 1).expect("valid smoothing");
        assert_eq!(if_zero, if_one);
        for result in &if_zero {
            for value in [result.p_smooth_m, result.p_if_m, result.l_if_m]
                .into_iter()
                .flatten()
            {
                assert!(value.is_finite());
            }
            assert!(result.window <= MIN_HATCH_WINDOW_CAP);
        }
    }

    #[test]
    fn ionosphere_free_hatch_smoothing_matches_python_oracle_bits() {
        let actual = smooth_iono_free_code(&oracle_arc(), CycleSlipOptions::default(), 100)
            .expect("valid smoothing");
        let expected = [
            (
                Some(0x4174689c0a4cab98),
                Some(0x4174689c0a4cab98),
                Some(0x41746197d93b3cb8),
                1,
                false,
            ),
            (
                Some(0x417468a1f8be0026),
                Some(0x417468a1f195bb1b),
                Some(0x4174619dced4d652),
                2,
                false,
            ),
            (
                Some(0x417468a7fbcfc0cc),
                Some(0x417468a80059a882),
                Some(0x417461a3cfa1a31e),
                3,
                false,
            ),
            (
                Some(0x417468ae0479118a),
                Some(0x417468adfa7503c6),
                Some(0x417461a9dba1a31e),
                4,
                false,
            ),
            (
                Some(0x417468b421b7b0f7),
                Some(0x417468b421b7b0f7),
                Some(0x417461b021565566),
                1,
                true,
            ),
            (
                Some(0x417468ba3c0eec2c),
                Some(0x417468ba33ffc0f9),
                Some(0x417461b643bcbbce),
                2,
                false,
            ),
            (
                Some(0x417468c0711ef759),
                Some(0x417468c0711ef759),
                Some(0x417461bc71565568),
                1,
                true,
            ),
            (
                Some(0x417468c6a35fe7ef),
                Some(0x417468c69cd40bb8),
                Some(0x417461c2aa232235),
                2,
                false,
            ),
            (None, None, None, 0, false),
        ];

        for (got, (p_smooth, p_if, l_if, window, reset)) in actual.iter().zip(expected) {
            assert_eq!(b(got.p_smooth_m), p_smooth);
            assert_eq!(b(got.p_if_m), p_if);
            assert_eq!(b(got.l_if_m), l_if);
            assert_eq!(got.window, window);
            assert_eq!(got.reset, reset);
        }
    }
}
