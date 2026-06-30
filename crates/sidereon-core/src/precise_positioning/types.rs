//! Public ABI data types for static multi-epoch PPP positioning.
//!
//! These are the language-independent input, configuration, and result structs
//! shared by the float ([`float`](super::float)) and fixed
//! ([`fixed`](super::fixed)) solve clusters and re-exported from the parent
//! module. They hold no orchestration logic; only the pure conversions tied to
//! a single type live here.

use std::collections::BTreeMap;

use crate::astro::math::interp::lerp;
use crate::ils::IlsError;
use crate::ppp_corrections::{CivilDateTime, PppCorrections, PppCorrectionsOptions};
use crate::tropo::Met;
use crate::GnssSatelliteId;

/// One ionosphere-free code/phase observation in a static PPP epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatObservation {
    /// Physical satellite used for ephemeris lookup.
    pub sat: GnssSatelliteId,
    /// Public satellite token, e.g. `"G07"`.
    pub satellite_id: String,
    /// Ambiguity state key. Split arcs use ids like `"G07#2"`.
    pub ambiguity_id: String,
    pub code_m: f64,
    pub phase_m: f64,
    /// Optional raw carrier frequencies, used by phase wind-up precompute when
    /// no explicit satellite ANTEX frequency pair is configured.
    pub freq1_hz: f64,
    pub freq2_hz: f64,
}

/// One static PPP epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatEpoch {
    pub epoch: CivilDateTime,
    pub jd_whole: f64,
    pub jd_fraction: f64,
    pub t_rx_j2000_s: f64,
    pub observations: Vec<FloatObservation>,
}

/// Initial static-arc state.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatState {
    pub position_m: [f64; 3],
    pub clocks_m: Vec<f64>,
    pub ambiguities_m: BTreeMap<String, f64>,
    pub ztd_m: f64,
}

/// Measurement weighting options. Values are inverse sigmas, matching Sidereon'
/// historical row scaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasurementWeights {
    pub code: f64,
    pub phase: f64,
    pub elevation_weighting: bool,
}

/// Iteration and convergence controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatSolveOptions {
    pub max_iterations: usize,
    pub position_tolerance_m: f64,
    pub clock_tolerance_m: f64,
    pub ambiguity_tolerance_m: f64,
    pub ztd_tolerance_m: f64,
}

impl Default for FloatSolveOptions {
    /// Canonical static-PPP iteration/convergence controls, read from
    /// [`super::defaults`]. This is the single source of truth bindings
    /// construct from instead of hardcoding literals; it does not change any
    /// solve, which still reads the caller's options.
    fn default() -> Self {
        Self {
            max_iterations: super::defaults::MAX_ITERATIONS,
            position_tolerance_m: super::defaults::POSITION_TOLERANCE_M,
            clock_tolerance_m: super::defaults::CLOCK_TOLERANCE_M,
            ambiguity_tolerance_m: super::defaults::AMBIGUITY_TOLERANCE_M,
            ztd_tolerance_m: super::defaults::ZTD_TOLERANCE_M,
        }
    }
}

/// Troposphere controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TroposphereOptions {
    pub enabled: bool,
    pub estimate_ztd: bool,
    pub met: Met,
    /// Mapping function applied to the zenith delays and the estimated ZTD.
    pub mapping: TropoMapping,
}

impl TroposphereOptions {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            estimate_ztd: false,
            met: Met::new_unchecked(1013.25, 288.15, 0.5),
            mapping: TropoMapping::Niell,
        }
    }
}

/// Tropospheric mapping-function selection for a PPP solve.
///
/// `Niell` uses the climatological Niell (1996) mapping with no external data.
/// `Vmf1` uses the Vienna Mapping Function 1 driven by a site-wise `a`
/// coefficient series ([`VmfSiteSeries`]) interpolated to each epoch; the
/// Saastamoinen zenith delays are unchanged, only the mapping differs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TropoMapping {
    /// Niell (1996) mapping functions.
    Niell,
    /// VMF1 site-wise mapping from a 6-hourly `a`-coefficient series.
    Vmf1(VmfSiteSeries),
}

/// Maximum number of 6-hourly VMF site samples carried for one arc.
///
/// VMF data products provide `a` coefficients at 00/06/12/18 UT; one day plus
/// the next 00 UT node (for interpolation across midnight) is five samples, so
/// eight is comfortable headroom while keeping [`VmfSiteSeries`] `Copy`.
pub const VMF_SITE_MAX_SAMPLES: usize = 8;

/// Clamp window (days) on each side of a single-sample VMF series.
///
/// With one sample there is no interval to size the allowed clamp from, so this
/// fixed window is used by [`VmfSiteSeries::interpolate_checked`]. One VMF
/// sampling step (6 h = 0.25 day) past the lone node is treated as covered;
/// beyond it, the epoch is out of VMF coverage.
pub const VMF_SITE_SINGLE_SAMPLE_CLAMP_DAYS: f64 = 0.25;

/// One VMF site-wise sample: the `a` coefficients at a single epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VmfSiteSample {
    /// Modified Julian date of the sample (VMF nodes are 00/06/12/18 UT).
    pub mjd: f64,
    /// Hydrostatic `a` coefficient from the VMF data product.
    pub ah: f64,
    /// Wet `a` coefficient from the VMF data product.
    pub aw: f64,
}

/// A short, strictly ascending VMF site-wise `a`-coefficient series for one
/// station, linearly interpolated to the observation epoch.
///
/// Fixed-capacity ([`VMF_SITE_MAX_SAMPLES`]) so the enclosing
/// [`TroposphereOptions`] stays `Copy`. Interpolation clamps to the endpoints
/// outside the sample span (no extrapolation of the slope).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VmfSiteSeries {
    samples: [VmfSiteSample; VMF_SITE_MAX_SAMPLES],
    len: usize,
}

impl VmfSiteSeries {
    /// Build a series from ascending samples (1..=[`VMF_SITE_MAX_SAMPLES`]).
    ///
    /// Errors if empty, over capacity, not strictly increasing in MJD, or if any
    /// `a` coefficient is non-finite or non-positive.
    pub fn new(samples: &[VmfSiteSample]) -> Result<Self, crate::error::Error> {
        use crate::error::Error;
        if samples.is_empty() {
            return Err(Error::InvalidInput("vmf site series empty".to_string()));
        }
        if samples.len() > VMF_SITE_MAX_SAMPLES {
            return Err(Error::InvalidInput(format!(
                "vmf site series length {} exceeds {VMF_SITE_MAX_SAMPLES}",
                samples.len()
            )));
        }
        for (idx, s) in samples.iter().enumerate() {
            if !s.mjd.is_finite() || !s.ah.is_finite() || !s.aw.is_finite() {
                return Err(Error::InvalidInput(
                    "vmf site sample not finite".to_string(),
                ));
            }
            if s.ah <= 0.0 || s.aw <= 0.0 {
                return Err(Error::InvalidInput(
                    "vmf site sample a-coefficient not positive".to_string(),
                ));
            }
            if idx > 0 && s.mjd <= samples[idx - 1].mjd {
                return Err(Error::InvalidInput(
                    "vmf site series mjd not strictly increasing".to_string(),
                ));
            }
        }
        let mut buf = [VmfSiteSample {
            mjd: 0.0,
            ah: 0.0,
            aw: 0.0,
        }; VMF_SITE_MAX_SAMPLES];
        buf[..samples.len()].copy_from_slice(samples);
        Ok(Self {
            samples: buf,
            len: samples.len(),
        })
    }

    /// Hydrostatic and wet `a` coefficients interpolated to `mjd`, or `None` when
    /// `mjd` lies more than one sampling step beyond either endpoint.
    ///
    /// Within the span this interpolates; just past an endpoint - up to the
    /// adjacent sampling interval, e.g. the final 6 h block after the last VMF
    /// node - it clamps to the endpoint value, matching [`Self::interpolate`].
    /// Beyond that it returns `None` instead of silently reusing a stale endpoint
    /// coefficient for an epoch hours or days outside the product; the caller must
    /// treat that as missing VMF coverage rather than extrapolate indefinitely.
    /// For a single-sample series (no interval to size the window) the clamp
    /// window is [`VMF_SITE_SINGLE_SAMPLE_CLAMP_DAYS`] on each side.
    pub(crate) fn interpolate_checked(&self, mjd: f64) -> Option<(f64, f64)> {
        let s = &self.samples[..self.len];
        let first = s[0];
        let last = s[self.len - 1];
        let lead = if self.len >= 2 {
            s[1].mjd - first.mjd
        } else {
            VMF_SITE_SINGLE_SAMPLE_CLAMP_DAYS
        };
        let trail = if self.len >= 2 {
            last.mjd - s[self.len - 2].mjd
        } else {
            VMF_SITE_SINGLE_SAMPLE_CLAMP_DAYS
        };
        if mjd < first.mjd - lead || mjd > last.mjd + trail {
            return None;
        }
        Some(self.interpolate(mjd))
    }

    /// Hydrostatic and wet `a` coefficients linearly interpolated to `mjd`,
    /// clamped to the endpoint values outside the sample span.
    ///
    /// This clamps unboundedly; prefer [`Self::interpolate_checked`] on the solve
    /// path so an epoch far outside the product is flagged rather than served a
    /// stale endpoint coefficient.
    pub(crate) fn interpolate(&self, mjd: f64) -> (f64, f64) {
        let s = &self.samples[..self.len];
        let first = s[0];
        if mjd <= first.mjd {
            return (first.ah, first.aw);
        }
        let last = s[self.len - 1];
        if mjd >= last.mjd {
            return (last.ah, last.aw);
        }
        for win in s.windows(2) {
            let (lo, hi) = (win[0], win[1]);
            if mjd <= hi.mjd {
                let f = (mjd - lo.mjd) / (hi.mjd - lo.mjd);
                return (lerp(lo.ah, hi.ah, f), lerp(lo.aw, hi.aw, f));
            }
        }
        (last.ah, last.aw)
    }
}

/// One ANTEX PCV sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcvSample {
    pub azimuth_deg: Option<f64>,
    pub zenith_deg: f64,
    pub value_m: f64,
}

/// Receiver antenna calibration at one frequency.
#[derive(Debug, Clone, PartialEq)]
pub struct ReceiverAntennaFrequency {
    pub label: String,
    pub pco_m: [f64; 3],
    pub pcv_samples: Vec<PcvSample>,
}

/// Receiver antenna correction options.
#[derive(Debug, Clone, PartialEq)]
pub struct ReceiverAntennaOptions {
    pub freq1_label: String,
    pub freq1_hz: f64,
    pub freq2_label: String,
    pub freq2_hz: f64,
    pub frequencies: Vec<ReceiverAntennaFrequency>,
}

/// Fine satellite clock series, keyed by GPS seconds.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SatelliteClockCorrections {
    pub series: BTreeMap<GnssSatelliteId, Vec<(f64, f64)>>,
}

/// Range-correction options and precomputed correction tables.
///
/// Disabled corrections must be selected explicitly; `Default` is intentionally
/// unavailable.
///
/// ```compile_fail
/// use sidereon_core::precise_positioning::RangeCorrections;
///
/// let _ = RangeCorrections::default();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RangeCorrections {
    pub receiver_antenna: Option<ReceiverAntennaOptions>,
    pub sat_clock_relativity: bool,
    pub satellite_clock: Option<SatelliteClockCorrections>,
    pub ppp: PppCorrectionLookup,
}

impl RangeCorrections {
    /// Create an explicit all-off correction set.
    pub fn disabled() -> Self {
        Self {
            receiver_antenna: None,
            sat_clock_relativity: false,
            satellite_clock: None,
            ppp: PppCorrectionLookup::default(),
        }
    }
}

/// Static float solve controls.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatSolveConfig {
    pub weights: MeasurementWeights,
    pub tropo: TroposphereOptions,
    pub corrections: RangeCorrections,
    pub opts: FloatSolveOptions,
    pub residual_screen: bool,
}

/// Indexed static PPP correction lookup tables.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PppCorrectionLookup {
    pub tide: BTreeMap<usize, [f64; 3]>,
    pub pole_tide: BTreeMap<usize, [f64; 3]>,
    pub ocean_loading: BTreeMap<usize, [f64; 3]>,
    pub windup_m: BTreeMap<(GnssSatelliteId, usize), f64>,
    pub sat_pco_ecef: BTreeMap<(GnssSatelliteId, usize), [f64; 3]>,
    pub sat_pcv_m: BTreeMap<(GnssSatelliteId, usize), f64>,
    pub tide_enabled: bool,
    pub pole_tide_enabled: bool,
    pub ocean_loading_enabled: bool,
    pub windup_enabled: bool,
    pub satellite_antenna_enabled: bool,
}

impl PppCorrectionLookup {
    pub fn from_options(value: PppCorrections, options: &PppCorrectionsOptions) -> Self {
        Self::from_parts(
            value,
            options.solid_earth_tide,
            options.pole_tide.is_some(),
            options.ocean_loading.is_some(),
            options.phase_windup,
            options.satellite_antenna.is_some(),
        )
    }

    fn from_parts(
        value: PppCorrections,
        tide_enabled: bool,
        pole_tide_enabled: bool,
        ocean_loading_enabled: bool,
        windup_enabled: bool,
        satellite_antenna_enabled: bool,
    ) -> Self {
        Self {
            tide: value
                .tide
                .into_iter()
                .map(|c| (c.epoch_index, c.vector_m))
                .collect(),
            pole_tide: value
                .pole_tide
                .into_iter()
                .map(|c| (c.epoch_index, c.vector_m))
                .collect(),
            ocean_loading: value
                .ocean_loading
                .into_iter()
                .map(|c| (c.epoch_index, c.vector_m))
                .collect(),
            windup_m: value
                .windup_m
                .into_iter()
                .map(|c| ((c.sat, c.epoch_index), c.value_m))
                .collect(),
            sat_pco_ecef: value
                .sat_pco_ecef
                .into_iter()
                .map(|c| ((c.sat, c.epoch_index), c.vector_m))
                .collect(),
            sat_pcv_m: value
                .sat_pcv_m
                .into_iter()
                .map(|c| ((c.sat, c.epoch_index), c.value_m))
                .collect(),
            tide_enabled,
            pole_tide_enabled,
            ocean_loading_enabled,
            windup_enabled,
            satellite_antenna_enabled,
        }
    }
}

impl From<PppCorrections> for PppCorrectionLookup {
    fn from(value: PppCorrections) -> Self {
        let tide_enabled = !value.tide.is_empty();
        let pole_tide_enabled = !value.pole_tide.is_empty();
        let ocean_loading_enabled = !value.ocean_loading.is_empty();
        let windup_enabled = !value.windup_m.is_empty();
        let satellite_antenna_enabled =
            !value.sat_pco_ecef.is_empty() || !value.sat_pcv_m.is_empty();
        Self::from_parts(
            value,
            tide_enabled,
            pole_tide_enabled,
            ocean_loading_enabled,
            windup_enabled,
            satellite_antenna_enabled,
        )
    }
}

/// Per-satellite residual row in the returned public solution.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatResidual {
    pub epoch_index: usize,
    pub satellite_id: String,
    pub code_m: f64,
    pub phase_m: f64,
    pub code_weight: f64,
    pub phase_weight: f64,
}

/// Static float solution.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatSolution {
    pub position_m: [f64; 3],
    pub epoch_clocks_m: Vec<f64>,
    pub ambiguities_m: BTreeMap<String, f64>,
    pub ztd_residual_m: Option<f64>,
    pub residuals_m: Vec<FloatResidual>,
    pub used_sats: Vec<String>,
    pub iterations: usize,
    pub converged: bool,
    pub status: FloatStatus,
    pub code_rms_m: f64,
    pub phase_rms_m: f64,
    pub weighted_rms_m: f64,
}

/// Static PPP solve termination status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatStatus {
    StateTolerance,
    MaxIterations,
}

/// Static PPP solve errors.
#[derive(Debug, Clone, PartialEq)]
pub enum FloatSolveError {
    NoEphemeris {
        satellite_id: String,
        reason: NoEphemerisReason,
    },
    SingularGeometry,
    InvalidClockCount {
        expected: usize,
        actual: usize,
    },
    InvalidSolveOption {
        field: &'static str,
        reason: &'static str,
    },
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    MissingAmbiguity(String),
    MissingCorrection {
        satellite_id: String,
        correction: MissingCorrection,
    },
}

impl core::fmt::Display for FloatSolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoEphemeris {
                satellite_id,
                reason,
            } => write!(
                f,
                "missing PPP ephemeris for satellite {satellite_id}: {reason}"
            ),
            Self::SingularGeometry => write!(f, "PPP float geometry is singular"),
            Self::InvalidClockCount { expected, actual } => write!(
                f,
                "invalid PPP clock vector length: expected {expected}, got {actual}"
            ),
            Self::InvalidSolveOption { field, reason } => {
                write!(f, "invalid PPP solve option {field}: {reason}")
            }
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid PPP input {field}: {reason}")
            }
            Self::MissingAmbiguity(id) => write!(f, "missing PPP ambiguity {id}"),
            Self::MissingCorrection {
                satellite_id,
                correction,
            } => write!(
                f,
                "missing PPP correction for satellite {satellite_id}: {correction}"
            ),
        }
    }
}

impl std::error::Error for FloatSolveError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissingCorrection {
    SolidEarthTide,
    PoleTide,
    OceanLoading,
    PhaseWindup,
    SatelliteAntennaPco,
    SatelliteAntennaPcv,
    ReceiverAntennaFrequency(String),
    ReceiverAntennaPcv(String),
    ReceiverAntennaGeometry,
}

impl core::fmt::Display for MissingCorrection {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SolidEarthTide => write!(f, "solid Earth tide correction"),
            Self::PoleTide => write!(f, "solid Earth pole tide correction"),
            Self::OceanLoading => write!(f, "ocean tide loading correction"),
            Self::PhaseWindup => write!(f, "phase wind-up correction"),
            Self::SatelliteAntennaPco => write!(f, "satellite antenna PCO"),
            Self::SatelliteAntennaPcv => write!(f, "satellite antenna PCV"),
            Self::ReceiverAntennaFrequency(label) => {
                write!(f, "receiver antenna frequency {label}")
            }
            Self::ReceiverAntennaPcv(label) => write!(f, "receiver antenna PCV {label}"),
            Self::ReceiverAntennaGeometry => write!(f, "receiver antenna geometry"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NoEphemerisReason {
    NoEphemeris,
    MissingSatelliteClock,
    Reason(String),
}

impl core::fmt::Display for NoEphemerisReason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoEphemeris => write!(f, "no ephemeris product covers the epoch"),
            Self::MissingSatelliteClock => write!(f, "satellite clock is unavailable"),
            Self::Reason(reason) => write!(f, "{reason}"),
        }
    }
}

/// Integer ambiguity resolution controls for fixed PPP.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedAmbiguityOptions {
    pub wavelengths_m: BTreeMap<String, f64>,
    pub offsets_m: BTreeMap<String, f64>,
    pub ratio_threshold: f64,
}

/// Static fixed-ambiguity PPP solve controls.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedSolveConfig {
    pub weights: MeasurementWeights,
    pub tropo: TroposphereOptions,
    pub corrections: RangeCorrections,
    pub opts: FloatSolveOptions,
    pub ambiguity: FixedAmbiguityOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerStatus {
    Fixed,
    NotFixed,
}

/// Frozen ambiguity-search metadata returned with a fixed PPP solution.
#[derive(Debug, Clone, PartialEq)]
pub struct AmbiguitySearch {
    pub order: Vec<String>,
    pub float_cycles: BTreeMap<String, f64>,
    pub covariance_cycles: Vec<Vec<f64>>,
    pub covariance_inverse_cycles: Vec<Vec<f64>>,
}

/// Integer-search summary returned with a fixed PPP solution.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedIntegerMetadata {
    pub integer_status: IntegerStatus,
    pub integer_ratio: f64,
    pub integer_best_score: f64,
    pub integer_second_best_score: Option<f64>,
    pub integer_candidates: usize,
    pub ambiguity_search: AmbiguitySearch,
}

/// Static integer-fixed PPP solution.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedSolution {
    pub position_m: [f64; 3],
    pub epoch_clocks_m: Vec<f64>,
    pub fixed_ambiguities_cycles: BTreeMap<String, i64>,
    pub fixed_ambiguities_m: BTreeMap<String, f64>,
    pub ztd_residual_m: Option<f64>,
    pub float_solution: FloatSolution,
    pub residuals_m: Vec<FloatResidual>,
    pub used_sats: Vec<String>,
    pub iterations: usize,
    pub converged: bool,
    pub status: FloatStatus,
    pub code_rms_m: f64,
    pub phase_rms_m: f64,
    pub weighted_rms_m: f64,
    pub integer: FixedIntegerMetadata,
}

/// Static fixed PPP solve errors.
#[derive(Debug, Clone, PartialEq)]
pub enum FixedSolveError {
    Float(FloatSolveError),
    Integer(IlsError),
    MissingWavelength(String),
    MissingOffset(String),
    MissingFixedAmbiguity(String),
}

impl core::fmt::Display for FixedSolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Float(error) => write!(f, "PPP float prerequisite failed: {error}"),
            Self::Integer(error) => write!(f, "PPP integer ambiguity search failed: {error}"),
            Self::MissingWavelength(id) => write!(f, "missing PPP wavelength for ambiguity {id}"),
            Self::MissingOffset(id) => write!(f, "missing PPP offset for ambiguity {id}"),
            Self::MissingFixedAmbiguity(id) => {
                write!(f, "missing fixed PPP ambiguity {id}")
            }
        }
    }
}

impl std::error::Error for FixedSolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Float(error) => Some(error),
            Self::Integer(error) => Some(error),
            _ => None,
        }
    }
}
