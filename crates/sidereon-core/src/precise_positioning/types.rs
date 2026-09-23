//! Public ABI data types for static multi-epoch PPP positioning.
//!
//! These are the language-independent input, configuration, and result structs
//! shared by the float ([`float`](super::float)) and fixed
//! ([`fixed`](super::fixed)) solve clusters and re-exported from the parent
//! module. They hold no orchestration logic; only the pure conversions tied to
//! a single type live here.

use std::collections::BTreeMap;

use crate::astro::math::interp::lerp;
use crate::dop::PositionCovariance;
use crate::ils::IlsError;
use crate::ppp_corrections::{CivilDateTime, PppCorrections, PppCorrectionsOptions};
use crate::ssr::{
    PhaseContinuityToken, SsrBiasStatus, SsrCodeBiasQueryResult, SsrCorrectedEphemeris,
    SsrPhaseBiasQueryResult, SsrSolution,
};
use crate::tropo::Met;
use crate::{GnssSatelliteId, GnssSystem};

/// One ionosphere-free code/phase observation in a static PPP epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatObservation {
    /// Physical satellite used for ephemeris lookup.
    pub sat: GnssSatelliteId,
    /// Public satellite token, e.g. `"G07"`.
    pub satellite_id: String,
    /// Ambiguity state key. Split arcs use ids like `"G07#2"`.
    pub ambiguity_id: String,
    /// Ionosphere-free code measurement in meters; the row builder subtracts
    /// the modeled code range from it for the code prefit residual.
    pub code_m: f64,
    /// Ionosphere-free phase measurement in meters; the row builder applies
    /// phase bias and wind-up corrections before subtracting the modeled range
    /// and ambiguity.
    pub phase_m: f64,
    /// Optional raw carrier frequencies, used by phase wind-up precompute when
    /// no explicit satellite ANTEX frequency pair is configured.
    pub freq1_hz: f64,
    /// Second raw carrier frequency passed to correction precomputation for
    /// phase-windup and code-bias ionosphere-free combinations.
    pub freq2_hz: f64,
    /// GLONASS FDMA frequency-channel number for this satellite, when known.
    pub glonass_channel: Option<i8>,
}

/// One static PPP epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatEpoch {
    /// Civil reception time copied into correction precomputation for
    /// epoch-dependent tide, antenna-validity, and bias evaluation.
    pub epoch: CivilDateTime,
    /// Whole part of the split Julian date used by the troposphere model to
    /// construct its validated instant and VMF site-series MJD.
    pub jd_whole: f64,
    /// Fractional part of the split Julian date paired with [`Self::jd_whole`]
    /// for validated time conversion and VMF site-series lookup.
    pub jd_fraction: f64,
    /// Receiver epoch in seconds from J2000, used for observable prediction,
    /// correction precomputation, and regular-spacing detection.
    pub t_rx_j2000_s: f64,
    /// Observations traversed in input order to build code/phase rows; an
    /// elevation cutoff replaces this vector with only the retained rows.
    pub observations: Vec<FloatObservation>,
}

/// Initial static-arc state.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatState {
    /// Receiver seed and updated position in ECEF meters, passed to satellite
    /// prediction and covariance extraction.
    pub position_m: [f64; 3],
    /// One receiver-clock state in meters per epoch; the solve boundary requires
    /// the vector length to equal the number of epochs.
    pub clocks_m: Vec<f64>,
    /// Float ambiguity estimates in meters keyed by ambiguity id; the row
    /// builder uses them for the estimated ambiguity columns.
    pub ambiguities_m: BTreeMap<String, f64>,
    /// Residual zenith total delay state in meters, applied through the wet
    /// mapping only when ZTD estimation is enabled.
    pub ztd_m: f64,
    /// North horizontal troposphere gradient state, in metres.
    pub tropo_gradient_north_m: f64,
    /// East horizontal troposphere gradient state, in metres.
    pub tropo_gradient_east_m: f64,
    /// Optional post-combination residual ionosphere states, keyed by ambiguity
    /// arc and expressed in metres on the ionosphere-free observable.
    pub residual_ionosphere_m: BTreeMap<String, f64>,
}

/// Measurement weighting options. Values are inverse sigmas, matching Sidereon'
/// historical row scaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasurementWeights {
    /// Inverse-sigma multiplier for code rows; it must be finite and positive.
    pub code: f64,
    /// Inverse-sigma multiplier for phase rows; it must be finite and positive.
    pub phase: f64,
    /// If true, multiplies each base weight by the sine-of-elevation scale,
    /// floored at `1e-3` for non-finite or very low elevations.
    pub elevation_weighting: bool,
}

/// Iteration and convergence controls.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct FloatSolveOptions {
    /// Inclusive iteration cap; zero and values above the 10,000 PPP cap are
    /// rejected before the solve loop starts.
    pub max_iterations: usize,
    /// Maximum allowed norm of the three-component position update in meters
    /// for a state-tolerance result.
    pub position_tolerance_m: f64,
    /// Maximum absolute receiver-clock update in meters required for a
    /// state-tolerance result.
    pub clock_tolerance_m: f64,
    /// Maximum absolute estimated-ambiguity update in meters required for a
    /// float state-tolerance result.
    pub ambiguity_tolerance_m: f64,
    /// Maximum absolute ZTD update in meters; the fixed path also uses this
    /// tolerance for each horizontal-gradient update component.
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
#[non_exhaustive]
pub struct TroposphereOptions {
    /// Enables troposphere modeling; when false, the model returns zero slant,
    /// ZTD-mapping, and gradient-mapping terms and skips meteorological checks.
    pub enabled: bool,
    /// Adds one ZTD normal-equation state only when [`Self::enabled`] is also
    /// true; otherwise the solution reports no ZTD residual.
    pub estimate_ztd: bool,
    /// Estimate constant north/east horizontal tropospheric gradients over a
    /// static PPP arc. This can help at sites with strong horizontal delay
    /// gradients, but is off by default because it adds two solve states.
    pub estimate_tropo_gradients: bool,
    /// Meteorological input passed to the Saastamoinen slant-delay calculation
    /// when enabled; pressure and temperature must be positive and humidity a
    /// fraction.
    pub met: Met,
    /// Mapping function applied to the zenith delays and the estimated ZTD.
    pub mapping: TropoMapping,
}

impl TroposphereOptions {
    /// Build disabled troposphere controls using the supplied meteorology.
    ///
    /// Estimation is disabled and the Niell mapping is selected; assign the
    /// boolean fields or mapping when a different model is required.
    #[must_use]
    pub const fn new(met: Met) -> Self {
        Self {
            enabled: false,
            estimate_ztd: false,
            estimate_tropo_gradients: false,
            met,
            mapping: TropoMapping::Niell,
        }
    }

    /// Return the explicit all-off troposphere configuration.
    ///
    /// The returned meteorological values are standard-atmosphere placeholders;
    /// the disabled model path does not consume them.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            estimate_ztd: false,
            estimate_tropo_gradients: false,
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
/// fixed window is used by `VmfSiteSeries::interpolate_checked`. One VMF
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
    /// `None` marks a no-azimuth sample; `Some` places the sample in an azimuth
    /// bracket for circular interpolation.
    pub azimuth_deg: Option<f64>,
    /// Zenith angle in degrees, validated in the inclusive range 0..=180
    /// before sorted zenith interpolation.
    pub zenith_deg: f64,
    /// Receiver PCV correction in meters carried into the interpolated value
    /// added to the PCO projection.
    pub value_m: f64,
}

/// Receiver antenna calibration at one frequency.
#[derive(Debug, Clone, PartialEq)]
pub struct ReceiverAntennaFrequency {
    /// Frequency label searched by the configured signal selectors.
    pub label: String,
    /// Receiver phase-center offset in local north/east/up meters, projected
    /// onto the predicted line of sight before the two-frequency combination.
    pub pco_m: [f64; 3],
    /// ANTEX PCV samples scanned into no-azimuth and azimuth-indexed grids; an
    /// empty or unusable grid produces a missing receiver PCV correction.
    pub pcv_samples: Vec<PcvSample>,
}

/// Receiver antenna correction options.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ReceiverAntennaOptions {
    /// Label selecting the first receiver-frequency calibration record.
    pub freq1_label: String,
    /// First carrier frequency in hertz used in the receiver antenna
    /// ionosphere-free combination; it must be finite and positive.
    pub freq1_hz: f64,
    /// Label selecting the second receiver-frequency calibration record.
    pub freq2_label: String,
    /// Second carrier frequency in hertz used with [`Self::freq1_hz`] in the
    /// receiver antenna ionosphere-free combination; it must differ from it.
    pub freq2_hz: f64,
    /// Frequency calibration records searched by label for local PCO and PCV
    /// data for each configured signal.
    pub frequencies: Vec<ReceiverAntennaFrequency>,
}

impl ReceiverAntennaOptions {
    /// Build receiver-antenna options from both carrier definitions and the
    /// supplied frequency calibrations.
    #[must_use]
    pub fn new(
        freq1_label: String,
        freq1_hz: f64,
        freq2_label: String,
        freq2_hz: f64,
        frequencies: Vec<ReceiverAntennaFrequency>,
    ) -> Self {
        Self {
            freq1_label,
            freq1_hz,
            freq2_label,
            freq2_hz,
            frequencies,
        }
    }
}

/// Fine satellite clock series, keyed by GPS seconds.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SatelliteClockCorrections {
    /// Per-satellite `(GPS seconds, clock bias seconds)` records. Each series is
    /// strictly increasing and is interpolated with one-interval endpoint bounds.
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
    /// Optional receiver antenna calibration; when present, both configured
    /// frequency records and usable PCV data are required.
    pub receiver_antenna: Option<ReceiverAntennaOptions>,
    /// Enables the relativistic satellite range term
    /// `2 * dot(position, velocity) / c`; false contributes zero.
    ///
    /// It contributes zero as well where the satellite clock in use already
    /// includes the term: with no external CLK series, a source whose
    /// [`ObservableEphemerisSource::clock_includes_relativity`] is true, such as a
    /// broadcast or SSR-corrected source. An SP3 or precise-interpolant source's clock
    /// leaves the term to the user, and the rows add it; so do they with a CLK series,
    /// which replaces the source's clock.
    ///
    /// [`ObservableEphemerisSource::clock_includes_relativity`]: crate::observables::ObservableEphemerisSource::clock_includes_relativity
    pub sat_clock_relativity: bool,
    /// Optional external CLK series used when an observable prediction has no
    /// satellite clock; a missing series or time gap reports a clock error.
    pub satellite_clock: Option<SatelliteClockCorrections>,
    /// Indexed tide, wind-up, antenna, and bias tables consumed by the row
    /// model; enabled tables must contain each requested key.
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
#[non_exhaustive]
pub struct FloatSolveConfig {
    /// Code/phase inverse-sigma weights copied into the solve model context.
    pub weights: MeasurementWeights,
    /// Troposphere and optional ZTD/gradient controls copied into the solve
    /// model context.
    pub tropo: TroposphereOptions,
    /// Receiver, satellite-clock, tide, antenna, wind-up, and bias data queried
    /// while assembling solve rows.
    pub corrections: RangeCorrections,
    /// Iteration cap and state tolerances used by the float solve loop.
    pub opts: FloatSolveOptions,
    /// Optional PPP observation elevation cutoff in degrees.
    ///
    /// `None` preserves the historical observation set. When set, the static
    /// solver predicts each observation's elevation at the seed receiver
    /// position and removes observations below the cutoff before active
    /// ambiguity ids, residual rows, normal rows, or fixed ambiguity search are
    /// assembled.
    pub elevation_cutoff_deg: Option<f64>,
    /// Enables leave-one-observation-per-satellite residual screening; a
    /// screened solution is accepted only when its weighted RMS is less than
    /// half the unscreened value.
    pub residual_screen: bool,
    /// Estimate one post-combination residual ionosphere state per ambiguity
    /// arc. This is disabled by default in repository configs and is intended
    /// for diagnosing residual dispersive error after ionosphere-free
    /// combination.
    pub estimate_residual_ionosphere: bool,
}

impl FloatSolveConfig {
    /// Build static float controls from all required model and solve settings.
    #[must_use]
    pub const fn new(
        weights: MeasurementWeights,
        tropo: TroposphereOptions,
        corrections: RangeCorrections,
        opts: FloatSolveOptions,
        elevation_cutoff_deg: Option<f64>,
        residual_screen: bool,
        estimate_residual_ionosphere: bool,
    ) -> Self {
        Self {
            weights,
            tropo,
            corrections,
            opts,
            elevation_cutoff_deg,
            residual_screen,
            estimate_residual_ionosphere,
        }
    }
}

/// Indexed static PPP correction lookup tables.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PppCorrectionLookup {
    /// ECEF solid-earth tide vectors keyed by zero-based epoch index and
    /// projected with the negative line-of-sight dot product when enabled.
    pub tide: BTreeMap<usize, [f64; 3]>,
    /// ECEF pole-tide vectors keyed by epoch index and projected with the
    /// negative line-of-sight dot product when enabled.
    pub pole_tide: BTreeMap<usize, [f64; 3]>,
    /// ECEF ocean-loading vectors keyed by epoch index and projected with the
    /// negative line-of-sight dot product when enabled.
    pub ocean_loading: BTreeMap<usize, [f64; 3]>,
    /// Ionosphere-free phase wind-up meters keyed by `(satellite, epoch index)`.
    pub windup_m: BTreeMap<(GnssSatelliteId, usize), f64>,
    /// Ionosphere-free satellite PCO vectors in ECEF meters keyed by satellite
    /// and epoch index, then projected onto the line of sight.
    pub sat_pco_ecef: BTreeMap<(GnssSatelliteId, usize), [f64; 3]>,
    /// Ionosphere-free satellite PCV meters keyed by satellite and epoch index.
    pub sat_pcv_m: BTreeMap<(GnssSatelliteId, usize), f64>,
    /// Clock-datum ionosphere-free code-bias meters keyed by satellite and
    /// epoch index and inserted into the modeled code range.
    pub code_bias_m: BTreeMap<(GnssSatelliteId, usize), f64>,
    /// HAS/SSR ionosphere-free code bias stored with the opposite sign of the
    /// observation-side bias because the row model adds it to modeled code. Keyed by
    /// satellite, epoch index and observation ambiguity state key, since the combination
    /// can use each observation's own carrier frequencies.
    pub ssr_code_bias_m: BTreeMap<(GnssSatelliteId, usize, String), f64>,
    /// HAS/SSR ionosphere-free phase bias stored with observation-side sign and
    /// added directly to the phase residual. Keyed by satellite, epoch index,
    /// and observation ambiguity state key.
    pub phase_bias_m: BTreeMap<(GnssSatelliteId, usize, String), f64>,
    /// The two SSR/HAS bias records each entry of [`Self::ssr_code_bias_m`] was formed
    /// from, under the same key. The row model requires both still to be the records
    /// available at the observation's transmission time, from the solution of the orbit and
    /// clock its ephemeris source applies there. An entry without records, as in a lookup
    /// filled directly, is applied as given.
    pub ssr_code_bias_records: BTreeMap<(GnssSatelliteId, usize, String), [SsrBiasRecord; 2]>,
    /// The two SSR/HAS bias records each entry of [`Self::phase_bias_m`] was formed from,
    /// checked the same way as [`Self::ssr_code_bias_records`].
    pub phase_bias_records: BTreeMap<(GnssSatelliteId, usize, String), [SsrBiasRecord; 2]>,
    /// Selects whether a solid-earth tide entry is required for each epoch.
    pub tide_enabled: bool,
    /// Selects whether a pole-tide entry is required for each epoch.
    pub pole_tide_enabled: bool,
    /// Selects whether an ocean-loading entry is required for each epoch.
    pub ocean_loading_enabled: bool,
    /// Selects whether a phase wind-up entry is required for each satellite and
    /// epoch.
    pub windup_enabled: bool,
    /// Selects whether both satellite PCO and PCV entries are required.
    pub satellite_antenna_enabled: bool,
    /// Selects whether a clock-datum code-bias entry is required.
    pub code_bias_enabled: bool,
    /// Selects whether an SSR/HAS code-bias entry is required.
    pub ssr_code_bias_enabled: bool,
    /// Selects whether an SSR/HAS phase-bias entry is required.
    pub phase_bias_enabled: bool,
    /// SSR/HAS PPP bias application report, populated when [`Self::with_ssr_biases`] is called.
    pub ssr_bias_report: Option<SsrPppBiasApplicationReport>,
}

/// SSR/HAS signal ids used to apply parsed per-signal biases to an IF PPP row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsrPppBiasSignalPair {
    /// Signal id queried for the first code bias in the correction store.
    pub code1_signal: u8,
    /// Signal id queried for the second code bias in the correction store.
    pub code2_signal: u8,
    /// Signal id queried for the first phase bias in the correction store.
    pub phase1_signal: u8,
    /// Signal id queried for the second phase bias in the correction store.
    pub phase2_signal: u8,
    /// First carrier frequency used for the ionosphere-free bias combination.
    pub freq1_hz: f64,
    /// Second carrier frequency paired with [`Self::freq1_hz`] for the
    /// ionosphere-free bias combination.
    pub freq2_hz: f64,
}

/// Status of an ionosphere-free bias combination for an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SsrIfCombinationStatus {
    /// Both signals were available, compatible, and successfully combined.
    Applied,
    /// Bias application was explicitly opted out by caller options.
    OptedOut,
    /// No signal pair was configured for this satellite / system.
    NoSignalPairConfigured,
    /// One or both requested signals were not available in the store at the transmission
    /// time: missing, transmitted as unavailable, not yet valid or expired. Each signal
    /// report's query result states which.
    SignalUnavailable,
    /// Carrier frequencies are invalid (non-finite, non-positive, or equal).
    InvalidFrequencies,
    /// Provider source or solution differs between the two signals.
    IncompatibleSourceOrSolution,
    /// IOD SSR differs between the two signals.
    IncompatibleIod,
    /// The ephemeris source applies no SSR orbit and clock correction to the satellite at
    /// the transmission time ([`SsrCorrectedEphemeris::applied_orbit_clock_solution`] is
    /// `None`): it declines the satellite or returns the plain broadcast state. Satellite
    /// biases belong to the solution whose clocks they were estimated with, so with no SSR
    /// clock in use there is no solution to apply them against.
    OrbitClockSolutionUnavailable,
    /// Both signals' biases come from a different source, provider or solution than the
    /// orbit and clock corrections the ephemeris source applies to the satellite at the
    /// transmission time. Applying them would mix the bias datum of one SSR solution with
    /// the clocks of another, so they are not applied.
    OrbitClockSolutionMismatch,
    /// An active Galileo HAS do-not-use indication excludes the satellite. Each signal
    /// report's query result carries the indication's reference epoch and validity interval
    /// in its `details`. The ephemeris source refuses an excluded satellite, so its
    /// transmission time cannot be predicted and the queries are made at the reception
    /// time.
    SatelliteExcluded,
    /// The transmission time could not be predicted, because the ephemeris source cannot
    /// place the satellite, for a reason other than a do-not-use exclusion. The signal
    /// reports hold the queries at the reception time.
    TransmitTimeUnavailable,
    /// A caller continuity token supplied for this ambiguity and signal does not continue the
    /// current phase-bias arc, so the caller has to reset that ambiguity.
    PhaseDiscontinuityNeedsReset,
}

/// Detailed bias query result for a single signal in an observation epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct SsrObsSignalReport<Q> {
    /// Zero-based epoch index in the input epoch slice.
    pub epoch_index: usize,
    /// Observation satellite ID.
    pub sat: GnssSatelliteId,
    /// Ambiguity ID from the observation.
    pub ambiguity_id: String,
    /// Requested signal ID.
    pub signal_id: u8,
    /// The full query result returned by the correction store.
    pub query_result: Q,
}

/// PPP bias application report for a single observation at a given epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct SsrObsApplicationReport {
    /// Zero-based epoch index.
    pub epoch_index: usize,
    /// Physical satellite ID.
    pub sat: GnssSatelliteId,
    /// Public satellite ID string (e.g. "G07").
    pub satellite_id: String,
    /// Ambiguity state key (e.g. "G07#1").
    pub ambiguity_id: String,
    /// Transmission time, seconds since J2000, at which the biases and the orbit and clock
    /// solution were evaluated, when it could be predicted.
    pub transmit_time_j2000_s: Option<f64>,
    /// Solution of the SSR orbit and clock corrections the ephemeris source applies to the
    /// satellite at the transmission time, if any.
    pub applied_orbit_clock_solution: Option<SsrSolution>,
    /// Code IF combination status.
    pub code_status: SsrIfCombinationStatus,
    /// Applied code IF bias in meters (opposite sign convention, added to modeled code range).
    pub applied_code_if_m: Option<f64>,
    /// First code signal query report.
    pub code1_report: Option<SsrObsSignalReport<SsrCodeBiasQueryResult>>,
    /// Second code signal query report.
    pub code2_report: Option<SsrObsSignalReport<SsrCodeBiasQueryResult>>,
    /// Phase IF combination status.
    pub phase_status: SsrIfCombinationStatus,
    /// Applied phase IF bias in meters (direct sign convention, added to phase residual).
    pub applied_phase_if_m: Option<f64>,
    /// First phase signal query report.
    pub phase1_report: Option<SsrObsSignalReport<SsrPhaseBiasQueryResult>>,
    /// Second phase signal query report.
    pub phase2_report: Option<SsrObsSignalReport<SsrPhaseBiasQueryResult>>,
}

/// Overall aggregate status of applying SSR biases across all requested epochs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SsrPppAggregateStatus {
    /// All requested code and phase corrections were successfully combined and applied.
    AllApplied,
    /// Some requested corrections were applied, but at least one failed (unavailable, reset needed, etc.).
    PartiallyApplied,
    /// All requested corrections were unavailable or failed to apply.
    NoneApplied,
    /// No signals were requested (e.g. empty epochs or both code and phase opted out).
    #[default]
    EmptyOrOptedOut,
}

/// Complete report of SSR/HAS PPP bias application across all epochs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SsrPppBiasApplicationReport {
    /// Overall aggregate status.
    pub status: SsrPppAggregateStatus,
    /// Number of observations where code IF bias was successfully applied.
    pub code_applied_count: usize,
    /// Number of observations where code IF bias was requested but failed.
    pub code_failed_count: usize,
    /// Number of observations where phase IF bias was successfully applied.
    pub phase_applied_count: usize,
    /// Number of observations where phase IF bias was requested but failed.
    pub phase_failed_count: usize,
    /// Number of phase observations that require ambiguity reset due to discontinuity.
    pub phase_discontinuity_resets_needed: usize,
    /// Detailed per-observation reports retaining every requested observation's results.
    pub observation_reports: Vec<SsrObsApplicationReport>,
}

/// Identity of one SSR/HAS bias record an applied ionosphere-free bias was formed from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsrBiasRecord {
    /// Signal identifier.
    pub signal: u8,
    /// Source, provider and solution of the record, which is also the solution of the
    /// orbit and clock corrections it was applied with.
    pub solution: SsrSolution,
    /// IOD SSR of the record.
    pub iod_ssr: u8,
    /// Reference epoch of the record, seconds since J2000.
    pub ref_epoch_j2000_s: f64,
}

/// Why SSR/HAS biases recorded in a [`PppCorrectionLookup`] do not hold at an
/// observation's transmission time in a solve.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SsrTransmitTimeFailure {
    /// The solve's ephemeris source applies no SSR corrections
    /// ([`crate::observables::ObservableEphemerisSource::ssr_corrections`] is `None`), so
    /// the biases have no orbit and clock of their solution to go with.
    SourceWithoutSsrCorrections,
    /// The transmission time could not be predicted from the solve's source.
    TransmitTimeUnavailable,
    /// The source applies orbit and clock corrections of another solution, or none, at the
    /// transmission time.
    OrbitClockSolution {
        /// Transmission time, seconds since J2000.
        transmit_time_j2000_s: f64,
        /// Solution the source applies there, if any.
        applied: Option<SsrSolution>,
    },
    /// A recorded bias is not the record available at the transmission time.
    BiasRecord {
        /// Transmission time, seconds since J2000.
        transmit_time_j2000_s: f64,
        /// Signal of the record.
        signal: u8,
        /// Status of the query for that signal at the transmission time. `Available` means
        /// another record, of a different solution, IOD SSR or reference epoch, is.
        status: SsrBiasStatus,
    },
}

/// Where a PPP solve found an [`SsrBiasExclusion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SsrBiasExclusionStage {
    /// Before a solve pass, from the state the pass starts at.
    BeforeSolve,
    /// While a solve pass iterated: the biases stopped holding at the transmission time of
    /// an intermediate state.
    DuringIteration,
    /// At the position a solve pass converged to.
    AtConvergence,
    /// While the fixed solve re-solved with its integer ambiguities held.
    FixedResolve,
}

/// An observation left out of a PPP solve because an SSR/HAS bias it requires was not
/// resolved.
///
/// The corrections require an SSR code bias when
/// [`PppCorrectionLookup::ssr_code_bias_enabled`] is set and an SSR phase bias when
/// [`PppCorrectionLookup::phase_bias_enabled`] is set. An observation with no entry for a
/// required bias is excluded rather than solved without it.
#[derive(Debug, Clone, PartialEq)]
pub struct SsrBiasExclusion {
    /// Zero-based index of the observation's epoch in the solve's input epochs.
    pub epoch_index: usize,
    /// Public satellite token of the observation.
    pub satellite_id: String,
    /// Ambiguity state key of the observation.
    pub ambiguity_id: String,
    /// A required SSR code bias is absent.
    pub code_bias_missing: bool,
    /// A required SSR phase bias is absent.
    pub phase_bias_missing: bool,
    /// Why the recorded biases do not hold at the observation's transmission time, when the
    /// lookup has them but the solve's source does not apply them there.
    pub transmit_time_failure: Option<SsrTransmitTimeFailure>,
    /// Solve pass that found the exclusion: 0 before the first solve, from the starting
    /// position; `k` during solve pass `k`, at the position it converged to, or before the
    /// next pass from the state it reached. A static solve adds the observations a pass
    /// finds and solves again from that state, until a pass finds none.
    pub pass: usize,
    /// Where the exclusion was found: before a solve pass, during its iteration, or at the
    /// position it converged to.
    pub stage: SsrBiasExclusionStage,
    /// The row [`PppCorrectionLookup::with_ssr_biases`] reported for this observation,
    /// whose code and phase statuses state why the bias was not resolved. `None` when the
    /// lookup's [`PppCorrectionLookup::ssr_bias_report`] has no row for it, as for a lookup
    /// filled directly.
    pub application: Option<SsrObsApplicationReport>,
}

/// Per-satellite and per-system default signal mapping for SSR/HAS PPP biases.
///
/// Both application flags default to `true`, so the default value *requests* code and phase
/// biases for every observation while configuring no signal pairs to resolve them from.
/// Because requested biases fail closed, passing the default value straight to
/// [`PppCorrectionLookup::with_ssr_biases`] makes every observation unresolvable rather than
/// a no-op; see that method for the full behaviour and for how to opt out explicitly.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SsrPppBiasOptions {
    /// Satellite-specific signal pairs checked first for each observation.
    pub per_satellite: BTreeMap<GnssSatelliteId, SsrPppBiasSignalPair>,
    /// Fallback signal pairs selected by the observation satellite's GNSS
    /// system when no satellite-specific pair exists.
    pub per_system: BTreeMap<GnssSystem, SsrPppBiasSignalPair>,
    /// Whether code biases are requested (defaults to `true`).
    ///
    /// When `true`, an observation whose code bias cannot be resolved fails closed rather
    /// than being omitted. Set it to `false` to opt out.
    pub apply_code_biases: bool,
    /// Whether phase biases are requested (defaults to `true`).
    ///
    /// When `true`, an observation whose phase bias cannot be resolved fails closed rather
    /// than being omitted. Set it to `false` to opt out.
    pub apply_phase_biases: bool,
    /// Per-(ambiguity_id, signal_id) phase continuity acknowledgement tokens.
    pub phase_continuity_tokens: BTreeMap<(String, u8), PhaseContinuityToken>,
}

impl Default for SsrPppBiasOptions {
    fn default() -> Self {
        Self {
            per_satellite: BTreeMap::new(),
            per_system: BTreeMap::new(),
            apply_code_biases: true,
            apply_phase_biases: true,
            phase_continuity_tokens: BTreeMap::new(),
        }
    }
}

impl SsrPppBiasOptions {
    /// Create new bias options with both code and phase biases enabled by default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the satellite-specific pair, or the pair configured for its GNSS
    /// system when no satellite-specific entry exists.
    pub fn signal_pair(&self, sat: GnssSatelliteId) -> Option<SsrPppBiasSignalPair> {
        self.per_satellite
            .get(&sat)
            .copied()
            .or_else(|| self.per_system.get(&sat.system).copied())
    }

    /// Builder method to configure code bias application opt-in/opt-out.
    pub fn with_apply_code_biases(mut self, apply: bool) -> Self {
        self.apply_code_biases = apply;
        self
    }

    /// Builder method to configure phase bias application opt-in/opt-out.
    pub fn with_apply_phase_biases(mut self, apply: bool) -> Self {
        self.apply_phase_biases = apply;
        self
    }

    /// Builder method to acknowledge continuity for a specific ambiguity and signal.
    pub fn with_phase_continuity_token(
        mut self,
        ambiguity_id: impl Into<String>,
        signal: u8,
        token: PhaseContinuityToken,
    ) -> Self {
        self.phase_continuity_tokens
            .insert((ambiguity_id.into(), signal), token);
        self
    }

    /// Builder method to register a fallback signal pair for a GNSS system.
    pub fn with_system_signal_pair(
        mut self,
        system: GnssSystem,
        pair: SsrPppBiasSignalPair,
    ) -> Self {
        self.per_system.insert(system, pair);
        self
    }

    /// Builder method to register a satellite-specific signal pair.
    pub fn with_satellite_signal_pair(
        mut self,
        sat: GnssSatelliteId,
        pair: SsrPppBiasSignalPair,
    ) -> Self {
        self.per_satellite.insert(sat, pair);
        self
    }
}

impl PppCorrectionLookup {
    /// Convert precomputed correction vectors and scalars into indexed lookup
    /// tables, copying option enablement and leaving SSR/HAS tables disabled.
    pub fn from_options(value: PppCorrections, options: &PppCorrectionsOptions) -> Self {
        Self::from_parts(
            value,
            options.solid_earth_tide,
            options.pole_tide.is_some(),
            options.ocean_loading.is_some(),
            options.phase_windup,
            options.satellite_antenna.is_some(),
            options.code_bias.is_some(),
        )
    }

    fn from_parts(
        value: PppCorrections,
        tide_enabled: bool,
        pole_tide_enabled: bool,
        ocean_loading_enabled: bool,
        windup_enabled: bool,
        satellite_antenna_enabled: bool,
        code_bias_enabled: bool,
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
            code_bias_m: value
                .code_bias_m
                .into_iter()
                .map(|c| ((c.sat, c.epoch_index), c.value_m))
                .collect(),
            ssr_code_bias_m: BTreeMap::new(),
            phase_bias_m: BTreeMap::new(),
            ssr_code_bias_records: BTreeMap::new(),
            phase_bias_records: BTreeMap::new(),
            ssr_bias_report: None,
            tide_enabled,
            pole_tide_enabled,
            ocean_loading_enabled,
            windup_enabled,
            satellite_antenna_enabled,
            code_bias_enabled,
            ssr_code_bias_enabled: false,
            phase_bias_enabled: false,
        }
    }

    /// Merge parsed SSR/HAS signal biases into this lookup for the given epochs,
    /// returning the updated lookup and a complete typed application report.
    ///
    /// SSR/HAS code biases are stored as code-only model-side corrections, so an
    /// observation-side HAS/SSR bias uses the opposite sign. Phase biases are
    /// stored as observation-side corrections and are added directly to the
    /// phase residual. Both are keyed by satellite, epoch index and ambiguity id.
    ///
    /// # Evaluated at the transmission time
    ///
    /// The solve evaluates `ephemeris` at each observation's transmission time, so every
    /// bias is judged there too. The transmission time is predicted from the reception
    /// time and `receiver_position_m`, the way the solve predicts it; an error of a
    /// kilometre in the position moves it by about 3 µs. Biases are resolved with
    /// [`crate::ssr::SsrCorrectionStore::query_code_bias`] and
    /// [`crate::ssr::SsrCorrectionStore::query_phase_bias`] on `ephemeris`'s store at that
    /// time, so their lifetime, any Galileo HAS do-not-use exclusion and phase continuity
    /// apply. The untimed `code_bias` and `phase_bias` accessors are not used. Each applied
    /// bias records its two signals' records ([`Self::ssr_code_bias_records`],
    /// [`Self::phase_bias_records`]), and the row model checks them again at the
    /// transmission time of every solve iteration.
    ///
    /// Ionosphere-free combination requires the satellite not to be excluded by a HAS
    /// do-not-use indication, valid finite unequal carrier frequencies, both signals
    /// available from one solution with matching IOD SSR, and no phase continuity break
    /// against a caller token. The frequencies are the signal pair's, or the observation's
    /// when either configured value is not positive. It also requires `ephemeris` to apply
    /// SSR orbit and clock corrections to the satellite at the transmission time
    /// ([`SsrCorrectedEphemeris::applied_orbit_clock_solution`]) and the biases to come
    /// from that same source, provider and solution. A bias from any other solution is
    /// reported as [`SsrIfCombinationStatus::OrbitClockSolutionMismatch`] and not applied;
    /// where `ephemeris` declines the satellite or falls back to the broadcast state it is
    /// reported as [`SsrIfCombinationStatus::OrbitClockSolutionUnavailable`].
    ///
    /// The solve has to be given this same `ephemeris` as its source: the row model reads
    /// the applied solution through
    /// [`crate::observables::ObservableEphemerisSource::ssr_corrections`], and a source
    /// without SSR corrections, such as SP3 and clock products, has no solution for the
    /// recorded biases, so the observation is left out.
    ///
    /// # Phase continuity
    ///
    /// A phase bias is queried with the token stored in
    /// [`SsrPppBiasOptions::phase_continuity_tokens`] under the observation's ambiguity id
    /// and the signal. A token that does not continue the current arc yields
    /// [`SsrIfCombinationStatus::PhaseDiscontinuityNeedsReset`] for that ambiguity only.
    /// Without a token the caller holds no arc for that ambiguity, so the bias is applied and
    /// any break recorded since the previous arc is reported in the signal report's
    /// `discontinuity_details`; the returned `continuity_token` is the one to pass on the
    /// next call. A Galileo HAS arc follows the source and the PDI only, so a new mask ID or
    /// IOD set ID keeps it.
    ///
    /// # Unresolved biases exclude the observation
    ///
    /// Calling this method *requests* bias application for every observation in `epochs`,
    /// according to [`SsrPppBiasOptions::apply_code_biases`] and
    /// [`SsrPppBiasOptions::apply_phase_biases`], which are both `true` by default. The
    /// corresponding requirement flags on the returned lookup are set from those options
    /// alone, before any observation is examined, and they stay set even when no bias could
    /// be resolved. A requested bias that cannot be produced is never treated as zero: it is
    /// reported with a typed [`SsrIfCombinationStatus`], and the float, fixed and kinematic
    /// solves leave the observation out and list it in `ssr_bias_exclusions`, with this
    /// report's row for it. A static solve fails only when the observations that remain
    /// cannot support it, with
    /// [`FloatSolveError::InsufficientObservationsAfterSsrBiasExclusion`].
    ///
    /// This includes the case where nothing was configured to resolve. Default options carry
    /// no `per_satellite` or `per_system` signal pairs, so
    /// `with_ssr_biases(&ephemeris, &epochs, position, &SsrPppBiasOptions::default())`
    /// reports every observation as [`SsrIfCombinationStatus::NoSignalPairConfigured`], and a
    /// solve on the result excludes every observation.
    ///
    /// To run without SSR/HAS biases, opt out explicitly with
    /// [`SsrPppBiasOptions::with_apply_code_biases`] and
    /// [`SsrPppBiasOptions::with_apply_phase_biases`] set to `false`, or do not call this
    /// method. Opting out reports [`SsrIfCombinationStatus::OptedOut`], counts no failures,
    /// and leaves the requirement flags clear.
    pub fn with_ssr_biases(
        mut self,
        ephemeris: &SsrCorrectedEphemeris<'_>,
        epochs: &[FloatEpoch],
        receiver_position_m: [f64; 3],
        options: &SsrPppBiasOptions,
    ) -> (Self, SsrPppBiasApplicationReport) {
        let store = ephemeris.store();
        let mut report = SsrPppBiasApplicationReport::default();
        self.ssr_code_bias_enabled = options.apply_code_biases;
        self.phase_bias_enabled = options.apply_phase_biases;

        for (epoch_index, epoch) in epochs.iter().enumerate() {
            for obs in &epoch.observations {
                let key = (obs.sat, epoch_index, obs.ambiguity_id.clone());
                self.ssr_code_bias_m.remove(&key);
                self.phase_bias_m.remove(&key);
                self.ssr_code_bias_records.remove(&key);
                self.phase_bias_records.remove(&key);
                let transmit_time_j2000_s =
                    transmit_time_j2000_s(ephemeris, obs, receiver_position_m, epoch.t_rx_j2000_s);
                let applied_orbit_clock_solution = transmit_time_j2000_s
                    .and_then(|t_tx| ephemeris.applied_orbit_clock_solution(obs.sat, t_tx));
                let signals = options.signal_pair(obs.sat);

                let (code_outcome, code1_report, code2_report) =
                    match (signals, transmit_time_j2000_s) {
                        _ if !options.apply_code_biases => {
                            (Err(SsrIfCombinationStatus::OptedOut), None, None)
                        }
                        (None, _) => (
                            Err(SsrIfCombinationStatus::NoSignalPairConfigured),
                            None,
                            None,
                        ),
                        (Some(signals), transmit_time) => {
                            let t_query = transmit_time.unwrap_or(epoch.t_rx_j2000_s);
                            let q1 = store.query_code_bias(obs.sat, signals.code1_signal, t_query);
                            let q2 = store.query_code_bias(obs.sat, signals.code2_signal, t_query);
                            let views = [BiasQueryView::code(&q1), BiasQueryView::code(&q2)];
                            let outcome = match transmit_time {
                                Some(_) => ionosphere_free_ssr_bias_m(
                                    combination_frequencies(&signals, obs),
                                    applied_orbit_clock_solution,
                                    views[0],
                                    views[1],
                                ),
                                None => Err(transmit_time_unavailable_status(&views)),
                            };
                            (
                                outcome,
                                Some(signal_report(epoch_index, obs, signals.code1_signal, q1)),
                                Some(signal_report(epoch_index, obs, signals.code2_signal, q2)),
                            )
                        }
                    };

                let (phase_outcome, phase1_report, phase2_report) =
                    match (signals, transmit_time_j2000_s) {
                        _ if !options.apply_phase_biases => {
                            (Err(SsrIfCombinationStatus::OptedOut), None, None)
                        }
                        (None, _) => (
                            Err(SsrIfCombinationStatus::NoSignalPairConfigured),
                            None,
                            None,
                        ),
                        (Some(signals), transmit_time) => {
                            let t_query = transmit_time.unwrap_or(epoch.t_rx_j2000_s);
                            let token = |signal: u8| {
                                options
                                    .phase_continuity_tokens
                                    .get(&(obs.ambiguity_id.clone(), signal))
                                    .copied()
                            };
                            let q1 = store.query_phase_bias(
                                obs.sat,
                                signals.phase1_signal,
                                t_query,
                                token(signals.phase1_signal),
                            );
                            let q2 = store.query_phase_bias(
                                obs.sat,
                                signals.phase2_signal,
                                t_query,
                                token(signals.phase2_signal),
                            );
                            let views = [BiasQueryView::phase(&q1), BiasQueryView::phase(&q2)];
                            let outcome = match transmit_time {
                                Some(_) => ionosphere_free_ssr_bias_m(
                                    combination_frequencies(&signals, obs),
                                    applied_orbit_clock_solution,
                                    views[0],
                                    views[1],
                                ),
                                None => Err(transmit_time_unavailable_status(&views)),
                            };
                            (
                                outcome,
                                Some(signal_report(epoch_index, obs, signals.phase1_signal, q1)),
                                Some(signal_report(epoch_index, obs, signals.phase2_signal, q2)),
                            )
                        }
                    };

                // Code biases are stored with the model-side sign: the row model adds them
                // to the modelled code, the equivalent of adding the transmitted bias to the
                // observation (RTKLIB `P = P_obs + cbias`).
                let applied_code_if_m = code_outcome.as_ref().ok().map(|(code_if_m, _)| -code_if_m);
                let applied_phase_if_m = phase_outcome
                    .as_ref()
                    .ok()
                    .map(|(phase_if_m, _)| *phase_if_m);
                if let (Some(value), Ok((_, records))) = (applied_code_if_m, &code_outcome) {
                    self.ssr_code_bias_m.insert(key.clone(), value);
                    self.ssr_code_bias_records.insert(key.clone(), *records);
                }
                if let Ok((value, records)) = &phase_outcome {
                    self.phase_bias_m.insert(key.clone(), *value);
                    self.phase_bias_records.insert(key.clone(), *records);
                }
                let code_status = report.count_code(code_outcome.map(|(value, _)| value));
                let phase_status = report.count_phase(phase_outcome.map(|(value, _)| value));
                report.observation_reports.push(SsrObsApplicationReport {
                    epoch_index,
                    sat: obs.sat,
                    satellite_id: obs.satellite_id.clone(),
                    ambiguity_id: obs.ambiguity_id.clone(),
                    transmit_time_j2000_s,
                    applied_orbit_clock_solution,
                    code_status,
                    applied_code_if_m,
                    code1_report,
                    code2_report,
                    phase_status,
                    applied_phase_if_m,
                    phase1_report,
                    phase2_report,
                });
            }
        }

        let total_requested = report.code_applied_count
            + report.code_failed_count
            + report.phase_applied_count
            + report.phase_failed_count;
        let total_applied = report.code_applied_count + report.phase_applied_count;
        report.status = if total_requested == 0 {
            SsrPppAggregateStatus::EmptyOrOptedOut
        } else if total_applied == total_requested {
            SsrPppAggregateStatus::AllApplied
        } else if total_applied == 0 {
            SsrPppAggregateStatus::NoneApplied
        } else {
            SsrPppAggregateStatus::PartiallyApplied
        };
        self.ssr_bias_report = Some(report.clone());
        (self, report)
    }
}

impl SsrPppBiasApplicationReport {
    fn count_code(
        &mut self,
        outcome: Result<f64, SsrIfCombinationStatus>,
    ) -> SsrIfCombinationStatus {
        match outcome {
            Ok(_) => {
                self.code_applied_count += 1;
                SsrIfCombinationStatus::Applied
            }
            Err(SsrIfCombinationStatus::OptedOut) => SsrIfCombinationStatus::OptedOut,
            Err(status) => {
                self.code_failed_count += 1;
                status
            }
        }
    }

    fn count_phase(
        &mut self,
        outcome: Result<f64, SsrIfCombinationStatus>,
    ) -> SsrIfCombinationStatus {
        match outcome {
            Ok(_) => {
                self.phase_applied_count += 1;
                SsrIfCombinationStatus::Applied
            }
            Err(SsrIfCombinationStatus::OptedOut) => SsrIfCombinationStatus::OptedOut,
            Err(status) => {
                self.phase_failed_count += 1;
                if status == SsrIfCombinationStatus::PhaseDiscontinuityNeedsReset {
                    self.phase_discontinuity_resets_needed += 1;
                }
                status
            }
        }
    }
}

/// Per-signal report row for one observation's bias query.
fn signal_report<Q>(
    epoch_index: usize,
    obs: &FloatObservation,
    signal_id: u8,
    query_result: Q,
) -> SsrObsSignalReport<Q> {
    SsrObsSignalReport {
        epoch_index,
        sat: obs.sat,
        ambiguity_id: obs.ambiguity_id.clone(),
        signal_id,
        query_result,
    }
}

/// The fields of a code- or phase-bias query result that the combination reads.
#[derive(Clone, Copy)]
struct BiasQueryView {
    signal: u8,
    status: SsrBiasStatus,
    bias_m: Option<f64>,
    solution: Option<SsrSolution>,
    iod_ssr: Option<u8>,
    ref_epoch_j2000_s: Option<f64>,
}

impl BiasQueryView {
    fn code(query: &SsrCodeBiasQueryResult) -> Self {
        Self {
            signal: query.signal,
            status: query.status,
            bias_m: query.bias_m,
            solution: query.solution,
            iod_ssr: query.iod_ssr,
            ref_epoch_j2000_s: query.ref_epoch_j2000_s,
        }
    }

    fn phase(query: &SsrPhaseBiasQueryResult) -> Self {
        Self {
            signal: query.signal,
            status: query.status,
            bias_m: query.bias_m,
            solution: query.solution,
            iod_ssr: query.iod_ssr,
            ref_epoch_j2000_s: query.ref_epoch_j2000_s,
        }
    }

    /// The record this view read, when the query found an available one.
    fn record(&self) -> Option<SsrBiasRecord> {
        Some(SsrBiasRecord {
            signal: self.signal,
            solution: self.solution?,
            iod_ssr: self.iod_ssr?,
            ref_epoch_j2000_s: self.ref_epoch_j2000_s?,
        })
    }
}

/// Status of a bias whose transmission time could not be predicted, from its signals'
/// queries at the reception time: a do-not-use exclusion, which is why the source cannot
/// place the satellite, or otherwise the missing transmission time itself.
fn transmit_time_unavailable_status(views: &[BiasQueryView; 2]) -> SsrIfCombinationStatus {
    if views
        .iter()
        .any(|view| view.status == SsrBiasStatus::Excluded)
    {
        SsrIfCombinationStatus::SatelliteExcluded
    } else {
        SsrIfCombinationStatus::TransmitTimeUnavailable
    }
}

/// Transmission time of `obs` predicted the way the solve predicts it, from the reception
/// time and an approximate receiver position, or `None` when the source cannot place the
/// satellite.
fn transmit_time_j2000_s(
    ephemeris: &SsrCorrectedEphemeris<'_>,
    obs: &FloatObservation,
    receiver_position_m: [f64; 3],
    t_rx_j2000_s: f64,
) -> Option<f64> {
    let options = super::predict_default(ephemeris, obs).ok()?;
    let t_tx = crate::observables::transmit_epoch_j2000_s(
        ephemeris,
        obs.sat,
        receiver_position_m,
        t_rx_j2000_s,
        crate::observables::TransmitTimeOptions {
            light_time: options.light_time,
            sagnac: options.sagnac,
        },
        crate::observables::flight_time_seed_s(obs.code_m),
    )
    .ok()?;
    t_tx.is_finite().then_some(t_tx)
}

/// Carrier frequencies for the ionosphere-free combination: the signal pair's, or the
/// observation's when either configured value is not positive and both of the
/// observation's are.
fn combination_frequencies(signals: &SsrPppBiasSignalPair, obs: &FloatObservation) -> (f64, f64) {
    if signals.freq1_hz > 0.0 && signals.freq2_hz > 0.0 {
        (signals.freq1_hz, signals.freq2_hz)
    } else if obs.freq1_hz > 0.0 && obs.freq2_hz > 0.0 {
        (obs.freq1_hz, obs.freq2_hz)
    } else {
        (signals.freq1_hz, signals.freq2_hz)
    }
}

/// Ionosphere-free SSR bias in metres with the store's sign and the two records it was
/// formed from, or the typed reason it cannot be formed. The checks run in the order the
/// statuses are reported: a continuity reset, a do-not-use exclusion, frequencies,
/// availability, agreement of the two signals, then agreement with the orbit and clock
/// solution the ephemeris source applies.
fn ionosphere_free_ssr_bias_m(
    (f1, f2): (f64, f64),
    applied_orbit_clock_solution: Option<SsrSolution>,
    first: BiasQueryView,
    second: BiasQueryView,
) -> Result<(f64, [SsrBiasRecord; 2]), SsrIfCombinationStatus> {
    let statuses = [first.status, second.status];
    if statuses.contains(&SsrBiasStatus::PhaseDiscontinuityNeedsReset) {
        return Err(SsrIfCombinationStatus::PhaseDiscontinuityNeedsReset);
    }
    if statuses.contains(&SsrBiasStatus::Excluded) {
        return Err(SsrIfCombinationStatus::SatelliteExcluded);
    }
    if !(f1.is_finite() && f2.is_finite() && f1 > 0.0 && f2 > 0.0 && f1 != f2) {
        return Err(SsrIfCombinationStatus::InvalidFrequencies);
    }
    if first.status != SsrBiasStatus::Available || second.status != SsrBiasStatus::Available {
        return Err(SsrIfCombinationStatus::SignalUnavailable);
    }
    if first.solution != second.solution {
        return Err(SsrIfCombinationStatus::IncompatibleSourceOrSolution);
    }
    if first.iod_ssr != second.iod_ssr {
        return Err(SsrIfCombinationStatus::IncompatibleIod);
    }
    if applied_orbit_clock_solution.is_none() {
        return Err(SsrIfCombinationStatus::OrbitClockSolutionUnavailable);
    }
    if first.solution != applied_orbit_clock_solution {
        return Err(SsrIfCombinationStatus::OrbitClockSolutionMismatch);
    }
    let (Some(b1), Some(b2), Some(record1), Some(record2)) =
        (first.bias_m, second.bias_m, first.record(), second.record())
    else {
        return Err(SsrIfCombinationStatus::SignalUnavailable);
    };
    let if_m = ionosphere_free_bias_m(b1, b2, f1, f2);
    if if_m.is_finite() {
        Ok((if_m, [record1, record2]))
    } else {
        Err(SsrIfCombinationStatus::InvalidFrequencies)
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
        let code_bias_enabled = !value.code_bias_m.is_empty();
        let phase_bias_enabled = false;
        Self::from_parts(
            value,
            tide_enabled,
            pole_tide_enabled,
            ocean_loading_enabled,
            windup_enabled,
            satellite_antenna_enabled,
            code_bias_enabled,
        )
        .with_phase_bias_enabled(phase_bias_enabled)
    }
}

impl PppCorrectionLookup {
    fn with_phase_bias_enabled(mut self, phase_bias_enabled: bool) -> Self {
        self.phase_bias_enabled = phase_bias_enabled;
        self
    }
}

fn ionosphere_free_bias_m(b1_m: f64, b2_m: f64, f1_hz: f64, f2_hz: f64) -> f64 {
    let gamma = f1_hz * f1_hz / (f1_hz * f1_hz - f2_hz * f2_hz);
    gamma * b1_m - (gamma - 1.0) * b2_m
}

/// Per-satellite residual row in the returned public solution.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatResidual {
    /// Zero-based input-epoch index for this residual row.
    pub epoch_index: usize,
    /// Public satellite token copied from the observation.
    pub satellite_id: String,
    /// Ambiguity state key copied from the observation, which identifies the observation
    /// within its epoch and groups temporal residual arcs.
    pub ambiguity_id: String,
    /// Code prefit residual returned by the undifferenced row model.
    pub code_m: f64,
    /// Phase prefit residual returned by the undifferenced row model.
    pub phase_m: f64,
    /// Code inverse-sigma row weight, including optional elevation scaling.
    pub code_weight: f64,
    /// Phase inverse-sigma row weight, including optional elevation scaling.
    pub phase_weight: f64,
}

/// Temporal correlation estimate used to deflate static PPP sample count.
///
/// The estimator pools lag-1 autocorrelation from standardized post-fit code
/// and phase residual arcs, split by satellite and observable. It models the
/// residual sequence as AR(1) in epoch units and converts the fitted lag-1
/// coefficient into an effective independent sample count. This captures
/// short-memory temporal correlation such as multipath, residual troposphere,
/// and orbit or clock interpolation errors that persist for minutes. It does
/// not model spatially correlated or day-constant systematics, antenna effects,
/// loading errors, or reference-frame errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemporalCorrelationSummary {
    /// Pooled nonnegative lag-1 autocorrelation coefficient.
    pub lag1_autocorrelation: f64,
    /// AR(1) decorrelation time in epoch spacings, computed as `-1 / ln(rho)`.
    pub decorrelation_time_epochs: f64,
    /// AR(1) decorrelation time in seconds when the input epochs have a regular
    /// positive spacing.
    pub decorrelation_time_s: Option<f64>,
    /// Number of residual samples contributing to the temporal estimate.
    pub nominal_sample_count: usize,
    /// Effective independent sample count after AR(1) deflation.
    pub effective_sample_count: f64,
    /// Multiplier applied to covariance to account for temporal correlation.
    pub variance_inflation_factor: f64,
    /// Number of satellite and observable arcs used in the pooled estimate.
    pub arcs_used: usize,
}

/// Static float solution.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatSolution {
    /// Estimated static receiver position in ECEF/ITRF metres.
    pub position_m: [f64; 3],
    /// Posterior covariance of [`Self::position_m`], scaled by
    /// [`Self::position_covariance_scale_factor`].
    ///
    /// The ECEF and ENU matrices follow [`PositionCovariance`]. They are the
    /// top-left position block of the inverse final weighted normal matrix after
    /// the per-epoch receiver clocks are eliminated and the remaining static
    /// states are marginalized, then multiplied by the a-posteriori residual
    /// variance factor reported in [`Self::posterior_variance_factor`]. This
    /// calibrates the covariance to the residual scale of this solve, but can
    /// still be optimistic when residuals are temporally correlated or dominated
    /// by multipath, antenna, loading, or other unmodelled systematics. When the
    /// residual factor is below 1.0, this covariance is smaller than the formal
    /// covariance.
    pub position_covariance: PositionCovariance,
    /// Unscaled formal covariance from the final weighted normal matrix.
    ///
    /// This is the unit-variance covariance before applying
    /// [`Self::posterior_variance_factor`].
    pub formal_position_covariance: PositionCovariance,
    /// A-posteriori unit-variance factor, computed as weighted SSR divided by
    /// unreduced degrees of freedom. Eliminated receiver clocks still count as
    /// estimated parameters in the denominator.
    pub posterior_variance_factor: f64,
    /// Multiplier applied to [`Self::formal_position_covariance`] to produce
    /// [`Self::position_covariance`]. This equals
    /// [`Self::posterior_variance_factor`].
    pub position_covariance_scale_factor: f64,
    /// Position covariance with temporal-correlation sample-count deflation.
    ///
    /// This additive field does not change [`Self::position_covariance`]. It is
    /// the formal covariance multiplied by the larger of 1.0 and the posterior
    /// variance factor, and then by
    /// [`Self::temporal_correlation`]'s variance inflation factor. The floor
    /// keeps this conservative covariance from shrinking below the formal
    /// normal-equation covariance.
    pub temporal_position_covariance: PositionCovariance,
    /// Multiplier applied to [`Self::formal_position_covariance`] to produce
    /// [`Self::temporal_position_covariance`].
    pub temporal_position_covariance_scale_factor: f64,
    /// Residual temporal-correlation estimate used by
    /// [`Self::temporal_position_covariance`].
    pub temporal_correlation: TemporalCorrelationSummary,
    /// Final receiver-clock states in meters copied from [`FloatState`], one for each
    /// solved epoch, in the order of [`Self::solved_epoch_indices`].
    pub epoch_clocks_m: Vec<f64>,
    /// Final float ambiguity estimates in meters keyed by ambiguity id; fixed
    /// resolution converts selected values to cycles with wavelength and offset.
    pub ambiguities_m: BTreeMap<String, f64>,
    /// Post-combination ionosphere states when residual-ionosphere estimation is
    /// enabled; otherwise this is an empty map.
    pub residual_ionosphere_m: BTreeMap<String, f64>,
    /// Estimated residual zenith total delay when troposphere and ZTD estimation
    /// are enabled; otherwise this is `None`.
    pub ztd_residual_m: Option<f64>,
    /// Estimated north horizontal troposphere gradient, in metres.
    pub tropo_gradient_north_m: Option<f64>,
    /// Estimated east horizontal troposphere gradient, in metres.
    pub tropo_gradient_east_m: Option<f64>,
    /// Posterior covariance of north/east troposphere gradients, in square
    /// metres, scaled by [`Self::position_covariance_scale_factor`].
    pub tropo_gradient_covariance_m2: Option<[[f64; 2]; 2]>,
    /// Unscaled formal covariance of north/east troposphere gradients.
    pub formal_tropo_gradient_covariance_m2: Option<[[f64; 2]; 2]>,
    /// Returned residual rows in epoch and observation traversal order.
    pub residuals_m: Vec<FloatResidual>,
    /// Sorted ambiguity ids collected from active observations and used as the
    /// default fixed integer-search order.
    pub used_sats: Vec<String>,
    /// Iteration counter retained from the float solve loop, which starts at 1.
    pub iterations: usize,
    /// True only when the state-tolerance branch ended the float solve.
    pub converged: bool,
    /// Records whether the float solve met state tolerances or reached its
    /// iteration cap.
    pub status: FloatStatus,
    /// Root-mean-square of the returned code prefit residuals.
    pub code_rms_m: f64,
    /// Root-mean-square of the returned phase prefit residuals.
    pub phase_rms_m: f64,
    /// Root-mean-square after each returned code and phase residual is
    /// multiplied by its row weight.
    pub weighted_rms_m: f64,
    /// Observations left out of the solve because an SSR/HAS bias the corrections
    /// require was not resolved for them, in epoch and observation order.
    pub ssr_bias_exclusions: Vec<SsrBiasExclusion>,
    /// Input epoch index of each solved epoch, ascending, paired with
    /// [`Self::epoch_clocks_m`]. An input epoch left with no observations by the elevation
    /// cutoff, SSR bias exclusion or the residual screen has no receiver clock to estimate
    /// and is not solved, so it is absent here.
    pub solved_epoch_indices: Vec<usize>,
    /// Observations, as (input epoch index, ambiguity id), the solve admitted again after
    /// an SSR/HAS bias exclusion. A later solve from this solution, such as the fixed
    /// solve, judges them only at convergence and does not admit them again.
    pub ssr_bias_readmissions: Vec<(usize, String)>,
    /// The last pass of the solve's SSR/HAS bias fixed point, from which a later solve from
    /// this solution, such as the fixed solve, continues the numbering.
    pub ssr_bias_last_pass: usize,
    /// Whether the solve ran the residual screen.
    pub residual_screen: bool,
    /// The iteration and convergence options the solve ran with.
    pub solve_options: FloatSolveOptions,
    /// Observations, as (input epoch index, ambiguity id), the residual screen removed from
    /// the accepted solution. The fixed solve leaves them out too.
    pub residual_screen_removals: Vec<(usize, String)>,
}

/// Static PPP solve termination status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatStatus {
    /// The current position, clock, ZTD, gradient, and ambiguity updates all
    /// met their configured tolerances.
    StateTolerance,
    /// The solve reached `max_iterations` before all active updates met their
    /// configured tolerances.
    MaxIterations,
}

/// Static PPP solve errors.
#[derive(Debug, Clone, PartialEq)]
pub enum FloatSolveError {
    /// Observable prediction or a required satellite clock could not provide
    /// the requested epoch.
    NoEphemeris {
        /// Public satellite token for the failed observation.
        satellite_id: String,
        /// Distinguishes absent ephemeris, an absent clock, and an upstream
        /// prediction or media error.
        reason: NoEphemerisReason,
    },
    /// The normal equations, covariance, or related geometry could not produce
    /// a valid nonsingular result.
    SingularGeometry,
    /// A receiver-clock vector does not contain one entry per input epoch.
    InvalidClockCount {
        /// Number of clock entries required by the input epoch slice.
        expected: usize,
        /// Number of clock entries supplied by the caller.
        actual: usize,
    },
    /// An iteration or convergence option failed pre-solve validation.
    InvalidSolveOption {
        /// Static option name reported by validation.
        field: &'static str,
        /// Static reason the option was rejected.
        reason: &'static str,
    },
    /// An epoch, observation, state, correction, covariance, or meteorological
    /// input failed field-level validation.
    InvalidInput {
        /// Static input label identifying the rejected value.
        field: &'static str,
        /// Static validation reason paired with the input label.
        reason: &'static str,
    },
    /// Elevation filtering left too few observations or fewer than four active
    /// satellites for the configured PPP parameter layout.
    InsufficientObservationsAfterElevationCutoff {
        /// Elevation threshold applied before row assembly.
        cutoff_deg: f64,
        /// Number of observations remaining after filtering.
        retained_observations: usize,
        /// Minimum retained count computed from the active PPP unknown count.
        required_observations: usize,
    },
    /// Leaving out the observations whose required SSR/HAS bias was not resolved left too
    /// few observations or fewer than four distinct satellites for the configured PPP
    /// parameter layout.
    InsufficientObservationsAfterSsrBiasExclusion {
        /// Number of observations left out for an unresolved SSR/HAS bias.
        excluded_observations: usize,
        /// Number of observations remaining.
        retained_observations: usize,
        /// Minimum retained count computed from the active PPP unknown count.
        required_observations: usize,
    },
    /// An estimated ambiguity id was absent when an update or fixed conversion
    /// required it.
    MissingAmbiguity(String),
    /// An enabled correction lookup did not contain the required datum.
    MissingCorrection {
        /// Public satellite token for the observation needing the datum.
        satellite_id: String,
        /// Correction or receiver-antenna datum that was absent.
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
            Self::InsufficientObservationsAfterElevationCutoff {
                cutoff_deg,
                retained_observations,
                required_observations,
            } => write!(
                f,
                "PPP elevation cutoff {cutoff_deg} deg retained {retained_observations} observations; at least {required_observations} are required"
            ),
            Self::InsufficientObservationsAfterSsrBiasExclusion {
                excluded_observations,
                retained_observations,
                required_observations,
            } => write!(
                f,
                "PPP SSR bias exclusion left out {excluded_observations} observations and retained {retained_observations}; at least {required_observations} are required"
            ),
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
/// Identifies the correction lookup or receiver-antenna datum missing from a
/// PPP measurement row.
pub enum MissingCorrection {
    /// The epoch-indexed solid-earth tide vector was absent while enabled.
    SolidEarthTide,
    /// The epoch-indexed pole-tide vector was absent while enabled.
    PoleTide,
    /// The epoch-indexed ocean-loading vector was absent while enabled.
    OceanLoading,
    /// The satellite/epoch phase wind-up value was absent while enabled.
    PhaseWindup,
    /// The satellite/epoch satellite PCO vector was absent while enabled.
    SatelliteAntennaPco,
    /// The satellite/epoch satellite PCV value was absent while enabled.
    SatelliteAntennaPcv,
    /// The satellite/epoch clock-datum code bias was absent while enabled.
    CodeBias,
    /// The satellite/epoch SSR/HAS code bias was absent while enabled.
    SsrCodeBias,
    /// The satellite/epoch SSR/HAS phase bias was absent while enabled.
    PhaseBias,
    /// The configured receiver frequency label had no matching calibration.
    ReceiverAntennaFrequency(String),
    /// The configured receiver frequency had no usable PCV value.
    ReceiverAntennaPcv(String),
    /// The satellite-to-receiver vector could not be normalized for projection.
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
            Self::CodeBias => write!(f, "code-bias correction"),
            Self::SsrCodeBias => write!(f, "SSR/HAS code-bias correction"),
            Self::PhaseBias => write!(f, "phase-bias correction"),
            Self::ReceiverAntennaFrequency(label) => {
                write!(f, "receiver antenna frequency {label}")
            }
            Self::ReceiverAntennaPcv(label) => write!(f, "receiver antenna PCV {label}"),
            Self::ReceiverAntennaGeometry => write!(f, "receiver antenna geometry"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Explains why observable prediction or satellite-clock lookup could not supply
/// a requested PPP epoch.
pub enum NoEphemerisReason {
    /// No ephemeris product covers the requested epoch.
    NoEphemeris,
    /// Predicted or external satellite clock data is unavailable at the transmit
    /// time.
    MissingSatelliteClock,
    /// Formatted upstream invalid-input, ephemeris, or media error text.
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
#[non_exhaustive]
pub struct FixedAmbiguityOptions {
    /// Per-ambiguity carrier wavelengths in meters used for meter/cycle
    /// conversion and ambiguity covariance scaling; each value must be finite
    /// and positive.
    pub wavelengths_m: BTreeMap<String, f64>,
    /// Per-ambiguity meter offsets removed before float-to-cycle conversion and
    /// added after integer cycles are converted back to meters.
    pub offsets_m: BTreeMap<String, f64>,
    /// Acceptance threshold passed to the LAMBDA integer lattice search.
    pub ratio_threshold: f64,
}

impl FixedAmbiguityOptions {
    /// Build ambiguity controls with an explicit ratio threshold.
    ///
    /// The wavelength and offset maps start empty because their values are
    /// signal-specific; assign them before a fixed solve.
    #[must_use]
    pub fn new(ratio_threshold: f64) -> Self {
        Self {
            wavelengths_m: BTreeMap::new(),
            offsets_m: BTreeMap::new(),
            ratio_threshold,
        }
    }
}

/// Static fixed-ambiguity PPP solve controls.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FixedSolveConfig {
    /// Code/phase inverse-sigma weights copied into the fixed solve context.
    pub weights: MeasurementWeights,
    /// Troposphere and optional ZTD/gradient controls copied into the fixed
    /// solve context.
    pub tropo: TroposphereOptions,
    /// Correction tables reused for integer-search covariance and fixed rows.
    pub corrections: RangeCorrections,
    /// Iteration cap and state tolerances used by the fixed re-solve.
    pub opts: FloatSolveOptions,
    /// Optional PPP observation elevation cutoff in degrees.
    ///
    /// `None` applies no cutoff. When set, the fixed path applies the cutoff
    /// before active ambiguity ids, integer covariance assembly, residual rows,
    /// and normal rows are built. A cutoff lower than the float solve's, or
    /// none, keeps any satellite the float solve cut, and the fixed solve then
    /// solves the float arc again over the fixed arc before it fixes.
    pub elevation_cutoff_deg: Option<f64>,
    /// Per-ambiguity wavelengths, offsets, and integer acceptance ratio used by
    /// the LAMBDA search and the ambiguity-held re-solve.
    pub ambiguity: FixedAmbiguityOptions,
    /// Estimate one post-combination residual ionosphere state per ambiguity
    /// arc during the fixed re-solve.
    pub estimate_residual_ionosphere: bool,
}

impl FixedSolveConfig {
    /// Build static fixed-solve controls from all model and ambiguity settings.
    #[must_use]
    pub const fn new(
        weights: MeasurementWeights,
        tropo: TroposphereOptions,
        corrections: RangeCorrections,
        opts: FloatSolveOptions,
        elevation_cutoff_deg: Option<f64>,
        ambiguity: FixedAmbiguityOptions,
        estimate_residual_ionosphere: bool,
    ) -> Self {
        Self {
            weights,
            tropo,
            corrections,
            opts,
            elevation_cutoff_deg,
            ambiguity,
            estimate_residual_ionosphere,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Records whether the integer lattice result passed its fixed-status condition.
pub enum IntegerStatus {
    /// The lattice resolver accepted an integer-fixed candidate.
    Fixed,
    /// The lattice resolver returned candidates without accepting an integer fix.
    NotFixed,
}

/// Frozen ambiguity-search metadata returned with a fixed PPP solution.
#[derive(Debug, Clone, PartialEq)]
pub struct AmbiguitySearch {
    /// Ambiguity ids in the exact order supplied to the integer lattice search.
    pub order: Vec<String>,
    /// Float ambiguity estimates converted from meters to cycles using the
    /// configured wavelength and offset.
    pub float_cycles: BTreeMap<String, f64>,
    /// Ambiguity covariance returned by the lattice resolver, in cycles squared
    /// and ordered by [`Self::order`].
    pub covariance_cycles: Vec<Vec<f64>>,
    /// Inverse ambiguity covariance returned by the lattice resolver in the same
    /// order as [`Self::covariance_cycles`].
    pub covariance_inverse_cycles: Vec<Vec<f64>>,
}

/// Integer-search summary returned with a fixed PPP solution.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedIntegerMetadata {
    /// Whether the integer lattice resolver accepted the selected candidate.
    pub integer_status: IntegerStatus,
    /// Candidate score ratio returned by the integer lattice resolver.
    pub integer_ratio: f64,
    /// Score of the best integer candidate.
    pub integer_best_score: f64,
    /// Score of the second candidate, when the resolver evaluated one.
    pub integer_second_best_score: Option<f64>,
    /// Number of integer candidates evaluated by the resolver.
    pub integer_candidates: usize,
    /// Frozen order, float values, and covariance matrices from the lattice
    /// search.
    pub ambiguity_search: AmbiguitySearch,
}

/// Static integer-fixed PPP solution.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedSolution {
    /// Estimated static receiver position in ECEF/ITRF metres.
    pub position_m: [f64; 3],
    /// Posterior covariance of [`Self::position_m`] after fixed ambiguities are
    /// held and the per-epoch receiver clocks are eliminated, scaled by
    /// [`Self::position_covariance_scale_factor`].
    ///
    /// The ECEF and ENU matrices follow [`PositionCovariance`]. The raw residual
    /// factor is reported in [`Self::posterior_variance_factor`]. This can still
    /// be optimistic when residuals are temporally correlated or dominated by
    /// multipath, antenna, loading, or other unmodelled systematics. When the
    /// residual factor is below 1.0, this covariance is smaller than the formal
    /// covariance.
    pub position_covariance: PositionCovariance,
    /// Unscaled formal covariance from the final weighted normal matrix.
    ///
    /// This is the unit-variance covariance before applying
    /// [`Self::posterior_variance_factor`].
    pub formal_position_covariance: PositionCovariance,
    /// A-posteriori unit-variance factor, computed as weighted SSR divided by
    /// unreduced degrees of freedom. Eliminated receiver clocks still count as
    /// estimated parameters in the denominator.
    pub posterior_variance_factor: f64,
    /// Multiplier applied to [`Self::formal_position_covariance`] to produce
    /// [`Self::position_covariance`]. This equals
    /// [`Self::posterior_variance_factor`].
    pub position_covariance_scale_factor: f64,
    /// Position covariance with temporal-correlation sample-count deflation.
    ///
    /// This additive field does not change [`Self::position_covariance`]. It is
    /// the formal covariance multiplied by the larger of 1.0 and the posterior
    /// variance factor, and then by
    /// [`Self::temporal_correlation`]'s variance inflation factor. The floor
    /// keeps this conservative covariance from shrinking below the formal
    /// normal-equation covariance.
    pub temporal_position_covariance: PositionCovariance,
    /// Multiplier applied to [`Self::formal_position_covariance`] to produce
    /// [`Self::temporal_position_covariance`].
    pub temporal_position_covariance_scale_factor: f64,
    /// Residual temporal-correlation estimate used by
    /// [`Self::temporal_position_covariance`].
    pub temporal_correlation: TemporalCorrelationSummary,
    /// Final receiver-clock states in meters from the ambiguity-held fixed
    /// re-solve, one for each solved epoch, in the order of
    /// [`Self::solved_epoch_indices`].
    pub epoch_clocks_m: Vec<f64>,
    /// Selected integer ambiguity values keyed by ambiguity id and retained in
    /// the integer-search order.
    pub fixed_ambiguities_cycles: BTreeMap<String, i64>,
    /// Selected integer ambiguities converted to meters as offset plus integer
    /// cycles times wavelength and held in every fixed phase row.
    pub fixed_ambiguities_m: BTreeMap<String, f64>,
    /// Post-combination ionosphere states from the fixed re-solve when enabled;
    /// otherwise this is an empty map.
    pub residual_ionosphere_m: BTreeMap<String, f64>,
    /// Fixed re-solve residual ZTD when ZTD estimation is enabled; otherwise
    /// this is `None`.
    pub ztd_residual_m: Option<f64>,
    /// Estimated north horizontal troposphere gradient, in metres.
    pub tropo_gradient_north_m: Option<f64>,
    /// Estimated east horizontal troposphere gradient, in metres.
    pub tropo_gradient_east_m: Option<f64>,
    /// Posterior covariance of north/east troposphere gradients, in square
    /// metres, scaled by [`Self::position_covariance_scale_factor`].
    pub tropo_gradient_covariance_m2: Option<[[f64; 2]; 2]>,
    /// Unscaled formal covariance of north/east troposphere gradients.
    pub formal_tropo_gradient_covariance_m2: Option<[[f64; 2]; 2]>,
    /// The float solution the integers were fixed from: the caller's, or the float arc
    /// solved again when the fixed solve changed the arc: an SSR/HAS bias stopped holding
    /// during the fixed re-solve, an excluded observation held at the fixed position and
    /// was admitted again, or the fixed arc observed an ambiguity the float solution did
    /// not solve. The re-solve uses the fixed configuration's weights, troposphere,
    /// corrections, residual ionosphere setting and elevation cutoff, with the caller's
    /// float solve options and residual screen setting; the residual screen runs again
    /// over the whole arc and its removals replace the caller's.
    pub float_solution: FloatSolution,
    /// Fixed re-solve residual rows in epoch and observation traversal order.
    pub residuals_m: Vec<FloatResidual>,
    /// Ambiguity ids from the integer-search order, exposed in that same order.
    pub used_sats: Vec<String>,
    /// Iteration counter from the ambiguity-held fixed re-solve.
    pub iterations: usize,
    /// True when the fixed re-solve met all active state tolerances.
    pub converged: bool,
    /// Records whether the fixed re-solve met state tolerances or reached its
    /// iteration cap.
    pub status: FloatStatus,
    /// Root-mean-square of the fixed code residuals.
    pub code_rms_m: f64,
    /// Root-mean-square of the fixed phase residuals.
    pub phase_rms_m: f64,
    /// Root-mean-square after fixed code and phase residuals are multiplied by
    /// their row weights.
    pub weighted_rms_m: f64,
    /// Integer-search metadata produced by the search that supplied the fixed
    /// ambiguity values.
    pub integer: FixedIntegerMetadata,
    /// Observations left out of the fixed re-solve because an SSR/HAS bias the
    /// corrections require was not resolved for them, in epoch and observation order.
    pub ssr_bias_exclusions: Vec<SsrBiasExclusion>,
    /// Input epoch index of each epoch of the fixed re-solve, ascending, paired with
    /// [`Self::epoch_clocks_m`]. An input epoch left with no observations by the elevation
    /// cutoff or SSR bias exclusion is not solved and is absent here.
    pub solved_epoch_indices: Vec<usize>,
}

/// Static fixed PPP solve errors.
#[derive(Debug, Clone, PartialEq)]
pub enum FixedSolveError {
    /// A float prerequisite, validation, prediction, correction, or normal
    /// equation operation failed before or during fixed processing.
    Float(FloatSolveError),
    /// The LAMBDA integer lattice search returned an [`IlsError`].
    Integer(IlsError),
    /// The ambiguity id has no configured carrier wavelength.
    MissingWavelength(String),
    /// The ambiguity id has no configured meter offset.
    MissingOffset(String),
    /// A fixed ambiguity was expected but absent from the fixed result path.
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
