//! Clock-stability estimators from IEEE Std 1139-2008 and NIST SP 1065.
//!
//! The natural RINEX receiver-clock phase input is
//! `crates/sidereon-core/src/rinex_obs/mod.rs:ObsEpoch::rcv_clock_offset_s`.
//! [`receiver_clock_phase_deviations`] exposes that field as a phase-deviation
//! series in seconds.
//!
//! Gap handling is explicit. [`GapPolicy::Reject`] requires a contiguous series.
//! [`GapPolicy::OmitTerms`] keeps the series indexed and excludes every estimator
//! term whose required phase samples cross a missing sample.

use crate::estimation::mad_spread;
use crate::rinex::observations::RinexObs;

const SQRT_3: f64 = 1.732_050_807_568_877_2;
const DEFAULT_POWER_LAW_MIN_POINTS_PER_OCTAVE: usize = 2;
const DEFAULT_POWER_LAW_SLOPE_TOLERANCE: f64 = 0.125;

/// Tagged input samples for Allan-family estimators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AllanSeries<'a> {
    /// Phase deviations in seconds, sampled every `tau0_s`.
    PhaseSeconds(&'a [f64]),
    /// Fractional-frequency samples, dimensionless, sampled every `tau0_s`.
    FractionalFrequency(&'a [f64]),
    /// Phase deviations in seconds with missing samples.
    PhaseSecondsWithGaps(&'a [Option<f64>]),
    /// Fractional-frequency samples with missing samples.
    FractionalFrequencyWithGaps(&'a [Option<f64>]),
}

/// Averaging-factor grid for requested estimator points.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TauGrid {
    /// `m = 1, 2, 4, 8, ...` while the estimator has at least one term.
    #[default]
    Octave,
    /// `m = 1..=m_max` while the estimator has at least one term.
    All,
    /// Caller-supplied averaging factors.
    Explicit(Vec<usize>),
}

/// Missing-sample policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GapPolicy {
    /// Reject any missing sample before estimation.
    #[default]
    Reject,
    /// Exclude estimator terms whose phase samples cross a missing sample.
    OmitTerms,
}

/// Estimators available through [`compute_allan_deviations`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllanEstimatorSet {
    /// Plain non-overlapping Allan deviation.
    pub adev: bool,
    /// Fully overlapping Allan deviation.
    pub overlapping_adev: bool,
    /// Modified Allan deviation.
    pub mdev: bool,
    /// Overlapping Hadamard deviation.
    pub hdev: bool,
    /// Time deviation derived from MDEV.
    pub tdev: bool,
}

impl AllanEstimatorSet {
    /// No estimators selected.
    pub const fn none() -> Self {
        Self {
            adev: false,
            overlapping_adev: false,
            mdev: false,
            hdev: false,
            tdev: false,
        }
    }

    /// Standard overlapping estimators plus TDEV.
    pub const fn standard() -> Self {
        Self {
            adev: false,
            overlapping_adev: true,
            mdev: true,
            hdev: true,
            tdev: true,
        }
    }

    /// Every implemented estimator.
    pub const fn all() -> Self {
        Self {
            adev: true,
            overlapping_adev: true,
            mdev: true,
            hdev: true,
            tdev: true,
        }
    }

    fn is_empty(self) -> bool {
        !self.adev && !self.overlapping_adev && !self.mdev && !self.hdev && !self.tdev
    }
}

impl Default for AllanEstimatorSet {
    fn default() -> Self {
        Self::standard()
    }
}

/// Options for the combined Allan-family driver.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct AllanOptions {
    /// Which estimators to compute.
    pub estimators: AllanEstimatorSet,
    /// Averaging-factor grid.
    pub tau_grid: TauGrid,
    /// Missing-sample policy.
    pub gap_policy: GapPolicy,
}

/// Input package for [`compute_allan_deviations`].
#[derive(Debug, Clone, PartialEq)]
pub struct AllanInput<'a> {
    /// Tagged sample series.
    pub series: AllanSeries<'a>,
    /// Basic sampling interval in seconds.
    pub tau0_s: f64,
    /// Estimator, tau-grid, and gap options.
    pub options: AllanOptions,
}

/// One estimator curve.
#[derive(Debug, Clone, PartialEq)]
pub struct AllanResult {
    /// Averaging times, seconds.
    pub tau_s: Vec<f64>,
    /// Deviation value for each averaging time.
    pub deviation: Vec<f64>,
    /// Number of estimator terms used at each averaging time.
    pub n: Vec<usize>,
}

impl AllanResult {
    fn new() -> Self {
        Self {
            tau_s: Vec::new(),
            deviation: Vec::new(),
            n: Vec::new(),
        }
    }

    fn push(&mut self, tau_s: f64, deviation: f64, n: usize) {
        self.tau_s.push(tau_s);
        self.deviation.push(deviation);
        self.n.push(n);
    }

    /// Number of tau points in the curve.
    pub fn len(&self) -> usize {
        self.tau_s.len()
    }

    /// Whether the curve has no tau points.
    pub fn is_empty(&self) -> bool {
        self.tau_s.is_empty()
    }
}

/// Combined output from [`compute_allan_deviations`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AllanDeviationCurves {
    /// Plain non-overlapping Allan deviation, if requested.
    pub adev: Option<AllanResult>,
    /// Fully overlapping Allan deviation, if requested.
    pub overlapping_adev: Option<AllanResult>,
    /// Modified Allan deviation, if requested.
    pub mdev: Option<AllanResult>,
    /// Overlapping Hadamard deviation, if requested.
    pub hdev: Option<AllanResult>,
    /// Time deviation, if requested.
    pub tdev: Option<AllanResult>,
}

/// IEEE 1139 fractional-frequency PSD power-law noise type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PowerLawNoiseType {
    /// Random-walk frequency modulation, `S_y(f) = h_-2 f^-2`.
    RandomWalkFM,
    /// Flicker frequency modulation, `S_y(f) = h_-1 f^-1`.
    FlickerFM,
    /// White frequency modulation, `S_y(f) = h_0`.
    WhiteFM,
    /// Flicker phase modulation, `S_y(f) = h_1 f`.
    FlickerPM,
    /// White phase modulation, `S_y(f) = h_2 f^2`.
    WhitePM,
}

impl PowerLawNoiseType {
    /// Spectral exponent `alpha` in `S_y(f) = h_alpha f^alpha`.
    pub const fn alpha(self) -> i32 {
        match self {
            Self::RandomWalkFM => -2,
            Self::FlickerFM => -1,
            Self::WhiteFM => 0,
            Self::FlickerPM => 1,
            Self::WhitePM => 2,
        }
    }

    /// Index into [`PowerLawNoiseFit::coefficients`].
    pub const fn coefficient_index(self) -> usize {
        match self {
            Self::RandomWalkFM => 0,
            Self::FlickerFM => 1,
            Self::WhiteFM => 2,
            Self::FlickerPM => 3,
            Self::WhitePM => 4,
        }
    }
}

/// Exact rational log-log slope of ADEV versus tau for a power-law type.
pub fn allan_deviation_power_law_slope(noise_type: PowerLawNoiseType) -> f64 {
    match noise_type {
        PowerLawNoiseType::RandomWalkFM => 0.5,
        PowerLawNoiseType::FlickerFM => 0.0,
        PowerLawNoiseType::WhiteFM => -0.5,
        PowerLawNoiseType::FlickerPM | PowerLawNoiseType::WhitePM => -1.0,
    }
}

/// Exact rational log-log slope of MDEV versus tau for a power-law type.
pub fn modified_allan_deviation_power_law_slope(noise_type: PowerLawNoiseType) -> f64 {
    match noise_type {
        PowerLawNoiseType::RandomWalkFM => 0.5,
        PowerLawNoiseType::FlickerFM => 0.0,
        PowerLawNoiseType::WhiteFM => -0.5,
        PowerLawNoiseType::FlickerPM => -1.0,
        PowerLawNoiseType::WhitePM => -1.5,
    }
}

/// Exact tau exponent of Allan variance for a power-law type.
pub fn allan_variance_power_law_tau_exponent(noise_type: PowerLawNoiseType) -> i32 {
    match noise_type {
        PowerLawNoiseType::RandomWalkFM => 1,
        PowerLawNoiseType::FlickerFM => 0,
        PowerLawNoiseType::WhiteFM => -1,
        PowerLawNoiseType::FlickerPM | PowerLawNoiseType::WhitePM => -2,
    }
}

/// Options for per-octave power-law identification and coefficient fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PowerLawNoiseOptions {
    /// Minimum tau points required before an octave can be classified.
    pub min_points_per_octave: usize,
    /// Maximum absolute slope error allowed for an exact-rational noise type.
    pub slope_tolerance: f64,
    /// Maximum robust local-slope scatter allowed before an octave is ambiguous.
    pub scatter_tolerance: f64,
    /// Basic sample interval used by the deviation calculation, seconds.
    pub basic_tau_s: f64,
    /// Upper measurement bandwidth, hertz, used by phase-modulation conversions.
    pub measurement_bandwidth_hz: f64,
}

impl PowerLawNoiseOptions {
    /// Construct options with the default octave point count and slope tolerances.
    pub const fn new(basic_tau_s: f64, measurement_bandwidth_hz: f64) -> Self {
        Self {
            min_points_per_octave: DEFAULT_POWER_LAW_MIN_POINTS_PER_OCTAVE,
            slope_tolerance: DEFAULT_POWER_LAW_SLOPE_TOLERANCE,
            scatter_tolerance: DEFAULT_POWER_LAW_SLOPE_TOLERANCE,
            basic_tau_s,
            measurement_bandwidth_hz,
        }
    }

    /// Construct options using the Nyquist bandwidth for samples spaced by `basic_tau_s`.
    pub fn sampled_at_nyquist(basic_tau_s: f64) -> Self {
        Self::new(basic_tau_s, 0.5 / basic_tau_s)
    }
}

/// Why a tau octave could not receive a dominant noise type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerLawOctaveFlag {
    /// Fewer than [`PowerLawNoiseOptions::min_points_per_octave`] tau points were present.
    UnderSampled,
    /// A zero deviation made the log-log slope undefined.
    DegenerateDeviation,
    /// MDEV did not have enough tau points to separate phase-modulation types.
    MissingModifiedAllan,
}

/// Dominant noise decision for one tau octave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerLawOctaveDominance {
    /// One power-law type was identified.
    Dominant(PowerLawNoiseType),
    /// The octave contains conflicting or off-table slopes.
    Ambiguous,
    /// The octave is not classifiable because required data are absent.
    Flagged(PowerLawOctaveFlag),
}

/// Per-octave power-law classification from local ADEV and MDEV slopes.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerLawOctave {
    /// First tau in the octave, seconds.
    pub tau_start_s: f64,
    /// Last tau used in the octave, seconds.
    pub tau_end_s: f64,
    /// Number of ADEV tau points used for the octave slope.
    pub point_count: usize,
    /// Fitted ADEV log-log slope for this octave, if available.
    pub adev_slope: Option<f64>,
    /// Fitted MDEV log-log slope for this octave, if available.
    pub mdev_slope: Option<f64>,
    /// Robust scatter of adjacent ADEV slopes inside the octave, if available.
    pub slope_scatter: Option<f64>,
    /// Dominant type, ambiguity, or faithful-bound flag for the octave.
    pub dominance: PowerLawOctaveDominance,
}

/// Consecutive tau span that supports one fitted power-law coefficient.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerLawNoiseRegion {
    /// Noise type identified across the region.
    pub noise_type: PowerLawNoiseType,
    /// First tau in the region, seconds.
    pub tau_start_s: f64,
    /// Last tau in the region, seconds.
    pub tau_end_s: f64,
    /// Number of classified octaves merged into this region.
    pub octave_count: usize,
    /// Number of deviation points used in the coefficient fit.
    pub point_count: usize,
    /// Mean local slope from the statistic used for classification.
    pub mean_slope: f64,
    /// Fitted PSD coefficient for this region.
    pub coefficient: f64,
}

/// IEEE 1139 power-law noise identification and coefficient fit.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerLawNoiseFit {
    /// Dominant classification for each tau octave.
    pub dominant_per_octave: Vec<PowerLawOctave>,
    /// PSD coefficients `[h_-2, h_-1, h_0, h_1, h_2]`.
    ///
    /// Entries are `NaN` when no faithful identified region supports that coefficient.
    pub coefficients: [f64; 5],
    /// Consecutive identified tau regions used by the coefficient fit.
    pub regions: Vec<PowerLawNoiseRegion>,
}

/// Error from power-law clock-noise identification setup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PowerLawNoiseError {
    /// An option field is outside its accepted range.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Reason the field is invalid.
        reason: &'static str,
    },
    /// A supplied deviation curve is structurally invalid.
    InvalidCurve {
        /// Curve label.
        curve: &'static str,
        /// Reason the curve is invalid.
        reason: &'static str,
    },
}

impl core::fmt::Display for PowerLawNoiseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidOptions { field, reason } => {
                write!(f, "invalid power-law option {field}: {reason}")
            }
            Self::InvalidCurve { curve, reason } => {
                write!(f, "invalid {curve} curve: {reason}")
            }
        }
    }
}

impl std::error::Error for PowerLawNoiseError {}

/// Estimator identifier used in errors and options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllanEstimator {
    /// Plain non-overlapping Allan deviation.
    Adev,
    /// Fully overlapping Allan deviation.
    OverlappingAdev,
    /// Modified Allan deviation.
    Mdev,
    /// Overlapping Hadamard deviation.
    Hdev,
    /// Time deviation.
    Tdev,
}

impl AllanEstimator {
    fn label(self) -> &'static str {
        match self {
            Self::Adev => "ADEV",
            Self::OverlappingAdev => "OADEV",
            Self::Mdev => "MDEV",
            Self::Hdev => "HDEV",
            Self::Tdev => "TDEV",
        }
    }
}

/// Error from Allan-family estimator setup or evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum AllanError {
    /// The input series has no samples.
    EmptySeries,
    /// The basic sampling interval is not finite and positive.
    InvalidTau0 { tau0_s: f64 },
    /// No estimator was requested.
    NoEstimators,
    /// An explicit tau grid was empty.
    EmptyTauGrid,
    /// Averaging factors must be positive.
    InvalidAveragingFactor { averaging_factor: usize },
    /// A requested estimator has no valid term for this sample count.
    TooFewSamples {
        estimator: AllanEstimator,
        averaging_factor: usize,
        available_phase_samples: usize,
    },
    /// A sample was not finite.
    NonFiniteSample { index: usize },
    /// A missing sample was present under [`GapPolicy::Reject`].
    Gap { index: usize },
    /// The computed tau was not finite.
    NonFiniteTau {
        estimator: AllanEstimator,
        averaging_factor: usize,
    },
    /// The computed deviation was not finite.
    NonFiniteDeviation {
        estimator: AllanEstimator,
        averaging_factor: usize,
    },
}

impl core::fmt::Display for AllanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptySeries => write!(f, "Allan series is empty"),
            Self::InvalidTau0 { tau0_s } => {
                write!(f, "tau0_s must be finite and positive, got {tau0_s}")
            }
            Self::NoEstimators => write!(f, "no Allan estimators selected"),
            Self::EmptyTauGrid => write!(f, "explicit tau grid is empty"),
            Self::InvalidAveragingFactor { averaging_factor } => {
                write!(
                    f,
                    "averaging factor must be positive, got {averaging_factor}"
                )
            }
            Self::TooFewSamples {
                estimator,
                averaging_factor,
                available_phase_samples,
            } => write!(
                f,
                "{} has no valid terms for averaging factor {} with {} phase samples",
                estimator.label(),
                averaging_factor,
                available_phase_samples
            ),
            Self::NonFiniteSample { index } => {
                write!(f, "sample {index} is not finite")
            }
            Self::Gap { index } => {
                write!(f, "sample {index} is missing")
            }
            Self::NonFiniteTau {
                estimator,
                averaging_factor,
            } => write!(
                f,
                "{} tau is not finite for averaging factor {}",
                estimator.label(),
                averaging_factor
            ),
            Self::NonFiniteDeviation {
                estimator,
                averaging_factor,
            } => write!(
                f,
                "{} deviation is not finite for averaging factor {}",
                estimator.label(),
                averaging_factor
            ),
        }
    }
}

impl std::error::Error for AllanError {}

/// Extract RINEX receiver-clock offsets as phase deviations in seconds.
///
/// Event epochs (`flag > 1`) are returned as gaps. The source field is
/// `crates/sidereon-core/src/rinex_obs/mod.rs:ObsEpoch::rcv_clock_offset_s`.
pub fn receiver_clock_phase_deviations(obs: &RinexObs) -> Vec<Option<f64>> {
    obs.epochs()
        .iter()
        .map(|epoch| {
            if epoch.flag > 1 {
                None
            } else {
                epoch.rcv_clock_offset_s
            }
        })
        .collect()
}

/// Compute the requested Allan-family curves.
pub fn compute_allan_deviations(
    input: &AllanInput<'_>,
) -> Result<AllanDeviationCurves, AllanError> {
    if input.options.estimators.is_empty() {
        return Err(AllanError::NoEstimators);
    }

    let phase = prepare_phase(input.series, input.tau0_s, input.options.gap_policy)?;
    let mut curves = AllanDeviationCurves::default();

    if input.options.estimators.adev {
        curves.adev = Some(compute_curve(
            &phase,
            input.tau0_s,
            &input.options.tau_grid,
            AllanEstimator::Adev,
        )?);
    }
    if input.options.estimators.overlapping_adev {
        curves.overlapping_adev = Some(compute_curve(
            &phase,
            input.tau0_s,
            &input.options.tau_grid,
            AllanEstimator::OverlappingAdev,
        )?);
    }
    if input.options.estimators.mdev {
        curves.mdev = Some(compute_curve(
            &phase,
            input.tau0_s,
            &input.options.tau_grid,
            AllanEstimator::Mdev,
        )?);
    }
    if input.options.estimators.hdev {
        curves.hdev = Some(compute_curve(
            &phase,
            input.tau0_s,
            &input.options.tau_grid,
            AllanEstimator::Hdev,
        )?);
    }
    if input.options.estimators.tdev {
        curves.tdev = Some(compute_curve(
            &phase,
            input.tau0_s,
            &input.options.tau_grid,
            AllanEstimator::Tdev,
        )?);
    }

    Ok(curves)
}

/// Plain non-overlapping Allan deviation for explicit averaging factors.
pub fn allan_deviation(
    series: AllanSeries<'_>,
    tau0_s: f64,
    averaging_factors: &[usize],
) -> Result<AllanResult, AllanError> {
    compute_explicit(series, tau0_s, averaging_factors, AllanEstimator::Adev)
}

/// Fully overlapping Allan deviation for explicit averaging factors.
pub fn overlapping_adev(
    series: AllanSeries<'_>,
    tau0_s: f64,
    averaging_factors: &[usize],
) -> Result<AllanResult, AllanError> {
    compute_explicit(
        series,
        tau0_s,
        averaging_factors,
        AllanEstimator::OverlappingAdev,
    )
}

/// Modified Allan deviation for explicit averaging factors.
pub fn modified_adev(
    series: AllanSeries<'_>,
    tau0_s: f64,
    averaging_factors: &[usize],
) -> Result<AllanResult, AllanError> {
    compute_explicit(series, tau0_s, averaging_factors, AllanEstimator::Mdev)
}

/// Overlapping Hadamard deviation for explicit averaging factors.
pub fn hadamard_deviation(
    series: AllanSeries<'_>,
    tau0_s: f64,
    averaging_factors: &[usize],
) -> Result<AllanResult, AllanError> {
    compute_explicit(series, tau0_s, averaging_factors, AllanEstimator::Hdev)
}

/// Time deviation for explicit averaging factors.
pub fn time_deviation(
    series: AllanSeries<'_>,
    tau0_s: f64,
    averaging_factors: &[usize],
) -> Result<AllanResult, AllanError> {
    compute_explicit(series, tau0_s, averaging_factors, AllanEstimator::Tdev)
}

/// Identify per-octave power-law noise and fit PSD coefficients.
///
/// The `adev` argument may be plain or overlapping ADEV. The function consumes
/// the supplied deviation curves as-is and does not recompute any Allan-family
/// statistic. ADEV identifies random-walk FM, flicker FM, and white FM regions.
/// For phase modulation, where ADEV has the same `tau^-1` slope for flicker PM
/// and white PM, the matching MDEV octave supplies the separating slope.
///
/// Coefficients are returned as `[h_-2, h_-1, h_0, h_1, h_2]`. FM coefficients
/// are fitted from ADEV using the IEEE 1139 Allan-variance conversion constants.
/// PM coefficients are fitted from MDEV using the modified-Allan conversion
/// constants derived from the NIST SP 1065 MVAR transfer function, with
/// `measurement_bandwidth_hz` and `basic_tau_s` applied to white PM.
pub fn fit_power_law_noise(
    adev: &AllanResult,
    mdev: &AllanResult,
    options: PowerLawNoiseOptions,
) -> Result<PowerLawNoiseFit, PowerLawNoiseError> {
    validate_power_law_options(options)?;
    validate_power_law_curve("ADEV", adev)?;
    validate_power_law_curve("MDEV", mdev)?;

    let dominant_per_octave = classify_power_law_octaves(adev, mdev, options);
    let regions = build_power_law_regions(&dominant_per_octave, adev, mdev, options);
    let mut coefficients = [f64::NAN; 5];

    for noise_type in POWER_LAW_NOISE_TYPES {
        let coefficient = fit_coefficient_for_type(noise_type, &regions, adev, mdev, options);
        if let Some(coefficient) = coefficient {
            coefficients[noise_type.coefficient_index()] = coefficient;
        }
    }

    Ok(PowerLawNoiseFit {
        dominant_per_octave,
        coefficients,
        regions,
    })
}

fn compute_explicit(
    series: AllanSeries<'_>,
    tau0_s: f64,
    averaging_factors: &[usize],
    estimator: AllanEstimator,
) -> Result<AllanResult, AllanError> {
    let phase = prepare_phase(series, tau0_s, GapPolicy::Reject)?;
    compute_curve(
        &phase,
        tau0_s,
        &TauGrid::Explicit(averaging_factors.to_vec()),
        estimator,
    )
}

const POWER_LAW_NOISE_TYPES: [PowerLawNoiseType; 5] = [
    PowerLawNoiseType::RandomWalkFM,
    PowerLawNoiseType::FlickerFM,
    PowerLawNoiseType::WhiteFM,
    PowerLawNoiseType::FlickerPM,
    PowerLawNoiseType::WhitePM,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdevSlopeClass {
    Noise(PowerLawNoiseType),
    PhaseModulation,
    Ambiguous,
}

fn validate_power_law_options(options: PowerLawNoiseOptions) -> Result<(), PowerLawNoiseError> {
    if options.min_points_per_octave < 2 {
        return Err(PowerLawNoiseError::InvalidOptions {
            field: "min_points_per_octave",
            reason: "must be at least 2",
        });
    }
    if !(options.slope_tolerance.is_finite() && options.slope_tolerance > 0.0) {
        return Err(PowerLawNoiseError::InvalidOptions {
            field: "slope_tolerance",
            reason: "must be finite and positive",
        });
    }
    if !(options.scatter_tolerance.is_finite() && options.scatter_tolerance >= 0.0) {
        return Err(PowerLawNoiseError::InvalidOptions {
            field: "scatter_tolerance",
            reason: "must be finite and nonnegative",
        });
    }
    if !(options.basic_tau_s.is_finite() && options.basic_tau_s > 0.0) {
        return Err(PowerLawNoiseError::InvalidOptions {
            field: "basic_tau_s",
            reason: "must be finite and positive",
        });
    }
    if !(options.measurement_bandwidth_hz.is_finite() && options.measurement_bandwidth_hz > 0.0) {
        return Err(PowerLawNoiseError::InvalidOptions {
            field: "measurement_bandwidth_hz",
            reason: "must be finite and positive",
        });
    }
    Ok(())
}

fn validate_power_law_curve(
    curve_name: &'static str,
    curve: &AllanResult,
) -> Result<(), PowerLawNoiseError> {
    if curve.tau_s.len() != curve.deviation.len() || curve.tau_s.len() != curve.n.len() {
        return Err(PowerLawNoiseError::InvalidCurve {
            curve: curve_name,
            reason: "tau, deviation, and term-count lengths must match",
        });
    }

    let mut previous_tau_s = None;
    for (index, (&tau_s, &deviation)) in curve.tau_s.iter().zip(curve.deviation.iter()).enumerate()
    {
        if !(tau_s.is_finite() && tau_s > 0.0) {
            return Err(PowerLawNoiseError::InvalidCurve {
                curve: curve_name,
                reason: "tau values must be finite and positive",
            });
        }
        if previous_tau_s.is_some_and(|previous| tau_s <= previous) {
            return Err(PowerLawNoiseError::InvalidCurve {
                curve: curve_name,
                reason: "tau values must be strictly increasing",
            });
        }
        previous_tau_s = Some(tau_s);

        if !(deviation.is_finite() && deviation >= 0.0) {
            return Err(PowerLawNoiseError::InvalidCurve {
                curve: curve_name,
                reason: "deviations must be finite and nonnegative",
            });
        }
        if curve.n[index] == 0 {
            return Err(PowerLawNoiseError::InvalidCurve {
                curve: curve_name,
                reason: "term counts must be positive",
            });
        }
    }

    Ok(())
}

fn classify_power_law_octaves(
    adev: &AllanResult,
    mdev: &AllanResult,
    options: PowerLawNoiseOptions,
) -> Vec<PowerLawOctave> {
    let mut octaves = Vec::new();
    let mut start = 0usize;
    while start < adev.tau_s.len() {
        let end = octave_end_index(&adev.tau_s, start);
        let point_count = end + 1 - start;
        if point_count < options.min_points_per_octave {
            let tau_start_s = adev.tau_s[start];
            octaves.push(PowerLawOctave {
                tau_start_s,
                tau_end_s: tau_start_s,
                point_count,
                adev_slope: None,
                mdev_slope: None,
                slope_scatter: None,
                dominance: PowerLawOctaveDominance::Flagged(PowerLawOctaveFlag::UnderSampled),
            });
            start += 1;
            continue;
        }

        octaves.push(classify_power_law_window(start, end, adev, mdev, options));
        if end >= adev.tau_s.len() - 1 {
            break;
        }
        start = end;
    }
    octaves
}

fn octave_end_index(tau_s: &[f64], start: usize) -> usize {
    let tau_limit_s = tau_s[start] * 2.0;
    let tolerance = tau_limit_s.abs().max(1.0) * 32.0 * f64::EPSILON;
    let mut end = start;
    while end + 1 < tau_s.len() && tau_s[end + 1] <= tau_limit_s + tolerance {
        end += 1;
    }
    end
}

fn classify_power_law_window(
    start: usize,
    end: usize,
    adev: &AllanResult,
    mdev: &AllanResult,
    options: PowerLawNoiseOptions,
) -> PowerLawOctave {
    let tau_start_s = adev.tau_s[start];
    let tau_end_s = adev.tau_s[end];
    let point_count = end + 1 - start;

    let Some(adev_slope) = log_log_slope_for_curve(adev, start, end) else {
        return PowerLawOctave {
            tau_start_s,
            tau_end_s,
            point_count,
            adev_slope: None,
            mdev_slope: None,
            slope_scatter: None,
            dominance: PowerLawOctaveDominance::Flagged(PowerLawOctaveFlag::DegenerateDeviation),
        };
    };

    let adjacent_slopes = adjacent_log_log_slopes(adev, start, end);
    let slope_scatter = robust_slope_scatter(&adjacent_slopes);
    let scatter_is_ambiguous =
        slope_scatter.is_some_and(|scatter| scatter > options.scatter_tolerance);
    let adev_class = classify_adev_slope(adev_slope, options.slope_tolerance);
    let local_adev_consistent =
        local_adev_slopes_consistent(&adjacent_slopes, adev_class, options.slope_tolerance);

    let (mdev_slope, mdev_scatter, local_mdev_consistent) =
        mdev_slope_in_range(mdev, tau_start_s, tau_end_s, options);

    let dominance = if scatter_is_ambiguous || !local_adev_consistent {
        PowerLawOctaveDominance::Ambiguous
    } else {
        match adev_class {
            AdevSlopeClass::Noise(noise_type) => {
                if let Some(mdev_slope) = mdev_slope {
                    match classify_mdev_slope(mdev_slope, options.slope_tolerance) {
                        Some(mdev_type) if mdev_type != noise_type => {
                            PowerLawOctaveDominance::Ambiguous
                        }
                        None => PowerLawOctaveDominance::Ambiguous,
                        Some(_) => PowerLawOctaveDominance::Dominant(noise_type),
                    }
                } else {
                    PowerLawOctaveDominance::Dominant(noise_type)
                }
            }
            AdevSlopeClass::PhaseModulation => match mdev_slope {
                Some(mdev_slope)
                    if !mdev_scatter.is_some_and(|scatter| scatter > options.scatter_tolerance)
                        && local_mdev_consistent =>
                {
                    match classify_mdev_slope(mdev_slope, options.slope_tolerance) {
                        Some(PowerLawNoiseType::FlickerPM) => {
                            PowerLawOctaveDominance::Dominant(PowerLawNoiseType::FlickerPM)
                        }
                        Some(PowerLawNoiseType::WhitePM) => {
                            PowerLawOctaveDominance::Dominant(PowerLawNoiseType::WhitePM)
                        }
                        _ => PowerLawOctaveDominance::Ambiguous,
                    }
                }
                Some(_) => PowerLawOctaveDominance::Ambiguous,
                None => PowerLawOctaveDominance::Flagged(PowerLawOctaveFlag::MissingModifiedAllan),
            },
            AdevSlopeClass::Ambiguous => PowerLawOctaveDominance::Ambiguous,
        }
    };

    PowerLawOctave {
        tau_start_s,
        tau_end_s,
        point_count,
        adev_slope: Some(adev_slope),
        mdev_slope,
        slope_scatter,
        dominance,
    }
}

fn log_log_slope_for_curve(curve: &AllanResult, start: usize, end: usize) -> Option<f64> {
    if end <= start {
        return None;
    }
    let tau = &curve.tau_s[start..=end];
    let deviation = &curve.deviation[start..=end];
    if deviation.iter().any(|&value| value <= 0.0) {
        return None;
    }

    let n = tau.len() as f64;
    let (sum_x, sum_y) = tau
        .iter()
        .zip(deviation.iter())
        .fold((0.0, 0.0), |(sum_x, sum_y), (&tau_s, &sigma)| {
            (sum_x + libm::log(tau_s), sum_y + libm::log(sigma))
        });
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let (num, den) =
        tau.iter()
            .zip(deviation.iter())
            .fold((0.0, 0.0), |(num, den), (&tau_s, &sigma)| {
                let dx = libm::log(tau_s) - mean_x;
                let dy = libm::log(sigma) - mean_y;
                (num + dx * dy, den + dx * dx)
            });

    if den > 0.0 {
        Some(num / den)
    } else {
        None
    }
}

fn adjacent_log_log_slopes(curve: &AllanResult, start: usize, end: usize) -> Vec<f64> {
    let mut slopes = Vec::new();
    for index in start..end {
        let tau0 = curve.tau_s[index];
        let tau1 = curve.tau_s[index + 1];
        let sigma0 = curve.deviation[index];
        let sigma1 = curve.deviation[index + 1];
        if sigma0 <= 0.0 || sigma1 <= 0.0 {
            continue;
        }
        let denominator = libm::log(tau1) - libm::log(tau0);
        if denominator > 0.0 {
            slopes.push((libm::log(sigma1) - libm::log(sigma0)) / denominator);
        }
    }
    slopes
}

fn robust_slope_scatter(slopes: &[f64]) -> Option<f64> {
    if slopes.len() < 2 {
        return Some(0.0);
    }
    mad_spread(slopes, 0.0).ok()
}

fn local_adev_slopes_consistent(slopes: &[f64], expected: AdevSlopeClass, tolerance: f64) -> bool {
    slopes
        .iter()
        .all(|&slope| classify_adev_slope(slope, tolerance) == expected)
}

fn local_mdev_slopes_consistent(
    slopes: &[f64],
    expected: Option<PowerLawNoiseType>,
    tolerance: f64,
) -> bool {
    match expected {
        Some(expected) => slopes
            .iter()
            .all(|&slope| classify_mdev_slope(slope, tolerance) == Some(expected)),
        None => true,
    }
}

fn mdev_slope_in_range(
    mdev: &AllanResult,
    tau_start_s: f64,
    tau_end_s: f64,
    options: PowerLawNoiseOptions,
) -> (Option<f64>, Option<f64>, bool) {
    let indices = curve_indices_in_range(mdev, tau_start_s, tau_end_s);
    if indices.len() < options.min_points_per_octave {
        return (None, None, false);
    }
    let start = indices[0];
    let end = *indices.last().expect("nonempty indices");
    let Some(slope) = log_log_slope_for_curve(mdev, start, end) else {
        return (None, None, false);
    };
    let adjacent = adjacent_log_log_slopes(mdev, start, end);
    let classified = classify_mdev_slope(slope, options.slope_tolerance);
    let consistent = local_mdev_slopes_consistent(&adjacent, classified, options.slope_tolerance);
    (Some(slope), robust_slope_scatter(&adjacent), consistent)
}

fn curve_indices_in_range(curve: &AllanResult, tau_start_s: f64, tau_end_s: f64) -> Vec<usize> {
    let tolerance = tau_end_s.abs().max(tau_start_s.abs()).max(1.0) * 32.0 * f64::EPSILON;
    curve
        .tau_s
        .iter()
        .enumerate()
        .filter_map(|(index, &tau_s)| {
            if tau_s + tolerance >= tau_start_s && tau_s <= tau_end_s + tolerance {
                Some(index)
            } else {
                None
            }
        })
        .collect()
}

fn classify_adev_slope(slope: f64, tolerance: f64) -> AdevSlopeClass {
    let phase_target = -1.0;
    if (slope - phase_target).abs() <= tolerance {
        return AdevSlopeClass::PhaseModulation;
    }

    let candidates = [
        PowerLawNoiseType::RandomWalkFM,
        PowerLawNoiseType::FlickerFM,
        PowerLawNoiseType::WhiteFM,
    ];
    let mut best = None;
    let mut best_distance = f64::INFINITY;
    for noise_type in candidates {
        let distance = (slope - allan_deviation_power_law_slope(noise_type)).abs();
        if distance < best_distance {
            best = Some(noise_type);
            best_distance = distance;
        }
    }
    if best_distance <= tolerance {
        AdevSlopeClass::Noise(best.expect("candidate selected"))
    } else {
        AdevSlopeClass::Ambiguous
    }
}

fn classify_mdev_slope(slope: f64, tolerance: f64) -> Option<PowerLawNoiseType> {
    let mut best = None;
    let mut best_distance = f64::INFINITY;
    for noise_type in POWER_LAW_NOISE_TYPES {
        let distance = (slope - modified_allan_deviation_power_law_slope(noise_type)).abs();
        if distance < best_distance {
            best = Some(noise_type);
            best_distance = distance;
        }
    }
    if best_distance <= tolerance {
        best
    } else {
        None
    }
}

fn build_power_law_regions(
    octaves: &[PowerLawOctave],
    adev: &AllanResult,
    mdev: &AllanResult,
    options: PowerLawNoiseOptions,
) -> Vec<PowerLawNoiseRegion> {
    let mut regions = Vec::new();
    let mut current: Option<PowerLawNoiseRegion> = None;

    for octave in octaves {
        let PowerLawOctaveDominance::Dominant(noise_type) = octave.dominance else {
            if let Some(region) = current.take() {
                regions.push(region);
            }
            continue;
        };
        let Some(slope) = octave_slope_for_region(noise_type, octave) else {
            continue;
        };

        if current
            .as_ref()
            .is_some_and(|region| region.noise_type == noise_type)
        {
            let region = current.as_mut().expect("current region");
            let next_count = region.octave_count + 1;
            region.tau_end_s = octave.tau_end_s;
            region.mean_slope =
                (region.mean_slope * region.octave_count as f64 + slope) / next_count as f64;
            region.octave_count = next_count;
        } else {
            if let Some(region) = current.take() {
                regions.push(region);
            }
            current = Some(PowerLawNoiseRegion {
                noise_type,
                tau_start_s: octave.tau_start_s,
                tau_end_s: octave.tau_end_s,
                octave_count: 1,
                point_count: 0,
                mean_slope: slope,
                coefficient: f64::NAN,
            });
        }
    }

    if let Some(region) = current {
        regions.push(region);
    }

    for region in &mut regions {
        region.point_count = count_points_for_region(region.noise_type, region, adev, mdev);
        if let Some(coefficient) = fit_coefficient_for_ranges(
            region.noise_type,
            &[(region.tau_start_s, region.tau_end_s)],
            adev,
            mdev,
            options,
        ) {
            region.coefficient = coefficient;
        }
    }

    regions
}

fn octave_slope_for_region(noise_type: PowerLawNoiseType, octave: &PowerLawOctave) -> Option<f64> {
    match noise_type {
        PowerLawNoiseType::FlickerPM | PowerLawNoiseType::WhitePM => octave.mdev_slope,
        _ => octave.adev_slope,
    }
}

fn count_points_for_region(
    noise_type: PowerLawNoiseType,
    region: &PowerLawNoiseRegion,
    adev: &AllanResult,
    mdev: &AllanResult,
) -> usize {
    let curve = coefficient_curve(noise_type, adev, mdev);
    curve_indices_in_range(curve, region.tau_start_s, region.tau_end_s).len()
}

fn fit_coefficient_for_type(
    noise_type: PowerLawNoiseType,
    regions: &[PowerLawNoiseRegion],
    adev: &AllanResult,
    mdev: &AllanResult,
    options: PowerLawNoiseOptions,
) -> Option<f64> {
    let ranges = regions
        .iter()
        .filter(|region| region.noise_type == noise_type)
        .map(|region| (region.tau_start_s, region.tau_end_s))
        .collect::<Vec<_>>();
    fit_coefficient_for_ranges(noise_type, &ranges, adev, mdev, options)
}

fn fit_coefficient_for_ranges(
    noise_type: PowerLawNoiseType,
    ranges: &[(f64, f64)],
    adev: &AllanResult,
    mdev: &AllanResult,
    options: PowerLawNoiseOptions,
) -> Option<f64> {
    if ranges.is_empty() {
        return None;
    }

    let curve = coefficient_curve(noise_type, adev, mdev);
    let mut sum_ab = 0.0;
    let mut sum_aa = 0.0;
    for (index, &tau_s) in curve.tau_s.iter().enumerate() {
        if !ranges
            .iter()
            .any(|&(start, end)| tau_in_range(tau_s, start, end))
        {
            continue;
        }
        let factor = power_law_variance_factor(noise_type, tau_s, options)?;
        if !(factor.is_finite() && factor > 0.0) {
            return None;
        }
        let variance = curve.deviation[index] * curve.deviation[index];
        if !(variance.is_finite() && variance >= 0.0) {
            return None;
        }
        sum_ab += factor * variance;
        sum_aa += factor * factor;
    }

    if sum_aa > 0.0 {
        Some(sum_ab / sum_aa)
    } else {
        None
    }
}

fn coefficient_curve<'a>(
    noise_type: PowerLawNoiseType,
    adev: &'a AllanResult,
    mdev: &'a AllanResult,
) -> &'a AllanResult {
    match noise_type {
        PowerLawNoiseType::FlickerPM | PowerLawNoiseType::WhitePM => mdev,
        _ => adev,
    }
}

fn tau_in_range(tau_s: f64, tau_start_s: f64, tau_end_s: f64) -> bool {
    let tolerance = tau_end_s.abs().max(tau_start_s.abs()).max(1.0) * 32.0 * f64::EPSILON;
    tau_s + tolerance >= tau_start_s && tau_s <= tau_end_s + tolerance
}

fn power_law_variance_factor(
    noise_type: PowerLawNoiseType,
    tau_s: f64,
    options: PowerLawNoiseOptions,
) -> Option<f64> {
    if !(tau_s.is_finite() && tau_s > 0.0) {
        return None;
    }
    let pi = core::f64::consts::PI;
    let factor = match noise_type {
        PowerLawNoiseType::RandomWalkFM => (2.0 * pi * pi / 3.0) * tau_s,
        PowerLawNoiseType::FlickerFM => 2.0 * core::f64::consts::LN_2,
        PowerLawNoiseType::WhiteFM => 0.5 / tau_s,
        PowerLawNoiseType::FlickerPM => {
            let numerator = 3.0 * libm::log(256.0_f64 / 27.0);
            numerator / (8.0 * pi * pi * tau_s * tau_s)
        }
        PowerLawNoiseType::WhitePM => {
            let numerator = 3.0 * options.measurement_bandwidth_hz * options.basic_tau_s;
            numerator / (4.0 * pi * pi * tau_s * tau_s * tau_s)
        }
    };
    if factor.is_finite() && factor > 0.0 {
        Some(factor)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy)]
struct PhasePoint {
    value_s: f64,
    run_id: usize,
}

fn prepare_phase(
    series: AllanSeries<'_>,
    tau0_s: f64,
    gap_policy: GapPolicy,
) -> Result<Vec<Option<PhasePoint>>, AllanError> {
    validate_tau0(tau0_s)?;
    match series {
        AllanSeries::PhaseSeconds(values) => phase_from_contiguous(values),
        AllanSeries::FractionalFrequency(values) => phase_from_contiguous_frequency(values, tau0_s),
        AllanSeries::PhaseSecondsWithGaps(values) => phase_from_gapped(values, gap_policy),
        AllanSeries::FractionalFrequencyWithGaps(values) => {
            phase_from_gapped_frequency(values, tau0_s, gap_policy)
        }
    }
}

fn validate_tau0(tau0_s: f64) -> Result<(), AllanError> {
    if tau0_s.is_finite() && tau0_s > 0.0 {
        Ok(())
    } else {
        Err(AllanError::InvalidTau0 { tau0_s })
    }
}

fn phase_from_contiguous(values: &[f64]) -> Result<Vec<Option<PhasePoint>>, AllanError> {
    if values.is_empty() {
        return Err(AllanError::EmptySeries);
    }
    values
        .iter()
        .enumerate()
        .map(|(index, &value_s)| {
            validate_sample(index, value_s).map(|value_s| Some(PhasePoint { value_s, run_id: 0 }))
        })
        .collect()
}

fn phase_from_contiguous_frequency(
    values: &[f64],
    tau0_s: f64,
) -> Result<Vec<Option<PhasePoint>>, AllanError> {
    if values.is_empty() {
        return Err(AllanError::EmptySeries);
    }
    let mut phase = Vec::with_capacity(values.len() + 1);
    let mut value_s = 0.0;
    phase.push(Some(PhasePoint { value_s, run_id: 0 }));
    for (index, &frequency) in values.iter().enumerate() {
        let frequency = validate_sample(index, frequency)?;
        value_s += tau0_s * frequency;
        if !value_s.is_finite() {
            return Err(AllanError::NonFiniteSample { index });
        }
        phase.push(Some(PhasePoint { value_s, run_id: 0 }));
    }
    Ok(phase)
}

fn phase_from_gapped(
    values: &[Option<f64>],
    gap_policy: GapPolicy,
) -> Result<Vec<Option<PhasePoint>>, AllanError> {
    if values.is_empty() {
        return Err(AllanError::EmptySeries);
    }

    let mut phase = Vec::with_capacity(values.len());
    let mut run_id = 0usize;
    let mut in_run = false;
    for (index, value) in values.iter().enumerate() {
        match value {
            Some(value_s) => {
                let value_s = validate_sample(index, *value_s)?;
                if !in_run {
                    run_id += 1;
                    in_run = true;
                }
                phase.push(Some(PhasePoint { value_s, run_id }));
            }
            None => {
                if gap_policy == GapPolicy::Reject {
                    return Err(AllanError::Gap { index });
                }
                in_run = false;
                phase.push(None);
            }
        }
    }
    Ok(phase)
}

fn phase_from_gapped_frequency(
    values: &[Option<f64>],
    tau0_s: f64,
    gap_policy: GapPolicy,
) -> Result<Vec<Option<PhasePoint>>, AllanError> {
    if values.is_empty() {
        return Err(AllanError::EmptySeries);
    }
    if gap_policy == GapPolicy::Reject {
        for (index, value) in values.iter().enumerate() {
            if value.is_none() {
                return Err(AllanError::Gap { index });
            }
        }
    }

    let mut phase = vec![None; values.len() + 1];
    let mut run_id = 0usize;
    let mut index = 0usize;
    while index < values.len() {
        if values[index].is_none() {
            index += 1;
            continue;
        };

        run_id += 1;
        let mut value_s = 0.0;
        phase[index] = Some(PhasePoint { value_s, run_id });
        let mut sample_index = index;
        while let Some(current) = values.get(sample_index).copied().flatten() {
            let current = validate_sample(sample_index, current)?;
            value_s += tau0_s * current;
            if !value_s.is_finite() {
                return Err(AllanError::NonFiniteSample {
                    index: sample_index,
                });
            }
            phase[sample_index + 1] = Some(PhasePoint { value_s, run_id });
            sample_index += 1;
        }
        index = sample_index + 1;
    }

    Ok(phase)
}

fn validate_sample(index: usize, value: f64) -> Result<f64, AllanError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AllanError::NonFiniteSample { index })
    }
}

fn compute_curve(
    phase: &[Option<PhasePoint>],
    tau0_s: f64,
    tau_grid: &TauGrid,
    estimator: AllanEstimator,
) -> Result<AllanResult, AllanError> {
    let averaging_factors = averaging_factors(phase.len(), estimator, tau_grid)?;
    let strict = matches!(tau_grid, TauGrid::Explicit(_));
    let mut result = AllanResult::new();

    for m in averaging_factors {
        if m == 0 {
            return Err(AllanError::InvalidAveragingFactor {
                averaging_factor: m,
            });
        }
        if candidate_count(phase.len(), estimator, m).is_none() {
            if strict {
                return Err(AllanError::TooFewSamples {
                    estimator,
                    averaging_factor: m,
                    available_phase_samples: phase.len(),
                });
            }
            continue;
        }

        let tau_s = tau0_s * m as f64;
        if !tau_s.is_finite() {
            return Err(AllanError::NonFiniteTau {
                estimator,
                averaging_factor: m,
            });
        }

        let (sum_sq, n) = estimator_sum_squares(phase, estimator, m);
        if n == 0 {
            if strict {
                return Err(AllanError::TooFewSamples {
                    estimator,
                    averaging_factor: m,
                    available_phase_samples: phase.len(),
                });
            }
            continue;
        }

        let deviation = deviation_from_sum(sum_sq, n, tau_s, m, estimator);
        if !deviation.is_finite() {
            return Err(AllanError::NonFiniteDeviation {
                estimator,
                averaging_factor: m,
            });
        }
        result.push(tau_s, deviation, n);
    }

    if result.is_empty() {
        return Err(AllanError::TooFewSamples {
            estimator,
            averaging_factor: 1,
            available_phase_samples: phase.len(),
        });
    }
    Ok(result)
}

fn averaging_factors(
    phase_len: usize,
    estimator: AllanEstimator,
    tau_grid: &TauGrid,
) -> Result<Vec<usize>, AllanError> {
    match tau_grid {
        TauGrid::Explicit(values) => {
            if values.is_empty() {
                Err(AllanError::EmptyTauGrid)
            } else {
                Ok(values.clone())
            }
        }
        TauGrid::All => Ok((1..=max_averaging_factor(phase_len, estimator)).collect()),
        TauGrid::Octave => {
            let max_m = max_averaging_factor(phase_len, estimator);
            let mut values = Vec::new();
            let mut m = 1usize;
            while m <= max_m {
                values.push(m);
                if m > max_m / 2 {
                    break;
                }
                m *= 2;
            }
            Ok(values)
        }
    }
}

fn max_averaging_factor(phase_len: usize, estimator: AllanEstimator) -> usize {
    match estimator {
        AllanEstimator::Adev | AllanEstimator::OverlappingAdev => phase_len.saturating_sub(1) / 2,
        AllanEstimator::Mdev | AllanEstimator::Tdev => phase_len / 3,
        AllanEstimator::Hdev => phase_len.saturating_sub(1) / 3,
    }
}

fn candidate_count(phase_len: usize, estimator: AllanEstimator, m: usize) -> Option<usize> {
    if m == 0 {
        return None;
    }
    match estimator {
        AllanEstimator::Adev => {
            let frequency_len = phase_len.checked_sub(1)?;
            (frequency_len / m).checked_sub(1)
        }
        AllanEstimator::OverlappingAdev => phase_len.checked_sub(m.checked_mul(2)?),
        AllanEstimator::Mdev | AllanEstimator::Tdev => {
            phase_len.checked_sub(m.checked_mul(3)?)?.checked_add(1)
        }
        AllanEstimator::Hdev => phase_len.checked_sub(m.checked_mul(3)?),
    }
}

fn estimator_sum_squares(
    phase: &[Option<PhasePoint>],
    estimator: AllanEstimator,
    m: usize,
) -> (f64, usize) {
    match estimator {
        AllanEstimator::Adev => plain_adev_sum_squares(phase, m),
        AllanEstimator::OverlappingAdev => overlapping_adev_sum_squares(phase, m),
        AllanEstimator::Mdev | AllanEstimator::Tdev => modified_adev_sum_squares(phase, m),
        AllanEstimator::Hdev => hadamard_sum_squares(phase, m),
    }
}

fn deviation_from_sum(
    sum_sq: f64,
    n: usize,
    tau_s: f64,
    m: usize,
    estimator: AllanEstimator,
) -> f64 {
    let n = n as f64;
    match estimator {
        AllanEstimator::Adev | AllanEstimator::OverlappingAdev => {
            (sum_sq / (2.0 * n * tau_s * tau_s)).sqrt()
        }
        AllanEstimator::Mdev => {
            let mf = m as f64;
            (sum_sq / (2.0 * mf * mf * n * tau_s * tau_s)).sqrt()
        }
        AllanEstimator::Hdev => (sum_sq / (6.0 * n * tau_s * tau_s)).sqrt(),
        AllanEstimator::Tdev => {
            let mf = m as f64;
            let mdev = (sum_sq / (2.0 * mf * mf * n * tau_s * tau_s)).sqrt();
            tau_s * mdev / SQRT_3
        }
    }
}

fn plain_adev_sum_squares(phase: &[Option<PhasePoint>], m: usize) -> (f64, usize) {
    let Some(count) = candidate_count(phase.len(), AllanEstimator::Adev, m) else {
        return (0.0, 0);
    };
    let mut sum_sq = 0.0;
    let mut n = 0usize;
    for j in 0..count {
        let i = j * m;
        if let Some((diff, _)) = second_difference(phase, i, m) {
            let square = diff * diff;
            sum_sq += square;
            n += 1;
        }
    }
    (sum_sq, n)
}

fn overlapping_adev_sum_squares(phase: &[Option<PhasePoint>], m: usize) -> (f64, usize) {
    let Some(count) = candidate_count(phase.len(), AllanEstimator::OverlappingAdev, m) else {
        return (0.0, 0);
    };
    let mut sum_sq = 0.0;
    let mut n = 0usize;
    for i in 0..count {
        if let Some((diff, _)) = second_difference(phase, i, m) {
            let square = diff * diff;
            sum_sq += square;
            n += 1;
        }
    }
    (sum_sq, n)
}

fn modified_adev_sum_squares(phase: &[Option<PhasePoint>], m: usize) -> (f64, usize) {
    let Some(count) = candidate_count(phase.len(), AllanEstimator::Mdev, m) else {
        return (0.0, 0);
    };
    let mut sum_sq = 0.0;
    let mut n = 0usize;
    for j in 0..count {
        let mut inner = 0.0;
        let mut run_id = None;
        let mut valid = true;
        for i in j..(j + m) {
            let Some((diff, diff_run_id)) = second_difference(phase, i, m) else {
                valid = false;
                break;
            };
            if run_id.is_some_and(|current| current != diff_run_id) {
                valid = false;
                break;
            }
            run_id = Some(diff_run_id);
            inner += diff;
        }
        if valid {
            let square = inner * inner;
            sum_sq += square;
            n += 1;
        }
    }
    (sum_sq, n)
}

fn hadamard_sum_squares(phase: &[Option<PhasePoint>], m: usize) -> (f64, usize) {
    let Some(count) = candidate_count(phase.len(), AllanEstimator::Hdev, m) else {
        return (0.0, 0);
    };
    let mut sum_sq = 0.0;
    let mut n = 0usize;
    for i in 0..count {
        if let Some(diff) = third_difference(phase, i, m) {
            let square = diff * diff;
            sum_sq += square;
            n += 1;
        }
    }
    (sum_sq, n)
}

fn second_difference(phase: &[Option<PhasePoint>], i: usize, m: usize) -> Option<(f64, usize)> {
    let x0 = phase.get(i).copied().flatten()?;
    let x1 = phase.get(i + m).copied().flatten()?;
    let x2 = phase.get(i + 2 * m).copied().flatten()?;
    if x0.run_id != x1.run_id || x0.run_id != x2.run_id {
        return None;
    }
    Some((x2.value_s - 2.0 * x1.value_s + x0.value_s, x0.run_id))
}

fn third_difference(phase: &[Option<PhasePoint>], i: usize, m: usize) -> Option<f64> {
    let x0 = phase.get(i).copied().flatten()?;
    let x1 = phase.get(i + m).copied().flatten()?;
    let x2 = phase.get(i + 2 * m).copied().flatten()?;
    let x3 = phase.get(i + 3 * m).copied().flatten()?;
    if x0.run_id != x1.run_id || x0.run_id != x2.run_id || x0.run_id != x3.run_id {
        return None;
    }
    Some(x3.value_s - 3.0 * x2.value_s + 3.0 * x1.value_s - x0.value_s)
}
