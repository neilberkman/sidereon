//! Closed-form navigation-signal figures of merit.
//!
//! This module evaluates modulation spectra and spectrum-derived metrics from
//! analytic chip-pulse transforms. It does not generate IF samples, local
//! replicas, acquisition grids, tracking loops, or receiver state.
//!
//! Shipped modulation variants:
//!
//! - BPSK(n), with code rate `n * 1.023 MHz`.
//! - sine-phased BOC(m,n), with `2m/n` an integer subchip count.
//! - cosine-phased BOC(m,n), with `4m/n` an integer quarter-subchip count.
//! - MBOC(6,1,1/11), as `10/11 BOC(1,1) + 1/11 BOC(6,1)`.
//! - GPS L1C pilot TMBOC(6,1,4/33), as
//!   `29/33 BOC(1,1) + 4/33 BOC(6,1)`.
//! - Galileo E1 CBOC(6,1,1/11) data or pilot chip pulses, formed as the
//!   weighted sum of BOC(1,1) and BOC(6,1) subcarriers with selectable sign.
//!
//! AltBOC, ACE-BOC, interplex products, receiver-specific filtered replicas,
//! and arbitrary channel multiplexing are non-goals for this pass.

use crate::constants::C_M_S;
use crate::validate;

/// Reference chipping-rate unit used by BPSK(n) and BOC(m,n), in hertz.
pub const REFERENCE_CHIP_RATE_HZ: f64 = 1_023_000.0;

/// Receiver bandwidth used by the Betz L1 SSC fixture, in hertz.
pub const BETZ_L1_RECEIVER_BANDWIDTH_HZ: f64 = 24_000_000.0;

const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = 2.0 * PI;
const MAX_SUBCHIPS: usize = 4096;
const MAX_WEIGHTED_COMPONENTS: usize = 32;
const MAX_QUADRATURE_PANELS: usize = 32768;
const QUADRATURE_PANEL_HZ: f64 = REFERENCE_CHIP_RATE_HZ / 4.0;
const ROOT_SCAN_STEPS: usize = 96;
const ROOT_BISECTION_STEPS: usize = 48;
const WEIGHT_SUM_TOL: f64 = 1.0e-12;
const INTEGER_RATIO_TOL: f64 = 1.0e-12;
const DEGENERATE_DENOMINATOR: f64 = 1.0e-300;

#[allow(clippy::excessive_precision)]
const GL64_POSITIVE_NODES: [f64; 32] = [
    2.43502926634244325e-02,
    7.29931217877990424e-02,
    1.21462819296120544e-01,
    1.69644420423992831e-01,
    2.17423643740007083e-01,
    2.64687162208767424e-01,
    3.11322871990210970e-01,
    3.57220158337668126e-01,
    4.02270157963991570e-01,
    4.46366017253464087e-01,
    4.89403145707052956e-01,
    5.31279464019894565e-01,
    5.71895646202634000e-01,
    6.11155355172393278e-01,
    6.48965471254657311e-01,
    6.85236313054233270e-01,
    7.19881850171610771e-01,
    7.52819907260531940e-01,
    7.83972358943341385e-01,
    8.13265315122797539e-01,
    8.40629296252580316e-01,
    8.65999398154092770e-01,
    8.89315445995114140e-01,
    9.10522137078502825e-01,
    9.29569172131939570e-01,
    9.46411374858402765e-01,
    9.61008799652053658e-01,
    9.73326827789910975e-01,
    9.83336253884625977e-01,
    9.91013371476744287e-01,
    9.96340116771955220e-01,
    9.99305041735772170e-01,
];

#[allow(clippy::excessive_precision)]
const GL64_POSITIVE_WEIGHTS: [f64; 32] = [
    4.86909570091397237e-02,
    4.85754674415033935e-02,
    4.83447622348029404e-02,
    4.79993885964583034e-02,
    4.75401657148303222e-02,
    4.69681828162099788e-02,
    4.62847965813143539e-02,
    4.54916279274180727e-02,
    4.45905581637566079e-02,
    4.35837245293234157e-02,
    4.24735151236535352e-02,
    4.12625632426234581e-02,
    3.99537411327204536e-02,
    3.85501531786155635e-02,
    3.70551285402399844e-02,
    3.54722132568822401e-02,
    3.38051618371418630e-02,
    3.20579283548514254e-02,
    3.02346570724024884e-02,
    2.83396726142594486e-02,
    2.63774697150548978e-02,
    2.43527025687111306e-02,
    2.22701738083829967e-02,
    2.01348231535300216e-02,
    1.79517157756973571e-02,
    1.57260304760250269e-02,
    1.34630478967191179e-02,
    1.11681394601309738e-02,
    8.84675982636339009e-03,
    6.50445796897848854e-03,
    4.14703326056443250e-03,
    1.78328072169469699e-03,
];

/// Phasing used for a BOC(m,n) square-wave subcarrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BocPhasing {
    /// Sine-phased BOC, represented by alternating half-subcarrier chips.
    Sine,
    /// Cosine-phased BOC, represented on quarter-subcarrier chips.
    Cosine,
}

/// Sign convention for Galileo E1 CBOC(6,1,1/11) chip pulses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CbocSign {
    /// Add the BOC(6,1) component to the BOC(1,1) component.
    Plus,
    /// Subtract the BOC(6,1) component from the BOC(1,1) component.
    Minus,
}

/// Navigation modulation model used by spectrum-domain metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct SignalModulation {
    kind: ModulationKind,
    label: &'static str,
}

/// One component in a normalized weighted composite spectrum.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedComponent {
    /// Fractional power assigned to this component.
    pub weight: f64,
    /// Component modulation whose normalized PSD is scaled by `weight`.
    pub modulation: SignalModulation,
}

/// Interfering signal and received power used in C/N0 degradation metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct InterferenceTerm {
    /// Interference modulation spectrum.
    pub modulation: SignalModulation,
    /// Interference received power divided by desired-signal received power.
    pub power_ratio_to_carrier: f64,
}

/// Effective C/N0 result with the corresponding degradation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cn0Degradation {
    /// Effective carrier-to-noise-density ratio in hertz.
    pub effective_cn0_hz: f64,
    /// Effective carrier-to-noise-density ratio in decibel-hertz.
    pub effective_cn0_db_hz: f64,
    /// Loss from the input C/N0 to the effective C/N0, in decibels.
    pub degradation_db: f64,
}

/// Processing mode for early-late DLL thermal-noise jitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DllProcessing {
    /// Coherent early-minus-late processing.
    Coherent,
    /// Non-coherent early-minus-late power processing with squaring loss.
    NonCoherent,
}

/// Inputs for code-tracking thermal-noise figures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DllTrackingOptions {
    /// Carrier-to-noise-density ratio, in decibel-hertz.
    pub cn0_db_hz: f64,
    /// One-sided DLL loop bandwidth, in hertz.
    pub loop_bandwidth_hz: f64,
    /// Predetection coherent integration time, in seconds.
    pub integration_time_s: f64,
    /// Early-late correlator spacing, in code chips.
    pub correlator_spacing_chips: f64,
    /// Two-sided receiver bandwidth, in hertz.
    pub receiver_bandwidth_hz: f64,
}

/// Code-tracking thermal-noise result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DllJitter {
    /// One-sigma delay jitter, in seconds.
    pub seconds: f64,
    /// One-sigma delay jitter, in code chips.
    pub chips: f64,
    /// One-sigma range jitter, in metres.
    pub meters: f64,
    /// Non-coherent squaring-loss multiplier. Coherent processing returns `1`.
    pub squaring_loss: f64,
}

/// Inputs for one-path specular multipath envelope metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultipathOptions {
    /// Reflected-path amplitude divided by direct-path amplitude, in `[0, 1)`.
    pub multipath_to_direct_ratio: f64,
    /// Early-late correlator spacing, in code chips.
    pub correlator_spacing_chips: f64,
    /// Two-sided receiver bandwidth, in hertz.
    pub receiver_bandwidth_hz: f64,
}

/// Multipath envelope value at one reflected-path delay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultipathEnvelopePoint {
    /// Reflected-path delay relative to the direct path, in code chips.
    pub delay_chips: f64,
    /// Reflected-path delay relative to the direct path, in seconds.
    pub delay_s: f64,
    /// In-phase multipath tracking error, in code chips.
    pub in_phase_chips: f64,
    /// In-phase multipath tracking error, in seconds.
    pub in_phase_s: f64,
    /// In-phase multipath tracking error, in metres.
    pub in_phase_m: f64,
    /// Anti-phase multipath tracking error, in code chips.
    pub anti_phase_chips: f64,
    /// Anti-phase multipath tracking error, in seconds.
    pub anti_phase_s: f64,
    /// Anti-phase multipath tracking error, in metres.
    pub anti_phase_m: f64,
    /// Running average of the absolute envelope, in code chips.
    pub running_average_chips: f64,
    /// Running average of the absolute envelope, in seconds.
    pub running_average_s: f64,
    /// Running average of the absolute envelope, in metres.
    pub running_average_m: f64,
}

/// Error returned by signal-analysis functions.
#[derive(Debug, Clone, PartialEq)]
pub enum SignalAnalysisError {
    /// A boundary input was malformed before a metric could be computed.
    InvalidInput {
        /// Name of the malformed field.
        field: &'static str,
        /// Stable validation reason.
        reason: &'static str,
    },
    /// A weighted composite was supplied without any components.
    EmptyComponents,
    /// A deterministic root search did not bracket a discriminator zero.
    NoDiscriminatorRoot {
        /// Reflected-path delay for the failed root, in code chips.
        delay_chips: f64,
        /// Multipath phase sign used by the failed root.
        phase_sign: f64,
    },
}

impl core::fmt::Display for SignalAnalysisError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid signal-analysis input {field}: {reason}")
            }
            Self::EmptyComponents => write!(f, "weighted composite has no components"),
            Self::NoDiscriminatorRoot {
                delay_chips,
                phase_sign,
            } => write!(
                f,
                "no early-late discriminator root for delay {delay_chips} chips and phase sign {phase_sign}"
            ),
        }
    }
}

impl std::error::Error for SignalAnalysisError {}

#[derive(Debug, Clone, PartialEq)]
enum ModulationKind {
    Pulse(ChipPulse),
    Weighted(Vec<WeightedComponent>),
}

#[derive(Debug, Clone, PartialEq)]
struct ChipPulse {
    code_rate_hz: f64,
    amplitudes: Vec<f64>,
}

struct CorrelationKernel<'a> {
    modulation: &'a SignalModulation,
    receiver_bandwidth_hz: f64,
    power: f64,
}

impl WeightedComponent {
    /// Construct one weighted component for [`SignalModulation::weighted`].
    pub fn new(weight: f64, modulation: SignalModulation) -> Self {
        Self { weight, modulation }
    }
}

impl InterferenceTerm {
    /// Construct one interference term from a modulation and power ratio.
    pub fn new(modulation: SignalModulation, power_ratio_to_carrier: f64) -> Self {
        Self {
            modulation,
            power_ratio_to_carrier,
        }
    }
}

impl SignalModulation {
    /// Construct BPSK(n), with code rate `n * 1.023 MHz`.
    pub fn bpsk(order: f64) -> Result<Self, SignalAnalysisError> {
        let order = positive(order, "order")?;
        Ok(Self::pulse(
            "BPSK(n)",
            order * REFERENCE_CHIP_RATE_HZ,
            vec![1.0],
        ))
    }

    /// Construct BPSK(1), the GPS C/A spectral shape for long random codes.
    pub fn bpsk1() -> Self {
        Self::pulse("BPSK(1)", REFERENCE_CHIP_RATE_HZ, vec![1.0])
    }

    /// Construct BOC(m,n) with the requested square-wave phasing.
    pub fn boc(m: f64, n: f64, phasing: BocPhasing) -> Result<Self, SignalAnalysisError> {
        let m = positive(m, "m")?;
        let n = positive(n, "n")?;
        let code_rate_hz = n * REFERENCE_CHIP_RATE_HZ;
        let amplitudes = match phasing {
            BocPhasing::Sine => {
                let subchips = integer_ratio(2.0 * m / n, "2m/n")?;
                alternating_subchips(subchips)
            }
            BocPhasing::Cosine => {
                let subchips = integer_ratio(4.0 * m / n, "4m/n")?;
                cosine_subchips(subchips)
            }
        };
        Ok(Self::pulse("BOC(m,n)", code_rate_hz, amplitudes))
    }

    /// Construct sine-phased BOC(m,n).
    pub fn boc_sine(m: f64, n: f64) -> Result<Self, SignalAnalysisError> {
        Self::boc(m, n, BocPhasing::Sine)
    }

    /// Construct cosine-phased BOC(m,n).
    pub fn boc_cosine(m: f64, n: f64) -> Result<Self, SignalAnalysisError> {
        Self::boc(m, n, BocPhasing::Cosine)
    }

    /// Construct the normalized MBOC(6,1,1/11) PSD.
    pub fn mboc_6_1_1_over_11() -> Self {
        Self::weighted_unchecked(
            "MBOC(6,1,1/11)",
            vec![
                WeightedComponent::new(10.0 / 11.0, Self::boc11()),
                WeightedComponent::new(1.0 / 11.0, Self::boc61()),
            ],
        )
    }

    /// Construct the GPS L1C pilot TMBOC(6,1,4/33) PSD.
    pub fn tmboc_6_1_4_over_33() -> Self {
        Self::weighted_unchecked(
            "TMBOC(6,1,4/33)",
            vec![
                WeightedComponent::new(29.0 / 33.0, Self::boc11()),
                WeightedComponent::new(4.0 / 33.0, Self::boc61()),
            ],
        )
    }

    /// Construct one Galileo E1 CBOC(6,1,1/11) chip pulse.
    pub fn cboc_6_1_1_over_11(sign: CbocSign) -> Self {
        let sign = match sign {
            CbocSign::Plus => 1.0,
            CbocSign::Minus => -1.0,
        };
        let low = libm::sqrt(10.0 / 11.0);
        let high = libm::sqrt(1.0 / 11.0);
        let mut amplitudes = Vec::with_capacity(12);
        for k in 0..12 {
            let boc11 = if k < 6 { 1.0 } else { -1.0 };
            let boc61 = if k % 2 == 0 { 1.0 } else { -1.0 };
            amplitudes.push(low * boc11 + sign * high * boc61);
        }
        Self::pulse("CBOC(6,1,1/11)", REFERENCE_CHIP_RATE_HZ, amplitudes)
    }

    /// Construct a custom normalized weighted PSD composite.
    ///
    /// The component weights must be finite, non-negative, and sum to `1`
    /// within `1e-12`. The resulting PSD is the weighted sum of the component
    /// PSDs, so it models time multiplexing or an explicitly published average
    /// spectrum rather than an arbitrary coherent channel sum.
    pub fn weighted(components: Vec<WeightedComponent>) -> Result<Self, SignalAnalysisError> {
        validate_weighted_components(&components)?;
        Ok(Self::weighted_unchecked("weighted composite", components))
    }

    /// Return the normalized power spectral density at offset frequency `f`.
    ///
    /// The return value has units of `1/Hz` and integrates to unit power over
    /// infinite bandwidth for every shipped modulation.
    pub fn psd_hz(&self, offset_hz: f64) -> Result<f64, SignalAnalysisError> {
        let offset_hz = finite(offset_hz, "offset_hz")?;
        match &self.kind {
            ModulationKind::Pulse(pulse) => pulse.psd_hz(offset_hz),
            ModulationKind::Weighted(components) => {
                let mut acc = 0.0;
                for component in components {
                    acc += component.weight * component.modulation.psd_hz(offset_hz)?;
                }
                finite(acc, "psd_hz")
            }
        }
    }

    /// Return the code rate in hertz when the modulation has one unambiguous rate.
    pub fn code_rate_hz(&self) -> Result<f64, SignalAnalysisError> {
        match &self.kind {
            ModulationKind::Pulse(pulse) => Ok(pulse.code_rate_hz),
            ModulationKind::Weighted(components) => {
                let mut iter = components.iter();
                let first = iter
                    .next()
                    .ok_or(SignalAnalysisError::EmptyComponents)?
                    .modulation
                    .code_rate_hz()?;
                for component in iter {
                    let rate = component.modulation.code_rate_hz()?;
                    if rate.to_bits() != first.to_bits() {
                        return Err(invalid("code_rate_hz", "mixed component rates"));
                    }
                }
                Ok(first)
            }
        }
    }

    /// Return a short stable label for the modulation.
    pub fn label(&self) -> &'static str {
        self.label
    }

    fn boc11() -> Self {
        Self::pulse(
            "BOCsin(1,1)",
            REFERENCE_CHIP_RATE_HZ,
            alternating_subchips(2),
        )
    }

    fn boc61() -> Self {
        Self::pulse(
            "BOCsin(6,1)",
            REFERENCE_CHIP_RATE_HZ,
            alternating_subchips(12),
        )
    }

    fn pulse(label: &'static str, code_rate_hz: f64, amplitudes: Vec<f64>) -> Self {
        Self {
            kind: ModulationKind::Pulse(ChipPulse {
                code_rate_hz,
                amplitudes,
            }),
            label,
        }
    }

    fn weighted_unchecked(label: &'static str, components: Vec<WeightedComponent>) -> Self {
        Self {
            kind: ModulationKind::Weighted(components),
            label,
        }
    }
}

impl ChipPulse {
    fn psd_hz(&self, offset_hz: f64) -> Result<f64, SignalAnalysisError> {
        let (real, imag) = self.transform(offset_hz);
        let energy = self.energy_s();
        let psd = (real * real + imag * imag) / energy;
        finite(psd, "psd_hz")
    }

    fn transform(&self, offset_hz: f64) -> (f64, f64) {
        let chip_s = 1.0 / self.code_rate_hz;
        let dt = chip_s / self.amplitudes.len() as f64;
        if offset_hz == 0.0 {
            let real = self.amplitudes.iter().sum::<f64>() * dt;
            return (real, 0.0);
        }

        let w = TWO_PI * offset_hz;
        let mut real = 0.0;
        let mut imag = 0.0;
        for (idx, amplitude) in self.amplitudes.iter().enumerate() {
            let a = idx as f64 * dt;
            let b = (idx + 1) as f64 * dt;
            real += amplitude * (libm::sin(w * b) - libm::sin(w * a)) / w;
            imag += amplitude * (libm::cos(w * b) - libm::cos(w * a)) / w;
        }
        (real, imag)
    }

    fn energy_s(&self) -> f64 {
        let chip_s = 1.0 / self.code_rate_hz;
        let dt = chip_s / self.amplitudes.len() as f64;
        self.amplitudes
            .iter()
            .map(|amplitude| amplitude * amplitude * dt)
            .sum()
    }
}

impl<'a> CorrelationKernel<'a> {
    fn new(
        modulation: &'a SignalModulation,
        receiver_bandwidth_hz: f64,
    ) -> Result<Self, SignalAnalysisError> {
        let power = power_in_band(modulation, receiver_bandwidth_hz)?;
        if power <= 0.0 {
            return Err(invalid("receiver_bandwidth_hz", "zero in-band power"));
        }
        Ok(Self {
            modulation,
            receiver_bandwidth_hz,
            power,
        })
    }

    fn autocorrelation(&self, delay_s: f64) -> Result<f64, SignalAnalysisError> {
        let corr = integrate_symmetric(self.receiver_bandwidth_hz, |f| {
            let psd = self.modulation.psd_hz(f)?;
            finite(psd * libm::cos(TWO_PI * f * delay_s), "autocorrelation")
        })?;
        finite(corr / self.power, "autocorrelation")
    }
}

/// Integrate the normalized PSD over a two-sided receiver bandwidth.
pub fn power_in_band(
    modulation: &SignalModulation,
    receiver_bandwidth_hz: f64,
) -> Result<f64, SignalAnalysisError> {
    let bandwidth = positive(receiver_bandwidth_hz, "receiver_bandwidth_hz")?;
    integrate_symmetric(bandwidth, |f| modulation.psd_hz(f))
}

/// Compute the fraction of total signal power inside a two-sided bandwidth.
pub fn fraction_power_in_band(
    modulation: &SignalModulation,
    receiver_bandwidth_hz: f64,
) -> Result<f64, SignalAnalysisError> {
    power_in_band(modulation, receiver_bandwidth_hz)
}

/// Compute the RMS, or Gabor, bandwidth over a two-sided receiver bandwidth.
pub fn rms_bandwidth_hz(
    modulation: &SignalModulation,
    receiver_bandwidth_hz: f64,
) -> Result<f64, SignalAnalysisError> {
    let bandwidth = positive(receiver_bandwidth_hz, "receiver_bandwidth_hz")?;
    let power = power_in_band(modulation, bandwidth)?;
    if power <= 0.0 {
        return Err(invalid("receiver_bandwidth_hz", "zero in-band power"));
    }
    let moment2 = integrate_symmetric(bandwidth, |f| {
        let psd = modulation.psd_hz(f)?;
        finite(f * f * psd, "rms_bandwidth_hz")
    })?;
    finite(libm::sqrt(moment2 / power), "rms_bandwidth_hz")
}

/// Compute the normalized in-band autocorrelation at a delay.
///
/// The value is normalized by the in-band power, so `delay_s = 0` returns `1`
/// for every modulation with non-zero in-band power.
pub fn autocorrelation(
    modulation: &SignalModulation,
    delay_s: f64,
    receiver_bandwidth_hz: f64,
) -> Result<f64, SignalAnalysisError> {
    let delay_s = finite(delay_s, "delay_s")?;
    let bandwidth = positive(receiver_bandwidth_hz, "receiver_bandwidth_hz")?;
    let power = power_in_band(modulation, bandwidth)?;
    if power <= 0.0 {
        return Err(invalid("receiver_bandwidth_hz", "zero in-band power"));
    }
    let corr = integrate_symmetric(bandwidth, |f| {
        let psd = modulation.psd_hz(f)?;
        finite(psd * libm::cos(TWO_PI * f * delay_s), "autocorrelation")
    })?;
    finite(corr / power, "autocorrelation")
}

/// Compute the spectral separation coefficient between two modulations.
///
/// This is the inner product `int Gs(f) Gi(f) df` over the stated two-sided
/// receiver bandwidth. Both PSDs are normalized over infinite bandwidth.
pub fn spectral_separation_coefficient_hz(
    desired: &SignalModulation,
    interference: &SignalModulation,
    receiver_bandwidth_hz: f64,
) -> Result<f64, SignalAnalysisError> {
    let bandwidth = positive(receiver_bandwidth_hz, "receiver_bandwidth_hz")?;
    integrate_symmetric(bandwidth, |f| {
        let desired_psd = desired.psd_hz(f)?;
        let interference_psd = interference.psd_hz(f)?;
        finite(
            desired_psd * interference_psd,
            "spectral_separation_coefficient_hz",
        )
    })
}

/// Compute a spectral separation coefficient in decibel-hertz.
pub fn spectral_separation_coefficient_db_hz(
    desired: &SignalModulation,
    interference: &SignalModulation,
    receiver_bandwidth_hz: f64,
) -> Result<f64, SignalAnalysisError> {
    let ssc = spectral_separation_coefficient_hz(desired, interference, receiver_bandwidth_hz)?;
    if ssc <= 0.0 {
        return Err(invalid(
            "spectral_separation_coefficient_hz",
            "not positive",
        ));
    }
    finite(
        10.0 * libm::log10(ssc),
        "spectral_separation_coefficient_db_hz",
    )
}

/// Compute the SSC against white interference normalized over the band.
pub fn white_noise_spectral_separation_hz(
    desired: &SignalModulation,
    receiver_bandwidth_hz: f64,
) -> Result<f64, SignalAnalysisError> {
    let bandwidth = positive(receiver_bandwidth_hz, "receiver_bandwidth_hz")?;
    let power = power_in_band(desired, bandwidth)?;
    finite(power / bandwidth, "white_noise_spectral_separation_hz")
}

/// Compute effective C/N0 and degradation for finite-band interference.
///
/// The input `cn0_db_hz` is the desired carrier power divided by thermal-noise
/// density before the listed interference terms are added. Interference powers
/// are supplied as ratios to desired carrier power.
pub fn effective_cn0_degradation(
    desired: &SignalModulation,
    cn0_db_hz: f64,
    receiver_bandwidth_hz: f64,
    interferences: &[InterferenceTerm],
) -> Result<Cn0Degradation, SignalAnalysisError> {
    let cn0_db_hz = finite(cn0_db_hz, "cn0_db_hz")?;
    let cn0_hz = db_hz_to_hz(cn0_db_hz)?;
    let bandwidth = positive(receiver_bandwidth_hz, "receiver_bandwidth_hz")?;
    let signal_power = power_in_band(desired, bandwidth)?;
    if signal_power <= 0.0 {
        return Err(invalid("receiver_bandwidth_hz", "zero in-band power"));
    }

    let mut inverse_effective = 1.0 / cn0_hz;
    for term in interferences {
        let ratio = nonnegative(
            term.power_ratio_to_carrier,
            "interference_power_ratio_to_carrier",
        )?;
        let ssc =
            spectral_separation_coefficient_hz(desired, &term.modulation, receiver_bandwidth_hz)?;
        inverse_effective += ratio * ssc / signal_power;
    }

    if inverse_effective <= 0.0 {
        return Err(invalid("effective_cn0_hz", "not positive"));
    }
    let effective_cn0_hz = 1.0 / inverse_effective;
    let effective_cn0_db_hz = hz_to_db_hz(effective_cn0_hz)?;
    Ok(Cn0Degradation {
        effective_cn0_hz,
        effective_cn0_db_hz,
        degradation_db: cn0_db_hz - effective_cn0_db_hz,
    })
}

/// Compute early-late DLL thermal-noise jitter for a modulation.
pub fn dll_thermal_noise_jitter(
    modulation: &SignalModulation,
    options: DllTrackingOptions,
    processing: DllProcessing,
) -> Result<DllJitter, SignalAnalysisError> {
    let code_rate_hz = modulation.code_rate_hz()?;
    let cn0_hz = db_hz_to_hz(finite(options.cn0_db_hz, "cn0_db_hz")?)?;
    let loop_factor = dll_loop_factor(options.loop_bandwidth_hz, options.integration_time_s)?;
    let spacing_chips = positive(options.correlator_spacing_chips, "correlator_spacing_chips")?;
    let spacing_s = spacing_chips / code_rate_hz;
    let bandwidth = positive(options.receiver_bandwidth_hz, "receiver_bandwidth_hz")?;

    let noise_integral = integrate_symmetric(bandwidth, |f| {
        let psd = modulation.psd_hz(f)?;
        let s = libm::sin(PI * f * spacing_s);
        finite(psd * s * s, "dll_noise_integral")
    })?;
    let discriminator_integral = integrate_symmetric(bandwidth, |f| {
        let psd = modulation.psd_hz(f)?;
        finite(
            f * psd * libm::sin(PI * f * spacing_s),
            "dll_discriminator_integral",
        )
    })?;
    if discriminator_integral.abs() <= DEGENERATE_DENOMINATOR {
        return Err(invalid(
            "correlator_spacing_chips",
            "degenerate discriminator",
        ));
    }

    let mut variance_s2 = loop_factor * noise_integral
        / ((TWO_PI * TWO_PI) * cn0_hz * discriminator_integral * discriminator_integral);
    let squaring_loss = match processing {
        DllProcessing::Coherent => 1.0,
        DllProcessing::NonCoherent => {
            let integration_time_s = positive(options.integration_time_s, "integration_time_s")?;
            let cos2_integral = integrate_symmetric(bandwidth, |f| {
                let psd = modulation.psd_hz(f)?;
                let c = libm::cos(PI * f * spacing_s);
                finite(psd * c * c, "dll_squaring_loss")
            })?;
            let cos_integral = integrate_symmetric(bandwidth, |f| {
                let psd = modulation.psd_hz(f)?;
                finite(psd * libm::cos(PI * f * spacing_s), "dll_squaring_loss")
            })?;
            if cos_integral.abs() <= DEGENERATE_DENOMINATOR {
                return Err(invalid(
                    "correlator_spacing_chips",
                    "degenerate discriminator",
                ));
            }
            let loss =
                1.0 + cos2_integral / (integration_time_s * cn0_hz * cos_integral * cos_integral);
            variance_s2 *= loss;
            finite(loss, "dll_squaring_loss")?
        }
    };
    let seconds = finite(libm::sqrt(variance_s2), "dll_thermal_noise_jitter")?;
    jitter_from_seconds(seconds, code_rate_hz, squaring_loss)
}

/// Compute the published lower bound for code-delay tracking jitter.
pub fn dll_lower_bound(
    modulation: &SignalModulation,
    options: DllTrackingOptions,
) -> Result<DllJitter, SignalAnalysisError> {
    let code_rate_hz = modulation.code_rate_hz()?;
    let cn0_hz = db_hz_to_hz(finite(options.cn0_db_hz, "cn0_db_hz")?)?;
    let loop_factor = dll_loop_factor(options.loop_bandwidth_hz, options.integration_time_s)?;
    let bandwidth = positive(options.receiver_bandwidth_hz, "receiver_bandwidth_hz")?;
    let moment2 = integrate_symmetric(bandwidth, |f| {
        let psd = modulation.psd_hz(f)?;
        finite(f * f * psd, "dll_lower_bound")
    })?;
    if moment2 <= 0.0 {
        return Err(invalid("receiver_bandwidth_hz", "zero spectral moment"));
    }
    let variance_s2 = loop_factor / ((TWO_PI * TWO_PI) * cn0_hz * moment2);
    let seconds = finite(libm::sqrt(variance_s2), "dll_lower_bound")?;
    jitter_from_seconds(seconds, code_rate_hz, 1.0)
}

/// Compute one-path early-late multipath error envelopes on a delay grid.
///
/// Delays must be finite, non-negative, and non-decreasing. The running average
/// is the trapezoidal cumulative average of
/// `max(abs(in_phase), abs(anti_phase))` over the supplied delay axis.
pub fn multipath_error_envelope(
    modulation: &SignalModulation,
    options: MultipathOptions,
    delay_chips: &[f64],
) -> Result<Vec<MultipathEnvelopePoint>, SignalAnalysisError> {
    let code_rate_hz = modulation.code_rate_hz()?;
    let amplitude = unit_interval_exclusive_upper(
        options.multipath_to_direct_ratio,
        "multipath_to_direct_ratio",
    )?;
    let spacing_chips = positive(options.correlator_spacing_chips, "correlator_spacing_chips")?;
    let bandwidth = positive(options.receiver_bandwidth_hz, "receiver_bandwidth_hz")?;
    let spacing_s = spacing_chips / code_rate_hz;
    let kernel = CorrelationKernel::new(modulation, bandwidth)?;
    let mut previous_delay = 0.0;
    let mut area_chips = 0.0;
    let mut previous_envelope = 0.0;
    let mut out = Vec::with_capacity(delay_chips.len());

    for (idx, &delay_chips) in delay_chips.iter().enumerate() {
        let delay_chips = nonnegative(delay_chips, "delay_chips")?;
        if idx > 0 && delay_chips < previous_delay {
            return Err(invalid("delay_chips", "not non-decreasing"));
        }
        let delay_s = delay_chips / code_rate_hz;
        let in_phase_s =
            multipath_tracking_error_s(&kernel, amplitude, delay_s, delay_chips, spacing_s, 1.0)?;
        let anti_phase_s =
            multipath_tracking_error_s(&kernel, amplitude, delay_s, delay_chips, spacing_s, -1.0)?;
        let in_phase_chips = in_phase_s * code_rate_hz;
        let anti_phase_chips = anti_phase_s * code_rate_hz;
        let envelope = in_phase_chips.abs().max(anti_phase_chips.abs());
        if idx == 0 {
            area_chips = 0.0;
        } else {
            area_chips += 0.5 * (previous_envelope + envelope) * (delay_chips - previous_delay);
        }
        let running_average_chips = if delay_chips > 0.0 {
            area_chips / delay_chips
        } else {
            envelope
        };
        out.push(MultipathEnvelopePoint {
            delay_chips,
            delay_s,
            in_phase_chips,
            in_phase_s,
            in_phase_m: in_phase_s * C_M_S,
            anti_phase_chips,
            anti_phase_s,
            anti_phase_m: anti_phase_s * C_M_S,
            running_average_chips,
            running_average_s: running_average_chips / code_rate_hz,
            running_average_m: running_average_chips / code_rate_hz * C_M_S,
        });
        previous_delay = delay_chips;
        previous_envelope = envelope;
    }

    Ok(out)
}

fn multipath_tracking_error_s(
    kernel: &CorrelationKernel<'_>,
    amplitude: f64,
    delay_s: f64,
    delay_chips: f64,
    spacing_s: f64,
    phase_sign: f64,
) -> Result<f64, SignalAnalysisError> {
    let search_half_width = delay_s + 2.0 * spacing_s + 2.0 / kernel.modulation.code_rate_hz()?;
    let left = -search_half_width;
    let right = search_half_width;
    let step = (right - left) / ROOT_SCAN_STEPS as f64;
    let mut best_root = None;
    let mut best_abs = f64::INFINITY;
    let mut prev_x = left;
    let mut prev_y =
        early_late_discriminator(kernel, amplitude, delay_s, spacing_s, phase_sign, prev_x)?;
    if prev_y == 0.0 {
        best_root = Some(prev_x);
        best_abs = prev_x.abs();
    }

    for i in 1..=ROOT_SCAN_STEPS {
        let x = left + i as f64 * step;
        let y = early_late_discriminator(kernel, amplitude, delay_s, spacing_s, phase_sign, x)?;
        if y == 0.0 {
            let abs_x = x.abs();
            if abs_x < best_abs {
                best_root = Some(x);
                best_abs = abs_x;
            }
        } else if (prev_y < 0.0 && y > 0.0) || (prev_y > 0.0 && y < 0.0) {
            let root = bisect_root(
                |candidate| {
                    early_late_discriminator(
                        kernel, amplitude, delay_s, spacing_s, phase_sign, candidate,
                    )
                },
                prev_x,
                x,
            )?;
            let abs_root = root.abs();
            if abs_root < best_abs {
                best_root = Some(root);
                best_abs = abs_root;
            }
        }
        prev_x = x;
        prev_y = y;
    }

    best_root.ok_or(SignalAnalysisError::NoDiscriminatorRoot {
        delay_chips,
        phase_sign,
    })
}

fn early_late_discriminator(
    kernel: &CorrelationKernel<'_>,
    amplitude: f64,
    delay_s: f64,
    spacing_s: f64,
    phase_sign: f64,
    error_s: f64,
) -> Result<f64, SignalAnalysisError> {
    let half = 0.5 * spacing_s;
    let early = kernel.autocorrelation(error_s + half)?
        + phase_sign * amplitude * kernel.autocorrelation(error_s - delay_s + half)?;
    let late = kernel.autocorrelation(error_s - half)?
        + phase_sign * amplitude * kernel.autocorrelation(error_s - delay_s - half)?;
    finite(early * early - late * late, "early_late_discriminator")
}

fn bisect_root<F>(mut f: F, mut left: f64, mut right: f64) -> Result<f64, SignalAnalysisError>
where
    F: FnMut(f64) -> Result<f64, SignalAnalysisError>,
{
    let mut f_left = f(left)?;
    for _ in 0..ROOT_BISECTION_STEPS {
        let mid = 0.5 * (left + right);
        let f_mid = f(mid)?;
        if f_mid == 0.0 {
            return Ok(mid);
        }
        if (f_left < 0.0 && f_mid > 0.0) || (f_left > 0.0 && f_mid < 0.0) {
            right = mid;
        } else {
            left = mid;
            f_left = f_mid;
        }
    }
    Ok(0.5 * (left + right))
}

fn jitter_from_seconds(
    seconds: f64,
    code_rate_hz: f64,
    squaring_loss: f64,
) -> Result<DllJitter, SignalAnalysisError> {
    Ok(DllJitter {
        seconds,
        chips: seconds * code_rate_hz,
        meters: seconds * C_M_S,
        squaring_loss,
    })
}

fn dll_loop_factor(
    loop_bandwidth_hz: f64,
    integration_time_s: f64,
) -> Result<f64, SignalAnalysisError> {
    let loop_bandwidth_hz = positive(loop_bandwidth_hz, "loop_bandwidth_hz")?;
    let integration_time_s = positive(integration_time_s, "integration_time_s")?;
    let factor = loop_bandwidth_hz * (1.0 - 0.5 * loop_bandwidth_hz * integration_time_s);
    if factor <= 0.0 {
        return Err(invalid("loop_bandwidth_hz", "loop factor not positive"));
    }
    finite(factor, "dll_loop_factor")
}

fn integrate_symmetric<F>(bandwidth_hz: f64, f: F) -> Result<f64, SignalAnalysisError>
where
    F: FnMut(f64) -> Result<f64, SignalAnalysisError>,
{
    let half_band = 0.5 * bandwidth_hz;
    let integral = integrate_interval(0.0, half_band, f)?;
    finite(2.0 * integral, "quadrature")
}

fn integrate_interval<F>(a: f64, b: f64, mut f: F) -> Result<f64, SignalAnalysisError>
where
    F: FnMut(f64) -> Result<f64, SignalAnalysisError>,
{
    if b < a {
        return Err(invalid("quadrature_interval", "out of range"));
    }
    if b == a {
        return Ok(0.0);
    }
    let width = b - a;
    let panels = panel_count(width)?;
    let panel_width = width / panels as f64;
    let mut acc = 0.0;
    for panel in 0..panels {
        let pa = a + panel as f64 * panel_width;
        let pb = if panel + 1 == panels {
            b
        } else {
            pa + panel_width
        };
        acc += integrate_panel(pa, pb, &mut f)?;
    }
    finite(acc, "quadrature")
}

fn integrate_panel<F>(a: f64, b: f64, f: &mut F) -> Result<f64, SignalAnalysisError>
where
    F: FnMut(f64) -> Result<f64, SignalAnalysisError>,
{
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let mut acc = 0.0;
    for (&node, &weight) in GL64_POSITIVE_NODES.iter().zip(GL64_POSITIVE_WEIGHTS.iter()) {
        acc += weight * (f(mid - half * node)? + f(mid + half * node)?);
    }
    Ok(half * acc)
}

fn panel_count(width_hz: f64) -> Result<usize, SignalAnalysisError> {
    let panels = libm::ceil(width_hz / QUADRATURE_PANEL_HZ).max(1.0);
    if panels > MAX_QUADRATURE_PANELS as f64 {
        return Err(invalid("receiver_bandwidth_hz", "too wide"));
    }
    Ok(panels as usize)
}

fn alternating_subchips(subchips: usize) -> Vec<f64> {
    (0..subchips)
        .map(|idx| if idx % 2 == 0 { 1.0 } else { -1.0 })
        .collect()
}

fn cosine_subchips(subchips: usize) -> Vec<f64> {
    (0..subchips)
        .map(|idx| match idx % 4 {
            0 | 3 => 1.0,
            _ => -1.0,
        })
        .collect()
}

fn integer_ratio(value: f64, field: &'static str) -> Result<usize, SignalAnalysisError> {
    let value = positive(value, field)?;
    let rounded = libm::round(value);
    if (value - rounded).abs() > INTEGER_RATIO_TOL {
        return Err(invalid(field, "not an integer"));
    }
    if rounded < 1.0 || rounded > MAX_SUBCHIPS as f64 {
        return Err(invalid(field, "out of range"));
    }
    Ok(rounded as usize)
}

fn validate_weighted_components(
    components: &[WeightedComponent],
) -> Result<(), SignalAnalysisError> {
    if components.is_empty() {
        return Err(SignalAnalysisError::EmptyComponents);
    }
    if components.len() > MAX_WEIGHTED_COMPONENTS {
        return Err(invalid("components", "too many"));
    }
    let mut sum = 0.0;
    for component in components {
        sum += nonnegative(component.weight, "component_weight")?;
    }
    if (sum - 1.0).abs() > WEIGHT_SUM_TOL {
        return Err(invalid("component_weight", "weights must sum to one"));
    }
    Ok(())
}

fn db_hz_to_hz(db_hz: f64) -> Result<f64, SignalAnalysisError> {
    let hz = libm::pow(10.0, db_hz / 10.0);
    if hz > 0.0 {
        finite(hz, "cn0_hz")
    } else {
        Err(invalid("cn0_db_hz", "out of range"))
    }
}

fn hz_to_db_hz(hz: f64) -> Result<f64, SignalAnalysisError> {
    let hz = positive(hz, "cn0_hz")?;
    finite(10.0 * libm::log10(hz), "cn0_db_hz")
}

fn positive(x: f64, field: &'static str) -> Result<f64, SignalAnalysisError> {
    validate::finite_positive(x, field).map_err(map_analysis_input)
}

fn nonnegative(x: f64, field: &'static str) -> Result<f64, SignalAnalysisError> {
    validate::finite_nonneg(x, field).map_err(map_analysis_input)
}

fn finite(x: f64, field: &'static str) -> Result<f64, SignalAnalysisError> {
    validate::finite(x, field).map_err(map_analysis_input)
}

fn unit_interval_exclusive_upper(x: f64, field: &'static str) -> Result<f64, SignalAnalysisError> {
    validate::finite_in_range_exclusive_upper(x, 0.0, 1.0, field).map_err(map_analysis_input)
}

fn map_analysis_input(error: validate::FieldError) -> SignalAnalysisError {
    invalid(error.field(), error.reason())
}

fn invalid(field: &'static str, reason: &'static str) -> SignalAnalysisError {
    SignalAnalysisError::InvalidInput { field, reason }
}

#[cfg(test)]
mod tests {
    //! Provenance:
    //! - PSD definitions follow Betz, "The Offset Carrier Modulation for GPS
    //!   Modernization", ION NTM 1999, and the Navipedia BOC sine/cosine PSD
    //!   derivations by J. A. Avila Rodriguez.
    //! - SSC fixture rows are transcribed from Betz, "Intersystem and
    //!   Intrasystem Interference with Signal Imperfections", Table 3, using
    //!   the 24 MHz receiver-bandwidth rows and the table note for theoretical
    //!   BPSK self SSC.
    //! - DLL formulas follow Betz and Kolodziejski as reproduced in Gazda et
    //!   al., "Simulation of Code Tracking Error Variance with Early-Late DLL
    //!   for BOC Modulation", equations 13 and 14.
    //! - MBOC/CBOC constants follow public GPS L1C IS-GPS-800 and the Galileo
    //!   MBOC/CBOC descriptions in Inside GNSS, "The MBOC Modulation".
    //! - The cosine-BOC fixture uses Galileo E6 PRS BOCcos(10,5), cited in
    //!   public Galileo signal-plan descriptions and Inside GNSS MBOC notes.

    use super::*;

    const WIDE_BAND_HZ: f64 = 512.0 * REFERENCE_CHIP_RATE_HZ;
    const HIGH_COMPONENT_WIDE_BAND_HZ: f64 = 2_048.0 * REFERENCE_CHIP_RATE_HZ;
    const COSINE_WIDE_BAND_HZ: f64 = 16_384.0 * REFERENCE_CHIP_RATE_HZ;
    const MULTIPATH_BAND_HZ: f64 = 64.0 * REFERENCE_CHIP_RATE_HZ;

    #[test]
    fn psd_closed_forms_match_pinned_values() {
        let bpsk = SignalModulation::bpsk1();
        let boc11 = SignalModulation::boc_sine(1.0, 1.0).unwrap();
        let boc61 = SignalModulation::boc_sine(6.0, 1.0).unwrap();
        let boc_cos_10_5 = SignalModulation::boc_cosine(10.0, 5.0).unwrap();
        let cboc_plus = SignalModulation::cboc_6_1_1_over_11(CbocSign::Plus);
        let cboc_minus = SignalModulation::cboc_6_1_1_over_11(CbocSign::Minus);

        // Betz random-code BPSK PSD with the public GPS C/A BPSK(1) rate.
        assert_close(
            bpsk.psd_hz(0.0).unwrap(),
            9.775171065493646e-7,
            1.0e-21,
            "BPSK(1) PSD at band center",
        );
        // Betz/Navipedia sine-BOC expression for the public BOC(1,1) family.
        assert_close(
            boc11.psd_hz(0.5 * REFERENCE_CHIP_RATE_HZ).unwrap(),
            3.9617276106485926e-7,
            1.0e-21,
            "BOCsin(1,1) PSD at 0.5 chip-rate offset",
        );
        // Betz/Navipedia sine-BOC expression at the BOC(6,1) high-frequency lobe.
        assert_close(
            boc61.psd_hz(5.5 * REFERENCE_CHIP_RATE_HZ).unwrap(),
            1.8890394898251426e-7,
            1.0e-21,
            "BOCsin(6,1) PSD at 5.5 chip-rate offset",
        );
        // Navipedia cosine-BOC expression for Galileo E6 PRS BOCcos(10,5).
        assert_close(
            boc_cos_10_5.psd_hz(10.0 * REFERENCE_CHIP_RATE_HZ).unwrap(),
            7.923455221297186e-8,
            1.0e-22,
            "BOCcos(10,5) PSD at 10.23 MHz offset",
        );
        // Inside GNSS MBOC/CBOC weights for the Galileo E1 CBOC plus chip pulse.
        assert_close(
            cboc_plus.psd_hz(0.5 * REFERENCE_CHIP_RATE_HZ).unwrap(),
            3.907695366836319e-7,
            1.0e-20,
            "CBOC plus PSD at 0.5 chip-rate offset",
        );
        // Inside GNSS MBOC/CBOC weights for the Galileo E1 CBOC minus chip pulse.
        assert_close(
            cboc_minus.psd_hz(0.5 * REFERENCE_CHIP_RATE_HZ).unwrap(),
            3.307930501412689e-7,
            1.0e-21,
            "CBOC minus PSD at 0.5 chip-rate offset",
        );
    }

    #[test]
    fn psds_have_unit_power_to_fixed_wide_band() {
        let cases = [
            (SignalModulation::bpsk1(), WIDE_BAND_HZ, 1.3e-3),
            (
                SignalModulation::boc_sine(1.0, 1.0).unwrap(),
                WIDE_BAND_HZ,
                1.3e-3,
            ),
            (
                SignalModulation::boc_cosine(10.0, 5.0).unwrap(),
                COSINE_WIDE_BAND_HZ,
                2.0e-3,
            ),
            (
                SignalModulation::mboc_6_1_1_over_11(),
                HIGH_COMPONENT_WIDE_BAND_HZ,
                1.0e-3,
            ),
            (
                SignalModulation::tmboc_6_1_4_over_33(),
                HIGH_COMPONENT_WIDE_BAND_HZ,
                1.0e-3,
            ),
            (
                SignalModulation::cboc_6_1_1_over_11(CbocSign::Plus),
                HIGH_COMPONENT_WIDE_BAND_HZ,
                1.0e-3,
            ),
        ];
        for (modulation, bandwidth_hz, tolerance) in cases {
            let power = power_in_band(&modulation, bandwidth_hz).unwrap();
            assert!(
                (power - 1.0).abs() < tolerance,
                "{} power {power}",
                modulation.label()
            );
        }
    }

    #[test]
    fn mboc_and_tmboc_weights_are_pinned_to_public_signal_definitions() {
        let f = 5.5 * REFERENCE_CHIP_RATE_HZ;
        let boc11 = SignalModulation::boc_sine(1.0, 1.0).unwrap();
        let boc61 = SignalModulation::boc_sine(6.0, 1.0).unwrap();
        let mboc = SignalModulation::mboc_6_1_1_over_11();
        let tmboc = SignalModulation::tmboc_6_1_4_over_33();

        let mboc_expected =
            10.0 / 11.0 * boc11.psd_hz(f).unwrap() + 1.0 / 11.0 * boc61.psd_hz(f).unwrap();
        let tmboc_expected =
            29.0 / 33.0 * boc11.psd_hz(f).unwrap() + 4.0 / 33.0 * boc61.psd_hz(f).unwrap();

        assert_eq!(mboc.psd_hz(f).unwrap().to_bits(), mboc_expected.to_bits());
        assert_eq!(tmboc.psd_hz(f).unwrap().to_bits(), tmboc_expected.to_bits());
    }

    #[test]
    fn ssc_rows_match_betz_l1_table_values() {
        let bpsk = SignalModulation::bpsk1();
        let boc11 = SignalModulation::boc_sine(1.0, 1.0).unwrap();

        let rows = [
            (
                "BPSK theoretical self, Betz Table 3 note",
                &bpsk,
                &bpsk,
                -61.8,
                0.08,
            ),
            (
                "C/A desired, BOC(1,1) interference",
                &bpsk,
                &boc11,
                -67.8,
                0.12,
            ),
            (
                "BOC(1,1) desired, C/A interference",
                &boc11,
                &bpsk,
                -67.9,
                0.12,
            ),
            ("BOC(1,1) self", &boc11, &boc11, -64.8, 0.12),
        ];

        for (label, desired, interference, expected_db_hz, tolerance_db) in rows {
            let got = spectral_separation_coefficient_db_hz(
                desired,
                interference,
                BETZ_L1_RECEIVER_BANDWIDTH_HZ,
            )
            .unwrap();
            assert!(
                (got - expected_db_hz).abs() <= tolerance_db,
                "{label}: got {got}, want {expected_db_hz}"
            );
        }
    }

    #[test]
    fn dll_jitter_reduces_to_textbook_bpsk_case() {
        let bpsk = SignalModulation::bpsk1();
        let options = DllTrackingOptions {
            cn0_db_hz: 45.0,
            loop_bandwidth_hz: 1.0,
            integration_time_s: 0.02,
            correlator_spacing_chips: 0.5,
            receiver_bandwidth_hz: WIDE_BAND_HZ,
        };
        let got = dll_thermal_noise_jitter(&bpsk, options, DllProcessing::Coherent).unwrap();
        let cn0 = db_hz_to_hz(options.cn0_db_hz).unwrap();
        let loop_factor = options.loop_bandwidth_hz
            * (1.0 - 0.5 * options.loop_bandwidth_hz * options.integration_time_s);
        let expected_chips =
            libm::sqrt(loop_factor * options.correlator_spacing_chips / (2.0 * cn0));
        assert_close(
            got.chips,
            expected_chips,
            8.0e-5,
            "BPSK coherent DLL jitter",
        );

        let noncoherent =
            dll_thermal_noise_jitter(&bpsk, options, DllProcessing::NonCoherent).unwrap();
        let expected_loss = 1.0
            + 1.0
                / (options.integration_time_s
                    * cn0
                    * (1.0 - 0.5 * options.correlator_spacing_chips));
        assert_close(
            noncoherent.squaring_loss,
            expected_loss,
            4.0e-4,
            "BPSK noncoherent squaring loss",
        );
    }

    #[test]
    fn lower_bound_tracks_rms_bandwidth_definition() {
        let boc11 = SignalModulation::boc_sine(1.0, 1.0).unwrap();
        let options = DllTrackingOptions {
            cn0_db_hz: 45.0,
            loop_bandwidth_hz: 1.0,
            integration_time_s: 0.02,
            correlator_spacing_chips: 0.1,
            receiver_bandwidth_hz: BETZ_L1_RECEIVER_BANDWIDTH_HZ,
        };
        let bound = dll_lower_bound(&boc11, options).unwrap();
        let power = power_in_band(&boc11, options.receiver_bandwidth_hz).unwrap();
        let beta = rms_bandwidth_hz(&boc11, options.receiver_bandwidth_hz).unwrap();
        let cn0 = db_hz_to_hz(options.cn0_db_hz).unwrap();
        let loop_factor = options.loop_bandwidth_hz
            * (1.0 - 0.5 * options.loop_bandwidth_hz * options.integration_time_s);
        let expected_s = libm::sqrt(loop_factor / ((TWO_PI * TWO_PI) * cn0 * beta * beta * power));
        assert_close(bound.seconds, expected_s, 1.0e-18, "DLL lower bound");
    }

    #[test]
    fn effective_cn0_degradation_matches_closed_form() {
        let desired = SignalModulation::bpsk1();
        let interference = SignalModulation::boc_sine(1.0, 1.0).unwrap();
        let cn0_db_hz = 45.0;
        let ratio = libm::pow(10.0_f64, -20.0 / 10.0);
        let result = effective_cn0_degradation(
            &desired,
            cn0_db_hz,
            BETZ_L1_RECEIVER_BANDWIDTH_HZ,
            &[InterferenceTerm::new(interference.clone(), ratio)],
        )
        .unwrap();
        let cn0 = db_hz_to_hz(cn0_db_hz).unwrap();
        let signal_power = power_in_band(&desired, BETZ_L1_RECEIVER_BANDWIDTH_HZ).unwrap();
        let ssc = spectral_separation_coefficient_hz(
            &desired,
            &interference,
            BETZ_L1_RECEIVER_BANDWIDTH_HZ,
        )
        .unwrap();
        let expected = 1.0 / (1.0 / cn0 + ratio * ssc / signal_power);
        assert_close(result.effective_cn0_hz, expected, 1.0e-11, "effective C/N0");
    }

    #[test]
    fn bpsk_multipath_envelope_matches_classic_canonical_case() {
        let bpsk = SignalModulation::bpsk1();
        let points = multipath_error_envelope(
            &bpsk,
            MultipathOptions {
                multipath_to_direct_ratio: 0.5,
                correlator_spacing_chips: 1.0,
                receiver_bandwidth_hz: MULTIPATH_BAND_HZ,
            },
            &[0.0, 0.5, 1.0],
        )
        .unwrap();
        assert_close(
            points[1].in_phase_chips,
            1.0 / 6.0,
            2.5e-3,
            "BPSK in-phase multipath envelope at 0.5 chip",
        );
        assert!(
            points[1].anti_phase_chips < 0.0,
            "anti-phase envelope should be negative for the canonical delay"
        );
        assert!(
            points[2].running_average_chips > 0.0,
            "running average envelope should accumulate"
        );
    }

    #[test]
    fn multipath_envelope_rejects_complete_cancellation_case() {
        let bpsk = SignalModulation::bpsk1();
        let got = multipath_error_envelope(
            &bpsk,
            MultipathOptions {
                multipath_to_direct_ratio: 1.0,
                correlator_spacing_chips: 1.0,
                receiver_bandwidth_hz: MULTIPATH_BAND_HZ,
            },
            &[0.0],
        );

        assert_eq!(
            got,
            Err(invalid("multipath_to_direct_ratio", "out of range"))
        );
    }

    fn assert_close(got: f64, expected: f64, tolerance: f64, label: &str) {
        assert!(
            (got - expected).abs() <= tolerance,
            "{label}: got {got:e}, want {expected:e}, tolerance {tolerance:e}"
        );
    }
}
