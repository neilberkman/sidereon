//! Static-arc PPP correction precomputation.
//!
//! This module owns the language-independent correction algebra for per-epoch
//! Sun/Moon and solid-earth tide evaluation, per-satellite carrier-phase wind-up
//! continuity, and satellite antenna PCO/PCV projection in the satellite body
//! frame.

use crate::astro::angles::beta_angle_from_cos_rad;
use crate::astro::bodies::{sun_moon_ecef, SunMoon};
use crate::astro::frames::transforms::{FrameTransformError, Ut1Gate};
use crate::astro::math::vec3::{add3, cross3, dot3, neg3, norm3, scale3, sub3, unit3};
use crate::astro::time::model::{Instant, JulianDateSplit, TimeScale};
use crate::astro::time::{
    CoverageError, TimeScaleInputErrorKind, TimeScales, Validated, ValidityMode,
};
use crate::validate;
use std::collections::BTreeMap;
use std::f64::consts::PI;

use crate::antenna;
use crate::bias::{BiasError, BiasLookup, BiasSet, ClockReferenceObservables};
use crate::constants::{
    C_M_S, F_L1_HZ, J2000_JD, MICROSECONDS_PER_SECOND, OMEGA_E_DOT_RAD_S, RAD_TO_DEG,
    SECONDS_PER_DAY, SECONDS_PER_HOUR,
};
use crate::ephemeris::Sp3;
use crate::frequencies;
use crate::observables::{
    predict, ObservablesError, ObservablesInputErrorKind, PredictOptions, PredictedObservables,
};
use crate::tides::{
    ocean_tide_loading, solid_earth_pole_tide, solid_earth_tide_with_constants,
    StationTideConstants, TideError,
};

// The ocean-loading types live in `tides` (the displacement math owns them), but
// `PppCorrectionsOptions::ocean_loading` is the public entry point that consumes
// them. Re-export them here so a caller configuring PPP corrections can name and
// build the option's type, and size the BLQ block with `NUM_OCEAN_CONSTITUENTS`
// rather than a hardcoded `11`, without reaching into `tides`. The pole-tide
// option (`PoleTideOptions`) is defined in this module because it is a
// PPP-correction switch with no role in the tide math itself; this keeps the
// `PppCorrectionsOptions` surface coherent from one module.
pub use crate::tides::{OceanLoadingBlq, NUM_OCEAN_CONSTITUENTS};
use crate::tolerances::{
    FREQUENCY_DENOMINATOR_EPS_HZ, PPP_FREQUENCY_ABS_EPS_HZ, PPP_FREQUENCY_REL_EPS,
    VECTOR_NORM_ZERO_EPS, YAW_SINGULARITY_EPS_RAD,
};
use crate::{GnssSatelliteId, GnssSystem};

/// Civil date/time fields used by PPP correction tables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CivilDateTime {
    /// Gregorian calendar year passed to civil-time validation, tide evaluation,
    /// and Julian-date conversion.
    pub year: i32,
    /// One-based Gregorian calendar month passed to civil-time validation, tide
    /// evaluation, and Julian-date conversion.
    pub month: u8,
    /// Gregorian calendar day passed to civil-time validation, tide evaluation,
    /// and Julian-date conversion.
    pub day: u8,
    /// Civil-time hour used directly and in the fractional hour sent to tide
    /// routines.
    pub hour: u8,
    /// Civil-time minute used directly and in the fractional hour sent to tide
    /// routines.
    pub minute: u8,
    /// Fractional civil seconds validated with the selected time-scale policy,
    /// converted to Julian-date time, and included in the tide fractional hour.
    pub second: f64,
}

/// One satellite observation row needed by the static correction precompute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PppCorrectionObservation {
    /// Physical satellite used for orbit prediction and per-satellite correction
    /// lookup.
    pub sat: GnssSatelliteId,
    /// First carrier frequency in hertz, used for wind-up and code-bias
    /// ionosphere-free calculations.
    pub freq1_hz: f64,
    /// Second carrier frequency in hertz, used for wind-up and code-bias
    /// ionosphere-free calculations.
    pub freq2_hz: f64,
    /// Optional GLONASS FDMA channel passed to RINEX-frequency lookup; when it is
    /// absent, the channel is inferred from the two carrier frequencies when possible.
    pub glonass_channel: Option<i8>,
}

/// One receiver epoch and its visible satellite rows.
#[derive(Debug, Clone, PartialEq)]
pub struct PppCorrectionEpoch {
    /// Civil epoch used for Sun/Moon, station-displacement, and antenna-validity
    /// evaluation.
    pub epoch: CivilDateTime,
    /// Receiver time in seconds from J2000 on the SP3 product's time scale,
    /// used for precise-orbit prediction and converted into the bias
    /// product's time scale for code-bias evaluation.
    pub t_rx_j2000_s: f64,
    /// Visible observations traversed in input order for per-satellite correction
    /// generation.
    pub observations: Vec<PppCorrectionObservation>,
}

/// Frequency-dependent satellite antenna calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct SatelliteAntennaFrequency {
    /// Frequency label used to select this record from a satellite antenna block.
    pub label: String,
    /// Satellite body-frame phase-center offset in meters, projected to ECEF
    /// before the ionosphere-free combination.
    pub pco_m: [f64; 3],
    /// No-azimuth PCV samples as `(nadir angle in degrees, correction in meters)`;
    /// samples are sorted before interpolation and non-finite values are rejected.
    pub noazi_pcv_m: Vec<(f64, f64)>,
}

/// Satellite antenna block selected by PRN and validity window.
#[derive(Debug, Clone, PartialEq)]
pub struct SatelliteAntenna {
    /// Satellite identifier used to select this antenna block.
    pub sat: GnssSatelliteId,
    /// Inclusive lower civil-epoch bound for this block; `None` leaves it open.
    pub valid_from: Option<CivilDateTime>,
    /// Inclusive upper civil-epoch bound for this block; `None` leaves it open.
    pub valid_until: Option<CivilDateTime>,
    /// Frequency-specific PCO and no-azimuth PCV records searched by trimmed
    /// frequency label.
    pub frequencies: Vec<SatelliteAntennaFrequency>,
}

/// Satellite antenna correction options.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SatelliteAntennaOptions {
    /// Trimmed label selecting the first signal's PCO and PCV records.
    pub freq1_label: String,
    /// First configured carrier frequency in hertz; it must be finite, positive,
    /// and distinct from [`Self::freq2_hz`]. When phase wind-up is enabled, this
    /// value overrides the observation's first frequency.
    pub freq1_hz: f64,
    /// Trimmed label selecting the second signal's PCO and PCV records.
    pub freq2_label: String,
    /// Second configured carrier frequency in hertz; it must be finite, positive,
    /// and distinct from [`Self::freq1_hz`]. When phase wind-up is enabled, this
    /// value overrides the observation's second frequency.
    pub freq2_hz: f64,
    /// Antenna blocks searched in vector order by satellite and civil validity.
    pub antennas: Vec<SatelliteAntenna>,
}

impl SatelliteAntennaOptions {
    /// Build satellite-antenna options from both carrier definitions and the
    /// supplied antenna records.
    #[must_use]
    pub fn new(
        freq1_label: String,
        freq1_hz: f64,
        freq2_label: String,
        freq2_hz: f64,
        antennas: Vec<SatelliteAntenna>,
    ) -> Self {
        Self {
            freq1_label,
            freq1_hz,
            freq2_label,
            freq2_hz,
            antennas,
        }
    }
}

/// Offline code-bias correction options.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CodeBiasOptions {
    /// Bias-SINEX or DCB model queried for the selected satellite and epoch.
    pub bias_set: BiasSet,
    /// Per-satellite used code-observable pairs, taking precedence over system
    /// defaults.
    pub used_observables_per_sat: BTreeMap<GnssSatelliteId, (String, String)>,
    /// Per-system used code-observable pairs used when no satellite-specific pair
    /// is present.
    pub used_observables_default: BTreeMap<GnssSystem, (String, String)>,
    /// Optional per-system clock-reference pairs overriding the pairs in
    /// [`BiasSet::clock_reference`].
    pub clock_reference: Option<ClockReferenceObservables>,
}

impl CodeBiasOptions {
    /// Build code-bias options for a bias set with no observable overrides.
    ///
    /// Both override maps are empty and `clock_reference` is `None`; assign
    /// those fields when the observation mapping needs to be configured.
    #[must_use]
    pub fn new(bias_set: BiasSet) -> Self {
        Self {
            bias_set,
            used_observables_per_sat: BTreeMap::new(),
            used_observables_default: BTreeMap::new(),
            clock_reference: None,
        }
    }
}

/// Solid-Earth pole tide correction options.
///
/// The pole tide needs the epoch's IERS polar motion, which the engine's
/// embedded EOP table does not carry (it holds UT1-UTC only). The caller
/// supplies it in arcseconds, sourced from IERS EOP exactly like the other
/// Earth-orientation inputs. Polar motion drifts only a few mas/day, so a single
/// daily value is representative across a static arc.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PoleTideOptions {
    /// IERS polar motion x of the date (arcsec).
    pub xp_arcsec: f64,
    /// IERS polar motion y of the date (arcsec).
    pub yp_arcsec: f64,
}

impl PoleTideOptions {
    /// Build pole-tide options from the date's required IERS polar motion.
    #[must_use]
    pub const fn new(xp_arcsec: f64, yp_arcsec: f64) -> Self {
        Self {
            xp_arcsec,
            yp_arcsec,
        }
    }
}

/// PPP correction precompute switches.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PppCorrectionsOptions {
    /// Enables one Sun/Moon-driven solid-earth displacement per input epoch.
    pub solid_earth_tide: bool,
    /// Enables one polar-motion-driven displacement per input epoch when present.
    pub pole_tide: Option<PoleTideOptions>,
    /// Ocean tide loading: the station's BLQ coefficients. The engine does not
    /// embed ocean-loading models, so the caller supplies the per-station BLQ
    /// block (Bos-Scherneck / OSO Chalmers or equivalent), exactly like the
    /// polar-motion data dependency of the pole tide.
    pub ocean_loading: Option<OceanLoadingBlq>,
    /// Enables continuous per-satellite ionosphere-free phase wind-up corrections
    /// in meters.
    pub phase_windup: bool,
    /// Enables satellite PCO/PCV corrections after validating the configured
    /// carrier pair and PCV samples.
    pub satellite_antenna: Option<SatelliteAntennaOptions>,
    /// Enables clock-datum code-bias corrections for observations with a used
    /// observable mapping.
    pub code_bias: Option<CodeBiasOptions>,
}

impl PppCorrectionsOptions {
    /// Build options with every optional PPP correction disabled.
    ///
    /// The two boolean switches are `false` and each optional correction is
    /// `None`; assign a field to enable that correction explicitly.
    #[allow(clippy::new_without_default)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: None,
        }
    }
}

/// Indexed vector result. The epoch index refers to the input epoch slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochVectorCorrection {
    /// Zero-based index written while `build` enumerates the input epochs and used
    /// as the station-correction lookup key.
    pub epoch_index: usize,
    /// ECEF station-displacement vector in meters, later projected against the
    /// predicted line of sight.
    pub vector_m: [f64; 3],
}

/// Indexed satellite scalar result. The epoch index refers to the input epoch slice.
#[derive(Debug, Clone, PartialEq)]
pub struct SatScalarCorrection {
    /// Satellite copied from the observation and combined with [`Self::epoch_index`]
    /// into the precise-positioning lookup key.
    pub sat: GnssSatelliteId,
    /// Zero-based source-epoch index combined with [`Self::sat`] into the precise-
    /// positioning lookup key.
    pub epoch_index: usize,
    /// Ionosphere-free correction in meters: phase wind-up, satellite PCV, or
    /// clock-datum code bias according to its containing table.
    pub value_m: f64,
}

/// Indexed satellite vector result. The epoch index refers to the input epoch slice.
#[derive(Debug, Clone, PartialEq)]
pub struct SatVectorCorrection {
    /// Satellite copied from the observation and combined with [`Self::epoch_index`]
    /// into the satellite PCO lookup key.
    pub sat: GnssSatelliteId,
    /// Zero-based source-epoch index combined with [`Self::sat`] into the satellite
    /// PCO lookup key.
    pub epoch_index: usize,
    /// Ionosphere-free satellite PCO vector in ECEF meters, projected onto the
    /// predicted line of sight by the PPP measurement model.
    pub vector_m: [f64; 3],
}

/// Precomputed PPP correction tables.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PppCorrections {
    /// Solid-earth tide displacements indexed by input epoch.
    pub tide: Vec<EpochVectorCorrection>,
    /// Solid-earth pole-tide displacements indexed by input epoch.
    pub pole_tide: Vec<EpochVectorCorrection>,
    /// Ocean-loading displacements indexed by input epoch.
    pub ocean_loading: Vec<EpochVectorCorrection>,
    /// Continuous ionosphere-free phase wind-up corrections indexed by satellite
    /// and input epoch.
    pub windup_m: Vec<SatScalarCorrection>,
    /// Ionosphere-free satellite PCO vectors in ECEF meters indexed by satellite
    /// and input epoch.
    pub sat_pco_ecef: Vec<SatVectorCorrection>,
    /// Ionosphere-free no-azimuth satellite PCV values in meters indexed by
    /// satellite and input epoch.
    pub sat_pcv_m: Vec<SatScalarCorrection>,
    /// Clock-datum code-bias corrections in meters indexed by satellite and input
    /// epoch.
    pub code_bias_m: Vec<SatScalarCorrection>,
    /// Diagnostics containing missing-metadata warnings for observations without a
    /// code-bias used-observable mapping.
    pub diagnostics: crate::format::Diagnostics,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
/// Errors encountered while constructing static PPP correction tables.
pub enum PppCorrectionsError {
    #[error("invalid PPP correction input {field}: {reason}")]
    /// A receiver, antenna, prediction, or media input failed validation.
    InvalidInput {
        /// Static label identifying the input that failed validation.
        field: &'static str,
        /// Validator or prediction reason, such as `not finite` or `out of range`.
        reason: &'static str,
    },
    #[error("invalid PPP correction epoch at epoch {epoch_index}: {source}")]
    /// Sun/Moon time-scale construction failed for an input epoch.
    Epoch {
        /// Zero-based index of the invalid or out-of-coverage input epoch.
        epoch_index: usize,
        #[source]
        /// Civil-time validation or embedded time-model coverage failure.
        source: CoverageError,
    },
    #[error("solid Earth tide correction failed at epoch {epoch_index}: {source}")]
    /// Solid-earth tide evaluation failed after Sun/Moon data were obtained.
    Tide {
        /// Zero-based index of the input epoch where evaluation failed.
        epoch_index: usize,
        #[source]
        /// Underlying solid-earth tide failure.
        source: TideError,
    },
    #[error("solid Earth pole tide correction failed at epoch {epoch_index}: {source}")]
    /// Solid-earth pole-tide evaluation failed.
    PoleTide {
        /// Zero-based index of the input epoch where evaluation failed.
        epoch_index: usize,
        #[source]
        /// Underlying polar-motion tide failure.
        source: TideError,
    },
    #[error("ocean tide loading correction failed at epoch {epoch_index}: {source}")]
    /// BLQ ocean-loading evaluation failed.
    OceanLoading {
        /// Zero-based index of the input epoch where evaluation failed.
        epoch_index: usize,
        #[source]
        /// Underlying ocean-loading failure.
        source: TideError,
    },
    #[error(
        "invalid phase wind-up carrier frequencies at epoch {epoch_index} for {sat}: {field} {reason}"
    )]
    /// A carrier pair selected for phase wind-up failed finite, positivity, or
    /// separation validation.
    WindupFrequency {
        /// Zero-based index of the observation epoch where validation failed.
        epoch_index: usize,
        /// Satellite whose selected carrier pair failed validation.
        sat: GnssSatelliteId,
        /// Static label identifying the failing frequency or frequency pair.
        field: &'static str,
        /// Validation reason, such as `not finite`, `not positive`, or `must differ`.
        reason: &'static str,
    },
    #[error("invalid satellite antenna carrier frequencies: {field} {reason}")]
    /// The configured satellite-antenna carrier pair failed validation before
    /// per-observation processing.
    SatelliteAntennaFrequency {
        /// Static label identifying the failing configured frequency or pair.
        field: &'static str,
        /// Validation reason, such as `not finite`, `not positive`, or `must differ`.
        reason: &'static str,
    },
    #[error("code-bias correction failed: {source}")]
    /// Code-bias setup failed because the required clock reference or bias epoch
    /// conversion was unavailable.
    Bias {
        #[source]
        /// Underlying bias-model failure, such as a missing clock reference.
        source: BiasError,
    },
    #[error("invalid code-bias observable at epoch {epoch_index} for {sat}: {field} {reason}")]
    /// A code-bias used observable has an invalid or mismatched carrier frequency.
    CodeBiasObservable {
        /// Zero-based index of the input epoch containing the invalid observation.
        epoch_index: usize,
        /// Satellite containing the invalid or mismatched observation frequency.
        sat: GnssSatelliteId,
        /// Static label identifying the used observable frequency field.
        field: &'static str,
        /// Validation reason, including `not finite`, `not positive`, or
        /// `frequency mismatch`.
        reason: &'static str,
    },
}

/// Build static PPP correction tables for a precise-orbit arc.
///
/// The Sun and Moon behind the solid Earth tide, phase wind-up and satellite
/// antenna corrections are rotated into ITRF with UT1, so an epoch outside the
/// UT1 table is refused when any of those is enabled; see
/// [`build_with_validity`].
pub fn build(
    sp3: &Sp3,
    epochs: &[PppCorrectionEpoch],
    receiver_ecef_m: [f64; 3],
    options: &PppCorrectionsOptions,
) -> Result<PppCorrections, PppCorrectionsError> {
    build_with_validity(sp3, epochs, receiver_ecef_m, options, ValidityMode::Strict)
        .map(|validated| validated.value)
}

/// [`build`] under an explicit UT1 [`ValidityMode`]: [`ValidityMode::Strict`]
/// refuses an epoch outside the UT1 table with
/// [`CoverageError::OutsideCoverage`]; [`ValidityMode::Permissive`] evaluates
/// the Sun and Moon there with the long-term UT1 and reports the first
/// departure in [`Validated::degraded`].
pub fn build_with_validity(
    sp3: &Sp3,
    epochs: &[PppCorrectionEpoch],
    receiver_ecef_m: [f64; 3],
    options: &PppCorrectionsOptions,
    mode: ValidityMode,
) -> Result<Validated<PppCorrections>, PppCorrectionsError> {
    build_with_validity_and_tide_constants(
        sp3,
        epochs,
        receiver_ecef_m,
        options,
        mode,
        StationTideConstants::Conventions,
    )
}

/// [`build_with_validity`] with an explicit Chapter 7 station-tide constant
/// table. [`StationTideConstants::Conventions`] is the default used by
/// [`build`] and [`build_with_validity`]; [`StationTideConstants::IersRoutine`]
/// reproduces the distributed IERS routine for comparisons with its outputs.
pub fn build_with_validity_and_tide_constants(
    sp3: &Sp3,
    epochs: &[PppCorrectionEpoch],
    receiver_ecef_m: [f64; 3],
    options: &PppCorrectionsOptions,
    mode: ValidityMode,
    tide_constants: StationTideConstants,
) -> Result<Validated<PppCorrections>, PppCorrectionsError> {
    let gate = Ut1Gate::new(mode);
    let corrections = build_gated(
        sp3,
        epochs,
        receiver_ecef_m,
        options,
        &gate,
        BuildPolicy {
            tide_constants,
            predictor: predict,
            receiver_neu_basis: crate::estimation::substrate::frames::geodetic_neu_basis,
        },
    )?;
    Ok(Validated {
        value: corrections,
        degraded: gate
            .finish(())
            .expect("every UT1 refusal is returned where it happens")
            .degraded,
    })
}

/// The observable predictor the wind-up and satellite antenna corrections read the
/// satellite geometry from: [`predict`], or in this repository's tests the replay of the
/// external reference's microsecond-rounded transmission epoch.
type Predictor = fn(
    &dyn crate::observables::ObservableEphemerisSource,
    GnssSatelliteId,
    [f64; 3],
    f64,
    PredictOptions,
) -> Result<PredictedObservables, ObservablesError>;

/// The receiver's local `(north, east, up)` basis the wind-up takes the receiver
/// dipole from: the geodetic one, or in this repository's tests the replay of the
/// external reference's geocentric one.
type ReceiverNeuBasis = fn([f64; 3]) -> ([f64; 3], [f64; 3], [f64; 3]);

#[derive(Clone, Copy)]
struct BuildPolicy {
    tide_constants: StationTideConstants,
    predictor: Predictor,
    receiver_neu_basis: ReceiverNeuBasis,
}

/// [`build`] with the transmission epoch rounded to whole microseconds and the receiver
/// dipole in the geocentric frame, as the external reference fixture computed it.
#[cfg(all(test, feature = "test-replays"))]
fn build_rounded_microsecond_replay_with_tide_constants(
    sp3: &Sp3,
    epochs: &[PppCorrectionEpoch],
    receiver_ecef_m: [f64; 3],
    options: &PppCorrectionsOptions,
    tide_constants: StationTideConstants,
) -> Result<PppCorrections, PppCorrectionsError> {
    let gate = Ut1Gate::new(ValidityMode::Strict);
    build_gated(
        sp3,
        epochs,
        receiver_ecef_m,
        options,
        &gate,
        BuildPolicy {
            tide_constants,
            predictor: crate::observables::rounded_microsecond_replay::predict,
            receiver_neu_basis: crate::frame::geocentric_neu_basis,
        },
    )
}

/// [`build`] with the receiver dipole in the geocentric frame the external reference
/// fixture used, and every other term live.
#[cfg(all(test, feature = "test-replays"))]
fn build_geocentric_dipole_replay(
    sp3: &Sp3,
    epochs: &[PppCorrectionEpoch],
    receiver_ecef_m: [f64; 3],
    options: &PppCorrectionsOptions,
) -> Result<PppCorrections, PppCorrectionsError> {
    let gate = Ut1Gate::new(ValidityMode::Strict);
    build_gated(
        sp3,
        epochs,
        receiver_ecef_m,
        options,
        &gate,
        BuildPolicy {
            tide_constants: StationTideConstants::IersRoutine,
            predictor: predict,
            receiver_neu_basis: crate::frame::geocentric_neu_basis,
        },
    )
}

fn build_gated(
    sp3: &Sp3,
    epochs: &[PppCorrectionEpoch],
    receiver_ecef_m: [f64; 3],
    options: &PppCorrectionsOptions,
    gate: &Ut1Gate,
    policy: BuildPolicy,
) -> Result<PppCorrections, PppCorrectionsError> {
    validate_receiver_state(receiver_ecef_m)?;

    let mut corrections = PppCorrections::default();
    if !options.solid_earth_tide
        && options.pole_tide.is_none()
        && options.ocean_loading.is_none()
        && !options.phase_windup
        && options.satellite_antenna.is_none()
        && options.code_bias.is_none()
    {
        return Ok(corrections);
    }

    let satellite_antenna_frequencies = options
        .satellite_antenna
        .as_ref()
        .map(validate_satellite_antenna_options)
        .transpose()?;

    let mut previous_windup_cycles: BTreeMap<GnssSatelliteId, f64> = BTreeMap::new();

    // Sun/Moon is needed only by the solid-earth tide and the satellite-yaw
    // corrections (phase wind-up + satellite antenna). Pole tide and ocean
    // loading are pure station displacements, so a pole/ocean-only config must
    // not be coupled to the Sun/Moon (and the EOP/SP3 time paths behind them).
    let need_sun_moon =
        options.solid_earth_tide || options.phase_windup || options.satellite_antenna.is_some();
    // The per-observation predict() loop only feeds the wind-up and satellite
    // antenna corrections; skip it entirely when neither is enabled.
    let need_obs_loop = options.phase_windup || options.satellite_antenna.is_some();

    for (epoch_index, epoch_row) in epochs.iter().enumerate() {
        let sun_moon = if need_sun_moon {
            Some(sun_moon_at(epoch_row.epoch, gate).map_err(|source| {
                PppCorrectionsError::Epoch {
                    epoch_index,
                    source,
                }
            })?)
        } else {
            None
        };

        if options.solid_earth_tide {
            let sun_moon = sun_moon.expect("Sun/Moon computed when solid-earth tide is enabled");
            let d = tide_at(
                receiver_ecef_m,
                epoch_row.epoch,
                sun_moon.sun,
                sun_moon.moon,
                policy.tide_constants,
            )
            .map_err(|source| PppCorrectionsError::Tide {
                epoch_index,
                source,
            })?;
            corrections.tide.push(EpochVectorCorrection {
                epoch_index,
                vector_m: d,
            });
        }

        if let Some(pole) = options.pole_tide {
            let d = pole_tide_at(receiver_ecef_m, epoch_row.epoch, pole).map_err(|source| {
                PppCorrectionsError::PoleTide {
                    epoch_index,
                    source,
                }
            })?;
            corrections.pole_tide.push(EpochVectorCorrection {
                epoch_index,
                vector_m: d,
            });
        }

        if let Some(blq) = options.ocean_loading.as_ref() {
            let d = ocean_loading_at(receiver_ecef_m, epoch_row.epoch, blq).map_err(|source| {
                PppCorrectionsError::OceanLoading {
                    epoch_index,
                    source,
                }
            })?;
            corrections.ocean_loading.push(EpochVectorCorrection {
                epoch_index,
                vector_m: d,
            });
        }

        if let Some(code_bias) = options.code_bias.as_ref() {
            for observation in &epoch_row.observations {
                let lookup = code_bias_correction_m(
                    code_bias,
                    observation,
                    epoch_row,
                    epoch_index,
                    sp3.header.time_scale,
                )?;
                let kind = match lookup {
                    BiasLookup::Available { value, .. } => {
                        corrections.code_bias_m.push(SatScalarCorrection {
                            sat: observation.sat,
                            epoch_index,
                            value_m: value,
                        });
                        continue;
                    }
                    // Conflicting bias records, and a receiver epoch that
                    // cannot be put on the product's time scale, leave the
                    // correction undetermined; the warning says so rather
                    // than reporting missing metadata.
                    BiasLookup::Ambiguous { .. }
                    | BiasLookup::UnsupportedScale {
                        product: Some(_), ..
                    } => crate::format::WarningKind::Mismatch,
                    // No correction: no used observables configured, no
                    // record, no carrier for an observable, or a product
                    // without a usable time system.
                    _ => crate::format::WarningKind::MissingMetadata,
                };
                corrections
                    .diagnostics
                    .push_warning(crate::format::Warning {
                        at: crate::format::RecordRef::at_record(epoch_index)
                            .with_satellite(observation.sat.to_string()),
                        kind,
                    });
            }
        }

        if !need_obs_loop {
            continue;
        }
        let sun_moon = sun_moon.expect("Sun/Moon computed when the observation loop runs");

        for observation in &epoch_row.observations {
            let obs = match (policy.predictor)(
                sp3,
                observation.sat,
                receiver_ecef_m,
                epoch_row.t_rx_j2000_s,
                PredictOptions {
                    carrier_hz: F_L1_HZ,
                    light_time: true,
                    sagnac: true,
                },
            ) {
                Ok(obs) => obs,
                Err(ObservablesError::InvalidInput { field, kind }) => {
                    return Err(PppCorrectionsError::InvalidInput {
                        field,
                        reason: observables_input_reason(kind),
                    });
                }
                Err(ObservablesError::Media(_)) => {
                    return Err(PppCorrectionsError::InvalidInput {
                        field: "media",
                        reason: "out of range",
                    });
                }
                Err(ObservablesError::NoEphemeris | ObservablesError::Ephemeris(_)) => continue,
            };

            if options.phase_windup {
                let prev = previous_windup_cycles.get(&observation.sat).copied();
                if let Some(phw) = windup_cycles(
                    &obs,
                    receiver_ecef_m,
                    (policy.receiver_neu_basis)(receiver_ecef_m),
                    sun_moon.sun,
                    prev,
                ) {
                    let (f1, f2) = windup_frequency_pair(options, observation, epoch_index)?;
                    corrections.windup_m.push(SatScalarCorrection {
                        sat: observation.sat,
                        epoch_index,
                        value_m: windup_metres(phw, f1, f2),
                    });
                    previous_windup_cycles.insert(observation.sat, phw);
                }
            }

            if let Some(sat_ant) = &options.satellite_antenna {
                if let Some((pco_ecef, pcv_m)) = satellite_antenna_correction(
                    &obs,
                    sun_moon.sun,
                    observation.sat,
                    epoch_row.epoch,
                    sat_ant,
                    satellite_antenna_frequencies
                        .expect("satellite antenna frequencies are validated when enabled"),
                ) {
                    corrections.sat_pco_ecef.push(SatVectorCorrection {
                        sat: observation.sat,
                        epoch_index,
                        vector_m: pco_ecef,
                    });
                    corrections.sat_pcv_m.push(SatScalarCorrection {
                        sat: observation.sat,
                        epoch_index,
                        value_m: pcv_m,
                    });
                }
            }
        }
    }

    Ok(corrections)
}

fn validate_receiver_state(receiver_ecef_m: [f64; 3]) -> Result<(), PppCorrectionsError> {
    validate::finite_vec3(receiver_ecef_m, "receiver_ecef_m").map_err(ppp_invalid_input)?;
    validate::finite_positive(norm3(receiver_ecef_m), "receiver radius_m")
        .map_err(ppp_invalid_input)?;
    Ok(())
}

fn ppp_invalid_input(error: validate::FieldError) -> PppCorrectionsError {
    PppCorrectionsError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

fn observables_input_reason(kind: ObservablesInputErrorKind) -> &'static str {
    match kind {
        ObservablesInputErrorKind::NonFinite => "not finite",
        ObservablesInputErrorKind::NotPositive => "not positive",
        ObservablesInputErrorKind::Negative => "negative",
        ObservablesInputErrorKind::OutOfRange => "out of range",
        ObservablesInputErrorKind::Missing => "missing",
        ObservablesInputErrorKind::FloatParse => "invalid float",
        ObservablesInputErrorKind::IntParse => "invalid integer",
        ObservablesInputErrorKind::InvalidCivilDate => "invalid civil date",
        ObservablesInputErrorKind::InvalidCivilTime => "invalid civil time",
    }
}

fn windup_frequency_pair(
    options: &PppCorrectionsOptions,
    observation: &PppCorrectionObservation,
    epoch_index: usize,
) -> Result<(f64, f64), PppCorrectionsError> {
    let (f1_hz, f2_hz) = options
        .satellite_antenna
        .as_ref()
        .map(|a| (a.freq1_hz, a.freq2_hz))
        .unwrap_or((observation.freq1_hz, observation.freq2_hz));
    validate_frequency_pair(
        f1_hz,
        f2_hz,
        FrequencyPairFields {
            freq1: "phase wind-up freq1_hz",
            freq2: "phase wind-up freq2_hz",
            pair: "phase wind-up frequency pair",
        },
        |field, reason| PppCorrectionsError::WindupFrequency {
            epoch_index,
            sat: observation.sat,
            field,
            reason,
        },
    )
}

fn validate_satellite_antenna_frequency_pair(
    options: &SatelliteAntennaOptions,
) -> Result<(f64, f64), PppCorrectionsError> {
    validate_frequency_pair(
        options.freq1_hz,
        options.freq2_hz,
        FrequencyPairFields {
            freq1: "satellite antenna freq1_hz",
            freq2: "satellite antenna freq2_hz",
            pair: "satellite antenna frequency pair",
        },
        |field, reason| PppCorrectionsError::SatelliteAntennaFrequency { field, reason },
    )
}

fn validate_satellite_antenna_options(
    options: &SatelliteAntennaOptions,
) -> Result<(f64, f64), PppCorrectionsError> {
    let frequencies_hz = validate_satellite_antenna_frequency_pair(options)?;
    validate_satellite_antenna_pcv_samples(options)?;
    Ok(frequencies_hz)
}

fn code_bias_correction_m(
    options: &CodeBiasOptions,
    observation: &PppCorrectionObservation,
    epoch_row: &PppCorrectionEpoch,
    epoch_index: usize,
    receiver_scale: TimeScale,
) -> Result<BiasLookup, PppCorrectionsError> {
    let Some(used) = options
        .used_observables_per_sat
        .get(&observation.sat)
        .or_else(|| {
            options
                .used_observables_default
                .get(&observation.sat.system)
        })
    else {
        return Ok(BiasLookup::Absent);
    };
    let glonass_channel =
        observation_glonass_channel(observation, epoch_index, (&used.0, &used.1))?;
    validate_code_observable_frequency(
        observation,
        epoch_index,
        "used observable 1",
        &used.0,
        observation.freq1_hz,
        glonass_channel,
    )?;
    validate_code_observable_frequency(
        observation,
        epoch_index,
        "used observable 2",
        &used.1,
        observation.freq2_hz,
        glonass_channel,
    )?;
    let reference = options
        .clock_reference
        .as_ref()
        .unwrap_or(options.bias_set.clock_reference());
    if reference.per_system.is_empty() {
        return Err(PppCorrectionsError::Bias {
            source: BiasError::MissingClockReference,
        });
    }
    let Some(clock_pair) = reference.per_system.get(&observation.sat.system) else {
        return Err(PppCorrectionsError::Bias {
            source: BiasError::MissingClockReference,
        });
    };
    // The used pair is the clock datum: the model is exactly zero, whatever
    // the product's time system.
    if used.0 == clock_pair.0 && used.1 == clock_pair.1 {
        return Ok(BiasLookup::Available {
            value: 0.0,
            records: Vec::new(),
            overridden: Vec::new(),
        });
    }
    // A product without a usable TIME_SYSTEM gives no scale to read the
    // receiver epoch on; that observation gets no correction and a warning.
    let Some(product_scale) = options.bias_set.time_scale() else {
        return Ok(BiasLookup::UnsupportedScale {
            product: None,
            query: receiver_scale,
        });
    };
    let Some(epoch) = code_bias_epoch(epoch_row.t_rx_j2000_s, receiver_scale, product_scale)
        .map_err(|source| PppCorrectionsError::Bias { source })?
    else {
        return Ok(BiasLookup::UnsupportedScale {
            product: Some(product_scale),
            query: receiver_scale,
        });
    };
    Ok(options.bias_set.code_bias_model_m(
        observation.sat,
        (&used.0, &used.1),
        (observation.freq1_hz, observation.freq2_hz),
        glonass_channel,
        (&clock_pair.0, &clock_pair.1),
        epoch,
    ))
}

/// The receiver epoch, seconds from J2000 on `receiver_scale`, as an instant
/// on the bias product's `product_scale`, converted through the crate's
/// time-scale offsets (leap seconds included for UTC). `None` when the two
/// scales have no modelled offset, such as TCG or TCB.
fn code_bias_epoch(
    t_rx_j2000_s: f64,
    receiver_scale: TimeScale,
    product_scale: TimeScale,
) -> Result<Option<Instant>, BiasError> {
    validate::finite(t_rx_j2000_s, "t_rx_j2000_s").map_err(|error| BiasError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    })?;
    let t_product_s = if receiver_scale == product_scale {
        t_rx_j2000_s
    } else {
        let Some(offset_s) = scale_offset_s(t_rx_j2000_s, receiver_scale, product_scale) else {
            return Ok(None);
        };
        t_rx_j2000_s + offset_s
    };
    let days_since_j2000 = t_product_s / SECONDS_PER_DAY;
    let whole_days = days_since_j2000.floor();
    let fraction = days_since_j2000 - whole_days;
    let jd = JulianDateSplit::new(J2000_JD + whole_days, fraction)
        .map_err(|_| BiasError::InvalidEpoch)?;
    Ok(Some(Instant::from_julian_date(product_scale, jd)))
}

/// `to` reading minus `from` reading, in seconds, at the instant `t_j2000_s`
/// reads on `from`. The UTC Julian date that picks the leap-second count is
/// refined once, so it is right on both sides of a leap second.
fn scale_offset_s(t_j2000_s: f64, from: TimeScale, to: TimeScale) -> Option<f64> {
    use crate::astro::time::scales::timescale_offset_at_s;
    let jd_from = J2000_JD + t_j2000_s / SECONDS_PER_DAY;
    let first = timescale_offset_at_s(from, TimeScale::Utc, jd_from).ok()?;
    let utc_jd = jd_from + first / SECONDS_PER_DAY;
    let refined = timescale_offset_at_s(from, TimeScale::Utc, utc_jd).ok()?;
    let utc_jd = jd_from + refined / SECONDS_PER_DAY;
    timescale_offset_at_s(from, to, utc_jd).ok()
}

fn validate_code_observable_frequency(
    observation: &PppCorrectionObservation,
    epoch_index: usize,
    field: &'static str,
    obs: &str,
    actual_hz: f64,
    glonass_channel: Option<i8>,
) -> Result<(), PppCorrectionsError> {
    validate::finite_positive(actual_hz, field).map_err(|error| {
        PppCorrectionsError::CodeBiasObservable {
            epoch_index,
            sat: observation.sat,
            field: error.field(),
            reason: error.reason(),
        }
    })?;
    let Some(expected_hz) = frequencies::rinex_observation_frequency_hz(
        observation.sat.system,
        obs,
        3.04,
        glonass_channel,
    ) else {
        return Ok(());
    };
    let tol_hz = (expected_hz.abs().max(actual_hz.abs()) * PPP_FREQUENCY_REL_EPS)
        .max(PPP_FREQUENCY_ABS_EPS_HZ);
    if (expected_hz - actual_hz).abs() > tol_hz {
        return Err(PppCorrectionsError::CodeBiasObservable {
            epoch_index,
            sat: observation.sat,
            field,
            reason: "frequency mismatch",
        });
    }
    Ok(())
}

/// The GLONASS FDMA channel of an observation: its stated channel, or else the one its
/// frequencies name for the bands of its used observables, `freq1_hz` for the first and
/// `freq2_hz` for the second ([`frequencies::infer_glonass_fdma_channel`]). Frequencies
/// that name two different channels are refused as a frequency mismatch.
fn observation_glonass_channel(
    observation: &PppCorrectionObservation,
    epoch_index: usize,
    used: (&str, &str),
) -> Result<Option<i8>, PppCorrectionsError> {
    if observation.glonass_channel.is_some() || observation.sat.system != GnssSystem::Glonass {
        return Ok(observation.glonass_channel);
    }
    let band = |code: &str| code.chars().nth(1).unwrap_or('0');
    frequencies::infer_glonass_fdma_channel(&[
        (band(used.0), observation.freq1_hz),
        (band(used.1), observation.freq2_hz),
    ])
    .map_err(|_| PppCorrectionsError::CodeBiasObservable {
        epoch_index,
        sat: observation.sat,
        field: "used observables",
        reason: "frequency mismatch",
    })
}

fn validate_satellite_antenna_pcv_samples(
    options: &SatelliteAntennaOptions,
) -> Result<(), PppCorrectionsError> {
    for antenna in &options.antennas {
        for frequency in &antenna.frequencies {
            for &(nadir_deg, pcv_m) in &frequency.noazi_pcv_m {
                validate::finite(nadir_deg, "satellite antenna noazi_pcv_m")
                    .map_err(ppp_invalid_input)?;
                validate::finite(pcv_m, "satellite antenna noazi_pcv_m")
                    .map_err(ppp_invalid_input)?;
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct FrequencyPairFields {
    freq1: &'static str,
    freq2: &'static str,
    pair: &'static str,
}

fn validate_frequency_pair(
    f1_hz: f64,
    f2_hz: f64,
    fields: FrequencyPairFields,
    invalid: impl Fn(&'static str, &'static str) -> PppCorrectionsError,
) -> Result<(f64, f64), PppCorrectionsError> {
    let f1_hz = validate::finite_positive(f1_hz, fields.freq1)
        .map_err(|e| invalid(e.field(), e.reason()))?;
    let f2_hz = validate::finite_positive(f2_hz, fields.freq2)
        .map_err(|e| invalid(e.field(), e.reason()))?;
    if (f1_hz - f2_hz).abs() < FREQUENCY_DENOMINATOR_EPS_HZ {
        Err(invalid(fields.pair, "must differ"))
    } else {
        Ok((f1_hz, f2_hz))
    }
}

fn sun_moon_at(epoch: CivilDateTime, gate: &Ut1Gate) -> Result<SunMoon, CoverageError> {
    let ts = gate
        .admit(time_scales_at(epoch)?)
        .map_err(|error| match error {
            FrameTransformError::Ut1OutsideCoverage { reason } => {
                CoverageError::OutsideCoverage(reason)
            }
            FrameTransformError::InvalidInput { field, .. } => CoverageError::InvalidInput {
                field,
                kind: TimeScaleInputErrorKind::NonFinite,
            },
        })?;
    Ok(sun_moon_ecef(&ts).expect("validated time scales produce Sun/Moon vectors"))
}

fn time_scales_at(epoch: CivilDateTime) -> Result<TimeScales, CoverageError> {
    let civil = validate::civil_datetime_with_second_policy(
        i64::from(epoch.year),
        i64::from(epoch.month),
        i64::from(epoch.day),
        i64::from(epoch.hour),
        i64::from(epoch.minute),
        epoch.second,
        validate::CivilSecondPolicy::UtcLike,
    )
    .map_err(|error| CoverageError::InvalidInput {
        field: error.field(),
        kind: TimeScaleInputErrorKind::from(&error),
    })?;

    // Permissive here only builds the value; the caller's `Ut1Gate` applies
    // the policy to its `ut1_degraded`.
    TimeScales::from_utc_validated(
        civil.year as i32,
        civil.month as i32,
        civil.day as i32,
        civil.hour as i32,
        civil.minute as i32,
        civil.second,
        ValidityMode::Permissive,
    )
    .map(|validated| validated.value)
}

fn tide_at(
    receiver_ecef_m: [f64; 3],
    epoch: CivilDateTime,
    sun_ecef_m: [f64; 3],
    moon_ecef_m: [f64; 3],
    constants: StationTideConstants,
) -> Result<[f64; 3], TideError> {
    let fhr = epoch.hour as f64 + epoch.minute as f64 / 60.0 + epoch.second / SECONDS_PER_HOUR;
    solid_earth_tide_with_constants(
        &receiver_ecef_m,
        epoch.year,
        epoch.month as i32,
        epoch.day as i32,
        fhr,
        &sun_ecef_m,
        &moon_ecef_m,
        constants,
    )
}

fn pole_tide_at(
    receiver_ecef_m: [f64; 3],
    epoch: CivilDateTime,
    pole: PoleTideOptions,
) -> Result<[f64; 3], TideError> {
    let fhr = epoch.hour as f64 + epoch.minute as f64 / 60.0 + epoch.second / SECONDS_PER_HOUR;
    solid_earth_pole_tide(
        &receiver_ecef_m,
        epoch.year,
        epoch.month as i32,
        epoch.day as i32,
        fhr,
        pole.xp_arcsec,
        pole.yp_arcsec,
    )
}

fn ocean_loading_at(
    receiver_ecef_m: [f64; 3],
    epoch: CivilDateTime,
    blq: &OceanLoadingBlq,
) -> Result<[f64; 3], TideError> {
    let fhr = epoch.hour as f64 + epoch.minute as f64 / 60.0 + epoch.second / SECONDS_PER_HOUR;
    ocean_tide_loading(
        &receiver_ecef_m,
        epoch.year,
        epoch.month as i32,
        epoch.day as i32,
        fhr,
        blq,
    )
}

fn windup_metres(phw_cycles: f64, f1_hz: f64, f2_hz: f64) -> f64 {
    let lam1 = C_M_S / f1_hz;
    let lam2 = C_M_S / f2_hz;
    let gamma = ionosphere_free_gamma(f1_hz, f2_hz);
    (gamma * lam1 - (gamma - 1.0) * lam2) * phw_cycles
}

/// Phase wind-up in cycles, as RTKLIB `windupcorr` forms it. The receiver dipole
/// is `x` north and `y` west in `receiver_neu`, the receiver's local basis:
/// geodetic (ellipsoid-normal), as `windupcorr` takes it from `xyz2enu` at the
/// `ecef2pos` position.
fn windup_cycles(
    pred: &PredictedObservables,
    receiver_ecef_m: [f64; 3],
    receiver_neu: ([f64; 3], [f64; 3], [f64; 3]),
    sun_ecef_m: [f64; 3],
    prev_phw: Option<f64>,
) -> Option<f64> {
    let rs = pred.sat_pos_ecef_m;
    let vs = pred.sat_velocity_m_s;
    let (exs, eys) = sat_yaw(rs, vs, sun_ecef_m)?;
    let ek = unit3(sub3(receiver_ecef_m, rs))?;

    let (n, e, _u) = receiver_neu;
    let exr = n;
    let eyr = neg3(e);

    let eks = cross3(ek, eys);
    let ekr = cross3(ek, eyr);
    let ds = sub3(exs, add3(scale3(ek, dot3(ek, exs)), eks));
    let dr = sub3(exr, sub3(scale3(ek, dot3(ek, exr)), ekr));

    let nds = norm3(ds);
    let ndr = norm3(dr);
    if nds == 0.0 || ndr == 0.0 {
        return None;
    }

    let cosp = clamp(dot3(ds, dr) / nds / ndr);
    let mut ph = libm::acos(cosp) / std::f64::consts::TAU;
    let drs = cross3(ds, dr);
    if dot3(ek, drs) < 0.0 {
        ph = -ph;
    }

    Some(match prev_phw {
        None => ph,
        Some(prev) => ph + (prev - ph + 0.5).floor(),
    })
}

fn sat_yaw(rs: [f64; 3], vs: [f64; 3], sun_ecef_m: [f64; 3]) -> Option<([f64; 3], [f64; 3])> {
    let ri_v = [
        vs[0] - OMEGA_E_DOT_RAD_S * rs[1],
        vs[1] + OMEGA_E_DOT_RAD_S * rs[0],
        vs[2],
    ];
    let n = cross3(rs, ri_v);
    let p = cross3(sun_ecef_m, n);

    let es = unit3(rs)?;
    let esun = unit3(sun_ecef_m)?;
    let en = unit3(n)?;
    let ep = unit3(p)?;

    let beta = beta_angle_from_cos_rad(dot3(esun, en));
    let ee = libm::acos(clamp(dot3(es, ep)));
    let mut mu = PI / 2.0 + if dot3(es, esun) <= 0.0 { -ee } else { ee };

    if mu < -PI / 2.0 {
        mu += std::f64::consts::TAU;
    } else if mu >= PI / 2.0 {
        mu -= std::f64::consts::TAU;
    }

    let yaw = yaw_nominal(beta, mu);
    let ex = cross3(en, es);
    let cosy = libm::cos(yaw);
    let siny = libm::sin(yaw);
    let exs = add3(scale3(en, -siny), scale3(ex, cosy));
    let eys = add3(scale3(en, -cosy), scale3(ex, -siny));
    Some((exs, eys))
}

fn yaw_nominal(beta: f64, mu: f64) -> f64 {
    if beta.abs() < YAW_SINGULARITY_EPS_RAD && mu.abs() < YAW_SINGULARITY_EPS_RAD {
        PI
    } else {
        libm::atan2(-libm::tan(beta), libm::sin(mu)) + PI
    }
}

fn satellite_antenna_correction(
    pred: &PredictedObservables,
    sun_ecef_m: [f64; 3],
    sat: GnssSatelliteId,
    epoch: CivilDateTime,
    options: &SatelliteAntennaOptions,
    frequencies_hz: (f64, f64),
) -> Option<([f64; 3], f64)> {
    let rs = pred.sat_pos_ecef_m;
    let ant = options.antenna_for(sat, epoch)?;

    let (ex, ey, ez) = satellite_sun_fixed_axes(rs, sun_ecef_m)?;

    let off1 = ant.pco(&options.freq1_label)?;
    let off2 = ant.pco(&options.freq2_label)?;
    let gamma = ionosphere_free_gamma(frequencies_hz.0, frequencies_hz.1);

    let dant1 = body_to_ecef(off1, ex, ey, ez);
    let dant2 = body_to_ecef(off2, ex, ey, ez);
    let dant_ecef = sub3(scale3(dant1, gamma), scale3(dant2, gamma - 1.0));
    let pcv_m = nadir_pcv_if(ant, pred, options, gamma)?;

    Some((dant_ecef, pcv_m))
}

/// Convert a satellite body-frame PCO to ECEF using satellite-Sun-fixed axes.
pub(crate) fn satellite_body_pco_to_ecef(
    pco_body_m: [f64; 3],
    sat_position_ecef_m: [f64; 3],
    sun_ecef_m: [f64; 3],
) -> Option<[f64; 3]> {
    let (ex, ey, ez) = satellite_sun_fixed_axes(sat_position_ecef_m, sun_ecef_m)?;
    Some(body_to_ecef(pco_body_m, ex, ey, ez))
}

fn satellite_sun_fixed_axes(
    sat_position_ecef_m: [f64; 3],
    sun_ecef_m: [f64; 3],
) -> Option<([f64; 3], [f64; 3], [f64; 3])> {
    let sat_norm_m = norm3(sat_position_ecef_m);
    if !sat_norm_m.is_finite() || sat_norm_m <= VECTOR_NORM_ZERO_EPS {
        return None;
    }
    let ez = scale3(neg3(sat_position_ecef_m), 1.0 / sat_norm_m);

    let sun_delta_m = sub3(sun_ecef_m, sat_position_ecef_m);
    let sun_delta_norm_m = norm3(sun_delta_m);
    if !sun_delta_norm_m.is_finite() || sun_delta_norm_m <= VECTOR_NORM_ZERO_EPS {
        return None;
    }
    let es = scale3(sun_delta_m, 1.0 / sun_delta_norm_m);

    let normal = cross3(ez, es);
    let normal_norm = norm3(normal);
    if !normal_norm.is_finite() || normal_norm <= VECTOR_NORM_ZERO_EPS {
        return None;
    }
    let ey = scale3(normal, 1.0 / normal_norm);
    let ex = cross3(ey, ez);
    Some((ex, ey, ez))
}

fn body_to_ecef(pco_body_m: [f64; 3], ex: [f64; 3], ey: [f64; 3], ez: [f64; 3]) -> [f64; 3] {
    add3(
        add3(scale3(ex, pco_body_m[0]), scale3(ey, pco_body_m[1])),
        scale3(ez, pco_body_m[2]),
    )
}

fn ionosphere_free_gamma(f1_hz: f64, f2_hz: f64) -> f64 {
    let f1_sq = f1_hz * f1_hz;
    f1_sq / (f1_sq - f2_hz * f2_hz)
}

fn nadir_pcv_if(
    ant: &SatelliteAntenna,
    pred: &PredictedObservables,
    options: &SatelliteAntennaOptions,
    gamma: f64,
) -> Option<f64> {
    let eu = unit3(neg3(pred.los_unit))?;
    let ez = unit3(neg3(pred.sat_pos_ecef_m))?;
    let nadir_deg = libm::acos(clamp(dot3(eu, ez))) * RAD_TO_DEG;
    let p1 = ant.pcv_noazi(&options.freq1_label, nadir_deg)?;
    let p2 = ant.pcv_noazi(&options.freq2_label, nadir_deg)?;
    Some(gamma * p1 - (gamma - 1.0) * p2)
}

impl SatelliteAntennaOptions {
    fn antenna_for(&self, sat: GnssSatelliteId, epoch: CivilDateTime) -> Option<&SatelliteAntenna> {
        self.antennas
            .iter()
            .find(|ant| ant.sat == sat && ant.valid_at(epoch))
    }
}

impl SatelliteAntenna {
    fn valid_at(&self, epoch: CivilDateTime) -> bool {
        let after_from = self
            .valid_from
            .is_none_or(|from| civil_cmp(epoch, from) != std::cmp::Ordering::Less);
        let before_until = self
            .valid_until
            .is_none_or(|until| civil_cmp(epoch, until) != std::cmp::Ordering::Greater);
        after_from && before_until
    }

    fn frequency(&self, label: &str) -> Option<&SatelliteAntennaFrequency> {
        self.frequencies
            .iter()
            .find(|f| f.label.trim() == label.trim())
    }

    fn pco(&self, label: &str) -> Option<[f64; 3]> {
        self.frequency(label).map(|f| f.pco_m)
    }

    fn pcv_noazi(&self, label: &str, zenith_deg: f64) -> Option<f64> {
        let frequency = self.frequency(label)?;
        interpolate_samples(&frequency.noazi_pcv_m, zenith_deg)
    }
}

fn civil_cmp(a: CivilDateTime, b: CivilDateTime) -> std::cmp::Ordering {
    (
        a.year,
        a.month,
        a.day,
        a.hour,
        a.minute,
        ordered_seconds(a.second),
    )
        .cmp(&(
            b.year,
            b.month,
            b.day,
            b.hour,
            b.minute,
            ordered_seconds(b.second),
        ))
}

fn ordered_seconds(second: f64) -> i64 {
    (second * MICROSECONDS_PER_SECOND).round() as i64
}

fn interpolate_samples(samples: &[(f64, f64)], zenith_deg: f64) -> Option<f64> {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    antenna::interpolate_zenith_sorted(&sorted, zenith_deg)
}

fn clamp(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests {
    use super::*;
    use crate::astro::time::split_julian_date;
    use crate::constants::F_L2_HZ;
    use crate::observables::j2000_seconds_from_split;
    use crate::GnssSystem;

    const REAL_CODE_BIA: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bias/CODE.BIA"
    ));

    fn sp3_fixture() -> Sp3 {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read SP3 fixture {path}: {e}"));
        Sp3::parse(&bytes).expect("parse SP3 fixture")
    }

    fn civil(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: f64) -> CivilDateTime {
        CivilDateTime {
            year,
            month,
            day,
            hour,
            minute,
            second,
        }
    }

    fn split_jd(epoch: CivilDateTime) -> (f64, f64) {
        split_julian_date(
            epoch.year,
            i32::from(epoch.month),
            i32::from(epoch.day),
            i32::from(epoch.hour),
            i32::from(epoch.minute),
            epoch.second,
        )
    }

    fn fake_antenna_options(sat: GnssSatelliteId) -> SatelliteAntennaOptions {
        SatelliteAntennaOptions {
            freq1_label: "G01".to_string(),
            freq1_hz: F_L1_HZ,
            freq2_label: "G02".to_string(),
            freq2_hz: F_L2_HZ,
            antennas: vec![SatelliteAntenna {
                sat,
                valid_from: Some(civil(2020, 1, 1, 0, 0, 0.0)),
                valid_until: Some(civil(2021, 1, 1, 0, 0, 0.0)),
                frequencies: vec![
                    SatelliteAntennaFrequency {
                        label: "G01".to_string(),
                        pco_m: [0.1, -0.2, 1.0],
                        noazi_pcv_m: vec![(0.0, 0.001), (5.0, 0.002), (10.0, 0.004)],
                    },
                    SatelliteAntennaFrequency {
                        label: "G02".to_string(),
                        pco_m: [-0.1, 0.3, 0.5],
                        noazi_pcv_m: vec![(0.0, -0.001), (5.0, -0.002), (10.0, -0.003)],
                    },
                ],
            }],
        }
    }

    fn windup_epoch(sat: GnssSatelliteId, freq1_hz: f64, freq2_hz: f64) -> PppCorrectionEpoch {
        let epoch = civil(2020, 6, 24, 12, 0, 0.0);
        let (jd_whole, jd_fraction) = split_jd(epoch);
        PppCorrectionEpoch {
            epoch,
            t_rx_j2000_s: j2000_seconds_from_split(jd_whole, jd_fraction)
                .expect("valid split Julian date"),
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz,
                freq2_hz,
                glonass_channel: None,
            }],
        }
    }

    /// The reference fixture was computed with the transmission epoch rounded to whole
    /// microseconds and the receiver dipole in the geocentric frame. Replayed through
    /// both it agrees to the bit. With the live transmission epoch, which keeps every bit
    /// of the flight time, it differs only through that epoch: the station tide not at
    /// all, and the wind-up and satellite antenna terms by at most what the satellite's
    /// turn over that epoch difference moves them. The live build then differs from that
    /// in the wind-up alone, through the receiver dipole's geodetic frame.
    #[test]
    fn ppp_corrections_match_elixir_reference_fixture() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let epoch = civil(2020, 6, 24, 12, 0, 0.0);
        let (jd_whole, jd_fraction) = split_jd(epoch);
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let epochs = vec![PppCorrectionEpoch {
            epoch,
            t_rx_j2000_s: j2000_seconds_from_split(jd_whole, jd_fraction)
                .expect("valid split Julian date"),
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz: F_L1_HZ,
                freq2_hz: F_L2_HZ,
                glonass_channel: None,
            }],
        }];
        let options = PppCorrectionsOptions {
            solid_earth_tide: true,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: true,
            satellite_antenna: Some(fake_antenna_options(sat)),
            code_bias: None,
        };

        let got = build_rounded_microsecond_replay_with_tide_constants(
            &sp3,
            &epochs,
            receiver,
            &options,
            StationTideConstants::IersRoutine,
        )
        .expect("valid PPP corrections");

        assert_eq!(got.tide.len(), 1);
        assert_eq!(
            got.tide[0].vector_m.map(f64::to_bits),
            [0x3FB8BC98E788ED00, 0x3FAA54D8C1097507, 0x3FB03498C46B3B4F]
        );
        assert_eq!(got.windup_m.len(), 1);
        assert_eq!(got.windup_m[0].value_m.to_bits(), 0xBF808DE79DBD2C16);
        assert_eq!(got.sat_pco_ecef.len(), 1);
        assert_eq!(
            got.sat_pco_ecef[0].vector_m.map(f64::to_bits),
            [0xBFE58ED947570048, 0x3FDEDBB280CEB1BE, 0xBFFE3BCA6A354E4A]
        );
        assert_eq!(got.sat_pcv_m.len(), 1);
        assert_eq!(got.sat_pcv_m[0].value_m.to_bits(), 0x3F77617E95BD232C);

        let conventions_replay = build_rounded_microsecond_replay_with_tide_constants(
            &sp3,
            &epochs,
            receiver,
            &options,
            StationTideConstants::Conventions,
        )
        .expect("valid Conventions PPP corrections");
        let default = build(&sp3, &epochs, receiver, &options).expect("valid PPP corrections");
        assert_eq!(
            default.tide[0].vector_m.map(f64::to_bits),
            conventions_replay.tide[0].vector_m.map(f64::to_bits)
        );
        assert_ne!(
            default.tide[0].vector_m.map(f64::to_bits),
            got.tide[0].vector_m.map(f64::to_bits)
        );
        assert!(default.tide[0]
            .vector_m
            .into_iter()
            .zip(got.tide[0].vector_m)
            .all(|(conventions, routine)| (conventions - routine).abs() <= 0.18e-3));

        let live_geocentric = build_geocentric_dipole_replay(&sp3, &epochs, receiver, &options)
            .expect("valid PPP corrections");
        assert_eq!(
            live_geocentric.tide[0].vector_m.map(f64::to_bits),
            got.tide[0].vector_m.map(f64::to_bits),
            "the station tide does not read the satellite"
        );
        let options_l1 = PredictOptions {
            carrier_hz: F_L1_HZ,
            light_time: true,
            sagnac: true,
        };
        let t_rx = epochs[0].t_rx_j2000_s;
        let exact = crate::observables::predict(&sp3, sat, receiver, t_rx, options_l1)
            .expect("live prediction");
        let rounded = crate::observables::rounded_microsecond_replay::predict(
            &sp3, sat, receiver, t_rx, options_l1,
        )
        .expect("rounded prediction");
        let dt = exact.transmit_time_j2000_s - rounded.transmit_time_j2000_s;
        assert!(
            dt.abs() <= 0.5e-6 + 1.0e-12,
            "the epochs differ by the rounding alone: {dt} s"
        );
        let position = |t: f64| {
            sp3.position_at_j2000_seconds(sat, t)
                .expect("SP3 state")
                .position
                .as_array()
        };
        let (plus, minus, at) = (
            position(exact.transmit_time_j2000_s + 0.5),
            position(exact.transmit_time_j2000_s - 0.5),
            position(exact.transmit_time_j2000_s),
        );
        let speed = norm3([plus[0] - minus[0], plus[1] - minus[1], plus[2] - minus[2]]);
        let turn_rad = 2.0
            * dt.abs()
            * (speed / exact.geometric_range_m
                + speed / norm3(at)
                + crate::constants::OMEGA_E_DOT_RAD_S);
        let metres_per_rad = 10.0;
        let bound_m = metres_per_rad * turn_rad + 1.0e-15;
        for axis in 0..3 {
            let moved =
                live_geocentric.sat_pco_ecef[0].vector_m[axis] - got.sat_pco_ecef[0].vector_m[axis];
            assert!(moved.abs() <= bound_m, "PCO axis {axis} moved {moved} m");
        }
        let pcv_moved = live_geocentric.sat_pcv_m[0].value_m - got.sat_pcv_m[0].value_m;
        assert!(pcv_moved.abs() <= bound_m, "PCV moved {pcv_moved} m");

        let receiver_neu = crate::frame::geocentric_neu_basis(receiver);
        let sun_ecef_m = sun_moon_at(epoch, &Ut1Gate::new(ValidityMode::Strict))
            .expect("valid Sun position")
            .sun;
        let projected_dipoles = |prediction: &PredictedObservables| {
            let line_of_sight = unit3(sub3(receiver, prediction.sat_pos_ecef_m))
                .expect("nonzero receiver-satellite vector");
            let (satellite_x, satellite_y) = sat_yaw(
                prediction.sat_pos_ecef_m,
                prediction.sat_velocity_m_s,
                sun_ecef_m,
            )
            .expect("valid satellite yaw frame");
            let satellite_cross = cross3(line_of_sight, satellite_y);
            let receiver_x = receiver_neu.0;
            let receiver_y = neg3(receiver_neu.1);
            let receiver_cross = cross3(line_of_sight, receiver_y);
            let satellite_dipole = sub3(
                satellite_x,
                add3(
                    scale3(line_of_sight, dot3(line_of_sight, satellite_x)),
                    satellite_cross,
                ),
            );
            let receiver_dipole = sub3(
                receiver_x,
                sub3(
                    scale3(line_of_sight, dot3(line_of_sight, receiver_x)),
                    receiver_cross,
                ),
            );
            (line_of_sight, satellite_dipole, receiver_dipole)
        };
        let (exact_los, exact_satellite_dipole, exact_receiver_dipole) = projected_dipoles(&exact);
        let (rounded_los, rounded_satellite_dipole, rounded_receiver_dipole) =
            projected_dipoles(&rounded);
        let asin_angle_upper = |argument: f64| {
            let argument_upper = libm::nextafter(argument, f64::INFINITY).min(1.0);
            let asin_upper = libm::nextafter(
                libm::asin(argument_upper) + 4.0 * f64::EPSILON,
                f64::INFINITY,
            );
            libm::nextafter(2.0 * asin_upper, f64::INFINITY)
        };
        let conditioned_direction_change = |first: [f64; 3], second: [f64; 3]| {
            let minimum_norm = norm3(first).min(norm3(second));
            assert!(minimum_norm > 0.0, "projected dipole is conditioned");
            let chord = norm3(sub3(first, second));
            asin_angle_upper((chord / minimum_norm).min(1.0))
        };
        let acos_roundoff_bound_rad = |first: [f64; 3], second: [f64; 3]| {
            let product_is_normal_or_exact_zero = |left: f64, right: f64| {
                let product = left * right;
                product.is_normal() || (product == 0.0 && (left == 0.0 || right == 0.0))
            };
            assert!(first.into_iter().all(f64::is_finite));
            assert!(second.into_iter().all(f64::is_finite));
            assert!(first
                .into_iter()
                .all(|value| product_is_normal_or_exact_zero(value, value)));
            assert!(second
                .into_iter()
                .all(|value| product_is_normal_or_exact_zero(value, value)));
            assert!(first
                .into_iter()
                .zip(second)
                .all(|(left, right)| product_is_normal_or_exact_zero(left, right)));
            assert!(norm3(first).is_normal() && norm3(second).is_normal());
            let unit_roundoff = 0.5 * f64::EPSILON;
            let gamma =
                |operations: f64| operations * unit_roundoff / (1.0 - operations * unit_roundoff);
            let dot_error = gamma(5.0);
            let norm_error = gamma(5.0);
            let norm_relative_error = ((1.0 + norm_error).sqrt() * (1.0 + unit_roundoff) - 1.0)
                .max(1.0 - (1.0 - norm_error).sqrt() * (1.0 - unit_roundoff));
            let division_lower =
                (1.0 - norm_relative_error).powi(2) * (1.0 - unit_roundoff).powi(2);
            let cosine_error = dot_error / division_lower + division_lower.recip() - 1.0;
            let cosine_error = cosine_error.min(1.0);
            asin_angle_upper((0.5 * cosine_error).sqrt()) + 4.0 * f64::EPSILON * PI
        };
        let orientation_roundoff_bound =
            |los: [f64; 3], satellite: [f64; 3], receiver: [f64; 3]| {
                let unit_roundoff = 0.5 * f64::EPSILON;
                let gamma = |operations: f64| {
                    operations * unit_roundoff / (1.0 - operations * unit_roundoff)
                };
                let underflow_allowance = 3.0 * f64::from_bits(1);
                let cross_error = [
                    gamma(3.0)
                        * (satellite[1].abs() * receiver[2].abs()
                            + satellite[2].abs() * receiver[1].abs()),
                    gamma(3.0)
                        * (satellite[2].abs() * receiver[0].abs()
                            + satellite[0].abs() * receiver[2].abs()),
                    gamma(3.0)
                        * (satellite[0].abs() * receiver[1].abs()
                            + satellite[1].abs() * receiver[0].abs()),
                ];
                let cross_error = cross_error.map(|error| error + underflow_allowance);
                let cross = cross3(satellite, receiver);
                gamma(5.0)
                    * (los[0].abs() * cross[0].abs()
                        + los[1].abs() * cross[1].abs()
                        + los[2].abs() * cross[2].abs())
                    + los[0].abs() * cross_error[0]
                    + los[1].abs() * cross_error[1]
                    + los[2].abs() * cross_error[2]
                    + 5.0 * f64::from_bits(1)
            };
        let dipole_phase_bound_rad =
            conditioned_direction_change(exact_satellite_dipole, rounded_satellite_dipole)
                + conditioned_direction_change(exact_receiver_dipole, rounded_receiver_dipole)
                + 2.0 * conditioned_direction_change(exact_los, rounded_los)
                + acos_roundoff_bound_rad(exact_satellite_dipole, exact_receiver_dipole)
                + acos_roundoff_bound_rad(rounded_satellite_dipole, rounded_receiver_dipole);
        let projected_phase = |los: [f64; 3], satellite: [f64; 3], receiver: [f64; 3]| {
            let cosp = clamp(dot3(satellite, receiver) / norm3(satellite) / norm3(receiver));
            let mut phase = libm::acos(cosp);
            if dot3(los, cross3(satellite, receiver)) < 0.0 {
                phase = -phase;
            }
            phase
        };
        let exact_phase = projected_phase(exact_los, exact_satellite_dipole, exact_receiver_dipole);
        let rounded_phase = projected_phase(
            rounded_los,
            rounded_satellite_dipole,
            rounded_receiver_dipole,
        );
        let signed_phase_uncertainty =
            |los: [f64; 3], satellite: [f64; 3], receiver: [f64; 3], phase: f64| {
                let orientation = dot3(los, cross3(satellite, receiver));
                if orientation.abs() <= orientation_roundoff_bound(los, satellite, receiver) {
                    2.0 * (phase.abs() + acos_roundoff_bound_rad(satellite, receiver))
                } else {
                    0.0
                }
            };
        let dipole_phase_bound_rad = dipole_phase_bound_rad
            + signed_phase_uncertainty(
                exact_los,
                exact_satellite_dipole,
                exact_receiver_dipole,
                exact_phase,
            )
            + signed_phase_uncertainty(
                rounded_los,
                rounded_satellite_dipole,
                rounded_receiver_dipole,
                rounded_phase,
            );
        assert!(
            dipole_phase_bound_rad < PI - exact_phase.abs()
                && dipole_phase_bound_rad < PI - rounded_phase.abs(),
            "wind-up phase remains on one principal branch"
        );
        let metres_per_cycle = windup_metres(1.0, F_L1_HZ, F_L2_HZ).abs();
        let conversion_roundoff_bound_m = |first_phase: f64, second_phase: f64| {
            let unit_roundoff = 0.5 * f64::EPSILON;
            let gamma_three = 3.0 * unit_roundoff / (1.0 - 3.0 * unit_roundoff);
            gamma_three * metres_per_cycle * (first_phase.abs() + second_phase.abs())
                / std::f64::consts::TAU
        };
        let windup_moved = live_geocentric.windup_m[0].value_m - got.windup_m[0].value_m;
        let windup_bound_m = metres_per_cycle * dipole_phase_bound_rad / std::f64::consts::TAU
            + conversion_roundoff_bound_m(exact_phase, rounded_phase);
        assert!(
            windup_moved.abs() <= windup_bound_m,
            "wind-up moved {windup_moved} m, above conditioned projected-dipole bound {windup_bound_m} m"
        );

        let live = build(&sp3, &epochs, receiver, &options).expect("valid PPP corrections");
        assert_eq!(
            live.tide[0].vector_m.map(f64::to_bits),
            conventions_replay.tide[0].vector_m.map(f64::to_bits)
        );
        assert_eq!(
            live.sat_pco_ecef[0].vector_m.map(f64::to_bits),
            live_geocentric.sat_pco_ecef[0].vector_m.map(f64::to_bits)
        );
        assert_eq!(
            live.sat_pcv_m[0].value_m.to_bits(),
            live_geocentric.sat_pcv_m[0].value_m.to_bits()
        );
        let (_, _, geodetic_up) =
            crate::estimation::substrate::frames::geodetic_neu_basis(receiver);
        let geocentric_up = crate::frame::geocentric_up(receiver);
        let vertical_angle_rad = libm::atan2(
            norm3(cross3(geodetic_up, geocentric_up)),
            dot3(geodetic_up, geocentric_up),
        );
        assert!(vertical_angle_rad > 1.0e-3 && vertical_angle_rad < 3.4e-3);
        let frame_moved = live.windup_m[0].value_m - live_geocentric.windup_m[0].value_m;
        assert!(frame_moved != 0.0, "the dipole frame moves the wind-up");
        let line_of_sight = unit3(sub3(receiver, exact.sat_pos_ecef_m)).expect("line of sight");
        let geodetic_neu = crate::estimation::substrate::frames::geodetic_neu_basis(receiver);
        let geocentric_neu = crate::frame::geocentric_neu_basis(receiver);
        let receiver_dipole = |neu: ([f64; 3], [f64; 3], [f64; 3])| {
            let (north, east, _) = neu;
            add3(
                sub3(north, scale3(line_of_sight, dot3(line_of_sight, north))),
                cross3(line_of_sight, neg3(east)),
            )
        };
        let geodetic_dipole = receiver_dipole(geodetic_neu);
        let geocentric_dipole = receiver_dipole(geocentric_neu);
        let conditioning = norm3(geodetic_dipole).min(norm3(geocentric_dipole));
        let dipole_delta_bound = 4.0 * libm::sin(0.5 * vertical_angle_rad);
        let projected_angle_bound = if conditioning > dipole_delta_bound {
            asin_angle_upper((dipole_delta_bound / conditioning).min(1.0))
        } else {
            std::f64::consts::PI
        };
        let geodetic_frame_phase =
            projected_phase(line_of_sight, exact_satellite_dipole, geodetic_dipole);
        let geocentric_frame_phase =
            projected_phase(line_of_sight, exact_satellite_dipole, geocentric_dipole);
        let frame_phase_bound = projected_angle_bound
            + acos_roundoff_bound_rad(exact_satellite_dipole, geodetic_dipole)
            + acos_roundoff_bound_rad(exact_satellite_dipole, geocentric_dipole)
            + signed_phase_uncertainty(
                line_of_sight,
                exact_satellite_dipole,
                geodetic_dipole,
                geodetic_frame_phase,
            )
            + signed_phase_uncertainty(
                line_of_sight,
                exact_satellite_dipole,
                geocentric_dipole,
                geocentric_frame_phase,
            );
        assert!(
            frame_phase_bound < PI - geodetic_frame_phase.abs()
                && frame_phase_bound < PI - geocentric_frame_phase.abs(),
            "receiver-frame wind-up remains on one principal branch"
        );
        let frame_bound_m = metres_per_cycle * frame_phase_bound / std::f64::consts::TAU
            + conversion_roundoff_bound_m(geodetic_frame_phase, geocentric_frame_phase);
        assert!(
            frame_moved.abs() <= frame_bound_m,
            "wind-up moved {frame_moved} m, above conditioned dipole bound {frame_bound_m} m"
        );
    }

    #[test]
    fn pole_tide_correction_is_emitted_and_matches_standalone() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let epoch = civil(2020, 6, 24, 12, 0, 0.0);
        let (jd_whole, jd_fraction) = split_jd(epoch);
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let epochs = vec![PppCorrectionEpoch {
            epoch,
            t_rx_j2000_s: j2000_seconds_from_split(jd_whole, jd_fraction)
                .expect("valid split Julian date"),
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz: F_L1_HZ,
                freq2_hz: F_L2_HZ,
                glonass_channel: None,
            }],
        }];
        let pole = PoleTideOptions {
            xp_arcsec: 0.169_051,
            yp_arcsec: 0.411_760,
        };
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: Some(pole),
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: None,
        };

        let got = build(&sp3, &epochs, receiver, &options).expect("valid PPP corrections");

        assert_eq!(got.pole_tide.len(), 1);
        assert_eq!(got.pole_tide[0].epoch_index, 0);
        let expected = crate::tides::solid_earth_pole_tide(
            &receiver,
            2020,
            6,
            24,
            12.0,
            pole.xp_arcsec,
            pole.yp_arcsec,
        )
        .expect("valid pole tide");
        assert_eq!(got.pole_tide[0].vector_m, expected);
        // Pole tide is opt-in and independent of the solid-earth tide.
        assert!(got.tide.is_empty());
    }

    // ZIM2 ocean-loading BLQ (GOT4.7), OLFG/Scherneck Onsala 2020-Jun-25,
    // holt.oso.chalmers.se; used here purely as a finite, real-valued BLQ to
    // exercise the precompute plumbing (the receiver below is not ZIM2).
    fn zim2_blq() -> OceanLoadingBlq {
        OceanLoadingBlq {
            amplitude_m: [
                [
                    0.00693, 0.00228, 0.00148, 0.00061, 0.00220, 0.00094, 0.00070, 0.00001,
                    0.00047, 0.00025, 0.00019,
                ],
                [
                    0.00272, 0.00076, 0.00061, 0.00020, 0.00036, 0.00025, 0.00011, 0.00005,
                    0.00004, 0.00001, 0.00002,
                ],
                [
                    0.00061, 0.00026, 0.00010, 0.00009, 0.00025, 0.00002, 0.00008, 0.00003,
                    0.00002, 0.00000, 0.00001,
                ],
            ],
            phase_deg: [
                [
                    -72.3, -44.2, -90.8, -44.1, -62.9, -94.5, -64.3, 171.0, 3.4, 3.6, 1.1,
                ],
                [
                    84.3, 115.4, 63.3, 113.7, 98.6, 20.7, 94.2, -44.5, -170.0, -162.7, -177.8,
                ],
                [
                    -29.3, 1.7, -44.0, -4.2, 44.2, -39.1, 43.7, 170.1, -93.3, -118.3, -176.4,
                ],
            ],
        }
    }

    #[test]
    fn ocean_loading_correction_is_emitted_and_matches_standalone() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let epoch = civil(2020, 6, 24, 12, 0, 0.0);
        let (jd_whole, jd_fraction) = split_jd(epoch);
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let epochs = vec![PppCorrectionEpoch {
            epoch,
            t_rx_j2000_s: j2000_seconds_from_split(jd_whole, jd_fraction)
                .expect("valid split Julian date"),
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz: F_L1_HZ,
                freq2_hz: F_L2_HZ,
                glonass_channel: None,
            }],
        }];
        let blq = zim2_blq();
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: Some(blq),
            phase_windup: false,
            satellite_antenna: None,
            code_bias: None,
        };

        let got = build(&sp3, &epochs, receiver, &options).expect("valid PPP corrections");

        assert_eq!(got.ocean_loading.len(), 1);
        assert_eq!(got.ocean_loading[0].epoch_index, 0);
        let expected = crate::tides::ocean_tide_loading(&receiver, 2020, 6, 24, 12.0, &blq)
            .expect("valid ocean loading");
        assert_eq!(got.ocean_loading[0].vector_m, expected);
        // Ocean loading is opt-in and independent of the other corrections.
        assert!(got.tide.is_empty());
        assert!(got.pole_tide.is_empty());
    }

    #[test]
    fn pole_or_ocean_only_skips_sun_moon_and_prediction() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];

        // An epoch crafted so the Sun/Moon and the per-observation predict path
        // would BOTH fail if they ran: the date is past the embedded EOP
        // coverage (sun_moon_at -> Epoch::OutsideCoverage) and t_rx is non-finite
        // (predict -> InvalidInput). Pole tide and ocean loading are pure station
        // displacements needing neither, so a pole/ocean-only build must still
        // succeed. Their only date requirement is a valid civil date, which
        // 2100-01-01 satisfies.
        let epochs = vec![PppCorrectionEpoch {
            epoch: civil(2100, 1, 1, 12, 0, 0.0),
            t_rx_j2000_s: f64::NAN,
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz: F_L1_HZ,
                freq2_hz: F_L2_HZ,
                glonass_channel: None,
            }],
        }];

        // Pole tide only.
        let pole = PoleTideOptions {
            xp_arcsec: 0.169_051,
            yp_arcsec: 0.411_760,
        };
        let got = build(
            &sp3,
            &epochs,
            receiver,
            &PppCorrectionsOptions {
                solid_earth_tide: false,
                pole_tide: Some(pole),
                ocean_loading: None,
                phase_windup: false,
                satellite_antenna: None,
                code_bias: None,
            },
        )
        .expect("pole-only build must not touch the Sun/Moon or predict paths");
        assert_eq!(got.pole_tide.len(), 1);
        assert!(got.tide.is_empty());
        assert!(got.ocean_loading.is_empty());
        assert!(got.windup_m.is_empty());
        assert!(got.sat_pco_ecef.is_empty());
        assert!(got.sat_pcv_m.is_empty());

        // Ocean loading only.
        let blq = zim2_blq();
        let got = build(
            &sp3,
            &epochs,
            receiver,
            &PppCorrectionsOptions {
                solid_earth_tide: false,
                pole_tide: None,
                ocean_loading: Some(blq),
                phase_windup: false,
                satellite_antenna: None,
                code_bias: None,
            },
        )
        .expect("ocean-only build must not touch the Sun/Moon or predict paths");
        assert_eq!(got.ocean_loading.len(), 1);
        assert!(got.tide.is_empty());
        assert!(got.pole_tide.is_empty());
        assert!(got.windup_m.is_empty());
        assert!(got.sat_pco_ecef.is_empty());
        assert!(got.sat_pcv_m.is_empty());
    }

    #[test]
    fn phase_windup_rejects_invalid_observation_frequency_pairs() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: true,
            satellite_antenna: None,
            code_bias: None,
        };
        let cases = [
            (0.0, F_L2_HZ, "phase wind-up freq1_hz", "not positive"),
            (-F_L1_HZ, F_L2_HZ, "phase wind-up freq1_hz", "not positive"),
            (
                F_L1_HZ,
                F_L1_HZ,
                "phase wind-up frequency pair",
                "must differ",
            ),
        ];

        for (freq1_hz, freq2_hz, field, reason) in cases {
            let epochs = vec![windup_epoch(sat, freq1_hz, freq2_hz)];
            let err = build(&sp3, &epochs, receiver, &options)
                .expect_err("invalid phase wind-up frequencies must error");

            assert_eq!(
                err,
                PppCorrectionsError::WindupFrequency {
                    epoch_index: 0,
                    sat,
                    field,
                    reason,
                }
            );
        }
    }

    #[test]
    fn phase_windup_observation_frequency_pair_computes_finite_correction() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: true,
            satellite_antenna: None,
            code_bias: None,
        };
        let epochs = vec![windup_epoch(sat, F_L1_HZ, F_L2_HZ)];

        let got =
            build(&sp3, &epochs, receiver, &options).expect("valid phase wind-up frequencies");

        assert_eq!(got.windup_m.len(), 1);
        assert!(got.windup_m[0].value_m.is_finite());
    }

    fn code_bias_product() -> crate::bias::BiasSet {
        let text = "\
%=BIA 1.00 TST 2020:176:00000 TST 2020:176:00000 2020:177:00000 A 00000003
+FILE/REFERENCE
 DESCRIPTION TEST
-FILE/REFERENCE
+BIAS/DESCRIPTION
 BIAS_MODE ABSOLUTE
 TIME_SYSTEM G
 SATELLITE_CLOCK_REFERENCE_OBSERVABLES G C1W C2W
-BIAS/DESCRIPTION
+BIAS/SOLUTION
 OSB  G021 G21           C1C       2020:176:00000 2020:177:00000 ns     -1.234567890000E+00 2.00000E-02
 OSB  G021 G21           C1W       2020:176:00000 2020:177:00000 ns      5.600000000000E-01 2.00000E-02
 OSB  G021 G21           C2W       2020:176:00000 2020:177:00000 ns     -3.000000000000E-01 2.00000E-02
-BIAS/SOLUTION
%=ENDBIA
";
        crate::bias::BiasSet::parse_bias_sinex(text.as_bytes())
            .expect("parse code-bias product")
            .value
    }

    fn real_code_bias_product() -> crate::bias::BiasSet {
        crate::bias::BiasSet::parse_bias_sinex(REAL_CODE_BIA)
            .expect("parse real CODE Bias-SINEX product")
            .value
    }

    /// A GPS satellite 21 OSB row at the Bias-SINEX 1.00 section 4.8
    /// columns: OBS1 at 25, the interval at 35 and 50, the unit at 65, the
    /// estimate right-aligned in 70..91 and its sigma in 92..103.
    fn g21_osb_row(obs: &str, start: &str, end: &str, value: &str) -> String {
        let row = format!(
            " OSB  G021 G21           {obs:<4}      {start} {end} ns   {value:>21} {:>11}",
            "2.00000E-02"
        );
        assert_eq!(row.len(), 103);
        row
    }

    /// A strictly conformant product with the given `TIME_SYSTEM` label and
    /// rows, read leniently when it has no label.
    fn g21_product(time_system: Option<&str>, rows: &[String]) -> crate::bias::BiasSet {
        let mut lines = vec![
            format!(
                "%=BIA 1.00 TST 2020:176:00000 TST 2020:176:00000 2020:177:00000 A {:08}",
                rows.len()
            ),
            "+FILE/REFERENCE".to_string(),
            " DESCRIPTION TEST".to_string(),
            "-FILE/REFERENCE".to_string(),
            "+BIAS/DESCRIPTION".to_string(),
            " BIAS_MODE ABSOLUTE".to_string(),
        ];
        if let Some(label) = time_system {
            lines.push(format!(" TIME_SYSTEM {label}"));
        }
        lines.push(" SATELLITE_CLOCK_REFERENCE_OBSERVABLES G C1W C2W".to_string());
        lines.push("-BIAS/DESCRIPTION".to_string());
        lines.push("+BIAS/SOLUTION".to_string());
        lines.extend(rows.iter().cloned());
        lines.push("-BIAS/SOLUTION".to_string());
        lines.push("%=ENDBIA".to_string());
        let text = lines.join("\n");
        crate::bias::BiasSet::parse_bias_sinex_with_policy(
            text.as_bytes(),
            crate::bias::BiasReadPolicy::Lenient,
        )
        .expect("parse code-bias product")
        .value
    }

    fn code_bias_warnings(got: &PppCorrections) -> Vec<crate::format::WarningKind> {
        got.diagnostics
            .warnings
            .iter()
            .map(|warning| warning.kind)
            .collect()
    }

    fn code_bias_epoch(sat: GnssSatelliteId) -> Vec<PppCorrectionEpoch> {
        let epoch = civil(2020, 6, 24, 12, 0, 0.0);
        let (jd_whole, jd_fraction) = split_jd(epoch);
        vec![PppCorrectionEpoch {
            epoch,
            t_rx_j2000_s: j2000_seconds_from_split(jd_whole, jd_fraction)
                .expect("valid split Julian date"),
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz: F_L1_HZ,
                freq2_hz: F_L2_HZ,
                glonass_channel: None,
            }],
        }]
    }

    fn code_bias_options(bias_set: crate::bias::BiasSet, used: (&str, &str)) -> CodeBiasOptions {
        let mut used_observables_default = BTreeMap::new();
        used_observables_default.insert(GnssSystem::Gps, (used.0.to_string(), used.1.to_string()));
        CodeBiasOptions {
            bias_set,
            used_observables_per_sat: BTreeMap::new(),
            used_observables_default,
            clock_reference: None,
        }
    }

    #[test]
    fn code_bias_builds_exact_zero_for_matched_clock_datum() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let epochs = code_bias_epoch(sat);
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: Some(code_bias_options(code_bias_product(), ("C1W", "C2W"))),
        };

        let got = build(&sp3, &epochs, receiver, &options).expect("code-bias build");

        assert_eq!(got.code_bias_m.len(), 1);
        assert_eq!(got.code_bias_m[0].value_m.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn code_bias_reads_the_receiver_epoch_on_the_product_time_scale() {
        // The product is in UTC and the SP3 epochs in GPS time. 12:00:00 GPST
        // on 2020-06-24 is 11:59:42 UTC (TAI-UTC 37 s, TAI-GPST 19 s), so the
        // C1C record ending at 11:59:50 UTC applies, not the one after it.
        let sp3 = sp3_fixture();
        assert_eq!(sp3.header.time_scale, TimeScale::Gpst);
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let rows = [
            g21_osb_row("C1C", "2020:176:00000", "2020:176:43190", "-1.0"),
            g21_osb_row("C1C", "2020:176:43190", "2020:177:00000", "5.0"),
            g21_osb_row("C1W", "2020:176:00000", "2020:177:00000", "0.56"),
            g21_osb_row("C2W", "2020:176:00000", "2020:177:00000", "-0.3"),
        ];
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: Some(code_bias_options(
                g21_product(Some("UTC"), &rows),
                ("C1C", "C2W"),
            )),
        };

        let got = build(&sp3, &code_bias_epoch(sat), receiver, &options).expect("build");
        let alpha = F_L1_HZ * F_L1_HZ / (F_L1_HZ * F_L1_HZ - F_L2_HZ * F_L2_HZ);
        let beta = -(F_L2_HZ * F_L2_HZ) / (F_L1_HZ * F_L1_HZ - F_L2_HZ * F_L2_HZ);
        let used_if = alpha * -(1.0_f64 * 1.0e-9) + beta * (-0.3_f64 * 1.0e-9);
        let ref_if = alpha * (0.56_f64 * 1.0e-9) + beta * (-0.3_f64 * 1.0e-9);
        let expected = (used_if - ref_if) * C_M_S;
        assert_eq!(got.code_bias_m.len(), 1);
        assert_eq!(got.code_bias_m[0].value_m.to_bits(), expected.to_bits());
    }

    #[test]
    fn code_bias_without_a_time_system_warns_per_observation() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let rows = [
            g21_osb_row("C1C", "2020:176:00000", "2020:177:00000", "-1.0"),
            g21_osb_row("C1W", "2020:176:00000", "2020:177:00000", "0.56"),
            g21_osb_row("C2W", "2020:176:00000", "2020:177:00000", "-0.3"),
        ];
        let product = g21_product(None, &rows);
        assert_eq!(product.time_scale(), None);
        let options = |used: (&str, &str)| PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: Some(code_bias_options(product.clone(), used)),
        };

        // No scale to read the epoch on: no correction, a warning, no error.
        let got = build(
            &sp3,
            &code_bias_epoch(sat),
            receiver,
            &options(("C1C", "C2W")),
        )
        .expect("build without a time system");
        assert!(got.code_bias_m.is_empty());
        assert_eq!(
            code_bias_warnings(&got),
            vec![crate::format::WarningKind::MissingMetadata]
        );

        // The clock datum itself is exactly zero whatever the time system.
        let got = build(
            &sp3,
            &code_bias_epoch(sat),
            receiver,
            &options(("C1W", "C2W")),
        )
        .expect("matched datum");
        assert_eq!(got.code_bias_m.len(), 1);
        assert_eq!(got.code_bias_m[0].value_m.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn ambiguous_code_bias_records_warn_as_a_mismatch() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let rows = [
            g21_osb_row("C1C", "2020:176:00000", "2020:177:00000", "-1.0"),
            g21_osb_row("C1C", "2020:176:00000", "2020:177:00000", "-2.0"),
            g21_osb_row("C1W", "2020:176:00000", "2020:177:00000", "0.56"),
            g21_osb_row("C2W", "2020:176:00000", "2020:177:00000", "-0.3"),
        ];
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: Some(code_bias_options(
                g21_product(Some("G"), &rows),
                ("C1C", "C2W"),
            )),
        };

        let got = build(&sp3, &code_bias_epoch(sat), receiver, &options).expect("build");
        assert!(got.code_bias_m.is_empty());
        assert_eq!(
            code_bias_warnings(&got),
            vec![crate::format::WarningKind::Mismatch]
        );
    }

    #[test]
    fn code_bias_builds_mismatched_pair_scalar() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let epochs = code_bias_epoch(sat);
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: Some(code_bias_options(code_bias_product(), ("C1C", "C2W"))),
        };

        let got = build(&sp3, &epochs, receiver, &options).expect("code-bias build");
        let alpha = F_L1_HZ * F_L1_HZ / (F_L1_HZ * F_L1_HZ - F_L2_HZ * F_L2_HZ);
        let beta = -(F_L2_HZ * F_L2_HZ) / (F_L1_HZ * F_L1_HZ - F_L2_HZ * F_L2_HZ);
        let used_if =
            alpha * (-1.234567890000_f64 * 1.0e-9) + beta * (-0.300000000000_f64 * 1.0e-9);
        let ref_if = alpha * (0.560000000000_f64 * 1.0e-9) + beta * (-0.300000000000_f64 * 1.0e-9);
        let expected = (used_if - ref_if) * C_M_S;

        assert_eq!(got.code_bias_m.len(), 1);
        assert_eq!(got.code_bias_m[0].value_m.to_bits(), expected.to_bits());
    }

    #[test]
    fn code_bias_build_applies_real_glonass_osb_with_fdma_channel() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Glonass, 2).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let channel = -4;
        let freq1_hz = frequencies::rinex_observation_frequency_hz(
            GnssSystem::Glonass,
            "C1C",
            3.04,
            Some(channel),
        )
        .expect("GLONASS G1 frequency");
        let freq2_hz = frequencies::rinex_observation_frequency_hz(
            GnssSystem::Glonass,
            "C2C",
            3.04,
            Some(channel),
        )
        .expect("GLONASS G2 frequency");
        let epoch = civil(2026, 6, 24, 12, 0, 0.0);
        let (jd_whole, jd_fraction) = split_jd(epoch);
        let epochs = vec![PppCorrectionEpoch {
            epoch,
            t_rx_j2000_s: j2000_seconds_from_split(jd_whole, jd_fraction)
                .expect("valid split Julian date"),
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz,
                freq2_hz,
                glonass_channel: Some(channel),
            }],
        }];
        let mut used_observables_default = BTreeMap::new();
        used_observables_default
            .insert(GnssSystem::Glonass, ("C1C".to_string(), "C2C".to_string()));
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: Some(CodeBiasOptions {
                bias_set: real_code_bias_product(),
                used_observables_per_sat: BTreeMap::new(),
                used_observables_default,
                clock_reference: None,
            }),
        };

        let got = build(&sp3, &epochs, receiver, &options).expect("GLONASS code-bias build");
        let (alpha, beta) = crate::bias::ionosphere_free_coefficients(freq1_hz, freq2_hz).unwrap();
        let used_if = alpha * (0.2114_f64 * 1.0e-9) + beta * (2.6597_f64 * 1.0e-9);
        let ref_if = alpha * (1.7840_f64 * 1.0e-9) + beta * (2.9490_f64 * 1.0e-9);
        let expected = (used_if - ref_if) * C_M_S;

        assert_eq!(got.code_bias_m.len(), 1);
        assert_eq!(got.code_bias_m[0].sat, sat);
        assert_eq!(got.code_bias_m[0].value_m.to_bits(), expected.to_bits());
    }

    #[test]
    fn code_bias_build_requires_clock_reference() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let epochs = code_bias_epoch(sat);
        let dcb = crate::bias::BiasSet::parse_code_dcb(
            b"# DCB P1-C1 2020-06 G\n G21 1.000 0.100\n",
            Some(crate::bias::CodeDcbOptions {
                pair: ("P1".to_string(), "C1".to_string()),
                year: 2020,
                month: 6,
                time_scale: crate::astro::time::model::TimeScale::Gpst,
                receiver_system: None,
            }),
        )
        .expect("parse DCB")
        .value;
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: Some(code_bias_options(dcb, ("C1C", "C2W"))),
        };

        let err = build(&sp3, &epochs, receiver, &options)
            .expect_err("missing clock reference must error");
        assert!(matches!(
            err,
            PppCorrectionsError::Bias {
                source: BiasError::MissingClockReference
            }
        ));
    }

    #[test]
    fn satellite_antenna_rejects_invalid_frequency_pairs_without_windup() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let cases = [
            (0.0, F_L2_HZ, "satellite antenna freq1_hz", "not positive"),
            (
                F_L1_HZ,
                f64::INFINITY,
                "satellite antenna freq2_hz",
                "not finite",
            ),
            (
                f64::NAN,
                F_L2_HZ,
                "satellite antenna freq1_hz",
                "not finite",
            ),
            (
                F_L1_HZ,
                F_L1_HZ,
                "satellite antenna frequency pair",
                "must differ",
            ),
        ];

        for (freq1_hz, freq2_hz, field, reason) in cases {
            let mut antenna = fake_antenna_options(sat);
            antenna.freq1_hz = freq1_hz;
            antenna.freq2_hz = freq2_hz;
            let options = PppCorrectionsOptions {
                solid_earth_tide: false,
                pole_tide: None,
                ocean_loading: None,
                phase_windup: false,
                satellite_antenna: Some(antenna),
                code_bias: None,
            };
            let epochs = vec![windup_epoch(sat, F_L1_HZ, F_L2_HZ)];

            let err = build(&sp3, &epochs, receiver, &options)
                .expect_err("invalid satellite antenna frequencies must error");

            assert_eq!(
                err,
                PppCorrectionsError::SatelliteAntennaFrequency { field, reason }
            );
        }
    }

    #[test]
    fn satellite_antenna_frequency_pair_computes_finite_corrections_without_windup() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: Some(fake_antenna_options(sat)),
            code_bias: None,
        };
        let epochs = vec![windup_epoch(sat, F_L1_HZ, F_L2_HZ)];

        let got =
            build(&sp3, &epochs, receiver, &options).expect("valid satellite antenna frequencies");

        assert!(got.windup_m.is_empty());
        assert_eq!(got.sat_pco_ecef.len(), 1);
        assert!(got.sat_pco_ecef[0]
            .vector_m
            .iter()
            .all(|value| value.is_finite()));
        assert_eq!(got.sat_pcv_m.len(), 1);
        assert!(got.sat_pcv_m[0].value_m.is_finite());
    }

    #[test]
    fn satellite_antenna_rejects_non_finite_pcv_samples() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let mut antenna = fake_antenna_options(sat);
        antenna.antennas[0].frequencies[0].noazi_pcv_m[1] = (5.0, f64::NAN);
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: Some(antenna),
            code_bias: None,
        };
        let epochs = vec![windup_epoch(sat, F_L1_HZ, F_L2_HZ)];

        let err = build(&sp3, &epochs, receiver, &options)
            .expect_err("non-finite satellite PCV samples must error");

        assert_eq!(
            err,
            PppCorrectionsError::InvalidInput {
                field: "satellite antenna noazi_pcv_m",
                reason: "not finite",
            }
        );
    }

    #[test]
    fn satellite_antenna_empty_pcv_grid_is_not_materialized_as_zero() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let epoch = civil(2020, 6, 24, 12, 0, 0.0);
        let (jd_whole, jd_fraction) = split_jd(epoch);
        let receiver = [3_512_900.0, 780_500.0, 5_248_700.0];
        let epochs = vec![PppCorrectionEpoch {
            epoch,
            t_rx_j2000_s: j2000_seconds_from_split(jd_whole, jd_fraction)
                .expect("valid split Julian date"),
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz: F_L1_HZ,
                freq2_hz: F_L2_HZ,
                glonass_channel: None,
            }],
        }];
        let mut antenna = fake_antenna_options(sat);
        antenna.antennas[0].frequencies[0].noazi_pcv_m.clear();
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: Some(antenna),
            code_bias: None,
        };

        let got = build(&sp3, &epochs, receiver, &options).expect("valid PPP corrections");

        assert!(got.sat_pco_ecef.is_empty());
        assert!(got.sat_pcv_m.is_empty());
    }

    #[test]
    fn build_rejects_non_finite_receive_time_for_satellite_corrections() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: true,
            satellite_antenna: None,
            code_bias: None,
        };
        let epochs = vec![PppCorrectionEpoch {
            epoch: civil(2020, 6, 24, 12, 0, 0.0),
            t_rx_j2000_s: f64::NAN,
            observations: vec![PppCorrectionObservation {
                sat,
                freq1_hz: F_L1_HZ,
                freq2_hz: F_L2_HZ,
                glonass_channel: None,
            }],
        }];

        let err = build(
            &sp3,
            &epochs,
            [3_512_900.0, 780_500.0, 5_248_700.0],
            &options,
        )
        .expect_err("non-finite receive time must be reported");

        assert_eq!(
            err,
            PppCorrectionsError::InvalidInput {
                field: "t_rx_j2000_s",
                reason: "not finite",
            }
        );
    }

    #[test]
    fn noazi_pcv_interpolation_clamps_and_interpolates() {
        let samples = vec![(10.0, 4.0), (0.0, 1.0), (5.0, 2.0)];

        assert_eq!(interpolate_samples(&samples, -1.0), Some(1.0));
        assert_eq!(interpolate_samples(&samples, 2.5), Some(1.5));
        assert_eq!(interpolate_samples(&samples, 99.0), Some(4.0));
    }

    #[test]
    fn build_rejects_invalid_receiver_state_before_disabled_short_circuit() {
        let sp3 = sp3_fixture();
        let options = PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: None,
        };

        for (receiver, field, reason) in [
            (
                [f64::NAN, 780_500.0, 5_248_700.0],
                "receiver_ecef_m",
                "not finite",
            ),
            ([0.0, 0.0, 0.0], "receiver radius_m", "not positive"),
        ] {
            let err = build(&sp3, &[], receiver, &options)
                .expect_err("invalid receiver state must error before empty success");

            assert_eq!(err, PppCorrectionsError::InvalidInput { field, reason });
        }
    }

    #[test]
    fn build_rejects_invalid_correction_epoch_without_panicking() {
        let sp3 = sp3_fixture();
        let options = PppCorrectionsOptions {
            solid_earth_tide: true,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: None,
        };
        let epochs = vec![PppCorrectionEpoch {
            epoch: civil(2021, 2, 29, 12, 0, 0.0),
            t_rx_j2000_s: 0.0,
            observations: Vec::new(),
        }];

        let err = build(
            &sp3,
            &epochs,
            [3_512_900.0, 780_500.0, 5_248_700.0],
            &options,
        )
        .expect_err("invalid PPP correction epoch must return an error");

        assert_eq!(
            err,
            PppCorrectionsError::Epoch {
                epoch_index: 0,
                source: CoverageError::InvalidInput {
                    field: "civil datetime",
                    kind: TimeScaleInputErrorKind::InvalidCivilDate,
                },
            }
        );
    }

    #[test]
    fn build_rejects_non_finite_correction_epoch_without_panicking() {
        let sp3 = sp3_fixture();
        let options = PppCorrectionsOptions {
            solid_earth_tide: true,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: None,
        };
        let epochs = vec![PppCorrectionEpoch {
            epoch: civil(2020, 6, 24, 12, 0, f64::NAN),
            t_rx_j2000_s: 0.0,
            observations: Vec::new(),
        }];

        let err = build(
            &sp3,
            &epochs,
            [3_512_900.0, 780_500.0, 5_248_700.0],
            &options,
        )
        .expect_err("non-finite PPP correction epoch must return an error");

        assert_eq!(
            err,
            PppCorrectionsError::Epoch {
                epoch_index: 0,
                source: CoverageError::InvalidInput {
                    field: "civil datetime",
                    kind: TimeScaleInputErrorKind::NonFinite,
                },
            }
        );
    }

    #[test]
    fn build_rejects_correction_epoch_after_eop_coverage() {
        let sp3 = sp3_fixture();
        let options = PppCorrectionsOptions {
            solid_earth_tide: true,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: None,
        };
        let epochs = vec![PppCorrectionEpoch {
            epoch: civil(2100, 1, 1, 0, 0, 0.0),
            t_rx_j2000_s: 0.0,
            observations: Vec::new(),
        }];

        let err = build(
            &sp3,
            &epochs,
            [3_512_900.0, 780_500.0, 5_248_700.0],
            &options,
        )
        .expect_err("post-coverage PPP correction epoch must return an error");

        assert_eq!(
            err,
            PppCorrectionsError::Epoch {
                epoch_index: 0,
                source: CoverageError::OutsideCoverage(
                    crate::astro::time::DegradeReason::AfterCoverage
                ),
            }
        );
    }

    #[test]
    fn build_accepts_valid_correction_epoch() {
        let sp3 = sp3_fixture();
        let options = PppCorrectionsOptions {
            solid_earth_tide: true,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: None,
        };
        let epochs = vec![PppCorrectionEpoch {
            epoch: civil(2020, 6, 24, 12, 0, 0.0),
            t_rx_j2000_s: 0.0,
            observations: Vec::new(),
        }];

        let got = build(
            &sp3,
            &epochs,
            [3_512_900.0, 780_500.0, 5_248_700.0],
            &options,
        )
        .expect("valid PPP correction epoch must build");

        assert_eq!(got.tide.len(), 1);
    }

    #[test]
    fn build_rejects_degenerate_receiver_state_before_tide() {
        let sp3 = sp3_fixture();
        let epoch = civil(2020, 6, 24, 12, 0, 0.0);
        let options = PppCorrectionsOptions {
            solid_earth_tide: true,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: None,
        };
        let epochs = vec![PppCorrectionEpoch {
            epoch,
            t_rx_j2000_s: 0.0,
            observations: Vec::new(),
        }];

        let err = build(&sp3, &epochs, [0.0, 0.0, 0.0], &options)
            .expect_err("degenerate tide geometry must error");

        assert_eq!(
            err,
            PppCorrectionsError::InvalidInput {
                field: "receiver radius_m",
                reason: "not positive",
            }
        );
    }
}
