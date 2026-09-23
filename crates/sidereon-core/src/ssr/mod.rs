//! Engineering-unit State Space Representation corrections.
//!
//! The RTCM module stores raw transmitted integers. This module stores scaled
//! correction values keyed by satellite, with enough provider and issue metadata
//! to apply orbit and clock corrections on top of broadcast ephemerides.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::antex::{Antex, AntexDateTime};
use crate::astro::bodies::sun_moon_ecef;
use crate::astro::frames::transforms::Ut1Gate;
use crate::astro::math::vec3::add3;
use crate::astro::time::civil::civil_from_j2000_seconds;
use crate::astro::time::eop::Ut1DepartureRecord;
use crate::astro::time::model::{GnssWeekTow, TimeScale};
use crate::astro::time::scales::TimeScales;
use crate::astro::time::{DegradeReason, Validated, ValidityMode};
use crate::constants::{C_M_S, GPS_EPOCH_TO_J2000_S, SECONDS_PER_HOUR, SECONDS_PER_WEEK};
use crate::ephemeris::{BroadcastEphemeris, BroadcastIssue, NavMessage};
use crate::error::{Error, Result};
use crate::has::{
    has_mt1_reference_j2000_s, has_validity_interval_s, HasMt1Message, HasPhaseBiasConversion,
};
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::observables::{ObservableEphemerisSource, ObservableState, ObservablesError};
use crate::ppp_corrections::satellite_body_pco_to_ecef;
use crate::rinex_nav::is_beidou_geo;
use crate::rtcm::{Message, SsrKind, SsrMessage};
use crate::spp::{EphemerisSource, PositionClock, PositionClockGroupDelay};
use crate::staleness::StalenessPolicy;

mod signal;
pub use signal::{has_signal, rtcm_ssr_signal, GnssSignal, SignalCode, SsrRawSignal, SsrSignalKey};

const DEFAULT_SSR_STALENESS_S: f64 = 90.0;
/// Largest age of an RTCM SSR orbit or clock correction, measured from its
/// transmitted epoch, that is applied: RTKLIB `MAXAGESSR`.
const RTCM_SSR_MAX_AGE_S: f64 = 90.0;
/// Age, measured from its transmitted epoch, below which an RTCM SSR
/// high-rate clock correction is applied: RTKLIB `MAXAGESSR_HRCLK`.
const RTCM_SSR_HIGH_RATE_CLOCK_MAX_AGE_S: f64 = 10.0;
/// UTC(SU) + 3 h is GLONASS time.
const GLONASS_MINUS_UTC_S: f64 = 3.0 * SECONDS_PER_HOUR;
const SECONDS_PER_DAY: f64 = 86_400.0;
/// Julian date of the GPS epoch, 1980-01-06 00:00:00.
const GPS_EPOCH_JD: f64 = 2_444_244.5;
use crate::rinex_nav::{ephpos_stepped_tk, EPHPOS_STEP_S};
/// RTCM 10403.x SSR radial orbit and clock C0 resolution, meters.
const RTCM_SSR_RADIAL_CLOCK_SCALE_M: f64 = 1.0e-4;
/// RTCM 10403.x SSR along-track and cross-track orbit resolution, meters.
const RTCM_SSR_ALONG_CROSS_SCALE_M: f64 = 4.0e-4;
/// RTCM 10403.x SSR radial-rate and clock C1 resolution, meters per second.
const RTCM_SSR_RADIAL_CLOCK_RATE_SCALE_M_S: f64 = 1.0e-6;
/// RTCM 10403.x SSR along/cross-rate resolution, meters per second.
const RTCM_SSR_ALONG_CROSS_RATE_SCALE_M_S: f64 = 4.0e-6;
/// RTCM 10403.x SSR clock C2 resolution, meters per second squared.
const RTCM_SSR_CLOCK_ACCEL_SCALE_M_S2: f64 = 2.0e-8;
/// RTCM 10403.x SSR code-bias resolution, meters.
const RTCM_SSR_CODE_BIAS_SCALE_M: f64 = 1.0e-2;
/// RTCM 10403.x SSR phase-bias resolution, meters.
const RTCM_SSR_PHASE_BIAS_SCALE_M: f64 = 1.0e-4;

/// Which stream produced a correction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SsrSource {
    /// RTCM SSR messages.
    RtcmSsr,
    /// Galileo HAS messages.
    GalileoHas,
}

/// Orbital basis used by stored RAC components.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OrbitBasis {
    /// Velocity-aligned basis used by RTCM SSR and HAS.
    VelocityAligned,
}

/// Reference point of the corrected orbit.
///
/// `rtcm_ssr_default` preserves the convention used by the current correction
/// path: no satellite PCO is applied because the corrected state is treated as
/// an antenna-phase-center state unless the caller explicitly selects CoM.
///
/// `igs_ssr_default` selects CoM for IGS SSR CoM streams. IGS SSR v1.00 defines
/// CoM/APC as the satellite reference point choices and IGS RTS product
/// documentation identifies CoM streams separately from antenna-reference
/// streams:
/// <https://files.igs.org/pub/data/format/igs_ssr_v1.pdf>
/// <https://igs.org/rts/products/>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SsrReferencePoint {
    /// Satellite antenna phase center.
    AntennaPhaseCenter,
    /// Satellite center of mass.
    CenterOfMass,
}

impl SsrReferencePoint {
    /// Default reference point for decoded RTCM SSR streams.
    pub const fn rtcm_ssr_default() -> Self {
        Self::AntennaPhaseCenter
    }

    /// Default reference point for IGS SSR CoM streams.
    pub const fn igs_ssr_default() -> Self {
        Self::CenterOfMass
    }

    /// Stable compact tag for storing this reference-point choice.
    pub const fn tag(self) -> u8 {
        match self {
            Self::AntennaPhaseCenter => 0,
            Self::CenterOfMass => 1,
        }
    }

    /// Decode a compact tag produced by [`SsrReferencePoint::tag`].
    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::AntennaPhaseCenter),
            1 => Some(Self::CenterOfMass),
            _ => None,
        }
    }
}

/// Backward-compatible name for an SSR orbit reference point.
pub type OrbitReferencePoint = SsrReferencePoint;

/// Satellite attitude model used for SSR CoM-to-APC conversion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SsrSatelliteAttitude {
    /// No satellite attitude model is available; CoM-to-APC conversion is declined.
    #[default]
    Unavailable,
    /// Nominal satellite-Sun-fixed axes used by the PPP antenna PCO path.
    ///
    /// This does not cover yaw maneuvers, eclipse turns, or provider-specific
    /// attitude laws. Use it only when that nominal model is the intended SSR
    /// convention for the correction stream.
    NominalSunFixed,
}

/// Provider and solution identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SsrSolution {
    /// Correction source.
    pub source: SsrSource,
    /// Provider id.
    pub provider_id: u16,
    /// Solution id.
    pub solution_id: u8,
}

/// Broadcast navigation message an SSR orbit or clock correction refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SsrNavigationMessage {
    /// An RTCM SSR correction. It refers to the broadcast record of the
    /// satellite's own navigation message that its issue names, as RTKLIB
    /// `satpos_ssr` selects it: GPS and QZSS LNAV and Galileo I/NAV by IODE,
    /// BeiDou D1 (D2 for a geostationary satellite) by the IOD
    /// `mod(toe/720, 240)` (IGS SSR v1.00, IDF012), GLONASS by `tb`.
    Rtcm,
    /// A Galileo HAS correction, with the navigation-message index NM its mask
    /// states, as transmitted (HAS SIS ICD 5.2.1.6, Table 21). Index 0 is GPS
    /// LNAV or Galileo I/NAV. Indices 1..=7 are reserved: the correction is
    /// kept, and [`SsrCorrectedEphemeris`] does not apply it, reporting
    /// [`SsrStateUnavailable::ReservedNavigationMessage`].
    Has(u8),
}

impl SsrNavigationMessage {
    /// The reserved HAS navigation-message index this correction refers to, if
    /// it refers to one.
    pub const fn reserved_has_index(self) -> Option<u8> {
        match self {
            Self::Has(index) if index != 0 => Some(index),
            _ => None,
        }
    }
}

/// Orbit correction for one satellite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsrOrbitCorrection {
    /// Provider and solution identity.
    pub solution: SsrSolution,
    /// Broadcast navigation message the correction refers to.
    pub nav_message: SsrNavigationMessage,
    /// Referenced broadcast issue.
    pub iode: u32,
    /// IOD SSR.
    pub iod_ssr: u8,
    /// Orbit basis.
    pub basis: OrbitBasis,
    /// True when the RTCM reference datum bit marks a regional CRS.
    pub crs_regional: bool,
    /// Orbit reference point policy.
    pub reference_point: SsrReferencePoint,
    /// Radial delta to add, meters.
    pub radial_m: f64,
    /// Along-track delta to add, meters.
    pub along_m: f64,
    /// Cross-track delta to add, meters.
    pub cross_m: f64,
    /// Radial delta rate to add, meters per second.
    pub radial_rate_m_s: f64,
    /// Along-track delta rate to add, meters per second.
    pub along_rate_m_s: f64,
    /// Cross-track delta rate to add, meters per second.
    pub cross_rate_m_s: f64,
    /// Reference epoch of the rate terms, seconds since J2000. For RTCM SSR it
    /// is the transmitted epoch plus half the update interval, or the
    /// transmitted epoch for update interval index 0 (IGS SSR v1.00); for
    /// Galileo HAS it is the TOH epoch.
    pub ref_epoch_j2000_s: f64,
    /// Transmitted epoch, seconds since J2000: the RTCM SSR epoch time, or the
    /// Galileo HAS TOH epoch. The age of an RTCM SSR correction is measured
    /// from it.
    pub transmitted_epoch_j2000_s: f64,
    /// Update interval, seconds: the RTCM SSR update interval, or the Galileo
    /// HAS validity interval.
    pub update_interval_s: f64,
}

/// High-rate clock correction for one satellite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsrHighRateClock {
    /// Provider and solution identity.
    pub solution: SsrSolution,
    /// IOD SSR.
    pub iod_ssr: u8,
    /// Additive C0 term, meters.
    pub c0_m: f64,
    /// Reference epoch, seconds since J2000. A high-rate clock has no rate
    /// terms, so this is its transmitted epoch.
    pub ref_epoch_j2000_s: f64,
    /// Transmitted epoch, seconds since J2000. Its age is measured from it.
    pub transmitted_epoch_j2000_s: f64,
    /// Update interval, seconds.
    pub update_interval_s: f64,
}

/// Clock correction for one satellite.
///
/// A corrected satellite clock is the broadcast clock polynomial, less the
/// relativistic term `2 r·v / c²`, plus the correction polynomial over `c`
/// (RTKLIB `satpos_ssr`; HAS SIS ICD 7.3, Eq. 23 and 24). The correction adds to
/// the clock for both sources.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsrClockCorrection {
    /// Provider and solution identity.
    pub solution: SsrSolution,
    /// Broadcast navigation message the correction refers to.
    pub nav_message: SsrNavigationMessage,
    /// IOD SSR.
    pub iod_ssr: u8,
    /// C0 term, meters, added to the satellite clock as `C0 / c`.
    pub c0_m: f64,
    /// C1 term, meters per second.
    pub c1_m_s: f64,
    /// C2 term, meters per second squared.
    pub c2_m_s2: f64,
    /// Reference epoch of the C1 and C2 terms, seconds since J2000. For RTCM
    /// SSR it is the transmitted epoch plus half the update interval, or the
    /// transmitted epoch for update interval index 0 (IGS SSR v1.00); for
    /// Galileo HAS it is the TOH epoch.
    pub ref_epoch_j2000_s: f64,
    /// Transmitted epoch, seconds since J2000: the RTCM SSR epoch time, or the
    /// Galileo HAS TOH epoch. The age of an RTCM SSR correction is measured
    /// from it.
    pub transmitted_epoch_j2000_s: f64,
    /// Update interval, seconds: the RTCM SSR update interval, or the Galileo
    /// HAS validity interval.
    pub update_interval_s: f64,
    /// Matching high-rate correction, when present.
    pub high_rate: Option<SsrHighRateClock>,
}

/// Typed lifetime metadata for SSR corrections.
///
/// Distinguishes Galileo HAS explicit wire validity duration from
/// RTCM SSR transmission update interval cadence.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum SsrLifetime {
    /// Galileo HAS explicit wire validity interval in seconds (Galileo HAS SIS ICD Section 5.2.2.1).
    GalileoHasValidityInterval(f64),
    /// RTCM SSR update interval in seconds. It is the transmission cadence, not
    /// a validity duration: it moves the reference time of orbit and clock rate
    /// terms to half an interval after the transmitted epoch (index 0: to the
    /// transmitted epoch itself), and bounds no
    /// use. RTCM SSR biases carry no age limit beyond the store staleness cap,
    /// as RTKLIB applies none; orbit and clock corrections are limited by age
    /// from the transmitted epoch (RTKLIB `MAXAGESSR`, 90 s).
    RtcmUpdateInterval(f64),
}

impl SsrLifetime {
    /// Duration or update cadence value in seconds.
    pub const fn seconds(self) -> f64 {
        match self {
            Self::GalileoHasValidityInterval(s) | Self::RtcmUpdateInterval(s) => s,
        }
    }

    /// Whether this lifetime represents a Galileo HAS explicit validity interval.
    pub const fn is_has(self) -> bool {
        matches!(self, Self::GalileoHasValidityInterval(_))
    }

    /// Whether this lifetime represents an RTCM SSR update interval cadence.
    pub const fn is_rtcm(self) -> bool {
        matches!(self, Self::RtcmUpdateInterval(_))
    }
}

/// Native phase discontinuity indicator and its source format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PhaseDiscontinuityIndicator {
    /// Galileo HAS 2-bit Phase Discontinuity Indicator (PDI, 0..=3 per HAS SIS ICD
    /// Table 40, laid out by Tables 38 and 39; §5.2.6.1 defines how it increments).
    GalileoHasPdi(u8),
    /// RTCM SSR phase discontinuity counter, a 4-bit wire field (0..=15), carried
    /// in a `u8` because that is the decoded record type.
    RtcmDiscontinuityCounter(u8),
}

impl PhaseDiscontinuityIndicator {
    /// Raw integer indicator value.
    pub const fn raw_value(self) -> u8 {
        match self {
            Self::GalileoHasPdi(v) | Self::RtcmDiscontinuityCounter(v) => v,
        }
    }
}

/// Opaque, read-only phase continuity token distinguishing source, solution,
/// and continuity reference epoch/context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhaseContinuityToken {
    sat: GnssSatelliteId,
    signal: SsrSignalKey,
    source: SsrSource,
    provider_id: u16,
    solution_id: u8,
    continuity_ref_epoch_bits: u64,
    raw_indicator: u8,
    generation: u32,
}

impl PhaseContinuityToken {
    /// Corrected satellite.
    pub const fn satellite(&self) -> GnssSatelliteId {
        self.sat
    }

    /// Key of the corrected signal: its physical signal, or the raw
    /// source-qualified index when the source's table assigns none.
    pub const fn signal(&self) -> SsrSignalKey {
        self.signal
    }

    /// Correction stream source.
    pub const fn source(&self) -> SsrSource {
        self.source
    }

    /// Solution provider identifier.
    pub const fn provider_id(&self) -> u16 {
        self.provider_id
    }

    /// Solution identifier.
    pub const fn solution_id(&self) -> u8 {
        self.solution_id
    }

    /// Reference epoch at which this continuous phase arc began, seconds since J2000.
    pub fn continuity_ref_epoch_j2000_s(&self) -> f64 {
        f64::from_bits(self.continuity_ref_epoch_bits)
    }

    /// Raw native discontinuity indicator value (HAS 2-bit PDI or RTCM counter).
    pub const fn raw_indicator(&self) -> u8 {
        self.raw_indicator
    }

    /// Arc transition generation index distinguishing same-epoch revisions and source switches.
    pub const fn generation(&self) -> u32 {
        self.generation
    }
}

/// Status of an SSR bias query at a requested epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SsrBiasStatus {
    /// Correction is available and valid at the query epoch.
    Available,
    /// Satellite or signal has no record in the store.
    Missing,
    /// Signal correction was transmitted as unavailable (or superseded by newer unavailable).
    Unavailable,
    /// Query epoch is before the correction's reference epoch.
    NotYetValid,
    /// Correction validity interval or update interval staleness cap has expired.
    Expired,
    /// Satellite is excluded by an active Galileo HAS do-not-use indication.
    Excluded,
    /// Query epoch is NaN or non-finite.
    InvalidEpoch,
    /// Phase discontinuity detected or token mismatch; caller must reset ambiguity arc.
    PhaseDiscontinuityNeedsReset,
    /// The record's signal index is one its source's table leaves reserved or
    /// unassigned, so it names no physical signal and applies to no observation.
    /// The record is kept, and the query reports its transmitted value: `bias_m`
    /// where the record has metres (RTCM SSR code and phase, HAS code), and for
    /// phase `bias_cycles` where it has cycles (HAS). The status, not the value,
    /// is what keeps it from being applied.
    UnknownSignal,
}

/// Details of phase continuity evaluation for a phase-bias query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SsrDiscontinuityDetails {
    /// Initial observation established a new continuity token without prior continuity.
    InitialTokenEstablished,
    /// Continuous phase arc confirmed with matching caller acknowledgement token.
    Continuous,
    /// Native Galileo HAS Phase Discontinuity Indicator (PDI) changed.
    HasPdiChanged {
        /// Previous PDI value.
        previous: u8,
        /// Current incoming or updated PDI value.
        current: u8,
    },
    /// Native RTCM phase discontinuity counter changed.
    RtcmDiscontinuityCounterChanged {
        /// Previous discontinuity counter value.
        previous: u8,
        /// Current incoming or updated discontinuity counter value.
        current: u8,
    },
    /// Correction provider or solution changed; receiver policy does not infer cross-service continuity.
    SolutionChanged {
        /// Previous solution metadata.
        previous: SsrSolution,
        /// Current solution metadata.
        current: SsrSolution,
    },
    /// Caller supplied a stale token from an earlier continuity arc.
    StaleToken,
    /// Caller supplied a future token not yet valid at current state.
    FutureToken,
    /// Caller supplied a token for another satellite, signal, or provider.
    MismatchedToken,
}

/// Detailed diagnostic reason for an SSR bias query status.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SsrBiasResolutionDetails {
    /// Correction is valid, fresh, and continuous.
    Available,
    /// No record exists for this satellite and signal.
    NoRecord,
    /// Transmitted wire unavailable sentinel.
    TransmittedUnavailable,
    /// Transmitted phase cycles present but metre conversion unavailable (missing carrier frequency).
    ConversionUnavailable,
    /// Query epoch is before reference epoch (Galileo HAS forward validity).
    EpochBeforeReference {
        /// Reference epoch in seconds since J2000.
        ref_epoch_j2000_s: f64,
        /// Query epoch in seconds since J2000.
        query_epoch_j2000_s: f64,
    },
    /// Correction has expired past validity interval or staleness cap.
    EpochExpired {
        /// Expiry epoch in seconds since J2000.
        expiry_epoch_j2000_s: f64,
        /// Query epoch in seconds since J2000.
        query_epoch_j2000_s: f64,
    },
    /// Active Galileo HAS satellite do-not-use exclusion.
    ExcludedByDoNotUse {
        /// Reference epoch in seconds since J2000.
        ref_epoch_j2000_s: f64,
        /// Validity interval duration in seconds.
        validity_interval_s: f64,
    },
    /// Query epoch is NaN or infinite.
    InvalidEpoch,
    /// Phase discontinuity encountered.
    PhaseDiscontinuity(SsrDiscontinuityDetails),
    /// The record's raw signal index names no physical signal in its source's table.
    UnknownSignal(SsrRawSignal),
}

/// Query result for a code-bias correction at a specific reception epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct SsrCodeBiasQueryResult {
    /// Corrected satellite identifier.
    pub sat: GnssSatelliteId,
    /// Key of the queried signal. A raw signal whose source table assigns it a
    /// physical signal is queried, and reported, as that physical signal.
    pub signal: SsrSignalKey,
    /// Raw signal index, as its source transmitted it, of the record found, if any.
    /// A physical signal's record may come from either source.
    pub source_signal: Option<SsrRawSignal>,
    /// Validity and availability status at the query epoch.
    pub status: SsrBiasStatus,
    /// Code bias correction value in meters, if available.
    pub bias_m: Option<f64>,
    /// Solution stream identity and provider metadata, if available.
    pub solution: Option<SsrSolution>,
    /// Issue of Data SSR matching the correction stream, if available.
    pub iod_ssr: Option<u8>,
    /// Reference epoch in seconds since J2000, if available.
    pub ref_epoch_j2000_s: Option<f64>,
    /// Lifetime or validity interval definition, if available.
    pub lifetime: Option<SsrLifetime>,
    /// Detailed diagnostic resolution reason.
    pub details: SsrBiasResolutionDetails,
}

/// Query result for a phase-bias correction at a specific reception epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct SsrPhaseBiasQueryResult {
    /// Corrected satellite identifier.
    pub sat: GnssSatelliteId,
    /// Key of the queried signal. A raw signal whose source table assigns it a
    /// physical signal is queried, and reported, as that physical signal.
    pub signal: SsrSignalKey,
    /// Raw signal index, as its source transmitted it, of the record found, if any.
    /// A physical signal's record may come from either source.
    pub source_signal: Option<SsrRawSignal>,
    /// Validity and availability status at the query epoch.
    pub status: SsrBiasStatus,
    /// Phase bias correction value in meters, if available.
    pub bias_m: Option<f64>,
    /// Native phase bias value in carrier cycles, if available.
    pub bias_cycles: Option<f64>,
    /// Solution stream identity and provider metadata, if available.
    pub solution: Option<SsrSolution>,
    /// Issue of Data SSR matching the correction stream, if available.
    pub iod_ssr: Option<u8>,
    /// Reference epoch in seconds since J2000, if available.
    pub ref_epoch_j2000_s: Option<f64>,
    /// Lifetime or validity interval definition, if available.
    pub lifetime: Option<SsrLifetime>,
    /// Opaque token identifying the continuous phase arc, if available.
    pub continuity_token: Option<PhaseContinuityToken>,
    /// Raw native phase discontinuity indicator, if available.
    pub discontinuity_indicator: Option<PhaseDiscontinuityIndicator>,
    /// Detailed continuity evaluation outcome, if continuity was evaluated.
    /// Queried with no acknowledgement token, it is the break recorded since the
    /// previous arc, or `InitialTokenEstablished` when there is none, and the
    /// status is not `PhaseDiscontinuityNeedsReset` either way.
    pub discontinuity_details: Option<SsrDiscontinuityDetails>,
    /// Detailed diagnostic resolution reason.
    pub details: SsrBiasResolutionDetails,
}

/// Galileo HAS status recorded in watermark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HasWatermarkStatus {
    /// Usable bias received.
    Usable,
    /// Unavailable bias received.
    Unavailable,
}

/// Minimal private watermark tracking accepted Galileo HAS state per signal across RTCM updates.
#[derive(Clone, Copy, Debug, PartialEq)]
struct HasStatusWatermark {
    /// Reference epoch in seconds since J2000 of the last accepted HAS record.
    ref_epoch_j2000_s: f64,
    /// Status (usable vs unavailable) of the last accepted HAS record.
    status: HasWatermarkStatus,
    /// Phase Discontinuity Indicator of the last accepted HAS record, if phase bias.
    pdi: Option<u8>,
}

/// Action outcome and reason for an ingested HAS record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum IngestionActionReason {
    /// Initial record accepted into store.
    AcceptedInitialRecord,
    /// Newer record accepted and replaced older record.
    AcceptedNewerRecord,
    /// Usable record accepted over equal-epoch unavailable.
    AcceptedUsableOverUnavailable,
    /// Updated record accepted at equal epoch (e.g. revision). Between two
    /// usable records at one reference epoch the later arrival wins, as RTKLIB
    /// overwrites a correction with each message it decodes.
    AcceptedUpdatedRecord,
    /// First unavailable record stored with full metadata on empty entry.
    AcceptedFirstUnavailableWithMetadata,
    /// Newer unavailable record cleared older HAS record.
    AcceptedUnavailableClearingOlderHas,
    /// Active RTCM record retained when a HAS unavailable record arrived. The watermark
    /// advances to the incoming reference epoch and PDI; the active RTCM correction,
    /// its continuity token and its arc generation are untouched.
    RetainedActiveRtcmOnHasUnavailable,
    /// Equal-epoch unavailable record carrying the same solution, IOD, lifetime and PDI
    /// as the active unavailable record, so nothing about the active state changed.
    RetainedEquivalentUnavailable,
    /// Record refused because it is older than the established HAS watermark.
    RefusedOlderThanWatermark,
    /// Unavailable record refused because equal-epoch usable takes precedence.
    RefusedEqualEpochUnavailableUnderUsable,
}

impl IngestionActionReason {
    /// Whether the incoming record replaced the active correction.
    ///
    /// An accepted record always advances the per-signal Galileo HAS watermark as well.
    pub const fn is_accepted(&self) -> bool {
        matches!(
            self,
            Self::AcceptedInitialRecord
                | Self::AcceptedNewerRecord
                | Self::AcceptedUsableOverUnavailable
                | Self::AcceptedUpdatedRecord
                | Self::AcceptedFirstUnavailableWithMetadata
                | Self::AcceptedUnavailableClearingOlderHas
        )
    }

    /// Whether the active correction, its continuity token and its arc generation were
    /// left exactly as they were.
    ///
    /// A retained record still advances the per-signal Galileo HAS watermark, so the
    /// reference epoch and PDI it carries are not lost.
    pub const fn is_retained(&self) -> bool {
        matches!(
            self,
            Self::RetainedActiveRtcmOnHasUnavailable | Self::RetainedEquivalentUnavailable
        )
    }

    /// Whether the incoming record was refused, leaving the active correction and the
    /// per-signal Galileo HAS watermark both unchanged.
    pub const fn is_refused(&self) -> bool {
        matches!(
            self,
            Self::RefusedOlderThanWatermark | Self::RefusedEqualEpochUnavailableUnderUsable
        )
    }
}

/// Provenance and status of the active entry resulting from ingestion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActiveProvenanceStatus {
    /// Active entry is Galileo HAS usable.
    ActiveHasUsable,
    /// Active entry is Galileo HAS unavailable (sentinel with metadata).
    ActiveHasUnavailable,
    /// Active entry is RTCM SSR usable.
    ActiveRtcmUsable,
    /// No active entry present.
    NoActiveEntry,
}

/// Diagnostic report entry for one incoming HAS code bias record.
#[derive(Clone, Debug, PartialEq)]
pub struct HasCodeBiasIngestionRecord {
    /// Corrected satellite identifier.
    pub sat: GnssSatelliteId,
    /// HAS signal index as transmitted.
    pub signal: SsrRawSignal,
    /// Key the record is stored under: the physical signal HAS SIS ICD Table 20
    /// assigns the index, or the raw index for a reserved one.
    pub key: SsrSignalKey,
    /// Correction stream source format.
    pub source: SsrSource,
    /// Solution stream identity and provider metadata.
    pub solution: SsrSolution,
    /// Reference epoch in seconds since J2000.
    pub ref_epoch_j2000_s: f64,
    /// Transmitted native code bias in meters, or None if unavailable sentinel.
    pub native_bias_m: Option<f64>,
    /// Ingestion decision outcome and reason.
    pub reason: IngestionActionReason,
    /// Resulting active provenance status after evaluating this record.
    pub resulting_status: ActiveProvenanceStatus,
}

/// Diagnostic report entry for one incoming HAS phase bias record.
#[derive(Clone, Debug, PartialEq)]
pub struct HasPhaseBiasIngestionRecord {
    /// Corrected satellite identifier.
    pub sat: GnssSatelliteId,
    /// HAS signal index as transmitted.
    pub signal: SsrRawSignal,
    /// Key the record is stored under: the physical signal HAS SIS ICD Table 20
    /// assigns the index, or the raw index for a reserved one.
    pub key: SsrSignalKey,
    /// Correction stream source format.
    pub source: SsrSource,
    /// Solution stream identity and provider metadata.
    pub solution: SsrSolution,
    /// Reference epoch in seconds since J2000.
    pub ref_epoch_j2000_s: f64,
    /// Transmitted native phase bias in cycles, or None if unavailable sentinel.
    pub native_cycles: Option<f64>,
    /// Transmitted Phase Discontinuity Indicator (PDI).
    pub pdi: u8,
    /// Continuity token assigned or matched for this phase arc, if available.
    pub continuity_token: Option<PhaseContinuityToken>,
    /// Ingestion decision outcome and reason.
    pub reason: IngestionActionReason,
    /// Resulting active provenance status after evaluating this record.
    pub resulting_status: ActiveProvenanceStatus,
}

/// Diagnostic report of all code and phase records evaluated during HAS MT1 ingestion.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HasIngestionReport {
    /// Evaluation records for all code biases in the ingested MT1 block.
    pub code_records: Vec<HasCodeBiasIngestionRecord>,
    /// Evaluation records for all phase biases in the ingested MT1 block.
    pub phase_records: Vec<HasPhaseBiasIngestionRecord>,
}

impl HasIngestionReport {
    /// Whether any incoming record was refused due to watermark protection.
    pub fn has_refusals(&self) -> bool {
        self.code_records.iter().any(|r| r.reason.is_refused())
            || self.phase_records.iter().any(|r| r.reason.is_refused())
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CodeBiasSignalRecord {
    /// Signal index as the record's source transmitted it.
    signal: SsrRawSignal,
    value_m: Option<f64>,
    solution: SsrSolution,
    iod_ssr: u8,
    ref_epoch_j2000_s: f64,
    /// Transmitted epoch: the RTCM SSR epoch time, or the Galileo HAS TOH
    /// epoch. RTCM SSR bias age is measured from it.
    transmitted_epoch_j2000_s: f64,
    lifetime: SsrLifetime,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct CodeBiasSignalEntry {
    active: Option<CodeBiasSignalRecord>,
    has_watermark: Option<HasStatusWatermark>,
    has_superseded_epoch_j2000_s: Option<f64>,
    last_has_epoch_j2000_s: Option<f64>,
}

/// SSR code-bias corrections keyed by signal: the physical signal where the
/// source's table assigns one, the raw source-qualified index otherwise.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SsrCodeBias {
    signals: BTreeMap<SsrSignalKey, CodeBiasSignalEntry>,
}

#[derive(Clone, Debug, PartialEq)]
struct PhaseBiasSignalRecord {
    /// Signal index as the record's source transmitted it.
    signal: SsrRawSignal,
    value_m: Option<f64>,
    value_cycles: Option<f64>,
    solution: SsrSolution,
    iod_ssr: u8,
    ref_epoch_j2000_s: f64,
    /// Transmitted epoch: the RTCM SSR epoch time, or the Galileo HAS TOH
    /// epoch. RTCM SSR bias age is measured from it.
    transmitted_epoch_j2000_s: f64,
    lifetime: SsrLifetime,
    discontinuity: PhaseDiscontinuityIndicator,
    token: PhaseContinuityToken,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct PhaseBiasSignalEntry {
    active: Option<PhaseBiasSignalRecord>,
    has_watermark: Option<HasStatusWatermark>,
    arc_generation: u32,
    prior_break: Option<SsrDiscontinuityDetails>,
    has_superseded_epoch_j2000_s: Option<f64>,
    last_has_epoch_j2000_s: Option<f64>,
    last_has_continuity: Option<(u8, u64)>,
    prior_rtcm_continuity: Option<(u8, u64)>,
}

/// SSR phase-bias corrections keyed by signal: the physical signal where the
/// source's table assigns one, the raw source-qualified index otherwise.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SsrPhaseBias {
    signals: BTreeMap<SsrSignalKey, PhaseBiasSignalEntry>,
}

/// Validity-bounded exclusion marker for an active Galileo HAS satellite do-not-use indication.
#[derive(Clone, Copy, Debug, PartialEq)]
struct HasExclusionMarker {
    /// J2000 reference epoch in seconds derived from TOH per Galileo HAS SIS ICD Section 5.2.2.1 / Section 7.7.
    ref_epoch_j2000_s: f64,
    /// Validity interval duration in seconds per Galileo HAS SIS ICD Section 5.2.2.1 and Table 23.
    validity_interval_s: f64,
}

impl HasExclusionMarker {
    /// Check whether the exclusion indication is active at the query epoch.
    ///
    /// Per Galileo HAS SIS ICD Section 5.2.2.1, validity starts at the time defined by TOH
    /// and lasts for the validity interval. In agreement with SSR `<=` validity convention,
    /// the indication is active when `t >= ref_epoch && t <= ref_epoch + validity_interval`.
    fn is_active(&self, t_j2000_s: f64) -> bool {
        t_j2000_s >= self.ref_epoch_j2000_s
            && t_j2000_s <= self.ref_epoch_j2000_s + self.validity_interval_s
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct SatCorrections {
    orbit: Option<SsrOrbitCorrection>,
    has_orbit_superseded_epoch_j2000_s: Option<f64>,
    clock: Option<SsrClockCorrection>,
    pending_high_rate: Option<SsrHighRateClock>,
    ura_index: Option<u8>,
    code_bias: SsrCodeBias,
    phase_bias: SsrPhaseBias,
    exclusion: Option<HasExclusionMarker>,
    has_clock_superseded_epoch_j2000_s: Option<f64>,
    /// Latest reference epoch of any HAS orbit record accepted for this
    /// satellite, usable or unavailable. It persists whatever source holds the
    /// orbit now, so a HAS record older than one already accepted is refused
    /// even after an RTCM correction has replaced the HAS one.
    has_orbit_watermark_j2000_s: Option<f64>,
    /// Latest reference epoch of any HAS clock record accepted for this
    /// satellite: usable, unavailable or do-not-use. It persists whatever
    /// source holds the clock now, and usable and unavailable records older
    /// than it are refused.
    has_clock_watermark_j2000_s: Option<f64>,
    /// Latest reference epoch of a usable HAS clock record accepted for this
    /// satellite. A do-not-use record older than it is refused. A do-not-use
    /// record older only than an unavailable one still applies for its own
    /// validity interval: "unavailable" states that no correction is sent, not
    /// that the satellite may be used.
    has_clock_usable_watermark_j2000_s: Option<f64>,
}

/// Whether a HAS record at `epoch` is older than one already accepted.
fn older_than_has_watermark(watermark: Option<f64>, epoch_j2000_s: f64) -> bool {
    watermark.is_some_and(|w| epoch_j2000_s < w)
}

/// Advance a HAS watermark to an accepted record's epoch.
fn advance_has_watermark(watermark: &mut Option<f64>, epoch_j2000_s: f64) {
    *watermark = Some(watermark.map_or(epoch_j2000_s, |w| w.max(epoch_j2000_s)));
}

/// Active SSR corrections keyed by satellite.
///
/// The store keeps Galileo HAS watermarks per satellite and signal, so it
/// refuses a HAS record older than one it has already accepted. Replaying
/// earlier data into a store that has seen later data therefore changes
/// nothing; replay into a new store.
#[derive(Clone, Debug, PartialEq)]
pub struct SsrCorrectionStore {
    corrections: BTreeMap<GnssSatelliteId, SatCorrections>,
    reference_point: SsrReferencePoint,
    staleness: StalenessPolicy,
}

impl Default for SsrCorrectionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SsrCorrectionStore {
    /// Build an empty correction store.
    pub fn new() -> Self {
        Self {
            corrections: BTreeMap::new(),
            reference_point: SsrReferencePoint::rtcm_ssr_default(),
            staleness: StalenessPolicy::seconds(DEFAULT_SSR_STALENESS_S),
        }
    }

    /// Set the orbit reference point policy for later ingests.
    pub fn with_reference_point(mut self, reference_point: SsrReferencePoint) -> Self {
        self.reference_point = reference_point;
        self
    }

    /// Orbit reference point policy used for later ingests.
    pub fn reference_point(&self) -> SsrReferencePoint {
        self.reference_point
    }

    /// Set the store staleness policy.
    pub fn with_staleness(mut self, policy: StalenessPolicy) -> Self {
        self.staleness = policy;
        self
    }

    /// The store-level staleness policy.
    pub fn staleness(&self) -> StalenessPolicy {
        self.staleness
    }

    /// Ingest one RTCM message, ignoring non-SSR messages.
    ///
    /// `week` is the receiver time, in any GNSS, UTC or GLONASS week scale. Each
    /// SSR epoch is placed in the week (GLONASS: the day) nearest it.
    pub fn ingest(&mut self, message: &Message, week: GnssWeekTow) -> Result<()> {
        if let Message::Ssr(ssr) = message {
            self.ingest_ssr(ssr, week)?;
        }
        Ok(())
    }

    /// Ingest one decoded RTCM SSR message.
    ///
    /// `week` is the receiver time, in any GNSS, UTC or GLONASS week scale. The
    /// message's epoch is placed in the week (GLONASS: the day) nearest it, as
    /// RTKLIB `adjweek` and `adjday_glot` place it.
    ///
    /// The message is applied to a staged copy of the store, and the copy
    /// replaces the store only when every record has been applied. A refused
    /// message, whichever record or check refuses it, leaves the store as it
    /// was. Galileo HAS ingestion commits the same way.
    pub fn ingest_ssr(&mut self, message: &SsrMessage, week: GnssWeekTow) -> Result<()> {
        let update_interval_s = update_interval_s(message.header.update_interval)?;
        let transmitted_epoch_j2000_s = ssr_epoch_j2000_s(
            message.system,
            message.message_number,
            week,
            message.header.epoch_time_s,
        )?;
        // The orbit and clock rate terms are referenced to the transmitted epoch plus
        // half the update interval, except for update interval index 0, which uses
        // the transmitted epoch itself (IGS SSR v1.00, section 5, and IDF004).
        // RTKLIB `satpos_ssr` shifts index 0 (1 s) by 0.5 s as well, since it
        // shifts whenever `udi >= 1.0`; the definition is followed here. Biases and
        // the high-rate clock have no rate terms and keep the transmitted epoch.
        let ref_epoch_j2000_s = if message.header.update_interval == 0 {
            transmitted_epoch_j2000_s
        } else {
            transmitted_epoch_j2000_s + update_interval_s / 2.0
        };
        let solution = SsrSolution {
            source: SsrSource::RtcmSsr,
            provider_id: message.header.provider_id,
            solution_id: message.header.solution_id,
        };

        let mut staged = self.corrections.clone();

        match message.kind {
            SsrKind::Orbit => {
                for record in &message.orbit {
                    let sat = ssr_satellite(message, record.satellite_id)?;
                    let orbit = orbit_from_rtcm(
                        self.reference_point,
                        message,
                        solution,
                        record,
                        ref_epoch_j2000_s,
                        transmitted_epoch_j2000_s,
                        update_interval_s,
                    );
                    let entry = staged.entry(sat).or_default();
                    entry.orbit = Some(orbit);
                }
            }
            SsrKind::Clock => {
                for record in &message.clock {
                    let sat = ssr_satellite(message, record.satellite_id)?;
                    let entry = staged.entry(sat).or_default();
                    let mut clock = SsrClockCorrection {
                        solution,
                        nav_message: SsrNavigationMessage::Rtcm,
                        iod_ssr: message.header.iod_ssr,
                        c0_m: f64::from(record.c0) * RTCM_SSR_RADIAL_CLOCK_SCALE_M,
                        c1_m_s: f64::from(record.c1) * RTCM_SSR_RADIAL_CLOCK_RATE_SCALE_M_S,
                        c2_m_s2: f64::from(record.c2) * RTCM_SSR_CLOCK_ACCEL_SCALE_M_S2,
                        ref_epoch_j2000_s,
                        transmitted_epoch_j2000_s,
                        update_interval_s,
                        high_rate: None,
                    };
                    if let Some(hr) = entry.pending_high_rate {
                        if high_rate_matches(&clock, &hr) {
                            clock.high_rate = Some(hr);
                        }
                    }
                    entry.clock = Some(clock);
                }
            }
            SsrKind::CombinedOrbitClock => {
                // A combined message carries one orbit and one clock correction
                // per satellite, written as pairs. Pairing by position alone
                // would drop the records past the shorter list, or put one
                // satellite's clock on another satellite's orbit.
                if message.orbit.len() != message.clock.len() {
                    return Err(Error::InvalidInput(format!(
                        "RTCM SSR {} combined orbit/clock message carries {} orbit records \
                         and {} clock records; each satellite needs one of each",
                        message.message_number,
                        message.orbit.len(),
                        message.clock.len()
                    )));
                }
                for (index, (orbit_record, clock_record)) in
                    message.orbit.iter().zip(&message.clock).enumerate()
                {
                    if orbit_record.satellite_id != clock_record.satellite_id {
                        return Err(Error::InvalidInput(format!(
                            "RTCM SSR {} combined orbit/clock record {index} names satellite \
                             id {} for its orbit and {} for its clock",
                            message.message_number,
                            orbit_record.satellite_id,
                            clock_record.satellite_id
                        )));
                    }
                }
                for (orbit_record, clock_record) in message.orbit.iter().zip(&message.clock) {
                    let sat = ssr_satellite(message, orbit_record.satellite_id)?;
                    let orbit = orbit_from_rtcm(
                        self.reference_point,
                        message,
                        solution,
                        orbit_record,
                        ref_epoch_j2000_s,
                        transmitted_epoch_j2000_s,
                        update_interval_s,
                    );
                    let entry = staged.entry(sat).or_default();
                    entry.orbit = Some(orbit);
                    let mut clock = SsrClockCorrection {
                        solution,
                        nav_message: SsrNavigationMessage::Rtcm,
                        iod_ssr: message.header.iod_ssr,
                        c0_m: f64::from(clock_record.c0) * RTCM_SSR_RADIAL_CLOCK_SCALE_M,
                        c1_m_s: f64::from(clock_record.c1) * RTCM_SSR_RADIAL_CLOCK_RATE_SCALE_M_S,
                        c2_m_s2: f64::from(clock_record.c2) * RTCM_SSR_CLOCK_ACCEL_SCALE_M_S2,
                        ref_epoch_j2000_s,
                        transmitted_epoch_j2000_s,
                        update_interval_s,
                        high_rate: None,
                    };
                    if let Some(hr) = entry.pending_high_rate {
                        if high_rate_matches(&clock, &hr) {
                            clock.high_rate = Some(hr);
                        }
                    }
                    entry.clock = Some(clock);
                }
            }
            SsrKind::Ura => {
                for &(satellite_id, ura_index) in &message.ura {
                    let sat = ssr_satellite(message, satellite_id)?;
                    staged.entry(sat).or_default().ura_index = Some(ura_index);
                }
            }
            SsrKind::CodeBias => {
                // A bias has no rate term, so its reference is the transmitted epoch.
                let ref_epoch_j2000_s = transmitted_epoch_j2000_s;
                // A signal listed twice for one satellite keeps its last value, as
                // RTKLIB `decode_ssr3` overwrites it.
                let mut last = BTreeMap::new();
                for record in &message.code_bias {
                    let sat = ssr_satellite(message, record.satellite_id)?;
                    for &(signal, bias) in &record.biases {
                        last.insert((sat, signal), bias);
                    }
                }
                for ((sat, index), bias) in last {
                    let bias_m = f64::from(bias) * RTCM_SSR_CODE_BIAS_SCALE_M;
                    // Keyed by the physical signal the RTCM table assigns the index,
                    // so a HAS record of the same signal shares the entry and one of
                    // another signal with the same index does not.
                    let signal = SsrRawSignal::rtcm_ssr(message.system, index);
                    let sig_entry = staged
                        .entry(sat)
                        .or_default()
                        .code_bias
                        .signals
                        .entry(signal.key())
                        .or_default();
                    sig_entry.active = Some(CodeBiasSignalRecord {
                        signal,
                        value_m: Some(bias_m),
                        solution,
                        iod_ssr: message.header.iod_ssr,
                        ref_epoch_j2000_s,
                        transmitted_epoch_j2000_s,
                        lifetime: SsrLifetime::RtcmUpdateInterval(update_interval_s),
                    });
                }
            }
            SsrKind::HighRateClock => {
                for record in &message.clock {
                    let sat = ssr_satellite(message, record.satellite_id)?;
                    let high_rate = SsrHighRateClock {
                        solution,
                        iod_ssr: message.header.iod_ssr,
                        c0_m: f64::from(record.c0) * RTCM_SSR_RADIAL_CLOCK_SCALE_M,
                        ref_epoch_j2000_s: transmitted_epoch_j2000_s,
                        transmitted_epoch_j2000_s,
                        update_interval_s,
                    };
                    let entry = staged.entry(sat).or_default();
                    entry.pending_high_rate = Some(high_rate);
                    if let Some(clock) = &mut entry.clock {
                        if high_rate_matches(clock, &high_rate) {
                            clock.high_rate = Some(high_rate);
                        }
                    }
                }
            }
            SsrKind::PhaseBias => {
                // A bias has no rate term, so its reference is the transmitted epoch.
                let ref_epoch_j2000_s = transmitted_epoch_j2000_s;
                // A signal listed twice for one satellite keeps its last value, as
                // RTKLIB `decode_ssr7` overwrites it; only that value is applied, so
                // an earlier duplicate does not register a phase break.
                let mut last = BTreeMap::new();
                for record in &message.phase_bias {
                    let sat = ssr_satellite(message, record.satellite_id)?;
                    for bias in &record.biases {
                        last.insert((sat, bias.signal_id), bias);
                    }
                }
                for ((sat, _), bias) in last {
                    let sat_entry = staged.entry(sat).or_default();
                    let bias_m = f64::from(bias.bias) * RTCM_SSR_PHASE_BIAS_SCALE_M;
                    let signal = SsrRawSignal::rtcm_ssr(message.system, bias.signal_id);
                    let key = signal.key();
                    let sig_entry = sat_entry.phase_bias.signals.entry(key).or_default();
                    let continuity_ref_epoch_bits = match &sig_entry.active {
                        None => {
                            sig_entry.prior_break = None;
                            sig_entry.prior_rtcm_continuity =
                                Some((bias.discontinuity_counter, ref_epoch_j2000_s.to_bits()));
                            ref_epoch_j2000_s.to_bits()
                        }
                        Some(prev) => {
                            if prev.solution.source != SsrSource::RtcmSsr
                                || prev.solution.provider_id != solution.provider_id
                                || prev.solution.solution_id != solution.solution_id
                            {
                                sig_entry.arc_generation =
                                    sig_entry.arc_generation.checked_add(1).ok_or_else(|| {
                                        Error::InvalidInput(
                                            "phase continuity generation overflow".to_string(),
                                        )
                                    })?;
                                sig_entry.prior_break =
                                    Some(SsrDiscontinuityDetails::SolutionChanged {
                                        previous: prev.solution,
                                        current: solution,
                                    });
                                sig_entry.prior_rtcm_continuity =
                                    Some((bias.discontinuity_counter, ref_epoch_j2000_s.to_bits()));
                                ref_epoch_j2000_s.to_bits()
                            } else {
                                let prev_counter = match sig_entry.prior_rtcm_continuity {
                                    Some((c, _)) => c,
                                    None => prev.discontinuity.raw_value(),
                                };
                                if bias.discontinuity_counter != prev_counter {
                                    sig_entry.arc_generation = sig_entry
                                        .arc_generation
                                        .checked_add(1)
                                        .ok_or_else(|| {
                                            Error::InvalidInput(
                                                "phase continuity generation overflow".to_string(),
                                            )
                                        })?;
                                    sig_entry.prior_break = Some(
                                        SsrDiscontinuityDetails::RtcmDiscontinuityCounterChanged {
                                            previous: prev_counter,
                                            current: bias.discontinuity_counter,
                                        },
                                    );
                                    sig_entry.prior_rtcm_continuity = Some((
                                        bias.discontinuity_counter,
                                        ref_epoch_j2000_s.to_bits(),
                                    ));
                                    ref_epoch_j2000_s.to_bits()
                                } else {
                                    sig_entry
                                        .prior_rtcm_continuity
                                        .map(|(_, b)| b)
                                        .unwrap_or_else(|| ref_epoch_j2000_s.to_bits())
                                }
                            }
                        }
                    };
                    let token = PhaseContinuityToken {
                        sat,
                        signal: key,
                        source: SsrSource::RtcmSsr,
                        provider_id: solution.provider_id,
                        solution_id: solution.solution_id,
                        continuity_ref_epoch_bits,
                        raw_indicator: bias.discontinuity_counter,
                        generation: sig_entry.arc_generation,
                    };
                    sig_entry.active = Some(PhaseBiasSignalRecord {
                        signal,
                        value_m: Some(bias_m),
                        value_cycles: None,
                        solution,
                        iod_ssr: message.header.iod_ssr,
                        ref_epoch_j2000_s,
                        transmitted_epoch_j2000_s,
                        lifetime: SsrLifetime::RtcmUpdateInterval(update_interval_s),
                        discontinuity: PhaseDiscontinuityIndicator::RtcmDiscontinuityCounter(
                            bias.discontinuity_counter,
                        ),
                        token,
                    });
                }
            }
            SsrKind::Vtec => {}
        }
        self.corrections = staged;
        Ok(())
    }

    /// Ingest one decoded Galileo HAS MT1 correction message and return a detailed ingestion report.
    pub fn ingest_has_mt1_with_report(
        &mut self,
        message: &HasMt1Message,
        reception_gst: GnssWeekTow,
    ) -> Result<HasIngestionReport> {
        let ref_epoch_j2000_s = has_mt1_reference_j2000_s(reception_gst, message.header.toh_s)?;
        let solution = SsrSolution {
            source: SsrSource::GalileoHas,
            provider_id: u16::from(message.header.mask_id),
            solution_id: message.header.iod_set_id,
        };
        // Preflight orbit, clock, code bias, and phase bias blocks upfront so contradictory records,
        // reserved validity intervals, and non-finite values do not leave partial state.
        if let Some(orbit) = &message.orbit {
            if has_validity_interval_s(orbit.validity_interval).is_none() {
                return Err(Error::Parse("HAS orbit VI is reserved".to_string()));
            }
            let mut seen_sat = BTreeSet::new();
            for record in &orbit.records {
                if !seen_sat.insert(record.sat) {
                    return Err(Error::InvalidInput(format!(
                        "duplicate HAS orbit record for {}",
                        record.sat
                    )));
                }
                check_has_record_nav_message(message, "orbit", record.sat, record.nav_message)?;
                for (name, val) in [
                    ("radial", record.radial_m),
                    ("along", record.along_m),
                    ("cross", record.cross_m),
                ] {
                    if let Some(v) = val {
                        if !v.is_finite() {
                            return Err(Error::InvalidInput(format!(
                                "non-finite HAS orbit {name} correction for {}",
                                record.sat
                            )));
                        }
                    }
                }
            }
        }
        for clock in [
            message.clock_full_set.as_ref(),
            message.clock_subset.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if has_validity_interval_s(clock.validity_interval).is_none() {
                return Err(Error::Parse("HAS clock VI is reserved".to_string()));
            }
            let mut seen_sat = BTreeSet::new();
            for record in &clock.records {
                if !seen_sat.insert(record.sat) {
                    return Err(Error::InvalidInput(format!(
                        "duplicate HAS clock record for {}",
                        record.sat
                    )));
                }
                check_has_record_nav_message(message, "clock", record.sat, record.nav_message)?;
                if record.correction_m.is_some() && record.do_not_use {
                    return Err(Error::InvalidInput(format!(
                        "contradictory HAS clock correction for {}: correction is Some while do_not_use is true",
                        record.sat
                    )));
                }
                if let Some(v) = record.correction_m {
                    if !v.is_finite() {
                        return Err(Error::InvalidInput(format!(
                            "non-finite HAS clock correction for {}",
                            record.sat
                        )));
                    }
                }
            }
        }
        if let Some(code_bias) = &message.code_bias {
            if has_validity_interval_s(code_bias.validity_interval).is_none() {
                return Err(Error::Parse("HAS code bias VI is reserved".to_string()));
            }
            let mut seen_sig = BTreeSet::new();
            for record in &code_bias.records {
                if !seen_sig.insert((record.sat, record.signal_id)) {
                    return Err(Error::InvalidInput(format!(
                        "duplicate HAS code bias record for {} signal {}",
                        record.sat, record.signal_id
                    )));
                }
                if let Some(v) = record.bias_m {
                    if !v.is_finite() {
                        return Err(Error::InvalidInput(format!(
                            "non-finite HAS code bias for {} signal {}",
                            record.sat, record.signal_id
                        )));
                    }
                }
            }
        }
        if let Some(phase_bias) = &message.phase_bias {
            if has_validity_interval_s(phase_bias.validity_interval).is_none() {
                return Err(Error::Parse("HAS phase bias VI is reserved".to_string()));
            }
            let mut seen_sig = BTreeSet::new();
            for record in &phase_bias.records {
                if !seen_sig.insert((record.sat, record.signal_id)) {
                    return Err(Error::InvalidInput(format!(
                        "duplicate HAS phase bias record for {} signal {}",
                        record.sat, record.signal_id
                    )));
                }
                if record.discontinuity_indicator > 3 {
                    return Err(Error::InvalidInput(format!(
                        "invalid HAS phase discontinuity indicator {} for {} signal {}",
                        record.discontinuity_indicator, record.sat, record.signal_id
                    )));
                }
                if let Some(v) = record.bias_cycles {
                    if !v.is_finite() {
                        return Err(Error::InvalidInput(format!(
                            "non-finite HAS phase bias cycles for {} signal {}",
                            record.sat, record.signal_id
                        )));
                    }
                }
                if record.conversion() == HasPhaseBiasConversion::InvalidInput {
                    return Err(Error::InvalidInput(format!(
                        "invalid HAS phase bias conversion for {} signal {}",
                        record.sat, record.signal_id
                    )));
                }
            }
        }

        let mut staged = self.corrections.clone();
        let mut report = HasIngestionReport::default();

        if let Some(orbit) = &message.orbit {
            let update_interval_s = has_validity_interval_s(orbit.validity_interval)
                .ok_or_else(|| Error::Parse("HAS orbit VI is reserved".to_string()))?;
            for record in &orbit.records {
                let entry = staged.entry(record.sat).or_default();
                if let (Some(radial_m), Some(along_m), Some(cross_m)) =
                    (record.radial_m, record.along_m, record.cross_m)
                {
                    // Available complete orbit vector. The persistent watermark is checked
                    // first: it holds even when an RTCM orbit replaced the last HAS one.
                    if older_than_has_watermark(
                        entry.has_orbit_watermark_j2000_s,
                        ref_epoch_j2000_s,
                    ) {
                        continue;
                    }
                    if let Some(superseded_epoch) = entry.has_orbit_superseded_epoch_j2000_s {
                        if ref_epoch_j2000_s < superseded_epoch {
                            continue;
                        }
                    }
                    if let Some(existing_orbit) = &entry.orbit {
                        if existing_orbit.solution.source == SsrSource::GalileoHas
                            && ref_epoch_j2000_s < existing_orbit.ref_epoch_j2000_s
                        {
                            continue;
                        }
                    }
                    advance_has_watermark(
                        &mut entry.has_orbit_watermark_j2000_s,
                        ref_epoch_j2000_s,
                    );
                    entry.has_orbit_superseded_epoch_j2000_s = None;
                    entry.orbit = Some(SsrOrbitCorrection {
                        solution,
                        nav_message: SsrNavigationMessage::Has(record.nav_message),
                        iode: record.iode,
                        iod_ssr: message.header.iod_set_id,
                        basis: OrbitBasis::VelocityAligned,
                        crs_regional: false,
                        reference_point: SsrReferencePoint::AntennaPhaseCenter,
                        radial_m,
                        along_m,
                        cross_m,
                        radial_rate_m_s: 0.0,
                        along_rate_m_s: 0.0,
                        cross_rate_m_s: 0.0,
                        ref_epoch_j2000_s,
                        transmitted_epoch_j2000_s: ref_epoch_j2000_s,
                        update_interval_s,
                    });
                } else {
                    // Unavailable orbit vector sentinel (any component None)
                    if older_than_has_watermark(
                        entry.has_orbit_watermark_j2000_s,
                        ref_epoch_j2000_s,
                    ) {
                        continue;
                    }
                    if let Some(existing_orbit) = &entry.orbit {
                        if existing_orbit.solution.source == SsrSource::GalileoHas
                            && ref_epoch_j2000_s <= existing_orbit.ref_epoch_j2000_s
                        {
                            continue;
                        }
                    }
                    if let Some(superseded_epoch) = entry.has_orbit_superseded_epoch_j2000_s {
                        if ref_epoch_j2000_s <= superseded_epoch {
                            continue;
                        }
                    }
                    if entry
                        .orbit
                        .as_ref()
                        .is_some_and(|o| o.solution.source == SsrSource::GalileoHas)
                    {
                        entry.orbit = None;
                    }
                    advance_has_watermark(
                        &mut entry.has_orbit_watermark_j2000_s,
                        ref_epoch_j2000_s,
                    );
                    entry.has_orbit_superseded_epoch_j2000_s = Some(ref_epoch_j2000_s);
                }
            }
        }
        for clock in [
            message.clock_full_set.as_ref(),
            message.clock_subset.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let update_interval_s = has_validity_interval_s(clock.validity_interval)
                .ok_or_else(|| Error::Parse("HAS clock VI is reserved".to_string()))?;
            for record in &clock.records {
                let entry = staged.entry(record.sat).or_default();
                // The persistent watermarks are checked before the source-conditional
                // guards: they hold even when an RTCM clock replaced the last HAS one,
                // so a delayed older record, do-not-use included, is refused.
                let watermark = if record.do_not_use {
                    entry.has_clock_usable_watermark_j2000_s
                } else {
                    entry.has_clock_watermark_j2000_s
                };
                if older_than_has_watermark(watermark, ref_epoch_j2000_s) {
                    continue;
                }
                if record.do_not_use {
                    let effective_vi = if let Some(existing_ex) = entry.exclusion {
                        if ref_epoch_j2000_s < existing_ex.ref_epoch_j2000_s {
                            continue;
                        }
                        if ref_epoch_j2000_s == existing_ex.ref_epoch_j2000_s {
                            existing_ex.validity_interval_s.max(update_interval_s)
                        } else {
                            update_interval_s
                        }
                    } else {
                        update_interval_s
                    };
                    if let Some(existing_clock) = &entry.clock {
                        if existing_clock.solution.source == SsrSource::GalileoHas
                            && ref_epoch_j2000_s < existing_clock.ref_epoch_j2000_s
                        {
                            continue;
                        }
                    }
                    if entry
                        .has_clock_superseded_epoch_j2000_s
                        .is_some_and(|epoch| ref_epoch_j2000_s >= epoch)
                    {
                        entry.has_clock_superseded_epoch_j2000_s = None;
                    }
                    // A do-not-use indication excludes the satellite for its own validity
                    // interval; the exclusion marker below is what prevents use, and it is
                    // consulted by every query path. Destroying state from other providers is
                    // neither required nor safe: a HAS message delayed past its own validity
                    // would otherwise discard newer, still-usable RTCM SSR data, leaving the
                    // satellite uncorrectable at an epoch where it is not even excluded. So
                    // clear only HAS-sourced clock state that this record supersedes, matching
                    // the provenance and epoch guards used by the unavailable-sentinel branch.
                    if entry
                        .clock
                        .as_ref()
                        .is_some_and(|c| c.solution.source == SsrSource::GalileoHas)
                    {
                        entry.clock = None;
                    }
                    if entry.pending_high_rate.as_ref().is_some_and(|hr| {
                        hr.solution.source == SsrSource::GalileoHas
                            && hr.ref_epoch_j2000_s <= ref_epoch_j2000_s
                    }) {
                        entry.pending_high_rate = None;
                    }
                    advance_has_watermark(
                        &mut entry.has_clock_watermark_j2000_s,
                        ref_epoch_j2000_s,
                    );
                    entry.exclusion = Some(HasExclusionMarker {
                        ref_epoch_j2000_s,
                        validity_interval_s: effective_vi,
                    });
                } else if let Some(c0_m) = record.correction_m {
                    if let Some(existing_ex) = entry.exclusion {
                        if ref_epoch_j2000_s <= existing_ex.ref_epoch_j2000_s {
                            continue;
                        }
                    }
                    if let Some(superseded_epoch) = entry.has_clock_superseded_epoch_j2000_s {
                        if ref_epoch_j2000_s < superseded_epoch {
                            continue;
                        }
                    }
                    if let Some(existing_clock) = &entry.clock {
                        if existing_clock.solution.source == SsrSource::GalileoHas
                            && ref_epoch_j2000_s < existing_clock.ref_epoch_j2000_s
                        {
                            continue;
                        }
                    }
                    advance_has_watermark(
                        &mut entry.has_clock_watermark_j2000_s,
                        ref_epoch_j2000_s,
                    );
                    advance_has_watermark(
                        &mut entry.has_clock_usable_watermark_j2000_s,
                        ref_epoch_j2000_s,
                    );
                    entry.exclusion = None;
                    entry.has_clock_superseded_epoch_j2000_s = None;
                    entry.clock = Some(SsrClockCorrection {
                        solution,
                        nav_message: SsrNavigationMessage::Has(record.nav_message),
                        iod_ssr: message.header.iod_set_id,
                        c0_m,
                        c1_m_s: 0.0,
                        c2_m_s2: 0.0,
                        ref_epoch_j2000_s,
                        transmitted_epoch_j2000_s: ref_epoch_j2000_s,
                        update_interval_s,
                        high_rate: None,
                    });
                } else {
                    if let Some(existing_ex) = entry.exclusion {
                        if ref_epoch_j2000_s < existing_ex.ref_epoch_j2000_s {
                            continue;
                        }
                    }
                    if let Some(existing_clock) = &entry.clock {
                        if existing_clock.solution.source == SsrSource::GalileoHas
                            && ref_epoch_j2000_s <= existing_clock.ref_epoch_j2000_s
                        {
                            continue;
                        }
                    }
                    if let Some(superseded_epoch) = entry.has_clock_superseded_epoch_j2000_s {
                        if ref_epoch_j2000_s <= superseded_epoch {
                            continue;
                        }
                    }
                    if entry
                        .clock
                        .as_ref()
                        .is_some_and(|c| c.solution.source == SsrSource::GalileoHas)
                    {
                        entry.clock = None;
                    }
                    if entry.pending_high_rate.as_ref().is_some_and(|hr| {
                        hr.solution.source == SsrSource::GalileoHas
                            && hr.ref_epoch_j2000_s <= ref_epoch_j2000_s
                    }) {
                        entry.pending_high_rate = None;
                    }
                    advance_has_watermark(
                        &mut entry.has_clock_watermark_j2000_s,
                        ref_epoch_j2000_s,
                    );
                    entry.has_clock_superseded_epoch_j2000_s = Some(ref_epoch_j2000_s);
                }
            }
        }
        if let Some(code_bias) = &message.code_bias {
            let update_interval_s = has_validity_interval_s(code_bias.validity_interval)
                .ok_or_else(|| Error::Parse("HAS code bias VI is reserved".to_string()))?;
            let lifetime = SsrLifetime::GalileoHasValidityInterval(update_interval_s);
            for record in &code_bias.records {
                let signal = SsrRawSignal::galileo_has(record.sat.system, record.signal_id);
                let key = signal.key();
                let sat_entry = staged.entry(record.sat).or_default();
                let sig_entry = sat_entry.code_bias.signals.entry(key).or_default();
                let incoming_status = if record.bias_m.is_some() {
                    HasWatermarkStatus::Usable
                } else {
                    HasWatermarkStatus::Unavailable
                };

                let is_rtcm_active = sig_entry
                    .active
                    .as_ref()
                    .is_some_and(|r| r.solution.source == SsrSource::RtcmSsr);

                let reason = match sig_entry.has_watermark {
                    Some(wm) => {
                        if ref_epoch_j2000_s < wm.ref_epoch_j2000_s {
                            IngestionActionReason::RefusedOlderThanWatermark
                        } else if ref_epoch_j2000_s == wm.ref_epoch_j2000_s {
                            match (wm.status, incoming_status) {
                                (HasWatermarkStatus::Usable, HasWatermarkStatus::Unavailable) => {
                                    IngestionActionReason::RefusedEqualEpochUnavailableUnderUsable
                                }
                                (HasWatermarkStatus::Unavailable, HasWatermarkStatus::Usable) => {
                                    IngestionActionReason::AcceptedUsableOverUnavailable
                                }
                                (HasWatermarkStatus::Usable, HasWatermarkStatus::Usable) => {
                                    IngestionActionReason::AcceptedUpdatedRecord
                                }
                                (
                                    HasWatermarkStatus::Unavailable,
                                    HasWatermarkStatus::Unavailable,
                                ) => {
                                    if is_rtcm_active {
                                        IngestionActionReason::RetainedActiveRtcmOnHasUnavailable
                                    } else if sig_entry.active.as_ref().is_some_and(|prev| {
                                        prev.value_m.is_none()
                                            && prev.solution == solution
                                            && prev.iod_ssr == message.header.iod_set_id
                                            && prev.ref_epoch_j2000_s == ref_epoch_j2000_s
                                            && prev.lifetime == lifetime
                                    }) {
                                        IngestionActionReason::RetainedEquivalentUnavailable
                                    } else {
                                        IngestionActionReason::AcceptedUpdatedRecord
                                    }
                                }
                            }
                        } else {
                            match incoming_status {
                                HasWatermarkStatus::Usable => {
                                    IngestionActionReason::AcceptedNewerRecord
                                }
                                HasWatermarkStatus::Unavailable => {
                                    if is_rtcm_active {
                                        IngestionActionReason::RetainedActiveRtcmOnHasUnavailable
                                    } else {
                                        IngestionActionReason::AcceptedUnavailableClearingOlderHas
                                    }
                                }
                            }
                        }
                    }
                    None => match incoming_status {
                        HasWatermarkStatus::Usable => IngestionActionReason::AcceptedInitialRecord,
                        HasWatermarkStatus::Unavailable => {
                            if is_rtcm_active {
                                IngestionActionReason::RetainedActiveRtcmOnHasUnavailable
                            } else {
                                IngestionActionReason::AcceptedFirstUnavailableWithMetadata
                            }
                        }
                    },
                };

                let resulting_status = if reason.is_refused() {
                    match &sig_entry.active {
                        Some(rec) if rec.solution.source == SsrSource::RtcmSsr => {
                            ActiveProvenanceStatus::ActiveRtcmUsable
                        }
                        Some(rec) if rec.value_m.is_some() => {
                            ActiveProvenanceStatus::ActiveHasUsable
                        }
                        Some(_) => ActiveProvenanceStatus::ActiveHasUnavailable,
                        None => ActiveProvenanceStatus::NoActiveEntry,
                    }
                } else {
                    sig_entry.has_watermark = Some(HasStatusWatermark {
                        ref_epoch_j2000_s,
                        status: incoming_status,
                        pdi: None,
                    });
                    sig_entry.last_has_epoch_j2000_s = Some(ref_epoch_j2000_s);

                    match incoming_status {
                        HasWatermarkStatus::Usable => {
                            sig_entry.has_superseded_epoch_j2000_s = None;
                            sig_entry.active = Some(CodeBiasSignalRecord {
                                signal,
                                value_m: record.bias_m,
                                solution,
                                iod_ssr: message.header.iod_set_id,
                                ref_epoch_j2000_s,
                                transmitted_epoch_j2000_s: ref_epoch_j2000_s,
                                lifetime,
                            });
                            ActiveProvenanceStatus::ActiveHasUsable
                        }
                        HasWatermarkStatus::Unavailable => {
                            if is_rtcm_active {
                                ActiveProvenanceStatus::ActiveRtcmUsable
                            } else {
                                sig_entry.has_superseded_epoch_j2000_s = Some(ref_epoch_j2000_s);
                                sig_entry.active = Some(CodeBiasSignalRecord {
                                    signal,
                                    value_m: None,
                                    solution,
                                    iod_ssr: message.header.iod_set_id,
                                    ref_epoch_j2000_s,
                                    transmitted_epoch_j2000_s: ref_epoch_j2000_s,
                                    lifetime,
                                });
                                ActiveProvenanceStatus::ActiveHasUnavailable
                            }
                        }
                    }
                };

                report.code_records.push(HasCodeBiasIngestionRecord {
                    sat: record.sat,
                    signal,
                    key,
                    source: SsrSource::GalileoHas,
                    solution,
                    ref_epoch_j2000_s,
                    native_bias_m: record.bias_m,
                    reason,
                    resulting_status,
                });
            }
        }
        if let Some(phase_bias) = &message.phase_bias {
            let update_interval_s = has_validity_interval_s(phase_bias.validity_interval)
                .ok_or_else(|| Error::Parse("HAS phase bias VI is reserved".to_string()))?;
            let lifetime = SsrLifetime::GalileoHasValidityInterval(update_interval_s);
            for record in &phase_bias.records {
                let signal = SsrRawSignal::galileo_has(record.sat.system, record.signal_id);
                let key = signal.key();
                let sat_entry = staged.entry(record.sat).or_default();
                let sig_entry = sat_entry.phase_bias.signals.entry(key).or_default();
                let incoming_status = if record.bias_cycles.is_some() {
                    HasWatermarkStatus::Usable
                } else {
                    HasWatermarkStatus::Unavailable
                };

                let is_rtcm_active = sig_entry
                    .active
                    .as_ref()
                    .is_some_and(|r| r.solution.source == SsrSource::RtcmSsr);

                let reason = match sig_entry.has_watermark {
                    Some(wm) => {
                        if ref_epoch_j2000_s < wm.ref_epoch_j2000_s {
                            IngestionActionReason::RefusedOlderThanWatermark
                        } else if ref_epoch_j2000_s == wm.ref_epoch_j2000_s {
                            match (wm.status, incoming_status) {
                                (HasWatermarkStatus::Usable, HasWatermarkStatus::Unavailable) => {
                                    IngestionActionReason::RefusedEqualEpochUnavailableUnderUsable
                                }
                                (HasWatermarkStatus::Unavailable, HasWatermarkStatus::Usable) => {
                                    IngestionActionReason::AcceptedUsableOverUnavailable
                                }
                                (HasWatermarkStatus::Usable, HasWatermarkStatus::Usable) => {
                                    IngestionActionReason::AcceptedUpdatedRecord
                                }
                                (
                                    HasWatermarkStatus::Unavailable,
                                    HasWatermarkStatus::Unavailable,
                                ) => {
                                    // An active RTCM correction is retained whether or not the
                                    // incoming unavailable record revises its PDI: the active
                                    // correction does not change, so the outcome is a retention.
                                    // The revised PDI still advances the HAS watermark and is
                                    // reported verbatim in the record's own `pdi` field.
                                    if is_rtcm_active {
                                        IngestionActionReason::RetainedActiveRtcmOnHasUnavailable
                                    } else if wm.pdi == Some(record.discontinuity_indicator)
                                        && sig_entry.active.as_ref().is_some_and(|prev| {
                                            prev.value_m.is_none()
                                                && prev.value_cycles.is_none()
                                                && prev.solution == solution
                                                && prev.iod_ssr == message.header.iod_set_id
                                                && prev.ref_epoch_j2000_s == ref_epoch_j2000_s
                                                && prev.lifetime == lifetime
                                                && prev.discontinuity.raw_value()
                                                    == record.discontinuity_indicator
                                        })
                                    {
                                        IngestionActionReason::RetainedEquivalentUnavailable
                                    } else {
                                        IngestionActionReason::AcceptedUpdatedRecord
                                    }
                                }
                            }
                        } else {
                            match incoming_status {
                                HasWatermarkStatus::Usable => {
                                    IngestionActionReason::AcceptedNewerRecord
                                }
                                HasWatermarkStatus::Unavailable => {
                                    if is_rtcm_active {
                                        IngestionActionReason::RetainedActiveRtcmOnHasUnavailable
                                    } else {
                                        IngestionActionReason::AcceptedUnavailableClearingOlderHas
                                    }
                                }
                            }
                        }
                    }
                    None => match incoming_status {
                        HasWatermarkStatus::Usable => IngestionActionReason::AcceptedInitialRecord,
                        HasWatermarkStatus::Unavailable => {
                            if is_rtcm_active {
                                IngestionActionReason::RetainedActiveRtcmOnHasUnavailable
                            } else {
                                IngestionActionReason::AcceptedFirstUnavailableWithMetadata
                            }
                        }
                    },
                };

                if reason.is_refused() {
                    let resulting_status = match &sig_entry.active {
                        Some(rec) if rec.solution.source == SsrSource::RtcmSsr => {
                            ActiveProvenanceStatus::ActiveRtcmUsable
                        }
                        Some(rec) if rec.value_m.is_some() || rec.value_cycles.is_some() => {
                            ActiveProvenanceStatus::ActiveHasUsable
                        }
                        Some(_) => ActiveProvenanceStatus::ActiveHasUnavailable,
                        None => ActiveProvenanceStatus::NoActiveEntry,
                    };
                    report.phase_records.push(HasPhaseBiasIngestionRecord {
                        sat: record.sat,
                        signal,
                        key,
                        source: SsrSource::GalileoHas,
                        solution,
                        ref_epoch_j2000_s,
                        native_cycles: record.bias_cycles,
                        pdi: record.discontinuity_indicator,
                        continuity_token: sig_entry.active.as_ref().map(|r| r.token),
                        reason,
                        resulting_status,
                    });
                    continue;
                }

                sig_entry.has_watermark = Some(HasStatusWatermark {
                    ref_epoch_j2000_s,
                    status: incoming_status,
                    pdi: Some(record.discontinuity_indicator),
                });
                sig_entry.last_has_epoch_j2000_s = Some(ref_epoch_j2000_s);

                // If HAS unavailable arrived while RTCM is active, retain active RTCM completely
                if incoming_status == HasWatermarkStatus::Unavailable && is_rtcm_active {
                    let token = sig_entry.active.as_ref().map(|r| r.token);
                    report.phase_records.push(HasPhaseBiasIngestionRecord {
                        sat: record.sat,
                        signal,
                        key,
                        source: SsrSource::GalileoHas,
                        solution,
                        ref_epoch_j2000_s,
                        native_cycles: record.bias_cycles,
                        pdi: record.discontinuity_indicator,
                        continuity_token: token,
                        reason,
                        resulting_status: ActiveProvenanceStatus::ActiveRtcmUsable,
                    });
                    continue;
                }

                // Continuity evaluation for newly active HAS record
                let continuity_ref_epoch_bits = match &sig_entry.active {
                    None => {
                        sig_entry.prior_break = None;
                        sig_entry.last_has_continuity =
                            Some((record.discontinuity_indicator, ref_epoch_j2000_s.to_bits()));
                        ref_epoch_j2000_s.to_bits()
                    }
                    Some(prev) => {
                        // A HAS phase arc breaks only on a source switch or a PDI change
                        // (HAS SIS ICD 5.2.6.1, 7.4). The mask ID and IOD set ID change
                        // with the mask and the message set, not with the carrier-phase
                        // arc, so they are kept in the record and left out of this test.
                        if prev.solution.source != SsrSource::GalileoHas {
                            sig_entry.arc_generation =
                                sig_entry.arc_generation.checked_add(1).ok_or_else(|| {
                                    Error::InvalidInput(
                                        "phase continuity generation overflow".to_string(),
                                    )
                                })?;
                            sig_entry.prior_break =
                                Some(SsrDiscontinuityDetails::SolutionChanged {
                                    previous: prev.solution,
                                    current: solution,
                                });
                            sig_entry.last_has_continuity =
                                Some((record.discontinuity_indicator, ref_epoch_j2000_s.to_bits()));
                            ref_epoch_j2000_s.to_bits()
                        } else {
                            let prev_pdi = match sig_entry.last_has_continuity {
                                Some((pdi, _)) => pdi,
                                None => prev.discontinuity.raw_value(),
                            };
                            if record.discontinuity_indicator != prev_pdi {
                                sig_entry.arc_generation =
                                    sig_entry.arc_generation.checked_add(1).ok_or_else(|| {
                                        Error::InvalidInput(
                                            "phase continuity generation overflow".to_string(),
                                        )
                                    })?;
                                sig_entry.prior_break =
                                    Some(SsrDiscontinuityDetails::HasPdiChanged {
                                        previous: prev_pdi,
                                        current: record.discontinuity_indicator,
                                    });
                                sig_entry.last_has_continuity = Some((
                                    record.discontinuity_indicator,
                                    ref_epoch_j2000_s.to_bits(),
                                ));
                                ref_epoch_j2000_s.to_bits()
                            } else {
                                sig_entry
                                    .last_has_continuity
                                    .map(|(_, b)| b)
                                    .unwrap_or_else(|| ref_epoch_j2000_s.to_bits())
                            }
                        }
                    }
                };

                let token = PhaseContinuityToken {
                    sat: record.sat,
                    signal: key,
                    source: SsrSource::GalileoHas,
                    provider_id: solution.provider_id,
                    solution_id: solution.solution_id,
                    continuity_ref_epoch_bits,
                    raw_indicator: record.discontinuity_indicator,
                    generation: sig_entry.arc_generation,
                };

                let resulting_status = match incoming_status {
                    HasWatermarkStatus::Usable => {
                        sig_entry.has_superseded_epoch_j2000_s = None;
                        sig_entry.active = Some(PhaseBiasSignalRecord {
                            signal,
                            value_m: record.bias_m(),
                            value_cycles: record.bias_cycles,
                            solution,
                            iod_ssr: message.header.iod_set_id,
                            ref_epoch_j2000_s,
                            transmitted_epoch_j2000_s: ref_epoch_j2000_s,
                            lifetime,
                            discontinuity: PhaseDiscontinuityIndicator::GalileoHasPdi(
                                record.discontinuity_indicator,
                            ),
                            token,
                        });
                        ActiveProvenanceStatus::ActiveHasUsable
                    }
                    HasWatermarkStatus::Unavailable => {
                        sig_entry.has_superseded_epoch_j2000_s = Some(ref_epoch_j2000_s);
                        sig_entry.active = Some(PhaseBiasSignalRecord {
                            signal,
                            value_m: None,
                            value_cycles: None,
                            solution,
                            iod_ssr: message.header.iod_set_id,
                            ref_epoch_j2000_s,
                            transmitted_epoch_j2000_s: ref_epoch_j2000_s,
                            lifetime,
                            discontinuity: PhaseDiscontinuityIndicator::GalileoHasPdi(
                                record.discontinuity_indicator,
                            ),
                            token,
                        });
                        ActiveProvenanceStatus::ActiveHasUnavailable
                    }
                };

                report.phase_records.push(HasPhaseBiasIngestionRecord {
                    sat: record.sat,
                    signal,
                    key,
                    source: SsrSource::GalileoHas,
                    solution,
                    ref_epoch_j2000_s,
                    native_cycles: record.bias_cycles,
                    pdi: record.discontinuity_indicator,
                    continuity_token: Some(token),
                    reason,
                    resulting_status,
                });
            }
        }

        self.corrections = staged;
        Ok(report)
    }

    /// Ingest one decoded Galileo HAS MT1 correction message.
    pub fn ingest_has_mt1(
        &mut self,
        message: &HasMt1Message,
        reception_gst: GnssWeekTow,
    ) -> Result<()> {
        self.ingest_has_mt1_with_report(message, reception_gst)?;
        Ok(())
    }

    /// Orbit correction for a satellite.
    pub fn orbit(&self, sat: GnssSatelliteId) -> Option<&SsrOrbitCorrection> {
        self.corrections.get(&sat)?.orbit.as_ref()
    }

    /// Clock correction for a satellite.
    pub fn clock(&self, sat: GnssSatelliteId) -> Option<&SsrClockCorrection> {
        self.corrections.get(&sat)?.clock.as_ref()
    }

    /// URA index for a satellite.
    pub fn ura_index(&self, sat: GnssSatelliteId) -> Option<u8> {
        self.corrections.get(&sat)?.ura_index
    }

    /// Query code bias for a satellite, signal, and reception epoch.
    ///
    /// `signal` is looked up by its canonical key ([`SsrSignalKey::canonical`]): a
    /// raw signal whose source table assigns it a physical signal is that physical
    /// signal, whichever source's record the store holds for it. A record on an
    /// index its source's table leaves unassigned is reported
    /// [`SsrBiasStatus::UnknownSignal`] within its lifetime, with its value, and never as
    /// available.
    pub fn query_code_bias(
        &self,
        sat: GnssSatelliteId,
        signal: impl Into<SsrSignalKey>,
        t_j2000_s: f64,
    ) -> SsrCodeBiasQueryResult {
        let signal = signal.into().canonical();
        if !t_j2000_s.is_finite() {
            return SsrCodeBiasQueryResult {
                sat,
                signal,
                source_signal: None,
                status: SsrBiasStatus::InvalidEpoch,
                bias_m: None,
                solution: None,
                iod_ssr: None,
                ref_epoch_j2000_s: None,
                lifetime: None,
                details: SsrBiasResolutionDetails::InvalidEpoch,
            };
        }
        if self.is_satellite_excluded(sat, t_j2000_s) {
            let ex = self.corrections.get(&sat).and_then(|c| c.exclusion);
            let (ref_epoch_j2000_s, validity_interval_s) = ex
                .map(|e| (e.ref_epoch_j2000_s, e.validity_interval_s))
                .unwrap_or((0.0, 0.0));
            return SsrCodeBiasQueryResult {
                sat,
                signal,
                source_signal: None,
                status: SsrBiasStatus::Excluded,
                bias_m: None,
                solution: None,
                iod_ssr: None,
                ref_epoch_j2000_s: Some(ref_epoch_j2000_s),
                lifetime: Some(SsrLifetime::GalileoHasValidityInterval(validity_interval_s)),
                details: SsrBiasResolutionDetails::ExcludedByDoNotUse {
                    ref_epoch_j2000_s,
                    validity_interval_s,
                },
            };
        }
        let Some(sat_entry) = self.corrections.get(&sat) else {
            return SsrCodeBiasQueryResult {
                sat,
                signal,
                source_signal: None,
                status: SsrBiasStatus::Missing,
                bias_m: None,
                solution: None,
                iod_ssr: None,
                ref_epoch_j2000_s: None,
                lifetime: None,
                details: SsrBiasResolutionDetails::NoRecord,
            };
        };
        let Some(sig_entry) = sat_entry.code_bias.signals.get(&signal) else {
            return SsrCodeBiasQueryResult {
                sat,
                signal,
                source_signal: None,
                status: SsrBiasStatus::Missing,
                bias_m: None,
                solution: None,
                iod_ssr: None,
                ref_epoch_j2000_s: None,
                lifetime: None,
                details: SsrBiasResolutionDetails::NoRecord,
            };
        };
        let Some(record) = &sig_entry.active else {
            return SsrCodeBiasQueryResult {
                sat,
                signal,
                source_signal: None,
                status: SsrBiasStatus::Missing,
                bias_m: None,
                solution: None,
                iod_ssr: None,
                ref_epoch_j2000_s: None,
                lifetime: None,
                details: SsrBiasResolutionDetails::NoRecord,
            };
        };

        // Lifetime evaluation applies FIRST before availability resolution
        let ref_epoch = record.ref_epoch_j2000_s;
        let staleness_cap = self.staleness.max_staleness_s;
        let time_valid = match record.lifetime {
            SsrLifetime::GalileoHasValidityInterval(vi) => {
                let effective_dur = vi.min(staleness_cap);
                if t_j2000_s < ref_epoch {
                    Err(SsrBiasResolutionDetails::EpochBeforeReference {
                        ref_epoch_j2000_s: ref_epoch,
                        query_epoch_j2000_s: t_j2000_s,
                    })
                } else if t_j2000_s > ref_epoch + effective_dur {
                    Err(SsrBiasResolutionDetails::EpochExpired {
                        expiry_epoch_j2000_s: ref_epoch + effective_dur,
                        query_epoch_j2000_s: t_j2000_s,
                    })
                } else {
                    Ok(())
                }
            }
            // RTKLIB applies no age limit to RTCM SSR biases; the update interval
            // is a cadence, so only the store staleness cap bounds them, measured
            // from the transmitted epoch either side.
            SsrLifetime::RtcmUpdateInterval(_) => {
                let t0 = record.transmitted_epoch_j2000_s;
                let age = (t_j2000_s - t0).abs();
                if age <= staleness_cap {
                    Ok(())
                } else if t_j2000_s > t0 + staleness_cap {
                    Err(SsrBiasResolutionDetails::EpochExpired {
                        expiry_epoch_j2000_s: t0 + staleness_cap,
                        query_epoch_j2000_s: t_j2000_s,
                    })
                } else {
                    Err(SsrBiasResolutionDetails::EpochBeforeReference {
                        ref_epoch_j2000_s: t0,
                        query_epoch_j2000_s: t_j2000_s,
                    })
                }
            }
        };

        if let Err(details) = time_valid {
            let status = match details {
                SsrBiasResolutionDetails::EpochBeforeReference { .. } => SsrBiasStatus::NotYetValid,
                _ => SsrBiasStatus::Expired,
            };
            return SsrCodeBiasQueryResult {
                sat,
                signal,
                source_signal: Some(record.signal),
                status,
                bias_m: None,
                solution: Some(record.solution),
                iod_ssr: Some(record.iod_ssr),
                ref_epoch_j2000_s: Some(ref_epoch),
                lifetime: Some(record.lifetime),
                details,
            };
        }

        if let SsrSignalKey::Unknown(raw) = signal {
            return SsrCodeBiasQueryResult {
                sat,
                signal,
                source_signal: Some(record.signal),
                status: SsrBiasStatus::UnknownSignal,
                bias_m: record.value_m,
                solution: Some(record.solution),
                iod_ssr: Some(record.iod_ssr),
                ref_epoch_j2000_s: Some(ref_epoch),
                lifetime: Some(record.lifetime),
                details: SsrBiasResolutionDetails::UnknownSignal(raw),
            };
        }

        let Some(bias_m) = record.value_m else {
            return SsrCodeBiasQueryResult {
                sat,
                signal,
                source_signal: Some(record.signal),
                status: SsrBiasStatus::Unavailable,
                bias_m: None,
                solution: Some(record.solution),
                iod_ssr: Some(record.iod_ssr),
                ref_epoch_j2000_s: Some(record.ref_epoch_j2000_s),
                lifetime: Some(record.lifetime),
                details: SsrBiasResolutionDetails::TransmittedUnavailable,
            };
        };

        SsrCodeBiasQueryResult {
            sat,
            signal,
            source_signal: Some(record.signal),
            status: SsrBiasStatus::Available,
            bias_m: Some(bias_m),
            solution: Some(record.solution),
            iod_ssr: Some(record.iod_ssr),
            ref_epoch_j2000_s: Some(ref_epoch),
            lifetime: Some(record.lifetime),
            details: SsrBiasResolutionDetails::Available,
        }
    }

    /// Query phase bias for a satellite, signal, reception epoch, and caller continuity acknowledgement.
    ///
    /// With `acknowledged_token` set, a token that does not continue the current
    /// arc yields `PhaseDiscontinuityNeedsReset`. With `None` the caller starts a
    /// new arc from the returned token: the status is not a reset, and any break
    /// recorded since the previous arc is reported in `discontinuity_details`.
    ///
    /// `signal` is looked up by its canonical key, as in [`Self::query_code_bias`]. A
    /// record on an index its source's table leaves unassigned is reported
    /// [`SsrBiasStatus::UnknownSignal`] within its lifetime, with its transmitted value
    /// and continuity, and never as available.
    pub fn query_phase_bias(
        &self,
        sat: GnssSatelliteId,
        signal: impl Into<SsrSignalKey>,
        t_j2000_s: f64,
        acknowledged_token: Option<PhaseContinuityToken>,
    ) -> SsrPhaseBiasQueryResult {
        let signal = signal.into().canonical();
        if !t_j2000_s.is_finite() {
            return SsrPhaseBiasQueryResult {
                sat,
                signal,
                source_signal: None,
                status: SsrBiasStatus::InvalidEpoch,
                bias_m: None,
                bias_cycles: None,
                solution: None,
                iod_ssr: None,
                ref_epoch_j2000_s: None,
                lifetime: None,
                continuity_token: None,
                discontinuity_indicator: None,
                discontinuity_details: None,
                details: SsrBiasResolutionDetails::InvalidEpoch,
            };
        }
        if self.is_satellite_excluded(sat, t_j2000_s) {
            let ex = self.corrections.get(&sat).and_then(|c| c.exclusion);
            let (ref_epoch_j2000_s, validity_interval_s) = ex
                .map(|e| (e.ref_epoch_j2000_s, e.validity_interval_s))
                .unwrap_or((0.0, 0.0));
            return SsrPhaseBiasQueryResult {
                sat,
                signal,
                source_signal: None,
                status: SsrBiasStatus::Excluded,
                bias_m: None,
                bias_cycles: None,
                solution: None,
                iod_ssr: None,
                ref_epoch_j2000_s: Some(ref_epoch_j2000_s),
                lifetime: Some(SsrLifetime::GalileoHasValidityInterval(validity_interval_s)),
                continuity_token: None,
                discontinuity_indicator: None,
                discontinuity_details: None,
                details: SsrBiasResolutionDetails::ExcludedByDoNotUse {
                    ref_epoch_j2000_s,
                    validity_interval_s,
                },
            };
        }
        let Some(sat_entry) = self.corrections.get(&sat) else {
            return SsrPhaseBiasQueryResult {
                sat,
                signal,
                source_signal: None,
                status: SsrBiasStatus::Missing,
                bias_m: None,
                bias_cycles: None,
                solution: None,
                iod_ssr: None,
                ref_epoch_j2000_s: None,
                lifetime: None,
                continuity_token: None,
                discontinuity_indicator: None,
                discontinuity_details: None,
                details: SsrBiasResolutionDetails::NoRecord,
            };
        };
        let Some(sig_entry) = sat_entry.phase_bias.signals.get(&signal) else {
            return SsrPhaseBiasQueryResult {
                sat,
                signal,
                source_signal: None,
                status: SsrBiasStatus::Missing,
                bias_m: None,
                bias_cycles: None,
                solution: None,
                iod_ssr: None,
                ref_epoch_j2000_s: None,
                lifetime: None,
                continuity_token: None,
                discontinuity_indicator: None,
                discontinuity_details: None,
                details: SsrBiasResolutionDetails::NoRecord,
            };
        };
        let Some(record) = &sig_entry.active else {
            return SsrPhaseBiasQueryResult {
                sat,
                signal,
                source_signal: None,
                status: SsrBiasStatus::Missing,
                bias_m: None,
                bias_cycles: None,
                solution: None,
                iod_ssr: None,
                ref_epoch_j2000_s: None,
                lifetime: None,
                continuity_token: None,
                discontinuity_indicator: None,
                discontinuity_details: None,
                details: SsrBiasResolutionDetails::NoRecord,
            };
        };

        // Lifetime evaluation applies FIRST before availability resolution
        let ref_epoch = record.ref_epoch_j2000_s;
        let staleness_cap = self.staleness.max_staleness_s;
        let time_valid = match record.lifetime {
            SsrLifetime::GalileoHasValidityInterval(vi) => {
                let effective_dur = vi.min(staleness_cap);
                if t_j2000_s < ref_epoch {
                    Err(SsrBiasResolutionDetails::EpochBeforeReference {
                        ref_epoch_j2000_s: ref_epoch,
                        query_epoch_j2000_s: t_j2000_s,
                    })
                } else if t_j2000_s > ref_epoch + effective_dur {
                    Err(SsrBiasResolutionDetails::EpochExpired {
                        expiry_epoch_j2000_s: ref_epoch + effective_dur,
                        query_epoch_j2000_s: t_j2000_s,
                    })
                } else {
                    Ok(())
                }
            }
            // RTKLIB applies no age limit to RTCM SSR biases; the update interval
            // is a cadence, so only the store staleness cap bounds them, measured
            // from the transmitted epoch either side.
            SsrLifetime::RtcmUpdateInterval(_) => {
                let t0 = record.transmitted_epoch_j2000_s;
                let age = (t_j2000_s - t0).abs();
                if age <= staleness_cap {
                    Ok(())
                } else if t_j2000_s > t0 + staleness_cap {
                    Err(SsrBiasResolutionDetails::EpochExpired {
                        expiry_epoch_j2000_s: t0 + staleness_cap,
                        query_epoch_j2000_s: t_j2000_s,
                    })
                } else {
                    Err(SsrBiasResolutionDetails::EpochBeforeReference {
                        ref_epoch_j2000_s: t0,
                        query_epoch_j2000_s: t_j2000_s,
                    })
                }
            }
        };

        // Continuity is evaluated once, up front, so that an availability or validity
        // failure still reports a supplied token that does not match the active arc
        // rather than silently discarding the acknowledgement.
        let current_token = record.token;
        let continuity = evaluate_phase_continuity(
            sat,
            signal,
            current_token,
            sig_entry.prior_break,
            acknowledged_token,
        );

        if let Err(details) = time_valid {
            let status = match details {
                SsrBiasResolutionDetails::EpochBeforeReference { .. } => SsrBiasStatus::NotYetValid,
                _ => SsrBiasStatus::Expired,
            };
            return SsrPhaseBiasQueryResult {
                sat,
                signal,
                source_signal: Some(record.signal),
                status,
                bias_m: None,
                bias_cycles: record.value_cycles,
                solution: Some(record.solution),
                iod_ssr: Some(record.iod_ssr),
                ref_epoch_j2000_s: Some(ref_epoch),
                lifetime: Some(record.lifetime),
                continuity_token: Some(current_token),
                discontinuity_indicator: Some(record.discontinuity),
                discontinuity_details: Some(continuity),
                details,
            };
        }

        // A signal its source's table does not assign names no observation to apply
        // the bias to; the record keeps its native value and continuity.
        if let SsrSignalKey::Unknown(raw) = signal {
            return SsrPhaseBiasQueryResult {
                sat,
                signal,
                source_signal: Some(record.signal),
                status: SsrBiasStatus::UnknownSignal,
                bias_m: record.value_m,
                bias_cycles: record.value_cycles,
                solution: Some(record.solution),
                iod_ssr: Some(record.iod_ssr),
                ref_epoch_j2000_s: Some(ref_epoch),
                lifetime: Some(record.lifetime),
                continuity_token: Some(current_token),
                discontinuity_indicator: Some(record.discontinuity),
                discontinuity_details: Some(continuity),
                details: SsrBiasResolutionDetails::UnknownSignal(raw),
            };
        }

        // Wire unavailable sentinel and a failed metre conversion keep their primary
        // status and raw native values; only the continuity diagnostic is added.
        if record.value_m.is_none() {
            let details = if record.value_cycles.is_some() {
                SsrBiasResolutionDetails::ConversionUnavailable
            } else {
                SsrBiasResolutionDetails::TransmittedUnavailable
            };
            return SsrPhaseBiasQueryResult {
                sat,
                signal,
                source_signal: Some(record.signal),
                status: SsrBiasStatus::Unavailable,
                bias_m: None,
                bias_cycles: record.value_cycles,
                solution: Some(record.solution),
                iod_ssr: Some(record.iod_ssr),
                ref_epoch_j2000_s: Some(ref_epoch),
                lifetime: Some(record.lifetime),
                continuity_token: Some(current_token),
                discontinuity_indicator: Some(record.discontinuity),
                discontinuity_details: Some(continuity),
                details,
            };
        }

        // Without an acknowledgement token the caller holds no arc to continue, so
        // a break recorded before this query obliges no reset; it is reported in
        // `discontinuity_details` only.
        if acknowledged_token.is_some() && continuity_needs_reset(continuity) {
            return SsrPhaseBiasQueryResult {
                sat,
                signal,
                source_signal: Some(record.signal),
                status: SsrBiasStatus::PhaseDiscontinuityNeedsReset,
                bias_m: None,
                bias_cycles: record.value_cycles,
                solution: Some(record.solution),
                iod_ssr: Some(record.iod_ssr),
                ref_epoch_j2000_s: Some(ref_epoch),
                lifetime: Some(record.lifetime),
                continuity_token: Some(current_token),
                discontinuity_indicator: Some(record.discontinuity),
                discontinuity_details: Some(continuity),
                details: SsrBiasResolutionDetails::PhaseDiscontinuity(continuity),
            };
        }

        SsrPhaseBiasQueryResult {
            sat,
            signal,
            source_signal: Some(record.signal),
            status: SsrBiasStatus::Available,
            bias_m: record.value_m,
            bias_cycles: record.value_cycles,
            solution: Some(record.solution),
            iod_ssr: Some(record.iod_ssr),
            ref_epoch_j2000_s: Some(ref_epoch),
            lifetime: Some(record.lifetime),
            continuity_token: Some(current_token),
            discontinuity_indicator: Some(record.discontinuity),
            discontinuity_details: Some(continuity),
            details: SsrBiasResolutionDetails::Available,
        }
    }

    /// Code bias in meters for a satellite and signal.
    ///
    /// Untimed raw latest-value inspector retained for compatibility. It ignores
    /// the correction's lifetime and staleness, any HAS do-not-use exclusion of
    /// the satellite, phase continuity and whether the signal is one its source's
    /// table assigns: it returns the latest stored value whether or not it applies
    /// now. Positioning callers must use [`Self::query_code_bias`].
    pub fn code_bias(&self, sat: GnssSatelliteId, signal: impl Into<SsrSignalKey>) -> Option<f64> {
        self.corrections
            .get(&sat)?
            .code_bias
            .signals
            .get(&signal.into().canonical())?
            .active
            .as_ref()?
            .value_m
    }

    /// Phase bias for a satellite.
    ///
    /// Untimed raw latest-value inspector retained for compatibility. It ignores
    /// the correction's lifetime and staleness, any HAS do-not-use exclusion of
    /// the satellite, phase continuity and whether the signal is one its source's
    /// table assigns: it returns the latest stored value whether or not it applies
    /// now. Positioning callers must use [`Self::query_phase_bias`].
    pub fn phase_bias(&self, sat: GnssSatelliteId, signal: impl Into<SsrSignalKey>) -> Option<f64> {
        self.corrections
            .get(&sat)?
            .phase_bias
            .signals
            .get(&signal.into().canonical())?
            .active
            .as_ref()?
            .value_m
    }

    /// Whether the satellite has an active HAS do-not-use exclusion at the given epoch.
    fn is_satellite_excluded(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> bool {
        self.corrections
            .get(&sat)
            .and_then(|entry| entry.exclusion)
            .is_some_and(|ex| ex.is_active(t_j2000_s))
    }

    /// HAS do-not-use exclusion marker for a satellite, if present (test inspection helper).
    #[cfg(test)]
    fn has_exclusion(&self, sat: GnssSatelliteId) -> Option<&HasExclusionMarker> {
        self.corrections.get(&sat)?.exclusion.as_ref()
    }

    /// High-rate clock correction attached to satellite clock, if present (test inspection helper).
    #[cfg(test)]
    fn pending_high_rate(&self, sat: GnssSatelliteId) -> Option<&SsrHighRateClock> {
        self.corrections.get(&sat)?.pending_high_rate.as_ref()
    }

    /// HAS clock superseded reference epoch for a satellite, if present (test inspection helper).
    #[cfg(test)]
    fn has_clock_superseded_epoch(&self, sat: GnssSatelliteId) -> Option<f64> {
        self.corrections
            .get(&sat)?
            .has_clock_superseded_epoch_j2000_s
    }

    /// HAS orbit superseded reference epoch for a satellite, if present (test inspection helper).
    #[cfg(test)]
    pub(crate) fn has_orbit_superseded_epoch(&self, sat: GnssSatelliteId) -> Option<f64> {
        self.corrections
            .get(&sat)?
            .has_orbit_superseded_epoch_j2000_s
    }

    /// HAS code bias superseded reference epoch for a satellite and signal (test inspection helper).
    #[cfg(test)]
    pub(crate) fn has_code_bias_superseded_epoch(
        &self,
        sat: GnssSatelliteId,
        signal: impl Into<SsrSignalKey>,
    ) -> Option<f64> {
        self.corrections
            .get(&sat)?
            .code_bias
            .signals
            .get(&signal.into().canonical())?
            .has_superseded_epoch_j2000_s
    }

    /// HAS phase bias superseded reference epoch for a satellite and signal (test inspection helper).
    #[cfg(test)]
    pub(crate) fn has_phase_bias_superseded_epoch(
        &self,
        sat: GnssSatelliteId,
        signal: impl Into<SsrSignalKey>,
    ) -> Option<f64> {
        self.corrections
            .get(&sat)?
            .phase_bias
            .signals
            .get(&signal.into().canonical())?
            .has_superseded_epoch_j2000_s
    }
}

/// Evaluate a caller acknowledgement against the active record's continuity token.
///
/// This is the single continuity decision for phase-bias queries: every query outcome,
/// including unavailable, unknown-conversion and out-of-validity ones, reports the
/// result of this evaluation so a mismatched or stale acknowledgement is never lost.
///
/// Two tokens agree only when satellite, signal, source, arc generation,
/// continuity reference epoch and raw native indicator all agree, and, for RTCM
/// SSR, provider and solution too, so a token minted by a different store or for
/// an earlier arc that reuses a generation counter is rejected. A Galileo HAS
/// token's mask ID and IOD set ID are not compared: the HAS arc does not follow
/// them.
///
/// With no acknowledgement token the caller holds no arc to continue, so the
/// query establishes one: the result is reported without a reset, and any break
/// recorded since the previous arc is reported as the details.
fn evaluate_phase_continuity(
    sat: GnssSatelliteId,
    signal: SsrSignalKey,
    current_token: PhaseContinuityToken,
    prior_break: Option<SsrDiscontinuityDetails>,
    acknowledged_token: Option<PhaseContinuityToken>,
) -> SsrDiscontinuityDetails {
    let Some(ack) = acknowledged_token else {
        // Without an acknowledgement the caller learns about any break recorded since
        // the arc it last saw, or that this is the first token for the arc.
        return prior_break.unwrap_or(SsrDiscontinuityDetails::InitialTokenEstablished);
    };
    if ack.sat != sat || ack.signal != signal {
        return SsrDiscontinuityDetails::MismatchedToken;
    }
    // A HAS arc does not follow the mask ID or IOD set ID (HAS SIS ICD 5.2.6.1,
    // 7.4), so they are left out of a HAS token's identity; an RTCM arc follows
    // its provider and solution.
    let solution_differs = match current_token.source {
        SsrSource::GalileoHas => false,
        SsrSource::RtcmSsr => {
            ack.provider_id != current_token.provider_id
                || ack.solution_id != current_token.solution_id
        }
    };
    if ack.source != current_token.source || solution_differs {
        return SsrDiscontinuityDetails::SolutionChanged {
            previous: SsrSolution {
                source: ack.source,
                provider_id: ack.provider_id,
                solution_id: ack.solution_id,
            },
            current: SsrSolution {
                source: current_token.source,
                provider_id: current_token.provider_id,
                solution_id: current_token.solution_id,
            },
        };
    }
    let indicator_changed = || match current_token.source {
        SsrSource::GalileoHas => SsrDiscontinuityDetails::HasPdiChanged {
            previous: ack.raw_indicator,
            current: current_token.raw_indicator,
        },
        SsrSource::RtcmSsr => SsrDiscontinuityDetails::RtcmDiscontinuityCounterChanged {
            previous: ack.raw_indicator,
            current: current_token.raw_indicator,
        },
    };
    // Position on the arc timeline. A store only ever advances the generation counter,
    // and it stamps the continuity reference epoch whenever it does, so generation and
    // reference epoch must agree on the ordering. A pair that disagrees cannot have
    // been minted by any single store and is a mismatch, not a position.
    let generation_order = ack.generation.cmp(&current_token.generation);
    let epoch_order = if ack.continuity_ref_epoch_bits == current_token.continuity_ref_epoch_bits {
        Ordering::Equal
    } else {
        match ack
            .continuity_ref_epoch_j2000_s()
            .partial_cmp(&current_token.continuity_ref_epoch_j2000_s())
        {
            // Distinct bit patterns that compare equal or do not order at all (signed
            // zero, a non-finite epoch) name no point on the timeline.
            Some(Ordering::Less) => Ordering::Less,
            Some(Ordering::Greater) => Ordering::Greater,
            Some(Ordering::Equal) | None => return SsrDiscontinuityDetails::MismatchedToken,
        }
    };
    let arc_order = match (generation_order, epoch_order) {
        (Ordering::Equal, Ordering::Equal) => Ordering::Equal,
        (Ordering::Greater, Ordering::Greater | Ordering::Equal)
        | (Ordering::Equal, Ordering::Greater) => Ordering::Greater,
        (Ordering::Less, Ordering::Less | Ordering::Equal) | (Ordering::Equal, Ordering::Less) => {
            Ordering::Less
        }
        _ => return SsrDiscontinuityDetails::MismatchedToken,
    };
    match arc_order {
        Ordering::Greater => SsrDiscontinuityDetails::FutureToken,
        Ordering::Less => {
            if ack.raw_indicator == current_token.raw_indicator {
                SsrDiscontinuityDetails::StaleToken
            } else {
                indicator_changed()
            }
        }
        Ordering::Equal => {
            if ack.raw_indicator == current_token.raw_indicator {
                SsrDiscontinuityDetails::Continuous
            } else {
                indicator_changed()
            }
        }
    }
}

/// Whether a continuity evaluation obliges the caller to reset its ambiguity arc.
const fn continuity_needs_reset(details: SsrDiscontinuityDetails) -> bool {
    !matches!(
        details,
        SsrDiscontinuityDetails::Continuous | SsrDiscontinuityDetails::InitialTokenEstablished
    )
}

fn orbit_from_rtcm(
    reference_point: SsrReferencePoint,
    message: &SsrMessage,
    solution: SsrSolution,
    record: &crate::rtcm::SsrOrbitRecord,
    ref_epoch_j2000_s: f64,
    transmitted_epoch_j2000_s: f64,
    update_interval_s: f64,
) -> SsrOrbitCorrection {
    SsrOrbitCorrection {
        solution,
        nav_message: SsrNavigationMessage::Rtcm,
        iode: record.iode,
        iod_ssr: message.header.iod_ssr,
        basis: OrbitBasis::VelocityAligned,
        crs_regional: message.header.satellite_reference_datum.unwrap_or(false),
        reference_point,
        radial_m: -f64::from(record.delta_radial) * RTCM_SSR_RADIAL_CLOCK_SCALE_M,
        along_m: -f64::from(record.delta_along) * RTCM_SSR_ALONG_CROSS_SCALE_M,
        cross_m: -f64::from(record.delta_cross) * RTCM_SSR_ALONG_CROSS_SCALE_M,
        radial_rate_m_s: -f64::from(record.dot_delta_radial) * RTCM_SSR_RADIAL_CLOCK_RATE_SCALE_M_S,
        along_rate_m_s: -f64::from(record.dot_delta_along) * RTCM_SSR_ALONG_CROSS_RATE_SCALE_M_S,
        cross_rate_m_s: -f64::from(record.dot_delta_cross) * RTCM_SSR_ALONG_CROSS_RATE_SCALE_M_S,
        ref_epoch_j2000_s,
        transmitted_epoch_j2000_s,
        update_interval_s,
    }
}

/// Which RTKLIB age limit an RTCM SSR correction is held to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RtcmAgeLimit {
    /// Orbit and clock corrections: `MAXAGESSR`.
    OrbitClock,
    /// High-rate clock corrections: `MAXAGESSR_HRCLK`.
    HighRateClock,
}

/// Behavior when a correction is missing or stale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingCorrectionAction {
    /// Decline the satellite.
    Decline,
    /// Return the plain broadcast state.
    FallBackToBroadcast,
}

/// Regional CRS handling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegionalPolicy {
    /// Decline regional corrections.
    DeclineRegional,
    /// Allow regional corrections from these provider ids.
    AllowProviders(BTreeSet<u16>),
}

/// Missing-correction and regional policy for corrected ephemerides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsrFallbackPolicy {
    /// Missing or stale correction behavior.
    pub on_missing_correction: MissingCorrectionAction,
    /// Regional CRS behavior.
    pub regional: RegionalPolicy,
}

impl Default for SsrFallbackPolicy {
    fn default() -> Self {
        Self {
            on_missing_correction: MissingCorrectionAction::Decline,
            regional: RegionalPolicy::DeclineRegional,
        }
    }
}

/// An ephemeris source that applies SSR orbit and clock corrections from a store, as seen
/// by a positioning solve that has to know which SSR solution a satellite state came from.
///
/// [`ObservableEphemerisSource::ssr_corrections`] returns it for the SSR-corrected sources,
/// so a solve can check, at each observation's transmission time, that the SSR biases it
/// applies belong to the solution of the orbit and clock it uses.
pub trait SsrCorrectionSource {
    /// Store holding the SSR corrections the source applies.
    fn ssr_store(&self) -> &SsrCorrectionStore;

    /// Solution of the SSR orbit and clock corrections the source applies to `sat` at
    /// `t_j2000_s`, or `None` when it declines the satellite or returns a state without
    /// SSR corrections.
    fn applied_orbit_clock_solution(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<SsrSolution>;

    /// [`Self::applied_orbit_clock_solution`] with a UT1 refusal kept as
    /// `Err(`[`Error::Ut1OutsideCoverage`]`)`: the source refused the
    /// satellite's state under a strict UT1 policy, so no solution is known
    /// to be in use. The default implementation never refuses.
    fn try_applied_orbit_clock_solution(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<SsrSolution>> {
        Ok(self.applied_orbit_clock_solution(sat, t_j2000_s))
    }
}

impl SsrCorrectionSource for SsrCorrectedEphemeris<'_> {
    fn ssr_store(&self) -> &SsrCorrectionStore {
        self.store
    }

    fn applied_orbit_clock_solution(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<SsrSolution> {
        SsrCorrectedEphemeris::applied_orbit_clock_solution(self, sat, t_j2000_s)
    }

    fn try_applied_orbit_clock_solution(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<SsrSolution>> {
        self.applied_orbit_clock_checked(sat, t_j2000_s)
    }
}

impl SsrCorrectionSource for SsrCorrectedEphemerisOwned {
    fn ssr_store(&self) -> &SsrCorrectionStore {
        &self.store
    }

    fn applied_orbit_clock_solution(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<SsrSolution> {
        SsrCorrectedEphemerisOwned::applied_orbit_clock_solution(self, sat, t_j2000_s)
    }

    fn try_applied_orbit_clock_solution(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<SsrSolution>> {
        self.borrowed().applied_orbit_clock_checked(sat, t_j2000_s)
    }
}

/// Why an SSR-corrected source applies no SSR orbit and clock corrections to a satellite
/// at an epoch.
///
/// [`SsrCorrectedEphemeris::applied_orbit_clock_status`] returns it. Where a broadcast
/// fallback is allowed, the source then returns the broadcast state instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SsrStateUnavailable {
    /// A Galileo HAS do-not-use indication excludes the satellite at the epoch.
    ExcludedByHas,
    /// The store holds no orbit correction for the satellite.
    NoOrbitCorrection,
    /// The store holds no clock correction for the satellite.
    NoClockCorrection,
    /// The orbit and clock corrections belong to different solutions or IOD SSR values.
    OrbitClockMismatch,
    /// The orbit or clock correction refers to a Galileo HAS navigation-message index
    /// that HAS SIS ICD Table 21 reserves (1..=7). The correction is kept in the store,
    /// with the index as transmitted, and is not applied to any broadcast record.
    ReservedNavigationMessage {
        /// The navigation-message index the mask states.
        index: u8,
    },
    /// The orbit correction does not apply at the epoch.
    OrbitNotFresh,
    /// The clock correction does not apply at the epoch.
    ClockNotFresh,
    /// The orbit correction is regional and its provider is not allowed.
    RegionalProviderNotAllowed,
    /// The satellite's system has no broadcast model SSR corrections are applied to here
    /// (GPS, GLONASS, Galileo, QZSS and BeiDou have one).
    NoBroadcastModel,
    /// No broadcast record valid at the epoch has the issue the orbit correction names
    /// (its IODE; for BeiDou the IOD `mod(toe/720, 240)`; for GLONASS `tb`).
    NoMatchingBroadcastRecord {
        /// The IODE the orbit correction refers to.
        iode: u32,
    },
    /// The broadcast record gives no finite position at the epoch or 1 ms later.
    InvalidBroadcastState,
    /// The broadcast position and velocity give no radial, along-track and cross-track
    /// axes.
    DegenerateOrbitFrame,
    /// A centre-of-mass orbit cannot be moved to the antenna phase centre: no nominal
    /// attitude model, no ANTEX calibration for the satellite, or no Sun position.
    CenterOfMassUnresolved,
    /// A centre-of-mass orbit's move to the antenna phase centre reads UT1 (for the Sun
    /// direction) outside the UT1 table, and the source's UT1 policy is
    /// [`ValidityMode::Strict`]. The satellite is not given a broadcast fallback state
    /// instead: [`SsrCorrectedEphemeris::corrected_state_checked`] reports
    /// [`Error::Ut1OutsideCoverage`].
    Ut1OutsideCoverage(DegradeReason),
}

/// The broadcast state RTKLIB `satpos_ssr` starts from for one satellite and epoch.
struct SsrBroadcastState {
    /// Broadcast position, metres.
    position_m: [f64; 3],
    /// 1 ms forward-difference velocity, metres per second.
    velocity_m_s: [f64; 3],
    /// Satellite clock before the SSR clock correction, seconds.
    clock_s: f64,
    /// Single-frequency group delay of the record, seconds.
    group_delay_s: Option<f64>,
}

/// An SSR-corrected state and the solution of the corrections applied to it.
struct SsrAppliedState {
    /// Corrected position, metres.
    position_m: [f64; 3],
    /// Corrected satellite clock, seconds.
    clock_s: f64,
    /// Solution of the applied orbit and clock corrections.
    solution: SsrSolution,
    /// Single-frequency group delay of the broadcast record, seconds.
    group_delay_s: Option<f64>,
    /// UT1 departure the CoM-to-APC conversion accepted under a permissive policy.
    ut1_degraded: Option<DegradeReason>,
}

/// Which state an SSR-corrected source returns for a satellite at an epoch.
enum VelocitySource {
    Ssr,
    Broadcast,
    None,
    /// The SSR state is refused for reading UT1 outside the table.
    Ut1Refused(DegradeReason),
}

/// Broadcast ephemeris corrected by an SSR store.
#[derive(Clone)]
pub struct SsrCorrectedEphemeris<'a> {
    broadcast: &'a BroadcastEphemeris,
    store: &'a SsrCorrectionStore,
    antex: Option<&'a Antex>,
    attitude: SsrSatelliteAttitude,
    staleness: StalenessPolicy,
    fallback: SsrFallbackPolicy,
    ut1_validity: ValidityMode,
    ut1_departures: Ut1DepartureRecord,
}

impl<'a> SsrCorrectedEphemeris<'a> {
    /// Build a corrected source from borrowed broadcast and SSR stores.
    pub fn new(broadcast: &'a BroadcastEphemeris, store: &'a SsrCorrectionStore) -> Self {
        Self {
            broadcast,
            store,
            antex: None,
            attitude: SsrSatelliteAttitude::Unavailable,
            staleness: store.staleness(),
            fallback: SsrFallbackPolicy::default(),
            ut1_validity: ValidityMode::Strict,
            ut1_departures: Ut1DepartureRecord::default(),
        }
    }

    /// Set the UT1 policy for the Sun direction of the CoM-to-APC conversion.
    ///
    /// The nominal Sun-fixed attitude rotates the Sun into ECEF with UT1. The
    /// default, [`ValidityMode::Strict`], refuses an epoch outside the UT1 table:
    /// [`Self::corrected_state_checked`] and
    /// [`ObservableEphemerisSource::observable_state_at_j2000_s`] return
    /// [`Error::Ut1OutsideCoverage`], and the `Option` methods return `None`.
    /// Under [`ValidityMode::Permissive`] the conversion uses the long-term UT1;
    /// [`Self::corrected_state_checked`] reports the departure for that state
    /// and [`Self::ut1_departure`] the first one this source and its clones
    /// have used.
    pub fn with_validity(mut self, validity: ValidityMode) -> Self {
        self.ut1_validity = validity;
        self
    }

    /// The first departure from the UT1 table this source or a clone of it
    /// accepted under [`ValidityMode::Permissive`], or `None`.
    pub fn ut1_departure(&self) -> Option<DegradeReason> {
        self.ut1_departures.first()
    }

    fn with_departure_record(mut self, record: Ut1DepartureRecord) -> Self {
        self.ut1_departures = record;
        self
    }

    /// Attach satellite ANTEX calibrations for CoM-to-APC orbit conversion.
    pub fn with_satellite_antennas(mut self, antex: &'a Antex) -> Self {
        self.antex = Some(antex);
        self
    }

    /// Set the satellite attitude model for CoM-to-APC conversion.
    pub fn with_satellite_attitude(mut self, attitude: SsrSatelliteAttitude) -> Self {
        self.attitude = attitude;
        self
    }

    /// Set the staleness policy.
    pub fn with_staleness(mut self, policy: StalenessPolicy) -> Self {
        self.staleness = policy;
        self
    }

    /// Set missing-correction and regional behavior.
    pub fn with_fallback(mut self, policy: SsrFallbackPolicy) -> Self {
        self.fallback = policy;
        self
    }

    /// Mark one regional provider as applicable.
    pub fn allow_regional_provider(mut self, provider_id: u16) -> Self {
        match &mut self.fallback.regional {
            RegionalPolicy::DeclineRegional => {
                let mut providers = BTreeSet::new();
                providers.insert(provider_id);
                self.fallback.regional = RegionalPolicy::AllowProviders(providers);
            }
            RegionalPolicy::AllowProviders(providers) => {
                providers.insert(provider_id);
            }
        }
        self
    }

    /// SSR correction store this source reads.
    pub fn store(&self) -> &'a SsrCorrectionStore {
        self.store
    }

    /// Corrected ECEF position and satellite clock at a J2000 epoch.
    ///
    /// The clock is built as RTKLIB `satpos_ssr` builds it: the broadcast clock
    /// polynomial `af0 + af1·tk + af2·tk²` of the record the orbit correction's
    /// IODE selects, with `tk` the time from its `toc`, less `2 r·v / c²` for the
    /// broadcast position `r` and the 1 ms forward-difference velocity `v` of that
    /// record, plus the clock correction over `c`. The broadcast group delay (GPS
    /// TGD, Galileo BGD, BeiDou TGD) is not in it, and the relativistic term is
    /// the `r·v` one, not the broadcast `F·e·√A·sin E` one. For Galileo HAS this
    /// is HAS SIS ICD 7.3, Eq. 23 and 24; the HAS code biases take the place of
    /// the group delays (HAS SIS ICD 7.4). For RTCM SSR the correction adds to
    /// the clock as it does in RTKLIB and IGS SSR.
    ///
    /// A broadcast fallback state keeps the broadcast clock of
    /// [`EphemerisSource::position_clock_at_j2000_s`].
    ///
    /// `None` also when the CoM-to-APC conversion is refused outside the UT1
    /// table; [`Self::corrected_state_checked`] returns that reason.
    pub fn corrected_state(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<([f64; 3], f64)> {
        self.corrected_state_with_group_delay(sat, t_j2000_s)
            .map(|(position, clock, _)| (position, clock))
    }

    /// [`Self::corrected_state`] with the UT1 policy's outcome.
    ///
    /// `Err(`[`Error::Ut1OutsideCoverage`]`)` when the CoM-to-APC conversion
    /// reads UT1 outside the table under [`ValidityMode::Strict`]; the
    /// satellite is not given the broadcast state instead. Otherwise the state
    /// [`Self::corrected_state`] returns, with the departure accepted under
    /// [`ValidityMode::Permissive`] in [`Validated::degraded`].
    pub fn corrected_state_checked(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Validated<Option<PositionClock>>> {
        let checked = self.corrected_state_with_group_delay_checked(sat, t_j2000_s)?;
        Ok(Validated {
            value: checked.value.map(|(position, clock, _)| (position, clock)),
            degraded: checked.degraded,
        })
    }

    /// [`Self::corrected_state`] with its single-frequency group delay (see
    /// [`Self::single_frequency_group_delay_s`]), from one evaluation.
    ///
    /// `None` also when the CoM-to-APC conversion is refused outside the UT1
    /// table; [`Self::corrected_state_with_group_delay_checked`] returns that
    /// reason.
    pub fn corrected_state_with_group_delay(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<PositionClockGroupDelay> {
        self.corrected_state_with_group_delay_checked(sat, t_j2000_s)
            .ok()
            .and_then(|checked| checked.value)
    }

    /// [`Self::corrected_state_with_group_delay`] with the UT1 policy's outcome,
    /// as [`Self::corrected_state_checked`] reports it.
    pub fn corrected_state_with_group_delay_checked(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Validated<Option<PositionClockGroupDelay>>> {
        self.corrected_state_with_group_delay_checked_selected(sat, t_j2000_s, t_j2000_s)
    }

    /// [`Self::corrected_state_with_group_delay_checked`] with the broadcast record selected
    /// at `selection_j2000_s`, the observation epoch RTKLIB `satpos_ssr` selects at
    /// (`seleph(teph, ...)`): the record the orbit correction's IODE names nearest that
    /// epoch, or for a broadcast fallback the record the broadcast store selects there.
    /// The corrections themselves are applied at `t_j2000_s`.
    pub fn corrected_state_with_group_delay_checked_selected(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<Validated<Option<PositionClockGroupDelay>>> {
        if self.store.is_satellite_excluded(sat, t_j2000_s) {
            return Ok(Validated::ok(None));
        }
        match self.ssr_corrected_state(sat, t_j2000_s, selection_j2000_s) {
            Ok(state) => Ok(Validated {
                value: Some((state.position_m, state.clock_s, state.group_delay_s)),
                degraded: state.ut1_degraded,
            }),
            Err(SsrStateUnavailable::Ut1OutsideCoverage(reason)) => {
                Err(Error::Ut1OutsideCoverage(reason))
            }
            Err(_) => Ok(Validated::ok(self.broadcast_fallback_with_group_delay(
                sat,
                t_j2000_s,
                selection_j2000_s,
            ))),
        }
    }

    /// Solution of the SSR orbit and clock corrections that [`Self::corrected_state`]
    /// applies for `sat` at `t_j2000_s`.
    ///
    /// `None` when `corrected_state` would not return an SSR-corrected state; the reason is
    /// [`Self::applied_orbit_clock_status`]. In those cases `corrected_state` declines the
    /// satellite or returns the plain broadcast state, so no SSR solution's clock is in use.
    /// Both methods evaluate the same function.
    pub fn applied_orbit_clock_solution(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<SsrSolution> {
        self.applied_orbit_clock_status(sat, t_j2000_s).ok()
    }

    /// Solution of the SSR orbit and clock corrections that [`Self::corrected_state`]
    /// applies for `sat` at `t_j2000_s`, or why it applies none.
    pub fn applied_orbit_clock_status(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> std::result::Result<SsrSolution, SsrStateUnavailable> {
        if self.store.is_satellite_excluded(sat, t_j2000_s) {
            return Err(SsrStateUnavailable::ExcludedByHas);
        }
        self.ssr_corrected_state(sat, t_j2000_s, t_j2000_s)
            .map(|state| state.solution)
    }

    /// [`Self::applied_orbit_clock_solution`] with a UT1 refusal returned as
    /// [`Error::Ut1OutsideCoverage`].
    fn applied_orbit_clock_checked(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<SsrSolution>> {
        match self.applied_orbit_clock_status(sat, t_j2000_s) {
            Ok(solution) => Ok(Some(solution)),
            Err(SsrStateUnavailable::Ut1OutsideCoverage(reason)) => {
                Err(Error::Ut1OutsideCoverage(reason))
            }
            Err(_) => Ok(None),
        }
    }

    /// Group delay, seconds, a single-frequency pseudorange model subtracts from the clock
    /// of the state [`Self::corrected_state`] returns for `sat` at `t_j2000_s`.
    ///
    /// - An SSR-corrected state, RTCM SSR or Galileo HAS: the broadcast group delay of the
    ///   record the orbit correction's IODE selects. The SSR clock, as `satpos_ssr` builds
    ///   it, has none, and RTKLIB `pntpos` applies the broadcast TGD or BGD to a
    ///   single-frequency pseudorange whatever the ephemeris option. HAS SIS ICD 7.4 has
    ///   the HAS code biases replace the group delays; the SPP, DGNSS and tightly coupled
    ///   code models here apply no SSR code bias, so the broadcast delay is the
    ///   single-frequency term they have.
    /// - A broadcast fallback state: the broadcast group delay of the record it uses.
    /// - No state: `None`.
    pub fn single_frequency_group_delay_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<f64> {
        self.corrected_state_with_group_delay(sat, t_j2000_s)?.2
    }

    /// Satellite ECEF velocity, metres per second, of the state [`Self::corrected_state`]
    /// returns at `t_j2000_s`, or `None` when it returns no state.
    ///
    /// For an SSR-corrected state this is the velocity RTKLIB `satpos_ssr` returns: the
    /// velocity of the broadcast record selected by the orbit correction's IODE, formed by
    /// `ephpos` as the difference of that record's positions at `t_j2000_s` and 1 ms later.
    /// `satpos_ssr` adds no orbit-correction rate to it. For a broadcast fallback state it
    /// is the velocity of the broadcast record the fallback uses. It never differences
    /// across a correction reference epoch or between an SSR and a broadcast state.
    pub fn corrected_velocity(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<[f64; 3]> {
        self.corrected_velocity_selected(sat, t_j2000_s, t_j2000_s)
    }

    /// [`Self::corrected_velocity`] with the broadcast record selected at
    /// `selection_j2000_s`, as [`Self::corrected_state_with_group_delay_checked_selected`]
    /// selects it.
    fn corrected_velocity_selected(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<[f64; 3]> {
        match self.velocity_source(sat, t_j2000_s, selection_j2000_s) {
            VelocitySource::Ssr => self.ssr_broadcast_velocity(sat, t_j2000_s, selection_j2000_s),
            VelocitySource::Broadcast => {
                self.broadcast
                    .selected_record_velocity_at(sat, t_j2000_s, selection_j2000_s)
            }
            VelocitySource::None | VelocitySource::Ut1Refused(_) => None,
        }
    }

    /// Which state [`Self::corrected_state`] returns for `sat` at `t_j2000_s`, with the
    /// broadcast record selected at `selection_j2000_s`.
    fn velocity_source(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> VelocitySource {
        if self.store.is_satellite_excluded(sat, t_j2000_s) {
            return VelocitySource::None;
        }
        match self.ssr_corrected_state(sat, t_j2000_s, selection_j2000_s) {
            Ok(_) => return VelocitySource::Ssr,
            Err(SsrStateUnavailable::Ut1OutsideCoverage(reason)) => {
                return VelocitySource::Ut1Refused(reason)
            }
            Err(_) => {}
        }
        if self
            .broadcast_fallback_with_group_delay(sat, t_j2000_s, selection_j2000_s)
            .is_some()
        {
            VelocitySource::Broadcast
        } else {
            VelocitySource::None
        }
    }

    /// Velocity of the broadcast record selected by the SSR orbit correction's IODE, as
    /// RTKLIB `ephpos` forms it: a 1 ms forward difference of that record's positions.
    fn ssr_broadcast_velocity(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<[f64; 3]> {
        let orbit = self.store.orbit(sat)?;
        self.ssr_broadcast_state(sat, orbit, t_j2000_s, selection_j2000_s)
            .ok()
            .map(|state| state.velocity_m_s)
    }

    /// The broadcast state `satpos_ssr` starts from for `sat` at `t_j2000_s`: the position
    /// and the 1 ms forward-difference velocity of the record the orbit correction names,
    /// the satellite clock before the SSR correction, and the record's single-frequency
    /// group delay.
    ///
    /// - GPS, Galileo, QZSS, BeiDou: the record by IODE (BeiDou: by the IOD
    ///   `mod(toe/720, 240)`, IGS SSR v1.00 IDF012, in the low eight bits of the
    ///   transmitted issue); the clock is `af0 + af1·tk + af2·tk²` with `tk` from `toc`,
    ///   not iterated, less `2 r·v / c / c` (RTKLIB `satpos_ssr`; HAS SIS ICD Eq. 24).
    /// - GLONASS: the record whose `tb` is the IODE; the clock is `geph2pos`'s,
    ///   `-TauN + GammaN·tk`, with no relativistic term, as `satpos_ssr` leaves it.
    fn ssr_broadcast_state(
        &self,
        sat: GnssSatelliteId,
        orbit: &SsrOrbitCorrection,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> std::result::Result<SsrBroadcastState, SsrStateUnavailable> {
        use SsrStateUnavailable as Unavailable;
        if sat.system == GnssSystem::Glonass {
            let (r, v, clock_s) = self
                .broadcast
                .glonass_ssr_state(sat, orbit.iode, t_j2000_s, selection_j2000_s)
                .ok_or(Unavailable::NoMatchingBroadcastRecord { iode: orbit.iode })?;
            return Ok(SsrBroadcastState {
                position_m: r,
                velocity_m_s: v,
                clock_s,
                group_delay_s: None,
            });
        }
        let nav_message = ssr_nav_message(sat).ok_or(Unavailable::NoBroadcastModel)?;
        let (sow, is_geo) =
            ssr_seconds_of_week(sat, t_j2000_s).ok_or(Unavailable::NoBroadcastModel)?;
        let record = if sat.system == GnssSystem::BeiDou {
            self.broadcast.select_by_beidou_ssr_iod_at(
                sat,
                orbit.iode & 0xFF,
                nav_message,
                selection_j2000_s,
            )
        } else {
            let issue = BroadcastIssue {
                issue: orbit.iode,
                message: nav_message,
            };
            self.broadcast
                .select_by_issue_at(sat, issue, nav_message, selection_j2000_s)
        }
        .ok_or(Unavailable::NoMatchingBroadcastRecord { iode: orbit.iode })?;
        let (r, v) = broadcast_position_velocity(record, sow, is_geo)
            .ok_or(Unavailable::InvalidBroadcastState)?;

        // Satellite clock by the clock parameters, then the relativity correction
        // (RTKLIB `satpos_ssr`; HAS SIS ICD Eq. 24). `tk` is not iterated, as
        // `satpos_ssr` evaluates it.
        let tk_clock_s = crate::broadcast::time_from_reference_s(sow, record.clock.toc_sow);
        let mut clock_s = record.clock.af0
            + record.clock.af1 * tk_clock_s
            + record.clock.af2 * tk_clock_s * tk_clock_s;
        clock_s -= 2.0 * (r[0] * v[0] + r[1] * v[1] + r[2] * v[2]) / C_M_S / C_M_S;
        Ok(SsrBroadcastState {
            position_m: r,
            velocity_m_s: v,
            clock_s,
            group_delay_s: Some(record.broadcast_clock_group_delay_s()),
        })
    }

    /// SSR-corrected state and the solution of the orbit and clock corrections applied to
    /// it, or why the corrections cannot be applied at `t_j2000_s`.
    ///
    /// The statements follow RTKLIB `satpos_ssr`: the broadcast position and velocity of
    /// the IODE-selected record, the clock from that record's polynomial less `2 r·v / c²`,
    /// the orbit correction along the velocity-aligned axes, then the clock correction.
    ///
    /// A centre-of-mass orbit's move to the antenna phase centre reads UT1 under this
    /// source's UT1 policy: a refusal is [`SsrStateUnavailable::Ut1OutsideCoverage`], and
    /// an accepted departure is recorded on the source and carried in the state.
    fn ssr_corrected_state(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> std::result::Result<SsrAppliedState, SsrStateUnavailable> {
        use SsrStateUnavailable as Unavailable;
        let gate = Ut1Gate::new(self.ut1_validity);
        let orbit = self
            .store
            .orbit(sat)
            .ok_or(Unavailable::NoOrbitCorrection)?;
        let clock = self
            .store
            .clock(sat)
            .ok_or(Unavailable::NoClockCorrection)?;
        if orbit.solution != clock.solution || orbit.iod_ssr != clock.iod_ssr {
            return Err(Unavailable::OrbitClockMismatch);
        }
        if let Some(index) = orbit
            .nav_message
            .reserved_has_index()
            .or_else(|| clock.nav_message.reserved_has_index())
        {
            return Err(Unavailable::ReservedNavigationMessage { index });
        }
        if !self.correction_fresh(
            orbit.solution.source,
            t_j2000_s,
            orbit.ref_epoch_j2000_s,
            orbit.transmitted_epoch_j2000_s,
            orbit.update_interval_s,
            RtcmAgeLimit::OrbitClock,
        ) {
            return Err(Unavailable::OrbitNotFresh);
        }
        if !self.correction_fresh(
            clock.solution.source,
            t_j2000_s,
            clock.ref_epoch_j2000_s,
            clock.transmitted_epoch_j2000_s,
            clock.update_interval_s,
            RtcmAgeLimit::OrbitClock,
        ) {
            return Err(Unavailable::ClockNotFresh);
        }
        if orbit.crs_regional && !self.regional_allowed(orbit.solution.provider_id) {
            return Err(Unavailable::RegionalProviderNotAllowed);
        }

        let SsrBroadcastState {
            position_m: r,
            velocity_m_s: v,
            mut clock_s,
            group_delay_s,
        } = self.ssr_broadcast_state(sat, orbit, t_j2000_s, selection_j2000_s)?;

        let (er, ea, ec) = velocity_aligned_basis(r, v).ok_or(Unavailable::DegenerateOrbitFrame)?;
        let dt_orbit = t_j2000_s - orbit.ref_epoch_j2000_s;
        let radial = orbit.radial_m + orbit.radial_rate_m_s * dt_orbit;
        let along = orbit.along_m + orbit.along_rate_m_s * dt_orbit;
        let cross = orbit.cross_m + orbit.cross_rate_m_s * dt_orbit;
        // RTKLIB `satpos_ssr`: rs[i]+=-(er[i]*deph[0]+ea[i]*deph[1]+ec[i]*deph[2])+dant[i];
        // `radial`, `along` and `cross` are the negated `deph`, and negation is exact, so
        // the sum of the products is `satpos_ssr`'s negated term bit for bit and is added
        // to the broadcast position as one term.
        let mut corrected_position = [
            r[0] + (radial * er[0] + along * ea[0] + cross * ec[0]),
            r[1] + (radial * er[1] + along * ea[1] + cross * ec[1]),
            r[2] + (radial * er[2] + along * ea[2] + cross * ec[2]),
        ];
        if orbit.reference_point == SsrReferencePoint::CenterOfMass {
            let pco_ecef_m = self
                .satellite_pco_to_apc(sat, t_j2000_s, corrected_position, &gate)
                .ok_or_else(|| match gate.finish(()) {
                    Err(
                        crate::astro::frames::transforms::FrameTransformError::Ut1OutsideCoverage {
                            reason,
                        },
                    ) => Unavailable::Ut1OutsideCoverage(reason),
                    _ => Unavailable::CenterOfMassUnresolved,
                })?;
            corrected_position = add3(corrected_position, pco_ecef_m);
        }

        let dt_clock = t_j2000_s - clock.ref_epoch_j2000_s;
        let mut dclock_m =
            clock.c0_m + clock.c1_m_s * dt_clock + clock.c2_m_s2 * dt_clock * dt_clock;
        if let Some(high_rate) = clock.high_rate {
            if high_rate_matches(clock, &high_rate)
                && self.correction_fresh(
                    high_rate.solution.source,
                    t_j2000_s,
                    high_rate.ref_epoch_j2000_s,
                    high_rate.transmitted_epoch_j2000_s,
                    high_rate.update_interval_s,
                    RtcmAgeLimit::HighRateClock,
                )
            {
                dclock_m += high_rate.c0_m;
            }
        }
        // t_corr = t_sv - (dts(brdc) + dclk(ssr) / c): the correction adds to the clock
        // for RTCM SSR and Galileo HAS alike.
        clock_s += dclock_m / C_M_S;
        let ut1_degraded = match gate.finish(()) {
            Ok(validated) => validated.degraded,
            Err(crate::astro::frames::transforms::FrameTransformError::Ut1OutsideCoverage {
                reason,
            }) => return Err(Unavailable::Ut1OutsideCoverage(reason)),
            Err(crate::astro::frames::transforms::FrameTransformError::InvalidInput { .. }) => None,
        };
        self.ut1_departures.record(ut1_degraded);
        Ok(SsrAppliedState {
            position_m: corrected_position,
            clock_s,
            solution: clock.solution,
            group_delay_s,
            ut1_degraded,
        })
    }

    /// Whether a correction applies at `t_j2000_s`.
    ///
    /// A Galileo HAS correction applies from its TOH epoch through its validity
    /// interval, clipped to the store staleness cap (HAS SIS ICD 5.2.2.1). An
    /// RTCM SSR correction applies while its age from the transmitted epoch is
    /// within the store staleness cap and RTKLIB's limit: at most 90 s for
    /// orbit and clock (`MAXAGESSR`), under 10 s for a high-rate clock
    /// (`MAXAGESSR_HRCLK`). The update interval bounds no RTCM correction; it
    /// only moves the reference time of the rate terms.
    fn correction_fresh(
        &self,
        source: SsrSource,
        t_j2000_s: f64,
        ref_epoch_j2000_s: f64,
        transmitted_epoch_j2000_s: f64,
        update_interval_s: f64,
        rtcm_limit: RtcmAgeLimit,
    ) -> bool {
        let cap_s = self.staleness.max_staleness_s;
        match source {
            SsrSource::GalileoHas => {
                if !t_j2000_s.is_finite()
                    || !ref_epoch_j2000_s.is_finite()
                    || !update_interval_s.is_finite()
                    || update_interval_s < 0.0
                {
                    return false;
                }
                t_j2000_s >= ref_epoch_j2000_s
                    && t_j2000_s <= ref_epoch_j2000_s + cap_s.min(update_interval_s)
            }
            SsrSource::RtcmSsr => {
                if !t_j2000_s.is_finite() || !transmitted_epoch_j2000_s.is_finite() {
                    return false;
                }
                let age_s = (t_j2000_s - transmitted_epoch_j2000_s).abs();
                match rtcm_limit {
                    RtcmAgeLimit::OrbitClock => age_s <= cap_s.min(RTCM_SSR_MAX_AGE_S),
                    RtcmAgeLimit::HighRateClock => {
                        age_s < RTCM_SSR_HIGH_RATE_CLOCK_MAX_AGE_S && age_s <= cap_s
                    }
                }
            }
        }
    }

    fn regional_allowed(&self, provider_id: u16) -> bool {
        match &self.fallback.regional {
            RegionalPolicy::DeclineRegional => false,
            RegionalPolicy::AllowProviders(providers) => providers.contains(&provider_id),
        }
    }

    fn broadcast_fallback_with_group_delay(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        if self
            .store
            .orbit(sat)
            .is_some_and(|orbit| orbit.reference_point == SsrReferencePoint::CenterOfMass)
        {
            return None;
        }
        if self.fallback.on_missing_correction == MissingCorrectionAction::FallBackToBroadcast {
            EphemerisSource::try_position_clock_group_delay_selected_at_j2000_s(
                self.broadcast,
                sat,
                t_j2000_s,
                selection_j2000_s,
            )
            .ok()
            .flatten()
            .map(|state| state.value)
        } else {
            None
        }
    }

    fn satellite_pco_to_apc(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        sat_position_ecef_m: [f64; 3],
        gate: &Ut1Gate,
    ) -> Option<[f64; 3]> {
        if self.attitude != SsrSatelliteAttitude::NominalSunFixed {
            return None;
        }
        let antex = self.antex?;
        let epoch = antex_epoch_from_j2000_gpst(t_j2000_s)?;
        let antenna = antex.satellite_antenna(&sat.to_string(), epoch)?;
        let frequency = ssr_apc_frequency(sat.system)?;
        let pco_body_m = antenna.pco(frequency).ok()?;
        let ts = time_scales_from_j2000_gpst(t_j2000_s)?;
        // A UT1 refusal is remembered by `gate` and returned by the caller as
        // `SsrStateUnavailable::Ut1OutsideCoverage`, not as an unresolved offset.
        let ts = gate.admit(ts).ok()?;
        let sun_ecef_m = sun_moon_ecef(&ts).ok()?.sun;
        satellite_body_pco_to_ecef(pco_body_m, sat_position_ecef_m, sun_ecef_m)
    }
}

impl EphemerisSource for SsrCorrectedEphemeris<'_> {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        self.corrected_state(sat, t_j2000_s)
    }

    fn single_frequency_group_delay_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        SsrCorrectedEphemeris::single_frequency_group_delay_s(self, sat, t_j2000_s)
    }

    fn position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        self.corrected_state_with_group_delay(sat, t_j2000_s)
    }

    /// [`Self::corrected_state_checked`]: `Err(`[`Error::Ut1OutsideCoverage`]`)`
    /// when this source's UT1 policy refuses the CoM-to-APC conversion, and
    /// the accepted departure on the state otherwise.
    fn try_position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<Validated<PositionClock>>> {
        Ok(Validated::transpose(
            self.corrected_state_checked(sat, t_j2000_s)?,
        ))
    }

    fn try_position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<Validated<PositionClockGroupDelay>>> {
        Ok(Validated::transpose(
            self.corrected_state_with_group_delay_checked(sat, t_j2000_s)?,
        ))
    }

    /// [`SsrCorrectedEphemeris::corrected_state_with_group_delay_checked_selected`].
    fn try_position_clock_group_delay_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<Option<Validated<PositionClockGroupDelay>>> {
        Ok(Validated::transpose(
            self.corrected_state_with_group_delay_checked_selected(
                sat,
                t_j2000_s,
                selection_j2000_s,
            )?,
        ))
    }

    /// The broadcast clock polynomial of the store this source corrects: RTKLIB
    /// `satposs` places the transmission epoch with `ephclk` for the SSR ephemeris
    /// options too.
    fn try_transmit_epoch_clock_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<Option<Validated<f64>>> {
        Ok(self
            .broadcast
            .transmit_epoch_clock_s(sat, t_j2000_s, selection_j2000_s)
            .map(Validated::ok))
    }
}

impl ObservableEphemerisSource for SsrCorrectedEphemeris<'_> {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> std::result::Result<ObservableState, ObservablesError> {
        self.try_observable_state_at_j2000_s(sat, t_j2000_s)
            .map(|state| state.value)
    }

    fn try_observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> std::result::Result<Validated<ObservableState>, ObservablesError> {
        self.try_observable_state_group_delay_at_j2000_s(sat, t_j2000_s)
            .map(|state| Validated {
                value: state.value.0,
                degraded: state.degraded,
            })
    }

    fn ssr_corrections(&self) -> Option<&dyn SsrCorrectionSource> {
        Some(self)
    }

    /// True: an SSR-corrected clock carries `-2 r·v / c²` (RTKLIB `satpos_ssr`, HAS SIS
    /// ICD Eq. 24), and a broadcast fallback clock carries the broadcast relativistic
    /// term.
    fn clock_includes_relativity(&self) -> bool {
        true
    }

    fn single_frequency_group_delay_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        SsrCorrectedEphemeris::single_frequency_group_delay_s(self, sat, t_j2000_s)
    }

    fn observable_state_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> std::result::Result<(ObservableState, Option<f64>), ObservablesError> {
        self.try_observable_state_group_delay_at_j2000_s(sat, t_j2000_s)
            .map(|state| state.value)
    }

    fn try_observable_state_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> std::result::Result<Validated<(ObservableState, Option<f64>)>, ObservablesError> {
        let checked = self
            .corrected_state_with_group_delay_checked(sat, t_j2000_s)
            .map_err(ObservablesError::Ephemeris)?;
        let (position_ecef_m, clock_s, group_delay) =
            checked.value.ok_or(ObservablesError::NoEphemeris)?;
        Ok(Validated {
            value: (
                ObservableState {
                    position_ecef_m,
                    clock_s: Some(clock_s),
                },
                group_delay,
            ),
            degraded: checked.degraded,
        })
    }

    /// The broadcast clock polynomial of the store this source corrects, as
    /// [`EphemerisSource::try_transmit_epoch_clock_s`] gives it.
    fn try_observable_transmit_epoch_clock_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> std::result::Result<Validated<Option<f64>>, ObservablesError> {
        self.broadcast
            .transmit_epoch_clock_s(sat, t_j2000_s, selection_j2000_s)
            .map(|clock_s| Validated::ok(Some(clock_s)))
            .ok_or(ObservablesError::NoEphemeris)
    }

    /// [`SsrCorrectedEphemeris::corrected_state_with_group_delay_checked_selected`].
    fn try_observable_state_group_delay_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> std::result::Result<Validated<(ObservableState, Option<f64>)>, ObservablesError> {
        let checked = self
            .corrected_state_with_group_delay_checked_selected(sat, t_j2000_s, selection_j2000_s)
            .map_err(ObservablesError::Ephemeris)?;
        let (position_ecef_m, clock_s, group_delay) =
            checked.value.ok_or(ObservablesError::NoEphemeris)?;
        Ok(Validated {
            value: (
                ObservableState {
                    position_ecef_m,
                    clock_s: Some(clock_s),
                },
                group_delay,
            ),
            degraded: checked.degraded,
        })
    }

    fn velocity_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<std::result::Result<[f64; 3], ObservablesError>> {
        self.velocity_selected_at_j2000_s(sat, t_j2000_s, t_j2000_s)
    }

    fn velocity_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<std::result::Result<[f64; 3], ObservablesError>> {
        match self.velocity_source(sat, t_j2000_s, selection_j2000_s) {
            VelocitySource::Ssr => Some(
                self.ssr_broadcast_velocity(sat, t_j2000_s, selection_j2000_s)
                    .ok_or(ObservablesError::NoEphemeris),
            ),
            // The broadcast record the fallback state comes from; a system with no
            // broadcast record model has no state here either way.
            VelocitySource::Broadcast => self
                .broadcast
                .selected_record_velocity_at(sat, t_j2000_s, selection_j2000_s)
                .map(Ok),
            VelocitySource::None => Some(Err(ObservablesError::NoEphemeris)),
            VelocitySource::Ut1Refused(reason) => Some(Err(ObservablesError::Ephemeris(
                Error::Ut1OutsideCoverage(reason),
            ))),
        }
    }
}

/// Owned corrected ephemeris source.
#[derive(Clone)]
pub struct SsrCorrectedEphemerisOwned {
    broadcast: Arc<BroadcastEphemeris>,
    store: Arc<SsrCorrectionStore>,
    antex: Option<Arc<Antex>>,
    attitude: SsrSatelliteAttitude,
    staleness: StalenessPolicy,
    fallback: SsrFallbackPolicy,
    ut1_validity: ValidityMode,
    ut1_departures: Ut1DepartureRecord,
}

impl SsrCorrectedEphemerisOwned {
    /// Build an owned corrected source.
    pub fn new(broadcast: Arc<BroadcastEphemeris>, store: Arc<SsrCorrectionStore>) -> Self {
        let staleness = store.staleness();
        Self {
            broadcast,
            store,
            antex: None,
            attitude: SsrSatelliteAttitude::Unavailable,
            staleness,
            fallback: SsrFallbackPolicy::default(),
            ut1_validity: ValidityMode::Strict,
            ut1_departures: Ut1DepartureRecord::default(),
        }
    }

    /// Set the UT1 policy; see [`SsrCorrectedEphemeris::with_validity`].
    pub fn with_validity(mut self, validity: ValidityMode) -> Self {
        self.ut1_validity = validity;
        self
    }

    /// See [`SsrCorrectedEphemeris::ut1_departure`].
    pub fn ut1_departure(&self) -> Option<DegradeReason> {
        self.ut1_departures.first()
    }

    /// See [`SsrCorrectedEphemeris::corrected_state_checked`].
    pub fn corrected_state_checked(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Validated<Option<PositionClock>>> {
        self.borrowed().corrected_state_checked(sat, t_j2000_s)
    }

    /// See [`SsrCorrectedEphemeris::corrected_state_with_group_delay_checked`].
    pub fn corrected_state_with_group_delay_checked(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Validated<Option<PositionClockGroupDelay>>> {
        self.borrowed()
            .corrected_state_with_group_delay_checked(sat, t_j2000_s)
    }

    /// Attach satellite ANTEX calibrations for CoM-to-APC orbit conversion.
    pub fn with_satellite_antennas(mut self, antex: Arc<Antex>) -> Self {
        self.antex = Some(antex);
        self
    }

    /// Set the satellite attitude model for CoM-to-APC conversion.
    pub fn with_satellite_attitude(mut self, attitude: SsrSatelliteAttitude) -> Self {
        self.attitude = attitude;
        self
    }

    /// Set the staleness policy.
    pub fn with_staleness(mut self, policy: StalenessPolicy) -> Self {
        self.staleness = policy;
        self
    }

    /// Set missing-correction and regional behavior.
    pub fn with_fallback(mut self, policy: SsrFallbackPolicy) -> Self {
        self.fallback = policy;
        self
    }

    /// Mark one regional provider as applicable.
    pub fn allow_regional_provider(mut self, provider_id: u16) -> Self {
        match &mut self.fallback.regional {
            RegionalPolicy::DeclineRegional => {
                let mut providers = BTreeSet::new();
                providers.insert(provider_id);
                self.fallback.regional = RegionalPolicy::AllowProviders(providers);
            }
            RegionalPolicy::AllowProviders(providers) => {
                providers.insert(provider_id);
            }
        }
        self
    }

    /// Corrected ECEF position and satellite clock at a J2000 epoch; see
    /// [`SsrCorrectedEphemeris::corrected_state`].
    pub fn corrected_state(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<([f64; 3], f64)> {
        self.borrowed().corrected_state(sat, t_j2000_s)
    }

    /// See [`SsrCorrectedEphemeris::corrected_velocity`].
    pub fn corrected_velocity(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<[f64; 3]> {
        self.borrowed().corrected_velocity(sat, t_j2000_s)
    }

    /// Solution of the SSR orbit and clock corrections that [`Self::corrected_state`]
    /// applies; see [`SsrCorrectedEphemeris::applied_orbit_clock_solution`].
    pub fn applied_orbit_clock_solution(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<SsrSolution> {
        self.borrowed().applied_orbit_clock_solution(sat, t_j2000_s)
    }

    /// [`Self::corrected_state`] with its single-frequency group delay; see
    /// [`SsrCorrectedEphemeris::corrected_state_with_group_delay`].
    pub fn corrected_state_with_group_delay(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        self.borrowed()
            .corrected_state_with_group_delay(sat, t_j2000_s)
    }

    /// Single-frequency group delay of the state [`Self::corrected_state`] returns; see
    /// [`SsrCorrectedEphemeris::single_frequency_group_delay_s`].
    pub fn single_frequency_group_delay_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<f64> {
        self.borrowed()
            .single_frequency_group_delay_s(sat, t_j2000_s)
    }

    /// Solution of the SSR orbit and clock corrections that [`Self::corrected_state`]
    /// applies, or why it applies none; see
    /// [`SsrCorrectedEphemeris::applied_orbit_clock_status`].
    pub fn applied_orbit_clock_status(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> std::result::Result<SsrSolution, SsrStateUnavailable> {
        self.borrowed().applied_orbit_clock_status(sat, t_j2000_s)
    }

    /// Borrowed source with the same store, broadcast data, antennas and policies, for
    /// callers such as `PppCorrectionLookup::with_ssr_biases` that take one.
    pub fn as_borrowed(&self) -> SsrCorrectedEphemeris<'_> {
        self.borrowed()
    }

    fn borrowed(&self) -> SsrCorrectedEphemeris<'_> {
        let source = SsrCorrectedEphemeris::new(&self.broadcast, &self.store)
            .with_staleness(self.staleness)
            .with_fallback(self.fallback.clone())
            .with_satellite_attitude(self.attitude)
            .with_validity(self.ut1_validity)
            .with_departure_record(self.ut1_departures.clone());
        if let Some(antex) = &self.antex {
            source.with_satellite_antennas(antex)
        } else {
            source
        }
    }
}

impl EphemerisSource for SsrCorrectedEphemerisOwned {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        self.borrowed().position_clock_at_j2000_s(sat, t_j2000_s)
    }

    fn single_frequency_group_delay_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        SsrCorrectedEphemerisOwned::single_frequency_group_delay_s(self, sat, t_j2000_s)
    }

    fn position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        self.borrowed()
            .corrected_state_with_group_delay(sat, t_j2000_s)
    }

    fn try_position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<Validated<PositionClock>>> {
        self.borrowed()
            .try_position_clock_at_j2000_s(sat, t_j2000_s)
    }

    fn try_position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<Validated<PositionClockGroupDelay>>> {
        self.borrowed()
            .try_position_clock_group_delay_at_j2000_s(sat, t_j2000_s)
    }

    fn try_position_clock_group_delay_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<Option<Validated<PositionClockGroupDelay>>> {
        self.borrowed()
            .try_position_clock_group_delay_selected_at_j2000_s(sat, t_j2000_s, selection_j2000_s)
    }

    fn try_transmit_epoch_clock_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<Option<Validated<f64>>> {
        self.borrowed()
            .try_transmit_epoch_clock_s(sat, t_j2000_s, selection_j2000_s)
    }
}

impl ObservableEphemerisSource for SsrCorrectedEphemerisOwned {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> std::result::Result<ObservableState, ObservablesError> {
        self.borrowed().observable_state_at_j2000_s(sat, t_j2000_s)
    }

    fn try_observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> std::result::Result<Validated<ObservableState>, ObservablesError> {
        self.borrowed()
            .try_observable_state_at_j2000_s(sat, t_j2000_s)
    }

    fn try_observable_state_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> std::result::Result<Validated<(ObservableState, Option<f64>)>, ObservablesError> {
        self.borrowed()
            .try_observable_state_group_delay_at_j2000_s(sat, t_j2000_s)
    }

    fn ssr_corrections(&self) -> Option<&dyn SsrCorrectionSource> {
        Some(self)
    }

    /// True: an SSR-corrected clock carries `-2 r·v / c²` (RTKLIB `satpos_ssr`, HAS SIS
    /// ICD Eq. 24), and a broadcast fallback clock carries the broadcast relativistic
    /// term.
    fn clock_includes_relativity(&self) -> bool {
        true
    }

    fn single_frequency_group_delay_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        SsrCorrectedEphemerisOwned::single_frequency_group_delay_s(self, sat, t_j2000_s)
    }

    fn observable_state_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> std::result::Result<(ObservableState, Option<f64>), ObservablesError> {
        self.borrowed()
            .observable_state_group_delay_at_j2000_s(sat, t_j2000_s)
    }

    fn try_observable_transmit_epoch_clock_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> std::result::Result<Validated<Option<f64>>, ObservablesError> {
        self.borrowed()
            .try_observable_transmit_epoch_clock_s(sat, t_j2000_s, selection_j2000_s)
    }

    fn try_observable_state_group_delay_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> std::result::Result<Validated<(ObservableState, Option<f64>)>, ObservablesError> {
        self.borrowed()
            .try_observable_state_group_delay_selected_at_j2000_s(sat, t_j2000_s, selection_j2000_s)
    }

    fn velocity_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<std::result::Result<[f64; 3], ObservablesError>> {
        self.borrowed()
            .velocity_selected_at_j2000_s(sat, t_j2000_s, selection_j2000_s)
    }

    fn velocity_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<std::result::Result<[f64; 3], ObservablesError>> {
        self.borrowed().velocity_at_j2000_s(sat, t_j2000_s)
    }
}

/// Seconds for the RTCM SSR 4-bit update interval field.
///
/// The header field is public, so a hand-built message can hold any `u8`.
/// Values past 15 do not fit the 4-bit field and name no interval; they are
/// refused before any table lookup rather than indexing past the table.
fn update_interval_s(index: u8) -> Result<f64> {
    const TABLE: [f64; 16] = [
        1.0,
        2.0,
        5.0,
        10.0,
        15.0,
        30.0,
        60.0,
        120.0,
        240.0,
        300.0,
        600.0,
        900.0,
        1800.0,
        SECONDS_PER_HOUR,
        7200.0,
        10800.0,
    ];
    TABLE.get(usize::from(index)).copied().ok_or_else(|| {
        Error::Parse(format!(
            "SSR update interval index {index} does not fit the 4-bit update interval field (0..=15)"
        ))
    })
}

/// RTCM message number of the IGS SSR container (IGS SSR v1.00).
const IGS_SSR_MESSAGE_NUMBER: u16 = 4076;

/// Transmitted epoch of an RTCM SSR message, seconds since J2000 in GPS time.
///
/// `receiver` is the receiver time, in any GNSS, UTC or GLONASS week scale.
///
/// - GPS, Galileo, QZSS and BeiDou messages carry a 20-bit time of week, read
///   as GPS time as RTKLIB `decode_ssr_epoch` reads every non-GLONASS SSR
///   epoch. For native BeiDou messages (1258..1263, 1270) this is the
///   reference reader's choice rather than a definition: in real IGS
///   `SSRA03IGS0` streams the 1261 frames sent in the batch whose GPS frames
///   are stamped 223660 s, and whose GLONASS frames state the same instant,
///   hold 223656 s. Read as GPS time they sit 4 s before that batch epoch;
///   read as BDT they would sit 10 s after it.
/// - IGS SSR messages (4076) carry a GPS time of week for every GNSS,
///   GLONASS and BeiDou included (IGS SSR v1.00, IDF003).
///
/// A time of week is placed in the week nearest the receiver time, as RTKLIB
/// `adjweek` does, so a message stamped just before a week boundary and
/// received just after it keeps its week. A GLONASS RTCM SSR message (1063..1068)
/// carries a 17-bit GLONASS time of day, UTC(SU) + 3 h, placed on the GLONASS
/// day nearest the receiver time and converted to GPS time with the leap seconds
/// in force, as RTKLIB `adjday_glot` does.
fn ssr_epoch_j2000_s(
    system: GnssSystem,
    message_number: u16,
    receiver: GnssWeekTow,
    epoch_time_s: u32,
) -> Result<f64> {
    let receiver_gps_s = receiver_gps_s(receiver)?;
    let igs_ssr = message_number == IGS_SSR_MESSAGE_NUMBER;
    if system == GnssSystem::Glonass && !igs_ssr {
        return Ok(glonass_ssr_epoch_gps_s(receiver_gps_s, epoch_time_s) - GPS_EPOCH_TO_J2000_S);
    }
    let week_start_s = (receiver_gps_s / SECONDS_PER_WEEK).floor() * SECONDS_PER_WEEK;
    let receiver_tow_s = receiver_gps_s - week_start_s;
    let mut tow_s = f64::from(epoch_time_s);
    if tow_s < receiver_tow_s - SECONDS_PER_WEEK / 2.0 {
        tow_s += SECONDS_PER_WEEK;
    } else if tow_s > receiver_tow_s + SECONDS_PER_WEEK / 2.0 {
        tow_s -= SECONDS_PER_WEEK;
    }
    Ok(week_start_s + tow_s - GPS_EPOCH_TO_J2000_S)
}

/// GPS minus UTC, seconds, in force at `utc_s` seconds of UTC after the GPS
/// epoch, counted without leap seconds as RTKLIB `gtime_t` counts them.
fn gps_minus_utc_at_utc_s(utc_s: f64) -> f64 {
    crate::astro::time::scales::gps_utc_offset_s(GPS_EPOCH_JD + utc_s / SECONDS_PER_DAY)
}

/// The receiver time in seconds of GPS time after the GPS epoch.
fn receiver_gps_s(receiver: GnssWeekTow) -> Result<f64> {
    if !receiver.tow_s.is_finite() {
        return Err(Error::Parse(
            "SSR receiver time of week is not finite".to_string(),
        ));
    }
    let continuous_s = f64::from(receiver.week) * SECONDS_PER_WEEK + receiver.tow_s;
    Ok(match receiver.system {
        TimeScale::Gpst | TimeScale::Gst | TimeScale::Qzsst => continuous_s,
        TimeScale::Bdt => {
            continuous_s
                + crate::constants::BDS_EPOCH_MINUS_GPS_EPOCH_S
                + crate::constants::GPST_MINUS_BDT_S
        }
        // RTKLIB `utc2gpst`: the leap count is the one in force at that UTC.
        TimeScale::Utc => continuous_s + gps_minus_utc_at_utc_s(continuous_s),
        TimeScale::Glonasst => {
            let utc_s = continuous_s - GLONASS_MINUS_UTC_S;
            utc_s + gps_minus_utc_at_utc_s(utc_s)
        }
        other => {
            return Err(Error::Parse(format!(
                "an SSR epoch is placed against a GNSS, UTC or GLONASS receiver week, \
                 not a {} one",
                other.abbrev()
            )))
        }
    })
}

/// RTKLIB `adjday_glot`: the GLONASS time of day `tod_s` placed on the GLONASS
/// day nearest the receiver time, returned in seconds of GPS time after the
/// GPS epoch.
fn glonass_ssr_epoch_gps_s(receiver_gps_s: f64, tod_s: u32) -> f64 {
    // RTKLIB `gpst2utc`: the leap count is the one in force at the resulting UTC.
    let first_guess_utc_s = receiver_gps_s - gps_minus_utc_at_utc_s(receiver_gps_s);
    let receiver_utc_s = receiver_gps_s - gps_minus_utc_at_utc_s(first_guess_utc_s);
    let receiver_glonass_s = receiver_utc_s + GLONASS_MINUS_UTC_S;
    let day_start_s = (receiver_glonass_s / SECONDS_PER_DAY).floor() * SECONDS_PER_DAY;
    let receiver_tod_s = receiver_glonass_s - day_start_s;
    let mut tod_s = f64::from(tod_s);
    if tod_s < receiver_tod_s - SECONDS_PER_DAY / 2.0 {
        tod_s += SECONDS_PER_DAY;
    } else if tod_s > receiver_tod_s + SECONDS_PER_DAY / 2.0 {
        tod_s -= SECONDS_PER_DAY;
    }
    let utc_s = day_start_s + tod_s - GLONASS_MINUS_UTC_S;
    // RTKLIB `utc2gpst`: the leap count is the one in force at that UTC.
    utc_s + gps_minus_utc_at_utc_s(utc_s)
}

/// Build the identifier for an SSR record's raw satellite field.
///
/// The field is read the way RTKLIB `decode_ssr1`..`decode_ssr7` read it:
///
/// - GLONASS: five bits, the slot as transmitted.
/// - GPS, Galileo: six bits, the satellite number as transmitted.
/// - BeiDou: six bits, the PRN as transmitted. The native messages this crate
///   decodes (1258..1263, 1270) carry a 10-bit issue and an 8-bit IOD after the
///   satellite field. That is the layout real IGS `SSRA03IGS0` 1261 frames
///   have - 23 satellites fill 5013 of the frame's 5016 bits - and the layout
///   RTKLIB `decode_ssr1` reads in rtklib-ex 2.5.1 (`np=6, ni=10, nj=8`). The
///   bit count establishes the layout, not the satellite offset: the offset
///   follows rtklib-ex, the reader whose layout matches these frames, which
///   adds nothing (`offp=0`). RTKLIB demo5 adds one (`offp=1`) with its older
///   draft layout (`ni=10, nj=24`), which these frames do not have.
/// - QZSS: four bits in the native messages (1246..1251 and the 1268 phase
///   bias), six bits otherwise (the IGS SSR layout). In both the broadcast PRN
///   is the field plus 192, which is the `Jnn` slot the field states.
///
/// SBAS is refused. RTKLIB adds 120 to the native field (1252..1257) and 119
/// to the IGS SSR field, so the offset depends on a layout an `SsrMessage` here
/// does not identify, and the native layout carries an IOD CRC these records do
/// not hold. Reading the field as the `Snn` slot itself, as this function once
/// did, matched neither layout. NavIC is refused because no SSR layout for it
/// is read here.
///
/// The raw width is checked separately from the identifier range. The shared
/// satellite-token range is `1..=99` for every constellation, so `R32` is a
/// valid identifier, but a five-bit GLONASS field holds at most 31; the
/// identifier constructor cannot reject 32 there, and the width check does.
fn ssr_satellite(message: &SsrMessage, satellite_id: u8) -> Result<GnssSatelliteId> {
    let system = message.system;
    let field_bits = match system {
        GnssSystem::Glonass => 5u32,
        GnssSystem::Gps | GnssSystem::Galileo | GnssSystem::BeiDou => 6,
        GnssSystem::Qzss if crate::rtcm::is_native_qzss_ssr(message.message_number) => 4,
        GnssSystem::Qzss => 6,
        GnssSystem::Navic | GnssSystem::Sbas => {
            return Err(Error::Parse(format!(
                "no SSR layout read here carries {system} corrections, \
                 so satellite id {satellite_id} has no defined field layout"
            )))
        }
    };
    let widest = (1u16 << field_bits) - 1;
    if u16::from(satellite_id) > widest {
        return Err(Error::Parse(format!(
            "SSR {system} satellite id {satellite_id} does not fit the \
             {field_bits}-bit raw satellite field (0..={widest})"
        )));
    }
    GnssSatelliteId::new(system, satellite_id)
        .map_err(|e| Error::Parse(format!("invalid SSR satellite id {satellite_id}: {e}")))
}

/// Refuse a HAS orbit or clock record whose navigation-message index cannot have
/// been transmitted: wider than the 3-bit NM field, or, in a message carrying its
/// mask, not the index that mask states for the record's GNSS. A reserved index
/// (1..=7) is kept; the corrected source declines to apply it.
fn check_has_record_nav_message(
    message: &HasMt1Message,
    block_name: &str,
    sat: GnssSatelliteId,
    nav_message: u8,
) -> Result<()> {
    if nav_message > 7 {
        return Err(Error::InvalidInput(format!(
            "HAS {block_name} record for {sat} holds navigation message index \
             {nav_message}, wider than the 3-bit NM field (0..=7)"
        )));
    }
    let Some(mask) = &message.mask else {
        return Ok(());
    };
    match mask.systems.iter().find(|m| m.system == sat.system) {
        Some(system) if system.nav_message != nav_message => Err(Error::InvalidInput(format!(
            "HAS {block_name} record for {sat} holds navigation message index \
             {nav_message}, but the message's mask states {}",
            system.nav_message
        ))),
        _ => Ok(()),
    }
}

fn high_rate_matches(clock: &SsrClockCorrection, high_rate: &SsrHighRateClock) -> bool {
    clock.solution == high_rate.solution && clock.iod_ssr == high_rate.iod_ssr
}

/// Navigation message whose records an SSR orbit and clock correction for `sat` refers
/// to: the one RTKLIB `satpos_ssr` selects by IODE for GPS, Galileo, QZSS and BeiDou.
/// BeiDou geostationary satellites broadcast D2, the others D1.
fn ssr_nav_message(sat: GnssSatelliteId) -> Option<NavMessage> {
    match sat.system {
        GnssSystem::Gps => Some(NavMessage::GpsLnav),
        GnssSystem::Qzss => Some(NavMessage::QzssLnav),
        GnssSystem::Galileo => Some(NavMessage::GalileoInav),
        GnssSystem::BeiDou if is_beidou_geo(sat) => Some(NavMessage::BeidouD2),
        GnssSystem::BeiDou => Some(NavMessage::BeidouD1),
        _ => None,
    }
}

fn ssr_apc_frequency(system: GnssSystem) -> Option<&'static str> {
    match system {
        GnssSystem::Gps => Some("G01"),
        GnssSystem::Glonass => Some("R01"),
        GnssSystem::Galileo => Some("E01"),
        GnssSystem::BeiDou => Some("C02"),
        GnssSystem::Qzss => Some("J01"),
        GnssSystem::Navic => Some("I05"),
        GnssSystem::Sbas => Some("S01"),
    }
}

fn antex_epoch_from_j2000_gpst(t_j2000_s: f64) -> Option<AntexDateTime> {
    let (year, month, day, hour, minute, second) = civil_fields_from_j2000_gpst(t_j2000_s)?;
    AntexDateTime::new(year, month, day, hour, minute, second.trunc() as u8).ok()
}

fn time_scales_from_j2000_gpst(t_j2000_s: f64) -> Option<TimeScales> {
    let (year, month, day, hour, minute, second) = civil_fields_from_j2000_gpst(t_j2000_s)?;
    TimeScales::from_scale(
        TimeScale::Gpst,
        year,
        i32::from(month),
        i32::from(day),
        i32::from(hour),
        i32::from(minute),
        second,
    )
    .ok()
}

fn civil_fields_from_j2000_gpst(t_j2000_s: f64) -> Option<(i32, u8, u8, u8, u8, f64)> {
    if !t_j2000_s.is_finite() {
        return None;
    }
    let whole = t_j2000_s.floor();
    if whole < i64::MIN as f64 || whole > i64::MAX as f64 {
        return None;
    }
    let fraction = t_j2000_s - whole;
    let (year, month, day, hour, minute, second) = civil_from_j2000_seconds(whole as i64);
    Some((
        i32::try_from(year).ok()?,
        u8::try_from(month).ok()?,
        u8::try_from(day).ok()?,
        u8::try_from(hour).ok()?,
        u8::try_from(minute).ok()?,
        second as f64 + fraction,
    ))
}

/// Seconds of week of `t_j2000_s` in the time scale of `sat`'s broadcast records (GPS
/// time for GPS, Galileo and QZSS, BDT for BeiDou), and whether `sat` takes the BeiDou
/// geostationary orbit branch; `None` for a system without an SSR broadcast model here.
/// They are `t_j2000_s`'s seconds of week rounded once, exact for every epoch after
/// mid-January 2000 (see `crate::rinex_nav::query_native_time`).
fn ssr_seconds_of_week(sat: GnssSatelliteId, t_j2000_s: f64) -> Option<(f64, bool)> {
    crate::rinex_nav::query_native_time(sat, t_j2000_s).map(|(_, sow, is_geo)| (sow, is_geo))
}

/// Position and velocity of one broadcast record as RTKLIB `ephpos` forms them for
/// `satpos_ssr`, at seconds of week `sow` in the record's time scale: the position at
/// `tk`, and the difference of the positions at `tk` and [`EPHPOS_STEP_S`] later over
/// the step. The SSR-corrected state uses both for its clock's relativistic term and
/// its radial, along-track and cross-track basis, as `satpos_ssr` uses `rs` and `rs+3`.
fn broadcast_position_velocity(
    record: &crate::rinex_nav::BroadcastRecord,
    sow: f64,
    is_geo: bool,
) -> Option<([f64; 3], [f64; 3])> {
    let tk = crate::broadcast::time_from_reference_s(sow, record.elements.toe_sow);
    let position = |tk_s: f64| -> Option<[f64; 3]> {
        crate::broadcast::satellite_position_ecef_at_tk_unchecked(
            &record.elements,
            None,
            &record.constants(),
            tk_s,
            is_geo,
        )
        .position()
        .ok()
        .map(|position| position.as_array())
    };
    let start = position(tk)?;
    let end = position(ephpos_stepped_tk(tk))?;
    Some((
        start,
        [
            (end[0] - start[0]) / EPHPOS_STEP_S,
            (end[1] - start[1]) / EPHPOS_STEP_S,
            (end[2] - start[2]) / EPHPOS_STEP_S,
        ],
    ))
}

fn velocity_aligned_basis(r: [f64; 3], v: [f64; 3]) -> Option<([f64; 3], [f64; 3], [f64; 3])> {
    let ea = normalize(v)?;
    let rc = cross(r, v);
    let ec = normalize(rc)?;
    let er = cross(ea, ec);
    Some((er, ea, ec))
}

fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 0.0 && n.is_finite() {
        Some([v[0] / n, v[1] / n, v[2] / n])
    } else {
        None
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    /// Store key of Galileo HAS signal `index` of `sat`'s system.
    fn has_sig(sat: GnssSatelliteId, index: u8) -> SsrSignalKey {
        SsrRawSignal::galileo_has(sat.system, index).key()
    }

    /// Store key of RTCM SSR signal `index` of `sat`'s system.
    fn rtcm_sig(sat: GnssSatelliteId, index: u8) -> SsrSignalKey {
        SsrRawSignal::rtcm_ssr(sat.system, index).key()
    }
    use crate::astro::math::vec3::dot3;
    use crate::constants::{F_L1_HZ, F_L2_HZ};
    use crate::has::{
        has_mt1_reference_j2000_s, HasClockBlock, HasClockCorrection, HasClockSystem, HasCodeBias,
        HasCodeBiasBlock, HasGnssMask, HasMaskBlock, HasMt1Header, HasMt1Message, HasOrbitBlock,
        HasOrbitCorrection, HasPhaseBias, HasPhaseBiasBlock,
    };
    use crate::rtcm::{
        Message, SsrClockRecord, SsrHeader, SsrOrbitRecord, SsrPhaseBiasRecord, SsrPhaseBiasSignal,
        SsrStreamAssembler,
    };
    use crate::sp3::Sp3;

    const REAL_SSRA02IGS0_1060_FRAME_HEX: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/ssr/SSRA02IGS0_2026181234930_1060.hex"
    ));
    const REAL_SSR_WEEK: u32 = 2425;
    const REAL_SSR_EPOCH_TOW_S: f64 = 344_970.0;
    const GPS_ANTEX_TEXT: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/antex/igs20_wettzell_trim.atx"
    ));

    fn header(kind: SsrKind) -> SsrHeader {
        SsrHeader {
            epoch_time_s: 100_000,
            update_interval: 3,
            multiple_message: false,
            iod_ssr: 4,
            provider_id: 7,
            solution_id: 2,
            satellite_reference_datum: matches!(kind, SsrKind::Orbit | SsrKind::CombinedOrbitClock)
                .then_some(false),
            dispersive_bias_consistency: None,
            mw_consistency: None,
            satellite_count: 1,
        }
    }

    #[test]
    fn rtcm_ingest_scales_orbit_and_clock() {
        let message = SsrMessage {
            message_number: 1060,
            system: GnssSystem::Gps,
            kind: SsrKind::CombinedOrbitClock,
            header: header(SsrKind::CombinedOrbitClock),
            orbit: vec![SsrOrbitRecord {
                satellite_id: 1,
                iode: 42,
                delta_radial: 10_000,
                delta_along: -20_000,
                delta_cross: 30_000,
                dot_delta_radial: 100,
                dot_delta_along: -200,
                dot_delta_cross: 300,
            }],
            clock: vec![SsrClockRecord {
                satellite_id: 1,
                c0: 10_000,
                c1: -2_000,
                c2: 300,
            }],
            code_bias: Vec::new(),
            phase_bias: Vec::<SsrPhaseBiasRecord>::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };
        let mut store = SsrCorrectionStore::new();
        let week = GnssWeekTow::new(TimeScale::Gpst, 2_400, 100_000.0).unwrap();
        store.ingest_ssr(&message, week).unwrap();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let orbit = store.orbit(sat).unwrap();
        assert_eq!(orbit.iode, 42);
        assert_eq!(orbit.reference_point, SsrReferencePoint::rtcm_ssr_default());
        assert_eq!(orbit.radial_m.to_bits(), (-1.0_f64).to_bits());
        assert_eq!(orbit.along_m.to_bits(), 8.0_f64.to_bits());
        assert_eq!(orbit.cross_m.to_bits(), (-12.0_f64).to_bits());
        assert!((orbit.radial_rate_m_s + 1.0e-4).abs() < 1.0e-18);
        let clock = store.clock(sat).unwrap();
        assert_eq!(clock.c0_m.to_bits(), 1.0_f64.to_bits());
        assert!((clock.c1_m_s + 0.002).abs() < 1.0e-18);
        assert!((clock.c2_m_s2 - 6.0e-6).abs() < 1.0e-18);
    }

    /// Build a one-record SSR orbit message for the given constellation and raw
    /// satellite field, using the message number the decoder assigns to that
    /// constellation's orbit family.
    fn orbit_message(system: GnssSystem, satellite_id: u8) -> SsrMessage {
        let message_number = match system {
            GnssSystem::Gps => 1057,
            GnssSystem::Glonass => 1063,
            GnssSystem::Galileo => 1240,
            GnssSystem::BeiDou => 1258,
            GnssSystem::Qzss => 1246,
            GnssSystem::Sbas => 1252,
            // No SSR layout for NavIC is read here; the number is a placeholder
            // so the refusal can be exercised through ingestion.
            GnssSystem::Navic => 1057,
        };
        SsrMessage {
            message_number,
            system,
            kind: SsrKind::Orbit,
            header: header(SsrKind::Orbit),
            orbit: vec![SsrOrbitRecord {
                satellite_id,
                iode: 42,
                delta_radial: 10_000,
                delta_along: -20_000,
                delta_cross: 30_000,
                dot_delta_radial: 0,
                dot_delta_along: 0,
                dot_delta_cross: 0,
            }],
            clock: Vec::new(),
            code_bias: Vec::new(),
            phase_bias: Vec::<SsrPhaseBiasRecord>::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        }
    }

    fn ssr_week() -> GnssWeekTow {
        GnssWeekTow::new(TimeScale::Gpst, 2_400, 100_000.0).expect("valid SSR week")
    }

    /// The GLONASS SSR satellite field is five bits, so slot 31 is the widest
    /// value it carries. `R32` is a valid identifier under the shared 1..=99
    /// range, so the identifier constructor accepts 32; no five-bit field holds
    /// it, and the width check refuses it.
    #[test]
    fn ssr_glonass_satellite_field_is_five_bits_wide() {
        let mut store = SsrCorrectionStore::new();
        store
            .ingest_ssr(&orbit_message(GnssSystem::Glonass, 31), ssr_week())
            .expect("R31 fits the five-bit GLONASS SSR satellite field");
        let r31 = GnssSatelliteId::new(GnssSystem::Glonass, 31).unwrap();
        assert_eq!(store.orbit(r31).unwrap().iode, 42);

        let mut refusing = SsrCorrectionStore::new();
        let err = refusing
            .ingest_ssr(&orbit_message(GnssSystem::Glonass, 32), ssr_week())
            .expect_err("R32 does not fit the five-bit GLONASS SSR satellite field");
        assert!(
            err.to_string().contains("5-bit"),
            "the refusal must name the raw field width, got {err}"
        );
        assert!(
            GnssSatelliteId::new(GnssSystem::Glonass, 32).is_ok(),
            "R32 is a valid identifier, so only the width check refuses it"
        );
        assert!(
            refusing.corrections.is_empty(),
            "a refused record must not leave a correction behind"
        );
    }

    /// GPS, Galileo and BeiDou SSR records carry a six-bit satellite field: 63
    /// is the widest value, 64 does not fit. Accepting 63 is a statement about
    /// the wire field, not a claim that the constellation flies that satellite.
    #[test]
    fn ssr_six_bit_constellations_stop_at_sixty_three() {
        for system in [GnssSystem::Gps, GnssSystem::Galileo, GnssSystem::BeiDou] {
            let mut store = SsrCorrectionStore::new();
            store
                .ingest_ssr(&orbit_message(system, 63), ssr_week())
                .unwrap_or_else(|e| panic!("{system:?} 63 fits the six-bit field: {e}"));
            let sat = GnssSatelliteId::new(system, 63).unwrap();
            assert_eq!(store.orbit(sat).unwrap().iode, 42, "{system:?}");

            let mut refusing = SsrCorrectionStore::new();
            let err = refusing
                .ingest_ssr(&orbit_message(system, 64), ssr_week())
                .expect_err("64 does not fit a six-bit raw satellite field");
            assert!(
                err.to_string().contains("6-bit"),
                "{system:?} refusal must name the raw field width, got {err}"
            );
            assert!(
                refusing.corrections.is_empty(),
                "{system:?}: a refused record must not leave a correction behind"
            );
        }
    }

    /// The native BeiDou SSR messages carry the PRN itself, in the layout real
    /// IGS 1261 frames have and rtklib-ex reads (`offp=0`): field 63 is C63 and
    /// 64 does not fit the six-bit field.
    #[test]
    fn ssr_native_beidou_field_is_the_prn() {
        for field in [1u8, 59, 63] {
            let mut store = SsrCorrectionStore::new();
            store
                .ingest_ssr(&orbit_message(GnssSystem::BeiDou, field), ssr_week())
                .unwrap_or_else(|e| panic!("native field {field}: {e}"));
            let sat = GnssSatelliteId::new(GnssSystem::BeiDou, field).unwrap();
            assert_eq!(store.orbit(sat).unwrap().iode, 42, "field {field}");
        }
        let err = SsrCorrectionStore::new()
            .ingest_ssr(&orbit_message(GnssSystem::BeiDou, 64), ssr_week())
            .expect_err("64 does not fit the six-bit field");
        assert!(err.to_string().contains("6-bit"), "{err}");
    }

    /// Raw satellite field values that no decoder produces can still be written
    /// straight into a public `SsrOrbitRecord`. Zero names no satellite and
    /// 255 fits no SSR field; neither may be normalised, truncated onto another
    /// satellite, or dropped in silence.
    #[test]
    fn ssr_refuses_raw_satellite_field_bypasses() {
        for system in [
            GnssSystem::Gps,
            GnssSystem::Glonass,
            GnssSystem::Galileo,
            GnssSystem::BeiDou,
        ] {
            for satellite_id in [0u8, 255] {
                let mut store = SsrCorrectionStore::new();
                let err = store
                    .ingest_ssr(&orbit_message(system, satellite_id), ssr_week())
                    .expect_err("out-of-domain raw satellite id must be refused");
                assert!(
                    matches!(err, Error::Parse(_)),
                    "{system:?} {satellite_id}: refusal must be a typed parse error, got {err}"
                );
                assert!(
                    store.corrections.is_empty(),
                    "{system:?} {satellite_id}: nothing may be stored"
                );
            }
        }
    }

    /// QZSS SSR records are read as RTKLIB reads them: the broadcast PRN is the
    /// field plus 192, which is the `Jnn` slot the field states, over four bits
    /// in the native messages 1246..1251 and the 1268 phase bias, and six bits
    /// in the IGS SSR layout.
    #[test]
    fn ssr_qzss_satellite_is_the_slot_the_field_states() {
        let mut native = SsrCorrectionStore::new();
        native
            .ingest_ssr(&orbit_message(GnssSystem::Qzss, 15), ssr_week())
            .expect("J15 fits the four-bit native QZSS field");
        let j15 = GnssSatelliteId::new(GnssSystem::Qzss, 15).unwrap();
        assert_eq!(native.orbit(j15).unwrap().iode, 42);

        let mut refusing = SsrCorrectionStore::new();
        let err = refusing
            .ingest_ssr(&orbit_message(GnssSystem::Qzss, 16), ssr_week())
            .expect_err("16 does not fit the four-bit native QZSS field");
        assert!(err.to_string().contains("4-bit"), "{err}");
        assert!(refusing.corrections.is_empty());

        // The native QZSS phase bias 1268 carries the same four-bit field.
        let mut phase_bias_number = orbit_message(GnssSystem::Qzss, 16);
        phase_bias_number.message_number = 1268;
        let err = SsrCorrectionStore::new()
            .ingest_ssr(&phase_bias_number, ssr_week())
            .expect_err("1268 carries a four-bit QZSS field");
        assert!(err.to_string().contains("4-bit"), "{err}");

        // Outside 1246..1251 the field is the six-bit IGS SSR one.
        let mut igs = orbit_message(GnssSystem::Qzss, 63);
        igs.message_number = 4076;
        let mut store = SsrCorrectionStore::new();
        store
            .ingest_ssr(&igs, ssr_week())
            .expect("J63 fits the six-bit IGS SSR field");
        let j63 = GnssSatelliteId::new(GnssSystem::Qzss, 63).unwrap();
        assert!(store.orbit(j63).is_some());
        igs.orbit[0].satellite_id = 64;
        let err = SsrCorrectionStore::new()
            .ingest_ssr(&igs, ssr_week())
            .expect_err("64 does not fit a six-bit field");
        assert!(err.to_string().contains("6-bit"), "{err}");
    }

    /// SBAS and NavIC have no SSR layout read here. RTKLIB offsets the SBAS
    /// field by 120 or 119 depending on the layout, and reading the field as
    /// the slot itself matched neither, so a hand-built message naming either
    /// system is refused rather than given a guessed width and offset.
    #[test]
    fn ssr_refuses_constellations_with_no_layout_read_here() {
        for system in [GnssSystem::Navic, GnssSystem::Sbas] {
            let mut store = SsrCorrectionStore::new();
            let err = store
                .ingest_ssr(&orbit_message(system, 5), ssr_week())
                .expect_err("no SSR layout read here carries this constellation");
            assert!(
                err.to_string().contains("no SSR layout read here"),
                "{system:?} refusal must say why, got {err}"
            );
            assert!(store.corrections.is_empty(), "{system:?}");
        }
    }

    /// A message is applied to a staged copy that replaces the store only when
    /// every record succeeds, so a message with one refused record changes
    /// nothing in the store, wherever in the message that record sits.
    #[test]
    fn ssr_refusal_leaves_the_store_unchanged() {
        let template = orbit_message(GnssSystem::Gps, 5);
        let good = template.orbit[0].clone();
        let bad = SsrOrbitRecord {
            satellite_id: 64,
            ..good.clone()
        };
        for records in [vec![good.clone(), bad.clone()], vec![bad, good]] {
            let mut message = template.clone();
            message.orbit = records;
            message.header.satellite_count = 2;

            let mut store = SsrCorrectionStore::new();
            store
                .ingest_ssr(&message, ssr_week())
                .expect_err("one record does not fit the six-bit field");
            assert!(
                store.corrections.is_empty(),
                "a refused message applies no record"
            );
        }
    }

    /// The update interval header field is four bits wide. A hand-built header
    /// can hold any `u8`; 16 and above name no interval and are refused by
    /// name before any lookup, leaving the store unchanged.
    #[test]
    fn ssr_update_interval_past_four_bits_is_refused_without_panicking() {
        let mut store = SsrCorrectionStore::new();
        store
            .ingest_ssr(&orbit_message(GnssSystem::Gps, 5), ssr_week())
            .expect("seed record");
        let before = store.clone();
        for index in [16u8, 255] {
            let mut message = orbit_message(GnssSystem::Gps, 6);
            message.header.update_interval = index;
            let err = store
                .ingest_ssr(&message, ssr_week())
                .expect_err("update interval index past 15 must be refused");
            assert!(matches!(err, Error::Parse(_)), "{index}: {err}");
            assert!(err.to_string().contains("4-bit update interval"), "{err}");
            assert_eq!(store, before, "index {index}: nothing may be applied");
        }
        let mut widest = orbit_message(GnssSystem::Gps, 6);
        widest.header.update_interval = 15;
        store
            .ingest_ssr(&widest, ssr_week())
            .expect("index 15 is the widest the 4-bit field holds");
        let g06 = GnssSatelliteId::new(GnssSystem::Gps, 6).unwrap();
        assert_eq!(store.orbit(g06).unwrap().update_interval_s, 10800.0);
    }

    fn combined_message(pairs: &[(u8, u8)]) -> SsrMessage {
        let template = orbit_message(GnssSystem::Gps, 1);
        let orbit_record = template.orbit[0].clone();
        SsrMessage {
            message_number: 1060,
            kind: SsrKind::CombinedOrbitClock,
            header: header(SsrKind::CombinedOrbitClock),
            orbit: pairs
                .iter()
                .map(|&(orbit_id, _)| SsrOrbitRecord {
                    satellite_id: orbit_id,
                    ..orbit_record.clone()
                })
                .collect(),
            clock: pairs
                .iter()
                .map(|&(_, clock_id)| SsrClockRecord {
                    satellite_id: clock_id,
                    c0: 100,
                    c1: 0,
                    c2: 0,
                })
                .collect(),
            ..template
        }
    }

    /// A combined orbit/clock message pairs records by position. A clock
    /// record naming a different satellite from its orbit record, or lists of
    /// different lengths, are refused by name instead of putting one
    /// satellite's clock on another's orbit or dropping the unpaired records.
    #[test]
    fn ssr_combined_orbit_clock_refuses_unpaired_records() {
        let mut store = SsrCorrectionStore::new();
        store
            .ingest_ssr(&combined_message(&[(5, 5), (7, 7)]), ssr_week())
            .expect("paired records apply");
        let g05 = GnssSatelliteId::new(GnssSystem::Gps, 5).unwrap();
        let g07 = GnssSatelliteId::new(GnssSystem::Gps, 7).unwrap();
        assert!(store.orbit(g05).is_some() && store.clock(g05).is_some());
        assert!(store.orbit(g07).is_some() && store.clock(g07).is_some());
        let before = store.clone();

        let err = store
            .ingest_ssr(&combined_message(&[(5, 5), (8, 9)]), ssr_week())
            .expect_err("clock satellite differs from orbit satellite");
        assert!(matches!(err, Error::InvalidInput(_)), "{err}");
        assert!(
            err.to_string()
                .contains("record 1 names satellite id 8 for its orbit and 9 for its clock"),
            "{err}"
        );
        assert_eq!(store, before, "a refused message applies no record");

        let mut short_clock = combined_message(&[(5, 5), (7, 7)]);
        short_clock.clock.pop();
        let err = store
            .ingest_ssr(&short_clock, ssr_week())
            .expect_err("unequal orbit and clock record counts");
        assert!(
            err.to_string()
                .contains("2 orbit records and 1 clock records"),
            "{err}"
        );
        assert_eq!(store, before, "a refused message applies no record");

        let mut short_orbit = combined_message(&[(5, 5), (7, 7)]);
        short_orbit.orbit.pop();
        let err = store
            .ingest_ssr(&short_orbit, ssr_week())
            .expect_err("unequal orbit and clock record counts");
        assert!(
            err.to_string()
                .contains("1 orbit records and 2 clock records"),
            "{err}"
        );
        assert_eq!(store, before, "a refused message applies no record");
    }

    /// A GLONASS SSR epoch is a GLONASS time of day, UTC(SU) + 3 h. It is placed
    /// on the GLONASS day nearest the receiver time and converted to GPS time with
    /// the leap seconds in force, as RTKLIB `adjday_glot` does, across day and GPS
    /// week boundaries and across a leap second.
    #[test]
    fn glonass_ssr_epoch_is_time_of_day_on_nearest_day_in_gps_time() {
        let r01 = GnssSatelliteId::new(GnssSystem::Glonass, 1).unwrap();
        let transmitted = |week: u32, tow: f64, tod: u32| {
            let mut message = orbit_message(GnssSystem::Glonass, 1);
            message.header.epoch_time_s = tod;
            let mut store = SsrCorrectionStore::new();
            store
                .ingest_ssr(
                    &message,
                    GnssWeekTow::new(TimeScale::Gpst, week, tow).unwrap(),
                )
                .unwrap();
            store.orbit(r01).unwrap().transmitted_epoch_j2000_s
        };
        let gps =
            |week: u32, tow: f64| f64::from(week) * SECONDS_PER_WEEK + tow - GPS_EPOCH_TO_J2000_S;

        // GPS - UTC is 18 s from 2017 on. Receiver Saturday 23:59:50 GPST, week 2400:
        // 23:59:32 UTC, Sunday 02:59:32 GLONASS in GPS week 2401. A time of day 2 s
        // earlier lies on that GLONASS day, which is still GPS week 2400.
        assert_eq!(transmitted(2400, 604_790.0, 10_770), gps(2400, 604_788.0));
        // Receiver Sunday 21:00:18 GPST, week 2401: Monday 00:00:00 GLONASS. A time
        // of day 10 s before GLONASS midnight belongs to the previous GLONASS day.
        assert_eq!(transmitted(2401, 75_618.0, 86_390), gps(2401, 75_608.0));
        // Receiver Sunday 00:00:10 GPST, week 2401: 02:59:52 GLONASS. 23:59:55 is the
        // previous GLONASS day, Saturday 20:59:55 UTC, in GPS week 2400.
        assert_eq!(transmitted(2401, 10.0, 86_395), gps(2400, 594_013.0));
        // Receiver Saturday 20:59:50 GPST, week 2400: 23:59:32 GLONASS. 00:00:05 is
        // the next GLONASS day, Saturday 21:00:05 UTC.
        assert_eq!(transmitted(2400, 593_990.0, 5), gps(2400, 594_023.0));

        // Across the 2016-12-31 leap second GPS - UTC went from 17 s to 18 s; GPS
        // week 1930 starts 2017-01-01 00:00:00. GLONASS 03:01:00 on 1 January is
        // 00:01:00 UTC; GLONASS 02:59:00 is 23:59:00 UTC on 31 December.
        assert_eq!(transmitted(1930, 100.0, 10_860), gps(1930, 78.0));
        assert_eq!(transmitted(1930, 100.0, 10_740), gps(1929, 604_757.0));
    }

    /// A non-GLONASS SSR time of week is placed in the week nearest the receiver
    /// time, as RTKLIB `adjweek` does, across a week boundary either way. Native
    /// BeiDou messages are read in GPS time, as RTKLIB reads them; IGS SSR (4076)
    /// messages carry GPS time for every GNSS.
    #[test]
    fn ssr_time_of_week_is_placed_in_the_week_nearest_the_receiver() {
        let transmitted = |system: GnssSystem, number: u16, week: u32, tow: f64, epoch: u32| {
            let sat = GnssSatelliteId::new(system, 5).unwrap();
            let mut message = orbit_message(system, 5);
            message.message_number = number;
            message.header.epoch_time_s = epoch;
            let mut store = SsrCorrectionStore::new();
            store
                .ingest_ssr(
                    &message,
                    GnssWeekTow::new(TimeScale::Gpst, week, tow).unwrap(),
                )
                .unwrap();
            store.orbit(sat).unwrap().transmitted_epoch_j2000_s
        };
        let gps =
            |week: u32, tow: f64| f64::from(week) * SECONDS_PER_WEEK + tow - GPS_EPOCH_TO_J2000_S;
        // Stamped just after the boundary, received just before it: next week.
        assert_eq!(
            transmitted(GnssSystem::Gps, 1057, 2400, 604_795.0, 5),
            gps(2401, 5.0)
        );
        // Stamped just before the boundary, received just after it: previous week.
        assert_eq!(
            transmitted(GnssSystem::Gps, 1057, 2401, 3.0, 604_798),
            gps(2400, 604_798.0)
        );
        // Nearest week of an ordinary epoch is the receiver's own.
        assert_eq!(
            transmitted(GnssSystem::Galileo, 1240, 2401, 100_000.0, 99_990),
            gps(2401, 99_990.0)
        );
        assert_eq!(
            transmitted(GnssSystem::BeiDou, 1258, 2401, 20.0, 604_795),
            gps(2400, 604_795.0)
        );
        assert_eq!(
            transmitted(GnssSystem::BeiDou, 1258, 2401, 100_000.0, 99_986),
            gps(2401, 99_986.0)
        );
        // An IGS SSR BeiDou or GLONASS epoch is GPS time.
        assert_eq!(
            transmitted(
                GnssSystem::BeiDou,
                IGS_SSR_MESSAGE_NUMBER,
                2401,
                100_000.0,
                99_990
            ),
            gps(2401, 99_990.0)
        );
        assert_eq!(
            transmitted(
                GnssSystem::Glonass,
                IGS_SSR_MESSAGE_NUMBER,
                2401,
                100_000.0,
                99_990
            ),
            gps(2401, 99_990.0)
        );
    }

    /// Real IGS `SSRA03IGS0` BeiDou 1261 frames are read in GPS time, as RTKLIB
    /// reads them. Each sits 4 s before the epoch of the batch it is sent in,
    /// which the batch's GPS frames state and its GLONASS time of day confirms.
    #[test]
    fn real_beidou_ssr_epochs_are_read_in_gps_time() {
        let bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/SSRA03IGS0_2026188140760_3epoch.rtcm3"
        ));
        let week = GnssWeekTow::new(TimeScale::Gpst, 2426, 223_690.0).unwrap();
        let mut store = SsrCorrectionStore::new();
        let mut assembler = SsrStreamAssembler::new();
        for decoded in assembler.push(bytes) {
            store
                .ingest(&decoded.expect("decode real SSR frame"), week)
                .expect("ingest real SSR frame");
        }
        let gps = |tow: f64| 2426.0 * SECONDS_PER_WEEK + tow - GPS_EPOCH_TO_J2000_S;
        let epoch_of = |system: GnssSystem| {
            store
                .corrections
                .iter()
                .find(|(sat, entry)| sat.system == system && entry.orbit.is_some())
                .and_then(|(_, entry)| entry.orbit)
                .map(|orbit| orbit.transmitted_epoch_j2000_s)
                .expect("orbit correction for the system")
        };
        // The last complete batch is stamped 223680 s by its GPS 1060 frames and
        // 61662 s GLONASS time of day, the same instant; its BeiDou 1261 frames
        // hold 223676 s.
        assert_eq!(epoch_of(GnssSystem::Gps), gps(223_680.0));
        assert_eq!(epoch_of(GnssSystem::Glonass), gps(223_680.0));
        assert_eq!(epoch_of(GnssSystem::BeiDou), gps(223_676.0));
    }

    /// An RTCM SSR bias applies while its age from the transmitted epoch is
    /// within the store staleness cap, whatever the update interval: at index
    /// 15 (10800 s) and index 9 (300 s) it applies from the transmitted epoch
    /// on, not from half an interval later.
    #[test]
    fn rtcm_bias_age_runs_from_transmitted_epoch_at_any_update_interval() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let empty = orbit_message(GnssSystem::Gps, 1);
        let t0 = ssr_epoch_j2000_s(GnssSystem::Gps, 1059, ssr_week(), 100_000).unwrap();
        for index in [15u8, 9] {
            let mut code_header = header(SsrKind::CodeBias);
            code_header.update_interval = index;
            let code = SsrMessage {
                message_number: 1059,
                kind: SsrKind::CodeBias,
                header: code_header,
                orbit: Vec::new(),
                code_bias: vec![crate::rtcm::SsrCodeBiasRecord {
                    satellite_id: 1,
                    biases: vec![(0, 100)],
                }],
                ..empty.clone()
            };
            let mut phase_header = header(SsrKind::PhaseBias);
            phase_header.update_interval = index;
            let phase = SsrMessage {
                message_number: 1265,
                kind: SsrKind::PhaseBias,
                header: phase_header,
                orbit: Vec::new(),
                phase_bias: vec![SsrPhaseBiasRecord {
                    satellite_id: 1,
                    yaw_angle: 0,
                    yaw_rate: 0,
                    biases: vec![SsrPhaseBiasSignal {
                        signal_id: 0,
                        integer_indicator: 0,
                        wide_lane_integer_indicator: 0,
                        discontinuity_counter: 0,
                        bias: 1000,
                    }],
                }],
                ..empty.clone()
            };
            let mut store = SsrCorrectionStore::new();
            store.ingest_ssr(&code, ssr_week()).unwrap();
            store.ingest_ssr(&phase, ssr_week()).unwrap();
            for dt in [0.0, 1.0, 90.0, -90.0] {
                let code_q = store.query_code_bias(sat, rtcm_sig(sat, 0), t0 + dt);
                assert_eq!(
                    code_q.status,
                    SsrBiasStatus::Available,
                    "index {index}, {dt} s"
                );
                assert_eq!(code_q.ref_epoch_j2000_s, Some(t0));
                let phase_q = store.query_phase_bias(sat, rtcm_sig(sat, 0), t0 + dt, None);
                assert_eq!(
                    phase_q.status,
                    SsrBiasStatus::Available,
                    "index {index}, {dt} s"
                );
            }
            assert_eq!(
                store
                    .query_code_bias(sat, rtcm_sig(sat, 0), t0 + 90.5)
                    .status,
                SsrBiasStatus::Expired
            );
            assert_eq!(
                store
                    .query_phase_bias(sat, rtcm_sig(sat, 0), t0 + 90.5, None)
                    .status,
                SsrBiasStatus::Expired
            );
            assert_eq!(
                store
                    .query_code_bias(sat, rtcm_sig(sat, 0), t0 - 90.5)
                    .status,
                SsrBiasStatus::NotYetValid
            );
        }
    }

    /// A HAS orbit or clock record older than one already accepted for the
    /// satellite is refused even after an RTCM correction has replaced the HAS
    /// one: HAS20 -> RTCM25 -> HAS10 leaves the RTCM correction in place, and a
    /// delayed do-not-use at 10 does not exclude the satellite.
    #[test]
    fn has_orbit_and_clock_watermarks_hold_across_an_intervening_rtcm_correction() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let t0_tow = 200_000.0;
        let at = |offset: u32| {
            GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + f64::from(offset)).unwrap()
        };
        let toh = |offset: u32| ((t0_tow as u32 % 3600) + offset) as u16;
        let epoch = |offset: u32| has_mt1_reference_j2000_s(at(offset), toh(offset)).unwrap();
        let has = |offset: u32, radial_m: Option<f64>, clock_m: Option<f64>, do_not_use: bool| {
            let mut message = has_message(
                vec![HasOrbitCorrection {
                    sat,
                    nav_message: 0,
                    iode: 7,
                    radial_m,
                    along_m: Some(0.25),
                    cross_m: Some(0.125),
                }],
                vec![HasClockCorrection {
                    sat,
                    nav_message: 0,
                    correction_m: clock_m,
                    do_not_use,
                }],
                Vec::new(),
                Vec::new(),
            );
            message.header.toh_s = toh(offset);
            message
        };

        let mut store = SsrCorrectionStore::new();
        store
            .ingest_has_mt1(&has(20, Some(0.5), Some(-0.25), false), at(20))
            .unwrap();
        assert_eq!(
            store.orbit(sat).unwrap().solution.source,
            SsrSource::GalileoHas
        );

        let mut rtcm = combined_message(&[(1, 1)]);
        rtcm.header.epoch_time_s = (t0_tow + 25.0) as u32;
        store
            .ingest_ssr(
                &rtcm,
                GnssWeekTow::new(TimeScale::Gpst, 1042, t0_tow + 25.0).unwrap(),
            )
            .unwrap();
        let rtcm_state = store.clone();
        assert_eq!(
            store.orbit(sat).unwrap().solution.source,
            SsrSource::RtcmSsr
        );
        assert_eq!(
            store.clock(sat).unwrap().solution.source,
            SsrSource::RtcmSsr
        );

        // Usable, unavailable and do-not-use HAS records at 10 are all older than
        // the HAS records already accepted at 20.
        for (label, message) in [
            ("usable", has(10, Some(0.5), Some(-0.25), false)),
            ("unavailable", has(10, None, None, false)),
            ("do-not-use", has(10, Some(0.5), None, true)),
        ] {
            store.ingest_has_mt1(&message, at(10)).unwrap();
            assert_eq!(store, rtcm_state, "{label} HAS10 changes nothing");
            assert!(
                !store.is_satellite_excluded(sat, epoch(10) + 1.0),
                "{label}: no exclusion from a record older than the watermark"
            );
        }

        // A newer do-not-use is accepted and excludes the satellite.
        store
            .ingest_has_mt1(&has(30, Some(0.5), None, true), at(30))
            .unwrap();
        assert!(store.is_satellite_excluded(sat, epoch(30)));
    }

    /// A Galileo HAS phase arc breaks only on a source switch or a PDI change
    /// (HAS SIS ICD 5.2.6.1, 7.4): a new mask ID or IOD set ID keeps the arc,
    /// and a token taken before them still continues it.
    #[test]
    fn has_phase_arc_follows_source_and_pdi_not_mask_or_iod_set() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let t0_tow = 200_000.0;
        let at = |offset: u32| {
            GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + f64::from(offset)).unwrap()
        };
        let toh = |offset: u32| ((t0_tow as u32 % 3600) + offset) as u16;
        let epoch = |offset: u32| has_mt1_reference_j2000_s(at(offset), toh(offset)).unwrap();
        let phase = |offset: u32, mask_id: u8, iod_set_id: u8, pdi: u8| {
            let mut message = has_message(
                Vec::new(),
                Vec::new(),
                Vec::new(),
                vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: Some(0.25),
                    discontinuity_indicator: pdi,
                }],
            );
            message.header.toh_s = toh(offset);
            message.header.mask_id = mask_id;
            message.header.iod_set_id = iod_set_id;
            message
        };

        let mut store = SsrCorrectionStore::new();
        store.ingest_has_mt1(&phase(0, 1, 1, 0), at(0)).unwrap();
        let token0 = store
            .query_phase_bias(sat, has_sig(sat, 0), epoch(0), None)
            .continuity_token
            .unwrap();

        store.ingest_has_mt1(&phase(30, 2, 5, 0), at(30)).unwrap();
        let q30 = store.query_phase_bias(sat, has_sig(sat, 0), epoch(30), Some(token0));
        assert_eq!(q30.status, SsrBiasStatus::Available);
        assert_eq!(
            q30.discontinuity_details,
            Some(SsrDiscontinuityDetails::Continuous)
        );
        let token30 = q30.continuity_token.unwrap();
        assert_eq!(token30.generation(), token0.generation());
        assert_eq!((token30.provider_id(), token30.solution_id()), (2, 5));

        store.ingest_has_mt1(&phase(60, 2, 6, 1), at(60)).unwrap();
        let q60 = store.query_phase_bias(sat, has_sig(sat, 0), epoch(60), Some(token0));
        assert_eq!(q60.status, SsrBiasStatus::PhaseDiscontinuityNeedsReset);
        assert_eq!(
            q60.discontinuity_details,
            Some(SsrDiscontinuityDetails::HasPdiChanged {
                previous: 0,
                current: 1
            })
        );
    }

    /// A code or phase bias signal listed twice for one satellite in one RTCM
    /// message keeps its last value, as RTKLIB overwrites it; the earlier
    /// phase value does not register a phase break.
    #[test]
    fn rtcm_duplicate_bias_signal_keeps_the_last_value() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let empty = orbit_message(GnssSystem::Gps, 1);
        let code = SsrMessage {
            message_number: 1059,
            kind: SsrKind::CodeBias,
            header: header(SsrKind::CodeBias),
            orbit: Vec::new(),
            code_bias: vec![
                crate::rtcm::SsrCodeBiasRecord {
                    satellite_id: 1,
                    biases: vec![(0, 100), (9, 300)],
                },
                crate::rtcm::SsrCodeBiasRecord {
                    satellite_id: 1,
                    biases: vec![(0, 200)],
                },
            ],
            ..empty.clone()
        };
        let signal = |counter: u8, bias: i32| SsrPhaseBiasSignal {
            signal_id: 0,
            integer_indicator: 0,
            wide_lane_integer_indicator: 0,
            discontinuity_counter: counter,
            bias,
        };
        let phase = SsrMessage {
            message_number: 1265,
            kind: SsrKind::PhaseBias,
            header: header(SsrKind::PhaseBias),
            orbit: Vec::new(),
            phase_bias: vec![SsrPhaseBiasRecord {
                satellite_id: 1,
                yaw_angle: 0,
                yaw_rate: 0,
                biases: vec![signal(1, 1000), signal(2, 2000)],
            }],
            ..empty
        };
        let mut store = SsrCorrectionStore::new();
        store.ingest_ssr(&code, ssr_week()).unwrap();
        store.ingest_ssr(&phase, ssr_week()).unwrap();
        assert_eq!(
            store.code_bias(sat, rtcm_sig(sat, 0)),
            Some(f64::from(200) * RTCM_SSR_CODE_BIAS_SCALE_M)
        );
        assert_eq!(
            store.code_bias(sat, rtcm_sig(sat, 9)),
            Some(f64::from(300) * RTCM_SSR_CODE_BIAS_SCALE_M)
        );
        assert_eq!(
            store.phase_bias(sat, rtcm_sig(sat, 0)),
            Some(f64::from(2000) * RTCM_SSR_PHASE_BIAS_SCALE_M)
        );
        let transmitted = ssr_epoch_j2000_s(GnssSystem::Gps, 1059, ssr_week(), 100_000).unwrap();
        let q = store.query_phase_bias(sat, rtcm_sig(sat, 0), transmitted, None);
        assert_eq!(q.status, SsrBiasStatus::Available);
        assert_eq!(
            q.discontinuity_details,
            Some(SsrDiscontinuityDetails::InitialTokenEstablished)
        );
        let token = q.continuity_token.unwrap();
        assert_eq!((token.raw_indicator(), token.generation()), (2, 0));
        assert_eq!(q.ref_epoch_j2000_s, Some(transmitted));
    }

    /// An RTCM high-rate clock applies while its age from its transmitted epoch
    /// is under 10 s (RTKLIB `MAXAGESSR_HRCLK`), and attaches to a combined
    /// orbit/clock message only when solution and IOD SSR match, as it does to a
    /// clock message.
    #[test]
    fn rtcm_high_rate_clock_age_limit_and_combined_attachment() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let iode = broadcast
            .select_record_at(sat, t)
            .expect("broadcast record at SSR epoch")
            .issue_of_data
            .expect("broadcast issue")
            .issue;
        let week = GnssWeekTow::new(TimeScale::Gpst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("valid SSR week");
        let mut combined = combined_message(&[(sat.prn, sat.prn)]);
        combined.header.epoch_time_s = REAL_SSR_EPOCH_TOW_S as u32;
        combined.header.update_interval = 0;
        combined.orbit[0].iode = iode;
        let mut high_rate = combined_message(&[(sat.prn, sat.prn)]);
        high_rate.message_number = 1062;
        high_rate.kind = SsrKind::HighRateClock;
        high_rate.header = combined.header.clone();
        high_rate.header.satellite_reference_datum = None;
        high_rate.orbit = Vec::new();
        high_rate.clock[0].c0 = 500;

        let mut plain = SsrCorrectionStore::new();
        plain.ingest_ssr(&combined, week).unwrap();
        let mut with_high_rate = SsrCorrectionStore::new();
        with_high_rate.ingest_ssr(&high_rate, week).unwrap();
        with_high_rate.ingest_ssr(&combined, week).unwrap();
        assert!(with_high_rate.clock(sat).unwrap().high_rate.is_some());

        let clock_at = |store: &SsrCorrectionStore, dt: f64| {
            SsrCorrectedEphemeris::new(&broadcast, store)
                .corrected_state(sat, t + dt)
                .expect("fresh RTCM correction")
                .1
        };
        // The high-rate clock adds to the clock correction, and the correction adds to
        // the clock (RTKLIB `satpos_ssr`: `dclk += hrclk`, `dts += dclk / CLIGHT`).
        let applied = clock_at(&with_high_rate, 9.5) - clock_at(&plain, 9.5);
        assert!((applied - 0.05 / C_M_S).abs() < 1.0e-18, "{applied}");
        assert_eq!(
            clock_at(&with_high_rate, 10.0).to_bits(),
            clock_at(&plain, 10.0).to_bits(),
            "a high-rate clock 10 s old is not applied"
        );

        // A pending high-rate clock from another IOD SSR is not attached.
        let mut other_iod = high_rate.clone();
        other_iod.header.iod_ssr = combined.header.iod_ssr.wrapping_add(1);
        let mut mismatched = SsrCorrectionStore::new();
        mismatched.ingest_ssr(&other_iod, week).unwrap();
        mismatched.ingest_ssr(&combined, week).unwrap();
        assert!(mismatched.clock(sat).unwrap().high_rate.is_none());
    }

    #[test]
    fn reference_point_tag_round_trips_through_store_ingest() {
        let message = SsrMessage {
            message_number: 1057,
            system: GnssSystem::Gps,
            kind: SsrKind::Orbit,
            header: header(SsrKind::Orbit),
            orbit: vec![SsrOrbitRecord {
                satellite_id: 1,
                iode: 42,
                delta_radial: 0,
                delta_along: 0,
                delta_cross: 0,
                dot_delta_radial: 0,
                dot_delta_along: 0,
                dot_delta_cross: 0,
            }],
            clock: Vec::new(),
            code_bias: Vec::new(),
            phase_bias: Vec::<SsrPhaseBiasRecord>::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };
        let mut store =
            SsrCorrectionStore::new().with_reference_point(SsrReferencePoint::igs_ssr_default());
        let week = GnssWeekTow::new(TimeScale::Gpst, 2_400, 100_000.0).unwrap();
        store.ingest_ssr(&message, week).unwrap();
        let tag = store.reference_point().tag();
        assert_eq!(
            SsrReferencePoint::from_tag(tag),
            Some(SsrReferencePoint::CenterOfMass)
        );
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        assert_eq!(
            store.clone().orbit(sat).unwrap().reference_point,
            SsrReferencePoint::CenterOfMass
        );
    }

    /// The rate terms are referenced half an update interval after the
    /// transmitted epoch, except for index 0, which uses the transmitted epoch
    /// itself (IGS SSR v1.00). RTKLIB `satpos_ssr` shifts index 0 by 0.5 s too.
    #[test]
    fn update_interval_moves_rate_reference_by_half_except_index_zero() {
        let week = GnssWeekTow::new(TimeScale::Gpst, 2_400, 0.0).unwrap();
        let transmitted = ssr_epoch_j2000_s(GnssSystem::Gps, 1057, week, 100_000).unwrap();
        let expected = f64::from(week.week) * SECONDS_PER_WEEK + 100_000.0 - GPS_EPOCH_TO_J2000_S;
        assert_eq!(transmitted.to_bits(), expected.to_bits());

        let sat = GnssSatelliteId::new(GnssSystem::Gps, 5).unwrap();
        for (index, half_s) in [(0u8, 0.0), (1, 1.0), (3, 5.0), (15, 5400.0)] {
            let mut message = orbit_message(GnssSystem::Gps, 5);
            message.header.update_interval = index;
            let mut store = SsrCorrectionStore::new();
            store.ingest_ssr(&message, week).unwrap();
            let orbit = store.orbit(sat).unwrap();
            assert_eq!(
                orbit.transmitted_epoch_j2000_s.to_bits(),
                expected.to_bits()
            );
            assert_eq!(
                orbit.ref_epoch_j2000_s.to_bits(),
                (expected + half_s).to_bits(),
                "index {index}"
            );
        }
    }

    #[test]
    fn satellite_pco_projection_matches_hand_geometry() {
        let sat_position_ecef_m = [1.0, 0.0, 0.0];
        let sun_ecef_m = [1.0, 1.0, 0.0];
        let pco_body_m = [0.25, 0.5, 0.75];
        let shift = satellite_body_pco_to_ecef(pco_body_m, sat_position_ecef_m, sun_ecef_m)
            .expect("non-degenerate satellite-Sun axes");
        assert_eq!(
            shift.map(f64::to_bits),
            [
                (-0.75_f64).to_bits(),
                0.25_f64.to_bits(),
                (-0.5_f64).to_bits()
            ]
        );
        let los_unit = [0.0, 0.0, 1.0];
        assert_eq!(dot3(shift, los_unit).to_bits(), (-0.5_f64).to_bits());
    }

    #[test]
    fn satellite_pco_projection_declines_near_degenerate_axes() {
        assert!(satellite_body_pco_to_ecef(
            [0.25, 0.5, 0.75],
            [1.0, 0.0, 0.0],
            [0.0, 1.0e-15, 0.0],
        )
        .is_none());
    }

    #[test]
    fn high_rate_clock_is_additive_when_identity_matches() {
        let mut store = SsrCorrectionStore::new();
        let week = GnssWeekTow::new(TimeScale::Gpst, 2_400, 100_000.0).unwrap();
        let low = SsrMessage {
            message_number: 1058,
            system: GnssSystem::Gps,
            kind: SsrKind::Clock,
            header: header(SsrKind::Clock),
            orbit: Vec::new(),
            clock: vec![SsrClockRecord {
                satellite_id: 1,
                c0: 1_000,
                c1: 0,
                c2: 0,
            }],
            code_bias: Vec::new(),
            phase_bias: Vec::<SsrPhaseBiasRecord>::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };
        let high = SsrMessage {
            message_number: 1062,
            system: GnssSystem::Gps,
            kind: SsrKind::HighRateClock,
            header: header(SsrKind::HighRateClock),
            orbit: Vec::new(),
            clock: vec![SsrClockRecord {
                satellite_id: 1,
                c0: 500,
                c1: 0,
                c2: 0,
            }],
            code_bias: Vec::new(),
            phase_bias: vec![SsrPhaseBiasRecord {
                satellite_id: 1,
                yaw_angle: 0,
                yaw_rate: 0,
                biases: vec![SsrPhaseBiasSignal {
                    signal_id: 1,
                    integer_indicator: 0,
                    wide_lane_integer_indicator: 0,
                    discontinuity_counter: 0,
                    bias: 0,
                }],
            }],
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };
        store.ingest_ssr(&high, week).unwrap();
        store.ingest_ssr(&low, week).unwrap();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        assert_eq!(
            store.clock(sat).unwrap().high_rate.unwrap().c0_m.to_bits(),
            0.05_f64.to_bits()
        );
    }

    #[test]
    fn rtcm_binary_orbit_clock_correction_matches_analytic_rac_application() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let record = broadcast
            .select_record_at(sat, t)
            .expect("broadcast record at SSR epoch");
        let iode = record.issue_of_data.expect("broadcast issue").issue;

        let wanted_rac_m = [2.0, -4.0, 1.2];
        let clock_correction_m = 0.5;
        let message = Message::Ssr(SsrMessage {
            message_number: 1060,
            system: GnssSystem::Gps,
            kind: SsrKind::CombinedOrbitClock,
            header: SsrHeader {
                epoch_time_s: REAL_SSR_EPOCH_TOW_S as u32,
                update_interval: 0,
                multiple_message: false,
                iod_ssr: 3,
                provider_id: 9,
                solution_id: 1,
                satellite_reference_datum: Some(false),
                dispersive_bias_consistency: None,
                mw_consistency: None,
                satellite_count: 1,
            },
            orbit: vec![SsrOrbitRecord {
                satellite_id: sat.prn,
                iode,
                delta_radial: -20_000,
                delta_along: 10_000,
                delta_cross: -3_000,
                dot_delta_radial: 0,
                dot_delta_along: 0,
                dot_delta_cross: 0,
            }],
            clock: vec![SsrClockRecord {
                satellite_id: sat.prn,
                c0: 5_000,
                c1: 0,
                c2: 0,
            }],
            code_bias: Vec::new(),
            phase_bias: Vec::<SsrPhaseBiasRecord>::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        });
        let frame = message.to_frame().expect("frame RTCM SSR");
        let mut assembler = SsrStreamAssembler::new();
        let mut store = SsrCorrectionStore::new();
        let week = GnssWeekTow::new(TimeScale::Gpst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("valid SSR week");
        for decoded in assembler.push(&frame) {
            let decoded = decoded.expect("decode RTCM SSR frame");
            store.ingest(&decoded, week).expect("ingest RTCM SSR frame");
        }

        let source = SsrCorrectedEphemeris::new(&broadcast, &store);
        let (corrected_position, corrected_clock) =
            source.corrected_state(sat, t).expect("corrected state");
        let (broadcast_position, _) = broadcast
            .position_clock_at_j2000_s(sat, t)
            .expect("broadcast state");
        let velocity = finite_difference_broadcast_velocity(&broadcast, sat, t);
        let (er, ea, ec) = analytic_velocity_aligned_basis(broadcast_position, velocity);
        let expected_position = [
            broadcast_position[0]
                + wanted_rac_m[0] * er[0]
                + wanted_rac_m[1] * ea[0]
                + wanted_rac_m[2] * ec[0],
            broadcast_position[1]
                + wanted_rac_m[0] * er[1]
                + wanted_rac_m[1] * ea[1]
                + wanted_rac_m[2] * ec[1],
            broadcast_position[2]
                + wanted_rac_m[0] * er[2]
                + wanted_rac_m[1] * ea[2]
                + wanted_rac_m[2] * ec[2],
        ];
        assert_vector_close(corrected_position, expected_position, 2.0e-9);
        // The clock correction adds to the clock, as RTKLIB `satpos_ssr` and IGS SSR add
        // it. This test once asserted the broadcast clock minus C0 / c: the subtraction was
        // the wrong sign, and that broadcast clock carried the broadcast group delay and
        // carries the `F·e·√A·sin E` relativistic term, where the SSR-corrected clock has
        // neither and has `-2 r·v / c²` instead.
        let dclock_m = store.clock(sat).expect("stored clock").c0_m;
        assert_eq!(dclock_m.to_bits(), (5_000.0_f64 * 1.0e-4).to_bits());
        assert_eq!(
            corrected_clock.to_bits(),
            satpos_ssr_clock_s(&broadcast, sat, REAL_SSR_EPOCH_TOW_S, dclock_m).to_bits()
        );
        let uncorrected = satpos_ssr_clock_s(&broadcast, sat, REAL_SSR_EPOCH_TOW_S, 0.0);
        assert!(
            (corrected_clock - (uncorrected + clock_correction_m / C_M_S)).abs() < 1.0e-18,
            "a positive C0 makes the satellite clock later"
        );
    }

    #[test]
    fn rtcm_iode_mismatch_declines_or_falls_back_without_applying_correction() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let record = broadcast
            .select_record_at(sat, t)
            .expect("broadcast record at SSR epoch");
        let stale_iode = (record.issue_of_data.expect("broadcast issue").issue + 1) & 0xff;
        let message = Message::Ssr(SsrMessage {
            message_number: 1060,
            system: GnssSystem::Gps,
            kind: SsrKind::CombinedOrbitClock,
            header: SsrHeader {
                epoch_time_s: REAL_SSR_EPOCH_TOW_S as u32,
                update_interval: 0,
                multiple_message: false,
                iod_ssr: 4,
                provider_id: 9,
                solution_id: 1,
                satellite_reference_datum: Some(false),
                dispersive_bias_consistency: None,
                mw_consistency: None,
                satellite_count: 1,
            },
            orbit: vec![SsrOrbitRecord {
                satellite_id: sat.prn,
                iode: stale_iode,
                delta_radial: -20_000,
                delta_along: 0,
                delta_cross: 0,
                dot_delta_radial: 0,
                dot_delta_along: 0,
                dot_delta_cross: 0,
            }],
            clock: vec![SsrClockRecord {
                satellite_id: sat.prn,
                c0: 5_000,
                c1: 0,
                c2: 0,
            }],
            code_bias: Vec::new(),
            phase_bias: Vec::<SsrPhaseBiasRecord>::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        });
        let mut store = SsrCorrectionStore::new();
        let week = GnssWeekTow::new(TimeScale::Gpst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("valid SSR week");
        let mut assembler = SsrStreamAssembler::new();
        for decoded in assembler.push(&message.to_frame().expect("frame RTCM SSR")) {
            store
                .ingest(&decoded.expect("decode RTCM SSR frame"), week)
                .expect("ingest SSR");
        }

        let strict = SsrCorrectedEphemeris::new(&broadcast, &store);
        assert!(strict.corrected_state(sat, t).is_none());

        let fallback =
            SsrCorrectedEphemeris::new(&broadcast, &store).with_fallback(SsrFallbackPolicy {
                on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                regional: RegionalPolicy::DeclineRegional,
            });
        let got = fallback
            .corrected_state(sat, t)
            .expect("broadcast fallback");
        let expected = broadcast
            .position_clock_at_j2000_s(sat, t)
            .expect("broadcast state");
        assert_eq!(got.0.map(f64::to_bits), expected.0.map(f64::to_bits));
        assert_eq!(got.1.to_bits(), expected.1.to_bits());
    }

    /// An RTCM SSR orbit and clock correction applies while its age from the
    /// transmitted epoch is at most min(store cap, 90 s), RTKLIB `MAXAGESSR`, on
    /// either side. The 1 s update interval (index 0) bounds nothing.
    #[test]
    fn rtcm_orbit_clock_age_runs_from_transmitted_epoch_not_update_interval() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let record = broadcast
            .select_record_at(sat, t)
            .expect("broadcast record at SSR epoch");
        let message = Message::Ssr(SsrMessage {
            message_number: 1060,
            system: GnssSystem::Gps,
            kind: SsrKind::CombinedOrbitClock,
            header: SsrHeader {
                epoch_time_s: REAL_SSR_EPOCH_TOW_S as u32,
                update_interval: 0,
                multiple_message: false,
                iod_ssr: 6,
                provider_id: 9,
                solution_id: 1,
                satellite_reference_datum: Some(false),
                dispersive_bias_consistency: None,
                mw_consistency: None,
                satellite_count: 1,
            },
            orbit: vec![SsrOrbitRecord {
                satellite_id: sat.prn,
                iode: record.issue_of_data.expect("broadcast issue").issue,
                delta_radial: 0,
                delta_along: 0,
                delta_cross: 0,
                dot_delta_radial: 0,
                dot_delta_along: 0,
                dot_delta_cross: 0,
            }],
            clock: vec![SsrClockRecord {
                satellite_id: sat.prn,
                c0: 0,
                c1: 0,
                c2: 0,
            }],
            code_bias: Vec::new(),
            phase_bias: Vec::<SsrPhaseBiasRecord>::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        });
        let week = GnssWeekTow::new(TimeScale::Gpst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("valid SSR week");
        let store_with_cap = |cap_s: f64| {
            let mut store =
                SsrCorrectionStore::new().with_staleness(StalenessPolicy::seconds(cap_s));
            let mut assembler = SsrStreamAssembler::new();
            for decoded in assembler.push(&message.to_frame().expect("frame RTCM SSR")) {
                store
                    .ingest(&decoded.expect("decode RTCM SSR frame"), week)
                    .expect("ingest SSR");
            }
            store
        };

        let store = store_with_cap(90.0);
        let orbit = store.orbit(sat).expect("RTCM orbit");
        assert_eq!(orbit.transmitted_epoch_j2000_s, t);
        assert_eq!(orbit.ref_epoch_j2000_s, t);
        let strict = SsrCorrectedEphemeris::new(&broadcast, &store);
        for dt in [1.25, 60.0, 90.0, -90.0] {
            assert!(
                strict.corrected_state(sat, t + dt).is_some(),
                "age {dt} s is within 90 s"
            );
        }
        for dt in [90.25, -90.25] {
            assert!(
                strict.corrected_state(sat, t + dt).is_none(),
                "age {dt} s is past 90 s"
            );
        }

        // A store cap below 90 s binds first; one above it does not extend the limit.
        let tight = store_with_cap(30.0);
        let tight_source = SsrCorrectedEphemeris::new(&broadcast, &tight);
        assert!(tight_source.corrected_state(sat, t + 30.0).is_some());
        assert!(tight_source.corrected_state(sat, t + 30.25).is_none());
        let loose = store_with_cap(300.0);
        assert!(SsrCorrectedEphemeris::new(&broadcast, &loose)
            .corrected_state(sat, t + 90.25)
            .is_none());

        let fallback =
            SsrCorrectedEphemeris::new(&broadcast, &store).with_fallback(SsrFallbackPolicy {
                on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                regional: RegionalPolicy::DeclineRegional,
            });
        let got = fallback
            .corrected_state(sat, t + 90.25)
            .expect("broadcast fallback past the age limit");
        let expected = broadcast
            .position_clock_at_j2000_s(sat, t + 90.25)
            .expect("broadcast state");
        assert_eq!(got.0.map(f64::to_bits), expected.0.map(f64::to_bits));
        assert_eq!(got.1.to_bits(), expected.1.to_bits());
    }

    #[test]
    fn rtcm_binary_phase_bias_ingests_for_ppp_bias_lookup() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let message = Message::Ssr(SsrMessage {
            message_number: 1265,
            system: GnssSystem::Gps,
            kind: SsrKind::PhaseBias,
            header: SsrHeader {
                epoch_time_s: REAL_SSR_EPOCH_TOW_S as u32,
                update_interval: 0,
                multiple_message: false,
                iod_ssr: 7,
                provider_id: 9,
                solution_id: 1,
                satellite_reference_datum: None,
                dispersive_bias_consistency: Some(true),
                mw_consistency: Some(false),
                satellite_count: 1,
            },
            orbit: Vec::new(),
            clock: Vec::new(),
            code_bias: Vec::new(),
            phase_bias: vec![SsrPhaseBiasRecord {
                satellite_id: sat.prn,
                yaw_angle: 127,
                yaw_rate: -12,
                biases: vec![
                    SsrPhaseBiasSignal {
                        signal_id: 0,
                        integer_indicator: 1,
                        wide_lane_integer_indicator: 2,
                        discontinuity_counter: 3,
                        bias: 1_250,
                    },
                    SsrPhaseBiasSignal {
                        signal_id: 9,
                        integer_indicator: 0,
                        wide_lane_integer_indicator: 1,
                        discontinuity_counter: 4,
                        bias: -2_500,
                    },
                ],
            }],
            ura: Vec::new(),
            padding_bits: Vec::new(),
        });
        let mut store = SsrCorrectionStore::new();
        let week = GnssWeekTow::new(TimeScale::Gpst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("valid SSR week");
        let mut assembler = SsrStreamAssembler::new();
        for decoded in assembler.push(&message.to_frame().expect("frame RTCM phase bias")) {
            store
                .ingest(&decoded.expect("decode RTCM phase-bias frame"), week)
                .expect("ingest SSR phase bias");
        }

        assert_eq!(
            store.phase_bias(sat, rtcm_sig(sat, 0)).unwrap().to_bits(),
            0.125_f64.to_bits()
        );
        assert_eq!(
            store.phase_bias(sat, rtcm_sig(sat, 9)).unwrap().to_bits(),
            (-0.25_f64).to_bits()
        );
    }

    /// A HAS MT1 message holding the given blocks, with no inline mask; ingestion
    /// reads the records the blocks hold and does not consult a mask.
    fn has_message(
        orbit: Vec<HasOrbitCorrection>,
        clock: Vec<HasClockCorrection>,
        code_bias: Vec<HasCodeBias>,
        phase_bias: Vec<HasPhaseBias>,
    ) -> HasMt1Message {
        HasMt1Message {
            header: HasMt1Header {
                toh_s: 10,
                mask: false,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: None,
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: orbit,
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: clock,
            }),
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: code_bias,
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: phase_bias,
            }),
            padding_bits: Vec::new(),
        }
    }

    fn has_reception() -> GnssWeekTow {
        GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, 3_620.0).expect("GST reception")
    }

    /// A store holding a full HAS correction set for G01 and G02: orbit, clock,
    /// and code and phase biases on signals 0 (L1 C/A) and 9 (L2).
    fn store_with_has_corrections(
        g01: GnssSatelliteId,
        g02: GnssSatelliteId,
    ) -> SsrCorrectionStore {
        let mut orbit = Vec::new();
        let mut clock = Vec::new();
        let mut code_bias = Vec::new();
        let mut phase_bias = Vec::new();
        for sat in [g01, g02] {
            orbit.push(HasOrbitCorrection {
                sat,
                nav_message: 0,
                iode: 7,
                radial_m: Some(0.5),
                along_m: Some(-0.25),
                cross_m: Some(0.125),
            });
            clock.push(HasClockCorrection {
                sat,
                nav_message: 0,
                correction_m: Some(-0.75),
                do_not_use: false,
            });
            for signal_id in [0, 9] {
                code_bias.push(HasCodeBias {
                    sat,
                    signal_id,
                    bias_m: Some(0.24),
                });
                phase_bias.push(HasPhaseBias {
                    sat,
                    signal_id,
                    bias_cycles: Some(1.25),
                    discontinuity_indicator: 0,
                });
            }
        }
        let mut store = SsrCorrectionStore::new();
        store
            .ingest_has_mt1(
                &has_message(orbit, clock, code_bias, phase_bias),
                has_reception(),
            )
            .expect("ingest full HAS correction set");
        for sat in [g01, g02] {
            assert!(store.orbit(sat).is_some() && store.clock(sat).is_some());
            for signal in [0, 9] {
                assert!(store.code_bias(sat, has_sig(sat, signal)).is_some());
                assert!(store.phase_bias(sat, has_sig(sat, signal)).is_some());
            }
        }
        store
    }

    /// The same message as `message`, sent 30 s later: TOH 40 s, received at
    /// TOW 3650 s, so its reference epoch follows the seeded corrections'.
    fn has_later(mut message: HasMt1Message) -> (HasMt1Message, GnssWeekTow) {
        message.header.toh_s = 40;
        let reception =
            GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, 3_650.0).expect("GST reception");
        (message, reception)
    }

    /// A later HAS message that marks a satellite's correction unavailable, or
    /// the satellite do-not-use, withdraws the stored HAS correction instead of
    /// leaving the earlier one (possibly for another IOD) to be applied. Other
    /// satellites and other signals keep theirs, and no zero is stored. An
    /// unavailable record at the same reference epoch as a stored usable one
    /// does not withdraw it: the usable record for that epoch is kept.
    #[test]
    fn has_unavailable_and_do_not_use_records_remove_stored_corrections() {
        let g01 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let g02 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let mut store = store_with_has_corrections(g01, g02);
        let g02_before = store.corrections.get(&g02).cloned();

        let (later, later_reception) = has_later(has_message(
            vec![HasOrbitCorrection {
                sat: g01,
                nav_message: 0,
                iode: 8,
                radial_m: Some(0.5),
                along_m: None,
                cross_m: Some(0.125),
            }],
            vec![HasClockCorrection {
                sat: g01,
                nav_message: 0,
                correction_m: None,
                do_not_use: true,
            }],
            vec![HasCodeBias {
                sat: g01,
                signal_id: 0,
                bias_m: None,
            }],
            vec![HasPhaseBias {
                sat: g01,
                signal_id: 9,
                bias_cycles: None,
                discontinuity_indicator: 0,
            }],
        ));
        store
            .ingest_has_mt1(&later, later_reception)
            .expect("ingest unavailable and do-not-use records");
        let later_epoch = has_mt1_reference_j2000_s(later_reception, 40).unwrap();

        assert!(
            store.orbit(g01).is_none(),
            "unavailable orbit removes G01's"
        );
        assert!(store.clock(g01).is_none(), "do-not-use removes G01's clock");
        assert!(store.is_satellite_excluded(g01, later_epoch));
        assert_eq!(
            store.code_bias(g01, has_sig(g01, 0)),
            None,
            "unavailable code bias removed"
        );
        assert_eq!(
            store.code_bias(g01, has_sig(g01, 9)),
            Some(0.24),
            "other signal kept"
        );
        assert_eq!(
            store.phase_bias(g01, has_sig(g01, 9)),
            None,
            "unavailable phase bias removed"
        );
        assert_eq!(
            store.phase_bias(g01, has_sig(g01, 0)).map(f64::to_bits),
            Some((1.25 * (C_M_S / F_L1_HZ)).to_bits()),
            "other signal kept"
        );
        assert_eq!(store.corrections.get(&g02).cloned(), g02_before);

        // A satellite with nothing stored gains no correction from unavailable
        // records; it gains only their watermarks and the do-not-use exclusion.
        let g03 = GnssSatelliteId::new(GnssSystem::Gps, 3).unwrap();
        let (unavailable_only, reception) = has_later(has_message(
            vec![HasOrbitCorrection {
                sat: g03,
                nav_message: 0,
                iode: 1,
                radial_m: None,
                along_m: None,
                cross_m: None,
            }],
            vec![HasClockCorrection {
                sat: g03,
                nav_message: 0,
                correction_m: None,
                do_not_use: true,
            }],
            Vec::new(),
            Vec::new(),
        ));
        store
            .ingest_has_mt1(&unavailable_only, reception)
            .expect("ingest unavailable records for an unknown satellite");
        assert!(store.orbit(g03).is_none() && store.clock(g03).is_none());
        assert!(store.is_satellite_excluded(g03, later_epoch));

        // Same reference epoch as G02's stored usable orbit: the usable one is kept.
        let same_epoch = has_message(
            vec![HasOrbitCorrection {
                sat: g02,
                nav_message: 0,
                iode: 7,
                radial_m: None,
                along_m: None,
                cross_m: None,
            }],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        store
            .ingest_has_mt1(&same_epoch, has_reception())
            .expect("ingest same-epoch unavailable orbit");
        assert!(store.orbit(g02).is_some());
    }

    /// A later clock record transmitted as unavailable removes the stored HAS
    /// clock. A caller-built record that holds a correction while marked
    /// do-not-use, which the encoder also refuses, is refused by name and the
    /// message changes nothing.
    #[test]
    fn has_unavailable_or_contradictory_clock_removes_stored_clock() {
        let g01 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let g02 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let mut store = store_with_has_corrections(g01, g02);
        let before = store.corrections.clone();
        let (contradictory, reception) = has_later(has_message(
            Vec::new(),
            vec![
                HasClockCorrection {
                    sat: g01,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: false,
                },
                HasClockCorrection {
                    sat: g02,
                    nav_message: 0,
                    correction_m: Some(0.5),
                    do_not_use: true,
                },
            ],
            Vec::new(),
            Vec::new(),
        ));
        let err = store
            .ingest_has_mt1(&contradictory, reception)
            .expect_err("a correction marked do-not-use is contradictory");
        assert!(
            err.to_string()
                .contains("contradictory HAS clock correction for G02"),
            "{err}"
        );
        assert_eq!(
            store.corrections, before,
            "a refused message applies nothing"
        );

        let (unavailable, reception) = has_later(has_message(
            Vec::new(),
            vec![HasClockCorrection {
                sat: g01,
                nav_message: 0,
                correction_m: None,
                do_not_use: false,
            }],
            Vec::new(),
            Vec::new(),
        ));
        store
            .ingest_has_mt1(&unavailable, reception)
            .expect("ingest clock records");
        assert!(store.clock(g01).is_none());
        assert!(store.clock(g02).is_some());
        assert!(store.orbit(g01).is_some() && store.orbit(g02).is_some());
    }

    /// A phase bias on a signal index the HAS table reserves keeps its cycles and
    /// has no metres: no signal or wavelength is assumed for it, the typed query
    /// reports it as an unknown signal, and the signals the table assigns apply.
    #[test]
    fn has_phase_bias_on_unassigned_signal_keeps_cycles_without_metres() {
        let g01 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let mut store = SsrCorrectionStore::new();
        let message = has_message(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![
                HasPhaseBias {
                    sat: g01,
                    signal_id: 10,
                    bias_cycles: Some(1.0),
                    discontinuity_indicator: 0,
                },
                HasPhaseBias {
                    sat: g01,
                    signal_id: 0,
                    bias_cycles: Some(1.0),
                    discontinuity_indicator: 0,
                },
            ],
        );
        store
            .ingest_has_mt1(&message, has_reception())
            .expect("ingest phase biases");
        let epoch = has_mt1_reference_j2000_s(has_reception(), 10).unwrap();
        assert_eq!(store.phase_bias(g01, has_sig(g01, 10)), None);
        // HAS SIS ICD Table 20 reserves GPS index 10: the record is kept under its
        // raw source-qualified signal and reported as naming no known signal.
        let raw = SsrRawSignal::galileo_has(GnssSystem::Gps, 10);
        assert_eq!(has_sig(g01, 10), SsrSignalKey::Unknown(raw));
        let unassigned = store.query_phase_bias(g01, raw, epoch, None);
        assert_eq!(unassigned.status, SsrBiasStatus::UnknownSignal);
        assert_eq!(unassigned.signal, SsrSignalKey::Unknown(raw));
        assert_eq!(unassigned.source_signal, Some(raw));
        assert_eq!(unassigned.bias_m, None);
        assert_eq!(unassigned.bias_cycles, Some(1.0));
        assert_eq!(
            unassigned.details,
            SsrBiasResolutionDetails::UnknownSignal(raw)
        );
        // RTCM SSR GPS index 10 is L2 P, a different signal with its own key.
        assert_eq!(
            store
                .query_phase_bias(g01, rtcm_sig(g01, 10), epoch, None)
                .status,
            SsrBiasStatus::Missing
        );
        assert_eq!(
            store.phase_bias(g01, has_sig(g01, 0)).map(f64::to_bits),
            Some((C_M_S / F_L1_HZ).to_bits())
        );
    }

    /// Every validity interval is read before any record is applied, so a
    /// reserved clock VI leaves the orbit records before it unapplied.
    #[test]
    fn has_reserved_clock_vi_leaves_store_unchanged() {
        let g01 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let g02 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let mut store = store_with_has_corrections(g01, g02);
        let before = store.corrections.clone();
        let mut message = has_message(
            vec![HasOrbitCorrection {
                sat: g01,
                nav_message: 0,
                iode: 9,
                radial_m: None,
                along_m: Some(0.0),
                cross_m: Some(0.0),
            }],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        message.clock_full_set.as_mut().unwrap().validity_interval = 15;
        let err = store
            .ingest_has_mt1(&message, has_reception())
            .expect_err("VI 15 is reserved");
        assert!(
            err.to_string().contains("HAS clock VI is reserved"),
            "{err}"
        );
        assert_eq!(store.corrections, before);
    }

    #[test]
    fn has_binary_orbit_clock_and_biases_ingest_with_additive_convention() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let record = broadcast
            .select_record_at(sat, t)
            .expect("broadcast record at SSR epoch");
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 2,
                iod_set_id: 5,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat.prn],
                    signals: vec![0, 9],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat,
                    nav_message: 0,
                    iode: record.issue_of_data.expect("broadcast issue").issue,
                    radial_m: Some(1.25),
                    along_m: Some(-2.0),
                    cross_m: Some(3.0),
                }],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    nav_message: 0,
                    correction_m: Some(-0.75),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasCodeBias {
                        sat,
                        signal_id: 0,
                        bias_m: Some(0.24),
                    },
                    HasCodeBias {
                        sat,
                        signal_id: 9,
                        bias_m: Some(-0.46),
                    },
                ],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasPhaseBias {
                        sat,
                        signal_id: 0,
                        bias_cycles: Some(1.25),
                        discontinuity_indicator: 1,
                    },
                    HasPhaseBias {
                        sat,
                        signal_id: 9,
                        bias_cycles: Some(-2.5),
                        discontinuity_indicator: 2,
                    },
                ],
            }),
            padding_bits: Vec::new(),
        };
        let decoded = HasMt1Message::decode(&message.encode().expect("encode HAS MT1"))
            .expect("decode HAS MT1");
        let mut store = SsrCorrectionStore::new();
        let reception = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception");
        store
            .ingest_has_mt1(&decoded, reception)
            .expect("ingest HAS MT1");
        let source = SsrCorrectedEphemeris::new(&broadcast, &store);
        let (position, clock) = source.corrected_state(sat, t).expect("HAS corrected state");
        let (broadcast_position, _) = broadcast
            .position_clock_at_j2000_s(sat, t)
            .expect("broadcast state");
        let velocity = finite_difference_broadcast_velocity(&broadcast, sat, t);
        let (er, ea, ec) = analytic_velocity_aligned_basis(broadcast_position, velocity);
        // The RAC correction is summed first and added to the broadcast position as one
        // term, as RTKLIB `satpos_ssr` adds it; summing term by term onto the position
        // rounds differently by up to an ulp of the coordinate (3.7e-9 m here).
        let expected_position = [
            broadcast_position[0] + (1.25 * er[0] - 2.0 * ea[0] + 3.0 * ec[0]),
            broadcast_position[1] + (1.25 * er[1] - 2.0 * ea[1] + 3.0 * ec[1]),
            broadcast_position[2] + (1.25 * er[2] - 2.0 * ea[2] + 3.0 * ec[2]),
        ];
        assert_vector_close(position, expected_position, 2.0e-9);
        // HAS SIS ICD Eq. 23 and 24: the broadcast clock polynomial, less `2 r·v / c²`,
        // plus the delta clock correction (-0.75 m) over c, with no TGD (HAS SIS ICD 7.4).
        // This test once asserted the broadcast clock minus 0.75 / c, but that broadcast
        // clock subtracted the LNAV TGD (4.190951585770e-09 s for G30), and it uses the
        // `F·e·√A·sin E` relativistic term, which Eq. 23 and 24 replace.
        let dclock_m = store.clock(sat).expect("stored HAS clock").c0_m;
        assert_eq!(
            clock.to_bits(),
            satpos_ssr_clock_s(&broadcast, sat, REAL_SSR_EPOCH_TOW_S, dclock_m).to_bits()
        );
        assert!((dclock_m + 0.75).abs() < 1.0e-12, "{dclock_m}");

        assert_eq!(
            store.code_bias(sat, has_sig(sat, 0)).unwrap().to_bits(),
            0.24_f64.to_bits()
        );
        assert_eq!(
            store.code_bias(sat, has_sig(sat, 9)).unwrap().to_bits(),
            (-0.46_f64).to_bits()
        );
        assert_eq!(
            store.phase_bias(sat, has_sig(sat, 0)).unwrap().to_bits(),
            (1.25 * (C_M_S / F_L1_HZ)).to_bits()
        );
        assert_eq!(
            store.phase_bias(sat, has_sig(sat, 9)).unwrap().to_bits(),
            (-2.5 * (C_M_S / F_L2_HZ)).to_bits()
        );
    }

    #[test]
    fn corrected_ephemeris_uses_real_rtcm_broadcast_and_sp3_products() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sp3_bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/IGS0OPSULT_20261811800_02D_15M_ORB.SP3"
        ))
        .expect("read SP3 fixture");
        let sp3 = Sp3::parse(&sp3_bytes).expect("parse SP3 fixture");
        let store = real_gps_ssr_store();
        let source = SsrCorrectedEphemeris::new(&broadcast, &store)
            .with_staleness(StalenessPolicy::seconds(60.0));
        let t = ssr_j2000(REAL_SSR_EPOCH_TOW_S);

        let mut orbit_error_sum_m2 = 0.0;
        let mut clock_error_sum_ns2 = 0.0;
        let mut count = 0_usize;
        for sat in [
            GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap(),
            GnssSatelliteId::new(GnssSystem::Gps, 31).unwrap(),
        ] {
            let (corrected_position, corrected_clock) =
                source.corrected_state(sat, t).expect("corrected state");
            let sp3_state = sp3.position_at_j2000_seconds(sat, t).expect("SP3 state");
            let sp3_position = sp3_state.position.as_array();
            let sp3_clock = sp3_state.clock_s.expect("SP3 clock");
            let orbit_error_m = norm([
                corrected_position[0] - sp3_position[0],
                corrected_position[1] - sp3_position[1],
                corrected_position[2] - sp3_position[2],
            ]);
            let clock_error_ns = (corrected_clock - sp3_clock) * 1.0e9;
            orbit_error_sum_m2 += orbit_error_m * orbit_error_m;
            clock_error_sum_ns2 += clock_error_ns * clock_error_ns;
            count += 1;
        }
        assert_eq!(count, 2);
        let orbit_rms_m = (orbit_error_sum_m2 / count as f64).sqrt();
        let clock_rms_ns = (clock_error_sum_ns2 / count as f64).sqrt();

        // The fixture is the closest public triple assembled on 2026-07-02:
        // captured SSRA02IGS0, matching broadcast records, and ultra-rapid SP3.
        // A rapid or final SP3 for this UTC day was not available at capture time.
        assert!(orbit_rms_m < 1.6, "{orbit_rms_m}");
        assert!(clock_rms_ns < 22.0, "{clock_rms_ns}");
    }

    /// RTKLIB `eph2pos`, `ephpos` and the position half of `satpos_ssr` for one GPS record,
    /// written out statement for statement with `libm` for `sin`, `cos` and `atan2`. With
    /// `fused_node` the node longitude `O=eph->OMG0+(eph->OMGd-omge)*tk-omge*eph->toes` is
    /// formed with two fused multiply-adds, as a C compiler that contracts floating-point
    /// expressions forms it (clang does by default on arm64); without it, each operation is
    /// rounded as the C source writes it.
    fn rtklib_satpos_ssr_position(
        record: &crate::rinex_nav::BroadcastRecord,
        tk: f64,
        deph: [f64; 3],
        fused_node: bool,
    ) -> [f64; 3] {
        let e = &record.elements;
        let eph2pos = |tk: f64| -> [f64; 3] {
            const MU_GPS: f64 = 3.9860050e14;
            const OMGE: f64 = 7.2921151467e-5;
            let a = e.sqrt_a * e.sqrt_a;
            let m = e.m0 + ((MU_GPS / (a * a * a)).sqrt() + e.delta_n) * tk;
            let (mut big_e, mut ek, mut n) = (m, 0.0_f64, 0);
            while (big_e - ek).abs() > 1e-13 && n < 30 {
                ek = big_e;
                big_e -= (big_e - e.e * libm::sin(big_e) - m) / (1.0 - e.e * libm::cos(big_e));
                n += 1;
            }
            let (sin_e, cos_e) = (libm::sin(big_e), libm::cos(big_e));
            let mut u = libm::atan2((1.0 - e.e * e.e).sqrt() * sin_e, cos_e - e.e) + e.omega;
            let mut r = a * (1.0 - e.e * cos_e);
            let mut i = e.i0 + e.idot * tk;
            let (sin2u, cos2u) = (libm::sin(2.0 * u), libm::cos(2.0 * u));
            u += e.cus * sin2u + e.cuc * cos2u;
            r += e.crs * sin2u + e.crc * cos2u;
            i += e.cis * sin2u + e.cic * cos2u;
            let (x, y, cosi) = (r * libm::cos(u), r * libm::sin(u), libm::cos(i));
            let o = if fused_node {
                libm::fma(
                    -OMGE,
                    e.toe_sow,
                    libm::fma(e.omega_dot - OMGE, tk, e.omega0),
                )
            } else {
                e.omega0 + (e.omega_dot - OMGE) * tk - OMGE * e.toe_sow
            };
            let (sin_o, cos_o) = (libm::sin(o), libm::cos(o));
            [
                x * cos_o - y * cosi * sin_o,
                x * sin_o + y * cosi * cos_o,
                y * libm::sin(i),
            ]
        };
        let rs = eph2pos(tk);
        let rst = eph2pos(crate::rinex_nav::ephpos_stepped_tk(tk));
        let v = [
            (rst[0] - rs[0]) / 1e-3,
            (rst[1] - rs[1]) / 1e-3,
            (rst[2] - rs[2]) / 1e-3,
        ];
        let normv = |a: [f64; 3]| {
            let r = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            [a[0] / r, a[1] / r, a[2] / r]
        };
        let cross3 = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let ea = normv(v);
        let ec = normv(cross3(rs, v));
        let er = cross3(ea, ec);
        let mut out = rs;
        for axis in 0..3 {
            out[axis] += -(er[axis] * deph[0] + ea[axis] * deph[1] + ec[axis] * deph[2]) + 0.0;
        }
        out
    }

    /// The corrected G30 state is RTKLIB `satpos_ssr`'s, bit for bit.
    ///
    /// Record: G30 LNAV of `BRDC00WRD_S_20261820000_G30_G31.rnx` (toe and toc 345600 s of
    /// week 2425). Correction: the SSRA02IGS0 1060 frame, IODE 90, epoch 344970 s with a
    /// 10 s update interval, so `t1 = -5 s` and `deph = (0.0807 + 3e-5·t1, 0.2484 -
    /// 4e-5·t1, -0.1396 - 3.2e-5·t1)` m, C0 0.0166 m. At 344970 s, `tk = -630 s`.
    ///
    /// Position. The broadcast position is `eph2pos`'s: Newton's method for Kepler's
    /// equation to 1e-13, `i = (i0 + IDOT·tk) + δi`. The velocity is the 1 ms forward
    /// difference `ephpos` forms, and the correction is added as
    /// `rs[i] += -(er[i]·deph[0] + ea[i]·deph[1] + ec[i]·deph[2])`. Replayed in IEEE double
    /// with each operation rounded as the C source writes it, this gives
    /// (-6327381.424159685, 15802129.789888276, -20121898.098271403) m, the bits pinned
    /// here. RTKLIB built with floating-point contraction (clang's default on arm64) forms
    /// the node longitude with fused multiply-adds and gives (-6327381.424159626,
    /// 15802129.789888298, -20121898.098271403) m instead, 64 and 12 ulp away in x and y;
    /// the second replay below reproduces those bits, so the difference is that
    /// contraction alone. Before this change the fixed-point Kepler solver (to 1e-12) and
    /// the `(i0 + δi) + IDOT·tk` inclination gave bits 13931924021901094555,
    /// 4714745314434008015 and 13939538677975909640.
    ///
    /// Clock. `f0 + f1·tk + f2·tk² = 2.8009318884866823e-04 s` (af0 2.801017835736e-04 s,
    /// af1 1.364242052659e-11 s/s, af2 0), less `2 r·v / c / c` from the position and
    /// velocity above, plus `0.0166 / c`: 2.800865527753679e-04 s. There is no TGD
    /// (4.190951585770e-09 s) in it. Before this change it was 4553802308601788245
    /// (2.8008655277821554e-04 s), from the fixed-point broadcast position.
    #[test]
    fn corrected_state_matches_rtklib_satpos_ssr_oracle_for_one_epoch() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let store = real_gps_ssr_store();
        let source = SsrCorrectedEphemeris::new(&broadcast, &store)
            .with_staleness(StalenessPolicy::seconds(60.0));
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let (position, clock) = source.corrected_state(sat, t).expect("corrected state");
        assert_eq!(
            position.map(f64::to_bits),
            [
                13_931_924_021_901_094_570,
                4_714_745_314_434_008_007,
                13_939_538_677_975_909_641,
            ]
        );
        assert_eq!(clock.to_bits(), 2.800_865_527_753_679e-4_f64.to_bits());
        assert_eq!(clock.to_bits(), 4_553_802_308_601_735_715);
        assert_eq!(
            clock.to_bits(),
            satpos_ssr_clock_s(
                &broadcast,
                sat,
                REAL_SSR_EPOCH_TOW_S,
                store.clock(sat).expect("stored clock").c0_m,
            )
            .to_bits()
        );

        let record = broadcast
            .select_record_at(sat, t)
            .expect("broadcast record");
        let orbit = store.orbit(sat).expect("stored orbit");
        let dt = t - orbit.ref_epoch_j2000_s;
        assert_eq!(dt, -5.0);
        let deph = [
            -(orbit.radial_m + orbit.radial_rate_m_s * dt),
            -(orbit.along_m + orbit.along_rate_m_s * dt),
            -(orbit.cross_m + orbit.cross_rate_m_s * dt),
        ];
        let tk = REAL_SSR_EPOCH_TOW_S - record.elements.toe_sow;
        assert_eq!(
            rtklib_satpos_ssr_position(record, tk, deph, false).map(f64::to_bits),
            position.map(f64::to_bits)
        );
        let contracted = [
            -6_327_381.424_159_626,
            15_802_129.789_888_298,
            -20_121_898.098_271_403,
        ];
        assert_eq!(
            rtklib_satpos_ssr_position(record, tk, deph, true).map(f64::to_bits),
            contracted.map(f64::to_bits)
        );
    }

    #[test]
    fn com_reference_point_requires_explicit_attitude_model() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let antex = Antex::parse(GPS_ANTEX_TEXT).expect("parse ANTEX fixture");
        let store = real_gps_ssr_store_with_reference_point(SsrReferencePoint::CenterOfMass);
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t = ssr_j2000(REAL_SSR_EPOCH_TOW_S);

        let no_attitude = SsrCorrectedEphemeris::new(&broadcast, &store)
            .with_staleness(StalenessPolicy::seconds(60.0))
            .with_satellite_antennas(&antex);
        assert!(no_attitude.corrected_state(sat, t).is_none());

        let fallback = SsrCorrectedEphemeris::new(&broadcast, &store)
            .with_staleness(StalenessPolicy::seconds(60.0))
            .with_satellite_antennas(&antex)
            .with_fallback(SsrFallbackPolicy {
                on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                regional: RegionalPolicy::DeclineRegional,
            });
        assert!(fallback.corrected_state(sat, t).is_none());

        let nominal = SsrCorrectedEphemeris::new(&broadcast, &store)
            .with_staleness(StalenessPolicy::seconds(60.0))
            .with_satellite_antennas(&antex)
            .with_satellite_attitude(SsrSatelliteAttitude::NominalSunFixed);
        assert!(nominal.corrected_state(sat, t).is_some());
    }

    #[test]
    fn com_to_apc_after_the_ut1_table_is_refused_or_reported() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let antex = Antex::parse(GPS_ANTEX_TEXT).expect("parse ANTEX fixture");
        let store = real_gps_ssr_store_with_reference_point(SsrReferencePoint::CenterOfMass);
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let source = SsrCorrectedEphemeris::new(&broadcast, &store)
            .with_staleness(StalenessPolicy::seconds(60.0))
            .with_satellite_antennas(&antex)
            .with_satellite_attitude(SsrSatelliteAttitude::NominalSunFixed);
        let satellite_ecef_m = [15_000.0e3, 10_000.0e3, 18_000.0e3];
        // 2027-09-01 00:00 GPST: JD 2461649.5 is 10104.5 days after J2000,
        // MJD 61649, past the UT1 table (it ends at MJD 61589).
        let after = 10_104.5 * 86_400.0;
        let refused = crate::astro::frames::transforms::FrameTransformError::Ut1OutsideCoverage {
            reason: DegradeReason::AfterCoverage,
        };

        // Strict: the conversion is refused and the refusal is kept, so the
        // caller reports it rather than an unavailable satellite.
        let strict = Ut1Gate::new(ValidityMode::Strict);
        assert_eq!(
            source.satellite_pco_to_apc(sat, after, satellite_ecef_m, &strict),
            None
        );
        assert_eq!(strict.finish(()), Err(refused));
        assert_eq!(
            Error::Ut1OutsideCoverage(DegradeReason::AfterCoverage).to_string(),
            "UT1 outside the table: instant follows the UT1 table coverage"
        );

        // Permissive: the offset is computed with the long-term UT1 and the
        // departure is reported.
        let permissive_source = source.clone().with_validity(ValidityMode::Permissive);
        let permissive = Ut1Gate::new(ValidityMode::Permissive);
        let offset = permissive_source
            .satellite_pco_to_apc(sat, after, satellite_ecef_m, &permissive)
            .expect("APC offset with the long-term UT1");
        assert!(offset.iter().all(|value| value.is_finite()));
        assert_eq!(
            permissive.finish(()).map(|validated| validated.degraded),
            Ok(Some(DegradeReason::AfterCoverage))
        );

        // Inside the table both modes give the same corrected state, with no
        // departure, and the source records none.
        let t = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let strict_state = source.corrected_state_checked(sat, t).expect("in table");
        let permissive_state = permissive_source
            .corrected_state_checked(sat, t)
            .expect("in table");
        assert_eq!(strict_state, permissive_state);
        assert_eq!(permissive_state.degraded, None);
        assert!(permissive_state.value.is_some());
        assert_eq!(permissive_source.ut1_departure(), None);
    }

    #[test]
    fn com_orbit_tag_blocks_fallback_after_store_default_changes_to_apc() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let antex = Antex::parse(GPS_ANTEX_TEXT).expect("parse ANTEX fixture");
        let mut store = real_gps_ssr_store_with_reference_point(SsrReferencePoint::CenterOfMass);
        store = store.with_reference_point(SsrReferencePoint::AntennaPhaseCenter);
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        assert_eq!(
            store.reference_point(),
            SsrReferencePoint::AntennaPhaseCenter
        );
        assert_eq!(
            store.orbit(sat).unwrap().reference_point,
            SsrReferencePoint::CenterOfMass
        );

        let source = SsrCorrectedEphemeris::new(&broadcast, &store)
            .with_staleness(StalenessPolicy::seconds(60.0))
            .with_satellite_antennas(&antex)
            .with_fallback(SsrFallbackPolicy {
                on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                regional: RegionalPolicy::DeclineRegional,
            });

        assert_eq!(
            source.observable_state_at_j2000_s(sat, t),
            Err(ObservablesError::NoEphemeris)
        );
    }

    fn real_gps_ssr_store() -> SsrCorrectionStore {
        real_gps_ssr_store_with_reference_point(SsrReferencePoint::rtcm_ssr_default())
    }

    fn real_gps_ssr_store_with_reference_point(
        reference_point: SsrReferencePoint,
    ) -> SsrCorrectionStore {
        let mut assembler = SsrStreamAssembler::new();
        let mut store = SsrCorrectionStore::new().with_reference_point(reference_point);
        let week = GnssWeekTow::new(TimeScale::Gpst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("valid SSR week");
        for decoded in assembler.push(&hex_bytes(REAL_SSRA02IGS0_1060_FRAME_HEX)) {
            let message = decoded.expect("decode SSR frame");
            store.ingest(&message, week).expect("ingest SSR frame");
        }
        assert_eq!(assembler.retained_len(), 0);
        store
    }

    fn ssr_j2000(tow_s: f64) -> f64 {
        f64::from(REAL_SSR_WEEK) * SECONDS_PER_WEEK + tow_s - GPS_EPOCH_TO_J2000_S
    }

    fn hex_bytes(hex: &str) -> Vec<u8> {
        let compact: String = hex.chars().filter(|c| c.is_ascii_hexdigit()).collect();
        assert_eq!(compact.len() % 2, 0);
        compact
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| {
                let hi = (chunk[0] as char).to_digit(16).unwrap();
                let lo = (chunk[1] as char).to_digit(16).unwrap();
                ((hi << 4) | lo) as u8
            })
            .collect()
    }

    /// The satellite clock RTKLIB `satpos_ssr` forms for a GPS satellite at `tow_s` of
    /// `REAL_SSR_WEEK`, written as `satpos_ssr` writes it: `f0 + f1·tk + f2·tk²` of the
    /// broadcast record with `tk = t - toc`, less `2 r·v / c / c` for the record's
    /// position and its 1 ms forward-difference velocity, plus `dclk / c`.
    fn satpos_ssr_clock_s(
        broadcast: &BroadcastEphemeris,
        sat: GnssSatelliteId,
        tow_s: f64,
        dclock_m: f64,
    ) -> f64 {
        let t = ssr_j2000(tow_s);
        let record = broadcast
            .select_record_at(sat, t)
            .expect("broadcast record");
        let tk = tow_s - record.clock.toc_sow;
        let mut dts = record.clock.af0 + record.clock.af1 * tk + record.clock.af2 * tk * tk;
        let (r, _) = broadcast
            .position_clock_at_j2000_s(sat, t)
            .expect("broadcast state");
        let v = finite_difference_broadcast_velocity(broadcast, sat, t);
        dts -= 2.0 * dot3(r, v) / C_M_S / C_M_S;
        dts += dclock_m / C_M_S;
        dts
    }

    fn finite_difference_broadcast_velocity(
        broadcast: &BroadcastEphemeris,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> [f64; 3] {
        // RTKLIB `ephpos`: the record's positions at `tk` and `tk` + 1 ms.
        broadcast
            .selected_record_velocity(sat, t_j2000_s)
            .expect("broadcast record velocity")
    }

    fn analytic_velocity_aligned_basis(
        position: [f64; 3],
        velocity: [f64; 3],
    ) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let along = unit(velocity);
        let cross_track = unit(cross3(position, velocity));
        let radial = cross3(along, cross_track);
        (radial, along, cross_track)
    }

    fn assert_vector_close(actual: [f64; 3], expected: [f64; 3], tolerance_m: f64) {
        let error = norm([
            actual[0] - expected[0],
            actual[1] - expected[1],
            actual[2] - expected[2],
        ]);
        assert!(
            error <= tolerance_m,
            "error {error}, expected {expected:?}, got {actual:?}"
        );
    }

    fn unit(v: [f64; 3]) -> [f64; 3] {
        let n = norm(v);
        [v[0] / n, v[1] / n, v[2] / n]
    }

    fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn norm(v: [f64; 3]) -> f64 {
        dot(v, v).sqrt()
    }

    #[test]
    fn has_ingest_refuses_caller_built_contradictory_record_without_leaving_cached_state() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let reception = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception");

        // 1. Contradictory full-set block: correction_m is Some while do_not_use is true.
        let mut store = SsrCorrectionStore::new();
        let contradictory_full_set = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat,
                    nav_message: 0,
                    iode: 10,
                    radial_m: Some(1.0),
                    along_m: Some(2.0),
                    cross_m: Some(3.0),
                }],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    nav_message: 0,
                    correction_m: Some(1.25),
                    do_not_use: true,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        let err = store
            .ingest_has_mt1(&contradictory_full_set, reception)
            .unwrap_err();
        let err_str = err.to_string();
        assert!(
            err_str.contains("contradictory HAS clock correction for G30"),
            "{err_str}"
        );
        assert!(
            err_str.contains("correction is Some while do_not_use is true"),
            "{err_str}"
        );
        // Verify store was not mutated: no orbit, clock, or exclusion was cached.
        assert!(store.orbit(sat).is_none());
        assert!(store.clock(sat).is_none());
        assert!(store.has_exclusion(sat).is_none());
        assert!(!store.is_satellite_excluded(sat, ssr_j2000(REAL_SSR_EPOCH_TOW_S)));

        // 2. Contradictory subset block: correction_m is Some while do_not_use is true.
        let contradictory_subset = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: None,
            clock_subset: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    nav_message: 0,
                    correction_m: Some(-0.5),
                    do_not_use: true,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        let err_sub = store
            .ingest_has_mt1(&contradictory_subset, reception)
            .unwrap_err();
        let err_sub_str = err_sub.to_string();
        assert!(
            err_sub_str.contains("contradictory HAS clock correction for G30"),
            "{err_sub_str}"
        );
        assert!(store.clock(sat).is_none());
        assert!(store.has_exclusion(sat).is_none());
        assert!(!store.is_satellite_excluded(sat, ssr_j2000(REAL_SSR_EPOCH_TOW_S)));
    }

    /// A newer HAS message that marks an orbit, clock, code bias or phase bias
    /// unavailable withdraws the older HAS value: neither the store accessors,
    /// the typed queries nor `corrected_state` serve the older value, although
    /// its own validity interval still covers the query epoch, and a delayed
    /// copy of the older message cannot bring it back.
    #[test]
    fn has_unavailable_components_are_never_served_from_older_state() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t0_tow = REAL_SSR_EPOCH_TOW_S;
        let t0 = ssr_j2000(t0_tow);
        let t1 = ssr_j2000(t0_tow + 30.0);
        let iode = broadcast
            .select_record_at(sat, t0)
            .expect("broadcast record at SSR epoch")
            .issue_of_data
            .expect("broadcast issue")
            .issue;
        let reception =
            |tow: f64| GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, tow).expect("GST reception");
        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });
        // Validity interval index 5 is 60 s, so the older message still covers t1.
        let build = |toh_offset_s: u32,
                     iod_set_id: u8,
                     orbit: Option<[Option<f64>; 3]>,
                     clock_m: Option<Option<f64>>,
                     code_m: Option<Option<f64>>,
                     phase_cycles: Option<Option<f64>>| HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + toh_offset_s) as u16,
                mask: true,
                orbit: orbit.is_some(),
                clock_full_set: clock_m.is_some(),
                clock_subset: false,
                code_bias: code_m.is_some(),
                phase_bias: phase_cycles.is_some(),
                reserved: 0,
                mask_id: 1,
                iod_set_id,
            },
            mask: mask.clone(),
            orbit: orbit.map(|[radial_m, along_m, cross_m]| HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat,
                    nav_message: 0,
                    iode,
                    radial_m,
                    along_m,
                    cross_m,
                }],
            }),
            clock_full_set: clock_m.map(|correction_m| HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    nav_message: 0,
                    correction_m,
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: code_m.map(|bias_m| HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m,
                }],
            }),
            phase_bias: phase_cycles.map(|bias_cycles| HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles,
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };
        let older = build(
            0,
            1,
            Some([Some(1.0), Some(0.5), Some(-0.5)]),
            Some(Some(-0.5)),
            Some(Some(0.25)),
            Some(Some(0.125)),
        );
        let fallback = SsrFallbackPolicy {
            on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
            regional: RegionalPolicy::DeclineRegional,
        };
        let broadcast_state = broadcast
            .position_clock_at_j2000_s(sat, t1)
            .expect("broadcast state at t1");

        let mut seeded = SsrCorrectionStore::new();
        seeded.ingest_has_mt1(&older, reception(t0_tow)).unwrap();
        let older_state = SsrCorrectedEphemeris::new(&broadcast, &seeded)
            .corrected_state(sat, t1)
            .expect("older HAS orbit and clock cover t1");
        assert_ne!(older_state, broadcast_state);
        assert_eq!(
            seeded.query_code_bias(sat, has_sig(sat, 0), t1).bias_m,
            Some(0.25)
        );
        assert!(seeded
            .query_phase_bias(sat, has_sig(sat, 0), t1, None)
            .bias_m
            .is_some());

        let newer_orbit_unavailable = build(
            30,
            2,
            Some([Some(1.0), None, Some(-0.5)]),
            None,
            Some(None),
            Some(None),
        );
        let newer_clock_unavailable = build(30, 2, None, Some(None), None, None);

        for (name, newer) in [
            (
                "orbit with one unavailable component",
                &newer_orbit_unavailable,
            ),
            ("unavailable clock", &newer_clock_unavailable),
        ] {
            let mut store = seeded.clone();
            store
                .ingest_has_mt1(newer, reception(t0_tow + 30.0))
                .unwrap();
            for pass in ["after the newer message", "after a delayed older copy"] {
                if name.starts_with("orbit") {
                    assert!(store.orbit(sat).is_none(), "{name}, {pass}");
                    assert_eq!(
                        store.code_bias(sat, has_sig(sat, 0)),
                        None,
                        "{name}, {pass}"
                    );
                    assert_eq!(
                        store.phase_bias(sat, has_sig(sat, 0)),
                        None,
                        "{name}, {pass}"
                    );
                    let code = store.query_code_bias(sat, has_sig(sat, 0), t1);
                    assert_eq!(code.status, SsrBiasStatus::Unavailable, "{name}, {pass}");
                    assert_eq!(code.bias_m, None, "{name}, {pass}");
                    let phase = store.query_phase_bias(sat, has_sig(sat, 0), t1, None);
                    assert_eq!(phase.status, SsrBiasStatus::Unavailable, "{name}, {pass}");
                    assert_eq!(phase.bias_m, None, "{name}, {pass}");
                } else {
                    assert!(store.clock(sat).is_none(), "{name}, {pass}");
                }
                assert_eq!(
                    SsrCorrectedEphemeris::new(&broadcast, &store).corrected_state(sat, t1),
                    None,
                    "{name}, {pass}: the older HAS state must not be served"
                );
                assert_eq!(
                    SsrCorrectedEphemeris::new(&broadcast, &store)
                        .with_fallback(fallback.clone())
                        .corrected_state(sat, t1),
                    Some(broadcast_state),
                    "{name}, {pass}: fallback serves broadcast, not the older HAS state"
                );
                store.ingest_has_mt1(&older, reception(t0_tow)).unwrap();
            }
        }
    }

    #[test]
    fn has_ingest_do_not_use_excludes_satellite_and_blocks_broadcast_fallback() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let record = broadcast
            .select_record_at(sat, t)
            .expect("broadcast record at SSR epoch");
        let reception = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception");

        let mut store = SsrCorrectionStore::new();

        // Step 1: Ingest usable orbit and clock for G30.
        let usable_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat,
                    nav_message: 0,
                    iode: record.issue_of_data.expect("broadcast issue").issue,
                    radial_m: Some(1.0),
                    along_m: Some(0.0),
                    cross_m: Some(0.0),
                }],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    nav_message: 0,
                    correction_m: Some(-0.5),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&usable_msg, reception)
            .expect("ingest usable HAS MT1");

        let strict_source = SsrCorrectedEphemeris::new(&broadcast, &store);
        assert!(strict_source.corrected_state(sat, t).is_some());
        let fallback_source =
            SsrCorrectedEphemeris::new(&broadcast, &store).with_fallback(SsrFallbackPolicy {
                on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                regional: RegionalPolicy::DeclineRegional,
            });
        assert!(fallback_source.corrected_state(sat, t).is_some());

        // Step 2: Ingest explicit do-not-use indication for G30.
        let dnu_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 2,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: true,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&dnu_msg, reception)
            .expect("ingest do-not-use HAS MT1");

        // Verify cached clock is removed and exclusion is active.
        assert!(store.clock(sat).is_none());
        assert!(store.has_exclusion(sat).is_some());
        assert!(store.is_satellite_excluded(sat, t));

        // Borrowed implementation: strict declines, and fallback CANNOT bypass exclusion.
        let strict_after = SsrCorrectedEphemeris::new(&broadcast, &store);
        assert!(strict_after.corrected_state(sat, t).is_none());
        let fallback_after =
            SsrCorrectedEphemeris::new(&broadcast, &store).with_fallback(SsrFallbackPolicy {
                on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                regional: RegionalPolicy::DeclineRegional,
            });
        assert!(fallback_after.corrected_state(sat, t).is_none());
        assert_eq!(
            fallback_after.observable_state_at_j2000_s(sat, t),
            Err(ObservablesError::NoEphemeris)
        );

        // Owned wrapper implementation: must also decline both corrected and broadcast fallback.
        let owned_fallback = SsrCorrectedEphemerisOwned::new(
            Arc::new(BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture")),
            Arc::new(store.clone()),
        )
        .with_fallback(SsrFallbackPolicy {
            on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
            regional: RegionalPolicy::DeclineRegional,
        });
        assert!(owned_fallback.corrected_state(sat, t).is_none());
        assert_eq!(
            owned_fallback.observable_state_at_j2000_s(sat, t),
            Err(ObservablesError::NoEphemeris)
        );

        // Contrast with unavailable sentinel (-4096): does not set an exclusion.
        let mut unavail_store = SsrCorrectionStore::new();
        let unavail_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 3,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        unavail_store
            .ingest_has_mt1(&unavail_msg, reception)
            .expect("ingest unavailable HAS MT1");
        assert!(unavail_store.has_exclusion(sat).is_none());
        assert!(!unavail_store.is_satellite_excluded(sat, t));
    }

    #[test]
    fn has_exclusion_temporal_bounds_and_fallback_resumption() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t_ref = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let reception = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception");

        let mut store = SsrCorrectionStore::new();
        // Indication with VI index 5 = 60 seconds validity interval.
        let dnu_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5, // 60s
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: true,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&dnu_msg, reception)
            .expect("ingest do-not-use HAS MT1");

        let strict = SsrCorrectedEphemeris::new(&broadcast, &store);
        let fallback =
            SsrCorrectedEphemeris::new(&broadcast, &store).with_fallback(SsrFallbackPolicy {
                on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                regional: RegionalPolicy::DeclineRegional,
            });

        // 1. Before reference time: t_ref - 1.0s.
        // Exclusion is not active; fallback resumes broadcast, while strict declines.
        let t_before = t_ref - 1.0;
        assert!(!store.is_satellite_excluded(sat, t_before));
        assert!(strict.corrected_state(sat, t_before).is_none());
        assert!(fallback.corrected_state(sat, t_before).is_some());

        // 2. At reference time: t_ref.
        // Exclusion is active; both strict and fallback decline.
        assert!(store.is_satellite_excluded(sat, t_ref));
        assert!(strict.corrected_state(sat, t_ref).is_none());
        assert!(fallback.corrected_state(sat, t_ref).is_none());

        // 3. Just inside validity interval: t_ref + 30.0s.
        // Exclusion is active; both decline.
        let t_inside = t_ref + 30.0;
        assert!(store.is_satellite_excluded(sat, t_inside));
        assert!(strict.corrected_state(sat, t_inside).is_none());
        assert!(fallback.corrected_state(sat, t_inside).is_none());

        // 4. At the chosen end boundary: t_ref + 60.0s.
        // In accordance with SSR <= validity convention, boundary is inclusive.
        // Exclusion is active; both decline.
        let t_end = t_ref + 60.0;
        assert!(store.is_satellite_excluded(sat, t_end));
        assert!(strict.corrected_state(sat, t_end).is_none());
        assert!(fallback.corrected_state(sat, t_end).is_none());

        // 5. Just after expiration: t_ref + 60.001s.
        // Exclusion has expired; strict missing-correction handling continues to decline,
        // while enabled broadcast fallback resumes.
        let t_after = t_ref + 60.001;
        assert!(!store.is_satellite_excluded(sat, t_after));
        assert!(strict.corrected_state(sat, t_after).is_none());
        let (pos_after, clk_after) = fallback
            .corrected_state(sat, t_after)
            .expect("broadcast fallback resumes after exclusion expires");
        let (b_pos, b_clk) = broadcast
            .position_clock_at_j2000_s(sat, t_after)
            .expect("broadcast state");
        assert_eq!(pos_after.map(f64::to_bits), b_pos.map(f64::to_bits));
        assert_eq!(clk_after.to_bits(), b_clk.to_bits());

        // Verify caller's tight staleness setting cannot shorten an active exclusion
        // into permission to use broadcast.
        let tight_staleness_fallback = SsrCorrectedEphemeris::new(&broadcast, &store)
            .with_staleness(StalenessPolicy::seconds(10.0))
            .with_fallback(SsrFallbackPolicy {
                on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                regional: RegionalPolicy::DeclineRegional,
            });
        // At t_ref + 30.0s, the age (30s) exceeds caller's max_staleness_s (10s),
        // but the exclusion remains active (validity interval is 60s) and fallback MUST be declined.
        assert!(tight_staleness_fallback
            .corrected_state(sat, t_inside)
            .is_none());
    }

    #[test]
    fn has_exclusion_cleared_by_later_correction_not_older_and_preserves_unrelated_satellites() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 31).unwrap();
        let t_ref = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let reception1 = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception 1");
        let record1 = broadcast
            .select_record_at(sat1, t_ref)
            .expect("broadcast record sat1");
        let record2 = broadcast
            .select_record_at(sat2, t_ref)
            .expect("broadcast record sat2");

        let mut store = SsrCorrectionStore::new();

        // Ingest initial message at t_ref:
        // Orbit for both G30 and G31.
        // Clock for G31: usable (-0.5m).
        // Clock for G30: do-not-use (+4095 wire sentinel).
        let initial_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn, sat2.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5, // 60s
                records: vec![
                    HasOrbitCorrection {
                        sat: sat1,
                        nav_message: 0,
                        iode: record1.issue_of_data.expect("broadcast issue").issue,
                        radial_m: Some(1.0),
                        along_m: Some(0.0),
                        cross_m: Some(0.0),
                    },
                    HasOrbitCorrection {
                        sat: sat2,
                        nav_message: 0,
                        iode: record2.issue_of_data.expect("broadcast issue").issue,
                        radial_m: Some(1.0),
                        along_m: Some(0.0),
                        cross_m: Some(0.0),
                    },
                ],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5, // 60s
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![
                    HasClockCorrection {
                        sat: sat1,
                        nav_message: 0,
                        correction_m: None,
                        do_not_use: true,
                    },
                    HasClockCorrection {
                        sat: sat2,
                        nav_message: 0,
                        correction_m: Some(-0.5),
                        do_not_use: false,
                    },
                ],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&initial_msg, reception1)
            .expect("ingest initial HAS MT1");

        assert!(store.is_satellite_excluded(sat1, t_ref));
        assert!(!store.is_satellite_excluded(sat2, t_ref));
        {
            let fallback_source =
                SsrCorrectedEphemeris::new(&broadcast, &store).with_fallback(SsrFallbackPolicy {
                    on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                    regional: RegionalPolicy::DeclineRegional,
                });
            assert!(fallback_source.corrected_state(sat1, t_ref).is_none());
            assert!(fallback_source.corrected_state(sat2, t_ref).is_some());
        }

        // Part A: An older usable correction delivered after the do-not-use does NOT clear the exclusion.
        // Reception is at t_ref + 10s, but TOH corresponds to t_ref - 30s (an older epoch).
        let reception_older =
            GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S + 10.0)
                .expect("GST reception older");
        let older_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) - 30) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 2,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(0.75),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&older_msg, reception_older)
            .expect("ingest older HAS MT1");

        // G30 exclusion MUST remain intact, and old clock must NOT be cached.
        assert!(store.is_satellite_excluded(sat1, t_ref));
        assert!(store.clock(sat1).is_none());
        {
            let fallback_source =
                SsrCorrectedEphemeris::new(&broadcast, &store).with_fallback(SsrFallbackPolicy {
                    on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                    regional: RegionalPolicy::DeclineRegional,
                });
            assert!(fallback_source.corrected_state(sat1, t_ref).is_none());
            // Unrelated satellite G31 must remain unaffected.
            assert_eq!(
                store.clock(sat2).unwrap().c0_m.to_bits(),
                (-0.5_f64).to_bits()
            );
            assert!(fallback_source.corrected_state(sat2, t_ref).is_some());
        }

        // Part B: Same-epoch conflict preserves conservative exclusion.
        let same_epoch_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 3,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(-0.25),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&same_epoch_msg, reception1)
            .expect("ingest same-epoch HAS MT1");
        assert!(store.is_satellite_excluded(sat1, t_ref));
        assert!(store.clock(sat1).is_none());

        // Part C: A later usable correction clears the exclusion and restores corrected output.
        // Epoch at t_ref + 60s.
        let t_later = t_ref + 60.0;
        let reception_later =
            GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S + 65.0)
                .expect("GST reception later");
        let later_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 60) as u16,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 4,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat: sat1,
                    nav_message: 0,
                    iode: record1.issue_of_data.expect("broadcast issue").issue,
                    radial_m: Some(1.0),
                    along_m: Some(0.0),
                    cross_m: Some(0.0),
                }],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(-0.35),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&later_msg, reception_later)
            .expect("ingest later HAS MT1");

        // G30 exclusion is cleared and new clock is cached.
        assert!(store.has_exclusion(sat1).is_none());
        assert!(!store.is_satellite_excluded(sat1, t_later));
        assert_eq!(
            store.clock(sat1).unwrap().c0_m.to_bits(),
            (-0.35_f64).to_bits()
        );
        let corrected_after = SsrCorrectedEphemeris::new(&broadcast, &store);
        assert!(corrected_after.corrected_state(sat1, t_later).is_some());
        // Satellite 2 still intact.
        assert_eq!(
            store.clock(sat2).unwrap().c0_m.to_bits(),
            (-0.5_f64).to_bits()
        );
    }

    #[test]
    fn has_exclusion_toh_hour_rollover_starts_at_reference_time_not_receipt() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        // Reception GST is 10 seconds past an hour boundary: week 2425, TOW = 3610.0 s.
        let reception_tow_s = 3610.0;
        let reception_gst = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, reception_tow_s)
            .expect("GST reception");

        // Indication has toh_s = 3590 s (59m 50s), which belongs to the preceding hour (TOW = 3590.0 s).
        let toh_s = 3590_u16;

        // Verify has_mt1_reference_j2000_s resolves reference time to the preceding hour per Eqs 28/29.
        let resolved_ref_j2000_s =
            has_mt1_reference_j2000_s(reception_gst, toh_s).expect("resolve reference J2000");
        let expected_ref_gst_s = f64::from(REAL_SSR_WEEK) * SECONDS_PER_WEEK + 3590.0;
        let expected_ref_j2000_s = expected_ref_gst_s - GPS_EPOCH_TO_J2000_S;
        assert_eq!(
            resolved_ref_j2000_s.to_bits(),
            expected_ref_j2000_s.to_bits(),
            "reference time must be 20 seconds before reception"
        );

        // Ingest do-not-use indication with validity interval index 3 = 20 seconds.
        // Valid interval is [expected_ref_j2000_s, expected_ref_j2000_s + 20.0s].
        let mut store = SsrCorrectionStore::new();
        let msg = HasMt1Message {
            header: HasMt1Header {
                toh_s,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 3, // 20s
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: true,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&msg, reception_gst)
            .expect("ingest rollover HAS MT1");

        let marker = store.has_exclusion(sat).expect("exclusion marker exists");
        assert_eq!(
            marker.ref_epoch_j2000_s.to_bits(),
            expected_ref_j2000_s.to_bits()
        );
        assert_eq!(marker.validity_interval_s, 20.0);

        // 1. Before reference time: not active.
        let t_before_ref = expected_ref_j2000_s - 1.0;
        assert!(!store.is_satellite_excluded(sat, t_before_ref));

        // 2. At reference time: active.
        assert!(store.is_satellite_excluded(sat, expected_ref_j2000_s));

        // 3. Pre-receipt epoch: 10s after reference time (TOW 3600.0s), which is 10s BEFORE reception (TOW 3610.0s).
        // The exclusion starts at reference time, not reception time, so it IS active here!
        let t_pre_receipt = expected_ref_j2000_s + 10.0;
        assert!(store.is_satellite_excluded(sat, t_pre_receipt));

        // 4. At reception time: expected_ref_j2000_s + 20.0s (TOW 3610.0s).
        // By SSR <= validity convention, the endpoint is active.
        let t_receipt = expected_ref_j2000_s + 20.0;
        assert!(store.is_satellite_excluded(sat, t_receipt));

        // 5. Just after expiry: expected_ref_j2000_s + 20.001s.
        // The exclusion has expired based on reference time, NOT continuing 20s from receipt!
        let t_expired = expected_ref_j2000_s + 20.001;
        assert!(!store.is_satellite_excluded(sat, t_expired));
    }

    #[test]
    fn has_clock_unavailable_full_set_lifecycle_and_fallback_resumption() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 31).unwrap();
        let t_ref = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let reception1 = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception 1");
        let record1 = broadcast
            .select_record_at(sat1, t_ref)
            .expect("broadcast record sat1");
        let record2 = broadcast
            .select_record_at(sat2, t_ref)
            .expect("broadcast record sat2");

        let mut store = SsrCorrectionStore::new();

        // 1. Initial message at t_ref:
        // Orbit for sat1 and sat2; usable clock for both (sat1: 0.75m, sat2: -0.50m).
        let initial_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn, sat2.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5, // 60s
                records: vec![
                    HasOrbitCorrection {
                        sat: sat1,
                        nav_message: 0,
                        iode: record1.issue_of_data.expect("broadcast issue").issue,
                        radial_m: Some(1.0),
                        along_m: Some(0.0),
                        cross_m: Some(0.0),
                    },
                    HasOrbitCorrection {
                        sat: sat2,
                        nav_message: 0,
                        iode: record2.issue_of_data.expect("broadcast issue").issue,
                        radial_m: Some(1.0),
                        along_m: Some(0.0),
                        cross_m: Some(0.0),
                    },
                ],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5, // 60s
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![
                    HasClockCorrection {
                        sat: sat1,
                        nav_message: 0,
                        correction_m: Some(0.75),
                        do_not_use: false,
                    },
                    HasClockCorrection {
                        sat: sat2,
                        nav_message: 0,
                        correction_m: Some(-0.50),
                        do_not_use: false,
                    },
                ],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&initial_msg, reception1)
            .expect("ingest initial HAS MT1");

        // Attach pending high-rate clock with HAS provenance for sat1 to test that HAS unavailable clears it
        store.corrections.entry(sat1).or_default().pending_high_rate = Some(SsrHighRateClock {
            solution: SsrSolution {
                source: SsrSource::GalileoHas,
                provider_id: 1,
                solution_id: 1,
            },
            iod_ssr: 1,
            c0_m: 0.75,
            ref_epoch_j2000_s: t_ref,
            transmitted_epoch_j2000_s: t_ref,
            update_interval_s: 5.0,
        });
        assert!(store.pending_high_rate(sat1).is_some());

        // Verify initial state: both satellites have usable clocks and corrected state
        assert_eq!(
            store.clock(sat1).unwrap().c0_m.to_bits(),
            0.75_f64.to_bits()
        );
        assert_eq!(
            store.clock(sat2).unwrap().c0_m.to_bits(),
            (-0.50_f64).to_bits()
        );
        {
            let strict = SsrCorrectedEphemeris::new(&broadcast, &store);
            assert!(strict.corrected_state(sat1, t_ref).is_some());
            assert!(strict.corrected_state(sat2, t_ref).is_some());
        }

        // 2. Strictly newer unavailable clock for sat1 at t_ref + 30s
        // sat1: data unavailable (-4096 wire sentinel: correction_m None, do_not_use false)
        // sat2: remains usable with updated value (-0.60m)
        let t_unavail = t_ref + 30.0;
        let reception2 =
            GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S + 30.0)
                .expect("GST reception 2");
        let unavail_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 30) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 2,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn, sat2.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![
                    HasClockCorrection {
                        sat: sat1,
                        nav_message: 0,
                        correction_m: None,
                        do_not_use: false,
                    },
                    HasClockCorrection {
                        sat: sat2,
                        nav_message: 0,
                        correction_m: Some(-0.60),
                        do_not_use: false,
                    },
                ],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&unavail_msg, reception2)
            .expect("ingest unavailable HAS MT1");

        // sat1: cached clock and pending high-rate state are cleared
        assert!(store.clock(sat1).is_none());
        assert!(store.pending_high_rate(sat1).is_none());
        assert_eq!(store.has_clock_superseded_epoch(sat1), Some(t_unavail));
        // Unavailable is NOT do-not-use: no exclusion marker is created
        assert!(store.has_exclusion(sat1).is_none());
        assert!(!store.is_satellite_excluded(sat1, t_unavail));

        // sat2: unrelated satellite keeps both values. The new clock replaces the
        // old one, while the orbit from the first message is untouched because
        // this message carries no orbit block.
        assert_eq!(
            store.clock(sat2).unwrap().c0_m.to_bits(),
            (-0.60_f64).to_bits()
        );
        let sat2_orbit = *store.orbit(sat2).expect("sat2 orbit retained");
        let sat2_clock = *store.clock(sat2).expect("sat2 clock retained");
        assert_eq!(sat2_orbit.iod_ssr, 1, "sat2 keeps the first message orbit");
        assert_eq!(sat2_orbit.solution.solution_id, 1);
        assert_eq!(sat2_clock.iod_ssr, 2, "sat2 takes the second message clock");
        assert_eq!(sat2_clock.solution.solution_id, 2);

        // Broadcast fallback view:
        // sat1 has no clock at all because the unavailable sentinel cleared it.
        // sat2 has both an orbit and a clock, but they come from different IOD
        // sets, so applying them together would mix provenance: strict output is
        // refused for both satellites, for different reasons.
        {
            let strict = SsrCorrectedEphemeris::new(&broadcast, &store);
            assert!(
                strict.corrected_state(sat1, t_unavail).is_none(),
                "sat1 has no clock after the unavailable sentinel"
            );
            assert!(
                strict.corrected_state(sat2, t_unavail).is_none(),
                "sat2 orbit IOD set 1 and clock IOD set 2 must not be combined"
            );

            // Neither refusal is a do-not-use exclusion, so broadcast fallback is
            // available for both satellites once the caller opts into it.
            let fallback =
                SsrCorrectedEphemeris::new(&broadcast, &store).with_fallback(SsrFallbackPolicy {
                    on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                    regional: RegionalPolicy::DeclineRegional,
                });
            assert!(fallback.corrected_state(sat1, t_unavail).is_some());
            assert!(fallback.corrected_state(sat2, t_unavail).is_some());
        }

        // 3. Stale usable arrival delivered afterward CANNOT revive superseded state
        // TOH corresponds to t_ref (epoch t_ref < t_unavail)
        let stale_usable_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 3,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(1.20),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&stale_usable_msg, reception2)
            .expect("ingest stale usable HAS MT1");

        // Stale usable clock MUST NOT be cached, and superseded epoch marker remains intact
        assert!(store.clock(sat1).is_none());
        assert_eq!(store.has_clock_superseded_epoch(sat1), Some(t_unavail));
        // Unrelated sat2 still preserved
        assert_eq!(
            store.clock(sat2).unwrap().c0_m.to_bits(),
            (-0.60_f64).to_bits()
        );

        // 4. Strictly later usable correction restores corrected output at t_ref + 60s
        let t_later = t_ref + 60.0;
        let reception3 =
            GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S + 60.0)
                .expect("GST reception 3");
        let later_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 60) as u16,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 4,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn, sat2.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![
                    HasOrbitCorrection {
                        sat: sat1,
                        nav_message: 0,
                        iode: record1.issue_of_data.expect("broadcast issue").issue,
                        radial_m: Some(1.0),
                        along_m: Some(0.0),
                        cross_m: Some(0.0),
                    },
                    HasOrbitCorrection {
                        sat: sat2,
                        nav_message: 0,
                        iode: record2.issue_of_data.expect("broadcast issue").issue,
                        radial_m: Some(1.0),
                        along_m: Some(0.0),
                        cross_m: Some(0.0),
                    },
                ],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![
                    HasClockCorrection {
                        sat: sat1,
                        nav_message: 0,
                        correction_m: Some(-0.45),
                        do_not_use: false,
                    },
                    HasClockCorrection {
                        sat: sat2,
                        nav_message: 0,
                        correction_m: Some(-0.70),
                        do_not_use: false,
                    },
                ],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&later_msg, reception3)
            .expect("ingest later HAS MT1");

        assert_eq!(
            store.clock(sat1).unwrap().c0_m.to_bits(),
            (-0.45_f64).to_bits()
        );
        assert_eq!(
            store.clock(sat2).unwrap().c0_m.to_bits(),
            (-0.70_f64).to_bits()
        );
        assert!(store.has_clock_superseded_epoch(sat1).is_none());
        {
            let strict = SsrCorrectedEphemeris::new(&broadcast, &store);
            assert!(strict.corrected_state(sat1, t_later).is_some());
            assert!(strict.corrected_state(sat2, t_later).is_some());
        }

        // 5. Stale unavailable cannot clear a newer usable correction
        // Ingest unavailable with TOH corresponding to t_ref + 45s (older than t_later = t_ref + 60s)
        let stale_unavail_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 45) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 5,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&stale_unavail_msg, reception3)
            .expect("ingest stale unavailable HAS MT1");

        // Stale unavailable MUST NOT clear the newer usable clock (-0.45m)
        assert_eq!(
            store.clock(sat1).unwrap().c0_m.to_bits(),
            (-0.45_f64).to_bits()
        );
        assert!(store.has_clock_superseded_epoch(sat1).is_none());
        {
            let strict = SsrCorrectedEphemeris::new(&broadcast, &store);
            assert!(strict.corrected_state(sat1, t_later).is_some());
        }
    }

    #[test]
    fn has_clock_unavailable_subset_and_do_not_use_interaction() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 31).unwrap();
        let t_ref = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let reception1 = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception 1");
        let record1 = broadcast
            .select_record_at(sat1, t_ref)
            .expect("broadcast record sat1");
        let record2 = broadcast
            .select_record_at(sat2, t_ref)
            .expect("broadcast record sat2");

        let mut store = SsrCorrectionStore::new();

        // 1. Initial message at t_ref:
        // sat1: do-not-use (+4095, exclusion active for 60s)
        // sat2: usable clock (-0.50m)
        let initial_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn, sat2.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5, // 60s
                records: vec![
                    HasOrbitCorrection {
                        sat: sat1,
                        nav_message: 0,
                        iode: record1.issue_of_data.expect("broadcast issue").issue,
                        radial_m: Some(1.0),
                        along_m: Some(0.0),
                        cross_m: Some(0.0),
                    },
                    HasOrbitCorrection {
                        sat: sat2,
                        nav_message: 0,
                        iode: record2.issue_of_data.expect("broadcast issue").issue,
                        radial_m: Some(1.0),
                        along_m: Some(0.0),
                        cross_m: Some(0.0),
                    },
                ],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5, // 60s
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![
                    HasClockCorrection {
                        sat: sat1,
                        nav_message: 0,
                        correction_m: None,
                        do_not_use: true,
                    },
                    HasClockCorrection {
                        sat: sat2,
                        nav_message: 0,
                        correction_m: Some(-0.50),
                        do_not_use: false,
                    },
                ],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&initial_msg, reception1)
            .expect("ingest initial HAS MT1");

        assert!(store.is_satellite_excluded(sat1, t_ref));
        assert!(store.has_exclusion(sat1).is_some());
        {
            let fallback =
                SsrCorrectedEphemeris::new(&broadcast, &store).with_fallback(SsrFallbackPolicy {
                    on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                    regional: RegionalPolicy::DeclineRegional,
                });
            // Fallback is blocked by active do-not-use exclusion!
            assert!(fallback.corrected_state(sat1, t_ref).is_none());
            assert!(fallback.corrected_state(sat2, t_ref).is_some());
        }

        // 2. Strictly newer unavailable indication delivered via clock_subset at t_ref + 20s
        // sat1: unavailable (-4096: correction_m None, do_not_use false)
        // Rule: unavailable NEVER erases a do-not-use exclusion!
        let t_sub = t_ref + 20.0;
        let reception2 =
            GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S + 20.0)
                .expect("GST reception 2");
        let subset_unavail_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 20) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 2,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn, sat2.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: None,
            clock_subset: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: false,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&subset_unavail_msg, reception2)
            .expect("ingest subset unavailable HAS MT1");

        // The do-not-use exclusion MUST NOT be erased by the unavailable record!
        assert!(store.has_exclusion(sat1).is_some());
        assert!(store.is_satellite_excluded(sat1, t_sub));
        assert!(store.clock(sat1).is_none());
        assert_eq!(store.has_clock_superseded_epoch(sat1), Some(t_sub));
        {
            let fallback =
                SsrCorrectedEphemeris::new(&broadcast, &store).with_fallback(SsrFallbackPolicy {
                    on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                    regional: RegionalPolicy::DeclineRegional,
                });
            // Still declined because exclusion remains active!
            assert!(fallback.corrected_state(sat1, t_sub).is_none());
            // Unrelated sat2 still intact
            assert!(fallback.corrected_state(sat2, t_sub).is_some());
        }

        // 3. After exclusion expires (at t_ref + 65s):
        // Exclusion is no longer active, but clock is unavailable (superseded).
        // Broadcast fallback resumes!
        let t_expired = t_ref + 65.0;
        assert!(!store.is_satellite_excluded(sat1, t_expired));
        assert!(store.clock(sat1).is_none());
        {
            let strict = SsrCorrectedEphemeris::new(&broadcast, &store);
            assert!(strict.corrected_state(sat1, t_expired).is_none());

            let fallback =
                SsrCorrectedEphemeris::new(&broadcast, &store).with_fallback(SsrFallbackPolicy {
                    on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                    regional: RegionalPolicy::DeclineRegional,
                });
            assert!(fallback.corrected_state(sat1, t_expired).is_some());
        }

        // 4. Stale usable arrival via clock_subset at t_ref + 10s (< t_sub = t_ref + 20s)
        // Delivered afterward: cannot revive superseded state!
        let stale_sub_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 10) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 3,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn, sat2.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: None,
            clock_subset: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(0.80),
                    do_not_use: false,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&stale_sub_msg, reception2)
            .expect("ingest stale subset HAS MT1");

        assert!(store.clock(sat1).is_none());
        assert_eq!(store.has_clock_superseded_epoch(sat1), Some(t_sub));

        // 5. Genuinely later usable correction via clock_subset at t_ref + 90s
        let t_later = t_ref + 90.0;
        let reception3 =
            GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S + 90.0)
                .expect("GST reception 3");
        let later_sub_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 90) as u16,
                mask: true,
                orbit: true,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 4,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn, sat2.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![
                    HasOrbitCorrection {
                        sat: sat1,
                        nav_message: 0,
                        iode: record1.issue_of_data.expect("broadcast issue").issue,
                        radial_m: Some(1.0),
                        along_m: Some(0.0),
                        cross_m: Some(0.0),
                    },
                    HasOrbitCorrection {
                        sat: sat2,
                        nav_message: 0,
                        iode: record2.issue_of_data.expect("broadcast issue").issue,
                        radial_m: Some(1.0),
                        along_m: Some(0.0),
                        cross_m: Some(0.0),
                    },
                ],
            }),
            clock_full_set: None,
            clock_subset: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(0.35),
                    do_not_use: false,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        let sat2_clock_before = store
            .clock(sat2)
            .copied()
            .expect("sat2 clock before later subset message");
        store
            .ingest_has_mt1(&later_sub_msg, reception3)
            .expect("ingest later subset HAS MT1");

        // Unrelated sat2 clock state must remain completely unaffected by the sat1 subset update
        assert_eq!(
            store.clock(sat2),
            Some(&sat2_clock_before),
            "sat2 clock state must remain unchanged across unrelated subset ingestion"
        );
        // Genuinely later usable correction clears exclusion and superseded marker, and restores corrected output
        assert!(store.has_exclusion(sat1).is_none());
        assert!(store.has_clock_superseded_epoch(sat1).is_none());
        assert_eq!(
            store.clock(sat1).unwrap().c0_m.to_bits(),
            0.35_f64.to_bits()
        );
        {
            let strict = SsrCorrectedEphemeris::new(&broadcast, &store);
            assert!(
                strict.corrected_state(sat1, t_later).is_some(),
                "sat1 corrected state restored at later epoch"
            );
            // sat2 strict corrected state at t_later (t_ref + 90s) is correctly refused:
            // sat2 clock was issued at t_ref with VI 60s and iod_set_id 1, whereas the later message
            // updated orbit with iod_set_id 4 at t_ref + 90s without a new sat2 clock.
            // Both freshness (age 90s > update_interval 60s) and solution matching
            // (iod_ssr / solution_id 4 vs 1) reject it.
            assert!(
                strict.corrected_state(sat2, t_later).is_none(),
                "sat2 strict corrected state must be refused due to clock expiry and solution mismatch"
            );

            // With fallback enabled, broadcast fallback succeeds because sat2 is not excluded
            let fallback =
                SsrCorrectedEphemeris::new(&broadcast, &store).with_fallback(SsrFallbackPolicy {
                    on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                    regional: RegionalPolicy::DeclineRegional,
                });
            assert!(
                fallback.corrected_state(sat2, t_later).is_some(),
                "sat2 broadcast fallback succeeds when correction is expired/mismatched"
            );
        }
    }

    #[test]
    fn has_do_not_use_same_epoch_retains_maximum_validity_interval_regardless_of_block_or_arrival_order(
    ) {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 31).unwrap();
        let t_ref = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let reception = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception");
        let toh_s = (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16;

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat1.prn, sat2.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        // Case A: Both block types together in one message.
        // clock_full_set has VI index 5 = 60s (longer VI).
        // clock_subset has VI index 0 = 5s (shorter VI).
        // Conservative equal-epoch receiver arbitration must retain max VI (60s).
        // Relevant official source: Galileo HAS SIS ICD Sections 5.2.2.1, 5.2.3.2 (Table 31), 5.2.4.1, and 7.7:
        // https://www.gsc-europa.eu/sites/default/files/sites/all/files/Galileo-HAS-SIS-ICD_in_force.pdf
        let msg_both = HasMt1Message {
            header: HasMt1Header {
                toh_s,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5, // 60s
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![
                    HasClockCorrection {
                        sat: sat1,
                        nav_message: 0,
                        correction_m: None,
                        do_not_use: true,
                    },
                    HasClockCorrection {
                        sat: sat2,
                        nav_message: 0,
                        correction_m: Some(-0.50),
                        do_not_use: false,
                    },
                ],
            }),
            clock_subset: Some(HasClockBlock {
                validity_interval: 0, // 5s
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: true,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let mut store_both = SsrCorrectionStore::new();
        store_both
            .ingest_has_mt1(&msg_both, reception)
            .expect("ingest both blocks HAS MT1");

        assert_eq!(
            store_both.has_exclusion(sat1).unwrap().validity_interval_s,
            60.0
        );

        let fallback_policy = SsrFallbackPolicy {
            on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
            regional: RegionalPolicy::DeclineRegional,
        };

        // Query point 1: beyond shorter VI (5s) but inside longer VI (60s): t_ref + 10s
        let t_inside = t_ref + 10.0;
        assert!(store_both.is_satellite_excluded(sat1, t_inside));
        {
            let strict = SsrCorrectedEphemeris::new(&broadcast, &store_both);
            assert!(strict.corrected_state(sat1, t_inside).is_none());
            let fallback = SsrCorrectedEphemeris::new(&broadcast, &store_both)
                .with_fallback(fallback_policy.clone());
            assert!(
                fallback.corrected_state(sat1, t_inside).is_none(),
                "broadcast fallback must remain excluded inside retained VI"
            );
        }

        // Query point 2: at longer endpoint: t_ref + 60s (inclusive per SSR <= convention)
        let t_endpoint = t_ref + 60.0;
        assert!(store_both.is_satellite_excluded(sat1, t_endpoint));
        {
            let strict = SsrCorrectedEphemeris::new(&broadcast, &store_both);
            assert!(strict.corrected_state(sat1, t_endpoint).is_none());
            let fallback = SsrCorrectedEphemeris::new(&broadcast, &store_both)
                .with_fallback(fallback_policy.clone());
            assert!(
                fallback.corrected_state(sat1, t_endpoint).is_none(),
                "broadcast fallback must remain excluded at retained VI endpoint"
            );
        }

        // Query point 3: just after longer endpoint: t_ref + 60.001s
        let t_after = t_ref + 60.001;
        assert!(!store_both.is_satellite_excluded(sat1, t_after));
        {
            let strict = SsrCorrectedEphemeris::new(&broadcast, &store_both);
            assert!(strict.corrected_state(sat1, t_after).is_none());
            let fallback = SsrCorrectedEphemeris::new(&broadcast, &store_both)
                .with_fallback(fallback_policy.clone());
            assert!(
                fallback.corrected_state(sat1, t_after).is_some(),
                "broadcast fallback must resume after retained VI expires"
            );
        }

        // Case B: Arrival order full-set first (VI 60s) then subset second (VI 5s)
        let msg_fs = HasMt1Message {
            header: HasMt1Header {
                toh_s,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 2,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5, // 60s
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: true,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let msg_sub = HasMt1Message {
            header: HasMt1Header {
                toh_s,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 3,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: Some(HasClockBlock {
                validity_interval: 0, // 5s
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: true,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let mut store_fs_first = SsrCorrectionStore::new();
        store_fs_first
            .ingest_has_mt1(&msg_fs, reception)
            .expect("ingest full-set first");
        store_fs_first
            .ingest_has_mt1(&msg_sub, reception)
            .expect("ingest subset second");

        assert_eq!(
            store_fs_first
                .has_exclusion(sat1)
                .unwrap()
                .validity_interval_s,
            60.0
        );
        assert!(store_fs_first.is_satellite_excluded(sat1, t_inside));
        assert!(store_fs_first.is_satellite_excluded(sat1, t_endpoint));
        assert!(!store_fs_first.is_satellite_excluded(sat1, t_after));
        {
            let fallback = SsrCorrectedEphemeris::new(&broadcast, &store_fs_first)
                .with_fallback(fallback_policy.clone());
            assert!(fallback.corrected_state(sat1, t_inside).is_none());
            assert!(fallback.corrected_state(sat1, t_endpoint).is_none());
            assert!(fallback.corrected_state(sat1, t_after).is_some());
        }

        // Case C: Reversed arrival order: subset first (VI 5s) then full-set second (VI 60s)
        let mut store_sub_first = SsrCorrectionStore::new();
        store_sub_first
            .ingest_has_mt1(&msg_sub, reception)
            .expect("ingest subset first");
        // Prior to full-set arrival, exclusion has 5s VI
        assert_eq!(
            store_sub_first
                .has_exclusion(sat1)
                .unwrap()
                .validity_interval_s,
            5.0
        );
        store_sub_first
            .ingest_has_mt1(&msg_fs, reception)
            .expect("ingest full-set second");

        // After same-epoch full-set arrival, max VI (60s) is retained
        assert_eq!(
            store_sub_first
                .has_exclusion(sat1)
                .unwrap()
                .validity_interval_s,
            60.0
        );
        assert!(store_sub_first.is_satellite_excluded(sat1, t_inside));
        assert!(store_sub_first.is_satellite_excluded(sat1, t_endpoint));
        assert!(!store_sub_first.is_satellite_excluded(sat1, t_after));
        {
            let fallback = SsrCorrectedEphemeris::new(&broadcast, &store_sub_first)
                .with_fallback(fallback_policy);
            assert!(fallback.corrected_state(sat1, t_inside).is_none());
            assert!(fallback.corrected_state(sat1, t_endpoint).is_none());
            assert!(fallback.corrected_state(sat1, t_after).is_some());
        }
    }

    #[test]
    fn has_reserved_clock_vi_preflight_prevents_partial_cache_mutation() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 31).unwrap();
        let t_ref = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let reception = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception");
        let record1 = broadcast
            .select_record_at(sat1, t_ref)
            .expect("broadcast record sat1");
        let record2 = broadcast
            .select_record_at(sat2, t_ref)
            .expect("broadcast record sat2");

        let mut store = SsrCorrectionStore::new();
        let baseline_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn, sat2.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![
                    HasOrbitCorrection {
                        sat: sat1,
                        nav_message: 0,
                        iode: record1.issue_of_data.expect("broadcast issue").issue,
                        radial_m: Some(1.0),
                        along_m: Some(0.0),
                        cross_m: Some(0.0),
                    },
                    HasOrbitCorrection {
                        sat: sat2,
                        nav_message: 0,
                        iode: record2.issue_of_data.expect("broadcast issue").issue,
                        radial_m: Some(2.0),
                        along_m: Some(0.0),
                        cross_m: Some(0.0),
                    },
                ],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![
                    HasClockCorrection {
                        sat: sat1,
                        nav_message: 0,
                        correction_m: Some(-0.50),
                        do_not_use: false,
                    },
                    HasClockCorrection {
                        sat: sat2,
                        nav_message: 0,
                        correction_m: Some(0.25),
                        do_not_use: false,
                    },
                ],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&baseline_msg, reception)
            .expect("ingest baseline HAS MT1");

        let expected_store = store.clone();

        // 1. Valid orbit + reserved full-set clock VI (index 15 is reserved per Table 23)
        let msg_valid_orbit_reserved_fs = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 10) as u16,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 2,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5, // valid orbit VI
                records: vec![HasOrbitCorrection {
                    sat: sat1,
                    nav_message: 0,
                    iode: record1.issue_of_data.expect("broadcast issue").issue,
                    radial_m: Some(99.0),
                    along_m: Some(0.0),
                    cross_m: Some(0.0),
                }],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 15, // RESERVED clock VI
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(99.0),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        let err1 = store
            .ingest_has_mt1(&msg_valid_orbit_reserved_fs, reception)
            .unwrap_err();
        assert!(matches!(&err1, Error::Parse(message) if message == "HAS clock VI is reserved"));
        assert_eq!(
            store, expected_store,
            "entire store must remain unchanged when full-set clock VI is reserved"
        );

        // 2. Valid full-set clock VI + reserved subset clock VI
        let msg_valid_fs_reserved_sub = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 20) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 3,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5, // valid full-set VI
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(88.0),
                    do_not_use: false,
                }],
            }),
            clock_subset: Some(HasClockBlock {
                validity_interval: 15, // RESERVED subset VI
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(77.0),
                    do_not_use: false,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        let err2 = store
            .ingest_has_mt1(&msg_valid_fs_reserved_sub, reception)
            .unwrap_err();
        assert!(matches!(&err2, Error::Parse(message) if message == "HAS clock VI is reserved"));
        assert_eq!(
            store, expected_store,
            "entire store must remain unchanged when subset clock VI is reserved"
        );
    }

    #[test]
    fn has_usable_clock_equal_epoch_replaces_and_older_arrival_refused() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t_ref = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let reception1 = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception 1");

        let mut store = SsrCorrectionStore::new();

        // 1. Initial message at t_ref with usable clock c0_m = -0.50
        let msg_initial = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(-0.50),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&msg_initial, reception1)
            .expect("ingest initial HAS MT1");

        assert_eq!(
            store.clock(sat1).unwrap().c0_m.to_bits(),
            (-0.50_f64).to_bits()
        );
        assert_eq!(
            store.clock(sat1).unwrap().ref_epoch_j2000_s.to_bits(),
            t_ref.to_bits()
        );

        // 2. Equal-epoch usable update at t_ref with c0_m = -0.75
        // Replaces the cached clock (equal-epoch replacement preserved via older-only guard)
        let msg_equal_epoch = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 2,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: None,
            clock_subset: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(-0.75),
                    do_not_use: false,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&msg_equal_epoch, reception1)
            .expect("ingest equal-epoch HAS MT1");

        assert_eq!(
            store.clock(sat1).unwrap().c0_m.to_bits(),
            (-0.75_f64).to_bits(),
            "equal-epoch usable arrival must replace cached clock"
        );
        assert_eq!(
            store.clock(sat1).unwrap().ref_epoch_j2000_s.to_bits(),
            t_ref.to_bits()
        );

        // 3. Strictly newer usable clock at t_ref + 30.0s with c0_m = -0.90
        let t_newer = t_ref + 30.0;
        let reception2 =
            GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S + 30.0)
                .expect("GST reception 2");
        let msg_newer = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 30) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 3,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(-0.90),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&msg_newer, reception2)
            .expect("ingest newer HAS MT1");

        assert_eq!(
            store.clock(sat1).unwrap().c0_m.to_bits(),
            (-0.90_f64).to_bits()
        );
        assert_eq!(
            store.clock(sat1).unwrap().ref_epoch_j2000_s.to_bits(),
            t_newer.to_bits()
        );

        // 4. Stale/older usable arrival delivered afterward (epoch t_ref < t_newer)
        // Delivered at reception3 (t_ref + 35s), but TOH corresponds to t_ref
        let reception3 =
            GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S + 35.0)
                .expect("GST reception 3");
        let msg_stale_older = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 4,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![sat1.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(1.20),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1(&msg_stale_older, reception3)
            .expect("ingest stale older HAS MT1");

        // The newer cached clock (-0.90m at t_newer) MUST NOT be replaced by the older arrival
        assert_eq!(
            store.clock(sat1).unwrap().c0_m.to_bits(),
            (-0.90_f64).to_bits(),
            "older usable arrival must not replace newer cached clock"
        );
        assert_eq!(
            store.clock(sat1).unwrap().ref_epoch_j2000_s.to_bits(),
            t_newer.to_bits()
        );
    }

    #[test]
    fn has_clock_equal_epoch_usable_unavailable_receiver_arbitration_and_ordering() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t_ref = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let reception1 = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception 1");
        let toh_s = (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16;

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat1.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        // Usable clock message at t_ref (c0 = -0.50m)
        let usable_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(-0.50),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        // Unavailable clock message at t_ref
        let unavail_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 2,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        // Case 1: Usable arrives first, then equal Unavailable arrives second
        // Bounded receiver arbitration policy: usable takes priority over unavailable at equal epochs.
        let mut store_usable_first = SsrCorrectionStore::new();
        store_usable_first
            .ingest_has_mt1(&usable_msg, reception1)
            .expect("ingest usable first");
        assert_eq!(
            store_usable_first.clock(sat1).unwrap().c0_m.to_bits(),
            (-0.50_f64).to_bits()
        );
        store_usable_first
            .ingest_has_mt1(&unavail_msg, reception1)
            .expect("ingest equal unavailable second");
        assert_eq!(
            store_usable_first.clock(sat1).unwrap().c0_m.to_bits(),
            (-0.50_f64).to_bits(),
            "equal-epoch unavailable must not clear existing usable clock"
        );
        assert!(
            store_usable_first
                .has_clock_superseded_epoch(sat1)
                .is_none(),
            "superseded epoch must not be set when equal unavailable is refused"
        );

        // Case 2: Unavailable arrives first, then equal Usable arrives second
        // Bounded receiver arbitration policy: usable takes priority over unavailable at equal epochs.
        // Unavailable supersession marker is cleared and usable clock is stored.
        let mut store_unavail_first = SsrCorrectionStore::new();
        store_unavail_first
            .ingest_has_mt1(&unavail_msg, reception1)
            .expect("ingest unavailable first");
        assert!(store_unavail_first.clock(sat1).is_none());
        assert_eq!(
            store_unavail_first.has_clock_superseded_epoch(sat1),
            Some(t_ref)
        );
        store_unavail_first
            .ingest_has_mt1(&usable_msg, reception1)
            .expect("ingest equal usable second");
        assert_eq!(
            store_unavail_first.clock(sat1).unwrap().c0_m.to_bits(),
            (-0.50_f64).to_bits(),
            "equal-epoch usable must clear unavailable supersession marker and store usable clock"
        );
        assert!(
            store_unavail_first
                .has_clock_superseded_epoch(sat1)
                .is_none(),
            "superseded marker must be cleared by equal-epoch usable arrival"
        );

        // Case 3: Strictly newer unavailable indication (t_ref + 20s) clears older usable clock (t_ref)
        let t_newer = t_ref + 20.0;
        let reception2 =
            GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S + 20.0)
                .expect("GST reception 2");
        let unavail_newer_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 20) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 3,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let mut store_newer_unavail = SsrCorrectionStore::new();
        store_newer_unavail
            .ingest_has_mt1(&usable_msg, reception1)
            .expect("ingest usable");
        store_newer_unavail
            .ingest_has_mt1(&unavail_newer_msg, reception2)
            .expect("ingest newer unavailable");
        assert!(
            store_newer_unavail.clock(sat1).is_none(),
            "strictly newer unavailable must clear older usable clock"
        );
        assert_eq!(
            store_newer_unavail.has_clock_superseded_epoch(sat1),
            Some(t_newer)
        );

        // Case 4: Strictly older usable (t_ref) arriving after newer unavailable (t_ref + 20s) is refused
        let mut store_older_usable = SsrCorrectionStore::new();
        store_older_usable
            .ingest_has_mt1(&unavail_newer_msg, reception2)
            .expect("ingest newer unavailable");
        store_older_usable
            .ingest_has_mt1(&usable_msg, reception2)
            .expect("ingest older usable delivered late");
        assert!(
            store_older_usable.clock(sat1).is_none(),
            "strictly older usable must remain blocked by newer unavailable supersession marker"
        );
        assert_eq!(
            store_older_usable.has_clock_superseded_epoch(sat1),
            Some(t_newer)
        );

        // Case 5: Do-not-use still wins over usable at equal epochs in BOTH arrival orders
        let dnu_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 4,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: true,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        // Order A: DNU first, then equal Usable second
        let mut store_dnu_first = SsrCorrectionStore::new();
        store_dnu_first
            .ingest_has_mt1(&dnu_msg, reception1)
            .expect("ingest DNU first");
        store_dnu_first
            .ingest_has_mt1(&usable_msg, reception1)
            .expect("ingest equal usable second");
        assert!(
            store_dnu_first.is_satellite_excluded(sat1, t_ref),
            "DNU must win over usable at equal epoch when DNU arrives first"
        );
        assert!(store_dnu_first.clock(sat1).is_none());

        // Order B: Usable first, then equal DNU second
        let mut store_usable_dnu = SsrCorrectionStore::new();
        store_usable_dnu
            .ingest_has_mt1(&usable_msg, reception1)
            .expect("ingest usable first");
        store_usable_dnu
            .ingest_has_mt1(&dnu_msg, reception1)
            .expect("ingest equal DNU second");
        assert!(
            store_usable_dnu.is_satellite_excluded(sat1, t_ref),
            "DNU must win over usable at equal epoch when Usable arrives first"
        );
        assert!(store_usable_dnu.clock(sat1).is_none());
    }

    #[test]
    fn has_unavailable_and_do_not_use_arrival_ordering_and_superseded_preservation() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 31).unwrap();
        let t0 = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let t10 = t0 + 10.0;
        let t5 = t0 + 5.0;
        let t15 = t0 + 15.0;
        let t_inside = t0 + 20.0;
        let t_expired = t0 + 65.0;

        let record1 = broadcast
            .select_record_at(sat1, t0)
            .expect("broadcast record sat1");
        let record2 = broadcast
            .select_record_at(sat2, t0)
            .expect("broadcast record sat2");

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat1.prn, sat2.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        let fallback_policy = SsrFallbackPolicy {
            on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
            regional: RegionalPolicy::DeclineRegional,
        };

        // 1. Message at T0:
        // Orbit for sat1 and sat2 (VI 60s)
        // Clock for sat1: Do-Not-Use (VI 60s)
        // Clock for sat2: Usable (-0.50m)
        let msg_dnu_t0 = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5, // 60s
                records: vec![
                    HasOrbitCorrection {
                        sat: sat1,
                        nav_message: 0,
                        iode: record1.issue_of_data.expect("broadcast issue").issue,
                        radial_m: Some(1.0),
                        along_m: Some(0.0),
                        cross_m: Some(0.0),
                    },
                    HasOrbitCorrection {
                        sat: sat2,
                        nav_message: 0,
                        iode: record2.issue_of_data.expect("broadcast issue").issue,
                        radial_m: Some(1.0),
                        along_m: Some(0.0),
                        cross_m: Some(0.0),
                    },
                ],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5, // 60s
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![
                    HasClockCorrection {
                        sat: sat1,
                        nav_message: 0,
                        correction_m: None,
                        do_not_use: true,
                    },
                    HasClockCorrection {
                        sat: sat2,
                        nav_message: 0,
                        correction_m: Some(-0.50),
                        do_not_use: false,
                    },
                ],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        assert!(msg_dnu_t0.encode().is_ok());

        // 2. Message at T10:
        // Clock for sat1: Unavailable (sentinel -4096: correction_m None, do_not_use false)
        let msg_unavail_t10 = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 10) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 2,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: false,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        assert!(msg_unavail_t10.encode().is_ok());

        // 3. Stale usable arrival at T5 (delivered late):
        let msg_usable_t5 = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 5) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 3,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(0.80),
                    do_not_use: false,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        assert!(msg_usable_t5.encode().is_ok());

        // 4. Genuinely later usable arrival at T15:
        let msg_usable_t15 = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 15) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1, // matches initial orbit iod_set_id 1
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(-0.30),
                    do_not_use: false,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        assert!(msg_usable_t15.encode().is_ok());

        let reception_t0 = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception t0");
        let reception_t10 =
            GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S + 10.0)
                .expect("GST reception t10");
        let reception_t15 =
            GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S + 15.0)
                .expect("GST reception t15");
        let reception_t20 =
            GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S + 20.0)
                .expect("GST reception t20");

        assert_eq!(
            has_mt1_reference_j2000_s(reception_t10, msg_usable_t5.header.toh_s).unwrap(),
            t5
        );

        // === Order A: DNU at T0 first, then Unavailable at T10 second ===
        let mut store_a = SsrCorrectionStore::new();
        store_a
            .ingest_has_mt1(&msg_dnu_t0, reception_t0)
            .expect("Order A: ingest DNU T0");
        store_a
            .ingest_has_mt1(&msg_unavail_t10, reception_t10)
            .expect("Order A: ingest unavail T10");

        // Verify unrelated sat2 clock is preserved:
        assert_eq!(
            store_a.clock(sat2).unwrap().c0_m.to_bits(),
            (-0.50_f64).to_bits(),
            "Order A: unrelated sat2 clock must remain preserved"
        );

        // Verify exclusion marker and superseded epoch retention:
        assert_eq!(
            store_a.has_exclusion(sat1).unwrap().validity_interval_s,
            60.0
        );
        assert_eq!(store_a.has_exclusion(sat1).unwrap().ref_epoch_j2000_s, t0);
        assert_eq!(
            store_a.has_clock_superseded_epoch(sat1),
            Some(t10),
            "Order A must retain newer unavailable timestamp"
        );
        assert!(store_a.is_satellite_excluded(sat1, t_inside));
        {
            let strict = SsrCorrectedEphemeris::new(&broadcast, &store_a);
            assert!(strict.corrected_state(sat1, t_inside).is_none());
            let fallback = SsrCorrectedEphemeris::new(&broadcast, &store_a)
                .with_fallback(fallback_policy.clone());
            assert!(
                fallback.corrected_state(sat1, t_inside).is_none(),
                "broadcast fallback must remain declined inside VI"
            );
        }

        // Stale usable arrival at T5: refused by newer superseded timestamp T10
        store_a
            .ingest_has_mt1(&msg_usable_t5, reception_t10)
            .expect("Order A: ingest stale usable T5");
        assert!(
            store_a.clock(sat1).is_none(),
            "Order A: stale usable T5 must not restore clock"
        );
        assert!(
            store_a.has_exclusion(sat1).is_some(),
            "Order A: stale usable T5 must not clear exclusion"
        );
        assert_eq!(store_a.has_clock_superseded_epoch(sat1), Some(t10));

        // After VI expiration: fallback resumes
        assert!(!store_a.is_satellite_excluded(sat1, t_expired));
        {
            let strict = SsrCorrectedEphemeris::new(&broadcast, &store_a);
            assert!(strict.corrected_state(sat1, t_expired).is_none());
            let fallback = SsrCorrectedEphemeris::new(&broadcast, &store_a)
                .with_fallback(fallback_policy.clone());
            assert!(
                fallback.corrected_state(sat1, t_expired).is_some(),
                "Order A: broadcast fallback resumes after exclusion expiration"
            );
        }

        // Genuinely later usable arrival at T15: clears exclusion and restores corrected state
        store_a
            .ingest_has_mt1(&msg_usable_t15, reception_t15)
            .expect("Order A: ingest later usable T15");
        assert!(store_a.has_exclusion(sat1).is_none());
        assert!(store_a.has_clock_superseded_epoch(sat1).is_none());
        assert_eq!(
            store_a.clock(sat1).unwrap().c0_m.to_bits(),
            (-0.30_f64).to_bits()
        );
        {
            let strict = SsrCorrectedEphemeris::new(&broadcast, &store_a);
            assert!(
                strict.corrected_state(sat1, t15).is_some(),
                "Order A: strict corrected state restored by genuinely later usable"
            );
        }

        // === Order B: Unavailable at T10 first, then DNU at T0 second ===
        let mut store_b = SsrCorrectionStore::new();
        store_b
            .ingest_has_mt1(&msg_unavail_t10, reception_t10)
            .expect("Order B: ingest unavail T10");
        assert_eq!(store_b.has_clock_superseded_epoch(sat1), Some(t10));
        assert!(store_b.has_exclusion(sat1).is_none());

        store_b
            .ingest_has_mt1(&msg_dnu_t0, reception_t10)
            .expect("Order B: ingest DNU T0");

        // Verify unrelated sat2 clock is preserved:
        assert_eq!(
            store_b.clock(sat2).unwrap().c0_m.to_bits(),
            (-0.50_f64).to_bits(),
            "Order B: unrelated sat2 clock must remain preserved"
        );

        // Order B must accept the DNU and retain the newer unavailable timestamp:
        assert_eq!(
            store_b.has_exclusion(sat1).unwrap().validity_interval_s,
            60.0
        );
        assert_eq!(store_b.has_exclusion(sat1).unwrap().ref_epoch_j2000_s, t0);
        assert_eq!(
            store_b.has_clock_superseded_epoch(sat1),
            Some(t10),
            "Order B must preserve newer unavailable timestamp when accepting older DNU"
        );
        assert!(store_b.is_satellite_excluded(sat1, t_inside));
        {
            let strict = SsrCorrectedEphemeris::new(&broadcast, &store_b);
            assert!(strict.corrected_state(sat1, t_inside).is_none());
            let fallback = SsrCorrectedEphemeris::new(&broadcast, &store_b)
                .with_fallback(fallback_policy.clone());
            assert!(
                fallback.corrected_state(sat1, t_inside).is_none(),
                "broadcast fallback must remain declined inside VI"
            );
        }

        // Stale usable arrival at T5: refused by preserved newer superseded timestamp T10
        store_b
            .ingest_has_mt1(&msg_usable_t5, reception_t10)
            .expect("Order B: ingest stale usable T5");
        assert!(
            store_b.clock(sat1).is_none(),
            "Order B: stale usable T5 must not restore clock"
        );
        assert!(
            store_b.has_exclusion(sat1).is_some(),
            "Order B: stale usable T5 must not clear exclusion"
        );
        assert_eq!(store_b.has_clock_superseded_epoch(sat1), Some(t10));

        // After VI expiration: fallback resumes
        assert!(!store_b.is_satellite_excluded(sat1, t_expired));
        {
            let strict = SsrCorrectedEphemeris::new(&broadcast, &store_b);
            assert!(strict.corrected_state(sat1, t_expired).is_none());
            let fallback = SsrCorrectedEphemeris::new(&broadcast, &store_b)
                .with_fallback(fallback_policy.clone());
            assert!(
                fallback.corrected_state(sat1, t_expired).is_some(),
                "Order B: broadcast fallback resumes after exclusion expiration"
            );
        }

        // Genuinely later usable arrival at T15: clears exclusion and restores corrected state
        store_b
            .ingest_has_mt1(&msg_usable_t15, reception_t15)
            .expect("Order B: ingest later usable T15");
        assert!(store_b.has_exclusion(sat1).is_none());
        assert!(store_b.has_clock_superseded_epoch(sat1).is_none());
        assert_eq!(
            store_b.clock(sat1).unwrap().c0_m.to_bits(),
            (-0.30_f64).to_bits()
        );
        {
            let strict = SsrCorrectedEphemeris::new(&broadcast, &store_b);
            assert!(
                strict.corrected_state(sat1, t15).is_some(),
                "Order B: strict corrected state restored by genuinely later usable"
            );
        }

        // Surrounding case: DNU at T20 clears older unavailable at T10
        let mut store_c = SsrCorrectionStore::new();
        store_c
            .ingest_has_mt1(&msg_unavail_t10, reception_t10)
            .expect("ingest unavail T10");
        assert_eq!(store_c.has_clock_superseded_epoch(sat1), Some(t10));
        let msg_dnu_t20 = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 20) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 5,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: true,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        assert!(msg_dnu_t20.encode().is_ok());
        assert_eq!(
            has_mt1_reference_j2000_s(reception_t20, msg_dnu_t20.header.toh_s).unwrap(),
            t0 + 20.0
        );
        store_c
            .ingest_has_mt1(&msg_dnu_t20, reception_t20)
            .expect("ingest DNU T20");
        assert!(
            store_c.has_clock_superseded_epoch(sat1).is_none(),
            "newer DNU at T20 must clear older unavailable superseded epoch"
        );
        assert_eq!(
            store_c.has_exclusion(sat1).unwrap().ref_epoch_j2000_s,
            t0 + 20.0
        );
    }

    #[test]
    fn has_clock_unavailable_provenance_aware_pending_high_rate_invalidation() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t_ref = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let reception1 = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception 1");
        let reception2 =
            GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S + 30.0)
                .expect("GST reception 2");

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat1.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        // HAS unavailable message for sat1 at t_ref + 30s
        let has_unavail_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 30) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 2,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        // 1. RTCM pending high-rate clock must NOT be cleared by HAS unavailable
        let mut store1 = SsrCorrectionStore::new();
        let mut clock_hdr = header(SsrKind::Clock);
        clock_hdr.epoch_time_s = REAL_SSR_EPOCH_TOW_S as u32;
        clock_hdr.update_interval = 5;
        let rtcm_clock = SsrMessage {
            message_number: 1058,
            system: GnssSystem::Gps,
            kind: SsrKind::Clock,
            header: clock_hdr,
            orbit: Vec::new(),
            clock: vec![SsrClockRecord {
                satellite_id: sat1.prn,
                c0: 2500,
                c1: 0,
                c2: 0,
            }],
            code_bias: Vec::new(),
            phase_bias: Vec::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };

        let mut hr_hdr = header(SsrKind::HighRateClock);
        hr_hdr.epoch_time_s = REAL_SSR_EPOCH_TOW_S as u32;
        hr_hdr.update_interval = 0;
        let rtcm_hr = SsrMessage {
            message_number: 1062,
            system: GnssSystem::Gps,
            kind: SsrKind::HighRateClock,
            header: hr_hdr,
            orbit: Vec::new(),
            clock: vec![SsrClockRecord {
                satellite_id: sat1.prn,
                c0: 500,
                c1: 0,
                c2: 0,
            }],
            code_bias: Vec::new(),
            phase_bias: Vec::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };

        store1
            .ingest_ssr(&rtcm_clock, reception1)
            .expect("ingest RTCM clock");
        store1
            .ingest_ssr(&rtcm_hr, reception1)
            .expect("ingest RTCM high-rate clock");
        assert!(store1.clock(sat1).is_some());
        assert!(store1.pending_high_rate(sat1).is_some());

        // Ingest HAS unavailable at t_ref + 30s
        store1
            .ingest_has_mt1(&has_unavail_msg, reception2)
            .expect("ingest HAS unavailable");

        // RTCM base clock and RTCM pending high-rate are preserved!
        assert!(
            store1.clock(sat1).is_some(),
            "RTCM base clock must be preserved across HAS unavailable indication"
        );
        assert!(
            store1.pending_high_rate(sat1).is_some(),
            "RTCM pending high-rate must be preserved across HAS unavailable indication"
        );
        assert_eq!(
            store1.has_clock_superseded_epoch(sat1),
            Some(t_ref + 30.0),
            "HAS superseded epoch marker must be tracked for HAS stream"
        );

        // 2. Genuinely newer pending HAS high-rate must NOT be cleared by older HAS unavailable
        let mut store2 = SsrCorrectionStore::new();
        store2
            .corrections
            .entry(sat1)
            .or_default()
            .pending_high_rate = Some(SsrHighRateClock {
            solution: SsrSolution {
                source: SsrSource::GalileoHas,
                provider_id: 1,
                solution_id: 1,
            },
            iod_ssr: 1,
            c0_m: 0.15,
            ref_epoch_j2000_s: t_ref + 45.0, // newer than unavailable at t_ref + 30s
            transmitted_epoch_j2000_s: t_ref + 45.0,
            update_interval_s: 5.0,
        });
        store2
            .ingest_has_mt1(&has_unavail_msg, reception2)
            .expect("ingest HAS unavailable");
        assert!(
            store2.pending_high_rate(sat1).is_some(),
            "genuinely newer HAS pending high-rate must be preserved"
        );

        // 3. Older or equal pending HAS high-rate IS cleared by HAS unavailable
        let mut store3 = SsrCorrectionStore::new();
        store3
            .corrections
            .entry(sat1)
            .or_default()
            .pending_high_rate = Some(SsrHighRateClock {
            solution: SsrSolution {
                source: SsrSource::GalileoHas,
                provider_id: 1,
                solution_id: 1,
            },
            iod_ssr: 1,
            c0_m: 0.15,
            ref_epoch_j2000_s: t_ref + 20.0, // older than unavailable at t_ref + 30s
            transmitted_epoch_j2000_s: t_ref + 20.0,
            update_interval_s: 5.0,
        });
        store3
            .ingest_has_mt1(&has_unavail_msg, reception2)
            .expect("ingest HAS unavailable");
        assert!(
            store3.pending_high_rate(sat1).is_none(),
            "older HAS pending high-rate must be cleared by HAS unavailable"
        );

        // 4. Removing a superseded HAS base clock removes its attached high-rate correction
        let mut store4 = SsrCorrectionStore::new();
        let has_usable_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: Some(-0.40),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store4
            .ingest_has_mt1(&has_usable_msg, reception1)
            .expect("ingest HAS usable clock");
        // Attach high-rate correction to HAS base clock:
        if let Some(clock) = store4
            .corrections
            .get_mut(&sat1)
            .and_then(|entry| entry.clock.as_mut())
        {
            clock.high_rate = Some(SsrHighRateClock {
                solution: SsrSolution {
                    source: SsrSource::GalileoHas,
                    provider_id: 1,
                    solution_id: 1,
                },
                iod_ssr: 1,
                c0_m: 0.05,
                ref_epoch_j2000_s: t_ref,
                transmitted_epoch_j2000_s: t_ref,
                update_interval_s: 5.0,
            });
        }
        assert!(store4.clock(sat1).unwrap().high_rate.is_some());

        // Ingest HAS unavailable:
        store4
            .ingest_has_mt1(&has_unavail_msg, reception2)
            .expect("ingest HAS unavailable");
        assert!(
            store4.clock(sat1).is_none(),
            "superseded HAS base clock must be cleared, removing attached high-rate value"
        );

        // 5. A do-not-use exclusion blocks use of the satellite without destroying RTCM state.
        //    Per Galileo HAS SIS ICD Section 5.2.2.1 the indication is bounded by its own
        //    validity interval; it is an instruction not to use the satellite while it is
        //    active, not a licence to discard another provider's still-valid corrections.
        let mut store5 = SsrCorrectionStore::new();
        store5
            .ingest_ssr(&rtcm_hr, reception1)
            .expect("ingest RTCM high-rate");
        assert!(store5.pending_high_rate(sat1).is_some());

        let has_dnu_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((REAL_SSR_EPOCH_TOW_S as u32 % 3600) + 30) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 5,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: true,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        let hr_before = store5
            .pending_high_rate(sat1)
            .copied()
            .expect("RTCM pending high-rate before DNU");
        store5
            .ingest_has_mt1(&has_dnu_msg, reception2)
            .expect("ingest HAS DNU");
        let hr_after = store5
            .pending_high_rate(sat1)
            .copied()
            .expect("RTCM pending high-rate is another provider's state and must survive DNU");
        assert_eq!(
            hr_after, hr_before,
            "DNU must leave RTCM pending high-rate byte-identical, not merely present"
        );
        assert_eq!(hr_after.solution.source, SsrSource::RtcmSsr);
        // The exclusion is what prevents use. VI index 5 is 60 s, so the indication runs
        // from its own reference epoch t_ref + 30 s through t_ref + 90 s inclusive.
        assert!(!store5.is_satellite_excluded(sat1, t_ref + 29.9));
        assert!(store5.is_satellite_excluded(sat1, t_ref + 30.0));
        assert!(store5.is_satellite_excluded(sat1, t_ref + 90.0));
        // Once the indication expires the preserved RTCM state is no longer blocked.
        assert!(!store5.is_satellite_excluded(sat1, t_ref + 90.1));
    }

    /// A HAS do-not-use indication must exclude the satellite for exactly its own validity
    /// interval without destroying another provider's state.
    ///
    /// Per Galileo HAS SIS ICD Section 5.2.2.1 the indication is bounded by the validity
    /// interval that starts at its TOH reference epoch. It is an instruction not to use the
    /// satellite while it is active, so it must gate use through the exclusion marker rather
    /// than by discarding RTCM SSR corrections that remain valid in their own right. Every
    /// case below drives the store exclusively through the public ingestion entry points; no
    /// pending or cached state is synthesised by writing private fields.
    #[test]
    fn has_do_not_use_gates_use_without_discarding_rtcm_state() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t_ref = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let iode = broadcast
            .select_record_at(sat, t_ref)
            .expect("broadcast record at SSR epoch")
            .issue_of_data
            .expect("broadcast issue")
            .issue;

        // RTCM reception week; only the week number is consulted for SSR epoch resolution.
        let rtcm_week = GnssWeekTow::new(TimeScale::Gpst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("RTCM reception week");
        // HAS reception must be GST and must not precede the resolved TOH epoch.
        let has_reception = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("HAS GST reception");

        // Combined orbit and clock transmitted at t_ref, update interval index 7 = 120 s.
        // The rate terms are referenced half an interval later, at t_ref + 60 s; the
        // correction applies while |t - t_ref| <= 90 s (RTKLIB `MAXAGESSR`).
        let rtcm_combined = SsrMessage {
            message_number: 1060,
            system: GnssSystem::Gps,
            kind: SsrKind::CombinedOrbitClock,
            header: SsrHeader {
                epoch_time_s: REAL_SSR_EPOCH_TOW_S as u32,
                update_interval: 7,
                multiple_message: false,
                iod_ssr: 3,
                provider_id: 9,
                solution_id: 1,
                satellite_reference_datum: Some(false),
                dispersive_bias_consistency: None,
                mw_consistency: None,
                satellite_count: 1,
            },
            orbit: vec![SsrOrbitRecord {
                satellite_id: sat.prn,
                iode,
                delta_radial: -20_000,
                delta_along: 10_000,
                delta_cross: -3_000,
                dot_delta_radial: 0,
                dot_delta_along: 0,
                dot_delta_cross: 0,
            }],
            clock: vec![SsrClockRecord {
                satellite_id: sat.prn,
                c0: 5_000,
                c1: 0,
                c2: 0,
            }],
            code_bias: Vec::new(),
            phase_bias: Vec::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };

        // High-rate clock sharing solution identity and IOD SSR with the base clock, so it
        // attaches on arrival. Transmitted at t_ref, update interval index 6 = 60 s.
        let rtcm_high_rate = SsrMessage {
            message_number: 1062,
            system: GnssSystem::Gps,
            kind: SsrKind::HighRateClock,
            header: SsrHeader {
                epoch_time_s: REAL_SSR_EPOCH_TOW_S as u32,
                update_interval: 6,
                multiple_message: false,
                iod_ssr: 3,
                provider_id: 9,
                solution_id: 1,
                satellite_reference_datum: None,
                dispersive_bias_consistency: None,
                mw_consistency: None,
                satellite_count: 1,
            },
            orbit: Vec::new(),
            clock: vec![SsrClockRecord {
                satellite_id: sat.prn,
                c0: 700,
                c1: 0,
                c2: 0,
            }],
            code_bias: Vec::new(),
            phase_bias: Vec::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };

        let has_mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });
        // VI index 5 = 60 s. TOH offsets are relative to the hour containing t_ref.
        let toh_at_t_ref = (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16;
        let dnu_message = |toh_s: u16| HasMt1Message {
            header: HasMt1Header {
                toh_s,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 4,
            },
            mask: has_mask.clone(),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: true,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        // Baseline: the same RTCM stream with no HAS message at all.
        let mut baseline = SsrCorrectionStore::new();
        baseline
            .ingest_ssr(&rtcm_combined, rtcm_week)
            .expect("baseline combined");
        baseline
            .ingest_ssr(&rtcm_high_rate, rtcm_week)
            .expect("baseline high-rate");
        let baseline_orbit = *baseline.orbit(sat).expect("baseline orbit");
        let baseline_clock = *baseline.clock(sat).expect("baseline clock");
        let baseline_pending = *baseline.pending_high_rate(sat).expect("baseline pending");
        assert_eq!(
            baseline_clock.high_rate,
            Some(baseline_pending),
            "high-rate attaches to the base clock on arrival"
        );

        // Assert the full retained RTCM state of a store is byte-identical to the baseline.
        let assert_rtcm_state_intact = |store: &SsrCorrectionStore, label: &str| {
            let orbit = store
                .orbit(sat)
                .unwrap_or_else(|| panic!("{label}: orbit retained"));
            let clock = store
                .clock(sat)
                .unwrap_or_else(|| panic!("{label}: clock retained"));
            let pending = store
                .pending_high_rate(sat)
                .unwrap_or_else(|| panic!("{label}: pending high-rate retained"));
            assert_eq!(*orbit, baseline_orbit, "{label}: orbit state unchanged");
            assert_eq!(*clock, baseline_clock, "{label}: clock state unchanged");
            assert_eq!(
                *pending, baseline_pending,
                "{label}: pending state unchanged"
            );
            assert_eq!(clock.solution.source, SsrSource::RtcmSsr);
            assert_eq!(clock.iod_ssr, 3);
            assert_eq!(clock.ref_epoch_j2000_s, t_ref + 60.0);
            assert_eq!(clock.transmitted_epoch_j2000_s, t_ref);
            assert_eq!(clock.update_interval_s, 120.0);
            assert_eq!(clock.c0_m.to_bits(), 0.5_f64.to_bits());
            assert_eq!(
                clock.high_rate.map(|hr| hr.c0_m.to_bits()),
                Some(0.07_f64.to_bits()),
                "{label}: attached high-rate value unchanged"
            );
        };

        let fallback_policy = SsrFallbackPolicy {
            on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
            regional: RegionalPolicy::DeclineRegional,
        };

        // 1. RTCM first, then an active HAS do-not-use.
        let mut store_rtcm_first = SsrCorrectionStore::new();
        store_rtcm_first
            .ingest_ssr(&rtcm_combined, rtcm_week)
            .expect("combined before DNU");
        store_rtcm_first
            .ingest_ssr(&rtcm_high_rate, rtcm_week)
            .expect("high-rate before DNU");
        store_rtcm_first
            .ingest_has_mt1(&dnu_message(toh_at_t_ref), has_reception)
            .expect("ingest active HAS DNU");
        assert_rtcm_state_intact(&store_rtcm_first, "RTCM then DNU");

        // 2. The reverse arrival order reaches the same state: an exclusion already on record
        //    does not reject later RTCM data, and later RTCM data does not clear the exclusion.
        let mut store_dnu_first = SsrCorrectionStore::new();
        store_dnu_first
            .ingest_has_mt1(&dnu_message(toh_at_t_ref), has_reception)
            .expect("ingest active HAS DNU");
        store_dnu_first
            .ingest_ssr(&rtcm_combined, rtcm_week)
            .expect("combined after DNU");
        store_dnu_first
            .ingest_ssr(&rtcm_high_rate, rtcm_week)
            .expect("high-rate after DNU");
        assert_rtcm_state_intact(&store_dnu_first, "DNU then RTCM");
        assert_eq!(
            store_dnu_first.has_exclusion(sat).copied(),
            store_rtcm_first.has_exclusion(sat).copied(),
            "arrival order does not change the resulting exclusion"
        );

        // 3. While the indication is active the satellite is unusable in both orders, and
        //    broadcast fallback must not bypass the exclusion either.
        let t_blocked = t_ref + 10.0;
        for (store, label) in [
            (&store_rtcm_first, "RTCM then DNU"),
            (&store_dnu_first, "DNU then RTCM"),
        ] {
            assert!(
                store.is_satellite_excluded(sat, t_blocked),
                "{label}: exclusion active"
            );
            assert!(
                SsrCorrectedEphemeris::new(&broadcast, store)
                    .corrected_state(sat, t_blocked)
                    .is_none(),
                "{label}: strict declines while excluded"
            );
            assert!(
                SsrCorrectedEphemeris::new(&broadcast, store)
                    .with_fallback(fallback_policy.clone())
                    .corrected_state(sat, t_blocked)
                    .is_none(),
                "{label}: fallback cannot bypass an active exclusion"
            );
        }

        // 4. Once the indication expires the preserved RTCM corrections resume, because they
        //    are still inside their own freshness window. Exclusion runs [t_ref, t_ref + 60];
        //    the RTCM combined message stays fresh to t_ref + 90.
        let t_resumed = t_ref + 60.001;
        assert!(!store_rtcm_first.is_satellite_excluded(sat, t_resumed));
        let resumed = SsrCorrectedEphemeris::new(&broadcast, &store_rtcm_first)
            .corrected_state(sat, t_resumed)
            .expect("preserved RTCM correction resumes after the exclusion expires");
        let expected = SsrCorrectedEphemeris::new(&broadcast, &baseline)
            .corrected_state(sat, t_resumed)
            .expect("baseline corrected state");
        assert_eq!(
            resumed.0.map(f64::to_bits),
            expected.0.map(f64::to_bits),
            "resumed position is bit-identical to the never-excluded baseline"
        );
        assert_eq!(
            resumed.1.to_bits(),
            expected.1.to_bits(),
            "resumed clock is bit-identical to the never-excluded baseline"
        );
        assert_rtcm_state_intact(&store_rtcm_first, "after expiry");

        // 5. A do-not-use that has already expired by the time the RTCM epoch is queried must
        //    not have removed anything: the satellite corrects normally at t_ref. The
        //    indication references t_ref - 90 s and lasts 60 s, so it ended at t_ref - 30 s.
        let mut store_stale_dnu = SsrCorrectionStore::new();
        store_stale_dnu
            .ingest_has_mt1(&dnu_message(toh_at_t_ref - 90), has_reception)
            .expect("ingest already-expired HAS DNU");
        store_stale_dnu
            .ingest_ssr(&rtcm_combined, rtcm_week)
            .expect("combined after stale DNU");
        store_stale_dnu
            .ingest_ssr(&rtcm_high_rate, rtcm_week)
            .expect("high-rate after stale DNU");
        assert_rtcm_state_intact(&store_stale_dnu, "expired DNU, RTCM after");
        assert!(!store_stale_dnu.is_satellite_excluded(sat, t_ref));
        let stale_state = SsrCorrectedEphemeris::new(&broadcast, &store_stale_dnu)
            .corrected_state(sat, t_ref)
            .expect("expired DNU does not suppress a fresh RTCM correction");
        let baseline_state = SsrCorrectedEphemeris::new(&broadcast, &baseline)
            .corrected_state(sat, t_ref)
            .expect("baseline corrected state at t_ref");
        assert_eq!(
            stale_state.0.map(f64::to_bits),
            baseline_state.0.map(f64::to_bits)
        );
        assert_eq!(stale_state.1.to_bits(), baseline_state.1.to_bits());

        // 6. The same stale indication arriving after the RTCM stream is equally harmless.
        let mut store_stale_dnu_last = SsrCorrectionStore::new();
        store_stale_dnu_last
            .ingest_ssr(&rtcm_combined, rtcm_week)
            .expect("combined before stale DNU");
        store_stale_dnu_last
            .ingest_ssr(&rtcm_high_rate, rtcm_week)
            .expect("high-rate before stale DNU");
        store_stale_dnu_last
            .ingest_has_mt1(&dnu_message(toh_at_t_ref - 90), has_reception)
            .expect("ingest already-expired HAS DNU");
        assert_rtcm_state_intact(&store_stale_dnu_last, "RTCM then expired DNU");
        assert!(!store_stale_dnu_last.is_satellite_excluded(sat, t_ref));
        let stale_last_state = SsrCorrectedEphemeris::new(&broadcast, &store_stale_dnu_last)
            .corrected_state(sat, t_ref)
            .expect("expired DNU arriving last does not suppress a fresh RTCM correction");
        assert_eq!(
            stale_last_state.0.map(f64::to_bits),
            baseline_state.0.map(f64::to_bits)
        );
        assert_eq!(stale_last_state.1.to_bits(), baseline_state.1.to_bits());

        // 7. A pending high-rate that arrives before its base clock survives a do-not-use and
        //    still attaches when the base clock finally arrives.
        let mut store_pending_first = SsrCorrectionStore::new();
        store_pending_first
            .ingest_ssr(&rtcm_high_rate, rtcm_week)
            .expect("high-rate before base clock");
        assert!(
            store_pending_first.clock(sat).is_none(),
            "no base clock yet, so the high-rate is only pending"
        );
        store_pending_first
            .ingest_has_mt1(&dnu_message(toh_at_t_ref), has_reception)
            .expect("ingest active HAS DNU");
        assert_eq!(
            store_pending_first.pending_high_rate(sat).copied(),
            Some(baseline_pending),
            "pending high-rate awaiting a base clock survives a do-not-use"
        );
        store_pending_first
            .ingest_ssr(&rtcm_combined, rtcm_week)
            .expect("base clock after DNU");
        assert_rtcm_state_intact(&store_pending_first, "pending first");
        assert_eq!(
            store_pending_first
                .clock(sat)
                .and_then(|c| c.high_rate)
                .map(|hr| hr.c0_m.to_bits()),
            Some(0.07_f64.to_bits()),
            "surviving pending high-rate attaches to the late base clock"
        );
        // It remains excluded while the indication is active, and usable once it expires.
        assert!(store_pending_first.is_satellite_excluded(sat, t_blocked));
        assert!(SsrCorrectedEphemeris::new(&broadcast, &store_pending_first)
            .corrected_state(sat, t_blocked)
            .is_none());
        assert!(!store_pending_first.is_satellite_excluded(sat, t_resumed));
        assert!(SsrCorrectedEphemeris::new(&broadcast, &store_pending_first)
            .corrected_state(sat, t_resumed)
            .is_some());
    }

    #[test]
    fn test_has_orbit_supersession_unavailability_and_restoration() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let t0_tow = 100_000.0;
        let reception_t0 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow).unwrap();
        let reception_t30 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 30.0).unwrap();
        let reception_t60 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 60.0).unwrap();
        // Reference epochs the TOH values below resolve to, computed independently of the
        // TOH arithmetic so the supersession watermark can be pinned to an exact epoch.
        let epoch_at =
            |tow_s: f64| f64::from(1042u32) * SECONDS_PER_WEEK + tow_s - GPS_EPOCH_TO_J2000_S;

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat1.prn, sat2.prn],
                signals: vec![0, 9],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        let mask_sat1_only = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat1.prn],
                signals: vec![0, 9],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        // 1. Usable HAS orbit at T0 for sat1 and sat2
        let has_t0 = HasMt1Message {
            header: HasMt1Header {
                toh_s: (t0_tow as u32 % 3600) as u16,
                mask: true,
                orbit: true,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![
                    HasOrbitCorrection {
                        sat: sat1,
                        nav_message: 0,
                        iode: 10,
                        radial_m: Some(0.10), // 40 * 0.0025
                        along_m: Some(0.20),  // 25 * 0.0080
                        cross_m: Some(0.32),  // 40 * 0.0080
                    },
                    HasOrbitCorrection {
                        sat: sat2,
                        nav_message: 0,
                        iode: 20,
                        radial_m: Some(0.40), // 160 * 0.0025
                        along_m: Some(0.48),  // 60 * 0.0080
                        cross_m: Some(0.56),  // 70 * 0.0080
                    },
                ],
            }),
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        assert!(has_t0.encode().is_ok());

        let mut store = SsrCorrectionStore::new();
        store.ingest_has_mt1(&has_t0, reception_t0).unwrap();
        assert!(store.orbit(sat1).is_some());
        assert!(store.orbit(sat2).is_some());

        // 2. Newer HAS message at T30: sat1 has partial vector (along_m: None makes entire vector unavailable)
        // Uses smaller inline mask for sat1 with mask_id: 2 to form a valid wire case
        let has_t30_unavail = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 30) as u16,
                mask: true,
                orbit: true,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 2,
                iod_set_id: 1,
            },
            mask: mask_sat1_only.clone(),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat: sat1,
                    nav_message: 0,
                    iode: 11,
                    radial_m: Some(0.15),
                    along_m: None, // partial -> vector unavailable!
                    cross_m: Some(0.32),
                }],
            }),
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        assert!(has_t30_unavail.encode().is_ok());

        store
            .ingest_has_mt1(&has_t30_unavail, reception_t30)
            .unwrap();
        assert!(
            store.orbit(sat1).is_none(),
            "partial HAS orbit vector must clear strictly older orbit"
        );
        assert!(
            store.orbit(sat2).is_some(),
            "unrelated sat2 orbit must remain preserved"
        );
        assert_eq!(
            store.has_orbit_superseded_epoch(sat1),
            Some(epoch_at(t0_tow + 30.0)),
            "unavailable sentinel records its own epoch as the orbit supersession watermark"
        );
        assert_eq!(
            store.has_orbit_superseded_epoch(sat2),
            None,
            "watermark is per satellite and must not leak to sat2"
        );

        // 3. Delayed stale usable HAS at T0 arrives: refused by watermark, cannot resurrect
        store.ingest_has_mt1(&has_t0, reception_t0).unwrap();
        assert!(
            store.orbit(sat1).is_none(),
            "stale usable HAS at T0 cannot resurrect superseded orbit"
        );
        assert!(store.orbit(sat2).is_some(), "sat2 remains preserved");
        assert_eq!(
            store.has_orbit_superseded_epoch(sat1),
            Some(epoch_at(t0_tow + 30.0)),
            "a refused stale update leaves the watermark where it was"
        );

        // 4. Newer usable HAS orbit at T60 restores sat1 orbit
        let has_t60_usable = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 60) as u16,
                mask: true,
                orbit: true,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 3,
                iod_set_id: 1,
            },
            mask: mask_sat1_only.clone(),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat: sat1,
                    nav_message: 0,
                    iode: 12,
                    radial_m: Some(0.20), // 80 * 0.0025
                    along_m: Some(0.24),  // 30 * 0.0080
                    cross_m: Some(0.40),  // 50 * 0.0080
                }],
            }),
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        assert!(has_t60_usable.encode().is_ok());

        store
            .ingest_has_mt1(&has_t60_usable, reception_t60)
            .unwrap();
        let restored = store.orbit(sat1).expect("newer usable HAS restores orbit");
        assert_eq!(restored.radial_m.to_bits(), 0.20_f64.to_bits());
        assert_eq!(restored.along_m.to_bits(), 0.24_f64.to_bits());
        assert_eq!(restored.cross_m.to_bits(), 0.40_f64.to_bits());
        assert!(store.orbit(sat2).is_some(), "sat2 remains preserved");
        assert_eq!(
            store.has_orbit_superseded_epoch(sat1),
            None,
            "a newer usable orbit clears the supersession watermark"
        );

        // 5. Delayed older unavailable at T30 arriving after T60 cannot clear newer usable
        store
            .ingest_has_mt1(&has_t30_unavail, reception_t30)
            .unwrap();
        assert!(
            store.orbit(sat1).is_some(),
            "older unavailable must not clear newer usable orbit"
        );
        assert_eq!(
            store.has_orbit_superseded_epoch(sat1),
            None,
            "a refused older sentinel must not re-arm the watermark"
        );

        // 6. Equal-epoch: usable wins over unavailable in both orders
        let mut store_eq_a = SsrCorrectionStore::new();
        store_eq_a
            .ingest_has_mt1(&has_t30_unavail, reception_t30)
            .unwrap();
        let has_t30_usable = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 30) as u16,
                mask: true,
                orbit: true,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 2,
                iod_set_id: 2,
            },
            mask: mask_sat1_only.clone(),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat: sat1,
                    nav_message: 0,
                    iode: 11,
                    radial_m: Some(0.15),
                    along_m: Some(0.24),
                    cross_m: Some(0.40),
                }],
            }),
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        assert!(has_t30_usable.encode().is_ok());
        store_eq_a
            .ingest_has_mt1(&has_t30_usable, reception_t30)
            .unwrap();
        assert!(
            store_eq_a.orbit(sat1).is_some(),
            "usable wins over unavailable in order A (unavail then usable)"
        );
        assert_eq!(
            store_eq_a.has_orbit_superseded_epoch(sat1),
            None,
            "an equal-epoch usable orbit clears the watermark the sentinel had armed"
        );

        let mut store_eq_b = SsrCorrectionStore::new();
        store_eq_b
            .ingest_has_mt1(&has_t30_usable, reception_t30)
            .unwrap();
        store_eq_b
            .ingest_has_mt1(&has_t30_unavail, reception_t30)
            .unwrap();
        assert!(
            store_eq_b.orbit(sat1).is_some(),
            "usable wins over unavailable in order B (usable then unavail)"
        );
        assert_eq!(
            store_eq_b.has_orbit_superseded_epoch(sat1),
            None,
            "an equal-epoch sentinel is refused outright and arms no watermark"
        );

        // 7. RTCM orbit is preserved when HAS unavailable arrives
        let mut store_rtcm = SsrCorrectionStore::new();
        let rtcm_msg = SsrMessage {
            message_number: 1057,
            system: GnssSystem::Gps,
            kind: SsrKind::Orbit,
            header: header(SsrKind::Orbit),
            orbit: vec![SsrOrbitRecord {
                satellite_id: 1,
                iode: 42,
                delta_radial: 100,
                delta_along: 200,
                delta_cross: 300,
                dot_delta_radial: 0,
                dot_delta_along: 0,
                dot_delta_cross: 0,
            }],
            clock: Vec::new(),
            code_bias: Vec::new(),
            phase_bias: Vec::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };
        store_rtcm.ingest_ssr(&rtcm_msg, reception_t30).unwrap();
        assert_eq!(
            store_rtcm.orbit(sat1).unwrap().solution.source,
            SsrSource::RtcmSsr
        );
        store_rtcm
            .ingest_has_mt1(&has_t30_unavail, reception_t30)
            .unwrap();
        assert_eq!(
            store_rtcm.orbit(sat1).unwrap().solution.source,
            SsrSource::RtcmSsr,
            "RTCM orbit preserved on HAS unavailable"
        );
        assert_eq!(
            store_rtcm.has_orbit_superseded_epoch(sat1),
            Some(epoch_at(t0_tow + 30.0)),
            "the sentinel still arms the HAS watermark while leaving RTCM orbit untouched"
        );
    }

    #[test]
    fn test_code_and_phase_bias_supersession_and_watermark_persistence() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let t0_tow = 200_000.0;
        let reception_t0 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow).unwrap();
        let reception_t30 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 30.0).unwrap();
        let reception_t60 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 60.0).unwrap();
        // Reference epochs the TOH values below resolve to, computed independently of the
        // TOH arithmetic so each supersession watermark can be pinned to an exact epoch.
        let epoch_at =
            |tow_s: f64| f64::from(1042u32) * SECONDS_PER_WEEK + tow_s - GPS_EPOCH_TO_J2000_S;

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat1.prn, sat2.prn],
                signals: vec![0, 9],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        let mask_sat1_sig0 = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat1.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        let pb_sat1_sig0_cycles = 0.25;
        let pb_sat1_sig9_cycles = -0.15;
        let pb_sat2_sig0_cycles = 0.0;
        let pb_sat2_sig9_cycles = 0.40;
        let pb_sat1_sig0_m = pb_sat1_sig0_cycles * (C_M_S / F_L1_HZ);
        let pb_sat1_sig9_m = pb_sat1_sig9_cycles * (C_M_S / F_L2_HZ);
        let pb_sat2_sig9_m = pb_sat2_sig9_cycles * (C_M_S / F_L2_HZ);

        // T0: sat1 sig0 = 0.5m, sig9 = -0.3m. sat2 sig0 = 0.0m (zero is usable!), sig9 = 0.8m.
        let has_t0 = HasMt1Message {
            header: HasMt1Header {
                toh_s: (t0_tow as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasCodeBias {
                        sat: sat1,
                        signal_id: 0,
                        bias_m: Some(0.5),
                    },
                    HasCodeBias {
                        sat: sat1,
                        signal_id: 9,
                        bias_m: Some(-0.3),
                    },
                    HasCodeBias {
                        sat: sat2,
                        signal_id: 0,
                        bias_m: Some(0.0),
                    }, // zero is usable!
                    HasCodeBias {
                        sat: sat2,
                        signal_id: 9,
                        bias_m: Some(0.8),
                    },
                ],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasPhaseBias {
                        sat: sat1,
                        signal_id: 0,
                        bias_cycles: Some(pb_sat1_sig0_cycles),
                        discontinuity_indicator: 0,
                    },
                    HasPhaseBias {
                        sat: sat1,
                        signal_id: 9,
                        bias_cycles: Some(pb_sat1_sig9_cycles),
                        discontinuity_indicator: 0,
                    },
                    HasPhaseBias {
                        sat: sat2,
                        signal_id: 0,
                        bias_cycles: Some(pb_sat2_sig0_cycles),
                        discontinuity_indicator: 0,
                    },
                    HasPhaseBias {
                        sat: sat2,
                        signal_id: 9,
                        bias_cycles: Some(pb_sat2_sig9_cycles),
                        discontinuity_indicator: 0,
                    },
                ],
            }),
            padding_bits: Vec::new(),
        };
        assert!(has_t0.encode().is_ok());

        let mut store = SsrCorrectionStore::new();
        store.ingest_has_mt1(&has_t0, reception_t0).unwrap();
        assert_eq!(store.code_bias(sat1, has_sig(sat1, 0)), Some(0.5));
        assert_eq!(store.code_bias(sat1, has_sig(sat1, 9)), Some(-0.3));
        assert_eq!(store.code_bias(sat2, has_sig(sat2, 0)), Some(0.0));
        assert_eq!(
            store.phase_bias(sat1, has_sig(sat1, 0)),
            Some(pb_sat1_sig0_m)
        );
        assert_eq!(
            store.phase_bias(sat1, has_sig(sat1, 9)),
            Some(pb_sat1_sig9_m)
        );
        assert_eq!(store.phase_bias(sat2, has_sig(sat2, 0)), Some(0.0));
        assert_eq!(
            store.phase_bias(sat2, has_sig(sat2, 9)),
            Some(pb_sat2_sig9_m)
        );

        // T30: sat1 sig 0 has bias_cycles: None (unavailable sentinel).
        // Uses smaller inline mask for sat1 sig 0 with mask_id: 2 to form a valid wire case
        let has_t30_unavail = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 30) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 2,
                iod_set_id: 1,
            },
            mask: mask_sat1_sig0.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![HasCodeBias {
                    sat: sat1,
                    signal_id: 0,
                    bias_m: None,
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat: sat1,
                    signal_id: 0,
                    bias_cycles: None,
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };
        assert!(has_t30_unavail.encode().is_ok());

        store
            .ingest_has_mt1(&has_t30_unavail, reception_t30)
            .unwrap();
        // Cleared strictly superseded numeric correction for sat1 sig 0
        assert_eq!(store.code_bias(sat1, has_sig(sat1, 0)), None);
        assert_eq!(store.phase_bias(sat1, has_sig(sat1, 0)), None);
        // Unrelated signals/satellites preserved
        assert_eq!(store.code_bias(sat1, has_sig(sat1, 9)), Some(-0.3));
        assert_eq!(store.code_bias(sat2, has_sig(sat2, 0)), Some(0.0));
        assert_eq!(store.code_bias(sat2, has_sig(sat2, 9)), Some(0.8));
        // The unavailable sentinel arms a per-signal supersession watermark at its own epoch.
        assert_eq!(
            store.has_code_bias_superseded_epoch(sat1, has_sig(sat1, 0)),
            Some(epoch_at(t0_tow + 30.0)),
            "code bias watermark armed at the sentinel epoch"
        );
        assert_eq!(
            store.has_phase_bias_superseded_epoch(sat1, has_sig(sat1, 0)),
            Some(epoch_at(t0_tow + 30.0)),
            "phase bias watermark armed at the sentinel epoch"
        );
        assert_eq!(
            store.has_code_bias_superseded_epoch(sat1, has_sig(sat1, 9)),
            None,
            "watermarks are per signal and must not leak to sig 9"
        );
        assert_eq!(
            store.has_phase_bias_superseded_epoch(sat2, has_sig(sat2, 0)),
            None
        );

        // Stale usable at T0 arrives again: refused by watermark, cannot resurrect
        store.ingest_has_mt1(&has_t0, reception_t0).unwrap();
        assert_eq!(
            store.code_bias(sat1, has_sig(sat1, 0)),
            None,
            "stale usable at T0 cannot resurrect sat1 sig 0"
        );
        assert_eq!(
            store.has_code_bias_superseded_epoch(sat1, has_sig(sat1, 0)),
            Some(epoch_at(t0_tow + 30.0)),
            "a refused stale record leaves the code bias watermark in place"
        );
        assert_eq!(
            store.has_phase_bias_superseded_epoch(sat1, has_sig(sat1, 0)),
            Some(epoch_at(t0_tow + 30.0)),
            "a refused stale record leaves the phase bias watermark in place"
        );

        // Newer usable HAS at T60: restores sat1 sig 0
        let pb_sat1_t60_cycles = 0.55;
        let pb_sat1_t60_m = pb_sat1_t60_cycles * (C_M_S / F_L1_HZ);
        let has_t60_usable = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 60) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 3,
                iod_set_id: 1,
            },
            mask: mask_sat1_sig0.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![HasCodeBias {
                    sat: sat1,
                    signal_id: 0,
                    bias_m: Some(0.60),
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat: sat1,
                    signal_id: 0,
                    bias_cycles: Some(pb_sat1_t60_cycles),
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };
        assert!(has_t60_usable.encode().is_ok());

        store
            .ingest_has_mt1(&has_t60_usable, reception_t60)
            .unwrap();
        assert_eq!(store.code_bias(sat1, has_sig(sat1, 0)), Some(0.60));
        assert_eq!(
            store.phase_bias(sat1, has_sig(sat1, 0)),
            Some(pb_sat1_t60_m)
        );
        assert_eq!(
            store.has_code_bias_superseded_epoch(sat1, has_sig(sat1, 0)),
            None,
            "a newer usable code bias clears the watermark"
        );
        assert_eq!(
            store.has_phase_bias_superseded_epoch(sat1, has_sig(sat1, 0)),
            None,
            "a newer usable phase bias clears the watermark"
        );

        // Delayed older unavailable at T30 arriving after T60 cannot clear newer usable
        store
            .ingest_has_mt1(&has_t30_unavail, reception_t30)
            .unwrap();
        assert_eq!(
            store.code_bias(sat1, has_sig(sat1, 0)),
            Some(0.60),
            "older unavailable cannot clear newer usable bias"
        );
        assert_eq!(
            store.has_code_bias_superseded_epoch(sat1, has_sig(sat1, 0)),
            None,
            "a refused older sentinel must not re-arm the code bias watermark"
        );
        assert_eq!(
            store.has_phase_bias_superseded_epoch(sat1, has_sig(sat1, 0)),
            None,
            "a refused older sentinel must not re-arm the phase bias watermark"
        );

        // Equal-epoch usable wins over unavailable in both orders
        let pb_sat1_t30_cycles = 0.35;
        let pb_sat1_t30_m = pb_sat1_t30_cycles * (C_M_S / F_L1_HZ);
        let mut store_eq1 = SsrCorrectionStore::new();
        store_eq1
            .ingest_has_mt1(&has_t30_unavail, reception_t30)
            .unwrap();
        let has_t30_usable = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 30) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 2,
                iod_set_id: 2,
            },
            mask: mask_sat1_sig0.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![HasCodeBias {
                    sat: sat1,
                    signal_id: 0,
                    bias_m: Some(0.52),
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat: sat1,
                    signal_id: 0,
                    bias_cycles: Some(pb_sat1_t30_cycles),
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };
        assert!(has_t30_usable.encode().is_ok());
        store_eq1
            .ingest_has_mt1(&has_t30_usable, reception_t30)
            .unwrap();
        assert_eq!(
            store_eq1.code_bias(sat1, has_sig(sat1, 0)),
            Some(0.52),
            "usable wins over unavail in order 1"
        );
        assert_eq!(
            store_eq1.phase_bias(sat1, has_sig(sat1, 0)),
            Some(pb_sat1_t30_m),
            "usable phase wins over unavail in order 1"
        );
        assert_eq!(
            store_eq1.has_code_bias_superseded_epoch(sat1, has_sig(sat1, 0)),
            None
        );
        assert_eq!(
            store_eq1.has_phase_bias_superseded_epoch(sat1, has_sig(sat1, 0)),
            None
        );

        let mut store_eq2 = SsrCorrectionStore::new();
        store_eq2
            .ingest_has_mt1(&has_t30_usable, reception_t30)
            .unwrap();
        store_eq2
            .ingest_has_mt1(&has_t30_unavail, reception_t30)
            .unwrap();
        assert_eq!(
            store_eq2.code_bias(sat1, has_sig(sat1, 0)),
            Some(0.52),
            "usable wins over unavail in order 2"
        );
        assert_eq!(
            store_eq2.phase_bias(sat1, has_sig(sat1, 0)),
            Some(pb_sat1_t30_m),
            "usable phase wins over unavail in order 2"
        );
        assert_eq!(
            store_eq2.has_code_bias_superseded_epoch(sat1, has_sig(sat1, 0)),
            None
        );
        assert_eq!(
            store_eq2.has_phase_bias_superseded_epoch(sat1, has_sig(sat1, 0)),
            None
        );

        // Mixed RTCM/HAS provenance: RTCM overwrites active value, HAS watermark persists privately
        let mut store_mix = SsrCorrectionStore::new();
        store_mix
            .ingest_has_mt1(&has_t30_unavail, reception_t30)
            .unwrap();
        let rtcm_code_msg = SsrMessage {
            message_number: 1059,
            system: GnssSystem::Gps,
            kind: SsrKind::CodeBias,
            header: header(SsrKind::CodeBias),
            orbit: Vec::new(),
            clock: Vec::new(),
            code_bias: vec![crate::rtcm::SsrCodeBiasRecord {
                satellite_id: 1,
                biases: vec![(0, 25)], // 0.25m
            }],
            phase_bias: Vec::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };
        let watermark_before_rtcm =
            store_mix.has_code_bias_superseded_epoch(sat1, has_sig(sat1, 0));
        assert_eq!(
            watermark_before_rtcm,
            Some(epoch_at(t0_tow + 30.0)),
            "the sentinel arms the code bias watermark before RTCM arrives"
        );
        store_mix.ingest_ssr(&rtcm_code_msg, reception_t30).unwrap();
        assert_eq!(store_mix.code_bias(sat1, has_sig(sat1, 0)), Some(0.25));
        assert_eq!(
            store_mix.has_code_bias_superseded_epoch(sat1, has_sig(sat1, 0)),
            watermark_before_rtcm,
            "RTCM replaces the active value but leaves the HAS watermark untouched"
        );

        // Stale HAS at T0 arriving now cannot overwrite RTCM due to persistent watermark
        store_mix.ingest_has_mt1(&has_t0, reception_t0).unwrap();
        assert_eq!(
            store_mix.code_bias(sat1, has_sig(sat1, 0)),
            Some(0.25),
            "stale HAS cannot overwrite RTCM across watermark"
        );
        assert_eq!(
            store_mix.has_code_bias_superseded_epoch(sat1, has_sig(sat1, 0)),
            watermark_before_rtcm,
            "the persistent watermark is what refuses the stale HAS record"
        );
    }

    #[test]
    fn test_phase_bias_cycles_without_metres_and_continuity_resets() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let t0_tow = 300_000.0;
        let reception_t0 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow).unwrap();
        let t0_j2000 =
            has_mt1_reference_j2000_s(reception_t0, (t0_tow as u32 % 3600) as u16).unwrap();

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        let unknown_sig_mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![1],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        // 1. HAS SIS ICD Table 20 reserves GPS index 1: the record keeps its cycles under
        // its raw source-qualified signal and is reported as an unknown signal.
        let raw = SsrRawSignal::galileo_has(GnssSystem::Gps, 1);
        let has_unknown_carrier = HasMt1Message {
            header: HasMt1Header {
                toh_s: (t0_tow as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: unknown_sig_mask,
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 1,
                    bias_cycles: Some(0.25),
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };
        assert!(has_unknown_carrier.encode().is_ok());

        let mut store = SsrCorrectionStore::new();
        store
            .ingest_has_mt1(&has_unknown_carrier, reception_t0)
            .unwrap();
        assert_eq!(
            store.phase_bias(sat, has_sig(sat, 1)),
            None,
            "cycles without metres must not return f64 value"
        );
        let q_cycles = store.query_phase_bias(sat, has_sig(sat, 1), t0_j2000, None);
        assert_eq!(q_cycles.status, SsrBiasStatus::UnknownSignal);
        assert_eq!(
            q_cycles.details,
            SsrBiasResolutionDetails::UnknownSignal(raw)
        );
        assert_eq!(q_cycles.source_signal, Some(raw));
        assert_eq!(q_cycles.bias_cycles, Some(0.25));
        assert_eq!(q_cycles.bias_m, None);
        assert_eq!(
            q_cycles.discontinuity_details,
            Some(SsrDiscontinuityDetails::InitialTokenEstablished)
        );
        let cycles_token = q_cycles
            .continuity_token
            .expect("unknown-signal record carries a token");
        assert_eq!(cycles_token.signal(), SsrSignalKey::Unknown(raw));

        // An unknown-signal record keeps its UnknownSignal status while still
        // evaluating a supplied acknowledgement, so a token from another arc is not lost.
        let mut later_store = SsrCorrectionStore::new();
        let mut has_unknown_carrier_later = has_unknown_carrier.clone();
        has_unknown_carrier_later.header.toh_s = ((t0_tow as u32 % 3600) + 30) as u16;
        let reception_cycles_t30 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 30.0).unwrap();
        let t_cycles_t30 =
            has_mt1_reference_j2000_s(reception_cycles_t30, ((t0_tow as u32 % 3600) + 30) as u16)
                .unwrap();
        later_store
            .ingest_has_mt1(&has_unknown_carrier_later, reception_cycles_t30)
            .unwrap();
        let later_cycles_token = later_store
            .query_phase_bias(sat, has_sig(sat, 1), t_cycles_t30, None)
            .continuity_token
            .unwrap();

        let q_cycles_stale =
            later_store.query_phase_bias(sat, has_sig(sat, 1), t_cycles_t30, Some(cycles_token));
        assert_eq!(q_cycles_stale.status, SsrBiasStatus::UnknownSignal);
        assert_eq!(
            q_cycles_stale.details,
            SsrBiasResolutionDetails::UnknownSignal(raw)
        );
        assert_eq!(q_cycles_stale.bias_cycles, Some(0.25));
        assert_eq!(
            q_cycles_stale.discontinuity_details,
            Some(SsrDiscontinuityDetails::StaleToken)
        );

        let q_cycles_future =
            store.query_phase_bias(sat, has_sig(sat, 1), t0_j2000, Some(later_cycles_token));
        assert_eq!(q_cycles_future.status, SsrBiasStatus::UnknownSignal);
        assert_eq!(
            q_cycles_future.discontinuity_details,
            Some(SsrDiscontinuityDetails::FutureToken)
        );

        // A token minted for another satellite on the same unknown signal is a
        // mismatch, not a position on this arc's timeline.
        let other_sat = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let mut other_sat_msg = has_unknown_carrier.clone();
        other_sat_msg.mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![other_sat.prn],
                signals: vec![1],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });
        if let Some(pb) = other_sat_msg.phase_bias.as_mut() {
            for rec in &mut pb.records {
                rec.sat = other_sat;
            }
        }
        let mut other_sat_store = SsrCorrectionStore::new();
        other_sat_store
            .ingest_has_mt1(&other_sat_msg, reception_t0)
            .unwrap();
        let other_sat_token = other_sat_store
            .query_phase_bias(other_sat, has_sig(other_sat, 1), t0_j2000, None)
            .continuity_token
            .unwrap();

        let q_cycles_mismatch =
            store.query_phase_bias(sat, has_sig(sat, 1), t0_j2000, Some(other_sat_token));
        assert_eq!(q_cycles_mismatch.status, SsrBiasStatus::UnknownSignal);
        assert_eq!(
            q_cycles_mismatch.discontinuity_details,
            Some(SsrDiscontinuityDetails::MismatchedToken)
        );

        let q_cycles_exact =
            store.query_phase_bias(sat, has_sig(sat, 1), t0_j2000, Some(cycles_token));
        assert_eq!(q_cycles_exact.status, SsrBiasStatus::UnknownSignal);
        assert_eq!(
            q_cycles_exact.discontinuity_details,
            Some(SsrDiscontinuityDetails::Continuous),
            "an exact acknowledgement is continuous even though the signal is unknown"
        );

        // 2. Continuous updates, discontinuity, reset, wrap, and rejection
        let mut store2 = SsrCorrectionStore::new();
        let make_phase_msg = |toh_offset: u32, pdi: u8| -> HasMt1Message {
            HasMt1Message {
                header: HasMt1Header {
                    toh_s: ((t0_tow as u32 % 3600) + toh_offset) as u16,
                    mask: true,
                    orbit: false,
                    clock_full_set: false,
                    clock_subset: false,
                    code_bias: false,
                    phase_bias: true,
                    reserved: 0,
                    mask_id: 1,
                    iod_set_id: 1,
                },
                mask: mask.clone(),
                orbit: None,
                clock_full_set: None,
                clock_subset: None,
                code_bias: None,
                phase_bias: Some(HasPhaseBiasBlock {
                    validity_interval: 5,
                    records: vec![HasPhaseBias {
                        sat,
                        signal_id: 0,
                        bias_cycles: Some(0.25),
                        discontinuity_indicator: pdi,
                    }],
                }),
                padding_bits: Vec::new(),
            }
        };

        // Message 0: PDI = 0 at T0
        let msg0 = make_phase_msg(0, 0);
        store2.ingest_has_mt1(&msg0, reception_t0).unwrap();
        let q0 = store2.query_phase_bias(sat, has_sig(sat, 0), t0_j2000, None);
        assert_eq!(q0.status, SsrBiasStatus::Available);
        assert_eq!(q0.details, SsrBiasResolutionDetails::Available);
        assert_eq!(
            q0.discontinuity_details,
            Some(SsrDiscontinuityDetails::InitialTokenEstablished)
        );
        let token0 = q0.continuity_token.expect("token established");

        // Message 1: PDI = 0 at T30 (continuous)
        let reception_t30 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 30.0).unwrap();
        let msg1 = make_phase_msg(30, 0);
        store2.ingest_has_mt1(&msg1, reception_t30).unwrap();
        let q1 = store2.query_phase_bias(sat, has_sig(sat, 0), t0_j2000 + 30.0, Some(token0));
        assert_eq!(q1.status, SsrBiasStatus::Available);
        assert_eq!(q1.details, SsrBiasResolutionDetails::Available);
        assert_eq!(
            q1.discontinuity_details,
            Some(SsrDiscontinuityDetails::Continuous)
        );

        // Message 2: PDI = 1 at T60 (discontinuity!)
        let reception_t60 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 60.0).unwrap();
        let msg2 = make_phase_msg(60, 1);
        store2.ingest_has_mt1(&msg2, reception_t60).unwrap();
        let q2_disc = store2.query_phase_bias(sat, has_sig(sat, 0), t0_j2000 + 60.0, Some(token0));
        assert_eq!(q2_disc.status, SsrBiasStatus::PhaseDiscontinuityNeedsReset);
        assert_eq!(
            q2_disc.discontinuity_indicator,
            Some(PhaseDiscontinuityIndicator::GalileoHasPdi(1))
        );
        assert_eq!(
            q2_disc.discontinuity_details,
            Some(SsrDiscontinuityDetails::HasPdiChanged {
                previous: 0,
                current: 1
            })
        );
        let token1 = q2_disc
            .continuity_token
            .expect("new token provided with reset request");

        // Caller resets ambiguity and resumes query with new token
        let q2_ack = store2.query_phase_bias(sat, has_sig(sat, 0), t0_j2000 + 60.0, Some(token1));
        assert_eq!(q2_ack.status, SsrBiasStatus::Available);
        assert_eq!(q2_ack.details, SsrBiasResolutionDetails::Available);
        assert_eq!(
            q2_ack.discontinuity_details,
            Some(SsrDiscontinuityDetails::Continuous)
        );

        // Advance through PDI wrap: 1 -> 2 -> 3 -> 0
        let reception_t90 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 90.0).unwrap();
        store2
            .ingest_has_mt1(&make_phase_msg(90, 2), reception_t90)
            .unwrap();
        let reception_t120 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 120.0).unwrap();
        store2
            .ingest_has_mt1(&make_phase_msg(120, 3), reception_t120)
            .unwrap();
        let reception_t150 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 150.0).unwrap();
        store2
            .ingest_has_mt1(&make_phase_msg(150, 0), reception_t150)
            .unwrap();

        // Stale token0 from the original PDI=0 arc must be REJECTED despite PDI numerically wrapping to 0
        let q_wrap = store2.query_phase_bias(sat, has_sig(sat, 0), t0_j2000 + 150.0, Some(token0));
        assert_eq!(
            q_wrap.status,
            SsrBiasStatus::PhaseDiscontinuityNeedsReset,
            "wrapped PDI must not resurrect stale token"
        );
        assert_eq!(
            q_wrap.discontinuity_details,
            Some(SsrDiscontinuityDetails::StaleToken)
        );

        // Mismatched signal token rejection: obtain token for signal 1 from actual store query
        let q_other = store.query_phase_bias(sat, has_sig(sat, 1), t0_j2000, None);
        let bad_token = q_other.continuity_token.expect("token for signal 1");
        let q_mismatch =
            store2.query_phase_bias(sat, has_sig(sat, 0), t0_j2000 + 150.0, Some(bad_token));
        assert_eq!(
            q_mismatch.status,
            SsrBiasStatus::PhaseDiscontinuityNeedsReset
        );
        assert_eq!(
            q_mismatch.discontinuity_details,
            Some(SsrDiscontinuityDetails::MismatchedToken)
        );
    }

    #[test]
    fn test_preflight_and_whole_store_transactionality() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let t0_tow = 400_000.0;
        let reception = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow).unwrap();

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        // Populate store with valid initial state
        let mut store = SsrCorrectionStore::new();
        let valid_initial = HasMt1Message {
            header: HasMt1Header {
                toh_s: (t0_tow as u32 % 3600) as u16,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat,
                    nav_message: 0,
                    iode: 10,
                    radial_m: Some(0.1),
                    along_m: Some(0.2),
                    cross_m: Some(0.3),
                }],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    nav_message: 0,
                    correction_m: Some(-0.4),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: Some(1.2),
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: Some(0.1),
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };
        store.ingest_has_mt1(&valid_initial, reception).unwrap();
        let initial_store = store.clone();

        // Construct message with valid earlier blocks, but INVALID phase bias VI index (0 is reserved!)
        let invalid_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 30) as u16,
                mask: true,
                orbit: true,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 2,
            },
            mask: mask.clone(),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat,
                    nav_message: 0,
                    iode: 11,
                    radial_m: Some(0.5),
                    along_m: Some(0.6),
                    cross_m: Some(0.7),
                }],
            }),
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: Some(2.5),
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 15, // reserved and invalid per Table 23
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: Some(0.2),
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };

        let reception2 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 30.0).unwrap();
        let res = store.ingest_has_mt1(&invalid_msg, reception2);
        assert!(res.is_err(), "ingest must fail on invalid phase bias VI");
        assert_eq!(
            store, initial_store,
            "whole-store equality: preflight must leave earlier blocks untouched"
        );
    }

    #[test]
    fn test_time_aware_bias_queries_boundaries_and_exclusion() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let t0_tow = 500_000.0;
        let reception = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow).unwrap();
        let t_ref = has_mt1_reference_j2000_s(reception, (t0_tow as u32 % 3600) as u16).unwrap();

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        let phase_cycles = 0.20;
        let expected_phase_m = phase_cycles * (C_M_S / F_L1_HZ);

        // HAS message with validity interval 10 (300 s per Table 23, interval [t_ref, t_ref + 300])
        let msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (t0_tow as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 10,
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: Some(1.5),
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 10,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: Some(phase_cycles),
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };

        // Explicitly configure receiver staleness policy to >= 300s so receiver cap does not clip wire VI
        let mut store = SsrCorrectionStore::new().with_staleness(StalenessPolicy::seconds(300.0));
        store.ingest_has_mt1(&msg, reception).unwrap();

        // 1. Before reference: NotYetValid
        let q_before = store.query_code_bias(sat, has_sig(sat, 0), t_ref - 0.001);
        assert_eq!(q_before.status, SsrBiasStatus::NotYetValid);
        assert_eq!(q_before.bias_m, None);

        // 2. Exact reference: Available
        let q_exact = store.query_code_bias(sat, has_sig(sat, 0), t_ref);
        assert_eq!(q_exact.status, SsrBiasStatus::Available);
        assert_eq!(q_exact.bias_m, Some(1.5));

        // 3. Middle of VI: Available
        let q_mid = store.query_code_bias(sat, has_sig(sat, 0), t_ref + 150.0);
        assert_eq!(q_mid.status, SsrBiasStatus::Available);
        assert_eq!(q_mid.bias_m, Some(1.5));

        // 4. Exact VI end: Available
        let q_end = store.query_code_bias(sat, has_sig(sat, 0), t_ref + 300.0);
        assert_eq!(q_end.status, SsrBiasStatus::Available);
        assert_eq!(q_end.bias_m, Some(1.5));

        // 5. After VI end: Expired
        let q_after = store.query_code_bias(sat, has_sig(sat, 0), t_ref + 300.001);
        assert_eq!(q_after.status, SsrBiasStatus::Expired);
        assert_eq!(q_after.bias_m, None);

        // 6. Non-finite epochs: InvalidEpoch
        assert_eq!(
            store.query_code_bias(sat, has_sig(sat, 0), f64::NAN).status,
            SsrBiasStatus::InvalidEpoch
        );
        assert_eq!(
            store
                .query_code_bias(sat, has_sig(sat, 0), f64::INFINITY)
                .status,
            SsrBiasStatus::InvalidEpoch
        );
        assert_eq!(
            store
                .query_code_bias(sat, has_sig(sat, 0), f64::NEG_INFINITY)
                .status,
            SsrBiasStatus::InvalidEpoch
        );

        // 7. Active DNU excludes time-aware query without erasing raw metadata
        // Table 23 index 5 = 60 s (index 2 is 15 s)
        let dnu_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (t0_tow as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5, // 60 s per Table 23
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    nav_message: 0,
                    correction_m: None,
                    do_not_use: true,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        store.ingest_has_mt1(&dnu_msg, reception).unwrap();
        // Inside DNU [t_ref, t_ref + 60]: Excluded
        assert_eq!(
            store
                .query_code_bias(sat, has_sig(sat, 0), t_ref + 30.0)
                .status,
            SsrBiasStatus::Excluded
        );
        // Outside DNU (after 60s, but still within bias VI of 300s): Available!
        assert_eq!(
            store
                .query_code_bias(sat, has_sig(sat, 0), t_ref + 100.0)
                .status,
            SsrBiasStatus::Available
        );

        // 8. Raw untimed inspector returns numeric value regardless of epoch
        assert_eq!(store.code_bias(sat, has_sig(sat, 0)), Some(1.5));
        assert_eq!(
            store.phase_bias(sat, has_sig(sat, 0)),
            Some(expected_phase_m)
        );
    }

    #[test]
    fn test_ssr_corrected_ephemeris_has_forward_validity() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t0 = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let reception_t0 =
            GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S).unwrap();

        let record = broadcast
            .select_record_at(sat, t0)
            .expect("broadcast record");

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        // HAS message with orbit and clock, VI = 5 (60 s for GPS per table, or 300 s depending on index)
        let vi_s = has_validity_interval_s(5).unwrap();

        let has_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat,
                    nav_message: 0,
                    iode: record.issue_of_data.expect("broadcast issue").issue,
                    radial_m: Some(0.1),
                    along_m: Some(0.2),
                    cross_m: Some(0.3),
                }],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    nav_message: 0,
                    correction_m: Some(-0.4),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let mut store = SsrCorrectionStore::new();
        store.ingest_has_mt1(&has_msg, reception_t0).unwrap();

        let ephemeris = SsrCorrectedEphemeris::new(&broadcast, &store);

        // 1. Before reference epoch (t0 - 1.0): NOT fresh, corrected_state returns None
        assert!(
            ephemeris.corrected_state(sat, t0 - 1.0).is_none(),
            "HAS correction before TOH must not be applied"
        );

        // 2. Exact reference epoch (t0): fresh, corrected_state returns Some
        assert!(
            ephemeris.corrected_state(sat, t0).is_some(),
            "HAS correction at TOH must be applied"
        );

        // 3. Exact VI end (t0 + vi_s): fresh, corrected_state returns Some
        assert!(
            ephemeris.corrected_state(sat, t0 + vi_s).is_some(),
            "HAS correction at exact VI end must be applied"
        );

        // 4. After VI end (t0 + vi_s + 1.0): expired, corrected_state returns None
        assert!(
            ephemeris.corrected_state(sat, t0 + vi_s + 1.0).is_none(),
            "HAS correction after VI end must not be applied"
        );
    }

    #[test]
    fn test_first_unavailable_fresh_store_retains_full_metadata() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let t0_tow = 100_000.0;
        let reception_t0 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow).unwrap();
        let t_ref = has_mt1_reference_j2000_s(reception_t0, (t0_tow as u32 % 3600) as u16).unwrap();

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0, 9],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        let msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (t0_tow as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5, // 60s
                records: vec![
                    HasCodeBias {
                        sat,
                        signal_id: 0,
                        bias_m: None,
                    },
                    HasCodeBias {
                        sat,
                        signal_id: 9,
                        bias_m: None,
                    },
                ],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5, // 60s
                records: vec![
                    HasPhaseBias {
                        sat,
                        signal_id: 0,
                        bias_cycles: None,
                        discontinuity_indicator: 2,
                    },
                    HasPhaseBias {
                        sat,
                        signal_id: 9,
                        bias_cycles: None,
                        discontinuity_indicator: 1,
                    },
                ],
            }),
            padding_bits: Vec::new(),
        };
        assert!(msg.encode().is_ok());

        let mut store = SsrCorrectionStore::new();
        let report = store
            .ingest_has_mt1_with_report(&msg, reception_t0)
            .unwrap();
        assert_eq!(report.code_records.len(), 2);
        assert_eq!(report.phase_records.len(), 2);
        assert!(!report.has_refusals());
        assert_eq!(
            report.code_records[0].reason,
            IngestionActionReason::AcceptedFirstUnavailableWithMetadata
        );
        assert_eq!(
            report.code_records[1].reason,
            IngestionActionReason::AcceptedFirstUnavailableWithMetadata
        );
        assert_eq!(
            report.phase_records[0].reason,
            IngestionActionReason::AcceptedFirstUnavailableWithMetadata
        );
        assert_eq!(
            report.phase_records[1].reason,
            IngestionActionReason::AcceptedFirstUnavailableWithMetadata
        );
        assert_eq!(
            report.code_records[0].resulting_status,
            ActiveProvenanceStatus::ActiveHasUnavailable
        );
        assert_eq!(
            report.phase_records[0].resulting_status,
            ActiveProvenanceStatus::ActiveHasUnavailable
        );

        // Raw inspector returns None for both signals
        assert_eq!(store.code_bias(sat, has_sig(sat, 0)), None);
        assert_eq!(store.code_bias(sat, has_sig(sat, 9)), None);
        assert_eq!(store.phase_bias(sat, has_sig(sat, 0)), None);
        assert_eq!(store.phase_bias(sat, has_sig(sat, 9)), None);

        // Time-aware query before reference: NotYetValid (lifetime check applies before availability resolution)
        let q_before_c = store.query_code_bias(sat, has_sig(sat, 0), t_ref - 1.0);
        assert_eq!(q_before_c.status, SsrBiasStatus::NotYetValid);
        assert_eq!(
            q_before_c.details,
            SsrBiasResolutionDetails::EpochBeforeReference {
                ref_epoch_j2000_s: t_ref,
                query_epoch_j2000_s: t_ref - 1.0,
            }
        );

        let q_before_p = store.query_phase_bias(sat, has_sig(sat, 0), t_ref - 1.0, None);
        assert_eq!(q_before_p.status, SsrBiasStatus::NotYetValid);

        // Time-aware query inside forward VI: Unavailable with full metadata, token, and PDI
        let q_inside_c = store.query_code_bias(sat, has_sig(sat, 0), t_ref + 10.0);
        assert_eq!(q_inside_c.status, SsrBiasStatus::Unavailable);
        assert_eq!(
            q_inside_c.details,
            SsrBiasResolutionDetails::TransmittedUnavailable
        );
        assert_eq!(q_inside_c.bias_m, None);
        assert_eq!(q_inside_c.ref_epoch_j2000_s, Some(t_ref));
        assert_eq!(
            q_inside_c.lifetime,
            Some(SsrLifetime::GalileoHasValidityInterval(60.0))
        );

        let q_inside_p0 = store.query_phase_bias(sat, has_sig(sat, 0), t_ref + 10.0, None);
        assert_eq!(q_inside_p0.status, SsrBiasStatus::Unavailable);
        assert_eq!(
            q_inside_p0.details,
            SsrBiasResolutionDetails::TransmittedUnavailable
        );
        assert_eq!(q_inside_p0.bias_m, None);
        assert_eq!(q_inside_p0.bias_cycles, None);
        assert_eq!(
            q_inside_p0.discontinuity_indicator,
            Some(PhaseDiscontinuityIndicator::GalileoHasPdi(2))
        );
        assert!(q_inside_p0.continuity_token.is_some());

        let q_inside_p9 = store.query_phase_bias(sat, has_sig(sat, 9), t_ref + 10.0, None);
        assert_eq!(q_inside_p9.status, SsrBiasStatus::Unavailable);
        assert_eq!(
            q_inside_p9.discontinuity_indicator,
            Some(PhaseDiscontinuityIndicator::GalileoHasPdi(1))
        );

        // Time-aware query after VI: Expired
        let q_after_c = store.query_code_bias(sat, has_sig(sat, 0), t_ref + 61.0);
        assert_eq!(q_after_c.status, SsrBiasStatus::Expired);
        let q_after_p = store.query_phase_bias(sat, has_sig(sat, 0), t_ref + 61.0, None);
        assert_eq!(q_after_p.status, SsrBiasStatus::Expired);
    }

    #[test]
    fn test_has20_rtcm_has10_refusal_and_watermark_protection() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let t0_tow = 100_000.0;
        let reception_t10 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 10.0).unwrap();
        let reception_t20 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 20.0).unwrap();
        let reception_t25 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 25.0).unwrap();
        let reception_t30 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 30.0).unwrap();

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        // 1. HAS at T=20
        let has_t20 = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 20) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: Some(0.50),
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: Some(0.20),
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };
        assert!(has_t20.encode().is_ok());

        let mut store = SsrCorrectionStore::new();
        let rep20 = store
            .ingest_has_mt1_with_report(&has_t20, reception_t20)
            .unwrap();
        assert!(!rep20.has_refusals());
        assert_eq!(store.code_bias(sat, has_sig(sat, 0)), Some(0.50));
        let t_ref20 =
            has_mt1_reference_j2000_s(reception_t20, ((t0_tow as u32 % 3600) + 20) as u16).unwrap();
        let q_has20 = store.query_phase_bias(sat, has_sig(sat, 0), t_ref20, None);
        let token_has20 = q_has20.continuity_token.unwrap();
        assert_eq!(token_has20.source(), SsrSource::GalileoHas);

        // 2. RTCM at T=25 arrives: active value/status/continuity transitions to RTCM
        let mut rtcm_code_hdr = header(SsrKind::CodeBias);
        rtcm_code_hdr.epoch_time_s = (t0_tow + 25.0) as u32;
        let rtcm_code = SsrMessage {
            message_number: 1059,
            system: GnssSystem::Gps,
            kind: SsrKind::CodeBias,
            header: rtcm_code_hdr,
            orbit: Vec::new(),
            clock: Vec::new(),
            code_bias: vec![crate::rtcm::SsrCodeBiasRecord {
                satellite_id: 1,
                biases: vec![(0, 70)], // 0.70 m
            }],
            phase_bias: Vec::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };
        store.ingest_ssr(&rtcm_code, reception_t25).unwrap();

        let mut rtcm_phase_hdr = header(SsrKind::PhaseBias);
        rtcm_phase_hdr.epoch_time_s = (t0_tow + 25.0) as u32;
        let rtcm_phase = SsrMessage {
            message_number: 1265,
            system: GnssSystem::Gps,
            kind: SsrKind::PhaseBias,
            header: rtcm_phase_hdr,
            orbit: Vec::new(),
            clock: Vec::new(),
            code_bias: Vec::new(),
            phase_bias: vec![SsrPhaseBiasRecord {
                satellite_id: 1,
                yaw_angle: 0,
                yaw_rate: 0,
                biases: vec![SsrPhaseBiasSignal {
                    signal_id: 0,
                    integer_indicator: 0,
                    wide_lane_integer_indicator: 0,
                    discontinuity_counter: 10,
                    bias: 3500, // 0.35 m
                }],
            }],
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };
        store.ingest_ssr(&rtcm_phase, reception_t25).unwrap();

        // The stored values are the raw wire counts scaled by the documented RTCM
        // factors, not the decimal literals they approximate: 70 * 1e-2 and
        // 3500 * 1e-4 each land one ULP above 0.70 and 0.35, so compare the bits
        // of the same products the ingest path computes.
        let rtcm_code_bias_m = 70.0 * RTCM_SSR_CODE_BIAS_SCALE_M;
        let rtcm_phase_bias_m = 3500.0 * RTCM_SSR_PHASE_BIAS_SCALE_M;
        assert_eq!(
            store.code_bias(sat, has_sig(sat, 0)).unwrap().to_bits(),
            rtcm_code_bias_m.to_bits()
        );
        assert_eq!(
            store.phase_bias(sat, has_sig(sat, 0)).unwrap().to_bits(),
            rtcm_phase_bias_m.to_bits()
        );
        let q_rtcm = store.query_phase_bias(sat, has_sig(sat, 0), t_ref20 + 5.0, None);
        let token_rtcm = q_rtcm.continuity_token.unwrap();
        assert_eq!(token_rtcm.source(), SsrSource::RtcmSsr);

        // 3. Older HAS at T=10 arrives: REFUSED because HAS watermark is at T=20
        // Active RTCM value/status/continuity remain completely unchanged!
        let has_t10 = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 10) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: Some(0.40),
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: Some(0.10),
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };
        assert!(has_t10.encode().is_ok());

        let rep10 = store
            .ingest_has_mt1_with_report(&has_t10, reception_t10)
            .unwrap();
        assert!(rep10.has_refusals());
        assert_eq!(
            rep10.code_records[0].reason,
            IngestionActionReason::RefusedOlderThanWatermark
        );
        assert_eq!(
            rep10.phase_records[0].reason,
            IngestionActionReason::RefusedOlderThanWatermark
        );
        assert_eq!(
            rep10.code_records[0].resulting_status,
            ActiveProvenanceStatus::ActiveRtcmUsable
        );
        assert_eq!(
            rep10.phase_records[0].resulting_status,
            ActiveProvenanceStatus::ActiveRtcmUsable
        );

        // Active RTCM value and continuity are still preserved, bit for bit
        assert_eq!(
            store.code_bias(sat, has_sig(sat, 0)).unwrap().to_bits(),
            rtcm_code_bias_m.to_bits()
        );
        assert_eq!(
            store.phase_bias(sat, has_sig(sat, 0)).unwrap().to_bits(),
            rtcm_phase_bias_m.to_bits()
        );
        let q_rtcm_cont =
            store.query_phase_bias(sat, has_sig(sat, 0), t_ref20 + 5.0, Some(token_rtcm));
        assert_eq!(q_rtcm_cont.status, SsrBiasStatus::Available);
        assert_eq!(
            q_rtcm_cont.discontinuity_details,
            Some(SsrDiscontinuityDetails::Continuous)
        );

        // 4. Newer HAS at T=30 arrives: accepted, returns distinct token from token_has20
        let has_t30 = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 30) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: Some(0.60),
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: Some(0.30),
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };
        assert!(has_t30.encode().is_ok());

        let rep30 = store
            .ingest_has_mt1_with_report(&has_t30, reception_t30)
            .unwrap();
        assert!(!rep30.has_refusals());
        assert_eq!(
            rep30.code_records[0].reason,
            IngestionActionReason::AcceptedNewerRecord
        );
        assert_eq!(
            rep30.phase_records[0].reason,
            IngestionActionReason::AcceptedNewerRecord
        );
        assert_eq!(store.code_bias(sat, has_sig(sat, 0)), Some(0.60));

        let t_ref30 =
            has_mt1_reference_j2000_s(reception_t30, ((t0_tow as u32 % 3600) + 30) as u16).unwrap();
        let q_has30 = store.query_phase_bias(sat, has_sig(sat, 0), t_ref30, None);
        let token_has30 = q_has30.continuity_token.unwrap();
        assert_eq!(token_has30.source(), SsrSource::GalileoHas);
        assert_ne!(
            token_has30, token_has20,
            "newer HAS returns distinct token from past HAS arc"
        );
        assert_ne!(token_has30, token_rtcm);
    }

    #[test]
    fn test_equal_epoch_unavail_vs_usable_both_orders_with_rtcm_intervening() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let t0_tow = 200_000.0;
        let reception_t30 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 30.0).unwrap();
        let t_ref30 =
            has_mt1_reference_j2000_s(reception_t30, ((t0_tow as u32 % 3600) + 30) as u16).unwrap();

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        let has_t30_unavail = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 30) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: None,
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: None,
                    discontinuity_indicator: 1,
                }],
            }),
            padding_bits: Vec::new(),
        };

        let has_t30_usable = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 30) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 2,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: Some(0.50),
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: Some(0.20),
                    discontinuity_indicator: 1,
                }],
            }),
            padding_bits: Vec::new(),
        };

        let mut rtcm_code_hdr = header(SsrKind::CodeBias);
        rtcm_code_hdr.epoch_time_s = (t0_tow + 30.0) as u32;
        let rtcm_code = SsrMessage {
            message_number: 1059,
            system: GnssSystem::Gps,
            kind: SsrKind::CodeBias,
            header: rtcm_code_hdr,
            orbit: Vec::new(),
            clock: Vec::new(),
            code_bias: vec![crate::rtcm::SsrCodeBiasRecord {
                satellite_id: 1,
                biases: vec![(0, 30)], // 0.30 m
            }],
            phase_bias: Vec::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };

        // The intervening provider update carries a real RTCM phase bias too, so the
        // phase half of this ordering is exercised against an active RTCM record.
        let mut rtcm_phase_hdr = header(SsrKind::PhaseBias);
        rtcm_phase_hdr.update_interval = 5;
        rtcm_phase_hdr.epoch_time_s = (t0_tow + 15.0) as u32;
        let rtcm_phase = SsrMessage {
            message_number: 1265,
            system: GnssSystem::Gps,
            kind: SsrKind::PhaseBias,
            header: rtcm_phase_hdr,
            orbit: Vec::new(),
            clock: Vec::new(),
            code_bias: Vec::new(),
            phase_bias: vec![SsrPhaseBiasRecord {
                satellite_id: sat.prn,
                yaw_angle: 0,
                yaw_rate: 0,
                biases: vec![SsrPhaseBiasSignal {
                    signal_id: 0,
                    integer_indicator: 0,
                    wide_lane_integer_indicator: 0,
                    discontinuity_counter: 9,
                    bias: 1500, // 0.1500 m
                }],
            }],
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };

        // Order A: HAS unavail -> RTCM -> HAS usable (usable beats unavail at equal epoch)
        let mut store_a = SsrCorrectionStore::new();
        store_a
            .ingest_has_mt1_with_report(&has_t30_unavail, reception_t30)
            .unwrap();
        assert_eq!(store_a.code_bias(sat, has_sig(sat, 0)), None);

        store_a.ingest_ssr(&rtcm_code, reception_t30).unwrap();
        store_a.ingest_ssr(&rtcm_phase, reception_t30).unwrap();
        assert_eq!(store_a.code_bias(sat, has_sig(sat, 0)), Some(0.30));
        assert_eq!(store_a.phase_bias(sat, has_sig(sat, 0)), Some(0.15));

        let rep_a = store_a
            .ingest_has_mt1_with_report(&has_t30_usable, reception_t30)
            .unwrap();
        assert_eq!(
            rep_a.code_records[0].reason,
            IngestionActionReason::AcceptedUsableOverUnavailable
        );
        assert_eq!(
            store_a.code_bias(sat, has_sig(sat, 0)),
            Some(0.50),
            "usable HAS wins over unavail at equal epoch"
        );
        // One ordered typed report per incoming record, phase included.
        assert_eq!(rep_a.code_records.len(), 1);
        assert_eq!(rep_a.phase_records.len(), 1);
        assert_eq!(
            rep_a.phase_records[0].reason,
            IngestionActionReason::AcceptedUsableOverUnavailable
        );
        assert_eq!(
            rep_a.phase_records[0].resulting_status,
            ActiveProvenanceStatus::ActiveHasUsable
        );
        assert_eq!(rep_a.phase_records[0].native_cycles, Some(0.20));
        assert_eq!(rep_a.phase_records[0].pdi, 1);
        // The equal-epoch usable record displaces the intervening RTCM phase bias.
        let has_phase_m = 0.20 * (C_M_S / F_L1_HZ);
        assert_eq!(store_a.phase_bias(sat, has_sig(sat, 0)), Some(has_phase_m));

        // Order B: HAS usable -> RTCM -> HAS unavail (unavail loses to usable at equal epoch, RTCM preserved)
        let mut store_b = SsrCorrectionStore::new();
        store_b
            .ingest_has_mt1_with_report(&has_t30_usable, reception_t30)
            .unwrap();
        assert_eq!(store_b.code_bias(sat, has_sig(sat, 0)), Some(0.50));

        store_b.ingest_ssr(&rtcm_code, reception_t30).unwrap();
        store_b.ingest_ssr(&rtcm_phase, reception_t30).unwrap();
        assert_eq!(store_b.code_bias(sat, has_sig(sat, 0)), Some(0.30));
        let rtcm_phase_token = store_b
            .query_phase_bias(sat, has_sig(sat, 0), t_ref30, None)
            .continuity_token
            .unwrap();
        assert_eq!(rtcm_phase_token.source(), SsrSource::RtcmSsr);

        let rep_b = store_b
            .ingest_has_mt1_with_report(&has_t30_unavail, reception_t30)
            .unwrap();
        assert_eq!(
            rep_b.code_records[0].reason,
            IngestionActionReason::RefusedEqualEpochUnavailableUnderUsable
        );
        assert_eq!(
            store_b.code_bias(sat, has_sig(sat, 0)),
            Some(0.30),
            "equal epoch unavail loses to prior usable, preserving active RTCM"
        );
        assert_eq!(rep_b.code_records.len(), 1);
        assert_eq!(rep_b.phase_records.len(), 1);
        assert_eq!(
            rep_b.phase_records[0].reason,
            IngestionActionReason::RefusedEqualEpochUnavailableUnderUsable
        );
        assert!(rep_b.phase_records[0].reason.is_refused());
        // A refused record still reports its own incoming values and the active
        // correction it failed to displace.
        assert_eq!(rep_b.phase_records[0].native_cycles, None);
        assert_eq!(rep_b.phase_records[0].pdi, 1);
        assert_eq!(
            rep_b.phase_records[0].resulting_status,
            ActiveProvenanceStatus::ActiveRtcmUsable
        );
        // The refused equal-epoch unavailable record left the intervening RTCM phase
        // bias and its acknowledgement exactly as they were.
        assert_eq!(store_b.phase_bias(sat, has_sig(sat, 0)), Some(0.15));
        let q_b = store_b.query_phase_bias(sat, has_sig(sat, 0), t_ref30, Some(rtcm_phase_token));
        assert_eq!(q_b.status, SsrBiasStatus::Available);
        assert_eq!(q_b.bias_m, Some(0.15));
        assert_eq!(
            q_b.discontinuity_details,
            Some(SsrDiscontinuityDetails::Continuous)
        );

        // The refusal left the per-signal HAS watermark at the usable record that
        // arrived before the intervening RTCM update.
        let entry_b = store_b
            .corrections
            .get(&sat)
            .unwrap()
            .phase_bias
            .signals
            .get(&has_sig(sat, 0))
            .unwrap();
        assert_eq!(
            entry_b.has_watermark,
            Some(HasStatusWatermark {
                ref_epoch_j2000_s: t_ref30,
                status: HasWatermarkStatus::Usable,
                pdi: Some(1),
            })
        );
    }

    #[test]
    fn test_newer_has_none_with_current_rtcm_preserves_numeric_rtcm_and_persists_watermark() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let t0_tow = 300_000.0;
        let reception_t50 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 50.0).unwrap();
        let reception_t55 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 55.0).unwrap();
        let reception_t60 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 60.0).unwrap();

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        // 1. Ingest RTCM at T=50
        let mut rtcm_code_hdr = header(SsrKind::CodeBias);
        rtcm_code_hdr.epoch_time_s = (t0_tow + 50.0) as u32;
        let rtcm_code = SsrMessage {
            message_number: 1059,
            system: GnssSystem::Gps,
            kind: SsrKind::CodeBias,
            header: rtcm_code_hdr,
            orbit: Vec::new(),
            clock: Vec::new(),
            code_bias: vec![crate::rtcm::SsrCodeBiasRecord {
                satellite_id: 1,
                biases: vec![(0, 25)], // 0.25 m
            }],
            phase_bias: Vec::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };
        let mut store = SsrCorrectionStore::new();
        store.ingest_ssr(&rtcm_code, reception_t50).unwrap();
        assert_eq!(store.code_bias(sat, has_sig(sat, 0)), Some(0.25));

        // 2. HAS unavailable arrives at T=60 (with PDI = 2)
        let has_t60_unavail = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 60) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: None,
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: None,
                    discontinuity_indicator: 2,
                }],
            }),
            padding_bits: Vec::new(),
        };
        let rep60 = store
            .ingest_has_mt1_with_report(&has_t60_unavail, reception_t60)
            .unwrap();
        assert_eq!(
            rep60.code_records[0].reason,
            IngestionActionReason::RetainedActiveRtcmOnHasUnavailable
        );
        assert_eq!(
            rep60.code_records[0].resulting_status,
            ActiveProvenanceStatus::ActiveRtcmUsable
        );
        // Active RTCM numeric value remains preserved!
        assert_eq!(store.code_bias(sat, has_sig(sat, 0)), Some(0.25));

        // 3. Older HAS at T=55 arrives: refused because HAS watermark is at T=60!
        let has_t55 = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 55) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: Some(0.80),
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: Some(0.40),
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };
        let rep55 = store
            .ingest_has_mt1_with_report(&has_t55, reception_t55)
            .unwrap();
        assert_eq!(
            rep55.code_records[0].reason,
            IngestionActionReason::RefusedOlderThanWatermark
        );
        assert_eq!(
            store.code_bias(sat, has_sig(sat, 0)),
            Some(0.25),
            "older HAS at T55 cannot overwrite RTCM protected by T60 watermark"
        );
    }

    #[test]
    fn test_same_epoch_pdi_transitions_and_wrap_invalidation() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let t0_tow = 400_000.0;
        let reception_t0 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow).unwrap();
        let t_ref = has_mt1_reference_j2000_s(reception_t0, (t0_tow as u32 % 3600) as u16).unwrap();

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        // The solution identity is deliberately held constant across the whole cycle so
        // that only the PDI revises. A changed IOD set id is a solution change and is
        // covered separately by test_source_switch_a_b_a_invalidates_old_token.
        let make_msg = |pdi: u8| -> HasMt1Message {
            HasMt1Message {
                header: HasMt1Header {
                    toh_s: (t0_tow as u32 % 3600) as u16,
                    mask: true,
                    orbit: false,
                    clock_full_set: false,
                    clock_subset: false,
                    code_bias: false,
                    phase_bias: true,
                    reserved: 0,
                    mask_id: 1,
                    iod_set_id: 1,
                },
                mask: mask.clone(),
                orbit: None,
                clock_full_set: None,
                clock_subset: None,
                code_bias: None,
                phase_bias: Some(HasPhaseBiasBlock {
                    validity_interval: 5,
                    records: vec![HasPhaseBias {
                        sat,
                        signal_id: 0,
                        bias_cycles: Some(0.10),
                        discontinuity_indicator: pdi,
                    }],
                }),
                padding_bits: Vec::new(),
            }
        };

        let mut store = SsrCorrectionStore::new();

        // 1. PDI = 0 at T=0
        store
            .ingest_has_mt1_with_report(&make_msg(0), reception_t0)
            .unwrap();
        let q0 = store.query_phase_bias(sat, has_sig(sat, 0), t_ref, None);
        assert_eq!(q0.status, SsrBiasStatus::Available);
        assert_eq!(
            q0.discontinuity_details,
            Some(SsrDiscontinuityDetails::InitialTokenEstablished)
        );
        let token_pdi0 = q0.continuity_token.unwrap();
        assert_eq!(token_pdi0.raw_indicator(), 0);
        assert_eq!(token_pdi0.generation(), 0);

        // 2. Same-epoch revision PDI 0 -> 1 (cannot be blanket <= rejected!)
        let rep1 = store
            .ingest_has_mt1_with_report(&make_msg(1), reception_t0)
            .unwrap();
        assert_eq!(
            rep1.phase_records[0].reason,
            IngestionActionReason::AcceptedUpdatedRecord
        );

        let q1 = store.query_phase_bias(sat, has_sig(sat, 0), t_ref, Some(token_pdi0));
        assert_eq!(q1.status, SsrBiasStatus::PhaseDiscontinuityNeedsReset);
        assert_eq!(
            q1.discontinuity_details,
            Some(SsrDiscontinuityDetails::HasPdiChanged {
                previous: 0,
                current: 1
            })
        );
        let token_pdi1 = q1.continuity_token.unwrap();
        assert_eq!(token_pdi1.raw_indicator(), 1);
        assert_eq!(token_pdi1.generation(), 1);

        // Acking with token_pdi1 succeeds
        let q1_ack = store.query_phase_bias(sat, has_sig(sat, 0), t_ref, Some(token_pdi1));
        assert_eq!(q1_ack.status, SsrBiasStatus::Available);
        assert_eq!(
            q1_ack.discontinuity_details,
            Some(SsrDiscontinuityDetails::Continuous)
        );

        // 3. Same-epoch revision PDI 1 -> 2
        store
            .ingest_has_mt1_with_report(&make_msg(2), reception_t0)
            .unwrap();
        // 4. Same-epoch revision PDI 2 -> 3
        store
            .ingest_has_mt1_with_report(&make_msg(3), reception_t0)
            .unwrap();
        // 5. Same-epoch revision PDI 3 -> 0 (wrapped back to 0 at the same epoch!)
        store
            .ingest_has_mt1_with_report(&make_msg(0), reception_t0)
            .unwrap();

        let q_curr = store.query_phase_bias(sat, has_sig(sat, 0), t_ref, None);
        let token_wrap0 = q_curr.continuity_token.unwrap();
        assert_eq!(token_wrap0.raw_indicator(), 0);
        assert_eq!(token_wrap0.generation(), 4);

        // Stale token_pdi0 from generation 0 MUST NOT be accepted or report false HasPdiChanged{0, 0}
        let q_stale = store.query_phase_bias(sat, has_sig(sat, 0), t_ref, Some(token_pdi0));
        assert_eq!(q_stale.status, SsrBiasStatus::PhaseDiscontinuityNeedsReset);
        assert_eq!(
            q_stale.discontinuity_details,
            Some(SsrDiscontinuityDetails::StaleToken)
        );
    }

    #[test]
    fn test_source_switch_a_b_a_invalidates_old_token() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let t0_tow = 500_000.0;
        let reception_t10 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 10.0).unwrap();
        let reception_t20 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 20.0).unwrap();
        let reception_t30 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 30.0).unwrap();
        let t_ref10 =
            has_mt1_reference_j2000_s(reception_t10, ((t0_tow as u32 % 3600) + 10) as u16).unwrap();
        let t_ref30 =
            has_mt1_reference_j2000_s(reception_t30, ((t0_tow as u32 % 3600) + 30) as u16).unwrap();

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        // 1. Source A: HAS at T=10, PDI = 0
        let has_t10 = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 10) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: Some(0.10),
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };
        let mut store = SsrCorrectionStore::new();
        store
            .ingest_has_mt1_with_report(&has_t10, reception_t10)
            .unwrap();
        let q1 = store.query_phase_bias(sat, has_sig(sat, 0), t_ref10, None);
        let token_has1 = q1.continuity_token.unwrap();
        assert_eq!(token_has1.source(), SsrSource::GalileoHas);

        // 2. Source B: RTCM at T=20
        let mut rtcm_hdr = header(SsrKind::PhaseBias);
        rtcm_hdr.epoch_time_s = (t0_tow + 20.0) as u32;
        let rtcm_phase = SsrMessage {
            message_number: 1265,
            system: GnssSystem::Gps,
            kind: SsrKind::PhaseBias,
            header: rtcm_hdr,
            orbit: Vec::new(),
            clock: Vec::new(),
            code_bias: Vec::new(),
            phase_bias: vec![SsrPhaseBiasRecord {
                satellite_id: 1,
                yaw_angle: 0,
                yaw_rate: 0,
                biases: vec![SsrPhaseBiasSignal {
                    signal_id: 0,
                    integer_indicator: 0,
                    wide_lane_integer_indicator: 0,
                    discontinuity_counter: 5,
                    bias: 2000,
                }],
            }],
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };
        store.ingest_ssr(&rtcm_phase, reception_t20).unwrap();
        let q2 = store.query_phase_bias(sat, has_sig(sat, 0), t_ref10 + 10.0, None);
        let token_rtcm = q2.continuity_token.unwrap();
        assert_eq!(token_rtcm.source(), SsrSource::RtcmSsr);

        // 3. Return to Source A: HAS at T=30, PDI = 0 (same raw PDI as step 1!)
        let has_t30 = HasMt1Message {
            header: HasMt1Header {
                toh_s: ((t0_tow as u32 % 3600) + 30) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: Some(0.10),
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };
        store
            .ingest_has_mt1_with_report(&has_t30, reception_t30)
            .unwrap();

        // HAS -> RTCM -> HAS must NOT revive the old token from step 1
        let q3 = store.query_phase_bias(sat, has_sig(sat, 0), t_ref30, Some(token_has1));
        assert_eq!(q3.status, SsrBiasStatus::PhaseDiscontinuityNeedsReset);
        assert_eq!(
            q3.discontinuity_details,
            Some(SsrDiscontinuityDetails::StaleToken)
        );

        // The RTCM token is also rejected at step 3, and because it names the other
        // stream the diagnostic identifies the solution change rather than mere staleness.
        let q4 = store.query_phase_bias(sat, has_sig(sat, 0), t_ref30, Some(token_rtcm));
        assert_eq!(q4.status, SsrBiasStatus::PhaseDiscontinuityNeedsReset);
        assert_eq!(
            q4.discontinuity_details,
            Some(SsrDiscontinuityDetails::SolutionChanged {
                previous: SsrSolution {
                    source: SsrSource::RtcmSsr,
                    provider_id: token_rtcm.provider_id(),
                    solution_id: token_rtcm.solution_id(),
                },
                current: SsrSolution {
                    source: SsrSource::GalileoHas,
                    provider_id: 1,
                    solution_id: 1,
                },
            })
        );
        assert_eq!(
            q4.details,
            SsrBiasResolutionDetails::PhaseDiscontinuity(q4.discontinuity_details.unwrap())
        );
    }

    #[test]
    fn test_phase_query_rejects_future_mismatched_and_stale_tokens() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let t0_tow = 600_000.0;
        let reception = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow).unwrap();
        let t_ref = has_mt1_reference_j2000_s(reception, (t0_tow as u32 % 3600) as u16).unwrap();

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat1.prn, sat2.prn],
                signals: vec![0, 9],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        let msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (t0_tow as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasPhaseBias {
                        sat: sat1,
                        signal_id: 0,
                        bias_cycles: Some(0.10),
                        discontinuity_indicator: 0,
                    },
                    HasPhaseBias {
                        sat: sat1,
                        signal_id: 9,
                        bias_cycles: Some(0.20),
                        discontinuity_indicator: 0,
                    },
                    HasPhaseBias {
                        sat: sat2,
                        signal_id: 0,
                        bias_cycles: Some(0.30),
                        discontinuity_indicator: 0,
                    },
                    HasPhaseBias {
                        sat: sat2,
                        signal_id: 9,
                        bias_cycles: Some(0.40),
                        discontinuity_indicator: 0,
                    },
                ],
            }),
            padding_bits: Vec::new(),
        };

        let mut store = SsrCorrectionStore::new();
        store.ingest_has_mt1_with_report(&msg, reception).unwrap();

        let token_sat1_sig0 = store
            .query_phase_bias(sat1, has_sig(sat1, 0), t_ref, None)
            .continuity_token
            .unwrap();
        let token_sat1_sig9 = store
            .query_phase_bias(sat1, has_sig(sat1, 9), t_ref, None)
            .continuity_token
            .unwrap();
        let token_sat2_sig0 = store
            .query_phase_bias(sat2, has_sig(sat2, 0), t_ref, None)
            .continuity_token
            .unwrap();

        // Query sat1 sig0 with token from sat1 sig9 (mismatched signal)
        let q_sig_mismatch =
            store.query_phase_bias(sat1, has_sig(sat1, 0), t_ref, Some(token_sat1_sig9));
        assert_eq!(
            q_sig_mismatch.status,
            SsrBiasStatus::PhaseDiscontinuityNeedsReset
        );
        assert_eq!(
            q_sig_mismatch.discontinuity_details,
            Some(SsrDiscontinuityDetails::MismatchedToken)
        );

        // Query sat1 sig0 with token from sat2 sig0 (mismatched satellite)
        let q_sat_mismatch =
            store.query_phase_bias(sat1, has_sig(sat1, 0), t_ref, Some(token_sat2_sig0));
        assert_eq!(
            q_sat_mismatch.status,
            SsrBiasStatus::PhaseDiscontinuityNeedsReset
        );
        assert_eq!(
            q_sat_mismatch.discontinuity_details,
            Some(SsrDiscontinuityDetails::MismatchedToken)
        );

        // A token carrying a later generation is obtained from a second store that was
        // driven through further same-epoch PDI revisions, so nothing here is forged.
        let mut ahead_store = SsrCorrectionStore::new();
        ahead_store
            .ingest_has_mt1_with_report(&msg, reception)
            .unwrap();
        for pdi in [1u8, 2, 3] {
            let mut revised = msg.clone();
            if let Some(pb) = revised.phase_bias.as_mut() {
                for rec in &mut pb.records {
                    rec.discontinuity_indicator = pdi;
                }
            }
            ahead_store
                .ingest_has_mt1_with_report(&revised, reception)
                .unwrap();
        }
        let ahead_token = ahead_store
            .query_phase_bias(sat1, has_sig(sat1, 0), t_ref, None)
            .continuity_token
            .unwrap();
        assert!(ahead_token.generation() > token_sat1_sig0.generation());
        assert_eq!(
            ahead_token.continuity_ref_epoch_j2000_s().to_bits(),
            token_sat1_sig0.continuity_ref_epoch_j2000_s().to_bits(),
            "both stores began this arc at the same reference epoch"
        );

        let q_future = store.query_phase_bias(sat1, has_sig(sat1, 0), t_ref, Some(ahead_token));
        assert_eq!(q_future.status, SsrBiasStatus::PhaseDiscontinuityNeedsReset);
        assert_eq!(
            q_future.discontinuity_details,
            Some(SsrDiscontinuityDetails::FutureToken)
        );

        // Exactly the current token still succeeds.
        let q_exact = store.query_phase_bias(sat1, has_sig(sat1, 0), t_ref, Some(token_sat1_sig0));
        assert_eq!(q_exact.status, SsrBiasStatus::Available);
        assert_eq!(
            q_exact.discontinuity_details,
            Some(SsrDiscontinuityDetails::Continuous)
        );
    }

    #[test]
    fn test_retained_rtcm_survives_repeated_and_revised_has_unavailable_for_code_and_phase() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let t0_tow = 200_000.0;
        let toh = |offset: u32| ((t0_tow as u32 % 3600) + offset) as u16;
        let reception =
            |offset: f64| GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + offset).unwrap();
        let reception_t20 = reception(20.0);
        let reception_t25 = reception(25.0);
        let reception_t30 = reception(30.0);
        let reception_t40 = reception(40.0);
        let t_ref20 = has_mt1_reference_j2000_s(reception_t20, toh(20)).unwrap();
        let t_ref30 = has_mt1_reference_j2000_s(reception_t30, toh(30)).unwrap();
        let t_ref40 = has_mt1_reference_j2000_s(reception_t40, toh(40)).unwrap();

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        let has_msg = |offset: u32,
                       iod: u8,
                       code_m: Option<f64>,
                       cycles: Option<f64>,
                       pdi: u8|
         -> HasMt1Message {
            HasMt1Message {
                header: HasMt1Header {
                    toh_s: toh(offset),
                    mask: true,
                    orbit: false,
                    clock_full_set: false,
                    clock_subset: false,
                    code_bias: true,
                    phase_bias: true,
                    reserved: 0,
                    mask_id: 1,
                    iod_set_id: iod,
                },
                mask: mask.clone(),
                orbit: None,
                clock_full_set: None,
                clock_subset: None,
                code_bias: Some(HasCodeBiasBlock {
                    validity_interval: 5,
                    records: vec![HasCodeBias {
                        sat,
                        signal_id: 0,
                        bias_m: code_m,
                    }],
                }),
                phase_bias: Some(HasPhaseBiasBlock {
                    validity_interval: 5,
                    records: vec![HasPhaseBias {
                        sat,
                        signal_id: 0,
                        bias_cycles: cycles,
                        discontinuity_indicator: pdi,
                    }],
                }),
                padding_bits: Vec::new(),
            }
        };

        // Real RTCM code and phase bias messages. The 30 s update interval keeps the
        // RTCM records valid across every HAS reference epoch this test queries.
        let mut rtcm_code_hdr = header(SsrKind::CodeBias);
        rtcm_code_hdr.update_interval = 5;
        rtcm_code_hdr.epoch_time_s = (t0_tow + 15.0) as u32;
        let rtcm_code = SsrMessage {
            message_number: 1059,
            system: GnssSystem::Gps,
            kind: SsrKind::CodeBias,
            header: rtcm_code_hdr,
            orbit: Vec::new(),
            clock: Vec::new(),
            code_bias: vec![crate::rtcm::SsrCodeBiasRecord {
                satellite_id: sat.prn,
                biases: vec![(0, 30)], // 0.30 m
            }],
            phase_bias: Vec::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };

        let mut rtcm_phase_hdr = header(SsrKind::PhaseBias);
        rtcm_phase_hdr.update_interval = 5;
        rtcm_phase_hdr.epoch_time_s = (t0_tow + 15.0) as u32;
        let rtcm_phase = SsrMessage {
            message_number: 1265,
            system: GnssSystem::Gps,
            kind: SsrKind::PhaseBias,
            header: rtcm_phase_hdr,
            orbit: Vec::new(),
            clock: Vec::new(),
            code_bias: Vec::new(),
            phase_bias: vec![SsrPhaseBiasRecord {
                satellite_id: sat.prn,
                yaw_angle: 0,
                yaw_rate: 0,
                biases: vec![SsrPhaseBiasSignal {
                    signal_id: 0,
                    integer_indicator: 0,
                    wide_lane_integer_indicator: 0,
                    discontinuity_counter: 7,
                    bias: 2000, // 0.2000 m
                }],
            }],
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };

        // 1. HAS usable at T+20 establishes an active HAS arc.
        let mut store = SsrCorrectionStore::new();
        store
            .ingest_has_mt1_with_report(&has_msg(20, 2, Some(0.50), Some(0.20), 1), reception_t20)
            .unwrap();
        assert_eq!(store.code_bias(sat, has_sig(sat, 0)), Some(0.50));
        assert_eq!(
            store
                .query_phase_bias(sat, has_sig(sat, 0), t_ref20, None)
                .continuity_token
                .unwrap()
                .source(),
            SsrSource::GalileoHas
        );

        // 2. RTCM code and phase take over the active correction for this signal.
        store.ingest_ssr(&rtcm_code, reception_t20).unwrap();
        store.ingest_ssr(&rtcm_phase, reception_t20).unwrap();
        assert_eq!(store.code_bias(sat, has_sig(sat, 0)), Some(0.30));

        let q_after_rtcm = store.query_phase_bias(sat, has_sig(sat, 0), t_ref30, None);
        assert_eq!(
            q_after_rtcm.status,
            SsrBiasStatus::Available,
            "with no acknowledgement the caller starts a new arc; no reset is demanded"
        );
        assert!(
            matches!(
                q_after_rtcm.discontinuity_details,
                Some(SsrDiscontinuityDetails::SolutionChanged { .. })
            ),
            "the stream switch is still reported in the details"
        );
        let rtcm_token = q_after_rtcm.continuity_token.unwrap();
        assert_eq!(rtcm_token.source(), SsrSource::RtcmSsr);
        assert_eq!(rtcm_token.raw_indicator(), 7);
        let rtcm_generation = rtcm_token.generation();

        let q_acked = store.query_phase_bias(sat, has_sig(sat, 0), t_ref30, Some(rtcm_token));
        assert_eq!(q_acked.status, SsrBiasStatus::Available);
        assert_eq!(q_acked.bias_m, Some(0.2));
        assert_eq!(
            q_acked.discontinuity_details,
            Some(SsrDiscontinuityDetails::Continuous)
        );
        assert_eq!(
            q_acked.discontinuity_indicator,
            Some(PhaseDiscontinuityIndicator::RtcmDiscontinuityCounter(7))
        );

        // Snapshot the exact active numeric state that must survive every HAS
        // unavailable record below.
        let active_before = store
            .corrections
            .get(&sat)
            .unwrap()
            .phase_bias
            .signals
            .get(&has_sig(sat, 0))
            .unwrap()
            .clone();
        let active_code_before = store
            .corrections
            .get(&sat)
            .unwrap()
            .code_bias
            .signals
            .get(&has_sig(sat, 0))
            .unwrap()
            .active
            .clone();
        assert_eq!(active_before.active.as_ref().unwrap().value_m, Some(0.2));

        // 3. HAS unavailable at T+30, then the identical record again, then the same
        //    epoch again with a revised PDI. None of them may disturb active RTCM.
        let unavailable_inputs: [(u8, IngestionActionReason); 3] = [
            (2, IngestionActionReason::RetainedActiveRtcmOnHasUnavailable),
            (2, IngestionActionReason::RetainedActiveRtcmOnHasUnavailable),
            (3, IngestionActionReason::RetainedActiveRtcmOnHasUnavailable),
        ];
        for (pdi, expected_reason) in unavailable_inputs {
            let report = store
                .ingest_has_mt1_with_report(&has_msg(30, 3, None, None, pdi), reception_t30)
                .unwrap();

            assert_eq!(report.code_records.len(), 1);
            assert_eq!(report.phase_records.len(), 1);

            let code_rec = &report.code_records[0];
            assert_eq!(
                code_rec.reason,
                IngestionActionReason::RetainedActiveRtcmOnHasUnavailable
            );
            assert!(code_rec.reason.is_retained());
            assert_eq!(
                code_rec.resulting_status,
                ActiveProvenanceStatus::ActiveRtcmUsable
            );
            assert_eq!(code_rec.native_bias_m, None);
            assert_eq!(code_rec.ref_epoch_j2000_s, t_ref30);

            let phase_rec = &report.phase_records[0];
            assert_eq!(phase_rec.reason, expected_reason);
            assert!(phase_rec.reason.is_retained());
            assert_eq!(
                phase_rec.resulting_status,
                ActiveProvenanceStatus::ActiveRtcmUsable
            );
            // Native cycles and PDI describe the incoming record verbatim.
            assert_eq!(phase_rec.native_cycles, None);
            assert_eq!(phase_rec.pdi, pdi);
            // The reported token describes the resulting active correction, which is
            // still the untouched RTCM arc.
            assert_eq!(phase_rec.continuity_token, Some(rtcm_token));

            // Active numeric record, solution, IOD, lifetime, token, prior break and
            // generation are all exactly as they were.
            let entry = store
                .corrections
                .get(&sat)
                .unwrap()
                .phase_bias
                .signals
                .get(&has_sig(sat, 0))
                .unwrap();
            assert_eq!(entry.active, active_before.active);
            assert_eq!(entry.arc_generation, active_before.arc_generation);
            assert_eq!(entry.prior_break, active_before.prior_break);
            assert_eq!(
                entry.prior_rtcm_continuity,
                active_before.prior_rtcm_continuity
            );
            assert_eq!(
                store
                    .corrections
                    .get(&sat)
                    .unwrap()
                    .code_bias
                    .signals
                    .get(&has_sig(sat, 0))
                    .unwrap()
                    .active,
                active_code_before
            );
            assert_eq!(store.code_bias(sat, has_sig(sat, 0)), Some(0.30));

            // The HAS watermark still advanced, carrying the incoming epoch and PDI.
            assert_eq!(
                entry.has_watermark,
                Some(HasStatusWatermark {
                    ref_epoch_j2000_s: t_ref30,
                    status: HasWatermarkStatus::Unavailable,
                    pdi: Some(pdi),
                })
            );

            // Queries with no acknowledgement and with the current RTCM
            // acknowledgement both behave exactly as before the retained input.
            let q_none = store.query_phase_bias(sat, has_sig(sat, 0), t_ref30, None);
            assert_eq!(q_none.status, SsrBiasStatus::Available);
            assert!(matches!(
                q_none.discontinuity_details,
                Some(SsrDiscontinuityDetails::SolutionChanged { .. })
            ));
            assert_eq!(q_none.continuity_token, Some(rtcm_token));
            let q_ack = store.query_phase_bias(sat, has_sig(sat, 0), t_ref30, Some(rtcm_token));
            assert_eq!(q_ack.status, SsrBiasStatus::Available);
            assert_eq!(q_ack.bias_m, Some(0.2));
            assert_eq!(
                q_ack.discontinuity_details,
                Some(SsrDiscontinuityDetails::Continuous)
            );
        }

        // 4. An older HAS record at T+25 is refused by the T+30 watermark.
        let report25 = store
            .ingest_has_mt1_with_report(&has_msg(25, 4, Some(0.99), Some(0.99), 0), reception_t25)
            .unwrap();
        assert_eq!(
            report25.code_records[0].reason,
            IngestionActionReason::RefusedOlderThanWatermark
        );
        assert_eq!(
            report25.phase_records[0].reason,
            IngestionActionReason::RefusedOlderThanWatermark
        );
        assert!(report25.has_refusals());
        // A refused record reports its own incoming values, not the active ones.
        assert_eq!(report25.phase_records[0].native_cycles, Some(0.99));
        assert_eq!(report25.phase_records[0].pdi, 0);
        assert_eq!(report25.phase_records[0].continuity_token, Some(rtcm_token));
        assert_eq!(
            report25.phase_records[0].resulting_status,
            ActiveProvenanceStatus::ActiveRtcmUsable
        );
        assert_eq!(store.code_bias(sat, has_sig(sat, 0)), Some(0.30));
        let entry_after_refusal = store
            .corrections
            .get(&sat)
            .unwrap()
            .phase_bias
            .signals
            .get(&has_sig(sat, 0))
            .unwrap();
        assert_eq!(entry_after_refusal.active, active_before.active);
        assert_eq!(
            entry_after_refusal.arc_generation,
            active_before.arc_generation
        );
        // The refused record leaves the watermark at the T+30 revision it had reached.
        assert_eq!(
            entry_after_refusal.has_watermark,
            Some(HasStatusWatermark {
                ref_epoch_j2000_s: t_ref30,
                status: HasWatermarkStatus::Unavailable,
                pdi: Some(3),
            })
        );

        // 5. A newer usable HAS record at T+40 reactivates HAS with a distinct token.
        let report40 = store
            .ingest_has_mt1_with_report(&has_msg(40, 5, Some(0.77), Some(0.33), 0), reception_t40)
            .unwrap();
        assert_eq!(
            report40.code_records[0].reason,
            IngestionActionReason::AcceptedNewerRecord
        );
        assert_eq!(
            report40.phase_records[0].reason,
            IngestionActionReason::AcceptedNewerRecord
        );
        assert!(report40.phase_records[0].reason.is_accepted());
        assert_eq!(
            report40.phase_records[0].resulting_status,
            ActiveProvenanceStatus::ActiveHasUsable
        );
        assert_eq!(store.code_bias(sat, has_sig(sat, 0)), Some(0.77));

        let has40_token = report40.phase_records[0].continuity_token.unwrap();
        assert_ne!(has40_token, rtcm_token);
        assert_eq!(has40_token.source(), SsrSource::GalileoHas);
        assert!(has40_token.generation() > rtcm_generation);

        // The stale RTCM acknowledgement is refused against the new HAS arc.
        let q_stale_rtcm = store.query_phase_bias(sat, has_sig(sat, 0), t_ref40, Some(rtcm_token));
        assert_eq!(
            q_stale_rtcm.status,
            SsrBiasStatus::PhaseDiscontinuityNeedsReset
        );
        assert!(matches!(
            q_stale_rtcm.discontinuity_details,
            Some(SsrDiscontinuityDetails::SolutionChanged { .. })
        ));
        // Acknowledging the new token restores a continuous arc.
        let q_new = store.query_phase_bias(sat, has_sig(sat, 0), t_ref40, Some(has40_token));
        assert_eq!(q_new.status, SsrBiasStatus::Available);
        assert_eq!(
            q_new.discontinuity_details,
            Some(SsrDiscontinuityDetails::Continuous)
        );
    }

    #[test]
    fn test_equal_epoch_unavailable_separates_equivalent_from_revised_records() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let t0_tow = 250_000.0;
        let reception = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow).unwrap();
        let t_ref = has_mt1_reference_j2000_s(reception, (t0_tow as u32 % 3600) as u16).unwrap();

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        let unavailable = |iod: u8, pdi: u8, validity_interval: u8| HasMt1Message {
            header: HasMt1Header {
                toh_s: (t0_tow as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: iod,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval,
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: None,
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: None,
                    discontinuity_indicator: pdi,
                }],
            }),
            padding_bits: Vec::new(),
        };

        let mut store = SsrCorrectionStore::new();
        let first = store
            .ingest_has_mt1_with_report(&unavailable(1, 2, 5), reception)
            .unwrap();
        assert_eq!(
            first.code_records[0].reason,
            IngestionActionReason::AcceptedFirstUnavailableWithMetadata
        );
        assert_eq!(
            first.phase_records[0].reason,
            IngestionActionReason::AcceptedFirstUnavailableWithMetadata
        );
        let established = store.clone();
        let established_token = first.phase_records[0].continuity_token.unwrap();

        // A byte-identical equal-epoch record is genuinely equivalent: nothing about
        // the active correction, its token or its generation moves.
        let repeat = store
            .ingest_has_mt1_with_report(&unavailable(1, 2, 5), reception)
            .unwrap();
        assert_eq!(
            repeat.code_records[0].reason,
            IngestionActionReason::RetainedEquivalentUnavailable
        );
        assert_eq!(
            repeat.phase_records[0].reason,
            IngestionActionReason::RetainedEquivalentUnavailable
        );
        assert!(repeat.phase_records[0].reason.is_retained());
        assert!(!repeat.phase_records[0].reason.is_accepted());
        assert!(!repeat.phase_records[0].reason.is_refused());
        assert_eq!(
            repeat.phase_records[0].continuity_token,
            Some(established_token)
        );
        assert_eq!(store, established, "an equivalent record changes nothing");

        // A revised IOD set id at the same epoch is not an equivalent record and is
        // not blanket rejected for being `<=`. It starts no new phase arc: a HAS arc
        // breaks only on a source switch or a PDI change (HAS SIS ICD 5.2.6.1, 7.4).
        let revised_iod = store
            .ingest_has_mt1_with_report(&unavailable(2, 2, 5), reception)
            .unwrap();
        assert_eq!(
            revised_iod.code_records[0].reason,
            IngestionActionReason::AcceptedUpdatedRecord
        );
        assert_eq!(
            revised_iod.phase_records[0].reason,
            IngestionActionReason::AcceptedUpdatedRecord
        );
        assert!(revised_iod.phase_records[0].reason.is_accepted());
        let iod_token = revised_iod.phase_records[0].continuity_token.unwrap();
        assert_eq!(iod_token.solution_id(), 2);
        assert_eq!(iod_token.generation(), established_token.generation());
        let across_iod =
            store.query_phase_bias(sat, has_sig(sat, 0), t_ref, Some(established_token));
        assert_eq!(
            across_iod.discontinuity_details,
            Some(SsrDiscontinuityDetails::Continuous),
            "the IOD set id is not part of a HAS token's identity"
        );
        assert_ne!(store, established);

        // A revised PDI at the same epoch and solution is also an update.
        let revised_pdi = store
            .ingest_has_mt1_with_report(&unavailable(2, 3, 5), reception)
            .unwrap();
        assert_eq!(
            revised_pdi.phase_records[0].reason,
            IngestionActionReason::AcceptedUpdatedRecord
        );
        assert_eq!(revised_pdi.phase_records[0].pdi, 3);
        let pdi_token = revised_pdi.phase_records[0].continuity_token.unwrap();
        assert_eq!(pdi_token.raw_indicator(), 3);
        assert_eq!(pdi_token.generation(), iod_token.generation() + 1);

        // A revised validity interval at the same epoch, solution and PDI changes the
        // stored lifetime, so it is an update rather than an equivalent retention, but
        // it starts no new arc.
        let before_lifetime = store.clone();
        let revised_lifetime = store
            .ingest_has_mt1_with_report(&unavailable(2, 3, 6), reception)
            .unwrap();
        assert_eq!(
            revised_lifetime.code_records[0].reason,
            IngestionActionReason::AcceptedUpdatedRecord
        );
        assert_eq!(
            revised_lifetime.phase_records[0].reason,
            IngestionActionReason::AcceptedUpdatedRecord
        );
        assert_eq!(
            revised_lifetime.phase_records[0]
                .continuity_token
                .unwrap()
                .generation(),
            pdi_token.generation(),
            "a lifetime revision is not an arc transition"
        );
        assert_ne!(store, before_lifetime);
        assert_eq!(
            store
                .query_phase_bias(sat, has_sig(sat, 0), t_ref, None)
                .lifetime
                .unwrap(),
            SsrLifetime::GalileoHasValidityInterval(has_validity_interval_s(6).unwrap())
        );
        assert_eq!(
            store
                .query_code_bias(sat, has_sig(sat, 0), t_ref)
                .lifetime
                .unwrap(),
            SsrLifetime::GalileoHasValidityInterval(has_validity_interval_s(6).unwrap())
        );
    }

    #[test]
    fn test_token_identity_spans_reference_epoch_across_distinct_stores() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let other_sat = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let t0_tow = 300_000.0;
        let toh = |offset: u32| ((t0_tow as u32 % 3600) + offset) as u16;

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn, other_sat.prn],
                signals: vec![0, 9],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        // Identical mask id, IOD set id, satellite, signal and PDI. Only the reference
        // epoch differs between the two stores, and both arcs are at generation 0.
        let msg = |offset: u32| HasMt1Message {
            header: HasMt1Header {
                toh_s: toh(offset),
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: true,
                reserved: 0,
                mask_id: 4,
                iod_set_id: 6,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasPhaseBias {
                        sat,
                        signal_id: 0,
                        bias_cycles: Some(0.10),
                        discontinuity_indicator: 2,
                    },
                    HasPhaseBias {
                        sat,
                        signal_id: 9,
                        bias_cycles: Some(0.20),
                        discontinuity_indicator: 2,
                    },
                    HasPhaseBias {
                        sat: other_sat,
                        signal_id: 0,
                        bias_cycles: Some(0.30),
                        discontinuity_indicator: 2,
                    },
                ],
            }),
            padding_bits: Vec::new(),
        };

        let reception_early = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 10.0).unwrap();
        let reception_late = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 40.0).unwrap();
        let t_early = has_mt1_reference_j2000_s(reception_early, toh(10)).unwrap();
        let t_late = has_mt1_reference_j2000_s(reception_late, toh(40)).unwrap();
        assert!(t_early < t_late);

        let mut early_store = SsrCorrectionStore::new();
        early_store
            .ingest_has_mt1_with_report(&msg(10), reception_early)
            .unwrap();
        let mut late_store = SsrCorrectionStore::new();
        late_store
            .ingest_has_mt1_with_report(&msg(40), reception_late)
            .unwrap();

        let early_token = early_store
            .query_phase_bias(sat, has_sig(sat, 0), t_early, None)
            .continuity_token
            .unwrap();
        let late_token = late_store
            .query_phase_bias(sat, has_sig(sat, 0), t_late, None)
            .continuity_token
            .unwrap();

        // Every part of the identity except the continuity reference epoch agrees.
        assert_eq!(early_token.generation(), 0);
        assert_eq!(late_token.generation(), 0);
        assert_eq!(early_token.satellite(), late_token.satellite());
        assert_eq!(early_token.signal(), late_token.signal());
        assert_eq!(early_token.source(), late_token.source());
        assert_eq!(early_token.provider_id(), late_token.provider_id());
        assert_eq!(early_token.solution_id(), late_token.solution_id());
        assert_eq!(early_token.raw_indicator(), late_token.raw_indicator());
        assert_ne!(
            early_token.continuity_ref_epoch_j2000_s().to_bits(),
            late_token.continuity_ref_epoch_j2000_s().to_bits()
        );
        assert_ne!(early_token, late_token);

        // The earlier token is stale against the later store.
        let stale = late_store.query_phase_bias(sat, has_sig(sat, 0), t_late, Some(early_token));
        assert_eq!(stale.status, SsrBiasStatus::PhaseDiscontinuityNeedsReset);
        assert_eq!(
            stale.discontinuity_details,
            Some(SsrDiscontinuityDetails::StaleToken)
        );

        // The later token is ahead of the earlier store.
        let future = early_store.query_phase_bias(sat, has_sig(sat, 0), t_early, Some(late_token));
        assert_eq!(future.status, SsrBiasStatus::PhaseDiscontinuityNeedsReset);
        assert_eq!(
            future.discontinuity_details,
            Some(SsrDiscontinuityDetails::FutureToken)
        );

        // Each store still accepts exactly its own current token.
        let ok_late = late_store.query_phase_bias(sat, has_sig(sat, 0), t_late, Some(late_token));
        assert_eq!(ok_late.status, SsrBiasStatus::Available);
        assert_eq!(
            ok_late.discontinuity_details,
            Some(SsrDiscontinuityDetails::Continuous)
        );
        let ok_early =
            early_store.query_phase_bias(sat, has_sig(sat, 0), t_early, Some(early_token));
        assert_eq!(ok_early.status, SsrBiasStatus::Available);
        assert_eq!(
            ok_early.discontinuity_details,
            Some(SsrDiscontinuityDetails::Continuous)
        );

        // Tokens for another signal or satellite are mismatched, not merely stale.
        let other_signal_token = late_store
            .query_phase_bias(sat, has_sig(sat, 9), t_late, None)
            .continuity_token
            .unwrap();
        let mismatch_signal =
            late_store.query_phase_bias(sat, has_sig(sat, 0), t_late, Some(other_signal_token));
        assert_eq!(
            mismatch_signal.discontinuity_details,
            Some(SsrDiscontinuityDetails::MismatchedToken)
        );
        let other_sat_token = late_store
            .query_phase_bias(other_sat, has_sig(other_sat, 0), t_late, None)
            .continuity_token
            .unwrap();
        let mismatch_sat =
            late_store.query_phase_bias(sat, has_sig(sat, 0), t_late, Some(other_sat_token));
        assert_eq!(
            mismatch_sat.discontinuity_details,
            Some(SsrDiscontinuityDetails::MismatchedToken)
        );

        // An unavailable record keeps its own primary status while still reporting
        // that the supplied acknowledgement does not name the current arc.
        let unavailable_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: toh(100),
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: true,
                reserved: 0,
                mask_id: 4,
                iod_set_id: 6,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                // The same PDI as the established arc, so the unavailable record does
                // not itself start a new generation and the reference epoch is the
                // only part of the token identity that separates the two stores.
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: None,
                    discontinuity_indicator: 2,
                }],
            }),
            padding_bits: Vec::new(),
        };
        let reception_unavail = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 100.0).unwrap();
        let t_unavail = has_mt1_reference_j2000_s(reception_unavail, toh(100)).unwrap();
        late_store
            .ingest_has_mt1_with_report(&unavailable_msg, reception_unavail)
            .unwrap();

        let q_unavail =
            late_store.query_phase_bias(sat, has_sig(sat, 0), t_unavail, Some(early_token));
        assert_eq!(q_unavail.status, SsrBiasStatus::Unavailable);
        assert_eq!(
            q_unavail.details,
            SsrBiasResolutionDetails::TransmittedUnavailable
        );
        assert_eq!(q_unavail.bias_cycles, None);
        assert_eq!(
            q_unavail.discontinuity_indicator,
            Some(PhaseDiscontinuityIndicator::GalileoHasPdi(2))
        );
        assert_ne!(
            q_unavail.discontinuity_details,
            Some(SsrDiscontinuityDetails::Continuous),
            "an invalid acknowledgement is never reported as continuous"
        );
        assert_eq!(
            q_unavail.discontinuity_details,
            Some(SsrDiscontinuityDetails::StaleToken)
        );

        // An out-of-validity query keeps its time status and still evaluates the token.
        let q_expired =
            late_store.query_phase_bias(sat, has_sig(sat, 0), t_unavail - 1.0, Some(early_token));
        assert_eq!(q_expired.status, SsrBiasStatus::NotYetValid);
        assert!(matches!(
            q_expired.details,
            SsrBiasResolutionDetails::EpochBeforeReference { .. }
        ));
        assert_eq!(
            q_expired.discontinuity_details,
            Some(SsrDiscontinuityDetails::StaleToken)
        );
    }

    #[test]
    fn test_generation_overflow_is_transactional_and_spares_untransitioned_records() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let t0_tow = 400_000.0;
        let toh = |offset: u32| ((t0_tow as u32 % 3600) + offset) as u16;
        let reception_t10 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 10.0).unwrap();
        let reception_t20 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 20.0).unwrap();
        let reception_t5 = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow + 5.0).unwrap();

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0, 9],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        let msg = |offset: u32, iod: u8, pdi_sig0: u8, pdi_sig9: u8| HasMt1Message {
            header: HasMt1Header {
                toh_s: toh(offset),
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: iod,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasCodeBias {
                        sat,
                        signal_id: 0,
                        bias_m: Some(0.10 + f64::from(offset)),
                    },
                    HasCodeBias {
                        sat,
                        signal_id: 9,
                        bias_m: Some(0.20 + f64::from(offset)),
                    },
                ],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasPhaseBias {
                        sat,
                        signal_id: 0,
                        bias_cycles: Some(0.10),
                        discontinuity_indicator: pdi_sig0,
                    },
                    HasPhaseBias {
                        sat,
                        signal_id: 9,
                        bias_cycles: Some(0.20),
                        discontinuity_indicator: pdi_sig9,
                    },
                ],
            }),
            padding_bits: Vec::new(),
        };

        let mut store = SsrCorrectionStore::new();
        store
            .ingest_has_mt1_with_report(&msg(10, 1, 0, 0), reception_t10)
            .unwrap();

        // Private near-maximum setup: driving the arc to u32::MAX through the public
        // ingest path would need billions of transitions. Only the second record in
        // the message is placed at the limit, so the first record is the one that
        // would already have been committed by a non-transactional implementation.
        {
            let sig_entry = store
                .corrections
                .get_mut(&sat)
                .unwrap()
                .phase_bias
                .signals
                .get_mut(&has_sig(sat, 9))
                .unwrap();
            sig_entry.arc_generation = u32::MAX;
            // The active record's token has to carry the same generation, otherwise the
            // injected state is one no sequence of ingests could have produced.
            sig_entry
                .active
                .as_mut()
                .expect("signal 9 has an active record")
                .token
                .generation = u32::MAX;
        }
        let before = store.clone();

        // A message whose records both transition (PDI 0 -> 1) overflows on signal 9.
        let err = store
            .ingest_has_mt1_with_report(&msg(20, 2, 1, 1), reception_t20)
            .unwrap_err();
        assert!(
            matches!(&err, Error::InvalidInput(m) if m.contains("phase continuity generation overflow")),
            "unexpected error: {err:?}"
        );
        assert_eq!(
            store, before,
            "the overflowing message left no code, phase or watermark state committed"
        );

        // A record that needs no transition is still legal at the limit: same epoch,
        // same solution, same PDI.
        let report_same = store
            .ingest_has_mt1_with_report(&msg(10, 1, 0, 0), reception_t10)
            .unwrap();
        assert_eq!(
            report_same.phase_records[1].reason,
            IngestionActionReason::AcceptedUpdatedRecord
        );
        assert_eq!(
            report_same.phase_records[1]
                .continuity_token
                .unwrap()
                .generation(),
            u32::MAX
        );
        assert_eq!(store, before, "an unchanged update mutates nothing");

        // An older record is refused rather than overflowing.
        let report_old = store
            .ingest_has_mt1_with_report(&msg(5, 9, 3, 3), reception_t5)
            .unwrap();
        assert!(report_old
            .phase_records
            .iter()
            .all(|r| r.reason == IngestionActionReason::RefusedOlderThanWatermark));
        assert_eq!(store, before, "a refused record mutates nothing");

        // RTCM ingestion is transactional across records for the same reason: the
        // first signal would transition, the second overflows. RTCM SSR GPS index 10 is
        // L2 P, the signal of HAS index 9 whose arc is at the limit; RTCM index 9 is
        // L2C(M+L), a signal of its own.
        let mut rtcm_hdr = header(SsrKind::PhaseBias);
        rtcm_hdr.update_interval = 5;
        rtcm_hdr.epoch_time_s = (t0_tow + 15.0) as u32;
        let rtcm_phase = SsrMessage {
            message_number: 1265,
            system: GnssSystem::Gps,
            kind: SsrKind::PhaseBias,
            header: rtcm_hdr,
            orbit: Vec::new(),
            clock: Vec::new(),
            code_bias: Vec::new(),
            phase_bias: vec![SsrPhaseBiasRecord {
                satellite_id: sat.prn,
                yaw_angle: 0,
                yaw_rate: 0,
                biases: vec![
                    SsrPhaseBiasSignal {
                        signal_id: 0,
                        integer_indicator: 0,
                        wide_lane_integer_indicator: 0,
                        discontinuity_counter: 1,
                        bias: 1000,
                    },
                    SsrPhaseBiasSignal {
                        signal_id: 10,
                        integer_indicator: 0,
                        wide_lane_integer_indicator: 0,
                        discontinuity_counter: 1,
                        bias: 2000,
                    },
                ],
            }],
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };
        let rtcm_err = store.ingest_ssr(&rtcm_phase, reception_t20).unwrap_err();
        assert!(
            matches!(&rtcm_err, Error::InvalidInput(m) if m.contains("phase continuity generation overflow")),
            "unexpected error: {rtcm_err:?}"
        );
        assert_eq!(
            store, before,
            "the overflowing RTCM message committed no earlier record"
        );

        // Duplicate edited signal records are rejected deterministically before any
        // mutation, rather than panicking or being silently deduplicated.
        let mut duplicate = msg(20, 3, 0, 0);
        if let Some(pb) = duplicate.phase_bias.as_mut() {
            pb.records.push(HasPhaseBias {
                sat,
                signal_id: 0,
                bias_cycles: Some(0.55),
                discontinuity_indicator: 0,
            });
        }
        let dup_err = store
            .ingest_has_mt1_with_report(&duplicate, reception_t20)
            .unwrap_err();
        assert!(
            matches!(&dup_err, Error::InvalidInput(m) if m.contains("duplicate HAS phase bias record")),
            "unexpected error: {dup_err:?}"
        );
        assert_eq!(store, before, "a rejected duplicate mutates nothing");
    }

    #[test]
    fn test_has_forward_validity_and_receiver_cap_separate() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let t0_tow = 100_000.0;
        let reception = GnssWeekTow::new(TimeScale::Gst, 1042, t0_tow).unwrap();
        let t_ref = has_mt1_reference_j2000_s(reception, (t0_tow as u32 % 3600) as u16).unwrap();

        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![sat.prn],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        // Case A: Short HAS VI (index 1 = 10 s), large receiver cap (100 s)
        let short_vi_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (t0_tow as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 1, // 10 s per Table 23
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: Some(1.0),
                }],
            }),
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let mut store_a = SsrCorrectionStore::new().with_staleness(StalenessPolicy::seconds(100.0));
        store_a
            .ingest_has_mt1_with_report(&short_vi_msg, reception)
            .unwrap();

        // 5s into interval: valid
        assert_eq!(
            store_a
                .query_code_bias(sat, has_sig(sat, 0), t_ref + 5.0)
                .status,
            SsrBiasStatus::Available
        );
        // 15s into interval: expired because HAS VI is 10s, even though receiver cap is 100s!
        assert_eq!(
            store_a
                .query_code_bias(sat, has_sig(sat, 0), t_ref + 15.0)
                .status,
            SsrBiasStatus::Expired
        );

        // Case B: Long HAS VI (index 10 = 300 s), small receiver cap (20 s)
        let long_vi_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: (t0_tow as u32 % 3600) as u16,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 2,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 10, // 300 s per Table 23
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: Some(2.0),
                }],
            }),
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let mut store_b = SsrCorrectionStore::new().with_staleness(StalenessPolicy::seconds(20.0));
        store_b
            .ingest_has_mt1_with_report(&long_vi_msg, reception)
            .unwrap();

        // 10s into interval: valid
        assert_eq!(
            store_b
                .query_code_bias(sat, has_sig(sat, 0), t_ref + 10.0)
                .status,
            SsrBiasStatus::Available
        );
        // 30s into interval: expired because receiver cap (20s) clips the 300s wire interval!
        assert_eq!(
            store_b
                .query_code_bias(sat, has_sig(sat, 0), t_ref + 30.0)
                .status,
            SsrBiasStatus::Expired
        );

        // Case C: RTCM centered policy is unchanged (allows epoch before ref within update interval)
        let mut rtcm_hdr = header(SsrKind::CodeBias);
        rtcm_hdr.epoch_time_s = t0_tow as u32;
        rtcm_hdr.update_interval = 4; // update interval 10 s
        let rtcm_msg = SsrMessage {
            message_number: 1059,
            system: GnssSystem::Gps,
            kind: SsrKind::CodeBias,
            header: rtcm_hdr,
            orbit: Vec::new(),
            clock: Vec::new(),
            code_bias: vec![crate::rtcm::SsrCodeBiasRecord {
                satellite_id: 1,
                biases: vec![(0, 50)],
            }],
            phase_bias: Vec::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };
        let mut store_c = SsrCorrectionStore::new().with_staleness(StalenessPolicy::seconds(60.0));
        store_c.ingest_ssr(&rtcm_msg, reception).unwrap();

        // RTCM centered window |t - ref| <= effective_limit allows t_ref - 5.0
        let q_rtcm_before = store_c.query_code_bias(sat, has_sig(sat, 0), t_ref - 5.0);
        assert_eq!(q_rtcm_before.status, SsrBiasStatus::Available);

        // But HAS at t_ref - 5.0 is NotYetValid due to forward-only validity
        assert_eq!(
            store_a
                .query_code_bias(sat, has_sig(sat, 0), t_ref - 5.0)
                .status,
            SsrBiasStatus::NotYetValid
        );
    }

    fn g30_g31_broadcast() -> BroadcastEphemeris {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        ))
        .expect("read NAV fixture");
        BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture")
    }

    /// The same records with every GPS TGD and Galileo BGD of `sat` moved by `shift_s`.
    fn with_group_delay_shift(
        broadcast: &BroadcastEphemeris,
        sat: GnssSatelliteId,
        shift_s: f64,
    ) -> BroadcastEphemeris {
        let records = broadcast
            .records()
            .iter()
            .map(|record| {
                let mut record = *record;
                if record.satellite_id == sat {
                    let delays = &mut record.group_delays;
                    for value in [
                        &mut delays.gps_tgd_s,
                        &mut delays.galileo_bgd_e5a_e1_s,
                        &mut delays.galileo_bgd_e5b_e1_s,
                    ]
                    .into_iter()
                    .flatten()
                    {
                        *value += shift_s;
                    }
                }
                record
            })
            .collect();
        BroadcastEphemeris::new(records).expect("records with shifted group delays")
    }

    /// A HAS MT1 message with a one-satellite mask stating `nav_message`, an orbit
    /// correction against `iode` and a clock correction of `clock_m`, ingested at
    /// `reception` (GST) with the TOH of that instant.
    fn has_orbit_clock_store(
        sat: GnssSatelliteId,
        iode: u32,
        nav_message: u8,
        clock_m: f64,
        reception: GnssWeekTow,
    ) -> SsrCorrectionStore {
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: (reception.tow_s as u32 % 3600) as u16,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 2,
                iod_set_id: 5,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: sat.system,
                    satellites: vec![sat.prn],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat,
                    nav_message,
                    iode,
                    radial_m: Some(1.25),
                    along_m: Some(-2.0),
                    cross_m: Some(3.0),
                }],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: sat.system,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    nav_message,
                    correction_m: Some(clock_m),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        let decoded = HasMt1Message::decode(&message.encode().expect("encode HAS MT1"))
            .expect("decode HAS MT1");
        let mut store = SsrCorrectionStore::new();
        store
            .ingest_has_mt1(&decoded, reception)
            .expect("ingest HAS MT1");
        store
    }

    /// An RTCM 1060 message for G30 against `iode` with clock C0 `c0` (0.1 mm units),
    /// update interval index 0, ingested at the fixture epoch.
    fn rtcm_g30_store(iode: u32, c0: i32) -> SsrCorrectionStore {
        let message = SsrMessage {
            message_number: 1060,
            system: GnssSystem::Gps,
            kind: SsrKind::CombinedOrbitClock,
            header: SsrHeader {
                epoch_time_s: REAL_SSR_EPOCH_TOW_S as u32,
                update_interval: 0,
                multiple_message: false,
                iod_ssr: 3,
                provider_id: 9,
                solution_id: 1,
                satellite_reference_datum: Some(false),
                dispersive_bias_consistency: None,
                mw_consistency: None,
                satellite_count: 1,
            },
            orbit: vec![SsrOrbitRecord {
                satellite_id: 30,
                iode,
                delta_radial: -20_000,
                delta_along: 10_000,
                delta_cross: -3_000,
                dot_delta_radial: 0,
                dot_delta_along: 0,
                dot_delta_cross: 0,
            }],
            clock: vec![SsrClockRecord {
                satellite_id: 30,
                c0,
                c1: 0,
                c2: 0,
            }],
            code_bias: Vec::new(),
            phase_bias: Vec::<SsrPhaseBiasRecord>::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };
        let week = GnssWeekTow::new(TimeScale::Gpst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("valid SSR week");
        let mut store = SsrCorrectionStore::new();
        store.ingest_ssr(&message, week).expect("ingest RTCM SSR");
        store
    }

    /// RTCM SSR and IGS SSR define the clock correction as added to the broadcast clock,
    /// and RTKLIB `satpos_ssr` adds `dclk / CLIGHT`. A positive C0 makes the corrected
    /// satellite clock later by exactly C0 / c.
    #[test]
    fn rtcm_clock_correction_adds_to_the_satellite_clock() {
        let broadcast = g30_g31_broadcast();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let iode = broadcast
            .select_record_at(sat, t)
            .expect("broadcast record")
            .issue_of_data
            .expect("broadcast issue")
            .issue;
        let zero = rtcm_g30_store(iode, 0);
        let positive = rtcm_g30_store(iode, 5_000);
        let negative = rtcm_g30_store(iode, -5_000);
        let clock = |store: &SsrCorrectionStore| {
            SsrCorrectedEphemeris::new(&broadcast, store)
                .corrected_state(sat, t)
                .expect("RTCM corrected state")
                .1
        };
        let c0_m = positive.clock(sat).expect("stored clock").c0_m;
        assert!(c0_m > 0.0);
        assert!(clock(&positive) > clock(&zero));
        assert!(clock(&negative) < clock(&zero));
        assert!(((clock(&positive) - clock(&zero)) - c0_m / C_M_S).abs() < 1.0e-18);
        assert!(((clock(&zero) - clock(&negative)) - c0_m / C_M_S).abs() < 1.0e-18);
    }

    /// Metamorphic check of the SSR clock against the broadcast group delay, for RTCM SSR
    /// and Galileo HAS on a GPS LNAV record. RTKLIB `satpos_ssr` builds the clock from the
    /// polynomial and `2 r·v / c²` only (its `satpos` notes: the clock "does not include
    /// code bias correction (tgd or bgd)"), and HAS SIS ICD 7.4 has the HAS code biases
    /// replace the TGD. Moving the TGD therefore leaves the corrected clock bit for bit,
    /// with the HAS correction zero and non-zero. The broadcast single-frequency group
    /// delay, which moves by the shift, is the control.
    #[test]
    fn ssr_corrected_gps_clock_is_independent_of_lnav_tgd() {
        let broadcast = g30_g31_broadcast();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let shift_s = 2.5e-8;
        let shifted = with_group_delay_shift(&broadcast, sat, shift_s);
        let record = broadcast
            .select_record_at(sat, t)
            .expect("broadcast record");
        let shifted_record = shifted.select_record_at(sat, t).expect("shifted record");
        assert_eq!(
            record.broadcast_clock_group_delay_s().to_bits(),
            4.190_951_585_770e-9_f64.to_bits()
        );
        assert!(
            (shifted_record.broadcast_clock_group_delay_s()
                - record.broadcast_clock_group_delay_s()
                - shift_s)
                .abs()
                < 1.0e-18
        );
        // Control: the shift reaches the single-frequency group delay, and the broadcast
        // state clock, which is RTKLIB's `satposs` clock without it, stays as it was.
        let plain = broadcast
            .position_clock_at_j2000_s(sat, t)
            .expect("state")
            .1;
        let plain_shifted = shifted.position_clock_at_j2000_s(sat, t).expect("state").1;
        assert_eq!(plain.to_bits(), plain_shifted.to_bits());
        let group_delay = broadcast
            .single_frequency_group_delay_s(sat, t)
            .expect("group delay");
        let group_delay_shifted = shifted
            .single_frequency_group_delay_s(sat, t)
            .expect("group delay");
        assert!(((group_delay_shifted - group_delay) - shift_s).abs() < 1.0e-18);

        let reception = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception");
        let iode = record.issue_of_data.expect("broadcast issue").issue;
        let stores = [
            ("RTCM SSR", real_gps_ssr_store()),
            ("RTCM SSR zero C0", rtcm_g30_store(iode, 0)),
            (
                "HAS zero DCC",
                has_orbit_clock_store(sat, iode, 0, 0.0, reception),
            ),
            (
                "HAS DCC",
                has_orbit_clock_store(sat, iode, 0, -0.75, reception),
            ),
        ];
        for (label, store) in &stores {
            let clock = SsrCorrectedEphemeris::new(&broadcast, store)
                .corrected_state(sat, t)
                .unwrap_or_else(|| panic!("{label} corrected state"))
                .1;
            let clock_shifted = SsrCorrectedEphemeris::new(&shifted, store)
                .corrected_state(sat, t)
                .unwrap_or_else(|| panic!("{label} corrected state, shifted TGD"))
                .1;
            assert_eq!(clock.to_bits(), clock_shifted.to_bits(), "{label}");
            let dclock_m = store.clock(sat).expect("stored clock").c0_m;
            assert_eq!(
                clock.to_bits(),
                satpos_ssr_clock_s(&broadcast, sat, REAL_SSR_EPOCH_TOW_S, dclock_m).to_bits(),
                "{label}"
            );
            // The single-frequency group delay, RTCM SSR and HAS alike, is the TGD of the
            // record the orbit correction's IODE selects, as RTKLIB `pntpos` applies it,
            // returned with the state from one evaluation.
            let source = SsrCorrectedEphemeris::new(&shifted, store);
            let (_, combined_clock, combined_delay) = source
                .corrected_state_with_group_delay(sat, t)
                .expect("state with group delay");
            assert_eq!(combined_clock.to_bits(), clock_shifted.to_bits(), "{label}");
            assert_eq!(
                combined_delay.map(f64::to_bits),
                Some(shifted_record.broadcast_clock_group_delay_s().to_bits()),
                "{label}"
            );
            assert_eq!(
                source
                    .single_frequency_group_delay_s(sat, t)
                    .map(f64::to_bits),
                combined_delay.map(f64::to_bits),
                "{label}"
            );
        }
    }

    /// The metamorphic check for Galileo HAS on an I/NAV record: moving both BGDs leaves
    /// the HAS-corrected clock bit for bit, and it is HAS SIS ICD Eq. 23 and 24 as RTKLIB
    /// `satpos_ssr` forms them, with no BGD. The I/NAV single-frequency group delay, BGD
    /// E5b/E1, moves by the shift.
    #[test]
    fn has_corrected_galileo_clock_is_independent_of_inav_bgd() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/nav/ESBC00DNK_R_20201770000_01D_MN.rnx"
        ))
        .expect("read NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse NAV fixture");
        let sat = GnssSatelliteId::new(GnssSystem::Galileo, 1).unwrap();
        // 2020-06-25 12:01:00 GST: one minute after the I/NAV record's toe (388800 s).
        let week = 2111_u32;
        let tow_s = 388_860.0;
        let t = f64::from(week) * SECONDS_PER_WEEK + tow_s - GPS_EPOCH_TO_J2000_S;
        let record = broadcast.select_record_at(sat, t).expect("I/NAV record");
        assert_eq!(record.message, NavMessage::GalileoInav);
        assert_eq!(record.clock.toc_sow.to_bits(), 388_800.0_f64.to_bits());
        let shift_s = -3.0e-9;
        let shifted = with_group_delay_shift(&broadcast, sat, shift_s);
        // Control: the shift reaches the single-frequency group delay, and the broadcast
        // state clock, which is RTKLIB's `satposs` clock without it, stays as it was.
        let plain = broadcast
            .position_clock_at_j2000_s(sat, t)
            .expect("state")
            .1;
        let plain_shifted = shifted.position_clock_at_j2000_s(sat, t).expect("state").1;
        assert_eq!(plain.to_bits(), plain_shifted.to_bits());
        let group_delay = broadcast
            .single_frequency_group_delay_s(sat, t)
            .expect("group delay");
        let group_delay_shifted = shifted
            .single_frequency_group_delay_s(sat, t)
            .expect("group delay");
        assert!(((group_delay_shifted - group_delay) - shift_s).abs() < 1.0e-18);

        let reception = GnssWeekTow::new(TimeScale::Gst, week, tow_s).expect("GST reception");
        for clock_m in [0.0, 0.5] {
            let store = has_orbit_clock_store(
                sat,
                record.issue_of_data.expect("broadcast issue").issue,
                0,
                clock_m,
                reception,
            );
            let clock = SsrCorrectedEphemeris::new(&broadcast, &store)
                .corrected_state(sat, t)
                .expect("HAS corrected state")
                .1;
            let clock_shifted = SsrCorrectedEphemeris::new(&shifted, &store)
                .corrected_state(sat, t)
                .expect("HAS corrected state, shifted BGD")
                .1;
            assert_eq!(clock.to_bits(), clock_shifted.to_bits(), "DCC {clock_m} m");

            let tk = tow_s - record.clock.toc_sow;
            let mut expected =
                record.clock.af0 + record.clock.af1 * tk + record.clock.af2 * tk * tk;
            let r = broadcast
                .position_clock_at_j2000_s(sat, t)
                .expect("state")
                .0;
            let v = finite_difference_broadcast_velocity(&broadcast, sat, t);
            expected -= 2.0 * dot3(r, v) / C_M_S / C_M_S;
            expected += store.clock(sat).expect("stored HAS clock").c0_m / C_M_S;
            assert_eq!(clock.to_bits(), expected.to_bits(), "DCC {clock_m} m");
        }
    }

    /// HAS SIS ICD Table 21 reserves navigation-message indices 1..=7. A correction
    /// for one is decoded, kept with its index and stored, and the corrected source does
    /// not apply it to the LNAV record: it reports the reserved index and, where
    /// allowed, falls back to the broadcast state as for a missing correction.
    #[test]
    fn has_reserved_navigation_message_is_kept_and_not_applied() {
        let broadcast = g30_g31_broadcast();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let t = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let iode = broadcast
            .select_record_at(sat, t)
            .expect("broadcast record")
            .issue_of_data
            .expect("broadcast issue")
            .issue;
        let reception = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception");

        let usable = has_orbit_clock_store(sat, iode, 0, -0.75, reception);
        assert_eq!(
            usable.orbit(sat).expect("orbit").nav_message,
            SsrNavigationMessage::Has(0)
        );
        assert!(SsrCorrectedEphemeris::new(&broadcast, &usable)
            .applied_orbit_clock_status(sat, t)
            .is_ok());

        for index in 1..=7_u8 {
            let store = has_orbit_clock_store(sat, iode, index, -0.75, reception);
            let orbit = store.orbit(sat).expect("reserved-index orbit is stored");
            let clock = store.clock(sat).expect("reserved-index clock is stored");
            assert_eq!(orbit.nav_message, SsrNavigationMessage::Has(index));
            assert_eq!(clock.nav_message, SsrNavigationMessage::Has(index));
            assert_eq!(orbit.nav_message.reserved_has_index(), Some(index));
            assert_eq!(orbit.iode, iode);
            assert!((clock.c0_m + 0.75).abs() < 1.0e-12);

            let declining = SsrCorrectedEphemeris::new(&broadcast, &store);
            assert_eq!(
                declining.applied_orbit_clock_status(sat, t),
                Err(SsrStateUnavailable::ReservedNavigationMessage { index })
            );
            assert_eq!(declining.applied_orbit_clock_solution(sat, t), None);
            assert_eq!(declining.corrected_state(sat, t), None);
            assert_eq!(declining.corrected_velocity(sat, t), None);

            let fallback =
                SsrCorrectedEphemeris::new(&broadcast, &store).with_fallback(SsrFallbackPolicy {
                    on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                    regional: RegionalPolicy::DeclineRegional,
                });
            let (position, clock_s) = fallback.corrected_state(sat, t).expect("fallback");
            let (broadcast_position, broadcast_clock) = broadcast
                .position_clock_at_j2000_s(sat, t)
                .expect("broadcast state");
            assert_eq!(
                position.map(f64::to_bits),
                broadcast_position.map(f64::to_bits)
            );
            assert_eq!(clock_s.to_bits(), broadcast_clock.to_bits());
        }
    }

    /// The navigation-message index is a mask field. A record holding an index its
    /// mask does not state, or one wider than the 3-bit field, cannot be transmitted:
    /// the encoder and the HAS ingest refuse it by name and leave the store unchanged.
    #[test]
    fn has_record_navigation_message_must_be_the_mask_index() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let reception = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception");
        let store = has_orbit_clock_store(sat, 90, 0, -0.75, reception);
        let good = HasMt1Message::decode(
            &HasMt1Message {
                header: HasMt1Header {
                    toh_s: (REAL_SSR_EPOCH_TOW_S as u32 % 3600) as u16,
                    mask: true,
                    orbit: true,
                    clock_full_set: false,
                    clock_subset: false,
                    code_bias: false,
                    phase_bias: false,
                    reserved: 0,
                    mask_id: 2,
                    iod_set_id: 6,
                },
                mask: Some(HasMaskBlock {
                    systems: vec![HasGnssMask {
                        system: GnssSystem::Gps,
                        satellites: vec![sat.prn],
                        signals: vec![0],
                        cell_mask: None,
                        nav_message: 0,
                    }],
                    reserved: 0,
                }),
                orbit: Some(HasOrbitBlock {
                    validity_interval: 5,
                    records: vec![HasOrbitCorrection {
                        sat,
                        nav_message: 0,
                        iode: 90,
                        radial_m: Some(0.5),
                        along_m: Some(0.5),
                        cross_m: Some(0.5),
                    }],
                }),
                clock_full_set: None,
                clock_subset: None,
                code_bias: None,
                phase_bias: None,
                padding_bits: Vec::new(),
            }
            .encode()
            .expect("encode HAS MT1"),
        )
        .expect("decode HAS MT1");

        let mut mismatched = good.clone();
        mismatched.orbit.as_mut().expect("orbit block").records[0].nav_message = 2;
        let error = mismatched
            .encode()
            .expect_err("record index differs from mask");
        assert!(
            error.to_string().contains("navigation message index 2"),
            "{error}"
        );
        let mut refused = store.clone();
        let error = refused
            .ingest_has_mt1(&mismatched, reception)
            .expect_err("record index differs from mask");
        assert!(error.to_string().contains("mask states 0"), "{error}");
        assert_eq!(refused, store);

        let mut too_wide = good;
        too_wide.mask = None;
        too_wide.header.mask = false;
        too_wide.orbit.as_mut().expect("orbit block").records[0].nav_message = 8;
        let mut refused = store.clone();
        let error = refused
            .ingest_has_mt1(&too_wide, reception)
            .expect_err("index wider than NM");
        assert!(error.to_string().contains("3-bit NM field"), "{error}");
        assert_eq!(refused, store);
    }

    /// The seconds of week an SSR state is evaluated at hold every bit of the J2000
    /// epoch. One ulp past a whole second (2^-23 s near 8.3e8 s) is lost when
    /// `GPS_EPOCH_TO_J2000_S` is added first: the sum, near 1.47e9 s, has 2^-22 s
    /// spacing and rounds the half-way case to the whole second.
    #[test]
    fn ssr_seconds_of_week_keep_every_bit_of_the_epoch() {
        let whole = ssr_j2000(REAL_SSR_EPOCH_TOW_S);
        let t = f64::from_bits(whole.to_bits() + 1);
        let ulp = t - whole;
        assert_eq!(ulp.to_bits(), 2.0_f64.powi(-23).to_bits());
        let gps = GnssSatelliteId::new(GnssSystem::Gps, 30).unwrap();
        let (sow, is_geo) = ssr_seconds_of_week(gps, t).expect("GPS seconds of week");
        assert!(!is_geo);
        assert_eq!(sow.to_bits(), (REAL_SSR_EPOCH_TOW_S + ulp).to_bits());
        let rounded = (t + GPS_EPOCH_TO_J2000_S).rem_euclid(SECONDS_PER_WEEK);
        assert_eq!(rounded.to_bits(), REAL_SSR_EPOCH_TOW_S.to_bits());

        let qzss = GnssSatelliteId::new(GnssSystem::Qzss, 2).unwrap();
        assert_eq!(ssr_seconds_of_week(qzss, t), Some((sow, false)));
        let beidou = GnssSatelliteId::new(GnssSystem::BeiDou, 30).unwrap();
        let (bdt_sow, _) = ssr_seconds_of_week(beidou, t).expect("BDT seconds of week");
        assert_eq!(
            bdt_sow.to_bits(),
            (REAL_SSR_EPOCH_TOW_S - 14.0 + ulp).to_bits()
        );
        let beidou_geo = GnssSatelliteId::new(GnssSystem::BeiDou, 3).unwrap();
        assert_eq!(
            ssr_seconds_of_week(beidou_geo, t).map(|(_, geo)| geo),
            Some(true)
        );
        let glonass = GnssSatelliteId::new(GnssSystem::Glonass, 3).unwrap();
        assert_eq!(ssr_seconds_of_week(glonass, t), None);

        // Week wrap: 1 s before the J2000 epoch's week ends.
        let end_of_week = SECONDS_PER_WEEK - crate::rinex_nav::J2000_GPS_SECONDS_OF_WEEK - 1.0;
        assert_eq!(
            ssr_seconds_of_week(gps, end_of_week),
            Some((SECONDS_PER_WEEK - 1.0, false))
        );
        assert_eq!(
            ssr_seconds_of_week(gps, end_of_week + 1.0),
            Some((0.0, false))
        );
        assert_eq!(
            ssr_seconds_of_week(beidou, end_of_week + 1.0),
            Some((SECONDS_PER_WEEK - 14.0, false))
        );
    }

    /// RTKLIB `ephpos` moves its `gtime_t` 1 ms with `timeadd`, which adds the step to
    /// the fraction of a second, and `eph2pos` takes `tk` as the whole seconds plus that
    /// fraction. For `tk = 1.6245591041273448 s` that rounds to 1.625559104127345 s,
    /// where adding 1 ms to `tk` itself rounds to 1.6255591041273447 s.
    #[test]
    fn ephpos_step_rounds_as_rtklib_timeadd() {
        let tk = 1.624_559_104_127_344_8;
        assert_eq!(
            ephpos_stepped_tk(tk).to_bits(),
            1.625_559_104_127_345_f64.to_bits()
        );
        assert_ne!(
            ephpos_stepped_tk(tk).to_bits(),
            (tk + EPHPOS_STEP_S).to_bits()
        );
        // A whole-second `tk` rounds once either way.
        assert_eq!(
            ephpos_stepped_tk(-630.0).to_bits(),
            (-630.0 + EPHPOS_STEP_S).to_bits()
        );
        // A fraction that carries into the next second: -0.0005 s is -1 s plus
        // 0.9995 s, the step makes the fraction 1.0005 s, `timeadd` carries the whole
        // second, and 0.0005 s is left as 1.0005 - 1 rounded.
        assert_eq!(
            ephpos_stepped_tk(-0.0005).to_bits(),
            0.000_499_999_999_999_944_9_f64.to_bits()
        );
    }

    /// GLONASS SSR corrections apply as RTKLIB `satpos_ssr` applies them: to the record
    /// whose `tb` the correction's IODE names (`selgeph`), with that record's `geph2pos`
    /// position, its 1 ms forward-difference velocity for the radial, along-track and
    /// cross-track axes, and its clock `-TauN + GammaN·tk` with no relativistic term, plus
    /// the clock correction over c. A correction naming a `tb` 12 h away matches no record.
    #[test]
    fn glonass_ssr_applies_to_the_tb_record_as_satpos_ssr() {
        let nav_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/nav/ESBC00DNK_R_20201770000_01D_RN.rnx"
        ))
        .expect("read GLONASS NAV fixture");
        let broadcast = BroadcastEphemeris::from_nav(&nav_text).expect("parse GLONASS NAV");
        let rec = broadcast.glonass_records()[0];
        let sat = rec.satellite_id;
        let leap_s = 18.0; // GPS - UTC in 2020, as the fixture header states
        let toe_gpst = rec.toe_utc_j2000_s + leap_s;
        let t = toe_gpst + 60.0;
        let tk = t - toe_gpst;
        // tb: the 15-min index of the reference epoch in UTC + 3 h (RTKLIB `readrnx`).
        let toe_utc_tod = (rec.toe_utc_j2000_s + 43_200.0).rem_euclid(86_400.0);
        let tb = ((toe_utc_tod + 10_800.0).rem_euclid(86_400.0) / 900.0 + 0.5) as u32;

        let t_gps = t + GPS_EPOCH_TO_J2000_S;
        let week = (t_gps / SECONDS_PER_WEEK).floor();
        let receiver = GnssWeekTow::new(
            TimeScale::Gpst,
            week as u32,
            t_gps - week * SECONDS_PER_WEEK,
        )
        .expect("receiver week");
        let glonass_tod =
            ((t - leap_s + 43_200.0).rem_euclid(86_400.0) + 10_800.0).rem_euclid(86_400.0) as u32;
        let store_for = |iode: u32| {
            let message = SsrMessage {
                message_number: 1066,
                system: GnssSystem::Glonass,
                kind: SsrKind::CombinedOrbitClock,
                header: SsrHeader {
                    epoch_time_s: glonass_tod,
                    update_interval: 0,
                    multiple_message: false,
                    iod_ssr: 2,
                    provider_id: 7,
                    solution_id: 1,
                    satellite_reference_datum: Some(false),
                    dispersive_bias_consistency: None,
                    mw_consistency: None,
                    satellite_count: 1,
                },
                orbit: vec![SsrOrbitRecord {
                    satellite_id: sat.prn,
                    iode,
                    delta_radial: -20_000,
                    delta_along: 10_000,
                    delta_cross: -3_000,
                    dot_delta_radial: 0,
                    dot_delta_along: 0,
                    dot_delta_cross: 0,
                }],
                clock: vec![SsrClockRecord {
                    satellite_id: sat.prn,
                    c0: 5_000,
                    c1: 0,
                    c2: 0,
                }],
                code_bias: Vec::new(),
                phase_bias: Vec::<SsrPhaseBiasRecord>::new(),
                ura: Vec::new(),
                padding_bits: Vec::new(),
            };
            let mut store = SsrCorrectionStore::new();
            store
                .ingest_ssr(&message, receiver)
                .expect("ingest GLONASS SSR");
            store
        };

        let store = store_for(tb);
        let orbit = store.orbit(sat).expect("GLONASS orbit correction");
        assert_eq!(orbit.transmitted_epoch_j2000_s.to_bits(), t.to_bits());
        let source = SsrCorrectedEphemeris::new(&broadcast, &store);
        assert!(source.applied_orbit_clock_status(sat, t).is_ok());
        let (position, clock, group_delay) = source
            .corrected_state_with_group_delay(sat, t)
            .expect("GLONASS SSR state");
        assert_eq!(group_delay, None);

        let state0 = [
            rec.pos_m[0],
            rec.pos_m[1],
            rec.pos_m[2],
            rec.vel_m_s[0],
            rec.vel_m_s[1],
            rec.vel_m_s[2],
        ];
        let start = crate::glonass::propagate(state0, rec.acc_m_s2, tk).expect("propagate");
        let end = crate::glonass::propagate(state0, rec.acc_m_s2, ephpos_stepped_tk(tk))
            .expect("propagate 1 ms later");
        let r = [start[0], start[1], start[2]];
        let v = [
            (end[0] - start[0]) / EPHPOS_STEP_S,
            (end[1] - start[1]) / EPHPOS_STEP_S,
            (end[2] - start[2]) / EPHPOS_STEP_S,
        ];
        let (er, ea, ec) = velocity_aligned_basis(r, v).expect("RAC axes");
        let (radial, along, cross) = (orbit.radial_m, orbit.along_m, orbit.cross_m);
        // RTKLIB `satpos_ssr` adds the summed RAC correction to the position as one term.
        let expected_position = [
            r[0] + (radial * er[0] + along * ea[0] + cross * ec[0]),
            r[1] + (radial * er[1] + along * ea[1] + cross * ec[1]),
            r[2] + (radial * er[2] + along * ea[2] + cross * ec[2]),
        ];
        assert_eq!(
            position.map(f64::to_bits),
            expected_position.map(f64::to_bits)
        );
        let c0_m = store.clock(sat).expect("GLONASS clock correction").c0_m;
        let mut expected_clock = rec.clk_bias + rec.gamma_n * tk;
        expected_clock += c0_m / C_M_S;
        assert_eq!(clock.to_bits(), expected_clock.to_bits());

        let other = store_for((tb + 48) % 96);
        assert_eq!(
            SsrCorrectedEphemeris::new(&broadcast, &other).applied_orbit_clock_status(sat, t),
            Err(SsrStateUnavailable::NoMatchingBroadcastRecord {
                iode: (tb + 48) % 96
            })
        );
    }

    /// A BeiDou SSR orbit correction names its record by the IOD `mod(toe/720, 240)` of IGS
    /// SSR v1.00 (IDF012), in the low eight bits of the transmitted issue; real SSRA03IGS0
    /// 1261 frames stamped 223656 s carry 70 for every satellite, `mod(223200/720, 240)`
    /// for the hourly BDT `toe` 223200 s, and 0 in the upper ten bits. The record's AODE
    /// does not name it.
    #[test]
    fn beidou_ssr_matches_the_record_by_toe_iod() {
        let recs = crate::rinex_nav::parse_nav(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/nav/ESBC00DNK_R_20201770000_01D_MN.rnx"
            ))
            .expect("read NAV fixture"),
        )
        .expect("parse NAV fixture");
        let record = *recs
            .iter()
            .find(|r| {
                r.satellite_id.system == GnssSystem::BeiDou
                    && !is_beidou_geo(r.satellite_id)
                    && r.message == NavMessage::BeidouD1
                    && r.elements.toe_sow.fract() == 0.0
            })
            .expect("BeiDou D1 record");
        let broadcast = BroadcastEphemeris::new(vec![record]).expect("store");
        let sat = record.satellite_id;
        let iod = (record.elements.toe_sow as u32 / 720) % 240;
        let t = f64::from(record.toe.week) * SECONDS_PER_WEEK
            + record.toe.tow_s
            + crate::constants::BDS_EPOCH_MINUS_GPS_EPOCH_S
            + crate::constants::GPST_MINUS_BDT_S
            - GPS_EPOCH_TO_J2000_S
            + 30.0;
        let found = broadcast
            .select_by_beidou_ssr_iod_at(sat, iod, NavMessage::BeidouD1, t)
            .expect("record by toe IOD");
        assert_eq!(found.issue_of_data, record.issue_of_data);
        assert!(broadcast
            .select_by_beidou_ssr_iod_at(sat, (iod + 1) % 240, NavMessage::BeidouD1, t)
            .is_none());
        if record.issue_of_data.expect("broadcast issue").issue != iod {
            // The AODE is not the SSR IOD: selecting by it finds nothing.
            assert!(broadcast
                .select_by_beidou_ssr_iod_at(
                    sat,
                    record.issue_of_data.expect("broadcast issue").issue,
                    NavMessage::BeidouD1,
                    t
                )
                .is_none());
        }
    }

    /// Galileo HAS and RTCM SSR bias records that carry the same raw signal index for
    /// different physical signals never share an entry, whichever arrives first. GPS
    /// index 9 is L2 P in HAS SIS ICD Table 20 and L2C(M+L) in RTCM SSR; Galileo index 0
    /// is E1-B in HAS and E1-A in RTCM SSR. Each record is accepted as the first on its
    /// own signal, keeps its value, solution and raw index, and starts its own phase arc.
    /// A record of the same physical signal from the other source (RTCM GPS index 10,
    /// L2 P) still shares the HAS entry and is arbitrated by arrival.
    #[test]
    fn has_and_rtcm_records_of_different_signals_stay_apart_in_both_orders() {
        let gps = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let gal = GnssSatelliteId::new(GnssSystem::Galileo, 1).unwrap();
        let tow = 100_000.0;
        let toh = ((tow as u32 % 3600) + 20) as u16;
        let reception = GnssWeekTow::new(TimeScale::Gst, 1042, tow + 20.0).unwrap();
        let t = has_mt1_reference_j2000_s(reception, toh).unwrap() + 1.0;

        let has = HasMt1Message {
            header: HasMt1Header {
                toh_s: toh,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![
                    HasGnssMask {
                        system: GnssSystem::Gps,
                        satellites: vec![gps.prn],
                        signals: vec![9],
                        cell_mask: None,
                        nav_message: 0,
                    },
                    HasGnssMask {
                        system: GnssSystem::Galileo,
                        satellites: vec![gal.prn],
                        signals: vec![0],
                        cell_mask: None,
                        nav_message: 0,
                    },
                ],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasCodeBias {
                        sat: gps,
                        signal_id: 9,
                        bias_m: Some(0.50),
                    },
                    HasCodeBias {
                        sat: gal,
                        signal_id: 0,
                        bias_m: Some(0.60),
                    },
                ],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasPhaseBias {
                        sat: gps,
                        signal_id: 9,
                        bias_cycles: Some(0.20),
                        discontinuity_indicator: 0,
                    },
                    HasPhaseBias {
                        sat: gal,
                        signal_id: 0,
                        bias_cycles: Some(0.30),
                        discontinuity_indicator: 0,
                    },
                ],
            }),
            padding_bits: Vec::new(),
        };
        assert!(has.encode().is_ok());

        let rtcm = |message_number: u16, system: GnssSystem, kind: SsrKind, index: u8| {
            let mut header = header(kind);
            header.epoch_time_s = (tow + 20.0) as u32;
            let mut message = SsrMessage {
                message_number,
                system,
                kind,
                header,
                orbit: Vec::new(),
                clock: Vec::new(),
                code_bias: Vec::new(),
                phase_bias: Vec::new(),
                ura: Vec::new(),
                padding_bits: Vec::new(),
            };
            match kind {
                SsrKind::CodeBias => {
                    message.code_bias = vec![crate::rtcm::SsrCodeBiasRecord {
                        satellite_id: 1,
                        biases: vec![(index, 70)],
                    }];
                }
                _ => {
                    message.phase_bias = vec![SsrPhaseBiasRecord {
                        satellite_id: 1,
                        yaw_angle: 0,
                        yaw_rate: 0,
                        biases: vec![SsrPhaseBiasSignal {
                            signal_id: index,
                            integer_indicator: 0,
                            wide_lane_integer_indicator: 0,
                            discontinuity_counter: 10,
                            bias: 3500,
                        }],
                    }];
                }
            }
            message
        };
        let rtcm_messages = [
            rtcm(1059, GnssSystem::Gps, SsrKind::CodeBias, 9),
            rtcm(1265, GnssSystem::Gps, SsrKind::PhaseBias, 9),
            rtcm(1242, GnssSystem::Galileo, SsrKind::CodeBias, 0),
            rtcm(1267, GnssSystem::Galileo, SsrKind::PhaseBias, 0),
        ];
        let rtcm_code_m = 70.0 * RTCM_SSR_CODE_BIAS_SCALE_M;
        let rtcm_phase_m = 3500.0 * RTCM_SSR_PHASE_BIAS_SCALE_M;
        let code = |band: char, attribute: char| SignalCode::new(band, attribute).unwrap();

        for has_first in [true, false] {
            let mut store = SsrCorrectionStore::new();
            let ingest_rtcm = |store: &mut SsrCorrectionStore| {
                for message in &rtcm_messages {
                    store.ingest_ssr(message, reception).unwrap();
                }
            };
            if !has_first {
                ingest_rtcm(&mut store);
            }
            let report = store.ingest_has_mt1_with_report(&has, reception).unwrap();
            if has_first {
                ingest_rtcm(&mut store);
            }

            // Whichever arrived first, each HAS record found no RTCM record on its signal.
            for (reason, status) in report
                .code_records
                .iter()
                .map(|r| (r.reason, r.resulting_status))
                .chain(
                    report
                        .phase_records
                        .iter()
                        .map(|r| (r.reason, r.resulting_status)),
                )
            {
                assert_eq!(reason, IngestionActionReason::AcceptedInitialRecord);
                assert_eq!(status, ActiveProvenanceStatus::ActiveHasUsable);
            }
            assert_eq!(
                report.code_records[0].signal,
                SsrRawSignal::galileo_has(GnssSystem::Gps, 9)
            );
            assert_eq!(
                report.code_records[0].key,
                SsrSignalKey::Physical(GnssSignal::new(GnssSystem::Gps, code('2', 'P')))
            );
            assert_eq!(
                report.phase_records[1].key,
                SsrSignalKey::Physical(GnssSignal::new(GnssSystem::Galileo, code('1', 'B')))
            );

            for (sat, index, has_code_m, has_cycles, rtcm_code) in [
                (gps, 9, 0.50, 0.20, code('2', 'X')),
                (gal, 0, 0.60, 0.30, code('1', 'A')),
            ] {
                let has_raw = SsrRawSignal::galileo_has(sat.system, index);
                let rtcm_raw = SsrRawSignal::rtcm_ssr(sat.system, index);
                assert_ne!(has_raw.key(), rtcm_raw.key());
                assert_eq!(
                    rtcm_raw.key(),
                    SsrSignalKey::Physical(GnssSignal::new(sat.system, rtcm_code))
                );

                let has_q = store.query_code_bias(sat, has_raw, t);
                assert_eq!(has_q.status, SsrBiasStatus::Available, "{sat} {has_first}");
                assert_eq!(has_q.bias_m, Some(has_code_m));
                assert_eq!(has_q.source_signal, Some(has_raw));
                assert_eq!(has_q.solution.unwrap().source, SsrSource::GalileoHas);
                let rtcm_q = store.query_code_bias(sat, rtcm_raw, t);
                assert_eq!(rtcm_q.status, SsrBiasStatus::Available, "{sat} {has_first}");
                assert_eq!(rtcm_q.bias_m.map(f64::to_bits), Some(rtcm_code_m.to_bits()));
                assert_eq!(rtcm_q.source_signal, Some(rtcm_raw));
                assert_eq!(rtcm_q.solution.unwrap().source, SsrSource::RtcmSsr);

                let has_p = store.query_phase_bias(sat, has_raw, t, None);
                assert_eq!(has_p.status, SsrBiasStatus::Available);
                assert_eq!(has_p.bias_cycles, Some(has_cycles));
                assert_eq!(has_p.source_signal, Some(has_raw));
                assert_eq!(
                    has_p.discontinuity_details,
                    Some(SsrDiscontinuityDetails::InitialTokenEstablished)
                );
                let has_token = has_p.continuity_token.unwrap();
                assert_eq!(has_token.source(), SsrSource::GalileoHas);
                assert_eq!(has_token.signal(), has_raw.key());
                assert_eq!(has_token.generation(), 0);
                let rtcm_p = store.query_phase_bias(sat, rtcm_raw, t, None);
                assert_eq!(rtcm_p.status, SsrBiasStatus::Available);
                assert_eq!(
                    rtcm_p.bias_m.map(f64::to_bits),
                    Some(rtcm_phase_m.to_bits())
                );
                assert_eq!(
                    rtcm_p.discontinuity_details,
                    Some(SsrDiscontinuityDetails::InitialTokenEstablished)
                );
                let rtcm_token = rtcm_p.continuity_token.unwrap();
                assert_eq!(rtcm_token.source(), SsrSource::RtcmSsr);
                assert_eq!(rtcm_token.generation(), 0);
                // Each token belongs to its own signal only.
                assert_eq!(
                    store
                        .query_phase_bias(sat, has_raw, t, Some(rtcm_token))
                        .discontinuity_details,
                    Some(SsrDiscontinuityDetails::MismatchedToken)
                );
            }

            // RTCM GPS index 10 is L2 P, the physical signal of HAS index 9: it shares
            // that entry, replaces the active record by arrival and breaks the phase arc.
            store
                .ingest_ssr(
                    &rtcm(1059, GnssSystem::Gps, SsrKind::CodeBias, 10),
                    reception,
                )
                .unwrap();
            store
                .ingest_ssr(
                    &rtcm(1265, GnssSystem::Gps, SsrKind::PhaseBias, 10),
                    reception,
                )
                .unwrap();
            let has_raw = SsrRawSignal::galileo_has(GnssSystem::Gps, 9);
            let shared = store.query_code_bias(gps, has_raw, t);
            assert_eq!(shared.status, SsrBiasStatus::Available);
            assert_eq!(
                shared.source_signal,
                Some(SsrRawSignal::rtcm_ssr(GnssSystem::Gps, 10))
            );
            assert_eq!(shared.solution.unwrap().source, SsrSource::RtcmSsr);
            let shared_p = store.query_phase_bias(gps, has_raw, t, None);
            assert!(matches!(
                shared_p.discontinuity_details,
                Some(SsrDiscontinuityDetails::SolutionChanged { .. })
            ));
            assert_eq!(shared_p.continuity_token.unwrap().generation(), 1);
            // The L2C(M+L) record at RTCM index 9 is untouched.
            let l2x =
                store.query_phase_bias(gps, SsrRawSignal::rtcm_ssr(GnssSystem::Gps, 9), t, None);
            assert_eq!(l2x.continuity_token.unwrap().generation(), 0);
            assert_eq!(
                l2x.source_signal,
                Some(SsrRawSignal::rtcm_ssr(GnssSystem::Gps, 9))
            );
        }
    }

    /// A bias on an index its source's table leaves unassigned is stored under its raw
    /// source-qualified signal, kept, reported as an unknown signal by the typed query
    /// once in its lifetime, and never collides with the other source's same index.
    #[test]
    fn unassigned_rtcm_and_has_indices_are_kept_apart_as_unknown_signals() {
        let gps = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let tow = 100_000.0;
        let reception = GnssWeekTow::new(TimeScale::Gst, 1042, tow + 20.0).unwrap();
        let mut header = header(SsrKind::CodeBias);
        header.epoch_time_s = (tow + 20.0) as u32;
        // RTCM SSR GPS index 12 is unassigned in RTKLIB `ssr_sig_gps`.
        let message = SsrMessage {
            message_number: 1059,
            system: GnssSystem::Gps,
            kind: SsrKind::CodeBias,
            header,
            orbit: Vec::new(),
            clock: Vec::new(),
            code_bias: vec![crate::rtcm::SsrCodeBiasRecord {
                satellite_id: 1,
                biases: vec![(12, 70), (0, 30)],
            }],
            phase_bias: Vec::new(),
            ura: Vec::new(),
            padding_bits: Vec::new(),
        };
        let mut store = SsrCorrectionStore::new();
        store.ingest_ssr(&message, reception).unwrap();
        let t = ssr_epoch_j2000_s(GnssSystem::Gps, 1059, reception, (tow + 20.0) as u32).unwrap();

        let raw = SsrRawSignal::rtcm_ssr(GnssSystem::Gps, 12);
        let q = store.query_code_bias(gps, raw, t);
        assert_eq!(q.status, SsrBiasStatus::UnknownSignal);
        assert_eq!(q.signal, SsrSignalKey::Unknown(raw));
        assert_eq!(q.source_signal, Some(raw));
        assert_eq!(q.details, SsrBiasResolutionDetails::UnknownSignal(raw));
        // The transmitted value is reported; the status keeps it from being applied.
        assert_eq!(
            q.bias_m.map(f64::to_bits),
            Some((70.0 * RTCM_SSR_CODE_BIAS_SCALE_M).to_bits())
        );
        assert_eq!(q.solution.unwrap().source, SsrSource::RtcmSsr);
        // The value is kept, and the untimed inspector returns it.
        assert_eq!(
            store.code_bias(gps, raw).map(f64::to_bits),
            Some((70.0 * RTCM_SSR_CODE_BIAS_SCALE_M).to_bits())
        );
        // HAS GPS index 12 is L5 Q, a different, assigned signal with no record here.
        assert_eq!(
            store
                .query_code_bias(gps, SsrRawSignal::galileo_has(GnssSystem::Gps, 12), t)
                .status,
            SsrBiasStatus::Missing
        );
        // Past its lifetime the record is expired, like any other.
        assert_eq!(
            store.query_code_bias(gps, raw, t + 1000.0).status,
            SsrBiasStatus::Expired
        );
        // The assigned signal of the same message applies.
        assert_eq!(
            store.query_code_bias(gps, rtcm_sig(gps, 0), t).status,
            SsrBiasStatus::Available
        );
    }
}
