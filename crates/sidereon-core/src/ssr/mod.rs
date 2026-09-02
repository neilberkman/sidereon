//! Engineering-unit State Space Representation corrections.
//!
//! The RTCM module stores raw transmitted integers. This module stores scaled
//! correction values keyed by satellite, with enough provider and issue metadata
//! to apply orbit and clock corrections on top of broadcast ephemerides.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::antex::{Antex, AntexDateTime};
use crate::astro::bodies::sun_moon_ecef;
use crate::astro::math::vec3::add3;
use crate::astro::time::civil::civil_from_j2000_seconds;
use crate::astro::time::model::{GnssWeekTow, TimeScale};
use crate::astro::time::scales::TimeScales;
use crate::broadcast::satellite_state_unchecked;
use crate::constants::{C_M_S, GPS_EPOCH_TO_J2000_S, SECONDS_PER_HOUR, SECONDS_PER_WEEK};
use crate::ephemeris::{BroadcastEphemeris, BroadcastIssue, NavMessage};
use crate::error::{Error, Result};
use crate::has::{has_mt1_reference_j2000_s, has_validity_interval_s, HasMt1Message};
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::observables::{ObservableEphemerisSource, ObservableState, ObservablesError};
use crate::ppp_corrections::satellite_body_pco_to_ecef;
use crate::rinex_nav::is_beidou_geo;
use crate::rtcm::{Message, SsrKind, SsrMessage};
use crate::spp::EphemerisSource;
use crate::staleness::StalenessPolicy;

const DEFAULT_SSR_STALENESS_S: f64 = 90.0;
const FD_HALF_S: f64 = 0.5;
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

/// Orbit correction for one satellite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsrOrbitCorrection {
    /// Provider and solution identity.
    pub solution: SsrSolution,
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
    /// Reference epoch, seconds since J2000.
    pub ref_epoch_j2000_s: f64,
    /// Update interval, seconds.
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
    /// Reference epoch, seconds since J2000.
    pub ref_epoch_j2000_s: f64,
    /// Update interval, seconds.
    pub update_interval_s: f64,
}

/// Clock correction for one satellite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsrClockCorrection {
    /// Provider and solution identity.
    pub solution: SsrSolution,
    /// IOD SSR.
    pub iod_ssr: u8,
    /// C0 term, meters.
    pub c0_m: f64,
    /// C1 term, meters per second.
    pub c1_m_s: f64,
    /// C2 term, meters per second squared.
    pub c2_m_s2: f64,
    /// Reference epoch, seconds since J2000.
    pub ref_epoch_j2000_s: f64,
    /// Update interval, seconds.
    pub update_interval_s: f64,
    /// Matching high-rate correction, when present.
    pub high_rate: Option<SsrHighRateClock>,
}

/// Code-bias correction placeholder for Phase B.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SsrCodeBias {
    biases_m: BTreeMap<u8, f64>,
}

/// Phase-bias correction placeholder for Phase B.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SsrPhaseBias {
    biases_m: BTreeMap<u8, f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct SatCorrections {
    orbit: Option<SsrOrbitCorrection>,
    clock: Option<SsrClockCorrection>,
    pending_high_rate: Option<SsrHighRateClock>,
    ura_index: Option<u8>,
    code_bias: SsrCodeBias,
    phase_bias: SsrPhaseBias,
}

/// Active SSR corrections keyed by satellite.
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
    pub fn ingest(&mut self, message: &Message, week: GnssWeekTow) -> Result<()> {
        if let Message::Ssr(ssr) = message {
            self.ingest_ssr(ssr, week)?;
        }
        Ok(())
    }

    /// Ingest one decoded RTCM SSR message.
    pub fn ingest_ssr(&mut self, message: &SsrMessage, week: GnssWeekTow) -> Result<()> {
        let update_interval_s = update_interval_s(message.header.update_interval);
        let ref_epoch_j2000_s = ssr_epoch_j2000_s(
            message.system,
            week,
            message.header.epoch_time_s,
            message.header.update_interval,
            update_interval_s,
        )?;
        let solution = SsrSolution {
            source: SsrSource::RtcmSsr,
            provider_id: message.header.provider_id,
            solution_id: message.header.solution_id,
        };

        match message.kind {
            SsrKind::Orbit => {
                for record in &message.orbit {
                    let sat = ssr_satellite(message.system, record.satellite_id)?;
                    let orbit = orbit_from_rtcm(
                        self.reference_point,
                        message,
                        solution,
                        record,
                        ref_epoch_j2000_s,
                        update_interval_s,
                    );
                    let entry = self.corrections.entry(sat).or_default();
                    entry.orbit = Some(orbit);
                }
            }
            SsrKind::Clock => {
                for record in &message.clock {
                    let sat = ssr_satellite(message.system, record.satellite_id)?;
                    let entry = self.corrections.entry(sat).or_default();
                    let mut clock = SsrClockCorrection {
                        solution,
                        iod_ssr: message.header.iod_ssr,
                        c0_m: f64::from(record.c0) * RTCM_SSR_RADIAL_CLOCK_SCALE_M,
                        c1_m_s: f64::from(record.c1) * RTCM_SSR_RADIAL_CLOCK_RATE_SCALE_M_S,
                        c2_m_s2: f64::from(record.c2) * RTCM_SSR_CLOCK_ACCEL_SCALE_M_S2,
                        ref_epoch_j2000_s,
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
                for (orbit_record, clock_record) in message.orbit.iter().zip(&message.clock) {
                    let sat = ssr_satellite(message.system, orbit_record.satellite_id)?;
                    let orbit = orbit_from_rtcm(
                        self.reference_point,
                        message,
                        solution,
                        orbit_record,
                        ref_epoch_j2000_s,
                        update_interval_s,
                    );
                    let entry = self.corrections.entry(sat).or_default();
                    entry.orbit = Some(orbit);
                    entry.clock = Some(SsrClockCorrection {
                        solution,
                        iod_ssr: message.header.iod_ssr,
                        c0_m: f64::from(clock_record.c0) * RTCM_SSR_RADIAL_CLOCK_SCALE_M,
                        c1_m_s: f64::from(clock_record.c1) * RTCM_SSR_RADIAL_CLOCK_RATE_SCALE_M_S,
                        c2_m_s2: f64::from(clock_record.c2) * RTCM_SSR_CLOCK_ACCEL_SCALE_M_S2,
                        ref_epoch_j2000_s,
                        update_interval_s,
                        high_rate: entry.pending_high_rate,
                    });
                }
            }
            SsrKind::Ura => {
                for &(satellite_id, ura_index) in &message.ura {
                    let sat = ssr_satellite(message.system, satellite_id)?;
                    self.corrections.entry(sat).or_default().ura_index = Some(ura_index);
                }
            }
            SsrKind::CodeBias => {
                for record in &message.code_bias {
                    let sat = ssr_satellite(message.system, record.satellite_id)?;
                    let entry = self.corrections.entry(sat).or_default();
                    for &(signal, bias) in &record.biases {
                        entry
                            .code_bias
                            .biases_m
                            .insert(signal, f64::from(bias) * RTCM_SSR_CODE_BIAS_SCALE_M);
                    }
                }
            }
            SsrKind::HighRateClock => {
                for record in &message.clock {
                    let sat = ssr_satellite(message.system, record.satellite_id)?;
                    let high_rate = SsrHighRateClock {
                        solution,
                        iod_ssr: message.header.iod_ssr,
                        c0_m: f64::from(record.c0) * RTCM_SSR_RADIAL_CLOCK_SCALE_M,
                        ref_epoch_j2000_s,
                        update_interval_s,
                    };
                    let entry = self.corrections.entry(sat).or_default();
                    entry.pending_high_rate = Some(high_rate);
                    if let Some(clock) = &mut entry.clock {
                        if high_rate_matches(clock, &high_rate) {
                            clock.high_rate = Some(high_rate);
                        }
                    }
                }
            }
            SsrKind::PhaseBias => {
                for record in &message.phase_bias {
                    let sat = ssr_satellite(message.system, record.satellite_id)?;
                    let entry = self.corrections.entry(sat).or_default();
                    for bias in &record.biases {
                        entry.phase_bias.biases_m.insert(
                            bias.signal_id,
                            f64::from(bias.bias) * RTCM_SSR_PHASE_BIAS_SCALE_M,
                        );
                    }
                }
            }
            SsrKind::Vtec => {}
        }
        Ok(())
    }

    /// Ingest one decoded Galileo HAS MT1 correction message.
    pub fn ingest_has_mt1(
        &mut self,
        message: &HasMt1Message,
        reception_gst: GnssWeekTow,
    ) -> Result<()> {
        let ref_epoch_j2000_s = has_mt1_reference_j2000_s(reception_gst, message.header.toh_s)?;
        let solution = SsrSolution {
            source: SsrSource::GalileoHas,
            provider_id: u16::from(message.header.mask_id),
            solution_id: message.header.iod_set_id,
        };
        if let Some(orbit) = &message.orbit {
            let update_interval_s = has_validity_interval_s(orbit.validity_interval)
                .ok_or_else(|| Error::Parse("HAS orbit VI is reserved".to_string()))?;
            for record in &orbit.records {
                self.corrections.entry(record.sat).or_default().orbit = Some(SsrOrbitCorrection {
                    solution,
                    iode: record.iode,
                    iod_ssr: message.header.iod_set_id,
                    basis: OrbitBasis::VelocityAligned,
                    crs_regional: false,
                    reference_point: SsrReferencePoint::AntennaPhaseCenter,
                    radial_m: record.radial_m,
                    along_m: record.along_m,
                    cross_m: record.cross_m,
                    radial_rate_m_s: 0.0,
                    along_rate_m_s: 0.0,
                    cross_rate_m_s: 0.0,
                    ref_epoch_j2000_s,
                    update_interval_s,
                });
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
                self.corrections.entry(record.sat).or_default().clock = Some(SsrClockCorrection {
                    solution,
                    iod_ssr: message.header.iod_set_id,
                    c0_m: record.correction_m,
                    c1_m_s: 0.0,
                    c2_m_s2: 0.0,
                    ref_epoch_j2000_s,
                    update_interval_s,
                    high_rate: None,
                });
            }
        }
        if let Some(code_bias) = &message.code_bias {
            for record in &code_bias.records {
                self.corrections
                    .entry(record.sat)
                    .or_default()
                    .code_bias
                    .biases_m
                    .insert(record.signal_id, record.bias_m);
            }
        }
        if let Some(phase_bias) = &message.phase_bias {
            for record in &phase_bias.records {
                self.corrections
                    .entry(record.sat)
                    .or_default()
                    .phase_bias
                    .biases_m
                    .insert(record.signal_id, record.bias_m);
            }
        }
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

    /// Code bias in meters for a satellite and raw signal id.
    pub fn code_bias(&self, sat: GnssSatelliteId, signal: u8) -> Option<f64> {
        self.corrections
            .get(&sat)?
            .code_bias
            .biases_m
            .get(&signal)
            .copied()
    }

    /// Phase bias for a satellite.
    pub fn phase_bias(&self, sat: GnssSatelliteId, signal: u8) -> Option<f64> {
        self.corrections
            .get(&sat)?
            .phase_bias
            .biases_m
            .get(&signal)
            .copied()
    }
}

fn orbit_from_rtcm(
    reference_point: SsrReferencePoint,
    message: &SsrMessage,
    solution: SsrSolution,
    record: &crate::rtcm::SsrOrbitRecord,
    ref_epoch_j2000_s: f64,
    update_interval_s: f64,
) -> SsrOrbitCorrection {
    SsrOrbitCorrection {
        solution,
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
        update_interval_s,
    }
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

/// Broadcast ephemeris corrected by an SSR store.
#[derive(Clone)]
pub struct SsrCorrectedEphemeris<'a> {
    broadcast: &'a BroadcastEphemeris,
    store: &'a SsrCorrectionStore,
    antex: Option<&'a Antex>,
    attitude: SsrSatelliteAttitude,
    staleness: StalenessPolicy,
    fallback: SsrFallbackPolicy,
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
        }
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

    /// Corrected ECEF position and satellite clock at a J2000 epoch.
    pub fn corrected_state(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<([f64; 3], f64)> {
        self.corrected_state_inner(sat, t_j2000_s)
            .or_else(|| self.broadcast_fallback_after_failure(sat, t_j2000_s))
    }

    fn corrected_state_inner(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        let orbit = self.store.orbit(sat)?;
        let clock = self.store.clock(sat)?;
        if orbit.solution != clock.solution || orbit.iod_ssr != clock.iod_ssr {
            return None;
        }
        if !self.correction_fresh(t_j2000_s, orbit.ref_epoch_j2000_s, orbit.update_interval_s) {
            return None;
        }
        if !self.correction_fresh(t_j2000_s, clock.ref_epoch_j2000_s, clock.update_interval_s) {
            return None;
        }
        if orbit.crs_regional && !self.regional_allowed(orbit.solution.provider_id) {
            return None;
        }

        let nav_message = default_nav_message(sat.system)?;
        let issue = BroadcastIssue {
            issue: orbit.iode,
            message: nav_message,
        };
        let record = self
            .broadcast
            .select_by_issue_at(sat, issue, nav_message, t_j2000_s)?;
        let (t_continuous_s, is_geo) = continuous_time_for_sat(sat, t_j2000_s)?;
        let sow = t_continuous_s.rem_euclid(SECONDS_PER_WEEK);
        let state = satellite_state_unchecked(
            &record.elements,
            &record.clock,
            &record.constants(),
            sow,
            record.broadcast_clock_group_delay_s(),
            is_geo,
        );
        let r = state.orbit.position().ok()?.as_array();
        let v = broadcast_velocity(record, sat, t_j2000_s, is_geo)?;
        let (er, ea, ec) = velocity_aligned_basis(r, v)?;
        let dt_orbit = t_j2000_s - orbit.ref_epoch_j2000_s;
        let radial = orbit.radial_m + orbit.radial_rate_m_s * dt_orbit;
        let along = orbit.along_m + orbit.along_rate_m_s * dt_orbit;
        let cross = orbit.cross_m + orbit.cross_rate_m_s * dt_orbit;
        let mut corrected_position = [
            r[0] + radial * er[0] + along * ea[0] + cross * ec[0],
            r[1] + radial * er[1] + along * ea[1] + cross * ec[1],
            r[2] + radial * er[2] + along * ea[2] + cross * ec[2],
        ];
        if orbit.reference_point == SsrReferencePoint::CenterOfMass {
            let pco_ecef_m = self.satellite_pco_to_apc(sat, t_j2000_s, corrected_position)?;
            corrected_position = add3(corrected_position, pco_ecef_m);
        }

        let dt_clock = t_j2000_s - clock.ref_epoch_j2000_s;
        let mut dclock_m =
            clock.c0_m + clock.c1_m_s * dt_clock + clock.c2_m_s2 * dt_clock * dt_clock;
        if let Some(high_rate) = clock.high_rate {
            if high_rate_matches(clock, &high_rate)
                && self.correction_fresh(
                    t_j2000_s,
                    high_rate.ref_epoch_j2000_s,
                    high_rate.update_interval_s,
                )
            {
                dclock_m += high_rate.c0_m;
            }
        }
        let corrected_clock_s = match clock.solution.source {
            SsrSource::RtcmSsr => state.clock.dt_clock_total_s - dclock_m / C_M_S,
            SsrSource::GalileoHas => state.clock.dt_clock_total_s + dclock_m / C_M_S,
        };
        Some((corrected_position, corrected_clock_s))
    }

    fn correction_fresh(
        &self,
        t_j2000_s: f64,
        ref_epoch_j2000_s: f64,
        update_interval_s: f64,
    ) -> bool {
        let age = (t_j2000_s - ref_epoch_j2000_s).abs();
        age.is_finite()
            && update_interval_s.is_finite()
            && update_interval_s >= 0.0
            && age <= self.staleness.max_staleness_s.min(update_interval_s)
    }

    fn regional_allowed(&self, provider_id: u16) -> bool {
        match &self.fallback.regional {
            RegionalPolicy::DeclineRegional => false,
            RegionalPolicy::AllowProviders(providers) => providers.contains(&provider_id),
        }
    }

    fn broadcast_fallback_after_failure(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        if self
            .store
            .orbit(sat)
            .is_some_and(|orbit| orbit.reference_point == SsrReferencePoint::CenterOfMass)
        {
            return None;
        }
        if self.fallback.on_missing_correction == MissingCorrectionAction::FallBackToBroadcast {
            self.broadcast.position_clock_at_j2000_s(sat, t_j2000_s)
        } else {
            None
        }
    }

    fn satellite_pco_to_apc(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        sat_position_ecef_m: [f64; 3],
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
}

impl ObservableEphemerisSource for SsrCorrectedEphemeris<'_> {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> std::result::Result<ObservableState, ObservablesError> {
        let Some((position_ecef_m, clock_s)) = self.corrected_state(sat, t_j2000_s) else {
            return Err(ObservablesError::NoEphemeris);
        };
        Ok(ObservableState {
            position_ecef_m,
            clock_s: Some(clock_s),
        })
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
        }
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

    fn borrowed(&self) -> SsrCorrectedEphemeris<'_> {
        let source = SsrCorrectedEphemeris::new(&self.broadcast, &self.store)
            .with_staleness(self.staleness)
            .with_fallback(self.fallback.clone())
            .with_satellite_attitude(self.attitude);
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
}

impl ObservableEphemerisSource for SsrCorrectedEphemerisOwned {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> std::result::Result<ObservableState, ObservablesError> {
        self.borrowed().observable_state_at_j2000_s(sat, t_j2000_s)
    }
}

fn update_interval_s(index: u8) -> f64 {
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
    TABLE[usize::from(index)]
}

fn ssr_epoch_j2000_s(
    system: GnssSystem,
    week: GnssWeekTow,
    epoch_time_s: u32,
    update_interval: u8,
    update_interval_s: f64,
) -> Result<f64> {
    let scale = match system {
        GnssSystem::Galileo => TimeScale::Gst,
        GnssSystem::BeiDou => TimeScale::Bdt,
        _ => TimeScale::Gpst,
    };
    let epoch_offset_s = if update_interval == 0 {
        0.0
    } else {
        update_interval_s / 2.0
    };
    let normalized = GnssWeekTow::new(scale, week.week, f64::from(epoch_time_s) + epoch_offset_s)
        .and_then(GnssWeekTow::normalized)
        .map_err(|_| Error::Parse("SSR epoch is out of range".to_string()))?;
    let continuous = f64::from(normalized.week) * SECONDS_PER_WEEK + normalized.tow_s;
    Ok(match system {
        GnssSystem::BeiDou => {
            continuous
                + crate::constants::BDS_EPOCH_MINUS_GPS_EPOCH_S
                + crate::constants::GPST_MINUS_BDT_S
                - GPS_EPOCH_TO_J2000_S
        }
        _ => continuous - GPS_EPOCH_TO_J2000_S,
    })
}

fn ssr_satellite(system: GnssSystem, satellite_id: u8) -> Result<GnssSatelliteId> {
    GnssSatelliteId::new(system, satellite_id)
        .map_err(|e| Error::Parse(format!("invalid SSR satellite id {satellite_id}: {e}")))
}

fn high_rate_matches(clock: &SsrClockCorrection, high_rate: &SsrHighRateClock) -> bool {
    clock.solution == high_rate.solution && clock.iod_ssr == high_rate.iod_ssr
}

fn default_nav_message(system: GnssSystem) -> Option<NavMessage> {
    match system {
        GnssSystem::Gps => Some(NavMessage::GpsLnav),
        GnssSystem::Galileo => Some(NavMessage::GalileoInav),
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

fn continuous_time_for_sat(sat: GnssSatelliteId, t_j2000_s: f64) -> Option<(f64, bool)> {
    if !matches!(
        sat.system,
        GnssSystem::Gps | GnssSystem::Galileo | GnssSystem::BeiDou
    ) {
        return None;
    }
    let gpst_continuous = t_j2000_s + GPS_EPOCH_TO_J2000_S;
    if sat.system == GnssSystem::BeiDou {
        Some((
            gpst_continuous
                - crate::constants::GPST_MINUS_BDT_S
                - crate::constants::BDS_EPOCH_MINUS_GPS_EPOCH_S,
            is_beidou_geo(sat),
        ))
    } else {
        Some((gpst_continuous, false))
    }
}

fn broadcast_velocity(
    record: &crate::rinex_nav::BroadcastRecord,
    sat: GnssSatelliteId,
    t_j2000_s: f64,
    is_geo: bool,
) -> Option<[f64; 3]> {
    let p_plus = broadcast_position_from_record(record, sat, t_j2000_s + FD_HALF_S, is_geo)?;
    let p_minus = broadcast_position_from_record(record, sat, t_j2000_s - FD_HALF_S, is_geo)?;
    let denom = 2.0 * FD_HALF_S;
    Some([
        (p_plus[0] - p_minus[0]) / denom,
        (p_plus[1] - p_minus[1]) / denom,
        (p_plus[2] - p_minus[2]) / denom,
    ])
}

fn broadcast_position_from_record(
    record: &crate::rinex_nav::BroadcastRecord,
    sat: GnssSatelliteId,
    t_j2000_s: f64,
    is_geo: bool,
) -> Option<[f64; 3]> {
    let (t_continuous_s, _) = continuous_time_for_sat(sat, t_j2000_s)?;
    let sow = t_continuous_s.rem_euclid(SECONDS_PER_WEEK);
    satellite_state_unchecked(
        &record.elements,
        &record.clock,
        &record.constants(),
        sow,
        record.broadcast_clock_group_delay_s(),
        is_geo,
    )
    .orbit
    .position()
    .ok()
    .map(|p| p.as_array())
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
    use crate::astro::math::vec3::dot3;
    use crate::constants::{F_L1_HZ, F_L2_HZ};
    use crate::has::{
        HasClockBlock, HasClockCorrection, HasCodeBias, HasCodeBiasBlock, HasGnssMask,
        HasMaskBlock, HasMt1Header, HasMt1Message, HasOrbitBlock, HasOrbitCorrection, HasPhaseBias,
        HasPhaseBiasBlock,
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

    #[test]
    fn interval_zero_epoch_uses_transmitted_time_not_midpoint() {
        let week = GnssWeekTow::new(TimeScale::Gpst, 2_400, 0.0).unwrap();
        let got_zero =
            ssr_epoch_j2000_s(GnssSystem::Gps, week, 100_000, 0, update_interval_s(0)).unwrap();
        let expected = f64::from(week.week) * SECONDS_PER_WEEK + 100_000.0 - GPS_EPOCH_TO_J2000_S;
        assert_eq!(got_zero.to_bits(), expected.to_bits());

        let got_nonzero =
            ssr_epoch_j2000_s(GnssSystem::Gps, week, 100_000, 1, update_interval_s(1)).unwrap();
        assert_eq!(got_nonzero.to_bits(), (expected + 1.0).to_bits());
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
        let iode = record.issue_of_data.issue;

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
        let (broadcast_position, broadcast_clock) = broadcast
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
        assert!((corrected_clock - (broadcast_clock - clock_correction_m / C_M_S)).abs() < 1.0e-18);
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
        let stale_iode = (record.issue_of_data.issue + 1) & 0xff;
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

    #[test]
    fn rtcm_update_interval_staleness_is_enforced_before_store_cap() {
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
                iode: record.issue_of_data.issue,
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
        let mut store = SsrCorrectionStore::new().with_staleness(StalenessPolicy::seconds(90.0));
        let week = GnssWeekTow::new(TimeScale::Gpst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("valid SSR week");
        let mut assembler = SsrStreamAssembler::new();
        for decoded in assembler.push(&message.to_frame().expect("frame RTCM SSR")) {
            store
                .ingest(&decoded.expect("decode RTCM SSR frame"), week)
                .expect("ingest SSR");
        }

        let strict = SsrCorrectedEphemeris::new(&broadcast, &store);
        assert!(strict.corrected_state(sat, t + 1.0).is_some());
        assert!(strict.corrected_state(sat, t + 1.25).is_none());

        let fallback =
            SsrCorrectedEphemeris::new(&broadcast, &store).with_fallback(SsrFallbackPolicy {
                on_missing_correction: MissingCorrectionAction::FallBackToBroadcast,
                regional: RegionalPolicy::DeclineRegional,
            });
        let got = fallback
            .corrected_state(sat, t + 1.25)
            .expect("broadcast fallback after update interval expiry");
        let expected = broadcast
            .position_clock_at_j2000_s(sat, t + 1.25)
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
            store.phase_bias(sat, 0).unwrap().to_bits(),
            0.125_f64.to_bits()
        );
        assert_eq!(
            store.phase_bias(sat, 9).unwrap().to_bits(),
            (-0.25_f64).to_bits()
        );
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
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat,
                    iode: record.issue_of_data.issue,
                    radial_m: 1.25,
                    along_m: -2.0,
                    cross_m: 3.0,
                }],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                records: vec![HasClockCorrection {
                    sat,
                    correction_m: -0.75,
                }],
            }),
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasCodeBias {
                        sat,
                        signal_id: 0,
                        bias_m: 0.24,
                    },
                    HasCodeBias {
                        sat,
                        signal_id: 9,
                        bias_m: -0.46,
                    },
                ],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasPhaseBias {
                        sat,
                        signal_id: 0,
                        bias_cycles: 1.25,
                        bias_m: 1.25 * C_M_S / F_L1_HZ,
                        discontinuity_indicator: 1,
                    },
                    HasPhaseBias {
                        sat,
                        signal_id: 9,
                        bias_cycles: -2.5,
                        bias_m: -2.5 * C_M_S / F_L2_HZ,
                        discontinuity_indicator: 2,
                    },
                ],
            }),
            padding_bits: Vec::new(),
        };
        let decoded = HasMt1Message::decode(&message.encode()).expect("decode HAS MT1");
        let mut store = SsrCorrectionStore::new();
        let reception = GnssWeekTow::new(TimeScale::Gst, REAL_SSR_WEEK, REAL_SSR_EPOCH_TOW_S)
            .expect("GST reception");
        store
            .ingest_has_mt1(&decoded, reception)
            .expect("ingest HAS MT1");
        let source = SsrCorrectedEphemeris::new(&broadcast, &store);
        let (position, clock) = source.corrected_state(sat, t).expect("HAS corrected state");
        let (broadcast_position, broadcast_clock) = broadcast
            .position_clock_at_j2000_s(sat, t)
            .expect("broadcast state");
        let velocity = finite_difference_broadcast_velocity(&broadcast, sat, t);
        let (er, ea, ec) = analytic_velocity_aligned_basis(broadcast_position, velocity);
        let expected_position = [
            broadcast_position[0] + 1.25 * er[0] - 2.0 * ea[0] + 3.0 * ec[0],
            broadcast_position[1] + 1.25 * er[1] - 2.0 * ea[1] + 3.0 * ec[1],
            broadcast_position[2] + 1.25 * er[2] - 2.0 * ea[2] + 3.0 * ec[2],
        ];
        assert_vector_close(position, expected_position, 2.0e-9);
        assert!((clock - (broadcast_clock - 0.75 / C_M_S)).abs() < 1.0e-18);
        assert_eq!(
            store.code_bias(sat, 0).unwrap().to_bits(),
            0.24_f64.to_bits()
        );
        assert_eq!(
            store.code_bias(sat, 9).unwrap().to_bits(),
            (-0.46_f64).to_bits()
        );
        assert_eq!(
            store.phase_bias(sat, 0).unwrap().to_bits(),
            (1.25 * C_M_S / F_L1_HZ).to_bits()
        );
        assert_eq!(
            store.phase_bias(sat, 9).unwrap().to_bits(),
            (-2.5 * C_M_S / F_L2_HZ).to_bits()
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

    #[test]
    fn corrected_position_matches_rtklib_satpos_ssr_oracle_for_one_epoch() {
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
        let (position, clock) = source
            .corrected_state(sat, ssr_j2000(REAL_SSR_EPOCH_TOW_S))
            .expect("corrected state");
        assert_eq!(
            position.map(f64::to_bits),
            [
                13_931_924_021_901_094_572,
                4_714_745_314_434_008_008,
                13_939_538_677_975_909_636,
            ]
        );
        assert_eq!(clock.to_bits(), 4_553_802_228_904_002_216);
        let rtklib_position = [
            -6_327_381.424_159_626,
            15_802_129.789_888_298,
            -20_121_898.098_271_403,
        ];
        let position_error_m = norm([
            position[0] - rtklib_position[0],
            position[1] - rtklib_position[1],
            position[2] - rtklib_position[2],
        ]);
        assert!(position_error_m < 1.0e-6, "{position_error_m}");
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

    fn finite_difference_broadcast_velocity(
        broadcast: &BroadcastEphemeris,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> [f64; 3] {
        let p_plus = broadcast
            .position_clock_at_j2000_s(sat, t_j2000_s + FD_HALF_S)
            .expect("broadcast plus state")
            .0;
        let p_minus = broadcast
            .position_clock_at_j2000_s(sat, t_j2000_s - FD_HALF_S)
            .expect("broadcast minus state")
            .0;
        [
            (p_plus[0] - p_minus[0]) / (2.0 * FD_HALF_S),
            (p_plus[1] - p_minus[1]) / (2.0 * FD_HALF_S),
            (p_plus[2] - p_minus[2]) / (2.0 * FD_HALF_S),
        ]
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
}
