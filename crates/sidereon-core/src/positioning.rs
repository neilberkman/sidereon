//! Single-point positioning and GNSS geometry diagnostics.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::constants::C_M_S;
pub use crate::dop::{dop, Dop, DopError, LineOfSight};
use crate::ephemeris::{BroadcastEphemeris, Sp3};
use crate::id::GnssSystem;
pub use crate::quality::{
    spp_robust_fde_driver, FdeError, FdeOptions, FdeResult, FdeSppError, FdeSppOptions,
};
use crate::rinex::observations::{pseudoranges, ObsEpochTime, ObservationFile, SignalPolicy};
use crate::rtcm::{self, MsmKind};
pub use crate::spp::{
    residual_rms, solve, solve_broadcast, solve_doppler_velocity, solve_spp_batch_parallel,
    solve_spp_batch_serial, solve_with_doppler_velocity, solve_with_fallback, solve_with_policy,
    solve_with_solver, BroadcastReason, ClockRelativity, Corrections, DopplerObservation,
    DopplerVelocityInputs, EphemerisSource, FallbackError, FixSource, GalileoNequickCoeffs,
    KlobucharCoeffs, Observation, PseudorangeCode, ReceiverSolution, RejectedSat, RejectionReason,
    RobustConfig, SolutionMetadata, SolveInputs, SolvePolicy, SolvePolicyError, SourcedSolution,
    SppDopplerSolution, SppError, SurfaceMet, DEFAULT_HUBER_K, DEFAULT_ROBUST_MAX_OUTER,
    DEFAULT_ROBUST_OUTER_TOL_M, DEFAULT_ROBUST_SCALE_FLOOR_M, ELEVATION_MASK_RAD, SIGMA0_M,
    TRANSMIT_TIME_ITERATIONS,
};
pub use crate::static_positioning::{
    solve_static, StaticClockBias, StaticCovariance, StaticEpoch, StaticEpochInfluence,
    StaticInfluenceStatus, StaticResidual, StaticSatelliteBatchInfluence, StaticSatelliteInfluence,
    StaticSolution, StaticSolutionMetadata, StaticSolveError, StaticSolveOptions,
};
pub use crate::static_reference_station::{
    solve_static_reference_station_rinex, StaticReferenceCarrierRinexOptions,
    StaticReferenceCarrierSolution, StaticReferenceCodeSolution, StaticReferenceEpochDiagnostic,
    StaticReferenceFixStatus, StaticReferenceModeError, StaticReferenceModeReport,
    StaticReferenceModeStatus, StaticReferenceStationCovariance, StaticReferenceStationError,
    StaticReferenceStationMode, StaticReferenceStationRinexOptions, StaticReferenceStationSolution,
};
use crate::{astro::time, Error as CoreError, GnssSatelliteId};

/// Role-oriented alias for a solved receiver state.
pub type Solution = ReceiverSolution;

/// Error type returned by [`solve`].
pub type Error = SppError;

/// Assembly-time error from building SPP inputs out of a parsed RINEX
/// observation file.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RinexSppError {
    /// A RINEX observation helper rejected malformed or non-finite input.
    Observation(CoreError),
    /// No initial receiver position was supplied and the observation header did
    /// not carry `APPROX POSITION XYZ`.
    MissingApproxPosition,
}

impl fmt::Display for RinexSppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Observation(error) => write!(f, "RINEX SPP observation assembly failed: {error}"),
            Self::MissingApproxPosition => {
                f.write_str("RINEX SPP assembly needs APPROX POSITION XYZ or an initial guess")
            }
        }
    }
}

impl std::error::Error for RinexSppError {}

impl From<CoreError> for RinexSppError {
    fn from(error: CoreError) -> Self {
        Self::Observation(error)
    }
}

/// Broadcast correction metadata used while converting RINEX observations into
/// SPP [`SolveInputs`].
///
/// A broadcast navigation product supplies ionosphere coefficients and GLONASS
/// FDMA channels. A precise SP3 product does not, so its default assembly
/// context is zero Klobuchar coefficients and no GLONASS channels. When solving
/// precise SP3 positions with broadcast atmosphere metadata, wrap the SP3 with
/// [`RinexSppSource::with_broadcast_context`].
#[derive(Debug, Clone, PartialEq)]
pub struct RinexSppBroadcastCorrections {
    /// GPS Klobuchar coefficients, also used as the fallback for systems
    /// without a dedicated correction set.
    pub klobuchar: KlobucharCoeffs,
    /// BeiDou-specific Klobuchar coefficients, when a NAV product provides
    /// `BDSA`/`BDSB`.
    pub beidou_klobuchar: Option<KlobucharCoeffs>,
    /// Galileo NeQuick-G coefficients, when a NAV product provides `GAL`.
    pub galileo_nequick: Option<GalileoNequickCoeffs>,
    /// GLONASS FDMA channel numbers keyed by GLONASS slot.
    pub glonass_channels: BTreeMap<u8, i8>,
}

impl Default for RinexSppBroadcastCorrections {
    fn default() -> Self {
        Self {
            klobuchar: zero_klobuchar(),
            beidou_klobuchar: None,
            galileo_nequick: None,
            glonass_channels: BTreeMap::new(),
        }
    }
}

/// Source of non-observation metadata needed during RINEX SPP assembly.
///
/// [`BroadcastEphemeris`] implements this trait with its parsed NAV ionosphere
/// coefficients and GLONASS FDMA channels. [`Sp3`] implements it with empty
/// broadcast metadata so precise-only callers can still assemble
/// troposphere-only or no-correction inputs.
pub trait RinexSppAssemblySource {
    /// Broadcast correction metadata available to RINEX SPP assembly.
    fn rinex_spp_broadcast_corrections(&self) -> RinexSppBroadcastCorrections;

    /// Replace the ionosphere coefficients of `corrections` with those in effect at
    /// `t_j2000_s` (GPS time), for a source whose coefficients change through the product
    /// (RINEX 4 `> ION` frames). The GLONASS channels are left as they are. The default
    /// leaves `corrections` unchanged.
    fn rinex_spp_ionosphere_at(
        &self,
        _t_j2000_s: f64,
        _corrections: &mut RinexSppBroadcastCorrections,
    ) {
    }
}

impl RinexSppAssemblySource for BroadcastEphemeris {
    fn rinex_spp_broadcast_corrections(&self) -> RinexSppBroadcastCorrections {
        let mut corrections = RinexSppBroadcastCorrections {
            glonass_channels: self.glonass_frequency_channels(),
            ..RinexSppBroadcastCorrections::default()
        };
        set_ionosphere(&mut corrections, self.iono_corrections());
        corrections
    }

    /// The ionosphere coefficients [`BroadcastEphemeris::iono_corrections_at`] selects for
    /// the epoch.
    fn rinex_spp_ionosphere_at(
        &self,
        t_j2000_s: f64,
        corrections: &mut RinexSppBroadcastCorrections,
    ) {
        set_ionosphere(corrections, self.iono_corrections_at(t_j2000_s));
    }
}

/// Put a NAV product's ionosphere coefficients in `corrections`.
fn set_ionosphere(
    corrections: &mut RinexSppBroadcastCorrections,
    iono: crate::ephemeris::IonoCorrections,
) {
    corrections.klobuchar = iono
        .gps
        .map(klobuchar_from_alpha_beta)
        .unwrap_or_else(zero_klobuchar);
    corrections.beidou_klobuchar = iono.beidou.map(klobuchar_from_alpha_beta);
    corrections.galileo_nequick = iono.galileo;
}

impl RinexSppAssemblySource for Sp3 {
    fn rinex_spp_broadcast_corrections(&self) -> RinexSppBroadcastCorrections {
        RinexSppBroadcastCorrections::default()
    }
}

/// Delegating ephemeris source that lets a precise product solve with broadcast
/// NAV metadata during RINEX SPP assembly.
///
/// Use [`Self::with_broadcast_context`] for the common precise-SP3-plus-RINEX-NAV
/// path: satellite position and clocks come from `ephemeris`, while
/// ionosphere coefficients and GLONASS FDMA channels come from `broadcast`.
pub struct RinexSppSource<'a, E: EphemerisSource + ?Sized> {
    ephemeris: &'a E,
    broadcast: Option<&'a BroadcastEphemeris>,
}

impl<'a, E: EphemerisSource + ?Sized> RinexSppSource<'a, E> {
    /// Build a delegating source with no broadcast assembly context.
    #[must_use]
    pub const fn new(ephemeris: &'a E) -> Self {
        Self {
            ephemeris,
            broadcast: None,
        }
    }

    /// Build a delegating source whose ephemeris is used for the solve and whose
    /// broadcast product is used for RINEX assembly metadata.
    #[must_use]
    pub const fn with_broadcast_context(
        ephemeris: &'a E,
        broadcast: &'a BroadcastEphemeris,
    ) -> Self {
        Self {
            ephemeris,
            broadcast: Some(broadcast),
        }
    }

    /// The ephemeris source delegated to during the SPP solve.
    #[must_use]
    pub const fn ephemeris(&self) -> &'a E {
        self.ephemeris
    }

    /// Broadcast metadata source, when one was supplied.
    #[must_use]
    pub const fn broadcast_context(&self) -> Option<&'a BroadcastEphemeris> {
        self.broadcast
    }
}

impl<E: EphemerisSource + ?Sized> EphemerisSource for RinexSppSource<'_, E> {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        self.ephemeris.position_clock_at_j2000_s(sat, t_j2000_s)
    }

    fn single_frequency_group_delay_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        EphemerisSource::single_frequency_group_delay_s(self.ephemeris, sat, t_j2000_s)
    }

    fn clock_relativity_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> crate::spp::ClockRelativity {
        EphemerisSource::clock_relativity_s(self.ephemeris, sat, t_j2000_s)
    }

    fn clock_relativity_for_state_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        position_m: [f64; 3],
    ) -> crate::spp::ClockRelativity {
        EphemerisSource::clock_relativity_for_state_s(self.ephemeris, sat, t_j2000_s, position_m)
    }

    fn position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        EphemerisSource::position_clock_group_delay_at_j2000_s(self.ephemeris, sat, t_j2000_s)
    }

    fn try_position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<crate::astro::time::Validated<crate::spp::PositionClock>>, crate::Error>
    {
        self.ephemeris.try_position_clock_at_j2000_s(sat, t_j2000_s)
    }

    fn try_position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<
        Option<crate::astro::time::Validated<crate::spp::PositionClockGroupDelay>>,
        crate::Error,
    > {
        self.ephemeris
            .try_position_clock_group_delay_at_j2000_s(sat, t_j2000_s)
    }
}

impl<E: EphemerisSource + ?Sized> RinexSppAssemblySource for RinexSppSource<'_, E> {
    fn rinex_spp_broadcast_corrections(&self) -> RinexSppBroadcastCorrections {
        self.broadcast
            .map(RinexSppAssemblySource::rinex_spp_broadcast_corrections)
            .unwrap_or_default()
    }

    fn rinex_spp_ionosphere_at(
        &self,
        t_j2000_s: f64,
        corrections: &mut RinexSppBroadcastCorrections,
    ) {
        if let Some(broadcast) = self.broadcast {
            broadcast.rinex_spp_ionosphere_at(t_j2000_s, corrections);
        }
    }
}

/// Options for assembling RINEX observation epochs into SPP [`SolveInputs`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RinexSppOptions {
    /// Per-constellation pseudorange-code selection policy.
    pub signal_policy: SignalPolicy,
    /// Correction terms to request in each assembled solve.
    pub corrections: Corrections,
    /// Optional initial guess `[x_m, y_m, z_m, b_m]`. When absent, the RINEX
    /// header's `APPROX POSITION XYZ` provides the position and the clock seed
    /// is zero.
    pub initial_guess: Option<[f64; 4]>,
    /// Optional satellite allow-list. `None` keeps every satellite with a
    /// selected pseudorange.
    pub satellites: Option<BTreeSet<GnssSatelliteId>>,
    /// Surface meteorology for troposphere correction.
    pub met: SurfaceMet,
    /// Optional robust reweighting for every assembled epoch.
    pub robust: Option<RobustConfig>,
}

impl RinexSppOptions {
    /// Build options from an explicit signal policy.
    #[must_use]
    pub fn new(signal_policy: SignalPolicy) -> Self {
        Self {
            signal_policy,
            corrections: Corrections::IONO_TROPO,
            initial_guess: None,
            satellites: None,
            met: SurfaceMet::default(),
            robust: None,
        }
    }

    /// Build options using the default single-frequency signal policy for the
    /// observation file's RINEX version.
    pub fn default_for(obs: &ObservationFile) -> Result<Self, RinexSppError> {
        Ok(Self::new(SignalPolicy::default_for(obs.header().version)?))
    }

    /// Replace the correction request.
    #[must_use]
    pub const fn with_corrections(mut self, corrections: Corrections) -> Self {
        self.corrections = corrections;
        self
    }

    /// Replace the initial solve guess.
    #[must_use]
    pub const fn with_initial_guess(mut self, initial_guess: [f64; 4]) -> Self {
        self.initial_guess = Some(initial_guess);
        self
    }

    /// Restrict assembly to the supplied satellites.
    #[must_use]
    pub fn with_satellites<I>(mut self, satellites: I) -> Self
    where
        I: IntoIterator<Item = GnssSatelliteId>,
    {
        self.satellites = Some(satellites.into_iter().collect());
        self
    }

    /// Replace surface meteorology.
    #[must_use]
    pub const fn with_surface_met(mut self, met: SurfaceMet) -> Self {
        self.met = met;
        self
    }

    /// Replace robust-reweighting config.
    #[must_use]
    pub const fn with_robust(mut self, robust: Option<RobustConfig>) -> Self {
        self.robust = robust;
        self
    }
}

/// One assembled RINEX observation epoch and its SPP inputs.
#[derive(Debug, Clone)]
pub struct RinexSppEpochInputs {
    /// Index in [`ObservationFile::epochs`].
    pub epoch_index: usize,
    /// Civil epoch exactly as it appears in the RINEX observation file.
    pub epoch: ObsEpochTime,
    /// Fully assembled SPP inputs for this epoch.
    pub inputs: SolveInputs,
}

/// One RINEX observation epoch paired with its serial SPP solve result.
#[derive(Debug, Clone)]
pub struct RinexSppEpochSolution {
    /// Index in [`ObservationFile::epochs`].
    pub epoch_index: usize,
    /// Civil epoch exactly as it appears in the RINEX observation file.
    pub epoch: ObsEpochTime,
    /// Result from solving the assembled epoch.
    pub solution: Result<ReceiverSolution, SolvePolicyError>,
}

/// One set of assembled RTCM MSM observations and its SPP inputs.
#[derive(Debug, Clone)]
pub struct RtcmSppEpochInputs {
    /// Index in the ordered set of assembled RTCM observation epochs.
    pub epoch_index: usize,
    /// Civil epoch reconstructed from the stream conversion logic.
    pub epoch: ObsEpochTime,
    /// Fully assembled SPP inputs for this epoch.
    pub inputs: SolveInputs,
}

/// Offset from an SBAS MSM satellite-mask number to the SBAS broadcast PRN:
/// number `n` is PRN `119 + n`, as RTKLIB `decode_msm7` applies it
/// (`prn += MINPRNSBS - 1`).
const MSM_SBAS_PRN_OFFSET: u16 = 119;

/// Map an MSM satellite-mask number to the satellite it names.
///
/// MSM numbers satellites from 1 within each system. For SBAS, number `n` is
/// broadcast PRN `119 + n`, so it is converted through the SBAS broadcast PRN
/// window (`n` 1..=39 gives `S20`..`S58`) instead of being read as the slot
/// itself. For QZSS, number `n` is PRN `192 + n`, which is already the `Jnn`
/// slot. Every other system uses the number as its PRN or slot. A number that
/// names no satellite in that form returns `None`.
fn msm_satellite_id(system: GnssSystem, number: u8) -> Option<GnssSatelliteId> {
    match system {
        GnssSystem::Sbas => {
            crate::sbas::store::sbas_prn_to_sat(u16::from(number).checked_add(MSM_SBAS_PRN_OFFSET)?)
        }
        _ => GnssSatelliteId::new(system, number).ok(),
    }
}

/// Convert RTCM MSM observation messages into SPP-ready epoch inputs.
///
/// Messages are grouped by `(system, epoch_time)` in stream order, so each
/// RTCM epoch produces one `RtcmSppEpochInputs` entry.
pub fn spp_inputs_from_rtcm_msm<S, F>(
    messages: &[rtcm::MsmMessage],
    source: &S,
    options: &RinexSppOptions,
    mut map_epoch: F,
) -> Result<Vec<RtcmSppEpochInputs>, RinexSppError>
where
    S: RinexSppAssemblySource + ?Sized,
    F: FnMut(GnssSystem, u32) -> Option<(f64, ObsEpochTime)>,
{
    if messages.is_empty() {
        return Ok(Vec::new());
    }

    let initial_guess = options.initial_guess.unwrap_or([0.0; 4]);
    let mut groups = Vec::<(GnssSystem, u32, Vec<usize>)>::new();
    let mut group_index = BTreeMap::<(GnssSystem, u32), usize>::new();

    for (index, message) in messages.iter().enumerate() {
        let key = (message.system, message.header.epoch_time);
        let slot = if let Some(index) = group_index.get(&key) {
            *index
        } else {
            let slot = groups.len();
            groups.push((key.0, key.1, Vec::new()));
            group_index.insert(key, slot);
            slot
        };
        groups[slot].2.push(index);
    }

    // The source's corrections, GLONASS channels included, are built once; only the
    // ionosphere coefficients are selected per epoch.
    let source_corrections = source.rinex_spp_broadcast_corrections();
    let mut out = Vec::new();
    for (epoch_index, (system, epoch_time, group_indexes)) in groups.into_iter().enumerate() {
        let Some((t_rx_j2000_s, epoch)) = map_epoch(system, epoch_time) else {
            continue;
        };
        // The broadcast ionosphere coefficients in effect at the epoch.
        let mut epoch_iono = ionosphere_only(&source_corrections);
        source.rinex_spp_ionosphere_at(t_rx_j2000_s, &mut epoch_iono);
        let Some(preferred_codes) = options.signal_policy.codes.get(&system) else {
            continue;
        };
        let preferred_codes = preferred_codes
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();

        // Each signal is read with the MSM type and satellite data of the
        // message that carries it: an epoch may mix MSM4 and MSM7, whose fine
        // pseudoranges have different scales and invalid values.
        let mut by_satellite = BTreeMap::<u8, Vec<RtcmCell<'_>>>::new();
        for message in group_indexes
            .iter()
            .copied()
            .filter_map(|index| messages.get(index))
        {
            for signal in &message.signals {
                let Some(satellite) = message
                    .satellites
                    .iter()
                    .find(|satellite| satellite.id == signal.satellite_id)
                else {
                    continue;
                };
                by_satellite
                    .entry(signal.satellite_id)
                    .or_default()
                    .push(RtcmCell {
                        kind: message.kind,
                        satellite: *satellite,
                        signal,
                    });
            }
        }

        let mut observations = Vec::new();
        for (satellite_id, cells) in by_satellite {
            let Some(pseudorange_m) = rtcm_msm_pseudorange_m(system, &cells, &preferred_codes)
            else {
                continue;
            };
            if let Some(satellite_id) = msm_satellite_id(system, satellite_id) {
                observations.push(Observation {
                    satellite_id,
                    pseudorange_m,
                });
            }
        }

        if observations.is_empty() {
            continue;
        }

        let t_rx_second_of_day_s =
            time::second_of_day(epoch.hour.into(), epoch.minute.into(), epoch.second);
        let day_of_year = time::day_of_year(
            epoch.year,
            i32::from(epoch.month),
            i32::from(epoch.day),
            epoch.hour.into(),
            epoch.minute.into(),
            epoch.second,
        );

        out.push(RtcmSppEpochInputs {
            epoch_index,
            epoch,
            inputs: SolveInputs {
                observations,
                t_rx_j2000_s,
                t_rx_second_of_day_s,
                day_of_year,
                initial_guess,
                corrections: options.corrections,
                klobuchar: epoch_iono.klobuchar,
                beidou_klobuchar: epoch_iono.beidou_klobuchar,
                galileo_nequick: epoch_iono.galileo_nequick,
                sbas_iono: None,
                glonass_channels: source_corrections.glonass_channels.clone(),
                met: options.met,
                robust: options.robust,
                pseudorange_code: crate::spp::PseudorangeCode::SingleFrequency,
            },
        });
    }

    Ok(out)
}

/// One MSM signal cell with the MSM type and satellite data of its message.
#[derive(Clone, Copy)]
struct RtcmCell<'a> {
    kind: MsmKind,
    satellite: rtcm::MsmSatellite,
    signal: &'a rtcm::MsmSignal,
}

/// The pseudorange of the selected cell, or `None` when its rough range or its
/// fine pseudorange is the field's invalid value (DF397 255; DF400 `-2^14` in
/// MSM4, DF405 `-2^19` in MSM7), as RTKLIB `decode_msm4` and `decode_msm7`
/// leave such a pseudorange unset. A whole-millisecond rough range of 0 also
/// gives `None`: RTKLIB `decode_msm4`..`decode_msm7` leave the satellite range
/// at zero, add no modulo-1-ms remainder to it, and `save_msm_obs` stores a
/// pseudorange only when that range is nonzero.
fn rtcm_msm_pseudorange_m(
    system: GnssSystem,
    cells: &[RtcmCell<'_>],
    preferred_codes: &[&str],
) -> Option<f64> {
    let cell = select_rtcm_signal(system, cells, preferred_codes)?;
    let satellite = cell.satellite;
    if satellite.rough_range_ms == rtcm::MSM_ROUGH_RANGE_INVALID || satellite.rough_range_ms == 0 {
        return None;
    }
    let rough_ms =
        f64::from(satellite.rough_range_ms) + f64::from(satellite.rough_range_mod1) / 1024.0;
    let fine = cell.signal.fine_pseudorange;
    let fine_ms = match cell.kind {
        MsmKind::Msm4 => {
            if fine == rtcm::MSM4_FINE_PSEUDORANGE_INVALID {
                return None;
            }
            f64::from(fine) / 2_f64.powi(24)
        }
        MsmKind::Msm7 => {
            if fine == rtcm::MSM7_FINE_PSEUDORANGE_INVALID {
                return None;
            }
            f64::from(fine) / 2_f64.powi(29)
        }
    };
    Some((rough_ms + fine_ms) * 1.0e-3 * C_M_S)
}

fn select_rtcm_signal<'a>(
    system: GnssSystem,
    cells: &[RtcmCell<'a>],
    preferred_codes: &[&str],
) -> Option<RtcmCell<'a>> {
    if cells.is_empty() {
        return None;
    }

    if preferred_codes.is_empty() {
        return cells.first().copied();
    }

    preferred_codes
        .iter()
        .find_map(|requested| {
            let normalized = requested
                .strip_prefix('C')
                .or_else(|| requested.strip_prefix('L'))
                .unwrap_or(requested);
            cells.iter().copied().find(|cell| {
                let Some(code) = rtcm::msm_signal_rinex_code(system, cell.signal.signal_id) else {
                    return false;
                };
                code == *requested || code == normalized
            })
        })
        .or_else(|| cells.first().copied())
}

/// Assemble every non-event RINEX observation epoch with at least one selected
/// pseudorange into SPP [`SolveInputs`].
///
/// The function preserves observation-file epoch order, skips RINEX event and
/// cycle slip epochs (`flag > 1`) and any epoch without an epoch time, selects
/// one single-frequency pseudorange per satellite
/// under [`RinexSppOptions::signal_policy`], derives receive time from the RINEX
/// civil epoch, seeds the receiver from the `APPROX POSITION XYZ` in effect at
/// each epoch unless `initial_guess` is supplied, and combines GLONASS channels
/// from the assembly source with the observation `GLONASS SLOT / FRQ #` entries
/// in effect at each epoch. Observation header channels take precedence. A
/// record an event carries is in effect from its epoch, as
/// [`ObservationFile::header_at`] gives it.
pub fn spp_inputs_from_rinex_obs<S>(
    obs: &ObservationFile,
    source: &S,
    options: &RinexSppOptions,
) -> Result<Vec<RinexSppEpochInputs>, RinexSppError>
where
    S: RinexSppAssemblySource + ?Sized,
{
    // A position, an antenna or a GLONASS channel an event declares applies to
    // the epochs after it.
    let timeline = obs.header_timeline()?;
    // The source's corrections, merged with the GLONASS channels of each header segment,
    // are built once; only the ionosphere coefficients are selected per epoch.
    let source_corrections = source.rinex_spp_broadcast_corrections();
    let segment_corrections: Vec<RinexSppBroadcastCorrections> = timeline
        .segments()
        .map(|(_, header)| merged_broadcast_corrections(header, &source_corrections))
        .collect();
    let mut out = Vec::new();

    for (epoch_index, epoch) in obs.epochs().iter().enumerate() {
        // An event, a cycle slip record, and an epoch with no epoch time hold
        // no measurement a receive time can be derived for.
        let Some(epoch_time) = epoch.epoch.filter(|_| epoch.flag <= 1) else {
            continue;
        };
        let header = timeline.at(epoch_index);
        let initial_guess = initial_guess(header, options)?;
        let mut selected = pseudoranges(obs, epoch, &options.signal_policy)?;
        if let Some(allowed) = &options.satellites {
            selected.retain(|(sat, _)| allowed.contains(sat));
        }
        if selected.is_empty() {
            continue;
        }

        let epoch_context = epoch_time_context(epoch_time);
        // The corrections of the header segment in effect at the epoch, with the
        // broadcast ionosphere coefficients in effect at it.
        let Some(base_corrections) = segment_corrections.get(timeline.segment_index(epoch_index))
        else {
            continue;
        };
        let mut epoch_iono = ionosphere_only(base_corrections);
        source.rinex_spp_ionosphere_at(epoch_context.t_rx_j2000_s, &mut epoch_iono);
        let observations = selected
            .into_iter()
            .map(|(satellite_id, pseudorange_m)| Observation {
                satellite_id,
                pseudorange_m,
            })
            .collect();

        out.push(RinexSppEpochInputs {
            epoch_index,
            epoch: epoch_time,
            inputs: SolveInputs {
                observations,
                t_rx_j2000_s: epoch_context.t_rx_j2000_s,
                t_rx_second_of_day_s: epoch_context.t_rx_second_of_day_s,
                day_of_year: epoch_context.day_of_year,
                initial_guess,
                corrections: options.corrections,
                klobuchar: epoch_iono.klobuchar,
                beidou_klobuchar: epoch_iono.beidou_klobuchar,
                galileo_nequick: epoch_iono.galileo_nequick,
                sbas_iono: None,
                glonass_channels: base_corrections.glonass_channels.clone(),
                met: options.met,
                robust: options.robust,
                pseudorange_code: crate::spp::PseudorangeCode::SingleFrequency,
            },
        });
    }

    Ok(out)
}

/// Assemble RINEX SPP epochs and solve them serially against the same source.
///
/// The returned vector has one entry per assembled epoch, not one entry per raw
/// RINEX epoch. Per-epoch solve failures are retained in
/// [`RinexSppEpochSolution::solution`], matching [`solve_spp_batch_serial`].
pub fn solve_spp_from_rinex_obs<S>(
    source: &S,
    obs: &ObservationFile,
    options: &RinexSppOptions,
    with_geodetic: bool,
    policy: SolvePolicy,
) -> Result<Vec<RinexSppEpochSolution>, RinexSppError>
where
    S: EphemerisSource + RinexSppAssemblySource,
{
    let epochs = spp_inputs_from_rinex_obs(obs, source, options)?;
    let inputs = epochs
        .iter()
        .map(|epoch| epoch.inputs.clone())
        .collect::<Vec<_>>();
    let results = solve_spp_batch_serial(source, &inputs, with_geodetic, policy);
    Ok(epochs
        .into_iter()
        .zip(results)
        .map(|(epoch, solution)| RinexSppEpochSolution {
            epoch_index: epoch.epoch_index,
            epoch: epoch.epoch,
            solution,
        })
        .collect())
}

fn klobuchar_from_alpha_beta(value: crate::ephemeris::KlobucharAlphaBeta) -> KlobucharCoeffs {
    KlobucharCoeffs {
        alpha: value.alpha,
        beta: value.beta,
    }
}

const fn zero_klobuchar() -> KlobucharCoeffs {
    KlobucharCoeffs {
        alpha: [0.0; 4],
        beta: [0.0; 4],
    }
}

fn initial_guess(
    header: &crate::rinex::observations::ObsHeader,
    options: &RinexSppOptions,
) -> Result<[f64; 4], RinexSppError> {
    if let Some(initial_guess) = options.initial_guess {
        return Ok(initial_guess);
    }
    let approx = header
        .approx_position_m
        .ok_or(RinexSppError::MissingApproxPosition)?;
    Ok([approx[0], approx[1], approx[2], 0.0])
}

/// The ionosphere coefficients of `corrections`, without its GLONASS channels.
fn ionosphere_only(corrections: &RinexSppBroadcastCorrections) -> RinexSppBroadcastCorrections {
    RinexSppBroadcastCorrections {
        klobuchar: corrections.klobuchar,
        beidou_klobuchar: corrections.beidou_klobuchar,
        galileo_nequick: corrections.galileo_nequick,
        glonass_channels: BTreeMap::new(),
    }
}

fn merged_broadcast_corrections(
    header: &crate::rinex::observations::ObsHeader,
    source_corrections: &RinexSppBroadcastCorrections,
) -> RinexSppBroadcastCorrections {
    let mut corrections = source_corrections.clone();
    corrections.glonass_channels.extend(
        header
            .glonass_slots
            .iter()
            .map(|(&slot, &channel)| (slot, channel)),
    );
    corrections
}

struct EpochTimeContext {
    t_rx_j2000_s: f64,
    t_rx_second_of_day_s: f64,
    day_of_year: f64,
}

fn epoch_time_context(epoch: ObsEpochTime) -> EpochTimeContext {
    let year = epoch.year;
    let month = i32::from(epoch.month);
    let day = i32::from(epoch.day);
    let hour = i32::from(epoch.hour);
    let minute = i32::from(epoch.minute);
    EpochTimeContext {
        t_rx_j2000_s: time::j2000_seconds(year, month, day, hour, minute, epoch.second),
        t_rx_second_of_day_s: time::second_of_day(hour, minute, epoch.second),
        day_of_year: time::day_of_year(year, month, day, hour, minute, epoch.second),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rinex_obs::SignalPolicy;
    use crate::rtcm::{MsmHeader, MsmKind, MsmMessage, MsmSatellite, MsmSignal};

    #[derive(Default)]
    struct NoCorrections;

    impl RinexSppAssemblySource for NoCorrections {
        fn rinex_spp_broadcast_corrections(&self) -> RinexSppBroadcastCorrections {
            RinexSppBroadcastCorrections::default()
        }
    }

    fn synthetic_rtcm_messages() -> Vec<MsmMessage> {
        vec![MsmMessage {
            message_number: 1074,
            system: GnssSystem::Gps,
            kind: MsmKind::Msm4,
            header: MsmHeader {
                reference_station_id: 0,
                epoch_time: 12_345,
                multiple_message: false,
                iods: 1,
                reserved: 0,
                clock_steering: 0,
                external_clock: 0,
                divergence_free_smoothing: false,
                smoothing_interval: 0,
            },
            signal_mask: 0xC000_0000,
            satellites: vec![MsmSatellite {
                id: 1,
                rough_range_ms: 100,
                rough_range_mod1: 512,
                extended_info: None,
                rough_phase_range_rate_m_s: None,
            }],
            signals: vec![
                MsmSignal {
                    satellite_id: 1,
                    signal_id: 1,
                    fine_pseudorange: 1 << 24,
                    lock_time_indicator: 0,
                    half_cycle_ambiguity: false,
                    cnr: 0,
                    fine_phase_range: 0,
                    fine_phase_range_rate: None,
                },
                MsmSignal {
                    satellite_id: 1,
                    signal_id: 2,
                    fine_pseudorange: 0,
                    lock_time_indicator: 0,
                    half_cycle_ambiguity: false,
                    cnr: 0,
                    fine_phase_range: 0,
                    fine_phase_range_rate: None,
                },
            ],
            trailing_bits: Vec::new(),
        }]
    }

    #[test]
    fn rtcm_msm_helper_assembles_single_epoch_with_signal_selection() {
        let messages = synthetic_rtcm_messages();
        let options = RinexSppOptions::new(SignalPolicy::default_for(3.03).expect("policy"));
        let inputs =
            spp_inputs_from_rtcm_msm(&messages, &NoCorrections, &options, |_system, _raw| {
                Some((
                    1_234_567.0,
                    ObsEpochTime {
                        year: 2026,
                        month: 7,
                        day: 7,
                        hour: 0,
                        minute: 0,
                        second: 0.0,
                    },
                ))
            })
            .expect("convert");

        assert_eq!(inputs.len(), 1);
        let epoch = &inputs[0];
        assert_eq!(epoch.inputs.observations.len(), 1);
        assert_eq!(epoch.inputs.observations[0].satellite_id.to_string(), "G01");
        assert_eq!(
            epoch.inputs.observations[0].pseudorange_m as i64,
            30_129_142
        );
    }

    fn single_signal_msm(kind: MsmKind, satellite_id: u8, fine_pseudorange: i32) -> MsmMessage {
        let (message_number, extended_info, fine_phase_range_rate) = match kind {
            MsmKind::Msm4 => (1074, None, None),
            MsmKind::Msm7 => (1077, Some(0), Some(0)),
        };
        let mut message = synthetic_rtcm_messages().remove(0);
        message.message_number = message_number;
        message.kind = kind;
        message.signal_mask = 1 << (32 - 2);
        message.satellites[0].id = satellite_id;
        message.satellites[0].extended_info = extended_info;
        message.signals = vec![MsmSignal {
            satellite_id,
            signal_id: 2,
            fine_pseudorange,
            lock_time_indicator: 0,
            half_cycle_ambiguity: false,
            cnr: 0,
            fine_phase_range: 0,
            fine_phase_range_rate,
        }];
        message
    }

    fn solve_inputs(messages: &[MsmMessage]) -> Vec<RtcmSppEpochInputs> {
        let options = RinexSppOptions::new(SignalPolicy::default_for(3.03).expect("policy"));
        spp_inputs_from_rtcm_msm(messages, &NoCorrections, &options, |_system, _raw| {
            Some((
                1_234_567.0,
                ObsEpochTime {
                    year: 2026,
                    month: 7,
                    day: 7,
                    hour: 0,
                    minute: 0,
                    second: 0.0,
                },
            ))
        })
        .expect("convert")
    }

    /// An MSM7 fine pseudorange of `-2^19` (DF405) is the invalid value, as
    /// RTKLIB `decode_msm7` tests it, and yields no pseudorange. It was read as
    /// a range about 293 km short of the rough range.
    #[test]
    fn msm7_invalid_fine_pseudorange_yields_no_observation() {
        let invalid =
            single_signal_msm(MsmKind::Msm7, 1, crate::rtcm::MSM7_FINE_PSEUDORANGE_INVALID);
        assert!(solve_inputs(&[invalid]).is_empty());
        let valid = single_signal_msm(MsmKind::Msm7, 1, 0);
        assert_eq!(solve_inputs(&[valid])[0].inputs.observations.len(), 1);
    }

    /// A rough range of 0 whole milliseconds gives no pseudorange, as RTKLIB
    /// `save_msm_obs` stores none while the satellite range is zero.
    #[test]
    fn msm_zero_rough_range_yields_no_observation() {
        let mut zero = single_signal_msm(MsmKind::Msm7, 1, 0);
        zero.satellites[0].rough_range_ms = 0;
        assert!(solve_inputs(&[zero]).is_empty());
    }

    /// Each signal is scaled by the MSM type of the message that carries it.
    /// An epoch whose first message was MSM4 read an MSM7 fine pseudorange at
    /// the MSM4 scale, 2^5 times too large.
    #[test]
    fn mixed_msm4_and_msm7_epoch_scales_each_signal_by_its_own_message() {
        // The same raw value, 2^13, is 2^-11 ms at the MSM4 scale (DF400,
        // 2^-24 ms) and 2^-16 ms at the MSM7 scale (DF405, 2^-29 ms).
        let msm4 = single_signal_msm(MsmKind::Msm4, 1, 1 << 13);
        let msm7 = single_signal_msm(MsmKind::Msm7, 2, 1 << 13);
        let inputs = solve_inputs(&[msm4, msm7]);
        assert_eq!(inputs.len(), 1);
        let observations = &inputs[0].inputs.observations;
        assert_eq!(observations.len(), 2);
        let rough_ms = 100.0 + 512.0 / 1024.0;
        for (observation, fine_ms) in observations.iter().zip([2_f64.powi(-11), 2_f64.powi(-16)]) {
            let expected = (rough_ms + fine_ms) * 1.0e-3 * C_M_S;
            assert!(
                (observation.pseudorange_m - expected).abs() < 1.0e-6,
                "{} {} vs {expected}",
                observation.satellite_id,
                observation.pseudorange_m
            );
        }
    }

    /// SBAS MSM number `n` is broadcast PRN `119 + n` (RTKLIB `decode_msm7`),
    /// so it names slot `n + 19`, not slot `n`. Numbers past the broadcast
    /// window name no SBAS satellite.
    #[test]
    fn msm_sbas_numbers_map_through_the_broadcast_prn_window() {
        assert_eq!(
            msm_satellite_id(GnssSystem::Sbas, 1).map(|sat| sat.to_string()),
            Some("S20".to_string())
        );
        assert_eq!(
            msm_satellite_id(GnssSystem::Sbas, 39).map(|sat| sat.to_string()),
            Some("S58".to_string())
        );
        for number in [0u8, 40, 64, 255] {
            assert_eq!(msm_satellite_id(GnssSystem::Sbas, number), None, "{number}");
        }
    }

    /// QZSS MSM number `n` is PRN `192 + n`, which is already the `Jnn` slot;
    /// the other systems use the number directly, across the whole 64-wide mask.
    #[test]
    fn msm_numbers_of_other_systems_are_the_satellite_number() {
        for system in [
            GnssSystem::Gps,
            GnssSystem::Glonass,
            GnssSystem::Galileo,
            GnssSystem::Qzss,
            GnssSystem::BeiDou,
            GnssSystem::Navic,
        ] {
            for number in 1..=64u8 {
                let sat = msm_satellite_id(system, number)
                    .unwrap_or_else(|| panic!("{system:?} MSM number {number}"));
                assert_eq!(sat.system, system);
                assert_eq!(sat.prn, number);
            }
            assert_eq!(msm_satellite_id(system, 0), None, "{system:?} 0");
        }
    }
}
