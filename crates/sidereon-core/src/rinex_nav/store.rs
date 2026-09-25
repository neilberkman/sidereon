//! Broadcast-store selection and SPP source adapter.

use crate::broadcast::{
    satellite_clock_bias_at_delta_unchecked, satellite_position_ecef_at_tk_unchecked,
    satellite_state, satellite_state_at_deltas_unchecked, satellite_state_cnav,
    satellite_state_cnav_unchecked, satellite_state_unchecked, time_from_reference_delta_s,
    time_from_reference_s, CnavRates, SatelliteState,
};
use crate::constants::{HALF_WEEK_S, SECONDS_PER_WEEK};
use crate::error::{Error, Result as CoreResult};
use crate::glonass;
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::spp::EphemerisSource;
use std::cmp::Ordering;

use super::{
    cnav_ura_nominal_m, gps_minus_utc_at_utc_j2000_s, is_beidou_geo, keplerian_max_dtoe_s,
    parse_nav_file, week_tow_native_j2000_s, BroadcastGroupDelays, BroadcastIssue, BroadcastRecord,
    CnavParameters, GlonassRecord, IonoCorrections, IonosphereFrame, NavDiagnostic, NavHeader,
    NavMessage, NavParseError, SbasRecord, SkippedNavBlock, EPHPOS_STEP_S, GLONASS_MAX_AGE_S,
    J2000_GPS_SECONDS_OF_WEEK, SBAS_MAX_AGE_S,
};
use super::{ephpos_stepped_tk, query_native_time, toe_native_j2000_s};
use crate::astro::time::model::GnssWeekTow;
use crate::astro::time::{ExactEpoch, ExactEpochQuery};

/// Which navigation-message generation a store prefers when a GPS/QZSS
/// satellite has both legacy and CNAV-family records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NavMessagePreference {
    /// Prefer legacy records and use CNAV-family records only as fallback.
    #[default]
    PreferLegacy,
    /// Prefer CNAV-family records and use legacy records as fallback.
    PreferModern,
}

/// A queryable set of parsed broadcast records, usable as an SPP
/// [`EphemerisSource`].
///
/// Records are selected as RTKLIB `seleph`, `selgeph` and `selseph` select them.
/// The candidates are ordered as RTKLIB `uniqnav` orders them - by transmission
/// time, then reference time, in file order among equals - with a record that repeats
/// an earlier one's satellite, reference time and issue (and, for Galileo, the F/NAV
/// clock set) left out. For a query, the record whose reference time is nearest in
/// **continuous** time is taken among those within the system's limit - GPS, QZSS and
/// NavIC 7201 s, Galileo 14400 s, BeiDou 21601 s, GLONASS 1800 s, SBAS 360 s - with a
/// tie going to the later candidate; a Galileo record is not used before its `toe`
/// (RTKLIB's age-of-data rule). The broadcast fit interval is metadata and does not
/// enter the selection. A query outside every limit has no ephemeris, so a stale or
/// wrong-week product cannot silently produce a position.
///
/// [`from_nav`](BroadcastStore::from_nav) keeps the records of the single-frequency
/// messages and applies RTKLIB `satexclude` to the record a query selects: the query
/// has no state when that record's health word is not 0 (QZSS with bit 0, the L1C/A
/// LEX flag, masked as `svh &= 0xFE`), when its URA or SISA variance exceeds 300² m²
/// (`MAX_VAR_EPH`; a URA above 192 m, which RTKLIB's URA index maps to 384 m or more,
/// or a SISA of no accurate prediction), when a GLONASS record is unhealthy by
/// [`GlonassRecord::is_healthy`], or when a CNAV record states no URA prediction. An
/// older healthy record is not used in its place, as RTKLIB does not use one. An
/// issue-matched lookup (IODE, SSR IOD) applies the health part only, since RTKLIB
/// takes the variance of such a state from the correction. [`new`](BroadcastStore::new)
/// keeps records verbatim and excludes nothing, for callers that want their own policy.
pub struct BroadcastStore {
    records: Vec<BroadcastRecord>,
    /// Indices into `records` in RTKLIB `uniqeph` order, repeats left out.
    selection: Vec<usize>,
    glonass: Vec<GlonassRecord>,
    glonass_selection: Vec<usize>,
    sbas: Vec<SbasRecord>,
    sbas_selection: Vec<usize>,
    header: Option<NavHeader>,
    iono: IonoCorrections,
    /// The RINEX 4 `> ION` frames in transmission order.
    iono_frames: Vec<IonosphereFrame>,
    skipped: Vec<SkippedNavBlock>,
    departures: Vec<NavDiagnostic>,
    message_preference: NavMessagePreference,
    /// Whether a selected record RTKLIB `satexclude` would exclude yields no state
    /// (set by [`BroadcastStore::from_nav`]).
    exclude_unusable: bool,
}

impl BroadcastStore {
    /// Build a store from already-parsed Keplerian records, verbatim (no policy
    /// filter, no GLONASS or SBAS records, and no ionosphere coefficients; use
    /// [`from_nav`](Self::from_nav) to capture those).
    pub fn new(records: Vec<BroadcastRecord>) -> CoreResult<Self> {
        for record in &records {
            validate_manual_record(record)?;
        }
        let selection = keplerian_selection_order(&records);
        Ok(Self {
            records,
            selection,
            glonass: Vec::new(),
            glonass_selection: Vec::new(),
            sbas: Vec::new(),
            sbas_selection: Vec::new(),
            header: None,
            iono: IonoCorrections::default(),
            iono_frames: Vec::new(),
            skipped: Vec::new(),
            departures: Vec::new(),
            message_preference: NavMessagePreference::default(),
            exclude_unusable: false,
        })
    }

    /// Parse a RINEX 2.xx/3.xx/4.xx navigation file and keep the records of the
    /// messages used for single-frequency positioning: GPS LNAV, GPS/QZSS CNAV-family,
    /// QZSS LNAV, Galileo I/NAV (and a Galileo record whose data sources name no single
    /// message, which RTKLIB's default selection also uses), BeiDou D1/D2, NavIC LNAV,
    /// GLONASS and SBAS. Health does not filter the records: among a satellite's
    /// records the store selects as RTKLIB `seleph`, `selgeph` and `selseph` do, and a
    /// selected record RTKLIB `satexclude` excludes yields no state (see
    /// [`BroadcastStore`]). The header's broadcast ionosphere coefficients, the RINEX 4
    /// ionosphere frames (see [`iono_corrections_at`](Self::iono_corrections_at)) and
    /// the header are captured.
    ///
    /// Only a header that cannot be read is an error. A block that cannot be read is
    /// left out and reported in [`skipped`](Self::skipped) with its line and reason,
    /// and a departure from the format the reader read through, including a header
    /// record whose values cannot be read, in [`departures`](Self::departures): one
    /// bad record does not cost the file's other records.
    pub fn from_nav(text: &str) -> Result<Self, NavParseError> {
        let file = parse_nav_file(text)?;
        let parse = file.nav_parse();
        let records: Vec<BroadcastRecord> = file
            .keplerian_records()
            .filter(|record| Self::is_default_message(record.message))
            .collect();
        let glonass: Vec<GlonassRecord> = file.glonass_records().collect();
        let sbas: Vec<SbasRecord> = file.sbas_records().collect();
        let mut iono_frames: Vec<IonosphereFrame> = file.ionosphere_frames().cloned().collect();
        super::frames::sort_by_transmission(&mut iono_frames);
        let departures = file.departures();
        let selection = keplerian_selection_order(&records);
        let glonass_selection = glonass_selection_order(&glonass);
        let sbas_selection = sbas_selection_order(&sbas);
        Ok(Self {
            records,
            selection,
            glonass,
            glonass_selection,
            sbas,
            sbas_selection,
            iono: file.header.iono,
            header: Some(file.header),
            iono_frames,
            skipped: parse.skipped,
            departures,
            message_preference: NavMessagePreference::default(),
            exclude_unusable: true,
        })
    }

    /// Set the GPS/QZSS legacy-vs-CNAV selection preference.
    pub fn set_message_preference(&mut self, preference: NavMessagePreference) {
        self.message_preference = preference;
    }

    /// The GPS/QZSS legacy-vs-CNAV selection preference.
    pub const fn message_preference(&self) -> NavMessagePreference {
        self.message_preference
    }

    /// The header of the file the store was read from; `None` for a store built with
    /// [`new`](Self::new).
    pub fn header(&self) -> Option<&NavHeader> {
        self.header.as_ref()
    }

    /// Blocks of the file that could not be read and are not in the store, each with
    /// its line and reason.
    pub fn skipped(&self) -> &[SkippedNavBlock] {
        &self.skipped
    }

    /// Departures from the format the reader read through, each with its line.
    pub fn departures(&self) -> &[NavDiagnostic] {
        &self.departures
    }

    /// The broadcast ionosphere coefficients with no epoch to select by: the header's
    /// sets (GPS, QZSS, BeiDou, NavIC Klobuchar, Galileo NeQuick-G, BeiDou BDGIM), each
    /// replaced by the RINEX 4 `> ION` frame of its system and model transmitted latest,
    /// as [`super::parse_iono_corrections`] reads them. Empty for a store built with
    /// [`new`](Self::new).
    pub fn iono_corrections(&self) -> IonoCorrections {
        super::frames::ionosphere_in_effect(self.iono, &self.iono_frames, None)
    }

    /// The broadcast ionosphere coefficients in effect at `t_j2000_s` (GPS time): the
    /// header's sets, each replaced by the RINEX 4 `> ION` frame of its system and model
    /// transmitted latest at or before the epoch. A set that neither the header nor a
    /// frame transmitted by the epoch states is `None`.
    pub fn iono_corrections_at(&self, t_j2000_s: f64) -> IonoCorrections {
        super::frames::ionosphere_in_effect(self.iono, &self.iono_frames, Some(t_j2000_s))
    }

    /// The held GLONASS records.
    pub fn glonass_records(&self) -> &[GlonassRecord] {
        &self.glonass
    }

    /// The held SBAS records.
    pub fn sbas_records(&self) -> &[SbasRecord] {
        &self.sbas
    }

    /// The GLONASS FDMA frequency channels carried by the held broadcast
    /// records, keyed by satellite PRN/slot, as the records state them. A
    /// channel outside the `-7..=6` allocation is kept, as the observation
    /// header keeps one, and consumers that resolve a carrier check the
    /// allocation; a stated channel too wide for `i8` names no FDMA channel at
    /// all and is left out of the map rather than wrapped onto another one.
    ///
    /// Lets a consumer source the per-satellite channel numbers - needed to
    /// scale the GLONASS ionospheric delay per carrier - from the broadcast
    /// navigation message when an observation file carries no `GLONASS SLOT /
    /// FRQ #` header records. Each GLONASS satellite broadcasts one channel, so
    /// the map has at most one entry per slot. The result keys/values match the
    /// `glonass_slots` layout of [`crate::rinex_obs::ObsHeader`], so a consumer
    /// can use this map directly where an OBS file would otherwise supply one.
    pub fn glonass_frequency_channels(&self) -> std::collections::BTreeMap<u8, i8> {
        self.glonass
            .iter()
            .filter_map(|r| {
                i8::try_from(r.freq_channel)
                    .ok()
                    .map(|channel| (r.satellite_id.prn, channel))
            })
            .collect()
    }

    /// The messages [`from_nav`](Self::from_nav) keeps.
    fn is_default_message(message: NavMessage) -> bool {
        matches!(
            message,
            NavMessage::GpsLnav
                | NavMessage::GpsCnav
                | NavMessage::GpsCnav2
                | NavMessage::QzssLnav
                | NavMessage::QzssCnav
                | NavMessage::QzssCnav2
                | NavMessage::GalileoInav
                | NavMessage::GalileoUnclassified
                | NavMessage::BeidouD1
                | NavMessage::BeidouD2
                | NavMessage::NavicLnav
        )
    }

    /// The held records.
    pub fn records(&self) -> &[BroadcastRecord] {
        &self.records
    }

    /// Select the broadcast record used for `sat` at `t_j2000_s`.
    ///
    /// The selection uses the same message preference, native-system time
    /// mapping, and limits as [`EphemerisSource::position_clock_at_j2000_s`].
    /// Non-Keplerian systems return `None`.
    pub fn select_record_at(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<&BroadcastRecord> {
        let (t_native_s, _, _) = query_native_time(sat, t_j2000_s)?;
        self.select(sat, t_native_s)
    }

    pub(crate) fn select_record_at_epoch_query(
        &self,
        sat: GnssSatelliteId,
        selection_epoch: &ExactEpochQuery,
    ) -> Option<&BroadcastRecord> {
        self.select_exact(sat, selection_epoch)
    }

    /// Broadcast group delay, seconds, of the record selected for `sat` at `t_j2000_s`,
    /// the one [`EphemerisSource::position_clock_at_j2000_s`] evaluates, for the
    /// single-frequency user of its message: GPS, QZSS and NavIC LNAV TGD, Galileo I/NAV
    /// BGD E5b/E1, Galileo F/NAV BGD E5a/E1, BeiDou TGD1, CNAV TGD less ISC L1C/A, and
    /// for GLONASS `-ΔτN / (γ - 1)` from the record's L1/L2 group delay difference
    /// ([`GlonassRecord::single_frequency_group_delay_s`]). The clock this store returns
    /// does not include it, as RTKLIB `satposs` returns none; a single-frequency
    /// pseudorange model subtracts it from that clock, as RTKLIB `pntpos` applies it to
    /// the pseudorange. `None` where no record is selected, for a GLONASS record that
    /// states no delay, and for SBAS.
    pub fn single_frequency_group_delay_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<f64> {
        match sat.system {
            GnssSystem::Glonass => self
                .select_glonass(sat, t_j2000_s)
                .and_then(|(rec, _)| rec.single_frequency_group_delay_s()),
            GnssSystem::Sbas => None,
            _ => self
                .select_record_at(sat, t_j2000_s)
                .map(BroadcastRecord::broadcast_clock_group_delay_s),
        }
    }

    /// Satellite clock offset, seconds, that RTKLIB `satposs` places a pseudorange's
    /// transmission epoch with (`ephclk`): the clock at satellite clock time `t_j2000_s` of
    /// the record selected for `sat` at `selection_j2000_s`, the observation epoch RTKLIB
    /// selects by (`seleph(teph, ...)`, `selgeph`, `selseph`), as `eph2clk` forms it for a
    /// Keplerian record ([`crate::ephemeris::satellite_clock_bias_s`]), `geph2clk` for
    /// GLONASS ([`GlonassRecord::clock_bias_s`]) and `seph2clk` for SBAS, without the
    /// relativistic term or a group delay. `None` where no record is selected.
    pub fn transmit_epoch_clock_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<f64> {
        match sat.system {
            GnssSystem::Glonass => {
                let (rec, _) = self.glonass_selected(sat, t_j2000_s, selection_j2000_s)?;
                Some(rec.clock_bias_s(t_j2000_s))
            }
            GnssSystem::Sbas => {
                // seph2clk: t=ts=timediff(time,seph->t0); for (i=0;i<2;i++)
                // t=ts-(seph->af0+seph->af1*t); return seph->af0+seph->af1*t;
                let (rec, ts) = self.sbas_selected(sat, t_j2000_s, selection_j2000_s)?;
                let mut t = ts;
                for _ in 0..2 {
                    t = ts - (rec.af0_s + rec.af1_s_s * t);
                }
                Some(rec.af0_s + rec.af1_s_s * t)
            }
            _ => {
                let (rec, sow, _) = self.keplerian_selected(sat, t_j2000_s, selection_j2000_s)?;
                Some(crate::broadcast::satellite_clock_bias_s_unchecked(
                    &rec.clock, sow,
                ))
            }
        }
    }

    /// Velocity, metres per second, of the record selected for `sat` at `t_j2000_s`, as
    /// RTKLIB `ephpos` forms it: the difference of that record's positions at the epoch and
    /// [`EPHPOS_STEP_S`] later over the step, the step added to the record's reduced time
    /// (`tk` for a Keplerian record, time from the reference epoch for GLONASS and SBAS) as
    /// RTKLIB adds it to its exact `gtime_t`. Differencing the source's positions instead
    /// would select a record on each side and blend two records across a record change.
    /// `None` where no record is selected or the system has no broadcast model here.
    pub(crate) fn selected_record_velocity(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<[f64; 3]> {
        self.selected_record_velocity_at(sat, t_j2000_s, t_j2000_s)
    }

    /// [`Self::selected_record_velocity`] of the record selected at `selection_j2000_s`,
    /// evaluated at `t_j2000_s`, as RTKLIB `satposs` selects it by the observation epoch.
    pub(crate) fn selected_record_velocity_at(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<[f64; 3]> {
        match sat.system {
            GnssSystem::Glonass => {
                let (rec, tk) = self.glonass_selected(sat, t_j2000_s, selection_j2000_s)?;
                glonass_record_velocity(rec, tk)
            }
            GnssSystem::Sbas => {
                let (rec, t) = self.sbas_selected(sat, t_j2000_s, selection_j2000_s)?;
                let start = rec.position_at(t);
                let end = rec.position_at(ephpos_stepped_tk(t));
                Some(difference_velocity(start, end))
            }
            _ => {
                let (rec, _, _) = self.keplerian_selected(sat, t_j2000_s, selection_j2000_s)?;
                keplerian_record_velocity(rec, sat, t_j2000_s)
            }
        }
    }

    /// Velocity of the Keplerian GPS LNAV record with issue byte `iode` valid at
    /// `t_j2000_s`: its positions at `t_j2000_s` and 1 ms later, differenced, as RTKLIB
    /// `ephpos` forms it for an IODE-selected record.
    pub(crate) fn iode_record_velocity(
        &self,
        sat: GnssSatelliteId,
        iode: u8,
        t_j2000_s: f64,
    ) -> Option<[f64; 3]> {
        self.iode_record_velocity_at(sat, iode, t_j2000_s, t_j2000_s)
    }

    /// [`Self::iode_record_velocity`] of the record selected at `selection_j2000_s`.
    pub(crate) fn iode_record_velocity_at(
        &self,
        sat: GnssSatelliteId,
        iode: u8,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<[f64; 3]> {
        let rec = self.select_by_iode_at(sat, iode, selection_j2000_s)?;
        keplerian_record_velocity(rec, sat, t_j2000_s)
    }

    /// Select the record for `sat` with a matching GPS LNAV issue byte at `t`: the first
    /// candidate in selection order within the system's limit, as RTKLIB `seleph` returns
    /// for an IODE.
    pub fn select_by_iode_at(
        &self,
        sat: GnssSatelliteId,
        iode: u8,
        t_j2000_s: f64,
    ) -> Option<&BroadcastRecord> {
        let (t_native_s, _, _) = query_native_time(sat, t_j2000_s)?;
        self.first_valid(sat, t_native_s, |r| {
            r.issue_of_data
                == Some(BroadcastIssue {
                    issue: u32::from(iode),
                    message: NavMessage::GpsLnav,
                })
        })
    }

    pub(crate) fn select_by_iode_at_epoch_query(
        &self,
        sat: GnssSatelliteId,
        iode: u8,
        selection_epoch: &ExactEpochQuery,
    ) -> Option<&BroadcastRecord> {
        self.first_valid_exact(sat, selection_epoch, |record| {
            record.issue_of_data
                == Some(BroadcastIssue {
                    issue: u32::from(iode),
                    message: NavMessage::GpsLnav,
                })
        })
    }

    /// Evaluate a matching issue-specific broadcast record at `t`.
    pub fn state_by_iode_at(
        &self,
        sat: GnssSatelliteId,
        iode: u8,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        let (_, sow, is_geo) = query_native_time(sat, t_j2000_s)?;
        let rec = self.select_by_iode_at(sat, iode, t_j2000_s)?;
        let state = evaluate_record_unchecked(rec, sow, is_geo);
        let position = state.orbit.position().ok()?;
        Some((position.as_array(), satposs_clock_s(&state)))
    }

    /// [`Self::state_by_iode_at`] with the single-frequency group delay of the same
    /// record.
    pub(crate) fn state_group_delay_by_iode_at(
        &self,
        sat: GnssSatelliteId,
        iode: u8,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        self.state_group_delay_by_iode_selected_at(sat, iode, t_j2000_s, t_j2000_s)
    }

    /// [`Self::state_group_delay_by_iode_at`] at `t_j2000_s` of the record selected at
    /// `selection_j2000_s`, as RTKLIB `satpos_sbas` selects it by the observation epoch.
    pub(crate) fn state_group_delay_by_iode_selected_at(
        &self,
        sat: GnssSatelliteId,
        iode: u8,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        let (_, sow, is_geo) = query_native_time(sat, t_j2000_s)?;
        let rec = self.select_by_iode_at(sat, iode, selection_j2000_s)?;
        let state = evaluate_record_unchecked(rec, sow, is_geo);
        let position = state.orbit.position().ok()?;
        Some((
            position.as_array(),
            satposs_clock_s(&state),
            Some(rec.broadcast_clock_group_delay_s()),
        ))
    }

    pub(crate) fn state_group_delay_by_iode_selected_at_epoch_query(
        &self,
        sat: GnssSatelliteId,
        iode: u8,
        epoch: &crate::astro::time::ExactEpochQuery,
        selection_epoch: &ExactEpochQuery,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        let rec = self.select_by_iode_at_epoch_query(sat, iode, selection_epoch)?;
        let (_, _, is_geo) = super::query_native_exact_time(sat, epoch.epoch())?;
        let state = evaluate_record_at_epoch_query(rec, epoch, is_geo)?;
        Some((
            state.orbit.position().ok()?.as_array(),
            satposs_clock_s(&state),
            Some(rec.broadcast_clock_group_delay_s()),
        ))
    }

    /// Select the BeiDou record for `sat` of message `nav_message` whose IOD, as IGS SSR
    /// v1.00 (IDF012) defines it for BDS, `mod(toe/720, 240)` with `toe` the BDT seconds of
    /// week, equals `iod`: the first candidate in selection order within the BeiDou
    /// limit. SSR orbit corrections name a BeiDou record by this IOD, not by its AODE.
    pub(crate) fn select_by_beidou_ssr_iod_at(
        &self,
        sat: GnssSatelliteId,
        iod: u32,
        nav_message: NavMessage,
        t_j2000_s: f64,
    ) -> Option<&BroadcastRecord> {
        if sat.system != GnssSystem::BeiDou {
            return None;
        }
        let (t_native_s, _, _) = query_native_time(sat, t_j2000_s)?;
        self.first_valid(sat, t_native_s, |r| {
            r.message == nav_message && beidou_ssr_iod(r) == Some(iod)
        })
    }

    /// The first valid record of `nav_message` for `sat` at `t_j2000_s`, in selection
    /// order, whose issue's `bits` least significant bits equal `low_bits`: how an IGS
    /// SSR Galileo correction names its I/NAV record by the eight low bits of IODnav
    /// (IGS SSR v1.00, IDF012).
    pub(crate) fn select_by_issue_low_bits_at(
        &self,
        sat: GnssSatelliteId,
        low_bits: u32,
        bits: u32,
        nav_message: NavMessage,
        t_j2000_s: f64,
    ) -> Option<&BroadcastRecord> {
        let mask = (1u32 << bits) - 1;
        let (t_native_s, _, _) = query_native_time(sat, t_j2000_s)?;
        self.first_valid(sat, t_native_s, |r| {
            r.message == nav_message
                && r.issue_of_data.is_some_and(|issue| {
                    issue.message == nav_message && issue.issue & mask == low_bits
                })
        })
    }

    pub(crate) fn select_by_issue_low_bits_at_epoch_query(
        &self,
        sat: GnssSatelliteId,
        low_bits: u32,
        bits: u32,
        nav_message: NavMessage,
        selection_epoch: &ExactEpochQuery,
    ) -> Option<&BroadcastRecord> {
        let mask = 1_u32.checked_shl(bits)?.checked_sub(1)?;
        self.first_valid_exact(sat, selection_epoch, |record| {
            record.message == nav_message
                && record.issue_of_data.is_some_and(|issue| {
                    issue.message == nav_message && issue.issue & mask == low_bits
                })
        })
    }

    pub(crate) fn select_by_issue_at_epoch_query(
        &self,
        sat: GnssSatelliteId,
        issue: BroadcastIssue,
        nav_message: NavMessage,
        selection_epoch: &ExactEpochQuery,
    ) -> Option<&BroadcastRecord> {
        if issue.message != nav_message {
            return None;
        }
        self.first_valid_exact(sat, selection_epoch, |record| {
            record.message == nav_message && record.issue_of_data == Some(issue)
        })
    }

    pub(crate) fn select_by_beidou_ssr_iod_at_epoch_query(
        &self,
        sat: GnssSatelliteId,
        iod: u32,
        nav_message: NavMessage,
        selection_epoch: &ExactEpochQuery,
    ) -> Option<&BroadcastRecord> {
        if sat.system != GnssSystem::BeiDou {
            return None;
        }
        self.first_valid_exact(sat, selection_epoch, |record| {
            record.message == nav_message && beidou_ssr_iod(record) == Some(iod)
        })
    }

    /// Position, velocity and clock of the GLONASS record for `sat` whose `tb`, the 15-min
    /// index of its reference epoch in UTC + 3 h, equals `iode`, as RTKLIB `satpos_ssr`
    /// forms them for a GLONASS SSR correction: `selgeph` by that issue (RTKLIB `readrnx`
    /// forms a GLONASS record's IODE as `tb`), then `geph2pos` at `t_j2000_s` and 1 ms
    /// later. The clock is `geph2pos`'s, `-TauN + GammaN·tk` with `tk` not iterated, and
    /// carries no relativistic term, which RTKLIB adds none of for GLONASS. The record's
    /// reference epoch lies within [`GLONASS_MAX_AGE_S`] of `selection_j2000_s`, the epoch
    /// `selgeph` selects at (`teph`), and among records of that `tb` the first in selection
    /// order is taken, as `selgeph` returns for an issue.
    pub(crate) fn glonass_ssr_state(
        &self,
        sat: GnssSatelliteId,
        iode: u32,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<([f64; 3], [f64; 3], f64)> {
        let rec = self
            .glonass_selection
            .iter()
            .map(|&index| &self.glonass[index])
            .find(|r| {
                r.satellite_id == sat
                    && glonass_tb(r) == Some(iode)
                    && (selection_j2000_s - r.toe_gpst_j2000_s()).abs() <= GLONASS_MAX_AGE_S
            })?;
        if self.exclude_unusable && !rec.is_healthy() {
            return None;
        }
        let tk = t_j2000_s - rec.toe_gpst_j2000_s();
        let state0 = glonass_state0(rec);
        let start = glonass::propagate(state0, rec.acc_m_s2, tk).ok()?;
        let end = glonass::propagate(state0, rec.acc_m_s2, ephpos_stepped_tk(tk)).ok()?;
        let velocity =
            difference_velocity([start[0], start[1], start[2]], [end[0], end[1], end[2]]);
        let clock = glonass::position_clock_offset_s(rec.clk_bias, rec.gamma_n, tk);
        Some(([start[0], start[1], start[2]], velocity, clock))
    }

    pub(crate) fn glonass_ssr_state_at_query(
        &self,
        sat: GnssSatelliteId,
        iode: u32,
        epoch: &crate::astro::time::ExactEpochQuery,
        selection_epoch: &ExactEpochQuery,
    ) -> Option<([f64; 3], [f64; 3], f64)> {
        let rec = self.select_glonass_by_issue_at_epoch_query(sat, iode, selection_epoch)?;
        if self.exclude_unusable && !rec.is_healthy() {
            return None;
        }
        let reference = exact_glonass_gpst_epoch_query(rec)?;
        let tk = epoch.seconds_since_query(&reference);
        let state0 = glonass_state0(rec);
        let start = glonass::propagate(state0, rec.acc_m_s2, tk).ok()?;
        let end = glonass::propagate(state0, rec.acc_m_s2, ephpos_stepped_tk(tk)).ok()?;
        let velocity =
            difference_velocity([start[0], start[1], start[2]], [end[0], end[1], end[2]]);
        let clock = glonass::position_clock_offset_s(rec.clk_bias, rec.gamma_n, tk);
        Some(([start[0], start[1], start[2]], velocity, clock))
    }

    /// Keep only the records matching a predicate (e.g. a custom message/health
    /// policy on a store built with [`new`](BroadcastStore::new)).
    pub fn retain(&mut self, keep: impl FnMut(&BroadcastRecord) -> bool) {
        self.records.retain(keep);
        self.selection = keplerian_selection_order(&self.records);
    }

    /// The record for `sat` RTKLIB `seleph` selects at `t_native_s`: among the candidates
    /// in selection order within the system's limit (a Galileo record only after its
    /// `toe`), the one whose `toe` is nearest, a tie going to the later candidate. A
    /// candidate of the preferred message family (see [`NavMessagePreference`]) is taken
    /// over one of the other family.
    fn select(&self, sat: GnssSatelliteId, t_native_s: f64) -> Option<&BroadcastRecord> {
        let tmax = keplerian_max_dtoe_s(sat.system);
        let mut preferred: Option<(usize, f64)> = None;
        let mut fallback: Option<(usize, f64)> = None;
        for &index in &self.selection {
            let record = &self.records[index];
            if record.satellite_id != sat {
                continue;
            }
            let Some(t) = distance_within(record, t_native_s, tmax) else {
                continue;
            };
            let slot = if self.is_preferred_family(record) {
                &mut preferred
            } else {
                &mut fallback
            };
            let better = match *slot {
                None => true,
                Some((current, tmin)) => {
                    t < tmin
                        || (t == tmin
                            && cnav_tie_rank(record.message)
                                <= cnav_tie_rank(self.records[current].message))
                }
            };
            if better {
                *slot = Some((index, t));
            }
        }
        let record = &self.records[preferred.or(fallback)?.0];
        (!self.exclude_unusable || !keplerian_excluded(record)).then_some(record)
    }

    fn select_exact(
        &self,
        sat: GnssSatelliteId,
        selection_epoch: &ExactEpochQuery,
    ) -> Option<&BroadcastRecord> {
        let mut preferred: Option<usize> = None;
        let mut fallback: Option<usize> = None;
        for &record_index in &self.selection {
            let record = &self.records[record_index];
            if record.satellite_id != sat || !record_within_exact_limit(record, selection_epoch) {
                continue;
            }
            let slot = if self.is_preferred_family(record) {
                &mut preferred
            } else {
                &mut fallback
            };
            let better = match *slot {
                None => true,
                Some(current_index) => {
                    let current = &self.records[current_index];
                    let candidate_epoch = exact_week_tow_epoch(record.toe)?.query();
                    let current_epoch = exact_week_tow_epoch(current.toe)?.query();
                    match selection_epoch.compare_distance_to(&candidate_epoch, &current_epoch) {
                        Ordering::Less => true,
                        Ordering::Equal => {
                            cnav_tie_rank(record.message) <= cnav_tie_rank(current.message)
                        }
                        Ordering::Greater => false,
                    }
                }
            };
            if better {
                *slot = Some(record_index);
            }
        }
        let record = &self.records[preferred.or(fallback)?];
        (!self.exclude_unusable || !keplerian_excluded(record)).then_some(record)
    }

    /// The first candidate for `sat` in selection order within the system's limit (a
    /// Galileo record only after its `toe`) that `matches`, as RTKLIB `seleph` returns a
    /// record for a given issue.
    fn first_valid(
        &self,
        sat: GnssSatelliteId,
        t_native_s: f64,
        matches: impl Fn(&BroadcastRecord) -> bool,
    ) -> Option<&BroadcastRecord> {
        let tmax = keplerian_max_dtoe_s(sat.system);
        let record = self
            .selection
            .iter()
            .map(|&index| &self.records[index])
            .find(|r| {
                r.satellite_id == sat
                    && matches(r)
                    && distance_within(r, t_native_s, tmax).is_some()
            })?;
        (!self.exclude_unusable || !keplerian_health_excluded(record)).then_some(record)
    }

    fn first_valid_exact(
        &self,
        sat: GnssSatelliteId,
        selection_epoch: &ExactEpochQuery,
        matches: impl Fn(&BroadcastRecord) -> bool,
    ) -> Option<&BroadcastRecord> {
        let record = self
            .selection
            .iter()
            .map(|&record_index| &self.records[record_index])
            .find(|record| {
                record.satellite_id == sat
                    && matches(record)
                    && record_within_exact_limit(record, selection_epoch)
            })?;
        (!self.exclude_unusable || !keplerian_health_excluded(record)).then_some(record)
    }

    fn is_preferred_family(&self, record: &BroadcastRecord) -> bool {
        if !matches!(
            record.satellite_id.system,
            GnssSystem::Gps | GnssSystem::Qzss
        ) {
            return true;
        }
        match self.message_preference {
            NavMessagePreference::PreferLegacy => !record.message.is_cnav_family(),
            NavMessagePreference::PreferModern => record.message.is_cnav_family(),
        }
    }

    /// Select the broadcast record matching a specific issue and message at the query
    /// epoch: the first candidate in selection order within the system's limit, as
    /// RTKLIB `seleph` returns for an issue.
    pub fn select_by_issue_at(
        &self,
        sat: GnssSatelliteId,
        issue: BroadcastIssue,
        nav_message: NavMessage,
        t_j2000_s: f64,
    ) -> Option<&BroadcastRecord> {
        if issue.message != nav_message {
            return None;
        }
        let (t_native_s, _, _) = query_native_time(sat, t_j2000_s)?;
        self.first_valid(sat, t_native_s, |r| {
            r.message == nav_message && r.issue_of_data == Some(issue)
        })
    }

    /// The GLONASS record for `sat` RTKLIB `selgeph` selects at the GPS-time query
    /// `t_j2000_s`: the nearest reference epoch within [`GLONASS_MAX_AGE_S`], a tie going
    /// to the later candidate in selection order, with `tk` = query − the record's
    /// reference epoch in GPS time.
    fn select_glonass(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<(&GlonassRecord, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for &index in &self.glonass_selection {
            let record = &self.glonass[index];
            if record.satellite_id != sat {
                continue;
            }
            let t = (record.toe_gpst_j2000_s() - t_j2000_s).abs();
            if t > GLONASS_MAX_AGE_S {
                continue;
            }
            if best.is_none_or(|(_, tmin)| t <= tmin) {
                best = Some((index, t));
            }
        }
        let rec = &self.glonass[best?.0];
        if self.exclude_unusable && !rec.is_healthy() {
            return None;
        }
        Some((rec, t_j2000_s - rec.toe_gpst_j2000_s()))
    }

    fn select_glonass_at_epoch_query(
        &self,
        sat: GnssSatelliteId,
        selection_epoch: &ExactEpochQuery,
    ) -> Option<&GlonassRecord> {
        let mut best: Option<usize> = None;
        for &record_index in &self.glonass_selection {
            let record = &self.glonass[record_index];
            if record.satellite_id != sat
                || !glonass_record_within_exact_limit(record, selection_epoch, GLONASS_MAX_AGE_S)
            {
                continue;
            }
            let better = match best {
                None => true,
                Some(current_index) => {
                    let candidate_epoch = exact_glonass_gpst_epoch_query(record)?;
                    let current_epoch =
                        exact_glonass_gpst_epoch_query(&self.glonass[current_index])?;
                    selection_epoch.compare_distance_to(&candidate_epoch, &current_epoch)
                        != Ordering::Greater
                }
            };
            if better {
                best = Some(record_index);
            }
        }
        let record = &self.glonass[best?];
        if self.exclude_unusable && !record.is_healthy() {
            return None;
        }
        Some(record)
    }

    fn select_glonass_by_issue_at_epoch_query(
        &self,
        sat: GnssSatelliteId,
        iode: u32,
        selection_epoch: &ExactEpochQuery,
    ) -> Option<&GlonassRecord> {
        self.glonass_selection
            .iter()
            .map(|&record_index| &self.glonass[record_index])
            .find(|record| {
                record.satellite_id == sat
                    && glonass_tb(record) == Some(iode)
                    && glonass_record_within_exact_limit(record, selection_epoch, GLONASS_MAX_AGE_S)
            })
            .filter(|record| !self.exclude_unusable || record.is_healthy())
    }

    /// The SBAS record for `sat` RTKLIB `selseph` selects at the GPS-time query
    /// `t_j2000_s`: the nearest reference epoch within [`SBAS_MAX_AGE_S`], a tie going to
    /// the later candidate in selection order, with `t` = query − `t0`.
    fn select_sbas(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<(&SbasRecord, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for &index in &self.sbas_selection {
            let record = &self.sbas[index];
            if record.satellite_id != sat {
                continue;
            }
            let t = (record.t0_j2000_s() - t_j2000_s).abs();
            if t > SBAS_MAX_AGE_S {
                continue;
            }
            if best.is_none_or(|(_, tmin)| t <= tmin) {
                best = Some((index, t));
            }
        }
        let rec = &self.sbas[best?.0];
        if self.exclude_unusable && sbas_excluded(rec) {
            return None;
        }
        Some((rec, t_j2000_s - rec.t0_j2000_s()))
    }

    /// SBAS broadcast state named by an SSR orbit issue, selecting among only the
    /// matching records at the observation epoch and evaluating the selected record at
    /// the transmit epoch. IGS SSR names `IODN`; native RTCM SSR names `t0 mod 8192 s`
    /// in 16-second units. RINEX SBAS navigation records do not carry the native IOD
    /// CRC, so that field is retained in the SSR correction but cannot be cross-checked
    /// against this broadcast source.
    pub(crate) fn sbas_ssr_state(
        &self,
        sat: GnssSatelliteId,
        issue: u32,
        igs_ssr: bool,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<([f64; 3], [f64; 3], f64)> {
        if sat.system != GnssSystem::Sbas
            || !t_j2000_s.is_finite()
            || !selection_j2000_s.is_finite()
        {
            return None;
        }
        let mut best: Option<(usize, f64)> = None;
        for &index in &self.sbas_selection {
            let record = &self.sbas[index];
            if record.satellite_id != sat {
                continue;
            }
            let age = (record.t0_j2000_s() - selection_j2000_s).abs();
            if age > SBAS_MAX_AGE_S || !sbas_issue_matches(record, issue, igs_ssr) {
                continue;
            }
            if best.is_none_or(|(_, best_age)| age <= best_age) {
                best = Some((index, age));
            }
        }
        let record = &self.sbas[best?.0];
        if self.exclude_unusable && sbas_excluded(record) {
            return None;
        }
        let t = t_j2000_s - record.t0_j2000_s();
        let position = record.position_at(t);
        let next_position = record.position_at(ephpos_stepped_tk(t));
        let velocity = [
            (next_position[0] - position[0]) / EPHPOS_STEP_S,
            (next_position[1] - position[1]) / EPHPOS_STEP_S,
            (next_position[2] - position[2]) / EPHPOS_STEP_S,
        ];
        let clock = record.af0_s + record.af1_s_s * t;
        Some((position, velocity, clock))
    }

    pub(crate) fn sbas_ssr_state_at_query(
        &self,
        sat: GnssSatelliteId,
        issue: u32,
        igs_ssr: bool,
        epoch: &ExactEpochQuery,
        selection_epoch: &ExactEpochQuery,
    ) -> Option<([f64; 3], [f64; 3], f64)> {
        if sat.system != GnssSystem::Sbas {
            return None;
        }
        let mut best: Option<usize> = None;
        for &record_index in &self.sbas_selection {
            let record = &self.sbas[record_index];
            if record.satellite_id != sat
                || !sbas_record_within_exact_limit(record, selection_epoch, SBAS_MAX_AGE_S)
                || !sbas_issue_matches_exact(record, issue, igs_ssr)
            {
                continue;
            }
            let better = match best {
                None => true,
                Some(current_index) => {
                    let candidate_epoch = exact_sbas_epoch_query(record)?;
                    let current_epoch = exact_sbas_epoch_query(&self.sbas[current_index])?;
                    selection_epoch.compare_distance_to(&candidate_epoch, &current_epoch)
                        != Ordering::Greater
                }
            };
            if better {
                best = Some(record_index);
            }
        }
        let record = &self.sbas[best?];
        if self.exclude_unusable && sbas_excluded(record) {
            return None;
        }
        let reference = exact_sbas_epoch_query(record)?;
        let elapsed_s = epoch.seconds_since_query(&reference);
        let stepped_s = epoch
            .clone()
            .checked_add_binary_seconds(EPHPOS_STEP_S)?
            .seconds_since_query(&reference);
        let position = record.position_at(elapsed_s);
        let next_position = record.position_at(stepped_s);
        let velocity = difference_velocity(position, next_position);
        let clock = record.af0_s + record.af1_s_s * elapsed_s;
        Some((position, velocity, clock))
    }

    fn select_sbas_at_epoch_query(
        &self,
        sat: GnssSatelliteId,
        selection_epoch: &ExactEpochQuery,
    ) -> Option<&SbasRecord> {
        let mut best: Option<usize> = None;
        for &record_index in &self.sbas_selection {
            let record = &self.sbas[record_index];
            if record.satellite_id != sat
                || !sbas_record_within_exact_limit(record, selection_epoch, SBAS_MAX_AGE_S)
            {
                continue;
            }
            let better = match best {
                None => true,
                Some(current_index) => {
                    let candidate_epoch = exact_sbas_epoch_query(record)?;
                    let current_epoch = exact_sbas_epoch_query(&self.sbas[current_index])?;
                    selection_epoch.compare_distance_to(&candidate_epoch, &current_epoch)
                        != Ordering::Greater
                }
            };
            if better {
                best = Some(record_index);
            }
        }
        let record = &self.sbas[best?];
        if self.exclude_unusable && sbas_excluded(record) {
            return None;
        }
        Some(record)
    }
}

fn sbas_issue_matches(record: &SbasRecord, issue: u32, igs_ssr: bool) -> bool {
    if igs_ssr {
        return issue <= u32::from(u8::MAX) && record.iodn == Some(f64::from(issue));
    }
    if issue > 0x1FF {
        return false;
    }
    let t0_sow = (record.t0_j2000_s().rem_euclid(SECONDS_PER_WEEK) + J2000_GPS_SECONDS_OF_WEEK)
        .rem_euclid(SECONDS_PER_WEEK);
    (t0_sow / 16.0).floor() as u32 % 512 == issue
}

fn sbas_issue_matches_exact(record: &SbasRecord, issue: u32, igs_ssr: bool) -> bool {
    if igs_ssr {
        return issue <= u32::from(u8::MAX) && record.iodn == Some(f64::from(issue));
    }
    if issue > 0x1FF {
        return false;
    }
    let Some(epoch) = ExactEpoch::from_civil(
        record.epoch.year,
        i32::from(record.epoch.month),
        i32::from(record.epoch.day),
        i32::from(record.epoch.hour),
        i32::from(record.epoch.minute),
        record.epoch.second,
    ) else {
        return false;
    };
    let before_whole_second = epoch.attoseconds() == 0 && epoch.sub_attosecond().0 < 0;
    let whole_sow = (epoch.whole_seconds().rem_euclid(604_800) - i64::from(before_whole_second)
        + J2000_GPS_SECONDS_OF_WEEK as i64)
        .rem_euclid(604_800);
    (whole_sow / 16) as u32 % 512 == issue
}

/// RTKLIB `MAX_VAR_EPH`: the largest ephemeris variance `satexclude` accepts, m².
const MAX_VAR_EPH_M2: f64 = 300.0 * 300.0;

/// RTKLIB `rinex.c` `ura_eph`: the URA values (m) of URA indices 0-14.
const URA_EPH_M: [f64; 15] = [
    2.4, 3.4, 4.85, 6.85, 9.65, 13.65, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0, 1536.0, 3072.0,
    6144.0,
];

/// RTKLIB `ephemeris.c` `STD_GAL_NAPA`: the Galileo error (m) for no accurate prediction.
const STD_GAL_NAPA_M: f64 = 500.0;

/// RTKLIB `ephemeris.c` `ERREPH_GLO`: the GLONASS broadcast ephemeris error (m).
const ERREPH_GLO_M: f64 = 5.0;

/// RTKLIB `uraindex`: the first URA index whose value is at least `value`, 15 past the
/// table.
fn ura_index(value: f64) -> usize {
    URA_EPH_M
        .iter()
        .position(|&ura| ura >= value)
        .unwrap_or(URA_EPH_M.len())
}

/// RTKLIB `var_uraeph` for a URA index, m² (6144² past the table).
pub(crate) fn ura_variance_m2(index: usize) -> f64 {
    let ura = URA_EPH_M.get(index).copied().unwrap_or(6144.0);
    ura * ura
}

/// RTKLIB `var_uraeph` of a Galileo SISA (m) through `sisa_index`, m²: a SISA below 0 or
/// above 6 m is no accurate prediction.
fn sisa_variance_m2(value: f64) -> f64 {
    let std = if !(0.0..=6.0).contains(&value) {
        STD_GAL_NAPA_M
    } else {
        let index = if value <= 0.49 {
            (value / 0.01).round()
        } else if value <= 0.98 {
            ((value - 0.5) / 0.02).round() + 50.0
        } else if value <= 1.96 {
            ((value - 1.0) / 0.04).round() + 75.0
        } else {
            ((value - 2.0) / 0.16).round() + 100.0
        };
        if index <= 49.0 {
            index * 0.01
        } else if index <= 74.0 {
            0.5 + (index - 50.0) * 0.02
        } else if index <= 99.0 {
            1.0 + (index - 75.0) * 0.04
        } else if index <= 125.0 {
            2.0 + (index - 100.0) * 0.16
        } else {
            STD_GAL_NAPA_M
        }
    };
    std * std
}

/// RTKLIB `var_uraeph` of a Keplerian record, m²: the Galileo SISA through `sisa_index`,
/// the URA index of the stated accuracy through `uraindex` otherwise. A blank accuracy is
/// read as 0, as RTKLIB `readrnx` reads it.
fn keplerian_variance_m2(record: &BroadcastRecord) -> f64 {
    let accuracy_m = record.sv_accuracy_m.unwrap_or(0.0);
    if record.satellite_id.system == GnssSystem::Galileo {
        sisa_variance_m2(accuracy_m)
    } else {
        ura_variance_m2(ura_index(accuracy_m))
    }
}

/// RTKLIB `var_uraeph(SYS_SBS, sva)` of an SBAS record, m², its URA read through
/// `uraindex`, a blank URA as 0.
fn sbas_variance_m2(record: &SbasRecord) -> f64 {
    ura_variance_m2(ura_index(record.ura_m.unwrap_or(0.0)))
}

/// The health part of RTKLIB `satexclude` for a Keplerian record: the health word, read
/// as RTKLIB's `(int)` reads it, is not 0, with QZSS masked by `0xFE`.
fn keplerian_health_excluded(record: &BroadcastRecord) -> bool {
    let mut svh = record.sv_health as i64;
    if record.satellite_id.system == GnssSystem::Qzss {
        svh &= 0xFE;
    }
    svh != 0
}

/// RTKLIB `satexclude` for a Keplerian record: the health part, then the ephemeris
/// variance against `MAX_VAR_EPH`, from the SISA for Galileo and the URA index
/// otherwise. A CNAV record with no URA prediction is excluded (RTKLIB has no CNAV
/// record); an accuracy the record leaves blank excludes nothing, as RTKLIB reads it as
/// 0.
fn keplerian_excluded(record: &BroadcastRecord) -> bool {
    if keplerian_health_excluded(record) {
        return true;
    }
    if record.message.is_cnav_family()
        && record
            .cnav
            .is_none_or(|cnav| cnav_ura_nominal_m(cnav.ura_ed_index).is_none())
    {
        return true;
    }
    let Some(accuracy_m) = record.sv_accuracy_m else {
        return false;
    };
    let variance = if record.satellite_id.system == GnssSystem::Galileo {
        sisa_variance_m2(accuracy_m)
    } else {
        ura_variance_m2(ura_index(accuracy_m))
    };
    variance > MAX_VAR_EPH_M2
}

/// RTKLIB `satexclude` for an SBAS record: health not 0, or the URA variance above
/// `MAX_VAR_EPH`.
fn sbas_excluded(record: &SbasRecord) -> bool {
    record.health as i64 != 0
        || record
            .ura_m
            .is_some_and(|ura| ura_variance_m2(ura_index(ura)) > MAX_VAR_EPH_M2)
}

/// `|toe - t|` for a candidate within `tmax` of the query, or `None`: RTKLIB `seleph`
/// skips a Galileo record whose `toe` is not before the query (`timediff(toe, time) >=
/// 0`, "AOD<=0") and any record farther than `tmax`.
fn distance_within(record: &BroadcastRecord, t_native_s: f64, tmax: f64) -> Option<f64> {
    let toe = toe_native_j2000_s(record);
    if record.satellite_id.system == GnssSystem::Galileo && toe - t_native_s >= 0.0 {
        return None;
    }
    let t = (toe - t_native_s).abs();
    (t <= tmax).then_some(t)
}

/// Move `t` by a week to lie within half a week of `reference`, as RTKLIB `adjweek` does.
fn adjust_week(t: f64, reference: f64) -> f64 {
    let dt = t - reference;
    if dt < -HALF_WEEK_S {
        t + SECONDS_PER_WEEK
    } else if dt > HALF_WEEK_S {
        t - SECONDS_PER_WEEK
    } else {
        t
    }
}

/// Move `t` by a day to lie within half a day of `reference`, as RTKLIB `adjday` does.
fn adjust_day(t: f64, reference: f64) -> f64 {
    let dt = t - reference;
    if dt < -43_200.0 {
        t + 86_400.0
    } else if dt > 43_200.0 {
        t - 86_400.0
    } else {
        t
    }
}

/// A Keplerian record's transmission time as RTKLIB `decode_eph` forms `ttr`: the
/// stated transmission time of message (0 when blank) in the stated week, moved to within
/// half a week of `toc`, in continuous seconds of the record's scale.
fn transmission_key(record: &BroadcastRecord) -> f64 {
    let toc = week_tow_native_j2000_s(record.toc);
    let sow = record.transmission_time_sow().unwrap_or(0.0);
    let ttr = week_tow_native_j2000_s(GnssWeekTow {
        system: record.toc.system,
        week: record.week,
        tow_s: sow,
    });
    adjust_week(ttr, toc)
}

/// Whether a Galileo record belongs to RTKLIB's F/NAV clock set (`code & (bit 8 | bit
/// 1)`), which `uniqeph` keeps apart from the I/NAV set.
fn galileo_fnav_set(record: &BroadcastRecord) -> bool {
    match record.galileo_data_sources() {
        Some(word) => word & ((1 << 8) | (1 << 1)) != 0,
        None => record.message == NavMessage::GalileoFnav,
    }
}

/// Indices of `records` in RTKLIB `uniqeph` order: sorted by transmission time, then
/// `toe`, then satellite (file order among equals; RTKLIB's `qsort` leaves that order
/// unspecified), with a record that repeats the last kept record's satellite, `toe` and
/// issue (and Galileo clock set and message) left out.
fn keplerian_selection_order(records: &[BroadcastRecord]) -> Vec<usize> {
    let keys: Vec<(f64, f64)> = records
        .iter()
        .map(|r| (transmission_key(r), toe_native_j2000_s(r)))
        .collect();
    let mut order: Vec<usize> = (0..records.len()).collect();
    order.sort_by(|&a, &b| {
        keys[a]
            .0
            .total_cmp(&keys[b].0)
            .then(keys[a].1.total_cmp(&keys[b].1))
            .then(records[a].satellite_id.cmp(&records[b].satellite_id))
            .then(galileo_fnav_set(&records[a]).cmp(&galileo_fnav_set(&records[b])))
    });
    let mut kept: Vec<usize> = Vec::with_capacity(order.len());
    for index in order {
        if let Some(&last) = kept.last() {
            let (a, b) = (&records[last], &records[index]);
            let repeat = a.satellite_id == b.satellite_id
                && keys[last].1 == keys[index].1
                && a.issue_of_data.map(|issue| issue.issue)
                    == b.issue_of_data.map(|issue| issue.issue)
                && a.message == b.message
                && (a.satellite_id.system != GnssSystem::Galileo
                    || galileo_fnav_set(a) == galileo_fnav_set(b));
            if repeat {
                continue;
            }
        }
        kept.push(index);
    }
    kept
}

/// A GLONASS record's frame time as RTKLIB `decode_geph` forms `tof`: the stated
/// message frame time within its UTC day (0 when blank) on the day of the stated epoch,
/// moved to within half a day of `tb`, in GPS time.
fn glonass_frame_key(record: &GlonassRecord) -> f64 {
    let day_start =
        ((record.epoch_utc_j2000_s + 43_200.0) / 86_400.0).floor() * 86_400.0 - 43_200.0;
    let tod = record.message_frame_time_s.unwrap_or(0.0) % 86_400.0;
    let tof = adjust_day(day_start + tod, record.toe_utc_j2000_s);
    tof + gps_minus_utc_at_utc_j2000_s(tof)
}

/// Indices of `records` in RTKLIB `uniqgeph` order: sorted by frame time, then
/// reference epoch, then satellite, with a record that repeats the last kept record's
/// satellite, reference epoch and health left out.
fn glonass_selection_order(records: &[GlonassRecord]) -> Vec<usize> {
    let keys: Vec<(f64, f64)> = records
        .iter()
        .map(|r| (glonass_frame_key(r), r.toe_gpst_j2000_s()))
        .collect();
    let mut order: Vec<usize> = (0..records.len()).collect();
    order.sort_by(|&a, &b| {
        keys[a]
            .0
            .total_cmp(&keys[b].0)
            .then(keys[a].1.total_cmp(&keys[b].1))
            .then(records[a].satellite_id.cmp(&records[b].satellite_id))
    });
    let mut kept: Vec<usize> = Vec::with_capacity(order.len());
    for index in order {
        if let Some(&last) = kept.last() {
            let (a, b) = (&records[last], &records[index]);
            if a.satellite_id == b.satellite_id
                && keys[last].1 == keys[index].1
                && a.sv_health == b.sv_health
                && a.health_flags_word() == b.health_flags_word()
            {
                continue;
            }
        }
        kept.push(index);
    }
    kept
}

/// An SBAS record's frame time as RTKLIB `decode_seph` forms `tof`: the stated
/// transmission time (0 when blank) in the GPS week of `t0`, moved to within half a week
/// of `t0`.
fn sbas_frame_key(record: &SbasRecord) -> f64 {
    let t0 = record.t0_j2000_s();
    let gps_s = t0 + crate::constants::GPS_EPOCH_TO_J2000_S;
    let week_start = (gps_s / SECONDS_PER_WEEK).floor() * SECONDS_PER_WEEK
        - crate::constants::GPS_EPOCH_TO_J2000_S;
    adjust_week(week_start + record.message_frame_time_s.unwrap_or(0.0), t0)
}

/// Indices of `records` in RTKLIB `uniqseph` order: sorted by frame time, then `t0`, then
/// satellite, with a record that repeats the last kept record's satellite and `t0` left
/// out.
fn sbas_selection_order(records: &[SbasRecord]) -> Vec<usize> {
    let keys: Vec<(f64, f64)> = records
        .iter()
        .map(|r| (sbas_frame_key(r), r.t0_j2000_s()))
        .collect();
    let mut order: Vec<usize> = (0..records.len()).collect();
    order.sort_by(|&a, &b| {
        keys[a]
            .0
            .total_cmp(&keys[b].0)
            .then(keys[a].1.total_cmp(&keys[b].1))
            .then(records[a].satellite_id.cmp(&records[b].satellite_id))
    });
    let mut kept: Vec<usize> = Vec::with_capacity(order.len());
    for index in order {
        if let Some(&last) = kept.last() {
            if records[last].satellite_id == records[index].satellite_id
                && keys[last].1 == keys[index].1
            {
                continue;
            }
        }
        kept.push(index);
    }
    kept
}

fn glonass_state0(rec: &GlonassRecord) -> [f64; 6] {
    [
        rec.pos_m[0],
        rec.pos_m[1],
        rec.pos_m[2],
        rec.vel_m_s[0],
        rec.vel_m_s[1],
        rec.vel_m_s[2],
    ]
}

fn difference_velocity(start: [f64; 3], end: [f64; 3]) -> [f64; 3] {
    [
        (end[0] - start[0]) / EPHPOS_STEP_S,
        (end[1] - start[1]) / EPHPOS_STEP_S,
        (end[2] - start[2]) / EPHPOS_STEP_S,
    ]
}

fn validate_manual_record(record: &BroadcastRecord) -> CoreResult<()> {
    validate_finite(record.toe.tow_s, "record.toe.tow_s")?;
    validate_finite(record.toc.tow_s, "record.toc.tow_s")?;
    validate_finite(record.sv_health, "record.sv_health")?;
    if let Some(accuracy) = record.sv_accuracy_m {
        validate_finite(accuracy, "record.sv_accuracy_m")?;
    }
    if let Some(fit) = record.fit_interval_s {
        validate_finite(fit, "record.fit_interval_s")?;
        if fit <= 0.0 {
            return Err(invalid_input("record.fit_interval_s", "not positive"));
        }
    }
    validate_group_delays(record.group_delays)?;
    validate_cnav_presence(record)?;
    if let Some(cnav) = record.cnav {
        validate_cnav_parameters(cnav)?;
    }

    if let Some(cnav) = record.cnav {
        satellite_state_cnav(
            &record.elements,
            &cnav_rates(cnav),
            &record.clock,
            &record.constants(),
            record.elements.toe_sow,
            record.broadcast_clock_group_delay_s(),
        )
        .map(|_| ())
    } else {
        satellite_state(
            &record.elements,
            &record.clock,
            &record.constants(),
            record.elements.toe_sow,
            record.broadcast_clock_group_delay_s(),
            is_beidou_geo(record.satellite_id),
        )
        .map(|_| ())
    }
}

fn validate_group_delays(delays: BroadcastGroupDelays) -> CoreResult<()> {
    for (field, value) in [
        ("group_delays.gps_tgd_s", delays.gps_tgd_s),
        (
            "group_delays.galileo_bgd_e5a_e1_s",
            delays.galileo_bgd_e5a_e1_s,
        ),
        (
            "group_delays.galileo_bgd_e5b_e1_s",
            delays.galileo_bgd_e5b_e1_s,
        ),
        ("group_delays.beidou_tgd1_s", delays.beidou_tgd1_s),
        ("group_delays.beidou_tgd2_s", delays.beidou_tgd2_s),
        ("group_delays.cnav_isc_l1ca_s", delays.cnav_isc_l1ca_s),
        ("group_delays.cnav_isc_l2c_s", delays.cnav_isc_l2c_s),
        ("group_delays.cnav_isc_l5i5_s", delays.cnav_isc_l5i5_s),
        ("group_delays.cnav_isc_l5q5_s", delays.cnav_isc_l5q5_s),
        ("group_delays.cnav_isc_l1cd_s", delays.cnav_isc_l1cd_s),
        ("group_delays.cnav_isc_l1cp_s", delays.cnav_isc_l1cp_s),
    ] {
        if let Some(value) = value {
            validate_finite(value, field)?;
        }
    }
    Ok(())
}

fn validate_cnav_presence(record: &BroadcastRecord) -> CoreResult<()> {
    if record.message.is_cnav_family() != record.cnav.is_some() {
        return Err(invalid_input(
            "record.cnav",
            "must be present only for CNAV-family messages",
        ));
    }
    Ok(())
}

fn validate_cnav_parameters(params: CnavParameters) -> CoreResult<()> {
    validate_finite(params.adot_m_s, "record.cnav.adot_m_s")?;
    validate_finite(
        params.delta_n0_dot_rad_s2,
        "record.cnav.delta_n0_dot_rad_s2",
    )?;
    validate_finite(params.top.tow_s, "record.cnav.top.tow_s")?;
    validate_finite(
        params.transmission_time_sow,
        "record.cnav.transmission_time_sow",
    )?;
    if !(-16..=15).contains(&params.ura_ed_index) {
        return Err(invalid_input("record.cnav.ura_ed_index", "out of range"));
    }
    if !(-16..=15).contains(&params.ura_ned0_index) {
        return Err(invalid_input("record.cnav.ura_ned0_index", "out of range"));
    }
    if params.ura_ned1_index > 7 {
        return Err(invalid_input("record.cnav.ura_ned1_index", "out of range"));
    }
    if params.ura_ned2_index > 7 {
        return Err(invalid_input("record.cnav.ura_ned2_index", "out of range"));
    }
    Ok(())
}

fn cnav_rates(params: CnavParameters) -> CnavRates {
    CnavRates {
        adot_m_s: params.adot_m_s,
        delta_n0_dot_rad_s2: params.delta_n0_dot_rad_s2,
    }
}

/// Velocity of GLONASS record `rec` at `tk` from its reference epoch: its propagated
/// positions at `tk` and 1 ms later, differenced, as RTKLIB `ephpos` forms it with
/// `geph2pos`.
fn glonass_record_velocity(rec: &GlonassRecord, tk: f64) -> Option<[f64; 3]> {
    let state0 = glonass_state0(rec);
    let start = glonass::propagate(state0, rec.acc_m_s2, tk).ok()?;
    let end = glonass::propagate(state0, rec.acc_m_s2, ephpos_stepped_tk(tk)).ok()?;
    Some(difference_velocity(
        [start[0], start[1], start[2]],
        [end[0], end[1], end[2]],
    ))
}

/// Velocity of Keplerian record `rec` at `t_j2000_s`: its positions at `tk` and
/// `tk + EPHPOS_STEP_S` differenced over the step, as RTKLIB `ephpos` forms it.
fn keplerian_record_velocity(
    rec: &BroadcastRecord,
    sat: GnssSatelliteId,
    t_j2000_s: f64,
) -> Option<[f64; 3]> {
    let (_, sow, is_geo) = query_native_time(sat, t_j2000_s)?;
    let tk = time_from_reference_s(sow, rec.elements.toe_sow);
    let rates = rec.cnav.map(cnav_rates);
    // The CNAV model has no GEO branch.
    let is_geo = is_geo && rates.is_none();
    let position = |tk_s: f64| -> Option<[f64; 3]> {
        satellite_position_ecef_at_tk_unchecked(
            &rec.elements,
            rates.as_ref(),
            &rec.constants(),
            tk_s,
            is_geo,
        )
        .position()
        .ok()
        .map(|position| position.as_array())
    };
    let start = position(tk)?;
    let end = position(ephpos_stepped_tk(tk))?;
    Some(difference_velocity(start, end))
}

fn evaluate_record_unchecked(rec: &BroadcastRecord, sow: f64, is_geo: bool) -> SatelliteState {
    if let Some(cnav) = rec.cnav {
        satellite_state_cnav_unchecked(
            &rec.elements,
            &cnav_rates(cnav),
            &rec.clock,
            &rec.constants(),
            sow,
            rec.broadcast_clock_group_delay_s(),
        )
    } else {
        satellite_state_unchecked(
            &rec.elements,
            &rec.clock,
            &rec.constants(),
            sow,
            rec.broadcast_clock_group_delay_s(),
            is_geo,
        )
    }
}

fn exact_week_tow_epoch(time: GnssWeekTow) -> Option<ExactEpoch> {
    let epoch_offset_s = match time.system {
        crate::astro::time::TimeScale::Bdt => crate::constants::BDS_EPOCH_MINUS_GPS_EPOCH_S as i64,
        _ => 0,
    };
    let whole_s = i64::from(time.week) * SECONDS_PER_WEEK as i64 + epoch_offset_s
        - crate::constants::GPS_EPOCH_TO_J2000_S as i64;
    ExactEpoch::from_j2000_seconds(whole_s as f64)?.checked_add_seconds(time.tow_s)
}

fn exact_selection_epoch(selection_j2000_s: f64) -> Option<ExactEpochQuery> {
    ExactEpoch::from_binary_j2000_seconds(selection_j2000_s)
}

fn within_exact_interval(
    query: &ExactEpochQuery,
    reference: &ExactEpochQuery,
    limit_s: f64,
) -> Option<bool> {
    let direction = query.compare_interval_query(reference, 0.0)?;
    if direction == Ordering::Less {
        Some(reference.compare_interval_query(query, limit_s)? != Ordering::Greater)
    } else {
        Some(query.compare_interval_query(reference, limit_s)? != Ordering::Greater)
    }
}

fn record_within_exact_limit(record: &BroadcastRecord, selection_epoch: &ExactEpochQuery) -> bool {
    let native_selection_epoch = if record.toe.system == crate::astro::time::TimeScale::Bdt {
        let Some(epoch) = selection_epoch
            .clone()
            .checked_sub_binary_seconds(crate::constants::GPST_MINUS_BDT_S)
        else {
            return false;
        };
        epoch
    } else {
        selection_epoch.clone()
    };
    let Some(reference) = exact_week_tow_epoch(record.toe).map(ExactEpoch::query) else {
        return false;
    };
    if record.satellite_id.system == GnssSystem::Galileo
        && native_selection_epoch
            .compare_interval_query(&reference, 0.0)
            .is_none_or(|direction| direction != Ordering::Greater)
    {
        return false;
    }
    within_exact_interval(
        &native_selection_epoch,
        &reference,
        keplerian_max_dtoe_s(record.satellite_id.system),
    )
    .unwrap_or(false)
}

fn exact_glonass_gpst_epoch_query(record: &GlonassRecord) -> Option<ExactEpochQuery> {
    ExactEpoch::from_binary_j2000_seconds(record.toe_utc_j2000_s)?
        .checked_add_binary_seconds(gps_minus_utc_at_utc_j2000_s(record.toe_utc_j2000_s))
}

fn glonass_record_within_exact_limit(
    record: &GlonassRecord,
    selection_epoch: &ExactEpochQuery,
    limit_s: f64,
) -> bool {
    exact_glonass_gpst_epoch_query(record)
        .and_then(|reference| within_exact_interval(selection_epoch, &reference, limit_s))
        .unwrap_or(false)
}

fn exact_sbas_epoch_query(record: &SbasRecord) -> Option<ExactEpochQuery> {
    ExactEpoch::from_civil(
        record.epoch.year,
        i32::from(record.epoch.month),
        i32::from(record.epoch.day),
        i32::from(record.epoch.hour),
        i32::from(record.epoch.minute),
        record.epoch.second,
    )
    .map(ExactEpoch::query)
}

fn sbas_record_within_exact_limit(
    record: &SbasRecord,
    selection_epoch: &ExactEpochQuery,
    limit_s: f64,
) -> bool {
    exact_sbas_epoch_query(record)
        .and_then(|reference| within_exact_interval(selection_epoch, &reference, limit_s))
        .unwrap_or(false)
}

pub(crate) fn exact_record_deltas(
    epoch: &crate::astro::time::ExactEpochQuery,
    record: &BroadcastRecord,
) -> Option<(f64, f64)> {
    let toe = exact_week_tow_epoch(record.toe)?;
    let toc = exact_week_tow_epoch(record.toc)?;
    let epoch = if record.toe.system == crate::astro::time::TimeScale::Bdt {
        epoch
            .clone()
            .checked_sub_binary_seconds(crate::constants::GPST_MINUS_BDT_S)?
    } else {
        epoch.clone()
    };
    Some((
        time_from_reference_delta_s(epoch.seconds_since(toe)),
        time_from_reference_delta_s(epoch.seconds_since(toc)),
    ))
}

fn evaluate_record_at_epoch_query(
    record: &BroadcastRecord,
    epoch: &crate::astro::time::ExactEpochQuery,
    is_geo: bool,
) -> Option<SatelliteState> {
    let (tk_s, toc_delta_s) = exact_record_deltas(epoch, record)?;
    let rates = record.cnav.map(cnav_rates);
    Some(satellite_state_at_deltas_unchecked(
        &record.elements,
        rates.as_ref(),
        &record.clock,
        &record.constants(),
        tk_s,
        toc_delta_s,
        record.broadcast_clock_group_delay_s(),
        is_geo && rates.is_none(),
    ))
}

const fn cnav_tie_rank(message: NavMessage) -> u8 {
    match message {
        NavMessage::GpsCnav2 | NavMessage::QzssCnav2 => 1,
        _ => 0,
    }
}

fn validate_finite(value: f64, field: &'static str) -> CoreResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "not finite"))
    }
}

fn invalid_input(field: &'static str, reason: &'static str) -> Error {
    Error::InvalidInput(format!("{field} {reason}"))
}

impl core::str::FromStr for BroadcastStore {
    type Err = NavParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_nav(s)
    }
}

impl BroadcastStore {
    /// Position, `satposs` clock and single-frequency group delay of the record selected
    /// for `sat` at `t_j2000_s`, from one selection.
    fn state_with_group_delay(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        self.state_with_group_delay_selected(sat, t_j2000_s, t_j2000_s)
    }

    /// [`Self::state_with_group_delay`] of the record selected at `selection_j2000_s`,
    /// evaluated at `t_j2000_s`: RTKLIB `satposs` selects the record by the observation
    /// epoch (`seleph(teph, ...)`, `selgeph`, `selseph`) and evaluates it at the
    /// transmission epoch.
    fn state_with_group_delay_selected(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        match sat.system {
            // GLONASS is not Keplerian: integrate its broadcast state vector with the RK4
            // propagator. The clock is `geph2pos`'s, `-TauN + GammaN·tk` with `tk` not
            // iterated, as RTKLIB `satposs` returns it.
            GnssSystem::Glonass => {
                let (rec, tk) = self.glonass_selected(sat, t_j2000_s, selection_j2000_s)?;
                let state = glonass::propagate(glonass_state0(rec), rec.acc_m_s2, tk).ok()?;
                let clock = glonass::position_clock_offset_s(rec.clk_bias, rec.gamma_n, tk);
                Some((
                    [state[0], state[1], state[2]],
                    clock,
                    rec.single_frequency_group_delay_s(),
                ))
            }
            // SBAS: `seph2pos`, a constant-acceleration state and a first-order clock.
            GnssSystem::Sbas => {
                let (rec, _) = self.sbas_selected(sat, t_j2000_s, selection_j2000_s)?;
                let (position, clock) = rec.position_clock_at_j2000_s(t_j2000_s);
                Some((position, clock, None))
            }
            // Keplerian systems. The query instant (J2000, GPST-aligned) is read in the
            // satellite system's own scale and seconds of week: BeiDou runs on BDT (= GPST
            // - 14 s), and its geostationary satellites take the GEO orbit branch.
            _ => {
                let (rec, sow, is_geo) =
                    self.keplerian_selected(sat, t_j2000_s, selection_j2000_s)?;
                let state = evaluate_record_unchecked(rec, sow, is_geo);
                let position = state.orbit.position().ok()?;
                Some((
                    position.as_array(),
                    satposs_clock_s(&state),
                    Some(rec.broadcast_clock_group_delay_s()),
                ))
            }
        }
    }

    /// Variance (m²) of the satellite position and clock error of the record selected
    /// for `sat` at `selection_j2000_s`, as RTKLIB `eph2pos`, `geph2pos` and `seph2pos`
    /// state it: `var_uraeph` of a Keplerian record's accuracy (the Galileo SISA through
    /// `sisa_index`, the URA index of any other system's stated URA through `uraindex`,
    /// a blank accuracy as 0), `ERREPH_GLO²` (25 m²) for a GLONASS record, and
    /// `var_uraeph` of an SBAS record's URA. `None` where no record is selected.
    pub fn ephemeris_variance_m2(
        &self,
        sat: GnssSatelliteId,
        selection_j2000_s: f64,
    ) -> Option<f64> {
        match sat.system {
            GnssSystem::Glonass => self
                .select_glonass(sat, selection_j2000_s)
                .map(|_| ERREPH_GLO_M * ERREPH_GLO_M),
            GnssSystem::Sbas => self
                .select_sbas(sat, selection_j2000_s)
                .map(|(record, _)| sbas_variance_m2(record)),
            _ => {
                let (selection_native_s, _, _) = query_native_time(sat, selection_j2000_s)?;
                self.select(sat, selection_native_s)
                    .map(keplerian_variance_m2)
            }
        }
    }

    /// The Keplerian record for `sat` selected at `selection_j2000_s`, with the seconds of
    /// week and the GEO flag of `t_j2000_s`, the epoch it is evaluated at.
    fn keplerian_selected(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<(&BroadcastRecord, f64, bool)> {
        let (selection_native_s, _, _) = query_native_time(sat, selection_j2000_s)?;
        let rec = self.select(sat, selection_native_s)?;
        let (_, sow, is_geo) = query_native_time(sat, t_j2000_s)?;
        Some((rec, sow, is_geo))
    }

    /// The GLONASS record for `sat` RTKLIB `selgeph` selects at `selection_j2000_s`, with
    /// `tk`, the time of `t_j2000_s` from its reference epoch in GPS time.
    fn glonass_selected(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<(&GlonassRecord, f64)> {
        let (rec, _) = self.select_glonass(sat, selection_j2000_s)?;
        Some((rec, t_j2000_s - rec.toe_gpst_j2000_s()))
    }

    /// The SBAS record for `sat` RTKLIB `selseph` selects at `selection_j2000_s`, with the
    /// time of `t_j2000_s` from its reference epoch `t0`.
    fn sbas_selected(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<(&SbasRecord, f64)> {
        let (rec, _) = self.select_sbas(sat, selection_j2000_s)?;
        Some((rec, t_j2000_s - rec.t0_j2000_s()))
    }
}

impl EphemerisSource for BroadcastStore {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        self.state_with_group_delay(sat, t_j2000_s)
            .map(|(position, clock, _)| (position, clock))
    }

    /// The broadcast group delay of the record [`Self::position_clock_at_j2000_s`] uses:
    /// GPS, QZSS and NavIC TGD, Galileo BGD E5b/E1 for I/NAV and E5a/E1 for F/NAV, BeiDou
    /// TGD1, TGD less ISC L1C/A for CNAV, and `-ΔτN / (γ - 1)` for a GLONASS record that
    /// states `ΔτN`. `None` for SBAS and a GLONASS record without `ΔτN`.
    fn single_frequency_group_delay_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        BroadcastStore::single_frequency_group_delay_s(self, sat, t_j2000_s)
    }

    fn position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        self.state_with_group_delay(sat, t_j2000_s)
    }

    /// The one-evaluation read above; a broadcast store never refuses a state.
    fn try_position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> crate::Result<Option<crate::astro::time::Validated<crate::spp::PositionClockGroupDelay>>>
    {
        Ok(self
            .state_with_group_delay(sat, t_j2000_s)
            .map(crate::astro::time::Validated::ok))
    }

    /// The record selected at `selection_j2000_s`, evaluated at `t_j2000_s`, as RTKLIB
    /// `satposs` selects it by the observation epoch.
    fn try_position_clock_group_delay_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> crate::Result<Option<crate::astro::time::Validated<crate::spp::PositionClockGroupDelay>>>
    {
        Ok(self
            .state_with_group_delay_selected(sat, t_j2000_s, selection_j2000_s)
            .map(crate::astro::time::Validated::ok))
    }

    fn try_position_clock_group_delay_selected_at_exact_epoch(
        &self,
        sat: GnssSatelliteId,
        epoch: crate::astro::time::ExactEpoch,
        selection_j2000_s: f64,
    ) -> crate::Result<Option<crate::astro::time::Validated<crate::spp::PositionClockGroupDelay>>>
    {
        let selection_epoch =
            exact_selection_epoch(selection_j2000_s).ok_or(Error::EpochOutOfRange)?;
        self.try_position_clock_group_delay_selected_at_epoch_query(
            sat,
            &epoch.query(),
            &selection_epoch,
        )
    }

    fn try_position_clock_group_delay_selected_at_epoch_query(
        &self,
        sat: GnssSatelliteId,
        epoch: &crate::astro::time::ExactEpochQuery,
        selection_epoch: &crate::astro::time::ExactEpochQuery,
    ) -> crate::Result<Option<crate::astro::time::Validated<crate::spp::PositionClockGroupDelay>>>
    {
        let state = match sat.system {
            GnssSystem::Gps
            | GnssSystem::Galileo
            | GnssSystem::Qzss
            | GnssSystem::BeiDou
            | GnssSystem::Navic => {
                let Some(record) = self.select_record_at_epoch_query(sat, selection_epoch) else {
                    return Ok(None);
                };
                let (_, _, is_geo) = super::query_native_exact_time(sat, epoch.epoch())
                    .ok_or(Error::EpochOutOfRange)?;
                let state = evaluate_record_at_epoch_query(record, epoch, is_geo)
                    .ok_or(Error::EpochOutOfRange)?;
                let Some(position) = state.orbit.position().ok() else {
                    return Ok(None);
                };
                Some((
                    position.as_array(),
                    satposs_clock_s(&state),
                    Some(record.broadcast_clock_group_delay_s()),
                ))
            }
            GnssSystem::Glonass => {
                let Some(record) = self.select_glonass_at_epoch_query(sat, selection_epoch) else {
                    return Ok(None);
                };
                let toe = exact_glonass_gpst_epoch_query(record).ok_or(Error::EpochOutOfRange)?;
                let tk = epoch.seconds_since_query(&toe);
                let state = glonass::propagate(glonass_state0(record), record.acc_m_s2, tk).ok();
                state.map(|state| {
                    (
                        [state[0], state[1], state[2]],
                        glonass::position_clock_offset_s(record.clk_bias, record.gamma_n, tk),
                        record.single_frequency_group_delay_s(),
                    )
                })
            }
            GnssSystem::Sbas => {
                let Some(record) = self.select_sbas_at_epoch_query(sat, selection_epoch) else {
                    return Ok(None);
                };
                let t0 = exact_sbas_epoch_query(record).ok_or(Error::EpochOutOfRange)?;
                let tk = epoch.seconds_since_query(&t0);
                Some((
                    record.position_at(tk),
                    record.af0_s + record.af1_s_s * tk,
                    None,
                ))
            }
        };
        Ok(state.map(crate::astro::time::Validated::ok))
    }

    /// [`BroadcastStore::transmit_epoch_clock_s`]: the clock polynomial alone, as RTKLIB
    /// `ephclk` reads it.
    fn try_transmit_epoch_clock_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> crate::Result<Option<crate::astro::time::Validated<f64>>> {
        Ok(
            BroadcastStore::transmit_epoch_clock_s(self, sat, t_j2000_s, selection_j2000_s)
                .map(crate::astro::time::Validated::ok),
        )
    }

    fn try_transmit_epoch_clock_at_exact_epoch(
        &self,
        sat: GnssSatelliteId,
        epoch: ExactEpoch,
        selection_j2000_s: f64,
    ) -> crate::Result<Option<crate::astro::time::Validated<f64>>> {
        let selection_epoch =
            exact_selection_epoch(selection_j2000_s).ok_or(Error::EpochOutOfRange)?;
        self.try_transmit_epoch_clock_at_epoch_query(sat, &epoch.query(), &selection_epoch)
    }

    fn try_transmit_epoch_clock_at_epoch_query(
        &self,
        sat: GnssSatelliteId,
        epoch: &crate::astro::time::ExactEpochQuery,
        selection_epoch: &crate::astro::time::ExactEpochQuery,
    ) -> crate::Result<Option<crate::astro::time::Validated<f64>>> {
        let clock = match sat.system {
            GnssSystem::Gps
            | GnssSystem::Galileo
            | GnssSystem::Qzss
            | GnssSystem::BeiDou
            | GnssSystem::Navic => {
                let Some(record) = self.select_record_at_epoch_query(sat, selection_epoch) else {
                    return Ok(None);
                };
                let (_, toc_delta_s) =
                    exact_record_deltas(epoch, record).ok_or(Error::EpochOutOfRange)?;
                Some(satellite_clock_bias_at_delta_unchecked(
                    &record.clock,
                    toc_delta_s,
                ))
            }
            GnssSystem::Glonass => {
                let Some(record) = self.select_glonass_at_epoch_query(sat, selection_epoch) else {
                    return Ok(None);
                };
                let reference =
                    exact_glonass_gpst_epoch_query(record).ok_or(Error::EpochOutOfRange)?;
                Some(crate::glonass::clock_offset_s(
                    record.clk_bias,
                    record.gamma_n,
                    epoch.seconds_since_query(&reference),
                ))
            }
            GnssSystem::Sbas => {
                let Some(record) = self.select_sbas_at_epoch_query(sat, selection_epoch) else {
                    return Ok(None);
                };
                let reference = exact_sbas_epoch_query(record).ok_or(Error::EpochOutOfRange)?;
                let ts = epoch.seconds_since_query(&reference);
                let mut time = ts;
                for _ in 0..2 {
                    time = ts - (record.af0_s + record.af1_s_s * time);
                }
                Some(record.af0_s + record.af1_s_s * time)
            }
        };
        Ok(clock.map(crate::astro::time::Validated::ok))
    }

    /// [`BroadcastStore::ephemeris_variance_m2`] of the record selected at
    /// `selection_j2000_s`, or `0.0` where no record is selected.
    fn ephemeris_variance_m2(
        &self,
        sat: GnssSatelliteId,
        _t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> f64 {
        BroadcastStore::ephemeris_variance_m2(self, sat, selection_j2000_s).unwrap_or(0.0)
    }

    fn ephemeris_variance_at_epoch_query(
        &self,
        sat: GnssSatelliteId,
        _state_epoch: &crate::astro::time::ExactEpochQuery,
        selection_epoch: &crate::astro::time::ExactEpochQuery,
    ) -> f64 {
        match sat.system {
            GnssSystem::Glonass => self
                .select_glonass_at_epoch_query(sat, selection_epoch)
                .map_or(0.0, |_| ERREPH_GLO_M * ERREPH_GLO_M),
            GnssSystem::Sbas => self
                .select_sbas_at_epoch_query(sat, selection_epoch)
                .map_or(0.0, sbas_variance_m2),
            _ => self
                .select_record_at_epoch_query(sat, selection_epoch)
                .map_or(0.0, keplerian_variance_m2),
        }
    }
}

/// BeiDou SSR IOD of a record, `mod(toe/720, 240)` with `toe` the BDT seconds of week of
/// the ephemeris reference time (IGS SSR v1.00, IDF012). `None` for a record whose `toe`
/// is not a whole number of seconds within the week.
fn beidou_ssr_iod(record: &BroadcastRecord) -> Option<u32> {
    let toe_s = record.elements.toe_sow;
    if !(0.0..SECONDS_PER_WEEK).contains(&toe_s) || toe_s.fract() != 0.0 {
        return None;
    }
    Some(((toe_s as u32) / 720) % 240)
}

/// GLONASS `tb` of a record as RTKLIB `readrnx` forms its IODE: the index of the 15-min
/// interval of the day, in UTC + 3 h, of the reference epoch,
/// `(int)(fmod(tow + 10800, 86400) / 900 + 0.5)`. The record's reference epoch is UTC
/// seconds since J2000, an epoch at 12:00, so its UTC time of day is that plus 43200 s.
fn glonass_tb(record: &GlonassRecord) -> Option<u32> {
    let toe_s = record.toe_utc_j2000_s;
    if !toe_s.is_finite() {
        return None;
    }
    let tod_s = (toe_s + 43_200.0).rem_euclid(86_400.0);
    Some(((tod_s + 10_800.0).rem_euclid(86_400.0) / 900.0 + 0.5) as u32)
}

/// The satellite clock RTKLIB `satposs` returns for a Keplerian broadcast record: the
/// polynomial at `t - toc`, not iterated, and the relativistic term, without the group
/// delay (`eph2pos`: "satellite clock includes relativity correction without code bias
/// (tgd or bgd)"). It is formed as `dt_clock_total_s` forms its first two terms, so
/// subtracting the group delay from it gives `dt_clock_total_s` bit for bit.
fn satposs_clock_s(state: &SatelliteState) -> f64 {
    state.clock.dt_clock_poly_s + state.clock.dt_rel_s
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod exact_epoch_tests {
    use super::{
        exact_selection_epoch, exact_week_tow_epoch, within_exact_interval, BroadcastStore,
    };
    use crate::astro::time::{ExactEpoch, GnssWeekTow, TimeScale};
    use crate::broadcast::{ClockPolynomial, KeplerianElements};
    use crate::id::{GnssSatelliteId, GnssSystem};
    use crate::rinex_nav::{
        BroadcastGroupDelays, BroadcastIssue, BroadcastRecord, GlonassRecord, NavEpoch, NavMessage,
        SbasRecord, StatedNavFields,
    };

    fn gps_selection_record(toe_sow: f64, issue: u32) -> BroadcastRecord {
        let message = NavMessage::GpsLnav;
        BroadcastRecord {
            satellite_id: GnssSatelliteId::new(GnssSystem::Gps, 30).expect("valid satellite id"),
            message,
            issue_of_data: Some(BroadcastIssue { issue, message }),
            week: 2111,
            toe: GnssWeekTow::new(TimeScale::Gpst, 2111, toe_sow).expect("valid toe"),
            toc: GnssWeekTow::new(TimeScale::Gpst, 2111, toe_sow).expect("valid toc"),
            elements: KeplerianElements {
                sqrt_a: 5153.0,
                e: 0.001,
                m0: 0.0,
                delta_n: 0.0,
                omega0: 0.0,
                i0: 0.9,
                omega: 0.0,
                omega_dot: 0.0,
                idot: 0.0,
                cuc: 0.0,
                cus: 0.0,
                crc: 0.0,
                crs: 0.0,
                cic: 0.0,
                cis: 0.0,
                toe_sow,
            },
            clock: ClockPolynomial {
                af0: 0.0,
                af1: 0.0,
                af2: 0.0,
                toc_sow: toe_sow,
            },
            group_delays: BroadcastGroupDelays::default(),
            cnav: None,
            sv_health: 0.0,
            sv_accuracy_m: Some(2.0),
            fit_interval_s: None,
            stated: StatedNavFields::default(),
        }
    }

    fn glonass_selection_record(toe_utc_j2000_s: f64) -> GlonassRecord {
        GlonassRecord {
            satellite_id: GnssSatelliteId::new(GnssSystem::Glonass, 1).expect("valid satellite id"),
            toe_utc_j2000_s,
            epoch_utc_j2000_s: toe_utc_j2000_s,
            pos_m: [0.0; 3],
            vel_m_s: [0.0; 3],
            acc_m_s2: [0.0; 3],
            clk_bias: 0.0,
            gamma_n: 0.0,
            sv_health: 0.0,
            freq_channel: 0,
            stated_freq_channel: 0,
            message_frame_time_s: None,
            age_days: None,
            status_flags: None,
            l1_l2_group_delay_field_s: None,
            urai: None,
            health_flags: None,
        }
    }

    fn sbas_selection_record() -> SbasRecord {
        SbasRecord {
            satellite_id: GnssSatelliteId::new(GnssSystem::Sbas, 20).expect("valid satellite id"),
            epoch: NavEpoch {
                year: 2000,
                month: 1,
                day: 1,
                hour: 12,
                minute: 0,
                second: 0.0,
            },
            af0_s: 0.0,
            af1_s_s: 0.0,
            message_frame_time_s: None,
            pos_m: [0.0; 3],
            vel_m_s: [0.0; 3],
            acc_m_s2: [0.0; 3],
            health: 0.0,
            ura_m: None,
            iodn: None,
        }
    }

    #[test]
    fn exact_igs_galileo_issue_uses_low_bits_after_but_not_at_toe() {
        let mut record = gps_selection_record(120_000.0, 0x123);
        record.satellite_id = GnssSatelliteId::new(GnssSystem::Galileo, 1).unwrap();
        record.message = NavMessage::GalileoInav;
        record.issue_of_data = Some(BroadcastIssue {
            issue: 0x123,
            message: NavMessage::GalileoInav,
        });
        let satellite = record.satellite_id;
        let toe = exact_week_tow_epoch(record.toe).unwrap().query();
        let after = toe.clone().checked_add_binary_seconds(1.0e-18).unwrap();
        assert_eq!(
            toe.j2000_seconds().to_bits(),
            after.j2000_seconds().to_bits()
        );
        let store = BroadcastStore::new(vec![record]).unwrap();
        assert!(store
            .select_by_issue_low_bits_at_epoch_query(
                satellite,
                0x23,
                8,
                NavMessage::GalileoInav,
                &toe,
            )
            .is_none());
        let selected = store
            .select_by_issue_low_bits_at_epoch_query(
                satellite,
                0x23,
                8,
                NavMessage::GalileoInav,
                &after,
            )
            .unwrap();
        assert_eq!(selected.issue_of_data.unwrap().issue, 0x123);
        assert!(store
            .select_by_issue_low_bits_at_epoch_query(
                satellite,
                0x24,
                8,
                NavMessage::GalileoInav,
                &after,
            )
            .is_none());
    }

    #[test]
    fn exact_sbas_ssr_selection_keeps_age_boundary_and_local_transmit_time() {
        let mut record = sbas_selection_record();
        record.epoch.year = 2020;
        record.pos_m = [1.0, 2.0, 3.0];
        record.vel_m_s = [1_000.0, 0.0, 0.0];
        record.af1_s_s = 1.0e-5;
        record.iodn = Some(7.0);
        let satellite = record.satellite_id;
        let reference = super::exact_sbas_epoch_query(&record).unwrap();
        let transmit = reference
            .clone()
            .checked_add_binary_seconds(20.0e-9)
            .unwrap();
        assert_eq!(
            reference.j2000_seconds().to_bits(),
            transmit.j2000_seconds().to_bits()
        );
        let at_limit = reference.clone().checked_add_binary_seconds(360.0).unwrap();
        let past_limit = at_limit
            .clone()
            .checked_add_binary_seconds(1.0e-18)
            .unwrap();
        assert_eq!(
            at_limit.j2000_seconds().to_bits(),
            past_limit.j2000_seconds().to_bits()
        );
        let native_issue = (0..512)
            .find(|&issue| super::sbas_issue_matches_exact(&record, issue, false))
            .unwrap();
        let mut store = BroadcastStore::new(Vec::new()).unwrap();
        store.sbas = vec![record];
        store.sbas_selection = vec![0];
        store.exclude_unusable = true;
        for (issue, igs_ssr) in [(native_issue, false), (7, true)] {
            let state = store
                .sbas_ssr_state_at_query(satellite, issue, igs_ssr, &transmit, &at_limit)
                .unwrap();
            assert_eq!(state.0, [1.0 + 1_000.0 * 20.0e-9, 2.0, 3.0]);
            assert_eq!(state.2.to_bits(), (1.0e-5_f64 * 20.0e-9).to_bits());
            assert!(store
                .sbas_ssr_state_at_query(satellite, issue, igs_ssr, &transmit, &past_limit)
                .is_none());
        }
        store.sbas[0].health = 1.0;
        assert!(store
            .sbas_ssr_state_at_query(satellite, 7, true, &transmit, &reference)
            .is_none());
    }

    #[test]
    fn exact_native_sbas_issue_does_not_round_across_a_sixteen_second_boundary() {
        let mut record = sbas_selection_record();
        record.epoch.year = 2020;
        record.epoch.second = 16.0 - 1.0e-9;
        let before = (0..512)
            .find(|&issue| super::sbas_issue_matches_exact(&record, issue, false))
            .unwrap();
        let rounded_before = record.t0_j2000_s();
        record.epoch.second = 16.0;
        assert_eq!(rounded_before.to_bits(), record.t0_j2000_s().to_bits());
        assert!(super::sbas_issue_matches_exact(
            &record,
            (before + 1) % 512,
            false
        ));
        assert!(!super::sbas_issue_matches_exact(&record, before, false));
    }

    #[test]
    fn exact_week_tow_difference_keeps_fraction_across_week_boundary() {
        let reference =
            exact_week_tow_epoch(GnssWeekTow::new(TimeScale::Gpst, 2200, 604_799.9).unwrap())
                .unwrap();
        let query =
            exact_week_tow_epoch(GnssWeekTow::new(TimeScale::Gpst, 2201, 0.0).unwrap()).unwrap();
        assert_eq!(query.seconds_since(reference), 0.1);

        let just_before = query.checked_sub_seconds(1.0e-30).unwrap();
        assert_eq!(just_before.seconds_since(query), -1.0e-30);
        assert!(just_before < ExactEpoch::from_j2000_seconds(query.j2000_seconds()).unwrap());
    }

    #[test]
    fn exact_record_selection_keeps_twenty_nanoseconds_before_midpoint() {
        let early = gps_selection_record(0.0, 1);
        let late = gps_selection_record(7_200.0, 2);
        let sat = early.satellite_id;
        let midpoint = exact_week_tow_epoch(
            GnssWeekTow::new(TimeScale::Gpst, 2111, 3_600.0).expect("valid midpoint"),
        )
        .expect("midpoint epoch");
        let selection_epoch = midpoint
            .query()
            .checked_sub_binary_seconds(20.0e-9)
            .expect("finite offset");
        assert_eq!(
            selection_epoch.j2000_seconds().to_bits(),
            midpoint.j2000_seconds().to_bits()
        );
        let store = BroadcastStore::new(vec![early, late]).expect("valid store");
        let selected_exact = store
            .select_record_at_epoch_query(sat, &selection_epoch)
            .expect("exact selection");
        let selected_rounded = store
            .select_record_at(sat, selection_epoch.j2000_seconds())
            .expect("rounded selection");
        assert_eq!(selected_exact.issue_of_data.expect("issue").issue, 1);
        assert_eq!(selected_rounded.issue_of_data.expect("issue").issue, 2);
        let selected_tie = store
            .select_record_at_epoch_query(sat, &midpoint.query())
            .expect("exact tie selection");
        assert_eq!(selected_tie.issue_of_data.expect("issue").issue, 2);
    }

    #[test]
    fn exact_variance_uses_the_record_selected_before_rounded_midpoint() {
        let mut early = gps_selection_record(0.0, 1);
        early.sv_accuracy_m = Some(2.0);
        let mut late = gps_selection_record(7_200.0, 2);
        late.sv_accuracy_m = Some(20.0);
        let sat = early.satellite_id;
        let early_variance = super::keplerian_variance_m2(&early);
        let late_variance = super::keplerian_variance_m2(&late);
        assert_ne!(early_variance, late_variance);

        let midpoint = exact_week_tow_epoch(
            GnssWeekTow::new(TimeScale::Gpst, 2111, 3_600.0).expect("valid midpoint"),
        )
        .expect("midpoint epoch");
        let selection_epoch = midpoint
            .query()
            .checked_sub_binary_seconds(20.0e-9)
            .expect("finite offset");
        let store = BroadcastStore::new(vec![early, late]).expect("valid store");

        assert_eq!(
            crate::spp::EphemerisSource::ephemeris_variance_at_epoch_query(
                &store,
                sat,
                &selection_epoch,
                &selection_epoch,
            ),
            early_variance
        );
        assert_eq!(
            store.ephemeris_variance_m2(sat, selection_epoch.j2000_seconds()),
            Some(late_variance)
        );
    }

    #[test]
    fn exact_selection_age_limit_distinguishes_one_attosecond() {
        let reference = ExactEpoch::new(1_000_000_000, 0)
            .expect("reference epoch")
            .query();
        let at_limit = ExactEpoch::new(1_000_000_090, 0)
            .expect("limit epoch")
            .query();
        let twenty_nanoseconds_inside = at_limit
            .clone()
            .checked_sub_binary_seconds(20.0e-9)
            .expect("finite offset");
        let one_attosecond_past = ExactEpoch::new(1_000_000_090, 1)
            .expect("one-attosecond-past epoch")
            .query();
        assert_eq!(
            at_limit.j2000_seconds().to_bits(),
            one_attosecond_past.j2000_seconds().to_bits()
        );
        assert_eq!(
            at_limit.j2000_seconds().to_bits(),
            twenty_nanoseconds_inside.j2000_seconds().to_bits()
        );
        assert!(within_exact_interval(&twenty_nanoseconds_inside, &reference, 90.0).unwrap());
        assert!(within_exact_interval(&at_limit, &reference, 90.0).unwrap());
        assert!(!within_exact_interval(&one_attosecond_past, &reference, 90.0).unwrap());
    }

    #[test]
    fn galileo_exact_selection_requires_toe_and_includes_age_limit() {
        let mut record = gps_selection_record(7_200.0, 1);
        record.satellite_id = GnssSatelliteId::new(GnssSystem::Galileo, 1).unwrap();
        record.message = NavMessage::GalileoInav;
        record.toe = GnssWeekTow::new(TimeScale::Gst, 2111, 7_200.0).unwrap();
        record.toc = record.toe;

        let toe = exact_week_tow_epoch(record.toe).unwrap().query();
        let just_before_toe = toe
            .clone()
            .checked_sub_binary_seconds(1.0e-18)
            .expect("finite offset");
        let just_after_toe = toe
            .clone()
            .checked_add_binary_seconds(1.0e-18)
            .expect("finite offset");
        assert!(!super::record_within_exact_limit(&record, &just_before_toe));
        assert!(super::record_within_exact_limit(&record, &just_after_toe));

        let at_age_limit = toe
            .clone()
            .checked_add_binary_seconds(14_400.0)
            .expect("finite offset");
        let past_age_limit = at_age_limit
            .clone()
            .checked_add_binary_seconds(1.0e-18)
            .expect("finite offset");
        assert!(super::record_within_exact_limit(&record, &at_age_limit));
        assert!(!super::record_within_exact_limit(&record, &past_age_limit));
    }

    #[test]
    fn exact_glonass_and_sbas_age_limits_include_boundary_but_not_one_attosecond_past() {
        let glonass = glonass_selection_record(0.0);
        let glonass_reference = super::exact_glonass_gpst_epoch_query(&glonass).unwrap();
        let glonass_limit = glonass_reference
            .clone()
            .checked_add_binary_seconds(1_800.0)
            .unwrap();
        let glonass_inside = ExactEpoch::new(1_812, ExactEpoch::ATTOSECONDS_PER_SECOND - 1)
            .unwrap()
            .query();
        let glonass_past = ExactEpoch::new(1_813, 1).unwrap().query();
        assert!(super::glonass_record_within_exact_limit(
            &glonass,
            &glonass_inside,
            1_800.0
        ));
        assert!(super::glonass_record_within_exact_limit(
            &glonass,
            &glonass_limit,
            1_800.0
        ));
        assert!(!super::glonass_record_within_exact_limit(
            &glonass,
            &glonass_past,
            1_800.0
        ));

        let sbas = sbas_selection_record();
        let sbas_reference = super::exact_sbas_epoch_query(&sbas).unwrap();
        let sbas_limit = sbas_reference
            .clone()
            .checked_add_binary_seconds(360.0)
            .unwrap();
        let sbas_inside = ExactEpoch::new(359, ExactEpoch::ATTOSECONDS_PER_SECOND - 1)
            .unwrap()
            .query();
        let sbas_past = ExactEpoch::new(360, 1).unwrap().query();
        assert!(super::sbas_record_within_exact_limit(
            &sbas,
            &sbas_inside,
            360.0
        ));
        assert!(super::sbas_record_within_exact_limit(
            &sbas,
            &sbas_limit,
            360.0
        ));
        assert!(!super::sbas_record_within_exact_limit(
            &sbas, &sbas_past, 360.0
        ));
    }

    #[test]
    fn exact_issue_selection_preserves_beidou_iod_modulo_and_full_issue() {
        let mut record = gps_selection_record(172_799.0, 0x1_2345);
        record.satellite_id = GnssSatelliteId::new(GnssSystem::BeiDou, 1).unwrap();
        record.message = NavMessage::BeidouD1;
        record.issue_of_data = Some(BroadcastIssue {
            issue: 0x1_2345,
            message: NavMessage::BeidouD1,
        });
        record.toe = GnssWeekTow::new(TimeScale::Bdt, 2111, 172_799.0).unwrap();
        record.toc = record.toe;

        assert_eq!(super::beidou_ssr_iod(&record), Some(239));
        let query = exact_week_tow_epoch(record.toe).unwrap().query();
        let store = BroadcastStore::new(vec![record]).expect("valid store");
        assert!(store
            .select_by_issue_at_epoch_query(
                GnssSatelliteId::new(GnssSystem::BeiDou, 1).unwrap(),
                BroadcastIssue {
                    issue: 0x1_2345,
                    message: NavMessage::BeidouD1,
                },
                NavMessage::BeidouD1,
                &query,
            )
            .is_some());
        assert!(store
            .select_by_issue_at_epoch_query(
                GnssSatelliteId::new(GnssSystem::BeiDou, 1).unwrap(),
                BroadcastIssue {
                    issue: 0x2345,
                    message: NavMessage::BeidouD1,
                },
                NavMessage::BeidouD1,
                &query,
            )
            .is_none());
        assert!(store
            .select_by_beidou_ssr_iod_at_epoch_query(
                GnssSatelliteId::new(GnssSystem::BeiDou, 1).unwrap(),
                239,
                NavMessage::BeidouD1,
                &query,
            )
            .is_some());
        assert!(store
            .select_by_beidou_ssr_iod_at_epoch_query(
                GnssSatelliteId::new(GnssSystem::BeiDou, 1).unwrap(),
                0x1_00ef,
                NavMessage::BeidouD1,
                &query,
            )
            .is_none());
    }

    #[test]
    fn binary_selection_epoch_keeps_the_input_f64_value() {
        let binary_epoch = 1_000_000_000.1_f64;
        let reference = ExactEpoch::new(1_000_000_000, 0)
            .expect("reference epoch")
            .query();
        let exact_binary = exact_selection_epoch(binary_epoch).expect("finite binary epoch");
        let decimal = ExactEpoch::from_j2000_seconds(binary_epoch)
            .expect("finite decimal epoch")
            .query();
        let binary_delta = exact_binary.seconds_since_query(&reference);
        let decimal_delta = decimal.seconds_since_query(&reference);
        assert_eq!(
            binary_delta.to_bits(),
            (binary_epoch - 1_000_000_000.0).to_bits()
        );
        assert_ne!(binary_delta.to_bits(), decimal_delta.to_bits());
    }
}
