//! Broadcast-store selection and SPP source adapter.

use crate::broadcast::{
    satellite_position_ecef_at_tk_unchecked, satellite_state, satellite_state_cnav,
    satellite_state_cnav_unchecked, satellite_state_unchecked, time_from_reference_s, CnavRates,
    SatelliteState,
};
use crate::constants::{HALF_WEEK_S, SECONDS_PER_WEEK};
use crate::error::{Error, Result as CoreResult};
use crate::glonass;
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::spp::EphemerisSource;

use super::{
    cnav_ura_nominal_m, gps_minus_utc_at_utc_j2000_s, is_beidou_geo, keplerian_max_dtoe_s,
    parse_nav_file, week_tow_native_j2000_s, BroadcastGroupDelays, BroadcastIssue, BroadcastRecord,
    CnavParameters, GlonassRecord, IonoCorrections, IonosphereFrame, NavDiagnostic, NavHeader,
    NavMessage, NavParseError, SbasRecord, SkippedNavBlock, EPHPOS_STEP_S, GLONASS_MAX_AGE_S,
    SBAS_MAX_AGE_S,
};
use super::{ephpos_stepped_tk, query_native_time, toe_native_j2000_s};
use crate::astro::time::model::GnssWeekTow;

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
        match sat.system {
            GnssSystem::Glonass => self.glonass_record_velocity(sat, t_j2000_s),
            GnssSystem::Sbas => {
                let (rec, t) = self.select_sbas(sat, t_j2000_s)?;
                let start = rec.position_at(t);
                let end = rec.position_at(ephpos_stepped_tk(t));
                Some(difference_velocity(start, end))
            }
            _ => {
                let rec = self.select_record_at(sat, t_j2000_s)?;
                keplerian_record_velocity(rec, sat, t_j2000_s)
            }
        }
    }

    /// Velocity of the GLONASS record selected at `t_j2000_s`: its propagated positions at
    /// `t_j2000_s` and 1 ms later, differenced, as RTKLIB `ephpos` forms it with `geph2pos`.
    fn glonass_record_velocity(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<[f64; 3]> {
        let (rec, tk) = self.select_glonass(sat, t_j2000_s)?;
        let state0 = glonass_state0(rec);
        let start = glonass::propagate(state0, rec.acc_m_s2, tk).ok()?;
        let end = glonass::propagate(state0, rec.acc_m_s2, ephpos_stepped_tk(tk)).ok()?;
        Some(difference_velocity(
            [start[0], start[1], start[2]],
            [end[0], end[1], end[2]],
        ))
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
        let rec = self.select_by_iode_at(sat, iode, t_j2000_s)?;
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
        let (_, sow, is_geo) = query_native_time(sat, t_j2000_s)?;
        let rec = self.select_by_iode_at(sat, iode, t_j2000_s)?;
        let state = evaluate_record_unchecked(rec, sow, is_geo);
        let position = state.orbit.position().ok()?;
        Some((
            position.as_array(),
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

    /// Position, velocity and clock of the GLONASS record for `sat` whose `tb`, the 15-min
    /// index of its reference epoch in UTC + 3 h, equals `iode`, as RTKLIB `satpos_ssr`
    /// forms them for a GLONASS SSR correction: `selgeph` by that issue (RTKLIB `readrnx`
    /// forms a GLONASS record's IODE as `tb`), then `geph2pos` at `t_j2000_s` and 1 ms
    /// later. The clock is `geph2pos`'s, `-TauN + GammaN·tk` with `tk` not iterated, and
    /// carries no relativistic term, which RTKLIB adds none of for GLONASS. The record's
    /// reference epoch lies within [`GLONASS_MAX_AGE_S`] of the query, and among records
    /// of that `tb` the first in selection order is taken, as `selgeph` returns for an
    /// issue.
    pub(crate) fn glonass_ssr_state(
        &self,
        sat: GnssSatelliteId,
        iode: u32,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], [f64; 3], f64)> {
        let rec = self
            .glonass_selection
            .iter()
            .map(|&index| &self.glonass[index])
            .find(|r| {
                r.satellite_id == sat
                    && glonass_tb(r) == Some(iode)
                    && (t_j2000_s - r.toe_gpst_j2000_s()).abs() <= GLONASS_MAX_AGE_S
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

/// RTKLIB `uraindex`: the first URA index whose value is at least `value`, 15 past the
/// table.
fn ura_index(value: f64) -> usize {
    URA_EPH_M
        .iter()
        .position(|&ura| ura >= value)
        .unwrap_or(URA_EPH_M.len())
}

/// RTKLIB `var_uraeph` for a URA index, m² (6144² past the table).
fn ura_variance_m2(index: usize) -> f64 {
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
        match sat.system {
            // GLONASS is not Keplerian: integrate its broadcast state vector with the RK4
            // propagator. The clock is `geph2pos`'s, `-TauN + GammaN·tk` with `tk` not
            // iterated, as RTKLIB `satposs` returns it.
            GnssSystem::Glonass => {
                let (rec, tk) = self.select_glonass(sat, t_j2000_s)?;
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
                let (rec, _) = self.select_sbas(sat, t_j2000_s)?;
                let (position, clock) = rec.position_clock_at_j2000_s(t_j2000_s);
                Some((position, clock, None))
            }
            // Keplerian systems. The query instant (J2000, GPST-aligned) is read in the
            // satellite system's own scale and seconds of week: BeiDou runs on BDT (= GPST
            // - 14 s), and its geostationary satellites take the GEO orbit branch.
            _ => {
                let (t_native_s, sow, is_geo) = query_native_time(sat, t_j2000_s)?;
                let rec = self.select(sat, t_native_s)?;
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
