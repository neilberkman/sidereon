//! Broadcast-store selection and SPP source adapter.

use crate::broadcast::{
    satellite_position_ecef_at_tk_unchecked, satellite_state, satellite_state_cnav,
    satellite_state_cnav_unchecked, satellite_state_unchecked, time_from_reference_s, CnavRates,
    SatelliteState,
};
use crate::error::{Error, Result as CoreResult};
use crate::glonass;
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::spp::EphemerisSource;

use super::{
    cnav_ura_nominal_m, is_beidou_geo, parse_glonass, parse_iono_corrections_checked,
    parse_leap_seconds_checked, parse_nav, BroadcastGroupDelays, BroadcastIssue, BroadcastRecord,
    CnavParameters, GlonassRecord, IonoCorrections, NavMessage, NavParseError, EPHPOS_STEP_S,
    GLONASS_MAX_AGE_S, MAX_EPHEMERIS_AGE_S,
};
use super::{ephpos_stepped_tk, query_native_time, toe_native_j2000_s};

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
/// For a satellite and epoch it selects the record whose reference time `toe` is
/// nearest in **continuous** GPS time (week number times the week length plus
/// the seconds of week, not the seconds of week alone), and rejects the query as
/// having no ephemeris if it falls outside that record's validity window - half
/// the broadcast GPS curve-fit interval, or the coarse [`MAX_EPHEMERIS_AGE_S`]
/// fallback for systems that do not broadcast one - so a stale or wrong-week
/// product cannot silently produce a position.
///
/// [`from_nav`](BroadcastStore::from_nav) applies a default usability policy:
/// healthy GPS LNAV, GPS/QZSS CNAV-family, Galileo I/NAV, BeiDou D1/D2, and
/// GLONASS records are kept, with CNAV no-prediction URA records excluded.
/// [`new`](BroadcastStore::new) keeps records verbatim for callers that want
/// their own policy.
pub struct BroadcastStore {
    records: Vec<BroadcastRecord>,
    glonass: Vec<GlonassRecord>,
    leap_seconds: Option<f64>,
    iono: IonoCorrections,
    message_preference: NavMessagePreference,
}

impl BroadcastStore {
    /// Build a store from already-parsed Keplerian records, verbatim (no policy
    /// filter, no GLONASS records, no leap-second offset, and no ionosphere
    /// coefficients; use [`from_nav`](Self::from_nav) to capture those).
    pub fn new(records: Vec<BroadcastRecord>) -> CoreResult<Self> {
        for record in &records {
            validate_manual_record(record)?;
        }
        Ok(Self {
            records,
            glonass: Vec::new(),
            leap_seconds: None,
            iono: IonoCorrections::default(),
            message_preference: NavMessagePreference::default(),
        })
    }

    /// Parse a RINEX 3.x/4.xx navigation file and keep the records usable for
    /// single-frequency positioning under the default policy: healthy GPS LNAV,
    /// GPS/QZSS CNAV-family records with URA prediction, Galileo I/NAV, BeiDou
    /// D1/D2, and GLONASS. The header's broadcast ionosphere coefficients (see
    /// [`iono_corrections`](Self::iono_corrections)) and leap-second offset are
    /// captured.
    pub fn from_nav(text: &str) -> Result<Self, NavParseError> {
        let records = parse_nav(text)?
            .into_iter()
            .filter(Self::is_default_usable)
            .collect();
        let glonass = parse_glonass(text)?
            .into_iter()
            .filter(|r| r.sv_health == 0.0)
            .collect();
        Ok(Self {
            records,
            glonass,
            leap_seconds: parse_leap_seconds_checked(text)?,
            iono: parse_iono_corrections_checked(text)?,
            message_preference: NavMessagePreference::default(),
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

    /// The broadcast ionosphere coefficients parsed from the navigation header
    /// or RINEX 4 body `> ION` frames (GPS `GPSA`/`GPSB`, BeiDou `BDSA`/`BDSB`,
    /// and Galileo `GAL`). Empty for a store built with [`new`](Self::new).
    pub fn iono_corrections(&self) -> IonoCorrections {
        self.iono
    }

    /// The held GLONASS records.
    pub fn glonass_records(&self) -> &[GlonassRecord] {
        &self.glonass
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

    /// The default usability policy: healthy and single-frequency-appropriate
    /// messages. CNAV-family records with no URA prediction are excluded.
    fn is_default_usable(r: &BroadcastRecord) -> bool {
        r.sv_health == 0.0
            && matches!(
                r.message,
                NavMessage::GpsLnav
                    | NavMessage::GpsCnav
                    | NavMessage::GpsCnav2
                    | NavMessage::QzssLnav
                    | NavMessage::QzssCnav
                    | NavMessage::QzssCnav2
                    | NavMessage::GalileoInav
                    | NavMessage::BeidouD1
                    | NavMessage::BeidouD2
            )
            && (!r.message.is_cnav_family()
                || r.cnav
                    .map(|cnav| cnav_ura_nominal_m(cnav.ura_ed_index).is_some())
                    .unwrap_or(false))
    }

    /// The held records.
    pub fn records(&self) -> &[BroadcastRecord] {
        &self.records
    }

    /// Select the broadcast record used for `sat` at `t_j2000_s`.
    ///
    /// The selection uses the same message preference, native-system time
    /// mapping, and validity-window checks as
    /// [`EphemerisSource::position_clock_at_j2000_s`]. Non-Keplerian systems
    /// return `None`.
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
    /// single-frequency user of its message: GPS and QZSS LNAV TGD, Galileo I/NAV BGD
    /// E5b/E1, Galileo F/NAV BGD E5a/E1, BeiDou TGD1, CNAV TGD less ISC L1C/A. The clock
    /// this store returns does not include it, as RTKLIB `satposs` returns none; a
    /// single-frequency pseudorange model subtracts it from that clock, as RTKLIB
    /// `pntpos` subtracts it from the pseudorange. `None` where no record is selected and
    /// for GLONASS.
    pub fn single_frequency_group_delay_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<f64> {
        self.select_record_at(sat, t_j2000_s)
            .map(BroadcastRecord::broadcast_clock_group_delay_s)
    }

    /// Velocity, metres per second, of the record selected for `sat` at `t_j2000_s`, as
    /// RTKLIB `ephpos` forms it: the difference of that record's positions at the epoch and
    /// [`EPHPOS_STEP_S`] later over the step, the step added to the record's reduced time
    /// (`tk` for a Keplerian record, time from the reference epoch for GLONASS) as RTKLIB
    /// adds it to its exact `gtime_t`. Differencing the source's positions instead would
    /// select a record on each side and blend two records across a record change. `None`
    /// where no record is selected or the system has no broadcast model here.
    pub(crate) fn selected_record_velocity(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<[f64; 3]> {
        if sat.system == GnssSystem::Glonass {
            return self.glonass_record_velocity(sat, t_j2000_s);
        }
        let rec = self.select_record_at(sat, t_j2000_s)?;
        keplerian_record_velocity(rec, sat, t_j2000_s)
    }

    /// Velocity of the GLONASS record selected at `t_j2000_s`: its propagated positions at
    /// `t_j2000_s` and 1 ms later, differenced, as RTKLIB `ephpos` forms it with `geph2pos`.
    fn glonass_record_velocity(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<[f64; 3]> {
        let (rec, tk) = self.select_glonass(sat, t_j2000_s)?;
        let state0 = [
            rec.pos_m[0],
            rec.pos_m[1],
            rec.pos_m[2],
            rec.vel_m_s[0],
            rec.vel_m_s[1],
            rec.vel_m_s[2],
        ];
        let start = glonass::propagate(state0, rec.acc_m_s2, tk).ok()?;
        let end = glonass::propagate(state0, rec.acc_m_s2, ephpos_stepped_tk(tk)).ok()?;
        Some([
            (end[0] - start[0]) / EPHPOS_STEP_S,
            (end[1] - start[1]) / EPHPOS_STEP_S,
            (end[2] - start[2]) / EPHPOS_STEP_S,
        ])
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

    /// Select the valid record for `sat` with a matching GPS issue byte at `t`.
    pub fn select_by_iode_at(
        &self,
        sat: GnssSatelliteId,
        iode: u8,
        t_j2000_s: f64,
    ) -> Option<&BroadcastRecord> {
        let (t_native_s, _, _) = query_native_time(sat, t_j2000_s)?;
        self.records
            .iter()
            .filter(|r| r.satellite_id == sat)
            .filter(|r| r.issue_of_data.message == NavMessage::GpsLnav)
            .filter(|r| r.issue_of_data.issue == u32::from(iode))
            .filter(|r| (t_native_s - toe_native_j2000_s(r)).abs() <= Self::half_window_s(r))
            .min_by(|a, b| {
                let da = (t_native_s - toe_native_j2000_s(a)).abs();
                let db = (t_native_s - toe_native_j2000_s(b)).abs();
                da.partial_cmp(&db).unwrap_or(core::cmp::Ordering::Equal)
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

    /// Select the valid BeiDou record for `sat` of message `nav_message` whose IOD, as
    /// IGS SSR v1.00 (IDF012) defines it for BDS, `mod(toe/720, 240)` with `toe` the BDT
    /// seconds of week, equals `iod`, nearest `t_j2000_s`. SSR orbit corrections name a
    /// BeiDou record by this IOD, not by its AODE.
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
        self.records
            .iter()
            .filter(|r| {
                r.satellite_id == sat
                    && r.message == nav_message
                    && beidou_ssr_iod(r) == Some(iod)
                    && (t_native_s - toe_native_j2000_s(r)).abs() <= Self::half_window_s(r)
            })
            .min_by(|a, b| {
                let da = (t_native_s - toe_native_j2000_s(a)).abs();
                let db = (t_native_s - toe_native_j2000_s(b)).abs();
                da.partial_cmp(&db).unwrap_or(core::cmp::Ordering::Equal)
            })
    }

    /// Position, velocity and clock of the GLONASS record for `sat` whose `tb`, the 15-min
    /// index of its reference epoch in UTC + 3 h, equals `iode`, as RTKLIB `satpos_ssr`
    /// forms them for a GLONASS SSR correction: `selgeph` by that issue (RTKLIB `readrnx`
    /// forms a GLONASS record's IODE as `tb`), then `geph2pos` at `t_j2000_s` and 1 ms
    /// later. The clock is `geph2pos`'s, `-TauN + GammaN·tk` with `tk` not iterated, and
    /// carries no relativistic term, which RTKLIB adds none of for GLONASS. The record's
    /// reference epoch lies within [`GLONASS_SSR_MAX_AGE_S`] of the query, as `selgeph`
    /// bounds it. `None` without a
    /// leap-second offset, which places a GLONASS epoch on the GPS timeline.
    pub(crate) fn glonass_ssr_state(
        &self,
        sat: GnssSatelliteId,
        iode: u32,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], [f64; 3], f64)> {
        let leap = self.leap_seconds?;
        let toe_gpst = |r: &GlonassRecord| r.toe_utc_j2000_s + leap;
        let rec = self
            .glonass
            .iter()
            .filter(|r| {
                r.satellite_id == sat
                    && glonass_tb(r) == Some(iode)
                    && (t_j2000_s - toe_gpst(r)).abs() <= GLONASS_SSR_MAX_AGE_S
            })
            .min_by(|a, b| {
                let da = (t_j2000_s - toe_gpst(a)).abs();
                let db = (t_j2000_s - toe_gpst(b)).abs();
                da.partial_cmp(&db).unwrap_or(core::cmp::Ordering::Equal)
            })?;
        let tk = t_j2000_s - toe_gpst(rec);
        let state0 = [
            rec.pos_m[0],
            rec.pos_m[1],
            rec.pos_m[2],
            rec.vel_m_s[0],
            rec.vel_m_s[1],
            rec.vel_m_s[2],
        ];
        let start = glonass::propagate(state0, rec.acc_m_s2, tk).ok()?;
        let end = glonass::propagate(state0, rec.acc_m_s2, ephpos_stepped_tk(tk)).ok()?;
        let velocity = [
            (end[0] - start[0]) / EPHPOS_STEP_S,
            (end[1] - start[1]) / EPHPOS_STEP_S,
            (end[2] - start[2]) / EPHPOS_STEP_S,
        ];
        let clock = rec.clk_bias + rec.gamma_n * tk;
        Some(([start[0], start[1], start[2]], velocity, clock))
    }

    /// Keep only the records matching a predicate (e.g. a custom message/health
    /// policy on a store built with [`new`](BroadcastStore::new)).
    pub fn retain(&mut self, keep: impl FnMut(&BroadcastRecord) -> bool) {
        self.records.retain(keep);
    }

    /// The half-validity window (seconds either side of `toe`) for a record: half
    /// the broadcast GPS fit interval, or [`MAX_EPHEMERIS_AGE_S`] when no fit
    /// interval is broadcast (Galileo, BeiDou).
    fn half_window_s(rec: &BroadcastRecord) -> f64 {
        match rec.fit_interval_s {
            Some(fit) => fit / 2.0,
            None => MAX_EPHEMERIS_AGE_S,
        }
    }

    /// The record for `sat` whose native-system `toe` is nearest
    /// `t_native_s` **among those whose validity window covers the
    /// query** (see [`half_window_s`](Self::half_window_s)). Filtering by
    /// validity before choosing the nearest means a query just past one record's
    /// fit interval can still be served by a farther record whose own window is
    /// wide enough, rather than being rejected outright.
    fn select(&self, sat: GnssSatelliteId, t_native_s: f64) -> Option<&BroadcastRecord> {
        let mut preferred = None;
        let mut fallback = None;
        for record in self.records.iter().filter(|r| r.satellite_id == sat) {
            if (t_native_s - toe_native_j2000_s(record)).abs() > Self::half_window_s(record) {
                continue;
            }
            if self.is_preferred_family(record) {
                select_better_candidate(&mut preferred, record, t_native_s);
            } else {
                select_better_candidate(&mut fallback, record, t_native_s);
            }
        }
        preferred.or(fallback)
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

    /// Select the broadcast record matching a specific issue and message at the
    /// query epoch.
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
        self.records
            .iter()
            .filter(|r| {
                r.satellite_id == sat
                    && r.message == nav_message
                    && r.issue_of_data == issue
                    && (t_native_s - toe_native_j2000_s(r)).abs() <= Self::half_window_s(r)
            })
            .min_by(|a, b| {
                let da = (t_native_s - toe_native_j2000_s(a)).abs();
                let db = (t_native_s - toe_native_j2000_s(b)).abs();
                da.partial_cmp(&db).unwrap_or(core::cmp::Ordering::Equal)
            })
    }

    /// The GLONASS record for `sat` nearest the GPST-aligned query `t_j2000_s`
    /// (within [`GLONASS_MAX_AGE_S`]), with `tk` = query − the record's reference
    /// epoch in GPS time. Returns `None` if no leap-second offset was parsed (the
    /// GLONASS UTC epoch then cannot be placed on the GPST timeline).
    fn select_glonass(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<(&GlonassRecord, f64)> {
        let leap = self.leap_seconds?;
        let toe_gpst = |r: &GlonassRecord| r.toe_utc_j2000_s + leap;
        let rec = self
            .glonass
            .iter()
            .filter(|r| r.satellite_id == sat)
            .min_by(|a, b| {
                let da = (t_j2000_s - toe_gpst(a)).abs();
                let db = (t_j2000_s - toe_gpst(b)).abs();
                da.partial_cmp(&db).unwrap_or(core::cmp::Ordering::Equal)
            })?;
        let tk = t_j2000_s - toe_gpst(rec);
        if tk.abs() <= GLONASS_MAX_AGE_S {
            Some((rec, tk))
        } else {
            None
        }
    }
}

fn validate_manual_record(record: &BroadcastRecord) -> CoreResult<()> {
    validate_finite(record.toe.tow_s, "record.toe.tow_s")?;
    validate_finite(record.toc.tow_s, "record.toc.tow_s")?;
    validate_finite(record.sv_health, "record.sv_health")?;
    validate_finite(record.sv_accuracy_m, "record.sv_accuracy_m")?;
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
    Some([
        (end[0] - start[0]) / EPHPOS_STEP_S,
        (end[1] - start[1]) / EPHPOS_STEP_S,
        (end[2] - start[2]) / EPHPOS_STEP_S,
    ])
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

fn select_better_candidate<'a>(
    best: &mut Option<&'a BroadcastRecord>,
    candidate: &'a BroadcastRecord,
    t_native_s: f64,
) {
    let Some(current) = *best else {
        *best = Some(candidate);
        return;
    };
    if candidate_is_better(candidate, current, t_native_s) {
        *best = Some(candidate);
    }
}

fn candidate_is_better(
    candidate: &BroadcastRecord,
    current: &BroadcastRecord,
    t_native_s: f64,
) -> bool {
    let da = (t_native_s - toe_native_j2000_s(candidate)).abs();
    let db = (t_native_s - toe_native_j2000_s(current)).abs();
    match da.partial_cmp(&db).unwrap_or(core::cmp::Ordering::Equal) {
        core::cmp::Ordering::Less => true,
        core::cmp::Ordering::Greater => false,
        core::cmp::Ordering::Equal => {
            cnav_tie_rank(candidate.message) < cnav_tie_rank(current.message)
        }
    }
}

const fn cnav_tie_rank(message: NavMessage) -> u8 {
    match message {
        NavMessage::GpsCnav | NavMessage::QzssCnav => 0,
        NavMessage::GpsCnav2 | NavMessage::QzssCnav2 => 1,
        NavMessage::QzssLnav => 0,
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
    /// for `sat` at `t_j2000_s`, from one selection. The group delay is `None` for GLONASS.
    fn state_with_group_delay(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        // GLONASS is not Keplerian: integrate its broadcast state vector with the
        // RK4 propagator. Its reference epoch is UTC, mapped onto the GPST-aligned
        // query via the parsed leap-second offset.
        if sat.system == GnssSystem::Glonass {
            let (rec, tk) = self.select_glonass(sat, t_j2000_s)?;
            let state0 = [
                rec.pos_m[0],
                rec.pos_m[1],
                rec.pos_m[2],
                rec.vel_m_s[0],
                rec.vel_m_s[1],
                rec.vel_m_s[2],
            ];
            let state = glonass::propagate(state0, rec.acc_m_s2, tk).ok()?;
            let clock = glonass::clock_offset_s(rec.clk_bias, rec.gamma_n, tk);
            return Some(([state[0], state[1], state[2]], clock, None));
        }

        // Supported Keplerian systems only; a record from any other system (for
        // example SBAS or NavIC) reports no ephemeris rather than being evaluated
        // with the wrong model. (`from_nav` already restricts records, but `new`
        // accepts arbitrary ones.) The query instant (J2000, GPST-aligned) is read in
        // the satellite system's own scale and seconds of week: BeiDou runs on BDT
        // (= GPST - 14 s), and its geostationary satellites take the GEO orbit branch.
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
    /// GPS and QZSS TGD, Galileo BGD E5b/E1 for I/NAV and E5a/E1 for F/NAV, BeiDou TGD1,
    /// or TGD less ISC L1C/A for CNAV. `None` for GLONASS, whose broadcast clock this
    /// store applies without a group delay.
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
}

/// Largest distance, seconds, between a query and the reference epoch of the GLONASS
/// record an SSR correction's `tb` names: RTKLIB `selgeph`'s `MAXDTOE_GLO`.
const GLONASS_SSR_MAX_AGE_S: f64 = 1800.0;

/// BeiDou SSR IOD of a record, `mod(toe/720, 240)` with `toe` the BDT seconds of week of
/// the ephemeris reference time (IGS SSR v1.00, IDF012). `None` for a record whose `toe`
/// is not a whole number of seconds within the week.
fn beidou_ssr_iod(record: &BroadcastRecord) -> Option<u32> {
    let toe_s = record.elements.toe_sow;
    if !(0.0..crate::constants::SECONDS_PER_WEEK).contains(&toe_s) || toe_s.fract() != 0.0 {
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
/// polynomial and the relativistic term, without the group delay (`eph2pos`: "satellite
/// clock includes relativity correction without code bias (tgd or bgd)"). It is formed
/// as `dt_clock_total_s` forms its first two terms, so subtracting the group delay from
/// it gives `dt_clock_total_s` bit for bit.
fn satposs_clock_s(state: &SatelliteState) -> f64 {
    state.clock.dt_clock_poly_s + state.clock.dt_rel_s
}
