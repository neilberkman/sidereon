use std::sync::Arc;

use crate::constants::C_M_S;
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::observables::{ObservableEphemerisSource, ObservableState, ObservablesError};
use crate::spp::EphemerisSource;

use super::store::{SbasCorrectionStore, SbasIonoGrid};

/// An [`EphemerisSource`] that can select a broadcast state by issue of data.
///
/// [`SbasCorrectedEphemeris`] calls the issue-specific lookup when a fresh GPS
/// long-term correction supplies the broadcast IODE. A failed lookup prevents
/// that correction branch from producing a state.
pub trait IssueAwareBroadcast: EphemerisSource {
    /// Return the ECEF position in meters and satellite clock offset in seconds
    /// for the requested broadcast IODE at seconds since J2000.
    ///
    /// Return `None` when no usable record for `sat` has the requested IODE at
    /// the query epoch.
    fn state_by_iode_at(
        &self,
        sat: GnssSatelliteId,
        iode: u8,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)>;

    /// [`Self::state_by_iode_at`] with the single-frequency group delay of the same
    /// record (see [`EphemerisSource::single_frequency_group_delay_s`]). The default takes
    /// the delay of the record the source selects by time at `t_j2000_s`, which is the IODE
    /// record wherever the two coincide; a source that can select by IODE overrides it.
    fn state_group_delay_by_iode_at(
        &self,
        sat: GnssSatelliteId,
        iode: u8,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        let (position, clock) = self.state_by_iode_at(sat, iode, t_j2000_s)?;
        Some((
            position,
            clock,
            self.single_frequency_group_delay_s(sat, t_j2000_s),
        ))
    }

    /// Velocity, metres per second, of the record with broadcast IODE `iode` at
    /// `t_j2000_s`, when the source defines one. `None` by default.
    fn velocity_by_iode_at(
        &self,
        _sat: GnssSatelliteId,
        _iode: u8,
        _t_j2000_s: f64,
    ) -> Option<[f64; 3]> {
        None
    }

    /// Velocity, metres per second, of the record [`EphemerisSource::position_clock_at_j2000_s`]
    /// uses at `t_j2000_s`, when the source defines one. `None` by default.
    fn broadcast_velocity_at(&self, _sat: GnssSatelliteId, _t_j2000_s: f64) -> Option<[f64; 3]> {
        None
    }
}

/// Selects the branch used when no complete or permitted partial SBAS
/// correction is available.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SbasSolveMode {
    /// Use the underlying broadcast state when correction data cannot be applied.
    #[default]
    MixedAugmentation,
    /// Return no state when correction data cannot be applied.
    SbasOnly,
}

/// Borrowed SBAS correction inputs and the resulting fallback behavior for one
/// source GEO.
///
/// `corrected_state` rejects a disabled source GEO or withdrawn satellite. GPS
/// states use fast and long-term corrections when both are fresh, or one
/// correction when the store permits partial use. The selected GEO uses fresh
/// GEO navigation plus its fast clock delta; without a fresh fast correction
/// it, like every other unresolved query, follows [`SbasSolveMode`]: the
/// uncorrected navigation state under
/// [`SbasSolveMode::MixedAugmentation`], no state under
/// [`SbasSolveMode::SbasOnly`].
pub struct SbasCorrectedEphemeris<'a> {
    broadcast: &'a dyn IssueAwareBroadcast,
    store: &'a SbasCorrectionStore,
    geo: GnssSatelliteId,
    mode: SbasSolveMode,
}

impl<'a> SbasCorrectedEphemeris<'a> {
    /// Retain `broadcast`, `store`, and `geo` by reference and initialize the
    /// fallback mode to [`SbasSolveMode::MixedAugmentation`].
    pub fn new(
        broadcast: &'a dyn IssueAwareBroadcast,
        store: &'a SbasCorrectionStore,
        geo: GnssSatelliteId,
    ) -> Self {
        Self {
            broadcast,
            store,
            geo,
            mode: SbasSolveMode::MixedAugmentation,
        }
    }

    /// Call [`SbasCorrectionStore::ready_geos`] at `t_j2000_s` and select its
    /// first GEO.
    ///
    /// The store orders ready GEOs by newest partition update, so this returns
    /// the newest eligible source or `None` when no GEO is ready.
    pub fn with_preferred_geo(
        broadcast: &'a dyn IssueAwareBroadcast,
        store: &'a SbasCorrectionStore,
        t_j2000_s: f64,
    ) -> Option<Self> {
        let geo = store.ready_geos(t_j2000_s).first().copied()?;
        Some(Self::new(broadcast, store, geo))
    }

    /// Replace the fallback mode used when a complete or permitted partial
    /// correction is unavailable.
    pub fn with_mode(mut self, mode: SbasSolveMode) -> Self {
        self.mode = mode;
        self
    }

    /// Delegate to [`SbasCorrectionStore::iono_grid`] for the selected GEO.
    ///
    /// `None` indicates that the GEO partition is absent or that its latest
    /// update has disabled the grid.
    pub fn iono_grid(&self) -> Option<&SbasIonoGrid> {
        self.store.iono_grid(self.geo)
    }

    fn corrected_state(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<([f64; 3], f64)> {
        self.corrected_state_with_group_delay(sat, t_j2000_s)
            .map(|(position, clock, _)| (position, clock))
    }

    /// [`Self::corrected_state`] with the single-frequency group delay of the broadcast
    /// record the state starts from: the record the long-term correction's IODE selects,
    /// or the one the broadcast state uses. RTKLIB `pntpos` applies that record's TGD to an
    /// SBAS-corrected solution, as `satpos_sbas` builds on `ephpos`, whose clock has none.
    /// `None` for the SBAS GEO, whose navigation clock carries no group delay.
    fn corrected_state_with_group_delay(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        if self.store.is_disabled(self.geo, t_j2000_s) || self.store.is_withdrawn(self.geo, sat) {
            return None;
        }

        if sat == self.geo {
            let geo_state = self.store.fresh_geo_nav(self.geo, t_j2000_s)?;
            let (position, clock) = geo_state.state_at(t_j2000_s);
            // Without a fresh fast correction the GEO's own navigation state
            // is its broadcast state, used or refused as the mode says, like
            // any other satellite without a correction. Its navigation clock
            // carries no group delay.
            return match self.fast_clock_delta_s(sat, t_j2000_s) {
                Some(delta_s) => Some((position, clock + delta_s, None)),
                None => match self.mode {
                    SbasSolveMode::MixedAugmentation => Some((position, clock, None)),
                    SbasSolveMode::SbasOnly => None,
                },
            };
        }

        let fast = self.store.fresh_fast(self.geo, sat, t_j2000_s);
        let long = (sat.system == GnssSystem::Gps)
            .then(|| self.store.fresh_long_term(self.geo, sat, t_j2000_s))
            .flatten();

        match (fast, long) {
            (Some(fast), Some(long)) => {
                let (mut position, mut clock, group_delay) = self
                    .broadcast
                    .state_group_delay_by_iode_at(sat, long.iode, t_j2000_s)?;
                let dt = t_j2000_s - long.t0_j2000_s;
                for (i, component) in position.iter_mut().enumerate() {
                    *component += long.delta_ecef_m[i] + long.delta_ecef_rate_m_s[i] * dt;
                }
                clock += long.delta_af0_s + long.delta_af1_s_s * dt;
                clock += (fast.prc_m + fast.rrc_m_s * (t_j2000_s - fast.t_of_j2000_s)) / C_M_S;
                Some((position, clock, group_delay))
            }
            (Some(fast), None) if self.store.allow_partial_corrections() => {
                let (position, mut clock, group_delay) = self
                    .broadcast
                    .position_clock_group_delay_at_j2000_s(sat, t_j2000_s)?;
                clock += (fast.prc_m + fast.rrc_m_s * (t_j2000_s - fast.t_of_j2000_s)) / C_M_S;
                Some((position, clock, group_delay))
            }
            (None, Some(long)) if self.store.allow_partial_corrections() => {
                let (mut position, mut clock, group_delay) = self
                    .broadcast
                    .state_group_delay_by_iode_at(sat, long.iode, t_j2000_s)?;
                let dt = t_j2000_s - long.t0_j2000_s;
                for (i, component) in position.iter_mut().enumerate() {
                    *component += long.delta_ecef_m[i] + long.delta_ecef_rate_m_s[i] * dt;
                }
                clock += long.delta_af0_s + long.delta_af1_s_s * dt;
                Some((position, clock, group_delay))
            }
            _ => match self.mode {
                SbasSolveMode::MixedAugmentation => self
                    .broadcast
                    .position_clock_group_delay_at_j2000_s(sat, t_j2000_s),
                SbasSolveMode::SbasOnly => None,
            },
        }
    }

    /// Velocity of the state [`Self::corrected_state`] returns, as RTKLIB `satpos_sbas`
    /// takes it: the broadcast velocity of the record the state starts from (by the
    /// long-term correction's IODE, or the one the broadcast state uses), with no
    /// long-term velocity correction added, and for the GEO the difference of its
    /// navigation state at `t_j2000_s` and 1 ms later. `None` inside the result when the
    /// source returns no state; `None` outside it when the broadcast source defines no
    /// record velocity.
    fn corrected_velocity(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<Result<[f64; 3], ObservablesError>> {
        const STEP_S: f64 = crate::rinex_nav::EPHPOS_STEP_S;
        if self.store.is_disabled(self.geo, t_j2000_s) || self.store.is_withdrawn(self.geo, sat) {
            return Some(Err(ObservablesError::NoEphemeris));
        }
        if sat == self.geo {
            let Some(geo_state) = self
                .store
                .fresh_geo_nav(self.geo, t_j2000_s)
                .filter(|_| self.corrected_state(sat, t_j2000_s).is_some())
            else {
                return Some(Err(ObservablesError::NoEphemeris));
            };
            // The step is added to the time from the navigation reference epoch, as
            // RTKLIB `seph2pos` takes it from its exact `gtime_t`.
            let dt = t_j2000_s - geo_state.t0_j2000_s;
            let (start, _) = geo_state.state_after(dt);
            let (end, _) = geo_state.state_after(dt + STEP_S);
            return Some(Ok([
                (end[0] - start[0]) / STEP_S,
                (end[1] - start[1]) / STEP_S,
                (end[2] - start[2]) / STEP_S,
            ]));
        }
        if self.corrected_state(sat, t_j2000_s).is_none() {
            return Some(Err(ObservablesError::NoEphemeris));
        }
        let fast = self.store.fresh_fast(self.geo, sat, t_j2000_s);
        let long = (sat.system == GnssSystem::Gps)
            .then(|| self.store.fresh_long_term(self.geo, sat, t_j2000_s))
            .flatten();
        let velocity = match (fast, long) {
            (Some(_), Some(long)) => self
                .broadcast
                .velocity_by_iode_at(sat, long.iode, t_j2000_s),
            (None, Some(long)) if self.store.allow_partial_corrections() => self
                .broadcast
                .velocity_by_iode_at(sat, long.iode, t_j2000_s),
            _ => self.broadcast.broadcast_velocity_at(sat, t_j2000_s),
        };
        velocity.map(Ok)
    }

    /// Group delay a single-frequency model subtracts from the clock of the state
    /// [`EphemerisSource::position_clock_at_j2000_s`] returns: the broadcast group delay
    /// of the record that state starts from. `None` for the SBAS GEO and where no state is
    /// returned.
    pub fn single_frequency_group_delay_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<f64> {
        self.corrected_state_with_group_delay(sat, t_j2000_s)?.2
    }

    fn fast_clock_delta_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        let fast = self.store.fresh_fast(self.geo, sat, t_j2000_s)?;
        Some((fast.prc_m + fast.rrc_m_s * (t_j2000_s - fast.t_of_j2000_s)) / C_M_S)
    }
}

impl EphemerisSource for SbasCorrectedEphemeris<'_> {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        self.corrected_state(sat, t_j2000_s)
    }

    fn single_frequency_group_delay_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        SbasCorrectedEphemeris::single_frequency_group_delay_s(self, sat, t_j2000_s)
    }

    fn position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        self.corrected_state_with_group_delay(sat, t_j2000_s)
    }

    /// The one-evaluation read above; an SBAS-corrected source never refuses a state.
    fn try_position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> crate::Result<Option<crate::astro::time::Validated<crate::spp::PositionClockGroupDelay>>>
    {
        Ok(self
            .position_clock_group_delay_at_j2000_s(sat, t_j2000_s)
            .map(crate::astro::time::Validated::ok))
    }
}

impl ObservableEphemerisSource for SbasCorrectedEphemeris<'_> {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        let Some((position_ecef_m, clock_s)) = self.corrected_state(sat, t_j2000_s) else {
            return Err(ObservablesError::NoEphemeris);
        };
        Ok(ObservableState {
            position_ecef_m,
            clock_s: Some(clock_s),
        })
    }

    fn velocity_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<Result<[f64; 3], ObservablesError>> {
        self.corrected_velocity(sat, t_j2000_s)
    }

    /// True: an SBAS-corrected clock is the broadcast clock, which carries the broadcast
    /// relativistic term, plus the SBAS clock corrections, as RTKLIB `satpos_sbas` builds
    /// it on `ephpos`.
    fn clock_includes_relativity(&self) -> bool {
        true
    }

    fn single_frequency_group_delay_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        SbasCorrectedEphemeris::single_frequency_group_delay_s(self, sat, t_j2000_s)
    }

    fn observable_state_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<(ObservableState, Option<f64>), ObservablesError> {
        let (position_ecef_m, clock_s, group_delay) = self
            .corrected_state_with_group_delay(sat, t_j2000_s)
            .ok_or(ObservablesError::NoEphemeris)?;
        Ok((
            ObservableState {
                position_ecef_m,
                clock_s: Some(clock_s),
            },
            group_delay,
        ))
    }

    /// The one-evaluation read above; an SBAS-corrected source never refuses a state.
    fn try_observable_state_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<crate::astro::time::Validated<(ObservableState, Option<f64>)>, ObservablesError>
    {
        ObservableEphemerisSource::observable_state_group_delay_at_j2000_s(self, sat, t_j2000_s)
            .map(crate::astro::time::Validated::ok)
    }
}

/// Owns thread-safe SBAS correction inputs for one source GEO.
///
/// The [`EphemerisSource`] and [`ObservableEphemerisSource`] implementations
/// delegate to a borrowed [`SbasCorrectedEphemeris`] with the same GEO and
/// fallback mode.
pub struct SbasCorrectedEphemerisOwned {
    broadcast: Arc<dyn IssueAwareBroadcast + Send + Sync>,
    store: Arc<SbasCorrectionStore>,
    geo: GnssSatelliteId,
    mode: SbasSolveMode,
}

impl SbasCorrectedEphemerisOwned {
    /// Store the supplied reference-counted inputs and `geo`, initializing the
    /// fallback mode to [`SbasSolveMode::MixedAugmentation`].
    pub fn new(
        broadcast: Arc<dyn IssueAwareBroadcast + Send + Sync>,
        store: Arc<SbasCorrectionStore>,
        geo: GnssSatelliteId,
    ) -> Self {
        Self {
            broadcast,
            store,
            geo,
            mode: SbasSolveMode::MixedAugmentation,
        }
    }

    /// Replace the fallback mode used when a complete or permitted partial
    /// correction is unavailable.
    pub fn with_mode(mut self, mode: SbasSolveMode) -> Self {
        self.mode = mode;
        self
    }

    /// Delegate to the owned correction store's [`SbasCorrectionStore::iono_grid`]
    /// lookup for the selected GEO.
    pub fn iono_grid(&self) -> Option<&SbasIonoGrid> {
        self.store.iono_grid(self.geo)
    }

    fn borrowed(&self) -> SbasCorrectedEphemeris<'_> {
        SbasCorrectedEphemeris::new(self.broadcast.as_ref(), self.store.as_ref(), self.geo)
            .with_mode(self.mode)
    }
}

impl EphemerisSource for SbasCorrectedEphemerisOwned {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        self.borrowed().position_clock_at_j2000_s(sat, t_j2000_s)
    }

    fn single_frequency_group_delay_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        self.borrowed()
            .single_frequency_group_delay_s(sat, t_j2000_s)
    }

    fn position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        self.borrowed()
            .corrected_state_with_group_delay(sat, t_j2000_s)
    }

    /// The one-evaluation read above; an SBAS-corrected source never refuses a state.
    fn try_position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> crate::Result<Option<crate::astro::time::Validated<crate::spp::PositionClockGroupDelay>>>
    {
        Ok(self
            .position_clock_group_delay_at_j2000_s(sat, t_j2000_s)
            .map(crate::astro::time::Validated::ok))
    }
}

impl ObservableEphemerisSource for SbasCorrectedEphemerisOwned {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        self.borrowed().observable_state_at_j2000_s(sat, t_j2000_s)
    }

    fn velocity_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<Result<[f64; 3], ObservablesError>> {
        self.borrowed().velocity_at_j2000_s(sat, t_j2000_s)
    }

    fn clock_includes_relativity(&self) -> bool {
        self.borrowed().clock_includes_relativity()
    }

    fn single_frequency_group_delay_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        self.borrowed()
            .single_frequency_group_delay_s(sat, t_j2000_s)
    }

    fn observable_state_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<(ObservableState, Option<f64>), ObservablesError> {
        ObservableEphemerisSource::observable_state_group_delay_at_j2000_s(
            &self.borrowed(),
            sat,
            t_j2000_s,
        )
    }

    /// The one-evaluation read above; an SBAS-corrected source never refuses a state.
    fn try_observable_state_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<crate::astro::time::Validated<(ObservableState, Option<f64>)>, ObservablesError>
    {
        ObservableEphemerisSource::observable_state_group_delay_at_j2000_s(self, sat, t_j2000_s)
            .map(crate::astro::time::Validated::ok)
    }
}

impl IssueAwareBroadcast for crate::rinex_nav::BroadcastStore {
    fn state_by_iode_at(
        &self,
        sat: GnssSatelliteId,
        iode: u8,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        crate::rinex_nav::BroadcastStore::state_by_iode_at(self, sat, iode, t_j2000_s)
    }

    fn state_group_delay_by_iode_at(
        &self,
        sat: GnssSatelliteId,
        iode: u8,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        crate::rinex_nav::BroadcastStore::state_group_delay_by_iode_at(self, sat, iode, t_j2000_s)
    }

    fn velocity_by_iode_at(
        &self,
        sat: GnssSatelliteId,
        iode: u8,
        t_j2000_s: f64,
    ) -> Option<[f64; 3]> {
        self.iode_record_velocity(sat, iode, t_j2000_s)
    }

    fn broadcast_velocity_at(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<[f64; 3]> {
        self.selected_record_velocity(sat, t_j2000_s)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::astro::time::model::{GnssWeekTow, TimeScale};
    use crate::sbas::message::{
        SbasDoNotUse, SbasFastCorrections, SbasGeoNav, SbasIgpDelay, SbasIgpMask, SbasIonoDelays,
        SbasMessage, SbasPrnMask, SpareBits,
    };
    use crate::sbas::store::{sbas_prn_to_sat, SbasLongTermCorrection};

    struct StaticBroadcast {
        sat: GnssSatelliteId,
        state: ([f64; 3], f64),
        iode: u8,
    }

    impl EphemerisSource for StaticBroadcast {
        fn position_clock_at_j2000_s(
            &self,
            sat: GnssSatelliteId,
            _t_j2000_s: f64,
        ) -> Option<([f64; 3], f64)> {
            (sat == self.sat).then_some(self.state)
        }
    }

    impl IssueAwareBroadcast for StaticBroadcast {
        fn state_by_iode_at(
            &self,
            sat: GnssSatelliteId,
            iode: u8,
            t_j2000_s: f64,
        ) -> Option<([f64; 3], f64)> {
            (iode == self.iode)
                .then(|| self.position_clock_at_j2000_s(sat, t_j2000_s))
                .flatten()
        }
    }

    fn epoch(tow_s: f64) -> GnssWeekTow {
        GnssWeekTow::new(TimeScale::Gpst, 2400, tow_s).expect("valid epoch")
    }

    #[test]
    fn mixed_mode_falls_back_to_broadcast_without_complete_correction() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid GPS PRN");
        let geo = sbas_prn_to_sat(120).unwrap();
        let broadcast = StaticBroadcast {
            sat,
            state: ([1.0, 2.0, 3.0], 4.0),
            iode: 7,
        };
        let store = SbasCorrectionStore::new();
        let source = SbasCorrectedEphemeris::new(&broadcast, &store, geo);
        assert_eq!(
            source.position_clock_at_j2000_s(sat, 0.0),
            Some(([1.0, 2.0, 3.0], 4.0))
        );
        assert_eq!(
            source
                .with_mode(SbasSolveMode::SbasOnly)
                .position_clock_at_j2000_s(sat, 0.0),
            None
        );
    }

    /// With no correction, the mixed-mode source returns the broadcast state, and its
    /// velocity is RTKLIB `ephpos`'s for that record: the positions at `tk` and `tk` + 1 ms,
    /// the step added to the reduced time. Added to the 2026 J2000 epoch instead, the step
    /// would not even be 1 ms.
    #[test]
    fn catch_all_velocity_is_the_broadcast_record_ephpos_velocity() {
        let nav = crate::rinex_nav::BroadcastStore::from_nav(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
        )))
        .expect("parse NAV fixture");
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 31).expect("valid GPS PRN");
        let geo = sbas_prn_to_sat(120).unwrap();
        let store = SbasCorrectionStore::new();
        let source = SbasCorrectedEphemeris::new(&nav, &store, geo);
        let t = 836_222_417.0;
        assert_ne!((t + 1.0e-3) - t, 1.0e-3, "the absolute step is inexact");

        let velocity = source
            .velocity_at_j2000_s(sat, t)
            .expect("the source defines its velocity")
            .expect("broadcast velocity");
        let record = nav.select_record_at(sat, t).expect("broadcast record");
        let sow = (t + crate::constants::GPS_EPOCH_TO_J2000_S)
            .rem_euclid(crate::constants::SECONDS_PER_WEEK);
        let tk = sow - record.elements.toe_sow;
        let position = |tk_s: f64| {
            crate::broadcast::satellite_position_ecef_at_tk_unchecked(
                &record.elements,
                None,
                &record.constants(),
                tk_s,
                false,
            )
            .position()
            .expect("finite position")
            .as_array()
        };
        let (start, end) = (position(tk), position(tk + 1.0e-3));
        for axis in 0..3 {
            let expected = (end[axis] - start[axis]) / 1.0e-3;
            assert_eq!(velocity[axis].to_bits(), expected.to_bits(), "axis {axis}");
        }
    }

    #[test]
    fn withdrawn_satellite_returns_no_state() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid GPS PRN");
        let geo = sbas_prn_to_sat(120).unwrap();
        let mut store = SbasCorrectionStore::new();
        let mut mask = [false; 210];
        mask[0] = true;
        store
            .ingest(
                &SbasMessage::PrnMask(SbasPrnMask {
                    preamble: 0x53,
                    iodp: 1,
                    mask,
                    reserved: SpareBits::new(),
                }),
                geo,
                epoch(10.0),
            )
            .unwrap();
        let mut udrei = [0u8; 13];
        udrei[0] = 15;
        store
            .ingest(
                &SbasMessage::FastCorrections(SbasFastCorrections {
                    preamble: 0x53,
                    message_type: 2,
                    iodf: 1,
                    iodp: 1,
                    prc: [0; 13],
                    udrei,
                    reserved: SpareBits::new(),
                }),
                geo,
                epoch(20.0),
            )
            .unwrap();
        let broadcast = StaticBroadcast {
            sat,
            state: ([1.0, 2.0, 3.0], 4.0),
            iode: 7,
        };
        let source = SbasCorrectedEphemeris::new(&broadcast, &store, geo);
        assert_eq!(source.position_clock_at_j2000_s(sat, 0.0), None);
    }

    #[test]
    fn complete_correction_adds_position_and_clock_terms() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid GPS PRN");
        let geo = sbas_prn_to_sat(120).unwrap();
        let broadcast = StaticBroadcast {
            sat,
            state: ([1.0, 2.0, 3.0], 4.0),
            iode: 9,
        };
        let mut store = SbasCorrectionStore::new();
        let mut mask = [false; 210];
        mask[0] = true;
        store
            .ingest(
                &SbasMessage::PrnMask(SbasPrnMask {
                    preamble: 0x53,
                    iodp: 1,
                    mask,
                    reserved: SpareBits::new(),
                }),
                geo,
                epoch(10.0),
            )
            .unwrap();
        let mut prc = [0i16; 13];
        prc[0] = 8;
        store
            .ingest(
                &SbasMessage::FastCorrections(SbasFastCorrections {
                    preamble: 0x53,
                    message_type: 2,
                    iodf: 1,
                    iodp: 1,
                    prc,
                    udrei: [0; 13],
                    reserved: SpareBits::new(),
                }),
                geo,
                epoch(20.0),
            )
            .unwrap();
        store.insert_long_term_for_test(
            geo,
            sat,
            SbasLongTermCorrection {
                iode: 9,
                delta_ecef_m: [10.0, 20.0, 30.0],
                delta_ecef_rate_m_s: [1.0, 0.0, 0.0],
                delta_af0_s: 0.5,
                delta_af1_s_s: 0.0,
                t0_j2000_s: epoch_to_j2000_s_for_test(epoch(20.0)),
            },
            epoch_to_j2000_s_for_test(epoch(20.0)),
        );
        let source = SbasCorrectedEphemeris::new(&broadcast, &store, geo);
        let t = epoch_to_j2000_s_for_test(epoch(21.0));
        let (pos, clock) = source.position_clock_at_j2000_s(sat, t).unwrap();
        assert_eq!(pos, [12.0, 22.0, 33.0]);
        assert!((clock - (4.5 + 1.0 / C_M_S)).abs() < 1.0e-15);
    }

    #[test]
    fn fresh_mt0_withholds_iono_grid_from_corrected_source() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid GPS PRN");
        let geo = sbas_prn_to_sat(120).unwrap();
        let broadcast = StaticBroadcast {
            sat,
            state: ([1.0, 2.0, 3.0], 4.0),
            iode: 7,
        };
        let mut store = SbasCorrectionStore::new();
        let mut mask = [false; 201];
        mask[0] = true;
        store
            .ingest(
                &SbasMessage::IgpMask(SbasIgpMask {
                    preamble: 0x53,
                    band_number: 0,
                    iodi: 1,
                    mask,
                    reserved: SpareBits(vec![(0, 4), (0, 1)]),
                }),
                geo,
                epoch(1.0),
            )
            .unwrap();
        let mut entries: [SbasIgpDelay; 15] = core::array::from_fn(|_| SbasIgpDelay::default());
        entries[0] = SbasIgpDelay {
            vertical_delay: 8,
            givei: 0,
        };
        store
            .ingest(
                &SbasMessage::IonoDelays(SbasIonoDelays {
                    preamble: 0x53,
                    band_number: 0,
                    block_id: 0,
                    iodi: 1,
                    entries,
                    reserved: SpareBits(vec![(0, 7)]),
                }),
                geo,
                epoch(2.0),
            )
            .unwrap();
        let source = SbasCorrectedEphemeris::new(&broadcast, &store, geo);
        assert!(source.iono_grid().is_some());

        store
            .ingest(
                &SbasMessage::DoNotUse(SbasDoNotUse {
                    preamble: 0x53,
                    data: vec![0; 27],
                }),
                geo,
                epoch(3.0),
            )
            .unwrap();
        let source = SbasCorrectedEphemeris::new(&broadcast, &store, geo);
        assert!(source.iono_grid().is_none());
    }

    /// A GEO without a fresh fast correction has only its broadcast
    /// navigation state, which the mode decides on as for any other satellite,
    /// rather than that state with a zero fast correction added.
    #[test]
    fn geo_without_fast_correction_follows_the_mode() {
        let geo = sbas_prn_to_sat(120).unwrap();
        let other = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid GPS PRN");
        let broadcast = StaticBroadcast {
            sat: other,
            state: ([1.0, 2.0, 3.0], 4.0),
            iode: 7,
        };
        let mut store = SbasCorrectionStore::new();
        store
            .ingest(
                &SbasMessage::GeoNav(SbasGeoNav {
                    preamble: 0x9A,
                    time_of_day_s: 1,
                    ura: 0,
                    x_m: 100,
                    y_m: 200,
                    z_m: 300,
                    x_rate_m_s: 0,
                    y_rate_m_s: 0,
                    z_rate_m_s: 0,
                    x_accel_m_s2: 0,
                    y_accel_m_s2: 0,
                    z_accel_m_s2: 0,
                    a_gf0_s: 64,
                    a_gf1_s_s: 0,
                    reserved: SpareBits(vec![(0, 8)]),
                }),
                geo,
                epoch(16.0),
            )
            .unwrap();
        let t = epoch_to_j2000_s_for_test(epoch(16.0));
        let expected = store.geo_nav(geo).expect("GEO nav").state_at(t);
        let source = SbasCorrectedEphemeris::new(&broadcast, &store, geo);
        assert_eq!(source.position_clock_at_j2000_s(geo, t), Some(expected));
        let sbas_only = source.with_mode(SbasSolveMode::SbasOnly);
        assert_eq!(sbas_only.position_clock_at_j2000_s(geo, t), None);
        assert!(matches!(
            sbas_only.velocity_at_j2000_s(geo, t),
            Some(Err(ObservablesError::NoEphemeris))
        ));
    }

    fn epoch_to_j2000_s_for_test(epoch: GnssWeekTow) -> f64 {
        f64::from(epoch.week) * crate::constants::SECONDS_PER_WEEK + epoch.tow_s
            - crate::constants::GPS_EPOCH_TO_J2000_S
    }
}
