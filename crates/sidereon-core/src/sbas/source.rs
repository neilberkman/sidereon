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
/// GEO navigation plus its fast clock delta; other unresolved queries follow
/// [`SbasSolveMode`].
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
        if self.store.is_disabled(self.geo, t_j2000_s) || self.store.is_withdrawn(self.geo, sat) {
            return None;
        }

        if sat == self.geo {
            let geo_state = self.store.fresh_geo_nav(self.geo, t_j2000_s)?;
            let (position, clock) = geo_state.state_at(t_j2000_s);
            let clock = clock + self.fast_clock_delta_s(sat, t_j2000_s).unwrap_or(0.0);
            return Some((position, clock));
        }

        let fast = self.store.fresh_fast(self.geo, sat, t_j2000_s);
        let long = (sat.system == GnssSystem::Gps)
            .then(|| self.store.fresh_long_term(self.geo, sat, t_j2000_s))
            .flatten();

        match (fast, long) {
            (Some(fast), Some(long)) => {
                let (mut position, mut clock) =
                    self.broadcast.state_by_iode_at(sat, long.iode, t_j2000_s)?;
                let dt = t_j2000_s - long.t0_j2000_s;
                for (i, component) in position.iter_mut().enumerate() {
                    *component += long.delta_ecef_m[i] + long.delta_ecef_rate_m_s[i] * dt;
                }
                clock += long.delta_af0_s + long.delta_af1_s_s * dt;
                clock += (fast.prc_m + fast.rrc_m_s * (t_j2000_s - fast.t_of_j2000_s)) / C_M_S;
                Some((position, clock))
            }
            (Some(fast), None) if self.store.allow_partial_corrections() => {
                let (position, mut clock) =
                    self.broadcast.position_clock_at_j2000_s(sat, t_j2000_s)?;
                clock += (fast.prc_m + fast.rrc_m_s * (t_j2000_s - fast.t_of_j2000_s)) / C_M_S;
                Some((position, clock))
            }
            (None, Some(long)) if self.store.allow_partial_corrections() => {
                let (mut position, mut clock) =
                    self.broadcast.state_by_iode_at(sat, long.iode, t_j2000_s)?;
                let dt = t_j2000_s - long.t0_j2000_s;
                for (i, component) in position.iter_mut().enumerate() {
                    *component += long.delta_ecef_m[i] + long.delta_ecef_rate_m_s[i] * dt;
                }
                clock += long.delta_af0_s + long.delta_af1_s_s * dt;
                Some((position, clock))
            }
            _ => match self.mode {
                SbasSolveMode::MixedAugmentation => {
                    self.broadcast.position_clock_at_j2000_s(sat, t_j2000_s)
                }
                SbasSolveMode::SbasOnly => None,
            },
        }
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
}

impl ObservableEphemerisSource for SbasCorrectedEphemerisOwned {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        self.borrowed().observable_state_at_j2000_s(sat, t_j2000_s)
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
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::astro::time::model::{GnssWeekTow, TimeScale};
    use crate::sbas::message::{
        SbasDoNotUse, SbasFastCorrections, SbasIgpDelay, SbasIgpMask, SbasIonoDelays, SbasMessage,
        SbasPrnMask, SpareBits,
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
                    reserved: SpareBits::new(),
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
                    reserved: SpareBits::new(),
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
                    data: Vec::new(),
                }),
                geo,
                epoch(3.0),
            )
            .unwrap();
        let source = SbasCorrectedEphemeris::new(&broadcast, &store, geo);
        assert!(source.iono_grid().is_none());
    }

    fn epoch_to_j2000_s_for_test(epoch: GnssWeekTow) -> f64 {
        f64::from(epoch.week) * crate::constants::SECONDS_PER_WEEK + epoch.tow_s
            - crate::constants::GPS_EPOCH_TO_J2000_S
    }
}
