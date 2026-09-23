use sidereon_core::astro::time::model::{GnssWeekTow, TimeScale};
use sidereon_core::constants::{C_M_S, GPS_EPOCH_TO_J2000_S, SECONDS_PER_WEEK};
use sidereon_core::positioning::EphemerisSource;
use sidereon_core::sbas::{
    sbas_prn_to_sat, IssueAwareBroadcast, SbasCorrectedEphemeris, SbasCorrectionStore,
    SbasFastCorrections, SbasLongTermCorrections, SbasLongTermHalf, SbasLongTermRecord,
    SbasMessage, SbasPrnMask, SpareBits,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

struct StaticBroadcast {
    sat: GnssSatelliteId,
    iode: u8,
    state: ([f64; 3], f64),
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

fn epoch_to_j2000_s(epoch: GnssWeekTow) -> f64 {
    f64::from(epoch.week) * SECONDS_PER_WEEK + epoch.tow_s - GPS_EPOCH_TO_J2000_S
}

#[test]
fn fast_and_long_term_corrections_apply_with_expected_signs() {
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid GPS PRN");
    let geo = sbas_prn_to_sat(120).expect("valid source GEO");
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

    let half = SbasLongTermHalf {
        velocity_code: false,
        iodp: 1,
        records: vec![
            SbasLongTermRecord {
                monitored_index: 1,
                iode: 7,
                delta_x: 8,
                delta_y: 16,
                delta_z: 24,
                delta_x_rate: 0,
                delta_y_rate: 0,
                delta_z_rate: 0,
                delta_a_f0: 0,
                delta_a_f1: 0,
                time_of_day_s: None,
            },
            unused_record(),
        ],
        reserved: SpareBits(vec![(0, 1)]),
    };
    store
        .ingest(
            &SbasMessage::LongTermCorrections(SbasLongTermCorrections {
                preamble: 0x53,
                halves: [
                    half,
                    SbasLongTermHalf {
                        velocity_code: false,
                        iodp: 0,
                        records: vec![unused_record(), unused_record()],
                        reserved: SpareBits(vec![(0, 1)]),
                    },
                ],
            }),
            geo,
            epoch(20.0),
        )
        .unwrap();

    let broadcast = StaticBroadcast {
        sat,
        iode: 7,
        state: ([10.0, 20.0, 30.0], 0.25),
    };
    let source = SbasCorrectedEphemeris::new(&broadcast, &store, geo);
    let (position, clock) = source
        .position_clock_at_j2000_s(sat, epoch_to_j2000_s(epoch(20.0)))
        .expect("corrected state");
    assert_eq!(position, [11.0, 22.0, 33.0]);
    assert!((clock - (0.25 + 1.0 / C_M_S)).abs() < 1.0e-15);
}

/// A record slot a half without the velocity code leaves unused: mask index 0,
/// every other field zero, as the wire carries it.
fn unused_record() -> SbasLongTermRecord {
    SbasLongTermRecord {
        monitored_index: 0,
        iode: 0,
        delta_x: 0,
        delta_y: 0,
        delta_z: 0,
        delta_x_rate: 0,
        delta_y_rate: 0,
        delta_z_rate: 0,
        delta_a_f0: 0,
        delta_a_f1: 0,
        time_of_day_s: None,
    }
}
