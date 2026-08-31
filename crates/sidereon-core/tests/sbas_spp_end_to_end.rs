use std::collections::BTreeMap;

use sidereon_core::constants::{C_M_S, F_L1_HZ};
use sidereon_core::frame::Wgs84Geodetic;
use sidereon_core::positioning::{
    solve, Corrections, EphemerisSource, KlobucharCoeffs, Observation, SolveInputs, SurfaceMet,
};
use sidereon_core::sbas::{
    sbas_prn_to_sat, IssueAwareBroadcast, SbasBlock, SbasCorrectedEphemeris, SbasCorrectionStore,
    SbasFastCorrections, SbasGeoState, SbasIgp, SbasIonoGrid, SbasLongTermCorrections,
    SbasLongTermHalf, SbasLongTermRecord, SbasMessage, SbasPrnMask, SbasWireForm, SpareBits,
};
use sidereon_core::{geodetic_to_itrf, GnssSatelliteId, GnssSystem};

const OMEGA_E_DOT_RAD_S: f64 = 7.292_115_146_7e-5;

#[derive(Clone, Copy)]
struct StaticState {
    position_ecef_m: [f64; 3],
    clock_s: f64,
    iode: u8,
}

struct StaticBroadcast {
    states: BTreeMap<GnssSatelliteId, StaticState>,
    dynamic_geo: Option<(GnssSatelliteId, SbasGeoState)>,
}

impl EphemerisSource for StaticBroadcast {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        if let Some((geo, state)) = &self.dynamic_geo {
            if *geo == sat {
                return Some(state.state_at(t_j2000_s));
            }
        }
        let state = self.states.get(&sat)?;
        Some((state.position_ecef_m, state.clock_s))
    }
}

impl IssueAwareBroadcast for StaticBroadcast {
    fn state_by_iode_at(
        &self,
        sat: GnssSatelliteId,
        iode: u8,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        let state = self.states.get(&sat)?;
        (state.iode == iode)
            .then(|| self.position_clock_at_j2000_s(sat, t_j2000_s))
            .flatten()
    }
}

fn body(hex: &str) -> Vec<u8> {
    assert!(hex.len().is_multiple_of(2));
    (0..hex.len())
        .step_by(2)
        .map(|idx| u8::from_str_radix(&hex[idx..idx + 2], 16).expect("hex byte"))
        .collect()
}

fn epoch(tow_s: f64) -> sidereon_core::astro::time::model::GnssWeekTow {
    sidereon_core::astro::time::model::GnssWeekTow::new(
        sidereon_core::astro::time::model::TimeScale::Gpst,
        2400,
        tow_s,
    )
    .expect("valid epoch")
}

fn epoch_to_j2000_s(epoch: sidereon_core::astro::time::model::GnssWeekTow) -> f64 {
    f64::from(epoch.week) * sidereon_core::constants::SECONDS_PER_WEEK + epoch.tow_s
        - sidereon_core::constants::GPS_EPOCH_TO_J2000_S
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    scale(v, 1.0 / norm(v))
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn rotate_for_sagnac(pos: [f64; 3], tau_s: f64) -> [f64; 3] {
    let theta = OMEGA_E_DOT_RAD_S * tau_s;
    let (sin_theta, cos_theta) = libm::sincos(theta);
    [
        cos_theta * pos[0] - sin_theta * pos[1],
        sin_theta * pos[0] + cos_theta * pos[1],
        pos[2],
    ]
}

struct IonoContext<'a> {
    receiver_geo: Wgs84Geodetic,
    grid: &'a SbasIonoGrid,
    east: [f64; 3],
    north: [f64; 3],
    up: [f64; 3],
}

fn pseudorange_from_model(
    eph: &dyn EphemerisSource,
    sat: GnssSatelliteId,
    t_rx_j2000_s: f64,
    receiver_ecef_m: [f64; 3],
    receiver_clock_m: f64,
    iono: &IonoContext<'_>,
) -> f64 {
    let (pos0, _) = eph
        .position_clock_at_j2000_s(sat, t_rx_j2000_s)
        .expect("truth ephemeris at receive time");
    let mut tau = norm(sub(pos0, receiver_ecef_m)) / C_M_S;
    let mut sat_position = pos0;
    let mut sat_clock_s = 0.0;
    for _ in 0..2 {
        let t_tx = t_rx_j2000_s - tau;
        let (pos, clock) = eph
            .position_clock_at_j2000_s(sat, t_tx)
            .expect("truth ephemeris at transmit time");
        sat_position = pos;
        sat_clock_s = clock;
        tau = norm(sub(sat_position, receiver_ecef_m)) / C_M_S;
    }
    let sat_rot = rotate_for_sagnac(sat_position, tau);
    let los = unit(sub(sat_rot, receiver_ecef_m));
    let elevation = libm::asin(dot(los, iono.up));
    let azimuth = libm::atan2(dot(los, iono.east), dot(los, iono.north));
    let iono_m = iono
        .grid
        .slant_delay_m(iono.receiver_geo, elevation, azimuth, F_L1_HZ)
        .expect("SBAS ionosphere covers synthetic line of sight");
    norm(sub(sat_rot, receiver_ecef_m)) + receiver_clock_m - C_M_S * sat_clock_s + iono_m
}

fn gps(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS PRN")
}

fn long_record(monitored_index: u8, delta_raw: [i32; 3]) -> SbasLongTermRecord {
    SbasLongTermRecord {
        monitored_index,
        iode: 7,
        delta_x: delta_raw[0],
        delta_y: delta_raw[1],
        delta_z: delta_raw[2],
        delta_x_rate: 0,
        delta_y_rate: 0,
        delta_z_rate: 0,
        delta_a_f0: 0,
        delta_a_f1: 0,
        time_of_day_s: None,
    }
}

fn long_half(records: Vec<SbasLongTermRecord>) -> SbasLongTermHalf {
    SbasLongTermHalf {
        velocity_code: false,
        iodp: 1,
        records,
        reserved: SpareBits::new(),
    }
}

fn empty_long_half() -> SbasLongTermHalf {
    long_half(Vec::new())
}

fn iono_grid(lon_deg: f64) -> SbasIonoGrid {
    let mut points = Vec::new();
    for lat in [-20.0, 0.0, 20.0] {
        for lon in [lon_deg - 20.0, lon_deg, lon_deg + 20.0] {
            points.push(SbasIgp {
                lat_deg: lat,
                lon_deg: lon,
                vertical_delay_m: 5.0,
                give_variance_m2: None,
            });
        }
    }
    SbasIonoGrid::new(points, 0)
}

#[test]
fn sbas_corrected_spp_with_geo_ranging_beats_uncorrected() {
    let geo = sbas_prn_to_sat(129).expect("valid source GEO");
    let rx_epoch = epoch(433_754.4);
    let t_rx_j2000_s = epoch_to_j2000_s(rx_epoch);

    let geo_nav = SbasBlock::decode(
        &body("9A25C80C8D3F574632853C69A015EEBFF2D7DF580018FE3FCFF79C38C0"),
        SbasWireForm::Body226,
    )
    .expect("captured MT9 decodes")
    .message;

    let mut store = SbasCorrectionStore::new();
    store.ingest(&geo_nav, geo, rx_epoch).unwrap();
    let geo_state = store.geo_nav(geo).expect("GEO navigation state").clone();
    let (geo_position, _) = geo_state.state_at(t_rx_j2000_s);

    let lon_rad = libm::atan2(geo_position[1], geo_position[0]);
    let receiver_geo = Wgs84Geodetic::new(0.0, lon_rad, 0.0).expect("valid receiver geodetic");
    let receiver_ecef = geodetic_to_itrf(receiver_geo)
        .expect("valid receiver ECEF")
        .as_array();
    let up = unit(receiver_ecef);
    let east = unit([-up[1], up[0], 0.0]);
    let north = [0.0, 0.0, 1.0];
    let grid = iono_grid(lon_rad.to_degrees());
    let iono_context = IonoContext {
        receiver_geo,
        grid: &grid,
        east,
        north,
        up,
    };

    let los_vectors = [
        unit(add(up, add(scale(east, 0.30), scale(north, 0.20)))),
        unit(add(up, add(scale(east, -0.35), scale(north, 0.15)))),
        unit(add(up, add(scale(east, 0.10), scale(north, -0.35)))),
        unit(add(up, add(scale(east, -0.20), scale(north, -0.30)))),
        unit(add(up, add(scale(east, 0.40), scale(north, -0.10)))),
    ];
    let sats = [gps(1), gps(2), gps(3), gps(4), gps(5)];
    let delta_raw = [
        [80, -64, 40],
        [-72, 56, -32],
        [64, 48, -48],
        [-56, -40, 72],
        [48, -72, -56],
    ];
    let prc_raw = [80, -64, 48, -40, 32];

    let mut true_states = BTreeMap::new();
    let mut broadcast_states = BTreeMap::new();
    for ((&sat, los), (&delta, &prc)) in sats
        .iter()
        .zip(los_vectors.iter())
        .zip(delta_raw.iter().zip(prc_raw.iter()))
    {
        let true_position = add(receiver_ecef, scale(*los, 26_600_000.0));
        let delta_m = [
            f64::from(delta[0]) * 0.125,
            f64::from(delta[1]) * 0.125,
            f64::from(delta[2]) * 0.125,
        ];
        let prc_m = f64::from(prc) * 0.125;
        true_states.insert(
            sat,
            StaticState {
                position_ecef_m: true_position,
                clock_s: 0.0,
                iode: 7,
            },
        );
        broadcast_states.insert(
            sat,
            StaticState {
                position_ecef_m: sub(true_position, delta_m),
                clock_s: -prc_m / C_M_S,
                iode: 7,
            },
        );
    }
    let mut mask = [false; 210];
    for active in mask.iter_mut().take(sats.len()) {
        *active = true;
    }
    store
        .ingest(
            &SbasMessage::PrnMask(SbasPrnMask {
                preamble: 0x53,
                iodp: 1,
                mask,
                reserved: SpareBits::new(),
            }),
            geo,
            rx_epoch,
        )
        .unwrap();
    let mut prc = [0i16; 13];
    for (slot, raw) in prc_raw.iter().enumerate() {
        prc[slot] = *raw as i16;
    }
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
            rx_epoch,
        )
        .unwrap();
    store
        .ingest(
            &SbasMessage::LongTermCorrections(SbasLongTermCorrections {
                preamble: 0x53,
                halves: [
                    long_half(vec![
                        long_record(1, delta_raw[0]),
                        long_record(2, delta_raw[1]),
                    ]),
                    long_half(vec![
                        long_record(3, delta_raw[2]),
                        long_record(4, delta_raw[3]),
                    ]),
                ],
            }),
            geo,
            rx_epoch,
        )
        .unwrap();
    store
        .ingest(
            &SbasMessage::LongTermCorrections(SbasLongTermCorrections {
                preamble: 0x9A,
                halves: [
                    long_half(vec![long_record(5, delta_raw[4])]),
                    empty_long_half(),
                ],
            }),
            geo,
            rx_epoch,
        )
        .unwrap();

    let true_broadcast = StaticBroadcast {
        states: true_states.clone(),
        dynamic_geo: Some((geo, geo_state.clone())),
    };
    let receiver_clock_m = 35.0;
    let mut observations = Vec::new();
    for sat in sats.into_iter().chain([geo]) {
        observations.push(Observation {
            satellite_id: sat,
            pseudorange_m: pseudorange_from_model(
                &true_broadcast,
                sat,
                t_rx_j2000_s,
                receiver_ecef,
                receiver_clock_m,
                &iono_context,
            ),
        });
    }

    let base_inputs = SolveInputs {
        observations,
        t_rx_j2000_s,
        t_rx_second_of_day_s: rx_epoch.tow_s % sidereon_core::constants::SECONDS_PER_DAY,
        day_of_year: 1.0,
        initial_guess: [
            receiver_ecef[0] + 750.0,
            receiver_ecef[1] - 500.0,
            receiver_ecef[2] + 250.0,
            0.0,
        ],
        corrections: Corrections::IONO,
        klobuchar: KlobucharCoeffs {
            alpha: [0.0; 4],
            beta: [0.0; 4],
        },
        beidou_klobuchar: None,
        galileo_nequick: None,
        sbas_iono: Some(grid),
        glonass_channels: BTreeMap::new(),
        met: SurfaceMet::default(),
        robust: None,
    };

    let uncorrected = StaticBroadcast {
        states: broadcast_states,
        dynamic_geo: Some((geo, geo_state)),
    };
    let uncorrected_solution = solve(&uncorrected, &base_inputs, false).expect("uncorrected solve");
    let reference_solution = solve(&true_broadcast, &base_inputs, false).expect("reference solve");
    let corrected = SbasCorrectedEphemeris::new(&uncorrected, &store, geo);
    let corrected_solution = solve(&corrected, &base_inputs, false).expect("SBAS corrected solve");

    assert!(corrected_solution.used_sats.contains(&geo));
    let reference_position = reference_solution.position.as_array();
    let uncorrected_error_m = norm(sub(
        uncorrected_solution.position.as_array(),
        reference_position,
    ));
    let corrected_error_m = norm(sub(
        corrected_solution.position.as_array(),
        reference_position,
    ));
    assert!(
        corrected_error_m < uncorrected_error_m * 0.5,
        "corrected={corrected_error_m} uncorrected={uncorrected_error_m}"
    );
    assert!(
        corrected_error_m < 1.0e-3,
        "corrected position should match the reference solve, got {corrected_error_m}"
    );
}
