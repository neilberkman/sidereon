//! BNC decoding oracle for the RTCM SSR and IGS SSR messages.
//!
//! BNC (the BKG NTRIP Client, 2.13.7) is the reference SSR codec of the IGS
//! real-time service. `fixtures-generators/bnc_ssr_oracle/generate.sh` builds
//! its SSR codec, writes `tests/fixtures/rtcm/families/bnc_encoded_ssr.rtcm3`
//! with BNC's encoder (every RTCM SSR message 1057..1270 including 1264, and
//! every IGS SSR subtype including the VTEC message 201, over every field's
//! range), and records what BNC's decoder reads from each frame of that stream,
//! of the RTKLIB-encoded 4076 stream and of the real IGS RTCM SSR stream
//! `tests/fixtures/ssr/SSRA03IGS0_2026188140760_3epoch.rtcm3`
//! (`<stream>.bnc.jsonl`, floating values as the hex of their IEEE 754 bits).
//!
//! Every frame here decodes strictly, re-encodes to its body, and gives, with
//! BNC's scale constants and order of operations, the bits BNC stores.

use serde_json::Value;
use sidereon_core::rtcm::{FrameScanner, Message, SsrKind, SsrMessage, SsrVtecMessage};
use sidereon_core::GnssSystem;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");
/// BNC's `MPI`, spelled as BNC spells it.
#[allow(clippy::approx_constant)]
const MPI: f64 = 3.141592653589793;

fn bits64(value: &Value) -> u64 {
    u64::from_str_radix(value.as_str().expect("hex bits"), 16).expect("hex bits")
}

fn bits32(value: &Value) -> u32 {
    u32::from_str_radix(value.as_str().expect("hex bits"), 16).expect("hex bits")
}

/// BNC's system index (`CLOCKORBIT_SATGPS` ...).
fn system_index(system: GnssSystem) -> usize {
    match system {
        GnssSystem::Gps => 0,
        GnssSystem::Glonass => 1,
        GnssSystem::Galileo => 2,
        GnssSystem::Qzss => 3,
        GnssSystem::Sbas => 4,
        GnssSystem::BeiDou => 5,
        GnssSystem::Navic => panic!("no NavIC SSR"),
    }
}

/// BNC `URAToValue`.
fn ura_to_value(ura: u8) -> f64 {
    let (urac, urav) = (i32::from(ura >> 3), f64::from(ura & 7));
    match ura {
        0 => 0.0,
        63 => 5.5,
        _ => (3f64.powi(urac) * (1.0 + urav / 4.0) - 1.0) / 1000.0,
    }
}

#[derive(Default, Debug)]
struct Coverage {
    frames: usize,
    satellites: usize,
    values: usize,
}

fn check_ssr(at: &str, ssr: &SsrMessage, bnc: &Value, coverage: &mut Coverage) {
    let s = system_index(ssr.system);
    let h = &ssr.header;
    let group = match ssr.kind {
        SsrKind::CodeBias => &bnc["cb"],
        SsrKind::PhaseBias => &bnc["pb"],
        _ => &bnc["co"],
    };
    assert!(group.is_object(), "{at}: BNC stored no {:?}", ssr.kind);
    assert_eq!(group["iod"].as_u64(), Some(u64::from(h.iod_ssr)), "{at}");
    assert_eq!(
        group["provider"].as_u64(),
        Some(u64::from(h.provider_id)),
        "{at}"
    );
    assert_eq!(
        group["solution"].as_u64(),
        Some(u64::from(h.solution_id)),
        "{at}"
    );
    assert_eq!(
        group["udi"].as_u64(),
        Some(u64::from(h.update_interval)),
        "{at}"
    );
    let system = &group["systems"][s];
    assert_eq!(
        system["epoch"].as_u64(),
        Some(u64::from(h.epoch_time_s)),
        "{at}"
    );
    let sats = system["sats"].as_array().expect("sats");
    let igs = ssr.igs_ssr_version.is_some();
    match ssr.kind {
        SsrKind::Orbit
        | SsrKind::Clock
        | SsrKind::CombinedOrbitClock
        | SsrKind::HighRateClock
        | SsrKind::Ura => {
            if let Some(datum) = h.satellite_reference_datum {
                assert_eq!(group["datum"].as_u64(), Some(u64::from(datum)), "{at}");
            }
            let ids: Vec<u8> = match ssr.kind {
                SsrKind::Orbit | SsrKind::CombinedOrbitClock => {
                    ssr.orbit.iter().map(|o| o.satellite_id).collect()
                }
                SsrKind::Ura => ssr.ura.iter().map(|&(id, _)| id).collect(),
                _ => ssr.clock.iter().map(|c| c.satellite_id).collect(),
            };
            assert_eq!(sats.len(), ids.len(), "{at}");
            for (index, (sat, &id)) in sats.iter().zip(&ids).enumerate() {
                let at = format!("{at} satellite {id}");
                assert_eq!(sat["id"].as_u64(), Some(u64::from(id)), "{at}");
                if let Some(o) = ssr.orbit.get(index) {
                    let (iod, toe) = match (igs, ssr.system) {
                        (false, GnssSystem::Sbas) => (o.iod_crc.expect("IOD CRC"), o.iode * 16),
                        (false, GnssSystem::BeiDou) => (o.iode & 0xFF, (o.iode >> 8) * 8),
                        _ => (o.iode, 0),
                    };
                    assert_eq!(sat["iod"].as_u64(), Some(u64::from(iod)), "{at} IOD");
                    assert_eq!(sat["toe"].as_u64(), Some(u64::from(toe)), "{at} toe");
                    let orbit = [
                        f64::from(o.delta_radial) * (1.0 / 10000.0),
                        f64::from(o.delta_along) * (1.0 / 2500.0),
                        f64::from(o.delta_cross) * (1.0 / 2500.0),
                        f64::from(o.dot_delta_radial) * (1.0 / 1000000.0),
                        f64::from(o.dot_delta_along) * (1.0 / 250000.0),
                        f64::from(o.dot_delta_cross) * (1.0 / 250000.0),
                    ];
                    for (k, value) in orbit.iter().enumerate() {
                        assert_eq!(bits64(&sat["orbit"][k]), value.to_bits(), "{at} orbit {k}");
                    }
                    coverage.values += 8;
                }
                match ssr.kind {
                    SsrKind::Clock | SsrKind::CombinedOrbitClock => {
                        let c = &ssr.clock[index];
                        let clock = [
                            f64::from(c.c0) * (1.0 / 10000.0),
                            f64::from(c.c1) * (1.0 / 1000000.0),
                            f64::from(c.c2) * (1.0 / 50000000.0),
                        ];
                        for (k, value) in clock.iter().enumerate() {
                            assert_eq!(bits64(&sat["clock"][k]), value.to_bits(), "{at} clock {k}");
                        }
                        coverage.values += 3;
                    }
                    SsrKind::HighRateClock => {
                        let hr = f64::from(ssr.clock[index].c0) * (1.0 / 10000.0);
                        assert_eq!(bits64(&sat["hr"]), hr.to_bits(), "{at} high-rate clock");
                        coverage.values += 1;
                    }
                    SsrKind::Ura => {
                        let ura = ura_to_value(ssr.ura[index].1);
                        assert_eq!(bits64(&sat["ura"]), ura.to_bits(), "{at} URA");
                        coverage.values += 1;
                    }
                    _ => {}
                }
            }
            coverage.satellites += ids.len();
        }
        SsrKind::CodeBias => {
            assert_eq!(sats.len(), ssr.code_bias.len(), "{at}");
            for (sat, record) in sats.iter().zip(&ssr.code_bias) {
                let at = format!("{at} satellite {}", record.satellite_id);
                assert_eq!(
                    sat["id"].as_u64(),
                    Some(u64::from(record.satellite_id)),
                    "{at}"
                );
                let biases = sat["biases"].as_array().expect("biases");
                assert_eq!(biases.len(), record.biases.len(), "{at}");
                for (bias, &(signal, raw)) in biases.iter().zip(&record.biases) {
                    assert_eq!(bias[0].as_u64(), Some(u64::from(signal)), "{at}");
                    let value = (f64::from(raw) * (1.0 / 100.0)) as f32;
                    assert_eq!(bits32(&bias[1]), value.to_bits(), "{at} signal {signal}");
                    coverage.values += 2;
                }
            }
            coverage.satellites += ssr.code_bias.len();
        }
        SsrKind::PhaseBias => {
            assert_eq!(
                group["dispersive"].as_u64(),
                h.dispersive_bias_consistency.map(u64::from),
                "{at}"
            );
            assert_eq!(
                group["mw"].as_u64(),
                h.mw_consistency.map(u64::from),
                "{at}"
            );
            assert_eq!(sats.len(), ssr.phase_bias.len(), "{at}");
            for (sat, record) in sats.iter().zip(&ssr.phase_bias) {
                let at = format!("{at} satellite {}", record.satellite_id);
                assert_eq!(
                    sat["id"].as_u64(),
                    Some(u64::from(record.satellite_id)),
                    "{at}"
                );
                let yaw = f64::from(record.yaw_angle) * (MPI / 256.0);
                let yaw_rate = f64::from(record.yaw_rate) * (MPI / 8192.0);
                assert_eq!(bits64(&sat["yaw"]), yaw.to_bits(), "{at} yaw");
                assert_eq!(
                    bits64(&sat["yaw_rate"]),
                    yaw_rate.to_bits(),
                    "{at} yaw rate"
                );
                let biases = sat["biases"].as_array().expect("biases");
                assert_eq!(biases.len(), record.biases.len(), "{at}");
                for (bias, signal) in biases.iter().zip(&record.biases) {
                    assert_eq!(bias[0].as_u64(), Some(u64::from(signal.signal_id)), "{at}");
                    assert_eq!(
                        bias[1].as_u64(),
                        Some(u64::from(signal.integer_indicator)),
                        "{at}"
                    );
                    assert_eq!(
                        bias[2].as_u64(),
                        Some(u64::from(signal.wide_lane_integer_indicator)),
                        "{at}"
                    );
                    assert_eq!(
                        bias[3].as_u64(),
                        Some(u64::from(signal.discontinuity_counter)),
                        "{at}"
                    );
                    let value = (f64::from(signal.bias) * (1.0 / 10000.)) as f32;
                    assert_eq!(bits32(&bias[4]), value.to_bits(), "{at} phase bias");
                    coverage.values += 5;
                }
                coverage.values += 2;
            }
            coverage.satellites += ssr.phase_bias.len();
        }
    }
}

fn check_vtec(at: &str, vtec: &SsrVtecMessage, bnc: &Value, coverage: &mut Coverage) {
    let v = &bnc["vtec"];
    assert_eq!(
        v["epoch"].as_u64(),
        Some(u64::from(vtec.epoch_time_s)),
        "{at}"
    );
    assert_eq!(
        v["udi"].as_u64(),
        Some(u64::from(vtec.update_interval)),
        "{at}"
    );
    assert_eq!(v["iod"].as_u64(), Some(u64::from(vtec.iod_ssr)), "{at}");
    assert_eq!(
        v["provider"].as_u64(),
        Some(u64::from(vtec.provider_id)),
        "{at}"
    );
    assert_eq!(
        v["solution"].as_u64(),
        Some(u64::from(vtec.solution_id)),
        "{at}"
    );
    let quality = f64::from(vtec.quality_indicator) * (1.0 / 20.0);
    assert_eq!(bits64(&v["quality"]), quality.to_bits(), "{at} quality");
    let layers = v["layers"].as_array().expect("layers");
    assert_eq!(layers.len(), vtec.layers.len(), "{at}");
    for (layer, ours) in layers.iter().zip(&vtec.layers) {
        let height = f64::from(ours.height) * 10000.0;
        assert_eq!(bits64(&layer["height"]), height.to_bits(), "{at} height");
        assert_eq!(
            layer["degree"].as_u64(),
            Some(u64::from(ours.degree)),
            "{at}"
        );
        assert_eq!(layer["order"].as_u64(), Some(u64::from(ours.order)), "{at}");
        for (key, list) in [("c", &ours.cosine), ("s", &ours.sine)] {
            let stored = layer[key].as_array().expect("coefficients");
            assert_eq!(stored.len(), list.len(), "{at} {key}");
            for (value, &raw) in stored.iter().zip(list) {
                let expected = f64::from(raw) * (1.0 / 200.0);
                assert_eq!(bits64(value), expected.to_bits(), "{at} {key}");
            }
            coverage.values += list.len();
        }
    }
}

fn check_stream(name: &str) -> Coverage {
    let bytes = std::fs::read(format!("{FIXTURES}{name}")).expect("stream");
    let records: Vec<Value> = std::fs::read_to_string(format!("{FIXTURES}{name}.bnc.jsonl"))
        .expect("BNC records")
        .lines()
        .map(|line| serde_json::from_str(line).expect("record"))
        .collect();
    let frames: Vec<_> = FrameScanner::new(&bytes).collect();
    assert_eq!(frames.len(), records.len(), "{name}");
    let mut coverage = Coverage::default();
    for (frame, bnc) in frames.iter().zip(&records) {
        let offset = frame.body.as_ptr() as usize - bytes.as_ptr() as usize - 3;
        let at = format!("{name} frame at {offset}");
        assert_eq!(bnc["offset"].as_u64(), Some(offset as u64), "{at}");
        assert!(
            matches!(bnc["ret"].as_i64(), Some(0 | 1)),
            "{at}: BNC refused it"
        );
        let message = Message::decode(frame.body).unwrap_or_else(|err| panic!("{at}: {err}"));
        assert_eq!(message.encode().expect("encode"), frame.body, "{at}");
        match &message {
            Message::Ssr(ssr) => check_ssr(&at, ssr, bnc, &mut coverage),
            Message::SsrVtec(vtec) => check_vtec(&at, vtec, bnc, &mut coverage),
            other => panic!("{at}: message {} is not SSR", other.message_number()),
        }
        coverage.frames += 1;
    }
    coverage
}

/// Every RTCM SSR message 1057..1270 (1264 included) and every IGS SSR
/// subtype (201 included) BNC's encoder writes, the RTKLIB-encoded 4076
/// frames, and a real IGS RTCM SSR stream: every value BNC stores, bit for
/// bit.
#[test]
fn ssr_messages_match_bnc() {
    for name in [
        "rtcm/families/bnc_encoded_ssr.rtcm3",
        "rtcm/families/rtklib_encoded_4076.rtcm3",
        "ssr/SSRA03IGS0_2026188140760_3epoch.rtcm3",
    ] {
        let coverage = check_stream(name);
        eprintln!("{name}: {coverage:?}");
        assert!(coverage.values > 0, "{name}");
    }
}
