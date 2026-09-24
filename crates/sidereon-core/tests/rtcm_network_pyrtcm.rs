//! Network RTK and coordinate transformation messages against an independent
//! layout reference.
//!
//! RTKLIB does not decode 1014-1017, 1021-1027, 1030-1032, 1034, 1035 or
//! 1037-1039, so these messages are checked against pyrtcm 1.2.0's message
//! layouts: `fixtures-generators/pyrtcm_layouts/generate_network_frames.py`
//! writes four frames of each message in pyrtcm's field order and widths, with
//! every field drawn over its whole range (the extremes included), checks that
//! pyrtcm reads each frame back to the values written, and records those raw
//! values. Each frame here must decode strictly to exactly those values, in
//! wire order, and re-encode to its exact body.

use serde_json::Value;
use sidereon_core::rtcm::{FrameScanner, Message, ProjectionParameters};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/rtcm/network/");

/// The decoded raw field values of a message, in wire order, after the
/// message number: counts included, each name character as its code point.
fn wire_values(message: &Message) -> Vec<i64> {
    let mut v = Vec::new();
    match message {
        Message::NetworkAuxiliaryStation(m) => v.extend([
            i64::from(m.network_id),
            i64::from(m.subnetwork_id),
            i64::from(m.auxiliary_station_count),
            i64::from(m.master_station_id),
            i64::from(m.auxiliary_station_id),
            i64::from(m.delta_latitude),
            i64::from(m.delta_longitude),
            i64::from(m.delta_height),
        ]),
        Message::NetworkCorrectionDifferences(m) => {
            v.extend([
                i64::from(m.network_id),
                i64::from(m.subnetwork_id),
                i64::from(m.epoch_time),
                i64::from(m.multiple_message),
                i64::from(m.master_station_id),
                i64::from(m.auxiliary_station_id),
                i64::from(m.satellite_count),
            ]);
            for s in &m.satellites {
                v.extend([
                    i64::from(s.satellite_id),
                    i64::from(s.ambiguity_status),
                    i64::from(s.non_sync_count),
                ]);
                v.extend(s.geometric.map(i64::from));
                v.extend(s.iod.map(i64::from));
                v.extend(s.ionospheric.map(i64::from));
            }
        }
        Message::HelmertTransformation(m) => {
            for name in [&m.source_name, &m.target_name] {
                v.push(name.chars().count() as i64);
                v.extend(name.chars().map(|c| i64::from(u32::from(c))));
            }
            v.extend([
                i64::from(m.system_id),
                i64::from(m.utilized_messages),
                i64::from(m.plate_number),
                i64::from(m.computation_indicator),
                i64::from(m.height_indicator),
                i64::from(m.validity_latitude),
                i64::from(m.validity_longitude),
                i64::from(m.validity_extension_latitude),
                i64::from(m.validity_extension_longitude),
                i64::from(m.dx),
                i64::from(m.dy),
                i64::from(m.dz),
                i64::from(m.r1),
                i64::from(m.r2),
                i64::from(m.r3),
                i64::from(m.ds),
            ]);
            if let Some(p) = m.rotation_point {
                v.extend([p.x, p.y, p.z]);
            }
            v.extend([
                i64::from(m.add_as),
                i64::from(m.add_bs),
                i64::from(m.add_at),
                i64::from(m.add_bt),
                i64::from(m.horizontal_quality),
                i64::from(m.vertical_quality),
            ]);
        }
        Message::ResidualGrid(m) => {
            v.extend([
                i64::from(m.system_id),
                i64::from(m.horizontal_shift),
                i64::from(m.vertical_shift),
                i64::from(m.origin_1),
                i64::from(m.origin_2),
                i64::from(m.extension_1),
                i64::from(m.extension_2),
                i64::from(m.mean_offset_1),
                i64::from(m.mean_offset_2),
                i64::from(m.mean_height_offset),
            ]);
            for r in &m.residuals {
                v.extend([
                    i64::from(r.horizontal_1),
                    i64::from(r.horizontal_2),
                    i64::from(r.height),
                ]);
            }
            v.extend([
                i64::from(m.horizontal_interpolation),
                i64::from(m.vertical_interpolation),
                i64::from(m.horizontal_quality),
                i64::from(m.vertical_quality),
                i64::from(m.mjd),
            ]);
        }
        Message::Projection(m) => {
            v.extend([i64::from(m.system_id), i64::from(m.projection_type)]);
            match m.parameters {
                ProjectionParameters::NaturalOrigin {
                    latitude,
                    longitude,
                    add_scale,
                    false_easting,
                    false_northing,
                } => v.extend([
                    latitude,
                    longitude,
                    i64::from(add_scale),
                    false_easting as i64,
                    false_northing,
                ]),
                ProjectionParameters::LambertConicConformal {
                    latitude,
                    longitude,
                    standard_parallel_1,
                    standard_parallel_2,
                    false_easting,
                    false_northing,
                } => v.extend([
                    latitude,
                    longitude,
                    standard_parallel_1,
                    standard_parallel_2,
                    false_easting as i64,
                    false_northing,
                ]),
                ProjectionParameters::ObliqueMercator {
                    rectification,
                    latitude,
                    longitude,
                    azimuth,
                    rectified_to_skew,
                    add_scale,
                    easting,
                    northing,
                } => v.extend([
                    i64::from(rectification),
                    latitude,
                    longitude,
                    azimuth as i64,
                    i64::from(rectified_to_skew),
                    i64::from(add_scale),
                    easting as i64,
                    northing,
                ]),
            }
        }
        Message::NetworkResiduals(m) => {
            v.extend([
                i64::from(m.epoch_time),
                i64::from(m.reference_station_id),
                i64::from(m.reference_station_count),
                i64::from(m.satellite_count),
            ]);
            for s in &m.satellites {
                v.extend([
                    i64::from(s.satellite_id),
                    i64::from(s.s_oc),
                    i64::from(s.s_od),
                    i64::from(s.s_oh),
                    i64::from(s.s_lc),
                    i64::from(s.s_ld),
                ]);
            }
        }
        Message::PhysicalReferenceStation(m) => v.extend([
            i64::from(m.non_physical_station_id),
            i64::from(m.physical_station_id),
            i64::from(m.itrf_realization_year),
            m.ecef_x,
            m.ecef_y,
            m.ecef_z,
        ]),
        Message::FkpGradients(m) => {
            v.extend([
                i64::from(m.reference_station_id),
                i64::from(m.epoch_time),
                i64::from(m.satellite_count),
            ]);
            for s in &m.satellites {
                v.extend([
                    i64::from(s.satellite_id),
                    i64::from(s.iod),
                    i64::from(s.geometric_north),
                    i64::from(s.geometric_east),
                    i64::from(s.ionospheric_north),
                    i64::from(s.ionospheric_east),
                ]);
            }
        }
        other => panic!(
            "message {} is not a network RTK message",
            other.message_number()
        ),
    }
    v
}

/// Every pyrtcm-laid-out frame of the network RTK and transformation messages
/// decodes strictly to the values written, field for field in wire order, and
/// re-encodes to its exact body.
#[test]
fn network_rtk_and_transformation_messages_match_pyrtcm_layouts() {
    let bytes = std::fs::read(format!("{DIR}pyrtcm_network_rtk.rtcm3")).expect("frames");
    let reference: Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{DIR}pyrtcm_network_rtk.json")).expect("values"),
    )
    .expect("json");
    let records = reference["frames"].as_array().expect("frames");
    let mut scanner = FrameScanner::new(&bytes);
    let frames: Vec<_> = scanner.by_ref().collect();
    assert_eq!(scanner.resync_bytes(), 0);
    assert_eq!(frames.len(), records.len());
    let mut types = std::collections::BTreeSet::new();
    let mut values = 0usize;
    for (index, (frame, record)) in frames.iter().zip(records).enumerate() {
        let number = record["type"].as_u64().expect("type");
        let message = Message::decode(frame.body)
            .unwrap_or_else(|err| panic!("frame {index} ({number}): {err}"));
        assert_eq!(u64::from(message.message_number()), number, "frame {index}");
        assert!(!matches!(message, Message::Unsupported(_)), "frame {index}");
        let expected: Vec<i64> = record["fields"]
            .as_array()
            .expect("fields")
            .iter()
            .map(|field| field[1].as_i64().expect("raw value"))
            .collect();
        assert_eq!(wire_values(&message), expected, "frame {index} ({number})");
        assert_eq!(
            message.encode().expect("re-encode"),
            frame.body,
            "frame {index} ({number})"
        );
        types.insert(number);
        values += expected.len();
    }
    eprintln!(
        "{} frames, {} message types, {values} values",
        frames.len(),
        types.len()
    );
    assert_eq!(types.len(), 19);
}
