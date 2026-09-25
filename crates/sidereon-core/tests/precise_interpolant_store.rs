use std::fs;
use std::mem;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use sidereon_core::ephemeris::{
    precise_interpolant_store_checksum64, MmapPreciseEphemerisInterpolant,
    PreciseEphemerisInterpolant, PreciseInterpolantStoreError, Sp3, Sp3InterpolationOptions,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

const COD_5M_FIXTURE: &str = "tests/fixtures/sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3";
/// 15-minute product with the G01 records from 07:30 through 10:00 GPST
/// removed, so its 07:15 and 10:15 nodes bracket a three-hour hole.
const GAP_15M_FIXTURE: &str = "tests/fixtures/sp3/GAP_G01_20201760000_15M.sp3";

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name)
}

fn temp_path(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{name}-{}-{nonce}", std::process::id()))
}

fn fixture_sp3() -> Sp3 {
    let bytes = fs::read(fixture_path(COD_5M_FIXTURE)).expect("read SP3 fixture");
    Sp3::parse(&bytes).expect("parse SP3 fixture")
}

fn gps(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS satellite")
}

fn assert_state_bits_eq(
    sat: GnssSatelliteId,
    epoch_j2000_s: f64,
    mapped: sidereon_core::ephemeris::Sp3State,
    memory: sidereon_core::ephemeris::Sp3State,
) {
    assert_eq!(
        mapped.position.as_array().map(f64::to_bits),
        memory.position.as_array().map(f64::to_bits),
        "{sat} position bits differ at {epoch_j2000_s}"
    );
    assert_eq!(
        mapped.clock_s.map(f64::to_bits),
        memory.clock_s.map(f64::to_bits),
        "{sat} clock bits differ at {epoch_j2000_s}"
    );
}

#[test]
fn precise_interpolant_store_build_is_deterministic_for_same_product() {
    let sp3 = fixture_sp3();
    let first = PreciseEphemerisInterpolant::from_sp3(&sp3)
        .to_mmap_store_bytes()
        .expect("first artifact build");
    let second = PreciseEphemerisInterpolant::from_sp3(&sp3)
        .to_mmap_store_bytes()
        .expect("second artifact build");

    assert_eq!(first, second);
    let mapped =
        MmapPreciseEphemerisInterpolant::from_bytes(&first).expect("artifact opens from bytes");
    assert_eq!(mapped.as_bytes().as_ptr(), first.as_ptr());
    assert_eq!(
        mapped.checksum64(),
        precise_interpolant_store_checksum64(&first)
    );
    assert_eq!(mapped.as_bytes(), first.as_slice());
}

#[test]
fn mapped_precise_interpolant_matches_in_memory_bits_at_records_and_midpoints() {
    let sp3 = fixture_sp3();
    let memory = PreciseEphemerisInterpolant::from_sp3(&sp3);
    let bytes = memory
        .to_mmap_store_bytes()
        .expect("build precise interpolant artifact");
    let store_path = temp_path("precise-interpolant-store").with_extension("bin");
    fs::write(&store_path, &bytes).expect("write artifact");

    {
        let mapped =
            MmapPreciseEphemerisInterpolant::from_path(&store_path).expect("read artifact");
        assert_eq!(mapped.time_scale(), memory.time_scale());
        assert_eq!(&mapped.satellites()[..3], &[gps(1), gps(2), gps(3)]);

        let epochs = sp3.epochs_j2000_seconds();
        let mut queries = Vec::new();
        queries.extend(epochs.iter().take(24).copied());
        queries.extend(
            epochs
                .windows(2)
                .take(24)
                .map(|window| 0.5 * (window[0] + window[1])),
        );

        for sat in [gps(1), gps(2), gps(3)] {
            for &epoch_j2000_s in &queries {
                let got = mapped
                    .position_at_j2000_seconds(sat, epoch_j2000_s)
                    .expect("mapped evaluation");
                let want = memory
                    .position_at_j2000_seconds(sat, epoch_j2000_s)
                    .expect("in-memory evaluation");
                assert_state_bits_eq(sat, epoch_j2000_s, got, want);
            }
        }
    }

    fs::remove_file(store_path).expect("remove temp artifact");
}

/// The mapped reader carries its own copy of the coverage-gap policy (query
/// gate plus contiguous-run bracket) over the byte-indexed node axis. Pin it to
/// the in-memory interpolant on the gapped fixture: bit-identical states where
/// both evaluate, and the same rejection at the edges and inside the hole.
#[test]
fn mapped_precise_interpolant_matches_in_memory_across_a_coverage_gap() {
    let bytes = fs::read(fixture_path(GAP_15M_FIXTURE)).expect("read gapped SP3 fixture");
    let sp3 = Sp3::parse(&bytes).expect("parse gapped SP3 fixture");
    let memory = PreciseEphemerisInterpolant::from_sp3(&sp3);
    let mapped = MmapPreciseEphemerisInterpolant::from_vec(
        memory
            .to_mmap_store_bytes()
            .expect("build precise interpolant artifact"),
    )
    .expect("read artifact");

    let epochs = sp3.epochs_j2000_seconds();
    let nominal_s = 900.0;
    let first = epochs[0];
    let last = epochs[epochs.len() - 1];
    // The G01 hole is bracketed by its 07:15 and 10:15 nodes; every other
    // satellite is contiguous across the whole day.
    let hole_lo = 646_254_900.0;
    let hole_hi = 646_265_700.0;

    let mut queries: Vec<f64> = epochs.clone();
    queries.extend(epochs.windows(2).map(|w| 0.5 * (w[0] + w[1])));
    // One nominal spacing past either edge is admitted; beyond it is refused.
    let edge_overreach = [first - nominal_s - 50.0, last + nominal_s + 50.0];
    queries.extend([first - nominal_s, last + nominal_s]);
    queries.extend(edge_overreach);
    // The admitted margins on either side of the hole, and in-gap points that
    // must be refused for G01 alone.
    let hole_margins = [hole_lo + nominal_s, hole_hi - nominal_s];
    let in_hole = [
        hole_lo + nominal_s + 1.0,
        646_260_300.0,
        hole_hi - nominal_s - 1.0,
    ];
    queries.extend(hole_margins);
    queries.extend(in_hole);

    for sat in [gps(1), gps(2)] {
        for &epoch_j2000_s in &queries {
            let got = mapped.position_at_j2000_seconds(sat, epoch_j2000_s);
            let want = memory.position_at_j2000_seconds(sat, epoch_j2000_s);
            match (got, want) {
                (Ok(got), Ok(want)) => assert_state_bits_eq(sat, epoch_j2000_s, got, want),
                (Err(got), Err(want)) => {
                    assert_eq!(got, want, "{sat} at {epoch_j2000_s}: rejection differs");
                }
                (got, want) => {
                    panic!("{sat} at {epoch_j2000_s}: mapped {got:?} vs in-memory {want:?}")
                }
            }
        }

        // The rejection path was exercised, not vacuously matched.
        for epoch_j2000_s in edge_overreach {
            assert!(
                mapped
                    .position_at_j2000_seconds(sat, epoch_j2000_s)
                    .is_err(),
                "{sat} at {epoch_j2000_s}: edge over-reach must be refused"
            );
        }
    }
    for epoch_j2000_s in hole_margins {
        mapped
            .position_at_j2000_seconds(gps(1), epoch_j2000_s)
            .unwrap_or_else(|e| panic!("G01 hole margin {epoch_j2000_s} must evaluate: {e:?}"));
    }
    for epoch_j2000_s in in_hole {
        assert!(
            mapped
                .position_at_j2000_seconds(gps(1), epoch_j2000_s)
                .is_err(),
            "G01 in-hole query {epoch_j2000_s} must be refused"
        );
        mapped
            .position_at_j2000_seconds(gps(2), epoch_j2000_s)
            .unwrap_or_else(|e| panic!("G02 has no hole at {epoch_j2000_s}: {e:?}"));
    }
}

#[test]
fn borrowed_precise_interpolant_store_rejects_unaligned_zero_copy_slice() {
    let sp3 = fixture_sp3();
    let bytes = sp3
        .precise_interpolant_store_bytes()
        .expect("build precise interpolant artifact");
    let mut padded = vec![0u8; bytes.len() + mem::align_of::<f64>()];

    for offset in 0..mem::align_of::<f64>() {
        padded[offset..offset + bytes.len()].copy_from_slice(&bytes);
        let candidate = &padded[offset..offset + bytes.len()];
        if (candidate.as_ptr() as usize).is_multiple_of(mem::align_of::<f64>()) {
            continue;
        }
        let err = MmapPreciseEphemerisInterpolant::from_bytes(candidate)
            .expect_err("unaligned borrowed artifact must fail");
        assert!(matches!(err, PreciseInterpolantStoreError::Parse { .. }));
        return;
    }

    panic!("test allocator did not provide any unaligned candidate slice");
}

#[test]
fn precise_interpolant_store_rejects_corrupt_and_truncated_artifacts() {
    let sp3 = fixture_sp3();
    let bytes = sp3
        .precise_interpolant_store_bytes()
        .expect("build precise interpolant artifact");

    let mut corrupt = bytes.clone();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 0x80;
    let err = MmapPreciseEphemerisInterpolant::from_bytes(&corrupt)
        .expect_err("corrupt artifact must fail");
    assert!(matches!(err, PreciseInterpolantStoreError::Checksum { .. }));

    // A short store is truncated, with both lengths, not a checksum mismatch.
    let truncated = &bytes[..bytes.len() - 1];
    let err = MmapPreciseEphemerisInterpolant::from_bytes(truncated)
        .expect_err("truncated artifact must fail");
    assert_eq!(
        err,
        PreciseInterpolantStoreError::Truncated {
            declared: bytes.len() as u64,
            available: bytes.len() as u64 - 1,
        }
    );
}

fn header_checksum(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(
        bytes[HEADER_CHECKSUM_OFFSET..HEADER_CHECKSUM_OFFSET + 8]
            .try_into()
            .unwrap(),
    )
}

const HEADER_VERSION_OFFSET: usize = 8;
const HEADER_SAT_COUNT_OFFSET: usize = 12;
const HEADER_INDEX_OFFSET_OFFSET: usize = 16;
const HEADER_DATA_OFFSET_OFFSET: usize = 24;
const HEADER_TOTAL_LEN_OFFSET: usize = 32;
const V1_STORE_VERSION: u16 = 1;
const V2_STORE_VERSION: u16 = 2;
const SAT_INDEX_RECORD_LEN: usize = 96;
const SAT_POSITION_COUNT_OFFSET: usize = 4;
const SAT_CLOCK_NODE_COUNT_OFFSET: usize = 8;
const SAT_CLOCK_ARC_COUNT_OFFSET: usize = 12;
const SAT_POS_X_OFFSET: usize = 16;
const SAT_POS_KX_OFFSET: usize = 24;
const SAT_POS_KY_OFFSET: usize = 32;
const SAT_POS_KZ_OFFSET: usize = 40;
const SAT_CLOCK_NODE_OFFSET: usize = 48;
const SAT_CLOCK_ARC_OFFSET: usize = 56;
const SAT_CHECKSUM_OFFSET: usize = 80;
const CLOCK_ARC_RECORD_LEN: usize = 64;
const CLOCK_ARC_ARRAY_OFFSETS: [usize; 5] = [8, 16, 24, 32, 40];
const ACCURACY_VALUE_RECORD_LEN: usize = 16;
const STORE_ALIGNMENT: usize = 4096;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn read_u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn read_u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn write_u16_at(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u64_at(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

fn legacy_v1_checksum64(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .enumerate()
        .fold(FNV_OFFSET_BASIS, |hash, (index, byte)| {
            let value = if (HEADER_CHECKSUM_OFFSET..HEADER_CHECKSUM_OFFSET + 8).contains(&index) {
                0
            } else {
                *byte
            };
            (hash ^ u64::from(value)).wrapping_mul(FNV_PRIME)
        })
}

fn align_store_offset(offset: usize) -> usize {
    (offset + STORE_ALIGNMENT - 1) & !(STORE_ALIGNMENT - 1)
}

fn project_v2_store_to_v1(bytes: &[u8]) -> Vec<u8> {
    assert_eq!(read_u16_at(bytes, HEADER_VERSION_OFFSET), V2_STORE_VERSION);

    let satellite_count = read_u32_at(bytes, HEADER_SAT_COUNT_OFFSET) as usize;
    let index_offset = read_u64_at(bytes, HEADER_INDEX_OFFSET_OFFSET) as usize;
    let data_offset = read_u64_at(bytes, HEADER_DATA_OFFSET_OFFSET) as usize;
    let mut legacy = bytes[..data_offset].to_vec();

    for satellite_index in 0..satellite_count {
        let record_offset = index_offset + satellite_index * SAT_INDEX_RECORD_LEN;
        let original_record = &bytes[record_offset..record_offset + SAT_INDEX_RECORD_LEN];
        let position_count = read_u32_at(original_record, SAT_POSITION_COUNT_OFFSET) as usize;
        let clock_node_count = read_u32_at(original_record, SAT_CLOCK_NODE_COUNT_OFFSET) as usize;
        let clock_arc_count = read_u32_at(original_record, SAT_CLOCK_ARC_COUNT_OFFSET) as usize;
        let original_data_offset = read_u64_at(original_record, SAT_DATA_OFFSET_OFFSET) as usize;
        let original_data_len = read_u64_at(original_record, SAT_DATA_LEN_OFFSET) as usize;
        let original_data_end = original_data_offset + original_data_len;
        let original_accuracy_start =
            read_u64_at(original_record, SAT_POS_KZ_OFFSET) as usize + position_count * 8;
        let accuracy_record_count = position_count * 3 + clock_node_count;
        let removed_accuracy_len = accuracy_record_count * ACCURACY_VALUE_RECORD_LEN;
        let original_accuracy_end = original_accuracy_start + removed_accuracy_len;
        assert!(original_accuracy_start >= original_data_offset);
        assert!(original_accuracy_end <= original_data_end);

        let new_data_offset = align_store_offset(legacy.len());
        legacy.resize(new_data_offset, 0);
        legacy.extend_from_slice(&bytes[original_data_offset..original_accuracy_start]);
        legacy.extend_from_slice(&bytes[original_accuracy_end..original_data_end]);
        let new_data_len = original_data_len - removed_accuracy_len;

        let relocate_offset = |original_offset: usize| {
            assert!(
                original_offset < original_accuracy_start
                    || original_offset >= original_accuracy_end
            );
            let original_relative = original_offset - original_data_offset;
            let new_relative = if original_offset < original_accuracy_start {
                original_relative
            } else {
                original_relative - removed_accuracy_len
            };
            new_data_offset + new_relative
        };

        let mut relocated_offsets = Vec::with_capacity(6);
        for field_offset in [
            SAT_POS_X_OFFSET,
            SAT_POS_KX_OFFSET,
            SAT_POS_KY_OFFSET,
            SAT_POS_KZ_OFFSET,
            SAT_CLOCK_NODE_OFFSET,
            SAT_CLOCK_ARC_OFFSET,
        ] {
            relocated_offsets.push((
                field_offset,
                relocate_offset(read_u64_at(original_record, field_offset) as usize),
            ));
        }
        for (field_offset, relocated_offset) in relocated_offsets {
            write_u64_at(
                &mut legacy,
                record_offset + field_offset,
                relocated_offset as u64,
            );
        }
        write_u64_at(
            &mut legacy,
            record_offset + SAT_DATA_OFFSET_OFFSET,
            new_data_offset as u64,
        );
        write_u64_at(
            &mut legacy,
            record_offset + SAT_DATA_LEN_OFFSET,
            new_data_len as u64,
        );

        let original_arc_offset = read_u64_at(original_record, SAT_CLOCK_ARC_OFFSET) as usize;
        for arc_index in 0..clock_arc_count {
            let original_arc_record = original_arc_offset + arc_index * CLOCK_ARC_RECORD_LEN;
            let relocated_arc_record = relocate_offset(original_arc_record);
            for field_offset in CLOCK_ARC_ARRAY_OFFSETS {
                let original_array_offset =
                    read_u64_at(bytes, original_arc_record + field_offset) as usize;
                write_u64_at(
                    &mut legacy,
                    relocated_arc_record + field_offset,
                    relocate_offset(original_array_offset) as u64,
                );
            }
        }

        let payload_checksum = fnv1a64(&legacy[new_data_offset..new_data_offset + new_data_len]);
        write_u64_at(
            &mut legacy,
            record_offset + SAT_CHECKSUM_OFFSET,
            payload_checksum,
        );
    }

    write_u16_at(&mut legacy, HEADER_VERSION_OFFSET, V1_STORE_VERSION);
    let legacy_total_len = legacy.len() as u64;
    write_u64_at(&mut legacy, HEADER_TOTAL_LEN_OFFSET, legacy_total_len);
    write_u64_at(&mut legacy, HEADER_CHECKSUM_OFFSET, 0);
    let checksum = legacy_v1_checksum64(&legacy);
    write_u64_at(&mut legacy, HEADER_CHECKSUM_OFFSET, checksum);
    legacy
}

/// Open `bytes` on the verified path and on the attested path, the attested
/// claim being the checksum the header declares.
fn open_both_ways(
    bytes: &[u8],
) -> [Result<MmapPreciseEphemerisInterpolant<'static>, PreciseInterpolantStoreError>; 2] {
    let claimed = if bytes.len() >= HEADER_CHECKSUM_OFFSET + 8 {
        header_checksum(bytes)
    } else {
        0
    };
    [
        MmapPreciseEphemerisInterpolant::from_vec(bytes.to_vec()),
        MmapPreciseEphemerisInterpolant::from_vec_attested(bytes.to_vec(), claimed),
    ]
}

#[test]
fn precise_interpolant_store_reports_framing_by_cause_on_both_paths() {
    let bytes = fixture_sp3()
        .precise_interpolant_store_bytes()
        .expect("build precise interpolant artifact");
    let len = bytes.len() as u64;

    for cut in [bytes.len() - 1, bytes.len() - 4096, STORE_HEADER_LEN] {
        for result in open_both_ways(&bytes[..cut]) {
            assert_eq!(
                result.err(),
                Some(PreciseInterpolantStoreError::Truncated {
                    declared: len,
                    available: cut as u64,
                }),
                "cut at {cut}"
            );
        }
    }
    for cut in [0, 7, 8, 40, STORE_HEADER_LEN - 1] {
        for result in open_both_ways(&bytes[..cut]) {
            assert_eq!(
                result.err(),
                Some(PreciseInterpolantStoreError::HeaderTruncated {
                    available: cut as u64,
                }),
                "cut at {cut}"
            );
        }
    }

    let mut longer = bytes.clone();
    longer.push(0);
    for result in open_both_ways(&longer) {
        assert_eq!(
            result.err(),
            Some(PreciseInterpolantStoreError::TrailingBytes {
                declared: len,
                available: len + 1,
            })
        );
    }

    let mut foreign = bytes.clone();
    foreign[..8].copy_from_slice(b"NOTASTOR");
    for result in open_both_ways(&foreign) {
        assert_eq!(
            result.err(),
            Some(PreciseInterpolantStoreError::BadMagic {
                found: *b"NOTASTOR"
            })
        );
    }
}

#[test]
fn precise_interpolant_store_reports_a_corrupt_span_by_cause_on_both_paths() {
    let bytes = fixture_sp3()
        .precise_interpolant_store_bytes()
        .expect("build precise interpolant artifact");
    let first_sat = MmapPreciseEphemerisInterpolant::from_vec(bytes.clone())
        .expect("pristine store opens")
        .satellites()[0];

    // The first index record's payload length, grown past the store. The
    // store keeps its length, so this is corruption, not truncation.
    let data_offset_at = STORE_HEADER_LEN + SAT_DATA_OFFSET_OFFSET;
    let data_len_at = STORE_HEADER_LEN + SAT_DATA_LEN_OFFSET;
    let data_offset = u64::from_le_bytes(
        bytes[data_offset_at..data_offset_at + 8]
            .try_into()
            .unwrap(),
    );
    let mut corrupt = bytes.clone();
    corrupt[data_len_at..data_len_at + 8].copy_from_slice(&(1u64 << 40).to_le_bytes());

    let [verified, attested] = open_both_ways(&corrupt);
    assert!(
        matches!(
            verified.err(),
            Some(PreciseInterpolantStoreError::Checksum { .. })
        ),
        "the verified path hashes the store first"
    );
    assert_eq!(
        attested.err(),
        Some(PreciseInterpolantStoreError::RangeOutOfBounds {
            region: "satellite data",
            sat: Some(first_sat),
            offset: data_offset,
            len: 1 << 40,
            available: bytes.len() as u64,
        })
    );

    // A payload byte flipped in place keeps every length: the verified path
    // names the checksum.
    let mut flipped = bytes.clone();
    let last = flipped.len() - 1;
    flipped[last] ^= 0x80;
    let [verified, _] = open_both_ways(&flipped);
    assert!(matches!(
        verified.err(),
        Some(PreciseInterpolantStoreError::Checksum { .. })
    ));
}

/// Midpoint of the G01 hole, bracketed by the 07:15 and 10:15 nodes.
const GAP_MID_HOLE_J2000_S: f64 = 646_260_300.0;
/// Header bytes 48..56 hold the gap threshold factor, all-zero for the default.
const HEADER_GAP_THRESHOLD_FACTOR_OFFSET: usize = 48;
const HEADER_CHECKSUM_OFFSET: usize = 40;
const STORE_HEADER_LEN: usize = 64;
/// Byte offsets of a satellite's payload offset and length in its 96-byte
/// index record.
const SAT_DATA_OFFSET_OFFSET: usize = 64;
const SAT_DATA_LEN_OFFSET: usize = 72;

fn gapped_sp3() -> Sp3 {
    let bytes = fs::read(fixture_path(GAP_15M_FIXTURE)).expect("read gapped SP3 fixture");
    Sp3::parse(&bytes).expect("parse gapped SP3 fixture")
}

#[test]
fn precise_interpolant_store_carries_a_non_default_gap_threshold() {
    let default_bytes = PreciseEphemerisInterpolant::from_sp3(&gapped_sp3())
        .to_mmap_store_bytes()
        .expect("default artifact");
    // V2 encodes the default gap policy with a zero-valued header field.
    assert!(
        default_bytes[HEADER_GAP_THRESHOLD_FACTOR_OFFSET..STORE_HEADER_LEN]
            .iter()
            .all(|&b| b == 0)
    );
    let default_mapped = MmapPreciseEphemerisInterpolant::from_vec(default_bytes).expect("open");
    assert_eq!(
        default_mapped.interpolation_options(),
        Sp3InterpolationOptions::default()
    );
    assert!(default_mapped
        .position_at_j2000_seconds(gps(1), GAP_MID_HOLE_J2000_S)
        .is_err());

    // The hole is twelve nominal spacings wide; 13 bridges it.
    let wide = Sp3InterpolationOptions::new(13.0).expect("valid policy");
    let memory =
        PreciseEphemerisInterpolant::from_sp3(&gapped_sp3().with_interpolation_options(wide));
    let wide_bytes = memory.to_mmap_store_bytes().expect("wide artifact");
    assert_eq!(
        f64::from_le_bytes(
            wide_bytes[HEADER_GAP_THRESHOLD_FACTOR_OFFSET..HEADER_GAP_THRESHOLD_FACTOR_OFFSET + 8]
                .try_into()
                .unwrap()
        ),
        13.0
    );
    let mapped = MmapPreciseEphemerisInterpolant::from_vec(wide_bytes).expect("open wide");
    assert_eq!(mapped.interpolation_options(), wide);

    let want = memory
        .position_at_j2000_seconds(gps(1), GAP_MID_HOLE_J2000_S)
        .expect("in-memory bridges the hole");
    let got = mapped
        .position_at_j2000_seconds(gps(1), GAP_MID_HOLE_J2000_S)
        .expect("mapped bridges the hole");
    assert_state_bits_eq(gps(1), GAP_MID_HOLE_J2000_S, got, want);
}

#[test]
fn precise_interpolant_store_rejects_an_unusable_gap_threshold() {
    let pristine = PreciseEphemerisInterpolant::from_sp3(&gapped_sp3())
        .to_mmap_store_bytes()
        .expect("default artifact");
    for factor in [1.0, 0.5, -1.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut bytes = pristine.clone();
        bytes[HEADER_GAP_THRESHOLD_FACTOR_OFFSET..HEADER_GAP_THRESHOLD_FACTOR_OFFSET + 8]
            .copy_from_slice(&factor.to_le_bytes());
        let checksum = precise_interpolant_store_checksum64(&bytes);
        bytes[HEADER_CHECKSUM_OFFSET..HEADER_CHECKSUM_OFFSET + 8]
            .copy_from_slice(&checksum.to_le_bytes());

        let err = MmapPreciseEphemerisInterpolant::from_bytes(&bytes)
            .expect_err("a factor the constructor rejects must not open");
        assert!(
            matches!(err, PreciseInterpolantStoreError::Parse { .. }),
            "factor {factor}: expected a parse rejection, got {err:?}"
        );
        assert!(
            err.to_string().contains("gap threshold factor"),
            "factor {factor}: {err}"
        );
    }
}

/// Project V2 default-policy artifacts to legacy V1 layout and compare against
/// the V1 pins captured at 2ddaf0b, before V2 accuracy blocks were added.
#[test]
fn default_policy_v2_projects_to_pinned_legacy_v1_artifacts() {
    let pins = [
        (
            "tests/fixtures/sp3/GAP_G01_20201760000_15M.sp3",
            922_656usize,
            0xf42f31591bdb97bd_u64,
        ),
        (COD_5M_FIXTURE, 2_591_808, 0xa2a11c9de566fda3),
        (
            "tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3",
            926_752,
            0x48ded418f39ddc05,
        ),
    ];
    for (fixture, len, checksum) in pins {
        let sp3 = Sp3::parse(&fs::read(fixture_path(fixture)).expect("read fixture"))
            .expect("parse fixture");
        assert_eq!(
            sp3.interpolation_options(),
            Sp3InterpolationOptions::default()
        );
        let bytes = sp3.precise_interpolant_store_bytes().expect("artifact");
        assert_eq!(read_u16_at(&bytes, HEADER_VERSION_OFFSET), V2_STORE_VERSION);
        let current = MmapPreciseEphemerisInterpolant::from_vec(bytes.clone()).expect("open V2");
        assert_eq!(
            current.interpolation_options(),
            Sp3InterpolationOptions::default()
        );

        let legacy_bytes = project_v2_store_to_v1(&bytes);
        assert_eq!(
            read_u16_at(&legacy_bytes, HEADER_VERSION_OFFSET),
            V1_STORE_VERSION
        );
        assert_eq!(legacy_bytes.len(), len, "{fixture}: legacy V1 length");
        assert_eq!(
            legacy_v1_checksum64(&legacy_bytes),
            checksum,
            "{fixture}: legacy V1 checksum"
        );
        assert_eq!(
            precise_interpolant_store_checksum64(&legacy_bytes),
            checksum,
            "{fixture}: reader checksum agrees with the legacy pin"
        );
        let mapped = MmapPreciseEphemerisInterpolant::from_vec(legacy_bytes)
            .expect("open legacy V1 artifact with unknown accuracy");
        assert_eq!(
            mapped.interpolation_options(),
            Sp3InterpolationOptions::default()
        );
        let satellite = mapped.satellites()[0];
        let epoch = sp3.epochs_j2000_seconds()[0];
        assert_eq!(
            sidereon_core::positioning::EphemerisSource::ephemeris_variance_m2(
                &mapped, satellite, epoch, epoch
            ),
            0.0,
            "{fixture}: V1 accuracy is unknown and retains the zero-variance fallback"
        );
    }
}

#[test]
fn projected_v1_preserves_a_non_default_gap_threshold() {
    let mut v2_bytes = PreciseEphemerisInterpolant::from_sp3(&gapped_sp3())
        .to_mmap_store_bytes()
        .expect("default V2 artifact");
    write_u64_at(
        &mut v2_bytes,
        HEADER_GAP_THRESHOLD_FACTOR_OFFSET,
        13.0_f64.to_bits(),
    );
    let v2_checksum = precise_interpolant_store_checksum64(&v2_bytes);
    write_u64_at(&mut v2_bytes, HEADER_CHECKSUM_OFFSET, v2_checksum);

    let v1_bytes = project_v2_store_to_v1(&v2_bytes);
    assert_eq!(
        read_u16_at(&v1_bytes, HEADER_VERSION_OFFSET),
        V1_STORE_VERSION
    );
    let mapped = MmapPreciseEphemerisInterpolant::from_vec(v1_bytes)
        .expect("open projected V1 with non-default gap policy");
    assert_eq!(
        mapped.interpolation_options(),
        Sp3InterpolationOptions::new(13.0).expect("valid gap policy")
    );
}
