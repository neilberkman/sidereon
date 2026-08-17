use std::fs;
use std::mem;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use sidereon_core::ephemeris::{
    precise_interpolant_store_checksum64, MmapPreciseEphemerisInterpolant,
    PreciseEphemerisInterpolant, PreciseInterpolantStoreError, Sp3,
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

    let truncated = &bytes[..bytes.len() - 1];
    let err = MmapPreciseEphemerisInterpolant::from_bytes(truncated)
        .expect_err("truncated artifact must fail");
    assert!(matches!(
        err,
        PreciseInterpolantStoreError::Checksum { .. } | PreciseInterpolantStoreError::Parse { .. }
    ));
}
