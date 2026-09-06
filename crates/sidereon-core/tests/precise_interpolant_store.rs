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

    let truncated = &bytes[..bytes.len() - 1];
    let err = MmapPreciseEphemerisInterpolant::from_bytes(truncated)
        .expect_err("truncated artifact must fail");
    assert!(matches!(
        err,
        PreciseInterpolantStoreError::Checksum { .. } | PreciseInterpolantStoreError::Parse { .. }
    ));
}

/// Midpoint of the G01 hole, bracketed by the 07:15 and 10:15 nodes.
const GAP_MID_HOLE_J2000_S: f64 = 646_260_300.0;
/// Header bytes 48..56 hold the gap threshold factor, all-zero for the default.
const HEADER_GAP_THRESHOLD_FACTOR_OFFSET: usize = 48;
const HEADER_CHECKSUM_OFFSET: usize = 40;
const STORE_HEADER_LEN: usize = 64;

fn gapped_sp3() -> Sp3 {
    let bytes = fs::read(fixture_path(GAP_15M_FIXTURE)).expect("read gapped SP3 fixture");
    Sp3::parse(&bytes).expect("parse gapped SP3 fixture")
}

#[test]
fn precise_interpolant_store_carries_a_non_default_gap_threshold() {
    let default_bytes = PreciseEphemerisInterpolant::from_sp3(&gapped_sp3())
        .to_mmap_store_bytes()
        .expect("default artifact");
    // A default-policy artifact leaves the header field zero, so its bytes are
    // what they were before the field existed.
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

/// Default-policy artifacts are byte-identical to those written before the
/// header carried a gap threshold factor. Lengths and checksums computed at
/// 2ddaf0b, the last commit without the field, on the same fixtures.
#[test]
fn default_policy_artifacts_are_byte_identical_to_those_written_before_the_header_field() {
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
        assert_eq!(bytes.len(), len, "{fixture}: length");
        assert_eq!(
            precise_interpolant_store_checksum64(&bytes),
            checksum,
            "{fixture}: checksum"
        );
        // And such an artifact opens as the default policy, which is also how
        // one written before the field existed opens.
        let mapped = MmapPreciseEphemerisInterpolant::from_vec(bytes).expect("open");
        assert_eq!(
            mapped.interpolation_options(),
            Sp3InterpolationOptions::default()
        );
    }
}
