use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use sidereon_core::ephemeris::{
    MmapPreciseEphemerisInterpolant, PreciseInterpolantStoreError, Sp3,
};
use sidereon_core::terrain_store::{
    dted_tile_list_to_mmap_store, terrain_store_checksum64, DtedTileListEntry, MmapTerrain,
    TerrainStoreError,
};
use sidereon_core::DigestProvenance;

const HEADER_INDEX_OFFSET_OFFSET: usize = 16;
const HEADER_DATA_OFFSET_OFFSET: usize = 24;
const INTERPOLANT_HEADER_CHECKSUM_OFFSET: usize = 40;
const INTERPOLANT_POS_KX_OFFSET_OFFSET: usize = 24;

fn terrain_store() -> &'static [u8] {
    static STORE: OnceLock<Vec<u8>> = OnceLock::new();
    STORE
        .get_or_init(|| {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dted/tiles");
            let entries = vec![
                DtedTileListEntry::from_indices(36, -106, root.join("n36_w106_1arc_v3.dt2")),
                DtedTileListEntry::from_indices(36, -107, root.join("n36_w107_1arc_v3.dt2")),
            ];
            dted_tile_list_to_mmap_store(&entries)
                .expect("build terrain store from committed DTED tiles")
        })
        .as_slice()
}

fn interpolant_store() -> &'static [u8] {
    static STORE: OnceLock<Vec<u8>> = OnceLock::new();
    STORE
        .get_or_init(|| {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3");
            let text = std::fs::read(&path).expect("read committed IGS fixture");
            Sp3::parse(&text)
                .expect("parse committed IGS fixture")
                .precise_interpolant_store_bytes()
                .expect("build interpolant artifact")
        })
        .as_slice()
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("fixture contains u64 field"),
    )
}

fn temp_path(file_name: &str) -> PathBuf {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "sidereon-attested-open-{}-{}",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("create temporary directory");
    dir.join(file_name)
}

#[test]
fn terrain_attested_open_skips_payload_hashing_and_verify_detects_corruption() {
    let mut corrupt = terrain_store().to_vec();
    let data_offset = read_u64(&corrupt, HEADER_DATA_OFFSET_OFFSET) as usize;
    corrupt[data_offset + 1] ^= 1;
    let path = temp_path("corrupt-terrain.tmm");
    std::fs::write(&path, &corrupt).expect("write corrupt terrain store");

    let verified_error = MmapTerrain::from_path(&path).expect_err("verified open must fail");
    assert!(matches!(verified_error, TerrainStoreError::Checksum { .. }));

    let claim = 0x0123_4567_89ab_cdef;
    let mut attested =
        MmapTerrain::from_path_attested(&path, claim).expect("attested open must skip tile hashes");
    assert_eq!(attested.digest_provenance(), DigestProvenance::Attested);
    assert_eq!(attested.checksum64(), claim);
    assert_eq!(attested.is_memory_mapped(), cfg!(feature = "mmap"));

    let verify_error = attested.verify().expect_err("verification must fail");
    assert_eq!(verify_error, verified_error);
    assert_eq!(attested.digest_provenance(), DigestProvenance::Attested);
}

#[test]
fn interpolant_attested_open_skips_hashing_and_verify_detects_corruption() {
    let mut corrupt = interpolant_store().to_vec();
    let declared = read_u64(&corrupt, INTERPOLANT_HEADER_CHECKSUM_OFFSET);
    let index_offset = read_u64(&corrupt, HEADER_INDEX_OFFSET_OFFSET) as usize;
    let pos_kx_offset =
        read_u64(&corrupt, index_offset + INTERPOLANT_POS_KX_OFFSET_OFFSET) as usize;
    corrupt[pos_kx_offset + 1] ^= 1;
    let path = temp_path("corrupt-interpolant.spi");
    std::fs::write(&path, &corrupt).expect("write corrupt interpolant store");

    let verified_error =
        MmapPreciseEphemerisInterpolant::from_path(&path).expect_err("verified open must fail");
    assert!(matches!(
        verified_error,
        PreciseInterpolantStoreError::Checksum { .. }
            | PreciseInterpolantStoreError::SatelliteChecksum { .. }
    ));

    let mut attested = MmapPreciseEphemerisInterpolant::from_path_attested(&path, declared)
        .expect("attested open must skip payload hashes");
    assert_eq!(attested.digest_provenance(), DigestProvenance::Attested);
    assert_eq!(attested.checksum64(), declared);
    assert_eq!(attested.is_memory_mapped(), cfg!(feature = "mmap"));

    let verify_error = attested.verify().expect_err("verification must fail");
    assert_eq!(verify_error, verified_error);
    assert_eq!(attested.digest_provenance(), DigestProvenance::Attested);
}

#[test]
fn interpolant_attested_open_rejects_a_claim_that_differs_from_the_header() {
    let artifact = interpolant_store();
    let declared = read_u64(artifact, INTERPOLANT_HEADER_CHECKSUM_OFFSET);
    let claimed = declared ^ 1;
    let path = temp_path("pristine-interpolant.spi");
    std::fs::write(&path, artifact).expect("write interpolant store");

    let error = MmapPreciseEphemerisInterpolant::from_path_attested(&path, claimed)
        .expect_err("header mismatch must fail closed");
    assert_eq!(
        error,
        PreciseInterpolantStoreError::AttestedChecksumMismatch { claimed, declared }
    );
    let message = error.to_string();
    assert!(message.contains(&format!("{claimed:#x}")));
    assert!(message.contains(&format!("{declared:#x}")));
}

#[test]
fn terrain_attested_identity_verifies_and_legacy_constructors_stay_verified() {
    let store = terrain_store();
    let claim = terrain_store_checksum64(store);
    let path = temp_path("pristine-terrain.tmm");
    std::fs::write(&path, store).expect("write terrain store");

    let mut verified = MmapTerrain::from_path(&path).expect("open verified terrain store");
    let mut attested =
        MmapTerrain::from_path_attested(&path, claim).expect("open attested terrain store");
    let owned = MmapTerrain::from_vec(store.to_vec()).expect("open owned terrain store");
    let borrowed = MmapTerrain::from_bytes(store).expect("open borrowed terrain store");

    assert_eq!(verified.digest_provenance(), DigestProvenance::Verified);
    assert_eq!(owned.digest_provenance(), DigestProvenance::Verified);
    assert_eq!(borrowed.digest_provenance(), DigestProvenance::Verified);
    assert_eq!(attested.digest_provenance(), DigestProvenance::Attested);
    assert_eq!(attested.checksum64(), claim);
    assert!(format!("{attested:?}").contains("digest_provenance: Attested"));

    for (longitude_deg, latitude_deg) in [
        (-105.5, 36.5),
        (-105.25, 36.75),
        (-106.5, 36.5),
        (-106.75, 36.25),
        (-105.999, 36.001),
    ] {
        let expected = verified
            .height_m(longitude_deg, latitude_deg)
            .expect("verified terrain query succeeds");
        let found = attested
            .height_m(longitude_deg, latitude_deg)
            .expect("attested terrain query succeeds");
        assert_eq!(found.to_bits(), expected.to_bits());
    }

    attested.verify().expect("attested terrain verifies");
    assert_eq!(attested.digest_provenance(), DigestProvenance::Verified);
    assert_eq!(attested.checksum64(), claim);
    attested.verify().expect("verified terrain re-verifies");

    let wrong_claim = claim ^ 1;
    let mut wrong = MmapTerrain::from_path_attested(&path, wrong_claim)
        .expect("terrain headers have no checksum cross-check");
    assert_eq!(wrong.checksum64(), wrong_claim);
    assert_eq!(
        wrong.verify().expect_err("wrong terrain claim must fail"),
        TerrainStoreError::AttestedChecksumMismatch {
            expected: wrong_claim,
            found: claim,
        }
    );
}

#[test]
fn interpolant_attested_identity_verifies_and_legacy_constructors_stay_verified() {
    let artifact = interpolant_store();
    let declared = read_u64(artifact, INTERPOLANT_HEADER_CHECKSUM_OFFSET);
    let path = temp_path("identity-interpolant.spi");
    std::fs::write(&path, artifact).expect("write interpolant store");

    let verified =
        MmapPreciseEphemerisInterpolant::from_path(&path).expect("open verified interpolant");
    let mut attested = MmapPreciseEphemerisInterpolant::from_path_attested(&path, declared)
        .expect("open attested interpolant");
    let owned = MmapPreciseEphemerisInterpolant::from_vec(artifact.to_vec())
        .expect("open owned interpolant");
    let borrowed =
        MmapPreciseEphemerisInterpolant::from_bytes(artifact).expect("open borrowed interpolant");

    assert_eq!(verified.digest_provenance(), DigestProvenance::Verified);
    assert_eq!(owned.digest_provenance(), DigestProvenance::Verified);
    assert_eq!(borrowed.digest_provenance(), DigestProvenance::Verified);
    assert_eq!(attested.digest_provenance(), DigestProvenance::Attested);
    assert_eq!(attested.checksum64(), declared);
    let debug = format!("{attested:?}");
    assert!(debug.contains("digest_provenance: Attested"));
    assert!(debug.contains("attested_checksum64: Some"));

    for &sat in verified.satellites() {
        for query in [646_315_200.0, 646_336_800.0, 646_358_400.0] {
            let expected = verified.position_at_j2000_seconds(sat, query);
            let found = attested.position_at_j2000_seconds(sat, query);
            match (expected, found) {
                (Ok(expected), Ok(found)) => {
                    for axis in 0..3 {
                        assert_eq!(
                            found.position.as_array()[axis].to_bits(),
                            expected.position.as_array()[axis].to_bits()
                        );
                    }
                    assert_eq!(
                        found.clock_s.map(f64::to_bits),
                        expected.clock_s.map(f64::to_bits)
                    );
                }
                (Err(_), Err(_)) => {}
                outcomes => panic!("verified and attested queries differ: {outcomes:?}"),
            }
        }
    }

    attested.verify().expect("attested interpolant verifies");
    assert_eq!(attested.digest_provenance(), DigestProvenance::Verified);
    assert_eq!(attested.checksum64(), declared);
    attested.verify().expect("verified interpolant re-verifies");
}

/// The byte-based attested constructors carry the identical contract as the
/// path-based ones. They exist for interface layers that hold bytes rather
/// than a path, and were added for the wasm binding; this pins them in core so
/// their behavior is owned here rather than implied by a downstream consumer.
#[test]
fn byte_based_attested_constructors_match_the_path_based_contract() {
    // Terrain: corrupted payload opens attested, fails verify.
    let mut corrupt = terrain_store().to_vec();
    let data_offset = read_u64(&corrupt, HEADER_DATA_OFFSET_OFFSET) as usize;
    corrupt[data_offset + 1] ^= 1;
    assert!(matches!(
        MmapTerrain::from_vec(corrupt.clone()),
        Err(TerrainStoreError::Checksum { .. })
    ));
    let mut terrain = MmapTerrain::from_vec_attested(corrupt, 0xDEAD_BEEF)
        .expect("attested byte open skips payload hashing");
    assert_eq!(terrain.digest_provenance(), DigestProvenance::Attested);
    assert_eq!(terrain.checksum64(), 0xDEAD_BEEF);
    assert!(terrain.verify().is_err());

    // Interpolant: claim must match the header declaration, byte path too.
    let pristine = interpolant_store().to_vec();
    let declared = read_u64(&pristine, INTERPOLANT_HEADER_CHECKSUM_OFFSET);
    assert!(matches!(
        MmapPreciseEphemerisInterpolant::from_vec_attested(pristine.clone(), declared ^ 1),
        Err(PreciseInterpolantStoreError::AttestedChecksumMismatch { .. })
    ));
    let mut artifact = MmapPreciseEphemerisInterpolant::from_vec_attested(pristine, declared)
        .expect("attested byte open with the declared checksum");
    assert_eq!(artifact.digest_provenance(), DigestProvenance::Attested);
    artifact.verify().expect("pristine artifact verifies");
    assert_eq!(artifact.digest_provenance(), DigestProvenance::Verified);
}
