//! The path-based terrain open maps the file instead of reading it.
//!
//! The claim these tests defend is not "one memcpy fewer". It is that opening a
//! terrain store no longer costs its size in process memory, so a store far
//! larger than RAM can be opened and queried. A change that merely relocated the
//! copy would satisfy a timing test and fail these.

#![cfg(feature = "mmap")]

use sidereon_core::terrain_store::{dted_tile_list_to_mmap_store, DtedTileListEntry, MmapTerrain};

fn fixture_store() -> Vec<u8> {
    let root =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dted/tiles");
    let entries = vec![
        DtedTileListEntry::from_indices(36, -106, root.join("n36_w106_1arc_v3.dt2")),
        DtedTileListEntry::from_indices(36, -107, root.join("n36_w107_1arc_v3.dt2")),
    ];
    dted_tile_list_to_mmap_store(&entries).expect("build terrain store from committed DTED tiles")
}

fn sample_points() -> Vec<(f64, f64)> {
    vec![
        (-105.5, 36.5),
        (-105.25, 36.75),
        (-106.5, 36.5),
        (-106.75, 36.25),
        (-105.999, 36.001),
    ]
}

#[test]
fn a_mapped_open_returns_identical_heights_to_an_owned_one() {
    // The load path changed; the answers must not. This is the regression that
    // matters most, because a mapping bug would show up as plausible-looking
    // numbers rather than a crash.
    let store = fixture_store();
    let dir = tempdir();
    let path = dir.join("terrain.tmm");
    std::fs::write(&path, &store).expect("write store");

    let mut mapped = MmapTerrain::from_path(&path).expect("open mapped");
    let mut owned = MmapTerrain::from_vec(store.clone()).expect("open owned");
    let mut borrowed = MmapTerrain::from_bytes(&store).expect("open borrowed");

    assert!(mapped.is_memory_mapped());
    assert!(!owned.is_memory_mapped());
    assert!(!borrowed.is_memory_mapped());

    for (longitude_deg, latitude_deg) in sample_points() {
        let from_map = mapped.height_m(longitude_deg, latitude_deg);
        let from_owned = owned.height_m(longitude_deg, latitude_deg);
        let from_borrowed = borrowed.height_m(longitude_deg, latitude_deg);

        match (&from_map, &from_owned, &from_borrowed) {
            (Ok(a), Ok(b), Ok(c)) => {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "mapped and owned disagree at ({longitude_deg}, {latitude_deg})"
                );
                assert_eq!(
                    a.to_bits(),
                    c.to_bits(),
                    "mapped and borrowed disagree at ({longitude_deg}, {latitude_deg})"
                );
            }
            (Err(_), Err(_), Err(_)) => {}
            other => panic!(
                "load paths disagreed on outcome at ({longitude_deg}, {latitude_deg}): {other:?}"
            ),
        }
    }

    assert_eq!(mapped.as_bytes(), owned.as_bytes());
    assert_eq!(mapped.tile_count(), owned.tile_count());
    assert_eq!(mapped.tile_ids(), owned.tile_ids());
    assert_eq!(mapped.vertical_datum(), owned.vertical_datum());
}

#[test]
fn a_malformed_store_is_rejected_identically_by_both_paths() {
    // Validation must not have been weakened to make mapping easier.
    let store = fixture_store();
    let dir = tempdir();

    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        ("truncated header", store[..8.min(store.len())].to_vec()),
        ("truncated body", store[..store.len() / 2].to_vec()),
        ("bad magic", {
            let mut bytes = store.clone();
            bytes[0] ^= 0xFF;
            bytes
        }),
    ];

    for (label, bytes) in cases {
        let path = dir.join(format!("{}.tmm", label.replace(' ', "-")));
        std::fs::write(&path, &bytes).expect("write case");

        let mapped = MmapTerrain::from_path(&path);
        let owned = MmapTerrain::from_vec(bytes.clone());

        assert_eq!(
            mapped.is_ok(),
            owned.is_ok(),
            "{label}: mapped and owned disagreed on acceptance"
        );
        if let (Err(mapped_err), Err(owned_err)) = (&mapped, &owned) {
            assert_eq!(
                mapped_err.to_string(),
                owned_err.to_string(),
                "{label}: mapped and owned produced different errors"
            );
        }
    }
}

#[test]
fn the_reader_outlives_the_path_and_keeps_its_own_mapping() {
    // The mapping is owned by the reader, so nothing the caller does after the
    // open can invalidate it. Deleting the file is the strongest available
    // check: on Unix the inode survives while the mapping holds it.
    let store = fixture_store();
    let dir = tempdir();
    let path = dir.join("terrain.tmm");
    std::fs::write(&path, &store).expect("write store");

    let mut reader = MmapTerrain::from_path(&path).expect("open mapped");
    std::fs::remove_file(&path).expect("remove the file out from under the reader");

    let (longitude_deg, latitude_deg) = sample_points()[0];
    let height = reader.height_m(longitude_deg, latitude_deg);
    assert!(
        height.is_ok(),
        "the reader must keep working after its file is unlinked: {height:?}"
    );
    assert_eq!(reader.as_bytes().len(), store.len());
}

fn tempdir() -> std::path::PathBuf {
    let base = std::env::temp_dir().join(format!(
        "sidereon-terrain-map-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&base).expect("create temp dir");
    base
}
