//! Opening a terrain store from a path must not allocate the artifact.
//!
//! This lives in its own test binary on purpose. The instrument is a global
//! allocator counter, which is process-wide: any other test allocating
//! concurrently in the same process pollutes the measurement, and the result is
//! a test that passes serially and fails under the parallel harness. One test
//! per process is what makes the number mean what it says.

#![cfg(feature = "mmap")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use sidereon_core::terrain_store::{dted_tile_list_to_mmap_store, DtedTileListEntry, MmapTerrain};

/// Counts bytes allocated while armed.
///
/// This is the instrument that makes "the whole file is not read" checkable
/// rather than asserted: `fs::read` must allocate the file's length, a mapping
/// must not.
struct CountingAllocator;

static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) && new_size > layout.size() {
            ALLOCATED.fetch_add(new_size - layout.size(), Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

fn measure<T>(body: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATED.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    let value = body();
    ARMED.store(false, Ordering::Relaxed);
    (value, ALLOCATED.load(Ordering::Relaxed))
}

fn fixture_store() -> Vec<u8> {
    let root =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dted/tiles");
    let entries = vec![
        DtedTileListEntry::from_indices(36, -106, root.join("n36_w106_1arc_v3.dt2")),
        DtedTileListEntry::from_indices(36, -107, root.join("n36_w107_1arc_v3.dt2")),
    ];
    dted_tile_list_to_mmap_store(&entries).expect("build terrain store from committed DTED tiles")
}

#[test]
fn opening_from_a_path_does_not_allocate_the_file() {
    let store = fixture_store();
    let dir = tempdir();
    let path = dir.join("terrain.tmm");
    std::fs::write(&path, &store).expect("write store");

    // Warm up both paths first: the first measured region in the process also
    // pays for one-time lazy initialization, which would be charged to
    // whichever path happened to run first and swamp the comparison.
    drop(MmapTerrain::from_path(&path).expect("warm up mapped"));
    drop(MmapTerrain::from_vec(store.clone()).expect("warm up owned"));

    let (reader, allocated) = measure(|| MmapTerrain::from_path(&path).expect("open mapped"));

    assert!(
        reader.is_memory_mapped(),
        "the path constructor must map the file, not read it"
    );
    assert_eq!(
        reader.as_bytes().len(),
        store.len(),
        "the mapped span must cover the whole artifact"
    );
    // Self-validating: measure a path that provably does copy, so the counter
    // is shown to discriminate rather than merely reporting a small number.
    // Without this, an instrument that always read zero would pass.
    let (_owned, owned_allocated) =
        measure(|| MmapTerrain::from_vec(store.clone()).expect("open owned"));
    assert!(
        owned_allocated >= store.len(),
        "the counter failed to observe a known copy: {owned_allocated} bytes for a {} byte store",
        store.len()
    );

    println!(
        "store={} bytes; mapped open allocated {}; copying open allocated {}",
        store.len(),
        allocated,
        owned_allocated
    );
    assert!(
        allocated < store.len(),
        "opening allocated {allocated} bytes for a {} byte store; a mapping must not \
         allocate the artifact (the copying path allocated {owned_allocated})",
        store.len()
    );
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
