#![cfg(not(target_arch = "wasm32"))]

use sidereon_core::data::{
    product, AnalysisCenter, DistributionSource, ProductDate, ProductIdentity, ProductType,
};
use sidereon_core::exact_cache::{ExactCacheError, ExactProductCache};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

const OLD_PRODUCT: &[u8] = b"old validated product";
const OLD_ARCHIVE: &[u8] = b"old archive";
const OLD_PROVENANCE: &[u8] = b"{\"acquisition\":\"old\"}";
const NEW_PRODUCT: &[u8] = b"new validated product";
const NEW_ARCHIVE: &[u8] = b"new archive";
const NEW_PROVENANCE: &[u8] = b"{\"acquisition\":\"new\"}";

fn identity() -> ProductIdentity {
    product(
        AnalysisCenter::CodUlt,
        ProductType::Sp3,
        ProductDate::new(2026, 7, 16).expect("date"),
        Some("05M"),
        Some("0000"),
    )
    .expect("product")
    .identity()
    .expect("identity")
}

fn cache(path: impl Into<PathBuf>) -> ExactProductCache {
    ExactProductCache::new(path, identity(), DistributionSource::Direct).expect("cache")
}

fn temp_directory(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let directory = std::env::temp_dir().join(format!(
        "sidereon-exact-cache-{label}-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&directory).expect("create temporary directory");
    directory
}

fn wait_for(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !path.exists() {
        assert!(Instant::now() < deadline, "timed out waiting for {path:?}");
        thread::sleep(Duration::from_millis(5));
    }
}

fn child(mode: &str, stable: &Path, root: &Path) -> std::process::Child {
    let mut command = Command::new(std::env::current_exe().expect("test executable"));
    command
        .arg("--exact")
        .arg("exact_cache_subprocess")
        .arg("--nocapture")
        .env("SIDEREON_EXACT_CACHE_CHILD_MODE", mode)
        .env("SIDEREON_EXACT_CACHE_CHILD_STABLE", stable)
        .env("SIDEREON_EXACT_CACHE_CHILD_ROOT", root)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if mode == "read_barrier" {
        command.env(
            "SIDEREON_TEST_EXACT_CACHE_READ_BARRIER",
            root.join("read-barrier"),
        );
    }
    command.spawn().expect("spawn cache helper")
}

#[test]
fn exact_cache_subprocess() {
    let Some(mode) = std::env::var_os("SIDEREON_EXACT_CACHE_CHILD_MODE") else {
        return;
    };
    let stable =
        PathBuf::from(std::env::var_os("SIDEREON_EXACT_CACHE_CHILD_STABLE").expect("stable path"));
    let root =
        PathBuf::from(std::env::var_os("SIDEREON_EXACT_CACHE_CHILD_ROOT").expect("root path"));
    let cache = cache(stable);
    match mode.to_str().expect("UTF-8 mode") {
        "hold" => {
            let guard = cache.lock(Duration::from_secs(5)).expect("lock");
            let orphan = root
                .join("cache")
                .join(".sidereon-cache-v3")
                .join("entries")
                .join("ffffffffffffffffffffffffffffffff");
            fs::create_dir_all(&orphan).expect("orphan");
            fs::write(root.join("ready"), b"ready").expect("ready");
            wait_for(&root.join("release"));
            drop(guard);
        }
        "publish" => {
            let guard = cache.lock(Duration::from_secs(5)).expect("lock");
            cache
                .publish(&guard, NEW_PRODUCT, NEW_ARCHIVE, NEW_PROVENANCE)
                .expect("publish");
        }
        "publish_if_miss" => {
            wait_for(&root.join("start"));
            let guard = cache.lock(Duration::from_secs(5)).expect("lock");
            if cache.read().expect("read").is_none() {
                let mut counter = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(root.join("downloads"))
                    .expect("download counter");
                counter.write_all(b"download\n").expect("count download");
                counter.sync_all().expect("sync download counter");
                cache
                    .publish(&guard, OLD_PRODUCT, OLD_ARCHIVE, OLD_PROVENANCE)
                    .expect("publish");
            }
        }
        "read_barrier" => {
            let entry = cache.read().expect("coherent read").expect("cache hit");
            fs::write(root.join("read-product"), entry.product).expect("write read result");
        }
        other => panic!("unknown child mode {other}"),
    }
}

#[test]
#[cfg(feature = "exact-cache-test-failpoints")]
fn unlocked_reader_retries_when_cleanup_removes_its_previous_entry() {
    let root = temp_directory("reader-cleanup-race");
    let stable = root.join("cache").join("product.SP3");
    let cache = cache(&stable);
    let guard = cache.lock(Duration::from_secs(1)).expect("initial lock");
    cache
        .publish(&guard, OLD_PRODUCT, OLD_ARCHIVE, OLD_PROVENANCE)
        .expect("initial publish");
    drop(guard);

    let mut reader = child("read_barrier", &stable, &root);
    wait_for(&root.join("read-barrier.ready"));

    let guard = cache.lock(Duration::from_secs(1)).expect("refresh lock");
    cache
        .publish(&guard, NEW_PRODUCT, NEW_ARCHIVE, NEW_PROVENANCE)
        .expect("refresh publish");
    cache
        .cleanup_abandoned(&guard)
        .expect("cleanup previous entry");
    drop(guard);
    fs::write(root.join("read-barrier.release"), b"release").expect("release reader");

    assert!(reader.wait().expect("reader status").success());
    assert_eq!(
        fs::read(root.join("read-product")).expect("read result"),
        NEW_PRODUCT
    );
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn two_processes_publish_one_download_for_the_same_identity() {
    let root = temp_directory("race");
    let stable = root.join("cache").join("product.SP3");
    let mut first = child("publish_if_miss", &stable, &root);
    let mut second = child("publish_if_miss", &stable, &root);
    fs::write(root.join("start"), b"start").expect("release start barrier");
    assert!(first.wait().expect("first status").success());
    assert!(second.wait().expect("second status").success());
    let downloads = fs::read_to_string(root.join("downloads")).expect("download counter");
    assert_eq!(downloads.lines().count(), 1);
    let entry = cache(&stable).read().expect("read").expect("entry");
    assert_eq!(entry.product, OLD_PRODUCT);
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn live_writer_lock_cannot_be_stolen_or_cleaned() {
    let root = temp_directory("live-lock");
    let stable = root.join("cache").join("product.SP3");
    let mut holder = child("hold", &stable, &root);
    wait_for(&root.join("ready"));
    let cache = cache(&stable);
    assert!(matches!(
        cache.lock(Duration::from_millis(50)),
        Err(ExactCacheError::LockTimeout)
    ));
    let orphan = root
        .join("cache")
        .join(".sidereon-cache-v3")
        .join("entries")
        .join("ffffffffffffffffffffffffffffffff");
    assert!(orphan.is_dir());
    fs::write(root.join("release"), b"release").expect("release holder");
    assert!(holder.wait().expect("holder status").success());
    let guard = cache
        .lock(Duration::from_secs(1))
        .expect("lock after release");
    cache.cleanup_abandoned(&guard).expect("cleanup abandoned");
    assert!(!orphan.exists());
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
#[cfg(feature = "exact-cache-test-failpoints")]
fn process_death_at_each_publication_boundary_leaves_old_or_complete_new_entry() {
    for step in [
        "after_payload",
        "after_archive",
        "after_metadata",
        "after_entry_sync",
        "after_marker_write",
        "after_marker_rename",
        "after_commit_sync",
    ] {
        let root = temp_directory(step);
        let stable = root.join("cache").join("product.SP3");
        let cache = cache(&stable);
        let guard = cache.lock(Duration::from_secs(1)).expect("initial lock");
        cache
            .publish(&guard, OLD_PRODUCT, OLD_ARCHIVE, OLD_PROVENANCE)
            .expect("initial publish");
        drop(guard);

        let mut publisher = Command::new(std::env::current_exe().expect("test executable"))
            .arg("--exact")
            .arg("exact_cache_subprocess")
            .arg("--nocapture")
            .env("SIDEREON_EXACT_CACHE_CHILD_MODE", "publish")
            .env("SIDEREON_EXACT_CACHE_CHILD_STABLE", &stable)
            .env("SIDEREON_EXACT_CACHE_CHILD_ROOT", &root)
            .env("SIDEREON_TEST_EXACT_CACHE_FAILPOINT", step)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn crashing publisher");
        assert_eq!(publisher.wait().expect("publisher status").code(), Some(86));

        let entry = cache
            .read()
            .expect("coherent read")
            .expect("old or new entry");
        let accepted = (entry.product == OLD_PRODUCT
            && entry.archive == OLD_ARCHIVE
            && entry.provenance == OLD_PROVENANCE)
            || (entry.product == NEW_PRODUCT
                && entry.archive == NEW_ARCHIVE
                && entry.provenance == NEW_PROVENANCE);
        assert!(accepted, "mixed entry accepted after {step}");
        fs::remove_dir_all(root).expect("cleanup");
    }
}

#[test]
fn reader_rejects_corruption_and_identity_or_source_substitution() {
    let root = temp_directory("corruption");
    let stable = root.join("cache").join("product.SP3");
    let cache = cache(&stable);
    let guard = cache.lock(Duration::from_secs(1)).expect("lock");
    let published = cache
        .publish(&guard, OLD_PRODUCT, OLD_ARCHIVE, OLD_PROVENANCE)
        .expect("publish");
    drop(guard);

    fs::write(&published.product_path, b"corrupt").expect("corrupt payload");
    assert!(matches!(
        cache.read(),
        Err(ExactCacheError::InvalidCommit(_))
    ));
    fs::write(&published.product_path, OLD_PRODUCT).expect("restore payload");

    let other_source = ExactProductCache::new(&stable, identity(), DistributionSource::NasaCddis)
        .expect("other source cache");
    assert!(matches!(
        other_source.read(),
        Err(ExactCacheError::InvalidCommit(_))
    ));

    let mut other_identity = identity();
    other_identity.format_version = Some("SP3-d".to_owned());
    let other_identity =
        ExactProductCache::new(&stable, other_identity, DistributionSource::Direct)
            .expect("other identity cache");
    assert!(matches!(
        other_identity.read(),
        Err(ExactCacheError::InvalidCommit(_))
    ));
    fs::remove_dir_all(root).expect("cleanup");
}
