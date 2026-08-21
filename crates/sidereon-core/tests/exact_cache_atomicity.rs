#![cfg(not(target_arch = "wasm32"))]

use sidereon_core::data::{
    product, AnalysisCenter, DistributionSource, ProductDate, ProductIdentity, ProductType,
};
#[cfg(feature = "exact-cache-test-failpoints")]
use sidereon_core::exact_cache::ExactCacheTestClock;
use sidereon_core::exact_cache::{
    ExactCacheError, ExactCacheOpen, ExactCacheSingleFlightOptions, ExactProductCache,
};
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

fn single_flight_options() -> ExactCacheSingleFlightOptions {
    ExactCacheSingleFlightOptions {
        poll_interval: Duration::from_millis(10),
        heartbeat_interval: Duration::from_millis(25),
        liveness_timeout: Duration::from_secs(2),
        wait_timeout: Duration::from_secs(10),
    }
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

#[cfg(feature = "exact-cache-test-failpoints")]
fn paused_child(
    mode: &str,
    stable: &Path,
    root: &Path,
    failpoint: &str,
    barrier: &Path,
) -> std::process::Child {
    let mut command = Command::new(std::env::current_exe().expect("test executable"));
    command
        .arg("--exact")
        .arg("exact_cache_subprocess")
        .arg("--nocapture")
        .env("SIDEREON_EXACT_CACHE_CHILD_MODE", mode)
        .env("SIDEREON_EXACT_CACHE_CHILD_STABLE", stable)
        .env("SIDEREON_EXACT_CACHE_CHILD_ROOT", root)
        .env("SIDEREON_TEST_EXACT_CACHE_PAUSE_FAILPOINT", failpoint)
        .env("SIDEREON_TEST_EXACT_CACHE_FAILPOINT_BARRIER", barrier)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command.spawn().expect("spawn paused cache helper")
}

fn count_download(root: &Path) {
    let mut counter = OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join("downloads"))
        .expect("download counter");
    counter.write_all(b"download\n").expect("count download");
    counter.sync_all().expect("sync download counter");
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
                count_download(&root);
                cache
                    .publish(&guard, OLD_PRODUCT, OLD_ARCHIVE, OLD_PROVENANCE)
                    .expect("publish");
            }
        }
        "singleflight_publish" => {
            match cache
                .open_single_flight(single_flight_options())
                .expect("single-flight open")
            {
                ExactCacheOpen::Hit(entry) => {
                    fs::write(root.join("singleflight-result"), entry.product)
                        .expect("write single-flight hit");
                }
                ExactCacheOpen::Owner(owner) => {
                    count_download(&root);
                    owner.heartbeat().expect("download heartbeat");
                    let entry = owner
                        .publish(NEW_PRODUCT, NEW_ARCHIVE, NEW_PROVENANCE)
                        .expect("single-flight publish");
                    fs::write(root.join("singleflight-result"), entry.product)
                        .expect("write single-flight publish result");
                }
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
#[cfg(feature = "exact-cache-test-failpoints")]
fn waiter_observes_the_owners_commit_without_downloading() {
    let root = temp_directory("singleflight-waiter");
    let stable = root.join("cache").join("product.SP3");
    let owner_barrier = root.join("owner-pause");
    let waiter_barrier = root.join("waiter-pause");
    let mut owner = paused_child(
        "singleflight_publish",
        &stable,
        &root,
        "after_inflight_heartbeat_refresh",
        &owner_barrier,
    );
    wait_for(&owner_barrier.with_extension("ready"));
    assert!(root
        .join("cache")
        .join(".sidereon-cache-v3")
        .join("in-flight.json")
        .is_file());

    let mut waiter = paused_child(
        "singleflight_publish",
        &stable,
        &root,
        "after_inflight_wait_observation",
        &waiter_barrier,
    );
    wait_for(&waiter_barrier.with_extension("ready"));
    fs::write(owner_barrier.with_extension("release"), b"release").expect("release owner");
    assert!(owner.wait().expect("owner status").success());
    fs::write(waiter_barrier.with_extension("release"), b"release").expect("release waiter");
    assert!(waiter.wait().expect("waiter status").success());

    let downloads = fs::read_to_string(root.join("downloads")).expect("download counter");
    assert_eq!(downloads.lines().count(), 1);
    assert_eq!(
        fs::read(root.join("singleflight-result")).expect("single-flight result"),
        NEW_PRODUCT
    );
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
#[cfg(all(feature = "exact-cache-test-failpoints", unix))]
fn sigkill_dead_owner_is_taken_over_after_injected_liveness_timeout() {
    let root = temp_directory("singleflight-dead-owner");
    let stable = root.join("cache").join("product.SP3");
    let owner_barrier = root.join("dead-owner-pause");
    let mut owner = paused_child(
        "singleflight_publish",
        &stable,
        &root,
        "after_inflight_heartbeat_refresh",
        &owner_barrier,
    );
    wait_for(&owner_barrier.with_extension("ready"));
    assert!(root
        .join("cache")
        .join(".sidereon-cache-v3")
        .join("in-flight.json")
        .is_file());
    owner.kill().expect("SIGKILL dead owner");
    assert!(!owner.wait().expect("dead owner status").success());

    let clock = ExactCacheTestClock::new();
    let waiter_clock = clock.clone();
    let waiter_root = root.clone();
    let waiter_stable = stable.clone();
    let takeover = thread::spawn(move || {
        let options = ExactCacheSingleFlightOptions {
            poll_interval: Duration::from_secs(1),
            heartbeat_interval: Duration::from_secs(1),
            liveness_timeout: Duration::from_secs(5),
            wait_timeout: Duration::from_secs(10),
        };
        let owner = match cache(waiter_stable)
            .open_single_flight_with_test_clock(options, &waiter_clock)
            .expect("dead-owner takeover")
        {
            ExactCacheOpen::Owner(owner) => owner,
            ExactCacheOpen::Hit(_) => panic!("dead owner unexpectedly committed"),
        };
        count_download(&waiter_root);
        owner
            .publish(NEW_PRODUCT, NEW_ARCHIVE, NEW_PROVENANCE)
            .expect("takeover publish")
    });
    clock.wait_for_sleepers(1);
    clock.advance(Duration::from_secs(5));
    let entry = takeover.join().expect("takeover thread");
    assert_eq!(entry.product, NEW_PRODUCT);

    let downloads = fs::read_to_string(root.join("downloads")).expect("download counter");
    assert_eq!(downloads.lines().count(), 2);
    let entries = root
        .join("cache")
        .join(".sidereon-cache-v3")
        .join("entries");
    assert_eq!(fs::read_dir(entries).expect("entries").count(), 1);
    assert_eq!(
        cache(&stable)
            .read()
            .expect("coherent read")
            .expect("committed takeover")
            .product,
        NEW_PRODUCT
    );
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
#[cfg(feature = "exact-cache-test-failpoints")]
fn live_owner_total_wait_timeout_does_not_create_a_second_owner() {
    let root = temp_directory("singleflight-timeout");
    let stable = root.join("cache").join("product.SP3");
    let first = match cache(&stable)
        .open_single_flight(single_flight_options())
        .expect("first owner")
    {
        ExactCacheOpen::Owner(owner) => owner,
        ExactCacheOpen::Hit(_) => panic!("cold cache returned a hit"),
    };

    let clock = ExactCacheTestClock::new();
    let waiter_clock = clock.clone();
    let waiter_stable = stable.clone();
    let waiter = thread::spawn(move || {
        cache(waiter_stable).open_single_flight_with_test_clock(
            ExactCacheSingleFlightOptions {
                poll_interval: Duration::from_secs(1),
                heartbeat_interval: Duration::from_secs(5),
                liveness_timeout: Duration::from_secs(10),
                wait_timeout: Duration::from_secs(3),
            },
            &waiter_clock,
        )
    });
    clock.wait_for_sleepers(1);
    clock.advance(Duration::from_secs(3));
    assert!(matches!(
        waiter.join().expect("waiter thread"),
        Err(ExactCacheError::SingleFlightTimeout)
    ));
    first.heartbeat().expect("first owner remains live");
    drop(first);
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
    assert!(!root
        .join("cache")
        .join(".sidereon-cache-v3")
        .join("in-flight.json")
        .exists());
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn single_flight_owner_publishes_and_the_next_open_is_a_hit() {
    let root = temp_directory("singleflight-cold-hit");
    let stable = root.join("cache").join("product.SP3");
    let cache = cache(&stable);
    let owner = match cache
        .open_single_flight(single_flight_options())
        .expect("cold single-flight open")
    {
        ExactCacheOpen::Owner(owner) => owner,
        ExactCacheOpen::Hit(_) => panic!("cold cache returned a hit"),
    };
    let published = owner
        .publish(OLD_PRODUCT, OLD_ARCHIVE, OLD_PROVENANCE)
        .expect("owner publish");
    assert_eq!(published.product, OLD_PRODUCT);

    let hit = match cache
        .open_single_flight(single_flight_options())
        .expect("warm single-flight open")
    {
        ExactCacheOpen::Hit(hit) => hit,
        ExactCacheOpen::Owner(_) => panic!("warm cache returned an owner"),
    };
    assert_eq!(hit.product, OLD_PRODUCT);
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
#[cfg(feature = "exact-cache-test-failpoints")]
fn process_death_at_each_single_flight_boundary_preserves_committed_atomicity() {
    for step in [
        "after_inflight_heartbeat",
        "after_inflight_marker",
        "after_inflight_sync",
        "after_inflight_heartbeat_refresh",
        "after_inflight_retire",
        "after_inflight_retire_sync",
        "after_inflight_reap",
        "after_inflight_reap_sync",
    ] {
        let root = temp_directory(step);
        let stable = root.join("cache").join("product.SP3");
        let cache = cache(&stable);

        let mut publisher = Command::new(std::env::current_exe().expect("test executable"))
            .arg("--exact")
            .arg("exact_cache_subprocess")
            .arg("--nocapture")
            .env("SIDEREON_EXACT_CACHE_CHILD_MODE", "singleflight_publish")
            .env("SIDEREON_EXACT_CACHE_CHILD_STABLE", &stable)
            .env("SIDEREON_EXACT_CACHE_CHILD_ROOT", &root)
            .env("SIDEREON_TEST_EXACT_CACHE_FAILPOINT", step)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn crashing single-flight publisher");
        assert_eq!(
            publisher.wait().expect("publisher status").code(),
            Some(86),
            "single-flight failpoint did not fire at {step}"
        );

        if let Some(entry) = cache.read().expect("coherent read") {
            assert_eq!(entry.product, NEW_PRODUCT, "product after {step}");
            assert_eq!(entry.archive, NEW_ARCHIVE, "archive after {step}");
            assert_eq!(entry.provenance, NEW_PROVENANCE, "provenance after {step}");
        }
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
