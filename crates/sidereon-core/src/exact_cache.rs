//! Exact-product cache identity binding and atomic publication.
//!
//! The pure functions in this module define one commit record for every
//! Sidereon interface, including WebAssembly hosts. On native targets,
//! [`ExactProductCache`] adds bounded cross-process locking, immutable entry
//! staging, durable writes, and an atomic reader-visible commit.
//!
//! Network transport and product parsing remain outside this module. Callers
//! must validate product semantics before publication and repeat that semantic
//! validation on bytes returned from a cache hit.

use crate::data::{DataCatalogError, DistributionSource, ProductIdentity};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::time::Duration;

/// Commit-record version shared by all Sidereon interfaces.
pub const EXACT_CACHE_SCHEMA_VERSION: u8 = 3;

/// Directory below one exact identity/source cache directory.
pub const EXACT_CACHE_CONTROL_DIRECTORY: &str = ".sidereon-cache-v3";

/// Name of the single reader-visible commit record.
pub const EXACT_CACHE_MARKER_FILENAME: &str = "current.json";

/// Error produced by the shared exact-product cache protocol.
#[derive(Debug, thiserror::Error)]
pub enum ExactCacheError {
    /// The product identity is internally inconsistent.
    #[error("invalid exact product identity: {0}")]
    Identity(#[from] DataCatalogError),
    /// An immutable transaction identifier is not 32 lower-case hexadecimal characters.
    #[error("invalid exact-cache entry identifier")]
    InvalidEntryId,
    /// A commit record is malformed or does not bind the supplied entry.
    #[error("invalid or mismatched exact-cache commit: {0}")]
    InvalidCommit(&'static str),
    /// A native filesystem operation failed.
    #[cfg(not(target_arch = "wasm32"))]
    #[error("exact-cache {operation} failed: {source}")]
    Io {
        /// Operation that failed.
        operation: &'static str,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },
    /// The per-entry cross-process lock was not acquired in time.
    #[cfg(not(target_arch = "wasm32"))]
    #[error("timed out waiting for the exact-cache lock")]
    LockTimeout,
    /// A live single-flight owner did not commit within the configured wait.
    #[error("timed out waiting for the exact-cache in-flight owner")]
    SingleFlightTimeout,
    /// The single-flight owner token is no longer current or its heartbeat failed.
    #[error("exact-cache single-flight ownership was lost")]
    SingleFlightOwnershipLost,
    /// Single-flight duration options are zero or internally inconsistent.
    #[error("invalid exact-cache single-flight options")]
    InvalidSingleFlightOptions,
    /// Cross-process durable cache publication is unsupported on this platform.
    #[cfg(not(target_arch = "wasm32"))]
    #[error("durable exact-cache publication is unsupported on this platform")]
    UnsupportedPlatform,
}

/// Digests and lengths bound by one exact-cache commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactCacheDigests {
    /// SHA-256 of canonical full [`ProductIdentity`] bytes.
    pub identity_sha256: String,
    /// Explicit distribution source.
    pub distribution_source: String,
    /// SHA-256 of decompressed, validated product bytes.
    pub product_sha256: String,
    /// Product byte length.
    pub product_byte_length: u64,
    /// SHA-256 of distributor archive bytes.
    pub archive_sha256: String,
    /// Archive byte length.
    pub archive_byte_length: u64,
    /// SHA-256 of the exact provenance bytes.
    pub provenance_sha256: String,
    /// Provenance byte length.
    pub provenance_byte_length: u64,
}

/// Parsed, verified commit record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedExactCacheCommit {
    /// Immutable transaction identifier referenced by the marker.
    pub entry_id: String,
    /// Verified identity, source, and byte bindings.
    pub digests: ExactCacheDigests,
}

/// Bounded timing policy for exact-cache single-flight coordination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactCacheSingleFlightOptions {
    /// Interval between committed-entry and heartbeat observations.
    pub poll_interval: Duration,
    /// Interval between automatic owner heartbeat writes.
    pub heartbeat_interval: Duration,
    /// Required continuous no-progress interval before owner retirement.
    pub liveness_timeout: Duration,
    /// Maximum total time spent waiting for another owner.
    pub wait_timeout: Duration,
}

impl Default for ExactCacheSingleFlightOptions {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_millis(50),
            heartbeat_interval: Duration::from_secs(5),
            liveness_timeout: Duration::from_secs(30),
            wait_timeout: Duration::from_secs(30 * 60),
        }
    }
}

impl ExactCacheSingleFlightOptions {
    fn validate(self) -> Result<Self, ExactCacheError> {
        if self.poll_interval.is_zero()
            || self.heartbeat_interval.is_zero()
            || self.liveness_timeout.is_zero()
            || self.wait_timeout.is_zero()
            || self.heartbeat_interval >= self.liveness_timeout
        {
            return Err(ExactCacheError::InvalidSingleFlightOptions);
        }
        Ok(self)
    }
}

/// Target-neutral next action for an exact-cache single-flight waiter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactCacheSingleFlightDecision {
    /// Observe the committed entry and owner revision again after this delay.
    Wait(Duration),
    /// Recheck the exact owner revision atomically and take over if unchanged.
    Takeover,
    /// Stop without downloading because the bounded total wait expired.
    Timeout,
}

/// Target-neutral liveness state shared by filesystem and browser substrates.
#[derive(Debug, Clone)]
pub struct ExactCacheSingleFlightWait {
    started: Duration,
    unchanged_since: Duration,
    observation_sha256: Option<[u8; 32]>,
}

impl ExactCacheSingleFlightWait {
    /// Begin observing an owner at one monotonic timestamp.
    #[must_use]
    pub fn new(now: Duration) -> Self {
        Self {
            started: now,
            unchanged_since: now,
            observation_sha256: None,
        }
    }

    /// Observe an opaque owner/heartbeat revision and select the next action.
    ///
    /// `now` and the returned delay use a caller-local monotonic clock. The
    /// revision must change whenever the owner token or heartbeat changes.
    pub fn observe(
        &mut self,
        now: Duration,
        revision: &[u8],
        options: ExactCacheSingleFlightOptions,
    ) -> Result<ExactCacheSingleFlightDecision, ExactCacheError> {
        let options = options.validate()?;
        let revision_sha256: [u8; 32] = Sha256::digest(revision).into();
        if self.observation_sha256 != Some(revision_sha256) {
            self.observation_sha256 = Some(revision_sha256);
            self.unchanged_since = now;
        }

        let no_progress = now.saturating_sub(self.unchanged_since);
        if no_progress >= options.liveness_timeout {
            return Ok(ExactCacheSingleFlightDecision::Takeover);
        }
        let elapsed = now.saturating_sub(self.started);
        if elapsed >= options.wait_timeout {
            return Ok(ExactCacheSingleFlightDecision::Timeout);
        }
        Ok(ExactCacheSingleFlightDecision::Wait(
            options
                .poll_interval
                .min(options.wait_timeout - elapsed)
                .min(options.liveness_timeout - no_progress),
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CommitRecord {
    schema_version: u8,
    entry: String,
    identity_sha256: String,
    distribution_source: String,
    product_sha256: String,
    product_byte_length: u64,
    archive_sha256: String,
    archive_byte_length: u64,
    provenance_sha256: String,
    provenance_byte_length: u64,
}

/// Return the SHA-256 binding for every field of an exact product identity.
pub fn identity_sha256(identity: &ProductIdentity) -> Result<String, ExactCacheError> {
    Ok(sha256_hex(&identity.canonical_bytes()?))
}

/// Build a canonical commit record for an immutable cache transaction.
///
/// `entry_id` must be a freshly allocated, immutable transaction directory.
/// Native callers normally use [`ExactProductCache::publish`], which allocates
/// it. WebAssembly hosts can generate 16 random bytes, encode them as lower-case
/// hexadecimal, store the three byte objects under that identifier, and then
/// atomically replace their commit marker with the returned bytes.
pub fn build_commit_record(
    identity: &ProductIdentity,
    source: DistributionSource,
    entry_id: &str,
    product: &[u8],
    archive: &[u8],
    provenance: &[u8],
) -> Result<Vec<u8>, ExactCacheError> {
    validate_entry_id(entry_id)?;
    let record = CommitRecord {
        schema_version: EXACT_CACHE_SCHEMA_VERSION,
        entry: entry_id.to_owned(),
        identity_sha256: identity_sha256(identity)?,
        distribution_source: source.code().to_owned(),
        product_sha256: sha256_hex(product),
        product_byte_length: byte_length(product)?,
        archive_sha256: sha256_hex(archive),
        archive_byte_length: byte_length(archive)?,
        provenance_sha256: sha256_hex(provenance),
        provenance_byte_length: byte_length(provenance)?,
    };
    serde_json::to_vec(&record).map_err(|_| ExactCacheError::InvalidCommit("serialization"))
}

/// Verify that one marker and immutable byte triple belong to the requested
/// full identity and explicit distribution source.
///
/// The returned provenance bytes are authenticated by the marker but remain
/// application data. The acquisition interface must parse them and confirm
/// requested/resolved identities and product semantics before accepting a hit.
pub fn verify_commit_record(
    identity: &ProductIdentity,
    source: DistributionSource,
    marker: &[u8],
    product: &[u8],
    archive: &[u8],
    provenance: &[u8],
) -> Result<VerifiedExactCacheCommit, ExactCacheError> {
    let record: CommitRecord = serde_json::from_slice(marker)
        .map_err(|_| ExactCacheError::InvalidCommit("malformed JSON"))?;
    if record.schema_version != EXACT_CACHE_SCHEMA_VERSION {
        return Err(ExactCacheError::InvalidCommit("schema version"));
    }
    validate_entry_id(&record.entry)?;
    let expected_identity = identity_sha256(identity)?;
    let expected = ExactCacheDigests {
        identity_sha256: expected_identity,
        distribution_source: source.code().to_owned(),
        product_sha256: sha256_hex(product),
        product_byte_length: byte_length(product)?,
        archive_sha256: sha256_hex(archive),
        archive_byte_length: byte_length(archive)?,
        provenance_sha256: sha256_hex(provenance),
        provenance_byte_length: byte_length(provenance)?,
    };
    let actual = ExactCacheDigests {
        identity_sha256: record.identity_sha256,
        distribution_source: record.distribution_source,
        product_sha256: record.product_sha256,
        product_byte_length: record.product_byte_length,
        archive_sha256: record.archive_sha256,
        archive_byte_length: record.archive_byte_length,
        provenance_sha256: record.provenance_sha256,
        provenance_byte_length: record.provenance_byte_length,
    };
    if actual != expected {
        return Err(ExactCacheError::InvalidCommit("identity, source, or bytes"));
    }
    Ok(VerifiedExactCacheCommit {
        entry_id: record.entry,
        digests: actual,
    })
}

fn validate_entry_id(entry_id: &str) -> Result<(), ExactCacheError> {
    if entry_id.len() == 32
        && entry_id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(ExactCacheError::InvalidEntryId)
    }
}

fn byte_length(bytes: &[u8]) -> Result<u64, ExactCacheError> {
    u64::try_from(bytes.len()).map_err(|_| ExactCacheError::InvalidCommit("byte length overflow"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use std::fs::{self, File, OpenOptions};
    use std::io::{ErrorKind, Write};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Condvar, Mutex, OnceLock};
    use std::thread::{self, JoinHandle};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    const LOCK_FILENAME: &str = ".sidereon-cache.lock";
    const INFLIGHT_FILENAME: &str = "in-flight.json";
    const INFLIGHT_HEARTBEAT_DIRECTORY: &str = "in-flight-heartbeats";
    const INFLIGHT_PROTOCOL_VERSION: u8 = 1;

    /// Paths and exact bytes from one verified immutable cache entry.
    #[derive(Debug, Clone)]
    pub struct CommittedExactCacheEntry {
        /// Immutable transaction identifier.
        pub entry_id: String,
        /// Validated product path.
        pub product_path: PathBuf,
        /// Distributor archive path.
        pub archive_path: PathBuf,
        /// Provenance path.
        pub provenance_path: PathBuf,
        /// Exact product bytes authenticated by the commit record.
        pub product: Vec<u8>,
        /// Exact archive bytes authenticated by the commit record.
        pub archive: Vec<u8>,
        /// Exact provenance bytes authenticated by the commit record.
        pub provenance: Vec<u8>,
    }

    /// Result of opening an exact cache with single-flight coordination.
    #[derive(Debug)]
    pub enum ExactCacheOpen {
        /// A complete committed entry was already available or was published
        /// by the owner observed while waiting.
        Hit(CommittedExactCacheEntry),
        /// This process owns acquisition and is the only caller that should fetch.
        Owner(ExactCacheOwner),
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    #[serde(deny_unknown_fields)]
    struct InflightRecord {
        protocol_version: u8,
        owner_token: String,
        process_id: u32,
        process_nonce: String,
        created_unix_ms: u64,
        identity_sha256: String,
        distribution_source: String,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct InflightSnapshot {
        marker: Vec<u8>,
        record: Option<InflightRecord>,
        heartbeat: Option<HeartbeatFingerprint>,
    }

    impl InflightSnapshot {
        fn revision(&self) -> [u8; 32] {
            let mut digest = Sha256::new();
            digest.update(self.marker.len().to_be_bytes());
            digest.update(&self.marker);
            match &self.heartbeat {
                Some(heartbeat) => {
                    digest.update([1]);
                    digest.update(heartbeat.byte_length.to_be_bytes());
                    match heartbeat.modified {
                        Some(modified) => match modified.duration_since(UNIX_EPOCH) {
                            Ok(duration) => {
                                digest.update([1]);
                                digest.update(duration.as_secs().to_be_bytes());
                                digest.update(duration.subsec_nanos().to_be_bytes());
                            }
                            Err(error) => {
                                digest.update([2]);
                                digest.update(error.duration().as_secs().to_be_bytes());
                                digest.update(error.duration().subsec_nanos().to_be_bytes());
                            }
                        },
                        None => digest.update([0]),
                    }
                }
                None => digest.update([0]),
            }
            digest.finalize().into()
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct HeartbeatFingerprint {
        byte_length: u64,
        modified: Option<SystemTime>,
    }

    #[derive(Debug)]
    struct HeartbeatControl {
        stopped: Mutex<bool>,
        wake: Condvar,
    }

    impl HeartbeatControl {
        fn wait(&self, interval: Duration) -> bool {
            let stopped = self.stopped.lock().expect("heartbeat mutex poisoned");
            let (stopped, _) = self
                .wake
                .wait_timeout_while(stopped, interval, |stopped| !*stopped)
                .expect("heartbeat condition variable poisoned");
            *stopped
        }

        fn stop(&self) {
            *self.stopped.lock().expect("heartbeat mutex poisoned") = true;
            self.wake.notify_all();
        }
    }

    /// Exclusive right to fetch and publish one single-flight cache miss.
    pub struct ExactCacheOwner {
        cache: ExactProductCache,
        token: String,
        options: ExactCacheSingleFlightOptions,
        heartbeat_control: Arc<HeartbeatControl>,
        heartbeat_failed: Arc<AtomicBool>,
        heartbeat_thread: Option<JoinHandle<()>>,
        released: bool,
    }

    impl std::fmt::Debug for ExactCacheOwner {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("ExactCacheOwner")
                .field("stable_path", &self.cache.stable_path)
                .field("token", &self.token)
                .field("options", &self.options)
                .field("heartbeat_failed", &self.heartbeat_failed)
                .field("released", &self.released)
                .finish_non_exhaustive()
        }
    }

    trait MonotonicClock {
        fn now(&self) -> Duration;
        fn sleep(&self, duration: Duration);
    }

    struct SystemMonotonicClock {
        origin: Instant,
    }

    impl SystemMonotonicClock {
        fn new() -> Self {
            Self {
                origin: Instant::now(),
            }
        }
    }

    impl MonotonicClock for SystemMonotonicClock {
        fn now(&self) -> Duration {
            self.origin.elapsed()
        }

        fn sleep(&self, duration: Duration) {
            thread::sleep(duration);
        }
    }

    #[cfg(feature = "exact-cache-test-failpoints")]
    #[derive(Debug, Clone)]
    #[doc(hidden)]
    pub struct ExactCacheTestClock {
        state: Arc<(Mutex<TestClockState>, Condvar)>,
    }

    #[cfg(feature = "exact-cache-test-failpoints")]
    #[derive(Debug)]
    struct TestClockState {
        now: Duration,
        sleepers: usize,
    }

    #[cfg(feature = "exact-cache-test-failpoints")]
    impl ExactCacheTestClock {
        /// Create a stopped monotonic clock for deterministic integration tests.
        #[must_use]
        pub fn new() -> Self {
            Self {
                state: Arc::new((
                    Mutex::new(TestClockState {
                        now: Duration::ZERO,
                        sleepers: 0,
                    }),
                    Condvar::new(),
                )),
            }
        }

        /// Advance the clock and wake every cache waiter.
        pub fn advance(&self, duration: Duration) {
            let (state, wake) = &*self.state;
            let mut state = state.lock().expect("test clock mutex poisoned");
            state.now = state.now.saturating_add(duration);
            wake.notify_all();
        }

        /// Block until at least `count` cache sleeps have begun.
        pub fn wait_for_sleepers(&self, count: usize) {
            let deadline = Instant::now() + Duration::from_secs(10);
            let (state, wake) = &*self.state;
            let mut state = state.lock().expect("test clock mutex poisoned");
            while state.sleepers < count {
                let remaining = deadline
                    .checked_duration_since(Instant::now())
                    .expect("timed out waiting for test-clock sleeper");
                let (next, timeout) = wake
                    .wait_timeout(state, remaining)
                    .expect("test clock condition variable poisoned");
                state = next;
                assert!(
                    !timeout.timed_out() || state.sleepers >= count,
                    "timed out waiting for test-clock sleeper"
                );
            }
        }
    }

    #[cfg(feature = "exact-cache-test-failpoints")]
    impl Default for ExactCacheTestClock {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "exact-cache-test-failpoints")]
    impl MonotonicClock for ExactCacheTestClock {
        fn now(&self) -> Duration {
            self.state.0.lock().expect("test clock mutex poisoned").now
        }

        fn sleep(&self, duration: Duration) {
            let (state, wake) = &*self.state;
            let mut state = state.lock().expect("test clock mutex poisoned");
            let deadline = state.now.saturating_add(duration);
            state.sleepers += 1;
            wake.notify_all();
            while state.now < deadline {
                state = wake
                    .wait(state)
                    .expect("test clock condition variable poisoned");
            }
        }
    }

    /// One exact identity/source cache rooted at the caller's stable product path.
    #[derive(Debug, Clone)]
    pub struct ExactProductCache {
        stable_path: PathBuf,
        identity: ProductIdentity,
        source: DistributionSource,
    }

    /// Held cross-process lock for one [`ExactProductCache`].
    pub struct ExactCacheGuard {
        lock_file: File,
        stable_path: PathBuf,
    }

    impl Drop for ExactCacheGuard {
        fn drop(&mut self) {
            let _ = self.lock_file.unlock();
        }
    }

    impl ExactCacheOwner {
        fn new(
            cache: ExactProductCache,
            token: String,
            options: ExactCacheSingleFlightOptions,
        ) -> Self {
            let heartbeat_control = Arc::new(HeartbeatControl {
                stopped: Mutex::new(false),
                wake: Condvar::new(),
            });
            let heartbeat_failed = Arc::new(AtomicBool::new(false));
            let thread_cache = cache.clone();
            let thread_token = token.clone();
            let thread_control = Arc::clone(&heartbeat_control);
            let thread_failed = Arc::clone(&heartbeat_failed);
            let heartbeat_thread = thread::Builder::new()
                .name("exact-cache-heartbeat".to_owned())
                .spawn(move || {
                    while !thread_control.wait(options.heartbeat_interval) {
                        if thread_cache
                            .refresh_inflight_heartbeat(&thread_token)
                            .is_err()
                        {
                            thread_failed.store(true, Ordering::Release);
                            break;
                        }
                    }
                })
                .ok();
            if heartbeat_thread.is_none() {
                heartbeat_failed.store(true, Ordering::Release);
            }
            Self {
                cache,
                token,
                options,
                heartbeat_control,
                heartbeat_failed,
                heartbeat_thread,
                released: false,
            }
        }

        /// Refresh this owner's liveness heartbeat immediately.
        pub fn heartbeat(&self) -> Result<(), ExactCacheError> {
            if self.heartbeat_failed.load(Ordering::Acquire) {
                return Err(ExactCacheError::SingleFlightOwnershipLost);
            }
            let result = self.cache.refresh_inflight_heartbeat(&self.token);
            if result.is_err() {
                self.heartbeat_failed.store(true, Ordering::Release);
            }
            result
        }

        /// Publish validated bytes and release single-flight ownership.
        pub fn publish(
            mut self,
            product: &[u8],
            archive: &[u8],
            provenance: &[u8],
        ) -> Result<CommittedExactCacheEntry, ExactCacheError> {
            self.stop_heartbeat();
            if self.heartbeat_failed.load(Ordering::Acquire) {
                return Err(ExactCacheError::SingleFlightOwnershipLost);
            }
            let guard = self.cache.lock(self.options.wait_timeout)?;
            if !self.cache.inflight_token_is_current(&self.token)? {
                return Err(ExactCacheError::SingleFlightOwnershipLost);
            }
            let entry = self.cache.publish(&guard, product, archive, provenance)?;
            self.cache.release_inflight(&guard, &self.token)?;
            self.released = true;
            Ok(entry)
        }

        fn stop_heartbeat(&mut self) {
            self.heartbeat_control.stop();
            if let Some(heartbeat_thread) = self.heartbeat_thread.take() {
                if heartbeat_thread.join().is_err() {
                    self.heartbeat_failed.store(true, Ordering::Release);
                }
            }
        }
    }

    impl Drop for ExactCacheOwner {
        fn drop(&mut self) {
            self.stop_heartbeat();
            if self.released {
                return;
            }
            if let Ok(guard) = self.cache.lock(Duration::ZERO) {
                let _ = self.cache.release_inflight(&guard, &self.token);
            }
        }
    }

    impl ExactProductCache {
        /// Create a cache handle after validating the complete identity.
        pub fn new(
            stable_path: impl Into<PathBuf>,
            identity: ProductIdentity,
            source: DistributionSource,
        ) -> Result<Self, ExactCacheError> {
            identity.validate()?;
            let stable_path = stable_path.into();
            if stable_path.file_name().is_none() || stable_path.parent().is_none() {
                return Err(ExactCacheError::InvalidCommit("stable product path"));
            }
            Ok(Self {
                stable_path,
                identity,
                source,
            })
        }

        /// Stable caller-facing path used to locate this cache entry.
        #[must_use]
        pub fn stable_path(&self) -> &Path {
            &self.stable_path
        }

        /// Open this cache with bounded single-flight miss coalescing.
        ///
        /// A hit contains bytes verified by the unchanged schema-v3 commit
        /// protocol. Only the returned owner should perform acquisition.
        pub fn open_single_flight(
            &self,
            options: ExactCacheSingleFlightOptions,
        ) -> Result<ExactCacheOpen, ExactCacheError> {
            let clock = SystemMonotonicClock::new();
            self.open_single_flight_with_clock(options, &clock)
        }

        /// Open using an injectable monotonic clock for deterministic tests.
        #[cfg(feature = "exact-cache-test-failpoints")]
        #[doc(hidden)]
        pub fn open_single_flight_with_test_clock(
            &self,
            options: ExactCacheSingleFlightOptions,
            clock: &ExactCacheTestClock,
        ) -> Result<ExactCacheOpen, ExactCacheError> {
            self.open_single_flight_with_clock(options, clock)
        }

        /// Acquire the per-entry cross-process lock with bounded waiting.
        pub fn lock(&self, timeout: Duration) -> Result<ExactCacheGuard, ExactCacheError> {
            ensure_supported_platform()?;
            let parent = self
                .stable_path
                .parent()
                .ok_or(ExactCacheError::InvalidCommit("stable product parent"))?;
            durable_create_dir_all(parent)?;
            let lock_path = parent.join(LOCK_FILENAME);
            let lock_file = OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(lock_path)
                .map_err(|source| io("open lock", source))?;
            lock_file
                .sync_all()
                .map_err(|source| io("sync lock", source))?;
            sync_directory(parent)?;
            let deadline = Instant::now()
                .checked_add(timeout)
                .ok_or(ExactCacheError::LockTimeout)?;
            loop {
                match lock_file.try_lock() {
                    Ok(()) => {
                        return Ok(ExactCacheGuard {
                            lock_file,
                            stable_path: self.stable_path.clone(),
                        });
                    }
                    Err(error) => {
                        let source: std::io::Error = error.into();
                        if source.kind() == ErrorKind::WouldBlock {
                            let now = Instant::now();
                            if now >= deadline {
                                return Err(ExactCacheError::LockTimeout);
                            }
                            thread::sleep((deadline - now).min(Duration::from_millis(10)));
                        } else {
                            return Err(io("lock", source));
                        }
                    }
                }
            }
        }

        /// Read and digest-verify the currently committed immutable entry.
        ///
        /// Returns `Ok(None)` only when no commit marker exists. A malformed,
        /// incomplete, or mismatched entry is an error, never a cache miss.
        pub fn read(&self) -> Result<Option<CommittedExactCacheEntry>, ExactCacheError> {
            let marker_path = self.marker_path();
            for _ in 0..16 {
                let marker = match fs::read(&marker_path) {
                    Ok(bytes) => bytes,
                    Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
                    Err(source) => return Err(io("read marker", source)),
                };
                test_read_barrier();
                match self.read_committed_entry(&marker) {
                    Ok(entry) => return Ok(Some(entry)),
                    Err(error) => match fs::read(&marker_path) {
                        Ok(current) if current != marker => continue,
                        Err(current_error) if current_error.kind() == ErrorKind::NotFound => {
                            continue;
                        }
                        _ => return Err(error),
                    },
                }
            }
            Err(ExactCacheError::InvalidCommit(
                "commit changed repeatedly during read",
            ))
        }

        fn read_committed_entry(
            &self,
            marker: &[u8],
        ) -> Result<CommittedExactCacheEntry, ExactCacheError> {
            let record: CommitRecord = serde_json::from_slice(marker)
                .map_err(|_| ExactCacheError::InvalidCommit("malformed JSON"))?;
            validate_entry_id(&record.entry)?;
            let paths = self.entry_paths(&record.entry)?;
            let product = fs::read(&paths.product).map_err(|source| io("read product", source))?;
            let archive = fs::read(&paths.archive).map_err(|source| io("read archive", source))?;
            let provenance =
                fs::read(&paths.provenance).map_err(|source| io("read provenance", source))?;
            let verified = verify_commit_record(
                &self.identity,
                self.source,
                marker,
                &product,
                &archive,
                &provenance,
            )?;
            Ok(CommittedExactCacheEntry {
                entry_id: verified.entry_id,
                product_path: paths.product,
                archive_path: paths.archive,
                provenance_path: paths.provenance,
                product,
                archive,
                provenance,
            })
        }

        /// Publish a complete validated candidate under the held entry lock.
        pub fn publish(
            &self,
            guard: &ExactCacheGuard,
            product: &[u8],
            archive: &[u8],
            provenance: &[u8],
        ) -> Result<CommittedExactCacheEntry, ExactCacheError> {
            self.require_guard(guard)?;
            ensure_supported_platform()?;
            let control = self.control_directory();
            let entries = control.join("entries");
            durable_create_dir_all(&entries)?;
            let entry_id = self.allocate_entry_id(&entries)?;
            let paths = self.entry_paths(&entry_id)?;
            let entry_directory = paths
                .product
                .parent()
                .ok_or(ExactCacheError::InvalidCommit("entry parent"))?;
            sync_directory(&entries)?;
            let marker_temp =
                control.join(format!(".{EXACT_CACHE_MARKER_FILENAME}.{entry_id}.tmp"));
            let marker = build_commit_record(
                &self.identity,
                self.source,
                &entry_id,
                product,
                archive,
                provenance,
            )?;
            let result: Result<(), ExactCacheError> = (|| {
                write_exclusive(&paths.product, product)?;
                test_failpoint("after_payload");
                write_exclusive(&paths.archive, archive)?;
                test_failpoint("after_archive");
                write_exclusive(&paths.provenance, provenance)?;
                test_failpoint("after_metadata");
                sync_directory(entry_directory)?;
                sync_directory(&entries)?;
                test_failpoint("after_entry_sync");
                write_exclusive(&marker_temp, &marker)?;
                test_failpoint("after_marker_write");
                fs::rename(&marker_temp, self.marker_path())
                    .map_err(|source| io("rename marker", source))?;
                test_failpoint("after_marker_rename");
                sync_directory(&control)?;
                test_failpoint("after_commit_sync");
                Ok(())
            })();
            if result.is_err() {
                let _ = fs::remove_file(&marker_temp);
                if self.current_entry_id().as_deref() != Some(entry_id.as_str()) {
                    let _ = fs::remove_dir_all(entry_directory);
                }
            }
            result?;
            Ok(CommittedExactCacheEntry {
                entry_id,
                product_path: paths.product,
                archive_path: paths.archive,
                provenance_path: paths.provenance,
                product: product.to_vec(),
                archive: archive.to_vec(),
                provenance: provenance.to_vec(),
            })
        }

        /// Remove unreferenced transactions while holding the entry lock.
        pub fn cleanup_abandoned(&self, guard: &ExactCacheGuard) -> Result<(), ExactCacheError> {
            self.require_guard(guard)?;
            let control = self.control_directory();
            let entries = control.join("entries");
            let current = match fs::read(self.marker_path()) {
                Ok(marker) => {
                    let record: CommitRecord = match serde_json::from_slice(&marker) {
                        Ok(record) => record,
                        Err(_) => return Ok(()),
                    };
                    if validate_entry_id(&record.entry).is_err() {
                        return Ok(());
                    }
                    Some(record.entry)
                }
                Err(error) if error.kind() == ErrorKind::NotFound => None,
                Err(_) => return Ok(()),
            };
            if let Ok(children) = fs::read_dir(&entries) {
                for child in children.flatten() {
                    let name = child.file_name();
                    if current.as_deref() != name.to_str() {
                        let _ = fs::remove_dir_all(child.path());
                    }
                }
            }
            if let Ok(children) = fs::read_dir(&control) {
                let prefix = format!(".{EXACT_CACHE_MARKER_FILENAME}.");
                for child in children.flatten() {
                    let name = child.file_name();
                    let Some(name) = name.to_str() else {
                        continue;
                    };
                    if name.starts_with(&prefix) && name.ends_with(".tmp") {
                        let _ = fs::remove_file(child.path());
                    }
                }
            }
            Ok(())
        }

        fn open_single_flight_with_clock<C: MonotonicClock>(
            &self,
            options: ExactCacheSingleFlightOptions,
            clock: &C,
        ) -> Result<ExactCacheOpen, ExactCacheError> {
            let options = options.validate()?;
            ensure_supported_platform()?;
            let started = clock.now();
            let mut wait_state = ExactCacheSingleFlightWait::new(started);

            loop {
                if let Some(entry) = self.read()? {
                    return Ok(ExactCacheOpen::Hit(entry));
                }

                let now = clock.now();
                let snapshot = self.read_inflight_snapshot()?;
                match snapshot {
                    None => {
                        if let Some(opened) = self.try_claim_transition(options)? {
                            return Ok(opened);
                        }
                        let elapsed = now.saturating_sub(started);
                        if elapsed >= options.wait_timeout {
                            return Err(ExactCacheError::SingleFlightTimeout);
                        }
                        clock.sleep(options.poll_interval.min(options.wait_timeout - elapsed));
                    }
                    Some(snapshot) => {
                        let decision = wait_state.observe(now, &snapshot.revision(), options)?;
                        test_failpoint("after_inflight_wait_observation");
                        match decision {
                            ExactCacheSingleFlightDecision::Wait(duration) => {
                                clock.sleep(duration);
                            }
                            ExactCacheSingleFlightDecision::Takeover => {
                                if let Some(opened) =
                                    self.try_takeover_transition(&snapshot, options)?
                                {
                                    return Ok(opened);
                                }
                                let elapsed = clock.now().saturating_sub(started);
                                if elapsed >= options.wait_timeout {
                                    return Err(ExactCacheError::SingleFlightTimeout);
                                }
                                clock.sleep(
                                    options.poll_interval.min(options.wait_timeout - elapsed),
                                );
                            }
                            ExactCacheSingleFlightDecision::Timeout => {
                                return Err(ExactCacheError::SingleFlightTimeout);
                            }
                        }
                    }
                }
            }
        }

        fn try_claim_transition(
            &self,
            options: ExactCacheSingleFlightOptions,
        ) -> Result<Option<ExactCacheOpen>, ExactCacheError> {
            let Some(guard) = self.try_transition_guard()? else {
                return Ok(None);
            };
            if let Some(entry) = self.read()? {
                return Ok(Some(ExactCacheOpen::Hit(entry)));
            }
            if self.read_inflight_snapshot()?.is_some() {
                return Ok(None);
            }
            let Some(token) = self.claim_inflight(&guard)? else {
                return Ok(None);
            };
            Ok(Some(ExactCacheOpen::Owner(ExactCacheOwner::new(
                self.clone(),
                token,
                options,
            ))))
        }

        fn try_takeover_transition(
            &self,
            observed: &InflightSnapshot,
            options: ExactCacheSingleFlightOptions,
        ) -> Result<Option<ExactCacheOpen>, ExactCacheError> {
            let Some(guard) = self.try_transition_guard()? else {
                return Ok(None);
            };
            if let Some(entry) = self.read()? {
                return Ok(Some(ExactCacheOpen::Hit(entry)));
            }
            if self.read_inflight_snapshot()?.as_ref() != Some(observed) {
                return Ok(None);
            }
            if !self.retire_inflight(&guard, observed)? {
                return Ok(None);
            }
            let Some(token) = self.claim_inflight(&guard)? else {
                return Ok(None);
            };
            Ok(Some(ExactCacheOpen::Owner(ExactCacheOwner::new(
                self.clone(),
                token,
                options,
            ))))
        }

        fn try_transition_guard(&self) -> Result<Option<ExactCacheGuard>, ExactCacheError> {
            match self.lock(Duration::ZERO) {
                Ok(guard) => Ok(Some(guard)),
                Err(ExactCacheError::LockTimeout) => Ok(None),
                Err(error) => Err(error),
            }
        }

        fn claim_inflight(
            &self,
            guard: &ExactCacheGuard,
        ) -> Result<Option<String>, ExactCacheError> {
            self.require_guard(guard)?;
            let control = self.control_directory();
            let heartbeats = control.join(INFLIGHT_HEARTBEAT_DIRECTORY);
            durable_create_dir_all(&heartbeats)?;
            if self.inflight_path().exists() {
                return Ok(None);
            }

            let token = random_identifier("random in-flight owner token")?;
            let heartbeat_path = self.inflight_heartbeat_path(&token)?;
            write_exclusive(&heartbeat_path, b"\0")?;
            sync_directory(&heartbeats)?;
            test_failpoint("after_inflight_heartbeat");

            let record = InflightRecord {
                protocol_version: INFLIGHT_PROTOCOL_VERSION,
                owner_token: token.clone(),
                process_id: std::process::id(),
                process_nonce: process_nonce()?,
                created_unix_ms: unix_milliseconds(),
                identity_sha256: identity_sha256(&self.identity)?,
                distribution_source: self.source.code().to_owned(),
            };
            let marker = serde_json::to_vec(&record)
                .map_err(|_| ExactCacheError::InvalidCommit("in-flight serialization"))?;
            if !write_exclusive_if_absent(&self.inflight_path(), &marker)? {
                let _ = fs::remove_file(heartbeat_path);
                return Ok(None);
            }
            test_failpoint("after_inflight_marker");
            sync_directory(&control)?;
            test_failpoint("after_inflight_sync");
            Ok(Some(token))
        }

        fn read_inflight_snapshot(&self) -> Result<Option<InflightSnapshot>, ExactCacheError> {
            let marker = match fs::read(self.inflight_path()) {
                Ok(marker) => marker,
                Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
                Err(source) => return Err(io("read in-flight marker", source)),
            };
            let record = serde_json::from_slice::<InflightRecord>(&marker)
                .ok()
                .filter(|record| {
                    record.protocol_version == INFLIGHT_PROTOCOL_VERSION
                        && validate_entry_id(&record.owner_token).is_ok()
                        && validate_entry_id(&record.process_nonce).is_ok()
                });
            if let Some(record) = &record {
                if record.identity_sha256 != identity_sha256(&self.identity)?
                    || record.distribution_source != self.source.code()
                {
                    return Err(ExactCacheError::InvalidCommit(
                        "in-flight identity or source",
                    ));
                }
            }
            let heartbeat = match &record {
                Some(record) => {
                    match fs::metadata(self.inflight_heartbeat_path(&record.owner_token)?) {
                        Ok(metadata) => Some(HeartbeatFingerprint {
                            byte_length: metadata.len(),
                            modified: metadata.modified().ok(),
                        }),
                        Err(error) if error.kind() == ErrorKind::NotFound => None,
                        Err(source) => return Err(io("read in-flight heartbeat", source)),
                    }
                }
                None => None,
            };
            Ok(Some(InflightSnapshot {
                marker,
                record,
                heartbeat,
            }))
        }

        fn refresh_inflight_heartbeat(&self, token: &str) -> Result<(), ExactCacheError> {
            if !self.inflight_token_is_current(token)? {
                return Err(ExactCacheError::SingleFlightOwnershipLost);
            }
            let heartbeat_path = self.inflight_heartbeat_path(token)?;
            let mut heartbeat = OpenOptions::new()
                .append(true)
                .open(heartbeat_path)
                .map_err(|source| io("open in-flight heartbeat", source))?;
            heartbeat
                .write_all(b"\0")
                .map_err(|source| io("write in-flight heartbeat", source))?;
            heartbeat
                .sync_all()
                .map_err(|source| io("sync in-flight heartbeat", source))?;
            test_failpoint("after_inflight_heartbeat_refresh");
            Ok(())
        }

        fn inflight_token_is_current(&self, token: &str) -> Result<bool, ExactCacheError> {
            Ok(self
                .read_inflight_snapshot()?
                .and_then(|snapshot| snapshot.record)
                .is_some_and(|record| record.owner_token == token))
        }

        fn release_inflight(
            &self,
            guard: &ExactCacheGuard,
            token: &str,
        ) -> Result<(), ExactCacheError> {
            self.require_guard(guard)?;
            let Some(snapshot) = self.read_inflight_snapshot()? else {
                return Ok(());
            };
            if snapshot
                .record
                .as_ref()
                .map(|record| record.owner_token.as_str())
                != Some(token)
            {
                return Ok(());
            }
            let _ = self.retire_inflight(guard, &snapshot)?;
            Ok(())
        }

        fn retire_inflight(
            &self,
            guard: &ExactCacheGuard,
            expected: &InflightSnapshot,
        ) -> Result<bool, ExactCacheError> {
            self.require_guard(guard)?;
            if self.read_inflight_snapshot()?.as_ref() != Some(expected) {
                return Ok(false);
            }
            let nonce = random_identifier("random retired marker id")?;
            let token = expected
                .record
                .as_ref()
                .map_or("malformed", |record| record.owner_token.as_str());
            let retired = self
                .control_directory()
                .join(format!(".in-flight.{token}.{nonce}.retired"));
            match fs::rename(self.inflight_path(), &retired) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::NotFound => return Ok(false),
                Err(source) => return Err(io("retire in-flight marker", source)),
            }
            test_failpoint("after_inflight_retire");

            let retired_marker =
                fs::read(&retired).map_err(|source| io("read retired in-flight marker", source))?;
            if retired_marker != expected.marker {
                let _ = write_exclusive_if_absent(&self.inflight_path(), &retired_marker)?;
                let _ = fs::remove_file(&retired);
                sync_directory(&self.control_directory())?;
                return Ok(false);
            }

            sync_directory(&self.control_directory())?;
            test_failpoint("after_inflight_retire_sync");
            fs::remove_file(&retired)
                .map_err(|source| io("remove retired in-flight marker", source))?;
            if let Some(record) = &expected.record {
                match fs::remove_file(self.inflight_heartbeat_path(&record.owner_token)?) {
                    Ok(()) => {}
                    Err(error) if error.kind() == ErrorKind::NotFound => {}
                    Err(source) => return Err(io("remove in-flight heartbeat", source)),
                }
            }
            test_failpoint("after_inflight_reap");
            sync_directory(&self.control_directory())?;
            let heartbeats = self.control_directory().join(INFLIGHT_HEARTBEAT_DIRECTORY);
            if heartbeats.is_dir() {
                sync_directory(&heartbeats)?;
            }
            test_failpoint("after_inflight_reap_sync");
            Ok(true)
        }

        fn require_guard(&self, guard: &ExactCacheGuard) -> Result<(), ExactCacheError> {
            if guard.stable_path == self.stable_path {
                Ok(())
            } else {
                Err(ExactCacheError::InvalidCommit("cache lock scope"))
            }
        }

        fn control_directory(&self) -> PathBuf {
            self.stable_path
                .parent()
                .expect("validated cache path has parent")
                .join(EXACT_CACHE_CONTROL_DIRECTORY)
        }

        fn marker_path(&self) -> PathBuf {
            self.control_directory().join(EXACT_CACHE_MARKER_FILENAME)
        }

        fn inflight_path(&self) -> PathBuf {
            self.control_directory().join(INFLIGHT_FILENAME)
        }

        fn inflight_heartbeat_path(&self, token: &str) -> Result<PathBuf, ExactCacheError> {
            validate_entry_id(token)?;
            Ok(self
                .control_directory()
                .join(INFLIGHT_HEARTBEAT_DIRECTORY)
                .join(format!("{token}.heartbeat")))
        }

        fn entry_paths(&self, entry_id: &str) -> Result<EntryPaths, ExactCacheError> {
            validate_entry_id(entry_id)?;
            let filename = self
                .stable_path
                .file_name()
                .ok_or(ExactCacheError::InvalidCommit("stable product filename"))?;
            let entry = self.control_directory().join("entries").join(entry_id);
            let product = entry.join(filename);
            let archive = entry.join(format!("{}.archive", filename.to_string_lossy()));
            let provenance = entry.join(format!("{}.provenance.json", filename.to_string_lossy()));
            Ok(EntryPaths {
                product,
                archive,
                provenance,
            })
        }

        fn allocate_entry_id(&self, entries: &Path) -> Result<String, ExactCacheError> {
            for _ in 0..128 {
                let entry_id = random_identifier("random entry id")?;
                match fs::create_dir(entries.join(&entry_id)) {
                    Ok(()) => return Ok(entry_id),
                    Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                    Err(source) => return Err(io("create entry", source)),
                }
            }
            Err(ExactCacheError::InvalidCommit("entry identifier collision"))
        }

        fn current_entry_id(&self) -> Option<String> {
            let marker = fs::read(self.marker_path()).ok()?;
            let record: CommitRecord = serde_json::from_slice(&marker).ok()?;
            validate_entry_id(&record.entry).ok()?;
            Some(record.entry)
        }
    }

    struct EntryPaths {
        product: PathBuf,
        archive: PathBuf,
        provenance: PathBuf,
    }

    fn io(operation: &'static str, source: std::io::Error) -> ExactCacheError {
        ExactCacheError::Io { operation, source }
    }

    fn ensure_supported_platform() -> Result<(), ExactCacheError> {
        if cfg!(any(target_os = "linux", target_os = "macos")) {
            Ok(())
        } else {
            Err(ExactCacheError::UnsupportedPlatform)
        }
    }

    fn durable_create_dir_all(path: &Path) -> Result<(), ExactCacheError> {
        if path.is_dir() {
            return Ok(());
        }
        let mut missing = Vec::new();
        let mut cursor = path;
        while !cursor.exists() {
            missing.push(cursor.to_path_buf());
            cursor = cursor
                .parent()
                .ok_or(ExactCacheError::InvalidCommit("cache directory parent"))?;
        }
        for directory in missing.iter().rev() {
            match fs::create_dir(directory) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::AlreadyExists && directory.is_dir() => {}
                Err(source) => return Err(io("create directory", source)),
            }
            let parent = directory
                .parent()
                .ok_or(ExactCacheError::InvalidCommit("cache directory parent"))?;
            sync_directory(parent)?;
        }
        Ok(())
    }

    fn write_exclusive(path: &Path, bytes: &[u8]) -> Result<(), ExactCacheError> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .map_err(|source| io("create immutable file", source))?;
        file.write_all(bytes)
            .map_err(|source| io("write immutable file", source))?;
        file.sync_all()
            .map_err(|source| io("sync immutable file", source))
    }

    fn write_exclusive_if_absent(path: &Path, bytes: &[u8]) -> Result<bool, ExactCacheError> {
        let mut file = match OpenOptions::new().create_new(true).write(true).open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::AlreadyExists => return Ok(false),
            Err(source) => return Err(io("create in-flight marker", source)),
        };
        let result = file
            .write_all(bytes)
            .map_err(|source| io("write in-flight marker", source))
            .and_then(|()| {
                file.sync_all()
                    .map_err(|source| io("sync in-flight marker", source))
            });
        if result.is_err() {
            let _ = fs::remove_file(path);
        }
        result.map(|()| true)
    }

    fn random_identifier(operation: &'static str) -> Result<String, ExactCacheError> {
        let mut random = [0_u8; 16];
        getrandom::getrandom(&mut random)
            .map_err(|error| io(operation, std::io::Error::other(error.to_string())))?;
        let mut identifier = String::with_capacity(32);
        for byte in random {
            write!(&mut identifier, "{byte:02x}").expect("writing to String cannot fail");
        }
        Ok(identifier)
    }

    fn process_nonce() -> Result<String, ExactCacheError> {
        static PROCESS_NONCE: OnceLock<String> = OnceLock::new();
        if let Some(nonce) = PROCESS_NONCE.get() {
            return Ok(nonce.clone());
        }
        let nonce = random_identifier("random in-flight process nonce")?;
        let _ = PROCESS_NONCE.set(nonce);
        Ok(PROCESS_NONCE
            .get()
            .expect("process nonce initialized")
            .clone())
    }

    fn unix_milliseconds() -> u64 {
        let milliseconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        u64::try_from(milliseconds).unwrap_or(u64::MAX)
    }

    fn sync_directory(path: &Path) -> Result<(), ExactCacheError> {
        File::open(path)
            .and_then(|directory| directory.sync_all())
            .map_err(|source| io("sync directory", source))
    }

    #[cfg(feature = "exact-cache-test-failpoints")]
    fn test_failpoint(name: &str) {
        if std::env::var_os("SIDEREON_TEST_EXACT_CACHE_PAUSE_FAILPOINT").as_deref()
            == Some(std::ffi::OsStr::new(name))
        {
            let barrier = PathBuf::from(
                std::env::var_os("SIDEREON_TEST_EXACT_CACHE_FAILPOINT_BARRIER")
                    .expect("exact-cache pause failpoint barrier"),
            );
            fs::write(barrier.with_extension("ready"), b"ready")
                .expect("write exact-cache failpoint barrier");
            let release = barrier.with_extension("release");
            let deadline = Instant::now() + Duration::from_secs(30);
            while !release.exists() {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for exact-cache failpoint release"
                );
                thread::sleep(Duration::from_millis(5));
            }
        }
        if std::env::var_os("SIDEREON_TEST_EXACT_CACHE_FAILPOINT").as_deref()
            == Some(std::ffi::OsStr::new(name))
        {
            std::process::exit(86);
        }
    }

    #[cfg(not(feature = "exact-cache-test-failpoints"))]
    fn test_failpoint(_name: &str) {}

    #[cfg(feature = "exact-cache-test-failpoints")]
    fn test_read_barrier() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let Some(barrier) = std::env::var_os("SIDEREON_TEST_EXACT_CACHE_READ_BARRIER") else {
                return;
            };
            let barrier = PathBuf::from(barrier);
            fs::write(barrier.with_extension("ready"), b"ready")
                .expect("write exact-cache read barrier");
            let release = barrier.with_extension("release");
            let deadline = Instant::now() + Duration::from_secs(10);
            while !release.exists() {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for exact-cache read barrier"
                );
                thread::sleep(Duration::from_millis(5));
            }
        });
    }

    #[cfg(not(feature = "exact-cache-test-failpoints"))]
    fn test_read_barrier() {}
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::{
    CommittedExactCacheEntry, ExactCacheGuard, ExactCacheOpen, ExactCacheOwner, ExactProductCache,
};

#[cfg(all(not(target_arch = "wasm32"), feature = "exact-cache-test-failpoints"))]
#[doc(hidden)]
pub use native::ExactCacheTestClock;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{product, AnalysisCenter, ProductDate, ProductType};

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

    #[test]
    fn commit_binds_every_byte_group_identity_and_source() {
        let identity = identity();
        let marker = build_commit_record(
            &identity,
            DistributionSource::Direct,
            "0123456789abcdef0123456789abcdef",
            b"product",
            b"archive",
            b"provenance",
        )
        .expect("commit");
        let verified = verify_commit_record(
            &identity,
            DistributionSource::Direct,
            &marker,
            b"product",
            b"archive",
            b"provenance",
        )
        .expect("verify");
        assert_eq!(verified.entry_id, "0123456789abcdef0123456789abcdef");

        for (product, archive, provenance) in [
            (&b"changed"[..], &b"archive"[..], &b"provenance"[..]),
            (&b"product"[..], &b"changed"[..], &b"provenance"[..]),
            (&b"product"[..], &b"archive"[..], &b"changed"[..]),
        ] {
            assert!(verify_commit_record(
                &identity,
                DistributionSource::Direct,
                &marker,
                product,
                archive,
                provenance,
            )
            .is_err());
        }
        assert!(verify_commit_record(
            &identity,
            DistributionSource::NasaCddis,
            &marker,
            b"product",
            b"archive",
            b"provenance",
        )
        .is_err());
    }

    #[test]
    fn malformed_entry_ids_are_rejected() {
        for entry in [
            "",
            "ABCDEF0123456789ABCDEF0123456789",
            "0123456789abcdef0123456789abcdeg",
            "0123456789abcdef",
        ] {
            assert!(build_commit_record(
                &identity(),
                DistributionSource::Direct,
                entry,
                b"product",
                b"archive",
                b"provenance",
            )
            .is_err());
        }
    }

    #[test]
    fn single_flight_wait_state_resets_on_progress_and_prefers_stale_takeover() {
        let options = ExactCacheSingleFlightOptions {
            poll_interval: Duration::from_secs(2),
            heartbeat_interval: Duration::from_secs(1),
            liveness_timeout: Duration::from_secs(5),
            wait_timeout: Duration::from_secs(20),
        };
        let mut wait = ExactCacheSingleFlightWait::new(Duration::ZERO);
        assert_eq!(
            wait.observe(Duration::ZERO, b"owner-a:1", options)
                .expect("first observation"),
            ExactCacheSingleFlightDecision::Wait(Duration::from_secs(2))
        );
        assert_eq!(
            wait.observe(Duration::from_secs(4), b"owner-a:1", options)
                .expect("unchanged observation"),
            ExactCacheSingleFlightDecision::Wait(Duration::from_secs(1))
        );
        assert_eq!(
            wait.observe(Duration::from_secs(4), b"owner-a:2", options)
                .expect("heartbeat progress"),
            ExactCacheSingleFlightDecision::Wait(Duration::from_secs(2))
        );
        assert_eq!(
            wait.observe(Duration::from_secs(9), b"owner-a:2", options)
                .expect("stale observation"),
            ExactCacheSingleFlightDecision::Takeover
        );

        let options = ExactCacheSingleFlightOptions {
            wait_timeout: Duration::from_secs(5),
            ..options
        };
        let mut equal_deadlines = ExactCacheSingleFlightWait::new(Duration::ZERO);
        equal_deadlines
            .observe(Duration::ZERO, b"owner", options)
            .expect("initial observation");
        assert_eq!(
            equal_deadlines
                .observe(Duration::from_secs(5), b"owner", options)
                .expect("equal deadlines"),
            ExactCacheSingleFlightDecision::Takeover
        );
    }

    #[test]
    fn single_flight_wait_state_times_out_while_owner_is_live() {
        let options = ExactCacheSingleFlightOptions {
            poll_interval: Duration::from_secs(1),
            heartbeat_interval: Duration::from_secs(2),
            liveness_timeout: Duration::from_secs(10),
            wait_timeout: Duration::from_secs(3),
        };
        let mut wait = ExactCacheSingleFlightWait::new(Duration::ZERO);
        wait.observe(Duration::ZERO, b"owner:1", options)
            .expect("initial observation");
        assert_eq!(
            wait.observe(Duration::from_secs(3), b"owner:2", options)
                .expect("live owner at wait bound"),
            ExactCacheSingleFlightDecision::Timeout
        );
    }
}
