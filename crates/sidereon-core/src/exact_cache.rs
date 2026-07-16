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
    use fs2::FileExt;
    use std::fs::{self, File, OpenOptions};
    use std::io::{ErrorKind, Write};
    use std::path::{Path, PathBuf};
    use std::thread;
    use std::time::{Duration, Instant};

    const LOCK_FILENAME: &str = ".sidereon-cache.lock";

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
            let _ = FileExt::unlock(&self.lock_file);
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
                match lock_file.try_lock_exclusive() {
                    Ok(()) => {
                        return Ok(ExactCacheGuard {
                            lock_file,
                            stable_path: self.stable_path.clone(),
                        });
                    }
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        let now = Instant::now();
                        if now >= deadline {
                            return Err(ExactCacheError::LockTimeout);
                        }
                        thread::sleep((deadline - now).min(Duration::from_millis(10)));
                    }
                    Err(source) => return Err(io("lock", source)),
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
                let mut random = [0_u8; 16];
                getrandom::getrandom(&mut random).map_err(|error| {
                    io("random entry id", std::io::Error::other(error.to_string()))
                })?;
                let mut entry_id = String::with_capacity(32);
                for byte in random {
                    write!(&mut entry_id, "{byte:02x}").expect("writing to String cannot fail");
                }
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

    fn sync_directory(path: &Path) -> Result<(), ExactCacheError> {
        File::open(path)
            .and_then(|directory| directory.sync_all())
            .map_err(|source| io("sync directory", source))
    }

    #[cfg(feature = "exact-cache-test-failpoints")]
    fn test_failpoint(name: &str) {
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
pub use native::{CommittedExactCacheEntry, ExactCacheGuard, ExactProductCache};

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
}
