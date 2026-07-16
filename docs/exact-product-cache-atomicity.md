# Exact-product cache atomicity audit

## Verdict for 0.29.2

Sidereon 0.29.2 did not provide a cross-process or crash-consistent exact-cache
commit. The Rust core derived the full distributor-independent identity and
source-specific cache path but intentionally performed no cache IO. The Python
and Elixir acquisition interfaces each wrote the archive, provenance, and
decompressed product as three independently renamed files. Their locks covered
only one Python process or one BEAM coordination domain.

The 0.29.2 readers did recheck the full requested and resolved identities,
distribution source, both SHA-256 digests and lengths, caller checksum, and
parsed product semantics. Those checks rejected ordinary corruption and most
mixed triples. They did not establish one atomic acquisition record, prevent
independent OS processes from interleaving publication, preserve the previous
entry across every crash boundary, or prevent duplicate cross-process
downloads. Therefore the required guarantee was absent.

## Unreleased correction

The acquisition-capable interfaces now share one on-disk protocol:

1. The source and complete `ProductIdentity` continue to select the existing
   collision-resistant identity directory. No identity field is inferred from
   a filename or modification time.
2. A POSIX advisory lock file in that directory covers cache check, acquisition,
   validation, and commit. Python uses `flock` directly; the Elixir NIF uses the
   same `flock` operation on Unix. Waiting is bounded. Process exit closes the
   descriptor and releases the lock without a stale-owner deletion decision.
3. A writer creates a cryptographically unique entry directory and writes the
   product, archive, and provenance there. Each file is synchronized before the
   entry and entries directories are synchronized.
4. The writer creates and synchronizes a small version-2 commit record. It
   contains the immutable entry identifier and the SHA-256 digest of the exact
   provenance bytes. A same-directory atomic rename replaces the prior commit
   record, followed by synchronization of its parent directory.
5. A reader follows only the commit record. The record binds provenance bytes;
   provenance binds product/archive digests, lengths, full requested and
   resolved identities, and source. The reader repeats those checks and fresh
   SP3/IONEX semantic validation before returning a hit.
6. A later lock owner removes unreferenced transaction directories. It cannot
   run cleanup while a cooperating writer is alive. Valid 0.29.0-0.29.2 triples
   are fully revalidated and republished as one transaction without a new
   download.

The commit record is the only reader-visible transition. A failure before its
rename leaves the prior record. A failure after its rename points to an entry
whose files and directory names were already synchronized. With no prior
record, the corresponding outcomes are a complete entry or no acceptable
entry.

## Test coverage

Deterministic tests use explicit barriers and named publication failpoints. The
Python and Elixir suites cover independent OS processes racing one source,
identical bytes acquired through different allowed sources, duplicate-request
reuse, a prior entry during refresh, process death after every payload,
archive, provenance, entry-sync, marker-write, marker-rename, and final-sync
boundary, corruption, same-filename/different-identity isolation, and abandoned
cleanup while another OS process owns the lock. Python additionally constructs
a deliberate payload/provenance mismatch and verifies legacy migration.

Manual bidirectional compatibility checks also prove that Python accepts an
Elixir-published entry and Elixir accepts a Python-published entry. Holding the
lock in either runtime makes the other runtime time out, confirming that both
use the same OS lock rather than two unrelated coordination mechanisms.

## Compatibility and residual risk

- The exact-acquisition function signatures remain compatible. Python adds
  optional `cache_lock_timeout_s`; Elixir adds optional
  `:cache_lock_timeout_ms`.
- `result.path` still names the official product filename, now inside the
  immutable transaction directory. Code that assumed the undocumented former
  parent directory should instead use the returned path.
- The legacy convenience fetch APIs and their separate cache layouts are not
  changed.
- The crash and cross-process guarantee applies to local Linux and macOS
  filesystems that honor POSIX `flock`, atomic same-directory rename, regular
  file synchronization, and directory synchronization. Network filesystems or
  mounts that weaken those primitives are outside the guarantee.
- The cache directory is assumed to be controlled by the caller. Protection
  from a malicious local user who can replace paths or rewrite committed files
  is not part of this contract; tampering is detected when it changes bound
  bytes, but this is not an authenticated storage format.
