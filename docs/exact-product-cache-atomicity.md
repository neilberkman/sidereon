# Exact-product cache atomicity audit

## Verdict for 0.29.2

Sidereon 0.29.2 did not provide a cross-process or crash-consistent exact-cache
commit. The Rust core derived distributor-independent identity and
source-specific paths but performed no cache IO. Python and Elixir separately
wrote product, archive, and provenance files, and their coordination did not
cover independent OS processes.

The 0.29.2 acquisition readers rechecked requested and resolved provenance,
distribution source, byte lengths and SHA-256 digests, and parsed product
semantics. Those checks rejected ordinary corruption and most mixed triples.
They did not make the three files one atomic acquisition record, prevent two
processes from interleaving publication, preserve the previous entry across
every process-death boundary, or prevent a duplicate cross-process download.
The required guarantee was therefore absent.

The audit also found that the Rust `ProductIdentity` omitted the catalog
analysis-center selector and parsed format version even though the acquisition
interfaces retained them. Those fields can distinguish exact products, so the
shared identity now includes both.

## 0.30.0 correction

One Rust implementation now defines the native protocol used by the Rust,
Python, C, and Elixir interfaces. The WASM/JavaScript interface uses the same
identity and commit-record builder/verifier, with browser-native coordination
and storage:

1. The complete `ProductIdentity` includes family, analysis center, publisher,
   solution class, campaign, filename version, coverage date, issue, span,
   cadence, official filename, format, parsed format version, and prediction
   horizon. Its canonical bytes and the explicit distribution source select
   and bind a cache entry. No field is inferred from filename or modification
   time.
2. Native writers hold a bounded advisory `flock` on the entry directory across
   cache check, acquisition, validation, commit, and cleanup. The kernel
   releases the lock when a process exits; recovery never guesses an owner PID
   or deletes a live writer's files.
3. A native writer creates a cryptographically random immutable entry
   directory, writes product, archive, and provenance bytes with exclusive file
   creation, synchronizes each file, and synchronizes the entry directories.
4. Schema-v3 commit bytes bind the SHA-256 digest and length of all three byte
   objects, the SHA-256 digest of the complete canonical identity, the explicit
   distribution source, and the immutable entry identifier.
5. A synchronized temporary marker is renamed over `current.json` in the same
   directory, then the directory is synchronized. That marker is the only
   reader-visible transition. A process death leaves the previous complete
   entry, the new complete entry, or no acceptable entry; it cannot authorize a
   mixed entry.
6. Readers follow only the marker and rehash all three objects. An unlocked
   native reader retries if the marker changes while a later lock owner removes
   the previous immutable directory, so refresh yields an old or new complete
   entry rather than a spurious partial read.
7. Cleanup runs only under the writer lock and removes only immutable entry
   directories not named by the current marker. Valid legacy Python and Elixir
   triples are fully revalidated before transaction migration and do not
   require another download.
8. Browser JavaScript uses a Web Lock for same-origin tab/worker coordination.
   One strict-durability IndexedDB read-write transaction stores the immutable
   entry and replaces its marker atomically. Reads invoke the shared WASM
   verifier before returning any bytes.

Transport and format parsing remain outside the low-level cache module. A
caller must validate product semantics before publication and must parse the
authenticated product and provenance bytes on a hit. Python and Elixir exact
acquisition perform those checks themselves.

## Interface parity

- Rust exposes the pure schema-v3 builder/verifier and the native
  `ExactProductCache` through both `sidereon-core` and `sidereon`.
- Python exposes `sidereon.exact_cache` and uses that same native implementation
  in exact acquisition.
- C exposes lock, locked and unlocked read, publish, cleanup, authenticated-byte
  and path accessors, and handle release functions in `sidereon.h`.
- WASM exposes the pure builder/verifier and `BrowserExactProductCache` from the
  `@neilberkman/sidereon/exact-cache` package export.
- Elixir exposes `Sidereon.GNSS.ExactCache` and uses that same native
  implementation in exact acquisition.

Host languages retain their normal ownership and async conventions, but the
identity fields, accepted sources, commit bindings, corruption behavior,
bounded coordination, immutable publication, and cleanup rules are shared.

## Test coverage

Core tests use child processes, explicit barriers, and named publication
failpoints compiled only by the non-default `exact-cache-test-failpoints`
feature. They cover two processes racing the same identity, one-download
warm-cache reuse, a live owner that cannot be displaced or cleaned, an unlocked
reader racing refresh cleanup, death after every payload/archive/provenance,
entry-sync, marker-write, marker-rename, and final-sync boundary, byte
corruption, source substitution, and same-path identity substitution.

Python and Elixir acquisition tests cover distributor races, legacy migration,
parsed-product validation, provenance mismatch, lock timeout, and duplicate
request reuse through the shared native implementation. C tests exercise lock
ownership, timeout, publication, locked and unlocked verified reads, and byte
copying. WASM tests exercise full identity fields, commit substitution
rejection, Web Lock serialization, one-acquisition reuse, IndexedDB atomic
publication, and stored-byte corruption rejection.

## Compatibility and residual risk

- Existing exact-acquisition calls remain compatible. Python adds optional
  `cache_lock_timeout_s`; Elixir adds optional `:cache_lock_timeout_ms`.
- `result.path` still names the official product, now inside its immutable
  entry directory. Consumers should not depend on the undocumented former
  parent directory.
- The native guarantee covers local Linux and macOS filesystems that implement
  advisory `flock`, atomic same-directory rename, file synchronization, and
  directory synchronization. Network filesystems or mounts that weaken those
  primitives are outside the guarantee.
- Browser coordination covers contexts participating in the same origin's Web
  Locks and IndexedDB databases. It is not an OS filesystem lock and does not
  coordinate unrelated origins or native processes.
- The cache directory is caller-controlled trusted storage. Digest verification
  detects changed bound bytes but is not a signature and does not defend
  against an attacker able to replace both data and commit records.
