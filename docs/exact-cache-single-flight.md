# Exact-cache single-flight coalescing

## Decision

Exact-cache single-flight is an opt-in open mode on `ExactProductCache`. A
single call returns either a verified cache hit or an owner object. Only the
owner may fetch, validate, and publish. Other callers remain inside the core
state machine until the owner publishes, abandons the attempt, loses its
lease, or their bounded wait expires.

The committed data format remains schema v3. Coordination uses sidecars under
the existing `.sidereon-cache-v3` control directory and never changes the
meaning or encoding of `current.json`. This is deliberately not schema v4:
coordination state is temporary, is not evidence that bytes are committed,
and may be ignored or removed without changing what a reader accepts.

Transport and semantic validation remain outside the cache. The core owns all
coordination transitions; a caller only branches on `Hit` versus `Owner` and
uses the owner object to publish validated bytes.

## On-disk representation

For one exact identity/source cache directory, the additions are:

```text
.sidereon-cache-v3/
  current.json                         # unchanged schema-v3 commit record
  entries/                             # unchanged immutable entries
  in-flight.json                       # current owner record, if any
  in-flight-heartbeats/
    <owner-token>.heartbeat            # append-only liveness pulse
  .in-flight.<token>.<nonce>.retired   # transient retired owner record
```

`in-flight.json` is JSON with a coordination protocol version, the canonical
identity digest, distribution source, 128-bit random owner token, process ID,
128-bit process nonce, and creation time in Unix milliseconds. The identity
digest and source prevent an accidentally shared stable path from coalescing
different requests.

The owner token is the authority. PID is diagnostic only: PIDs are reused and
can collide across containers or hosts sharing a cache. A portable boot ID is
not available on Linux, macOS, and Windows with consistent privacy and
permission properties. A cryptographically random process nonce therefore
provides the boot/session disambiguation that a boot-derived value was meant
to provide, without depending on a host-global identifier. Creation time is
also diagnostic and is never used to decide liveness. The token plus process
nonce makes accidental identity collision independent of PID, host, container,
and clock values.

The heartbeat is initialized before `in-flight.json` becomes visible. Each
refresh appends one byte and synchronizes the file. Waiters observe its length
and modification time. Length makes progress visible even on filesystems with
coarse mtime resolution; mtime remains useful on implementations that delay
length metadata. Heartbeat contents carry no authority and need no parser.

Exclusive creation elects an owner when `in-flight.json` is absent. A partial
or malformed record left by process death is not a cache error and cannot
authorize data; if it remains unchanged for a full liveness window it is
retired like a dead owner's record. A well-formed record for a different
identity or source is an error rather than a request to steal unrelated work.

## Liveness mechanism

The primary mechanism is a progress heartbeat, not PID probing, an absolute
expiry timestamp, or an advisory lock. It uses regular file create, append,
metadata, rename, and directory synchronization operations available on local
and network filesystems on Linux, macOS, and Windows.

A waiter records the complete owner-record bytes and the heartbeat metadata,
then starts an injectable local monotonic timer. Any observed change resets
that timer. An owner is eligible for retirement only after one complete
`liveness_timeout` with no observed progress. A newly arrived waiter therefore
waits its own full liveness window even if the file's wall-clock mtime looks
old. This deliberately trades slower dead-owner recovery for immunity to host
clock skew, timestamp truncation, clock steps, copied directories, and a
future-dated mtime.

The owner refreshes in a background thread at `heartbeat_interval`; callers
may also request an immediate heartbeat. The default interval is one fifth of
the liveness window. Publication stops the heartbeat thread and checks that
the same token is still current before committing.

The existing advisory file lock is retained around short coordination
transitions and publication. On reliable local implementations it cheaply
serializes the final recheck/retire/create sequence and fences publication. It
is not the liveness oracle, is not held during a download, and a waiter's
liveness decision does not depend on the kernel reporting owner death. On a
network mount that ignores or weakens advisory locks, exclusive creation still
elects one marker and the immutable-entry/schema-v3 commit protocol still
prevents torn entries. A lease protocol cannot guarantee exactly-once work
during a partition or metadata-coherence failure; in that case redundant
downloads or bounded timeout errors are allowed, while cache safety is not.

Failure behavior is explicit:

- A slow but refreshing owner remains owner. Waiters do not download.
- A process death stops progress. After a full liveness window, one waiter
  retires the unchanged record and competes for ownership.
- A suspended process longer than the liveness window is treated as dead. It
  is fenced by its token when it resumes and may not publish through the owner
  API.
- A heartbeat write failure marks the owner unusable. Publication returns an
  ownership-lost error; dropping the owner attempts to release its record.
- A transient waiter read or metadata error is returned. IO failure is not
  silently reinterpreted as owner death.
- A network partition can make both sides believe progress stopped. Each side
  may fetch, but every publication is still a complete atomic schema-v3
  transaction. After coherence returns, only `current.json` selects an entry.
- A filesystem without the atomic create/rename and durability guarantees
  already required by the native exact cache remains outside the native
  durability guarantee. Heartbeats do not weaken or enlarge that guarantee.

## Waiter algorithm and bounds

The public options are `poll_interval`, `heartbeat_interval`,
`liveness_timeout`, and `wait_timeout`. All are positive and the heartbeat
interval must be shorter than the liveness timeout.

Each pass performs these steps:

1. Read and fully verify `current.json`. If present, return `Hit`.
2. Read `in-flight.json` and its token-specific heartbeat fingerprint.
3. If no marker exists, take the short transition lock, repeat steps 1 and 2,
   and attempt exclusive marker creation. The winner returns `Owner`; losers
   resume waiting.
4. If the snapshot changed, reset the monotonic no-progress timer.
5. If the snapshot is unchanged for `liveness_timeout`, take the short
   transition lock and repeat the committed-entry and exact-snapshot checks.
   Only an exact match may be retired. The retiring waiter then attempts
   exclusive creation of its own record while still in that transition.
6. Otherwise sleep for the smaller of the poll interval and the remaining
   liveness/total-wait budget.

The total wait is bounded by `wait_timeout`. An unchanged owner reaching the
liveness threshold is considered for takeover before the total-time check, so
equal liveness and wait deadlines do not spuriously fail. If total wait expires
while the owner is still live, the call returns `SingleFlightTimeout`. It does
not steal a live lease and does not tell the caller to download independently.
Applications that deliberately want today's duplicate-download behavior can
continue using `read`, `lock`, and `publish` without selecting single-flight.

### Takeover race interleavings

The difficult interleavings are handled as follows:

- Two waiters see the same stale snapshot. Both request the transition lock.
  The first rechecks, renames the exact record to a unique retired name, and
  creates a new marker. The second then sees a different snapshot and cannot
  retire it. If advisory locking is ineffective, exclusive creation still
  selects one visible successor; a contender that moved a record other than
  its observed bytes restores/abandons it and restarts without becoming owner.
- The old owner refreshes before retirement. The heartbeat fingerprint no
  longer matches, so retirement is rejected and the no-progress timer resets.
- The old owner refreshes after its record was retired. The pulse is attached
  to the old token and cannot refresh a successor. Its next explicit heartbeat
  or publication check reports lost ownership.
- The old owner begins publication while a waiter attempts retirement. On
  normal local filesystems the short advisory lock orders the final token
  check and publication against retirement. If a mount violates advisory-lock
  exclusion, both writers can at worst publish complete immutable entries;
  schema-v3 atomic replacement still admits no mixed entry.
- A contender is paused after its stale recheck. On resumption it must compare
  the record it retired with the bytes it observed. A newer token is not
  accepted as stale. A brief marker absence can cause another contender to win
  exclusive creation, but neither contender can publish without later proving
  that its own token is current.

Strict exactly-once execution is impossible for a filesystem lease during an
unbounded partition or pause: choosing availability permits duplicate work,
while choosing strict exclusion can strand the cache forever. This design
chooses bounded recovery and token-fenced publication. In the ordinary coherent
filesystem case, the transition recheck plus exclusive creation provides one
owner and one download.

## Crash safety and failpoints

`in-flight.json` is never read as a commit record and never points at entry
bytes. Readers continue to accept only a verified schema-v3 `current.json`.
The existing publication ordering is unchanged: immutable files and their
directories are synchronized before the temporary commit marker is renamed,
and the control directory is synchronized afterward.

New crash boundaries are instrumented by the existing
`exact-cache-test-failpoints` feature:

- `after_inflight_heartbeat`: initial token heartbeat is durable but no owner
  record need exist.
- `after_inflight_marker`: the owner record file is synchronized but its
  directory entry may not be durable.
- `after_inflight_sync`: the visible owner record and directory are durable.
- `after_inflight_heartbeat_refresh`: one progress pulse is synchronized.
- `after_inflight_retire`: the active record was renamed to a unique retired
  name but the directory rename may not be durable.
- `after_inflight_retire_sync`: retirement is directory-durable.
- `after_inflight_reap`: retired owner/heartbeat files were removed but the
  removals may not be durable.
- `after_inflight_reap_sync`: retirement cleanup is directory-durable.

Death at the first four boundaries leaves no committed change. Death during
retirement can leave an active record, no active record, or an ignored retired
record. If publication preceded retirement, `current.json` already selects the
complete new entry; otherwise it still selects the old entry or is absent.
Recovery treats remaining coordination artifacts only as liveness state. Thus
every new boundary preserves the original invariant: old complete entry, new
complete entry, or no entry, never a torn entry.

The process-kill suite covers every listed boundary. Separate process tests
cover a live owner/waiter handoff and dead-owner takeover. Expiry tests use an
injectable monotonic clock; filesystem and scheduling wall time is used only
for test-harness safety deadlines.

## API and compatibility

`ExactProductCache::new`, `read`, `lock`, `publish`, and
`cleanup_abandoned` retain their signatures and behavior. The additive API is:

- `ExactCacheSingleFlightOptions`, including the four bounded durations;
- target-neutral `ExactCacheSingleFlightWait` and
  `ExactCacheSingleFlightDecision`, which own progress, liveness, takeover,
  and total-timeout decisions for native and browser substrates;
- `ExactProductCache::open_single_flight(options)`;
- `ExactCacheOpen::{Hit, Owner}`; and
- `ExactCacheOwner::{heartbeat, publish}`.

Dropping an owner abandons the attempt and best-effort releases its marker.
Publishing consumes the owner, verifies its token, performs the unchanged
schema-v3 commit, and retires coordination state. Code that never calls
`open_single_flight` creates no sidecars and behaves exactly as before.

The coordination representation and decision kernel use portable primitives.
The native cache as a whole retains its existing Linux/macOS platform gate
because schema-v3 directory durability has not yet been implemented and
qualified on Windows. A Windows native port can reuse this single-flight
protocol, but must first provide the pre-existing atomic publication guarantee;
single-flight does not relax that prerequisite.

### Mixed-version matrix

| Participants | Committed-entry safety | Download coalescing |
| --- | --- | --- |
| old + old | Unchanged schema-v3 lock/commit guarantee | Only when both old callers hold the legacy lock across acquisition, as before |
| old writer + new waiter | New code respects the old transition/publication lock and verifies schema v3 | The new waiter normally returns the old writer's commit; no sidecar is required |
| new owner + old requester | Both publish only complete schema-v3 entries; the old code ignores sidecars safely | Not guaranteed: the old requester cannot know about the new sidecar and may download concurrently |
| new + new | Unchanged schema-v3 safety plus token fencing | One download on coherent filesystems; bounded recovery rules apply on failure |
| old reader + new writer | Old reader follows only unchanged `current.json` | Not applicable to reads |

No new code removes or rewrites an old version's committed marker because of
in-flight state. Old cleanup ignores the new filenames. New cleanup may remove
only retired coordination artifacts and the heartbeat belonging to the token
it retired. This makes the sidecar safe for rolling upgrades and downgrades.

## Binding follow-up

- Python adds one optional exact-cache single-flight option object/field with
  poll, heartbeat, liveness, and total-wait durations. Its extension maps
  `Hit` and `Owner`; acquisition and filesystem transitions stay in Rust.
- Elixir adds the equivalent option keys in `Sidereon.GNSS.ExactCache`; the NIF
  maps the same two outcomes and owner publication. The Elixir module adds no
  filesystem logic.
- C adds an options struct (size/version guarded in the normal ABI style), an
  open result discriminant, and opaque owner heartbeat/publish/release calls.
- WASM keeps schema-v3 `buildExactCacheCommit`/`verifyExactCacheCommit` bytes.
  `BrowserExactProductCache` maps the open decision into one IndexedDB
  read-write transaction: return the committed record if present, otherwise
  create/read the in-flight object and choose owner or waiter. IndexedDB
  serializes conflicting read-write transactions, so compare-and-swap,
  takeover, and commit ordering are transaction outcomes; the JavaScript file
  does not reproduce native rename races or advisory-lock reasoning. A timer
  updates the token's heartbeat object; bounded polling passes the transaction's
  opaque owner/heartbeat revision and browser monotonic time to core's
  `ExactCacheSingleFlightWait`, then performs the returned wait, takeover, or
  timeout action. JavaScript therefore owns storage/timer adaptation but no
  liveness rules. Web Locks may remain a same-origin accelerator, not a second
  state machine.
