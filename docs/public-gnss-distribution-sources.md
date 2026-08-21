# Public GNSS distribution sources

## Decision

Sidereon models an exact GNSS product independently from the distributor used
to obtain it. A distributor may change the public URL and transport compression;
it cannot change the publisher, product line, solution class, issue, date,
cadence, family, format, or official decompressed filename.

The Rust core remains network-free. It owns catalog selection, exact identity,
official filename and source-location derivation, safe cache-relative paths,
and SP3/IONEX parsing. Bindings that already own acquisition, Python and
Elixir, own authenticated HTTP, retries, cookies, cache IO, and credential
configuration. C and WebAssembly expose the same pure identity and location
derivation without adding hidden network behavior.

## Public model

`ProductIdentity` contains the public product family, publisher, solution class,
campaign token, filename version, date, issue/start time, coverage token,
sampling token, official filename, format, and prediction horizon when the
catalog line defines one. `ProductIdentity::validate` rejects a caller-built
value when those fields disagree with the filename. URL, request, and cache-key
helpers all invoke that validation before using the value.

`DistributionSource` has four explicit values:

- `direct`: the existing cataloged analysis-center or IGS archive;
- `nasa_cddis`: NASA CDDIS over HTTPS;
- `local_file`: bytes read from a caller-selected path;
- `in_memory`: bytes supplied by the caller.

`DistributionLocation` records the source, public URL when one exists, archive
filename, and transport compression. The official filename in the identity is
the decompressed standard-product filename. That permits two distributors to
serve the same exact bytes with different transport packaging without treating
compression as a different scientific product.

`ProductRequest` holds one validated identity and a non-empty ordered list of
acceptable distributors. It grants no permission to try another analysis
center, tier, issue, date, cadence, or family.

## Complete exact product sets

Workflows that require several products can declare the full identity inventory
and call `validate_exact_product_set(expected, available)` after every product
has passed acquisition validation. The gate rejects an empty declaration,
duplicates on either side, missing identities, and undeclared identities. It
compares complete identities rather than filenames, so two prediction tiers
that publish the same filename remain distinct.

The gate is sans-IO and does not make cache writes transactional. Its contract
is that dependent processing starts only after it returns `Ok(())`. Pass only
resolved identities from successful acquisitions as `available`.

SP3 observed/predicted timing is a separate content property. Read it from
`Sp3::prediction_summary()`, which aggregates the product's record flags. Do
not derive that boundary from issue times, nominal durations, or catalog
prediction fields.

## CDDIS paths

IGS combined final SP3 identity is date-aware. The official rapid/final orbit
combination begins at GPS week 0730 (1994-01-02), so earlier final-SP3 requests
are rejected. Before GPS week 2238 the official decompressed filename is
`igs<gps-week><day>.sp3`; from week 2238 it is
`IGS0OPSFIN_<YYYY><DDD>0000_01D_15M_ORB.SP3`. CDDIS uses a four-digit GPS-week
directory and preserves the matching transport compression in each era:

```text
https://cddis.nasa.gov/archive/gnss/products/<four-digit-gps-week>/igs<gps-week><day>.sp3.Z
https://cddis.nasa.gov/archive/gnss/products/<four-digit-gps-week>/<official-filename>.gz
```

The cutoff comes from the IGS transition guideline, and archive objects on
both sides of the boundary confirm it. Current IGS final SP3 is classified as
`final`; IGS broadcast navigation remains `broadcast`. Call
`product_solution_class(center, family)` when the family is known. The legacy
center-only `AnalysisCenter::solution_class()` remains for source
compatibility, but cannot express both IGS product lines.

The BKG archive supports the current
`IGS/products/<gps-week>/<long-filename>.gz` layout. Its historical listings do
not establish one uniform direct path: week 2235 legacy products are under
`IGS/products/orbits/2235`, while week 2236 contains long-name trial products
under `IGS/products/2236` and only a partial legacy set under
`IGS/products/orbits/2236`. Sidereon therefore returns
`UnsupportedDistributionEra` for a pre-week-2238 IGS final-SP3 direct-BKG
location instead of guessing. The same exact historical identity can be
resolved through the verified CDDIS layout.

The long-name transition is also a distributor boundary. CDDIS resolution
rejects every long-name SP3 or IONEX identity dated before week 2238,
regardless of analysis center. The only pre-2238 SP3 identity modeled for
CDDIS is the IGS combined final product's verified legacy short name and `.Z`
packaging. This prevents a valid direct-archive long name from being projected
into an unverified historical CDDIS path.

CDDIS support is exact-product specific, not publisher-wide. Sidereon does not
map ESA's `ESA0MGNFIN` final SP3 identity to CDDIS: the official ESA archive
serves that line directly, while the public CDDIS catalog evidence used in this
audit does not establish an exact `ESA0MGNFIN` object. This is an unsupported
distribution, not permission to substitute another ESA final-orbit family.

Current long-name IONEX products resolve to:

```text
https://cddis.nasa.gov/archive/gnss/products/ionex/<year>/<day-of-year>/<official-filename>.gz
```

The core rejects CDDIS requests for product families for which this mapping is
not implemented. It does not relabel another file as the requested product.

## CODE product routes

AIUB publishes product families in distinct directories behind its HTTPS
download service. Sidereon routes each catalog family independently:

```text
CODE MGEX final SP3/clock:  https://www.aiub.unibe.ch/download/CODE_MGEX/CODE/<year>/...
CODE final IONEX:          https://www.aiub.unibe.ch/download/CODE/<year>/...
CODE rapid IONEX:          https://www.aiub.unibe.ch/download/CODE/...
CODE ultra-rapid SP3:      https://www.aiub.unibe.ch/download/CODE/...
```

The `cod` SP3 and clock catalog entries describe the current MGEX final line;
its IONEX entry describes the operational final line. Historical CODE
short-name products use different identities and layouts. Until those are
modeled explicitly, `AnalysisCenter::Cod` rejects SP3, clock, and IONEX dates
before GPS week 2238 with `UnsupportedProductEra`; it never fabricates a
current long filename for a historical request.

CODE P1 and P2 predicted maps use separate AIUB tiers. Direct locations resolve
the exact product identity to:

```text
https://www.aiub.unibe.ch/download/CODE/IONO/P1/<identity-year>/<official-filename>.gz
https://www.aiub.unibe.ch/download/CODE/IONO/P2/<identity-year>/<official-filename>.gz
```

The HTTPS redirect chain is restricted to AIUB's download host and public
object-store host. A missing exact URL remains a not-published result; direct
location derivation performs no date lookback or tier substitution.

## Nominal next-issue schedule

`next_issue_due(center, product_type, now)` is a network-free query over the
same catalog identities used by publication status. It returns the first due
time at or after `now`, the exact `ProductIdentity`, and half-open observed and
predicted coverage intervals. It does not fetch a listing and does not claim
that an archive has posted the issue.

The due-time rules distinguish a filename coverage epoch from publication.
For IGS combined ultra-rapid SP3, a two-day filename names 24 observed hours
followed by 24 predicted hours and is released 27 hours after its coverage
start. The analysis-center ultra lines use the 26 h 50 min submission deadline.
GFZ rapid SP3 and CLK use the next-day 15:45 UTC analysis-center deadline. CODE
rapid IONEX uses the following 00:00 UTC boundary for its published less-than-24
hour latency. CODE predicted IONEX uses its cataloged one-day or two-day horizon.

Final products are weekly batches of daily identities. The query names the
Saturday identity, which is the newest identity publication-status monitoring
expects from that batch. Analysis-center final SP3 and CLK batches are due at
Wednesday 05:00 UTC, 11 days after GPS week end. IGS combined final is due by
Friday, 13 days after week end. Because that source gives a day but no hour, the
catalog represents the deadline as Friday 23:59:59 UTC. Final IONEX uses its
published approximately 11-day weekly latency and the same end-of-day rule for
the otherwise date-only deadline.

These resolutions are catalog policy, not inferred timestamps. Their source
URLs, access date, `curl 8.7.1` retrieval record, `shasum -a 256` digests, and
the date-only resolution rules are committed in
`crates/sidereon-core/tests/fixtures/data/nominal_issue_schedule_provenance.json`.
The sources are the IGS products page, the IGS Analysis Center Coordinator
schedule, and the CODE, ESA, and GFZ analysis-center descriptions.

The current source set does not establish a nominal due-time rule for WUM's
near-real-time line or broadcast navigation. Those pairs return
`UnsupportedNominalSchedule` rather than inheriting a nearby cadence.

The scoreboard publication-status outcome carries `next_issue` beside
`behind_nominal_minutes`. Its CLI renders both with:

```text
sidereon-scoreboard --publication-status CENTER PRODUCT --at YYYY-MM-DDTHH:MM:SSZ
```

## GFZ rapid SP3 cadence eras

GFZ changed its operational rapid-orbit cadence inside GPS week 2158. Its
official listing publishes `GFZ0OPSRAP_20211370000_01D_15M_ORB.SP3.gz` for
2021 day 137 and `GFZ0OPSRAP_20211380000_01D_05M_ORB.SP3.gz` for day 138; the
subsequent products in that directory retain `05M`. Current rapid listings
also publish `05M`, including the verified 2026 day-200 object.

`default_sample(AnalysisCenter::Gfz, ProductType::Sp3)` retains its date-free
signature and now reports the current `05M` convention. Code deriving a dated
product should use `default_sample_for_date`, which returns `15M` through 2021
day 137 and `05M` from day 138. All catalog helpers that receive `sample=None`,
including `product` and `mgex_sp3`, use the date-aware query. An explicit
sampling token remains explicit and is not silently rewritten.
`supported_samples` reports the complete evidenced set for a date and issue;
constructors reject any other cadence before deriving a filename or URL.

```text
through 2021-05-17: https://isdc-data.gfz.de/gnss/products/rapid/w2158/GFZ0OPSRAP_20211370000_01D_15M_ORB.SP3.gz
from 2021-05-18:    https://isdc-data.gfz.de/gnss/products/rapid/w2158/GFZ0OPSRAP_20211380000_01D_05M_ORB.SP3.gz
current example:    https://isdc-data.gfz.de/gnss/products/rapid/w2428/GFZ0OPSRAP_20262000000_01D_05M_ORB.SP3.gz
```

## SP3 product-era floors and ultra-rapid cadence

The catalog does not extend a verified long-name series backward merely
because its filename can be formatted. Direct product derivation starts at the
first official archive object established for each modeled line:

```text
ESA final SP3/clock: 2014-01-05
GFZ rapid SP3/clock: 2020-05-13
IGS ultra SP3:       GPS week 2238 (2022-11-27)
CODE ultra SP3:      GPS week 2238 (2022-11-27)
ESA ultra SP3:       2022-10-04
GFZ ultra SP3:       2020-10-06
```

Earlier requests return `UnsupportedProductEra`. Ultra-rapid issue selection
also applies the floor: a target on the first publication date never fabricates
previous-day candidates, and a target before the floor is rejected.

ESA ultra-rapid SP3 used `15M` through the 0600 issue on 2025-02-02 and `05M`
from that day's 1200 issue onward. GFZ ultra-rapid SP3 defaults to `15M` through
2021-05-15 and `05M` from 2021-05-16. The GFZ listing contains one overlapping
`05M` object at the 0000 issue on 2021-05-15; the rest of that day's published
issues remain `15M`, so the deterministic default remains `15M` for that date
and `05M` remains an explicit alternate candidate.

`default_sample_for_date` keeps its date-only signature. For an issue-based
product it reports the `0000`/start-of-day convention; consequently ESA ultra
returns `15M` for 2025-02-02. Product construction uses the actual issue, so an
omitted sample resolves to `15M` at 0600 and `05M` at 1200. Ultra-SP3 location
candidates contain only the cadence and span published for that exact issue.
The only cataloged two-cadence issue is GFZ `2021-05-15 0000`, where the
official listing contains both `02D_15M` and `02D_05M` objects. CODE additionally
publishes the moving `COD0OPSULT.SP3` snapshot documented by AIUB. It is not an
exact alias for the dated one-day product and is outside the exact-candidate
API.

Current official listings confirm one dated orbit variant per issue for IGS,
ESA, and GFZ: IGS `02D_15M`, ESA `02D_05M`, and GFZ `02D_05M`. AIUB documents
CODE's dated ultra-rapid orbit as `01D_05M` and separately documents the moving
snapshot. Corresponding speculative alternate-span and cross-cadence paths
returned 404 during the 2026-07-21 audit and are not catalog candidates.

```text
ESA last 15M issue:  https://navigation-office.esa.int/products/gnss-products/2352/ESA0OPSULT_20250330600_02D_15M_ORB.SP3.gz
ESA first 05M issue: https://navigation-office.esa.int/products/gnss-products/2352/ESA0OPSULT_20250331200_02D_05M_ORB.SP3.gz
GFZ last 15M date:   https://isdc-data.gfz.de/gnss/products/ultra/w2157/GFZ0OPSULT_20211352100_02D_15M_ORB.SP3.gz
GFZ first 05M date:  https://isdc-data.gfz.de/gnss/products/ultra/w2158/GFZ0OPSULT_20211360000_02D_05M_ORB.SP3.gz
```

## SP3 filename epoch and content start

An SP3 long filename identifies an epoch, but official archive content shows
that the epoch was not always the first epoch inside the product. Current
cataloged lines use the filename epoch as their content start. Historical GFZ
ultra-rapid products instead began 24 hours before the filename epoch through
2022-09-06. GFZ changed conventions issue by issue over the next two days:

| Filename date | 0000 | 0300 | 0600 | 0900 | 1200 | 1500 | 1800 | 2100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| through 2022-09-06 | -1 day | -1 day | -1 day | -1 day | -1 day | -1 day | -1 day | -1 day |
| 2022-09-07 | aligned | -1 day | -1 day | -1 day | -1 day | -1 day | -1 day | -1 day |
| 2022-09-08 | aligned | -1 day | -1 day | aligned | aligned | aligned | aligned | aligned |
| from 2022-09-09 | aligned | aligned | aligned | aligned | aligned | aligned | aligned | aligned |

The convention before and during the transition is based on a complete header
sweep of the official GFZ ultra-rapid series from 2020-10-06 through
2022-09-08, performed on 2026-07-21. The audit requested all eight documented
issues per day: 5,586 official objects were present, 38 requests were ordinary
HTTP 404 publication absences, 5,579 present objects used the one-day offset,
and exactly seven present transition issues were aligned. No present object had
an unexpected content start and no transport or header decode failed. The
3.9-MiB audit manifest is retained as release evidence with SHA-256
`f592591880c6325e32e48ed48a291d93f079bf8cabad18f83c80f0e1fd5c5131`.
The table is intentionally explicit because the 2022-09-08 0300 and 0600
issues return to the older convention after the aligned 0000 issue; a single
issue-time threshold would be incorrect.

`sp3_content_start_convention` exposes this product- and issue-aware catalog
fact and rejects issues that the selected center does not publish.
`ExactSp3Request::from_identity` derives the required first epoch from it before
reading product bytes. There is no caller override. Exact validation still
requires the line-1 declared start, line-2 GPS week and seconds-of-week and MJD
fields, and first parsed epoch all to equal that one required instant.
`ExactSp3Request::new`, which has no product-line identity, continues to require
the supplied date and issue themselves as the first epoch.

## SP3 terminal-record interoperability

SP3-d identifies the last logical record by the `A3` field `EOF` in columns
1-3. It does not prescribe a transport line ending and does not explicitly
assign columns after column 3 on that record. Official products nevertheless
establish three interoperable encodings:

| Official product inspected | Logical terminal record |
| --- | --- |
| ESA `ESA0MGNFIN_20251960000_01D_05M_ORB.SP3.gz` | `EOF` plus 77 ASCII spaces (80 bytes), then LF |
| GFZ `GFZ0OPSRAP_20262000000_01D_05M_ORB.SP3.gz` | `EOF` plus 37 ASCII spaces (40 bytes), then LF |
| BKG-hosted `IGS0OPSFIN_20251960000_01D_15M_ORB.SP3.gz` | bare `EOF`, then LF |

Sidereon therefore recognizes a complete logical record beginning with `EOF`
in columns 1-3 and followed only by ASCII spaces, up to a conservative
80-column bound. The bound is Sidereon interoperability and input-robustness
policy, not a claim that SP3-d explicitly mandates an 80-column EOF record. LF
and CRLF are accepted as text-record separators, and a final record without a
separator is accepted. Empty or ASCII-space-only records after `EOF` are also
tolerated; that is a Sidereon compatibility policy rather than a requirement of
SP3-d.

Recognition is anchored to the complete logical record, not an unrestricted
substring search. Missing markers, leading whitespace, `EOFX`, `EOF X`, tab
padding, padding past column 80, a lone CR, nonblank records after the marker,
and a marker before the declared final epoch are rejected by exact validation.
A malformed EOF-like record is reported separately from a genuinely absent
record, while both remain terminal product-integrity failures.

The ESA gzip transport was independently checked end to end: the HTTP body is
966,204 bytes, `gzip -t` validates the member and its CRC32/ISIZE trailer, and
the 2,740,975-byte decompressed payload ends in the 80-byte padded record plus
one LF with no later data. Its compressed SHA-256 is
`c4f42d3c49c1d692b6c7584360c1621db75ad465ea44825fa9ce0581834050b9`; the
decompressed SHA-256 is
`8a5dfa77d3bad74fcd22a9f19140b608c67e9c4f3c273c655bb97b037fdab9aa`.

## Exact SP3 acceptance

`Sp3::parse` remains the general SP3 reader. Exact-product acquisition uses
`ExactSp3Request` with `parse_exact_sp3` or `validate_exact_sp3` before it
accepts bytes for a declared identity. That gate:

- accepts only a positive, fixed-duration requested cadence and rejects
  unknown units, `00U`, noncanonical equivalents such as `60M` or `24H`, and
  non-finite, zero, negative, or out-of-range header cadence;
- requires the header cadence to equal the requested sample interval;
- requires the line-1 start, line-2 GPS-week/seconds-of-week/MJD start fields,
  and first parsed epoch to represent the requested cataloged content start,
  and requires the header epoch count to equal the parsed count;
- requires the mandatory SP3 header/EOF structure, exact agreement between the
  line-3 satellite declaration and per-epoch P/V record count and order,
  P-then-V pairing for velocity products, at least four header comment records,
  and no nonblank records after `EOF`;
- optionally binds the SP3 line-1 producing-agency field to the exact catalog
  identity (including the official `ESOC` and `AIUB` header codes, which differ
  from the `ESA` and `COD` filename producer codes);
- requires a strictly increasing, regular parsed epoch grid at the requested
  cadence; and
- derives the permitted count from the validated request span and cadence. A
  one-day, five-minute product may contain 288 half-open epochs ending at
  23:55, or 289 inclusive epochs ending at the next midnight. Shorter, longer,
  or irregular grids fail integrity validation.

The requested duration, rather than an untrusted SP3 header value, is the
source of truth for the coverage check. The SP3-d header still supplies an
independent cadence and count that must agree.

## Candidate fallback

Candidate resolution distinguishes ordinary publication absence from product
integrity failure. An explicitly recognized not-posted response can advance to
the next officially cataloged candidate. Malformed or unparseable bytes,
digest failure, start/identity mismatch, cadence mismatch, irregularity, and
span mismatch are terminal by default, and the first such error is preserved.
Caller configuration errors and unsupported center/product pairs are also not
reported as publication absence.

`ultra_sp3_locations` applies the same date/issue-aware cadence gate and the
cataloged span before emitting any URL. It does not generate a different span
or a cadence merely because that token would form a syntactically plausible
long filename. Moving latest-product snapshots are excluded from this exact
dated-product API. CODE's current moving snapshot covers a different time
window and duration from the dated one-day product, so ordinary absence of the
dated object does not authorize trying it as the same identity.

No official material reviewed for this audit documented a moving-alias rule
that permits broad fallback after failed content validation, so Sidereon adds
no such exception. A future exception would require a documented archive race
and a narrow dedicated state; it must not catch the general integrity-error
class.

## Authentication and transport boundary

Credentials are binding inputs, never core state. Python and Elixir accept a
caller-supplied Earthdata bearer token or the documented netrc mechanism.
Authentication headers are restricted to the approved CDDIS and Earthdata
Login hosts. Redirects are explicit and HTTPS-only for that flow. Cookies obey
their host/domain, path, and secure restrictions. Recorded URLs omit user
information, queries, and fragments, and neither errors nor provenance contain
headers, cookies, tokens, or passwords.

Acquisition distinguishes authentication required, authentication failed,
authorization denied, absent/not-yet-published, retired endpoint, redirect
policy, malformed URL, transport, content type, obvious HTML error document,
content length, decompression, caller checksum, product validation, and cache
failures. Retries are bounded and limited to connection/timeouts, HTTP 408/429,
and server errors.

## Content validation and provenance

A successful network status is not sufficient. Acquisition applies archive and
decompressed size limits, checks declared content length, rejects HTML, verifies
gzip completion and caller checksums, parses the standard product, and checks
its start date/time and cadence against the exact request. The resolved identity
adds the observed SP3 or IONEX format version.

Success returns the verified local path plus provenance containing requested
and resolved identity, publisher, distributor, official filename, sanitized
original and final URLs, retrieval time, decompressed and archive byte lengths
and SHA-256 hashes, compression, ETag/Last-Modified when available, cache-hit
state, and sanitized failures from earlier explicitly allowed distributors.

## Cache policy

The cache separates distributor and every exact identity discriminator. The
decompressed product, original downloaded archive, and JSON provenance sidecar
are retained. A cache hit rechecks identities, byte counts, both hashes, caller
checksum, and a fresh product parse with semantic checks.

## Merged-SP3 input identity

Every contributor accepted by a merged-SP3 acquisition has two separate public
records. `Sp3ArtifactIdentity` is reproducible: it binds requested and resolved
`ProductIdentity`, the selected distributor, official decompressed filename,
SHA-256 digest and byte length of both product and distributor archive, and
archive compression. Retrieval time, cache-hit status, sanitized URLs, HTTP
metadata, and failed attempts are acquisition observations and do not enter the
artifact identity.

`Sp3MergeInputIdentity::new` validates complete artifact records and binds the
canonical contributor set plus every `MergeOptions` control to a versioned
`sidereon-sp3-merge-input-v1:<sha256>` identifier. Contributor enumeration and
set/map iteration order do not affect that identifier for mean or median
combination. With precedence combination, contributor order is an effective
merge-policy control and is therefore bound in order; reversing it can change
the merged bytes and changes the identifier. A different verified artifact,
resolved identity, contributor set, or merge option also changes it. Empty,
duplicate, malformed, non-SP3, or internally mismatched contributor records are
rejected rather than inferred from filenames or cache contents.

The stable identifier deliberately contains no retrieval observations,
credentials, cookies, headers, URLs, or filesystem paths. Persist the public
artifact records and merge policy alongside it; `verify` recomputes the
canonical identifier from those records. Single-contributor and
multi-contributor merges use the same schema.

Accepted negative-zero tolerances canonicalize to positive zero because both
values execute identically. The literal public contract vectors in
[`sp3-merge-input-v1.json`](../crates/sidereon-core/golden/sp3-merge-input-v1.json) bind
the complete policy and exact artifacts for Rust, Python, Elixir, C, and WASM.

The acquisition-capable Python and Elixir interfaces publish the product,
original archive, and JSON provenance as one immutable transaction. A single
SHA-256-bound commit record names that transaction and is atomically replaced
only after the entry files and directories have been synchronized. Readers
follow only that record and then repeat the identity, source, digest, length,
caller-checksum, and semantic checks.

On Linux and macOS, both interfaces use the same per-entry POSIX advisory lock
across cache validation, acquisition, and commit. The wait is bounded; a lock or
cache-write failure is terminal rather than permission to try another source.
OS process death releases the lock automatically, allowing a later owner to
clean abandoned transactions without deleting a live writer's work. Valid
0.29.0-0.29.2 three-file entries are revalidated and migrated into the committed
layout without a new download.

The crash guarantee relies on a local filesystem providing atomic
same-directory rename, POSIX advisory locks, regular-file synchronization, and
directory synchronization. Under those Linux/macOS guarantees, a process death
or power loss during publication leaves the previous complete entry or no
acceptable entry; it cannot expose a mixed payload/provenance pair. A verified
existing entry is returned without contacting a remote service, including in
offline mode.

The [cache atomicity audit](exact-product-cache-atomicity.md) records the 0.29.2
verdict, corrected protocol, process/failpoint coverage, compatibility, and
residual risks.

## Compatibility and extension

Python and Elixir route the legacy IONEX convenience API through exact
acquisition. Its explicit lookback option still controls candidate dates, but
each candidate now uses the versioned exact cache and full semantic validation;
unverified entries in the former flat cache are not accepted implicitly.
Adding another public distributor requires a location/compression mapping for
an existing identity plus the same redirect, size, content, parse, provenance,
and cache gates. It must not modify identity fields.

The product-aware solution-class query and exact-SP3 validator are additive.
Existing IGS broadcast-navigation derivation and the legacy center-only
solution-class query retain their signatures. The additive
`default_sample_for_date` query preserves historical GFZ derivation, while the
legacy date-free query now returns GFZ's corrected current `05M` value. For
issue-based products, the date-only query represents the 0000 issue; product
construction and candidate ordering use the actual issue. Behavior is
deliberately stricter for invalid caller-built identities, pre-series dates,
unsupported center/product combinations, pre-transition `cod` long-name
requests, unverified pre-transition CDDIS long-name paths, and acquired SP3
bytes that do not meet an exact request. Serialized SP3 text now includes at
least four comment records; blank structural padding is not returned as
semantic `Sp3::comments` text. The new
`ArchiveCompression::UnixCompress` variant and the added typed catalog and
scoreboard error variants are source-visible API additions for exhaustive Rust
matches.

Those catalog, validation, and cache changes shipped in `0.33.0` as a minor
release because they added public API and made previously accepted ambiguous or
integrity-invalid inputs fail.

The terminal-record correction, historical GFZ content-start catalog, and
public content-start query are prepared for `0.34.0`. The EOF change is parser
compatibility, but the inspectable catalog enum/query and the newly enforced
historical GFZ identity semantics warrant a minor release. No numerical
calculation changes.

## Public evidence for this audit

Unless a row states otherwise, sources were accessed on 2026-07-20. Object and
directory links record the archive evidence observed on the listed date;
availability of an individual file is not a promise that it will remain
mirrored by every distributor.

| Catalog or validation decision | Primary public evidence | Accessed |
| --- | --- | --- |
| The official IGS rapid/final orbit combination began on 1994-01-02, GPS week 0730; earlier dates are outside that product series. | [1994 IGS Annual Report, Analysis Centre Coordinator section](https://files.igs.org/pub/resource/pubs/94an_repta.pdf) | 2026-07-20 |
| IGS final, rapid, and ultra-rapid products switched to long filenames at the start of GPS week 2238 on 2022-11-27; final orbit changed from `igs<week><day>.sp3.Z` to `IGS0OPSFIN_<epoch>_01D_15M_ORB.SP3.gz`. November 26 is the final day of week 2237. | [IGS transition guideline](https://files.igs.org/pub/resource/guidelines/Guideline_for_the_transition_of_the_IGS_products_to_IGS20_and_long_filenames_v2.0.pdf), [IGSMAIL-8256](https://lists.igs.org/pipermail/igsmail/2022/008252.html), [IGSMAIL-8274](https://lists.igs.org/pipermail/igsmail/2022/008270.html), [IGS products](https://igs.org/products/) | 2026-07-20 |
| CDDIS documents operational orbit product paths as `WWWW/AAAWWWWD.TYP.Z`, with the GPS-week field represented by four characters. | [NASA CDDIS precise-orbit documentation](https://www.earthdata.nasa.gov/data/space-geodesy-techniques/gnss/precise-orbits-product) | 2026-07-20 |
| CDDIS has the legacy week-2237 final object with Unix-compress packaging and the long-name week-2238 object with gzip packaging. | [week 2237 object](https://cddis.nasa.gov/archive/gnss/products/2237/igs22370.sp3.Z), [week 2238 object](https://cddis.nasa.gov/archive/gnss/products/2238/IGS0OPSFIN_20223310000_01D_15M_ORB.SP3.gz) | 2026-07-20 |
| CDDIS's documented legacy orbit convention and the IGS week-2238 transition support one modeled pre-transition mapping: the IGS combined-final short name with `.Z`. Sidereon does not project other centers' pre-2238 long names into CDDIS. | [NASA precise-orbit convention](https://www.earthdata.nasa.gov/data/space-geodesy-techniques/gnss/precise-orbits-product), [IGS transition guideline](https://files.igs.org/pub/resource/guidelines/Guideline_for_the_transition_of_the_IGS_products_to_IGS20_and_long_filenames_v2.0.pdf), [week 2237 legacy object](https://cddis.nasa.gov/archive/gnss/products/2237/igs22370.sp3.Z), [week 2238 long-name object](https://cddis.nasa.gov/archive/gnss/products/2238/IGS0OPSFIN_20223310000_01D_15M_ORB.SP3.gz) | 2026-07-20 |
| BKG's current direct layout is `IGS/products/<week>`; its transition-era listings do not support one uniform historical direct rule. | [week 2238 current listing](https://igs.bkg.bund.de/root_ftp/IGS/products/2238/), [week 2235 legacy listing](https://igs.bkg.bund.de/root_ftp/IGS/products/orbits/2235/), [week 2236 long-name listing](https://igs.bkg.bund.de/root_ftp/IGS/products/2236/), [week 2236 legacy listing](https://igs.bkg.bund.de/root_ftp/IGS/products/orbits/2236/) | 2026-07-20 |
| Long-name LEN/SMP syntax documents `D`, `W`, `L`, and `Y` units, while the official archive publishes `07D` despite the guideline's longest-unit prose. Sidereon therefore does not invent `D`-to-`W` or `L`-to-`Y` rewriting. Exact sub-day equivalents such as `60M` and `24H` remain noncanonical, and `00U` is unspecified rather than an exact positive cadence. | [IGS long product filename guidelines v2.2](https://files.igs.org/pub/resource/guidelines/Guidelines_for_Long_Product_Filenames_in_the_IGS_v2.2_EN.pdf), [official week-2420 `07D` product](https://igs.bkg.bund.de/root_ftp/IGS/products/2420/IGS0OPSFIN_20261440000_07D_01D_ERP.ERP.gz) | 2026-07-20 |
| SP3 line 1 declares start and epoch count; line 2 repeats the start as GPS week/seconds-of-week and MJD/fraction and declares an epoch interval strictly between 0 and 100,000 seconds. | [SP3-d specification](https://files.igs.org/pub/data/format/sp3d.pdf) | 2026-07-20 |
| SP3-d requires at least five `+` and five `++` records, at least four header comment records, line-3 satellite-count agreement, a complete ordered satellite record set at every epoch, each V record after its matching P record, and `EOF` as the last record. | [SP3-d specification](https://files.igs.org/pub/data/format/sp3d.pdf) | 2026-07-20 |
| SP3-d defines the final field as `EOF` in columns 1-3 (`A3`). Official ESA output demonstrates an 80-byte record padded with 77 ASCII spaces; a BKG-hosted IGS final product demonstrates the bare three-byte record. | [SP3-d specification](https://files.igs.org/pub/data/format/sp3d.pdf), [official ESA product](https://navigation-office.esa.int/products/gnss-products/2375/ESA0MGNFIN_20251960000_01D_05M_ORB.SP3.gz), [official BKG-hosted IGS product](https://igs.bkg.bund.de/root_ftp/IGS/products/2375/IGS0OPSFIN_20251960000_01D_15M_ORB.SP3.gz) | 2026-07-21 |
| GFZ rapid output demonstrates another valid producer convention: `EOF` followed by 37 ASCII spaces, for a 40-byte logical record. | [official GFZ rapid product](https://isdc-data.gfz.de/gnss/products/rapid/w2428/GFZ0OPSRAP_20262000000_01D_05M_ORB.SP3.gz) | 2026-07-21 |
| Official SP3 bodies identify their producing agency as `IGS`, `ESOC`, `GFZ`, and `AIUB`; these content fields bind IGS, ESA, GFZ, and CODE catalog identities without assuming the filename producer token is identical. | [IGS rapid SP3](https://igs.bkg.bund.de/root_ftp/IGS/products/2428/IGS0OPSRAP_20262000000_01D_15M_ORB.SP3.gz), [ESA rapid SP3](https://navigation-office.esa.int/products/gnss-products/2428/ESA0OPSRAP_20262000000_01D_05M_ORB.SP3.gz), [GFZ rapid SP3](https://isdc-data.gfz.de/gnss/products/rapid/w2428/GFZ0OPSRAP_20262000000_01D_05M_ORB.SP3.gz), [CODE final SP3](https://www.aiub.unibe.ch/download/CODE_MGEX/CODE/2026/COD0MGXFIN_20261920000_01D_05M_ORB.SP3.gz) | 2026-07-20 |
| AIUB identifies its current product service and CODE product series. | [AIUB services](https://www.aiub.unibe.ch/services/index_eng.html), [CODE Analysis Center](https://www.aiub.unibe.ch/research/code___analysis_center/index_eng.html) | 2026-07-20 |
| AIUB documents operational, rapid, ultra-rapid, predicted, final, MGEX, clock, SP3, and IONEX names and directories. | [AIUB_AFTP.TXT](https://www.aiub.unibe.ch/download/AIUB_AFTP.TXT) | 2026-07-20 |
| Current AIUB listings confirm MGEX final SP3/clock under `CODE_MGEX/CODE/<year>`, final products under `CODE/<year>`, and rapid/ultra-rapid products at `CODE`. | [MGEX 2026 listing](https://code.aiub.unibe.ch/s3_script/aiub_s3_bucket_listing.php?path=CODE_MGEX%2FCODE%2F2026), [CODE 2026 listing](https://code.aiub.unibe.ch/s3_script/aiub_s3_bucket_listing.php?path=CODE%2F2026), [CODE current listing](https://code.aiub.unibe.ch/s3_script/aiub_s3_bucket_listing.php?path=CODE) | 2026-07-20 |
| AIUB's P1 and P2 predicted IONEX tiers are separate paths. | [P1 2026 listing](https://code.aiub.unibe.ch/s3_script/aiub_s3_bucket_listing.php?path=CODE%2FIONO%2FP1%2F2026), [P2 2026 listing](https://code.aiub.unibe.ch/s3_script/aiub_s3_bucket_listing.php?path=CODE%2FIONO%2FP2%2F2026) | 2026-07-20 |
| GFZ rapid SP3 used `15M` through 2021 day 137 and `05M` from day 138 within GPS week 2158; its current rapid series remains `05M`. The current day-200 `05M` object returned HTTP 200 while the corresponding `15M` URL returned 404. | [GFZ week-2158 listing](https://isdc-data.gfz.de/gnss/products/rapid/w2158/), [GFZ current week-2428 listing](https://isdc-data.gfz.de/gnss/products/rapid/w2428/), [current 05M object](https://isdc-data.gfz.de/gnss/products/rapid/w2428/GFZ0OPSRAP_20262000000_01D_05M_ORB.SP3.gz), [absent 15M path](https://isdc-data.gfz.de/gnss/products/rapid/w2428/GFZ0OPSRAP_20262000000_01D_15M_ORB.SP3.gz) | 2026-07-20 |
| ESA's MGEX final SP3 and clock archive begins on 2014-01-05; the preceding week has no corresponding final-orbit or clock object. | [preceding week 1773](https://navigation-office.esa.int/products/gnss-products/1773/), [first week 1774 listing](https://navigation-office.esa.int/products/gnss-products/1774/), [first SP3 object](https://navigation-office.esa.int/products/gnss-products/1774/ESA0MGNFIN_20140050000_01D_05M_ORB.SP3.gz), [first clock object](https://navigation-office.esa.int/products/gnss-products/1774/ESA0MGNFIN_20140050000_01D_30S_CLK.CLK.gz) | 2026-07-20 |
| GFZ's rapid SP3 and clock listing begins on 2020-05-13 (2020 day 134). | [GFZ week-2105 listing](https://isdc-data.gfz.de/gnss/products/rapid/w2105/), [first rapid SP3 object](https://isdc-data.gfz.de/gnss/products/rapid/w2105/GFZ0OPSRAP_20201340000_01D_15M_ORB.SP3.gz), [first rapid clock object](https://isdc-data.gfz.de/gnss/products/rapid/w2105/GFZ0OPSRAP_20201340000_01D_30S_CLK.CLK.gz) | 2026-07-20 |
| IGS operational ultra-rapid long names start with the week-2238 transition. | [IGS transition guideline](https://files.igs.org/pub/resource/guidelines/Guideline_for_the_transition_of_the_IGS_products_to_IGS20_and_long_filenames_v2.0.pdf), [BKG week-2238 listing](https://igs.bkg.bund.de/root_ftp/IGS/products/2238/), [first long-name ultra SP3 object](https://igs.bkg.bund.de/root_ftp/IGS/products/2238/IGS0OPSULT_20223310000_02D_15M_ORB.SP3.gz) | 2026-07-20 |
| CODE ultra-rapid SP3 is modeled from the week-2238 long-name transition; earlier CODE ultra identities require a distinct legacy convention that this catalog does not invent. | [IGS transition guideline](https://files.igs.org/pub/resource/guidelines/Guideline_for_the_transition_of_the_IGS_products_to_IGS20_and_long_filenames_v2.0.pdf), [AIUB CODE listing](https://code.aiub.unibe.ch/s3_script/aiub_s3_bucket_listing.php?path=CODE) | 2026-07-20 |
| ESA's operational ultra-rapid SP3 line begins on 2022-10-04 (day 277); the preceding week contains final products but no ultra-rapid SP3. | [preceding week 2229](https://navigation-office.esa.int/products/gnss-products/2229/), [week-2230 listing](https://navigation-office.esa.int/products/gnss-products/2230/), [first ultra SP3 object](https://navigation-office.esa.int/products/gnss-products/2230/ESA0OPSULT_20222770000_02D_15M_ORB.SP3.gz) | 2026-07-20 |
| GFZ's operational ultra-rapid SP3 listing begins on 2020-10-06 (day 280). | [GFZ week-2126 listing](https://isdc-data.gfz.de/gnss/products/ultra/w2126/), [first ultra SP3 object](https://isdc-data.gfz.de/gnss/products/ultra/w2126/GFZ0OPSULT_20202800000_02D_15M_ORB.SP3.gz) | 2026-07-20 |
| ESA ultra-rapid SP3 changes from `15M` at the 2025-02-02 0600 issue to `05M` at 1200. | [ESA week-2352 listing](https://navigation-office.esa.int/products/gnss-products/2352/), [0600 15M object](https://navigation-office.esa.int/products/gnss-products/2352/ESA0OPSULT_20250330600_02D_15M_ORB.SP3.gz), [1200 05M object](https://navigation-office.esa.int/products/gnss-products/2352/ESA0OPSULT_20250331200_02D_05M_ORB.SP3.gz) | 2026-07-20 |
| GFZ ultra-rapid SP3 defaults to `15M` through 2021-05-15 and `05M` from 2021-05-16. One 0000 `05M` object overlaps the otherwise-`15M` final day. | [GFZ week-2157 listing](https://isdc-data.gfz.de/gnss/products/ultra/w2157/), [last 15M issue](https://isdc-data.gfz.de/gnss/products/ultra/w2157/GFZ0OPSULT_20211352100_02D_15M_ORB.SP3.gz), [overlapping 05M object](https://isdc-data.gfz.de/gnss/products/ultra/w2157/GFZ0OPSULT_20211350000_02D_05M_ORB.SP3.gz), [GFZ week-2158 listing](https://isdc-data.gfz.de/gnss/products/ultra/w2158/), [first next-day 05M object](https://isdc-data.gfz.de/gnss/products/ultra/w2158/GFZ0OPSULT_20211360000_02D_05M_ORB.SP3.gz) | 2026-07-20 |
| Current ultra-rapid dated orbit variants are IGS `02D_15M`, ESA `02D_05M`, GFZ `02D_05M`, and CODE `01D_05M`. The official ESA/GFZ/IGS directory listings contain no second current cadence or span, AIUB documents no dated CODE `02D_05M` orbit, and the corresponding synthesized paths returned 404. AIUB separately documents `COD0OPSULT.SP3`, but a live comparison proves it is a different moving snapshot: the 3,337,569-byte alias (SHA-256 `6a51fcc8eb22c232680b04c06b57124616ff51a76e984e8e39dcbb6477d55bb4`) starts 2026-07-20 12:00 and declares 577 epochs at 300 seconds, while the 1,473,962-byte dated product (SHA-256 `36d10bc035aebb1fbfbe8ac9fa00608c1951bf17d6f51b9380508e729da80584`) starts 00:00 and declares 289 epochs. It is therefore excluded from exact dated candidates. | [IGS transition guideline](https://files.igs.org/pub/resource/guidelines/Guideline_for_the_transition_of_the_IGS_products_to_IGS20_and_long_filenames_v2.0.pdf), [BKG week-2428 listing](https://igs.bkg.bund.de/root_ftp/IGS/products/2428/), [ESA week-2370 listing](https://navigation-office.esa.int/products/gnss-products/2370/), [GFZ week-2370 listing](https://isdc-data.gfz.de/gnss/products/ultra/w2370/), [AIUB product inventory](https://www.aiub.unibe.ch/download/AIUB_AFTP.TXT), [AIUB current CODE listing](https://code.aiub.unibe.ch/s3_script/aiub_s3_bucket_listing.php?path=CODE), [moving snapshot](https://www.aiub.unibe.ch/download/CODE/COD0OPSULT.SP3), [dated product](https://www.aiub.unibe.ch/download/CODE/COD0OPSULT_20262010000_01D_05M_ORB.SP3) | 2026-07-21 |
| GFZ ultra-rapid SP3 content began one day before its filename epoch through 2022-09-06. The transition was issue-specific on September 7 and 8, including old-convention reversions after aligned issues; products are aligned from September 9. A complete official-archive header sweep requested 5,624 date/issue combinations from series start through the transition: 5,586 objects were present, 38 were ordinary 404 publication absences, 5,579 used the one-day offset, seven transition objects were aligned, and no present object had an unexpected offset. The retained sweep manifest has SHA-256 `f592591880c6325e32e48ed48a291d93f079bf8cabad18f83c80f0e1fd5c5131`. | [GFZ ultra-rapid archive root](https://isdc-data.gfz.de/gnss/products/ultra/), [GFZ week-2226 listing](https://isdc-data.gfz.de/gnss/products/ultra/w2226/), [GPS-week-crossing pre-transition issue](https://isdc-data.gfz.de/gnss/products/ultra/w2226/GFZ0OPSULT_20222470000_02D_05M_ORB.SP3.gz), [September 7 aligned 0000](https://isdc-data.gfz.de/gnss/products/ultra/w2226/GFZ0OPSULT_20222500000_02D_05M_ORB.SP3.gz), [September 8 old-convention 0600](https://isdc-data.gfz.de/gnss/products/ultra/w2226/GFZ0OPSULT_20222510600_02D_05M_ORB.SP3.gz), [September 8 aligned 0900](https://isdc-data.gfz.de/gnss/products/ultra/w2226/GFZ0OPSULT_20222510900_02D_05M_ORB.SP3.gz), [post-transition issue](https://isdc-data.gfz.de/gnss/products/ultra/w2226/GFZ0OPSULT_20222520000_02D_05M_ORB.SP3.gz) | 2026-07-21 |
| CDDIS IONEX filenames transitioned from historical short names toward long names beginning at week 2238, with center-specific timing. Sidereon therefore does not derive a pre-transition CDDIS URL for a caller's long-name IONEX identity. | [IGS ionospheric products](https://igs.org/products/#ionosphere), [NASA Earthdata support clarification](https://forum.earthdata.nasa.gov/viewtopic.php?t=4779) | 2026-07-20 |

The archive and format sources above do not document a general permission to
continue after an exact candidate fails integrity validation. This audit found
no narrower moving-alias integrity exception to implement.

## Remaining limits in official material

- BKG publishes historical orbit files, but the reviewed public listings do
  not define one complete rule for directory buckets, filename case, and
  compression across the legacy archive. Sidereon therefore does not derive a
  generic legacy direct-BKG URL.
- No reviewed archive document identifies a moving-alias validation failure
  that is safe to recover from by selecting another product. Only ordinary
  absence has fallback semantics.

The transition date is not unresolved. IGSMAIL-8256, IGSMAIL-8274, the
transition guideline, GPS-week arithmetic, and archive objects agree that week
2238 began on 2022-11-27. The IGS products page's November 26 parenthetical is
an isolated off-by-one statement; the same page identifies that date as the
end of week 2237.

AIUB's legacy CODE names and directories are documented, so they are not an
evidence gap. Supporting them is deferred because they are distinct public
identities requiring product-specific short-name validation and distribution
handling; current long names are never substituted for them.

## Other public evidence

- [NASA CDDIS archive access](https://www.earthdata.nasa.gov/centers/cddis-daac/archive-access)
- [Earthdata Login curl and wget access](https://urs.earthdata.nasa.gov/documentation/for_users/data_access/curl_and_wget)
- [Earthdata bearer-token Python example](https://urs.earthdata.nasa.gov/documentation/for_users/data_access/python_user_token_script)
- [NASA GNSS atmospheric products](https://www.earthdata.nasa.gov/data/space-geodesy-techniques/gnss/atmospheric-products)
- [NASA Earth science data-use policy](https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy)
