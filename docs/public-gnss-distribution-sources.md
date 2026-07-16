# Public GNSS distribution sources

## Decision

Sidereon models an exact GNSS product independently from the distributor used
to obtain it. A distributor may change the public URL and transport compression;
it cannot change the publisher, product line, solution class, issue, date,
cadence, family, format, or official decompressed filename.

The Rust core remains network-free. It owns catalog selection, exact identity,
official filename and source-location derivation, safe cache-relative paths,
and SP3/IONEX parsing. Bindings that already own acquisition—Python and
Elixir—own authenticated HTTP, retries, cookies, cache IO, and credential
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

Current long-name SP3 products resolve to:

```text
https://cddis.nasa.gov/archive/gnss/products/<gps-week>/<official-filename>.gz
```

Current long-name IONEX products resolve to:

```text
https://cddis.nasa.gov/archive/gnss/products/ionex/<year>/<day-of-year>/<official-filename>.gz
```

The core rejects CDDIS requests for product families for which this mapping is
not implemented. It does not relabel another file as the requested product.

## CODE predicted IONEX paths

CODE P1 and P2 predicted maps use separate AIUB tiers. Direct locations resolve
the exact product identity to:

```text
https://www.aiub.unibe.ch/download/CODE/IONO/P1/<identity-year>/<official-filename>.gz
https://www.aiub.unibe.ch/download/CODE/IONO/P2/<identity-year>/<official-filename>.gz
```

The HTTPS redirect chain is restricted to AIUB's download host and public
object-store host. A missing exact URL remains a not-published result; direct
location derivation performs no date lookback or tier substitution.

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

This is an additive public API and should ship as the next minor release rather
than a patch release.

## Public evidence

- [NASA CDDIS archive access](https://www.earthdata.nasa.gov/centers/cddis-daac/archive-access)
- [Earthdata Login curl and wget access](https://urs.earthdata.nasa.gov/documentation/for_users/data_access/curl_and_wget)
- [Earthdata bearer-token Python example](https://urs.earthdata.nasa.gov/documentation/for_users/data_access/python_user_token_script)
- [NASA precise orbit products](https://www.earthdata.nasa.gov/data/space-geodesy-techniques/gnss/precise-orbits-product)
- [NASA GNSS atmospheric products](https://www.earthdata.nasa.gov/data/space-geodesy-techniques/gnss/atmospheric-products)
- [IGS long product filename guidelines](https://files.igs.org/pub/resource/guidelines/Guidelines_for_Long_Product_Filenames_in_the_IGS.pdf)
- [NASA Earth science data-use policy](https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy)
