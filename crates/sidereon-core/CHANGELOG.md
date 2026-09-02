# Changelog

All notable changes to `sidereon-core` are documented here.

## [Unreleased]

### Changed

- Portable dynamic matrix and matrix-vector products are pinned to nalgebra's
  fixed-order scalar path by a randomized bit-identity test (normal-equation
  orders through 500 and a 2000x200 Jacobian). CI benchmarks every
  application hotpath and portable linear-algebra case against the same-job
  merge base and fails on a regression above 25 percent.
- `nalgebra` 0.33.3 and `simba` 0.9.1 are now exact dependency pins. The
  decomposition algorithms and scalar-dispatch companion participate in the
  crate's bit-exact identity claim, so a semver-compatible update could change
  operation order, convergence thresholds, or covariance bits; any upgrade is
  now a deliberate re-pinning release rather than a side effect of `cargo
  update`.
- Native exact-cache locking now uses the stable standard-library file-locking
  API, removing the unmaintained fs2 dependency while preserving bounded,
  non-blocking retry behavior and error handling.
- **Breaking:** `terrain`, `ionex::tec_grid`, and
  `astro::propagator::dense_output` now return typed error enums from their
  public parsing, interpolation, and dense-output evaluation APIs; the error
  messages remain unchanged.
- `libm` is pinned to exactly 0.2.16. The crate's cross-platform bit-exactness
  is a property of that implementation's rounding, so a semver-compatible
  update of it could change results without any change here. The pin makes
  such a move a deliberate edit with a re-pinning pass, not a side effect of
  `cargo update`.
- OMM parsing and SP3 merging are organized into documented private stages;
  frozen serializer outputs, diagnostics, and numerical results are unchanged.

### Added

- A supply-chain gate (`cargo deny`: advisories, licenses, bans, sources)
  runs in CI, together with an MSRV check at Rust 1.89. One advisory is an
  explicit, documented exception: RUSTSEC-2024-0436 (`paste`, an unmaintained
  compile-time proc-macro reached only through the exact `simba` pin); it is
  revisited when `nalgebra`/`simba` are upgraded.
- Batch APIs now use an optional default-on `parallel` feature. Disabling it
  removes rayon and compiles the same order-preserving batch entry points with
  plain iterators, keeping results bit-identical and retaining the public
  serial variants.
- `JulianDate::new`, `whole`, `fraction`, and `from_unix_microseconds`: named
  construction and accessors for the split Julian date, and a Unix-microsecond
  conversion that shares its floor-and-remainder arithmetic with pass
  prediction (bit-identical, proven by a test over negative and day-boundary
  inputs). The tuple representation is unchanged.

### Fixed

- Documentation now states the actual single-crate layout: the GNSS layer is
  always present alongside propagation, and the units policy permits bare
  solver-space positions while keeping frame and datum names on georeferenced
  quantities.

## [1.4.1] - 2026-08-31

Supersedes 1.4.0. Solutions are unchanged; a diagnostic is restored.

### Fixed

- A RINEX NAV record probe sliced a line at a fixed byte offset and panicked
  when the header bytes were not a UTF-8 character boundary (found by fuzzing);
  it now inspects bytes and never panics on malformed input.
- A DTED coordinate field ending in a multi-byte character was sliced at an
  invalid boundary and panicked (found by fuzzing); it now returns a typed
  error.
- The core trust-region backend no longer overrides the solver's dot-product
  and matrix-vector reductions. Those fallbacks were already portable
  fixed-order arithmetic; overriding them in 1.4.0 summed the terminal
  gradient in a different order, which left every converged solution
  bit-identical but moved the reported first-order optimality (a
  cancellation-dominated number) by an order of magnitude on some fits, and
  could change the evaluation counts of a fit. 1.4.1 reports the same
  optimality and counts as 1.3.3 for fits whose path is otherwise unchanged.

## [1.4.0] - 2026-08-31

Results are bit-identical across x86_64 and arm64 targets. Relative to 1.3.3
the frozen outputs of iterative fits move in their last bits, and SVD-derived
covariance and geometry diagnostics now agree with the values 1.3.3 produced
on x86_64/glibc; position and clock results of the bundled static and SPP
fixtures are unchanged.

### Changed

- All core transcendental evaluation now delegates to the portable `libm`
  kernels, including the trust-region solve paths and numerical weighting,
  so solved results do not depend on the platform C math library.
- Core nalgebra decompositions and dynamic matrix products now run through a
  transparent portable binary64 scalar. This bypasses architecture-selected
  SIMD matrix-product kernels while retaining binary64 arithmetic and making
  SVD-derived covariance and geometry diagnostics bit-identical across targets.
- Core trust-region solves now inject their own portable numerical backend for
  SVD, powers, dot products, and matrix-vector products; the general-purpose
  trust-region crate's default backend and published parity behavior are
  unchanged.
- Robust-loss evaluation in core-driven trust-region solves (Cauchy, Arctan)
  uses portable `log1p` and `atan` through the new defaulted hooks in
  `trust-region-least-squares` 0.11.0.
- Fused multiply-add in the SP3 interpolant, geoid, frame, and vector helpers
  goes through `libm::fma` instead of the platform C library's `fma`; no
  result changed on any tested platform, the dependency did.
- A workspace lint (`clippy.toml` `disallowed-methods`) rejects any new
  platform-libm transcendental or `mul_add` call, and a guard test keeps
  production decompositions on the portable scalar.

## [1.3.3] - 2026-08-30

### Fixed

- The Moon's geocentric distance in the analytic Sun/Moon series lost its
  parallax sine in 1.3.2 (`a / parallax` instead of `a / sin(parallax)`),
  moving the Moon by about 17 km. The solid-earth tide it feeds was off by
  roughly 140 mm. 1.3.2 is superseded; every downstream interface skips it.
  A bit-pinned Moon regression now guards the series, since the DE440 golden
  tolerates the model's own ~1% and cannot see a slip of this size.

## [1.3.2] - 2026-08-30

### Fixed

- `parse_archive_listing` deduplicated by scanning every object parsed so far
  for each incoming row, which is quadratic. On AIUB's whole-tree CSV
  (~426k rows) the parse took 154 s; it now indexes each path's position and
  takes 0.23 s. Listing order and the `observed_at` backfill are unchanged.

### Changed

- Transcendental math (sin, cos, tan, atan2, asin, acos, exp, log, pow) now
  goes through portable Rust kernels rather than the platform C math library,
  so results are bit-identical across x86_64 and arm64. The full test suite
  now runs on both architectures in CI and passes bit-for-bit on each. The
  owned trust-region solver's complete subproblem assembly, not only its
  factorization, uses fixed-order scalar arithmetic.

## [1.3.1] - 2026-08-29

### Changed

- Coordination release keeping the shared release number across the language
  interfaces. Ships the Go interface relicense from Apache-2.0 to MIT
  (matching the engine and every other language interface). No numerical,
  algorithmic, or API changes in the engine.

## [1.3.0] - 2026-08-29

### Changed

- Coordination release keeping the shared release number across the language
  interfaces, which now include a Go interface. No numerical, algorithmic, or
  API changes in the engine.

## [1.2.0] - 2026-08-28

### Added

- `rinex_band_frequency_hz_classified` reports
  `Error::MissingGlonassChannel` when GLONASS G1/G2 needs an FDMA channel,
  while existing lookup behavior and frequency values remain unchanged.

### Fixed

- Lenient RINEX 4 NAV parsing now decodes GPS/QZSS CNAV-family EPH frames
  instead of skipping them.
- RTKLIB SBAS parsing now preserves the `Framed250` wire form for valid
  32-byte blocks without changing the parsed bytes.

## [1.1.1] - 2026-08-26

### Changed

- Coordination release restoring the shared release number across the language
  interfaces after the Elixir 1.1.1 patch. No numerical, algorithmic, or API
  changes.

## [1.1.0] - 2026-08-24

### Added

- `locate_source_with` and `SourceLocateConfig`, a `#[non_exhaustive]`
  configuration wrapping `SourceLocateOptions`, whose `include_influence`
  can skip the one-full-re-solve-per-sensor leave-one-out diagnostics and
  return an empty influence vector. `locate_source` is unchanged and
  equivalent to `include_influence = true`.
- `closed_form_initial_guess` names the source-localization seed for its actual
  Schau-Robinson spherical-intersection method. `chan_ho_initial_guess` remains
  as a deprecated compatibility wrapper.

### Changed

- `SourceLocateOptions` keeps its 1.0 shape; settings added from here on live
  on `SourceLocateConfig` so they stay additive.
- Source-solution rank, condition number, covariance, and GDOP now come from one
  thin SVD of the final Jacobian. The covariance is assembled as
  `V * diag(1 / sigma_i^2) * V^T` over retained singular values instead of a
  separate Cholesky inverse of the normal matrix.
- Sensor influence `score` is exactly the larger absolute full/leave-one-out
  ToA residual divided by `timing_sigma_s`. Robust downweighting remains
  available separately in `loss_weight`.
- TDOA origin time uses one robust-loss reweighting refinement after the
  position solve when a non-linear loss is selected. Linear loss retains the
  original arithmetic-mean path exactly.
- Source-localization documentation now specifies the ToA/TDOA models, state,
  sensor minima, seed method, covariance/CRLB interpretation, influence cost,
  solver termination codes, and fallible API errors.

### Fixed

- Closed-form quadratic degeneracy and discriminant checks are relative to the
  coefficient magnitudes. The ToA seed no longer rejects an otherwise finite
  candidate with an arbitrary absolute-distance cutoff.
- Empty source-localization sensor input now reports `InvalidInput` for
  `sensors` instead of a dimension-assuming `TooFewSensors { needed: 3 }`.

## [1.0.1] - 2026-08-22

### Changed

- `trust-region-least-squares` updated to 0.10.0: the injected backend contract
  is now the `HostNumerics` seam (SVD, BLAS reductions, and NumPy power
  dispatch in one fail-closed contract), and the host backend reproduces
  NumPy's stride-0 scalar-exponent power fast paths bit-for-bit. sidereon-core
  drives the solver through its data/model entry points and is unaffected at
  its own API surface.

## [1.0.0] - 2026-08-21

Sidereon 1.0.0. The public API carries a stability commitment from here:
additions arrive without breaking existing callers (MergeOptions and its
non-exhaustive construction pattern are the template), and anything that
must break waits for 2.0.0.

### Added

- Window-scoped continuity verdicts: `EpochWindow`, `StencilExtent`
  (derived from the interpolator's sliding-window order and the product's
  epoch interval, never caller-supplied), `defects_influencing` on
  `ContinuityReport` and `MergeReport`, and accept/refuse verdict helpers
  that name the influencing defects either way. A consumer evaluating a
  bounded span no longer refuses a product for a seam its stencil cannot
  reach, and cannot silently accept one whose stencil reaches it. The
  `inspect` CLI gains `--window FROM THROUGH`.
- `next_issue_due`: a network-free answer, over the same catalog the
  publication-status query uses, for when the next issue of a cataloged
  product line is nominally due, naming the ultra lines' observed and
  predicted halves. Schedules cited to the published IGS product
  descriptions in committed provenance; boundary behavior pinned across
  UTC midnight and a GPS week rollover. The scoreboard prints the next
  due issue beside the current lag.
- Oracle version pinning documented in `docs/oracle-version-pinning.md`
  with measured (not transcribed) cross-version deltas for the SciPy and
  NumPy reference stack, and a one-command reproduction from pinned
  environments.

(0.40.0's exact-cache single-flight coalescing and non-exhaustive
`MergeOptions` ship to every interface with this release.)

## [0.40.0] - 2026-08-21

### Added

- `ExactProductCache::open_single_flight`: concurrent requesters for one
  product identity coalesce onto a single download. A waiter observes the
  owner's in-flight marker and blocks, bounded, on the committed entry
  instead of re-downloading - the answer to the alias-prone ultra-target
  pairs that previously had to stay serial. Ownership is a random
  128-bit token (PID and wall clocks are diagnostic only, so containers
  sharing a cache directory cannot be confused); liveness is append-only
  heartbeat growth judged on each waiter's own monotonic clock; takeover
  re-verifies the marker snapshot under the existing transition lock and
  claims by exclusive creation; a slow live owner yields a bounded
  `SingleFlightTimeout`, never a second download. The sidecar is
  schema-v3-compatible - commit encoding and `current.json` are
  byte-unchanged - and the mixed-version matrix is documented in
  `docs/exact-cache-single-flight.md`. Nine new failpoint boundaries
  carry process-kill tests; real-process integration tests cover
  waiter-observes-commit-without-downloading, SIGKILLed-owner takeover
  with exactly one commit, and live-owner timeout.

### Changed

- **Breaking**: `MergeOptions` is `#[non_exhaustive]`. It gains a field
  whenever the merge learns a new policy - 0.37.0 alone added two - and
  each addition was source-breaking for every downstream exhaustive
  literal. Construct by mutating `MergeOptions::default()`; future
  options then arrive without breakage.

## [0.39.1] - 2026-08-11

### Fixed

- DTED terrain lookups now compute the grid cell and intra-cell fraction
  in exact integer arithmetic. The 1-arc-second scaling is a roughly
  65-bit product, so the binary64 multiply rounded away the low fraction
  bits - up to 4096 ULP at representative CONUS coordinates - and, at a
  posting boundary, could round a coordinate strictly below a posting
  onto the exact integer, flipping the lookup into the next cell's
  stencil with fraction 0.0. The offset's integer significand is now
  multiplied by postings-per-degree before the power-of-two division,
  with Euclidean flooring for negative offsets and correctly rounded
  dyadic-to-binary64 conversion. All three lookup paths (bilinear DTED,
  bilinear mmap store, nearest-posting) share the one helper; the
  nearest-posting ties-to-even policy is unchanged, now computed on the
  exact remainder. Dyadic-exact coordinates are byte-identical before
  and after.

## [0.39.0] - 2026-08-10

### Added

- Attested opens for both mapped artifact readers:
  `MmapTerrain::from_path_attested` / `from_vec_attested` and
  `MmapPreciseEphemerisInterpolant::from_path_attested` /
  `from_vec_attested`, taking a caller-attested content checksum in place
  of the O(payload) hash pass the verified constructors perform. 0.38
  mapped the file but still hashed every payload byte at open (~90 s cold
  / ~47 s warm on a ~34 GB store, measured downstream); a caller who
  already holds a trustworthy measurement - fs-verity, a signed manifest,
  a content-addressed store - can now hand it over instead.

  The handle carries its digest provenance (`DigestProvenance::Verified`
  vs `Attested`) everywhere the digest appears, so an attested handle can
  never masquerade as a verified one. `checksum64()` on an attested
  handle returns the claim without hashing. `verify()` escalates to the
  full hash pass on demand and flips provenance on success. Everything
  O(header) and O(index) stays unconditional; the interpolant's attested
  open cross-checks the claim against the header's declared checksum in
  O(8) and fails closed with `AttestedChecksumMismatch` - a wrong digest
  for the file is caught without hashing a byte. The terrain header
  carries no file-level checksum, so its claim is recorded as-is and
  checked only by `verify()`.

## [0.38.0] - 2026-08-09

### Added

- `mmap` feature (off by default). With it enabled, `MmapTerrain::from_path`
  and `MmapPreciseEphemerisInterpolant::from_path` memory-map the file
  read-only and the reader owns the mapping, instead of reading the whole
  artifact into process memory. The entry point is unchanged, so every
  existing caller benefits without migrating to a new constructor.

  The copy avoided is the smaller half. A mapping is demand-paged, so a
  reader that queries a geographically local region faults in the pages
  covering those tiles and never touches the rest; construction parses only
  the header, datum tag, and index. That is the difference between opening a
  30+ GB terrain store and being unable to start. Measured on the committed
  fixture: a mapped open allocates ~1 KB regardless of artifact size, where
  the copying open allocates the artifact.

  Neither reader becomes self-referential and no `unsafe` appears at any
  interface boundary. Bytes live in a new `ArtifactBytes` enum
  (`Borrowed` / `Owned` / `Mapped`) and every lookup derives its span on
  demand. The interpolant's mapped parse uses offset-backed arrays rather
  than the borrowed `&[f64]` arrays its borrowed path uses, so the promotion
  to an owning reader is expressible in safe code.

- `MmapTerrain::is_memory_mapped` and
  `MmapPreciseEphemerisInterpolant::is_memory_mapped`, so a caller or a test
  can assert that a path open actually mapped rather than read. A change that
  quietly relocated the copy would otherwise be indistinguishable from a fix.

## [0.37.0] - 2026-08-09

### Added

- `check_continuity` attests that a precise-ephemeris sample series is
  physically continuous, or reports each violation with the epochs, the
  interval, and the magnitude that exceeded its bound. Two checks with
  different jobs: a speed gate whose bound is a true physical upper bound
  for the orbit class (`sqrt(mu/a_min) + omega_e*r_max`), so it cannot
  false-positive and catches gross corruption; and a hold-out
  interpolation residual evaluated through the product's own Lagrange
  substrate, which supplies the sensitivity. On a real GFZ ultra product
  earth-fixed chord speeds run 2757-3187 m/s against a ~6 km/s class
  bound, leaving hundreds of kilometres of displacement undetectable per
  epoch pair, so a speed gate alone cannot see a metre-scale splice; the
  residual check resolves a 5 m splice against a 1 m tolerance. Ordering
  is the library's responsibility - input is sorted internally, so a
  shuffled sequence and a sorted one produce identical reports - and
  after duplicate epochs are split out as their own defect class, a zero
  or negative interval is unrepresentable in the comparison path.
- `MergeOptions::provenance` records per-epoch merge provenance as the
  merge decides: which contributor supplied each accepted cell, where
  selection changed and why, and what each contributor covered.
  `Summary` mode is bounded by the number of selection changes; `Full`
  adds one entry per accepted cell. `MergeReport::provenance` is an
  `Option` so "not requested" stays distinguishable from "one
  contributor". `CellSelection` records a combined value as combined
  rather than nominating a supplier: under `Mean` or `Median` the written
  value is a combination of the members, so no single contributor
  supplied it.
- `MergeOptions::verify_continuity` runs the continuity check over the
  merged product as a post-condition and attributes each violation to the
  contributors on both sides, distinguishing a splice across a
  contributor change from a discontinuity inside one contributor's arc.
  It reports without refusing: the merge still returns the product.

### Changed

- `MergeOptions` gains the `provenance` and `verify_continuity` fields.
  This is source-breaking for exhaustive struct literals; construction
  sites using `..MergeOptions::default()` are unaffected. Neither option
  changes the merged product - the SP3 output is byte-identical whether
  or not they are enabled, pinned by test.

## [0.36.3] - 2026-08-04

### Fixed

- `parse_archive_listing` no longer rejects an AIUB whole-tree CSV listing
  over a path containing spaces. `;` is the field delimiter, so a space is
  legal path content, and the live 426k-row listing carries unrelated
  objects (conference PDFs, tarballs) with spaces in their names; one such
  row rejected the entire listing, so `publication_status` for every CODE
  line followed the redirect and then died at the parser. The four-field
  structure remains the malformed-row signal; closed dialect detection is
  unchanged. Found by downstream 0.36.1 verification; the recorded fixture
  now ends with the verbatim offending row, and the full live listing
  (425,132 objects) is the reproduction.

## [0.36.2] - 2026-08-04

### Changed

- Version-alignment release; no engine changes. The Python interface's
  0.36.2 adds anonymous-FTP transport for the `wum_nrt` line (parity with
  Elixir 0.36.1) and its release gate enforces exact engine-version
  lockstep.

## [0.36.1] - 2026-08-04

### Changed

- Version-alignment release; no engine changes. The Python and WASM
  interfaces enforce exact version lockstep with the engine crates, and
  their 0.36.1 patch (accepting the `WUM` publisher and `near_real_time`
  solution-class tokens in caller-built identities) requires matching
  engine versions on the registry.

## [0.36.0] - 2026-08-04

### Added

- Added an opt-in cross-line candidate walk for CODE's predicted ionosphere:
  `predicted_ionex_line_candidates` enumerates the `P1` and `P2` artifacts for
  one map date (both lines publish the same official filename for a map date,
  but the two-day line is produced a day earlier, so `P2` is routinely
  published while `P1` is still absent when CODE runs behind). Candidates
  never substitute a neighboring date's map, each keeps its own exact
  identity and cache path, and `resolve_first_published` preserves the line
  actually served in provenance. Single-line requests keep their fail-closed
  behavior.
- Added a publication-status API: `parse_archive_listing` (Apache and XHTML
  autoindexes, AIUB's whole-tree CSV, FTP `LIST` output - each verified live
  on 2026-08-04 and recorded as fixtures), `newest_published_product`,
  `published_issue_age_minutes`, and the bounded `publication_listing_urls`
  (current week directory plus previous, or one whole-tree listing). The
  scoreboard's one-call `publication_status` query reports the newest
  published issue and its lag behind nominal without fetching product bytes,
  and reports a transport failure as `Unreachable` rather than answering
  from an older directory - "nothing published" and "archive did not answer"
  are distinct outcomes.
- Added Wuhan University's hourly MGEX near-real-time orbit line
  (`wum_nrt`, `WUM0MGXNRT`, 02D span at 05M over anonymous FTP), verified
  against the live archive: the series begins 2024-07-03 (GPS week 2321) and
  the previously published `WUM0MGXULA` hourly line ended around GPS week
  2230 with a publication gap between; pre-NRT dates are refused. The line
  is not projected onto CDDIS (no exact mapping is cataloged). `ArchiveProtocol`
  gains `Ftp`, `SolutionClass` gains `NearRealTime`, and `ProductPublisher`
  gains `Whu`.
- The IGS combined ultra (`IGS0OPSULT`) and the Wuhan NRT line participate
  in the multi-center SP3 merge-consensus path behind their catalog entries,
  with exact-validation agency pins (`IGS`, `WHU`) and a four-center
  merge-input identity test alongside ESA/GFZ.
- Documented the case for broadcast ephemerides as the acquisition
  resilience floor (`docs/broadcast-ephemeris-resilience-floor.md`), as a
  design issue without implementation.

## [0.35.1] - 2026-08-01

### Fixed

- The RTK double-difference row builder now rejects a rover position that
  overflows to infinity instead of panicking. The rover is formed as
  `base + baseline_m`, and the boundary check validated each operand
  separately, so two individually finite inputs near `f64::MAX` summed to an
  infinite position and tripped a debug assertion inside the internal `add3`
  primitive. The sum is now built with the checked helper and surfaces as a
  typed `InvalidInput { field: "rtk.rover_pos", kind: NonFinite }` through all
  three RTK paths. Physically realizable baselines are unaffected. Added the
  scheduled-fuzz crash artifact (run 30695232523) as a committed corpus seed.
- SP3 now rejects an epoch-record (`*`) seconds value its own field cannot
  re-emit. The writer renders the epoch instant through an `F11.8` field, so
  seconds carrying more precision silently shifted the epoch on re-encode
  (`0.0000009999` came back as `0.00000100`, moving the instant by ~0.1 ns).
  This completes the fixed-column re-emission rule already applied to record
  values and the header line-2 fields. Conforming products are unaffected:
  their epoch seconds already round-trip through the field exactly.
- The RTK row builder now validates a supplied receiver-antenna calibration.
  A non-finite `pco_neu_m` reached the PCO/NEU projection and tripped a debug
  assertion inside the vector primitives; it is now rejected by field as
  `InvalidInput { field: "rtk.receiver_antenna.{base,rover}.pco_neu_m" }`. An
  offset that is finite but large enough to overflow when its three basis
  components are summed is reported as `ReceiverAntenna(InvalidGeometry)` by
  the projection itself. Published antenna calibrations are unaffected: real
  PCOs are centimetre-scale.

### Testing

- `sp3_round_trip` now compares the product's public content - header, epoch
  instants, comments, and every epoch's satellite states - instead of asserting
  whole-struct equality against the pre-normalization product. `to_sp3_string`
  is a normalizing writer, and `Sp3` retains raw acquisition-validation
  provenance describing the *input* text, so `parse(write(x)) == x` was false by
  construction for malformed or sparse inputs and reported those as crashes.
  The content comparison still catches a writer that drops or mangles data.
- `fuzz_rtk` now exercises the receiver-antenna path. It previously passed
  `None` for the corrections at all three RTK entry points, so PCO/PCV
  projection was never fuzzed; the harness now supplies arbitrary base/rover
  calibrations on roughly half of inputs and keeps the `None` path covered.
- RINEX observation headers now reject a code list the fixed-column format
  cannot carry, instead of parsing into a product that cannot be serialized.
  Observation descriptors are `A3` fields in `SYS / # / OBS TYPES`,
  `# / TYPES OF OBSERV`, `SYS / SCALE FACTOR`, and `SYS / PHASE SHIFT`
  (RINEX 2.11 section 5.1, RINEX 3.05/4.02 section 5.1), and the
  `SYS / # / OBS TYPES` count is an `I3` field. A wider descriptor or count is
  now a typed `Error::Parse` at the record that carries it. Previously such a
  header re-emitted a record that overran its 60-column content area, was
  truncated, and re-parsed with fewer codes than its own count declared, so
  `repair -> to_rinex_string -> parse` failed on input the parser had accepted.
- Added the exact 632-byte scheduled-fuzz crash artifact (run 30197879510) as a
  core regression and a committed fuzz-corpus seed.
- A `SYS / PHASE SHIFT` correction far from unity is now written in exponent
  form. Rust's `Display` never switches to an exponent, so a value such as
  `1e-300` rendered as 302 columns of plain decimal: the record was truncated
  into the content area, the correction collapsed to zero, and the satellite
  list disappeared. Corrections that fit the record's `F8.5` field keep their
  existing plain-decimal spelling byte for byte.
- A `SYS / PHASE SHIFT` satellite list that cannot be re-emitted inside the
  60-column content area is now a typed `Error::Parse`. Single-digit PRNs are
  read from two-column tokens (`G1`) but written into `1X,A3` fields, so a
  readable record was not always a writable one.

- SP3 now rejects a value its own fixed-column field cannot re-emit unchanged.
  Record positions, velocities, clocks, and clock rates are `F14.6` fields, and
  the header line-2 (`##`) seconds-of-week, epoch interval, and MJD fraction are
  `F15.8`, `F14.8`, and 13-decimal fields (SP3-c section 3, SP3-d Hilla 2016).
  A value carrying more precision than its field expresses, or one too wide for
  its columns, is now a typed `Error::Parse`. Previously such a value fit the
  columns but not the format, so `parse -> to_sp3_string -> parse` silently
  changed it (`36.019431257` km came back as `36.019431`). Conforming files are
  unaffected: their values already round-trip through their own fields exactly.

### Documentation

- Pinned the SP3 serialization contract for unrepresentable satellites:
  `Sp3::skipped_records` counts entries the input text carried but the product
  cannot represent (an extended GLONASS slot such as `R28` beyond the engine's
  PRN cap). They are deliberately dropped rather than aborting the parse, so
  nothing of them reaches the writer and a re-encoded product always re-parses
  with no skips. The `sp3_round_trip` fuzz target asserted that the two counts
  matched, which no correct implementation can satisfy for such a file; it now
  asserts the re-encode reports zero skips - stricter, since the writer must
  never emit a record the parser cannot represent - while still comparing every
  other field. Added a core regression and two committed fuzz-corpus seeds from
  scheduled run 30262991024. No parser, writer, or numerical behavior changed.

### Compatibility

- Parser compatibility patch. Conforming RINEX 2/3/4 observation files are
  unaffected: their descriptors already fit the `A3` code fields and their
  per-system counts the `I3` field. No public API, numerical kernel, or output
  formatting changed.

## [0.35.0] - 2026-07-24

### Fixed

- Observation QC no longer panics when a successfully parsed RINEX OBS product
  carries `INTERVAL = 0`. RINEX 2.11 section 5.3 and RINEX 3.05/4.02 section
  6.5 permit zero, a blank field, or an omitted optional record when metadata is
  unknown. Blank `INTERVAL` fields are now parsed as absent; zero is retained
  as unavailable metadata and reported as informational `OBS-H19`.
- An unavailable source interval is never used as cadence. QC instead labels a
  cadence inferred from the actual epoch grid as `Inferred`, or reports
  `Unresolved` and skips interval-dependent gap calculations. Negative or
  caller-constructed non-finite source intervals produce error `OBS-H20`.
  Explicit zero, negative, or non-finite caller overrides continue to return
  `InvalidInterval`, and interval repair remains opt-in.
- Gap accounting now saturates instead of overflowing for an extremely small
  positive caller interval, ignores non-finite public-structure epoch deltas,
  and never rounds a sub-millisecond inferred interval down to zero.
- Added the exact 583-byte scheduled-fuzz crash artifact, its human-readable
  reduction, core and CLI regressions, and a committed fuzz-corpus seed. The
  scheduled workflow timeout is now 90 minutes: its 31 two-minute target
  budgets alone require 62 minutes before runner setup and compilation.

### Compatibility

- Parser and QC compatibility patch. `Finding` gains the additive,
  non-exhaustive `ObsIntervalUnavailable` and `ObsInvalidInterval` variants.
  Existing positive source intervals and explicit positive overrides behave as
  before. Solver, orbit, propagation, positioning, frame, timing, and other
  numerical kernels are unchanged.

## [0.34.0] - 2026-07-21

### Added

- Added `supported_samples`, a date- and issue-aware catalog query for the
  officially evidenced cadences of one product. The same gate now rejects a
  syntactically plausible but unpublished cadence before filename, URL,
  identity, or cache-key derivation.
- Added the product- and issue-aware `sp3_content_start_convention` catalog
  query and `Sp3ContentStartConvention`. The result states whether an exact
  SP3 product starts at its filename epoch or one day earlier and exposes the
  corresponding whole-second offset. Invalid issues and issues on product
  lines that do not publish them are rejected.

### Fixed

- Exact SP3 validation now recognizes the terminal record as a complete logical
  record: `EOF` in columns 1-3 followed only by ASCII-space padding, bounded by
  Sidereon's 80-column interoperability policy. This accepts both bare records
  and the padded records published by the audited ESA and GFZ product lines,
  with LF, CRLF, or no final line separator. The previous whole-line equality
  check falsely reported `MissingEof` for those valid public products.
- Malformed EOF-like records now report `MalformedEofRecord` rather than being
  misdiagnosed as absent. Missing markers, `EOFX`, tab padding, padding beyond
  the policy width, leading whitespace, lone-CR framing, premature markers, and
  nonblank data after a valid marker still fail closed. Empty and ASCII-space-
  only records after the marker remain an explicitly documented Sidereon
  tolerance.
- Exact requests derived from historical GFZ ultra-rapid identities now
  distinguish the epoch encoded by the official filename from the product's
  first content epoch. Products through 2022-09-06 require the archive-observed
  one-day offset; the non-monotonic 2022-09-07/08 transition is cataloged per
  issue, and products from 2022-09-09 remain aligned. Declared-start, line-2,
  first-epoch, cadence, grid, and span checks remain strict, and callers cannot
  override the cataloged offset.
- Ultra-rapid location candidates now contain only dated span/cadence variants
  evidenced for the exact center, date, and issue. This removes speculative
  cross-cadence and alternate-span URLs; the documented GFZ `0000` overlap on
  2021-05-15 remains. CODE's moving latest-product snapshot is excluded because
  it is not the dated one-day exact product. All caller-built product
  identities now require the cataloged span, not merely a
  syntactically valid span embedded in a matching filename.
- Hardened auxiliary gzip ingestion for the Rust Bias-SINEX/CODE DCB path
  loaders and the validation scoreboard. They now decode every RFC 1952 member,
  accept optional header fields up to the archive limit, validate FHCRC plus
  every member CRC32/ISIZE and trailer, and reject truncation or trailing data.
  Local loaders enforce explicit 64 MiB archive and 500 MiB product limits.
  Scoreboard downloads use one authoritative bounded GET: final 404/410 remains
  ordinary publication absence, while transport and 5xx retries start with a
  fresh process and buffer so partial attempts cannot contaminate a success.
  Its curl status is carried in a dedicated terminal frame, and publication
  absence is authorized only by curl's HTTP-failure exit plus 404/410; a
  truncated transfer whose partial body ends in those digits remains a
  transport failure.

### Compatibility

- Parser, catalog, and transport compatibility only. No orbit, propagation,
  merge, positioning, solver, frame, timing, or other numerical calculation
  changed.
  All language interfaces inherit the same core behavior; each interface
  carries its own exact-parser regression, and the acquisition-owning
  interfaces also test the complete acquisition path. The GFZ correction
  changes only which cataloged start instant exact validation requires.
- This is a minor release because the new catalog query and enum are public and
  because exact validation now applies newly cataloged historical GFZ
  semantics. `ultra_sp3_locations` can return fewer candidates because
  unsupported alternate spans/cadences and CODE's non-exact moving snapshot
  are no longer represented as dated products. Caller-built identities with a
  noncatalog span now fail validation. Existing SP3 parsing and all numerical
  APIs remain compatible.
- Gzip changes affect transport integrity and resource limits only; no product
  parser, orbit, positioning, merge, or other numerical calculation changed.

## [0.33.1] - 2026-07-20

### Fixed

- Included the repository MIT license, the intact IERS Conventions derived-work
  notice, and the applicable ERFA and RTKLIB notices in the crates.io source
  package. The tide source now points directly to the packaged IERS terms.
- Renamed the private Rust translations of the IERS/SOFA companion routines and
  added the license-required statement that the derived work is not distributed
  or endorsed by the IERS Conventions Center.

### Compatibility

- Packaging, licensing documentation, and private source identifiers only;
  public APIs and numerical behavior are unchanged from 0.33.0.

## [0.33.0] - 2026-07-20

### Added

- Added IGS combined final-SP3 catalog support with date-aware official names:
  legacy `igs<week><day>.sp3` identities from GPS week 0730 through 2237 and
  `IGS0OPSFIN_<epoch>_01D_15M_ORB.SP3` identities from week 2238 onward.
  Historical CDDIS locations use `.Z`; current CDDIS and direct-BKG locations
  use `.gz`.
- Added `product_solution_class` so callers can distinguish IGS final SP3 from
  IGS broadcast navigation without changing the legacy center-only query.
- Added `default_sample_for_date` for product lines whose published sampling
  interval changed over time. GFZ rapid SP3 resolves to `15M` through 2021 day
  137 and `05M` from day 138.
- Added verified series floors for ESA final SP3/clock, GFZ rapid SP3/clock,
  and IGS, CODE, ESA, and GFZ ultra-rapid SP3 products. Ultra issue lookback
  stops at the applicable floor instead of emitting a previous-day identity.
- Made omitted ultra-rapid SP3 cadence issue-aware. ESA uses `15M` through the
  2025-02-02 0600 issue and `05M` from 1200; GFZ uses `15M` through 2021-05-15
  and `05M` from 2021-05-16. Candidate order follows the issue-era default.
- Added `ExactSp3Request`, `parse_exact_sp3`, and `validate_exact_sp3`. Exact
  validation binds the line-1 start/count, line-2 GPS-week/seconds-of-week/MJD
  start metadata, mandatory header/EOF records, producing agency, complete
  per-epoch satellite record sequences, finite positive cadence, parsed regular
  epoch grid, requested cadence, requested span, and optional format revision.
  It accepts both the half-open and inclusive regular-grid representations of
  an exact span.

### Fixed

- RINEX observation repair now canonicalizes malformed non-ASCII and control
  characters before fixed-column header parsing while preserving byte offsets,
  so repaired output remains printable, parseable, and byte-idempotent.
- CODE rapid and final catalog entries now use AIUB's current HTTPS download
  service with product-specific routes for MGEX final SP3/clock, operational
  final IONEX, and rapid IONEX; the already-correct ultra-rapid SP3 route is
  preserved. Historical `cod` requests are rejected until their distinct
  short-name identities are modeled instead of being assigned current long
  filenames.
- Caller-built identities for unsupported center/product combinations now fail
  before URL derivation or acquisition.
- IGS combined final-SP3 requests now reject dates before the official start at
  GPS week 0730 (1994-01-02), and legacy CDDIS product paths zero-pad the GPS
  week directory to four digits.
- CDDIS location derivation rejects pre-week-2238 long-name SP3 and IONEX
  identities while retaining the verified IGS final short-name `.Z` series.
  It also refuses to substitute a different CDDIS product for ESA's exact
  `ESA0MGNFIN` final-SP3 identity.
- Corrected the current GFZ rapid-SP3 catalog default to `05M`. Date-derived
  requests preserve the historical `15M` default through 2021 day 137 and use
  the published `05M` convention from day 138, including current products.
- Exact-SP3 candidate selection now advances only after ordinary publication
  absence. Parse, digest, identity, cadence, grid, and span failures remain
  terminal and preserve the first integrity error rather than accepting a
  later candidate. Candidate product codes and SP3 producing-agency fields are
  bound to the selected public product family.
- SP3 serialization now pads the mandatory header comment section to four
  records without adding semantic comments. Exact validation also enforces the
  line-3 satellite count, per-epoch record order, and P/V pairing required for
  velocity products.

### Compatibility

- Existing IGS broadcast-navigation derivation and
  `AnalysisCenter::solution_class()` are unchanged. The product-aware query and
  exact-SP3 validator and date-aware default-sample query are additive;
  `ArchiveCompression::UnixCompress` adds a public enum variant, and the
  catalog and scoreboard errors add typed public variants. The legacy
  date-free `default_sample` now reports GFZ's current `05M` rapid-SP3 cadence;
  dated default derivation remains `15M` for historical products through 2021
  day 137. For issue-based products, the date-only default represents the 0000
  issue while product construction uses the actual issue. Invalid identities,
  pre-series dates, unsupported combinations, unmodeled historical CODE
  products, and integrity-invalid exact SP3 content now fail earlier.
  Serialized SP3 text with fewer than four semantic comments gains blank
  mandatory comment records; blank structural padding is no longer surfaced as
  semantic text in `Sp3::comments`.
  These public additions and stricter semantics require a minor `0.33.0`
  release rather than a patch.

## [0.32.0] - 2026-07-18

### Added

- Added deterministic `parse_navcen_at` and `merge_navcen_at` APIs for
  evaluating NAVCEN operational usability at an explicit UTC instant. Returned
  assessments preserve NANU type, subject, raw Outage Start text, the evaluation
  instant, and parsed/unparseable/not-applicable timing provenance.

### Fixed

- Active bounded forecast NANUs now affect the time-aware path only during
  their validated half-open UTC interval. Future forecasts no longer disable a
  satellite early, completed temporary outages no longer remain active, and an
  incomplete interval remains usable with explicit ambiguity provenance.

### Compatibility

- Existing `parse_navcen` and `merge_navcen` signatures and clock-free behavior
  are unchanged. Callers making operational decisions should migrate to the
  explicit-time APIs; see `NAVCEN_TIME_SEMANTICS.md`. The time-aware path also
  recognizes active `UNUSUFN` notices as immediately unusable; the legacy path's
  pre-existing omission of that code is intentionally preserved.

## [0.31.2] - 2026-07-16

### Fixed

- Included the public merged-SP3 v1 golden fixture inside the published
  `sidereon-core` crate so its integration tests compile from an isolated
  crates.io source archive.

## [0.31.1] - 2026-07-16

### Fixed

- Canonical merged-SP3 identities now normalize accepted negative-zero
  tolerances to positive zero, matching merge execution semantics.
- Merge execution and provenance identity validation now reject the same empty
  system filters and incomplete asserted frame-label sets.
- Added literal cross-interface v1 golden vectors covering the complete merge
  policy, canonical contributor ordering, precedence ordering, and malformed
  inputs.

## [0.31.0] - 2026-07-16

### Added

- Added `Sp3ArtifactIdentity` and `Sp3MergeInputIdentity`, a versioned,
  order-independent stable identity for the complete set of exact SP3
  artifacts and merge controls. The canonical identity binds requested and
  resolved product identities, distributor, product and archive digests and
  lengths, compression, and every merge option while excluding acquisition
  observations, URLs, credentials, and local paths. Mean/median contributor
  enumeration is canonicalized; precedence contributor order is bound as an
  effective policy control.
- Added exact-identity distribution-location derivation so alternate cataloged
  SP3 duration and sampling candidates retain their declared identity instead
  of being reconstructed as the catalog default.

### Compatibility

- This release is additive. Existing SP3 merge and product-location APIs retain
  their prior signatures; exact provenance identity construction is opt-in.

## [0.30.0] - 2026-07-16

### Added

- Added the schema-v3 exact-product cache protocol. Commit records bind the
  complete product identity, explicit distribution source, and SHA-256 digest
  and length of validated product, distributor archive, and provenance bytes.
- Added native Linux/macOS `ExactProductCache` transactions with bounded
  cross-process locking, cryptorandom immutable entries, synchronized files and
  directories, one atomic commit marker, unlocked-reader refresh retry, and
  lock-scoped abandoned-entry cleanup. The ergonomic `sidereon` crate re-exports
  the same API.
- Added `analysis_center` and `format_version` to `ProductIdentity`. Canonical
  identity bytes and portable keys now include every exact identity field.

### Compatibility

- This is a source-breaking identity-model correction: Rust struct literals
  must provide the two new fields. The minor version advances because
  `ProductIdentity` is externally constructible; `cargo-semver-checks` confirms
  a patch release would be incorrect.

### Fixed

- Prevented finite even-count robust-median inputs near `f64` limits from
  overflowing their central-pair addition to infinity.

## [0.29.2] - 2026-07-16

### Added

- Added `validate_exact_product_set`, a sans-IO completion gate for workflows
  that require several exact products. It rejects empty declarations,
  duplicate expected or available identities, missing identities, and
  undeclared identities before dependent processing begins.
- Exact-set comparison uses the complete distributor-independent identity, so
  same-filename products with different prediction tiers remain distinct.
  SP3 observed/predicted timing remains authoritative only through
  `Sp3::prediction_summary()` and its record-level flags.

## [0.29.1] - 2026-07-15

### Fixed

- CODE predicted IONEX direct locations now use AIUB's supported HTTPS
  download endpoint and the exact `CODE/IONO/P1/<year>` or
  `CODE/IONO/P2/<year>` directory selected by the requested prediction tier.
- The directory year is derived from the resolved product identity, including
  a P2 request whose one-day offset crosses into a new year. Exact filenames,
  prediction horizons, and distributor-independent cache identities are
  unchanged; no older date, alternate tier, or provider is substituted.

## [0.29.0] - 2026-07-15

### Added

- Added an exact public GNSS product identity model that keeps product family,
  publisher, solution class, campaign, issue, cadence, coverage date, official
  filename version, format, and prediction horizon separate from distribution.
- Added explicit direct-archive, NASA CDDIS/Earthdata, local-file, and in-memory
  distribution sources, including exact CDDIS SP3 and IONEX locations and
  deterministic source-specific cache paths.
- Added validated product requests and distribution locations so selecting a
  distributor cannot silently change the requested center, tier, issue,
  cadence, date, family, or official filename.

### Compatibility

- This release is additive. Existing product selection, URL generation, and
  cache behavior remain unchanged; the new exact-identity and distribution
  APIs are opt-in.

## [0.28.1] - 2026-07-15

### Fixed

- CODE ultra-rapid SP3 candidates now use AIUB's official HTTPS download
  endpoint while retaining the daily `0000` issue and the dated, alternate,
  and latest-alias filenames. The remaining AIUB-backed catalog entries were
  audited separately and remain unchanged because current and historical trees
  use mixed naming and availability conventions.
- The SP3 validation harness no longer treats access denial or transport
  failure as product absence. Candidate-URL 404/410 statuses are retained in
  reports, while transport failures preserve source, filename, URL, and the
  available HTTP status or network diagnostic.
- Sequential RTK solves with baseline process noise now enforce the exact
  symmetry of the information-form time update. The rank-3 correction is
  symmetric mathematically, but independently evaluated matrix triangles could
  accumulate phase-precision roundoff and destabilize long held-ambiguity arcs.

### Evaluation-bit stability

- Process-noise-enabled sequential RTK updates can move in their last bits when
  the two information-matrix triangles are averaged. The zero-process-noise
  path and public interfaces are unchanged.

## [0.28.0] - 2026-07-13

### Added

- Added deterministic contested-cell outlier rejection, per-cell precedence,
  mixed-cadence SP3 coverage merging, clock-outlier provenance, and an opt-in
  whole-satellite precedence mode.
- Added SP3 per-epoch observed/predicted metadata and the contiguous
  observed-through boundary derived from record flags.
- Added current and alternate ultra-rapid SP3 catalog locations for IGS, CODE,
  ESA, and GFZ.

## [0.27.1] - 2026-07-13

### Fixed

- `lambda_ils_search` now rejects ambiguity values outside the `i64` lattice
  domain before reduction and checks back-transformed candidates before integer
  conversion. Previously, an extreme finite input could saturate to
  `i64::MAX`, overflow canonical rescoring, and return `Ok` with non-finite
  scores and ratio.

### Evaluation-bit stability

- Calls whose ambiguity inputs and back-transformed candidates remain within
  the `i64` lattice domain retain the same LAMBDA arithmetic, candidate
  ordering, scores, and fix decisions. Inputs or internally produced candidates
  outside that domain now return the existing typed `InvalidInput` error.

## [0.27.0] - 2026-07-12

### Added

- `GeoidGrid::from_proj_egm96_gtx` loads the public PROJ EGM96 15-arcminute
  GTX grid. `GeoidGrid::undulation_proj_rad` reproduces PROJ 9.3.0's radian
  indexing and interpolation order with an explicit
  `ProjVgridshiftArithmetic` selection for contracted or separately rounded
  multiply-add evaluation. The new path is pinned against 13,051 public-grid
  reference points; invalid coordinates return typed errors.

### Evaluation-bit stability

- Existing geoid loaders and `GeoidGrid::undulation_rad` retain their previous
  evaluation bits. The new PROJ path has no implicit platform-dependent
  default: callers select fused or separately rounded arithmetic explicitly.

## [0.26.1]

### Security and availability

- RINEX observation parsing now rejects epoch record counts that exceed the
  format's three-character `I3` maximum of 999 before reserving record storage.
  A malformed RINEX 2 epoch could previously pass an effectively unbounded
  count to `Vec::with_capacity`, allowing memory exhaustion or a process abort
  instead of a parse error. `sidereon-core` releases 0.11.1 through 0.26.0,
  inclusive, are affected.

### Evaluation-bit stability

- Valid RINEX inputs retain identical parsed values and evaluation bits. Only
  malformed epochs with an over-width record count change behavior: they now
  return a deterministic parse error before allocation.

## [0.26.0]

### Breaking

- Removed the generic sequential-RTK innovation-screen API and its result
  fields: `InnovationScreenOpts`, `InnovationScreen`,
  `UpdateOpts::innovation_screen`, `EpochUpdate::innovation_screen`,
  `RtkArcEpochSolution::innovation_screen`,
  `ScreenKind::RtkSequentialInnovation`, and
  `ResidualNormRecipe::RtkInverseVarianceInnovation`. The removed mechanism
  divided residuals by measurement variance, omitted predicted-state
  covariance and shared-reference correlation during classification, and
  treated carrier-phase events as ordinary row outliers. Sequential RTK now
  consistently assimilates the complete correlated double-difference block;
  carrier anomalies remain handled by the causal slip/arc lifecycle.
- Removing those variants also compacts the enums' compiler-assigned numeric
  discriminants. Code that casts them to integers will observe
  `ResidualNormRecipe::RtkInverseSigmaResidual` changing from 1 to 0,
  `ResidualNormRecipe::PppInverseSigmaMagnitude` from 2 to 1, and
  `ScreenKind::PppFloatLeaveOneOut` from 3 to 2. These enums do not promise a
  stable numeric representation; 0.26.0 does not preserve unused discriminant
  holes.

### Fixed

- Ionospheric pierce-point evaluation now remains finite when floating-point
  rounding puts a valid near-polar latitude sine just outside `[-1, 1]`.
- The locked dependency graph now uses `crossbeam-epoch` 0.9.20, which fixes
  RUSTSEC-2026-0204.

### Evaluation-bit stability

- The near-polar TEC correction intentionally changes affected pierce-point
  results from non-finite latitude/longitude values to finite values. Existing
  in-range TEC evaluations require no golden re-pin, and the ordinary
  sequential-RTK path remains bit-identical to its former no-screen execution.

## [0.25.0]

### Added

- The `sidereon` facade root now exposes the CRINEX encoder convenience
  `encode_crinex`, matching the existing lower core module and bindings.
- The `sidereon` facade root now re-exports existing Sun/Moon azimuth/elevation
  helpers, geodetic/topocentric transform helpers, TLE look-angle and
  ground-track helpers, and Doppler shift helpers that were already available
  through lower core modules.

## [0.24.0]

### Changed

- ARAIM now returns an unavailable `AraimResult` with `available: false` when
  geometry cannot support the integrity budget, instead of returning
  `UnmonitorableFaultMass`.

## [0.23.0]

### Added

- RTCM 3 broadcast ephemeris decode/encode and solver conversion for Galileo
  1045/1046, BeiDou 1042, and QZSS 1044. Galileo 1046 is covered by a real HAS
  IDD capture propagated against a CNES/CLS ultra-rapid SP3 trim; BeiDou 1042,
  QZSS 1044, and Galileo F/NAV 1045 are covered by a real BKG BCEP capture
  propagated against matching CNES/CLS and QZSS ultra-rapid SP3 trims.
- Static PPP float and fixed solve configs now accept an optional
  `elevation_cutoff_deg`; when set, observations below the seed-position
  elevation cutoff are removed before ambiguity ids, residual rows, normal rows,
  and fixed ambiguity search are assembled. `None` preserves the existing
  observation set.
- optional tropospheric horizontal gradient estimation for static PPP (off by
  default).
- static multi-epoch positioning (`solve_static`) is now public, with
  covariance, leave-one-out redundancy diagnostics, and robust weighting.

- Static PPP float and fixed solutions add temporal-correlation covariance
  reporting: `temporal_position_covariance`,
  `temporal_position_covariance_scale_factor`, and `temporal_correlation`.

## [0.22.0]

### Added

- Static PPP float and fixed solutions expose posterior receiver-position
  covariance in ECEF and ENU coordinates through `PositionCovariance`, plus the
  raw posterior unit-variance factor and the applied covariance scale factor
  (which equals that factor). Every solver in the library now reports position
  covariance.
  The estimator pools lag-1 post-fit residual autocorrelation by satellite arc
  and observable, reports AR(1) effective sample count and decorrelation time,
  and keeps the existing posterior-scaled covariance fields unchanged.
- SP3 multi-center merge coordinate-label reconciliation: caller-asserted label
  equivalence and catalog Helmert reconciliation between known ITRF/IGS
  realizations, with merge-report audit fields for the selected method, affected
  records, published parameters, rates, provenance, and catalog direction.
  Strict label matching remains the default; unresolvable mismatches still fail.
- RTCM MSM stream-to-SPP conversion for live workflows: `RtcmSppEpochInputs` and
  `spp_inputs_from_rtcm_msm` assemble RTCM MSM observations into the same
  per-epoch solve input shape used by RINEX replay.
- Allocation-free warm hot path for high-rate serving:
  `emission_media_batch_at_j2000_s_into` writes the correction bundle into
  caller buffers, and `EmissionMediaReceiverContext` plus
  `emission_media_batch_at_j2000_s_with_receiver_context_into` cache the
  per-receiver setup so a staged, repeated single-call path allocates nothing.
  Results are bit-identical to the allocating form (gated across the fixture
  sweep). Staged precise-ephemeris interpolants gain `EphemerisSource` impls,
  with a bit-identity gate pinning the staged-interpolant SPP solve to the raw
  SP3 solve.

### Changed

- Static PPP eliminates per-epoch receiver clocks from the normal equations and
  back-substitutes them after solving the reduced static system, making
  day-length arcs tractable without changing the public clock output.
- Static PPP result covariance is multiplied by the posterior residual variance
  factor, with the unscaled formal covariance retained for callers.
- PPP GF/MW cycle-slip splitting confirms GF/MW-only events before creating new
  ambiguity states, while LLI and data-gap splits remain immediate.

### Breaking

- `FloatSolution` and `FixedSolution` in precise positioning gained required
  `position_covariance`, formal covariance, and posterior variance scale
  fields. Callers constructing these structs directly must populate them;
  callers only reading results are unaffected.

## [0.21.0]

### Added

- Loose GNSS/INS field-mode options: stationary ZUPT/ZARU pseudo-updates with
  a configurable accel and gyro magnitude window, wheeled-vehicle
  non-holonomic lateral and vertical velocity constraints, and
  per-fix-status GNSS covariance weighting for single, float, and fixed
  updates. The inertial filter config also accepts a fixed IMU-to-body
  direction-cosine matrix for callers that do not pre-rotate IMU samples.
- Standalone first-fix velocity matching helpers for GNSS outage spans,
  including `velocity_match_outage_to_state` for blending an outage segment to
  a caller-supplied post-update endpoint instead of only to the raw GNSS fix.
- RINEX-to-SPP assembly helpers that convert parsed observation epochs plus a
  broadcast or precise ephemeris context into per-epoch `SolveInputs`, with a
  serial batch solve convenience that preserves per-epoch solve errors.
- Static reference-station RINEX solve that composes code-DGNSS and carrier RTK
  modes, returning one station coordinate with covariance, fix status, and
  per-epoch diagnostics.

### Changed

- Fusion state checkpoints use codec version 4 and still read earlier v1-v3
  streams. Checkpoints now preserve the stationary-detector window plus the
  last stationary and non-holonomic pseudo-update epochs, so restored filters
  keep detector state and duplicate-update guards.
- `GnssFixMeasurement` now carries public `fix_status`; JSON
  `SerializableLooseMeasurement` defaults the field for older payloads.
- RTK `FloatBaselineSolution` and `FixedBaselineSolution` now expose
  `baseline_covariance_m2`; computing that covariance can surface
  `SingularGeometry` on degenerate final normal equations.
- Fusion RTS histories accept same-epoch predicted/updated checkpoints for
  measurement-only updates, synthesize an identity transition for those
  updates, and permit zero-duration smoothing transitions.
- Static reference-station selection now prefers fixed carrier RTK, then code
  DGNSS, then float carrier fallback; reports keep the fixed-solution
  measurement count, label fixed-mode failures correctly, carry typed
  per-mode errors, and format all-mode failures by mode instead of dumping
  debug structs.
- Code-DGNSS covariance now accounts for both rover and reference code noise,
  including the multi-epoch static reference-station path.
- Stationary ZUPT/ZARU and non-holonomic pseudo-updates no longer inherit GNSS
  IGG-III measurement reweighting or Yang prediction-adaptation settings.
- Tight-coupling range-rate gyro-bias rows now honor `imu_to_body_dcm` for
  non-identity IMU mounting.
- Real BKG/IGS-IP SSRA03IGS0 SSR integration fixture covering GPS, GLONASS,
  Galileo, and BeiDou RTCM SSR orbit, clock, and code-bias decode. The test
  validates IODE-matched GPS SSR-corrected broadcast satellite positions
  against the IGS ultra-rapid SP3 with a non-vacuous broadcast-only error
  margin.

## [0.20.0]

### Added

- RINEX RTK arc builders as library API: rover and base observations plus
  ephemeris and base coordinates in, double-differenced carrier-phase arcs
  built by the library, static float and wide-lane fixed baselines out with
  fix status. On the real WTZR/WTZZ station pair the fixed
  baseline lands within 2.8 mm of the published ITRF antenna-reference-point
  baseline (float: 8.3 mm).

- SSR and Galileo HAS corrections now drive the PPP solve: an SSR-corrected
  ephemeris provider applies orbit and clock corrections over broadcast
  ephemeris with strict IODE matching, update-interval staleness handling, and
  explicit antenna-phase-center versus center-of-mass reference handling; RTCM
  SSR code and phase biases for GPS and Galileo decode into the correction
  store and apply in the PPP measurement model. On the end-to-end fixture the
  SSR-corrected solve closes an 11.6 m broadcast-only error to below 0.1 mm
  against the SP3-backed reference.

## [0.19.0]

### Changed

- The fusion smoother's transition-combination step uses a specialized square
  matrix product, making fixed-interval smoothing tractable over histories
  recorded at inertial sample rates (found during the deep-urban field
  rematch).
- Correction to an earlier draft of this entry: a one-ULP evaluation shift on
  Earth-orientation-chain paths appeared mid-cycle from the tide-force wiring
  and was reversed by the station-displacement refactor before release. Net
  evaluation bits for these surfaces are UNCHANGED relative to 0.18.0.

### Added

- Station displacement corrections now have a public `tides` entry that accepts
  ITRF/ECEF or WGS84 geodetic station positions, UTC epochs, per-epoch IERS
  polar motion, and optional caller-supplied BLQ ocean-loading coefficients.
  The scalar and batch APIs return component-resolved ITRF/ECEF displacements
  for solid Earth tide, pole tide, and ocean tidal loading. BLQ parsing supports
  standard Bos-Scherneck/HARDISP six-row station blocks and reports typed errors
  for unsupported constituents.
- Solid Earth tide and solid Earth pole tide propagation forces. The solid
  Earth tide force ships the IERS 2010 Chapter 6 Step 1 frequency-independent
  anelastic Love-number `Cnm`/`Snm` corrections from Sun and Moon positions,
  including degree 3 terms and degree 4 `k+` terms from degree-2 tides. Step 2
  frequency-dependent constituent corrections are documented as a follow-up.
  The pole tide force uses polar-motion samples from a series-backed
  body-fixed provider and the IERS 2010 mean pole model, and both forces are
  opt-in builder components.

## [0.18.0]

### Added

- Loose GNSS updates can opt into IGG-III measurement variance inflation and a
  Yang two-segment prediction adaptive factor. The prediction factor is gated
  by a Jiang-Zhang Mahalanobis measurement-outlier check so measurement faults
  use measurement reweighting rather than innovation-driven covariance scaling.
- Fusion RTS fixed-interval smoothing over recorded error-state histories,
  with recorded forward-pass transitions, predicted and updated checkpoints,
  smoothed covariances, and loose/tight measurement-agnostic entry points.
- Simulator-backed field-behavior pins for loose fusion smoothing, outage
  coast, and low-satellite tight consistency.

## [0.17.0]

### Fixed

- Tight GNSS C1C and carrier-phase code-row prediction now uses the same
  measured-pseudorange transmit-time model as SPP, removing centimetre-level
  frozen-state residual differences from the prior observable transmit-time
  approximation.
- Sample-backed SP3 interpolation now reconstructs the whole-second node axis
  from the split epoch before reducing it to continuous J2000 seconds, with
  only an epoch-ULP bound for accepting whole-second candidates. An earlier
  attempt used an absolute snap that did not fire for real converted epochs,
  which land one `f64` ULP below affected record seconds; the record-epoch
  oracle now runs a conversion-path fixture on both construction paths.

### Changed

- IONEX slant-delay evaluation now reports out-of-coverage epochs and
  pierce-point latitude or longitude as typed errors by default instead of a
  silent hold. Callers can opt into the legacy hold behavior with
  `IonexCoveragePolicy::Hold`, which returns an explicit status marker, and the
  new batch result helper reports coverage per element.

## [0.16.1]

### Fixed

- SP3 interpolation on the parsed-product path now uses an exact parsed
  J2000-second epoch axis for record nodes, preventing one-second node bucketing
  errors at affected 45-minute cadence boundaries. Record-epoch positions and
  clocks are gated directly against public SP3 text records, including the
  cached batch path. (A clock quantization initially reported alongside this
  was traced to the reporting consumer's own time conversion, not to this
  library; the record-epoch clock oracle it prompted remains, at 5e-13 s
  against the file text.)

## [0.11.1]

### Added

- GNSS observation QC now computes teqc-style multipath (MP1/MP2 RMS) per
  satellite and per constellation with per-arc moving-average bias removal,
  matching teqc `+qc` to sub-micrometer on a real captured stream; a
  receiver clock-jump detector; and an aggregate per-constellation cycle-slip
  tally over the existing dual-frequency slip detector.
- QC report renderers: a fixed-width teqc-style text summary, an HTML summary,
  and JSON serialization of the full `ObservationQcReport`.
- RINEX 2.x observation-file ingest into the shared canonical observation IR,
  so RINEX 2 and CRINEX 1.0 archives parse and flow through QC and lint
  unchanged.
- Fuzz targets for the space-weather CSV/txt parser, the RINEX QC repair
  round-trip, and the EGM96 DTED grid parser.

### Changed

- Rust, Python, C, WASM, and Elixir interfaces expose the new QC surface
  (multipath, clock jumps, cycle-slip tally, and the report renderers)
  with uniform parity.

## [0.11.0]

### Added

- Core 6x6 orbit covariance transport with frame-labeled nodes,
  RTN acceleration process noise, caller-supplied transport segments,
  PSD-safe Log-Cholesky interpolation, covariance unit conversion helpers, and
  TCA Pc integration for propagated covariances.
- Space-weather ingestion: CSSI space-weather CSV and txt parsing with a
  time-indexed table, NRLMSISE-00 selection conventions (previous-day F10.7,
  81-day centered average, daily and 3-hourly Ap), format-faithful
  serializers, a CelesTrak data-catalog entry, and a `SpaceWeatherSource`
  hook feeding atmospheric drag and orbital-decay estimation.
- GNSS observation quality control: per-satellite and per-signal completeness,
  gap, and signal-strength summaries over RINEX observation data, plus a RINEX
  lint and repair pass with typed finding codes, cross-checked against an
  independent extraction oracle on real IGS stations. (Multipath, cycle-slip,
  and clock-jump metrics land in 0.11.1.)
- RTCM MSM carrier-phase lock-time indicator to RINEX loss-of-lock indicator
  derivation: DF402 and DF407 lock-time bucket tables, conservative
  decrease detection with same-bucket ambiguity handling, half-cycle ambiguity
  mapping, and a per-signal lock-time tracker, cross-checked against an
  independent RTKLIB convbin decode of a real MSM stream. Adds RTCM stream
  decode diagnostics and typed truncation classification.
- NTRIP sans-IO protocol: a caster handshake and streaming state machine
  (request builder, response classification, chunked and sourcetable decoding,
  GGA position feed policy) with no transport in the core, plus idiomatic
  streaming clients in the Python and Elixir interfaces.
- NMEA 0183 support: a forgiving sentence parser, an epoch accumulator, and a
  GGA writer over a format-agnostic representation.
- CNAV and RINEX-4 broadcast evaluation: CNAV and CNAV2 clock and orbit
  parameters, user range accuracy and inter-signal correction accessors,
  mixed broadcast-store selection, and a lenient navigation parse that reports
  skipped blocks.
- TLE mean-element fitting: fit SGP4 elements to a span of states on the
  shared trust-region least-squares engine, with observability diagnostics,
  epoch selection, and observation weighting. NDM epochs gain femtosecond
  precision through a single shared parser.
- Geoid undulation evaluation matching PROJ on the EGM96 15-arcminute grid:
  node-registered bilinear interpolation with antimeridian and pole handling,
  batch lookup, and orthometric to ellipsoidal height conversion, pinned to
  PROJ-computed reference values.

### Changed

- `Covariance6Error` now includes interpolation-specific
  `NotFactorizable` and `InvalidInterpolationParameter` variants, and
  propagated-covariance TCA option structs now carry `process_noise`.
  Exhaustive matches and struct literals may need source updates.
- Rust, Python, C, WASM, and Elixir interfaces expose uniform capability parity
  for the 0.11.0 surface.

## [0.10.1]

### Fixed

- DTED ten-degree block directories now follow the layout production stores
  use: the hemisphere letter comes from the tile index and the magnitude is the
  truncated absolute value, so `n36_w107` buckets under `n30_w100/`,
  `n32_w118` under `n30_w110/`, and `s01_w001` under `s00_w000/` (with `n00`
  and `s00` kept distinct). The previous flooring convention mis-bucketed
  every western and southern index that was not an exact multiple of ten,
  making caches invisible to existing tile stores. Tile naming itself is
  unchanged. A cache directory populated by 0.10.0 can be migrated by moving
  the affected tiles into the corrected block directories, or simply
  regenerated. The derivation is validated against an observed 888-tile
  listing captured from a production-style store.
- `PreciseEphemerisSamples::from_samples` now rejects a sample epoch whose
  derived J2000 seconds is not finite and a finite clock offset that overflows
  to a non-finite value in native microseconds, instead of poisoning the
  interpolation node axis or emitting non-finite clock values downstream.

## [0.10.0]

### Added

- Astrodynamics coverage for anomaly conversions, analytic Kepler propagation,
  equinoctial and modified-equinoctial elements, solar beta angle,
  RIC/RTN/LVLH relative frames, Clohessy-Wiltshire motion, angular separation,
  position angle, general body observation, almanac events, atmospheric drag
  force, orbital decay, source-agnostic ephemeris grid sampling, and
  terrain/DTED lookup.
- GNSS DCB/OSB bias ingestion, SBAS augmentation with decode and corrected SPP,
  SSR/HAS real-time corrections, and robust SPP with a fault
  detection/exclusion driver.
- Cache-first data acquisition support for SP3, IONEX, CLK, NAV, and SRTM
  terrain to DTED products, using a single sans-IO core catalog and bit-exact
  hgt to DTED conversion.

### Changed

- Rust, Python, C, WASM, and Elixir interfaces now expose uniform capability
  parity for the 0.10.0 surface.
- GNSS constellation labels now use conventional styling: GPS, GLONASS, Galileo,
  BeiDou, QZSS, NavIC, and SBAS.

## [0.1.0]

Initial release.

- SGP4/SDP4 propagation (Vallado port), TLE and OMM (KVN/XML/JSON) parsing.
- Coordinate and time transforms (TEME/GCRS/ITRS/geodetic/topocentric, leap
  seconds, UT1), Sun/Moon ephemeris, solid-earth tides.
- RINEX navigation/observation/clock and CRINEX parsing, SP3 load and merge,
  ANTEX antenna corrections, broadcast and precise ephemeris evaluation.
- GNSS positioning: SPP (with robust estimation), RTK (LAMBDA ambiguity
  resolution, dual-frequency, multi-GNSS), and static PPP.
- Carrier-phase combinations and cycle-slip detection, DOP, visibility and
  pass prediction, velocity/Doppler, and observation quality weighting.
- Conjunction assessment and collision probability.
