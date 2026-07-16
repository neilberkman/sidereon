# Changelog

All notable changes to `sidereon-core` are documented here.

## [Unreleased]

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
