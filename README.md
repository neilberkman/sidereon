# sidereon

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21174996-1682D4.svg)](https://doi.org/10.5281/zenodo.21174996)

GNSS positioning and astrodynamics in Rust, with first-class interfaces in [Python](https://github.com/neilberkman/sidereon-python), [C](https://github.com/neilberkman/sidereon-c), [WebAssembly](https://github.com/neilberkman/sidereon-wasm), and [Elixir](https://github.com/neilberkman/sidereon-ex). Reference-validated, and bit-exact to public oracles where it counts.

sidereon is one engine: a Rust core for satellite orbit propagation, GNSS positioning, time and frame transforms, atmosphere models, and the standard exchange formats, exposed through idiomatic interfaces in five languages so the same validated math is reachable wherever you work.

**Live demo: [sidereon.dev](https://sidereon.dev)**: a real-time satellite tracker (globe, ground tracks, coverage, conjunction screening, orbit determination) computed in the browser via the WebAssembly build.

## Capabilities

- **Orbit propagation:** SGP4/SDP4 from TLE/OMM, numerical propagation with a composable force model (spherical-harmonic geopotential to selectable degree and order, solid Earth and pole tides per the IERS conventions, Sun/Moon third-body, solar radiation pressure with conical shadow, Earth albedo and infrared radiation pressure, relativistic correction, NRLMSISE-00 drag), decay/reentry prediction with a post-decay validity latch, Kepler propagation and anomaly conversion, batch/constellation propagation, ground tracks, passes, visibility, and coverage.
- **Orbit determination:** initial orbit determination (Gibbs, Herrick-Gibbs, Gauss angles-only), batch least-squares fit of the numerical propagator to precise ephemerides with a per-satellite RTN residual ledger, and covariance propagation.
- **GNSS positioning:** single-point (SPP), RINEX observation to SPP assembly and solve helpers, public multi-epoch `static_positioning` / `solve_static` solves with covariance, leave-one-out redundancy diagnostics, and robust weighting (including one-call reference-station static: rover and reference RINEX in, station coordinate with covariance out), Doppler velocity with clock drift, RTK float and fixed (LAMBDA) with baselines built straight from raw RINEX (verified to millimeters against a published ITRF station pair), PPP float and fixed with SSR or Galileo HAS corrections driving the solve over broadcast ephemeris, static PPP with temporal-correlation covariance (calibrated day-length bounds), optional elevation cutoff, and optional tropospheric gradient estimation, DGNSS, across GPS/GLONASS/Galileo/BeiDou/QZSS, with DOP (G/P/H/V/T).
- **Integrity and error bounds:** RAIM fault detection and exclusion, multi-constellation ARAIM (MHSS protection levels), SBAS protection levels (DO-229), classical reliability (per-observation minimal detectable bias, internal/external reliability), observability classification of every solve (rank, redundancy, conditioning), and covariance-derived error metrics (CEP, R95, drms, SEP, error ellipse) that report wide or flagged bounds for weak geometry rather than fabricated confidence, and uncertainty-aware geodesic geofencing (containment and crossing probabilities from a position covariance, with hysteresis).
- **GNSS corrections:** SBAS message decode and correction application, RTCM SSR and Galileo HAS orbit/clock/bias correction stores with explicit provider reference-point handling, NTRIP client stream handling, and differential code biases (DCB/OSB) from Bias-SINEX and CODE products.
- **Ephemeris and time:** broadcast and precise (SP3) ephemeris, RTCM 3 broadcast ephemeris decode for GPS (1019), GLONASS (1020), Galileo (1045/1046), BeiDou (1042), and QZSS (1044), each real-data validated, JPL SPK kernels, source-agnostic satellite state sampling across all three, batched multi-satellite interpolation, scale-aware time (UTC/TAI/TT/UT1/TDB/TCG/TCB and the GNSS system times) with leap-second handling and caller-updatable leap and UT1 tables, and Earth orientation (EOP).
- **Timing and clocks:** Allan-family stability analysis (ADEV/MDEV/HDEV/TDEV), power-law clock-noise identification with a five-coefficient fit (IEEE 1139), and clock comparison across products.
- **Estimation and detection:** a covariance-weighted track filter for position fixes (no IMU required: weak-geometry fixes with wide covariances cannot spike the track) with a fixed-interval RTS smoother, scalar Kalman and alpha-beta trackers, innovation gating (NIS), MAD statistics, CFAR detection thresholds, and source localization (ToA/TDOA) from arrival times at known sensors.
- **Geodesy and monitoring:** geodesic direct and inverse problems on the ellipsoid (Karney), an epoch-aware terrestrial reference frame catalog with published ITRF and ETRF Helmert parameter sets, station displacement corrections (solid Earth tide, pole tide, and ocean loading from caller-supplied BLQ coefficients), station velocity (MIDAS), trajectory fitting with seasonal terms and offsets, step detection, network motion fields with common-mode removal, and repeating-geometry (sidereal) filtering with coverage-aware templates.
- **GNSS/INS fusion:** field mode for real receivers (zero-velocity and zero-angular-rate updates, non-holonomic vehicle constraints, per-fix-status weighting, IMU-to-body mounting DCM, velocity matching across outages), plus ECEF strapdown mechanization with rigorous attitude integration, an error-state EKF (with a UKF option) using Joseph-form updates, loose and tight coupling (per-satellite pseudorange and range-rate measurements, valid from a single satellite), IGG-III loose updates (measurement reweighting and adaptive prediction scaling behind an outlier guard, from the published schemes), an RTS fixed-interval smoother over recorded histories, time synchronization with checkpointed late-measurement replay, a serializable filter state, and a deterministic IMU error simulator. Field behavior is pinned by simulator-backed tests: fused beats own GNSS under an outlier budget, outages coast within the IMU-grade bound, and sub-4-satellite windows stay covariance-consistent.
- **Geometry and events:** TEME/GCRS/ITRS/geodetic/topocentric transforms (IAU/IERS), a precise Earth-orientation rotation provider, look angles, eclipse, relative motion in RIC/RTN/LVLH frames with Clohessy-Wiltshire propagation, conjunction screening with collision probability (TCA/Pc), and classical and equinoctial element conversion.
- **Observation and almanac:** apparent places (geocentric and topocentric RA/Dec and az/el) for the Sun, Moon, and planets, Sun and Moon rise/set, Moon illumination, seasons, moon phases, planetary transits, lunar and solar eclipses, sub-solar/sub-observer and terminator geometry, angular separation and position/phase/beta angles, and satellite visual magnitude.
- **Observation quality:** RINEX observation QC (completeness, multipath, cycle slips) validated against the standard toolchain, with [explicit interval-metadata handling](docs/rinex-observation-interval-qc.md), carrier-phase combinations, and Hatch smoothing.
- **Simulation:** a deterministic scenario simulator that turns the library into a validation instrument: versioned scenarios with a per-term error budget produce bit-reproducible synthetic observables plus a ground-truth ledger attributing solver error to each budget term.
- **Atmosphere and terrain:** Klobuchar and full Galileo NeQuick-G ionosphere, IONEX grids (vertical TEC and slant delay), tropospheric delay, NRLMSISE-00 density, DTED terrain elevation lookup with batch probes, a memory-mappable terrain store, EGM96/EGM2008 geoid grids, and a PROJ 9.3-compatible EGM96 GTX loader with explicit fused or separately rounded radian interpolation.
- **RF link and signal analysis:** free-space path loss, EIRP, carrier-to-noise (C/N0), link margin, and closed-form navigation-signal figures of merit (BPSK/BOC spectra, spectral separation coefficients, DLL thermal-noise jitter, multipath error envelopes) validated against published constants.
- **Formats:** TLE/OMM (Alpha-5 catalog numbers and CelesTrak GP CSV/JSON), CCSDS OEM/OPM/CDM/TDM, RINEX observation/navigation/clock, CRINEX (Hatanaka encode/decode), SP3, IONEX, ANTEX, Bias-SINEX, CODE DCB, RTCM 3.x, NMEA 0183, with forgiving parsers and round-trippable serializers for the formats that support it.
- **Public product distribution:** exact GNSS product identity is independent of
  its direct archive, NASA CDDIS/Earthdata, local-file, or in-memory source. The
  network-free core derives official SP3/IONEX names, source locations, and
  collision-resistant cache paths; the Python and Elixir interfaces add
  authenticated acquisition, validation, typed failures, and secret-free
  provenance. See the [design note](docs/public-gnss-distribution-sources.md).
  Resilience under analysis-center publication lag: an opt-in cross-line walk
  for CODE's predicted ionosphere (`P1` then `P2`, same map date, provenance
  naming the line served), a bounded publication-status query (newest
  published issue per center and line, its archive-reported publication text,
  and its lag behind nominal - distinguishing "nothing published" from an
  unreachable archive), and a wider ultra pool including the IGS combined
  ultra and Wuhan's hourly MGEX NRT line. Broadcast ephemerides as the
  resilience floor are a recorded
  [design issue](docs/broadcast-ephemeris-resilience-floor.md).

## Install

```sh
cargo add sidereon
```

```rust
use sidereon::astro::passes::{look_angle, GroundStation, UtcInstant};

let line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9009";
let line2 = "2 25544  51.6400 208.8657 0002644 250.3037 109.7782 15.49560812999990";

let elements = sidereon::astro::tle::parse(line1, line2)?.elements.to_element_set()?;
let station = GroundStation { latitude_deg: 51.5, longitude_deg: -0.1, altitude_m: 10.0 };
let when = UtcInstant::from_utc(2024, 1, 1, 12, 0, 0, 0).ok_or("bad datetime")?;

let look = look_angle(&elements, station, when)?;
println!("az {:.2} el {:.2} range {:.1} km", look.azimuth_deg, look.elevation_deg, look.range_km);
```

More runnable examples, in all five languages, are on the [live demo](https://sidereon.dev).

## The command line

The repo ships a single binary for working without writing code:

```text
sidereon solve --obs rover.obs --nav brdc.nav     # SPP per epoch, bounds always shown
sidereon qc --obs rover.obs                       # teqc-style observation quality report
sidereon inspect FILE                             # detect and summarize RINEX/SP3/ANTEX/TLE
sidereon metrics --enu-cov "..."                  # CEP / DRMS / R95 / ellipse from a covariance
sidereon tui --obs rover.obs --nav brdc.nav       # replay a session in the terminal monitor
sidereon tui --ntrip host:2101 --mount MP ...     # watch a live stream solve in real time
sidereon serve-mcp                                # expose the engine to AI agents over MCP
```

## Crates

- [`sidereon-core`](crates/sidereon-core): the engine. SGP4/SDP4 propagation, coordinate and time transforms, RINEX/SP3/ANTEX/OMM/RTCM parsing, broadcast and precise ephemeris, SPP/RTK/PPP/static/DGNSS positioning, DOP and visibility, conjunction assessment, and the supporting numerical kernels.
- [`sidereon`](crates/sidereon): the ergonomic Rust interface over `sidereon-core`. Product loaders, CRINEX encode/decode, Sun/Moon sky helpers, geodetic/topocentric and TLE look-angle/ground-track shortcuts, Doppler helpers, plus SPP/RTK/PPP/static solves with result structs and one error enum. This is the Rust interface, held to the same parity bar as the bindings below.
- [`sidereon-cli`](crates/sidereon-cli): the `sidereon` command-line tool over the engine: `solve` (SPP from RINEX with covariance-derived bounds always shown), `qc`, `metrics`, `inspect`, a terminal solution monitor (`tui`) that replays a logged RINEX session or watches a live NTRIP/TCP RTCM stream, and `serve-mcp`, a Model Context Protocol server exposing the engine to AI agents through task tools plus a typed capability graph. Built from this repo with `cargo build -p sidereon-cli`; not yet on crates.io.
- [`trust-region-least-squares`](crates/trust-region-least-squares): a standalone, independently publishable nonlinear least-squares solver that reproduces SciPy's trust-region-reflective `least_squares` bit-for-bit. It does not depend on the engine.

## Interfaces

The Rust interface is the `sidereon` crate above. The other language interfaces live in their own repositories, each over the same core:

- Python: [`sidereon-python`](https://github.com/neilberkman/sidereon-python) (PyPI: `sidereon`)
- C: [`sidereon-c`](https://github.com/neilberkman/sidereon-c)
- Elixir: [`sidereon-ex`](https://github.com/neilberkman/sidereon-ex) (Hex: `sidereon`)
- JavaScript / WebAssembly: [`sidereon-wasm`](https://github.com/neilberkman/sidereon-wasm) (npm: `@neilberkman/sidereon`)

## Validation

Evaluation is deterministic: the same product, the same version, and the same platform give
identical bits, and releases that change evaluation bits for an existing surface say so in the
changelog. Cross-platform bit identity holds only where a test pins it.

Every numerical routine is cross-checked against the reference implementation or published standard for its domain, not just internal goldens. Each check runs in the test suite against a committed reference fixture whose provenance (tool, version, source) is recorded, and many gates are bit-for-bit.

| Capability | Cross-checked against | Reference |
|---|---|---|
| SGP4 / SDP4 propagation | Vallado / CelesTrak verification states | Vallado, Crawford, Hujsak & Kelso, *Revisiting Spacetrack Report #3*, AIAA 2006-6753 |
| Frames, time, ephemerides (TEME/GCRS/ITRS, IAU 2006/2000A) | Pinned Skyfield vectors (ERFA/SOFA routines under the hood) | IAU SOFA / ERFA |
| Earth orientation | IERS Earth-orientation parameters | Petit & Luzum (eds.), *IERS Conventions (2010)*, IERS TN 36 |
| Least-squares engine (TRF) | SciPy `least_squares`, bit-exact | Virtanen et al., *SciPy 1.0*, Nature Methods 17 (2020) |
| GNSS positioning (SPP / RTK / PPP) | RTKLIB oracle arcs and real IGS precise products | RTKLIB; International GNSS Service (IGS) |
| Geoid undulation (EGM96 15-arcminute) | PROJ (`us_nga_egm96_15`), to 5 mm | PROJ |
| RINEX observation QC - multipath (MP1/MP2) | teqc `+qc` on a real captured stream, to sub-micrometer | teqc |
| RTCM MSM lock-time to RINEX LLI | RTKLIB `convbin` decode of a real MSM stream | RTKLIB |

The per-crate test suites document the exact references, fixtures, and tolerances.

## License

MIT
