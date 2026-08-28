# sidereon

GNSS and astrodynamics for Rust: propagate satellites, predict passes, solve
precise positions (SPP / RTK / PPP), and convert between coordinate frames and
time scales, checked against the references the field trusts (Vallado, Skyfield,
IGS, IERS).

It's a pure-Rust engine, fast and `#![forbid(unsafe_code)]` at the surface, with
one ergonomic crate that re-exports the whole stack. You just `cargo add
sidereon`.

## Install

```
cargo add sidereon
```

## Quickstart: when does the ISS fly over you?

No data files, no setup: give it a two-line element set and a ground station,
and ask when the satellite is above the horizon.

```rust
use std::time::{SystemTime, UNIX_EPOCH};

use sidereon::passes::{find_passes_for_satellite, GroundStation, PassFinderOptions, UtcInstant};
use sidereon::sgp4::Satellite;

fn main() {
    // Real ISS orbital elements (grab fresh ones from CelesTrak any time).
    let iss = Satellite::from_tle(
        "1 25544U 98067A   26178.50947090  .00006280  00000+0  12016-3 0  9996",
        "2 25544  51.6322 248.9966 0004278 238.4942 121.5629 15.49454046573359",
    )
    .expect("valid TLE");

    // A ground station: latitude, longitude in degrees, altitude in metres.
    let berkeley = GroundStation {
        latitude_deg: 37.87,
        longitude_deg: -122.27,
        altitude_m: 52.0,
    };

    // The next 24 hours, as UTC unix microseconds (the time unit everywhere here).
    let now_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after 1970")
        .as_micros() as i64;
    let start = UtcInstant::from_unix_microseconds(now_us);
    let end = UtcInstant::from_unix_microseconds(now_us + 24 * 3_600 * 1_000_000);

    // Every pass that climbs above 10 degrees.
    let options = PassFinderOptions {
        elevation_mask_deg: 10.0,
        ..PassFinderOptions::default()
    };
    let passes = find_passes_for_satellite(&iss, berkeley, start, end, options)
        .expect("valid pass-finder inputs");

    for pass in &passes {
        let secs = pass.aos.unix_microseconds() / 1_000_000;
        let (hh, mm) = ((secs % 86_400) / 3_600, (secs % 3_600) / 60);
        let minutes = (pass.los.unix_microseconds() - pass.aos.unix_microseconds()) as f64 / 60.0e6;
        println!("{hh:02}:{mm:02} UTC | {minutes:4.1} min | peak {:2.0} deg", pass.max_elevation_deg);
    }
}
```

A typical run prints something like:

```
08:30 UTC |  6.8 min | peak 88 deg
15:01 UTC |  6.6 min | peak 56 deg
16:39 UTC |  3.5 min | peak 14 deg
```

Each [`SatellitePass`] gives you acquisition (`aos`), loss (`los`), culmination
time, and peak elevation. The same `sidereon::passes` module has `look_angle`
(azimuth / elevation / range to a satellite at an instant), `ground_track`, and
`propagate_teme_arc` for raw state vectors; `Satellite` from `sidereon::sgp4` is
the propagator behind all of it.

## Precise positioning

The positioning engine is the other half of the library: feed it pseudoranges
and a precise-ephemeris (SP3) product and it returns a least-squares fix.

```rust
use sidereon::positioning::{Corrections, Observation, SolveInputs, SolvePolicy};
use sidereon::{load_sp3, solve_spp, GnssSatelliteId, GnssSystem};

let sp3 = load_sp3(&std::fs::read("igs_product.sp3")?)?;

let inputs = SolveInputs {
    observations: vec![
        Observation { satellite_id: GnssSatelliteId::new(GnssSystem::Gps, 1)?, pseudorange_m: 21_000_123.4 },
        Observation { satellite_id: GnssSatelliteId::new(GnssSystem::Gps, 8)?, pseudorange_m: 22_517_889.1 },
        // ...more satellites
    ],
    t_rx_j2000_s: receive_epoch_j2000_s,
    corrections: Corrections::IONO_TROPO,
    // ...time-of-day / day-of-year, Klobuchar coefficients, surface met, initial guess
    ..spp_inputs
};

let fix = solve_spp(&sp3, &inputs, /* with_geodetic */ true, policy)?;
println!("{:?}", fix.position);    // ItrfPositionM: ECEF metres
println!("{:?}", fix.geodetic);    // Some(Wgs84Geodetic): lat / lon / height
println!("{:?}", fix.used_sats);   // the satellites that contributed
```

`solve_rtk_float_with`, `solve_rtk_fixed_with`, `solve_ppp_float_with`, and
`solve_ppp_fixed_with` follow the same pattern: a typed config in, a result struct
with ECEF/geodetic position, residuals, DOP, and status out. One [`Error`] enum
unifies every product-parse and solve failure, and `solve_spp_batch` fans a fleet
of epochs across a rayon pool, bit-identical to the serial path.

## What's in the box

- **Orbits:** SGP4/TLE and OMM, numerical propagation with atmospheric drag and decay/reentry prediction, Kepler and anomaly conversions, classical and equinoctial elements, passes, look angles, ground tracks
- **Frames, time & geodesy:** TEME ↔ GCRS ↔ ITRS, GMST/GAST, geodetic ↔ ECEF, topocentric coordinates, UTC/TT/TDB/UT1, EGM96/EGM2008 geoid grids, PROJ EGM96 GTX loading with explicit fused or separately rounded interpolation, DTED terrain elevation
- **Bodies & almanac:** Sun/Moon/planet apparent places (geocentric or topocentric RA/Dec and az/el), Sun and Moon rise/set, Moon illumination, seasons, moon phases, eclipses, planetary transits, plus JPL SPK (DAF/.bsp) kernels
- **Observation geometry:** angular separation and position angle, phase/beta/parallactic angles, sub-solar and sub-observer points, terminator, satellite visual magnitude
- **Positioning:** SPP, RINEX observation to SPP assembly and solve helpers, RTK (float/fixed), PPP (float/fixed), DOP, velocity, robust fault detection and exclusion
- **Estimation and detection:** scalar alpha-beta and Kalman trackers, NIS innovation gates, EWMA/MAD/CFAR detectors, covariance-weighted `TrackFilter`, and fixed-interval RTS smoothing
- **GNSS/INS fusion:** loose and tight updates, inertial checkpoint serialization, RTS smoothing, outage velocity matching, stationary and non-holonomic pseudo-updates
- **GNSS data:** SP3, RINEX (obs/nav/clock), CRINEX encode/decode, ANTEX, broadcast ephemeris, Bias-SINEX / CODE DCB biases, source-agnostic ephemeris sampling
- **Protocol and simulation:** sans-I/O NTRIP requests, handshakes, sourcetables, chunked streams, and GGA formatting (live transport remains caller-owned); deterministic synthetic GNSS scenarios with media/source variants and a typed truth ledger
- **Corrections:** SBAS, RTCM SSR and Galileo HAS orbit/clock/bias correction stores
- **Space situational awareness:** conjunction/TCA screening, collision probability, CDM, covariance, relative motion (RIC/RTN/LVLH, Clohessy-Wiltshire)
- **RF:** link budget (FSPL, EIRP, C/N0, antenna gain)

The product parsers, CRINEX encoder, Sun/Moon sky helpers, look-angle,
ground-track, geodetic/topocentric, Doppler, and propagation shortcuts live at
the crate root (`load_sp3`, `encode_crinex`, `solve_spp`, `passes`, `sgp4`,
`tle`, `tca`, `relative`, `almanac`, `InertialFilter`,
`velocity_match_outage_to_state`); the full astrodynamics tree is under `sidereon::astro`. Lower-level RTK/PPP internals stay
behind the explicit `sidereon::raw` escape hatch so the ergonomic surface stays
small.

The Bias-SINEX and CODE DCB convenience path loaders treat `.gz` files as
complete RFC 1952 member series and validate every member header and trailer.
They cap local archives at 64 MiB and cumulative output at 500 MiB. Consumers
with a different I/O policy can decode bytes externally and pass them to the
existing `parse_bias_sinex*` or `parse_code_dcb*` functions.

## Other languages

sidereon is one validated engine with first-class interfaces in **Rust**,
**Python**, **C**, **Elixir**, and **WebAssembly**: same numbers everywhere.
See the live demo and docs at [sidereon.dev](https://sidereon.dev).

## How it's validated

The SGP4 propagator is a Rust port of David Vallado's reference implementation,
bit-exact to it. Frames and time are checked against Skyfield and IERS; the
positioning stack is checked against IGS products.

MIT licensed. The engine's SGP4 propagation credits David Vallado (AIAA 2006);
see the `sidereon-core` crate for full attribution.
