# sidereon-core

The complete engine behind [`sidereon`](https://crates.io/crates/sidereon): satellite
propagation and observation plus GNSS positioning, in one pure-Rust crate. It is the fold
of the propagation and GNSS layers. The GNSS layer sits behind a
default `gnss` feature, so a propagation-only consumer can build with `--no-default-features`
and never compile the SP3/RINEX/IONEX/solver code.

## Propagation and observation

- **SGP4 / TLE:** TLE parsing and SGP4 propagation, validated bit-exact against the Vallado
  reference (see Credits).
- **Numerical propagation:** Cartesian state propagation under two-body and J2 gravity, with
  fixed-step RK4 and adaptive Dormand-Prince 5(4) integrators.
- **Frames and time:** ITRS/GCRS transforms with precession, nutation, and GAST; TT/TAI/UT1
  time scales backed by IERS/IAU tables.
- **Geometry and events:** topocentric look angles, pass prediction, and eclipse/shadow
  geometry.
- **Conjunctions:** close-approach screening, collision probability, and RTN-frame covariance.
- **I/O:** TLE, CCSDS Orbit Mean-Elements Message / OMM (KVN, XML, and JSON; JSON behind the
  default-on `json` feature), and CCSDS Conjunction Data Message / CDM (KVN and XML). An OMM
  drives SGP4 bit-identically (0 ULP) to the equivalent TLE.

## GNSS positioning (default `gnss` feature)

- SP3 precise ephemeris and RINEX 3.x/4.x navigation (GPS, Galileo, BeiDou, GLONASS).
- Single-point positioning, double-differenced RTK with LAMBDA ambiguity resolution, and
  static precise point positioning.
- Broadcast Klobuchar ionosphere and Saastamoinen plus Niell troposphere.

## Parity bar

Every independently reproducible, libm-bound component (propagation, frames, time, SGP4,
ionosphere, troposphere, DOP, orbit and clock evaluation) is held to bit-exact (0 ULP)
parity against pinned references, proven by committed hex-float golden vectors. Solver
converged positions are sub-micron solver-agreement results, not a 0-ULP claim (the
linear-algebra step is BLAS-bound). 0 ULP is certified against a pinned target (OS/arch,
libm, toolchain, FMA policy); other platforms run the same algorithms but need their own
fixtures. Units are SI, with frame and datum encoded in the type system.

## Credits

The SGP4/SDP4 model is the public AIAA/Spacetrack theory. sidereon-core's SGP4 is a Rust
port of David Vallado's reference C++ (companion code to "Fundamentals of Astrodynamics and
Applications"), the same source the `sgp4` Python package ports. Credit to David Vallado and
the 2006 AIAA paper (Vallado, Crawford, Hujsak, Kelso). Vallado's C++ is used only as a
development-time parity oracle and is not distributed with this crate.

## License

MIT
