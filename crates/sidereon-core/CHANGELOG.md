# Changelog

All notable changes to `sidereon-core` are documented here.

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
