# Broadcast ephemerides as the product-acquisition resilience floor

Status: design issue - no implementation. This records the case for promoting
the broadcast navigation line from a comparison input to an explicit orbit
source of last resort, and the constraints any implementation must honor.

## Motivating incident

On 2026-08-04 (UTC) the precise ultra-rapid ecosystem ran roughly a day
behind schedule, together:

- GFZ's newest published ultra was the day-215 00:00 issue, published
  2026-08-04 04:26 archive-local - about 28 hours behind its nominal issue
  time (recorded in `crates/sidereon-core/tests/fixtures/listings/`).
- ESA's newest ultra was the day-215 00:00 issue.
- The IGS combined ultra at BKG had not yet created the current week's
  directory; its newest published issue was day-209 18:00, six days old.
- CODE's one-day predicted ionosphere had not published the current map date.

Every one of those lines is an analysis-center *computation* product: a
center's pipeline stalls, its line stalls. The catalog work that followed the
incident widened the precise pool (IGS combined ultra, Wuhan MGEX NRT) and
made lag observable (`publication_listing_urls`, `newest_published_product`,
`published_issue_age_minutes`, the scoreboard `publication_status` query),
but the pool is still a monoculture in one respect: with no analysis center
publishing, there is no orbit source at all.

Broadcast ephemerides break that monoculture. They are computed onboard the
operational control segments and published near-continuously by archive
mirrors as merged daily RINEX NAV files; a global multi-center outage of the
kind above does not interrupt them, because receivers world-wide keep
decoding the signal. Meter-class orbits (roughly 1 m SISRE for GPS/Galileo,
worse for GLONASS/BeiDou GEO) are three orders coarser than a precise ultra,
but they are the difference between a degraded solve and no solve.

## What already exists

The pieces are present and tested; they are just not wired as an acquisition
fallback:

- Catalog: the merged broadcast line `BRDC00WRD_R_<YYYYDDD>0000_01D_MN.rnx`
  is a first-class entry (`ProductType::Nav`, `mgex_nav`, BKG
  `BRDC/<year>/<doy>` layout, `SolutionClass::Broadcast`), with its exact
  identity and cache path.
- Evaluation: `BroadcastEphemeris` evaluates RINEX NAV records to ECEF
  position and clock, and `broadcast_comparison` already quantifies its
  error against SP3 (SISRE decomposition), which is exactly the accuracy
  label a fallback needs.
- Publication status: `publication_listing_urls` handles year/day-of-year
  layouts with the same bounded walk-back as week layouts, so "is the BRDC
  line current" is answerable by the same one-query API as the precise
  lines.

## Proposed shape (for a future change)

1. **A `nav_date_candidates` walk** mirroring `gim_date_candidates`: the
   merged BRDC file for the target civil day, then the previous day (whose
   records still cover the boundary hours through their fit intervals).
   Bounded, newest first, never a silent substitution: each candidate is its
   own exact identity.
2. **An explicit orbit-source ladder at the caller-facing solve boundary**:
   precise merged consensus, then a single precise line, then broadcast -
   with the resolved rung carried in provenance, never blended. A solution
   computed from broadcast orbits must be distinguishable from a precise
   solution in every report that names its inputs.
3. **Fail-closed defaults preserved**: the ladder is opt-in exactly like the
   cross-line predicted-IONEX walk. A caller that requested a precise
   product keeps getting "unavailable" rather than silently receiving
   meter-class orbits.
4. **Accuracy labeling, not accuracy guessing**: when a broadcast rung
   serves, the report should carry the solution-class code (`broadcast`)
   and, where a recent precise product exists for a *previous* window, may
   cite a measured SISRE from `broadcast_comparison` rather than a nominal
   constant.

## Constraints for any implementation

- Provenance and cache identity semantics are load-bearing. The broadcast
  artifact keeps its own identity (`BRDC00WRD...`, `SolutionClass::
  Broadcast`); a fallback that reuses a cache entry under a different
  identity than it was keyed by is wrong even if convenient.
- The RINEX NAV health and fit-interval flags must gate per-satellite use;
  a stalled precise line must not be replaced by an unhealthy broadcast
  record.
- Time-system care: BRDC records carry per-system time frames; the existing
  `BroadcastEphemeris` evaluation and the interface-boundary calendar rules
  own this, and the ladder must not add its own conversions.
- No polling loops; availability questions go through the same bounded
  publication-status query as every other line.
