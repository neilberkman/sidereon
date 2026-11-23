#!/usr/bin/env bash
# Print the Orekit fixture for the mean-of-date to GCRS rotation and the GCRS
# Sun and Moon, tests/fixtures/bodies/sun_moon_gcrs_orekit.json, on standard
# output:
#
#   ./generate.sh > ../../tests/fixtures/bodies/sun_moon_gcrs_orekit.json
#
# SunMoonGcrsOracle.java transforms fixed vectors from Orekit's mean-of-date
# frame (FramesFactory.getMOD with the IERS 2010 conventions: IAU 2006
# precession from EME2000, which is GCRF with the IAU 2006 frame bias) to GCRF,
# and records the JPL DE 440 Sun and Moon in GCRF and in that frame, at TT epochs
# from 1990 to 2050. Read by tests/sun_moon_gcrs_orekit_oracle.rs.
#
# Needs a JDK (javac and java, 8 or later), git, and two environment variables:
#   OREKIT_CP    classpath of the Orekit 13.1.10 and Hipparchus 4.0.3 jars
#                (a directory glob such as /path/to/jars/* is accepted)
#   OREKIT_DATA  a checkout of https://gitlab.orekit.org/orekit/orekit-data
#                (DE 440 ephemerides, UTC-TAI history); its commit is recorded
#                in the output
set -euo pipefail

: "${OREKIT_CP:?set OREKIT_CP to the Orekit 13.1.10 and Hipparchus 4.0.3 classpath}"
: "${OREKIT_DATA:?set OREKIT_DATA to an orekit-data checkout}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

DATA_COMMIT="$(git -C "$OREKIT_DATA" rev-parse HEAD)"

javac -d "$BUILD" -cp "$OREKIT_CP" "$HERE/SunMoonGcrsOracle.java"
java -cp "$BUILD:$OREKIT_CP" SunMoonGcrsOracle "$OREKIT_DATA" "$DATA_COMMIT" > "$BUILD/out.json"

# The fixture is only meaningful for the pinned versions.
grep -q '"orekit": "13.1.10"' "$BUILD/out.json" || { echo "Orekit is not 13.1.10" >&2; exit 1; }
grep -q '"hipparchus": "4.0.3"' "$BUILD/out.json" || { echo "Hipparchus is not 4.0.3" >&2; exit 1; }
cat "$BUILD/out.json"
