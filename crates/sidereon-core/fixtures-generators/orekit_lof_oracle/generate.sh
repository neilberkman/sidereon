#!/usr/bin/env bash
# Print the Orekit local-orbital-frame fixture for the orbit scoring tests,
# tests/fixtures/orbit_scoring/orekit_lof.json, on standard output:
#
#   ./generate.sh > ../../tests/fixtures/orbit_scoring/orekit_lof.json
#
# LofOracle.java resolves difference vectors on Orekit's LOFType.QSW and
# LOFType.TNW for synthetic Keplerian states and for the IGS ultra-rapid
# predicted records against the IGS final records of 2026-05-13 00:00 to 03:00
# GPST (tests/fixtures/sp3). Read by tests/orbit_scoring.rs.
#
# Needs a JDK (javac and java, 8 or later) and two environment variables:
#   OREKIT_CP    classpath of the Orekit 13.1.10 and Hipparchus 4.0.3 jars
#                (a directory glob such as /path/to/jars/* is accepted)
#   OREKIT_DATA  an orekit-data checkout; its commit is recorded in the output
set -euo pipefail

: "${OREKIT_CP:?set OREKIT_CP to the Orekit 13.1.10 and Hipparchus 4.0.3 classpath}"
: "${OREKIT_DATA:?set OREKIT_DATA to an orekit-data checkout}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP3="$HERE/../../tests/fixtures/sp3"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

DATA_COMMIT="$(git -C "$OREKIT_DATA" rev-parse HEAD)"

javac -d "$BUILD" -cp "$OREKIT_CP" "$HERE/LofOracle.java"
java -cp "$BUILD:$OREKIT_CP" LofOracle "$OREKIT_DATA" "$DATA_COMMIT" \
    "$SP3/IGS0OPSFIN_20261330000_03H_15M_ORB.SP3" \
    "$SP3/IGS0OPSULT_20261311800_02D_15M_ORB_2245_0415.SP3" \
    "$SP3/IGS0OPSULT_20261320000_02D_15M_ORB_2145_0415.SP3" > "$BUILD/out.json"

# The fixture is only meaningful for the pinned versions.
grep -q '"orekit": "13.1.10"' "$BUILD/out.json" || { echo "Orekit is not 13.1.10" >&2; exit 1; }
grep -q '"hipparchus": "4.0.3"' "$BUILD/out.json" || { echo "Hipparchus is not 4.0.3" >&2; exit 1; }
cat "$BUILD/out.json"
