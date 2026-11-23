#!/usr/bin/env bash
# Regenerate tests/fixtures/eclipse/moon_eclipse_orekit.json: the lighting ratio
# and acceleration of Orekit's SolarRadiationPressure with the Moon added as an
# occulting body, its occultation angles, and the Montenbruck & Gill shadow
# function evaluated on those angles with 40 digits (MoonEclipseOracle.java).
# Read by the Rust tests of astro::events::eclipse and astro::forces::srp.
#
# usage: OREKIT_CP='/path/to/jars/*' OREKIT_DATA=/path/to/orekit-data \
#            ./generate.sh > ../../tests/fixtures/eclipse/moon_eclipse_orekit.json
#
# OREKIT_CP is a Java class path holding the Orekit jar and the Hipparchus jars
# it needs (core, geometry, ode, fitting, optim, filtering, stat). OREKIT_DATA
# is a checkout of https://gitlab.orekit.org/orekit/orekit-data, which supplies
# the JPL DE 440 ephemeris and the UTC-TAI history. The JSON goes to standard
# output and records the Orekit and Hipparchus versions found on the class path
# and the orekit-data commit. Needs a JDK (javac and java, 8 or later) and git.
set -euo pipefail

: "${OREKIT_CP:?set OREKIT_CP to the class path of the Orekit and Hipparchus jars}"
: "${OREKIT_DATA:?set OREKIT_DATA to an orekit-data checkout}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

DATA_COMMIT="$(git -C "$OREKIT_DATA" rev-parse HEAD)"
javac -d "$BUILD" -cp "$OREKIT_CP" "$HERE/MoonEclipseOracle.java"
java -cp "$BUILD:$OREKIT_CP" MoonEclipseOracle "$OREKIT_DATA" "$DATA_COMMIT"
