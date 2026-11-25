#!/usr/bin/env bash
set -euo pipefail

: "${OREKIT_CP:?Orekit and Hipparchus classpath}"
: "${OREKIT_DATA:?Orekit data directory}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT
javac -d "$build" -cp "$OREKIT_CP" "$here/EarthOrientationOracle.java"
java -cp "$build:$OREKIT_CP" org.orekit.frames.EarthOrientationOracle "$OREKIT_DATA" "$@"
