#!/usr/bin/env bash
# Write the DE440 oracle and its binary records to the supplied directory.
set -euo pipefail

: "${OREKIT_CP:?set OREKIT_CP to the Orekit 13.1.10 and Hipparchus 4.0.3 classpath}"
: "${OREKIT_DATA:?set OREKIT_DATA to an orekit-data checkout}"
OUT="${1:?supply the output directory}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
DATA_COMMIT="$(git -C "$OREKIT_DATA" rev-parse HEAD)"

javac -d "$BUILD" -cp "$OREKIT_CP" "$HERE/DeOracle.java"
java -cp "$BUILD:$OREKIT_CP" DeOracle "$OREKIT_DATA" "$OUT" "$DATA_COMMIT"
grep -q '"orekit": "13.1.10"' "$OUT/de440_orekit.json"
grep -q '"hipparchus": "4.0.3"' "$OUT/de440_orekit.json"
