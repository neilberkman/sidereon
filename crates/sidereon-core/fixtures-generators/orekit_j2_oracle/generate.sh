#!/usr/bin/env bash
set -euo pipefail

: "${OREKIT_CP:?set OREKIT_CP to the Orekit 13.1.10 and Hipparchus 4.0.3 classpath}"
OUT="${1:?supply the output file}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

javac -d "$BUILD" -cp "$OREKIT_CP" "$HERE/J2Oracle.java"
java -cp "$BUILD:$OREKIT_CP" J2Oracle "$OUT"
