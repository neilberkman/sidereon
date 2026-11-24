#!/usr/bin/env bash
# Regenerate tests/fixtures/eclipse/orekit_shadow_srp_oracle.json: bit-exact
# hex-encoded lighting ratios, cannonball SRP accelerations and their
# FieldSpacecraftState<Gradient> position and C_R partials from Orekit 13.1.10 /
# Hipparchus 4.0.3 (ShadowSrpOracle.java).
#
# usage: OREKIT_CP='/path/to/jars/*' OREKIT_DATA=/path/to/orekit-data \
#            ./generate.sh > ../../tests/fixtures/eclipse/orekit_shadow_srp_oracle.json
set -euo pipefail

: "${OREKIT_CP:?set OREKIT_CP to the class path of the Orekit and Hipparchus jars}"
: "${OREKIT_DATA:?set OREKIT_DATA to an orekit-data checkout}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

DATA_COMMIT="$(git -C "$OREKIT_DATA" rev-parse HEAD)"
javac -d "$BUILD" -cp "$OREKIT_CP" "$HERE/ShadowSrpOracle.java"
java -cp "$BUILD:$OREKIT_CP" ShadowSrpOracle "$OREKIT_DATA" "$DATA_COMMIT"
