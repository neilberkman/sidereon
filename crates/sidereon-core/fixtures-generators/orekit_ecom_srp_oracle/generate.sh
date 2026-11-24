#!/usr/bin/env bash
# Regenerate tests/fixtures/forces/ecom_srp_orekit.json: Orekit's
# org.orekit.forces.radiation.ECOM2 double and FieldSpacecraftState<Gradient>
# acceleration, state partials and parameter partials (one realistic
# coefficient set per case) with SolarRadiationPressure.getLightingRatio on
# circular orbits with a fixed Sun (and optional Moon), at instants in
# sunlight, penumbra and umbra (EcomOracle.java). Read by
# crates/sidereon-core/tests/ecom_srp_orekit.rs.
#
# usage: OREKIT_CP='<dir>/*' OREKIT_DATA=<orekit-data checkout> ./generate.sh > fixture.json
#
# OREKIT_CP is a Java classpath holding the Orekit and Hipparchus jars (the
# fixture was made with Orekit 13.1.10 and Hipparchus 4.0.3); a 'dir/*' entry
# stands for the jars in dir. OREKIT_DATA is an orekit-data directory, recorded
# by its git commit. Needs a JDK (javac, java).
# Writes the fixture JSON to standard output.
set -euo pipefail

: "${OREKIT_CP:?set OREKIT_CP to the Orekit and Hipparchus jar classpath}"
: "${OREKIT_DATA:?set OREKIT_DATA to an orekit-data directory}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

javac -nowarn -d "$BUILD" -cp "$OREKIT_CP" "$HERE/EcomOracle.java" >&2

jar_version() {
    # The version in the file name of the jar named $1-<version>.jar on the
    # classpath; a classpath entry ending in '*' stands for its directory's jars.
    local jar
    # Unquoted on purpose: a 'dir/*' entry expands to the jars in dir.
    for jar in ${OREKIT_CP//:/ }; do
        case "$(basename "$jar")" in
            "$1"-[0-9]*.jar) basename "$jar" .jar | sed "s/^$1-//"; return ;;
        esac
    done
    echo unknown
}

OREKIT_VERSION="$(jar_version orekit)"
HIPPARCHUS_VERSION="$(jar_version hipparchus-core)"
DATA_COMMIT="$(git -C "$OREKIT_DATA" rev-parse HEAD 2>/dev/null || echo unknown)"
JAVA_VERSION="$(java -version 2>&1 | head -1 | sed 's/"/\\"/g')"

PROVENANCE="$(printf '{"generator": "fixtures-generators/orekit_ecom_srp_oracle/generate.sh", "oracle": "Orekit %s org.orekit.forces.radiation.ECOM2 double and FieldSpacecraftState<Gradient> accelerations and partials with SolarRadiationPressure.getLightingRatio", "orekit": "%s", "hipparchus": "%s", "orekit_data_commit": "%s", "java": "%s"}' \
    "$OREKIT_VERSION" "$OREKIT_VERSION" "$HIPPARCHUS_VERSION" "$DATA_COMMIT" "$JAVA_VERSION")"

java -cp "$BUILD:$OREKIT_CP" EcomOracle "$OREKIT_DATA" "$PROVENANCE"
