#!/usr/bin/env bash
# Regenerate tests/fixtures/attitude/gnss_attitude_orekit.json: Orekit's
# org.orekit.gnss.attitude providers (GPSBlockIIA, GPSBlockIIR, GPSBlockIIF,
# Glonass, Galileo, BeidouMeo, BeidouIGSO, BeidouGeo; Galileo also with the
# GPS III limits) evaluated on closed-form circular and eccentric Keplerian
# orbits and a closed-form Sun (GnssAttitudeOracle.java). Read by
# crates/sidereon-core/tests/gnss_attitude_orekit.rs.
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

javac -nowarn -d "$BUILD" -cp "$OREKIT_CP" "$HERE/GnssAttitudeOracle.java" >&2

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

PROVENANCE="$(printf '{"generator": "fixtures-generators/orekit_gnss_attitude_oracle/generate.sh", "oracle": "Orekit %s org.orekit.gnss.attitude providers, one provider per case and block evaluated at increasing times", "orekit": "%s", "hipparchus": "%s", "orekit_data_commit": "%s", "java": "%s"}' \
    "$OREKIT_VERSION" "$OREKIT_VERSION" "$HIPPARCHUS_VERSION" "$DATA_COMMIT" "$JAVA_VERSION")"

java -cp "$BUILD:$OREKIT_CP" GnssAttitudeOracle "$OREKIT_DATA" "$PROVENANCE"
