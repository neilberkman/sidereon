#!/usr/bin/env bash
# Regenerate tests/fixtures/orbit_determination/orbit_parameter_fit_orekit.json:
# Orekit's BatchLSEstimator fitting GPS G03 of the CODE MGEX final orbit
# tests/fixtures/sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3 (2020-06-25, 289
# epochs at 5 minutes) with ParameterDrivers selected on the cannonball SRP
# coefficient, on empirical RTN accelerations, and on ECOM2
# (OrbitParameterOracle.java).
# Read by crates/sidereon-core/tests/orbit_parameter_fit_orekit.rs.
#
# usage: OREKIT_CP='<dir>/*' OREKIT_DATA=<orekit-data checkout> ./generate.sh > fixture.json
#
# OREKIT_CP is a Java classpath holding the Orekit and Hipparchus jars (the
# fixture was made with Orekit 13.1.10 and Hipparchus 4.0.3); a 'dir/*' entry
# stands for the jars in dir. OREKIT_DATA is an orekit-data directory, recorded
# by its git commit; its Earth orientation parameters rotate the SP3 positions
# to GCRF. Needs a JDK (javac, java).
# Writes the fixture JSON to standard output.
set -euo pipefail

: "${OREKIT_CP:?set OREKIT_CP to the Orekit and Hipparchus jar classpath}"
: "${OREKIT_DATA:?set OREKIT_DATA to an orekit-data directory}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP3="$HERE/../../tests/fixtures/sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3"
SATELLITE=G03
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

javac -nowarn -d "$BUILD" -cp "$OREKIT_CP" "$HERE/OrbitParameterOracle.java" >&2

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

PROVENANCE="$(printf '{"generator": "fixtures-generators/orekit_orbit_parameter_oracle/generate.sh", "oracle": "Orekit %s BatchLSEstimator, Gauss-Newton, ParameterDrivers selected on SolarRadiationPressure, ParametricAcceleration, and ECOM2", "orekit": "%s", "hipparchus": "%s", "orekit_data_commit": "%s", "java": "%s"}' \
    "$OREKIT_VERSION" "$OREKIT_VERSION" "$HIPPARCHUS_VERSION" "$DATA_COMMIT" "$JAVA_VERSION")"

java -cp "$BUILD:$OREKIT_CP" OrbitParameterOracle "$OREKIT_DATA" "$PROVENANCE" "$SP3" "$SATELLITE"
