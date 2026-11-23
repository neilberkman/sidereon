#!/usr/bin/env bash
# Print the Orekit oracle for the ICGEM gravity field reader as JSON on
# standard output; the fixture lives at tests/fixtures/icgem/icgem_orekit.json.
#
# usage: OREKIT_CP='/path/to/jars/*' OREKIT_DATA=/path/to/orekit-data ./generate.sh > ../../tests/fixtures/icgem/icgem_orekit.json
#
# OREKIT_CP is a Java classpath holding Orekit 13.1.10 and Hipparchus 4.0.3
# (core, geometry, ode, fitting, optim, filtering, stat); OREKIT_DATA is a git
# checkout of orekit-data. Needs a JDK (javac and java, 21 or later for the
# shortest round-trip Double.toString).
#
# The small fields in tests/fixtures/icgem come from make_fields.py. The
# degree 2190 field is written to a scratch directory by IcgemOracle.java
# from an integer formula the Rust test repeats.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX="$HERE/../../tests/fixtures/icgem"
: "${OREKIT_CP:?set OREKIT_CP to the Orekit and Hipparchus jars}"
: "${OREKIT_DATA:?set OREKIT_DATA to an orekit-data checkout}"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
mkdir "$BUILD/classes" "$BUILD/scratch"

javac -d "$BUILD/classes" -cp "$OREKIT_CP" "$HERE/IcgemOracle.java"

# Versions from the jar names on the classpath.
jar_version() {
    local prefix="$1" jar
    for jar in ${OREKIT_CP//:/ }; do
        case "$(basename "$jar")" in
            "$prefix"-[0-9]*.jar) basename "$jar" .jar | sed "s/^$prefix-//"; return ;;
        esac
    done
    echo unknown
}
OREKIT_VERSION="$(jar_version orekit)"
HIPPARCHUS_VERSION="$(jar_version hipparchus-core)"
DATA_COMMIT="$(git -C "$OREKIT_DATA" rev-parse HEAD)"
JAVA_VERSION="$(java -version 2>&1 | head -1 | sed 's/"/\\"/g')"

printf '{\n'
printf '  "_source": {\n'
printf '    "generator": "fixtures-generators/icgem_orekit_oracle/generate.sh",\n'
printf '    "oracle": "Orekit %s ICGEMFormatReader (missing coefficients not allowed, TT dates) with HolmesFeatherstoneAttractionModel.gradient and nonCentralPart in a body frame equal to the input frame",\n' "$OREKIT_VERSION"
printf '    "orekit": "%s",\n' "$OREKIT_VERSION"
printf '    "hipparchus": "%s",\n' "$HIPPARCHUS_VERSION"
printf '    "orekit_data_commit": "%s",\n' "$DATA_COMMIT"
printf '    "java": "%s",\n' "$JAVA_VERSION"
printf '    "units": "position_m in m (body-fixed), acceleration_m_s2 in m/s^2, potential_m2_s2 the degree 2 and higher potential in m^2/s^2, epoch_tt_s TT seconds since J2000; coefficients fully normalized",\n'
printf '    "epoch_shift": "Orekit is evaluated at epoch_tt_s + orekit_epoch_shift_s; Orekit reads an icgem1.0 yyyymmdd epoch as 12:00, the ICGEM specification as 00:00, so the icgem1.0 case shifts by 43200 s to compare the same time arguments"\n'
printf '  },\n'
java -cp "$BUILD/classes:$OREKIT_CP" IcgemOracle "$FIX" "$OREKIT_DATA" "$BUILD/scratch"
printf '}\n'
