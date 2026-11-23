#!/usr/bin/env bash
# Regenerate tests/fixtures/analytic/orekit_analytic_propagators.json, the Orekit
# oracle for the Eckstein-Hechler and Brouwer-Lyddane propagators in
# src/astro/propagator/analytic: osculating states propagated over seven days,
# osculating-to-mean conversions and mean-to-osculating conversions for LEO,
# MEO, GNSS, IGSO and GEO-like orbits, and the cases Orekit refuses.
#
# usage: ./generate.sh > ../../tests/fixtures/analytic/orekit_analytic_propagators.json
#
# Needs a JDK (javac and java, 11 or later), OREKIT_CP set to a classpath that
# holds the Orekit 13.1.10 and Hipparchus 4.0.3 jars (a directory glob such as
# "/path/to/jars/*" works), and OREKIT_DATA set to an orekit-data checkout. The
# JSON goes to standard output; progress and compiler messages go to standard
# error.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OREKIT_CP:?set OREKIT_CP to the Orekit and Hipparchus jars}"
: "${OREKIT_DATA:?set OREKIT_DATA to an orekit-data checkout}"

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

# Record the exact jars on the classpath.
jar_record() {
    local pattern="$1" found=""
    local IFS=':'
    for entry in $OREKIT_CP; do
        for jar in $entry; do
            case "$(basename "$jar")" in
                $pattern)
                    found="$(basename "$jar") sha256 $(sha256sum "$jar" | cut -d' ' -f1)" ;;
            esac
        done
    done
    [ -n "$found" ] || { echo "no $pattern on OREKIT_CP" >&2; exit 1; }
    printf '%s' "$found"
}
OREKIT_JAR="$(jar_record 'orekit-[0-9]*.jar')"
HIPPARCHUS_JAR="$(jar_record 'hipparchus-core-[0-9]*.jar')"
case "$OREKIT_JAR" in orekit-13.1.10.jar*) ;; *) echo "expected Orekit 13.1.10, found $OREKIT_JAR" >&2; exit 1 ;; esac
case "$HIPPARCHUS_JAR" in hipparchus-core-4.0.3.jar*) ;; *) echo "expected Hipparchus 4.0.3, found $HIPPARCHUS_JAR" >&2; exit 1 ;; esac
DATA_COMMIT="$(git -C "$OREKIT_DATA" rev-parse HEAD 2>/dev/null || echo unknown)"

javac -nowarn -d "$BUILD" -cp "$OREKIT_CP" "$HERE/AnalyticPropagatorOracle.java" >&2
JAVA_VERSION="$(java -version 2>&1 | head -1 | sed 's/"/\\"/g')"

printf '{\n'
printf '  "_source": {\n'
printf '    "generator": "fixtures-generators/orekit_analytic_oracle/generate.sh",\n'
printf '    "oracle": "Orekit EcksteinHechlerPropagator, BrouwerLyddanePropagator, FixedPointConverter with EcksteinHechlerTheory and BrouwerLyddaneTheory (damping 1)",\n'
printf '    "orekit": "13.1.10",\n'
printf '    "orekit_jar": "%s",\n' "$OREKIT_JAR"
printf '    "hipparchus": "4.0.3",\n'
printf '    "hipparchus_jar": "%s",\n' "$HIPPARCHUS_JAR"
printf '    "orekit_data_commit": "%s",\n' "$DATA_COMMIT"
printf '    "java": "%s",\n' "$JAVA_VERSION"
printf '    "frame": "EME2000; both theories are frame-agnostic and take the pole as +Z",\n'
printf '    "units": "metres, metres per second, radians, seconds; m2 in rad/s^2",\n'
printf '    "states": "initial is the osculating Cartesian state each propagator starts from; samples are Orekit propagateOrbit at epoch + dt_s; from_mean starts a propagator from the printed mean elements (PropagationType.MEAN)",\n'
printf '    "eckstein_hechler_elements": "circular parameters a, ex = e cos(pa), ey = e sin(pa), i, raan, alpha_m = pa + mean anomaly",\n'
printf '    "brouwer_lyddane_elements": "Keplerian a, e, i, pa, raan, m (mean anomaly)"\n'
printf '  },\n'
java -cp "$BUILD:$OREKIT_CP" AnalyticPropagatorOracle
printf '}\n'
