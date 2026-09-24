#!/usr/bin/env bash
# Regenerate the two Orekit solid Earth tide fixtures in tests/fixtures/tides:
#
# - geopotential_tide_step2_orekit.json: the IERS Conventions (2010) Chapter 6
#   Step 2 corrections to C20, C21, S21, C22 and S22 that Orekit computes from
#   its own copies of Tables 6.5a-c, with the fundamental arguments it formed,
#   at 64 epochs from 1975 to 2050 (TideStep2Oracle.java). Read by the Rust
#   tests `step2_matches_orekit_for_the_same_arguments` and
#   `step2_at_epoch_matches_orekit`.
# - geopotential_tide_orekit_field.json: the degree 2 to 4 corrections of
#   Orekit's SolidTidesField (Steps 1 and 2, and Step 3 for a zero-tide field)
#   for supplied Sun and Moon positions at 48 epochs from 1980 to 2045
#   (SolidTidesFieldOracle.java). Read by the Rust test
#   `steps_1_to_3_match_orekit_solid_tides_field`.
#
# usage: ./generate.sh
#
# Needs a JDK (javac and java, 8 or later). The Orekit and Hipparchus jars are
# downloaded from Maven Central and checked against the SHA-256 digests below.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX="$HERE/../../tests/fixtures/tides"
OUT="$FIX/geopotential_tide_step2_orekit.json"
FIELD_OUT="$FIX/geopotential_tide_orekit_field.json"
MAVEN="https://repo1.maven.org/maven2"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

JARS="
5d035486d916a458d6a541866db99885a21d5bc0cbdb3785bb8f42ee8a70ddf4 org/orekit/orekit/13.1.8/orekit-13.1.8.jar
5af54dc3c21b8d85e1264157ee91d93fa67e68c825cfb38834dc2bbb9b0f6583 org/hipparchus/hipparchus-core/4.0.3/hipparchus-core-4.0.3.jar
19594d5106ec6a46951e859e2924bfcade02a834c06c56bd3d2130a74f0fbb7e org/hipparchus/hipparchus-geometry/4.0.3/hipparchus-geometry-4.0.3.jar
75f0f0b3fdcd99d13e9c394b7ae809cce5a9c22f1c14b1b90a5109ed951467fe org/hipparchus/hipparchus-ode/4.0.3/hipparchus-ode-4.0.3.jar
3712c3b4f43d9c12bf7b7aad187764f4b3c09cc962504899d8951eb310b1e4be org/hipparchus/hipparchus-fitting/4.0.3/hipparchus-fitting-4.0.3.jar
4bc0dbb5c6d5fefda5db9d92141c6cbb807174a1d491a85df46bcc57a67b8992 org/hipparchus/hipparchus-optim/4.0.3/hipparchus-optim-4.0.3.jar
279eece43411f372bd8340355cb89101def0bf9b1b6824231b76181d1ec63a81 org/hipparchus/hipparchus-filtering/4.0.3/hipparchus-filtering-4.0.3.jar
8b066e3060dc119efcce241fb0720a3b9e59c026d0f5fe9e70abfd5e22d3817b org/hipparchus/hipparchus-stat/4.0.3/hipparchus-stat-4.0.3.jar
"
CP=""
while read -r sum path; do
    [ -n "$path" ] || continue
    jar="$BUILD/$(basename "$path")"
    curl -fsS -o "$jar" "$MAVEN/$path"
    echo "$sum  $jar" | shasum -a 256 -c - >/dev/null
    CP="$CP${CP:+:}$jar"
done <<< "$JARS"

javac -d "$BUILD" -cp "$CP" "$HERE/TideStep2Oracle.java" "$HERE/SolidTidesFieldOracle.java"
JAVA_VERSION="$(java -version 2>&1 | head -1 | sed 's/"/\\"/g')"

{
    printf '{\n'
    printf '  "_source": {\n'
    printf '    "generator": "fixtures-generators/orekit_solid_tide_oracle/generate.sh",\n'
    printf '    "oracle": "Orekit 13.1.8 IERSConventions.IERS_2010.getTideFrequencyDependenceFunction, IERS Conventions (2010) Chapter 6 Tables 6.5a, 6.5b, 6.5c and Equations 6.8a, 6.8b",\n'
    printf '    "arguments": "gamma = GMST + pi with TAI as the UT1 scale, and the Delaunay arguments l, l_prime, f, d, omega, in radians, as Orekit formed them",\n'
    printf '    "epochs": "TT Julian date jd_whole + tt_fraction; UT1 = TT - 32.184 s",\n'
    printf '    "java": "%s"\n' "$JAVA_VERSION"
    printf '  },\n'
    printf '  "epochs": [\n'
    java -cp "$BUILD:$CP" TideStep2Oracle
    printf '  ]\n'
    printf '}\n'
} > "$OUT"
echo "wrote $OUT"

{
    printf '{\n'
    printf '  "_source": {\n'
    printf '    "generator": "fixtures-generators/orekit_solid_tide_oracle/generate.sh",\n'
    printf '    "oracle": "Orekit 13.1.8 org.orekit.forces.gravity.SolidTidesField with IERSConventions.IERS_2010 Love numbers, frequency-dependence function and permanent tide, pole tide off; IERS Conventions (2010) Section 6.2.1 Steps 1 and 2 and Section 6.2.2",\n'
    printf '    "inputs": "sun_km and moon_km are Earth-fixed positions given to SolidTidesField; mu = 398600.4415 km^3/s^2, ae = 6378.1363 km, GM Sun = 132712440041.93938 km^3/s^2, GM Moon = 4902.800066 km^3/s^2; gamma and the Delaunay arguments are those Orekit formed with TAI as the UT1 scale",\n'
    printf '    "fields": "tide_free and zero_tide are the corrections for a tide-free and a zero-tide central field, degrees 2 to 4, orders 0 to n",\n'
    printf '    "java": "%s"\n' "$JAVA_VERSION"
    printf '  },\n'
    printf '  "epochs": [\n'
    java -cp "$BUILD:$CP" org.orekit.forces.gravity.SolidTidesFieldOracle
    printf '  ]\n'
    printf '}\n'
} > "$FIELD_OUT"
echo "wrote $FIELD_OUT"
