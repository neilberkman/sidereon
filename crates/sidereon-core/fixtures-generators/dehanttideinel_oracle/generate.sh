#!/usr/bin/env bash
# Regenerate tests/fixtures/tides/dehanttideinel_oracle.json: the station
# displacement of the IERS Conventions (2010) Chapter 7 routine DEHANTTIDEINEL,
# built from the IERS source, for the four test cases of its header and a grid
# of stations and UTC dates (see dehant_oracle.f90).
#
# usage: ./generate.sh
#
# The IERS files are downloaded from the IERS Conventions Centre and checked
# against the SHA-256 digests below before they are compiled. They are compiled
# with gfortran at -O0 with floating-point contraction off, so every operation
# is a separately rounded binary64 operation as in the Rust translation. The
# fixture is read by the Rust test
# `solid_earth_tide_matches_dehanttideinel_built_from_source`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/../../tests/fixtures/tides/dehanttideinel_oracle.json"
BASE="https://iers-conventions.obspm.fr/content/chapter7/software/dehanttideinel"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

FILES="
686af399ea3a493e6c0ca659d2d64f367280f5bb331584ded0800ae8d45964d3 CAL2JD.F
64a5a69c38d41f6d9b64204f353f5ce46750d59a1067ca507cb9b6e71e934462 DAT.F
bc6039a1704761881bb785ce44ce084ea82783107ff64c576e69155a4914e2cb DEHANTTIDEINEL.F
636b6399dc6ab273a7b6104dd9341bf3c36995754aeee696511dbf19c9e909b6 NORM8.F
817761a92bb5416eb38322ea2d43d41cf8ea435e208a0ce229acf94311e9fa1e SPROD.F
d2976b8b76be8dd1d57e57a8d6b48f5764676126515bada3592753a07d3acd1e ST1IDIU.F
efdf284bd977826a1f4aea4c79c5dbd0c38fc1c403a2376010048b511a11f2c6 ST1ISEM.F
b1dfd0e797a3ce950631ad7dbbf5f576bf6b58dc515688253a3d84ca059bc282 ST1L1.F
898c70d4b8d50e09e0c717c911c4117b3ad1ca4996d369c258b68d00ea3a5674 STEP2DIU.F
f9d3bf0317222986d22e53557020bb13a6fbb90f8e3c9915137da6184d82813a STEP2LON.F
5ea9ab87e298d377f6dbe69c46b040906b6575a9132fab3830a4dc02c9139cef ZERO_VEC8.F
"
echo "$FILES" | while read -r sum name; do
    [ -n "$name" ] || continue
    curl -fsS -o "$BUILD/$name" "$BASE/$name"
    echo "$sum  $BUILD/$name" | shasum -a 256 -c - >/dev/null
done

FFLAGS="-O0 -ffp-contract=off -std=legacy -w"
for src in "$BUILD"/*.F; do
    gfortran $FFLAGS -c "$src" -o "${src%.F}.o"
done
gfortran -O0 -ffp-contract=off -c "$HERE/dehant_oracle.f90" -o "$BUILD/dehant_oracle.o"
gfortran -o "$BUILD/dehant_oracle" "$BUILD"/*.o

{
    printf '{\n'
    printf '  "_source": {\n'
    printf '    "generator": "fixtures-generators/dehanttideinel_oracle/generate.sh",\n'
    printf '    "routine": "IERS Conventions (2010) Chapter 7 DEHANTTIDEINEL.F with CAL2JD.F, DAT.F, NORM8.F, SPROD.F, ST1IDIU.F, ST1ISEM.F, ST1L1.F, STEP2DIU.F, STEP2LON.F, ZERO_VEC8.F",\n'
    printf '    "source_url": "%s/",\n' "$BASE"
    printf '    "compiler": "%s",\n' "$(gfortran --version | head -1)"
    printf '    "units": "metres for positions and displacement, hours for fhr_hours; dates are UTC"\n'
    printf '  },\n'
    printf '  "cases": [\n'
    "$BUILD/dehant_oracle"
    printf '  ]\n'
    printf '}\n'
} > "$OUT"
echo "wrote $OUT"
