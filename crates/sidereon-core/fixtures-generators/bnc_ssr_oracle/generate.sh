#!/usr/bin/env bash
# Regenerate the BNC SSR oracle fixtures: the SSR stream BNC's encoder writes
# (tests/fixtures/rtcm/families/bnc_encoded_ssr.rtcm3) and, for it and for the
# other SSR streams under tests/fixtures, what BNC decodes from each frame
# (`<stream>.bnc.jsonl`). See bnc_ssr_oracle.cpp.
#
# usage: BNC_SRC=/path/to/BNC_2.13.7/src ./generate.sh
#
# BNC_SRC is the `src` directory of bnc-2.13.7-sources.zip. Qt's QtCore
# framework (for QList, which the SSR codec uses) comes from QT_FRAMEWORKS, by
# default Homebrew's qtbase. Run the RTKLIB generator after this one: it decodes
# bnc_encoded_ssr.rtcm3 as well.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX="$HERE/../../tests/fixtures"
SRC="${BNC_SRC:?set BNC_SRC to the BNC src directory}"
QT="${QT_FRAMEWORKS:-/opt/homebrew/opt/qtbase/lib}"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

# The SSR codec includes bncutils.h for CRC24 alone; this stand-in declares it
# (bnc_ssr_oracle.cpp defines it) without the rest of BNC.
mkdir -p "$BUILD/shim"
printf '#include <cmath>\nunsigned long CRC24(long size, const unsigned char *buf);\n' \
    > "$BUILD/shim/bncutils.h"
CXXFLAGS="-std=c++17 -O2 -ffp-contract=off -w -I$BUILD/shim -I$SRC -I$SRC/RTCM3 -F$QT -I$QT/QtCore.framework/Headers"
for unit in clock_orbit clock_orbit_rtcm clock_orbit_igs; do
    c++ $CXXFLAGS -c "$SRC/RTCM3/clock_and_orbit/$unit.cpp" -o "$BUILD/$unit.o"
done
c++ $CXXFLAGS -o "$BUILD/oracle" "$HERE/bnc_ssr_oracle.cpp" "$BUILD"/*.o -F$QT -framework QtCore -Wl,-rpath,$QT

"$BUILD/oracle" encode "$FIX/rtcm/families/bnc_encoded_ssr.rtcm3"
for stream in rtcm/families/bnc_encoded_ssr.rtcm3 rtcm/families/rtklib_encoded_4076.rtcm3 \
    ssr/SSRA03IGS0_2026188140760_3epoch.rtcm3; do
    "$BUILD/oracle" decode "$FIX/$stream" > "$FIX/$stream.bnc.jsonl"
    echo "wrote $FIX/$stream.bnc.jsonl"
done
