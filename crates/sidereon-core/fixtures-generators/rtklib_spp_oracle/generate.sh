#!/usr/bin/env bash
# Regenerate tests/fixtures/rtk/rtklib_spp_selection_oracle.json: RTKLIB `pntpos`
# single-point solutions of the ESBC and WTZR 120-epoch RINEX fixtures from four
# initial positions each (see rtklib_spp_oracle.c), with the troposphere corrected
# and uncorrected.
#
# usage: RTKLIB_SRC=/path/to/RTKLIB/src ./generate.sh
#
# RTKLIB_SRC is the `src` directory of RTKLIB demo5 at commit
# 75a2e56275485b21a67bd35bc94bbeb8936e1a74. It is compiled with the options of the
# demo5 `rnx2rtkp` makefile, less tracing; the fixture is read by the Rust test
# `spp_selection_matches_rtklib_pntpos_from_every_initial_position`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX="$HERE/../../tests/fixtures"
OUT="$FIX/rtk/rtklib_spp_selection_oracle.json"
SRC="${RTKLIB_SRC:?set RTKLIB_SRC to the RTKLIB demo5 src directory}"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

# rtkcmn.c defines _POSIX_C_SOURCE, which on macOS hides snprintf unless
# _DARWIN_C_SOURCE is also defined; elsewhere the macro has no effect.
OPTS="-DENAGLO -DENAQZS -DENAGAL -DENACMP -DENAIRN -DNFREQ=4 -DNEXOBS=3 -D_DARWIN_C_SOURCE"
CFLAGS="-std=gnu99 -O2 -ffp-contract=off -fno-fast-math -w -I$SRC $OPTS"
for unit in rtkcmn trace rinex rtkpos postpos solution lambda geoid sbas preceph \
    pntpos ephemeris options ppp ppp_ar rtcm rtcm2 rtcm3 rtcm3e ionex tides sofa; do
    cc $CFLAGS -c "$SRC/$unit.c" -o "$BUILD/$unit.o"
done
cc $CFLAGS -o "$BUILD/rtklib_spp_oracle" "$HERE/rtklib_spp_oracle.c" "$BUILD"/*.o -lm

NAV="$FIX/nav/ESBC00DNK_R_20201770000_01D_MN.rnx"
{
    printf '{"generator": "fixtures-generators/rtklib_spp_oracle/generate.sh",\n'
    printf ' "rtklib": "rtklibexplorer/RTKLIB demo5 75a2e56275485b21a67bd35bc94bbeb8936e1a74",\n'
    printf ' "nav": "nav/ESBC00DNK_R_20201770000_01D_MN.rnx",\n'
    printf ' "runs": [\n'
    "$BUILD/rtklib_spp_oracle" esbc_iono_tropo \
        "$FIX/obs/ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx" "$NAV" 1 1
    printf ',\n'
    "$BUILD/rtklib_spp_oracle" wtzr_iono_tropo \
        "$FIX/obs/WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx" "$NAV" 1 1
    printf ',\n'
    "$BUILD/rtklib_spp_oracle" esbc_tropo \
        "$FIX/obs/ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx" "$NAV" 0 1
    printf ',\n'
    "$BUILD/rtklib_spp_oracle" esbc_iono \
        "$FIX/obs/ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx" "$NAV" 1 0
    printf ',\n'
    "$BUILD/rtklib_spp_oracle" wtzr_iono \
        "$FIX/obs/WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx" "$NAV" 1 0
    printf ']}\n'
} > "$OUT"
echo "wrote $OUT"
