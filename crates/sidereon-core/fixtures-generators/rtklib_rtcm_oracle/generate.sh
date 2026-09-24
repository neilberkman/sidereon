#!/usr/bin/env bash
# Regenerate the RTKLIB RTCM 3 oracle fixtures in tests/fixtures/rtcm/families/:
# the streams RTKLIB's encoder writes from real data (`rtklib_encoded_*.rtcm3`)
# and, for every stream there, what RTKLIB decodes from each frame
# (`<stream>.rtklib.jsonl`). See rtklib_rtcm_oracle.c for the modes.
#
# usage: RTKLIB_SRC=/path/to/RTKLIB/src RTKLIB_DATA=/path/to/RTKLIB/test/data ./generate.sh
#
# RTKLIB_SRC is the `src` directory of RTKLIB demo5 at commit
# 75a2e56275485b21a67bd35bc94bbeb8936e1a74 and RTKLIB_DATA its `test/data`
# directory. The library is compiled with the options of the demo5 `convbin`
# makefile, less tracing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX="$HERE/../../tests/fixtures"
FAM="$FIX/rtcm/families"
SRC="${RTKLIB_SRC:?set RTKLIB_SRC to the RTKLIB demo5 src directory}"
DATA="${RTKLIB_DATA:?set RTKLIB_DATA to the RTKLIB demo5 test/data directory}"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

# rtkcmn.c defines _POSIX_C_SOURCE, which on macOS hides snprintf unless
# _DARWIN_C_SOURCE is also defined; elsewhere the macro has no effect.
OPTS="-DENAGLO -DENAQZS -DENAGAL -DENACMP -DENAIRN -DNFREQ=4 -DNEXOBS=3 -D_DARWIN_C_SOURCE"
CFLAGS="-std=gnu99 -O2 -ffp-contract=off -fno-fast-math -w -I$SRC $OPTS"
for unit in rtkcmn trace rinex rtcm rtcm2 rtcm3 rtcm3e sbas preceph ephemeris ionex \
    geoid solution options pntpos rtkpos postpos lambda ppp ppp_ar tides sofa; do
    cc $CFLAGS -c "$SRC/$unit.c" -o "$BUILD/$unit.o"
done
cc $CFLAGS -o "$BUILD/oracle" "$HERE/rtklib_rtcm_oracle.c" "$BUILD"/*.o -lm
ORACLE="$BUILD/oracle"

# Receiver times RTKLIB starts from (GPS week, time of week), each within half
# a week of its stream.
GMSD7="1710 138"      # GMSD7_20121014.rtcm3, 2012-10-14
TESTGLO="1562 515220" # testglo.rtcm3, 2009-12
RTK2GO="2437 361600"  # the rtk2go captures, 2026-09-24 04:26 UTC
NAVIC="2437 172800"   # BRDM00DLR_S_20262650000_01D_MN_navic.rnx, 2026-09-22
SSRA03="2426 223638"  # SSRA03IGS0_2026188140760_3epoch.rtcm3, 2026-07-07

# Streams RTKLIB's encoder writes from real data.
"$ORACLE" encode-msm "$DATA/rcvraw/GMSD7_20121014.rtcm3" $GMSD7 12 \
    "$FAM/rtklib_encoded_msm1_to_msm4.rtcm3"
"$ORACLE" encode-legacy "$DATA/rcvraw/testglo.rtcm3" $TESTGLO 12 \
    "$FAM/rtklib_encoded_legacy.rtcm3"
"$ORACLE" encode-1041 "$FIX/nav/BRDM00DLR_S_20262650000_01D_MN_navic.rnx" \
    "$FAM/rtklib_encoded_1041.rtcm3"
"$ORACLE" encode-4076 "$FIX/ssr/SSRA03IGS0_2026188140760_3epoch.rtcm3" $SSRA03 2 \
    "$FAM/rtklib_encoded_4076.rtcm3"

decode() {
    "$ORACLE" decode "$FAM/$1" $2 $3 > "$FAM/$1.rtklib.jsonl"
    echo "wrote $FAM/$1.rtklib.jsonl"
}

# MSM1..MSM7.
decode rtklib_encoded_msm1_to_msm4.rtcm3 $GMSD7
decode rtk2go_tiftga_msm3_msm4.rtcm3 $RTK2GO
decode rtk2go_ormalingen_msm5.rtcm3 $RTK2GO
decode rtk2go_sejongnav_msm5.rtcm3 $RTK2GO
decode rtk2go_mirmenhof_msm6.rtcm3 $RTK2GO

# Legacy RTK observations 1001..1004, 1009..1012.
decode rtklib_encoded_legacy.rtcm3 $TESTGLO
decode rtklib_testglo_legacy.rtcm3 $TESTGLO
decode rtk2go_granthamall_legacy.rtcm3 $RTK2GO
decode rtk2go_jacksbay_legacy.rtcm3 $RTK2GO
decode rtk2go_mirmenhof_legacy.rtcm3 $RTK2GO

# NavIC ephemeris 1041 and GLONASS code-phase biases 1230.
decode rtklib_encoded_1041.rtcm3 $NAVIC
decode rtk2go_1230.rtcm3 $RTK2GO

# IGS SSR 4076.
decode rtklib_encoded_4076.rtcm3 $SSRA03

# System parameters 1013 (RTKLIB reads none of it) and text 1029.
decode rtk2go_1013.rtcm3 $RTK2GO
decode text_1029.rtcm3 $RTK2GO
