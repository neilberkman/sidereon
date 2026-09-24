#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX="$HERE/../../tests/fixtures"
OUT="${1:-$FIX/sp3/rtklib_sp3_node_oracle.json}"
SRC="${RTKLIB_SRC:?set RTKLIB_SRC to the RTKLIB demo5 src directory}"
IONO_MODE="${RTKLIB_IONO_MODE:?set RTKLIB_IONO_MODE to off or brdc}"
TROPO_MODE="${RTKLIB_TROPO_MODE:?set RTKLIB_TROPO_MODE to off or saas}"
NAV_FILE="$FIX/nav/BRDM00DLR_S_20201760000_01D_MN.rnx"
NAV_SOURCE_URL="https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2020/176/BRDM00DLR_S_20201760000_01D_MN.rnx.gz"
case "$IONO_MODE" in
    off|brdc) ;;
    *) printf 'unsupported RTKLIB ionosphere mode: %s\n' "$IONO_MODE" >&2; exit 2 ;;
esac
case "$TROPO_MODE" in
    off|saas) ;;
    *) printf 'unsupported RTKLIB troposphere mode: %s\n' "$TROPO_MODE" >&2; exit 2 ;;
esac
EXPECTED_REV=75a2e56275485b21a67bd35bc94bbeb8936e1a74
ACTUAL_REV="$(git -C "$SRC/.." rev-parse HEAD)"
if [[ "$ACTUAL_REV" != "$EXPECTED_REV" ]]; then
    printf 'RTKLIB revision mismatch: expected %s, found %s\n' "$EXPECTED_REV" "$ACTUAL_REV" >&2
    exit 1
fi
DIRTY_SOURCES="$(git -C "$SRC/.." status --porcelain --untracked-files=all -- src)"
if [[ -n "$DIRTY_SOURCES" ]]; then
    printf 'RTKLIB source tree is dirty; refusing to compile these sources:\n%s\n' "$DIRTY_SOURCES" >&2
    exit 1
fi

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
OUT_TMP="$BUILD/rtklib_sp3_node_oracle.json"

COMPILE_FLAGS=(-std=gnu99 -O2 -ffp-contract=off -fno-fast-math -w "-I$SRC"
    -DENAGLO -DENAQZS -DENAGAL -DENACMP -DENAIRN -DNFREQ=4 -DNEXOBS=3 -D_DARWIN_C_SOURCE)
for unit in rtkcmn trace rinex rtkpos postpos solution lambda geoid sbas preceph \
    pntpos ephemeris options ppp ppp_ar rtcm rtcm2 rtcm3 rtcm3e ionex tides sofa; do
    cc "${COMPILE_FLAGS[@]}" -c "$SRC/$unit.c" -o "$BUILD/$unit.o"
done
cc "${COMPILE_FLAGS[@]}" -o "$BUILD/rtklib_sp3_oracle" \
    "$HERE/rtklib_sp3_oracle.c" "$BUILD"/*.o -lm

GEOMETRY_SP3="$FIX/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
STATIC_SP3="$FIX/sp3/trimmed_go_static.sp3"
GEOMETRY_SHA="$(shasum -a 256 "$GEOMETRY_SP3" | cut -d ' ' -f 1)"
STATIC_SHA="$(shasum -a 256 "$STATIC_SP3" | cut -d ' ' -f 1)"
NAV_SHA="$(shasum -a 256 "$NAV_FILE" | cut -d ' ' -f 1)"
NAV_NAME="$(basename "$NAV_FILE")"
EXPECTED_NAV_SHA=778e99a30b9fc3f2ea2844219f459535a9060645ec6accf89d83410154db5c53
if [[ "$NAV_SHA" != "$EXPECTED_NAV_SHA" ]]; then
    printf 'broadcast navigation fixture SHA-256 mismatch: expected %s, found %s\n' \
        "$EXPECTED_NAV_SHA" "$NAV_SHA" >&2
    exit 1
fi
mkdir -p "$(dirname "$OUT")"
if "$BUILD/rtklib_sp3_oracle" \
    "$ACTUAL_REV" "$GEOMETRY_SHA" "$STATIC_SHA" "$NAV_SHA" \
    "$NAV_NAME" "$NAV_SOURCE_URL" "$IONO_MODE" "$TROPO_MODE" \
    "$GEOMETRY_SP3" "$STATIC_SP3" "$NAV_FILE" > "$OUT_TMP"; then
    :
else
    RESULT=$?
    printf 'RTKLIB SP3 oracle executable failed with status %d; fixture was not installed\n' \
        "$RESULT" >&2
    exit "$RESULT"
fi
mv "$OUT_TMP" "$OUT"
printf 'wrote %s\n' "$OUT"
